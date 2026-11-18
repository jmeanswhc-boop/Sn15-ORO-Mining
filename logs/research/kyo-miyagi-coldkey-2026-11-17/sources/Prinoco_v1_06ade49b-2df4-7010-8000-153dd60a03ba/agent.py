"""A shopping agent for ORO world episodes (runtime contract 8): one shopper and one store world per run.

The store is learnt at run time. policy_view.tools lists what the world offers; a model reads those tools and says
which one does which buying job, every answer is checked against the schemas, and keyword rules cover whatever stays
unassigned. The request sets the price ceiling, the ranking wish, the message and move allowances and any procedure,
and every action is counted against them. Replies are read by their shape - an id next to a title, a price or a
property table makes a listing - so no key names are fixed in advance. Lines from the shopper or the store that arrive
between moves are read before the next one.

Course of a run: understand the request, search, sort the result rows, open the likeliest listings, put one property
per message to the shopper when something may be held back, judge each opened listing against every requirement, then
basket the pick and check out - or decline plainly, naming no listing, when nothing qualifies. Models only read text;
the code decides every action. Steps, messages, model requests and wall time pace the run, and a run is always closed
while steps and time remain.
"""

from __future__ import annotations

import json
import math
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# ================================================================ model seats
# Each seat mixes vendors. chutes hosts only the TEE builds of two open models and has no ask seat (deep stands in).
# The ask seat starts with sonnet, which named the held-back property most often on recorded pre-question states.
_pda_SOL, _pda_HAIKU = "openai/gpt-6-sol", "anthropic/claude-haiku-5.5"
_pda_GEMINI, _pda_SONNET = "google/gemini-3.8-flash", "anthropic/claude-sonnet-5.5"
_pda_OPEN_PAIR = "zai-org/GLM-5.2-TEE deepseek-ai/DeepSeek-V3.2-TEE".split()
ROUTES = {
    "openrouter": dict(deep=[_pda_SOL, _pda_HAIKU, _pda_GEMINI], quick=[_pda_HAIKU, _pda_GEMINI, _pda_SOL],
                       ask=[_pda_SONNET, _pda_SOL, _pda_HAIKU]),
    "chutes": dict(deep=_pda_OPEN_PAIR[:], quick=_pda_OPEN_PAIR[:]),
}
_pda_FREE_CHOICE = frozenset({_pda_SONNET})   # answers 400 to a forced tool choice: sent "auto" plus a nudge
_pda_MODELS_PER_ASK, _pda_ASKS_PER_RUN = 3, 18  # tries for one question; requests per run, failed ones included
_pda_SPEND_PER_RUN = 0.054    # dollars of model use per run, every request included; past it code decides alone
_pda_SPEND_KEPT = 0.010       # the last part of it, kept for judging listings and hearing lines
_pda_SPEND_APART = 0.02       # most of it the records a stated value is kept on may take
_pda_SPEND_FIRST = 0.004      # what a kind of request not met yet in the run is expected to cost
_pda_KEPT_FOR = ("judge_listings", "read_line")   # this agent's own request names that may use the kept part
_pda_ASKS_UNPRICED = _pda_ASKS_PER_RUN  # no cost reported (a test bench; the validator always reports it): the count cap

# ================================================================ pacing, as shares of the step limit and counts
# held back for basket + checkout + a refused check-out and its replacement (>= 5); opened before asking (>= 4); read
# before the first question (>= 3)
END_SHARE, OPEN_SHARE, ASK_SHARE = 0.2, 0.11, 0.13
SEARCH_SHARE = 0.3            # how much of the free steps the first searches may use
JUDGE_SIZE, REVIEW_ROWS = 12, 40                  # listings per judging request; result rows per screening request
PAGES_PER_QUERY, UNSTATED_MESSAGES, FALLBACK_STEPS = 3, 2, 30
# pages of a search whose schema states no limit; messages when the request names no number; steps when
# policy_view gives none

# ================================================================ links to the proxy
_env_net, _net = (ProxyClient(max_retries=1, timeout=wait) for wait in (45, 90))
LOCAL_ENV_CALL = None         # a local test bench may set this; otherwise steps go to the validator proxy
ENV_RETRIES, ENV_SENDS_MAX, ENV_TIMEOUT = 3, 9, 45
# sends before a step counts as lost (unless time allows more); most sends of a finishing step; longest world wait

# ================================================================ wall clock (sandbox allows 300 s, start-up included)
CLOCK_BUDGET, WRAP_LEFT, MODEL_FLOOR, FINISH_LEFT = 265.0, 60.0, 20.0, 3.0
# usable seconds; under WRAP_LEFT extras are dropped; under MODEL_FLOOR (+ usual reply time) code answers alone;
# a finishing step is still tried in the last FINISH_LEFT seconds
MODEL_TIMEOUT = 90            # longest wait for one model reply
_pda_REFUSED = frozenset((400, 401, 403, 404, 413, 422))   # the route rejects the request: benched for the run
_pda_NO_CREDIT = 402                                        # the run's key has no credit left: no model can answer
_pda_THROTTLED = 429                                       # rate limit: pause, then the same model again
_pda_GATEWAY = (502, 503)                                  # a gateway fault: the same model once more, at once
_pda_PAUSES, _pda_PAUSES_PATIENT = [2.0, 4.0], [8.0] * 2   # the patient extra is for requests a run needs


class _PaClock:
    """Seconds left in the run, the time a model reply usually takes, and paced sleeping."""

    def __init__(self, budget=CLOCK_BUDGET):
        self._pda_limit = budget
        self._pda_lags = []
        self._pda_slept = 0
        self._pda_t0 = time.monotonic()

    def left(self):
        return self._pda_limit - (time.monotonic() - self._pda_t0)

    def hurry(self):
        return WRAP_LEFT > self.left()

    def model_room(self):
        recent = self._pda_lags[-4:]
        usual = sum(recent) / len(recent) if recent else 0.0
        return MODEL_FLOOR + 1.5 * usual < self.left()

    def model_timeout(self):
        wait = min(MODEL_TIMEOUT, self.left() - 15)
        return int(wait) if wait > 5 else 5

    def nap(self, seconds, floor=FINISH_LEFT):
        """Sleep unless that would leave fewer than `floor` seconds; report whether it slept. The pause is
        stretched by a factor that cycles with the nap count, so runs failing together drift apart."""
        pause = seconds * (0.75 + 0.05 * (self._pda_slept % 11))
        self._pda_slept += 1
        if self.left() - pause >= floor:
            time.sleep(pause)
            return True
        return False


DECLINE_NOTE = "Nothing I found meets every requirement at your price limit, so I am stopping here and buying nothing."
ASK_TRAIT = "Which {} would you like the {} to have?"


# ================================================================ words and figures
_pa_UNSIGNED = re.compile(r"\d[\d,]*(?:\.\d+)?")
_pda_FIGURE = re.compile("-?" + _pa_UNSIGNED.pattern)
_pda_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def norm(raw):
    """Lower-cased text with every run of white space turned into one blank and the ends cut."""
    return " ".join(str(raw or "").lower().split())


def words(raw):
    return _pda_WORD.findall(norm(raw))


def number(raw):
    """A JSON number as float, or the first figure written in a text; True/False are not numbers."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    hit = _pda_FIGURE.search(str(raw or ""))
    return None if hit is None else float("".join(hit.group().split(",")))


def as_written(amount, source, ref=None, avoid=()):
    """A model's figure checked against the figures `source` really contains. It stands when the text has it;
    otherwise the text's figure nearest to `ref` (or to the model's figure) replaces it, since models sometimes
    convert units unasked. A figure the model moved by whole powers of ten (a smaller money unit read as a larger
    one) goes back to the text's figure of that scale, and so does one equal to a figure in `avoid` (counts the
    text gives for other things) when the text has such a scaled figure."""
    read = number(amount)
    if read is None:
        return None
    present = [x for x in map(number, _pa_UNSIGNED.findall(source)) if (x or 0) > 0]
    scaled = [x for x in present if read > 0 and abs(x - read) > 1e-6
              and abs(math.log10(x / read) - round(math.log10(x / read))) < 1e-9]
    pick_scaled = lambda: min(scaled, key=lambda x: abs(math.log10(x / read)))
    if scaled and any(isinstance(a, (int, float)) and abs(read - a) < 1e-6 for a in avoid):
        return pick_scaled()
    if not present or min(abs(x - read) for x in present) < 1e-6:
        return read
    if scaled:
        return pick_scaled()
    near = ref or read
    gap = (lambda x: abs(math.log(x / near))) if near > 0 else (lambda x: abs(x - near))
    return min(present, key=gap)


def _pa_trim(token):
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"           # a -y word's plural comes back to its singular
    for tail in ("ing", "ed", "es", "s"):
        keep = len(token) - len(tail)
        if keep > 2 and token.endswith(tail):
            token = token[:keep]
            break
    # a silent final -e goes too, so a word and its -ing / -ed / -es forms share one root
    return token[:-1] if len(token) > 4 and token.endswith("e") and not token[-2].isdigit() else token


def stems(raw):
    """Words with one -s, -es, -ed or -ing cut off (a final -e too, -ies back to -y), for loose matching."""
    return [_pa_trim(w) for w in words(raw)]


def label_words(raw):
    return " ".join(words(re.sub(r"[_.]", " ", str(raw))))


# ================================================================ JSON schemas of the forced tools, from short layouts
# A layout is "name=code ..." where a trailing "!" on the name marks it required. Codes: s i b n (string, integer,
# boolean, number), a trailing "?" also allows null, "[code]" is an array, "a|b|c" a string enum, "pairs" a list of
# attribute/value pairs, and "*" takes the shape passed by keyword under that name.
_pda_TYPES = dict(s="string", i="integer", b="boolean", n="number")


def _pda_enum(*values):
    return {"type": "string", "enum": list(values)}


def _pda_list(item):
    return dict(type="array", items=item)


def _pda_shape(code):
    if code.startswith("["):
        return _pda_list(_pda_shape(code[1:-1]))
    if "|" in code:
        return _pda_enum(*code.split("|"))
    if code == "pairs":
        return _pda_list(_pda_object("attribute!=s value!=s bound=*", bound=_pda_enum("exact", "at least", "at most")))
    if code.endswith("?"):
        return {"type": [_pda_TYPES[code[:-1]], "null"]}
    return {"type": _pda_TYPES[code]}


def _pda_layout(layout, notes, made):
    """(properties, required names) of a layout; `notes` adds descriptions by field name."""
    props, must = {}, []
    for part in layout.split():
        name, _, code = part.partition("=")
        if name.endswith("!"):
            name = name[:-1]
            must.append(name)
        shape = made[name] if code == "*" else _pda_shape(code)
        if name in notes:
            shape["description"] = notes[name]
        props[name] = shape
    return props, must


def _pda_object(layout, notes=None, **made):
    props, must = _pda_layout(layout, notes or {}, made)
    return {"type": "object", "properties": props, "required": must}


def _pda_notes(text):
    """'name: description' lines -> {name: description}."""
    pairs = (line.split(":", 1) for line in text.strip().splitlines())
    return {name.strip(): about.strip() for name, about in pairs}


def _pa_forced(name, about, needed, fields):
    params = dict(type="object", properties=fields, required=needed)
    return {"type": "function", "function": dict(name=name, description=about, parameters=params)}


def _pda_tool(name, about, layout, notes=None, **made):
    props, must = _pda_layout(layout, notes or {}, made)
    return _pa_forced(name, about, must, props)


# ================================================================ which tool does which job
# role, then its value slots | what the job is
_pda_ROLE_TABLE = """
keyword_search  text_arg page_arg                         | look products up by typed words
filter_search   category_arg brand_arg price_arg page_arg | list products by a field such as a category, a brand or a price ceiling
open_listing    id_arg                                    | return the full record of one listing (specifications, attributes, datasheet); if more than one tool shows a listing, pick the tool whose reply carries those properties
open_offer      id_arg                                    | return one listing's offer (what it costs, whether it is in stock, the seller's remarks) when a tool other than the open_listing one shows them; "" when that tool shows them already
open_many       ids_arg                                   | take a list of listing ids and return all of them in full detail in a single call
basket_add      id_arg qty_arg                            | put a listing into the cart or basket
basket_remove   id_arg qty_arg                            | take a listing out of the cart or basket
basket_view                                               | show what is in the cart
checkout        id_arg qty_arg                            | place the order: pay for the cart, or for a single named listing
decline         text_arg                                  | end the task without buying anything
message         text_arg                                  | write to the shopper, a message or a question
"""
ROLE_FIELDS, _pda_JOB_TEXT = {}, {}
# Roles an answer may leave out, and worth no follow-up ask when it leaves them open or names a tool unfit for them.
_pdb_OPTIONAL = frozenset({"open_offer"})
for _pda_row in _pda_ROLE_TABLE.strip().splitlines():
    _pda_left, _pda_job = _pda_row.split("|")
    _pda_role, *_pda_slots = _pda_left.split()
    ROLE_FIELDS[_pda_role], _pda_JOB_TEXT[_pda_role] = _pda_slots, _pda_job.strip()

TOOL_ROLES_GUIDE = (
    "Below are the jobs of a buying assistant. Per job, report:\n"
    "- tool: which store tool carries it out (\"\" if no tool can). The same tool may serve two jobs provided "
    "the fixed values each one needs are not the same.\n"
    "- one name per value slot of the job: the tool parameter that receives that value, or \"\" when the tool has "
    "no such parameter. Slots: text_arg - the words, or the message; page_arg - whatever moves on to later results "
    "(page number, row offset, cursor); category_arg; brand_arg; price_arg - a price ceiling; qty_arg - how many "
    "units; id_arg - the id of a listing.\n"
    "- fixed: any parameter that must hold one set value before the tool performs this job (e.g. an action like add "
    "or remove, or an enum mode that picks field search over keyword search), with that value in the schema's own "
    "spelling.\n\n"
    "Tools arrive with a name, a description and a JSON parameter schema. Odd names happen; judge mainly by what "
    "the description and the parameter types say.\n\nBuying jobs:\n"
    + "\n".join("- %s: %s" % pair for pair in _pda_JOB_TEXT.items())
)

ROLES_AGAIN_NOTE = (
    "\n\nNote on fixed: when a tool can do a job only if one parameter is given one specific value - say an enum "
    "selecting a mode, an ending or what to do - write the parameter with that value in fixed, spelled to match the "
    "schema. Your earlier reply named no workable tool for the jobs below, so reply again in full, covering all jobs:"
    "\n{gaps}")

FINISH_ROLES_GUIDE = (
    "For each ending give the tool, the parameter and the value that make the tool carry it out, written the way "
    "the schema permits; use \"\" when no tool can. For close, add the parameter that takes a short explanation, if "
    "the tool has one.\n\nThe endings: order = buy what the cart holds, or one named listing; close = finish the "
    "task with nothing bought. Every tool shown has a required parameter limited to a few values.")

FILL_CALL_GUIDE = (
    "Write the arguments of one call to a shop tool. Keep exactly to its JSON schema: every required field, the "
    "types and the nesting it gives, and only values its option lists allow. Use the listing id and the quantity "
    "named, as written, and put nothing else in the basket.")
_pdb_FILL_HEAD = "The tool, as JSON:\n"
_pdb_FILL_ASK = "\n\nCall it so that the basket holds listing {pid}, quantity {count}."

FINISH_ROLES_SPEC = _pda_tool("finish_roles", "How the store closes a task, both ways.", "order!=* close!=*",
                              order=_pda_object("tool!=s param=s value=s"),
                              close=_pda_object("tool!=s param=s value=s text_param=s"))

TOOL_ROLES_SPEC = _pda_tool(
    "tool_roles", "Job-by-job mapping onto the store's tools.", " ".join(role + ("=*" if role in _pdb_OPTIONAL else "!=*") for role in ROLE_FIELDS),
    **{role: _pda_object(" ".join(["tool!=s"] + [slot + "=s" for slot in slots] + ["fixed=*"]),
                         fixed=_pda_list(_pda_object("param!=s value!=s")))
       for role, slots in ROLE_FIELDS.items()})

# ================================================================ the opening request
READ_REQUEST_GUIDE = """Turn what a shopper asks of a store assistant into the fields listed here. Rely only on what the request actually says, and pass over any chat unrelated to buying.

THE PRODUCT
* item: what sort of product it is, said in a few everyday shop words. A word that serves only to name that sort or the section it is sold in (which garment, which drink) must stay in.
* requirements: every further property the item needs, as attribute/value pairs - but only properties that a datasheet might list, e.g. material, shape, colour, style, the type of closure or of heel.
  - Leave out of requirements: stock, quantity, the spending limit, when or how it is delivered, the way the outcome will be judged, the sort of product or its section, and phrases that merely say the purchase replaces something worn out (nothing a datasheet lists is named by them).
  - One exception: if the request itself names a property and assigns that property a value (a kettle that must be of the gooseneck kind), list that pair even though its value is a sort (kind: gooseneck); item then remains the wider product (kettle).
  - attribute: in plain datasheet terms, the property that the value describes, in the request's framing (a style value goes under style, an occasion value under occasion). Keep all of the request's words for that property, leave out the noun naming the item, and a value ruled out stays filed under the same attribute.
  - value: a single plain value; "X, not Y" gives X.
  - bound: "at least" or "at most" if the request limits a measured quantity instead of naming a value, in which case value is just the figure with its unit; else "exact".
* exclusions: attribute/value pairs for specific values the request mentions and rules out - which the product must NOT carry. A requirement phrased as "anything other than" some value is not an exclusion; that requirement already handles it.

HOW FITTING PRODUCTS ARE RANKED
* priority: "cheapest" when the shopper wants the lowest price among the fitting products; "best" when fitting products are to be ranked on some measure that is not price (name that measure in best_measure and set best_direction to "high" or "low"); in every other case "none".

MONEY, UNITS, STOCK
* max_price: the number stated as the most the shopper will spend in total, exactly as written and never converted between units; null if there is no such number. currency: as written.
* quantity: number of units wanted, 1 when not given.
* in_stock: true if stock on hand or immediate availability is a must.

RULES THE REQUEST LAYS DOWN
* step_budget is the count of moves or actions allowed to the assistant, message_budget the count of messages it may send to the shopper; null for either when no number appears.
* question_cost: what each question or message to the shopper costs by the request, as a number, or null if no cost is given.
* decline_rule: what the request itself says to do if nothing fits, in its words; "" if it is silent on that.
* rules: further steps the request asks for.

PURPOSE AND SEARCHES
* reasons: the purpose of the purchase or who will use it, a few words each.
* searches: between 3 and 5 brief catalogue queries, most specific first. Each is two or three words: the most distinctive value the shopper requires beside a plain noun for the product, worded as a listing title would word it and never the name of a store section. One query may be that noun alone."""

# a requirement whose value the shopper keeps on another listing
_PDA_POINTER_NOTE = ("One entry per requirement whose value the %s withholds and sends you to read off another listing "
                     "instead: attribute as in requirements; shop, the seller or brand of that other listing; kind, "
                     "the sort of product it is; details, the facts given to single it out, each a property with its "
                     "value in a few words (nothing about pages or links); margin, how far the pick may stray from "
                     "that value (a number, with its unit), or null when it must be equal; differs, true when "
                     "the pick's value must be a different one from that listing's.")
def _pda_pointers():
    return _pda_list(_pda_object("attribute!=s shop!=s kind=s details!=[s] margin=n? unit=s differs=b"))

READ_REQUEST_SPEC = _pda_tool(
    "read_request", "What the opening message asks for, split into fields.",
    "item!=s requirements!=pairs max_price!=n? currency=s priority!=cheapest|best|none best_measure=s "
    "best_direction=high|low| quantity!=i in_stock!=b message_budget=i? step_budget=i? question_cost=n? "
    "decline_rule=s rules=[s] message_steps=[s] exclusions=pairs reasons=[s] searches!=[s] pointers=* products=*",
    _pda_notes("""
rules: Other steps of the procedure the request asks for.
message_steps: Quote, word for word, each step of the request's procedure that can only be done by messaging the shopper - e.g. asking for confirmation before buying. Do not include offers to answer questions, or the wording to use when no product fits.
reasons: For whom, or why, the product is being bought.
searches: Short catalogue searches of a few words, most specific first.
pointers: %s
products: Only when the request asks for two or more different sorts of product at once: one entry per sort, in the order asked, each with its own item, quantity, requirements and exclusions (as above) and two or three short searches for it. Leave it empty when one sort of product is wanted; when it is filled, the other product fields describe its first entry.
""" % (_PDA_POINTER_NOTE % "request")), pointers=_pda_pointers(),
    products=_pda_list(_pda_object("item!=s quantity!=i requirements!=pairs exclusions=pairs searches=[s]")))

# ================================================================ a value kept on another listing
REFER_GUIDE = """A shopper needs one property to take the value held by one particular listing, described below. \
The full records of listings that may be it follow, numbered. Go through every record: say whether its seller or \
brand is the one described, whether it is that sort of product, and for each fact given, in order, what the record \
itself says on that point and whether it agrees. A figure written in another unit that comes to the same amount \
agrees; a different value for the same point does not. A fact agrees only when the record says this very item has \
it: words saying that some other item or version has it do not count, and leave this record disagreeing. Then give the record's own value for the property, copied as \
the record writes it. Read the whole record, notes included: when its text sends the reader to another listing for \
that property, copy those words exactly in onward and describe the listing they name using only those words."""

REFER_SPEC = _pda_tool(
    "read_records", "Each record checked against the wanted listing, with its value or the listing it points to.",
    "records!=*", records=_pda_list(_pda_object(
        "num!=i seller!=yes|no sort!=yes|no|unclear facts!=* value=s? onward!=s shop!=s kind=s details!=[s]",
        _pda_notes("""
num: The record's number.
seller: yes when the record's seller or brand is the one described.
sort: Whether the record is the sort of product described.
facts: One entry per fact given, in the same order.
value: The record's own value for the property, word for word; null when it holds none.
onward: Only when the record sends the reader to another listing for the property: those words of the record, copied exactly; else "".
shop: The seller or brand of the listing the onward words name; else "".
kind: The sort of product the onward words name; else "".
details: The facts the onward words give to single that listing out, a few words each, copied from them.
"""), facts=_pda_list(_pda_object("says!=s agrees!=yes|no|absent", _pda_notes("""
says: What the record says on that point, in its own words; "" when it says nothing.
agrees: yes or no when the record speaks to the fact; absent when it says nothing on it.
"""))))))

# ================================================================ sorting search rows
SCREEN_GUIDE = """For every row, keyed by its number, give:
- kind: "yes" if the row is the sort of item wanted; "no" if it is clearly some other product (another device or garment, or an accessory for one); "maybe" if that cannot be told.
- stated: "yes" if the row by itself shows every stated requirement; "no" if it shows something that goes against one of them; "unknown" otherwise.

The rows were returned by a store search a shopping assistant made. Base both labels on nothing but each row's title, category and description."""

SCREEN_SPEC = _pda_tool("screen_rows", "Two labels per row.", "rows!=*",
                        rows=_pda_list(_pda_object("num!=i kind!=yes|no|maybe stated!=yes|no|unknown")))

# ================================================================ questions for the shopper
TOPICS_GUIDE = """Output store property names to raise with the shopper as questions, the one most likely to hide a requirement first. Give each name once, plainly and exactly as the store labels it, reading dots and underscores as spaces.

Order:
- First the property the request's story points to for this item: a remark about what the shopper or someone else will notice, look at, check, measure, fit, match it with, need it to work with, taste, or read on its label, even when made in passing or in figurative words. Name the recorded property that remark is about (one label), also in story_points_to. A remark about another product of the order does not count for this item.
- Then the properties most opened listings record, with values that differ from listing to listing - above all those about the item's look, fit, make or kind. Among them, one whose values are different sorts of the item goes ahead of one whose values grade a single sort by a figure or rating.
- Next any other property the request's story or reasons make matter.
- When listings differ mostly in the variant a buyer picks, the property those variants express, in plain words.

Skip:
- terms of the sale rather than features of the item;
- a property the request or an answer already settles (though a stated value that equals none of the store's values in full but is part of some leaves the store's longer label open), or one the shopper said they do not mind about;
- labels that list a listing's codes or variant choices;
- a label that merely repeats another.

Background: shoppers sometimes keep a requirement back. It is one value of one property the listings record, and it comes out only when a message names that property. On some properties the shopper has no view at all. A held-back requirement nearly always sits on a property that most listings of the kind record, with values that change between listings.

You receive the request with its story, the requirements already fixed, earlier questions with their answers, and, for each property the store records for this kind of item, how many opened listings record it and which values appear."""

TOPICS_SPEC = _pda_tool("ask_topics", "Ranked store labels worth a question.", "story_points_to=s? topics!=[s]",
                        {"story_points_to": "The recorded store label the story's remarks about this item point to; "
                                            "null when the story says nothing about any of its properties.",
                         "topics": "Each a store label; put first the one most likely to hold a requirement kept back."})

# ================================================================ what a line says
READ_LINE_GUIDE = """Fields to fill:
* reveals: property and value of every requirement the line states - a new demand, or an answer giving a value. The value is what the line wants the property to be, never a remark on the answer itself (how firm, final or negotiable it is). Each property the line names gets exactly one reveal, under the line's own name for it, even if the question was about another property. bound: "at least" or "at most" when the line limits a measured number (value then holds the number and its unit), otherwise "exact".
* changes: property and new value of a requirement that now takes another value.
* drops: numbers of the current requirements the shopper plainly no longer needs, however they refer to them; answering "I don't mind" is no drop.
* max_price: a new spending limit stated in the line, copied exactly as written with no unit conversion; else null.
* pick_unavailable: true if the line says the item the assistant chose or proposed cannot be ordered, is unavailable, sold out or gone.
* no_preference: true if the shopper does not mind about the thing asked.
* not_understood: true if the shopper did not understand the message.
* pointers: %s

Only for a line that replies to a question:
* reply_class: "reveal" if it gives a value for the property asked about; "not_picky" if the shopper has no preference on it; "split" if the shopper did not take the message as one question about one property (wants a single subject per message, or asks for it to be narrowed down); "other" otherwise.
* quote: the exact words of the line that show the reply_class.

The line comes from the shopper, or from the store speaking for the shopper, and may answer a question the assistant asked about a single property. Say what it changes, quoting values in the shopper's own words.""" % (_PDA_POINTER_NOTE % "line")

# a second pass when the first gave one property two values
READ_LINE_AGAIN = """

One property, two values: an earlier pass over this line split %s apart, though together they are one property. Look at the line again and return a single reveal for it, named the way the line itself names it. Its value is only what the line asks that property to be - words on how firm or final the wish is do not count."""

READ_LINE_SPEC = _pda_tool(
    "read_line", "Changes the line brings.",
    "reveals!=pairs drops!=[i] changes!=pairs max_price=n? pick_unavailable!=b no_preference=b not_understood=b "
    "reply_class=reveal|not_picky|split|other quote=s pointers=*",
    {"drops": "Positions in the numbered list of requirements the shopper gave up."}, pointers=_pda_pointers())

# ================================================================ judging opened listings
JUDGE_GUIDE = """Give one row per numbered listing: right_kind, a verdict for each of the shopper's requirements, and a figure when a ranking measure is named.

Verdicts, in the order of the requirements:
- yes: the value shows up, in the same or clearly equivalent words, alone or among other listed values, in an entry for the property or a closely named one (for style, e.g. "skirt style"); or, when no entry concerns the property, the title or description states it.
- no: the item clearly cannot have the value; or entries for the property exist and none of them gives the value, whatever the title claims.
- unclear: nothing on the listing decides it.
Recorded entries decide; title and description count only for a property the listing records nothing about.

right_kind: false only for an accessory or part of the item asked for, or an obviously different product. A listing filed in the store section for this kind counts as the right kind even when its title names a related style.

figure: only when a ranking measure is named - the listing's number for it, or null when it states none. Every figure in one unit, converted where a listing writes another: the unit named with the measure, otherwise one you choose and name in figure_unit.

Each listing arrives with its title, category, description and every property it records; a shopping assistant reads your rows before it buys."""

JUDGE_SPEC = _pda_tool("judge_listings", "Verdict rows for the opened listings.", "listings!=* figure_unit=s?",
                       listings=_pda_list(_pda_object("num!=i right_kind!=b verdicts!=[yes|no|unclear] figure=n?",
                                                      _pda_notes("verdicts: One per requirement, in the requirements' order."))))

_pda_NUDGE = "\n\nRespond through the tool, not in prose."


# ================================================================ talking to the proxy
def _pda_post(client, path, body):
    """(data, status) of one proxy request; status is the failure's HTTP code when the client reports one."""
    verbose = getattr(client, "post_verbose", None)
    if verbose is None:
        return client.post(path, json_data=body), None
    result = verbose(path, json_data=body)
    try:
        code = result.error["status"]
    except (AttributeError, LookupError, TypeError):
        code = None
    if not isinstance(code, int):
        code = None
    return getattr(result, "data", None), code


def _pa_env_post(envelope, clock):
    """One send to the world, its wait limited by the clock."""
    room = clock.left() - FINISH_LEFT
    _env_net.timeout = int(min(ENV_TIMEOUT, room)) if room > 3 else 3
    return _pda_post(_env_net, "/environment/call", envelope)


def post_inference(body, clock=None):
    """One chat completion request: (parsed reply or None, failure status or None)."""
    if clock is not None:
        _net.timeout = clock.model_timeout()
    return _pda_post(_net, "/inference/chat/completions", body)


def _pda_request(model, cap, system, prompt, spec, provider):
    if model in _pda_FREE_CHOICE:
        choice, prompt = "auto", prompt + _pda_NUDGE
    else:
        choice = dict(type="function", function=dict(name=spec["function"]["name"]))
    turns = [dict(role="system", content=system), dict(role="user", content=prompt)]
    body = {"model": model, "temperature": 0, "max_tokens": cap, "messages": turns, "tools": [spec],
            "tool_choice": choice}
    if provider == "chutes":                # its thinking models would spend the cap on hidden reasoning
        body["chat_template_kwargs"] = dict(thinking=False, enable_thinking=False)
    return body


def _pda_call_args(said, wanted):
    """Argument objects of the tool calls that may answer `wanted` (or carry no name), JSON text decoded."""
    for call in said.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") is not None and fn.get("name") != wanted:
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                continue
        if isinstance(args, dict):
            yield args


def _pda_unwrap_values(args):
    # some models send a list or object as a JSON string; such strings are opened when they parse
    packed = [k for k, v in args.items() if isinstance(v, str) and v.lstrip()[:1] in ("[", "{")]
    for key in packed:
        try:
            args[key] = json.loads(args[key])
        except ValueError:
            pass
    return args


def _pda_unpack(reply, wanted):
    try:
        said = reply["choices"][0]["message"]
    except (LookupError, TypeError):
        return None
    for args in _pda_call_args(said, wanted):
        return _pda_unwrap_values(args)
    text = said.get("content")
    start = text.find("{") if isinstance(text, str) else -1
    if start < 0:
        return None
    try:
        return json.loads(text[start:text.rfind("}") + 1])
    except ValueError:
        return None


class Seer:
    """Gets one forced tool call answered by the models of a seat, in turn; None leaves the decision to code.
    Routes that reject a request are benched for the run, a model with two empty replies too; a rate limit pauses
    and repeats the same model without counting the request; the clock says whether another request fits."""
    _pda_dark = 0                     # requests in a row sent to models and answered by none (models down)

    def __init__(self, clock=None):
        self._pda_vendor = getenv("INFERENCE_PROVIDER", "openrouter")
        self._pda_seats = ROUTES.get(self._pda_vendor) or ROUTES["openrouter"]
        self.clock = _PaClock() if clock is None else clock
        self.last_model = ""
        self._pda_used = 0
        self._pda_spent = None        # reported cost so far; None until a reply reports one
        self._pda_empty = {}
        self._pda_benched = set()
        self._pda_vital = False       # the request in hand is one the episode cannot order without
        self._pda_apart = False       # the request in hand is paid from its own purse, not the run's
        self.more_asks = 0            # requests allowed past the per-run count (one order of several products)
        self.part_cap = None          # spend this product of a several-product order may reach; None: the run's
        self._pda_aside = 0.0
        self._pda_kind = ""
        self._pda_dear = {}           # request name -> the dearest reply of that name so far
        self._pda_rate = {}           # model -> its dearest price per character of request text so far
        self._pda_size = 0            # characters of the request in hand
        self._pda_model = ""          # the model the request in hand goes to

    def _pda_room(self, keep):
        """Does one more request of the kind in hand fit under the run's spend, keeping `keep` back? A request is
        expected to cost what the dearest of its kind did, and no less than its text at its model's dearest price
        per character so far (a model not met yet: the dearest of any), with room for the answer; a kind not met
        yet, at least a small request's price."""
        total = (self._pda_spent or 0.0) + self._pda_aside
        rate = self._pda_rate.get(self._pda_model, max(self._pda_rate.values(), default=0.0))
        guess = max(self._pda_dear.get(self._pda_kind, _pda_SPEND_FIRST), 1.5 * rate * self._pda_size)
        limit = _pda_SPEND_PER_RUN
        if self.part_cap is not None and not self._pda_vital:
            # later products of the order keep their share; judging and hearing lines may dip into the part
            # of that share kept back for those same kinds, so the product in hand is never left unjudged
            lift = _pda_SPEND_KEPT if self._pda_kind in _pda_KEPT_FOR else 0.0
            limit = min(limit, self.part_cap + lift)
        return total + guess < limit - keep

    def spent_so_far(self) -> float:
        return (self._pda_spent or 0.0) + self._pda_aside

    def _pda_may_try(self, tries):
        if not (tries < _pda_MODELS_PER_ASK and self.clock.model_room()):
            return False
        priced = self._pda_spent is not None or self._pda_aside > 0
        kept = self._pda_vital or self._pda_kind in _pda_KEPT_FOR
        if priced and not self._pda_room(0.0 if kept else _pda_SPEND_KEPT):
            return False
        if self._pda_apart:
            return self._pda_aside < _pda_SPEND_APART
        if self._pda_vital:           # past the run's request count: one such request is worth it
            return True
        if not priced:
            return self._pda_used < _pda_ASKS_UNPRICED + self.more_asks
        return self._pda_used < _pda_ASKS_PER_RUN + self.more_asks

    def _pda_charge(self, reply):
        cost = number((reply.get("usage") or {}).get("cost")) if isinstance(reply.get("usage"), dict) else None
        if cost is not None:
            self._pda_dear[self._pda_kind] = max(self._pda_dear.get(self._pda_kind, 0.0), cost)
            per = cost / max(1, self._pda_size)
            self._pda_rate[self._pda_model] = max(self._pda_rate.get(self._pda_model, 0.0), per)
        if cost is not None and self._pda_apart:
            self._pda_aside += max(0.0, cost)
        elif cost is not None:
            self._pda_spent = (self._pda_spent or 0.0) + max(0.0, cost)

    def _pda_note_failure(self, model, status):
        if status == _pda_NO_CREDIT:
            self._pda_benched.update(m for seat in self._pda_seats.values() for m in seat)
        elif status in _pda_REFUSED:
            self._pda_benched.add(model)
        elif status != _pda_THROTTLED:
            self._pda_empty[model] = self._pda_empty.get(model, 0) + 1
            if self._pda_empty[model] > 1:
                self._pda_benched.add(model)

    def _pda_send(self, body, pauses, tries):
        """(reply, status, seconds) of one request, repeated after a pause while rate-limited; None when the
        budget or the clock runs out during such a pause."""
        again = True
        while True:
            began = time.monotonic()
            reply, status = post_inference(body, self.clock)
            if status in _pda_GATEWAY and again and self.clock.nap(0.4, MODEL_FLOOR):
                again = False
                continue
            if status != _pda_THROTTLED or not pauses or not self.clock.nap(pauses[0] * (1.0 + (id(body) % 9) / 32), MODEL_FLOOR):
                return reply, status, time.monotonic() - began
            pauses.pop(0)
            if not self._pda_may_try(tries):
                return None

    def ask(self, system, prompt, spec, tier="deep", cap=1500, skip=(), patient=False, vital=False, apart=False):
        """patient: the request matters enough to sit out a longer rate limit. vital: without an answer the episode
        cannot order at all, so the run's request count does not stop it and it may use the kept part of the spend
        (never more than the run's spend, and the clock still rules). apart: paid from a small purse inside the run's
        spend, so it does not use up the run's request count."""
        self._pda_vital, self._pda_apart = vital, apart
        used = self._pda_used
        try:
            return self._pda_ask(system, prompt, spec, tier, cap, skip, patient)
        finally:
            self._pda_vital = self._pda_apart = False
            if apart:
                self._pda_used = used

    def _pda_ask(self, system, prompt, spec, tier, cap, skip, patient):
        fn = spec["function"]
        wanted, must = fn["name"], fn["parameters"].get("required") or []
        self._pda_kind, self._pda_size = wanted, len(system) + len(prompt)
        pauses = _pda_PAUSES + (_pda_PAUSES_PATIENT if patient else [])
        seat = self._pda_seats.get(tier) or self._pda_seats["deep"]
        extra, tries = "", 0
        for model in [m for m in seat if m not in skip]:
            if model in self._pda_benched:
                continue
            self._pda_model = model
            if not self._pda_may_try(tries):
                break
            sent = self._pda_send(_pda_request(model, cap, system, prompt + extra, spec, self._pda_vendor),
                                  pauses, tries)
            if sent is None:
                break
            reply, status, took = sent
            tries += 1
            self._pda_used += status != _pda_THROTTLED
            if not isinstance(reply, dict):
                self._pda_note_failure(model, status)
                continue
            self.clock._pda_lags.append(took)
            self._pda_charge(reply)
            args = _pda_unpack(reply, wanted)
            if isinstance(args, dict) and all(k in args for k in must):
                self.last_model, self._pda_dark = model, 0
                return args
            extra = _pda_NUDGE
        if tries:
            self._pda_dark += 1           # models were asked and none answered; held back by the spend cap is not that
        return None

    def dark(self) -> bool:
        """No model answered the last two requests that were sent: the models are down, code decides alone. (Spend
        used up is not this: the judge has spoken on what was opened, and a decline then is a real one.)"""
        return self._pda_dark >= 2


# ================================================================ reading observations by shape
def _pda_patterns(**sources):
    return {name: re.compile(src, re.I) for name, src in sources.items()}


_pda_KEY = _pda_patterns(
    id=r"(^|[_\s.-])(id|pid|sku|uid|code|key|ref|no|num|number|handle)$|^(id|pid|sku|ref)[_\s.-]|_pid$|^pid|^ref$",
    title=r"title|^name$|product_name|item_name|heading|label$",
    price=r"price|cost|amount|fee",
    in_stock=r"stock|avail|on_?hand|in_?store|sold_?out",
    category=r"categ|department|aisle|section|shelf",
    brand=r"brand|maker|company|manufacturer|vendor",
    description=r"desc|summary|detail|blurb|about",
    next=r"next|continu|cursor|more|further|after|following|token",
    total=r"total|matched|count|found|hits",
    error=r"^(error|issue|problem|fault|refus\w*)$",
    fault=r"(^|_)(error|issue|problem|fault)s?(_|$)",
    refusal=r"refus|reject|denied|declin",
    sold=r"sold|out",
    gone=r"out|none|unavail|no\b|sold",
    pair_name=r"name|key|label|attr|prop",
    pair_value=r"value|val$|text",
)
BRAND_KEY, CATEGORY_KEY = _pda_KEY["brand"], _pda_KEY["category"]
_pda_ID_LOOK = re.compile(r"^(?=[^\s]*\d)[A-Za-z0-9][A-Za-z0-9_.:/-]{2,63}$")   # short, a digit, no blanks
_pda_SLOT_ORDER = tuple(_pda_KEY)[1:7]        # title, price, in_stock, category, brand, description
_pda_SKIP = object()          # the field does not suit this slot; the next slot may take it
_pda_NO_TOKEN = (0, "", False, None)
_pda_LOOSE = "_loose"         # a record's unnamed texts, held only until find_records has seen the whole reply


def _pda_plain(raw):
    return isinstance(raw, (str, int)) and raw is not True and raw is not False


def _pda_id_of(fields):
    for field, val in fields.items():
        if not (_pda_plain(val) and _pda_KEY["id"].search(str(field))):
            continue
        # a number always serves; a text must look like an id (at least three characters, so never blank)
        if isinstance(val, int) or _pda_ID_LOOK.match(val.strip()):
            return str(val)
    return None


def _pda_flat_table(node):
    """(size, table) for something that may hold properties: a list of name/value entries or a flat map."""
    if isinstance(node, dict):
        nested = [x for x in node.values() if isinstance(x, (dict, list))]
        if not node or nested:
            return None
        return len(node), {str(label): str(x) for label, x in node.items() if x is not None}
    if not (isinstance(node, list) and node) or any(not isinstance(e, dict) for e in node):
        return None
    table = {}
    for entry in node:
        names, values = [], []
        for label, x in entry.items():
            if _pda_KEY["pair_name"].search(str(label)):
                names.append(x)
            if _pda_KEY["pair_value"].search(str(label)):
                values.append(x)
        if names and values:
            table[str(names[0])] = str(values[0])
    return len(table), table


def _traits_of(fields):
    """The biggest property table nested in a listing's fields."""
    chosen = {}
    for held in fields.values():
        found = _pda_flat_table(held)
        if found and found[0] > len(chosen):
            chosen = found[1]
    return chosen


def _pda_slot_value(slot, key, raw):
    """What one plain field gives `slot`: a value, None (claimed but empty) or _pda_SKIP."""
    if slot == "title":
        return raw if isinstance(raw, str) else _pda_SKIP
    if slot != "in_stock":
        if slot == "price":
            raw = number(raw)
            return _pda_SKIP if raw is None else raw
        return _pda_SKIP if raw is None else str(raw)
    # a stock-named field always takes the slot, even when its value tells nothing
    if isinstance(raw, bool):
        return raw != bool(_pda_KEY["sold"].search(key))
    count = number(raw)
    if count is not None:
        return count > 0
    if isinstance(raw, str):
        return not _pda_KEY["gone"].search(raw)
    return None


def as_record(fields):
    ident = _pda_id_of(fields)
    if ident is None:
        return None
    card = {"id": ident}
    plain = ((str(k), v) for k, v in fields.items() if not isinstance(v, (list, dict)))
    for name, raw in plain:
        if _pda_KEY["id"].search(name):
            continue
        for slot in (s for s in _pda_SLOT_ORDER if s not in card and _pda_KEY[s].search(name)):
            got = _pda_slot_value(slot, name, raw)
            if got is not _pda_SKIP:
                if got is not None:
                    card[slot] = got
                break
    traits = _traits_of(fields)
    if traits:
        card["traits"] = traits
    if not {"title", "price", "traits"} & card.keys():
        return None
    if "title" not in card:
        # no field is named like a title: the texts no slot took wait for find_records to tell title from blurb
        taken = {k for k in fields if any(_pda_KEY[s].search(str(k)) for s in ("id",) + _pda_SLOT_ORDER)}
        loose = {str(k): v.strip() for k, v in fields.items() if k not in taken and isinstance(v, str) and " " in v.strip()}
        if loose:
            card[_pda_LOOSE] = loose
    return card


def _pda_heading_shape(text):
    """How much a text looks like a heading: the share of its words written with a capital, less one when it ends
    like a sentence."""
    found = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    share = sum(w[0].isupper() for w in found) / len(found) if found else 0.0
    return share - (1.0 if text.rstrip()[-1:] in ".!?" else 0.0)


def _pda_shape_texts(cards):
    """Titles and blurbs for records that name neither, decided per reply: the loose text field that reads most like
    a heading is the title, the longest of the others the blurb (a lone field is the title)."""
    held = [c for c in cards if _pda_LOOSE in c]
    if not held:
        return cards
    looks, sizes = {}, {}
    for card in held:
        for k, v in card[_pda_LOOSE].items():
            looks[k] = looks.get(k, 0.0) + _pda_heading_shape(v)
            sizes[k] = sizes.get(k, 0) + len(v)
    head = max(looks, key=lambda k: (looks[k] / len(held), k))
    rest = [k for k in sizes if k != head]
    blurb = max(rest, key=lambda k: (sizes[k], k)) if rest else None
    for card in held:
        loose = card.pop(_pda_LOOSE)
        if loose.get(head):
            card["title"] = loose[head]
        if blurb and "description" not in card and loose.get(blurb):
            card["description"] = loose[blurb]
    return cards


def _pda_scan(node, key=""):
    """Walks an observation in document order: yields (card, None) for each record, whose insides are not entered,
    and (None, (text, key)) for each string, with the name of the field that holds it."""
    if isinstance(node, str):
        yield None, (node, key)
    elif isinstance(node, list):
        for part in node:
            yield from _pda_scan(part, key)
    elif isinstance(node, dict):
        card = as_record(node)
        if card is not None:
            yield card, None
            return
        for name, part in node.items():
            yield from _pda_scan(part, str(name))


def find_records(tree):
    """Every listing record of an observation, in document order."""
    return _pda_shape_texts([card for card, _ in _pda_scan(tree) if card is not None])


def find_notes(tree):
    """Sentences outside the listing records: what the shopper or the store says."""
    kept = []
    for card, found in _pda_scan(tree):
        if card is not None:
            continue
        text, holder = found[0].strip(), found[1]
        complaint = _pda_KEY["error"].search(holder) or _pda_KEY["fault"].search(holder)
        if " " in text and len(text) > 11 and complaint is None:
            kept.append(text)
    return kept


def error_of(reply, need_data=False):
    """The world's complaint about a call, or "". need_data: the call should have brought listings or a sheet, so
    one bare sentence under a key of the world's own naming (and nothing else) is read as a refusal too."""
    if not isinstance(reply, dict):
        return ""
    found = (v for k, v in reply.items() if _pda_KEY["error"].search(str(k)) and _pda_plain(v)
             or _pda_KEY["fault"].search(str(k)) and isinstance(v, str) and " " in v.strip())
    named = str(next(found, ""))
    if not named and not need_data:
        flags = [k for k, v in reply.items() if isinstance(v, bool)]
        if flags and not any(reply[k] for k in flags) and not find_records(reply):
            # every yes/no flag of the reply is false and nothing came back: the call did not take effect
            said = [str(v) for v in reply.values() if isinstance(v, str) and v.strip()]
            return (said[0] if said else "%s: false" % flags[0])[:200]
    if named or not need_data or len(reply) != 1 or find_records(reply):
        return named
    only = next(iter(reply.values()))
    return only.strip() if isinstance(only, str) and " " in only.strip() else ""


def refusal_of(reply):
    if isinstance(reply, dict):
        for field, got in reply.items():
            if not got or not _pda_KEY["refusal"].search(str(field)):
                continue
            if isinstance(got, list):
                return list(map(str, got))
            return [got if isinstance(got, str) else json.dumps(got)]
    return []


def paging_of(reply):
    """(next-page token, total match count) of a search reply, each None when not reported; nested maps are
    tried in order when the top level reports neither."""
    if not isinstance(reply, dict):
        return None, None
    found = [None, None]
    for field, val in reply.items():
        label = str(field)
        if _pda_KEY["next"].search(label) and val not in _pda_NO_TOKEN:
            found[0] = val
            continue
        figure = None if isinstance(val, bool) else number(val)
        if figure is not None and _pda_KEY["total"].search(label):
            found[1] = figure
    if found == [None, None]:
        for inner in filter(lambda v: isinstance(v, dict), reply.values()):
            found = list(paging_of(inner))
            if found != [None, None]:
                break
    return found[0], found[1]


# ============================================================================ store tools and the jobs they do
# Value slots a role cannot work without.
_pdb_VITAL = dict(keyword_search="text_arg", open_listing="id_arg", open_many="ids_arg", basket_add="id_arg",
                  basket_remove="id_arg", message="text_arg", decline="text_arg", open_offer="id_arg")
# Parameter-name hints for a vital slot the proposal left empty (a list of ids has none).
_pdb_HINTS = dict(
    id_arg=r"id|ref|item|product|article|listing|sku|target|code",
    text_arg=r"text|query|q$|term|keyword|word|phrase|search|note|body|message|remark|reason|content|question",
)
_pdb_NOT_A_HANDLE = r"categ|aisle|depart|section|shelf|group|class|brand|maker|make|company|manufacturer|vendor"
# JSON types each value slot may go into; a slot missing here takes any type.
_pdb_WORD, _pdb_FIGURE = {"string"}, {"integer", "number"}
_pdb_KINDS_FOR = dict(text_arg=_pdb_WORD, category_arg=_pdb_WORD, brand_arg=_pdb_WORD,
                      id_arg={"string", "integer"}, qty_arg=_pdb_FIGURE, price_arg=_pdb_FIGURE)
_pdb_PAGING = r"page|^pg$|^p$|offset|cursor|token|after|continu|next"
_pdb_ENDS = r"give up|no purchase|close out|abandon|decline|final|check ?out|settle|purchase|place (?:the |an )?order"
_pdb_QUIT = r"give up|no purchase|close out|abandon|decline|without buy"
_pdb_CART = r"cart|basket|bag"


def _pdb_kinds(spec):
    kind = spec.get("type")
    if isinstance(kind, list):
        return set(kind)
    return {kind} if kind else set()


def _pdb_fits_field(slot, spec):
    """Could this parameter carry the slot? Words and ids are strings, amounts are numbers."""
    have = _pdb_kinds(spec)
    if slot == "ids_arg":
        inner = (spec.get("items") or {}).get("type")
        return "array" in have and (not inner or inner in ("string", "integer"))
    accepted = _pdb_KINDS_FOR.get(slot)
    if accepted is None:
        return True
    if accepted is not _pdb_FIGURE and spec.get("enum"):
        return slot == "category_arg"         # shelves from a fixed list: the closest option is sent later
    return bool(have & accepted) if have else True


def _pdb_one_option(spec):
    opts = spec.get("enum") if isinstance(spec, dict) else None
    return isinstance(opts, list) and len(opts) == 1


def _pdb_line_shape(spec, named=(), path=()):
    """Where a parameter takes basket lines: an array, at any depth below objects, whose items are objects holding a
    listing id (a string) and, if any, one whole-number count. (path inside the parameter, id key, count key) or
    None. Of several strings the id is the one the proposal named, else the only required one, else the only one;
    an object on the way may require nothing but its next step, and a line nothing but id, count and fixed values."""
    if not isinstance(spec, dict):
        return None
    kinds = _pdb_kinds(spec)
    if "array" in kinds:
        line = spec.get("items") if isinstance(spec.get("items"), dict) else {}
        fields = line.get("properties") or {}
        if not fields:
            return None
        words = [k for k, v in fields.items() if "string" in _pdb_kinds(v) and not v.get("enum")]
        counts = [k for k, v in fields.items() if _pdb_kinds(v) & {"integer"}]
        need = set(line.get("required") or [])
        pick = [w for w in words if w in named] or [w for w in words if w in need] or words
        if len(pick) != 1 or len(counts) > 1:
            return None
        count = counts[0] if counts else ""
        if any(k not in (pick[0], count) and not _pdb_one_option(fields.get(k)) for k in need):
            return None
        return tuple(path), pick[0], count
    for key, sub in (spec.get("properties") or {}).items():
        need = spec.get("required") or []
        if any(k != key and not _pdb_one_option((spec.get("properties") or {}).get(k)) for k in need):
            continue
        found = _pdb_line_shape(sub, named, path + (key,))
        if found:
            return found
    return None


_pdb_TYPE_TEST = dict(string=lambda v: isinstance(v, str), boolean=lambda v: isinstance(v, bool),
                      integer=lambda v: isinstance(v, int) and not isinstance(v, bool),
                      number=lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
                      array=lambda v: isinstance(v, list), object=lambda v: isinstance(v, dict), null=lambda v: v is None)


def _pdb_shape_fault(spec, val, where="$"):
    """Where a call value breaks its schema - a required field missing, a wrong type, a list too short or too long,
    a value off a fixed list - at any depth; "" when it fits."""
    if not isinstance(spec, dict) or not spec:
        return ""
    opts = spec.get("enum")
    if isinstance(opts, list) and opts and val not in opts:
        return where + " is not one of the options"
    tests = [_pdb_TYPE_TEST[k] for k in _pdb_kinds(spec) if k in _pdb_TYPE_TEST]
    if tests and not any(test(val) for test in tests):
        return where + " has the wrong type"
    if isinstance(val, dict):
        fields = spec.get("properties") or {}
        gap = next((k for k in spec.get("required") or [] if k not in val), None)
        if gap is not None:
            return "%s.%s is missing" % (where, gap)
        for key, inner in val.items():
            fault = _pdb_shape_fault(fields.get(key) or {}, inner, "%s.%s" % (where, key))
            if fault:
                return fault
    elif isinstance(val, list):
        least, most = number(spec.get("minItems")), number(spec.get("maxItems"))
        if (least is not None and len(val) < least) or (most is not None and len(val) > most):
            return where + " has the wrong number of entries"
        for n, inner in enumerate(val):
            fault = _pdb_shape_fault(spec.get("items") or {}, inner, "%s[%d]" % (where, n))
            if fault:
                return fault
    return ""


def _pdb_count_spots(args, pid, count):
    """Paths in a model-written cart call where the quantity sits: the value under the listing id used as a key,
    or a whole number beside the listing id. None when the listing id appears nowhere in the call."""
    spots, seen = [], [False]

    def walk(obj, path):
        if isinstance(obj, dict):
            if pid in obj:
                seen[0] = True
                if isinstance(obj[pid], int) and not isinstance(obj[pid], bool) and obj[pid] == count:
                    spots.append(path + (pid,))
            if any(v == pid for v in obj.values()):
                seen[0] = True
                spots.extend(path + (k,) for k, v in obj.items()
                             if isinstance(v, int) and not isinstance(v, bool) and v == count)
            for k, v in obj.items():
                walk(v, path + (k,))
        elif isinstance(obj, list):
            seen[0] = seen[0] or pid in obj
            for n, v in enumerate(obj):
                walk(v, path + (n,))

    walk(args, ())
    return spots if seen[0] else None


def _pdb_from_template(template, pid, count):
    """A cart call shaped like the model-written one: its quantity spots set to the count, then the listing id it
    was written for replaced (as a value or a key) by this one."""
    shape, old, spots = template
    call = json.loads(json.dumps(shape))
    for path in spots if count is not None else ():
        holder = call
        for key in path[:-1]:
            holder = holder[key]
        holder[path[-1]] = count

    def swap(obj):
        if isinstance(obj, dict):
            return {(str(pid) if k == old else k): swap(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [swap(v) for v in obj]
        return str(pid) if obj == old else obj

    return swap(call) if pid is not None else call


def _pdb_named_parts(text):
    """The name pieces of a parameter path as a model may write it (outer[].inner, outer.inner, inner)."""
    return tuple(piece for piece in re.split(r"[^\w-]+", str(text or "")) if piece)


def _pdb_coerce(spec, val):
    """A pinned value in the schema's own type or enum spelling (None when no option matches)."""
    opts = spec.get("enum")
    if opts:
        key = norm(val)
        return next((o for o in opts if norm(o) == key), None)
    kind = spec.get("type")
    if kind == "boolean":
        return norm(val) in ("true", "1", "yes")
    if kind not in ("integer", "number"):
        return str(val)
    amount = number(val)
    if amount is None or kind == "number":
        return amount
    return int(amount)


def _pdb_closest_option(spec, val):
    """For a shelf name: the enum option sharing most stems with it; None on no overlap or a tie."""
    target = set(stems(val))

    def score(opt):
        own = set(stems(opt))
        return len(target & own), -len(own - target), str(opt)

    board = sorted(map(score, spec.get("enum") or []), reverse=True)
    if not board or not board[0][0]:
        return None
    if len(board) > 1 and board[0][:2] == board[1][:2]:
        return None
    winner = board[0][2]
    return next(o for o in spec["enum"] if str(o) == winner)


def _pdb_clamp(spec, val):
    amount = number(val)
    if amount is None:
        return None
    to = float if spec.get("type") == "number" else int
    amount = to(amount)
    if "maximum" in spec:
        amount = min(amount, to(spec["maximum"]))
    if "minimum" in spec:
        amount = max(amount, to(spec["minimum"]))
    return amount


def _pdb_id_shaped(spec, val):
    """An id as the schema types it: a whole number when the field takes numbers and the id is all digits, else text."""
    kind = spec.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    text = str(val)
    if ("integer" in kinds or "number" in kinds) and "string" not in kinds and text.isdigit():
        return int(text)
    return text


def _pdb_fit_value(spec, val):
    """A call value adjusted to its parameter: number bounds, enum spelling, list size, text length."""
    kind = spec.get("type")
    if kind in ("integer", "number"):
        return _pdb_clamp(spec, val)
    if spec.get("enum"):
        return _pdb_coerce(spec, val)
    if kind == "array" or (isinstance(kind, list) and "array" in kind):
        many = list(val) if isinstance(val, (list, tuple)) else [val]
        many = [_pdb_id_shaped(spec.get("items") or {}, v) for v in many]
        most = spec.get("maxItems")
        return many[:int(most)] if isinstance(most, int) else many
    line, room = str(val), spec.get("maxLength")
    if not isinstance(room, int) or len(line) <= room:
        return line
    head = line[:room]
    return head.rsplit(" ", 1)[0] if " " in head else head


def _pdb_first(params, test):
    """First parameter name passing the test (a regex over the name, or a callable); "" when none does."""
    check = (lambda n: re.search(test, n, re.I)) if isinstance(test, str) else test
    return next((n for n in params if check(n)), "")


def _pdb_enum_pins(params, pattern):
    """Per parameter, the last enum option whose text matches the pattern."""
    pins = {}
    for pname, spec in params.items():
        hits = [o for o in spec.get("enum") or [] if re.search(pattern, str(o), re.I)]
        if hits:
            pins[pname] = hits[-1]
    return pins


def _pb_page_ref(token, read, numeric=False):
    """Page argument for the following page: the world's own token as given (a number, text or a structured
    cursor); a bare "more follows" flag becomes the next page number only when the page parameter is a number in
    the schema, else None (never a guessed format)."""
    if isinstance(token, bool) or token is None or isinstance(token, float):
        return read + 1 if numeric and token is not False and token is not None else None
    return token


# ---------------------------------------------------------------- drafts built from tool names and descriptions
def _pdb_draft_search(tm, tname, blurb, params):
    return dict(text_arg=_pdb_first(params, r"text|query|q$|term|keyword|word|search"),
                fixed=_pdb_enum_pins(params, r"typed|text|keyword|word|query|free"))


def _pdb_draft_listing(tm, tname, blurb, params):
    if len(params) <= 2:
        return dict(id_arg=_pdb_first(params, r"id|item|product|target|listing|sku"), fixed={})
    return None


def _pdb_cart_drafter(verb, bare_ok):
    """Cart add/remove; a tool with several modes needs its mode pinned by an enum option."""
    def draft(tm, tname, blurb, params):
        pins = _pdb_enum_pins(params, verb)
        multi = any(s.get("enum") for s in params.values())
        if not (pins or re.search(verb, blurb)) and (not bare_ok or multi):
            return None
        return dict(id_arg=_pdb_first(params, r"id|item|product|chosen|listing|sku"),
                    qty_arg=_pdb_first(params, r"qty|quant|how_?many|count|number|units"), fixed=pins)
    return draft


def _pdb_draft_view(tm, tname, blurb, params):
    shows = re.search(r"view|show|review|content|see|list|your cart", blurb)
    return dict(fixed={}) if shows and not tm.required(tname) else None


def _pdb_draft_order(tm, tname, blurb, params):
    if re.search(r"give up|decline|no purchase|without", blurb):
        return None
    return dict(id_arg=_pdb_first(params, r"id|item|product|listing|sku"),
                qty_arg=_pdb_first(params, r"qty|quant|how_?many|count|units"), fixed={})


def _pdb_draft_note(tm, tname, blurb, params):
    return dict(text_arg=_pdb_first(params, r"text|note|body|remark|message|reason|content"), fixed={})


# role, a description cue, a ruling-out cue, the drafter; tried in this order
_pdb_NAME_RULES = (
    ("keyword_search", r"search|find|query|look ?up", _pdb_ENDS, _pdb_draft_search),
    ("open_listing", r"detail|datasheet|propert|spec|attribute|view|inspect|listing|item info",
     _pdb_ENDS + "|" + _pdb_CART, _pdb_draft_listing),
    ("basket_add", _pdb_CART, _pdb_ENDS, _pdb_cart_drafter(r"add|include|put|insert|plus|keep", True)),
    ("basket_remove", _pdb_CART, _pdb_ENDS, _pdb_cart_drafter(r"remov|discard|delete|drop|minus|take", False)),
    ("basket_view", _pdb_CART, _pdb_ENDS, _pdb_draft_view),
    ("checkout", r"check ?out|purchase|buy|order|settle|pay", _pdb_QUIT, _pdb_draft_order),
    ("decline", _pdb_QUIT, r"question|ask the|message the|check ?out|purchase (?:the|your) cart", _pdb_draft_note),
    ("message", r"message|ask|question|shopper|customer|write|contact|chat", _pdb_ENDS, _pdb_draft_note),
)


class _pdb_NameGuesser:
    """Fills the roles nobody named by reading tool names, descriptions and parameter shapes."""

    def __init__(self, tm):
        self.tm = tm
        self.memo = {}

    def blurb(self, tname):
        if tname not in self.memo:
            spoken = tname.replace("_", " ") + " " + self.tm.tools[tname]["description"]
            self.memo[tname] = norm(spoken)
        return self.memo[tname]

    def run(self):
        tm = self.tm
        for duty, cue, veto, drafter in _pdb_NAME_RULES:
            if duty not in tm.roles:
                self._by_words(duty, cue, veto, drafter)
        if "open_many" not in tm.roles:
            self._many()
        searcher = tm.roles.get("keyword_search")
        if searcher and "filter_search" not in tm.roles:
            self._filters(searcher["tool"])
        for duty in ("keyword_search", "filter_search"):
            job = tm.roles.get(duty)
            if job and not job.get("page_arg"):
                pager = _pdb_first(tm.props(job["tool"]), _pdb_PAGING) or self._page_from_shape(job)
                if pager:
                    job["page_arg"] = pager

    def _by_words(self, duty, cue, veto, drafter):
        for tname in self.tm.tools:
            said = self.blurb(tname)
            if not re.search(cue, said) or re.search(veto, said):
                continue
            sketch = drafter(self.tm, tname, said, self.tm.props(tname))
            if sketch is not None and self.tm._pdb_try(duty, tname, sketch):
                return

    def _many(self):
        for tname in self.tm.tools:
            said = self.blurb(tname)
            if re.search(_pdb_ENDS, said) or re.search(r"cart|basket|bag|trolley", said):
                continue
            params = self.tm.props(tname)
            lists = [p for p in params if _pdb_fits_field("ids_arg", params[p])]
            if len(params) <= 2 and len(lists) == 1 and self.tm._pdb_try("open_many", tname, dict(ids_arg=lists[0], fixed={})):
                return

    def _filters(self, tname):
        """The word search frequently takes shelf and price filters too."""
        params = self.tm.props(tname)
        pricey = re.compile(r"price|cap|budget|max|spend|ceil|limit|upto|most|cost", re.I)
        sketch = dict(
            category_arg=_pdb_first(params, CATEGORY_KEY.search),
            price_arg=_pdb_first(params, lambda n: pricey.search(n) and _pdb_fits_field("price_arg", params[n])),
            brand_arg=_pdb_first(params, BRAND_KEY.search),
            page_arg=_pdb_first(params, _pdb_PAGING),
            fixed=_pdb_enum_pins(params, r"field|filter|facet|struct"))
        if sketch["category_arg"] or sketch["price_arg"]:
            self.tm._pdb_try("filter_search", tname, sketch)

    def _page_from_shape(self, job):
        """Paging read from the schema only: an optional whole number starting at 0 or 1, else a short optional
        string that is not in use yet."""
        tname = job["tool"]
        params, must = self.tm.props(tname), set(self.tm.required(tname))
        busy = set(job.get("fixed") or {})
        busy.update(v for k, v in job.items() if k.endswith("_arg") and isinstance(v, str))
        spare = [p for p in params if p not in must and p not in busy and not params[p].get("enum")]
        for p in spare:
            if params[p].get("type") == "integer" and number(params[p].get("minimum")) in (0, 1):
                return p
        for p in spare:
            if params[p].get("type") == "string" and (number(params[p].get("maxLength")) or 999) <= 32:
                return p
        return None


class ToolMap:
    """Role -> job: the tool, the parameter for each value slot, and the values pinned for that role."""

    def __init__(self, tools):
        self.tools = {}
        self.roles: dict[str, dict] = {}
        self.rejected: dict[str, str] = {}     # role -> why its last model proposal was refused
        self._pdb_barred: set = set()          # (role, tool) pairs that failed in use
        self._pdb_from_model = False           # refusal notes are kept for model proposals only
        self.unplanned: dict[str, tuple] = {}   # role -> (tool, pinned values) a model named but code could not shape
        for entry in tools:
            fn = entry.get("function", entry)
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            self.tools[fn["name"]] = dict(description=fn.get("description") or "",
                                          parameters=fn.get("parameters") or {})

    # ---------------------------------------------------------------- reading the schemas
    def _pdb_sheet(self, tname):
        return self.tools.get(tname, {}).get("parameters") or {}

    def props(self, tname):
        return self._pdb_sheet(tname).get("properties") or {}

    def required(self, tname):
        return list(self._pdb_sheet(tname).get("required") or [])

    def has(self, duty):
        return duty in self.roles

    def _pdb_cards(self, names):
        cards = []
        for n in names:
            info = self.tools[n]
            cards.append(dict(name=n, description=info["description"], parameters=info["parameters"]))
        return cards

    def schema_text(self):
        return json.dumps(self._pdb_cards(self.tools))

    def open_enums(self):
        """Tool -> {required parameter: options} for required enums with two or more options."""
        found = {}
        for tname in self.tools:
            params = self.props(tname)
            many = {p: list(params.get(p, {}).get("enum") or []) for p in self.required(tname)}
            many = {p: o for p, o in many.items() if len(o) > 1}
            if many:
                found[tname] = many
        return found

    def gaps(self):
        """Open roles the episode needs, plus open roles whose proposal was refused."""
        vital = ["checkout", "decline"]
        if not any(self.has(d) for d in ("keyword_search", "filter_search")):
            vital.append("keyword_search")
        if not (self.roles.get("checkout") or {}).get("id_arg"):
            vital.append("basket_add")
        return [d for d in ROLE_FIELDS if d not in self.roles
                and (d in vital or (d in self.rejected and d not in _pdb_OPTIONAL))]

    # ---------------------------------------------------------------- taking proposals
    def adopt(self, answer, open_only=False):
        """Keep each model proposal that passes the schema check (open_only: roles still open only), then guess
        the rest from names and settle the two endings."""
        offers = answer or {}
        self._pdb_from_model = True
        for duty in ROLE_FIELDS:
            offer = offers.get(duty)
            if isinstance(offer, dict) and not (open_only and duty in self.roles):
                job = self._pdb_plan(duty, offer)
                if job:
                    self.roles[duty] = job
                    self.rejected.pop(duty, None)
        self._pdb_from_model = False
        self._pdb_refill()

    def drop(self, duty, why):
        """A role that misbehaved in use: unmapped, its tool barred for it, the gap filled again."""
        job = self.roles.pop(duty, None)
        if job:
            self._pdb_barred.add((duty, job["tool"]))
            self.rejected[duty] = "%s %s" % (job["tool"], why)
        self._pdb_refill()

    def _pdb_refill(self):
        _pdb_NameGuesser(self).run()
        self._pdb_split_finish()
        self._pdb_settle_endings()

    def _pdb_split_finish(self):
        """Endings neither the model nor the names could map: a tool whose one required parameter is a two-way choice,
        each option named in its description, is both endings. The option whose own part of the description shares
        more words with the cart tools' names and descriptions buys, the other closes; a tie maps nothing. With one
        ending already on such a tool, the other ending takes the remaining option."""
        buy, close = self.roles.get("checkout"), self.roles.get("decline")
        if buy and close:
            return
        held = {job["tool"] for job in self.roles.values()}
        cart = set()
        for duty in ("basket_add", "basket_view", "basket_remove"):
            name = (self.roles.get(duty) or {}).get("tool")
            if name:
                cart |= set(stems(label_words(name) + " " + self.tools[name]["description"]))
        for tname in self.tools:
            req = self.required(tname)
            opts = [o for o in (self.props(tname).get(req[0], {}).get("enum") or [])
                    if isinstance(o, str)] if len(req) == 1 else []
            if len(opts) != 2:
                continue
            mine = buy or close
            if mine:
                if mine["tool"] == tname and mine["fixed"].get(req[0]) in opts:
                    other = [o for o in opts if o != mine["fixed"][req[0]]][0]
                    self._pdb_try("decline" if buy else "checkout", tname, {"fixed": {req[0]: other}})
                    return
                continue
            if tname in held:
                continue
            text = norm(self.tools[tname]["description"])
            spots = [re.search(r"\b%s\b" % re.escape(norm(o)), text) for o in opts]
            if not all(spots):
                continue
            at = sorted((hit.start(), o) for hit, o in zip(spots, opts))
            part = {o: text[start:(at[n + 1][0] if n + 1 < len(at) else len(text))].split(".")[0]
                    for n, (start, o) in enumerate(at)}
            a, b = opts
            score_a = len((set(stems(part[a])) - set(stems(part[b]))) & cart)
            score_b = len((set(stems(part[b])) - set(stems(part[a]))) & cart)
            if score_a == score_b:
                continue
            order, end = (a, b) if score_a > score_b else (b, a)
            if self._pdb_try("checkout", tname, {"fixed": {req[0]: order}}):
                if not self._pdb_try("decline", tname, {"fixed": {req[0]: end}}):
                    self.roles.pop("checkout", None)
            return

    def _pdb_note(self, duty, why):
        if self._pdb_from_model:
            self.rejected[duty] = why

    def _pdb_try(self, duty, tname, sketch):
        pairs = [dict(param=k, value=v) for k, v in sketch["fixed"].items()]
        job = self._pdb_plan(duty, dict(sketch, tool=tname, fixed=pairs))
        if job:
            self.roles[duty] = job
        return bool(job)

    def _pdb_plan(self, duty, offer):
        """A proposal checked against the tool's schema: a job, or None when the tool cannot do the role."""
        tname = str(offer.get("tool") or "")
        if tname not in self.tools:
            if tname:
                self._pdb_note(duty, "unknown tool %s" % tname)
            return None
        if (duty, tname) in self._pdb_barred:
            return None
        params = self.props(tname)
        pins: dict[str, Any] = {}
        for pair in offer.get("fixed") or []:
            if not isinstance(pair, dict) or pair.get("param") not in params:
                continue
            cast = _pdb_coerce(params[pair["param"]], pair.get("value"))
            if cast is not None:
                pins[pair["param"]] = cast
        job: dict[str, Any] = dict(tool=tname, fixed=pins)
        slots = ROLE_FIELDS[duty]
        for slot in slots:
            pname = str(offer.get(slot) or "")
            if pname and pname in params and pname not in pins and _pdb_fits_field(slot, params[pname]):
                job[slot] = pname
        if duty.startswith("basket_") and "id_arg" not in job:
            self._pdb_line_slot(params, job, offer)
        if not self._pdb_vital_slot(duty, tname, params, job, slots):
            return None
        return self._pdb_pin_rest(duty, tname, params, job, slots)

    def _pdb_line_slot(self, params, job, offer):
        """A cart tool whose listing id sits inside a list of line objects: job["lines"] = (parameter path to the
        list, id key, count key)."""
        named = _pdb_named_parts(offer.get("id_arg")) + _pdb_named_parts(offer.get("qty_arg"))
        for pname, spec in params.items():
            if pname in job["fixed"]:
                continue
            found = _pdb_line_shape(spec, named)
            if found:
                job["lines"] = ((pname,) + found[0], found[1], found[2])
                return

    def _pdb_vital_slot(self, duty, tname, params, job, slots):
        must = _pdb_VITAL.get(duty)
        if not must or must in job or job.get("lines"):
            return True
        busy = set(job["fixed"]) | {job[s] for s in slots if s in job}
        spare = self._pdb_free_param(tname, params, must, busy)
        if spare:
            job[must] = spare
            return True
        if duty == "decline":                 # closing may go without a remark
            return True
        if self._pdb_from_model and duty == "basket_add":
            self.unplanned[duty] = (tname, dict(job["fixed"]))   # a model may still write this call (_PddCart.fill)
        self._pdb_note(duty, "the %s slot fits no parameter of %s" % (must, tname))
        return False

    def _pdb_pin_rest(self, duty, tname, params, job, slots):
        """Every required parameter still without a value must be a one-option enum, pinned here."""
        pins = job["fixed"]
        done = set(pins) | {job[s] for s in slots if s in job} | ({job["lines"][0][0]} if job.get("lines") else set())
        for pname in self.required(tname):
            if pname in done:
                continue
            opts = params.get(pname, {}).get("enum")
            if opts and len(opts) == 1:
                pins[pname] = opts[0]
                continue
            choices = " - options %s" % ", ".join(map(str, opts)) if opts else ""
            self._pdb_note(duty, "%s needs %s, which got no value%s" % (tname, pname, choices))
            return None
        return job

    def _pdb_free_param(self, tname, params, slot, busy):
        """An unused parameter able to carry the slot: the sole required one, a well-named one, or the only one."""
        loose = [p for p in params if p not in busy and _pdb_fits_field(slot, params[p])]
        if not loose:
            return ""
        must = self.required(tname)
        sole = [p for p in loose if p in must]
        hint = _pdb_HINTS[slot]
        if len(sole) == 1:
            return sole[0]
        good = next((p for p in loose if re.search(hint, p, re.I) and not re.search(_pdb_NOT_A_HANDLE, p, re.I)), "")
        if good:
            return good
        return loose[0] if len(loose) == 1 else ""

    def take_template(self, duty, tname, pins, args, pid, count):
        """Keep a model-written call as the pattern for a role; False when it does not carry the listing id."""
        spots = _pdb_count_spots(args, pid, count)
        if spots is None:
            return False
        self.roles[duty] = dict(tool=tname, fixed=dict(pins), template=(json.loads(json.dumps(args)), pid, spots))
        self.unplanned.pop(duty, None)
        self.rejected.pop(duty, None)
        return True

    # ---------------------------------------------------------------- the two endings
    def _pdb_settle_endings(self):
        """Ordering and closing must be different calls, and neither may equal the message or the cart view; when
        one tool does both endings the deciding value is always sent. A role breaking this is unmapped."""
        buy, close = self.roles.get("checkout"), self.roles.get("decline")
        if buy and close and buy["tool"] == close["tool"]:
            self._pdb_pin_twin(buy, close)
            if buy["fixed"] == close["fixed"]:
                note = "same call (%s, same values) for buying and for closing empty-handed" % buy["tool"]
                for duty in ("checkout", "decline"):
                    self.roles.pop(duty, None)
                    self.rejected[duty] = note
        for duty, other in (("decline", "message"), ("checkout", "basket_view"), ("decline", "basket_view"),
                            ("checkout", "message")):
            mine, theirs = self.roles.get(duty), self.roles.get(other)
            if mine and theirs and (mine["tool"], mine["fixed"]) == (theirs["tool"], theirs["fixed"]):
                del self.roles[duty]
                self.rejected[duty] = "same call as %s (%s, same values)" % (other, mine["tool"])

    def _pdb_pin_twin(self, buy, close):
        """One tool, two endings: a deciding value pinned on one side only gives the other side the remaining
        option of a two-way enum or boolean."""
        params = self.props(buy["tool"])
        for src, dst in ((buy, close), (close, buy)):
            for pname, val in list(src["fixed"].items()):
                if pname in dst["fixed"]:
                    continue
                spec = params.get(pname) or {}
                opts = spec.get("enum") or ([True, False] if spec.get("type") == "boolean" else [])
                others = [o for o in opts if o != val]
                if len(opts) == 2 and len(others) == 1:
                    dst["fixed"][pname] = others[0]

    # ---------------------------------------------------------------- building a call
    def call_args(self, duty, **slots):
        job = self.roles.get(duty)
        if not job:
            return None
        if job.get("template"):
            return job["tool"], _pdb_from_template(job["template"], slots.get("id_arg"), slots.get("qty_arg"))
        params = self.props(job["tool"])
        payload = dict(job["fixed"])
        for slot, val in slots.items():
            pname = job.get(slot)
            if pname and val is not None:
                spec = params.get(pname, {})
                shaped = _pdb_fit_value(spec, val)
                if shaped is None and slot == "category_arg":
                    shaped = _pdb_closest_option(spec, val)
                if shaped is not None:
                    payload[pname] = shaped
        if job.get("lines"):
            wanted = slots.get("lines_arg")
            if wanted is None:
                wanted = [(slots.get("id_arg"), slots.get("qty_arg"))] if slots.get("id_arg") is not None else []
            path = job["lines"][0]
            payload[path[0]] = self._pdb_lines_value(params, job["lines"], wanted)
        return job["tool"], payload

    def _pdb_lines_value(self, params, lines, wanted):
        """The nested value for a lines job: [{id, count, fixed values}] wrapped in the objects on the path, built
        here (not through _pdb_fit_value, which would turn each line into text)."""
        path, id_key, count_key = lines
        spec = params.get(path[0]) or {}
        for key in path[1:]:
            spec = (spec.get("properties") or {}).get(key) or {}
        line = spec.get("items") or {}
        fields = line.get("properties") or {}
        rows = []
        for lid, count in wanted:
            row = {id_key: _pdb_id_shaped(fields.get(id_key) or {}, lid)}
            if count_key:
                row[count_key] = _pdb_clamp(fields.get(count_key) or {}, count if count is not None else 1)
            for k in line.get("required") or []:
                if k not in row and _pdb_one_option(fields.get(k)):
                    row[k] = fields[k]["enum"][0]
            rows.append(row)
        most = spec.get("maxItems")
        value: Any = rows[:int(most)] if isinstance(most, int) else rows
        for key in reversed(path[1:]):
            value = {key: value}
        return value


# ============================================================================ services the mission is built from
_pdb_ENDGAME = ("checkout", "decline", "basket_add", "basket_remove")   # still sent while the world is down
_PB_LOST = {"error": "no_response"}       # returned for a step that never got a reply (readers leave it alone)
_pdb_CLOSERS = ("checkout", "decline")
_pdb_CART_DUTIES = ("basket_add", "basket_remove", "basket_view")
_pdb_TALKERS = ("checkout", "decline", "message")
_pdb_TOOLS_HEAD = "Tool catalogue of this shop, given as a JSON array:\n"
_pdb_ENUM_HEAD = "Candidate tools, as a JSON array:\n"
_pdb_REQUEST_HEAD = "Exact text the shopper sent (do not paraphrase it):\n\n"


def _pdb_deliver(envelope, tname, payload, clock, finishing):
    """Send one envelope, resending the identical one while no reply arrives (the world replays a step it already
    took). Waits double from 1 s up to 8 s; a finishing step may wait until the last seconds, others stop at half
    the wrap-up margin. 409/422 are final; 402/403 get one resend only. Returns (reply or None, last HTTP code)."""
    hook = LOCAL_ENV_CALL                 # a local test world when one is plugged in
    margin = FINISH_LEFT if finishing else WRAP_LEFT / 2
    tries = ENV_SENDS_MAX if finishing else ENV_RETRIES + 2
    delay, n, code = 1.0, 0, None
    while n < tries:
        got, code = (hook(tname, payload), None) if hook is not None else _pa_env_post(envelope, clock)
        if isinstance(got, dict):
            return got, code
        if code in (409, 422) or (n > 0 and code in (402, 403)):
            break
        if not clock.nap(delay, margin):
            break
        delay, n = min(8.0, delay * 2), n + 1
    return None, code


class _pdb_RoleDesk:
    """Model side of role mapping: the first full answer, at most two follow-up asks for gaps, and tracking of
    finishing calls that ended nothing."""

    def __init__(self, seer, tm):
        self.seer, self.tm = seer, tm
        self.asks = 0
        self.misses: dict[str, str] = {}

    def _room(self):
        return self.seer.clock.model_room()

    def first(self):
        self.tm.adopt(self.seer.ask(TOOL_ROLES_GUIDE, _pdb_TOOLS_HEAD + self.tm.schema_text(), TOOL_ROLES_SPEC,
                                    "deep", 1800, (), True))
        self.again()

    def again(self):
        """Ask once more (on another model) for each open or refused role, with the reason; then, if an ending is
        still open, ask narrowly over tools with an open required enum. Code never guesses an enum value."""
        missing = self.tm.gaps()
        if not missing or self.asks >= 2 or not self._room():
            return
        self.asks += 1
        self._remap(missing)
        enums = self.tm.open_enums()
        if enums and not (self.tm.has("checkout") and self.tm.has("decline")) and self._room():
            self._endings(enums)

    def _remap(self, missing):
        why = self.tm.rejected
        listed = "\n".join("- %s: %s" % (d, why.get(d, "no tool proposed")) for d in missing)
        last = self.seer.last_model
        text = _pdb_TOOLS_HEAD + self.tm.schema_text() + ROLES_AGAIN_NOTE.format(gaps=listed)
        self.tm.adopt(self.seer.ask(TOOL_ROLES_GUIDE, text, TOOL_ROLES_SPEC, "deep", 1800, (last,) if last else ()),
                      open_only=True)

    def _endings(self, enums):
        said = self.seer.ask(FINISH_ROLES_GUIDE, _pdb_ENUM_HEAD + json.dumps(self.tm._pdb_cards(enums)),
                             FINISH_ROLES_SPEC, "deep", 600)
        offers = {}
        for key, duty in (("order", "checkout"), ("close", "decline")):
            part = (said or {}).get(key)
            if isinstance(part, dict) and part.get("tool") and part.get("param"):
                offers[duty] = dict(tool=part["tool"], text_arg=part.get("text_param") or "",
                                    fixed=[dict(param=part["param"], value=part.get("value"))])
        if len(offers) == 2:
            a, b = offers["checkout"], offers["decline"]
            if a["tool"] == b["tool"] and norm(a["fixed"][0]["value"]) == norm(b["fixed"][0]["value"]):
                return                    # one value cannot do both endings
        self.tm.adopt(offers, open_only=True)

    def missed_end(self, duty, ans):
        """A finishing call neither ended the episode nor was refused; two equal misses in a row drop the role."""
        if not self.tm.has(duty):
            self.again()
            return
        if ans is _PB_LOST:
            return                        # no reply at all says nothing about the mapping
        if isinstance(ans, dict):
            trouble = error_of(ans)
        else:
            trouble = "no call made" if ans is None else ""
        prior = self.misses.get(duty)
        self.misses[duty] = trouble
        if prior is not None and prior == trouble:
            self.misses.pop(duty)
            self.tm.drop(duty, "answered %s twice" % trouble[:60] if trouble else "ended nothing when called")
            self.again()


# plain request fields: attribute -> (reply key, reader)
_pdb_PLAIN = dict(
    item=("item", lambda v: str(v or "")),
    priority=("priority", lambda v: v if v in ("cheapest", "best", "none") else "none"),
    best_measure=("best_measure", lambda v: str(v or "")),
    best_high=("best_direction", lambda v: v != "low"),
    quantity=("quantity", lambda v: max(1, int(number(v) or 1))),
    stock_rule=("in_stock", lambda v: v is not False),
)


class _pdb_Intake:
    """Reads the request once through the model and writes wants, budgets and searches onto the mission."""

    def __init__(self, m):
        self.m = m

    def run(self):
        m = self.m
        text = _pdb_REQUEST_HEAD + m.request
        got = m.seer.ask(READ_REQUEST_GUIDE, text, READ_REQUEST_SPEC, "deep", 1800, (), True)
        got = got or m.seer.ask(READ_REQUEST_GUIDE, text, READ_REQUEST_SPEC, "quick", 1800)
        if got:
            self.take(got)
        if got and m.searches:
            self.lead_with_values()
        else:
            m.searches = self.fallback()
        self.values_first()
        held = min(m.kept_messages, max(0, int(m.messages_left)))
        m.kept_messages, m.messages_left = held, m.messages_left - held

    def take(self, got):
        m = self.m
        self._wants(got)
        talk, moves = number(got.get("message_budget")), number(got.get("step_budget"))
        # figures the request gives for steps, messages and counts are no price, whatever the model took
        counts = [number(got.get("quantity"))] + [number(p.get("quantity")) for p in got.get("products") or []
                                                  if isinstance(p, dict)]
        m.max_price = as_written(got.get("max_price"), m.request, avoid=(m.max_steps, talk, moves, *counts))
        for attr, (key, read) in _pdb_PLAIN.items():
            setattr(m, attr, read(got.get(key)))
        if talk is not None:
            m.messages_left = int(talk)
        if moves is not None:
            m.max_steps = min(m.max_steps, int(moves))
            m._pc_pace()
        m.reasons = [str(r) for r in got.get("reasons") or [] if r]
        if (number(got.get("question_cost")) or 0) > 0:
            m.messages_left = min(m.messages_left, 1)     # each question costs: only the best one is worth it
        # messages the stated procedure needs are kept apart (each must quote the request itself)
        whole = norm(m.request)
        quoted = [q for q in map(str, got.get("message_steps") or []) if q.strip() and norm(q) in whole]
        m.kept_messages = len(quoted)
        m.searches = [s for s in map(str, got.get("searches") or []) if s.strip()]
        m.pointers = [p for p in got.get("pointers") or [] if isinstance(p, dict) and p.get("attribute")
                      and str(p.get("shop") or "").strip()]
        said = set(stems(m.request))
        sorts = [p for p in got.get("products") or [] if isinstance(p, dict) and str(p.get("item") or "").strip()]
        named = all(set(stems(p["item"])) <= said for p in sorts)    # every sort is one the request itself names
        if named and len(sorts) >= 2:
            sorts = self._one_name(sorts)
            if len(sorts) == 1:
                m.item = sorts[0]["item"]         # one sort named for two kinds: the whole name is the item
        if named and len({norm(p["item"]) for p in sorts}) == len(sorts) >= 2 and not self._one_section(sorts):
            m.products = sorts                    # several sorts of product in one order

    def _one_section(self, sorts) -> bool:
        """Do two of the sorts share one section of the shop? A section named for two kinds (its name holds both)
        is one sort of product asked for by that name, not two."""
        m = self.m
        sections = []
        for tname in m.tools.tools:
            for spec in m.tools.props(tname).values():
                opts = spec.get("enum") if isinstance(spec, dict) else None
                if opts and len(opts) >= 5:
                    sections += [set(stems(str(o))) for o in opts]
        items = [set(stems(p["item"])) for p in sorts]
        return any(sum(1 for it in items if it and it <= sec) >= 2 for sec in sections)

    def _one_name(self, sorts) -> list:
        """Sorts the request writes as one name, side by side with one short joining mark between them, asked for
        in the same number and with the same wants, are one sort under that name (a section named for two kinds)."""
        text = norm(self.m.request)
        wants = lambda p: sorted(json.dumps(x, sort_keys=True, default=str) for x in p.get("requirements") or [])
        kept = [dict(sorts[0])] if sorts else []
        for two in sorts[1:]:
            one = kept[-1]
            joined = re.search(r"\b%s\s+\S{1,3}\s+%s\b" % (re.escape(norm(one["item"])), re.escape(norm(two["item"]))),
                               text)
            if (joined and number(one.get("quantity")) == number(two.get("quantity"))
                    and (wants(one) == wants(two) or not wants(one) or not wants(two))):
                one["item"] = joined.group(0)
                one["requirements"] = (one.get("requirements") or []) or (two.get("requirements") or [])
                one["searches"] = list(one.get("searches") or []) + list(two.get("searches") or [])
                continue
            kept.append(dict(two))
        return kept

    def _wants(self, got):
        for key, against in (("requirements", False), ("exclusions", True)):
            for pair in got.get(key) or []:
                if isinstance(pair, dict) and pair.get("attribute") and pair.get("value"):
                    self._want(pair, against)

    def _want(self, pair, against):
        want = dict(attr=str(pair["attribute"]), value=str(pair["value"]))
        want["from"] = "brief"
        bound = None if against else _pc_bound(pair)
        if against:
            want["not"] = True
        elif bound:
            want["rel"] = bound
        else:
            label = norm(label_words(want["attr"]))
            plain = [n for n in self.m.musts if not n.get("not") and not n.get("rel")]
            if any(norm(label_words(n["attr"])) == label for n in plain):
                return                    # a second plain value for a property already wanted
        self.m.musts.append(want)

    def fallback(self):
        """Value plus item for each want, then the item; with no item, the request's first three words."""
        m = self.m
        noun = m.item or ""
        if not noun:
            return [" ".join(words(m.request)[:3])]
        return [f"{n['value']} {noun}".strip() for n in m.musts] + [noun]

    def value_queries(self) -> list:
        """Searches built from the shopper's own words: the first two wanted values side by side, then each value
        beside the item, every value spelled as the request (or a line) writes it."""
        m = self.m
        texts = [m.request] + list(m.said)
        noun = _pdd_as_written(m.item, texts) or m.item
        item = set(stems(noun))
        vals = []
        for need in m.musts:
            if need.get("not"):
                continue
            v = _pdd_as_written(need["value"], texts)
            if v and len(words(v)) <= 4 and not set(stems(v)) <= item and norm(v) not in map(norm, vals):
                vals.append(v)
        room = m.text_room()
        out = [" ".join(vals[:2])] if len(vals) > 1 else []
        return out + [_pdd_with_noun(v, noun, room) for v in vals]

    def values_first(self) -> None:
        """The value searches go ahead of the model's own, as many as the first round's share of the free steps
        leaves room for beside one of the model's; the rest follow the model's searches."""
        m = self.m
        made = [q for q in dict.fromkeys(self.value_queries()) if q]
        room = max(0, int(m.free_steps() * SEARCH_SHARE) - 1)
        taken = set(map(norm, made))
        own = [q for q in m.searches if norm(q) not in taken]
        m.searches = made[:room] + own + made[room:]

    def lead_with_values(self):
        """Add "<value> <item>" just after the first search for each plain want of 1-3 stems (one of 3+ letters),
        unless a search already opens with the value then an item word. Stores often weigh only the first words,
        and a search naming the property misses other spellings of the value."""
        m = self.m
        noun = m.item or ""
        if not noun:
            return
        item = set(stems(noun))
        added = []
        for need in m.musts:
            lead = stems(need["value"])
            if need.get("not") or need.get("rel") or not 0 < len(lead) <= 3 or max(map(len, lead)) < 3:
                continue
            k = len(lead)
            if not any(s[:k] == lead and item & set(s[k:k + 1]) for s in map(stems, m.searches + added)):
                added.append(f"{need['value']} {noun}")
        m.searches[1:1] = added


_pdb_EMPTY = {list: "musts searches reasons told_queries order_seen ask_log asked topics pending_notes pointers "
                    "products",
              dict: "rows categories screen verdicts basket next_pages query_totals query_rows faults unsent "
                    "sheet_stock",
              set: "paged_out opened struck done_queries dropped"}
_pdb_SCALARS = dict(
    item="", max_price=None, priority="none", best_measure="", best_high=True, quantity=1, stock_rule=True,
    messages_left=UNSTATED_MESSAGES, page_rows=0, details_checked=False, many_cap=0, sweep_cap=None, sweep_shelf="",
    figure_unit="", kept_messages=0, no_messages=False, dead_ends=0, role_asks=0, kept=None, whole_noun=False,
    hand_on=False, offer_probed=False, rescued_cart=False, cap_rose=False, zero_stays=False,
)


def _pdb_request_text(view):
    said = str(view.get("query") or "")
    opening = view.get("opening")
    return said if opening is None else said + "\n\n" + json.dumps(opening)


class Mission:
    """One episode: world state, with role mapping and request reading done by small helper services."""

    def __init__(self, problem):
        world = problem["environment"]
        bind, view = world.get("binding"), world.get("policy_view")
        self.binding, self.policy = bind or {}, view or {}
        view = self.policy
        self.request = _pdb_request_text(view)
        self.tools = ToolMap(view.get("tools") or [])
        self.clock = _PaClock()
        self.seer = Seer(self.clock)
        self._pdb_roles = _pdb_RoleDesk(self.seer, self.tools)
        self.dialogue = []                # answered steps, as returned at the end
        self.step = 0
        self.finished = bool(view.get("finished"))
        self.world_down = False           # unanswered steps in a row: only finishing steps go out
        self.cart_fills = 0               # cart calls a model wrote because code could not shape one
        self.late_reads = 0               # unprompted lines read with the request and spend caps lifted
        self.vital_answers = 0            # answers read past those caps (set only by an order of several products)
        self.heard: set = set()           # lines read already, so a repeated line counts once
        self.said: list = []              # the shopper's lines as written, for values quoted in their own words
        self.reread: set = set()          # fresh reads of the pick after a refusal or a line
        limit = number(view.get("max_steps"))
        self.calls_per_turn = int(number(view.get("max_calls_per_turn")) or 1)   # this code sends one call per turn
        self.max_steps = int(limit or FALLBACK_STEPS)
        self._pc_pace()
        for make, names in _pdb_EMPTY.items():
            for name in names.split():
                setattr(self, name, make())
        for name in _pdb_SCALARS:
            setattr(self, name, _pdb_SCALARS[name])

    # ---------------------------------------------------------------- steps to the world
    def _pb_blocked(self, duty, tname, payload):
        """True when a non-ending role would make an ending's call (its tool with all its pinned values). Cart
        roles may share it when check-out takes a listing id."""
        if duty in _pdb_CLOSERS:
            return False
        jobs = self.tools.roles
        hits = [jobs[e] for e in _pdb_CLOSERS if jobs.get(e) and jobs[e]["tool"] == tname]
        if not any(all(payload.get(k) == v for k, v in j["fixed"].items()) for j in hits):
            return False
        cart_ok = duty.startswith("basket") and bool((jobs.get("checkout") or {}).get("id_arg"))
        return not cart_ok

    def act(self, duty, **slots):
        call = self.tools.call_args(duty, **slots)
        if call is None or self.finished or self.steps_remaining() <= 0:
            return None
        tname, payload = call
        sheet = dict(self.tools._pdb_sheet(tname), type="object")
        if _pdb_shape_fault(sheet, payload) or self._pb_blocked(duty, tname, payload):
            return self._pdb_hold_back(duty)
        final = duty in _pdb_ENDGAME
        if self.world_down and not final:
            return None
        envelope = dict(self.binding, step=self.step + 1, action={"name": tname, "args": payload})
        got, code = _pdb_deliver(envelope, tname, payload, self.clock, final)
        if got is not None:
            return self._pdb_answered(duty, tname, payload, got)
        if code == 422:
            return self._pdb_hold_back(duty)  # the call itself was refused: a mapping fault, not a lost step
        return self._pb_lost_step()

    def _pdb_hold_back(self, duty):
        """A call kept back (it would end the episode or lacks a required value). On a third try, for a role that
        is no ending, the role counts as wrongly mapped and is mapped again."""
        times = self.unsent.get(duty, 0) + 1
        self.unsent[duty] = times
        if times == 3 and duty not in _pdb_CLOSERS:
            self.tools.drop(duty, "stayed unsendable three times")
            self.repair_roles()
        return None

    def _pdb_answered(self, duty, tname, payload, got):
        self.dead_ends, self.world_down = 0, False
        theirs = got.get("step")            # follow the world's own counter when it runs ahead
        ours = self.step + 1
        self.step = max(ours, theirs) if isinstance(theirs, int) else ours
        self.finished = bool(got.get("finished") or got.get("done") is True)
        spoken = got.get("user_message")          # a shopper line some worlds send beside the observation
        if isinstance(spoken, str) and spoken.strip() and spoken not in json.dumps(got.get("observation"), default=str):
            self.pending_notes.append(spoken)
        self.dialogue.append(dict(role="assistant", tool_call=dict(name=tname, arguments=payload),
                                  environment_result=got))
        seen = got.get("observation")
        self._pb_note_fault(duty, seen)
        if duty in _pdb_CART_DUTIES:
            self._pdb_foreign_lines(seen, payload)
        return seen

    def _pdb_foreign_lines(self, seen, payload):
        """A cart reply that lists a line this run never put there (nor took out): it joins the basket record, so
        the clearing before an order takes it out. Whole-basket writers replace such lines anyway."""
        if (self.tools.roles.get("basket_add") or {}).get("lines") or not isinstance(seen, dict) or error_of(seen):
            return
        sent = json.dumps(payload, default=str)
        for rec in find_records(seen):
            lid = rec["id"]
            if lid not in self.basket and lid not in self.dropped and str(lid) not in sent:
                self.basket[lid] = 1

    def _pb_note_fault(self, duty, seen):
        """Three equal errors from one ordinary role: unmapped and asked for again (endings and messages are
        handled elsewhere)."""
        trouble = error_of(seen) if duty not in _pdb_TALKERS and isinstance(seen, dict) else ""
        if not trouble:
            return
        mark = duty, trouble[:80]
        self.faults[mark] = 1 + self.faults.get(mark, 0)
        if self.faults[mark] > 2 and self.tools.has(duty):
            self.tools.drop(duty, "returned %s three times" % trouble[:60])
            self.repair_roles()

    def _pb_lost_step(self):
        self.dead_ends += 1
        self.world_down = self.world_down or self.dead_ends >= ENV_RETRIES - 1
        return _PB_LOST

    # ---------------------------------------------------------------- roles
    def map_roles(self) -> None:
        self._pdb_roles.first()

    def repair_roles(self) -> None:
        self._pdb_roles.again()

    def _pb_end_missed(self, role: str, reply: Any) -> None:
        self._pdb_roles.missed_end(role, reply)

    # ---------------------------------------------------------------- fresh read of the pick
    def recheck(self, pid: str) -> bool:
        """Read the held pick again (after a refused finish or a line) so its price and stock are current; True
        when it is still buyable."""
        stale_ok = self.world_down or self.steps_remaining() <= 2 or len(self.reread) >= 3
        if stale_ok or self.clock.left() < WRAP_LEFT / 2:
            return self.buyable(pid)         # no fresh read: judge on what is known
        self.reread.add((pid, self.step))
        tm = self.tools
        roomy = self.steps_remaining() >= _PDD_SWAP_STEPS
        priced = roomy and (self._pdd_offer_ready() or self._pdd_offer_probe()) and self._pdd_offer_read(pid)
        priced = priced or (tm.has("open_listing") and self._pdb_listing_priced(pid))
        cart_can_tell = pid in self.basket and tm.has("basket_view")
        if not priced and cart_can_tell and self.steps_remaining() > 2:
            self._pdb_basket_price(pid)
        if not self.buyable(pid):
            return False
        self.kept = pid                      # still fits after the fresh read
        return True

    def _pdb_listing_priced(self, pid):
        page = self.act("open_listing", id_arg=pid)
        if not isinstance(page, dict) or error_of(page, need_data=True):
            return False
        self.absorb(page)
        self._pc_file_listing(pid, page)
        return any(isinstance(r.get("price"), float) for r in _pc_listing_records(pid, page))

    def _pdd_offer_ready(self):
        """An offer tool is mapped and it is not the tool the listing sheet is read with."""
        offer, sheet = self.tools.roles.get("open_offer"), self.tools.roles.get("open_listing")
        return bool(offer) and (not sheet or offer["tool"] != sheet["tool"])

    def _pdd_offer_probe(self):
        """No offer tool was named, but the listing sheet showed no stock flag: once per episode, the one other
        single-id tool no role holds (none when there are several) is taken as the offer tool on trial (one step on the pick); it stays only when
        its reply carries a price or a stock flag (see _pdd_offer_read)."""
        home = self.tools.roles.get("open_listing") or {}
        if self.offer_probed or self.sheet_stock.get(home.get("tool")) is not False:
            return False
        self.offer_probed = True
        spare = [(nm, arg) for nm, arg in self._pdc_Opener(self)._spare_one_id_tools(home)
                 if not self.tools.props(nm).get(arg, {}).get("enum")]
        if len(spare) != 1:
            return False       # a call on trial must be the only one it could be: an unknown tool may change the cart
        nm, arg = spare[0]
        self.tools.roles["open_offer"] = {"tool": nm, "fixed": {}, "id_arg": arg, "trial": True}
        return True

    def _pdd_offer_read(self, pid):
        """Read one listing's offer: its price and stock go on the record, with the seller's own remarks beside them
        (the judge reads them); a listing the offer shows out of stock is struck. True when the reply gave a price or
        a stock flag."""
        page = self.act("open_offer", id_arg=pid)
        job = self.tools.roles.get("open_offer") or {}
        recs = [] if not isinstance(page, dict) or error_of(page, need_data=True) else _pc_listing_records(pid, page)
        shows = any(isinstance(rec.get("price"), float) or isinstance(rec.get("in_stock"), bool) for rec in recs)
        if job.pop("trial", False) and not shows:
            self.tools.roles.pop("open_offer", None)     # the trial tool shows no offer: it is not kept
            return False
        if not recs:
            return False
        self.absorb(page)
        remarks = _pdd_offer_remarks(pid, page)
        for rec in recs:
            self.merge(dict(rec, notes=remarks) if remarks else rec)
        if self.stock_rule and any(rec.get("in_stock") is False for rec in recs):
            self.struck.add(pid)
        return shows

    def _pdb_basket_price(self, pid):
        """No price on the listing: the cart shows what the order would cost."""
        page = self.act("basket_view")
        self.absorb(page)
        for row in find_records(page):
            if row["id"] == pid:
                self.merge({k: row[k] for k in row if k in ("id", "price", "in_stock")})

    # ---------------------------------------------------------------- pacing
    def steps_remaining(self):
        return self.max_steps - self.step

    def _pc_pace(self):
        """End reserve, listings to open first and listings before asking, as shares of the step limit."""
        limit = self.max_steps
        self.reserve = max(5, round(END_SHARE * limit))
        self.open_min = max(4, round(OPEN_SHARE * limit))
        self.ask_after = max(3, round(ASK_SHARE * limit))

    def room(self, cost: int) -> bool:
        return cost <= self.steps_remaining() - self.reserve

    def text_room(self) -> Any:
        """The most characters the word search takes, when its schema says."""
        plan = self.tools.roles.get("keyword_search") or {}
        spec = self.tools.props(plan.get("tool", "")).get(plan.get("text_arg", "")) or {}
        return spec.get("maxLength") if isinstance(spec.get("maxLength"), int) else None

    def free_steps(self):
        """Steps left once the end game and the remaining questions are set aside."""
        talk = 0 if self.no_messages else self.messages_left + self.kept_messages
        return self.steps_remaining() - self.reserve - talk

    def _pb_roomy(self):
        return self.open_min < self.free_steps()

    # ---------------------------------------------------------------- the request
    def read_request(self):
        _pdb_Intake(self).run()

    # ---------------------------------------------------------------- searching
    def _pb_first_time(self, tag):
        if tag in self.done_queries:
            return False
        self.done_queries.add(tag)
        return True

    def _pdb_run_listing(self, duty, tag, page, **slots):
        """One search-like step unless already made, or a later page with no page parameter (page one again)."""
        if not self._pb_first_time(tag):
            return None
        if page is not None and not (self.tools.roles.get(duty) or {}).get("page_arg"):
            return None
        return self._pb_ingest(self.act(duty, **slots, page_arg=page))

    def search(self, query, page=None):
        phrase = norm(query)
        got = self._pdb_run_listing("keyword_search", f"{phrase}|{page}", page, text_arg=query)
        if got is None:
            return ([], None, None)
        ids, token, total = got
        self.next_pages[phrase] = (token, 1 + self.next_pages.get(phrase, (None, 0))[1])
        self.query_rows.setdefault(phrase, []).extend(ids)
        if total is not None:
            self.query_totals[phrase] = total
        return got

    def browse(self, category, cap, page=None):
        # a lower cap only narrows the shelf; the shopper's own limit still decides what may be bought
        got = None
        if self.tools.has("filter_search"):
            got = self._pdb_run_listing("filter_search", _pdd_filter_tag(category, cap, page), page,
                                        category_arg=category, price_arg=cap)
        return ([], None, None) if got is None else got

    def _pb_ingest(self, seen):
        """File a search reply's records; gives their ids, the next-page token and the reported total."""
        if seen is None:
            return ([], None, None)
        self.absorb(seen)
        found = find_records(seen)
        for row in found:
            self.merge(row)
            shelf = row.get("category")
            if shelf:
                self.categories[shelf] = 1 + self.categories.get(shelf, 0)
        ids = [row["id"] for row in found]
        token, total = paging_of(seen)
        if token is None and isinstance(seen, dict):
            token = _pdd_loose_token(seen, ids)
        if token is not None:
            self.page_rows = max(len(ids), self.page_rows)        # only a full page has a successor
        return ids, token, total

    def page_numeric(self, duty="keyword_search"):
        """Is the page parameter of this role a number in the tool's schema?"""
        job = self.tools.roles.get(duty) or {}
        kind = self.tools.props(job.get("tool", "")).get(job.get("page_arg", ""), {}).get("type")
        return any(k in ("integer", "number") for k in (kind if isinstance(kind, list) else [kind]))

    def _pb_page_ceiling(self):
        """Pages one search may read: the page parameter's schema maximum, else the default. Counted from 0 it is
        a first-row offset, so its maximum is a row number and pages lie page_rows apart."""
        job = self.tools.roles.get("keyword_search") or {}
        spec = self.tools.props(job.get("tool", "")).get(job.get("page_arg", ""), {})
        top = number(spec.get("maximum"))
        if not top:
            return PAGES_PER_QUERY
        offset = self.page_rows > 1 and number(spec.get("minimum")) == 0
        return int(top // self.page_rows) + 1 if offset else int(top)

    # ================================================================ public face of the reading services
    # Each job below lives in a small service object built on the mission; the methods here are the names the
    # rest of the agent calls.
    def _pb_ranked_reading(self, query, total):
        return self._pdc_Pager(self).reads_all(query, total)

    def pages_for(self, total, query=""):
        return self._pdc_Pager(self).page_count(total, query)

    def look_round(self):
        self._pdc_Pager(self).first_round()

    def cheap_sweep(self, shelf, start=None, floor=None):
        self._pdc_Pager(self).sweep(shelf, start, floor)

    def screen_rows(self, restate=False, only=None):
        self._pdc_Triage(self).screen(restate, only)

    def _pc_named_shelves(self) -> set:
        return self._pdc_Triage(self).named_sections()

    def open_order(self):
        return self._pdc_Triage(self).ranked()

    def census_rows(self, size: int) -> list[str]:
        return self._pdc_Triage(self).census(size)

    def open(self, pid: str) -> None:
        self._pdc_Opener(self).one(pid)

    def _pc_file_listing(self, pid: str, seen: Any) -> None:
        self._pdc_Opener(self).file(pid, seen)

    def open_batch(self, pids: list[str]) -> None:
        self._pdc_Opener(self).several(pids)

    def batch_size(self) -> int:
        if self.tools.has("open_many") and self.many_cap != 1:
            return self.many_cap or 6
        return 3

    def two_sided(self, topic: str) -> bool:
        return self._pdc_Asker(self).ambiguous(topic)

    def ask_text(self, topic: str, sharper: bool = False) -> str:
        return self._pdc_Asker(self).question(topic, sharper)

    def choose_topics(self) -> None:
        self._pdc_Asker(self).plan()

    def _determined_by(self, topic: str, attr: str) -> bool:
        return self._pdc_Asker(self).follows(topic, attr)

    def hear(self, lines: list[str], asked: str = "") -> dict:
        return self._pdc_Listener(self).take(lines, asked)

    def _set_need(self, attr: str, value: str, source: str, replace: bool = True, rel: str = "") -> None:
        self._pdc_Listener(self).record(attr, value, source, replace, rel)

    def judge(self, pids: list[str]) -> None:
        self._pdc_Checker(self).run(pids)

    def _grounded(self, pid: str, verdict: dict) -> dict:
        return self._pdc_Checker(self).verify(pid, verdict)

    def _code_judge(self, pid: str) -> dict:
        return self._pdc_Checker(self).by_rules(pid)

    # ---------------------------------------------------------------- record keeping shared by the services
    def merge(self, rec):
        key = rec["id"]
        if key not in self.rows:          # first sight: remember the order rows were met in
            self.order_seen.append(key)
            self.rows[key] = dict(rec)
            return
        held = self.rows[key]
        for field in rec:
            incoming = rec[field]
            if field != "traits":
                if incoming is not None:
                    held[field] = incoming
                continue
            held.setdefault("traits", {}).update(incoming)

    def buyable(self, pid):
        if pid in self.struck:
            return False
        rec = self.rows.get(pid) or {}
        if rec.get("in_stock") is False and self.stock_rule:
            return False
        price = rec.get("price")
        if isinstance(price, float) and self.max_price is not None:
            return price * self.quantity <= self.max_price + 1e-9
        return True

    def _req_text(self) -> str:
        parts = []
        for need in self.musts:
            rel = need.get("rel")
            lead = rel + " " if rel and rel not in norm(need["value"]) else ""
            parts.append("%s: %s%s%s" % (need["attr"], "not " if need.get("not") else "", lead, need["value"]))
        return "; ".join(parts) or "(none)"

    @staticmethod
    def _pdc_numbered(reply, field, size):
        """(position, row) for each reply row whose number points inside a batch of the given size."""
        found = []
        for entry in (reply or {}).get(field) or []:
            at = entry.get("num") if isinstance(entry, dict) else None
            if isinstance(at, int) and 0 <= at < size:
                found.append((at, entry))
        return found

    # ---------------------------------------------------------------- label comparison
    @staticmethod
    def _pdc_label_set(name: str) -> set:
        return set(words(label_words(name)))

    @staticmethod
    def _pdc_covered(small: set, big: set) -> bool:
        """Each word of small has a match in big; a plural and its singular match."""
        variants = lambda w: {w, w + "s", _pa_trim(w)}
        return all(any(variants(a) & variants(b) for b in big) for a in small)

    @staticmethod
    def _overlaps(a: str, b: str) -> bool:
        """True when the two names share a word that carries meaning."""
        filler = {"type", "kind", "of", "the", "and", "or"}
        mine = Mission._pdc_label_set(a) - filler
        return bool(mine & (Mission._pdc_label_set(b) - filler))

    @staticmethod
    def _pc_one_property(a: str, b: str) -> bool:
        """True when one name's words all sit inside the other's (plural forms allowed); a mere shared word is
        not enough."""
        x, y = Mission._pdc_label_set(a), Mission._pdc_label_set(b)
        if not (x and y):
            return False
        return Mission._pdc_covered(x, y) or Mission._pdc_covered(y, x)

    @staticmethod
    def _same_attr(a: str, b: str) -> bool:
        x, y = Mission._pdc_label_set(a), Mission._pdc_label_set(b)
        return bool(x and y) and (x <= y or y <= x)

    # ================================================================ service: how far to read searches
    class _pdc_Pager:
        def __init__(self, mission):
            self.m = mission

        def _shelf_filter(self) -> bool:
            spec = self.m.tools.roles.get("filter_search") or {}
            return bool(spec.get("price_arg") and spec.get("category_arg"))

        def reads_all(self, qtext, total) -> bool:
            """A ranked choice without a capped shelf listing reads every reachable page when the result fits,
            when only price matters and later pages are not dearer by design, or when the search names a value
            the shopper asked for."""
            m = self.m
            if self._shelf_filter() or m.priority not in ("cheapest", "best"):
                return False
            if total is not None and m.page_rows and total <= m._pb_page_ceiling() * m.page_rows:
                return True
            plain = [rq for rq in m.musts if not rq.get("not")]
            if not plain and m.priority == "cheapest":
                return not self._climbing(qtext)
            asked_for = {w for rq in plain for w in stems(rq["value"])}
            return not asked_for.isdisjoint(stems(qtext))

        def _climbing(self, qtext) -> bool:
            m = self.m
            seq = [m.rows[it].get("price") for it in m.query_rows.get(norm(qtext), []) if it in m.rows]
            seq = [x for x in seq if isinstance(x, float)]
            if len(seq) < 2:
                return False
            return not any(later < earlier for earlier, later in zip(seq, seq[1:]))

        def page_count(self, total, qtext=""):
            m = self.m
            limit = m._pb_page_ceiling()
            if self.reads_all(qtext, total) or not (total and m.page_rows):
                return limit
            needed = -(-int(total) // m.page_rows)
            return 1 if needed > max(limit, m.free_steps() // 3) else needed

        def _single_title(self) -> bool:
            names = {norm(rec.get("title") or "") for rec in self.m.rows.values()}
            names.discard("")
            return len(names) == 1

        def _keep_paging(self, pages, total, qtext) -> bool:
            if pages < self.page_count(total, qtext):
                return True
            # rows that are all one product tell nothing apart yet: read on to the ceiling
            return pages < self.m._pb_page_ceiling() and self._single_title()

        def first_round(self):
            m = self.m
            allowance = max(2, int(m.free_steps() * SEARCH_SHARE))
            used = 0
            for qtext in m.searches:
                if used >= allowance or not m._pb_roomy():
                    break
                before = len(m.rows)
                ids, token, total = m.search(qtext)
                used += 1
                if ids and before == len(m.rows):
                    continue
                pages = 1
                while (token is not None and used < allowance + 2 and m._pb_roomy()
                       and self._keep_paging(pages, total, qtext)):
                    _, token, total = m.search(qtext, _pb_page_ref(token, pages, m.page_numeric()))
                    pages += 1
                    used += 1
            if m.priority == "cheapest" and m.categories:
                self.sweep(max(m.categories, key=m.categories.get))

        def sweep(self, aisle, start=None, floor=None):
            """Narrow the shelf listing's upper price until all its matches fit in the readable pages, then read them.
            The cap moves between a crowded value and an empty (or already read) one."""
            m = self.m
            if not self._shelf_filter():
                return
            m.sweep_shelf = aisle
            cp = start if start is not None else m.max_price
            crowded, empty = None, floor
            tries = 5
            while tries:
                tries -= 1
                if cp is None or not m._pb_roomy():
                    return
                shown, token, count = m.browse(aisle, cp)
                if count is None:
                    if not shown:
                        return
                    count = len(shown)
                space = m._pb_page_ceiling() * max(m.page_rows, len(shown), 1)
                if count == 0:
                    if crowded is None:
                        return
                    empty, cp = cp, round((cp + crowded) / 2, 2)
                    continue
                if 0 < count <= space:
                    self._finish_shelf(aisle, cp, token)
                    m.sweep_cap = cp
                    return
                crowded = cp
                cp = self._lower_cap(aisle, shown, count, space, cp, empty)
                m.sweep_cap = cp

        def _finish_shelf(self, aisle, cp, token):
            m = self.m
            page = 1
            while token is not None and page < m._pb_page_ceiling() and m._pb_roomy():
                token = m.browse(aisle, cp, _pb_page_ref(token, page, m.page_numeric("filter_search")))[1]
                page += 1

        def _lower_cap(self, aisle, shown, count, space, cp, empty):
            """Price under which roughly the readable share of matches falls, judged from the sampled page; kept
            above an empty cap and away from caps already listed."""
            m = self.m
            priced = sorted(x for x in (m.rows[i].get("price") for i in shown) if isinstance(x, float))
            if not priced:
                aim = cp / 2
            else:
                at = int(len(priced) * space / count)
                aim = priced[min(max(at, 0), len(priced) - 1)]
            if empty is not None:
                aim = max(aim, (empty + cp) / 2)
            while aim < cp and _pdd_filter_tag(aisle, aim) in m.done_queries:
                aim = (aim + cp) / 2
            return round(aim if aim < cp else cp * 0.6, 2)

    # ================================================================ service: row screening and open order
    class _pdc_Triage:
        def __init__(self, mission):
            self.m = mission

        def _ask_model(self, chunk_ids, wanted) -> dict:
            m = self.m
            body = "Item: %s\nShopper's stated needs: %s\n\nLabel these rows (row number, then title | category | " \
                   "description):" % (m.item, wanted)
            for n, lid in enumerate(chunk_ids):
                look = m.rows[lid].get
                body += "\n%d) %s | category %s | %s" % (n, look("title", "")[:150], look("category", ""),
                                                         str(look("description", ""))[:160])
            ans = m.seer.ask(SCREEN_GUIDE, body, SCREEN_SPEC, "quick", 2500)
            return {chunk_ids[at]: (str(ln.get("kind")), str(ln.get("stated")))
                    for at, ln in Mission._pdc_numbered(ans, "rows", len(chunk_ids))}

        def _word_marks(self, lid):
            """Fallback marks from plain word presence."""
            m = self.m
            look = m.rows[lid].get
            parts = [str(look(k)) for k in ("title", "category", "description")]
            blob = norm(" ".join(parts + [json.dumps(look("traits") or {})]))
            present = lambda ws: all(w in blob for w in ws)
            brief = [w for rq in m.musts if rq["from"] == "brief" for w in words(rq["value"])]
            kind = "yes" if m.item and present(words(m.item)[-1:]) else "maybe"
            return kind, "yes" if present(brief) else "unknown"

        def screen(self, restate, only=None):
            """Mark every unmarked buyable row (only those in `only` when given). With restate, unopened rows marked
            before get a fresh stated mark (their kind mark stays) so the next openings follow the new answer."""
            m = self.m
            mks = m.screen
            again = {it: mks[it] for it in m.order_seen if restate and it in mks and it not in m.opened}
            todo = [it for it in m.order_seen if (it not in mks or it in again) and m.buyable(it)
                    and (only is None or it in only)]
            for at in range(0, len(todo), REVIEW_ROWS):
                part = todo[at:at + REVIEW_ROWS]
                got = self._ask_model(part, m._req_text())
                for it in part:
                    fresh = got.get(it)
                    if it in again:
                        mks[it] = (again[it][0], fresh[1]) if fresh else again[it]
                    else:
                        mks[it] = fresh or self._word_marks(it)
            section = lambda it: m.rows[it].get("category")
            good = {section(it) for it, mk in mks.items() if mk[0] == "yes"} - {None, ""}
            named = self.named_sections()
            for it, (kind, stated) in list(mks.items()):
                if kind == "no" and section(it) in good:
                    kind = "maybe"        # another row of this section was the right kind
                    mks[it] = (kind, stated)
                if named and section(it) and section(it) not in named:
                    mks[it] = ("no", stated)

        @staticmethod
        def _names_item(aisle: list, item: list) -> bool:
            """The section's words appear in the item phrase in order, from its first word to its last, with at
            most one other word between two of them."""
            pos, prev = 0, -1
            for w in aisle:
                hit = item.index(w, pos) if w in item[pos:] else -1
                gap = 1 if prev < 0 else 2
                if hit < 0 or hit - prev > gap:
                    return False
                pos, prev = hit + 1, hit
            return bool(aisle) and prev == len(item) - 1

        def named_sections(self) -> set:
            item = stems(self.m.item)
            sections = {str(rec.get("category") or "") for rec in self.m.rows.values()} - {""}
            return {s for s in sections if self._names_item(stems(s), item)}

        def row_writes(self, lid) -> bool:
            """The row's own title or description holds every word of each worded value (bounds and exclusions
            aside); False when there is no such value."""
            look = self.m.rows[lid].get
            texts = [_pdd_bits(look("title")), _pdd_bits(look("description"))]
            wanted = [_pdd_bits(rq.get("value")) for rq in getattr(self.m, "musts", ()) or ()
                      if not (rq.get("not") or rq.get("rel"))]
            wanted = [w for w in wanted if w]
            return bool(wanted) and all(any(w <= t for t in texts) for w in wanted)

        def ranked(self):
            """Unopened buyable rows not marked wrong. By price when the cheapest is wanted, the two cheapest rows
            writing every worded value in their own title or description first (a cheaper pass is still sought
            below the pick they set); else right kind and
            shown needs first. Rows listed by a search for a told value go ahead within the same marks."""
            m = self.m
            tl = {it for q in m.told_queries for it in m.query_rows.get(norm(q), [])}
            keyed = []
            for place, it in enumerate(m.order_seen):
                kind, stated = m.screen.get(it, ("maybe", "unknown"))
                if kind == "no" or it in m.opened or not m.buyable(it):
                    continue
                late = it not in tl
                if m.priority == "cheapest":
                    key = (stated == "no", late, m.rows[it].get("price") or 0.0, place)
                else:
                    key = (kind != "yes", stated != "yes", stated == "no", late, place)
                keyed.append((key, it))
            order = [it for _, it in sorted(keyed)]
            if m.priority == "cheapest":
                # two rows at most in the whole run go ahead of the price order: more would spend the looks a cheaper pass needs
                had = self.m.__dict__.setdefault("written_ahead", {}).setdefault(str(getattr(self.m, "item", "")), [])
                had += [it for it in order if self.row_writes(it)][:2 - len(had)]
                ahead = [it for it in had if it in order]
                return ahead + [it for it in order if it not in ahead]
            tops = self._store_tops(set(order))
            return tops + [it for it in order if it not in tops]

        def _store_tops(self, open_now):
            """The store's own first row of the first search that listed rows, and of each told-value search: a
            buyable row the screen did not rule out, one per title, two at most. The screen's labels (read from a
            title alone) may put such a row behind every other one; the store lists it first."""
            m = self.m
            lists = [rows for rows in m.query_rows.values() if rows][:1]
            lists += [m.query_rows.get(norm(q)) or [] for q in m.told_queries]
            tops, titles = [], set()
            for rows in lists:
                head = next((it for it in rows if (m.screen.get(it) or ("maybe",))[0] != "no" and m.buyable(it)), None)
                title = norm((m.rows.get(head) or {}).get("title") or "") if head else ""
                if head in open_now and head not in tops and (not title or title not in titles):
                    tops.append(head)
                    titles.add(title)
                if len(tops) >= 2:
                    break
            return tops

        def census(self, size):
            """Open order with one row per title first, then the remaining rows."""
            order = self.ranked()
            lead, titles = [], set()
            for it in order:
                title = norm(self.m.rows[it].get("title") or "")
                if len(lead) < size and (not title or title not in titles):
                    lead.append(it)
                    titles.add(title)
            tail = [it for it in order if it not in lead]
            return lead + tail[:max(0, size - len(lead))]

    # ================================================================ service: reading listings
    class _pdc_Opener:
        def __init__(self, mission):
            self.m = mission

        def one(self, lid):
            m = self.m
            if lid in m.opened or not m.tools.has("open_listing"):
                return
            m.opened.add(lid)                 # marked first: a step lost on the way is undone below
            seen = m.act("open_listing", id_arg=lid)
            if seen is _PB_LOST:
                m.opened.discard(lid)
            elif seen is not None:
                self.file(lid, seen)

        def file(self, lid, seen):
            m = self.m
            m.absorb(seen)
            recs = _pc_listing_records(lid, seen)
            for rec in recs:
                m.merge(rec)
            home = (m.tools.roles.get("open_listing") or {}).get("tool")
            told = any(isinstance(rec.get("in_stock"), bool) for rec in recs)
            m.sheet_stock[home] = bool(m.sheet_stock.get(home) or told)   # does the sheet tool ever show stock?
            if m.details_checked:
                return
            m.details_checked = True
            if not m._pc_card(lid).get("traits"):
                self._other_tools(lid)

        def several(self, pids):
            m = self.m
            left = [it for it in pids if it not in m.opened]
            width = min(len(left), m.many_cap or len(left)) if m.tools.has("open_many") else 0
            while width > 1 and len(left) > 1 and m.room(1):
                width, left = self._many(left, width)
            for it in left:
                m.open(it)

        def _many(self, left, width):
            """One multi-open step. An empty or failed reply halves the width; the first width that works is
            kept from then on."""
            m = self.m
            ids = left[:width]
            ans = m.act("open_many", ids_arg=ids)
            got = {} if ans is None or error_of(ans, need_data=True) else {r["id"]: r for r in find_records(ans)}
            if not got:
                m.many_cap = width // 2
                return m.many_cap, left
            m.absorb(ans)
            m.opened.update(ids)
            for it in ids:
                if it in got:
                    m.merge(dict(got[it], id=it))
            if not m.details_checked:
                m.details_checked = any(r.get("traits") for r in got.values())
            m.many_cap = m.many_cap or width
            rest = left[width:]
            return min(len(rest), m.many_cap), rest

        def _spare_one_id_tools(self, home):
            tools = self.m.tools
            taken = {plan["tool"] for role, plan in tools.roles.items() if role != "open_listing"}
            found = []
            for nm in tools.tools:
                req = tools.required(nm)
                if len(req) != 1 or nm in taken or nm == home.get("tool"):
                    continue
                if tools.props(nm).get(req[0], {}).get("type") in ("string", "integer", None):
                    found.append((nm, req[0]))
            return found

        def _other_tools(self, lid):
            """The listing tool showed no properties: try each other single-id tool, keep the first that does."""
            roles = self.m.tools.roles
            home = roles.get("open_listing") or {}
            for nm, arg in self._spare_one_id_tools(home):
                roles["open_listing"] = {"tool": nm, "fixed": {}, "id_arg": arg}
                facts = self._props_from(self.m.act("open_listing", id_arg=lid))
                if facts:
                    self.m.merge({"id": lid, "traits": facts})
                    return
                roles["open_listing"] = home

        def _props_from(self, seen):
            if seen is None or error_of(seen, need_data=True):
                return {}
            self.m.absorb(seen)
            lead = (find_records(seen) or [{}])[0]
            if lead.get("traits"):
                return lead["traits"]
            return _traits_of(seen) if isinstance(seen, dict) else {}

    # ================================================================ service: questions to the shopper
    class _pdc_Asker:
        CODE_LABELS = frozenset({"option", "options", "variant", "variants", "model", "sku", "code"})

        def __init__(self, mission):
            self.m = mission

        def _values(self):
            return _pc_label_tally(self.m.opened, self.m.rows)["values"]

        def ambiguous(self, subj):
            """One word, or a label nested in another opened label (either way): such a label may mean two
            properties, so a split reply earns a sharper re-ask."""
            lab = label_words(subj) or subj
            if len(words(lab)) < 2:
                return True
            return any(norm(n) != norm(lab) and self.m._pc_one_property(n, lab) for n in self._values())

        def question(self, subj, sharper):
            lab = label_words(subj) or subj
            txt = ASK_TRAIT.format(lab, self._noun(lab))
            if not sharper:
                return txt
            whole = ASK_TRAIT.format(lab, self.m.item) if self.m.item else txt
            if whole != txt:
                return whole                      # the item named in full, still one property and no "or"
            mine, elsewhere = [], set()
            for nm, vals in self._values().items():
                if norm(nm) == norm(lab):
                    mine.extend(vals)
                else:
                    elsewhere.update(vals)
            examples = [v for v in mine if v not in elsewhere and len(words(v)) <= 3][:1]
            if examples:
                txt = txt[:-1] + " - for example %s?" % examples[0]   # two values read as two questions
            return txt

        def _noun(self, lab):
            """The last word of the item name, spelled as most opened titles spell it, when titles carry it; else
            the item name in full (an earlier word alone may name whom it is for rather than what it is)."""
            m = self.m
            if m.whole_noun and m.item:
                return m.item                     # several products: the question names which one, in full
            titles = [words((m.rows.get(it) or {}).get("title") or "") for it in sorted(m.opened)]
            avoid = set(words(lab))
            pool = [w for w in words(m.item) if w not in avoid]
            if not pool:
                return m.item or "item"
            root = _pa_trim(pool[-1])
            spellings = [t for title in titles for t in title if _pa_trim(t) == root]
            if spellings:
                return max(sorted(set(spellings)), key=spellings.count)
            return m.item if len(pool) > 1 else pool[-1]

        def _kin(self):
            """Opened listings of the wanted sort (not screened out, not judged the wrong kind), when at least three
            are; else every opened one. Labels of other sorts would only offer questions about them."""
            m = self.m
            kin = {lid for lid in m.opened if (m.screen.get(lid) or ("maybe",))[0] != "no"
                   and (m.verdicts.get(lid) or {}).get("kind", True) is not False}
            return kin if len(kin) >= 3 else m.opened

        def plan(self):
            m = self.m
            tally = _pc_label_tally(self._kin(), m.rows)
            table = _pc_merged_labels(tally)
            ans = m.seer.ask(TOPICS_GUIDE, self._prompt(table), TOPICS_SPEC, "ask", 900)
            proposed = [str(t) for t in (ans or {}).get("topics") or [] if str(t).strip()]
            pointed = self._recorded_label(table, (ans or {}).get("story_points_to"))
            if pointed:
                proposed = [pointed] + [t for t in proposed if not m._pc_one_property(t, pointed)]
            lead = self._open_classes(table, tally)
            proposed = lead + [t for t in proposed if not any(m._pc_one_property(t, x) for x in lead)]
            m.topics = self._usable(proposed + _pc_spare_topics(table, proposed, m._pc_one_property))

        def _recorded_label(self, table, named):
            """The recorded label the model said the story points to: a table head or one of its other spellings
            holding the named words (or held by them); "" when it names none of them."""
            m = self.m
            if not isinstance(named, str) or not named.strip():
                return ""
            bare = lambda text: norm(text).replace(" ", "")
            for e in table:
                for n in e["names"]:
                    if bare(n) == bare(named):
                        return n
            for e in table:
                if any(m._pc_one_property(n, named) for n in e["names"]):
                    return e["head"]
            return ""

        def _open_classes(self, table, tally):
            """Heads of stated properties whose one-word value only names a class of the recorded values (the
            last word of a longer value, with no figure in front): the shopper's pick inside the class is still
            unknown. Not once that property was asked or answered."""
            m = self.m
            answered = [n["attr"] for n in m.musts if n.get("from") != "brief"]

            def still_open(rq):
                spoken = words(rq["value"])
                if rq.get("from") != "brief" or len(spoken) != 1 or self._settled(rq, tally):
                    return None
                prop = rq["attr"]
                if any(m._same_attr(a, prop) for a in answered) or any(m._pc_one_property(q, prop) for q, _ in m.ask_log):
                    return None
                for e in table:
                    if any(m._same_attr(n, prop) for n in e["names"]):
                        return e if self._class_word(e["values"], spoken[0]) else None
                return None
            return [e["head"] for e in map(still_open, m.musts) if e]

        @staticmethod
        def _class_word(values, last):
            for v in values:
                ws = words(v)
                if len(ws) > 1 and ws[-1] == last and not re.search(r"\d", " ".join(ws[:-1])):
                    return True       # the front names a sort, not an amount
            return False

        def _usable(self, candidates):
            m = self.m
            tally = _pc_label_tally(m.opened, m.rows)
            fixed = [rq["attr"] for rq in m.musts if self._settled(rq, tally)]
            kept, seen = [], set()
            for subj in candidates:
                key = norm(subj)
                if key in seen or any(m._pc_one_property(a, subj) for a in fixed):
                    continue
                if set(words(label_words(subj))) <= self.CODE_LABELS:
                    continue          # codes and variant pickers are no shopper wish
                seen.add(key)
                kept.append(subj)
            return kept

        def _settled(self, rq, tally):
            """True unless the stated value matches no recorded value of its property yet sits inside one."""
            if rq.get("not"):
                return True
            spoken_in = norm(rq["value"])
            if not spoken_in:
                return True
            recorded = {v for nm, counts in tally["values"].items() if self.m._same_attr(nm, rq["attr"])
                        for v in counts}
            if spoken_in in recorded:
                return True
            mine = set(words(spoken_in))
            return not any(mine <= set(words(v)) for v in recorded)

        def _prompt(self, table):
            m = self.m
            parts = ["Shopper's message:\n" + m.request, "Item these questions are about: " + (m.item or "(the item asked for)"),
                     "Stated up front (no need to ask): " + m._req_text()]
            if m.ask_log:
                parts.append("Questions sent so far (property -> how the reply was classed):\n"
                             + "\n".join("- %s -> %s" % pair for pair in m.ask_log))
            rows = []
            for e in table[:40]:
                tally = e["values"]
                cells = [e["head"], "%d listings" % e["count"], "%d values" % e["kinds"],
                         ", ".join(sorted(tally, key=lambda v: (-tally[v], v))[:6])]
                if len(e["names"]) > 1:
                    cells.append("also written as " + ", ".join(n for n in e["names"] if n != e["head"]))
                rows.append("- " + "; ".join(cells))
            parts.append("Recorded properties, %d listings opened - each line: name; listings with it; count of "
                         "different values; values seen most:\n" % len(m.opened) + ("\n".join(rows) or "- none yet, nothing has been opened"))
            pickers = [e["values"] for e in table if e["kinds"] > 1 and re.search(r"option|variant", e["head"])][:2]
            offered = "; ".join(", ".join(tuple(v)[:8]) for v in pickers)
            if offered:
                parts.append("Variant choices a buyer makes on these listings: " + offered)
            return "\n\n".join(parts)

        def follows(self, subj, prop):
            """Each value of attr pairs with a single value of topic across the opened listings (at least two
            values of each seen)."""
            m = self.m
            pairs: dict[str, set] = {}
            for lid in m.opened:
                facts = m._pc_card(lid).get("traits") or {}
                flat = {label_words(k): norm(facts[k]) for k in facts}
                a = _pc_first_match(flat, prop, m._pc_one_property)
                b = _pc_first_match(flat, subj, m._pc_one_property)
                if a is not None and b is not None:
                    pairs.setdefault(a, set()).add(b)
            if len(pairs) < 2:
                return False
            # a topic with one value everywhere is told apart by nothing
            return len(set().union(*pairs.values())) >= 2 and all(len(g) == 1 for g in pairs.values())

    # ================================================================ service: lines from the shopper or store
    class _pdc_Listener:
        def __init__(self, mission):
            self.m = mission

        @staticmethod
        def _items(rd, field):
            return [x for x in rd.get(field) or [] if isinstance(x, dict) and x.get("attribute") and x.get("value")]

        def take(self, lines, q_about):
            m = self.m
            fl = {"unavailable": False, "changed": False}
            spoken_in = [x for x in lines if x and x.strip()]
            if not q_about:
                # the same line reaching us twice is read once
                fresh, keys = [], set()
                for x in spoken_in:
                    k = norm(x)
                    if k not in m.heard and k not in keys:
                        keys.add(k)
                        fresh.append(x)
                spoken_in = fresh
                m.heard.update(keys)
            if not spoken_in:
                return fl
            fl["heard"] = True
            txt = "\n".join(spoken_in)
            m.said.append(txt)
            prompt = self._prompt(txt, q_about)
            # a line nobody asked for (a new limit, a dropped need, a change) can overturn the pick; two such reads
            # go ahead even when the run's request or spend cap is used up
            vital = not q_about and m.late_reads < 2
            m.late_reads += vital
            if q_about and m.vital_answers > 0:
                # each product of a several-product order hears its few answers in full, whatever the caps
                vital, m.vital_answers = True, m.vital_answers - 1
            rd = m.seer.ask(READ_LINE_GUIDE, prompt, READ_LINE_SPEC, "quick", 700, vital=vital)
            clash = self._clash(rd) if q_about else []
            if clash:
                # two values on overlapping names for one answer: one more read by the deep seat
                second = m.seer.ask(READ_LINE_GUIDE, prompt + READ_LINE_AGAIN % " and ".join(clash), READ_LINE_SPEC,
                                    "deep", 1500)
                if isinstance(second, dict) and not self._clash(second):
                    rd = second
            if isinstance(rd, dict) and q_about and rd.get("no_preference") and isinstance(rd.get("reveals"), list):
                # an answer of no preference states no requirement: keep only values the line itself spells out
                rd["reveals"] = [x for x in rd["reveals"] if isinstance(x, dict) and _pdd_quoted(str(x.get("value") or ""), txt)]
            if isinstance(rd, dict) and q_about:
                self._as_spoken(rd, txt, q_about)
            self._apply(rd or self._plain(txt, q_about), txt, q_about, fl)
            return fl

        def _as_spoken(self, rd, txt, q_about):
            """An answer's property as the reply names it: a reveal or change filed under a property the reply never
            names, while the reply names the one asked about, is about the one asked about (a near name would
            overwrite another stated requirement)."""
            said, asked = set(stems(txt)), set(stems(label_words(q_about)))
            if not asked or not asked <= said:
                return
            for field in ("reveals", "changes"):
                for x in rd.get(field) or []:
                    if isinstance(x, dict) and not set(stems(label_words(str(x.get("attribute") or "")))) <= said:
                        x["attribute"] = q_about

        def _prompt(self, spoken_in, q_about):
            m = self.m
            rules = ["  [%d] %s -> %s%s" % (n, rq["attr"], "any value except " if rq.get("not") else "", rq["value"])
                     for n, rq in enumerate(m.musts, 1)] or ["  (none yet)"]
            budget = "not given" if m.max_price is None else str(m.max_price)
            context = "\n\nThe assistant's last question was about: " + q_about if q_about else ""
            return ("Shopping for: %s\nCurrent requirements, numbered:\n%s\nBudget ceiling: %s%s\n\nThe line to read:\n%s"
                    % (m.item, "\n".join(rules), budget, context, spoken_in))

        def _triples(self, rd, field, cp):
            """(property, value, bound) for each usable pair of one field, minus pairs carrying the budget. A pair
            met twice takes the bound of its last appearance."""
            found = [(str(x["attribute"]), str(x["value"]), _pc_bound(x)) for x in self._items(rd, field)]
            last = {(a, v): b for a, v, b in found}
            return [(a, v, last[(a, v)]) for a, v, _ in found if not self._budget_like(v, cp)]

        def _reveals(self, tl, txt, q_about, fl):
            m = self.m
            single = bool(q_about) and len(tl) == 1
            spoken = set(words(txt))
            for nm, wv, lim in tl:
                if q_about and not m._pc_one_property(nm, q_about) and self._repeats(nm, wv):
                    continue          # a requirement in force said again; the question stays unanswered
                if single and not set(words(label_words(nm))) <= spoken:
                    # a lone answer goes to the asked property unless it names its own: a recorded label the
                    # line spells out in full
                    nm = self._spelled_label(spoken) or q_about
                if not q_about and not self._recorded(nm):
                    continue          # unprompted: only properties the opened listings carry
                fl.setdefault("revealed", []).append(nm)
                m._set_need(nm, wv, "told", replace=False, rel=lim)
                fl["changed"] = True

        def _spelled_label(self, spoken):
            """The recorded property label whose words the line holds in full, the one of most words; "" when
            none does or two of that size do."""
            m = self.m
            named = {}
            for lab in _pc_label_tally(m.opened, m.rows)["values"]:
                parts = set(words(label_words(lab)))
                if parts and parts <= spoken:
                    named.setdefault(len(parts), set()).add(" ".join(words(label_words(lab))))
            if not named:
                return ""
            top = named[max(named)]
            return next(iter(top)) if len(top) == 1 else ""

        def _changes(self, tl, q_about, fl):
            m = self.m
            for nm, wv, lim in tl:
                if q_about or any(m._same_attr(n["attr"], nm) for n in m.musts) or self._recorded(nm):
                    m._set_need(nm, wv, "told", replace=True, rel=lim)
                    fl["changed"] = True

        def _apply(self, rd, txt, q_about, fl):
            m = self.m
            cp = as_written(rd.get("max_price"), txt, m.max_price)
            # the facts that single out a pointed-to listing describe that listing, not the pick
            facts = [set(words(str(d))) for p in rd.get("pointers") or [] if isinstance(p, dict)
                     for d in (p.get("details") or []) if isinstance(d, (str, int, float))]
            tl = [t for t in self._triples(rd, "reveals", cp)
                  if not any(set(words(t[1])) and set(words(t[0] + " " + t[1])) <= f for f in facts)]
            self._reveals(tl, txt, q_about, fl)
            self._changes(self._triples(rd, "changes", cp), q_about, fl)
            # only a line nobody asked for may withdraw a requirement
            before = len(m.musts)
            gone = set() if q_about else self._drop(rd.get("drops") or [])
            if gone:
                # a need given up removes its own verdicts; the other verdicts still hold
                for lid, vd in list(m.verdicts.items()):
                    by = list(vd.get("by") or [])
                    if len(by) != before:
                        del m.verdicts[lid]
                        continue
                    m.verdicts[lid] = m._grounded(lid, dict(vd, by=[mk for at, mk in enumerate(by) if at not in gone]))
                fl["changed"] = True
            if cp is not None and cp > 0:
                m.cap_rose = m.cap_rose or (m.max_price is not None and cp > m.max_price)
                m.max_price, fl["changed"] = cp, True
            if rd.get("pick_unavailable"):
                fl["unavailable"] = True
            pointed = [p for p in rd.get("pointers") or [] if isinstance(p, dict) and p.get("attribute")
                       and str(p.get("shop") or "").strip()]
            if pointed:
                m.pointers.extend(pointed)
                _PddReferee(m).resolve(pointed[:1])
                m.verdicts.clear()
                fl["changed"] = True
            if not q_about or fl.get("revealed"):
                return
            kind = str(rd.get("reply_class") or "")
            if kind == "split" and rd.get("not_understood"):
                kind = "other"              # not understood at all: a fresh topic pays more than the same one again
            quote = norm(str(rd.get("quote") or ""))
            if kind in ("not_picky", "split") and quote and quote in norm(txt):
                fl["reply_class"] = kind    # trusted only with its quote found in the reply

        def _repeats(self, nm, wv):
            m = self.m
            return any(not rq.get("not") and m._same_attr(rq["attr"], nm) and norm(rq["value"]) == norm(wv)
                       for rq in m.musts)

        def _clash(self, rd):
            if not isinstance(rd, dict):
                return []
            m = self.m
            tl = [(str(x["attribute"]), str(x["value"])) for x in self._items(rd, "reveals")]
            names = []
            for i in range(len(tl)):
                for j in range(i + 1, len(tl)):
                    (a, va), (b, vb) = tl[i], tl[j]
                    if norm(va) != norm(vb) and (m._overlaps(a, b) or m._same_attr(a, b)):
                        names += [x for x in (a, b) if x not in names]
            return names

        def _recorded(self, nm):
            m = self.m
            labels = _pc_label_tally(m.opened, m.rows)["values"]
            return any(m._overlaps(lab, nm) or m._same_attr(lab, nm) for lab in labels)

        @staticmethod
        def _budget_like(given, cp):
            """A value carrying the line's own spending figure is the budget, not a property."""
            if cp is None:
                return False
            figures = [number(x) for x in _pa_UNSIGNED.findall(str(given))]
            return any(f is not None and abs(f - cp) < 1e-6 for f in figures)

        def _drop(self, drops):
            """Removes requirements named, or numbered from 1; the positions that went."""
            m = self.m
            numeric = lambda d: isinstance(d, int) or str(d).strip().isdigit()
            places = {int(d) - 1 for d in drops if numeric(d)}
            labels = [str(d) for d in drops if not numeric(d)]
            stays = lambda at, rq: at not in places and not any(m._same_attr(rq["attr"], d) for d in labels)
            gone = {at for at, rq in enumerate(m.musts) if not stays(at, rq)}
            m.musts = [rq for at, rq in enumerate(m.musts) if at not in gone]
            return gone

        def _plain(self, spoken_in, q_about):
            """Reading without a model: nothing is read from wording alone; only an answer is taken, as the one
            recorded value of the asked property that the reply spells out (the longest when others sit in it)."""
            m = self.m
            result = {"reveals": [], "changes": [], "drops": [], "pick_unavailable": False}
            if q_about:
                spaced = lambda raw: " " + " ".join(words(raw)) + " "
                heard_line = spaced(spoken_in)
                values = _pc_label_tally(m.opened, m.rows)["values"]
                near = {v for lab, vals in values.items() if m._overlaps(lab, q_about) or m._same_attr(lab, q_about)
                        for v in vals}
                hits = sorted({v for v in near if words(v) and spaced(v) in heard_line}, key=len, reverse=True)
                # two separate values say nothing; shorter ones inside the longest are part of it
                if hits and all(spaced(v) in spaced(hits[0]) for v in hits):
                    result["reveals"] = [{"attribute": q_about, "value": hits[0]}]
            return result

        def record(self, prop, given, source, replace, lim):
            m = self.m
            if source == "told" and m.item:
                self._search_for(given, prop)
            target = norm(given)
            for rq in m.musts:
                if rq.get("not") or not m._same_attr(rq["attr"], prop):
                    continue
                if norm(rq["value"]) == target and rq.get("rel", "") == lim:
                    return            # nothing new: verdicts stay
                # an answer adds to the request; only a change overwrites a stated value
                if replace or rq["from"] != "brief":
                    rq.update({"value": given, "from": source, "rel": lim})
                    break
            else:
                m.musts.append(dict(attr=prop, value=given, rel=lim, **{"from": source}))
            m.verdicts.clear()

        def _search_for(self, given, prop=""):
            """Put searches for a told value first, in the shopper's own words: the value beside the shortest value
            the request stated for another property (the listings holding both come up together), then the value
            beside the item as the request names it."""
            m = self.m
            texts = [m.request] + list(m.said)
            told = _pdd_as_written(given, list(reversed(m.said)) + [m.request]) or given
            noun = _pdd_as_written(m.item, texts) or m.item
            stated = []
            for n in m.musts:
                if n.get("from") != "brief" or n.get("not") or norm(n["value"]) == norm(given):
                    continue
                if prop and (m._same_attr(n["attr"], prop) or m._overlaps(n["attr"], prop)):
                    continue
                v = _pdd_as_written(n["value"], texts)
                if v and len(words(v)) <= 4 and not set(stems(v)) <= set(stems(noun)):
                    stated.append(v)
            stated.sort(key=lambda v: len(words(v)))
            both = " ".join([told] + stated[:1]) if stated else ""
            known = {norm(x) for x in m.searches}
            fresh = [q for q in (both, _pdd_with_noun(told, noun, m.text_room())) if q and norm(q) not in known]
            m.searches[0:0] = fresh
            m.told_queries.extend(fresh)

    # ================================================================ service: judging opened listings
    class _pdc_Checker:
        def __init__(self, mission):
            self.m = mission

        def run(self, pids):
            m = self.m
            pending = [it for it in pids if it not in m.verdicts]
            if not pending:
                return
            # sections where screened rows were the right kind make their listings the right kind
            good = {c for it, (look, _) in m.screen.items() for c in [m.rows[it].get("category")] if look == "yes" and c}
            named = m._pc_named_shelves()
            for at in range(0, len(pending), JUDGE_SIZE):
                bunch = pending[at:at + JUDGE_SIZE]
                ans = m.seer.ask(JUDGE_GUIDE, self._prompt(bunch), JUDGE_SPEC, "deep", 2500)
                parsed = self._rows(ans, bunch)
                unit = str((ans or {}).get("figure_unit") or "").strip()
                if parsed and unit and not m.figure_unit:
                    m.figure_unit = unit[:40]       # later groups use the same unit
                for it in bunch:
                    ruling = m._grounded(it, parsed[it] if it in parsed else m._code_judge(it))
                    section = m._pc_card(it).get("category")
                    if section in good:
                        ruling["kind"] = True
                    if named and section and section not in named:
                        ruling["kind"] = False
                    m.verdicts[it] = ruling

        def _rows(self, ans, bunch):
            """Usable model rows by id; a row whose verdict count differs from the requirement count is dropped."""
            wanted = len(self.m.musts)
            out = {}
            for at, ln in Mission._pdc_numbered(ans, "listings", len(bunch)):
                mks = list(map(str, ln.get("verdicts") or []))
                if len(mks) != wanted:
                    continue
                out[bunch[at]] = {"kind": ln.get("right_kind") is not False, "by": mks,
                                  "figure": number(ln.get("figure"))}
            return out

        @staticmethod
        def _rule_line(n, rq):
            shown, lim = rq["value"], rq.get("rel")
            if lim and lim not in norm(shown):
                shown = lim + " " + shown
            if rq.get("not"):
                shown = "NOT %s (its absence earns a yes)" % (shown,)
            return "%d. %s: %s" % (n, rq["attr"], shown)

        @staticmethod
        def _card_block(i, info):
            facts = info.get("traits") or {}
            recorded = " | ".join("%s = %s" % (label_words(k), facts[k]) for k in facts)
            said = " ".join(info.get("notes") or [])          # an offer's remarks, only once its offer was read
            return ("[%d] %s\n    category: %s\n    properties: %s\n    description: %s\n"
                    % (i, info.get("title", ""), info.get("category", ""), recorded or "(nothing recorded)",
                       str(info.get("description", ""))[:300])
                    + ("    seller remarks: %s\n" % said[:400] if said else ""))

        def _prompt(self, bunch):
            m = self.m
            rules = [self._rule_line(n, rq) for n, rq in enumerate(m.musts, 1)] or ["(no requirements)"]
            head = "Rule on every requirement below, one verdict per line and in the same order:\n" + "\n".join(rules)
            head += "\n\nShopper is buying: %s" % (m.item,)
            if m.best_measure and m.priority == "best":
                head += "\nMeasure to read off each listing as a number: %s" % (m.best_measure,)
                if m.figure_unit:
                    head += " (in %s)" % (m.figure_unit,)
            cards = "\n".join(self._card_block(i, m._pc_card(lid)) for i, lid in enumerate(bunch))
            return (head + "\n\nLISTINGS follow (the first one is 0):\n" + cards).rstrip("\n")

        def verify(self, lid, verdict):
            """Hold model marks to the listing text: a yes needs the value to show (figures as figures); an
            excluded value on the listing is a no. Counts how well each value is backed."""
            m = self.m
            info = m._pc_card(lid)
            facts = info.get("traits") or {}
            txt = " ".join([str(info.get("title")), str(info.get("category")), str(info.get("description"))]
                            + [str(x) for x in facts.values()])
            loose, flat = set(stems(txt)), norm(txt)
            mks = list(verdict["by"])
            tally = {"exact": 0, "stated": 0, "proven": 0}
            for pos, rq in enumerate(m.musts[:len(mks)]):
                mks[pos], gains = self._one(rq, mks[pos], info, facts, loose, flat)
                for nm in gains:
                    tally[nm] += 1
            return dict(verdict, by=mks, **tally)

        def _one(self, rq, cur, info, facts, loose, flat):
            m = self.m
            wv = rq["value"]
            ents = m._pc_entries_for(facts, rq["attr"])
            own_e = _pc_own_entries(facts, rq["attr"], m.item)
            if rq.get("not"):
                if _pc_excluded_shows(wv, facts, {k for k, _ in ents}):
                    return "no", ()
                return cur, (("proven",) if own_e or ents else ())
            if rq.get("rel") and own_e:
                measured = _pc_measure(str(wv), rq["rel"], own_e)
                if measured is not None:
                    if cur != "no" or measured != "yes":
                        cur = measured     # another scale may hide a figure: never lift the model's no
                    return cur, (("stated", "exact") if cur == "yes" else ())
                if cur == "yes" and _pc_other_scale(str(wv), own_e):
                    return cur, ()         # figures only in another unit: code cannot compare, the model's yes stands
            value_words = set(stems(wv))
            if value_words and value_words <= set(stems(info.get("category"))) and not re.search(r"\d", str(wv)):
                return "yes", ("stated", "exact")      # filed under a section named by the value
            checked = _pc_check_positive(cur, str(wv), ents, loose, flat)
            if checked is None:
                return cur, ()
            if own_e and not re.search(r"\d", str(wv)) and norm(wv) in set().union(*(_pc_pieces(v) for _, v in own_e)):
                checked = "yes"
            gains = []
            joined = " ".join(str(v) for _, v in ents)
            if joined and set(stems(joined)) & set(stems(wv)):
                gains.append("stated")
            if norm(wv) in set().union(set(), *(_pc_pieces(v) for _, v in ents)):
                gains.append("exact")
            return checked, gains

        def by_rules(self, lid):
            """Judgement without a model: the property's own entry decides, else the title."""
            m = self.m
            info = m._pc_card(lid)
            by_label = {label_words(k): norm(v) for k, v in (info.get("traits") or {}).items()}
            title = norm(info.get("title"))
            mks = []
            for rq in m.musts:
                want = norm(rq["value"])
                texts = [t for lab, t in by_label.items() if m._same_attr(lab, rq["attr"])]
                shows = any(want in t for t in texts) or bool(want and want in title)
                if rq.get("not"):
                    # an exclusion: the value seen anywhere rules the listing out; recorded without it, it passes
                    mks.append("no" if shows else "yes" if texts else "unclear")
                elif texts and rq.get("rel"):
                    # a limit is held against comparable figures only; another unit leaves it undecided
                    mks.append(_pc_measure(str(rq["value"]), rq["rel"], [(None, t) for t in texts]) or "unclear")
                elif texts:
                    # an entry whose every word sits inside the wanted phrase may be one of the values it allows
                    within = any(set(stems(t)) and set(stems(t)) <= set(stems(want)) for t in texts)
                    mks.append("yes" if any(want in t for t in texts) else "unclear" if within else "no")
                else:
                    mks.append("yes" if want and want in title else "unclear")
            return {"kind": m.screen.get(lid, ("maybe", ""))[0] != "no", "by": mks, "figure": None}

    # ------------------------------------------------------------------ shared record access
    def absorb(self, obs: Any) -> None:
        """Queue every new note an observation carries; they are read at the next decision point."""
        if isinstance(obs, str) or obs is None:
            return
        backlog = self.pending_notes
        for line in find_notes(obs):
            if line not in backlog:
                backlog.append(line)

    def _pc_card(self, pid: str) -> dict:
        """Everything known about one listing; an empty dict for an id never seen."""
        known = self.rows.get(pid)
        return known if known else {}

    def _pc_entries_for(self, props: dict, attr: str) -> list[tuple[Any, Any]]:
        """The (key, value) entries of props whose key names attr or a property overlapping it."""
        related = lambda key: self._overlaps(key, attr) or self._same_attr(key, attr)
        return [pair for pair in props.items() if related(pair[0])]

    def go(self) -> list[dict]:
        """Run the whole episode through the coordinator; the answered steps are returned."""
        _PddCoordinator(self).run()
        return self.dialogue


# ---------------------------------------------------------------------- text and record helpers
_PDD_CUT = re.compile(r"[,/;|]")
_PDD_SELLER_SIDE = re.compile(r"brand|price|stock|condition|warrant|ship|deliver|seller|model|name|options?$|variants?$|"
                              r"sku|id$")
_PDD_LEAD_FIGURE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*([a-z]*)")
_PDD_DIGIT = re.compile(r"\d")
_PDD_NUMBERS = re.compile(r"\d+(?:\.\d+)?")
_PDD_BARE_YES = re.compile(r"\s*(?:yes|true)\s*")


def _pdd_bump(counter: dict, key: Any, by: int = 1) -> None:
    counter[key] = counter.get(key, 0) + by


def _pdd_leaves(obj: Any):
    """Every list and plain value inside a reply, at any depth."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _pdd_leaves(v)
    elif isinstance(obj, list):
        yield obj
        for v in obj:
            if isinstance(v, (dict, list)):
                yield from _pdd_leaves(v)
    else:
        yield obj


def _pdd_loose_token(reply: dict, ids: list[str]) -> Any:
    """A next-page token under a name no reader knows: the one top-level text of a search reply shaped like a token
    (short, a digit, no blanks) that is no listing id; None when there is not exactly one."""
    shaped = [v for k, v in reply.items() if isinstance(v, str) and _pda_ID_LOOK.match(v) and v not in ids
              and not _pda_KEY["id"].search(str(k))]
    return shaped[0] if len(shaped) == 1 else None


def _pdd_quoted(part: str, whole: str) -> bool:
    """Is `part` copied from `whole`? Its words in a row, or nearly all of a longer part (a copy may slip on a sign)."""
    said, there = words(part), words(whole)
    if not said:
        return False
    if " %s " % " ".join(said) in " %s " % " ".join(there):
        return True
    seen = set(there)
    return len(said) >= 6 and sum(w in seen for w in said) >= 0.9 * len(said)


def _pdd_keys(obj: Any) -> list[str]:
    """Every field name in a reply, at any depth."""
    out, todo = [], [obj]
    while todo:
        node = todo.pop()
        if isinstance(node, dict):
            out += [str(k) for k in node]
            todo += list(node.values())
        elif isinstance(node, list):
            todo += node
    return out


def _pdd_texts(obj: Any) -> list[str]:
    """Every text value in a reply, at any depth (figures held as numbers are left out)."""
    out, todo = [], [obj]
    while todo:
        node = todo.pop()
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            todo += list(node.values())
        elif isinstance(node, list):
            todo += node
    return out


def _pdd_raw_records(obj: Any, ids: list[str]) -> dict:
    """id -> the first raw object in a reply that carries one of these listing ids, at any depth."""
    found, todo = {}, [obj]
    while todo:
        node = todo.pop(0)
        if isinstance(node, dict):
            ident = _pda_id_of(node)
            if ident in ids and ident not in found:
                found[ident] = node
                continue
            todo.extend(node.values())
        elif isinstance(node, list):
            todo.extend(node)
    return found


def _pdd_entries_of(opened: set, rows: dict):
    """(listing id, property name, raw value) for every named entry of the opened listings, ids in sorted order."""
    for lid in sorted(opened):
        record = rows.get(lid) or {}
        for label, raw in (record.get("traits") or {}).items():
            name = label_words(label)
            if name:
                yield lid, name, raw


def _pc_label_tally(opened: set, rows: dict) -> dict:
    """For the opened listings (taken in sorted id order so a prompt never depends on set order): how often each
    piece of each property label occurs, how many listings carry the label, and which ones."""
    values: dict[str, dict] = {}
    carried: dict[str, int] = {}
    holders: dict[str, set] = {}
    for lid, name, raw in _pdd_entries_of(opened, rows):
        bucket = values.setdefault(name, {})
        cut = [norm(part)[:40] for part in _PDD_CUT.split(str(raw))]
        for piece in filter(None, cut):
            _pdd_bump(bucket, piece)
        _pdd_bump(carried, name)
        holders.setdefault(name, set()).add(lid)
    return dict(values=values, carried=carried, holders=holders)


def _pdd_spellings(word: str) -> set:
    return {word, word + "s", _pa_trim(word)}


def _pdd_covers(inner: set, outer: set) -> bool:
    """True when inner is non-empty and each of its words matches some word of outer (singular or plural)."""
    if not inner:
        return False
    outer_forms = [_pdd_spellings(o) for o in outer]
    return all(any(_pdd_spellings(w) & forms for forms in outer_forms) for w in inner)


def _pdd_shorter_first(name: str) -> tuple:
    return len(words(name)), name


def _pdd_widest_first(row: dict) -> tuple:
    return -row["count"], -row["kinds"], row["head"]


def _pdd_fold(head: str, names: list[str], tally: dict) -> dict:
    """One table row: the members' value counts summed, and the listings carrying any member."""
    merged: dict[str, int] = {}
    carriers: set = set()
    for member in names:
        carriers.update(tally["holders"][member])
        for val, times in tally["values"][member].items():
            _pdd_bump(merged, val, times)
    return dict(count=len(carriers), kinds=len(merged), head=head, names=names, values=merged)


def _pc_merged_labels(tally: dict) -> list[dict]:
    """One row per property: label spellings whose words contain a shorter label's words are folded under that
    shorter head. Rows come most widely carried first, then by value variety, then by head."""
    heads: list[str] = []
    members: dict[str, list] = {}
    for name in sorted(tally["values"], key=_pdd_shorter_first):
        mine = set(words(name))
        home = next((h for h in heads if _pdd_covers(set(words(h)), mine)), name)
        if home == name and name not in members:
            heads.append(name)
        members.setdefault(home, []).append(name)
    return sorted((_pdd_fold(h, members[h], tally) for h in heads), key=_pdd_widest_first)


def _pc_spare_topics(table: list[dict], ranked: list[str], overlaps: Any) -> list[str]:
    """Extra properties for unused questions, in table order; seller-side labels and ones already ranked are left out."""
    out = []
    for row in table:
        head = row["head"]
        if _PDD_SELLER_SIDE.search(head):
            continue
        if any(overlaps(head, r) for r in ranked):
            continue
        out.append(head)
    return out


def _pc_first_match(entries: dict, name: str, overlaps: Any) -> Any:
    return next((text for label, text in entries.items() if overlaps(label, name)), None)


def _pdd_has_figure(fig: str, text: str) -> bool:
    """fig occurs in text as a whole number (no digit or decimal point glued to its front, no digit after)."""
    return bool(re.search(r"(?<![\d.])" + re.escape(fig) + r"(?![\d])", text))


def _pc_excluded_shows(value: str, traits: dict, own_keys: set) -> bool:
    """True when the excluded value appears as one listed value of any entry, or all its stems sit in an entry
    that belongs to the property itself."""
    flat = norm(value)
    banned = set(stems(value))
    for key, text in traits.items():
        if flat in [norm(part) for part in _PDD_CUT.split(str(text))]:
            return True
        if banned and key in own_keys and banned.issubset(stems(text)):
            return True
    return False


def _pc_bound(pair: Any) -> str:
    """"at least" or "at most" when a reading limits a number that way; "" otherwise."""
    if not isinstance(pair, dict):
        return ""
    said = norm(pair.get("bound"))
    return said if said in ("at most", "at least") else ""


def _pdd_core_stems(attr: str, item: str) -> set:
    """Meaningful stems (longer than two letters) of a property name, minus the item's own words when anything
    else is left."""
    long_ones = [s for s in stems(label_words(attr)) if len(s) > 2]
    item_bag = set(stems(label_words(item)))
    narrowed = [s for s in long_ones if s not in item_bag]
    return set(narrowed) if narrowed else set(long_ones)


def _pc_own_entries(props: dict, attr: str, item: str) -> list[tuple[Any, Any]]:
    """Entries whose key holds all of the property's core stems (more words in the key are fine)."""
    core = _pdd_core_stems(attr, item)
    if not core:
        return []
    return [(k, v) for k, v in props.items() if core.issubset(stems(label_words(k)))]


def _pc_pieces(raw: Any) -> set:
    return set(map(norm, _PDD_CUT.split(str(raw))))


def _pdd_letters(text: Any) -> set:
    return {s for s in stems(text) if not _PDD_DIGIT.search(s)}


def _pc_measure(val: str, rel: str, own: list) -> str | None:
    """Check a numeric limit against the property's own entries. "yes": some comparable figure keeps the limit;
    "no": comparable figures exist but none keeps it; None: nothing to compare. An entry is comparable when it
    names no unit or shares a unit word with the wanted value; units are never converted."""
    goal = number(val)
    if goal is None:
        return None
    want_units = _pdd_letters(val)
    keeps = (lambda f: f <= goal) if rel == "at most" else (lambda f: f >= goal)
    compared = False
    for _label, entry in own:
        have_units = _pdd_letters(entry)
        if want_units and have_units and want_units.isdisjoint(have_units):
            continue
        figures = [f for f in map(number, _pa_UNSIGNED.findall(str(entry))) if f is not None]
        compared = compared or bool(figures)
        if any(keeps(f) for f in figures):
            return "yes"
    return "no" if compared else None


def _pc_other_scale(val: str, own: list) -> bool:
    """Do the property's own entries give figures, every one of them in a unit the wanted value does not use?"""
    want_units = _pdd_letters(val)
    if not want_units:
        return False
    seen = False
    for _label, entry in own:
        if not _pa_UNSIGNED.findall(str(entry)):
            continue
        have_units = _pdd_letters(entry)
        if not have_units or not want_units.isdisjoint(have_units):
            return False
        seen = True
    return seen


def _pc_check_positive(mark: str, val: str, entries: list, loose: set, flat: str) -> str | None:
    """Hold a model mark against the listing text. None for a plain yes/true value (nothing to check); a "yes" turns
    "unclear" when the value is nowhere to be seen and no own entry exists, when a wanted figure is missing from the
    own entries, or when the own entries share no word with the value."""
    digits = _PDD_NUMBERS.findall(val)
    if not digits and _PDD_BARE_YES.fullmatch(norm(val)):
        return None
    if mark != "yes":
        return mark
    content = [] if digits else [s for s in stems(val) if len(s) > 2]
    if not entries:
        if digits:
            shown = any(_pdd_has_figure(d, flat) for d in digits)
        else:
            shown = not content or any(s in loose for s in content)
        return mark if shown else "unclear"
    own_text = norm(" ".join(str(v) for _, v in entries))
    if digits:
        return mark if any(_pdd_has_figure(d, own_text) for d in digits) else "unclear"
    if content and set(content).isdisjoint(stems(own_text)):
        return "unclear"
    return mark


def _pc_listing_records(pid: str, seen: Any) -> list[dict]:
    """Records from an opened listing's reply, all filed under the id that was opened."""
    recs = find_records(seen)
    if not recs:
        usable = isinstance(seen, dict) and not error_of(seen, need_data=True)
        sheet = _traits_of(seen) if usable else None
        return [{"id": pid, "traits": sheet}] if sheet else []
    keep_all = len(recs) == 1
    return [{**rec, "id": pid} for rec in recs if keep_all or rec["id"] == pid]


_PDD_SWAP_STEPS = 5     # an offer read is worth its step only with room left to swap the pick: remove, add, read, order


def _pdd_offer_remarks(pid: str, seen: Any) -> list[str]:
    """Sentences a listing's own record carries under a field that no record slot reads (an offer's remark, say);
    find_records passes them over and find_notes does not enter records. The listing opened, or a reply's only
    record, is read."""
    cards, found = [], []

    def walk(node):
        if isinstance(node, list):
            for part in node:
                walk(part)
        elif isinstance(node, dict):
            card = as_record(node)
            if card is not None:
                cards.append((card, node))
                return
            for part in node.values():
                walk(part)

    walk(seen)
    named = ("id",) + _pda_SLOT_ORDER
    for card, node in cards:
        if card["id"] != pid and len(cards) > 1:
            continue
        for key, val in node.items():
            text = val.strip() if isinstance(val, str) else ""
            if " " in text and not any(_pda_KEY[s].search(str(key)) for s in named) and text not in found:
                found.append(text)
    return found


def _pdd_is_lines(val: Any) -> bool:
    return isinstance(val, list) and bool(val) and all(isinstance(x, str) for x in val)


def _pdd_split_reply(obs: Any) -> tuple[list[str], list[str]]:
    """Separate a reply into (answer lines, lines pushed alongside). Only a dict of two or more keys with exactly
    one key holding a list of strings is split; anything else is all answer."""
    if isinstance(obs, str):
        return [obs], []
    side_keys = [key for key, val in obs.items() if _pdd_is_lines(val)] if isinstance(obs, dict) and len(obs) > 1 else []
    if len(side_keys) == 1:
        side = side_keys[0]
        main = {key: val for key, val in obs.items() if key != side}
        spoken = [val for val in main.values() if isinstance(val, str) and val.strip()] or find_notes(main)
        if spoken:
            return spoken, [line for line in obs[side] if line.strip()]
    return find_notes(obs), []


def _pdd_as_written(value: str, texts: list) -> str:
    """The value as the shopper wrote it: the run of their own words that spells it, case, spaces and marks aside
    (a value given as two words may be written as one, or the other way round); "" when no text holds it."""
    want = re.sub(r"[^0-9a-z]", "", str(value).lower())
    if not want:
        return ""
    for text in texts:
        toks = [re.sub(r"^\W+|\W+$", "", t) for t in str(text).split()]
        flat = [re.sub(r"[^0-9a-z]", "", t.lower()) for t in toks]
        for i in range(len(toks)):
            acc = ""
            for j in range(i, min(len(toks), i + 6)):
                acc += flat[j]
                if acc == want and flat[i]:
                    return " ".join(toks[i:j + 1])
                if not want.startswith(acc):
                    break
    return ""


def _pdd_with_noun(value: str, noun: str, room: Any) -> str:
    """The value beside the item, within the search box's length: the full item name when it fits, else its last
    word, else the value alone."""
    for tail in (noun, " ".join(noun.split()[-1:]), ""):
        query = " ".join([value, tail]).strip()
        if not isinstance(room, int) or len(query) <= room:
            return query
    return value


def _pdd_page_ref(token: Any, read: int, numeric: bool = False) -> Any:
    """What to send for the next page (see _pb_page_ref): never a guessed page number unless the schema asks for one."""
    return _pb_page_ref(token, read, numeric)


def _pdd_quiet() -> dict:
    return {"unavailable": False, "changed": False}


def _pdd_bits(raw: Any) -> set:
    """Stems of a text with figures cut away from the letters they are glued to ("x16" holds 16, "4k" 4 and k)."""
    return {_pa_trim(bit) for bit in re.findall(r"[a-z]+|\d+(?:[.,]\d+)*", norm(raw))}


def _pdd_plain_count(job: Mission) -> int:
    return len([need for need in job.musts if not need.get("not")])


def _pdd_filter_tag(shelf: Any, cap: Any, page: Any = None) -> str:
    """The done-mark of one shelf listing: a cap is written the same whether it came as 5995 or 5995.0."""
    figure = number(cap)
    return "filter|%s|%s|%s" % (norm(shelf), "None" if figure is None else ("%d" % figure if figure == int(figure) else "%.2f" % figure), page)


def _pdd_first_page_tag(query: str) -> str:
    return norm(query) + "|None"


def _pdd_mentions_listing(text: str, rows: dict) -> bool:
    """True when an id or title (longer than three characters) of any known row sits inside text."""
    squashed = norm(text)
    labels = [norm(key) for key in rows] + [norm(rec.get("title")) for rec in rows.values()]
    return any(len(label) > 3 and label in squashed for label in labels)


_PDD_DECLINE_TEXT = "None of the listings for sale{limit} has {need} together with the rest of what you asked for, so I am buying nothing."


# ---------------------------------------------------------------------- shopping services
class _PddService:
    """Common access to the shared mission state."""

    def __init__(self, job: Mission) -> None:
        self.job = job

    def _can(self, role: str) -> bool:
        return self.job.tools.has(role)

    def _params(self, role: str) -> dict:
        return self.job.tools.roles.get(role) or {}

    def _cost(self, lid: str) -> float | None:
        value = self.job.rows[lid].get("price")
        return value if isinstance(value, float) else None

    def _spare(self, keep: int = 0) -> int:
        return self.job.steps_remaining() - self.job.reserve - keep


class _PddRanker(_PddService):
    """Ranking service: which judged listings pass, which one is bought, which unopened rows could still win."""

    def passes(self, pid: str) -> bool:
        job = self.job
        verdict = job.verdicts.get(pid)
        sound = bool(verdict) and bool(verdict["kind"]) and all(mark == "yes" for mark in verdict["by"])
        return sound and job.buyable(pid) and pid in job.opened

    def fits(self) -> list[str]:
        return [lid for lid in self.job.verdicts if self.passes(lid)]

    def own_measure(self, pid: str) -> tuple | None:
        """(figure, unit word) of the ranking measure from the listing's own entry: the first key containing every
        word of the measure's name gives its first number and the letters that follow it."""
        name = set(words(label_words(self.job.best_measure)))
        if name:
            for key, text in (self.job._pc_card(pid).get("traits") or {}).items():
                if not name.issubset(words(label_words(key))):
                    continue
                found = _PDD_LEAD_FIGURE.search(norm(text))
                if found is not None:
                    figure, unit = found.groups()
                    return float(figure.replace(",", "")), unit
        return None

    def _cheap_key(self, lid: str, short_named: bool) -> tuple:
        cost = self._cost(lid)
        return (1e18 if cost is None else cost, short_named, self.job.order_seen.index(lid))

    def _top_by_measure(self, pool: list[str]) -> str | None:
        """Best-by-measure choice: own-entry figures when they agree on at most one unit word, else the judge's
        figures; None when neither ranks anything. Ties go to the lower price, then to pool order."""
        job = self.job
        direction = 1 if job.best_high else -1

        def standing(figure: float, lid: str) -> tuple:
            return direction * figure, -(job._pc_card(lid).get("price") or 0.0)

        own = {}
        for lid in pool:
            got = self.own_measure(lid)
            if got is not None:
                own[lid] = got
        units = {unit for _, unit in own.values()}
        units.discard("")
        if own and len(units) < 2:
            return max(own, key=lambda lid: standing(own[lid][0], lid))
        judged = [lid for lid in pool if job.verdicts[lid].get("figure") is not None]
        if judged:
            return max(judged, key=lambda lid: standing(job.verdicts[lid]["figure"], lid))
        return None

    def pick(self) -> str | None:
        job = self.job
        pool = self.fits()
        if not pool:
            return None
        by_price = job.priority == "cheapest"
        # a listing that writes every wanted value out (its own property values or its title) before one whose
        # yes rests on its prose or on the judge alone: bought lines nearly always write the value out
        pool = [lid for lid in pool if self.writes_values(lid)] or pool
        if job.kept in pool and not by_price:
            return job.kept                   # a pick that survived a fresh read stays the pick
        plain = _pdd_plain_count(job)
        score = lambda lid, field: job.verdicts[lid].get(field, 0)

        def prefer(cands: list[str], test: Any) -> list[str]:
            return [lid for lid in cands if test(lid)] or cands

        if not by_price:
            # entries that write out every wanted value first; failing that, those writing out the most
            most = max(score(lid, "stated") for lid in pool)
            pool = [lid for lid in pool if score(lid, "stated") >= min(plain, most)]
        # recording another value for an excluded property beats saying nothing about it
        pool = prefer(pool, lambda lid: score(lid, "proven") >= len(job.musts) - plain)
        if not by_price:
            pool = prefer(pool, lambda lid: score(lid, "exact") >= plain)
        if job.priority == "best":
            top = self._top_by_measure(pool)
            if top is not None:
                return top
        return min(pool, key=lambda lid: self._cheap_key(lid, score(lid, "stated") < plain))

    def writes_values(self, pid: str) -> bool:
        """Every wanted value (bounds and exclusions aside) has all its words in one of the listing's own property
        values, its section or its title."""
        info = self.job._pc_card(pid)
        texts = [_pdd_bits(v) for v in (info.get("traits") or {}).values()]
        texts += [_pdd_bits(info.get("title")), _pdd_bits(info.get("category"))]
        for need in self.job.musts:
            if need.get("not") or need.get("rel"):
                continue
            wanted = _pdd_bits(need["value"])
            if wanted and not any(wanted <= text for text in texts):
                return False
        return True

    def closest(self) -> str | None:
        """The opened, buyable listing of the right kind that no requirement ruled out, one writing every wanted
        value out first, then most requirements shown and as the pick would rank it; None when every judged listing
        missed something outright."""
        job = self.job
        near = [lid for lid, verdict in job.verdicts.items()
                if verdict.get("kind") and "no" not in verdict.get("by", ()) and job.buyable(lid) and lid in job.opened]
        if not near:
            return None
        near = [lid for lid in near if self.writes_values(lid)] or near     # as the pick: values written out first
        shown = lambda lid: sum(mark == "yes" for mark in job.verdicts[lid]["by"])
        return min(near, key=lambda lid: (-shown(lid), self._cheap_key(lid, False)))

    def _held_up(self) -> list[str]:
        job = self.job
        return [lid for lid, verdict in job.verdicts.items()
                if verdict.get("kind") and job.buyable(lid) and lid in job.opened]

    def _slot_shows(self, pid: str, slot: int, mark: str) -> bool:
        """A worded requirement (no figure, not an exclusion) whose every word sits in one of the listing's own
        entries for that property counts as shown, whatever the judge marked; any other needs the judge's yes."""
        job = self.job
        if mark == "yes":
            return True
        need = job.musts[slot] if slot < len(job.musts) else None
        if need is None or need.get("not") or need.get("rel") or _PDD_DIGIT.search(str(need["value"])):
            return False
        wanted = set(stems(need["value"]))
        facts = job._pc_card(pid).get("traits") or {}
        return bool(wanted) and any(wanted <= set(stems(v)) for _, v in _pc_own_entries(facts, need["attr"], job.item))

    def written_fit(self) -> str | None:
        """Before a decline: an opened, buyable listing of the right kind that writes every requirement in its own
        entries (worded values) or got the judge's yes (the rest), cheapest first; None when there is none."""
        fit = [lid for lid in self._held_up()
               if all(self._slot_shows(lid, n, mark) for n, mark in enumerate(self.job.verdicts[lid].get("by", ())))]
        return min(fit, key=lambda lid: self._cheap_key(lid, False)) if fit else None

    def least_missed(self) -> str | None:
        """After a decline was turned down: the opened, buyable listing of the right kind, one writing every wanted
        value out first, that misses the fewest requirements outright, then shows the most; None when none was
        judged."""
        job = self.job
        pool = self._held_up()
        if not pool:
            return None
        pool = [lid for lid in pool if self.writes_values(lid)] or pool     # as the pick: values written out first
        count = lambda lid, mark: sum(m == mark for m in job.verdicts[lid].get("by", ()))
        return min(pool, key=lambda lid: (count(lid, "no"), -count(lid, "yes"), self._cheap_key(lid, False)))

    def cheaper_unopened(self, pick: str | None) -> list[str]:
        """Unopened rows in open order that could still beat the pick (only cheaper ones when price decides)."""
        queue = self.job.open_order()
        if pick is None or self.job.priority != "cheapest":
            return queue
        bar = self.job._pc_card(pick).get("price")
        if not isinstance(bar, float):
            return []
        cheaper = []
        for lid in queue:
            cost = self._cost(lid)
            if cost is not None and cost < bar:
                cheaper.append(lid)
        return cheaper


class _PddTalk(_PddService):
    """Conversation service: questions to the shopper, their answers, and notes that ride on replies."""

    def __init__(self, job: Mission) -> None:
        super().__init__(job)
        self.resend: set = set()       # topics to ask again in sharper words
        self.resplit: set = set()      # topics already re-asked once
        self.restate = False           # an answer added a need: row marks must be redone

    def flush(self) -> dict:
        """Read the queued notes (if any) and clear the queue."""
        queued, self.job.pending_notes = self.job.pending_notes, []
        return self.job.hear(queued) if queued else _pdd_quiet()

    def _left_to_send(self) -> bool:
        return self.job.messages_left > 0 and not self.job.no_messages

    def can_ask(self) -> bool:
        return self._left_to_send() and self._can("message")

    def ready_to_ask(self) -> bool:
        """Questions start only when messages remain and listings were opened (or none can be opened at all)."""
        job = self.job
        nothing_seen = not job.opened and (self._can("open_listing") or self._can("open_many"))
        return job.messages_left > 0 and self._can("message") and not nothing_seen

    def ask_one(self) -> Any:
        """Send the next question. None: questions are over; False: topic skipped (asked before); else (topic, obs)."""
        job = self.job
        if not (self.can_ask() and job.topics and job.room(job.kept_messages + 1)):
            return None
        if job.hand_on and any(rq.get("from") == "told" for rq in job.musts):
            return None                   # this product's need is out: its spare messages go to the products after it
        topic = job.topics.pop(0)
        repeat = topic in self.resend
        if not repeat:
            if norm(topic) in {norm(x) for x in job.asked}:
                return False
            job.asked.append(topic)
        self.resend.discard(topic)
        wording = job.ask_text(topic, sharper=repeat)
        reply = job.act("message", text_arg=wording)
        job.messages_left -= 1
        return topic, reply

    def _worth_resend(self, topic: str) -> bool:
        """Re-ask a split reply once, and only when sharper words exist (the same words would split again)."""
        job = self.job
        if topic in self.resplit or not job.two_sided(topic):
            return False
        return job.ask_text(topic, sharper=True) != job.ask_text(topic)

    def take_answer(self, topic: str, reply: Any) -> bool:
        """Apply an answer; False when the world refused the message and no more may be sent."""
        job = self.job
        if error_of(reply):
            job.no_messages = True
            return False
        answer, side = _pdd_split_reply(reply)
        reading = job.hear(answer, asked=topic)
        verdict_class = reading.get("reply_class")
        if verdict_class == "split" and self._worth_resend(topic):
            self.resplit.add(topic)
            self.resend.add(topic)
            job.topics.insert(0, topic)       # ahead of every other topic
        if side:
            job.hear(side)                    # side lines are read as if they came unprompted
        job.ask_log.append((label_words(topic) or topic, verdict_class or "other"))
        news = reading.get("revealed") or []
        if news:
            self.restate = True
            if self._left_to_send() and topic not in self.resend:
                job.choose_topics()           # a new need changes which property is worth asking next
        for attr in news:
            job.topics = [t for t in job.topics if not job._determined_by(t, attr)]
        return True

    def question_loop(self, gate: Any) -> None:
        """Ask until nothing is left to ask; gate runs before every question and every answer."""
        while True:
            gate(True)
            sent = self.ask_one()
            if sent is None:
                return
            if sent is False:
                continue
            gate(False)
            if not self.take_answer(*sent):
                return

    def confirm(self, pick: str) -> dict | None:
        """The pre-order word with the shopper when the request keeps messages for it. Returns the reading only
        when it changed something or ruled the pick out; None otherwise."""
        job = self.job
        if job.kept_messages < 1 or job.no_messages or not self._can("message"):
            return None
        job.kept_messages -= 1
        name = str(job._pc_card(pick).get("title") or "the item I picked")[:90]
        back = job.act("message", text_arg="I have picked %s for you. Shall I place the order?" % name)
        heard: list[str] = []
        if back is not None and not error_of(back):
            heard = sum(_pdd_split_reply(back), [])
        outcome = job.hear(heard) if heard else _pdd_quiet()
        return outcome if (outcome["changed"] or outcome["unavailable"]) else None


class _PddFinder(_PddService):
    """Search service: first look, told-value searches, more pages and wider looks when nothing fits."""

    def __init__(self, job: Mission, ranker: _PddRanker) -> None:
        super().__init__(job)
        self.ranker = ranker
        self.read_out: set = set()     # sources the reach has read to their end (or could not read on)
        self.shelf: list | None = None  # the wanted section's listing as a reach source: [name, token, pages read]

    def first_look(self) -> None:
        self.job.look_round()
        self.job.screen_rows()

    def _fresh(self, query: str) -> bool:
        return _pdd_first_page_tag(query) not in self.job.done_queries

    def told_now(self) -> None:
        """Search right away for every value the shopper named, while steps allow."""
        job = self.job
        for query in list(job.told_queries):
            if job.free_steps() > job.open_min and self._fresh(query):
                job.search(query)

    def _queries(self) -> list[str]:
        return list(filter(None, self.job.searches + [self.job.item]))

    def _page_state(self, query: str) -> tuple:
        key = norm(query)
        token, read = self.job.next_pages.get(key, (None, 0))
        return key, token, read

    def page_further(self) -> bool:
        """Read one more page past the plan (told values first) while two steps fit above the reserve. A query
        whose page held nothing new of the wanted kind is not paged again."""
        job = self.job
        if not job.room(2):
            return False
        for query in job.told_queries + self._queries():
            key, token, read = self._page_state(query)
            if token is None or key in job.paged_out:
                continue
            before = set(job.rows)
            got = job.search(query, _pdd_page_ref(token, read, job.page_numeric()))[0]
            if not got:
                job.paged_out.add(key)
                continue
            job.screen_rows()
            useful = [x for x in got if x not in before and job.screen.get(x, ("no",))[0] != "no" and job.buyable(x)]
            if not useful:
                job.paged_out.add(key)
            return True
        return False

    # ---- reach: the lowest price wanted and a pick in hand, the spare steps read on for a cheaper fit
    def reach(self, pick: str | None) -> bool:
        """Read one more page of the source most likely to hold a cheaper fit than the pick, while steps beyond the
        end reserve allow. Rows are listed in no price order, so only reading on finds the lowest price; rows under
        the pick's price are the only ones screened, so a page bringing none costs no model call. False when no
        step is spare or nothing is left to read."""
        job = self.job
        bar = None if pick is None else self._cost(pick)
        if job.priority != "cheapest" or bar is None or job.free_steps() <= 3:
            return False
        while True:
            source = self._reach_source()
            if source is None:
                return False
            before, step = set(job.rows), job.step
            got = self._read_on(source)
            if job.step == step:
                self.read_out.add(source)          # no step was taken: this source cannot be read on
                continue
            cheaper = [lid for lid in got if lid not in before and job.buyable(lid)
                       and self._cost(lid) is not None and self._cost(lid) < bar]
            if cheaper:
                job.screen_rows(only=set(cheaper))
            return True

    def _sound(self, text: str) -> bool:
        """A search whose every word names the product or a value still wanted covers all listings that fit;
        one built on a value since dropped covers fewer."""
        job = self.job
        allowed = set(stems(job.item))
        for need in job.musts:
            if not need.get("not"):
                allowed.update(stems(need["value"]))
        return set(stems(text)) <= allowed

    def _left(self, total: Any, read: int) -> float:
        rows = self.job.page_rows
        if not (isinstance(total, (int, float)) and rows):
            return float("inf")
        return max(0.0, -(-(float(total) - read * rows) // rows))

    def _reach_source(self) -> Any:
        """The open source with the fewest pages left: a sound search that listed rows of the wanted kind, else the
        wanted section's own listing. None when every source is read out."""
        job = self.job
        best, least = None, None
        for key, (token, read) in job.next_pages.items():
            if token is None or key in self.read_out or not self._sound(key):
                continue
            listed = job.query_rows.get(key) or []
            if not any(job.screen.get(lid, ("maybe", ""))[0] != "no" for lid in listed):
                continue
            left = self._left(job.query_totals.get(key), read)
            if least is None or left < least:
                best, least = key, left
        if best is not None:
            return best
        if self.shelf is None:
            self.shelf = [self._reach_shelf(), None, 0]
        if self.shelf[0] is None or "shelf" in self.read_out or not self._can("filter_search"):
            return None
        return "shelf"

    def _reach_shelf(self) -> str | None:
        """The section the request names, else the one most screened-yes rows sit in."""
        job = self.job
        votes: dict = {}
        for lid, (kind, _) in job.screen.items():
            section = (job.rows.get(lid) or {}).get("category")
            if section and kind == "yes":
                _pdd_bump(votes, section)
        named = job._pc_named_shelves()
        if named:
            votes = {s: votes.get(s, 0) + 1 for s in named}
        return max(votes, key=votes.get) if votes else None

    def _read_on(self, source: Any) -> list[str]:
        job = self.job
        if source != "shelf":
            token, read = job.next_pages[source]
            return job.search(source, _pdd_page_ref(token, read, job.page_numeric()))[0]
        name, token, read = self.shelf
        if read and token is None:
            return []
        ids, token, _total = job.browse(name, job.max_price, None if not read else _pdd_page_ref(token, read, job.page_numeric("filter_search")))
        self.shelf = [name, token, read + 1]
        if token is None:
            self.read_out.add("shelf")
        return ids

    def beat_backup(self) -> None:
        """Cheapest wanted and the pick is gone: sweep the shelf below the next fit's price."""
        job = self.job
        fallback = self.ranker.pick()
        if job.priority != "cheapest" or fallback is None:
            return
        if not job.sweep_shelf or not self._params("filter_search").get("price_arg"):
            return
        start = self._cost(fallback)
        if start is not None and job.room(4):
            job.cheap_sweep(job.sweep_shelf, start=start, floor=job.sweep_cap)
            job.screen_rows()

    # one wider look when nothing fits: the first move that applies runs
    def widen(self) -> bool:
        """One more look for the round: the first source that lists rows not seen before; a look that brings none
        gives way to the next while free steps remain. True when anything was looked at."""
        job, looked = self.job, False
        for move in (self._told_search, self._price_band, self._unrun_search, self._planned_page, self._whole_shelf):
            while job.free_steps() > 0:
                before = len(job.rows)
                if not move():
                    break
                looked = True
                job.screen_rows()
                if len(job.rows) > before:
                    return True
        return looked

    def _search_first_unrun(self, queries: list[str]) -> bool:
        for query in queries:
            if self._fresh(query):
                self.job.search(query)
                return True
        return False

    def _told_search(self) -> bool:
        return self._search_first_unrun(self.job.told_queries)

    def _unrun_search(self) -> bool:
        return self._search_first_unrun(self._queries())

    def _price_band(self) -> bool:
        """Cheapest wanted: double the sweep cap (never past the shopper's limit) and list the shelf again."""
        job = self.job
        cap, limit = job.sweep_cap, job.max_price
        if job.priority != "cheapest" or cap is None or not job.sweep_shelf:
            return False
        if limit is not None and cap >= limit:
            return False
        job.sweep_cap = cap * 2 if limit is None else min(limit, cap * 2)
        job.browse(job.sweep_shelf, job.sweep_cap)
        return True

    def _planned_page(self) -> bool:
        job = self.job
        for query in self._queries():
            key, token, read = self._page_state(query)
            if token is None or read >= job.pages_for(job.query_totals.get(key), query):
                continue
            ref = _pdd_page_ref(token, read, job.page_numeric())
            if "%s|%s" % (key, ref) not in job.done_queries and self._params("keyword_search").get("page_arg"):
                job.search(query, ref)
                return True
        return False

    def _whole_shelf(self) -> bool:
        job = self.job
        if not (self._can("filter_search") and job.categories):
            return False
        busiest = max(job.categories, key=job.categories.get)
        if _pdd_filter_tag(busiest, job.max_price) in job.done_queries:
            return False
        job.browse(busiest, job.max_price)
        return True


class _PddChecker(_PddService):
    """Verification service: which listings get opened, judging them, and the extra checks before a decline."""

    def __init__(self, job: Mission, ranker: _PddRanker, finder: _PddFinder) -> None:
        super().__init__(job)
        self.ranker = ranker
        self.finder = finder
        self.wide = 0                  # how many listings the first look opens in all
        self.proved = False            # the one re-check before a decline has run

    def first_batch(self, asking: bool) -> None:
        """Open the first listings: with questions to send, a few distinct ones (their datasheets give the topics),
        the rest held back until the questions are done."""
        job = self.job
        floor = job.open_min
        self.wide = max(floor, min(floor * 2, job.free_steps() // 4))
        if asking:
            batch = job.census_rows(min(self.wide, max(3, min(job.ask_after, job.free_steps() - floor))))
        else:
            batch = job.open_order()[:self.wide]
        if job.free_steps() > 0:
            job.open_batch(batch)

    def open_held_back(self) -> None:
        job = self.job
        unopened = [lid for lid in job.open_order() if lid not in job.opened]
        batch = unopened[:max(0, self.wide - len(job.opened))]
        if batch and job.free_steps() > 0:
            job.open_batch(batch)

    def _opened_known(self) -> list[str]:
        return [lid for lid in self.job.opened if lid in self.job.rows]

    def judge_opened(self) -> None:
        self.job.judge(self._opened_known())

    def open_and_judge(self, pids: list[str], room: int | None = None) -> None:
        batch = pids if room is None else pids[:max(1, min(self.job.batch_size(), room))]
        self.job.open_batch(batch)
        self.job.judge(batch)

    def cheaper_step(self) -> bool:
        """Open one batch that could change the pick; with a pick in hand, room for a basket swap stays."""
        pick = self.ranker.pick()
        todo = self.ranker.cheaper_unopened(pick)
        room = self._spare()
        if not todo or room <= (2 if pick else 0):
            return False
        if pick is not None and self.job.priority not in ("cheapest", "best"):
            return False
        self.open_and_judge(todo, room)
        return True

    def wording_step(self) -> tuple[str | None, bool]:
        """The pick writes some value only in other words: open two more rows looking for one that writes all."""
        job = self.job
        pick = self.ranker.pick()
        wanted = (pick is not None and job.priority != "cheapest"
                  and job.verdicts[pick].get("exact", 0) < _pdd_plain_count(job) and job.room(5))
        if not wanted:
            return pick, False
        extra = [lid for lid in job.open_order() if lid != pick][:2]
        if extra:
            self.open_and_judge(extra)
        return pick, bool(extra)

    def second_step(self, pick: str) -> bool:
        """Before the first check-out, secure a second passing listing so a refusal needs no new search."""
        rivals = [lid for lid in self.job.open_order() if lid != pick]
        room = self._spare(2)
        if rivals and room > 0 and len(self.ranker.fits()) < 2:
            self.open_and_judge(rivals, room)
            return True
        return False

    def _wanted_shelf(self) -> str | None:
        """The section most screened-yes rows sit in (else most opened ones); a section the request names wins."""
        job = self.job
        votes: dict = {}
        for lid, (kind, _) in job.screen.items():
            section = job.rows[lid].get("category")
            if section and kind == "yes":
                _pdd_bump(votes, section)
        if not votes:
            for lid in job.opened:
                section = (job.rows.get(lid) or {}).get("category")
                if section:
                    _pdd_bump(votes, section)
        named = job._pc_named_shelves()
        if named:
            votes = {s: votes.get(s, 0) + 1 for s in named}
        return max(votes, key=votes.get) if votes else None

    def decline_proof(self) -> bool:
        """Before the first decline, once: judge every opened listing again under the needs now in force, page
        further, then open one batch of the cheapest unopened rows within the limit on the wanted shelf. Later
        calls page further, else open the next batch of those rows while steps are spare. True when anything ran,
        so a round comes before the decline."""
        if self.proved:
            return self.finder.page_further() or self._open_unseen()
        self.proved = True
        job = self.job
        rejudge = self._opened_known()
        for lid in rejudge:
            job.verdicts.pop(lid, None)
        job.judge(rejudge)
        if self.ranker.pick() is not None or self.finder.page_further():
            return True
        return self._open_unseen() or bool(rejudge)

    def _open_unseen(self) -> bool:
        """One batch of the cheapest listed rows not yet opened (not ruled out by the screen) on the wanted shelf;
        False when none is left or the steps are spoken for."""
        job = self.job
        shelf = self._wanted_shelf()
        unseen = []
        for lid in job.order_seen:
            if lid in job.opened or not job.buyable(lid) or job.screen.get(lid, ("maybe", ""))[0] == "no":
                continue
            if shelf is None or job.rows[lid].get("category") == shelf:
                unseen.append(lid)
        unseen.sort(key=lambda lid: 1e18 if self._cost(lid) is None else self._cost(lid))
        room = self._spare()
        if not unseen or room < 2:
            return False
        self.open_and_judge(unseen, room - 1)
        return True


class _PddCart(_PddService):
    """Cart and finish service: basket lines, check-out, decline, and closing an episode left open."""

    def __init__(self, job: Mission, talk: _PddTalk, ranker: _PddRanker) -> None:
        super().__init__(job)
        self.talk = talk
        self.ranker = ranker

    def _lines(self, role: str):
        return self._params(role).get("lines")

    def add(self, pid: str) -> bool:
        job = self.job
        if not self._can("basket_add"):
            return False
        count = job.quantity
        if self._lines("basket_add"):
            # a tool taking basket lines sets the whole basket: the one held line is the pick
            first = job.act("basket_add", lines_arg=[(pid, count)])
            if first is None or error_of(first) or not self._shows(first, pid):
                return False
            job.absorb(first)
            job.basket.clear()
            _pdd_bump(job.basket, pid, count)
            return True
        first = job.act("basket_add", id_arg=pid, qty_arg=count)
        written = bool(job.tools.roles["basket_add"].get("template"))
        if first is None or error_of(first) or (written and not self._shows(first, pid)):
            return self._refill(pid) and self.add(pid) if written else False
        job.absorb(first)
        if not (job.tools.roles["basket_add"].get("qty_arg") or written):
            for _ in range(count - 1):        # the tool takes no quantity: one more add per further unit
                job.act("basket_add", id_arg=pid)
        _pdd_bump(job.basket, pid, count)
        return True

    def fill(self, pid: str) -> bool:
        """A cart tool the model named but code could not shape a call for: the model writes one call for this
        listing, code checks it against the schema and keeps it as the pattern for the episode (twice at most)."""
        job, tools = self.job, self.job.tools
        plan = tools.unplanned.get("basket_add")
        if tools.has("basket_add") or not plan or job.cart_fills >= 2 or not job.clock.model_room():
            return False
        job.cart_fills += 1
        tname, pins = plan
        sheet = tools._pdb_sheet(tname)
        spec = dict(type="function", function=dict(name=tname, description=tools.tools[tname]["description"],
                                                   parameters=sheet))
        text = _pdb_FILL_HEAD + json.dumps(tools._pdb_cards([tname])) + _pdb_FILL_ASK.format(pid=pid,
                                                                                             count=job.quantity)
        got = job.seer.ask(FILL_CALL_GUIDE, text, spec, "deep", 900, vital=True)
        if not isinstance(got, dict):
            return False
        for pname, val in pins.items():
            got.setdefault(pname, val)
        if _pdb_shape_fault(sheet, got):
            return False
        return tools.take_template("basket_add", tname, pins, got, pid, job.quantity)

    def _refill(self, pid: str) -> bool:
        """The written call was refused or did not land: one fresh call from the model, else the role goes."""
        tools = self.job.tools
        job = tools.roles.get("basket_add") or {}
        tools.drop("basket_add", "its model-written call was refused")
        if job.get("template") and not tools.has("basket_add"):
            tools.unplanned["basket_add"] = (job["tool"], dict(job["fixed"]))
        return self.fill(pid)

    def _shows(self, written: Any, pid: str) -> bool:
        """Whether a basket-lines write took: its own reply naming the pick settles it; otherwise the basket view (when
        there is one) must not contradict it - a view listing other known listings but not the pick, or holding
        only empty lists, means the write did not land."""
        job = self.job
        if pid in json.dumps(written, default=str) or not self._can("basket_view"):
            return True
        view = job.act("basket_view")
        if not isinstance(view, dict) or error_of(view):
            return True
        job.absorb(view)                          # lines riding on the view are heard before any check-out
        shown = json.dumps(view, default=str)
        if pid in shown:
            return True
        others = any(lid in shown for lid in job.rows if lid != pid)
        lists = [v for v in _pdd_leaves(view) if isinstance(v, list)]
        return not (others or (lists and not any(lists)))

    def remove(self, pid: str) -> None:
        job = self.job
        role = "basket_remove" if self._lines("basket_remove") else "basket_add"
        if pid in job.basket and self._lines(role):
            rest = [(lid, n) for lid, n in job.basket.items() if lid != pid]
            if rest or self._lines_may_empty(role):
                job.act(role, lines_arg=rest)        # the lines written are the basket that stays
            job.basket.pop(pid)
            return
        if pid not in job.basket:
            return
        role = "basket_remove" if self._can("basket_remove") else "basket_add"
        least = self._zero_floor(role)
        if least is not None:
            self._count_out(role, pid, least)
        elif role == "basket_remove":
            job.act("basket_remove", id_arg=pid, qty_arg=job.basket[pid])
        else:
            return                                    # no way to take it out
        job.basket.pop(pid)
        job.dropped.add(pid)

    def _zero_floor(self, role: str) -> Any:
        """When taking a line out goes through the adding tool's own count slot (no value of its own tells the two
        apart), the lowest count that slot takes; None when removing is a job of its own, or the slot takes no zero.
        Without a removal job the schema must say outright that the count goes down to zero."""
        add, drop = self._params("basket_add"), self._params(role)
        slot = drop.get("qty_arg")
        if not slot or (add.get("tool"), add.get("qty_arg")) != (drop.get("tool"), slot):
            return None
        if (add.get("fixed") or {}) != (drop.get("fixed") or {}):
            return None
        least = number((self.job.tools.props(drop["tool"]).get(slot) or {}).get("minimum"))
        if least is None:
            return 0.0 if role == "basket_remove" else None
        return least if least <= 0 else None

    def _count_out(self, role: str, pid: str, least: float) -> None:
        """Send zero for the line; a count that adds to the line leaves it there, so when the reply still lists it
        and the slot takes counts below zero, send the line's count taken away (and from then on only that)."""
        job = self.job
        back = int(max(least, -job.basket[pid]))
        if job.zero_stays and least < 0:
            job.act(role, id_arg=pid, qty_arg=back)  # this world's zero was seen to leave the line: one call
            return
        got = job.act(role, id_arg=pid, qty_arg=0)
        if least >= 0 or not isinstance(got, dict) or error_of(got):
            return
        if pid in {rec["id"] for rec in find_records(got)}:
            job.zero_stays = True
            job.act(role, id_arg=pid, qty_arg=back)

    def _lines_may_empty(self, role: str) -> bool:
        path = self._lines(role)[0]
        spec = self.job.tools.props(self._params(role)["tool"]).get(path[0]) or {}
        for key in path[1:]:
            spec = (spec.get("properties") or {}).get(key) or {}
        return not number(spec.get("minItems"))

    def clear_others(self, pick: str) -> None:
        if self._lines("basket_add"):
            return                                # the next lines write replaces the basket anyway
        for held in [lid for lid in self.job.basket if lid != pick]:
            self.remove(held)

    def _finish(self, role: str, **args: Any) -> tuple[bool, list[str]]:
        """Send an ending. (True, []) when the episode is over; else (False, refusal lines). Notes are queued."""
        job = self.job
        over = lambda: job.finished or job.steps_remaining() <= 0
        if job.finished:
            return True, []
        answer = job.act(role, **args)
        if over():
            return True, []
        lines = refusal_of(answer)
        job.absorb(answer)
        if not lines:
            job._pb_end_missed(role, answer)
            fault = error_of(answer) if role == "checkout" and isinstance(answer, dict) else ""
            if fault:
                # a world that answers a check-out with a fault: the basket is read again, so a pick the cart no
                # longer holds is added anew (the fault text is not read as the shopper's: it may name the call)
                self._resync()
        return False, lines

    def _resync(self) -> None:
        job = self.job
        if not self._can("basket_view") or self._lines("basket_add") or job.steps_remaining() <= 2:
            return
        view = job.act("basket_view")
        if not isinstance(view, dict) or error_of(view):
            return
        job.absorb(view)
        listed = {rec["id"] for rec in find_records(view)}
        if listed or any(isinstance(v, list) for v in view.values()):
            for lid in [lid for lid in job.basket if lid not in listed]:
                job.basket.pop(lid)

    def checkout(self, pid: str) -> tuple[bool, list[str]]:
        return self._finish("checkout", id_arg=pid, qty_arg=self.job.quantity)

    def decline_note(self) -> str:
        """Name the need the judged listings missed most often (first seen wins a tie); never a listing."""
        job = self.job
        needs = job.musts
        misses: dict[int, int] = {}
        for verdict in job.verdicts.values():
            for slot, mark in enumerate(verdict.get("by") or []):
                if mark != "yes":
                    _pdd_bump(misses, slot)
        slot = max(misses, key=misses.get) if misses and needs else len(needs)
        if slot >= len(needs):
            return DECLINE_NOTE
        target = needs[slot]
        cap = job.max_price
        text = _PDD_DECLINE_TEXT.format(
            limit=" within {cap:g}".format(cap=cap) if cap else "",
            need="{} {}{}".format(target["attr"], "other than " if target.get("not") else "", target["value"]))
        return DECLINE_NOTE if _pdd_mentions_listing(text, job.rows) else text

    def decline(self) -> bool:
        """True when the decline ended the episode; a refusal is read and False returned."""
        over, lines = self._finish("decline", text_arg=self.decline_note())
        if not over:
            if lines:
                self.job.hear(lines)
            self.talk.flush()
        return over

    def _can_end(self) -> bool:
        return self._can("checkout") or self._can("decline")

    def _buy_now(self, choice: str) -> bool:
        """One attempt to order the choice; True when the episode ended."""
        job = self.job
        if choice not in job.basket and self._can("basket_add") and not job.tools.roles["checkout"].get("id_arg"):
            self.add(choice)
        ended, lines = self.checkout(choice)
        if ended:
            return True
        if lines:
            job.hear(lines)
        self.talk.flush()
        if choice in job.basket:
            job.recheck(choice)
        return False

    def rescue(self) -> None:
        """Close an episode still open after the coordinator (a fault, or the plan ran out): buy the judged pick
        when it passes, else decline; up to three tries while steps and time remain. An ending role is never
        probed: when none is mapped the roles are repaired once per try, and nothing is sent if still none."""
        job = self.job
        for _ in range(3):
            if job.finished or job.steps_remaining() <= 0 or job.clock.left() < FINISH_LEFT:
                return
            if not self._can_end():
                job.repair_roles()
                if not self._can_end():
                    return
            choice = self.ranker.pick() if job.verdicts else None
            held = [lid for lid in job.basket if lid not in job.struck]
            if choice and self._can("checkout") and job.buyable(choice):
                if self._buy_now(choice):
                    return
            elif held and len(held) == len(job.basket) and self._can("checkout") and not job.rescued_cart:
                # the order's lines are written and none was taken off sale: they are bought, not given up
                job.rescued_cart = True
                if self.checkout(held[0])[0]:
                    return
            elif self._can("decline") and self.decline():
                return


def _pdd_fact_value(fact: str) -> str:
    """The value part of a fact written "label: value" (the text after its first colon and space; a colon inside
    a figure such as a ratio stays); the fact itself else."""
    head, sep, tail = str(fact).partition(": ")
    return (tail if sep and tail.strip() and head.strip() else str(fact)).strip()


class _PddReferee(_PddService):
    """Requirements whose value the request keeps on another listing: find that listing, follow where its record
    points (a chain of a few listings at most) and put the value it holds in the requirement's place. A value never
    found leaves nothing to rule on, so that requirement is dropped."""

    HOPS, PAGES, SHARE = 6, 3, 0.45   # listings followed per requirement; pages read per search; step share

    def settle(self) -> None:
        if self.job.pointers:
            self.resolve(self.job.pointers[:2])

    def resolve(self, pointers: list[dict]) -> None:
        """Follow each pointer within a share of the steps left, then put the value found in place."""
        job = self.job
        if not (self._can("keyword_search") and (self._can("open_listing") or self._can("open_many"))):
            return
        self.stop_at = job.step + max(6, int(self.SHARE * job.free_steps()))
        for ptr in pointers:
            self._place(ptr, self._follow(ptr))

    def _room(self) -> bool:
        return self.job.step < self.stop_at and self.job.room(3)

    def _follow(self, ptr: dict) -> str:
        self.path = []
        return self._hop(ptr, str(ptr["attribute"]), set(), self.HOPS)

    def _hop(self, ptr: dict, attr: str, tried: set, left: int) -> str:
        """The value at the end of the chain from this description. The records that may be the listing are tried
        best first; one whose pointer leads nowhere gives way to the next (a near twin can point elsewhere)."""
        if left <= 0:
            return ""
        for got in self._candidates(ptr, attr, tried)[:3]:
            if got.get("value"):
                self.path.append(got["id"])
                return str(got["value"])
            if got.get("shop") and self._room():
                self.path.append(got["id"])
                found = self._hop(dict(got, attribute=attr), attr, tried, left - 1)
                if found:
                    return found
                self.path.pop()
        return ""

    def _candidates(self, ptr: dict, attr: str, tried: set) -> list[dict]:
        """Readings of the records that may be the described listing, most facts agreeing first; reading stops
        at the first page holding a record that agrees on every fact."""
        seen: list[dict] = []
        for spare in (False, True):           # the shop's own searches; one fact at a time only when they gave none
            if spare and seen:
                break
            for ids in self._finds(ptr, spare):
                fresh = [x for x in ids if x not in tried]
                seen += self._read(fresh, attr, ptr, tried) if fresh else []
                if any(g["whole"] for g in seen):
                    break
        return sorted(seen, key=lambda g: -g["score"])

    def _finds(self, ptr: dict, spare: bool = False):
        """Pages of ids from searches for the shop alone, then for the shop with the facts' values (spare: for the
        shop with one fact's value at a time); the rows that share most with the facts come first on each page. A
        fact written as "label: value" searches by its value, so a long label never pushes the value past the
        search box's length."""
        job = self.job
        shop = str(ptr.get("shop") or "").strip()
        facts = [str(d).strip() for d in ptr.get("details") or [] if str(d).strip()]
        marks = [m for m in [stems(shop)] + [stems(f) for f in facts] if m]
        values = [v for v in (_pdd_fact_value(f) for f in facts) if v]
        queries = [shop + " " + v for v in values] if spare else [shop, " ".join([shop] + values)]
        if spare and len(values) < 2:
            return

        def fit(lid):
            card = job._pc_card(lid)
            bag = set(stems(" ".join(str(v) for k, v in card.items() if k != "traits")))
            return -sum(len(bag & set(m)) / len(m) for m in marks)
        for query in dict.fromkeys(q.strip() for q in queries if q.strip()):
            page = None
            for _ in range(self.PAGES):
                if not self._room():
                    return
                ids, token, _total = job.search(query, page)
                if not ids:
                    break
                yield sorted(dict.fromkeys(ids), key=fit)
                if token is None:
                    break
                page = _pdd_page_ref(token, job.next_pages.get(norm(query), (None, 1))[1], job.page_numeric())

    def _opened(self, ids: list[str]):
        """Raw records of the listings, one step's worth at a time: multi-open steps of the widest size that works
        when the store has one, else one listing per step, while the share of steps lasts."""
        job, left = self.job, list(ids)
        width = min(len(left), job.many_cap or len(left)) if self._can("open_many") and job.many_cap != 1 else 1
        while left and self._room():
            if width > 1 and len(left) > 1:
                part = left[:width]
                ans = job.act("open_many", ids_arg=part)
                got = {} if ans is None or ans is _PB_LOST else _pdd_raw_records(ans, part)
                if not got:
                    width //= 2
                    continue
                job.many_cap = job.many_cap or width
            elif self._can("open_listing"):
                part = left[:1]
                ans = job.act("open_listing", id_arg=part[0])
                bad = ans is None or ans is _PB_LOST or error_of(ans, need_data=True)
                got = {} if bad else (_pdd_raw_records(ans, part) or {part[0]: ans})
            else:
                return
            left = left[len(part):]
            job.absorb(ans)
            for lid, raw in got.items():
                job.opened.add(lid)
                for rec in _pc_listing_records(lid, raw):
                    job.merge(rec)
            if got:
                yield got

    def _read(self, ids: list[str], attr: str, ptr: dict, tried: set) -> list[dict]:
        """Readings of the records that may be the wanted listing, judged by code from the model's per-fact check:
        the seller agrees, no fact disagrees and at least one agrees. Each carries
        its value, or the listing its own words point to, only when the record's text bears it out. Records go to
        the model a step's worth at a time; a batch holding a record that agrees on every fact ends the reading."""
        details = [str(d) for d in ptr.get("details") or [] if str(d).strip()]
        wanted = "seller or brand: %s; sort of product: %s; facts: %s" % (
            ptr.get("shop") or "?", ptr.get("kind") or "?", "; ".join(details) or "none")
        if ptr.get("onward"):
            wanted += '; named by another listing as: "%s"' % ptr["onward"]
        kept: list[dict] = []
        for records in self._opened(ids):
            tried.update(records)
            raws = list(records.values())
            said = self._ask(attr, wanted, raws)
            for entry in (said or {}).get("records") or []:
                num = entry.get("num") if isinstance(entry, dict) else None
                if not (isinstance(num, int) and 0 <= num < len(raws)):
                    continue
                got = self._judged(entry, raws[num], ptr, details)
                if got is not None and len(raws) > 1 and not (got.get("own") or got.get("details")):
                    alone = next(iter((self._ask(attr, wanted, raws[num:num + 1]) or {}).get("records") or []), None)
                    if isinstance(alone, dict):
                        got = self._judged(alone, raws[num], ptr, details)
                if got is not None and (got.get("value") or got.get("shop")):
                    kept.append(got)
            if any(g["whole"] for g in kept):
                break
        return kept

    def _ask(self, attr: str, wanted: str, raws: list):
        listed = "\n\n".join("[%d] %s" % (n, json.dumps(raw)[:2500]) for n, raw in enumerate(raws))
        text = "Property: %s\nListing wanted: %s\n\nRecords:\n%s" % (attr, wanted, listed)
        return self.job.seer.ask(REFER_GUIDE, text, REFER_SPEC, "deep", 400 + 160 * len(raws), apart=True)

    def _judged(self, entry: dict, raw: Any, ptr: dict, details: list[str]):
        """One record's reading, kept only when it is the wanted listing and what it reports is in its own text."""
        body = json.dumps(raw, ensure_ascii=False)
        bag = set(stems(body))
        shop, attr = str(ptr.get("shop") or ""), str(ptr["attribute"])
        keyed = set(stems(" ".join(label_words(k) for k in _pdd_keys(raw))))
        checks = [c for c in entry.get("facts") or [] if isinstance(c, dict)][:len(details)]
        said = [self._agrees(c, details[n], keyed) for n, c in enumerate(checks)]
        if entry.get("seller") != "yes" or entry.get("sort") == "no" or "no" in said:
            return None
        if not set(stems(shop)) <= bag or (details and said.count("yes") < 1):
            return None
        out = dict(score=said.count("yes") - said.count("absent"), whole=bool(said) and set(said) == {"yes"},
                   id=_pda_id_of(raw) if isinstance(raw, dict) else None)
        onward = str(entry.get("onward") or "").strip()
        told = set(stems(onward))
        if not (told and _pdd_quoted(onward, body)):
            own = self._own_value(raw, attr)
            value = str(entry.get("value") or "").strip()
            spoken = set(stems(label_words(attr))) <= bag and _pdd_quoted(value, " ".join(_pdd_texts(raw)))
            return dict(out, own=bool(own), value=own or (value if value and spoken else ""))
        named = str(entry.get("shop") or "").strip()
        facts = [str(d).strip() for d in entry.get("details") or [] if str(d).strip() and set(stems(d)) <= told]
        echo = norm(named) == norm(shop) and set(map(norm, facts)) <= set(map(norm, details))
        if not named or not set(stems(named)) <= told or echo:
            return dict(out, value="")
        kind = str(entry.get("kind") or "").strip()
        return dict(out, shop=named, kind=kind if set(stems(kind)) <= told else "", details=facts, onward=onward)

    @staticmethod
    def _agrees(check: dict, fact: str, keyed: set) -> str:
        """The model's yes/no/absent on one fact. A yes counts only when the words it quotes share a word with the
        fact's value (the fact's words that name no field of the record); otherwise it counts as absent."""
        agrees, says = str(check.get("agrees") or ""), str(check.get("says") or "")
        if agrees == "yes" and fact:
            value = set(stems(fact)) - keyed or set(stems(fact))
            if not set(stems(says)) & value:
                return "absent"
        return agrees

    def _own_value(self, raw: Any, attr: str) -> str:
        """The record's value for the property under a field of that name, read by code; "" when there is none."""
        found = []
        for rec in _pc_listing_records(str(_pda_id_of(raw) if isinstance(raw, dict) else ""), raw):
            sheet = rec.get("traits") if isinstance(rec.get("traits"), dict) else {}
            for key, val in sheet.items():
                if Mission._same_attr(str(key), attr) and isinstance(val, (str, int, float)) and str(val).strip():
                    found.append((norm(label_words(key)) != norm(label_words(attr)), str(val).strip()))
        return min(found)[1] if found else ""

    def _place(self, ptr: dict, value: str) -> None:
        """Swap the requirement the pointer stands for (same or overlapping property) for the value found."""
        job = self.job
        attr = str(ptr["attribute"])
        near = [n for n in job.musts if not n.get("not") and job._overlaps(n["attr"], attr)]
        same = [n for n in near if norm(label_words(n["attr"])) == norm(label_words(attr))]
        for n in same or sorted(near, key=lambda n: -len(words(n["value"])))[:1]:
            job.musts.remove(n)
        if not value:
            return
        if ptr.get("differs") is True:
            # the listing's value is the one the pick must not have
            job.musts.append({"attr": attr, "value": value, "not": True, "from": "brief"})
            return
        lead = _PDD_LEAD_FIGURE.search(norm(value))
        margin = number(ptr.get("margin"))
        if margin and lead:
            figure, unit = float(lead.group(1).replace(",", "")), lead.group(2) or str(ptr.get("unit") or "")
            for rel, at in (("at least", figure - margin), ("at most", figure + margin)):
                job.musts.append({"attr": attr, "value": ("%g %s" % (at, unit)).strip(), "rel": rel, "from": "brief"})
            return
        job.musts.append({"attr": attr, "value": value, "from": "brief"})
        if 0 < len(words(value)) <= 3:
            job.searches.insert(0, "%s %s" % (value, job.item) if job.item else value)


class _PddKit:
    """The services of one mission, wired together."""

    def __init__(self, job: Mission) -> None:
        self.ranker = _PddRanker(job)
        self.talk = _PddTalk(job)
        self.finder = _PddFinder(job, self.ranker)
        self.checker = _PddChecker(job, self.ranker, self.finder)
        self.cart = _PddCart(job, self.talk, self.ranker)
        self.referee = _PddReferee(job)


# ---------------------------------------------------------------------- coordinator
class _PddHalt(Exception):
    """The episode is over: nothing more runs."""


class _PddRestart(Exception):
    """Time is short or the world is down: optional work is dropped and a fresh round starts."""


class _PddCoordinator:
    """Calls the services in order: open, ask, then rounds of judge / look more / decide until the episode ends.
    Before each piece of work a gate stops at a finished episode and, for optional work, restarts the round when
    the clock is short or the world is down."""

    def __init__(self, job: Mission) -> None:
        self.job = job
        kit = _PddKit(job)
        self.ranker, self.talk, self.finder = kit.ranker, kit.talk, kit.finder
        self.checker, self.cart, self.referee = kit.checker, kit.cart, kit.referee
        self.refusals = 0              # refused endings (check-out or decline) so far
        self.decline_refused = False   # the world turned a decline down: it expects a purchase
        self.checkout_refused = False  # the world turned a check-out down, or took the held pick off sale
        self.tried_written: set = set()  # listings ordered because their own entries carry every worded need

    def short(self) -> bool:
        return self.job.clock.hurry() or self.job.world_down

    def gate(self, optional: bool = True) -> None:
        if self.job.finished:
            raise _PddHalt()
        if optional and self.short():
            raise _PddRestart()

    def run(self) -> None:
        try:
            self._opening()
        except _PddRestart:
            pass
        except _PddHalt:
            return
        while True:
            try:
                if self._round():
                    return
            except _PddRestart:
                continue
            except _PddHalt:
                return

    # ---------------------------------------------- opening
    def _opening(self) -> None:
        job = self.job
        self.gate(False)
        job.map_roles()
        job.read_request()
        self.gate()
        if _PddBasket.suits(job):
            _PddBasket(job).run()
            raise _PddHalt()
        self.referee.settle()
        self.gate()
        self.finder.first_look()
        self.gate()
        self.checker.first_batch(asking=self.talk.can_ask())
        self.gate()
        if self.talk.ready_to_ask():
            job.choose_topics()
            self.talk.question_loop(self.gate)
        self.gate()
        self._after_questions()

    def _after_questions(self) -> None:
        job = self.job
        self.talk.flush()
        if self.talk.restate:
            job.screen_rows(restate=True)     # marks from before the answer: held-back rows are chosen again
            self.talk.restate = False
        self.checker.open_held_back()
        self.finder.told_now()
        job.screen_rows()

    # ---------------------------------------------- end game; True means the episode stops
    def _round(self) -> bool:
        job = self.job
        self.gate(False)
        if job.finished or job.steps_remaining() <= 0:
            return True
        self.checker.judge_opened()
        if not self.short() and (job.cap_rose or not (self.refusals and self.ranker.pick() is not None)):
            job.cap_rose = False              # a raised limit opens dearer listings: look before ordering again
            self._look_more()                 # after a refused ending a ready pick goes straight back to the end
        self.gate(False)
        pick = self.ranker.pick()
        if pick is None:
            return self._nothing_fits()
        return self._pick_ready(pick)

    def _look_more(self) -> None:
        ck = self.checker
        while True:
            self.gate()
            if not ck.cheaper_step():
                break
        while not self.refusals:
            self.gate()
            if not self.finder.reach(self.ranker.pick()):
                break
            while True:
                self.gate()
                if not ck.cheaper_step():
                    break
        while True:
            self.gate()
            pick, opened_more = ck.wording_step()
            if not opened_more:
                break
        if pick is None or self.refusals:
            return
        while True:
            self.gate()
            if not ck.second_step(pick):
                return

    def _nothing_fits(self) -> bool:
        self.gate(False)
        if not self.short() and self.job.room(1) and self.finder.widen():
            return False
        if not self.short() and self.checker.decline_proof():
            return False
        orders = self.job.tools.has("checkout")
        written = self.ranker.written_fit() if orders else None
        if written is not None and written not in self.tried_written:
            # a listing whose own entries carry every worded requirement is no reason to decline; one that only
            # leaves requirements undecided is (a store that has no fit expects a decline, not a substitute)
            self.tried_written.add(written)
            return self._pick_ready(written)
        if self.decline_refused and orders:
            # a decline was already turned down (its line has been heard): order the likeliest judged listing
            # rather than decline again, the one missing least when none is clear of misses
            held = self.ranker.closest() or self.ranker.least_missed()
            if held is not None:
                return self._pick_ready(held)
        if (self.checkout_refused or self.job.seer.dark()) and orders:
            # the store meant to sell (it turned the check-out down, not the request), or no model is left to
            # judge with (code passes only what a listing spells out): a judged listing that no requirement rules
            # out is ordered, once, rather than closing the task empty-handed
            held = self.ranker.closest()
            if held is not None and held not in self.tried_written:
                self.tried_written.add(held)
                return self._pick_ready(held)
        if getattr(self.job, "pending_notes", None) and self.talk.flush().get("changed"):
            return False                      # a line still unread (a need dropped, a new limit) is heard first
        if self.cart.decline():
            return True
        self.decline_refused = True
        self.refusals += 1
        return self.refusals >= 4

    def _pick_ready(self, pick: str) -> bool:
        job = self.job
        self.gate(False)
        self.cart.clear_others(pick)
        if not job.tools.has("checkout"):
            return self._nothing_fits()       # no way to order: the episode still ends, by a decline
        if not job.tools.has("basket_add"):
            self.cart.fill(pick)              # a cart tool code could not shape: the model writes the call once
        if pick in job.basket or not job.tools.has("basket_add"):
            return self._checkout(pick)
        if not self.cart.add(pick):
            job.struck.add(pick)
            return False
        return self._settle(pick, self.talk.flush(), landed=True)

    def _settle(self, pick: str, effect: dict, landed: bool = False) -> bool:
        """Lines arrived while the pick is in the basket, or it has just landed there (a world may move a listing's
        price or stock once it is held): read it again, and go on to check out only when nothing ruled it out and
        it is still the pick."""
        job = self.job
        self.gate(False)
        if (landed or effect.get("heard")) and not effect["unavailable"]:
            if not job.recheck(pick):
                if job.tools.has("open_offer"):
                    self.talk.flush()             # a line on the offer read speaks of this pick, not of the next one
                self.cart.remove(pick)            # the fresh read took the held pick away: the store meant to sell
                self.checkout_refused = True
                return False
            later = self.talk.flush()             # lines on the fresh read
            effect = {key: bool(effect.get(key) or later.get(key)) for key in ("unavailable", "changed")}
            if not effect["unavailable"] and self.ranker.pick() not in (None, pick):
                return False                      # the fresh figures put another listing first
        if effect["unavailable"]:
            job.struck.add(pick)
            self.cart.remove(pick)
            return False
        if effect["changed"]:
            return False
        return self._checkout(pick)

    def _checkout(self, pick: str) -> bool:
        self.gate(False)
        effect = self.talk.confirm(pick)
        if effect is not None:
            return self._settle(pick, effect)
        self.cart.clear_others(pick)          # a line the cart showed that this run never planned goes first
        done, refused = self.cart.checkout(pick)
        return True if done else self._refused(pick, refused)

    def _refused(self, pick: str, refused: list[str]) -> bool:
        job = self.job
        self.gate(False)
        self.refusals += 1
        self.checkout_refused = True
        first = job.hear(refused) if refused else _pdd_quiet()
        later = self.talk.flush()
        if first["unavailable"] or later["unavailable"]:
            job.struck.add(pick)
            self.cart.remove(pick)
            if not self.short():
                self.finder.beat_backup()
        elif pick is not None and pick not in job.struck:
            job.recheck(pick)                 # price or stock may have moved: the next round sees the fresh read
        if self.refusals < 4:
            return False
        self.cart.decline()
        return True


# ---------------------------------------------------------------------- several sorts of product in one order
# mission fields that belong to one product of the order; everything else (steps, cart, messages, tools) is shared
_PDD_PART_FIELDS = ("musts searches told_queries order_seen ask_log asked topics pointers rows screen verdicts "
                    "next_pages query_totals query_rows paged_out opened struck done_queries categories").split()
_PDD_ASKS_PER_SORT = 6   # model requests each further product may add to the run's count
# single-valued mission fields (attribute names, not words to match) that belong to one product of the order
_PDD_PART_SCALARS = ("item", "quantity", "figure_unit", "dead_ends", "kept", "details_checked", "sweep_cap",
                     "sweep_shelf")


class _PddPart(_PddCoordinator):
    """One product of a several-product order: the usual opening and rounds, but a ready pick is held instead of
    bought, and a round with nothing that passes ends the part with the likeliest listing instead of a decline."""

    def __init__(self, job: Mission) -> None:
        super().__init__(job)
        self.held: str | None = None

    def play(self) -> None:
        try:
            self._opening()
        except _PddRestart:
            pass
        while True:
            try:
                if self._round():
                    return
            except _PddRestart:
                continue

    def _opening(self) -> None:
        job = self.job
        self.gate(False)
        self.referee.settle()              # this product's values kept on other listings, before the first look
        self.gate()
        self.finder.first_look()
        self.gate()
        self.checker.first_batch(asking=self.talk.can_ask())
        self.gate()
        if self.talk.ready_to_ask():
            job.choose_topics()
            self.talk.question_loop(self.gate)
        self.gate()
        self._after_questions()

    def _nothing_fits(self) -> bool:
        self.gate(False)
        if not self.short() and self.job.room(1) and self.finder.widen():
            return False
        self.held = self.ranker.closest() or self.ranker.written_fit() or self.ranker.least_missed()
        return True

    def _pick_ready(self, pick: str) -> bool:
        self.held = pick
        return True


class _PddBasket(_PddService):
    """A request for several sorts of product: each sort gets its share of the steps and messages and is found,
    asked about and judged on its own; then the cart is written with every line at once and the order placed."""

    @staticmethod
    def suits(job: Mission) -> bool:
        roles = job.tools.roles
        return (len(job.products) >= 2 and job.tools.has("basket_add") and job.tools.has("checkout")
                and not (roles.get("checkout") or {}).get("id_arg"))

    def __init__(self, job: Mission) -> None:
        super().__init__(job)
        self.parts: list[dict] = []
        self.talk = _PddTalk(job)

    # ---------------------------------------------- one product at a time
    def run(self) -> None:
        job = self.job
        job.whole_noun = True
        limit, pool = job.max_steps, job.messages_left
        sorts = job.products
        job.seer.more_asks = _PDD_ASKS_PER_SORT * (len(sorts) - 1)   # the spend cap still bounds the run
        job.vital_answers = max(0, pool)     # one answer read per message the order may send
        asked = list(job.pointers)           # values the request keeps on other listings, for the product they name
        for left, sort in zip(range(len(sorts), 0, -1), sorts):
            share = max(0, pool // left + (1 if pool % left else 0)) if pool > 0 else 0
            self._enter(sort, asked)
            job.hand_on = left > 1
            spent = job.seer.spent_so_far()
            job.seer.part_cap = spent + (_pda_SPEND_PER_RUN - spent) / left if left > 1 else None
            job.messages_left = min(share, pool)
            free = max(0, limit - job.step - job.reserve - pool - job.kept_messages)
            job.max_steps = min(limit, job.step + free // left + job.reserve + job.messages_left + job.kept_messages)
            part = _PddPart(job)
            try:
                part.play()
            finally:
                job.max_steps = limit
            pool -= share - max(0, job.messages_left)
            self._leave(part.held)
            if job.finished:
                return
        job.messages_left = max(0, pool)
        self._close()

    def _enter(self, sort: dict, asked: list | None = None) -> None:
        job = self.job
        for name in _PDD_PART_FIELDS:
            setattr(job, name, type(getattr(job, name))())
        for name in _PDD_PART_SCALARS:
            setattr(job, name, _pdb_SCALARS[name])
        intake = _pdb_Intake(job)
        job.item = str(sort.get("item") or "").strip()
        job.quantity = max(1, int(number(sort.get("quantity")) or 1))
        intake._wants(sort)
        # a value kept on another listing goes to the product whose own requirement names that property
        job.pointers = [p for p in asked or [] if any(job._same_attr(str(rq["attr"]), str(p.get("attribute") or ""))
                                                      for rq in job.musts) or self._kind_owner(p) is sort]
        job.searches = [q for q in map(str, sort.get("searches") or []) if q.strip()] or intake.fallback()
        intake.lead_with_values()
        intake.values_first()

    def _kind_owner(self, ptr: dict):
        """The product a pointer names by its sort when no product's own requirements name its property (the
        request withholds that value, so the pointed property may be missing from every product's list)."""
        sorts, attr = self.job.products, str(ptr.get("attribute") or "")
        for s in sorts:
            if any(isinstance(r, dict) and self.job._same_attr(str(r.get("attribute") or ""), attr)
                   for r in s.get("requirements") or []):
                return None
        kind = set(stems(str(ptr.get("kind") or "")))
        shares = [len(kind & set(stems(str(s.get("item") or "")))) for s in sorts]
        best = max(shares or [0])
        return sorts[shares.index(best)] if best and shares.count(best) == 1 else None

    def _leave(self, held: str | None) -> None:
        job = self.job
        state = {name: getattr(job, name) for name in _PDD_PART_FIELDS + list(_PDD_PART_SCALARS)}
        fits = sorted((lid for lid in _PddRanker(job).fits() if self._cost(lid) is not None), key=self._cost)
        self.parts.append(dict(state=state, held=held, fits=fits, qty=job.quantity))

    # ---------------------------------------------- the whole order
    def _merged(self) -> None:
        """Every product's listings at once, for the cart, the fresh reads and the prices."""
        job = self.job
        job.rows, job.struck, job.opened = {}, set(), set()
        for part in self.parts:
            job.rows.update(part["state"]["rows"])
            job.struck |= part["state"]["struck"]
            job.opened |= part["state"]["opened"]
        job.verdicts = {}                         # nothing is bought on its own after this

    def _total(self) -> float:
        return sum((self._cost(p["held"]) or 0.0) * p["qty"] for p in self.parts)

    def _within_budget(self) -> bool:
        """Swap held listings for cheaper passing ones of the same product until the order's sum is under the cap."""
        cap = self.job.max_price
        while cap is not None and self._total() > cap + 1e-9:
            best, gain = None, 0.0
            for part in self.parts:
                now = self._cost(part["held"]) or 0.0
                for lid in part["fits"]:
                    cut = (now - self._cost(lid)) * part["qty"]
                    if lid not in self.job.struck and cut > gain:
                        best, gain = (part, lid), cut
            if best is None:
                return False
            best[0]["held"] = best[1]
        return True

    def _fallback(self, part: dict) -> bool:
        """The held listing is gone: the cheapest other passing listing of that product, if any."""
        job = self.job
        for lid in part["fits"]:
            if lid != part["held"] and lid not in job.struck and job.buyable(lid):
                part["held"] = lid
                return True
        return False

    def _lines(self) -> list[tuple[str, int]]:
        return [(p["held"], p["qty"]) for p in self.parts]

    def _write(self) -> bool:
        """Put every line in the cart; True when the cart (as far as it can be seen) holds them all."""
        job, cart = self.job, _PddCart(self.job, self.talk, _PddRanker(self.job))
        lines = self._lines()
        if self._params("basket_add").get("lines"):
            first = job.act("basket_add", lines_arg=lines)
            if first is None or error_of(first):
                return False
            job.absorb(first)
            job.basket.clear()
            for lid, n in lines:
                _pdd_bump(job.basket, lid, n)
            return self._shown([lid for lid, _ in lines], first)
        for held in [lid for lid in job.basket if lid not in dict(lines)]:
            cart.remove(held)
        for lid, n in lines:
            if job.basket.get(lid) == n:
                continue
            job.quantity = n
            if not cart.add(lid):
                return False
        return True

    def _shown(self, ids: list[str], reply: Any) -> bool:
        job = self.job
        text = json.dumps(reply, default=str)
        if all(lid in text for lid in ids) or not self._can("basket_view"):
            return True
        view = job.act("basket_view")
        if not isinstance(view, dict) or error_of(view):
            return True
        job.absorb(view)
        shown = json.dumps(view, default=str)
        return all(lid in shown for lid in ids)

    def _fresh(self) -> bool:
        """Read each held listing again once the cart is written (a world may move price or stock then); swap out
        any that no longer passes. True when every line still stands."""
        job, steady = self.job, True
        for part in self.parts:
            if job.steps_remaining() <= len(self.parts) + 1:
                break
            job.quantity = part["qty"]
            if not job.recheck(part["held"]):
                job.struck.add(part["held"])
                steady = self._fallback(part) and False
        self.talk.flush()
        return steady

    def _close(self) -> None:
        job = self.job
        cart = _PddCart(job, self.talk, _PddRanker(job))
        self._merged()
        if any(p["held"] is None for p in self.parts) or not self._within_budget():
            cart.decline()
            return
        for _ in range(3):
            if job.finished or job.steps_remaining() <= 1:
                return
            if not self._write():
                part = next((p for p in self.parts if p["held"] not in job.basket), None)
                if part is None or not self._fallback(part):
                    break
                continue
            if not self._fresh():
                if any(p["held"] in job.struck for p in self.parts) or not self._within_budget():
                    break
                continue
            self._confirm()
            for stray in [lid for lid in job.basket if lid not in dict(self._lines())]:
                cart.remove(stray)                # a line the cart showed that the plan does not hold
            done, refused = cart.checkout(self.parts[0]["held"])
            if done:
                return
            if refused:
                job.hear(refused)
            self.talk.flush()
            for part in self.parts:
                job.quantity = part["qty"]
                if not job.recheck(part["held"]):
                    job.struck.add(part["held"])
                    self._fallback(part)
            if not self._within_budget():
                break
        if not job.finished:
            cart.decline()

    def _confirm(self) -> None:
        """The pre-order word with the shopper when the request keeps a message for it, naming every line."""
        job = self.job
        if job.kept_messages < 1 or job.no_messages or not self._can("message"):
            return
        job.kept_messages -= 1
        names = [str(job._pc_card(p["held"]).get("title") or "an item")[:60] for p in self.parts]
        back = job.act("message", text_arg="Ready to place one order with: %s. May I go ahead?" % "; ".join(names))
        if back is not None and not error_of(back):
            job.absorb(back)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        mission = Mission(problem_data)
    except Exception:
        return []                         # a world this code cannot even read: nothing sent, no crash
    try:
        mission.go()
    except Exception:
        pass
    # whatever happened above, an episode is not left open while steps and time remain
    try:
        _PddKit(mission).cart.rescue()
    except Exception:
        pass
    return mission.dialogue


__all__ = ["agent_main"]
