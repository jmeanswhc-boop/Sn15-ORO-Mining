from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import math
import random
import re
import threading
import time
import traceback
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, Literal, NotRequired, Protocol, TypedDict, get_args, get_origin, get_type_hints

from src.agent.proxy_client import ProxyClient


GC_MODEL_LIST: list[str] = [
    "z-ai/glm-5.3",
    "z-ai/glm-5.2",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-v4-flash",
]
GC_PRIMARY_MODEL: str = GC_MODEL_LIST[0]
GC_BUDGET_MODEL: str = GC_MODEL_LIST[0]
GC_FALLBACK_MODELS: tuple = tuple(GC_MODEL_LIST[1:])
GC_MAX_TOKENS: int = 2000
GC_TASK_MAX_TOKENS: int = 2500
GC_DECIDER_MAX_TOKENS: int = 3000
GC_RECHECK_MAX_TOKENS: int = 1500
GC_ROLE_REASONING: str = "low"
GC_ANALYSIS_REASONING: str = "low"
GC_REQUIREMENTS_REASONING: str = "medium"
GC_DECISION_REASONING: str = "medium"
GC_OPENROUTER_TIMEOUT: float = 90.0
GC_OPENROUTER_RETRIES: int = 1
GC_RETRY_BASE_SECONDS: float = 1.5
GC_RETRY_CAP_SECONDS: float = 12.0
GC_RETRY_RATE_LIMIT_FACTOR: float = 3.0
GC_NO_RETRY_STATUSES: frozenset = frozenset({400, 401, 402, 403})
GC_GONE_STATUSES: frozenset = frozenset({403, 404})
GC_TRANSIENT_STATUSES: tuple = (429, 502, 503, 504)
GC_TOTAL_SECONDS: float = 270.0
GC_FINISH_RESERVE_SECONDS: float = 45.0
GC_ENV_TIMEOUT: float = 90.0
GC_HARD_SECONDS: float = 295.0
GC_MIN_REQUEST_SECONDS: float = 10.0
GC_PHRASE_COUNT: int = 8
GC_QUERY_PROBE_STEPS: int = 5
GC_PAGE_STEPS: int = 4
GC_CATEGORY_LIMIT: int = 6
GC_CATEGORY_STEPS: int = 12
GC_CANDIDATE_COUNT: int = 8
GC_QUESTION_LIMIT: int = 3
GC_QUESTION_TEMPLATE: str = (
    "You required product as following:\n"
    "\"{user_request}\"\n"
    "I have got one quick question related your request:\n"
    "\"{question}\""
)
GC_QUESTION_SENTENCES: int = 3
GC_FIELD_VALUE_WORDS: int = 6
GC_FIELD_MIN_SHARE: float = 0.4
GC_FIELD_WEIGHTS: dict[str, float] = {"distinct": 0.85, "in_request": -0.87, "name_words": -1.46, "numeric": 0.32,
                                      "place": -0.37, "share": -0.9, "value_words": 0.78, "varies": -0.9}
GC_USE_DECLINE_AUDIT: bool = True
GC_DEFAULT_MAX_STEPS: int = 38
GC_FINISH_RESERVE_STEPS: int = 6
GC_DECLINE_FINISH_STEPS: int = 3
GC_TARGET_MATCHES: int = 21
GC_TASK_COST_BUDGET: float = 0.035
GC_MAX_ACTIONS: int = 16
GC_LONG_TEXT_MIN_LEN: int = 200
GC_ID_MAX_LEN: int = 128
GC_ROLE_CACHE_VERSION: int = 1
GC_BINDING_VERSION: int = 1
GC_DECLINE_REMARK: str = "None of the available items satisfies all of the requirements, so nothing is ordered."

GC_ROLE_DESCRIPTIONS: dict[str, str] = {
    "search": "locate items through free text, department, maker, price or similar filters; returns a page of items",
    "listing": "show one item in full by its id (title, price, stock, description)",
    "details": "show one item's structured attributes, specifications or properties by its id",
    "cart_add": "put an item into the basket (by id, optional quantity)",
    "cart_remove": "take an item out of the basket",
    "cart_edit": "change the basket; a single tool that adds or removes depending on a parameter",
    "cart_view": "show what is currently in the basket",
    "order": "FINAL: check out to buy (ends the episode)",
    "decline": "FINAL: end the episode with no purchase",
    "message": "send a question or note to the buyer; the buyer's reply is returned as the result",
    "compare": "show attributes for a list of item ids in one call",
    "other": "anything that fits none of the roles above",
}

GC_ROLE_SLOTS: dict[str, tuple] = {
    "search": ("text", "category", "brand", "max_price", "min_price", "page", "cursor", "mode"),
    "listing": ("id",),
    "details": ("id",),
    "cart_add": ("id", "quantity"),
    "cart_remove": ("id", "quantity"),
    "cart_edit": ("id", "quantity", "action"),
    "cart_view": (),
    "order": ("id", "quantity", "finish", "remark"),
    "decline": ("remark", "finish"),
    "message": ("message",),
    "compare": ("ids",),
    "other": (),
}

GV_MEMO_VALUES: dict = {}
GV_MEMO_LOCK = threading.Lock()
GV_RETRY_POLICIES: dict = {}
GV_RETRY_LOCK = threading.Lock()
GV_OPENROUTER_GATEWAYS: dict = {}
GV_GATEWAY_LOCK = threading.Lock()
GV_DEAD_MODELS: set = set()
GV_PLAIN_JSON_MODELS: set = set()
GV_OUTPUT_FIELDS: dict = {}
GV_CALL_OUTCOMES: dict = {}
GV_DEAD_LOCK = threading.Lock()
GV_LLM_FAILURES: list = []
GV_ROLE_MEMORY: dict = {}

RE_NORMALIZE_SPACE = re.compile(r"\s+")
RE_WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
RE_CODE_FENCE = re.compile(r"```(?:json)?")
RE_OPEN_BRACE = re.compile(r"\{")
RE_DIGIT_SEPARATORS = re.compile(r"[,\s]")
RE_NUMBER_TEXT = re.compile(r"-?\d+(?:,\d{3})*(?:[.,]\d+)?")
RE_THOUSANDS_COMMA = re.compile(r",(?=\d{3}\b)")
RE_SENTENCE_BREAK = re.compile(r"(?<=[.?!])\s+")
RE_ATTRIBUTE_SEPARATOR = re.compile(r"[._]+")
RE_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
RE_PARENTHESIS = re.compile(r"\s*\([^)]*\)")
RE_NOTE_PARAM = re.compile(r"explan|remark|note|message|comment|reason|summary|body|text|why", re.I)
RE_FINISH_ORDER = re.compile(r"check|order|buy|purchas|confirm|commit|place|submit|pay", re.I)
RE_FINISH_DECLINE = re.compile(r"cancel|declin|abort|skip|quit|give|none|stop|walk|reject|abandon|no_|^no$", re.I)

GP_TOOL_ROLES = """\
You classify the tools of an online-shopping environment by role. Tool names are arbitrary and may mislead; \
rely on each description and its parameters.

Roles:
{roles}

Rules: assign exactly one role per tool, chosen from its allowed roles. Two tools with identical parameters \
usually serve different roles. A description stating that the call is final or ends the session means order or \
decline.
classifications maps each tool name to its role.
Reply with one JSON object that follows the reply schema."""

GP_SLOT_MAPPER = """\
You map the arguments of online-shop tools to their meanings. Argument names are arbitrary; rely on \
descriptions, types, enums and limits.
Argument meanings per role:
- search: text (free words to match), category (department filter), brand (maker filter), max_price (price \
ceiling), min_price, page (page number), cursor (token from the previous reply that fetches the next page), \
mode (enum choosing free-text versus field search)
- listing/details/cart_add/cart_remove/order: id (item id), quantity
- cart_edit: id, quantity, action (enum choosing add versus remove)
- message: message (text sent to the buyer); order/decline: remark (a note, explanation or summary sent \
with the final call)
- order/decline: finish (enum choosing order versus decline when one tool does both)
- compare: ids (list of item ids)
Also name the enum values: mode_text (search mode for free text), mode_filter (search mode for filters), add \
and remove (basket action values), finish_order and finish_decline (finish values that order or decline). Use \
exact argument names and exact enum values; omit whatever does not exist.
tool_bindings holds one entry per tool: argument_map maps each meaning to the tool's argument name, enum_map \
maps each value name to the tool's enum value.
Reply with one JSON object that follows the reply schema."""

GP_SEARCH_PHRASES = """\
You convert a buyer's brief into keyword queries for a catalogue search engine. The engine returns a listing \
only when EVERY word of a query appears somewhere in that listing (title, description or attribute values).
Guidelines:
- Draw words only from the brief.
- Each query must include the word naming the item type.
- Every other word must be a characteristic the buyer insists on for some attribute, copied from the brief.
- Leave out narrative or situational wording (recipient, occasion, timing, place, motive), budget or cost \
figures, limits on messages or actions, and stock conditions.
- Use 2 to 5 words per query. Begin with the widest query (item type plus one attribute, or a two-word item \
name) and add one insisted-upon attribute at a time.
- Offer at most 8 distinct queries, widest first.
item_noun is the item's noun phrase (1-3 words); keyword_queries are the queries, widest first.
Reply with one JSON object that follows the reply schema."""

GP_PURCHASE_RULES = """\
Read a buyer's brief and extract the purchase rules. Fill the reply fields as follows:
- spend_limit: the overall spending ceiling for the order, as a number in the brief's own unit, or null if \
none is given
- unit_count: how many units the buyer wants in this order (integer). Count only units bought now; people \
mentioned in the narrative and items already owned are not a count; an item sold as a matched pair or set is \
one unit; 1 if not stated
- unit_count_quote: the brief's exact words stating that count, or an empty string
- lowest_price_wanted: true if the buyer wants the lowest price among items that meet the brief
- stock_only: true if the buyer wants only items available right now
- contact_limit: the cap on how many messages the assistant may send to the buyer, or null
- action_limit: the maximum number of steps or actions allowed, or null
- item_kind: the item being bought, 2-4 words
- skip_alternatives: true if the buyer forbids proposing a stand-in item when nothing suits
- ranking_metric: set only if the brief singles out one measurable attribute to maximise or minimise across \
suitable items; give it in listing terms. Otherwise use "", likewise when the extreme wanted is the price
- ranking_descending: true to maximise ranking_metric, false to minimise it
- ranking_quote: the brief's own words asking for that highest or lowest value, or "" when absent
- inspection_count: how many suitable listings the brief asks to be examined in detail before the pick, else \
null
- inspection_quote: the brief's own words giving that count, or "" when absent
- working_notes: verbatim sentences from the brief that direct how the assistant works and are not covered by \
any key above (inspection counts, what to say or withhold, wording of a closing note); an empty list if none. \
Item attributes, availability, money, message or step caps, the cheapest key, the fallback when nothing suits, \
and the buyer's own plans to inspect something are not working notes
- pre_pick_inquiries: questions the assistant must put to the buyer before picking. Each entry has kind, \
choices and inquiry. Choose kind budget solely when the brief states no spending cap and leaves the cap to be \
agreed. Use kind choice only if the brief names two desires and admits one item might not satisfy both, so the \
buyer must rank them; choices then carries those two desires copied exactly. inquiry is one brief sentence to \
the buyer on that point. An open invitation to ask anything is not an entry; in every other case give []
Reply with one JSON object that follows the reply schema."""

GP_REQUIREMENTS = """\
Extract the item criteria from a buyer's brief. Each criterion is one entry of the reply.
Guidelines:
- quote is a short span lifted verbatim from the brief.
- relation says how the item must relate to the value. equals: the value is required. excludes: the buyer \
rules that value out; required_value is the ruled-out value itself, and each ruled-out value is its own entry. \
at_least: the item must reach a level (a range or capability extending up to a figure). at_most: a limit the \
item must stay within.
- required_value holds only the value, as short as a listing would write it: never with a negation or a \
comparison word in front, and without the attribute's own name around it. For at_least and at_most it is the \
number plus its unit, while figure and unit carry them apart; otherwise figure is null.
- options: when the brief accepts any one of several values for this attribute, every accepted value, each on \
its own; otherwise a list with required_value alone.
- criterion_type: product_type for the bare noun naming the item bought, without its attributes (each \
attribute is a separate entry); property for a feature belonging to the item; logistics for delivery, \
shipping, timing, stock or availability and the item's condition; quantity for how many to buy; rule for \
budgets, message or step limits and other purchase rules.
- An item property is mandatory unless it is phrased as a wish: a property the buyer insists on (has to, must \
or should), or will check personally, is mandatory; numeric properties are mandatory.
- A property expressed only as a preference, hope or nice-to-have is not mandatory.
- alternatives: up to 8 other spellings or phrasings a seller might use for this value: synonyms, other \
spellings (British or American, singular or plural), gendered word forms, wording sellers commonly use, the \
figure converted into another unit (figure and unit together), and other notations for dimensions.
- Skip an entry whose value only repeats its attribute's name; it pins nothing down.
- Skip an exclusion of something whose absence is already implied by another entry's wanted value.
- List only what a listing could state: no narrative details, no wishes about records or documents, no buyer's \
own plans. A remark about how the buyer will inspect, judge, photograph or verify the item, and a general \
hope about quality, comfort or fit that names no specific value, is not a criterion. At most 12 criteria.
Reply with one JSON object that follows the reply schema."""

GP_DEPARTMENTS = """\
A buyer wants one item. From the store department names below, pick the departments where that item would be \
filed, strongest match first, no more than {limit}. Copy names exactly, character for character. Skip any \
department that is unrelated.
department_names lists the chosen names exactly as given.
Reply with one JSON object that follows the reply schema."""

GP_SHORTLIST = """\
You select items that can meet a buyer's needs. Read the brief and its mandatory criteria, then choose up to \
{n} item ids from the list that most probably meet ALL mandatory criteria (item type first). Favour exact \
matches stated in the title, department or description. Drop items that plainly conflict with a criterion. \
{cheap}
shortlisted_ids lists the chosen ids, most likely first; rationale is one sentence.
Reply with one JSON object that follows the reply schema."""

GP_QUESTIONS = """\
Before choosing an item, a store assistant may put up to {n} questions to the buyer. The buyer holds some \
preferences back and discloses a preference solely when a question targets precisely that attribute; a \
question about anything else gets no stated preference, and when one question names several attributes only \
the first is answered.
Choose {n} distinct attributes, most promising first:
1. the intended wearer or user (the person's gender), when one person will use the item, the brief leaves it \
open and the candidates do not all target the same group; give it origin who_for;
2. then attributes from catalogue_attributes (listing attribute names, how often each occurs, sample values), \
origin datasheet, in this preference order: its material; its pattern or print; its overall style or cut; then \
other attributes whose values differ across listings. Within that order, prefer an attribute the narrative \
hints at, and skip one the brief already settles.
Every attribute except the who_for one must be a name from catalogue_attributes.
Skip attributes the brief already settles, logistics (delivery, warranty, condition, seller), price, stock, \
quantity, lists of variants or sizes, and near-duplicates of an attribute already chosen.
Each inquiry is one short plain sentence naming the attribute the way listings name it and may offer two or \
three sample values. No instructions, lists, labels or item ids.
Each inquiry names attribute_name, its origin (datasheet, or who_for for the intended wearer or user) and \
inquiry_text, the question itself.
Reply with one JSON object that follows the reply schema."""

GP_OPENING_QUESTIONS = """\
A store assistant is about to shop for a buyer and may put up to {n} questions to the buyer before looking at \
any listing. The buyer holds some preferences back and discloses one solely when a question names precisely \
that attribute; a question naming several attributes gets only the first answered.
From the buyer's brief alone, choose {n} distinct attributes of the wanted item most likely to hide an \
unstated preference, most promising first:
1. the intended wearer or user (the person's gender) when one person will use the item and the brief leaves it \
open; give it origin who_for;
2. then attributes of that kind of item in this preference order: material; pattern or print; overall style, \
cut or form; colour; size or capacity; then whatever else buyers of that item kind usually settle. Prefer an \
attribute the narrative hints at (someone whose taste decides, how the item looks, where it goes, what it holds).
Skip attributes the brief already settles, logistics (delivery, warranty, condition, seller), price, stock and \
quantity. Name each attribute with one or two plain lower-case words, the way a listing would label it.
Each inquiry names attribute_name, its origin (datasheet, or who_for) and inquiry_text, one short plain \
question. Reply with one JSON object that follows the reply schema."""

GP_REPLY_READER = """\
A store assistant asked a buyer several questions, each concerning a single attribute of an item; the \
exchanges are numbered. Read each exchange on its own, using only its question and its reply, and decide \
whether that reply fixes a value the buyer requires.
- A reply states a value when it insists on one specific value in any wording: it says the attribute must be, \
has to be or will be accepted only as that value, that the value is fixed, settled or not open to negotiation, \
or that no other value will do.
- A reply states nothing when it lets the assistant decide, accepts any option, expresses no preference, or is \
unclear.
- relation is excludes only when the buyer rejects that value itself; equals when the buyer wants it. Wording \
saying the value cannot be traded, changed or given up means the buyer keeps it: equals, not excludes. \
at_least or at_most when the reply sets a numeric limit; then figure and unit hold it.
- wanted_value is just the wanted term, worded as the buyer put it, leaving out the attribute's own name, with \
no negation or comparison word in front.
- The attribute is the one the reply itself names; if the reply names no attribute, use the one that was asked \
about.
readings holds one entry per exchange, with its exchange number and the values that reply states.
Reply with one JSON object that follows the reply schema."""

GP_CHANGES = """\
A buying task received change notices, from the buyer or the store, after it started. Given the current \
criteria, budget and the items considered, state what changed. Report only what a notice clearly states.
retired_values lists values of criteria no longer needed, exactly as listed; new_criteria are criteria a \
notice adds (with relation, bare required_value, figure and unit for a numeric limit); revised_ceiling is a \
new spending ceiling or null; withdrawn_ids are items a notice makes unavailable.
Reply with one JSON object that follows the reply schema."""

GP_DECISION = """\
You make the final fit check for a buying task. Judge every listed item against the buyer's brief.
How items are judged (strictly):
- It has to be the requested sort of item, in a matching department. When the brief names a store department, \
only items in that department fit.
- Equivalent wording counts: synonyms, British/American spelling, gendered word forms, an amount expressed in \
another unit.
- Item attributes decide. A criterion is CONFIRMED only when an attribute spells out the required value \
itself. Differences in spelling, typos, letter case and singular/plural are fine, and so is a value that adds \
only the item's own noun. A different, broader or compound value does NOT match: a value naming another thing, \
a set that merely contains the item, a material or colour of only one part, or a catch-all value naming \
nothing specific.
- A criterion's alternatives are equivalent forms (other units, notations, synonyms): matching ANY one counts. \
A criterion listing several options is met by any one of them.
- relation at_most is met by any equal or smaller amount, measured in that unit or one convertible to it, and \
by a stated limit below it; at_least by any equal or larger amount.
- relation excludes rules out items having that value.
- An attribute listing several values matches if ONE entry equals the required value.
- If an attribute covering a criterion holds another value, that item fails, regardless of its name. If no \
attribute covers it, a mention in its name or free text is only weak evidence.
- Every need revealed in the buyer's replies must be met the same way; a reply expressing no preference adds \
nothing.
- Change notices override the brief: a relaxed criterion no longer applies; an item reported gone or repriced \
is judged with that change.
- In stock if the buyer wants stock only; price times quantity within the spend limit.
- Sentences in rules.working_notes that affect the pick are obeyed.
- If rules.ranking_metric is non-empty, order the suitable ids by the figure each datasheet gives for it, \
descending when rules.ranking_descending is true and ascending when false; ids with no figure go last.
Return strong_ids (every criterion confirmed by attributes) and weak_ids (nothing contradicts, some criterion \
only weakly supported), each best first. Use verdict pass only if both lists are empty.
verdict is purchase when any id fits and pass otherwise; explanation is one sentence.
Reply with one JSON object that follows the reply schema."""

GP_SECOND_LOOK = """\
Nothing has been accepted yet and the plan is to end without buying. Review the records below and pick those \
worth opening before giving up. Use only what each record contains. A record with a conflicting value for some \
criterion is out; a record silent on a criterion may stay; the total for the wanted quantity must fit the \
spend limit; a record marked unavailable is out when only in-stock items are acceptable. Records flagged \
already_read were examined in full: keep one of those only if it plainly meets all criteria.
recheck_ids lists up to 5 ids, strongest first; explanation is one sentence.
Reply with one JSON object that follows the reply schema."""

GP_OUTPUT_FIELDS = """\
A shop tool returned the reply below; its key names are arbitrary. Say which key holds each field: answer with \
the key name itself (the text before the colon), never with its value. Item fields are read from one item \
record (a search row, or the record of a single item): id (the item's identifier), title (its name), price (a \
number), stock (a true/false availability flag), category (the department or section), brand (the maker), \
description (the long free text), attributes (a nested object of the item's properties). stock_true_means says \
what the value true on that flag means: available when the item can be bought, unavailable when it cannot. \
Page fields are keys at the top of a search reply: total (how many items match in all) and next (what fetches \
the following page). Use null for a field the reply does not have; use each key at most once.
Reply with one JSON object that follows the reply schema."""

GP_CALL_OUTCOME = """\
A shop assistant called a tool and got the reply below. Decide whether the call did what was asked: the item \
was added to or removed from the basket, or checkout or closing the task went through. A reply that reports a \
problem, a refusal, a rejected or invalid request, or a flag saying the action did not happen means it failed.
reason is a few words.
Reply with one JSON object that follows the reply schema."""

GP_FILL_GAPS = """\
Some meanings of these shop tools' arguments are still unknown. For each tool below, name the argument or the \
enum value for each listed gap, using exact argument names and exact enum values from the tool; use null when \
the tool truly has none. Gap names: text (free search words), category (department filter), brand (maker \
filter), max_price (price ceiling), min_price (price floor), page (page number), cursor (token that fetches \
the next page), mode (enum choosing free-text versus field search), id (item id), quantity (how many), action \
(enum choosing add versus remove), finish (enum choosing order versus closing without buying), message (text \
for the buyer), remark (closing note), ids (list of item ids); enum value gaps: mode_text, mode_filter, add, \
remove, finish_order, finish_decline.
tool_bindings holds one entry per tool: argument_map maps each gap to an argument name, enum_map maps each \
value gap to an enum value.
Reply with one JSON object that follows the reply schema."""


RoleName = Literal[tuple(GC_ROLE_DESCRIPTIONS)]
RelationName = Literal["equals", "excludes", "at_least", "at_most"]
CriterionType = Literal["product_type", "property", "logistics", "quantity", "rule"]
InquiryKind = Literal["budget", "choice"]
QuestionOrigin = Literal["datasheet", "who_for"]
VerdictName = Literal["purchase", "pass"]
StockMeaning = Literal["available", "unavailable"]
EnumScalar = str | int | float | bool | None


class T_ToolRoles(TypedDict):
    classifications: dict[str, RoleName]


class T_ToolBinding(TypedDict):
    argument_map: dict[str, str | None]
    enum_map: dict[str, EnumScalar]


class T_ToolBindings(TypedDict):
    tool_bindings: dict[str, T_ToolBinding]


class T_SearchPhrases(TypedDict):
    item_noun: str
    keyword_queries: list[str]


class T_Inquiry(TypedDict):
    kind: InquiryKind
    choices: list[str]
    inquiry: str


class T_PurchaseRules(TypedDict):
    spend_limit: float | None
    unit_count: int
    unit_count_quote: str
    lowest_price_wanted: bool
    stock_only: bool
    contact_limit: int | None
    action_limit: int | None
    item_kind: str
    skip_alternatives: bool
    ranking_metric: str
    ranking_descending: bool
    ranking_quote: str
    inspection_count: int | None
    inspection_quote: str
    working_notes: list[str]
    pre_pick_inquiries: list[T_Inquiry]


class T_Criterion(TypedDict):
    quote: str
    property_name: str
    relation: RelationName
    required_value: str
    options: list[str]
    figure: float | None
    unit: str
    criterion_type: CriterionType
    is_mandatory: bool
    alternatives: list[str]


class T_Requirements(TypedDict):
    criteria: list[T_Criterion]


class T_Departments(TypedDict):
    department_names: list[str]


class T_Shortlist(TypedDict):
    shortlisted_ids: list[str]
    rationale: str


class T_Question(TypedDict):
    attribute_name: str
    origin: QuestionOrigin
    inquiry_text: str


class T_Questions(TypedDict):
    inquiries: list[T_Question]


class T_StatedWant(TypedDict):
    property_name: str
    relation: RelationName
    wanted_value: str
    figure: float | None
    unit: str
    alternatives: list[str]


class T_ReplyReading(TypedDict):
    exchange: int
    stated_wants: list[T_StatedWant]


class T_ReplyReadings(TypedDict):
    readings: list[T_ReplyReading]


class T_NewCriterion(TypedDict):
    property_name: str
    relation: RelationName
    required_value: str
    figure: float | None
    unit: str
    alternatives: list[str]


class T_Changes(TypedDict):
    retired_values: list[str]
    new_criteria: list[T_NewCriterion]
    revised_ceiling: float | None
    withdrawn_ids: list[str]


class T_Decision(TypedDict):
    verdict: VerdictName
    strong_ids: list[str]
    weak_ids: list[str]
    explanation: str


class T_SecondLook(TypedDict):
    recheck_ids: list[str]
    explanation: str


class T_ItemFields(TypedDict):
    id: str | None
    title: str | None
    price: str | None
    stock: str | None
    stock_true_means: StockMeaning
    category: str | None
    brand: str | None
    description: str | None
    attributes: str | None


class T_PageFields(TypedDict):
    total: str | None
    next: str | None


class T_OutputFields(TypedDict):
    item: T_ItemFields
    page: T_PageFields


class T_CallOutcome(TypedDict):
    succeeded: bool
    reason: str


GV_SCHEMAS: dict = {}


def gm_schema_of(tp: Any) -> dict:
    if isinstance(tp, dict):
        return tp
    if isinstance(tp, type) and issubclass(tp, dict) and hasattr(tp, "__annotations__"):
        if tp not in GV_SCHEMAS:
            hints = get_type_hints(tp, globalns=globals(), include_extras=True)
            props, required = {}, []
            for name, hint in hints.items():
                optional = get_origin(hint) is NotRequired
                props[name] = gm_schema_of(get_args(hint)[0] if optional else hint)
                if not optional:
                    required.append(name)
            GV_SCHEMAS[tp] = {"type": "object", "properties": props, "required": required,
                              "additionalProperties": False}
        return GV_SCHEMAS[tp]
    origin, args = get_origin(tp), get_args(tp)
    if origin is Literal:
        return {"type": "string", "enum": list(args)}
    if origin is list:
        return {"type": "array", "items": gm_schema_of(args[0]) if args else {}}
    if origin is dict:
        return {"type": "object", "additionalProperties": gm_schema_of(args[1]) if len(args) > 1 else {}}
    if args and type(None) in args:
        return {"anyOf": [gm_schema_of(a) for a in args]}
    if args:
        return {"anyOf": [gm_schema_of(a) for a in args]}
    simple = {str: "string", bool: "boolean", int: "integer", float: "number", type(None): "null"}
    return {"type": simple[tp]} if tp in simple else {}


_MISSING = object()


def gm_conform(value: Any, schema: dict) -> Any:
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            fitted = gm_conform(value, option)
            if fitted is not _MISSING:
                return fitted
        return _MISSING
    if "enum" in schema:
        return value if value in schema["enum"] else _MISSING
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return _MISSING
        out = {}
        props = schema.get("properties")
        if props is not None:
            for name, sub in props.items():
                fitted = gm_conform(value[name], sub) if name in value else _MISSING
                if fitted is not _MISSING:
                    out[name] = fitted
            return out
        extra = schema.get("additionalProperties", {})
        for name, item in value.items():
            fitted = gm_conform(item, extra) if isinstance(extra, dict) else item
            if fitted is not _MISSING:
                out[str(name)] = fitted
        return out
    if kind == "array":
        if not isinstance(value, list):
            return _MISSING
        items = [gm_conform(v, schema.get("items", {})) for v in value]
        return [v for v in items if v is not _MISSING]
    if kind == "string":
        return value if isinstance(value, str) else _MISSING
    if kind == "boolean":
        return value if isinstance(value, bool) else _MISSING
    if kind == "integer":
        if isinstance(value, bool):
            return _MISSING
        if isinstance(value, int):
            return value
        return int(value) if isinstance(value, float) and value.is_integer() else _MISSING
    if kind == "number":
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else _MISSING
    if kind == "null":
        return None if value is None else _MISSING
    return value


def gm_with_choices(schema: dict, choices: dict[str, list[str]]) -> dict:
    out = json.loads(json.dumps(schema))
    for path, values in choices.items():
        node = out
        *parents, last = path.split(".")
        for part in parents:
            node = node["properties"][part]
        node["properties"][last] = {"anyOf": [{"type": "string", "enum": list(values)}, {"type": "null"}]}
    return out


def gm_reply_format(tp: Any) -> dict:
    name = tp.__name__.lower() if isinstance(tp, type) else "reply"
    return {"type": "json_schema", "json_schema": {"name": name, "strict": False, "schema": gm_schema_of(tp)}}




def gm_time_budget(deadline: float | None, cap: float) -> float:
    if deadline is None:
        return cap
    return max(GC_MIN_REQUEST_SECONDS, min(cap, deadline - time.monotonic()))


def gm_hash_32(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def gm_normalize_space(text: Any) -> str:
    return RE_NORMALIZE_SPACE.sub(" ", str(text or "")).strip()


def gm_get_words(text: Any) -> list[str]:
    return RE_WORD.findall(str(text or "").lower())


def gm_extract_json_object(text: str) -> dict:
    text = RE_CODE_FENCE.sub("", text)
    decoder = json.JSONDecoder()
    for match in RE_OPEN_BRACE.finditer(text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in reply")


def gm_reply_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        strings = [v for v in payload.values() if isinstance(v, str)]
        if len(strings) == 1:
            return strings[0]
    return json.dumps(payload, ensure_ascii=False)


def gm_request_has_number(request: str, value: float) -> bool:
    digits = RE_DIGIT_SEPARATORS.sub("", request)
    text = str(int(value)) if float(value).is_integer() else str(value)
    return re.search(rf"(?<![\d.]){re.escape(text)}(?![\d])", digits) is not None


def gm_request_has_count(request: str, value: int, words: str = "") -> bool:
    quoted = gm_normalize_space(words).lower()
    return gm_request_has_number(request, value) or bool(quoted and quoted in gm_normalize_space(request).lower())


def gm_criterion(item: dict, value_key: str, quote: str, attribute: str, critical: bool = True,
                 kind: str = "property") -> "C_Requirement | None":
    value = gm_normalize_space(item.get(value_key, "")) if isinstance(item.get(value_key), str) else ""
    relation = item.get("relation") if item.get("relation") in C_Requirement.RELATIONS else "equals"
    figure = item.get("figure")
    figure = float(figure) if isinstance(figure, (int, float)) and not isinstance(figure, bool) else None
    unit = gm_normalize_space(item.get("unit", "")) if isinstance(item.get("unit"), str) else ""
    if relation in ("at_least", "at_most") and figure is None:
        found = C_Requirement.cm_figure_of(value)
        if found is None:
            return None
        figure = found[0]
    if not value:
        if figure is None:
            return None
        value = gm_normalize_space(f"{figure:g} {unit}")
    options = [gm_normalize_space(o) for o in item.get("options") or [] if isinstance(o, str) and o.strip()][:8]
    variants = [gm_normalize_space(v) for v in item.get("alternatives") or [] if isinstance(v, str) and v.strip()][:8]
    return C_Requirement(quote or value, attribute or "requirement", value, critical, variants, kind,
                         relation, options, figure, unit)


def gm_slot_for_value(ev: "C_EnumValue") -> "C_Slot":
    if ev in (C_EnumValue.MODE_TEXT, C_EnumValue.MODE_FILTER):
        return C_Slot.MODE
    if ev in (C_EnumValue.FINISH_ORDER, C_EnumValue.FINISH_DECLINE):
        return C_Slot.FINISH
    return C_Slot.ACTION


def gm_measure_value(product: "C_Product", measure: str) -> float | None:
    wanted = {C_Canon.sm_singular(w) for w in gm_get_words(measure) if len(w) > 2}
    best: tuple[int, float] | None = None
    for key, value in (product.attrs or {}).items():
        names = {C_Canon.sm_singular(w) for w in gm_get_words(str(key)) if len(w) > 2}
        overlap = len(wanted & names)
        figure = RE_NUMBER_TEXT.search(str(value))
        if overlap and figure and (best is None or overlap > best[0]):
            text = RE_THOUSANDS_COMMA.sub("", figure.group(0)).replace(",", ".")
            best = (overlap, float(text))
    return best[1] if best else None


def gm_rank_by_measure(ids: list[str], catalog: "C_Catalog", measure: str, most: bool) -> list[str]:
    stated = [(i, gm_measure_value(catalog.m_get(i), measure)) for i in ids if catalog.m_get(i) is not None]
    with_value = sorted([x for x in stated if x[1] is not None], key=lambda x: -x[1] if most else x[1])
    return [i for i, _ in with_value] + [i for i in ids if i not in {x[0] for x in with_value}]


class C_LLMError(RuntimeError):

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class C_EnvError(RuntimeError):

    def __init__(self, status: int | None, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class C_EpisodeFinished(RuntimeError):
    pass


class C_StepBudgetExhausted(RuntimeError):
    pass


class C_RetryPolicy:

    def __init__(self, key: str, base: float = GC_RETRY_BASE_SECONDS, cap: float = GC_RETRY_CAP_SECONDS) -> None:
        seed = int(hashlib.sha256((key or "session").encode()).hexdigest()[:12], 16)
        self.rng = random.Random(seed)
        self.base = base
        self.cap = cap
        self.offset = self.rng.uniform(0.0, base)
        self.lock = threading.Lock()

    def m_delay(self, attempt: int, status: int | None = None) -> float:
        base = self.base * (GC_RETRY_RATE_LIMIT_FACTOR if status == 429 else 1.0)
        with self.lock:
            factor = self.rng.uniform(0.5, 1.5)
        return min(self.cap, base * (2 ** attempt)) * factor + self.offset

    def m_wait(self, attempt: int, status: int | None = None) -> None:
        time.sleep(self.m_delay(attempt, status))

    @classmethod
    def cm_for_session(cls, key: str | None) -> "C_RetryPolicy":
        key = key or "session"
        with GV_RETRY_LOCK:
            if key not in GV_RETRY_POLICIES:
                GV_RETRY_POLICIES[key] = cls(key)
            return GV_RETRY_POLICIES[key]


@dataclass(frozen=True)
class C_EpisodeContext:

    session_id: str
    spend: list = field(default_factory=lambda: [0.0])
    deadline: float | None = None

    @classmethod
    def cm_for_episode(cls, session_id: str, deadline: float | None = None) -> "C_EpisodeContext":
        return cls(session_id, deadline=deadline)


class C_Memo:

    @classmethod
    def cm_get_or_compute(cls, key: str, compute: Callable[[], tuple[Any, bool]]) -> Any:
        with GV_MEMO_LOCK:
            if key in GV_MEMO_VALUES:
                return GV_MEMO_VALUES[key]
        value, keep = compute()
        if keep:
            with GV_MEMO_LOCK:
                GV_MEMO_VALUES[key] = value
        return value


class C_ChatClient(Protocol):

    def m_text(self, messages: list[dict], **kw: Any) -> str: ...

    def m_json(self, messages: list[dict], **kw: Any) -> dict: ...


class C_Gateway:

    @classmethod
    def cm_client(cls, timeout: float) -> Any:
        key = max(5, int(round(timeout / 5.0)) * 5)
        with GV_GATEWAY_LOCK:
            if key not in GV_OPENROUTER_GATEWAYS:
                GV_OPENROUTER_GATEWAYS[key] = ProxyClient(timeout=key, max_retries=1, retry_delay=0)
            return GV_OPENROUTER_GATEWAYS[key]

    @classmethod
    def cm_post(cls, path: str, body: dict, timeout: float) -> tuple[dict | None, int | None, str]:
        result = cls.cm_client(timeout).post_verbose(path, body)
        if result.data is not None:
            return result.data, 200, ""
        error = result.error or {}
        body_text = error.get("body")
        if not isinstance(body_text, str):
            body_text = json.dumps(body_text, ensure_ascii=False, default=str)
        return None, error.get("status"), body_text or str(error.get("kind") or "no response")


class C_LLMClient:

    PATH = "/inference/chat/completions"
    IN_FLIGHT = re.compile(r"in[-_ ]flight", re.I)
    EXHAUSTED = re.compile(r"insufficient credits|requires more credits|key limit exceeded", re.I)
    GONE = re.compile(r"not allowed|no endpoints|not a valid model|model .{0,60}not found", re.I)

    def __init__(self) -> None:
        self.exhausted = False
        self.calls = 0
        self.cost = 0.0
        self.lock = threading.Lock()

    def m_chat(self, messages: list[dict], *, model: str | None = None, max_tokens: int = GC_MAX_TOKENS,
               timeout: float | None = None, retry_key: str | None = None,
               cost_sink: list | None = None, **extra: Any) -> dict:
        body = {"model": model or GC_PRIMARY_MODEL, "messages": messages,
                "temperature": 0, "max_tokens": max_tokens, **extra}
        response = self.m_post(body, timeout or GC_OPENROUTER_TIMEOUT, retry_key)
        cost = (response.get("usage") or {}).get("cost")
        if cost_sink is not None and isinstance(cost, (int, float)):
            cost_sink[0] += float(cost)
        return response

    @staticmethod
    def sm_content(response: dict) -> str:
        return response["choices"][0]["message"].get("content") or ""

    @staticmethod
    def sm_truncated(response: dict) -> bool:
        return response["choices"][0].get("finish_reason") == "length"

    def m_text(self, messages: list[dict], **kw: Any) -> str:
        return self.sm_content(self.m_chat(messages, **kw))

    def m_json(self, messages: list[dict], *, attempts: int = 2, **kw: Any) -> dict:
        deadline = kw.pop("deadline", None)
        primary = kw.get("model") or GC_PRIMARY_MODEL
        chain = [primary] + [m for m in GC_FALLBACK_MODELS if m != primary]
        failure: Exception = C_LLMError("no model available")
        cycle = 0
        while True:
            with GV_DEAD_LOCK:
                usable = [m for m in chain if m not in GV_DEAD_MODELS] or [primary]
            transient = False
            for index, model in enumerate(usable):
                try:
                    tries = attempts if index == 0 and cycle == 0 else 1
                    return self.m_json_once(messages, attempts=tries, **{**kw, "model": model})
                except C_LLMError as exc:
                    if self.exhausted:
                        raise
                    if exc.status in GC_GONE_STATUSES or self.GONE.search(str(exc)):
                        with GV_DEAD_LOCK:
                            GV_DEAD_MODELS.add(model)
                    elif exc.status not in (400, 401):
                        transient = True
                    failure = exc
                except ValueError as exc:
                    transient = True
                    failure = exc
            if not transient:
                raise failure
            if deadline is not None and time.monotonic() + 5 > deadline:
                raise failure
            C_RetryPolicy.cm_for_session(kw.get("retry_key")).m_wait(min(cycle, 4), 429)
            cycle += 1

    def m_json_once(self, messages: list[dict], *, attempts: int = 2, reply_type: Any = None, **kw: Any) -> dict:
        model = kw.get("model") or GC_PRIMARY_MODEL
        if reply_type is not None:
            schema = gm_schema_of(reply_type)
            messages = [{**messages[0], "content": messages[0]["content"] + "\n\nReply schema (JSON Schema):\n"
                         + json.dumps(schema, ensure_ascii=False)}, *messages[1:]]
            structured = model not in GV_PLAIN_JSON_MODELS
            kw["response_format"] = gm_reply_format(reply_type) if structured else {"type": "json_object"}
        else:
            kw.setdefault("response_format", {"type": "json_object"})
        problem = "no reply"
        for _ in range(attempts):
            try:
                response = self.m_chat(messages, **kw)
            except C_LLMError as exc:
                if exc.status != 400 or kw["response_format"].get("type") != "json_schema":
                    raise
                GV_PLAIN_JSON_MODELS.add(model)
                kw["response_format"] = {"type": "json_object"}
                continue
            try:
                answer = gm_extract_json_object(self.sm_content(response))
            except (ValueError, KeyError, IndexError, TypeError):
                problem = "reply cut off at max_tokens" if self.sm_truncated(response) else "no JSON object in reply"
                continue
            if reply_type is None:
                return answer
            fitted = gm_conform(answer, gm_schema_of(reply_type))
            if fitted is not _MISSING:
                return fitted
            problem = "reply does not follow the reply schema"
        raise ValueError(problem)

    def m_post(self, body: dict, timeout: float, retry_key: str | None = None) -> dict:
        if self.exhausted:
            raise C_LLMError("inference key exhausted")
        retry = C_RetryPolicy.cm_for_session(retry_key)
        last = "unknown error"
        status = None
        attempts = GC_OPENROUTER_RETRIES + 1
        attempt = -1
        while attempt + 1 < attempts:
            attempt += 1
            with self.lock:
                self.calls += 1
            data, status, error = C_Gateway.cm_post(self.PATH, body, timeout)
            if isinstance(data, dict) and data.get("choices"):
                cost = (data.get("usage") or {}).get("cost")
                if isinstance(cost, (int, float)):
                    with self.lock:
                        self.cost += float(cost)
                return data
            last = f"HTTP {status}: {error[:300]}" if data is None else f"no choices: {json.dumps(data)[:300]}"
            if status in (402, 403) and self.EXHAUSTED.search(last):
                self.exhausted = True
                break
            if status == 402 and self.IN_FLIGHT.search(last):
                attempts = min(attempts + 1, GC_OPENROUTER_RETRIES + 4)
            elif status in GC_NO_RETRY_STATUSES:
                break
            if attempt + 1 < attempts:
                retry.m_wait(attempt, status)
        raise C_LLMError(last, status)


class C_LLMTask:

    model = GC_PRIMARY_MODEL
    reasoning = GC_ANALYSIS_REASONING
    token_limit = GC_TASK_MAX_TOKENS
    reply_type: Any = None

    def __init__(self, client: C_ChatClient | None, episode: C_EpisodeContext | None = None) -> None:
        self.client = client
        self.episode = episode
        self.last_error: str | None = None

    def m_ask(self, system: str, user: str, reply_type: Any = None) -> dict | None:
        if self.client is None:
            self.last_error = "no LLM client"
            return None
        extra: dict[str, Any] = {"reasoning": {"effort": self.reasoning}} if self.reasoning else {}
        model = self.model
        if self.episode is not None:
            extra["timeout"] = gm_time_budget(self.episode.deadline, GC_OPENROUTER_TIMEOUT)
            extra["retry_key"] = self.episode.session_id
            extra["deadline"] = self.episode.deadline
            extra["cost_sink"] = self.episode.spend
            if self.episode.spend[0] >= GC_TASK_COST_BUDGET and model != GC_BUDGET_MODEL:
                model = GC_BUDGET_MODEL
        try:
            return self.client.m_json([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                      model=model, max_tokens=self.token_limit,
                                      reply_type=reply_type or self.reply_type, **extra)
        except (C_LLMError, ValueError, KeyError, IndexError, TypeError) as exc:
            self.last_error = f"{type(self).__name__}: {exc}"
            GV_LLM_FAILURES.append(self.last_error[:300])
            return None


class C_Role(str, Enum):
    SEARCH = "search"
    LISTING = "listing"
    DETAILS = "details"
    CART_ADD = "cart_add"
    CART_REMOVE = "cart_remove"
    CART_EDIT = "cart_edit"
    CART_VIEW = "cart_view"
    ORDER = "order"
    DECLINE = "decline"
    MESSAGE = "message"
    COMPARE = "compare"
    OTHER = "other"

    @property
    def m_description(self) -> str:
        return GC_ROLE_DESCRIPTIONS[self.value]

    @property
    def m_is_final(self) -> bool:
        return self in (C_Role.ORDER, C_Role.DECLINE)


class C_Kind(str, Enum):

    ENUM = "enum"
    ID = "id"
    SHORT_TEXT = "short_text"
    LONG_TEXT = "long_text"
    INT = "int"
    NUMBER = "number"
    BOOL = "bool"
    ARRAY = "array"
    OBJECT = "object"
    OTHER = "other"

    @classmethod
    def cm_of(cls, schema: dict) -> "C_Kind":
        raw = schema.get("type")
        types = (set(raw) if isinstance(raw, list) else {raw} if isinstance(raw, str) else set()) - {"null"}
        if "enum" in schema or "const" in schema:
            return cls.ENUM
        if types == {"string"}:
            max_len = schema.get("maxLength")
            if isinstance(max_len, int) and max_len >= GC_LONG_TEXT_MIN_LEN:
                return cls.LONG_TEXT
            if isinstance(max_len, int) and max_len <= GC_ID_MAX_LEN and schema.get("minLength", 0) >= 1:
                return cls.ID
            return cls.SHORT_TEXT
        return {
            frozenset({"integer"}): cls.INT,
            frozenset({"number"}): cls.NUMBER,
            frozenset({"integer", "number"}): cls.NUMBER,
            frozenset({"boolean"}): cls.BOOL,
            frozenset({"array"}): cls.ARRAY,
            frozenset({"object"}): cls.OBJECT,
        }.get(frozenset(types), cls.OTHER)


@dataclass(frozen=True)
class C_ToolParam:

    name: str
    required: bool
    kind: C_Kind
    schema: dict

    @property
    def m_token(self) -> str:
        return f"{self.kind.value}{'!' if self.required else '?'}"

    @property
    def m_enum_values(self) -> list:
        if "enum" in self.schema and isinstance(self.schema["enum"], list):
            return list(self.schema["enum"])
        if "const" in self.schema:
            return [self.schema["const"]]
        return []


@dataclass(frozen=True)
class C_ToolSpec:

    name: str
    description: str
    parameters: dict
    params: tuple

    @classmethod
    def cm_from_dict(cls, tool: dict) -> "C_ToolSpec | None":
        fn = tool.get("function", tool) if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            return None
        parameters = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        props = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
        required = set(parameters.get("required") or [])
        params = tuple(
            C_ToolParam(name, name in required, C_Kind.cm_of(s if isinstance(s, dict) else {}),
                        s if isinstance(s, dict) else {})
            for name, s in props.items()
        )
        return cls(fn["name"], fn.get("description") or "", parameters, params)

    @classmethod
    def cm_parse_all(cls, tools: list[dict]) -> list["C_ToolSpec"]:
        return [spec for spec in (cls.cm_from_dict(t) for t in tools or []) if spec is not None]

    @property
    def m_signature(self) -> str:
        return "|".join(sorted(p.m_token for p in self.params)) or "-"

    def m_kinds(self, required_only: bool = False) -> list[C_Kind]:
        return sorted((p.kind for p in self.params if p.required or not required_only), key=lambda k: k.value)

    def m_param(self, name: str | None) -> C_ToolParam | None:
        return next((p for p in self.params if p.name == name), None) if name else None

    def m_to_prompt(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class C_SchemaFilter:

    ALL = list(C_Role)

    def m_candidates(self, spec: C_ToolSpec) -> list[C_Role]:
        kinds = spec.m_kinds()
        n = len(kinds)
        has = kinds.count
        K, R = C_Kind, C_Role

        if n == 0:
            return [R.CART_VIEW, R.ORDER, R.DECLINE, R.OTHER]
        if kinds == [K.LONG_TEXT]:
            required = bool(spec.m_kinds(required_only=True))
            return [R.MESSAGE, R.DECLINE, R.ORDER] if required else [R.DECLINE, R.MESSAGE, R.ORDER]
        if n == 2 and has(K.ENUM) and has(K.LONG_TEXT):
            return [R.ORDER, R.DECLINE, R.MESSAGE, R.SEARCH, R.OTHER]
        if n == 2 and has(K.ENUM) and has(K.SHORT_TEXT):
            return [R.SEARCH, R.ORDER, R.DECLINE, R.MESSAGE, R.OTHER]
        if n == 1 and kinds[0] in (K.ID, K.SHORT_TEXT):
            return [R.LISTING, R.DETAILS, R.CART_ADD, R.CART_REMOVE, R.ORDER, R.SEARCH, R.MESSAGE]
        if has(K.ENUM) and has(K.ID) and n <= 3 and not has(K.LONG_TEXT):
            if has(K.SHORT_TEXT) or has(K.NUMBER):
                return [R.SEARCH, R.CART_EDIT]
            return [R.CART_EDIT, R.CART_ADD, R.CART_REMOVE]
        if n == 2 and has(K.ID) and has(K.INT):
            return [R.CART_ADD, R.CART_REMOVE, R.CART_EDIT, R.ORDER]
        if has(K.SHORT_TEXT) or (n >= 2 and (has(K.NUMBER) or has(K.ENUM))):
            return [R.SEARCH, R.OTHER]
        if has(K.ID) and has(K.LONG_TEXT):
            return [R.MESSAGE, R.DECLINE, R.ORDER, R.OTHER]
        if kinds == [K.ARRAY]:
            return [R.COMPARE, R.ORDER, R.CART_EDIT, R.OTHER]
        if has(K.ARRAY) or has(K.OBJECT):
            return [R.ORDER, R.CART_EDIT, R.SEARCH, R.OTHER]
        return list(self.ALL)


class C_LLMRoleResolver:

    def __init__(self, client: C_ChatClient) -> None:
        self.client = client

    def m_resolve(self, specs: list[C_ToolSpec], ambiguous: dict[str, list[C_Role]],
                  episode: C_EpisodeContext | None = None) -> dict[str, C_Role]:
        extra: dict[str, Any] = {"reasoning": {"effort": GC_ROLE_REASONING}} if GC_ROLE_REASONING else {}
        if episode is not None:
            extra["retry_key"] = episode.session_id
            extra["deadline"] = episode.deadline
            extra["timeout"] = gm_time_budget(episode.deadline, GC_OPENROUTER_TIMEOUT)
        answer = self.client.m_json(self.sm_prompt(specs, ambiguous), model=GC_PRIMARY_MODEL,
                                    max_tokens=GC_MAX_TOKENS, reply_type=T_ToolRoles, **extra)
        picks = answer.get("classifications") if isinstance(answer, dict) else None
        picks = picks if isinstance(picks, dict) else {}
        resolved: dict[str, C_Role] = {}
        for name, allowed in ambiguous.items():
            value = picks.get(name)
            if isinstance(value, str):
                with contextlib.suppress(ValueError):
                    role = C_Role(value.strip())
                    if role in allowed:
                        resolved[name] = role
        return resolved

    @staticmethod
    def sm_prompt(specs: list[C_ToolSpec], ambiguous: dict[str, list[C_Role]]) -> list[dict]:
        roles = "\n".join(f"- {r.value}: {r.m_description}" for r in C_Role)
        tools = json.dumps([s.m_to_prompt() for s in specs], ensure_ascii=False)
        todo = "\n".join(f"- {name}: allowed roles {[r.value for r in allowed]}" for name, allowed in ambiguous.items())
        return [
            {"role": "system", "content": GP_TOOL_ROLES.replace("{roles}", roles)},
            {"role": "user", "content": f"Every tool:\n{tools}\n\nClassify these tools:\n{todo}"},
        ]


@dataclass
class C_RoleMap:

    roles: dict[str, C_Role]
    sources: dict[str, str]
    candidates: dict[str, list[C_Role]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def m_by_role(self) -> dict[C_Role, list[str]]:
        out: dict[C_Role, list[str]] = {}
        for name, role in self.roles.items():
            out.setdefault(role, []).append(name)
        return out

    def m_tool_for(self, role: C_Role) -> str | None:
        names = self.m_by_role().get(role)
        return names[0] if names else None

    def m_has(self, role: C_Role) -> bool:
        return role in self.roles.values()

    def m_to_dict(self) -> dict:
        return {"roles": {n: r.value for n, r in self.roles.items()}, "sources": dict(self.sources)}


class C_RoleDetector:

    def __init__(self, client: C_ChatClient | None, *, schema_filter: C_SchemaFilter | None = None) -> None:
        self.resolver = C_LLMRoleResolver(client) if client is not None else None
        self.schema_filter = schema_filter or C_SchemaFilter()
        self.episode: C_EpisodeContext | None = None

    def m_cache_key(self, specs: list[C_ToolSpec]) -> str:
        return gm_hash_32({"tools": [s.m_to_prompt() for s in specs],
                           "model": GC_PRIMARY_MODEL if self.resolver else None,
                           "version": GC_ROLE_CACHE_VERSION})

    def m_detect(self, tools: list[dict], *, use_cache: bool = True,
                 episode: C_EpisodeContext | None = None) -> C_RoleMap:
        self.episode = episode
        specs = C_ToolSpec.cm_parse_all(tools)
        key = self.m_cache_key(specs)
        if not use_cache:
            return self.m_detect_specs(specs)
        if key in GV_ROLE_MEMORY:
            return GV_ROLE_MEMORY[key]
        result = self.m_detect_specs(specs)
        if all(src in ("schema", "llm") for src in result.sources.values()):
            GV_ROLE_MEMORY[key] = result
        return result

    def m_detect_specs(self, specs: list[C_ToolSpec]) -> C_RoleMap:
        candidates = {s.name: self.schema_filter.m_candidates(s) for s in specs}
        shape_count: dict[str, int] = {}
        for s in specs:
            shape_count[s.m_signature] = shape_count.get(s.m_signature, 0) + 1

        roles: dict[str, C_Role] = {}
        sources: dict[str, str] = {}
        notes: list[str] = []
        ambiguous: dict[str, list[C_Role]] = {}
        for s in specs:
            allowed = candidates[s.name]
            if len(allowed) == 1 and shape_count[s.m_signature] == 1:
                roles[s.name], sources[s.name] = allowed[0], "schema"
            else:
                ambiguous[s.name] = allowed

        for _ in range(2):
            todo = {n: a for n, a in ambiguous.items() if n not in roles}
            if not todo or self.resolver is None:
                break
            try:
                for name, role in self.resolver.m_resolve(specs, todo, self.episode).items():
                    roles[name], sources[name] = role, "llm"
            except (C_LLMError, ValueError, KeyError, IndexError, TypeError) as exc:
                notes.append(f"llm failed: {exc}")

        for s in specs:
            if s.name not in roles:
                roles[s.name], sources[s.name] = candidates[s.name][0], "shape"

        ordered = {s.name: roles[s.name] for s in specs}
        self.sm_reconcile(ordered, sources, candidates, notes)
        return C_RoleMap(ordered, sources, candidates, notes)

    @staticmethod
    def sm_reconcile(roles: dict[str, C_Role], sources: dict[str, str],
                     candidates: dict[str, list[C_Role]], notes: list[str]) -> None:
        assigned = list(roles.values())
        if C_Role.LISTING not in assigned and assigned.count(C_Role.DETAILS) == 1:
            name = next(n for n, r in roles.items() if r is C_Role.DETAILS)
            if sources[name] == "shape" and C_Role.LISTING in candidates.get(name, []):
                roles[name] = C_Role.LISTING
                notes.append(f"{name}: only lookup tool, details -> listing")


class C_Slot(str, Enum):
    TEXT = "text"
    CATEGORY = "category"
    BRAND = "brand"
    MAX_PRICE = "max_price"
    MIN_PRICE = "min_price"
    PAGE = "page"
    CURSOR = "cursor"
    MODE = "mode"
    ID = "id"
    QUANTITY = "quantity"
    ACTION = "action"
    MESSAGE = "message"
    IDS = "ids"
    REMARK = "remark"
    FINISH = "finish"


class C_EnumValue(str, Enum):
    MODE_TEXT = "mode_text"
    MODE_FILTER = "mode_filter"
    ADD = "add"
    REMOVE = "remove"
    FINISH_ORDER = "finish_order"
    FINISH_DECLINE = "finish_decline"


@dataclass
class C_ToolBinding:

    tool: str
    role: C_Role
    slots: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)

    def m_to_dict(self) -> dict:
        return {"tool": self.tool, "role": self.role.value, "slots": {k.value: v for k, v in self.slots.items()},
                "values": {k.value: v for k, v in self.values.items()}}


class C_HeuristicSlotMapper:

    def m_map(self, spec: C_ToolSpec, role: C_Role) -> C_ToolBinding:
        binding = C_ToolBinding(spec.name, role)
        params = list(spec.params)
        texts = [p for p in params if p.kind in (C_Kind.SHORT_TEXT, C_Kind.ID)]
        longs = [p for p in params if p.kind is C_Kind.LONG_TEXT]
        notes = [p for p in texts + longs if RE_NOTE_PARAM.search(p.name) or RE_NOTE_PARAM.search(str(p.schema.get("description") or ""))]
        plain = [p for p in texts if p not in notes]
        enums = [p for p in params if p.kind is C_Kind.ENUM]
        ints = [p for p in params if p.kind is C_Kind.INT]
        numbers = [p for p in params if p.kind is C_Kind.NUMBER]
        arrays = [p for p in params if p.kind is C_Kind.ARRAY]

        def only(group: list[C_ToolParam]) -> C_ToolParam | None:
            return group[0] if len(group) == 1 else None

        def take(slot: C_Slot, param: C_ToolParam | None) -> None:
            if param is not None and param.name not in binding.slots.values():
                binding.slots[slot] = param.name

        if role is C_Role.SEARCH:
            take(C_Slot.TEXT, only(plain) if not longs else None)
            take(C_Slot.CATEGORY, only([e for e in enums if len(e.m_enum_values) >= 4]))
            take(C_Slot.MAX_PRICE, only(numbers))
            take(C_Slot.PAGE, only([p for p in ints if p.schema.get("minimum") in (0, 1)]))
        elif role in (C_Role.LISTING, C_Role.DETAILS):
            take(C_Slot.ID, only(texts))
        elif role in (C_Role.CART_ADD, C_Role.CART_REMOVE, C_Role.CART_EDIT, C_Role.ORDER):
            take(C_Slot.ID, only(plain) if notes else only(texts))
            take(C_Slot.QUANTITY, only(ints))
            if role is C_Role.ORDER:
                take(C_Slot.REMARK, only(longs) or only(notes))
                take(C_Slot.FINISH, only([e for e in enums if len(e.m_enum_values) >= 2]))
        elif role is C_Role.COMPARE:
            take(C_Slot.IDS, only(arrays))
        elif role is C_Role.MESSAGE:
            take(C_Slot.MESSAGE, only(longs) or (only(texts) if not longs else None))
        elif role is C_Role.DECLINE:
            take(C_Slot.REMARK, only(longs) or only(notes) or (only(texts) if not longs else None))
            take(C_Slot.FINISH, only([e for e in enums if len(e.m_enum_values) >= 2]))
        if role.m_is_final and C_Slot.FINISH in binding.slots:
            param = spec.m_param(binding.slots[C_Slot.FINISH])
            for ev, rx in ((C_EnumValue.FINISH_ORDER, RE_FINISH_ORDER), (C_EnumValue.FINISH_DECLINE, RE_FINISH_DECLINE)):
                hits = [v for v in param.m_enum_values if isinstance(v, str) and rx.search(v)]
                if len(hits) == 1:
                    binding.values[ev] = hits[0]
        return binding


class C_LLMSlotMapper(C_LLMTask):

    model = GC_PRIMARY_MODEL
    reasoning = GC_ROLE_REASONING
    token_limit = GC_MAX_TOKENS

    reply_type = T_ToolBindings
    SYSTEM = GP_SLOT_MAPPER

    def m_map(self, specs: list[C_ToolSpec], roles: C_RoleMap) -> dict[str, dict] | None:
        listed = [{**s.m_to_prompt(), "role": roles.roles[s.name].value} for s in specs
                  if roles.roles.get(s.name) not in (None, C_Role.OTHER, C_Role.CART_VIEW)]
        if not listed:
            return {}
        answer = self.m_ask(self.SYSTEM, json.dumps(listed, ensure_ascii=False))
        bindings = answer.get("tool_bindings") if isinstance(answer, dict) else None
        return bindings if isinstance(bindings, dict) else None

    def m_fill(self, specs: list[C_ToolSpec], roles: C_RoleMap, gaps: dict[str, list[str]]) -> dict | None:
        listed = [{**s.m_to_prompt(), "role": roles.roles[s.name].value, "gaps": gaps[s.name]} for s in specs]
        answer = self.m_ask(GP_FILL_GAPS, json.dumps(listed, ensure_ascii=False))
        bindings = answer.get("tool_bindings") if isinstance(answer, dict) else None
        return bindings if isinstance(bindings, dict) else None


class C_WorldBindings:

    def __init__(self, bindings: dict[str, C_ToolBinding], source: str) -> None:
        self.bindings = bindings
        self.source = source

    def m_to_dict(self) -> dict:
        return {"source": self.source, "bindings": {k: b.m_to_dict() for k, b in self.bindings.items()}}

    SLOT_KINDS = {
        C_Slot.TEXT: (C_Kind.SHORT_TEXT, C_Kind.LONG_TEXT, C_Kind.ID),
        C_Slot.MESSAGE: (C_Kind.SHORT_TEXT, C_Kind.LONG_TEXT, C_Kind.ID),
        C_Slot.REMARK: (C_Kind.SHORT_TEXT, C_Kind.LONG_TEXT, C_Kind.ID),
        C_Slot.ID: (C_Kind.SHORT_TEXT, C_Kind.LONG_TEXT, C_Kind.ID),
        C_Slot.CATEGORY: (C_Kind.SHORT_TEXT, C_Kind.ID, C_Kind.ENUM),
        C_Slot.BRAND: (C_Kind.SHORT_TEXT, C_Kind.ID, C_Kind.ENUM),
        C_Slot.MAX_PRICE: (C_Kind.INT, C_Kind.NUMBER),
        C_Slot.MIN_PRICE: (C_Kind.INT, C_Kind.NUMBER),
        C_Slot.PAGE: (C_Kind.INT, C_Kind.NUMBER),
        C_Slot.QUANTITY: (C_Kind.INT, C_Kind.NUMBER),
        C_Slot.IDS: (C_Kind.ARRAY,),
        C_Slot.MODE: (C_Kind.ENUM,),
        C_Slot.ACTION: (C_Kind.ENUM,),
        C_Slot.FINISH: (C_Kind.ENUM,),
    }

    @classmethod
    def sm_slot_fits(cls, slot: C_Slot, param: C_ToolParam) -> bool:
        kinds = cls.SLOT_KINDS.get(slot)
        return kinds is None or param.kind in kinds

    @classmethod
    def sm_merge(cls, binding: C_ToolBinding, spec: C_ToolSpec, raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        allowed = set(GC_ROLE_SLOTS.get(binding.role.value, ()))
        for slot_name, param_name in (raw.get("argument_map") or {}).items():
            with contextlib.suppress(ValueError):
                slot = C_Slot(slot_name)
                param = spec.m_param(param_name)
                if slot.value in allowed and slot not in binding.slots and param is not None \
                        and param_name not in binding.slots.values() and cls.sm_slot_fits(slot, param):
                    binding.slots[slot] = param_name
        for value_name, value in (raw.get("enum_map") or {}).items():
            with contextlib.suppress(ValueError):
                ev = C_EnumValue(value_name)
                slot = gm_slot_for_value(ev)
                owners = [q.name for q in spec.params if value in q.m_enum_values]
                if slot not in binding.slots and slot.value in allowed and len(owners) == 1 \
                        and owners[0] not in binding.slots.values():
                    binding.slots[slot] = owners[0]
                param = spec.m_param(binding.slots.get(slot))
                if param is not None and value in param.m_enum_values and ev not in binding.values:
                    binding.values[ev] = value

    @staticmethod
    def sm_gaps(binding: C_ToolBinding, spec: C_ToolSpec, roles: C_RoleMap) -> list[str]:
        free = [p for p in spec.params if p.name not in binding.slots.values()]
        role = binding.role
        gaps = [slot for slot in GC_ROLE_SLOTS.get(role.value, ()) if C_Slot(slot) not in binding.slots and free]
        shared = role.m_is_final and not (roles.m_has(C_Role.ORDER) and roles.m_has(C_Role.DECLINE))
        if C_Slot.FINISH.value in gaps and not (shared and any(p.m_enum_values for p in free)):
            gaps.remove(C_Slot.FINISH.value)
        needs = {C_Slot.MODE: (C_EnumValue.MODE_TEXT, C_EnumValue.MODE_FILTER),
                 C_Slot.ACTION: (C_EnumValue.ADD, C_EnumValue.REMOVE),
                 C_Slot.FINISH: (C_EnumValue.FINISH_ORDER, C_EnumValue.FINISH_DECLINE)}
        for slot, values in needs.items():
            if slot in binding.slots or slot.value in gaps:
                gaps += [v.value for v in values if v not in binding.values]
        return gaps

    @classmethod
    def cm_build(cls, specs: list[C_ToolSpec], roles: C_RoleMap,
                 llm: C_LLMSlotMapper | None) -> tuple["C_WorldBindings", bool]:
        shapes = C_HeuristicSlotMapper()
        mapped = llm.m_map(specs, roles) if llm is not None else None
        bindings: dict[str, C_ToolBinding] = {}
        for spec in specs:
            binding = C_ToolBinding(spec.name, roles.roles.get(spec.name, C_Role.OTHER))
            cls.sm_merge(binding, spec, (mapped or {}).get(spec.name) if isinstance(mapped, dict) else None)
            bindings[spec.name] = binding
        by_name = {s.name: s for s in specs}
        gaps = {n: cls.sm_gaps(b, by_name[n], roles) for n, b in bindings.items()
                if b.role not in (C_Role.OTHER, C_Role.CART_VIEW)}
        gaps = {n: g for n, g in gaps.items() if g}
        if gaps and llm is not None:
            filled = llm.m_fill([by_name[n] for n in gaps], roles, gaps)
            for name in gaps:
                cls.sm_merge(bindings[name], by_name[name], (filled or {}).get(name))
        for spec in specs:
            binding = bindings[spec.name]
            shaped = shapes.m_map(spec, binding.role)
            for slot, param_name in shaped.slots.items():
                if slot not in binding.slots and param_name not in binding.slots.values():
                    binding.slots[slot] = param_name
            for ev, value in shaped.values.items():
                if ev not in binding.values and value not in binding.values.values() \
                        and binding.slots.get(C_Slot.FINISH) == shaped.slots.get(C_Slot.FINISH):
                    binding.values[ev] = value
        source = "llm" if mapped is not None else "shape"
        return cls(bindings, source), mapped is not None


class C_Transport(Protocol):

    def m_call(self, envelope: dict) -> dict: ...


class C_ProxyTransport:

    PATH = "/environment/call"

    def __init__(self, timeout: float = GC_ENV_TIMEOUT) -> None:
        self.timeout = timeout
        self.deadline: float | None = None

    @staticmethod
    def sm_detail(text: str) -> str:
        with contextlib.suppress(ValueError, TypeError, AttributeError):
            detail = json.loads(text).get("detail")
            if isinstance(detail, dict):
                return str(detail.get("error", detail))
            if detail:
                return str(detail)
        return text

    def m_call(self, envelope: dict) -> dict:
        retry = C_RetryPolicy.cm_for_session(str(envelope.get("session_id") or ""))
        attempt = 0
        while True:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise C_EnvError(None, "the task's time is up")
            data, status, error = C_Gateway.cm_post(self.PATH, envelope, gm_time_budget(self.deadline, self.timeout))
            if data is not None:
                return data
            if status is not None and status not in GC_TRANSIENT_STATUSES:
                raise C_EnvError(status, self.sm_detail(error))
            retry.m_wait(min(attempt, 4), status)
            attempt += 1


@dataclass
class C_StepResult:

    tool: str
    args: dict
    step: int
    payload: Any = None
    notices: list[str] = field(default_factory=list)
    refusal: list[str] = field(default_factory=list)
    error: str | None = None
    detail: str | None = None
    finished: bool = False
    raw: Any = None

    def m_to_log(self) -> dict:
        return {"step": self.step, "tool": self.tool, "args": self.args, "finished": self.finished,
                "error": self.error, "detail": self.detail, "notices": self.notices, "refusal": self.refusal,
                "payload": self.payload}


class C_ObservationParser:

    RUNTIME_ERRORS = {"unknown_tool", "invalid_arguments", "budget_exceeded"}

    @staticmethod
    def sm_is_notice_list(value: Any) -> bool:
        return (isinstance(value, list) and len(value) > 0 and all(isinstance(v, str) for v in value)
                and any(len(v.split()) >= 3 for v in value) and not any(gm_is_record(v) for v in value))

    def m_parse(self, obs: Any) -> tuple[Any, list[str], list[str], str | None, str | None]:
        notices: list[str] = []
        refusal: list[str] = []
        error = detail = None
        for _ in range(4):
            if isinstance(obs, dict) and set(obs) == {"observation", "refusal"}:
                if isinstance(obs["refusal"], list):
                    refusal.extend(str(r) for r in obs["refusal"])
                elif obs["refusal"]:
                    refusal.append(json.dumps(obs["refusal"], ensure_ascii=False))
                obs = obs["observation"]
                continue
            if isinstance(obs, dict) and len(obs) == 2:
                keys = list(obs)
                lists = [k for k in keys if self.sm_is_notice_list(obs[k])]
                if len(lists) == 1:
                    other = keys[1 - keys.index(lists[0])]
                    notices.extend(obs[lists[0]])
                    obs = obs[other]
                    continue
            break
        if isinstance(obs, dict) and obs.get("error") in self.RUNTIME_ERRORS and set(obs) <= {"error", "detail"}:
            error, detail = obs["error"], obs.get("detail")
        return obs, notices, refusal, error, detail


class C_Environment:

    MAX_ACTIONS = GC_MAX_ACTIONS

    def __init__(self, transport: C_Transport, binding: dict, max_steps: int, log: Callable[[str, dict], None],
                 max_actions: int = GC_MAX_ACTIONS) -> None:
        self.transport = transport
        self.max_actions = max(1, min(self.MAX_ACTIONS, int(max_actions or 1)))
        self.binding = dict(binding)
        self.max_steps = int(max_steps)
        self.step = 0
        self.finished = False
        self.history: list[C_StepResult] = []
        self.notices: list[str] = []
        self.refusals: list[str] = []
        self.parser = C_ObservationParser()
        self.log = log

    @property
    def m_remaining(self) -> int:
        return max(0, self.max_steps - self.step)

    def m_call(self, tool: str, args: dict) -> C_StepResult:
        if self.finished:
            raise C_EpisodeFinished("the episode has finished")
        if self.m_remaining <= 0:
            raise C_StepBudgetExhausted("no steps left")
        resp = None
        for _ in range(3):
            envelope = {**self.binding, "step": self.step + 1, "action": {"name": tool, "args": args}}
            try:
                resp = self.transport.m_call(envelope)
                break
            except C_EnvError as exc:
                detail = str(exc.detail)
                expected = re.search(r"expected (\d+)", detail)
                if exc.status == 409 and expected:
                    self.step = int(expected.group(1)) - 1
                    continue
                if exc.status == 409 and "finished" in detail:
                    self.finished = True
                    raise C_EpisodeFinished(detail) from exc
                result = C_StepResult(tool, args, self.step, error=f"http:{exc.status}", detail=detail)
                self.m_record(result)
                if exc.status == 402 or (exc.status == 409 and "quarantined" in detail):
                    self.finished = True
                return result
        if resp is None:
            result = C_StepResult(tool, args, self.step, error="http:409", detail="could not resync the step")
            self.m_record(result)
            return result
        self.step = int(resp.get("step") or self.step + 1)
        self.finished = bool(resp.get("finished"))
        payload, notices, refusal, error, detail = self.parser.m_parse(resp.get("observation"))
        result = C_StepResult(tool, args, self.step, payload, notices, refusal, error, detail, self.finished,
                              resp.get("observation"))
        self.notices.extend(n for n in notices if n not in self.notices)
        self.refusals.extend(refusal)
        self.m_record(result)
        return result

    def m_call_many(self, calls: list[tuple[str, dict]]) -> list[C_StepResult]:
        calls = list(calls)[: max(0, min(self.max_actions, self.m_remaining))]
        if len(calls) <= 1:
            return [self.m_call(*calls[0])] if calls else []
        if self.finished:
            raise C_EpisodeFinished("the episode has finished")
        resp = None
        for _ in range(3):
            envelope = {**self.binding, "step": self.step + 1,
                        "actions": [{"name": tool, "args": args} for tool, args in calls]}
            try:
                resp = self.transport.m_call(envelope)
                break
            except C_EnvError as exc:
                detail = str(exc.detail)
                expected = re.search(r"expected (\d+)", detail)
                if exc.status == 409 and expected:
                    self.step = int(expected.group(1)) - 1
                    continue
                if exc.status == 409 and "finished" in detail:
                    self.finished = True
                    raise C_EpisodeFinished(detail) from exc
                tool, args = calls[0]
                result = C_StepResult(tool, args, self.step, error=f"http:{exc.status}", detail=detail)
                self.m_record(result)
                if exc.status == 402 or (exc.status == 409 and "quarantined" in detail):
                    self.finished = True
                return [result]
        if resp is None:
            tool, args = calls[0]
            result = C_StepResult(tool, args, self.step, error="http:409", detail="could not resync the step")
            self.m_record(result)
            return [result]
        observations = resp.get("observations")
        if not isinstance(observations, list) or not observations:
            observations = [resp.get("observation")]
        start = self.step
        self.step = int(resp.get("step") or start + len(observations))
        self.finished = bool(resp.get("finished"))
        results = []
        for index, (observation, (tool, args)) in enumerate(zip(observations, calls)):
            payload, notices, refusal, error, detail = self.parser.m_parse(observation)
            last = index == len(observations) - 1
            result = C_StepResult(tool, args, start + index + 1, payload, notices, refusal, error, detail,
                                  self.finished and last, observation)
            self.notices.extend(n for n in notices if n not in self.notices)
            self.refusals.extend(refusal)
            self.m_record(result)
            results.append(result)
        return results

    def m_record(self, result: C_StepResult) -> None:
        self.history.append(result)
        self.log("step", result.m_to_log())


@dataclass
class C_Product:

    id: str
    title: str = ""
    price: float | None = None
    in_stock: bool | None = None
    category: str | None = None
    brand: str | None = None
    description: str = ""
    attrs: dict = field(default_factory=dict)
    seen_in: set = field(default_factory=set)
    detailed: bool = False

    def m_merge(self, other: "C_Product") -> None:
        for name in ("title", "price", "in_stock", "category", "brand", "description"):
            value = getattr(other, name)
            if value not in (None, ""):
                setattr(self, name, value)
        self.attrs.update(other.attrs)
        self.seen_in |= other.seen_in
        self.detailed = self.detailed or other.detailed

    def m_text(self) -> str:
        attrs = " ".join(f"{k} {v}" for k, v in self.attrs.items())
        return " ".join(str(x) for x in (self.title, self.category or "", self.brand or "", self.description, attrs))

    def m_brief(self, desc_chars: int = 220) -> dict:
        out = {"id": self.id, "title": self.title, "price": self.price, "in_stock": self.in_stock,
               "category": self.category, "brand": self.brand}
        if self.description:
            out["description"] = self.description[:desc_chars]
        if self.attrs:
            out["attributes"] = self.attrs
        return out


@dataclass
class C_SearchPage:

    products: list
    total: int | None
    next_page: Any
    ok: bool

    @property
    def m_count(self) -> int:
        return self.total if self.total is not None else len(self.products)


class C_ProductReader:
    ID_VALUE = re.compile(r"^[A-Za-z]{1,6}[-_:][A-Za-z0-9_-]{3,}$")
    ITEM_FIELDS = tuple(k for k, v in get_type_hints(T_ItemFields).items() if v is not StockMeaning)

    def __init__(self, namer: Callable[[Any], dict | None] | None = None) -> None:
        self.known_ids: set[str] = set()
        self.key_roles: dict[str, str] = {}
        self.page_roles: dict[str, str] = {}
        self.inverted_stock: set[str] = set()
        self.namer = namer
        self.asked: set[str] = set()

    def m_learn(self, payload: Any, rows: list[dict]) -> None:
        self.m_learn_shapes(rows)
        if isinstance(payload, dict) and rows:
            self.m_learn_page_shapes(payload)
        self.m_learn_names(payload, rows)

    def m_learn_names(self, payload: Any, rows: list[dict]) -> None:
        if self.namer is None or not rows:
            return
        row_keys = sorted({k for r in rows for k in r})
        top_keys = sorted(k for k, v in payload.items() if not isinstance(v, (dict, list))) \
            if isinstance(payload, dict) else []
        open_row = [k for k in row_keys if k not in self.key_roles]
        open_page = [k for k in top_keys if k not in self.page_roles and k not in row_keys]
        signature = gm_hash_32({"rows": row_keys, "top": top_keys})
        if signature in self.asked or not (open_row or open_page):
            return
        self.asked.add(signature)
        with GV_MEMO_LOCK:
            answer = GV_OUTPUT_FIELDS.get(signature)
        if answer is None:
            answer = self.namer(gm_reply_sample(payload), row_keys, top_keys) or {}
            with GV_MEMO_LOCK:
                GV_OUTPUT_FIELDS[signature] = answer
        placed = set(self.key_roles.values())
        item = answer.get("item") or {}
        first = rows[0]
        for field_name in self.ITEM_FIELDS:
            key = item.get(field_name)
            if isinstance(key, str) and key not in row_keys:
                holders = [k for k, v in first.items() if not isinstance(v, (dict, list))
                           and str(v).strip() == key.strip().rstrip("…")]
                key = holders[0] if len(holders) == 1 else None
            if not isinstance(key, str) or key not in row_keys:
                continue
            if key not in self.key_roles and field_name not in placed:
                self.key_roles[key] = field_name
                placed.add(field_name)
            if self.key_roles.get(key) == field_name == "stock" and item.get("stock_true_means") == "unavailable":
                flags = [r.get(key) for r in rows if isinstance(r.get(key), bool)]
                if len(flags) < 4 or sum(flags) < 0.75 * len(flags):
                    self.inverted_stock.add(key)
        page = answer.get("page") or {}
        for field_name in ("total", "next"):
            key = page.get(field_name)
            if isinstance(key, str) and key not in top_keys and isinstance(payload, dict):
                holders = [k for k in top_keys if str(payload.get(k)) == key.strip()]
                key = holders[0] if len(holders) == 1 else None
            if isinstance(key, str) and key in top_keys and key not in self.page_roles \
                    and field_name not in self.page_roles.values():
                self.page_roles[key] = field_name

    def m_learn_shapes(self, rows: list[dict]) -> None:
        if not rows:
            return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        values = {k: [r.get(k) for r in rows if r.get(k) is not None] for k in keys}
        placed = set(self.key_roles.values())

        def place(field_name: str, candidates: list[str]) -> None:
            if len(candidates) == 1 and candidates[0] not in self.key_roles and field_name not in placed:
                self.key_roles[candidates[0]] = field_name
                placed.add(field_name)

        def strings(k: str) -> list[str]:
            return [v for v in values[k] if isinstance(v, str)]

        place("id", [k for k in keys if strings(k) and len(strings(k)) == len(values[k]) == len(rows)
                     and len(set(strings(k))) == len(rows) and all(self.ID_VALUE.match(v) for v in strings(k))])
        place("stock", [k for k in keys if values[k] and all(isinstance(v, bool) for v in values[k])])
        for field_name in ("title", "category", "brand", "description"):
            place(field_name, [k for k in keys if k.lower() == field_name and strings(k)])
        place("price", [k for k in keys if values[k] and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                              for v in values[k])])

    def m_learn_page_shapes(self, payload: dict) -> None:
        scalars = {k: v for k, v in payload.items() if not isinstance(v, (dict, list))}
        ints = [k for k, v in scalars.items() if isinstance(v, int) and not isinstance(v, bool)]
        if "total" not in self.page_roles.values() and len(ints) == 1:
            self.page_roles[ints[0]] = "total"

    def m_find_rows(self, payload: Any) -> list[dict]:
        best: list[dict] = []
        stack = [gm_unpack_records(payload)]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                dicts = [x for x in item if isinstance(x, dict)]
                if dicts and len(dicts) == len(item) and any(self.m_id_of(d) for d in dicts) and len(dicts) > len(best):
                    best = dicts
                stack.extend(x for x in item if isinstance(x, (dict, list)))
            elif isinstance(item, dict):
                stack.extend(v for v in item.values() if isinstance(v, (dict, list)))
        return best

    def m_id_of(self, row: dict) -> str | None:
        for key, value in row.items():
            if self.key_roles.get(key) == "id" and isinstance(value, str) and value.strip():
                return value
        for value in row.values():
            if isinstance(value, str) and (value in self.known_ids or self.ID_VALUE.match(value)):
                return value
        return None

    def m_product(self, row: dict, label: str = "", detailed: bool = False) -> C_Product | None:
        pid = self.m_id_of(row)
        if not pid:
            return None
        p = C_Product(pid, seen_in={label} if label else set(), detailed=detailed)
        learned = set(self.key_roles.values())
        strings: list[tuple[str, str]] = []
        for key, value in row.items():
            if value == pid:
                continue
            role = self.key_roles.get(key)
            if role == "attributes" and isinstance(value, dict):
                p.attrs.update({str(a): b for a, b in value.items() if not isinstance(b, (dict, list))})
                continue
            if role is not None and not isinstance(value, (dict, list)):
                if role == "stock" and isinstance(value, bool):
                    p.in_stock = (not value) if key in self.inverted_stock else value
                elif role == "price" and isinstance(value, (int, float)) and not isinstance(value, bool):
                    p.price = float(value)
                elif role in ("title", "category", "brand", "description") and isinstance(value, str):
                    setattr(p, role, value)
                elif role != "id":
                    p.attrs[key] = value
                continue
            if isinstance(value, dict):
                p.attrs.update({str(a): b for a, b in value.items() if not isinstance(b, (dict, list))})
            elif isinstance(value, bool):
                if "stock" not in learned and p.in_stock is None:
                    p.in_stock = value
                else:
                    p.attrs[key] = value
            elif isinstance(value, (int, float)):
                if "price" not in learned and p.price is None:
                    p.price = float(value)
                else:
                    p.attrs[key] = value
            elif isinstance(value, str):
                strings.append((key, value))
            elif isinstance(value, list) and all(not isinstance(v, (dict, list)) for v in value):
                p.attrs[key] = value
        for key, value in sorted(strings, key=lambda kv: -len(kv[1])):
            if not p.description and "description" not in learned and len(value) > 80:
                p.description = value
            elif not p.title and "title" not in learned and len(value) <= 200:
                p.title = value
            else:
                p.attrs[key] = value
        self.known_ids.add(pid)
        return p

    def m_page(self, payload: Any, label: str) -> C_SearchPage:
        payload = gm_unpack_records(payload)
        rows = self.m_find_rows(payload)
        if rows:
            self.m_learn(payload, rows)
        products = [p for p in (self.m_product(r, label) for r in rows) if p is not None]
        total = next_page = None
        if isinstance(payload, dict):
            for key, value in payload.items():
                role = self.page_roles.get(key)
                if role == "total" and isinstance(value, int) and not isinstance(value, bool):
                    total = value
                elif role == "next" and value not in (None, "", False) \
                        and not (isinstance(value, str) and value in self.known_ids):
                    next_page = value
        ok = isinstance(payload, (dict, list)) and (bool(products) or total == 0 or isinstance(payload, list)
                                                    or (isinstance(payload, dict) and bool(payload)))
        return C_SearchPage(products, total, next_page, ok)

    def m_detail(self, payload: Any, pid: str) -> C_Product | None:
        payload = gm_unpack_records(payload)
        if not isinstance(payload, dict) or not payload:
            return None
        row = payload
        for key, value in payload.items():
            if value == pid and key not in self.key_roles and "id" not in self.key_roles.values():
                self.key_roles[key] = "id"
        if self.m_id_of(row) is None:
            rows = self.m_find_rows(payload)
            inner = [v for v in payload.values() if isinstance(v, dict) and self.m_id_of(v) == pid]
            row = next((r for r in rows if self.m_id_of(r) == pid), rows[0] if len(rows) == 1 else None)
            if row is None and len(inner) == 1:
                row = inner[0]
            if row is None:
                row = {**payload, "id": pid}
        self.m_learn_names(payload, [row])
        p = self.m_product(row, detailed=True)
        if p is not None and p.id != pid:
            p.id = pid
        return p


RE_RECORD_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]{0,40}): ")


def gm_record_scalar(text: str) -> Any:
    value = text.strip()
    if value in ("True", "False"):
        return value == "True"
    if value in ("None", "null", ""):
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def gm_is_record(text: Any) -> bool:
    return isinstance(text, str) and " | " in text and bool(RE_RECORD_KEY.match(text)) \
        and sum(1 for seg in text.split(" | ") if RE_RECORD_KEY.match(seg)) >= 2


def gm_record_dict(text: str) -> dict:
    """A listing rendered as 'key: value | key: value | facts: a=b; c=d' becomes a dict (facts nested)."""
    fields: list[tuple[str, str]] = []
    for segment in text.split(" | "):
        m = RE_RECORD_KEY.match(segment)
        if m:
            fields.append((m.group(1), segment[m.end():]))
        elif fields:
            key, prior = fields[-1]
            fields[-1] = (key, prior + " | " + segment)
    out: dict[str, Any] = {}
    for key, value in fields:
        if "=" in value and ";" in value or (key in ("facts", "attributes", "specs") and "=" in value):
            pairs = {}
            for part in value.split(";"):
                name, _, val = part.partition("=")
                if name.strip() and _:
                    pairs[name.strip()] = gm_record_scalar(val)
            out[key] = pairs if pairs else value
        else:
            out[key] = gm_record_scalar(value)
    return out


def gm_unpack_records(payload: Any) -> Any:
    if isinstance(payload, list):
        return [gm_record_dict(x) if gm_is_record(x) else gm_unpack_records(x) for x in payload]
    if isinstance(payload, dict):
        return {k: gm_record_dict(v) if gm_is_record(v) else gm_unpack_records(v) for k, v in payload.items()}
    return gm_record_dict(payload) if gm_is_record(payload) else payload


def gm_masked(value: Any, reader: "C_ProductReader") -> Any:
    if isinstance(value, dict):
        return {k: gm_masked(v, reader) for k, v in value.items()}
    if isinstance(value, list):
        return [gm_masked(v, reader) for v in value[:1]]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return 0
    if isinstance(value, str) and (value in reader.known_ids or reader.ID_VALUE.match(value)):
        return "<id>"
    if isinstance(value, str) and len(value) > 24:
        return "<text>"
    return value


def gm_reply_sample(value: Any) -> Any:
    if isinstance(value, list):
        return [gm_reply_sample(v) for v in value[:2]]
    if isinstance(value, dict):
        return {k: gm_reply_sample(v) for k, v in list(value.items())[:30]}
    if isinstance(value, str) and len(value) > 80:
        return value[:80] + "…"
    return value


class C_CallOutcomeReader(C_LLMTask):

    reply_type = T_CallOutcome
    SYSTEM = GP_CALL_OUTCOME

    def m_run(self, call: dict) -> dict | None:
        return self.m_ask(self.SYSTEM, json.dumps(call, ensure_ascii=False, default=str))


class C_OutputFieldReader(C_LLMTask):

    reply_type = T_OutputFields
    SYSTEM = GP_OUTPUT_FIELDS

    def m_run(self, sample: Any, row_keys: list[str], top_keys: list[str]) -> dict | None:
        choices = {f"item.{f}": row_keys for f in C_ProductReader.ITEM_FIELDS}
        choices.update({"page.total": top_keys or [""], "page.next": top_keys or [""]})
        schema = gm_with_choices(gm_schema_of(T_OutputFields), choices)
        return self.m_ask(self.SYSTEM, json.dumps(sample, ensure_ascii=False, default=str), schema)


class C_Catalog:

    def __init__(self) -> None:
        self.products: dict[str, C_Product] = {}

    def m_add(self, product: C_Product) -> C_Product:
        if product.id in self.products:
            self.products[product.id].m_merge(product)
        else:
            self.products[product.id] = product
        return self.products[product.id]

    def m_add_all(self, products: Iterable[C_Product]) -> None:
        for p in products:
            self.m_add(p)

    def m_get(self, pid: str) -> C_Product | None:
        return self.products.get(pid)

    def m_categories(self) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for p in self.products.values():
            if p.category:
                counts[p.category] = counts.get(p.category, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])

    def __len__(self) -> int:
        return len(self.products)


class C_ToolBox:

    FINISH_VALUES = {C_Role.ORDER: C_EnumValue.FINISH_ORDER, C_Role.DECLINE: C_EnumValue.FINISH_DECLINE}

    def __init__(self, env: C_Environment, specs: dict[str, C_ToolSpec], roles: C_RoleMap,
                 bindings: C_WorldBindings, reader: C_ProductReader) -> None:
        self.env = env
        self.specs = specs
        self.roles = roles
        self.bindings = bindings
        self.reader = reader
        self.compare_limit: int | None = None
        self.judge: Callable[[dict], dict | None] | None = None

    def m_has(self, role: C_Role) -> bool:
        return self.roles.m_has(role) or self.m_shared_finish(role) is not None

    def m_shared_finish(self, role: C_Role) -> C_ToolBinding | None:
        other = {C_Role.ORDER: C_Role.DECLINE, C_Role.DECLINE: C_Role.ORDER}.get(role)
        name = self.roles.m_tool_for(other) if other is not None else None
        binding = self.bindings.bindings.get(name) if name is not None else None
        if binding is None or C_Slot.FINISH not in binding.slots or self.FINISH_VALUES[role] not in binding.values:
            return None
        return binding

    def m_binding(self, role: C_Role, want: tuple = ()) -> C_ToolBinding | None:
        names = self.roles.m_by_role().get(role) or []
        if not names:
            return self.m_shared_finish(role)
        found = [self.bindings.bindings.get(n) or C_ToolBinding(n, role) for n in names]
        if want and len(found) > 1:
            # the first wanted slot is the essential one (free text, or the department filter)
            found.sort(key=lambda b: (want[0] not in b.slots, -sum(1 for s in want if s in b.slots)))
        return found[0]

    @staticmethod
    def sm_search_want(kw: dict) -> tuple:
        if kw.get("text") is not None:
            return (C_Slot.TEXT, C_Slot.PAGE, C_Slot.CURSOR)
        return (C_Slot.CATEGORY, C_Slot.MAX_PRICE, C_Slot.PAGE, C_Slot.CURSOR)

    @staticmethod
    def sm_fit(value: Any, schema: dict) -> Any:
        kind = C_Kind.cm_of(schema)
        if kind in (C_Kind.SHORT_TEXT, C_Kind.ID, C_Kind.LONG_TEXT):
            text = gm_normalize_space(str(value))
            max_len = schema.get("maxLength")
            if isinstance(max_len, int) and len(text) > max_len:
                cut = text[:max_len]
                text = cut.rsplit(" ", 1)[0] if " " in cut and kind is not C_Kind.ID else cut
            min_len = schema.get("minLength", 0)
            if len(text) < min_len:
                text = text.ljust(min_len, ".")
            return text
        if kind in (C_Kind.INT, C_Kind.NUMBER):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            if math.isnan(number) or math.isinf(number):
                return None
            lo, hi = schema.get("minimum"), schema.get("maximum")
            if isinstance(lo, (int, float)):
                number = max(number, lo)
            if isinstance(hi, (int, float)):
                number = min(number, hi)
            if kind is C_Kind.INT:
                return int(math.floor(number))
            return int(number) if float(number).is_integer() else round(number, 2)
        if kind is C_Kind.ENUM:
            values = C_ToolParam("", False, kind, schema).m_enum_values
            return value if value in values else None
        if kind is C_Kind.BOOL:
            return bool(value)
        return value

    def m_build_args(self, role: C_Role, values: dict[C_Slot, Any],
                     binding: C_ToolBinding | None = None) -> tuple[str, dict] | None:
        binding = binding or self.m_binding(role)
        if binding is None:
            return None
        spec = self.specs[binding.tool]
        args: dict[str, Any] = {}
        if role.m_is_final and C_Slot.FINISH in binding.slots:
            value = binding.values.get(self.FINISH_VALUES[role])
            if value is None and binding.role is not role:
                return None
            values = {**values, C_Slot.FINISH: value}
        for slot, value in values.items():
            pname = binding.slots.get(slot)
            param = spec.m_param(pname)
            if param is None or value is None:
                continue
            fitted = self.sm_fit(value, param.schema)
            if fitted is not None and fitted != "":
                args[param.name] = fitted
        for param in spec.params:
            if param.required and param.name not in args:
                if role is C_Role.DECLINE and len(param.m_enum_values) > 1 and not self.roles.m_has(C_Role.ORDER):
                    return None
                if param.m_enum_values:
                    args[param.name] = param.m_enum_values[0]
                elif param.kind in (C_Kind.INT, C_Kind.NUMBER):
                    args[param.name] = self.sm_fit(param.schema.get("minimum", 1), param.schema)
                elif param.kind in (C_Kind.SHORT_TEXT, C_Kind.ID, C_Kind.LONG_TEXT):
                    args[param.name] = self.sm_fit("item", param.schema)
        return binding.tool, args

    def m_call(self, role: C_Role, values: dict[C_Slot, Any]) -> C_StepResult | None:
        built = self.m_build_args(role, values)
        if built is None:
            return None
        tool, args = built
        return self.env.m_call(tool, args)

    def m_search_values(self, *, text: str | None = None, category: str | None = None, brand: str | None = None,
                        max_price: float | None = None, page: int = 1, cursor: Any = None,
                        label: str = "") -> dict[C_Slot, Any] | None:
        binding = self.m_binding(C_Role.SEARCH, self.sm_search_want({"text": text}))
        if binding is None:
            return None
        filter_mode = text is None
        if not filter_mode and C_Slot.TEXT not in binding.slots:
            return None
        if (page and page > 1 and C_Slot.PAGE not in binding.slots) or (cursor not in (None, "", False) and C_Slot.CURSOR not in binding.slots):
            return None
        if filter_mode and not ((category is not None and C_Slot.CATEGORY in binding.slots)
                                or (brand is not None and C_Slot.BRAND in binding.slots)
                                or (max_price is not None and C_Slot.MAX_PRICE in binding.slots)):
            return None
        values: dict[C_Slot, Any] = {}
        if text is not None:
            values[C_Slot.TEXT] = " ".join(dict.fromkeys(gm_get_words(text))) or text
        if category is not None:
            values[C_Slot.CATEGORY] = category
        if brand is not None:
            values[C_Slot.BRAND] = brand
        if max_price is not None and (filter_mode or C_Slot.MODE not in binding.slots):
            values[C_Slot.MAX_PRICE] = max_price
        if page and page > 1:
            values[C_Slot.PAGE] = page
        if cursor not in (None, "", False):
            values[C_Slot.CURSOR] = cursor
        if C_Slot.MODE in binding.slots:
            mode = binding.values.get(C_EnumValue.MODE_FILTER if filter_mode else C_EnumValue.MODE_TEXT)
            if mode is not None:
                values[C_Slot.MODE] = mode
        return values

    def m_search(self, **kw: Any) -> tuple[C_StepResult | None, C_SearchPage]:
        values = self.m_search_values(**kw)
        if values is None:
            return None, C_SearchPage([], None, None, False)
        built = self.m_build_args(C_Role.SEARCH, values, self.m_binding(C_Role.SEARCH, self.sm_search_want(kw)))
        result = self.env.m_call(*built) if built is not None else None
        if result is None or result.error:
            return result, C_SearchPage([], None, None, False)
        return result, self.reader.m_page(result.payload, kw.get("label", ""))

    def m_search_many(self, requests: list[dict]) -> list[tuple[C_StepResult | None, C_SearchPage]]:
        empty = C_SearchPage([], None, None, False)
        built = []
        for kw in requests:
            values = self.m_search_values(**kw)
            built.append(self.m_build_args(C_Role.SEARCH, values, self.m_binding(C_Role.SEARCH, self.sm_search_want(kw)))
                         if values is not None else None)
        sendable = [i for i, b in enumerate(built) if b is not None]
        results: dict[int, C_StepResult] = {}
        for start in range(0, len(sendable), self.env.max_actions):
            chunk = sendable[start:start + self.env.max_actions]
            for index, result in zip(chunk, self.env.m_call_many([built[i] for i in chunk])):
                results[index] = result
        out = []
        for index, kw in enumerate(requests):
            result = results.get(index)
            if result is None or result.error:
                out.append((result, empty))
            else:
                out.append((result, self.reader.m_page(result.payload, kw.get("label", ""))))
        return out

    def m_compare(self, ids: list[str]) -> list[C_Product]:
        binding = self.m_binding(C_Role.COMPARE)
        if binding is None or not ids:
            return []
        spec = self.specs[binding.tool]
        param = spec.m_param(binding.slots.get(C_Slot.IDS))
        if param is None:
            return []
        ids = list(ids)[: self.m_compare_batch_size()]
        result = self.env.m_call(binding.tool, {param.name: ids})
        rows = self.reader.m_find_rows(result.payload) if result is not None and not result.error else []
        if not rows and len(ids) > 1 and not self.env.finished:
            self.compare_limit = max(1, len(ids) // 2)
            return self.m_compare(ids[: self.compare_limit])
        out = []
        for row in rows:
            product = self.reader.m_product(row, detailed=True)
            if product is not None:
                out.append(product)
        return out

    def m_compare_batch_size(self) -> int:
        binding = self.m_binding(C_Role.COMPARE)
        if binding is None:
            return 0
        param = self.specs[binding.tool].m_param(binding.slots.get(C_Slot.IDS))
        size = param.schema.get("maxItems") if param is not None else None
        size = size if isinstance(size, int) and size > 0 else 4
        return min(size, self.compare_limit or size)

    def m_lookup(self, role: C_Role, pid: str) -> tuple[C_StepResult | None, C_Product | None]:
        result = self.m_call(role, {C_Slot.ID: pid})
        if result is None or result.error:
            return result, None
        return result, self.reader.m_detail(result.payload, pid)

    def m_lookup_many(self, role: C_Role, pids: list[str]) -> list[C_Product]:
        out: list[C_Product] = []
        pids = list(dict.fromkeys(pids))
        for start in range(0, len(pids), self.env.max_actions):
            chunk = pids[start:start + self.env.max_actions]
            built = [self.m_build_args(role, {C_Slot.ID: pid}) for pid in chunk]
            if not all(built):
                break
            for pid, result in zip(chunk, self.env.m_call_many(built)):
                if result is not None and not result.error:
                    product = self.reader.m_detail(result.payload, pid)
                    if product is not None:
                        out.append(product)
        return out

    def m_message_many(self, texts: list[str]) -> list[C_StepResult | None]:
        built = [self.m_build_args(C_Role.MESSAGE, {C_Slot.MESSAGE: t}) for t in texts]
        if not texts or not all(built):
            return [None] * len(texts)
        results = self.env.m_call_many(built)
        return list(results) + [None] * (len(texts) - len(results))

    def m_failed(self, result: C_StepResult | None) -> bool:
        if result is None or result.error:
            return True
        payload = result.payload
        if payload in ({}, None, "", []):
            return True
        if self.judge is None:
            return False
        shape = gm_hash_32({"tool": result.tool, "reply": gm_masked(payload, self.reader)})
        with GV_MEMO_LOCK:
            known = GV_CALL_OUTCOMES.get(shape)
        if known is None:
            answer = self.judge({"tool": result.tool, "arguments": result.args, "reply": gm_reply_sample(payload)})
            if not isinstance(answer, dict) or not isinstance(answer.get("succeeded"), bool):
                return False
            known = not answer["succeeded"]
            with GV_MEMO_LOCK:
                GV_CALL_OUTCOMES[shape] = known
        return known


@dataclass
class C_Requirement:

    text: str
    attribute: str
    value: str
    critical: bool = True
    variants: list = field(default_factory=list)
    kind: str = "property"
    relation: str = "equals"
    options: list = field(default_factory=list)
    figure: float | None = None
    unit: str = ""

    RELATIONS = get_args(RelationName)
    FIGURE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*([^\d\s,;]*)")

    @property
    def m_negative(self) -> bool:
        return self.relation == "excludes"

    @property
    def m_positive_value(self) -> str:
        return self.value

    @property
    def m_is_product_type(self) -> bool:
        return self.kind == "product_type"

    @property
    def m_is_property(self) -> bool:
        return self.kind in ("property", "product_type")

    @classmethod
    def cm_figure_of(cls, text: Any) -> tuple[float, str] | None:
        match = cls.FIGURE.search(str(text or ""))
        if not match:
            return None
        return float(match.group(1).replace(",", ".")), C_Canon.cm_unit(match.group(2))

    @property
    def m_bound(self) -> tuple[str, float, str] | None:
        if self.relation not in ("at_least", "at_most"):
            return None
        found = (self.figure, C_Canon.cm_unit(self.unit)) if self.figure is not None else self.cm_figure_of(self.value)
        if found is None:
            return None
        return ("min" if self.relation == "at_least" else "max"), found[0], found[1]

    def m_bounds(self) -> list[tuple[str, float, str]]:
        bound = self.m_bound
        if bound is None:
            return []
        found = [bound] + [(bound[0], *f) for f in (self.cm_figure_of(v) for v in self.variants) if f is not None]
        return list(dict.fromkeys(found))

    def m_alternatives(self) -> list[str]:
        if self.m_bound is not None:
            return [self.value]
        options = [o for o in self.options if isinstance(o, str) and o.strip()]
        return options if len(options) > 1 else [self.value]

    def m_forms(self) -> list[str]:
        out: list[str] = []
        variants = self.variants
        alternatives = self.m_alternatives()
        bare = RE_PARENTHESIS.sub("", self.m_positive_value).strip()
        if bare and bare != self.m_positive_value:
            alternatives = [bare, *alternatives]
        expanded = [f for alt in alternatives for f in C_VariantExpander.cm_expand(alt)]
        for form in [*alternatives, *variants, *expanded]:
            n = C_Canon.cm_key(form)
            if n and n not in out:
                out.append(n)
        return out

    def m_to_dict(self) -> dict:
        out = {"quote": self.text, "property_name": self.attribute, "relation": self.relation,
               "required_value": self.value, "is_mandatory": self.critical, "criterion_type": self.kind,
               "alternatives": self.m_forms()[1:12]}
        if len(self.m_alternatives()) > 1:
            out["options"] = self.m_alternatives()
        if self.m_bound is not None:
            out["figure"], out["unit"] = self.m_bound[1], self.m_bound[2]
        return out


class C_Canon:

    @staticmethod
    def sm_singular(word: str) -> str:
        if word.isdigit() or len(word) <= 3:
            return word
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith(("ches", "shes", "sses", "xes", "zes")):
            return word[:-2]
        if word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        if word.endswith("fe") and len(word) > 3:
            return word[:-2] + "ve"
        if word.endswith("f") and not word.endswith("ff") and len(word) > 3:
            return word[:-1] + "ve"
        return word

    @classmethod
    def cm_unit(cls, text: str) -> str:
        return cls.sm_singular(re.sub(r"[^a-z]", "", str(text or "").lower()))

    @staticmethod
    def sm_lowered(text: Any) -> str:
        t = str(text or "").lower()
        t = RE_THOUSANDS_COMMA.sub("", re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t))
        t = re.sub(r"(?<=\d)\s*(?:\*|×|by)\s*(?=\d)", "x", t)
        t = re.sub(r"(?<=\d)\s+x\s+(?=\d)", "x", t)
        return t

    @classmethod
    def cm_tokens(cls, text: Any) -> list[str]:
        out = [cls.sm_singular(raw) for raw in re.findall(r"[a-z0-9]+(?:[.x][0-9]+)*(?:x[0-9]+)*", cls.sm_lowered(text))]
        joined: list[str] = []
        for word in out:
            if joined and re.fullmatch(r"\d+(?:\.\d+)?", joined[-1]) and re.fullmatch(r"[a-z]{1,4}", word):
                joined[-1] += word
            else:
                joined.append(word)
        return joined

    @classmethod
    def cm_key(cls, text: Any) -> str:
        return " ".join(cls.cm_tokens(text))

    @staticmethod
    def sm_same_word(one: str, other: str) -> bool:
        if one == other:
            return True
        short, long = sorted((one, other), key=len)
        return len(short) >= 4 and not short[0].isdigit() and long.startswith(short) and len(long) - len(short) <= 2


class C_ProductText:

    def __init__(self, text: str) -> None:
        self.key = C_Canon.cm_key(text)
        self.compact = self.key.replace(" ", "")
        self.words = set(self.key.split())

    def m_has_phrase(self, form: str) -> bool:
        key = C_Canon.cm_key(form)
        if not key:
            return False
        if re.search(rf"(?<!\S){re.escape(key)}(?!\S)", self.key):
            return True
        compact = key.replace(" ", "")
        if len(compact) < 4 or compact not in self.compact:
            return False
        tokens = self.key.split()
        return any("".join(tokens[i:j]) == compact
                   for i in range(len(tokens)) for j in range(i + 1, min(len(tokens), i + 4) + 1))

    def m_match(self, form: str) -> float:
        key = C_Canon.cm_key(form)
        if not key:
            return 0.0
        if self.m_has_phrase(form):
            return 1.0
        words = [w for w in key.split() if len(w) > 1]
        if not words:
            return 0.0
        found = sum(1 for w in words if w in self.words or any(C_Canon.sm_same_word(w, o) for o in self.words))
        return found / len(words)


class C_VariantExpander:

    DIMENSION = re.compile(r"(\d+(?:\.\d+)?)\s*(?:x|\*|×|by)\s*(\d+(?:\.\d+)?)"
                           r"(?:\s*(?:x|\*|×|by)\s*(\d+(?:\.\d+)?))?", re.I)

    @classmethod
    def cm_expand(cls, value: str) -> list[str]:
        out: list[str] = []
        for match in cls.DIMENSION.finditer(str(value or "")):
            parts = [g for g in match.groups() if g]
            for sep in ("x", " x ", "*", " by ", "×"):
                out.append(sep.join(parts))
        return list(dict.fromkeys(out))


@dataclass
class C_Policy:

    price_cap: float | None = None
    quantity: int = 1
    cheapest: bool = False
    in_stock_only: bool = True
    message_budget: int | None = None
    step_budget: int | None = None
    product_type: str = ""
    no_substitute: bool = True
    peak_property: str = ""
    peak_high: bool = True
    open_at_least: int | None = None
    conduct: list = field(default_factory=list)
    opening_asks: list = field(default_factory=list)

    def m_to_dict(self) -> dict:
        return {
            "spend_limit": self.price_cap,
            "unit_count": self.quantity,
            "lowest_price_wanted": self.cheapest,
            "stock_only": self.in_stock_only,
            "contact_limit": self.message_budget,
            "action_limit": self.step_budget,
            "item_kind": self.product_type,
            "skip_alternatives": self.no_substitute,
            "ranking_metric": self.peak_property,
            "ranking_descending": self.peak_high,
            "inspection_count": self.open_at_least,
            "working_notes": list(self.conduct),
            "pre_pick_inquiries": list(self.opening_asks),
        }


class C_PhraseFinder(C_LLMTask):

    reply_type = T_SearchPhrases
    SYSTEM = GP_SEARCH_PHRASES

    def m_run(self, request: str) -> tuple[list[str], str]:
        answer = self.m_ask(self.SYSTEM, f"Brief:\n{request}")
        request_words = set(gm_get_words(request))
        request_stems = {C_Canon.sm_singular(r) for r in request_words}
        item_noun = answer.get("item_noun", "") if isinstance(answer, dict) else ""
        item_noun = gm_normalize_space(item_noun) if isinstance(item_noun, str) else ""
        type_words = {C_Canon.sm_singular(w) for w in gm_get_words(item_noun) if len(w) > 2}
        phrases: list[str] = []
        raw = answer.get("keyword_queries") if isinstance(answer, dict) else None
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, str):
                continue
            tokens = [w for w in gm_get_words(item) if w in request_words or C_Canon.sm_singular(w) in request_stems]
            phrase = " ".join(tokens)
            if type_words and not any(C_Canon.sm_singular(w) in type_words for w in phrase.split()):
                continue
            if len(phrase.split()) >= 2 and phrase not in phrases:
                phrases.append(phrase)
        order = {p: i for i, p in enumerate(phrases)}
        phrases.sort(key=lambda p: (len(p.split()), order[p]))
        return phrases[:GC_PHRASE_COUNT], item_noun


class C_PolicyReader(C_LLMTask):

    reply_type = T_PurchaseRules
    SYSTEM = GP_PURCHASE_RULES

    def m_run(self, request: str) -> C_Policy:
        answer = self.m_ask(self.SYSTEM, f"Brief:\n{request}") or {}
        policy = C_Policy()
        cap = answer.get("spend_limit")
        if isinstance(cap, (int, float)) and not isinstance(cap, bool) and cap > 0 and gm_request_has_number(request, cap):
            policy.price_cap = float(cap)
        qty = answer.get("unit_count")
        if isinstance(qty, int) and not isinstance(qty, bool) and 1 <= qty <= 20:
            words = answer.get("unit_count_quote") if isinstance(answer.get("unit_count_quote"), str) else ""
            policy.quantity = qty if qty == 1 or gm_request_has_count(request, qty, words) else 1
        policy.cheapest = bool(answer.get("lowest_price_wanted"))
        stock = answer.get("stock_only")
        policy.in_stock_only = True if stock is None else bool(stock)
        for field_name, key in (("message_budget", "contact_limit"), ("step_budget", "action_limit")):
            value = answer.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0  \
                   and gm_request_has_number(request, value):
                setattr(policy, field_name, value)
        ptype = answer.get("item_kind")
        policy.product_type = gm_normalize_space(ptype) if isinstance(ptype, str) else ""
        policy.no_substitute = bool(answer.get("skip_alternatives", True))
        measure = answer.get("ranking_metric")
        said = answer.get("ranking_quote") if isinstance(answer.get("ranking_quote"), str) else ""
        if isinstance(measure, str) and measure.strip() and not policy.cheapest and said.strip() \
               and gm_normalize_space(said).lower() in gm_normalize_space(request).lower():
            policy.peak_property = gm_normalize_space(measure)[:60]
            policy.peak_high = answer.get("ranking_descending") is not False
        reads = answer.get("inspection_count")
        if isinstance(reads, int) and not isinstance(reads, bool) and 1 < reads <= 12 \
               and gm_request_has_count(request, reads, answer.get("inspection_quote")
                                         if isinstance(answer.get("inspection_quote"), str) else ""):
            policy.open_at_least = reads
        flat = gm_normalize_space(request).lower()
        policy.conduct = [gm_normalize_space(r) for r in answer.get("working_notes") or []
                          if isinstance(r, str) and r.strip() and gm_normalize_space(r).lower() in flat][:6]
        for item in answer.get("pre_pick_inquiries") or []:
            if not isinstance(item, dict) or not isinstance(item.get("inquiry"), str):
                continue
            wishes = [gm_normalize_space(w) for w in item.get("choices") or [] if isinstance(w, str) and w.strip()]
            if item.get("kind") == "budget":
                usable = policy.price_cap is None
                topic = "budget"
            else:
                usable = len(wishes) == 2 and all(w.lower() in flat for w in wishes)
                topic = " or ".join(wishes)[:60]
            question = C_QuestionWriter.sm_clip(item["inquiry"]) if usable else ""
            if question:
                policy.opening_asks.append({"topic": topic, "question": question})
        policy.opening_asks = policy.opening_asks[:2]
        return policy


class C_RequirementReader(C_LLMTask):

    reasoning = GC_REQUIREMENTS_REASONING

    reply_type = T_Requirements
    SYSTEM = GP_REQUIREMENTS
    KINDS = {"product_type", "property", "logistics", "quantity", "rule"}

    @staticmethod
    def sm_value_from_request(req: C_Requirement, request: str) -> bool:
        if req.m_bound is not None:
            return gm_request_has_number(request, req.m_bound[1])
        said = {C_Canon.sm_singular(w) for w in gm_get_words(request)}
        words = [C_Canon.sm_singular(w) for w in gm_get_words(req.value) if len(w) > 1]
        if not words:
            return False
        found = sum(1 for w in words if w in said or any(C_Canon.sm_same_word(w, o) for o in said))
        return found / len(words) >= 0.6

    def m_run(self, request: str) -> list[C_Requirement]:
        answer = self.m_ask(self.SYSTEM, f"Brief:\n{request}") or {}
        raw = answer.get("criteria")
        flat = gm_normalize_space(request).lower()
        out: list[C_Requirement] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            text = gm_normalize_space(item.get("quote", "")) if isinstance(item.get("quote"), str) else ""
            attribute = gm_normalize_space(item.get("property_name", ""))  \
               if isinstance(item.get("property_name"), str) else ""
            kind = item.get("criterion_type") if item.get("criterion_type") in self.KINDS else "property"
            if kind == "rule":
                continue
            req = gm_criterion(item, "required_value", text, attribute, bool(item.get("is_mandatory", True)), kind)
            if req is None:
                continue
            if not text or text.lower() not in flat:
                if req.value.lower() in flat:
                    req.text = req.value
                else:
                    continue
            if not self.sm_value_from_request(req, request):
                continue
            if kind == "property" and not req.m_negative and self.cm_restates_attribute(req.value, req.attribute):
                continue
            out.append(req)
            if len(out) >= 12:
                break
        return self.cm_drop_contradicted_exclusions(out)[:12]

    @staticmethod
    def sm_related(one: str, other: str) -> bool:
        short, long = sorted((one, other), key=len)
        return len(short) >= 3 and long.startswith(short)

    @classmethod
    def cm_restates_attribute(cls, value: str, attribute: str) -> bool:
        values = [C_Canon.sm_singular(w) for w in gm_get_words(value) if len(w) > 2]
        names = [C_Canon.sm_singular(w) for w in gm_get_words(attribute) if len(w) > 2]
        return bool(values) and bool(names) and all(any(cls.sm_related(v, n) for n in names) for v in values)

    @classmethod
    def cm_drop_contradicted_exclusions(cls, reqs: list[C_Requirement]) -> list[C_Requirement]:
        wanted = [C_Canon.sm_singular(w) for r in reqs if not r.m_negative and r.kind == "property"
                  for w in gm_get_words(r.m_positive_value) if len(w) > 2]
        out = []
        for r in reqs:
            if r.m_negative:
                excluded = [C_Canon.sm_singular(w) for w in gm_get_words(r.m_positive_value) if len(w) > 3]
                if excluded and all(any(w.startswith(e) for w in wanted) for e in excluded):
                    continue
            out.append(r)
        return out


@dataclass
class C_Probe:

    phrase: str
    page: C_SearchPage
    step: int


class C_CategoryPicker(C_LLMTask):

    reply_type = T_Departments
    SYSTEM = GP_DEPARTMENTS

    def m_run(self, request: str, product_type: str, categories: list[tuple[str, int]]) -> list[str]:
        if not categories:
            return []
        listing = "\n".join(f"- {name}  ({count} seen)" for name, count in categories[:80])
        answer = self.m_ask(self.SYSTEM.replace("{limit}", str(GC_CATEGORY_LIMIT)),
                            f"Brief:\n{request}\n\nItem: {product_type}\n\nDepartments:\n{listing}") or {}
        known = {name for name, _ in categories}
        lower = {name.lower(): name for name in known}
        out: list[str] = []
        for item in answer.get("department_names") or []:
            if not isinstance(item, str):
                continue
            name = item if item in known else lower.get(item.strip().lower())
            if name and name not in out:
                out.append(name)
        if not out:
            nouns = {w.rstrip("s") for w in gm_get_words(product_type) if len(w) > 2}
            out = [n for n, _ in categories if nouns & {w.rstrip("s") for w in gm_get_words(n)}][:2]  \
               or [categories[0][0]]
        return out[:GC_CATEGORY_LIMIT]


class C_SearchPlanner:

    def __init__(self, runner: "C_EpisodeRunner") -> None:
        self.runner = runner
        self.probes: list[C_Probe] = []
        self.tokens_used: set = set()

    def m_search(self, label: str, **kw: Any) -> C_SearchPage:
        _, page = self.runner.tools.m_search(label=label, **kw)
        self.runner.catalog.m_add_all(page.products)
        return page

    def m_search_many(self, requests: list[dict]) -> list[C_SearchPage]:
        pages = []
        for _, page in self.runner.tools.m_search_many(requests):
            self.runner.catalog.m_add_all(page.products)
            pages.append(page)
        return pages

    def m_numbered_pages(self, want: tuple = ()) -> bool:
        binding = self.runner.tools.m_binding(C_Role.SEARCH, want)
        return binding is not None and C_Slot.PAGE in binding.slots

    def m_next_page_args(self, previous: C_SearchPage, page_no: int, want: tuple = ()) -> dict | None:
        binding = self.runner.tools.m_binding(C_Role.SEARCH, want)
        if binding is None:
            return None
        if C_Slot.PAGE in binding.slots:
            return {"page": page_no}
        token = previous.next_page
        if C_Slot.CURSOR in binding.slots and token not in (None, "", False, True):
            if token in self.tokens_used:
                return None
            self.tokens_used.add(token)
            return {"cursor": token}
        return None

    def m_choose_value_queries(self, phrases: list[str], max_steps: int) -> list[C_Probe]:
        batch = phrases[:max_steps]
        while batch and not self.runner.m_can_spend(len(batch)):
            batch = batch[:-1]
        results: list[C_Probe] = []
        pages = self.m_search_many([{"label": f"q:{phrase}", "text": phrase} for phrase in batch])
        for phrase, page in zip(batch, pages):
            probe = C_Probe(phrase, page, self.runner.env.step)
            self.probes.append(probe)
            results.append(probe)
        window = self.runner.m_visible_window()
        useful = [p for p in results if p.page.m_count > 0]
        useful.sort(key=lambda p: (abs(p.page.m_count - window), -len(p.phrase.split())))
        return useful[:2]

    def m_first_hit(self, phrases: list[str], max_steps: int) -> list[C_Probe]:
        for phrase in phrases[:max_steps]:
            if not self.runner.m_can_spend(1):
                break
            page = self.m_search(f"q:{phrase}", text=phrase)
            probe = C_Probe(phrase, page, self.runner.env.step)
            self.probes.append(probe)
            if page.m_count > 0:
                return [probe]
        return []

    def m_choose_queries(self, phrases: list[str], max_steps: int) -> list[C_Probe]:
        target = self.runner.m_visible_window()
        if not phrases:
            return []
        results: dict[int, C_Probe] = {}

        def probe(i: int) -> C_Probe | None:
            if i in results:
                return results[i]
            if not self.runner.m_can_spend(1):
                return None
            page = self.m_search(f"q:{phrases[i]}", text=phrases[i])
            results[i] = C_Probe(phrases[i], page, self.runner.env.step)
            self.probes.append(results[i])
            return results[i]

        def count(p: C_Probe | None) -> int:
            return p.page.m_count if p is not None else -1

        lo, hi = 0, len(phrases) - 1
        first = probe(lo)
        steps = 1
        if first is not None and count(first) > target and hi > lo and steps < max_steps:
            last = probe(hi)
            steps += 1
            if last is not None and count(last) > target:
                lo = hi
            else:
                while hi - lo > 1 and steps < max_steps:
                    mid = (lo + hi) // 2
                    p = probe(mid)
                    steps += 1
                    if p is None:
                        break
                    if count(p) > target:
                        lo = mid
                    else:
                        hi = mid
        elif first is not None and count(first) == 0 and len(phrases) > 1 and steps < max_steps:
            probe(1)

        def score(p: C_Probe) -> tuple:
            c = p.page.m_count
            return (c == 0, abs(c - target), -len(p.phrase.split()))

        ranked = sorted(results.values(), key=score)
        return [p for p in ranked if p.page.m_count > 0][:2]

    def m_fetch_more_pages(self, chosen: list[C_Probe], max_steps: int) -> None:
        if self.m_numbered_pages():
            wanted = []
            for p in chosen:
                for page_no in (2, 3):
                    if self.sm_has_more(p.page, page_no - 1):
                        wanted.append({"label": f"q:{p.phrase}", "text": p.phrase, "page": page_no})
            spend = min(len(wanted), max_steps, max(0, self.runner.env.m_remaining - GC_FINISH_RESERVE_STEPS))
            if spend > 0:
                self.m_search_many(wanted[:spend])
            return
        spent = 0
        for p in chosen:
            page_no = 1
            page = p.page
            while spent < max_steps and self.sm_has_more(page, page_no) and self.runner.m_can_spend(1):
                page_no += 1
                more = self.m_next_page_args(page, page_no)
                if more is None:
                    break
                page = self.m_search(f"q:{p.phrase}", text=p.phrase, **more)
                spent += 1
                if not page.products:
                    break

    @staticmethod
    def sm_has_more(page: C_SearchPage, page_no: int) -> bool:
        if page_no >= 3:
            return False
        if page.next_page is not None and page.next_page is not False:
            return True
        if page.total is not None:
            return page.total > page_no * max(1, len(page.products) or 7)
        return False

    def m_category_search(self, categories: list[str], cap: float | None, max_steps: int, cheapest: bool) -> None:
        by_category = (C_Slot.CATEGORY, C_Slot.MAX_PRICE, C_Slot.PAGE, C_Slot.CURSOR)
        binding = self.runner.tools.m_binding(C_Role.SEARCH, by_category)
        if binding is None or C_Slot.CATEGORY not in binding.slots:
            self.runner.m_note("search has no department filter; using more phrase pages instead")
            return
        if self.m_numbered_pages(by_category):
            room = min(max_steps, max(0, self.runner.env.m_remaining - GC_FINISH_RESERVE_STEPS))
            firsts = [{"label": f"c:{c}", "category": c, "max_price": cap} for c in categories][:room]
            pages = self.m_search_many(firsts)
            room -= len(firsts)
            more = []
            for category, page in zip(categories, pages):
                too_many = bool(page.total and page.total > self.runner.m_visible_window())
                if too_many and page.m_count > 3 * self.runner.m_visible_window():
                    continue
                for page_no in (2, 3):
                    if self.sm_has_more(page, page_no - 1):
                        more.append({"label": f"c:{category}", "category": category, "max_price": cap,
                                     "page": page_no})
            if room > 0 and more:
                self.m_search_many(more[:room])
            return
        spent = 0
        queue: list[tuple[str, float | None]] = [(c, cap) for c in categories]
        while queue and spent < max_steps:
            category, price_cap = queue.pop(0)
            page_no = 1
            page = self.m_search(f"c:{category}", category=category, max_price=price_cap)
            spent += 1
            too_many = bool(page.total and page.total > self.runner.m_visible_window())
            if too_many and page.m_count > 3 * self.runner.m_visible_window():
                continue
            while spent < max_steps and self.sm_has_more(page, page_no) and self.runner.m_can_spend(1):
                page_no += 1
                more = self.m_next_page_args(page, page_no, by_category)
                if more is None:
                    break
                page = self.m_search(f"c:{category}", category=category, max_price=price_cap, **more)
                spent += 1

    def m_narrow_price(self, category: str, cap: float, total: int, budget: int, text: str | None = None) -> int:
        target = self.runner.m_visible_window()
        hi, hi_total = cap, total
        lo, lo_total = 0.0, 0
        spent = 0
        while spent < budget and hi - lo > 1 and self.runner.m_can_spend(1):
            share = (0.7 * target - lo_total) / max(1, hi_total - lo_total)
            guess = math.floor(lo + (hi - lo) * min(0.9, max(0.1, share)))
            if not lo < guess < hi:
                break
            extra = {"text": text} if text else {}
            page = self.m_search(f"c:{category}", category=category, max_price=float(guess), **extra)
            spent += 1
            found = page.total if page.total is not None else len(page.products)
            if found > target:
                hi, hi_total = guess, found
            elif found < 0.3 * target and lo_total < found < hi_total or found == 0:
                lo, lo_total = guess, found
            else:
                page_no = 1
                while spent < budget and self.sm_has_more(page, page_no) and self.runner.m_can_spend(1):
                    page_no += 1
                    more = self.m_next_page_args(page, page_no)
                    if more is None:
                        break
                    page = self.m_search(f"c:{category}", category=category, max_price=float(guess), **more, **extra)
                    spent += 1
                break
        return spent


class C_CompletenessChecker:

    def __init__(self, requirements: list[C_Requirement], categories: list[str], product_type: str,
                 products: Iterable[C_Product] = ()) -> None:
        products = list(products)
        self.requirements = [r for r in requirements if r.critical and not r.m_negative and r.m_is_property]
        noun = gm_get_words(product_type)[-1:] if product_type else []
        if noun and not any(r.m_is_product_type for r in self.requirements):
            self.requirements.insert(0, C_Requirement(product_type, "product type", noun[0], True, [], "product_type"))
        self.categories = set(categories)
        self.top_category = categories[0] if categories else None
        self.noun_words = {C_Canon.sm_singular(w) for w in gm_get_words(product_type) if len(w) > 2}
        self.facts = C_FactMatcher(gm_get_words(product_type), products)
        self.unverifiable = {id(r) for r in self.requirements
                             if products and not self.sm_is_type(r) and not self.m_verifiable(r, products)}
        detailed = [p for p in products if p.detailed and p.attrs]
        self.keyed = {id(r) for r in self.requirements if not self.sm_is_type(r) and len(detailed) >= 3
                      and sum(any(self.facts.m_key_matches(r.attribute, k) for k in p.attrs) for p in detailed)
                      >= 0.5 * len(detailed)}

    def m_verifiable(self, req: C_Requirement, products: list[C_Product]) -> bool:
        forms = req.m_forms()
        for p in products:
            if any(self.facts.m_key_matches(req.attribute, k) for k in p.attrs):
                return True
            if any(C_ProductText(f"{p.title} ; {p.description}").m_has_phrase(f) for f in forms[:4]):
                return True
        return False

    @staticmethod
    def sm_is_type(req: C_Requirement) -> bool:
        return req.m_is_product_type

    def m_status(self, req: C_Requirement, p: C_Product) -> str:
        forms = req.m_forms()
        if self.sm_is_type(req):
            if p.category and self.categories:
                if p.category not in self.categories:
                    return "unmet"
                cat = C_ProductText(p.category)
                if any(cat.m_has_phrase(f) for f in forms):
                    return "met"
                if p.category != self.top_category:
                    return "unmet"
                fact = self.facts.m_check(req, p)
                if fact != C_FactMatcher.UNKNOWN:
                    return "met" if fact == C_FactMatcher.CONFIRMED else "unmet"
                text = C_ProductText(f"{p.title} ; {p.description}")
                words = {C_Canon.sm_singular(w) for f in forms for w in f.split() if len(w) > 2}
                return "met" if any(text.m_has_phrase(f) for f in forms) or words & self.noun_words & text.words  \
                   else "unknown"
            text = C_ProductText(f"{p.title} ; {p.category or ''}")
            words = {C_Canon.sm_singular(w) for f in forms for w in f.split() if len(w) > 2} | self.noun_words
            return "met" if any(text.m_has_phrase(f) for f in forms) or words & text.words else "unmet"
        fact = self.facts.m_check(req, p)
        if fact == C_FactMatcher.CONFIRMED:
            return "met"
        if fact == C_FactMatcher.CONTRADICTED:
            return "unmet"
        if p.category and any(C_ProductText(p.category).m_has_phrase(f) for f in forms):
            return "met"
        if id(req) in self.keyed:
            return "unknown"
        title = C_ProductText(f"{p.title} ; {p.category or ''}")
        text = C_ProductText(f"{p.title} ; {p.category or ''} ; {p.description}")
        for f in forms:
            source = title if len(f.split()) == 1 else text
            if req.m_bound is None and source.m_has_phrase(f):
                return "met"
        return "unknown"

    def m_exact_count(self, p: C_Product | None) -> int:
        if p is None:
            return 0
        return sum(self.facts.m_check(replace(r, variants=[]), p) == C_FactMatcher.CONFIRMED
                   for r in self.requirements if not self.sm_is_type(r))

    def m_verdict(self, p: C_Product) -> dict:
        statuses = {f"{r.attribute}: {r.value}": ("unverifiable" if id(r) in self.unverifiable else self.m_status(r, p))
                    for r in self.requirements}
        statuses_checked = {k: v for k, v in statuses.items() if v != "unverifiable"}
        return {"complete": bool(statuses_checked) and all(v == "met" for v in statuses_checked.values()),
                "requirements": statuses}


class C_ExclusionFilter:

    def __init__(self, requirements: list[C_Requirement]) -> None:
        self.rules = [r for r in requirements if r.m_negative]
        self.cache: dict[str, tuple[tuple, str | None]] = {}

    def __bool__(self) -> bool:
        return bool(self.rules)

    @staticmethod
    def sm_evidence(p: C_Product) -> str:
        values = " ; ".join(str(v) for v in p.attrs.values() if not isinstance(v, (dict, list)))
        return " ; ".join(x for x in (p.title, p.category or "", values) if x)

    def m_reason(self, p: C_Product) -> str | None:
        if not self.rules:
            return None
        stamp = (p.title, p.category, len(p.attrs))
        cached = self.cache.get(p.id)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        text = C_ProductText(self.sm_evidence(p))
        found = None
        for rule in self.rules:
            hit = next((f for f in rule.m_forms() if text.m_has_phrase(f)), None)
            if hit is not None:
                found = f"{rule.text!r} excludes it (has {hit!r})"
                break
        self.cache[p.id] = (stamp, found)
        return found

    def m_excluded(self, p: C_Product) -> bool:
        return self.m_reason(p) is not None


class C_FactMatcher:

    CONFIRMED, CONTRADICTED, UNKNOWN = "confirmed", "contradicted", "unknown"
    LIST_ENTRIES = 30
    SPLIT = re.compile(r"\s*(?:,|;|\||\s/\s)\s*")

    def __init__(self, product_words: Iterable[str] = (), products: Iterable[C_Product] = ()) -> None:
        self.product_words = {C_Canon.sm_singular(w) for w in product_words}
        self.domains: dict[frozenset, set[str]] = {}
        for p in products:
            for key, value in (p.attrs or {}).items():
                if isinstance(value, (dict, list)):
                    continue
                label = frozenset(self.cm_key_words(key))
                if label:
                    entries = [C_Canon.cm_key(e) for e in self.SPLIT.split(str(value)) if e.strip()]
                    self.domains.setdefault(label, set()).update(e for e in entries if e)

    @staticmethod
    def sm_ordered_words(text: str) -> list[str]:
        return [w for w in C_Canon.cm_tokens(RE_ATTRIBUTE_SEPARATOR.sub(" ", str(text))) if len(w) > 2]

    @classmethod
    def cm_key_words(cls, text: str) -> set[str]:
        return set(cls.sm_ordered_words(text))

    def m_key_matches(self, attribute: str, fact_key: str) -> bool:
        parts = [a for a in re.split(r"\s*/\s*|\s+or\s+", str(attribute)) if a.strip()]
        if len(parts) > 1:
            return any(self.m_key_matches(a, fact_key) for a in parts)
        ordered = self.sm_ordered_words(attribute)
        want, have = set(ordered), self.cm_key_words(fact_key)
        if not want or not have:
            return False
        if want <= have:
            return True
        return have <= want and ordered[0] in have

    @staticmethod
    def sm_distinctive(form: str) -> bool:
        letters = re.sub(r"[^a-z]", "", form)
        return len(letters) >= 4

    @staticmethod
    def sm_typo(one: str, other: str) -> bool:
        if len(one) < 5 or len(one) != len(other) or one.isdigit():
            return False
        diff = [i for i, (a, b) in enumerate(zip(one, other)) if a != b]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and one[diff[0]] == other[diff[1]] and one[diff[1]] == other[diff[0]])

    def m_entry_matches(self, form: str, entry: str, attribute_words: set[str] | None = None) -> bool:
        full, have = form.split(), C_Canon.cm_key(entry).split()
        trimmed = [w for w in full if w not in (attribute_words or ())]
        return any(self.m_words_match(want, have) for want in ({tuple(full), tuple(trimmed or full)}))

    def m_words_match(self, want: tuple, have: list) -> bool:
        want = list(want)
        if not want or not have:
            return False
        if want == have:
            return True
        if len(have) < len(want) and set(have) <= set(want) and want[-1] in have:
            return True
        if len(have) > len(want) and set(want) <= set(have) and have[-1] == want[-1]:
            return True
        extra = [w for w in have if w not in want]
        if all(w in have for w in want) and extra and all(w in self.product_words for w in extra):
            return True
        return len(want) == len(have) and all(a == b or self.sm_typo(a, b) for a, b in zip(want, have))

    def m_value_matches(self, forms: list[str], fact_value: Any, attribute_words: set[str] | None = None) -> bool:
        entries = [e for e in self.SPLIT.split(str(fact_value)) if e.strip()]
        if not entries or len(entries) > self.LIST_ENTRIES:
            return False
        return any(self.m_entry_matches(form, entry, attribute_words) for form in forms for entry in entries)

    def m_owns(self, label: frozenset, forms: list[str], attribute_words: set[str]) -> bool:
        return any(self.m_entry_matches(f, e, attribute_words) for f in forms for e in self.domains.get(label, ()))

    @staticmethod
    def sm_quantities(value: Any) -> list[tuple[float, str]]:
        out = []
        for match in re.finditer(r"(\d+(?:[.,]\d+)?)\s*([a-z\"]{0,8})", C_Canon.sm_lowered(value)):
            out.append((float(match.group(1).replace(",", ".")), C_Canon.cm_unit(match.group(2))))
        return out

    def m_check_bound(self, req: C_Requirement, facts: dict[str, Any], keyed: list[str]) -> str:
        bounds = req.m_bounds()
        verdicts = []
        for key in keyed:
            for number, unit in self.sm_quantities(facts[key]):
                for op, limit, bound_unit in bounds:
                    if unit and bound_unit and unit != bound_unit:
                        continue
                    verdicts.append(number <= limit + 1e-9 if op == "max" else number >= limit - 1e-9)
        if not verdicts:
            return self.UNKNOWN
        return self.CONFIRMED if any(verdicts) else self.CONTRADICTED

    def m_check(self, req: C_Requirement, p: C_Product) -> str:
        if not p.detailed or not p.attrs or req.m_negative:
            return self.UNKNOWN
        forms = req.m_forms()
        facts = {k: v for k, v in p.attrs.items() if not isinstance(v, (dict, list))}
        quoted = set(C_Canon.cm_tokens(req.text))
        keyed = [k for k in facts if self.m_key_matches(req.attribute, k)
                 or (self.cm_key_words(k) and len(self.cm_key_words(k)) >= 2 and self.cm_key_words(k) <= quoted)]
        bare = self.cm_key_words(req.attribute)
        if req.m_bound is not None:
            return self.m_check_bound(req, facts, keyed)
        if not keyed:
            related = [k for k in facts if bare & self.cm_key_words(k)]
            if any(self.m_value_matches(forms, facts[k], self.cm_key_words(req.attribute)) for k in related):
                return self.CONFIRMED
            if not related and any(self.sm_distinctive(f) for f in forms) \
                   and any(self.m_value_matches([f for f in forms if self.sm_distinctive(f)], v)
                            for v in facts.values()):
                return self.CONFIRMED
            return self.UNKNOWN
        attribute_words = self.cm_key_words(req.attribute)
        groups: dict[frozenset, bool] = {}
        for k in keyed:
            label = frozenset(self.cm_key_words(k))
            groups[label] = groups.get(label, False) or self.m_value_matches(forms, facts[k], attribute_words)
        if all(groups.values()):
            return self.CONFIRMED
        if not any(groups.values()):
            return self.CONTRADICTED
        owners = {label: ok for label, ok in groups.items() if self.m_owns(label, forms, attribute_words)}
        owned = list(owners.values())
        if owned and (all(owned) or not any(owned)):
            return self.CONFIRMED if all(owned) else self.CONTRADICTED
        specific = [ok for label, ok in (owners or groups).items() if label & self.product_words]
        if specific and (all(specific) or not any(specific)):
            return self.CONFIRMED if all(specific) else self.CONTRADICTED
        return self.UNKNOWN


class C_CandidateScorer:

    def __init__(self, policy: C_Policy, requirements: list[C_Requirement], categories: list[str]) -> None:
        self.policy = policy
        self.requirements = [r for r in requirements if not r.m_negative]
        self.exclusions = C_ExclusionFilter(requirements)
        self.categories = set(categories)
        self.text_cache: dict[str, tuple] = {}

    def m_passes_hard_rules(self, p: C_Product) -> bool:
        if self.policy.in_stock_only and p.in_stock is False:
            return False
        if self.policy.price_cap is not None and p.price is not None \
               and p.price * self.policy.quantity > self.policy.price_cap + 1e-9:
            return False
        return not self.exclusions.m_excluded(p)

    def m_texts(self, p: C_Product) -> tuple[C_ProductText, C_ProductText]:
        stamp = (p.id, p.title, p.description, len(p.attrs))
        cached = self.text_cache.get(p.id)
        if cached is None or cached[0] != stamp:
            cached = (stamp, C_ProductText(p.m_text()), C_ProductText(p.title))
            self.text_cache[p.id] = cached
        return cached[1], cached[2]

    def m_requirement_hit(self, req: C_Requirement, p: C_Product) -> tuple[float, float]:
        text, title = self.m_texts(p)
        forms = req.m_forms()
        return (max((text.m_match(f) for f in forms), default=0.0),
                max((title.m_match(f) for f in forms), default=0.0))

    def m_score(self, p: C_Product) -> float:
        total = 0.0
        for req in self.requirements:
            hit, in_title = self.m_requirement_hit(req, p)
            weight = 2.0 if req.critical else 1.0
            total += weight * hit + 0.5 * in_title
        if p.category and p.category in self.categories:
            total += 1.5
        total += 0.2 * len(p.seen_in)
        return total

    def m_rank(self, products: Iterable[C_Product]) -> list[C_Product]:
        eligible = [p for p in products if self.m_passes_hard_rules(p)]
        return sorted(eligible, key=lambda p: (-self.m_score(p), p.price if p.price is not None else 1e18))


class C_Shortlister(C_LLMTask):

    reply_type = T_Shortlist
    SYSTEM = GP_SHORTLIST

    def m_run(self, request: str, requirements: list[C_Requirement], products: list[C_Product], n: int,
              cheapest: bool) -> list[str]:
        if not products:
            return []
        cheap = ("The buyer wants the cheapest fitting option: among equally fitting items favour the cheapest. "
                 if cheapest else "")
        user = json.dumps({"brief": request, "criteria": [r.m_to_dict() for r in requirements],
                           "items": [p.m_brief(160) for p in products]}, ensure_ascii=False)
        system = self.SYSTEM.replace("{n}", str(n)).replace("{cheap}", cheap)
        answer = self.m_ask(system, user) or {}
        known = {p.id for p in products}
        ids = [i for i in answer.get("shortlisted_ids") or [] if isinstance(i, str) and i in known]
        return list(dict.fromkeys(ids))[:n]


class C_QuestionWriter(C_LLMTask):

    reply_type = T_Questions
    SYSTEM = GP_QUESTIONS

    @classmethod
    def cm_varying_attributes(cls, candidates: list[C_Product], product_words: Iterable[str] = ()) -> list[dict]:
        values: dict[str, set[str]] = {}
        holders: dict[str, set[str]] = {}
        for p in candidates:
            for key, value in p.attrs.items():
                name = " ".join(RE_ATTRIBUTE_SEPARATOR.sub(" ", str(key)).lower().split())
                if not name or isinstance(value, (dict, list)):
                    continue
                values.setdefault(name, set()).add(str(value).lower()[:40])
                holders.setdefault(name, set()).add(p.id)
        words = {n: set(n.split()) for n in values}
        nouns = {C_Canon.sm_singular(w) for w in product_words} | set(product_words)
        home = {}
        for name in values:
            parents = [o for o in values if o != name and words[o] < words[name]
                       and (len(words[o]) > 1 or all(len(w) <= 2 or w in nouns for w in words[name] - words[o]))]
            home[name] = min(parents, key=lambda o: (len(words[o]), o)) if parents else name
        merged_values: dict[str, set[str]] = {}
        merged_holders: dict[str, set[str]] = {}
        for name, root in home.items():
            merged_values.setdefault(root, set()).update(values[name])
            merged_holders.setdefault(root, set()).update(holders[name])
        rows = [{"attribute": k, "listings": len(merged_holders[k]), "values": sorted(v)[:5]}
                for k, v in merged_values.items()]
        rows.sort(key=lambda r: (-r["listings"], -len(r["values"]), r["attribute"]))
        return rows[:24]

    @staticmethod
    def sm_listed(prop: str, rows: list[dict]) -> str | None:
        want = set(RE_ATTRIBUTE_SEPARATOR.sub(" ", prop.lower()).split())
        for row in rows:
            have = set(row["attribute"].split())
            if want and (want == have or want <= have or have <= want):
                return row["attribute"]
        return None

    @staticmethod
    def sm_field_name(key: Any) -> str:
        spaced = RE_CAMEL_BOUNDARY.sub(" ", str(key or ""))
        return " ".join(RE_ATTRIBUTE_SEPARATOR.sub(" ", spaced).lower().split())

    @classmethod
    def cm_field_features(cls, candidates: list[C_Product], settled: set[str], request: str) -> list[tuple[str, dict]]:
        values: dict[str, list[str]] = {}
        places: dict[str, list[float]] = {}
        for p in candidates:
            attributes = [(k, v) for k, v in p.attrs.items() if not isinstance(v, (dict, list))]
            for index, (key, value) in enumerate(attributes):
                name, text = cls.sm_field_name(key), gm_normalize_space(value).lower()
                if not name or not text or len(text.split()) > GC_FIELD_VALUE_WORDS:
                    continue
                values.setdefault(name, []).append(text)
                places.setdefault(name, []).append(index / max(1, len(attributes)))
        asked = set(C_Canon.cm_tokens(request))
        out = []
        for name, seen in values.items():
            words = set(C_Canon.cm_tokens(name))
            if words & settled or any(set(C_Canon.cm_tokens(v)) and set(C_Canon.cm_tokens(v)) <= settled for v in seen):
                continue
            share = len(seen) / max(1, len(candidates))
            if share < GC_FIELD_MIN_SHARE:
                continue
            distinct = len(set(seen))
            out.append((name, {
                "share": share, "distinct": min(1.0, distinct / max(1, len(candidates))),
                "varies": float(distinct > 1), "value_words": min(1.0, sum(len(v.split()) for v in seen) / len(seen) / 4),
                "numeric": sum(any(ch.isdigit() for ch in v) for v in seen) / len(seen),
                "name_words": min(1.0, len(name.split()) / 3), "place": sum(places[name]) / len(places[name]),
                "in_request": float(bool(words & asked))}))
        return out

    def m_run(self, request: str, requirements: list[C_Requirement], candidates: list[C_Product], n: int,
              product_words: Iterable[str] = ()) -> list[dict]:
        if n <= 0:
            return []
        settled = {w for r in requirements for w in C_Canon.cm_tokens(f"{r.attribute} {r.value}")}
        scored = [(sum(GC_FIELD_WEIGHTS.get(k, 0.0) * v for k, v in features.items()), name)
                  for name, features in self.cm_field_features(candidates, settled, request)]
        scored.sort(key=lambda pair: -pair[0])
        return [{"property": name, "question": f"Which {name} would you like?"} for _, name in scored[:n]]

    @staticmethod
    def sm_clip(text: str) -> str:
        sentences = [x for x in RE_SENTENCE_BREAK.split(gm_normalize_space(text)) if x]
        result = next((x for x in sentences if x.endswith("?")), sentences[0] if sentences else "")
        words = result.split()
        if len(words) > 30:
            head = " ".join(words[:30])
            result = head[: head.rfind(",")] if "," in head else head
        if result.count("(") > result.count(")"):
            result = result[: result.rfind("(")]
        result = result.strip(" ,;:-")
        if result and result[-1] not in ".?!":
            result += "?"
        return result


class C_ChangeReader(C_LLMTask):

    reply_type = T_Changes
    SYSTEM = GP_CHANGES

    def m_run(self, notices: list[str], requirements: list[C_Requirement], policy: C_Policy,
              products: list[C_Product], ordered: str | None = None) -> dict:
        answer = self.m_ask(self.SYSTEM, json.dumps({
            "notices": notices,
            "criteria": [{"property_name": r.attribute, "relation": r.relation, "required_value": r.value}
                         for r in requirements],
            "budget": policy.price_cap,
            "items": [p.m_brief(60) for p in products[:15]],
            "item_just_ordered": ordered,
            "note": "a notice about the item the assistant just suggested or picked is about item_just_ordered",
        }, ensure_ascii=False)) or {}
        values = {r.value for r in requirements}
        known = {p.id for p in products}
        dropped = [v for v in answer.get("retired_values") or [] if isinstance(v, str) and v in values]
        added = []
        for item in answer.get("new_criteria") or []:
            if isinstance(item, dict):
                req = gm_criterion(item, "required_value", "", gm_normalize_space(item.get("property_name") or ""))
                if req is not None:
                    added.append(req)
        budget = answer.get("revised_ceiling")
        budget = float(budget) if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 \
           and any(gm_request_has_number(n, budget) for n in notices) else None
        unavailable = [i for i in answer.get("withdrawn_ids") or [] if isinstance(i, str) and i in known]
        return {"dropped": dropped, "added": added, "new_budget": budget, "unavailable": unavailable}


class C_OpeningQuestionWriter(C_LLMTask):

    reply_type = T_Questions
    SYSTEM = GP_OPENING_QUESTIONS
    FALLBACK = ("material", "color", "style")

    def m_run(self, request: str, product_type: str, requirements: list[C_Requirement], n: int) -> list[dict]:
        if n <= 0:
            return []
        settled = {w for r in requirements for w in C_Canon.cm_tokens(f"{r.attribute} {r.value}")}
        brief = {"brief": request, "item_kind": product_type,
                 "settled_already": [f"{r.attribute}: {r.value}" for r in requirements]}
        answer = self.m_ask(self.SYSTEM.replace("{n}", str(n)), json.dumps(brief, ensure_ascii=False)) or {}
        names: list[str] = []
        for q in answer.get("inquiries") or []:
            name = C_QuestionWriter.sm_field_name(q.get("attribute_name") if isinstance(q, dict) else "")
            words = set(C_Canon.cm_tokens(name))
            if name and 1 <= len(name.split()) <= 3 and name not in names and not (words and words <= settled):
                names.append(name)
        for name in self.FALLBACK:
            if len(names) >= n:
                break
            if name not in names and not {name} <= settled:
                names.append(name)
        return [{"property": name, "question": f"Which {name} would you like?"} for name in names[:n]]


class C_AnswerReader(C_LLMTask):

    reply_type = T_ReplyReadings
    SYSTEM = GP_REPLY_READER

    def m_run(self, answers: list[dict]) -> list[C_Requirement]:
        if not answers:
            return []
        numbered = [{"exchange": i + 1, **a} for i, a in enumerate(answers)]
        answer = self.m_ask(self.SYSTEM, json.dumps(numbered, ensure_ascii=False)) or {}
        parsed: list[dict] = [{} for _ in answers]
        for reading in answer.get("readings") or []:
            index = reading.get("exchange", 0) - 1
            if 0 <= index < len(answers) and not parsed[index]:
                parsed[index] = reading
        found: list[tuple[C_Requirement, set[str]]] = []
        for answer, asked in zip(parsed, answers):
            said = set(C_Canon.cm_tokens(asked.get("reply", "")))
            for need in self.m_needs(answer, [asked]):
                found.append((need, said))
        out: list[C_Requirement] = []
        for need, said in found:
            same = [(n, w) for n, w in found if C_Canon.cm_key(n.value) == C_Canon.cm_key(need.value)]
            named = [n for n, w in same if set(C_Canon.cm_tokens(n.attribute)) <= w]
            keep = named or [same[0][0]]
            if need in keep and all(need.value != r.value or need.attribute != r.attribute for r in out):
                out.append(need)
        return out

    def m_needs(self, answer: dict, answers: list[dict]) -> list[C_Requirement]:
        out = []
        replies = " ".join(C_Canon.cm_key(a.get("reply", "")) for a in answers)
        said = replies.split()
        for item in answer.get("stated_wants") or []:
            if not isinstance(item, dict):
                continue
            need = gm_criterion(item, "wanted_value", "", gm_normalize_space(item.get("property_name") or ""))
            if need is None:
                continue
            if need.m_bound is not None:
                if not any(gm_request_has_number(a.get("reply", ""), need.m_bound[1]) for a in answers):
                    continue
            else:
                words = [w for w in C_Canon.cm_key(need.value).split() if len(w) > 1]
                if not words or not all(w in said or any(C_Canon.sm_same_word(w, r) for r in said) for w in words):
                    continue
            need.text = need.value
            need.variants = need.variants[:6]
            out.append(need)
        return out


@dataclass
class C_Decision:

    action: str
    ids: list
    reason: str = ""
    source: str = "llm"
    confirmed: list = field(default_factory=list)


class C_Decider(C_LLMTask):

    model = GC_PRIMARY_MODEL
    reasoning = GC_DECISION_REASONING
    token_limit = GC_DECIDER_MAX_TOKENS

    reply_type = T_Decision
    SYSTEM = GP_DECISION

    def m_run(self, request: str, policy: C_Policy, requirements: list[C_Requirement], answers: list[dict],
              notices: list[str], candidates: list[C_Product]) -> C_Decision | None:
        items = [p.m_brief(300) for p in candidates]
        user = json.dumps({
            "brief": request, "rules": policy.m_to_dict(), "criteria": [r.m_to_dict() for r in requirements],
            "buyer_replies": answers, "change_notices": notices, "items": items}, ensure_ascii=False)
        answer = self.m_ask(self.SYSTEM, user)
        if not isinstance(answer, dict):
            return None
        known = {p.id for p in candidates}
        strong = [i for i in answer.get("strong_ids") or [] if isinstance(i, str) and i in known]
        weak = [i for i in answer.get("weak_ids") or [] if isinstance(i, str) and i in known and i not in strong]
        if answer.get("verdict") == "pass" and not strong and not weak:
            return C_Decision("decline", [], gm_normalize_space(answer.get("explanation") or ""))
        ids = list(dict.fromkeys(strong + weak))
        if not ids:
            return None
        return C_Decision("order", ids, gm_normalize_space(answer.get("explanation") or ""), confirmed=strong)


class C_SecondLook(C_LLMTask):

    model = GC_PRIMARY_MODEL
    token_limit = GC_RECHECK_MAX_TOKENS

    reply_type = T_SecondLook
    SYSTEM = GP_SECOND_LOOK

    def m_run(self, request: str, policy: C_Policy, requirements: list[C_Requirement], answers: list[dict],
              products: list[C_Product], judged: set[str]) -> list[str]:
        if not products:
            return []
        user = json.dumps({
            "brief": request, "rules": policy.m_to_dict(), "criteria": [r.m_to_dict() for r in requirements],
            "buyer_replies": answers,
            "items": [{**p.m_brief(160), "already_read": p.id in judged} for p in products]}, ensure_ascii=False)
        answer = self.m_ask(self.SYSTEM, user)
        known = {p.id for p in products}
        ids = answer.get("recheck_ids") if isinstance(answer, dict) else None
        return [i for i in dict.fromkeys(ids or []) if isinstance(i, str) and i in known][:5]


class C_Finisher:

    def __init__(self, runner: "C_EpisodeRunner") -> None:
        self.runner = runner
        self.cart: dict[str, int] = {}
        self.stuck: set[str] = set()

    def m_cart_values(self, add: bool, pid: str, qty: int) -> tuple[C_Role, dict] | None:
        tools = self.runner.tools
        if tools.m_has(C_Role.CART_EDIT):
            binding = tools.m_binding(C_Role.CART_EDIT)
            action = binding.values.get(C_EnumValue.ADD if add else C_EnumValue.REMOVE)
            if action is not None or add:
                values = {C_Slot.ID: pid, C_Slot.QUANTITY: qty}
                if action is not None:
                    values[C_Slot.ACTION] = action
                return C_Role.CART_EDIT, values
        role = C_Role.CART_ADD if add else C_Role.CART_REMOVE
        if tools.m_has(role):
            return role, {C_Slot.ID: pid, C_Slot.QUANTITY: qty}
        return None

    def m_cart_call(self, add: bool, pid: str, qty: int) -> C_StepResult | None:
        chosen = self.m_cart_values(add, pid, qty)
        if chosen is not None:
            return self.runner.tools.m_call(*chosen)
        if not add and self.runner.tools.m_has(C_Role.CART_ADD):
            return self.m_set_zero(pid)
        return None

    def m_add_and_order(self, pid: str, qty: int) -> tuple[bool, C_StepResult | None]:
        tools = self.runner.tools
        chosen = self.m_cart_values(True, pid, qty)
        add = tools.m_build_args(*chosen) if chosen is not None else None
        order = tools.m_build_args(C_Role.ORDER, self.m_order_values(pid, qty))
        if add is None or order is None:
            return self.m_add(pid, qty), None
        results = self.runner.env.m_call_many([add, order])
        added = results[0] if results else None
        if added is None or tools.m_failed(added):
            return False, None
        self.cart[pid] = self.cart.get(pid, 0) + qty
        return True, results[1] if len(results) > 1 else None

    def m_set_zero(self, pid: str) -> C_StepResult | None:
        tools = self.runner.tools
        binding = tools.m_binding(C_Role.CART_ADD)
        param = tools.specs[binding.tool].m_param(binding.slots.get(C_Slot.QUANTITY))
        if param is None or C_ToolBox.sm_fit(0, param.schema) != 0:
            return None
        result = tools.m_call(C_Role.CART_ADD, {C_Slot.ID: pid, C_Slot.QUANTITY: 0})
        if result is None or tools.m_failed(result) or pid in json.dumps(result.payload, ensure_ascii=False):
            return None
        return result

    def m_add(self, pid: str, qty: int) -> bool:
        result = self.m_cart_call(True, pid, qty)
        if result is None or self.runner.tools.m_failed(result):
            return False
        self.cart[pid] = self.cart.get(pid, 0) + qty
        return True

    def m_remove(self, pid: str) -> bool:
        qty = self.cart.get(pid, 0)
        if qty <= 0:
            return True
        if pid in self.stuck:
            return False
        result = self.m_cart_call(False, pid, qty)
        if result is None or self.runner.tools.m_failed(result):
            self.stuck.add(pid)
            return False
        self.cart.pop(pid, None)
        return True

    def m_cart_exactly(self, pid: str, qty: int) -> bool:
        for other in [p for p in self.cart if p != pid]:
            if self.runner.env.m_remaining < 3 or not self.m_remove(other):
                return False
        if self.cart.get(pid) == qty:
            return True
        if pid in self.cart and (self.runner.env.m_remaining < 3 or not self.m_remove(pid)):
            return False
        return self.runner.env.m_remaining >= 2 and self.m_add(pid, qty)

    def m_still_fits(self, pid: str) -> bool:
        role = C_Role.LISTING if self.runner.tools.m_has(C_Role.LISTING) else C_Role.DETAILS
        if not self.runner.tools.m_has(role):
            return True
        _, fresh = self.runner.tools.m_lookup(role, pid)
        if fresh is not None:
            self.runner.catalog.m_add(fresh)
        product = self.runner.catalog.m_get(pid)
        return product is None or (product.in_stock is not False and self.runner.scorer.m_passes_hard_rules(product))

    def m_order_values(self, pid: str | None = None, qty: int | None = None) -> dict:
        binding = self.runner.tools.m_binding(C_Role.ORDER)
        values: dict[C_Slot, Any] = {}
        if binding is not None and C_Slot.ID in binding.slots:
            if pid is None and self.cart:
                pid, qty = next(iter(self.cart.items()))
            if pid is not None:
                values = {C_Slot.ID: pid, C_Slot.QUANTITY: qty or 1}
        if binding is not None and C_Slot.REMARK in binding.slots:
            values[C_Slot.REMARK] = self.m_order_remark(pid, qty)
        return values

    def m_order_remark(self, pid: str | None, qty: int | None) -> str:
        lines = dict(self.cart)
        if pid is not None and pid not in lines:
            lines[pid] = qty or 1
        parts = []
        for item, count in lines.items():
            product = self.runner.catalog.m_get(item)
            title = (product.title[:80] if product and product.title else item)
            price = f" at {product.price:g}" if product and product.price is not None else ""
            parts.append(f"{count} x {title} ({item}){price}")
        return "Placing the order: " + "; ".join(parts) + "." if parts else "Placing the order for the basket as it stands."

    def m_order(self, pid: str | None = None, qty: int | None = None) -> C_StepResult | None:
        return self.runner.tools.m_call(C_Role.ORDER, self.m_order_values(pid, qty))

    def m_decline(self) -> C_StepResult | None:
        return self.runner.tools.m_call(C_Role.DECLINE, {C_Slot.REMARK: GC_DECLINE_REMARK})

    def m_finish_any(self) -> C_StepResult | None:
        runner = self.runner
        if runner.env.finished or runner.env.m_remaining <= 0:
            return None
        carted_ok = [p for p in self.cart if p not in runner.unavailable
                     and (runner.catalog.m_get(p) is None or runner.scorer.m_passes_hard_rules(runner.catalog.m_get(p)))]
        if carted_ok and len(self.cart) == 1 and runner.tools.m_has(C_Role.ORDER):
            return self.m_order()
        if runner.tools.m_has(C_Role.DECLINE):
            return self.m_decline()
        if self.cart and runner.tools.m_has(C_Role.ORDER):
            return self.m_order()
        return None

    def m_run(self, decision: C_Decision) -> None:
        runner = self.runner
        qty = max(1, runner.policy.quantity)
        direct = not (runner.tools.m_has(C_Role.CART_EDIT) or runner.tools.m_has(C_Role.CART_ADD))
        for _ in range(6):
            if runner.env.finished or runner.env.m_remaining <= 0:
                return
            ids = [i for i in decision.ids if i not in runner.unavailable] if decision.action == "order" else []
            if not ids or not runner.tools.m_has(C_Role.ORDER):
                if ids == [] and self.cart:
                    result = self.m_finish_any()
                elif runner.tools.m_has(C_Role.DECLINE):
                    result = self.m_decline()
                else:
                    result = self.m_finish_any()
                if result is not None and result.refusal and not runner.env.finished:
                    runner.decline_refused = True
                    decision = runner.m_redecide("the decline was refused", result.refusal)
                    continue
                return
            pid = ids[0]
            if direct:
                result = self.m_order(pid, qty)
            elif not self.cart and runner.env.m_remaining >= 2 and runner.tools.m_has(C_Role.ORDER):
                added, result = self.m_add_and_order(pid, qty)
                if not added:
                    runner.m_note(f"could not add {pid} x{qty}")
                    if runner.env.m_remaining <= 1:
                        break
                    runner.unavailable.add(pid)
                    decision = runner.m_redecide(f"{pid} could not be added to the cart")
                    continue
                if result is None:
                    result = self.m_order()
            else:
                if not self.m_cart_exactly(pid, qty):
                    runner.m_note(f"could not cart exactly {pid} x{qty}")
                    if runner.env.m_remaining <= 1:
                        break
                    if self.stuck & (set(self.cart) - {pid}):
                        runner.m_note(f"cart lines that cannot be removed: {sorted(self.stuck & set(self.cart))}")
                        break
                    runner.unavailable.add(pid)
                    decision = runner.m_redecide(f"{pid} could not be added to the cart")
                    continue
                result = self.m_order()
            if result is None or runner.env.finished:
                return
            if result.refusal:
                runner.m_note(f"finish refused: {result.refusal}")
                if runner.env.m_remaining >= 3:
                    if not self.m_still_fits(pid):
                        runner.unavailable.add(pid)
                else:
                    runner.unavailable.add(pid)
                decision = runner.m_redecide(f"checkout of {pid} was refused", result.refusal, ordered=pid)
                continue
            if runner.tools.m_failed(result):
                runner.m_note(f"order call failed: {json.dumps(result.payload, ensure_ascii=False)[:200]}")
                if not direct:
                    self.cart = {}
                else:
                    runner.unavailable.add(pid)
                continue
        if not runner.env.finished:
            self.m_finish_any()


class C_EpisodeRunner:

    def __init__(self, client: C_ChatClient | None, transport: C_Transport, problem: dict, *,
                 role_detector: C_RoleDetector | None = None) -> None:
        self.client = client
        self.started = time.monotonic()
        self.log: list[dict] = []
        env_data = problem["environment"]
        self.policy_view = env_data["policy_view"]
        self.request = gm_normalize_space(self.policy_view.get("query") or problem.get("query") or "")
        self.user_request = self.request
        if self.policy_view.get("opening") is not None:
            self.request += "\n\nOpening: " + json.dumps(self.policy_view["opening"], ensure_ascii=False)
        max_steps = int(self.policy_view.get("max_steps") or GC_DEFAULT_MAX_STEPS)
        self.env = C_Environment(transport, env_data["binding"], max_steps, self.m_log_step,
                                 int(self.policy_view.get("max_calls_per_turn") or GC_MAX_ACTIONS))
        self.env.finished = bool(self.policy_view.get("finished"))
        if hasattr(transport, "deadline"):
            transport.deadline = self.started + GC_HARD_SECONDS
        self.episode = C_EpisodeContext.cm_for_episode(str(env_data["binding"].get("session_id") or ""),
                                                       self.started + GC_TOTAL_SECONDS - GC_FINISH_RESERVE_SECONDS)
        self.specs = {s.name: s for s in C_ToolSpec.cm_parse_all(self.policy_view.get("tools") or [])}
        self.role_detector = role_detector or C_RoleDetector(client)
        self.catalog = C_Catalog()
        self.reader = C_ProductReader(
            namer=(lambda sample, rows, top: C_OutputFieldReader(self.client, self.episode).m_run(sample, rows, top))
            if client else None)
        self.policy = C_Policy()
        self.requirements: list[C_Requirement] = []
        self.phrases: list[str] = []
        self.categories: list[str] = []
        self.answers: list[dict] = []
        self.candidates: list[C_Product] = []
        self.scorer = C_CandidateScorer(self.policy, [], [])
        self.decline_refused = False
        self.shopper_lines: list[str] = []
        self.queued_reads: list[str] = []
        self.unavailable: set[str] = set()
        self.changes_seen = 0
        self.tools: C_ToolBox | None = None
        self.finisher: C_Finisher | None = None
        self.decision: C_Decision | None = None

    def m_note(self, text: str) -> None:
        self.log.append({"stage": "note", "t": self.m_elapsed(), "text": text})

    def m_stage(self, name: str, **data: Any) -> None:
        self.log.append({"stage": name, "t": self.m_elapsed(), "step": self.env.step, **data})

    def m_log_step(self, kind: str, data: dict) -> None:
        self.log.append({"stage": kind, "t": self.m_elapsed(), **data})

    def m_elapsed(self) -> float:
        return round(time.monotonic() - self.started, 2)

    def m_time_left(self) -> float:
        return GC_TOTAL_SECONDS - (time.monotonic() - self.started)

    def m_can_spend(self, n: int, *, finishing: bool = False) -> bool:
        if self.env.finished:
            return False
        reserve = 0 if finishing else GC_FINISH_RESERVE_STEPS
        if self.env.m_remaining - n < reserve:
            return False
        if not finishing and self.m_time_left() < GC_FINISH_RESERVE_SECONDS:
            return False
        return True

    def m_visible_window(self) -> int:
        sizes = [item["rows"] for item in self.log if item.get("stage") == "search_page" and item.get("rows")]
        return max(sizes) * 3 if sizes else GC_TARGET_MATCHES

    def m_run(self) -> list[dict]:
        try:
            self.m_plan()
        except (C_EpisodeFinished, C_StepBudgetExhausted) as exc:
            self.m_note(f"stopped: {exc}")
        except Exception as exc:
            self.m_note(f"error: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}")
        finally:
            self.m_emergency_finish()
            if GV_LLM_FAILURES:
                self.m_stage("llm_failures", errors=GV_LLM_FAILURES[-20:])
            self.m_stage("end", finished=self.env.finished, steps=self.env.step, seconds=self.m_elapsed(),
                         llm_cost_usd=round(self.episode.spend[0], 5),
                         decision=self.decision.__dict__ if self.decision else None,
                         llm_calls=getattr(self.client, "calls", None))
        return self.log

    def m_plan(self) -> None:
        if self.env.finished:
            return
        tools = self.policy_view.get("tools") or []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            f_roles = pool.submit(self.role_detector.m_detect, tools, episode=self.episode)
            f_phrases = pool.submit(C_PhraseFinder(self.client, self.episode).m_run, self.request)
            f_policy = pool.submit(C_PolicyReader(self.client, self.episode).m_run, self.request)
            f_reqs = pool.submit(C_RequirementReader(self.client, self.episode).m_run, self.request)
            roles = f_roles.result()
            bindings = self.m_bindings(roles)
            self.phrases, product_type = f_phrases.result()
            self.policy = f_policy.result()
            self.requirements = f_reqs.result()
        if not self.policy.product_type:
            self.policy.product_type = product_type
        if self.policy.step_budget and self.policy.step_budget < self.env.max_steps:
            self.m_note(f"step limit {self.env.max_steps} -> {self.policy.step_budget} (the brief's own limit)")
            self.env.max_steps = self.policy.step_budget
        self.tools = C_ToolBox(self.env, self.specs, roles, bindings, self.reader)
        if self.client is not None:
            self.tools.judge = lambda call: C_CallOutcomeReader(self.client, self.episode).m_run(call)
        self.m_stage("analysis", roles=roles.m_to_dict()["roles"], bindings=bindings.m_to_dict(),
                     phrases=self.phrases, policy=self.policy.m_to_dict(),
                     requirements=[r.m_to_dict() for r in self.requirements])
        self.scorer = C_CandidateScorer(self.policy, self.requirements, [])

        if not self.tools.m_has(C_Role.ORDER) and not self.tools.m_has(C_Role.DECLINE):
            self.m_note("world has no finishing tool detected")

        self.m_ask_questions(opening=True)

        planner = C_SearchPlanner(self)
        self.m_wrap_search_logging()
        if self.tools.m_has(C_Role.SEARCH):
            if not self.phrases and self.policy.product_type:
                self.phrases = [self.policy.product_type]
            chosen = planner.m_choose_value_queries(self.m_value_phrases(), GC_QUERY_PROBE_STEPS)
            if not chosen:
                chosen = planner.m_choose_queries(self.phrases, GC_QUERY_PROBE_STEPS)
            if not chosen:
                chosen = planner.m_first_hit(self.m_fallback_phrases({p.phrase for p in planner.probes}), 4)
            self.m_stage("queries", probes=[{"phrase": p.phrase, "count": p.page.m_count} for p in planner.probes],
                         chosen=[p.phrase for p in chosen])
            planner.m_fetch_more_pages(chosen, GC_PAGE_STEPS)
            categories = self.catalog.m_categories()
            self.categories = C_CategoryPicker(self.client, self.episode).m_run(
                self.request, self.policy.product_type, categories) if categories else []
            self.m_stage("categories", seen=categories[:30], chosen=self.categories)
            self.scorer = C_CandidateScorer(self.policy, self.requirements, self.categories)
            search_budget = self.m_search_budget()
            if self.categories and search_budget > 0:
                planner.m_category_search(self.categories, self.policy.price_cap, search_budget, self.policy.cheapest)
            elif search_budget > 0 and len(chosen) < 2:
                for phrase in [p for p in self.phrases if p not in {x.phrase for x in planner.probes}][:3]:
                    if not self.m_can_spend(1 + self.m_later_steps()):
                        break
                    planner.m_search(f"q:{phrase}", text=phrase)
        else:
            self.m_note("no search tool detected")
        self.m_stage("catalog", products=len(self.catalog))

        excluded = {p.id: self.scorer.exclusions.m_reason(p) for p in self.catalog.products.values()
                    if self.scorer.exclusions.m_excluded(p)}
        if excluded:
            self.m_stage("exclusions", removed=len(excluded), examples=dict(list(excluded.items())[:5]))
        ranked = self.scorer.m_rank(self.catalog.products.values())
        top = ranked[:40]
        ids = C_Shortlister(self.client, self.episode).m_run(self.request, self.requirements, top,
                                                             GC_CANDIDATE_COUNT, self.policy.cheapest)
        if not ids:
            ids = [p.id for p in top[:GC_CANDIDATE_COUNT]]
        self.candidates = [self.catalog.m_get(i) for i in ids if self.catalog.m_get(i)]
        if self.policy.cheapest and top:
            best = self.scorer.m_score(top[0])
            tier = [p for p in top if self.scorer.m_score(p) >= 0.75 * best and p.price is not None]
            cheap_first = sorted(tier, key=lambda p: p.price)
            picked = cheap_first[:GC_CANDIDATE_COUNT]
            self.candidates = picked + [p for p in self.candidates if p not in picked]
            self.candidates = self.candidates[: GC_CANDIDATE_COUNT + 2]
        if self.policy.open_at_least and len(self.candidates) < self.policy.open_at_least:
            more = [p for p in ranked if p not in self.candidates and self.scorer.m_passes_hard_rules(p)]
            self.candidates += more[: self.policy.open_at_least - len(self.candidates)]
        self.m_stage("shortlist", ranked=len(ranked), candidates=[p.m_brief(60) for p in self.candidates])

        detail_role = self.m_detail_role()
        batch = self.tools.m_compare_batch_size()
        if batch > 1:
            want = [p.id for p in self.candidates]
            want += [p.id for p in self.scorer.m_rank(self.catalog.products.values()) if p.id not in want][: batch * 2]
            start = 0
            while start < len(want) and self.m_can_spend(1 + self.m_question_steps()):
                size = self.tools.m_compare_batch_size()
                for fresh in self.tools.m_compare(want[start:start + size]):
                    merged = self.catalog.m_add(fresh)
                    if merged not in self.candidates and merged.id in want[: len(self.candidates) + batch]:
                        self.candidates.append(merged)
                start += size
        if detail_role is not None:
            missing = [p.id for p in self.candidates if not p.detailed]
            room = max(0, self.env.m_remaining - GC_FINISH_RESERVE_STEPS - self.m_question_steps())
            self.m_read_ids(detail_role, missing[:room])

        self.m_ask_questions()
        self.m_read_spare()

        self.m_apply_changes()
        self.decision = self.m_decide()
        self.m_stage("decision", **self.decision.__dict__)
        self.m_refine()
        self.m_audit_decline()
        self.m_cheaper_check()
        self.m_ensure_pick_read()

        self.finisher = C_Finisher(self)
        self.finisher.m_run(self.decision)

    def m_fallback_phrases(self, tried: set[str]) -> list[str]:
        noun = self.m_product_noun()
        out: list[str] = []
        for r in self.requirements:
            if r.m_is_product_type:
                for form in r.variants:
                    words = gm_get_words(form)
                    if words:
                        out.append(" ".join(words) if noun in words or len(words) > 1
                                   else f"{' '.join(words)} {noun}")
                        out.append(" ".join(words))
        out += [" ".join(gm_get_words(self.policy.product_type)), noun]
        seen = set(tried)
        result = []
        for phrase in out:
            if phrase and phrase not in seen:
                seen.add(phrase)
                result.append(phrase)
        return result[:6]

    def m_product_noun(self) -> str:
        words = C_Canon.cm_tokens(self.policy.product_type)
        return words[-1] if words else ""

    def m_value_phrases(self, extra: list[C_Requirement] | None = None) -> list[str]:
        type_words = C_Canon.cm_tokens(self.policy.product_type)
        noun = " ".join(gm_get_words(self.policy.product_type)) if 1 <= len(type_words) <= 3 else self.m_product_noun()
        stated_types = [" ".join(gm_get_words(r.m_positive_value)) for r in self.requirements
                        if r.m_is_product_type and r.critical and 1 <= len(gm_get_words(r.m_positive_value)) <= 2]
        if len(type_words) > 3 and stated_types:
            noun = stated_types[0]
        if not noun:
            return []
        props = [r for r in (extra if extra is not None else self.requirements)
                 if r.critical and r.kind == "property" and not r.m_negative and r.m_bound is None]
        values = [RE_PARENTHESIS.sub("", r.m_positive_value).strip() for r in props]
        values = [v for v in values if v and len(v.split()) <= 4]
        out = [v if set(C_Canon.cm_tokens(noun)) <= set(C_Canon.cm_tokens(v)) else f"{v} {noun}" for v in values]
        if len(values) >= 2:
            out.append(f"{values[0]} {values[1]} {noun}")
        head = self.m_product_noun()
        if head and head != noun:
            out += [v if head in C_Canon.cm_tokens(v) else f"{v} {head}" for v in values[:2]]
        return list(dict.fromkeys(out))

    def m_need_search(self, needs: list[C_Requirement]) -> None:
        if not needs or not self.tools.m_has(C_Role.SEARCH):
            return
        planner = C_SearchPlanner(self)
        searched = {item.get("label") for item in self.log if item.get("stage") == "search_page"}
        stated = self.m_value_phrases()
        singles = self.m_value_phrases(needs)
        need_values = [RE_PARENTHESIS.sub("", n.m_positive_value).strip() for n in needs if n.kind == "property"]
        phrases = [f"{v} {stated[0]}" for v in need_values[:2]] if stated else []
        if len(need_values) >= 2:
            phrases.append(f"{need_values[0]} {need_values[1]} {self.m_product_noun()}")
        phrases = list(dict.fromkeys(phrases + singles))
        before = set(self.catalog.products)
        batch = [p for p in phrases if f"q:{p}" not in searched][:2]
        while batch and not self.m_can_spend(len(batch) + 1):
            batch = batch[:-1]
        planner.m_search_many([{"label": f"q:{phrase}", "text": phrase} for phrase in batch])
        fresh = [p for p in self.scorer.m_rank(self.catalog.products.values())
                 if not p.detailed and p.id not in self.unavailable]
        fresh = [p for p in fresh if p.id not in before][:3] + [p for p in fresh if p.id in before][:1]
        self.queued_reads = [p.id for p in fresh]
        self.m_stage("need_search", phrases=phrases[:3], new_rows=len(set(self.catalog.products) - before),
                     queued=len(self.queued_reads))

    def m_bindings(self, roles: C_RoleMap) -> C_WorldBindings:
        specs = list(self.specs.values())
        key = gm_hash_32({"tools": [s.m_to_prompt() for s in specs], "roles": roles.m_to_dict()["roles"],
                          "v": GC_BINDING_VERSION, "model": GC_PRIMARY_MODEL if self.client else None})
        llm = C_LLMSlotMapper(self.client, self.episode) if self.client is not None else None
        return C_Memo.cm_get_or_compute(key, lambda: C_WorldBindings.cm_build(specs, roles, llm))

    def m_wrap_search_logging(self) -> None:
        original = self.tools.m_search

        def logged(**kw: Any) -> tuple[C_StepResult | None, C_SearchPage]:
            result, page = original(**kw)
            self.log.append({"stage": "search_page", "t": self.m_elapsed(), "label": kw.get("label"),
                             "page": kw.get("page", 1), "max_price": kw.get("max_price"),
                             "total": page.total, "rows": len(page.products),
                             "ok": page.ok, "error": result.error if result else "no tool"})
            return result, page

        self.tools.m_search = logged
        original_many = self.tools.m_search_many

        def logged_many(requests: list[dict]) -> list[tuple[C_StepResult | None, C_SearchPage]]:
            out = original_many(requests)
            for kw, (result, page) in zip(requests, out):
                self.log.append({"stage": "search_page", "t": self.m_elapsed(), "label": kw.get("label"),
                                 "page": kw.get("page", 1), "max_price": kw.get("max_price"),
                                 "total": page.total, "rows": len(page.products), "ok": page.ok,
                                 "error": result.error if result else "no tool", "batched": True})
            return out

        self.tools.m_search_many = logged_many

    def m_question_steps(self) -> int:
        if not self.tools or not self.tools.m_has(C_Role.MESSAGE):
            return 0
        budget = self.policy.message_budget
        limit = GC_QUESTION_LIMIT if budget is None else min(budget, GC_QUESTION_LIMIT)
        return max(0, limit - len(self.answers))

    def m_later_steps(self) -> int:
        details = GC_CANDIDATE_COUNT if self.tools and (
            self.tools.m_has(C_Role.DETAILS) or self.tools.m_has(C_Role.LISTING)) else 0
        return details + self.m_question_steps()

    def m_search_budget(self) -> int:
        available = self.env.m_remaining - GC_FINISH_RESERVE_STEPS - self.m_later_steps()
        return max(0, min(GC_CATEGORY_STEPS, available))

    @staticmethod
    def sm_request_sentences(text: str) -> list[str]:
        parts = [p.strip() for p in re.split(r"(?<=\.)\s+", (text or "").strip())]
        return [p for p in parts if p]

    def m_question_request(self, index: int = 0, last: bool = False) -> str:
        sentences = self.sm_request_sentences(self.user_request)
        if not sentences:
            return self.user_request
        start = index * GC_QUESTION_SENTENCES
        if start >= len(sentences):
            start = 0
        stop = len(sentences) if last else start + GC_QUESTION_SENTENCES
        return " ".join(sentences[start:stop])

    def m_question_message(self, question: str, index: int = 0, last: bool = False) -> str:
        return GC_QUESTION_TEMPLATE.format(user_request=self.m_question_request(index, last), question=question)

    def m_ask_questions(self, opening: bool = False) -> None:
        n = self.m_question_steps()
        if n <= 0 or (not self.candidates and not opening):
            return
        first = [{"property": a["topic"], "question": a["question"]} for a in self.policy.opening_asks][:n]
        if opening:
            # every question goes out before the first search: the brief alone decides what to ask
            writer = C_OpeningQuestionWriter(self.client, self.episode)
            questions = first + (writer.m_run(self.request, self.policy.product_type or "", self.requirements,
                                              n - len(first)) if n > len(first) else [])
        else:
            questions = first + (C_QuestionWriter(self.client, self.episode).m_run(
                self.request, self.requirements, self.candidates, n - len(first),
                C_Canon.cm_tokens(self.policy.product_type)) if n > len(first) else [])
        self.m_stage("questions", questions=questions)
        room = len(questions)
        while room > 0 and not self.m_can_spend(room):
            room -= 1
        questions = questions[:room]
        results = self.tools.m_message_many([self.m_question_message(q["question"], i, i == len(questions) - 1)
                                             for i, q in enumerate(questions)])
        for index, (q, result) in enumerate(zip(questions, results)):
            if result is None:
                break
            if result.error == "budget_exceeded":
                self.m_note("message budget reached")
                break
            reply = gm_reply_text(result.payload)
            self.answers.append({"property": q["property"], "question": q["question"], "reply": reply})
            if index < len(first):
                self.shopper_lines.append(reply)
        needs = C_AnswerReader(self.client, self.episode).m_run(self.answers)
        if needs:
            self.requirements = self.requirements + needs
            self.scorer = C_CandidateScorer(self.policy, self.requirements, self.categories)
            self.m_stage("revealed_needs", needs=[r.m_to_dict() for r in needs])
            self.m_need_search(needs)

    def m_refine(self) -> None:
        detail_role = self.m_detail_role()
        for round_no in range(2):
            if self.decision is None or self.env.finished:
                return
            if self.decision.confirmed:
                return
            spare = self.env.m_remaining - GC_FINISH_RESERVE_STEPS
            if spare < 2 or self.m_time_left() < GC_FINISH_RESERVE_SECONDS + 40:
                return
            self.m_stage("refine", round=round_no + 1, spare_steps=spare)
            searched = {item.get("label") for item in self.log if item.get("stage") == "search_page"}
            noun = self.policy.product_type or ""
            phrases: list[str] = []
            for req in self.requirements:
                if req.m_negative or req.kind != "property":
                    continue
                for form in [req.value, *req.variants[:2]]:
                    phrase = gm_normalize_space(f"{form} {noun}")
                    if len(phrase.split()) >= 2 and f"q:{phrase}" not in searched and phrase not in phrases:
                        phrases.append(phrase)
            planner = C_SearchPlanner(self)
            batch = phrases[: max(1, spare // 3)]
            while batch and not self.m_can_spend(len(batch) + 1):
                batch = batch[:-1]
            planner.m_search_many([{"label": f"q:{phrase}", "text": phrase} for phrase in batch])
            if detail_role is not None:
                fresh = [p.id for p in self.scorer.m_rank(self.catalog.products.values())
                         if not p.detailed and p.id not in self.unavailable]
                room = max(0, self.env.m_remaining - GC_FINISH_RESERVE_STEPS)
                self.m_read_ids(detail_role, fresh[:room])
            self.decision = self.m_decide([f"second look (round {round_no + 1}) after more searching"])
            self.m_stage("decision", **self.decision.__dict__)

    def m_cheaper_check(self, search: bool = True) -> None:
        if not (self.policy.cheapest and self.decision and self.decision.action == "order" and self.decision.ids):
            return
        role = self.m_detail_role()
        pick = self.catalog.m_get(self.decision.ids[0])
        if role is None or pick is None or pick.price is None:
            return
        ceiling = math.floor(pick.price) - 1
        category = pick.category if pick.category in self.categories  \
           else (self.categories[0] if self.categories else None)
        planner = C_SearchPlanner(self)
        if search and category and ceiling > 0 and self.tools.m_has(C_Role.SEARCH) and self.m_can_spend(4):
            page = planner.m_search(f"c:{category}", category=category, max_price=float(ceiling))
            total = page.total if page.total is not None else len(page.products)
            if total > self.m_visible_window():
                planner.m_narrow_price(category, float(ceiling), total, 5)
        checker = C_CompletenessChecker(self.requirements, self.categories, self.policy.product_type,
                                        self.catalog.products.values())
        exclusions = self.scorer.exclusions
        cheaper = sorted((p for p in self.catalog.products.values()
                          if p.price is not None and p.price <= ceiling and not p.detailed
                          and p.id not in self.unavailable and self.scorer.m_passes_hard_rules(p)
                          and not exclusions.m_excluded(p)
                          and (not self.categories or p.category in self.categories)),
                         key=lambda p: p.price)
        read, found = 0, None
        queue = [p.id for p in cheaper][:6]
        while queue and found is None:
            room = max(0, self.env.m_remaining - GC_FINISH_RESERVE_STEPS)
            chunk, queue = queue[: min(3, room)], queue[min(3, room):]
            if not chunk:
                break
            for got in self.tools.m_lookup_many(role, chunk):
                merged = self.catalog.m_add(got)
                if merged not in self.candidates:
                    self.candidates.append(merged)
                read += 1
                if found is None and checker.m_verdict(merged)["complete"]:
                    found = merged.id
        self.m_stage("cheaper_check", pick=pick.id, pick_price=pick.price, ceiling=ceiling, rows=len(cheaper),
                     read=read, found=found)
        if read:
            self.decision = self.m_decide(["read cheaper products before ordering"])
            self.m_stage("decision", **self.decision.__dict__)

    def m_decision_pool(self, limit: int = 14) -> list[C_Product]:
        pool: list[C_Product] = [p for p in self.candidates if p.id not in self.unavailable]
        for p in self.scorer.m_rank(self.catalog.products.values()):
            if len(pool) >= limit:
                break
            if p.id not in self.unavailable and p not in pool:
                pool.append(p)
        return pool

    def m_decide(self, extra_notes: list[str] | None = None) -> C_Decision:
        pool = self.m_decision_pool()
        notices = list(self.env.notices) + [n for n in self.env.refusals if n not in self.env.notices]
        notices += [f"(agent note) {n}" for n in extra_notes or []]
        if self.policy.peak_property and not any(gm_measure_value(p, self.policy.peak_property) is not None
                                                 for p in pool):
            self.m_note(f"no figure for {self.policy.peak_property}: ranking by it dropped")
            self.policy.peak_property = ""
        decider = C_Decider(self.client, self.episode)
        decision = decider.m_run(self.request, self.policy, self.requirements, self.answers, notices, pool)  \
           if pool else None
        if decision is None:
            fitting = [p.id for p in self.scorer.m_rank(pool) if self.scorer.m_score(p) > 0]
            decision = C_Decision("order" if fitting else "decline", fitting, "fallback: best scored candidates",
                                  "fallback")
            self.m_note(f"decider unavailable: {decider.last_error or ('empty pool' if not pool else 'no usable reply')}")
        self.m_apply_strict(decision, pool)
        decision.ids = [i for i in decision.ids if i not in self.unavailable and self.catalog.m_get(i)
                        and self.scorer.m_passes_hard_rules(self.catalog.m_get(i))]
        decision.confirmed = [i for i in decision.confirmed if i in decision.ids]
        if decision.action == "order" and not decision.ids:
            decision.action = "decline"
            decision.reason += " (no fitting id passed the stock/price rules)"
        if self.policy.cheapest and decision.ids:
            def price(i: str) -> float:
                value = self.catalog.m_get(i).price
                return value if value is not None else 1e18
            confirmed = sorted(decision.confirmed, key=price)
            decision.ids = confirmed + sorted([i for i in decision.ids if i not in confirmed], key=price)
        elif self.policy.peak_property and decision.ids:
            decision.ids = gm_rank_by_measure(decision.ids, self.catalog, self.policy.peak_property,
                                              self.policy.peak_high)
        return decision

    def m_audit_decline(self) -> None:
        if self.decision is None or self.decision.action != "decline" or not GC_USE_DECLINE_AUDIT:
            return
        role = self.m_detail_role()
        room = max(0, self.env.m_remaining - GC_DECLINE_FINISH_STEPS)
        if role is None or room <= 0:
            return
        seen = [p for p in self.scorer.m_rank(self.catalog.products.values())
                if p.id not in self.unavailable and self.scorer.m_passes_hard_rules(p)][:40]
        judged = {p.id for p in self.m_decision_pool() if p.detailed}
        ids = C_SecondLook(self.client, self.episode).m_run(
            self.request, self.policy, self.requirements, self.answers, seen, judged)
        fresh = [i for i in ids if i not in judged]
        self.m_stage("decline_audit", possible=ids, unread=fresh)
        if not fresh:
            return
        self.m_read_ids(role, [i for i in fresh if not self.catalog.m_get(i).detailed][:room])
        self.candidates = [self.catalog.m_get(i) for i in fresh] + [p for p in self.candidates if p.id not in fresh]
        self.decision = self.m_decide(["products named by a check before declining were read; judge them too"])
        self.m_stage("decision", **self.decision.__dict__)

    def m_apply_strict(self, decision: C_Decision, pool: list[C_Product]) -> None:
        checker = C_CompletenessChecker(self.requirements, self.categories, self.policy.product_type,
                                        self.catalog.products.values())
        verdicts = {p.id: checker.m_verdict(p) for p in pool}
        complete = [p.id for p in pool if verdicts[p.id]["complete"]]
        if not complete and self.decline_refused:
            complete = [p.id for p in pool if verdicts[p.id]["requirements"]
                        and "unmet" not in verdicts[p.id]["requirements"].values()]
        exact = {pid: checker.m_exact_count(self.catalog.m_get(pid)) for pid in complete}
        ordered = [i for i in decision.ids if i in complete] + [i for i in complete if i not in decision.ids]
        ordered.sort(key=lambda pid: -exact[pid])
        dropped = [i for i in decision.ids if i not in complete]
        decision.ids = ordered
        decision.confirmed = list(ordered)
        decision.action = "order" if ordered else "decline"
        if dropped:
            decision.reason += f" (strict: dropped {len(dropped)} incomplete)"
        self.m_stage("strict_check", complete=ordered,
                     verdicts={i: v for i, v in verdicts.items() if i in set(dropped) | set(ordered)})

    def m_apply_changes(self) -> None:
        seen = self.env.notices + [n for n in self.env.refusals if n not in self.env.notices] + self.shopper_lines
        if len(seen) <= self.changes_seen:
            return
        self.changes_seen = len(seen)
        ordered = self.decision.ids[0] if self.decision and self.decision.ids else None
        change = C_ChangeReader(self.client, self.episode).m_run(
            seen, self.requirements, self.policy, self.m_decision_pool(), ordered)
        if change["dropped"]:
            self.requirements = [r for r in self.requirements if r.value not in change["dropped"]]
        if change["added"]:
            self.requirements = self.requirements + change["added"]
        if change["new_budget"]:
            self.policy.price_cap = change["new_budget"]
        self.unavailable |= set(change["unavailable"])
        self.scorer = C_CandidateScorer(self.policy, self.requirements, self.categories)
        self.m_stage("changes", dropped=change["dropped"], added=[r.m_to_dict() for r in change["added"]],
                     new_budget=change["new_budget"], unavailable=change["unavailable"])

    def m_redecide(self, why: str, notices: list[str] | None = None, *, ordered: str | None = None) -> C_Decision:
        self.m_note(f"redecide: {why}")
        self.m_apply_changes()
        notes = [why]
        if ordered is not None:
            status = "is now unavailable or changed (re-checked)" if ordered in self.unavailable  \
               else "was re-checked and still passes stock/price"
            notes.append(f"The notices refer to the order just attempted ({ordered}); that item {status}. "
                         "Notices change no other item unless they name it.")
        self.decision = self.m_decide(notes)
        self.m_stage("decision", **self.decision.__dict__)
        if ordered is not None and self.decision.action == "order":
            self.m_cheaper_check(search=False)
        if self.decision.action == "decline" and ordered is not None:
            if self.m_read_more(max(0, self.env.m_remaining - 4)):
                self.decision = self.m_decide(notes + ["read more datasheets after the refusal"])
                self.m_stage("decision", **self.decision.__dict__)
        return self.decision

    def m_read_ids(self, role: C_Role | None, pids: list[str]) -> int:
        if role is None or not pids:
            return 0
        read = 0
        for got in self.tools.m_lookup_many(role, pids):
            merged = self.catalog.m_add(got)
            if merged not in self.candidates:
                self.candidates.append(merged)
            read += 1
        return read

    def m_detail_role(self) -> C_Role | None:
        if self.tools.m_has(C_Role.DETAILS):
            return C_Role.DETAILS
        if self.tools.m_has(C_Role.LISTING):
            return C_Role.LISTING
        return None

    def m_read_spare(self) -> None:
        role = self.m_detail_role()
        queued, self.queued_reads = self.queued_reads, []
        if role is None:
            return
        queued = queued[:max(0, self.env.m_remaining - GC_FINISH_RESERVE_STEPS)]
        room = self.env.m_remaining - GC_FINISH_RESERVE_STEPS - 4 - len(queued)
        spare = [p.id for p in self.scorer.m_rank(self.catalog.products.values())
                 if not p.detailed and p.id not in self.unavailable and p.id not in queued]
        spare = spare[:max(0, min(room, GC_CANDIDATE_COUNT))]
        if queued or spare:
            read = self.m_read_ids(role, queued + spare)
            self.m_stage("spare_reads", read=read, queued=len(queued))

    def m_ensure_pick_read(self) -> None:
        if not self.decision or self.decision.action != "order" or not self.decision.ids:
            return
        role = self.m_detail_role()
        unread = [i for i in self.decision.ids[:4] if self.catalog.m_get(i) and not self.catalog.m_get(i).detailed]
        if role is None or not unread or self.env.m_remaining - len(unread) < GC_FINISH_RESERVE_STEPS // 2:
            return
        read = self.m_read_ids(role, unread)
        self.m_stage("pick_reads", read=read)
        if read:
            self.decision = self.m_decide(["read the datasheets of the picks before ordering"])
            self.m_stage("decision", **self.decision.__dict__)

    def m_read_more(self, budget: int) -> int:
        role = self.m_detail_role()
        if role is None or budget <= 0:
            return 0
        fresh = [p.id for p in self.scorer.m_rank(self.catalog.products.values())
                 if not p.detailed and p.id not in self.unavailable][: min(budget, 4)]
        return self.m_read_ids(role, fresh)

    def m_emergency_finish(self) -> None:
        if self.env.finished or self.env.m_remaining <= 0:
            return
        if self.tools is None:
            self.m_finish_without_tools()
            return
        self.m_note("emergency finish")
        finisher = self.finisher or C_Finisher(self)
        try:
            finisher.m_run(self.decision or self.m_decide())
        except Exception as exc:
            self.m_note(f"emergency finish failed: {type(exc).__name__}: {exc}")
        for _ in range(3):
            if self.env.finished or self.env.m_remaining <= 0:
                break
            try:
                finisher.m_finish_any()
            except Exception as exc:
                self.m_note(f"last resort failed: {type(exc).__name__}: {exc}")
                break

    def m_finish_without_tools(self) -> None:
        with contextlib.suppress(Exception):
            roles = C_RoleDetector(None).m_detect(self.policy_view.get("tools") or [], use_cache=False)
            name = roles.m_tool_for(C_Role.DECLINE) or roles.m_tool_for(C_Role.ORDER)
            if name:
                self.env.m_call(name, {})


class C_Agent:

    def __init__(self, client: C_ChatClient | None = None, role_detector: C_RoleDetector | None = None,
                 transport: C_Transport | None = None) -> None:
        self.client = client if client is not None else C_LLMClient()
        self.role_detector = role_detector if role_detector is not None else C_RoleDetector(self.client)
        self.transport = transport

    def m_run(self, problem_data: dict) -> list[dict]:
        transport = self.transport or C_ProxyTransport(timeout=GC_ENV_TIMEOUT)
        runner = C_EpisodeRunner(self.client, transport, problem_data, role_detector=self.role_detector)
        return runner.m_run()


def agent_main(problem_data: dict) -> list[dict]:
    return C_Agent().m_run(problem_data)