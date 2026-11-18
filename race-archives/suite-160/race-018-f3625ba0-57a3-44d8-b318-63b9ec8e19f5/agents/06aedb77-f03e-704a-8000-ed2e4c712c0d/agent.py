from __future__ import annotations
from dataclasses import dataclass, field
from os import environ
from typing import Any, Callable, Optional
import concurrent.futures
import hashlib
import json
import math
import queue
import re
import sys
import threading
import time
import traceback
import unicodedata
from src.agent.proxy_client import ProxyClient

ENV_PATH = "/environment/call"
INFERENCE_PATH = "/inference/chat/completions"
OBS_KEY = "observation"
FINISHED_KEY = "finished"
STEP_KEY = "step"
REPLAYED_KEY = "replayed"
REFUSAL_KEY = "refusal"
RUNTIME_ERROR_KEY = "error"
ACTION_KEY = "action"
NAME_KEY = "name"
ARGS_KEY = "args"
RUNTIME_ERROR_CODES = ("unknown_tool", "invalid_arguments", "budget_exceeded")
CODE_BUDGET_EXCEEDED = "budget_exceeded"

T_OPTIONAL_LLM_STOP = 150.0
T_EXPLORE_STOP = 190.0
T_DETERMINISTIC_CHECKOUT = 225.0
T_LLM_DEADLINE = 235.0
T_FALLBACK = 240.0
T_LAST_ENV_SEND = 255.0              # the last non-finish environment send
T_ENV_TIMEOUT_HORIZON = 290.0
T_LAST_FINISH_SEND = T_ENV_TIMEOUT_HORIZON - 10.0   # the last finish send (ORDER / DECLINE), 280 s

USD_SOFT = 0.035
USD_HARD = 0.06
USD_STOP = 0.08

ROLE_SEARCH = "role:search"
ROLE_FILTER = "role:filter"
ROLE_VIEW = "role:view"
ROLE_SPEC = "role:spec"
ROLE_BATCH = "role:batch"
ROLE_CART_WRITE = "role:cart_write"
ROLE_CART_VIEW = "role:cart_view"
ROLE_STATUS = "role:status"
ROLE_ORDER = "role:order"
ROLE_DECLINE = "role:decline"
ROLE_FINISH = "role:finish"          # one tool carrying both finishes through an enum
ROLE_MESSAGE = "role:message"
ROLE_OTHER = "role:other"
RECORD_ROLES = (ROLE_VIEW, ROLE_SPEC, ROLE_BATCH)
FINISH_ROLES = (ROLE_ORDER, ROLE_DECLINE, ROLE_FINISH)
ALL_ROLES = (ROLE_SEARCH, ROLE_FILTER, ROLE_VIEW, ROLE_SPEC, ROLE_BATCH, ROLE_CART_WRITE, ROLE_CART_VIEW,
             ROLE_STATUS, ROLE_ORDER, ROLE_DECLINE, ROLE_FINISH, ROLE_MESSAGE, ROLE_OTHER)

SLOT_QUERY = "slot:query"
SLOT_PAGE = "slot:page"
SLOT_OFFSET = "slot:offset"
SLOT_TOKEN = "slot:token"
SLOT_MODE = "slot:mode"
SLOT_SHELF = "slot:shelf"
SLOT_MAKER = "slot:maker"
SLOT_CEILING = "slot:ceiling"
SEARCH_SLOTS = (SLOT_QUERY, SLOT_PAGE, SLOT_OFFSET, SLOT_TOKEN, SLOT_MODE, SLOT_SHELF, SLOT_MAKER, SLOT_CEILING)
SLOT_ID = "slot:id"
SLOT_IDS = "slot:ids"
SLOT_QTY = "slot:qty"
SLOT_CART_MODE = "slot:cart_mode"
SLOT_ITEMS = "slot:items"
ITEM_SLOTS = (SLOT_ID, SLOT_IDS, SLOT_QTY, SLOT_CART_MODE, SLOT_ITEMS)
SLOT_TEXT = "slot:text"
SLOT_REMARK = "slot:remark"
SLOT_FINISH_MODE = "slot:finish_mode"
SLOT_LINE = "slot:line"
SLOT_KEY = "slot:key"
SLOT_OTHER = "slot:other"
TEXT_SLOTS = (SLOT_TEXT, SLOT_REMARK, SLOT_FINISH_MODE, SLOT_LINE, SLOT_KEY, SLOT_OTHER)
PAGING_SLOTS = (SLOT_PAGE, SLOT_OFFSET, SLOT_TOKEN)

SEM_TEXT = "sem:text"
SEM_FILTER_SHELF = "sem:filter_shelf"
SEM_FILTER_MAKER = "sem:filter_maker"
SEM_FILTER_ANY = "sem:filter_any"
SEM_ADD = "sem:add"
SEM_REMOVE = "sem:remove"
SEM_SET = "sem:set"
SEM_ORDER = "sem:order"
SEM_DECLINE = "sem:decline"
SEM_PRICE_ASC = "sem:price_asc"
SEM_OTHER = "sem:other"
SEARCH_MEANINGS = (SEM_TEXT, SEM_FILTER_SHELF, SEM_FILTER_MAKER, SEM_FILTER_ANY, SEM_OTHER)
CART_MEANINGS = (SEM_ADD, SEM_REMOVE, SEM_SET, SEM_OTHER)
FINISH_MEANINGS = (SEM_ORDER, SEM_DECLINE, SEM_OTHER)
SORT_MEANINGS = (SEM_PRICE_ASC, SEM_OTHER)

FR_ID = "fr:id"
FR_PRICE = "fr:price"
FR_STOCK = "fr:stock"
FR_TITLE = "fr:title"
FR_SHELF = "fr:shelf"
FR_MERCHANT = "fr:merchant"
FR_MAKER = "fr:maker"
FR_OVERVIEW = "fr:overview"
FR_NOTES = "fr:notes"
FR_ATTRS = "fr:attrs"
FR_RECORDS = "fr:records"
FR_COUNT = "fr:count"
FR_CONT = "fr:cont"
FR_LINES = "fr:lines"
FR_LINE_ID = "fr:line_id"
FR_LINE_QTY = "fr:line_qty"
FR_LINE_PRICE = "fr:line_price"
FR_TOTAL = "fr:total"
RECORD_FIELD_ROLES = (FR_ID, FR_PRICE, FR_STOCK, FR_TITLE, FR_SHELF, FR_MERCHANT, FR_MAKER, FR_OVERVIEW,
                      FR_NOTES, FR_ATTRS)
FAMILIES = (FR_ATTRS, FR_NOTES, FR_PRICE, FR_STOCK, FR_TITLE, FR_SHELF, FR_MERCHANT, FR_OVERVIEW)

FIN_ORDER = "finish:order"
FIN_DECLINE = "finish:decline"
SORT_ID = "sort:id"
SORT_PRICE = "sort:price"
SORT_UNKNOWN = "sort:unknown"

NEED_ATTRS = "need:attrs"
NEED_NOTES = "need:notes"
NEED_STOCK = "need:stock"
NEED_FULL = "need:full"
NEED_FAMILIES = {
    NEED_ATTRS: (FR_ATTRS,),
    NEED_NOTES: (FR_NOTES,),
    NEED_STOCK: (FR_STOCK, FR_PRICE),
    NEED_FULL: (FR_ATTRS, FR_NOTES, FR_STOCK, FR_PRICE),
}

STOPWORDS = frozenset(("a", "an", "the", "of", "and", "or", "to", "for", "in", "with"))

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{2,39}$")
_NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w])")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")

def monotonic() -> float:
    return time.monotonic()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        return 0
    return -(-int(a) // int(b))

def text_tokens(text: Any) -> list:
    """Lowercase alphanumeric tokens of any value."""
    return _TOKEN_RE.findall(str(text or "").lower())

def canon_key(raw: Any) -> str:
    """Canonical form of a key or label: lowercase, non-alphanumerics become spaces."""
    return " ".join(text_tokens(raw))

def name_tokens(name: Any) -> list:
    """Tokens of an identifier, splitting camelCase and snake_case."""
    return text_tokens(_CAMEL_RE.sub(r"\1 \2", str(name or "")))

def norm_space(text: Any) -> str:
    """Case- and whitespace-normalised text, used for verbatim-span checks."""
    return " ".join(str(text or "").lower().split())

def content_tokens(text: Any) -> list:
    return [t for t in text_tokens(text) if t not in STOPWORDS]

def looks_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, str):
        try:
            float(value.replace(",", ""))
            return True
        except ValueError:
            return False
    return False

def as_number(value: Any) -> Optional[float]:
    if not looks_numeric(value):
        return None
    if isinstance(value, str):
        return float(value.replace(",", ""))
    return float(value)

def is_id_shaped(value: Any) -> bool:
    """An identifier: no spaces, 3-40 characters, at least one digit, not a plain number."""
    return (isinstance(value, str) and bool(_ID_RE.match(value)) and any(c.isdigit() for c in value)
            and not looks_numeric(value))

def is_sentences(value: Any) -> bool:
    """A non-empty list of strings, each with >= 3 words and not id-shaped (DESIGN 5.3)."""
    return (isinstance(value, list) and len(value) > 0
            and all(isinstance(s, str) and len(s.split()) >= 3 and not is_id_shaped(s) for s in value))

def numbers_in(text: Any) -> list:
    """Every number written with digits: [(value, raw, start, end)]."""
    out = []
    for m in _NUMBER_RE.finditer(str(text or "")):
        whole = m.group(1).replace(",", "")
        frac = m.group(2)
        value = float(whole + "." + frac) if frac else int(whole)
        out.append((value, m.group(0), m.start(), m.end()))
    return out

def has_currency_symbol_near(text: str, start: int, end: int) -> bool:
    """True when a Unicode currency symbol (category Sc) touches the span (vocabulary-free)."""
    left = text[max(0, start - 2):start]
    right = text[end:end + 2]
    return any(unicodedata.category(ch) == "Sc" for ch in left + right)

def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

def wire_size(value: Any) -> int:
    """Bytes of value as the sandbox serialises it: plain json.dumps (ASCII escapes, ', ' and ': ')."""
    try:
        return len(json.dumps(value, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return len(json.dumps(dump_json(value)).encode("utf-8"))

def stable_hash(*parts: Any) -> str:
    return hashlib.sha256(dump_json(list(parts)).encode("utf-8")).hexdigest()

def token_overlap(a: Any, b: Any) -> float:
    """Jaccard overlap of two token bags (content tokens only)."""
    sa, sb = set(content_tokens(a)), set(content_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def clip_text(text: Any, n: int = 120) -> str:
    s = str(text or "")
    return s if len(s) <= n else s[: n - 1] + "…"

class StopEpisode(Exception):
    """The episode cannot continue: 402, quarantine, step budget gone or already finished."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason

class EnvelopeBug(Exception):
    """HTTP 422 or another malformed-envelope answer: the fallback rebuilds the envelope."""

class StepShift(Exception):
    """A 409 re-sync moved the step counter so that a non-finish call would take the last step (G4): the
    call was not sent; the executor turns this into BLOCKED G4 and the finisher sends the finish."""

    def __init__(self, n: int, limit: int):
        super().__init__(f"step shifted to n={n} of B={limit}")
        self.n, self.limit = n, limit

class GoFallback(Exception):
    """Raised inside a phase to jump straight to the deterministic finish (time, cost, pressure)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason

@dataclass
class Budget:
    B: int = 40
    M: int = 3
    M_source: str = "default"            # stated | status | default
    L: int = 0
    R: int = 4
    F: int = 36
    u: int = 0                           # steps sent (mirrors Transport.n)
    msgs_used: int = 0
    t0: float = 0.0
    usd: float = 0.0
    llm_calls: int = 0
    first_finish_step: Optional[int] = None
    contract_error_streak: int = 0
    known_good_shapes: set = field(default_factory=set)
    regime: str = "normal"               # normal | soft | hard | stop
    dead_inference: bool = False
    stop_reason: Optional[str] = None
    B_source: str = "default"            # extra: max_steps | stated | both | default
    M_probe_open: bool = False           # extra: default M may probe a 4th/5th message

@dataclass
class ToolBind:
    tool: str
    role: str = ROLE_OTHER
    role_conf: int = 0                   # 0 missing, 1 tentative, 2 agreed, 3 confirmed
    slots: dict = field(default_factory=dict)        # arg -> (slot, conf)
    enum_map: dict = field(default_factory=dict)     # arg -> {literal: (meaning, conf)}
    provides_expected: set = field(default_factory=set)
    provides_seen: set = field(default_factory=set)
    strikes: int = 0
    first_use_ok: Optional[bool] = None
    schema: dict = field(default_factory=dict)       # extra: the tool's JSON-schema parameters
    description: str = ""                            # extra
    plaus: dict = field(default_factory=dict)        # extra: role -> structural plausibility (0..1)
    vote_roles: list = field(default_factory=list)   # extra: role labels proposed by each vote
    decline_votes: int = 0                           # extra: how many votes called it a decline

@dataclass
class Binding:
    tools: dict = field(default_factory=dict)        # name -> ToolBind
    by_role: dict = field(default_factory=dict)      # role -> [tool names, best first]
    finish_kind: str = "uncertain"                   # two_tools | enum | uncertain
    item_list_cart: bool = False
    finish_conf: int = 0                             # extra: confidence of the finish arrangement (votes only)
    finish_tool: Optional[str] = None                # extra: enum arrangement tool
    finish_arg: Optional[str] = None                 # extra: enum arrangement argument
    order_literal: Optional[str] = None              # extra
    decline_literal: Optional[str] = None            # extra
    votes_used: int = 0                              # extra: number of valid BIND votes
    source: str = "structure"                        # extra: votes | arbiter | structure | cache
    demoted: dict = field(default_factory=dict)      # extra: role -> set of tools proven wrong for it

@dataclass
class Contract:
    P: Optional[int] = None
    W: Optional[int] = None
    W_source: str = "assumed"
    cont_kind: dict = field(default_factory=dict)    # tool -> page | offset | token
    sort: str = SORT_UNKNOWN                         # sort:id | sort:price | sort:unknown
    count_cap: Optional[int] = None
    token_cap: int = 5
    filter_honours_keywords: Optional[bool] = None
    text_honours_filters: Optional[bool] = None
    ceiling_inclusive: Optional[bool] = None
    batch_limit: Optional[int] = None
    id_sig: dict = field(default_factory=dict)       # {prefix, length, alphabet}
    unsearchable: set = field(default_factory=set)
    merchant_path_confirmed: bool = False
    round_counts: dict = field(default_factory=dict)  # extra: count value -> number of distinct queries
    sort_votes: dict = field(default_factory=dict)    # extra: sort:id|sort:price -> ascending multi-row pages

@dataclass
class FieldMap:
    """Per-tool field map, learned from value shapes (DESIGN 5.3)."""
    records_path: Optional[list] = None              # path to the record list; [] = the observation is one record
    role_paths: dict = field(default_factory=dict)   # fr:* -> key path inside a record
    count_path: Optional[list] = None
    cont_path: Optional[list] = None
    cart: dict = field(default_factory=dict)         # fr:lines/fr:line_id/fr:line_qty/fr:line_price/fr:total -> path
    attrs_mode: Optional[str] = None                 # attrs:dict | attrs:pairs | attrs:nested
    keyset: frozenset = frozenset()                  # extra: record keys when learned (re-learn on change)
    learned_from: int = 0                            # extra: observations used
    ambiguous: list = field(default_factory=list)    # extra: roles whose top candidates were within 10%
    row_version: int = 0                             # extra: FieldModel.version when learned
    pair_keys: Optional[tuple] = None                # extra: (name key, value key) of name/value attribute pairs
    pair_samples: list = field(default_factory=list)  # extra: pair lists seen (orientation by shape)
    meta_pages: dict = field(default_factory=dict)   # extra: qkey -> {page: sibling snapshot} (meta confirmation)
    count_confirmed: bool = False                    # extra: count path held constant across pages of one query
    cont_confirmed: bool = False                     # extra: continuation path changed or turned null across pages

@dataclass
class Record:
    id: str
    price_c: Optional[int] = None
    price_hist: list = field(default_factory=list)   # [(u, cents)]
    stock: Optional[bool] = None
    stock_hist: list = field(default_factory=list)   # [(u, bool)]
    title: str = ""
    shelf: Optional[str] = None
    merchant: Optional[str] = None
    maker: Optional[str] = None
    overview_head: str = ""
    attrs: dict = field(default_factory=dict)        # canon -> [(raw_key, raw_value)]
    notes: Optional[str] = None
    note_read: dict = field(default_factory=lambda: {"restates": [], "pointers": [], "not_versions": [],
                                                      "pair_rules": []})
    codes: dict = field(default_factory=lambda: {"own": [], "alias": [], "fits": [], "more_on": [],
                                                  "more_fits": []})
    seen_rows_at: list = field(default_factory=list)
    opened_by: dict = field(default_factory=dict)    # tool -> u
    chain_page: bool = False
    families_seen: set = field(default_factory=set)  # extra: field families observed for this id

@dataclass
class QueryRec:
    tool: str
    canon_args: str
    tokens: list = field(default_factory=list)
    count: Optional[int] = None
    capped: bool = False
    pages_read: set = field(default_factory=set)
    next_cont: Any = None
    complete: bool = False
    ids: list = field(default_factory=list)
    line: Optional[int] = None
    kind: str = "q:value"                            # q:value|q:ladder|q:bracket|q:merchant|q:twin|q:part|q:relaxed|q:probe
    args: dict = field(default_factory=dict)         # extra: the first page's arguments
    query_text: str = ""                             # extra
    has_cont: bool = False                           # extra: last page carried a continuation

@dataclass
class Constraint:
    cid: str
    attr: str                                        # canonical attribute
    op: str = "eq"                                   # eq|in|ne|ge|le|between|pair_pending
    vals: list = field(default_factory=list)
    bound: Optional[dict] = None                     # {num, unit, raw}
    source: str = "stated"                           # stated|disclosed|reference|event|review
    span: str = ""
    active: bool = True
    ref_id: Optional[str] = None

@dataclass
class KeyStats:
    canon: str
    spoken: str = ""
    variants: list = field(default_factory=list)
    cov: float = 0.0
    distinct: int = 0
    div: float = 0.0
    categorical: float = 0.3
    shelf_overlap: float = 0.0
    vote: float = 0.0
    score: float = 0.0
    twins: set = field(default_factory=set)
    sibling_of: Optional[str] = None
    meta: bool = False
    unseen: bool = False

@dataclass
class Verdict:
    per_constraint: dict = field(default_factory=dict)  # cid -> (state, level, quote)
    tier: str = "T3"                                   # T1 | T2P | T2 | T3 | X
    rank_penalty: float = 0.0
    compat_ok: Optional[bool] = None
    reasons: list = field(default_factory=list)        # extra

@dataclass
class Cert:
    kind: str                                          # A | B | C | P | D
    line: int
    queries: list = field(default_factory=list)
    bracket_c: Optional[int] = None
    floor_proof: bool = False
    valid: bool = True
    invalidated_by: Optional[str] = None

@dataclass
class Line:
    idx: int
    words: str = ""
    shelf_words: str = ""
    shelf_label: Optional[str] = None
    shelf_conf: int = 0
    qty: int = 1
    role: str = "independent"                          # base | accessory | independent
    constraints: list = field(default_factory=list)
    menu: dict = field(default_factory=dict)           # canon -> KeyStats
    asked: dict = field(default_factory=dict)          # canon -> outcome
    closed: set = field(default_factory=set)
    disclosed: int = 0
    table: dict = field(default_factory=dict)          # id -> row view (dict)
    verdicts: dict = field(default_factory=dict)       # id -> Verdict
    choice: Optional[str] = None
    backups: list = field(default_factory=list)
    cert: Optional[Cert] = None
    LB_c: int = 0
    LB_proof: Optional[str] = None
    query_alts: list = field(default_factory=list)     # extra: ordered fallback query texts (dead parse)
    guessed: bool = False                              # extra: line came from the deterministic parse
    cheapest: bool = False                             # extra: the objective applies to this line
    opened: list = field(default_factory=list)         # extra: ids opened for this line, in order

@dataclass
class Hop:
    merchant: str = ""
    shelf_words: str = ""
    descriptors: list = field(default_factory=list)
    candidate_ids: list = field(default_factory=list)
    chosen_id: Optional[str] = None
    opened: dict = field(default_factory=dict)
    pointer: Optional[dict] = None

@dataclass
class Reference:
    rid: str
    line: int
    attr: str
    relation: str = "equal"                            # equal | differ | pair
    origin: str = "request"                            # request | reply | trap
    source: dict = field(default_factory=lambda: {"merchant": "", "shelf_words": "", "descriptors": []})
    hops: list = field(default_factory=list)
    value: Optional[str] = None
    pair_rule: Optional[dict] = None
    status: str = "pending"                            # pending | resolved | partial | failed
    obligation: Optional[str] = None
    span: str = ""                                     # extra

@dataclass
class Ledger:
    lines: list = field(default_factory=list)
    cap_c: Optional[int] = None
    cap_span: str = ""
    objective: str = "any"                             # cheapest | any
    in_stock: bool = True
    decline_rule: dict = field(default_factory=lambda: {"allowed": True, "no_substitute": False,
                                                         "remark_rules": []})
    procedures: list = field(default_factory=list)
    refs: list = field(default_factory=list)
    hints: list = field(default_factory=list)
    compat: dict = field(default_factory=lambda: {"required": False, "base_line": None})
    version: int = 0
    stated_msgs: Optional[int] = None                  # extra: message budget stated in the request
    stated_steps: Optional[int] = None                 # extra: step or call budget stated in the request
    source: str = "none"                               # extra: votes | arbiter | fallback | none
    conflicts: list = field(default_factory=list)      # extra
    sentences: list = field(default_factory=list)      # extra: [(span, route)] from the parse

@dataclass
class Cart:
    observed: Optional[dict] = None                    # id -> qty, from a returned cart or a cart view
    observed_u: Optional[int] = None
    expected: dict = field(default_factory=dict)       # id -> qty, after every accepted write
    semantics: str = "unknown"                         # set | add | unknown
    sem_source: str = "schema"                         # schema | observed
    add_max: Optional[int] = None
    first_added: Optional[str] = None
    last_change_u: Optional[int] = None
    last_reopen_u: dict = field(default_factory=dict)  # id -> u
    total_observed: Optional[int] = None
    verified_u: Optional[int] = None                   # extra: last read-back that matched
    writes: int = 0                                    # extra
    returns_cart: Optional[bool] = None                # extra: writes return the cart (True) or an ACK (False)
    unverified: bool = False                           # extra: a write answered an unreadable shape; read back first
    observed_ambiguous: bool = False                   # extra: the last cart read fit two (qty, price) readings

@dataclass
class Event:
    text: str
    u: int
    via: str = "pending"                               # pending | refusal
    kind: str = "unknown"
    amount_c: Optional[int] = None
    attr: Optional[str] = None
    line: Optional[int] = None
    new_value: Optional[str] = None
    applied: bool = False

@dataclass
class Events:
    queue: list = field(default_factory=list)
    seen_texts: set = field(default_factory=set)
    refused_last: Optional[str] = None                 # finish:order | finish:decline | None
    decline_locked: bool = False
    dirty: bool = False
    before_first_finish: int = 0                       # extra: texts seen before the first finish

@dataclass
class Proof:
    kind: str                                          # E1|E2|E3|E1x|OVERCAP|NOFIT|FALLBACK
    lines: list = field(default_factory=list)
    numbers: dict = field(default_factory=dict)
    ids: list = field(default_factory=list)
    quotes: list = field(default_factory=list)
    certs: list = field(default_factory=list)
    audited: bool = False
    orderable: dict = field(default_factory=dict)      # extra: line idx -> id the audit found orderable (M7)

IK_SEARCH = "SEARCH"
IK_FILTER = "FILTER"
IK_OPEN = "OPEN"
IK_STATUS = "STATUS"
IK_ASK = "ASK"
IK_CART_SET = "CART_SET"
IK_CART_VIEW = "CART_VIEW"
IK_ORDER = "ORDER"
IK_DECLINE = "DECLINE"
IK_RAW = "RAW"
FINISH_INTENTS = (IK_ORDER, IK_DECLINE)

P_EXPLORE = "EXPLORE"
P_ASK = "ASK"
P_RESOLVE = "RESOLVE"
P_COVER = "COVER"
P_COMMIT = "COMMIT"
P_VERIFY = "VERIFY"
P_FINISH = "FINISH"
P_RECOVER = "RECOVER"
P_PROBE = "PROBE"

D_RECORDS = "RECORDS"
D_RECORD = "RECORD"
D_REPLY = "REPLY"
D_CART = "CART"
D_ACK = "ACK"
D_STATUS = "STATUS"
D_FINISHED = "FINISHED"
D_REFUSED = "REFUSED"
D_NONFINAL = "NONFINAL"
D_EMPTY = "EMPTY"
D_EMPTY_CART = "EMPTY_CART"
D_WORLD_FAULT = "WORLD_FAULT"
D_SESSION_ERROR = "SESSION_ERROR"
D_UNKNOWN = "UNKNOWN"
D_BLOCKED = "BLOCKED"
FAULT_KINDS = (D_EMPTY, D_WORLD_FAULT, D_SESSION_ERROR, D_UNKNOWN)

@dataclass
class Intent:
    """One environment action an engine wants (DESIGN 2). Payload keys by kind:
    SEARCH {query, cont?, qkey?, qkind?}; FILTER {shelf?, maker?, ceiling?, cont?, qkey?};
    OPEN {ids, need}; STATUS {}; ASK {key, text}; CART_SET {item, quantity};
    CART_VIEW {}; ORDER {}; DECLINE {remark?, basis}; RAW {tool, args}."""
    kind: str
    payload: dict = field(default_factory=dict)
    purpose: str = P_EXPLORE
    line: Optional[int] = None
    why: str = ""

    def signature(self) -> str:
        return self.kind + ":" + dump_json(self.payload)

@dataclass
class Call:
    tool: str
    args: dict
    role: str = ROLE_OTHER
    slots_used: list = field(default_factory=list)   # [(arg, slot)]
    path: str = "det"                                # det | op | det_fallback
    intent_kind: str = ""
    ids: list = field(default_factory=list)          # ids carried by the call
    shape: str = ""                                  # (tool, arg names, enum literals) signature
    write: Optional[tuple] = None                    # cart writes: (id, expected quantity after the write)

    def canonical(self) -> str:
        return self.tool + ":" + dump_json(self.args)

@dataclass
class Raw:
    observation: Any = None
    finished: bool = False
    step: Optional[int] = None
    replayed: bool = False
    http_status: Optional[int] = None
    err_body: Optional[str] = None

@dataclass
class Decoded:
    kind: str
    records: list = field(default_factory=list)      # record dicts as observed (after unwrapping)
    meta: dict = field(default_factory=dict)         # {"n": count|None, "cont": value, "has_cont": bool}
    cart: Optional[dict] = None                      # {"lines": [(id, qty, price_c|None)], "total_c": int|None}
    reply_text: Optional[str] = None
    messages: list = field(default_factory=list)     # pending-wrapper sentences
    refusal: list = field(default_factory=list)      # protocol refusal sentences
    fault_text: Optional[str] = None
    session_code: Optional[str] = None
    finished: bool = False
    inner: Any = None
    gid: Optional[str] = None                        # guard id when BLOCKED
    absent_ids: list = field(default_factory=list)   # ids a batch view reported missing
    records_path: Optional[list] = None
    container: Optional[dict] = None                 # the dict that holds the record list (meta siblings)
    tool: Optional[str] = None
    replayed: bool = False

    @staticmethod
    def blocked(gid: str, reason: str = "") -> "Decoded":
        return Decoded(kind=D_BLOCKED, gid=gid, fault_text=reason)

    @property
    def ok(self) -> bool:
        return self.kind not in FAULT_KINDS and self.kind != D_BLOCKED

@dataclass
class Admission:
    ok: bool
    call: Optional[Call] = None
    gid: Optional[str] = None
    reason: str = ""
    repaired: list = field(default_factory=list)     # guard ids that repaired the call

    @property
    def blocked(self) -> bool:
        return not self.ok

TRACE_CAP_BYTES = 1_500_000
TRACE_SUMMARY_RESERVE = 60_000
TRACE_OBS_CHARS = 6000
TRACE_OBS_CHARS_TIGHT = 800

class Trace:
    def __init__(self, clock: Callable[[], float], diag: Optional[Callable[[], dict]] = None,
                 cap_bytes: int = TRACE_CAP_BYTES):
        self.clock = clock
        self.diag = diag or (lambda: {})
        self.cap_bytes = cap_bytes
        self.entries: list = []
        self.decisions: list = []        # indices into entries
        self.llm: list = []              # compact llm call records (also decision entries)
        self.guard_counts: dict = {}
        self.size = 2
        self.dropped = 0
        self.compacted = 0
        self.summary_entry: Optional[dict] = None
        self.to_stderr = environ.get("LODESTAR_LOG") == "1"       # logging only; never changes the dialogue
        self.lock = threading.Lock()

    def _budget_left(self) -> int:
        return self.cap_bytes - TRACE_SUMMARY_RESERVE - self.size

    def _append(self, entry: dict, compactable: bool) -> None:
        size = wire_size(entry) + 2
        if size > self._budget_left():
            if not compactable:
                entry = self._tighten_step(entry)
            else:
                entry = {k: entry[k] for k in ("role", "decision", "u", "t") if k in entry}
                entry["compacted"] = True
                self.compacted += 1
            size = wire_size(entry) + 2
            if size > self._budget_left():
                self.dropped += 1
                return
        with self.lock:
            self.entries.append(entry)
            self.size += size
        if self.to_stderr:
            try:
                sys.stderr.write(dump_json(entry) + "\n")
            except Exception:
                pass

    def _tighten_step(self, entry: dict) -> dict:
        env = dict(entry.get("environment_result") or {})
        obs = env.get(OBS_KEY)
        env[OBS_KEY] = clip_text(dump_json(obs), TRACE_OBS_CHARS_TIGHT)
        out = dict(entry)
        out["environment_result"] = env
        return out

    def step(self, phase: str, intent: "Intent", call: "Call", raw: "Raw", dec: "Decoded", guards: list,
             extra: Optional[dict] = None) -> None:
        obs_text = dump_json(raw.observation)
        obs_out = raw.observation if len(obs_text) <= TRACE_OBS_CHARS else obs_text[:TRACE_OBS_CHARS]
        diag = dict(self.diag())
        diag.update({"path": call.path, "guards": list(guards), "intent": intent.kind, "decoded": dec.kind})
        if extra:
            diag.update(extra)
        entry = {
            "role": "assistant",
            "content": f"{phase}/{intent.purpose}: {clip_text(intent.why, 120)}",
            "tool_call": {"id": f"ls-{raw.step if raw.step is not None else diag.get('u')}", "type": "function",
                          "function": {"name": call.tool, "arguments": dump_json(call.args)}},
            "environment_result": {OBS_KEY: obs_out, FINISHED_KEY: bool(raw.finished), STEP_KEY: raw.step,
                                   REPLAYED_KEY: bool(raw.replayed)},
            "diag": diag,
        }
        self._append(entry, compactable=False)

    def decision(self, kind: str, **fields: Any) -> None:
        diag = self.diag()
        entry = {"role": "lodestar", "decision": kind, "u": diag.get("u"), "t": round(self.clock(), 2)}
        for k, v in fields.items():
            entry[k] = v
        self.decisions.append(len(self.entries))
        self._append(entry, compactable=True)

    def guard(self, gid: str, reason: str, intent_kind: str = "", tool: str = "") -> None:
        with self.lock:
            self.guard_counts[gid] = self.guard_counts.get(gid, 0) + 1
        self.decision("guard_block", gid=gid, reason=clip_text(reason, 200), intent=intent_kind, tool=tool)

    def llm_call(self, role: str, model: str, ms: float, tokens: tuple, usd: float, ok: bool, fallback: bool,
                 note: str = "", late: bool = False) -> None:
        rec = {"llm_role": role, "model": model, "ms": int(ms), "tokens": list(tokens), "usd": round(usd, 6),
               "success": ok, "fallback": fallback}
        if note:
            rec["note"] = clip_text(note, 160)
        if late:
            rec["late"] = True
        with self.lock:
            self.llm.append(rec)
        if not late:
            self.decision("llm", **rec)

    def summary(self, data: dict) -> None:
        data = dict(data)
        data["guard_counts"] = dict(self.guard_counts)
        data["trace_dropped"] = self.dropped
        data["trace_compacted"] = self.compacted
        self.summary_entry = {"role": "lodestar", "decision": "summary", "t": round(self.clock(), 2), **data}

    def export(self) -> list:
        out = list(self.entries)
        if self.summary_entry is not None:
            if wire_size(self.summary_entry) > TRACE_SUMMARY_RESERVE:
                text = dump_json(self.summary_entry)
                cut = TRACE_SUMMARY_RESERVE
                short = {"role": "lodestar", "decision": "summary", "truncated": True, "text": text[:cut]}
                while wire_size(short) > TRACE_SUMMARY_RESERVE - 200 and cut > 100:
                    cut //= 2
                    short["text"] = text[:cut]
                self.summary_entry = short
            out.append(self.summary_entry)
            if self.to_stderr:
                try:
                    sys.stderr.write(dump_json(self.summary_entry) + "\n")
                except Exception:
                    pass
        return out

    def export_size(self) -> int:
        """Bytes of the exported list as the sandbox writes it."""
        return wire_size(self.export())

MODEL_HAIKU = "anthropic/claude-haiku-5.5"
MODEL_LUNA = "openai/gpt-6-luna"
MODEL_V32 = "deepseek/deepseek-v3.2"
MODEL_V4PRO = "deepseek/deepseek-v4-pro"
MODEL_FLASH = "google/gemini-3.8-flash"
MODEL_SONNET = "anthropic/claude-sonnet-5.5"
MODEL_SOL = "openai/gpt-6-sol"
MODEL_V4FLASH = "deepseek/deepseek-v4-flash"
MODEL_QWEN = "qwen/qwen3.8-flash"           # on no ladder; its reasoning policy is pinned (12_llm)
LADDER_CHEAP = (MODEL_HAIKU, MODEL_LUNA, MODEL_V4FLASH, MODEL_V32)
LADDER_MID = (MODEL_V4PRO, MODEL_FLASH)
LADDER_STRONG = (MODEL_SOL, MODEL_V4PRO, MODEL_SONNET)

def s_obj(fields: dict, required: tuple = ()) -> dict:
    return {"t": "obj", "f": dict(fields), "r": tuple(required)}

def s_arr(item: dict, max_items: int = 64) -> dict:
    return {"t": "arr", "i": item, "n": max_items}

def s_map(value: dict) -> dict:
    return {"t": "map", "v": value}

def s_enum(*values: str) -> dict:
    return {"t": "enum", "e": tuple(values)}

S_STR = {"t": "str"}
S_NUM = {"t": "num"}
S_INT = {"t": "int"}
S_BOOL = {"t": "bool"}
S_ANY = {"t": "any"}
S_PATH = s_arr(S_STR, 8)
S_PAIR = s_arr(S_STR, 2)

def canon_label(value: Any, labels: tuple) -> Optional[str]:
    """An oracle label in lodestar's vocabulary: an exact (case-insensitive) match, or a bare label that
    matches exactly one namespaced label by its part after the colon ("order" -> "role:order")."""
    if not isinstance(value, str):
        return None
    low = value.strip().lower()
    for option in labels:
        if option.lower() == low:
            return option
    if not low or ":" in low:
        return None
    bare = low.replace(" ", "_").replace("-", "_")
    hits = {option for option in labels if option.lower().split(":", 1)[-1] == bare}
    return hits.pop() if len(hits) == 1 else None

def conform(value: Any, spec: dict) -> tuple:
    """Validate and coerce an oracle answer: (ok, value). Unknown keys are dropped, numeric strings
    become numbers, enum labels match case-insensitively, invalid list items are dropped."""
    kind = spec.get("t")
    if kind == "any":
        return True, value
    if kind == "obj":
        if not isinstance(value, dict):
            return False, None
        out = {}
        for key, sub in spec["f"].items():
            if key not in value or value[key] is None:
                if key in spec["r"]:
                    return False, None
                out[key] = None
                continue
            ok, coerced = conform(value[key], sub)
            if not ok:
                if key in spec["r"]:
                    return False, None
                out[key] = None
                continue
            out[key] = coerced
        return True, out
    if kind == "arr":
        if not isinstance(value, list):
            return False, None
        out = []
        for item in value[: spec["n"]]:
            ok, coerced = conform(item, spec["i"])
            if ok:
                out.append(coerced)
        return True, out
    if kind == "map":
        if not isinstance(value, dict):
            return False, None
        out = {}
        for key, item in value.items():
            ok, coerced = conform(item, spec["v"])
            if ok:
                out[str(key)] = coerced
        return True, out
    if kind == "str":
        if isinstance(value, str):
            return True, value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True, str(value)
        return False, None
    if kind in ("num", "int"):
        number = as_number(value)
        if number is None:
            return False, None
        if kind == "int":
            if abs(number - round(number)) > 1e-9:
                return False, None
            return True, int(round(number))
        return True, int(number) if float(number).is_integer() else number
    if kind == "bool":
        if isinstance(value, bool):
            return True, value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return True, value.strip().lower() == "true"
        return False, None
    if kind == "enum":
        label = canon_label(value, spec["e"])
        return (True, label) if label is not None else (False, None)
    return False, None

COMMON_RULES = (
    "Rules for every answer:\n"
    "- Reply with exactly one JSON object and nothing else: no prose, no code fences.\n"
    "- Copy every quoted value, span or name verbatim from the source you were given; never paraphrase a span.\n"
    "- Never turn story, background or chit-chat sentences into requirements.\n"
    "- Use only the labels listed for enumerated fields; when nothing fits use the 'other' label or null.\n"
    "- Placeholders such as <thing>, <attr>, <value>, <seller> are neutral; they are not hints about any shop.\n"
)
REPAIR_TEXT = "Reply with the JSON object only, in exactly the requested shape."

@dataclass(frozen=True)
class RoleSpec:
    key: str                        # registry key, e.g. BIND or BIND_ARBITER
    role: str                       # role tag used in the prompt, e.g. BIND
    rules: str
    output: str
    schema: Optional[dict]
    max_tokens: int
    ladder: tuple
    cap: float                      # per-call timeout cap in seconds (DESIGN 3.5)
    optional: bool = False          # optional work stops after 150 s and in the soft regime (REVIEW only;

    @property
    def tag(self) -> str:
        return f"lodestar:{self.role}:v1"

    def system_prompt(self) -> str:
        return f"{self.tag}\n{COMMON_RULES}\n{self.rules}\nOutput shape:\n{self.output}\n"

_ROLE_LABELS = ("role:search, role:filter, role:view, role:spec, role:batch, role:cart_write, role:cart_view, "
                "role:status, role:order, role:decline, role:finish, role:message, role:other")
BIND_RULES = (
    "Task: you receive the tool list of a shopping environment whose names may be disguised. Assign each tool "
    "a role by what it does, judged from its description and parameters, never from its name alone.\n"
    "Roles: role:search (free-text search over listings, possibly with paging or filter arguments); role:filter "
    "(browse by shelf/category, maker or price ceiling; may be the search tool through a mode argument); "
    "role:view (one listing as shown in results: price, stock, seller note); role:spec (one listing's attribute "
    "sheet); role:batch (several listings for a list of ids); role:cart_write (puts an item in the cart, sets or "
    "changes its quantity, or removes it); role:cart_view (shows the cart); role:status (counters such as "
    "messages left); role:order (orders the cart and ends the task); role:decline (ends the task without "
    "buying); role:finish (one tool that ends the task, with an argument choosing order or decline); "
    "role:message (sends a message to the shopper and returns the reply); role:other.\n"
    "There is exactly one finish arrangement: one role:order tool plus one role:decline tool, or one "
    "role:finish tool.\n"
    "Give a slot for every argument of every tool: slot:query, slot:page (page number), slot:offset (row "
    "offset), slot:token (opaque continuation), slot:mode (search mode), slot:shelf, slot:maker, slot:ceiling "
    "(price limit), slot:id, slot:ids (list of ids), slot:qty, slot:cart_mode, slot:items (list of id and "
    "quantity objects), slot:text (message body), slot:remark, slot:finish_mode, slot:line, slot:key, slot:other.\n"
    "Give a meaning for every enum literal. Search mode: sem:text, sem:filter_shelf, sem:filter_maker, "
    "sem:filter_any, sem:other. Cart mode: sem:add, sem:remove, sem:set, sem:other. Finish choice: sem:order, "
    "sem:decline, sem:other. Sort: sem:price_asc, sem:other.\n"
    "List the field families each tool returns: fr:attrs, fr:notes, fr:price, fr:stock, fr:title, fr:shelf, "
    "fr:merchant, fr:overview. conf is your confidence from 0 to 3.\n"
    "If the payload carries 'votes', two earlier answers disagreed: decide every tool yourself in the same shape."
)
BIND_OUTPUT = ('{"tools": [{"tname": "<tool name>", "role": "role:...", "conf": 0-3, "args": {"<arg>": "slot:..."}, '
               '"enum_map": {"<arg>": {"<literal>": "sem:..."}}, "returns": ["fr:..."]}]}')
BIND_SCHEMA = s_obj({"tools": s_arr(s_obj({
    "tname": S_STR,
    "role": S_STR,                  # mapped and counted by Binder.vote_validator (bare labels are repaired)
    "conf": S_NUM,
    "args": s_map(S_STR),
    "enum_map": s_map(s_map(S_STR)),
    "returns": s_arr(S_STR, 12),
}, ("tname", "role")), 40)}, ("tools",))

PARSE_RULES = (
    "Task: turn the shopper's request into a typed order ledger.\n"
    "1. Route every sentence or clause: requirement, objective, cap, stock, budget, decline_rule, procedure, "
    "reference, hint, story or restatement; copy each span verbatim.\n"
    "2. lines: one per distinct product wanted: n (0-based index), words naming it, shelf_words (category "
    "words), qty_n (a number; 1 when unstated) with qty_span (verbatim, \"\" when unstated), line_role (base, "
    "accessory = must fit a base item, or independent).\n"
    "3. reqs on a line come only from requirement or reference sentences: attr, op (eq, in, ne, ge, le, "
    "between), vals (verbatim values), num and unit for numeric bounds, span. Story and restatement sentences "
    "give at most hints.\n"
    "4. refs: a value that must equal, differ from or pair with the value on another listing: line, attr, rel "
    "(equal, differ, pair) by meaning, merchant, shelf_words, descriptors as [attr, value] pairs that locate the "
    "source listing, span. Descriptors locate the source; they never constrain the wanted item.\n"
    "5. cap: the one whole-basket spending limit: amount (number as written), unit as written, span. Never split "
    "it per line.\n"
    "6. stock_rule {required, span}. objective {kind, span}: kind is \"cheapest\" only when the shopper "
    "explicitly asks for the lowest price, otherwise \"any\".\n"
    "7. msgs {n, span}: the message budget. steps {n, kind, span}: the step or call budget, kind \"steps\" or "
    "\"calls\".\n"
    "8. decline_rule {allowed, no_substitute, remark_rules, span}: may the task end without buying, and verbatim "
    "rules about a closing remark.\n"
    "9. procedures [{kind, text, n, span}] (kind: view_n = view at least n listings, confirm_before_order = send "
    "the shopper a message before ordering, else other); compat {required, base_line, span}: items must fit each other; "
    "hints [{line, attr_guess, span}]: attributes the story suggests.\n"
    "Use null for anything not stated. Numbers inside spans must be copied exactly as written. Merchants, "
    "descriptor values, remark rules and procedure texts are copied verbatim from the request; a rule without "
    "its verbatim span is ignored.\n"
    "If the payload carries 'votes', two earlier answers disagreed: decide every field yourself in the same "
    "shape, quoting verbatim spans."
)
PARSE_OUTPUT = (
    '{"sentences": [{"span": "...", "route": "requirement"}], "lines": [{"n": 0, "words": "...", '
    '"shelf_words": "...", "qty_n": 1, "qty_span": "", "line_role": "independent", "reqs": [{"attr": "<attr>", '
    '"op": "eq", "vals": ["<value>"], "num": null, "unit": null, "span": "..."}]}], "refs": [{"line": 0, '
    '"attr": "<attr>", "rel": "equal", "merchant": "<seller>", "shelf_words": "...", "descriptors": '
    '[["<attr>", "<value>"]], "span": "..."}], "cap": {"amount": 0, "unit": "...", "span": "..."}, '
    '"stock_rule": {"required": true, "span": "..."}, "objective": {"kind": "any", "span": ""}, '
    '"msgs": {"n": null, "span": ""}, "steps": {"n": null, "kind": "steps", "span": ""}, "decline_rule": '
    '{"allowed": true, "no_substitute": false, "remark_rules": [], "span": ""}, "procedures": [], '
    '"compat": {"required": false, "base_line": null, "span": ""}, "hints": []}'
)
_SPANNED_NUM = s_obj({"n": S_NUM, "span": S_STR})
PARSE_SCHEMA = s_obj({
    "sentences": s_arr(s_obj({"span": S_STR, "route": s_enum(
        "requirement", "objective", "cap", "stock", "budget", "decline_rule", "procedure", "reference", "hint",
        "story", "restatement")}, ("span", "route")), 80),
    "lines": s_arr(s_obj({
        "n": S_INT, "words": S_STR, "shelf_words": S_STR, "qty_n": S_NUM, "qty_span": S_STR,
        "line_role": s_enum("base", "accessory", "independent"),
        "reqs": s_arr(s_obj({"attr": S_STR, "op": s_enum("eq", "in", "ne", "ge", "le", "between"),
                             "vals": s_arr(S_STR, 12), "num": S_NUM, "unit": S_STR, "span": S_STR},
                            ("attr", "op", "span")), 20),
    }, ("words",)), 8),
    "refs": s_arr(s_obj({"line": S_INT, "attr": S_STR, "rel": s_enum("equal", "differ", "pair"),
                         "merchant": S_STR, "shelf_words": S_STR, "descriptors": s_arr(S_PAIR, 8),
                         "span": S_STR}, ("attr", "rel", "span")), 8),
    "cap": s_obj({"amount": S_NUM, "unit": S_STR, "span": S_STR}),
    "stock_rule": s_obj({"required": S_BOOL, "span": S_STR}),
    "objective": s_obj({"kind": s_enum("cheapest", "any"), "span": S_STR}),
    "msgs": _SPANNED_NUM,
    "steps": s_obj({"n": S_NUM, "kind": s_enum("steps", "calls"), "span": S_STR}),
    "decline_rule": s_obj({"allowed": S_BOOL, "no_substitute": S_BOOL, "remark_rules": s_arr(S_STR, 6),
                           "span": S_STR}),
    "procedures": s_arr(s_obj({"kind": S_STR, "text": S_STR, "n": S_NUM, "span": S_STR}, ("kind",)), 8),
    "compat": s_obj({"required": S_BOOL, "base_line": S_INT, "span": S_STR}),
    "hints": s_arr(s_obj({"line": S_INT, "attr_guess": S_STR, "span": S_STR}, ("attr_guess",)), 12),
}, ("lines",))

FIELDS_RULES = (
    "Task: you receive samples of listings returned by one tool of a shopping environment (samples[].obs is a "
    "list of listings) and the field roles to resolve (roles). For each role give the path to the field inside "
    "one listing as a JSON array of keys from the listing inward, or null when absent. Every path must exist in "
    "the samples.\n"
    "record roles: fr:id, fr:price (the price charged), fr:stock (available or not), fr:title, fr:shelf "
    "(category or department), fr:merchant (seller or shop), fr:maker (brand or manufacturer), fr:overview "
    "(long description), fr:notes (seller note), fr:attrs (attribute table). Put them under record. search "
    "roles (fr:records, fr:count, fr:cont) and cart roles (fr:lines, fr:line_id, fr:line_qty, fr:line_price, "
    "fr:total) go under search and cart when asked. Answer only the roles listed in the payload's roles."
)
FIELDS_OUTPUT = ('{"search": {"fr:records": ["<key>"], "fr:count": ["<key>"], "fr:cont": null}, "record": '
                 '{"fr:id": ["<key>"], "fr:price": ["<key>"]}, "cart": {"fr:lines": ["<key>"]}}')
FIELDS_SCHEMA = s_obj({"search": s_map(S_PATH), "record": s_map(S_PATH), "cart": s_map(S_PATH)})

RANK_RULES = (
    "Task: you receive the shopper's request and, per order line, its shelf label, its words, the attributes the "
    "request already states, and its menu: attribute keys seen on opened listings, each with sample values and "
    "coverage. For each line choose up to 6 menu keys that the shopper most likely has an unstated personal "
    "requirement about, weighing the story and what shoppers usually specify for this kind of product; a key whose "
    "values do not vary can still be one. Score each key s0to3 (3 = very likely) and quote the request span that "
    "suggests it (\"\" if none). Copy keys exactly as the menu writes them. Never choose keys the request already "
    "states, and never identity, description, price, seller or brand keys; list such menu keys under meta. unseen: "
    "up to 2 keys the story implies that are missing from the menu, each with its verbatim request span. stated_map: "
    "map each stated attribute, as the request words it, to the menu key holding it. shelf_ok: whether the shelf "
    "label fits the line."
)
RANK_OUTPUT = ('{"lines": [{"line": 0, "shelf_ok": true, "keys": [{"key": "<attr>", "s0to3": 2, "span": ""}], '
               '"unseen": [{"key": "<attr>", "span": "..."}], "stated_map": {"<attr words>": "<attr>"}, '
               '"meta": ["<attr>"]}]}')
RANK_SCHEMA = s_obj({"lines": s_arr(s_obj({
    "line": S_INT, "shelf_ok": S_BOOL,
    "keys": s_arr(s_obj({"key": S_STR, "s0to3": S_NUM, "span": S_STR}, ("key",)), 6),
    "unseen": s_arr(s_obj({"key": S_STR, "span": S_STR}, ("key", "span")), 2),
    "stated_map": s_map(S_STR), "meta": s_arr(S_STR, 40)}, ("line",)), 8)}, ("lines",))

REPLY_RULES = (
    "Task: classify the shopper's reply to one question. The payload has the request, the question sent, the reply, "
    "the asked line and key, and the order lines with their labels. cls: disclosure (states a requirement or "
    "preference), indifferent (no preference), confused (did not understand), one_at_a_time (asks for one question "
    "at a time), unclear. For a disclosure give attr word for word as the reply names it (it may differ from the "
    "key asked), "
    "line_phrase (product words the reply names, or \"\"), op (eq, in for a set of allowed values, ne for anything "
    "except a value, ge, le, between, ref), vals (verbatim values), bound {num, unit, raw} for numeric limits, ref "
    "{rel: equal|differ|pair, merchant, shelf_words, descriptors as [attr, value] pairs} when the value is what "
    "another listing states (op ref), and value_span: the verbatim part of the reply stating the value. Values and "
    "bounds must be copied from the reply; numbers must appear in it."
)
REPLY_OUTPUT = ('{"cls": "disclosure", "attr": "<attr>", "line_phrase": "", "op": "eq", "vals": ["<value>"], '
                '"bound": null, "ref": null, "value_span": "..."}')
REPLY_SCHEMA = s_obj({
    "cls": s_enum("disclosure", "indifferent", "confused", "one_at_a_time", "unclear"),
    "attr": S_STR, "line_phrase": S_STR, "op": s_enum("eq", "in", "ne", "ge", "le", "between", "ref"),
    "vals": s_arr(S_STR, 12), "bound": s_obj({"num": S_NUM, "unit": S_STR, "raw": S_STR}),
    "ref": s_obj({"rel": s_enum("equal", "differ", "pair"), "merchant": S_STR, "shelf_words": S_STR,
                  "descriptors": s_arr(S_PAIR, 8)}),
    "value_span": S_STR}, ("cls",))

NOTES_RULES = (
    "Task: read the seller notes of at most 6 listings (rid, head = listing title, notes). For each record list: "
    "restates (attributes the note states about this listing itself: attr, val, span); pointers (the note says "
    "the value of an attribute must be taken from another listing, not from this one: attr, merchant = the "
    "other listing's seller exactly as written, shelf_words = its category words, descriptors = [attr, value] "
    "pairs identifying it, span); not_versions (the note says this listing is not some other version: merchant "
    "or \"\", descriptors of that other version, span); pair_rules (a store rule pairing values of an attribute: "
    "attr, pairs as [value_a, value_b], span). A note saying the item is not another version is never a pointer. "
    "In descriptors give attr only when the note names the attribute, else \"\". Copy every span verbatim from "
    "that record's note and every value, merchant and descriptor verbatim from its span. Omit records whose "
    "note has none of these."
)
NOTES_OUTPUT = ('{"recs": [{"rid": "<id>", "restates": [{"attr": "<attr>", "val": "<value>", "span": "..."}], '
                '"pointers": [{"attr": "<attr>", "merchant": "<seller>", "shelf_words": "...", "descriptors": '
                '[["<attr>", "<value>"]], "span": "..."}], "not_versions": [{"merchant": "<seller>", '
                '"descriptors": [], "span": "..."}], "pair_rules": [{"attr": "<attr>", "pairs": [["<a>", "<b>"]], '
                '"span": "..."}]}]}')
NOTES_SCHEMA = s_obj({"recs": s_arr(s_obj({
    "rid": S_STR,
    "restates": s_arr(s_obj({"attr": S_STR, "val": S_STR, "span": S_STR}, ("attr", "span")), 12),
    "pointers": s_arr(s_obj({"attr": S_STR, "merchant": S_STR, "shelf_words": S_STR,
                             "descriptors": s_arr(S_PAIR, 8), "span": S_STR}, ("attr", "span")), 8),
    "not_versions": s_arr(s_obj({"merchant": S_STR, "descriptors": s_arr(S_PAIR, 8), "span": S_STR},
                                ("span",)), 8),
    "pair_rules": s_arr(s_obj({"attr": S_STR, "pairs": s_arr(S_PAIR, 12), "span": S_STR}, ("attr", "span")), 6),
}, ("rid",)), 6)}, ("recs",))

EVENT_RULES = (
    "Task: type each message (events[], by index i) the shopper or store sent during the task. kind by "
    "meaning: budget (a new total spending limit for the whole order), drop (a requirement no longer matters), "
    "change (a requirement now has another value), quantity (a different number of units), price_shift (the "
    "price of a chosen listing changed), stock_out (a chosen listing is no longer available), other. amount: "
    "the number the message states, or null, with amount_span: the shortest part of the message holding it, "
    "copied verbatim. Never infer a direction (up or down) from the wording. attr: the attribute as the message "
    "names it, copied verbatim; line: the index of the order line (from lines) when identifiable; new_val: the "
    "new value of a change, copied verbatim."
)
EVENT_OUTPUT = ('{"events": [{"i": 0, "kind": "budget", "amount": 0, "amount_span": "...", "attr": null, '
                '"line": null, "new_val": null}]}')
EVENT_SCHEMA = s_obj({"events": s_arr(s_obj({
    "i": S_INT, "kind": s_enum("budget", "drop", "change", "quantity", "price_shift", "stock_out", "other"),
    "amount": S_NUM, "amount_span": S_STR, "attr": S_STR, "line": S_INT, "new_val": S_STR}, ("i", "kind")), 12)},
    ("events",))

JUDGE_RULES = (
    "Task: judge attribute value pairs and convert numbers between units.\n"
    "pairs: a is the value a shopper asked for, b is a listing's value, attr the attribute. rel is \"same\" only "
    "when a and b differ by spelling, inflection, word order or abbreviation, or are true synonyms naming the same "
    "value; a broader, narrower, related or numerically different value is \"different\". Answer every i.\n"
    "nums: raw is a listing value text and n a number written in it; val is n expressed in the target unit, and "
    "unit repeats the target unit exactly as given. Leave out a number whose unit cannot be converted."
)
JUDGE_OUTPUT = '{"pairs": [{"i": 0, "rel": "different"}], "nums": [{"i": 0, "val": 0, "unit": "<unit>"}]}'
JUDGE_SCHEMA = s_obj({"pairs": s_arr(s_obj({"i": S_INT, "rel": s_enum("same", "different")}, ("i", "rel")), 24),
                      "nums": s_arr(s_obj({"i": S_INT, "val": S_NUM, "unit": S_STR}, ("i", "val")), 24)})

COMPAT_RULES = (
    "Task: the payload's heads list 4-8 listings (rid, head = title, overview = the opening of the description). "
    "Find the phrases that introduce codes: the listing's own model code (own), another code for the same listing "
    "(alias), codes of the products it fits or works with (fits), the part code of another listing the text refers "
    "to for more (more_on), and the phrase saying another part also fits a code (also_fits). Copy each anchor "
    "phrase exactly as written but without any code; [] when a kind is absent. For each example record (rid) list "
    "the codes found, copied verbatim."
)
COMPAT_OUTPUT = ('{"anchors": {"own": ["..."], "alias": ["..."], "fits": ["..."], "more_on": ["..."], '
                 '"also_fits": ["..."]}, "examples": [{"rid": "<id>", "own": [], "alias": [], "fits": [], '
                 '"more_on": []}]}')
_CODES = s_arr(S_STR, 12)
COMPAT_SCHEMA = s_obj({"anchors": s_obj({"own": _CODES, "alias": _CODES, "fits": _CODES, "more_on": _CODES,
                                         "also_fits": _CODES}),
                       "examples": s_arr(s_obj({"rid": S_STR, "own": _CODES, "alias": _CODES, "fits": _CODES,
                                                "more_on": _CODES}, ("rid",)), 8)}, ("anchors",))

OPERATOR_RULES = (
    "Task: a shopping agent wants one action done and its own mapping onto the tools is not trusted. Answer with a "
    "single call of exactly one offered tool. payload holds the exact values to send: ids, query, text, item, "
    "quantity, items (a full list of item and quantity), shelf, maker, ceiling, cont (a continuation, passed "
    "unchanged), or handle (a confirmation handle, passed unchanged). Put each value into the argument meant for it, "
    "unchanged. meaning names the mode to choose for a mode argument: sem:text (keyword search), sem:filter_shelf, "
    "sem:filter_maker, sem:filter_any, sem:add, sem:set. For a message tool with line or topic arguments, choose the "
    "literals naming payload.line and payload.key. Never invent ids, text or quantities; leave out arguments you "
    "have no value for; never choose a tool or literal that ends the task without buying."
)
OPERATOR_OUTPUT = "a single tool call"

REVIEW_RULES = (
    "Task: check a planned purchase against the shopper's request before anything goes into the cart. payload.ledger "
    "is what was understood (lines with quantities, constraints, the whole-basket cap in cents, budgets, rules); "
    "payload.cart is the planned cart. List only requirements the ledger or the cart misses or gets wrong. kind: "
    "quantity (a line's quantity is wrong), cap (the spending limit is wrong or missing), missing_constraint (a stated "
    "requirement on a line is missing), missing_line (a wanted product has no line), other. line: the line index. "
    "request_span: the request words stating it, copied verbatim. attr: the attribute as the request words it. op: eq, "
    "or ne for an exclusion. val: the number, value or product words exactly as written in the span. explain: a few "
    "words. Story, background and restatement sentences are not requirements. clean is true when nothing is missing."
)
REVIEW_OUTPUT = ('{"clean": false, "issues": [{"kind": "missing_constraint", "line": 0, "request_span": "...", '
                 '"attr": "<attr>", "op": "eq", "val": "<value>", "explain": "..."}]}')
REVIEW_SCHEMA = s_obj({"clean": S_BOOL, "issues": s_arr(s_obj({
    "kind": s_enum("quantity", "cap", "missing_constraint", "missing_line", "other"), "line": S_INT,
    "request_span": S_STR, "attr": S_STR, "op": s_enum("eq", "ne"), "explain": S_STR, "val": S_STR},
    ("kind", "request_span")), 12)}, ("clean",))

AUDIT_RULES = (
    "Task: audit a proposed decision to end the task without buying. The payload holds the request, the proof (its "
    "kind and numbers, money in cents), the order lines (label, words, units, requirements with their source) and "
    "compact records of each line's candidates (rid, line, head, cost_c, instock, read = its full page was opened, "
    "on_label, attrs). orderable: every record that could satisfy its line: nothing in its own fields or head "
    "contradicts a requirement, it is in stock, and its cost leaves room for the other lines within the budget; give "
    "line, rid and why, and list none when no record qualifies. Use only rids from the records. proof_sound: false "
    "when the proof rests on something the shopper never required (story words, descriptors that only identify "
    "another listing, a guessed attribute), on the wrong shelf, or on a misread budget; true otherwise. "
    "remark_wanted: true only when the request asks for a closing note or explanation if nothing is bought. issues: "
    "short notes."
)
AUDIT_OUTPUT = ('{"orderable": [{"line": 0, "rid": "<id>", "why": "..."}], "proof_sound": true, "remark_wanted": false, '
                '"issues": ["..."]}')
AUDIT_SCHEMA = s_obj({"orderable": s_arr(s_obj({"line": S_INT, "rid": S_STR, "why": S_STR}, ("rid",)), 40),
                      "proof_sound": S_BOOL, "remark_wanted": S_BOOL, "issues": s_arr(S_STR, 12)}, ("proof_sound",))

PROC_RULES = (
    "Task: map each procedure the shopper requires onto one step a shopping agent can schedule. proc_i: the "
    "procedure index. when: before_order, before_first_write, after_selection or any. action: message (send the "
    "shopper a message), open (open the listings being bought), view_count (view at least a number of listings) or "
    "other. what: a few words."
)
PROC_OUTPUT = '{"steps": [{"proc_i": 0, "when": "before_order", "action": "open", "what": "..."}]}'
PING_RULES = "Task: availability check. Answer with the object shown, unchanged."
PING_OUTPUT = '{"pong": true}'
PING_SCHEMA = s_obj({"pong": S_BOOL}, ("pong",))

PROC_SCHEMA = s_obj({"steps": s_arr(s_obj({
    "proc_i": S_INT, "when": s_enum("before_order", "before_first_write", "after_selection", "any"),
    "action": s_enum("message", "open", "view_count", "other"), "what": S_STR}, ("proc_i", "action")), 12)},
    ("steps",))

_ROLE_SPECS = (
    RoleSpec("BIND", "BIND", BIND_RULES, BIND_OUTPUT, BIND_SCHEMA, 900, LADDER_CHEAP, 25.0),
    RoleSpec("BIND_ARBITER", "BIND", BIND_RULES, BIND_OUTPUT, BIND_SCHEMA, 900, LADDER_MID, 20.0),
    RoleSpec("PARSE", "PARSE", PARSE_RULES, PARSE_OUTPUT, PARSE_SCHEMA, 1400, LADDER_CHEAP, 25.0),
    RoleSpec("PARSE_ARBITER", "PARSE", PARSE_RULES, PARSE_OUTPUT, PARSE_SCHEMA, 1400, LADDER_MID, 20.0),
    RoleSpec("FIELDS", "FIELDS", FIELDS_RULES, FIELDS_OUTPUT, FIELDS_SCHEMA, 400, (MODEL_HAIKU, MODEL_LUNA), 15.0),
    RoleSpec("RANK", "RANK", RANK_RULES, RANK_OUTPUT, RANK_SCHEMA, 700, LADDER_CHEAP, 25.0),
    RoleSpec("REPLY", "REPLY", REPLY_RULES, REPLY_OUTPUT, REPLY_SCHEMA, 300, (MODEL_HAIKU, MODEL_LUNA), 8.0),
    RoleSpec("NOTES", "NOTES", NOTES_RULES, NOTES_OUTPUT, NOTES_SCHEMA, 700, (MODEL_LUNA, MODEL_HAIKU), 15.0),
    RoleSpec("EVENT", "EVENT", EVENT_RULES, EVENT_OUTPUT, EVENT_SCHEMA, 300, (MODEL_HAIKU, MODEL_LUNA), 8.0),
    RoleSpec("JUDGE", "JUDGE", JUDGE_RULES, JUDGE_OUTPUT, JUDGE_SCHEMA, 400, (MODEL_HAIKU, MODEL_LUNA), 15.0),
    RoleSpec("COMPAT", "COMPAT", COMPAT_RULES, COMPAT_OUTPUT, COMPAT_SCHEMA, 600, (MODEL_LUNA, MODEL_HAIKU), 25.0),
    RoleSpec("OPERATOR", "OPERATOR", OPERATOR_RULES, OPERATOR_OUTPUT, None, 300, (MODEL_HAIKU, MODEL_LUNA), 25.0),
    RoleSpec("REVIEW", "REVIEW", REVIEW_RULES, REVIEW_OUTPUT, REVIEW_SCHEMA, 600, (MODEL_V4PRO, MODEL_FLASH), 25.0,
             optional=True),
    RoleSpec("AUDIT", "AUDIT", AUDIT_RULES, AUDIT_OUTPUT, AUDIT_SCHEMA, 700, LADDER_STRONG, 25.0),
    RoleSpec("AUDIT_E1", "AUDIT", AUDIT_RULES, AUDIT_OUTPUT, AUDIT_SCHEMA, 700, (MODEL_V4PRO, MODEL_FLASH), 25.0),
    RoleSpec("PROC", "PROC", PROC_RULES, PROC_OUTPUT, PROC_SCHEMA, 300, (MODEL_LUNA, MODEL_HAIKU), 25.0),
    RoleSpec("PING", "PING", PING_RULES, PING_OUTPUT, PING_SCHEMA, 20, LADDER_CHEAP, 8.0),
)
ROLE_REGISTRY = {spec.key: spec for spec in _ROLE_SPECS}
LATE_ROLES = ("REPLY", "EVENT")
SOFT_SKIP_ROLES = ("REVIEW", "COMPAT", "JUDGE")

PRICES = (
    (MODEL_HAIKU, 0.10, 0.50),
    (MODEL_LUNA, 0.10, 0.50),
    (MODEL_V32, 0.28, 0.46),
    (MODEL_V4PRO, 0.62, 1.60),
    (MODEL_FLASH, 0.75, 3.75),
    (MODEL_SONNET, 3.00, 15.00),
    (MODEL_SOL, 2.50, 10.00),
    (MODEL_V4FLASH, 0.07, 0.19),
)  # USD per million tokens (input, output), as billed in the first real run; only when usage.cost is missing

_MODEL_BLACKLIST: set = set()        # process-wide (DESIGN A8): models answering 400/403 that name the model
_BLACKLIST_LOCK = threading.Lock()

_RZ_POLICY = ((MODEL_V32, "none_only"), (MODEL_V4PRO, "none_only"), (MODEL_V4FLASH, "none_only"),
              (MODEL_QWEN, "none_only"), (MODEL_SONNET, "budgeted"), (MODEL_FLASH, "budgeted"))
_RZ_MODE: dict = {}                  # learned: model -> "budgeted"; absent means reasoning off
_RZ_LOCK = threading.Lock()
_RZ_HEADROOM = 1152
_RZ_SLACK = 32                       # reasoning within this of completion_tokens took the budget
_RZ_DEMOTE_AT = 2                    # exhaustions per episode before a model goes last in every ladder
_RZ_TRUNC_CAP = 4096                 # bound of the x1.5 retry of a truncated answer
_RZ_MSG_RE = re.compile(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"')

def _rz_policy(model: str) -> str:
    """"none_only", "budgeted" (seeded) or "learn"."""
    for name, policy in _RZ_POLICY:
        if name == model:
            return policy
    return "learn"

def _rz_mode(model: str) -> str:
    policy = _rz_policy(model)
    if policy != "learn":
        return "none" if policy == "none_only" else "budgeted"
    with _RZ_LOCK:
        return _RZ_MODE.get(model, "none")

def _rz_request(model: str, max_tokens: int) -> tuple:
    """(mode, reasoning setting, max_tokens to send)."""
    if _rz_mode(model) == "budgeted":
        return "budgeted", {"effort": "minimal"}, int(max_tokens) + _RZ_HEADROOM
    return "none", {"effort": "none"}, int(max_tokens)

def _rz_message(text: Any) -> str:
    """The error's own message: error.message of a JSON body (and of a provider's JSON error in metadata.raw), or
    the message strings of a clipped one, never an echoed prompt. A JSON body without a message reads as empty; only
    a body holding no JSON at all is read whole."""
    raw = str(text or "")
    try:
        obj = json.loads(raw)
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        err = obj.get("error", obj)
        msg = err.get("message") if isinstance(err, dict) else err
        meta = err.get("metadata") if isinstance(err, dict) else None
        inner = meta.get("raw") if isinstance(meta, dict) else None
        inner = _rz_message(inner) if isinstance(inner, str) and inner.lstrip()[:1] == "{" else ""
        return " ".join(m for m in (msg if isinstance(msg, str) else "", inner) if m)
    found = _RZ_MSG_RE.findall(raw.replace('\\"', '"'))
    if found:
        return " ".join(found)
    return "" if raw.lstrip()[:1] in ("{", "[") else raw

def _rz_rejects(status: Optional[int], text: Any) -> bool:
    """An HTTP 400 about the reasoning setting (mandatory reasoning, an unsupported effort), read from the error's
    message; never a size rejection (two or more numbers of 1000 or more)."""
    if status != 400:
        return False
    low = _rz_message(text).lower()
    return sum(n[0] >= 1000 for n in numbers_in(low)) < 2 and (
        "reasoning" in low or "effort" in low or "thinking" in low)

def _rz_int(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

def _rz_usage(data: Any) -> tuple:
    """(finish_reason, completion_tokens, reasoning_tokens); None where absent."""
    finish = ct = rt = None
    if isinstance(data, dict):
        try:
            finish = (data.get("choices") or [{}])[0].get("finish_reason")
        except (AttributeError, IndexError, TypeError):
            pass
        usage = data.get("usage")
        if isinstance(usage, dict):
            ct = usage.get("completion_tokens")
            det = usage.get("completion_tokens_details")
            rt = det.get("reasoning_tokens") if isinstance(det, dict) else usage.get("reasoning_tokens")
    return finish, _rz_int(ct), _rz_int(rt)

def _rz_classify(data: Any) -> str:
    """An unusable answer's length stop: "exhausted" (reported reasoning took the budget), "truncated" or "" (no
    length stop). Zero reported reasoning is a truncation; with none reported, only an empty answer is exhausted."""
    finish, ct, rt = _rz_usage(data)
    if finish != "length":
        return ""
    if rt == 0:
        return "truncated"
    if rt is None or ct is None:
        try:
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
        except (AttributeError, IndexError, TypeError):
            content = None
        return "truncated" if str(content or "").strip() else "exhausted"
    return "exhausted" if rt >= ct - _RZ_SLACK else "truncated"

def _rz_longer(max_tokens: int) -> int:
    return max(int(max_tokens), min(int(max_tokens * 1.5), _RZ_TRUNC_CAP))

def price_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    for name, per_in, per_out in PRICES:
        if name == model:
            return (prompt_tokens * per_in + completion_tokens * per_out) / 1_000_000.0
    return (prompt_tokens * 0.5 + completion_tokens * 1.5) / 1_000_000.0

def _first_json_object(text: str) -> Optional[dict]:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except ValueError:
                        obj = None
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    return None

def extract_json(text: Any) -> Optional[dict]:
    """Strip code fences and return the first balanced JSON object, or None."""
    if not isinstance(text, str):
        return None
    if "```" in text:
        pieces = text.split("```")
        for i in range(1, len(pieces), 2):
            block = pieces[i]
            if block[:4].lower() == "json":
                block = block[4:]
            obj = _first_json_object(block)
            if obj is not None:
                return obj
    return _first_json_object(text)

@dataclass
class LlmJob:
    key: str                                   # ROLE_REGISTRY key, e.g. "BIND" or "PARSE_ARBITER"
    payload: Any
    models: Optional[tuple] = None             # override the role's ladder
    validator: Optional[Callable] = None       # semantic validator: value -> (ok, value, note)
    max_tokens: Optional[int] = None
    cap: Optional[float] = None
    optional: bool = False
    label: str = ""

@dataclass
class _HttpOutcome:
    status: Optional[int]
    data: Any
    error_text: str
    ms: float

TRANSIENT_SPAN_S = 10.0          # transient failures count toward quick-dead only across >= 10 s
RETRY_BACKOFF_S = 1.0            # one same-model retry after 429/5xx
MIN_CALL_S = 4.0                 # never start a call that cannot get this long before T_LLM_DEADLINE
START_POLL_S = 0.1               # ask_many re-reads a not-yet-started call's deadline this often

def _transient_status(status: Optional[int]) -> bool:
    return status is None or status == 429 or (isinstance(status, int) and status >= 500)

class _DaemonPool:
    """Elastic pool of daemon worker threads. Interpreter shutdown never joins daemon threads, so an
    abandoned inference call cannot hold the sandbox child open past its deadline; and while fewer than
    MAX_WORKERS threads exist a new call never queues behind abandoned ones."""
    MAX_WORKERS = 12

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.jobs: Any = queue.Queue()
        self.lock = threading.Lock()
        self.threads = 0
        self.idle = 0
        self.closed = False

    def submit(self, fn: Callable, *args: Any) -> Any:
        fut: Any = concurrent.futures.Future()
        fut.started_at = None
        with self.lock:
            if self.closed:
                raise RuntimeError("pool closed")
            spawn = self.jobs.qsize() + 1 > self.idle and self.threads < self.MAX_WORKERS
            if spawn:
                self.threads += 1
        self.jobs.put((fut, fn, args))
        if spawn:
            threading.Thread(target=self._work, name=f"{self.prefix}-{self.threads}", daemon=True).start()
        return fut

    def _work(self) -> None:
        while True:
            with self.lock:
                self.idle += 1
            item = self.jobs.get()
            with self.lock:
                self.idle -= 1
            if item is None:
                return
            fut, fn, args = item
            if not fut.set_running_or_notify_cancel():
                continue
            fut.started_at = monotonic()
            try:
                result = fn(*args)
            except BaseException as exc:  # noqa: BLE001 - delivered through the future
                fut.set_exception(exc)
            else:
                fut.set_result(result)

    def shutdown(self) -> None:
        with self.lock:
            self.closed = True
            threads = self.threads
        while True:
            try:
                item = self.jobs.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                item[0].cancel()
        for _ in range(threads):
            self.jobs.put(None)

class _JobRun:
    """State of one job's ladder walk inside ask_many (main thread only)."""

    def __init__(self, job: LlmJob, spec: RoleSpec, ladder: list, messages: list):
        self.job, self.spec, self.ladder = job, spec, ladder
        self.base_messages = messages
        self.model_i = 0
        self.repairing = False
        self.messages = messages
        self.future = None
        self.timeout = 0.0
        self.submitted = 0.0
        self.started = 0.0
        self.retry_at = 0.0                  # a same-model retry waits until this monotonic time
        self.retried = False                 # the current model already had its transient retry
        self.done = False
        self.value = None
        self.attempts = 0
        self.hard_fail = False               # some model failed for a non-transient reason
        self.index = 0
        self.rz_sent: Optional[tuple] = None   # (reasoning mode, max_tokens) of the attempt in flight
        self.rz_used = False                   # this model had its reasoning / truncation retry
        self.longer = False                    # max_tokens x1.5 (this model)

    @property
    def model(self) -> Optional[str]:
        return self.ladder[self.model_i] if self.model_i < len(self.ladder) else None

    def deadline(self) -> float:
        """The call's deadline, counted from the moment a worker started it (queued time is not the
        model's fault; a call still queued gets twice its timeout from submission)."""
        started = getattr(self.future, "started_at", None) if self.future is not None else None
        if started is None:
            return self.submitted + 2 * self.timeout
        return started + self.timeout

class Llm:
    def __init__(self, ep: Any, client: Any = None):
        self.ep = ep
        self.client = client if client is not None else ProxyClient(timeout=30, max_retries=1)
        self.pool = _DaemonPool("lodestar-llm")
        self.cache: dict = {}
        self.cost_lock = threading.Lock()
        self.abandoned: set = set()
        self.late_calls = 0
        self.fail_streak = 0
        self.failed_roles: list = []
        self.calls_by_role: dict = {}
        self.any_success = False
        self.failed_models: set = set()      # models that failed hard (400/403, invalid output after repair)
        self.transient_models: set = set()   # models that failed transiently (429, 5xx, network, timeout)
        self.first_fail_t: Optional[float] = None   # monotonic time of the first failure since the last success
        self.provisional_dead = False        # dead_inference was set by transient-looking failures
        self.reprobed = False
        self.rz_exhausted: dict = {}         # model -> budgeted-mode exhaustions this episode

    def regime(self) -> str:
        b = self.ep.budget
        if b.dead_inference or b.usd >= USD_STOP:
            r = "stop"
        elif b.usd >= USD_HARD:
            r = "hard"
        elif b.usd >= USD_SOFT:
            r = "soft"
        else:
            r = "normal"
        if r != b.regime:
            b.regime = r
            self.ep.trace.decision("regime", regime=r, usd=round(b.usd, 5), dead=b.dead_inference)
        return r

    def overrun(self) -> bool:
        """DESIGN 3.5 overrun projection: the exploration steps still planned (to u_c_soft) at the running mean
        environment latency, plus one oracle call, would end after 190 s: optional oracle work is skipped."""
        plan = getattr(self.ep, "plan", None)
        return plan is not None and plan.overrun_risk(plan.explore_steps_left(), 1)

    def mark_dead(self, reason: str, provisional: bool = False) -> None:
        if not self.ep.budget.dead_inference:
            self.ep.budget.dead_inference = True
            self.provisional_dead = provisional
            self.ep.trace.decision("oracles_dead", reason=reason, regime="stop", provisional=provisional)
        elif not provisional:
            self.provisional_dead = False

    def allowed(self, key: str, optional: bool = False) -> bool:
        spec = ROLE_REGISTRY.get(key)
        if spec is None:
            return False
        t = self.ep.clock()
        if t >= T_FALLBACK:
            return False
        r = self.regime()
        if r == "stop":
            return False
        late_role = spec.role in LATE_ROLES
        if (r == "hard" or t >= T_DETERMINISTIC_CHECKOUT) and not late_role:
            return False
        optional = optional or spec.optional
        if r == "soft" and (optional or key in SOFT_SKIP_ROLES):
            return False
        if optional and (t >= T_OPTIONAL_LLM_STOP or self.overrun()):
            return False
        return True

    def vote_ladders(self, key: str, votes: int) -> list:
        """Per-vote ladders for parallel votes (BIND, PARSE: 2; RANK: 3). One vote after 150 s, in the soft
        regime, or when the overrun projection says time is short (DESIGN 3.5, 5.1)."""
        if self.regime() != "normal" or self.ep.clock() >= T_OPTIONAL_LLM_STOP or self.overrun():
            votes = 1
        base = self._rz_order(list(ROLE_REGISTRY[key].ladder))
        with _BLACKLIST_LOCK:                # votes start on different usable models
            usable = [m for m in base if m not in _MODEL_BLACKLIST and self.rz_exhausted.get(m, 0) < _RZ_DEMOTE_AT]
        base = usable + [m for m in base if m not in usable]
        votes = min(votes, max(1, len(usable)))
        out = []
        for i in range(min(votes, len(base))):
            out.append(tuple(base[i:] + base[:i]))
        return out

    def _ladder(self, job: LlmJob, spec: RoleSpec) -> list:
        models = list(job.models or spec.ladder)
        with _BLACKLIST_LOCK:
            models = [m for m in models if m not in _MODEL_BLACKLIST]
        models = self._rz_order(models)
        if self.regime() == "hard" or self.ep.clock() >= T_DETERMINISTIC_CHECKOUT:
            cheap = self._rz_order([m for m in LADDER_CHEAP if m in models]) or models
            models = cheap[:1]
        return models

    def _rz_order(self, models: list) -> list:
        """Demoted models last (stable)."""
        late = [m for m in models if self.rz_exhausted.get(m, 0) >= _RZ_DEMOTE_AT]
        return [m for m in models if m not in late] + late

    def _timeout(self, spec: RoleSpec, job: LlmJob) -> float:
        cap = job.cap or spec.cap
        if self.regime() == "hard" or self.ep.clock() >= T_DETERMINISTIC_CHECKOUT:
            cap = min(cap, 8.0)
        return float(clamp(min(cap, T_LLM_DEADLINE - self.ep.clock()), MIN_CALL_S, cap))

    def _may_start(self) -> bool:
        """A call starts only if its shortest timeout still ends by T_LLM_DEADLINE."""
        return self.ep.clock() + MIN_CALL_S <= T_LLM_DEADLINE

    def _http(self, model: str, messages: list, max_tokens: int, tools: Optional[list] = None,
              reasoning: Optional[dict] = None) -> _HttpOutcome:
        body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": int(max_tokens)}
        if reasoning is not None:
            body["reasoning"] = reasoning
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        t0 = monotonic()
        try:
            res = self.client.post_verbose(INFERENCE_PATH, body)
        except Exception as exc:  # network stack errors become a failed attempt
            return _HttpOutcome(None, None, repr(exc)[:300], (monotonic() - t0) * 1000.0)
        ms = (monotonic() - t0) * 1000.0
        data = getattr(res, "data", None)
        if data is not None:
            return _HttpOutcome(200, data, "", ms)
        err = getattr(res, "error", None) or {}
        status = err.get("status") if isinstance(err, dict) else None
        text = err.get("body") if isinstance(err, dict) else err
        return _HttpOutcome(status if isinstance(status, int) else None, None, clip_text(text, 400), ms)

    def _account(self, future: Any, model: str, token: int, sent_tokens: int = 0) -> None:
        """Done-callback (worker thread): cost accounting only, under a lock. usage.cost, else prices x tokens with
        billed reasoning; a length stop without usage bills max_tokens."""
        try:
            out = future.result()
        except Exception:
            return
        usd = 0.0
        if out.status == 200 and isinstance(out.data, dict):
            usage = out.data.get("usage") or {}
            cost = usage.get("cost") if isinstance(usage, dict) else None
            finish, ct, rt = _rz_usage(out.data)
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
                usd = float(cost)
            elif isinstance(usage, dict) and usage:
                ct = ct or 0
                usd = price_of(model, int(usage.get("prompt_tokens") or 0), ct + (rt if rt and rt > ct else 0))
            elif finish == "length":
                usd = price_of(model, 0, sent_tokens)
        with self.cost_lock:
            self.ep.budget.usd += usd
            self.ep.budget.llm_calls += 1
            if token in self.abandoned:
                self.late_calls += 1

    def _submit(self, model: str, messages: list, max_tokens: int, tools: Optional[list] = None) -> tuple:
        token = id(messages) ^ int(monotonic() * 1e6)
        mode, reasoning, sent = _rz_request(model, max_tokens)
        fut = self.pool.submit(self._http, model, messages, sent, tools, reasoning)
        fut.rz_sent = (mode, sent)
        fut.add_done_callback(lambda f, m=model, k=token, n=sent: self._account(f, m, k, n))
        return fut, token

    def ask(self, key: str, payload: Any, *, models: Optional[tuple] = None, validator: Optional[Callable] = None,
            max_tokens: Optional[int] = None, cap: Optional[float] = None, optional: bool = False) -> Optional[dict]:
        return self.ask_many([LlmJob(key, payload, models, validator, max_tokens, cap, optional)])[0]

    def ask_many(self, jobs: list, deadline: Optional[float] = None) -> list:
        """Run jobs in parallel; each walks its ladder (repair once per model, one same-model retry after a
        429/5xx, then the next model). deadline: episode clock seconds after which unfinished jobs are
        abandoned (setup uses it)."""
        results: list = [None] * len(jobs)
        runs: list = []
        for idx, job in enumerate(jobs):
            spec = ROLE_REGISTRY.get(job.key)
            if spec is None or spec.schema is None or not self.allowed(job.key, job.optional):
                continue
            ladder = self._ladder(job, spec)
            if not ladder:
                continue
            cached = self._cache_get(spec, ladder, job.payload)
            if cached is not None:
                results[idx] = cached
                continue
            messages = [{"role": "system", "content": spec.system_prompt()},
                        {"role": "user", "content": dump_json(job.payload)}]
            run = _JobRun(job, spec, ladder, messages)
            run.index = idx
            runs.append(run)
        tokens: dict = {}
        while True:
            if self.ep.budget.dead_inference:
                break
            active = [r for r in runs if not r.done]
            if not active:
                break
            if deadline is not None and self.ep.clock() >= deadline:
                for run in active:
                    if run.future is not None:
                        with self.cost_lock:
                            self.abandoned.add(tokens.get(id(run)))
                    run.done = True
                break
            now = monotonic()
            for run in active:
                if run.future is None and run.retry_at <= now:
                    if run.model is None or not self._may_start():
                        run.done = True
                        continue
                    run.attempts += 1
                    run.timeout = self._timeout(run.spec, run.job)
                    run.submitted = run.started = monotonic()
                    mt = run.job.max_tokens or run.spec.max_tokens
                    run.future, tok = self._submit(run.model, run.messages, _rz_longer(mt) if run.longer else mt)
                    run.rz_sent = getattr(run.future, "rz_sent", None)
                    tokens[id(run)] = tok
            live = [r for r in runs if not r.done]
            pending = [r.future for r in live if r.future is not None]
            waits = [r.deadline() for r in live if r.future is not None] + \
                    [r.retry_at for r in live if r.future is None and r.retry_at > now]
            if not waits:
                continue
            wait_for = max(0.05, min(waits) - monotonic())
            if deadline is not None:
                wait_for = max(0.05, min(wait_for, deadline - self.ep.clock()))
            if any(getattr(f, "started_at", None) is None for f in pending):
                wait_for = min(wait_for, START_POLL_S)
            if pending:
                concurrent.futures.wait(pending, timeout=wait_for, return_when=concurrent.futures.FIRST_COMPLETED)
            else:
                time.sleep(min(wait_for, RETRY_BACKOFF_S))
            now = monotonic()
            for run in runs:
                if run.done or run.future is None:
                    continue
                if run.future.done():
                    try:
                        outcome = run.future.result()
                    except Exception as exc:  # cancelled or failed worker: a network-class failure
                        outcome = _HttpOutcome(None, None, repr(exc)[:200], 0.0)
                    run.future = None
                    self._process(run, outcome)
                elif now >= run.deadline():
                    with self.cost_lock:
                        self.abandoned.add(tokens.get(id(run)))
                    self._log_attempt(run, None, (now - run.started) * 1000.0, False, "timeout")
                    self._model_failed(run.model, transient=True)
                    run.future = None
                    self._next_model(run)
            self._check_dead()
        for run in runs:
            results[run.index] = run.value
            self._note_outcome(run)
        return results

    def tool_call(self, key: str, messages: list, tools: list, models: Optional[tuple] = None,
                  max_tokens: Optional[int] = None, cap: Optional[float] = None) -> Optional[tuple]:
        """OPERATOR-style call: one tool call from the offered tools, or None (DESIGN 5.20). Governed like ask():
        every attempt is logged; 429/5xx get one same-model retry; 400/403 naming the model blacklist it; 401/402
        kill inference; a timeout, a network error or an answer without a usable tool call moves down the ladder.
        A reasoning 400 or a length stop is handled as in ask()."""
        spec = ROLE_REGISTRY.get(key)
        if spec is None or not tools or not self.allowed(key):
            return None
        names = {str((t.get("function") or {}).get(NAME_KEY)) for t in tools if isinstance(t, dict)}
        msgs = [{"role": "system", "content": spec.system_prompt()}] + list(messages)
        for i, model in enumerate(self._ladder(LlmJob(key, None, models), spec)):
            retry = rz_used = longer = False
            while True:
                if not self._may_start() or self.ep.budget.dead_inference:
                    return None
                mt = max_tokens or spec.max_tokens
                fut, tok = self._submit(model, msgs, _rz_longer(mt) if longer else mt, tools)
                sent = getattr(fut, "rz_sent", None)
                t0 = monotonic()
                try:
                    out = fut.result(timeout=self._timeout(spec, LlmJob(key, None, cap=cap)))
                except Exception as exc:          # our timeout, or a failed worker: transient, next model
                    with self.cost_lock:
                        self.abandoned.add(tok)
                    timed_out = isinstance(exc, concurrent.futures.TimeoutError)
                    self._log(key, model, None, (monotonic() - t0) * 1000.0, False, i > 0,
                              "timeout" if timed_out else "network")
                    self._model_failed(model, transient=True)
                    break
                try:
                    self.ep.plan.llm_ms.append(out.ms)
                except AttributeError:
                    pass
                if out.status == 200 and isinstance(out.data, dict):
                    pick = self._tool_pick(out.data, names)
                    kind = "" if pick is not None else _rz_classify(out.data)
                    self._log(key, model, out, out.ms, pick is not None, i > 0 or retry or rz_used,
                              "" if pick is not None else (kind or "no usable tool call"))
                    if pick is not None:
                        self.any_success = True
                        self.first_fail_t = None
                        return pick
                    act = self._rz_action(key, model, out.data, kind, sent, rz_used) if kind else "next"
                    if act != "next":
                        rz_used, longer = True, longer or act == "longer"
                        continue
                    break
                note = f"http {out.status}" if out.status else "network"
                self._log(key, model, out, out.ms, False, i > 0 or retry or rz_used, note)
                if out.status in (401, 402):
                    self.mark_dead(note)
                    return None
                if _rz_rejects(out.status, out.error_text):     # never a blacklist
                    if self._rz_reject(key, model, sent, rz_used):
                        rz_used = True
                        continue
                    self._model_failed(model, transient=True)   # as in ask_many
                    break
                if out.status in (400, 403) and model.lower() in out.error_text.lower():
                    with _BLACKLIST_LOCK:
                        _MODEL_BLACKLIST.add(model)
                if out.status is not None and _transient_status(out.status) and not retry \
                        and self.ep.clock() + RETRY_BACKOFF_S + MIN_CALL_S <= T_LLM_DEADLINE:
                    retry = True
                    time.sleep(RETRY_BACKOFF_S)   # 429 / 5xx: the same model once more
                    continue
                self._model_failed(model, transient=_transient_status(out.status))
                break
        return None

    @staticmethod
    def _tool_pick(data: dict, names: set) -> Optional[tuple]:
        try:
            message = (data.get("choices") or [{}])[0].get("message") or {}
            call = (message.get("tool_calls") or [])[0].get("function") or {}
            raw_args = call.get("arguments")
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except (IndexError, AttributeError, TypeError, ValueError):
            return None
        name = call.get(NAME_KEY)
        return (name, args) if name in names and isinstance(args, dict) else None

    def reprobe(self) -> bool:
        """One cheap PING when inference died provisionally (transient-looking failures); a valid answer
        revives the oracles for the rest of the episode. Returns True when revived."""
        b = self.ep.budget
        if not (b.dead_inference and self.provisional_dead) or self.reprobed:
            return False
        self.reprobed = True
        if self.ep.clock() >= T_OPTIONAL_LLM_STOP or b.usd >= USD_SOFT:
            return False
        b.dead_inference = False
        self.failed_models.clear()
        self.transient_models.clear()
        self.first_fail_t = None
        self.fail_streak, self.failed_roles = 0, []
        with _BLACKLIST_LOCK:
            cheap = [m for m in LADDER_CHEAP if m not in _MODEL_BLACKLIST]
        cheap = self._rz_order(cheap)
        answer = self.ask("PING", {"ping": True}, models=tuple(cheap[:1]), cap=8.0) if cheap else None
        ok = isinstance(answer, dict) and answer.get("pong") is True
        if not ok:
            b.dead_inference = True
        else:
            self.provisional_dead = False
        self.ep.trace.decision("oracles_reprobe", revived=ok)
        return ok

    def close(self) -> None:
        try:
            self.pool.shutdown()
        except Exception:
            pass

    def _cache_get(self, spec: RoleSpec, ladder: list, payload: Any) -> Optional[dict]:
        for model in ladder:
            hit = self.cache.get(stable_hash(spec.key, model, payload))
            if hit is not None:
                return hit
        return None

    def _process(self, run: _JobRun, out: _HttpOutcome) -> None:
        model = run.model
        try:
            self.ep.plan.llm_ms.append(out.ms)
        except AttributeError:
            pass
        if out.status == 200 and isinstance(out.data, dict):
            content = None
            try:
                content = ((out.data.get("choices") or [{}])[0].get("message") or {}).get("content")
            except (AttributeError, IndexError, TypeError):
                content = None
            obj = extract_json(content)
            ok, value, note = False, None, "no json object"
            if obj is not None:
                ok, value = conform(obj, run.spec.schema)
                note = "" if ok else "schema mismatch"
                if ok and run.job.validator is not None:
                    try:
                        ok, value, note = run.job.validator(value)
                    except Exception as exc:
                        ok, note = False, f"validator error {exc!r}"[:120]
            kind = "" if ok else _rz_classify(out.data)
            self._log_attempt(run, out, out.ms, ok, kind or note)
            if ok:
                self.any_success = True
                self.first_fail_t = None
                run.done = True
                run.value = value
                self.cache[stable_hash(run.spec.key, model, run.job.payload)] = value
                return
            if kind:                            # a length stop: never the same-model repair re-ask
                act = self._rz_action(run.spec.key, model, out.data, kind, run.rz_sent, run.rz_used)
                if act != "next":               # resubmitted at once
                    run.rz_used, run.longer = True, run.longer or act == "longer"
                    run.messages = run.base_messages
                    return
                run.hard_fail = True
                self._model_failed(model, transient=False)
                self._next_model(run)
                return
            if not run.repairing:
                run.repairing = True
                run.messages = run.base_messages + [{"role": "assistant", "content": str(content or "")[:4000]},
                                                    {"role": "user", "content": REPAIR_TEXT}]
                return
            run.hard_fail = True
            self._model_failed(model, transient=False)
            self._next_model(run)
            return
        note = f"http {out.status}" if out.status else "network"
        self._log_attempt(run, out, out.ms, False, note)
        if out.status in (401, 402):
            self.mark_dead(note)
            run.done = True
            return
        if model and _rz_rejects(out.status, out.error_text):     # never a blacklist, never a hard failure
            if self._rz_reject(run.spec.key, model, run.rz_sent, run.rz_used):
                run.rz_used = True
                run.messages = run.base_messages
                return
            self._model_failed(model, transient=True)
            self._next_model(run)
            return
        if out.status in (400, 403) and model and model.lower() in out.error_text.lower():
            with _BLACKLIST_LOCK:
                _MODEL_BLACKLIST.add(model)
        if out.status is not None and _transient_status(out.status) and not run.retried:
            run.retried = True                  # 429 / 5xx: the same model once more after a short backoff
            run.retry_at = monotonic() + RETRY_BACKOFF_S
            self._note_failure_time()
            return
        transient = _transient_status(out.status)
        if not transient:
            run.hard_fail = True
        self._model_failed(model, transient=transient)
        self._next_model(run)

    def _note_failure_time(self) -> None:
        if self.first_fail_t is None and not self.any_success:
            self.first_fail_t = monotonic()

    def _model_failed(self, model: Optional[str], transient: bool = False) -> None:
        self._note_failure_time()
        if model:
            (self.transient_models if transient else self.failed_models).add(model)

    def _transient_span_ok(self) -> bool:
        return self.first_fail_t is not None and monotonic() - self.first_fail_t >= TRANSIENT_SPAN_S

    def _check_dead(self) -> None:
        """Quick dead-backend detection: before any oracle answered, every available cheap model has failed
        hard, or transiently with the failure streak spanning TRANSIENT_SPAN_S."""
        if self.any_success or self.ep.budget.dead_inference:
            return
        with _BLACKLIST_LOCK:
            cheap = [m for m in LADDER_CHEAP if m not in _MODEL_BLACKLIST]
        if not cheap:
            return
        span_ok = self._transient_span_ok()
        if all(m in self.failed_models or (span_ok and m in self.transient_models) for m in cheap):
            provisional = any(m not in self.failed_models for m in cheap)
            self.mark_dead("every cheap model failed before any oracle answered", provisional=provisional)

    def _next_model(self, run: _JobRun) -> None:
        run.model_i += 1
        run.repairing = False
        run.retried = False
        run.retry_at = 0.0
        run.messages = run.base_messages
        run.rz_used = run.longer = False
        run.rz_sent = None
        if run.model is None:
            run.done = True

    def _rz_switch(self, key: str, model: str, why: str) -> None:
        with _RZ_LOCK:
            changed = _RZ_MODE.get(model) != "budgeted"
            _RZ_MODE[model] = "budgeted"
        if changed:
            self.ep.trace.decision("llm_reasoning_mode", llm_role=key, model=model, mode="budgeted", why=why,
                                   headroom=_RZ_HEADROOM)

    def _rz_reject(self, key: str, model: str, sent: Optional[tuple], used: bool) -> bool:
        """A reasoning 400. A learning model sent with reasoning off: switch it, retry at once (True, once). Rejected
        when budgeted, or a none-only model (never switched to minimal): demote it for the episode (False)."""
        mode = (sent or ("none", 0))[0]
        if mode != "budgeted" and _rz_policy(model) == "learn":
            self._rz_switch(key, model, "http 400")
            return not used
        self.rz_exhausted[model] = max(self.rz_exhausted.get(model, 0), _RZ_DEMOTE_AT)
        self.ep.trace.decision("llm_reasoning_mode", llm_role=key, model=model, mode=mode, why="http 400",
                               action="next")
        return False

    def _rz_action(self, key: str, model: str, data: Any, kind: str, sent: Optional[tuple], used: bool) -> str:
        """A length stop with an unusable answer: "budgeted" (a learning model reasoned with reasoning off: switch,
        retry at once), "longer" (plain truncation: retry with max_tokens x1.5) or "next"; one such retry per model
        and job. A none-only model that reasoned goes "next" (effort minimal would only reason longer). Exhaustions
        count toward demotion when budgeted or none-only: a learning model's exhaustion with reasoning off only
        reveals its mode, and parallel votes reveal it together."""
        mode, mt = sent or ("none", 0)
        _, ct, rt = _rz_usage(data)
        policy = _rz_policy(model)
        reasoned = kind == "exhausted" or bool(rt and ct is not None and rt * 4 >= ct)
        if mode != "budgeted" and reasoned and policy == "learn":
            act = "budgeted"
            self._rz_switch(key, model, kind)
        elif mode != "budgeted" and reasoned:
            act = "next"
        else:
            act = "longer" if kind == "truncated" else "next"
        act = "next" if used else act
        fields = dict(llm_role=key, model=model, mode=mode, max_tokens=mt, completion=ct, reasoning=rt, action=act)
        if kind == "exhausted":
            counts = mode == "budgeted" or policy == "none_only"
            n = self.rz_exhausted[model] = self.rz_exhausted.get(model, 0) + counts
            self.ep.trace.decision("llm_exhausted", n=n, demoted=n >= _RZ_DEMOTE_AT, **fields)
        else:
            self.ep.trace.decision("llm_truncated", **fields)
        return act

    def _log_attempt(self, run: _JobRun, out: Optional[_HttpOutcome], ms: float, ok: bool, note: str) -> None:
        self._log(run.spec.key, run.model or "", out, ms, ok, run.model_i > 0 or run.repairing, note)

    def _log(self, key: str, model: str, out: Optional[_HttpOutcome], ms: float, ok: bool, fallback: bool,
             note: str) -> None:
        """One `llm` decision per attempt (DESIGN 9: role, model, ms, tokens, usd, ok, fallback)."""
        usage = (out.data.get("usage") or {}) if out is not None and isinstance(out.data, dict) else {}
        tokens = (int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)) \
            if isinstance(usage, dict) else (0, 0)
        cost = usage.get("cost") if isinstance(usage, dict) else None
        usd = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else 0.0
        self.calls_by_role[key] = self.calls_by_role.get(key, 0) + 1
        self.ep.trace.llm_call(key, model, ms, tokens, usd, ok, fallback, note)

    def _note_outcome(self, run: _JobRun) -> None:
        """Fast dead-backend detection: three whole-ladder failures across two roles, with no success in
        between, switch the episode to deterministic substitutes (deviation, see IMPLEMENTATION.md). A
        ladder that failed only transiently counts once the failure streak spans TRANSIENT_SPAN_S."""
        if run.value is not None:
            self.fail_streak = 0
            self.failed_roles = []
            return
        if run.model_i >= len(run.ladder) and run.attempts >= min(2, len(run.ladder)):
            if not run.hard_fail and not self._transient_span_ok():
                return
            self.fail_streak += 1
            self.failed_roles.append(run.spec.role)
            if self.fail_streak >= 3 and len(set(self.failed_roles)) >= 2:
                self.mark_dead("every model failed on the first roles", provisional=not run.hard_fail)

_EXPECTED_STEP_RE = re.compile(r"expected\s+(\d+)")
TRANSIENT_STATUSES = (429, 502, 503, 504)

class Transport:
    BACKOFF = (1.0, 2.0, 4.0)
    MAX_RESYNCS = 4

    def __init__(self, binding: dict, limit: Callable[[], int], clock: Callable[[], float],
                 calls_needed: Callable[[], int] = lambda: 1, client: Any = None,
                 sleep: Callable[[float], None] = time.sleep, log: Optional[Callable[..., None]] = None):
        self.client = client if client is not None else ProxyClient(timeout=20, max_retries=1)
        self.binding = dict(binding or {})
        self.limit = limit
        self.clock = clock
        self.calls_needed = calls_needed
        self.sleep = sleep
        self.log = log or (lambda *a, **k: None)
        self.n = 0
        self.finished = False
        self.state_uncertain = False
        self.sanitize = False          # set after an EnvelopeBug: args are JSON round-tripped
        self.retries = 0
        self.resyncs = 0

    def env_timeout(self, finish: bool = True) -> float:
        """clamp(min(20, (290 - t) / (calls_needed + 1)), 3, 20); a non-finish attempt is also capped at
        max(3, T_LAST_ENV_SEND - 5 - t) so it cannot outlive the non-finish send window."""
        t = self.clock()
        base = float(clamp(min(20.0, (T_ENV_TIMEOUT_HORIZON - t) / (max(1, self.calls_needed()) + 1)), 3.0, 20.0))
        if not finish:
            base = min(base, max(3.0, T_LAST_ENV_SEND - 5.0 - t))
        return base

    def _envelope(self, tool: str, args: dict) -> dict:
        if self.sanitize:
            args = json.loads(json.dumps(args, default=str))
        envelope = dict(self.binding)
        envelope[STEP_KEY] = self.n + 1
        envelope[ACTION_KEY] = {NAME_KEY: tool, ARGS_KEY: args}
        return envelope

    def send(self, tool: str, args: dict, deadline: float = T_FALLBACK, finish: bool = False) -> Raw:
        """finish: the call is an ORDER/DECLINE (or a finish completion); it may use the finish window and
        the last step. deadline: no transient retry starts at or after this episode time."""
        if self.finished:
            raise StopEpisode("finished")
        if self.n >= self.limit():
            raise StopEpisode("step_budget")
        if not finish:
            deadline = min(deadline, T_LAST_ENV_SEND)
        envelope = self._envelope(tool, args)
        attempt = 0
        resyncs = 0
        while True:
            try:
                self.client.timeout = self.env_timeout(finish)
            except Exception:
                pass
            try:
                res = self.client.post_verbose(ENV_PATH, envelope)
            except Exception as exc:  # the client itself failed: treat like a network error
                res = None
                err_status, err_text = None, repr(exc)[:300]
            else:
                data = getattr(res, "data", None)
                if isinstance(data, dict):
                    self.n = envelope[STEP_KEY]
                    raw = Raw(observation=data.get(OBS_KEY), finished=bool(data.get(FINISHED_KEY)),
                              step=data.get(STEP_KEY) if isinstance(data.get(STEP_KEY), int) else self.n,
                              replayed=bool(data.get(REPLAYED_KEY)), http_status=200)
                    if raw.finished:
                        self.finished = True
                    return raw
                err = getattr(res, "error", None) or {}
                err_status = err.get("status") if isinstance(err, dict) else None
                body = err.get("body") if isinstance(err, dict) else err
                err_text = body if isinstance(body, str) else dump_json(body)
            low = (err_text or "").lower()
            if err_status is None or err_status in TRANSIENT_STATUSES:
                if self.clock() >= deadline:
                    raise StopEpisode("environment_unavailable")
                delay = self.BACKOFF[min(attempt, len(self.BACKOFF) - 1)]
                delay = max(0.0, min(delay, deadline - self.clock()))
                self.log("env_retry", status=err_status, attempt=attempt + 1, delay=delay)
                self.retries += 1
                attempt += 1
                if delay > 0:
                    self.sleep(delay)
                if self.clock() >= deadline:                # no retry starts at the deadline (M8 time gates)
                    raise StopEpisode("environment_unavailable")
                continue                                    # identical envelope: replay-safe
            if err_status == 500 or "quarantin" in low:
                raise StopEpisode("quarantined")
            if err_status == 402:
                raise StopEpisode("key_exhausted")
            if err_status == 409:
                if "has finished" in low:
                    self.finished = True
                    raise StopEpisode("finished")
                resyncs += 1
                self.resyncs += 1
                if resyncs > self.MAX_RESYNCS:
                    raise StopEpisode("step_conflict")
                match = _EXPECTED_STEP_RE.search(low)
                if match:
                    expected = int(match.group(1))
                    if expected - 1 != self.n:
                        self.state_uncertain = True
                    self.log("env_resync", expected=expected, had=self.n + 1)
                    self.n = expected - 1
                elif "different action" in low:
                    self.log("env_step_reused", step=self.n + 1)
                    self.n += 1
                    self.state_uncertain = True
                else:
                    raise StopEpisode("conflict")
                if self.n >= self.limit():
                    raise StopEpisode("step_budget")
                if not finish and self.limit() - self.n <= 1:
                    self.log("env_step_shift", n=self.n, B=self.limit(), tool=tool)
                    raise StepShift(self.n, self.limit())
                envelope = self._envelope(tool, args)
                continue
            if err_status == 422:
                self.log("env_422", text=clip_text(err_text, 200))
                raise EnvelopeBug(clip_text(err_text, 300))
            raise EnvelopeBug(f"http {err_status}: {clip_text(err_text, 200)}")

ID_INJECT_KEY = "ls:id"     # synthetic record key for id-keyed record dicts

def unwrap(x: Any) -> tuple:
    """Peel two-key pending wrappers {result, [sentences]} in either key order (DESIGN 5.3)."""
    msgs: list = []
    for _ in range(4):
        if not (isinstance(x, dict) and len(x) == 2):
            break
        a, b = list(x.values())
        if is_sentences(a) and not is_sentences(b):
            msgs += a
            x = b
            continue
        if is_sentences(b) and not is_sentences(a):
            msgs += b
            x = a
            continue
        break
    return x, msgs

def get_path(x: Any, path: Optional[list]) -> Any:
    if path is None:
        return None
    cur = x
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur

def has_path(x: Any, path: Optional[list]) -> bool:
    if path is None:
        return False
    cur = x
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return False
    return True

def is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))

def is_session_error(x: Any) -> bool:
    return (isinstance(x, dict) and 1 <= len(x) <= 2 and isinstance(x.get(RUNTIME_ERROR_KEY), str)
            and all(isinstance(v, str) for v in x.values()))

def one_key_string(x: Any) -> Optional[str]:
    if isinstance(x, dict) and len(x) == 1:
        v = next(iter(x.values()))
        if isinstance(v, str):
            return v
    return None

def amount_text(v: str) -> bool:
    """v7: "12.30", "1,250", "-5", "$3.61"; not bare digits (a numeric code)."""
    signed = v[0] in "+-" or unicodedata.category(v[0]) == "Sc"
    m = re.fullmatch(r"\d[\d,]*(\.\d+)?", v[1:] if signed else v)
    return m is not None and (signed or m.group(1) is not None or "," in v)

def token_like(v: Any) -> bool:
    """An opaque token (a possible two-step checkout handle), not an all-capital fault code nor an amount (v7)."""
    return (isinstance(v, str) and 3 <= len(v) <= 64 and " " not in v and any(c.isdigit() for c in v)
            and not re.fullmatch(r"[A-Z_]+", v) and not amount_text(v))

def record_like(d: Any, min_keys: int = 3, digit_keys: Any = ()) -> bool:
    return isinstance(d, dict) and len(d) >= min_keys and (
        any(is_id_shaped(v) for v in d.values())
        or any(k in digit_keys and _digit_string(d[k]) for k in d))

def _digit_string(v: Any) -> bool:
    return isinstance(v, str) and 3 <= len(v) <= 40 and v.isdigit()

def digit_id_key(rows: list) -> Optional[str]:
    """List-level id test for digit-only string ids (DESIGN 5.3: string, unique, no spaces, <= 40 characters,
    shared alphabet): a key present on every row whose values are digit strings of one length, unique across
    the rows and never numbers elsewhere, when no row carries an id-shaped value."""
    dicts = [r for r in rows if isinstance(r, dict)]
    if len(dicts) < 2 or any(is_id_shaped(v) for r in dicts for v in r.values()):
        return None
    for key in dicts[0]:
        values = [r.get(key) for r in dicts]
        if all(_digit_string(v) for v in values) and len(set(values)) == len(values) \
                and len({len(v) for v in values}) == 1:
            return key
    return None

def _walk_lists(x: Any, path: tuple = (), depth: int = 0):
    if depth > 4 or not isinstance(x, dict):
        return
    for key, value in x.items():
        if isinstance(value, list):
            yield path + (key,), value, x
        elif isinstance(value, dict):
            yield from _walk_lists(value, path + (key,), depth + 1)

def _id_keyed(x: Any) -> bool:
    return (isinstance(x, dict) and len(x) >= 2 and sum(1 for k in x if is_id_shaped(k)) >= max(2, len(x) - 1)
            and all(isinstance(v, dict) for k, v in x.items() if is_id_shaped(k)))

def find_record_list(x: Any, count_path: Optional[list] = None, digit_keys: Any = None) -> Optional[tuple]:
    """(path, records, container) for the best record list inside x, a zero-count list, or None.
    count_path: the tool's learned count path (a zero there marks an empty result list); digit_keys: a set
    that collects keys found to hold digit-only string ids (list-level id test)."""
    known = digit_keys if digit_keys is not None else set()
    if isinstance(x, list):
        recs = [r for r in x if isinstance(r, dict)]
        key = digit_id_key(recs)
        if key is not None and digit_keys is not None:
            digit_keys.add(key)
        if recs and sum(1 for r in recs if record_like(r, 2, known)) >= max(1, len(recs) // 2):
            return [], recs, None
        return None
    best, best_score = None, None
    empties = []
    for path, value, container in _walk_lists(x):
        dicts = [r for r in value if isinstance(r, dict)]
        if dicts:
            key = digit_id_key(dicts)
            if key is not None and digit_keys is not None:
                digit_keys.add(key)
            n_like = sum(1 for r in dicts if record_like(r, 2, known))
            if n_like >= max(1, len(dicts) // 2):
                score = (n_like, -len(path))
                if best_score is None or score > best_score:
                    best, best_score = (list(path), dicts, container), score
        elif not value:
            empties.append((list(path), container))
    if best is not None:
        return best
    if _id_keyed(x):  # an id-keyed dict of records: the key becomes the record's id field
        return [], [dict(v, **{ID_INJECT_KEY: k}) for k, v in x.items() if is_id_shaped(k)], x
    zero_at_count = count_path is not None and get_path(x, count_path) == 0 \
        and not isinstance(get_path(x, count_path), bool)
    for path, container in empties:
        direct = [v for v in container.values() if isinstance(v, int) and not isinstance(v, bool)]
        nested = [v for k, v in _scalar_siblings(container, skip_key=path[-1], depth=1).items()
                  if len(k) == 2 and isinstance(v, int) and not isinstance(v, bool) and v == 0]
        if direct or nested or zero_at_count:
            return path, [], container
    return None

def holds_empty_list(x: Any, depth: int = 0) -> bool:
    """A dict that contains an empty list (one level of nesting at most): a zero-hit result shape."""
    if not isinstance(x, dict) or depth > 2:
        return False
    return any((isinstance(v, list) and not v) or holds_empty_list(v, depth + 1) for v in x.values())

def _scalar_siblings(container: Optional[dict], skip_key: Any = None, depth: int = 1) -> dict:
    """Scalar values beside the record list, one level into sibling dicts: {path tuple: value}."""
    out: dict = {}
    if not isinstance(container, dict):
        return out
    for key, value in container.items():
        if key == skip_key:
            continue
        if is_scalar(value) and not isinstance(value, bool):
            out[(key,)] = value
        elif isinstance(value, dict) and depth > 0:
            for k2, v2 in value.items():
                if is_scalar(v2) and not isinstance(v2, bool):
                    out[(key, k2)] = v2
    return out

@dataclass
class DecodeContext:
    tool: str = ""
    intent_kind: str = ""
    role: str = ROLE_OTHER
    is_finish: bool = False
    is_message: bool = False
    is_cart_write: bool = False
    is_cart_view: bool = False
    expected_empty: bool = True
    written_id: Optional[str] = None
    written_qty: Optional[int] = None
    requested_ids: list = field(default_factory=list)
    known_prices: dict = field(default_factory=dict)
    expected_qty: dict = field(default_factory=dict)   # the expected cart (id -> qty): breaks cart-key ties

class Decoder:
    def __init__(self, fields: "FieldModel"):
        self.fields = fields

    def decode(self, ctx: DecodeContext, raw: Raw) -> Decoded:
        obs = raw.observation
        dec = Decoded(kind=D_UNKNOWN, finished=bool(raw.finished), tool=ctx.tool, replayed=bool(raw.replayed))
        refusal_present = False
        if isinstance(obs, dict) and REFUSAL_KEY in obs and (len(obs) == 1 or (len(obs) == 2 and OBS_KEY in obs)):
            refusal_present = True
            value = obs.get(REFUSAL_KEY)
            if isinstance(value, list):
                dec.refusal = [s for s in value if isinstance(s, str)]
            elif isinstance(value, str):
                dec.refusal = [value]
            obs = obs.get(OBS_KEY)
        x, msgs = unwrap(obs)
        dec.messages = msgs
        dec.inner = x
        if ctx.is_finish:
            kind = self._finish_kind(ctx, dec, x, refusal_present)
            if kind is not None:
                dec.kind = kind
                return dec
        self._classify(ctx, dec, x)
        return dec

    def _finish_kind(self, ctx: DecodeContext, dec: Decoded, x: Any, refusal_present: bool) -> Optional[str]:
        if dec.finished:
            return D_FINISHED
        if refusal_present:
            return D_REFUSED
        if is_session_error(x):
            dec.session_code = x.get(RUNTIME_ERROR_KEY)
            dec.fault_text = dump_json(x)
            return D_SESSION_ERROR
        fault = one_key_string(x)
        empty = x is None or x == {}
        if (fault is not None or empty) and ctx.expected_empty:
            dec.fault_text = fault or "{}"
            return D_EMPTY_CART
        if dec.messages:
            return D_REFUSED
        if isinstance(x, dict) and 1 <= len(x) <= 2 and any(token_like(v) for v in x.values()):
            return D_NONFINAL
        if fault is not None:
            dec.fault_text = fault
            return D_WORLD_FAULT
        if empty:
            dec.fault_text = "{}"
            return D_EMPTY
        return None

    def _classify(self, ctx: DecodeContext, dec: Decoded, x: Any) -> None:
        if x is None or x == {} or x == []:
            dec.kind = D_EMPTY
            dec.fault_text = "{}"
            return
        if is_session_error(x):
            dec.kind = D_SESSION_ERROR
            dec.session_code = x.get(RUNTIME_ERROR_KEY)
            dec.fault_text = dump_json(x)
            return
        if ctx.is_message:
            if isinstance(x, str):
                dec.kind, dec.reply_text = D_REPLY, x
                return
            text = one_key_string(x) or reply_with_counters(x)
            if text is not None:
                dec.kind, dec.reply_text = D_REPLY, text
                return
        if isinstance(x, dict) and len(x) == 1:
            value = next(iter(x.values()))
            if isinstance(value, bool):
                dec.kind = D_ACK
                return
            if isinstance(value, str) and not is_id_shaped(value):
                dec.kind, dec.fault_text = D_WORLD_FAULT, value
                return
        if ctx.is_cart_write or ctx.is_cart_view or ctx.is_finish:
            cart = find_cart(x, ctx, self.fields)   # carts only where a cart can answer (F03)
            if cart is not None:
                dec.kind, dec.cart = D_CART, cart
                return
        if ctx.is_cart_write and isinstance(x, dict) and len(x) <= 4 and ctx.written_id is not None \
                and ctx.written_id in [v for v in x.values() if isinstance(v, str)]:
            dec.kind = D_ACK
            qty = [v for v in x.values() if isinstance(v, int) and not isinstance(v, bool)]
            dec.cart = {"lines": [(ctx.written_id, qty[0] if len(qty) == 1 else ctx.written_qty, None)],
                        "total_c": None, "echo": True}
            return
        if ctx.is_cart_write and isinstance(x, dict) and 1 < len(x) <= 4 and all(is_scalar(v) for v in x.values()):
            flags = [v for v in x.values() if isinstance(v, bool)]
            if flags == [True]:
                dec.kind = D_ACK                       # a success flag plus counters or notes (LS-05)
                return
            if flags == [False]:
                texts = [v for v in x.values() if isinstance(v, str) and v.strip()]
                dec.kind, dec.fault_text = D_WORLD_FAULT, "; ".join(texts) or dump_json(x)
                return
        found = find_record_list(x, self.fields.fmap(ctx.tool).count_path if ctx.tool else None,
                                 self.fields.digit_id_keys)
        if found is not None:
            path, recs, container = found
            dec.kind = D_RECORDS
            dec.records = recs
            dec.records_path = path
            dec.container = container
            if isinstance(container, dict):
                for key, value in container.items():
                    if path and key == path[-1]:
                        continue
                    if isinstance(value, list) and value and all(is_id_shaped(v) for v in value):
                        dec.absent_ids = list(value)
            return
        if record_like(x, 3, self.fields.digit_id_keys):
            dec.kind = D_RECORD
            dec.records = [x]
            dec.records_path = []
            return
        if isinstance(x, dict) and _small_int_dict(x):
            dec.kind = D_STATUS
            return
        if isinstance(x, str):
            dec.kind, dec.fault_text = D_WORLD_FAULT, x
            return
        dec.kind = D_UNKNOWN

def reply_with_counters(x: Any) -> Optional[str]:
    """A message reply beside counters: exactly one string value of >= 3 words, every other value a number
    or a boolean (LS-17). The counters are read as a status reading by the executor."""
    if not isinstance(x, dict) or len(x) < 2:
        return None
    texts = [v for v in x.values() if isinstance(v, str)]
    others = [v for v in x.values() if not isinstance(v, str)]
    if len(texts) == 1 and len(texts[0].split()) >= 3 and all(isinstance(v, (int, float, bool)) for v in others):
        return texts[0]
    return None

def _small_int_dict(x: dict) -> bool:
    if len(x) > 4:
        return False
    ints = 0
    for value in x.values():
        if isinstance(value, dict):
            if len(value) > 4 or not all(isinstance(v, int) and not isinstance(v, bool) for v in value.values()):
                return False
            ints += len(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            ints += 1
        else:
            return False
    return ints >= 1

def _price_to_cents(value: Any, major: bool) -> Optional[int]:
    number = as_number(value)
    if number is None:
        return None
    return int(round(number * 100)) if major else int(round(number))

def complete_cart_map(cmap: Optional[dict]) -> bool:
    """A learned cart map that can read lines: it names the lines, the line id and the line quantity."""
    return bool(cmap) and bool(cmap.get(FR_LINES)) and bool(cmap.get(FR_LINE_ID)) and bool(cmap.get(FR_LINE_QTY))

def find_cart(x: Any, ctx: DecodeContext, fields: Optional["FieldModel"] = None) -> Optional[dict]:
    """A cart: line-like dicts plus a total equal to sum(price*qty) (or sum(qty) without prices).
    An empty line list beside a zero is an empty cart, but only for cart intents. A learned map is used
    only when it can read lines (F01): the tool's own map first, then the last map from any cart call."""
    if not isinstance(x, dict):
        return None
    digit_keys = fields.digit_id_keys if fields is not None else set()
    maps = []
    if fields is not None:
        maps = [m for m in (fields.fmap(ctx.tool).cart if ctx.tool else None, fields.cart_map)
                if complete_cart_map(m)]
    for learned in maps:
        lines = get_path(x, learned.get(FR_LINES))
        if isinstance(lines, list):
            out = _cart_from_map(x, lines, learned, digit_keys)
            if out is not None:
                return out
    for path, value, container in _walk_lists(x):
        siblings = _scalar_siblings(container, skip_key=path[-1], depth=1)
        numbers = {p: v for p, v in siblings.items() if looks_numeric(v) and not isinstance(v, str)}
        if not value:
            if (ctx.is_cart_write or ctx.is_cart_view) and any(as_number(v) == 0 for v in numbers.values()):
                total_path = next(p for p, v in numbers.items() if as_number(v) == 0)
                return {"lines": [], "total_c": 0, "map": {FR_LINES: list(path), FR_TOTAL: list(total_path)}}
            continue
        if not all(isinstance(it, dict) for it in value):
            continue
        id_keys = [k for k in value[0] if all(isinstance(it, dict) and (is_id_shaped(it.get(k)) or (
            k in digit_keys and _digit_string(it.get(k)))) for it in value)]
        if not id_keys:
            continue
        id_key = id_keys[0]
        num_keys = [k for k in value[0] if k != id_key and all(
            isinstance(it.get(k), (int, float)) and not isinstance(it.get(k), bool) for it in value)]
        int_keys = [k for k in num_keys if all(float(it[k]).is_integer() and it[k] >= 0 for it in value)]
        match = None
        matches: list = []
        for total_path, total in numbers.items():
            total_v = as_number(total)
            for qk in int_keys:
                for pk in num_keys:
                    if pk == qk:
                        continue
                    if abs(sum(float(it[pk]) * it[qk] for it in value) - total_v) < 0.005 * max(1.0, abs(total_v)):
                        cand = (qk, pk, total_path)
                        matches.append(cand)
                        match = _prefer_cart_assignment(match, cand, value, ctx)
            if match is None:
                for qk in int_keys:
                    if sum(it[qk] for it in value) == total_v:
                        match = match or (qk, None, total_path)
        ambiguous = False
        if match is not None and match[1] is not None:
            best = _cart_assignment_score(match, value, ctx)
            ambiguous = any(m[0] != match[0] and _cart_assignment_score(m, value, ctx) >= best - 1 for m in matches)
        if match is None and (ctx.is_cart_write or ctx.is_cart_view) and int_keys:
            qk = min(int_keys, key=lambda k: sum(it[k] for it in value))
            others = [k for k in num_keys if k != qk]
            match = (qk, others[0] if others else None, None)
        if match is None:
            continue
        qk, pk, total_path = match
        cmap = {FR_LINES: list(path), FR_LINE_ID: [id_key], FR_LINE_QTY: [qk],
                FR_LINE_PRICE: [pk] if pk else None, FR_TOTAL: list(total_path) if total_path else None}
        out = _cart_from_map(x, value, cmap, digit_keys)
        if out is not None:
            out["ambiguous"] = ambiguous           # two readings fit: a read-back never acts on it
            return out
    return None

def _cart_assignment_score(assignment: tuple, lines: list, ctx: DecodeContext) -> int:
    qk, pk, _ = assignment
    s = 0
    for it in lines:
        ident = next((v for v in it.values() if is_id_shaped(v)), None)
        if ctx.written_id is not None and ident == ctx.written_id and ctx.written_qty is not None \
                and it.get(qk) == ctx.written_qty:
            s += 2
        if ident in ctx.expected_qty and it.get(qk) == ctx.expected_qty[ident]:
            s += 2
        known = ctx.known_prices.get(ident)
        if known is not None and pk is not None and int(round(float(it.get(pk) or 0))) == known:
            s += 2
        if pk is not None and float(it.get(pk) or 0) >= float(it.get(qk) or 0):
            s += 1
    return s

def _prefer_cart_assignment(current: Optional[tuple], cand: tuple, lines: list, ctx: DecodeContext) -> tuple:
    """Two numeric keys can both satisfy the total identity: prefer the quantity key that equals the written
    or expected quantity and the price key that equals a known catalog price."""
    if current is None:
        return cand
    return cand if _cart_assignment_score(cand, lines, ctx) > _cart_assignment_score(current, lines, ctx) else current

def _cart_from_map(x: dict, lines: list, cmap: dict, digit_keys: Any = ()) -> Optional[dict]:
    id_key = (cmap.get(FR_LINE_ID) or [None])[0]
    qty_key = (cmap.get(FR_LINE_QTY) or [None])[0]
    price_key = (cmap.get(FR_LINE_PRICE) or [None])[0] if cmap.get(FR_LINE_PRICE) else None
    if lines and (id_key is None or qty_key is None):
        return None                                   # a map without line keys cannot read lines (F01)
    out_lines = []
    for it in lines:
        if not isinstance(it, dict):
            return None
        if id_key is not None:
            ident, qty = it.get(id_key), it.get(qty_key)
            id_ok = is_id_shaped(ident) or (id_key in digit_keys and _digit_string(ident))
            if not id_ok or not isinstance(qty, (int, float)) or isinstance(qty, bool):
                return None
            price = it.get(price_key) if price_key else None
            out_lines.append((ident, int(qty), price))
    total = get_path(x, cmap.get(FR_TOTAL)) if cmap.get(FR_TOTAL) else None
    return {"lines": out_lines, "total_c": total, "map": cmap}

EVT_NAME_ROLES = (FR_SHELF, FR_MERCHANT, FR_MAKER)

class FieldModel:
    def __init__(self):
        self.maps: dict = {}             # tool -> FieldMap
        self.key_roles: dict = {}        # record key -> fr:* role, learned from multi-record results
        self.row_keys: set = set()       # keys seen in search/browse rows (not in record opens)
        self.cart_map: Optional[dict] = None   # the last map learned from a cart call (never from other calls)
        self.price_major: Optional[bool] = None
        self.rescale_pending = False     # price_major flipped False -> True: stored cents need x 100
        self.digit_id_keys: set = set()  # keys holding digit-only string ids (list-level id test)
        self.attr_names: set = set()     # canonical attribute names seen as dict keys (pair orientation hint)
        self.oracle_asked: set = set()
        self.version = 0                 # bumps when row-learned key roles change
        self.pinned: dict = {}           # tool -> {fr:*: path} accepted from the FIELDS oracle
        self.raw_rows: dict = {}         # id -> [(FieldMap, {key: number})]: numeric scalars seen (price re-learning)
        self.price_confirmed = False     # a cart showed the catalog prices
        self.name_stats: dict = {}       # key -> [row batches, sum of distinct ratios, ever null]

    def fmap(self, tool: str) -> FieldMap:
        if tool not in self.maps:
            self.maps[tool] = FieldMap()
        return self.maps[tool]

    def learn(self, tool: str, dec: Decoded, qkey: Optional[str] = None, page: Any = None,
              sent: Optional[dict] = None, opened: bool = False) -> FieldMap:
        """qkey/page/sent: the query identity, the paging value sent and the call's arguments (meta
        confirmation across pages); opened: the call opened records by id (batch views learn notes)."""
        fm = self.fmap(tool)
        if dec.kind == D_CART and dec.cart and dec.cart.get("map"):
            new = dec.cart["map"]
            has_lines = bool(dec.cart.get("lines"))
            if has_lines or not complete_cart_map(fm.cart):
                fm.cart = dict(new)
            if has_lines or not complete_cart_map(self.cart_map):
                self.cart_map = dict(new)
            return fm
        if dec.kind not in (D_RECORDS, D_RECORD):
            return fm
        recs = [r for r in dec.records if isinstance(r, dict)]
        keyset = frozenset(k for r in recs for k in r)
        fm.records_path = list(dec.records_path or [])
        if dec.kind == D_RECORDS:
            self._learn_meta(fm, dec, qkey, page, sent)
        if not recs:
            return fm
        multi = dec.kind == D_RECORDS and len(recs) >= 2
        as_rows = multi and not opened
        if as_rows:
            self._note_names(recs)
        stale = fm.learned_from > 0 and getattr(fm, "row_version", 0) < self.version and not multi
        core_missing = any(role not in fm.role_paths for role in (FR_ID, FR_PRICE, FR_TITLE)) and as_rows
        known_missing = as_rows and any(self.key_roles.get(k) not in (None, *fm.role_paths) for k in keyset)
        if fm.keyset and keyset <= fm.keyset and fm.role_paths and not stale and not core_missing \
                and not known_missing:
            fm.learned_from += 1
            if as_rows:
                self._name_fix(fm, tool)
            self._after_learn(fm, recs)
            return fm
        fm.keyset = fm.keyset | keyset
        ambiguous: list = []
        if as_rows:
            roles = self._roles_from_records(recs, True, self.digit_id_keys, ambiguous)
            proved: dict = {}                        # M6: the earliest tool that proved a role for a key wins, so
            for key, role in self.key_roles.items():  # every tool reads the same title and overview (codes)
                if key in keyset and role not in proved:
                    proved[role] = key
            for role, key in proved.items():
                for other in [r for r, k in roles.items() if k == key and r != role]:
                    del roles[other]
                roles[role] = key
            changed = False
            for role, key in roles.items():
                if self.key_roles.get(key) != role:
                    self.key_roles[key] = role
                    changed = True
            self.row_keys |= keyset
            if changed:
                self.version += 1
        else:
            roles = {}
            pins = {(r, p[0]) for paths in self.pinned.values() for r, p in paths.items() if len(p) == 1}
            for key in dict.fromkeys(k for r in recs for k in r):
                role = self.key_roles.get(key)
                if role and (role not in roles or ((role, key) in pins and (role, roles[role]) not in pins)):
                    roles[role] = key
            for role, key in self._roles_from_records(recs, False, self.digit_id_keys, ambiguous).items():
                if role not in roles and key not in roles.values() and key not in self.row_keys \
                        and not (role == FR_OVERVIEW and self.row_keys):   # M4: a row-absent text is the notes
                    roles[role] = key
                elif role == FR_OVERVIEW and role not in roles and key in self.row_keys \
                        and key not in self.key_roles:
                    roles[role] = key                        # M6: a full view proves the overview that rows
                    self.key_roles[key] = role               # truncate below the long-text threshold
                    self.version += 1
            self._notes_rule(roles, recs, keyset)
        fm.role_paths = {role: [key] for role, key in roles.items()}
        for role, path in (self.pinned.get(tool) or {}).items():
            fm.role_paths = {r: p for r, p in fm.role_paths.items() if p != path}
            fm.role_paths[role] = list(path)
        if as_rows:
            self._name_fix(fm, tool)
        fm.attrs_mode = self._attrs_mode(recs, (fm.role_paths.get(FR_ATTRS) or [None])[0])
        fm.ambiguous = sorted(set(ambiguous) - set(self.pinned.get(tool) or {}))
        fm.learned_from += 1
        fm.row_version = self.version
        self._after_learn(fm, recs)
        return fm

    def _notes_rule(self, roles: dict, recs: list, keyset: frozenset) -> None:
        """notes: a key present in opened records but not in search rows, sentence-like or null."""
        if not self.row_keys or FR_NOTES in roles:
            return
        for key in sorted(keyset - self.row_keys, key=str):
            values = [r.get(key) for r in recs]
            if key not in roles.values() and all(
                    v is None or (isinstance(v, str) and len(v.split()) >= 3) for v in values):
                roles[FR_NOTES] = key
                return

    def _after_learn(self, fm: FieldMap, recs: list) -> None:
        """Every observation: re-evaluate the price unit (F05) and the pair orientation (F08); keep the numeric
        scalars of each record for the price check at the first cart."""
        id_path = fm.role_paths.get(FR_ID)
        for r in recs:
            ident = get_path(r, id_path) if id_path else r.get(ID_INJECT_KEY)
            if isinstance(ident, str) and (ident in self.raw_rows or len(self.raw_rows) < 600):
                nums = {k: v for k, v in r.items() if looks_numeric(v) and not isinstance(v, (bool, str))}
                self.raw_rows[ident] = (self.raw_rows.get(ident, []) + [(fm, nums)])[-2:]
        price_path = fm.role_paths.get(FR_PRICE)
        if price_path:
            values = [get_path(r, price_path) for r in recs]
            major = any(isinstance(v, float) or (isinstance(v, str) and "." in v and looks_numeric(v))
                        for v in values if v is not None and not isinstance(v, bool))
            if self.price_major is None:
                self.price_major = major
            elif major and not self.price_major:
                self.price_major = True
                self.rescale_pending = True
        attrs_path = fm.role_paths.get(FR_ATTRS)
        if attrs_path and fm.attrs_mode == "attrs:pairs":
            for r in recs:
                value = get_path(r, attrs_path)
                if isinstance(value, list) and value:
                    fm.pair_samples.append(value)
            fm.pair_samples = fm.pair_samples[-12:]
            fm.pair_keys = orient_pairs(fm.pair_samples, self.attr_names) or fm.pair_keys

    def consult(self, ep: Any, tool: str, dec: Decoded) -> bool:
        """One FIELDS call per tool when record roles are ambiguous; a path is accepted only when it resolves on
        at least half of the sample and its values pass the role's shape test. Without an oracle the
        structural choice stands."""
        fm = self.fmap(tool)
        roles = [r for r in fm.ambiguous if r in RECORD_FIELD_ROLES]
        recs = [r for r in dec.records if isinstance(r, dict)][:3]
        if not roles or not recs or tool in self.oracle_asked or not ep.llm.allowed("FIELDS"):
            return False
        self.oracle_asked.add(tool)
        sample = [{k: (clip_text(v, 80) if isinstance(v, str) else v if is_scalar(v) or len(dump_json(v)) <= 400
                       else clip_text(dump_json(v), 200)) for k, v in r.items()} for r in recs]
        payload = {"roles": roles, "samples": [{"tname": tool, "obs": sample}]}
        ans = ep.llm.ask("FIELDS", payload, validator=lambda v: self.fields_validator(v, roles, recs))
        fixed = dict((ans or {}).get("record") or {})
        for role, path in list(fixed.items()):
            if fm.role_paths.get(role) == path:
                del fixed[role]
                continue
            fm.role_paths = {r: p for r, p in fm.role_paths.items() if p != path}
            fm.role_paths[role] = list(path)
            if len(path) == 1:
                self.key_roles[path[0]] = role
        if fixed:
            self.pinned.setdefault(tool, {}).update(fixed)
            fm.attrs_mode = self._attrs_mode(recs, (fm.role_paths.get(FR_ATTRS) or [None])[0])
        fm.ambiguous = [r for r in fm.ambiguous if r not in roles]
        ep.trace.decision("fieldmap", tool=tool, why="FIELDS", asked=roles, fixed=fixed, answered=ans is not None)
        return bool(fixed)

    def fields_validator(self, value: dict, roles: list, recs: list) -> tuple:
        good: dict = {}
        for raw_role, path in (value.get("record") or {}).items():
            role = canon_label(raw_role, RECORD_FIELD_ROLES)
            if role not in roles or not path or not all(isinstance(k, str) for k in path):
                continue
            hits = [get_path(r, path) for r in recs if has_path(r, path)]
            if 2 * len(hits) >= len(recs) and _evt_shape_ok(role, hits, len(hits)):
                good[role] = list(path)
        return True, {"record": good}, "" if good else "no path passed the shape test"

    def check_cart_prices(self, ep: Any, dec: Decoded) -> None:
        """DESIGN 5.3: the price path is confirmed when a cart shows the catalog prices (or a total equal to
        sum(price * qty)); when every shown price differs but one other numeric key of the stored rows matches
        them all, that key becomes the price path and the stored prices are re-read (decision fieldmap)."""
        if self.price_confirmed or not dec.cart:
            return
        major = bool(self.price_major)
        lines = [(i, q, p) for i, q, p in dec.cart.get("lines") or [] if q]
        known = [(i, q, _price_to_cents(p, major) if p is not None else None, ep.catalog[i].price_c)
                 for i, q, p in lines if i in ep.catalog and ep.catalog[i].price_c]
        if not known:
            return
        if all(cp is not None for _, _, cp, _ in known):
            if all(cp == ours for _, _, cp, ours in known):
                self.price_confirmed = True
            elif all(cp != ours for _, _, cp, ours in known):
                self._relearn_price(ep, {i: cp for i, _, cp, _ in known})
        else:
            total = _price_to_cents(dec.cart.get("total_c"), major)
            if total is not None and len(known) == len(lines) and total == sum(q * ours for _, q, _, ours in known):
                self.price_confirmed = True
        if self.price_confirmed:
            ep.trace.decision("fieldmap", why="price path confirmed by a cart", price_confirmed=True)

    def _relearn_price(self, ep: Any, target: dict) -> None:
        keys: Optional[set] = None
        for ident, cents in target.items():
            found = {k for fm, nums in self.raw_rows.get(ident, []) for k, v in nums.items()
                     if _price_to_cents(v, bool(self.price_major)) == cents}
            keys = found if keys is None else keys & found
        if not keys or len(keys) != 1:
            return                                     # a price change, not a misread field
        key = keys.pop()
        for fm in {id(f): f for rows in self.raw_rows.values() for f, nums in rows if key in nums}.values():
            fm.role_paths = {r: p for r, p in fm.role_paths.items() if p != [key]}
            fm.role_paths[FR_PRICE] = [key]
        self.key_roles[key] = FR_PRICE
        u = ep.transport.n
        for ident, rows in self.raw_rows.items():
            r = ep.catalog.get(ident)
            value = next((nums[key] for _, nums in reversed(rows) if key in nums), None)
            cents = _price_to_cents(value, bool(self.price_major)) if value is not None else None
            if r is not None and cents and r.price_c != cents:
                r.price_c = cents
                r.price_hist.append((u, cents))
        self.price_confirmed = True
        ep.trace.decision("fieldmap", why="price path re-learned at the first cart", price_key=key)

    def _note_names(self, recs: list) -> None:
        """Distinct ratio per short string key over each row batch (>= 3 rows)."""
        if len(recs) < 3:
            return
        for key in dict.fromkeys(k for r in recs for k in r):
            values = [r.get(key) for r in recs]
            texts = [v for v in values if isinstance(v, str) and v.strip()]
            if not texts or len(texts) + sum(1 for v in values if v is None or v == "") != len(values) \
                    or sum(len(t) for t in texts) / len(texts) > 80:
                continue
            st = self.name_stats.setdefault(key, [0, 0.0, False])
            st[0] += 1
            st[1] += len(set(texts)) / len(texts)
            st[2] = st[2] or len(texts) < len(values)

    def _name_fix(self, fm: FieldMap, tool: str) -> None:
        """Shelf, merchant and maker from accumulated batch ratios (two batches at least): a mixed first batch
        must not leave the shelf key labelled as the merchant. Roles the FIELDS oracle pinned are kept."""
        if any(role in (self.pinned.get(tool) or {}) for role in EVT_NAME_ROLES):
            return
        cur = {role: fm.role_paths[role][0] for role in EVT_NAME_ROLES if len(fm.role_paths.get(role) or []) == 1}
        taken = {p[0] for r, p in fm.role_paths.items() if r not in EVT_NAME_ROLES and len(p) == 1}
        stats = {k: v for k, v in self.name_stats.items() if v[0] >= 2 and k in fm.keyset and k not in taken
                 and (k in cur.values() or self.key_roles.get(k) in (None,) + EVT_NAME_ROLES)}
        if len(stats) < 2:
            return

        def ratio(k: str) -> float:
            return stats[k][1] / stats[k][0]
        never = sorted((k for k in stats if not stats[k][2]), key=lambda k: (ratio(k), str(k)))
        new: dict = {}
        if never and ratio(never[0]) <= 0.5:
            new[FR_SHELF] = never[0]
        rest = [k for k in never if k not in new.values()]
        if rest:
            new[FR_MERCHANT] = max(rest, key=lambda k: (ratio(k), str(k)))
        left = [k for k in stats if k not in new.values()]
        if left:
            new[FR_MAKER] = max(left, key=lambda k: (ratio(k), str(k)))
        if new == cur or not set(cur.values()) <= set(new.values()):
            return                                     # every key keeps a name role; nothing is dropped
        for role in EVT_NAME_ROLES:
            fm.role_paths.pop(role, None)
        for role, key in new.items():
            fm.role_paths[role] = [key]
            self.key_roles[key] = role
        self.version += 1

    def _learn_meta(self, fm: FieldMap, dec: Decoded, qkey: Optional[str], page: Any, sent: Optional[dict]) -> None:
        """count: an integer >= rows beside the list, constant across pages of one query; continuation: the
        sibling that changes across pages or turns null on the last page (DESIGN 5.3). First-page ranking:
        the next page number or offset, then token-like strings, then null, then other integers; strings
        with spaces or equal to the sent query never qualify. Confirmed paths are kept until contradicted."""
        container = dec.container
        if not isinstance(container, dict):
            return
        skip = dec.records_path[-1] if dec.records_path else None
        snap = _scalar_siblings(container, skip_key=skip, depth=1)
        prefix = list(dec.records_path[:-1]) if dec.records_path else []
        rows = len(dec.records)
        sent_text = {norm_space(v) for v in (sent or {}).values() if isinstance(v, str)}
        prev = None
        if qkey is not None:
            pages = fm.meta_pages.setdefault(qkey, {})
            here = dump_json(page)
            prev = next((v for k, v in pages.items() if k != here), None)
            if len(pages) < 4:
                pages[here] = dict(snap)
            if len(fm.meta_pages) > 24:
                fm.meta_pages.pop(next(iter(fm.meta_pages)))

        def rel(path: Optional[list]) -> Optional[tuple]:
            if not path or path[:len(prefix)] != prefix:
                return None
            return tuple(path[len(prefix):])

        ints = {p: v for p, v in snap.items() if isinstance(v, int) and not isinstance(v, bool)}
        counts = [p for p, v in ints.items() if v >= rows]
        if prev is not None:
            steady = [p for p in counts if prev.get(p) == ints[p]]
            counts = steady or counts
        count_p = None
        known = rel(fm.count_path)
        if fm.count_confirmed and known in counts:
            count_p = known
        elif counts:
            has_null = any(v is None for v in snap.values())
            exact = [p for p in counts if ints[p] == rows]
            bigger = [p for p in counts if ints[p] > rows]
            if has_null and exact and bigger and len(exact) == 1:
                count_p = exact[0]                     # a complete page: the count equals the rows seen
            else:
                count_p = max(counts, key=lambda p: ints[p])
            if prev is not None and prev.get(count_p) == ints[count_p]:
                fm.count_confirmed = True
        base = page if isinstance(page, int) and not isinstance(page, bool) else None
        nexts = {(base or 1) + 1, (base or 0) + rows}
        scores: dict = {}
        for p, v in snap.items():
            if p == count_p:
                continue
            if isinstance(v, str):
                text = v.strip()
                if not text or " " in text or norm_space(text) in sent_text:
                    continue
                score = 3.0 if any(c.isdigit() for c in text) else 1.0
            elif v is None:
                score = 2.5
            elif isinstance(v, int) and not isinstance(v, bool):
                score = 4.0 if v in nexts else 2.0
            else:
                continue
            if prev is not None and p in prev:
                if prev[p] == v and v is not None:
                    continue                           # equal across pages: static, never a continuation
                if prev[p] != v:
                    score += 2.0                       # changed, or turned null on the last page
            scores[p] = score
        known_cont = rel(fm.cont_path)
        if fm.cont_confirmed and known_cont in scores:
            cont_p = known_cont
        else:
            if fm.cont_confirmed and known_cont not in scores:
                fm.cont_confirmed = False              # contradicted: re-learn
            cont_p = max(scores, key=lambda p: (scores[p], -list(snap).index(p))) if scores else None
            if cont_p is not None and prev is not None and cont_p in prev:
                fm.cont_confirmed = True
        fm.count_path = prefix + list(count_p) if count_p else None
        fm.cont_path = prefix + list(cont_p) if cont_p else None

    def meta(self, tool: str, dec: Decoded) -> dict:
        fm = self.fmap(tool)
        n = None
        cont = None
        root = dec.inner
        if fm.count_path:
            v = get_path(root, fm.count_path)
            n = v if isinstance(v, int) and not isinstance(v, bool) else None
        if fm.cont_path:
            cont = get_path(root, fm.cont_path)
        return {"n": n, "cont": cont, "has_cont": cont is not None and cont != ""}

    @staticmethod
    def _roles_from_records(recs: list, multi: bool, digit_keys: Any = (), ambiguous: Optional[list] = None) -> dict:
        n = len(recs)
        keys: list = []
        for r in recs:
            for k in r:
                if k not in keys:
                    keys.append(k)
        roles: dict = {}
        stats = {}
        for k in keys:
            values = [r.get(k) for r in recs if k in r]
            values = [None if isinstance(v, str) and not v.strip() else v for v in values]   # blank = missing
            nonnull = [v for v in values if v is not None]
            stats[k] = (values, nonnull)
        ids = [k for k in keys if len(stats[k][1]) == n and (
            all(is_id_shaped(v) for v in stats[k][1]) or (k in digit_keys and all(_digit_string(v) for v in stats[k][1])))
            and len(set(stats[k][1])) == n]
        if ids:
            roles[FR_ID] = max(ids, key=lambda k: (_common_prefix_len(stats[k][1]), -keys.index(k)))
        prices = []
        for k in keys:
            values, nonnull = stats[k]
            if k == roles.get(FR_ID) or not nonnull:
                continue
            if all(looks_numeric(v) and not isinstance(v, bool) for v in nonnull):
                numbers = [as_number(v) for v in nonnull]
                if all(x > 0 for x in numbers):
                    varies = len(set(numbers)) > 1
                    prices.append((1.0 + (0.3 if varies or not multi else 0.0), statistics_median(numbers), k,
                                   _money_like(nonnull), all(float(x).is_integer() for x in numbers)))
        if prices:
            prices.sort(key=lambda t: (t[0], t[1]), reverse=True)
            top = prices[0]
            money = [t for t in prices if t[3] and t[0] >= top[0]]
            if top[4] and money and money[0] is not top:
                top = money[0]                         # a cents-precise decimal beats an integer counter (LS-10)
                if ambiguous is not None:
                    ambiguous.append(FR_PRICE)
            elif ambiguous is not None and len(prices) > 1 and prices[1][0] == top[0] \
                    and prices[1][1] >= 0.9 * top[1]:
                ambiguous.append(FR_PRICE)
            roles[FR_PRICE] = top[2]
        bools = [(sum(1 for v in stats[k][1] if v) / max(1, len(stats[k][1])), k) for k in keys
                 if stats[k][1] and all(isinstance(v, bool) for v in stats[k][1])]
        if bools:
            share, key = max(bools)
            if share >= 0.5 or len(bools) == 1:
                roles[FR_STOCK] = key
            if ambiguous is not None and len(bools) >= 2:
                ambiguous.append(FR_STOCK)
        for k in keys:
            nonnull = stats[k][1]
            if not nonnull:
                continue
            if all(isinstance(v, dict) for v in nonnull) and statistics_median(
                    [max(len(v), len(flatten_attrs(v))) for v in nonnull]) >= 3:
                roles.setdefault(FR_ATTRS, k)              # grouped tables count their leaves
            elif all(isinstance(v, list) and v and all(isinstance(x, dict) and len(x) == 2 for x in v)
                     for v in nonnull):
                roles.setdefault(FR_ATTRS, k)
        text_keys = []
        for k in keys:
            if k in roles.values():
                continue
            values, nonnull = stats[k]
            if not nonnull or not all(isinstance(v, str) for v in nonnull):
                continue
            mean_len = sum(len(v) for v in nonnull) / len(nonnull)
            spaced = sum(1 for v in nonnull if " " in v.strip()) / len(nonnull)
            unique = len(set(nonnull)) / len(nonnull)
            nulls = len(values) - len(nonnull)
            words = sum(len(v.split()) for v in nonnull) / len(nonnull)
            text_keys.append((k, mean_len, spaced, unique, len(set(nonnull)), nulls, words))
        def prose(key: Any) -> float:                # share of values with a sentence break (". " or a final ".")
            vals = [v for v in stats[key][1] if isinstance(v, str)]
            return sum(1 for v in vals if v.rstrip().endswith(".") or ". " in v) / max(1, len(vals))
        long_text = [t for t in text_keys
                     if t[1] > 150 or (multi and t[6] >= 4 and t[2] >= 0.8 and prose(t[0]) >= 0.5)]
        if long_text:                                 # M6: prose first (rows truncate overviews; titles can be long)
            roles[FR_OVERVIEW] = max(long_text, key=lambda t: (prose(t[0]) >= 0.5, t[1]))[0]
        titles = [t for t in text_keys if t[0] != roles.get(FR_OVERVIEW) and t[2] >= 0.8 and 10 <= t[1] <= 400
                  and (t[3] >= 0.8 or n < 3) and t[5] == 0]
        if titles:
            ranked = sorted(titles, key=lambda t: (-round(prose(t[0])), t[3], -abs(t[1] - 60)), reverse=True)
            roles[FR_TITLE] = ranked[0][0]
            if ambiguous is not None and len(ranked) >= 2 and round(prose(ranked[0][0])) == round(
                    prose(ranked[1][0])) and abs(ranked[0][3] - ranked[1][3]) < 0.1:
                ambiguous.append(FR_TITLE)
        else:
            spaced = [t for t in text_keys if t[2] >= 0.8 and t[5] == 0 and t[1] <= 400]
            plain = [t for t in spaced if t[0] != roles.get(FR_OVERVIEW)]
            if plain:
                roles[FR_TITLE] = max(plain, key=lambda t: (t[3], -abs(t[1] - 60)))[0]
            elif len(spaced) == 1:
                roles[FR_TITLE] = spaced[0][0]
        rest = [t for t in text_keys if t[0] not in roles.values() and t[1] <= 80]
        if multi:
            shelves = [t for t in rest if t[5] == 0 and t[4] <= max(1, n // 2)]
            if shelves:
                roles[FR_SHELF] = min(shelves, key=lambda t: (t[4], t[1]))[0]
            names = [t for t in rest if t[0] != roles.get(FR_SHELF) and t[6] <= 4]
            never_null = [t for t in names if t[5] == 0]
            if never_null:
                roles[FR_MERCHANT] = max(never_null, key=lambda t: t[4])[0]
                if ambiguous is not None and len(never_null) >= 2:
                    ambiguous.append(FR_MERCHANT)
            makers = [t for t in names if t[0] != roles.get(FR_MERCHANT)]
            if makers:
                roles[FR_MAKER] = max(makers, key=lambda t: t[4])[0]
        else:
            names = [t for t in rest if t[6] <= 4]
            if names:
                roles.setdefault(FR_SHELF, names[0][0])
        notes = [t for t in text_keys if t[0] not in roles.values() and t[6] >= 5]
        if notes and not multi:
            roles.setdefault(FR_NOTES, notes[0][0])
        return roles

    @staticmethod
    def _attrs_mode(recs: list, key: Optional[str]) -> Optional[str]:
        if key is None:
            return None
        sample = next((r.get(key) for r in recs if r.get(key) is not None), None)
        if isinstance(sample, list):
            return "attrs:pairs"
        if isinstance(sample, dict) and any(isinstance(v, dict) for v in sample.values()):
            return "attrs:nested"
        return "attrs:dict"

    def extract(self, tool: str, rec: dict) -> dict:
        """Field values of one record by role: {fr:*: value}; prices in integer cents."""
        fm = self.fmap(tool)
        out: dict = {}
        for role, path in fm.role_paths.items():
            if not has_path(rec, path):
                continue
            value = get_path(rec, path)
            if role == FR_PRICE:
                value = _price_to_cents(value, bool(self.price_major))
            elif role == FR_ATTRS:
                if isinstance(value, dict) and len(self.attr_names) < 400:
                    self.attr_names |= {canon_key(k) for k in value if isinstance(k, str)}
                value = flatten_attrs(value, pair_keys=fm.pair_keys)
            elif role == FR_STOCK:
                value = value if isinstance(value, bool) else None
            out[role] = value
        if FR_ID not in out:
            ident = next((v for v in rec.values() if is_id_shaped(v)), None)
            if ident is None:
                ident = next((v for k, v in rec.items() if k in self.digit_id_keys and _digit_string(v)), None)
            if ident is not None:
                out[FR_ID] = ident
        return out

def _evt_shape_ok(role: str, values: list, n: int) -> bool:
    """The shape test of a field role (DESIGN 5.3) on the values a proposed path resolves to."""
    vals = [v for v in values if v is not None]
    if role == FR_ID:
        return len(vals) == n and len({str(v) for v in vals}) == n and all(
            is_id_shaped(v) or _digit_string(v) for v in vals)
    if not vals:
        return role in (FR_NOTES, FR_MAKER)
    if role == FR_PRICE:
        return all(looks_numeric(v) and not isinstance(v, bool) and as_number(v) > 0 for v in vals)
    if role == FR_STOCK:
        return all(isinstance(v, bool) for v in vals)
    if role == FR_ATTRS:
        return all(isinstance(v, (dict, list)) and v for v in vals)
    if not all(isinstance(v, str) for v in vals):
        return False
    if role == FR_TITLE:
        return all(" " in v.strip() and 5 <= len(v) <= 400 for v in vals)
    return role not in EVT_NAME_ROLES or all(len(v) <= 80 for v in vals)

def _money_like(values: list) -> bool:
    """Every value is a number with at most two decimals and at least one has exactly two (cents)."""
    two = False
    for v in values:
        if isinstance(v, int) and not isinstance(v, bool):
            continue                                   # a whole amount: no decimals
        text = v if isinstance(v, str) else (repr(v) if isinstance(v, float) else None)
        if text is None:
            return False
        text = text.strip().replace(",", "")
        if "e" in text.lower():
            return False
        if "." in text:
            decimals = len(text.split(".", 1)[1])
            if decimals > 2:
                return False
            two = two or decimals == 2
    return two

def orient_pairs(samples: list, attr_names: Any = ()) -> Optional[tuple]:
    """(name key, value key) of name/value attribute pairs, decided by shape (DESIGN 5.3 'name-like and
    value-like'): names are unique within one list and recur across records; values carry digits and units
    more often; names match attribute keys seen elsewhere. None when the evidence is even."""
    pairs = [[x for x in lst if isinstance(x, dict) and len(x) == 2] for lst in samples if isinstance(lst, list)]
    pairs = [p for p in pairs if p]
    if not pairs:
        return None
    keysets = {}
    for lst in pairs:
        for item in lst:
            ks = tuple(sorted(item, key=str))
            keysets[ks] = keysets.get(ks, 0) + 1
    ka, kb = max(keysets, key=lambda ks: keysets[ks])

    def score(key: Any) -> float:
        lists = [[item.get(key) for item in lst if key in item] for lst in pairs]
        lists = [[str(v) for v in vals if is_scalar(v) and v is not None] for vals in lists]
        lists = [vals for vals in lists if vals]
        if not lists:
            return 0.0
        unique_in = sum(1 for vals in lists if len(set(vals)) == len(vals)) / len(lists)
        flat = [v for vals in lists for v in vals]
        digits = sum(1 for v in flat if any(c.isdigit() for c in v)) / len(flat)
        across = 0.0
        if len(lists) >= 2:
            distinct = set(flat)
            across = sum(1 for v in distinct if sum(1 for vals in lists if v in vals) >= 2) / max(1, len(distinct))
        known = sum(1 for v in flat if canon_key(v) in attr_names) / len(flat) if attr_names else 0.0
        return 2.0 * across + unique_in - digits + 2.0 * known

    sa, sb = score(ka), score(kb)
    if abs(sa - sb) < 0.25:
        return None
    return (ka, kb) if sa > sb else (kb, ka)

def statistics_median(values: list) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return float(s[mid]) if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0

def _common_prefix_len(values: list) -> int:
    if not values:
        return 0
    first = min(values)
    last = max(values)
    i = 0
    while i < min(len(first), len(last)) and first[i] == last[i]:
        i += 1
    return i

def flatten_attrs(value: Any, prefix: str = "", depth: int = 0, pair_keys: Optional[tuple] = None) -> list:
    """Attribute tables as [(raw_key, raw_value)]: dicts, name/value pair lists and nested groups. pair_keys:
    the (name key, value key) orientation learned by shape; without it, pairs are read in dict order."""
    out: list = []
    if depth > 3 or value is None:
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix} {k}".strip() if prefix and isinstance(v, dict) else str(k)
            if isinstance(v, dict):
                out += flatten_attrs(v, key, depth + 1, pair_keys)
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                out += flatten_attrs(v, key, depth + 1, pair_keys)
            elif isinstance(v, list):
                out.append((str(k), ", ".join(str(x) for x in v)))
            elif v is not None:
                out.append((str(k), str(v)))
        return out
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and len(item) == 2:
                if pair_keys and pair_keys[0] in item and pair_keys[1] in item:
                    a, b = item[pair_keys[0]], item[pair_keys[1]]
                else:
                    a, b = list(item.values())
                if isinstance(a, str) and is_scalar(b) and b is not None:
                    out.append((a, str(b)))
                elif isinstance(a, str) and isinstance(b, (list, dict)):
                    out += flatten_attrs(b, a, depth + 1, pair_keys)
            elif isinstance(item, dict):
                out += flatten_attrs(item, prefix, depth + 1, pair_keys)
    return out

DEFAULT_PAGE_SIZE = 8
DEFAULT_BATCH_LIMIT = 6

class ContractLearner:
    def __init__(self, contract: Contract):
        self.c = contract
        self.round_firsts: dict = {}

    def on_page(self, tool: str, qrec: "QueryRec", rows: list, n: Optional[int], cont: Any, page_sent: Any,
                paging_slot: Optional[str], paging_schema: Optional[dict], prices: list, ids: list) -> None:
        c = self.c
        has_cont = cont is not None and cont != ""
        if has_cont and rows:
            c.P = max(c.P or 0, len(rows))
        if has_cont and tool not in c.cont_kind:
            if isinstance(cont, str) and not looks_numeric(cont):
                c.cont_kind[tool] = "token"
            elif looks_numeric(cont):
                value = int(as_number(cont))
                first = page_sent is None
                sent = int(as_number(page_sent)) if looks_numeric(page_sent) else (1 if first else None)
                offset_sent = int(as_number(page_sent)) if looks_numeric(page_sent) else 0
                if sent is not None and value == sent + 1 and paging_slot != SLOT_OFFSET:
                    c.cont_kind[tool] = "page"
                elif value == offset_sent + len(rows):
                    c.cont_kind[tool] = "offset"
                else:
                    c.cont_kind[tool] = "token"
        if c.W is None and paging_schema:
            maximum = paging_schema.get("maximum")
            if isinstance(maximum, int) and c.P:
                if paging_slot == SLOT_PAGE:
                    c.W, c.W_source = maximum * c.P, "schema"
                elif paging_slot == SLOT_OFFSET:
                    c.W, c.W_source = maximum + c.P, "schema"
        seen = len(set(qrec.ids))
        if not has_cont and isinstance(n, int) and n > seen and seen > 0 and c.W is None:
            c.W, c.W_source = seen, "observed"
        self._sort_vote(rows, prices, ids)
        if isinstance(n, int) and n >= 100 and n % 50 == 0 and page_sent is None:
            firsts = self.round_firsts.setdefault(n, [])
            ident = list(ids)
            if not any((f == ident) if ident else (q == qrec.canon_args) for q, f in firsts):
                firsts.append((qrec.canon_args, ident))
            c.round_counts[n] = len(firsts)
            if len(firsts) >= 2:
                c.count_cap = n
        self.learn_id_sig(ids)

    def _sort_vote(self, rows: list, prices: list, ids: list) -> None:
        if len(ids) < 3:
            return
        ids_up = all(a <= b for a, b in zip(ids, ids[1:]))
        clean = [p for p in prices if p is not None]
        prices_up = len(clean) == len(prices) and all(a <= b for a, b in zip(clean, clean[1:]))
        if ids_up == prices_up:
            return
        key = SORT_ID if ids_up else SORT_PRICE
        self.c.sort_votes[key] = self.c.sort_votes.get(key, 0) + 1
        other = SORT_PRICE if key == SORT_ID else SORT_ID
        if self.c.sort_votes[key] >= 2 and not self.c.sort_votes.get(other):
            self.c.sort = key

    def window(self) -> int:
        if self.c.W:
            return self.c.W
        return 5 * (self.c.P or DEFAULT_PAGE_SIZE)

    def page_size(self) -> int:
        return self.c.P or DEFAULT_PAGE_SIZE

    def on_ceiling(self, ceiling_c: Optional[int], prices: list, left_out_at_ceiling: bool = False) -> None:
        """Rows priced exactly at a ceiling prove it inclusive; a complete result that leaves out a known
        on-shelf listing priced exactly at it proves it exclusive (unless already proven inclusive)."""
        if ceiling_c is None:
            return
        if any(p == ceiling_c for p in prices if p is not None):
            self.c.ceiling_inclusive = True
        elif left_out_at_ceiling and self.c.ceiling_inclusive is None:
            self.c.ceiling_inclusive = False

    def note_interplay(self, which: str, count_with: Optional[int], count_without: Optional[int]) -> None:
        """The one count comparison a ladder makes (DESIGN 5.7): which is "filter" (does a filter call honour
        keywords) or "text" (does a text search honour filters); equal counts mean the extra input is ignored."""
        if not isinstance(count_with, int) or not isinstance(count_without, int):
            return
        honoured = count_with != count_without
        if which == "filter":
            self.c.filter_honours_keywords = honoured
        else:
            self.c.text_honours_filters = honoured

    def batch_limit(self, schema_max_items: Optional[int]) -> int:
        if self.c.batch_limit is None:
            self.c.batch_limit = schema_max_items if isinstance(schema_max_items, int) and schema_max_items > 0 \
                else DEFAULT_BATCH_LIMIT
        return self.c.batch_limit

    def on_batch_rejected(self, sent: int) -> int:
        """{} or a fault on a batch view: halve the learned limit (never below 1)."""
        self.c.batch_limit = max(1, min(self.c.batch_limit or sent, sent) // 2)
        return self.c.batch_limit

    def learn_id_sig(self, ids: list) -> None:
        """Prefix (up to its last separator), fixed length when constant, and the character class of the
        tail (hex, digits, lower alphanumeric, alphanumeric or any)."""
        ids = [i for i in ids if isinstance(i, str)]
        if len(ids) < 2:
            return
        sig = self.c.id_sig
        known_prefix = sig.get("prefix") if sig else None
        pool = ids + ([known_prefix] if known_prefix is not None else [])
        common = text_common_prefix(pool) if known_prefix != "" else ""
        cut = max(common.rfind(ch) for ch in "-_:.")
        prefix = common[:cut + 1] if cut >= 0 else ""
        lengths = {len(i) for i in ids}
        if sig and sig.get("length") is not None:
            lengths.add(sig["length"])
        tails = "".join(i[len(prefix):] for i in ids)
        klass = id_char_class(tails)
        if sig and sig.get("alphabet"):
            klass = widest_class(klass, sig["alphabet"])
        self.c.id_sig = {"prefix": prefix, "length": lengths.pop() if len(lengths) == 1 else None, "alphabet": klass}

    def id_matches(self, ident: str) -> bool:
        sig = self.c.id_sig
        if not sig or not isinstance(ident, str):
            return True
        prefix = sig.get("prefix") or ""
        if prefix and not ident.startswith(prefix):
            return False
        if sig.get("length") is not None and len(ident) != sig["length"]:
            return False
        tail = ident[len(prefix):]
        return widest_class(id_char_class(tail), sig.get("alphabet") or "any") == (sig.get("alphabet") or "any")

ID_CLASSES = ("digits", "hex", "lower_alnum", "alnum", "any")     # each class contains the previous one

def id_char_class(text: str) -> str:
    chars = set(text)
    if all(c.isdigit() for c in chars):
        return "digits"
    if chars <= set("0123456789abcdef"):
        return "hex"
    if all(c.isalnum() and not c.isupper() for c in chars):
        return "lower_alnum"
    if all(c.isalnum() for c in chars):
        return "alnum"
    return "any"

def widest_class(a: str, b: str) -> str:
    rank = max(ID_CLASSES.index(a) if a in ID_CLASSES else len(ID_CLASSES) - 1,
               ID_CLASSES.index(b) if b in ID_CLASSES else len(ID_CLASSES) - 1)
    return ID_CLASSES[rank]

def text_common_prefix(values: list) -> str:
    """The longest prefix shared by every string (the lexicographic min and max bound it)."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    n = 0
    while n < len(lo) and n < len(hi) and lo[n] == hi[n]:
        n += 1
    return lo[:n]

class QueryBook:
    """One QueryRec per (tool, canonical non-paging arguments)."""

    def __init__(self):
        self.recs: dict = {}

    @staticmethod
    def key_of(tool: str, args: dict, paging_args: tuple) -> str:
        core = {k: v for k, v in args.items() if k not in paging_args}
        return tool + "|" + dump_json(core)

    def get(self, key: str) -> Optional[QueryRec]:
        return self.recs.get(key)

    def record_page(self, key: str, tool: str, args: dict, query_text: str, line: Optional[int], kind: str,
                    page_sent: Any, ids: list, n: Optional[int], cont: Any, count_cap: Optional[int]) -> QueryRec:
        rec = self.recs.get(key)
        if rec is None:
            rec = QueryRec(tool=tool, canon_args=key, tokens=content_tokens(query_text), line=line, kind=kind,
                           args=dict(args), query_text=query_text)
            self.recs[key] = rec
        rec.pages_read.add(dump_json(page_sent))
        for ident in ids:
            if ident not in rec.ids:
                rec.ids.append(ident)
        if isinstance(n, int):
            rec.count = n
            rec.capped = count_cap is not None and n >= count_cap
        rec.next_cont = cont
        rec.has_cont = cont is not None and cont != ""
        rec.complete = not rec.has_cont
        return rec

    def for_line(self, line: int) -> list:
        return [r for r in self.recs.values() if r.line == line]

_BIND_CACHE: dict = {}               # module-level (DESIGN A8): sha256(tools JSON) -> valid BIND votes
_BIND_CACHE_LOCK = threading.Lock()
SHORT_STRING_MAX = 100

@dataclass
class ArgFeat:
    name: str
    type: str                     # string | integer | numeric | boolean | array | object | other
    enum: Optional[list] = None
    max_len: Optional[int] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    max_items: Optional[int] = None
    items_type: Optional[str] = None
    items_props: dict = field(default_factory=dict)
    required: bool = False

    @property
    def is_enum(self) -> bool:
        return bool(self.enum)

    @property
    def is_string(self) -> bool:
        return self.type == "string" and not self.is_enum

    @property
    def free_text(self) -> bool:
        return self.is_string and (self.max_len is None or self.max_len >= SHORT_STRING_MAX)

    @property
    def short_key_arg(self) -> bool:
        """A short string argument that can carry an identifier (maxLength absent or <= 100)."""
        return self.is_string and (self.max_len is None or self.max_len <= SHORT_STRING_MAX)

def schema_type(spec: Any) -> str:
    """Normalised JSON-schema type. Any type that is not string/integer/boolean/array/object/null is
    the JSON-schema numeric type."""
    if not isinstance(spec, dict):
        return "other"
    t = spec.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t is None:
        if spec.get("enum"):
            return "string"
        if "properties" in spec:
            return "object"
        return "other"
    if t in ("string", "integer", "boolean", "array", "object"):
        return t
    if t == "null":
        return "other"
    return "numeric"

@dataclass
class ToolFeat:
    name: str
    description: str
    args: dict
    required: list
    schema: dict

    @property
    def text(self) -> str:
        return " ".join(name_tokens(self.name)) + " " + self.description

def tool_features(tool: dict) -> Optional[ToolFeat]:
    fn = tool.get("function", tool) if isinstance(tool, dict) else None
    if not isinstance(fn, dict) or not isinstance(fn.get(NAME_KEY), str):
        return None
    params = fn.get("parameters") or {}
    props = params.get("properties") or {} if isinstance(params, dict) else {}
    required = [r for r in (params.get("required") or []) if isinstance(r, str)] if isinstance(params, dict) else []
    stated = [int(x) for x in re.findall(r"(?<![\w.])\d{1,2}(?!\w|\.\d)", str(fn.get("description") or ""))]
    args = {}
    for name, spec in props.items():
        if not isinstance(spec, dict):
            continue
        t = schema_type(spec)
        items = spec.get("items") if isinstance(spec.get("items"), dict) else {}
        enum = spec.get("enum") if isinstance(spec.get("enum"), list) else None
        max_items = spec.get("maxItems") if isinstance(spec.get("maxItems"), int) else None
        least = spec.get("minItems") if isinstance(spec.get("minItems"), int) else 1
        limits = [n for n in stated if max(1, least) < n < (max_items or 65)]
        if t == "array" and limits and (not items or schema_type(items) == "string"):
            max_items = max(limits)
        args[name] = ArgFeat(
            name=name, type=t, enum=[e for e in enum if isinstance(e, (str, int, float))] if enum else None,
            max_len=spec.get("maxLength") if isinstance(spec.get("maxLength"), int) else None,
            minimum=spec.get("minimum") if isinstance(spec.get("minimum"), (int, float)) else None,
            maximum=spec.get("maximum") if isinstance(spec.get("maximum"), (int, float)) else None,
            max_items=max_items,
            items_type=schema_type(items) if items else None,
            items_props=items.get("properties") or {} if isinstance(items, dict) else {},
            required=name in required)
    return ToolFeat(name=fn[NAME_KEY], description=str(fn.get("description") or ""), args=args,
                    required=required, schema=params if isinstance(params, dict) else {})

def plausibility(tf: ToolFeat) -> dict:
    """Structural plausibility per role (0 = the structural requirement is violated: a veto)."""
    args = list(tf.args.values())
    req = [a for a in args if a.required]
    strings = [a for a in args if a.is_string]
    enums = [a for a in args if a.is_enum]
    ints = [a for a in args if a.type == "integer"]
    nums = [a for a in args if a.type == "numeric"]
    lists_str = [a for a in args if a.type == "array" and a.items_type in ("string", None, "other")]
    lists_obj = [a for a in args if a.type == "array" and a.items_type == "object"]
    out = {role: 0.0 for role in ALL_ROLES}

    req_text = [a for a in req if a.free_text]
    if len(req_text) == 1 and len(args) <= 2 and len(req) <= 2 and all(a is req_text[0] or a.is_enum for a in args):
        out[ROLE_MESSAGE] = (1.0 if (req_text[0].max_len or 0) >= SHORT_STRING_MAX else 0.75) - \
            (0.15 if len(args) == 2 else 0.0)
    if len(args) == 1 and len(req) == 1 and req[0].short_key_arg:
        out[ROLE_VIEW] = out[ROLE_SPEC] = 1.0 if req[0].max_len is not None else 0.7
    elif len(req) == 1 and req[0].short_key_arg and all(
            a.required or a.is_enum or a.type == "boolean" for a in args):
        out[ROLE_VIEW] = out[ROLE_SPEC] = 0.6        # one required id plus optional flags or enums (F21)
    req_lists = [a for a in req if a in lists_str]
    if len(req_lists) == 1 and len(req) == 1:
        out[ROLE_BATCH] = 1.0 if len(args) == 1 else 0.8
    id_req = [a for a in req if a.short_key_arg]
    cartish = bool(id_req) and any(a.required for a in ints + enums) and not req_lists and len(req) <= 3
    weak_cart = (len(req) == 1 and len(id_req) == 1 and len(strings) == 1 and not enums and not nums
                 and not lists_str and not lists_obj and len(ints) == 1 and not ints[0].required)
    if any(a.required for a in lists_obj):
        out[ROLE_CART_WRITE] = 0.9
    elif cartish and len(strings) == 1:
        out[ROLE_CART_WRITE] = 1.0 if ints else 0.85
    elif weak_cart:
        out[ROLE_CART_WRITE] = 0.3
    if strings and not req_lists and not (cartish and len(strings) == 1):
        paging = [a for a in ints if not a.required] + [a for a in strings if not a.required and a.short_key_arg
                                                         and a.max_len is not None and a.max_len <= 40]
        shortish = [a for a in strings if a.short_key_arg and a not in paging]
        if len(args) == 1 and len(req) == 1:
            score = 0.1 if req[0].free_text else 0.05      # a lone required string is a view or message
        else:
            score = 0.4
            score += 0.25 if paging else 0.0
            score += 0.15 if any(a.required for a in strings + enums) else 0.0
            score += 0.1 if enums else 0.0
            score += 0.05 if any(a.free_text for a in strings) else 0.0
            if not req and len(shortish) >= 2 and nums:
                score -= 0.25                                   # a filter-only tool
            if not req and not paging and not enums:
                score -= 0.2
        out[ROLE_SEARCH] = clamp(score, 0.05, 1.0)
        if (shortish or nums) and not (len(args) == 1 and len(req) == 1):
            f = 0.4 + (0.25 if nums else 0.0) + (0.15 if len(shortish) >= 2 else 0.0) + (0.1 if enums else 0.0) \
                + (0.15 if paging and shortish else 0.0)
            if req_text and not enums:
                f -= 0.2
            out[ROLE_FILTER] = clamp(f, 0.05, 1.0)
    if not args:
        out[ROLE_CART_VIEW] = out[ROLE_STATUS] = 1.0
        out[ROLE_ORDER] = out[ROLE_DECLINE] = 1.0
    elif not req and len(args) == 1 and strings and strings[0].free_text:
        out[ROLE_ORDER] = out[ROLE_DECLINE] = 0.9           # an optional remark only
    elif not req and len(args) == 1 and not lists_str and not lists_obj:
        out[ROLE_CART_VIEW] = out[ROLE_STATUS] = 0.6        # one optional page, flag or enum (F21)
        out[ROLE_ORDER] = out[ROLE_DECLINE] = 0.5
    req_enums = [a for a in req if a.is_enum and len(a.enum) >= 2]
    if len(req) == 1 and req_enums and all(a.required or a.free_text for a in args):
        out[ROLE_FINISH] = 1.0
    return out

def guess_slots(tf: ToolFeat, role: str) -> dict:
    """Structural slot guess per argument: {arg: (slot, conf)} (conf 2 = unambiguous by type)."""
    out: dict = {}
    args = list(tf.args.values())
    strings = [a for a in args if a.is_string]
    enums = [a for a in args if a.is_enum]
    ints = [a for a in args if a.type == "integer"]
    nums = [a for a in args if a.type == "numeric"]
    if role in (ROLE_SEARCH, ROLE_FILTER):
        for a in enums:
            out[a.name] = (SLOT_MODE, 2 if len(enums) == 1 else 1)
        for a in ints:
            slot = SLOT_OFFSET if a.minimum is not None and a.minimum <= 0 else SLOT_PAGE
            out[a.name] = (slot, 2 if len(ints) == 1 else 1)
        for a in nums:
            out[a.name] = (SLOT_CEILING, 2 if len(nums) == 1 else 1)
        query = None
        if role == ROLE_SEARCH:
            req_strings = [a for a in strings if a.required]
            if len(req_strings) == 1:
                query = (req_strings[0], 2)
            elif len(strings) == 1:
                query = (strings[0], 2)
            elif len(strings) >= 3:
                lengths = [a.max_len for a in strings]
                odd = [a for a in strings if lengths.count(a.max_len) == 1 and not (
                    a.max_len is not None and a.max_len <= 40 and not a.required)]
                if len(odd) == 1:
                    query = (odd[0], 1)
            if query is None and strings:
                query = (max(strings, key=lambda a: (a.max_len or 10_000)), 1)
        rest = [a for a in strings if query is None or a is not query[0]]
        if query is not None:
            out[query[0].name] = (SLOT_QUERY, query[1])
        tokens = [a for a in rest if not a.required and a.max_len is not None and a.max_len <= 40]
        for a in tokens:
            out[a.name] = (SLOT_TOKEN, 1)
        pair = [a for a in rest if a not in tokens]
        for i, a in enumerate(pair):
            out[a.name] = ((SLOT_SHELF, SLOT_MAKER)[i] if i < 2 else SLOT_OTHER, 1)
        return out
    if role in (ROLE_VIEW, ROLE_SPEC):
        for a in strings:
            out[a.name] = (SLOT_ID, 2 if len(strings) == 1 else 1)
        return out
    if role == ROLE_BATCH:
        for a in args:
            if a.type == "array":
                out[a.name] = (SLOT_IDS, 2)
        return out
    if role == ROLE_CART_WRITE:
        for a in args:
            if a.type == "array":
                out[a.name] = (SLOT_ITEMS, 2)
            elif a.is_string:
                out[a.name] = (SLOT_ID, 2 if len(strings) == 1 else 1)
            elif a.type == "integer":
                out[a.name] = (SLOT_QTY, 2 if len(ints) == 1 else 1)
            elif a.is_enum:
                out[a.name] = (SLOT_CART_MODE, 2 if len(enums) == 1 else 1)
        return out
    if role == ROLE_MESSAGE:
        for a in args:
            out[a.name] = (SLOT_TEXT, 2) if a.free_text and a.required else ((SLOT_KEY, 1) if a.is_enum else
                                                                             (SLOT_OTHER, 1))
        return out
    if role in (ROLE_ORDER, ROLE_DECLINE, ROLE_FINISH):
        for a in args:
            out[a.name] = (SLOT_FINISH_MODE, 2) if a.is_enum and a.required else (SLOT_REMARK, 2)
        return out
    return out

def role_fits(primary: str, wanted: str) -> bool:
    """May a tool whose primary role is `primary` serve an intent that needs `wanted`?"""
    if wanted in (ROLE_VIEW, ROLE_SPEC):
        return primary in RECORD_ROLES
    if wanted == ROLE_FILTER:
        return primary in (ROLE_FILTER, ROLE_SEARCH)
    if wanted == ROLE_ORDER:
        return primary in (ROLE_ORDER, ROLE_OTHER)      # never a cart view, status or decline tool (PROT-5)
    if wanted == ROLE_DECLINE:
        return primary in (ROLE_DECLINE, ROLE_OTHER)
    return primary == wanted

REPEATABLE_READ_ROLES = (ROLE_SEARCH, ROLE_FILTER, ROLE_VIEW, ROLE_SPEC, ROLE_BATCH)

SLOT_LABELS = SEARCH_SLOTS + ITEM_SLOTS + TEXT_SLOTS
MEANING_LABELS = SEARCH_MEANINGS + CART_MEANINGS + FINISH_MEANINGS + SORT_MEANINGS

class Binder:
    """Learns the world's disguised tools. Reads/writes ep.binding, ep.budget.known_good_shapes."""

    def __init__(self, ep: Any):
        self.ep = ep
        self.binding = Binding()
        self.features: dict = {}
        self.votes: list = []
        self.repairs: dict = {}
        self.revoted = False
        self.failed_calls: set = set()
        self.structural_best: dict = {}
        self.tools_hash = ""
        self.mode_flips = 0
        self.dropped_labels: list = []       # oracle labels that could not be mapped (logged in the bind decision)
        self.pending_acks: dict = {}         # item -> cart tool: ACK-only writes awaiting a cart read (F18)

    def tool_digest(self) -> list:
        out = []
        for tf in self.features.values():
            out.append({"tname": tf.name, "about": clip_text(tf.description, 300),
                        "params": {a.name: {k: v for k, v in (("kind", a.type), ("enum", a.enum),
                                                               ("maxLength", a.max_len), ("min", a.minimum),
                                                               ("max", a.maximum), ("required", a.required))
                                            if v is not None} for a in tf.args.values()}})
        return out

    def load_tools(self, tools_raw: list) -> None:
        self.tools_hash = stable_hash(tools_raw)
        for tool in tools_raw or []:
            tf = tool_features(tool)
            if tf is None:
                continue
            self.features[tf.name] = tf
            tb = ToolBind(tool=tf.name, schema=tf.schema, description=tf.description, plaus=plausibility(tf))
            self.binding.tools[tf.name] = tb
        for role in ALL_ROLES:
            ranked = sorted((t for t in self.binding.tools.values() if t.plaus.get(role, 0) > 0),
                            key=lambda t: -t.plaus[role])
            if ranked:
                top = ranked[0].plaus[role]
                second = ranked[1].plaus[role] if len(ranked) > 1 else 0.0
                self.structural_best[role] = ranked[0].tool if top - second >= 0.1 else None

    def cached_votes(self) -> Optional[list]:
        with _BIND_CACHE_LOCK:
            hit = _BIND_CACHE.get(self.tools_hash)
        return list(hit) if hit else None

    def store_votes(self, votes: list) -> None:
        if len(votes) >= 2:
            with _BIND_CACHE_LOCK:
                _BIND_CACHE[self.tools_hash] = list(votes)

    def unmapped_labels(self, answer: Any) -> list:
        """Role and enum-meaning labels of a BIND answer that map to no lodestar label, even bare (LS-06)."""
        bad: list = []
        if not isinstance(answer, dict):
            return bad
        names = {n.lower(): n for n in self.binding.tools}
        for entry in answer.get("tools") or []:
            if not isinstance(entry, dict):
                continue
            name = names.get(str(entry.get("tname") or "").strip().lower())
            if name is None:
                continue
            if canon_label(entry.get("role"), ALL_ROLES) is None:
                bad.append(f"{name}:{entry.get('role')}")
            for arg, table in (entry.get("enum_map") or {}).items():
                feat = self.features[name].args.get(arg)
                if feat is None or not feat.is_enum or not isinstance(table, dict):
                    continue
                for lit, meaning in table.items():
                    if str(lit) in {str(e) for e in feat.enum} and canon_label(meaning, MEANING_LABELS) is None:
                        bad.append(f"{name}.{arg}={lit}:{meaning}")
        return bad

    def vote_validator(self, answer: dict) -> tuple:
        """BIND validator: every role and enum meaning must map to a lodestar label (bare labels are mapped);
        otherwise the gateway's repair re-ask runs."""
        bad = self.unmapped_labels(answer)
        if bad:
            return False, None, clip_text("unmapped labels: " + ", ".join(bad), 120)
        return True, answer, ""

    def normalize_vote(self, answer: Optional[dict]) -> Optional[dict]:
        """A BIND answer mapped onto known tools: bare labels become namespaced ones when the match is unique,
        vetoed roles and labels that still map to nothing are removed (and remembered for the bind log)."""
        if not isinstance(answer, dict):
            return None
        out = {}
        names = {n.lower(): n for n in self.binding.tools}
        for entry in answer.get("tools") or []:
            if not isinstance(entry, dict):
                continue
            name = names.get(str(entry.get("tname") or "").strip().lower())
            if name is None:
                continue
            tb = self.binding.tools[name]
            role = canon_label(entry.get("role"), ALL_ROLES)
            if role is None and entry.get("role"):
                self.dropped_labels.append(f"{name}:{entry.get('role')}")
            if role is None or (role != ROLE_OTHER and tb.plaus.get(role, 0) <= 0):
                role = None                      # a structural violation vetoes the pairing
            conf = clamp(float(as_number(entry.get("conf")) or 1), 0, 3)
            slots = {}
            for a, label in (entry.get("args") or {}).items():
                slot = canon_label(label, SLOT_LABELS)
                if a in self.features[name].args and slot is not None:
                    slots[a] = slot
                elif a in self.features[name].args:
                    self.dropped_labels.append(f"{name}.{a}:{label}")
            meanings = {}
            for arg, table in (entry.get("enum_map") or {}).items():
                feat = self.features[name].args.get(arg)
                if feat is None or not feat.is_enum or not isinstance(table, dict):
                    continue
                literals = {str(e) for e in feat.enum}
                mapped = {}
                for lit, label in table.items():
                    meaning = canon_label(label, MEANING_LABELS)
                    if str(lit) in literals and meaning is not None:
                        mapped[str(lit)] = meaning
                    elif str(lit) in literals:
                        self.dropped_labels.append(f"{name}.{arg}={lit}:{label}")
                meanings[arg] = mapped
            returns = {f for f in (canon_label(x, FAMILIES) for x in (entry.get("returns") or [])) if f}
            out[name] = {"role": role, "conf": conf, "slots": slots, "meanings": meanings, "returns": returns}
        return out if out else None

    def conflicts(self, votes: list) -> list:
        """Critical roles or enum meanings on which two votes disagree."""
        if len(votes) < 2:
            return []
        a, b = votes[0], votes[1]
        out = []
        for role in (ROLE_SEARCH, ROLE_CART_WRITE, ROLE_MESSAGE, ROLE_ORDER, ROLE_DECLINE, ROLE_FINISH):
            ta = sorted(t for t, e in a.items() if e["role"] == role)
            tb = sorted(t for t, e in b.items() if e["role"] == role)
            if ta != tb:
                out.append(role)
        rec_a = sorted(t for t, e in a.items() if e["role"] in RECORD_ROLES)
        rec_b = sorted(t for t, e in b.items() if e["role"] in RECORD_ROLES)
        if not set(rec_a) & set(rec_b):
            out.append("record")
        for tool in [t for t in a if t in b]:
            if a[tool]["meanings"] != b[tool]["meanings"] and (a[tool]["meanings"] or b[tool]["meanings"]):
                out.append(f"enum:{tool}")
        return out

    def build(self, votes: list, arbiter: Optional[dict] = None) -> Binding:
        """Merge votes (and an arbiter answer) with structure into ep.binding."""
        bnd = self.binding
        sources = list(votes) + ([arbiter] if arbiter else [])
        bnd.votes_used = len(votes)
        bnd.source = "arbiter" if arbiter else ("votes" if votes else "structure")
        self.votes = sources
        for name, tb in bnd.tools.items():
            tf = self.features[name]
            tb.vote_roles = [s[name]["role"] for s in sources if name in s and s[name]["role"]]
            tb.decline_votes = sum(1 for r in tb.vote_roles if r == ROLE_DECLINE)
            tb.provides_expected = set().union(*[s[name]["returns"] for s in sources if name in s]) \
                if any(name in s for s in sources) else set()
        for role in ALL_ROLES:
            scored = []
            for name, tb in bnd.tools.items():
                plaus = tb.plaus.get(role, 0.0)
                if plaus <= 0:
                    continue
                naming = [s[name] for s in sources if name in s and s[name]["role"] == role]
                agree = 1.0 if len(naming) >= 2 else 0.0
                mean_conf = (sum(e["conf"] for e in naming) / len(naming) / 3.0) if naming else 0.0
                score = 0.45 * agree + 0.35 * mean_conf + 0.20 * plaus
                scored.append((score, -list(bnd.tools).index(name), name, len(naming)))
            scored.sort(reverse=True)
            bnd.by_role[role] = [name for _, _, name, _ in scored]
        self._assign_primary_roles(sources)
        self._assign_slots(sources)
        self._assign_finish(sources)
        self._assign_leftovers()
        return bnd

    def _assign_primary_roles(self, sources: list) -> None:
        bnd = self.binding
        taken: set = set()
        if sources:
            for name, tb in bnd.tools.items():
                if tb.vote_roles:
                    role = max(dict.fromkeys(tb.vote_roles),     # first-vote order breaks ties, never a set's
                               key=lambda r: (tb.vote_roles.count(r), r == tb.vote_roles[0]))
                    tb.role = role
                    agreeing = tb.vote_roles.count(role)
                    if role not in FINISH_ROLES and self.structural_best.get(role) == name:
                        agreeing += 1
                    tb.role_conf = min(2, agreeing)
                    taken.add(name)
        order = (ROLE_CART_WRITE, ROLE_BATCH, ROLE_FINISH, ROLE_MESSAGE, ROLE_SEARCH, ROLE_FILTER, ROLE_VIEW)
        for role in order:
            if any(bnd.tools[n].role == role for n in taken):
                continue
            cands = [n for n in bnd.by_role.get(role, []) if n not in taken]
            if role == ROLE_FILTER and not cands:
                search = [n for n in bnd.by_role.get(ROLE_FILTER, []) if bnd.tools[n].role == ROLE_SEARCH]
                if search:
                    continue                     # the search tool doubles as the filter tool
            if not cands:
                continue
            if role == ROLE_CART_WRITE and not sources:
                cands = [n for n in cands if bnd.tools[n].plaus[role] >= 0.5]   # weak cart shapes need votes
                if not cands:
                    continue
            if role == ROLE_VIEW:
                for n in cands:                  # every remaining single-id tool is a record tool
                    tb = bnd.tools[n]
                    tb.role = ROLE_VIEW
                    tb.role_conf = (2 if tb.plaus[ROLE_VIEW] >= 0.7 else 1) if not sources else max(tb.role_conf, 1)
                    taken.add(n)
                continue
            best = cands[0]
            tb = bnd.tools[best]
            tb.role = role
            clear = [n for n in bnd.by_role.get(role, []) if n not in taken and bnd.tools[n].plaus[role] >= 0.5]
            unique = clear == [best]
            tb.role_conf = 2 if (unique and role != ROLE_FINISH and not sources) else max(tb.role_conf, 1)
            if sources and self.structural_best.get(role) == best:
                tb.role_conf = max(tb.role_conf, 1)
            taken.add(best)
        for name, tb in bnd.tools.items():
            if name not in taken and tb.role == ROLE_OTHER:
                tb.role_conf = 0

    def _assign_slots(self, sources: list) -> None:
        for name, tb in self.binding.tools.items():
            tf = self.features[name]
            guess = guess_slots(tf, tb.role)
            for arg in tf.args:
                voted = [s[name]["slots"].get(arg) for s in sources if name in s and s[name]["slots"].get(arg)]
                g = guess.get(arg)
                if voted:
                    slot = max(sorted(set(voted)), key=lambda x: (voted.count(x), g is not None and x == g[0],
                                                                  -voted.index(x)))
                    conf = min(2, voted.count(slot) + (1 if g and g[0] == slot else 0))
                    tb.slots[arg] = (slot, conf)
                elif g:
                    tb.slots[arg] = g if tb.role_conf >= 2 else (g[0], min(g[1], 1))
                feat = tf.args[arg]
                if feat.is_enum:
                    table = {}
                    for lit in feat.enum:
                        lit_s = str(lit)
                        ms = [s[name]["meanings"].get(arg, {}).get(lit_s) for s in sources if name in s]
                        ms = [m for m in ms if m]
                        if not ms:
                            continue
                        ranked = sorted(set(ms), key=lambda x: (-ms.count(x), ms.index(x)))
                        if len(ranked) > 1 and ms.count(ranked[0]) == ms.count(ranked[1]):
                            continue                     # an exact tie is unresolved (LS-07)
                        table[lit_s] = (ranked[0], min(2, ms.count(ranked[0])))
                    if not table:
                        table = self._positional_meanings(tb, arg, feat)
                    tb.enum_map[arg] = table

    @staticmethod
    def _positional_meanings(tb: ToolBind, arg: str, feat: ArgFeat) -> dict:
        """No votes: the first literal is the primary action (text search, add, order); conf 1."""
        slot = tb.slots.get(arg, (SLOT_OTHER, 0))[0]
        lits = [str(e) for e in feat.enum]
        if slot == SLOT_MODE:
            primary, rest = SEM_TEXT, SEM_FILTER_ANY
        elif slot == SLOT_CART_MODE:
            primary, rest = SEM_ADD, SEM_REMOVE
        elif slot == SLOT_FINISH_MODE:
            primary, rest = SEM_ORDER, SEM_DECLINE
        else:
            return {}
        table = {lits[0]: (primary, 1)}
        for lit in lits[1:]:
            table[lit] = (rest, 1)
        return table

    def _assign_finish(self, sources: list) -> None:
        bnd = self.binding
        enum_tools = [n for n, tb in bnd.tools.items() if tb.role == ROLE_FINISH]
        if enum_tools:
            name = enum_tools[0]
            tb = bnd.tools[name]
            arg = next((a for a, (s, _) in tb.slots.items() if s == SLOT_FINISH_MODE), None)
            if arg is None:
                arg = next((a.name for a in self.features[name].args.values() if a.is_enum), None)
            table = tb.enum_map.get(arg, {}) if arg else {}
            order_lit = next((lit for lit, (m, _) in table.items() if m == SEM_ORDER), None)
            decline_lit = next((lit for lit, (m, _) in table.items() if m == SEM_DECLINE and lit != order_lit),
                               None)
            if arg and table and (order_lit is None or decline_lit is None):
                lits = list(table)
                order_lit = order_lit or next((x for x in lits if x != decline_lit), None)
                decline_lit = decline_lit or next((x for x in lits if x != order_lit), None)
            if order_lit is not None and order_lit == decline_lit:
                decline_lit = None                 # never one literal for both finishes (LS-07)
            bnd.finish_kind = "enum"
            bnd.finish_tool, bnd.finish_arg = name, arg
            bnd.order_literal, bnd.decline_literal = order_lit, decline_lit
            confs = [c for lit, (m, c) in table.items() if m in (SEM_ORDER, SEM_DECLINE)]
            vote_ok = sum(1 for r in tb.vote_roles if r == ROLE_FINISH) >= 2 and confs and min(confs) >= 2
            bnd.finish_conf = 2 if vote_ok and decline_lit is not None else 1
            return
        order_votes = {n: tb.vote_roles.count(ROLE_ORDER) for n, tb in bnd.tools.items()}
        decline_votes = {n: tb.vote_roles.count(ROLE_DECLINE) for n, tb in bnd.tools.items()}
        if sources and max(order_votes.values() or [0]) >= 1 and max(decline_votes.values() or [0]) >= 1:
            named = [n for n in bnd.tools if order_votes[n] or decline_votes[n]]
            cart_text = self._cart_text()

            def lean(n: str) -> float:
                """Vocabulary-free order lean: an optional free-text remark marks a decline tool; more
                overlap with the cart-write text marks the order tool."""
                return token_overlap(self.features[n].text, cart_text) - (1.0 if self._remark_only(n) else 0.0)

            order = max(named, key=lambda n: (order_votes[n] - decline_votes[n], lean(n), -list(bnd.tools).index(n)))
            decline = max((n for n in named if n != order),
                          key=lambda n: (decline_votes[n] - order_votes[n], -lean(n), -list(bnd.tools).index(n)),
                          default=None)
            margins = sorted((order_votes[n] - decline_votes[n] for n in named), reverse=True)
            tied = order_votes[order] <= decline_votes[order] or (len(margins) > 1 and margins[0] == margins[1])
            bnd.finish_kind = "two_tools"
            bnd.by_role[ROLE_ORDER] = [order] + [n for n in bnd.by_role.get(ROLE_ORDER, []) if n != order]
            if decline:
                bnd.by_role[ROLE_DECLINE] = [decline] + [n for n in bnd.by_role.get(ROLE_DECLINE, []) if n != decline]
            bnd.tools[order].role = ROLE_ORDER
            if decline:
                bnd.tools[decline].role = ROLE_DECLINE
            agreed = order_votes[order] >= 2 and decline and decline_votes[decline] >= 2
            bnd.finish_conf = 2 if agreed and not tied else 1
            if tied:
                self.ep.trace.decision("finish_uncertain", why="finish votes tie; structure breaks it",
                                       order_tool=order, decline_tool=decline)
            return
        self._structural_finish()

    def _remark_only(self, name: str) -> bool:
        """The tool's only argument is an optional free-text string (a closing remark)."""
        args = list(self.features[name].args.values())
        return len(args) == 1 and not args[0].required and args[0].free_text

    def _structural_finish(self, exclude: tuple = ()) -> None:
        """No agreeing votes on a two-tool finish: rank the no-argument tools (DESIGN 5.4 'uncertain'
        finish). Vocabulary-free heuristics, recorded as a deviation: an optional free-text remark marks
        the decline tool; otherwise the two most alike no-argument tools are {order, decline} (two remaining
        tools are paired even when nothing is alike) and the one more like the cart-write tool orders. The
        finish stays conf 1 (uncertain) and is logged. exclude: tools proven not to finish (re-pairing)."""
        bnd = self.binding
        cands = [n for n, tb in bnd.tools.items() if tb.plaus.get(ROLE_ORDER, 0) > 0 and n not in exclude
                 and tb.role in (ROLE_OTHER, ROLE_ORDER, ROLE_DECLINE, ROLE_CART_VIEW, ROLE_STATUS)
                 and not (tb.vote_roles and tb.role in (ROLE_CART_VIEW, ROLE_STATUS))]
        if not cands:
            bnd.finish_kind = "uncertain"
            bnd.finish_conf = 0
            bnd.by_role[ROLE_ORDER] = []
            bnd.by_role[ROLE_DECLINE] = []
            return
        cart_text = self._cart_text()

        def text(n: str) -> str:
            return self.features[n].text

        remark_tools = [n for n in cands if self._remark_only(n)]
        decline = remark_tools[0] if len(remark_tools) == 1 else None
        plain = [n for n in cands if n != decline]
        if decline is None and len(plain) >= 2:
            best_pair, best_sim = None, 0.0
            for i, a in enumerate(plain):
                for b in plain[i + 1:]:
                    sim = token_overlap(text(a), text(b))
                    if sim > best_sim:
                        best_pair, best_sim = (a, b), sim
            if best_pair is None and len(plain) == 2:
                best_pair = (plain[0], plain[1])      # two tools left: they are the pair (LS-12)
            if best_pair is not None:
                a, b = best_pair
                if token_overlap(text(b), cart_text) > token_overlap(text(a), cart_text):
                    a, b = b, a
                decline = b
                plain = [a] + [n for n in plain if n not in (a, b)]

        def order_score(n: str) -> tuple:
            fam = token_overlap(text(n), text(decline)) if decline else 0.0
            return (fam + 0.5 * token_overlap(text(n), cart_text), -list(bnd.tools).index(n))

        ranked = sorted(plain, key=order_score, reverse=True)
        if decline is None and len(ranked) >= 3:      # v7: no pair stands out: the tool least like an order
            decline = ranked.pop()                    # declines (a wrong guess is demoted on first use)
        bnd.finish_kind = "two_tools"
        bnd.by_role[ROLE_ORDER] = ranked
        bnd.by_role[ROLE_DECLINE] = [decline] if decline else []
        if ranked:
            bnd.tools[ranked[0]].role = ROLE_ORDER
            bnd.tools[ranked[0]].role_conf = 1
        if decline:
            bnd.tools[decline].role = ROLE_DECLINE
            bnd.tools[decline].role_conf = 1
        for n in ranked[1:]:
            if bnd.tools[n].role in (ROLE_OTHER, ROLE_ORDER, ROLE_DECLINE):
                bnd.tools[n].role = ROLE_OTHER
        bnd.finish_conf = 1

    def repair_finish(self, why: str) -> None:
        """A structural finish tool answered like a cart view or a status tool: pair the remaining
        no-argument tools again (LS-12). Vote-agreed finishes are left alone."""
        bnd = self.binding
        if bnd.finish_kind != "two_tools" or bnd.finish_conf >= 2:
            return
        if any(r in (ROLE_ORDER, ROLE_DECLINE) for tb in bnd.tools.values() for r in tb.vote_roles):
            return                                     # a voted pairing is kept; tool_for skips the demoted tool
        proven = set(bnd.demoted.get(ROLE_ORDER, set())) | set(bnd.demoted.get(ROLE_DECLINE, set()))
        self._structural_finish(exclude=tuple(sorted(proven)))
        self._assign_leftovers()
        self.ep.trace.decision("bind_repair", bind_role=ROLE_ORDER, rung="re-pair", why=clip_text(why, 160),
                               order_tool=self.tool_for(ROLE_ORDER), decline_tool=self.tool_for(ROLE_DECLINE))

    def decline_not(self, call: Call, kind: str) -> None:
        """v7: an uncertain decline answered by an empty-cart error (it needs a cart: it orders) or by an unreadable
        answer: an enum finish swaps its literals; a two-tool finish pairs the other tools again."""
        bnd = self.binding
        if call.tool == bnd.finish_tool and bnd.finish_arg:
            if kind == D_EMPTY_CART and call.args.get(bnd.finish_arg) == bnd.decline_literal and bnd.order_literal:
                bnd.order_literal, bnd.decline_literal = bnd.decline_literal, bnd.order_literal
                self.ep.trace.decision("bind_repair", bind_role=ROLE_FINISH, rung="swap literals", why=kind)
            return
        if any(r in (ROLE_ORDER, ROLE_DECLINE) for tb in bnd.tools.values() for r in tb.vote_roles):
            return                                     # a voted pairing is kept
        self.demote(call.tool, ROLE_DECLINE, f"a decline answered {kind}")
        if bnd.tools[call.tool].role == ROLE_DECLINE:
            bnd.tools[call.tool].role = ROLE_OTHER
        self.repair_finish(f"{call.tool} answered a decline with {kind}")
        if kind == D_EMPTY_CART:
            bnd.by_role[ROLE_ORDER] = [call.tool] + [n for n in bnd.by_role.get(ROLE_ORDER, []) if n != call.tool]
            bnd.tools[call.tool].role = ROLE_ORDER

    def finish_kind_of(self, call: Call, intent_kind: str) -> str:
        """FIN_ORDER or FIN_DECLINE for a finish call that was accepted, read from the tool (and literal)
        actually used rather than from the intent (PROT-5)."""
        bnd = self.binding
        if call.tool == bnd.finish_tool and bnd.finish_arg:
            lit = call.args.get(bnd.finish_arg)
            if lit == bnd.decline_literal:
                return FIN_DECLINE
            if lit == bnd.order_literal:
                return FIN_ORDER
        tb = bnd.tools.get(call.tool)
        if tb is not None and tb.role == ROLE_DECLINE:
            return FIN_DECLINE
        if tb is not None and tb.role == ROLE_ORDER:
            return FIN_ORDER
        return FIN_DECLINE if intent_kind == IK_DECLINE else FIN_ORDER

    def _cart_text(self) -> str:
        cart = next((n for n, tb in self.binding.tools.items() if tb.role == ROLE_CART_WRITE), None)
        return self.features[cart].text if cart else ""

    def _assign_leftovers(self) -> None:
        """No-argument tools nobody claimed: the one most like the cart-write tool views the cart, the
        rest are status tools (conf 1; first use confirms or demotes)."""
        bnd = self.binding
        cart_text = self._cart_text()
        left = [n for n, tb in bnd.tools.items() if tb.role == ROLE_OTHER and tb.plaus.get(ROLE_CART_VIEW, 0) > 0]
        left.sort(key=lambda n: (-token_overlap(self.features[n].text, cart_text), list(bnd.tools).index(n)))
        has_view = any(tb.role == ROLE_CART_VIEW for tb in bnd.tools.values())
        for n in left:
            tb = bnd.tools[n]
            like_cart = token_overlap(self.features[n].text, cart_text) > 0
            tb.role = ROLE_CART_VIEW if (like_cart and not has_view) else ROLE_STATUS
            tb.role_conf = 1
            has_view = has_view or tb.role == ROLE_CART_VIEW
        bnd.by_role[ROLE_CART_VIEW] = [n for n, tb in bnd.tools.items() if tb.role == ROLE_CART_VIEW]
        bnd.by_role[ROLE_STATUS] = [n for n, tb in bnd.tools.items() if tb.role == ROLE_STATUS]

    def tool_for(self, role: str, exclude: tuple = ()) -> Optional[str]:
        """Best tool for a role. A non-finish role never resolves to a finish tool, and ORDER never
        resolves to a tool labelled decline while an alternative exists."""
        bnd = self.binding
        demoted = bnd.demoted.get(role, set())
        fallback = None
        for name in bnd.by_role.get(role, []):
            tb = bnd.tools.get(name)
            if tb is None or name in demoted or name in exclude or tb.strikes >= 2:
                continue
            if not role_fits(tb.role, role):
                continue
            if role == ROLE_ORDER and (tb.decline_votes or tb.role == ROLE_DECLINE):
                fallback = fallback or name
                continue
            return name
        return fallback if role == ROLE_ORDER else None

    def record_tools(self) -> list:
        return [n for n, tb in self.binding.tools.items() if tb.role in RECORD_ROLES and tb.strikes < 2
                and n not in self.binding.demoted.get(tb.role, set())]

    def _covers(self, name: str, families: set) -> float:
        """How well a record tool covers some field families: the share it is known to provide, ranked
        observed (seen) > expected (votes); 0.5 when nothing is known yet (worth trying)."""
        tb = self.binding.tools[name]
        if not families:
            return 1.0
        if tb.provides_seen:
            return 2.0 * len(families & tb.provides_seen) / len(families)
        if tb.provides_expected:
            return 1.5 * len(families & tb.provides_expected) / len(families)
        return 0.5

    def open_plan(self, need: str) -> list:
        """[(tool, families)]: record tools that together cover `need`, greedy by coverage (DESIGN 5.4 'the
        cheapest tool set covering its need'). Engines issue one OPEN per entry when tools are split."""
        left = set(NEED_FAMILIES.get(need, ()))
        out = []
        tools = self.record_tools()
        while left and tools:
            best = max(tools, key=lambda n: (self._covers(n, left), -list(self.binding.tools).index(n)))
            tb = self.binding.tools[best]
            known = tb.provides_seen or tb.provides_expected
            got = (left & known) if known else set(left)
            if not got:
                break
            out.append((best, sorted(got)))
            left -= got
            tools = [n for n in tools if n != best]
        return out

    def route_open(self, ids: list, need: str) -> tuple:
        """(tool, ids for this call): the record tool that best covers `need` (DESIGN 5.4); a tool known to
        cover none of it is never chosen (F24). open_plan() lists the tools for a split need."""
        families = set(NEED_FAMILIES.get(need, ()))
        tools = self.record_tools()
        if not tools:
            return None, []
        ranked = sorted(tools, key=lambda n: (self._covers(n, families),
                                             self.binding.tools[n].role == ROLE_BATCH and len(ids) > 1,
                                             -list(self.binding.tools).index(n)), reverse=True)
        best = ranked[0]
        if self._covers(best, families) <= 0:
            return None, []
        if len(ids) > 1:     # M4: batch first (IMPLEMENTATION 4); an unobserved batch tool is tried once
            best = next((n for n in ranked if self.binding.tools[n].role == ROLE_BATCH and self._covers(n, families) > 0
                         and (not self.binding.tools[n].provides_seen
                              or self._covers(n, families) >= self._covers(best, families))), best)
        if self.binding.tools[best].role == ROLE_BATCH:
            ids_arg = self.slot_arg(best, SLOT_IDS)
            max_items = self.features[best].args[ids_arg].max_items if ids_arg else None
            limit = self.ep.contract_learner.batch_limit(max_items)
            return best, list(ids)[:max(1, limit)]
        return best, list(ids)[:1]

    def slot_arg(self, tool: str, slot: str) -> Optional[str]:
        tb = self.binding.tools.get(tool)
        if tb is None:
            return None
        best = None
        for arg, (s, conf) in tb.slots.items():
            if s == slot and (best is None or conf > tb.slots[best][1]):
                best = arg
        return best

    def literal_for(self, tool: str, arg: str, meaning: str) -> Optional[str]:
        table = self.binding.tools[tool].enum_map.get(arg, {})
        best = None
        for lit, (m, conf) in table.items():
            if m == meaning and (best is None or conf > table[best][1]):
                best = lit
        return best

    def meaning_conf(self, tool: str, arg: str, literal: str) -> int:
        return self.binding.tools[tool].enum_map.get(arg, {}).get(literal, (None, 0))[1]

    def demote(self, tool: str, role: str, why: str) -> None:
        self.binding.demoted.setdefault(role, set()).add(tool)
        self.ep.trace.decision("bind_demote", tool=tool, bind_role=role, why=clip_text(why, 160))

    def flip_cart_mode(self) -> bool:
        """An add literal that leaves the cart empty may be the remove literal: swap positional meanings. A
        second flip swaps them back (a write that was merely dropped looks the same once); never more."""
        if self.mode_flips >= 2:
            return False
        tool = self.tool_for(ROLE_CART_WRITE)
        arg = self.slot_arg(tool, SLOT_CART_MODE) if tool else None
        if not arg:
            return False
        table = self.binding.tools[tool].enum_map.get(arg, {})
        if any(conf >= 2 for _, conf in table.values()):
            return False
        lits = list(table)
        if len(lits) < 2:
            return False
        add = self.literal_for(tool, arg, SEM_ADD)
        other = next(l for l in lits if l != add)
        table[other] = (SEM_ADD, 1)
        table[add] = (SEM_REMOVE, 1)
        self.mode_flips += 1
        self.ep.trace.decision("bind_flip_cart_mode", tool=tool, arg=arg, add_literal=other)
        return True

    def confirm(self, intent: Intent, call: Call, dec: Decoded) -> None:
        """First-use signatures and confidence updates (DESIGN 5.4)."""
        tb = self.binding.tools.get(call.tool)
        if tb is None or dec.kind == D_BLOCKED:
            return
        role = call.role
        ok: Optional[bool] = None
        attributable = False
        if dec.kind == D_CART:
            self._settle_acks(dec)
        if dec.kind == D_SESSION_ERROR:
            ok = role == ROLE_MESSAGE and dec.session_code == CODE_BUDGET_EXCEEDED
            attributable = dec.session_code in ("unknown_tool", "invalid_arguments")
        elif role in (ROLE_SEARCH, ROLE_FILTER):
            ok = dec.kind == D_RECORDS
            if ok and role == ROLE_FILTER and intent.kind == IK_FILTER:
                ok = self._filter_holds(intent, call, dec)
            attributable = dec.kind in (D_EMPTY, D_WORLD_FAULT, D_REPLY, D_UNKNOWN)
            if dec.kind == D_UNKNOWN and holds_empty_list(dec.inner):
                attributable = False               # a zero-hit shape the decoder could not read (LS-03)
        elif role in RECORD_ROLES:
            got = set(self.ep.fields.extract(call.tool, r).get(FR_ID) for r in dec.records)
            ok = dec.kind in (D_RECORDS, D_RECORD) and bool(set(call.ids) & got)
            attributable = dec.kind in (D_EMPTY, D_WORLD_FAULT) and all(i in self.ep.catalog for i in call.ids)
            if dec.kind in (D_EMPTY, D_WORLD_FAULT) and tb.role == ROLE_BATCH and len(call.ids) > 1:
                limit = self.ep.contract_learner.on_batch_rejected(len(call.ids))
                self.ep.trace.decision("contract", batch_limit=limit, why="batch rejected")
                attributable = False
        elif role == ROLE_MESSAGE:
            ok = dec.kind == D_REPLY and len(str(dec.reply_text or "").split()) >= 3
            attributable = not ok
        elif role == ROLE_CART_WRITE:
            if dec.kind == D_CART:
                ok = any(line[0] in call.ids for line in dec.cart["lines"]) or (
                    call.write is not None and call.write[1] == 0)
            elif dec.kind == D_ACK:
                if tb.first_use_ok is None and self.tool_for(ROLE_CART_VIEW) is not None and call.write \
                        and call.write[1] > 0 and not (dec.cart and dec.cart.get("echo")):
                    self.pending_acks[call.write[0]] = call.tool
                    self.ep.budget.known_good_shapes.add(call.shape)
                    return
                ok = True if tb.first_use_ok is None else tb.first_use_ok
            elif dec.kind == D_WORLD_FAULT:
                ok = False
                self.failed_calls.add(call.canonical())
                self.ep.trace.decision("cart_fault", tool=call.tool, item=(call.write or (None,))[0],
                                       fault=clip_text(dec.fault_text, 120))
                return
            elif dec.kind == D_UNKNOWN:
                return                             # an unreadable answer: verified by a read-back (LS-05)
            else:
                ok = False
                attributable = dec.kind == D_EMPTY
        elif role in FINISH_ROLES:
            ok = dec.kind in (D_FINISHED, D_REFUSED, D_EMPTY_CART)
            if intent.kind == IK_DECLINE and dec.kind in (D_EMPTY_CART, D_UNKNOWN) and self.binding.finish_conf < 2:
                self.decline_not(call, dec.kind)
            elif dec.kind in (D_CART, D_STATUS, D_RECORDS, D_RECORD, D_REPLY):
                self.demote(call.tool, role, f"finish call returned {dec.kind}")
                if dec.kind == D_CART:
                    tb.role = ROLE_CART_VIEW
                    self.binding.by_role.setdefault(ROLE_CART_VIEW, []).insert(0, call.tool)
                elif dec.kind == D_STATUS:
                    tb.role = ROLE_STATUS
                    self.binding.by_role.setdefault(ROLE_STATUS, []).insert(0, call.tool)
                elif tb.role in (ROLE_ORDER, ROLE_DECLINE):
                    tb.role = ROLE_OTHER
                self.repair_finish(f"{call.tool} answered {dec.kind}")
        elif role == ROLE_CART_VIEW:
            ok = dec.kind == D_CART
            if dec.kind == D_STATUS:
                self.demote(call.tool, ROLE_CART_VIEW, "returned a status reading")
                tb.role = ROLE_STATUS
            attributable = not ok
        elif role == ROLE_STATUS:
            ok = dec.kind == D_STATUS
            attributable = not ok
        if ok:
            tb.role_conf = 3
            for arg, slot in call.slots_used:
                if arg in tb.slots:
                    tb.slots[arg] = (tb.slots[arg][0], 3)
            if tb.first_use_ok is None:
                tb.first_use_ok = True
            self.ep.budget.known_good_shapes.add(call.shape)
        elif ok is False and attributable:
            tb.strikes += 1
            tb.role_conf = max(1, tb.role_conf - 1)
            if tb.first_use_ok is None:
                tb.first_use_ok = False
            if role in REPEATABLE_READ_ROLES:
                self.failed_calls.add(call.canonical())
            self.ep.trace.decision("bind_strike", tool=call.tool, bind_role=role, decoded=dec.kind,
                                   strikes=tb.strikes, conf=tb.role_conf)
            if tb.strikes >= 2:
                self.repair(role, f"{call.tool} struck twice")
        elif ok is False and dec.kind in (D_EMPTY, D_WORLD_FAULT, D_SESSION_ERROR) and role in REPEATABLE_READ_ROLES:
            self.failed_calls.add(call.canonical())

    def _settle_acks(self, dec: Decoded) -> None:
        """A cart read settles pending ACK-only writes: the item shown confirms the cart tool, an item missing
        strikes it (DESIGN 5.4: an ACK followed by a read that shows the item)."""
        if not self.pending_acks or not dec.cart:
            return
        shown = {line[0] for line in dec.cart.get("lines") or []}
        for item, tool in list(self.pending_acks.items()):
            tb = self.binding.tools.get(tool)
            del self.pending_acks[item]
            if tb is None or tb.first_use_ok is not None:
                continue
            if item in shown:
                tb.first_use_ok = True
                tb.role_conf = 3
            else:
                tb.first_use_ok = False
                tb.strikes += 1
                tb.role_conf = max(1, tb.role_conf - 1)
                self.ep.trace.decision("bind_strike", tool=tool, bind_role=ROLE_CART_WRITE, decoded="ACK_UNSEEN",
                                       strikes=tb.strikes, conf=tb.role_conf)

    def _filter_holds(self, intent: Intent, call: Call, dec: Decoded) -> bool:
        """Every row satisfies the filter: shelf equal, price <= ceiling, maker equal (DESIGN 5.4)."""
        p = intent.payload
        maker_role = FR_MAKER if FR_MAKER in self.ep.fields.fmap(call.tool).role_paths else FR_MERCHANT
        for rec in dec.records:
            f = self.ep.fields.extract(call.tool, rec)
            if p.get("ceiling") is not None and f.get(FR_PRICE) is not None and f[FR_PRICE] > p["ceiling"]:
                return False
            if p.get("shelf") and f.get(FR_SHELF) and canon_key(f[FR_SHELF]) != canon_key(p["shelf"]):
                return False
            if p.get("maker") and f.get(maker_role) and canon_key(f[maker_role]) != canon_key(p["maker"]):
                return False
        return True

    def repair(self, role: str, why: str, failing: Optional[str] = None) -> str:
        """The repair ladder, each rung at most once per role (DESIGN 5.4). The 'next_tool' rung moves the
        role off the failing tool (it is demoted for the role) when another fitting tool exists; an ORDER is
        never moved onto a tool any vote called decline."""
        used = self.repairs.setdefault(role, set())
        for rung in ("next_tool", "operator", "revote", "degrade"):
            if rung in used:
                continue
            used.add(rung)
            if rung == "next_tool":
                alt = self.tool_for(role, exclude=(failing,) if failing else ())
                if alt is None or alt == failing or (role == ROLE_ORDER and (
                        self.binding.tools[alt].decline_votes or self.binding.tools[alt].role == ROLE_DECLINE)):
                    continue
                if failing:
                    self.demote(failing, role, why)
                self.ep.trace.decision("bind_repair", bind_role=role, rung=rung, why=why, tool=alt, failing=failing)
                return rung
            if rung == "operator":
                continue                                # M8: Operator.map is consulted per intent
            if rung == "revote" and not self.revoted and self.ep.llm.allowed("BIND"):
                self.revoted = True
                if self.revote(why):
                    return rung
                continue
            if rung == "degrade":
                self.ep.trace.decision("bind_repair", bind_role=role, rung=rung, why=why)
                return rung
        return "exhausted"

    def revote(self, evidence: str) -> bool:
        """One BIND re-vote per episode with the observed signatures as evidence."""
        payload = {"tools": self.tool_digest(), "evidence": {
            "observed": {n: {"first_use_ok": tb.first_use_ok, "strikes": tb.strikes,
                             "provides": sorted(tb.provides_seen)} for n, tb in self.binding.tools.items()},
            "note": clip_text(evidence, 200)}}
        answer = self.ep.llm.ask("BIND", payload)
        vote = self.normalize_vote(answer)
        if not vote:
            return False
        self.build(self.votes + [vote])
        self.ep.trace.decision("bind", source="revote", table=self.table())
        return True

    def det_ok(self, intent: Intent, call: Optional[Call]) -> bool:
        """DET_OK (DESIGN 5.4) without the guard term, which the executor applies to every path."""
        if call is None:
            return False
        tb = self.binding.tools.get(call.tool)
        if tb is None or tb.strikes >= 2 or call.canonical() in self.failed_calls:
            return False
        if intent.kind in FINISH_INTENTS:
            return self.binding.finish_conf >= 2
        if tb.role_conf < 2:
            return False
        return all(tb.slots.get(arg, (None, 0))[1] >= 2 for arg, _ in call.slots_used)

    def fallback_ok(self, call: Optional[Call]) -> bool:
        """The best structural call may still be sent (path det_fallback) when DET_OK failed only on
        confidence: the tool is not struck out and this exact call has not failed before."""
        if call is None:
            return False
        tb = self.binding.tools.get(call.tool)
        return tb is not None and tb.strikes < 2 and call.canonical() not in self.failed_calls

    def table(self) -> dict:
        return {n: {"bind_role": tb.role, "conf": tb.role_conf,
                    "slots": {a: [s, c] for a, (s, c) in tb.slots.items()},
                    "enum": {a: {lit: [m, c] for lit, (m, c) in t.items()} for a, t in tb.enum_map.items()},
                    "provides": sorted(tb.provides_seen or tb.provides_expected)}
                for n, tb in self.binding.tools.items()}

class CallBuilder:
    def __init__(self, ep: Any):
        self.ep = ep

    @property
    def binder(self) -> Binder:
        return self.ep.binder

    def compile(self, intent: Intent) -> tuple:
        """(Call | None, reason)."""
        kind = intent.kind
        try:
            if kind == IK_RAW:
                tool = intent.payload.get("tool")
                tb = self.binder.binding.tools.get(tool)
                if tb is None:
                    return None, "unknown tool"
                return self._finish_call(Call(tool=tool, args=dict(intent.payload.get(ARGS_KEY) or {}),
                                              role=tb.role, path="op", intent_kind=kind)), ""
            builder = {IK_SEARCH: self._search, IK_FILTER: self._filter, IK_OPEN: self._open,
                       IK_STATUS: self._noargs, IK_CART_VIEW: self._noargs, IK_ASK: self._ask,
                       IK_CART_SET: self._cart_set, IK_ORDER: self._finish, IK_DECLINE: self._finish}.get(kind)
            if builder is None:
                return None, f"no builder for {kind}"
            return builder(intent)
        except Exception as exc:  # a builder bug must not crash the episode
            return None, f"builder error {exc!r}"[:200]

    def _finish_call(self, call: Call) -> Call:
        tf = self.binder.features.get(call.tool)
        lits = []
        if tf:
            for arg, value in sorted(call.args.items()):
                if arg in tf.args and tf.args[arg].is_enum:
                    lits.append(f"{arg}={value}")
        call.shape = call.tool + "(" + ",".join(sorted(call.args)) + ")" + ";".join(lits)
        return call

    def money_out(self, cents: Any) -> Any:
        """A money payload (integer cents) in the world's unit: major units with two decimals when the world
        prices in major units (FieldModel.price_major), else integer cents (F06)."""
        number = as_number(cents)
        if number is None:
            return cents
        if self.ep.fields.price_major:
            return round(number / 100.0, 2)
        return int(round(number))

    def coerce(self, tool: str, arg: str, value: Any) -> Any:
        feat = self.binder.features[tool].args.get(arg)
        if feat is None or value is None:
            return value
        if feat.type == "integer":
            number = as_number(value)
            if number is None:
                return value
            number = int(math.floor(number + 1e-9))
            if feat.minimum is not None:
                number = max(number, int(math.ceil(feat.minimum)))
            if feat.maximum is not None:
                number = min(number, int(math.floor(feat.maximum)))
            return number
        if feat.type == "numeric":
            number = as_number(value)
            return value if number is None else (int(number) if float(number).is_integer() else number)
        if feat.type == "string" and not feat.is_enum:
            text = str(value)
            if feat.max_len is not None and len(text) > feat.max_len:
                text = trim_at_token(text, feat.max_len)
            return text
        if feat.type == "array":
            items = list(value) if isinstance(value, (list, tuple)) else [value]
            if feat.max_items is not None:
                items = items[: feat.max_items]
            return items
        return value

    def _required_filled(self, tool: str, args: dict) -> bool:
        return all(r in args for r in self.binder.features[tool].required)

    def _put(self, tool: str, args: dict, used: list, slot: str, value: Any) -> bool:
        arg = self.binder.slot_arg(tool, slot)
        if arg is None:
            return False
        args[arg] = self.coerce(tool, arg, value)
        used.append((arg, slot))
        return True

    def _mode(self, tool: str, args: dict, used: list, meanings: tuple) -> None:
        arg = self.binder.slot_arg(tool, SLOT_MODE)
        if arg is None:
            return
        for meaning in meanings:
            lit = self.binder.literal_for(tool, arg, meaning)
            if lit is not None:
                args[arg] = lit
                used.append((arg, SLOT_MODE))
                return

    def _paging(self, tool: str, args: dict, used: list, cont: Any) -> None:
        if cont is None:
            return
        for slot in (SLOT_TOKEN, SLOT_PAGE, SLOT_OFFSET):
            arg = self.binder.slot_arg(tool, slot)
            if arg is None:
                continue
            feat = self.binder.features[tool].args[arg]
            if slot == SLOT_TOKEN and isinstance(cont, str) and not looks_numeric(cont):
                args[arg] = cont                              # verbatim (G8)
            elif slot != SLOT_TOKEN and feat.type == "integer" and looks_numeric(cont):
                args[arg] = int(as_number(cont))
            elif slot == SLOT_TOKEN and feat.type == "string":
                args[arg] = str(cont)
            else:
                continue
            used.append((arg, slot))
            return

    def _search(self, intent: Intent) -> tuple:
        tool = self.binder.tool_for(ROLE_SEARCH)
        if tool is None:
            return None, "no search tool"
        args, used = {}, []
        query = clamp_query(intent.payload.get("query", ""), self.ep.contract.token_cap)
        if not self._put(tool, args, used, SLOT_QUERY, query):
            return None, "search tool has no query slot"
        self._mode(tool, args, used, (SEM_TEXT,))
        self._paging(tool, args, used, intent.payload.get("cont"))
        if not self._required_filled(tool, args):
            return None, "required argument without a slot value"
        return self._finish_call(Call(tool=tool, args=args, role=ROLE_SEARCH, slots_used=used,
                                      intent_kind=intent.kind)), ""

    def _filter(self, intent: Intent) -> tuple:
        tool = self.binder.tool_for(ROLE_FILTER) or self.binder.tool_for(ROLE_SEARCH)
        if tool is None:
            return None, "no filter tool"
        p = intent.payload
        args, used = {}, []
        if p.get("shelf"):
            if not self._put(tool, args, used, SLOT_SHELF, p["shelf"]):
                return None, "no shelf slot"
        if p.get("maker"):
            if not self._put(tool, args, used, SLOT_MAKER, p["maker"]):
                return None, "no maker slot"
        if p.get("ceiling") is not None:
            if not self._put(tool, args, used, SLOT_CEILING, self.money_out(p["ceiling"])):
                return None, "no ceiling slot"
        meanings = (SEM_FILTER_SHELF, SEM_FILTER_ANY) if p.get("shelf") else (SEM_FILTER_MAKER, SEM_FILTER_ANY)
        self._mode(tool, args, used, meanings)
        self._paging(tool, args, used, p.get("cont"))
        if not used or not self._required_filled(tool, args):
            return None, "filter call incomplete"
        return self._finish_call(Call(tool=tool, args=args, role=ROLE_FILTER, slots_used=used,
                                      intent_kind=intent.kind)), ""

    def _open(self, intent: Intent) -> tuple:
        ids = [i for i in intent.payload.get("ids") or [] if isinstance(i, str)]
        if not ids:
            return None, "no ids"
        tool, chunk = self.binder.route_open(ids, intent.payload.get("need", NEED_FULL))
        if tool is None:
            return None, "no record tool"
        args, used = {}, []
        role = self.binder.binding.tools[tool].role
        if role == ROLE_BATCH:
            if not self._put(tool, args, used, SLOT_IDS, chunk):
                return None, "batch tool has no ids slot"
        elif not self._put(tool, args, used, SLOT_ID, chunk[0]):
            return None, "view tool has no id slot"
        if not self._required_filled(tool, args):
            return None, "required argument without a slot value"
        return self._finish_call(Call(tool=tool, args=args, role=role, slots_used=used, intent_kind=intent.kind,
                                      ids=list(chunk))), ""

    def _noargs(self, intent: Intent) -> tuple:
        role = ROLE_CART_VIEW if intent.kind == IK_CART_VIEW else ROLE_STATUS
        tool = self.binder.tool_for(role)
        if tool is None:
            return None, f"no {role} tool"
        if self.binder.features[tool].required:
            return None, "tool needs arguments"
        return self._finish_call(Call(tool=tool, args={}, role=role, intent_kind=intent.kind)), ""

    def _ask(self, intent: Intent) -> tuple:
        tool = self.binder.tool_for(ROLE_MESSAGE)
        if tool is None:
            return None, "no message tool"
        args, used = {}, []
        if not self._put(tool, args, used, SLOT_TEXT, intent.payload.get("text", "")):
            return None, "message tool has no text slot"
        if not self._required_filled(tool, args):
            return None, "structured message tool needs the operator"
        return self._finish_call(Call(tool=tool, args=args, role=ROLE_MESSAGE, slots_used=used,
                                      intent_kind=intent.kind)), ""

    def _cart_set(self, intent: Intent) -> tuple:
        tool = self.binder.tool_for(ROLE_CART_WRITE)
        if tool is None:
            return None, "no cart tool"
        item = intent.payload.get("item")
        target = int(intent.payload.get("quantity", 0))
        cart = self.ep.cart
        current = int(cart.expected.get(item, 0))
        args, used = {}, []
        items_arg = self.binder.slot_arg(tool, SLOT_ITEMS)
        if items_arg is not None:                      # item-list cart: rebuilt in full from expected
            feat = self.binder.features[tool].args[items_arg]
            want = dict(cart.expected)
            if target > 0:
                want[item] = target
            else:
                want.pop(item, None)
            id_key, qty_key = item_list_keys(feat)
            if id_key is None:
                return None, "item-list cart needs the operator"
            lines = [{id_key: i, qty_key: q} if qty_key else {id_key: i} for i, q in want.items()]
            args[items_arg] = lines
            used.append((items_arg, SLOT_ITEMS))
            call = Call(tool=tool, args=args, role=ROLE_CART_WRITE, slots_used=used, intent_kind=intent.kind,
                        ids=[item])
            call.write = (item, target)
            return self._finish_call(call), ""
        sem = cart.semantics
        mode_arg = self.binder.slot_arg(tool, SLOT_CART_MODE)
        qty_arg = self.binder.slot_arg(tool, SLOT_QTY)
        qty_feat = self.binder.features[tool].args.get(qty_arg) if qty_arg else None
        if not self._put(tool, args, used, SLOT_ID, item):
            return None, "cart tool has no id slot"
        if sem == "add":
            delta = target - current
            if delta == 0:
                return None, "noop"
            meaning = SEM_ADD if delta > 0 else SEM_REMOVE
            lit = self.binder.literal_for(tool, mode_arg, meaning) if mode_arg else None
            if delta < 0 and lit is None:
                return None, "cannot remove without a remove literal"   # never an add in place of a removal (F09)
            if mode_arg and lit is None:
                return None, f"no literal for {meaning}"
            if mode_arg:
                args[mode_arg] = lit
                used.append((mode_arg, SLOT_CART_MODE))
            step = abs(delta)
            limit = cart.add_max or (int(qty_feat.maximum) if qty_feat and qty_feat.maximum else None)
            if limit:
                step = min(step, limit)
            if qty_arg:
                args[qty_arg] = self.coerce(tool, qty_arg, step)
                used.append((qty_arg, SLOT_QTY))
            elif step != 1:
                pass                                   # no quantity argument: one unit per call
            new_qty = current + (step if delta > 0 else -step)
        else:
            if target == 0 and qty_feat is not None and qty_feat.minimum is not None and qty_feat.minimum > 0:
                lit = self.binder.literal_for(tool, mode_arg, SEM_REMOVE) if mode_arg else None
                if lit is None:
                    return None, "cannot remove: quantity minimum above zero"
                args[mode_arg] = lit
                used.append((mode_arg, SLOT_CART_MODE))
                if qty_arg:
                    args[qty_arg] = self.coerce(tool, qty_arg, max(1, current))
                    used.append((qty_arg, SLOT_QTY))
            else:
                if mode_arg:
                    lit = self.binder.literal_for(tool, mode_arg, SEM_SET) or self.binder.literal_for(
                        tool, mode_arg, SEM_ADD)
                    if lit is not None:
                        args[mode_arg] = lit
                        used.append((mode_arg, SLOT_CART_MODE))
                if qty_arg:
                    args[qty_arg] = self.coerce(tool, qty_arg, target)
                    used.append((qty_arg, SLOT_QTY))
            new_qty = target
        if not self._required_filled(tool, args):
            return None, "required argument without a slot value"
        call = Call(tool=tool, args=args, role=ROLE_CART_WRITE, slots_used=used, intent_kind=intent.kind, ids=[item])
        call.write = (item, max(0, new_qty))
        return self._finish_call(call), ""

    def _finish(self, intent: Intent) -> tuple:
        """Finish intents never go to the operator: the best-scored binding is used and logged."""
        bnd = self.binder.binding
        is_order = intent.kind == IK_ORDER
        if bnd.finish_kind == "enum" and bnd.finish_tool and bnd.finish_arg:
            lit = bnd.order_literal if is_order else bnd.decline_literal
            if lit is None:
                return None, "no finish literal"
            args = {bnd.finish_arg: lit}
            remark = intent.payload.get("remark")
            remark_arg = self.binder.slot_arg(bnd.finish_tool, SLOT_REMARK)
            if not is_order and remark and remark_arg and remark_arg != bnd.finish_arg:
                args[remark_arg] = self.coerce(bnd.finish_tool, remark_arg, remark)   # the prover decides (F23)
            call = Call(tool=bnd.finish_tool, args=args, role=ROLE_FINISH,
                        slots_used=[(bnd.finish_arg, SLOT_FINISH_MODE)], intent_kind=intent.kind)
            return self._finish_call(call), ""
        role = ROLE_ORDER if is_order else ROLE_DECLINE
        tool = self.binder.tool_for(role)
        if is_order and tool is not None and self.binder.binding.tools[tool].decline_votes:
            alt = next((n for n in bnd.by_role.get(ROLE_ORDER, []) if n != tool
                        and not self.binder.binding.tools[n].decline_votes
                        and n not in bnd.demoted.get(ROLE_ORDER, set())), None)
            tool = alt or tool                     # never order through a tool any vote called decline
        if tool is None:
            return None, f"no {role} tool"
        args = {}
        tf = self.binder.features[tool]
        remark = intent.payload.get("remark")
        for arg, feat in tf.args.items():
            if feat.required:
                if feat.is_string and remark:
                    args[arg] = self.coerce(tool, arg, remark)
                else:
                    return None, "finish tool needs an unknown required argument"
        if remark and not args:
            remark_arg = next((a for a, f in tf.args.items() if f.free_text), None)
            if remark_arg is not None:
                args[remark_arg] = self.coerce(tool, remark_arg, remark)     # optional remark slot (F23)
        return self._finish_call(Call(tool=tool, args=args, role=role, intent_kind=intent.kind)), ""

def trim_at_token(text: str, max_len: int) -> str:
    """Trim free text to max_len at a token boundary (G1)."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut).strip()

def clamp_query(query: Any, token_cap: int) -> str:
    """At most token_cap whitespace tokens (the world's AND-matching token cap, DESIGN 5.7)."""
    words = str(query or "").split()
    return " ".join(words[: max(1, token_cap)])

def item_list_keys(feat: ArgFeat) -> tuple:
    """(id key, quantity key) inside an item-list cart's object items, by value type."""
    props = feat.items_props or {}
    id_key = next((k for k, s in props.items() if schema_type(s) == "string"), None)
    qty_key = next((k for k, s in props.items() if schema_type(s) == "integer"), None)
    return id_key, qty_key

G_SCHEMA, G_PHASE, G_STEPS, G_LAST, G_ENDGAME, G_REPEAT, G_IDS, G_CONT = ("G1", "G2", "G3", "G4", "G5", "G6", "G7",
                                                                         "G8")
G_HYGIENE, G_FILTER, G_MSG_BUDGET, G_QUESTION, G_MSG_DUP, G_SHAPE, G_FINISH_PRE, G_BATCH = (
    "G9", "G10", "G11", "G12", "G13", "G14", "G15", "G16")
G_TIME = "TIME"
_QZ_G12S = "G12S"           # e_q: the question's key is already stated or not an attribute
DECLINE_BASES = ("proof:E1", "proof:E2", "proof:E3", "proof:E1x", "proof:OVERCAP", "proof:NOFIT",
                 "fallback:proof", "fallback:impossible", "fallback:last_resort")

def is_finish_intent(intent: Intent) -> bool:
    """ORDER, DECLINE, or the operator's completion of a two-step finish (F25)."""
    return intent.kind in FINISH_INTENTS or (intent.kind == IK_RAW and bool(intent.payload.get("finish_completion")))

class Guard:
    def __init__(self, ep: Any):
        self.ep = ep
        self.sent_messages: list = []

    def admit(self, call: Optional[Call], intent: Intent) -> Admission:
        if call is None:
            return Admission(False, None, G_SCHEMA, "no call")
        repaired: list = []
        for check in (self._g3_steps, self._time_gate, self._g4_last, self._g2_phase, self._g15_finish,
                      self._g5_endgame, self._g16_batch, self._g1_schema, self._g14_shape, self._g6_repeat,
                      self._g7_ids, self._g8_cont, self._g9_hygiene, self._g10_filter, self._g11_messages,
                      self._g12_question, self._g12s_stated, self._g13_duplicate):
            verdict = check(call, intent)
            if verdict is None:
                continue
            gid, ok, reason = verdict
            if ok:
                repaired.append(gid)
                continue
            return Admission(False, call, gid, reason, repaired)
        return Admission(True, call, None, "", repaired)

    def _rem(self) -> int:
        return self.ep.budget.B - self.ep.transport.n

    def _g3_steps(self, call: Call, intent: Intent):
        if self.ep.transport.n >= self.ep.budget.B:
            return G_STEPS, False, f"n={self.ep.transport.n} >= B={self.ep.budget.B}"
        return None

    def _time_gate(self, call: Call, intent: Intent):
        t = self.ep.clock()
        if is_finish_intent(intent):
            if t >= T_LAST_FINISH_SEND:
                return G_TIME, False, f"t={t:.1f} past the finish window"
            return None
        if t >= T_LAST_ENV_SEND:
            return G_TIME, False, f"t={t:.1f}: only finishes after {T_LAST_ENV_SEND:.0f} s"
        return None

    def _g4_last(self, call: Call, intent: Intent):
        if self._rem() <= 1 and not is_finish_intent(intent):
            return G_LAST, False, "the last step must be a finish"
        return None

    def _g2_phase(self, call: Call, intent: Intent):
        tb = self.ep.binder.binding.tools.get(call.tool)
        if tb is None:
            return G_PHASE, False, f"unknown tool {call.tool}"
        finish_tool = tb.role in FINISH_ROLES or call.tool == self.ep.binder.binding.finish_tool
        if intent.kind in FINISH_INTENTS:
            if intent.purpose not in (P_FINISH, P_RECOVER):
                return G_PHASE, False, "finishes come from the Finisher only"
            return None
        if finish_tool and not (intent.kind == IK_RAW and intent.payload.get("finish_completion")):
            return G_PHASE, False, f"{call.tool} is a finish tool; a {intent.kind} intent may not use it"
        if tb.role == ROLE_MESSAGE or intent.kind == IK_ASK:
            if not (self.ep.phase == "ELICIT" or intent.payload.get("procedure")):
                return G_PHASE, False, "messages only in ELICIT or for a procedure"
        return None

    def _g15_finish(self, call: Call, intent: Intent):
        if intent.kind == IK_ORDER and not self.ep.cart.expected:
            return G_FINISH_PRE, False, "ORDER needs a non-empty expected cart"
        if intent.kind == IK_DECLINE:
            basis = intent.payload.get("basis")
            if basis not in DECLINE_BASES:
                return G_FINISH_PRE, False, f"DECLINE needs Q8 (basis {basis!r})"
            if self.ep.events.decline_locked and basis != "fallback:last_resort":
                return G_FINISH_PRE, False, "declines are locked after a refused decline"
            if self.ep.events.refused_last:
                return G_FINISH_PRE, False, "never decline right after a refusal"
        return None

    def _g5_endgame(self, call: Call, intent: Intent):
        plan = self.ep.plan
        kind = intent.kind
        if is_finish_intent(intent) or kind in (IK_CART_SET, IK_CART_VIEW):
            return None                                 # a two-step finish completion is a finish (M8)
        first_finish = self.ep.budget.first_finish_step is not None
        endgame = (not first_finish) and self._rem() <= plan.R + plan.need_finish()
        if first_finish:
            if kind == IK_OPEN:
                return None
            if kind in (IK_SEARCH, IK_FILTER) and intent.purpose in (P_RECOVER, P_COVER) \
                    and plan.recover_search_used < max(0, self._rem() - 2):
                return None
            return G_ENDGAME, False, f"after the first finish only recovery actions ({kind})"
        if endgame:
            if kind == IK_OPEN:
                ids = set(intent.payload.get("ids") or [])
                cart_ids = set(self.ep.cart.expected)
                if ids <= cart_ids or intent.payload.get("replacement") or intent.payload.get("chain_page") \
                        or intent.payload.get("procedure"):
                    return None
            return G_ENDGAME, False, f"endgame allowlist excludes {kind}"
        return None

    def _g14_shape(self, call: Call, intent: Intent):
        streak = self.ep.budget.contract_error_streak
        if streak <= 0 or is_finish_intent(intent):
            return None
        if call.shape in self.ep.budget.known_good_shapes:
            return None
        if streak == 1 and call.path == "det" and self.ep.binder.binding.tools[call.tool].first_use_ok:
            return None
        return G_SHAPE, False, "after a contract error only known-good call shapes"

    def _g6_repeat(self, call: Call, intent: Intent):
        if intent.kind in FINISH_INTENTS or intent.kind in (IK_CART_SET, IK_CART_VIEW, IK_STATUS, IK_ASK):
            return None                                 # state-dependent calls may be repeated
        if call.canonical() in self.ep.binder.failed_calls:
            return G_REPEAT, False, "identical call already returned an error, {} or not-found"
        return None

    def _g7_ids(self, call: Call, intent: Intent):
        for ident in call.ids:
            if ident not in self.ep.catalog:
                return G_IDS, False, f"id {ident!r} was never observed"
            if ident in self.ep.absent_ids:
                return G_IDS, False, f"id {ident!r} was reported absent by a batch view"
            if not self.ep.contract_learner.id_matches(ident):
                return G_IDS, False, f"id {ident!r} does not match the learned signature"
        return None

    def _g8_cont(self, call: Call, intent: Intent):
        cont = intent.payload.get("cont")
        if intent.kind not in (IK_SEARCH, IK_FILTER) or cont is None:
            return None
        for arg, slot in call.slots_used:
            if slot == SLOT_TOKEN:
                qrec = self.ep.queries.get(intent.payload.get("qkey") or "")
                if qrec is None or call.args.get(arg) != qrec.next_cont:
                    return G_CONT, False, "continuation tokens are passed verbatim from the same query"
            if slot in (SLOT_PAGE, SLOT_OFFSET):
                feat = self.ep.binder.features[call.tool].args.get(arg)
                if feat is None or feat.type != "integer":
                    return G_CONT, False, "page arithmetic only on integer arguments"
        return None

    def _g16_batch(self, call: Call, intent: Intent):
        arg = next((a for a, s in call.slots_used if s == SLOT_IDS), None)
        if arg is None:
            return None
        limit = self.ep.contract.batch_limit or DEFAULT_BATCH_LIMIT
        ids = call.args.get(arg) or []
        if len(ids) > limit:
            call.args[arg] = list(ids)[:limit]
            call.ids = list(call.args[arg])
            return G_BATCH, True, f"batch split at {limit}"
        return None

    def _g1_schema(self, call: Call, intent: Intent):
        tf = self.ep.binder.features.get(call.tool)
        if tf is None:
            return G_SCHEMA, False, "tool not in the schema list"
        repaired = False
        for arg in list(call.args):
            if arg not in tf.args:
                del call.args[arg]
                repaired = True
        for arg in tf.required:
            if arg not in call.args:
                return G_SCHEMA, False, f"required argument {arg} missing"
        slots = dict(call.slots_used)
        for arg, value in list(call.args.items()):
            feat = tf.args[arg]
            fixed, ok, why = check_value(feat, value, slots.get(arg))
            if not ok:
                return G_SCHEMA, False, f"{arg}: {why}"
            if fixed is not value and fixed != value:
                call.args[arg] = fixed
                repaired = True
        if repaired:
            self.ep.builder._finish_call(call)
            return G_SCHEMA, True, "schema repair"
        return None

    def _g9_hygiene(self, call: Call, intent: Intent):
        if intent.kind != IK_SEARCH:
            return None
        ok, why = self.ep.coverage.hygiene_ok(self.ep, intent.payload.get("query", ""))
        return None if ok else (G_HYGIENE, False, why)

    def _g10_filter(self, call: Call, intent: Intent):
        tool = call.tool
        if call.role not in (ROLE_SEARCH, ROLE_FILTER):
            return None
        mode_arg = self.ep.binder.slot_arg(tool, SLOT_MODE)
        mode = self.ep.binder.binding.tools[tool].enum_map.get(mode_arg, {}).get(call.args.get(mode_arg), (None, 0))[0] \
            if mode_arg else None
        used = {s for _, s in call.slots_used}
        filters = used & {SLOT_SHELF, SLOT_MAKER, SLOT_CEILING}
        if intent.kind == IK_FILTER:
            if not filters:
                return G_FILTER, False, "filter mode needs at least one filter argument"
            if SLOT_QUERY in used:
                return G_FILTER, False, "filter calls carry no keywords"
            if intent.payload.get("merchant_value") and SLOT_MAKER in used:
                return G_FILTER, False, "a merchant never goes into a maker filter"
            shelf = intent.payload.get("shelf")
            if shelf and not self._shelf_known(tool, shelf):
                return G_FILTER, False, "a shelf value must be an observed label or an enum literal"
        elif intent.kind == IK_SEARCH:
            if filters and self.ep.contract.text_honours_filters is not True:
                return G_FILTER, False, "text mode carries no filters"
            if mode in (SEM_FILTER_SHELF, SEM_FILTER_MAKER, SEM_FILTER_ANY):
                return G_FILTER, False, "text search sent in a filter mode"
        return None

    def _shelf_known(self, tool: str, shelf: str) -> bool:
        want = canon_key(shelf)
        if any(canon_key(r.shelf) == want for r in self.ep.catalog.values() if r.shelf):
            return True
        arg = self.ep.binder.slot_arg(tool, SLOT_SHELF)
        feat = self.ep.binder.features[tool].args.get(arg) if arg else None
        return bool(feat and feat.enum and any(canon_key(e) == want for e in feat.enum))

    def _g11_messages(self, call: Call, intent: Intent):
        if intent.kind != IK_ASK:
            return None
        b = self.ep.budget
        if b.msgs_used >= b.M and not b.M_probe_open:
            return G_MSG_BUDGET, False, f"message budget spent ({b.msgs_used}/{b.M})"
        if self.ep.cart.writes > 0 and not intent.payload.get("procedure"):
            return G_MSG_BUDGET, False, "no questions after the first cart write"
        return None

    def _g12_question(self, call: Call, intent: Intent):
        if intent.kind != IK_ASK or intent.payload.get("procedure"):
            return None
        ok, why = self.ep.elicitor.question_ok(self.ep, intent)
        return None if ok else (G_QUESTION, False, why)

    def _g12s_stated(self, call: Call, intent: Intent):
        """G12S (_qz_stated). A passing question is sent with key separators spaced; the intent text is kept."""
        if intent.kind != IK_ASK or intent.payload.get("procedure"):
            return None
        try:
            why = _qz_stated(self.ep, intent)
        except Exception:                       # a safety net never ends the episode
            why = ""
        arg = next((a for a, s in call.slots_used if s == SLOT_TEXT), None)
        text = call.args.get(arg) if arg else None
        if why or not isinstance(text, str) or _QZ_SEP_RE.sub(" ", text) == text:
            return (_QZ_G12S, False, why) if why else None
        call.args[arg] = _QZ_SEP_RE.sub(" ", text)
        return _QZ_G12S, True, "key separators spaced"

    def _g13_duplicate(self, call: Call, intent: Intent):
        if intent.kind != IK_ASK:
            return None
        text = norm_space(intent.payload.get("text", ""))
        if text in self.sent_messages:
            return G_MSG_DUP, False, "duplicate message text"
        return None

    def note_sent(self, intent: Intent) -> None:
        if intent.kind == IK_ASK:
            self.sent_messages.append(norm_space(intent.payload.get("text", "")))

def check_value(feat: ArgFeat, value: Any, slot: Optional[str]) -> tuple:
    """(value, ok, why): type checks with the safe repairs G1 allows."""
    if feat.is_enum:
        literals = [e for e in feat.enum]
        if value in literals:
            return value, True, ""
        match = [e for e in literals if isinstance(e, str) and isinstance(value, str) and e.lower() == value.lower()]
        if len(match) == 1:
            return match[0], True, ""
        return value, False, "not an allowed literal"
    if feat.type == "string":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str):
            return value, False, "expected a string"
        if feat.max_len is not None and len(value) > feat.max_len:
            value = trim_at_token(value, feat.max_len)
        return value, True, ""
    if feat.type == "integer":
        number = as_number(value)
        if number is None or abs(number - round(number)) > 1e-9:
            return value, False, "expected an integer"
        number = int(round(number))
        if feat.minimum is not None and number < feat.minimum or feat.maximum is not None and number > feat.maximum:
            if slot in (SLOT_PAGE, SLOT_OFFSET, SLOT_CEILING):
                number = int(clamp(number, feat.minimum if feat.minimum is not None else number,
                                   feat.maximum if feat.maximum is not None else number))
            else:
                return value, False, f"{number} outside [{feat.minimum}, {feat.maximum}]"
        return number, True, ""
    if feat.type == "numeric":
        number = as_number(value)
        if number is None:
            return value, False, "expected a number"
        if feat.minimum is not None and number < feat.minimum:
            return value, False, "below the minimum"
        if feat.maximum is not None and number > feat.maximum:
            return value, False, "above the maximum"
        return (int(number) if float(number).is_integer() else number), True, ""
    if feat.type == "boolean":
        return value, isinstance(value, bool), "expected a boolean"
    if feat.type == "array":
        if not isinstance(value, list):
            return value, False, "expected a list"
        return value, True, ""
    return value, True, ""

_QZ_SEP_RE = re.compile(r"(?<=[A-Za-z])[._](?=[A-Za-z])")
_QZ_END_RE = re.compile(r"[.!?]+(?=\s|$)")
_QZ_ROLES = (FR_PRICE, FR_STOCK, FR_ID, FR_TITLE, FR_MERCHANT, FR_MAKER, FR_SHELF, FR_OVERVIEW, FR_NOTES)

def _qz_stem(t: str) -> str:
    """Crude plural strip ('styles', 'watches'); never an 'ss', 'us' or 'is' ending."""
    if len(t) > 4 and t.endswith(("ches", "shes", "sses", "xes")):
        return t[:-2]
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")) else t

def _qz_toks(text: Any) -> list:
    return [_qz_stem(t) for t in text_tokens(text)]

def _qz_sentences(text: str, spans: bool = False) -> list:
    ends = [m.end() for m in _QZ_END_RE.finditer(text)]
    out = [(a, b) for a, b in zip([0] + ends, ends + [len(text)]) if text[a:b].strip()]
    return out if spans else [text[a:b] for a, b in out]

def _qz_line_words(line: Line, naming: bool = True) -> set:
    src = line.shelf_label if line.guessed and line.shelf_label and not naming else \
        f"{line.shelf_label or ''} {line.shelf_words} {line.words}"
    return set(_qz_toks(src)) - STOPWORDS

def _qz_line_text(ep: Any, line: Line) -> list:
    led, text = ep.ledger, str(ep.query or "")
    marks = {n for n in (ep.max_steps, led.stated_steps, led.stated_msgs) if type(n) is int and n > 0}
    routed = [sp for sp, route in led.sentences if sp and route in ("requirement", "restatement")]
    blanks = [norm_space(x) for x in [led.compat.get("span")] + [y for r in led.refs for y in [r.span] + [
        d[1] for d in r.source.get("descriptors") or [] if len(d) == 2]] if len(str(x or "").strip()) > 2]
    words = {ln.idx: _qz_line_words(ln) for ln in led.lines}
    out = []
    if _qz_parsed(ep):          # PARSE alive: only the line's own validated requirement spans
        for c in line.constraints:
            low = norm_space(c.span) if c.source == "stated" and c.active and c.span else ""
            for b in blanks:
                low = low.replace(b, " ")
            if _qz_toks(low):
                out.append(_qz_seg(low))
        return out
    for sent in _qz_sentences(text):
        low = norm_space(sent)
        if any(v in marks or (led.cap_c is not None and cap_to_cents(v, raw) == led.cap_c)
               for v, raw, _a, _b in numbers_in(sent)) or (led.sentences and not any(
                   sp in low or norm_span(sent) in sp for sp in routed)):
            continue
        for b in blanks:
            low = low.replace(b, " ")
        if not led.sentences:      # no routes: a named listing's descriptors follow its name; cut at the first
            w = sent.split()       # capitalised word after a lowercase one
            low = norm_space(" ".join(w[:next((i for i in range(1, len(w)) if w[i][:1].isupper() and w[i][1:2].islower()
                                               and w[i - 1][:1].islower()), len(w))]))
        toks, owner = _qz_seg(low), None
        if len(led.lines) > 1:
            for i, t in enumerate(toks):
                hits = [k for k, ws in words.items() if t in ws]
                owner = (hits[0] if len(hits) == 1 else None) if hits else owner
                toks[i] = t if owner == line.idx else ""
        elif not (led.sentences or line.guessed or set(toks) & words[line.idx]):
            continue
        if any(toks):
            out.append(toks)
    return out

def _qz_parsed(ep: Any) -> bool:
    """PARSE is alive: a ledger from the votes (or arbiter), or with sentence routes."""
    return bool(ep.ledger.sentences) or ep.ledger.source not in ("fallback", "none")

def _qz_seg(text: str) -> list:
    """Tokens with "" where ';', ',', '(' or ')' end a phrase."""
    return [t for piece in re.split(r"[;,()]", text) for t in _qz_toks(piece) + [""]][:-1]

def _qz_values(ep: Any, line: Line, ks: KeyStats) -> set:
    out = set()
    for rid in dict.fromkeys(list(line.opened) + list(line.table)):
        rec = ep.catalog.get(rid)
        for canon in (ks.variants or [ks.canon]) if rec else []:
            for _raw, value in rec.attrs.get(canon) or []:
                out |= {v for v in (tuple(_qz_toks(p)) for p in re.split(r"[,;|]", str(value))) if 0 < len(v) < 5}
    return out

def _qz_stated(ep: Any, intent: Intent) -> str:
    """Why an ASK must not go out (G12S): (iii) an unseen key (no opened listing carries it) names a non-attribute
    field's learned path; (i) for the key or a twin, every key word that is not a line word occurs in the line's
    text (not as the tail of another key's or constraint attribute's phrase, as 'type' in 'warranty type'; never
    when those words are another key of the line); (ii) in the fallback only, a value only that family holds occurs
    there as a whole phrase that no longer value of another family overlaps."""
    lines = ep.ledger.lines
    line = lines[intent.line] if isinstance(intent.line, int) and 0 <= intent.line < len(lines) else None
    key = intent.payload.get("key")
    ks = line.menu.get(key) if line else None
    if ks is None:
        return ""
    forms = {tuple(_qz_toks(n)) for n in [key, ks.spoken] + list(ks.variants) if n} if ks.unseen else set()
    for fm in list(ep.fields.maps.values()) if forms else []:
        for role in _QZ_ROLES:
            path = fm.role_paths.get(role)
            if path and isinstance(path[-1], str) and tuple(_qz_toks(path[-1])) in forms:
                return f"'{key}' names the {role} field, not an attribute"
    texts = _qz_line_text(ep, line)
    family = [key] + sorted(k for k in ks.twins if k in line.menu)
    fam = [set(_qz_toks(k)) - STOPWORDS for k in family]    # v5: no tail from a shorter family name
    tails = {tuple(t[-2:]) for ln in lines for k in [*ln.menu, *(c.attr or "" for c in ln.constraints if not (
                 ln is line and any(set(_qz_toks(c.attr or "")) - STOPWORDS < f for f in fam)))]
             if k not in family for t in [[x for x in _qz_toks(k) if x not in STOPWORDS]] if len(t) > 1}
    lw, names = _qz_line_words(line, False), _qz_line_words(line)
    bare = {tuple(t for t in _qz_toks(m) if t not in STOPWORDS) for m in line.menu if m not in family}
    for k in family if texts else []:
        own = [t for t in _qz_toks(k) if t not in STOPWORDS]
        need = [t for t in dict.fromkeys(own) if t not in lw]
        if need and (len(need) == len(set(own)) or tuple(need) not in bare) and all(any(
                tok == w and not (p and toks[p - 1] not in own and (toks[p - 1], w) in tails)
                for toks in texts for p, tok in enumerate(toks)) for w in need):
            return f"stated: '{' '.join(need)}' in the request" + (f" (twin '{k}')" if k != key else "")
        mine = set() if line.menu[k].unseen or _qz_parsed(ep) else _qz_values(ep, line, line.menu[k])
        longer = set()
        for k2, ks2 in line.menu.items():
            if mine and k2 != k and k2 not in family:
                other = _qz_values(ep, line, ks2)
                mine -= other
                longer |= {v for v in other if len(v) > 1}
        for val in sorted(mine, key=lambda v: (-len(v), v)):   # one word: 4+ letters, a sentence naming line/key
            if any(len(t) > 2 and not t.isdigit() and t not in STOPWORDS and t not in lw for t in val) and any(
                    tuple(toks[i:i + len(val)]) == val and (len(val) > 1 or len(val[0]) > 3 and set(toks) & (
                        names | set(own))) and not any(
                        tuple(toks[j:j + len(w)]) == w
                        for w in longer if len(w) > len(val) for j in range(max(0, i - len(w) + 1), i + len(val)))
                    for toks in texts for i in range(len(toks))):
                return f"stated: value '{' '.join(val)}' in the request" + (f" (twin '{k}')" if k != key else "")
    return ""

HARMLESS_INTENTS = (IK_SEARCH, IK_FILTER, IK_OPEN, IK_CART_VIEW, IK_STATUS)

class Executor:
    def __init__(self, ep: Any):
        self.ep = ep
        self.last_qkey: Optional[str] = None
        self.operator_calls = 0

    def run(self, intent: Intent) -> Decoded:
        ep = self.ep
        call, why = ep.builder.compile(intent)
        path = "det"
        if intent.kind not in FINISH_INTENTS and intent.kind != IK_RAW:
            if call is None or not ep.binder.det_ok(intent, call):
                op_call = ep.operator.map(ep, intent, call)
                if op_call is not None:
                    call, path = op_call, "op"
                    self.operator_calls += 1
                elif call is not None and intent.kind in HARMLESS_INTENTS:
                    path = "det_fallback"      # the operator is unavailable: try the best structural call
                elif call is not None and intent.kind == IK_CART_SET and ep.binder.fallback_ok(call) and (
                        intent.purpose in (P_COMMIT, P_RECOVER) or ep.phase == "FALLBACK"):
                    path = "det_fallback"      # cart writes like finishes: the best structural call
                else:
                    reason = why or "DET_OK failed and the operator gave no mapping"
                    ep.trace.guard("DET", reason, intent.kind, call.tool if call else "")
                    return Decoded.blocked("DET", reason)
        elif intent.kind in FINISH_INTENTS and call is not None and ep.binder.binding.finish_conf < 2:
            ep.trace.decision("finish_uncertain", intent=intent.kind, tool=call.tool, args=call.args,
                              finish_kind=ep.binder.binding.finish_kind)
        if call is None:
            ep.trace.guard("BUILD", why or "no call", intent.kind, "")
            return Decoded.blocked("BUILD", why)
        call.path = "op" if intent.kind == IK_RAW else path
        adm = ep.guard.admit(call, intent)
        if adm.blocked:
            ep.trace.guard(adm.gid, adm.reason, intent.kind, call.tool)
            return Decoded.blocked(adm.gid, adm.reason)
        call = adm.call
        finish = is_finish_intent(intent)
        if ep.phase == "FALLBACK":
            deadline = T_LAST_FINISH_SEND if finish else T_LAST_ENV_SEND
        else:
            deadline = T_FALLBACK
        sent_at = monotonic()
        try:
            raw = ep.transport.send(call.tool, call.args, deadline=deadline, finish=finish)
        except EnvelopeBug as exc:
            ep.binder.failed_calls.add(call.canonical())
            ep.transport.sanitize = True
            ep.trace.decision("envelope_bug", tool=call.tool, error=clip_text(str(exc), 200))
            return Decoded.blocked("ENVELOPE", str(exc))
        except StepShift as exc:              # a 409 re-sync left only the last step: keep it for the finish
            ep.budget.u = ep.transport.n
            ep.trace.guard(G_LAST, str(exc), intent.kind, call.tool)
            return Decoded.blocked(G_LAST, str(exc))
        ep.plan.note_env_ms((monotonic() - sent_at) * 1000.0)
        ep.budget.u = ep.transport.n
        if finish and ep.budget.first_finish_step is None:
            ep.budget.first_finish_step = ep.transport.n
        ep.guard.note_sent(intent)
        dec = ep.decoder.decode(self._context(intent, call), raw)
        qkey, page = self._query_key(intent, call)
        ep.fields.learn(call.tool, dec, qkey=qkey, page=page, sent=call.args, opened=call.role in RECORD_ROLES)
        if dec.kind in (D_RECORDS, D_RECORD) and ep.fields.fmap(call.tool).ambiguous:
            ep.fields.consult(ep, call.tool, dec)          # FIELDS once per tool on close candidates (M2)
        if ep.fields.rescale_pending:
            self._rescale_prices()
        if dec.kind == D_RECORDS:
            dec.meta = ep.fields.meta(call.tool, dec)
        ep.binder.confirm(intent, call, dec)
        self.absorb(intent, call, dec)
        if not finish:
            ep.events.refused_last = None          # the action after a refusal was not a finish (G15)
        ep.events_machine.ingest(ep, dec, intent)
        ep.trace.step(ep.phase, intent, call, raw, dec, adm.repaired)
        return dec

    def _query_key(self, intent: Intent, call: Call) -> tuple:
        """(query key, paging value sent) for search and filter calls; (None, None) otherwise."""
        if call.role not in (ROLE_SEARCH, ROLE_FILTER):
            return None, None
        paging = [(a, s) for a, s in call.slots_used if s in PAGING_SLOTS]
        key = intent.payload.get("qkey") or QueryBook.key_of(call.tool, call.args, tuple(a for a, _ in paging))
        return key, (call.args.get(paging[0][0]) if paging else None)

    def _rescale_prices(self) -> None:
        """The world turned out to price in major units after integer-looking prices were stored as cents:
        multiply every stored price by 100 (F05)."""
        ep = self.ep
        ep.fields.rescale_pending = False
        for r in ep.catalog.values():
            if r.price_c is not None:
                r.price_c *= 100
            r.price_hist = [(u, c * 100) for u, c in r.price_hist]
        for line in ep.ledger.lines:
            for row in line.table.values():
                if isinstance(row.get("price_c"), int):
                    row["price_c"] *= 100
        ep.trace.decision("fieldmap", why="prices are major units: stored cents x 100", price_major=True)

    def _context(self, intent: Intent, call: Call) -> DecodeContext:
        ep = self.ep
        return DecodeContext(
            tool=call.tool, intent_kind=intent.kind, role=call.role,
            is_finish=is_finish_intent(intent) or call.role in FINISH_ROLES,
            is_message=call.role == ROLE_MESSAGE, is_cart_write=call.role == ROLE_CART_WRITE,
            is_cart_view=call.role == ROLE_CART_VIEW or intent.kind == IK_CART_VIEW,
            expected_empty=not ep.cart.expected,
            written_id=call.write[0] if call.write else None,
            written_qty=(call.args.get(ep.binder.slot_arg(call.tool, SLOT_QTY)) if call.write else None),
            requested_ids=list(call.ids),
            known_prices={i: r.price_c for i, r in ep.catalog.items() if r.price_c is not None},
            expected_qty=dict(ep.cart.expected))

    def absorb(self, intent: Intent, call: Call, dec: Decoded) -> None:
        ep = self.ep
        if dec.kind == D_SESSION_ERROR:
            if dec.session_code in ("invalid_arguments", "unknown_tool"):
                ep.budget.contract_error_streak += 1
            if dec.session_code == CODE_BUDGET_EXCEEDED and intent.kind == IK_ASK:
                ep.budget.M = ep.budget.msgs_used
                ep.budget.M_probe_open = False
                ep.trace.decision("budget", M=ep.budget.M, why="message budget error")
        elif dec.kind != D_BLOCKED:
            ep.budget.contract_error_streak = 0
        if dec.kind in (D_RECORDS, D_RECORD):
            self._absorb_records(intent, call, dec)
        elif dec.kind == D_CART:
            ep.fields.check_cart_prices(ep, dec)       # confirm or re-learn the price path (M2)
            ep.cartman.observe(ep, call, dec)
        elif dec.kind == D_ACK and call.role == ROLE_CART_WRITE:
            ep.cartman.on_ack(ep, call, dec)
        elif dec.kind == D_UNKNOWN and call.role == ROLE_CART_WRITE:
            ep.cartman.on_unverified(ep, call, dec)
        elif dec.kind == D_STATUS:
            self._absorb_status(dec)
        if intent.kind == IK_ASK and dec.kind != D_SESSION_ERROR:
            if dec.kind == D_REPLY and isinstance(dec.inner, dict) and len(dec.inner) > 1:
                self._absorb_status(dec)               # counters beside the reply text (LS-17)
            ep.budget.msgs_used += 1
            ep.elicitor.on_reply(ep, intent, dec)
        if dec.kind == D_FINISHED:
            ep.finished = True
            ep.finish_kind = FIN_ORDER if intent.kind == IK_RAW else ep.binder.finish_kind_of(call, intent.kind)
            ep.finish_tool = call.tool

    def _absorb_records(self, intent: Intent, call: Call, dec: Decoded) -> None:
        ep = self.ep
        u = ep.transport.n
        as_row = call.role in (ROLE_SEARCH, ROLE_FILTER)
        opened = call.role in RECORD_ROLES
        ids, prices = [], []
        for rec in dec.records:
            f = ep.fields.extract(call.tool, rec)
            ident = f.get(FR_ID)
            if not isinstance(ident, str):
                continue
            r = merge_record(ep.catalog, ident, f, u, call.tool, as_row, opened)
            ids.append(ident)
            prices.append(r.price_c)
            if intent.line is not None and 0 <= intent.line < len(ep.ledger.lines) and as_row:
                ep.ledger.lines[intent.line].table[ident] = {"price_c": r.price_c, "stock": r.stock,
                                                            "shelf": r.shelf, "u": u}
        tb = ep.binder.binding.tools.get(call.tool)
        if tb is not None and opened:
            fm = ep.fields.fmap(call.tool)
            tb.provides_seen |= {role for role in fm.role_paths if role in FAMILIES}
            if FR_ATTRS in tb.provides_seen and not any(ep.catalog[i].attrs for i in ids if i in ep.catalog):
                tb.provides_seen.discard(FR_ATTRS)
        ep.contract_learner.learn_id_sig(ids)
        if opened:
            for ident in ids:
                if ident in ep.cart.expected:
                    ep.cart.last_reopen_u[ident] = u
        if dec.absent_ids:
            ep.absent_ids.update(i for i in dec.absent_ids if isinstance(i, str))
        if as_row:
            paging = [(a, s) for a, s in call.slots_used if s in PAGING_SLOTS]
            paging_args = tuple(a for a, _ in paging)
            key = intent.payload.get("qkey") or QueryBook.key_of(call.tool, call.args, paging_args)
            page_sent = call.args.get(paging[0][0]) if paging else None
            n, cont = dec.meta.get("n"), dec.meta.get("cont")
            qrec = ep.queries.record_page(key, call.tool, call.args, intent.payload.get("query", "") or "",
                                          intent.line, intent.payload.get("qkind", "q:value"), page_sent, ids, n,
                                          cont, ep.contract.count_cap)
            paging_arg = ep.binder.slot_arg(call.tool, SLOT_PAGE) or ep.binder.slot_arg(call.tool, SLOT_OFFSET)
            paging_slot = ep.binder.binding.tools[call.tool].slots.get(paging_arg, (None, 0))[0] if paging_arg else None
            paging_schema = (ep.binder.features[call.tool].schema.get("properties") or {}).get(paging_arg) \
                if paging_arg else None
            ep.contract_learner.on_page(call.tool, qrec, dec.records, n, cont, page_sent, paging_slot,
                                        paging_schema, prices, ids)
            if intent.kind == IK_FILTER:
                ceiling, shelf = intent.payload.get("ceiling"), canon_key(intent.payload.get("shelf") or "")
                complete = not qrec.has_cont and isinstance(n, int) and n <= len(set(qrec.ids))
                left_out = bool(shelf) and complete and ceiling is not None and not intent.payload.get("maker") \
                    and any(r.price_c == ceiling and r.stock is not False and r.id not in qrec.ids
                            and canon_key(r.shelf or "") == shelf for r in ep.catalog.values())
                ep.contract_learner.on_ceiling(ceiling, prices, left_out)
            self.last_qkey = key
        if ep.fields.fmap(call.tool).learned_from == 1:
            fm = ep.fields.fmap(call.tool)
            ep.trace.decision("fieldmap", tool=call.tool, roles={k: v for k, v in fm.role_paths.items()},
                              count=fm.count_path, cont=fm.cont_path, attrs_mode=fm.attrs_mode)

    def _absorb_status(self, dec: Decoded) -> None:
        """A status reading of two counters (left, used) gives M = their sum (only when M is not stated)."""
        ep = self.ep
        if ep.budget.M_source == "stated":
            return
        ints: list = []
        for value in (dec.inner or {}).values():
            if isinstance(value, dict):
                ints += [v for v in value.values() if isinstance(v, int) and not isinstance(v, bool)]
            elif isinstance(value, int) and not isinstance(value, bool):
                ints.append(value)
        if len(ints) == 2 and 0 < sum(ints) <= 20:
            ep.budget.M = sum(ints)
            ep.budget.M_source = "status"
            ep.trace.decision("budget", M=ep.budget.M, why="status reading")

def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())

def merge_record(catalog: dict, ident: str, f: dict, u: int, tool: str, as_row: bool, opened: bool) -> Record:
    """Records merge by id across rows, views, batch views and carts; each field keeps a history."""
    r = catalog.get(ident)
    if r is None:
        r = Record(id=ident)
        catalog[ident] = r
    price = f.get(FR_PRICE)
    if isinstance(price, int) and price > 0:
        if r.price_c != price:
            r.price_hist.append((u, price))
        r.price_c = price
    stock = f.get(FR_STOCK)
    if isinstance(stock, bool):
        if r.stock != stock:
            r.stock_hist.append((u, stock))
        r.stock = stock
    if _nonblank(f.get(FR_TITLE)):
        r.title = f[FR_TITLE]
    if _nonblank(f.get(FR_SHELF)):
        r.shelf = f[FR_SHELF]
    if _nonblank(f.get(FR_MERCHANT)):
        r.merchant = f[FR_MERCHANT]
    if _nonblank(f.get(FR_MAKER)):
        r.maker = f[FR_MAKER]
    if isinstance(f.get(FR_OVERVIEW), str):
        r.overview_head = f[FR_OVERVIEW][:240]
    if FR_NOTES in f:
        r.notes = f[FR_NOTES] if isinstance(f[FR_NOTES], str) else r.notes
    for raw_key, raw_value in f.get(FR_ATTRS) or []:
        bucket = r.attrs.setdefault(canon_key(raw_key), [])
        if (raw_key, raw_value) not in bucket:
            bucket.append((raw_key, raw_value))
    if as_row:
        r.seen_rows_at.append(u)
    if opened:
        r.opened_by[tool] = u
    r.families_seen |= {role for role, v in f.items() if role in FAMILIES and (v not in (None, "", []) or
                                                                               role == FR_NOTES)}
    return r

REQUIREMENT_ROUTES = ("requirement", "reference")
_V5P_ROUTES = ("requirement", "reference", "restatement", "objective")
_V5P_STOP_RE = re.compile(r"[();]|[.!?](?= |$)")
_V6F_CARDINALS = "one two three four five six seven eight nine ten eleven twelve".split()

def _v6f_counts(span: str) -> set:
    return {_V6F_CARDINALS.index(t) + 1 for t in text_tokens(span) if t in _V6F_CARDINALS}

def _v5p_at(text: str, needle: str, lo: int = 0, hi: int = 1 << 30) -> list:
    """Word-bounded starts of needle in text[lo:hi]."""
    return [m.start() for m in re.finditer(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", text[:hi])
            if m.start() >= lo] if needle else []

def norm_span(text: Any) -> str:
    return norm_space(str(text or "")).strip(" \t\"'.,;:!?()")

def number_in_span(value: Any, span: str) -> bool:
    if value is None:
        return False
    v = float(value)
    return any(abs(float(n) - v) < 1e-6 for n, _, _, _ in numbers_in(span))

class RequestParser:
    def __init__(self, ep: Any):
        self.ep = ep
        self.request = str(ep.query or "")
        self.req_norm = norm_space(self.request)
        self.opening_text = dump_json(ep.opening) if ep.opening is not None else ""

    def payload(self) -> dict:
        tools = [{"tname": t["tname"], "about": t["about"]} for t in self.ep.binder.tool_digest()]
        out = {"request": self.request, "tools": tools}
        if self.opening_text:
            out["opening"] = clip_text(self.opening_text, 1500)
        return out

    def vote_jobs(self) -> list:
        return [LlmJob("PARSE", self.payload(), models=ladder, validator=self.validator, label=f"parse_vote_{i}")
                for i, ladder in enumerate(self.ep.llm.vote_ladders("PARSE", 2))]

    def span_ok(self, span: Any) -> bool:
        s = norm_span(span)
        return bool(s) and s in self.req_norm

    def validator(self, vote: dict) -> tuple:
        """Semantic validation (lenient: invalid items are dropped; no valid line fails the vote). v5: reqs stand in
        requirement, reference, restatement or objective sentences; none inside the stock rule's span, among a
        reference's descriptors or naming its merchant."""
        routes = [(norm_span(s["span"]), s["route"]) for s in vote.get("sentences") or [] if self.span_ok(s["span"])]
        refs = [self._ref_ok(r) for r in vote.get("refs") or [] if self.span_ok(r["span"])]
        stock = norm_span((vote.get("stock_rule") or {}).get("span"))
        names = [set(text_tokens(ln.get(k))) for ln in vote.get("lines") or [] for k in ("words", "shelf_words")]
        zones = [z for r in refs if r["merchant"] and not any(set(text_tokens(r["merchant"])) <= n for n in names)
                 for z in self._v5p_zone(r, routes)]

        def route_of(span: str) -> Optional[str]:
            target = norm_span(span)
            for sent, route in routes:
                if target and (target in sent or sent in target):
                    return route
            return None

        lines, hints = [], list(vote.get("hints") or [])
        for i, ln in enumerate(vote.get("lines") or []):
            words = (ln.get("words") or "").strip()
            if not words:
                continue
            qty = ln.get("qty_n")
            qspan = ln.get("qty_span") or ""
            wordy = False
            if qspan and self.span_ok(qspan):
                if numbers_in(qspan):
                    if not number_in_span(qty, qspan):
                        qty = None
                elif qty in _v6f_counts(qspan):
                    wordy = True                 # a number in words: kept only when both votes agree (to_ledger)
                else:
                    qty = None                   # v6: no digit, no cardinal: one item
            else:
                qty = None                       # a quantity needs a verbatim span that holds it (F10)
            qty = int(qty) if isinstance(qty, (int, float)) and 1 <= qty <= 99 else 1
            reqs = []
            for rq in ln.get("reqs") or []:
                span_n = norm_span(rq["span"])
                if not self.span_ok(span_n) or self.span_ok(stock) and span_n in stock or self._v5p_desc(
                        rq, span_n, zones):
                    continue
                if route_of(rq["span"]) not in _V5P_ROUTES:   # story, hint, a field sentence, or no routed sentence
                    hints.append({"line": i, "attr_guess": rq["attr"], "span": rq["span"]})
                    continue
                vals = [v for v in (rq.get("vals") or []) if norm_span(v) and norm_span(v) in span_n]
                if rq["op"] in ("ge", "le", "between"):
                    if rq.get("num") is None or not number_in_span(rq["num"], rq["span"]):
                        continue
                elif not vals:
                    continue
                reqs.append(dict(rq, vals=vals))
            lines.append(dict(ln, words=words, qty_n=qty, reqs=reqs, _qty_wordy=wordy and qty != 1))
        if not lines:
            return False, None, "no line with a valid span"
        out = dict(vote)
        out["lines"] = lines
        out["hints"] = [h for h in hints if self.span_ok(h.get("span"))]
        out["refs"] = refs
        out["procedures"] = [self._proc_ok(p) for p in vote.get("procedures") or [] if self.span_ok(p.get("span"))]
        out["_routes"], out["_zones"] = routes, zones
        self._rules_ok(out, vote)
        for key in ("msgs", "steps"):
            item = vote.get(key) or {}
            n, span = item.get("n"), item.get("span") or ""
            if n is not None and not (self.span_ok(span) and (number_in_span(n, span) or not numbers_in(span))):
                out[key] = dict(item, n=None)
            elif n is not None and self.span_ok(span) and not numbers_in(span):
                out[key] = dict(item, wordy=True)
        cap = vote.get("cap") or {}
        if cap.get("amount") is not None:
            span = cap.get("span") or ""
            if not (self.span_ok(span) and (number_in_span(cap["amount"], span) or not numbers_in(span))):
                out["cap"] = dict(cap, amount=None)
            elif not numbers_in(span):
                out["cap"] = dict(cap, wordy=True)
        obj = vote.get("objective") or {}
        if obj.get("kind") == "cheapest" and not self.span_ok(obj.get("span")):
            out["objective"] = {"kind": "any", "span": ""}
        return True, out, ""

    def verbatim(self, text: Any) -> bool:
        return isinstance(text, str) and bool(norm_span(text)) and norm_span(text) in self.req_norm

    def _ref_ok(self, ref: dict) -> dict:
        """A reference keeps its merchant only when the request writes it, and descriptors whose value it writes
        (descriptors locate the source listing; they never constrain the wanted item)."""
        merchant = ref.get("merchant") if self.verbatim(ref.get("merchant")) else ""
        descriptors = [list(d) for d in ref.get("descriptors") or [] if len(d) == 2 and self.verbatim(d[1])]
        return dict(ref, merchant=merchant, descriptors=descriptors)

    def _v5p_zone(self, r: dict, routes: list) -> list:
        """[(start, end, attr, merchant)] of a reference's descriptors: after its merchant's last mention up to its
        span's end, up to the end of the last descriptor value it lists, never past the clause end (';', an unopened
        ')', the sentence end). A span that is a whole routed sentence bounds nothing: a clause written after the
        reference (a stated requirement) is no descriptor."""
        text, m, span = self.req_norm, norm_span(r["merchant"]), norm_span(r["span"])
        end = text.find(span) + len(span)
        for a in _v5p_at(text, m, 0, end)[-1:]:
            depth, a, stop = 0, a + len(m), len(text)
            for x in _V5P_STOP_RE.finditer(text, a):
                depth += {"(": 1, ")": -1}.get(x.group(), 0)
                if depth < 0 or depth == 0 and x.group() not in "()":
                    stop = x.start()
                    break
            last = [p + len(v) for v in (norm_span(d[1]) for d in r["descriptors"]) for p in _v5p_at(text, v, a, stop)[:1]]
            last += [end] if all(span != s for s, _r in routes) else []
            return [(a, min(stop, max(last, default=a)), canon_key(r["attr"]), m)]
        return []

    def _v5p_desc(self, rq: dict, span_n: str, zones: list) -> bool:
        """A value naming a merchant, or every value (else the span) among another attribute's descriptors."""
        vals = [v for v in map(norm_span, rq.get("vals") or []) if v]
        pos, attr = self.req_norm.find(span_n), canon_key(rq["attr"])
        spots = [(_v5p_at(self.req_norm, v, pos, pos + len(span_n)) or [-1])[0] for v in vals] or [pos]
        return any(_v5p_at(v, z[3]) for z in zones for v in vals) or bool(zones) and all(
            any(a <= s < b and k != attr for a, b, k, _m in zones) for s in spots)

    def _proc_ok(self, proc: dict) -> dict:
        n = proc.get("n")
        if n is not None and not number_in_span(n, proc.get("span") or ""):
            n = None
        text = proc.get("text") if self.verbatim(proc.get("text")) else proc.get("span")
        return dict(proc, n=n, text=text)

    def _rules_ok(self, out: dict, vote: dict) -> None:
        """Stock, decline and compatibility rules move off their safe defaults only on a verbatim span; remark
        rules are verbatim request text."""
        stock = vote.get("stock_rule") or {}
        if stock.get("required") is False and not self.span_ok(stock.get("span")):
            out["stock_rule"] = dict(stock, required=True)
        dr = dict(vote.get("decline_rule") or {})
        if not self.span_ok(dr.get("span")):
            dr["allowed"], dr["no_substitute"] = True, False
        dr["remark_rules"] = [r for r in dr.get("remark_rules") or [] if self.verbatim(r)]
        out["decline_rule"] = dr
        comp = dict(vote.get("compat") or {})
        if comp.get("required") and not self.span_ok(comp.get("span")):
            comp["required"] = False
        out["compat"] = comp

    @staticmethod
    def _wordy_agreed(v: dict, votes: list) -> dict:
        """DESIGN 5.8: a number whose span has no digits stands only when both votes give the same number."""
        out = dict(v)
        for key, num in (("cap", "amount"), ("msgs", "n"), ("steps", "n")):
            item = out.get(key) or {}
            if not item.get("wordy") or item.get(num) is None:
                continue
            values = [(x.get(key) or {}).get(num) for x in votes]
            if len(values) < 2 or any(val != item[num] for val in values):
                out[key] = dict(item, **{num: None})
        return out

    @staticmethod
    def _cset(line: dict) -> set:
        return {(canon_key(r["attr"]), r["op"], tuple(sorted(canon_key(v) for v in r.get("vals") or [])),
                 r.get("num")) for r in line.get("reqs") or []}

    def _v5p_cross(self, votes: list) -> list:
        """A detail one validated answer delimits as a reference's descriptor (or that names its merchant) is no req of
        another: each answer's bounded zones apply to all."""
        zones = [z for v in votes for z in v.get("_zones") or []]
        return [dict(v, lines=[dict(ln, reqs=[r for r in ln.get("reqs") or [] if not self._v5p_desc(
            r, norm_span(r["span"]), zones)]) for ln in v["lines"]]) for v in votes] if zones else votes

    def compare(self, a: dict, b: dict) -> list:
        a, b = self._v5p_cross([a, b])
        out = []
        la, lb = a["lines"], b["lines"]
        if len(la) != len(lb):
            out.append("lines:count")
        else:
            for x, y in zip(self._align(la, lb), lb):
                if x["qty_n"] != y["qty_n"]:
                    out.append("quantity")
                if self._cset(x) != self._cset(y):
                    out.append("constraints")
        if (a.get("cap") or {}).get("amount") != (b.get("cap") or {}).get("amount"):
            out.append("cap")
        if (a.get("objective") or {}).get("kind") != (b.get("objective") or {}).get("kind"):
            out.append("objective")
        for key in ("msgs", "steps"):
            if (a.get(key) or {}).get("n") != (b.get(key) or {}).get("n"):
                out.append(key)
        refs_a = {(canon_key(r["attr"]), r["rel"], canon_key(r.get("merchant"))) for r in a.get("refs") or []}
        refs_b = {(canon_key(r["attr"]), r["rel"], canon_key(r.get("merchant"))) for r in b.get("refs") or []}
        if refs_a != refs_b:
            out.append("refs")
        return sorted(set(out))

    @staticmethod
    def _align(la: list, lb: list) -> list:
        """Reorder la to match lb by word overlap (lines are aligned by their words)."""
        remaining = list(la)
        out = []
        for y in lb:
            best = max(remaining, key=lambda x: token_overlap(x["words"] + " " + (x.get("shelf_words") or ""),
                                                              y["words"] + " " + (y.get("shelf_words") or "")))
            remaining.remove(best)
            out.append(best)
        return out

    def resolve(self, a: dict, b: dict) -> dict:
        """Field-wise resolution when the arbiter fails (DESIGN 5.8)."""
        out = dict(a)
        for key in ("cap", "msgs", "steps"):
            av = (a.get(key) or {})
            bv = (b.get(key) or {})
            first = av.get("amount" if key == "cap" else "n")
            out[key] = av if first is not None else bv
        lines = a["lines"] if a["lines"] else b["lines"]
        other = b["lines"] if lines is a["lines"] else a["lines"]
        routes_a = dict(a.get("_routes") or [])
        routes_b = dict(b.get("_routes") or [])
        merged = []
        if len(lines) == len(other):
            for x, y in zip(lines, self._align(other, lines)):
                both = self._cset(x) & self._cset(y)
                keep = [r for r in x.get("reqs") or [] if (canon_key(r["attr"]), r["op"], tuple(sorted(
                    canon_key(v) for v in r.get("vals") or [])), r.get("num")) in both]
                for r in (x.get("reqs") or []) + (y.get("reqs") or []):
                    if r in keep:
                        continue
                    sent = norm_span(r["span"])
                    ra = next((rt for s, rt in routes_a.items() if sent and (sent in s or s in sent)), None)
                    rb = next((rt for s, rt in routes_b.items() if sent and (sent in s or s in sent)), None)
                    if ra == "requirement" and rb == "requirement":
                        keep.append(r)
                merged.append(dict(x, reqs=keep))
        else:
            merged = list(lines)
        if len(lines) == len(other):
            for x, y in zip(merged, self._align(other, lines)):
                if not (x.get("qty_span") and self.span_ok(x["qty_span"])) and y.get("qty_span") \
                        and self.span_ok(y["qty_span"]) and not y.get("_qty_wordy"):
                    x["qty_n"] = y["qty_n"]          # the quantity whose span validates
        out["lines"] = merged
        out["objective"] = a.get("objective") if (a.get("objective") or {}).get("kind") else b.get("objective")

        def routed(span: Any, wanted: tuple) -> bool:
            sent = norm_span(span)
            hits = [next((rt for s, rt in routes.items() if sent and (sent in s or s in sent)), None)
                    for routes in (routes_a, routes_b)]
            return all(h in wanted for h in hits)

        def ref_key(r: dict) -> tuple:
            return canon_key(r["attr"]), r["rel"], canon_key(r.get("merchant"))

        keys_b = {ref_key(r) for r in b.get("refs") or []}
        refs = [r for r in a.get("refs") or [] if ref_key(r) in keys_b]
        for r in (a.get("refs") or []) + (b.get("refs") or []):
            if r not in refs and ref_key(r) not in {ref_key(x) for x in refs} and routed(r["span"], REQUIREMENT_ROUTES):
                refs.append(r)
        out["refs"] = refs
        procs = list(a.get("procedures") or [])
        for p in b.get("procedures") or []:
            if norm_span(p.get("span")) not in {norm_span(x.get("span")) for x in procs} and routed(p.get("span"),
                                                                                              ("procedure",)):
                procs.append(p)
        out["procedures"] = procs
        for key in ("stock_rule", "decline_rule", "compat"):
            out[key] = a.get(key) if self.span_ok((a.get(key) or {}).get("span")) else (b.get(key) or a.get(key))
        hints = list(a.get("hints") or [])
        hints += [h for h in b.get("hints") or [] if h not in hints]
        out["hints"] = hints
        return out

    def parse(self, answers: list, arbiter: Optional[dict] = None, conflicts: Optional[list] = None) -> Ledger:
        """Merge validated votes. The orchestrator runs the v4-pro arbiter (in parallel with the BIND arbiter)
        when compare() finds conflicts and passes its answer here."""
        votes = [v for v in answers if isinstance(v, dict)]
        if not votes:
            return self.fallback()
        both = self._v5p_cross(votes + [arbiter] if isinstance(arbiter, dict) else votes)
        votes, arbiter = (both[:-1], both[-1]) if isinstance(arbiter, dict) else (both, arbiter)
        if len(votes) == 1:
            return self.to_ledger(self._wordy_agreed(votes[0], votes), "single_vote", [])
        conflicts = conflicts if conflicts is not None else self.compare(votes[0], votes[1])
        if not conflicts:
            return self.to_ledger(self._wordy_agreed(votes[0], votes[:2]), "votes", [])
        if isinstance(arbiter, dict):
            return self.to_ledger(self._wordy_agreed(arbiter, votes[:2]), "arbiter", conflicts)
        return self.to_ledger(self._wordy_agreed(self.resolve(votes[0], votes[1]), votes[:2]), "resolved", conflicts)

    def to_ledger(self, v: dict, source: str, conflicts: list) -> Ledger:
        led = Ledger(source=source, conflicts=list(conflicts))
        objective = (v.get("objective") or {}).get("kind") or "any"
        wordy_ok = source == "votes" or (source in ("arbiter", "resolved")
                                         and not ({"quantity", "lines:count"} & set(conflicts)))
        for i, ln in enumerate(v["lines"]):
            qty = int(ln["qty_n"]) if (wordy_ok or not ln.get("_qty_wordy")) else 1
            line = Line(idx=i, words=ln["words"], shelf_words=ln.get("shelf_words") or "", qty=qty,
                        role=ln.get("line_role") or "independent", cheapest=objective == "cheapest")
            for j, rq in enumerate(ln.get("reqs") or []):
                bound = None
                if rq["op"] in ("ge", "le", "between") and rq.get("num") is not None:
                    bound = {"num": rq["num"], "unit": rq.get("unit"), "raw": rq["span"]}
                line.constraints.append(Constraint(cid=f"c{i}.{j}", attr=canon_key(rq["attr"]), op=rq["op"],
                                                   vals=list(rq.get("vals") or []), bound=bound, source="stated",
                                                   span=rq["span"]))
            led.lines.append(line)
        cap = v.get("cap") or {}
        if cap.get("amount") is not None:
            led.cap_c = cap_to_cents(cap["amount"], cap.get("span") or "")
            led.cap_span = cap.get("span") or ""
        led.objective = objective
        stock = v.get("stock_rule") or {}
        led.in_stock = stock.get("required") is not False
        dr = v.get("decline_rule") or {}
        led.decline_rule = {"allowed": dr.get("allowed") is not False, "no_substitute": bool(dr.get("no_substitute")),
                            "remark_rules": list(dr.get("remark_rules") or []), "span": dr.get("span") or ""}
        led.procedures = [dict(p) for p in v.get("procedures") or [] if self.span_ok(p.get("span"))]
        for k, r in enumerate(v.get("refs") or []):
            line_idx = r.get("line") if isinstance(r.get("line"), int) and 0 <= r["line"] < len(led.lines) else 0
            led.refs.append(Reference(rid=f"r{k}", line=line_idx, attr=canon_key(r["attr"]), relation=r["rel"],
                                      origin="request", source={"merchant": r.get("merchant") or "",
                                                                "shelf_words": r.get("shelf_words") or "",
                                                                "descriptors": list(r.get("descriptors") or [])},
                                      span=r["span"]))
        led.hints = [dict(h, line=h["line"] if isinstance(h.get("line"), int) and 0 <= h["line"] < len(led.lines)
                          else None) for h in v.get("hints") or [] if self.span_ok(h.get("span"))]
        comp = v.get("compat") or {}
        base = comp.get("base_line")
        led.compat = {"required": bool(comp.get("required")),
                      "base_line": base if isinstance(base, int) and 0 <= base < len(led.lines) else None,
                      "span": comp.get("span") or ""}
        msgs = (v.get("msgs") or {}).get("n")
        steps = (v.get("steps") or {}).get("n")
        led.stated_msgs = int(msgs) if isinstance(msgs, (int, float)) and 0 <= msgs <= 50 else None
        led.stated_steps = int(steps) if isinstance(steps, (int, float)) and 1 <= steps <= 500 else None
        led.sentences = list(v.get("_routes") or [])
        return led

    def fallback(self) -> Ledger:
        """v5: the cap is never the step budget or a count ('3 messages', '16 inches'); a lone count is M only in
        DEFAULT_M.._ELI_PROBE_MAX, without a bound status tool, if its word does not recur."""
        text = self.request
        led = Ledger(source="fallback")
        nums = numbers_in(text)
        ms = self.ep.max_steps
        if type(ms) is int and any(n[0] == ms for n in nums):
            led.stated_steps = ms                        # e_q: the step-budget sentence
        strong, lone = self._qz_msgs(nums, True), self._qz_msgs(nums, False)
        counts = {c[2] for c in strong + lone}           # the counts themselves: a cap in the same sentence stands
        money = [n for n in nums if has_currency_symbol_near(text, n[2], n[3])]
        cap = money[0] if money else max((n for n in nums if n[0] != led.stated_steps and n[2] not in counts),
                                         key=lambda n: n[0], default=None)
        if cap:
            led.cap_c = cap_to_cents(cap[0], cap[1])
            led.cap_span = cap[1]
        status = any(tb.role == ROLE_STATUS and tb.role_conf >= 2 for tb in self.ep.binder.binding.tools.values())
        lone = [c for c in lone if DEFAULT_M <= c[0] <= _ELI_PROBE_MAX and not status
                and _qz_stem(c[3]) not in _qz_toks(text[:c[4][0]] + " " + text[c[4][1]:])]
        m = [{c[0] for c in cands} for cands in (strong, lone)]   # any reading makes M stated
        led.stated_msgs = next((v.pop() for v in m if len(v) == 1), None)
        drop = {led.stated_steps, led.stated_msgs, cap[0] if cap else None} - {None}
        story = " ".join(x for x in _qz_sentences(text) if not drop & {n[0] for n in numbers_in(x)})
        queries = guess_queries(story) or guess_queries(text)   # budget sentences name no product
        line = Line(idx=0, words=queries[0] if queries else "", qty=1, guessed=True, query_alts=queries)
        if line.words:
            led.lines.append(line)
        led.objective = "any"
        return led

    def _qz_msgs(self, nums: list, strong: bool) -> list:
        """M candidates (value, raw, start, next word, sentence span), never the step budget or money. Strong: a small
        integer before a word sharing a 4-letter stem with the message or status tool's name or description; lone:
        one before a plural word, alone in a sentence without parentheses."""
        text, b, out = self.request, self.ep.binder, []
        stems = set()
        for tool in (b.tool_for(ROLE_MESSAGE), b.tool_for(ROLE_STATUS)) if strong else ():
            tf = b.features.get(tool) if tool else None
            words = name_tokens(tool) + content_tokens(tf.description) if tf else []
            stems |= {t[:4] for t in words if len(t) >= 4}
        for value, raw, start, end in nums:
            nxt = (text_tokens(text[end:end + 24]) or [""])[0]
            span = next((s for s in _qz_sentences(text, True) if s[0] <= start < s[1]), (start, end))
            sent = text[span[0]:span[1]]
            if type(value) is not int or not 1 <= value <= 20 or value == self.ep.max_steps or len(nxt) < 3 \
                    or not nxt.isalpha() or has_currency_symbol_near(text, start, end):
                continue
            if nxt[:4] in stems if strong else (nxt.endswith("s") and "(" not in sent and ")" not in sent
                                                and len(numbers_in(sent)) == 1):
                out.append((value, raw, start, nxt, span))
        return out

    def probe_key(self, query: str, ids: list, n: int) -> tuple:
        """Scout contract (guessed lines): rank probes by the share of rows whose title or shelf names the word."""
        want = set(_qz_toks(query)) - STOPWORDS
        rows = [self.ep.catalog[i] for i in ids if i in self.ep.catalog]
        share = sum(1 for r in rows if want and want <= set(_qz_toks(f"{r.title} {r.shelf or ''}"))) / max(1, len(rows))
        return share >= 0.5, round(share * max(n, len(rows)), 3), n

def check_cap_unit(ep: Any) -> bool:
    """DESIGN 5.8 cap unit check, at selection time (F13): an integer cap written without decimals is read
    in the catalog's minor unit. When no row of any line fits q * price <= cap, the cheapest row is more
    than ten times the cap and fits q * price <= cap * 100, the cap was written in major units: x 100.
    Done at most once; the first cart total stays the confirmation."""
    led = ep.ledger
    cap = led.cap_c
    if cap is None or cap <= 0 or getattr(ep, "cap_unit_checked", False):
        return False
    if any(isinstance(n, float) for n, _, _, _ in numbers_in(led.cap_span)):
        return False
    costs = [line.qty * ep.catalog[i].price_c for line in led.lines for i in line.table
             if i in ep.catalog and ep.catalog[i].price_c and not (ep.catalog[i].stock is False and led.in_stock)]
    if not costs:
        return False
    cheapest = min(costs)
    if cheapest <= cap or cap * 10 >= cheapest or cheapest > cap * 100:
        return False
    ep.cap_unit_checked = True
    led.cap_c = cap * 100
    led.version += 1
    ep.trace.decision("cap_unit", cap_c=led.cap_c, was=cap, cheapest=cheapest,
                      why="no row fits the cap read in minor units; major units fit")
    ep.plan.compute("cap unit")
    return True

def cap_to_cents(amount: Any, span: str) -> Optional[int]:
    """Integer cents. A decimal amount (as written) is in major units: x 100."""
    number = as_number(amount)
    if number is None:
        return None
    decimal_written = any(isinstance(n, float) for n, _, _, _ in numbers_in(span) if abs(float(n) - number) < 1e-6)
    if not float(number).is_integer() or decimal_written:
        return int(round(number * 100))
    return int(number)

def guess_queries(text: str, max_n: int = 4) -> list:
    """Dead-inference line guess: content words ranked by repetition and by following an indefinite
    article; a word right after a number (unit, budget noun; v5: only there) and capitalised mid-sentence words
    (names) are skipped. A heuristic of the deterministic fallback, not a vocabulary."""
    stripped = re.sub(r"\([^)]*\)", " ", str(text or ""))
    raw = re.findall(r"[A-Za-z][A-Za-z'\-]*|\d+(?:[.,]\d+)?|[.!?]", stripped)
    lower = [w.lower().strip("'-") for w in raw]
    skip, after = set(), {i + 1 for i, w in enumerate(lower) if w[:1].isdigit()}
    for i, w in enumerate(lower):
        if i > 0 and raw[i][:1].isupper() and lower[i - 1] not in (".", "!", "?"):
            skip.add(w)
    counts: dict = {}
    for w in lower:
        if w.isalpha() and len(w) >= 3 and w not in STOPWORDS:
            counts[w] = counts.get(w, 0) + 1
    scores: dict = {}
    for i, w in enumerate(lower):
        if not w.isalpha() or len(w) < 3 or w in STOPWORDS or w in skip or i in after:
            continue
        s = float(counts.get(w, 0))
        if i > 0 and lower[i - 1] in ("a", "an"):
            s += 2.0
        s += min(len(w), 10) / 20.0
        scores[w] = max(scores.get(w, 0.0), s)
    ranked = sorted(scores, key=lambda w: (-scores[w], lower.index(w)))
    return ranked[:max_n]

DEFAULT_B = 40
DEFAULT_M = 3

def resolve_B(max_steps: Any, stated: Optional[int]) -> tuple:
    """B = min(policy_view.max_steps, stated budget); either alone if the other is missing; 40 if both."""
    ms = max_steps if isinstance(max_steps, int) and not isinstance(max_steps, bool) and max_steps > 0 else None
    st = stated if isinstance(stated, int) and stated > 0 else None
    if ms and st:
        return min(ms, st), "both"
    if ms:
        return ms, "max_steps"
    if st:
        return st, "stated"
    return DEFAULT_B, "default"

def plan_numbers(B: int, L: int, k: int, M: int, *, status_step: bool = False, ref_hops: Optional[list] = None,
                 writes_return_cart: Optional[bool] = True, cart_view: bool = False, fast_path: bool = True,
                 events_before_finish: bool = False, weights: Optional[list] = None, degrade: int = 0) -> dict:
    """Every step-budget quantity of DESIGN 3.2. ref_hops: one entry per reference, the known hop count
    or None (assume 3). degrade: how many rungs of the Free < L degrade ladder to apply."""
    L = max(0, int(L))
    k = max(1, int(k))
    if L <= 1:
        R = clamp(ceil_div(15 * B, 100), 4, 10)
    else:
        R = clamp(ceil_div(10 * B, 100) + 1, 4, 10)
    if events_before_finish:
        R = min(12, R + 2)
    n_l = min(k, 6) if k >= 3 else (3 if L == 1 else 2)
    hops = [h if isinstance(h, int) else 3 for h in (ref_hops or [])]
    hops = [min(5, max(0, h)) for h in hops]
    if degrade >= 1:
        n_l = 1
    if degrade >= 2:
        hops = [h if isinstance(h0, int) else 1 for h, h0 in zip(hops, ref_hops or [])]
    if degrade >= 4:
        R = 4
    F = B - R
    scout_l = 1 + ceil_div(n_l, k)
    sigma_scout = scout_l * L
    s_status = 1 if status_step else 0
    sigma_ref = sum(2 + 2 * h for h in hops)
    if L <= 1:
        V = 0 if fast_path else 1
    else:
        V = ceil_div(L, k) if k >= 2 else 1
        if writes_return_cart is False and cart_view:
            V += 1
    u_c_hard = F - (L + V + 1)
    u_c_soft = min(u_c_hard, ceil_div(6 * B, 10))
    T_ask = min(ceil_div(3 * B, 10), sigma_scout + s_status + 2)
    fixed = sigma_scout + s_status + M + sigma_ref + L + V + 1 + R
    free = B - fixed
    w = weights or [1.0] * L
    total_w = sum(w) or 1.0
    C = [max(0, int(math.floor(free * wi / total_w))) for wi in w]
    if degrade >= 3:
        C = [max(1, c) if free < L else c for c in C]
    return dict(B=B, L=L, k=k, M=M, R=R, F=F, n_l=n_l, scout_l=scout_l, sigma_scout=sigma_scout, S_status=s_status,
                sigma_ref=sigma_ref, V=V, u_c_hard=u_c_hard, u_c_soft=u_c_soft, T_ask=T_ask, fixed=fixed, free=free,
                C=C, degrade=degrade)

class Planner:
    """Reads ep.budget, ep.ledger, ep.binder, ep.cart, ep.events, ep.contract; writes ep.budget.R/F/L."""

    def __init__(self, ep: Any):
        self.ep = ep
        self.nums: dict = {}
        self.recover_search_used = 0
        self.env_ms: list = []
        self.llm_ms: list = []

    def batch_k(self) -> int:
        ep = self.ep
        batch = ep.binder.tool_for(ROLE_BATCH) if ep.binder.binding.tools else None
        if batch is None:
            return 1
        ids_arg = ep.binder.slot_arg(batch, SLOT_IDS)
        max_items = ep.binder.features[batch].args[ids_arg].max_items if ids_arg else None
        return max(1, ep.contract_learner.batch_limit(max_items))

    def weights(self) -> list:
        led = self.ep.ledger
        out = []
        for line in led.lines:
            has_ref = any(r.line == line.idx for r in led.refs)
            out.append(1.0 + (1.0 if line.cheapest else 0.0) + (0.5 if line.role == "accessory" else 0.0)
                       + (0.5 if has_ref else 0.0))
        return out

    def compute(self, why: str = "") -> dict:
        ep = self.ep
        b = ep.budget
        L = len(ep.ledger.lines)
        k = self.batch_k()
        status_agreed = b.M_source == "default" and any(
            tb.role == ROLE_STATUS and tb.role_conf >= 2 for tb in ep.binder.binding.tools.values())
        ref_hops = [(len(r.hops) if r.status in ("resolved", "partial", "failed") else None) for r in ep.ledger.refs]
        cart_view = ep.binder.tool_for(ROLE_CART_VIEW) is not None
        events_early = ep.events.before_first_finish > 0 and b.first_finish_step is None
        degrade = 0
        while True:
            nums = plan_numbers(b.B, L, k, b.M, status_step=status_agreed and b.M_source == "default",
                                ref_hops=ref_hops, writes_return_cart=ep.cart.returns_cart, cart_view=cart_view,
                                fast_path=True, events_before_finish=events_early, weights=self.weights(),
                                degrade=degrade)
            if nums["free"] >= L or degrade >= 4:
                break
            degrade += 1
        b.L, b.R, b.F = L, nums["R"], nums["F"]
        self.nums = nums
        ep.trace.decision("plan", why=why, **{k2: v for k2, v in nums.items()})
        return nums

    @property
    def R(self) -> int:
        return self.nums.get("R", self.ep.budget.R)

    def u(self) -> int:
        return self.ep.transport.n

    def rem(self) -> int:
        return self.ep.budget.B - self.ep.transport.n

    def need_finish(self) -> int:
        """(lines not in the cart at plan quantity) + (1 if the cart is unverified since its last change and
        the fast path does not apply) + 1."""
        ep = self.ep
        missing = 0
        for line in ep.ledger.lines:
            if not line.choice or ep.cart.expected.get(line.choice) != line.qty:
                missing += 1
        unverified = 0
        if ep.cart.expected and len(ep.ledger.lines) >= 2:
            if ep.cart.verified_u is None or (ep.cart.last_change_u or 0) > ep.cart.verified_u:
                unverified = 1
        return missing + unverified + 1

    def protected(self) -> int:
        """Steps reserved for graded procedures: unsent procedure messages, unopened chain pages and the
        procedure gate's own opens (F11)."""
        ep = self.ep
        return ep.elicitor.reserved_unsent(ep) + ep.resolver.unopened_chain_pages(ep) + ep.reviewer.reserved_opens(ep)

    def can_spend(self, k: int, purpose: str) -> bool:
        ep = self.ep
        t = ep.clock()
        if ep.finished or ep.stop_reason in ("quarantined", "key_exhausted", "finished") or t > T_FALLBACK:
            return False
        if purpose in (P_EXPLORE, P_COVER) and (t > T_EXPLORE_STOP or k > self.explore_room()):
            return False
        if purpose in (P_ASK, P_RESOLVE, P_PROBE) and t >= T_DETERMINISTIC_CHECKOUT:
            return False                     # 225 s: deterministic checkout only (DESIGN 3.5)
        rem = self.rem()
        need = self.need_finish()
        reserve = self.R if ep.budget.first_finish_step is None else 0
        if purpose in (P_EXPLORE, P_COVER, P_PROBE):
            return rem - k >= need + reserve + self.protected()
        if purpose in (P_ASK, P_RESOLVE):
            return rem - k >= need + reserve
        if purpose in (P_COMMIT, P_VERIFY, P_RECOVER):
            return rem - k >= 1
        if purpose == P_FINISH:
            return rem >= 1
        return False

    def endgame(self) -> bool:
        return self.ep.budget.first_finish_step is None and self.rem() <= self.R + self.need_finish()

    def pressure(self) -> bool:
        """Pressure release: quality gates Q1-Q7 become advisory; Q8 stays hard (DESIGN 3.3)."""
        return self.rem() <= self.need_finish() + 2

    def allows(self, intent: Intent) -> bool:
        ep = self.ep
        ok = self.can_spend(1, intent.purpose)
        why = "" if ok else "can_spend"
        if ok and intent.purpose in (P_EXPLORE, P_COVER, P_PROBE) and self.nums:
            if self.u() >= self.nums.get("u_c_hard", 10 ** 6):
                ok, why = False, "commit deadline (u_c_hard)"      # certificates included (F20)
            elif self.u() >= self.nums.get("u_c_soft", 10 ** 6) and not intent.payload.get("certificate"):
                ok, why = False, "commit window (u_c_soft)"
        if ok and intent.kind == IK_ASK and ep.cart.writes > 0 and not intent.payload.get("procedure"):
            ok, why = False, "questions come before the first cart write"
        if ok and intent.purpose in (P_RECOVER,) and intent.kind in (IK_SEARCH, IK_FILTER):
            self.recover_search_used += 1
        if not ok:
            ep.trace.decision("plan_block", intent=intent.kind, purpose=intent.purpose, why=why, rem=self.rem(),
                              need=self.need_finish())
        return ok

    def deadline(self, phase: str) -> bool:
        ep = self.ep
        t = ep.clock()
        if t >= T_FALLBACK:
            return True
        if phase == "SCOUT":
            return self.u() >= self.nums.get("T_ask", 10 ** 6)      # DESIGN 3.1: a candidate per line or T_ask
        if phase in ("SELECT", "COVER"):
            if t >= T_EXPLORE_STOP:
                return True
            return self.u() >= self.nums.get("u_c_soft", 10 ** 6) and not ep.coverage.cert_pending(ep)
        if phase in ("ELICIT", "RESOLVE"):
            return self.pressure()
        return False

    def note_env_ms(self, ms: float) -> None:
        self.env_ms.append(ms)
        if len(self.env_ms) > 200:
            del self.env_ms[:100]

    def mean_env_s(self) -> float:
        """Running mean environment latency in seconds (1 s before any call), without the slowest tenth of the
        samples (at least one from three samples on): one hung call must not make every later step look impossible."""
        if not self.env_ms:
            return 1.0
        ms = sorted(self.env_ms)
        keep = ms[:len(ms) - (ceil_div(len(ms), 10) if len(ms) >= 3 else 0)]
        return sum(keep) / len(keep) / 1000.0

    def explore_room(self) -> int:
        """Environment steps that still end before T_EXPLORE_STOP at the running mean latency (DESIGN 3.5 overrun
        projection; coverage may cap C_l with it). EXPLORE and COVER calls beyond it are not spent."""
        left = T_EXPLORE_STOP - self.ep.clock()
        return int(left / max(self.mean_env_s(), 0.05)) if left > 0 else 0

    def explore_steps_left(self) -> int:
        """Exploration steps still planned: up to the target commit start u_c_soft."""
        return max(0, int(self.nums.get("u_c_soft", self.u())) - self.u())

    def overrun_risk(self, planned_env: int, planned_llm: int = 0) -> bool:
        """t + planned_env * lambda_env + planned_llm * lambda_llm > 190 (DESIGN 3.5)."""
        lam_env = self.mean_env_s()
        lam_llm = (sum(self.llm_ms) / len(self.llm_ms) / 1000.0) if self.llm_ms else 3.0
        return self.ep.clock() + planned_env * lam_env + planned_llm * lam_llm > T_EXPLORE_STOP

class Engine:
    """Base engine: a pure planner over the ledger. Engines never call the environment."""
    name = "engine"

    def __init__(self, ep: Any = None):
        self.ep = ep

    def next_intent(self, ep: Any) -> Optional[Intent]:
        return None

    def after(self, ep: Any, intent: Intent, dec: Decoded) -> None:
        return None

    def done(self, ep: Any) -> bool:
        return True

@dataclass
class _ScoutLine:
    tried: list = field(default_factory=list)
    hits: int = 0
    qkey: Optional[str] = None
    opened: bool = False
    exhausted: bool = False
    failed_ids: set = field(default_factory=set)
    open_tries: int = 0
    counts: dict = field(default_factory=dict)      # query -> match count (guessed lines)
    best_ids: list = field(default_factory=list)    # rows of the best-matching query (guessed lines)
    best_key: Optional[tuple] = None                # v3: RequestParser.probe_key of that query (guessed lines)
    open_calls: int = 0                             # v2: open calls sent for menu growth (pass 2)
    open_fails: int = 0                             # v2: growth opens that failed
    grown: bool = False                             # v2: the thin-menu open was planned (pass 3)
    topups: int = 0                                 # h2: menu top-up opens requested by the elicitor

class Scout(Engine):
    name = "scout"
    OPEN_TRIES = 2
    GUESS_PROBES = 4          # dead-parse lines: probe up to 4 guessed queries (v3) ...
    GUESS_ENOUGH_HITS = 50    # ... until one matches this many listings and its rows mostly name its word
    GROW_FAILS = 2            # failed growth opens before a line stops growing
    THIN_MENU = 6             # usable menu keys below which one more batch opens (DESIGN 5.9) ...
    THIN_FREE = 12            # ... while Free >= 12

    def __init__(self, ep: Any = None):
        super().__init__(ep)
        self.state: dict = {}

    def st(self, line: Line) -> _ScoutLine:
        if line.idx not in self.state:
            self.state[line.idx] = _ScoutLine()
        return self.state[line.idx]

    def queries_for(self, ep: Any, line: Line) -> list:
        """Value query first (top stated literals, then the shelf noun), then its relaxations."""
        cap = ep.contract.token_cap
        if line.guessed:
            return [q for q in line.query_alts if q]
        values: list = []
        for c in line.constraints:
            if c.active and c.op in ("eq", "in"):
                for v in c.vals:
                    if v not in values:
                        values.append(v)
        values = values[:2]
        noun = content_tokens(line.shelf_words or line.words)
        value_tokens: list = []
        for v in values:
            for t in content_tokens(v):
                if t not in value_tokens and t not in noun:
                    value_tokens.append(t)
        out = []
        base = value_tokens + noun
        out.append(" ".join(_dedupe(base)[:cap]))
        relaxed = list(value_tokens)
        for _ in range(2):
            if not relaxed:
                break
            relaxed = relaxed[1:]
            out.append(" ".join(_dedupe(relaxed + noun)[:cap]))
        out.append(" ".join(_dedupe(noun)[:cap]))
        out.append(" ".join(_dedupe(content_tokens(line.words))[:cap]))
        return [q for q in _dedupe(out) if q.strip()]

    def candidates(self, ep: Any, line: Line, menu: bool = False) -> list:
        """Rows on the label, stock not false, q * price <= cap; titles with stated value tokens first; on
        cheapest lines the 2 cheapest first; then diversified by merchant / maker / first two title tokens.
        menu (h2): the label's over-cap rows follow, cheapest first. They are read for the question menu only
        (the shelf's attribute keys); the checker's budget check keeps them out of any basket."""
        st = self.st(line)
        pool = st.best_ids if line.guessed else list(line.table)
        rows = [ep.catalog[i] for i in pool if i in ep.catalog and i not in st.failed_ids]
        cap = ep.ledger.cap_c
        value_tokens = {t for c in line.constraints if c.active for v in c.vals for t in content_tokens(v)}
        out = []
        over_cap = 0
        over: list = []
        for r in rows:
            if r.stock is False and ep.ledger.in_stock:
                continue
            if line.shelf_label and r.shelf and canon_key(r.shelf) != canon_key(line.shelf_label):
                continue
            if cap is not None and r.price_c is not None and line.qty * r.price_c > cap:
                over_cap += 1
                over.append((r.price_c, pool.index(r.id), r.id))
                continue
            title_hits = len(value_tokens & set(text_tokens(r.title)))
            out.append((-title_hits, r.price_c if (line.cheapest and r.price_c is not None) else 0,
                        pool.index(r.id), r.id))
        out.sort()
        if not out and over_cap and check_cap_unit(ep):
            return self.candidates(ep, line, menu)       # the cap was written in major units (F13)
        ids = [ident for *_, ident in out]
        head = ids[:1]
        if line.cheapest:
            priced = sorted((i for i in ids if ep.catalog[i].price_c is not None),
                            key=lambda i: (ep.catalog[i].price_c, ids.index(i)))
            head += [i for i in priced[:2] if i not in head]

        def sig(ident: str) -> tuple:
            r = ep.catalog[ident]
            return canon_key(r.merchant or r.maker or ""), tuple(text_tokens(r.title)[:2])

        seen_sig = {sig(i) for i in head}
        fresh, rest = [], []
        for ident in ids:
            if ident in head:
                continue
            (rest if sig(ident) in seen_sig else fresh).append(ident)
            seen_sig.add(sig(ident))
        return head + fresh + rest + ([ident for *_, ident in sorted(over)] if menu else [])

    def next_query(self, ep: Any, line: Line, st: "_ScoutLine") -> Optional[str]:
        untried = [q for q in self.queries_for(ep, line) if q not in st.tried]
        if line.guessed:              # v3 (e_q contract): probes ranked by RequestParser.probe_key
            enough = st.best_key is not None and st.best_key[0] and st.best_key[2] >= self.GUESS_ENOUGH_HITS
            if untried and not enough and len(st.tried) < self.GUESS_PROBES:
                return untried[0]
            return None
        if (st.hits == 0 or not self.candidates(ep, line)) and untried and len(st.tried) < 4:
            return untried[0]
        return None

    def n_l(self, ep: Any) -> int:
        return max(1, int(ep.plan.nums.get("n_l") or 1))

    def open_intent(self, ep: Any, line: Line, want: int, why: str, menu: bool = False) -> Optional[Intent]:
        """OPEN(need=attrs) of the next unopened candidates: as many as one call can open (the batch limit),
        one id without a batch tool."""
        st = self.st(line)
        chunk = max(1, min(want, ep.plan.batch_k()))
        ids = [i for i in self.candidates(ep, line, menu) if i not in line.opened][:chunk]
        if not ids:
            return None
        return Intent(IK_OPEN, {"ids": ids, "need": NEED_ATTRS}, P_EXPLORE, line.idx,
                      why=f"scout line {line.idx}: {why} ({len(ids)})")

    def next_intent(self, ep: Any) -> Optional[Intent]:
        lines = ep.ledger.lines
        for line in lines:                                   # pass 1: query, label, first open
            st = self.st(line)
            if st.opened or st.exhausted:
                continue
            query = self.next_query(ep, line, st)
            if query is not None:
                return Intent(IK_SEARCH, {"query": query, "qkind": "q:value" if not st.tried else "q:relaxed"},
                              P_EXPLORE, line.idx, why=f"scout line {line.idx}: value query '{query}'")
            if st.open_tries >= self.OPEN_TRIES:
                st.exhausted = True
                continue
            it = self.open_intent(ep, line, self.n_l(ep), "open candidates")
            if it is None:
                st.exhausted = True
                continue
            return it
        n_l = self.n_l(ep)
        calls = ceil_div(n_l, max(1, int(ep.plan.nums.get("k") or 1)))
        growing = [l for l in lines if self.st(l).opened and not self.st(l).exhausted
                   and self.st(l).open_fails < self.GROW_FAILS]
        for line in sorted(growing, key=lambda l: (len(l.opened), l.idx)):    # pass 2: round-robin to n_l
            st = self.st(line)
            if len(line.opened) < n_l and st.open_calls < calls:
                it = self.open_intent(ep, line, n_l - len(line.opened), "grow the menu")
                if it is not None:
                    return it
                st.open_calls = calls                        # no candidate left
        if ep.plan.nums.get("free", 0) >= self.THIN_FREE:    # pass 3: one more batch on a thin menu
            for line in growing:
                st = self.st(line)
                if not st.grown and ep.elicitor.usable_keys(ep, line) < self.THIN_MENU:
                    st.grown = True
                    it = self.open_intent(ep, line, n_l, "thin menu")
                    if it is not None:
                        return it
        return None

    def _h2_topup(self, ep: Any) -> Optional[Intent]:
        """Menu readiness before the first question (h2): a line whose menu holds fewer than 2 listings gets
        up to two more opens (EXPLORE), from rows found since scouting (feasibility probes, relaxed queries),
        then the label's over-cap rows. Real races: a need key is on the menu of 0% of need-lines with no
        listing opened, 54% with one, 66-70% with two or more. Only while the planner keeps every unsent
        message and the reserve affordable and the commit window is open, so a top-up never ends ELICIT."""
        plan, b = ep.plan, ep.budget
        if not ep.elicitor.msg_room(ep):
            return None                                   # no question can follow: no menu to ready
        if plan.u() >= plan.nums.get("u_c_soft", 10 ** 6) or plan.explore_room() < 1 or \
                plan.rem() - 1 < plan.need_finish() + plan.R + plan.protected() + max(0, b.M - b.msgs_used):
            return None
        want = min(2, self.n_l(ep))
        for line in ep.ledger.lines:
            st = self.st(line)
            have = len(ep.elicitor.menu_items(ep, line))
            if have < want and st.topups < 2 and st.open_fails < self.GROW_FAILS:
                it = self.open_intent(ep, line, self.n_l(ep) - have, "menu top-up before asking", menu=True)
                if it is not None:
                    st.topups += 1
                    return it
        return None

    def after(self, ep: Any, intent: Intent, dec: Decoded) -> None:
        if intent.line is None or intent.line >= len(ep.ledger.lines):
            return
        line = ep.ledger.lines[intent.line]
        st = self.st(line)
        if intent.kind == IK_SEARCH:
            query = intent.payload.get("query", "")
            st.tried.append(query)
            if dec.kind == D_RECORDS and dec.records:
                n = dec.meta.get("n") if isinstance(dec.meta.get("n"), int) else len(dec.records)
                st.counts[query] = n
                st.hits += len(dec.records)
                ids = [i for i in (ep.fields.extract(dec.tool, r).get(FR_ID) for r in dec.records) if isinstance(i, str)]
                key = ep.parser.probe_key(query, ids, n) if line.guessed else None
                if not line.guessed or st.best_key is None or key > st.best_key:
                    st.best_key = key        # guessed lines: the probe whose rows name its word, not the most hits
                    st.qkey = ep.executor.last_qkey
                    st.best_ids = ids
                    self.learn_label(ep, line, dec)
            elif dec.kind == D_RECORDS:
                st.counts[query] = 0
            elif dec.kind == D_BLOCKED and dec.gid in ("DET", "BUILD"):
                st.exhausted = True
        elif intent.kind == IK_OPEN:
            ids = intent.payload.get("ids") or []
            first = not st.opened
            if first:
                st.open_tries += 1
            else:
                st.open_calls += 1
            got: list = []
            if dec.ok and dec.kind in (D_RECORDS, D_RECORD):
                back = {ep.fields.extract(dec.tool, r).get(FR_ID) for r in dec.records}
                got = [i for i in ids if i in back] or (ids[:1] if dec.kind == D_RECORD and len(ids) == 1 else [])
            if got:
                st.opened = True
                for ident in got:
                    if ident not in line.opened:
                        line.opened.append(ident)
                st.failed_ids |= {i for i in ids if i in dec.absent_ids}
                return
            if dec.kind == D_BLOCKED and dec.gid in ("DET", "BUILD"):
                st.exhausted = True
            if not first:
                st.open_fails += 1
            if len(ids) == 1 or dec.kind not in (D_EMPTY, D_BLOCKED, D_WORLD_FAULT):
                st.failed_ids |= set(ids)              # a rejected batch is retried at the halved limit

    def learn_label(self, ep: Any, line: Line, dec: Decoded) -> None:
        words = set(content_tokens(line.words + " " + line.shelf_words))
        tally: dict = {}
        for rec in dec.records:
            ident = ep.fields.extract(dec.tool, rec).get(FR_ID)
            r = ep.catalog.get(ident)
            if r is None or not r.shelf:
                continue
            tally[r.shelf] = tally.get(r.shelf, 0) + 1
        if not tally:
            return
        best = max(tally, key=lambda lab: (tally[lab] * (1 + len(words & set(content_tokens(lab)))), -len(lab)))
        overlap = len(words & set(content_tokens(best)))
        line.shelf_label = best
        line.shelf_conf = 2 if (tally[best] >= 3 and overlap > 0) else 1
        ep.trace.decision("shelf", line=line.idx, label=best, conf=line.shelf_conf, tally=tally)

    def done(self, ep: Any) -> bool:
        return all(self.st(line).exhausted for line in ep.ledger.lines)

def _dedupe(items: list) -> list:
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out

_DEC_INF = 10 ** 12                     # the floor of a line whose every enumerated on-shelf row is unavailable
_DEC_SAFE_EVENTS = ("budget", "quantity", "drop", "change", "stock_out")   # event kinds that never lower a floor

def e1_holds(lines: list, cap_c: Optional[int]) -> bool:
    """E1: sum over lines of qty * proven lower bound exceeds the cap. Unproven bounds count 0."""
    if cap_c is None:
        return False
    return sum(line.qty * max(0, line.LB_c if line.LB_proof else 0) for line in lines) > cap_c

def _dec_on_shelf(line: Line, r: Record) -> bool:
    """On the line's shelf; an unknown shelf counts as on it (it can only lower a floor)."""
    return not (line.shelf_label and r.shelf and canon_key(r.shelf) != canon_key(line.shelf_label))

def _dec_available(ep: Any, r: Record) -> bool:
    return not (r.stock is False and ep.ledger.in_stock)

def _dec_line_ids(line: Line) -> list:
    return list(dict.fromkeys(list(line.opened) + list(line.table)))

def _dec_hard(ep: Any, line: Line, ops: tuple = ()) -> list:
    """Constraints a proof may use: active, not pair_pending, references once resolved, never a reference span."""
    refs = ep.ledger.refs
    spans = [norm_space(r.span) for r in refs if r.span]
    out = []
    for c in line.constraints:
        if not c.active or c.op == "pair_pending" or (ops and c.op not in ops):
            continue
        span = norm_space(c.span)
        if c.source == "reference":
            if not any(r.rid == c.ref_id and r.status == "resolved" for r in refs):
                continue
        elif span and any(s in span or span in s for s in spans):
            continue
        out.append(c)
    return out

def _dec_searchable(value: Any) -> bool:
    """Tokens that can narrow a search safely: letters only, no one-character tokens."""
    toks = content_tokens(value)
    return bool(toks) and all(t.isalpha() and len(t) > 1 for t in toks)

def _dec_rec_for(ep: Any, query: str) -> Optional[QueryRec]:
    """The text query already run with these tokens in any order (AND search: the same rows), the most rows read;
    v1 compared the text, so a value query the coverage had read fully never became an enumeration floor."""
    want, best = frozenset(text_tokens(query)), None
    for rec in ep.queries.recs.values():
        if want and rec.query_text and frozenset(text_tokens(rec.query_text)) == want \
                and (best is None or len(rec.ids) > len(best.ids)):
            best = rec
    return best

def _dec_enumerated(rec: Optional[QueryRec]) -> bool:
    """Fully read: count == distinct ids read, not capped, no continuation left."""
    return (rec is not None and isinstance(rec.count, int) and not rec.capped and not rec.has_cont
            and len(set(rec.ids)) == rec.count)

def _dec_cents(ep: Any, value: Any) -> Optional[int]:
    """A money argument as sent (world units) back in integer cents."""
    n = as_number(value)
    return None if n is None else (int(round(n * 100)) if ep.fields.price_major else int(math.floor(n + 1e-9)))

def _dec_slot(ep: Any, tool: str, slot: str) -> Optional[str]:
    """The argument of a slot whose meaning is agreed (conf >= 2): proofs never rest on a guess."""
    arg = ep.binder.slot_arg(tool, slot)
    tb = ep.binder.binding.tools.get(tool)
    return arg if arg and tb is not None and tb.slots.get(arg, (None, 0))[1] >= 2 else None

def _dec_pages_left(ep: Any, rec: QueryRec) -> int:
    """Pages still to read; 10**6 (never) without a count or once every counted id was read yet a continuation stays."""
    left = rec.count - len(set(rec.ids)) if isinstance(rec.count, int) else 0
    return ceil_div(left, ep.contract_learner.page_size()) if left > 0 else 10 ** 6

class Feasibility(Engine):
    name = "feasibility"

    def __init__(self, ep: Any = None):
        super().__init__(ep)
        self.spent = 0                   # probe steps sent
        self.tried: set = set()          # probe keys done or abandoned
        self.proofs: dict = {}           # line idx -> floor record {how, lb, u, major, cids, vals, ids, probe}
        self.harvested: set = set()      # lines whose enumeration was evaluated
        self.unanchored: dict = {}       # line idx -> (anchor key, floor record) of a zero-count bracket
        self.anchors: dict = {}          # (tool, shelf, maker) -> the filter answered rows that obey it
        self.pending: Optional[dict] = None
        self.sent_at = 0

    def budget_steps(self, ep: Any) -> int:
        return min(8, ceil_div(2 * ep.budget.B, 10))

    def _open(self, ep: Any) -> bool:
        led = ep.ledger
        return (led.cap_c is not None and bool(led.lines) and led.source not in ("fallback", "none")
                and led.decline_rule.get("allowed", True) is not False and not ep.events.decline_locked
                and ep.llm.allowed("AUDIT_E1"))

    def ceiling_bracket(self, ep: Any, line: Line) -> Optional[int]:
        """c = m (inclusive ceilings) or m + 1, m = floor((cap - sum_{o != l} q_o * LB_o) / q_l): 0 rows => LB > m."""
        cap = ep.ledger.cap_c
        if cap is None or line.qty <= 0:
            return None
        fl = self.floors(ep)
        others = sum(o.qty * fl.get(o.idx, 0) for o in ep.ledger.lines if o.idx != line.idx)
        return (cap - others) // line.qty + (0 if ep.contract.ceiling_inclusive else 1)

    def value_plan(self, ep: Any, line: Line) -> Optional[dict]:
        """Two searchable eq values in one query (as the scout), else one query per value of an allowed set."""
        hard = [c for c in _dec_hard(ep, line, ("eq", "in")) if c.vals]
        singles = [c for c in hard if c.op == "eq" and len(c.vals) == 1 and _dec_searchable(c.vals[0])][:2]
        sets = [c for c in hard if len(c.vals) <= 3 and all(_dec_searchable(v) for v in c.vals)][:1]
        if not singles and not sets:
            return None
        noun = content_tokens(line.shelf_words or line.words)

        def query(vals: list) -> str:
            toks = [t for v in vals for t in content_tokens(v) if t not in noun]
            return " ".join(_dedupe(toks + noun)[:ep.contract.token_cap])
        if singles:
            vals = [c.vals[0] for c in singles]
            return {"cids": [c.cid for c in singles], "vals": vals, "queries": [query(vals)]}
        return {"cids": [sets[0].cid], "vals": list(sets[0].vals), "queries": [query([v]) for v in sets[0].vals]}

    def seen_min(self, ep: Any, line: Line) -> Optional[int]:
        """Cheapest on-shelf available value-query row: the most a floor can prove (an estimate, never a proof)."""
        plan = self.value_plan(ep, line)
        recs = [r for r in ([_dec_rec_for(ep, q) for q in plan["queries"]] if plan else []) if r] or \
            [r for r in ep.queries.for_line(line.idx) if r.kind == "q:value"][:1]
        ids = [i for r in recs for i in r.ids] if plan and recs else _dec_line_ids(line)
        rows = [r for r in (ep.catalog.get(i) for i in ids) if r is not None and _dec_on_shelf(line, r)]
        prices = [r.price_c for r in rows if r.price_c is not None and _dec_available(ep, r)]
        return min(prices) if prices else (_DEC_INF if rows else None)

    def potential(self, ep: Any) -> int:
        fl = self.floors(ep)
        return sum(l.qty * (fl[l.idx] or (self.seen_min(ep, l) or 0 if l.shelf_conf >= 2 and not l.guessed else 0))
                   for l in ep.ledger.lines)

    def triggered(self, ep: Any, line: Line) -> bool:
        check_cap_unit(ep)                 # a cap read 100 times too small is rescaled once (F13)
        if not self._open(ep) or line.shelf_conf < 2 or line.guessed:
            return False
        if self.potential(ep) > ep.ledger.cap_c:
            return True
        rec = next((r for r in ep.queries.for_line(line.idx) if r.kind == "q:value"), None)
        rows = [r for r in (ep.catalog.get(i) for i in (rec.ids if rec else [])) if r and _dec_on_shelf(line, r)]
        window = min(ep.contract_learner.window(), 3 * ep.contract_learner.page_size())
        return (bool(rows) and isinstance(rec.count, int) and rec.count <= window and not any(
            _dec_available(ep, r) and r.price_c is not None and line.qty * r.price_c <= ep.ledger.cap_c for r in rows))

    def next_intent(self, ep: Any) -> Optional[Intent]:
        self.pending = None
        if not self._open(ep) or self.spent >= self.budget_steps(ep) or self.e1(ep) is not None:
            return None
        anchor = self._anchor_intent(ep)
        if anchor is not None or self.potential(ep) <= ep.ledger.cap_c:
            return anchor
        lines = sorted((l for l in ep.ledger.lines if l.shelf_conf >= 2 and not l.guessed),
                       key=lambda l: (-(l.qty * (self.seen_min(ep, l) or 0)), l.idx))
        for line in lines:
            it = (self._enum_intent(ep, line) or self._bracket_intent(ep, line, False)
                  or self._bracket_intent(ep, line, True))
            if it is not None:
                return it
        return None

    def _probe(self, ep: Any, key: tuple, line: Line, kind: str, payload: dict, why: str) -> Intent:
        self.pending = {"key": key, "line": line.idx, "payload": dict(payload)}
        self.sent_at = ep.transport.n
        return Intent(kind, payload, P_PROBE, line.idx, why=f"feasibility: {why} for line {line.idx}")

    def _enum_intent(self, ep: Any, line: Line) -> Optional[Intent]:
        plan = self.value_plan(ep, line)
        if plan is None or line.idx in self.harvested:
            return None
        for q in plan["queries"]:
            key = ("enum", line.idx, q)
            rec = _dec_rec_for(ep, q)
            if not q or key in self.tried:
                return None
            if rec is None:
                return self._probe(ep, key, line, IK_SEARCH, {"query": q, "qkind": "q:value"}, f"value query '{q}'")
            if _dec_enumerated(rec):
                continue
            if not (rec.has_cont and not rec.capped and rec.count <= ep.contract_learner.window()
                    and _dec_pages_left(ep, rec) <= self.budget_steps(ep) - self.spent):
                self.tried.add(key)        # cannot be read fully within the screen: no enumeration proof
                return None
            return self._probe(ep, key, line, IK_SEARCH, {"query": rec.query_text, "cont": rec.next_cont,
                                                          "qkey": rec.canon_args, "qkind": "q:value"},
                               f"page the value query '{rec.query_text}'")   # its own text: tokens are opaque
        return None

    def _maker_value(self, ep: Any, line: Line) -> Optional[str]:
        """A hard value that is a maker on the shelf and shows on no other maker's listing there."""
        shelf = [r for r in ep.catalog.values() if r.shelf and line.shelf_label
                 and canon_key(r.shelf) == canon_key(line.shelf_label)]
        for c in _dec_hard(ep, line, ("eq",)):
            v = canon_key(c.vals[0]) if len(c.vals) == 1 else ""
            toks = set(text_tokens(v))
            own = [r for r in shelf if r.maker and canon_key(r.maker) == v]
            if own and toks and not any(not (r.maker and canon_key(r.maker) == v) and toks <= set(text_tokens(
                    r.title + " " + " ".join(str(x) for vals in r.attrs.values() for _, x in vals))) for r in shelf):
                return own[0].maker
        return None

    def _bracket_intent(self, ep: Any, line: Line, maker: bool) -> Optional[Intent]:
        key = ("maker" if maker else "bracket", line.idx)
        tool = ep.binder.tool_for(ROLE_FILTER) or ep.binder.tool_for(ROLE_SEARCH)
        slots = (SLOT_SHELF, SLOT_CEILING) + ((SLOT_MAKER,) if maker else ())
        c = self.ceiling_bracket(ep, line)
        if key in self.tried or tool is None or not all(_dec_slot(ep, tool, s) for s in slots) \
                or not line.shelf_label or c is None or c <= 0:
            return None
        payload = {"shelf": line.shelf_label, "ceiling": c, "qkind": "q:bracket"}
        rows = [r for r in ep.catalog.values() if r.shelf and canon_key(r.shelf) == canon_key(line.shelf_label)]
        if maker:
            v = self._maker_value(ep, line)
            if v is None:
                return None
            payload["maker"] = v
            rows = [r for r in rows if r.maker and canon_key(r.maker) == canon_key(v)]
        m = c if ep.contract.ceiling_inclusive else c - 1
        if any(r.price_c is not None and r.price_c <= m and _dec_available(ep, r) for r in rows):
            self.tried.add(key)            # a seen row already fits under the ceiling: the probe cannot prove
            return None
        return self._probe(ep, key, line, IK_FILTER, payload, f"{'maker ' + payload['maker'] if maker else 'shelf'} "
                                                              f"at or under {c}")

    def _anchor_intent(self, ep: Any) -> Optional[Intent]:
        """A zero-count bracket needs the same filter to answer the cheapest seen row (the anchor)."""
        for idx in list(self.unanchored):
            akey, rec = self.unanchored[idx]
            line = ep.ledger.lines[idx] if idx < len(ep.ledger.lines) else None
            key = ("anchor",) + akey
            if line is not None and self.anchors.get(akey):
                del self.unanchored[idx]
                self._record(ep, line, rec)
                continue
            maker = rec.get("maker")
            rows = [r for r in ep.catalog.values() if line is not None and r.shelf and r.price_c is not None
                    and canon_key(r.shelf) == akey[1] and _dec_available(ep, r)
                    and (not maker or (r.maker and canon_key(r.maker) == akey[2]))]
            if line is None or key in self.tried or not rows or self.spent >= self.budget_steps(ep):
                del self.unanchored[idx]
                continue
            payload = {"shelf": line.shelf_label, "ceiling": min(r.price_c for r in rows) + (
                0 if ep.contract.ceiling_inclusive else 1), "qkind": "q:probe"}
            if maker:
                payload["maker"] = maker
            return self._probe(ep, key, line, IK_FILTER, payload, "anchor the bracket")
        return None

    def after(self, ep: Any, intent: Intent, dec: Decoded) -> None:
        p, self.pending = self.pending, None
        if p is None or p["line"] >= len(ep.ledger.lines):
            return
        self.spent += ep.transport.n > self.sent_at
        line, key = ep.ledger.lines[p["line"]], p["key"]
        if key[0] == "enum":
            if dec.kind != D_RECORDS:
                self.tried.add(key)
            return
        self.tried.add(key)
        qrec = ep.queries.get(ep.executor.last_qkey or "") if dec.kind == D_RECORDS else None
        maker = p["payload"].get("maker")
        akey = key[1:] if key[0] == "anchor" else (qrec.tool if qrec else "", canon_key(line.shelf_label),
                                                     canon_key(maker or ""))
        if qrec is None:
            if key[0] == "anchor":
                self._unanchor(ep, akey, "the anchor probe failed")
            return
        ceiling = _dec_cents(ep, qrec.args.get(ep.binder.slot_arg(qrec.tool, SLOT_CEILING) or ""))
        if dec.records:
            obeys = all(self._row_obeys(ep, dec.tool, r, akey, ceiling) for r in dec.records)
            self.anchors[akey] = self.anchors.get(akey, False) or obeys
            if key[0] == "anchor" and obeys:
                for idx, (k, rec) in list(self.unanchored.items()):
                    if k == akey:
                        del self.unanchored[idx]
                        self._record(ep, ep.ledger.lines[idx], rec)
            elif key[0] == "anchor":
                self._unanchor(ep, akey, "the anchor rows break the filter")
            return
        if key[0] == "anchor":
            self._unanchor(ep, akey, "the anchor probe found nothing")
            return
        if dec.meta.get("has_cont") or dec.meta.get("n") not in (0, None) or ceiling is None:
            return
        how = "maker bracket" if maker else "bracket"
        rec = {"how": how, "lb": ceiling + (1 if ep.contract.ceiling_inclusive else 0), "maker": maker,
               "probe": f"{how} {line.shelf_label} under {ceiling}" + (f" by {maker}" if maker else "")}
        cons = [c for c in _dec_hard(ep, line, ("eq",)) if maker and len(c.vals) == 1
                and canon_key(c.vals[0]) == canon_key(maker)][:1]
        rec.update(cids=[c.cid for c in cons], vals=[c.vals[0] for c in cons])
        if self.anchors.get(akey):
            self._record(ep, line, rec)
        else:
            self.unanchored[line.idx] = (akey, rec)

    @staticmethod
    def _row_obeys(ep: Any, tool: str, raw: dict, akey: tuple, ceiling: Optional[int]) -> bool:
        """The row shows the shelf (and maker) asked for and a price under the ceiling, all readable."""
        f = ep.fields.extract(tool, raw)
        shelf, maker, price = f.get(FR_SHELF), f.get(FR_MAKER), f.get(FR_PRICE)
        return (isinstance(shelf, str) and canon_key(shelf) == akey[1]
                and (not akey[2] or (isinstance(maker, str) and canon_key(maker) == akey[2]))
                and (ceiling is None or (isinstance(price, int) and price <= ceiling)))

    def _unanchor(self, ep: Any, akey: tuple, why: str) -> None:
        self.anchors[akey] = False
        for idx, (k, rec) in list(self.unanchored.items()):
            if k == akey:
                del self.unanchored[idx]
                ep.trace.decision("proof", stage="floor_rejected", line=idx, how=rec["how"], why=why)

    def _harvest(self, ep: Any) -> None:
        """Every fully read value query becomes an enumeration floor, once per line."""
        for line in ep.ledger.lines:
            plan = self.value_plan(ep, line)
            recs = [_dec_rec_for(ep, q) for q in plan["queries"]] if plan else []
            if not recs or line.idx in self.harvested or line.shelf_conf < 2 or not all(map(_dec_enumerated, recs)):
                continue
            self.harvested.add(line.idx)
            ids = list(dict.fromkeys(i for r in recs for i in r.ids))
            lb = self._enum_floor(ep, line, ids, plan["vals"])
            read = [ep.catalog[i].seen_rows_at[0] for i in ids if i in ep.catalog and ep.catalog[i].seen_rows_at]
            if lb is None:
                ep.trace.decision("proof", stage="floor_none", line=line.idx, how="enumeration", rows=len(ids))
            else:
                self._record(ep, line, {"how": "enumeration", "lb": lb, "cids": plan["cids"], "vals": plan["vals"],
                                        "ids": ids, "probe": " | ".join(plan["queries"]), "u": min(read, default=None)})

    def _enum_floor(self, ep: Any, line: Line, ids: list, vals: list) -> Optional[int]:
        """min price of the enumerated on-shelf available rows and of line rows with a value in the title."""
        rows = [r for r in (ep.catalog.get(i) for i in ids) if r is not None and _dec_on_shelf(line, r)]
        toks, known = [set(content_tokens(v)) for v in vals], set(ids)
        if not rows:
            return None
        rows += [r for r in (ep.catalog.get(i) for i in _dec_line_ids(line) if i not in known)
                 if r is not None and _dec_on_shelf(line, r)
                 and any(t and t <= set(text_tokens(r.title)) for t in toks)]
        avail = [r for r in rows if _dec_available(ep, r)]
        if any(r.price_c is None for r in avail):
            return None
        return min(r.price_c for r in avail) if avail else _DEC_INF

    def _record(self, ep: Any, line: Line, rec: dict) -> None:
        rec = dict(rec, u=ep.transport.n if rec.get("u") is None else rec["u"], major=bool(ep.fields.price_major))
        if not self.valid(ep, line) or rec["lb"] > line.LB_c:
            self.proofs[line.idx] = rec
            line.LB_c, line.LB_proof = rec["lb"], rec["how"]
        ep.trace.decision("proof", stage="floor", line=line.idx, how=rec["how"], floor_c=rec["lb"],
                          probe=clip_text(rec.get("probe"), 160))

    def valid(self, ep: Any, line: Line) -> bool:
        """The floor still holds (see the header); a floor written elsewhere is taken as given."""
        p = self.proofs.get(line.idx)
        if not line.LB_proof or p is None:
            return bool(line.LB_proof)
        cons = {c.cid: c for c in line.constraints}
        why = None
        if p["major"] != bool(ep.fields.price_major):
            why = "price unit changed"
        elif any(e.u >= p["u"] and e.kind not in _DEC_SAFE_EVENTS for e in ep.events.queue):
            why = "an event may have changed prices"
        elif any(cons.get(cid) is None or not cons[cid].active or v not in cons[cid].vals
                 for cid, v in zip(p.get("cids") or [], p.get("vals") or [])):
            why = "its constraint changed"
        elif p["how"] == "enumeration":
            lb = self._enum_floor(ep, line, p["ids"], p["vals"])
            if lb is None:
                why = "its rows changed"
            else:
                line.LB_c = min(line.LB_c, lb)
        if why is None:
            return True
        line.LB_c, line.LB_proof = 0, None
        del self.proofs[line.idx]
        ep.trace.decision("proof", stage="floor_void", line=line.idx, how=p["how"], why=why)
        return False

    def floors(self, ep: Any) -> dict:
        self._harvest(ep)
        return {line.idx: (line.LB_c if self.valid(ep, line) else 0) for line in ep.ledger.lines}

    def e1(self, ep: Any) -> Optional[Proof]:
        led = ep.ledger
        fl = self.floors(ep)
        if not e1_holds(led.lines, led.cap_c) or led.cap_c in (led.stated_msgs, led.stated_steps):   # misread (v5)
            return None
        used = [i for i, v in fl.items() if v > 0]
        recs = [self.proofs.get(i) or {} for i in used]
        return Proof(kind="E1", lines=used, ids=[x for r in recs for x in r.get("ids", [])][:40],
                     numbers={"cap_c": led.cap_c, "floors": {str(i): fl[i] for i in used},
                              "total_c": sum(led.lines[i].qty * fl[i] for i in used),
                              "how": {str(i): led.lines[i].LB_proof for i in used}},
                     quotes=[r.get("probe", "") for r in recs])

    def done(self, ep: Any) -> bool:
        return self.spent >= self.budget_steps(ep) or self.e1(ep) is not None

_ELI_TOP = 6                  # Borda depth: points 6..1
_ELI_MIN_SCORE = 0.10
_ELI_Q_MAX = 120
_ELI_PROBE_MAX = 5            # an unknown M probes a 4th and 5th message (DESIGN 3.2)
_ELI_TWIN = 0.6
_ELI_LIST_RE = re.compile(r"[,/;|]")
_ELI_UNIT_RE = re.compile(r"^\d+(?:[.,]\d+)?[a-z]{1,5}$")
_ELI_DISCLOSED = "disclosure"

def _eli_stems(text: Any) -> set:
    """Tokens with a plural s stripped (overlap tests only)."""
    return {t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in text_tokens(text)}

def _eli_run(toks: list, sub: list) -> int:
    """Index of the first contiguous run of sub in toks, or -1."""
    n = len(sub)
    for i in range(len(toks) - n + 1 if n else 0):
        if toks[i:i + n] == sub:
            return i
    return -1

def _eli_clean(text: Any) -> str:
    return " ".join(str(text or "").replace("_", " ").replace("?", " ").split()).lower()

def _eli_shared(a: dict, b: dict, min_both: int = 1) -> float:
    """Share of co-carrying items where both keys hold an equal (canonical) value; -1 below min_both items.
    Equality, not token sharing: 'aged brass' / 'brass' are two attributes."""
    both = [i for i in a if i in b]
    if len(both) < min_both:
        return -1.0
    return sum(1 for i in both if a[i] & b[i]) / len(both)

def _eli_affixes(recs: list) -> set:
    """First/last tokens whose removal maps >= 2 keys onto existing keys with equal values (>= 50%)."""
    keyed: dict = {}
    for r in recs:
        for canon, pairs in r.attrs.items():
            vals = {canon_key(v) for _, v in pairs} - {""}
            if vals:
                keyed.setdefault(canon, {})[r.id] = vals
    support: dict = {}
    for k, vals in keyed.items():
        toks = k.split()
        if len(toks) < 2:
            continue
        for t, rest in ((toks[0], toks[1:]), (toks[-1], toks[:-1])):
            other = keyed.get(" ".join(rest))
            if other is not None and _eli_shared(vals, other) >= 0.5:
                support.setdefault(t, set()).add(k)
    return {t for t, ks in support.items() if len(ks) >= 2}

def _eli_base(canon: str, affixes: set) -> str:
    toks = canon.split()
    while len(toks) > 1 and toks[0] in affixes:
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in affixes:
        toks = toks[:-1]
    return " ".join(toks)

def _eli_spoken(raws: dict, affixes: set, base: str) -> str:
    """The most frequent raw spelling without the affix, '_' -> space; one whose canon is the base wins."""
    cands = [_eli_base(_eli_clean(r), affixes) for r, _ in sorted(raws.items(), key=lambda kv: (-kv[1], str(kv[0])))]
    return next((c for c in cands if canon_key(c) == base), cands[0] if cands else base)

def _eli_fold(fams: dict, lstems: set, min_both: int) -> None:
    """Subset twins are one attribute: b folds into a, or a into b when b's extra tokens are label words."""
    order = sorted(fams, key=lambda k: (len(k.split()), k))
    for a in order:
        for b in order:
            if a not in fams or b not in fams or not set(a.split()) < set(b.split()):
                continue
            ta, tb = ({r: {canon_key(x) for x in v} for r, v in fams[k][2].items()} for k in (a, b))
            if _eli_shared(ta, tb, min_both) < _ELI_TWIN:
                continue
            extra = _eli_stems(" ".join(t for t in b.split() if t not in a.split()))
            win, lose = (b, a) if extra <= lstems else (a, b)
            wv, wr, wi = fams[win]
            lv, lr, li = fams.pop(lose)
            wv.update(lv)
            for raw, n in lr.items():
                wr[raw] = wr.get(raw, 0) + n
            for rid, vs in li.items():
                got = wi.setdefault(rid, [])
                got += [v for v in vs if v not in got]

def _eli_meta(ep: Any, by_item: dict) -> str:
    """Why a key is meta by the shape of its values (DESIGN 5.10, plus variant selectors), or ''."""
    firsts, same, price, code, lists, multi = [], 0, 0, 0, 0, 0
    for rid, vs in by_item.items():
        v = vs[0]
        firsts.append(v)
        r = ep.catalog.get(rid)
        if r is not None:
            if canon_key(v) in {canon_key(x) for x in (r.title, r.merchant, r.maker, r.shelf, r.id) if x}:
                same += 1
            p = r.price_c
            if p and p >= 100 and any(abs(n - p) < 1e-6 or abs(n * 100 - p) < 0.5 for n, *_ in numbers_in(v)):
                price += 1
        low = v.lower()
        if " " not in v and len(v) >= 4 and re.search(r"[a-z]", low) and re.search(r"\d", low) \
                and not _ELI_UNIT_RE.match(low):
            code += 1
        lists += len([p for p in _ELI_LIST_RE.split(v) if p.strip()]) >= 4
        multi += " / " in v
    n = len(firsts)
    uniq = len({canon_key(v) for v in firsts}) / n
    med = statistics_median([len(v) for v in firsts])
    if n >= 3 and uniq > 0.9 and med > 25:
        return "unique long values"
    if code > 0.6 * n and (uniq >= 0.75 if n >= 3 else min(len(v) for v in firsts) >= 8):
        return "code-like values"
    if same > 0.5 * n:
        return "values repeat the listing identity"
    if price > 0.5 * n:
        return "values hold the price"
    if med > 60:
        return "long values"
    if lists > 0.5 * n:
        return "list values"
    if n >= 3 and multi > 0.5 * n and uniq > 0.9:
        return "variant selector values"
    return ""

class Elicitor(Engine):
    name = "elicitor"

    def __init__(self, ep: Any = None):
        super().__init__(ep)
        self.vals, self.votes, self.hint, self.unseen = {}, {}, {}, {}
        self.meta_votes, self.stated_votes, self.rank_sizes, self.asks = {}, {}, {}, {}
        self.shelf_bad: set = set()        # lines whose label a RANK majority rejected
        self.reranked: set = set()
        self.ranked = self.status_tried = self.loss_logged = False
        self.reask: list = []              # queued narrower re-asks
        self.reasked: set = set()          # (line, key) that had theirs
        self.seen_replies: dict = {}       # normalised reply -> {(line, key)}
        self.stopped = ""                  # why asking ended for good
        self.no_reply = self.proc_sent = 0
        self._affix_cache: tuple = (-1, set())

    def affixes(self, ep: Any, line: Optional[Line] = None) -> set:
        """Learned over every opened record; for a line, minus its label words (they qualify keys)."""
        recs = [r for r in ep.catalog.values() if r.attrs]
        sig = (len(recs), sum(len(p) for r in recs for p in r.attrs.values()))    # a re-open may add attributes
        if self._affix_cache[0] != sig:
            self._affix_cache = (sig, _eli_affixes(recs))
        if line is None:
            return self._affix_cache[1]
        lstems = _eli_stems(line.shelf_label or line.shelf_words or line.words)
        return {t for t in self._affix_cache[1] if _eli_stems(t).isdisjoint(lstems)}

    def menu_items(self, ep: Any, line: Line) -> list:
        label = canon_key(line.shelf_label) if line.shelf_label else ""
        out = []
        for i in list(line.opened) + [i for i in line.table if i not in line.opened]:
            r = ep.catalog.get(i)
            if r is None or not r.attrs or r.chain_page or (label and r.shelf and canon_key(r.shelf) != label):
                continue
            out.append(r)
        return out

    def build_menu(self, ep: Any, line: Line) -> dict:
        items = self.menu_items(ep, line)
        lstems = _eli_stems(line.shelf_label or line.shelf_words or line.words)
        affixes = self.affixes(ep, line)
        fams: dict = {}
        for r in items:
            for canon, pairs in r.attrs.items():
                variants, raws, by_item = fams.setdefault(_eli_base(canon, affixes), ({}, {}, {}))
                variants[canon] = 1
                for raw, value in pairs:
                    raws[raw] = raws.get(raw, 0) + 1
                    v = " ".join(str(value).split())
                    if v:
                        by_item.setdefault(r.id, []).append(v)
        min_both = 2 if len(items) >= 3 else 1
        _eli_fold(fams, lstems, min_both)
        menu, vals = {}, {}
        for base, (variants, raws, by_item) in fams.items():
            if not by_item:
                continue
            firsts = [vs[0] for vs in by_item.values()]
            counts: dict = {}
            for v in firsts:
                counts[canon_key(v)] = counts.get(canon_key(v), 0) + 1
            n = len(firsts)
            ks = KeyStats(canon=base, spoken=_eli_spoken(raws, affixes, base), variants=sorted(variants),
                          cov=n / len(items), distinct=len(counts),
                          div=(1.0 - max(counts.values()) / n) if len(counts) > 1 else 0.0,
                          categorical=1.0 if 2 <= len(counts) <= 60 and sum(map(len, firsts)) / n <= 30 else 0.3,
                          shelf_overlap=len(_eli_stems(base) & lstems) / max(1, len(set(base.split()))))
            ks.meta = bool(_eli_meta(ep, by_item))
            menu[base] = ks
            vals[base] = by_item
        keys = sorted(menu)
        toks = {k: {rid: {canon_key(v) for v in vs} for rid, vs in vals[k].items()} for k in keys}
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if _eli_shared(toks[a], toks[b], min_both) >= _ELI_TWIN:
                    menu[a].twins.add(b)
                    menu[b].twins.add(a)
        for a in keys:
            sa = set(a.split())
            sups = [b for b in keys if sa < set(b.split()) and b not in menu[a].twins and not menu[b].meta
                    and menu[b].cov >= 0.5 * menu[a].cov]
            if sups:
                menu[a].sibling_of = max(sups, key=lambda b: (menu[b].cov, b))
        for c, spoken in self.unseen.get(line.idx, {}).items():
            if c not in menu:
                menu[c] = KeyStats(canon=c, spoken=spoken, unseen=True,
                                   shelf_overlap=len(_eli_stems(c) & lstems) / max(1, len(set(c.split()))))
        line.menu = menu
        self.vals[line.idx] = vals
        self.score(ep, line)
        return menu

    def score(self, ep: Any, line: Line) -> None:
        votes = self.votes.get(line.idx)
        hinted = self.parse_hints(ep, line) if not votes else set()
        seen = any(not k2.unseen and not k2.meta for k2 in line.menu.values())   # else unseen keys keep the full
        for k, ks in line.menu.items():                          # vote (DESIGN 3.3: no candidate -> story keys)
            ks.vote = votes.get(k, 0.0) if votes else (0.5 if k in hinted else 0.0)
            ks.score = round(0.45 * ks.vote * (0.5 if ks.unseen and seen else 1.0) + 0.20 * ks.cov + 0.15 * ks.div
                             + 0.10 * ks.categorical + 0.10 * ks.shelf_overlap, 4)

    def family(self, ep: Any, line: Line, attr: Any) -> Optional[str]:
        """The menu key an attribute name denotes (canon, variant or affix-free form)."""
        c = canon_key(attr)
        if not c:
            return None
        if c in line.menu:
            return c
        for k, ks in line.menu.items():
            if c in ks.variants or c == canon_key(ks.spoken):
                return k
        base = _eli_base(c, self.affixes(ep, line))
        return base if base in line.menu else None

    def parse_hints(self, ep: Any, line: Line) -> set:
        """Menu keys that a validated PARSE story hint names for this line."""
        out = set()
        for h in ep.ledger.hints:
            if isinstance(h, dict) and h.get("line") in (line.idx, None):
                k = self.family(ep, line, h.get("attr_guess"))
                if k:
                    out.add(k)
        return out

    def h1_ref_attrs(self, ep: Any, line: Line) -> list:
        """Attributes the request's references fix on the line (resolved only after ELICIT); h3: only references
        naming a listing (a story sentence parsed as a bare pairing hid a real need)."""
        return [r.attr for r in ep.ledger.refs if r.line == line.idx and r.origin == "request" and r.attr
                and (r.source.get("merchant") or r.source.get("descriptors"))]

    def stated(self, ep: Any, line: Line) -> set:
        """Keys the request fixes: constraint and request-reference attributes, the key holding a stated value on
        most items, and RANK's stated_map majority."""
        out = set(self.stated_votes.get(line.idx, ()))
        out |= {k for k in (self.family(ep, line, a) for a in self.h1_ref_attrs(ep, line)) if k}
        vals = self.vals.get(line.idx, {})
        for c in line.constraints:
            if not c.active:
                continue
            k = self.family(ep, line, c.attr)
            if k:
                out.add(k)
            wanted = [text_tokens(w) for w in c.vals if text_tokens(w)] \
                if c.op in ("eq", "in", "ne") and c.source in ("stated", "review") else []
            best, top = None, 0
            for key, by_item in vals.items():
                if line.menu.get(key) is None or line.menu[key].meta:
                    continue
                n = sum(1 for vs in by_item.values()
                        if any(_eli_run(text_tokens(v), w) >= 0 for v in vs for w in wanted))
                if n > top:
                    best, top = key, n
            if best:
                out.add(best)
        return out

    def usable_keys(self, ep: Any, line: Line) -> int:
        menu = self.build_menu(ep, line)
        stated = self.stated(ep, line)
        return sum(1 for k, ks in menu.items() if not ks.meta and not ks.unseen and k not in stated)

    def top_values(self, line: Line, key: str, n: int) -> list:
        counts: dict = {}
        for vs in self.vals.get(line.idx, {}).get(key, {}).values():
            for v in vs:
                counts[v] = counts.get(v, 0) + 1
        return sorted(counts, key=lambda v: (-counts[v], v))[:n]

    def _h3_ident(self, line: Line, key: str) -> bool:
        """h3: an identity field: values never repeat and most hold a digit or separator ('blue / int:xl'). Real
        meta votes also hit gender, style, warranty and origin keys, hidden needs in real runs."""
        firsts = [vs[0] for vs in self.vals.get(line.idx, {}).get(key, {}).values() if vs]
        return len({canon_key(v) for v in firsts}) == len(firsts) and \
            2 * sum(1 for v in firsts if re.search(r"[\d/:#]", v)) >= len(firsts)

    def _h3_example(self, ep: Any, line: Line, ks: KeyStats) -> str:
        """h3: a value to cite (1-4 words, <= 30 chars, one value: no list separator, quote or non-ASCII) with a word
        of its own; no other line's label word, no menu key's name."""
        own = _eli_stems(self.label(ep, line)) | _eli_stems(f"{ks.canon} {ks.spoken}")
        bad = {t for l in ep.ledger.lines if l is not line for t in _eli_stems(self.label(ep, l))}
        names = [text_tokens(n) for l in ep.ledger.lines for k2, k2s in l.menu.items()
                 if not (l is line and k2 == ks.canon) for n in (k2, k2s.spoken)]
        for v in self.top_values(line, ks.canon, 8):
            vs, vt = _eli_stems(v), text_tokens(v)
            if 1 <= len(vt) <= 4 and len(v) <= 30 and v.isascii() and not _ELI_LIST_RE.search(v) and '"' not in v \
                    and not vs & bad and any(len(t) > 1 and t not in own for t in vs) \
                    and not any(_eli_run(vt, n) >= 0 for n in names if n):
                return _eli_clean(v)
        return ""

    def eligible(self, ep: Any, line: Line) -> list:
        """Askable keys, best first; a key yields to its eligible specific sibling."""
        stated = self.stated(ep, line)
        meta = self.meta_votes.get(line.idx, set())
        hint = self.hint.get(line.idx, {})
        out = [ks for k, ks in line.menu.items() if not ks.meta and k not in meta and k not in stated
               and k not in line.closed and k not in line.asked and ks.score >= _ELI_MIN_SCORE
               and self.question(ep, line, ks) is not None]
        out.sort(key=lambda ks: (-ks.score, -hint.get(ks.canon, 0), -ks.cov, ks.canon))
        if out and out[0].sibling_of:
            sib = next((ks for ks in out if ks.canon == out[0].sibling_of), None)
            if sib is not None and (sib.vote > 0 or not self.votes.get(line.idx)):
                out.remove(sib)
                out.insert(0, sib)
        return out

    def hint_strength(self, ep: Any, line: Line, keys: list) -> int:
        hint = self.hint.get(line.idx, {})
        hinted = self.parse_hints(ep, line)
        return max((hint.get(ks.canon, 0) + (1 if ks.canon in hinted else 0) for ks in keys), default=0)

    def label(self, ep: Any, line: Line) -> str:
        """The exact shelf label (else the opened candidates' shelf), lowercased; the line's words when there is
        none, RANK rejected it or another line shares it; '' for a lone guessed line."""
        shelves = [ep.catalog[i].shelf for i in line.opened if i in ep.catalog and ep.catalog[i].shelf]
        lab = line.shelf_label or (max(sorted(set(shelves)), key=shelves.count) if shelves else None)
        if lab and line.idx not in self.shelf_bad and not any(
                o.idx != line.idx and o.shelf_label and canon_key(o.shelf_label) == canon_key(lab)
                for o in ep.ledger.lines):
            return _eli_clean(lab)
        if line.guessed and len(ep.ledger.lines) == 1:
            return ""                                     # dead-parse words are no name; one line needs none
        return _eli_clean(line.words or line.shelf_words or lab)

    def question(self, ep: Any, line: Line, ks: KeyStats, example: str = "") -> Optional[str]:
        lab = self.label(ep, line)
        if not ks.spoken or (lab and set(text_tokens(ks.spoken)) <= set(text_tokens(lab))) \
                or (not lab and len(ep.ledger.lines) > 1):
            return None
        where = f" for the {lab}" if lab else ""
        tail = f", for example {example}" if example else ""
        for text in (f"Which {ks.spoken} would you like{where}{tail}?", f"Which {ks.spoken}{where}{tail}?"):
            if len(text) <= _ELI_Q_MAX:
                return text
        return None

    def rank(self, ep: Any, lines: Optional[list] = None) -> None:
        lines = list(ep.ledger.lines) if lines is None else lines
        for line in lines:
            self.rank_sizes[line.idx] = len(line.menu)
        if not lines or not ep.llm.allowed("RANK"):
            ep.trace.decision("rank", votes=0, why="oracle unavailable: facet-only scoring")
            return
        lookups, out = {}, []
        for line in lines:
            table, menu = {}, []
            for k, ks in sorted(line.menu.items(), key=lambda kv: (-kv[1].cov, kv[0])):
                if ks.meta or ks.unseen:
                    continue
                for name in [k, canon_key(ks.spoken)] + ks.variants:
                    table.setdefault(name, k)
                if len(menu) < 30:
                    menu.append({"key": ks.spoken, "vals": [clip_text(v, 40) for v in self.top_values(line, k, 4)],
                                 "cov": round(ks.cov, 2)})
            lookups[line.idx] = table
            out.append({"line": line.idx, "label": self.label(ep, line), "words": line.words.split(),
                        "stated": [c.attr for c in line.constraints if c.active] + self.h1_ref_attrs(ep, line),
                        "menu": menu})
        payload = {"request": ep.query, "lines": out}
        check = self.rank_validator(ep, lookups)
        tokens = max(ROLE_REGISTRY["RANK"].max_tokens, 300 + 200 * len(lines))     # ~200 tokens per line
        jobs = [LlmJob("RANK", payload, models=ladder, validator=check, max_tokens=tokens, label=f"rank_vote_{i}")
                for i, ladder in enumerate(ep.llm.vote_ladders("RANK", 3))]
        self.merge(ep, lines, [a for a in ep.llm.ask_many(jobs) if isinstance(a, dict)])

    def rank_validator(self, ep: Any, lookups: dict) -> Callable:
        req = norm_space(ep.query)

        def verbatim(span: Any) -> bool:
            s = norm_span(span)
            return len(s) >= 3 and s in req

        def check(v: dict) -> tuple:
            out = []
            for ln in v.get("lines") or []:
                table = lookups.get(ln.get("line"))
                if table is None:
                    continue
                keys, unseen = [], []
                for k in (ln.get("keys") or []) + [dict(u, s0to3=None) for u in ln.get("unseen") or []]:
                    c = canon_key(k.get("key"))
                    quoted = verbatim(k.get("span"))
                    s = as_number(k.get("s0to3"))
                    if c in table:
                        keys.append([table[c], 2 if s is None else clamp(s, 0, 3), quoted])
                    elif c and (quoted or k.get("s0to3") is None):
                        unseen.append([c, _eli_clean(k.get("key")), quoted])
                stated = [[canon_key(a), table[canon_key(k)]] for a, k in (ln.get("stated_map") or {}).items()
                          if canon_key(k) in table]
                meta = [table[canon_key(m)] for m in ln.get("meta") or [] if canon_key(m) in table]
                out.append({"line": ln["line"], "shelf_ok": ln.get("shelf_ok"), "keys": keys, "unseen": unseen,
                            "stated": stated, "meta": meta})
            if not out:
                return False, None, "no line of the payload"
            return True, {"lines": out}, ""
        return check

    def merge(self, ep: Any, lines: list, answers: list) -> None:
        """Borda 6..1 per vote / (6 x voters); unseen keys need 2 votes or a span; majorities for the rest."""
        info = []
        for line in lines:
            idx = line.idx
            pts, quotes, unseen, meta, stated, shelf, voters = {}, {}, {}, {}, {}, [], 0
            for a in answers:
                ln = next((x for x in a["lines"] if x["line"] == idx), None)
                if ln is None:
                    continue
                voters += 1
                ranked = [(c, s, q) for c, s, q in ln["keys"] if s > 0]
                for c, spoken, q in ln["unseen"]:
                    u = unseen.setdefault(c, [spoken, 0, False])
                    u[1] += 1
                    u[2] = u[2] or q
                    ranked.append((c, 3 if q else 2, q))
                ranked.sort(key=lambda x: -x[1])
                done: set = set()
                for c, _, q in ranked:
                    if c in done:
                        continue
                    if len(done) < _ELI_TOP:
                        pts[c] = pts.get(c, 0) + _ELI_TOP - len(done)
                    done.add(c)
                    quotes[c] = quotes.get(c, 0) + (1 if q else 0)
                for c in set(ln["meta"]):
                    meta[c] = meta.get(c, 0) + 1
                for attr, c in ln["stated"]:      # h3: never onto another key than a stated attribute's own
                    src = [x.attr for x in line.constraints if x.active and _eli_stems(attr) & (
                        _eli_stems(x.attr) | _eli_stems(x.span))] + [
                        a for a in self.h1_ref_attrs(ep, line) if _eli_stems(attr) & _eli_stems(a)]
                    if any(self.family(ep, line, a) in (None, c) for a in src):
                        stated[c] = stated.get(c, 0) + 1
                if isinstance(ln.get("shelf_ok"), bool):
                    shelf.append(ln["shelf_ok"])
            if not voters:
                continue
            admitted = {c: u[0] for c, u in unseen.items() if u[1] >= 2 or u[2]}
            self.unseen.setdefault(idx, {}).update(admitted)
            soft = {c for c, k in meta.items() if 2 * k > voters}      # h3: no vote ...
            self.votes[idx] = {c: p / (_ELI_TOP * voters) for c, p in pts.items()
                               if (c in line.menu or c in admitted) and c not in soft}
            hint = self.hint.setdefault(idx, {})
            for c, q in quotes.items():
                hint[c] = hint.get(c, 0) + q
            self.meta_votes[idx] = {c for c in soft if self._h3_ident(line, c)}   # ... closes identity keys
            self.stated_votes[idx] = {c for c, k in stated.items() if 2 * k > voters}
            if shelf and line.shelf_label:
                if 2 * sum(shelf) > len(shelf) and line.shelf_conf == 1:
                    line.shelf_conf = 2
                elif 2 * sum(shelf) < len(shelf):
                    self.shelf_bad.add(idx)
            self.build_menu(ep, line)
            top = sorted(self.votes[idx].items(), key=lambda kv: (-kv[1], kv[0]))[:_ELI_TOP]
            info.append(dict(line=idx, top=[[c, round(v, 3)] for c, v in top], unseen=sorted(admitted),
                             meta=sorted(self.meta_votes[idx]), meta_soft=sorted(soft - self.meta_votes[idx]),
                             stated=sorted(self.stated_votes[idx]),
                             shelf_conf=line.shelf_conf, label_rejected=idx in self.shelf_bad))
        ep.trace.decision("rank", votes=len(answers), lines=info)

    def log_menu(self, ep: Any, line: Line) -> None:
        stated = self.stated(ep, line)
        keys = sorted(line.menu.values(), key=lambda ks: (-ks.score, ks.canon))
        ep.trace.decision("menu", line=line.idx, label=self.label(ep, line), items=len(self.menu_items(ep, line)),
                          keys=[[ks.spoken, ks.score, round(ks.cov, 2), round(ks.div, 2)] for ks in keys
                                if not ks.meta and ks.canon not in stated][:12],
                          meta=sorted(ks.canon for ks in keys if ks.meta)[:20], stated=sorted(stated),
                          twins=sorted([k, sorted(ks.twins)] for k, ks in line.menu.items() if ks.twins)[:12],
                          siblings=sorted([k, ks.sibling_of] for k, ks in line.menu.items() if ks.sibling_of),
                          affixes=sorted(self.affixes(ep)))

    def msg_room(self, ep: Any) -> str:
        """'budget' within M, 'probe' for an unknown M's 4th/5th, else ''."""
        b = ep.budget
        reserved = self.reserved_unsent(ep)
        if b.msgs_used + reserved < b.M:
            return "budget"
        if b.M_source == "default" and b.msgs_used + reserved < _ELI_PROBE_MAX \
                and ep.plan.rem() > ep.plan.nums.get("fixed", 10 ** 6):
            return "probe"
        return ""

    def status_intent(self, ep: Any) -> Optional[Intent]:
        if self.status_tried:
            return None
        self.status_tried = True
        if ep.budget.M_source != "default" or ep.binder.tool_for(ROLE_STATUS) is None:
            return None
        if any((e.get("diag") or {}).get("intent") == IK_STATUS for e in ep.trace.entries):
            return None                                   # the setup already read it
        if not ep.plan.can_spend(1, P_PROBE) or ep.plan.u() >= ep.plan.nums.get("u_c_soft", 10 ** 6):
            return None                                   # a blocked probe would end the phase
        return Intent(IK_STATUS, {}, P_PROBE, None, why="read the message budget (M unknown)")

    def next_intent(self, ep: Any) -> Optional[Intent]:
        ep.budget.M_probe_open = False                    # re-armed only for a probe question
        if self.stopped or ep.cart.writes > 0 or not ep.ledger.lines:
            return None
        if ep.binder.tool_for(ROLE_MESSAGE) is None:
            self.stopped = "no message tool"
            return None
        if e1_holds(ep.ledger.lines, ep.ledger.cap_c):
            return None                                   # an E1 decline is pending
        if not self.ranked and ep.budget.msgs_used == 0:
            top = ep.scout._h2_topup(ep)                  # h2: a thin menu gets listings before RANK and asking
            if top is not None:
                return top
        for line in ep.ledger.lines:
            self.build_menu(ep, line)
        if not self.ranked:
            self.ranked = True
            for line in ep.ledger.lines:
                self.log_menu(ep, line)
            if self.msg_room(ep):
                self.rank(ep)
        else:
            grown = [l for l in ep.ledger.lines if l.idx not in self.reranked
                     and len(l.menu) - self.rank_sizes.get(l.idx, len(l.menu)) >= 3 and self.eligible(ep, l)]
            if grown:
                self.reranked |= {l.idx for l in grown}
                self.rank(ep, grown)                      # second pass: the menu gained >= 3 keys
        st = self.status_intent(ep) if self.reask or any(self.eligible(ep, l) for l in ep.ledger.lines) else None
        if st is not None:
            return st                                     # M unknown: read it before the first question
        room = self.msg_room(ep)
        if not room or not ep.plan.can_spend(1, P_ASK):
            self.note_loss(ep)
            return None
        while self.reask:
            it = self.reask.pop(0)
            line = ep.ledger.lines[it.line]
            if it.payload["key"] not in line.closed:
                return self.arm(ep, it, room)
        pick = self.pick(ep)
        if pick is None:
            return None
        line, ks = pick
        it = Intent(IK_ASK, {"key": ks.canon, "text": self.question(ep, line, ks)}, P_ASK, line.idx,
                    why=f"ask line {line.idx}: {ks.spoken}")
        return self.arm(ep, it, room)

    def arm(self, ep: Any, it: Intent, room: str) -> Intent:
        if room == "probe":
            ep.budget.M_probe_open = True
            it.payload["probe"] = True
        return it

    def pick(self, ep: Any) -> Optional[tuple]:
        lines = ep.ledger.lines
        live = [(line, keys) for line, keys in ((l, self.eligible(ep, l)) for l in lines) if keys]
        if not live:
            return None
        if len(lines) == 1:
            line, keys = live[0]
            return line, keys[0]
        pool = [x for x in live if not self.asks.get(x[0].idx)]
        if not pool:
            cand = [x for x in live if x[0].disclosed == 0] or live
            least = min(self.asks.get(x[0].idx, 0) for x in cand)
            pool = [x for x in cand if self.asks.get(x[0].idx, 0) == least]
        line, keys = max(pool, key=lambda x: (self.hint_strength(ep, x[0], x[1]), x[1][0].score, -x[0].idx))
        return line, keys[0]

    def note_loss(self, ep: Any) -> None:
        if self.loss_logged or len(ep.ledger.lines) < 2:
            return
        unasked = [l.idx for l in ep.ledger.lines if not self.asks.get(l.idx) and self.eligible(ep, l)]
        if unasked:
            self.loss_logged = True
            ep.trace.decision("ask", known_loss=unasked, why="messages spent before these lines were asked")

    def close(self, line: Optional[Line], key: Optional[str], twins: bool = True) -> None:
        if line is None or not key:
            return
        line.closed.add(key)
        ks = line.menu.get(key)
        if twins and ks is not None:
            line.closed |= ks.twins

    def after(self, ep: Any, intent: Intent, dec: Decoded) -> None:
        if intent.kind == IK_OPEN:
            ep.scout.after(ep, intent, dec)               # h2 menu top-up: the scout absorbs its own opens
            return
        if intent.kind == IK_STATUS:
            ep.trace.decision("budget", M=ep.budget.M, source=ep.budget.M_source, why="elicitor status read")
            ep.plan.compute("status read")
            return
        if intent.kind != IK_ASK or intent.payload.get("procedure"):
            return
        line = ep.ledger.lines[intent.line] if isinstance(intent.line, int) else None
        key = intent.payload.get("key")
        if dec.kind == D_BLOCKED:
            ep.budget.M_probe_open = False
            if dec.gid in (G_QUESTION, G_MSG_DUP, _QZ_G12S):
                self.close(line, key, twins=False)        # this question cannot go out (G12S: stated): next key
                if line is not None:
                    line.asked.setdefault(key, "blocked")
            else:
                self.stopped = f"ask blocked by {dec.gid}"
        elif dec.kind == D_SESSION_ERROR:
            ep.budget.M_probe_open = False
            self.stopped = f"message error {dec.session_code}"
        ep.trace.decision("ask", line=intent.line, key=key, text=intent.payload.get("text"),
                          sent=dec.kind != D_BLOCKED, decoded=dec.kind, gid=dec.gid,
                          reask=bool(intent.payload.get("reask")), probe=bool(intent.payload.get("probe")),
                          score=line.menu[key].score if line is not None and key in line.menu else None,
                          msgs=f"{ep.budget.msgs_used}/{ep.budget.M}", stopped=self.stopped or None)

    def on_reply(self, ep: Any, intent: Intent, dec: Decoded) -> None:
        b = ep.budget
        if b.M_probe_open and b.msgs_used > b.M:
            b.M = b.msgs_used                             # the probed message was answered
        b.M_probe_open = False
        if intent.payload.get("procedure"):
            self.proc_sent += 1
            ep.trace.decision("reply", procedure=True, text=clip_text(dec.reply_text, 300))
            return
        lines = ep.ledger.lines
        idx = intent.line if isinstance(intent.line, int) and 0 <= intent.line < len(lines) else None
        line = lines[idx] if idx is not None else None
        key = intent.payload.get("key")
        if idx is not None:
            self.asks[idx] = self.asks.get(idx, 0) + 1
        text = dec.reply_text if dec.kind == D_REPLY else None
        if line is None or not key:
            return
        if not text or not str(text).strip():
            line.asked[key] = "no_reply"
            self.close(line, key)
            self.no_reply += 1
            if self.no_reply >= 2:
                self.stopped = "the message tool gives no replies"
            ep.trace.decision("reply", line=idx, key=key, cls="no_reply", decoded=dec.kind)
            return
        norm = norm_space(text)
        template = any(lk != (idx, key) for lk in self.seen_replies.get(norm, ()))
        self.seen_replies.setdefault(norm, set()).add((idx, key))
        s = self.structural(ep, text, line, key)
        j = self.oracle(ep, intent, text) if ep.llm.allowed("REPLY") else None
        src = "oracle"
        if j is not None and (j["cls"] == _ELI_DISCLOSED) != (s["cls"] == _ELI_DISCLOSED):
            j, src = self.majority(s, j, self.oracle(ep, intent, text, second=True))
        if j is None:
            j, src = s, "structural"
        j = dict(j)
        cls = j["cls"]
        nar = None
        if cls == _ELI_DISCLOSED and not template:
            self.disclose(ep, line, key, j, text)
        elif cls == "one_at_a_time" and (idx, key) not in self.reasked:
            nar = self.narrower(ep, line, key)
        line.asked[key] = cls if not template else "template"
        if nar is not None:
            self.reask.append(nar)
            self.reasked |= {(idx, key), (idx, nar.payload["key"])}
        elif cls != _ELI_DISCLOSED or template:
            self.close(line, key)
        ep.trace.decision("reply", line=idx, key=key, cls=cls, src=src, template=template,
                          attr=j.get("attr"), op=j.get("op"), vals=j.get("vals"), bound=j.get("bound"),
                          ref=j.get("ref"), named_line=j.get("_line"), reask=nar.payload["text"] if nar else None,
                          text=clip_text(text, 300))

    @staticmethod
    def majority(s: dict, j: dict, j2: Optional[dict]) -> tuple:
        """Disclosure by majority of structure and two oracle opinions; the answer is an oracle's."""
        if j2 is None:
            return j, "oracle"
        d = _ELI_DISCLOSED
        if (j["cls"] == d) == (j2["cls"] == d):
            return j, "oracle+second"
        return (j if (j["cls"] == d) == (s["cls"] == d) else j2), "majority"

    def disclose(self, ep: Any, asked_line: Line, asked_key: str, j: dict, text: str) -> None:
        """A Constraint (disclosed) or Reference (reply) on the line and attribute the reply names."""
        target = self.line_named(ep, j.get("line_phrase"), text) if len(ep.ledger.lines) > 1 else None
        target = target or (ep.ledger.lines[j["_line"]] if isinstance(j.get("_line"), int) else asked_line)
        j["_line"] = target.idx
        s, at = self.structural(ep, text, target, asked_key), text_tokens(j.get("attr"))
        if s.get("_line") == target.idx and s.get("attr") and not j.get("ref") and (
                _eli_run(text_tokens(text), at) < 0 or set(at) < set(text_tokens(s["attr"]))):
            j["attr"] = s["attr"]     # h3: real votes said 'heel type' for 'type of heels'
        fam = self.family(ep, target, j.get("attr"))
        if fam is None:
            fam = canon_key(j.get("attr")) or asked_key
            if fam not in target.menu:
                self.unseen.setdefault(target.idx, {})[fam] = _eli_clean(j.get("attr")) or fam
                target.menu[fam] = KeyStats(canon=fam, spoken=_eli_clean(j.get("attr")) or fam, unseen=True)
        op, span = j.get("op") or "eq", j.get("value_span") or ""
        ref = j.get("ref")
        if ref and ref.get("merchant"):
            src = {"merchant": ref["merchant"], "shelf_words": ref.get("shelf_words") or "",
                   "descriptors": list(ref.get("descriptors") or [])}
            if j.get("_rel_guessed"):
                src["rel_guessed"] = True
            ep.ledger.refs.append(Reference(rid=f"q{len(ep.ledger.refs)}", line=target.idx, attr=fam,
                                            relation=ref.get("rel") or "equal", origin="reply", source=src,
                                            span=span or text))
            ep.plan.compute("reference from a reply")
        elif op in ("ge", "le", "between") and j.get("bound"):
            target.constraints.append(Constraint(cid=f"d{target.idx}.{len(target.constraints)}", attr=fam, op=op,
                                                 bound=dict(j["bound"]), source="disclosed", span=span or text))
        elif j.get("vals"):
            target.constraints.append(Constraint(cid=f"d{target.idx}.{len(target.constraints)}", attr=fam,
                                                 op=op if op in ("eq", "in", "ne") else "eq", vals=list(j["vals"]),
                                                 source="disclosed", span=span or text))
        target.disclosed += 1
        target.asked.setdefault(fam, _ELI_DISCLOSED)
        self.close(target, fam)
        self.close(asked_line, asked_key)
        ep.ledger.version += 1
        ep.coverage.invalidate(ep, target, "disclosure")

    def narrower(self, ep: Any, line: Line, key: str) -> Optional[Intent]:
        """After one_at_a_time: a more specific sibling key, else the key pinned by one example value (h3: it may
        hold the label word, as 'a-line skirts')."""
        ks = line.menu.get(key)
        if ks is None:
            return None
        sa = set(key.split())
        for other in self.eligible(ep, line):
            if sa < set(other.canon.split()):
                return Intent(IK_ASK, {"key": other.canon, "text": self.question(ep, line, other), "reask": True},
                              P_ASK, line.idx, why=f"narrower re-ask line {line.idx}: {other.spoken}")
        ex = self._h3_example(ep, line, ks)
        text = self.question(ep, line, ks, example=ex) if ex else None
        if text:
            return Intent(IK_ASK, {"key": key, "text": text, "reask": True}, P_ASK, line.idx,
                          why=f"narrower re-ask line {line.idx}: {ks.spoken} (example)")
        return None

    def line_named(self, ep: Any, phrase: Any, text: str) -> Optional[Line]:
        """The line a phrase names best (labels, words), or whose label the text names first."""
        best, top, tie = None, 0.0, False
        if phrase:
            ps = _eli_stems(phrase)
            for line in ep.ledger.lines:
                names = (self.label(ep, line), line.words, line.shelf_words)
                score = max((len(ps & _eli_stems(n)) / max(1, len(_eli_stems(n))) for n in names if n), default=0.0)
                if score > top:
                    best, top, tie = line, score, False
                elif score == top and score > 0:
                    tie = True
            return None if tie else best
        toks = text_tokens(text)
        hits = [(i, -len(lt), line.idx) for line in ep.ledger.lines for lt in [text_tokens(self.label(ep, line))]
                for i in [_eli_run(toks, lt) if lt else -1] if i >= 0]
        return ep.ledger.lines[min(hits)[2]] if hits else None       # the label named first

    def structural(self, ep: Any, text: str, line: Line, key: str) -> dict:
        """Vocabulary-free: a menu key of the named (else asked) line plus a known merchant (reference,
        relation unknown) or a known value of that key; else unclear."""
        toks = text_tokens(text)
        target = (self.line_named(ep, None, text) if len(ep.ledger.lines) > 1 else None) or line
        best = None
        for k, ks in target.menu.items():
            for name in dict.fromkeys((k, canon_key(ks.spoken), *ks.variants)):
                nt = name.split()
                i = _eli_run(toks, nt)
                if i >= 0 and (best is None or len(nt) > best[1]):
                    best = (k, len(nt), i)
        if best is None:
            return {"cls": "unclear"}
        fam, n, at = best
        out = {"cls": _ELI_DISCLOSED, "attr": target.menu[fam].spoken, "op": "eq", "vals": [], "_line": target.idx}
        rest = toks[:at] + ["\0"] * n + toks[at + n:]
        merchants = sorted({r.merchant for r in ep.catalog.values() if r.merchant and len(r.merchant) >= 4},
                           key=lambda m: (-len(m), m))
        for m in merchants:
            if _eli_run(rest, text_tokens(m)) >= 0:
                out.update(op="ref", ref={"rel": "equal", "merchant": m, "shelf_words": "", "descriptors": []},
                           _rel_guessed=True)
                return out
        ks = target.menu[fam]
        names = set(ks.variants) | {fam}
        found = []                                        # known values, nearest after the key first
        for v in {str(v) for r in ep.catalog.values() for c in names & set(r.attrs) for _, v in r.attrs[c]}:
            vt = text_tokens(v)
            i = _eli_run(rest, vt) if vt and len(v.strip()) > 1 else -1
            if i >= 0:
                found.append((i < at, i if i >= at else -i, -len(vt), v))
        if found:
            vt = text_tokens(min(found)[3])
            m = re.search(r"(?<![A-Za-z0-9])" + r"[^A-Za-z0-9]+".join(map(re.escape, vt)) + r"(?![A-Za-z0-9])",
                          text, re.I)
            out["vals"] = [m.group(0) if m else min(found)[3]]
            out["value_span"] = out["vals"][0]
        return out

    def oracle(self, ep: Any, intent: Intent, text: str, second: bool = False) -> Optional[dict]:
        if not ep.llm.allowed("REPLY", optional=second):
            return None
        line = ep.ledger.lines[intent.line]
        ks = line.menu.get(intent.payload.get("key"))
        payload = {"request": ep.query, "question": intent.payload.get("text", ""), "reply": text,
                   "asked": {"line": line.idx, "key": ks.spoken if ks else intent.payload.get("key")},
                   "lines": [{"line": l.idx, "label": self.label(ep, l)} for l in ep.ledger.lines]}
        models = None
        if second:
            payload["opinion"] = 2
            models = (MODEL_LUNA,)
        return ep.llm.ask("REPLY", payload, models=models, validator=self.reply_validator(text), optional=second)

    @staticmethod
    def reply_validator(text: str) -> Callable:
        """Spans, values, merchants, descriptors verbatim in the reply; a bound's number re-extracted from its
        span; a disclosure needs values, a bound or a reference."""
        whole = norm_space(text)

        def verb(s: Any) -> bool:
            s2 = norm_span(s)
            return bool(s2) and s2 in whole

        def check(v: dict) -> tuple:
            if v.get("cls") != _ELI_DISCLOSED:
                return True, {"cls": v.get("cls")}, ""
            out = {"cls": _ELI_DISCLOSED, "attr": str(v.get("attr") or "").strip(), "op": v.get("op") or "eq",
                   "vals": [x for x in v.get("vals") or [] if verb(x)], "bound": None, "ref": None,
                   "line_phrase": v.get("line_phrase") if verb(v.get("line_phrase")) else "",
                   "value_span": v.get("value_span") if verb(v.get("value_span")) else ""}
            bd = v.get("bound") or {}
            if bd.get("num") is not None:
                raw = next((s for s in (bd.get("raw"), out["value_span"]) if verb(s)), text)
                nums = [n for n, *_ in numbers_in(raw)]
                num = next((n for n in nums if abs(float(n) - float(bd["num"])) < 1e-6), None)
                if num is not None:
                    out["bound"] = {"num": num, "unit": bd.get("unit") if verb(bd.get("unit")) else "", "raw": raw}
            rf = v.get("ref") or {}
            if verb(rf.get("merchant")):
                out["ref"] = {"rel": rf.get("rel") or "equal", "merchant": rf["merchant"],
                              "shelf_words": rf.get("shelf_words") if verb(rf.get("shelf_words")) else "",
                              "descriptors": [d for d in rf.get("descriptors") or [] if len(d) == 2 and verb(d[1])]}
                out["op"] = "ref"
            elif out["op"] == "ref":
                return False, None, "a reference needs a verbatim merchant"
            elif out["op"] in ("ge", "le", "between"):
                if out["bound"] is None:
                    return False, None, "the bound's number is not in the reply"
            elif not out["vals"]:
                return False, None, "no verbatim value"
            if not out["attr"]:
                return False, None, "no attribute"
            out["value_span"] = out["value_span"] or (out["vals"][0] if out["vals"] else "")
            return True, out, ""
        return check

    def question_ok(self, ep: Any, intent: Intent) -> tuple:
        text = str(intent.payload.get("text") or "")
        if not text.strip():
            return False, "empty question"
        if text.count("?") != 1 or len(text) > _ELI_Q_MAX:
            return False, "one '?' and at most 120 characters"
        lines = ep.ledger.lines
        if not isinstance(intent.line, int) or not 0 <= intent.line < len(lines):
            return False, "no line"
        line = lines[intent.line]
        key = intent.payload.get("key")
        ks = line.menu.get(key)
        if ks is None or ks.meta or key in self.meta_votes.get(line.idx, ()):
            return False, "not an askable menu key"
        if key in line.closed:
            return False, "the (line, key) is closed"
        toks = text_tokens(text)
        lab = text_tokens(self.label(ep, line))
        if not lab and len(lines) > 1:
            return False, "the line's label must be named"
        for part in [text_tokens(ks.spoken)] + ([lab] if lab else []):    # key first: it may hold a label word
            i = _eli_run(toks, part)
            if i < 0:
                return False, "the line's label and the key must be named"
            toks = toks[:i] + ["\0"] + toks[i + len(part):]
        for other in lines:
            for k2, k2s in other.menu.items():
                if k2 != key and any(n and _eli_run(toks, n) >= 0 for n in (k2.split(), text_tokens(k2s.spoken))):
                    return False, f"names a second key ({k2})"
        return True, ""

    def reserved_unsent(self, ep: Any) -> int:
        fn = getattr(ep.reviewer, "reserved_messages", None)
        try:
            total = int(fn(ep) or 0) if callable(fn) else 0
        except Exception:
            total = 0
        return max(0, total - self.proc_sent)

    def has_menu(self, ep: Any) -> bool:
        return all(line.opened for line in ep.ledger.lines)

    def done(self, ep: Any) -> bool:
        return bool(self.stopped)

RES_PAIR_SEARCH = "pair_rule_search"     # Reference.obligation while the pair-rule search may still run
RES_MAX_PAGES = 6                        # chain pages per reference (races: 1-5)
RES_TRAP_HOPS = 2                        # trap chains are resolved only when <= 2 hops fit
RES_HOP_SEARCHES = 3                     # search calls per hop, pages included
RES_HOP_OPENS = 4                        # candidate open calls per hop
RES_HOP_SPECS = 2 * RES_HOP_OPENS        # v7: attribute reads per hop (split view/spec)
RES_PAIR_BATCHES = 2                     # pairing obligation: opens of the line's unopened candidates
RES_NOTES_BATCH = 6                      # records per NOTES call
RES_HEAD_WORDS = 4                       # a pointer names its attribute within the first 4 words of a sentence
RES_NEAR_PRICE = 1.05                    # trap: a clean alternative within 5% of the price (cheapest lines)
_RES_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’/&.\-]*|[(),;:]")
_RES_SENT_RE = re.compile(r"(?<=[.!?…])\s+")
_RES_SPLIT_RE = re.compile(r"[,:]|\s(?:" + "|".join(sorted(STOPWORDS - {"a", "an", "the"})) + r")\s")  # not articles
_RES_EMPTY = {"restates": [], "pointers": [], "not_versions": [], "pair_rules": []}

def _res_stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 5 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 4 and t.endswith("ed"):
        return t[:-2]
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t

def _res_toks(text: Any) -> list:
    """Stemmed content tokens; letter/digit runs split ("512gb" -> 512 gb)."""
    return [_res_stem(p) for t in text_tokens(text) for p in re.findall(r"\d+|[a-z]+", t) if p not in STOPWORDS]

def _res_has(hay: list, needle: list) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))

def _res_val(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return ", ".join(_res_val(x) for x in v)
    return dump_json(v) if isinstance(v, dict) else ("" if v is None else str(v))

def _res_name(x: Any) -> str:
    return canon_key(re.sub(r"['’]s\b", "", str(x or "")))

def _res_digit(text: Any) -> bool:
    return any(ch.isdigit() for ch in str(text))

def _res_near(x: str, y: str) -> bool:
    """Edit distance <= 1."""
    if len(x) == len(y):
        return sum(a != b for a, b in zip(x, y)) <= 1
    x, y = sorted((x, y), key=len)
    return len(y) - len(x) == 1 and any(y[:i] + y[i + 1:] == x for i in range(len(y)))

def _res_attr_match(a: Any, b: Any) -> bool:
    """Same attribute: equal stemmed tokens (short extra words like "its" ignored) or one-letter variants."""
    ta, tb = _res_toks(a), _res_toks(b)
    if len(ta) != len(tb):
        short, long_ = sorted((ta, tb), key=len)
        return bool(short) and _res_has(long_, short) and all(len(t) <= 3 for t in long_ if t not in short)
    return bool(ta) and all(x == y or (min(len(x), len(y)) >= 4 and _res_near(x, y)) for x, y in zip(ta, tb))

def _res_same(a: Any, b: Any) -> bool:
    ta, tb = _res_toks(a), _res_toks(b)
    return bool(ta) and (ta == tb or _res_has(tb, ta))

def _res_sentences(text: Any) -> list:
    return [s.strip() for s in _RES_SENT_RE.split(str(text or "")) if s.strip()]

def _res_descs(raw: Any) -> list:
    """Descriptors as [(attr, value)]; a bare value has attr ""."""
    out = []
    for d in raw or []:
        if isinstance(d, (list, tuple)) and d:
            a, v = (d[0], d[-1]) if len(d) >= 2 else ("", d[0])
        elif isinstance(d, str):
            a, v = "", d
        else:
            continue
        a, v = str(a or "").strip(), str(v or "").strip()
        if _res_toks(v):
            out.append((a, v))
    return out[:8]

def _res_own(rec: Any, attr: Any) -> Optional[tuple]:
    """(key rank, value): rank 0 exact key, 1 a spelling variant, 2 the shortest key holding attr."""
    at, best = _res_toks(attr), None
    for key, pairs in (rec.attrs or {}).items():
        val, kt = (_res_val(pairs[0][1]).strip() if pairs else ""), _res_toks(key)
        rank = 0 if kt == at else 1 if _res_attr_match(key, attr) else 2 if _res_has(kt, at) else None
        if val and at and rank is not None and (best is None or (rank, len(kt)) < best[0]):
            best = ((rank, len(kt)), val)
    return (best[0][0], best[1]) if best else None

def _res_own_value(rec: Any, attr: Any) -> Optional[str]:
    own = _res_own(rec, attr)
    return own[1] if own else None

def _res_score(rec: Any, desc: list) -> tuple:
    """(strong, shown, quality, contradictions). Strong: in an attribute (own key if tagged) or the shelf; shown:
    anywhere. Quality: same text 4, tokens 3, inside a value 2, key + value or shelf 1.5, title/notes 1."""
    pairs = [(_res_toks(k), _res_toks(_res_val(v))) for k, vals in (rec.attrs or {}).items() for _, v in vals]
    texts = {norm_space(_res_val(v)) for vals in (rec.attrs or {}).values() for _, v in vals}
    shelf = _res_toks(rec.shelf or "")
    weak = [_res_toks(rec.title), _res_toks(rec.notes or ""), _res_toks(rec.overview_head)]
    strong = shown = bad = 0
    quality = 0.0
    for a, v in desc:
        vt = _res_toks(v)
        sv = set(vt)
        own = _res_own(rec, a) if a else None
        if own is not None:
            ot = _res_toks(own[1])
            hit = _res_has(ot, vt) or sv <= set(ot)
            strong, shown, bad = strong + hit, shown + hit, bad + (not hit)
            quality += ((4 if norm_space(own[1]) == norm_space(v) else 3 if ot == vt else 2) - 0.5 * own[0]) * hit
            continue
        if norm_space(v) in texts:
            got = 4.0
        elif any(t == vt for _, t in pairs):
            got = 3.0
        elif any(_res_has(t, vt) for _, t in pairs):
            got = 2.0
        elif any(sv <= set(k + t) for k, t in pairs) or sv <= set(shelf):
            got = 1.5
        elif any(sv <= set(f) for f in weak):
            got = 1.0
        else:
            quality += 0.5 * max(len(sv & set(f)) / len(sv) for f in [k + t for k, t in pairs] + weak + [shelf])
            continue
        strong, shown, quality = strong + (got > 1), shown + 1, quality + got
    return strong, shown, quality, bad

def _res_split_key(phrase: str, keys: set) -> tuple:
    """(key, value) when a learned key opens or closes the phrase ("closure slip-on"), else ("", phrase)."""
    words = [w for w in (x.strip("()[],;:.\"'") for x in phrase.split()) if w]
    canon = [canon_key(w) for w in words]
    for n in range(len(words) - 1, 0, -1):
        if " ".join(canon[:n]) in keys:
            return " ".join(words[:n]), " ".join(words[n:])
        if " ".join(canon[-n:]) in keys:
            return " ".join(words[-n:]), " ".join(words[:-n])
    return "", phrase

def _res_informative(v: str, seen: Any) -> bool:
    """False for long, number-free phrases mostly unseen in values or titles ("not this item")."""
    toks = _res_toks(v)
    return len(toks) <= 2 or _res_digit(v) or 2 * sum(t in seen for t in toks) >= len(toks)

def _res_query_words(desc: list, key_words: set, value_words: Any = ()) -> list:
    """AND-search words (a key name empties it): tagged values; bare phrases give numbers and their unit, else
    words seen in values or titles that are not key words. Phrases with numbers first."""
    out = []
    for a, v in sorted(desc, key=lambda d: not _res_digit(d[1])):
        words = [w for w in (x.strip("()[],;:.\"'") for x in v.split()) if w]
        if not a:
            digit = [_res_digit(w) for w in words]
            seen = [set(_res_toks(w)) <= set(value_words) for w in words]
            key = [set(_res_toks(w)) <= key_words and not s for w, s in zip(words, seen)]
            words = [w for i, w in enumerate(words) if digit[i] or (not key[i] and (
                (i and digit[i - 1]) if any(digit) else seen[i]))]
        out += [w for w in words if len(w) > 1 and w.lower() not in STOPWORDS]
    return out

def _res_fb_partner(note: str, attr: Any, v: Any) -> Optional[tuple]:
    """Pair rule by shape: clauses sharing a connector ("X <conn> Y, Z <conn> W"); v's partner is across it."""
    vt, at = _res_toks(v), set(_res_toks(attr))
    for sent in _res_sentences(note):
        st = _res_toks(sent)
        if not _res_has(st, vt) or (at and not at & set(st)):
            continue
        clauses = [c for c in ([w.lower() for w in _RES_WORD_RE.findall(x) if w not in "(),;:"]
                               for x in re.split(r"[,;:]|\s(?:and|or)\s", sent)) if c]
        counts: dict = {}
        for c in clauses:
            for w in set(c):
                counts[w] = counts.get(w, 0) + 1
        for c in clauses:
            flat = [(i, s) for i, w in enumerate(c) for s in _res_toks(w)]
            pos = next((k for k in range(len(flat)) if [s for _, s in flat[k:k + len(vt)]] == vt), None)
            if pos is None:
                continue
            i0, i1 = flat[pos][0], flat[pos + len(vt) - 1][0] + 1
            vw = set(c[i0:i1])
            top = max((n for w, n in counts.items() if w not in vw), default=0)
            conn = {w for w, n in counts.items() if n == top and n >= 2 and w not in vw}
            j = i1
            while j < len(c) and c[j] in conn:
                j += 1
            if i1 < j < len(c):
                return " ".join(c[j:j + 3]).strip(".…"), sent
            j = i0
            while j > 0 and c[j - 1] in conn:
                j -= 1
            if 0 < j < i0:
                return c[j - 1].strip(".…"), sent
    return None

def _res_check_notes(ans: dict, texts: dict) -> tuple:
    """NOTES validator: known rids, spans verbatim in the notes, values and merchants verbatim in their span;
    a record that lost items is lossy (read by shape too)."""
    def inside(x: Any, text: Any) -> bool:
        x = norm_space(x).strip(" .;,:")
        return bool(x) and x in norm_space(text)

    out, recs = [], ans.get("recs") or []
    for x in recs:
        note = texts.get(x.get("rid"))
        if note is None:
            continue
        rd = {"rid": x["rid"]}
        for kind in _RES_EMPTY:
            keep = []
            for it in x.get(kind) or []:
                span = it["span"]
                desc = [d for d in it.get("descriptors") or [] if d and inside(d[-1], span)]
                pairs = [p for p in it.get("pairs") or [] if len(p) == 2 and inside(p[0], span) and inside(p[1], span)]
                who = re.sub(r"['’]s$", "", str(it.get("merchant") or "").strip())
                if not inside(span, note) or (kind == "restates" and not inside(it.get("val"), span)) \
                        or (kind == "pair_rules" and not pairs) or (who and not inside(who, span)) \
                        or (kind == "pointers" and not who):
                    continue
                keep.append(dict(it, merchant=who, descriptors=desc, pairs=pairs, shelf_words=it.get(
                    "shelf_words") if inside(it.get("shelf_words"), span) else ""))
            rd[kind] = keep
        given = len(x.get("pointers") or []) + len(x.get("pair_rules") or [])
        rd["lossy"] = len(rd["pointers"]) + len(rd["pair_rules"]) < given
        out.append(rd)
    if recs and not out:
        return False, None, "no record id of the payload"
    return True, {"recs": out}, ""

@dataclass
class _ResWalk:
    """Engine-local walk state of one reference (request, reply or trap)."""
    ref: Reference
    limit: int = RES_MAX_PAGES
    item: Optional[str] = None             # trap walks: the finalist whose own value is pointed elsewhere
    alts: list = field(default_factory=list)       # merchant name alternatives of the current hop
    shelf: str = ""
    desc: list = field(default_factory=list)
    visited: list = field(default_factory=list)    # chain pages, in order
    queries: list = field(default_factory=list)    # query texts sent or reused for the current hop
    qm: dict = field(default_factory=dict)         # query text -> merchant it searched
    searches: int = 0
    opens: int = 0
    specs: int = 0                                 # v7: the hop's attribute reads
    bad: set = field(default_factory=set)          # ids that failed to open
    stage: str = "locate"                          # locate | end | pair | done
    end_open: bool = False
    pair_opens: int = 0
    gate: list = field(default_factory=list)       # located, unopened next page of a stopped walk

class Resolver(Engine):
    name = "resolver"

    def __init__(self, ep: Any = None):
        super().__init__(ep)
        self.walks: dict = {}              # id(Reference) -> _ResWalk
        self.readings: dict = {}           # id -> validated NOTES reading
        self.asked: set = set()            # ids sent to NOTES
        self.issued: set = set()           # intent signatures (never repeated)
        self.last: Optional[list] = None   # [intent, u at issue, walk, absorbed]
        self.n_refs = 0
        self.traps: list = []              # trap References (never in the ledger)
        self.effective: dict = {}          # (id, attr) -> value at the end of its trap chain
        self.dirty: set = set()            # lines whose choice is re-checked before the write
        self.mfield = FR_MERCHANT          # field holding merchant names
        self.qkeys: dict = {}              # query text -> query-book key
        self._vocab: tuple = (-1, set(), set(), set(), set())
        self._in_trap = False

    def next_intent(self, ep: Any) -> Optional[Intent]:
        self._sync(ep)
        if len(ep.ledger.refs) != self.n_refs:
            self.n_refs = len(ep.ledger.refs)
            ep.plan.compute("references")
        for ref in list(ep.ledger.refs):
            w = self._walk_of(ref)
            if ref.status == "pending" or w.stage == "pair":
                it = self._advance(ep, w)
                if it is not None:
                    return self._issue(ep, it, w)
        return None

    def after(self, ep: Any, intent: Intent, dec: Decoded) -> None:
        if self.last is not None and self.last[0] is intent and not self.last[3]:
            self._absorb(ep, dec)

    def done(self, ep: Any) -> bool:
        return not any(r.status == "pending" or self._walk_of(r).stage == "pair" for r in ep.ledger.refs)

    def pending(self, ep: Any) -> bool:
        return any(r.status == "pending" for r in ep.ledger.refs)

    def unopened_chain_pages(self, ep: Any) -> int:
        n = 0
        for ref in ep.ledger.refs:
            w = self.walks.get(id(ref))
            if ref.status == "pending":
                n += 1 + (w is not None and w.stage == "end" and not w.end_open)
            elif w is not None and w.gate and w.gate[0] in ep.catalog and not ep.catalog[w.gate[0]].opened_by:
                n += 1
        return n

    def chain_gate(self, ep: Any) -> Optional[Intent]:
        """Q7: one OPEN of a known, unopened chain page."""
        self._sync(ep)
        for ref in ep.ledger.refs:
            w = self.walks.get(id(ref))
            if w is None or not w.gate or w.gate[0] not in ep.catalog or ep.catalog[w.gate[0]].opened_by:
                continue
            it = Intent(IK_OPEN, {"ids": w.gate[:1], "need": self._need(ep), "chain_page": True, "procedure": True},
                        P_VERIFY, None, why=f"procedure: open chain page of {ref.rid}")
            w.gate = []
            if it.signature() not in self.issued:
                return self._issue(ep, it, w)
        return None

    def pairing_obligation(self, ep: Any, line: Line) -> Optional[Intent]:
        """Q3 before a cart write on the line (see the contract)."""
        if ep.finished or ep.phase == "FALLBACK" or ep.clock() >= T_FALLBACK \
                or ep.budget.first_finish_step is not None or ep.plan.pressure():
            return None
        self._sync(ep)
        walks = [self._walk_of(r) for r in ep.ledger.refs if r.line == line.idx]
        for w in walks + [self.walks[id(r)] for r in self.traps if r.line == line.idx]:
            if w.ref.status == "pending" or w.stage == "pair":
                it = self._advance(ep, w, {"procedure": True})
                if it is not None and not (it.kind == IK_SEARCH and ep.plan.endgame()):
                    return self._issue(ep, it, w)
        if line.idx not in self.dirty or not line.choice:
            return None
        self.dirty.discard(line.idx)
        options = ep.selector.orderable(ep, line)
        if not options or line.choice in options:
            return None
        old, line.choice, line.backups = line.choice, options[0], options[1:4]
        ep.trace.decision("ref_hop", stage="repick", line=line.idx, was=old, choice=line.choice)
        it = Intent(IK_OPEN, {"ids": [line.choice], "need": NEED_STOCK, "replacement": True}, P_COMMIT, line.idx,
                    why=f"re-picked line {line.idx} after a reference result")
        return None if it.signature() in self.issued else self._issue(ep, it, None)

    def void_attrs(self, ep: Any, ident: str, keys: Any) -> set:
        rec = ep.catalog.get(ident)
        if rec is None or not rec.notes:
            return set()
        reading = self.readings.get(ident)
        return {k for k in keys if any(_res_attr_match(p["attr"], k) for p in (reading or _RES_EMPTY)["pointers"])
                or ((reading is None or reading.get("lossy")) and self._fb_pointer(ep, rec, k) is not None)}

    def trap_check(self, ep: Any, line: Line, ids: list) -> dict:
        out = {i: False for i in ids}
        cons = [c for c in line.constraints if c.active]
        if self._in_trap or not cons or not ids:
            return out
        self._in_trap = True
        try:
            noted = [i for i in ids if i in ep.catalog and ep.catalog[i].notes]
            if noted and ep.llm.allowed("NOTES", optional=True):
                self._reading(ep, noted[:RES_NOTES_BATCH], optional=True)
            keys = list(dict.fromkeys(c.attr for c in cons if c.attr))
            trapped = {i: a for i, a in ((i, self.void_attrs(ep, i, keys)) for i in noted) if a}
            unpaired = {i for c in cons if c.op == "pair_pending" and c.vals for i in ids if i in ep.catalog
                        and _res_same(c.vals[0], _res_own_value(ep.catalog[i], c.attr) or "")}
            if not trapped and not unpaired:
                return out
            clean = []
            for i in [i for i in ids if i not in trapped and i not in unpaired][:12]:
                try:
                    if ep.checker.verdict(ep, line, i, ep.ledger.cap_c).tier != "X":
                        clean.append(i)
                except Exception:
                    continue
            first = True
            for i, attrs in trapped.items():
                eff = self._effective_ok(line, i, attrs)
                price = ep.catalog[i].price_c
                alt = [c for c in clean if not line.cheapest or price is None or ep.catalog[c].price_c is None
                       or ep.catalog[c].price_c <= price * RES_NEAR_PRICE]
                out[i] = (not eff) if eff is not None else bool(alt)
                if eff is None and not alt and first:
                    first = False
                    self._want_trap(ep, line, i, sorted(attrs))
            for i in unpaired:
                out[i] = out[i] or bool(clean)
            if all(out.values()):
                return {i: False for i in ids}          # never void every id
            if any(out.values()):
                ep.trace.decision("ref_hop", stage="trap", line=line.idx, void=sorted(i for i, v in out.items() if v),
                                  trapped=sorted(trapped), unpaired=sorted(unpaired))
            return out
        finally:
            self._in_trap = False

    def _effective_ok(self, line: Line, ident: str, attrs: set) -> Optional[bool]:
        """True/False once every trapped attribute has its trap-chain value, else None."""
        if any(self.effective.get((ident, a)) is None for a in attrs):
            return None
        for a in sorted(attrs):
            v = self.effective[(ident, a)]
            for c in line.constraints:
                hit = c.active and c.attr == a and any(_res_same(x, v) for x in c.vals)
                if c.active and c.attr == a and c.vals and ((c.op in ("eq", "in") and not hit) or (c.op == "ne" and hit)):
                    return False
        return True

    def _want_trap(self, ep: Any, line: Line, ident: str, attrs: list) -> None:
        if any(r.line == line.idx and self.walks[id(r)].item == ident for r in self.traps) \
                or not ep.plan.can_spend(2 * RES_TRAP_HOPS, P_RESOLVE):
            return
        for a in attrs:
            ptr = self._pointer(ep, ident, a)
            if ptr is not None:
                ref = Reference(rid=f"t{len(self.traps)}", line=line.idx, attr=a, origin="trap", span=ptr["span"],
                                source={"merchant": ptr["alts"][0], "shelf_words": ptr["shelf"], "descriptors": ptr["desc"]})
                w = _ResWalk(ref=ref, limit=1 + RES_TRAP_HOPS, item=ident, visited=[ident])
                self._target(w, ptr)
                self.traps.append(ref)
                self.walks[id(ref)] = w
                ep.trace.decision("ref_hop", stage="trap_ref", line=line.idx, item=ident, attr=a, merchant=ptr["alts"][0])
                return

    def _walk_of(self, ref: Reference) -> _ResWalk:
        w = self.walks.get(id(ref))
        if w is None:
            src = ref.source or {}
            w = self.walks[id(ref)] = _ResWalk(ref=ref)
            self._target(w, {"alts": [src.get("merchant")] if src.get("merchant") else [],
                             "shelf": src.get("shelf_words") or "", "desc": _res_descs(src.get("descriptors"))})
        return w

    def _target(self, w: _ResWalk, ptr: dict) -> None:
        w.alts = [m for m in ptr.get("alts") or [] if _res_name(m)][:3]
        w.shelf, w.desc = ptr.get("shelf") or "", list(ptr.get("desc") or [])
        w.queries, w.qm, w.searches, w.opens, w.specs, w.bad = [], {}, 0, 0, 0, set()
        w.ref.hops.append(Hop(merchant=w.alts[0] if w.alts else "", shelf_words=w.shelf, descriptors=list(w.desc)))

    def _advance(self, ep: Any, w: _ResWalk, extra: Optional[dict] = None) -> Optional[Intent]:
        for _ in range(16):
            step = {"locate": self._hop, "end": self._end, "pair": self._pair}.get(w.stage)
            if step is None:
                return None
            it = step(ep, w)
            if it is not None:
                it.payload.update(extra or {})
                if it.signature() not in self.issued:
                    return it
                if it.kind == IK_OPEN:                 # a repeat: its ids count as failed
                    w.bad |= set(it.payload.get("ids") or [])
                    w.pair_opens += w.stage == "pair"
                else:
                    w.queries.append(it.payload.get("query"))
                    w.searches += 1
        return None

    def _hop(self, ep: Any, w: _ResWalk) -> Optional[Intent]:
        if len(w.visited) >= w.limit:
            return self._finish(ep, w, None, "hop limit")
        if not w.queries and not w.opens and not ep.plan.can_spend(2, P_RESOLVE):
            return self._finish(ep, w, None, "no budget for the hop")
        cands = self._cands(ep, w)
        unknown = [c for c in cands if not self._known(ep, c, w)]
        best, full = self._choose(ep, w, cands, bool(unknown) and w.opens < RES_HOP_OPENS)
        if full:
            return self._enter(ep, w, best, cands)
        rowfull = [c for c in unknown if self._score(ep, c, w.desc)[1] >= len(w.desc) > 0]
        n = len(w.queries)
        if not w.queries and not rowfull:
            it = self._search(ep, w)
            if it is not None or len(w.queries) > n:
                return it                              # None: an earlier query's rows were reused; look again
        if unknown and w.opens < RES_HOP_OPENS and ep.plan.can_spend(1, P_RESOLVE):
            order = rowfull + [c for c in unknown if c not in rowfull]
            ids, need = [c for c in order if not self._known(ep, c)][:max(1, min(ep.plan.batch_k(), 4))], self._need(ep)
            extra = {}
            if not ids or ids[0] != order[0]:          # the first candidate's notes are read: its attributes (v7;
                ids, need, extra = order[:1], NEED_ATTRS, {"spec": True}      # not counted as a candidate open)
            if not extra or w.specs < RES_HOP_SPECS:   # v7: bounded on their own
                return Intent(IK_OPEN, {"ids": ids, "need": need, "chain_page": True, **extra}, P_RESOLVE, None,
                              why=f"{w.ref.rid} hop {len(w.visited) + 1}: open {len(ids)} candidate(s)")
        if not unknown:                                # every located candidate was read: look further
            it = self._search(ep, w)
            if it is not None or len(w.queries) > n:
                return it
        if best is not None:
            return self._enter(ep, w, best, cands)
        w.gate = unknown[:1]
        return self._finish(ep, w, None, "chain page not found")

    def _search(self, ep: Any, w: _ResWalk) -> Optional[Intent]:
        if w.searches >= RES_HOP_SEARCHES or not ep.plan.can_spend(1, P_RESOLVE) or ep.plan.endgame():
            return None
        for q in w.queries:                            # more rows of a query that found the merchant
            rec = ep.queries.get(self.qkeys.get(q, ""))
            if rec is not None and rec.has_cont and len(rec.pages_read) < 3 and rec.ids:
                return Intent(IK_SEARCH, {"query": q, "cont": rec.next_cont, "qkey": self.qkeys[q], "qkind": "q:merchant",
                                          "merchant_value": w.qm.get(q, "")}, P_RESOLVE, None, why=f"next page of '{q}'")
        tool = ep.binder.tool_for(ROLE_SEARCH)
        for q, m in self._ladder(ep, w):
            if q in w.queries:
                continue
            w.qm[q] = m
            seen = next((r for r in ep.queries.recs.values() if r.tool == tool and r.pages_read
                         and norm_space(r.query_text) == norm_space(q)), None)
            if seen is not None:                       # another engine ran it: reuse its rows
                self.qkeys[q] = seen.canon_args
                w.queries.append(q)
                return None
            self.qkeys.setdefault(q, "res|" + norm_space(q))
            return Intent(IK_SEARCH, {"query": q, "qkey": self.qkeys[q], "qkind": "q:merchant", "merchant_value": m},
                          P_RESOLVE, None, why=f"{w.ref.rid} hop {len(w.visited) + 1}: locate '{q}'")
        return None

    def _ladder(self, ep: Any, w: _ResWalk) -> list:
        """(query, merchant): merchant + value words (+ noun without values), merchant + noun, merchant alone; a
        merchant alternative whose search returned no row of it is dropped."""
        vocab = self._vocab_of(ep)
        desc = [(a, v) for a, v in w.desc if a or _res_informative(v, vocab[4])]
        values = _res_query_words(desc, {t for k in vocab[3] for t in _res_toks(k)}, vocab[4])
        noun = next((t for t in reversed(str(w.shelf).split()) if len(t) >= 4 and t.isalpha()), "")
        alts = [m for m in w.alts if any(self._is_m(ep, r, [m]) for r in ep.catalog.values())][:1] or w.alts
        for q, m in w.qm.items():
            rec = ep.queries.get(self.qkeys.get(q, ""))
            if m and rec is not None and rec.ids and len(alts) > 1:
                alts = [a for a in alts if a != m] or alts
        out, cap = [], max(1, ep.contract.token_cap)
        for m in alts[:2] or [""]:
            for extra in (values or [noun], [noun] if values else [], []):
                kept: list = []                        # the cap counts search tokens as G9 does ("10/11" is two)
                for x in dict.fromkeys(m.split() + [x for x in extra if x]):
                    if not kept or len(text_tokens(" ".join(kept + [x]))) <= cap:
                        kept.append(x)
                q = " ".join(kept)
                if q.strip() and (q, m) not in out:
                    out.append((q, m))
        return out

    def _cands(self, ep: Any, w: _ResWalk) -> list:
        rows = []
        for q in w.queries:
            rec = ep.queries.get(self.qkeys.get(q, ""))
            rows += [i for i in (rec.ids if rec is not None else []) if i not in rows]
        named = w.alts and any(r.merchant or r.maker for r in ep.catalog.values())
        pool = [r for r in ep.catalog.values() if r.id not in w.visited and r.id not in w.bad and r.id != w.item
                and r.id not in ep.absent_ids and (self._is_m(ep, r, w.alts) if named else r.id in rows)]
        if w.shelf:
            sw = set(_res_toks(w.shelf))
            pool = [r for r in pool if r.shelf and len(sw & set(_res_toks(r.shelf))) >= (len(sw) + 1) // 2] or pool
        order = {i: n for n, i in enumerate(rows)}
        score = {r.id: self._score(ep, r.id, w.desc) for r in pool}
        pool.sort(key=lambda r: (-score[r.id][1], score[r.id][3] - score[r.id][2], order.get(r.id, len(order))))
        return [r.id for r in pool]

    def _is_m(self, ep: Any, r: Record, alts: list) -> bool:
        names = {_res_name(a) for a in alts}
        if ep.contract.merchant_path_confirmed:
            return _res_name(r.maker if self.mfield == FR_MAKER else r.merchant) in names
        return _res_name(r.merchant) in names or _res_name(r.maker) in names

    def _score(self, ep: Any, ident: str, desc: list) -> tuple:
        rec = ep.catalog.get(ident)
        return _res_score(rec, desc) if rec is not None and desc else (0, 0, 0.0, 0)

    def _known(self, ep: Any, ident: str, w: Optional[_ResWalk] = None) -> bool:
        """Opened with its notes read; v7: with a walk's descriptors, its attributes read too (a split view)."""
        rec = ep.catalog.get(ident)
        ok = rec is not None and bool(rec.opened_by) and (FR_NOTES in rec.families_seen or not self._notes_tool(ep))
        return ok and not (w is not None and w.desc and self._attrs_tool(ep, rec))

    @staticmethod
    def _attrs_tool(ep: Any, rec: Record) -> Optional[str]:
        """v7: the record tool that shows attributes when no tool that opened rec did."""
        tools = ep.binder.binding.tools
        if rec.attrs or any(FR_ATTRS in tools[n].provides_seen for n in rec.opened_by if n in tools):
            return None
        plan = ep.binder.open_plan(NEED_ATTRS)
        return plan[0][0] if plan and plan[0][0] not in rec.opened_by else None

    def _notes_tool(self, ep: Any) -> bool:
        return bool(ep.binder.open_plan(NEED_NOTES))

    def _need(self, ep: Any) -> str:
        return NEED_NOTES if self._notes_tool(ep) else NEED_ATTRS

    def _choose(self, ep: Any, w: _ResWalk, cands: list, more: bool = False) -> tuple:
        """(best opened candidate, final). Several strong matches are read for decoys; an inexact one waits."""
        known = [c for c in cands if self._known(ep, c, w)]
        if not known or not w.desc:
            return (known[0] if known else None), bool(known)
        seen = self._vocab_of(ep)[4]
        desc = [(a, v) for a, v in w.desc if a or _res_informative(v, seen)
                or any(self._score(ep, c, [(a, v)])[1] for c in known)]
        if not desc:
            return known[0], False
        n = len(desc)
        sc = {c: self._score(ep, c, desc) for c in known}
        rank = sorted(known, key=lambda c: (-sc[c][2], known.index(c)))
        strong = [c for c in rank if sc[c][0] == n and not sc[c][3]]
        if len(strong) > 1:
            self._reading(ep, strong)
            strong = [c for c in strong if not self._decoy(c, w)] or strong
        if strong:
            return strong[0], not more or sc[strong[0]][2] >= 4 * n
        return next((c for c in rank if sc[c][1] >= 0.75 * n and not sc[c][3]), None), False

    def _decoy(self, ident: str, w: _ResWalk) -> bool:
        names = {_res_name(a) for a in w.alts}
        return any((not n["merchant"] or _res_name(n["merchant"]) in names)
                   and any(_res_same(d[-1], v) or _res_same(v, d[-1]) for d in n["descriptors"] for _, v in w.desc)
                   for n in (self.readings.get(ident) or _RES_EMPTY)["not_versions"])

    def _enter(self, ep: Any, w: _ResWalk, t: str, cands: list) -> None:
        rec = ep.catalog[t]
        rec.chain_page = True
        w.visited.append(t)
        hop = w.ref.hops[-1]
        hop.chosen_id, hop.candidate_ids, hop.opened = t, list(cands[:8]), dict(rec.opened_by)
        hop.merchant = next((a for a in w.alts if self._is_m(ep, rec, [a])), hop.merchant)
        ptr = self._pointer(ep, t, w.ref.attr)
        ep.trace.decision("ref_hop", rid=w.ref.rid, stage="page", hop=len(w.visited), chosen=t,
                          title=clip_text(rec.title, 60), cands=len(cands),
                          pointer=(ptr["alts"][:1] + [ptr["src"]]) if ptr else None)
        if ptr is None:
            w.stage = "end"
        elif len(w.visited) >= w.limit:
            self._finish(ep, w, None, "hop limit")
        else:
            self._target(w, ptr)

    def _end(self, ep: Any, w: _ResWalk) -> Optional[Intent]:
        if not w.visited:
            return self._finish(ep, w, None, "no chain page")
        t = w.visited[-1]
        rec, tools = ep.catalog[t], ep.binder.binding.tools
        attrs_read = any(FR_ATTRS in tools[n].provides_seen for n in rec.opened_by if n in tools)
        if not rec.attrs and not attrs_read and not w.end_open and ep.binder.open_plan(NEED_ATTRS) \
                and ep.plan.can_spend(1, P_RESOLVE):
            w.end_open = True
            return Intent(IK_OPEN, {"ids": [t], "need": NEED_ATTRS, "chain_page": True}, P_RESOLVE, None,
                          why=f"{w.ref.rid}: chain end attributes")
        restated = next((r["val"] for r in (self.readings.get(t) or _RES_EMPTY)["restates"]
                         if _res_attr_match(r["attr"], w.ref.attr)), None)
        return self._finish(ep, w, _res_own_value(rec, w.ref.attr) or restated or self._title_value(ep, rec, w.ref.attr),
                            "chain end")

    def _title_value(self, ep: Any, rec: Record, attr: str) -> Optional[str]:
        """A value of attr seen on other records, written in the title or the notes."""
        vocab = {v for r in list(ep.catalog.values())[:400] if r is not rec for v in [_res_own_value(r, attr)]
                 if v and len(v) <= 40}
        text = _res_toks(f"{rec.title} {rec.notes or ''}")
        hits = [v for v in vocab if _res_has(text, _res_toks(v))]
        return max(hits, key=lambda v: (len(_res_toks(v)), v)) if hits else None

    def _finish(self, ep: Any, w: _ResWalk, v: Optional[str], why: str) -> None:
        ref = w.ref
        ref.hops = [h for h in ref.hops if h.chosen_id]
        ref.value, w.stage = v, "done"
        line = ep.ledger.lines[ref.line] if 0 <= ref.line < len(ep.ledger.lines) else None
        ref.status = "resolved" if v is not None else "partial" if ref.hops else "failed"
        if v is not None and ref.origin == "trap":
            self.effective[(w.item, ref.attr)] = v
            self.dirty.add(ref.line)
        elif v is not None and line is not None:
            op, vals = "ne" if ref.relation == "differ" else "eq", [v]
            if ref.relation == "pair":
                found = self._partner(ep, ref.attr, v, [h.chosen_id for h in ref.hops])
                if found is not None:
                    vals, ref.pair_rule, ref.obligation = [found[0]], {"from": v, "to": found[0], "span": found[1]}, None
                else:
                    op, ref.obligation, w.stage = "pair_pending", RES_PAIR_SEARCH, "pair"
            self._constrain(ep, line, ref, op, vals)
        ep.trace.decision("ref_hop", rid=ref.rid, stage=ref.status, why=why, attr=ref.attr, relation=ref.relation,
                          found=v, pages=[h.chosen_id for h in ref.hops], obligation=ref.obligation,
                          constraint=[(c.op, c.vals) for c in (line.constraints if line else []) if c.ref_id == ref.rid])
        if ref.origin != "trap":
            ep.plan.compute(f"reference {ref.rid} {ref.status}")

    def _constrain(self, ep: Any, line: Line, ref: Reference, op: str, vals: list) -> None:
        cid = f"{ref.rid}.ref"
        line.constraints = [c for c in line.constraints if c.cid != cid] + [Constraint(
            cid=cid, attr=canon_key(ref.attr), op=op, vals=list(vals), source="reference", span=ref.span or vals[0],
            ref_id=ref.rid)]
        ep.ledger.version += 1
        ep.coverage.invalidate(ep, line, "reference")
        self.dirty.add(line.idx)

    def _partner(self, ep: Any, attr: str, v: str, ids: list) -> Optional[tuple]:
        """(partner of v, rule span) from the notes of ids."""
        ids = [i for i in ids if i in ep.catalog and ep.catalog[i].notes]
        if ids:
            self._reading(ep, ids)
        for i in ids:
            reading = self.readings.get(i)
            for rule in (reading or _RES_EMPTY)["pair_rules"]:
                if not rule["attr"] or _res_attr_match(rule["attr"], attr):
                    for a, b in rule["pairs"]:
                        if _res_same(a, v) != _res_same(b, v):
                            return (b if _res_same(a, v) else a), rule["span"]
            if reading is None or reading.get("lossy"):
                found = _res_fb_partner(ep.catalog[i].notes, attr, v)
                if found is not None and not _res_same(found[0], v):
                    return found
        return None

    def _pair(self, ep: Any, w: _ResWalk) -> Optional[Intent]:
        ref = w.ref
        if not 0 <= ref.line < len(ep.ledger.lines):
            w.stage = "done"
            return None
        line = ep.ledger.lines[ref.line]
        pool = [i for i in dict.fromkeys(list(line.opened) + list(line.table)) if i in ep.catalog]
        known = [i for i in pool if self._known(ep, i)]
        found = self._partner(ep, ref.attr, ref.value, known)
        if found is not None:
            ref.pair_rule, ref.obligation, w.stage = {"from": ref.value, "to": found[0], "span": found[1]}, None, "done"
            self._constrain(ep, line, ref, "eq", [found[0]])
            ep.trace.decision("ref_hop", rid=ref.rid, stage="pair_rule", found=found[0], span=clip_text(found[1], 160))
            return None
        unopened = [i for i in pool if i not in known and i not in w.bad
                    and not (ep.ledger.in_stock and ep.catalog[i].stock is False)]
        if not unopened or w.pair_opens >= RES_PAIR_BATCHES or not ep.plan.can_spend(1, P_RESOLVE):
            ref.obligation, w.stage = None, "done"
            ep.trace.decision("ref_hop", rid=ref.rid, stage="pair_open", why="no pair rule found; stays pair_pending")
            return None
        w.pair_opens += 1
        return Intent(IK_OPEN, {"ids": unopened[:max(1, min(ep.plan.batch_k(), RES_NOTES_BATCH))], "need": self._need(ep)},
                      P_RESOLVE, line.idx, why=f"{ref.rid}: look for the pair rule in line {line.idx} candidates")

    def _reading(self, ep: Any, ids: list, optional: bool = False) -> Optional[dict]:
        """The NOTES reading of ids[0]; uncached ids go in one call (<= 6)."""
        for i in ids:
            if i in ep.catalog and not ep.catalog[i].notes:
                self.readings.setdefault(i, _RES_EMPTY)
        batch = [i for i in dict.fromkeys(ids) if i not in self.readings and i not in self.asked
                 and i in ep.catalog][:RES_NOTES_BATCH]
        if batch and ep.llm.allowed("NOTES", optional):
            self.asked.update(batch)
            recs = [ep.catalog[i] for i in batch]
            texts = {r.id: norm_space(r.notes) for r in recs}
            payload = {"request": ep.query, "recs": [{"rid": r.id, "head": clip_text(r.title, 160), "notes": r.notes}
                                                     for r in recs]}
            ans = ep.llm.ask("NOTES", payload, validator=lambda a: _res_check_notes(a, texts), optional=optional)
            if ans is not None:
                got = {x["rid"]: x for x in ans["recs"]}
                for r in recs:
                    reading = {k: got.get(r.id, _RES_EMPTY)[k] for k in _RES_EMPTY}
                    r.note_read = {k: list(v) for k, v in reading.items()}
                    self.readings[r.id] = dict(reading, lossy=got.get(r.id, {}).get("lossy", False))
        return self.readings.get(ids[0]) if ids else None

    def _pointer(self, ep: Any, ident: str, attr: str) -> Optional[dict]:
        rec = ep.catalog.get(ident)
        if rec is None or not rec.notes or not _res_toks(attr):
            return None
        reading = self._reading(ep, [ident])
        for p in (reading or _RES_EMPTY)["pointers"]:
            if _res_attr_match(p["attr"], attr):
                return {"alts": [p["merchant"]], "shelf": p["shelf_words"] or "", "desc": _res_descs(p["descriptors"]),
                        "span": p["span"], "src": "oracle"}
        return self._fb_pointer(ep, rec, attr) if reading is None or reading.get("lossy") else None

    def _vocab_of(self, ep: Any) -> tuple:
        """(catalog size, merchants, shelf stems, keys, value and title stems)."""
        if self._vocab[0] != len(ep.catalog):
            recs = list(ep.catalog.values())
            self._vocab = (len(ep.catalog), {_res_name(r.merchant) for r in recs if r.merchant},
                           {s for r in recs if r.shelf for s in _res_toks(r.shelf)},
                           {k for r in recs for k in r.attrs} | set(ep.fields.attr_names),
                           {t for r in recs for t in _res_toks(r.title + " " + " ".join(
                               _res_val(v) for pairs in r.attrs.values() for _, v in pairs))})
        return self._vocab

    def _fb_pointer(self, ep: Any, rec: Record, attr: str) -> Optional[dict]:
        """NOTES fallback by shape: a sentence naming attr in its first words, then a capitalised name (first word
        of the last capitalised run; known merchants, possessives first) that is not the item's own merchant, a
        title or shelf word or a word written in lower case; the parenthesised list, else the clauses after the
        name, give the descriptors; the other capitalised words the shelf."""
        at = _res_toks(attr)
        merchants, shelves, keys = self._vocab_of(ep)[1:4]
        own = {_res_name(rec.merchant), _res_name(rec.maker)} - {""}
        avoid = set(_res_toks(f"{rec.title} {rec.shelf or ''}")) | shelves | set(
            _res_toks(" ".join(x for x in re.findall(r"[A-Za-z]+", rec.notes or "") if x.islower())))
        for sent in _res_sentences(rec.notes):
            words = _RES_WORD_RE.findall(re.sub(r"(?<=\w)\(([^()]*)\)", r" \1 ", sent))   # "Milk(Ready to Drink)"
            flat = [(i, s) for i, x in enumerate(words) for s in _res_toks(x)]
            pos = next((k for k in range(len(flat)) if [s for _, s in flat[k:k + len(at)]] == at), None)
            if pos is None or flat[pos][0] >= RES_HEAD_WORDS:
                continue
            runs, cur, depth = [], [], 0
            for i in range(flat[pos + len(at) - 1][0] + 1, len(words)):
                x = words[i]
                depth += (x == "(") - (x == ")")
                if (depth == 0 and x[:1].isupper()) or (cur and (x == "&" or x.lower() in STOPWORDS)
                                                        and i + 1 < len(words) and words[i + 1][:1].isupper()):
                    cur.append(i)
                elif cur:
                    runs, cur = runs + [cur], []
            cands = []
            for run in reversed(runs + [cur] if cur else runs):
                mi = next((i for i in run if words[i][:1].isupper() and any(ch.islower() for ch in words[i])
                           and len(words[i]) >= 3), None)
                bare = re.sub(r"['’]s$", "", words[mi]) if mi is not None else ""
                nm = _res_name(bare)
                if nm and nm not in own and (nm in merchants or not set(_res_toks(bare)) & avoid):
                    cands.append((nm not in merchants, bare == words[mi], bare, mi))
            if not cands:
                continue
            bare, mi = sorted(cands, key=lambda c: c[:2])[0][2:]
            shelf = " ".join(words[i] for r in runs + [cur] for i in r if i != mi and words[i][:1].isupper())
            if "(" in words[mi:]:
                stop = words.index("(", mi)
                inner = words[stop + 1:words.index(")", stop) if ")" in words[stop:] else len(words)]
                parts = re.split(r"[,;]", " ".join(inner).replace(" ,", ",").replace(" ;", ";"))
            else:
                parts = _RES_SPLIT_RE.split(" " + " ".join(x for x in words[mi + 1:] if x not in "()").split(";")[0] + " ")
            desc = [_res_split_key(p.strip(), keys) for p in parts if set(_res_toks(p)) - set(_res_toks(shelf))][:6]
            entry = {"attr": attr, "merchant": bare, "shelf_words": shelf, "descriptors": [list(d) for d in desc],
                     "span": sent}
            if entry not in rec.note_read["pointers"]:
                rec.note_read["pointers"].append(entry)
            return {"alts": [c[2] for c in sorted(cands, key=lambda c: c[:2])[:3]], "shelf": shelf, "desc": desc,
                    "span": sent, "src": "fallback"}
        return None

    def _issue(self, ep: Any, it: Intent, w: Optional[_ResWalk]) -> Intent:
        self._sync(ep)
        self.issued.add(it.signature())
        self.last = [it, ep.transport.n, w, False]
        if w is not None and it.kind == IK_SEARCH:
            w.searches += 1
            w.queries += [] if it.payload["query"] in w.queries else [it.payload["query"]]
        elif w is not None and it.kind == IK_OPEN and w.stage == "locate" and it.payload.get("spec"):
            w.specs += 1
        elif w is not None and it.kind == IK_OPEN and w.stage == "locate":
            w.opens += 1
        return it

    def _sync(self, ep: Any) -> None:
        """Absorb an intent of this engine that ran in another loop (Q3 hook, procedure gate)."""
        if self.last is not None and not self.last[3]:
            self._absorb(ep, None)

    def _absorb(self, ep: Any, dec: Optional[Decoded]) -> None:
        it, u0, w, _ = self.last
        self.last[3] = True
        sent = ep.transport.n > u0 and (dec is None or dec.kind != D_BLOCKED)
        if it.kind == IK_SEARCH:
            rec = ep.queries.get(it.payload.get("qkey") or "")
            m = _res_name(it.payload.get("merchant_value"))
            if sent and rec is not None and m and not ep.contract.merchant_path_confirmed:
                rows = [ep.catalog[i] for i in rec.ids if i in ep.catalog]
                mh, kh = sum(_res_name(r.merchant) == m for r in rows), sum(_res_name(r.maker) == m for r in rows)
                if mh or kh:
                    self.mfield = FR_MAKER if kh > mh else FR_MERCHANT
                    ep.contract.merchant_path_confirmed = True
                    ep.trace.decision("contract", merchant_field=self.mfield, why="first merchant search",
                                      rows=len(rows), merchant_hits=mh, maker_hits=kh)
        elif it.kind == IK_OPEN and w is not None:
            ids = list(it.payload.get("ids") or [])
            opened = [i for i in ids if i in ep.catalog and max(ep.catalog[i].opened_by.values(), default=-1) > u0]
            if not sent or not opened:
                w.bad |= set(ids if not sent else ids[:1])
            elif w.stage == "pair" and it.line is not None and 0 <= it.line < len(ep.ledger.lines):
                ep.ledger.lines[it.line].opened += [i for i in opened if i not in ep.ledger.lines[it.line].opened]

_COV_SET_MAX = 4           # an allowed set with more values is not expanded into one query per value
_COV_RELAX_MAX = 2         # zero hits: drop the least selective value token at most twice
_COV_RUNGS = 6             # FILTER first pages per generation
_COV_PART_MAX = 4          # sub-queries per partition
_COV_TWIN_JACCARD = 0.6    # title-token Jaccard of a twin listing
_COV_TWIN_DF = 0.3         # twin-query tokens appear in under 30% of the titles seen

def _cov_stem(tok: str) -> str:
    """Light plural stemming (the checker's norm), so a request noun matches a label token."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("sses"):
        return tok[:-2]
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok

def _cov_toks(text: Any) -> list:
    return [t for t in content_tokens(text) if len(t) >= 2]

def _cov_stems(text: Any) -> frozenset:
    return frozenset(_cov_stem(t) for t in text_tokens(text))

_V5S_NP = ("stock", "stock_rule", "decline_rule", "procedure", "objective", "cap", "budget", "msgs", "steps")

def _v5s_votes(ep: Any, tags: tuple = _V5S_NP) -> list:
    """v5: each PARSE vote's spans of the rules in tags (two votes or none)."""
    sets = []
    for v in [v for v in ep.parse_votes if isinstance(v, dict)][:2] if isinstance(ep.parse_votes, list) else []:
        items = [v.get(k) for k in tags] + list(v.get("procedures") or [] if "procedure" in tags else [])
        out = [x.get("span") for x in items if isinstance(x, dict) and x.get("required") is not False]
        out += [s for s, route in v.get("_routes") or [] if route in tags]
        sets.append({norm_span(x) for x in out if isinstance(x, str)} - {""})
    return sets if len(sets) == 2 else []

def _v5s_np_spans(ep: Any) -> list:
    """Rule spans of one vote nested with one of the other's."""
    a, b = _v5s_votes(ep) or (set(), set())
    return sorted(s for s in a | b if any(s in t or t in s for t in (b if s in a else a)))

def _v5s_scrub(ep: Any) -> None:
    """Deactivate a requirement restating the stock rule ('stock: on hand'): each vote has a stock span nested with
    its span and holding its values (other rule sentences can state real requirements)."""
    cov, led = ep.coverage, ep.ledger
    key = (id(led), led.version, sum(len(l.constraints) for l in led.lines))
    if cov.v5s_key == key:
        return
    cov.v5s_key, votes, hit = key, _v5s_votes(ep, _V5S_NP[:2]), []
    for line in led.lines:
        for c in line.constraints:
            s, vt = norm_span(c.span), set(content_tokens(" ".join(map(str, c.vals))))
            if c.active and c.source in ("stated", "review") and s and vt and votes and all(any(
                    (s in t or t in s) and vt <= set(content_tokens(t)) for t in spans) for spans in votes):
                c.active = False
                cov.v5s_np.add(c.cid)
                hit.append([line.idx, c.attr, c.vals])
    if hit:
        ep.trace.decision("parse", scrubbed=hit, why="restates a rule")

@dataclass
class _CovLine:
    gen: int = 0
    sig: Optional[str] = None
    spent: int = 0                                   # coverage steps sent
    judged: dict = field(default_factory=dict)       # id -> (eligible, tier, (price, stock))
    xtier: dict = field(default_factory=dict)        # id -> (tier before opening, stamp)
    open_fail: set = field(default_factory=set)
    relax: dict = field(default_factory=dict)        # literal query -> relaxed texts
    dead: set = field(default_factory=set)           # texts whose first page failed or was blocked
    rungs: list = field(default_factory=list)        # [ceiling_c, qkey | None, count | None]
    parts: dict = field(default_factory=dict)        # sub-query -> (parent qkey, exhaustive, depth)
    split: set = field(default_factory=set)
    twin_q: Optional[str] = None
    lit_keys: set = field(default_factory=set)
    best: Optional[str] = None
    idle: bool = False
    filter_bad: bool = False                         # a bracket returned rows off the label or over the ceiling
    onshelf: set = field(default_factory=set)        # rows of brackets that held
    noted: set = field(default_factory=set)

class Coverage(Engine):
    name = "coverage"

    def __init__(self, ep: Any = None):
        super().__init__(ep)
        self.lines: dict = {}
        self.sent: set = set()           # intent signatures sent or blocked: never proposed again
        self.qkeys: dict = {}            # query text -> key of its first page
        self.cache: dict = {}
        self.zero_done: set = set()
        self.broken = False              # a coverage bug ends coverage, never the episode
        self.v5s_np: set = set()         # cids _v5s_scrub deactivated (v5)
        self.v5s_key: Optional[tuple] = None

    def _safe(self, ep: Any, where: str, default: Any, fn: Callable, *args: Any) -> Any:
        if self.broken:
            return default
        try:
            return fn(ep, *args)
        except Exception as exc:
            self.broken = True
            ep.trace.decision("crash", err=repr(exc)[:300], where="coverage." + where,
                              trace=clip_text(traceback.format_exc(), 1200))
            return default

    def st(self, line: Line) -> _CovLine:
        return self.lines.setdefault(line.idx, _CovLine())

    def _cached(self, name: str, key: Any, build: Callable) -> Any:
        hit = self.cache.get(name)
        if hit is None or hit[0] != key:
            hit = self.cache[name] = (key, build())
        return hit[1]

    @staticmethod
    def _purpose(ep: Any) -> str:
        return P_RECOVER if ep.budget.first_finish_step is not None else P_COVER

    def _note(self, ep: Any, line: Line, st: _CovLine, what: str, key: Any = "", **kw: Any) -> None:
        tag = what + "|" + str(key)
        if tag not in st.noted:
            st.noted.add(tag)
            ep.trace.decision("cert", what=what, line=line.idx, gen=st.gen, **kw)

    @staticmethod
    def _indexed(r: Record) -> set:
        """Tokens a search matches: title, shelf, merchant, maker, attribute values."""
        vals = " ".join(str(v) for bucket in r.attrs.values() for _, v in bucket)
        return set(text_tokens(" ".join((r.title or "", r.shelf or "", r.merchant or "", r.maker or "", vals))))

    def _stats(self, ep: Any) -> tuple:
        """(listings holding each indexed token, share of titles holding each token)."""
        def build() -> tuple:
            df: dict = {}
            tf: dict = {}
            for r in ep.catalog.values():
                for t in self._indexed(r):
                    df[t] = df.get(t, 0) + 1
                for t in set(text_tokens(r.title)):
                    tf[t] = tf.get(t, 0) + 1
            n = max(1, sum(1 for r in ep.catalog.values() if r.title))
            return df, {t: c / n for t, c in tf.items()}
        return self._cached("stats", (len(ep.catalog), ep.transport.n), build)

    def _rank(self, ep: Any, toks: list) -> list:
        """Most selective first (fewest listings), then the given order."""
        df, toks = self._stats(ep)[0], _dedupe(toks)
        return sorted(toks, key=lambda t: (df.get(t, 0), toks.index(t)))

    def _max_len(self, ep: Any) -> Optional[int]:
        tool = ep.binder.tool_for(ROLE_SEARCH)
        arg = ep.binder.slot_arg(tool, SLOT_QUERY) if tool else None
        feat = ep.binder.features[tool].args.get(arg) if arg else None
        return feat.max_len if feat is not None else None

    def _banned(self, ep: Any) -> frozenset:
        _v5s_scrub(ep)                  # before any search
        led = ep.ledger
        key = (id(led), led.version, len(ep.catalog), len(led.refs), len(led.sentences),
               tuple((len(l.constraints), len(l.menu), l.shelf_label, l.words) for l in led.lines))
        return self._cached("ban", key, lambda: self._build_ban(ep))

    @staticmethod
    def _build_ban(ep: Any) -> frozenset:
        led = ep.ledger
        rescue, names, neg, story, titles = set(), set(), set(), set(), set()
        for line in led.lines:
            rescue.update(text_tokens(" ".join((line.words, line.shelf_words, line.shelf_label or ""))))
            for c in line.constraints:
                names.update(text_tokens(c.attr))
                if c.cid in ep.coverage.v5s_np:
                    continue
                vals = {t for v in c.vals for t in text_tokens(v)}
                if c.op == "ne":
                    neg |= vals | (set(content_tokens(c.span)) - set(text_tokens(c.attr)))
                else:
                    rescue |= vals
            names.update(t for k in line.menu for t in text_tokens(k))
        for ref in led.refs:
            for src in [ref.source or {}] + [vars(h) for h in ref.hops]:
                pairs = [str(d[1]) for d in src.get("descriptors") or [] if isinstance(d, (list, tuple)) and len(d) > 1]
                rescue.update(text_tokens(" ".join([str(src.get("merchant") or ""), str(src.get("shelf_words") or "")]
                                                   + pairs)))
        for r in ep.catalog.values():
            rescue.update(text_tokens(" ".join((r.merchant or "", r.maker or "", r.shelf or ""))))
            titles.update(text_tokens(r.title))
        for span, route in led.sentences:
            if route not in REQUIREMENT_ROUTES:
                story.update(content_tokens(span))
        rules = {t for s in _v5s_np_spans(ep) for t in content_tokens(s)}
        return frozenset(((names | neg) - rescue) | (story - rescue - titles) | (rules - rescue))

    def hygiene_ok(self, ep: Any, query: str) -> tuple:
        return self._safe(ep, "hygiene_ok", (bool(str(query or "").strip()), ""), self._hygiene, query)

    def _hygiene(self, ep: Any, query: str) -> tuple:
        cap = max(1, ep.contract.token_cap)
        text, limit = str(query or ""), self._max_len(ep)
        toks = text_tokens(trim_at_token(text, limit) if limit and len(text) > limit else text)
        if not toks:
            return False, "empty query"
        if len(toks) > cap:
            return False, f"{len(toks)} tokens over the token cap {cap}"
        banned = self._banned(ep)
        for t in toks:
            if len(t) < 2 and t not in STOPWORDS:
                return False, f"one-character token {t!r}"
            if t in ep.contract.unsearchable:
                return False, f"token {t!r} was learned unsearchable"
            if t in banned:
                return False, f"{t!r} is an attribute name, a negation or a story word"
        return True, ""

    def _token_ok(self, ep: Any, t: str) -> bool:
        return len(t) >= 2 and t not in STOPWORDS and t not in ep.contract.unsearchable and t not in self._banned(ep)

    @staticmethod
    def _label_ok(ep: Any, line: Line) -> bool:
        """shelf_conf 2, or the scout's rule on the line's rows (three rows, a shared word)."""
        if not line.shelf_label:
            return False
        if line.shelf_conf >= 2:
            return True
        lab = canon_key(line.shelf_label)
        on = sum(1 for i in line.table if i in ep.catalog and canon_key(ep.catalog[i].shelf or "") == lab)
        return on >= 3 and bool(_cov_stems(line.shelf_label) & _cov_stems(line.words + " " + line.shelf_words))

    def _noun(self, ep: Any, line: Line) -> list:
        """Two label tokens (confirmed or sharing a word with the line), else the line's noun."""
        if line.shelf_label and (self._label_ok(ep, line) or _cov_stems(line.shelf_label)
                                 & _cov_stems(line.words + " " + line.shelf_words)):
            toks = [t for t in _cov_toks(line.shelf_label) if self._token_ok(ep, t)]
            if toks:
                return self._rank(ep, toks)[:2]
        return [t for t in _cov_toks(line.shelf_words or line.words) if self._token_ok(ep, t)][:2]

    def _fit(self, ep: Any, vals: list, noun: list) -> str:
        """Values then nouns within the token cap and maxLength."""
        cap = max(1, ep.contract.token_cap)
        vals = [t for t in _dedupe(vals) if t not in noun and self._token_ok(ep, t)]
        noun = list(noun[:max(1, cap - len(vals))]) if noun else []
        vals = vals[:max(0, cap - len(noun))]
        limit = self._max_len(ep)
        while limit and vals and len(" ".join(vals + noun)) > limit:
            vals.pop()
        while limit and len(noun) > 1 and len(" ".join(vals + noun)) > limit:
            noun.pop()
        return " ".join(vals + noun)

    def _bases(self, ep: Any, line: Line) -> list:
        """[(allowed-set value | None, query)]."""
        noun, eq, sets = self._noun(ep, line), [], []
        for c in line.constraints:
            if c.active and c.op == "eq":
                eq += [t for v in c.vals for t in _cov_toks(v)]
            elif c.active and c.op == "in" and c.vals:
                sets.append(list(c.vals))
        sets.sort(key=len)
        eq, out = self._rank(ep, eq), []
        for v in (sets[0] if sets and len(sets[0]) <= _COV_SET_MAX else [None]):
            q = self._fit(ep, (self._rank(ep, _cov_toks(v)) if v is not None else []) + eq, noun)
            if q and q not in [t for _, t in out]:
                out.append((v, q))
        return out

    def value_queries(self, ep: Any, line: Line) -> list:
        return [text for _, text in self._bases(ep, line)]

    def _literal(self, ep: Any, line: Line, text: str, v: Optional[str] = None) -> bool:
        """Only necessary tokens: eq values, confirmed-label tokens and one allowed-set value v (stem-equal)."""
        need = {_cov_stem(t) for c in line.constraints if c.active and c.op == "eq" for x in c.vals
                for t in text_tokens(x)}
        need |= (_cov_stems(line.shelf_label) if self._label_ok(ep, line) else set()) | _cov_stems(v or "")
        return bool(text_tokens(text)) and _cov_stems(text) <= need

    def _rec(self, ep: Any, text: str) -> Optional[QueryRec]:
        """Recorded key, the executor's key, else a search record with the same stems."""
        key = self.qkeys.get(text)
        if key is None:
            call, _ = ep.builder.compile(Intent(IK_SEARCH, {"query": text}, P_COVER))
            key = QueryBook.key_of(call.tool, call.args, tuple(a for a, s in call.slots_used if s in PAGING_SLOTS)) \
                if call is not None else None
        rec = ep.queries.get(key) if key else None
        if rec is None and text:
            tool, want = ep.binder.tool_for(ROLE_SEARCH), _cov_stems(text)
            rec = next((r for r in ep.queries.recs.values() if r.tool == tool and r.query_text
                        and _cov_stems(r.query_text) == want), None)
        return rec

    @staticmethod
    def _zero(rec: QueryRec) -> bool:
        return rec.count == 0 or (not rec.ids and not rec.has_cont)

    @staticmethod
    def _complete(rec: Optional[QueryRec]) -> bool:
        return rec is not None and not rec.has_cont and not rec.capped and (
            rec.count is None or len(set(rec.ids)) >= rec.count)

    @staticmethod
    def _beyond(ep: Any, rec: Optional[QueryRec]) -> bool:
        """Capped, or more rows than a known window (an unknown window is paged)."""
        W = ep.contract.W
        return rec is not None and (rec.capped or (W is not None and rec.count is not None and rec.count > W))

    def _lits(self, ep: Any, line: Line, st: _CovLine, relax: bool = True) -> list:
        """[(value, text, rec)]: the most selective literal query read stands in; zero hits drop the least selective
        value token (twice at most)."""
        out, tool, noun = [], ep.binder.tool_for(ROLE_SEARCH), set(self._noun(ep, line))
        for v, base in self._bases(ep, line):
            need = _cov_stems(" ".join(t for t in text_tokens(base) if t not in noun))
            live = [r for r in ep.queries.recs.values() if r.tool == tool and r.query_text and not self._zero(r)
                    and need <= _cov_stems(r.query_text) and self._literal(ep, line, r.query_text, v)]
            if live:
                rec = min(live, key=lambda r: (r.count if r.count is not None else 10 ** 9, -len(r.ids), r.query_text))
                out.append((v, rec.query_text, rec))
                continue
            chain = [base] + st.relax.get(base, [])
            text = chain[-1]
            rec = self._rec(ep, text)
            if relax and len(chain) <= _COV_RELAX_MAX and (text in st.dead or (rec is not None and self._zero(rec))):
                df, toks = self._stats(ep)[0], text_tokens(text)
                drop = [t for t in toks if t not in noun]
                if drop:
                    worst = max(drop, key=lambda t: (df.get(t, 0), toks.index(t)))
                    nxt = " ".join(t for t in toks if t != worst)
                    if nxt and nxt not in chain:
                        st.relax.setdefault(base, []).append(nxt)
                        text, rec = nxt, self._rec(ep, nxt)
            out.append((v, text, rec))
        st.lit_keys = {rec.canon_args for _, _, rec in out if rec is not None}
        return out

    def _mk(self, ep: Any, kind: str, payload: dict, line: Line, cert: bool, why: str) -> Optional[Intent]:
        if cert:
            payload["certificate"] = True
        it = Intent(kind, payload, self._purpose(ep), line.idx, why=f"cover line {line.idx}: {why}")
        return None if it.signature() in self.sent else it

    def _search(self, ep: Any, line: Line, st: _CovLine, text: str, qkind: str, cert: bool = False):
        ok, why = self.hygiene_ok(ep, text)
        if not ok:
            st.dead.add(text)
            self._note(ep, line, st, "hygiene", key=text, query=text, why=why)
            return None
        return self._mk(ep, IK_SEARCH, {"query": text, "qkind": qkind}, line, cert, f"{qkind[2:]} query '{text}'")

    def _page(self, ep: Any, line: Line, rec: QueryRec, cert: bool, base: Optional[dict] = None):
        """The next page with the query's own continuation, verbatim (never guessed, G8)."""
        if not rec.has_cont or rec.next_cont is None:
            return None
        payload = dict(base or {"query": rec.query_text})
        payload.update({"cont": rec.next_cont, "qkey": rec.canon_args, "qkind": rec.kind})
        label = rec.query_text or f"bracket {payload.get('ceiling')}"
        return self._mk(ep, IK_FILTER if base else IK_SEARCH, payload, line, cert,
                        f"next page of '{clip_text(label, 50)}'")

    def _open(self, ep: Any, line: Line, ids: list, cert: bool, why: str) -> Optional[Intent]:
        _, chunk = ep.binder.route_open(ids, NEED_ATTRS) if ids else (None, [])
        return self._mk(ep, IK_OPEN, {"ids": list(chunk), "need": NEED_ATTRS}, line, cert, why) if chunk else None

    @staticmethod
    def _opened(ep: Any, r: Record) -> bool:
        """Read by a record tool that shows attributes (any record tool while none is known to)."""
        if not r.opened_by:
            return False
        if FR_ATTRS in r.families_seen:
            return True
        tools = {n for n in ep.binder.record_tools() if FR_ATTRS in ep.binder.binding.tools[n].provides_seen}
        return not tools or bool(tools & set(r.opened_by))

    def _on_shelf(self, ep: Any, line: Line, r: Record, st: _CovLine) -> bool:
        """An unconfirmed label (perhaps a mislearned field) never excludes a row."""
        return not (r.shelf and self._label_ok(ep, line)) or canon_key(r.shelf) == canon_key(line.shelf_label) \
            or r.id in st.onshelf

    def _pool(self, ep: Any, line: Line, st: _CovLine, lits: bool = True) -> list:
        """The line's table and opened ids plus the rows of every query coverage used for it."""
        ids = list(line.table) + list(line.opened)
        keys = [self.qkeys.get(t) for t in list(st.parts) + [st.twin_q or ""]] + [r[1] for r in st.rungs]
        for key in keys + (sorted(st.lit_keys) if lits else []):
            rec = ep.queries.get(key) if key else None
            ids += rec.ids if rec is not None else []
        return _dedupe(ids)

    @staticmethod
    def _id_sorted(ep: Any, rec: QueryRec) -> bool:
        """Id order: the learned contract, else the query's own rows (>= 4, ids ascending, prices not)."""
        if ep.contract.sort != SORT_UNKNOWN:
            return ep.contract.sort == SORT_ID
        prices = [ep.catalog[i].price_c for i in rec.ids if i in ep.catalog]
        return len(rec.ids) >= 4 and rec.ids == sorted(rec.ids) and None not in prices and prices != sorted(prices)

    def _excluded(self, ep: Any, line: Line, st: _CovLine) -> set:
        """Rows every literal query would have shown if they held its tokens, and did not."""
        def build() -> set:
            out: Optional[set] = None
            for v, text, rec in self._lits(ep, line, st, relax=False):
                if rec is None or not rec.ids or not self._literal(ep, line, text, v):
                    return set()
                top = None if self._complete(rec) else max(rec.ids) if self._id_sorted(ep, rec) else ""
                if top == "":
                    return set()
                excl = {i for i in set(self._pool(ep, line, st, lits=False)) - set(rec.ids) if top is None or i <= top}
                out = excl if out is None else out & excl
            return out or set()
        return self._cached(f"excl{line.idx}", (ep.transport.n, st.gen, len(ep.queries.recs)), build)

    def _xtier(self, ep: Any, line: Line, st: _CovLine, r: Record, lb: Optional[int]) -> str:
        """The checker's tier before opening (stock, shelf, budget, title contradictions)."""
        stamp = (r.price_c, r.stock, lb)
        hit = st.xtier.get(r.id)
        if hit is None or hit[1] != stamp:
            hit = st.xtier[r.id] = (ep.checker.verdict(ep, line, r.id, lb).tier, stamp)
        return hit[0]

    def _unverified(self, ep: Any, line: Line, st: _CovLine, below: Optional[int], lb: Optional[int],
                    prio: frozenset = frozenset(), limit: Optional[int] = None, hits: Optional[set] = None,
                    failed: bool = False) -> list:
        """Rows to open, priority then price (`hits`: title hits first); `failed` counts failed opens."""
        excl = self._excluded(ep, line, st) if line.cheapest else set()
        rows = []
        for i in self._pool(ep, line, st):
            r = ep.catalog.get(i)
            if r is None or r.price_c is None or (i in st.open_fail and not failed) or i in ep.absent_ids \
                    or i in excl or self._opened(ep, r) or (below is not None and r.price_c >= below) \
                    or (ep.ledger.in_stock and r.stock is False) or (lb is not None and line.qty * r.price_c > lb) \
                    or not self._on_shelf(ep, line, r, st):
                continue
            rows.append(r)
        rows.sort(key=(lambda r: (-len(hits & set(text_tokens(r.title))), r.price_c, r.id)) if hits else
                  (lambda r: (r.id not in prio, r.price_c, r.id)))
        out = []
        for r in rows:
            if self._xtier(ep, line, st, r, lb) != "X":
                out.append(r.id)
                if limit and len(out) >= limit:
                    break
        return out

    def _judge(self, ep: Any, line: Line, st: _CovLine, lb: Optional[int]) -> Optional[int]:
        """Checker verdicts on opened rows not judged at their current price and stock; returns the best price. The
        line's eligible class only (T1/T2P on cheapest lines): eligible() falls back to T2/T3 within one batch, and
        such a row as the best stopped the ladder and certified a best that selection never takes (v2_d)."""
        todo = [i for i in self._pool(ep, line, st) if i in ep.catalog and self._opened(ep, ep.catalog[i])
                and (st.judged.get(i) or (0, 0, None))[2] != (ep.catalog[i].price_c, ep.catalog[i].stock)]
        if todo:
            ep.checker.eligible(ep, line, todo, lb, subset=True)     # verdicts and the JUDGE round (a subset call)
            for i in todo:
                v = line.verdicts.get(i)
                st.judged[i] = (v is not None and ep.checker.is_eligible(line, v), v.tier if v is not None else "T3",
                                (ep.catalog[i].price_c, ep.catalog[i].stock))
        pool = [(ep.catalog[i].price_c, tier_rank(j[1]), i) for i, j in st.judged.items()
                if j[0] and i in ep.catalog and ep.catalog[i].price_c is not None]
        best = min(pool) if pool else None
        st.best = best[2] if best else None
        return best[0] if best else None

    def _priority(self, ep: Any, st: _CovLine, lits: list) -> frozenset:
        """Hybrid intersection: bracket rows a value query also returned go first."""
        bracket = {i for _, key, _ in st.rungs if ep.queries.get(key or "") for i in ep.queries.get(key).ids}
        return frozenset(bracket & {i for _, _, rec in lits if rec is not None for i in rec.ids})

    def _filter_ready(self, ep: Any, line: Line, st: _CovLine) -> bool:
        """FILTER(shelf, ceiling) on the confirmed label compiles (in a filter mode when the tool has one)."""
        if st.filter_bad or not self._label_ok(ep, line):
            return False
        tool = ep.binder.tool_for(ROLE_FILTER)
        if tool and ep.contract.filter_honours_keywords is None and not ep.binder.slot_arg(tool, SLOT_QUERY):
            ep.contract.filter_honours_keywords = False
            ep.trace.decision("contract", filter_honours_keywords=False, why="the filter tool takes no keywords")
        call, _ = ep.builder.compile(Intent(IK_FILTER, {"shelf": line.shelf_label, "ceiling": 1}, P_COVER))
        if call is None or not ep.binder.slot_arg(call.tool, SLOT_CEILING):
            return False
        mode = ep.binder.slot_arg(call.tool, SLOT_MODE)
        return mode is None or mode in call.args

    @staticmethod
    def _anchor(ep: Any, line: Line, best_p: Optional[int], lb: Optional[int]) -> Optional[int]:
        """Ceiling reaching every price below the best (best - 1 if inclusive, else best); else the budget per unit."""
        if best_p is not None:
            return best_p - (1 if ep.contract.ceiling_inclusive is True else 0)
        return lb // max(1, line.qty) if lb is not None and lb > 0 else None

    def _bracket(self, ep: Any, line: Line, st: _CovLine, best_p: Optional[int], lb: Optional[int]):
        """(ceiling, key, rec) of a complete bracket reaching the anchor (certificate B)."""
        anchor = self._anchor(ep, line, best_p, lb)
        if anchor is None or st.filter_bad:
            return None
        for c, key, n in sorted(st.rungs, key=lambda x: x[0]):
            rec = ep.queries.get(key) if key else None
            if c >= anchor and rec is not None and not rec.has_cont and (n is None or len(set(rec.ids)) >= n):
                return c, key, rec
        return None

    def _next_ceiling(self, ep: Any, line: Line, st: _CovLine, anchor: int, target: int) -> Optional[int]:
        """The anchor, then about `target` rows above the highest complete rung, interpolated to the lowest
        too-large rung over seen label prices (linear under four)."""
        tried = {r[0] for r in st.rungs}
        if st.rungs and st.rungs[-1][1] is None:
            return None                                       # the last bracket failed
        done = [(c, n or 0) for c, key, n in st.rungs if self._complete(ep.queries.get(key) if key else None)]
        lo, n_lo = max(done) if done else (0, 0)
        big = [(c, n) for c, key, n in st.rungs if lo < c <= anchor and n is not None and (c, n) not in done]
        if not big:
            return anchor if anchor not in tried and anchor > lo else None
        hi, n_hi = min(big)
        lab = canon_key(line.shelf_label)
        seen = sorted(r.price_c for r in ep.catalog.values() if r.price_c is not None and lo < r.price_c < hi
                      and r.shelf and canon_key(r.shelf) == lab)
        frac = min(1.0, target / max(1, n_hi - n_lo))
        nxt = seen[max(0, int(len(seen) * frac) - 1)] if len(seen) >= 4 else lo + int((hi - lo) * frac)
        return nxt if lo < nxt < hi and nxt not in tried else None

    def _ladder(self, ep: Any, line: Line, st: _CovLine, room: int, soft: bool, best_p: Optional[int],
                lb: Optional[int]) -> Optional[Intent]:
        """Rungs fitting W and half the room (>= 3P rows) are paged to the end."""
        target = min(ep.contract.W or 10 ** 6, ep.contract_learner.page_size() * max(3, room // 2))
        anchor = self._anchor(ep, line, best_p, lb)
        if anchor is None or self._bracket(ep, line, st, best_p, lb) is not None:
            return None
        for c, key, n in st.rungs:
            rec = ep.queries.get(key) if key else None
            if rec is not None and rec.has_cont and n is not None and (n <= target or len(rec.pages_read) > 1) \
                    and not self._beyond(ep, rec):
                it = self._page(ep, line, rec, True, {"shelf": line.shelf_label, "ceiling": c})
                if it is not None:
                    return it
        if soft or room < 2 or len(st.rungs) >= _COV_RUNGS or ep.plan.overrun_risk(2):
            return None
        c = self._next_ceiling(ep, line, st, anchor, target)
        if c is None:
            return None
        return self._mk(ep, IK_FILTER, {"shelf": line.shelf_label, "ceiling": c,
                                        "qkind": "q:ladder" if st.rungs else "q:bracket"}, line, False,
                        f"shelf bracket at ceiling {c}")

    def _split(self, ep: Any, line: Line, rec: QueryRec) -> tuple:
        """(kind, sub-queries): allowed-set values (exhaustive), an unconstrained key's values, merchants."""
        label = _cov_stems(line.shelf_label or "")
        toks = [t for t in text_tokens(rec.query_text) if _cov_stem(t) not in label]
        toks += [t for t in self._noun(ep, line) if _cov_stem(t) in label][:1]
        free, have, groups = ep.contract.token_cap - len(toks), set(toks), []
        if free < 1:
            return "none", []
        for c in line.constraints:
            if c.active and c.op == "in" and 1 < len(c.vals) <= _COV_PART_MAX \
                    and not any(set(_cov_toks(v)) <= have for v in c.vals):
                groups.append(("values", [_cov_toks(v) for v in c.vals]))
        opened = [ep.catalog[i] for i in line.opened if i in ep.catalog and ep.catalog[i].attrs]
        if len(opened) >= 2:
            common = set(opened[0].attrs).intersection(*[set(r.attrs) for r in opened[1:]])
            best: list = []
            for k in sorted(common - {canon_key(c.attr) for c in line.constraints if c.active}):
                vals = _dedupe([str(r.attrs[k][0][1]) for r in opened])
                if len(best) < len(vals) <= _COV_PART_MAX and all(_cov_toks(v) and len(v) <= 30 for v in vals):
                    best = vals
            if len(best) >= 2:
                groups.append(("key", [_cov_toks(v) for v in best]))
        counts: dict = {}
        for i in rec.ids:
            m = ep.catalog[i].merchant if i in ep.catalog else None
            if m:
                counts[m] = counts.get(m, 0) + 1
        if counts:
            groups.append(("merchants", [_cov_toks(m) for m in sorted(counts, key=lambda m: (-counts[m], m))
                                         [:_COV_PART_MAX]]))
        for kind, vals in groups:
            subs = _dedupe([" ".join(_dedupe(toks + [t for t in v if t not in have][:free])) for v in vals])
            subs = [s for s in subs if s != rec.query_text and self.hygiene_ok(ep, s)[0]]
            if subs:
                return kind, subs
        return "none", []

    def _partition(self, ep: Any, line: Line, st: _CovLine, lits: list, room: int) -> Optional[Intent]:
        if room < 2:
            return None
        parents = [(rec, 1) for _, _, rec in lits if rec is not None]
        parents += [(self._rec(ep, s), d + 1) for s, (_, _, d) in list(st.parts.items()) if d < 2]
        for rec, depth in parents:
            if not self._beyond(ep, rec):
                continue
            if rec.canon_args not in st.split:
                st.split.add(rec.canon_args)
                kind, subs = self._split(ep, line, rec)
                for s in subs:
                    st.parts.setdefault(s, (rec.canon_args, kind == "values", depth))
                self._note(ep, line, st, "partition", key=rec.canon_args, parent=rec.query_text, by=kind,
                           parts=len(subs))
            for s, (parent, _, _) in list(st.parts.items()):
                srec = self._rec(ep, s) if parent == rec.canon_args else None
                if parent == rec.canon_args and srec is None and s not in st.dead:
                    it = self._search(ep, line, st, s, "q:part")
                elif srec is not None and srec.has_cont and not self._beyond(ep, srec):
                    it = self._page(ep, line, srec, False)
                else:
                    continue
                if it is not None:
                    return it
        return None

    def _parts_cert(self, ep: Any, st: _CovLine, lits: list) -> Optional[list]:
        """Keys when every literal query is complete or split into complete exhaustive parts (depth <= 2)."""
        keys: list = []

        def complete(rec: Optional[QueryRec], depth: int) -> bool:
            if self._complete(rec):
                keys.append(rec.canon_args)
                return True
            subs = [s for s, (p, full, _) in st.parts.items() if rec is not None and p == rec.canon_args and full]
            return depth < 2 and bool(subs) and all(complete(self._rec(ep, s), depth + 1) for s in subs)
        return keys if st.parts and lits and all(complete(rec, 0) for _, _, rec in lits) else None

    def _twins(self, ep: Any, line: Line, st: _CovLine, room: int, best_p: Optional[int], lb: Optional[int]):
        """Distinctive title tokens of the best; title twins at or below its price are opened."""
        best = ep.catalog.get(st.best) if st.best else None
        if best is None or best_p is None or room < 2 or ep.plan.overrun_risk(2):
            return None
        if st.twin_q is None:
            share = self._stats(ep)[1]
            toks = sorted((t for t in _dedupe(_cov_toks(best.title)) if share.get(t, 0) < _COV_TWIN_DF
                           and self._token_ok(ep, t)), key=lambda t: share.get(t, 0))
            st.twin_q = self._fit(ep, toks[:3], []) if toks else ""
        if not st.twin_q:
            return None
        if self._rec(ep, st.twin_q) is None:
            return None if st.twin_q in st.dead else self._search(ep, line, st, st.twin_q, "q:twin")
        twins = [i for i in self._unverified(ep, line, st, best_p + 1, lb)
                 if token_overlap(ep.catalog[i].title, best.title) >= _COV_TWIN_JACCARD]
        return self._open(ep, line, twins[:max(1, ep.plan.batch_k())], False, "open twin listings")

    @staticmethod
    def _dx_lb(ep: Any, line: Line) -> Optional[int]:
        """The line budget; None once an over-cap basket leaves the line none (then any cheaper row helps)."""
        lb = ep.selector.line_budget(ep, line)
        return lb if lb is None or lb > 0 else None

    def _dx_tight(self, ep: Any, line: Line, st: _CovLine) -> bool:
        """A non-cheapest line no known row fits within its budget is covered like a cheapest line (cheaper rows
        first), so an over-cap basket rests on a search for cheaper items (DESIGN 5.15, 5.18 OVERCAP)."""
        lb = ep.selector.line_budget(ep, line)
        if line.cheapest or lb is None or lb <= 0:
            return lb is not None and not line.cheapest
        return not any(j[0] and line.qty * (ep.catalog[i].price_c or 0) <= lb for i, j in st.judged.items()
                       if i in ep.catalog) and not self._unverified(ep, line, st, None, lb, limit=1)

    def _room(self, ep: Any, line: Line, st: _CovLine) -> int:
        """C_l plus idle lines' unspent share, within can_spend(COVER) and u_c_hard; rem - need - 2 in recovery."""
        plan = ep.plan
        rem, need = plan.rem(), plan.need_finish()
        if ep.budget.first_finish_step is not None:
            return rem - need - 2
        C = plan.nums.get("C") or []
        own = (lambda o: C[o.idx] if o.idx < len(C) else 0)
        spare = sum(max(0, own(o) - self.st(o).spent) for o in ep.ledger.lines if o.idx != line.idx and self.st(o).idle)
        return min(own(line) + spare - st.spent, rem - need - plan.R - plan.protected(),
                   plan.nums.get("u_c_hard", 10 ** 6) - plan.u())

    def _page_lit(self, ep: Any, line: Line, st: _CovLine, rec: Optional[QueryRec], room: int,
                  best_p: Optional[int], lb: Optional[int], filt: bool, partial: bool = False) -> Optional[Intent]:
        """Full paging if pages + 2 fit; `partial`: a page while 3 steps remain (no filter); price-sorted: while
        every row read is below the best."""
        if rec is None or not rec.has_cont or self._beyond(ep, rec):
            return None
        if ep.contract.sort == SORT_PRICE:
            seen = [ep.catalog[i].price_c for i in rec.ids if i in ep.catalog and ep.catalog[i].price_c is not None]
            if partial or (best_p is not None and seen and max(seen) >= best_p) \
                    or self._unverified(ep, line, st, best_p, lb, limit=1):
                return None
            return self._page(ep, line, rec, True)
        P = ep.contract_learner.page_size()
        left = ceil_div(max(0, (rec.count if rec.count is not None else len(rec.ids) + P) - len(set(rec.ids))), P)
        if partial:
            return None if filt or room < 3 else self._page(ep, line, rec, False)
        if left + 2 > room or ep.plan.overrun_risk(left + 2):
            self._note(ep, line, st, "degrade", key=rec.canon_args, component="paging", pages=left, room=room,
                       then="bracket" if filt else "partial paging")
            return None
        return self._page(ep, line, rec, True)

    def _lit_cert(self, ep: Any, line: Line, lits: list, best_p: Optional[int]) -> Optional[str]:
        """A: every literal query complete; P: price-sorted, each read to a row >= the best or to its end."""
        if not lits or not any(rec is not None and rec.ids for _, _, rec in lits):
            return None
        price_sorted = ep.contract.sort == SORT_PRICE
        for v, text, rec in lits:
            if rec is None or not self._literal(ep, line, text, v):
                return None
            seen = [ep.catalog[i].price_c for i in rec.ids if i in ep.catalog and ep.catalog[i].price_c is not None]
            if not (self._complete(rec) or (price_sorted and best_p is not None and seen and max(seen) >= best_p)):
                return None
        return "P" if price_sorted else "A"

    def _plan_cheapest(self, ep: Any, line: Line, st: _CovLine, room: int, soft: bool) -> Optional[Intent]:
        lits = self._lits(ep, line, st)
        bases, started = set(self.value_queries(ep, line)), any(rec is not None for _, _, rec in lits)
        for _, text, rec in lits:                             # 1. first pages; an allowed set completes A
            if rec is None and text not in st.dead and (started or not soft):
                it = self._search(ep, line, st, text, "q:value" if text in bases else "q:relaxed", cert=started)
                if it is not None:
                    return it
        lb = self._dx_lb(ep, line)
        best_p, filt = self._judge(ep, line, st, lb), self._filter_ready(ep, line, st)
        for _, _, rec in lits:                                # 2. full paging inside the window
            it = self._page_lit(ep, line, st, rec, room, best_p, lb, filt)
            if it is not None:
                return it
        ids = self._unverified(ep, line, st, best_p, lb, self._priority(ep, st, lits),    # 3. verify ascending
                               limit=max(1, ep.plan.batch_k()))
        it = self._open(ep, line, ids, True, "verify the cheapest unverified rows")
        for _, _, rec in ([] if soft or it is not None else lits):   # partial paging (no filter)
            it = self._page_lit(ep, line, st, rec, room, best_p, lb, filt, partial=True)
            if it is not None:
                break
        if it is not None:
            return it
        complete = self._lit_cert(ep, line, lits, best_p) is not None
        if filt and not complete:                             # 4. anchored ladder, bracket paging
            it = self._ladder(ep, line, st, room, soft, best_p, lb)
            if it is not None:
                return it
        if soft or complete or self._bracket(ep, line, st, best_p, lb) is not None:
            return None
        it = self._partition(ep, line, st, lits, room)        # 5. partitions, then visible pages of past-window queries
        for _, _, rec in ([] if it is not None or filt or room < 3 else lits):
            it = self._page(ep, line, rec, False) if rec is not None and self._beyond(ep, rec) else None
            if it is not None:
                break
        return it or self._twins(ep, line, st, room, best_p, lb)   # 6. twin sweep

    def _plan_other(self, ep: Any, line: Line, st: _CovLine, room: int, soft: bool) -> Optional[Intent]:
        """Non-cheapest lines stop at the first T1 item plus one backup."""
        if soft:
            return None
        lb = self._dx_lb(ep, line)
        self._judge(ep, line, st, lb)
        ok = [i for i, j in st.judged.items() if j[0]]
        if len(ok) >= 2 and any(st.judged[i][1] == "T1" for i in ok):
            return None
        hits = {t for c in line.constraints if c.active and c.op in ("eq", "in") for v in c.vals for t in text_tokens(v)}
        it = self._open(ep, line, self._unverified(ep, line, st, None, lb, limit=max(1, ep.plan.batch_k()),
                                                   hits=hits or None), False, "open candidates for a T1 and a backup")
        for _, text, rec in ([] if it is not None else self._lits(ep, line, st)):
            if rec is None and text not in st.dead:
                it = self._search(ep, line, st, text, "q:value")
            elif rec is not None and rec.has_cont and room >= 2:
                it = self._page(ep, line, rec, False)
            if it is not None:
                break
        return it

    def _learn_unsearchable(self, ep: Any) -> None:
        """A zero-hit token seen only in an overview or codes of a listing that holds the other tokens."""
        for rec in list(ep.queries.recs.values()):
            if not rec.query_text or rec.canon_args in self.zero_done or not self._zero(rec):
                continue
            self.zero_done.add(rec.canon_args)
            toks = set(text_tokens(rec.query_text))
            for r in ep.catalog.values():
                idx = self._indexed(r)
                found = toks & (set(text_tokens(r.overview_head + " " + " ".join(
                    str(x) for v in r.codes.values() for x in v))) - idx)
                new = found - ep.contract.unsearchable
                if new and toks - found <= idx:
                    ep.contract.unsearchable |= new
                    ep.trace.decision("contract", unsearchable=sorted(new), why="zero hits; only in an overview")

    def _sig(self, ep: Any, line: Line) -> str:
        return dump_json([[c.attr, c.op, c.vals, c.bound, c.active] for c in line.constraints]
                         + [line.shelf_label, line.shelf_conf, line.qty, ep.ledger.cap_c, ep.ledger.in_stock])

    def _sync(self, ep: Any, line: Line, st: _CovLine) -> None:
        sig = self._sig(ep, line)
        if st.sig is not None and sig != st.sig:
            self.invalidate(ep, line, "requirements changed")
        st.sig = sig

    def next_intent(self, ep: Any) -> Optional[Intent]:
        return self._safe(ep, "next_intent", None, self._next)

    def _next(self, ep: Any) -> Optional[Intent]:
        plan = ep.plan
        if not ep.ledger.lines or not plan.nums or ep.clock() > T_EXPLORE_STOP:
            return None
        first = ep.budget.first_finish_step is not None
        if not first and plan.u() >= plan.nums.get("u_c_hard", 10 ** 6):
            return None
        soft = not first and plan.u() >= plan.nums.get("u_c_soft", 10 ** 6)
        self._learn_unsearchable(ep)
        for line in sorted(ep.ledger.lines, key=lambda l: (not l.cheapest, l.idx)):
            st = self.st(line)
            self._sync(ep, line, st)
            room = self._room(ep, line, st)
            cheap = line.cheapest or self._dx_tight(ep, line, st)
            it = (self._plan_cheapest if cheap else self._plan_other)(ep, line, st, room, soft) \
                if room > 0 else None
            st.idle = it is None
            if it is not None:
                return it
            if cheap:
                self._certificate(ep, line)
        return None

    def after(self, ep: Any, intent: Intent, dec: Decoded) -> None:
        self.sent.add(intent.signature())
        if intent.line is not None and 0 <= intent.line < len(ep.ledger.lines):
            self._safe(ep, "after", None, self._after, intent, dec)

    def _after(self, ep: Any, intent: Intent, dec: Decoded) -> None:
        line = ep.ledger.lines[intent.line]
        st, p = self.st(line), intent.payload
        st.spent += dec.kind != D_BLOCKED
        key = ep.executor.last_qkey if dec.kind == D_RECORDS else None
        if intent.kind == IK_SEARCH and p.get("cont") is None:
            if key:
                self.qkeys[p.get("query", "")] = key
            else:
                st.dead.add(p.get("query", ""))
        elif intent.kind == IK_FILTER:
            if p.get("cont") is None:
                n = dec.meta.get("n") if key else None
                n = len(dec.records) if key and n is None and not dec.meta.get("has_cont") else n
                st.rungs.append([p.get("ceiling"), key, n])
                self._note(ep, line, st, "ladder", key=p.get("ceiling"), ceiling=p.get("ceiling"), rows=n)
            if key and not self._rows_hold(ep, line, dec, p.get("ceiling")):
                st.filter_bad = True
                self._note(ep, line, st, "filter", why="bracket rows off the label or over the ceiling")
            elif key:
                st.onshelf |= set(ep.queries.get(key).ids)
        elif intent.kind == IK_OPEN:
            ids, u = list(p.get("ids") or []), ep.transport.n
            got = [i for i in ids if i in ep.catalog and u in ep.catalog[i].opened_by.values()] \
                if dec.kind in (D_RECORDS, D_RECORD) else []
            if got or len(ids) == 1:                          # a rejected batch is retried smaller (limit halved)
                st.open_fail.update(i for i in ids if i not in got)
            line.opened.extend(i for i in got if i not in line.opened)
            if got:
                best_c = self._judge(ep, line, st, self._dx_lb(ep, line))
                ep.trace.decision("verdict", line=line.idx, opened=got, best=st.best, best_c=best_c,
                                  eligible=[i for i in got if st.judged.get(i, (False,))[0]])

    @staticmethod
    def _rows_hold(ep: Any, line: Line, dec: Decoded, ceiling: Any) -> bool:
        """Rows at or below the ceiling carrying the label among their raw values."""
        lab = canon_key(line.shelf_label or "")
        for rec in dec.records:
            price = ep.fields.extract(dec.tool, rec).get(FR_PRICE)
            if (ceiling is not None and isinstance(price, int) and price > ceiling) or (
                    lab and not any(isinstance(v, str) and canon_key(v) == lab for v in rec.values())):
                return False
        return True

    def certificate(self, ep: Any, line: Line) -> Optional[Cert]:
        return self._safe(ep, "certificate", line.cert, self._certificate, line) if line.cheapest else line.cert

    def _certificate(self, ep: Any, line: Line) -> Optional[Cert]:
        st = self.st(line)
        self._sync(ep, line, st)
        lb = self._dx_lb(ep, line)
        best_p = self._judge(ep, line, st, lb)
        lits = self._lits(ep, line, st, relax=False)
        kind, keys, bc, floor, missing = "D", [], None, False, []
        br = lit = parts = None
        if self._unverified(ep, line, st, best_p, lb, limit=1, failed=True):
            missing.append("rows below the best are unverified")
        else:
            br = self._bracket(ep, line, st, best_p, lb)
            lit = None if br else self._lit_cert(ep, line, lits, best_p)
            parts = None if br or lit else self._parts_cert(ep, st, lits)
        if br is not None:
            kind, keys, bc = "B", [br[1]], br[0]
            floor = not any(ep.catalog[i].price_c is not None and (best_p is None or ep.catalog[i].price_c < best_p)
                            for i in br[2].ids if i in ep.catalog)
        elif lit:
            kind, keys = lit, [rec.canon_args for _, _, rec in lits if rec is not None]
        elif parts:
            kind, keys = "C", parts
        elif not missing:
            missing.append("no complete literal query, bracket or partition")
        old = line.cert
        if old is not None and old.valid and (old.kind, old.queries, old.bracket_c, old.floor_proof) == (
                kind, keys, bc, floor):
            return old
        line.cert = Cert(kind=kind, line=line.idx, queries=keys, bracket_c=bc, floor_proof=floor)
        ep.trace.decision("cert", what="cert", line=line.idx, cert=kind, best=st.best, best_c=best_p, bracket_c=bc,
                          floor_proof=floor, queries=len(keys), missing=missing, spent=st.spent, gen=st.gen)
        return line.cert

    def q4_ok(self, ep: Any, line: Line) -> bool:
        return self._safe(ep, "q4_ok", True, self._q4, line)

    def q4_rows(self, ep: Any, line: Line) -> list:
        """Q4's cheaper unverified rows, cheapest first (one batch; read by recovery after a swap)."""
        r = ep.catalog.get(line.choice) if line.choice and line.cheapest else None
        return self._safe(ep, "q4_rows", [], self._unverified, line, self.st(line), r.price_c, self._dx_lb(ep, line),
                          frozenset(), max(1, ep.plan.batch_k())) if r is not None and r.price_c is not None else []

    def _q4(self, ep: Any, line: Line) -> bool:
        if line.cheapest and line.cert is None:
            self._certificate(ep, line)                       # every cheapest line logs a certificate
        r = ep.catalog.get(line.choice) if line.choice else None
        if not line.cheapest or r is None or r.price_c is None:
            return True
        st = self.st(line)
        self._sync(ep, line, st)
        cheaper = self._unverified(ep, line, st, r.price_c, self._dx_lb(ep, line), limit=1)
        if not cheaper:
            return True
        advisory = self._room(ep, line, st) < 1 or ep.clock() > T_EXPLORE_STOP or not ep.plan.can_spend(
            1, self._purpose(ep))
        self._note(ep, line, st, "q4", key=(line.choice, cheaper[0], advisory), choice=line.choice,
                   cheaper=cheaper[0], advisory=advisory)
        return advisory

    def invalidate(self, ep: Any, line: Line, why: str) -> None:
        st = self.st(line)
        if line.cert is not None and line.cert.valid:
            line.cert.valid, line.cert.invalidated_by = False, why
        st.gen += 1
        st.judged, st.xtier, st.rungs, st.parts, st.split, st.noted = {}, {}, [], {}, set(), set()
        st.twin_q, st.best, st.idle, st.sig = None, None, False, self._sig(ep, line)
        ep.trace.decision("cert", what="invalidate", line=line.idx, why=clip_text(why, 120), gen=st.gen)

    def cert_pending(self, ep: Any) -> bool:
        it = self.next_intent(ep)
        return it is not None and bool(it.payload.get("certificate"))

    def best_of(self, ep: Any, line: Line) -> Optional[str]:
        self._judge(ep, line, self.st(line), self._dx_lb(ep, line))
        return self.st(line).best

    def done(self, ep: Any) -> bool:
        return self.next_intent(ep) is None

TIERS = ("T1", "T2P", "T2", "T3", "X")
CHK_STATES = ("CONFIRMED", "WEAK", "NEAR", "UNCONFIRMED", "CONTRADICTED")
CHK_LEVELS = ("EXACT", "LIST", "PHRASE", "NUM", "OWN", "OTHERKEY", "TITLE", "NOTE", "PARTIAL", "NEAR", "UNIT",
              "POINTER", "PENDING", "MISSING")          # best first; [:5] confirm, [5:8] weak, [8:10] near
CHK_JUDGE_MAX_CALLS = 4         # JUDGE calls per episode
CHK_JUDGE_BATCH = 16            # pairs (and numbers) per call
CHK_COMMON = 2                  # a vocabulary value seen on >= 2 records is common (rule c, PHRASE penalty)
CHK_DEMOTE = 1.0                # rank_penalty at or above this ranks an item below undemoted ones
_CHK_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])")
_CHK_LIST_RE = re.compile(r"[,/;|]")
_CHK_SENT_RE = re.compile(r"(?<=[.!?;])\s+")
_CHK_CAP_RE = re.compile(r"[A-Z][a-z]{2,}")
_CHK_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_CHK_QTY_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)(?:\s+(\d+)/(\d+)|/(\d+))?"
                         r"(?:\s*(?:-|–|~|to)\s*(\d+(?:\.\d+)?))?\s*([a-z]+|\"|')?")

def tier_rank(tier: str) -> int:
    return TIERS.index(tier) if tier in TIERS else len(TIERS)

def _chk_rank(level: Optional[str]) -> int:
    return CHK_LEVELS.index(level) if level in CHK_LEVELS else len(CHK_LEVELS)

def _chk_stem(t: str) -> str:
    """Light suffix stemming (s/es/ies/ing/ed, then a final e), applied alike to both sides of a match."""
    if len(t) <= 3 or not t.isalpha():
        return t
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    if len(t) > 6 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 4 and t.endswith("ed"):
        t = t[:-2]
    elif len(t) >= 4 and t.endswith("e"):
        t = t[:-1]
    return t

def _chk_tokens(text: Any) -> tuple:
    """norm(): lowercase, accents folded, punctuation to space, letter/digit runs split, light stemming,
    STOPWORDS dropped (kept right after a number, where they are units)."""
    s = _CHK_SPLIT_RE.sub(" ", unicodedata.normalize("NFKD", str(text or "")).lower())
    raw = _TOKEN_RE.findall(s)
    out = [_chk_stem(t) for i, t in enumerate(raw) if t not in STOPWORDS or (i and raw[i - 1].isdigit())]
    return tuple(out) if out else tuple(_chk_stem(t) for t in raw)

def _chk_tok_eq(a: str, b: str) -> bool:
    """Token equality for NEAR: equal, or two long alphabetic tokens sharing a long prefix (inflection)."""
    if a == b:
        return True
    if len(a) < 5 or len(b) < 5 or not (a.isalpha() and b.isalpha()):
        return False
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= max(5, int(0.75 * min(len(a), len(b))))

def _chk_ed1(a: str, b: str) -> bool:
    """Edit distance at most one (key spellings such as colour/color), for tokens of five letters or more."""
    if a == b:
        return True
    if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i + (len(a) == len(b)):] == b[i + 1:]

def _sx_same_words(a: frozenset, b: frozenset) -> bool:
    """Two key word sets name one attribute: equal up to inflection or derivation (a shared prefix of at least 5
    letters and three quarters of the longer word: water resistant / water resistance), word for word. No one-edit
    substitution and no prefix measured against the shorter word (weight/height, waist/waistline differ)."""
    def eq(x: str, y: str) -> bool:
        n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        return x == y or (x.isalpha() and y.isalpha() and min(len(x), len(y)) >= 5
                          and n >= max(5, int(0.75 * max(len(x), len(y)))))
    return len(a) == len(b) and all(any(eq(x, y) for y in b) for x in a) and all(any(eq(x, y) for x in a) for y in b)

def _chk_run(seq: tuple, sub: tuple) -> bool:
    n = len(sub)
    return n > 0 and any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))

def _v5s_mask(seq: tuple, lab: tuple) -> tuple:
    """seq with every run that spells the shelf name lab blanked."""
    n, out, i = len(lab), [], 0
    while n and i < len(seq):
        hit = seq[i:i + n] == lab
        out.append("" if hit else seq[i])
        i += n if hit else 1
    return tuple(out) if n else seq

def _s2_run(seq: tuple, sub: tuple) -> bool:
    """_chk_run up to inflection (rectangular / rectangle)."""
    n = len(sub)
    return n > 0 and any(all(map(_chk_tok_eq, seq[i:i + n], sub)) for i in range(len(seq) - n + 1))

def _sx_inform(t: str, aw: tuple) -> bool:
    """t says something about the value: it is not a word of the attribute's name (or a prefix form of one,
    neck/neckline), which values of that attribute share whatever they are (block heel / wedge heel)."""
    return not any(t == w or (min(len(t), len(w)) >= 4 and (t.startswith(w) or w.startswith(t))) for w in aw)

def _chk_share(vt: tuple, xt: tuple, aw: tuple = ()) -> bool:
    """x shares a word with v (numbers and one-letter tokens never count: the numeric rules judge them; nor do
    the attribute-name words aw)."""
    xs = [t for t in xt if len(t) > 1 and not t.isdigit() and _sx_inform(t, aw)]
    return any(_chk_tok_eq(a, b) for a in vt if len(a) > 1 and not a.isdigit() and _sx_inform(a, aw) for b in xs)

def _chk_valued(x: Any) -> bool:
    """A value that states something: some token of two characters or a number (not '', '-', 'n/a')."""
    return any(len(t) > 1 or t.isdigit() for t in _TOKEN_RE.findall(str(x or "").lower()))

def _chk_code(x: Any) -> str:
    return "".join(text_tokens(x))

def _chk_qty(text: Any) -> list:
    """Quantities written in a text: [(lo, hi, unit stem, unit as written)]; mixed fractions and ranges
    included; a STOPWORDS word after a number is a unit unless another number follows it ('10 and 20')."""
    out = []
    s = _CHK_COMMA_RE.sub("", str(text or "").lower())
    for m in _CHK_QTY_RE.finditer(s):
        a = float(m.group(1))
        if m.group(2):
            a += float(m.group(2)) / max(1.0, float(m.group(3)))
        elif m.group(4):
            a /= max(1.0, float(m.group(4)))
        b = float(m.group(5)) if m.group(5) else a
        unit = m.group(6)
        if unit in STOPWORDS and s[m.end():].lstrip()[:1].isdigit():
            unit = None
        out.append((min(a, b), max(a, b), _chk_stem(unit) if unit and unit.isalpha() else unit, unit))
    return out

@dataclass
class _ChkItem:
    """One record parsed for matching (cached per record state)."""
    sig: tuple
    title: tuple
    entries: list            # [(canon key, raw key, raw value, tokens, list-element tokens)]
    notes: list              # [(tokens, foreign, sentence)]
    known: bool              # attributes were read (an opened record): a missing key is then evidence

class Checker:
    def __init__(self):
        self.judged: dict = {}       # (v tokens, x tokens) -> same | different (validated JUDGE answers)
        self.factors: dict = {}      # (from unit, to unit) -> factor (validated JUDGE conversions)
        self.tries: dict = {}        # pair or unit-pair key -> JUDGE attempts
        self.judge_calls = 0
        self._tok: dict = {}
        self._qty: dict = {}
        self._items: dict = {}
        self._stats: Optional[dict] = None
        self._want: Optional[dict] = None       # JUDGE requests collected during a round
        self._pin: Optional[dict] = None        # statistics pinned for a round
        self._v5: Optional[tuple] = None
        self._logged: dict = {}

    def toks(self, text: Any) -> tuple:
        key = str(text or "")
        hit = self._tok.get(key)
        if hit is None:
            hit = self._tok[key] = _chk_tokens(key)
        return hit

    def qty(self, text: Any) -> list:
        key = str(text or "")
        hit = self._qty.get(key)
        if hit is None:
            hit = self._qty[key] = _chk_qty(key)
        return hit

    def elems(self, x: str) -> list:
        parts = _CHK_LIST_RE.split(x)
        return [self.toks(p) for p in parts] if len(parts) > 1 else []

    def stats(self, ep: Any) -> dict:
        if self._pin is not None:
            return self._pin                   # one selection round reads one catalog state
        sig = (len(ep.catalog), sum(len(b) for r in ep.catalog.values() for b in r.attrs.values()),
               sum(1 for r in ep.catalog.values() if r.merchant or r.maker),
               sum(1 for r in ep.catalog.values() if r.stock is not None))
        st = self._stats
        if st is None or st["sig"] != sig:
            recs = [r for r in ep.catalog.values() if r.attrs]
            keys = {k: frozenset(self.toks(k)) for r in recs for k in r.attrs}
            names = {self.toks(n) for r in ep.catalog.values() for n in (r.merchant, r.maker) if n}
            makers = {self.toks(r.maker) for r in ep.catalog.values() if r.maker}
            st = self._stats = dict(sig=sig, recs=recs, keys=keys, names=names - {()}, makers=makers - {()},
                                    memo={}, stock=any(r.stock is not None for r in ep.catalog.values()))
        return st

    def _vals_of(self, r: Record, keys: tuple) -> list:
        return [x for k in keys for _, x in r.attrs.get(k, []) if _chk_valued(x)]

    def _hits(self, st: dict, key: str, vts: list) -> int:
        """Records whose values under key hold a constraint value as a contiguous run."""
        return sum(1 for r in st["recs"] if key in r.attrs and any(
            _chk_run(self.toks(x), vt) for _, x in r.attrs[key] for vt in vts))

    def attr_keys(self, ep: Any, attr: str, vals: Any = ()) -> tuple:
        """(base keys, variant keys) for an attribute as the request words it. Base: the key with the same
        words (spelling within one edit), else the one related key (words contained either way) that holds a
        stated value, else the key whose values hold it. Variants: base plus affix words, kept when their
        values agree with the base's on co-occurring records (DESIGN 5.10 affixes, learned per episode)."""
        st = self.stats(ep)
        memo = ("keys", attr, tuple(str(v) for v in vals or ()))
        if memo in st["memo"]:
            return st["memo"][memo]
        keys = st["keys"]
        alphabet = sorted({t for ks in keys.values() for t in ks})
        known = set(alphabet)
        ta = frozenset(t if t in known else next((u for u in alphabet if _chk_ed1(t, u)), t)
                       for t in self.toks(attr))
        vts = [self.toks(v) for v in vals or () if self.toks(v)]
        base = sorted(k for k, t in keys.items() if ta and (t == ta or _sx_same_words(t, ta)))
        if not base and ta:
            near = sorted(k for k, t in keys.items() if t and (ta < t or t < ta))
            hit = [k for k in near if self._hits(st, k, vts)]
            base = self._head(keys, hit) or ([] if hit else self._head(keys, near))
        if not base and vts and not any(t and ta and (ta <= t or t <= ta) for t in keys.values()):
            scored = sorted(((self._hits(st, k, vts), k) for k in keys), key=lambda s: (-s[0], s[1]))
            if scored and scored[0][0] and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                base = [scored[0][1]]
        variants = []
        if base:
            bt = frozenset().union(*(keys[k] for k in base))
            for k, t in sorted(keys.items()):
                if k not in base and (bt < t or (ta and ta < t)) and self._agrees(st, k, base):
                    variants.append(k)
        out = st["memo"][memo] = (tuple(base), tuple(variants))
        return out

    @staticmethod
    def _head(keys: dict, ks: list) -> list:
        """The key of ks that every other key of ks extends (closure for shoes closure type), else none."""
        if not ks:
            return []
        m = min(ks, key=lambda k: (len(keys[k]), k))
        return [m] if all(keys[m] <= keys[k] for k in ks) else []

    def _agrees(self, st: dict, key: str, base: list) -> bool:
        both = agree = 0
        bv: set = set()
        kv: set = set()
        for r in st["recs"]:
            bt = {t for x in self._vals_of(r, tuple(base)) for t in self.toks(x)}
            kt = {t for x in self._vals_of(r, (key,)) for t in self.toks(x)}
            bv |= bt
            kv |= kt
            if bt and kt:
                both += 1
                agree += bool(bt & kt)
        if both >= 2:
            return agree >= 0.75 * both
        if both == 1:
            return agree == 1
        return bool(kv) and len(kv & bv) >= 0.5 * len(kv)

    def _vocab(self, st: dict, own: tuple) -> dict:
        """vocab(A): token tuples of A's values (and list elements) -> [records carrying them, a raw form]."""
        memo = ("vocab", own)
        if memo not in st["memo"]:
            counts: dict = {}
            for r in st["recs"]:
                seen: dict = {}
                for x in self._vals_of(r, own):
                    seen.setdefault(self.toks(x), x)
                    parts = _CHK_LIST_RE.split(x)
                    for part in parts if len(parts) > 1 else ():
                        seen.setdefault(self.toks(part), part.strip())
                for t, raw in seen.items():
                    if t:
                        counts.setdefault(t, [0, raw])[0] += 1
            st["memo"][memo] = counts
        return st["memo"][memo]

    def _mirrors(self, st: dict, own: tuple) -> tuple:
        """Options-like keys: other keys whose value restates an own value of A on >= half of >= 2 records."""
        memo = ("mirror", own)
        if memo not in st["memo"]:
            tally: dict = {}
            for r in st["recs"]:
                ov = [self.toks(x) for x in self._vals_of(r, own)]
                if not ov:
                    continue
                for k, entries in r.attrs.items():
                    if k not in own:
                        n, h = tally.get(k, (0, 0))
                        tally[k] = (n + 1, h + any(_chk_run(self.toks(x), o) for _, x in entries for o in ov))
            st["memo"][memo] = tuple(sorted(k for k, (n, h) in tally.items() if n >= 2 and 2 * h >= n))
        return st["memo"][memo]

    def item(self, ep: Any, st: dict, r: Record) -> _ChkItem:
        sig = (r.title, r.notes, sum(len(b) for b in r.attrs.values()), len(st["names"]), tuple(r.opened_by))
        it = self._items.get(r.id)
        if it is not None and it.sig == sig:
            return it
        entries = [(k, rk, x, self.toks(x), self.elems(x)) for k, bucket in r.attrs.items() for rk, x in bucket
                   if _chk_valued(x)]
        tools = ep.binder.binding.tools
        known = bool(r.attrs) or any(t in tools and FR_ATTRS in tools[t].provides_seen for t in r.opened_by)
        own_names = {self.toks(n) for n in (r.merchant, r.maker) if n}
        others = st["names"] - own_names
        local = set(self.toks(r.title)) | {t for e in entries for t in e[3]} | {t for n in own_names for t in n}
        notes = []
        for s in _CHK_SENT_RE.split(str(r.notes or "")):
            stoks = self.toks(s)
            caps = [m.group(0) for m in _CHK_CAP_RE.finditer(s) if m.start() > 0]
            foreign = any(_chk_run(stoks, n) for n in others) or ("(" in s and "," in s) or any(
                not set(self.toks(cap)) <= local for cap in caps)
            if stoks:
                notes.append((stoks, foreign, s.strip()))
        it = self._items[r.id] = _ChkItem(sig, self.toks(r.title), entries, notes, known)
        return it

    @staticmethod
    def _level(vt: tuple, xt: tuple, elems: list, aw: tuple = ()) -> Optional[str]:
        if not vt or not xt:
            return None
        if vt == xt:
            return "EXACT"
        if vt in elems:
            return "LIST"
        if _chk_run(xt, vt):
            return "PHRASE"
        if set(vt) <= set(xt):
            return "PARTIAL"
        return "NEAR" if _chk_share(vt, xt, aw) else None

    def _convert(self, unit: Optional[str], target: Optional[str]) -> tuple:
        """(factor, same unit) from a listing unit to a target unit; factor None while a conversion is pending."""
        if unit == target or unit is None or target is None:
            return 1.0, True
        return self.factors.get((unit, target)), False

    def _numrel(self, qv: list, qx: list, bare: bool = False) -> tuple:
        """A one-quantity value against a listing value: (eq | near | ne | unit | None, pending unit pair); bare:
        listing numbers with a unit are skipped (v5: 'hdmi 2.0' is no '2 hdmi')."""
        if len(qv) != 1 or not qx:
            return None, None
        lo, hi, u = qv[0][:3]
        pend, near, n = None, False, 0
        for a, b, ux, _ in qx:
            if bare and ux:
                continue
            n += 1
            f, same = self._convert(ux, u)
            if f is None:
                pend = (ux, u)
                continue
            tol = (1e-9 if same else 0.005) * max(1.0, abs(hi))
            a, b = a * f, b * f
            if abs(a - lo) <= tol and abs(b - hi) <= tol:
                return "eq", None
            near = near or (a - tol <= lo and hi <= b + tol)
        if pend:
            return "unit", pend
        return ("near" if near else "ne") if n else None, None

    def _pair(self, v: str, x: str, elems: Any = (), aw: tuple = ()) -> tuple:
        """(level or None, kind) for requested value v against listing value x; kind is match | num |
        judged | open | unit (a unit conversion is pending, requested from JUDGE). A word of the attribute's
        name (aw) shared by v and x makes no NEAR pair: JUDGE decides such a pair (rule a)."""
        vt, xt = self.toks(v), self.toks(x)
        lvl = self._level(vt, xt, list(elems), aw)
        qv = self.qty(v)
        words = {t for t in vt if t.isalpha() and len(t) > 1} - set(aw) - {qv[0][2] if qv else None}
        nr, pend = self._numrel(qv, self.qty(x), bool(qv) and qv[0][2] is None and bool(words))
        if nr == "ne":
            return None, "num"
        if nr == "near" and lvl not in CHK_LEVELS[:3]:
            return "PARTIAL", "match"
        if nr == "eq":                    # v5: v's words must be in x too ('316l stainless steel' vs '316l')
            return (lvl if lvl in CHK_LEVELS[:3] else "PARTIAL" if words - set(xt) else "NUM"), "match"
        if lvl:
            return lvl, "match"
        if nr == "unit":
            self._ask_num(x, qv[0], pend)
            return None, "unit"
        rel = self.judged.get((vt, xt))
        if rel == "same":
            return "NEAR", "match"
        return None, ("judged" if rel else "open")

    def _matches_any(self, vals: list, x: str, elems: Any = (), aw: tuple = ()) -> bool:
        return any(self._pair(a, x, elems, aw)[1] == "match" for a in vals)

    def _ask_pair(self, v: str, x: str, attr: str, prio: int) -> None:
        want = self._want
        key = (self.toks(v), self.toks(x))
        if want is None or key in self.judged or self.tries.get(key, 0) >= 2:
            return
        old = want["p"].get(key)
        if old is None or prio < old[3]:
            want["p"][key] = (v, x, attr, prio)

    def _ask_num(self, raw: str, q: tuple, pend: Optional[tuple]) -> None:
        want = self._want
        if want is None or not pend or pend in self.factors or self.tries.get(pend, 0) >= 2 or pend in want["n"]:
            return
        n = next((a for a, _, ux, _ in self.qty(raw) if ux == pend[0] and a > 0), None)
        if n is not None and q[3]:
            want["n"][pend] = (raw, n, q[3])

    def _judge_open(self, ep: Any) -> bool:
        return (ep.phase != "FALLBACK" and self.judge_calls < CHK_JUDGE_MAX_CALLS
                and ep.clock() < T_OPTIONAL_LLM_STOP and ep.llm.allowed("JUDGE"))

    def _judge(self, ep: Any, want: dict) -> bool:
        """One JUDGE call for the round's pending pairs and numbers; code validates every answer: indexes must
        be ones sent, a converted number must come back in the target unit and give the same factor for every
        number of one unit pair, and "same" never overrides a numeric mismatch."""
        pairs = sorted(want["p"].items(), key=lambda kv: kv[1][3])[:CHK_JUDGE_BATCH]
        nums = sorted(want["n"].items(), key=lambda kv: str(kv[0]))[:CHK_JUDGE_BATCH]
        if not pairs and not nums:
            return False
        payload = {"pairs": [{"i": i, "a": p[0], "b": clip_text(p[1], 120), "attr": p[2]}
                             for i, (_, p) in enumerate(pairs)],
                   "nums": [{"i": i, "raw": clip_text(n[0], 120), "n": n[1], "unit": n[2]}
                            for i, (_, n) in enumerate(nums)]}
        for key, _ in pairs + nums:
            self.tries[key] = self.tries.get(key, 0) + 1
        self.judge_calls += 1

        def check(ans: dict) -> tuple:
            rels, facs = {}, {}
            for item in ans.get("pairs") or []:
                i = item.get("i")
                if isinstance(i, int) and 0 <= i < len(pairs) and item.get("rel") in ("same", "different"):
                    key, (v, x, _, _) = pairs[i]
                    numeric = self._numrel(self.qty(v), self.qty(x))[0] in ("ne", "near")
                    rels[key] = "different" if numeric else item["rel"]
            groups: dict = {}
            for item in ans.get("nums") or []:
                i, val, unit = item.get("i"), as_number(item.get("val")), item.get("unit")
                if not (isinstance(i, int) and 0 <= i < len(nums)) or val is None or not math.isfinite(val) \
                        or val <= 0:
                    continue
                key, (_, n, _) = nums[i]
                if unit and (self.toks(unit) or ("",))[-1] != key[1]:
                    continue                                   # converted into some other unit
                groups.setdefault(key, []).append(val / n)
            for key, fs in groups.items():
                if max(fs) <= min(fs) * 1.02:
                    facs[key] = sum(fs) / len(fs)
            if not rels and not facs:
                return False, None, "no valid pair or number"
            return True, {"rels": rels, "facs": facs}, ""

        ans = ep.llm.ask("JUDGE", payload, validator=check)
        got = ans if isinstance(ans, dict) else {}
        self.judged.update(got.get("rels") or {})
        for key, f in (got.get("facs") or {}).items():
            self.factors.setdefault(key, f)
        ep.trace.decision("verdict", what="judge", pairs=len(pairs), nums=len(nums),
                          answered=len(got.get("rels") or {}) + len(got.get("facs") or {}))
        return bool(got)

    def verdict(self, ep: Any, line: Line, ident: str, line_budget: Optional[int] = None) -> Verdict:
        _v5s_scrub(ep)                     # v5: also before REVIEW's re-check
        r = ep.catalog.get(ident)
        v = Verdict()
        if r is None:
            v.tier = "X"
            v.reasons.append("unknown id")
            return v
        st = self.stats(ep)
        pc = v.per_constraint
        if ep.ledger.in_stock and r.stock is False:
            pc["chk:stock"] = ("CONTRADICTED", "STOCK", "")
            v.reasons.append("out of stock")
        elif ep.ledger.in_stock and r.stock is None and st["stock"]:
            pc["chk:stock"] = ("UNCONFIRMED", "STOCK", "")            # ranks lower; the tier is unaffected
            v.rank_penalty += 0.25
        if line.shelf_label and r.shelf and canon_key(r.shelf) != canon_key(line.shelf_label):
            pc["chk:shelf"] = ("CONTRADICTED" if line.shelf_conf >= 2 else "WEAK", "SHELF", clip_text(r.shelf, 60))
            v.reasons.append("off shelf")
        if line_budget is not None and r.price_c is not None and line.qty * r.price_c > line_budget:
            pc["chk:budget"] = ("CONTRADICTED", "BUDGET", str(r.price_c))
            v.reasons.append("over the line budget")
        v.compat_ok = self._compat_for(ep, line, ident)
        if v.compat_ok is False:
            pc["chk:compat"] = ("CONTRADICTED", "COMPAT", "")
            v.reasons.append("codes miss the base item")
        want = self._want
        if any(s[0] == "CONTRADICTED" for s in pc.values()):
            self._want = None                  # an unavailable item never spends a JUDGE pair
        owns = {}
        noun = {t for x in (line.words, line.shelf_words, line.shelf_label, r.shelf) for t in self.toks(x)}
        self._v5 = self.toks(r.shelf or line.shelf_label)
        try:
            it = self.item(ep, st, r)
            for c in line.constraints:
                if not c.active:
                    continue
                state, level, quote, pen, own = self._eval(ep, st, r, it, c, noun)
                pc[c.cid] = (state, level, clip_text(quote, 80))
                owns[c.cid] = own
                v.rank_penalty += pen
                if state != "CONFIRMED" and len(v.reasons) < 6:
                    v.reasons.append(f"{c.cid} {state} {level}")
        finally:
            self._want = want
        v.tier = self._tier(pc, owns)
        v.rank_penalty = round(v.rank_penalty, 3)
        return v

    @staticmethod
    def _tier(pc: dict, owns: dict) -> str:
        if any(s[0] == "CONTRADICTED" for s in pc.values()):
            return "X"
        attr = [(cid, s, lvl) for cid, (s, lvl, _) in pc.items() if cid in owns]
        if any(s in ("NEAR", "UNCONFIRMED") for _, s, _ in attr):
            return "T3"
        weak = [(cid, lvl) for cid, s, lvl in attr if s == "WEAK"]
        shelf = pc.get("chk:shelf", ("",))[0] == "WEAK"
        if not weak and not shelf:
            return "T1"
        if not shelf and all(lvl in CHK_LEVELS[5:8] and owns[cid] is False for cid, lvl in weak):
            return "T2P"
        return "T2"

    def _eval(self, ep: Any, st: dict, r: Record, it: _ChkItem, c: Constraint, noun: set) -> tuple:
        """(state, level, quote, rank penalty, has own key: True | False | None unknown) for one constraint."""
        if c.op == "pair_pending":
            return "UNCONFIRMED", "PENDING", "", 0.0, None     # never forced to the unpaired value
        base, var = self.attr_keys(ep, c.attr, c.vals)
        own_keys = base + var
        ta = set(self.toks(c.attr))
        for p in (r.note_read or {}).get("pointers") or []:
            pa = set(self.toks(p.get("attr") if isinstance(p, dict) else ""))
            if pa and ((ta and (pa <= ta or ta <= pa)) or any(st["keys"].get(k) == pa for k in own_keys)):
                return "UNCONFIRMED", "POINTER", str(p.get("span") or ""), 0.5, None   # own value void (R11)
        own = [e for e in it.entries if e[0] in own_keys]
        has_own = True if own else (False if it.known else None)
        vals = [str(x) for x in c.vals if self.toks(x)]
        if c.op in ("ge", "le", "between"):
            return self._eval_bound(r, it, c, own) + (has_own,)
        if not vals:
            return "UNCONFIRMED", "MISSING", "", 0.0, has_own
        if not own_keys and c.op in ("eq", "in"):
            names = {self.toks(n): n for n in (r.maker, r.merchant) if n}
            for x in vals:
                if self.toks(x) in names:
                    return "CONFIRMED", "EXACT", names[self.toks(x)], 0.0, True
            if r.maker and any(self.toks(x) in st["makers"] for x in vals):
                return "CONTRADICTED", "OWN", r.maker, 0.0, True
        if c.op == "ne":
            return self._eval_ne(r, it, vals, [e for e in own if e[0] in base] or own) + (has_own,)
        res = [self._eval_value(st, r, it, c, x, vals, own, own_keys, noun) for x in vals]
        order = (lambda t: (t[0] == "CONTRADICTED", CHK_STATES.index(t[0]), _chk_rank(t[1])))
        pick = min(res, key=order) if c.op == "in" else max(res, key=order)
        return pick + (has_own,)

    def _eval_value(self, st: dict, r: Record, it: _ChkItem, c: Constraint, v: str, vals: list, own: list,
                    own_keys: tuple, noun: set) -> tuple:
        """One requested value: rules a, b and c of the contradiction scan, then the best evidence level. The
        item's own values decide: other keys and the title are consulted only when it has none; a seller-note
        restatement of v still lifts an own NEAR value to WEAK NOTE (r5: 12 passes on such a note)."""
        vt, aw = self.toks(v), self.toks(c.attr)
        best, bx, bel, pend, pen = None, None, (), False, 0.0
        miss = []
        for _, rk, x, xt, el in own:
            lvl, kind = self._pair(v, x, el, aw)
            if lvl and _chk_rank(lvl) < _chk_rank(best):
                best, bx, bel = lvl, (rk, x, xt), el
            elif lvl is None and kind == "unit":
                pend = True
            elif lvl is None and not self._matches_any(vals, x, el, aw):
                miss.append((rk, x, xt, kind))
        if best is None and miss and not pend:                                    # rule a
            for rk, x, _, kind in miss:
                if kind == "open":
                    self._ask_pair(v, x, c.attr, 0)
            level = "UNJUDGED" if any(m[3] == "open" for m in miss) else "OWN"
            return "CONTRADICTED", level, f"{miss[0][0]}={miss[0][1]}", 0.0
        if best in CHK_LEVELS[:4] and miss:                                       # rule b
            for rk, x, _, kind in miss:
                if kind == "open":
                    self._ask_pair(v, x, c.attr, 0)
            composite = _chk_run(it.title, vt) or any(
                _chk_run(e[3], vt) and _s2_run(tuple(t for t in e[3] if t not in vt), m[2]) for e in own for m in miss)
            if not composite:
                return "CONTRADICTED", "DUPKEY", f"{miss[0][0]}={miss[0][1]}", 0.0
            pen += 0.5                                                            # composite: tolerated
        vocab = self._vocab(st, own_keys) if own_keys else {}
        if vocab:                                                                 # rule c
            mirrors = self._mirrors(st, own_keys)
            mv = [e for e in it.entries if e[0] in mirrors and not (vt in e[4] or _chk_run(e[3], vt))]   # lists
            found = None
            for wt, (n, raw) in sorted(vocab.items()):
                if n < CHK_COMMON or any(t.isdigit() for t in wt) or not any(len(t) > 2 for t in wt) \
                        or set(wt) & noun:                        # a product noun of the line says nothing of A
                    continue
                if any(self._level(self.toks(a), wt, []) or self.judged.get((self.toks(a), wt)) == "same"
                       for a in vals) or any(_chk_run(e[3], wt) for e in own):
                    continue                                      # related to v, or composite (own has w)
                src = ("TITLE", r.title) if _chk_run(it.title, wt) else next(
                    (("MIRROR", f"{e[1]}={e[2]}") for e in mv if _chk_run(e[3], wt)), None)
                if src:
                    self._ask_pair(v, raw, c.attr, 0)
                    found = found or src
            if found:
                return "CONTRADICTED", found[0], found[1], 0.0
        if best in CHK_LEVELS[:4]:
            cont = next((e for e in bel if _chk_run(e, vt)), bx[2])
            if best == "PHRASE" and cont != vt and vocab.get(cont, (0,))[0] >= CHK_COMMON and vt in vocab:
                pen += CHK_DEMOTE                     # inside a distinct common value of A that stands beside v
            return "CONFIRMED", best, f"{bx[0]}={bx[1]}", pen
        note = self._weak(r, it, c, vt, own_keys, kinds=("NOTE",)) if best == "NEAR" else None
        if best in ("PARTIAL", "NEAR") and note is None:
            rel = self.judged.get((vt, bx[2]))
            if best == "NEAR" and rel is None:
                self._ask_pair(v, bx[1], c.attr, 1)
            pen += 0.0 if rel == "same" else 0.5 if rel == "different" else 0.25
            return "NEAR", best, f"{bx[0]}={bx[1]}", pen
        weak = note or (None if own else self._weak(r, it, c, vt, own_keys))
        if weak:
            return "WEAK", weak[0], weak[1], pen
        return "UNCONFIRMED", ("UNIT" if pend else "MISSING"), "", pen

    def _weak(self, r: Record, it: _ChkItem, c: Constraint, vt: tuple, own_keys: tuple,
              kinds: tuple = ("OTHERKEY", "TITLE", "NOTE")) -> Optional[tuple]:
        """OTHERKEY / TITLE / NOTE: v as a contiguous run in another key's value, the title, or a sentence of the
        item's own note (sentences naming another seller or a descriptor list are excluded; M4 restatements).
        v5: a value inside the listing's shelf name never matches there ('This hoodies & sweatshirts from ...')."""
        lab = self._v5 if self._v5 and len(self._v5) > len(vt) and _chk_run(self._v5, vt) else ()
        for k, rk, x, xt, el in it.entries if "OTHERKEY" in kinds else ():
            if k not in own_keys and (vt in el or _chk_run(_v5s_mask(xt, lab), vt)):
                return "OTHERKEY", f"{rk}={x}"
        if "TITLE" in kinds and _chk_run(_v5s_mask(it.title, lab), vt):
            return "TITLE", r.title
        ta = set(self.toks(c.attr))
        for p in (r.note_read or {}).get("restates") or []:
            if isinstance(p, dict) and p.get("span") and _chk_run(self.toks(p.get("val")), vt) and \
                    set(self.toks(p.get("attr"))) & ta and norm_space(p["span"]) in norm_space(r.notes):
                return "NOTE", str(p.get("span"))
        for stoks, foreign, s in it.notes:
            if not foreign and _chk_run(_v5s_mask(stoks, lab), vt):
                return "NOTE", s
        return None

    def _eval_ne(self, r: Record, it: _ChkItem, vals: list, pool: list) -> tuple:
        """ne: CONTRADICTED when v shows at EXACT, LIST or PHRASE in the attribute or in the title; CONFIRMED
        only when the attribute is present with another value; otherwise UNCONFIRMED."""
        for v in vals:
            for _, rk, x, _, el in pool:
                if self._pair(v, x, el)[0] in CHK_LEVELS[:4]:
                    return "CONTRADICTED", "OWN", f"{rk}={x}", 0.0
            if _chk_run(it.title, self.toks(v)):
                return "CONTRADICTED", "TITLE", r.title, 0.0
        if pool:
            return "CONFIRMED", "OWN", f"{pool[0][1]}={pool[0][2]}", 0.0
        return "UNCONFIRMED", "MISSING", "", 0.0

    def _limits(self, c: Constraint) -> Optional[tuple]:
        """(lo, hi, unit stem, unit as written) of a ge/le/between constraint."""
        b = c.bound or {}
        num = as_number(b.get("num"))
        qs = self.qty(b.get("raw") or c.span) + [q for x in c.vals for q in self.qty(x)]
        unit = b.get("unit") or next((q[3] for q in qs if q[3] and (num is None or abs(q[0] - num) < 1e-9)), None)
        stem = (self.toks(unit) or (None,))[-1] if unit else None
        if c.op == "between":
            nums = sorted({q[0] for q in qs} | {q[1] for q in qs} | ({num} if num is not None else set()))
            return (nums[0], nums[-1], stem, unit) if len(nums) >= 2 else None
        if num is None:
            return None
        return (num, math.inf, stem, unit) if c.op == "ge" else (-math.inf, num, stem, unit)

    def _bound_status(self, text: str, lim: tuple, title: bool = False) -> Optional[str]:
        """sat | viol | part (a range must satisfy the bound as a whole) | unit (conversion pending) | None."""
        lo, hi, u, _ = lim
        out, pend = [], None
        for a, b, ux, _ in self.qty(text):
            if title and ux is None:
                continue                       # a bare number in a title is no evidence
            f, same = self._convert(ux, u)
            if f is None:
                pend = (ux, u)
                continue
            tol = (1e-9 if same else 0.005) * max(1.0, abs(lo if math.isfinite(lo) else hi))
            a, b = a * f, b * f
            out.append("sat" if a >= lo - tol and b <= hi + tol else "viol" if b < lo - tol or a > hi + tol
                       else "part")
        if out:
            return out[0] if len(set(out)) == 1 else "part"
        if pend and not title:
            self._ask_num(text, lim, pend)
            return "unit"
        return None

    def _eval_bound(self, r: Record, it: _ChkItem, c: Constraint, own: list) -> tuple:
        """ge / le / between: numbers and unit tokens from the own values, else from the title; same unit in
        code, other units through validated JUDGE factors; a range must satisfy the bound as a whole."""
        lim = self._limits(c)
        if lim is None:
            return "UNCONFIRMED", "MISSING", "", 0.0
        res = [(self._bound_status(x, lim), rk, x) for _, rk, x, _, _ in own]
        comp = [t for t in res if t[0] in ("sat", "viol", "part")]
        if comp:
            kinds = {t[0] for t in comp}
            q = f"{comp[0][1]}={comp[0][2]}"
            if kinds == {"viol"}:
                return "CONTRADICTED", "OWN", q, 0.0
            if kinds == {"sat"}:
                return "CONFIRMED", "NUM", q, 0.0
            return "NEAR", "NEAR", q, 0.25
        if any(t[0] == "unit" for t in res):
            return "UNCONFIRMED", "UNIT", "", 0.0
        if own:
            return "UNCONFIRMED", "MISSING", "", 0.0
        s = self._bound_status(r.title, lim, title=True)
        if s == "sat":
            return "WEAK", "TITLE", r.title, 0.0
        if s == "viol":
            return "CONTRADICTED", "TITLE", r.title, 0.0
        if s == "part":
            return "NEAR", "NEAR", r.title, 0.25
        return "UNCONFIRMED", "MISSING", "", 0.0

    def compatible(self, ep: Any, acc_id: str, base_id: str) -> Optional[bool]:
        a, b = ep.catalog.get(acc_id), ep.catalog.get(base_id)
        if a is None or b is None:
            return None
        fits = {_chk_code(x) for k in ("fits", "more_fits") for x in a.codes.get(k) or []} - {""}
        owns = {_chk_code(x) for k in ("own", "alias") for x in b.codes.get(k) or []} - {""}
        return bool(fits & owns) if fits and owns else None

    def _compat_for(self, ep: Any, line: Line, ident: str) -> Optional[bool]:
        if line.role != "accessory":
            return None
        lines = ep.ledger.lines
        bl = ep.ledger.compat.get("base_line")
        base = lines[bl] if isinstance(bl, int) and 0 <= bl < len(lines) and bl != line.idx else next(
            (o for o in lines if o.role == "base" and o.idx != line.idx), None)
        return self.compatible(ep, ident, base.choice) if base is not None and base.choice else None

    def verdicts(self, ep: Any, line: Line, ids: list, line_budget: Optional[int] = None) -> dict:
        self._want = {"p": {}, "n": {}} if self._judge_open(ep) else None
        self._pin = None
        self._pin = self.stats(ep)
        try:
            out = {i: self.verdict(ep, line, i, line_budget) for i in ids}
            want, self._want = self._want, None
            decisive = want and (want["n"] or any(p[3] == 0 for p in want["p"].values()))
            if decisive and self._judge(ep, want):           # NEAR pairs (ranking only) ride along
                out = {i: self.verdict(ep, line, i, line_budget) for i in ids}
        finally:
            self._want = self._pin = None
        line.verdicts.update(out)
        return out

    @staticmethod
    def is_eligible(line: Line, verdict: Verdict) -> bool:
        return verdict.tier in (TIERS[:2] if line.cheapest else TIERS[:4])

    def deficit(self, line: Line, verdict: Verdict) -> float:
        """Weighted count of non-confirmed attribute verdicts: weak 1, near 2, unconfirmed 3, doubled unless the
        constraint is a disclosed need (never trade a requirement of the request for a disclosed value)."""
        src = {c.cid: c.source for c in line.constraints}
        states = {cid: (s[0] if isinstance(s, (tuple, list)) and s else s) for cid, s in verdict.per_constraint.items()}
        return sum(CHK_STATES.index(s) * (1.0 if src.get(cid) == "disclosed" else 2.0)
                   for cid, s in states.items() if cid in src and s in CHK_STATES[1:4])

    def eligible(self, ep: Any, line: Line, ids: list, line_budget: Optional[int] = None,
                 subset: bool = False) -> list:
        ids = list(dict.fromkeys(ids))
        vs = self.verdicts(ep, line, ids, line_budget)
        pos = {i: n for n, i in enumerate(ids)}

        def price(i: str) -> int:
            r = ep.catalog.get(i)
            return r.price_c if r is not None and r.price_c is not None else 10 ** 12

        def unopened(i: str) -> bool:
            return not ep.catalog[i].opened_by           # verified listings first among equals

        live = [i for i in ids if vs[i].tier != "X"]
        top = [i for i in live if self.is_eligible(line, vs[i])]
        held = subset and any(self.is_eligible(line, v) for i, v in line.verdicts.items() if i not in vs)
        if line.cheapest and (top or held):  # a subset call (coverage judges new rows) falls back to T2/T3 only
            out = sorted(top, key=lambda i: (vs[i].rank_penalty >= CHK_DEMOTE, price(i), tier_rank(vs[i].tier),
                                             unopened(i), vs[i].rank_penalty, pos[i]))
        else:
            out = sorted(live, key=lambda i: (tier_rank(vs[i].tier), vs[i].rank_penalty >= CHK_DEMOTE,
                                              self.deficit(line, vs[i]), unopened(i), price(i),
                                              vs[i].rank_penalty, pos[i]))
        head = (out[0], vs[out[0]].tier) if out else (None, None)
        if self._logged.get(line.idx) != head:
            self._logged[line.idx] = head
            ep.trace.decision("verdict", line=line.idx, candidates=len(ids), eligible=len(out), top=[
                dict(rid=i, tier=vs[i].tier, pen=vs[i].rank_penalty,
                     why={cid: f"{s[0][:4]}:{s[1]}" for cid, s in vs[i].per_constraint.items()}) for i in out[:3]])
        return out

_SEL_CODE_RE = re.compile(r"(?<![\w-])(?:[A-Z][A-Za-z]{2,} \d{2,6}|(?=[A-Z0-9-]*[A-Z])(?=(?:[A-Z-]*\d){2})"
                          r"[A-Z0-9]+(?:-[A-Z0-9]+)*)(?![\w-])")
_SEL_CLAUSE_RE = re.compile(r"(?<=[.;!?])\s+")
_SEL_ITEM_RE = re.compile(r"\s*(?:[,;&]|\band\b|\bor\b)\s*", re.I)
_SEL_KINDS = ("own", "alias", "fits", "more_on", "also_fits")
_SEL_BASE_K = 40           # bases compared per bundle selection (code matching costs no step)
_SEL_LOOKUPS = 4           # more-on part searches per episode
_SEL_LEAD_WORDS = 12       # a longer clause ends the leading code block
_SEL_VERIFY = 6            # verification opens (batches) per episode
_BX_PAGES = 8              # basket pages per line while no complete basket fits the cap
_SEL_BIG = 10 ** 12
_SX_OPENS = 4              # finalist verification opens (batches, failed ones included) per episode

def _sel_norm(code: Any) -> str:
    """Case-, space- and hyphen-insensitive code form."""
    return re.sub(r"[^a-z0-9]", "", str(code or "").lower())

def _sel_shape(code: Any) -> str:
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", re.sub(r"[^A-Za-z0-9]", "", str(code or ""))))

def _sel_clauses(text: Any) -> list:
    """Sentences of an overview head; a short code-free clause ("Model no.") joins the code that follows."""
    out: list = []
    for part in _SEL_CLAUSE_RE.split(str(text or "")):
        if out and len(out[-1].split()) <= 3 and not any(ch.isdigit() for ch in out[-1]) \
                and re.match(r"\S*\d", part):
            out[-1] += " " + part
        elif part:
            out.append(part)
    return out

def _sel_codes_in(text: str) -> list:
    """Codes in a span after an anchor: shape codes, else listed items with a digit and at most 3 words."""
    shaped = [m.group(0) for m in _SEL_CODE_RE.finditer(text)]
    if shaped:
        return shaped
    items = [i.strip(" .:;,-()[]\"'") for i in _SEL_ITEM_RE.split(text)]
    return [i for i in items if i and any(ch.isdigit() for ch in i) and len(i.split()) <= 3]

def _sel_anchor(phrase: Any) -> Optional[tuple]:
    """(clean phrase, regex) for an oracle anchor: its longest code-free run of words, 1-6 tokens."""
    runs = [r.strip(" .:;,-") for r in _SEL_CODE_RE.split(str(phrase or ""))]
    run = max(runs, key=len) if runs else ""
    toks = text_tokens(run)
    if not toks or len(toks) > 6 or all(t in STOPWORDS for t in toks):
        return None
    return run, re.compile(r"(?<![a-z0-9])" + r"[^a-z0-9]+".join(map(re.escape, toks)) + r"(?![a-z0-9])", re.I)

class Selector(Engine):
    name = "selector"

    def __init__(self, ep: Any = None):
        super().__init__(ep)
        self.anchors: dict = {}            # kind -> [(phrase, regex)] learned by COMPAT
        self.anchor_v = 0
        self.compat_done = False
        self.cache: dict = {}              # id -> (signature, parsed codes)
        self._tindex: tuple = (-1, {})     # (catalog size, {norm title code: [ids]})
        self._memo: dict = {}              # memos keyed by catalog/table sizes
        self.base: Optional[str] = None    # the base of the last bundle selection
        self.parts: dict = {}              # norm part code -> searched (once)
        self.opened: set = set()           # ids this engine opened
        self.open_lines: dict = {}         # id -> line idx (records the open on the line)
        self.pages: dict = {}              # line idx -> pages and brackets requested
        self.brackets: dict = {}           # line idx -> ceiling of its shelf bracket; ("q", idx) -> its query key
        self.verify_n = 0
        self.sx_tried: list = []          # finalist batches opened (single-line selection policy)
        self.rematch_failed: Optional[tuple] = None
        self._partial: Optional[str] = None  # the base most accessory lines fit when no basket is complete

    def _parse(self, r: Record) -> dict:
        sig = (r.title, r.overview_head, self.anchor_v)
        hit = self.cache.get(r.id)
        if hit is not None and hit[0] == sig:
            return hit[1]
        out: dict = {k: [] for k in _SEL_KINDS}
        out.update(lead=[], also=[], tcodes=[m.group(0) for m in _SEL_CODE_RE.finditer(r.title or "")])
        shapes = {_sel_shape(c) for c in out["tcodes"]}
        block = True
        for cl in _sel_clauses(r.overview_head):
            spans = list(_SEL_CODE_RE.finditer(cl))
            codes = [m.group(0) for m in spans]
            gap = next((k for k in range(1, len(spans)) if any(
                t not in STOPWORDS for t in text_tokens(cl[spans[k - 1].end():spans[k].start()]))), None)
            if gap is not None and _sel_shape(codes[gap - 1]) in shapes and "also_fits" not in self.anchors:
                out["also"].append((codes[gap - 1], codes[gap:]))      # "Part P also fits: C" (by structure)
                continue
            hits = self._anchor_hits(cl)
            for k, (s, e, kind) in enumerate(hits):
                found = _sel_codes_in(cl[e:hits[k + 1][0] if k + 1 < len(hits) else len(cl)])
                if kind == "also_fits":
                    subj = _sel_codes_in(cl[hits[k - 1][1] if k else 0:s])
                    if subj and found:
                        out["also"].append((subj[-1], found))
                else:
                    out[kind] += found
            if hits:
                continue
            if not spans or len(re.split(r"[\s;]+", cl)) > _SEL_LEAD_WORDS:
                block = False
            elif block:
                out["lead"].append(codes)
        if len(self.cache) > 5000:
            self.cache.clear()
        self.cache[r.id] = (sig, out)
        return out

    def _anchor_hits(self, clause: str) -> list:
        found = sorted((m.start(), m.start() - m.end(), m.end(), kind) for kind, rows in self.anchors.items()
                       for _, rx in rows for m in rx.finditer(clause))
        out: list = []
        for s, _, e, kind in found:
            if not out or s >= out[-1][1]:
                out.append((s, e, kind))
        return out

    def _title_index(self, ep: Any) -> dict:
        if self._tindex[0] != len(ep.catalog):
            idx: dict = {}
            for rid, r in ep.catalog.items():
                for m in _SEL_CODE_RE.finditer(r.title or ""):
                    idx.setdefault(_sel_norm(m.group(0)), []).append(rid)
            self._tindex = (len(ep.catalog), idx)
        return self._tindex[1]

    def base_codes(self, ep: Any, ident: Optional[str]) -> list:
        r = ep.catalog.get(ident) if ident else None
        if r is None:
            return []
        p = self._parse(r)
        return p["own"] + p["alias"] + [c for cl in p["lead"] for c in cl]

    def fit_codes(self, ep: Any, ident: Optional[str]) -> list:
        r = ep.catalog.get(ident) if ident else None
        if r is None:
            return []
        p = self._parse(r)
        return p["fits"] + [c for cl in p["lead"] for c in cl]

    def _pointers(self, r: Record) -> list:
        p = self._parse(r)
        return p["more_on"] + [cl[0] for cl in p["lead"][1:] if len(cl) == 1]

    def more_fits(self, ep: Any, ident: str) -> list:
        """Codes a 'more on part P' pointer adds: the row titled with P states 'Part <own part> also fits: C'."""
        r = ep.catalog.get(ident)
        if r is None:
            return []
        own = {_sel_norm(c) for c in self._parse(r)["tcodes"]}
        idx = self._title_index(ep)
        out: list = []
        for code in self._pointers(r):
            for host in idx.get(_sel_norm(code), []):
                if host != ident and host in ep.catalog:
                    for subj, targets in self._parse(ep.catalog[host])["also"]:
                        if not own or _sel_norm(subj) in own:
                            out += [t for t in targets if t not in out]
        return out

    def pending_parts(self, ep: Any, ident: str) -> list:
        """Pointer codes of ident whose page has not been seen and was never searched."""
        r = ep.catalog.get(ident)
        if r is None:
            return []
        p = self._parse(r)
        shapes = {_sel_shape(c) for c in p["tcodes"]}
        idx = self._title_index(ep)
        out = []
        for code in p["more_on"] + [cl[0] for cl in p["lead"][1:] if len(cl) == 1 and _sel_shape(cl[0]) in shapes]:
            n = _sel_norm(code)
            if n not in self.parts and not any(h != ident for h in idx.get(n, [])) and code not in out:
                out.append(code)
        return out

    def codes(self, ep: Any, ident: str) -> dict:
        r = ep.catalog.get(ident)
        if r is None:
            return {}
        p = self._parse(r)
        lead = [c for cl in p["lead"] for c in cl]
        r.codes = {"own": p["own"] or lead, "alias": p["alias"], "fits": p["fits"] or lead,
                   "more_on": p["more_on"] + [c for c in self._pointers(r) if c not in p["more_on"]],
                   "more_fits": self.more_fits(ep, ident)}
        return r.codes

    def compatible(self, ep: Any, acc: str, base: str) -> Optional[bool]:
        bc = {_sel_norm(c) for c in self.base_codes(ep, base)}
        if not bc:
            return None
        if {_sel_norm(c) for c in self.fit_codes(ep, acc) + self.more_fits(ep, acc)} & bc:
            return True
        return None if self.pending_parts(ep, acc) else False

    def shares_code(self, ep: Any, a: str, b: str) -> bool:
        return bool({_sel_norm(c) for c in self.base_codes(ep, a)} & {_sel_norm(c) for c in self.base_codes(ep, b)})

    def _seen(self, ep: Any, line: Line) -> list:
        return [i for i in list(line.opened) + [i for i in line.table if i not in line.opened] if i in ep.catalog]

    def _key(self, ep: Any, *extra: Any) -> tuple:
        if len(self._memo) > 4000:
            self._memo.clear()
        return extra + (len(ep.catalog), self.anchor_v, tuple(len(l.table) + len(l.opened) for l in ep.ledger.lines))

    def enforced(self, ep: Any, line: Line) -> bool:
        """Codes decide only when this accessory line's rows carry fit codes and the base line's rows carry codes."""
        b_idx, _ = self.roles(ep)
        if b_idx is None:
            return False
        key = self._key(ep, "enf", line.idx)
        if key not in self._memo:
            self._memo[key] = any(self.fit_codes(ep, i) for i in self._seen(ep, line)) and any(
                self.base_codes(ep, i) for i in self._seen(ep, ep.ledger.lines[b_idx]))
        return self._memo[key]

    def compat_ok(self, ep: Any, line: Line, ident: str, base: Optional[str]) -> bool:
        return not base or not self.enforced(ep, line) or self.compatible(ep, ident, base) is True

    def compat_verdict(self, ep: Any, line: Line, ident: str) -> Optional[bool]:
        b_idx, accs = self.roles(ep)
        if line.idx not in accs or not ep.ledger.lines[b_idx].choice or not self.enforced(ep, line):
            return None
        return self.compatible(ep, ident, ep.ledger.lines[b_idx].choice) is True

    def roles(self, ep: Any) -> tuple:
        """(base line, accessory lines) of a bundle the reader declared (a base/accessory role or compat.required);
        anchors may re-type the base by overview structure; with compat.required every other line joins."""
        lines = ep.ledger.lines
        comp = ep.ledger.compat or {}
        if len(lines) < 2 or not (comp.get("required") or any(l.role in ("base", "accessory") for l in lines)):
            return None, []
        bl = comp.get("base_line")
        base = bl if isinstance(bl, int) and not isinstance(bl, bool) and 0 <= bl < len(lines) else \
            next((l.idx for l in lines if l.role == "base"), None)
        accs = [l.idx for l in lines if l.role == "accessory"]
        if base is None and accs and len(lines) - len(accs) == 1:
            base = next(l.idx for l in lines if l.role != "accessory")
        typed = self._typed(ep) if self.anchors else {}
        if typed.get("base") is not None and (base is None or (not typed["own"].get(base)
                                                               and len(self._seen(ep, lines[base])) >= 2)):
            base = typed["base"]
        if base is None:
            return None, []
        return base, [l.idx for l in lines if l.idx != base and (l.role == "accessory" or comp.get("required")
                                                                 or typed.get("fits", {}).get(l.idx))]

    def _typed(self, ep: Any) -> dict:
        key = self._key(ep, "roles")
        if key not in self._memo:
            own, fits = {}, {}
            for line in ep.ledger.lines:
                parsed = [self._parse(ep.catalog[i]) for i in self._seen(ep, line)]
                own[line.idx] = sum(1 for p in parsed if p["own"] or p["alias"])
                fits[line.idx] = sum(1 for p in parsed if p["fits"])
            best = max(own, key=lambda i: (own[i], -i), default=None)
            base = best if best is not None and own[best] >= 2 and own[best] > fits.get(best, 0) else None
            self._memo[key] = {"base": base, "own": own, "fits": {i: n for i, n in fits.items() if n > own.get(i, 0)}}
        return self._memo[key]

    def ensure_anchors(self, ep: Any) -> None:
        """One COMPAT call per task, for a declared bundle whose rows on 2+ lines carry code shapes."""
        lines = ep.ledger.lines
        comp = ep.ledger.compat or {}
        if self.compat_done or len(lines) < 2 or not (
                comp.get("required") or any(l.role in ("base", "accessory") for l in lines)):
            return
        coded = {l.idx: [ep.catalog[i] for i in self._seen(ep, l) if any(
            ch.isdigit() for ch in (ep.catalog[i].overview_head or "")[:80])] for l in lines}
        if sum(1 for v in coded.values() if v) < 2:
            return
        b_idx = self.roles(ep)[0]
        order = sorted(coded, key=lambda i: (i != b_idx, i))
        samples: list = []
        for k in range(8):
            samples += [coded[i][k] for i in order if k < len(coded[i]) and coded[i][k] not in samples]
        samples = samples[:8]
        self.compat_done = True
        if not ep.llm.allowed("COMPAT"):
            ep.trace.decision("compat", source="shape", why="COMPAT not allowed now")
            return
        heads = [{"rid": r.id, "head": clip_text(r.title, 160), "overview": r.overview_head} for r in samples]
        ans = ep.llm.ask("COMPAT", {"heads": heads}, validator=lambda v: self._compat_check(v, samples))
        if ans and ans.get("anchors"):
            self.anchors = {k: [a for a in (_sel_anchor(p) for p in ps) if a] for k, ps in ans["anchors"].items()}
            self.anchor_v += 1
        ep.trace.decision("compat", source="compat" if self.anchors else "shape",
                          anchors={k: [p for p, _ in v] for k, v in self.anchors.items()},
                          examples=(ans or {}).get("examples_ok"), dropped=(ans or {}).get("dropped"))

    @staticmethod
    def _compat_check(value: Any, samples: list) -> tuple:
        """Anchors must occur in a sample overview (codes cut away); example codes must appear verbatim."""
        if not isinstance(value, dict):
            return False, None, "no object"
        texts = [r.overview_head or "" for r in samples]
        clean: dict = {}
        dropped: list = []
        for kind in _SEL_KINDS:
            for phrase in ((value.get("anchors") or {}).get(kind) or [])[:6]:
                got = _sel_anchor(phrase)
                if got and any(got[1].search(t) for t in texts) and got[0] not in clean.get(kind, []):
                    clean.setdefault(kind, []).append(got[0])
                else:
                    dropped.append(phrase)
        hay = {r.id: _sel_norm((r.title or "") + " " + (r.overview_head or "")) for r in samples}
        ok_n = bad_n = 0
        for ex in value.get("examples") or []:
            for kind in ("own", "alias", "fits", "more_on"):
                for code in ex.get(kind) or []:
                    good = bool(_sel_norm(code)) and _sel_norm(code) in hay.get(ex.get("rid"), "")
                    ok_n, bad_n = ok_n + good, bad_n + (not good)
        return True, {"anchors": clean, "examples_ok": [ok_n, bad_n], "dropped": dropped[:12]}, \
            f"{len(dropped)} anchors dropped" if dropped else ""

    @staticmethod
    def _price(ep: Any, ident: Optional[str]) -> int:
        r = ep.catalog.get(ident) if ident else None
        return r.price_c if r is not None and r.price_c is not None else 0

    def chosen_price(self, ep: Any, line: Line) -> int:
        if line.choice and line.choice in ep.catalog and ep.catalog[line.choice].price_c is not None:
            return ep.catalog[line.choice].price_c
        return self.cheapest_seen(ep, line)

    def cheapest_seen(self, ep: Any, line: Line) -> int:
        """The line's cheapest orderable price seen (in stock, not X), else 0."""
        prices = [ep.catalog[i].price_c for i in self.candidates(ep, line) if ep.catalog[i].price_c is not None
                  and not (ep.catalog[i].stock is False and ep.ledger.in_stock)
                  and getattr(line.verdicts.get(i), "tier", None) != "X"]
        return min(prices) if prices else 0

    def line_budget(self, ep: Any, line: Line) -> Optional[int]:
        cap = ep.ledger.cap_c
        if cap is None:
            return None
        return cap - sum(o.qty * self.chosen_price(ep, o) for o in ep.ledger.lines if o.idx != line.idx)

    def candidates(self, ep: Any, line: Line) -> list:
        seen = list(line.opened) + [i for i in line.table if i not in line.opened]
        if line.cheapest:
            seen.sort(key=lambda i: (ep.catalog[i].price_c if i in ep.catalog and ep.catalog[i].price_c is not None
                                     else _SEL_BIG, i not in line.opened))
        return [i for i in seen if i in ep.catalog]

    def rank(self, ep: Any, line: Line, ids: list) -> list:
        """The selection policy: the most exact evidence first (tier, then the checker's weighted deficit: a near
        value before a missing one); on cheapest lines the price first among T1/T2P (DESIGN 5.14), demoted items
        last."""
        vals = {t for c in line.constraints if c.active and c.op in ("eq", "in") for v in c.vals
                for t in content_tokens(v)}
        pos = {i: k for k, i in enumerate(ids)}

        def key(i: str) -> tuple:
            r, v = ep.catalog[i], line.verdicts.get(i)
            states = [s if isinstance(s, (tuple, list)) else (s,) for s in (v.per_constraint.values() if v else ())]
            tier = tier_rank(v.tier) if v else 3
            weak = sum(1 for s in states if s and str(s[0]).upper() != "CONFIRMED")
            inexact = sum(1 for s in states if len(s) > 1 and str(s[0]).upper() == "CONFIRMED"
                          and str(s[1]).upper() != "EXACT")
            ev = (ep.checker.deficit(line, v) if v else 0.0, weak, v.rank_penalty if v else 0.0, inexact,
                  0 if r.opened_by else 1, -len(vals & set(text_tokens(r.title))))
            price = r.price_c if r.price_c is not None else _SEL_BIG
            if line.cheapest:
                demoted = bool(v) and v.rank_penalty >= CHK_DEMOTE
                return (0, demoted, price, tier) + ev + (pos[i],) if tier <= 1 else (1, tier) + ev + (price, pos[i])
            return (tier,) + ev + (price, pos[i])
        return sorted(ids, key=key)

    def orderable(self, ep: Any, line: Line, budget: Optional[int] = None, compat: bool = True) -> list:
        ids = self.candidates(ep, line)
        priced = [i for i in ids if ep.catalog[i].price_c is not None or ep.ledger.cap_c is None]
        void = ep.resolver.trap_check(ep, line, priced)          # R11 pointer traps (M4) are never orderable
        priced = [i for i in priced if not void.get(i)]
        out = self.rank(ep, line, ep.checker.eligible(ep, line, priced, budget if budget is not None
                                                       else ep.ledger.cap_c))
        if not compat:
            return out
        b_idx, accs = self.roles(ep)
        base = ep.ledger.lines[b_idx].choice if b_idx is not None else None
        if line.idx in accs and base and self.enforced(ep, line):
            keep = []
            for i in out:
                ok = self.compatible(ep, i, base) is True
                v = line.verdicts.get(i)
                if v is not None:
                    v.compat_ok = ok
                    if not ok:
                        v.tier = "X"
                        v.reasons.append("codes miss the base")
                if ok:
                    keep.append(i)
            return keep
        if line.idx == b_idx and line.choice and accs:
            out.sort(key=lambda i: not (i != line.choice and self.shares_code(ep, i, line.choice)))
        return out

    def _sx_finalists(self, ep: Any) -> Optional[Intent]:
        """Open and verify finalists: a line whose best orderable item is no undemoted T1 (on a cheapest line: no
        undemoted T1/T2P; a clean pick there opens only cheaper rows carrying every value) opens the unopened rows
        that carry its requested values, most constraints covered first, then price. Plain baskets only (a single
        line first; a multi-line basket once it is complete inside the cap, or after B's row fetch), before
        u_c_soft, at most _SX_OPENS batches (a refused batch is retried at the size the contract learned)."""
        if len(self.sx_tried) >= _SX_OPENS or not ep.plan.can_spend(1, P_COVER) \
                or ep.plan.u() >= ep.plan.nums.get("u_c_soft", 10 ** 6):
            return None
        for line in ep.ledger.lines:
            try:
                ids = self._sx_rows(ep, line)[:max(1, ep.plan.batch_k())]
            except Exception as exc:                  # a finalist step never ends the selection drive
                ep.trace.decision("crash", where="selector finalists", err=repr(exc)[:200])
                self.sx_tried.extend([[]] * _SX_OPENS)
                return None
            if ids:
                self.sx_tried.append(ids)
                self.open_lines.update({i: line.idx for i in ids})
                return Intent(IK_OPEN, {"ids": ids, "need": NEED_ATTRS}, P_COVER, line.idx,
                              why=f"select: verify finalists for line {line.idx}")
        return None

    def _sx_rows(self, ep: Any, line: Line) -> list:
        """Unopened rows that can beat the line's best evidence. A row's evidence is the set of constraints its row
        verdict confirms or weakly supports (title, maker) or a query of the line names a value of (attribute
        values are indexed, keys are not); rows the checker already vetoes are never opened."""
        cons = [c for c in line.constraints if c.active and c.op in ("eq", "in", "ge", "le", "between")]
        if not cons:
            return []
        pool = self.orderable(ep, line)
        best = line.verdicts.get(pool[0]) if pool else None
        clean = best is not None and best.rank_penalty < CHK_DEMOTE and (
            best.tier == "T1" or (line.cheapest and best.tier == "T2P"))
        if clean and not line.cheapest:
            return []
        below = self._price(ep, pool[0]) if clean else None
        twice = {i for b in self.sx_tried for i in b if sum(i in o for o in self.sx_tried) >= 2}
        qcov: dict = {}
        words = {c.cid: [({t for t in content_tokens(v) if len(t) > 1} - set(content_tokens(c.attr)))
                         or set(content_tokens(v)) for v in c.vals] for c in cons}   # searchable value words
        for rec in ep.queries.for_line(line.idx):
            qt = set(rec.tokens)
            got = {cid for cid, ws in words.items() if any(w and w <= qt for w in ws)}
            for i in rec.ids if got else ():
                qcov.setdefault(i, set()).update(got)
        cap, cids, out = ep.ledger.cap_c, {c.cid for c in cons}, []
        lb = None if cap is None else cap - sum(o.qty * self.cheapest_seen(ep, o) for o in ep.ledger.lines
                                                if o.idx != line.idx)  # a basket of the others' cheapest can fit
        for i in self.candidates(ep, line):
            r = ep.catalog[i]
            if r.opened_by or i in ep.absent_ids or i in twice or r.price_c is None \
                    or (below is not None and r.price_c >= below):
                continue
            v = ep.checker.verdict(ep, line, i, lb)
            ev = qcov.get(i, set()) | {cid for cid, s in v.per_constraint.items()
                                       if cid in cids and s[0] in ("CONFIRMED", "WEAK")}
            if v.tier != "X" and len(ev) >= (len(cids) if clean else 1):
                out.append((-len(ev), r.price_c, i))
        return [i for _, _, i in sorted(out)]

    def basket_total(self, ep: Any) -> Optional[int]:
        return self._total(ep, {l.idx: l.choice for l in ep.ledger.lines})

    def _total(self, ep: Any, picks: dict) -> Optional[int]:
        total = 0
        for line in ep.ledger.lines:
            r = ep.catalog.get(picks.get(line.idx)) if picks.get(line.idx) else None
            if r is None or r.price_c is None:
                return None
            total += line.qty * r.price_c
        return total

    @staticmethod
    def _tier(line: Line, ident: str) -> int:
        v = line.verdicts.get(ident)
        return tier_rank(v.tier) if v else 3

    def _ladder(self, ep: Any, picks: dict, pools: dict) -> tuple:
        """Over the cap: swap the line with the best saving per tier loss to its next cheaper candidate, and
        repeat. Returns (picks, swaps)."""
        cap, lines, picks, swaps = ep.ledger.cap_c, ep.ledger.lines, dict(picks), []
        for _ in range(sum(len(p) for p in pools.values()) + 1):
            total = self._total(ep, picks)
            if cap is None or total is None or total <= cap:
                break
            best = None
            for idx, cur in sorted(picks.items()):
                cp = self._price(ep, cur)
                nxt = next((i for i in pools.get(idx) or [] if self._price(ep, i) < cp), None)
                if nxt is None:
                    continue
                loss = max(0, self._tier(lines[idx], nxt) - self._tier(lines[idx], cur))
                score = lines[idx].qty * (cp - self._price(ep, nxt)) / (1.0 + loss)
                if best is None or score > best[0]:
                    best = (score, idx, nxt)
            if best is None:
                break
            picks[best[1]] = best[2]
            swaps.append(list(best[1:]))
        return picks, swaps

    def select(self, ep: Any) -> dict:
        check_cap_unit(ep)                         # a cap read 100 times too small (F13)
        self.ensure_anchors(ep)
        lines = ep.ledger.lines
        b_idx, accs = self.roles(ep)
        out, partial = None, None
        if b_idx is not None and accs:
            cur = lines[b_idx].choice
            if cur and ep.cart.expected and ep.ledger.objective != "cheapest" \
                    and cur in self.orderable(ep, lines[b_idx], compat=False):
                out = self.select_bundle(ep, base=cur, need_fit=True)    # keep the written base when it still fits
            if out is None:
                out = self.select_bundle(ep)
            partial = self._partial
        if out is None:
            pools, swaps = {}, []
            for line in sorted(lines, key=lambda l: l.idx != b_idx):     # the base first: accessories filter on it
                pools[line.idx] = self.orderable(ep, line)
                if line.idx == b_idx and partial in pools[line.idx]:
                    pools[line.idx].remove(partial)
                    pools[line.idx].insert(0, partial)                    # the base most accessories fit
                line.choice = pools[line.idx][0] if pools[line.idx] else None
            picks = {l.idx: l.choice for l in lines}
            if all(picks.values()):
                picks, swaps = self._ladder(ep, picks, pools)
            for line in lines:
                line.choice = picks.get(line.idx)
                line.backups = [i for i in pools[line.idx] if i != line.choice][:3]
            out = dict(picks)
            ep.trace.decision("select", choices={str(k): v for k, v in out.items()}, total=self.basket_total(ep),
                              cap=ep.ledger.cap_c, swaps=swaps, tiers={str(l.idx): (
                                  l.verdicts.get(l.choice).tier if l.choice in l.verdicts else None) for l in lines})
        return out

    def _bx_pool(self, ep: Any, line: Line) -> list:
        """Every orderable candidate of a bundle line in policy order. orderable() cuts a cheapest line to its top
        tier; here the unverified rows below it stay (after the top tier): fit codes sit in the row overview and
        the attributes need an open (race 5), so the accessory that fits a base is often still unverified, and
        compatibility must decide before the tier cut."""
        top = self.orderable(ep, line, compat=False)
        if not line.cheapest:
            return top
        ids = [i for i in self.candidates(ep, line) if ep.catalog[i].price_c is not None or ep.ledger.cap_c is None]
        void = ep.resolver.trap_check(ep, line, ids)
        return top + self.rank(ep, line, [i for i in ids if i not in top and not void.get(i)
                                          and getattr(line.verdicts.get(i), "tier", "X") != "X"])

    def _bx_settled(self, ep: Any, line: Line, ident: str) -> bool:
        """Read and still T2/T3: another open cannot make it eligible."""
        return self._tier(line, ident) >= 2 and bool(ep.catalog[ident].opened_by)

    def _options(self, ep: Any, base: Optional[str] = None) -> tuple:
        """(complete baskets per base, best first; the lines the most promising partial base lacks, then the
        base line). Also sets self._partial, that base."""
        b_idx, accs = self.roles(ep)
        self._partial = None
        if b_idx is None or not accs:
            return [], []
        lines, cap = ep.ledger.lines, ep.ledger.cap_c
        raw = {l.idx: self._bx_pool(ep, l) for l in lines}
        cheapest = ep.ledger.objective == "cheapest" or any(l.cheapest for l in lines)
        index = {}                                     # accessory line -> {norm code: candidates that fit it}
        for a in accs:
            if self.enforced(ep, lines[a]):
                index[a] = {}
                for i in raw[a]:
                    for c in {_sel_norm(x) for x in self.fit_codes(ep, i) + self.more_fits(ep, i)}:
                        index[a].setdefault(c, set()).add(i)
        opts, partial = [], []
        for rank_b, b in enumerate([base] if base else raw[b_idx][:_SEL_BASE_K]):
            if b not in ep.catalog:
                continue
            pools = {b_idx: [b]}
            codes = {_sel_norm(c) for c in self.base_codes(ep, b)}
            for line in lines:
                if line.idx in index:
                    fit = set().union(*(index[line.idx].get(c, set()) for c in codes))
                    pools[line.idx] = [i for i in raw[line.idx] if i in fit]
                elif line.idx != b_idx:
                    pools[line.idx] = list(raw[line.idx])
            miss = [i for i, p in pools.items() if not p]
            if miss:
                partial.append((len(miss), rank_b, miss, b))
                continue
            twin = 0 if (self.base and b != self.base and self.shares_code(ep, b, self.base)) else 1
            first = {k: p[0] for k, p in pools.items()}      # the policy's picks, and the basket of picks not read
            ok = {k: [i for i in p if not self._bx_settled(ep, lines[k], i)] or p[:1] for k, p in pools.items()}
            fresh = {k: min(p, key=lambda i: self._price(ep, i)) if cheapest else p[0] for k, p in ok.items()}
            for start in [first] + ([fresh] if fresh != first else []):     # as weak (cheapest: the floor basket)
                picks, swaps = self._ladder(ep, start, pools)
                total = self._total(ep, picks)
                tr = {k: self._tier(lines[k], i) for k, i in picks.items()}
                tiers = sum(tr.values())
                fits = cap is None or (total is not None and total <= cap)
                tot = total if total is not None else _SEL_BIG
                weak = sum(1 for t in tr.values() if t >= 2)          # T2/T3: not eligible on a cheapest line
                settled = sum(1 for k, i in picks.items() if self._bx_settled(ep, lines[k], i))
                key = (settled, weak > 0, tiers if weak else tot, tot) if cheapest else (tiers, settled, tot)
                opts.append(dict(key=(not fits,) + key + (twin, rank_b), base=b, sel=picks, pools=pools, total=total,
                                 swaps=swaps, fits=fits, settled=settled, weak=weak,
                                 pending=[(k, i) for k, i in sorted(picks.items()) if not ep.catalog[i].opened_by]))
        opts.sort(key=lambda o: o["key"])
        if opts or not partial:
            return opts, ([] if opts else [b_idx])
        best = min(partial, key=lambda p: p[:2])
        self._partial = best[3]
        return opts, best[2]

    def select_bundle(self, ep: Any, base: Optional[str] = None, need_fit: bool = False) -> Optional[dict]:
        opts, short = self._options(ep, base)
        if not opts or (need_fit and not opts[0]["fits"]):
            ep.trace.decision("select", bundle=False, base=base, short=short)
            return None
        best = opts[0]
        b_idx = self.roles(ep)[0]
        for idx, ident in best["sel"].items():
            line = ep.ledger.lines[idx]
            line.choice = ident
            if idx == b_idx:                       # base backups: same-code twins first
                pool = sorted((i for i in self.orderable(ep, line, compat=False) if i != ident),
                              key=lambda i: not self.shares_code(ep, i, ident))
            else:
                pool = [i for i in best["pools"][idx] if i != ident]
            line.backups = pool[:3]
        self.base = best["base"]
        self.rematch_failed = None
        ep.trace.decision("select", bundle=True, base=best["base"], forced=base is not None,
                          choices={str(k): v for k, v in best["sel"].items()}, total=best["total"],
                          cap=ep.ledger.cap_c, within_cap=best["fits"], options=len(opts), swaps=best["swaps"],
                          codes={v: (lambda c: c["own"] + c["alias"] if k == b_idx else c["fits"] + c["more_fits"])(
                              self.codes(ep, v)) for k, v in best["sel"].items()})
        return dict(best["sel"])

    def ensure_bundle(self, ep: Any) -> bool:
        """Re-match the accessories after any base swap (constrained to the new base, else a full re-selection)."""
        b_idx, accs = self.roles(ep)
        lines = ep.ledger.lines
        if b_idx is None or not accs or not lines[b_idx].choice:
            return True
        b = lines[b_idx].choice
        bad = [a for a in accs if lines[a].choice and not self.compat_ok(ep, lines[a], lines[a].choice, b)]
        if not bad:
            self.base = b
            return True
        state = (b, tuple(l.choice for l in lines), len(ep.catalog))
        if state == self.rematch_failed:
            return False
        ep.trace.decision("select", why="re-match after a base swap", base=b, was=self.base, incompatible=bad)
        if self.select_bundle(ep, base=b) is None and self.select_bundle(ep) is None:
            self.rematch_failed = state
            return False
        return True

    def basket_status(self, ep: Any, picks: bool = False) -> dict:
        """For the prover (OVERCAP / NOFIT, 5.18) and the fallback (v3): is a complete basket formable from what was
        seen (a declared bundle: compatible baskets only), does the best one fit the cap, and which lines have no
        (compatible) orderable item; with picks=True also the best basket's picks (`sel`)."""
        b_idx, accs = self.roles(ep)
        if b_idx is not None and accs:
            opts, short = self._options(ep)
            best = opts[0] if opts else {}
            out = dict(complete=bool(opts), within_cap=bool(best.get("fits")), short=short, total=best.get("total"))
            sel = dict(best.get("sel") or {})
        else:
            pools = {l.idx: self.orderable(ep, l) for l in ep.ledger.lines}
            short = [i for i, p in pools.items() if not p]
            sel = {} if short or not pools else self._ladder(ep, {i: p[0] for i, p in pools.items()}, pools)[0]
            total = self._total(ep, sel) if sel else None
            cap = ep.ledger.cap_c
            out = dict(complete=bool(sel), within_cap=bool(sel) and (cap is None or (total is not None and total <= cap)),
                       short=short, total=total)
        if picks:
            out["sel"] = dict(sel)
        return out

    def bundle_ok(self, ep: Any) -> bool:
        b_idx, accs = self.roles(ep)
        lines = ep.ledger.lines
        base = lines[b_idx].choice if b_idx is not None else None
        return all(self.compat_ok(ep, lines[a], lines[a].choice, base) for a in accs if lines[a].choice)

    def next_intent(self, ep: Any) -> Optional[Intent]:
        lines = ep.ledger.lines
        b_idx, accs = self.roles(ep)
        if b_idx is None or not accs:                 # a plain basket: rows for a missing or over-cap line
            if len(lines) < 2:
                return self._sx_finalists(ep)         # finalists of a single line
            pools = {l.idx: self.orderable(ep, l) for l in lines}
            picks = {i: p[0] for i, p in pools.items() if p}
            if len(picks) == len(lines):
                picks = self._ladder(ep, picks, pools)[0]
                total = self._total(ep, picks)
                if ep.ledger.cap_c is None or total is None or total <= ep.ledger.cap_c:
                    return self._sx_finalists(ep)     # a complete basket inside the cap: finalists next
            order = sorted(lines, key=lambda l: (l.idx in picks, -l.qty * self._price(ep, picks.get(l.idx))))
            return next((it for it in (self._fetch(ep, l.idx) for l in order) if it is not None), None) \
                or self._sx_finalists(ep)
        self.ensure_anchors(ep)
        opts, short = self._options(ep)
        picks = opts[0]["sel"] if opts else {}
        if len(self.parts) < _SEL_LOOKUPS:            # a pointer can make a top candidate fit another base
            for a in accs:
                for ident in self._bx_pool(ep, ep.ledger.lines[a])[:4]:
                    if picks and self.compatible(ep, ident, picks[b_idx]):
                        continue
                    for code in self.pending_parts(ep, ident):
                        self.parts[_sel_norm(code)] = True
                        return Intent(IK_SEARCH, {"query": code, "qkind": "q:part"}, P_RESOLVE, None,
                                      why=f"bundle: page of part {code} (more-on pointer of {ident})")
        fits = bool(opts) and opts[0]["fits"]
        todo = self.bx_unread(ep, opts) if fits and self.verify_n < _SEL_VERIFY else []
        order = sorted(picks, key=lambda i: -lines[i].qty * self._price(ep, picks[i]))
        if not fits:                                 # no complete basket, or none inside the cap: more rows
            order = order if opts else short + [b_idx]
        elif todo or not opts[0]["settled"]:
            order = []
        else:                                        # complete, but a pick read as T2/T3: rows for its line first
            order = [i for i, j in picks.items() if self._bx_settled(ep, lines[i], j)] + order
        for idx in self._bx_turns(order):
            it = self._fetch(ep, idx)
            if it is not None:
                return it
        if not todo:
            return None
        self.verify_n += 1
        self.open_lines.update(dict((j, i) for i, j in todo))
        self.opened |= {j for _, j in todo}
        return Intent(IK_OPEN, {"ids": [j for _, j in todo], "need": NEED_ATTRS, "certificate": True}, P_COVER,
                      todo[0][0], why=f"bundle: verify the picks for base {opts[0]['base']}")

    def bx_weak(self, ep: Any) -> bool:
        """A declared bundle whose chosen basket holds a pick read as T2/T3: more rows may give a better basket."""
        return bool(self.roles(ep)[1]) and any(l.choice in ep.catalog and self._bx_settled(ep, l, l.choice)
                                               for l in ep.ledger.lines)

    def bx_unread(self, ep: Any, opts: Optional[list] = None, skip: Any = ()) -> list:
        """[(line, id)] worth one batch read before a basket is committed (at most batch_k): the unopened picks of
        the best basket, then of the next ones; when cheapest, of every basket below the best eligible total,
        cheapest first (verification in ascending price, ANALYSIS T6). A plain basket: its unopened choices."""
        lines = ep.ledger.lines
        if opts is None:
            opts = self._options(ep)[0] if self.roles(ep)[1] else []
        if not opts:
            opts = [dict(fits=True, settled=0, weak=1, total=0, pending=[(l.idx, l.choice) for l in lines if l.choice
                                                                and not ep.catalog[l.choice].opened_by])]
        cheap = ep.ledger.objective == "cheapest" or any(l.cheapest for l in lines)
        bar = min([o["total"] or 0 for o in opts if o["fits"] and not o.get("weak")] or [_SEL_BIG]) if cheap \
            else (_SEL_BIG if opts[0]["pending"] or opts[0]["settled"] else 0)   # else when the best one is unsure
        cands = [o for o in opts if o["fits"] and (o is opts[0] or (not o["settled"] and (o["total"] or 0) < bar))]
        todo = dict.fromkeys((i, j) for o in (sorted(cands, key=lambda o: o["total"] or 0) if cheap else cands)
                             for i, j in o["pending"] if j not in self.opened and j not in skip)
        return list(todo)[:max(1, ep.plan.batch_k())]

    def _bx_turns(self, order: list) -> list:
        """The lines to page in turn: fewest basket pages first, then the given order (no line takes them all)."""
        return sorted(dict.fromkeys(order), key=lambda i: (self.pages.get(i, 0), order.index(i)))

    def _fetch(self, ep: Any, idx: int) -> Optional[Intent]:
        """More rows for a line: one shelf bracket under the line's share of the cap (when the world filters by
        ceiling), then the next pages of its search (rows carry overviews), within C_l. A declared bundle (codes
        couple its lines; coverage never matches them) pages up to _BX_PAGES per line while no basket fits the cap
        or the best one holds a pick read as weak, with the commit-window exemption (certificate, to u_c_hard)."""
        line, cap = ep.ledger.lines[idx], ep.ledger.cap_c
        C, cert = ep.plan.nums.get("C") or [], bool(self.roles(ep)[1])
        if self.pages.get(idx, 0) >= (_BX_PAGES if cert else min(4, max(1, C[idx] if idx < len(C) else 1))):
            return None
        rest = sum(o.qty * self.cheapest_seen(ep, o) for o in ep.ledger.lines if o.idx != idx)
        ceiling = (cap - rest) // max(1, line.qty) if cap is not None else None    # DESIGN's line budget
        tool = ep.binder.tool_for(ROLE_FILTER) or ep.binder.tool_for(ROLE_SEARCH)
        if idx not in self.brackets and line.shelf_label and ceiling and ceiling > 0 and tool \
                and ep.binder.slot_arg(tool, SLOT_CEILING) and ep.binder.slot_arg(tool, SLOT_SHELF):
            self.brackets[idx] = ceiling
            self.pages[idx] = self.pages.get(idx, 0) + 1
            return Intent(IK_FILTER, {"shelf": line.shelf_label, "ceiling": ceiling, "qkind": "q:bracket",
                                      "certificate": cert}, P_COVER, idx,
                          why=f"basket: line {idx} rows under {ceiling} (no basket inside the cap yet)")
        own = ep.queries.get(self.brackets.get(("q", idx)) or "")
        recs = [q for q in ep.queries.for_line(idx) + ([own] if own else []) if q.has_cont and q.next_cont is not None
                and (q is own or (q.query_text and q.kind != "q:part"
                                  and getattr(ep.binder.binding.tools.get(q.tool), "role", None) == ROLE_SEARCH))]
        if not recs:
            return None
        q = max(recs, key=lambda x: (x is own, len(x.ids), x.canon_args))
        self.pages[idx] = self.pages.get(idx, 0) + 1
        if q is own:
            return Intent(IK_FILTER, {"shelf": line.shelf_label, "ceiling": self.brackets.get(idx), "cont": q.next_cont,
                                      "qkey": q.canon_args, "qkind": q.kind, "certificate": cert}, P_COVER, idx,
                          why=f"basket: next bracket page for line {idx}")
        return Intent(IK_SEARCH, {"query": q.query_text, "cont": q.next_cont, "qkey": q.canon_args, "qkind": q.kind,
                                  "certificate": cert}, P_COVER, idx,
                      why=f"basket: next page for line {idx} (no basket inside the cap yet)")

    def after(self, ep: Any, intent: Intent, dec: Decoded) -> None:
        if intent.kind == IK_FILTER and dec.kind == D_RECORDS and intent.line is not None \
                and "cont" not in intent.payload:
            self.brackets[("q", intent.line)] = ep.executor.last_qkey      # this bracket's query, for its next pages
        if intent.kind == IK_OPEN and dec.ok and dec.kind in (D_RECORDS, D_RECORD):
            for ident in intent.payload.get("ids") or []:
                idx = self.open_lines.get(ident)
                r = ep.catalog.get(ident)
                if r is not None and not r.opened_by:
                    self.opened.discard(ident)             # a single-id view opened only the first: retry later
                elif idx is not None and idx < len(ep.ledger.lines) and ident not in ep.ledger.lines[idx].opened:
                    ep.ledger.lines[idx].opened.append(ident)

    def done(self, ep: Any) -> bool:
        return False

class CartManager(Engine):
    name = "cartman"

    def __init__(self, ep: Any = None):
        super().__init__(ep)
        self.failures: dict = {}
        self.q4_deferred: set = set()      # line idx deferred by Q4 in this pass
        self.q4_advisory = False           # set after one coverage pass: Q4 no longer defers writes
        self.logged_order: list = []
        self.bx_read: set = set()          # basket picks this manager opened before writing them
        self.bx_rounds = 0
        self.bx_part = 0                   # v7: basket reads that opened only some of their ids
        self.bx_search: Optional[Intent] = None    # the selector search step this manager issued last

    def writable(self, item: Optional[str]) -> bool:
        return bool(item) and self.failures.get(item, 0) < 2

    def note_failure(self, item: Optional[str], blocked: bool = False) -> None:
        if item:
            self.failures[item] = self.failures.get(item, 0) + (2 if blocked else 1)

    def setup(self, ep: Any) -> None:
        cart = ep.cart
        tool = ep.binder.tool_for(ROLE_CART_WRITE)
        if tool is None:
            cart.semantics = "unknown"
            return
        binder = ep.binder
        mode_arg = binder.slot_arg(tool, SLOT_CART_MODE)
        qty_arg = binder.slot_arg(tool, SLOT_QTY)
        qty_feat = binder.features[tool].args.get(qty_arg) if qty_arg else None
        meanings = {m for m, _ in binder.binding.tools[tool].enum_map.get(mode_arg, {}).values()} if mode_arg else set()
        if binder.slot_arg(tool, SLOT_ITEMS):
            cart.semantics = "set"
            binder.binding.item_list_cart = True
        elif mode_arg and SEM_SET in meanings:
            cart.semantics = "set"
        elif mode_arg and (SEM_ADD in meanings or SEM_REMOVE in meanings):
            cart.semantics = "add"
        elif qty_feat is not None:
            cart.semantics = "set"
        else:
            cart.semantics = "add"
        cart.sem_source = "schema"
        if qty_feat is not None and qty_feat.maximum is not None:
            cart.add_max = int(qty_feat.maximum)
        ep.trace.decision("cart", semantics=cart.semantics, tool=tool, mode_arg=mode_arg, qty_arg=qty_arg,
                          add_max=cart.add_max, item_list=binder.binding.item_list_cart)

    def plan(self, ep: Any) -> dict:
        ep.selector.ensure_bundle(ep)
        out: dict = {}
        for line in ep.ledger.lines:
            if line.choice:
                out[line.choice] = out.get(line.choice, 0) + max(1, int(line.qty))
        return out

    def resilience(self, ep: Any, line: Line) -> int:
        """2: a verified twin backup at the same or lower price that stays compatible; 1: a verified compatible
        backup within the line budget; 0: none (DESIGN 5.16)."""
        r = ep.catalog.get(line.choice) if line.choice else None
        if r is None:
            return 0
        sel = ep.selector
        b_idx, accs = sel.roles(ep)
        lines = ep.ledger.lines
        base = lines[b_idx].choice if b_idx is not None else None
        budget = sel.line_budget(ep, line)
        best = 0
        for alt in line.backups:
            a = ep.catalog.get(alt)
            if a is None or alt == line.choice or not a.opened_by or (a.stock is False and ep.ledger.in_stock) \
                    or getattr(line.verdicts.get(alt), "tier", None) == "X":
                continue
            if line.idx == b_idx:
                twin = sel.shares_code(ep, alt, line.choice)
                if not twin and not all(sel.compat_ok(ep, lines[k], lines[k].choice, alt) for k in accs
                                        if lines[k].choice):
                    continue
            else:
                if line.idx in accs and not sel.compat_ok(ep, line, alt, base):
                    continue
                twin = token_overlap(a.title, r.title) >= 0.6 or bool(a.attrs and a.attrs == r.attrs)
            if twin and a.price_c is not None and r.price_c is not None and a.price_c <= r.price_c:
                return 2
            if budget is None or a.price_c is None or line.qty * a.price_c <= budget:
                best = 1
        return best

    def over_cap(self, ep: Any) -> bool:
        """A multi-line plan whose known total exceeds the cap (DESIGN 5.15: an over-cap order is a certain
        failure). Single lines never pick above the cap: the checker's budget check makes such items X."""
        total = ep.selector.basket_total(ep) if len(ep.ledger.lines) > 1 else None
        return ep.ledger.cap_c is not None and total is not None and total > ep.ledger.cap_c

    def _bx_search_ok(self, ep: Any) -> bool:
        """Only while the planner would spend the selector's search step (no step it refuses ends the drive)."""
        if len(ep.ledger.lines) < 2 or ep.phase != "COMMIT" or ep.plan.endgame() \
                or ep.budget.first_finish_step is not None:
            return False
        nums = ep.plan.nums
        end = nums.get("u_c_hard", 10 ** 6) if ep.selector.roles(ep)[1] else nums.get("u_c_soft", 10 ** 6)
        return ep.plan.u() < end and ep.plan.can_spend(1, P_COVER)

    def _bx_reads_ok(self, ep: Any) -> bool:
        """Multi-line baskets in the COMMIT phase, two read batches at most, never in the endgame."""
        n = len(ep.ledger.lines)                   # v7: two reads of every pick, batch_k picks a read
        return n > 1 and ep.phase == "COMMIT" and self.bx_rounds < 2 * -(-n // ep.plan.batch_k()) \
            and not ep.plan.endgame() and ep.budget.first_finish_step is None

    def write_order(self, ep: Any) -> list:
        lines = list(ep.ledger.lines)
        if len(lines) < 2:
            return lines
        b_idx = ep.selector.roles(ep)[0]
        res = {l.idx: self.resilience(ep, l) for l in lines}

        def key(line: Line) -> tuple:
            r = ep.catalog.get(line.choice) if line.choice else None
            recent = max(r.opened_by.values()) if r is not None and r.opened_by else -1
            return -res[line.idx], int(line.idx == b_idx and res[line.idx] < 2), -recent, line.idx
        order = sorted(lines, key=key)
        sig = [[l.idx, l.choice, res[l.idx]] for l in order]
        if sig != self.logged_order:
            self.logged_order = sig
            ep.trace.decision("cart", write_order=sig, why="resilience: the first-added item takes add-window events")
        return order

    def next_intent(self, ep: Any) -> Optional[Intent]:
        plan = self.plan(ep)
        expected = ep.cart.expected
        empty = [l.idx for l in ep.ledger.lines if not l.choice]
        over = not empty and self.over_cap(ep)
        weak = not (empty or over) and not ep.cart.writes and ep.selector.bx_weak(ep)
        if (empty or over or weak) and self._bx_search_ok(ep):
            it = ep.selector.next_intent(ep)       # an order cannot pass yet: search on for a complete basket
            if it is not None:                     # inside the cap while the planner allows (up to u_c_hard)
                self.bx_search = it
                return it
        if empty or over:                          # a partial or over-cap basket is a certain failure: the
            sig = ["partial", empty] if empty else ["over_cap", ep.selector.basket_total(ep)]   # finisher decides
            if self.logged_order != sig:
                self.logged_order = sig
                ep.trace.decision("cart", partial=empty, over_cap=sig[1] if over else None, cap=ep.ledger.cap_c,
                                  why="no complete basket inside the cap: nothing is written")
            return None
        blind = ep.selector.bx_unread(ep, skip=self.bx_read) if self._bx_reads_ok(ep) and any(
            expected.get(i, 0) != q for i, q in plan.items()) else []
        if blind:                                  # basket picks are read before they are written (once per id)
            return Intent(IK_OPEN, {"ids": [j for _, j in blind], "need": NEED_ATTRS, "basket_read": True}, P_VERIFY,
                          blind[0][0], why="read the unopened basket picks before writing them")
        notes = self._v7_notes_due(ep, plan)
        if notes:                                  # v7: a split world's picks have their notes read (traps)
            self.bx_read.update(notes)
            return Intent(IK_OPEN, {"ids": notes, "need": NEED_NOTES, "notes_read": True}, P_VERIFY, None,
                          why="read the notes of picks opened only for their attributes before writing them")
        for line in self.write_order(ep):
            item = line.choice
            if item and expected.get(item, 0) != plan[item] and self.failures.get(item, 0) < 2:
                obligation = ep.resolver.pairing_obligation(ep, line)
                if obligation is not None:
                    return obligation              # Q3 pairing obligation before the write (M4)
                if not (self.q4_advisory or ep.coverage.q4_ok(ep, line) or ep.plan.pressure()):
                    self.q4_deferred.add(line.idx)  # Q4: a cheaper unverified row first (M5)
                    continue
                return Intent(IK_CART_SET, {"item": item, "quantity": plan[item], "held": expected.get(item, 0)},
                              P_COMMIT, line.idx, why=f"write line {line.idx}: {item} x{plan[item]}")
        for item, qty in list(expected.items()):
            if item not in plan and qty > 0 and self.failures.get(item, 0) < 2:
                return Intent(IK_CART_SET, {"item": item, "quantity": 0, "held": qty}, P_COMMIT, None,
                              why=f"remove {item}")
        return None

    def _v7_notes_due(self, ep: Any, plan: dict) -> list:
        """Picks to write that only tools without notes opened, while a record tool shows notes (v7: their notes
        can point an attribute elsewhere)."""
        tools, out = ep.binder.binding.tools, []
        noted = ep.binder.open_plan(NEED_NOTES)
        if not noted or ep.plan.endgame() or not ep.plan.can_spend(1, P_VERIFY):
            return out
        for line in ep.ledger.lines:
            r = ep.catalog.get(line.choice) if line.choice else None
            if r is None or line.choice in self.bx_read or ep.cart.expected.get(line.choice, 0) == plan.get(line.choice) \
                    or not r.opened_by or FR_NOTES in r.families_seen or noted[0][0] in r.opened_by \
                    or any(FR_NOTES in tools[n].provides_seen for n in r.opened_by if n in tools) \
                    or not any(c.active for c in line.constraints):
                continue
            out.append(line.choice)
        return out[:max(1, ep.plan.batch_k())]

    def after(self, ep: Any, intent: Intent, dec: Decoded) -> None:
        if intent.payload.get("notes_read"):       # v7: the picks' notes were read: select again
            ep.selector.select(ep)
            return
        if intent is self.bx_search:               # the selector's own search step: absorb it, select again
            self.bx_search = None
            ep.selector.after(ep, intent, dec)
            ep.selector.select(ep)
            return
        if intent.payload.get("basket_read"):      # the picks were read: select again on their own attributes
            ids = list(intent.payload.get("ids") or [])
            got = [i for i in ids if i in ep.catalog and max(ep.catalog[i].opened_by.values(), default=-1)
                   >= ep.transport.n]              # v7: a single-id tool (or a batch limit) opened only some
            self.bx_part += 1 if len(got) < len(ids) else 0
            self.bx_rounds += 1 if len(got) >= len(ids) or self.bx_part > 4 else 0
            self.bx_read.update(got or ids)
            ep.selector.select(ep)
            return
        if intent.kind != IK_CART_SET:
            return
        if dec.kind not in (D_CART, D_ACK, D_UNKNOWN):
            self.note_failure(intent.payload.get("item"), blocked=dec.kind == D_BLOCKED)

    def observe(self, ep: Any, call: Call, dec: Decoded) -> None:
        cart = ep.cart
        u = ep.transport.n
        observed: dict = {}
        for ident, qty, price in dec.cart.get("lines") or []:
            observed[ident] = observed.get(ident, 0) + int(qty)
            cents = _price_to_cents(price, bool(ep.fields.price_major)) if price is not None else None
            r = ep.catalog.get(ident)
            if r is not None and cents and r.price_c != cents:
                r.price_hist.append((u, cents))
                r.price_c = cents
        total = dec.cart.get("total_c")
        if call.write is not None and call.role == ROLE_CART_WRITE:
            item, target = call.write
            prior = cart.expected.get(item, 0)
            sent = call.args.get(ep.binder.slot_arg(call.tool, SLOT_QTY) or "", None)
            got = observed.get(item, 0)
            if prior > 0 and isinstance(sent, int):
                learned = None
                if got == sent and got != prior + sent:
                    learned = "set"
                elif got == prior + sent and got != sent:
                    learned = "add"
                if learned and learned != cart.semantics:
                    ep.trace.decision("cart", semantics=learned, why="observed on a rewrite")
                if learned:
                    cart.semantics, cart.sem_source = learned, "observed"
            cart.writes += 1
            cart.returns_cart = True
            if cart.first_added is None and got > 0:
                cart.first_added = item
            cart.expected = {i: q for i, q in observed.items() if q > 0}
            cart.last_change_u = u
        cart.observed = observed
        cart.observed_u = u
        cart.observed_ambiguous = bool(dec.cart.get("ambiguous"))
        cart.unverified = False
        cart.total_observed = int(total) if isinstance(total, (int, float)) and not isinstance(total, bool) else None
        if {i: q for i, q in observed.items() if q > 0} == {i: q for i, q in cart.expected.items() if q > 0}:
            cart.verified_u = u

    def on_ack(self, ep: Any, call: Call, dec: Decoded) -> None:
        cart = ep.cart
        if call.write is None:
            return
        item, new_qty = call.write
        if new_qty > 0:
            cart.expected[item] = new_qty
        else:
            cart.expected.pop(item, None)
        cart.writes += 1
        if cart.returns_cart is None and not (dec.cart and dec.cart.get("echo")):
            cart.returns_cart = False
        if cart.first_added is None and new_qty > 0:
            cart.first_added = item
        cart.last_change_u = ep.transport.n

    def on_unverified(self, ep: Any, call: Call, dec: Decoded) -> None:
        self.on_ack(ep, call, dec)
        ep.cart.unverified = True
        ep.trace.decision("cart", unverified=True, item=call.write[0] if call.write else None,
                          why="the write answered an unreadable shape; read back before ordering")

    def fresh_observed(self, ep: Any) -> Optional[dict]:
        """The observed cart (positive lines) when it was read after the last change, else None."""
        cart = ep.cart
        if cart.observed is None or cart.observed_u is None or cart.observed_u < (cart.last_change_u or 0) \
                or cart.observed_ambiguous:
            return None
        return {i: q for i, q in cart.observed.items() if q > 0}

    def readback(self, ep: Any) -> tuple:
        cart = ep.cart
        plan = self.plan(ep)
        issues = []
        fresh = cart.observed is not None and cart.observed_u is not None and cart.observed_u >= (cart.last_change_u or 0)
        lines = {i: q for i, q in (cart.observed if fresh else cart.expected).items() if q > 0}
        if lines != plan:
            issues.append("lines differ from the plan")
        for ident in lines:
            r = ep.catalog.get(ident)
            if r is not None and r.stock is False and ep.ledger.in_stock:
                issues.append(f"dead line {ident}")
        total = 0
        for ident, qty in lines.items():
            r = ep.catalog.get(ident)
            if r is None or r.price_c is None:
                total = None
                break
            total += qty * r.price_c
        if total is not None and ep.ledger.cap_c is not None and total > ep.ledger.cap_c:
            issues.append("total over the cap")
        if fresh and cart.total_observed is not None and total is not None and cart.total_observed != total \
                and not ep.fields.price_major:
            issues.append("observed total differs from sum(price * qty)")
        if not ep.selector.bundle_ok(ep):
            issues.append("accessory codes miss the base")
        return not issues, issues

    def done(self, ep: Any) -> bool:
        plan = self.plan(ep)
        expected = {i: q for i, q in ep.cart.expected.items() if q > 0}
        short = not all(l.choice for l in ep.ledger.lines) or self.over_cap(ep)
        if (short or (not ep.cart.writes and ep.selector.bx_weak(ep))) and self._bx_search_ok(ep):
            return False                           # next_intent has the selector's search step
        if short:
            return True                            # nothing is written: the finisher decides
        return expected == plan or self.next_intent(ep) is None

EVT_LEDGER_KINDS = ("budget", "drop", "change", "quantity")   # kinds that change the ledger, not a listing
EVT_MAX_ROUNDS = 4                 # recovery rounds per episode
EVT_ROUND_STEPS = 8                # environment steps per recovery round
_S2_HUNT = 4                       # hunt steps per line (Q4 reads: as many again)
EVT_TWIN = 0.6                     # title-token Jaccard of twin listings (DESIGN 5.13)
EVT_RATIO_ORACLE = (0.05, 20.0)    # a typed budget must stay within these factors of the old cap
EVT_RATIO_GUESS = (0.2, 5.0)       # ... and the substitute's lone number within these
EVT_SOURCE = "src:event"           # Constraint.source of a changed value (the bare word is a world key)

def _evt_stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token

def _evt_mentions(text: Any, attr: Any) -> bool:
    """The attribute's content tokens appear as one contiguous run in the text (light plural folding)."""
    want = [_evt_stem(t) for t in content_tokens(attr)]
    have = [_evt_stem(t) for t in text_tokens(text)]
    n = len(want)
    return n > 0 and any(have[i:i + n] == want for i in range(len(have) - n + 1))

def _evt_same_attr(a: Any, b: Any) -> bool:
    ta = {_evt_stem(t) for t in content_tokens(a)}
    tb = {_evt_stem(t) for t in content_tokens(b)}
    return bool(ta and tb) and (ta <= tb or tb <= ta)

def _s2_cls(line: Line, ident: str) -> int:
    """Tier rank, T1/T2P as one on a cheapest line."""
    t = tier_rank(getattr(line.verdicts.get(ident), "tier", "T3"))
    return 0 if line.cheapest and t < 2 else t

def _v6_hard(v: Verdict, sold: bool = False) -> bool:
    """v6: an own check contradicted (stock, budget, shelf, codes); sold: out of stock within the budget."""
    pc = v.per_constraint
    if sold:
        return pc.get("chk:stock", ("",))[0] == "CONTRADICTED" and "chk:budget" not in pc
    return any(str(k).startswith("chk:") and s[0] == "CONTRADICTED" for k, s in pc.items())

def _evt_number(text: str, span: Any, amount: Any) -> Optional[tuple]:
    """(value, raw, start, end) re-extracted by code: from the oracle's span when it is verbatim in the text
    (the number equal to the oracle's amount, else the span's only number), else the oracle's amount when the
    text writes it; None otherwise."""
    pool = numbers_in(text)
    in_span = []
    if isinstance(span, str) and norm_space(span) and norm_space(span) in norm_space(text):
        raws = [n[1] for n in numbers_in(span)]
        in_span = [n for n in pool if n[1] in raws]
    if amount is not None and looks_numeric(amount):
        hit = [n for n in (in_span or pool) if abs(float(n[0]) - float(as_number(amount))) < 1e-6]
        if hit:
            return hit[0]
    return in_span[0] if len(in_span) == 1 else None

class EventMachine(Engine):
    name = "events"

    def __init__(self, ep: Any = None):
        super().__init__(ep)
        self.seq = 0                   # recovery triggers raised
        self.done_seq = 0              # triggers covered by a finished round
        self.trigger_u: Optional[int] = None
        self.why: list = []
        self.item_risk = False         # a trigger may have changed a listing: re-open the cart
        self.next_u: Optional[int] = None   # first trigger raised while a round was active
        self.next_risk = False
        self.reopen = False
        self.rounds = 0
        self.active = False
        self.round_seq = 0
        self.round_from = 0            # observations at or after this step are fresh in the round
        self.steps = 0
        self.tried: set = set()        # replacement ids opened this round
        self.failed: set = set()       # ids whose open failed or was blocked this round
        self.searched: set = set()     # line idx searched this round
        self.emitted: set = set()      # intent signatures sent this round
        self.reselected = False
        self.drop_lines: set = set()   # lines with a dropped constraint (cheapest re-search)
        self.round_drops: set = set()
        self.facts: set = set()        # silent changes already raised
        self.endgame_opens: dict = {}  # line idx -> replacement opens in the endgame
        self.hunts: dict = {}          # line idx -> hunt and Q4 steps
        self.swapped: set = set()      # lines whose pick changed this round
        self.read: set = set()         # lines whose best unread backup was read this round
        self.batch_fault = False       # a batch open failed: one id at a time
        self.by_oracle: set = set()    # id() of events typed by the EVENT oracle
        self.raws: dict = {}           # id(event) -> the budget number as written (cap_span)

    def ingest(self, ep: Any, dec: Decoded, intent: Optional[Intent] = None) -> None:
        ev = ep.events
        u = ep.transport.n
        new = []
        for text, via in [(t, "refusal") for t in dec.refusal] + [(t, "pending") for t in dec.messages]:
            if not isinstance(text, str) or text in ev.seen_texts:
                continue
            ev.seen_texts.add(text)
            event = Event(text=text, u=u, via=via)
            ev.queue.append(event)
            new.append(event)
            if ep.budget.first_finish_step is None:
                ev.before_first_finish += 1
        if new:
            ev.dirty = True
            ep.trace.decision("event_texts", texts=[clip_text(e.text, 200) for e in new], via=[e.via for e in new])
        if dec.kind == D_REFUSED and intent is not None:
            ev.refused_last = FIN_DECLINE if intent.kind == IK_DECLINE else FIN_ORDER
            ev.decline_locked = True       # a refusal means a valid order exists (0/93 re-declines, 1/144 after)
            ep.trace.decision("refusal", finish=ev.refused_last, texts=[clip_text(t, 200) for t in dec.refusal],
                              decline_locked=ev.decline_locked)
            self.raise_trigger(ep, "refusal", not (dec.refusal or dec.messages))   # texts: their kinds decide
        self.note_silent(ep, dec)

    def note_silent(self, ep: Any, dec: Decoded) -> None:
        """A cart item or line choice read out of stock or at a new price is a (possibly silent) event."""
        if self.active or dec.kind not in (D_RECORDS, D_RECORD, D_CART):
            return
        watch = {i for i, q in ep.cart.expected.items() if q > 0}
        if not watch:
            return
        if dec.kind == D_CART:
            ids = [line[0] for line in (dec.cart or {}).get("lines") or []]
        else:
            ids = [ep.fields.extract(dec.tool, r).get(FR_ID) for r in dec.records if isinstance(r, dict)]
        u = ep.transport.n
        for ident in [i for i in ids if i in watch]:
            r = ep.catalog.get(ident)
            if r is None:
                continue
            if r.stock is False and ep.ledger.in_stock and (ident, "stock") not in self.facts:
                self.facts.add((ident, "stock"))
                self.raise_trigger(ep, f"{ident} out of stock", True)
            if len(r.price_hist) >= 2 and r.price_hist[-1][0] == u and (ident, r.price_c) not in self.facts:
                self.facts.add((ident, r.price_c))
                self.raise_trigger(ep, f"{ident} price {r.price_hist[-2][1]}->{r.price_c}", True)

    def raise_trigger(self, ep: Any, why: str, item_risk: bool = False) -> None:
        self.seq += 1
        u = ep.transport.n
        if self.active:                    # handled by the next round, fresh from its own step
            self.next_u = u if self.next_u is None else self.next_u
            self.next_risk = self.next_risk or item_risk
        else:
            self.trigger_u = u if self.trigger_u is None else self.trigger_u
            self.item_risk = self.item_risk or item_risk
        self.why = (self.why + [why])[-8:]

    def needs_recovery(self, ep: Any) -> bool:
        return not ep.finished and self.seq > self.done_seq and self.rounds < EVT_MAX_ROUNDS

    def process(self, ep: Any) -> list:
        ev = ep.events
        pending = [e for e in ev.queue if not e.applied]
        ev.dirty = False
        if not pending:
            return []
        self.type_events(ep, pending)
        applied = []
        for e in pending:
            e.applied = True
            if self.apply(ep, e):
                applied.append(e.kind)
        ev.decline_locked = True           # never decline after a change (race data: 1 of 242 passed)
        ep.ledger.version += 1
        touched = ep.cart.writes > 0 or bool(ep.cart.expected)
        if touched:
            risk = any(e.kind not in EVT_LEDGER_KINDS for e in pending)
            self.raise_trigger(ep, "events " + ",".join(e.kind for e in pending), risk)
        ep.trace.decision("events_processed", n=len(pending), kinds=[e.kind for e in pending], applied=applied,
                          ledger_v=ep.ledger.version, cap_c=ep.ledger.cap_c, recover=touched)
        ep.plan.compute("events")          # the reserve R grows by 2 once event text appears early
        return pending

    def type_events(self, ep: Any, events: list) -> None:
        led = ep.ledger
        answers: dict = {}
        if ep.llm.allowed("EVENT"):
            payload = {"request": ep.query, "events": [{"i": k, "text": e.text} for k, e in enumerate(events)],
                       "lines": [{"line": l.idx, "label": l.shelf_label or l.words} for l in led.lines]}
            ans = ep.llm.ask("EVENT", payload, validator=lambda v: self.validate(ep, events, v))
            if isinstance(ans, dict):
                answers = {it["i"]: it for it in ans.get("events") or []}
        for k, e in enumerate(events):
            it = answers.get(k)
            if it is None or it.get("kind") is None:
                self.substitute(ep, e)
                continue
            self.by_oracle.add(id(e))
            e.kind, e.attr, e.line, e.new_value = it["kind"], it.get("attr"), it.get("line"), it.get("new_val")
            num = it.get("num")
            if e.kind == "budget":
                e.amount_c = self.budget_cents(ep, e.text, num, EVT_RATIO_ORACLE)
                e.kind = "budget" if e.amount_c else "other"
                self.raws[id(e)] = num[1] if num else ""
            elif e.kind == "quantity":
                e.amount_c = int(num[0]) if num and float(num[0]).is_integer() and 1 <= num[0] <= 99 else None
                e.kind = "quantity" if e.amount_c else "other"
        ep.trace.decision("event_typed", by=["oracle" if id(e) in self.by_oracle else "substitute" for e in events],
                          kinds=[e.kind for e in events], amounts=[e.amount_c for e in events],
                          attrs=[e.attr for e in events], lines=[e.line for e in events])

    def validate(self, ep: Any, events: list, value: dict) -> tuple:
        """EVENT answers: indexes exist; numbers re-extracted from verbatim spans; a drop or change names its
        attribute in the text; a changed value is verbatim; the line exists. Invalid items fall back to the
        substitute (kind None)."""
        out, seen = [], set()
        n_lines = len(ep.ledger.lines)
        for it in value.get("events") or []:
            i = it.get("i")
            if not isinstance(i, int) or not 0 <= i < len(events) or i in seen:
                continue
            seen.add(i)
            text, kind = events[i].text, it.get("kind")
            num = _evt_number(text, it.get("amount_span"), it.get("amount"))
            attr, new_val = it.get("attr"), it.get("new_val")
            if kind in ("budget", "quantity") and num is None:
                kind = None
            if kind in ("drop", "change") and not (isinstance(attr, str) and _evt_mentions(text, attr)):
                kind = None
            if kind == "change" and not (isinstance(new_val, str) and norm_space(new_val)
                                         and norm_space(new_val) in norm_space(text)):
                kind = None
            line = it.get("line") if isinstance(it.get("line"), int) and 0 <= it["line"] < n_lines else None
            out.append({"i": i, "kind": kind, "num": num, "attr": attr if isinstance(attr, str) else None,
                        "line": line, "new_val": new_val if isinstance(new_val, str) else None})
        if not out:
            return False, None, "no event typed"
        return True, {"events": out}, ""

    def substitute(self, ep: Any, e: Event) -> None:
        """Dead-oracle typing from numbers and structure only (DESIGN 5.17): a lone number the cart does not
        explain is a budget; a digit-free text naming exactly one active constraint's attribute is a drop."""
        nums = numbers_in(e.text)
        e.kind = "unknown"
        if len(nums) == 1:
            cents = self.budget_cents(ep, e.text, nums[0], EVT_RATIO_GUESS)
            if cents and not self.explained(ep, nums[0][0], cents):
                e.kind, e.amount_c = "budget", cents
                self.raws[id(e)] = nums[0][1]
            return
        if nums:
            return
        hits = [(line.idx, c.attr) for line in ep.ledger.lines for c in line.constraints
                if c.active and _evt_mentions(e.text, c.attr)]
        if len(hits) == 1:
            e.kind, e.line, e.attr = "drop", hits[0][0], hits[0][1]

    def explained(self, ep: Any, value: Any, cents: int) -> bool:
        """The number is a quantity, a listing price or the basket total already known: not a budget."""
        ids = {i for i, q in ep.cart.expected.items() if q > 0} | {l.choice for l in ep.ledger.lines if l.choice}
        prices = {ep.catalog[i].price_c for i in ids if i in ep.catalog and ep.catalog[i].price_c}
        total = ep.selector.basket_total(ep)
        quantities = {l.qty for l in ep.ledger.lines} | set(ep.cart.expected.values())
        return cents in prices or (total is not None and cents == total) or value in quantities

    @staticmethod
    def budget_cents(ep: Any, text: str, num: Optional[tuple], ratio: tuple) -> Optional[int]:
        """The new cap in cents under the episode's cap-unit rule: a decimal amount is major units (x 100); an
        integer is scaled x 100 when the original cap was (check_cap_unit), or, for an oracle-typed budget only,
        when only that reading is plausible against the old cap. None when implausible."""
        if not num:
            return None
        value, raw, start, end = num
        cents = cap_to_cents(value, raw)
        if not cents or cents <= 0:
            return None
        decimal = cents != int(value)
        old = ep.ledger.cap_c
        if not decimal and getattr(ep, "cap_unit_checked", False):
            cents *= 100
        elif not decimal and old and ratio is EVT_RATIO_ORACLE and not ratio[0] * old <= cents <= ratio[1] * old \
                and ratio[0] * old <= cents * 100 <= ratio[1] * old:
            cents *= 100
        if old:
            return cents if ratio[0] * old <= cents <= ratio[1] * old else None
        if ratio is EVT_RATIO_GUESS and not has_currency_symbol_near(text, start, end):
            return None
        return cents

    def apply(self, ep: Any, e: Event) -> bool:
        """Apply one typed event to the ledger; True when the ledger changed."""
        led = ep.ledger
        lines = led.lines
        target = [lines[e.line]] if isinstance(e.line, int) and 0 <= e.line < len(lines) else list(lines)
        if e.kind == "budget" and e.amount_c:
            if led.cap_c == e.amount_c:
                return False
            led.cap_c, led.cap_span = e.amount_c, self.raws.get(id(e)) or str(e.amount_c)
            for line in lines:
                self.invalidate(ep, line, "budget")
            return True
        if e.kind == "drop" and e.attr:
            hit = False
            for line in target:
                for c in line.constraints:
                    if c.active and _evt_same_attr(c.attr, e.attr):
                        c.active = False
                        hit = True
                        self.drop_lines.add(line.idx)
                        self.invalidate(ep, line, "drop")
                line.closed.add(canon_key(e.attr))          # a dropped need is never asked; other needs stay
            return hit
        if e.kind == "change" and e.attr and e.new_value:
            for line in target[:1] if len(target) == 1 else []:
                old = [c for c in line.constraints if c.active and _evt_same_attr(c.attr, e.attr)]
                for c in old:
                    c.active = False
                op = old[0].op if old and old[0].op in ("eq", "in", "ne") else "eq"
                line.constraints.append(Constraint(cid=f"e{len(line.constraints)}", attr=canon_key(e.attr), op=op,
                                                   vals=[e.new_value], source=EVT_SOURCE, span=e.new_value))
                self.invalidate(ep, line, "change")
                return True
            return False
        if e.kind == "quantity" and e.amount_c and len(target) == 1 and target[0].qty != e.amount_c:
            target[0].qty = int(e.amount_c)
            self.invalidate(ep, target[0], "quantity")
            return True
        return False

    @staticmethod
    def invalidate(ep: Any, line: Line, why: str) -> None:
        ep.coverage.invalidate(ep, line, why)
        line.verdicts.clear()

    def next_intent(self, ep: Any) -> Optional[Intent]:
        if not self.active:
            if not self.needs_recovery(ep):
                return None
            self.start(ep)
        it = self.plan_step(ep) if self.steps < EVT_ROUND_STEPS else None
        if it is not None and it.signature() in self.emitted:
            it = None                      # never the same action twice in a round
        if it is None:
            self.finish(ep)
            return None
        self.emitted.add(it.signature())
        return it

    def start(self, ep: Any) -> None:
        self.active = True
        self.rounds += 1
        self.round_seq = self.seq
        self.round_from = self.trigger_u if self.trigger_u is not None else ep.transport.n
        self.steps, self.reselected = 0, False
        self.tried, self.failed, self.searched, self.emitted, self.swapped = set(), set(), set(), set(), set()
        self.read = set()
        self.reopen = self.item_risk
        self.round_drops = set(self.drop_lines)
        ep.trace.decision("recover", round=self.rounds, why=list(self.why), since_u=self.round_from,
                          reopen=self.reopen)

    def finish(self, ep: Any) -> None:
        if not self.active:
            return
        self.rescue(ep)
        self.active = False
        self.done_seq = self.round_seq
        self.trigger_u, self.item_risk = self.next_u, self.next_risk
        self.next_u, self.next_risk = None, False
        if self.seq <= self.done_seq:
            self.why = []
        self.drop_lines -= self.round_drops
        ep.cartman.q4_advisory = True      # Q4 was read here; a deferred write would empty the cart
        ep.trace.decision("recover", round=self.rounds, done=True, steps=self.steps,
                          choices={str(l.idx): l.choice for l in ep.ledger.lines})

    def rescue(self, ep: Any) -> None:
        """v6: a sold-out choice never outlives a round: the best other row replaces it, no step, if writes fit."""
        for line in ep.ledger.lines:
            budget = ep.selector.line_budget(ep, line)
            if not line.choice or not _v6_hard(ep.checker.verdict(ep, line, line.choice, budget), True):
                continue
            on = self._v6_on(ep, line)
            pool = sorted((i for i in ep.selector.orderable(ep, line, budget) if i != line.choice
                           and i not in self.failed and on(i)), key=lambda i: not ep.catalog[i].opened_by)
            pool += self.v6_salvage(ep, line, on, budget)
            if pool and self.room(ep, 0, extra=2):
                self.swap(ep, line, pool[0], "dead at the round's end")

    def done(self, ep: Any) -> bool:
        return not self.active

    def after(self, ep: Any, intent: Intent, dec: Decoded) -> None:
        if dec.kind != D_BLOCKED:
            self.steps += 1
        ids = [i for i in intent.payload.get("ids") or [] if isinstance(i, str)]
        if intent.kind == IK_OPEN:
            if len(ids) > 1 and not dec.ok:           # v5: no id failed
                self.batch_fault = True
                return
            if dec.gid in ("DET", "BUILD") and intent.payload.get("need") == NEED_ATTRS:
                self.hunts[intent.line] = 2 * _S2_HUNT   # v6: no attribute reader (not the id)
                return
            if not dec.ok or any(not self.fresh(ep, i) for i in ids):
                self.failed |= {i for i in ids if not self.fresh(ep, i)}
            if intent.payload.get("replacement"):
                self.tried |= set(ids)
                if ep.plan.endgame() and intent.line is not None:
                    self.endgame_opens[intent.line] = self.endgame_opens.get(intent.line, 0) + 1
        elif intent.kind in (IK_SEARCH, IK_FILTER) and intent.line is not None:
            self.searched.add(intent.line)

    def plan_step(self, ep: Any) -> Optional[Intent]:
        if self.reopen:
            it = self.reopen_intent(ep)
            if it is not None:
                return it
        if not self.reselected:
            self.reselected = True
            self.reselect(ep)
        for line in ep.ledger.lines:
            if not line.choice:                           # nothing orderable: one more search, then re-select
                it = self.search_more(ep, line, "no candidate")
                self.reselected = it is None
                if it is not None:
                    return it
                continue
            it = self.fix_line(ep, line)
            if it is not None:
                return it
        for line in ep.ledger.lines:                      # Q4: a swapped cheapest pick waits for cheaper rows
            n = self.hunts.get(line.idx, 0)
            ids = [i for i in ep.coverage.q4_rows(ep, line) if i not in self.tried | self.failed] \
                if line.idx in self.swapped and n < 2 * _S2_HUNT and self.hunt_ok(ep) else []
            if ids:
                self.hunts[line.idx] = n + 1
                return Intent(IK_OPEN, {"ids": ids[:1] if self.batch_fault else ids, "need": NEED_ATTRS,
                                        "replacement": True}, P_RECOVER, line.idx,
                              why=f"recovery: Q4 rows of line {line.idx}")
        return None

    def fresh(self, ep: Any, ident: str) -> bool:
        r = ep.catalog.get(ident)
        return bool(r is not None and r.opened_by and max(r.opened_by.values()) >= self.round_from)

    def writes_needed(self, ep: Any) -> int:
        plan = ep.cartman.plan(ep)
        have = {i: q for i, q in ep.cart.expected.items() if q > 0}
        return sum(1 for i in plan if have.get(i) != plan[i]) + sum(1 for i in have if i not in plan)

    def room(self, ep: Any, k: int, extra: int = 0) -> bool:
        """k more steps leave the writes the plan needs, `extra` more, and the order."""
        return ep.plan.rem() - k >= self.writes_needed(ep) + extra + 1

    def reopen_intent(self, ep: Any) -> Optional[Intent]:
        """OPEN(cart ids, need=stock): carts carry no stock flag. The first-added item first; all in one batch
        with a batch tool, otherwise the other items only while the budget is comfortable."""
        cart = [i for i, q in ep.cart.expected.items() if q > 0]
        first = ep.cart.first_added if ep.cart.first_added in cart else (cart[0] if cart else None)
        order = ([first] if first else []) + [i for i in cart if i != first]
        stale = [i for i in order if i not in self.failed and not self.fresh(ep, i)]
        if not stale or not self.room(ep, 1):
            return None
        if ep.binder.tool_for(ROLE_BATCH) is not None and not self.batch_fault:
            ids = stale
        elif stale[0] == first or self.room(ep, 1, extra=len(stale) + 1):
            ids = stale[:1]
        else:
            return None
        return Intent(IK_OPEN, {"ids": ids, "need": NEED_STOCK}, P_RECOVER, None,
                      why="re-open the cart after " + clip_text(",".join(self.why[-2:]), 60))

    def reselect(self, ep: Any) -> None:
        """A line without a choice or a multi-line basket over the (new) cap: re-select through the selector,
        then keep every earlier choice that is still orderable while the basket fits (minimal cart churn)."""
        led = ep.ledger
        total = ep.selector.basket_total(ep)
        over = total is not None and led.cap_c is not None and total > led.cap_c and len(led.lines) > 1
        if not over and all(l.choice for l in led.lines):
            return                         # a single over-cap line is a dead choice for fix_line
        before = {l.idx: l.choice for l in led.lines}
        ep.selector.select(ep)
        for line in led.lines:
            old = before.get(line.idx)
            if not old or old == line.choice:
                continue
            if old not in ep.selector.orderable(ep, line, ep.selector.line_budget(ep, line)):
                continue
            new, line.choice = line.choice, old
            total = ep.selector.basket_total(ep)
            if total is not None and led.cap_c is not None and total > led.cap_c:
                line.choice = new
        self.swapped |= {l.idx for l in led.lines if l.choice != before.get(l.idx)}
        ep.trace.decision("recover", reselect=True, over_cap=over, cap_c=led.cap_c,
                          choices={str(l.idx): l.choice for l in led.lines}, was={str(k): v for k, v in before.items()})

    def price(self, ep: Any, ident: Optional[str]) -> int:
        r = ep.catalog.get(ident) if ident else None
        return r.price_c if r is not None and r.price_c is not None else 10 ** 12

    def fix_line(self, ep: Any, line: Line) -> Optional[Intent]:
        """Swap a dead or (on cheapest lines) undercut choice; open every new pick before it is written. v5: on the
        pick's shelf when it is the label and names the line (_v6_on). v6: _v6_clash; v6_salvage after the hunt."""
        cur = line.choice
        if not cur:
            return None
        budget = ep.selector.line_budget(ep, line)
        dv = ep.checker.verdict(ep, line, cur, budget)
        dead = dv.tier == "X"
        in_cart = ep.cart.expected.get(cur, 0) > 0
        if not dead and not in_cart and not self.fresh(ep, cur) and cur not in self.failed:
            return self.open_pick(ep, line, cur)          # a new pick (re-selection): verify before writing
        if not dead and not line.cheapest:
            return None                                   # nothing got worse: the same item again
        old, on = ep.catalog.get(cur), self._v6_on(ep, line)
        options = [i for i in ep.selector.orderable(ep, line, budget) if i != cur and i not in self.failed and on(i)]
        here = canon_key(old.shelf) if old is not None and old.shelf else ""
        same = [i for i in options if here and canon_key(ep.catalog[i].shelf or "") == here]
        cls = {i: _s2_cls(line, i) for i in options}
        why = "dead" if dead else "cheaper"
        if dead:
            ref = old.price_hist[-2][1] if old is not None and len(old.price_hist) >= 2 else self.price(ep, cur)
            title = old.title if old is not None else ""
            clash = self._v6_clash(ep, line, old)
            options.sort(key=lambda i: (cls[i] > 2, clash(i), cls[i]) + (() if line.cheapest else (i not in same, not (
                token_overlap(title, ep.catalog[i].title) >= EVT_TWIN and self.price(ep, i) <= ref))))
            unread = [i for i in options if not (ep.catalog[i].attrs or ep.coverage._opened(ep, ep.catalog[i]))]
            unread.sort(key=lambda i: (cls[i], self.price(ep, i) if line.cheapest else 0))
            it = self.hunt(ep, line, on, unread) if not options or cls[options[0]] or clash(options[0]) else None
            if it is not None or not options:
                it = it or self.search_more(ep, line, "no backup")
                if it is not None or not _v6_hard(dv, True):
                    return it
                options, why = self.v6_salvage(ep, line, on, budget), "dead, lower class"
                if not options:                           # no row: page on
                    self.searched.discard(line.idx)
                    return self.search_more(ep, line, "dead pick, no row")
        else:
            options = [i for i in (same or options)[:1] if self.price(ep, i) < self.price(ep, cur)]
            if not options:
                if line.idx in self.drop_lines:
                    return self.search_more(ep, line, "constraint dropped")
                return None
        for cand in options:
            if self.fresh(ep, cand):
                self.swap(ep, line, cand, why)
                return None
            if cand in self.tried:
                continue
            it = self.open_pick(ep, line, cand)
            if it is not None:
                return it
            if ep.catalog[cand].opened_by or (dead and self.room(ep, 0, extra=2)):
                self.swap(ep, line, cand, why + ", unverified")
            return None
        return None

    def _v6_on(self, ep: Any, line: Line) -> Callable:
        old, lab = ep.catalog.get(line.choice or ""), canon_key(line.shelf_label or "")
        shelf = lab if old is not None and old.shelf and canon_key(old.shelf) == lab and _cov_stems(lab) & _cov_stems(
            line.words + " " + line.shelf_words) else ""
        return lambda i: not (shelf and ep.catalog[i].shelf) or canon_key(ep.catalog[i].shelf) == shelf

    def _v6_clash(self, ep: Any, line: Line, old: Optional[Record]) -> Callable:
        """v6: id -> keys it contradicts of the dead pick's 2+ keys (attribute word) holding an unnamed value."""
        chk, keys = ep.checker, []
        holds = (lambda c, x: any(chk._level(chk.toks(v), chk.toks(x), chk.elems(str(x))) in CHK_LEVELS[:3]
                                  for v in c.vals))
        for c in line.constraints if old is not None else ():
            if c.active and c.op in ("eq", "in") and c.span and not _evt_mentions(c.span, c.attr):
                aw = {_evt_stem(t) for t in content_tokens(c.attr)}
                ks = [(c, k) for k, b in old.attrs.items() if aw & {_evt_stem(t) for t in content_tokens(k)}
                      and any(holds(c, x) for _, x in b)]
                keys += ks if len(ks) > 1 else []
        return lambda i: sum(1 for c, k in keys if any(_chk_valued(x) for _, x in ep.catalog[i].attrs.get(k, ()))
                             and not any(holds(c, x) for _, x in ep.catalog[i].attrs[k]))

    def v6_salvage(self, ep: Any, line: Line, on: Callable = bool, budget: Optional[int] = None) -> list:
        """v6: rows not dead for good, by contradictions of what the dead pick confirmed, all of them, rank()."""
        chk, out = ep.checker, {}
        kept = {k for k, s in chk.verdict(ep, line, line.choice, budget).per_constraint.items() if s[0] == "CONFIRMED"}
        ids = [i for i in ep.selector.candidates(ep, line) if i != line.choice and i not in self.failed and on(i)
               and ep.catalog[i].price_c is not None]
        void = ep.resolver.trap_check(ep, line, ids)
        for i in ids:
            v = chk.verdict(ep, line, i, budget)
            bad = {k for k, s in v.per_constraint.items() if s[0] == "CONTRADICTED"}
            if not (void.get(i) or _v6_hard(v)):
                out[i] = (len(bad & kept), len(bad))
        return sorted(ep.selector.rank(ep, line, list(out)), key=out.get)

    def open_pick(self, ep: Any, line: Line, cand: str) -> Optional[Intent]:
        """OPEN(cand) before it is written (payload replacement: allowed in the endgame, once per line there)."""
        if not self.room(ep, 1, extra=2):
            return None
        if ep.plan.endgame() and self.endgame_opens.get(line.idx, 0) >= 1:
            return None
        r = ep.catalog.get(cand)
        known = r is not None and (r.attrs or FR_ATTRS in r.families_seen)
        need = NEED_STOCK if known or not any(c.active for c in line.constraints) else NEED_FULL
        return Intent(IK_OPEN, {"ids": [cand], "need": need, "replacement": True}, P_RECOVER, line.idx,
                      why=f"verify replacement {cand} for line {line.idx}")

    def hunt(self, ep: Any, line: Line, on: Callable = bool, unread: Any = ()) -> Optional[Intent]:
        """No backup of class 0: the best unread backup (v5, once a round), else rows with its values, else search."""
        n = self.hunts.get(line.idx, 0)
        if n >= _S2_HUNT or not self.hunt_ok(ep):
            return None
        try:
            ids = [i for i in unread if i not in self.tried][:1] if line.idx not in self.read else []
            self.read.add(line.idx)
            ids = ids or sorted((i for i in ep.selector._sx_rows(ep, line) if i not in self.tried | self.failed
                                 and on(i)), key=lambda i: _s2_cls(line, i))   # v6: class first
            k = 1 if self.batch_fault else max(1, ep.plan.batch_k())
            it = Intent(IK_OPEN, {"ids": ids[:k], "need": NEED_ATTRS, "replacement": True},
                        P_RECOVER, line.idx, why=f"recovery: rows with line {line.idx}'s values") if ids \
                else self.search_more(ep, line, "hunt", value=True)
        except Exception as exc:                          # a hunt never ends the recovery
            ep.trace.decision("crash", where="hunt", err=repr(exc)[:200])
            it, n = None, _S2_HUNT
        self.hunts[line.idx] = n + (it is not None)
        return it

    def hunt_ok(self, ep: Any) -> bool:
        """Room for a hunt step, a verifying open, the writes and the order, before 190 s. endgame() is off once a
        finish was tried, so after a refusal room() is the bound."""
        return not ep.plan.endgame() and ep.clock() < T_EXPLORE_STOP and self.steps + 2 < EVT_ROUND_STEPS \
            and self.room(ep, 1, extra=3)

    def search_more(self, ep: Any, line: Line, why: str, value: bool = False) -> Optional[Intent]:
        """One SEARCH per line and round: the next page of the line's best query, else its value query
        without dropped values (value: the literal value query first). Only while time and steps allow a later
        open, swap and order."""
        if line.idx in self.searched or ep.clock() >= T_EXPLORE_STOP or not self.room(ep, 1, extra=3):
            return None
        recs = ep.queries.for_line(line.idx)
        if value:
            vw = {t for c in line.constraints if c.active for v in c.vals for t in content_tokens(v) if len(t) > 1}
            recs = [q for q in recs if vw & set(q.tokens)]
            ran = {_cov_stems(q.query_text) for q in ep.queries.recs.values()}
            fresh = [q for q in ep.coverage.value_queries(ep, line) if _cov_stems(q) not in ran]
        else:
            queries = ep.scout.queries_for(ep, line) if hasattr(ep.scout, "queries_for") else []
            ran = {q.query_text for q in recs}
            fresh = [q for q in queries + [line.words] if str(q or "").strip() and q not in ran]
        best = max((q for q in recs if q.ids and q.query_text and q.has_cont), key=lambda q: len(q.ids), default=None)
        payload: dict = {"qkind": "q:relaxed"}
        if best is not None and not (fresh and (value or line.idx in self.drop_lines)):
            payload.update(query=best.query_text, cont=best.next_cont, qkey=best.canon_args)
        elif fresh:
            payload["query"] = fresh[0]                   # without dropped values; never the same query twice
        else:
            return None
        self.searched.add(line.idx)
        return Intent(IK_SEARCH, payload, P_RECOVER, line.idx, why=f"recovery search for line {line.idx}: {why}")

    def swap(self, ep: Any, line: Line, cand: str, why: str) -> None:
        old = line.choice
        line.choice = cand
        self.swapped.add(line.idx)
        line.backups = [i for i in line.backups if i not in (cand, old)][:3]
        ep.trace.decision("recover", line=line.idx, swap=[old, cand], why=why,
                          price=[self.price(ep, old) if old else None, self.price(ep, cand)])
        if line.role == "base":
            bundle = ep.selector.select_bundle(ep)          # any base swap re-matches the accessories (M6)
            for idx, ident in (bundle or {}).items():
                if isinstance(idx, int) and 0 <= idx < len(ep.ledger.lines) and ident in ep.catalog:
                    ep.ledger.lines[idx].choice = ident

_DEC_REMARK = "No available listing meets every requirement within the budget, so no order was placed."
_DEC_QUOTED = re.compile(r"[\"“”]([^\"“”]{2,80})[\"“”]")
_DEC_EARLY = ("SETUP", "SCOUT", "FEASIBILITY", "ELICIT", "RESOLVE")

def _dec_state(v: Any, cid: str) -> tuple:
    """(state, level, quote) of a constraint in a checker verdict."""
    st = (getattr(v, "per_constraint", None) or {}).get(cid)
    st = [st.get(k) for k in ("state", "level", "quote")] if isinstance(st, dict) else st
    st = list(st) if isinstance(st, (list, tuple)) else []
    return tuple((st + [None] * 3)[:3])

def _dec_run(a: list, b: list) -> bool:
    return bool(a) and any(b[i:i + len(a)] == a for i in range(len(b) - len(a) + 1))

def _dec_own_quote(r: Record, quote: Any) -> bool:
    """The quote is read from the item's own attribute values (never a title or a note)."""
    q = text_tokens(quote)
    return bool(q) and any(_dec_run(q, x) or _dec_run(x, q) for x in
                           (text_tokens(v) for vals in r.attrs.values() for _, v in vals))

_V5D_PROVES = ("OWN", "UNJUDGED", "TITLE", "MIRROR")     # contradiction levels that prove

def _v5d_pos(ep: Any, r: Record, c: Constraint, level: str = "") -> bool:
    """A key holding every word of the attribute ('skirt style' for 'style') satisfies c (a bound: a number inside
    it): no decline rests on the item. ne: only a title-only contradiction, no such key holding the value."""
    chk = ep.checker
    tok, aw = chk.toks, set(chk.toks(c.attr))
    lim = chk._limits(c) if c.op in ("ge", "le", "between") else None
    xs = [x for k, vals in r.attrs.items() if aw and aw <= set(tok(k)) for _, x in vals if tok(x)]
    hit = [any(_chk_run(tok(x), tok(v)) for v in c.vals if tok(v)) for x in xs]
    if c.op == "ne":
        return level == "TITLE" and bool(xs) and not any(hit)
    return any((h and c.op in ("eq", "in")) or (lim is not None and chk._bound_status(str(x), lim) == "sat")
               for x, h in zip(xs, hit))

class DeclineProver(Engine):
    name = "prover"

    def __init__(self, ep: Any = None):
        super().__init__(ep)
        self.cancelled: set = set()      # proof keys an audit refused or cancelled
        self.audits: dict = {}           # proof key -> audit attempts
        self.last: Optional[Proof] = None
        self.logged: Any = None
        self._ck: Any = None
        self.ev_spent = 0                # decline-evidence steps sent
        self.ev_tried: set = set()
        self.ev_key: Optional[tuple] = None
        self.ev_at = 0
        self.ev_last: Optional[str] = None
        self.ev_exhausted = False
        self.gathered = False            # the evidence engine read something (the orchestrator re-selects)
        self._dx_wide = self._dx_fit = self._dx_sweep = True   # gather(): every unread row / only the choice;
        self.remark_wanted = False       # the last passing audit read a request for a closing remark

    @staticmethod
    def key(proof: Proof) -> tuple:
        return proof.kind, tuple(sorted(proof.lines))

    def proof(self, ep: Any) -> Optional[Proof]:
        ck = (ep.transport.n, ep.ledger.version, ep.phase, ep.ledger.cap_c, len(ep.catalog), len(self.cancelled),
              self.ev_exhausted, ep.events.decline_locked)
        if ck != self._ck:
            self._ck, self.last = ck, self._find(ep)
            p = self.last
            if p is not None and (self.key(p), p.numbers.get("total_c")) != self.logged:
                self.logged = (self.key(p), p.numbers.get("total_c"))
                ep.trace.decision("proof", proof_kind=p.kind, lines=p.lines, numbers=p.numbers, ids=p.ids[:12],
                                  quotes=p.quotes[:6], certs=p.certs)
        return self.last

    def _find(self, ep: Any) -> Optional[Proof]:
        led = ep.ledger
        if not led.lines or led.source in ("fallback", "none") or led.decline_rule.get("allowed", True) is False:
            return None
        late = ep.phase not in _DEC_EARLY
        makers = (lambda: [ep.feasibility.e1(ep)],
                  lambda: [self._settled(ep, l) for l in led.lines] if late else [],
                  lambda: self._overcap_nofit(ep) if late and self.deadline(ep) else [])
        for make in makers:
            for p in make():
                if p is not None and self.key(p) not in self.cancelled:
                    return p
        return None

    def deadline(self, ep: Any) -> bool:
        hard = ep.plan.nums.get("u_c_hard")
        return (ep.phase in ("COMMIT", "CHECKOUT", "RECOVER", "FALLBACK") or self.ev_exhausted
                or (hard is not None and ep.plan.u() >= hard) or ep.clock() >= T_EXPLORE_STOP
                or not ep.plan.can_spend(1, P_PROBE))

    def _line_ok(self, ep: Any, line: Line) -> bool:
        return (line.shelf_conf >= 2 and not line.guessed
                and not any(r.line == line.idx and r.status == "pending" for r in ep.ledger.refs)
                and not any(c.active and c.op == "pair_pending" for c in line.constraints))

    def budget(self, ep: Any, line: Line) -> Optional[int]:
        cap, fl = ep.ledger.cap_c, ep.feasibility.floors(ep)
        others = sum(o.qty * fl.get(o.idx, 0) for o in ep.ledger.lines if o.idx != line.idx)
        return None if cap is None else cap - others

    def complete_set(self, ep: Any, line: Line) -> Optional[list]:
        """Fully read noun/value queries (one value each), one relaxed, or a fully read shelf bracket reaching the
        budget (no lexical assumption). v2_d: no coverage certificate (only cheapest lines have one)."""
        noun = set(content_tokens(line.shelf_words or line.words)) | set(content_tokens(line.shelf_label or ""))
        vals = [set(content_tokens(c.vals[0])) for c in _dec_hard(ep, line, ("eq",)) if len(c.vals) == 1]
        ids, relaxed = [], False
        for rec in [r for r in ep.queries.for_line(line.idx) if _dec_enumerated(r)]:
            toks = set(rec.tokens)
            if toks and rec.query_text and (toks <= noun or any(toks <= noun | v for v in vals)):
                ids.extend(rec.ids)
                relaxed = relaxed or (len(rec.tokens) <= 3 and not any(v and v <= toks for v in vals))
            elif self._dx_reach(ep, line, rec):
                ids.extend(rec.ids)
                relaxed = True
        return list(dict.fromkeys(ids)) if relaxed and ids else None

    def _dx_reach(self, ep: Any, line: Line, rec: QueryRec) -> bool:
        """A label bracket (no maker, rows obey it) reaching the budget per unit (unknown inclusivity: exclusive)."""
        arg, budget = ep.binder.slot_arg(rec.tool, SLOT_SHELF), self.budget(ep, line)
        sent = _dec_cents(ep, rec.args.get(ep.binder.slot_arg(rec.tool, SLOT_CEILING) or ""))
        return (not rec.query_text and sent is not None and budget is not None and bool(arg and line.shelf_label)
                and canon_key(rec.args.get(arg)) == canon_key(line.shelf_label) and not ep.coverage.st(line).filter_bad
                and ep.binder.slot_arg(rec.tool, SLOT_MAKER) not in rec.args
                and sent - (0 if ep.contract.ceiling_inclusive else 1) >= budget // max(1, line.qty))

    def bad(self, ep: Any, line: Line, r: Record, v: Any = None) -> Optional[tuple]:
        """(why, attr, quote) when provably invalid: gone, or a proof constraint (no review guess) contradicted at a
        _V5D_PROVES level without _v5d_pos."""
        if not _dec_available(ep, r) or (line.shelf_conf >= 2 and not _dec_on_shelf(line, r)):
            return "gone", "", ""
        v = v if v is not None else ep.checker.verdict(ep, line, r.id, None)
        for c in _dec_hard(ep, line):
            state, level, quote = _dec_state(v, c.cid)
            numeric = c.op in ("ge", "le", "between")
            if state == "CONTRADICTED" and level in _V5D_PROVES and c.source != "review" and (
                    level == "TITLE" or _dec_own_quote(r, quote)) and (
                    not numeric or any(ch.isdigit() for ch in str(quote))) and not _v5d_pos(ep, r, c, level):
                return ("bound" if numeric else "contra"), c.attr, clip_text(quote, 80)
        return None

    def orderable(self, ep: Any, line: Line) -> list:
        """[(price, id)] of priced candidates not provably invalid, cheapest first."""
        ids = [i for i in ep.selector.candidates(ep, line) if ep.catalog[i].price_c is not None
               and i not in ep.absent_ids]
        void = ep.resolver.trap_check(ep, line, ids)
        return sorted((ep.catalog[i].price_c, i) for i in ids if not void.get(i) and self.bad(ep, line, ep.catalog[i])
                      is None)

    def basket(self, ep: Any, held: dict, todo: list, opts: Optional[dict] = None) -> Optional[dict]:
        """{idx: id} for todo keeping held {idx: id} (unpriced: 0) inside the cap, accessories fitting the base: least
        tier sum, then total, from opts {idx: [ids]} or orderable(); per line the cheapest per tier."""
        cap, lines, sel, cat = ep.ledger.cap_c, ep.ledger.lines, ep.selector, ep.catalog
        b_idx, accs = sel.roles(ep)
        ranked = {line.idx: sorted((tier_rank(ep.checker.verdict(ep, line, i, None).tier), cat[i].price_c, i) for i in (
            opts[line.idx] if opts is not None else [i for _, i in self.orderable(ep, line)])
            if i in cat and cat[i].price_c is not None) for line in todo}
        rb = ranked.get(b_idx, [])
        best = None
        for base in [held.get(b_idx)] if b_idx not in ranked else dict.fromkeys(
                [x[2] for x in rb[:4]] + [x[2] for x in sorted(rb, key=lambda x: x[1])[:6]]):
            def fits(k: int, i: str) -> bool:
                return not (base and k in accs and sel.enforced(ep, lines[k]) and sel.compatible(ep, i, base) is False)
            combos = [(0, sum(lines[k].qty * sel._price(ep, i) for k, i in held.items()), {})] if all(
                fits(k, i) for k, i in held.items()) else []
            for line in todo:
                front, low = [], None
                for t, p, i in ranked[line.idx]:
                    if (low is None or p < low) and fits(line.idx, i) and (line.idx != b_idx or i == base):
                        front.append((t, p, i))
                        low = p
                combos = [(n + t, tot + line.qty * p, {**c, line.idx: i}) for n, tot, c in combos for t, p, i in front
                          if cap is None or tot + line.qty * p <= cap]
            best = min(([best] if best else []) + combos, key=lambda x: x[:2], default=None)
        return best[2] if best else None

    def _settled(self, ep: Any, line: Line) -> Optional[Proof]:
        ids = self.complete_set(ep, line) if self._line_ok(ep, line) else None
        if ids is None:
            return None
        budget = self.budget(ep, line)
        cands = [r for r in (ep.catalog.get(i) for i in ids) if r is not None and _dec_on_shelf(line, r)]
        fits = [r for r in cands if _dec_available(ep, r)       # an unknown price counts as affordable
                and (budget is None or r.price_c is None or line.qty * r.price_c <= budget)]
        free = [i for p, i in self.orderable(ep, line) if budget is None or line.qty * p <= budget]
        hits = {r.id: self.bad(ep, line, r) for r in fits if r.opened_by}
        if not fits or free or not all(hits.get(r.id) for r in fits):
            return None
        return Proof(kind="E2" if all(h[0] == "bound" for h in hits.values()) else "E3", lines=[line.idx],
                     ids=[r.id for r in cands][:40], quotes=[[i, h[1], h[2]] for i, h in hits.items()][:20],
                     numbers={"budget_c": budget, "candidates": len(cands), "contradicted": len(hits)},
                     certs=[line.cert.kind if line.cert else "enum"])

    def _overcap_nofit(self, ep: Any) -> list:
        led = ep.ledger
        minima, nofit = {}, []
        for line in led.lines:
            ok = self.orderable(ep, line) if self._line_ok(ep, line) else None
            read = [i for i in ep.selector.candidates(ep, line)
                    if ep.catalog[i].opened_by and _dec_on_shelf(line, ep.catalog[i])]
            if ok is None or not (ok or read):
                return []                  # an uncertain line, or one never read: no evidence either way
            if ok:
                minima[line.idx] = ok[0]
            else:
                nofit.append(Proof(kind="NOFIT", lines=[line.idx], ids=read[:40], numbers={"read": len(read)}))
        total = sum(led.lines[i].qty * p for i, (p, _) in minima.items())
        if nofit or led.cap_c is None:
            return nofit[:1]
        b_idx, accs = ep.selector.roles(ep)
        st = ep.selector.basket_status(ep) if b_idx is not None and accs else {}
        if (st.get("complete") and st.get("within_cap")) or self.basket(ep, {}, led.lines) is not None:
            return []
        hard = ep.plan.nums.get("u_c_hard")
        searched = ep.phase in ("CHECKOUT", "RECOVER", "FALLBACK") or (hard is not None and ep.plan.u() >= hard)
        if total <= led.cap_c and not (st.get("complete") and searched):
            return []
        return [Proof(kind="OVERCAP", lines=sorted(minima), ids=[i for _, i in minima.values()],
                      numbers={"cap_c": led.cap_c, "total_c": total if total > led.cap_c else st.get("total"),
                               "minima": {str(i): p for i, (p, _) in minima.items()},
                               "compatible": bool(st and total <= led.cap_c)})]

    def preconditions_ok(self, ep: Any, proof: Proof) -> bool:
        ev, lines = ep.events, ep.ledger.lines
        return (not (ev.refused_last or ev.dirty or ev.decline_locked)
                and all(e.applied and e.u < ep.transport.n for e in ev.queue)   # never right after an event
                and ep.ledger.decline_rule.get("allowed", True) is not False and bool(proof.lines)
                and all(0 <= i < len(lines) and lines[i].shelf_conf >= 2 and not lines[i].guessed for i in proof.lines))

    def ready(self, ep: Any) -> bool:
        proof = self.proof(ep)
        return proof is not None and self.preconditions_ok(ep, proof)

    def audit(self, ep: Any, proof: Proof) -> bool:
        key = self.key(proof)
        role = "AUDIT_E1" if proof.kind == "E1" else "AUDIT"
        if key in self.cancelled or self.audits.get(key, 0) >= 2 or not ep.llm.allowed(role):
            ep.trace.decision("audit", proof_kind=proof.kind, audit_role=role, passed=False, why="no audit available")
            return False
        self.audits[key] = self.audits.get(key, 0) + 1
        payload, shown = self.payload(ep, proof)

        def check(ans: dict) -> tuple:
            bad = [o.get("rid") for o in ans.get("orderable") or [] if o.get("rid") not in shown]
            return (False, None, f"rid not among the records: {clip_text(bad[0], 40)}") if bad else (True, ans, "")
        ans = ep.llm.ask(role, payload, validator=check)
        if ans is None:
            ep.trace.decision("audit", proof_kind=proof.kind, audit_role=role, passed=False, why="no answer")
            return False
        listed = [(o["line"] if o.get("line") in proof.lines + [shown[o["rid"]]] else shown[o["rid"]], o["rid"])
                  for o in ans.get("orderable") or []]
        hits = self.recheck(ep, proof, listed)
        sound = bool(ans.get("proof_sound"))
        self.remark_wanted = ans.get("remark_wanted") is True
        proof.orderable, proof.audited = hits, sound and not hits
        if not proof.audited:
            self.cancelled.add(key)
        ep.trace.decision("audit", proof_kind=proof.kind, audit_role=role, passed=proof.audited, proof_sound=sound,
                          listed=[list(x) for x in listed][:12], orderable={str(k): v for k, v in hits.items()},
                          issues=[clip_text(x, 120) for x in (ans.get("issues") or [])][:6])
        return proof.audited

    def recheck(self, ep: Any, proof: Proof, listed: list) -> dict:
        """{line: id} of listed ids, not provably invalid, fitting the cap beside the other lines' cheapest."""
        led, fl = ep.ledger, ep.feasibility.floors(ep)
        cheapest = {l.idx: (self.orderable(ep, l) or [(fl.get(l.idx, 0), None)])[0] for l in led.lines}
        out: dict = {}
        for idx, rid in listed:
            r = ep.catalog.get(rid)
            if r is None or not (0 <= idx < len(led.lines)) or rid in ep.absent_ids \
                    or self.bad(ep, led.lines[idx], r) is not None:
                continue
            others = sum(o.qty * cheapest[o.idx][0] for o in led.lines if o.idx != idx)
            if led.cap_c is not None and r.price_c is not None and led.lines[idx].qty * r.price_c + others > led.cap_c:
                continue
            if idx not in out or (r.price_c or 0) < (ep.catalog[out[idx]].price_c or 0):
                out[idx] = rid
        if out and proof.kind == "OVERCAP":
            out.update({i: ident for i, (_, ident) in cheapest.items() if i not in out and ident})
        return out

    def payload(self, ep: Any, proof: Proof) -> tuple:
        """({request, proof, lines, recs}, {rid: line} of the records shown)."""
        led = ep.ledger
        idxs = [i for i in proof.lines if 0 <= i < len(led.lines)] or [l.idx for l in led.lines]
        per = max(8, 32 // max(1, len(idxs)))
        lines, recs, shown = [], [], {}
        for idx in idxs:
            line = led.lines[idx]
            lines.append({"line": idx, "label": line.shelf_label or line.words, "words": line.words, "units": line.qty,
                          "reqs": [clip_text(f"{c.attr} {c.op} {', '.join(map(str, c.vals))} "
                                             f"{(c.bound or {}).get('raw') or ''} ({c.source})", 120)
                                   for c in line.constraints if c.active]})
            pool = [ep.catalog[i] for i in self._pool(ep, proof, line) if _dec_on_shelf(line, ep.catalog[i])]
            for r in sorted(pool, key=lambda r: (r.price_c if r.price_c is not None else 10 ** 12, r.id))[:per]:
                if r.id in shown:
                    continue
                shown[r.id] = idx
                recs.append({"rid": r.id, "line": idx, "head": clip_text(r.title, 90), "cost_c": r.price_c,
                             "instock": r.stock, "read": bool(r.opened_by),
                             "on_label": bool(r.shelf and line.shelf_label
                                              and canon_key(r.shelf) == canon_key(line.shelf_label)),
                             "attrs": {clip_text(k, 40): clip_text("; ".join(str(x) for _, x in vals), 60)
                                       for k, vals in list(r.attrs.items())[:12]}})
        facts = dict(proof.numbers, proof_kind=proof.kind, cap_c=led.cap_c,
                     quotes=[q if isinstance(q, str) else list(q) for q in proof.quotes][:12])
        return {"request": ep.query, "proof": facts, "lines": lines, "recs": recs}, shown

    @staticmethod
    def _pool(ep: Any, proof: Proof, line: Line) -> list:
        """E1: its evidence rows only (rows that never carried the value cannot cancel it); else every candidate."""
        if proof.kind != "E1":
            return list(dict.fromkeys([i for i in proof.ids if i in ep.catalog] + ep.selector.candidates(ep, line)))
        rec = ep.feasibility.proofs.get(line.idx) or {}
        if rec.get("how") != "enumeration":
            return [i for i in _dec_line_ids(line) if i in ep.catalog]
        toks = [set(content_tokens(v)) for v in rec.get("vals") or []]
        return list(dict.fromkeys([i for i in rec.get("ids") or [] if i in ep.catalog] + [
            i for i in _dec_line_ids(line) if i in ep.catalog
            and any(t and t <= set(text_tokens(ep.catalog[i].title)) for t in toks)]))

    def remark_slot(self, ep: Any) -> tuple:
        """(argument, required) where CallBuilder._finish puts a decline remark, or (None, False)."""
        bnd = ep.binder.binding
        if bnd.finish_kind == "enum" and bnd.finish_tool:
            tool = bnd.finish_tool
            arg = ep.binder.slot_arg(tool, SLOT_REMARK)
            arg = None if arg == bnd.finish_arg else arg
        else:
            tool = ep.binder.tool_for(ROLE_DECLINE)
            args = ep.binder.features[tool].args if tool in ep.binder.features else {}
            arg = next((a for a, f in args.items() if f.required and f.is_string), None) or next(
                (a for a, f in args.items() if f.free_text), None)
        feat = ep.binder.features[tool].args.get(arg) if tool in ep.binder.features and arg else None
        return (arg, bool(feat.required)) if feat else (None, False)

    def remark(self, ep: Any) -> Optional[str]:
        arg, required = self.remark_slot(ep)
        rules = [r for r in (ep.ledger.decline_rule.get("remark_rules") or []) if isinstance(r, str) and r.strip()]
        if arg is None or not (required or (rules and self.remark_wanted)):
            return None
        quoted = [m.strip() for r in rules for m in _DEC_QUOTED.findall(r)]
        return next((t for t in (quoted if len(quoted) == 1 else []) + [_DEC_REMARK] if self.remark_ok(ep, t)), None)

    @staticmethod
    def remark_ok(ep: Any, text: str) -> bool:
        """No catalog id, id-shaped word or listing title."""
        t = str(text or "").strip()
        words = [w.strip(".,;:!?()[]{}\"'") for w in t.split()]
        return (0 < len(t) <= 240 and not any(w in ep.catalog or is_id_shaped(w) for w in words)
                and not any(len(r.title) >= 12 and norm_space(r.title) in norm_space(t) for r in ep.catalog.values()))

    def next_intent(self, ep: Any) -> Optional[Intent]:
        self.ev_key, led = None, ep.ledger
        total = ep.selector.basket_total(ep)
        over = total is not None and led.cap_c is not None and total > led.cap_c
        if ep.phase == "PROVE" and led.lines and led.source not in ("fallback", "none") \
                and not (ep.events.decline_locked or ep.events.refused_last) \
                and led.decline_rule.get("allowed", True) is not False and ep.llm.allowed("AUDIT"):
            todo = []
            for line in [l for l in led.lines if self._line_ok(ep, l)]:
                v = ep.checker.verdict(ep, line, line.choice, None) if line.choice else None   # fresh: reads change it
                wide = over or v is None or v.tier == "X"         # a choice read as X leaves the line without one
                if wide or v.tier in ("T2", "T3"):
                    todo.append((not wide, line.idx, line, wide))
            for _, _, line, wide in sorted(todo, key=lambda x: x[:2]):
                it = self.gather(ep, line, wide)
                if it is not None:
                    return it
        self.ev_exhausted = True
        return None

    def gather(self, ep: Any, line: Line, wide: bool = True) -> Optional[Intent]:
        """The line's next evidence step (PROBE, certificate payload): see the header. v1 stopped after two opens
        and the relaxed query, so an OVERCAP rested on rows never read (v2_d)."""
        if self.ev_spent >= ceil_div(3 * ep.budget.B, 10):
            return None
        self._dx_wide = wide
        self._dx_sweep = wide or ep.plan.u() < ep.plan.nums.get("u_c_soft", 10 ** 6)   # an orderable basket within
        rows = (self._dx_page_intent, self._dx_bracket_intent, self._relaxed_intent) if wide else (  # the cap: hold
            (self._dx_page_intent,) if self._dx_sweep else ())             # a complete cart by the commit window
        sel = ep.selector                  # an accessory line whose rows all miss the base's codes: pages first
        fit = self._dx_fit = bool(wide and line.idx in sel.roles(ep)[1] and sel.enforced(ep, line)
                                  and not sel.orderable(ep, line) and sel.orderable(ep, line, compat=False))
        for make in rows + (self._open_intent,) if fit else (self._open_intent,) + rows:
            it = make(ep, line)
            if it is not None and it.signature() == self.ev_last:
                self.ev_tried.add(self.ev_key)           # no progress since the last step: give it up
                it = None
            if it is not None:
                self.ev_at, self.ev_last = ep.transport.n, it.signature()
                return it
        return None

    def relaxed_query(self, ep: Any, line: Line) -> str:
        """<= 3 tokens missing part of every hard value: a long value's first token and the noun."""
        noun = content_tokens(line.shelf_words or line.words)
        vals = [c.vals[0] for c in _dec_hard(ep, line, ("eq",)) if len(c.vals) == 1 and _dec_searchable(c.vals[0])]
        toks = [t for t in content_tokens(vals[0]) if t not in noun] if vals else []
        return " ".join(_dedupe(toks[:1] + noun)[:3] if len(toks) > 1 else noun[:3])

    def _relaxed_intent(self, ep: Any, line: Line) -> Optional[Intent]:
        q = self.relaxed_query(ep, line)
        key, rec = ("rq", line.idx, q), _dec_rec_for(ep, q)
        payload = {"query": q, "qkind": "q:relaxed", "certificate": True}
        if not q or key in self.ev_tried:
            return None
        if rec is not None:
            if _dec_enumerated(rec) or not rec.has_cont or rec.capped or rec.count > ep.contract_learner.window() \
                    or _dec_pages_left(ep, rec) > ceil_div(3 * ep.budget.B, 10) - self.ev_spent:
                self.ev_tried.add(key)
                return None
            payload.update(query=rec.query_text, cont=rec.next_cont, qkey=rec.canon_args)   # tokens are opaque
        self.ev_key = key
        return Intent(IK_SEARCH, payload, P_PROBE, line.idx, why=f"decline evidence: relaxed query '{q}'")

    def _open_intent(self, ep: Any, line: Line) -> Optional[Intent]:
        """Unread rows within the budget, each once: the choice first, then cheapest, title-contradicted last."""
        budget, cs = self.budget(ep, line), set(self.complete_set(ep, line) or ())
        rows = sorted((r for r in (ep.catalog.get(i) for i in _dec_line_ids(line))
                       if r is not None and not r.opened_by and ("id", r.id) not in self.ev_tried
                       and r.id not in ep.absent_ids and _dec_on_shelf(line, r) and _dec_available(ep, r)
                       and r.price_c is not None and (budget is None or line.qty * r.price_c <= budget)
                       and (self._dx_wide or r.id == line.choice or (self._dx_sweep and r.id in cs))),
                      key=lambda r: (ep.checker.verdict(ep, line, r.id, None).tier == "X", r.id != line.choice,
                                     r.price_c, r.id))
        _, chunk = ep.binder.route_open([r.id for r in rows[:6]], NEED_ATTRS) if rows else (None, [])
        if not chunk:
            return None
        self.ev_key = ("open", line.idx, tuple(chunk))
        return Intent(IK_OPEN, {"ids": list(chunk), "need": NEED_ATTRS, "certificate": True}, P_PROBE, line.idx,
                      why=f"decline evidence: read unread rows of line {line.idx}")

    def _dx_page_intent(self, ep: Any, line: Line) -> Optional[Intent]:
        """A partly read query's next page, brackets first: whole enumerations only (partial pages of a broad query
        surface rows that lack the values), except on a line short of a fitting accessory (codes ride on rows)."""
        left = ceil_div(3 * ep.budget.B, 10) - self.ev_spent
        for rec in sorted(ep.queries.for_line(line.idx), key=lambda r: (bool(r.query_text), _dec_pages_left(ep, r))):
            key = ("pg", rec.canon_args, len(rec.pages_read))
            if not rec.has_cont or rec.next_cont is None or rec.capped or key in self.ev_tried \
                    or (not self._dx_fit and _dec_pages_left(ep, rec) > left) \
                    or not (self._dx_wide or self._dx_reach(ep, line, rec)):
                continue
            payload = {"query": rec.query_text}
            if not rec.query_text:
                c = _dec_cents(ep, rec.args.get(ep.binder.slot_arg(rec.tool, SLOT_CEILING) or ""))
                if c is None or not line.shelf_label or ep.binder.slot_arg(rec.tool, SLOT_MAKER) in rec.args:
                    continue
                payload = {"shelf": line.shelf_label, "ceiling": c}
            payload.update(cont=rec.next_cont, qkey=rec.canon_args, qkind=rec.kind, certificate=True)
            self.ev_key = key
            return Intent(IK_SEARCH if rec.query_text else IK_FILTER, payload, P_PROBE, line.idx,
                          why=f"decline evidence: next page of '{clip_text(rec.query_text or 'bracket', 40)}'")
        return None

    def _dx_bracket_intent(self, ep: Any, line: Line) -> Optional[Intent]:
        """One shelf bracket reaching the budget, or below the cheapest orderable price; not once complete literal
        queries cover the line (rows they lack are lexically excluded, DESIGN 5.13)."""
        cov = ep.coverage
        if cov._lit_cert(ep, line, cov._lits(ep, line, cov.st(line), relax=False), None):
            return None
        key, budget, own = ("br", line.idx), self.budget(ep, line), self.orderable(ep, line)
        cs = [x for x in (own[0][0] - 1 if own and not self._dx_fit else None, budget // max(1, line.qty) + (
            0 if ep.contract.ceiling_inclusive else 1) if budget is not None else None) if x is not None]
        c = min(cs) if cs else None
        if key in self.ev_tried or c is None or c <= 0 or not ep.coverage._filter_ready(ep, line, ep.coverage.st(line)):
            return None
        for rec in ep.queries.for_line(line.idx):
            sent = _dec_cents(ep, rec.args.get(ep.binder.slot_arg(rec.tool, SLOT_CEILING) or ""))
            if not rec.query_text and not rec.has_cont and sent is not None and sent >= c \
                    and ep.binder.slot_arg(rec.tool, SLOT_MAKER) not in rec.args:
                return None
        self.ev_key = key
        return Intent(IK_FILTER, {"shelf": line.shelf_label, "ceiling": c, "qkind": "q:bracket", "certificate": True},
                      P_PROBE, line.idx, why=f"decline evidence: line {line.idx} rows at or under {c}")

    def after(self, ep: Any, intent: Intent, dec: Decoded) -> None:
        key, self.ev_key = self.ev_key, None
        if key is not None and ep.transport.n > self.ev_at:
            self.ev_spent += 1
            self.gathered = True
        if key is not None and key[0] == "open":
            self.ev_tried.update(("id", i) for i in key[2])
        elif key is not None and (key[0] in ("br", "pg") or dec.kind != D_RECORDS):
            self.ev_tried.add(key)

    def done(self, ep: Any) -> bool:
        return self.ev_exhausted

class Finisher:
    MAX_ORDER_ATTEMPTS = 5
    MAX_CHECKOUT_LOOPS = 14
    MAX_GATE_STEPS = 6
    FALLBACK_ROUNDS = 10
    WRITE_RETRIES = 2
    MAX_CART_SYNCS = 2

    def __init__(self):
        self.cart_syncs = 0                # read-backs that rewrote the cart (bounded: no verify/rewrite cycle)
        self.v5d_fix = False               # v5: an over-cap cart went to the fallback

    def fast_path_ok(self, ep: Any) -> bool:
        """L = 1, the write returned a cart or an ACK, the item was opened <= 3 steps earlier, no event pending."""
        if len(ep.ledger.lines) != 1 or ep.cart.unverified:
            return False
        line = ep.ledger.lines[0]
        if not line.choice or ep.cart.expected.get(line.choice) != line.qty or ep.cart.writes == 0 \
                or line.choice in ep.cartman.bx_read:  # v7: a notes read before the write is no stock read after it
            return False
        r = ep.catalog.get(line.choice)
        opened_u = max(r.opened_by.values()) if r is not None and r.opened_by else None
        if opened_u is None or ep.transport.n - opened_u > 3:
            return False
        return not ep.events.dirty and all(e.applied for e in ep.events.queue)

    def needs_verify(self, ep: Any) -> bool:
        if self.fast_path_ok(ep):
            return False
        cart = ep.cart
        return cart.unverified or cart.verified_u is None or (cart.last_change_u or 0) > cart.verified_u

    def verify(self, ep: Any) -> None:
        """Batch re-open of the cart (or the first-added item without a batch tool), a cart read when writes
        return no cart or a write was unreadable, then read-back (Q6). An observed cart that differs from the
        expected one replaces it and the plan is rewritten (PROT-4)."""
        cart = ep.cart
        ids = list(cart.expected)
        if not ids:
            return
        batch = ep.binder.tool_for(ROLE_BATCH) is not None
        targets = ids if batch else [cart.first_added if cart.first_added in ids else ids[0]]
        it = Intent(IK_OPEN, {"ids": targets, "need": NEED_STOCK}, P_VERIFY, None, why="re-open the cart before ordering")
        if ep.plan.can_spend(1, P_VERIFY):
            ep.run(it)
        if (cart.returns_cart is False or cart.unverified) and ep.binder.tool_for(ROLE_CART_VIEW) \
                and ep.plan.can_spend(1, P_VERIFY) and ep.plan.rem() >= 3:
            ep.run(Intent(IK_CART_VIEW, {}, P_VERIFY, None, why="read back the cart"))
        ok, issues = ep.cartman.readback(ep)
        ep.trace.decision("readback", ok=ok, issues=issues)
        if self.adopt_observed(ep, "read-back"):
            return                                 # rewritten: the next round reads it back again
        cart.verified_u = ep.transport.n
        if ep.cartman.fresh_observed(ep) is not None:
            cart.unverified = False

    def adopt_observed(self, ep: Any, why: str) -> bool:
        """A cart read after the last change that differs from the expected cart is the truth: it becomes the
        expected cart and the plan is rewritten (writes that never landed are sent again)."""
        fresh = ep.cartman.fresh_observed(ep)
        expected = {i: q for i, q in ep.cart.expected.items() if q > 0}
        if fresh is None or fresh == expected or self.cart_syncs >= self.MAX_CART_SYNCS:
            return False
        self.cart_syncs += 1
        ep.trace.decision("cart_sync", why=why, expected=expected, observed=fresh)
        missing_writes = [i for i in expected if fresh.get(i, 0) < expected[i]]
        if missing_writes and not fresh:
            ep.binder.flip_cart_mode()             # nothing landed: a positional add literal may be the remove
        ep.cart.expected = dict(fresh)
        ep.cart.last_change_u = ep.transport.n
        if ep.plan.rem() >= 2:
            self.resync(ep)
        return True

    def settle_uncertain(self, ep: Any) -> None:
        """DESIGN 5.2: after a step conflict (state_uncertain) the cart is re-read before any irreversible call:
        a cart view, or a re-open of the cart ids without one (F19)."""
        if not ep.transport.state_uncertain or ep.plan.rem() < 2:
            return
        ep.transport.state_uncertain = False
        if ep.binder.tool_for(ROLE_CART_VIEW):
            ep.run(Intent(IK_CART_VIEW, {}, P_VERIFY, None, why="re-read the cart after a step conflict"))
        elif ep.cart.expected:
            ep.run(Intent(IK_OPEN, {"ids": list(ep.cart.expected), "need": NEED_STOCK}, P_VERIFY, None,
                          why="re-open the cart after a step conflict"))
        self.adopt_observed(ep, "step conflict")

    def resync(self, ep: Any, forget: bool = False) -> None:
        """Rewrite the planned cart (after an empty-cart fault, a lost write or a changed plan). In the fallback,
        or once the 240 s mark has passed, the writes go out directly (bounded and time-checked): the phase
        loop would raise GoFallback and abort the finish."""
        if forget:
            ep.cart.expected = {}
            ep.cart.verified_u = None
        if not any(line.choice for line in ep.ledger.lines):
            ep.selector.select(ep)
        phase = ep.phase
        if phase == "FALLBACK" or ep.clock() >= T_FALLBACK:
            self.write_plan(ep)
        else:
            ep.orchestrator.drive(ep.cartman, "COMMIT")
        ep.phase = phase

    def write_plan(self, ep: Any) -> None:
        """CartManager's writes without the phase loop: at most two per line, each one time-checked."""
        for _ in range(2 * max(1, len(ep.ledger.lines)) + 2):
            it = ep.cartman.next_intent(ep)
            if it is None or it.kind != IK_CART_SET or not self.time_for(ep, 1) or ep.plan.rem() < 2:
                return
            dec = ep.run(it)
            ep.cartman.after(ep, it, dec)

    def sync_plan(self, ep: Any) -> None:
        """Write the plan when it differs from the expected cart (a recovery swap, F02)."""
        expected = {i: q for i, q in ep.cart.expected.items() if q > 0}
        if expected and ep.cartman.plan(ep) != expected:
            self.resync(ep)

    def reread_cart(self, ep: Any) -> bool:
        """An ORDER fault with a cart view available: read the cart and rewrite what never landed."""
        if not ep.binder.tool_for(ROLE_CART_VIEW) or ep.plan.rem() < 3:
            return False
        dec = ep.run(Intent(IK_CART_VIEW, {}, P_VERIFY, None, why="read the cart after an order fault"))
        return dec.kind == D_CART and self.adopt_observed(ep, "order fault")

    def checkout_loop(self, ep: Any) -> bool:
        ep.phase = "CHECKOUT"
        attempts = 0
        fault_resyncs = 0
        loops = 0
        em = ep.events_machine
        while not ep.finished:
            if ep.clock() >= T_FALLBACK:
                raise GoFallback("time")
            if ep.plan.rem() < 1:
                raise StopEpisode("step_budget")
            if attempts >= self.MAX_ORDER_ATTEMPTS:
                raise GoFallback("order attempts exhausted")
            loops += 1
            if loops > self.MAX_CHECKOUT_LOOPS:
                raise GoFallback("checkout loop")
            if ep.events.dirty:
                em.process(ep)
            if ep.events.refused_last or em.needs_recovery(ep):
                ep.orchestrator.drive(em, "RECOVER")     # re-open, swap, re-select; never writes the cart
                ep.phase = "CHECKOUT"
            self.sync_plan(ep)
            if not ep.cart.expected:
                self.resync(ep)
                if not ep.cart.expected:
                    raise GoFallback("empty cart at checkout")
            if self.needs_verify(ep) and ep.plan.rem() >= 3:
                self.verify(ep)
                if not ep.cart.expected:
                    continue
                if ep.events.dirty or em.needs_recovery(ep):
                    continue                             # a change seen while verifying: recover first
            self.settle_uncertain(ep)
            seq0 = em.seq
            for _ in range(self.MAX_GATE_STEPS):          # Q7: every gate action the planner allows, before the ORDER
                gate = ep.reviewer.procedure_gate(ep)     # (each is issued at most once; integration of M4 + M8)
                if gate is None or not ep.plan.allows(gate):
                    break
                ep.run(gate)
            if ep.events.dirty or (em.seq > seq0 and em.needs_recovery(ep)):
                continue                                 # v6: a gate step's news, or a silent change it read, first
            if not self.v5d_fix and self._v5d_rewrite(ep) is not None:    # v5: a budget cut, after the gate
                self.v5d_fix = True
                raise GoFallback("cart over the cap")
            dec = ep.run(Intent(IK_ORDER, {}, P_FINISH, None, why="order the cart"))
            attempts += 1
            if dec.kind == D_FINISHED:
                return True
            if dec.kind == D_REFUSED:
                continue
            if dec.kind == D_EMPTY_CART:
                fault_resyncs += 1
                ep.binder.flip_cart_mode()
                self.resync(ep, forget=True)
                continue
            if dec.kind in (D_WORLD_FAULT, D_EMPTY) and fault_resyncs == 0:
                fault_resyncs += 1
                if self.reread_cart(ep):
                    continue
                if ep.cart.returns_cart is not True:
                    ep.binder.flip_cart_mode()
                    self.resync(ep, forget=True)
                    continue
            if dec.kind == D_NONFINAL:
                if not self.complete_two_step(ep, dec):
                    raise GoFallback("two-step finish needs the operator")
                continue
            if dec.kind == D_BLOCKED:
                raise GoFallback(f"order blocked by {dec.gid}")
            if dec.kind in (D_CART, D_STATUS, D_RECORDS, D_RECORD, D_REPLY, D_ACK):
                if ep.binder.binding.finish_kind != "enum" and ep.binder.tool_for(ROLE_ORDER) is None:
                    raise GoFallback("no order tool left")
                continue
            role = ROLE_ORDER if ep.binder.binding.finish_kind != "enum" else ROLE_FINISH
            rung = ep.binder.repair(role, f"order returned {dec.kind}", failing=dec.tool)
            if rung in ("degrade", "exhausted"):
                raise GoFallback(f"order failed: {dec.kind}")
        return ep.finished

    def complete_two_step(self, ep: Any, dec: Decoded) -> bool:
        """NONFINAL (a two-step checkout): the operator's completion (M8), else token_completion; v7: one more
        deterministic candidate after a completion that does not finish."""
        call, tried = ep.operator.complete_finish(ep, dec), set()
        for _ in range(2):
            call = call or self.token_completion(ep, dec, tuple(tried))
            if call is None or not ep.can_act():
                return False
            tried.add(call.tool)
            out = ep.run(Intent(IK_RAW, {"tool": call.tool, ARGS_KEY: call.args, "finish_completion": True}, P_FINISH,
                                None, why="complete a two-step finish"))
            if out.kind == D_FINISHED:
                return True
            call = None
        return False

    @staticmethod
    def token_completion(ep: Any, dec: Decoded, exclude: tuple = ()) -> Optional[Call]:
        """The handle verbatim to a tool whose one required argument is a short string, bound as other or as a
        record tool that never returned records (v7), ranked by word overlap with the order tool and the answer's
        keys, then role other first."""
        inner = dec.inner if isinstance(dec.inner, dict) else {}
        token = next((v for v in inner.values() if token_like(v)), None)
        if token is None:
            return None
        bnd, feats = ep.binder.binding, ep.binder.features
        hint = " ".join([feats[dec.tool].text if dec.tool in feats else ""] + [" ".join(name_tokens(k)) for k in inner]
                        + [str(v) for v in inner.values() if v != token])
        cands = []
        for i, (name, tb) in enumerate(bnd.tools.items()):
            tf = feats[name]
            feat = tf.args.get(tf.required[0]) if len(tf.required) == 1 else None
            if name in exclude or name == dec.tool or tb.decline_votes or tb.provides_seen or feat is None \
                    or not feat.short_key_arg or tb.role not in (ROLE_OTHER, ROLE_VIEW, ROLE_SPEC):
                continue
            text = tf.text + " " + " ".join(" ".join(name_tokens(a)) for a in tf.args)
            cands.append((-token_overlap(text, hint), tb.role != ROLE_OTHER, i, name, feat.name))
        if not cands:
            return None
        _, _, _, name, arg = min(cands)
        return Call(tool=name, args={arg: token}, role=bnd.tools[name].role, path="det_fallback")

    def try_decline(self, ep: Any) -> bool:
        """M7: a proven, audited decline (DESIGN 5.18). An audit that finds an orderable id cancels the decline
        and that id becomes the line's choice (proof.orderable, F27)."""
        proof = ep.prover.proof(ep)
        if proof is None or not ep.prover.preconditions_ok(ep, proof):
            return False
        if not ep.prover.audit(ep, proof):
            for idx, ident in (proof.orderable or {}).items():
                if isinstance(idx, int) and 0 <= idx < len(ep.ledger.lines) and ident in ep.catalog:
                    ep.ledger.lines[idx].choice = ident
                    ep.trace.decision("audit_order", line=idx, choice=ident)
            return False
        payload = {"basis": f"proof:{proof.kind}"}
        remark = ep.prover.remark(ep)
        if remark:
            payload["remark"] = remark
        dec = ep.run(Intent(IK_DECLINE, payload, P_FINISH, None, why=f"proven decline {proof.kind}"))
        return dec.kind == D_FINISHED

    def candidate_for(self, ep: Any, line: Line) -> Optional[str]:
        """The line's choice when it is orderable and its writes never failed, else the first orderable backup
        with no failed write, else one with a single failed write (PROT-1)."""
        cm = ep.cartman
        choice_ok = bool(line.choice) and ep.checker.verdict(ep, line, line.choice, ep.ledger.cap_c).tier != "X"
        if choice_ok and cm.failures.get(line.choice, 0) == 0:
            return line.choice
        options = ep.selector.orderable(ep, line) or self._dx_unknown_fit(ep, line)
        clean = [i for i in options if cm.failures.get(i, 0) == 0]
        if clean:
            return clean[0]
        if choice_ok and cm.writable(line.choice):
            return line.choice
        retry = [i for i in options if cm.writable(i)]
        return retry[0] if retry else None

    @staticmethod
    def _dx_unknown_fit(ep: Any, line: Line) -> list:
        """An accessory line with no row known to fit the chosen base: rows whose fit is unknown (codes unread, a
        pointer unresolved) are orderable, as DESIGN 5.14 vetoes only codes that miss the base (v2_d)."""
        b_idx, accs = ep.selector.roles(ep)
        base = ep.ledger.lines[b_idx].choice if b_idx is not None else None
        if line.idx not in accs or not base:
            return []
        return [i for i in ep.selector.orderable(ep, line, compat=False) if ep.selector.compatible(ep, i, base) is None]

    @staticmethod
    def time_for(ep: Any, writes: int) -> bool:
        """Room for `writes` non-finish sends and one finish at the running mean latency (>= 1 s each)."""
        lam = max(ep.plan.mean_env_s(), 1.0)
        t = ep.clock()
        return t + writes * lam < T_LAST_ENV_SEND and t + (writes + 1) * lam < T_LAST_FINISH_SEND

    def decide_fallback(self, ep: Any, exclude: tuple = ()) -> tuple:
        """(action, detail). The decision table of DESIGN 5.19, in order:
        1 order: the cart holds every line at plan quantity (then order_retry once; a complete cart is never
          declined); v5: over the cap, a known basket inside it first (fb:rewrite), else a read;
        2 write_then_order: every missing line has an orderable candidate, rem >= missing + 1 and time allows, and
          the plan is no multi-line plan over the cap (else fb:refit, the best completion inside the cap keeping
          the cart's lines; else no write, and the plan counts as impossible);
        3 decline_proof on E1/E2/E3;
        gather (v2_d): a line without a candidate (v5: or the dearest line of an over-cap plan) gets the
          decline-evidence steps while rem >= missing + 3 and time allows, then the table is re-read;
        npi (v5): the best basket inside the cap not provably invalid; then OVERCAP/NOFIT declines;
        4 decline_impossible: a complete cart is impossible within rem (too few steps or too little time, or a
          missing line still without such an item once nothing is left to gather);
        then (deviation) order_partial when declines are locked or blocked and the cart is not empty, and
        decline_last_resort when no line can be written (or writing failed) and nothing else can finish.
        No decline is chosen right after a refusal (G15)."""
        if ep.finished or ep.plan.rem() <= 0:
            return "fb:none", None
        lines = ep.ledger.lines
        expected = ep.cart.expected
        missing = [l for l in lines if not (l.choice and expected.get(l.choice) == l.qty)]
        gather = "fb:gather" not in exclude and ep.plan.rem() >= len(missing) + 3 and self.time_for(ep, len(missing) + 2)
        if lines and not missing:
            alt = None if "fb:rewrite" in exclude else self._v5d_rewrite(ep)
            it = self._v5d_read(ep, []) if alt == () and gather else None
            if alt or it:
                return ("fb:rewrite", alt) if alt else ("fb:gather", it)
            for action in ("fb:order", "fb:order_retry"):
                if action not in exclude:
                    return action, None
            return "fb:none", None
        writes: Optional[list] = []
        for line in missing:
            ident = self.candidate_for(ep, line)
            if ident is None:
                writes = None
                break
            writes.append((line.idx, ident))
        writes, over = self._dx_fit_writes(ep, missing, writes)
        room = ep.plan.rem() >= len(missing) + 1
        timely = self.time_for(ep, len(missing))
        if lines and writes is not None and not over and room and timely and "fb:write_then_order" not in exclude:
            return "fb:write_then_order", writes
        decline_open = not ep.events.decline_locked and not ep.events.refused_last
        proof = ep.prover.proof(ep) if decline_open and "fb:decline_proof" not in exclude else None
        if proof is not None and proof.kind in ("E1", "E2", "E3"):
            return "fb:decline_proof", None
        it = self._v5d_read(ep, [l for l in missing if self.candidate_for(ep, l) is None]) if (
            writes is None or over) and gather else None
        if it is not None:
            return "fb:gather", it
        held = {l.idx: l.choice for l in lines if l not in missing}
        npi = self._v5d_fit(ep, held, missing, False) if lines and room and timely and "fb:npi" not in exclude \
            else None
        if npi or proof is not None:
            return ("fb:npi", [(l.idx, npi[l.idx]) for l in missing]) if npi else ("fb:decline_proof", None)
        impossible = not lines or writes is None or over or not room or not timely
        if decline_open and impossible and "fb:decline_impossible" not in exclude:
            return "fb:decline_impossible", None
        if expected and "fb:order_partial" not in exclude:
            return "fb:order_partial", None
        if "fb:decline_last_resort" not in exclude and not ep.events.refused_last \
                and (impossible or "fb:write_then_order" in exclude):
            return "fb:decline_last_resort", None
        return "fb:none", None

    @staticmethod
    def _v5d_fit(ep: Any, held: dict, todo: list, eligible: bool = True) -> Optional[dict]:
        """DeclineProver.basket over writable eligible items, or any not provably invalid."""
        cm, pv = ep.cartman, ep.prover
        return pv.basket(ep, held, todo, {l.idx: [i for i in (ep.selector.orderable(ep, l, compat=False) if eligible
                                                               else [x for _, x in pv.orderable(ep, l)])
                                                  if cm.writable(i)] for l in todo})

    def _v5d_rewrite(self, ep: Any) -> Optional[tuple]:
        """Over the cap: writes [(idx, id)] to a known basket inside it if steps allow, else (); None: inside it."""
        lines, cap, cart = ep.ledger.lines, ep.ledger.cap_c, {i: q for i, q in ep.cart.expected.items() if q > 0}
        if cap is None or sum(q * ep.selector._price(ep, i) for i, q in cart.items()) <= cap:
            return None
        picks = self._v5d_fit(ep, {}, lines) or self._v5d_fit(ep, {}, lines, False) or {}
        writes = [(l.idx, picks[l.idx]) for l in lines if picks and cart.get(picks[l.idx]) != l.qty]
        return writes if picks and ep.plan.rem() > 2 * len(writes) and self.time_for(ep, 2 * len(writes)) else ()

    @staticmethod
    def _v5d_read(ep: Any, first: list) -> Optional[Intent]:
        """The next decline-evidence read: first, else the dearest line (a recovery read after a finish)."""
        for line in first or sorted(ep.ledger.lines, key=lambda l: -l.qty * ep.selector._price(ep, l.choice)):
            it = ep.prover.gather(ep, line)
            if it is not None:
                it.purpose = P_RECOVER if ep.budget.first_finish_step is not None else it.purpose
                return it
        return None

    def _v5d_apply(self, ep: Any, writes: list) -> bool:
        """write_lines, then removals of cart items no line holds (a rewrite, a swap)."""
        if not self.write_lines(ep, writes):
            return False
        keep = {l.choice for l in ep.ledger.lines}
        for ident in [i for i, q in ep.cart.expected.items() if q > 0 and i not in keep]:
            if not self.time_for(ep, 1) or ep.plan.rem() < 2:
                return False
            it = Intent(IK_CART_SET, {"item": ident, "quantity": 0, "held": ep.cart.expected[ident]}, P_COMMIT, None,
                        why=f"remove {ident}")
            dec = ep.run(it)
            ep.cartman.after(ep, it, dec)
            if dec.kind not in (D_CART, D_ACK, D_UNKNOWN):
                return False
        return True

    def _dx_fit_writes(self, ep: Any, missing: list, writes: Optional[list]) -> tuple:
        """(writes, over). Over the cap (a certain failure, DESIGN 5.15), the best completion inside the cap keeping
        the cart's lines replaces the writes (fb:refit); else over."""
        lines, cap = ep.ledger.lines, ep.ledger.cap_c
        if writes is None or len(lines) < 2 or cap is None:
            return writes, False
        todo = {l.idx for l in missing}
        plan = {l.idx: l.choice for l in lines if l.idx not in todo}
        plan.update(dict(writes))
        total = ep.selector._total(ep, plan)
        if total is None or total <= cap:
            return writes, False
        picks = self._v5d_fit(ep, {k: v for k, v in plan.items() if k not in todo}, missing)
        if picks:
            alt = [(l.idx, picks[l.idx]) for l in missing]
            ep.trace.decision("fallback", action="fb:refit", over_cap=total, cap=cap, writes=alt)
            return alt, False
        ep.trace.decision("fallback", action="fb:skip_over_cap", over_cap=total, cap=cap)
        return writes, True

    def fallback(self, ep: Any, reason: str) -> None:
        ep.phase = "FALLBACK"
        excluded: list = []
        first = True
        force_order = False
        write_rounds = 0
        flipped = False
        recovery_reads = gathers = repaired = 0
        for _ in range(self.FALLBACK_ROUNDS):
            if ep.finished or not ep.can_act():
                return
            action, detail = self.decide_fallback(ep, tuple(excluded))
            if force_order and ep.cart.expected and action not in ("fb:order", "fb:order_retry", "fb:order_partial",
                                                                   "fb:rewrite", "fb:gather"):
                action, detail = "fb:order_partial", None    # never decline right after a refusal
            force_order = False
            if action == "fb:none" and ep.events.refused_last and recovery_reads < 1:
                recovery_reads += 1                         # a decline may follow one non-finish step (G15)
                if self.recovery_read(ep):
                    continue
            ep.trace.decision("fallback" if first else "fallback_step", reason=reason, action=action,
                              rem=ep.plan.rem(), t=round(ep.clock(), 1))
            first = False
            if action == "fb:none":
                return
            if action == "fb:gather":
                gathers += 1
                dec = self._optional_read(ep, detail)
                if dec is None or dec.kind == D_BLOCKED or gathers >= 4:
                    excluded.append("fb:gather")
                if dec is not None:
                    ep.prover.after(ep, detail, dec)
                    ep.selector.select(ep)
                continue
            if action in ("fb:write_then_order", "fb:npi", "fb:rewrite"):
                write_rounds += 1
                if write_rounds >= self.WRITE_RETRIES or action != "fb:write_then_order":
                    excluded.append(action)
                if not self._v5d_apply(ep, detail):
                    continue
                action = "fb:order_after_writes"      # the ORDER in the same round (PROT-6)
            if action in ("fb:order", "fb:order_retry", "fb:order_partial", "fb:order_after_writes"):
                if action == "fb:order" and ep.plan.rem() >= 2 and ep.clock() < 245.0 and self._stale(ep):
                    read = self._optional_read(ep, Intent(IK_OPEN, {"ids": list(ep.cart.expected)[:1],
                                                                    "need": NEED_STOCK}, P_VERIFY, None,
                                                          why="fallback re-open"))
                    if read is not None and not ep.cart.expected:
                        continue                            # (environment down: the order keeps its own window)
                dec = self.order(ep, f"fallback {action}")
                if dec.kind == D_FINISHED:
                    return
                if dec.kind == D_REFUSED:
                    ep.events_machine.process(ep)
                    self.recover_inline(ep)             # swap a dead line or apply the budget when steps allow
                    force_order = True                  # the same (or rewritten) cart again; never a decline next
                    continue
                if dec.kind == D_EMPTY_CART:
                    ep.cart.expected = {}
                    continue
                if dec.kind in (D_WORLD_FAULT, D_EMPTY) and not flipped:
                    flipped = True                      # the writes may never have landed (PROT-4)
                    if not self.reread_cart(ep) and ep.cart.returns_cart is not True:
                        ep.binder.flip_cart_mode()
                        ep.cart.expected = {}
                    if not ep.cart.expected or {i: q for i, q in ep.cart.expected.items() if q > 0} \
                            != ep.cartman.plan(ep):
                        if "fb:write_then_order" in excluded:
                            excluded.remove("fb:write_then_order")
                        write_rounds = 0
                        continue
                if dec.kind == D_NONFINAL and self.complete_two_step(ep, dec):
                    return
                if dec.kind in (D_CART, D_STATUS, D_RECORDS, D_RECORD) and ep.binder.tool_for(ROLE_ORDER):
                    continue
                excluded.append(action)
                continue
            basis = {"fb:decline_proof": "fallback:proof", "fb:decline_impossible": "fallback:impossible",
                     "fb:decline_last_resort": "fallback:last_resort"}[action]
            self.settle_uncertain(ep)
            dec = ep.run(Intent(IK_DECLINE, {"basis": basis}, P_FINISH, None, why=f"fallback {action}"))
            if dec.kind == D_FINISHED:
                return
            if dec.kind == D_REFUSED:
                ep.events_machine.process(ep)          # declines are now locked: back to an order (PROT-7)
                self.recover_inline(ep)                # re-select under the new ledger, search when nothing fits
                if not all(line.choice for line in ep.ledger.lines):
                    ep.selector.select(ep)
                if "fb:write_then_order" in excluded:
                    excluded.remove("fb:write_then_order")
                write_rounds = 0
                continue
            if dec.kind in (D_CART, D_STATUS, D_RECORDS, D_RECORD, D_EMPTY_CART, D_UNKNOWN) and repaired < 3 \
                    and ep.binder.tool_for(ROLE_DECLINE) not in (None, dec.tool):
                repaired += 1                          # v7: the uncertain decline tool was re-paired: try again
                continue
            excluded.append(action)

    def recover_inline(self, ep: Any) -> None:
        """The recovery engine without the phase loop (the fallback may run after 240 s, where the planner
        refuses everything): its intents one by one, each checked against the clock and the steps the engine
        reserves for the writes and the order (the guard still applies), then the changed plan is written."""
        em = ep.events_machine
        for _ in range(EVT_ROUND_STEPS + 2):
            if ep.plan.rem() < 3 or not self.time_for(ep, 3):
                break
            it = em.next_intent(ep)
            if it is None:
                break
            dec = self._optional_read(ep, it)
            if dec is None:
                break
            em.after(ep, it, dec)
        if em.active:
            em.finish(ep)
        expected = {i: q for i, q in ep.cart.expected.items() if q > 0}
        if expected and ep.cartman.plan(ep) != expected:
            self.write_plan(ep)

    def order(self, ep: Any, why: str) -> Decoded:
        self.settle_uncertain(ep)
        return ep.run(Intent(IK_ORDER, {}, P_FINISH, None, why=why))

    def write_lines(self, ep: Any, detail: list) -> bool:
        """Fallback writes, the clock re-checked before each; failures are recorded against the item so the
        next round moves the line to a backup (PROT-1). True when every write was accepted."""
        for idx, ident in detail:
            if not self.time_for(ep, 1) or ep.plan.rem() < 2:
                return False
            line = ep.ledger.lines[idx]
            line.choice = ident
            it = Intent(IK_CART_SET, {"item": ident, "quantity": line.qty}, P_COMMIT, idx,
                        why=f"fallback write line {idx}")
            dec = ep.run(it)
            ep.cartman.after(ep, it, dec)
            if dec.kind not in (D_CART, D_ACK, D_UNKNOWN):
                return False
        lines = ep.ledger.lines
        return bool(lines) and all(l.choice and ep.cart.expected.get(l.choice) == l.qty for l in lines)

    def recovery_read(self, ep: Any) -> bool:
        """One harmless non-finish step after a refusal, so the next action may be a decline: a fresh search for
        a line without candidates (purpose RECOVER, its rows feed selection), else a cart or status read."""
        if ep.plan.rem() < 2 or not self.time_for(ep, 1):
            return False
        for line in ep.ledger.lines:
            if self.candidate_for(ep, line) is None and line.words:
                dec = ep.run(Intent(IK_SEARCH, {"query": line.words, "qkind": "q:relaxed"}, P_RECOVER, line.idx,
                                    why=f"recovery search for line {line.idx}"))
                if dec.kind != D_BLOCKED:
                    ep.selector.select(ep)
                    return True
        for kind in (IK_CART_VIEW, IK_STATUS):
            dec = ep.run(Intent(kind, {}, P_RECOVER, None, why="read before the next finish"))
            if dec.kind != D_BLOCKED:
                return True
        return False

    @staticmethod
    def _optional_read(ep: Any, intent: Intent) -> Optional[Decoded]:
        """Run a non-finish step of the fallback that the finish does not need (a re-open, a recovery read). An
        environment that stays down until the non-finish deadline (255 s) ends that step, not the episode: the
        finish has its own window up to T_LAST_FINISH_SEND (integration: such a read used to cancel the order)."""
        try:
            return ep.run(intent)
        except StopEpisode as stop:
            if stop.reason != "environment_unavailable":
                raise
            ep.trace.decision("fallback_step", action="skip_read", why="environment unavailable", intent=intent.kind,
                              t=round(ep.clock(), 1))
            return None

    def _stale(self, ep: Any) -> bool:
        ids = list(ep.cart.expected)
        if not ids:
            return False
        r = ep.catalog.get(ids[0])
        opened_u = max(r.opened_by.values()) if r is not None and r.opened_by else None
        return opened_u is None or ep.transport.n - opened_u > 3

OP_MAX_LLM = 6
OP_KIND_TRIES = 2
OP_TIME_S = 24.0
OP_CAP_S = 8.0
OP_MAX_TOOLS = 6
OP_LEDGER_CHARS = 4000         # about 1k tokens; halved outside the normal regime (observations dropped too)
OP_OBS_CHARS = 600
OP_ROLES = {IK_SEARCH: (ROLE_SEARCH, ROLE_FILTER), IK_FILTER: (ROLE_FILTER, ROLE_SEARCH), IK_OPEN: RECORD_ROLES,
            IK_CART_SET: (ROLE_CART_WRITE,), IK_ASK: (ROLE_MESSAGE,)}
OP_KEY_SLOT = (("ids", SLOT_IDS), ("item", SLOT_ID), ("items", SLOT_ITEMS), ("cont", SLOT_TOKEN),
               ("quantity", SLOT_QTY), ("ceiling", SLOT_CEILING), ("query", SLOT_QUERY), ("text", SLOT_TEXT),
               ("shelf", SLOT_SHELF), ("maker", SLOT_MAKER))
OP_ENUM_SLOT = {IK_SEARCH: SLOT_MODE, IK_FILTER: SLOT_MODE, IK_CART_SET: SLOT_CART_MODE, IK_ASK: SLOT_KEY}

def _op_fits(key: str, feat: Any) -> bool:
    """Can an argument of this type carry the payload key?"""
    if feat.is_enum or feat.type == "boolean":
        return False
    if key in ("ids", "items"):
        return feat.type == "array" or (key == "ids" and feat.is_string)
    if key in ("quantity", "ceiling"):
        return feat.type in ("integer", "numeric")
    return feat.is_string or (key == "cont" and feat.type == "integer")

def _op_same(key: str, want: Any, value: Any) -> bool:
    """Does a proposed value carry this payload value?"""
    if key == "ids":
        return any(isinstance(g, str) and g in want for g in (value if isinstance(value, list) else [value]))
    if key == "items":
        return isinstance(value, list) and any(isinstance(x, dict) for x in value)
    if key in ("quantity", "ceiling"):
        a, b = as_number(value), as_number(want)
        return a is not None and b is not None and abs(a - b) < 1e-9
    if key in ("cont", "item"):
        return value is not None and str(value) == str(want)
    return isinstance(value, str) and (norm_space(value) == norm_space(want) or token_overlap(value, want) >= 0.5)

def _op_hint(ep: Any, intent: Intent, literal: Any) -> float:
    """How well an enum literal names the asked line and key (structured message tools)."""
    lines = ep.ledger.lines
    line = lines[intent.line] if isinstance(intent.line, int) and 0 <= intent.line < len(lines) else None
    words = f"{line.words} {line.shelf_words} {line.shelf_label or ''}" if line else ""
    return token_overlap(literal, words) + token_overlap(literal, intent.payload.get("key") or "")

def _op_safe(ep: Any, kind: str, fn: Callable, *args: Any) -> Any:
    """Run an M8 hook: an internal error is logged and means "nothing to do"; control flow passes through."""
    try:
        return fn(*args)
    except (StopEpisode, GoFallback):
        raise
    except Exception as exc:
        ep.trace.decision(kind, error=clip_text(repr(exc), 160))
        return None

class Operator:
    def __init__(self):
        self.plans: dict = {}          # kind -> (tool, {arg: key | ("lit", v) | ("items", id key, qty key)}, strikes)
        self.tries: dict = {}
        self.requests = 0
        self.spent = 0.0
        self.completions = 0

    def map(self, ep: Any, intent: Intent, call: Optional[Call]) -> Optional[Call]:
        return _op_safe(ep, "operator", self._map, ep, intent, call)

    def _map(self, ep: Any, intent: Intent, call: Optional[Call]) -> Optional[Call]:
        kind = intent.kind
        roles = OP_ROLES.get(kind)
        if roles is None or (call is not None and call.canonical() in ep.binder.failed_calls):
            return None                            # G6 would block the same call again
        vals = self._values(ep, intent)
        tools = self._family(ep, kind, roles) if vals else []
        if not tools:
            return None
        plan = self.plans.pop(kind, None)
        tb = ep.binder.binding.tools.get(plan[0]) if plan else None
        if tb is not None and plan[0] in tools and tb.strikes <= plan[2]:
            got, _ = self._assemble(ep, intent, vals, plan[0], self._replay(ep, plan, vals))
            if got is not None:
                self.plans[kind] = plan
                self._log(ep, kind, got, "plan")
                return got
        if ep.phase == "FALLBACK" or not self._may_ask(ep, kind):
            return None
        pick = self._ask_tool(ep, self._message(ep, intent, vals, call), tools)
        got, binding = self._assemble(ep, intent, vals, pick[0], pick[1]) if pick else (None, "no usable tool call")
        if got is None:
            ep.trace.decision("operator", intent=kind, mapped=False, why=binding,
                              proposed=clip_text(dump_json(pick), 300) if pick else None)
            return None
        self.plans[kind] = (pick[0], binding, ep.binder.binding.tools[pick[0]].strikes)
        self._log(ep, kind, got, "llm")
        return got

    def _may_ask(self, ep: Any, kind: str) -> bool:
        if self.requests >= OP_MAX_LLM or self.spent >= OP_TIME_S or self.tries.get(kind, 0) >= OP_KIND_TRIES \
                or not ep.llm.allowed("OPERATOR"):
            return False
        self.requests += 1
        self.tries[kind] = self.tries.get(kind, 0) + 1
        return True

    def _ask_tool(self, ep: Any, user: dict, names: list) -> Optional[tuple]:
        t0 = monotonic()
        pick = ep.llm.tool_call("OPERATOR", [{"role": "user", "content": dump_json(user)}],
                                [self._schema(ep, n) for n in names], cap=OP_CAP_S)
        self.spent += monotonic() - t0
        return pick if pick and pick[0] in names else None

    def _values(self, ep: Any, intent: Intent) -> Optional[dict]:
        """The payload values the call must carry, in the world's units; None when there is nothing to map."""
        p, kind, v = intent.payload, intent.kind, {}
        if kind == IK_SEARCH:
            v.update(query=clamp_query(p.get("query", ""), ep.contract.token_cap), meaning=(SEM_TEXT,))
            if not v["query"].strip():
                return None
        elif kind == IK_FILTER:
            v.update({k: str(p[k]) for k in ("shelf", "maker") if p.get(k)})
            if p.get("ceiling") is not None:
                v["ceiling"] = ep.builder.money_out(p["ceiling"])
            if not v:
                return None
            v["meaning"] = (SEM_FILTER_SHELF if p.get("shelf") else SEM_FILTER_MAKER, SEM_FILTER_ANY)
        elif kind == IK_OPEN:
            v.update(ids=[i for i in p.get("ids") or [] if isinstance(i, str)], need=p.get("need", NEED_FULL))
            if not v["ids"]:
                return None
        elif kind == IK_CART_SET:
            item, target, cart = p.get("item"), int(p.get("quantity", 0) or 0), ep.cart
            if not isinstance(item, str):
                return None
            current = int(cart.expected.get(item, 0))
            want = {i: q for i, q in cart.expected.items() if i != item}
            if target > 0:
                want[item] = target
            v.update(items=list(want.items()), item=item)
            if cart.semantics == "add":
                step = min(target - current, cart.add_max) if cart.add_max else target - current
                v.update(quantity=step, meaning=(SEM_ADD,), new=current + step)
            else:
                v.update(quantity=target, meaning=(SEM_SET, SEM_ADD), new=target)
            if v["quantity"] <= 0:
                del v["quantity"]                  # removals: item lists only (never an add for a removal)
        elif kind == IK_ASK:
            v["text"] = str(p.get("text") or "").strip()
            if not v["text"]:
                return None
        if kind in (IK_SEARCH, IK_FILTER) and p.get("cont") is not None:
            v["cont"] = p["cont"]
        return v

    @staticmethod
    def _family(ep: Any, kind: str, roles: tuple) -> list:
        """Tools bound to the intent's roles, or structurally plausible for them, best first."""
        bnd = ep.binder.binding
        floor = 0.5 if kind in (IK_OPEN, IK_ASK) else 0.3
        ranked = []
        for i, (name, tb) in enumerate(bnd.tools.items()):
            if tb.strikes >= 2 or tb.role in FINISH_ROLES or name == bnd.finish_tool or tb.decline_votes \
                    or any(r in FINISH_ROLES for r in tb.vote_roles) or name in bnd.demoted.get(roles[0], ()) \
                    or (tb.role == ROLE_MESSAGE and kind != IK_ASK) \
                    or (tb.role not in roles and ep.binder._remark_only(name)):
                continue
            fit = 1.0 + tb.role_conf if tb.role in roles else max(tb.plaus.get(r, 0.0) for r in roles)
            if fit >= floor:
                ranked.append((-fit, i, name))
        return [name for _, _, name in sorted(ranked)][:OP_MAX_TOOLS]

    @staticmethod
    def _schema(ep: Any, name: str) -> dict:
        for t in ep.tools_raw:
            fn = t.get("function", t) if isinstance(t, dict) else None
            if isinstance(fn, dict) and fn.get(NAME_KEY) == name:
                return {"type": "function", "function": fn}
        return {"type": "function", "function": {NAME_KEY: name}}

    def _message(self, ep: Any, intent: Intent, vals: dict, call: Optional[Call]) -> dict:
        normal = ep.llm.regime() == "normal"
        shown = {k: ([{"item": i, "quantity": q} for i, q in x] if k == "items" else x[0] if k == "meaning" else x)
                 for k, x in vals.items() if k != "new"}
        if intent.kind == IK_ASK:
            lines = ep.ledger.lines
            if isinstance(intent.line, int) and 0 <= intent.line < len(lines):
                shown["line"] = lines[intent.line].shelf_label or lines[intent.line].words
            shown["key"] = intent.payload.get("key")
        if call is None:
            why = ep.builder.compile(intent)[1] or "no call could be built"
        else:
            tb = ep.binder.binding.tools.get(call.tool)
            weak = [a for a, _ in call.slots_used if tb is None or tb.slots.get(a, (None, 0))[1] < 2]
            why = f"{call.tool}: role confidence {tb.role_conf if tb else 0}, weak arguments {weak}"
        user = {"goal": clip_text(f"{ep.phase}/{intent.purpose}: {intent.why}", 200), "intent": intent.kind,
                "payload": shown, "why": clip_text(why, 160),
                "ledger": self._digest(ep, OP_LEDGER_CHARS if normal else OP_LEDGER_CHARS // 2)}
        if normal:
            user["recent"] = self._recent(ep)
        return user

    @staticmethod
    def _digest(ep: Any, cap: int) -> dict:
        d = {"lines": [{"line": l.idx, "words": l.words, "units": l.qty, "label": l.shelf_label, "choice": l.choice}
                       for l in ep.ledger.lines],
             "cap_c": ep.ledger.cap_c, "cart": dict(ep.cart.expected), "steps": f"{ep.transport.n}/{ep.budget.B}",
             "msgs": f"{ep.budget.msgs_used}/{ep.budget.M}"}
        while d["lines"] and len(dump_json(d)) > cap:
            d["lines"].pop()
        return d

    @staticmethod
    def _recent(ep: Any) -> list:
        out = []
        for e in reversed(ep.trace.entries):
            if isinstance(e, dict) and e.get("role") == "assistant" and len(out) < 2:
                fn = (e.get("tool_call") or {}).get("function") or {}
                obs = (e.get("environment_result") or {}).get(OBS_KEY)
                out.append({"tname": fn.get(NAME_KEY), "obs": clip_text(dump_json(obs), OP_OBS_CHARS)})
        return out[::-1]

    @staticmethod
    def _replay(ep: Any, plan: tuple, vals: dict) -> dict:
        """A remembered mapping as a proposal carrying this intent's values."""
        feats, prop = ep.binder.features[plan[0]].args, {}
        for arg, b in plan[1].items():
            if isinstance(b, tuple):
                prop[arg] = b[1] if b[0] == "lit" else [{b[1]: i, **({b[2]: q} if b[2] else {})}
                                                        for i, q in vals.get("items") or []]
            elif b in vals and arg in feats:
                prop[arg] = vals[b][0] if b == "ids" and feats[arg].type != "array" else vals[b]
        return prop

    def _assemble(self, ep: Any, intent: Intent, vals: dict, tool: str, prop: Any) -> tuple:
        """(Call, binding) with the payload injected, or (None, reason)."""
        binder, kind = ep.binder, intent.kind
        tf, tb = binder.features.get(tool), binder.binding.tools.get(tool)
        if tf is None or tb is None or not isinstance(prop, dict):
            return None, "unknown tool or arguments"
        feats = tf.args
        if kind == IK_CART_SET:
            listy = any(a.type == "array" for a in feats.values())
            if not listy and "quantity" not in vals:
                return None, "removals are not mapped"
            keys = ["items"] if listy else ["item", "quantity"]
            counts = any(a.type in ("integer", "numeric") for a in feats.values())
            need = keys if listy or counts or vals["quantity"] != 1 else ["item"]
        else:
            keys = [k for k, _ in OP_KEY_SLOT if k in vals]
            need = keys if kind != IK_ASK else ["text"] if any(a.is_string for a in feats.values()) else []
        assign: dict = {}
        for arg, value in prop.items():
            feat = feats.get(arg)
            key = next((k for k in keys if k not in assign.values() and _op_fits(k, feat)
                        and _op_same(k, vals[k], value)), None) if feat is not None else None
            if key is not None:
                assign[arg] = key
        for key in need:
            if key not in assign.values():
                arg = self._home(tb, feats, key, prop, assign)
                if arg is None:
                    return None, f"no argument carries {key}"
                assign[arg] = key
        args, used, binding, ids, slot_of = {}, [], {}, [], dict(OP_KEY_SLOT)
        for arg, key in assign.items():
            feat, value, slot = feats[arg], vals[key], slot_of[key]
            if key == "ids":
                if feat.type == "array":
                    value = list(value)[:max(1, ep.contract_learner.batch_limit(feat.max_items))]
                else:
                    value, slot = value[0], SLOT_ID
                ids = value if isinstance(value, list) else [value]
            elif key == "items":
                pair = self._item_keys(feat, prop.get(arg), vals)
                if pair is None:
                    return None, "item keys unknown"
                value = [{pair[0]: i, pair[1]: q} if pair[1] else {pair[0]: i} for i, q in value]
                binding[arg] = ("items",) + pair
            elif key == "cont":
                label = tb.slots.get(arg, (None, 0))[0]
                slot = label if label in PAGING_SLOTS else (SLOT_TOKEN if feat.is_string else SLOT_PAGE)
            else:
                value = ep.builder.coerce(tool, arg, value)
            args[arg] = value
            used.append((arg, slot))
            binding.setdefault(arg, key)
        for arg, feat in feats.items():
            if arg in args:
                continue
            if feat.is_enum:
                value = self._literal(ep, intent, vals, tool, arg, feat, prop.get(arg))
                used += [(arg, OP_ENUM_SLOT.get(kind, SLOT_OTHER))] if value is not None else []
            elif feat.type == "boolean" and isinstance(prop.get(arg), bool):
                value = prop[arg]
            elif feat.required and feat.type == "integer" and kind in (IK_SEARCH, IK_FILTER):
                value = int(feat.minimum) if feat.minimum is not None else 1    # a first-page default
            else:
                value = None
            if value is None:
                if feat.required:
                    return None, f"required argument {arg} has no value"
                continue
            args[arg] = value
            binding[arg] = ("lit", value)
        slots = dict(used)
        for arg in list(args):
            fixed, ok, why = check_value(feats[arg], args[arg], slots.get(arg))
            if not ok:
                return None, f"{arg}: {why}"
            args[arg] = fixed
        role = {IK_SEARCH: ROLE_SEARCH, IK_FILTER: ROLE_FILTER, IK_CART_SET: ROLE_CART_WRITE, IK_ASK: ROLE_MESSAGE}.get(
            kind, ROLE_BATCH if SLOT_IDS in slots.values() else (tb.role if tb.role in RECORD_ROLES else ROLE_VIEW))
        out = Call(tool=tool, args=args, role=role, slots_used=used, path="op", intent_kind=kind,
                   ids=list(ids) if kind == IK_OPEN else ([vals["item"]] if kind == IK_CART_SET else []))
        if kind == IK_CART_SET:
            out.write = (vals["item"], max(0, vals["new"]))
        return ep.builder._finish_call(out), binding

    @staticmethod
    def _home(tb: Any, feats: dict, key: str, prop: dict, assign: dict) -> Optional[str]:
        """The argument for a payload key the proposal did not place: the binder's slot label first, then one the
        proposal named, then a required one; a tie is ambiguous (None)."""
        want = PAGING_SLOTS if key == "cont" else (SLOT_IDS, SLOT_ID) if key == "ids" else (dict(OP_KEY_SLOT)[key],)
        ranked = sorted((tb.slots.get(a, (None, 0))[0] not in want, a not in prop, not f.required, i, a)
                        for i, (a, f) in enumerate(feats.items()) if a not in assign and _op_fits(key, f))
        if not ranked or (len(ranked) > 1 and ranked[0][:3] == ranked[1][:3]):
            return None
        return ranked[0][-1]

    @staticmethod
    def _item_keys(feat: Any, proposed: Any, vals: dict) -> Optional[tuple]:
        """(id key, quantity key) of item-list objects: declared item properties, else read from the proposal."""
        id_key, qty_key = item_list_keys(feat)
        ids = {i for i, _ in vals["items"]}
        for x in (proposed if id_key is None and isinstance(proposed, list) else []):
            if isinstance(x, dict):
                id_key = next((k for k, v in x.items() if isinstance(v, str) and v in ids), id_key)
                qty_key = qty_key or next((k for k, v in x.items() if isinstance(v, int)
                                           and not isinstance(v, bool)), None)
        return (id_key, qty_key) if id_key else None

    @staticmethod
    def _literal(ep: Any, intent: Intent, vals: dict, tool: str, arg: str, feat: Any, proposed: Any) -> Any:
        """The binder's literal for the intent's meaning, else the proposed literal when valid; never one meaning order
        or decline; for messages the literal naming the line or key wins over one naming neither."""
        table = ep.binder.binding.tools[tool].enum_map.get(arg, {})
        bad = {lit for lit, (m, _) in table.items() if m in (SEM_ORDER, SEM_DECLINE)}
        allowed = [e for e in feat.enum if str(e) not in bad]
        for meaning in vals.get("meaning") or ():
            lit = ep.binder.literal_for(tool, arg, meaning)
            hit = next((e for e in allowed if str(e) == lit), None)
            if hit is not None:
                return hit
        pick = None
        if proposed is not None:
            fixed, ok, _ = check_value(feat, proposed, None)
            pick = fixed if ok and str(fixed) not in bad else None
        if intent.kind == IK_ASK and allowed:
            best = max(allowed, key=lambda e: (_op_hint(ep, intent, e), -allowed.index(e)))
            if _op_hint(ep, intent, best) > 0 and (pick is None or _op_hint(ep, intent, pick) == 0):
                return best
        return pick

    @staticmethod
    def _log(ep: Any, kind: str, call: Call, source: str) -> None:
        ep.trace.decision("operator", intent=kind, source=source, tool=call.tool,
                          args=clip_text(dump_json(call.args), 300), slots=[s for _, s in call.slots_used])

    def complete_finish(self, ep: Any, dec: Decoded) -> Optional[Call]:
        return _op_safe(ep, "operator", self._complete, ep, dec)

    def _complete(self, ep: Any, dec: Decoded) -> Optional[Call]:
        inner = dec.inner if isinstance(dec.inner, dict) else {}
        handle = next((v for v in inner.values() if token_like(v)), None)
        names = self._completers(ep) if handle is not None and self.completions < 2 else []
        if not names or not self._may_ask(ep, "complete"):
            return None
        self.completions += 1
        user = {"goal": "complete the purchase: the order call answered with a confirmation handle instead of "
                        "finishing", "intent": IK_ORDER, "payload": {"handle": handle, "answer": clip_text(
                            dump_json(inner), 400)}, "ledger": self._digest(ep, OP_LEDGER_CHARS // 2)}
        pick = self._ask_tool(ep, user, names)
        args = self._completion_args(ep, pick[0], pick[1], handle) if pick else None
        if args is None:
            ep.trace.decision("operator", intent="complete", mapped=False,
                              proposed=clip_text(dump_json(pick), 300) if pick else None)
            return None
        call = Call(tool=pick[0], args=args, role=ep.binder.binding.tools[pick[0]].role, path="op")
        self._log(ep, "complete", call, "llm")
        return call

    @staticmethod
    def _completers(ep: Any) -> list:
        """Never a decline, message or decline-shaped tool, nor one confidently bound to a non-finish role; a tool
        other than the order tool must take a string (the handle)."""
        bnd = ep.binder.binding
        decline = (bnd.by_role.get(ROLE_DECLINE) or [None])[0] if bnd.finish_kind != "enum" else None
        out = []
        for name, tb in bnd.tools.items():
            if name == decline or tb.role in (ROLE_DECLINE, ROLE_MESSAGE) or tb.decline_votes \
                    or ROLE_DECLINE in tb.vote_roles or ep.binder._remark_only(name):
                continue
            if tb.role in (ROLE_ORDER, ROLE_FINISH) or (
                    (tb.role == ROLE_OTHER or tb.role_conf < 2)
                    and any(a.is_string for a in ep.binder.features[name].args.values())):
                out.append(name)
        return out[:OP_MAX_TOOLS]

    @staticmethod
    def _completion_args(ep: Any, tool: str, prop: Any, handle: str) -> Optional[dict]:
        bnd = ep.binder.binding
        tb, feats = bnd.tools[tool], ep.binder.features[tool].args
        if not isinstance(prop, dict):
            return None
        args, carried = {}, False
        req_strings = [a for a, f in feats.items() if f.is_string and f.required]
        for arg, feat in feats.items():
            value = prop.get(arg)
            if feat.is_enum:
                bad = {lit for lit, (m, _) in tb.enum_map.get(arg, {}).items() if m == SEM_DECLINE}
                if tool == bnd.finish_tool and arg == bnd.finish_arg:
                    bad.add(str(bnd.decline_literal))
                    value = bnd.order_literal if value is None else value
                if value is None:
                    if feat.required:
                        return None
                    continue
                fixed, ok, _ = check_value(feat, value, None)
                if not ok or str(fixed) in bad:
                    return None                    # a decline literal is never accepted
                args[arg] = fixed
            elif feat.is_string and (value == handle or (feat.required and len(req_strings) == 1)):
                args[arg], carried = handle, True
            elif feat.type == "boolean" and isinstance(value, bool):
                args[arg] = value
            elif feat.required:
                return None
        return args if carried or tb.role in (ROLE_ORDER, ROLE_FINISH) else None

RV_CONFIRM_TEXT = "Everything you asked for is in the cart now. Shall I go ahead and place the order?"
RV_NOT_CAP_ROUTES = ("budget", "story", "restatement", "hint", "procedure", "decline_rule", "stock")

def _rv_route(vote: dict, span: str) -> Optional[str]:
    """How a PARSE vote routed the sentence holding span (None when no routed sentence holds it)."""
    target = norm_span(span)
    return next((route for sent, route in vote.get("_routes") or [] if target and (target in sent or sent in target)),
                None)

def _rv_stems(text: Any) -> set:
    return {t[:4] for t in content_tokens(text) if len(t) >= 3}

def _rv_covers(a: Any, b: Any) -> bool:
    """At least half of a's word stems occur in b."""
    sa = _rv_stems(a)
    return bool(sa) and 2 * len(sa & _rv_stems(b)) >= len(sa)

def _rv_qty_near(span: str, num: float, line: Any) -> bool:
    """The line's words follow the number within 4 words and before any other number ("2 skirts and 3 belts"):
    a quantity of that line, not a cap, a budget or another line's quantity."""
    stems = _rv_stems(f"{line.words} {line.shelf_words}")
    for value, _, _, end in numbers_in(span):
        after = text_tokens(span[end:])[:4]
        cut = next((i for i, t in enumerate(after) if t[0].isdigit()), len(after))
        if abs(float(value) - num) < 1e-9 and stems & {t[:4] for t in after[:cut]}:
            return True
    return False

def _rv_proc_valid(answer: dict, n: int) -> tuple:
    steps, seen = [], set()
    for s in answer.get("steps") or []:
        key = (s.get("proc_i"), s.get("action"))
        if isinstance(key[0], int) and 0 <= key[0] < n and key not in seen:
            seen.add(key)
            steps.append(s)
    return True, dict(answer, steps=steps), ""

class Reviewer:
    def __init__(self):
        self.reviewed = False
        self.steps: list = []          # scheduled procedure steps {i, when, action, n, span}
        self.gate_done: set = set()
        self.noted = False

    def review(self, ep: Any) -> list:
        if self.reviewed:
            return []
        self.reviewed = True
        why = self._skip(ep)
        ans = None if why else ep.llm.ask("REVIEW", self._payload(ep), validator=lambda a: self._valid(ep, a))
        if ans is None:
            ep.trace.decision("review", skipped=why or "no answer")
            return []
        applied, log, reselect = [], [], False
        for issue in ans.get("issues") or []:
            try:
                act, note, again = self._act(ep, issue)
            except (StopEpisode, GoFallback):
                raise
            except Exception as exc:            # one bad issue never stops the commit
                act, note, again = None, f"error {exc!r}"[:80], False
            log.append({"kind": issue.get("kind"), "line": issue.get("line"), "val": issue.get("val"),
                        "span": clip_text(issue.get("request_span"), 120), "acted": act, "note": note})
            if act:
                applied.append(act)
                reselect = reselect or again
        if applied:
            ep.ledger.version += 1
            ep.plan.compute("review")
            if reselect:
                ep.selector.select(ep)
        ep.trace.decision("review", clean=ans.get("clean"), issues=log, applied=applied)
        return applied

    @staticmethod
    def _skip(ep: Any) -> str:
        if not ep.ledger.lines:
            return "no lines"
        if ep.cart.writes or ep.cart.expected:
            return "after the first cart write"
        if ep.clock() >= T_OPTIONAL_LLM_STOP:
            return "time"
        regime = ep.llm.regime()
        return "" if regime == "normal" and ep.llm.allowed("REVIEW") else f"regime {regime}"

    @staticmethod
    def _payload(ep: Any) -> dict:
        led = ep.ledger
        ledger = {"lines": [{"line": l.idx, "words": l.words, "shelf_words": l.shelf_words, "units": l.qty,
                             "line_role": l.role} for l in led.lines],
                  "constraints": [{"line": l.idx, "attr": c.attr, "op": c.op, "vals": c.vals, "span": c.span}
                                  for l in led.lines for c in l.constraints if c.active],
                  "cap": {"cents": led.cap_c, "span": led.cap_span},
                  "budgets": {"messages": ep.budget.M, "steps": ep.budget.B},
                  "rules": {"objective": led.objective, "stock_rule": led.in_stock, "decline_rule": led.decline_rule,
                            "procedures": [p.get("span") for p in led.procedures if isinstance(p, dict)],
                            "refs": [{"line": r.line, "attr": r.attr, "rel": r.relation, "span": r.span}
                                     for r in led.refs]}}
        return {"request": ep.query, "ledger": ledger,
                "cart": [{"line": l.idx, "rid": l.choice, "n": l.qty} for l in led.lines if l.choice]}

    @staticmethod
    def _valid(ep: Any, answer: dict) -> tuple:
        """Lenient: issues whose request_span is not verbatim are dropped."""
        return True, dict(answer, issues=[i for i in answer.get("issues") or []
                                          if ep.parser.span_ok(i.get("request_span"))]), ""

    def _act(self, ep: Any, issue: dict) -> tuple:
        """(action label | None, note, re-select)."""
        kind, span, lines = issue.get("kind"), issue.get("request_span") or "", ep.ledger.lines
        idx = issue.get("line")
        line = lines[idx] if isinstance(idx, int) and 0 <= idx < len(lines) else (lines[0] if len(lines) == 1 else None)
        val = str(issue.get("val") or "").strip()
        if kind == "quantity":
            return self._quantity(ep, line, span, val)
        if kind == "cap":
            return self._cap(ep, span, val)
        if kind == "missing_constraint":
            return self._constraint(ep, line, span, val, issue)
        if kind == "missing_line":
            return self._line(ep, span, val)
        return None, "logged only", False

    @staticmethod
    def _over_cap(ep: Any) -> bool:
        total, cap = ep.selector.basket_total(ep), ep.ledger.cap_c
        return total is not None and cap is not None and total > cap

    def _quantity(self, ep: Any, line: Any, span: str, val: str) -> tuple:
        """The number is in the span, next to the line's words, 1..99 and differs from the line's quantity."""
        n = as_number(val)
        if line is None or n is None or n != int(n) or not 1 <= n <= 99 or not number_in_span(n, span):
            return None, "untestable", False
        if int(n) == line.qty or not _rv_qty_near(span, n, line):
            return None, "same quantity, or the number is not next to the line's words", False
        old, line.qty = line.qty, int(n)
        return f"quantity:{line.idx}:{old}->{line.qty}", "", self._over_cap(ep)

    def _cap(self, ep: Any, span: str, val: str) -> tuple:
        """The number is in the span and differs; no PARSE vote routed the span as another kind; a budget number, or
        a cap both votes agreed on, changes only when a vote routed the span as the cap."""
        led, n = ep.ledger, as_number(val)
        cents = cap_to_cents(n, span) if n is not None and number_in_span(n, span) else None
        if cents is None or cents <= 0 or cents == led.cap_c:
            return None, "untestable or same cap", False
        routes = [_rv_route(v, span) for v in ep.parse_votes]
        if any(r in RV_NOT_CAP_ROUTES for r in routes):
            return None, "a parse vote routed the span as another kind", False
        if "cap" not in routes and (n in (led.stated_msgs, led.stated_steps, ep.budget.B, ep.budget.M)
                                    or (led.cap_c is not None and led.source == "votes")):
            return None, "a budget number, or a cap both votes agreed on", False
        old, led.cap_c, led.cap_span = led.cap_c, cents, span
        return f"cap:{old}->{cents}", "", self._over_cap(ep)

    def _constraint(self, ep: Any, line: Any, span: str, val: str, issue: dict) -> tuple:
        """The value is literal in the span, >= 1 PARSE vote routed it as a requirement and the line has no constraint
        on that attribute or value: Constraint(source="review"); the certificate is invalidated."""
        attr = canon_key(issue.get("attr"))
        if line is None or not val or not attr or norm_span(val) not in norm_span(span):
            return None, "untestable (no line, attribute, or value literal in the span)", False
        if "requirement" not in [_rv_route(v, span) for v in ep.parse_votes]:
            return None, "no parse vote routed the span as a requirement", False
        if any(c.active and (c.attr == attr or canon_key(val) in [canon_key(x) for x in c.vals])
               for c in line.constraints):
            return None, "already in the ledger", False
        cid = f"rv{line.idx}.{len(line.constraints)}"
        line.constraints.append(Constraint(cid=cid, attr=attr, op="ne" if issue.get("op") == "ne" else "eq",
                                           vals=[val], source="review", span=span))
        ep.coverage.invalidate(ep, line, "review constraint")
        return (f"constraint:{line.idx}:{attr}",) + self._recheck(ep, line, cid)

    @staticmethod
    def _recheck(ep: Any, line: Any, cid: str) -> tuple:
        """(note, re-select): a CONTRADICTED pick is re-selected; an UNCONFIRMED one is kept unless a CONFIRMED
        orderable alternative exists."""
        if not line.choice:
            return "", False
        v = ep.checker.verdict(ep, line, line.choice, ep.selector.line_budget(ep, line))
        state = (v.per_constraint.get(cid) or ("UNCONFIRMED",))[0]
        if v.tier == "X" or state in ("CONFIRMED", "CONTRADICTED"):
            return state.lower(), v.tier == "X" or state == "CONTRADICTED"
        for ident in ep.selector.orderable(ep, line):
            alt = line.verdicts.get(ident)
            if ident != line.choice and alt is not None and (alt.per_constraint.get(cid) or ("",))[0] == "CONFIRMED":
                old, line.choice = line.choice, ident
                return f"swapped {old} for {ident}", False
        return "kept unconfirmed", False

    def _line(self, ep: Any, span: str, val: str) -> tuple:
        """One PARSE vote had the line and no ledger line covers it: added, scouted through the planner-gated SCOUT
        drive and selected; removed again when it gets no candidate."""
        led = ep.ledger
        cand = next((ln for vote in ep.parse_votes for ln in vote.get("lines") or []
                     if _rv_covers(ln.get("words"), f"{val} {span}")
                     and not any(_rv_covers(ln.get("words"), f"{l.words} {l.shelf_words}") for l in led.lines)), None)
        if cand is None:
            return None, "no parse vote had that line", False
        idx, qty = len(led.lines), cand.get("qty_n")
        line = Line(idx=idx, words=cand["words"], shelf_words=cand.get("shelf_words") or "",
                    qty=int(qty) if isinstance(qty, (int, float)) and 1 <= qty <= 99 else 1,
                    role=cand.get("line_role") or "independent", cheapest=led.objective == "cheapest")
        for j, rq in enumerate(cand.get("reqs") or []):
            bound = {"num": rq["num"], "unit": rq.get("unit"), "raw": rq["span"]} \
                if rq["op"] in ("ge", "le", "between") and rq.get("num") is not None else None
            line.constraints.append(Constraint(cid=f"c{idx}.{j}", attr=canon_key(rq["attr"]), op=rq["op"],
                                               vals=list(rq.get("vals") or []), bound=bound, span=rq["span"]))
        led.lines.append(line)
        ep.plan.compute("review line")
        phase = ep.phase
        ep.orchestrator.drive(ep.scout, "SCOUT")
        ep.phase = phase
        ep.selector.select(ep)
        if line.choice:
            return f"line:{idx}:{line.words}", "", False
        if led.lines and led.lines[-1] is line:
            led.lines.pop()
        ep.plan.compute("review line removed")
        return None, "no candidate for the added line: removed", False

    def plan_procedures(self, ep: Any) -> None:
        _op_safe(ep, "proc", self._plan_procedures, ep)

    def _plan_procedures(self, ep: Any) -> None:
        procs = [p for p in ep.ledger.procedures or [] if isinstance(p, dict)]
        if not procs:
            return
        steps, source = None, "substitute"
        if ep.llm.allowed("PROC"):
            payload = {"request": ep.query, "procedures": [{"proc_i": i, "text": p.get("span") or p.get("text") or ""}
                                                           for i, p in enumerate(procs)],
                       "tools": [{"tname": t["tname"], "about": t["about"]} for t in ep.binder.tool_digest()]}
            ans = ep.llm.ask("PROC", payload, validator=lambda a: _rv_proc_valid(a, len(procs)))
            if ans is not None:
                steps, source = ans["steps"], "oracle"
        if steps is None:
            labels = {"view n": "view_count", "view count": "view_count", "confirm before order": "message"}
            steps = [{"proc_i": i, "when": "before_order", "action": labels[canon_key(p.get("kind"))]}
                     for i, p in enumerate(procs) if canon_key(p.get("kind")) in labels]
        self.steps, dropped = [], []
        for s in steps:
            p = procs[s["proc_i"]]
            n, span = p.get("n"), p.get("span") or ""
            n = int(n) if isinstance(n, (int, float)) and n == int(n) and 1 <= n <= 20 and number_in_span(n, span) \
                else None
            if (s["action"] == "view_count" and n is None) or s["action"] == "other" or (
                    s["action"] == "message" and s.get("when") == "before_first_write"):
                dropped.append([s.get("proc_i"), s.get("when"), s.get("action")])
            else:
                self.steps.append({"i": s["proc_i"], "when": s.get("when") or "any", "action": s["action"], "n": n,
                                   "span": clip_text(span, 160)})
        ep.trace.decision("proc", source=source, procedures=len(procs), steps=self.steps, logged_only=dropped)

    @staticmethod
    def _viewed(ep: Any) -> int:
        return len({i for line in ep.ledger.lines for i in list(line.opened) + list(line.table)
                    if i in ep.catalog and ep.catalog[i].opened_by and not ep.catalog[i].chain_page})

    def reserved_opens(self, ep: Any) -> int:
        if not self.steps:
            return 0
        k, total = max(1, ep.plan.batch_k()), 0
        for j, s in enumerate(self.steps):
            if s["action"] == "view_count":
                total += ceil_div(max(0, s["n"] - self._viewed(ep)), k)
            elif s["action"] == "open" and ("open", j) not in self.gate_done:
                total += 1
        return total

    def reserved_messages(self, ep: Any) -> int:
        due = any(s["action"] == "message" for s in self.steps)
        return 1 if due and norm_space(RV_CONFIRM_TEXT) not in ep.guard.sent_messages else 0

    def procedure_gate(self, ep: Any) -> Optional[Intent]:
        return _op_safe(ep, "procedure", self._gate, ep)

    def _gate(self, ep: Any) -> Optional[Intent]:
        self._note(ep)
        chain = ep.resolver.chain_gate(ep)          # M4 (Q7): a located chain page that a stopped walk never opened
        if chain is not None:
            return chain
        k = max(1, ep.plan.batch_k())
        pages = [r.id for r in ep.catalog.values() if r.chain_page and not r.opened_by
                 and ("page", r.id) not in self.gate_done][:k]
        if pages:
            self.gate_done.update(("page", i) for i in pages)
            return Intent(IK_OPEN, {"ids": pages, "need": NEED_NOTES, "chain_page": True, "procedure": True}, P_VERIFY,
                          None, why="procedure gate: open the missing chain pages")
        if self.reserved_messages(ep) and ("msg",) not in self.gate_done:
            self.gate_done.add(("msg",))
            return Intent(IK_ASK, {"key": None, "text": RV_CONFIRM_TEXT, "procedure": True}, P_VERIFY, None,
                          why="procedure gate: the message the request asks for before ordering")
        for j, s in enumerate(self.steps):
            if s["action"] == "view_count" and self._viewed(ep) < s["n"]:
                ids = []
                for line in ep.ledger.lines:
                    ids += [i for i in ep.selector.orderable(ep, line) if not ep.catalog[i].opened_by and i not in ids]
                ids = ids[:min(k, s["n"] - self._viewed(ep))]
                if ids and ("view", tuple(ids)) not in self.gate_done:
                    self.gate_done.add(("view", tuple(ids)))
                    return Intent(IK_OPEN, {"ids": ids, "need": NEED_ATTRS, "procedure": True}, P_VERIFY, None,
                                  why=f"procedure gate: view at least {s['n']} listings")
            if s["action"] == "open" and ("open", j) not in self.gate_done and ep.cart.expected:
                self.gate_done.add(("open", j))
                return Intent(IK_OPEN, {"ids": list(ep.cart.expected)[:k], "need": NEED_FULL, "procedure": True},
                              P_VERIFY, None, why="procedure gate: open the listings before ordering")
        return None

    def _note(self, ep: Any) -> None:
        """The always-on checks that allow no action at checkout, logged once."""
        refs, b = [r.rid for r in ep.ledger.refs if r.status != "resolved"], ep.budget
        if not self.noted and (refs or b.msgs_used < b.M):
            ep.trace.decision("procedure", unresolved_refs=refs, msgs=f"{b.msgs_used}/{b.M}",
                              why="advisory: references unresolved or questions unspent at checkout")
        self.noted = True

    @staticmethod
    def remark_ok(ep: Any, text: Any) -> bool:
        if {w.strip(".,;:!?()[]\"'") for w in str(text or "").split()} & set(ep.catalog):
            return False
        low = norm_space(text)
        return not any(len(r.title) >= 12 and norm_space(r.title) in low for r in ep.catalog.values())

SETUP_HARD_LIMIT = 35.0
SETUP_VOTE_DEADLINE = 30.0
ARBITER_CAP = 20.0
TERMINAL_STOPS = ("quarantined", "key_exhausted", "finished", "step_budget")

class Episode:
    """One task: the typed ledger, the learned world model and every engine. Engines read and plan; only
    Executor.run (ep.run) touches the environment."""

    def __init__(self, problem_data: Any, env_client: Any = None, llm_client: Any = None,
                 sleep: Optional[Callable[[float], None]] = None):
        self.t0 = monotonic()
        data = problem_data if isinstance(problem_data, dict) else {}
        env = data.get("environment") or {}
        pv = env.get("policy_view") or {}
        self.binding_env = dict(env.get("binding") or {})
        self.query = str(pv.get("query") or data.get("query") or "")
        self.opening = pv.get("opening")
        self.tools_raw = list(pv.get("tools") or [])
        self.max_steps = pv.get("max_steps")
        self.already_finished = bool(pv.get(FINISHED_KEY))
        self.phase = "SETUP"
        self.finished = False
        self.finish_kind: Optional[str] = None
        self.finish_tool: Optional[str] = None
        self.stop_reason: Optional[str] = None
        self.absent_ids: set = set()              # ids a batch view reported missing (G7)
        self.parse_votes: list = []               # validated PARSE votes with their _routes (REVIEW, M8)
        self.cap_unit_checked = False             # the cap-unit check ran (check_cap_unit)
        self.budget = Budget(t0=self.t0)
        self.trace = Trace(self.clock, self.diag)
        self.catalog: dict = {}
        self.ledger = Ledger()
        self.cart = Cart()
        self.events = Events()
        self.contract = Contract()
        self.contract_learner = ContractLearner(self.contract)
        self.queries = QueryBook()
        self.fields = FieldModel()
        self.decoder = Decoder(self.fields)
        self.binder = Binder(self)
        self.builder = CallBuilder(self)
        self.guard = Guard(self)
        self.llm = Llm(self, client=llm_client)
        self.plan = Planner(self)
        self.transport = Transport(self.binding_env, limit=lambda: self.budget.B, clock=self.clock,
                                   calls_needed=self.plan.need_finish, client=env_client,
                                   sleep=sleep or time.sleep, log=self.trace.decision)
        self.executor = Executor(self)
        self.parser = RequestParser(self)
        self.scout = Scout(self)
        self.feasibility = Feasibility(self)
        self.elicitor = Elicitor(self)
        self.resolver = Resolver(self)
        self.coverage = Coverage(self)
        self.checker = Checker()
        self.selector = Selector()
        self.cartman = CartManager(self)
        self.events_machine = EventMachine()
        self.prover = DeclineProver()
        self.finisher = Finisher()
        self.operator = Operator()
        self.reviewer = Reviewer()
        self.orchestrator = Orchestrator(self)

    def clock(self) -> float:
        return monotonic() - self.t0

    def diag(self) -> dict:
        transport = getattr(self, "transport", None)
        return {"u": transport.n if transport is not None else 0, "B": self.budget.B,
                "msgs": f"{self.budget.msgs_used}/{self.budget.M}", "t": round(self.clock(), 2),
                "usd": round(self.budget.usd, 5), "ledger_v": self.ledger.version}

    def run(self, intent: Intent) -> Decoded:
        return self.executor.run(intent)

    def can_act(self) -> bool:
        """A finish can still be attempted: not finished or stopped, a step left, inside the finish window
        (T_LAST_FINISH_SEND; non-finish sends stop at T_LAST_ENV_SEND through the guard's TIME gate)."""
        return (not self.finished and self.stop_reason not in TERMINAL_STOPS and not self.transport.finished
                and self.transport.n < self.budget.B and self.clock() < T_LAST_FINISH_SEND)

class Orchestrator:
    def __init__(self, ep: Episode):
        self.ep = ep

    def run(self) -> None:
        ep = self.ep
        if ep.already_finished:
            ep.finished = True
            return
        self.setup()
        self.check_time()
        self.drive(ep.scout, "SCOUT")
        self.check_time()
        if self.feasibility():
            return
        ep.llm.reprobe()                       # a provisional oracle death gets one cheap re-probe (LS-04)
        self.drive(ep.elicitor, "ELICIT")
        self.drive(ep.resolver, "RESOLVE")
        self.select_all()
        self.prove()
        if ep.prover.ready(ep) and ep.finisher.try_decline(ep):
            return
        self.commit()
        ep.finisher.checkout_loop(ep)

    def check_time(self) -> None:
        if self.ep.clock() >= T_FALLBACK:
            raise GoFallback("time")

    def setup(self) -> None:
        ep = self.ep
        ep.phase = "SETUP"
        ep.binder.load_tools(ep.tools_raw)
        cached = ep.binder.cached_votes()
        digest = ep.binder.tool_digest()
        bind_jobs = [] if cached else [LlmJob("BIND", {"tools": digest}, models=ladder, label=f"bind_vote_{i}",
                                              validator=ep.binder.vote_validator)
                                       for i, ladder in enumerate(ep.llm.vote_ladders("BIND", 2))]
        parse_jobs = ep.parser.vote_jobs()
        answers = ep.llm.ask_many(bind_jobs + parse_jobs, deadline=SETUP_VOTE_DEADLINE) \
            if (bind_jobs or parse_jobs) else []
        bind_answers = answers[:len(bind_jobs)]
        parse_answers = answers[len(bind_jobs):]
        votes = cached or [v for v in (ep.binder.normalize_vote(a) for a in bind_answers) if v]
        bind_conflicts = ep.binder.conflicts(votes)
        parse_votes = [p for p in parse_answers if isinstance(p, dict)]
        parse_conflicts = ep.parser.compare(parse_votes[0], parse_votes[1]) if len(parse_votes) >= 2 else []
        arbiter_jobs = []
        if ep.clock() + ARBITER_CAP <= SETUP_HARD_LIMIT:
            if bind_conflicts and ep.llm.allowed("BIND_ARBITER"):
                raw_votes = [a for a in bind_answers if isinstance(a, dict)]
                arbiter_jobs.append(LlmJob("BIND_ARBITER", {"tools": digest, "votes": raw_votes}, label="bind_arbiter",
                                           validator=ep.binder.vote_validator))
            if parse_conflicts and ep.llm.allowed("PARSE_ARBITER"):
                payload = dict(ep.parser.payload())
                payload["votes"] = [{k: v for k, v in x.items() if not k.startswith("_")} for x in parse_votes[:2]]
                arbiter_jobs.append(LlmJob("PARSE_ARBITER", payload, validator=ep.parser.validator,
                                           label="parse_arbiter"))
        arb_answers = ep.llm.ask_many(arbiter_jobs, deadline=SETUP_HARD_LIMIT) if arbiter_jobs else []
        bind_arbiter = parse_arbiter = None
        for job, ans in zip(arbiter_jobs, arb_answers):
            if job.key == "BIND_ARBITER":
                bind_arbiter = ep.binder.normalize_vote(ans)
            else:
                parse_arbiter = ans
        ep.binder.build(votes, bind_arbiter)
        if not cached:
            ep.binder.store_votes(votes)
        bnd = ep.binder.binding
        ep.trace.decision("bind", source="cache" if cached else bnd.source, votes=len(votes), conflicts=bind_conflicts,
                          finish_kind=bnd.finish_kind, finish_conf=bnd.finish_conf,
                          order_tool=bnd.finish_tool or (bnd.by_role.get(ROLE_ORDER) or [None])[0],
                          decline_tool=bnd.finish_tool or (bnd.by_role.get(ROLE_DECLINE) or [None])[0],
                          dropped_labels=ep.binder.dropped_labels[:20], table=ep.binder.table())
        ep.parse_votes = list(parse_votes)
        ep.ledger = ep.parser.parse(parse_answers, arbiter=parse_arbiter, conflicts=parse_conflicts)
        led = ep.ledger
        ep.trace.decision("parse", source=led.source, conflicts=led.conflicts, cap_c=led.cap_c, cap_span=led.cap_span,
                          objective=led.objective, in_stock=led.in_stock, msgs=led.stated_msgs, steps=led.stated_steps,
                          lines=[{"words": l.words, "shelf_words": l.shelf_words, "quantity": l.qty,
                                  "constraints": [[c.attr, c.op, c.vals] for c in l.constraints],
                                  "query_alts": l.query_alts} for l in led.lines],
                          refs=len(led.refs), hints=len(led.hints))
        B, src = resolve_B(ep.max_steps, led.stated_steps)
        ep.budget.B, ep.budget.B_source = B, src
        if led.stated_msgs is not None:
            ep.budget.M, ep.budget.M_source = led.stated_msgs, "stated"
        else:
            ep.budget.M, ep.budget.M_source = DEFAULT_M, "default"
            ep.budget.M_probe_open = False
        ep.cartman.setup(ep)
        ep.reviewer.plan_procedures(ep)
        self.ingest_opening()
        nums = ep.plan.compute("setup")
        if nums.get("S_status") and ep.binder.tool_for(ROLE_STATUS):
            ep.run(Intent(IK_STATUS, {}, P_PROBE, None, why="read the message budget"))
            ep.plan.compute("status read")

    def ingest_opening(self) -> None:
        ep = self.ep
        if ep.opening is None:
            return
        dec = ep.decoder.decode(DecodeContext(tool="opening"), Raw(observation=ep.opening))
        if dec.kind in (D_RECORDS, D_RECORD):
            ep.fields.learn("opening", dec)
            for rec in dec.records:
                f = ep.fields.extract("opening", rec)
                if isinstance(f.get(FR_ID), str):
                    merge_record(ep.catalog, f[FR_ID], f, 0, "opening", True, False)
        ep.events_machine.ingest(ep, dec, None)

    def drive(self, engine: Engine, phase: str) -> None:
        """The shape of every phase (DESIGN 3.4)."""
        ep = self.ep
        ep.phase = phase
        last_sig, repeats = None, 0
        for _ in range(2 * max(1, ep.budget.B) + 10):
            if ep.finished:
                return
            self.check_time()
            it = engine.next_intent(ep)
            if it is None:
                return
            sig = it.signature()
            if sig == last_sig:
                repeats += 1
                if repeats >= 2:
                    ep.trace.decision("engine_loop", engine=engine.name, intent=it.kind)
                    return
            else:
                last_sig, repeats = sig, 0
            if not ep.plan.allows(it):
                return
            dec = ep.run(it)
            engine.after(ep, it, dec)
            if ep.events.dirty:
                ep.events_machine.process(ep)
            if engine.done(ep) or ep.plan.deadline(phase):
                return

    def feasibility(self) -> bool:
        ep = self.ep
        if not any(ep.feasibility.triggered(ep, line) for line in ep.ledger.lines):
            return False
        self.drive(ep.feasibility, "FEASIBILITY")
        return ep.prover.ready(ep) and ep.finisher.try_decline(ep)

    def select_all(self) -> None:
        ep = self.ep
        self.drive(ep.coverage, "SELECT")
        self.drive(ep.selector, "SELECT")          # M6: bundle lookups, pick verification, accessory pages
        ep.phase = "SELECT"
        ep.selector.select(ep)

    def prove(self) -> None:
        """M7 decline evidence before the commit-deadline proofs (DESIGN 5.18): for lines without an orderable item, or
        an over-cap basket, one relaxed query and opens of unread rows up to u_c_hard; selection is redone if read."""
        ep = self.ep
        ep.prover.gathered = False
        self.drive(ep.prover, "PROVE")
        if ep.prover.gathered:
            ep.selector.select(ep)

    def commit(self) -> None:
        ep = self.ep
        ep.phase = "COMMIT"
        ep.reviewer.review(ep)
        self.drive(ep.cartman, "COMMIT")
        if ep.cartman.q4_deferred and not ep.finished:
            ep.trace.decision("q4_defer", lines=sorted(ep.cartman.q4_deferred))
            ep.cartman.q4_deferred.clear()
            self.drive(ep.coverage, "SELECT")      # coverage owes the cheaper rows (Q4); then write anyway
            ep.selector.select(ep)
            ep.cartman.q4_advisory = True
            self.drive(ep.cartman, "COMMIT")

def build_summary(ep: Episode) -> dict:
    b = ep.budget
    lines = []
    for line in ep.ledger.lines:
        verdict = line.verdicts.get(line.choice) if line.choice else None
        lines.append(dict(line=line.idx, words=line.words, shelf=line.shelf_label, quantity=line.qty,
                          choice=line.choice, tier=verdict.tier if verdict else None,
                          cert=line.cert.kind if line.cert else None, opened=list(line.opened)))
    out = dict(B=b.B, B_source=b.B_source, M=b.M, M_source=b.M_source, R=b.R, F=b.F, u_first_finish=b.first_finish_step,
               steps=ep.transport.n, msgs_used=b.msgs_used, lines=lines,
               needs_disclosed=sum(line.disclosed for line in ep.ledger.lines),
               refs=[dict(rid=r.rid, status=r.status, hops=len(r.hops)) for r in ep.ledger.refs],
               refusals=[e.text for e in ep.events.queue if e.via == "refusal"], finish_kind=ep.finish_kind,
               finish_tool=ep.finish_tool,
               fallback_reason=ep.stop_reason, usd=round(b.usd, 5), llm_calls=b.llm_calls, t=round(ep.clock(), 2),
               operator_calls=ep.executor.operator_calls, bind_source=ep.binder.binding.source,
               parse_source=ep.ledger.source, oracles_dead=b.dead_inference, cart=dict(ep.cart.expected),
               cart_semantics=ep.cart.semantics, env_retries=ep.transport.retries, env_resyncs=ep.transport.resyncs)
    out[FINISHED_KEY] = ep.finished
    return out

def agent_main(problem_data: dict) -> list:
    """Entry point. Never raises; always attempts a finish while a step and about 10 s remain."""
    try:
        ep = Episode(problem_data)
    except Exception as exc:
        return [{"role": "lodestar", "decision": "crash", "err": repr(exc)[:300], "where": "episode init"}]
    try:
        try:
            ep.orchestrator.run()
        except StopEpisode as stop:
            ep.stop_reason = stop.reason
        except GoFallback as go:
            ep.stop_reason = go.reason
        except Exception as exc:
            ep.stop_reason = ep.stop_reason or "exception"
            ep.trace.decision("crash", err=repr(exc)[:300], where=clip_text(traceback.format_exc(), 1500))
        finally:
            if not ep.finished and ep.can_act():
                try:
                    ep.finisher.fallback(ep, ep.stop_reason or "exception")
                except StopEpisode as stop:
                    ep.stop_reason = ep.stop_reason or stop.reason
                except Exception as exc:
                    ep.trace.decision("crash", err=repr(exc)[:300], where=clip_text(traceback.format_exc(), 1500))
    except Exception:
        pass
    try:
        ep.trace.summary(build_summary(ep))
    except Exception as exc:
        ep.trace.decision("crash", err=repr(exc)[:300], where="summary")
    try:
        ep.llm.close()
    except Exception:
        pass
    try:
        return ep.trace.export()
    except Exception:
        return []

__all__ = ["agent_main"]
