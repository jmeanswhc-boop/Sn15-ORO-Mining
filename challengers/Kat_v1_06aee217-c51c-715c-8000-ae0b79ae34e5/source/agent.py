"""A shopping agent for ORO Bench store worlds, run by an event-driven controller.

Code owns the flow. The controller keeps a queue of episode events (the request read, a search round due, a
question due, a shopper reply, picks due, a cart ready to order, a refusal, nothing fitting, the clock or the steps
running out, a pilot turn) and hands each one to the handler registered for it; a handler does its one job and
queues the events that follow. Models only read: the request and the tool list once, then short calls that judge
listings, read the shopper and plan questions. Where the routine is stuck (nothing buyable, an order or exit turned
down, an unknown store line or tool) a model pilot chooses a few moves that the same engines carry out.
"""

from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

from src.agent.proxy_client import ProxyClient

FAST_MODEL = "anthropic/claude-haiku-5.5"
INTAKE_MODEL = "anthropic/claude-sonnet-5.5"
VERIFY_MODEL = "anthropic/claude-sonnet-5.5"
PLAN_MODEL = "anthropic/claude-sonnet-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"
# wall clock: tasks get 300 s, we plan within CLOCK_BUDGET and wind down inside END_MARGIN
END_MARGIN = 45.0
CLOCK_BUDGET = 255.0
# a store-only sweep after a refused close (no model call) may run this close to CLOCK_BUDGET
SWEEP_FLOOR = 10.0
# seconds added once when a refused close brings a change to act on (the last model call then ends near 250 s)
GRACE_SECONDS = 20.0
# a last model review of the read listings may still start this many seconds before CLOCK_BUDGET (with no store step
# of its own it ends near 250 s, inside the 300 s limit)
LAST_LOOK_LEFT = 20.0
# seconds one question takes (the message and the reading of its reply), held back from the first search pass
QUESTION_SECONDS = 10.0
# seconds per product beyond the first that the first search pass leaves for the work after the questions
AFTER_ASK_HOLD = 20.0
# text sizes handed to judges and screeners
ROW_CHARS = 260
PAGE_CHARS = 2200
# output room for the second request reading, after the first came back cut off or unreadable
INTAKE_TOKENS = 12000
# output room for the question plan: the fast and spare models think before they answer and were cut at 700 tokens
PLAN_TOKENS = 1600
# output caps for answers that grow with the records (Arena, race-15 world: at 600 tokens 15 of 90 reference-page
# reads, 9 of 18 rankings and 3 of 17 basket checks came back empty at the cap; row sorts 18 of 245 at 1200)
POINTER_TOKENS = 1600
MATCH_TOKENS = 1200
RANK_TOKENS = 1600
BASKET_TOKENS = 1200
TRIAGE_TOKENS = 2000
# closing line for a reply a program parses: some hosts wrap the call in their own JSON answer, and the models then
# summed up or left out the requested JSON (Arena login: 34% empty planner replies without it, 1 in 52 with it)
JSON_CLOSER = ("\n\nThe text of your reply is read by a program: write out the JSON object itself in full as that text "
               "(never a summary of it, never an empty reply).")
# spend lines in USD per task: Sonnet roles fall back to the fast model, exploring stops, the pilot stops
PILOT_TURNS = 6
PILOT_SPEND = 0.065
SPEND_CAP = 0.085
SEARCH_SPEND = 0.05
STRONG_SPEND = 0.03
LIST_WIDTH = 6
STORE = ProxyClient(timeout=90, max_retries=2, retry_delay=1)
MODELS = ProxyClient(timeout=75, max_retries=1, retry_delay=1)
# per-world memory kept across tasks run in the same process
STORE_MEMORY = {}
RETRY_STATUSES = (500, 502, 503, 504, 429)
CART_ACCEPTED = r"\b(ok|okay|success\w*|added|done|accepted|applied|updated|in (the |your )?(cart|basket))\b"
# selector-style fields bundle several choices into one value; we never ask the shopper about them
SELECTOR_WORDS = set(["variant", "option"])
# relation against a referenced page -> the relation it turns into once that page has been read
REFERENCE_OPS = dict([
    ("same as reference", "is"),
    ("different from reference", "is not"),
    ("counterpart of reference", "is"),
    ("at least reference", "at least"),
    ("at most reference", "at most"),
    ("within reference", "at least"),
])
# wording that points the reader at some other listing's page
POINTS_ELSEWHERE = re.compile(r"\b(?:more|details?|info(?:rmation)?|see|read|look)\b[^.;:!?]{0,24}?\b(?:part|model|item|listing|"
                              r"product|page|code)\s*(?:no\.?|number|#)?\s*([A-Za-z0-9][A-Za-z0-9-]{2,})", re.I)
# the series prefix of a dashed code, and part / model codes in general
SERIES_PREFIX = re.compile(r"\b([A-Z]{1,5}\d{1,3})-\d{2,}\b")
CODE_WORD = re.compile(r"\b[A-Za-z]{2,}-?\d{3,}[A-Za-z]?\b|\b[A-Za-z]{2,}\d*-\d{2,}\b")
# a part or model code ("AB1234", "AB-1234"), read the way related_pages and pointed_codes read them
MODEL_CODE = re.compile(r"\b[A-Z]{1,6}-?\d{3,}[A-Z0-9]*\b")
# a model named by a word and a number ("Corvo 512"): stores list such names among codes in what a part fits
MODEL_NAME = re.compile(r"\b[A-Z][A-Za-z]{2,} [A-Z]?\d{2,}[A-Za-z]?\b")
# a code family: a code whose tail is placeholder letters ("AB-12xx" stands for every AB-12.. code)
CODE_FAMILY = re.compile(r"\b([A-Z]{1,6}-?\d+)([xX]+)\b")
EXCEPT_WORDS = re.compile(r"\b(?:except|excluding|but not|other than|apart from)\b", re.I)
SPAN_IN_TEXT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[-\u2013]\s*(\d+(?:\.\d+)?)(?![\w.])")
SPAN_VALUE = re.compile(r"\s*(-?\d+(?:\.\d+)?)\s*([a-z\"']*)\s*(?:to|-|~)\s*(-?\d+(?:\.\d+)?)\s*([a-z\"' ]*)")
FIELD_TEXT = re.compile(r"\s*([A-Za-z_][\w ]{0,40}?)\s*:\s(.*)", re.S)
FIELD_CUT = re.compile(r"\s\|\s(?=[A-Za-z_][\w ]{0,40}?\s*:\s)")
PAIR_CUT = re.compile(r";\s*(?=[A-Za-z_][\w ]{0,40}?=)")
PIECE_SPLIT = r"[,;/|]"
CAMEL_GAP = r"(?<=[a-z0-9])(?=[A-Z])"
JSON_TYPES = {"string": str, "integer": (int, float), "number": (int, float), "array": list, "boolean": bool}
UNFIT = object()


# -------------------- words, numbers and json in replies


def plain_text(text):
    s = str(text or "").lower()
    for old, new in (("&", " and "), ("_", " "), ("-", " ")):
        s = s.replace(old, new)
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def singular(w):
    # crude singular: drop a trailing s from words longer than three letters
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w


def word_set(text):
    wset = set()
    for w in plain_text(text).split():
        if len(w) >= 2:
            wset.add(singular(w))
    return wset


def all_words(text):
    """Like word_set, but single characters survive."""
    return set(map(singular, plain_text(text).split()))


def stem_set(text):
    stems = set()
    for w in word_set(text):
        stems.add(w[:4])
    return stems


def stems_of(words):
    return set(wrd[:4] for wrd in words)


def same_name(a, b):
    """True when two property names differ only in spelling."""
    if plain_text(a) != plain_text(b):
        left = word_set(a)
        return bool(left) and left == word_set(b)
    return True


def overlap(a, b):
    left = word_set(a)
    right = word_set(b)
    if left and right:
        return len(left & right) / len(left | right)
    return 0.0


def near_name(aa, bb):
    """True when two property names read as one property."""
    return same_name(aa, bb) or overlap(aa, bb) >= 0.5


def squeeze(text_s):
    """Collapse whitespace runs to single spaces and trim the ends."""
    return re.sub(r"\s+", " ", text_s).strip()


def split_camel(kkey):
    """Put spaces between the words of a camelCase key."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(kkey))


def spaced_key(key):
    # property key with camelCase split, then flattened
    return plain_text(re.sub(CAMEL_GAP, " ", str(key)))


def is_number(item_x):
    return isinstance(item_x, (int, float))


def is_count(item_x):
    # a real number, booleans excluded
    return isinstance(item_x, (int, float)) and not isinstance(item_x, bool)


def to_number(value):
    raw_v = value
    if isinstance(raw_v, str):
        hit = re.search(r"-?\d+(?:\.\d+)?", raw_v.replace(",", ""))
        if hit is None:
            return None
        try:
            return float(hit.group(0))
        except Exception:
            return None
    if is_number(raw_v) and raw_v is not True and raw_v is not False:
        return float(raw_v)
    return None


def bracket_end(s, at, opener, closer):
    """Position right after the bracket matching the one at `at`, skipping quoted text; -1 if unmatched."""
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


def try_json(s):
    try:
        return True, json.loads(s)
    except Exception:
        return False, None


def first_json(text):
    """The first JSON object in a reply (an array if no object parses), or None."""
    if not text:
        return None
    body = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fenced:
        body = fenced.group(1).strip()
    for pair in ("{}", "[]"):
        at = body.find(pair[0])
        end = -1 if at == -1 else bracket_end(body, at, pair[0], pair[1])
        if end > 0:
            ok, read_d = try_json(body[at:end])
            if ok:
                return read_d
    return try_json(body)[1]


def compact(value, limit):
    s = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(s) > limit:
        s = s[:limit] + "..."
    return s


def as_json(val_s):
    """Prompt-ready JSON that keeps non-ASCII characters."""
    return json.dumps(val_s, ensure_ascii=False)


# -------------------- tool schemas and argument values


def function_of(tool):
    inner = tool.get("function")
    return inner if isinstance(inner, dict) else tool


def schema_type(schema):
    t = schema.get("type")
    if isinstance(t, list):
        t = next((item_x for item_x in t if item_x != "null"), None)
    return t


def enum_value(choices, raw_v):
    if raw_v in choices:
        return raw_v
    spelled = {}
    for c in choices:
        if isinstance(c, str):
            spelled[plain_text(c)] = c
    if not spelled or not isinstance(raw_v, str):
        return UNFIT
    flat = plain_text(raw_v)
    if flat in spelled:
        return spelled[flat]
    top = max(spelled, key=lambda c: overlap(c, raw_v))
    return spelled[top] if overlap(top, raw_v) >= 0.5 else UNFIT


def text_value(schema, raw_v):
    if not isinstance(raw_v, str):
        raw_v = str(raw_v[0]) if (isinstance(raw_v, list) and raw_v) else str(raw_v)
    ceiling_v = schema.get("maxLength")
    if isinstance(ceiling_v, int) and len(raw_v) > ceiling_v:
        head_t = raw_v[:ceiling_v]
        raw_v = head_t.rsplit(" ", 1)[0] if " " in head_t else head_t
    shortest = schema.get("minLength")
    too_short = isinstance(shortest, int) and len(raw_v) < shortest
    return UNFIT if too_short else raw_v


def fit_to_schema(schema, value):
    """Coerce a value to its argument schema while keeping its meaning: convert the type, wrap in a list, clip to
    the bounds, or adopt the schema's spelling of an enum entry. UNFIT if none of that works."""
    t = schema_type(schema)
    raw_v = value
    choices = schema.get("enum")
    if isinstance(choices, list) and choices:
        return enum_value(choices, raw_v)
    if t in ("number", "integer"):
        num = to_number(raw_v)
        if num is None:
            return UNFIT
        lowest, highest = schema.get("minimum"), schema.get("maximum")
        if is_number(lowest) and num < lowest:
            num = lowest
        if is_number(highest) and highest < num:
            num = highest
        return int(round(num)) if t == "integer" else num
    if t == "string":
        return text_value(schema, raw_v)
    if t == "boolean":
        if raw_v is True or raw_v is False:
            return raw_v
        return {"false": False, "true": True}.get(str(raw_v).strip().lower(), UNFIT)
    if t == "array":
        many = raw_v if isinstance(raw_v, list) else [raw_v]
        most = schema.get("maxItems")
        if isinstance(most, int):
            many = many[:most]
        return many
    return raw_v


def filler_value(schema):
    """The plainest value a schema accepts, for a required argument nobody supplied."""
    t = schema_type(schema)
    choices = schema.get("enum")
    if isinstance(choices, list) and choices:
        return choices[0]
    if t == "boolean":
        return False
    if t == "string":
        return fit_to_schema(schema, "n/a")
    if t in ("number", "integer"):
        return fit_to_schema(schema, 1)
    if t != "array" or schema.get("minItems"):
        return UNFIT
    return []


def first_of_type(fields, typ):
    for kn, val in fields.items():
        if isinstance(val, dict) and val.get("type") == typ:
            return kn
    return None


def is_object_list(val):
    if not isinstance(val, dict) or val.get("type") != "array":
        return False
    inner = val.get("items")
    return isinstance(inner, dict) and inner.get("type") == "object"


def is_plain_type(val, typ):
    # an argument of this type that is not a fixed choice list
    return isinstance(val, dict) and val.get("type") == typ and not val.get("enum")


def first_named(order, pool):
    # the first required name found in the pool, else the pool's first, else None
    for name in order:
        if name in pool:
            return name
    return pool[0] if pool else None


def mode_argument(schema):
    mode = schema.get("mode")
    if isinstance(mode, dict):
        return mode.get("arg")
    return None


def mode_switch(tool_d):
    """A role's mode argument with its values, or None."""
    mode_d = tool_d.get("mode")
    return mode_d if isinstance(mode_d, dict) else None


# -------------------- store results


def text_field_value(raw):
    """Parse one textual field: none/true/false and bare numbers become values, semicolon-separated name=value
    pairs become a dict, everything else stays text."""
    val = raw.strip()
    low = val.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    if re.fullmatch(r"-?\d+", val):
        return int(val)
    if re.fullmatch(r"-?\d+\.\d+", val):
        return float(val)
    parts_l = PAIR_CUT.split(val)
    if "=" in val and all(re.fullmatch(r"[A-Za-z_][\w ]{0,40}?=.*", pt.strip(), re.S) for pt in parts_l):
        return {pt.split("=", 1)[0].strip(): pt.split("=", 1)[1].strip() for pt in parts_l}
    return val


def records_from_text(obs, depth=0):
    """Turn records that arrive as text into records: JSON text, or a line of "name: value" fields joined by
    " | ". One store wrote every row and page like that, and none of it was being read."""
    if depth > 8:
        return obs
    if isinstance(obs, list):
        return [records_from_text(item_x, depth + 1) for item_x in obs]
    if isinstance(obs, dict):
        return {kk: records_from_text(vv, depth + 1) for kk, vv in obs.items()}
    if isinstance(obs, str) and obs.lstrip()[:1] in ("{", "["):
        try:
            return records_from_text(json.loads(obs), depth + 1)
        except ValueError:
            return obs
    if not isinstance(obs, str) or " | " not in obs:
        return obs
    parts_l = FIELD_CUT.split(obs.strip())
    if len(parts_l) < 3:
        return obs
    fields = []
    for pt in parts_l:
        hit = FIELD_TEXT.fullmatch(pt)
        if hit is None:
            return obs
        fields.append([hit.group(1).strip(), hit.group(2)])

    # the store writes its field names one way (case, spaces); a cut before a name written otherwise was text inside
    # the field before it, such as a description holding " | Size: ..."
    def style(name):
        return " " in name, name != name.lower()
    styles = {}
    for nm, _ in fields:
        styles[style(nm)] = styles.get(style(nm), 0) + 1
    usual = max(styles, key=styles.get)
    count = styles[usual]
    kept = []
    for nm, raw in fields:
        if kept and count >= 3 and style(nm) != usual:
            kept[-1][1] += f" | {nm}: {raw}"
        else:
            kept.append([nm, raw])
    return {nm: text_field_value(raw) for nm, raw in kept}


def find_rows(obs, depth=0):
    if isinstance(obs, list):
        every = bool(obs)
        for item_x in obs:
            if not isinstance(item_x, dict):
                every = False
                break
        return obs if every else None
    # rows may sit several envelopes down; a two-level limit read a deeper-wrapped result as empty, so nothing was
    # ever read and every order task ended in a decline
    if depth >= 6 or not isinstance(obs, dict):
        return None
    longest, size = None, 0
    for inner in obs.values():
        rows = find_rows(inner, depth + 1)
        if rows and (longest is None or len(rows) > size):
            longest, size = rows, len(rows)
    return longest


def is_sentence(value):
    if not isinstance(value, str):
        return False
    return len(value) > 24 and " " in value.strip()


def separate_lines(obs):
    """Split a result into (the result proper, the shopper lines that rode along with it)."""
    said_t = []
    if not isinstance(obs, dict):
        return obs, said_t
    nested = sum(isinstance(item_x, dict) for item_x in obs.values())
    # a record with an id-like field is a record, not an envelope around one
    bare_word = False
    for item_x in obs.values():
        if isinstance(item_x, str):
            trimmed = item_x.strip()
            if trimmed and " " not in trimmed:
                bare_word = True
    envelope = nested == 1 and len(obs) <= 3 and not bare_word
    kept = {}
    for key in obs:
        item_x = obs[key]
        if isinstance(item_x, list) and item_x and all(map(is_sentence, item_x)):
            said_t += [str(y) for y in item_x]
        elif envelope and is_sentence(item_x):
            said_t.append(str(item_x))
        else:
            kept[key] = item_x
    if said_t and len(kept) == 1:
        only = list(kept.values())[0]
        if isinstance(only, dict):
            kept = only
    return kept, said_t


def cart_reply_refuses(obs):
    """A single-field cart reply whose text explains a refusal instead of confirming the change."""
    if not isinstance(obs, dict) or len(obs) != 1:
        return False
    sole = list(obs.values())[0]
    if not isinstance(sole, str):
        return False
    return re.search(CART_ACCEPTED, sole, re.I) is None


def nonempty_table(table):
    clean = {}
    for kn, val in table.items():
        if val not in (None, ""):
            clean[str(kn)] = val
    return clean


def shopper_words(obs):
    """What the shopper said in a message result: the text, or a record's longest text field."""
    if isinstance(obs, str):
        return obs
    if not isinstance(obs, dict):
        return None
    texts = [fv_ for fv_ in obs.values() if isinstance(fv_, str)]
    return max(texts, key=len) if texts else compact(obs, 400)


def refusal_record(name_s, reply_d):
    """The view of a refused closing call that the pilot gets."""
    return {"tool": name_s, "result": compact(reply_d["obs"], 600), "lines": reply_d["lines"][:3]}


# -------------------- model calls


def quietly(duty):
    try:
        out = duty()
    except Exception:
        out = None
    return out


def in_parallel(jobs, workers=3):
    """Run independent jobs concurrently; the results come back in job order, None for a job that raised."""
    count = len(jobs)
    if count < 2:
        return list(map(quietly, jobs))
    with ThreadPoolExecutor(max_workers=min(count, workers)) as pool:
        outs = pool.map(quietly, jobs)
        return list(outs)


def has_choices(step_spec):
    return isinstance(step_spec, dict) and bool(step_spec.get("choices"))


def reply_content(step_spec):
    said_t = (step_spec["choices"][0].get("message") or {}).get("content")
    if isinstance(said_t, list):
        chunks = []
        for part in said_t:
            if isinstance(part, dict):
                chunks.append(str(part.get("text") or ""))
        said_t = "".join(chunks)
    return said_t


class ModelLine:
    def __init__(self, episode):
        self.calls = 0
        self.spent = 0.0
        self.episode = episode
        self.key_exhausted = False
        self.unparsed = []
        self.late_floor = None

    def model_for(self, model):
        """Which model really serves a call for `model`: past the spend line the strong seats drop to the fast model."""
        strong = (INTAKE_MODEL, PLAN_MODEL, VERIFY_MODEL)
        if model != FAST_MODEL and model in strong and STRONG_SPEND <= self.spent:
            return FAST_MODEL
        return model

    def complete(self, model, system, user, max_tokens):
        name = self.model_for(model)
        clock = self.episode.time_left
        remaining = clock()
        floor = END_MARGIN - 12 if self.late_floor is None else self.late_floor
        if floor > remaining:
            return None
        if self.key_exhausted:
            return None
        if self.spent >= SPEND_CAP:
            # the run's key pays the shopper too and stops every call once spent: past this line the task goes on
            # with code alone
            self.key_exhausted = True
            self.episode.trace_event("spend_stop", dollars=round(self.spent, 4))
            return None
        wait = min(45, remaining - END_MARGIN + 20) if self.late_floor is None else min(25, remaining)
        MODELS.timeout = int(max(10, wait))
        head_t = system
        if name.startswith("anthropic/"):
            head_t = [dict(type="text", text=system, cache_control=dict(type="ephemeral"))]
        chat = [dict(role="system", content=head_t), dict(role="user", content=user)]
        payload = dict(model=name, messages=chat, temperature=0, max_tokens=max_tokens)
        send = lambda: MODELS.post_verbose("/inference/chat/completions", payload)
        self.calls = self.calls + 1
        answer_d = send()
        tries = 0
        # retry only refusals (rate limit, gateway errors), never timeouts, and only while time allows
        while tries < 2 and not has_choices(answer_d.data):
            exc = answer_d.error
            code = exc.get("status") if isinstance(exc, dict) else None
            if code not in RETRY_STATUSES or clock() < END_MARGIN + 10:
                break
            tries += 1
            time.sleep(2.0 * tries)
            answer_d = send()
        got = answer_d.data
        if has_choices(got):
            spent = (got.get("usage") or {}).get("cost")
            if is_number(spent):
                self.spent = self.spent + float(spent)
            said_t = reply_content(got)
            if said_t:
                return str(said_t)
        why = str(getattr(answer_d, "error", ""))
        exc = getattr(answer_d, "error", None)
        if isinstance(exc, dict) and exc.get("status") in (402, 403):
            # the key is spent or capped: every later call would fail the same way, so none is sent
            self.key_exhausted = True
        self.episode.trace_event("llm_failure", model=name, error=why[:200])
        return None

    def json_reply(self, models, system, user, max_tokens):
        # replies that held no JSON are kept for callers able to read their answer from plain sentences
        self.unparsed = []
        tried = []
        # every parsed reply gets the closing line the planner carries (empty replies on some hosts cost a fallback
        # call each: about a third of a task's time went to them on the Arena login)
        if not user.endswith(JSON_CLOSER):
            user = user + JSON_CLOSER
        for name in models:
            # past the spend line two seats of a chain can be one model: a second identical call is skipped
            if self.model_for(name) in tried:
                continue
            tried.append(self.model_for(name))
            said_t = self.complete(name, system, user, max_tokens) or ""
            got = first_json(said_t)
            if got is not None:
                return got
            if said_t.strip():
                self.unparsed.append(said_t)
        return None


def quick_json(ep, system, user_t, max_tokens):
    """A JSON answer from the fast model, falling back to the spare one."""
    return ep.models.json_reply([FAST_MODEL, BACKUP_MODEL], system, user_t, max_tokens)


# -------------------- listings, demands and products


class Listing:
    def __init__(self, id_):
        self.ident = id_
        self.heading = self.blurb = self.remarks = ""
        self.kind = self.maker = self.vendor = ""
        self.price = None
        self.available = None
        self.table = {}
        self.fully_read = False
        # True once a compare row or the listing's own page was read, so its notes are known
        self.page_seen = False
        self.screened = False
        self.screen = "maybe"

    def view(self, limit=PAGE_CHARS):
        row_rec = dict(id=self.ident, title=self.heading[:160])
        for kn, val in (("category", self.kind), ("brand", self.maker), ("seller", self.vendor)):
            if val:
                row_rec[kn] = val
        for kn, val in (("price", self.price), ("can_buy", self.available)):
            if val is not None:
                row_rec[kn] = val
        if self.table:
            # stores repeat one value under several spellings of the key: show each pair once
            usage, marks = {}, set()
            for label_s, val in self.table.items():
                mark = (spaced_key(label_s), json.dumps(val, sort_keys=True, default=str))
                if mark in marks:
                    continue
                marks.add(mark)
                usage[label_s] = val
            row_rec["properties"] = usage
        if self.remarks:
            row_rec["seller_notes"] = self.remarks[:600]
        if self.blurb:
            row_rec["description"] = self.blurb[:400] if self.table else self.blurb[:500]
        size = len(json.dumps(row_rec, ensure_ascii=False))
        if self.table and size > limit:
            usage = row_rec["properties"]
            admit = max(6, int(len(usage) * limit / size))
            row_rec["properties"] = dict(list(usage.items())[:admit])
        if row_rec.get("description") and len(json.dumps(row_rec, ensure_ascii=False)) > limit:
            row_rec["description"] = row_rec["description"][:200]
        return row_rec


class Demand:
    def __init__(self, prop, op, value, reference=None, origin="request"):
        self.origin = origin
        self.reference = reference if isinstance(reference, dict) else None
        self.value = str(value or "").strip()
        self.op = str(op or "is").strip().lower()
        self.prop = str(prop or "").strip()
        self.traced_key = None
        self.first_traced_key = None
        self.owned_codes = None
        self.prop_aligned = False

    def as_line(self):
        pending = self.reference and self.op in REFERENCE_OPS
        suffix = "(not yet resolved)" if pending else self.value
        return f"{self.prop}: {self.op} {suffix}"

    def key_text(self):
        label_s = self.prop
        return plain_text(label_s)


class Product:
    def __init__(self, schema, index):
        pick = schema.get
        self.position = index
        self.name = str(pick("name") or pick("category") or "the product").strip()
        self.kind = str(pick("category") or self.name).strip()
        amount = pick("quantity")
        self.quantity = int(amount) if (is_count(amount) and amount >= 1) else 1
        self.demands = []
        for frow in pick("requirements") or []:
            if not isinstance(frow, dict) or not frow.get("property"):
                continue
            self.demands.append(Demand(frow.get("property"), frow.get("op"), frow.get("value"), frow.get("reference")))
        self.hints = [str(host) for host in (pick("hints") or []) if host]
        # most one unit may cost; a zero or negative cap means none
        ceiling = to_number(pick("price_cap"))
        self.unit_cap = None if (ceiling is not None and ceiling <= 0) else ceiling
        terms = [str(term_s) for term_s in (pick("search_terms") or []) if term_s]
        self.planned_queries = terms or [self.kind]
        # the reader puts the bare product noun last
        self.noun = self.planned_queries[-1]
        self.seen = {}
        self.verdicts = {}           # id -> {"failed", "unsure", ...}
        self.judged_on = {}       # id -> requirement lines judged
        self.asked = []
        self.no_view = []
        self.exhausted = False
        self.choice = None
        self.categories_ok = None
        self.paging = {}             # search signature -> paging state
        self.rows_by_query = {}        # keyword search -> ids listed
        self.value_ids = set()     # ids listed by value queries
        self.values_complete = False
        self.value_signature = None
        self.value_queries_sent = []
        self.heard_queried = 0
        self.heard_search_done = False
        self.bound_searched = False
        self.bounds_sent = []
        self.span_searched = False
        self.asked_queries = False
        self.more_hidden = False
        self.narrowed = False
        self.swept_below = None
        self.measure_looked = False
        self.measure_calls = 0
        self.measure_positions = {}
        self.misfit_ids = set()
        self.fit_lines_seen = set()
        self.bound_host = None
        self.bound_host_listing = None
        self.need_replies = []
        self.reworded = []
        self.listed_props = []
        self.unclear_replies = 0
        self.store_order = {}

    def plan_query(self, query):
        """Queue a query for this product's next keyword search unless it is planned already."""
        if not query:
            return
        flat = plain_text(query)
        for planned in self.planned_queries:
            if plain_text(planned) == flat:
                return
        # right after the planned queries already sent (seeks counts page calls too, so it is no position here)
        sent = {plain_text(kk.split(":", 1)[1]) for kk in self.paging if kk.startswith("search_text:")}
        at = next((ix for ix, planned in enumerate(self.planned_queries) if plain_text(planned) not in sent), len(self.planned_queries))
        self.planned_queries.insert(at, query[:60])

    def over_unit_cap(self, listing):
        price = listing.price
        if self.unit_cap is None or price is None:
            return False
        return price > self.unit_cap + 1e-6

    def firm_demands(self):
        acc_p = []
        for dm in self.demands:
            if dm.reference and dm.op in REFERENCE_OPS:
                continue
            acc_p.append(dm)
        return acc_p

    def demand_lines(self):
        return list(number.as_line() for number in self.firm_demands())

    def buried_values(self, listing):
        """Plain-value demands whose matching field has the value only inside a longer phrase."""
        matches = 0
        for dm in self.firm_demands():
            if dm.op not in ("is", "one of") or not dm.value:
                continue
            want_bags = field_wanted_sets(dm.value, dm.prop)
            own_keys = [val for kn, val in listing.table.items() if key_names_property(kn, dm.prop)]
            if want_bags and own_keys:
                chunks = value_pieces(own_keys)
                if not any(wrd in chunks for wrd in want_bags):
                    matches += 1
        return matches

    def exact_values(self, listing):
        """Plain-value demands stated exactly by one listed piece of some property."""
        if not listing.table:
            return 0
        chunks = value_pieces(listing.table.values())
        hit_n = 0
        for dm in self.firm_demands():
            if dm.op in ("is", "one of") and dm.value:
                if any(wrd in chunks for wrd in wanted_alternatives(dm.value)):
                    hit_n += 1
        return hit_n

    def fits(self, include_unsure=False, cheapest=False, strong_only=False):
        lines_now = set(self.demand_lines())
        certain, doubtful = [], []
        for offer_id, offer in self.seen.items():
            if not offer.fully_read:
                continue
            verdict_d = self.verdicts.get(offer_id)
            if verdict_d is None or verdict_d["failed"] or self.over_unit_cap(offer):
                continue
            if strong_only and verdict_d.get("weak"):
                continue
            if not (self.judged_on.get(offer_id, set()) >= lines_now):
                continue
            if verdict_d["unsure"] or self.loose_lines(offer_id):
                doubtful.append(offer)
            else:
                certain.append(offer)
        if cheapest:
            # by price; at equal price a listing stating the value as written goes before one burying it in a phrase
            def order(o):
                return sort_price(o), self.buried_values(o) > 0
        else:
            # table evidence first, then fewer weak readings, then price
            def order(o):
                return -self.exact_values(o), len(self.verdicts[o.ident].get("weak") or ()), sort_price(o)
        certain.sort(key=order)

        def doubt(o):
            return len(self.verdicts[o.ident]["unsure"]) + len(self.loose_lines(o.ident)), sort_price(o)
        doubtful.sort(key=doubt)
        if include_unsure and not certain:
            return doubtful
        return certain

    def loose_lines(self, lid):
        """Word-value requirement lines this listing meets only via a synonym or broader word, not its own words."""
        verdict_d = self.verdicts.get(lid) or {}
        line_set = set(verdict_d.get("loose") or ())
        if not line_set:
            return []
        offer = self.seen.get(lid)
        acc_p = []
        for dm in self.firm_demands():
            line_s = dm.as_line()
            if line_s not in line_set or dm.op not in ("is", "one of"):
                continue
            if not dm.value or re.search(r"\d", dm.value) or dm.origin == "fit":
                continue
            if offer is not None and self.holds_words(offer, dm):
                continue
            acc_p.append(line_s)
        return acc_p

    @staticmethod
    def holds_words(listing, req):
        """True when the listing's field holds every word of the value, or one listed piece uses only the value's words."""
        own_keys = [str(val) for kn, val in listing.table.items() if key_names_property(kn, req.prop)]
        wordset = set()
        for line_s in own_keys:
            wordset |= word_set(line_s)
        chunks = [word_set(issue) for line_s in own_keys for issue in re.split(PIECE_SPLIT, line_s) if word_set(issue)]
        for option in req.value.split("|"):
            wset = word_set(option)
            if not wset:
                continue
            if wset <= wordset:
                return True
            for piece in chunks:
                if piece <= wset:
                    return True
        return False


def reference_identity(ref):
    """Seller, category and sorted attribute values: what identifies a referenced listing."""
    if not isinstance(ref, dict):
        return ()
    named = ref.get("attributes")
    values_l = []
    if isinstance(named, dict):
        values_l = sorted(plain_text(item_x) for item_x in named.values() if item_x)
    seller = plain_text(ref.get("seller"))
    return seller, plain_text(ref.get("category")), tuple(values_l)


def value_pieces(values):
    # every comma / slash / semicolon / bar piece of the given values, as word sets
    found = set()
    for val in values:
        for piece in re.split(PIECE_SPLIT, str(val)):
            wset = word_set(piece)
            if wset:
                found.add(frozenset(wset))
    return found


def wanted_alternatives(want):
    # the alternatives of a required value (split at |), as word sets
    acc_p = []
    for variant in want.split("|"):
        wset = word_set(variant)
        if wset:
            acc_p.append(frozenset(wset))
    return acc_p


def field_wanted_sets(want, field):
    # a required value that repeats words of its own property's name: the remaining words carry the value, so a
    # field of that property writing only them states it (kept whole when too little would remain)
    name_bag = word_set(field)
    acc_p = []
    for wset in wanted_alternatives(want):
        rest = wset - name_bag
        if rest and rest != wset and any(len(wrd) >= 3 or wrd.isdigit() for wrd in rest):
            wset = frozenset(rest)
        acc_p.append(wset)
    for variant in want.split("|"):
        # short joining words between a value's outer words (an article, a preposition) carry no value: a field
        # writing the outer words without them writes the same value
        toks = [singular(wrd) for wrd in plain_text(variant).split() if len(wrd) >= 2]
        middle = [wrd for wrd in toks[1:-1] if not (wrd.isalpha() and len(wrd) <= 3)]
        if len(toks) >= 3 and len(middle) < len(toks) - 2:
            bare = frozenset([toks[0], toks[-1]] + middle)
            if len(bare) >= 2 and bare not in acc_p:
                acc_p.append(bare)
    return acc_p


def key_names_property(key, field):
    # a store key that names the required property (same name in other spelling, or mostly the same words)
    return same_name(str(key), field) or overlap(str(key), field) >= 0.5


def any_key_names(offers, field):
    for one in offers:
        for kn in one.table:
            if key_names_property(kn, field):
                return True
    return False


def notes_point_elsewhere(listing, field):
    """True when the seller notes mention another property this listing has its own key for, while naming
    neither the required property nor any of its keys."""
    notes = " %s " % " ".join(singular(wrd) for wrd in plain_text(listing.remarks or "").split())
    if not notes.strip():
        return False

    def named(label):
        words = [singular(wrd) for wrd in spaced_key(label).split()]
        return bool(words) and (" %s " % " ".join(words)) in notes
    if named(field):
        return False
    others = []
    for kn in listing.table:
        if key_names_property(kn, field):
            if named(kn):
                return False
        else:
            others.append(kn)
    return any(named(kn) for kn in others)


def span_holds_bound(listing, dm):
    """A shopper floor or ceiling that falls inside a two-figure span written in the property's field: the listing
    may or may not meet it. A span entirely beyond the bound is left to the judge."""
    # real race 14: every order of such a listing was turned down (10 of 10)
    if dm.op not in ("at least", "at most") or dm.reference:
        return False
    bound = to_number(dm.value)
    if bound is None:
        return False
    want_marks = set(re.findall(r"[a-z]{2,}", str(dm.value).lower()))
    for kn, val in listing.table.items():
        if not isinstance(val, str) or not key_names_property(kn, dm.prop):
            continue
        hit = SPAN_VALUE.fullmatch(val.lower().replace(",", ""))
        if hit is None:
            continue
        marks = {mk.strip("\"' ") for mk in (hit.group(2), hit.group(4))} - {""}
        # a span written in some other unit is not compared
        if marks and want_marks and not any(mk[0] == wm[0] and (len(mk) == 1 or mk[:2] == wm[:2])
                                            for mk in marks for wm in want_marks):
            continue
        low, high = sorted([float(hit.group(1)), float(hit.group(3))])
        if (dm.op == "at least" and low < bound <= high) or (dm.op == "at most" and low <= bound < high):
            return True
    return False


def sort_price(offer):
    return 1e18 if offer.price is None else offer.price


def verdict_of(pd, offer):
    """The verdict stored for a listing ({} if none)."""
    return pd.verdicts.get(offer.ident) or {}


def clean_verdict():
    """An empty verdict: no failure, no doubt."""
    return {"failed": set(), "unsure": set(), "weak": set(), "loose": set()}


def over_cap(ceiling_v, offer_x, count_n):
    """True if count_n units of the listing exceed the cap."""
    if ceiling_v is None or offer_x.price is None:
        return False
    return offer_x.price * count_n > ceiling_v + 1e-6


def queries_sent(pd):
    """Keyword queries the product has already sent."""
    return {kk.split(":", 1)[1] for kk in pd.paging if kk.startswith("search_text:")}


def first_unsent(pd, qlist):
    """Position of the first planned query the product has not sent yet (len(qlist) once all went out)."""
    sent = {plain_text(tq) for tq in queries_sent(pd)}
    return next((ix for ix, qq in enumerate(qlist) if plain_text(qq) not in sent), len(qlist))


def names_in(verdict, key):
    return set(str(el) for el in (verdict.get(key) or []) if el)


# -------------------- the engines the controller drives (mixed into the episode)


class StoreSession:
    """Clock, counters, the trace and every store call."""

    def time_left(self) -> float:
        return CLOCK_BUDGET - (time.monotonic() - self.clock_start)

    def steps_left(self) -> int:
        return self.step_cap - self.steps

    def extend_clock(self, why: str) -> None:
        """Once per task, a refused close that brought a change (budget, dropped need) earns more time to act on it.
        The wind-down otherwise starts about 90 s before the hard limit, and a second decline after a refused
        one was paid 0 times in 270 (races 8-11; Arena, race-15 world: 5 of 33 declines were such repeats)."""
        if not self.grace_used:
            self.grace_used = True
            self.clock_start += GRACE_SECONDS
            left = round(self.time_left(), 1)
            self.trace_event("close_grace", why=why, left=left)

    def reads_cutoff(self) -> float:
        """Seconds-left threshold below which searching and reading stop: the closing margin, or the sweep floor during a store-only sweep."""
        floor = END_MARGIN if self.sweep_floor is None else self.sweep_floor
        return floor + self.question_hold

    def trace_event(self, kind: str, **fields: object) -> None:
        stamp = round(time.monotonic() - self.clock_start, 1)
        self.trace.append({"kind": kind, "step": self.steps, "t": stamp, **fields})

    def remember_store(self) -> None:
        STORE_MEMORY[self.store_key] = {"maps": self.record_keys, "compare_limit": self.compare_size,
                                        "thin_compare": self.compare_thin, "all_words": self.all_words_needed}

    def spent_for_search(self) -> bool:
        """True once spending reached the cap for exploring."""
        return self.models.spent >= SEARCH_SPEND

    def batch_reads(self) -> bool:
        """True when one compare call reads several listings together with their tables."""
        return not (self.compare_thin or self.compare_off) and bool(self.roles.get("compare"))

    def call_tool(self, name: str, args: dict) -> dict:
        """Call a tool with schema-fitted arguments, retrying once with repaired ones after an argument refusal. The
        result's "bad_args" tells whether the refusal stands."""
        fitted = self.fit_args(name, args)
        result = self.send_action(name, fitted)
        if result["error"] and self.args_rejected(result):
            retry = self.repair_args(name, fitted, result["obs"])
            if retry not in (None, fitted):
                result = self.send_action(name, retry)
            result["bad_args"] = bool(result["error"]) and self.args_rejected(result)
        return result

    @staticmethod
    def args_rejected(out: dict) -> bool:
        blob = " ".join((f"{out['error']}", compact(out["obs"], 400)))
        return bool(re.search(r"invalid|missing|required|argument|parameter|schema|malformed|validation|unexpected|must be",
                              blob, re.I))

    def roles_for(self, name: str) -> list[str]:
        return [duty for duty, spec in self.roles.items() if isinstance(spec, dict) and spec.get("name") == name]

    def fit_args(self, name: str, args: dict) -> dict:
        """Make arguments fit the schema: convert or clip values, move an unknown-named value onto a missing required
        argument (or drop it), and fill required arguments still missing."""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        if not props:
            return args
        loose = {key: val for key, val in args.items() if val is not None and key not in props}
        fitted = ((key, fit_to_schema(props[key], val)) for key, val in args.items() if val is not None and key in props)
        shaped = {key: val for key, val in fitted if val is not UNFIT}
        lacking = [key for key in (schema.get("required") or []) if key not in shaped]
        for target in list(lacking):
            spec = props.get(target) or {}
            want = JSON_TYPES.get(spec.get("type"), object)
            # same-typed values first (a count never stands in for an id), then the closest name
            order = sorted(loose, key=lambda c: (not isinstance(loose[c], want), -overlap(str(c), target)))
            tried = ((c, fit_to_schema(spec, loose[c])) for c in order)
            moved = next(((c, value) for c, value in tried if value is not UNFIT), None)
            if moved:
                donor, value = moved
                shaped[target] = value
                del loose[donor]
                lacking.remove(target)
        if lacking:
            shaped.update(self.fill_args(name, shaped, lacking))
        return shaped

    def fill_args(self, name: str, have: dict, missing: list, refusal: object = None) -> dict:
        """Supply missing required arguments: the schema default, else one cheap model call, else the plainest allowed
        value, remembered per tool. Choice arguments of an ending tool are left for closing_args."""
        stash = self.filled_args
        props = tool_props(self, name)
        ending = not {"checkout", "decline"}.isdisjoint(self.roles_for(name))
        filled, ask = {}, []
        for key in missing:
            spec = props.get(key) or {}
            remembered = stash.get((name, key), UNFIT)
            if remembered is not UNFIT:
                filled[key] = remembered
            elif "default" in spec:
                filled[key] = spec["default"]
            elif not (ending and isinstance(spec.get("enum"), list)):
                ask.append(key)
        if not ask:
            return filled
        reply = (self.args_from_model(name, have, refusal) or {}).get("args")
        offered = reply if isinstance(reply, dict) else {}
        for key in ask:
            spec = props.get(key) or {}
            val = UNFIT if offered.get(key) is None else fit_to_schema(spec, offered[key])
            if val is UNFIT:
                val = filler_value(spec)
            if val is not UNFIT:
                filled[key] = stash[(name, key)] = val
        self.trace_event("args_filled", tool=name, args=compact(filled, 200))
        return filled

    def args_from_model(self, name: str, have: dict, refusal: object) -> dict | None:
        lines = [f"TOOL: {name}", f"DESCRIPTION: {self.tool_description(name)}",
                 "SCHEMA: " + as_json(self.schemas.get(name) or {}),
                 "ARGUMENTS OF THE CALL: " + json.dumps(have, ensure_ascii=False, default=str),
                 "USED HERE AS: " + (", ".join(self.roles_for(name)) or "unknown")]
        if refusal is not None:
            lines.append("THE STORE'S REFUSAL: " + compact(refusal, 400))
        reply = self.models.json_reply([FAST_MODEL, BACKUP_MODEL], ARGS_RULES, "\n".join(lines), 400)
        return reply if isinstance(reply, dict) else None

    def repair_args(self, name: str, args: dict, refusal: object) -> dict | None:
        """Corrected arguments after an argument refusal (at most two repairs per tool per task); None means no retry."""
        used = self.repairs.get(name, 0)
        if used >= 2 or self.steps_left() < 2:
            return None
        self.repairs[name] = used + 1
        said = compact(refusal, 400)
        cited = [key for key in tool_props(self, name) if key not in args
                 and re.search(r"(?<![A-Za-z0-9_])" + re.escape(key) + r"(?![A-Za-z0-9_])", said)]
        self.trace_event("args_refused", tool=name, args=compact(args, 200), refusal=said[:200])
        if cited:
            return {**args, **self.fill_args(name, args, cited, refusal)}
        reply = (self.args_from_model(name, args, refusal) or {}).get("args")
        return self.fit_args(name, reply) if isinstance(reply, dict) and reply else None

    def send_action(self, name: str, args: dict) -> dict:
        """Send a single action to the store; returns {"obs", "lines", "error", "ok"}."""
        if name not in self.schemas or self.finished or self.stopped:
            return failed_action("unavailable")
        ending = {(self.roles.get(r) or {}).get("name") for r in ("checkout", "decline")}
        if self.steps_left() < (1 if name in ending else 2):
            return failed_action("no steps left")
        packet = dict(self.binding, step=self.steps + 1, action={"name": name, "args": args})
        for _ in range(3):
            remaining = self.time_left()
            if remaining < 3:
                return failed_action("no time")
            STORE.timeout = int(max(4, min(90, remaining - 1)))
            reply = STORE.post_verbose("/environment/call", packet)
            if isinstance(reply.data, dict) and ("observation" in reply.data or "finished" in reply.data):
                return self.absorb_reply(name, args, reply.data)
            why = self.store_failure(reply.error, packet)
            if why is not None:
                return failed_action(why)
        return failed_action("env failure")

    def store_failure(self, error: object, packet: dict) -> str | None:
        """Read a failed store call: why to give up, or None to try again (a step conflict first resyncs the count)."""
        err = error or {}
        status = err.get("status") if isinstance(err, dict) else None
        detail = str(err.get("body") if isinstance(err, dict) else err)[:300]
        self.trace_event("env_failure", status=status, body=detail)
        if status == 402:
            self.stopped = True
            return "payment"
        if status != 409:
            return f"refused {status}" if status in (403, 422) else None
        if "finished" in detail:
            self.finished = True
            return "finished"
        expected = re.search(r"expected (\d+)", detail)
        if expected is None:
            return "conflict"
        self.steps = int(expected.group(1)) - 1
        packet["step"] = self.steps + 1
        return None

    def absorb_reply(self, name: str, args: dict, data: dict) -> dict:
        """Book a store reply: step count, finish flag, queued shopper lines, the trace and the call log."""
        step_no = data.get("step")
        self.tools_used.add(name)
        self.steps = step_no if isinstance(step_no, int) else self.steps + 1
        if data.get("finished"):
            self.finished = True
        obs, said = separate_lines(records_from_text(data.get("observation")))
        self.line_queue += [line for line in said if line not in self.lines_seen]
        problem = obs.get("error") if isinstance(obs, dict) else None
        if isinstance(problem, str) and len(obs) <= 2:
            result = dict(obs=obs, lines=said, error=problem, ok=False)
        else:
            result = dict(obs=obs, lines=said, error=None, ok=True)
        self.trace_event("call", tool=name, args=compact(args, 160), result=compact(obs, 200), lines=said[:3])
        spoke = name == (self.roles.get("message") or {}).get("name")
        width = 300 if spoke else 120
        note = {"step": self.steps, "tool": name, "args": compact(args, width), "result": compact(obs, 160)}
        if said:
            note["shopper_lines"] = [line[:200] for line in said[:3]]
        if spoke:
            note["basket_then"] = [list(pair) for pair in self.cart_signature()]
        self.call_log.append(note)
        return result

    def key_map(self, kind: str, sample: dict) -> dict:
        sig_key = kind + ":" + ",".join(sorted(sample.keys()))
        mapping = self.record_keys.get(sig_key)
        if mapping is None:
            prefix = kind + ":"
            mapping = next((known for sig, known in self.record_keys.items()
                            if known and sig.startswith(prefix)
                            and sum(key in sample for key in known.values()) >= max(2, int(0.6 * len(known)))), None)
            if mapping is not None:
                self.record_keys[sig_key] = mapping
        if mapping is not None:
            # a map learned while text fields were empty lacks them: remap once per newly filled key
            mapped = mapping.values()
            fresh = [key for key, val in sample.items() if key not in mapped and (key, sig_key) not in self.remapped_keys
                     and ((isinstance(val, str) and len(val) > 25) or (isinstance(val, dict) and val))]
            if not fresh or all(role in mapping for role in ("notes", "text", "facts")):
                return mapping
            self.remapped_keys.update((key, sig_key) for key in fresh)
        shown = {}
        for key, val in sample.items():
            text = compact(val, 160) if isinstance(val, (dict, list)) else val
            shown[key] = text[:160] + "..." if isinstance(text, str) and len(text) > 160 else text
        reply = self.models.json_reply([FAST_MODEL, BACKUP_MODEL], KEY_RULES, as_json(shown), 700)
        found = dict(mapping or {})
        if isinstance(reply, dict):
            found.update((role, key) for role, key in reply.items()
                         if role not in found and isinstance(key, str) and key in sample)
        guess_key_roles(found, sample)
        self.record_keys[sig_key] = found
        self.remember_store()
        return found

    def listing_from(self, rec: dict, kind: str) -> Listing | None:
        keys = self.key_map(kind, rec)
        id_key = keys.get("id")
        if not (id_key and isinstance(rec.get(id_key), (str, int))):
            return None
        offer = Listing(str(rec[id_key]))
        typed_fields = {"title": ("heading", str), "text": ("blurb", str), "category": ("kind", str),
                        "brand": ("maker", str), "seller": ("vendor", str), "notes": ("remarks", str),
                        "stock": ("available", bool)}
        for role, (attr, typ) in typed_fields.items():
            key = keys.get(role)
            if key and isinstance(rec.get(key), typ):
                setattr(offer, attr, rec[key])
        if keys.get("price"):
            offer.price = to_number(rec.get(keys["price"]))
        if keys.get("facts") and isinstance(rec.get(keys["facts"]), dict):
            offer.table = nonempty_table(rec[keys["facts"]])
        return offer

    def merge_listing(self, prod: Product, listing: Listing, opened: bool) -> Listing:
        prior = prod.seen.setdefault(listing.ident, listing)
        if prior is not listing:
            texts = {attr: getattr(listing, attr) for attr in ("heading", "blurb", "kind", "maker", "vendor", "remarks")}
            for attr, value in texts.items():
                if value and (opened or not getattr(prior, attr)):
                    setattr(prior, attr, value)
            for attr in ("price", "available"):
                if getattr(listing, attr) is not None:
                    setattr(prior, attr, getattr(listing, attr))
            fresh = listing.table
            if fresh:
                if fresh != prior.table:
                    # changed records: judge again
                    prod.judged_on.pop(prior.ident, None)
                prior.table = fresh
            if not prior.page_seen:
                prior.page_seen = listing.page_seen
        if opened:
            prior.fully_read = True
        return prior

    def tool_description(self, name: str) -> str:
        func = next((f for f in map(function_of, self.tool_entries) if isinstance(f, dict) and f.get("name") == name), None)
        return "" if func is None else str(func.get("description") or "")

    def role_schema(self, role: str) -> dict:
        tool = (self.roles.get(role) or {}).get("name") or ""
        return self.schemas.get(tool, {})

    def paging_style(self, role: str) -> str | None:
        """"index" (page number), "offset" (row offset), "token", or None when the search cannot page."""
        spec = self.roles.get(role) or {}
        page_arg = spec.get("page")
        if not page_arg:
            return None
        if arg_schema(self, role, page_arg).get("type") != "integer":
            return "token"
        hint = str(spec.get("page_kind") or "").lower()
        if hint == "number":
            return "index"
        if hint == "offset" or any(word in page_arg.lower() for word in ("offset", "start", "skip")):
            return "offset"
        return "index"

    def enum_spelling(self, role: str, arg: str, value: object) -> object:
        """The schema's spelling of a listed value; None when no listed value is close."""
        allowed = arg_schema(self, role, arg).get("enum")
        if not (isinstance(allowed, list) and allowed and isinstance(value, str)):
            return value
        spelled = {plain_text(e): e for e in allowed if isinstance(e, str)}
        if not spelled:
            return value
        exact = spelled.get(plain_text(value))
        if exact is not None:
            return exact
        peak = max(spelled, key=lambda e: overlap(e, value))
        return spelled[peak] if overlap(peak, value) >= 0.5 else None

    def clip_arg(self, role: str, arg: str, value: str) -> str:
        ceiling = arg_schema(self, role, arg).get("maxLength")
        if isinstance(ceiling, int) and len(value) > ceiling:
            head = value[:ceiling]
            return head.rsplit(" ", 1)[0] if " " in head else head
        return value

    def clip_text(self, prop: dict, value: str) -> str:
        ceiling = prop.get("maxLength")
        return value[:ceiling] if isinstance(ceiling, int) and len(value) > ceiling else value


def failed_action(why: str) -> dict:
    """A store-call result that never reached the store, or that the store turned down for good."""
    return dict(obs=None, lines=[], error=why, ok=False)


def guess_key_roles(found: dict, sample: dict) -> None:
    """Fill the facts, id and price roles a reading left out, from the record's shape."""
    shapes = (
        ("facts", lambda key, val: isinstance(val, dict) and val
         and all(isinstance(el, (str, int, float)) or el is None for el in val.values())),
        ("id", lambda key, val: isinstance(val, str) and re.fullmatch(r"[A-Za-z]{0,3}-?[0-9a-f]{8,}", val)),
        ("price", lambda key, val: is_count(val) and "price" in key.lower()),
    )
    for role, fits_shape in shapes:
        if role in found:
            continue
        hit = next((key for key, val in sample.items() if fits_shape(key, val)), None)
        if hit is not None:
            found[role] = hit


def tool_props(ep: StoreSession, tool: str) -> dict:
    return (ep.schemas.get(tool) or {}).get("properties") or {}


def required_choices(schema: dict) -> list:
    """Required arguments of a tool schema whose values are listed (an enum)."""
    props = schema.get("properties") or {}
    return [key for key in schema.get("required") or [] if isinstance((props.get(key) or {}).get("enum"), list)]


def arg_schema(ep: StoreSession, role: str, arg: str | None) -> dict:
    return (ep.role_schema(role).get("properties") or {}).get(arg) or {}


def room_left(ep: StoreSession) -> int:
    """Steps beyond those reserved for the ending."""
    return ep.steps_left() - ep.closing_steps()


def low_on_time(ep: StoreSession, more_q: float) -> bool:
    """True when fewer seconds remain than the closing margin plus more_q."""
    return ep.time_left() < END_MARGIN + more_q


def time_for(ep: StoreSession, more_q: float) -> bool:
    """True when more seconds remain than the closing margin plus more_q."""
    return ep.time_left() > END_MARGIN + more_q


class RequestReading:
    """The shopper's request and the store's tools, read once at the start."""

    def read_brief(self) -> bool:
        prompt = self.intake_text()
        ask = self.models.json_reply
        intake = ask([INTAKE_MODEL], INTAKE_RULES, prompt, 2500)
        if not (isinstance(intake, dict) and isinstance(intake.get("tools"), dict)):
            # hidden thinking counts against max_tokens, so a long request (several products) can run out of room
            # before its JSON closes: the retry gets room enough to finish it (the spare thinks at length, so it
            # only comes last and with that room: inside the short room it ran out and cost the most time)
            intake = ask([INTAKE_MODEL, FAST_MODEL, BACKUP_MODEL], INTAKE_RULES, prompt, INTAKE_TOKENS)
        if not isinstance(intake, dict):
            self.trace_event("setup_failed")
            return False
        tools = intake.get("tools")
        offered = tools if isinstance(tools, dict) else {}
        self.roles = {duty: spec for duty, spec in offered.items()
                      if isinstance(spec, dict) and isinstance(spec.get("name"), str) and spec["name"] in self.schemas}
        self.align_roles()
        for duty in ("search_text", "open", "checkout", "decline", "message"):
            if duty not in self.roles:
                self.trace_event("role_missing", role=duty)
        self.read_items(intake.get("items"))
        self.read_limits(intake)
        for product in self.products:
            self.add_owned_fit(product)
        self.conditions = self.read_conditions(intake.get("conditions"))
        self.read_procedure(intake.get("procedure"))
        summary = []
        for item in self.products:
            row = (item.name, item.quantity, item.demand_lines(), item.hints)
            summary.append(row if item.unit_cap is None else row + (item.unit_cap,))
        named = {duty: spec.get("name") for duty, spec in self.roles.items()}
        self.trace_event("plan", items=summary, cap=self.budget, cheapest=self.cheapest, nearest=self.nearest_wanted,
                         messages=self.message_cap, steps=self.step_cap, roles=named)
        return True

    def read_items(self, listed: object) -> None:
        """The products the reading lists (one stand-in from the request text when it lists none), tidied and
        recounted."""
        rows = listed if isinstance(listed, list) else []
        self.products = [Product(row, slot) for slot, row in enumerate(rows) if isinstance(row, dict)]
        if not self.products:
            stand_in = {"name": "the product", "category": "", "search_terms": [self.request_text[:60]]}
            self.products = [Product(stand_in, 0)]
        for product in self.products:
            self.tidy_product(product)
        self.recheck_quantities()

    def read_limits(self, intake: dict) -> None:
        """Cheapest and stock rules, best-of measure, the ending when nothing fits, budget and unit caps, message and
        call budgets, subjects to ask about and fit rules."""
        get = intake.get
        self.cheapest = bool(get("cheapest"))
        self.buyable_only = bool(get("in_stock_only", True))
        best = get("best")
        self.measure = best if isinstance(best, str) and best else None
        when_none = str(get("nothing_fits") or "").strip().lower()
        # "nearest": with no full fit, order the closest listing
        self.nearest_wanted = when_none == "nearest"
        # the request says how to end when nothing fits, so its decline is stated
        self.decline_stated = when_none == "decline" or bool(get("decline_rule"))
        self.budget = to_number(get("total_cap"))
        written = {float(fig) for fig in re.findall(r"\d+(?:\.\d+)?", self.request_text.replace(",", ""))}
        if self.budget is not None:
            for product in self.products:
                cap = product.unit_cap
                if cap is None or cap in written:
                    continue
                # a per-unit cap whose figure the request never writes is the reader's own arithmetic, not the
                # shopper's limit; the total still bounds the basket (Arena, race-15 world: the only two such caps,
                # read from totals the request gave, priced every listing out and both tasks declined)
                product.unit_cap = None
                self.trace_event("unwritten_cap_dropped", item=product.name, cap=cap)
        msgs = get("message_budget")
        self.message_cap = int(msgs) if is_count(msgs) and msgs >= 0 else None
        calls = get("call_budget")
        if is_count(calls) and 0 < calls < self.step_cap:
            self.step_cap = int(calls)
        self.topics = [str(topic) for topic in get("ask_about") or [] if topic]
        self.fit_rules = [rule for rule in get("fit_rules") or [] if isinstance(rule, dict)]

    def read_procedure(self, raw: object) -> None:
        """Steps the request demands (six at most), less message steps that restate an ask_about subject."""
        steps = [step for step in raw or [] if isinstance(step, dict) and step.get("goal")][:6]
        for step in steps:
            step["moment"] = str(step.get("moment") or "before_finish").lower()
            step["kind"] = "message" if str(step.get("kind") or "").lower() == "message" else "tool"
        invited = [step for step in steps if self.echoes_topic(step)]
        if invited:
            # a message step that restates a subject the reading also put under ask_about is the request's
            # invitation to be asked: the questions answer it, and a reserved closing message would cost one of them
            steps = [step for step in steps if step not in invited]
            self.trace_event("steps_dropped", goals=[str(step.get("goal"))[:120] for step in invited])
        self.steps_demanded = steps
        self.needs_approval = any(step["kind"] == "message" and step["moment"] in ("before_order", "before_finish")
                                  for step in steps)

    def intake_text(self) -> str:
        tools = [dict(name=func.get("name"), description=func.get("description", ""), parameters=func.get("parameters"))
                 for func in map(function_of, self.tool_entries)]
        opening = "" if self.opening is None else "\n\nOPENING OBSERVATION:\n" + compact(self.opening, 1500)
        return "REQUEST:\n" + self.request_text + opening + "\n\nTOOLS:\n" + as_json(tools)

    def recheck_quantities(self) -> None:
        """Counts above one get a separate second reading, since a story number or a bundle word can pass for a
        quantity. The second reading can only lower a count."""
        if all(item.quantity <= 1 for item in self.products):
            return
        shown = as_json([dict(name=item.name, category=item.kind, quantity=item.quantity) for item in self.products])
        reply = self.models.json_reply([INTAKE_MODEL, FAST_MODEL], COUNT_RULES,
                                       "REQUEST:\n" + self.request_text + "\n\nITEMS AS READ:\n" + shown, 1500)
        rows = reply.get("items") if isinstance(reply, dict) else None
        if not isinstance(rows, list) or len(rows) != len(self.products):
            return
        lowered = [(item, row) for item, row in zip(self.products, rows)
                   if isinstance(row, dict) and is_count(row.get("quantity")) and 1 <= row["quantity"] < item.quantity]
        for item, row in lowered:
            now = int(row["quantity"])
            self.trace_event("count_checked", item=item.name, was=item.quantity, now=now,
                             words=str(row.get("words") or "")[:80])
            item.quantity = now

    def tidy_product(self, prod: Product) -> None:
        # values the request gives for a referenced page are that page's, not this item's
        ref_pairs = {(plain_text(key), plain_text(val))
                     for need in prod.demands if need.reference and isinstance(need.reference.get("attributes"), dict)
                     for key, val in need.reference["attributes"].items()}
        if ref_pairs:
            prod.demands = [need for need in prod.demands
                            if need.reference or (plain_text(need.prop), plain_text(need.value)) not in ref_pairs]
            ref_values = [val for _, val in ref_pairs if val]
            kept = [query for query in prod.planned_queries if not any(val in plain_text(query) for val in ref_values)]
            prod.planned_queries = kept or [prod.kind]
        # a plain "is" value made only of the category's words restates the kind; a word beyond it narrows the kind
        noun_words = word_set(prod.kind)
        kept_needs = []
        for need in prod.demands:
            words = word_set(need.value)
            restates = need.op == "is" and not need.reference and words and words <= noun_words
            if not restates:
                kept_needs.append(need)
        prod.demands = kept_needs
        # every stated value is also a query on its own (fewer words match more listings)
        for need in prod.demands:
            if need.op not in ("is", "one of", "contains") or not need.value:
                continue
            for part in map(str.strip, need.value.split("|")):
                forms = (part, part.replace("-", " ")) if "-" in part else (part,)
                for phrase in forms:
                    if not phrase or len(phrase) > 40:
                        continue
                    if plain_text(phrase) not in {plain_text(query) for query in prod.planned_queries}:
                        prod.planned_queries.append(phrase)

    def echoes_topic(self, step: dict) -> bool:
        """True for a message step whose goal holds at least half the longer words of one ask_about subject."""
        if step.get("kind") != "message":
            return False
        goal = {w for w in word_set(step.get("goal")) if len(w) >= 4}
        subjects = ({w for w in word_set(topic) if len(w) >= 4} for topic in self.topics)
        return any(len(words) >= 2 and 2 * len(words & goal) >= len(words) for words in subjects)

    def align_roles(self) -> None:
        """Fit cart and ending roles to the real schemas: a cart tool taking a list of lines becomes cart_set, unknown
        argument names yield to the schema's own, and an ending tool needing a choice covers both endings."""
        roles = self.roles
        for duty in ("cart_add", "cart_set"):
            if roles.get(duty):
                self.align_cart_role(duty, roles[duty])
        # other roles: an argument name the schema lacks becomes its argument of the right type
        plain_roles = {"message": ("text", "string"), "open": ("id", "string"), "facts": ("id", "string"),
                       "compare": ("ids", "array"), "search_text": ("query", "string")}
        for duty, (slot, type_name) in plain_roles.items():
            spec = roles.get(duty) or {}
            schema = self.schemas.get(spec.get("name")) or {}
            props = schema.get("properties") or {}
            if props and spec.get(slot) not in props:
                skip = mode_argument(spec)
                typed = [key for key, val in props.items() if key != skip and is_plain_type(val, type_name)]
                spec[slot] = first_named(schema.get("required") or [], typed)
        # an ending tool with a required choice ends the task either way
        for held, other in (("checkout", "decline"), ("decline", "checkout")):
            tool = (roles.get(held) or {}).get("name")
            if tool and other not in roles and required_choices(self.schemas.get(tool) or {}):
                roles[other] = {"name": tool}

    def align_cart_role(self, duty: str, spec: dict) -> None:
        tool = spec["name"]
        props = tool_props(self, tool)
        lines_arg = next((key for key in props if is_object_list(props[key])), None)
        if lines_arg is not None:
            entry = props[lines_arg]["items"].get("properties") or {}
            own = spec if duty == "cart_set" else {}
            id_key, qty_key = (own.get(key) if own.get(key) in entry else first_of_type(entry, typ)
                               for key, typ in (("id", "string"), ("qty", "integer")))
            if id_key:
                self.roles["cart_set"] = dict(name=tool, lines=lines_arg, id=id_key, qty=qty_key)
            if duty == "cart_add":
                self.roles.pop("cart_add", None)
        elif duty == "cart_add":
            required = (self.schemas.get(tool) or {}).get("required") or []
            if spec.get("id") not in props:
                texts = [key for key, val in props.items() if is_plain_type(val, "string")]
                spec["id"] = first_named(required, texts)
            if spec.get("qty") and spec["qty"] not in props:
                spec["qty"] = first_of_type(props, "integer")
            if not spec.get("qty"):
                # an unclaimed whole-number argument is the quantity, whatever it is called
                skip = (spec.get("id"), mode_argument(spec))
                counts = (key for key in [*required, *props] if key not in skip and isinstance(props.get(key), dict))
                spec["qty"] = next((key for key in counts if props[key].get("type") in ("integer", "number")), None)
            if not spec.get("id"):
                self.roles.pop("cart_add", None)

    def closing_args(self, role: str) -> dict:
        """Arguments that select this ending on its tool: the intake's values snapped to the schema's listed values,
        and any still-open required choice decided by one cheap model call."""
        spec = self.roles.get(role) or {}
        schema = self.role_schema(role)
        props = schema.get("properties") or {}
        given = spec.get("args")
        selected = {}
        for key, val in (given if isinstance(given, dict) else {}).items():
            if key not in props or val in (None, ""):
                continue
            listed = props[key].get("enum")
            if isinstance(listed, list) and listed and val not in listed:
                best = max(listed, key=lambda e: overlap(str(e), str(val)))
                # below half an overlap nothing listed is meant, and the choice call below decides
                val = best if overlap(str(best), str(val)) >= 0.5 else None
            if val is not None:
                selected[key] = val
        open_keys = [key for key in required_choices(schema) if key not in selected]
        if not open_keys:
            return selected
        chosen = self.closing_choices.get(role)
        if chosen is None:
            chosen = self.choose_closing(role, spec.get("name"), {key: props[key] for key in open_keys})
        picks = {key: chosen.get(key) for key in open_keys}
        selected.update((key, val) for key, val in picks.items() if val in props[key]["enum"])
        return selected

    def choose_closing(self, role: str, tool: str | None, open_props: dict) -> dict:
        """One cheap model call picks the listed values that select this ending."""
        aim = "place the order for what is in the cart" if role == "checkout" else "end the task without buying anything"
        prompt = "\n".join([f"TOOL: {tool}", f"DESCRIPTION: {self.tool_description(tool or '')}",
                            "ARGUMENTS: " + as_json(open_props), "GOAL: " + aim])
        reply = self.models.json_reply([FAST_MODEL, BACKUP_MODEL], CHOICE_RULES, prompt, 300)
        if not isinstance(reply, dict) or not reply:
            return {}
        # only a useful answer is remembered; a failed call is retried next time
        self.closing_choices = {**self.closing_choices, role: reply}
        return reply

    def read_conditions(self, raw: object) -> list:
        """Demands on one product that depend on another product's outcome, as {"src", "prop", "values",
        "branches": [(outcome, product, demand spec)]}."""
        tests = [(rule, rule.get("if")) for rule in (raw if isinstance(raw, list) else []) if isinstance(rule, dict)]
        rules = []
        for rule, test in tests:
            if not (isinstance(test, dict) and test.get("property")):
                continue
            source = self.product_named(test.get("item"))
            listed = test.get("values")
            values = [str(val) for val in (listed if isinstance(listed, list) else [listed]) if val]
            arms = self.condition_arms(rule, source)
            if source is None or not values or not arms:
                continue
            rules.append({"src": source, "prop": str(test["property"]), "values": values, "branches": arms})
            for _, target, branch in arms:
                # search for the values a branch may demand, so listings carrying them are at hand
                pieces = [val for val in map(str.strip, str(branch["value"]).split("|")) if val and len(val) <= 40]
                for val in pieces:
                    if all(plain_text(val) not in plain_text(query) for query in target.planned_queries):
                        target.planned_queries.append(f"{val} {target.noun}"[:60])
        if rules:
            shown = []
            for rule in rules:
                arms = [(outcome, item.name, branch.get("property"), branch.get("value"))
                        for outcome, item, branch in rule["branches"]]
                shown.append((rule["src"].name, rule["prop"], rule["values"], arms))
            self.trace_event("conditions", rules=shown)
        return rules

    def condition_arms(self, rule: dict, source: Product | None) -> list:
        """A condition's then / otherwise demands, each on a requested product other than the condition's own."""
        arms = []
        for key, outcome in (("then", True), ("otherwise", False)):
            branch = rule.get(key)
            if isinstance(branch, dict) and branch.get("property") and branch.get("value"):
                target = self.product_named(branch.get("item"))
                if target is not None and target is not source:
                    arms.append((outcome, target, branch))
        return arms

    def product_named(self, name: object) -> Product | None:
        wanted = str(name or "")
        if not (wanted and self.products):
            return None
        scored = [(max(overlap(item.name, wanted), overlap(item.kind, wanted)), item) for item in self.products]
        score, best = max(scored, key=lambda pair: pair[0])
        return best if score >= 0.2 else None

    def is_fit_clause(self, hint: object) -> bool:
        """A story hint quoting a model code from the request: it ties the product to a device the shopper owns, which
        the listing pages settle; it is no hidden taste of the shopper."""
        said = self.request_text.lower()
        return len(self.products) > 1 and any(code.lower() in said for code in CODE_WORD.findall(str(hint)))

    def outside_hosts(self, prod: Product) -> list[str]:
        """Hosts of fit rules that tie this product (their accessory) to a device no requested product is."""
        hosts = []
        for link in self.fit_rules:
            part, host = (str(link.get(key) or "") for key in ("accessory", "host"))
            if not (part and host) or any(overlap(item.name, host) >= 0.2 for item in self.products):
                continue
            if max(self.products, key=lambda item: overlap(item.name, part)) is prod and overlap(prod.name, part) >= 0.2:
                hosts.append(host)
        return hosts

    def bound_to_owned(self, prod: Product) -> bool:
        """Several products, and this one must fit a device the shopper is not buying (a fit rule whose host is no
        requested product, or a hint with that clause). Like a part fitting a host in the cart, the pages
        settle it: in race 13 no hidden need on such parts concerned their fit."""
        if len(self.products) < 2:
            return False
        return bool(self.outside_hosts(prod)) or any(self.is_fit_clause(hint) for hint in prod.hints)

    def owned_device_codes(self, prod: Product) -> list[str]:
        """Model codes of devices this product must fit without being bought: codes the request names, taken from
        a fit rule whose host is no requested product, else (several products) from a hint with the fit
        clause. A host named without a code yields none."""
        said = self.request_text.upper()
        found = [code for host in self.outside_hosts(prod) for code in MODEL_CODE.findall(host.upper()) if code in said]
        if not found:
            found = [code for hint in prod.hints if self.is_fit_clause(hint)
                     for code in MODEL_CODE.findall(str(hint).upper()) if code in said]
        own_words = (prod.name + " " + prod.kind).upper()
        return [code for code in dict.fromkeys(found) if code not in own_words]

    def add_owned_fit(self, prod: Product) -> None:
        """Add a requirement that the product fit the shopper's own device (named by model code), judged on every
        listing (code lists first, see outside_fit_codes) but never asked about (fixed_properties)."""
        codes = self.owned_device_codes(prod)
        if not codes:
            return
        listed = " / ".join(codes)
        need = Demand("compatibility", "fits", "the shopper's own %s (owned already, not part of this order)" % listed,
                      origin="fit_out")
        need.owned_codes = set(codes)
        prod.demands.append(need)
        self.trace_event("outside_fit", item=prod.name, req=need.as_line())
class Searching:
    """Keyword and field searches, paging and row screening."""

    def query_text(self, prod: Product, role: str, arg: str, query: str) -> str:
        """The query as it is sent: value words ahead of category words (values decide matches, and some stores read
        only the first words); over the length limit category words are dropped first, then it is cut at a word."""
        cat_bag = word_set(prod.kind) | word_set(prod.name)
        values, cat_words = [], []
        for word in query.split():
            own = word_set(word)
            (cat_words if own and own <= cat_bag else values).append(word)
        joined = " ".join(values + cat_words)
        limit = arg_schema(self, role, arg).get("maxLength")
        if isinstance(limit, int) and len(joined) > limit:
            kept = list(values)
            for word in cat_words:
                # category words come back in order while they still fit
                if len(" ".join(kept + [word])) <= limit:
                    kept.append(word)
            return self.clip_arg(role, arg, " ".join(kept) or joined)
        return joined

    def sent_form(self, prod: Product, query: str) -> str:
        """A keyword query as the text search tool receives it (the key its match count is kept under)."""
        arg = (self.roles.get("search_text") or {}).get("query") or ""
        return self.query_text(prod, "search_text", arg, query)

    def search_args(self, prod: Product, duty: str, spec: dict, query: str | None, fields: dict | None) -> dict | None:
        """One page's arguments before paging: the fixed mode argument, then the query text or the field values.
        None when a field value names a category the store does not list."""
        mode_arg = spec.get("mode")
        args = {mode_arg["arg"]: mode_arg.get("value")} if isinstance(mode_arg, dict) and mode_arg.get("arg") else {}
        if query is not None:
            args[spec["query"]] = self.query_text(prod, duty, spec["query"], query)
            return args
        for field, val in (fields or {}).items():
            param = spec.get(field)
            if param and val not in (None, ""):
                spelled = self.enum_spelling(duty, param, val)
                if spelled is None:
                    return None
                args[param] = self.clip_arg(duty, param, spelled) if isinstance(spelled, str) else spelled
        return args

    def take_rows(self, prod: Product, raw_rows: list, query_key: str | None, own_rows: set) -> tuple[list, int, int]:
        """Merge one page's rows into the product's pool: (listings read from the rows, how many were new to the
        pool, how many this query had not listed before). query_key files the rows under a keyword query."""
        offers, new_n, fresh_n = [], 0, 0
        for raw in raw_rows:
            offer = self.listing_from(raw, "row")
            if offer is None:
                continue
            offers.append(offer)
            if offer.ident not in prod.seen:
                new_n += 1
            if offer.ident not in own_rows:
                own_rows.add(offer.ident)
                fresh_n += 1
            self.merge_listing(prod, offer, opened=False)
            if query_key is not None:
                prod.rows_by_query.setdefault(query_key, set()).add(offer.ident)
        return offers, new_n, fresh_n

    def run_search(self, prod: Product, query: str | None = None, fields: dict | None = None, pages: int = 1,
                   budget: int | None = None) -> int:
        """Search and turn pages; returns how many listings were new to the product's pool."""
        by_text = query is not None
        duty = "search_text" if by_text else "search_fields"
        spec = self.roles.get(duty)
        if not spec:
            return 0
        mode = self.paging_style(duty)
        sig_key = duty + ":" + (query if by_text else json.dumps(fields or {}, sort_keys=True))
        state = prod.paging.get(sig_key) or {}
        if state.get("done"):
            return 0
        cursor = state.get("token")
        first_page = state.get("page", 0) + 1
        if mode is None and first_page > 1:
            return 0
        floor = 0 if budget is None else budget
        own_rows = set(prod.rows_by_query.get(sig_key) or ()) if by_text else set()
        top_price = state.get("top_price")
        added, finished, done_page = 0, False, first_page - 1
        for page in range(first_page, first_page + pages):
            if self.steps_left() <= floor or self.reads_cutoff() > self.time_left():
                break
            if by_text and not spec.get("query"):
                return 0
            call = self.search_args(prod, duty, spec, query, fields)
            if call is None:
                # the store lists no such category: this field search cannot run
                prod.paging[sig_key] = dict(page=0, token=None, done=True)
                return added
            if page > 1:
                step = page_arg(mode, page, self.page_size, cursor)
                if step is None:
                    finished = True
                    break
                call[spec["page"]] = step
                peak = arg_schema(self, duty, spec["page"]).get("maximum")
                if isinstance(step, int) and isinstance(peak, (int, float)) and step > peak:
                    # the store will not page this far: what lies beyond stays hidden
                    prod.more_hidden = True
                    finished = True
                    break
            result = self.call_tool(spec["name"], call)
            done_page = page
            if not result["ok"]:
                finished = True
                self.trace_event("search_error", error=result["error"])
                break
            rows = find_rows(result["obs"]) or []
            cursor, matches = page_meta(result["obs"])
            if rows:
                self.page_size = max(self.page_size, len(rows))
            if page == 1 and by_text and (matches is not None or not rows):
                self.note_match_count(call[spec["query"]], matches if matches is not None and rows else 0)
            offers, new_n, fresh_n = self.take_rows(prod, rows, sig_key if by_text else None, own_rows)
            added += new_n
            if len(offers) > 1:
                # rows with price and stock leave little for the page to add when a table tool exists
                self.rows_complete = all(o.price is not None and o.available is not None for o in offers)
            if prod.position != 99 and prod.categories_ok is None:
                self.settle_categories(prod)
            watch_price_order(self, offers)
            if offers:
                top_price = max((o.price for o in offers if o.price is not None), default=None)
            # a short page, a page holding only rows this query had already listed, or every reported match shown
            # ends the paging
            if (not rows or len(rows) < self.page_size or (page > 1 and offers and not fresh_n)
                    or (matches is not None and page * max(len(rows), 1) >= matches)):
                finished = True
                break
            if mode == "token" and not cursor:
                if matches is not None and page * len(rows) < matches:
                    # more matches than the store would list
                    prod.more_hidden = True
                finished = True
                break
        prod.paging[sig_key] = {"page": max(done_page, 0), "token": cursor, "done": finished or mode is None,
                                "top_price": top_price}
        return added

    def note_match_count(self, query: str, count: int) -> None:
        """Remember a query's match count. Two queries where one's words include the other's reveal how the store
        matches: more words and fewer matches means every word must match."""
        mine = frozenset(all_words(query))
        if not mine:
            return
        for other, other_n in self.match_counts.items():
            theirs = frozenset(all_words(other))
            if len(theirs) == len(mine) or not theirs & mine:
                continue
            pairs = ((theirs, other_n), (mine, count))
            (small, small_n), (big, big_n) = pairs if len(theirs) < len(mine) else pairs[::-1]
            verdict = None
            if small < big and big_n > small_n:
                verdict = False
            elif big_n * 10 < small_n * 3 and (small < big or len(big) >= len(small) + 2):
                # far fewer matches for more words (a small dip can be the store's count cap)
                verdict = True
            if verdict is not None and verdict is not self.all_words_needed:
                self.all_words_needed = verdict
                self.trace_event("search_words", all_words=verdict, small=small_n, big=big_n)
        self.match_counts[query] = count

    @staticmethod
    def value_groups(prod: Product) -> list[list[str]]:
        """Plain required values as keyword groups, one per requirement with its alternatives, shopper answers first."""
        ranked = []
        for dm in prod.firm_demands():
            if dm.op in ("is", "one of", "contains") and dm.value:
                # part codes are matched through fitment wording, never typed into a keyword query
                options = [v.strip() for v in dm.value.split("|") if word_set(v) and not CODE_WORD.search(v)]
                if options and len(options) <= 4 and all(len(o) <= 24 for o in options):
                    ranked.append((dm.origin != "shopper", options))
        return [options for _, options in sorted(ranked, key=lambda pair: pair[0])]

    @staticmethod
    def groups_carried(listing: Listing, groups: list[list[str]]) -> int:
        """Number of value groups found in the listing's own words."""
        own = [listing.heading, listing.blurb or "", listing.kind, listing.maker, listing.vendor]
        words = all_words(" ".join(own + [f"{kn} {val}" for kn, val in listing.table.items()]))
        return sum(1 for bunch in groups if any(all_words(variant) <= words for variant in bunch))

    def value_queries(self, prod: Product) -> list[str]:
        groups = self.value_groups(prod)
        if not groups:
            return []
        noun = prod.kind if len(prod.noun) > 22 else prod.noun
        # a store ranking by any word credits every listing with the category words: then values go alone
        suffix = [] if self.all_words_needed is False else [noun]
        out = []
        # where every word must match, word values go first without figures (a figure matches only in its own
        # spelling); a store ranking by matched words puts rows holding every value first, figures included
        worded = [grp[0] for grp in groups if not all(re.search(r"\d", v) for v in grp)]
        if self.all_words_needed and worded and len(worded) < len(groups):
            out.append(" ".join(worded[:3] + suffix))
        leaders = [grp[0] for grp in groups]
        for idx, bunch in enumerate(groups):
            others = leaders[:idx] + leaders[idx + 1:]
            for variant in bunch[:3]:
                qry = " ".join(([variant] + others)[:3] + suffix)
                if query_words(qry) not in {query_words(q) for q in out}:
                    out.append(qry)
            if len(out) >= 4:
                break
        return out

    def pending_value_queries(self, prod: Product) -> list[str]:
        """Value queries (see value_queries) that no query sent so far covers."""
        wanted = self.value_queries(prod)
        covered = [query_words(kn.split(":", 1)[1]) for kn in prod.paging if kn.startswith("search_text:")]
        covered += [query_words(q) for q in prod.value_queries_sent]
        out = []
        for qry in wanted:
            words = query_words(qry)
            if all(not words <= c for c in covered):
                out.append(qry)
        return out

    def search_values(self, prod: Product, keep_steps: int, first: str | None = None) -> bool:
        """Send the next value query (or the requested one if still pending), plus shorter variants when it matches
        nothing. True if a search ran."""
        pending = self.pending_value_queries(prod)
        if not pending or self.steps_left() <= keep_steps + 1:
            return False
        opening = not prod.value_queries_sent
        qry = first if first in pending else pending[0]
        steps_before = self.steps
        prod.value_queries_sent = list(prod.value_queries_sent) + [qry]
        self.trace_event("value_query", item=prod.name, query=qry)
        # the first value query pages as deep as the item's own queries; later ones less, unless every word counts
        if not opening:
            depth = 3 if self.all_words_needed else 2
        elif len(self.products) == 1:
            depth = 6
        else:
            depth = 3
        self.run_search(prod, query=qry, pages=depth, budget=keep_steps)
        self.collect_value_rows(prod, qry)
        if self.match_counts.get(self.sent_form(prod, qry)) == 0 and self.steps_left() > keep_steps + 1:
            # no listing has every word: try the values without the noun, then the first value with the noun
            noun_bag = word_set(prod.noun)
            bare = []
            for wrd in qry.split():
                own = word_set(wrd)
                if not own or not own <= noun_bag:
                    bare.append(wrd)
            lead_with_noun = " ".join([self.value_groups(prod)[0][0], prod.noun])
            for reduced in (" ".join(bare), lead_with_noun):
                # skip one the store would receive exactly as an earlier query
                if (not reduced or plain_text(reduced) == plain_text(qry) or self.steps_left() <= keep_steps + 1
                        or self.sent_form(prod, reduced) in self.match_counts):
                    continue
                prod.value_queries_sent.append(reduced)
                self.run_search(prod, query=reduced, pages=2, budget=keep_steps)
                self.collect_value_rows(prod, reduced)
                if self.match_counts.get(self.sent_form(prod, reduced)):
                    break
        return self.steps > steps_before

    def listed_whole(self, prod: Product, query: str) -> bool:
        """True when a value query went out with all its value words and its pages listed every reported match."""
        key = "search_text:" + query
        sent = self.sent_form(prod, query)
        total = self.match_counts.get(sent)
        if not total or not (prod.paging.get(key) or {}).get("done", False):
            return False
        if len(prod.rows_by_query.get(key) or ()) < total:
            return False
        return all_words(sent) >= all_words(query) - all_words(prod.noun)

    def collect_value_rows(self, prod: Product, query: str) -> None:
        """Add a value query's rows to the product's value ids; once every value query went out whole and listed
        all its matches, the value ids are complete."""
        prod.value_ids |= prod.rows_by_query.get("search_text:" + query) or set()
        wanted = self.value_queries(prod)
        if wanted and all(self.listed_whole(prod, qry) for qry in wanted):
            prod.values_complete = True

    def search_heard(self, prod: Product, keep_steps: int) -> bool:
        """Once per product, search a heard value as keywords: kind + a stated value + it, kind + it, then the value
        alone (an any-word store ranks the bare value best; value queries already covered all-words stores).
        True if a search ran."""
        heard, preset = [], []
        for rq in prod.firm_demands():
            if rq.value and rq.op in ("is", "one of"):
                (heard if rq.origin == "shopper" else preset).append(rq.value.split("|")[0].strip())
        if self.all_words_needed is True or prod.heard_search_done or not heard:
            return self.search_bounds(prod, keep_steps)
        prod.heard_search_done = True
        lead = heard[0]
        queries = [f"{prod.kind} {preset[0]} {lead}"] if preset else []
        queries += [f"{prod.kind} {lead}", lead]
        done = {query_words(q) for q in queries_sent(prod)}
        ran = False
        for qry in queries:
            if self.steps_left() <= keep_steps:
                break
            if query_words(qry) in done:
                continue
            ran = True
            self.run_search(prod, query=qry, pages=3, budget=keep_steps)
        return ran

    def send_bound_queries(self, prod: Product, queries: object, tried: set, floor: int, keep_steps: int) -> bool:
        """Send bound queries one page each, in turn, until one adds a listing; queries already tried are skipped and
        the steps stop at floor. True if one ran."""
        ran = False
        for qry in queries:
            if self.steps_left() <= floor:
                break
            if query_words(qry) in tried:
                continue
            ran = True
            self.trace_event("bound_query", item=prod.name, query=qry)
            if self.run_search(prod, query=qry, pages=1, budget=keep_steps + 1) > 0:
                break
        return ran

    def search_bounds(self, prod: Product, keep_steps: int, any_source: bool = False) -> bool:
        """Once per numeric bound the shopper revealed ("at least <n> <unit>"), search the figure as the product's
        listings write it, with a plain required value and the noun. Plain queries rank rows near the bound
        low, and under the cheapest rule the cheapest fit usually sits right at the bound. One page per query;
        the bare figure goes out only if the first adds nothing. True if a search ran."""
        terms, stated_for = [], {}
        for rq in prod.firm_demands():
            if rq.op not in ("at least", "at most") or "|" in rq.value:
                continue
            heard = rq.origin == "shopper"
            if heard:
                term = figure_as_titles(prod, rq.value)
            elif any_source:
                # a stated bound: the figure as the read property tables write it, else as the request states it
                # (a title's spelling, figure and unit joined, matched no table value in an all-words store)
                term = figure_as_tables(prod, rq.prop, rq.value) or rq.value.strip()
            else:
                continue
            if term and term not in prod.bounds_sent and term not in terms:
                terms.append(term)
                stated_for[term] = None if heard else rq
        if not terms or self.steps_left() <= keep_steps + 1:
            return False
        term = terms[0]
        prod.bounds_sent = list(prod.bounds_sent) + [term]
        tried = {query_words(q) for q in queries_sent(prod)}
        stated = stated_for[term]
        if stated is None:
            values = (rq.value.split("|")[0].strip() for rq in prod.firm_demands()
                      if rq.op in ("is", "one of", "contains") and rq.value and not re.search(r"\d", rq.value))
            lead = next((v for v in values if v and len(v) <= 24), None)
            queries = ([f"{lead} {term} {prod.noun}"] if lead else []) + [f"{term} {prod.noun}"]
            # one step stays free beyond the floor so a row the search lists can still be read
            return self.send_bound_queries(prod, queries, tried, keep_steps + 1, keep_steps)
        # the property's own name with the figure ranks the listings whose tables carry the figure first; the
        # rows near the bound follow on the next page
        first_w = next(([rq.value] for rq in prod.firm_demands()
                        if rq.op == "is" and rq.value and len(rq.value) <= 24 and not CODE_WORD.search(rq.value)), [])
        qry = " ".join([prod.noun] + first_w + [stated.prop, term])[:60]
        if query_words(qry) in tried or self.steps_left() <= keep_steps + 1:
            return False
        self.trace_event("bound_query", item=prod.name, query=qry)
        self.run_search(prod, query=qry, pages=2, budget=keep_steps + 1)
        hits_n = self.match_counts.get(self.sent_form(prod, qry))
        if hits_n is not None and hits_n < 6:
            # few rows hold the property's name with the figure: where every word must match, a title stating only
            # the figure is shut out (an Arena host listing wrote "350 cd/m2" in its title and never got listed), so
            # the figure goes with the noun alone, then the bare number
            num_m = re.match(r"\s*(\d+(?:[.,]\d+)?)", term)
            bare = [f"{term} {prod.noun}"] + ([f"{num_m.group(1)} {prod.noun}"] if num_m else [])
            self.send_bound_queries(prod, dict.fromkeys(bare), tried | {query_words(qry)}, keep_steps + 2, keep_steps)
        return True

    def suggest_queries(self, prod: Product) -> list[str]:
        """Ask the fast model for new keyword queries when the product's own queries found too few fits."""
        shown = sorted(prod.seen.values(), key=lambda cand: (not cand.fully_read, cand.ident))[:14]
        any_read = any(cand.fully_read for cand in shown)
        fit_ids = {fx.ident for fx in prod.fits(include_unsure=True)} if any_read else set()
        rows_out = [{"title": cand.heading[:110], "fits": (cand.ident in fit_ids) if cand.fully_read else None}
                    for cand in shown]
        attempted = sorted(queries_sent(prod))
        qname = (self.roles.get("search_text") or {}).get("query") or ""
        longest = (self.role_schema("search_text").get("properties") or {}).get(qname, {}).get("maxLength")
        heard_vals = "\n- ".join(prod.demand_lines()) or "none"
        message = (f"PRODUCT: {prod.name} (category: {prod.kind})\nREQUIREMENTS:\n- " + heard_vals
                   + f"\nTRIED: {as_json(attempted)}\nLISTINGS SEEN: {as_json(rows_out)}")
        if longest:
            message += f"\nMAX QUERY LENGTH: {longest} characters"
        answer_d = quick_json(self, QUERY_RULES, message, 700)
        known = {plain_text(tt) for tt in attempted + prod.planned_queries}
        incoming = []
        for raw in (answer_d.get("queries") or []) if isinstance(answer_d, dict) else []:
            qry = str(raw).strip()
            if qry and len(qry) <= 60 and plain_text(qry) not in known:
                known.add(plain_text(qry))
                incoming.append(qry)
        self.trace_event("suggested", item=prod.name, queries=incoming[:5])
        return incoming[:5]

    def is_kind(self, prod: Product, listing: Listing) -> bool:
        if not listing.kind or not prod.kind:
            return True
        label = plain_text(listing.kind)
        if prod.categories_ok:
            return label in prod.categories_ok
        target = plain_text(prod.kind)
        return label in target or target in label or overlap(label, target) >= 0.5

    def settle_categories(self, prod: Product) -> None:
        """Decide with one cheap model call which seen category labels are the requested kind."""
        if prod.categories_ok is not None:
            return
        labels = sorted({o.kind for o in prod.seen.values() if o.kind})
        if not labels:
            return
        flat = {plain_text(label) for label in labels}
        if len(labels) == 1:
            only, target = plain_text(labels[0]), plain_text(prod.kind)
            # a single foreign label leaves the word rule in force; more labels may still come
            if overlap(labels[0], prod.kind) >= 0.3 or target in only or only in target:
                prod.categories_ok = flat
            return
        prompt = "\n".join([f"REQUESTED PRODUCT: {prod.name} (category wording: {prod.kind})",
                            f"REQUEST EXCERPT: {self.request_text[:600]}",
                            "STORE CATEGORY LABELS SEEN: " + json.dumps(labels, ensure_ascii=False)])
        answer_d = quick_json(self, CATEGORY_RULES, prompt, 500)
        given = answer_d.get("labels") if isinstance(answer_d, dict) else None
        if not isinstance(given, list):
            return
        chosen = {plain_text(label) for label in given if isinstance(label, str) and plain_text(label) in flat}
        if chosen:
            prod.categories_ok = chosen
            self.trace_event("labels", item=prod.name, labels=sorted(chosen))

    def candidates(self, prod: Product) -> list[Listing]:
        rank = {"yes": 0, "no": 2}
        kept = [o for o in prod.seen.values()
                if not (self.buyable_only and o.available is False) and not over_cap(self.budget, o, prod.quantity)
                and self.is_kind(prod, o)]
        return sorted(kept, key=lambda o: (rank.get(o.screen, 1), sort_price(o)))

    def screen_rows(self, prod: Product, listings: list[Listing]) -> None:
        waiting = [o for o in listings if not (o.fully_read or o.screened) and o.screen == "maybe"]
        if len(waiting) < 8 or not prod.demand_lines():
            return
        lot = waiting[:30]
        rows_r = [dict(n=number, title=o.heading[:140], text=(o.blurb or "")[:ROW_CHARS], category=o.kind,
                       price=o.price) for number, o in enumerate(lot, 1)]
        prompt = ("REQUESTED PRODUCT: " + f"{prod.kind}" + "\nREQUIREMENTS:\n- " + "\n- ".join(prod.demand_lines())
                  + "\n\nROWS:\n" + json.dumps(rows_r, ensure_ascii=False))
        # numbers in, numbers out: a short answer comes back fast
        answer_d = quick_json(self, SCREEN_RULES, prompt, TRIAGE_TOKENS)
        for o in lot:
            o.screened = True
        if isinstance(answer_d, dict):
            for verdict_d in ("no", "yes"):
                picks = [str(n) for n in answer_d.get(verdict_d) or [] if isinstance(n, (int, str))]
                for number in (int(p) for p in picks if p.strip().isdigit()):
                    if 1 <= number <= len(lot):
                        lot[number - 1].screen = verdict_d


def page_arg(mode: str | None, page: int, size: int, cursor: str | None) -> int | str | None:
    """The paging argument for a page; None when the store cannot be paged that far this way."""
    if mode == "offset":
        return (page - 1) * max(size, 1)
    if mode == "index":
        return page
    return cursor if mode == "token" and cursor else None


def page_meta(obs: object) -> tuple[str | None, int | None]:
    """(next-page cursor, reported match count) of one search result: the cursor is the last short text without
    spaces, the count the largest integer beside the rows (an offset is smaller, and may come first)."""
    if not isinstance(obs, dict):
        return None, None
    values = list(obs.values())
    tokens = [v for v in values if isinstance(v, str) and " " not in v and 0 < len(v) <= 20]
    counts = [v for v in values if isinstance(v, int) and not isinstance(v, bool)]
    return (tokens[-1] if tokens else None), max(counts, default=None)


def query_words(text: str) -> frozenset:
    """The word set two keyword queries are compared by."""
    return frozenset(all_words(text))


def watch_price_order(ep: Episode, rows_p: list[Listing]) -> None:
    """Spot a store that lists its rows cheapest first: two pages of at least four priced rows in rising order,
    and no page out of order (one race in fourteen behaved so)."""
    prices = [o.price for o in rows_p if o.price is not None]
    if len(prices) < 4:
        return
    tally = ep.price_page_tally
    rising = all(a <= b for a, b in zip(prices, prices[1:]))
    tally[0 if rising else 1] += 1
    sorted_now = tally[1] == 0 and tally[0] >= 2
    if sorted_now != ep.cheapest_first:
        ep.cheapest_first = sorted_now
        ep.trace_event("price_sorted", value=sorted_now, pages=list(tally))


def value_groups_hit(ep: Episode, pd: Product, vgroups: list[list[str]], offer: Listing) -> int:
    """How many value groups a row carries in its own words, or, in an all-words store, how many a value query
    listed it for."""
    if offer.ident in pd.value_ids:
        return len(vgroups)
    carried = ep.groups_carried(offer, vgroups)
    if ep.all_words_needed and carried < len(vgroups):
        listed_for = [all_words(key.split(":", 1)[1]) for key, ids in pd.rows_by_query.items() if offer.ident in ids]
        by_query = sum(any(all_words(v) <= words for v in grp for words in listed_for) for grp in vgroups)
        return max(carried, by_query)
    return carried


def field_filters(ep: Episode, pd: Product) -> dict:
    """Field-search filters for a product: its category, with a price ceiling from the budget share and its own cap."""
    filters = {"category": pd.kind}
    ceilings = [] if ep.budget is None else [ep.budget / max(pd.quantity, 1)]
    if pd.unit_cap is not None:
        ceilings.append(pd.unit_cap)
    if ceilings:
        filters["price_max"] = min(ceilings)
    return filters


def figure_as_titles(pd: Product, value: str) -> str | None:
    """Write a "<n> <unit>" value the way the product's seen titles and texts write that unit after a number
    (attached or spaced, whichever is commoner). None when no seen row uses the unit."""
    hit = re.match(r"\s*(\d+(?:[.,]\d+)?)\s*([A-Za-z]+)", str(value or ""))
    if not hit:
        return None
    number, unit = hit.group(1), hit.group(2).lower()
    if len(unit) > 3:
        unit_re = re.escape(unit[:max(3, len(unit) - 2)]) + r"[a-z]*\b"
    else:
        unit_re = re.escape(unit) + r"\b"
    pattern = re.compile(r"\d(\s?)(" + unit_re + r")", re.I)
    spellings = [(not mm.group(1), mm.group(2).lower()) for offer in pd.seen.values()
                 for mm in pattern.finditer(f"{offer.heading} {offer.blurb or ''}")]
    if not spellings:
        return None
    attached, written = max(sorted(set(spellings), reverse=True), key=spellings.count)
    return number + ("" if attached else " ") + written


def figure_as_tables(pd: Product, field: str, value: str) -> str | None:
    """Write a figure the way the read property tables write units after numbers in the required property
    (attached or spaced, the commoner way). None when no read table has the property with a figure."""
    hit = re.match(r"\s*(\d+(?:[.,]\d+)?)\s*([A-Za-z]+)", str(value or ""))
    if not hit:
        return None
    gaps = [mm.group(1) for offer in pd.seen.values() if offer.fully_read
            for kn, val in offer.table.items() if key_names_property(kn, field)
            for mm in re.finditer(r"\d(\s?)[A-Za-z]", str(val))]
    if not gaps:
        return None
    spaced = sum(1 for gap in gaps if gap)
    return hit.group(1) + (" " if 2 * spaced >= len(gaps) else "") + hit.group(2).lower()


def bound_search_due(ep: Episode, pd: Product) -> bool:
    """Once per product: it has a stated numeric bound, at least four of its listings were read and none fits.
    The figure is then searched before the remaining rows are read, since titles rarely show it."""
    due = (not pd.bound_searched and not pd.fits()
           and any(rq.op in ("at least", "at most") and re.search(r"\d", rq.value) for rq in pd.firm_demands())
           and sum(1 for cand in pd.seen.values() if cand.fully_read and ep.is_kind(pd, cand)) >= 4)
    if due:
        pd.bound_searched = True
    return due


def request_spans(ep: Episode, pd: Product) -> list[str]:
    """"<low>-<high>" spans in the request whose ends the product holds as two bounds on one property, spelled
    as the request spells them (the intake splits such a span into two bounds)."""
    lows, highs = {}, {}
    for rq in pd.firm_demands():
        amount = to_number(rq.value) if rq.op in ("at least", "at most") else None
        if amount is not None:
            (lows if rq.op == "at least" else highs).setdefault(plain_text(rq.prop), amount)
    spans = []
    for mm in SPAN_IN_TEXT.finditer(ep.request_text):
        low, high = float(mm.group(1)), float(mm.group(2))
        term_s = f"{mm.group(1)}-{mm.group(2)}"
        if term_s not in spans and any(lo == low and highs.get(fd) == high for fd, lo in lows.items()):
            spans.append(term_s)
    return spans


def search_span(ep: Episode, pd: Product, keep_steps: int) -> bool:
    """Once per product, search a stated span exactly as the request writes it; stores that keep the property as
    a span hold it verbatim. True if the search ran."""
    # races 7-14: a query of the span alone listed a paid listing on its first page in 55 of 56 sends, the span with
    # the product noun 19 of 25, with other words added 52 of 394
    if pd.span_searched or ep.steps_left() <= keep_steps + 1:
        return False
    spans = request_spans(ep, pd)
    if not spans:
        return False
    pd.span_searched = True
    span = spans[0]
    if any(query_words(tq) == query_words(span) for tq in queries_sent(pd)):
        return False
    ep.trace_event("span_query", item=pd.name, query=span)
    steps_before = ep.steps
    ep.run_search(pd, query=span, pages=1, budget=keep_steps + 1)
    return ep.steps > steps_before


def widen_search(ep: Episode, pd: Product, qlist: list[str], cursor: list[int], lim: object) -> str:
    """Next search for a product whose listed rows are all read, trying in turn: heard values as keywords, a
    stated span, the next planned query, a field search, more pages of the first queries, model-suggested
    queries. Returns "stop" once nothing is left."""
    if ep.search_heard(pd, lim()) or search_span(ep, pd, lim()):
        return "again"
    nxt = max(cursor[0], first_unsent(pd, qlist))
    if nxt >= len(qlist):
        cursor[0] = nxt
    else:
        cursor[0] = nxt + 1
        found_n = ep.run_search(pd, query=qlist[nxt], pages=6 if len(ep.products) == 1 else 3, budget=lim())
        if not found_n and nxt + 1 >= len(qlist):
            pd.exhausted = True
        return "again"
    if ep.roles.get("search_fields"):
        filters = field_filters(ep, pd)
        state = pd.paging.get("search_fields:" + json.dumps(filters, sort_keys=True)) or {}
        if not state.get("done"):
            ep.run_search(pd, fields=filters, pages=2, budget=lim())
            return "again"
    for qq in qlist[:2]:
        state = pd.paging.get("search_text:" + qq) or {}
        if not state.get("done") and ep.run_search(pd, query=qq, pages=2, budget=lim()) > 0:
            return "again"
    if not pd.asked_queries and ep.steps_left() > lim() + 1:
        pd.asked_queries = True
        suggested = ep.suggest_queries(pd)
        if suggested:
            pd.planned_queries.extend(suggested)
            qlist.extend(suggested)
            return "again"
    pd.exhausted = True
    return "stop"


def cheapest_exhausted(ep: Episode, pd: Product, best_p: float, all_seen: bool, qlist: list[str],
                       lim: object) -> str | None:
    """Cheapest rule and nothing cheaper left unread. Tries heard values, then a field search capped below the
    best price, then all required values as one query if the store hides matches, then the next page of
    the main query. "again" repeats the round, "stop" ends the search, None falls through."""
    if ep.search_heard(pd, lim()):
        return "again"
    if not all_seen:
        field_spec = ep.roles.get("search_fields")
        if best_p < 1e17 and field_spec and field_spec.get("price_max") and pd.swept_below != best_p \
                and ep.steps_left() > lim() + 2:
            # only listings cheaper than the best fit come back
            pd.swept_below = best_p
            below = {"category": pd.kind, "price_max": max(0.0, best_p - 0.01)}
            ep.run_search(pd, fields=below, pages=3, budget=lim())
            return "again"
        if pd.more_hidden and not pd.narrowed and ep.steps_left() > lim() + 2:
            pd.narrowed = True
            values = [rr.value.split("|")[0].strip() for rr in pd.firm_demands()
                      if rr.op in ("is", "one of") and rr.value]
            if values:
                for qq in (" ".join(values + [pd.kind]), " ".join(values)):
                    if ep.steps_left() <= lim() + 1:
                        break
                    ep.run_search(pd, query=qq, pages=2, budget=lim())
                return "again"
    # rows show prices, so paging on costs a step a page and only cheaper rows get read
    states = ((qq, pd.paging.get("search_text:" + qq)) for qq in qlist)
    open_q = next((qq for qq, st in states if st and not st.get("done")
                   and not (ep.cheapest_first and (st.get("top_price") or 0) >= best_p)), None)
    if open_q and ep.steps_left() > lim() + 2:
        steps_before = ep.steps
        ep.run_search(pd, query=open_q, pages=1, budget=lim() + 1)
        if ep.steps > steps_before:
            return "again"
    return "stop" if len(queries_sent(pd)) >= min(len(qlist), 2) else None


def ranked_ids(ep: Episode, models: list[str], prod: Product, rows_out: list[dict], chars: int) -> list[str]:
    """Listing ids in the order a ranking model put the rows by the request's measure (empty when the reply holds
    no order); chars caps the rows' JSON."""
    message = f"MEASURE: {ep.measure}\nPRODUCT: {prod.name}\nLISTINGS: " + compact(rows_out, chars)
    answer_d = ep.models.json_reply(models, RANK_RULES, message, RANK_TOKENS)
    if not isinstance(answer_d, dict):
        return []
    return [str(xv) for xv in answer_d.get("order") or []]


def order_by_measure(ep: Episode, prod: Product, unread: list[Listing], vgroups: list[list[str]]) -> None:
    """Sort unread rows by the request's measure of best, ranking each batch of new rows from title and text
    with one call; ranked rows keep their place and unrankable rows follow. Opening cheapest first left
    the strongest rows unread, because the search stops once enough fits are known."""
    # real race 14: 4 of 6 best-by episodes lost (the field's leaders 0-2), the bought fit beaten by a row never read
    positions = prod.measure_positions
    fresh = [cand for cand in unread if cand.ident not in positions and cand.screen != "no"][:24]
    if prod.measure_calls < 3 and len(fresh) >= 3:
        prod.measure_calls += 1
        rows_out = [{"id": cand.ident, "title": cand.heading[:120], "price": cand.price,
                     "description": str(cand.blurb or "")[:160]} for cand in fresh]
        order = ranked_ids(ep, [FAST_MODEL, BACKUP_MODEL], prod, rows_out, 7000)
        fresh_ids = {cand.ident for cand in fresh}
        start = len(positions)
        for pos, pid in enumerate(order):
            if pid in fresh_ids and pid not in positions:
                positions[pid] = start + pos
        ep.trace_event("best_order", item=prod.name, ranked=len(positions), lead=order[:3])
    unknown = len(positions) + len(unread)

    def place(cand: Listing) -> tuple:
        groups_hit = value_groups_hit(ep, prod, vgroups, cand) if vgroups else 0
        return cand.screen == "no", -groups_hit, positions.get(cand.ident, unknown), sort_price(cand)
    unread.sort(key=place)


def look_above_fits(ep: Episode, prod: Product, fit_list: list[Listing], unread: list[Listing],
                    keep_steps: int) -> bool:
    """Final pass with a measure of best, once per product. The search stops at a few read fits, yet a row seen
    only in results may beat them (titles often state the measure). Read fits and unread rows are ranked
    together and at most two unread rows ranked above the best fit are opened and judged. True if one
    was opened."""
    if prod.measure_looked or not unread or ep.steps_left() <= keep_steps + 2:
        return False
    prod.measure_looked = True
    pool = [cand for cand in unread if cand.screen != "no"][:10]
    if not pool:
        return False
    leaders = ep.rank_by_measure(prod, list(fit_list))[:3]
    rows_out = [{"id": cand.ident, "title": cand.heading[:120], "price": cand.price, "properties": cand.table,
                 "description": str(cand.blurb or "")[:200]} for cand in leaders + pool]
    order = ranked_ids(ep, [PLAN_MODEL, FAST_MODEL, BACKUP_MODEL], prod, rows_out, 9000)
    read_ids = {cand.ident for cand in leaders}
    cut = next((pos for pos, pid in enumerate(order) if pid in read_ids), len(order))
    pool_ids = {cand.ident for cand in pool}
    ahead = [pid for pid in order[:cut] if pid in pool_ids]
    ep.trace_event("best_look", item=prod.name, measure=ep.measure, ahead=ahead[:2])
    return bool(ahead) and bool(ep.read_listings(prod, ahead[:2], keep_steps=keep_steps))


def outside_first(ep: Episode, prod: Product, rows: list[Listing]) -> None:
    """Reorder unread rows in place for a product tied to a device the shopper owns: rows whose visible code lists
    name the device's code first, rows without a list next, rows whose list leaves it out last; rows the
    screen rejected stay at the back."""
    codes = {code for dm in prod.demands if dm.origin == "fit_out" for code in dm.owned_codes or ()}
    if len(rows) < 2 or not codes:
        return
    fitbook = fit_book(ep)

    def place(row: Listing) -> tuple[bool, int]:
        if stated_fits(fitbook, row.ident)[0] & codes:
            return row.screen == "no", 0
        return row.screen == "no", 2 if fitbook["own"].get(row.ident) else 1
    rows.sort(key=place)
class ListingReading:
    """Opening listings: pages, compare batches and property tables."""

    def tables_suffice(self) -> bool:
        """A table tool next to the page tool, no usable compare tool, and rows showing price and stock: a listing is
        judged from row plus table (one step), and pages are read only where they matter."""
        roles = self.roles
        if roles.get("compare") and not self.compare_off:
            return False
        return bool(roles.get("facts") and roles.get("open") and self.rows_complete)

    def read_by_table(self, prod: Product | None) -> bool:
        if prod is None or not self.tables_suffice():
            return False
        # a fit is read off both products' pages
        tied = any(prod is lhs or prod is host for lhs, host in self.fit_pairs())
        return not tied and not any(dm.reference for dm in prod.demands)

    def read_listings(self, prod: Product | None, ids: list, keep_steps: int = 0, page: bool = False) -> list:
        """Read listings in full (through the compare tool when there is one) and return them; page=True reads a
        listing's own page even where its table would do."""
        found = []
        queue = [lid for lid in ids if lid]
        if not queue:
            return found
        roles = self.roles
        table_tool, page_tool = roles.get("facts"), roles.get("open")
        side = None if self.compare_off else roles.get("compare")
        if side and not self.compare_checked:
            # honour a limit stated on the list argument
            self.compare_checked = True
            stated = arg_schema(self, "compare", side.get("ids") or "").get("maxItems")
            if isinstance(stated, int) and 2 <= stated < self.compare_size:
                self.compare_size = stated
        tables_only = not page and self.read_by_table(prod)

        def short(cost: int) -> bool:
            # true when this read would eat into the steps kept for later
            if not keep_steps:
                return False
            return self.steps_left() < keep_steps + cost

        def settle(listing: Listing) -> None:
            found.append(listing if prod is None else self.merge_listing(prod, listing, True))

        def table_read(lid: str) -> None:
            got = self.read_table_only(prod, lid)
            if got is not None:
                found.append(got)
        at, resent = 0, set()
        while at < len(queue) and self.steps_left() > 0 and self.time_left() > self.reads_cutoff():
            if tables_only or (self.compare_thin and table_tool and prod is not None):
                # each table on its own: compare rows carry none in this store
                if short(1):
                    break
                lid = queue[at]
                at += 1
                table_read(lid)
                continue
            if side and at + 1 < len(queue):
                if short(1):
                    break
                batch = queue[at:at + self.compare_size]
                reply = self.call_tool(side["name"], {side["ids"]: batch})
                rows = find_rows(reply["obs"])
                if not (reply["ok"] and rows):
                    if reply["error"] or reply["ok"]:
                        # whatever the error says, the tool may take fewer ids: halve, and below two stop using it
                        cap = min(self.compare_size, len(batch))
                        if cap > 2:
                            self.compare_size = max(2, cap // 2)
                            self.remember_store()
                        else:
                            self.compare_off = True
                            side = None
                        continue
                else:
                    listings = [o for o in (self.listing_from(r, "detail") for r in rows) if o is not None]
                    bare = bool(table_tool) and bool(listings) and not any(o.table for o in listings)
                    if bare and not self.compare_thin:
                        self.compare_thin = True
                        self.remember_store()
                        self.trace_event("thin_compare", tool=side["name"], facts_tool=table_tool.get("name"))
                    split = bare and prod is not None
                    for o in listings:
                        o.page_seen = True
                        if split:
                            # keep the row's notes now; its table is read below
                            self.merge_listing(prod, o, False)
                            continue
                        o.fully_read = True
                        settle(o)
                    if split:
                        for o in listings:
                            if short(1) or low_on_time(self, 0):
                                break
                            table_read(o.ident)
                    at += len(batch)
                    returned = {o.ident for o in listings}
                    missed = [lid for lid in batch if lid not in returned and lid not in resent]
                    if listings and missed and len(returned) < len(batch):
                        # fewer rows than ids: the tool caps the batch, the others go round again
                        if 2 <= len(returned) < self.compare_size:
                            self.compare_size = len(returned)
                            self.remember_store()
                        resent.update(missed)
                        queue[at:at] = missed
                    continue
            if not page_tool or short(2 if table_tool else 1):
                break
            lid = queue[at]
            at += 1
            reply = self.call_tool(page_tool["name"], {page_tool["id"]: lid})
            if not (reply["ok"] and isinstance(reply["obs"], dict)):
                continue
            one = self.listing_from(reply["obs"], "detail")
            if one is None:
                one = Listing(lid)
            one.page_seen = True
            known = None if prod is None else prod.seen.get(lid)
            if not one.table and table_tool and self.steps_left() > 0 and not (known is not None and known.table):
                one.table = self.fetch_table(lid) or {}
            one.ident = lid
            one.fully_read = True
            settle(one)
        return found

    def fetch_table(self, lid: str) -> dict | None:
        """A listing's property table from the facts tool (one step): {} if it has none, None if the call failed."""
        tool = self.roles.get("facts")
        if not tool:
            return None
        reply = self.call_tool(tool["name"], {tool["id"]: lid})
        record = reply["obs"]
        if not (reply["ok"] and isinstance(record, dict)):
            return None
        key = self.key_map("facts", record).get("facts")
        nested = record.get(key) if key else None
        if isinstance(nested, dict):
            return nonempty_table(nested)
        if any(v is not None and not isinstance(v, (str, int, float)) for v in record.values()):
            return {}
        return nonempty_table(record)

    def read_table_only(self, prod: Product, lid: str) -> Listing | None:
        """Read a listing from its table alone; the seen row provides title, price and text."""
        known = prod.seen.get(lid)
        table = self.fetch_table(lid)
        if table is None or (not table and known is None):
            return None
        fresh = Listing(lid)
        fresh.table = table
        return self.merge_listing(prod, fresh, True)

    def read_sample(self, prod: Product, allowance: int | None = None) -> None:
        """Read more listings of the product's kind so its property menu is representative (about twelve with a
        multi-read tool, five otherwise), within the steps it may spend."""
        reserve = self.closing_steps() + 2
        if "message" not in self.roles or self.steps_left() <= reserve:
            return
        floor_n = max(0 if allowance is None else self.steps_left() - allowance, reserve)
        goal = 12 if self.batch_reads() else 5

        def unread() -> list:
            rows = [c for c in prod.seen.values()
                    if not c.fully_read and c.available is not False and self.is_kind(prod, c)]
            rows.sort(key=lambda c: (c.screen == "no", c.screen != "yes", sort_price(c)))
            return rows
        while self.steps_left() > floor_n and self.time_left() > END_MARGIN + 20 + self.question_hold \
                and not self.spent_for_search():
            missing = goal - sum(1 for c in prod.seen.values() if c.fully_read and self.is_kind(prod, c))
            if missing <= 0:
                return
            rows = unread()
            if len(rows) < 2 and len(queries_sent(prod)) <= len(prod.planned_queries):
                self.run_search(prod, query=prod.kind, pages=1, budget=floor_n)
                rows = unread()
            if not rows:
                return
            width = self.compare_size if self.batch_reads() else (4 if self.compare_thin else 1)
            got = self.read_listings(prod, [c.ident for c in rows[:min(missing, width)]], keep_steps=floor_n)
            if not got:
                return
            self.judge(prod, read_pool(self, prod))

    def read_backups(self) -> None:
        """Cheapest rule, before the first cart add: read the cheapest unread rows priced up to the second known fit,
        one batch per product, so a pick that sells out or reprices at carting has a checked replacement."""
        self.backups_read = True        # its reserved step is released from now on
        if not all(self.buyable(wp) for wp in self.products):
            return                      # an item still without a fit needs the steps more
        for pd in self.products:
            if room_left(self) < 1 or low_on_time(self, 20):
                break
            # the price of the second known fit (of the only one when there is one)
            leaders = self.buyable(pd)[:2]
            bound = leaders[-1].price if leaders else None
            # rows a title screen rejected are read as well: the table decides and a title may contradict it
            rows = sorted((c for c in self.candidates(pd)
                           if not (c.fully_read or c.available is False or c.price is None)
                           and (bound is None or c.price <= bound) and not over_cap(self.budget, c, pd.quantity)),
                          key=lambda c: c.price)
            if not rows:
                continue
            got = self.read_listings(pd, [c.ident for c in rows[:self.compare_size]], keep_steps=self.closing_steps())
            self.trace_event("backups", item=pd.name, read=[c.ident for c in got], bound=bound)
            if got:
                self.judge(pd, got)


def read_pool(ep: Episode, pd: Product) -> list:
    """Candidate rows of the product that have been read in full."""
    return [cr for cr in ep.candidates(pd) if cr.fully_read]


def nearest_line(lines: list[str], text: str) -> str:
    """The requirement line a free-text mention reads closest to, or the mention itself when none is close."""
    if not lines:
        return text
    closest = max(lines, key=lambda cond: overlap(cond, text))
    return closest if closest and overlap(closest, text) > 0.2 else text


def table_holds(listing: Listing, cond: dict) -> bool | None:
    """What the listing's property table says on a condition: True or False when it is clear, None otherwise."""
    texts = [str(pv) for pk, pv in listing.table.items() if near_name(str(pk).replace("_", " "), cond["prop"])]
    if not texts:
        return None
    have = all_words(" ".join(texts))
    bags = [bg for bg in map(all_words, cond["values"]) if bg]
    if any(bg <= have for bg in bags):
        return True
    return None if any(bg & have for bg in bags) else False


def cited_lines(labels: set, lines: list[str]) -> set:
    """Requirement references (R2, a paraphrase) mapped back to the requirement lines."""
    back = set()
    for label in labels:
        hit = re.fullmatch(r"\s*R?(\d+)\s*", label)
        number = int(hit.group(1)) if hit else 0
        back.add(lines[number - 1] if 0 < number <= len(lines) else nearest_line(lines, label))
    return back


class Judging:
    """Deciding whether listings meet the demands, and ranking those that do."""

    def judge_all(self) -> None:
        """Judge the read candidates of every product in one parallel round."""
        in_parallel([lambda each=each: self.judge(each, read_pool(self, each)) for each in self.products])

    def judge(self, prod: Product, listings: list) -> None:
        self.drop_label_demands(prod)
        self.align_fields(prod)
        self.trim_spoken_words(prod)
        lines = prod.demand_lines()
        if not lines:
            # only the product kind to judge: each read listing stands
            for o in listings:
                if o.fully_read and o.ident not in prod.verdicts:
                    prod.verdicts[o.ident] = {"failed": set(), "unsure": set()}
                    prod.judged_on[o.ident] = set()
            return
        wanted = set(lines)
        pending = []
        for o in listings:
            if not o.fully_read or prod.judged_on.get(o.ident, set()) >= wanted:
                continue
            # failing a requirement still in force decides it: no call needed. The lines it was judged on stay as
            # they were: if the shopper later lets the failed requirement go, the requirements it never was judged
            # on still need a call before it counts as a fit
            prior = prod.verdicts.get(o.ident) if o.ident in prod.judged_on else None
            if prior and set(prior.get("failed") or ()) & wanted:
                continue
            pending.append(o)
        if not pending or self.time_left() < END_MARGIN:
            outside_fit_codes(self, prod)
            return
        usage = self.field_usage(prod)
        batches, replies = self.ask_judges(prod, pending, lines, usage)
        for batch, reply in zip(batches, replies):
            listed = reply.get("verdicts") if isinstance(reply, dict) else None
            said = {v["id"]: v for v in (listed if isinstance(listed, list) else ())
                    if isinstance(v, dict) and v.get("id") in prod.seen}
            for o in batch:
                if o.ident in said:
                    self.settle_verdict(prod, o, said[o.ident], lines, usage)
        outside_fit_codes(self, prod)

    def ask_judges(self, prod: Product, pending: list, lines: list[str], usage: list[dict]) -> tuple[list, list]:
        """One fast-model judgement per batch of four listings, run in parallel: (batches, replies)."""
        notes = "\n\nSTORE FIELDS (from this store's records read so far):\n" + self.field_usage_text(usage) \
            if usage else ""
        numbered = "\n".join(f"R{n}: {line}" for n, line in enumerate(lines, 1))
        head = "".join(["REQUESTED PRODUCT: ", f"{prod.kind}", "\nREQUIREMENTS (judge each):\n", numbered, notes,
                        "\n\nLISTINGS:\n"])

        def card_of(o: Listing) -> dict:
            shown, links = o.view(), self.related_pages(prod, o)
            if links:
                shown["related_pages"] = links
            return shown
        batches = [pending[cut:cut + 4] for cut in range(0, len(pending), 4)]
        prompts = [head + json.dumps(list(map(card_of, batch)), ensure_ascii=False) for batch in batches]
        # independent batches run in parallel: model wait is the cost here, not steps
        return batches, in_parallel([(lambda u=u: quick_json(self, JUDGE_RULES, u, 2000)) for u in prompts])

    def settle_verdict(self, prod: Product, o: Listing, said: dict, lines: list[str], usage: list[dict]) -> None:
        """Store a judge's verdict on one listing, overruled where the store's own fields or records decide."""
        loose = names_in(said, "loose")
        verdict = dict(failed=cited_lines(names_in(said, "failed"), lines) | self.field_failures(prod, o, usage),
                       unsure=cited_lines(names_in(said, "unsure"), lines),
                       weak=cited_lines(names_in(said, "weak") | loose, lines),
                       loose=cited_lines(loose, lines))
        prod.verdicts[o.ident] = verdict
        prod.judged_on[o.ident] = set(lines)
        failed, unsure = verdict["failed"], verdict["unsure"]
        for dm in prod.firm_demands():
            line = dm.as_line()
            if not self.field_is_value(o, dm):
                continue
            # a field of the property holding only the required value meets it, whatever
            # the judge said (a value repeating the property's name is read without those words)
            if line in failed:
                failed.discard(line)
                self.trace_event("failed_overruled", id=o.ident, req=line)
            # notes handing a differently named property of this listing to another page leave this
            # property's own field standing: the doubt is cleared and the final check leaves the line out
            if line in unsure and notes_point_elsewhere(o, dm.prop):
                unsure.discard(line)
                self.proven_fits.add((prod.position, o.ident, line))
                self.trace_event("unsure_overruled", id=o.ident, req=line)
        for dm in prod.firm_demands():
            # a value met as written must have its words somewhere in the records
            line = dm.as_line()
            if line in failed or line in unsure or line in verdict["loose"]:
                continue
            if self.value_unwritten(prod, o, dm):
                unsure.add(line)
            elif span_holds_bound(o, dm):
                # a span of figures with the bound inside it does not hold the bound for certain
                unsure.add(line)
                self.trace_event("span_straddles", id=o.ident, req=line)

    def drop_label_demands(self, prod: Product) -> None:
        """Drop a request value that merely repeats the store's category label, on a property no read listing
        has: it names the category rather than a requirement."""
        if not prod.categories_ok:
            return
        read = read_of_kind(self, prod)
        if len(read) < 3:
            return
        label_words = set().union(*map(word_set, prod.categories_ok))
        kept = []
        for dm in prod.demands:
            words = word_set(dm.value)
            label_only = dm.origin == "request" and dm.op == "is" and words and words <= label_words
            if not label_only or any_key_names(read, dm.prop):
                kept.append(dm)
                continue
            self.trace_event("label_restatement", item=prod.name, req=dm.as_line())
        prod.demands = kept

    def align_fields(self, prod: Product) -> None:
        """When no read listing has a key for a required property but they hold the required value under one other
        key, the requirement moves to that key (this store files it differently). Word values only."""
        read = read_of_kind(self, prod)
        if len(read) < 3:
            return
        for dm in prod.firm_demands():
            plain_word = dm.op in ("is", "one of", "contains") and dm.value and not re.search(r"\d", dm.value)
            if dm.prop_aligned or not plain_word or any_key_names(read, dm.prop):
                continue
            wants = wanted_alternatives(dm.value)
            # per read listing, the keys whose field holds one of the wanted values
            holding = [{spaced_key(kn) for kn, val in o.table.items()
                        if any(w <= piece for piece in value_pieces([val]) for w in wants)} for o in read]
            tally = {kn: sum(1 for keys in holding if kn in keys) for kn in set().union(*holding)}
            counts = sorted(tally.values(), reverse=True)
            if not counts or (len(counts) > 1 and counts[0] == counts[1]):
                continue
            was, peak = dm.prop, max(tally, key=tally.get)
            dm.prop, dm.prop_aligned = peak, True
            self.trace_event("prop_aligned", item=prod.name, was=was, now=peak, value=dm.value)

    def shown_stems(self) -> set:
        """Stems of every word the store has shown on a seen row: title, description, notes, labels and properties."""
        stems = set()
        for each in self.products:
            for o in each.seen.values():
                parts = [o.heading, o.blurb, o.remarks, o.kind, o.maker, o.vendor]
                parts += [f"{kk} {vv}" for kk, vv in o.table.items()]
                stems |= stems_of(word_set(" ".join(str(pt) for pt in parts if pt)))
        return stems

    def trim_spoken_words(self, prod: Product) -> None:
        """A stated plain value whose leading or trailing words appear nowhere the store has shown anything (no
        title, description, note or property of any row, though a search carried those words), while the
        rest is written in the property's field on read listings: the extra words are the request's manner
        of speaking, so the value keeps only the words the store writes."""
        read = read_of_kind(self, prod)
        if len(read) < 3:
            return
        vocab = None
        for dm in prod.firm_demands():
            if dm.origin != "request" or dm.reference or dm.op not in ("is", "contains") or not dm.value:
                continue
            tokens = dm.value.split()
            if "|" in dm.value or re.search(r"\d", dm.value) or len(tokens) < 2:
                continue
            pieces = [ws for o in read for kn, val in o.table.items() if key_names_property(kn, dm.prop)
                      for ws in map(word_set, re.split(PIECE_SPLIT, str(val))) if ws]
            whole = word_set(dm.value)
            if not pieces or any(whole <= pc for pc in pieces):
                continue
            if vocab is None:
                vocab = self.shown_stems()
            hidden = [bool(ws) and not (stems_of(ws) & vocab) for ws in map(word_set, tokens)]
            shown_at = [ix for ix, gone in enumerate(hidden) if not gone]
            if not shown_at:
                continue
            lo, hi = shown_at[0], shown_at[-1] + 1
            if hi - lo == len(tokens):
                continue
            core = " ".join(tokens[lo:hi])
            core_bag = word_set(core)
            if not core_bag or not any(core_bag <= pc for pc in pieces):
                continue
            dropped = word_set(" ".join(tokens[:lo] + tokens[hi:]))
            if any(dropped <= word_set(tq) for tq in queries_sent(prod)):
                self.trace_event("value_trimmed", item=prod.name, prop=dm.prop, was=dm.value, now=core)
                dm.value = core

    @staticmethod
    def field_is_value(listing: Listing, req: Demand) -> bool:
        """A key naming the required property whose entire value is the required value (an 'is' or 'one of' word demand)."""
        if req.op not in ("is", "one of") or req.reference or not req.value:
            return False
        wants = set(field_wanted_sets(req.value, req.prop))
        own = (value_pieces([val]) for kn, val in listing.table.items() if same_name(kn, req.prop))
        return any(len(pieces) == 1 and pieces <= wants for pieces in own)

    @staticmethod
    def value_unwritten(prod: Product, listing: Listing, req: Demand) -> bool:
        """True when no record of the listing has a word of the value beyond the product's own name words."""
        if req.op not in ("is", "one of") or req.origin == "fit" or not req.value:
            return False
        # a key naming the property is for the judge to read (other spellings, ranges, ages)
        if any(key_names_property(kn, req.prop) for kn in listing.table):
            return False
        name_words = all_words(prod.name) | all_words(prod.kind)
        # short alphabetic words (articles, joiners) prove nothing; figures do
        variants = [ws for ws in ({w for w in all_words(option) - name_words if not (w.isalpha() and len(w) <= 3)}
                                  for option in req.value.split("|")) if ws]
        if not variants:
            return False
        table_text = " ".join(f"{kk} {value_v}" for kk, value_v in listing.table.items())
        texts = [listing.heading or "", listing.blurb or "", listing.remarks or "", listing.kind or "", table_text]
        stems = stems_of(all_words(" ".join(texts)))
        return not any(stems_of(ws) & stems for ws in variants)

    def field_usage(self, prod: Product) -> list[dict]:
        """How this store writes each plain required value, from the listings read so far: keys naming the property,
        how many listings carry one, the values found there, and which listings write the value."""
        read = read_of_kind(self, prod)
        if len(read) < 3:
            return []
        usage = []
        for dm in prod.firm_demands():
            if dm.op in ("is", "one of") and dm.value and not re.search(r"\d", dm.value):
                record = self.field_record(dm, read)
                if record is not None:
                    usage.append(record)
        return usage

    @staticmethod
    def field_record(dm: Demand, read: list) -> dict | None:
        """One required value's row of field_usage; None when the value has no words or no read listing a field."""
        wants = field_wanted_sets(dm.value, dm.prop)
        if not wants:
            return None
        holders = [(o, named) for o, named in ((o, [kk for kk in o.table if same_name(kk, dm.prop)]) for o in read)
                   if named]
        if not holders:
            return None
        written = [(o.ident, piece) for o, named in holders for kn in named
                   for piece in (" ".join(raw.lower().split()) for raw in re.split(PIECE_SPLIT, str(o.table[kn])))
                   if piece]
        counts = {}
        for _, piece in written:
            counts[piece] = counts.get(piece, 0) + 1
        writers = {ident for ident, piece in written if any(map(word_set(piece).issuperset, wants))}
        keys = set().union(*(named for _, named in holders))
        return dict(req=dm.as_line(), prop=dm.prop, keys=sorted(keys), carriers=len(holders), read=len(read),
                    values=counts, wants=wants, writers=writers)

    def field_failures(self, prod: Product, listing: Listing, table: list[dict] | None = None) -> set:
        """Requirements the listing fails by the store's own wording: its field lacks the value that other read listings write exactly there."""
        failed = set()
        for row in self.field_usage(prod) if table is None else table:
            own = [kk for kk in listing.table if same_name(kk, row["prop"])]
            if not own or not row["writers"] - {listing.ident}:
                continue
            pieces = value_pieces([listing.table[kk] for kk in own])
            # compare by stems so another spelling of the value is not a plain failure
            if pieces and not any(stems_of(w) <= stems_of(p) for w in row["wants"] for p in pieces):
                failed.add(row["req"])
        return failed

    def field_usage_text(self, table: list[dict]) -> str:
        lines = []
        for frow in table:
            knames = ", ".join(map(repr, frow["keys"][:3]))
            counted = sorted(frow["values"].items(), key=lambda kv: kv[1], reverse=True)
            written = " | ".join(f"{value_v} ({number})" for value_v, number in counted[:8])
            lines.append(f"- {frow['req']}: field {knames} on {frow['carriers']} of {frow['read']} read listings; values "
                         f"written: {written}; {len(frow['writers'])} listing(s) write the required value itself")
        return "\n".join(lines)

    def related_pages(self, prod: Product, listing: Listing, limit: int = 2) -> list[dict]:
        """Known listings linked to this one by a part code: named in its text by their title code, or naming its title
        code in theirs (the compatibility list may sit on the other page)."""
        def notes(o: Listing) -> str:
            return " ".join([o.blurb or "", o.remarks or ""])
        own_codes = set(MODEL_CODE.findall(listing.heading or ""))
        cited = set(MODEL_CODE.findall(notes(listing))) - own_codes
        found, taken = [], {listing.ident}
        pools = [prod.seen] + [each.seen for each in self.products if each is not prod]
        for other in (o for pool in pools for o in pool.values()):
            if len(found) >= limit:
                break
            if other.ident in taken:
                continue
            if set(MODEL_CODE.findall(other.heading or "")) & cited or own_codes & set(MODEL_CODE.findall(notes(other))):
                taken.add(other.ident)
                found.append({"id": other.ident, "title": other.heading[:120], "text": notes(other)[:400]})
        return found

    def verified(self, prod: Product, listing: Listing) -> bool:
        key = (prod.position, listing.ident, tuple(prod.demand_lines()),
               json.dumps(listing.table, sort_keys=True, default=str))
        if key not in self.verify_cache:
            self.verify_cache[key] = self.verify(prod, listing)
        return self.verify_cache[key]

    def verify_prompt(self, prod: Product, listing: Listing, lines: list[str]) -> str:
        """The final check's message: the product, the request and later shopper lines, budget, requirements, listing."""
        later = ""
        if self.store_lines:
            # lines that came later change the request: the budget and requirements below already follow them
            recent = "\n- ".join(ln[:200] for ln in self.store_lines[-6:])
            later = "LATER LINES FROM THE SHOPPER (where they differ from the request, they decide; the budget and " \
                    "requirements below already follow them):\n- " + recent + "\n"
        ceiling = self.budget if self.budget is not None else "none stated"
        listed = "\n- ".join(lines) or "(none beyond the product kind, budget and availability)"
        card = json.dumps(listing.view(3500), ensure_ascii=False)
        return (f"REQUESTED PRODUCT: {prod.name} (category: {prod.kind}; quantity {prod.quantity})\n"
                + f"REQUEST: {self.request_text[:1500]}\n" + later + f"BUDGET, SUM OF EVERY LINE: {ceiling}\n"
                + "REQUIREMENTS:\n- " + listed + "\n\nLISTING:\n" + card)

    def verify(self, prod: Product, listing: Listing, strict: bool = False) -> bool:
        # a fit another page's code list proves is settled; this check reads the listing's own records only
        lines = [ln for ln in prod.demand_lines() if (prod.position, listing.ident, ln) not in self.proven_fits]
        reply = self.models.json_reply([VERIFY_MODEL, FAST_MODEL, BACKUP_MODEL], VERIFY_RULES,
                                       self.verify_prompt(prod, listing, lines), 800)
        if not isinstance(reply, dict):
            return not strict
        fits = reply.get("fits")
        if fits is not True and strict:
            return False
        problems = [str(issue) for issue in reply.get("problems") or []]
        self.trace_event("verify", id=listing.ident, fits=fits, problems=problems[:4])
        if fits is False and problems:
            marks = prod.verdicts.setdefault(listing.ident, {"failed": set(), "unsure": set()})
            if not marks["failed"]:
                self.checker_failed.add((prod.position, listing.ident))
            marks["failed"].update(nearest_line(lines, issue) for issue in problems)
            return False
        return True

    def buyable(self, prod: Product) -> list:
        """Fits in choosing order, without listings known to be sold out or failed by a store field."""
        usage = self.field_usage(prod)
        return [fx for fx in prod.fits(cheapest=self.cheapest)
                if fx.available is not False and not self.field_failures(prod, fx, usage)]

    def rank_by_measure(self, prod: Product, cands: list) -> list:
        """Sort fits by the request's measure of best with one model call; unchanged when there is no measure."""
        if not self.measure or len(cands) < 2:
            return cands
        ids = [cand.ident for cand in cands]
        memo = self.measure_ranks.get(prod.position)
        known = memo is not None and memo[0] == ids
        if not known:
            listed = compact([{"id": c.ident, "title": c.heading[:120], "price": c.price, "properties": c.table,
                               "description": str(c.blurb or "")[:300]} for c in cands[:12]], 9000)
            reply = self.models.json_reply([PLAN_MODEL, FAST_MODEL, BACKUP_MODEL], RANK_RULES,
                                           f"MEASURE: {self.measure}\nPRODUCT: {prod.name}\nLISTINGS: " + listed,
                                           RANK_TOKENS)
            order = [str(xv) for xv in reply.get("order") or []] if isinstance(reply, dict) else []
            self.trace_event("best_rank", measure=self.measure, order=order[:12])
            memo = (ids, order)
            if order:
                self.measure_ranks = {**self.measure_ranks, prod.position: memo}
        rank = {pid: ix for ix, pid in enumerate(memo[1])}
        unranked = len(rank)
        # listings the model left out follow the ranked ones, in their given order
        return sorted(cands, key=lambda c: rank[c.ident] if c.ident in rank else unranked + ids.index(c.ident))

    def nearest_miss(self, prod: Product) -> Listing | None:
        """For a request that accepts a near miss: the read listing of the right kind with the fewest failed
        requirements (sold out, over the product's cap and store-field failures each count), then fewest
        doubts, then lowest price."""
        usage = self.field_usage(prod)
        best, best_key = None, None
        for cand in prod.seen.values():
            verdict = prod.verdicts.get(cand.ident) if cand.fully_read else None
            if verdict is None or not self.is_kind(prod, cand):
                continue
            failed = set(verdict.get("failed") or ())
            if "cart refused" in failed:
                continue
            misses = len(failed | self.field_failures(prod, cand, usage))
            flags = (cand.available is False, prod.over_unit_cap(cand), over_cap(self.budget, cand, prod.quantity))
            misses += sum(1 for flag in flags if flag)
            # a requirement never judged on this listing (time ran out) counts as a doubt, not as met
            unjudged = len(set(prod.demand_lines()) - prod.judged_on.get(cand.ident, set()))
            rank = (misses, len(verdict.get("unsure") or ()) + unjudged, sort_price(cand))
            if best_key is None or rank < best_key:
                best, best_key = cand, rank
        return best

    def condition_holds(self, cond: dict, listing: Listing) -> bool | None:
        """Is the listing's value for the condition's property among the condition's values? The property table
        decides when it is clear either way, otherwise one fast model call."""
        cache = self.condition_cache
        key = (id(cond), listing.ident, json.dumps(listing.table, sort_keys=True, default=str))
        if key not in cache:
            holds = table_holds(listing, cond)
            if holds is None:
                ask = (f"PROPERTY: {cond['prop']}\nVALUES: {as_json(cond['values'])}\n\nLISTING:\n"
                       + as_json(listing.view(2500)))
                reply = quick_json(self, CONDITION_RULES, ask, 300)
                said = reply.get("holds") if isinstance(reply, dict) else None
                holds = said if isinstance(said, bool) else None
            cache[key] = holds
            self.trace_event("condition", src=cond["src"].name, listing=listing.ident, prop=cond["prop"], holds=holds)
        return cache[key]

    def apply_conditions(self, prod: Product, chosen: dict) -> None:
        """Apply conditions whose source product has a pick to this product's demands and judge its listings again."""
        derived, touched = [], False
        for cn in self.conditions:
            branches = [(flag, sp) for flag, target, sp in cn["branches"] if target is prod]
            if not branches:
                continue
            touched = True
            pick = chosen.get(cn["src"].position)
            if pick is None:
                continue
            holds = self.condition_holds(cn, pick)
            if holds is not None:
                derived += [Demand(sp["property"], sp.get("op") or "is", sp["value"], origin="condition")
                            for flag, sp in branches if flag == holds]
        if not touched:
            return
        fresh = [dm.as_line() for dm in derived]
        if sorted(fresh) == sorted(dm.as_line() for dm in prod.demands if dm.origin == "condition"):
            return
        prod.demands = [dm for dm in prod.demands if dm.origin != "condition"] + derived
        self.trace_event("condition_applied", item=prod.name, reqs=fresh)
        self.judge(prod, read_pool(self, prod))
        if not self.buyable(prod) and room_left(self) >= 3 and time_for(self, 20):
            prod.exhausted = False
            self.hunt(prod, room_left(self), final=True)

    def fit_pairs(self) -> list[tuple]:
        """"X must fit Y" rules of the request as (accessory product, host product) pairs."""
        pairs = []
        if len(self.products) < 2:
            return pairs
        for rule in self.fit_rules:
            part_s, host_s = str(rule.get("accessory") or ""), str(rule.get("host") or "")
            if not (part_s and host_s):
                continue
            # the products whose names read closest to each side of the rule
            part_p, host_p = (max(self.products, key=lambda wp: overlap(wp.name, text)) for text in (part_s, host_s))
            if part_p is host_p or overlap(part_p.name, part_s) < 0.2 or overlap(host_p.name, host_s) < 0.2:
                continue
            if (part_p, host_p) not in pairs:
                pairs.append((part_p, host_p))
        return pairs

    def bind_to_host(self, prod: Product, host_p: Product, host: Listing) -> None:
        """Tie the accessory to the chosen host listing: a demand quoting the host's page, a lookup of the part codes
        its candidates point to, and a new judgement of the candidates."""
        quoted = " ".join(filter(None, (host.heading[:120], (host.blurb or "")[:300], (host.remarks or "")[:150])))
        prod.demands = [dm for dm in prod.demands if dm.origin != "fit"] + [
            Demand("compatibility", "is", f"works with the {host_p.name} chosen, whose page reads: {quoted}", origin="fit")]
        sought = pointed_codes(self, prod)
        series_codes = host_series(self, host_p, host)
        search_series(self, prod, sought, series_codes)
        self.trace_event("fit_bound", item=prod.name, host=host.ident, series=series_codes[:2], looked_up=sought[:3])
        # listings whose stated codes already prove the fit need no model call for it
        codes_prove_fit(self, prod, host)
        self.judge(prod, read_pool(self, prod))
        if series_codes:
            clear_series_doubt(self, prod, series_codes)
        codes_prove_fit(self, prod, host)

    @staticmethod
    def in_series(listing: Listing, series: list[str]) -> bool:
        return any(code in series for code in SERIES_PREFIX.findall(listing.heading or ""))
def read_of_kind(ep, prod):
    acc_p = []
    for one in prod.seen.values():
        if one.fully_read and one.table and ep.is_kind(prod, one):
            acc_p.append(one)
    return acc_p


def ranked_fits(ep, pd):
    """Buyable fits of a product in the request's order of best, a host chosen by the fit plan in front."""
    ranked_l = ep.rank_by_measure(pd, ep.buyable(pd))
    held = (ep.host_plan or {}).get(pd.position)
    if held is not None:
        first = [cand for cand in ranked_l if cand.ident == held.ident]
        ranked_l = first + [cand for cand in ranked_l if cand.ident != held.ident]
    return ranked_l


def title_words(heading):
    """A title as plain words, part and model codes removed."""
    return " ".join(plain_text(CODE_WORD.sub(" ", str(heading or ""))).split())


def title_twins(pd, offer):
    """Other listings of the product with the same title once part codes are removed, priced the same or higher."""
    # races 5-14, one-product tasks: where one of two title twins was turned down for the product, so was the other
    # in 59 of 73 pairs (both paid in 538 more)
    core = title_words(offer.heading)
    if len(core.split()) < 3:
        return []
    twins = []
    for other in pd.seen.values():
        if other.ident == offer.ident or title_words(other.heading) != core:
            continue
        if offer.price is not None and other.price is not None and other.price < offer.price:
            continue
        twins.append(other)
    return twins


def basket_problems(ep):
    """Check a multi-product basket before checkout: first in code (one listing per product, each of the right
    kind), then one model reading of cross-product rules and the total. Returns [(product, reason)] for
    lines to rechoose; every basket is read at most once."""
    wants = [pq for pq in ep.products if pq.choice is not None]
    if len(wants) < 2:
        return []
    bad = []
    owner = {}
    for pq in wants:
        first = owner.get(pq.choice.ident)
        if first is not None and plain_text(first.kind) != plain_text(pq.kind):
            bad.append((pq, "the same listing is the pick for another product"))
            continue
        if first is None:
            owner[pq.choice.ident] = pq
        if not ep.is_kind(pq, pq.choice):
            bad.append((pq, "not the kind of product requested"))
    if bad:
        return bad
    sig = tuple(sorted(pq.choice.ident for pq in wants))
    if sig in ep.baskets_checked or low_on_time(ep, 45) or ep.models.spent >= PILOT_SPEND:
        return []
    ep.baskets_checked.add(sig)
    total = sum(((pq.choice.price or 0) * pq.quantity for pq in wants), 0.0)
    lines = [json.dumps({"item": pq.name, "kind": pq.kind, "quantity": pq.quantity,
                         "requirements": pq.demand_lines()[:8], "listing": pq.choice.view(1400)}, ensure_ascii=False)
             for pq in wants]
    ceiling = "none stated" if ep.budget is None else str(ep.budget)
    prompt = (f"REQUEST: {ep.request_text[:1500]}\n" + f"BUDGET CAP (SUM OF LINES): {ceiling}\n"
              + f"ORDER TOTAL: {round(total, 2)}\n" + "ORDER LINES:\n" + "\n".join(lines))
    answer = ep.models.json_reply([VERIFY_MODEL, FAST_MODEL], BASKET_RULES, prompt, BASKET_TOKENS)
    if not isinstance(answer, dict) or answer.get("ok") is not False:
        return []
    by_name = dict((plain_text(pq.name), pq) for pq in wants)
    for row in answer.get("bad") or []:
        pq = by_name.get(plain_text(str(row.get("item") or ""))) if isinstance(row, dict) else None
        if pq is not None and any((pq.position, pq.choice.ident, ln) in ep.proven_fits for ln in pq.demand_lines()):
            # a fit the code proved from the store's own code lists is not overturned by a reading of the whole
            # order (Arena, race-15 world: the cart held a paid basket, the review called the proven part unfit,
            # the swap that followed ended in a decline)
            ep.trace_event("basket_row_kept", item=pq.name, why=str(row.get("why") or "")[:120])
            continue
        if pq is not None and not any(pq is b[0] for b in bad):
            bad.append((pq, "basket: " + str(row.get("why") or "breaks a rule between products")[:120]))
    return bad


def choose_confirmed(ep, pd, chosen_map):
    """Confirm up to three ranked fits of the product; with no time left, take the best one unconfirmed."""
    tries_n = 0
    for offer in ranked_fits(ep, pd):
        if tries_n >= 3 or low_on_time(ep, 0):
            break
        tries_n += 1
        if ep.verified(pd, offer):
            chosen_map[pd.position] = offer
            return
    if pd.position in chosen_map:
        return
    # three listings the checker turned down leave no pick (the close then reviews and, with nothing verified,
    # declines); only a clock too short for any check takes the best one unconfirmed
    fallback = ranked_fits(ep, pd)
    if fallback and low_on_time(ep, 0):
        chosen_map[pd.position] = fallback[0]


def try_other_host(ep, pd, host_p, chosen_map):
    """If no accessory fits the chosen host, look for another host listing one of them fits."""
    for alt_l in ep.buyable(host_p)[:3]:
        if chosen_map.get(host_p.position) is alt_l or low_on_time(ep, 0):
            continue
        ep.bind_to_host(pd, host_p, alt_l)
        fit_list = ep.buyable(pd)
        if fit_list:
            chosen_map[host_p.position] = alt_l
            chosen_map[pd.position] = fit_list[0]
            return


def flip_condition_source(ep, pd, chosen_map, cond_list, fit_pairs):
    """When nothing satisfies what a condition demands under the source's pick, try a source pick with the
    opposite outcome, unless a chosen accessory has to fit that source."""
    linked = [cc for cc in cond_list if cc["src"].position in chosen_map and any(dd2 is pd for _, dd2, _ in cc["branches"])]
    for cond_d in linked:
        src_w = cond_d["src"]
        if low_on_time(ep, 0) or any(hh2 is src_w and aa.position in chosen_map for aa, hh2 in fit_pairs):
            continue
        current = chosen_map[src_w.position]
        result = ep.condition_holds(cond_d, current)
        for alt_l in ep.buyable(src_w)[:4]:
            if alt_l is chosen_map[src_w.position] or ep.condition_holds(cond_d, alt_l) == result:
                continue
            chosen_map[src_w.position] = alt_l
            ep.apply_conditions(pd, chosen_map)
            fit_list = ep.buyable(pd)
            if fit_list and ep.verified(src_w, alt_l):
                chosen_map[pd.position] = fit_list[0]
                break
            chosen_map[src_w.position] = current
            ep.apply_conditions(pd, chosen_map)
        if pd.position in chosen_map:
            break


def fit_to_budget(ep, chosen_map):
    """Move the most expensive line to its next cheaper fit until the total fits the budget (at most ten moves); returns the total."""
    def spend():
        return sum((chosen_map[pq.position].price or 0) * pq.quantity for pq in ep.products)

    def line_cost(pq):
        return -((chosen_map[pq.position].price or 0) * pq.quantity)
    for _ in range(10):
        if spend() <= ep.budget + 1e-6:
            break
        stepped = False
        for pq in sorted(ep.products, key=line_cost):
            ladder = [ff for ff in pq.fits(cheapest=True) if ff.available is not False]
            refs = [ff.ident for ff in ladder]
            at = refs.index(chosen_map[pq.position].ident) if chosen_map[pq.position].ident in refs else -1
            if 0 <= at < len(ladder) - 1:
                chosen_map[pq.position] = ladder[at + 1]
                stepped = True
                break
        if not stepped:
            break
    return spend()


def pointed_codes(ep, prod):
    """Part codes that the product's live candidates mention in their text, that no seen title carries, and whose
    letter family appears in the product's titles."""
    pattern = re.compile(r"\b[A-Z]{1,6}-?\d{3,}[A-Z0-9]*\b")
    in_titles = set()
    for w_ in ep.products:
        for ob in w_.seen.values():
            in_titles.update(pattern.findall(ob.heading or ""))
    heads = set()
    for ob in prod.seen.values():
        heads.update(code_letters(code) for code in pattern.findall(ob.heading or ""))
    found_c = []
    for ob in ep.candidates(prod):
        if ob.fully_read and not verdict_of(prod, ob).get("failed"):
            for code in pattern.findall(" ".join([ob.blurb or "", ob.remarks or ""])):
                fresh_c = code not in in_titles and code not in found_c
                if fresh_c and code_letters(code) in heads:
                    found_c.append(code)
    return found_c


def code_letters(code):
    """A part code's leading capitals."""
    return re.match(r"[A-Z]+", code).group(0)


def search_series(ep, prod, sought, series):
    """Within the steps the ending leaves, search the host's series codes (with the product kind) and the
    pointed-to part codes, then open the series' accessories."""
    rest = []
    for code in sought:
        if code not in series:
            rest.append(code)
    for code in series[:2] + rest[:3]:
        if ep.steps_left() <= ep.closing_steps() + 2 or low_on_time(ep, 10):
            break
        ep.run_search(prod, query=f"{code} {prod.kind}" if code in series else code, pages=1, budget=ep.closing_steps())
    if not series or ep.steps_left() <= ep.closing_steps() + 2:
        return
    members = []
    for ob in ep.candidates(prod):
        if not ob.fully_read and ep.in_series(ob, series):
            members.append(ob.ident)
    if members:
        width_n = ep.compare_size if ep.batch_reads() else 3
        ep.read_listings(prod, members[:width_n], keep_steps=ep.closing_steps() + 1)


def host_series(ep, host_p, host):
    """Short codes on the host page ("AB3") that prefix part codes in the other products' titles ("AB3-1234")."""
    blob_s = " ".join([host.blurb or "", host.remarks or "", json.dumps(host.table, ensure_ascii=False)])
    heads = {hd_ for w_ in ep.products if w_ is not host_p for ob in w_.seen.values()
             for hd_ in SERIES_PREFIX.findall(ob.heading or "")}
    own_heads = SERIES_PREFIX.findall(host.heading or "")
    found_c = []
    for code in re.findall(r"\b[A-Z]{1,5}\d{1,3}\b", blob_s):
        if code in found_c or code in own_heads:
            continue
        fam_s = re.match(r"[A-Z]+", code).group(0)
        if code in heads or any(re.match(r"[A-Z]+", hd_).group(0) == fam_s for hd_ in heads):
            found_c.append(code)
    return found_c


def clear_series_doubt(ep, prod, series):
    """Read candidates whose own part code belongs to the host's series state the fit: clear that doubt."""
    fit_txt = prod.demands[-1].as_line()
    for ob in ep.candidates(prod):
        fitbook = prod.verdicts.get(ob.ident)
        if fitbook and ob.fully_read and ep.in_series(ob, series):
            for slot_k in ("unsure", "weak", "failed"):
                if slot_k in fitbook:
                    fitbook[slot_k].discard(fit_txt)


def codes_listed(rest_s):
    """The codes of a text that consists only of codes or named models ("AB1234, Corvo 512."); else None."""
    got = set()
    for row_e in re.split(r"\s*[,;/]\s*|\s+(?:and|or)\s+", rest_s.strip().rstrip(".")):
        row_e = row_e.strip()
        if not row_e:
            continue
        if MODEL_CODE.fullmatch(row_e) or MODEL_NAME.fullmatch(row_e):
            got.add(row_e)
            continue
        bits = row_e.split()
        if not all(MODEL_CODE.fullmatch(bt) for bt in bits):
            return None
        got.update(bits)
    return got or None


def codes_in(text):
    """All codes and named models in a text."""
    return set(MODEL_CODE.findall(text or "")) | set(MODEL_NAME.findall(text or ""))


def family_codes(rest_s, universe):
    """Known codes covered by a short fit statement naming code families ("every AB-12xx model, except
    AB-1220"): the seen codes each named family stands for plus plainly named codes, minus whatever follows
    an exception. None without a family or when the text is more than a few words."""
    if not CODE_FAMILY.search(rest_s):
        return None
    if len(re.findall(r"[A-Za-z]+", MODEL_CODE.sub(" ", CODE_FAMILY.sub(" ", rest_s)))) > 6:
        return None
    cut = EXCEPT_WORDS.search(rest_s)
    parts_l = [(rest_s[:cut.start()], True), (rest_s[cut.end():], False)] if cut else [(rest_s, True)]
    got, gone = set(), set()
    for text, keep in parts_l:
        pool_l = got if keep else gone
        pool_l.update(MODEL_CODE.findall(text))
        for stem_c, marks in CODE_FAMILY.findall(text):
            size = len(stem_c) + len(marks)
            pool_l.update(code for code in universe
                          if len(code) == size and code.startswith(stem_c) and code[len(stem_c):].isalnum())
    return got - gone


def fit_book(ep):
    """Fit statements found in plain code lists of the listings seen so far. In "<lead>: <codes>", a lead naming part
    codes speaks for those parts, a lead naming none for the listing that holds the text. A known-family code
    mentioned elsewhere in a listing's text (not in its own title) points to a page that may list more."""
    seen = {}
    for pq in ep.products:
        seen.update(pq.seen)
    titled, universe = set(), set()
    for ob in seen.values():
        titled.update(MODEL_CODE.findall(ob.heading or ""))
        universe.update(MODEL_CODE.findall(" ".join([ob.heading or "", ob.blurb or "", ob.remarks or ""])))
    heads = set(code_letters(code) for code in titled)
    own, about, aims = {}, {}, {}
    for ref, ob in seen.items():
        mine = set(MODEL_CODE.findall(ob.heading or ""))
        for piece in re.split(r"(?<=[.;!?])\s+", " ".join([ob.blurb or "", ob.remarks or ""])):
            lead, colon, rest = piece.partition(":")
            listed = colon and (codes_listed(rest) or family_codes(rest, universe))
            if not listed:
                for code in MODEL_CODE.findall(piece):
                    if code_letters(code) in heads and code not in mine:
                        aims.setdefault(ref, set()).add(code)
                continue
            subjects = set(MODEL_CODE.findall(lead)) - mine
            for code in subjects:
                about.setdefault(code, set()).update(listed)
            if not subjects:
                own.setdefault(ref, set()).update(listed)
    return {"seen": seen, "own": own, "about": about, "aims": aims, "titled": titled}


def stated_fits(fitbook, ref):
    """(codes the listing is said to fit, by its own lists or other pages' lists for its title codes; whether a page it points to is unseen)."""
    got = set(fitbook["own"].get(ref, ()))
    ob = fitbook["seen"].get(ref)
    if ob is not None:
        for code in MODEL_CODE.findall(ob.heading or ""):
            got |= fitbook["about"].get(code, set())
    return got, bool(fitbook["aims"].get(ref, set()) - fitbook["titled"])


def listed_fit_codes(fitbook, items):
    """All codes named by the fit lists of these products' listings, plus every code a page lists for some part."""
    out = set()
    for pq in items:
        for ref in pq.seen:
            out |= fitbook["own"].get(ref, set())
    for values_l in fitbook["about"].values():
        out |= values_l
    return out


def host_fit_codes(host, known):
    """Codes in a host listing's own records that some accessory's fit list names."""
    blob = " ".join([host.heading or "", host.blurb or "", host.remarks or "",
                     json.dumps(host.table, ensure_ascii=False, default=str)])
    return codes_in(blob) & known


def fits_but_host(ep, prod):
    """Read, buyable listings of the product that meet everything except the fit to a host, cheapest first."""
    fit_lines = {nd.as_line() for nd in prod.demands if nd.origin == "fit"}
    lines_now = set(prod.demand_lines()) - fit_lines
    out = []
    for ref, ob in prod.seen.items():
        if not ob.fully_read or ob.available is False or prod.over_unit_cap(ob) or over_cap(ep.budget, ob, prod.quantity):
            continue
        verdict_d = prod.verdicts.get(ref)
        if verdict_d is None or not prod.judged_on.get(ref, set()) >= lines_now:
            continue
        if (set(verdict_d.get("failed") or ()) | set(verdict_d.get("unsure") or ())) - fit_lines:
            continue
        out.append(ob)
    out.sort(key=sort_price)
    return out


def fit_evidence(ep, host_p, parts):
    """Code evidence for one host and its accessories: the fit book, each accessory's fitting listings with their
    codes, and the host's possible listings (read fits, or unread rows not ruled out) with their codes."""
    fitbook = fit_book(ep)
    known = listed_fit_codes(fitbook, parts)
    fit_list = {}
    for acc_p in parts:
        fit_list[acc_p.position] = [(ob, stated_fits(fitbook, ob.ident)) for ob in fits_but_host(ep, acc_p)]
    good_refs = {ob.ident for ob in ep.buyable(host_p)}
    host_rows = []
    for ob in host_p.seen.values():
        if ob.available is False or host_p.over_unit_cap(ob) or not ep.is_kind(host_p, ob):
            continue
        if (ob.fully_read and ob.ident not in good_refs) or (not ob.fully_read and ob.screen == "no"):
            continue
        cds = host_fit_codes(ob, known) if known else set()
        if cds:
            host_rows.append((ob, cds, ob.ident in good_refs))
    return fitbook, fit_list, host_rows


def covering(parts, fit_list, code):
    """Accessories having a fitting listing stated to fit this code."""
    return [acc_p for acc_p in parts if any(code in cds for _, (cds, _) in fit_list[acc_p.position])]


def chase_fit_pages(ep, parts, fit_list, fitbook, floor_k, cap_n):
    """Search the unseen pages that fitting accessory listings point to: one store call per code, no model call, each code once."""
    done_n = 0
    looked = ep.fit_codes_searched
    for acc_p in parts:
        for ob, (_, open_) in fit_list[acc_p.position]:
            if not open_:
                continue
            for code in sorted(fitbook["aims"].get(ob.ident, set()) - fitbook["titled"] - looked):
                if done_n >= cap_n or ep.steps_left() <= floor_k + 1 or low_on_time(ep, 10):
                    return done_n
                done_n += 1
                looked.add(code)
                ep.run_search(acc_p, query=code, pages=1, budget=floor_k)
    return done_n


def plan_host(ep, reads):
    """Choose the host of "X must fit Y" rules from the accessories' side. Accessories carry the other demands and so
    have few fits, while hosts have many: the host is taken with a code that every accessory has a fitting
    listing stated (in a plain code list) to fit. Pages the fits point to are looked up, and before the last
    searches (reads=True) a few unread rows that could close a gap are opened. Sets ep.host_plan
    {host slot: listing} and ep.settled_slots (slots needing no more search). Stores without code lists are
    left as they are."""
    groups = {}
    for acc, host in ep.fit_pairs():
        if host.position not in groups:
            groups[host.position] = (host, [])
        groups[host.position][1].append(acc)
    plan = dict(ep.host_plan or {})
    settled = set(ep.settled_slots or ())
    floor = ep.closing_steps() + 2
    for host, parts in groups.values():
        chosen = choose_host(ep, host, parts, plan.get(host.position), floor, reads)
        if chosen is None:
            continue
        plan[host.position] = chosen
        settled.add(host.position)
        settled.update(p.position for p in parts)
    ep.host_plan, ep.settled_slots = plan, settled


def codes_covered(parts, fits, rows):
    """Host codes for which every accessory has a fitting listing, sorted."""
    return sorted(set(code for _, codes, _ in rows for code in codes if len(covering(parts, fits, code)) == len(parts)))


def choose_host(ep, host, parts, current, floor, reads):
    """The host listing whose code every accessory is stated to fit (the current pick when it already does), or None."""
    book, fits, rows = fit_evidence(ep, host, parts)
    if not rows:
        return None
    if current is None:
        ranked = ep.rank_by_measure(host, ep.buyable(host))
        current = ranked[0] if ranked else None
    own = set()
    if current is not None:
        for ob, codes, _ in rows:
            if ob.ident == current.ident:
                own = codes
    if any(len(covering(parts, fits, code)) == len(parts) for code in own):
        return current
    if chase_fit_pages(ep, parts, fits, book, floor, 4):
        book, fits, rows = fit_evidence(ep, host, parts)
    if reads:
        book, fits, rows = fill_fit_gaps(ep, host, parts, (book, fits, rows), floor)
    codes = codes_covered(parts, fits, rows)
    if not codes:
        ep.trace_event("fit_plan_none", host=host.name, hosts=len(rows))
        return None
    codes.sort(key=lambda code: sum((min(sort_price(ob) for ob, (cds, _) in fits[acc.position] if code in cds)
                                     * acc.quantity for acc in parts), 0.0))
    if reads and ep.steps_left() > floor + 2 and time_for(ep, 10):
        # unread hosts of the codes ahead of the first code with a known good host: read in one batch and judged
        # in one parallel round (one by one, each read costs a full model round)
        batch = []
        for code in codes:
            if any(good and code in cds for _, cds, good in rows):
                break
            for ob in sorted((ob for ob, cds, good in rows if code in cds and not good), key=sort_price)[:2]:
                if ob not in batch:
                    batch.append(ob)
        if batch:
            ep.read_listings(host, [ob.ident for ob in batch[:4]], keep_steps=floor + 1)
            ep.judge(host, read_pool(ep, host))
    good_now = dict((ob.ident, ob) for ob in ep.buyable(host))
    for code in codes:
        ready = [good_now[ob.ident] for ob, cds, _ in rows if code in cds and ob.ident in good_now]
        if ready:
            best = min(ready, key=sort_price)
            ep.trace_event("fit_plan", host=host.name, code=code, pick=best.ident,
                           fits={acc.name: [ob.ident for ob, (cds, _) in fits[acc.position] if code in cds][:3] for acc in parts})
            return best
    return None


def fill_fit_gaps(ep, host, parts, evidence, floor):
    """Open a few unread rows of an accessory that could close a host code only it still lacks; returns the fit
    evidence, refreshed after each read."""
    book, fits, rows = evidence
    for acc in parts:
        if codes_covered(parts, fits, rows) or not time_for(ep, 15) or ep.steps_left() <= floor + 4:
            break
        others = [a for a in parts if a is not acc]
        # host codes every other accessory already covers: only this accessory is missing there
        goals = set(code for _, cds, _ in rows for code in cds if all(a in covering(parts, fits, code) for a in others))
        if not goals:
            continue
        groups = ep.value_groups(acc)
        unread = []
        for ob in acc.seen.values():
            if ob.fully_read or ob.available is False or ob.screen == "no" or acc.over_unit_cap(ob) \
                    or over_cap(ep.budget, ob, acc.quantity) or not ep.is_kind(acc, ob):
                continue
            codes, unlisted = stated_fits(book, ob.ident)
            if unlisted or codes & goals:
                unread.append(ob)
        if not unread:
            continue
        unread.sort(key=lambda c: (c.screen != "yes", -value_groups_hit(ep, acc, groups, c), sort_price(c)))
        ids = [ob.ident for ob in unread[:4]]
        ep.trace_event("fit_gap_read", item=acc.name, targets=sorted(goals)[:6], ids=list(ids))
        ep.read_listings(acc, ids, keep_steps=floor + 2)
        ep.judge(acc, read_pool(ep, acc))
        book, fits, rows = fit_evidence(ep, host, parts)
        if chase_fit_pages(ep, [acc], fits, book, floor, 2):
            book, fits, rows = fit_evidence(ep, host, parts)
    return book, fits, rows


def bind_accessories(ep):
    """Before the last searches, tie each accessory of a fit rule to its host's best pick and judge the fit
    (stated codes first, then one batch of model calls per accessory). An accessory with nothing
    fitting that pick then gets searched first instead of ending with an unjudged near miss at the
    last second. Groups the fit plan settled are skipped."""
    settled = set(ep.settled_slots or ())
    for acc_w, host_p in ep.fit_pairs():
        if acc_w.position in settled or not time_for(ep, 10) or room_left(ep) < 3:
            continue
        ranked_l = ranked_fits(ep, host_p)
        if not ranked_l:
            continue
        ep.bind_to_host(acc_w, host_p, ranked_l[0])
        acc_w.bound_host = ranked_l[0].ident
        acc_w.bound_host_listing = ranked_l[0]
        ep.trace_event("fit_early", item=acc_w.name, host=ranked_l[0].ident, fits=[ob.ident for ob in ep.buyable(acc_w)][:3])
        if not ep.buyable(acc_w):
            search_twins(ep, acc_w, ranked_l[0])


def near_fits(ep, prod):
    """Read, buyable listings that fail nothing but the host fit, fewest doubts first, then cheapest."""
    fit_lines = {nd.as_line() for nd in prod.demands if nd.origin == "fit"}
    out = []
    for ref, ob in prod.seen.items():
        if not ob.fully_read or ob.available is False or prod.over_unit_cap(ob) or over_cap(ep.budget, ob, prod.quantity):
            continue
        verdict_d = prod.verdicts.get(ref)
        if verdict_d is None or set(verdict_d.get("failed") or ()) - fit_lines:
            continue
        out.append((len(set(verdict_d.get("unsure") or ()) - fit_lines), sort_price(ob), ob))
    out.sort(key=lambda row: row[:2])
    return [ob for _, _, ob in out]


def twin_query(heading):
    """Title words that name the product itself: no figures, codes or bracketed parts."""
    kept = []
    for word in (w.strip(",.;:/|-+&") for w in re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", heading or "").split()):
        if len(word) > 1 and word.isalpha() and all(word.lower() != k.lower() for k in kept):
            kept.append(word)
    return " ".join(kept[:5])


def search_twins(ep, prod, host):
    """An accessory with no listing fitting its host: stores often sell one product as several listings that
    differ only in what they fit, and searching the product's own words rarely lists them all. The title
    words of listings missing only the fit are searched (one page, at most two queries), and new rows
    whose stated codes name the host's code are opened and judged."""
    if ep.steps_left() <= ep.closing_steps() + 4 or not time_for(ep, 20):
        return
    tried = set(plain_text(t) for t in queries_sent(prod))
    queries = []
    for ob in near_fits(ep, prod)[:4]:
        query = twin_query(ob.heading)
        key = plain_text(query)
        if len(query.split()) > 1 and key not in tried and all(key != plain_text(had) for had in queries):
            queries.append(query)
            if len(queries) == 2:
                break
    if not queries:
        return
    for query in queries:
        if ep.steps_left() <= ep.closing_steps() + 3 or low_on_time(ep, 15):
            break
        ep.run_search(prod, query=query, pages=1, budget=ep.closing_steps() + 2)
    codes_prove_fit(ep, prod, host)
    book = fit_book(ep)
    host_codes = host_fit_codes(host, listed_fit_codes(book, [prod]))
    fresh = sorted((ob for ob in ep.candidates(prod)
                    if not (ob.fully_read or ob.available is False or prod.over_unit_cap(ob)) and ep.is_kind(prod, ob)
                    and host_codes and stated_fits(book, ob.ident)[0] & host_codes), key=sort_price)
    picks = [ob.ident for ob in fresh[:3]]
    ep.trace_event("twin_search", item=prod.name, queries=queries, fitting=picks)
    if fresh and ep.steps_left() > ep.closing_steps() + 2:
        ep.read_listings(prod, list(picks), keep_steps=ep.closing_steps() + 1)
        codes_prove_fit(ep, prod, host)
        ep.judge(prod, read_pool(ep, prod))
        codes_prove_fit(ep, prod, host)


def settled_groups(ep):
    """Slots of fit groups that need no backup search (no cheapest rule): every accessory was tied early to the host's best pick and has a buyable fit."""
    if ep.cheapest:
        return set()
    groups = {}
    for acc_w, host_p in ep.fit_pairs():
        groups.setdefault(host_p.position, (host_p, []))[1].append(acc_w)
    done = set()
    for host_p, parts in groups.values():
        ranked_l = ranked_fits(ep, host_p)
        if not ranked_l:
            continue
        if all(acc_p.bound_host == ranked_l[0].ident and ep.buyable(acc_p) for acc_p in parts):
            done.add(host_p.position)
            done.update(acc_p.position for acc_p in parts)
    return done


def codes_prove_fit(ep, prod, host):
    """Read candidates whose plain code lists (their own, or another page's for their title code) name a code from
    the chosen host's records fit that host: the fit demand is marked met and judged, and the final
    check skips it."""
    fit_needs = [nd for nd in prod.demands if nd.origin == "fit"]
    if not fit_needs:
        return
    fit_txt = fit_needs[-1].as_line()
    fitbook = fit_book(ep)
    host_cds = host_fit_codes(host, listed_fit_codes(fitbook, [prod]))
    if not host_cds:
        return
    proven = ep.proven_fits
    # where the store never states fit for parts of a listing's code series on other pages, the listing's own code
    # list (with no pointer to a page not yet seen) is complete: one leaving out every code of the host rules it out
    spoken_for = {code_letters(code) for code in fitbook["about"]}
    misfits = set()
    # fit lines of hosts bound before: a verdict this check set on one of them no longer applies
    stale = prod.fit_lines_seen - {fit_txt}
    prod.fit_lines_seen.add(fit_txt)
    for ob in ep.candidates(prod):
        verdict_d = prod.verdicts.get(ob.ident)
        if verdict_d is not None and stale:
            for slot_k in ("unsure", "weak", "failed", "loose"):
                if slot_k in verdict_d:
                    verdict_d[slot_k] -= stale
        cds, open_ = stated_fits(fitbook, ob.ident)
        if not (cds & host_cds):
            closed = not any(code_letters(code) in spoken_for for code in MODEL_CODE.findall(ob.heading or ""))
            if closed and not open_ and fitbook["own"].get(ob.ident):
                misfits.add(ob.ident)
                if verdict_d is not None and ob.fully_read:
                    verdict_d.setdefault("failed", set()).add(fit_txt)
                    prod.judged_on.setdefault(ob.ident, set()).add(fit_txt)
            continue
        if verdict_d is None or not ob.fully_read:
            continue
        for slot_k in ("unsure", "weak", "failed", "loose"):
            if slot_k in verdict_d:
                verdict_d[slot_k].discard(fit_txt)
        prod.judged_on.setdefault(ob.ident, set()).add(fit_txt)
        proven.add((prod.position, ob.ident, fit_txt))
    prod.misfit_ids = misfits


def outside_fit_codes(ep, prod):
    """Read listings whose plain code lists (own, or another page's for their title code) name the code of the
    shopper's own device meet the owned-device demand: the judge's doubt is cleared and the final
    check skips it. Other listings keep their verdict."""
    for dm in prod.demands:
        codes = dm.owned_codes
        if dm.origin != "fit_out" or not codes:
            continue
        line_s = dm.as_line()
        fitbook = fit_book(ep)
        proven = ep.proven_fits
        for ob in prod.seen.values():
            verdict_d = prod.verdicts.get(ob.ident)
            if verdict_d is None or not ob.fully_read or not (stated_fits(fitbook, ob.ident)[0] & codes):
                continue
            for slot_k in ("unsure", "weak", "failed", "loose"):
                if slot_k in verdict_d:
                    verdict_d[slot_k].discard(line_s)
            prod.judged_on.setdefault(ob.ident, set()).add(line_s)
            proven.add((prod.position, ob.ident, line_s))


class ShopperTalk:
    """Messages to the shopper: questions, their replies, approvals and the process gate."""

    def can_message(self):
        ceiling_v = self.message_cap
        if "message" in self.roles and (ceiling_v is None or self.messages_sent < ceiling_v):
            return self.steps_left() > 0
        return False

    def reserved_messages(self):
        """Messages held back for the closing steps (a confirmation, a notice)."""
        if not self.steps_demanded or "message" not in self.roles:
            return 0
        held_back = 0
        for dem in self.steps_demanded:
            # a decline keeps nothing: the ending tool carries its own note
            if dem["kind"] != "message" or dem["moment"] not in ("before_order", "before_finish"):
                continue
            held_back += max(1, int(dem.get("messages") or 1))
        ceiling_v = self.message_cap
        if ceiling_v is None:
            return held_back
        return min(held_back, max(0, ceiling_v - self.messages_sent))

    def messages_owed(self):
        """Reserved messages still owed for the current basket (none once this basket was shown)."""
        if self.cart and self.approved_cart == self.cart_signature():
            return 0
        return self.reserved_messages()

    def can_ask(self):
        """True while a question still fits next to the reserved closing messages."""
        if self.can_message():
            ceiling_v = self.message_cap
            return ceiling_v is None or self.messages_sent < ceiling_v - self.reserved_messages()
        return False

    def cart_signature(self):
        fit_pairs = []
        for ref_d, product_count in self.cart.items():
            if product_count:
                fit_pairs.append((ref_d, int(product_count or 0)))
        fit_pairs.sort()
        return tuple(fit_pairs)

    def ask_approval(self):
        """Show the shopper the basket and ask for a go-ahead; True once the message went out."""
        if not (self.can_message() and self.steps_left() >= 2):
            return False
        spec = self.roles["message"]
        parts = []
        total = 0.0
        for ident, count in self.cart_signature():
            owner = self.line_owner.get(ident)
            offer = owner.seen.get(ident) if owner else None
            label = squeeze((offer.heading if offer else ident) or ident)[:70]
            price = offer.price if offer and offer.price is not None else None
            if price is None:
                parts.append(f"{count} x {label}")
                continue
            total += price * count
            parts.append(f"{count} x {label}" + f" ({price:g} each)")
        ending = f". Total {total:g}." if total else "."
        text = self.clip_arg("message", spec["text"], "Your final basket: " + "; ".join(parts) + ending
                             + " Shall I check out this basket? A yes from you lets me place it.")
        resp = self.call_tool(spec["name"], {spec["text"]: text})
        self.messages_sent += 1
        self.approved_cart = self.cart_signature()
        self.trace_event("confirm_asked", basket=self.approved_cart, ok=resp["ok"], obs=compact(resp["obs"], 160))
        return True

    def run_process_gate(self, moment, intent):
        """Ask the planner which demanded steps are due now and run them as checked tool calls. True if one ran,
        False if none was due, None if no answer came."""
        attempts = self.gate_rounds.get(moment, 0)
        if attempts >= 3 or self.steps_left() < 2 or low_on_time(self, 0):
            return False
        self.gate_rounds[moment] = attempts + 1
        closing = set(name for name in ((self.roles.get(role) or {}).get("name")
                                        for role in ("checkout", "decline", "cart_add", "cart_set")) if name is not None)
        offered = [{"name": name, "description": self.tool_description(name), "parameters": spec}
                   for name, spec in self.schemas.items() if name not in closing]
        basket = []
        for ident, count in self.cart_signature():
            owner = self.line_owner.get(ident)
            offer = owner.seen.get(ident) if owner else None
            basket.append({"id": ident, "quantity": count, "title": offer.heading[:120] if offer else "",
                           "price": offer.price if offer else None, "for": owner.name if owner else None})
        total = sum((row["price"] or 0) * row["quantity"] for row in basket)
        spare = "no limit" if self.message_cap is None else self.message_cap - self.messages_sent
        prompt = ("REQUEST:\n" + self.request_text + "\n\nSTEPS THE REQUEST DEMANDS:\n" + as_json(self.steps_demanded)
                  + f"\n\nMOMENT: {moment} (the assistant is about to {intent})"
                  + "\n\nBASKET NOW: " + as_json({"lines": basket, "total": total})
                  + f"\nMESSAGES LEFT: {spare}; STEPS LEFT: {self.steps_left()}"
                  + "\n\nTOOLS:\n" + as_json(offered)
                  + "\n\nDONE SO FAR (tool calls and results, oldest first):\n" + as_json(self.call_log[-30:]))
        answer = self.models.json_reply([PLAN_MODEL, FAST_MODEL, BACKUP_MODEL], PROCESS_RULES, prompt, 900)
        if not isinstance(answer, dict):
            self.trace_event("gate_failed", moment=moment)
            return None
        actions = [act for act in (answer.get("actions") or [])[:2] if isinstance(act, dict)]
        speak = (self.roles.get("message") or {}).get("name")
        acted = False
        for spot, act in enumerate(actions):
            name = act.get("tool")
            if name in closing or name not in self.schemas:
                continue
            if name != speak:
                # keep steps for messages still owed (later in this answer or reserved) and the checkout
                due = max(sum(1 for later in actions[spot + 1:] if later.get("tool") == speak), self.messages_owed())
                if self.steps_left() < due + 2:
                    self.trace_event("gate_step_skipped", moment=moment, tool=name, steps=self.steps_left(), due=due)
                    continue
            spec = self.schemas[name]
            args = self.gate_args(spec, act.get("args"))
            if any(need not in args for need in spec.get("required") or []):
                continue
            talking = name == (self.roles.get("message") or {}).get("name")
            if talking and not self.can_message():
                continue
            if self.steps_left() < 2:
                break
            resp = self.call_tool(name, args)
            if talking:
                self.messages_sent += 1
                self.approved_cart = self.cart_signature()
            self.trace_event("gate_step", moment=moment, tool=name, why=str(act.get("why") or "")[:160], ok=resp["ok"])
            acted = True
        if not acted:
            self.trace_event("gate_clear", moment=moment)
        return acted

    def gate_args(self, spec, preset):
        """The planner's arguments the tool takes, enum values checked and texts clipped to the field."""
        fields = spec.get("properties") or {}
        args = {key: val for key, val in preset.items() if key in fields} if isinstance(preset, dict) else {}
        for key in list(args):
            field = fields.get(key) or {}
            allowed = field.get("enum")
            if isinstance(allowed, list) and allowed and args[key] not in allowed:
                del args[key]
            elif isinstance(args[key], str):
                args[key] = self.clip_text(field, args[key])
        return args

    def ask_shopper(self, prod, prop, narrow=False, also=None):
        spec = self.roles["message"]
        also = [also] if isinstance(also, str) else [p_ for p_ in (also or []) if p_]
        subjects = [prop] + also
        text = self.clip_arg("message", spec["text"], question_wording(prod, prop, narrow, also))
        sent = self.call_tool(spec["name"], {spec["text"]: text})
        self.messages_sent += 1
        if not narrow:
            prod.asked.extend(plain_text(p_) for p_ in subjects)
        heard = shopper_words(sent["obs"]) if sent["ok"] else None
        if not heard:
            return None
        told = [f"{rq.prop}: {rq.op} {rq.value}" for rq in prod.demands if rq.origin == "shopper" and rq.value]
        about = ", ".join(subjects)
        reading = self.models.json_reply([FAST_MODEL, BACKUP_MODEL, PLAN_MODEL], REPLY_RULES,
                                         f"ITEM: {prod.name}\nPROPERTY ASKED ABOUT: {about}\nQUESTION: {text}\nKNOWN NEEDS: {json.dumps(told)}\nREPLY: {heard}",
                                         600)
        kind = str(reading.get("kind") or "").lower() if isinstance(reading, dict) else None
        if kind == "need":
            self.settle_need_reading(prod, subjects, about, heard, reading)
        self.trace_event("reply", item=prod.name, prop=prop, reply=heard[:200], read=compact(reading, 300))
        if kind is None:
            return None
        if kind in ("need", "repeat"):
            # remember the shopper's wording so a later question on the same subject counts as a repeat
            prod.need_replies = list(prod.need_replies) + [" " + plain_text(heard) + " "]
        return reading

    def settle_need_reading(self, prod, subjects, about, heard, reading):
        """Tidy a need reading in place: joined values, polarity, bounds inside the value, the property named."""
        whole = joined_alternatives(heard, reading.get("value"))
        if whole:
            # parts the reply joins with '+' or '&' make one combined value, not a choice between them
            self.trace_event("value_joined", item=prod.name, was=str(reading.get("value")), now=whole)
            reading["value"] = whole
            if str(reading.get("op") or "").strip().lower() == "one of":
                reading["op"] = "is"
        op = str(reading.get("op") or "").strip().lower()
        if op == "is not" and reading.get("value") and not reading.get("reference"):
            self.recheck_polarity(prod, about, heard, reading)
        elif op == "is":
            # "anything other than at least N gets a no" reads as op "is" with the bound inside the value
            split_bound(reading)
        named = self.property_in_reply(prod, heard, subjects + [str(reading.get("property") or "")],
                                       str(reading.get("value") or ""))
        if named:
            # the reply names a different listing property: the need sits on that one
            reading["property"] = named

    def recheck_polarity(self, prod, about_p, heard, read_d):
        """A reply read as excluding a plain value is read again with a narrow question: refusing everything but
        the value (a double negative) wants that very value, which the first reading often misses."""
        value_s = str(read_d.get("value"))
        message = f"PROPERTY: {read_d.get('property') or about_p}\nVALUE: {value_s}\nREPLY: {heard}"
        answer_d = self.models.json_reply([VERIFY_MODEL, FAST_MODEL], POLARITY_RULES, message, 600)
        rule = str(answer_d.get("rule") or "").strip().lower() if isinstance(answer_d, dict) else ""
        self.trace_event("polarity", item=prod.name, value=value_s[:80], rule=rule)
        if rule != "only":
            return
        read_d["op"] = "is"
        split_bound(read_d)

    @staticmethod
    def mentioned_before(prod, prop):
        """True when a multi-word property name already appeared in a reply that stated a need."""
        wbag = plain_text(prop)
        if len(wbag.split()) < 2:
            return False
        lookup_p = f" {wbag} "
        for ln_ in prod.need_replies:
            if lookup_p in ln_:
                return True
        return False

    def property_in_reply(self, prod, answer_d, asked, value):
        """If a reply names none of the asked properties, the one listing property it does name; else None."""
        padded = " " + plain_text(answer_d) + " "
        for other_name in asked:
            if other_name and plain_text(other_name) and " " + plain_text(other_name) + " " in padded:
                return None
        aliases = set()
        for offer in prod.seen.values():
            for kn in offer.table:
                nm = plain_text(split_camel(kn))
                if len(nm) < 4 or word_set(nm) & word_set(value):
                    continue
                if any(same_name(nm, other_name) for other_name in asked if other_name):
                    continue
                aliases.add(nm)
        hit_count = [nm for nm in aliases if " " + nm + " " in padded]
        kept = []
        for host_p in hit_count:
            # drop a name that is only part of a longer hit
            inner = False
            for oo in hit_count:
                if host_p != oo and " " + host_p + " " in " " + oo + " ":
                    inner = True
                    break
            if not inner:
                kept.append(host_p)
        if len(kept) == 1:
            return kept[0]
        return None

    def apply_reply(self, prod, prop, answer):
        if not answer:
            return
        for step in answer.get("steps") or []:
            if isinstance(step, dict):
                self.add_step(step)
        kind = str(answer.get("kind") or "").lower()
        cap = to_number(answer.get("budget"))
        if cap is not None and cap > 0:
            self.budget = cap
            self.trace_event("budget", cap=cap)
        named = str(answer.get("property") or prop)
        if kind == "free":
            prod.no_view.append(plain_text(named))
            return
        if kind != "need" or not (answer.get("value") or answer.get("reference")):
            return
        need = Demand(named, answer.get("op") or "is", answer.get("value") or "", answer.get("reference"), origin="shopper")
        if need.reference:
            # the page gets read for the need's own property
            need.reference["property"] = need.prop
            sig = reference_identity(need.reference)
            if sig and any(sig in (old.traced_key, old.first_traced_key)
                           or (old.reference and reference_identity(old.reference) == sig) for old in prod.demands):
                # the same page reference again: the value already taken from it stands
                self.trace_event("reference_repeat", item=prod.name, prop=need.prop)
                return
        if need.value and any(old.origin == "shopper" and plain_text(old.value) == plain_text(need.value)
                              and old.op == need.op for old in prod.demands):
            return
        # only a shopper-given requirement on the same property gets replaced
        prod.demands = [old for old in prod.demands if old.origin != "shopper" or old.key_text() != need.key_text()] + [need]
        if need.op in ("is", "one of") and need.value and not need.reference and len(need.value) <= 40:
            prod.plan_query(f"{need.value.split('|')[0].strip()} {prod.noun}")
        if need.reference and need.op in REFERENCE_OPS:
            self.trace_reference(prod, need)

    def apply_reply_guarded(self, prod, prop, answer):
        """apply_reply while questions are pending: a reference chain must not take their steps."""
        if self.message_cap is None:
            steps_now = 2
        else:
            steps_now = self.message_cap - self.messages_sent
        self.trace_floor = max(0, steps_now) + self.closing_steps()
        try:
            self.apply_reply(prod, prop, answer)
        finally:
            self.trace_floor = 0

    def read_stated_values(self):
        """A request sentence "the <property> is <value>" on a property the read listings carry and no requirement
        covers yet gets a short check; a confirmed demand joins the product's requirements."""
        whole = padded_flat(self.request_text)
        chunks = [pc.strip() for pc in re.split(r"(?<=[.!?;])\s+", self.request_text) if pc.strip()]
        for pd in self.products:
            names = set().union(*spelled_tables([o for o in pd.seen.values() if o.fully_read and self.is_kind(pd, o)]))
            tried = set()
            for nm in sorted(names, key=lambda n: (-len(n), n)):
                key = re.sub(r"\s+", " ", plain_text(nm)).strip()
                probe = f" the {key} is "
                if not key or key in tried or probe not in whole \
                        or any(plain_text(rq.prop) == key or overlap(rq.prop, key) >= 0.67 for rq in pd.demands):
                    continue
                line = next((pc for pc in chunks if probe in padded_flat(pc)), "")
                if line:
                    tried.add(key)
                    self.check_stated(pd, key, line)

    def check_stated(self, pd, key, line):
        """One short model check of a request sentence stating the property; a confirmed demand is added."""
        others = [wp.name for wp in self.products if wp is not pd]
        got = self.models.json_reply([FAST_MODEL, BACKUP_MODEL], STATED_RULES,
                                     f"ITEM: {pd.name}\nOTHER ITEMS: {json.dumps(others)}\nPROPERTY: {key}\nSENTENCE: {line}", 300)
        self.trace_event("stated_check", item=pd.name, prop=key, read=compact(got, 200))
        if not isinstance(got, dict) or got.get("demand") is not True:
            return
        val = str(got.get("value") or "").strip()
        rel = str(got.get("op") or "is").strip().lower()
        if val and rel in ("is", "is not", "at least", "at most", "one of"):
            pd.demands.append(Demand(key, rel, val))
            if rel in ("is", "one of") and len(val) <= 40:
                pd.plan_query(f"{val.split('|')[0].strip()} {pd.noun}")

    def property_menu(self, prod, carried=False):
        """Properties of the product's opened listings, ranked by how much their values vary (with carried, by how
        many listings have them); keys holding equal values are merged."""
        pool = [c for c in prod.seen.values() if c.fully_read and self.is_kind(prod, c)]
        live = [c for c in pool if not verdict_of(prod, c).get("failed")]
        if len(live) >= 4:
            pool = live
        tables = spelled_tables(pool)
        counts = {}
        for table in tables:
            for name in table:
                counts[name] = counts.get(name, 0) + 1
        # on a tie the plainer (shorter) name speaks for its aliases
        names = sorted(counts, key=lambda nm: (-counts[nm], len(nm), nm))
        folded = fold_spellings(names, tables, alias_keys(names, tables))
        decided = {rq.key_text() for rq in prod.demands} | set(prod.asked)
        rows = []
        for name in names:
            if name in folded:
                continue
            group = [name] + [alias for alias, lead in folded.items() if lead == name]
            if any(plain_text(nm) in decided or any(overlap(nm, fx) >= 0.67 for fx in decided) for nm in group) \
                    or word_set(name) & SELECTOR_WORDS:
                continue            # settled, or a variant picker (those confuse the shopper)
            values = [vv for vv in group_values(tables, group) if vv is not None]
            rows.append(menu_line(name, group, values, len(pool), carried))
        # equal weights go to the key more listings carry: on real races 11-13 a widely carried key held the hidden
        # need more often than a rarely carried one in two races of three, too mixed for a weight of its own
        rows.sort(key=lambda row: (-row["_w"], -row["listings"], row["property"]))
        for row in rows:
            del row["_w"]
        return rows[:40]

    def fixed_properties(self, prod):
        """Properties the request already settles (condition sources and targets, fit pairings); never asked."""
        any_hit = []
        for link in self.conditions:
            if link["src"] is prod:
                any_hit.append(link["prop"])
            for _, target_p, branch in link["branches"]:
                if target_p is prod:
                    any_hit.append(str(branch["property"]))
        for host_s, _ in self.fit_pairs():
            if host_s is prod:
                any_hit.append("compatibility")
                break
        if "compatibility" not in any_hit and self.bound_to_owned(prod):
            any_hit.append("compatibility")
        if self.measure:
            # the measure the request ranks by: the shopper wants the most (or least) of it, so a question on it draws
            # no need (real races 12-14, every agent: 0 of 49 such questions revealed one on that property)
            any_hit.append(str(self.measure))
        return list(dict.fromkeys(any_hit))

    def plan_questions(self, count, model=PLAN_MODEL):
        """Question order: the planner proposes it once; code removes repeats and settled properties and fills gaps
        from the property menus."""
        if count <= 0:
            return []
        if not self.question_order and not self.plan_requested:
            self.question_order = self.first_question_plan(count, model)
        queue = []
        if len(self.products) > 1:
            return self.round_robin_plan(count)
        for pd, prop in self.question_order:
            if len(queue) >= count:
                break
            if not self.prop_covered(pd, prop, queue):
                queue.append((pd, prop))
        # one item from here on: its two most carried properties are likely hidden needs, put them early
        if count >= 3:
            for pd in self.products:
                tops = [row["property"] for row in self.property_menu(pd, carried=True) if row["listings"] >= 3]
                for spot, prop in enumerate(tops[:2]):
                    if not self.prop_covered(pd, prop, queue):
                        queue.insert(min(len(queue), 2 * spot + 1), (pd, prop))
            del queue[count:]
        if len(queue) < count:
            self.fill_from_menus(queue, count)
        return queue[:count]

    def prop_covered(self, pd, prop, queue):
        """True when the property is asked, reworded, decided, demanded, mentioned or already queued."""
        if any(same_name(prop, name) for name in pd.asked + pd.reworded):
            return True
        if any(near_name(prop, name) for name in self.fixed_properties(pd)):
            return True
        if any(same_name(prop, rq.prop) for rq in pd.demands) or self.mentioned_before(pd, prop):
            return True
        return any(owner is pd and same_name(prop, name) for owner, name in queue)

    def fill_from_menus(self, queue, count):
        """Fill the queue from the property menus, the least asked product first, one property per product a turn."""
        menus = dict((wp.position, [row for row in self.property_menu(wp) if row["listings"] >= 2 or row["of"] <= 2])
                     for wp in self.products)
        turns = sorted(self.products, key=lambda wp: (sum(1 for owner, _ in queue if owner is wp) + len(wp.asked),
                                                      wp.position))
        while len(queue) < count:
            before = len(queue)
            for pd in turns:
                menu = menus[pd.position]
                while menu:
                    prop = menu.pop(0)["property"]
                    if not self.prop_covered(pd, prop, queue):
                        queue.append((pd, prop))
                        break
                if len(queue) >= count:
                    break
            if len(queue) == before:
                break

    def first_question_plan(self, count, model):
        """The one planner round of the task, planned again from a thinner menu when a pick is rarely carried."""
        # one planner round per task: when no model gives a plan, a second round later costs the same time again
        self.plan_requested = True
        size = max(count if self.message_cap is None else self.message_cap, count)
        picks = self.ask_plan(size, model, False)
        source = "model"
        if not picks:
            picks, source = self.plan_from_hints(), "hints"
        # (K4 reordering by coverage was measured harmful against b08: off)
        self.trace_event("plan_questions", questions=[(wp.name, prop) for wp, prop in picks], source=source)
        # one product: a planned question on a property that few of its read listings carry is planned again
        # from a menu without such properties (see common_entries)
        if source == "model" and len(self.products) == 1 and self.time_left() > END_MARGIN + 60:
            menu = self.property_menu(self.products[0])
            rare = [prop for _, prop in picks[:size] if planned_is_rare(menu, prop)]
            if rare:
                again = self.ask_plan(size, model, True)
                self.trace_event("plan_questions_again", thin=rare, questions=[(wp.name, prop) for wp, prop in again])
                picks = again or picks
        return picks

    def ask_plan(self, size, model, thin):
        """One planner reading: [(product, property)] in the order to ask, repeats dropped."""
        blocks = []
        for pd in self.products:
            menu = self.property_menu(pd)
            # the fit clause reads to the planner as a remark on "how it fits" and drew questions on the very
            # property the pages settle (race 13: the hidden need sat elsewhere on that product, or on another)
            blocks.append({"item": pd.name, "category": pd.kind, "known_requirements": pd.demand_lines(),
                           "decided_by_the_request": self.fixed_properties(pd),
                           "hints": [cl for cl in pd.hints if not self.is_fit_clause(cl)], "already_asked": pd.asked,
                           "property_menu": common_entries(menu) if thin else menu})
        message = ("REQUEST:\n" + self.request_text + "\n\nSUBJECTS THE REQUEST TELLS THE ASSISTANT TO RAISE: "
                   + json.dumps(self.topics) + "\n\nITEMS:\n" + as_json(blocks) + f"\n\nQUESTION_COUNT = {size}")
        backup = BACKUP_MODEL if model == FAST_MODEL else FAST_MODEL
        self.models.unparsed = []
        chain = [model, backup, BACKUP_MODEL if backup != BACKUP_MODEL else FAST_MODEL]
        answer = self.models.json_reply(chain, QUESTION_RULES, message + JSON_CLOSER, PLAN_TOKENS)
        listed = answer.get("questions") if isinstance(answer, dict) else None
        picks = []
        for q in listed if isinstance(listed, list) else []:
            if not isinstance(q, dict) or not q.get("property"):
                continue
            target = max(self.products, key=lambda wp: overlap(wp.name, q.get("item") or wp.name))
            prop = squeeze(str(q["property"]).replace("_", " ")).lower()
            if prop and not any(wp is target and same_name(prop, name) for wp, name in picks):
                picks.append((target, prop))
        if picks:
            return picks
        # no JSON from any model: a reply naming its picks in sentences still says which properties it chose
        for said in self.models.unparsed or []:
            picks = self.plan_from_prose(said)
            if picks:
                break
        return picks

    def plan_from_prose(self, said_t):
        """Questions recovered from a planner reply written in sentences: menu properties in the order named, each
        given to the product whose menu has it (if several, the product named last before it, else the
        first). A product's own name words never count as a property; one-letter spelling slips match."""
        words = plain_text(said_t).split()

        def spans(name):
            seq = plain_text(name).split()
            width = len(seq)
            return [(i, i + width) for i in range(len(words) - width + 1)
                    if all(near_spelling(words[i + j], part) for j, part in enumerate(seq))] if seq else []
        named, masked = {}, set()
        for wp in self.products:
            found = spans(wp.name) + spans(wp.kind)
            named[wp.position] = [start for start, _ in found]
            masked.update(k for start, end in found for k in range(start, end))
        name_words = [w for wp in self.products for w in plain_text(wp.name + " " + str(wp.kind)).split()]
        hits = []
        for wp in self.products:
            for row in self.property_menu(wp):
                prop = row["property"]
                for start, end in spans(prop):
                    # a lone word shared with an item's name most likely names that item, not the property
                    lone = end - start == 1 and any(near_spelling(words[start], w) for w in name_words)
                    if not lone and not masked.intersection(range(start, end)):
                        hits.append((start, end, wp, prop))
        # a shorter name inside a longer one ("type" in "plug type") is part of the longer name
        hits = [h for h in hits if not any(o[0] <= h[0] and h[1] <= o[1] and (o[1] - o[0]) > (h[1] - h[0]) for o in hits)]
        picks = []
        for at in sorted(set(h[0] for h in hits)):
            owners = [h for h in hits if h[0] == at]
            if len(owners) > 1:
                owners.sort(key=lambda h: max([p for p in named[h[2].position] if p < at] or [-1]), reverse=True)
            wp = owners[0][2]
            prop = squeeze(str(owners[0][3]).replace("_", " ")).lower()
            if prop and not any(pw is wp and same_name(prop, pp) for pw, pp in picks):
                picks.append((wp, prop))
        return picks

    def plan_from_hints(self):
        """No model produced a plan: per product, the (at most two) menu properties its story hints point at most
        closely (similar name, else the share of the property's words found in a hint), products with such
        a property first; turn order and menus fill the rest."""
        chosen_map = []
        for wp in self.products:
            scored = []
            for pos, row_d in enumerate(self.property_menu(wp)):
                prop_n = squeeze(str(row_d["property"]).replace("_", " ")).lower()
                wset = stem_set(prop_n)
                best = 0.0
                for clue in wp.hints:
                    if same_name(prop_n, clue) or overlap(prop_n, clue) >= 0.6:
                        best = max(best, 2.0)
                    elif wset:
                        best = max(best, len(wset & stem_set(clue)) / len(wset))
                if best > 0:
                    scored.append((-best, pos, prop_n))
            chosen_map.extend((wp, prop_n) for _, _, prop_n in sorted(scored)[:2])
        return chosen_map

    def reconsider_next(self, pd, queue_q):
        """After a "no preference" reply the pre-made plan may next ask a mere rewording of the declined property
        on the same product (no view on one rarely means a view on the other); that message then goes to a
        property recording a separate fact, one most listings carry."""
        if not queue_q or queue_q[0][0] is not pd or not pd.no_view or self.time_left() <= END_MARGIN + 60:
            return queue_q
        planned = queue_q[0][1]
        menu = self.property_menu(pd)
        if not menu:
            return queue_q
        no_view = [{"property": prop, "values": shown_values(pd, prop)} for prop in dict.fromkeys(pd.no_view)]
        message = ("PRODUCT: " + pd.name + " (" + str(pd.kind) + ")" + "\nKNOWN REQUIREMENTS: " + as_json(pd.demand_lines())
                   + "\nALREADY ASKED: " + as_json(pd.asked) + "\nNO_VIEW_ON: " + as_json(no_view)
                   + "\nPLANNED: " + as_json({"property": planned, "values": shown_values(pd, planned)})
                   + "\nPROPERTY MENU: " + as_json(menu))
        answer = self.models.json_reply([FAST_MODEL, BACKUP_MODEL], REWORD_RULES, message, 600)
        if not isinstance(answer, dict):
            return queue_q
        choice = squeeze(str(answer.get("property") or "").replace("_", " ")).lower()
        restates = answer.get("restates") is True
        self.trace_event("recheck_question", item=pd.name, no_view=list(pd.no_view), planned=planned, restates=restates,
                         chosen=choice, why=str(answer.get("why") or "")[:200])
        if not restates or not choice or same_name(choice, planned):
            return queue_q
        # the new property must be one the listings carry and still open (not asked, fixed or settled)
        name = next((row["property"] for row in menu if same_name(choice, row["property"])), None)
        if not name or any(same_name(name, asked) for asked in pd.asked) \
                or any(same_name(name, rq.prop) for rq in pd.demands) or self.mentioned_before(pd, name) \
                or any(near_name(name, fixed) for fixed in self.fixed_properties(pd)):
            return queue_q
        pd.reworded = list(pd.reworded) + [plain_text(planned)]
        return [(pd, name)] + [pair for pair in queue_q[1:] if not (pair[0] is pd and same_name(pair[1], name))]

    def round_robin_plan(self, count):
        """Several products: a queue per product (planner picks, its most carried property second, then its menu)
        served in turns, so every product is asked before any gets a third question."""
        queues = dict((wp.position, []) for wp in self.products)
        first_seen = []
        for wp, name in self.question_order:
            line = queues.get(wp.position)
            if line is None or self.prop_covered(wp, name, ()) or any(same_name(name, had) for had in line):
                continue
            line.append(name)
            if wp.position not in first_seen:
                first_seen.append(wp.position)
        for wp in self.products:
            options = [row for row in self.property_menu(wp) if row["listings"] >= 2 or row["of"] <= 2]
            line = queues[wp.position]
            lead = next((row["property"] for row in self.property_menu(wp, carried=True) if row["listings"] >= 3), None)
            if lead and not self.prop_covered(wp, lead, ()) and not any(same_name(lead, had) for had in line[:2]):
                line[:] = [had for had in line if not same_name(had, lead)]
                line.insert(min(1, len(line)), lead)
            for row in options:
                name = row["property"]
                if not self.prop_covered(wp, name, ()) and not any(same_name(name, had) for had in line):
                    line.append(name)
            if wp.position not in first_seen:
                first_seen.append(wp.position)
        heard = set(wp.position for wp in self.products if any(rq.origin == "shopper" for rq in wp.demands))
        by_slot = dict((wp.position, wp) for wp in self.products)
        # least-asked items first; the plan is rebuilt after every told need
        turns = sorted(first_seen, key=lambda ix: (ix in heard, len(by_slot[ix].asked), first_seen.index(ix)))
        picked = []
        # a product the request binds the other products to (they have to fit it) is the main purchase, and the parts
        # are mostly settled by that fit: in race 13 the hidden need sat on the host in 22 of 25 such tasks, and after
        # a missed first key a second key on the host revealed it .25 of the time, one on a part .00 (182 tries).
        # So until a need is heard on it, the host takes the first two questions; a heard need rebuilds this plan
        # with the host last.
        hosts = list(dict.fromkeys(host_p.position for _, host_p in self.fit_pairs()))
        if len(hosts) == 1 and hosts[0] in queues and hosts[0] not in heard and not by_slot[hosts[0]].unclear_replies:
            host, line = by_slot[hosts[0]], queues[hosts[0]]
            for _ in range(max(0, 2 - len(host.asked))):
                if len(picked) >= count or not line:
                    break
                picked.append((host, line.pop(0)))
        # a product whose hidden need is out waits until the others' queues run dry: in race 12 no multi-product task
        # that some agent passed held a second hidden need on a product that had told one
        # a product whose question drew a "say it again" reply comes last of all (.03 needs on later questions there)
        lost = set(ix for ix in turns if by_slot[ix].unclear_replies)
        rings = ([ix for ix in turns if ix not in heard and ix not in lost],
                 [ix for ix in turns if ix in heard and ix not in lost], [ix for ix in turns if ix in lost])
        for ring in rings:
            while len(picked) < count and any(queues[ix] for ix in ring):
                for ix in ring:
                    if len(picked) < count and queues[ix]:
                        picked.append((by_slot[ix], queues[ix].pop(0)))
        return picked


def question_wording(prod, prop, narrow, also):
    """Question text for one property or a list; narrow asks for that one thing, with up to two sample values."""
    noun_s = prod.name
    if not narrow:
        if also:
            named = [prop] + list(also)
            return f"Thinking of the {noun_s}: which {', '.join(named[:-1])} or {named[-1]} would you like?"
        return f"Which {prop} would you like the {noun_s} to have?"
    samples = []
    for co in prod.seen.values():
        for pkey, pval in co.table.items():
            if same_name(split_camel(pkey).replace("_", " "), prop) and str(pval) not in samples:
                samples.append(str(pval)[:30])
    examples_s = " or ".join(samples[:2])
    return f"Just this one thing: for the {noun_s}, what {prop} do you need" + (f", for example {examples_s}?" if examples_s else "?")


def split_bound(read_d):
    """A wanted value that starts with a bound ("at least 320 ml") becomes that op plus the figure, so it is
    compared as a bound rather than matched as text (op names are this agent's own)."""
    value_s = str(read_d.get("value") or "")
    bare = plain_text(value_s)
    for bound in ("at least", "at most"):
        if bare.startswith(bound + " ") and len(bare) > len(bound) + 1:
            read_d["op"] = bound
            read_d["value"] = value_s.strip()[len(bound):].strip()
            return


def joined_alternatives(answer_d, value):
    """The parsed alternatives as the reply writes them joined by '+' or '&' (wanted together), or None."""
    alts = [part.strip() for part in str(value or "").split("|") if part.strip()]
    if len(alts) < 2 or not answer_d:
        return None
    pattern = r"\s*[+&]\s*".join(re.escape(part) for part in alts)
    found = re.search(r"(?<!\w)" + pattern + r"(?!\w)", str(answer_d), re.I)
    return " ".join(found.group(0).split()) if found else None


def spelled_tables(offers):
    """Every listing's property table with plainly spelled keys (camelCase split, separators as spaces, lower case)."""
    table_list = []
    for cr in offers:
        table_d = {}
        for key_x, val_x in cr.table.items():
            nm = squeeze(split_camel(key_x).replace("_", " ").replace(".", " ")).lower()
            if nm and nm not in table_d:
                table_d[nm] = str(val_x).strip().lower()[:60]
        table_list.append(table_d)
    return table_list


def alias_keys(key_names, table_list):
    """Keys equal in at least 80% of the (two or more) listings carrying both: later key -> the key that stands for it."""
    folded = {}
    for pos_i, first_w in enumerate(key_names):
        if first_w in folded:
            continue
        for other_k in key_names[pos_i + 1:]:
            if other_k in folded:
                continue
            both_rows = [table_d for table_d in table_list if first_w in table_d and other_k in table_d]
            if len(both_rows) >= 2 and sum(table_d[first_w] == table_d[other_k] for table_d in both_rows) >= 0.8 * len(both_rows):
                folded[other_k] = first_w
    return folded


def names_related(name_a, name_b):
    """Two property names that may be one property: one's words contain the other's, or both end in the same word."""
    seq_a = [singular(w) for w in plain_text(name_a).split()]
    seq_b = [singular(w) for w in plain_text(name_b).split()]
    if not seq_a or not seq_b:
        return False
    return set(seq_a) <= set(seq_b) or set(seq_b) <= set(seq_a) or seq_a[-1] == seq_b[-1]


def fold_spellings(key_names, table_list, folded):
    """Keys never carried together, with related names (see names_related) and values sharing a word (or both
    wordless, or both mostly starting with a number), are one property each seller spells differently.
    They form one group led by the fullest name; folded keys and their aliases point at the lead. On the
    Arena copies of races 8-13 a hidden requirement split like that looked like several rare properties."""
    open_k = [name for name in key_names if name not in folded]
    carriers, vwords, figured = {}, {}, {}
    for name in open_k:
        spellings = [name] + [alias for alias, lead in folded.items() if lead == name]
        found = [(ix, table[s]) for ix, table in enumerate(table_list) for s in spellings if s in table]
        carriers[name] = set(ix for ix, _ in found)
        vwords[name] = set(w for _, raw in found for w in re.findall(r"[a-z]{2,}", raw))
        texts = [str(raw) for _, raw in found]
        # values that open on a figure ("20 g", "25g per serving") are one kind of value whatever unit word follows
        figured[name] = bool(texts) and 2 * sum(1 for t in texts if re.match(r"\s*\d", t)) > len(texts)
    owner = dict((name, name) for name in open_k)
    members = dict((name, [name]) for name in open_k)
    for at, a in enumerate(open_k):
        for b in open_k[at + 1:]:
            ga, gb = owner[a], owner[b]
            if ga == gb or not names_related(a, b):
                continue
            alike = vwords[a] & vwords[b] or not (vwords[a] or vwords[b]) or (figured[a] and figured[b])
            # the two groups must not share a listing
            if not alike or any(carriers[x] & carriers[y] for x in members[ga] for y in members[gb]):
                continue
            for m in members[gb]:
                owner[m] = ga
            members[ga].extend(members.pop(gb))
    for grp in list(members.values()):
        if len(grp) > 1:
            # the fullest name leads (it holds the other spellings' words and says most about the property), then the
            # name linked to most of the group, then the most carried
            lead = max(grp, key=lambda m: (len(set(plain_text(m).split())), sum(1 for o in grp if o != m and names_related(m, o)),
                                           len(carriers[m]), -grp.index(m)))
            folded.update((m, lead) for m in grp if m != lead)
    for name in list(folded):
        lead = folded[name]
        if lead in folded:
            folded[name] = folded[lead]
    return folded


def group_values(table_list, group):
    """Per listing, its value under the first group name it carries, None when it carries none."""
    return [next((table_d[nm_x] for nm_x in group if nm_x in table_d), None) for table_d in table_list]


def spread(seen_values):
    """Natural-log entropy of the values seen."""
    spread_v = 0.0
    for val_x in set(seen_values):
        frac_v = seen_values.count(val_x) / len(seen_values)
        spread_v -= frac_v * math.log(frac_v)
    return spread_v


def menu_line(kn, group, seen_values, n_total, carried):
    """One menu entry with its sort weight under "_w": value spread times the share of listings carrying it (a
    need separates listings; one value everywhere tells nothing), or the carrier count when carried."""
    carriers_n = len(seen_values)
    shown_vals = list(dict.fromkeys(val_x for val_x in seen_values if val_x))
    w_score = spread(seen_values) * carriers_n / max(n_total, 1)
    n_distinct = len(set(seen_values))
    w_count = carriers_n * (0.5 if n_distinct == 1 and carriers_n >= 3 else 1.0)
    if carriers_n >= 4 and n_distinct >= carriers_n:
        # a distinct value on every listing reads like an id (a year, a code)
        w_score *= 0.5
        w_count *= 0.5
    row_e = {"property": kn, "listings": carriers_n, "of": n_total, "values": shown_vals[:3]}
    if len(group) > 1:
        row_e["same_as"] = group[1:4]
    row_e["_w"] = w_count if carried else w_score
    return row_e


def rarely_carried(row_d):
    """A menu property carried by fewer than a third of three or more read listings."""
    return row_d["of"] >= 3 and row_d["listings"] * 3 < row_d["of"]


def planned_is_rare(menu, prop):
    """Whether the planned property (or one of its aliases) is a rarely carried menu entry."""
    for row_d in menu:
        if same_name(prop, row_d["property"]) or any(same_name(prop, other_name) for other_name in row_d.get("same_as", [])):
            return rarely_carried(row_d)
    return False


def common_entries(menu, floor_k=3):
    """The menu minus rarely carried properties, keeping a few when fewer than floor_k would remain. Across 429
    single-product planner menus on the Arena copies of races 8-13, the hidden requirement never sat on a
    property carried by under a third of the read listings, and none of 96 planned questions on such a
    property found it."""
    kept = [row_d for row_d in menu if not rarely_carried(row_d)]
    if len(kept) >= floor_k:
        return kept
    room = floor_k - len(kept)
    back = [row_d for row_d in menu if rarely_carried(row_d)][:room]
    return [row_d for row_d in menu if any(row_d is item_x for item_x in kept + back)]


class ReferenceTracing:
    """Values a demand quotes from another listing's page."""

    def trace_reference(self, prod, req, hop=0):
        if req not in prod.demands:
            # the shopper let the requirement go: its chain is not traced any further
            return
        ref_d = req.reference or {}
        if not req.first_traced_key:
            # remember the first page named: a later repeat points at this one
            req.first_traced_key = reference_identity(ref_d)
        floor = max(4, self.trace_floor)
        if hop > 7 or self.steps_left() < floor or low_on_time(self, 20):
            return
        seller, kind, attrs, wanted = reference_parts(ref_d, req.prop)
        page, tried = self.locate_reference(seller, kind, attrs, ref_d, floor)
        if page is None:
            self.trace_event("reference_unmatched" if tried else "reference_unfound", seller=seller, category=kind)
            return
        pairing = req.op == "counterpart of reference"
        relation = str(ref_d.get("relation") or "").strip()
        head = f"PROPERTY WANTED: {wanted}\n" + (f"RELATION THE SHOPPER ASKED FOR: {relation}\n" if relation else "")
        # stores whose rows lack a property table: read the notes first, so a page that only points on costs no table
        thin = self.compare_thin and not page.table
        if thin and not page.remarks and self.steps_left() >= 3:
            self.load_table(page)
            thin = False
        read = self.page_value(page, head)
        # a pairing depends on the page's own value, which only its table states
        if thin and self.steps_left() >= 3 and isinstance(read, dict) and not read.get("redirect") \
                and (pairing or not read.get("value")) and self.load_table(page):
            read = self.page_value(page, head)
        self.trace_event("reference", hop=hop, page=page.ident, read=compact(read, 200))
        link = read.get("redirect") if isinstance(read, dict) else None
        if not isinstance(link, dict) or not link:
            take_page_value(self, prod, req, ref_d, read, pairing, relation, page)
            return
        about = str(link.get("property") or wanted)
        if not redirect_targets(about, wanted):
            # the notes redirect some other property: the wanted one is read off this page
            if thin and not page.table and not read.get("value") and self.steps_left() >= 3 and self.load_table(page):
                again = self.page_value(page, head)
                if isinstance(again, dict):
                    read.update({fld: again.get(fld) or read.get(fld) for fld in ("value", "paired")})
            if not read.get("value"):
                for key, val in page.table.items():
                    if near_name(str(key).replace("_", " "), wanted):
                        if val not in (None, ""):
                            read["value"] = str(val)
                        break
            self.trace_event("reference_other_redirect", about=about, wanted=wanted, value=read.get("value"))
            take_page_value(self, prod, req, ref_d, read, pairing, relation, page)
            return
        link["property"] = wanted
        if relation:
            link["relation"] = relation
        if ref_d.get("tolerance"):
            link["tolerance"] = ref_d["tolerance"]
        req.reference = link
        self.trace_reference(prod, req, hop + 1)

    def page_value(self, page, head):
        return quick_json(self, PAGE_RULES, head + "\nPAGE:\n" + as_json(page.view(3000)), POINTER_TOKENS)

    def load_table(self, page):
        page.table = self.fetch_table(page.ident) or {}
        return page.table

    def locate_reference(self, seller, kind, attrs, ref_d, floor):
        """Search for the page a reference describes; returns that page (or None) and the ids already tried."""
        # one search state per described page, so a retry pages further and skips listings already tried
        scratch, tried = reference_search(self, seller, kind, attrs)
        words = reference_words(attrs, seller)
        described = " ".join(words[:3])
        # the store ranks a query's rows by the described words found anywhere in a listing, the property table
        # included, while a result row shows only its title and a cut description: rows returned for a query
        # naming the described values keep the store's order
        order = scratch.store_order
        figures = word_set(described) - word_set(seller)

        def shortlist():
            pool = [c for c in scratch.seen.values() if c.ident not in tried and seller_matches(seller, c)]
            kept = pool if not kind else [c for c in pool if self.is_kind(scratch, c)]
            if not kept and seller and len(pool) <= 2 * self.compare_size:
                kept = pool          # loosely described category: trust the seller's own listings
            place = dict((pid, spot) for spot, pid in enumerate(scratch.seen))
            # equal scores keep the store's order (best keyword matches first)
            return sorted(kept, key=lambda c: (order.get(c.ident) is None, order.get(c.ident) or 0,
                                               -described_overlap(described, words, c),
                                               overlap(c.vendor, seller) < 0.5, place.get(c.ident, 0)))
        for query in reference_queries(seller, kind, described):
            if self.steps_left() < floor:
                break
            keyed = bool(figures & word_set(query))
            # the store shows few rows per page: up to three pages before moving on
            for _ in range(3):
                if self.steps_left() < floor:
                    break
                before = set(scratch.seen)
                if self.run_search(scratch, query=query, pages=1, budget=3) == 0:
                    break
                for pid in (scratch.seen if keyed else ()):
                    if pid not in before:
                        order.setdefault(pid, len(order))
                if shortlist():
                    break
            hits = shortlist()
            if not hits:
                continue
            page = self.match_page(scratch, hits, ref_d, seller, kind, attrs, floor)
            # only the pages actually opened are spent: a candidate left unread (the reading stops after a few
            # pages) stays in line for the next query or a later retry
            tried.update(c.ident for c in hits if getattr(scratch.seen.get(c.ident), "fully_read", False))
            if page is not None:
                return page, tried
        return None, tried

    def match_page(self, scratch, found, ref, seller, category, attrs, keep_steps=0):
        """Find the listing a reference describes among the candidates, opening them in batches and never going below
        the caller's step floor (the ending keeps its steps)."""
        keep = max(3, keep_steps)
        pick = str(ref.get("pick") or "").strip()
        count = ref.get("count")
        count = count if isinstance(count, int) else None
        intro = "DESCRIPTION: category " + category + "; seller " + seller + "; attributes " + as_json(attrs)
        seen = scratch.seen
        if pick or (count or 0) > 1:
            page = self.pick_described(scratch, found, intro, pick, count, keep)
            if page is not None:
                return page
            # pages already read and rejected are not opened again
            found = [c for c in found if not getattr(seen.get(c.ident), "fully_read", False)]
        alone = not self.roles.get("compare") or self.compare_off
        size, rounds = (1, 5) if alone else (self.compare_size, 3)
        pending = []
        for _ in range(rounds):
            if not found or self.steps_left() < keep:
                break
            chunk, found = found[:size], found[size:]
            for got in self.read_listings(None, [c.ident for c in chunk], keep_steps=keep):
                seen[got.ident] = got
            batch = [seen[c.ident] for c in chunk if c.ident in seen and seen[c.ident].fully_read]
            exact = next((c for c in batch if table_states_all(attrs, c)), None)
            if exact is not None:
                return exact
            pending += batch
            if alone and found and len(pending) < 3:
                continue          # gather a few single pages before one model comparison
            shown, pending = (pending if alone else batch), []
            cards = [c.view(1600) for c in shown]
            if not cards:
                continue
            page, settled = self.judge_pages(cards, shown, intro, attrs, seen)
            if page is not None:
                return page
            if not settled and len(cards) == 1 and not found:
                return seen[cards[0]["id"]]
        if not pending:
            return None
        # pages opened one at a time but never compared before the rounds or steps ran out: judge them now (no
        # store call), so a page already paid for is not dropped as unmatched
        return self.judge_pages([c.view(1600) for c in pending], pending, intro, attrs, seen)[0]

    def judge_pages(self, cards, shown, intro, attrs, seen):
        """One matcher reading of opened pages: (the page it names, True) when it answers with an id, else (the single
        page whose own text carries every described value, or None; False)."""
        verdict = quick_json(self, MATCH_RULES, intro + "\n\nCANDIDATES:\n" + as_json(cards), MATCH_TOKENS)
        if isinstance(verdict, dict) and "id" in verdict:
            # the matcher saw them all and none fits: keep looking
            return seen.get(verdict["id"]), True
        hits = [c for c in shown if text_states_all(attrs, c)]
        return (hits[0] if len(hits) == 1 else None), False

    def pick_described(self, scratch, found, intro, pick, count, keep):
        """Choosing among several seller listings: read them all first (a few more than the stated count), then one
        matcher reading names the page."""
        size = self.compare_size
        upto = min(len(found), 3 * size, max(4, 2 * (count or 2)))
        for start in range(0, upto, size):
            if self.steps_left() < keep:
                break
            ids = [c.ident for c in found[start:min(start + size, upto)]
                   if not getattr(scratch.seen.get(c.ident), "fully_read", False)]
            for got in self.read_listings(None, ids, keep_steps=keep):
                scratch.seen[got.ident] = got
        cards = [scratch.seen[c.ident].view(1200) for c in found
                 if c.ident in scratch.seen and scratch.seen[c.ident].fully_read]
        if not cards:
            return None
        text = intro + (f"\nPICK: {pick}" if pick else "") + (f"\nLISTINGS THAT FIT: {count}" if count else "")
        verdict = quick_json(self, MATCH_RULES, text + "\n\nCANDIDATES:\n" + as_json(cards), MATCH_TOKENS)
        if isinstance(verdict, dict) and verdict.get("id") in scratch.seen:
            return scratch.seen[verdict["id"]]
        return None


def reference_parts(ref_d, prop):
    """A reference's seller, category, described attributes and wanted property."""
    attrs = ref_d.get("attributes")
    return (str(ref_d.get("seller") or "").strip(), str(ref_d.get("category") or "").strip(),
            attrs if isinstance(attrs, dict) else {}, str(ref_d.get("property") or prop).strip())


def reference_search(ep, vendor, category_s, attr_d):
    """The search state kept per described page: a scratch product and the ids already tried."""
    key = json.dumps([plain_text(vendor), plain_text(category_s),
                      sorted((plain_text(k), plain_text(v)) for k, v in attr_d.items())])
    state = ep.reference_searches.get(key)
    if state is None:
        state = ep.reference_searches[key] = (
            Product({"name": category_s or "listing", "category": category_s, "search_terms": []}, 99), set())
    return state


def reference_words(attr_d, vendor):
    """Search words for a described page: short values whole, a few words of long ones, figures last (units may differ)."""
    texts = [str(v) for v in attr_d.values()]
    words = [t.strip() for t in texts if len(t.split()) <= 3 and t.strip()]
    for t in texts:
        if len(t.split()) > 3:
            words += [w for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", t)
                      if w not in words and plain_text(w) != plain_text(vendor)][:4]
    return sorted(words, key=lambda w: bool(re.search(r"\d", w)))


def reference_queries(vendor, category_s, described_text):
    """Queries for a described page, most specific first ("kettles and tea makers" names its kind twice)."""
    last = category_s.split()[-1] if category_s else ""
    head = re.split(r"\s+and\s+", category_s)[0].split() if category_s else []
    nouns = list(dict.fromkeys(n for n in (last, head[-1] if head else "") if n))
    out = []
    if vendor:
        out.append(f"{vendor} {described_text}".strip())
        # the described figures without their units: a store whose titles or fields write "170ml" or "170 ML" still
        # lists the page for the seller's name and the bare number (race 9: the paid runs looked the page up so)
        figures = re.findall(r"\d+(?:[.,]\d+)?", described_text)
        if figures:
            out.append(" ".join([vendor] + figures[:2]))
        out.extend(f"{vendor} {n}" for n in nouns)
        out.append(vendor)
    if category_s and described_text:
        out.append(f"{described_text} {last}".strip())
    return list(dict.fromkeys(filter(None, out)))


def seller_matches(vendor, offer):
    """True when no seller is named or the listing is that seller's."""
    if not vendor:
        return True
    if any(overlap(name, vendor) >= 0.5 for name in (offer.vendor, offer.maker)):
        return True
    return plain_text(vendor) in plain_text(" ".join([offer.heading, offer.blurb, offer.remarks]))


def described_overlap(described_text, wlist, offer):
    """Count of described words in the listing's text; a figure counts with or without the space before its unit."""
    sought = word_set(described_text)
    if not sought:
        return 0
    blob = " ".join([offer.heading, offer.blurb, offer.remarks, json.dumps(offer.table)])
    have = word_set(blob)
    tight = plain_text(blob).replace(" ", "")
    extra = 0
    for piece in wlist:
        own = word_set(piece)
        if re.search(r"\d", piece) and not (own and own <= have):
            packed = plain_text(piece).replace(" ", "")
            if len(packed) >= 3 and packed in tight:
                extra += 1
    return len(sought & have) + extra


def text_states_all(attr_d, offer):
    """True when the words of every described value appear in the listing's own text (title, notes, description, table)."""
    if not attr_d:
        return False
    rows = " ".join(f"{k} {v}" for k, v in offer.table.items())
    have = all_words(" ".join([offer.heading or "", offer.remarks or "", offer.blurb or "", rows]))
    return all(not str(v).strip() or (all_words(v) and all_words(v) <= have) for v in attr_d.values())


def table_states_all(attr_d, offer):
    """True when the listing's property table states every described value."""
    if not offer.fully_read or not attr_d:
        return False
    facts = [(plain_text(split_camel(k)), plain_text(v)) for k, v in offer.table.items()]
    return all(plain_text(v) and any(have == plain_text(v) and near_name(name, k) for name, have in facts)
               for k, v in attr_d.items())


def redirect_targets(about_p, wanted_p):
    """True when a redirect concerns the wanted property (same name, or one name contained in the other)."""
    if near_name(about_p, wanted_p):
        return True
    about, wanted = plain_text(about_p), plain_text(wanted_p)
    return wanted in about or about in wanted


def settle_reference(nd_q, ref_d, want_s, rel):
    """Fix a reference demand to the value its page gave."""
    nd_q.traced_key = reference_identity(ref_d)
    nd_q.value = want_s
    nd_q.op = rel
    nd_q.reference = None


def range_demands(ep, pd, nd_q, ref_d, val):
    """A figure plus or minus the reference's tolerance, in the page's unit, becomes a pair of bounds; False when a
    figure is unreadable."""
    centre, give = to_number(val), to_number(str(ref_d.get("tolerance") or "").strip())
    if centre is None or give is None:
        return False
    unit = re.sub(r"^[\s\d.,+-]+", "", str(val)).strip()
    high = f"{centre + give:g} {unit}".strip()
    settle_reference(nd_q, ref_d, f"{centre - give:g} {unit}".strip(), "at least")
    pd.demands.append(Demand(nd_q.prop, "at most", high, origin=nd_q.origin))
    ep.trace_event("reference_range", prop=nd_q.prop, low=nd_q.value, high=high)
    return True


def take_page_value(ep, pd, nd_q, ref_d, page_read, pairing, relation, page_o):
    """Turn the page reading into the demand's value, or into a pair of bounds around it."""
    got = page_read if isinstance(page_read, dict) else {}
    val = got.get("paired" if pairing else "value")
    twin = got.get("paired")
    if nd_q.op == "same as reference" and twin and got.get("wanted") \
            and plain_text(got["wanted"]) == plain_text(twin) != plain_text(got.get("value")):
        # the shopper wants what goes with the page's value, and the page names that
        val = twin
        ep.trace_event("relation_paired", page=page_o.ident, relation=relation[:80], value=val)
    if val and nd_q.op == "within reference":
        if range_demands(ep, pd, nd_q, ref_d, val):
            return
        nd_q.op = "same as reference"
    if not val:
        if pairing and got.get("value"):
            # no pairing on the page: its own value is the nearest reading of what was pointed at
            ep.trace_event("pair_unstated", page=page_o.ident, value=got.get("value"))
            settle_reference(nd_q, ref_d, str(got["value"]), "is")
        return
    settle_reference(nd_q, ref_d, str(val), REFERENCE_OPS.get(nd_q.op, "is"))
    # the page's value makes the item's sharpest query
    if nd_q.op == "is" and len(nd_q.value) <= 40 and pd.planned_queries \
            and not any(plain_text(nd_q.value) in plain_text(q) for q in pd.planned_queries):
        pd.plan_query(f"{nd_q.value} {pd.noun}")


class StoreLines:
    """Shopper lines that arrive inside tool results."""

    def read_store_lines(self):
        """Read queued shopper lines once; True when they changed the task."""
        lines_new = [ln for ln in self.line_queue if ln not in self.lines_seen]
        self.line_queue = []
        if not lines_new:
            return False
        self.lines_seen.update(lines_new)
        self.store_lines.extend(lines_new)
        prompt_txt = "LINES:\n- " + "\n- ".join(lines_new)
        prompt_txt += "\n\nREQUESTED ITEMS: " + json.dumps([pq.name for pq in self.products])
        # while a requirement tied to another page is still untraced, the requirements in force go along: without them
        # the reader took "ease off on <property>" for small talk when the property sat in a reference. Other lines
        # read alike either way (recorded lines, both inputs), so only then
        held = {pq.name: [rq.as_line() for rq in pq.demands] for pq in self.products}
        sent_held = any(rq.reference and rq.op in REFERENCE_OPS for pq in self.products for rq in pq.demands)
        if sent_held:
            prompt_txt += "\n\nREQUIREMENTS NOW: " + json.dumps(held)
        touched = self.apply_line_reading(lines_new, quick_json(self, LINE_RULES, prompt_txt, 700))
        if touched is None and not sent_held and any(held.values()):
            # the reading gave nothing to act on, or a drop naming no requirement held: read once more with the
            # requirements in force (real race lines dropping a property in slang: 4 of 8 reads named it alone, 8 of 8
            # with the requirements)
            touched = self.apply_line_reading(lines_new, quick_json(self, LINE_RULES, prompt_txt + "\n\nREQUIREMENTS NOW: " + json.dumps(held), 700))
        if touched is None:
            # nothing to act on here: the pilot sees these lines on its next look
            self.unplaced_lines.extend(lines_new)
        return bool(touched)

    def apply_line_reading(self, lines_new, answer_d):
        """Act on one reading of shopper lines. True if the task changed, False if acted on without change, None
        if there was nothing to act on (no answer, only remarks, or a drop of no held requirement)."""
        self.trace_event("event", lines=[ln[:160] for ln in lines_new], read=compact(answer_d, 300))
        if not isinstance(answer_d, dict):
            return None
        entries = [change for change in (answer_d.get("changes") or []) if isinstance(change, dict)]
        kinds = [str(change.get("kind") or "").lower() for change in entries]
        if set(kinds) <= {"other", "", "drop"} and not any(
                kind == "drop" and change.get("property") and self.has_demand_on(str(change["property"]))
                for kind, change in zip(kinds, entries)):
            return None
        touched = False
        for kind, change in zip(kinds, entries):
            aim = self.line_target(change)
            if kind == "budget" and to_number(change.get("budget")):
                before = self.budget
                self.budget = checked_budget(to_number(change.get("budget")), lines_new, self.budget)
                touched = True
                if self.budget != before:
                    self.widened_step = self.steps     # what fits, and so the cheapest fit, may differ now
            elif kind == "drop" and change.get("property"):
                if self.drop_demand([aim] if aim else self.products, str(change["property"])):
                    touched = True
                    self.widened_step = self.steps
            elif kind == "add" and change.get("property") and change.get("value"):
                targets = [aim] if aim else self.products[:1]
                for pq in targets:
                    pq.demands.append(Demand(change["property"], change.get("op") or "is", change["value"], None, origin="shopper"))
                touched = touched or bool(targets)
            elif kind in ("price", "stock"):
                touched = True
                self.refresh_cart()
            elif kind == "step" and isinstance(change.get("step"), dict):
                self.add_step(change["step"])
        if touched:
            self.changed_at = self.steps       # last step at which the request changed
        return touched

    def line_target(self, change):
        """The product a change line names, or None (no item, or a weak match among several products)."""
        item = change.get("item")
        if not item:
            return None
        closest = max(self.products, key=lambda pq: overlap(pq.name, item))
        return None if len(self.products) > 1 and overlap(closest.name, item) < 0.2 else closest

    def has_demand_on(self, prop):
        """True when a requested product has a requirement on this property."""
        return any(overlap(rq.prop, prop) >= 0.5 for pq in self.products for rq in pq.demands)

    def drop_demand(self, goals_t, prop):
        """Remove a need the shopper gave up, from the products and from every stored verdict; True if one went."""
        dropped = False
        for pd in goals_t:
            gone = [rq for rq in pd.demands if overlap(rq.prop, prop) >= 0.5]
            pd.demands = [rq for rq in pd.demands if overlap(rq.prop, prop) < 0.5]
            if not gone:
                continue
            dropped = True
            # the item's searches named the dropped value, so every listing without it went unlisted: queries holding
            # its words leave the plan, the bare product query goes first, and the value rows start over (real race
            # 15: after "take <property> off ... and search without it", 85 of 92 runs searched nothing more and kept
            # a dearer pick; no run passed)
            words = set().union(*(all_words(str(rq.value or "").replace("|", " ")) for rq in gone))
            # the bare product query: the product noun without the dropped value's words, else the category
            bare = " ".join(w for w in str(pd.noun).split() if not all_words(w) & words).strip()
            if not all_words(bare):
                bare = str(pd.kind or pd.noun)
            pd.planned_queries = [bare] + [q for q in pd.planned_queries
                                           if not all_words(q) & words and plain_text(q) != plain_text(bare)]
            pd.value_ids, pd.values_complete, pd.value_queries_sent, pd.swept_below = set(), False, [], None
            self.trace_event("drop_requery", item=pd.name, prop=prop, queries=pd.planned_queries[:3])
            for verdict in pd.verdicts.values():
                for kn in ("failed", "unsure", "weak", "loose"):
                    if verdict.get(kn):
                        verdict[kn] = {fx for fx in verdict[kn] if overlap(str(fx).split(":")[0], prop) < 0.5}
        return dropped

    def add_step(self, step):
        """Add a process step the shopper asked for mid-task to the steps the gate checks."""
        if not step.get("goal") or len(self.steps_demanded) >= 8:
            return
        row_e = dict(step)
        row_e["moment"] = str(row_e.get("moment") or "before_finish").lower()
        is_msg = str(row_e.get("kind") or "").lower() == "message"
        row_e["kind"] = "message" if is_msg else "tool"
        for step_e in self.steps_demanded:
            if overlap(row_e["goal"], step_e.get("goal")) >= 0.6:
                return
        self.steps_demanded.append(row_e)
        candidates = row_e["kind"] == "message" and row_e["moment"] in ("before_order", "before_finish")
        self.needs_approval = self.needs_approval or candidates
        self.trace_event("step_added", step=row_e)

    def refresh_cart(self):
        for pid, pd in list(self.line_owner.items()):
            if self.steps_left() <= 2:
                return
            self.read_listings(pd, [pid], page=True)
            read_back = pd.seen[pid] if pid in pd.seen else None
            self.trace_event("refreshed", id=pid, price=None if read_back is None else read_back.price,
                             stock=None if read_back is None else read_back.available)


def checked_budget(ceiling_v, line_list, old_cap):
    """Check a budget read from shopper lines against the figures in those lines: the reader may be off by a factor
    of 100; failing that, take the first figure near the old budget."""
    nums = [float(xx.replace(",", "")) for xx in re.findall(r"\d[\d,]*(?:\.\d+)?", " ".join(line_list))]
    if ceiling_v in nums:
        return ceiling_v
    if ceiling_v * 100 in nums:
        return ceiling_v * 100
    if ceiling_v / 100 in nums:
        return ceiling_v / 100
    if nums and old_cap is not None:
        near_l = [xx for xx in nums if old_cap / 20 <= xx <= old_cap * 20]
        if near_l:
            return near_l[0]
    return ceiling_v


def sweep_after_news(ep, since):
    """Cheapest rule, after a refused close whose lines dropped a need or moved the budget (at or after step
    `since`): rejected listings may fit now and unread rows may beat every known fit. Per product, a field
    search capped below its cheapest fit (with no need left to judge, the cap moves below the cheapest
    row listed while more rows hide), then the cheapest new rows are read, keeping steps for a cart swap
    and the order. Store calls only, so it may run past the closing margin; True if a listing was read."""
    if ep.finished or not ep.cheapest or ep.widened_step < since or ep.sweeps_done >= 2 \
            or not (ep.roles.get("search_fields") or {}).get("price_max"):
        return False
    ep.sweeps_done += 1
    # a swap is a removal and an add per line, then the order
    keep = 2 * len(ep.products) + 1
    found = False
    ep.sweep_floor = SWEEP_FLOOR
    try:
        for pd in ep.products:
            bare = not pd.firm_demands()
            if ep.steps_left() < keep + 2 or ep.time_left() <= SWEEP_FLOOR:
                break
            if not bare and ep.time_left() <= END_MARGIN + 10:
                continue    # a read row could not be judged in time
            top = sweep_ceiling(ep, pd)
            if top is not None:
                found = sweep_product(ep, pd, top, bare, keep) or found
    finally:
        ep.sweep_floor = None
    return found


def sweep_ceiling(ep, pd):
    """The price a swept row has to beat: the cheapest fit's, else just above the field search's price cap."""
    fits = ep.buyable(pd)
    top = sort_price(fits[0]) if fits else None
    if top is not None and top < 1e17:
        return top
    cap = field_filters(ep, pd).get("price_max")
    return None if cap is None else cap + 0.01


def cheaper_rows(ep, pd, below):
    """Unread buyable rows of the kind priced under `below`, not refused by the cart, cheapest first."""
    rows = []
    for row in pd.seen.values():
        if row.fully_read or row.available is False or row.price is None or row.price >= below \
                or pd.over_unit_cap(row) or over_cap(ep.budget, row, pd.quantity):
            continue
        if ep.is_kind(pd, row) and "cart refused" not in verdict_of(pd, row).get("failed", ()):
            rows.append(row)
    return sorted(rows, key=sort_price)


def sweep_product(ep, pd, top, bare, keep):
    """Field searches under the price to beat, then the cheapest new rows read; True if a listing was read."""
    cap = top
    for _ in range(3):
        if ep.steps_left() < keep + 2:
            break
        filters = {"category": pd.kind, "price_max": max(0.0, cap - 0.01)}
        state = "search_fields:" + json.dumps(filters, sort_keys=True)
        if (pd.paging.get(state) or {}).get("done"):
            break
        ep.run_search(pd, fields=filters, pages=1, budget=keep + 1)
        hidden = not (pd.paging.get(state) or {}).get("done")
        under = cheaper_rows(ep, pd, cap)
        ep.trace_event("late_sweep", item=pd.name, cap=filters["price_max"], cheaper=len(under), more=hidden)
        if not under or not hidden or not bare:
            break
        # every listed row of the kind fits now: only rows under the cheapest one listed can beat it
        cap = sort_price(under[0])
    read = False
    for row in cheaper_rows(ep, pd, top):
        if ep.steps_left() < keep + 1 or ep.time_left() <= SWEEP_FLOOR or (not bare and ep.time_left() <= END_MARGIN):
            break
        got = ep.read_listings(pd, [row.ident], keep_steps=keep)
        if not got:
            break
        read = True
        ep.judge(pd, got)
        if any(fit.ident == row.ident for fit in ep.buyable(pd)):
            break
    return read


class Cart:
    """The cart, the order and the ways of ending without one."""

    def closing_steps(self):
        """Steps kept for the close: cart adds, a look at carted listings, a swap, the checkout and a retry."""
        decided = self.fixed_reserve
        if decided is not None:
            return decided
        look_steps = min(len(self.products), 3) if self.tables_suffice() else 2
        total_v = len(self.products) + 6
        if self.roles.get("facts") and not self.roles.get("compare"):
            total_v += look_steps
        later = 0
        for step_e in self.steps_demanded:
            if step_e["moment"] != "start":
                later += 1
        if self.needs_approval:
            later += 1
        total_v += min(4, later)
        if self.cheapest and not self.backups_read:
            total_v += 1
        return total_v

    def set_cart(self, lines):
        """Replace the whole cart with exactly these lines (id -> quantity) through the replacing tool."""
        tool_d = self.roles.get("cart_set") or {}
        line_arg = tool_d.get("lines") or ""
        qty_arg = tool_d.get("qty")
        line_schema = (self.role_schema("cart_set").get("properties") or {}).get(line_arg) or {}
        cap_n = (((line_schema.get("items") or {}).get("properties") or {}).get(qty_arg or "") or {}).get("maximum")
        payload_rows = []
        for line_ref, qty in lines.items():
            row_e = {tool_d["id"]: line_ref}
            if qty_arg:
                row_e[qty_arg] = qty if not isinstance(cap_n, int) else min(qty, cap_n)
            payload_rows.append(row_e)
        reply_d = self.call_tool(tool_d["name"], {tool_d["lines"]: payload_rows})
        if reply_d["ok"]:
            return reply_d["ok"]
        self.trace_event("cart_set_failed", error=reply_d["error"], obs=compact(reply_d["obs"], 200))
        return reply_d["ok"]

    def add_line(self, prod, listing):
        return change_cart(self, listing.ident, prod)

    def remove_line(self, listing_id):
        return change_cart(self, listing_id, None)

    def search_pointed_parts(self):
        """Before ordering, search once for part codes a cart line's text points to that no seen title carries."""
        spec = self.roles.get("search_text")
        if not spec or not spec.get("query") or not self.cart:
            return
        done = self.parts_searched
        titles = " ".join(c.heading or "" for wp in self.products for c in wp.seen.values())
        codes = []
        for pid in self.cart:
            own = next((wp.seen[pid] for wp in self.products if pid in wp.seen), None)
            if own is None:
                continue
            for mt in POINTS_ELSEWHERE.finditer(" ".join([own.blurb or "", own.remarks or ""])):
                code = mt.group(1).strip("-")
                if len(code) < 4 or not re.search(r"\d", code) or code.lower() in done or code in codes:
                    continue
                pat = r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![A-Za-z0-9])"
                if re.search(pat, own.heading or "", re.I):
                    continue          # the listing's own code
                if re.search(pat, titles, re.I):
                    done.add(code.lower())
                else:
                    codes.append(code)
        if not codes or self.steps_left() < self.messages_owed() + 2:
            return
        mode = spec.get("mode")
        args = {mode["arg"]: mode.get("value")} if isinstance(mode, dict) and mode.get("arg") else {}
        batch = codes[:5]
        args[spec["query"]] = self.clip_arg("search_text", spec["query"], " ".join(batch))
        resp = self.call_tool(spec["name"], args)
        done.update(c.lower() for c in batch)
        self.trace_event("pointed_parts", codes=batch, rows=len(find_rows(resp["obs"]) or []))

    def place_order(self):
        tool_d = self.roles.get("checkout")
        if not tool_d:
            return False
        if self.cart and self.steps_demanded:
            intent = "check out the basket below as the order"
            step_ran = self.run_process_gate("before_order", intent)
            if step_ran and self.messages_owed() and self.steps_left() >= 2:
                # steps ran yet a demanded message for this basket is still owed
                step_ran = self.run_process_gate("before_order", intent) or step_ran
            if step_ran is not False and self.needs_approval and self.approved_cart != self.cart_signature():
                step_ran = self.ask_approval() or step_ran
            if step_ran and (self.read_store_lines() or self.picks_invalid()):
                # the reply carried a change: let the caller choose again
                self.trace_event("procedure_changed")
                return False
        self.search_pointed_parts()
        reply_d = self.call_tool(tool_d["name"], self.closing_args("checkout"))
        if self.finished:
            self.trace_event("ordered", cart=self.cart)
            return True
        self.trace_event("order_refused", obs=compact(reply_d["obs"], 200), lines=reply_d["lines"][:3])
        self.last_refusal = refusal_record(tool_d["name"], reply_d)
        return False

    def cart_early(self):
        """Cart the best known pick before the final search: stores announce changes (sold out, new price, dropped
        need, new budget) once the cart holds something, so they arrive while steps remain. Checkout makes the
        cart match the final picks, at no cost when the pick holds."""
        if self.early_carted or self.cart:
            return
        if not (self.roles.get("cart_add") or self.roles.get("cart_set")):
            return
        if room_left(self) < 3 or low_on_time(self, 30):
            self.early_carted = True
            return
        # no fit yet: leave the cart empty; a later fit gets carted when the hunt calls again
        self.judge_all()
        for pd in self.products:
            order_c = ranked_fits(self, pd)
            if not order_c:
                continue
            choice = order_c[0]
            if over_cap(self.budget, choice, pd.quantity):
                continue
            self.early_carted = True
            if self.add_line(pd, choice):
                self.trace_event("provisional_cart", item=pd.name, id=choice.ident)
            return

    def picks_invalid(self):
        spend = None
        for pd in self.products:
            choice = pd.choice
            if choice is None:
                continue
            if choice.available is False:
                return True
            if self.budget is None or choice.price is None:
                continue
            if spend is None:
                spend = 0
                for wp in self.products:
                    if wp.choice:
                        spend += (wp.choice.price or 0) * wp.quantity
            if spend > self.budget + 1e-6:
                return True
        return False

    def repick(self):
        """After a change, search again where needed and choose each product's pick from its fits again (cheapest rule kept)."""
        for pd in self.products:
            self.judge(pd, read_pool(self, pd))
            if not self.cheapest and any(fx.available is not False for fx in pd.fits()):
                continue
            pd.heard_search_done, pd.swept_below = False, None
            room = max(2, room_left(self))
            if room >= 3 and time_for(self, 20):
                self.hunt(pd, room, final=True)
        for pd in self.products:
            fits = [fx for fx in pd.fits(cheapest=self.cheapest) if fx.available is not False]
            if not fits:
                pd.choice = None
            elif self.cheapest or pd.choice is None or pd.choice.ident not in [fx.ident for fx in fits]:
                pd.choice = fits[0]
        choices = {wp.position: wp.choice for wp in self.products if wp.choice is not None}
        targets = {}
        for rule in self.conditions:
            targets.update((dd.position, dd) for _, dd, _ in rule["branches"])
        for target in targets.values():
            # a new pick of a condition source may change what the condition demands
            self.apply_conditions(target, choices)
            fit_ids = [fx.ident for fx in self.buyable(target)]
            if target.choice is not None and target.choice.ident not in fit_ids:
                target.choice = target.seen[fit_ids[0]] if fit_ids else None
        if any(wp.choice is None for wp in self.products):
            self.run(Event(PICKS_DUE))
        elif self.budget is not None and sum((wp.choice.price or 0) * wp.quantity for wp in self.products) > self.budget + 1e-6:
            self.run(Event(PICKS_DUE))

    def order_nearest(self):
        """The request allows the nearest listing: cart each product's pick (chosen, best fit or nearest miss) and order."""
        if self.nearest_tried or not self.products or not self.roles.get("checkout"):
            return False
        self.nearest_tried = True
        preset = self.nearest_picks or {}
        lineup = {}
        for pq in self.products:
            pick = preset.get(pq.position)
            if pick is None:
                ranking = ranked_fits(self, pq)
                pick = ranking[0] if ranking else self.nearest_miss(pq)
            if pick is not None:
                lineup[pq.position] = pick
        if not lineup:
            return False
        covered = [pq for pq in self.products if pq.position in lineup]
        wanted = set(cr.ident for cr in lineup.values())
        stale = [pid for pid in self.cart if pid not in wanted]
        missing = [pq for pq in covered if lineup[pq.position].ident not in self.cart]
        self.trace_event("nearest_order", picks={pq.name: lineup[pq.position].ident for pq in covered})
        for pid in stale:
            if self.steps_left() > len(missing) + 1:
                self.remove_line(pid)
        for pq in missing:
            if self.steps_left() < 2:
                break
            pq.choice = lineup[pq.position]
            self.add_line(pq, pq.choice)
        if self.steps_left() < 1 or not self.cart:
            return False
        moves_at = self.steps
        if self.place_order() or self.finished:
            return True
        # refused: what the checkout brought (a dropped need, a larger budget) may let something fit now
        self.nearest_rounds += 1
        moved = self.read_store_lines() or self.changed_at >= moves_at
        if not moved:
            return False
        self.extend_clock("order refused")
        if self.nearest_rounds > 2 or self.steps_left() < len(self.products) + 1:
            return False
        # unsure fits may be ordered again: the change can leave them the best known
        self.nearest_tried, self.unconfirmed_tried, self.nearest_picks = False, False, None
        sweep_after_news(self, moves_at)
        self.run(Event(PICKS_DUE))
        return self.finished

    def order_unconfirmed(self):
        """No rule for the nothing-fits case: rather than declining, order listings no requirement plainly fails
        (some unconfirmed), provided every product has one and the total is within budget."""
        if self.unconfirmed_tried or self.nearest_tried or not self.roles.get("checkout"):
            return False
        self.unconfirmed_tried = True
        choices = {}
        for pd in self.products:
            pick = self.unconfirmed_pick(pd)
            if pick is None:
                return False
            choices[pd.position] = pick
        total = sum((choices[wp.position].price or 0) * wp.quantity for wp in self.products)
        if self.budget is not None and total > self.budget + 1e-6:
            return False
        self.trace_event("unsure_order", picks={wp.name: choices[wp.position].ident for wp in self.products})
        self.nearest_picks = choices
        return self.order_nearest()

    def unconfirmed_pick(self, pd):
        """The product's listing for an unconfirmed order, or None (nothing left, or nothing passed the checker)."""
        line = ranked_fits(self, pd)
        if not line:
            usage = self.field_usage(pd)
            line = [c for c in pd.fits(include_unsure=True, cheapest=self.cheapest)
                    if c.available is not False and not self.field_failures(pd, c, usage) and not pd.loose_lines(c.ident)
                    and "cart refused" not in verdict_of(pd, c).get("failed", ())]
            if not line:
                return None
        # an order nobody verified seldom pays (real races 11-14: a careful decliner's no-fit declines would have
        # earned at most 1.7-5.1 per race as orders, against 3-4 paid declines): an unsure listing goes to the
        # cart only once the independent checker passes it
        for c in line[:2]:
            if low_on_time(self, 25):
                # no time left for the checker: a listing the judge found clean on every requirement goes in
                # unchecked rather than the whole order being declined (Arena, race-15 world: 24 of 87 tasks
                # declined at the clock, 5 of the 8 with a known answer had every paid listing read and judged
                # clean; real race 15: the crown declined 23 of 108 three-product runs, a single asker 39)
                for other in line:
                    verdict = verdict_of(pd, other)
                    if pd.judged_on.get(other.ident) and not verdict.get("failed") and not verdict.get("unsure"):
                        self.trace_event("unsure_clock_pass", item=pd.name, id=other.ident)
                        return other
                break
            if c.ident in self.rechecked_ids and not verdict_of(pd, c).get("unsure"):
                return c
            self.rechecked_ids.add(c.ident)
            if self.verify(pd, c, strict=True):
                return c
        self.trace_event("unsure_unverified", item=pd.name)
        return None

    def decline_remark(self, reason):
        """A plain closing note: what was looked for, the budget, and why nothing was bought."""
        name_parts = []
        for wp in self.products:
            name_parts.append(f"{wp.quantity} x {wp.name}" if wp.quantity > 1 else wp.name)
        sought = ", ".join(name_parts) or "the request"
        note = f"Search covered {sought}"
        if self.budget is not None:
            note += f" with at most {self.budget:g} to spend"
        note += "; this ends without a purchase: "
        first_w = "no fitting listing for "
        reason_txt = ""
        if reason.startswith(first_w):
            reason_txt = "none of the listings read for " + reason[len(first_w):] + " meets every requirement"
        elif reason.startswith("the cheapest fitting combination"):
            reason_txt = "the listings that meet every requirement cost more together than that"
        reason_txt = re.sub(r"\bp-[0-9a-f]{6,}\b", "", reason_txt).strip()
        return note + (reason_txt or "none of what the store shows meets every requirement") + "."

    def audit_search(self, missing):
        """Before declining, an independent look at everything seen for each product still without a fit (possible
        fits, unread rows, untried searches), acted on by code. True when something new came of it."""
        gained = False
        self.fixed_reserve = closing_reserve(self)
        try:
            for pd in missing:
                if low_on_time(self, 20):
                    break
                cards = self.audit_cards(pd)
                if not cards:
                    continue
                tried = sorted(queries_sent(pd))
                money = "none stated" if self.budget is None else self.budget
                message = (f"PRODUCT: {pd.name} (category: {pd.kind}; quantity {pd.quantity})\nBUDGET, SUM OF EVERY LINE: "
                           f"{money}\nREQUIREMENTS:\n- ") + ("\n- ".join(pd.demand_lines()) or "none") \
                    + f"\nQUERIES TRIED: {as_json(tried)}\n\nLISTINGS SEEN:\n" + compact(cards, 14000)
                answer = self.models.json_reply([PLAN_MODEL, FAST_MODEL, BACKUP_MODEL], AUDIT_RULES, message, 1500)
                if not isinstance(answer, dict):
                    continue
                self.trace_event("audit", item=pd.name, answer=compact(answer, 300))
                gained = self.act_on_audit(pd, answer, tried) or gained
                self.judge(pd, read_pool(self, pd))
        finally:
            self.fixed_reserve = None
        return gained

    def audit_cards(self, pd):
        """Up to 40 buyable listings of the right kind seen for the product, read ones first, cheapest first."""
        pool = [c for c in pd.seen.values()
                if self.is_kind(pd, c) and c.available is not False and not over_cap(self.budget, c, pd.quantity)]
        cards = []
        for c in sorted(pool, key=lambda c: (not c.fully_read, sort_price(c)))[:40]:
            card = {"id": c.ident, "title": c.heading[:110], "price": c.price, "read": c.fully_read}
            if c.fully_read:
                verdict = verdict_of(pd, c)
                card.update({"failed": sorted(verdict.get("failed") or [])[:3],
                             "unsure": sorted(verdict.get("unsure") or [])[:3],
                             "properties": dict(list(c.table.items())[:16])})
                if c.remarks:
                    card["notes"] = c.remarks[:200]
            else:
                card["text"] = (c.blurb or "")[:160]
            cards.append(card)
        return cards

    def act_on_audit(self, pd, answer, tried):
        """Carry out an audit answer: recheck named fits, open named rows, run untried queries. True on a gain."""
        gained = False
        for pid in [str(x) for x in answer.get("fits_after_all") or []][:2]:
            c = pd.seen.get(pid)
            if c is None or not c.fully_read or self.field_failures(pd, c):
                continue
            if self.verify(pd, c, strict=True):
                # a second independent reading agrees: set the old verdict aside
                pd.verdicts[c.ident] = clean_verdict()
                pd.judged_on[c.ident] = set(pd.demand_lines())
                gained = True
        unopened = [str(x) for x in answer.get("unopened") or [] if str(x) in pd.seen and not pd.seen[str(x)].fully_read][:4]
        if unopened and room_left(self) >= 1 and self.read_listings(pd, unopened, keep_steps=self.closing_steps()):
            gained = True
        queries = [str(x).strip() for x in answer.get("queries") or [] if str(x).strip()]
        earlier = set(plain_text(t) for t in tried)
        for query in queries[:2]:
            if room_left(self) < 3 or low_on_time(self, 15):
                break
            if plain_text(query) in earlier \
                    or self.run_search(pd, query=query[:60], pages=1, budget=self.closing_steps()) <= 0:
                continue
            self.screen_rows(pd, [c for c in self.candidates(pd) if not c.fully_read])
            fresh = [c for c in self.candidates(pd) if not c.fully_read and c.screen != "no"][:2]
            if fresh and self.read_listings(pd, [c.ident for c in fresh], keep_steps=self.closing_steps()):
                gained = True
        return gained


def whole_cart_only(ep):
    """True when the store's only cart tool replaces the entire cart."""
    return bool(not ep.roles.get("cart_add") and ep.roles.get("cart_set"))


def change_cart(ep, line_ref, owner_p):
    """Add a cart line for the owner's product, or remove it when owner is None, using the replacing tool if that
    is the store's only cart tool. True when the store accepted the change."""
    adding = owner_p is not None
    if whole_cart_only(ep):
        lines = {ref: n for ref, n in ep.cart.items() if adding or ref != line_ref}
        if adding:
            lines[line_ref] = owner_p.quantity
        if not ep.set_cart(lines):
            return False
        ep.cart = lines
        if adding:
            ep.line_owner[line_ref] = owner_p
        else:
            ep.line_owner.pop(line_ref, None)
        return True
    tool = ep.roles.get("cart_add")
    if not tool:
        return False
    args, qty = cart_args(ep, tool, line_ref, owner_p)
    reply = ep.call_tool(tool["name"], args)
    if not adding:
        if reply["ok"]:
            ep.cart.pop(line_ref, None)
            ep.line_owner.pop(line_ref, None)
        return bool(reply["ok"])
    if reply["ok"] and not cart_reply_refuses(reply["obs"]):
        ep.cart[line_ref] = qty
        ep.line_owner[line_ref] = owner_p
        return True
    ep.cart_rejects_args = bool(reply.get("bad_args"))
    ep.trace_event("cart_add_failed", error=reply["error"], obs=compact(reply["obs"], 200))
    return False


def cart_args(ep, tool, ref, owner):
    """The cart tool's arguments for adding the owner's line (owner set) or removing it, and the quantity added."""
    args = {tool["id"]: ref}
    mode = mode_switch(tool)
    field = tool.get("qty")
    if owner is None:
        if mode and mode.get("arg") and mode.get("remove") is not None:
            args[mode["arg"]] = mode["remove"]
            if field:
                args[field] = ep.cart.get(ref, 1)
        elif field:
            low = arg_schema(ep, "cart_add", field).get("minimum")
            args[field] = low if isinstance(low, int) and low > 0 else 0
        return args, None
    qty = owner.quantity
    if field:
        top = arg_schema(ep, "cart_add", field).get("maximum")
        if isinstance(top, int) and qty > top:
            qty = top
        args[field] = qty
    if mode and mode.get("arg") and mode.get("add") is not None:
        args[mode["arg"]] = mode["add"]
    return args, qty


def final_search(ep, unfit):
    """One more search for the products without a pick, keeping just the steps the close needs; True if one fits
    now. The decline keeps steps too: a closing call may bring a new budget or a dropped need (race 12)."""
    ep.fixed_reserve = closing_reserve(ep)
    try:
        for pq in unfit:
            if room_left(ep) >= 3:
                pq.exhausted = False
                ep.hunt(pq, room_left(ep), final=True)
    finally:
        ep.fixed_reserve = None
    return any(ep.buyable(pq) for pq in unfit)


def closing_reserve(ep):
    """Steps held for the close during a last search: cart adds, the shopper's yes, checkout and retries."""
    return len(ep.products) + 5 + max(ep.reserved_messages(), 1 if ep.needs_approval else 0)


class Pilot:
    """A model pilot for the moments the routine cannot settle alone."""

    def pilot_reserve(self):
        """Steps withheld from the pilot: a cart add per product, the closing call, a retry and owed messages."""
        return self.messages_owed() + len(self.products) + 2

    def unused_tools(self):
        """Store tools that no routine role uses."""
        used_names = set()
        for tool_spec in self.roles.values():
            if isinstance(tool_spec, dict):
                used_names.add(tool_spec.get("name"))
        added_q = []
        for tt in self.tool_entries:
            fdef = tt.get("function")
            if not isinstance(fdef, dict):
                fdef = tt
            if not isinstance(fdef, dict) or not fdef.get("name"):
                continue
            nm = fdef["name"]
            if nm in used_names:
                continue
            added_q.append({"name": nm, "description": str(fdef.get("description") or "")[:300],
                            "parameters": compact(fdef.get("parameters") or {}, 300), "called": nm in self.tools_used})
        return added_q

    def pilot_look_due(self):
        if self.pilot_looked:
            return False
        if low_on_time(self, 40) or self.models.spent >= PILOT_SPEND:
            return False
        if self.steps_left() < self.pilot_reserve() + 1:
            return False
        if len(self.unplaced_lines) > self.lines_shown:
            return True
        for tt in self.unused_tools():
            if not tt["called"]:
                return True
        return False

    def pilot_record(self):
        """The pilot's view: allowances, each product's needs and findings, the cart, store lines, recent calls."""
        held = self.pilot_reserve()
        asks = 0
        if self.can_ask():
            asks = 1 if self.message_cap is None else self.message_cap - self.messages_sent - self.reserved_messages()
        items = [self.pilot_item(wp) for wp in self.products]
        steps = self.steps_left()
        return {"allowances": {"steps_left": steps, "steps_you_may_use": max(0, steps - held),
                               "questions_left": max(0, asks),
                               "seconds_left": max(0, int(self.time_left() - END_MARGIN))},
                "budget_total": self.budget, "cheapest_wanted": self.cheapest, "best_by": self.measure, "items": items,
                "cart": [{"id": pid, "quantity": n} for pid, n in self.cart.items()],
                "store_lines": [line[:240] for line in self.store_lines[-8:]],
                "lines_the_routine_found_no_change_in": [line[:240] for line in self.unplaced_lines[-4:]],
                "latest_calls": self.call_log[-6:], "other_tools": self.unused_tools()}

    def pilot_item(self, wp):
        """One product as the pilot sees it: needs, fits, unsure and near-miss pages, unread rows, queries tried."""
        fits = self.buyable(wp)
        fit_ids = set(c.ident for c in fits)
        misses, doubtful = [], []
        for c in wp.seen.values():
            if not c.fully_read or c.ident in fit_ids or not self.is_kind(wp, c):
                continue
            verdict = verdict_of(wp, c)
            row = brief_card(c)
            if c.available is False:
                row["can_buy"] = False
            if verdict.get("failed"):
                row["failed"] = sorted(verdict["failed"])[:3]
                misses.append(row)
            elif verdict.get("unsure") or wp.loose_lines(c.ident):
                row["not_shown"] = sorted(set(verdict.get("unsure") or ()) | set(wp.loose_lines(c.ident)))[:3]
                doubtful.append(row)
        misses.sort(key=lambda r: (len(r["failed"]), 1e18 if r["price"] is None else r["price"]))
        unread = [brief_card(c) for c in self.candidates(wp) if not c.fully_read and c.screen != "no"]
        block = {"name": wp.name, "category": wp.kind, "quantity": wp.quantity}
        if wp.unit_cap is not None:
            block["price_cap_each"] = wp.unit_cap
        block.update({"requirements": wp.demand_lines(), "asked_shopper_about": wp.asked[:8],
                      "fits": [brief_card(c) for c in fits[:4]], "unsure": doubtful[:3], "near_misses": misses[:5],
                      "rows_not_read": unread[:6], "queries_tried": sorted(queries_sent(wp))[:10],
                      "search_complete": bool(wp.exhausted or wp.values_complete)})
        return block

    def refuse_finish(self):
        """Once per session, turn down a finish while some product has no fit, an unfinished search and promising
        unread rows the steps could still cover."""
        state_d = self.pilot_state
        if state_d["pushed"]:
            return ""
        if self.steps_left() - self.pilot_reserve() < 1:
            return ""
        for wp in self.products:
            if wp.exhausted or wp.values_complete or self.buyable(wp):
                continue
            rows_out = []
            for cand in self.candidates(wp):
                if not cand.fully_read and cand.screen != "no":
                    rows_out.append(cand.ident)
            rows_out = rows_out[:4]
            if not rows_out:
                continue
            state_d["pushed"] = True
            return f"no fit for {wp.name} and its search is not complete; rows not read: {rows_out}"
        return ""

    def move_summary(self, prod, rows, opened, fits_before):
        verdict_map = {}
        for cand in opened:
            val = verdict_of(prod, cand)
            if val.get("failed"):
                said_t = "fails " + "; ".join(sorted(val["failed"])[:2])
            elif val.get("unsure"):
                said_t = "not shown: " + "; ".join(sorted(val["unsure"])[:2])
            else:
                said_t = "meets every requirement"
            verdict_map[cand.ident] = said_t
        fit_now = [cand.ident for cand in self.buyable(prod)]
        gained = False
        for ix in fit_now:
            if ix not in fits_before:
                gained = True
                break
        return f"{rows} new rows; read {verdict_map or 'none'}; fits now {fit_now[:4]}", gained

    def pilot_news(self, move):
        """A change the pilot attributes to a store line: the line has to exist and state the change itself (the new
        amount, or the dropped requirement)."""
        claim = str(move.get("line") or "").strip()
        quoted = next((ln for ln in reversed(self.store_lines) if claim[:60] in ln or overlap(claim, ln) >= 0.6),
                      None) if claim else None
        if quoted is None:
            return unchanged("not done: no store line says that")
        act = {"budget": self.news_budget, "drop": self.news_drop, "gone": self.news_gone}.get(
            str(move.get("kind") or "").lower())
        return act(move, quoted) if act else unchanged("not done: kind must be budget, drop or gone")

    def news_budget(self, move, quoted):
        amount = to_number(move.get("budget"))
        if amount is None or amount not in [float(t.replace(",", "")) for t in re.findall(r"\d[\d,]*(?:\.\d+)?", quoted)]:
            return unchanged("not done: the line does not state that amount")
        self.budget = amount
        self.changed_at = self.steps
        self.pilot_state["news"] = True
        self.trace_event("pilot_news", change="budget", budget=amount)
        return changed(f"budget now {amount:g}")

    def news_drop(self, move, quoted):
        prop = str(move.get("property") or "")
        who = self.product_named(move.get("item"))
        scope = [who] if who else self.products
        bag = word_set(quoted)
        if not any(word_set(rq.prop) & bag or word_set(rq.value) & bag
                   for pq in scope for rq in pq.demands if overlap(rq.prop, prop) >= 0.5):
            return unchanged("not done: the line does not name that requirement")
        if not self.drop_demand(scope, prop):
            return unchanged("not done: no such requirement")
        self.changed_at = self.steps
        self.pilot_state["news"] = True
        self.trace_event("pilot_news", change="drop", property=prop)
        return changed(f"requirement on {prop} dropped")

    def news_gone(self, move, quoted):
        lid = str(move.get("id") or "")
        hit = next((pq.seen[lid] for pq in self.products if lid in pq.seen), None)
        if hit is None or not (lid in self.cart or any(pq.choice is hit for pq in self.products)):
            return unchanged("not done: name a listing in the cart or picked")
        hit.available = False
        self.changed_at = self.steps
        self.trace_event("pilot_news", change="gone", id=lid)
        return changed(f"{lid} marked as not buyable")

    def recheck_unsure(self):
        """Before declining, for each product without a fit the independent checker rereads at most two listings that
        fail nothing plainly (merely unconfirmed); one it passes becomes a fit."""
        found = False
        for pd in self.products:
            if self.buyable(pd) or low_on_time(self, 25):
                continue
            usage = self.field_usage(pd)
            pool = []
            for c in pd.seen.values():
                verdict = verdict_of(pd, c)
                doubtful = c.fully_read and not verdict.get("failed") and (verdict.get("unsure") or pd.loose_lines(c.ident))
                if doubtful and c.available is not False and not pd.over_unit_cap(c) and self.is_kind(pd, c) \
                        and not self.field_failures(pd, c, usage) and not over_cap(self.budget, c, pd.quantity) \
                        and c.ident not in self.rechecked_ids:
                    pool.append(c)
            pool.sort(key=lambda c: (len(verdict_of(pd, c).get("unsure") or ()), sort_price(c)))
            for c in pool[:2]:
                self.rechecked_ids.add(c.ident)
                if self.verify(pd, c, strict=True):
                    pd.verdicts[c.ident] = clean_verdict()
                    pd.judged_on[c.ident] = set(pd.demand_lines())
                    self.trace_event("recheck_fit", item=pd.name, id=c.ident)
                    found = True
                    break
        return found

    def last_look(self, reason):
        """Before a decline for want of a fit: one model reading of the read listings no reader plainly failed, for the
        products still without a pick. True when every product then has a pick inside the budget (the cart follows)."""
        # Arena, race-15 world, build 46: 5 of 11 declines with a known answer had every line of a paid basket read and
        # passed by the reader, but softer gates (an unjudged later line, a store-field doubt, an unsure word) kept them
        # out; the review before a decline ran in 3 of 33, as it needs 70 s while declines came at 205-228 s
        if self.last_look_done or self.finished or not str(reason or "").startswith("no fitting listing"):
            return False
        if not self.products or not self.roles.get("checkout") or self.time_left() < LAST_LOOK_LEFT:
            return False
        picks = dict(self.nearest_picks or {})
        unfit = [pd for pd in self.products if pd.position not in picks]
        if not unfit or self.steps_left() < len(unfit) + 2:
            return False
        # condition outcomes stay with the code; a fit rule the code could not settle is read here with the chosen
        # device's record in view (a fit the code proved never reaches this point: that product has its pick)
        linked = set()
        for rule in self.conditions:
            if rule["src"].position in picks:
                # the source's pick already set the condition's line on its targets: the review reads that line
                continue
            linked.add(rule["src"].position)
            linked.update(target.position for _, target, _ in rule["branches"])
        if any(pd.position in linked for pd in unfit):
            return False
        decks = {}
        for pd in unfit:
            decks[pd.position] = self.last_look_cards(pd)
            if not decks[pd.position]:
                return False
        self.last_look_done = True
        ceiling = "none stated" if self.budget is None else self.budget
        rule = "the cheapest listing that meets every requirement" if self.cheapest else (
            "the best by: " + self.measure if self.measure else "any listing that meets every requirement")
        held = [{"item": pd.name, "quantity": pd.quantity, "id": c.ident, "title": c.heading[:140], "price": c.price,
                 "properties": dict(list(c.table.items())[:12])}
                for pd in self.products for c in [picks.get(pd.position)] if c is not None]
        bonds = [f"{part.name} must work with the {host.name}" for part, host in self.fit_pairs()]
        if bonds:
            parts_note = "FIT RULES: " + "; ".join(bonds)
        else:
            parts_note = ""
        parts = [f"REQUEST: {self.request_text[:1500]}", f"BUDGET, SUM OF EVERY LINE: {ceiling}",
                 f"WHICH LISTING THE SHOPPER WANTS: {rule}", "ALREADY CHOSEN: " + (as_json(held) if held else "none")]
        if parts_note:
            parts.append(parts_note)
        for num, pd in enumerate(unfit, 1):
            parts.append(f"PRODUCT {num}: {pd.name} (category: {pd.kind}; quantity {pd.quantity})\nREQUIREMENTS:\n- "
                         + ("\n- ".join(pd.demand_lines()) or "none beyond the kind")
                         + "\nLISTINGS:\n" + compact(decks[pd.position], 7000))
        before = self.models.late_floor
        self.models.late_floor = 8.0
        try:
            answer = self.models.json_reply([VERIFY_MODEL, FAST_MODEL], LAST_LOOK_RULES, "\n\n".join(parts), 1500)
        finally:
            self.models.late_floor = before
        named = answer.get("picks") if isinstance(answer, dict) else None
        if not isinstance(named, dict):
            self.trace_event("last_look", picks=None)
            return False
        options = {}
        for num, pd in enumerate(unfit, 1):
            allowed = {card["id"] for card in decks[pd.position]}
            ids = [str(x) for x in (named.get(str(num)) or []) if str(x) in allowed]
            found = [pd.seen[x] for x in ids]
            if self.cheapest:
                found.sort(key=sort_price)
            if not found:
                self.trace_event("last_look", picks=compact(named, 200), none_for=pd.name)
                return False
            options[pd.position] = found
        chosen = {pos: found[0] for pos, found in options.items()}
        if self.budget is not None:
            fixed = sum((c.price or 0) * pd.quantity for pd in self.products for c in [picks.get(pd.position)] if c)
            total = fixed + sum((chosen[pd.position].price or 0) * pd.quantity for pd in unfit)
            if total > self.budget + 1e-6:
                chosen = {pos: min(found, key=sort_price) for pos, found in options.items()}
                total = fixed + sum((chosen[pd.position].price or 0) * pd.quantity for pd in unfit)
                if total > self.budget + 1e-6:
                    self.trace_event("last_look", over_budget=total)
                    return False
        for pd in unfit:
            pick = chosen[pd.position]
            pd.verdicts[pick.ident] = clean_verdict()
            pd.judged_on[pick.ident] = set(pd.demand_lines())
            self.checker_failed.discard((pd.position, pick.ident))
        for pd in self.products:
            pd.choice = picks.get(pd.position) or chosen[pd.position]
        self.trace_event("last_look", chosen={pd.name: pd.choice.ident for pd in self.products})
        return True

    def last_look_cards(self, pd):
        """Read listings of the product's kind, in stock and inside the caps, that no reader failed (a failure by the
        checker alone stays in): the soft doubts on each are shown, fewest doubts first, then by price."""
        usage = self.field_usage(pd)
        lines = set(pd.demand_lines())
        rows = []
        for c in pd.seen.values():
            if not c.fully_read or c.available is False or not self.is_kind(pd, c):
                continue
            if over_cap(self.budget, c, pd.quantity) or pd.over_unit_cap(c):
                continue
            verdict = verdict_of(pd, c)
            failed = set(verdict.get("failed") or ())
            if "cart refused" in failed or (failed and (pd.position, c.ident) not in self.checker_failed):
                continue
            doubts = sorted(set(verdict.get("unsure") or ()) | set(pd.loose_lines(c.ident)))
            doubts += sorted(failed) + sorted(self.field_failures(pd, c, usage))
            doubts += sorted(lines - set(pd.judged_on.get(c.ident) or ()))
            card = {"id": c.ident, "title": c.heading[:140], "price": c.price,
                    "properties": dict(list(c.table.items())[:20]), "text": str(c.blurb or "")[:240]}
            if c.remarks:
                card["notes"] = c.remarks[:240]
            if doubts:
                card["doubted on"] = list(dict.fromkeys(doubts))[:4]
            rows.append((len(doubts), sort_price(c), card))
        rows.sort(key=lambda row: (row[0], row[1]))
        return [row[2] for row in rows[:10]]

    def review_before_decline(self, reason):
        """Last look before ending without a purchase: recheck unsure listings, then let the pilot review the
        record (once, and again after a refused exit). True when the picks should be chosen again."""
        if self.finished or not self.review_pending or reason in ("forced", "the cart tool refuses its arguments"):
            return False
        if not self.products or low_on_time(self, 30):
            return False
        self.review_pending = False
        any_hit = self.recheck_unsure()
        if not any_hit and self.steps_left() >= self.pilot_reserve() and self.models.spent < PILOT_SPEND:
            reason_txt = "Nothing it may buy was found for some product: " + (reason or "no fit")
            any_hit = self.pilot(reason_txt, self.last_refusal, turns=PILOT_TURNS)
            if any_hit and self.pilot_state.get("news"):
                # budget raised or need dropped: earlier rejects deserve another look
                self.last_search_done = False
                self.audited = False
        if any_hit:
            self.trace_event("decline_review", found=True)
        return any_hit


def pilot_prompt(ep, point, detail, history):
    """The pilot's input: the request, the reason it was called, the record and its moves so far."""
    text_s = "REQUEST:\n" + ep.request_text + "\n\nWHY YOU ARE CALLED: " + point
    if detail:
        text_s += "\nDETAIL: " + compact(detail, 800)
    text_s += "\n\nRECORD:\n" + compact(ep.pilot_record(), 9000)
    if history:
        text_s += "\n\nYOUR MOVES SO FAR:\n" + "\n".join(history)
    return text_s


def brief_card(c):
    """A listing's id, short title and price."""
    return {"id": c.ident, "title": c.heading[:90], "price": c.price}


def near_spelling(a, b):
    """Same word (singular or plural), or one letter apart in words of five letters or more."""
    a, b = singular(a), singular(b)
    if a == b:
        return True
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    if len(shorter) < 5 or len(longer) - len(shorter) != 1:
        return False
    return any(shorter == longer[:i] + longer[i + 1:] for i in range(len(longer)))


def shown_values(pd, prop):
    """Up to four values the product's listings write for the property, each cut to 30 characters."""
    values = []
    for c in pd.seen.values():
        for key, val in c.table.items():
            text = str(val)[:30]
            if text not in values and same_name(split_camel(key).replace("_", " "), prop):
                values.append(text)
    return values[:4]


def padded_flat(text):
    """Plain words of a text on single spaces, padded with a space at each end."""
    return " " + re.sub(r"\s+", " ", plain_text(text)) + " "


def unchanged(note):
    """Outcome of a pilot move that left the buyable set as it was."""
    return note, False


def changed(note):
    """Outcome of a pilot move that altered the buyable set."""
    return note, True


# -------------------- the event-driven controller

REQUEST_READ = "request_read"
REFERENCES_DUE = "references_due"
OPENING_STEP = "opening_step"
FIRST_PASS = "first_pass"
STATED_VALUES = "stated_values"
QUESTIONS_DUE = "questions_due"
QUESTION_DUE = "question_due"
EXTRA_QUESTION = "extra_question"
ASK = "ask"
SHOPPER_REPLIED = "shopper_replied"
FINAL_PASS = "final_pass"
SEARCH_ROUND = "search_round"
PICKS_DUE = "picks_due"
CART_READY = "cart_ready"
CHECKOUT_REFUSED = "checkout_refused"
REPICK = "repick"
NOTHING_FITS = "nothing_fits"
DECLINE_REFUSED = "decline_refused"
FORCED_CLOSE = "forced_close"
PILOT_TURN = "pilot_turn"
PILOT_MOVE = "pilot_move"


class Event:
    """Something that happened in the episode, with the facts its handler needs."""

    __slots__ = ("kind", "facts")

    def __init__(self, kind, **facts):
        self.kind = kind
        self.facts = facts


class SearchRun:
    """One product's search, kept across its rounds: the step floor, the query list and the paging cursor."""

    def __init__(self, prod, floor, queries, enough, sweep, final, hold):
        self.prod = prod
        self.floor = floor
        self.queries = queries
        self.cursor = [first_unsent(prod, queries)]
        self.enough = enough
        self.sweep = sweep
        self.final = final
        self.hold = hold


class Controller:
    """Runs the episode as events: each handler does one job and returns the events that follow from it.

    Follow-up events run before anything queued earlier, so a handler's consequences play out first, the way a
    call returns before its caller goes on. A handler that needs the outcome of a whole sub-flow before it can go
    on (a decline that first lets the picks be chosen again) runs that sub-flow to its end with a nested run."""

    def run(self, first):
        pending = [first]
        while pending:
            event = pending.pop(0)
            pending[:0] = self.HANDLERS[event.kind](self, **event.facts)

    def go(self):
        try:
            if not self.read_brief():
                self.run(Event(FORCED_CLOSE))
                return self.trace
            self.run(Event(REQUEST_READ))
        except Exception as exc:   # an episode must always end
            self.trace_event("exception", error=repr(exc)[:300])
        try:
            if not self.finished:
                self.run(Event(FORCED_CLOSE))
        except Exception as exc:
            self.trace_event("exception", error=repr(exc)[:300])
        self.trace_event("done", cost=round(self.models.spent, 4), llm_calls=self.models.calls, steps=self.steps)
        return self.trace

    # ---------------------------------------------------------------- from the request to the questions
    def on_request_read(self):
        # no stated cap: plan on 4 messages (caps seen were 4 or 5); the extra-question round keeps asking while allowed
        cap = 4 if self.message_cap is None else self.message_cap
        self.question_count = cap - self.reserved_messages()
        if self.question_count < 0 or "message" not in self.roles:
            self.question_count = 0
        self.spare_question = 1 if self.question_count else 0
        return [Event(REFERENCES_DUE, early=True), Event(OPENING_STEP), Event(FIRST_PASS), Event(STATED_VALUES),
                Event(QUESTIONS_DUE), Event(REFERENCES_DUE, early=False), Event(FINAL_PASS), Event(PICKS_DUE)]

    def on_references_due(self, early):
        """Trace the demands that quote another listing's page. Early (from the request itself) the question and
        closing steps stay untouched; late (a reply may have added some) only the closing steps are kept."""
        if early:
            self.trace_floor = (2 * len(self.products) + self.question_count + self.spare_question
                                + self.closing_steps())
        for prod in self.products:
            for dm in list(prod.demands):
                if not dm.reference or dm.op not in REFERENCE_OPS:
                    continue
                if not early:
                    self.trace_floor = self.closing_steps() + 2
                self.trace_reference(prod, dm)
                if not early:
                    self.trace_floor = 0
        self.trace_floor = 0
        return []

    def on_opening_step(self):
        """Share out the first search pass, then run a demanded opening step that is not a message (an opening
        message becomes one of the questions)."""
        # the questions (plus one re-ask) and the ending are held back from the first pass
        free = self.steps_left() - self.question_count - self.spare_question - self.closing_steps()
        self.first_share = max(2, free // max(len(self.products), 1)) if free > 0 else 0
        for step_e in self.steps_demanded:
            if step_e["moment"] == "start" and step_e.get("kind") != "message":
                self.run_process_gate("start", "start the task (search the store)")
                self.read_store_lines()
                if self.message_cap is not None:
                    self.question_count = min(self.question_count,
                                              self.message_cap - self.messages_sent - self.reserved_messages())
                self.question_count = max(0, self.question_count)
                break
        return []

    def on_first_pass(self):
        """First search and read per product. On a slow run it stops early enough for the questions to be sent and
        their replies read (an unread reply leaves a told need out of the pick)."""
        count = len(self.products)
        self.question_hold = self.question_count * QUESTION_SECONDS + 15.0 if self.question_count > 0 else 0.0
        if count > 1:
            # the work after the questions (fit to the chosen host, told needs, the final searches) needs time of its
            # own; without it the first pass ran until the questions and the fits were never searched
            self.question_hold += min(AFTER_ASK_HOLD * (count - 1), 2 * AFTER_ASK_HOLD)
        try:
            for prod in self.products:
                # several products: each gets the same share before the questions; steps one leaves unused wait
                # for the final searches, when the told needs say what to look for
                share = self.first_share
                if count == 1:
                    share = max(share, self.steps_left() - self.question_count - 1 - self.closing_steps())
                if share <= 0:
                    break
                before = self.steps_left()
                self.hunt(prod, max(2, share // 2))
                self.read_sample(prod, max(0, share - (before - self.steps_left())))
        finally:
            self.question_hold = 0.0
        return []

    def on_stated_values(self):
        self.read_stated_values()
        return []

    def on_questions_due(self):
        """Questions name one product each and list up to LIST_WIDTH of its properties: on real race 14 the shopper
        answered a hidden need wherever the list named it (need replies .34 of list messages against .17-.18 of
        one-property ones, five agents). A reply asking for one subject at a time turns lists off."""
        if self.question_count <= 0 or not self.can_ask():
            return []
        # One product asks lists too. Real race 15, one-product order tasks: a hidden need came back on .24 of
        # messages naming 7 or more properties against .15 of single ones, and the list-asking leader drew 32 of 44
        # such needs against 24 for a single asker. With two messages or fewer one product keeps single questions:
        # a strict shopper's split then costs half of them (Arena, race-13 world, cap 2)
        if len(self.products) < 2 and self.message_cap is not None and self.message_cap < 3:
            self.single_questions = True
        self.question_queue = self.plan_questions(self.question_count)
        return [Event(QUESTION_DUE)]

    def on_question_due(self):
        """The next planned question, unless the messages or steps ran out (then the extra round decides)."""
        queue = self.question_queue
        if not queue or not self.can_ask() or self.steps_left() <= self.closing_steps():
            return [Event(EXTRA_QUESTION)]
        prod, prop_n = queue.pop(0)
        if prod.unclear_replies and any(not other.unclear_replies for other, _ in queue):
            # real race 15, two or more products: after a "say it again" reply a later question on that product
            # drew the same reply .97 of the time (a need .03), one on another product a need .18
            queue.append((prod, prop_n))
            return [Event(QUESTION_DUE)]
        known = any(overlap(dm.prop, prop_n) >= 0.67 for dm in prod.demands) or self.mentioned_before(prod, prop_n)
        # a pick an earlier message already asked (it led a list in place of a pick named before) is not sent
        # again: the same text gets the same reply
        if known or any(same_name(prop_n, other_name) for other_name in prod.asked):
            return [Event(QUESTION_DUE)]
        return [Event(ASK, product=prod, prop=prop_n, planned=True, no_view_before=len(prod.no_view))]

    def on_extra_question(self):
        """Messages left after the plan go to further properties, since every hidden need has to come out."""
        if not (self.can_ask() and self.steps_left() > self.closing_steps() + 1 and time_for(self, 30)):
            return []
        plan = self.plan_questions(len(self.products))
        if not plan:
            return []
        lists_on = not self.single_questions

        def turn(pair):
            # a list per message: the product with the fewest needs heard gets the next one; a product whose question
            # drew a "say it again" reply goes last
            heard_n = sum(1 for dm in pair[0].demands if dm.origin == "shopper") if lists_on else 0
            return (pair[0].unclear_replies > 0, heard_n)
        plan.sort(key=turn)
        return [Event(ASK, product=plan[0][0], prop=plan[0][1], planned=False)]

    def message_props(self, prod, prop_n):
        """The lead and the further properties of the product for one message: its later planned picks, then its
        menu (widely carried first), none asked, settled or known. A planned pick that an earlier list already named
        hands the lead to the best property no list named yet and follows it."""
        # real race 14: a need re-asked first after an earlier listing was revealed 4 times of 17, a fresh first .85
        if self.single_questions:
            return prop_n, []
        named = []

        def still_open(pname):
            if any(same_name(pname, other_name) for other_name in named + prod.asked + prod.reworded):
                return False
            if any(same_name(pname, dm.prop) for dm in prod.demands) or self.mentioned_before(prod, pname):
                return False
            return not any(near_name(pname, fixed) for fixed in self.fixed_properties(prod))
        lined = [pname for owner, pname in self.question_queue if owner is prod]
        lined += [row_d["property"] for row_d in common_entries(self.property_menu(prod))]
        # a store key whose name is itself several names ("a, b, c") is no single property the shopper keeps
        lined = [pname for pname in lined if pname == prop_n or "," not in str(pname)]
        listed_before = set(prod.listed_props)
        lead = prop_n
        if plain_text(prop_n) in listed_before:
            named.append(prop_n)
            fresh = next((pname for pname in lined if plain_text(pname) not in listed_before and still_open(pname)),
                         None)
            named.pop()
            if fresh is not None:
                lead = fresh
        named.append(lead)
        if lead != prop_n:
            named.append(prop_n)
        # the task's first list names two properties. Real races 9-13: a strict shopper splits a list more often the
        # more subjects it names; a lenient shopper (races 8, 14) answers the first two places alike, so later lists
        # carry the full width
        width = LIST_WIDTH if self.lists_asked else 2
        for pname in lined:
            if len(named) >= width:
                break
            if still_open(pname):
                named.append(pname)
        return lead, named[1:]

    def on_ask(self, product, prop, planned, no_view_before=None):
        prod, prop_n = product, prop
        demands_before = len(prod.demands)
        lead, more = self.message_props(prod, prop_n)
        if lead != prop_n:
            self.trace_event("fresh_lead", item=prod.name, lead=lead, listed_before=prop_n)
        reply = self.ask_shopper(prod, lead, also=more)
        if more:
            self.lists_asked += 1
        return [Event(SHOPPER_REPLIED, product=prod, prop=prop_n, planned=planned, no_view_before=no_view_before,
                      reply=reply, lead=lead, more=more, demands_before=demands_before)]

    def on_shopper_replied(self, product, prop, planned, no_view_before, reply, lead, more, demands_before):
        prod, prop_n = product, prop
        kind = str(reply.get("kind") or "").lower() if isinstance(reply, dict) else ""
        # a "repeat" with no need of the product heard yet restates nothing: it asked for the message again
        lost = kind == "unclear" or (kind == "repeat" and not any(dm.origin == "shopper" for dm in prod.demands))
        if lost and len(self.products) > 1:
            prod.unclear_replies += 1
            self.trace_event("unclear_moved_on", item=prod.name)
        if kind == "split":
            gone = self.settle_split(prod, prop_n, lead, more)
            # a split marks a strict shopper: one subject per message from here on
            self.single_questions = True
            if more:
                # no listed property counts as asked; the split list's lead goes next on its own (real races 9-13:
                # each key of a split list asked alone with its product revealed .91-1.00)
                prod.asked = [other_name for other_name in prod.asked if other_name not in gone]
                self.question_queue.insert(0, (prod, lead))
            self.trace_event("lists_off", item=prod.name, unasked=[lead] + more if more else [])
        elif more:
            self.keep_list_open(prod, lead, more, kind, reply)
        self.apply_reply_guarded(prod, lead, reply)
        self.read_store_lines()
        if not planned:
            return [Event(EXTRA_QUESTION)]
        queue = self.question_queue
        if len(prod.demands) != demands_before and queue:
            # a need was heard: the rest is planned again so its property (and aliases) is skipped
            room = len(queue)
            if self.message_cap is not None:
                room = min(room, self.message_cap - self.messages_sent - self.reserved_messages())
            self.question_queue = self.plan_questions(room)
        elif queue and len(prod.no_view) > no_view_before and self.single_questions:
            # "no preference": the next planned question may only reword the declined property
            self.question_queue = self.reconsider_next(prod, queue)
        return [Event(QUESTION_DUE)]

    def settle_split(self, prod, prop_n, lead, more):
        """A strict shopper answers a list it does not split for a need named in any place (real races 9-13:
        .84-1.00), so the places after the lead of its earlier unsplit lists are settled. The split message's own
        properties stay open on its product. Returns those properties (flattened)."""
        gone = {plain_text(pname) for pname in [lead] + more} if more else set()
        if more and lead != prop_n:
            # the planned pick sat in an earlier unsplit list and was answered there
            gone.discard(plain_text(prop_n))
        cleared = []
        for other in self.products:
            done_here = [pname for pname in set(other.listed_props)
                         if not (other is prod and pname in gone) and pname not in other.asked]
            other.asked.extend(done_here)
            other.listed_props = []
            if done_here:
                cleared.append((other, set(done_here)))
        if cleared:
            self.question_queue[:] = [(owner, pname) for owner, pname in self.question_queue
                                      if not any(owner is done and plain_text(pname) in keys for done, keys in cleared)]
            self.trace_event("strict_cleared", keys={done.name: sorted(keys) for done, keys in cleared})
        return gone

    def keep_list_open(self, prod, lead, more, kind, reply):
        """Only a list's lead counts as asked: on real race 14 its later places often drew no answer, and 9% of the
        list-asking leader's reveals came from such a property listed again later. A shopper answers the first need
        a list names, so the places before an answered one were heard and hold none (real race 15: .40 of list
        reveals came from the first place, .16 the second, .12 the third)."""
        gone = {plain_text(pname) for pname in more}
        if kind == "need":
            named = [lead] + list(more)
            heard = str(reply.get("property") or "")
            places = [ix for ix, pname in enumerate(named)
                      if heard and (same_name(heard, pname) or overlap(heard, pname) >= 0.67)]
            if places:
                gone -= {plain_text(pname) for pname in named[:places[0]]}
        prod.asked = [other_name for other_name in prod.asked if other_name not in gone]
        prod.listed_props = list(set(prod.listed_props) | gone)

    # ---------------------------------------------------------------- searching
    def on_final_pass(self):
        """Cart the best known pick, judge everything open, settle fit hosts, then search more where no fit exists
        (products without a fit first)."""
        self.early_wanted = True
        self.cart_early()
        self.judge_all()
        # fit rules ("X must fit Y"): settle the host by what the accessories fit, before the last searches
        plan_host(self, True)
        # accessories of fit rules are judged against their host's best pick now, while time remains
        bind_accessories(self)
        order = sorted(self.products, key=lambda prod: (bool(prod.fits()), -len(prod.seen)))
        order = [prod for prod in order if prod.position not in self.settled_slots]
        for at, prod in enumerate(order):
            if prod.position in settled_groups(self):
                continue
            self.hunt(prod, max(2, room_left(self) // max(len(order) - at, 1)), final=True)
        return []

    def hunt(self, prod, steps, final=False):
        """Search and read listings for one product, round by round, until enough fits are known or its step
        allowance runs out."""
        stop_at = self.steps_left() - steps

        def floor():
            return max(stop_at, self.closing_steps())
        # several products under the cheapest rule: the first pass only finds fits; the sweep of every cheaper row
        # waits for the final pass, when told needs and the fit to the chosen host decide which rows can count
        search = SearchRun(prod, floor, list(prod.planned_queries), 3 if self.cheapest else (4 if self.measure else 2),
                           self.cheapest and (final or len(self.products) < 2), final, 10 + self.question_hold)
        self.run(Event(SEARCH_ROUND, search=search))

    def on_search_round(self, search):
        prod, floor, final = search.prod, search.floor, search.final
        again = [Event(SEARCH_ROUND, search=search)]
        if not (self.steps_left() > floor() and time_for(self, search.hold) and not self.spent_for_search()):
            return []
        self.read_store_lines()
        self.screen_rows(prod, self.candidates(prod))
        if prod.bound_host_listing is not None:
            codes_prove_fit(self, prod, prod.bound_host_listing)
        rows = self.candidates(prod)
        # rows whose stated codes leave out the chosen host's code are not worth a read
        unread = [cand for cand in rows if not cand.fully_read and cand.ident not in prod.misfit_ids]
        self.judge(prod, [cand for cand in rows if cand.fully_read])
        fit_list = prod.fits()
        if final and fit_list and not self.cart and self.early_wanted:
            self.cart_early()
        # with the cheapest rule every unread row under the best fit's price still has to be seen
        if fit_list and not search.sweep and len(fit_list) >= search.enough:
            if final and self.measure and look_above_fits(self, prod, fit_list, unread, floor()):
                return again
            return []
        groups = self.value_groups(prod)
        if prod.value_signature != repr(groups):
            # values changed (a need was heard): earlier value-query coverage no longer applies
            prod.value_signature = repr(groups)
            prod.value_ids = set()
            prod.values_complete = False
        if groups and self.query_heard_value(prod, floor):
            return again
        cheap_top = sort_price(fit_list[0]) if fit_list and search.sweep else None
        self.order_unread(prod, unread, groups)
        if groups:
            # no unread row (under the best fit's price, with the cheapest rule) carries every required value: the
            # values are queried
            pool = unread if cheap_top is None else [cand for cand in unread
                                                     if cand.price is not None and cand.price < cheap_top]
            if not any(value_groups_hit(self, prod, groups, cand) == len(groups) for cand in pool) \
                    and self.search_values(prod, floor()):
                return again
        outside_first(self, prod, unread)
        all_seen = bool(groups) and self.all_words_needed and prod.values_complete
        if cheap_top is not None:
            unread = [cand for cand in unread if cand.price is not None and cand.price < cheap_top]
            if all_seen:
                # the value query returned every listing with the values: the rest cannot fit
                unread = [cand for cand in unread if cand.screen == "yes" or cand.ident in prod.value_ids]
            if not unread:
                verdict = cheapest_exhausted(self, prod, cheap_top, all_seen, search.queries, floor)
                if verdict == "again":
                    return again
                if verdict == "stop":
                    return []
        if unread and bound_search_due(self, prod) and self.search_bounds(prod, floor(), any_source=True):
            return again
        if unread:
            # many products mean many steps and little time: more pages per judging call
            width = 2
            if self.batch_reads():
                width = self.compare_size
            elif self.compare_thin or len(self.products) >= 3:
                width = 4
            if not self.read_listings(prod, [cand.ident for cand in unread[:width]], keep_steps=floor()):
                return []
            return again
        # every row is read: search further
        if widen_search(self, prod, search.queries, search.cursor, floor) == "stop":
            return []
        return again

    def query_heard_value(self, prod, floor):
        """A newly heard word value leads a query before more rows of the earlier queries are read (real race 14: a
        new query first found the paid listing .123 of the time, a third page .056). A heard bound has no words to
        lead one. True when the query ran."""
        heard = [dm for dm in prod.firm_demands()
                 if dm.origin == "shopper" and dm.op in ("is", "one of", "contains") and dm.value]
        if len(heard) <= prod.heard_queried:
            return False
        prod.heard_queried = len(heard)
        heard_words = word_set(" ".join(dm.value.replace("|", " ") for dm in heard))
        # the first value query carrying the heard words (a heard figure is left out of an all-words store's lead
        # query, which then is not the one to send)
        lead = next((qry for qry in self.pending_value_queries(prod) if word_set(qry) & heard_words), "")
        if lead and self.search_values(prod, floor(), first=lead):
            self.trace_event("heard_value_query", item=prod.name)
            return True
        return False

    def order_unread(self, prod, unread, groups):
        """Unread rows in reading order: rows carrying the required values first; under the cheapest rule in a store
        listing the cheapest rows first, by price (real race 11: leaders who did so passed .65 of the tasks where they
        saw the paid listing, an engine that did not .26); with a measure of best, best on it first."""
        if groups:
            unread.sort(key=lambda cand: (cand.screen == "no", -value_groups_hit(self, prod, groups, cand),
                                          cand.screen != "yes", sort_price(cand)))
        if self.cheapest and self.cheapest_first:
            unread.sort(key=lambda cand: (cand.screen == "no", sort_price(cand)))
        elif self.measure and not self.cheapest and len(unread) > 2:
            order_by_measure(self, prod, unread, groups)

    # ---------------------------------------------------------------- choosing and ordering
    def on_picks_due(self):
        """Choose one confirmed listing per product (hosts before their accessories, condition sources before the
        products depending on them), then order, search once more, or end without a purchase."""
        self.read_store_lines()
        if self.cheapest and not self.backups_read:
            self.read_backups()
        self.judge_all()
        plan_host(self, False)
        chosen = {}
        pairs = self.fit_pairs()
        hosts = {host.position for _, host in pairs}
        host_of = {part.position: host for part, host in pairs}
        sources = {rule["src"].position for rule in self.conditions}
        waiting = {target.position for rule in self.conditions for _, target, _ in rule["branches"]}
        # products whose choice waits on nobody get their first pick confirmed in parallel
        jobs = []
        for prod in self.products:
            if prod.position in host_of or prod.position in waiting:
                continue
            ranked = ranked_fits(self, prod)
            if ranked:
                jobs.append(lambda prod=prod, first=ranked[0]: self.verified(prod, first))
        in_parallel(jobs)

        def turn(prod):
            return (prod.position in host_of or prod.position in waiting,
                    not (prod.position in hosts or prod.position in sources), prod.position)
        for prod in sorted(self.products, key=turn):
            host = host_of.get(prod.position)
            if host is not None and host.position in chosen:
                self.bind_to_host(prod, host, chosen[host.position])
            if prod.position in waiting:
                self.apply_conditions(prod, chosen)
            choose_confirmed(self, prod, chosen)
            if host is not None and prod.position not in chosen:
                # no accessory suits this host: try another host that one suits
                try_other_host(self, prod, host, chosen)
            if prod.position in waiting and prod.position not in chosen:
                # nothing meets what the condition demands for that pick: try a source pick with the other outcome
                flip_condition_source(self, prod, chosen, self.conditions, pairs)
        return self.after_choice(chosen)

    def after_choice(self, chosen):
        unfit = [prod for prod in self.products if prod.position not in chosen]
        if unfit and not self.last_search_done and time_for(self, 20) and not self.spent_for_search():
            # nothing to order for these yet: the reserved ending steps go to one more search
            self.last_search_done = True
            if final_search(self, unfit):
                return [Event(PICKS_DUE)]
        # (re-verifying a listing the audit names costs no step; its reads and searches keep the ending's steps)
        if unfit and not self.audited and time_for(self, 25) and not self.spent_for_search():
            self.audited = True
            if self.audit_search(unfit) and all(self.buyable(prod) for prod in unfit):
                return [Event(PICKS_DUE)]
        self.nearest_picks = dict(chosen)
        if unfit:
            labels = [prod.name for prod in unfit]
            self.decline_reason = "no fitting listing for " + ", ".join(labels)
            self.trace_event("no_fit", items=labels, pools={prod.name: len(prod.seen) for prod in unfit})
            return [Event(NOTHING_FITS, reason=self.decline_reason)]
        if self.budget is not None:
            total = fit_to_budget(self, chosen)
            if total > self.budget + 1e-6:
                self.decline_reason = "the cheapest fitting combination is over the budget"
                self.trace_event("over_budget", total=total, cap=self.budget)
                return [Event(NOTHING_FITS, reason=self.decline_reason)]
        for prod in self.products:
            prod.choice = chosen[prod.position]
        if not self.pilot_look_due():
            return [Event(CART_READY, attempt=0)]
        # unplaced store lines or never-used store tools: the pilot looks once before the order (new world rules
        # surface there first)
        self.pilot_looked = True
        note = "Every product has a pick; before the order, check the store lines and tools the routine could not place."
        if self.pilot(note, {"picks": {prod.name: chosen[prod.position].ident for prod in self.products}}, turns=3):
            return [Event(PICKS_DUE)]
        return [Event(CART_READY, attempt=0)]

    def on_cart_ready(self, attempt):
        """Make the cart hold exactly the picks, look at them once carted, check the basket, then check out."""
        # every re-choice opens a new round; rounds are capped as well
        self.order_rounds += 1
        if self.finished or attempt > 2 or self.order_rounds > 8:
            return []
        picked = {prod.choice.ident for prod in self.products if prod.choice}
        for lid in [lid for lid in self.cart if lid not in picked]:
            self.remove_line(lid)
        for prod in self.products:
            pick = prod.choice
            if not pick or pick.ident in self.cart or self.add_line(prod, pick):
                continue
            if self.cart_rejects_args:
                # the tool rejects the call itself: another listing would fail the same way
                self.trace_event("cart_faulty")
                return [Event(NOTHING_FITS, reason="the cart tool refuses its arguments")]
            prod.verdicts.setdefault(pick.ident, {"failed": set(), "unsure": set()})["failed"].add("cart refused")
            prod.choice = None
        if not all(prod.choice is not None for prod in self.products):
            return [Event(PICKS_DUE)]
        self.look_at_cart()
        changed_now = self.read_store_lines()
        for prod in self.products:
            # a pick read after the add is judged again; a contradicted one is swapped before checkout
            pick = prod.choice
            if pick is None or pick.ident in prod.judged_on:
                continue
            self.judge(prod, [pick])
            failed = verdict_of(prod, prod.choice).get("failed")
            if failed:
                self.trace_event("chosen_failed", id=prod.choice.ident, failed=sorted(failed)[:3])
                changed_now = True
        if changed_now or self.picks_invalid():
            self.trace_event("rechoose", cart=self.cart)
            for prod in self.products:
                self.judge(prod, read_pool(self, prod))
            return [Event(REPICK, attempt=attempt)]
        if len(self.products) > 1 and attempt < 2 and self.basket_rejected():
            return [Event(REPICK, attempt=attempt)]
        before = self.steps
        if self.place_order():
            return []
        return [Event(CHECKOUT_REFUSED, attempt=attempt, since=before)]

    def look_at_cart(self):
        """The carted listings are looked at after the add, so changes show up in the next result; listings judged
        from tables only get their own page read (stock, price, seller notes)."""
        if self.tables_suffice():
            for prod in self.products:
                pick = prod.choice
                if pick is None or pick.page_seen:
                    continue
                if self.steps_left() - self.messages_owed() > 2:
                    self.read_listings(prod, [pick.ident], page=True)
        elif self.steps_left() > 2 + self.messages_owed() and self.products and self.products[0].choice:
            self.read_listings(self.products[0], [self.products[0].choice.ident])

    def basket_rejected(self):
        """A basket of several products is checked as a whole once before the checkout: each line's own demands were
        judged one listing at a time, so this looks only at what spans lines. True when lines must be chosen again."""
        bad_lines = basket_problems(self)
        if not bad_lines:
            return False
        twins_out = {}
        for prod, why in bad_lines:
            prod.verdicts.setdefault(prod.choice.ident, {"failed": set(), "unsure": set()})["failed"].add(why)
            # the same listing under another part code carries the same flaw: with time short the next basket goes
            # unread, and the twin was bought instead
            for twin in title_twins(prod, prod.choice):
                prod.verdicts.setdefault(twin.ident, {"failed": set(), "unsure": set()})["failed"].add(why)
                twins_out.setdefault(prod.name, []).append(twin.ident)
        self.trace_event("basket_gate", bad={prod.name: why for prod, why in bad_lines}, twins=twins_out)
        return True

    def on_checkout_refused(self, attempt, since):
        """Read why the order was turned down, let the pilot look when nothing explains it, then choose again."""
        changed_now = self.read_store_lines()
        if not changed_now:
            self.refresh_cart()
            changed_now = self.read_store_lines() or self.picks_invalid()
        if changed_now:
            self.extend_clock("order refused")
        if not (changed_now or self.finished) and self.steps_left() >= len(self.products) + 3:
            self.pilot("The store turned down the checkout.", self.last_refusal, turns=4, refusal=True)
        # a dropped need or a new budget under the cheapest rule: look below the known fits before choosing again
        sweep_after_news(self, since)
        return [Event(REPICK, attempt=attempt)]

    def on_repick(self, attempt):
        self.repick()
        if self.finished:
            return []
        return [Event(CART_READY, attempt=attempt + 1)]

    # ---------------------------------------------------------------- ending without the planned order
    def on_nothing_fits(self, reason):
        """No order of verified fits: a last review, the nearest listing or an unconfirmed fit where the request
        allows, else a decline that carries its reason."""
        decline = self.roles.get("decline")
        if self.review_before_decline(reason):
            self.run(Event(PICKS_DUE))
            if self.finished:
                return []
        if self.last_look(reason):
            self.run(Event(CART_READY, attempt=1))
            if self.finished:
                return []
        if self.nearest_wanted:
            if self.order_nearest():
                return []
            if not self.finished and self.cart and self.steps_left() >= 1:
                # the request wants the closest listing bought when nothing fits, so a close with no purchase cannot
                # meet it: the cart goes to the checkout once more (a refused close may only have carried news)
                self.trace_event("closest_resend", cart=self.cart)
                if self.place_order() or self.finished:
                    return []
        elif not self.decline_stated:
            # no decline rule in the request: with no verified fit after the searches and reviews, declining is the
            # better bet (real races 11-14: such declines paid 3-4 decline tasks per race)
            if self.order_unconfirmed():
                return []
        if self.finished or not decline:
            return []
        if self.steps_demanded and not self.decline_gated:
            self.decline_gated = True
            self.run_process_gate("before_decline", "end the task without buying anything")
        payload = self.decline_args(decline, reason)
        result = self.call_tool(decline["name"], payload)
        self.trace_event("declined", reason=reason or self.decline_reason)
        if self.finished or self.decline_refused_once or self.steps_left() < 1:
            return []
        return [Event(DECLINE_REFUSED, result=result, payload=payload, tool=decline["name"])]

    def decline_args(self, decline, reason):
        payload = self.closing_args("decline")
        schema = self.role_schema("decline").get("properties") or {}
        slot = decline.get("remark")
        if not slot or slot not in schema:
            # unnamed free-text argument: the single open string field left after the fixed ones
            free = [key for key, spec_d in schema.items() if isinstance(spec_d, dict) and spec_d.get("type") == "string"
                    and not spec_d.get("enum") and key not in payload]
            slot = free[0] if len(free) == 1 else None
        if slot and slot in schema:
            # the decline carries its reason when it can (what was sought, the budget, why nothing fit; no ids, since
            # a note citing one is graded as a missed step)
            payload[slot] = self.clip_arg("decline", slot, self.decline_remark(reason or self.decline_reason))
        return payload

    def on_decline_refused(self, result, payload, tool):
        """The store refused the exit: a change announced at the close (budget, price) may make something fit."""
        self.decline_refused_once = True
        self.trace_event("decline_refused", obs=compact(result["obs"], 200), lines=result["lines"][:3])
        self.last_refusal = refusal_record(tool, result)
        count = len(self.products)
        # lines the closing call carried count as a change even if already read
        news = self.read_store_lines() or result["lines"] or self.changed_at >= self.steps - 1
        if news:
            self.extend_clock("decline refused")
        if news and self.steps_left() >= count + 2:
            # listings ruled out before (old budget, a dropped need) may fit now: look once more
            self.review_pending = True
            look_again = True
        else:
            look_again = self.steps_left() >= count + 3 and self.pilot(
                "The store refused to close the task with no purchase.", self.last_refusal, turns=4, refusal=True)
        if look_again:
            self.last_search_done = False
            self.audited = False
            self.run(Event(PICKS_DUE))
        if self.finished:
            return []
        # a second decline after a refused one was paid 0 times of 270 (races 8-11): the nearest read listing per
        # product is ordered instead when every product has one
        if not self.nearest_tried and self.products and self.steps_left() >= count + 1 \
                and all(ranked_fits(self, prod) or self.nearest_miss(prod) is not None for prod in self.products):
            self.trace_event("nearest_after_refused_decline")
            if self.order_nearest() or self.finished:
                return []
        # some stores refuse to close with items in the cart: it is emptied first
        for lid in list(self.cart):
            if self.steps_left() > 1:
                self.remove_line(lid)
        self.call_tool(tool, payload)
        return []

    def on_forced_close(self):
        """Out of time, steps or ideas: finish with what is known (a buyable fitting cart inside the budget)."""
        if self.finished:
            return []
        for prod in self.products:
            if prod.choice is not None and prod.choice.available is not False:
                continue
            fit_now = self.buyable(prod)
            prod.choice = fit_now[0] if fit_now else None
        ready = bool(self.products) and all(prod.choice for prod in self.products)
        if ready and not self.nearest_wanted:
            # out of steps with picks the checker never passed: an unverified order seldom pays (a clock too short for
            # a check lets them through, since the checker then gives no answer)
            for prod in self.products:
                if not self.verified(prod, prod.choice):
                    self.trace_event("close_unverified", item=prod.name, id=prod.choice.ident)
                    ready = False
                    break
        if ready and self.budget is not None:
            ready = sum((prod.choice.price or 0) * prod.quantity for prod in self.products) <= self.budget + 1e-6
        if ready:
            picked = {prod.choice.ident for prod in self.products}
            obsolete = [lid for lid in self.cart if lid not in picked]
            to_add = [prod for prod in self.products if prod.choice.ident not in self.cart]
            if self.steps_left() >= len(obsolete) + len(to_add) + 1:
                for lid in obsolete:
                    self.remove_line(lid)
                for prod in to_add:
                    self.add_line(prod, prod.choice)
                if self.place_order():
                    return []
        if not self.finished:
            return [Event(NOTHING_FITS, reason="forced")]
        return []

    # ---------------------------------------------------------------- the pilot
    def pilot(self, point, detail=None, turns=PILOT_TURNS, refusal=False):
        """Let a model choose a few moves where the routine is stuck, one per turn, carried out by the routine's
        engines inside the steps the ending needs. True when the moves changed what can be bought."""
        if self.finished or self.stopped:
            return False
        self.pilot_state = {"accepts": 0, "pushed": False, "refusal": refusal, "news": False}
        self.lines_shown = len(self.unplaced_lines)
        session = {"point": point, "detail": detail, "turns": turns, "history": [], "gained": False}
        self.run(Event(PILOT_TURN, session=session, turn=0))
        if session["gained"]:
            self.lines_shown = len(self.unplaced_lines)
        return session["gained"]

    def on_pilot_turn(self, session, turn):
        if turn >= session["turns"] or self.finished or low_on_time(self, 30) or self.models.spent >= PILOT_SPEND:
            return []
        answer = quick_json(self, PILOT_RULES, pilot_prompt(self, session["point"], session["detail"],
                                                            session["history"]), 2000)
        move = answer.get("move") if isinstance(answer, dict) else None
        if not isinstance(move, dict):
            return []
        verb = str(move.get("do") or "").lower()
        if verb and verb != "finish":
            return [Event(PILOT_MOVE, session=session, turn=turn, move=move, verb=verb,
                          think=str(answer.get("think") or "")[:200])]
        held = self.refuse_finish()
        self.trace_event("pilot", point=session["point"][:40], turn=turn, move="finish", held=held)
        if not held:
            return []
        session["history"].append(f"{turn + 1}. finish -> not yet: {held}")
        return [Event(PILOT_TURN, session=session, turn=turn + 1)]

    def on_pilot_move(self, session, turn, move, verb, think):
        report, gained = self.carry_out(verb, move)
        if gained:
            session["gained"] = True
        self.trace_event("pilot", point=session["point"][:40], turn=turn, move=compact(move, 200), report=report[:300],
                         think=think)
        session["history"].append(f"{turn + 1}. {compact(move, 240)} -> {report[:700]}")
        return [Event(PILOT_TURN, session=session, turn=turn + 1)]

    def carry_out(self, verb, move):
        """One pilot move: (what it brought, whether the buyable set changed)."""
        prod = self.product_named(move.get("item"))
        if not prod:
            prod = self.products[0] if len(self.products) == 1 else None
        reserve = self.pilot_reserve()
        if verb in ("search", "read", "ask", "tool") and self.steps_left() - reserve < 1:
            return unchanged("refused: the remaining steps are held for the ending")
        if verb in ("search", "read", "ask", "accept") and prod is None:
            return unchanged("refused: give the exact name of one item")
        if verb in self.MOVES:
            return self.MOVES[verb](self, prod, move)
        if verb == "tool":
            return self.move_tool(move)
        if verb == "news":
            return self.pilot_news(move)
        return unchanged("refused: that move does not exist")

    def judge_read(self, prod, ids):
        read = self.read_listings(prod, ids, keep_steps=self.pilot_reserve()) if ids else []
        self.judge(prod, [cand for cand in read if cand.fully_read])
        return read

    def move_search(self, prod, move):
        query = squeeze(str(move.get("query") or ""))[:60]
        if not query or plain_text(query) in {plain_text(qry) for qry in queries_sent(prod)}:
            return unchanged("refused: the query is empty or was already run")
        if not self.roles.get("search_text"):
            return unchanged("refused: this store has no keyword search")
        known = set(prod.seen)
        fits_before = {cand.ident for cand in self.buyable(prod)}
        self.run_search(prod, query=query, pages=1, budget=self.pilot_reserve())
        new_rows = [cand for cand in self.candidates(prod) if not cand.fully_read and cand.ident not in known]
        self.screen_rows(prod, new_rows)
        hopeful = sorted((cand for cand in new_rows if cand.screen != "no"),
                         key=lambda cand: (cand.screen != "yes", sort_price(cand)))
        width = self.compare_size if self.batch_reads() else 2
        read = self.judge_read(prod, [cand.ident for cand in hopeful[:width]])
        return self.move_summary(prod, len(set(prod.seen) - known), read, fits_before)

    def move_read(self, prod, move):
        ids = [str(raw_id) for raw_id in move.get("ids") or [] if raw_id][:4]
        unread_ids = [lid for lid in ids if not getattr(prod.seen.get(lid), "fully_read", False)]
        if not unread_ids:
            return unchanged("refused: none of the named listings is unread")
        fits_before = {cand.ident for cand in self.buyable(prod)}
        return self.move_summary(prod, 0, self.judge_read(prod, unread_ids), fits_before)

    def move_ask(self, prod, move):
        prop = squeeze(str(move.get("property") or "").replace("_", " ")).lower()[:40]
        if not prop or not self.can_ask():
            return unchanged("refused: no question is left, or no property was named")
        if any(same_name(prop, asked) for asked in prod.asked) or any(same_name(prop, dm.prop) for dm in prod.demands):
            return unchanged("refused: that property was asked or is known already")
        known_n = len(prod.demands)
        reply = self.ask_shopper(prod, prop)
        self.apply_reply_guarded(prod, prop, reply)
        self.read_store_lines()
        new_lines = [dm.as_line() for dm in prod.demands[known_n:]]
        if new_lines:
            self.judge(prod, [cand for cand in prod.seen.values() if cand.fully_read])
        return f"the reply reads as {compact(reply, 200)}; the requirements are now {prod.demand_lines()}", bool(new_lines)

    def move_tool(self, move):
        name = str(move.get("name") or "")
        args = move.get("args") if isinstance(move.get("args"), dict) else {}
        roles = set(self.roles_for(name))
        if name not in self.schemas:
            return unchanged("refused: the store has no tool by that name")
        if roles & {"checkout", "decline", "cart_add", "cart_set"}:
            return unchanged("refused: the routine fills the cart and ends the task itself")
        if "message" in roles:
            if not self.can_ask() and not (self.pilot_state["refusal"] and self.can_message()):
                return unchanged("refused: no message is left to send")
            self.messages_sent += 1
        elif roles & {"search_text", "search_fields", "open", "facts", "compare"}:
            return unchanged("refused: use the search or read move for this tool")
        result = self.call_tool(name, args)
        news = self.read_store_lines()
        lines = f" lines {result['lines'][:3]}" if result["lines"] else ""
        return f"result {compact(result['obs'], 1500)}{lines}", news

    def move_accept(self, prod, move):
        cand = prod.seen.get(str(move.get("id") or ""))
        if cand is None or not cand.fully_read:
            return unchanged("refused: open that listing before this move")
        if cand.ident in {fit.ident for fit in self.buyable(prod)}:
            return unchanged("refused: it already counts as a fit")
        if self.pilot_state["accepts"] >= 2:
            return unchanged("refused: two claims were already put to the checker")
        if (cand.available is False or prod.over_unit_cap(cand) or not self.is_kind(prod, cand)
            or self.field_failures(prod, cand) or over_cap(self.budget, cand, prod.quantity)):
            return unchanged("refused: it is not buyable within the allowances, or a store field fails it")
        self.pilot_state["accepts"] += 1
        if not self.verify(prod, cand, strict=True):
            return unchanged("the checker disagrees, so it stays out")
        prod.verdicts[cand.ident] = clean_verdict()
        prod.judged_on[cand.ident] = set(prod.demand_lines())
        self.trace_event("pilot_accept", item=prod.name, id=cand.ident)
        return changed("the checker agrees: it counts as a fit now")

    MOVES = {"search": move_search, "read": move_read, "ask": move_ask, "accept": move_accept}
    HANDLERS = {
        REQUEST_READ: on_request_read, REFERENCES_DUE: on_references_due, OPENING_STEP: on_opening_step,
        FIRST_PASS: on_first_pass, STATED_VALUES: on_stated_values, QUESTIONS_DUE: on_questions_due,
        QUESTION_DUE: on_question_due, EXTRA_QUESTION: on_extra_question, ASK: on_ask,
        SHOPPER_REPLIED: on_shopper_replied, FINAL_PASS: on_final_pass, SEARCH_ROUND: on_search_round,
        PICKS_DUE: on_picks_due, CART_READY: on_cart_ready, CHECKOUT_REFUSED: on_checkout_refused, REPICK: on_repick,
        NOTHING_FITS: on_nothing_fits, DECLINE_REFUSED: on_decline_refused, FORCED_CLOSE: on_forced_close,
        PILOT_TURN: on_pilot_turn, PILOT_MOVE: on_pilot_move,
    }


class Episode(Controller, Pilot, Cart, StoreLines, ReferenceTracing, ShopperTalk, Judging, ListingReading, Searching,
              RequestReading, StoreSession):
    """One shopping task, from the request to its last store call."""

    def __init__(self, problem_data):
        self.clock_start = time.monotonic()
        env = problem_data["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        look = self.policy.get
        self.request_text = str(look("query") or "")
        self.opening = look("opening")
        self.tool_entries = list(filter(lambda term_s: isinstance(term_s, dict), look("tools") or []))
        self.schemas = {}
        for tool_entry in self.tool_entries:
            func = function_of(tool_entry)
            if isinstance(func, dict) and func.get("name"):
                self.schemas[func["name"]] = func.get("parameters") or {"type": "object", "properties": {}}
        self.step_cap = int(look("max_steps") or 30)
        self.finished = bool(look("finished"))
        self.models = ModelLine(self)
        self.store_key = "|".join(sorted(self.schemas))
        notes = STORE_MEMORY.get(self.store_key) or {}
        self.record_keys = dict(notes.get("maps") or {})
        # compare batches start at 4: stores refuse more without naming a limit, and each refusal costs a step
        self.compare_size = int(notes.get("compare_limit") or 4)
        # compare rows lacked the property table that a separate facts tool provides
        self.compare_thin = bool(notes.get("thin_compare"))
        # does the store need every query word (None = not known yet)
        self.all_words_needed = notes.get("all_words")
        self.steps = 0
        self.stopped = False
        self.trace = []
        self.roles = {}
        self.products = []
        self.budget = None
        self.measure = None
        self.cheapest = False
        self.needs_approval = False   # the procedure wants a shopper message before ordering
        self.steps_demanded = []
        self.call_log = []           # calls so far, as the process gate reads them
        self.gate_rounds = {}
        self.approved_cart = None
        self.messages_sent = 0
        self.message_cap = None
        self.buyable_only = True
        self.topics = []
        self.fit_rules = []
        self.conditions = []
        self.lines_seen = set()
        self.line_queue = []
        self.line_owner = {}
        self.cart = {}
        self.page_size = 0
        self.decline_reason = ""
        self.match_counts = {}       # sent query -> reported match count
        self.store_lines = []          # store lines in arrival order
        self.unplaced_lines = []          # lines where the line reader saw no change
        self.last_refusal = None
        self.review_pending = True      # the pilot may review the record before a decline
        self.tools_used = set()
        self.decline_stated = False
        self.question_hold = 0.0
        self.sweep_floor = None
        self.trace_floor = 0
        self.fixed_reserve = None
        self.grace_used = False
        self.compare_off = False
        self.compare_checked = False
        self.rows_complete = False
        self.cheapest_first = False
        self.price_page_tally = [0, 0]
        self.filled_args = {}
        self.repairs = {}
        self.remapped_keys = set()
        self.closing_choices = {}
        self.verify_cache = {}
        self.condition_cache = {}
        self.reference_searches = {}
        self.parts_searched = set()
        self.baskets_checked = set()
        self.fit_codes_searched = set()
        self.proven_fits = set()
        self.rechecked_ids = set()
        self.checker_failed = set()      # (product, listing) the checker failed while the reader had not
        self.last_look_done = False
        self.plan_requested = False
        self.question_order = None
        self.single_questions = False
        self.lists_asked = 0
        self.measure_ranks = {}
        self.host_plan = None
        self.settled_slots = set()
        self.early_carted = False
        self.early_wanted = False
        self.backups_read = False
        self.last_search_done = False
        self.audited = False
        self.pilot_looked = False
        self.lines_shown = 0
        self.nearest_tried = False
        self.nearest_rounds = 0
        self.nearest_picks = None
        self.unconfirmed_tried = False
        self.decline_gated = False
        self.decline_refused_once = False
        self.order_rounds = 0
        self.cart_rejects_args = False
        self.changed_at = -1
        self.widened_step = -1
        self.sweeps_done = 0
        self.question_count = 0
        self.spare_question = 0
        self.first_share = 0
        self.question_queue = []


# -------------------- model instructions (the JSON keys each one asks for are read by code)

INTAKE_RULES = """Job: turn one shopper's message plus a simulated store's tool catalogue (OpenAI function schemas) into a task sheet for a buying assistant. Output exactly one JSON object following the outline at the bottom; no prose around it.

How to fill it in
1. Only hard demands become requirements. Soft wishes ("ideally", "would be nice") and background (who will use it, where it will sit) do not; note the properties they point to under "hints". Words that only name the product type or its category stay in "name"/"category". Still, when a named property gets a value ("in the <value> style", "with <value> as its <property>", "a <property> of <value>"), that is a requirement on the property, even if the value sounds like a product type; the category remains what the shopper asked for.
2. "3 x gadgets where the colour is red and the size is not large" means a single item, quantity 3, carrying two requirements (colour is red; size is not large).
3. A count of units inside one listing (box, pack or set of N) is a requirement on pack/set size; "quantity" counts listings to purchase.
4. Wording that merely explains one demand ("hook fastening only, so nothing with snaps or laces") is still a single requirement (fastening: hook); add no properties for the things it excludes.
5. If the shopper invites questions or wants to be asked first, that is not a procedure step (the assistant asks anyway); list any subject it mentions in "ask_about".
6. Ops: alternatives "A or B" -> "one of" with "A | B"; "not X" or "anything but X" -> "is not"; "at least N units" -> "at least" with "N units". Units and spelling stay as the shopper wrote them.
7. Some demands lean on a DIFFERENT listing's page, picked out by category, seller and a few attribute values. Use "same as reference" for the value that page states, "different from reference" for anything except it, "counterpart of reference" for the value that goes with it rather than the value itself (what the page or the shop pairs with it, what suits, works with or is compatible with it), "at least reference" / "at most reference" for no lower / no higher than it, and "within reference" for a margin either side of its figure (margin in "tolerance"). Put category, seller, identifying attributes and the property to read into "reference". Describe the page the shopper names even if it may point onward; the assistant chases pointers on its own. If several of that seller's listings are said to qualify with a rule for choosing (the bigger, the cheaper, the one with more of something), store the rule in "pick" and the number in "count", keeping "attributes" to plain identifying values without the comparison. Identifying values belong to that other listing only: never make them requirements or search terms of the wanted item. Invented illustration: "for the colour, avoid whatever Fenwick's garden hose page (length 15 m, finish matte) states" yields exactly one requirement {"property": "colour", "op": "different from reference", "value": "", "reference": {"category": "garden hoses", "seller": "Fenwick", "attributes": {"length": "15 m", "finish": "matte"}, "property": "colour"}} and none about length or finish.
8. A demand on item Y that depends on how item X comes out ("should X's P turn out to be V, I need Y's Q to be W"; "Y's Q must be W, except when X's P is V") is a condition: "then" covers X's P being one of the values, "otherwise" covers the rest. It never goes into "requirements" or "procedure".
9. "category" is the product type; when the shopper names a store category ("the category is garden tools"), copy that wording verbatim.
10. Money uses one unit throughout, the one the shopper counts in. Cents mean cents: "total_cap" is then a number of cents and dollar or spelled-out amounts ("twelve dollars forty") get converted (1240).
11. Tool and argument names are copied from the schemas character for character. When a single final tool closes the task both ways via an argument that chooses between buying the cart and leaving empty-handed, "checkout" and "decline" both point to it, each with "args" carrying its selecting value spelled as in the schema. A final tool that ends without buying is "decline"; the one that pays for the cart is "checkout". When one search tool switches between keyword and field search via an enum argument, "search_text" and "search_fields" both point to it with their own mode values.

Outline (the // notes are guidance only; never write them out)
{
 "items": [  // each distinct product to buy, usually just one
   {"name": "short noun phrase for it, using the request's words (for a list, that entry's words)",
    "category": "product type or store category requested, in the request's wording",
    "quantity": 1,
    "requirements": [ // hard demands on properties of THIS product; never price, stock, quantity or budget
      {"property": "property name the way a listing would label it", "op": "is | is not | at least | at most | one of | contains | same as reference | different from reference | counterpart of reference | at least reference | at most reference | within reference",
       "value": "value, bound or list as written (for 'one of', join the values with ' | ')",
       "reference": null  // only with the ops ending in 'reference': {"category": "...", "seller": "...", "attributes": {"property": "value", ...}, "property": "property to read on that page", "count": null or how many listings the description says qualify, "pick": null or the rule choosing among several (e.g. "whichever is wider"), "relation": "the shopper's own words linking the wanted value to that page's value", "tolerance": null or, for 'within reference', the margin with its unit}
      }],
    "price_cap": null,  // ceiling on ONE unit's own price (not the total), bare number in the request's money unit; else null
    "hints": ["properties the story suggests the shopper secretly cares about (a remark about the user, or about what goes inside, points at the property that records exactly that); empty if none"],
    "search_terms": ["2-4 store-search queries, sharpest first. Listings must contain every query word, so keep them short: the product noun plus one or two required value words (figures keep units); no bare property names, no values that only describe a referenced listing; the final one is the bare product noun"]
   }],
 "total_cap": 12345 or null,        // overall price ceiling, bare number in the request's counting unit (cents if it says cents)
 "cheapest": false,                 // true only when the shopper wants the cheapest / lowest-priced / cheapest-total option among fitting ones (ordering near misses for a nothing-fits case does not count)
 "best": null,                      // name of the measure when the shopper wants the best by some measure, else null
 "message_budget": null,            // integer number of messages the shopper allows, or null
 "call_budget": null,               // integer number of tool calls / steps allowed, or null
 "in_stock_only": true,             // the shopper only wants listings currently purchasable
 "ask_about": ["topics the shopper wants raised with them, in their words; empty if none"],
 "fit_rules": [{"accessory": "item name", "host": "item name"}],   // 'make sure X fits Y' rules; empty if none
 "conditions": [{"if": {"item": "item name", "property": "...", "values": ["..."]}, "then": {"item": "item name", "property": "...", "op": "is | is not | at least | at most | one of", "value": "..."} or null, "otherwise": the same shape or null}],   // demands on one product that hinge on another product's outcome; empty if none
 "decline_rule": null,              // the shopper's instruction for when nothing fits (e.g. just tell me, no substitutes), or null
 "nothing_fits": null,              // "nearest" when, if every listing misses somewhere, the shopper wants the closest one bought, offered or recommended; "decline" when they want the task ended without buying then; null when they say neither
 "procedure": [                     // demands on HOW the assistant goes about the job (not requirements, budgets, message or step limits, and never the nothing-fits rule such as reporting it or proposing a near miss, which belongs in decline_rule / nothing_fits): approvals or confirmations to get, things to show or tell, tools to use, an ordering of steps; empty if none
   {"moment": "start | before_order | before_decline | before_finish", "kind": "message | tool", "tool": "tool name or null", "goal": "what must happen, in the shopper's terms", "messages": 1}  // messages: shopper messages the step uses (0 for a tool); an in-store check (stock, availability, a value) is kind "tool" with 0 messages unless the shopper wants to hear the result
 ],
 "tools": {
   "search_text": {"name": "tool", "query": "argument taking the keywords", "page": "argument for later results (page number, row offset, or a token from the previous result), or null",
                   "page_kind": "number | offset | token or null", "mode": {"arg": "argument switching to keyword search", "value": "its value"} or null},
   "search_fields": {"name": "tool searching by category/brand/price fields, or the same tool in field mode", "category": "arg or null", "brand": "arg or null", "price_max": "arg or null", "page": "arg or null", "page_kind": "number | offset | token or null", "mode": {"arg": "...", "value": "..."} or null} or null,
   "open": {"name": "tool returning one complete listing", "id": "its id argument"},
   "facts": {"name": "separate tool returning a listing's property table, if any", "id": "arg"} or null,
   "compare": {"name": "tool returning several complete listings", "ids": "its list argument"} or null,
   "cart_add": {"name": "tool adding a listing to the cart", "id": "arg", "qty": "quantity arg or null",
                "mode": {"arg": "argument choosing add or remove", "add": "value that adds", "remove": "value that removes"} or null},
   "cart_set": {"name": "tool replacing the whole cart with a list of lines, if any", "lines": "its list argument", "id": "key of the listing id inside a line", "qty": "key of the quantity inside a line"} or null,
   "cart_view": {"name": "tool displaying the cart"} or null,
   "checkout": {"name": "final tool that completes the purchase", "args": {"argument": "value"} or null},
   "decline": {"name": "final tool that ends with no purchase", "remark": "a text argument or null", "args": {"argument": "value"} or null},
   "message": {"name": "tool that messages the shopper", "text": "its text argument"},
   "status": {"name": "tool reporting the remaining allowances"} or null
 }
}"""

COUNT_RULES = """Situation: a buying assistant believes the shopper wants several units of some product. Verify each such count against the shopper's message. Quantity means how many units of one store listing go into the cart, i.e. how many of that product this order buys.
Decide like this:
- A number attached to the product where it is being requested ("N x product", "N products", "N packs of product") is the quantity, no matter what the story adds.
- Failing that, the message's own wording about how many to buy decides. Story numbers (how many people will use it, each needing one, items owned or tried earlier, how often something happens) are not quantities when the request names no such count.
- A word for the bundle a product is normally sold as asks for one unit if a single listing already is that bundle; several bundles mean that many units.
- Units packed inside one box, set or pack describe the listing; order one unit per pack wanted.
- Nothing about a count means 1.
Output one JSON object, no other text: {"items": [{"name": "the item's name as given", "words": "the request's exact words asking for this many, or null", "quantity": 1}]}, one entry per given item, same order."""

SCREEN_RULES = """Pre-sort search hits for a buying assistant. Every numbered row carries an id, title, a bit of text, a category and a price. Using that row only, guess whether the full listing could pass the requirements:
- answer "no" when the row's own wording plainly conflicts with a required value;
- answer "yes" when the row already shows the required values;
- otherwise (the row is silent) it is "maybe".
Product type is settled by the category label, so a row filed under the wanted category is the right type whatever its title implies; never mark "no" for the type.
Output a single JSON object, no other text: {"yes": [row numbers], "no": [row numbers]}. Unlisted rows are "maybe"."""

JUDGE_RULES = """Role: verdict writer. For each store listing, decide per numbered shopper requirement (R1, R2, ...) whether its own records satisfy it.

Satisfaction
1. Look up the listing property that corresponds to the requirement. Spelling of keys varies (spaces, underscores, prefixes, abbreviations), and several keys may hold one property; whichever key names it counts. Match by meaning rather than by string: a single comma-separated entry is enough; plural = singular; near synonyms and everyday abbreviations pass; capitals do not matter; a store value made from the required words with extra qualifiers passes; words repeating the property's own name inside the required value are ignored (that field holding the remaining words states it); convert units before comparing (numbers numerically, edges of a bound included, a range passes a bound only when it lies wholly inside). When one figure is required and the property lists a supported span or several figures containing it, it passes.
2. Property table first. Once a property covers the requirement, its value decides, and another value fails even against a title claiming otherwise. Without any such key, seller notes, an options/variants field, the title or the description may supply the value, but only by saying it outright (own words, standard abbreviation, or the figure). Mere implication by a product name or type (a garment type for a length, "blend" for a particular mix) says nothing. Unstated -> "unsure". A related but different property's value does not count (a component's property is not the product's).
3. Disagreeing keys for one property: pass if any of them gives the required value. Alternatives listed in a property, with the options field naming which variant this is: only that variant counts; a value found only among the other alternatives misses.
4. "is not" / "different from": fail once the excluded value is stated. "one of": any listed value. "contains": value among the stated ones. Feature must be absent (none, no, without): passes when no record mentions it.

Special situations
- Category label = product type. Labelled with the wanted category means right type, whatever the title. Only listings with no label can fail as accessory, spare part or case.
- Compatibility with products named by model or code: passes only if this listing's own text lists one of them among things it fits or works with. Another product's page listing a series or family of compatible part codes makes a listing pass whose own part code (often in brackets in the title) starts with that series. A sentence saying some OTHER part fits a product speaks for the listing whose title carries that part's code, not for the page holding the sentence; "related_pages" shows such pages, and their statements about this listing's own code count for it. A plain link to another page proves nothing.
- Seller notes sending a property to a DIFFERENT listing (use/follow that one's value): ignore this listing's own value for it; a requirement on it is "unsure" unless a related page shown here states it. Sending away a property no requirement mentions changes nothing.
- STORE FIELDS (when supplied): for each required property, the keys and values used by this store's read listings of the category. If some carry the exact required value in that field, a listing whose field shows something else fails, even a broader or related value (required words plus qualifiers are not something else); a value carried by no read listing may still pass through an obvious synonym.
- Only the listed requirements; invent none.

Output: one JSON object, no commentary: {"verdicts": [{"id": "...", "failed": ["R numbers the records plainly fail"], "unsure": ["R numbers no record speaks to"], "weak": ["R numbers passed only via title, description or an options field, not via a property or seller note"], "loose": ["R numbers passed via a synonym or wider term instead of the value's own words; unit conversion or a series-covered code is not loose"]}, ...]}. One entry per listing; omit empty lists. Nothing under "failed" or "unsure" = all requirements met."""

QUESTION_RULES = """Pick the shopper questions for a buying assistant. Hidden demands: each is one property's value for one wanted product, disclosed only if a message names that exact property (or an obvious synonym); asking about something irrelevant to them yields "no preference".
Given: the request; per wanted item its name, known demands, story hints, and a property menu built from its candidate listings (carrier counts, sample values).
Return exactly QUESTION_COUNT picks, strongest first: {"questions": [{"item": "item name exactly as given", "property": "property name spelled as in the menu (words separated by spaces)"}, ...]}.
Order of preference:
1. Hinted properties: comments on the user, on mounting or fitting, on contents or on appearance map to the menu property that records exactly that; match each hint to its nearest menu name.
2. Then properties people buying such a product tend to have opinions about; prefer high "listings" counts and values that differ across listings. Bad picks: one value everywhere, option/variant selectors, identifiers, model codes, warranty or shipping notes.
3. Never: properties fixed by a known demand, price, stock, quantity, repeats (two spellings = one property). Share picks across items according to how undecided each is; strongest hints first."""

REWORD_RULES = """Setting: one question per message, one property each. Hidden demands sit on single properties and appear only when that exact property is asked about. Just now the shopper answered "no preference" for the NO_VIEW_ON property or properties; the plan (written before any answer) asks about PLANNED next.
Given: product, known demands, properties already asked, NO_VIEW_ON and PLANNED with sample values, plus a menu of properties its listings carry (carrier counts, samples).
Return JSON: {"why": "one short sentence", "restates": true or false, "property": "the property to ask next, spelled as in the menu"}.
- restates = true when PLANNED is the same aspect as some NO_VIEW_ON property under another label, or when its values could serve equally as that property's values (both general descriptions of overall look, design or theme); indifference to one usually means indifference to the other. A property holding its own separate fact (a different measurement, a single concrete feature) never restates, even within the same broad area such as looks or size.
- restates = false -> property = PLANNED.
- restates = true -> property = the menu entry likeliest to hide a demand: a fact separate from all NO_VIEW_ON properties, carried by the most listings, varying in value. Exclude anything asked already, fixed by a known demand, price, stock, quantity, identifiers, option selectors, warranty and shipping notes."""

STATED_RULES = """Sentence check. The request sentence gives PROPERTY some value; listings of ITEM have a field called PROPERTY. Question: is this value required of the ITEM purchased here, or does the sentence talk about another thing (the shopper's house, plans or belongings, a different product from OTHER ITEMS, money, the product type)? Respond with JSON only: {"demand": true or false, "op": "is | is not | at least | at most | one of", "value": "wording from the sentence (several allowed values: join with ' | ')"}."""

REPLY_RULES = """Task: label how a shopper responded when a buying assistant asked about one or more properties of a product. Respond with JSON only:
{"kind": "need|repeat|free|split|unclear|other",
 "property": "which property the response is about: shopper wording, or the asked property if that is what they mean",
 "op": "is|is not|at least|at most|one of|same as reference|different from reference|counterpart of reference|at least reference|at most reference|within reference",
 "value": "stated value, bound or list (lists joined with ' | ')",
 "reference": null or {"category": "...", "seller": "...", "attributes": {"property": "value"}, "property": "property to look up there", "count": null or qualifying listings by their count, "pick": null or selection rule among several, "relation": "their own phrasing of how the wanted value relates to that page", "tolerance": null or allowed deviation around that page's figure},
 "budget": null or a number when a new spending total is given,
 "answer": "" or a short plain reply to any question they put,
 "steps": [] or new process demands (get approval, show or tell something, use a tool before ordering), each {"moment": "before_order|before_decline|before_finish", "kind": "message|tool", "tool": null, "goal": "required outcome", "messages": 1}}
Picking the kind
- "need" = they insist on some value, limit, exclusion or set (settled, firm, nothing but, only X will do, anything except X, X or Y alone). "anything except / but / apart from / other than X": follow what happens to that "anything". They keep, accept or ask for it -> X is excluded (op "is not"; "different from reference" when X is a page's value). They reject or discard it, or rule out settling for it -> X is wanted (op "is"; "same as reference"). A pointer to a DIFFERENT listing's page (category, seller, attribute values): wanting its value -> "same as reference" with "reference"; anything but that -> "different from reference"; whatever that page matches with its value (counterpart, goes with) -> "counterpart of reference"; not lower / not higher -> "at least reference" / "at most reference".
- "repeat" = no answer, merely an item from KNOWN NEEDS said again (same property and value).
- "free" = indifference: don't mind, open, anything goes, no opinion.
- "split" = asks to break the message up, take topics one at a time, or answer only its first part (it held several topics).
- "unclear" = did not get it, or wants it repeated or simplified.
- "other" = all remaining cases (greeting, counter-question, budget change).
For "property" use what the response itself names, in their words, even when it is not the asked one; if it names nothing, the asked property (of several, the one carrying the value). One "no preference" over several named properties covers all of them; "property" is then the first asked.
Limit words ("at most N", "at least N", "no more than N", "under N") decide op and value even in garbled sentences ("the length I need sits at an at most 12 cm" means op "at most", value "12 cm").
An insisted value with no limit words is op "is" however it is worded: declining to look beyond it, or to accept anything else, asks for exactly it, numbers too (op "is", value "50 cm")."""

LINE_RULES = """Mid-task, a store tool's output included a sentence from the shopper. Describe its consequences in one JSON object:
{"changes": [{"kind": "budget|drop|add|price|stock|step|other",
  "budget": null or new spending ceiling as a number,
  "property": null or property of the requirement removed or introduced,
  "op": "is|is not|at least|at most|one of", "value": null or value of the new requirement,
  "item": null or affected product, "note": "a short sentence"}]}
- "budget": spending limit changed either way; copy the figure exactly, in the sentence's own unit (cents remain cents).
- "drop": a requirement abandoned ("let X go", "drop X", "never mind X").
- "add": a requirement introduced.
- "price": per the shopper, the chosen listing now costs something else.
- "stock": per the shopper, the chosen listing is no longer for sale.
- "step": process demand (approval, show or tell something, a tool before ordering) as "step": {"moment": "before_order|before_decline|before_finish", "kind": "message|tool", "tool": null, "goal": "required outcome", "messages": 1}."""

PAGE_RULES = """Reference tracing: one listing page is open and the value of one property is needed from it. The page may (a) give the value in its table or seller notes, (b) point the reader at a DIFFERENT listing that holds it (category, seller, some attribute values), or (c) state which property values belong together (one pair for this page's value, or the whole store's pair list).
JSON: {"value": "value given by this page itself, verbatim, or null", "paired": "value this page pairs with its own, or null", "wanted": "what the shopper's RELATION calls for: the paired value when it wants something going with, suiting or compatible with the page's value and a pairing is stated; the page's own value when it wants the same value", "redirect": null or {"category": "null unless the notes name one", "seller": "...", "attributes": {"property": "value"}, "property": "..."}}
- Handing the WANTED property to another listing = redirect, no value. Handing some other property away is no redirect here.
- An outright stated value in the notes = that value.
- Else the table under any key naming the property; options field, title or description only when no key names it.
- A sentence just listing the store's pairs does not tell this page's value.
- "paired" only from a stated pairing; in a pair list use the pair holding this page's own value (either direction). Null when nothing is paired or the own value is unknown."""

QUERY_RULES = """Propose search-box queries. Store matching: every query word must occur in title, brand, category or property values (stems ok, tiny words ignored); results come in a fixed order, a few per page, so each query should be narrow enough that most hits meet the requirements. Given: product, requirements, queries tried, some seen listings flagged as meeting them or not. JSON only: {"queries": ["...", ...]}, max 5 fresh queries of 1-3 words, best first: the product noun as store titles write it with a required value in listing spelling, or other store vocabulary for product or value. No query that was already tried."""

ARGS_RULES = """A store tool rejected (or would reject) the arguments of a call. Given: tool name and description, schema, current arguments, its use here, and the refusal text if any. JSON only: {"args": {the complete argument set for the call}}. Keep current arguments unless the refusal faults them; fill gaps with values fitting the use (count 1 unless stated otherwise, first page of results, the option serving the use). Schema-allowed names and values only."""

CHOICE_RULES = """Final tool, closing argument needed. Given: tool description, still-open arguments with permitted values, aim. JSON only: {"<argument name>": "<the allowed value that achieves the aim>", ...}."""

PROCESS_RULES = """Audit, at a given moment of a shopping task, whether the assistant follows the shopper's instructions about process. Inputs: the shopper's message; the process steps it demands; the moment (what the assistant will do next); the current basket; remaining allowances; the store's tools (tools that end the task or touch the cart are omitted, the assistant handles those); and every call made so far with its result.
Find the demanded steps that are due right now, before the assistant's next move, and not yet done for the current state.
- Approvals and confirmations only count for the basket exactly as it now stands; one given for an earlier basket is void.
- A step already done for this state is not due again; a step tied to another moment is not due.
- A shopper message must be self-contained: for a basket, give every line (quantity, title, price) and the total, and state plainly what is being asked. Never ask about product properties in it, and never spend the final allowed message on anything but a demanded step.
- In-store checks (stock, availability, a value) use the store's reading tools rather than a message, unless the shopper wants to be told; since the assistant reads every listing it buys, such a check is only due when the basket's listings are unread.
Output a single JSON object, no other text: {"actions": [{"tool": "exact tool name", "args": {"argument": "value", ...}, "why": "the demanded step it fulfils"}]} in order, at most 2, or {"actions": []} if nothing is due."""

RANK_RULES = """These listings all satisfy the requirements; the shopper wants the top performer on one measure (something to maximise or minimise). Take each listing's figure for it from properties, title or description, normalising units. JSON only: {"order": ["id", ...], "values": {"id": "the value as read"}} with all ids, best first; unreadable ones at the end; on a tie, cheaper first."""

AUDIT_RULES = """Second opinion before a product is given up. Given: product, requirements, budget, queries run, and all listings encountered: READ ones (properties, notes, requirements marked failed or unaddressed) and rows seen only in results. JSON only: {"fits_after_all": ["READ listing ids whose own records in fact meet every requirement (property value decides; a different value fails; a property absent from all records is unmet)"], "unopened": ["max 4 unread row ids likeliest to fit, likeliest first"], "queries": ["max 2 new 1-3 word keyword queries likely to surface fits, none run before"]}. Empty lists allowed."""

POLARITY_RULES = """A shopper answered a question on one product property. Which rule does the answer set for that property regarding VALUE? JSON only, fields filled in this order: {"refused": "plainly, the property values the answer rejects, discards or refuses to see", "accepted": "plainly, the values it calls acceptable", "rule": "only" if VALUE alone is acceptable and all else refused, "except" if VALUE itself is refused while others are fine, "unclear" if the answer does not settle it}."""

CONDITION_RULES = """Condition check on one listing: by its own records, is its value of the named property among the named values (meaning-equal, units converted)? JSON only: {"holds": true or false, "value": "listing's value verbatim, or null"}; no record stating the property -> holds false, value null."""

MATCH_RULES = """Find the single listing matching a description (category, seller, some attribute values) among the candidates with records below. JSON only, no explanation: {"id": "id of the described listing, or null when none qualifies"}.
- Seller must match; the category label must name the described kind; every described attribute must agree with the records (meaning-equal, units converted).
- Notes, title and description are records: a value missing from the table still counts if the listing's own notes state it of this listing (remarks about other versions or listings do not), so stated-in-notes values make a fit.
- Any described value left unstated or contradicted = no fit; answer null, never the nearest candidate (searching continues).
- A PICK rule may leave several fitting; apply it among them and return its choice."""

BASKET_RULES = """Whole-order review. Lines were each checked against their own requirements already, so skip that. Assess just three order-level things: (1) rules from the request connecting products (fit, suit or match one another; a value of one hinging on another), using each line's records; (2) the total versus the budget; (3) that every line clearly is the product it stands for. Respond with JSON only, no explanation: {"ok": true or false, "bad": [{"item": "item name exactly as given", "why": "short reason"}]}. Mark a line bad only on a plain breach in its records; if records leave a rule undecided, treat the order as ok."""

LAST_LOOK_RULES = """Closing review. A buying assistant is about to end a task without buying, because its routine could not settle a listing for the numbered products below. Every listing shown was read in full, and none was marked as plainly failing a requirement; "doubted on" names the requirements its routine was not sure of. For each product, test each listing against every requirement using only that listing's own record (title, properties, text, notes): a different stated value fails it; a requirement the record never touches is unmet. Respect the budget for the sum of all lines, the quantities, and which listing the shopper wants. Under a fit rule, a product meant to work with another one must suit, by the records, the listing already chosen or the one you name for that other product. JSON only: {"picks": {"<product number>": ["ids of listings meeting every requirement, best first"]}, "note": "one short sentence"}. Give an empty list for a product where none qualifies."""
VERIFY_RULES = """Pre-purchase gate. Settle EACH listed requirement from this listing's own records, nothing else.
- Passes when the listing's own property states the value. Valid: any key naming the property (disagreeing keys: one stating the value suffices), one comma-list entry, synonyms, abbreviations, singular/plural, converted units, a supported span or list of figures including the required one, a store value of the required words plus qualifiers. Store wording beats outside notions of what such products really are.
- Alternatives in a property plus an options field naming this listing's variant: the variant decides; a value only among the other alternatives fails.
- No covering key: seller notes, options field, title or description may state it; stated nowhere = fail. Another property's value does not count.
- Compatibility passes if the listing names the other product among what it fits, or that product's page lists a compatible part-code series this listing's own code starts with.
- Notes deferring a property to another listing's value: this listing's own value cannot pass it.
- Exclusions and closed lists bind. Wanted-category label = right product type, unless title, description or properties plainly show another physical form than requested (liquid or bar instead of powder, single component instead of the whole item): then fail and list it under "problems".
- Price x quantity within budget (price units). Records showing it cannot be bought = fail.
Only the listed requirements; add none.
JSON: {"fits": true or false, "problems": ["each requirement plainly unmet or unaddressed by any record"]}."""

CATEGORY_RULES = """Store category labels vs one requested product kind. Return each seen label that IS that kind: its meaning matches the request's wording taken whole, broader labels joining it with a neighbouring kind included. Neighbouring kinds stay out (accessories, parts, cases for it, narrower or different types). JSON only: {"labels": ["label exactly as given", ...]}; empty if none fit."""

KEY_RULES = """Map roles to keys of the JSON record a store tool returned (below): for each role, the exact key, or null if absent. One JSON object:
{"id": "identifier", "title": "name", "text": "descriptive text", "price": "price", "stock": "whether purchasable", "category": "category label", "brand": "brand", "seller": "seller", "facts": "object holding its property table", "notes": "seller's free-text remarks", "total": "cart total", "lines": "cart lines", "next": "marker or token for the following results page", "count": "overall match count"}"""

PILOT_RULES = """Control passes to you from a buying assistant's scripted routine, which is stuck: for some product it holds nothing it may buy, or the store refused its order or exit, or a store line or tool appeared that the script cannot handle. The script has searched, read and judged listings so far; its record is below. Choose a single move per turn; its result comes back and you choose again.
Reply format, ONE JSON object only: {"think": "one or two short sentences", "move": {...}}. Available moves:
- {"do": "search", "item": "product name", "query": "1-3 words"} store search for that product; the most promising new rows are opened and judged. Each keyword has to occur in a listing's own wording; repeating a tried query yields nothing.
- {"do": "read", "item": "product name", "ids": ["id", ...]} opens at most 4 listings and judges them on that product's requirements.
- {"do": "ask", "item": "product name", "property": "a single property in the listings' spelling"} one message to the buyer asking which value of that property they want (allowed while questions_left > 0).
- {"do": "tool", "name": "tool", "args": {...}} a single call of a tool from other_tools (after a refusal also the message tool), used per its description.
- {"do": "accept", "item": "product name", "id": "id", "why": "the records meeting each requirement"} claim that a read listing meets all requirements after all; an independent checker rereads it and only its consent turns it into a fit.
- {"do": "news", "line": "exact copy of the store line", "kind": "budget, drop or gone", "item": "product name or null", "property": "drop only: requirement released by the line", "budget": "budget only: new total, numeric", "id": "gone only: listing the line reports unavailable"} apply a change announced by a store line that the record does not yet reflect.
- {"do": "finish"} return control: the script orders the known fits, or ends without a purchase if some product has none.
Advice:
- Check store lines and fresh results before anything else; a line may lift or lower the budget, cancel a requirement, or give the reason an order or exit was refused, and you act on that.
- Next, look for what the script overlooked: unread rows with a fitting title and price, a near miss whose own records actually satisfy the requirement it failed, a query in the store's vocabulary for the product or required value.
- Only a listing's own records satisfy a requirement; accepting a listing that leaves one unstated or gives a different value is never right.
- Moves of type search, read, ask and tool cost store steps: keep within steps_you_may_use, repeat nothing, and finish once nothing promising is left."""


def agent_main(problem_data):
    return Episode(problem_data).go()


__all__ = ["agent_main"]
