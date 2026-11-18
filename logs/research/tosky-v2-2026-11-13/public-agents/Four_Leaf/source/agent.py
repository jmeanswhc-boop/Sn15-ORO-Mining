"""ORO ShoppingBench agent - core runtime v0 (owner claude-1, builder A).

One file, standard library plus the sandbox's ProxyClient. Entry point: agent_main(problem_data).

Layers, top to bottom:
  1. CONFIG          budgets, deadline and model preferences (all tunable in one place).
  2. Schema check    local validation and repair of tool arguments, so no step is wasted on a
                     call the runtime would reject.
  3. WorldAdapter    binds tool ROLES from tool names, descriptions and parameter schemas, and
                     normalises every observation by VALUE SHAPE. Worlds rename tools and keys,
                     and wrap observations in renamed side-channel envelopes mid-episode.
  4. EnvSession      the /environment/call protocol: step counter, batched reads, replay-safe
                     retries, finished flag, deadline.
  5. ModelChain      model preferences per provider with failure-triggered switching.
  6. REASONING LAYER owner claude-2. Placeholder rules today; the interface is below.
  7. Shop            catalogue, pagination state, cart, cheapest-viable selection.
  8. Drivers         RuleDriver (deterministic, no LLM) and ModelDriver (LLM actor inside a
                     gated harness). Every episode ends with an order or a decline.

REASONING INTERFACE (claude-2 replaces the method bodies of class Reasoning; the signatures and
the Ledger fields are the contract). `llm` is a ModelChain or None; every method must work with
llm=None, because the deterministic path calls them too. Run.viable() calls judge() with
llm=None on every product, so judge must be cheap; spend model calls in check_order().
Use llm.ask_json(system, user, purpose="judge") for a JSON answer (None on failure).

  Ledger                                    fields documented on the class
  extract(query, opening, llm) -> Ledger    read budgets, quantity, preference and needs
  search_terms(ledger, shop, llm) -> list   search phrases, most promising first
  judge(ledger, product, llm) -> Verdict    does one product meet the known needs?
  rank(ledger, candidates, llm) -> list     order viable products (default: cheapest first)
  next_question(ledger, shop, messages_left, llm) -> str | None
                                            one question about one property, or None
  absorb(ledger, lines, source, shop, llm) -> set
                                            take shopper text (source: reply, change, refusal)
                                            and return event tags (see EVENT TAGS)
  check_order(ledger, cart, shop, llm) -> (ok, problems)
                                            last gate before an order is placed
  should_decline(ledger, shop, exhausted, llm) -> bool
  decline_note(ledger) -> str               text for a decline tool's free-text argument

EVENT TAGS returned by absorb: "budget" (budget changed), "need" (a need was added),
"drop" (a need was removed), "price" (the cart item's price changed), "stock" (the cart item
sold out), "quantity" (the wanted quantity changed). The core re-checks the cart on any of these.

Money: Ledger.budget is in CENTS. Every world seen so far prices in cents; Run.budget_world()
converts only when the adapter learns the world uses dollars.

CORE GUARANTEES: no call is sent that fails the local schema check; repeat reads are answered
from cache for free; the message budget is never exceeded on purpose; the first checkout leaves
CONFIG["refusal_reserve"] steps for a refused order (every pending change fires at checkout);
the episode always ends with an order or a decline when any finishing call is possible.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from os import getenv

try:  # the sandbox ships the repo's client; offline tests inject their own
    from src.agent.proxy_client import ProxyClient
except Exception:  # noqa: BLE001 - any import failure means "inject a client"
    ProxyClient = None


# ======================================================================================
# 1. CONFIG
# ======================================================================================

CONFIG = {
    "deadline_s": 265.0,        # our wall-clock budget per episode; the hard limit is 300 s
    "finish_margin_s": 30.0,    # below this much time left, stop exploring and finish
    "reserve_steps": 6,         # kept for the final cart fix, the pre-checkout re-read and the order
    "refusal_reserve": 6,       # kept after the first checkout: every pending change fires there
    "max_queries": 4,           # distinct search phrases in the rule path
    "max_pages": 6,             # pages fetched per phrase (a schema maximum also stops paging)
    "inspect_top_k": 6,         # cheapest viable listings opened before choosing
    "batch_max": 16,            # validator limit of actions in one envelope
    "max_order_attempts": 3,
    "ref_max_hops": 6,          # C0017: listing-to-listing hops a reference may take
    "p_neg": True,
    "p_cmp": True,
    "p_gref": True,
    "p_free": True,
    "q_plain": True,            # QT (claude-3-14): questions without "and" / "&" or a second sentence
    "jv_support": True,         # JV (claude-3-14): a reply value must be shown by some observed listing
    "rq_page": True,              # RQ (claude-3-14): page-form references in the request
    "rq_note": True,              # RQN: note-shaped redirect sentences in the request
    "pair_inline": False,       # V4-P1: "That <v> <attr> goes together with <w>" does not replace a reference's value
    "match_caption_override": False,  # C0017: a caption phrase may override a contradicting fact
    "env_timeout_s": 90,
    "hard_limit_s": 300.0,      # C0054: the sandbox's per-task limit; env calls never wait past it
    "hard_margin_s": 25.0,      # C0054: kept free below the hard limit (process start, the final reply)
    "env_attempts": 6,          # C0054: attempts per envelope (one HTTP try each), jittered back-off between them
    "env_backoff_s": 1.0,       # C0054: first back-off; doubles per attempt, capped below
    "env_backoff_cap_s": 8.0,          # outer attempts per envelope (each replays the same step)
    "llm_timeout_s": 40,        # A-R: per attempt (was 15: under race load a slow reply was dropped and counted failed)
    "llm_retries": 0,
    "llm_bench_after": 2,       # A-R: straight TRANSIENT failures before a model cools down (401-404 bench at once)
    "llm_attempts": 2,          # A-R: attempts per model per call on a transient error (429, 408, 5xx, timeout, network)
    "llm_backoff_s": 1.0,       # A-R: wait before the same model again (doubles; a 429 keeps RL1's Retry-After wait)
    "llm_cooldown_s": 45.0,     # A-R: a cooled model is skipped this long, then probed again
    "llm_time_budget_s": 110.0,  # A-R: model wall time per episode; past it the rules go on alone
    "llm_deadline_reserve_s": 60.0,  # A-R3: no new model call with less than this left before the deadline
    "llm_deadline_margin_s": 45.0,   # A-R3: an attempt's wait ends at least this long before the deadline
    "llm_wall_slack_s": 1.0,    # A-R4: a model attempt's hard wall-clock limit is its timeout plus this
    "llm_min_max_tokens": 2000,  # A-R4: room for a reasoning-by-default model's reasoning plus the answer (F-038)
    "llm_min_time_s": 20.0,     # never start a model call with less model time than this left (A-R4: a reasoning reply)
    "actor": "assist",          # "assist": rules driver + a few model calls (default);
                                # "model": LLM actor in the harness; "rules": no model at all
    "max_calls": 20,            # hard cap on model calls per episode (every mode)
    "max_cost_usd": 0.045,      # hard cap on model cost per episode (R1: ORO allows $2 for 30 qualifying tasks)
    "cost_guard_usd": {"read": 0.02, "judge": 0.012, "actor": 0.004},   # R1: worst case of the next call, by purpose
    "token_price_usd": {"read": [3e-6, 15e-6], "judge": [3e-6, 15e-6], "actor": [1e-6, 5e-6]},  # C1: for unlisted models
    "model_price_usd": {        # C1: list prices per token (input, output), ORO's model list of 2026-10-28
        "anthropic/claude-sonnet-5.5": [2e-6, 10e-6], "openai/gpt-6-sol": [2e-6, 10e-6],
        "anthropic/claude-haiku-5.5": [0.1e-6, 0.5e-6], "openai/gpt-6-luna": [0.1e-6, 0.5e-6],
        "deepseek/deepseek-v4-flash": [0.091e-6, 0.182e-6]},
    "model_read": True,         # R1: one structured model reading of the request; the rules parse is the fallback
    "read_timeout_s": 60,       # R1: that one call may take longer than the others (A-R: was 35)
    "assist_max_judge_calls": 4,  # judge-model questions per episode in assist mode
    "final_max_calls": 2,       # of those, kept for the final check of the picks (the rest compare values)
    "compare_max_calls": 0,     # judge value comparisons per episode (Reasoning.compare_values); rules-first: off
    "model_replies": False,     # a model reading of replies the parsers find no value in (rules-first: off)
    "model_extract_needs": False,  # model needs from the request (off: no upside with real models on p54)
    "assist_max_calls": 10,     # model calls per episode in assist mode (extract 1, questions 1-2,
                                # unclear answers <= 2; the judge questions are kept back for the judge)
    "max_blocked": 6,           # gate refusals before the model actor hands over to the rules
    "discover_models": True,    # one GET /inference/models per process to drop unserved models
    "discover_timeout_s": 4,
    "actor_keep_results": 8,    # older tool results are shown to the model as short digests
    "models": {                 # preference order; all are on ORO's public allowlists (2026-10-26)
        "openrouter": {
            "actor": ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "deepseek/deepseek-v4-flash"],
            "judge": ["anthropic/claude-sonnet-5.5", "openai/gpt-6-sol"],
            "read": ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"],   # R1: one fallback at most
        },
        "chutes": {
            "actor": ["deepseek-ai/DeepSeek-V3.2-TEE", "zai-org/GLM-5.2-TEE",
                      "moonshotai/Kimi-K2.6-TEE", "Qwen/Qwen3.5-397B-A17B-TEE"],
            "read": ["deepseek-ai/DeepSeek-V3.2-TEE", "zai-org/GLM-5.2-TEE"],   # R1: one fallback at most
            "judge": ["deepseek-ai/DeepSeek-V3.2-TEE", "zai-org/GLM-5.2-TEE",
                      "moonshotai/Kimi-K2.6-TEE"],
        },
    },
}


# ======================================================================================
# Text helpers
# ======================================================================================

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_ID_RE = re.compile(r"^[A-Za-z]{1,4}-[0-9A-Za-z]{6,}$")


def words(text) -> list:
    """Lower-case word tokens; splits camelCase, snake_case and punctuation."""
    if not isinstance(text, str):
        return []
    return [w for w in re.split(r"[^a-z0-9]+", re.sub(r"(?<=[a-z])(?=\d+(?:[^a-z0-9]|$))", " ", _CAMEL.sub(" ", text).lower())) if w]
    #     R5-S16: a counter glued to a word ("department1", "bookmark2") is split off, like a case or underscore boundary


def looks_like_id(value) -> bool:
    return isinstance(value, str) and bool(_ID_RE.match(value))


def clip(text, n) -> str:
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    return text if len(text) <= n else text[: n - 1] + "~"


def is_sentence_list(value) -> bool:
    """A non-empty list of strings, at least one of which reads like a sentence."""
    return (isinstance(value, list) and bool(value) and all(isinstance(x, str) for x in value)
            and any(" " in x.strip() for x in value))


_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a": 1, "an": 1,
                 "single": 1, "pair": 2, "couple": 2, "dozen": 12}
_MONEY_RE = re.compile(
    r"(?P<pre>(?:US\s*)?\$|USD\s*)\s*(?P<a>\d[\d,]*(?:\.\d+)?)"
    r"|(?P<b>\d[\d,]*(?:\.\d+)?)\s*(?P<post>(?:US\s+)?(?:cents?|cts?\.?|¢|c\b)|(?:US\s+)?(?:dollars?|bucks|usd))",
    re.IGNORECASE)


def parse_money(text) -> list:
    """All money mentions in `text` as (cents, unit, start) with unit 'cents' or 'dollars'."""
    out = []
    for m in _MONEY_RE.finditer(text or ""):
        if m.group("a") is not None:
            amount, unit = float(m.group("a").replace(",", "")), "dollars"
        else:
            amount = float(m.group("b").replace(",", ""))
            post = m.group("post").lower()
            unit = "dollars" if any(k in post for k in ("dollar", "buck", "usd")) else "cents"
        cents = amount * 100.0 if unit == "dollars" else amount
        out.append((round(cents, 2), unit, m.start()))
    return out


def parse_count(token):
    token = (token or "").lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


# ======================================================================================
# 2. Schema check and repair (our implementation of the documented JSON-schema subset)
# ======================================================================================

def _is_kind(value, kind) -> bool:
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list)
    return isinstance(value, dict)


def _equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def _kinds(schema) -> list:
    kind = schema.get("type")
    if kind is None:
        return []
    return list(kind) if isinstance(kind, list) else [kind]


def schema_problems(value, schema, path="$") -> list:
    """Every way `value` breaks `schema` (empty list when it is valid)."""
    if not isinstance(schema, dict):
        return []
    kinds = _kinds(schema)
    if kinds and not any(_is_kind(value, k) for k in kinds):
        return [f"{path}: expected {'/'.join(kinds)}"]
    problems = []
    if "const" in schema and not _equal(value, schema["const"]):
        problems.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and not any(_equal(value, v) for v in schema["enum"]):
        problems.append(f"{path}: must be one of {schema['enum']}")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                problems.append(f"{path}: missing {name!r}")
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            if name in props:
                problems += schema_problems(item, props[name], f"{path}.{name}")
            elif extra is False:
                problems.append(f"{path}: unknown property {name!r}")
            elif isinstance(extra, dict):
                problems += schema_problems(item, extra, f"{path}.{name}")
    if isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            problems.append(f"{path}: wrong number of items")
        for i, item in enumerate(value):
            if isinstance(schema.get("items"), dict):
                problems += schema_problems(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, str) and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", len(value)):
        problems.append(f"{path}: length out of range")
    if _is_kind(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: above maximum {schema['maximum']}")
    return problems


def _closest_key(name, keys):
    """The schema key a misspelt or re-cased argument name most likely means."""
    target = set(words(name))
    best, best_score = None, 0
    for key in keys:
        if key.lower() == str(name).lower():
            return key
        score = len(target & set(words(key)))
        if score > best_score:
            best, best_score = key, score
    return best


def _closest_enum(value, options):
    text = str(value).lower()
    for option in options:
        if str(option).lower() == text:
            return option
    toks = set(words(text))
    scored = sorted(((len(toks & set(words(str(o)))), i) for i, o in enumerate(options)), reverse=True)
    if scored and scored[0][0] > 0:
        return options[scored[0][1]]
    return options[0] if options else value


def default_for(schema):
    """A safe value for a required property the caller did not give (None when unsure)."""
    if not isinstance(schema, dict):
        return None
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return schema["enum"][0]
    kinds = _kinds(schema)
    if "integer" in kinds or "number" in kinds:
        return schema.get("minimum", 1)
    if "array" in kinds:
        return []
    if "object" in kinds:
        return {}
    if "boolean" in kinds:
        return False
    return None


def repair(value, schema):
    """Coerce `value` towards `schema`: types, enums, bounds, lengths and unknown keys."""
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if "const" in schema:
        return schema["const"]
    if "object" in kinds or (not kinds and isinstance(value, dict)):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                value = {}
        if not isinstance(value, dict):
            value = {}
        props = schema.get("properties") or {}
        extra = schema.get("additionalProperties", True)
        out = {}
        for name, item in value.items():
            if name in props:
                out[name] = repair(item, props[name])
            elif extra is not False:
                out[name] = item
            else:
                alt = _closest_key(name, [k for k in props if k not in value])
                if alt is not None and alt not in out:
                    out[alt] = repair(item, props[alt])
        for name in schema.get("required") or []:
            if name not in out:
                fill = default_for(props.get(name))
                if fill is not None:
                    out[name] = fill
        return out
    if schema.get("enum"):
        if not any(_equal(value, v) for v in schema["enum"]):
            value = _closest_enum(value, schema["enum"])
        return value
    if ("integer" in kinds or "number" in kinds) and not _is_kind(value, "number"):
        try:
            value = float(str(value).replace(",", "").strip())
        except ValueError:
            return default_for(schema)
    if "integer" in kinds and not ("number" in kinds) and _is_kind(value, "number"):
        value = int(round(value))
    if _is_kind(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            value = schema["minimum"]
        if "maximum" in schema and value > schema["maximum"]:
            value = schema["maximum"]
        return value
    if "string" in kinds:
        if not isinstance(value, str):
            value = value if value is None else (json.dumps(value) if isinstance(value, (dict, list)) else str(value))
        if isinstance(value, str) and "maxLength" in schema and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
        return value
    if "array" in kinds:
        if not isinstance(value, list):
            value = [] if value is None else [value]
        if "maxItems" in schema:
            value = value[: schema["maxItems"]]
        if isinstance(schema.get("items"), dict):
            value = [repair(v, schema["items"]) for v in value]
        return value
    if "boolean" in kinds and not isinstance(value, bool):
        return str(value).strip().lower() in ("1", "true", "yes", "y")
    return value


# ======================================================================================
# 3. WorldAdapter: tool roles from names, descriptions and schemas; observations by shape
# ======================================================================================

ROLES = ("search", "browse", "detail", "spec", "compare", "message", "cart_add", "cart_view",
         "order", "decline", "account", "status")

_ROLE_NAME = {
    "search": {"search", "find", "query", "lookup", "look", "seek", "goods", "products", "items",
               "catalog", "catalogue"},
    "browse": {"shelf", "aisle", "section", "department", "category", "browse", "categories"},
    "detail": {"detail", "details", "view", "inspect", "offer", "listing", "show", "info",
               "describe", "open"},
    "spec": {"spec", "specs", "sheet", "datasheet", "properties", "attributes", "facts", "traits",
             "specification"},
    "compare": {"contrast", "compare", "comparison", "versus", "side"},
    "message": {"message", "contact", "customer", "shopper", "ask", "write", "chat", "reply",
                "question", "talk", "tell", "send"},
    "cart_add": {"add", "put", "keep", "set", "cart", "basket", "bag", "trolley", "quantity",
                 "modify", "update", "to"},
    "cart_view": {"cart", "basket", "bag", "trolley", "summary", "contents", "list", "view",
                  "review", "show"},
    "order": {"order", "checkout", "purchase", "buy", "submit", "settle", "conclude", "wrap",
              "place", "complete", "pay", "finalize", "finalise", "finish", "end", "done", "commit"},
    "decline": {"decline", "give", "abstain", "cancel", "quit", "refuse", "abandon", "nothing",
                "skip", "pass", "without", "reject", "none"},
    "account": {"account", "points", "wallet", "balance", "report", "loyalty", "profile",
                "credit", "member"},
    "status": {"status", "remaining"},
}
_ROLE_DESC = {
    "search": {"search", "keyword", "keywords", "find", "matching", "query", "look"},
    "browse": {"category", "section", "shelf", "aisle", "department", "browse"},
    "detail": {"full", "details", "detail", "single", "inspect", "offer", "listing"},
    "spec": {"property", "properties", "attribute", "attributes", "spec", "specification",
             "datasheet", "facts"},
    "compare": {"compare", "contrast", "side", "several", "multiple", "together"},
    "message": {"shopper", "customer", "question", "ask", "message", "pose", "reply"},
    "cart_add": {"cart", "basket", "bag", "trolley", "add", "modify", "quantity", "remove"},
    "cart_view": {"cart", "basket", "bag", "trolley", "contents", "your", "total"},
    "order": {"checkout", "check", "purchase", "final", "order", "buy", "place", "submit"},
    "decline": {"decline", "without", "give", "close", "nothing", "abstain", "refuse", "cancel"},
    "account": {"account", "points", "balance", "loyalty", "credit"},
    "status": {"status", "remaining", "left"},
}
_ROLE_PHRASES = {   # description phrases: strong evidence when tool and argument names are opaque
    "decline": (re.compile(r"no purchase|without (?:a |an |any )?(?:buying|purchas|ordering|order\b)|nothing bought|give up|close out|"
                           r"abstain|decline|end without|order nothing", re.IGNORECASE), 7),
    "order": (re.compile(r"check ?out|place (?:the |an )?order|buy what|purchase", re.IGNORECASE), 3),
    "cart_view": (re.compile(r"your (?:cart|bag|basket|trolley)|cart contents|what is in the|show the (?:bag|cart|basket)|summary", re.IGNORECASE), 3),
    "cart_add": (re.compile(r"modify cart|add to (?:the )?(?:cart|bag|basket)|change the quantity|units of a listing|"
                            r"quantity of (?:a |an |the )?(?:listing|item|product|article)|in the (?:cart|bag|basket|trolley)\b|"
                            r"in the basket|into the cart", re.IGNORECASE), 3),
    "detail": (re.compile(r"in full|single article|full details|current offer|one listing", re.IGNORECASE), 2),
    "spec": (re.compile(r"property list|properties|spec(?:ification)? sheet|datasheet|attributes", re.IGNORECASE), 3),
    "message": (re.compile(r"question to (?:the )?(?:shopper|customer)|message (?:the )?(?:shopper|customer)|write to the|"
                           r"ask the (?:shopper|customer)|send the customer", re.IGNORECASE), 4),
    "search": (re.compile(r"\bsearch|keyword|look up items|query items", re.IGNORECASE), 2),
    "browse": (re.compile(r"shelf section|by section|filter search|category filter", re.IGNORECASE), 2),
    "account": (re.compile(r"account|loyalty|points", re.IGNORECASE), 2),
}
_NAME_DECLINE = re.compile(r"\b(?:(?:stand|step) down|(?:bow|back|opt|walk) out|walk away)\b")   # G1: two-word name cues
_SHARED_ROLES = ({"detail", "spec"}, {"search", "browse"})
_BIND_MIN = 4

_P_CATEGORY = {"category", "department", "section", "shelf", "aisle", "dept", "categories", "range", "class"}
_P_BRAND = {"brand", "maker", "manufacturer", "company"}
_P_TOTAL = {"wanted", "needed", "total"}   # G3: a count that names the line's total
_P_ID = {"id", "ids", "code", "pid", "sku", "item", "product", "listing", "article", "target",
         "chosen", "asin", "ref", "listings", "items", "products"}
_P_PAGE = {"page", "pageno", "start", "offset", "row", "skip", "cursor", "token", "after", "next",
           "continue", "further", "bookmark", "resume", "first", "onward"}
_P_COUNT = {"count", "qty", "quantity", "many", "units", "number", "howmany", "amount", "needed",
            "wanted"}
_P_CAP = {"cap", "ceiling", "max", "maximum", "budget", "limit", "bound", "price", "spend",
          "upto", "under"}
_P_TEXT = {"text", "query", "keywords", "keyword", "wanted", "search", "term", "terms", "phrase",
           "free", "q", "body", "note", "message", "remark", "explanation", "reply", "question",
           "comment", "reason", "words", "string"}

_KNOWN_PARAM_WORDS = (_P_CATEGORY | _P_BRAND | _P_ID | _P_PAGE | _P_COUNT | _P_CAP | _P_TEXT
                      | {"mode", "type", "using", "by", "route", "choice", "verb", "step", "action", "operation",
                         "reason", "remark", "note", "explanation", "body", "lines", "line", "basket", "cart",
                         "bag", "trolley", "method", "how", "change"})
_E_BUY = {"checkout", "buy", "purchase", "order", "place", "confirm", "complete", "pay", "proceed",
          "submit", "commit", "finish", "accept", "yes"}
_E_ABSTAIN = {"abstain", "cancel", "decline", "none", "skip", "nothing", "abort", "quit", "refuse",
              "pass", "giveup", "give", "without", "empty", "no", "withdraw", "forfeit", "reject", "nobuy", "not"}
_E_ADD = {"include", "plus", "add", "put", "insert", "increase", "more", "in", "hold"}
_E_REMOVE = {"discard", "remove", "minus", "delete", "drop", "decrease", "take", "out", "less", "release"}
_E_KEYWORD = {"typed", "text", "keyword", "keywords", "free", "query", "search", "fulltext", "words",
              "phrase", "freetext", "term", "terms"}
_E_FILTER = {"fields", "narrow", "filter", "category", "facet", "attribute", "attributes",
             "browse", "field", "structured", "department", "shelf", "section", "aisle"}
_ENUM_SENSES = (("route", _E_BUY, _E_ABSTAIN), ("verb", _E_ADD, _E_REMOVE),
                ("mode", _E_KEYWORD, _E_FILTER))

_F_TITLE = {"title", "name", "caption", "heading", "label", "headline"}
_F_CATEGORY = {"category", "department", "aisle", "shelf", "section", "dept"}
_F_BRAND = {"brand", "company", "maker", "manufacturer"}
_F_SELLER = {"seller", "outlet", "store", "vendor", "merchant", "shop", "sold"}
_F_STOCK = {"stock", "available", "availability", "buy", "instock", "can", "ok", "avail",
            "sellable", "purchasable", "orderable"}
_F_PRICE = {"price", "cost", "amount", "payable", "fee", "charge", "usd", "cents", "money"}
_F_STOCK_NOT = {"sold", "unavailable", "oos", "backorder", "backordered", "outofstock"}
_STOCK_WORDS = {"available": True, "instock": True, "onshelf": True, "yes": True, "true": True, "soldout": False,
                "outofstock": False, "unavailable": False, "backorder": False, "backordered": False, "no": False,
                "false": False}   # looked up with spaces, dashes and underscores removed
_F_QTY = {"qty", "quantity", "count", "units", "number", "line", "pieces"}
_F_RATING = {"rating", "stars", "score", "reviews", "rated"}
_F_ID = {"id", "pid", "code", "sku", "asin"}
_O_TOTAL = {"count", "total", "matched", "matches", "hits", "size", "number", "found", "results"}
_O_NEXT = {"next", "after", "afterwards", "further", "continue", "cursor", "token", "more", "page",
           "offset", "following", "key"}
_O_ERROR = {"error", "fault", "issue", "problem", "failure", "invalid", "warning"}
_O_CART_TOTAL = {"total", "payable", "grand", "sum", "subtotal", "cost", "amount", "price"}


def classify_param(name, schema) -> str:
    """Semantic kind of one tool parameter: id, ids, text, category, brand, page, count, cap,
    enum, bool, array or other."""
    schema = schema if isinstance(schema, dict) else {}
    w = set(words(name))
    kinds = _kinds(schema)
    if schema.get("enum"):
        return "enum"
    if "array" in kinds:
        items = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        inner = items.get("properties") if isinstance(items.get("properties"), dict) else {}
        has_id = any(set(words(k)) & _P_ID and "string" in _kinds(v if isinstance(v, dict) else {}) for k, v in inner.items())
        has_count = any("integer" in _kinds(v if isinstance(v, dict) else {}) for v in inner.values())
        if has_id and has_count:
            return "lines"        # a whole cart as [{id, count}, ...]
        return "ids" if w & _P_ID else "array"
    if "boolean" in kinds:
        return "bool"
    if "integer" in kinds or "number" in kinds:
        if w & _P_PAGE:
            return "page"
        if w & _P_CAP:
            return "cap"
        if w & _P_COUNT:
            return "count"
        return "count" if "integer" in kinds else "cap"
    if w & _P_PAGE and not w & _P_TEXT:
        return "page"
    if w & _P_CATEGORY:
        return "category"
    if w & _P_BRAND:
        return "brand"
    if w & _P_ID and not w & _P_TEXT:
        return "id"
    return "text"


_PAGE_STRONG = {"page", "pageno", "cursor", "token", "bookmark", "offset"}   # the words of _P_PAGE prose uses only for paging


def described_kind(text, schema):
    """R5-S13: the kind an argument's own description gives, or None when it names none."""
    w, kinds = set(words(text)), _kinds(schema)
    if "array" in kinds:
        found = classify_param(text, schema)
        return found if found in ("ids", "lines") else None
    if "boolean" in kinds:
        return None
    if "integer" in kinds or "number" in kinds:
        return "page" if w & _PAGE_STRONG else "cap" if w & _P_CAP else "count" if w & _P_COUNT else None
    if w & _PAGE_STRONG:
        return "page"
    if w & _P_CATEGORY:
        return "category"
    if w & _P_BRAND:
        return "brand"
    if w & _P_ID and not w & _P_TEXT:
        return "id"
    return "text" if w & _P_TEXT else None


_S_LEFT = {"left", "leftover", "remaining"}
_S_USED = {"used", "so", "far", "count"}


def status_counts(raw, depth=0):
    """G4: (used, left) from a status observation: in one object, the integer whose name says what is left and
    the one that counts what was used."""
    if not isinstance(raw, dict) or depth > 3:
        return None
    ints = {k: v for k, v in raw.items() if isinstance(v, int) and not isinstance(v, bool)}
    left = [v for k, v in ints.items() if set(words(k)) & _S_LEFT]
    used = [v for k, v in ints.items() if set(words(k)) & _S_USED and not set(words(k)) & _S_LEFT]
    if len(left) == 1 and len(used) == 1:
        return used[0], left[0]
    for v in raw.values():
        found = status_counts(v, depth + 1)
        if found:
            return found
    return None


def enum_sense(values) -> dict:
    """What an enum parameter switches: route (buy/abstain), verb (add/remove) or mode
    (keyword/filter search), with the value for each side."""
    values = list(values or [])
    for one_sided in (False, True):
        if one_sided and len(values) == 2:
            rem = [v for v in values if set(words(str(v))) & _E_REMOVE and not set(words(str(v))) & _E_ADD]
            if len(rem) == 1:   # C1: one value removes, so the pair is an add/remove switch and the other value adds
                return {"sense": "verb", "pos": values[1 - values.index(rem[0])], "neg": rem[0]}
        for sense, pos_set, neg_set in _ENUM_SENSES:
            pos = next((v for v in values if set(words(str(v))) & pos_set and not set(words(str(v))) & neg_set), None)
            neg = next((v for v in values if set(words(str(v))) & neg_set and v != pos), None)
            if pos is not None and (neg is not None or one_sided):
                return {"sense": sense, "pos": pos, "neg": neg}
    return {"sense": None, "pos": values[0] if values else None, "neg": None}


class ToolInfo:
    """One declared tool, with its words and parameter kinds pre-computed."""

    def __init__(self, decl):
        fn = decl.get("function") if isinstance(decl.get("function"), dict) else decl
        self.decl = decl
        self.name = str(fn.get("name") or "")
        self.desc = str(fn.get("description") or "")
        schema = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        self.schema = schema or {"type": "object", "properties": {}}
        self.props = self.schema.get("properties") or {}
        self.required = [r for r in (self.schema.get("required") or []) if r in self.props]
        self.name_words = set(words(self.name))
        self.desc_words = set(words(self.desc))
        self.kind = {k: classify_param(k, s) for k, s in self.props.items()}
        self.opaque = {k for k in self.props if not set(words(k)) & _KNOWN_PARAM_WORDS}
        for k in self.opaque:
            self.kind[k] = self._shape_kind(self.props[k])
        for k, v in self.props.items():
            about = v.get("description") if isinstance(v, dict) and not v.get("enum") else None
            found = described_kind(about, v) if isinstance(about, str) else None
            if found:
                self.kind[k] = found   # R5-S13: an argument's own description says what it is
                self.opaque.discard(k)
        plain = [k for k, v in self.kind.items() if v == "text" and not (self.props.get(k) or {}).get("enum")]
        named_text = [k for k in plain if set(words(k)) & _P_TEXT]
        loose_int = [k for k, v in self.kind.items() if v == "count" and k not in self.required
                     and not set(words(k)) & _P_COUNT and "integer" in _kinds(self.props.get(k) or {})]
        if len(self.props) == 2 and len(named_text) == 1 and len(loose_int) == 1:
            self.kind[loose_int[0]] = "page"   # V1: beside a query string, the optional nameless integer is the offset
        if (len(self.props) == 2 and list(self.kind.values()).count("count") == 1 and len(plain) == 1
                and "string" in _kinds(self.props.get(plain[0]) or {}) and not set(words(plain[0])) & _P_TEXT):
            self.kind[plain[0]] = "id"   # R5-S3: a count and one plain string: the string names what is counted (V2: never a query)
        shapes = {k: (v if isinstance(v, dict) else {}) for k, v in self.props.items()}
        if not self.has("page") and re.search(r"\bcontinu|next page|more results", self.desc, re.IGNORECASE):
            loose = [k for k, v in shapes.items() if k not in self.required and self.kind[k] in ("text", "id")
                     and "string" in _kinds(v) and not v.get("enum")]
            named = [k for k in loose if re.search(r"\b%s\b" % re.escape(k), self.desc)]
            free = [k for k in loose if not shapes[k].get("maxLength")]
            pick = named or free
            if len(pick) == 1:
                self.kind[pick[0]] = "page"   # R5-S7: the optional string the description says to continue with
        if (not self.has("category") and self.desc_words & _ROLE_DESC["browse"] and not self.has("count")
                and (self.has("cap") or self.has("page")) and not self.desc_words & {"search", "keyword", "keywords", "query"}
                and not any(v.get("enum") for v in shapes.values())):
            slots = [k for k, v in shapes.items() if self.kind[k] in ("text", "id") and k not in self.required
                     and v.get("maxLength") and "string" in _kinds(v)]
            if slots:
                self.kind[slots[0]] = "category"   # R5-S8: a filter tool "by category": its first bounded string
        self.enums = {k: enum_sense(s.get("enum")) for k, s in self.props.items()
                      if isinstance(s, dict) and s.get("enum")}

    def _shape_kind(self, schema):
        """Kind of a parameter whose name says nothing, from its schema and the tool's purpose."""
        schema = schema if isinstance(schema, dict) else {}
        kinds = _kinds(schema)
        if schema.get("enum"):
            vals = [str(v) for v in schema["enum"]]
            if len(vals) >= 2 and all(re.search(r"\d$", v) for v in vals) and len({re.sub(r"\d+$", "", v) for v in vals}) == 1 \
                    and any(re.search(r"[a-z]", v, re.IGNORECASE) for v in vals):
                return "page"   # R5-B3: numbered continuation tokens sharing one stem ("<stem>1", "<stem>2", ...)
            return "enum"
        if "array" in kinds:
            return "lines" if classify_param("items", schema) == "lines" else "ids"
        if "number" in kinds and "integer" not in kinds:
            return "cap"
        if "integer" in kinds:
            searchy = bool(self.desc_words & {"search", "query", "list", "rows", "articles", "items", "find"})
            if "multipleOf" in schema or (searchy and schema.get("maximum", 99) <= 50):
                return "page"
            return "count"
        longest = schema.get("maxLength", 0)
        searchy = bool(self.desc_words & {"search", "find", "query", "look", "catalogue", "catalog", "keyword", "keywords", "matching"})
        if longest and longest <= 64 and not searchy and (schema.get("minLength", 0) >= 1 or len(self.props) <= 2):
            return "id"
        if not longest and not searchy and len(self.props) <= 2 and _ROLE_PHRASES["detail"][0].search(self.desc):
            return "id"   # R5-S4: an unbounded string of a one-listing tool names the listing
        return "text"

    def param(self, kind, sense=None):
        """The parameter of `kind` (required ones first); for enums, the one with `sense`."""
        names = [k for k, v in self.kind.items() if v == kind
                 and (sense is None or self.enums.get(k, {}).get("sense") == sense)]
        names.sort(key=lambda k: (k not in self.required, k))
        return names[0] if names else None

    def has(self, kind) -> bool:
        return kind in self.kind.values()

    def fit(self, role) -> int:
        """How well the parameter schema fits a role (positive = plausible)."""
        n, req = len(self.props), [self.kind[r] for r in self.required]
        if role == "search":
            score = 3 if self.has("text") else (1 if self.param("enum", "mode") else -4)
            score += 1 if self.has("page") else 0
            score += 2 if self.param("enum", "mode") else 0
            return score - (5 if "id" in req else 0)
        if role == "browse":
            return (4 if self.has("category") else -3) + (1 if self.has("cap") else 0)
        if role in ("detail", "spec"):
            if self.has("lines") or any(self.kind[k] == "count" and (set(words(k)) & (_P_COUNT | _F_QTY) or k in self.opaque)
                                        for k in self.required):
                return -4   # D1: an id with a required quantity is a cart; reading through it changes the basket
            if req.count("id") == 1 and n <= 2:
                return 4
            return 0 if self.has("id") else -4
        if role == "compare":
            return 5 if self.has("ids") else (-8 if self.has("lines") else -5)
        if role == "message":
            if self.has("id") or not self.has("text") or n > 2:
                return -4
            longest = max((s.get("maxLength", 0) for s in self.props.values() if isinstance(s, dict)), default=0)
            return 3 + (1 if longest >= 200 else 0)
        if role == "cart_add":
            if (self.has("id") and self.has("count")) or self.has("lines"):
                return 6
            return 1 if self.has("id") else -6
        if role == "cart_view":
            return (2 if n == 0 else 1) if not self.required else -4
        if role == "order":
            if self.has("id") or "text" in req:
                return -4
            if self.param("enum", "route"):
                return 3
            return 2 if not self.required else -2
        if role == "decline":
            if self.has("id"):
                return -4
            return 2 if not [k for k in req if k != "text"] else -2
        if role == "account":
            return 1 if not self.required else -2
        if role == "status":
            return 2 if not self.props else -4
        return 0


class Binding:
    """A role bound to a tool, with argument values the role always sends."""

    def __init__(self, tool, fixed=None):
        self.tool = tool
        self.fixed = dict(fixed or {})

    def __repr__(self):
        return f"Binding({self.tool.name}, {self.fixed})"


class Obs:
    """One observation, normalised by value shape (never by literal key names)."""

    def __init__(self, raw):
        self.raw = raw
        self.kind = "other"   # listing | product | cart | reply | ack | error | empty | other
        self.products = []    # normalised products (listing rows or one detailed view)
        self.cart = None      # {"lines": [{"id", "qty", "price", "title"}], "total": x}
        self.text = None      # shopper reply to a message
        self.lines = []       # side-channel shopper lines (changes announced mid-episode)
        self.refusal = None   # list of lines when an order was refused
        self.error = None
        self.ok = None        # outcome of an acknowledgement
        self.next = None      # token for the next page (None = no more pages)
        self.total = None     # number of matches the world reported

    def summary(self) -> dict:
        """Compact view for logs and for the model actor."""
        out = {"kind": self.kind}
        if self.error:
            out["error"] = clip(self.error, 200)
        if self.products:
            out["products"] = [compact_product(p, full=self.kind == "product") for p in self.products[:12]]
            if self.total is not None:
                out["total_matches"] = self.total
            out["more_pages"] = self.next is not None
        if self.cart is not None:
            out["cart"] = self.cart
        if self.text:
            out["shopper_reply"] = clip(self.text, 600)
        if self.ok is not None:
            out["ok"] = self.ok
        if self.lines:
            out["shopper_update"] = self.lines
        if self.refusal:
            out["order_refused"] = self.refusal
        return out


def compact_product(p, full=False) -> dict:
    out = {"id": p.get("id"), "title": clip(p.get("title") or "", 110), "price": p.get("price"),
           "in_stock": p.get("in_stock")}
    if full:
        for key in ("brand", "category", "seller"):
            if p.get(key):
                out[key] = p[key]
        props = p.get("props") or {}
        if props:
            out["props"] = {clip(k, 40): clip(str(v), 70) for k, v in list(props.items())[:30]}
        if p.get("text"):
            out["text"] = clip(p["text"], 400)
    return out


class WorldAdapter:
    """Binds roles to tools once, builds role calls, and normalises every observation."""

    def __init__(self, tools):
        self.tools = [ToolInfo(t) for t in tools or [] if isinstance(t, dict)]
        self.by_name = {t.name: t for t in self.tools}
        self.roles = {}
        self.cart_mode = None     # "verb" | "absolute" | "add"
        self.money_unit = None    # "cents" | "dollars" when the world says so
        self.stock_inverted = any(re.search(r"\btrue means (?:un|not |sold|out)", t.desc, re.IGNORECASE)
                                  for t in self.tools)   # a stock flag whose true means unavailable
        self.field_roles = {}     # R5-B1: roles of opaque observation fields, learned from listing rows
        self.opaque_world = False   # R5-B1: no field of the listing rows carries a role word
        self._bind()
        for t in self.tools:
            unit = re.search(r"\b(?:in|as) (cents|dollars|usd)\b|prices?[^.]{0,30}\b(cents|dollars)\b", t.desc, re.IGNORECASE)
            if unit:
                found = (unit.group(1) or unit.group(2)).lower()
                self.money_unit = "cents" if found == "cents" else "dollars"
            elif {"cents", "cent"} & t.desc_words:
                self.money_unit = "cents"

    # ---- binding -------------------------------------------------------------------
    def _bind(self):
        scored = []
        for t in self.tools:
            for role in ROLES:
                score = (3 * len(t.name_words & _ROLE_NAME[role])
                         + min(3, len(t.desc_words & _ROLE_DESC[role])) + t.fit(role))
                phrase = _ROLE_PHRASES.get(role)
                if phrase and phrase[0].search(t.desc):
                    if role != "order" or not _ROLE_PHRASES["decline"][0].search(t.desc):
                        score += phrase[1]
                if role == "status" and not t.props and t.desc_words & _S_LEFT and t.desc_words & _S_USED:
                    score += 6   # R5-S9: no arguments, and it reports what is left and what was used
                if _NAME_DECLINE.search(" ".join(words(t.name))):   # G1: "stand down" ends without an order
                    score += 6 if role == "decline" else (-6 if role == "order" else 0)
                scored.append((score, role, t.name))
        scored.sort(key=lambda x: (-x[0], ROLES.index(x[1]), x[2]))
        tool_roles = {}
        for score, role, name in scored:
            if score < _BIND_MIN or role in self.roles:
                continue
            held = tool_roles.get(name, set())
            if held and not all({role, r} in _SHARED_ROLES for r in held):
                continue
            self.roles[role] = Binding(self.by_name[name])
            tool_roles.setdefault(name, set()).add(role)
        both = self.roles.get("detail")
        if both and "spec" in self.roles and self.roles["spec"].tool is both.tool:
            held = {b.tool.name for b in self.roles.values()}
            spare = [n for sc, r, n in scored if r == "detail" and n not in held and sc >= _BIND_MIN]
            spec_of = {n: sc for sc, r, n in scored if r == "spec"}
            if spare and spec_of.get(both.tool.name, 0) > spec_of.get(spare[0], 0):
                self.roles["detail"] = Binding(self.by_name[spare[0]])   # R5-S15: the property-list tool keeps spec; the
                #            free tool that fits a detail view takes detail, never left unbound
        if "spec" not in self.roles and "detail" in self.roles:
            self.roles["spec"] = Binding(self.roles["detail"].tool)
        if "detail" not in self.roles and "spec" in self.roles:
            self.roles["detail"] = Binding(self.roles["spec"].tool)
        # keyword and filter search on one combined tool
        for role, side in (("search", "pos"), ("browse", "neg")):
            b = self.roles.get(role)
            mode = b.tool.param("enum", "mode") if b else None
            if mode:
                value = b.tool.enums[mode][side]
                if value is None and role == "browse":
                    del self.roles["browse"]
                    continue
                b.fixed[mode] = value
        # order and decline routes on one tool
        order = self.roles.get("order")
        route = order.tool.param("enum", "route") if order else None
        if route:
            sense = order.tool.enums[route]
            order.fixed[route] = sense["pos"]
            if "decline" not in self.roles and sense["neg"] is not None:
                self.roles["decline"] = Binding(order.tool, {route: sense["neg"]})
        dec = self.roles.get("decline")
        route = dec.tool.param("enum", "route") if dec and "order" not in self.roles else None
        if route and dec.tool.enums[route]["pos"] is not None:   # R3: a route tool taken as decline first also orders
            sense = dec.tool.enums[route]
            self.roles["order"] = Binding(dec.tool, {route: sense["pos"]})
            if sense["neg"] is not None:
                dec.fixed[route] = sense["neg"]
        self._fallback_bindings()
        order = self.roles.get("order")
        route = order.tool.param("enum", "route") if order else None
        if route and route not in order.fixed:
            sense = order.tool.enums[route]
            order.fixed[route] = sense["pos"]
            if "decline" not in self.roles and sense["neg"] is not None:
                self.roles["decline"] = Binding(order.tool, {route: sense["neg"]})
        cart = self.roles.get("cart_add")
        if cart:
            if cart.tool.has("lines"):
                self.cart_mode = "vector"
            elif re.search(r"exact (?:quantity|count|number)|zero removes|set (?:the )?(?:quantity|count)", cart.tool.desc, re.IGNORECASE):
                self.cart_mode = "absolute"
            elif cart.tool.param("enum", "verb"):
                self.cart_mode = "verb"
            elif ({"set", "update", "change"} & cart.tool.name_words
                  or {"set", "sets", "replace", "replaces", "exact"} & cart.tool.desc_words
                  or (cart.tool.props.get(cart.tool.param("count")) or {}).get("minimum") == 0
                  or set(words(cart.tool.param("count") or "")) & _P_TOTAL):
                self.cart_mode = "absolute"  # a count that may be 0 sets the line rather than adds
            else:
                self.cart_mode = "add"

    def _fallback_bindings(self):
        """Never leave a world without a finish or a cart role: take the unbound tool with the
        most order-like (or cart-like) evidence in its name, description and arguments."""
        held = {b.tool.name for b in self.roles.values()}
        free = [t for t in self.tools if t.name not in held]
        if "order" not in self.roles:
            cue = re.compile(r"check ?out|order|purchase|buy|finish|final|\bend\b|commit|conclude|wrap|complete|submit", re.IGNORECASE)
            def said(t):   # B2: the name's words (camelCase split), the description and the enum values
                values = [str(v) for spec in t.props.values() if isinstance(spec, dict) for v in (spec.get("enum") or [])]
                return " ".join(words(t.name)) + " " + t.desc + " " + " ".join(values)
            ranked = sorted(((len(cue.findall(said(t))) - (5 if t.has("id") else 0), t.name) for t in free), reverse=True)
            if ranked and ranked[0][0] > 0:
                tool = self.by_name[ranked[0][1]]
                fixed = {}
                for name, sense in tool.enums.items():
                    values = list(tool.props[name].get("enum") or [])
                    neg = next((v for v in values if set(words(str(v))) & _E_ABSTAIN), None)
                    buy = [v for v in values if v != neg and cue.search(" ".join(words(str(v))))]
                    pos = buy[0] if buy else next((v for v in values if v != neg), None)   # R3: the value that says buy
                    if pos is not None:
                        fixed[name] = pos
                        tool.enums[name] = {"sense": "route", "pos": pos, "neg": neg}
                self.roles["order"] = Binding(tool, fixed)
                free = [t for t in free if t is not tool]
        if "cart_add" not in self.roles:
            ranked = [t for t in free if (t.has("id") and t.has("count")) or t.has("lines")]
            if ranked:
                self.roles["cart_add"] = Binding(ranked[0])
        if "status" not in self.roles and "message" in self.roles and all(r in self.roles for r in ("order", "decline", "cart_view")):
            held = {b.tool.name for b in self.roles.values()}
            bare = [t for t in self.tools if t.name not in held and not t.props]
            if len(bare) == 1:
                self.roles["status"] = Binding(bare[0])   # ST1: the one argument-free tool no role took reads the status

    def cart_vector(self, lines):
        """The one call that sets the whole cart for an array-of-lines cart tool."""
        b = self.roles.get("cart_add")
        if b is None or self.cart_mode != "vector":
            return None
        name = b.tool.param("lines")
        inner = ((b.tool.props.get(name) or {}).get("items") or {}).get("properties") or {}
        idk = next((k for k, v in inner.items() if set(words(k)) & _P_ID and "string" in _kinds(v)), None)
        cnt = next((k for k, v in inner.items() if "integer" in _kinds(v)), None)
        return self._call("cart_add", {name: [{idk: pid, cnt: q} for pid, q in lines.items()]})

    def role_of(self, name, args=None):
        """The role a call plays (the decline route on a shared tool is told apart by args)."""
        hits = [r for r, b in self.roles.items() if b.tool.name == name]
        if not hits:
            return None
        if len(hits) > 1 and isinstance(args, dict):
            for r in hits:
                b = self.roles[r]
                if b.fixed and all(args.get(k) == v for k, v in b.fixed.items()):
                    return r
            for r in ("order", "search", "detail"):
                if r in hits:
                    return r
        return hits[0]

    def describe(self) -> dict:
        return {r: [b.tool.name, b.fixed] for r, b in sorted(self.roles.items())}

    # ---- call builders: each returns (tool name, repaired args) or None --------------
    def _call(self, role, args):
        b = self.roles.get(role)
        if b is None:
            return None
        merged = dict(b.fixed)
        merged.update({k: v for k, v in args.items() if k is not None and v is not None})
        lost = [v for k, v in args.items() if k is None and v is not None]
        missing = [r for r in b.tool.required if r not in merged]
        if len(lost) == 1 and len(missing) == 1:
            merged[missing[0]] = lost[0]   # R5-S6: the value whose argument went unnamed fills the one required argument left
        return b.tool.name, repair(merged, b.tool.schema)

    @staticmethod
    def _page_arg(tool, token):
        """The page parameter and value for `token`; (None, None) when it cannot be sent."""
        if token is None:
            return None, None
        name = tool.param("page")
        if name is None:
            return None, None
        schema = tool.props.get(name) or {}
        if "integer" in _kinds(schema) or "number" in _kinds(schema):
            try:
                token = int(token)
            except (TypeError, ValueError):
                return None, None
            if "maximum" in schema and token > schema["maximum"]:
                return None, None
        else:
            token = str(token)
        return name, token

    def search(self, query, token=None):
        b = self.roles.get("search")
        if b is None:
            return None
        page, value = self._page_arg(b.tool, token)
        if token is not None and page is None:
            return None
        named = sorted((k for k, v in b.tool.kind.items() if v == "text" and k not in b.tool.opaque),
                       key=lambda k: (k not in b.tool.required, k))
        text = named[0] if named else b.tool.param("text")   # R5-S12: the argument whose name says text takes the query
        args = {text: query, page: value}
        if text in b.tool.opaque and b.tool.param("enum", "mode"):
            for name, kind in b.tool.kind.items():   # the keyword mode ignores filter slots
                if kind == "text" and name in b.tool.opaque:
                    args[name] = query
        return self._call("search", args)

    def browse(self, category, cap=None, token=None):
        b = self.roles.get("browse")
        if b is None:
            return None
        page, value = self._page_arg(b.tool, token)
        if token is not None and page is None:
            return None
        return self._call("browse", {b.tool.param("category"): category,
                                     b.tool.param("cap"): cap, page: value})

    def detail(self, pid, role="detail"):
        b = self.roles.get(role)
        return self._call(role, {b.tool.param("id"): pid}) if b else None

    def compare(self, ids):
        b = self.roles.get("compare")
        return self._call("compare", {b.tool.param("ids"): list(ids)}) if b else None

    def message(self, text):
        b = self.roles.get("message")
        return self._call("message", {b.tool.param("text"): text}) if b else None

    def cart_view(self):
        return self._call("cart_view", {})

    def order(self):
        return self._call("order", self._noted("order", {}, "Ordering the basket as confirmed."))

    def decline(self, note):
        b = self.roles.get("decline")
        return self._call("decline", self._noted("decline", {b.tool.param("text"): note}, note)) if b else None

    def _noted(self, role, args, note):
        """N1: every required free-text argument of a finish tool that the call leaves empty carries the schema's own
        default, else the note."""
        b = self.roles.get(role)
        for k in (b.tool.required if b else []):
            s = b.tool.props.get(k) or {}
            if k not in args and k not in b.fixed and "string" in _kinds(s) and not s.get("enum"):
                given = s.get("default")
                args[k] = given if isinstance(given, str) and given.strip() else note
        return args

    def account(self):
        return self._call("account", {})

    def status(self):
        return self._call("status", {})

    def step_tools(self):
        """P1: the tools a refusal may name as a required step: unbound, or bound only to read-only roles whose arguments
        the basket can fill (cart view, status, account, a product view). Never a cart, order, decline, message, search,
        browse or compare tool. (The binder takes any read-only tool with one listing id as the spec view.)"""
        held = {}
        for r, b in self.roles.items():
            held.setdefault(b.tool.name, set()).add(r)
        return sorted(t.name for t in self.tools if t.name and held.get(t.name, set()) <= {"cart_view", "status", "account", "detail", "spec"})

    def named_step(self, name, cart, text):
        """P1: the call for the declared tool a refused order named (Run.named_step); None when a required argument
        has no value from its schema or from the basket."""
        t = self.by_name.get(name)
        if t is None:
            return None
        ids = sorted(pid for pid, q in cart.items() if q > 0)
        said = set(words(text))
        args = {}
        for k in t.required:
            s = t.props.get(k) if isinstance(t.props.get(k), dict) else {}
            kinds = _kinds(s)
            if "const" in s or "default" in s:
                args[k] = s["const"] if "const" in s else s["default"]
            elif s.get("enum"):
                values = list(s["enum"])   # the value the refusal's words name, else the first
                args[k] = max(values, key=lambda v: (len(set(words(str(v))) & said), -values.index(v)))
            elif "boolean" in kinds:
                args[k] = True   # a required flag on a required step asserts it
            elif "array" in kinds and not isinstance(s.get("items") or {}, dict):
                return None   # P1c: a list "items" schema (tuple form): no value from the basket
            elif "array" in kinds and ids and "string" in (_kinds((s.get("items") or {})) or ["string"]):
                args[k] = ids
            elif t.kind.get(k) == "id" and len(ids) == 1:
                args[k] = ids[0]
            elif t.kind.get(k) == "count" and len(ids) == 1:
                args[k] = cart[ids[0]]
            elif t.kind.get(k) == "text" and "string" in kinds:
                args[k] = "Checked the basket before ordering."
            else:
                return None
        return name, repair(args, t.schema)

    def cart_effect(self, args, have_of):
        """(product id, units after) for an acknowledged cart call, from its own arguments;
        for an array-of-lines cart, the whole new cart as a dict."""
        b = self.roles.get("cart_add")
        if b is None or not isinstance(args, dict):
            return None
        t = b.tool
        if self.cart_mode == "vector":
            lines = args.get(t.param("lines")) or []
            out = {}
            for line in lines if isinstance(lines, list) else []:
                if isinstance(line, dict):
                    pid = next((v for k, v in line.items() if isinstance(v, str)), None)
                    qty = next((v for k, v in line.items() if isinstance(v, int) and not isinstance(v, bool)), 1)
                    if pid:
                        out[pid] = qty
            return out
        pid, count = args.get(t.param("id")), args.get(t.param("count"))
        if pid is None:
            return None
        count = count if isinstance(count, int) and not isinstance(count, bool) else 1
        have = have_of(pid)
        if self.cart_mode == "absolute":
            return pid, count
        verb = t.param("enum", "verb")
        if verb and args.get(verb) == t.enums[verb]["neg"]:
            return pid, max(0, have - count)
        if set(words(t.param("count") or "")) & _P_TOTAL:
            return pid, count   # G3: the count names the line's total
        return pid, have + count

    def learn_cart(self, args, before, cart):
        """G3: an add-mode cart call on a line already held that comes back at the sent count (or gone at 0)
        shows a world that sets lines."""
        b = self.roles.get("cart_add")
        if self.cart_mode != "add" or b is None or not isinstance(args, dict):
            return
        pid, count = args.get(b.tool.param("id")), args.get(b.tool.param("count"))
        if pid is None or not isinstance(count, int) or isinstance(count, bool):
            return
        now = {line.get("id"): line.get("qty") for line in cart.get("lines") or []}
        if before.get(pid, 0) > 0 and now.get(pid, 0) == count:
            self.cart_mode = "absolute"

    def cart_ops(self, pid, have, want):
        """Calls that move the cart line for `pid` from `have` to `want` units: [] when equal,
        None when this world's cart tool cannot do it."""
        b = self.roles.get("cart_add")
        if b is None:
            return None
        if have == want:
            return []
        if self.cart_mode == "vector":
            return None   # handled for the whole cart by Run.ensure_cart
        t = b.tool
        idp, cnt = t.param("id"), t.param("count")
        bounds = t.props.get(cnt) or {}
        lo, hi = bounds.get("minimum", 0), bounds.get("maximum", 10 ** 6)
        if self.cart_mode == "absolute":
            return None if want < lo or want > hi else [self._call("cart_add", {idp: pid, cnt: want})]
        verb = t.param("enum", "verb")
        value = None
        if self.cart_mode == "verb":
            value = t.enums[verb]["pos"] if want > have else t.enums[verb]["neg"]
            if value is None:
                return None
        elif want < have:
            if want == 0 and lo <= 0:   # G3: 0 removes the line where lines are set; learn_cart then knows
                return [self._call("cart_add", {idp: pid, cnt: 0})]
            return None  # an add-only cart cannot shrink
        if self.cart_mode == "verb" and want > have and want <= hi and set(words(cnt or "")) & _P_TOTAL:
            return [self._call("cart_add", {idp: pid, cnt: want, verb: value})]   # G3: the count names the line's total
        calls, delta = [], abs(want - have)
        while delta > 0:
            chunk = min(delta, hi)
            calls.append(self._call("cart_add", {idp: pid, cnt: chunk, verb: value}))
            delta -= chunk
        return calls

    # ---- observation normalisation -------------------------------------------------
    def normalise(self, raw, role=None) -> Obs:
        obs = Obs(raw)
        x = raw
        for _ in range(4):
            if isinstance(x, dict) and set(x) == {"observation", "refusal"}:
                refusal = x["refusal"] if isinstance(x["refusal"], list) else [x["refusal"]]
                obs.refusal = [s for s in refusal if isinstance(s, str)] or [json.dumps(x["refusal"])]
                x = x["observation"]
                continue
            unwrapped = self._unwrap(x)
            if unwrapped is None:
                break
            x, lines = unwrapped
            obs.lines += [s for s in lines if s not in obs.lines]
        if role == "order" and self.opaque_world and obs.lines and not obs.refusal:
            obs.refusal, obs.lines = list(obs.lines), []   # R5-B4: an opaque world's checkout that answers with shopper lines was refused
        elif role in ("order", "decline") and obs.lines and not obs.refusal and isinstance(x, dict):
            obs.refusal, obs.lines = list(obs.lines), []   # R5-S11: a finishing call answered by {<result>, [shopper lines]}
            #            was held (when it ended the episode instead, env.finished says so and nothing follows)
        self._classify(obs, x, role)
        return obs

    @staticmethod
    def _unwrap(x):
        """A {<inner>, [shopper lines]} side-channel envelope, whatever its two key names."""
        if not isinstance(x, dict) or len(x) != 2:
            return None
        (_, va), (_, vb) = x.items()
        for lines, inner in ((va, vb), (vb, va)):
            if (is_sentence_list(lines) and not is_sentence_list(inner)
                    and isinstance(inner, (dict, str, list))
                    and any(re.search(r"[.!?,']", s) or len(s.split()) >= 6 for s in lines)):
                return inner, lines
        return None

    @staticmethod
    def _error_of(x, role):
        if "error" in x and not isinstance(x["error"], (dict, list)):
            return str(x["error"])
        for k, v in x.items():
            if isinstance(v, str) and set(words(k)) & _O_ERROR:
                return v
        if len(x) == 1 and role != "message":
            v = next(iter(x.values()))
            if isinstance(v, str):
                return v
        return None

    def _classify(self, obs, x, role):
        if x is None or x == {} or x == []:
            obs.kind = "empty"
            return
        if isinstance(x, str):
            obs.kind, obs.text = ("reply" if role == "message" else "other"), x
            return
        if isinstance(x, list):
            if role in ("search", "browse", "compare"):
                self._learn_roles(x)   # R5-B1
            rows = [p for p in (self.product(v) for v in x if isinstance(v, dict)) if p]
            if rows:
                obs.kind, obs.products = "listing", rows
            return
        if not isinstance(x, dict):
            obs.kind, obs.text = "other", str(x)
            return
        error = self._error_of(x, role)
        if error is not None:
            obs.kind, obs.error = "error", error
            return
        if role == "message":
            texts = [v for v in x.values() if isinstance(v, str)]
            if texts:
                obs.kind, obs.text = "reply", max(texts, key=len)
                return
        list_key, rows = None, []
        for k, v in x.items():
            if isinstance(v, list):
                if role in ("search", "browse", "compare"):
                    self._learn_roles(v)   # R5-B1
                parsed = [p for p in (self.product(i) for i in v if isinstance(i, dict)) if p]
                if len(parsed) > len(rows) or (list_key is None and v == []):
                    list_key, rows = k, parsed
        if list_key is not None:
            items = [i for i in x[list_key] if isinstance(i, dict)]
            line_like = any(self._qty_of(i) is not None for i in items) or (
                role in ("cart_add", "cart_view") and bool(rows))
            scalars = {k: v for k, v in x.items() if k != list_key and not isinstance(v, (dict, list))}
            if line_like or (role in ("cart_add", "cart_view") and not items):
                obs.kind = "cart"
                obs.cart = {"lines": [self._line(i) for i in items], "total": self._total(scalars)}
                return
            obs.kind, obs.products = "listing", rows
            self._paging(obs, scalars, self.opaque_world)   # R5-B2
            return
        product = self.product(x)
        if product is not None:
            if self._qty_of(x) is not None and role in ("cart_add", "cart_view"):
                obs.kind, obs.ok = "ack", True
                obs.cart = {"lines": [self._line(x)], "total": None, "partial": True}
                return
            obs.kind, obs.products = "product", [product]
            return
        values = list(x.values())
        if values and all(isinstance(v, bool) for v in values):
            obs.kind, obs.ok = "ack", all(values)
            return
        total = self._total(x)
        if total is not None and role in ("cart_view", "cart_add"):
            obs.kind, obs.cart = "cart", {"lines": [], "total": total}
            return
        obs.kind = "other"

    @staticmethod
    def _paging(obs, scalars, opaque_world=False):
        candidates = []
        for k, v in scalars.items():
            w = set(words(k))
            if w & (_O_NEXT | _PAGE_STRONG):   # R5-S18: a paging word marks the next token, even beside a total word
                candidates.append(v)
            elif isinstance(v, int) and not isinstance(v, bool) and w & _O_TOTAL:
                obs.total = v
            elif v is None or isinstance(v, str):
                candidates.append(v)
        obs.next = next((v for v in candidates if v not in (None, "", False)), None)
        ints = sorted(v for k, v in scalars.items() if isinstance(v, int) and not isinstance(v, bool))
        if (opaque_world and obs.next is None and len(ints) == 2 and ints[0] > 0
                and all(WorldAdapter._opaque(k) and not set(words(k)) & (_O_NEXT | _O_TOTAL) for k in scalars)):
            obs.next = ints[0]   # R5-B2: opaque scalars: the smaller integer is the next page or offset, the larger the total
            if obs.total is None:
                obs.total = ints[1]
        spare = [v for k, v in scalars.items() if isinstance(v, int) and not isinstance(v, bool) and not set(words(k)) & _O_NEXT]
        if not opaque_world and obs.total is None and len(spare) == 1 and spare[0] >= len(obs.products):
            obs.total = spare[0]   # R5-S10: the one count beside the rows that covers them all is the number of matches
        elif (not opaque_world and obs.total is None and obs.next is None and len(spare) == 2 and min(spare) > 0
                and max(spare) >= len(obs.products)):
            obs.next, obs.total = min(spare), max(spare)   # R5-S14: two unnamed counts: the next page, and the matches (as B2)

    @staticmethod
    def _qty_of(d):
        for k, v in d.items():
            w = set(words(k))
            if isinstance(v, int) and not isinstance(v, bool) and w & _F_QTY and not w & _F_PRICE:
                return v
        return None

    def _line(self, d):
        """One cart line. With opaque keys the quantity is the small integer and the price the
        other number, so a quantity is never read as a price."""
        p = self.product(d) or {}
        qty, price = self._qty_of(d), p.get("price")
        numbers = [(k, v) for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        priced = sorted((v, k) for k, v in numbers if set(words(k)) & _F_PRICE)
        if (qty is None and len(priced) >= 2 and isinstance(priced[0][0], int) and 0 <= priced[0][0] <= 50
                and priced[-1][0] > priced[0][0]):
            qty, price = priced[0][0], priced[-1][0]   # R5-S17: several money-like numbers: the small count is the quantity
        if qty is None and numbers:
            named_price = [k for k, v in numbers if set(words(k)) & _F_PRICE]
            others = [(k, v) for k, v in numbers if k not in named_price]
            ints = [(k, v) for k, v in others if isinstance(v, int) and 0 <= v <= 50]
            if ints:
                qkey, qty = min(ints, key=lambda kv: kv[1])
                rest = [v for k, v in numbers if k != qkey]
                price = d[named_price[0]] if named_price else (rest[0] if rest else None)
            elif len(numbers) == 1:
                qty = 1
        if qty is None and price is not None:   # B1: every number carries a price word; the count is the other small one
            small = [v for k, v in numbers if isinstance(v, int) and 0 <= v <= 50 and v != price]
            if len(small) == 1:
                qty = small[0]
        return {"id": p.get("id"), "qty": qty, "price": price, "title": clip(p.get("title") or "", 90)}

    @staticmethod
    def _total(scalars):
        best = None
        for k, v in scalars.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and set(words(k)) & _O_CART_TOTAL:
                best = v
        return best

    @staticmethod
    def _props_from_list(v):
        """Property tables written as [[name, value], ...] or [{"property": .., "value": ..}, ...]."""
        props = {}
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
                props[item[0]] = item[1] if isinstance(item[1], (str, int, float, bool)) else json.dumps(item[1])
            elif isinstance(item, dict) and len(item) >= 2:
                keys = list(item)
                kname = next((k for k in keys if set(words(k)) & {"property", "name", "key", "attribute", "label", "field", "spec"}), None)
                vname = next((k for k in keys if set(words(k)) & {"value", "val", "detail", "setting", "content"}), None)
                if kname is None or vname is None:
                    strs = [k for k in keys if isinstance(item[k], str)]
                    if len(keys) != 2 or len(strs) < 1:
                        return None
                    kname, vname = keys[0], keys[1]
                if not isinstance(item[kname], str):
                    return None
                props[item[kname]] = item[vname] if isinstance(item[vname], (str, int, float, bool)) else json.dumps(item[vname])
            else:
                return None
        return props

    @staticmethod
    def _opaque(key) -> bool:
        """R5-B1: a field name that carries none of the role words the adapter reads."""
        return not set(words(key)) & (_F_TITLE | _F_CATEGORY | _F_BRAND | _F_SELLER | _F_STOCK | _F_STOCK_NOT
                                       | _F_PRICE | _F_QTY | _F_RATING | _F_ID)

    def _learn_roles(self, items):
        """R5-B1: roles of opaque fields from their values across a listing's rows: the 0/1 integer is the stock flag,
        the number with the most distinct values the price; of the phrases of 8+ characters the most repeated names the
        aisle, the shortest of the rest the item and longer ones its description; the most repeated short label names
        the seller."""
        rows = [i for i in items if isinstance(i, dict)]
        if len(rows) < 2 or any(not self._opaque(k) for r in rows for k in r):
            return   # a world that names any of its fields is bound by those names
        self.opaque_world = True
        cols = {}
        for r in rows:
            for k, v in r.items():
                if self._opaque(k) and v is not None:
                    cols.setdefault(k, []).append(v)
        nums = {k: vs for k, vs in cols.items() if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vs)}
        strs = {k: vs for k, vs in cols.items() if all(isinstance(v, str) and v.strip() for v in vs)
                and not any(looks_like_id(v) for v in vs)
                and not all(re.sub(r"[\s_\-]+", "", v.strip().lower()) in _STOCK_WORDS for v in vs)}   # R5-B8: a stock-word column is the stock flag
        flags = [k for k, vs in nums.items() if set(vs) <= {0, 1}]
        rest = [k for k in nums if k not in flags]
        if rest:
            self.field_roles[max(rest, key=lambda k: (len(set(nums[k])), k))] = "price"
            for k in flags:
                self.field_roles[k] = "stock"
        if not strs:
            return
        distinct = {k: len(set(vs)) / len(vs) for k, vs in strs.items()}
        mean = {k: sum(len(v) for v in vs) / len(vs) for k, vs in strs.items()}
        long_ = sorted((k for k in strs if mean[k] >= 8), key=lambda k: (distinct[k], mean[k], k))
        if len(long_) >= 2 and (distinct[long_[0]] <= 0.5 or distinct[long_[0]] < distinct[long_[1]]):
            self.field_roles[long_[0]] = "category"   # the most repeated phrase: the aisle the rows share
            long_ = long_[1:]
        for i, k in enumerate(sorted(long_, key=lambda k: (mean[k], k))):
            self.field_roles[k] = "title" if i == 0 else "text"   # the shortest varied phrase names the item
        short = sorted((k for k in strs if mean[k] < 8), key=lambda k: (distinct[k], k))
        if short:
            self.field_roles[short[0]] = "seller"

    def product(self, d):
        """Normalise one product-like dict; None when it carries no product id."""
        if not isinstance(d, dict):
            return None
        out = {"id": None, "title": None, "price": None, "in_stock": None, "category": None,
               "brand": None, "seller": None, "rating": None, "text": "", "props": {}}
        texts, stock_guess, price_extra = [], None, None
        if self.opaque_world:   # R5-B5: renamed fields no listing has shown yet: the lone 0/1 integer is the stock flag
            unknown = [k for k, v in d.items() if k not in self.field_roles and self._opaque(k)
                       and isinstance(v, (int, float)) and not isinstance(v, bool)]
            flags = [k for k in unknown if isinstance(d[k], int) and d[k] in (0, 1)]
            if len(unknown) == 2 and len(flags) == 1:
                self.field_roles[flags[0]] = "stock"
                self.field_roles[next(k for k in unknown if k != flags[0])] = "price"
        for k, v in d.items():
            role = self.field_roles.get(k)   # R5-B1: an opaque field whose role the listing rows showed
            if role == "price" and isinstance(v, (int, float)) and not isinstance(v, bool):
                out["price"] = v
                continue
            if role == "stock" and isinstance(v, (int, float, bool)):
                out["in_stock"] = (not v) if (isinstance(v, bool) and self.stock_inverted) else bool(v)
                continue
            if role in ("title", "category", "seller") and isinstance(v, str) and out[role] is None:
                out[role] = v
                continue
            if role == "text" and isinstance(v, str):
                texts.append(v)
                continue
            w = set(words(k))
            if isinstance(v, bool):
                if w & _F_STOCK_NOT:
                    out["in_stock"] = not v
                elif w & _F_STOCK:
                    out["in_stock"] = (not v) if self.stock_inverted else v
                elif stock_guess is None:
                    stock_guess = (not v) if self.stock_inverted else v
            elif isinstance(v, str):
                low = v.strip().lower()
                key = re.sub(r"[\s_\-]+", "", low)
                if key in _STOCK_WORDS and (w & (_F_STOCK | _F_STOCK_NOT) or "_" in low or " " in low or low in ("available", "unavailable")):
                    state = _STOCK_WORDS[key]
                    out["in_stock"] = (not state) if w & _F_STOCK_NOT and low in ("yes", "no", "true", "false") else state
                elif out["id"] is None and (looks_like_id(v) or (w & _F_ID and len(v) <= 64 and " " not in v)):
                    out["id"] = v
                elif w & _F_CATEGORY and out["category"] is None:  # before title: "aisle_name"
                    out["category"] = v
                elif w & _F_BRAND and out["brand"] is None:
                    out["brand"] = v
                elif w & _F_SELLER and out["seller"] is None:
                    out["seller"] = v
                elif w & _F_TITLE and out["title"] is None:
                    out["title"] = v
                elif v.strip():
                    texts.append(v)
            elif isinstance(v, (int, float)):
                if w & _F_RATING:
                    out["rating"] = v
                elif w & (_F_STOCK | _F_STOCK_NOT) and isinstance(v, int):
                    out["in_stock"] = (v <= 0) if w & _F_STOCK_NOT else (v > 0)
                elif w & _F_ID and not w & _F_PRICE and out["id"] is None:
                    out["id"] = str(v)
                elif w & _F_PRICE or (out["price"] is None and not w & _F_QTY):
                    extra = len(w - _F_PRICE)   # B3: "amount" before "amount_total": the plainest price word wins
                    if out["price"] is not None and price_extra is not None and extra >= price_extra:
                        continue
                    out["price"], price_extra = v, extra
                    if {"cents", "cent"} & w:
                        self.money_unit = "cents"
            elif isinstance(v, dict):
                for pk, pv in v.items():
                    if isinstance(pv, (str, int, float, bool)):
                        out["props"][str(pk)] = pv
                    elif isinstance(pv, list):
                        out["props"][str(pk)] = ", ".join(str(i) for i in pv)
            elif isinstance(v, list) and v:
                table = self._props_from_list(v)
                if table:
                    out["props"].update(table)
                elif all(isinstance(i, str) for i in v):
                    texts.append("; ".join(v))
        if out["id"] is None:
            return None
        if out["in_stock"] is None:
            out["in_stock"] = stock_guess
        if out["title"] is None and texts:
            short = sorted((t for t in texts if len(t) >= 8), key=len)
            out["title"] = short[0] if short else texts[0]
            texts = [t for t in texts if t != out["title"]]
        out["text"] = " ".join(texts)
        return out


# ======================================================================================
# 4. EnvSession: the /environment/call protocol
# ======================================================================================

class _TimeUp(BaseException):
    """A-R4: past the hard limit minus its margin only a terminal call may go out. A BaseException, so that no
    `except Exception` guard on the way can swallow it."""


class EnvSession:
    """Step counter, envelopes, batching and replay-safe retries for one episode.

    The validator replays a step it already ran when the same step number arrives with the
    same actions, so resending after a lost response is safe."""

    def __init__(self, client, binding, max_steps, clock=time.monotonic, deadline_s=None):
        self.client = client
        self.binding = dict(binding or {})
        try:
            self.max_steps = int(max_steps or 30)
        except (TypeError, ValueError):     # C0055: an unreadable step budget falls back to the default
            self.max_steps = 30
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + (deadline_s if deadline_s is not None else CONFIG["deadline_s"])
        self.steps = 0
        self.finished = False
        self.failures = 0
        self.last_error = None

    def time_left(self) -> float:
        return self.deadline - self.clock()

    def past_guard(self) -> bool:
        """A-R4: past the hard limit minus its margin (300 - 25 s)."""
        return self.clock() - self.started >= CONFIG.get("hard_limit_s", 300.0) - CONFIG.get("hard_margin_s", 25.0)

    def _bound_client(self):
        """C0054: one attempt may wait at most half the time left before the hard limit; with little
        time left, one short attempt. The client makes one try per attempt: send() retries."""
        if not hasattr(self.client, "timeout"):
            return
        left = self.started + CONFIG.get("hard_limit_s", 300.0) - CONFIG.get("hard_margin_s", 25.0) - self.clock()
        if hasattr(self.client, "max_retries"):
            self.client.max_retries = 1     # retries happen in send(), with a jittered back-off
        self.client.timeout = max(2.0, min(CONFIG["env_timeout_s"], (left - 6) / 2 if left > 20 else left - 1))

    def _post(self, envelope):
        try:
            self._bound_client()
            verbose = getattr(self.client, "post_verbose", None)
            if verbose is not None:
                result = verbose("/environment/call", envelope)
                data = getattr(result, "data", None)
                return (data if isinstance(data, dict) else None), getattr(result, "error", None)
            data = self.client.post("/environment/call", envelope)
            return (data if isinstance(data, dict) else None), None
        except Exception as exc:  # noqa: BLE001 - a transport failure must not kill the episode
            return None, {"kind": "exception", "body": f"{type(exc).__name__}: {exc}"}

    def send(self, actions):
        """Run 1-16 (name, args) actions as consecutive steps. Returns the raw observations
        (fewer when the episode finished part-way), or None when the call failed."""
        room = self.max_steps - self.steps
        if self.finished or not actions or room <= 0:
            return None
        actions = list(actions)[: min(CONFIG["batch_max"], room)]
        items = [{"name": n, "args": a} for n, a in actions]
        envelope = dict(self.binding)
        envelope["step"] = self.steps + 1
        if len(items) == 1:
            envelope["action"] = items[0]
        else:
            envelope["actions"] = items
        data = None
        for attempt in range(CONFIG["env_attempts"]):
            data, error = self._post(envelope)
            if data is not None:
                break
            self.failures += 1
            text = json.dumps(error, default=str) if error else ""
            self.last_error = clip(text, 300)
            if "has finished" in text:
                self.finished = True
                return None
            expected = re.search(r"expected (\d+)", text)
            if expected and int(expected.group(1)) != envelope["step"]:
                self.steps = int(expected.group(1)) - 1
                envelope["step"] = self.steps + 1
                continue        # a corrected step number is retried at once
            if self.time_left() < 5:
                break
            base = min(CONFIG.get("env_backoff_cap_s", 8.0), CONFIG.get("env_backoff_s", 1.0) * 2 ** attempt)
            hard_left = self.started + CONFIG.get("hard_limit_s", 300.0) - CONFIG.get("hard_margin_s", 25.0) - self.clock()
            if attempt + 1 < CONFIG["env_attempts"] and hard_left > 1.5 * base + 20:
                time.sleep(base * (0.75 + 0.25 * (attempt % 2)))    # C0054: back off (alternating spread) before the next attempt
        if data is None:
            return None
        if isinstance(data.get("observations"), list):
            observations = data["observations"]
        else:
            observations = [data.get("observation")]
        step = data.get("step")
        self.steps = step if isinstance(step, int) and not isinstance(step, bool) else self.steps + len(observations)
        self.finished = bool(data.get("finished"))
        return observations


# ======================================================================================
# 5. ModelChain: model preferences with failure-triggered switching
# ======================================================================================

def detect_provider() -> str:
    """Mirror of the proxy's rule: an OpenRouter key starts with sk-or-; anything else is Chutes.
    Only the prefix is inspected and nothing is stored or printed."""
    named = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if named in CONFIG["models"]:
        return named
    return "openrouter" if (getenv("INFERENCE_ACCESS_TOKEN") or "").startswith("sk-or-") else "chutes"


def parse_json_loose(text):
    """The first JSON object or array in a model reply (code fences and chatter tolerated)."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"```(?:json)?", "", text)
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start: end + 1])
            except ValueError:
                continue
    return None


def _rate_limit_pause(error):
    """RL1: seconds to wait before retrying a rate-limited call (Retry-After, capped at 2 s); None when not rate-limited."""
    if not isinstance(error, dict):
        return None
    body = str(error.get("body") or "")
    if error.get("status") != 429 and not re.search(r"rate.?limit|too many requests", body, re.IGNORECASE):
        return None
    headers = error.get("headers") if isinstance(error.get("headers"), dict) else {}
    after = next((v for k, v in headers.items() if str(k).lower() == "retry-after"), None)
    m = re.search(r"retry[ _-]?after\D{0,5}(\d+(?:\.\d+)?)", body, re.IGNORECASE)
    try:
        wait = float(after if after is not None else (m.group(1) if m else 1.0))
    except (TypeError, ValueError):
        wait = 1.0
    return max(0.0, min(2.0, wait))


def _error_kind(error):
    """A-R: the status class of a failed model call, for the log (never the body)."""
    if not isinstance(error, dict):
        return "malformed"
    if error.get("kind") == "length":
        return "length"   # A-R4: cut off by the token limit
    status = error.get("status")
    if isinstance(status, int) and not isinstance(status, bool):
        if status in (401, 402, 403, 404, 408, 429):
            return str(status)
        return "5xx" if status >= 500 else ("4xx" if status >= 400 else "malformed")
    if re.search(r"time.?out|timed out", str(error.get("body") or ""), re.IGNORECASE):
        return "timeout"
    return "network" if error.get("kind") in ("network", "exception") else "malformed"


class ModelChain:
    """Calls /inference/chat/completions down a preference list. A model is benched after
    repeated failures or when the proxy says it is not allowed; a 403 that lists the allowed
    models re-targets the chain. chat() returns None when every model has failed, and the
    caller then takes the deterministic path."""

    _served = None  # provider model ids seen by discover(), cached per process

    def __init__(self, client, provider=None, clock=time.monotonic, deadline=None, prefs=None):
        self.client = client
        self.provider = provider or detect_provider()
        base = prefs or CONFIG["models"].get(self.provider) or CONFIG["models"]["chutes"]
        self.prefs = {purpose: list(models) for purpose, models in base.items()}
        forced = getenv("SANDBOX_MODEL")
        if forced:
            for purpose, models in self.prefs.items():
                self.prefs[purpose] = [forced] + [m for m in models if m != forced]
        self.clock = clock
        self.deadline = deadline
        self.streak = {}
        self.benched = set()
        self.cool = {}              # A-R: model -> clock time until which it is skipped (transient failures)
        self.spent_s = 0.0          # A-R: model wall time this episode
        self.attempts = 0           # A-R: HTTP attempts (calls counts logical calls)
        self.judge_calls = 0
        self.allowed = None
        self.calls = 0
        self.failures = 0
        self.cost = 0.0
        self.log = []               # one entry per call: purpose, model, ok, seconds, cost
        self.last_error = None

    def spent(self) -> bool:
        """True once the per-episode call or cost cap is reached."""
        return (self.calls >= CONFIG["max_calls"] or self.cost >= CONFIG["max_cost_usd"]
                or self.attempts >= 2 * CONFIG["max_calls"])

    def usable(self, purpose="actor") -> list:
        models = self.prefs.get(purpose) or self.prefs.get("actor") or []
        now = self.clock()
        return [m for m in models if m not in self.benched and (self.allowed is None or m in self.allowed)
                and self.cool.get(m, 0.0) <= now]   # A-R: a cooled model comes back after its cooldown

    def alive(self, purpose="actor") -> bool:
        return bool(self.usable(purpose)) and not self.spent()

    def discover(self, client=None):
        """Move models the provider does not list to the back (one GET per process). The
        listing is the provider's catalogue, a superset of ORO's allowlist; the allowlist itself
        is learnt from a 403 (see _learn)."""
        if not CONFIG["discover_models"]:
            return
        if ModelChain._served is None:
            ModelChain._served = set()
            try:
                data = (client or self.client).get("/inference/models")
                rows = data.get("data") if isinstance(data, dict) else data
                for row in rows or []:
                    ident = row.get("id") if isinstance(row, dict) else row
                    if isinstance(ident, str):
                        ModelChain._served.add(ident)
            except Exception:  # noqa: BLE001 - discovery is optional
                pass
        served = ModelChain._served
        if served and not (getenv("INFERENCE_PROVIDER") or "").strip():
            # no provider named: take the chain whose models this proxy actually lists
            counts = {p: sum(m in served for ms in chains.values() for m in ms) for p, chains in CONFIG["models"].items()}
            best = max(counts, key=counts.get)
            if counts[best] > 0 and best != self.provider:
                self.provider = best
                self.prefs = {purpose: list(models) for purpose, models in CONFIG["models"][best].items()}
        if served:
            for purpose, models in self.prefs.items():
                self.prefs[purpose] = [m for m in models if m in served] + [m for m in models if m not in served]

    def _post(self, payload, timeout=None):
        try:
            if hasattr(self.client, "timeout"):   # set on every call, so the read call's longer wait never sticks
                left = ((self.deadline - self.clock() - CONFIG["llm_deadline_margin_s"]) if self.deadline is not None
                        else float("inf"))   # A-R3: never wait into the deadline's margin
                left = min(left, CONFIG["llm_time_budget_s"] - self.spent_s)   # A-R: nor past the episode's model time
                self.client.timeout = max(3, min(timeout or CONFIG["llm_timeout_s"], left))   # R1: the read call may wait longer
            box = {}

            def attempt():
                try:
                    verbose = getattr(self.client, "post_verbose", None)
                    if verbose is not None:
                        result = verbose("/inference/chat/completions", payload)
                        box["out"] = (getattr(result, "data", None), getattr(result, "error", None))
                    else:
                        box["out"] = (self.client.post("/inference/chat/completions", payload), None)
                except Exception as exc:  # noqa: BLE001
                    box["out"] = (None, {"kind": "exception", "body": f"{type(exc).__name__}: {exc}"})
            wall = getattr(self.client, "timeout", None)
            if not isinstance(wall, (int, float)) or isinstance(wall, bool):
                attempt()
                return box["out"]
            worker = threading.Thread(target=attempt, daemon=True)
            worker.start()
            worker.join(wall + CONFIG["llm_wall_slack_s"])
            if worker.is_alive() or "out" not in box:   # A-R4: a reply still dribbling in is abandoned
                return None, {"kind": "network", "status": None, "body": "wall-clock limit reached (timeout)"}
            return box["out"]
        except Exception as exc:  # noqa: BLE001
            return None, {"kind": "exception", "body": f"{type(exc).__name__}: {exc}"}

    def _learn(self, model, error):
        """True when the model is out for this call: benched (401/402: no key or credit; 403 / not allowed; 404 / model
        not found) or, A-R, cooled down after llm_bench_after straight transient failures (probed again later)."""
        body = str((error or {}).get("body") or "")
        status = (error or {}).get("status")
        if status in (401, 402) or re.search(r"exhausted|insufficient (credits|funds)", body, re.IGNORECASE):
            self.benched.update(m for models in self.prefs.values() for m in models)  # no key or no credit
            return True
        if status == 403 or "not allowed" in body:
            self.benched.add(model)
            tail = body.split("allowed_models", 1)[1] if "allowed_models" in body else ""
            allowed = re.findall(r'"([^"\s]+/[^"\s]+)"', tail)
            if allowed:
                self.allowed = set(allowed)
                for purpose, models in self.prefs.items():
                    if not [m for m in models if m in self.allowed]:
                        self.prefs[purpose] = models + allowed[:3]
            return True
        if status == 404 or re.search(r"no such model|unknown model|invalid model|model\W.{0,60}(?:not found|does not exist)",
                                      body, re.IGNORECASE):
            self.benched.add(model)
            return True
        self.streak[model] = self.streak.get(model, 0) + 1
        if self.streak[model] >= CONFIG["llm_bench_after"]:
            self.streak[model] = 0
            self.cool[model] = self.clock() + CONFIG["llm_cooldown_s"]
            return True
        return False

    def chat(self, messages, purpose="actor", tools=None, max_tokens=1200, temperature=0, need_json=False):
        """One assistant message (dict, with '_model' added) or None when every model failed. A-R: a transient error is
        retried on the same model (llm_attempts, short back-off), then the next model is tried; a logical call counts
        once against the call and judge budgets."""
        if purpose == "judge":
            if self.judge_calls >= CONFIG.get("assist_max_judge_calls", 2):
                return None     # C0054: the judge budget (now per logical call; A-R)
            self.judge_calls += 1
        max_tokens = max(max_tokens, CONFIG.get("llm_min_max_tokens", 0))   # A-R4: room for reasoning plus the answer
        tried, tries, counted = set(), {}, False
        while True:
            fresh = [m for m in self.usable(purpose) if m not in tried]  # re-read: a 403 may re-target
            if not fresh:
                return None
            model = fresh[0]
            if self.deadline is not None and self.deadline - self.clock() < CONFIG["llm_deadline_reserve_s"]:
                return None     # A-R3: near the deadline the env steps keep the time
            if CONFIG["llm_time_budget_s"] - self.spent_s < CONFIG["llm_min_time_s"]:
                return None     # A-R: the episode's model time is used up; the rules go on
            if self.spent():
                return None
            p_in, p_out = ((CONFIG.get("model_price_usd") or {}).get(model)
                           or (CONFIG.get("token_price_usd") or {}).get(purpose, [0.0, 0.0]))
            worst = len(json.dumps(messages, default=str)) / 4 * p_in + max_tokens * p_out   # A-R4: this call at full length
            if self.cost + max(worst, (CONFIG.get("cost_guard_usd") or {}).get(purpose, 0.0)) > CONFIG["max_cost_usd"]:
                return None     # R1: spend guard: a call that could cross the episode cap is not made (rules go on)
            payload = {"model": model, "messages": messages, "temperature": temperature,
                       "max_tokens": max_tokens}
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            started = self.clock()
            data, error = self._post(payload, CONFIG.get("read_timeout_s") if purpose == "read" and not tried else None)
            seconds = self.clock() - started
            self.spent_s += seconds
            self.attempts += 1
            if not counted:
                self.calls += 1
                counted = True
            usage = data.get("usage") if isinstance(data, dict) else None
            p_in, p_out = ((CONFIG.get("model_price_usd") or {}).get(model)
                           or (CONFIG.get("token_price_usd") or {}).get(purpose, [0.0, 0.0]))
            if isinstance(usage, dict):   # C1: fixed token prices only, never a billed cost (F-010)
                cost = (usage.get("prompt_tokens") or 0) * p_in + (usage.get("completion_tokens") or 0) * p_out
            else:   # C1: a reply without usage is priced from its prompt size and token limit, never as free
                cost = len(json.dumps(messages, default=str)) / 4 * p_in + max_tokens * p_out if data else 0.0
            self.cost += cost
            message, finish = None, None
            if isinstance(data, dict) and isinstance(data.get("choices"), list) and data["choices"]:
                first = data["choices"][0]
                message = first.get("message") if isinstance(first, dict) else None
                finish = first.get("finish_reason") if isinstance(first, dict) else None
            if isinstance(message, dict) and finish == "length" and not message.get("tool_calls") and (
                    not str(message.get("content") or "").strip()
                    or (need_json and parse_json_loose(message.get("content")) is None)):
                message, error = None, {"kind": "length", "status": None, "body": "finish_reason length: reply cut off"}
            details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
            entry = {"purpose": purpose, "model": model, "ok": isinstance(message, dict),
                     "seconds": round(seconds, 2), "cost": round(cost, 6),
                     "tokens": [usage.get("prompt_tokens"), usage.get("completion_tokens")] if isinstance(usage, dict) else None,
                     "finish": finish, "reasoning": details.get("reasoning_tokens") if isinstance(details, dict) else None}
            n = tries.get(model, 0)
            if n:
                entry["try"] = n
            if isinstance(message, dict):
                self.log.append(entry)
                self.streak[model] = 0
                self.cool.pop(model, None)
                message = dict(message)
                message["_model"] = model
                return message
            kind = entry["err"] = _error_kind(error)   # A-R: the status class, so a race can be diagnosed
            self.log.append(entry)
            self.failures += 1
            self.last_error = clip(json.dumps(error, default=str), 300) if error else "malformed reply"
            tries[model] = n + 1
            if self._learn(model, error):
                tried.add(model)    # benched, or cooled down: the next model
                continue
            if tries[model] < CONFIG["llm_attempts"] and kind not in ("timeout", "408", "length"):   # the next model
                pause = _rate_limit_pause(error)   # RL1: a 429's own wait
                wait = pause if pause is not None else CONFIG["llm_backoff_s"] * 2 ** n
                if ((self.deadline is None or self.deadline - self.clock() > CONFIG["llm_deadline_reserve_s"] + wait)
                        and self.spent_s + wait <= CONFIG["llm_time_budget_s"] - CONFIG["llm_min_time_s"]):   # A-R4 W2
                    self.spent_s += wait   # the waits count against the model time budget too
                    time.sleep(wait)
                    continue
            tried.add(model)

    def ask_json(self, system, user, purpose="judge", max_tokens=700):
        """A JSON object or list from the model, or None."""
        reply = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                          purpose=purpose, max_tokens=max_tokens, need_json=True)
        return parse_json_loose(reply.get("content")) if reply else None


# ======================================================================================
# 6. REASONING LAYER. Deterministic reasoning on the Bench language parsers (6a). The
#    signatures of class Reasoning are the contract; LLM fallbacks plug in behind them.
# ======================================================================================

class Need:
    """One requirement on the product. `prop` is the attribute (None = anywhere in the facts),
    `op` is eq | in | ge | le | gt | lt | between | ne | contains | not_contains | avoid | ref
    (a pending reference to another listing)."""

    def __init__(self, prop, value, source="request", hard=True, op="eq", scope=None, line=None):
        self.prop, self.value, self.source, self.hard, self.op = prop, value, source, hard, op
        self.scope = scope      # C0017: the product phrase the shopper tied it to ("internal hard drives")
        self.line = line        # C0017: index of the order line it belongs to (None = every line / single line)

    def as_dict(self) -> dict:
        d = {"prop": self.prop, "op": self.op, "value": self.value, "source": self.source, "hard": self.hard}
        if self.scope or self.line is not None:
            d.update(scope=self.scope, line=self.line)
        return d


class Verdict:
    """Judgement of one product: ok is True (meets every need), False (violates one) or None
    (not enough evidence). `missing` names the needs that were not confirmed."""

    def __init__(self, ok, missing=None, why=""):
        self.ok, self.missing, self.why = ok, list(missing or []), why


class Ledger:
    """Everything known about what the shopper wants. Owned by the reasoning layer; the core
    reads budget, quantity, msg_budget, step_budget, prefer, needs, product, category and
    notes["refs"] (pending references to resolve)."""

    def __init__(self, query, opening=None):
        self.query = query or ""
        self.opening = opening
        self.budget = None          # in CENTS (None = no budget stated)
        self.budget_unit = None     # "cents" | "dollars": how the request wrote it
        self.quantity = 1           # units of the one product to order
        self.msg_budget = None      # messages allowed (None = not stated)
        self.step_budget = None     # steps allowed by the request (None = not stated)
        self.prefer = "cheapest"    # "cheapest" | "best"
        self.stock_only = False     # only items on the shelf (never back-ordered)
        self.decline_ok = True      # the request allows "nothing found"
        self.product = None         # the product phrase ("hoodie or sweatshirt")
        self.terms = []             # product words for search
        self.category = None        # a category phrase from the request, if any
        self.needs = []             # list of Need
        self.asked = []             # properties already asked about
        self.heard = []             # (source, text) shopper lines in arrival order
        self.notes = {}             # free-form state (refs, open, unknown)
        self.lines = []             # multi-line orders: [{"product", "quantity", "needs", "category"}]

    def summary(self) -> dict:
        return {"budget_cents": self.budget, "quantity": self.quantity, "prefer": self.prefer,
                "messages_allowed": self.msg_budget, "steps_allowed": self.step_budget,
                "product": self.product or " ".join(self.terms), "category": self.category,
                "needs": [n.as_dict() for n in self.needs], "asked": list(self.asked)}


_STOP = set("""a an the and or but for with without from into onto of to in on at by is are be
been being it its this that these those i me my we our you your he she they them their his her
im i'm ive i've id i'd ill i'll just so very really some any one ones something thing things
please thanks need needs want wants looking searching shopping get got find buy
order pick choose whichever whatever which what who whom where when why how can could would
should will shall may might must also too as if then than there here about around over under
up down out off again more most less least only own same such no nor not now all both each few
other""".split())
_BUDGET_CUE = re.compile(r"(budget|over|under|up to|at most|max|maximum|(?:no|not) more|spend|cap|"
                         r"ceiling|limit|afford|less than|price)", re.IGNORECASE)


def _count_near(text, nouns):
    """The largest count stated before one of `nouns` (the count next to the noun, not a nearer unrelated one)."""
    found = [parse_count(m.group(1)) for m in re.finditer(
        r"\b(\d+|" + "|".join(k for k in _NUMBER_WORDS if len(k) > 2) + r")[\s\-]+(?:[a-z\-]+\s+){0,2}?(?:"
        + nouns + r")\b", text, re.IGNORECASE)]
    found = [f for f in found if f]
    return max(found) if found else None


def _tokens(text) -> set:
    return {w for w in words(text) if w not in _STOP and len(w) > 1}


# ======================================================================================
# 6a. BENCH LANGUAGE: requests, shopper answers, unprompted events, listing redirects.
#     Deterministic parsers, tested on real race strings (tests/fixtures). The reasoning
#     layer may add an LLM fallback for whatever these return as "unknown".
# ======================================================================================

_LANG_STOP = {"a", "an", "the", "of", "on", "in", "to", "it", "its", "for", "and", "or", "with",
              "that", "this", "one", "some", "any", "my", "me", "i", "is", "be", "tag", "label",
              "listing", "please", "just", "only", "really", "kind", "sort"}


def stem(token) -> str:
    """A light plural fold: heels->heel, cans->can, boxes->box (never shortens below 3)."""
    t = token.lower()
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith(("ches", "shes", "xes", "sses")):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def toks(text) -> list:
    """Normalised content tokens (numbers kept, punctuation dropped, plurals folded)."""
    text = re.sub(r"(\d)\s*[x\u00d7]\s*(\d)", r"\1x\2", str(text or "").lower().replace("'", ""))
    raw = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", text)
    return [stem(t) for t in raw if t not in _LANG_STOP]


_GE = re.compile(r"\b(?:at least|minimum(?: of)?|(?:no|not) less than|min\.?)\s+(.+)|(.+?)\s+(?:or (?:more|bigger|larger|higher|greater|above|longer|over)|and (?:up|above|over)|\+)$")
_LE = re.compile(r"\b(?:at most|up to|maximum(?: of)?|(?:no|not) more than|under|below|less than|max\.?)\s+(.+)|(.+?)\s+(?:or (?:less|smaller|lower|under|below|shorter|fewer))$")


def value_op(text):
    """('ge'|'le'|'in'|'eq', value) for a stated value, e.g. 'at least <n> <unit>',
    '<n> <unit> or bigger', '<a> or <b>'."""
    text = str(text or "").strip().strip(" .,;")
    m = _GE.search(text)
    if m and re.search(r"\d", m.group(1) or m.group(2) or ""):
        return "ge", (m.group(1) or m.group(2)).strip()
    m = _LE.search(text)
    if m and re.search(r"\d", m.group(1) or m.group(2) or ""):
        return "le", (m.group(1) or m.group(2)).strip()
    if text.startswith("[") and text.endswith("]"):
        text = re.sub(r"['\"\[\]]", "", text).replace(",", "|")
    if "|" in text:
        return "in", [p.strip() for p in text.split("|") if p.strip()]
    if (re.search(r"\s(?:or|/)\s", text) and len(text.split()) <= 8
            and not re.search(r"\bor (?:so|thereabouts|similar|close|more|less|bigger|smaller|larger|higher|lower|"
                              r"above|below|greater|fewer|longer|shorter|over|under)\b", text)):
        return "in", [p.strip() for p in re.split(r"\s(?:or|/)\s|,\s*", text) if p.strip()]
    return "eq", text


# >>>>>>>> C0017 claude-2: reference parsing (BEGIN) >>>>>>>>
# ---- references to another listing (C0017/C0028: hops, selectors, pairing rules, disclaimers) ----
# Structural detection only: a reference is a listing mention next to a capitalised seller name
# ("listing ... sold by <Name>", "... from <Name>", "<Name>'s ..."); its qualifiers are SELECTORS.

_REF = re.compile(r"\blisting\b(?:\s+(?:for|of))?\s+(?:the |a |an )?(?P<item>[^,;:.]{1,80}?)\s+(?:sold by|from|by)\s+"
                  r"(?P<seller>[A-Z][\w'\-]+)(?P<rest>(?:[^.;]|\.(?=\d))*)")


def _quals_of(rest):
    """The qualifier clauses after a seller name (", with A, B and C, ..."): every clause up to the
    first one that talks about listings again (a note on how to follow the chain)."""
    rest = re.sub(r"^\s*,?\s*(?:with|having|that has|in)\s+", "", rest or "", count=1)
    if rest == "" or rest[:1] == ",":
        return ""
    keep = []
    for clause in re.split(r",\s*", rest):
        if re.search(r"\blisting", clause, re.IGNORECASE):
            break
        keep.append(clause)
    return ", ".join(keep).strip(" ,.")


_REF_PAGE = re.compile(r"\b(?:page|listing|entry|post)\s+(?:for|of|on)\s+(?:the |a |an )?(?P<item>[^,;:.]{1,80}?)\s+(?:that|which)\s+"
                       r"(?P<seller>[A-Z][\w'\-]+)\s+(?:has |have |had )?(?:listed|lists|sells|sold|offers|offered|posted|carries|stocks)\b"
                       r"(?P<rest>(?:[^.;]|\.(?=\d))*)")   # H1: "<page> for the <item> that <Seller> has listed, with ..."
_REF_FROM = re.compile(r"\b(?:page|entry|post)\s+(?:for|of|on)\s+(?:the |a |an )?(?P<item>[^;:.]{1,80}?)\s+"
                       r"(?:from|by|sold by|listed by)\s+(?P<seller>[A-Z][\w'\-]+)(?P<rest>(?:[^.;]|\.(?=\d))*)")   # claude-3-18: "<page> of <item> from <Seller>, with ..."
_REF_NEG = re.compile(r"\b(?:anything|everything|whatever)\s+(?:bar|but|besides|except|other than|apart from|save)\b|"
                      r"\b(?:different from|unlike|other than|not (?:the )?same as)\b", re.IGNORECASE)
_REF_POS = re.compile(r"\b(?:same as|identical to|just like|exactly like|equal to)\b", re.IGNORECASE)


def parse_reference(text):
    """{'item', 'seller', 'quals', 'neg', 'at'} when `text` points at another listing, else None. `neg`: the last
    polarity cue in the sentence before the reference excludes that listing's value (H1); `at`: where the
    reference (or its polarity cue) starts."""
    t = text or ""
    m = _REF.search(t)
    page = _REF_PAGE.search(t) or _REF_FROM.search(t)
    if page and (not m or page.start() < m.start()):
        m = page
    if not m:
        return None
    item = re.sub(r"^(?:the |a |an )", "", m.group("item").strip(), flags=re.IGNORECASE)
    before = re.split(r"[.!?;]\s", t[:m.start()])[-1]
    neg = [x for x in _REF_NEG.finditer(before)]
    pos = [x for x in _REF_POS.finditer(before)]
    last = max(neg + pos, key=lambda x: x.end(), default=None)
    at = m.start() - len(before) + last.start() if last is not None else m.start()
    return {"item": item, "seller": m.group("seller").strip(), "quals": _quals_of(m.group("rest")),
            "neg": bool(last is not None and last in neg), "at": at, "page": m is page}


_AUX = r"(?:has|have|must|should|needs?|is|are|'s|to be|be|being|matches|match|matching)"
_ATTR_FOR = re.compile(r"\bfor (?:the |its |their )?(?P<attr>[a-z][a-z0-9 /&\-]{0,40}?)(?: of (?:the |my )?(?P<scope>[a-z][a-z' \-]{1,40}?))?,\s*"
                       r"(?:it (?:has to|must|should) be |(?:it's|it is) |go for |pick |take )?[^,;]*$", re.IGNORECASE)
_SCOPE_LEAD = re.compile(r"(?:\d+|one|two|three|a|an|some)\s+(?P<scope>[a-z][a-z' \-]{1,40}?)\s+(?:whose|with|where)\s*$", re.IGNORECASE)


def _attr_before(before):
    """(attr, scope, start) of the property a request clause ties to a reference: the noun phrase that
    opens the clause, up to its first auxiliary verb ("<attr> of the <scope> <verb> ...")."""
    m = _ATTR_FOR.search(before)
    if m:
        return m.group("attr"), m.group("scope"), m.start("attr")
    cut = max(before.rfind(","), before.rfind(";"), before.rfind("."))
    for w in (" whose ", " with ", " where ", " and ", " that "):
        cut = max(cut, before.lower().rfind(w) + len(w) - 1 if w in before.lower() else -1)
    clause = before[cut + 1:]
    m = re.match(r"\s*(?:the |its |their )?(?P<attr>[a-z][a-z0-9 /&\-]{0,40}?)(?: of (?:the |my )?(?P<scope>[a-z][a-z' \-]{1,40}?))?\s+"
                 + _AUX + r"\b", clause, re.IGNORECASE)
    if not m:
        return None, None, None
    return m.group("attr"), m.group("scope"), cut + 1 + m.start("attr")


def parse_request_refs(text) -> list:
    """References stated in a request: [{'attr', 'scope', 'ref'}], one per listing mention with a
    seller. The property is the noun phrase opening that clause; `scope` names the order line when
    the request gives one (a count and a product phrase before "whose")."""
    out = []
    t = " ".join(str(text or "").split())
    if CONFIG.get("rq_page"):   # RQ: listing-form and page-form mentions in order; polarity from the sentence before each
        found, prev = [], 0
        for m in sorted(list(_REF.finditer(t)) + list(_REF_PAGE.finditer(t)), key=lambda x: x.start()):
            if m.start() < prev:
                continue   # one mention matched by both forms
            sent = max([prev] + [t.rfind(p, 0, m.start()) + len(p) for p in (". ", "; ", "! ", "? ") if t.rfind(p, 0, m.start()) >= 0])
            ref = parse_reference(t[sent:m.end()])
            if ref:
                found.append((m, ref, sent + ref.get("at", m.start() - sent)))
            prev = m.end()
    else:
        found = [(m, parse_reference(t[m.start():]), m.start()) for m in _REF.finditer(t)]
    for m, ref, cue in found:
        if not ref:
            continue
        attr, scope, start = _attr_before(t[max(0, cue - 160): cue])
        if not attr:
            continue
        attr = _clean_attr(re.sub(r"^(?:and|with|whose|where|that)\s+", "", attr.strip().lower()))
        if not attr or len(attr.split()) > 4:
            continue
        if not scope:
            s = _SCOPE_LEAD.search(t[max(0, cue - 160): cue][:start])
            scope = s.group("scope") if s else None
        out.append({"attr": attr, "scope": scope, "ref": ref})
    if CONFIG.get("rq_note"):   # RQN: a request sentence shaped like a listing note names a listing for one property
        for r in parse_redirects(t):
            if any(o["ref"].get("seller") == r["seller"] for o in out):
                continue
            attr = _clean_attr(r["attr"])
            if attr and len(attr.split()) <= 4:
                out.append({"attr": attr, "scope": None,
                            "ref": {"item": r.get("item") or "", "seller": r["seller"], "quals": r.get("quals") or "", "neg": False}})
    return out


_RD_FOR = re.compile(r"\bfor (?:its |the |their )?(?P<attr>[a-z][a-z0-9 /&\-]{0,40}?),\s*(?P<body>.+)$", re.IGNORECASE)
_RD_TAKEN = re.compile(r"\b(?:its |the |their )?(?P<attr>[a-z][a-z0-9 /&\-]{0,40}?)\s+(?:is to be|are to be|should be|must be|is|are)\s+"
                       r"(?:taken|read|copied) from\s+(?P<body>.+)$", re.IGNORECASE)
_RD_VERB = re.compile(r"^(?:please\s+)?(?:refer to|look to|look at|go by|follow|use|see|check|consult|"
                      r"take (?:it|the value|the answer|that|this detail)?\s*from|take)\s+", re.IGNORECASE)
_RD_STOP = re.compile(r"(?:,|;)?\s*(?:\bnot (?:to |from |at |by )?(?:this|these|that)\b|\brather than\b|\binstead\b|\bsince\b|"
                      r"\bbecause\b|\bas the reference\b|\bfor that\b|;|\.\s*$)", re.IGNORECASE)
_RD_MORE = (   # other short redirect shapes (each: attr, body, whether the body starts with a verb)
    (re.compile(r"\b(?:its |the )?(?P<attr>[a-z][a-z0-9 /&\-]{0,40}?)\s+(?:is|are) not taken from (?:this|these) \w+[:;,]?\s*(?P<body>.+)$", re.IGNORECASE), True),
    (re.compile(r"\b(?:its |the )?(?P<attr>[a-z][a-z0-9 /&\-]{0,40}?)\s+comes from\s+(?P<body>.+)$", re.IGNORECASE), False),
    (re.compile(r"\btake (?:the |its )?(?P<attr>[a-z][a-z0-9 /&\-]{0,40}?) from\s+(?P<body>.+)$", re.IGNORECASE), False),
)
_FUNCTION_CAPS = {"The", "A", "An", "This", "That", "These", "Those", "It", "Its", "For", "Our", "And", "Or", "With", "Not"}


def _seller_in(target):
    """The seller a redirect target names: the capitalised word after 'from'/'by', or before "'s",
    else its first capitalised word (function words and all-capital acronyms skipped)."""
    for pattern in (r"\bfrom ([A-Z][\w\-]+)", r"\bby ([A-Z][\w\-]+)", r"\b([A-Z][\w\-]+)'s\b",
                    r"\b([A-Z][a-z][\w\-]*) one\b"):   # H4: "the black <Seller> one"
        s = re.search(pattern, target)
        if s and s.group(1) not in _FUNCTION_CAPS:
            return s.group(1)
    for s in re.finditer(r"\b([A-Z][a-z][\w\-]*)\b", target):
        if s.group(1) not in _FUNCTION_CAPS:
            return s.group(1)
    return None


def parse_redirects(text) -> list:
    """Redirect sentences in a listing's text ("For <attr>, refer to <a listing> from <Seller> (...),
    rather than <this listing>"): [{'attr', 'seller', 'item', 'quals', 'raw'}]. `quals` is the whole target
    description minus the seller: its words are SELECTORS for the next listing, not needs."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", str(text or "")):
        m, body = _RD_FOR.search(sentence), None
        if m and _RD_VERB.search(m.group("body")):
            body = _RD_VERB.sub("", m.group("body"), count=1)
        else:
            m = _RD_TAKEN.search(sentence)
            body = m.group("body") if m else None
        for rx, verb in _RD_MORE:
            if body:
                break
            m = rx.search(sentence)
            if m and (not verb or _RD_VERB.search(m.group("body"))):
                body = _RD_VERB.sub("", m.group("body"), count=1) if verb else m.group("body")
        if not body:
            continue
        attr = re.sub(r"^(?:its|the|their)\s+", "", m.group("attr").strip().lower())
        if (not attr or len(attr.split()) > 4
                or attr.split()[0] in ("example", "details", "sizing", "more", "reference", "answer", "value", "it", "that", "this")):
            continue
        target = _RD_STOP.split(body, 1)[0].strip(" ,.;")
        target = re.sub(r"^(?:a different \w+[:,]?\s*)", "", target, flags=re.IGNORECASE)
        seller = _seller_in(target)
        if not seller:
            continue
        quals = re.sub(r"\b(?:from|by)\s+" + re.escape(seller) + r"\b|\b" + re.escape(seller) + r"(?:'s)?\b", " ", target)
        quals = " ".join(quals.split()).strip(" ,.")
        out.append({"attr": attr, "seller": seller, "item": "", "quals": quals, "raw": sentence.strip()})
    return out


_PAIR_HEAD = re.compile(r"\b(?:(?:the )?store's |our )?(?:store )?(?:rule for (?P<a1>[a-z][a-z /&\-]{0,40}?)(?:\s+is that|\s*:|\s+pairs)|"
                        r"(?P<a2>[a-z][a-z /&\-]{0,30}?) (?:pairing )?rule(?:\s+pairs|\s*:|\s+is that))\s*(?P<body>[^.]+)", re.IGNORECASE)
_PAIR_INLINE = (
    re.compile(r"\b(?:that|this|and that)\s+(?P<v>[a-z0-9][a-z0-9\-]*(?: [a-z0-9\-]+)?)\s+(?P<a>[a-z][a-z ]{1,25}?)\s+goes (?:together )?with\s+(?P<w>[a-z0-9][a-z0-9 \-]{0,25}?)(?:[,.;]|$)", re.IGNORECASE),
    re.compile(r"\b(?P<v>[a-z0-9][a-z0-9\-]*) in (?P<a>[a-z][a-z ]{1,25}?),?\s+which goes (?:together )?with\s+(?P<w>[a-z0-9][a-z0-9 \-]{0,25}?)(?:[,.;]|$)", re.IGNORECASE),
)


def parse_pairings(text) -> list:
    """Store value-pairing rules: [{'attr', 'pairs': {canon(value): paired value}}]. A value read from a
    listing that states such a rule for that attribute is replaced by its partner."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", str(text or "")):
        m = _PAIR_HEAD.search(sentence)
        if m:
            attr = (m.group("a1") or m.group("a2") or "").strip().lower()
            attr = re.sub(r"^(?:the|our|store|s)\s+", "", attr)
            pairs = {}
            for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group("body")):
                p = re.match(r"\s*(?P<x>.+?)\s+(?:goes together with|goes with|pairs with|with)\s+(?P<y>.+?)\s*$", part)
                if p:
                    pairs[canon(p.group("x"))] = p.group("y").strip(" .")
            if pairs:
                out.append({"attr": attr, "pairs": pairs})
            continue
        if CONFIG.get("pair_inline"):   # V4-P1: an inline companion sentence names a match, not a replacement
            for rx in _PAIR_INLINE:
                m = rx.search(sentence)
                if m:
                    out.append({"attr": m.group("a").strip().lower(), "pairs": {canon(m.group("v")): m.group("w").strip()}})
                    break
    return out


_DISCLAIM = (
    re.compile(r"\b(?:it|they|this|these)(?: \w+){0,3} (?:is|are) not the (?P<x>[^.;]+)", re.IGNORECASE),
    re.compile(r"(?P<x>[^.;]+?) is a different \w+, and this is not it", re.IGNORECASE),
    re.compile(r"\b(?:a|an) (?P<x>[^.;]+?) (?:version|model) (?:exists|is sold)(?: elsewhere)?", re.IGNORECASE),
    re.compile(r"\b(?:a|an) version (?:with|in) (?P<x>[^.;]+?) (?:exists|is sold)", re.IGNORECASE),
    re.compile(r"\bnot (?:the|a) (?P<x>[^.;]+?) version", re.IGNORECASE),
    re.compile(r"\b(?:is|are) not (?:a|an|in) (?P<x>[^.;,]+)", re.IGNORECASE),          # "it is not a <x>"
    re.compile(r",\s*not (?P<x>\d[^.;,]*)", re.IGNORECASE),                            # "<n> <unit>, not <m> <unit>"
)


def parse_disclaimers(text) -> list:
    """Token sets of look-alike listings this listing says it is NOT (a sentence denying it is the
    version with some qualifiers, or saying such a version exists elsewhere)."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", str(text or "")):
        if _RD_FOR.search(sentence) and _RD_VERB.search(_RD_FOR.search(sentence).group("body")):
            continue
        for rx in _DISCLAIM:
            for m in rx.finditer(sentence):
                t = set(canon(m.group("x")).split()) - _SEL_STOP
                if t:
                    out.append(t)
    return out
# <<<<<<<< C0017 claude-2: reference parsing (END) <<<<<<<<


# ---- shopper answers -----------------------------------------------------------------------
# Structural reading, no phrase lists from any suite: a reply is a constraint when it names a
# value of the property asked (the catalogue's own values), a bounded number, or "<property>
# <linking words> <value>"; anything else carries no information.

_FUNCTION_WORDS = {"a", "an", "the", "to", "be", "is", "are", "as", "at", "on", "in", "of", "with", "for", "by",
                   "that", "this", "it", "its", "my", "me", "i", "i'm", "im", "i'll", "you", "your", "we", "our",
                   "and", "or", "but", "so", "if", "then", "than", "just", "only", "really", "very", "still", "also",
                   "has", "have", "had", "got", "get", "must", "should", "will", "would", "can", "could",
                   "need", "want", "keep", "put", "say", "says", "said", "set", "down", "up", "out", "into", "from",
                   "when", "where", "what", "which", "who", "how", "there", "here", "all", "any", "some", "no", "not",
                   "don't", "won't", "can't", "isn't", "it's", "that's", "what's", "let's", "please", "sure", "now"}
_LINK = re.compile(r"\s+(?:being|showing|matching|reading|equal to|equals|set to|(?:must|has to|should|needs to|to) be|"
                   r"is|are|=|:)\s+", re.IGNORECASE)
_CLAUSE_END = re.compile(r"[,;!?()—]|\.(?!\d)|\s(?:and|because|so|but|same|like|while|since|unless|which)\s|"
                         r"\sor\s(?=[^,.;]*\b(?:i|i'm|i'll|you|we|they)\b)", re.IGNORECASE)
_BOUND = re.compile(r"\b(?:at (?:least|most)|up to|no (?:more|less) than|minimum(?: of)?|maximum(?: of)?)\s+\d+(?:\.\d+)?\s*[a-z\"]*"
                    r"|\b\d+(?:\.\d+)?\s*[a-z\"]*\s+or (?:more|less|bigger|smaller|larger|higher|lower|above|below)\b", re.IGNORECASE)


def _clean_value(v):
    """Cut a value at the first clause boundary and at a trailing qualifier phrase ("on the ...",
    "at the ..."); drop brackets and leading articles."""
    v = re.sub(r"\s*\(.*?\)\s*", " ", str(v or ""))
    v = _CLAUSE_END.split(" " + v.strip() + " ", 1)[0]
    head = re.split(r"\s+(?:on|at|from|for|in|to)\s+(?:the|a|an|my|your|its|their|it)\b", v.strip(), 1)[0]
    v = head if toks(head) else v
    return re.sub(r"^(?:a|an|the)\s+", "", v.strip(" .,;!\"'"), flags=re.IGNORECASE)


def _snap_value(value, attr, shop=None, line=None):
    """A revealed value cut to the catalogue value it starts with ("<value>, <more words>" -> "<value>"):
    candidates' own values, never a fixed list. A value no candidate shows gives way to the one catalogue
    value of the property the reply itself names (not right after a negation)."""
    if not isinstance(value, str) or shop is None:
        return value
    cv = canon(value)
    known = sorted(shop.values_of(attr or ""), key=len, reverse=True)
    if not cv or any(canon(k) == cv for k in known):
        return value
    for k in known:
        ck = canon(k)
        if ck and cv.startswith(ck + " "):
            return k
    if line and not any(set(cv.split()) & set(canon(k).split()) for k in known):
        said = " " + canon(line) + " "
        named = [k for k in known if canon(k) and " " + canon(k) + " " in said
                 and not re.search(r"\b(?:no|not|never|nothing|without|except|nor)\s+(?:\w+\s+)?" + re.escape(canon(k)) + r"\b",
                                   canon(line))]
        named = [k for k in named if not any(o != k and " " + canon(k) + " " in " " + canon(o) + " " for o in named)]
        if len(named) == 1:
            return named[0]
    return value


def _scope_of(a):
    """The product phrase a reveal names: '<attr> of the <phrase>' -> '<phrase>'."""
    m = re.search(r"\s+of (?:the|my|his|her|their) (.+)$", str(a or "").strip(" ,.").lower())
    return m.group(1).strip() if m else None


def _clean_attr(a):
    a = a.strip(" ,.").lower()
    a = re.sub(r"^(?:and|but|also|so|plus|then)\s+", "", a)   # ", and the pattern has to be ..."
    a = re.sub(r"\s+(?:can|could|should|must|will|would|may|might)$", "", a)
    a = re.sub(r"\s+of (?:the|my|his|her|their) .+$", "", a)   # "<property> of the <product>"
    a = re.sub(r"\s+(?:on|for|of|in) (?:it|them)$", "", a)
    a = re.sub(r"\b(col|flav)our\b", r"\1or", a)
    return re.sub(r"^(?:the|my|its|his|her)\s+", "", a).strip()


def _prop_before(text, end):
    """The last 1-4 content words before position `end` (the property name)."""
    head = re.split(r"[,;.!?()]", text[:end])[-1]
    parts = [w for w in re.findall(r"[a-z0-9][a-z0-9'/&\-]*", head.lower())]
    out = []
    for w in reversed(parts):
        if w in _FUNCTION_WORDS:
            if out:
                break
            continue
        out.append(w)
        if len(out) == 4:
            break
    return " ".join(reversed(out))


_RV_END = re.compile(r"\s*(?:[,;:!?()\[\]]|\.(?!\d)|\s[–—-]\s)\s*")   # a decimal point ends no clause
_RV_JOIN = re.compile(r"\s+(?:but|since|so|because|though|while)\s+")
_RV_WHO = {"i", "me", "it", "that", "this", "they", "them", "we", "us", "you", "there"}
_RV_BRIDGE = re.compile(r"\s*(?:(?:that|which|to|is|are|was|be|will|must|should|would)\s+)?(?P<b>[a-z][a-z'\-]*)\s+"
                        r"(?:(?:as|on|in|at|to|be|out|up|down|for|into)\s+)*(?P<v>\S.*)$")


_RV_NOBRIDGE = {"of", "in", "on", "at", "for", "with", "by", "from", "into", "to", "the", "a", "an", "my", "your", "our",
                "its", "his", "her", "their", "and", "or"}


def _rv_words(text):
    return set(re.findall(r"[a-z0-9']+", str(text or "").lower()))


def _rv_value(clause):
    """RV1: '<bridge> <value>' -> value; a clause joined after it (an "and" or "since" opening a clause about it) is cut off."""
    m = _RV_BRIDGE.match(clause or "")
    if not m or m.group("b") in _RV_NOBRIDGE:
        return None   # a preposition or an article is no bridge ("in the form of ...", "of the <line> ...")
    v = m.group("v")
    if " as " in f" {v} ":
        v = re.split(r"(?:^|\s)as\s+", v)[-1]   # "<verb> in <words> as <value>": the last "as" introduces it
    j = _RV_JOIN.search(v)
    if j:
        v = v[:j.start()]
    ands = [a for a in re.finditer(r"\s+and\s+", v) if _rv_words(v[a.end():]) & _RV_WHO]
    if ands:   # "<value> and <a clause about it>": the last such "and" ends the value
        v = v[:ands[-1].start()]
    v = re.sub(r"^(?:a|an|the)\s+", "", v.strip(" .\"'"))
    words = toks(v)
    if not words or len(v.split()) > 5 or set(words) <= _STOP or _rv_words(v) & _RV_WHO:
        return None
    return v


def _rv_relabel(later, value):
    """RV1: 'as <w>, and <w> <bridge> <value>': the value the sentence gives <w> after naming it again, or None."""
    again = re.search(r"\b" + re.escape(str(value).lower()) + r"\b", (later or "").lower())
    return _rv_value(later.lower()[again.end():]) if again else None


def _rv_read(text, asked, scopes=()):
    """RV1: (value, scope) of a reply about `asked`, read from the property's own clause or the I/we clause after it."""
    low = (text or "").lower().replace("’", "'")
    words = toks(asked or "")
    m = re.search(r"\b" + r"\W+".join(re.escape(w) for w in words) + r"\b", low) if words else None
    if not m:
        return None
    rest, scope = low[m.end():], None
    for s in sorted({str(s).lower() for s in scopes or () if s}, key=len, reverse=True):
        k = re.match(r"\s+of (?:the|my|our|his|her|their|these|those) " + re.escape(s) + r"(?![\w])", rest)
        if k:
            rest, scope = rest[k.end():], s
            break
    else:
        k = re.match(r"\s+of (?:the|my|our|his|her|their|these|those) ([a-z0-9'&/\-]+(?: [a-z0-9'&/\-]+){0,4}?)"
                     r"(?=\s+(?:that|which|to|is|are|as)\s)", rest)
        if k:
            rest, scope = rest[k.end():], k.group(1)
    clauses = _RV_END.split(rest)
    own = _rv_value(clauses[0]) if clauses else None
    after = None
    for c in clauses[1:2]:   # "<I/we> <bridge> <value>" right after the property's clause
        c = re.sub(r"^(?:and|but|so)\s+", "", c.strip())
        lead = re.match(r"(?:i|we)(?:'m|'ll|'d|'ve|'re)?\s+(?:(?:only|just|still|really|do|will|would|can|shall)\s+)?", c)
        after = _rv_value(c[lead.end():]) if lead else None
    if own and after and re.match(r"\s*(?:is|are|was|were)\s+\S+\s*$", clauses[0]):
        own = None   # "<property> is <word>, I <bridge> <value>": the clause after it holds the value
    if own:
        return (_rv_relabel(" ".join(clauses[1:]), own) or own), scope
    if clauses and (not clauses[0].strip() or re.match(r"\s*(?:is|are|was|were)\s+\S+\s*$", clauses[0])):
        return (after, scope) if after else None   # only a property clause without content hands over to the next one
    return None


def classify_answer(text, asked=None, values=None, properties=None, scopes=None) -> dict:
    """A shopper reply: constraint (attr, op, value) | reference | noinfo.
    `values` are the catalogue values seen for the asked property and `properties` the
    property names seen in this category; both are optional. `scopes`: line and category names (RV1)."""
    t = (text or "").strip()
    if not t:
        return {"kind": "noinfo"}
    ref = parse_reference(t)
    if ref and ref.get("page"):   # H2: the property is a known name (or the one asked) written before the reference
        head = " " + " ".join(toks(t[:ref["at"]])) + " "
        names = sorted({p for p in list(properties or []) + [asked] if p}, key=lambda p: (-len(toks(p)), p))
        attr = next((p for p in names if toks(p) and " " + " ".join(toks(p)) + " " in head), None)
        if attr is None:
            attr = _prop_before(t, ref["at"]) or asked
        low = t[:ref["at"]].lower()
        k = low.rfind(" ".join(str(attr).lower().split()[-1:]) or "\0")
        scope = re.match(r"\w*\s+of (?:the|my|his|her|their) ((?:[a-z&'\-]+ ?){1,5}?)(?=\s*[,;]|\s+(?:is|are|stays|coming|which|that|"
                         r"but|as|in|to|be|i|i'm|i'll|will|must)\b)", low[k:]) if k >= 0 else None
        return {"kind": "reference", "attr": _clean_attr(attr) if attr else asked, "ref": ref,
                "scope": scope.group(1).strip() if scope else None}
    if ref:
        m = _LINK.search(t)
        attr = _prop_before(t, m.start()) if m else None
        return {"kind": "reference", "attr": _clean_attr(attr) if attr else asked, "ref": ref,
                "scope": _scope_of(t[:m.start()]) if m else None}
    low = " " + " ".join(toks(t)) + " "
    names_asked = bool(asked) and set(toks(asked)) <= set(low.split())
    for value in sorted(values or [], key=len, reverse=True) if names_asked else []:   # property named + its value
        vt = " ".join(toks(value))
        if not vt or set(vt.split()) <= _FUNCTION_WORDS:
            continue   # a value like "can" or "no" is ordinary English, not evidence
        if f" {vt} " in low:
            got = _rv_read(t, asked, scopes) if CONFIG.get("rv_clause", True) else None
            gt = " ".join(toks(got[0])) if got else ""
            extra = f" {gt} ".replace(f" {vt} ", " ", 1).split() if gt and f" {vt} " in f" {gt} " else []
            if (gt and gt != vt and extra and len(extra) <= 2 and not set(extra) & _FUNCTION_WORDS
                    and not _rv_words(got[0]) & _FUNCTION_WORDS):   # RV1: "<modifier> <listed value>" holds the listed value
                return {"kind": "constraint", "attr": asked, "op": "eq", "value": got[0], "scope": got[1], "clause": "rv"}
            return {"kind": "constraint", "attr": asked, "op": "eq", "value": str(value),
                    "clause": "rv" if CONFIG.get("rv_clause", True) else None}   # RV1: a listed value is kept whole
    if names_asked:   # S2-F1: "<asked> [of the <product>] <up to 3 words> as|to be <value>"
        m = re.search(r"\b" + r"\s+".join(re.escape(w) for w in toks(asked)) + r"\b(?:\s+of (?:the|my|his|her|their|these|those)"
                      r" [^,.;!?]{1,60}?)?\s+(?:[a-z']+\s+){0,3}?(?:as|to be)\s+(?P<v>(?:[^,.;!?()]|\.(?=\d))+)", t, re.IGNORECASE)
        value = _clean_value(m.group("v")) if m else ""
        if value and CONFIG.get("rv_clause", True):
            value = _rv_relabel(t[m.end("v"):], value) or value   # RV1: "as <w>, and <w> <bridge> <value>"
        if (value and toks(value) and len(value.split()) <= 4 and not set(toks(value)) <= _FUNCTION_WORDS
                and not set(toks(value)) & {"whatever", "anything", "any", "long", "well", "much", "many", "possible"}):
            op, value = value_op(value)
            scope = _scope_of(re.sub(r"\s+(?:as|to be)\s*$", "", t[m.start():m.start("v")].rstrip()))
            return {"kind": "constraint", "attr": asked, "op": op, "value": value, "scope": scope,
                    "clause": "as" if CONFIG.get("rv_clause", True) else None}
    got = _rv_read(t, asked, scopes) if names_asked and CONFIG.get("rv_clause", True) else None
    if got:   # RV1: the value from the property's own clause (or the I/we clause after it)
        value, scope = got
        if re.match(r"with\s", value) and not any(str(v).lower().startswith("with ") for v in values or ()):
            value = value[5:].strip()   # "<verb> with <value>": a preposition, unless the key's own values start with it
        op, bare = value_op(value)
        if op != "eq" and toks(bare) and toks(value)[:len(toks(bare))] == toks(bare):
            op, bare = "eq", value   # "<n> <unit> and above" is one catalogue value
        return {"kind": "constraint", "attr": asked, "op": op, "value": bare, "scope": scope, "clause": "rv"}
    for name in sorted(properties or [], key=len, reverse=True):   # "<known property> <link> <value>"
        m = re.search(r"\b" + re.escape(name) + r"\b", t, re.IGNORECASE)
        if m:
            rest = _LINK.match(t[m.end():].replace(" ", " ")) or re.match(r"\s+", t[m.end():])
            value = _clean_value(t[m.end() + (rest.end() if rest else 0):])
            if value and toks(value) and len(value.split()) <= 6:
                op, value = value_op(value)
                return {"kind": "constraint", "attr": name, "op": op, "value": value, "scope": _scope_of(t[m.start():])}
    bound = _BOUND.search(t)
    if bound and asked:
        op, value = value_op(bound.group(0))
        return {"kind": "constraint", "attr": asked, "op": op, "value": value}
    for m in _LINK.finditer(t):                                 # generic "<words> <link> <value>"
        attr = _prop_before(t, m.start())
        value = _clean_value(t[m.end():])
        if attr and value and toks(value) and len(value.split()) <= 6 and not set(toks(value)) & {"whatever", "anything", "any"}:
            if asked and not set(toks(asked)) & set(toks(attr)):
                continue
            op, value = value_op(value)
            return {"kind": "constraint", "attr": _clean_attr(attr), "op": op, "value": value, "scope": _scope_of(attr)}
    return {"kind": "noinfo"}


# ---- unprompted messages and refusal lines --------------------------------------------------
# Structural: money is a budget; a line naming a property the shopper needs, with no value,
# relaxes that need; any other line concerns the proposed item (price or stock), which the core
# re-reads before the next checkout.

_STOCK_GONE = re.compile(r"\b(?:sold out|out of (?:the )?stock|no longer (?:available|in stock|for sale|sold)|unavailable|"
                          r"can(?:no|')t be (?:ordered|bought|purchased)|cannot be (?:ordered|bought|purchased)|"
                          r"(?:none|nothing|zero) left|no stock|(?:ran|run|has run) out|(?:is|are|went) gone)\b")


def classify_event(text, need_props=()) -> dict:
    """Kind of an unprompted shopper line: budget | stock | relax | item."""
    t = (text or "").strip()
    money = parse_money(t)
    if money:
        return {"kind": "budget", "cents": money[-1][0], "unit": money[-1][1]}
    if _STOCK_GONE.search(t.lower().replace("\u2019", "'")):
        return {"kind": "stock"}
    low = " " + " ".join(toks(t)) + " "
    for prop in sorted({p for p in need_props if p}, key=len, reverse=True):
        pt = " ".join(toks(prop))
        if pt and f" {pt} " in low:
            return {"kind": "relax", "attr": prop}
    return {"kind": "item"}


_TAILS = re.compile(r"(?:,|;|\s\u2014|\s-\s).*$")   # structural: a value ends at its clause


# ---- the shopper's request ---------------------------------------------------------------------

_PRODUCT_PATTERNS = [   # "<want-verb> <noun phrase>": plain English verbs of buying, wanting and looking
    r"\b(?:want|need|like|after|wanting|needing|take|see) (?:something|anything) (?:from|in|out of) (?:the |your )?(?P<p>[a-z][^,.;:()]+)",
    r"^\s*(?P<p>[a-z][^,.;:()]{1,40}?) (?:is|are) what (?:i|we)(?:'m| am|'re| are)? [a-z]+",
    r"\b(?:item|product|thing)(?:'s| is)?\s*:\s*(?:a |an |some |one )?(?P<p>[a-z][^,.;:()]+)",
    r"^\s*(?P<p>[a-z][a-z0-9'&/\-]*(?: [a-z][a-z0-9'&/\-]*){0,3}),? for (?:me|us)(?=[,.!]| please\b| and\b|$)",
    r"^\s*(?P<p>[a-z][^,.;:()]{1,40}?) (?:is|are) the (?:category|range|section|department)\b",
    r"^\s*(?:please\s+)?(?:buy|get|order|find)\s+(?:me\s+)?(?:a (?:pair|couple) of |one |a |an |some |\d+\s*x |\d+ |two |three )?(?P<p>[^,.;:()]+)",
    r"^\s*(?:(?:i|we)(?:'m| am|'re| are)\s+)?(?:looking|shopping|searching) for\s+(?:a (?:pair|couple) of |a |an |some |one )?(?P<p>[^,.;:()]+)",   # RQF
    r"\b(?:i|we)(?:'m| am|'re| are) (?:looking|shopping|searching) for\s+(?:a (?:pair|couple) of |a |an |some |one )?(?P<p>[^,.;:()]+)",
    r"\b(?:i|we)(?:'m| am|'re| are) after\s+(?:a (?:pair|couple) of |a |an |some |one )?(?P<p>[^,.;:()]+)",
    r"\b(?:i|we)(?:'d| would) (?:also )?like (?:to (?:buy|get|order|see|find|try) )?(?:a (?:pair|couple) of |a |an |some |one )?(?P<p>[^,.;:()]+)",
    r"\b(?:i|we) (?:also )?(?:want|need)(?: to (?:buy|get|order))?\s+(?:a (?:pair|couple) of |a |an |some |one )?(?P<p>[^,.;:()]+)",
    r"\b(?:show|find|get) me (?:a (?:pair|couple) of |a |an |some )?(?P<p>[^,.;:()]+)",
    r"\b(?:i|we)(?:'m| am|'re| are) [a-z]+ing for\s+(?:a (?:pair|couple) of |a |an |some |one )?(?P<p>[^,.;:()]+)",
    r"\b(?:i|we)(?:'m| am|'re| are) [a-z]+ing (?:a (?:pair|couple) of |a |an |some )(?P<p>[^,.;:()]+)",
    r"\bstart with (?:a (?:pair|couple) of |a |an |some |the )?(?P<p>[^,.;:()]+)",
    r"\bit(?:'s| is) (?:a |an |some )?(?P<p>[a-z][^,.;:()]{1,40}?) (?:that )?(?:i|we)(?:'m| am|'re| are)? [a-z]+(?:ing)?\b",
    r"\bit(?:'s| is) (?:a|an) (?P<p>[a-z][^,.;:()]{1,40})",
    r"\b(?:have|has|got|carry|sell|stock)(?: you)? any (?P<p>[a-z][^,.;:()?]{1,40})",
    r"\bit (?:has to|must|should|needs to) be (?:a |an |some )(?P<p>[^,.;:()]+)",
    r"\b(?:a|an) (?P<p>[a-z][a-z \-/&]{2,30}?) with (?:a |an )?[a-z]",
]
_PRODUCT_RES = [re.compile(p, re.IGNORECASE) for p in _PRODUCT_PATTERNS]
_PRODUCT_WEAK = 4   # the last patterns: "it's a <NP>", "have any <NP>", "it must be a <NP>", "a <NP> with"
_NARRATIVE = {"i", "me", "my", "we", "our", "you", "your", "he", "she", "her", "his", "they", "it", "its", "here",
              "there", "before", "then", "when", "something", "anything", "whatever", "thing"}
_PRODUCT_CUT = re.compile(r"\s+(?:range|with|in the|in a|in an|that|that'd|which|where|from|for|so|but|and the|and it|made of|"
                          r"in|one where|whose|whichever|please|to)\b.*$", re.IGNORECASE)
_NEED_PATTERNS = [   # "<property> <modal> be <value>" and a few other plain constructions
    r"(?:^|[.;]\s*)(?P<a>[a-z][a-z0-9_ /&\-]{1,30}?) (?P<o>eq|in|le|ge|lt|gt|==|<=|>=) (?P<v>(?:[^.;]|\.(?=\d))+)",
    r"(?:^|[.;,]\s*|\bfor (?:the )?[^,.;]{2,60},\s*)(?P<a>[a-z][a-z0-9_ /&\-]{1,30}?) (?:must|should|has to|needs to) be (?:one of |including |inclusive of |matching )?(?P<v>(?:[^.;]|\.(?=\d))+)",
    r"\bthe (?P<a>[a-z][a-z0-9 /&\-]{1,30}?)(?:'s| has| must| should| needs?| is)?\s*(?:has to|have to|must|should|needs? to|got to|gotta|ought to|is to|to)?\s*"
    r"\b(?:be|say|read|come in|come from|measure|match|equal|go (?:up )?to|land on|land at|come to)\s+(?:a |an |the |one of |including |inclusive of |matching )?(?P<v>(?:[^,.;()]|\.(?=\d))+)",
    r"\b(?:it|it's|it is)\s+(?:has to|have to|must|should|needs? to|got to|gotta|ought to|is to)\s+(?:be|say|read|come in|come from|measure|match|equal|go (?:up )?to)\s+(?P<v>(?:[^,.;()]|\.(?=\d))+)",
    r"\b(?:where|and) the (?P<a>[a-z][a-z0-9 /&\-]{1,30}?) (?:is|are) (?P<v>(?:[^,.;()]|\.(?=\d))+)",
    r"\bwith (?:a |an )?(?P<v>[a-z0-9][^,.;()]{1,30}?) as the (?P<a>[a-z][a-z /&\-]{1,30}?)\b",
    r"\bwith (?P<v>[a-z0-9][a-z0-9 \-]{1,25}?) on the (?P<a>[a-z][a-z ]{1,20}?)\b",
    r"\bmade of (?P<v>[a-z0-9][^,.;()?]{1,25})",
    r"\bfor an? (?P<v>[a-z][a-z\-]{2,15}) (?P<a>occasion)\b",
    r"\bthe (?P<a>[a-z][a-z /&\-]{1,25}?) can be (?P<v>(?:[^.;()]|\.(?=\d))+?)(?:\.|,|$)",
    r"\b(?:be|is|are) (?P<v>[a-z0-9][a-z0-9 %\-]{0,25}?),? as (?:far|long) as (?:the )?(?P<a>[a-z][a-z ]{1,25}?) go(?:es)?\b",
    r"\b(?:be|is|are) (?P<v>[a-z0-9][a-z0-9 %\-]{0,25}?), that(?:'s| is) the (?P<a>[a-z][a-z ]{1,25}?) (?:i|we) (?:want|need|like)\b",
    r"\bit(?:'s| is) the (?P<v>[a-z][a-z\-]*(?: [a-z][a-z\-]*){0,2}?) (?P<a>[a-z]+ (?:type|kind)|[a-z]+) (?:that )?(?:i|we)(?:'m| am| are)? (?:want|need|after|wanting)\b",
    r"(?:^|[.;!?]\s+)(?P<v>[a-z][a-z\-]{2,20}) is the (?P<a>[a-z][a-z ]{1,20}?) (?:that|i|we) ",
    r"(?:^|[.;]\s*)on (?:the )?(?P<a>[a-z][a-z /&\-]{1,25}?), it(?:'s| is| (?:has to|must|should) be) (?P<v>[a-z0-9][^,.;()]{0,40})",
    r"(?:^|[.;!?]\s+)(?P<a>[a-z][a-z0-9 /&\-]{1,30}?)\s*:\s*(?P<v>[a-z0-9](?:(?!\b(?:calls?|tool calls?)\b)[^,.;()?]){0,40})",   # RQF: "<a>: <v>"
    r"\bwhose (?P<a>[a-z][a-z0-9 /&\-]{1,30}?) (?:is|are|must be|should be) (?P<v>[a-z0-9][^,.;()]{0,40})",   # RQF
    r"\b(?:have|has|having) (?P<v>[a-z0-9][a-z0-9 \-]{1,25}?) as (?:the|its|their) (?P<a>[a-z][a-z /&\-]{1,30}?)\b",   # RQF
    r"\bin (?P<v>[a-z0-9][a-z0-9 \-]{1,25}?) \((?P<a>[a-z][a-z /&\-]{1,30}?)\)",   # RQF: "in <v> (<a>)"
]
_NEED_RES = [re.compile(p, re.IGNORECASE) for p in _NEED_PATTERNS]
# Tentative phrases (C0028): "<value> <property>" groups and bare values whose property is only known
# once candidates are seen. Reasoning.late_needs keeps one only when a candidate fact key names the
# property (or, for a bare value, when candidates carry that exact value under some key).
_TENTATIVE_RES = [re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:in|with) (?P<art>the |a |an )?(?P<g>[a-z0-9][a-z0-9 \-]{1,40}?)(?=[,.;()]|$| and | but | for | that | which | so | or )",
    r"(?:^|[,;:]\s*)(?P<g>[a-z][a-z]+[- ][a-z]+)(?=\s*[,.;(]|$)",
    r"\b(?:in|for a|for an|for my|meant for a|meant for an) (?P<v>[a-z][a-z\-]{2,20})\b",
    r"\b(?P<v>[a-z][a-z\-]{2,20}) is the (?P<a>[a-z][a-z ]{1,20}?)\b(?= that| i| here|[,.;]|$)",
    r"\bit(?:'s| is) the (?P<g>[a-z][a-z\- ]{2,40}?) (?:that )?(?:i|we)(?:'m| am| are)? (?:want|need|after|wanting)\b",
    r"\b(?:take|have|want|need) (?P<o>[a-z][a-z\-]{2,20}) only\b",
)]
_AVOID_RES = [re.compile(p, re.IGNORECASE) for p in (   # "leave out / skip / exclude ... <phrase>"
    r"\b(?:leave out|skip|exclude|rule out|avoid) (?:whatever |anything )?(?:that )?(?:has |with )(?P<x>[^.;(),]+)",
    r"\b(?:nothing|anything) with (?P<x>[^.;(),]+)",
    r"\bwithout (?:a |an |any )?(?P<x>[a-z][a-z \-]{2,25}?)(?=[,.;]|$)",
)]
_ATTR_JUNK = {"price", "budget", "thing", "one", "item", "cost", "label", "listing", "website", "store", "shelf", "deal", "total", "number", "it"}
_VALUE_JUNK = re.compile(r"\b(?:cts|cents|cent|messages?|moves|steps|website|shelf|on hand|substitute|price)\b", re.IGNORECASE)


_COUNT_NOUN = r"(?:of (?:them|these|those)|units|pieces|identical|separate|individual|matching|copies|packs of)"
# F-033 count guard: the closed-class cue words that say whether a count is the order's (G1) or a value (G2)
_COUNT_CUES = re.compile(
    r"\b(?P<ask>want|wants|need|needs|get|getting|order|buy|buying|find|look(?:ing)? for|after|take|grab|pick|send|add|please|"
    r"give me|require|(?:would|'d|i) like|(?:could|can|would|will) you|(?:do|does) you (?:have|carry|sell|stock)|shopping for)\b"
    r"|\b(?P<past>had|(?<!\bto )(?<!\byou )(?<!\bplease )(?:have|has)|own|owns|owned|already|did|bought|got|gotten|ordered|"
    r"purchased|wore|worn|received|gave|"
    r"given|tried|returned|kept|(?:(?<=\bi )|(?<=\bwe )|(?<=\bthey )|(?<=\bhe )|(?<=\bshe )|(?<=\byou ))used(?: to [a-z]+)?|"
    r"broke|broken|lost|sold|paid|picked up|grabbed|brought|found|chose|took|went|came|ago|last|"
    r"yesterday|previously|before|earlier|because|since)\b", re.IGNORECASE)
_VALUE_LEAD = {"be", "is", "are", "was", "were", "under", "over", "than", "below", "above"}


def _order_count(text, start, end):
    """F-033: may the count at text[start:end] set the order quantity? Not right after "be", "was" or a bound word (G2),
    and not when the nearest cue before it in its clause is a past, possession or reason word (G1)."""
    before = re.split(r"[.!?;,\n]", text[:start])[-1]
    lead = before.lower().split()[-1:]
    if lead and lead[0] in _VALUE_LEAD:
        return False
    cues = list(_COUNT_CUES.finditer(before))
    return not cues or cues[-1].group("ask") is not None
_INDEFINITE = re.compile(r"\b(?:a|an|some|any|another)\s+(?P<p>[a-z][a-z0-9'\-/&]*(?:\s+[a-z][a-z0-9'\-/&]*){0,3})",
                         re.IGNORECASE)


def _indefinite_np(sentences):
    """The first "a / an / some / any <noun phrase>" (cut at its clause words) outside the sentences about
    the budget and the limits: plain grammar, used when no want-verb introduces the product."""
    for sentence in sentences:
        if re.search(r"(cts|cents|messages|moves|steps|website|on hand|shelf|substitute)", sentence, re.IGNORECASE):
            continue
        sentence, measures = _mask_measures(sentence)   # A-R7 M1
        for m in _INDEFINITE.finditer(sentence):
            before = sentence[:m.start()].lower().split()[-1:] or [""]
            if re.fullmatch(r"(?:be|is|are|was|were|'s|it's|for|like|as|than|of|into|about)", before[0]):
                continue   # a complement or a modifier phrase, not the item itself
            phrase = _PRODUCT_CUT.sub("", m.group("p")).strip(" ,.?!'")
            phrase = re.sub(r"^(?:new |pair of )+", "", phrase, flags=re.IGNORECASE)
            ws = set(words(phrase))
            if (phrase and len(phrase.split()) <= 4 and not _VALUE_JUNK.search(phrase)
                    and not ws & (_NARRATIVE | _NOT_PRODUCT | _PRONOUNS) and not ws <= _FUNCTION_WORDS
                    and not _order_head(phrase)):   # A-R5 T1
                return _unmask_measures(phrase.lower(), measures)
    return None


def parse_request(text) -> dict:
    """Structured fields of a shopper request (practice and qualifying phrasing)."""
    t = " ".join(str(text or "").split())
    low = t.lower()
    out = {"budget": None, "budget_unit": None, "msg_budget": None, "step_budget": None, "quantity": 1,
           "prefer": "cheapest", "stock_only": False, "decline_ok": False, "product": None, "category": None,
           "needs": [], "avoid": [], "tentative": []}
    money = parse_money(t)
    for cents, unit, pos in money:
        if _BUDGET_CUE.search(t[max(0, pos - 80): pos + 40]):
            out["budget"], out["budget_unit"] = cents, unit
            break
    if out["budget"] is None and money:
        out["budget"], out["budget_unit"] = money[0][0], money[0][1]
    out["msg_budget"] = _count_near(t, r"messages?|questions?|texts?")
    out["step_budget"] = _count_near(t, r"moves?|steps?|actions?|turns?|tool calls?|calls?")   # CAP: "40 calls"
    out["stock_only"] = bool(re.search(r"(on hand|\bshelf\b|in stock|out of (?:the )?stock|back ?order|sold out)", low))
    out["decline_ok"] = bool(re.search(r"(nothing (?:found|fits)|say so|substitute)", low))
    if re.search(r"(best[- ]rated|highest[- ]rated|top[- ]rated|best reviewed)", low) and not re.search(
            r"\b(?:cheapest|lowest|least)\b", low):
        out["prefer"] = "best"
    qty = next((m for m in re.finditer(r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+" + _COUNT_NOUN + r"\b", low)
                if _order_count(low, m.start(), m.end())), None)   # F-033: the first count the request asks for
    if qty:
        out["quantity"] = parse_count(qty.group(1)) or 1
    sentences = re.split(r"(?<=[.!?])\s+", t)
    tiers = (_PRODUCT_RES[:-_PRODUCT_WEAK], _PRODUCT_RES[-_PRODUCT_WEAK:])   # a want-verb in any sentence first
    for s, tier in [(x, tier) for tier in tiers for x in sentences]:
        if out["product"]:
            break
        s = _strip_money_clause(s)   # A-R7c C1: "<item> under 3000 cents" keeps its item
        if re.search(r"(cts|cents|messages|moves|steps|website|on hand|shelf|substitute)", s, re.IGNORECASE):
            s = ", ".join(c for c in re.split(r"[,;]", s)   # RQF: drop the budget clause, keep the product clause
                          if not re.search(r"(cts|cents|messages|moves|steps|website|on hand|shelf|substitute)", c, re.IGNORECASE))
            if not s.strip():
                continue
        s, measures = _mask_measures(s)   # A-R7 M1: "a 10.1 inch <item>" is not the product "10"
        for rx in tier:
            m = rx.search(s)
            if m:
                phrase = _PRODUCT_CUT.sub("", _drop_order_clause(m.group("p"))).strip(" ,.?!'")   # A-R7c C2
                phrase = re.sub(r"^(?:something from |some |a |an |the |one |new |another |pair of )+",
                                "", phrase, flags=re.IGNORECASE)
                if (phrase and len(phrase.split()) <= 6 and not _VALUE_JUNK.search(phrase)
                        and not set(words(phrase)) & _NARRATIVE and not _order_head(phrase)):   # A-R5 T1
                    out["product"] = _unmask_measures(phrase.lower(), measures)
                    out["product_weak"] = rx is _PRODUCT_RES[-1]   # "a <NP> with ...": the most narrative construction
                    break
    if not out["product"]:   # grammar fallback: the first indefinite noun phrase of a sentence about the item
        out["product"] = _indefinite_np(sentences)
        out["product_weak"] = out["product"] is not None
    cat = (re.search(r"\b(?:from|in) (?:the |your )(?P<c>[a-z][a-z &/'\-]{2,40}?) (?:range|section|category|department)\b", low)
           or re.search(r"\b(?:something|anything) (?:from|in) (?:the |your )?(?P<c>[a-z][a-z &/'\-]{2,40}?)"
                        r"(?= where\b| and the\b| with\b|[,.;]|$)", low)
           or re.search(r"(?:^|[.!?]\s+)(?P<c>[a-z][a-z &/'\-]{2,40}?) (?:is|are) the (?:category|range|section|department)\b", low))
    if not cat and out["product"]:
        head = re.escape(out["product"].split()[-1].rstrip("s"))
        cat = re.search(re.escape(out["product"]) + r" (?:in|from) (?:the |your )?(?P<c>[a-z][a-z &/'\-]{0,40}?\b" + head
                        + r"(?:e?s)?)\b", low)
    if cat:
        out["category"] = next(g for g in cat.groups() if g).strip()
        if out.get("product_weak") or not out["product"]:   # a story noun phrase never outranks the named category
            out["product"], out["product_weak"] = out["category"], False
    seen = set()
    for rx in _NEED_RES:
        for m in rx.finditer(t):
            gd = m.groupdict()
            attr = (gd.get("a") or gd.get("a2") or "").strip().lower() or None
            raw = gd.get("v") or ""
            if attr and (attr in _ATTR_JUNK or _VALUE_JUNK.search(attr)
                         or attr.split()[0] in ("nothing", "anything", "everything", "something")):
                continue
            if attr and attr.split()[0] in ("i", "we"):
                attr = None   # "I want <product> in <value>": the value stands, its property is unknown
            listed = re.match(r"\s*(\[[^\]]*\])", raw)
            if listed:
                raw = listed.group(1).replace(",", "|")
            value = raw.strip(" []").replace("'", "").replace('"', "") if listed else _clean_value(raw)
            word = (gd.get("o") or "").lower()
            if word in ("le", "lt", "<="):
                value = "at most " + value
            elif word in ("ge", "gt", ">="):
                value = "at least " + value
            if not value or _VALUE_JUNK.search(value) or len(value.split()) > 6 or not toks(value):
                continue
            if set(re.findall(r"[a-z']+", value.lower())) & _PRONOUNS:
                continue   # a phrase built on a pronoun is not a product value
            if attr:
                attr = _clean_attr(attr)
            op, val = value_op(value)
            key = (attr, str(val).lower())
            if key in seen or (out["product"] and str(val).lower() == out["product"]):
                continue
            seen.add(key)
            out["needs"].append({"attr": attr, "op": op, "value": val})
    if out["product"]:
        head = re.escape(out["product"].split()[-1].rstrip("s"))
        for m in re.finditer(head + r"s?,? (?:in|with) (?:a|an|the) (?P<v>[a-z][a-z\-]*(?: [a-z][a-z\-]*){0,2}?) "
                             r"(?P<a>[a-z]+ (?:type|kind)|[a-z]+)(?=[,.;!?]| that\b| which\b| please\b| for\b|$)", low):
            attr, value = _clean_attr(m.group("a")), _clean_value(m.group("v"))
            if value and toks(value) and attr not in _ATTR_JUNK and (attr, value.lower()) not in seen:
                seen.add((attr, value.lower()))
                out["needs"].append({"attr": attr, "op": "eq", "value": value})
    named = [set(toks(f"{n['attr']} {n['value']}")) for n in out["needs"] if n["attr"] and isinstance(n["value"], str)]
    out["needs"] = [n for n in out["needs"] if n["attr"] or not isinstance(n["value"], str)
                    or not any(set(toks(n["value"])) <= w for w in named)]
    for m in re.finditer(r"\b(?:do not|don't|never) (?:choose|pick|select|take|buy|go with) (?P<v>[^,.;]{1,40}?) "
                         r"(?:for|as) (?:the |its |their )?(?P<a>[a-z][a-z0-9_ /&\-]{1,30}?)\s*(?:[.;!]|$)", t, re.IGNORECASE):
        value, attr = _clean_value(m.group("v")), _clean_attr(m.group("a"))
        if value and toks(value) and attr and (attr, value.lower()) not in seen:
            seen.add((attr, value.lower()))
            out["needs"].append({"attr": attr, "op": "ne", "value": value})
    for rx in _AVOID_RES:
        for m in rx.finditer(t):
            phrase = _clean_value(m.group("x"))
            if phrase and not _VALUE_JUNK.search(phrase):
                out["avoid"].append(phrase.lower())
    out["tentative"] = []
    for rx in _TENTATIVE_RES:
        for m in rx.finditer(t):
            gd = {k: (v or "").strip().lower() for k, v in m.groupdict().items()}
            if gd.get("o"):
                if not _VALUE_JUNK.search(gd["o"]) and not set(toks(gd["o"])) <= set(toks(out["product"] or "")) and all(
                        set(toks(gd["o"])) - set(toks(str(n["value"]))) for n in out["needs"]):
                    out["needs"].append({"attr": None, "op": "eq", "value": gd["o"]})
                continue
            if gd.get("g") or (gd.get("v") and not _VALUE_JUNK.search(gd["v"])):
                out["tentative"].append(gd)
    return out


# ---- multi-line requests (2-3 different products, a quantity for each) ---------------------

_QTY_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a couple of": 2,
              "both": 2, "pair": 1, "single": 1, "couple": 2, "trio": 3}


# A-R5 T1: a phrase of order-level words only (closed class) is the order's total, budget or price, not an item
_ORDER_HEADS = {"total", "totals", "subtotal", "sum", "overall", "altogether", "combined", "whole", "everything", "all",
                "budget", "amount", "cost", "costs", "price", "prices", "bill", "spend", "spending", "payment", "order",
                "orders", "purchase", "checkout"}


# A-R7f M1: a measure by its shape alone (no unit vocabulary): never a count or an item
_MEASURE_HELD = re.compile(
    r"\b\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?(?:\s*[x×]\s*\d+(?:\.\d+)?)?\b"   # a dimension: "3 x 5", "2x4", "10 x 12"
    r"|\b\d+\.\d+\s*-?\s*[a-z]+\b"                                              # a decimal and its word: "10.1 inch"
    r"|\b\d+(?:\.\d+)?(?!x\b)[a-z]+\b"                                          # glued: "16gb", "10mm" (not the sign "2x")
    r"|\b\d+\s*-?\s*in\s*-?\s*1\b"                                              # "3 in 1", "2-in-1" (before "a <n>")
    r"|(?<=\ba )\d+(?:\.\d+)?\b|(?<=\ban )\d+(?:\.\d+)?\b", re.IGNORECASE)      # "a 3 seater <item>": one item
_MONEY_TOKEN = re.compile(r"\d\s*¢|[$€£]\s*\d|\b\d+\s*(?:cents?|cts)\b", re.IGNORECASE)   # A-R7 M3


_MONEY_AMOUNT = r"(?:[$€£]\s*\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s*(?:¢|cents?\b|cts\b))"   # A-R7f: A-R5's money words
_MONEY_CLAUSE = re.compile(   # A-R7c C1: "under 2000¢", "for less than $30 each", "within 5000 cents in total"
    r"\s*,?\s*\b(?:for |at |costing |priced )?(?:under|below|at most|up to|within|less than|(?:no|not) more than|"
    r"not over|no higher than|max(?:imum)?(?: of)?)\s+(?:a total of\s+)?" + _MONEY_AMOUNT
    + r"(?:\s+(?:each|in total|in all|overall|for everything|altogether))?", re.IGNORECASE)


def _strip_money_clause(text):
    """A-R7c C1: the request sentence or phrase without its money clause."""
    return " ".join(_MONEY_CLAUSE.sub(" ", text).split()) if isinstance(text, str) else text


def _drop_order_clause(phrase):
    """A-R7c C2: "the total to be and a blue mug" -> "blue mug": a leading order-level part joined by "and" goes."""
    if not isinstance(phrase, str):
        return phrase
    parts = re.split(r"\s+and\s+", phrase)
    while len(parts) > 1 and _order_head(parts[0]):
        parts = parts[1:]
    return re.sub(r"^(?:a |an |the |some )+", "", " and ".join(parts), flags=re.IGNORECASE)


def _mask_measures(text):
    """A-R7 M1: (text with every measure held as one letters-only word, the measures) for _unmask_measures."""
    found = []

    def hold(m):
        found.append(m.group(0))
        return "zqmeasure" + "abcdefghijklmnopqrstuvwxy"[min(len(found) - 1, 24)] + "z"
    return _MEASURE_HELD.sub(hold, text) if isinstance(text, str) else text, found


def _unmask_measures(phrase, found):
    if not found or not isinstance(phrase, str):
        return phrase
    return re.sub(r"zqmeasure([a-y])z", lambda m: found[min("abcdefghijklmnopqrstuvwxy".index(m.group(1)), len(found) - 1)].lower(),
                  phrase)


def _order_head(phrase) -> bool:
    """A-R5 T1: every word of the phrase, function words aside, is an order-level word (a lone "total", say).
    A-R7f M2: a bare number, or a held measure alone, is no item either."""
    ws = [w for w in (words(phrase) if isinstance(phrase, str) else []) if w not in _FUNCTION_WORDS]
    if ws and all(re.fullmatch(r"\d+|zqmeasure[a-y]z", w) for w in ws):
        return True
    return bool(ws) and all(w in _ORDER_HEADS for w in ws)


def _times_sign(text) -> str:
    """A-R6 X1: a multiplication sign next to a count is a plain count ("2 x <item>" -> "2 <item>", "<item> x 2" ->
    "<item>, 2 units"); a dimension, a size word, an x-word and a determiner's modifier ("a 4x ...") are left alone."""
    if not isinstance(text, str):
        return text
    # A-R7 (claude-3 review): "2 x <item>" / "2x <item>" only where a list item starts or right after the request verb;
    # after a comma or "and" only once the text already writes its counts this way ("looking for 4x zoom ..." is no count)
    strong = r"(^|[.;:!?\n•]\s*|\s-\s*|\b(?:get|buy|order|find|want|need)\s+(?:me\s+|us\s+)?)"
    sign = r"(\d+)\s*[x×]\s+(?=[A-Za-z])"
    t = text
    if re.search(strong + sign, t, re.IGNORECASE):
        t = re.sub(strong + sign, r"\1\2 ", t, flags=re.IGNORECASE)
        t = re.sub(r"(,\s*|\band\s+)" + sign, r"\1\2 ", t, flags=re.IGNORECASE)
    # "x 2 <item>": the sign before the count, at the start of a list item (text start, ":", ",", ";", "-", a bullet)
    t = re.sub(r"(^|[:;,\n•]\s*|\s-\s*)[x×]\s*(\d+)\s+(?=[A-Za-z])", r"\1\2 ", t)

    def trail(m):   # "<item> x 2"; not after a word that holds a digit ("a 16GB kit 8GB x 2" is a kit spec)
        before = m.string[:m.start()].split()
        return m.group(0) if before and re.search(r"\d", before[-1]) else f", {m.group(1)} units"
    return re.sub(r"(?<=[A-Za-z])\s+[x×]\s*(\d+)(?=\s*(?:[.,;:!?)]|$|\band\b|\bor\b|\bplease\b))", trail, t)


def _product_in(sentence):
    """The product phrase a sentence introduces, or None."""
    if re.search(r"(cts|cents|messages|moves|steps|website|on hand|shelf|substitute)", sentence, re.IGNORECASE):
        return None
    sentence, measures = _mask_measures(_strip_money_clause(sentence))   # A-R7 M1, A-R7c C1
    for rx in _PRODUCT_RES:
        m = rx.search(sentence)
        if m:
            phrase = _PRODUCT_CUT.sub("", m.group("p")).strip(" ,.?!'")
            phrase = re.sub(r"^(?:something from |some |a |an |the |one |new |another |pair of |"
                            r"\d+ |two |three |four |five )+", "", phrase, flags=re.IGNORECASE)
            if (phrase and len(phrase.split()) <= 6 and not _VALUE_JUNK.search(phrase)
                    and not set(words(phrase)) & _NARRATIVE and not _order_head(phrase)):   # A-R5 T1
                return _unmask_measures(phrase.lower(), measures)
    return None


def capture_key_values(text, product, table, category=None):
    """[(key or None, value)] stated next to listing keys or values. `table`: key words -> values the inspected
    listings show; `category`: the store aisle. Reads "<v> <key>" (a key's head word too when its values hold <v>;
    a 1-word key only when its values use the words of <v>), leading item-phrase words the aisle name lacks,
    "<item> in <v>", and "<v> only" for a listed <v>. Skips a value right after a negation. With no key, the value
    is matched anywhere in the facts."""
    low = " ".join(str(text or "").lower().split())
    seq = re.findall(r"[a-z0-9%+]+(?:[-/][a-z0-9%+]+)*", low)
    ptoks = set(toks(product or ""))
    vwords = {key: {w for v in values for w in toks(v)} for key, values in table.items()}
    stop = _FUNCTION_WORDS | _PRONOUNS | _NARRATIVE
    neg = {"no", "not", "nothing"}
    out, seen = [], set()

    def keep(key, value):
        vt = " ".join(toks(value))
        if vt and vt not in seen:
            seen.add(vt)
            out.append((key, value))

    def holder(value):
        keys = [k for k, values in table.items() if value in values]
        return keys[0] if len(keys) == 1 else None

    for key in table:
        kw = key.split()
        if not kw or set(kw) & stop or len(key) < 3:
            continue
        heads = [kw] + ([kw[:1]] if len(kw) >= 2 else [])
        for hw in heads:
            for i in range(len(seq) - len(hw) + 1):
                if seq[i:i + len(hw)] != hw:
                    continue
                span, j = [], i - 1
                while j >= 0 and len(span) < 3 and seq[j] not in stop:
                    span.insert(0, seq[j])
                    j -= 1
                if j >= 0 and seq[j] in neg or (j >= 1 and seq[j - 1] in neg):
                    continue
                span = [w for w in span if not set(toks(w)) & ptoks]
                if not span or any(w.isdigit() for w in span):
                    continue
                value = " ".join(span)
                if hw is kw and len(kw) >= 2:
                    keep(key, value)
                elif value in table[key] or (hw is kw and set(toks(value)) <= vwords[key]):
                    keep(key, value)
    pw = re.findall(r"[a-z0-9%+]+(?:[-/][a-z0-9%+]+)*", (product or "").lower())
    cw = set(toks(category or ""))
    if cw and len(pw) >= 2:   # leading item-phrase words the aisle name lacks: "<v> <item>"
        lead = []
        for w in pw:
            if set(toks(w)) & cw or w in stop:
                break
            lead.append(w)
        if lead and len(lead) < len(pw) and set(toks(" ".join(pw[len(lead):]))) & cw:
            value = " ".join(lead)
            keep(holder(value), value)
    if product:   # "<item> in <v>"
        m = re.search(re.escape(product.lower()) + r" in (?:a |an )?([a-z][a-z-]*(?: [a-z][a-z-]*)?)(?=[ ,.;:!?]|$)", low)
        if m:
            first = m.group(1).split()[0]
            if first not in stop and first not in ("the", "your"):
                cand = m.group(1) if holder(m.group(1)) else first
                keep(holder(cand), cand)
    for i, w in enumerate(seq):   # "<v> only" with a listed value
        if w == "only" and i >= 1:
            for n in (2, 1):
                if i - n >= 0:
                    value = " ".join(seq[i - n:i])
                    if holder(value) and not (i - n - 1 >= 0 and seq[i - n - 1] in neg):
                        keep(holder(value), value)
                        break
    return out


_COUNT_WORDS = {"of them", "units", "pieces", "items", "ones", "packs", "sets"}   # count words, never an item


def _r9_count_only(phrase) -> bool:
    """A-R8b G2: the phrase is only count words and numbers ("packs", "3 pcs", "1 set"): a restated count."""
    ws = [w for w in words(phrase or "") if not (w.isdigit() or w in _NUMBER_WORDS)]
    return bool(ws) and all(w in _COUNT_WORDS or w + "s" in _COUNT_WORDS or re.fullmatch(_COUNT_NOUN + r"|of|them|these|those|\d*pcs", w)
                            for w in ws)


_R9_PER_ITEM = re.compile(r"\b(?:each|apiece|per (?:item|unit|piece))\b", re.IGNORECASE)


def _r9_total_budget(led) -> bool:
    """A-R8b G1: the budget caps the whole basket: it came from a shopper's line, or the clause holding the amount
    says total / altogether / overall and nothing in it makes the amount per item."""
    if led.notes.get("budget_event"):
        return True
    t = " ".join(str(led.query or "").split())
    for m in _MONEY_RE.finditer(t):
        got = parse_money(m.group(0))
        if not got or got[0][0] != led.budget:
            continue
        lo = max(t.rfind(c, 0, m.start()) for c in ".;!?,") + 1
        hi = min([j for j in (t.find(c, m.end()) for c in ".;!?,") if j >= 0] or [len(t)])
        clause = t[lo:hi]
        return bool(re.search(r"\b(?:total|altogether|overall)\b", clause, re.IGNORECASE)) and not _R9_PER_ITEM.search(clause)
    return False


def line_quantity(text, product):
    """Units wanted for one line: "three of them", "2 x", "a pair", "I need two <product>"."""
    text = _glued_count(text)
    head = set(toks(product or "")) | set(words(product or "")) | _COUNT_WORDS
    head_rx = "|".join(re.escape(h) for h in sorted(head, key=len, reverse=True) if h)
    num = r"(\d+|one|two|three|four|five|six|a couple of|both)"

    def wanted(pattern):
        for m in re.finditer(pattern, text, re.IGNORECASE):
            clause = re.split(r"[.!?;]", text[:m.start()])[-1]
            rest = text[m.end():].split("?", 1)
            if len(rest) > 1 and not re.search(r"[.!;]", rest[0]) and not re.search(r"\b(?:could|can|would|will) you\b",
                                                                                     clause, re.IGNORECASE):
                continue   # inside a question (Q2c: "could you get me two ...?" is a request)
            if set(clause.lower().split()[-4:]) & {"own", "owned", "owns", "had", "wear", "wore", "bought", "already"}:
                continue   # a count the shopper has, not one they want
            if not _order_count(text, m.start(), m.end()):
                continue   # F-033: a count in a clause about the past, or an attribute's value
            return m
        return None
    m = wanted(num + r"\s+(?:[a-z\-]+\s+){0,3}?(?:" + head_rx + r")s?\b") if head_rx else None
    if not m:
        m = wanted(num + r"\s+(?:of (?:them|those|these)|units|pieces|identical|separate|matching|copies)\b")
    if not m:
        m = wanted(r"\b(?:quantity|qty|count)(?: of)?\s*(?:is|:|=)?\s*(\d+|one|two|three|four|five)\b")
    if not m:
        return 1
    word = m.group(1).lower()
    value = int(word) if word.isdigit() else _QTY_WORDS.get(word, 1)
    return value if 1 <= value <= 20 else 1


def _glued_count(text):
    """Q2b: "2x <item>" / "2 x <item>" at a phrase start or right after the request verb reads as "2 <item>"."""
    return re.sub(r"(^|[.,;:!?]\s*|\b(?:get|buy|order|find|want|need)\s+(?:me\s+|us\s+)?)(\d+)\s*x\s+(?=[a-z])",
                  r"\1\2 ", str(text or ""), flags=re.IGNORECASE)


def product_count(text, product):
    """Q2a: the units wanted of `product` (a phrase the model read): a count directly before the product phrase, only
    the product's own words between, or "<count> of them"; 1 when none."""
    text = _glued_count(text)
    pw = [w for w in re.findall(r"[a-z0-9'\-]+", str(product or "").lower()) if w not in _STOP]
    if not pw:
        return 1
    own = {w.rstrip("s") for w in pw}
    num = r"(\d+|one|two|three|four|five|six|a couple of|both)"
    for m in re.finditer(r"(?<![\w\-])" + num + r"\s+((?:[a-z0-9'\-]+\s+){0,4}?)" + re.escape(pw[-1].rstrip("s")) + r"s?\b"
                         r"|(?<![\w\-])" + num + r"\s+of (?:them|those|these)\b", text, re.IGNORECASE):
        word = (m.group(1) or m.group(3)).lower()
        between = (m.group(2) or "").split()
        before = text[:m.start()].split()
        if (word in own or any(w.lower().rstrip("s") not in own for w in between)
                or (before and (before[-1].lower() == "of" or re.search(r"\d", before[-1])
                                or (len(before[-1]) >= 2 and before[-1].isupper())))):
            continue   # a number in the product, after "of", or after a model token is no count
        rest = text[m.end():].split("?", 1)
        clause = re.split(r"[.!?]", text[:m.start()])[-1]
        if len(rest) > 1 and not re.search(r"[.!;]", rest[0]) and not re.search(r"\b(?:could|can|would|will) you\b",
                                                                                 clause, re.IGNORECASE):
            continue   # inside a question
        if not _order_count(text, m.start(), m.end()):
            continue   # F-033: a count in a clause about the past, or an attribute's value
        value = int(word) if word.isdigit() else _QTY_WORDS.get(word, 1)
        return value if 1 <= value <= 20 else 1
    return 1


_PRONOUNS = {"i", "me", "my", "mine", "you", "your", "we", "us", "our", "they", "them", "their", "it",
             "he", "she", "him", "her", "something", "anything", "everything", "nothing", "whatever", "someone",
             "anyone", "somebody", "anybody"}
_NOT_PRODUCT = {"one", "next", "few", "only", "ever", "thing", "stuff", "something", "anything", "option", "part", "bit", "lot", "time", "way",
                "little", "couple", "while", "day", "week", "month", "year", "moment", "second", "minute", "hour"}


_R9_ITEM = r"([A-Za-z][A-Za-z'&\- ]{2,40}?)(?=\s*(?:,|\band\b|\.|$))"


def _r9_items(held):
    """R9 P1: [(start, end, count, phrase, soft)] of one masked sentence: counted items, then "a/an <item>" singles
    list-joined to an accepted item (transitively); a masked "a/an <n> <word>" item counts as one and anchors."""
    counted = [(m.start(), m.end(), m.group(1), m.group(2), False) for m in
               re.finditer(r"\b(\d+|one|two|three|four|five|six)\s+" + _R9_ITEM, held)]
    for m in re.finditer(r"\b(?:an?)\s+(zqmeasure[a-y]z\s+" + _R9_ITEM[1:], held, re.IGNORECASE):
        if not any(s < m.end() and m.start() < e for s, e, _, _, _ in counted):
            counted.append((m.start(), m.end(), "1", m.group(1), False))
    if not counted:
        return []
    anchors = [c for c in counted if not set(words(c[3])) & _NARRATIVE]
    cands = [(m.start(), m.end(), "1", m.group(1), True) for m in
             re.finditer(r"\b(?:an?)\s+" + _R9_ITEM, held, re.IGNORECASE)
             if not any(s < m.end() and m.start() < e for s, e, _, _, _ in counted)
             and not set(words(m.group(1))) & (_NARRATIVE | _NOT_PRODUCT)]
    accepted, grew = list(anchors), True
    while grew:   # transitive: "2 mugs, a teapot and a tray"
        grew = False
        for c in cands:
            if c in accepted:
                continue
            if any(re.fullmatch(r"\s*(?:,|,?\s*and)\s*", held[e:c[0]] if e <= c[0] else held[c[1]:s])
                   for s, e, _, _, _ in accepted):
                accepted.append(c)
                grew = True
    singles = [c for c in accepted if c[4]]
    return sorted(counted + singles)


def split_lines(text) -> list:
    """Order lines: [{product, sentences}]. A sentence naming a new product opens a line; a
    sentence restating a known product (shared word) and need sentences join the latest line."""
    lines, current = [], None
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(str(text or "").split())):
        held, measures = _mask_measures(sentence)   # A-R7 M1: a measure is never "<n> <item>"
        found = _r9_items(held)   # R9 P1
        soft = {_unmask_measures(ph, measures).strip().lower() for _, _, _, ph, sf in found if sf}
        items = [(c, _unmask_measures(ph, measures)) for _, _, c, ph, _ in found]
        items = [(c, ph) for c, ph in items if not re.fullmatch(_COUNT_NOUN, ph.strip().lower())   # "three units" restates a count
                 and not _VALUE_JUNK.search(ph) and not _order_head(ph)]   # A-R5 T1: "<n> in total" is no item
        if re.search(r"(cts|cents|messages|moves|steps|website|on hand|shelf|substitute)", sentence, re.IGNORECASE):
            items = []   # the budget and the limits are not order lines
        if len(items) >= 2 and not lines:   # "I want <n> <product>, and <m> <other product>."
            for count, phrase in items:
                lines.append({"product": phrase.strip().lower(), "sentences": [], "quantity": parse_count(count) or 1,
                              "soft": phrase.strip().lower() in soft})   # R9 P1
            current = None
            continue
        scoped = re.match(r"\s*for (?:the )?([^,]{2,60}),", sentence, re.IGNORECASE)
        if scoped and lines:
            target = set(toks(scoped.group(1)))
            match = max(lines, key=lambda l: len(target & set(toks(l["product"]))))
            if target & set(toks(match["product"])):
                match["sentences"].append(sentence)
                continue
        product = _product_in(sentence)
        if product and set(toks(product)) & _NOT_PRODUCT:
            product = None
        if product and lines and not (any(rx.search(sentence) for rx in _NEED_RES)
                                      or re.search(r"\b(?:\d+|one|two|three|four|five|six)\s+" + _COUNT_NOUN + r"\b", sentence, re.IGNORECASE)):
            product = None   # a further line must state a need or a quantity in its own sentence
        if product:
            ptoks = set(toks(product))
            same = next((l for l in lines if ptoks & set(toks(l["product"]))), None)
            if same is None:
                current = {"product": product, "sentences": [sentence]}
                lines.append(current)
            else:
                current = same
                current["sentences"].append(sentence)
        elif current is not None:
            current["sentences"].append(sentence)
    return lines


# >>>>>>>> C0017 claude-2: need matcher + reference resolver (BEGIN) >>>>>>>>
# ---- matching a need against a product (C0017/C0028: exact canonical values, keys from the facts) ----
# A need is MET only by the value the world wrote ("modern" is not "mid-century modern": a race verdict
# shows it). The need's words inside a longer value ("wood" in "solid wood (walnut, oak)") are UNCONFIRMED
# (None): such a product stays a candidate, ranked after confirmed ones (a race pass did). Numbers are
# compared only when the need and the fact write the same unit; a different unit leaves the need
# unverified (None), so a verified candidate is preferred and the model judge may settle it. No unit,
# attribute or synonym tables: property names come from the candidates' own fact keys.

_SEL_STOP = {"a", "an", "the", "of", "in", "on", "to", "with", "and", "or", "that", "which", "is", "are", "it", "its",
             "from", "by", "for", "at", "as", "this", "these", "not", "item", "product", "listing", "version",
             "one", "set", "pair", "sold", "sell", "offer", "offers", "made", "has", "have", "uses", "use", "comes",
             "come"}


def canon(text) -> str:
    """Lower case, '_'/'-' as spaces, punctuation dropped (decimal points and % kept), plurals folded."""
    t = str(text if text is not None else "").lower().replace("’", "'").replace("'s ", " ")
    t = re.sub(r"(\d)\s*[x×]\s*(\d)", r"\1x\2", t)
    t = re.sub(r"[_\-/]+", " ", t)
    return " ".join(stem(x) for x in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", t))


def _alternatives(value, key=None) -> list:
    """The values one fact offers: 'powder-coated or natural' -> both; 'black, white' -> both;
    '100% cotton' -> also 'cotton'; key 'assembly required' with value 'no' -> 'no assembly required'."""
    raw = re.sub(r"\s*\([^)]*\)", "", str(value if value is not None else ""))
    parts = [raw] + re.split(r"\s*(?:,|;|/|\||\bor\b)\s*", raw)
    if key is not None and len(raw.split()) <= 2:
        k, low = re.sub(r"[._]", " ", str(key)), raw.strip().lower()
        if low in ("yes", "true"):
            parts.append(k)
        elif low in ("no", "false", "none"):
            parts.append("no " + k)
        else:
            parts += [f"{raw} {k}", f"{k} {raw}"]
    out = []
    for p in parts:
        c = canon(p)
        d = re.sub(r"^(?:\d+(?:\.\d+)?%\s*)+", "", c)
        for x in (c, d):
            if x and x not in out:
                out.append(x)
    return out


_MEASURE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?:\s*(?:-|to|~)\s*(\d+(?:\.\d+)?))?\s*(\"|[a-z]+)?", re.IGNORECASE)


def _unit(text):
    """A unit as written, normalised only generically (case, spacing, a plural ending); None when absent."""
    u = str(text or "").strip().lower()
    if not u or u in _LANG_STOP:
        return None
    if len(u) == 3 and u.endswith("s") and not u.endswith("ss"):
        return u[:-1]   # F1: a short unit word's plural ending
    return u if u == '"' else stem(u)


def _fractions(text):
    """F2: mixed and simple fractions as decimals: "1 3/8", "1-3/8" -> 1.375; "1/2" -> 0.5."""
    text = re.sub(r"(?<![\d./])(\d+)[\s\-]+(\d+)/(\d+)(?![\d/])",
                  lambda m: "%g" % (int(m.group(1)) + int(m.group(2)) / max(1, int(m.group(3)))), str(text))
    return re.sub(r"(?<![\d./])(\d+)/(\d+)(?![\d/])", lambda m: "%g" % (int(m.group(1)) / max(1, int(m.group(2)))), text)


def measures(text) -> list:
    """Every number in `text` as (low, high, unit as written or None)."""
    text = re.sub(r"\d+(?:\.\d+)?\s*[x×*]\s*\d+(?:\.\d+)?(?:\s*[x×*]\s*\d+(?:\.\d+)?)?", " ",   # "3440 x 1440" is a
                  str(text if text is not None else "").lower())                                      # size, not two numbers
    text = _fractions(text)   # F2
    return [(float(m.group(1)), float(m.group(2) or m.group(1)), _unit(m.group(3))) for m in _MEASURE.finditer(text)]


def _num_hit(op, want, have, key=None):
    """True/False for a numeric need against a fact; None when no number there carries the need's unit
    as written (a unitless number counts only when the fact key itself writes that unit). The rules
    never convert units: a different unit is unverified."""
    w = measures(want if not isinstance(want, (list, tuple)) else " ".join(map(str, want)))
    if not w:
        return None
    got = measures(have)
    if not got:
        return False
    unit = w[0][2]
    if unit:
        same = [g for g in got if g[2] == unit]
        if not same and unit in {_unit(t) for t in re.findall(r"[a-z]+|\"", str(key or "").lower())}:
            same = [g for g in got if g[2] is None]
    else:
        same = [g for g in got if g[2] is None] or got
    if not same:
        return None
    eps = 1e-6 + 1e-3 * abs(w[0][0])
    if op == "between":
        lo, hi = w[0][0], (w[1][0] if len(w) > 1 else w[0][1])
        return any(g[0] >= lo - eps and g[1] <= hi + eps for g in same)
    if op == "ge":
        return any(g[1] >= w[0][0] - eps for g in same)
    if op == "gt":
        return any(g[1] > w[0][0] + eps for g in same)
    if op == "le":
        return any(g[0] <= w[0][0] + eps for g in same)
    if op == "lt":
        return any(g[0] < w[0][0] - eps for g in same)
    return any(abs(g[0] - w[0][0]) <= eps and abs(g[1] - g[0]) <= eps for g in same)    # eq / near


def _same_word(a, b) -> bool:
    """Generic spelling match: equal, or a one-letter spelling variant (one letter inserted in a 5+
    letter word, or one letter changed in a 4+ letter word). Digits never vary."""
    if a == b:
        return True
    if re.search(r"\d", a + b):
        return False
    s, l = sorted((a, b), key=len)
    if len(s) >= 5 and len(l) == len(s) + 1:
        it = iter(l)
        return all(c in it for c in s)
    if len(s) >= 4 and len(l) == len(s):
        return sum(x != y for x, y in zip(s, l)) == 1
    return False


def _attr_tokens(attr):
    return set(toks(re.sub(r"\([^)]*\)", " ", str(attr)))) if attr else set()     # "(watch line)" is a scope note


def _key_parts(key, attr_tokens=()):
    """(name words, noise words) of a fact key: the words of its last dotted segment are the name;
    earlier segments (a namespace), parenthesised notes, short codes and numbers are noise. Name
    words that abbreviate or respell a word of `attr_tokens` are read as that word."""
    raw = re.sub(r"\([^)]*\)", " ", str(key))
    segs = [x for x in raw.split(".") if x.strip()] or [raw]
    name = toks(segs[-1].replace("_", " "))
    noise = {t for x in segs[:-1] for t in toks(x.replace("_", " "))}
    noise |= set(toks(" ".join(re.findall(r"\(([^)]*)\)", str(key)))))
    noise |= {t for t in name if len(t) <= 2 or re.search(r"\d", t)}
    fixed = {next((a for a in attr_tokens if _same_word(t, a)), t) for t in name if t not in noise}
    return fixed, noise - fixed


def tentative_values(ledger) -> list:
    """First words of the request's article-led "<value> <property>" groups ("in the X style"): used
    as search and inspection hints before candidates show which fact key they belong to."""
    return [t["g"].split()[0] for t in ledger.notes.get("tentative") or []
            if t.get("g") and t.get("art") and len(t["g"].split()) >= 2 and not _VALUE_JUNK.search(t["g"])]


def plain_key(key) -> str:
    """A fact key as a shopper would name it: the words of its last dotted segment, without codes."""
    last = [x for x in re.sub(r"\([^)]*\)", " ", str(key)).split(".") if x.strip()]
    found = re.findall(r"[a-z][a-z']+", (last[-1] if last else str(key)).replace("_", " ").lower())
    return " ".join(found) or str(key).lower()


def _key_tokens(key):
    name, noise = _key_parts(key)
    return name | noise


def key_score(attr, key) -> float:
    """How well a fact key names `attr`: 3 same words, 2.x key adds only noise (a namespace, codes),
    1.x key is a shorter form (a two-word key for a three-word attribute) or adds real words, 0 unrelated."""
    a = _attr_tokens(attr)
    k, noise = _key_parts(key, a)
    if not a or not k:
        return 0.0
    if a == k:
        return 3.0 if not noise else 2.5 - 0.1 * len(noise)
    if a < k:
        return 1.5 - 0.1 * len(k - a)
    if k < a:
        return 1.2 + 0.1 * len(k)
    return 0.0   # a shared word alone ("sleeve length" for "sleeve type") names another attribute


def _is_options(key) -> bool:
    return bool(set(words(key)) & {"options", "option"})


def related_facts(product, attr):
    """(key, value) facts that name `attr`, best-named first: the same words, a longer key (a dotted
    key whose last segment is the attribute) or a shorter one. A key sharing only one word with a
    two-word attribute names another attribute. Option lists are excluded."""
    if not attr:
        return []
    a, own = _attr_tokens(attr), set(toks(str(product.get("category") or "")))
    scored = []
    for k, v in (product.get("props") or {}).items():
        if _is_options(k):
            continue
        name, noise = _key_parts(k, a)
        if name and name < a and not all(any(_same_word(w, c) or (min(len(w), len(c)) >= 4 and (w.startswith(c) or c.startswith(w)))
                                             for c in own | noise) for w in a - name):
            continue   # the words the key lacks qualify the attribute ("seat" height): another attribute
        scored.append((key_score(attr, k), k, v))
    return [(k, v) for s, k, v in sorted(scored, key=lambda x: -x[0]) if s >= 1.0]


_NEGATED = re.compile(r"\b(?:not|isn't|aren't|nor|never)\s+(?:a |an |the |in |of )?[^,.;]*", re.IGNORECASE)


def own_text(product, generated=False) -> str:
    """Caption, summary and the listing's description of ITSELF: redirect, rule and disclaimer
    sentences removed, and negated phrases cut ("it is not a pack", "not 4 channel"). generated=True
    keeps only the store's generated sentences (those naming the seller), not raw marketplace text."""
    seller = str(product.get("seller") or product.get("brand") or "").lower()
    parts = [] if generated else [str(product.get("title") or "")]
    for sentence in re.split(r"(?<=[.!?])\s+", str(product.get("text") or "")):
        if parse_redirects(sentence) or parse_pairings(sentence) or re.search(
                r"\b(?:is|are) not (?:the|it)\b|elsewhere|different \w+, and this", sentence, re.IGNORECASE):
            continue
        if generated and not (seller and seller in sentence.lower()):
            continue
        parts.append(_NEGATED.sub(" ", sentence))
    return " ".join(p for p in parts if p)


def _phrase_in(value, text) -> bool:
    v, t = canon(value), " " + canon(text) + " "
    return bool(v) and (" " + v + " ") in t


_NUM_OPS = {"ge", "le", "gt", "lt", "between", "near"}


def _qualifiers_only(tokens, skip):
    """True when every token outside the index range `skip` is a number, a code with digits, '%', or
    the word written right after a number (its unit as written)."""
    for j, x in enumerate(tokens):
        if j in skip:
            continue
        if not (re.search(r"\d|%", x) or (j > 0 and re.search(r"\d", tokens[j - 1]))):
            return False
    return True


def _any(results):
    """True if any True, else None if any None, else False."""
    results = list(results)
    return True if True in results else (None if None in results else False)


def _value_hit(op, value, have, key=None):
    """One need value against one fact value.
    True: the world's value, or it plus qualifiers only ("<value> <spec>" for <value>, "<n>% <value>").
    None: a near form, the need's phrase inside a longer value ("side seam invisible zip", "solid
    wood", "mid-century modern"), all its words with one inserted ("invisible back zip"), a one-letter
    respelling, or a number in a different unit. Confirmed candidates rank first, so an exact match
    wins when one exists. False: anything else."""
    if op == "in":
        vals = value if isinstance(value, (list, tuple)) else re.split(r"\s*\|\s*", str(value))
        return _any(_value_hit("eq", v, have, key) for v in vals if str(v).strip())
    if op in _NUM_OPS:
        return _num_hit("between" if op == "between" else op, value, have, key)
    if op == "contains":
        if _phrase_in(value, have):
            return True
        want = canon(value)
        return None if " " not in want and any(_same_word(want, x) for x in canon(have).split()) else False
    want = canon(value)
    if not want:
        return False
    alts = _alternatives(have, key)
    if want in alts:
        return True
    other_unit = False
    if re.search(r"\d", want):
        hit = _num_hit("eq", value, have, key)
        if hit is not None:
            return hit
        other_unit = bool(measures(have))      # numbers there, but in another unit: unverified
    w = want.split()
    near = False
    for a in alts:
        t = a.split()
        at = [i for i in range(len(t) - len(w) + 1) if t[i:i + len(w)] == w]
        if at and len(t) > len(w):
            if _qualifiers_only(t, range(at[0], at[0] + len(w))):
                return True
            near = near or len(t) <= 6
    if near:
        return None
    if len(w) > 1 and any(set(w) <= set(a.split()) and len(a.split()) <= 6 for a in alts):
        return None     # a word inserted inside the phrase: "invisible back zip" failed, "simple leg assembly" passed
    if len(w) == 1 and any(len(a.split()) == 1 and _same_word(want, a) for a in alts):
        return None     # a respelling: unconfirmed, never folded by a table
    return None if other_unit else False


def _word_form(a, b) -> bool:
    """Two forms of one word: no digits, a shared start of 5+ letters that leaves at most 2 letters of the shorter."""
    if re.search(r"[0-9]", a + b) or min(len(a), len(b)) < 5:
        return False
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n >= max(5, min(len(a), len(b)) - 2)


def need_status(need, product):
    """True (met), False (contradicted) or None (unconfirmed) for one need on one product.
    Facts whose key names the attribute decide. Without such a fact the row is ABSENT: the listing's
    generated description, other facts or the caption may confirm the value; otherwise it stays
    unconfirmed, never contradicted (the store holds normalized values a listing may not show).
    A listing that redirects the attribute elsewhere is still judged on its own facts."""
    op = getattr(need, "op", "eq")
    value = need.value
    if value is None or value == "" or op == "ref":
        return None
    negative = op in ("avoid", "ne", "not_contains")
    base = {"avoid": "contains", "ne": "eq", "not_contains": "contains"}.get(op, op)
    props = product.get("props") or {}
    if op == "avoid" or not need.prop:
        blob = " ".join(f"{re.sub(r'[._]', ' ', str(k))} {v}" for k, v in props.items() if not _is_options(k))
        strong = _phrase_in(value, blob) or _phrase_in(value, own_text(product, generated=True))
        weak = strong or _phrase_in(value, own_text(product)) or _phrase_in(
            value, " ".join(str(v) for k, v in props.items() if _is_options(k)))
        if negative:
            return not weak
        return True if strong else (None if weak or not props else False)   # a caption-only hit is unconfirmed
    facts = related_facts(product, need.prop)
    if facts:
        hits = [(_value_hit(base, value, v, k), key_score(need.prop, k)) for k, v in facts]
        hit = _any(h for h, _ in hits)
        if hit is not True and base == "eq" and isinstance(value, str):   # M5: the value inside the property's own value
            want = canon(value).split()
            for (h, sc), (k, v) in zip(hits, facts):
                have = canon(v).split()
                if sc >= 3.0 and want and (any(have[i:i + len(want)] == want for i in range(len(have) - len(want) + 1))
                                           or (len(want) == 1 and any(_word_form(want[0], x) for x in have))):
                    hit = True   # "light <v>", a list holding <v>, "60% <v> ...", a word form ("<stem>ed" for "<stem>y")
                    break
        if (hit is None and base in _NUM_OPS and any(sc >= 3.0 for _, sc in hits)
                and all(h is False for h, sc in hits if sc >= 3.0)):
            hit = False   # S2-F2: the facts named exactly after the attribute contradict the bound in its own unit
        if hit is not True and CONFIG.get("match_caption_override") and _loose_hit(base, value, need.prop, product, text_only=True):
            hit = True
    else:
        hit = _loose_hit(base, value, need.prop, product)
        if hit is False:
            hit = None      # an absent row is not a contradiction
    if negative:
        return {True: False, False: True}.get(hit)
    return hit


def _loose_hit(op, value, attr, product, text_only=False):
    """A need with no fact named after its attribute. True: the listing's generated description
    (the sentences naming its seller, which restate world attributes), another fact (numbers only
    with the same unit) or the caption gives the value. None: only option text or the marketing
    summary shows it (the grader ignores the summary). False: nowhere."""
    own = [x for x in re.split(r"\s*;\s*", own_caption_facts(product, attr)) if x]
    if any(_value_hit(op, value, x) is True for x in own):
        return True
    props = product.get("props") or {}
    if op in _NUM_OPS:
        if not text_only:
            w = measures(value)
            if w and w[0][2]:
                for k, v in props.items():
                    if not _is_options(k) and any(g[2] == w[0][2] for g in measures(v)) and _num_hit(op, value, v, k):
                        return None   # another attribute's number: unconfirmed
        return False
    vals = value if isinstance(value, (list, tuple)) else (str(value).split("|") if op == "in" else [value])
    if any(_phrase_in(x, own_text(product, generated=True)) or _phrase_in(x, product.get("title") or "") for x in vals):
        return True
    if not text_only and any(_value_hit("eq", x, v, k) is True for k, v in props.items() if not _is_options(k) for x in vals):
        return True
    options = " ".join(str(v) for k, v in props.items() if _is_options(k))
    if any(_phrase_in(x, options) or _phrase_in(x, own_text(product)) for x in vals):
        return None
    return False


def own_caption_facts(product, attr) -> str:
    """Values the listing's generated description gives for `attr` ("in the <value> <attr>", "a
    <attr> of <n> <unit>", "made of <value>"), ';'-joined."""
    text, out = own_text(product, generated=True), []
    a = canon(attr)
    if not a:
        return ""
    low = canon(text)
    for m in re.finditer(r"([a-z0-9.%]+(?: [a-z0-9.%]+)?) " + re.escape(a) + r"\b", low):
        out += [m.group(1), m.group(1).split()[-1]]
    for m in re.finditer(r"\b" + re.escape(a) + r" (?:of |is |in )?([a-z0-9.%]+(?: [a-z0-9.%]+)?)", low):
        out += [m.group(1), m.group(1).split()[0]]
    out += [m.group(1) for m in re.finditer(r"made (?:of|from) ([a-z0-9]+(?: [a-z0-9]+)?)", low)]
    return "; ".join(out)


def value_of(product, attr, vocab=()):
    """The listing's own value for `attr`: the best-named fact (when it lists several values, the
    one the listing's own description names), else its own description ("a capacity of 1 tb",
    "the color red", "in the regular size class"), else a word the description gives ("made of X",
    "in X") that other listings use as a value of `attr` (`vocab`: values seen at runtime)."""
    facts = related_facts(product, attr)
    desc = canon(_NEGATED.sub(" ", " ".join(s for s in re.split(r"(?<=[.!?])\s+", str(product.get("text") or ""))
                                             if not parse_redirects(s))))
    if facts:
        exact = [v for k, v in facts if key_score(attr, k) >= 3.0]
        value = str((exact or [facts[0][1]])[0])
        alts = [a for a in _alternatives(value)[1:] if a != canon(value)]
        named = [a for a in alts if (" " + a + " ") in (" " + desc + " ")]
        return named[0] if len(alts) > 1 and len(named) == 1 else value
    low, a = own_text(product).lower(), re.escape(str(attr).lower())
    m = re.search(r"\b" + a + r"(?: of| is|:| as)? (?:the |a |an )?([a-z0-9.%\-]+(?: (?!and\b|with\b)[a-z0-9.%\-]+){0,2}?)(?=[,.;]|$| and | with )", low)
    if m:
        return m.group(1)
    m = re.search(r"\b(?:in|of|with|has|is) (?:the |a |an )?([a-z0-9.%\-]+(?: [a-z0-9.%\-]+)?) " + a + r"\b", low)
    if m:
        return m.group(1)
    known = {canon(v) for v in vocab or ()}
    for m in re.finditer(r"\b(?:made of|made from|in|comes in|offered in|is) ([a-z0-9\-]+)\b", low):
        if canon(m.group(1)) in known:
            return m.group(1)
    title = " " + canon(product.get("title") or "") + " "
    for m in re.finditer(r"\b(?:comes in|offered in|in) ([a-z][a-z\-]{2,})\b", low):
        if canon(m.group(1)) not in _STOP and " " + canon(m.group(1)) + " " in title:
            return m.group(1)    # the listing's own title says it too
    return None


def paired_value(product, attr, value):
    """Apply the listing's value-pairing rule for `attr` (if any) to `value`: when the last listing of
    a reference chain pairs value A with value B, the reference fixes B."""
    for rule in parse_pairings(product.get("text")):
        if rule["attr"] and key_score(attr, rule["attr"]) <= 0 and not (_attr_tokens(rule["attr"]) & _attr_tokens(attr)):
            continue
        hit = rule["pairs"].get(canon(value))
        if hit is None:
            hit = next((w for v, w in rule["pairs"].items() if v in _alternatives(value)), None)
        if hit is not None:
            return hit
    return value


# ---- settling unverified values with the judge model (C0045) ------------------------------------
# Another unit or another wording leaves a need unverified or unmet: the rules never convert or equate.
# One judge call per need (at most CONFIG["compare_max_calls"] per episode) lists which candidate values
# satisfy it, so the knowledge stays in the model. Without a model nothing changes.

def need_key(need) -> str:
    return f"{need.prop}|{getattr(need, 'op', 'eq')}|{need.value}"


def compare_plan(needs, products, done=(), budget=2) -> list:
    """[(key, need, values)] for at most `budget` needs worth one judge call each: needs naming an
    attribute whose candidates show values that do not match exactly. Needs that no candidate meets
    come first, then those with the most unverified candidates."""
    plan = []
    for need in needs:
        if not need.prop or need.value in (None, "") or getattr(need, "op", "eq") in ("ref", "avoid", "ne", "not_contains"):
            continue
        key = need_key(need)
        if key in done:
            continue
        met = unverified = 0
        values = {}
        for p in products:
            facts = related_facts(p, need.prop)
            if not facts:
                continue
            status = need_status(need, p)
            met += status is True
            unverified += status is None
            if status is not True:
                for k, v in facts:
                    values.setdefault(canon(v), str(v).strip())
        if values:
            plan.append(((met > 0, -unverified), key, need, list(values.values())[:24]))
    plan.sort(key=lambda x: x[0])
    return [(key, need, values) for _, key, need, values in plan[:max(0, budget)]]


def judged_status(need, product, equivalents=None):
    """need_status with the judge model's answer for this need applied: a value it listed as
    satisfying is met. For a numeric need (a unit comparison) an unverified value it was shown and did
    not list is not met; a near wording it did not list stays unverified (ranked after met ones)."""
    status = need_status(need, product)
    eq = (equivalents or {}).get(need_key(need))
    if not eq or status is not None or not need.prop:
        return status   # MG2: the model settles only what the rules leave unverified
    vals = [canon(v) for k, v in related_facts(product, need.prop)]
    if any(v in eq["yes"] for v in vals):
        return True
    return status


# ---- selecting the listing a reference names ----------------------------------------------------

def _listing_tokens(product) -> set:
    """Words of the listing's VALUES, own description and category (fact keys are not evidence)."""
    facts = " ".join(str(v) for k, v in (product.get("props") or {}).items() if not _is_options(k))
    return set(canon(" ".join([facts, own_text(product), str(product.get("category") or "")])).split())


_SEG_SPLIT = re.compile(r"\s*(?:,|;|:|\(|\)|\band\b|\bwith\b|\bthat (?:is|are|has|have|uses)\b|\bwhich (?:is|are|has)\b|"
                        r"\bin\b|\bat\b|\bmade of\b|\bsold in\b)\s*", re.IGNORECASE)


def _segment_score(seg, product, have, mine, drop):
    """One selector segment ("<key words> <value>", or the value alone) against a listing: 1 shown,
    0.5 near form, 0 absent, -1 contradicted by the fact it names."""
    st = set(seg.split())
    best_key, best_n, floor = None, 0, None
    for k, v in (product.get("props") or {}).items():
        kt = _key_parts(k)[0]
        if not kt - {"type", "kind"}:      # "product type" names nothing on its own
            continue
        if kt and kt <= st and len(kt) > best_n and not _is_options(k):
            best_key, best_n = k, len(kt)
    if best_key is not None:
        rest = " ".join(t for t in seg.split() if t not in _key_tokens(best_key) and t not in drop)
        if rest:
            hit = _value_hit("eq", rest, product["props"][best_key], best_key)
            if hit is False:   # H7: the value shown elsewhere on the page is a near form, not a contradiction
                elsewhere = " ".join(str(v) for k, v in (product.get("props") or {}).items() if k != best_key and not _is_options(k))
                if _value_hit("contains", rest, elsewhere + " " + own_text(product)) or any(
                        abs(w[0] - g[0]) <= 1e-3 * max(1.0, abs(w[0])) and w[2] == g[2]
                        for w in measures(rest) for g in measures(elsewhere + " " + own_text(product))):
                    hit = None
            if hit is not None:
                return 1.0 if hit else -1.0
            floor = 0.5     # a near form or another unit: the listing's own words may still show it
    words_ = [t for t in seg.split() if not re.match(r"^\d", t) and t not in drop]
    nums = measures(seg)
    got = sum(1 for t in words_ if t in have)
    for w in nums:      # the same number in the same unit as written (or one side unitless)
        got += any(abs(w[0] - g[0]) <= 1e-3 * max(1.0, abs(w[0])) and (w[2] == g[2] or w[2] is None or g[2] is None) for g in mine)
    total = len(words_) + len(nums)
    score = got / total if total else None
    return score if floor is None else max(floor, score or 0.0)


def selector_score(ref, product) -> float:
    """How well `product` fits a reference: the seller must match; then each selector segment (the
    category words and every qualifier) is checked against the fact it names, or against the
    listing's words when it names none; minus a penalty when the listing disclaims being the
    version with those qualifiers. Selectors pick the listing; they are never needs."""
    seller = str(ref.get("seller") or "").lower()
    if seller and seller != str(product.get("seller") or product.get("brand") or "").lower():
        return -1.0
    drop = _SEL_STOP | set(canon(seller).split())
    have = _listing_tokens(product)
    mine = measures(" ".join(str(v) for k, v in (product.get("props") or {}).items() if not _is_options(k))
                    + " " + own_text(product))
    text = re.sub(r"\b" + re.escape(seller) + r"(?:'s)?\b", " ", _fractions(ref.get("quals") or ""), flags=re.IGNORECASE)   # F2
    text = re.sub(r"\((?:in|measured in) [a-z ]+\)", " ", text, flags=re.IGNORECASE)     # "(in lb)" unit notes
    segs = [canon(p) for p in _SEG_SPLIT.split(text)]
    segs = [s for s in segs if set(s.split()) - drop]
    item = set(canon(ref.get("item") or "").split()) - drop
    scores = [x for x in (_segment_score(s, product, have, mine, drop) for s in segs) if x is not None]
    if item:
        scores.append(len(item & have) / len(item))
    score = sum(scores) / len(scores) if scores else 1.0
    sel = (set(" ".join(segs).split()) - drop) | item
    for d in parse_disclaimers(product.get("text")):
        if len(d & sel) >= max(1, (len(d) + 1) // 2):
            score -= 0.5
    return score


_SUPERLATIVE = re.compile(r"\b(?:that has |with )?the (?P<dir>lowest|smallest|least|shortest|highest|largest|biggest|longest|most) "
                          r"(?P<attr>[a-z][a-z ]{1,30}?)(?: of (?:the )?(?:\d+|two|three|those|them)\b[^,;]*)?(?=[,;]|$)", re.IGNORECASE)


def _talks_of(product, attr) -> float:
    """H6: 1 when a listing's notes redirect `attr` to another listing, 0.5 when its text names `attr`, else 0."""
    if not attr:
        return 0.0
    redirects = product.get("redirects")
    if redirects is None:
        redirects = parse_redirects(product.get("text"))
    if any(key_score(attr, r["attr"]) > 0 or _attr_tokens(attr) & _attr_tokens(r["attr"]) for r in redirects):
        return 1.0
    return 0.5 if " " + " ".join(toks(attr)) + " " in " " + " ".join(toks(product.get("text") or "")) + " " else 0.0


def pick_listing(ref, products, exclude=()):
    """The best listing for a reference among `products` (None if no listing of that seller). A
    superlative qualifier ("the lowest <attr> of the <n> that match") picks, among the listings matching
    the other qualifiers about equally well, the one with the lowest (highest) number for <attr>."""
    sup = _SUPERLATIVE.search(str(ref.get("quals") or ""))
    if sup:
        ref = dict(ref, quals=(ref["quals"][:sup.start()] + ref["quals"][sup.end():]).strip(" ,"))
    ranked = sorted(((selector_score(ref, p) + 0.2 * _talks_of(p, ref.get("follows")), -len(p.get("props") or {}), p["id"])
                     for p in products
                     if p.get("id") not in exclude), reverse=True)
    ranked = [r for r in ranked if r[0] >= 0]
    if not ranked:
        return None
    by_id = {p["id"]: p for p in products}
    if sup:
        def number(p):      # the attribute's own number, else one from a fact sharing a word with it
            nums = measures(value_of(p, sup.group("attr")) or "")
            if not nums:
                near = [v for k, v in (p.get("props") or {}).items() if _attr_tokens(sup.group("attr")) & _key_parts(k)[0]]
                nums = measures(" ".join(map(str, near)))
            return nums[0][0] if nums else None
        close = [(number(by_id[r[2]]), r[2]) for r in ranked if r[0] >= ranked[0][0] - 0.25]
        close = [c for c in close if c[0] is not None]
        if len(close) >= 2:
            low = sup.group("dir").lower() in ("lowest", "smallest", "least", "shortest")
            return by_id[(min(close) if low else max(close))[1]]
    return by_id[ranked[0][2]]


def resolve_chain(attr, ref, lookup, max_hops=6, vocab=()):
    """Follow a reference to the value it fixes: open the named listing; if it redirects `attr`
    elsewhere ("For <attr>, refer to ..."), follow that, up to `max_hops` times, never trusting a
    redirecting listing's own value; apply the last listing's pairing rule. `lookup(ref, seen)`
    returns the chosen listing (a product dict with props and text) or None; `vocab` holds values
    of `attr` seen at runtime. Returns (value or None, [listing ids])."""
    trail, target = [], ref
    for hop in range(max_hops + 1):
        listing = lookup(dict(target, follows=attr), set(trail))   # H6: the property this chain passes on
        if listing is None:
            return None, trail
        trail.append(listing["id"])
        redirects = listing.get("redirects")
        if redirects is None:
            redirects = parse_redirects(listing.get("text"))
        nxt = next((r for r in redirects if key_score(attr, r["attr"]) > 0 or _attr_tokens(attr) & _attr_tokens(r["attr"])), None)
        if nxt is not None and hop < max_hops:
            target = nxt
            continue
        value = value_of(listing, attr, vocab)
        return (paired_value(listing, attr, value) if value is not None else None), trail
    return None, trail


# ---- attaching needs to order lines ---------------------------------------------------------------

def line_of(scope, lines):
    """Index of the order line a scope phrase names ("<product words>"), given one phrase per
    line (its product words and/or category name); None when no line shares a word."""
    want = set(toks(scope or ""))
    if not want or not lines:
        return None
    scored = sorted(((len(want & set(toks(l or ""))) / len(want), -i) for i, l in enumerate(lines)), reverse=True)
    return -scored[0][1] if scored[0][0] > 0 else None


def attach_lines(needs, lines):
    """Give each scoped need its line index; unscoped needs stay None (they apply to every line,
    which is right for a one-line order)."""
    for n in needs:
        if getattr(n, "line", None) is None and getattr(n, "scope", None):
            n.line = line_of(n.scope, lines)
    return needs


def needs_for_line(needs, index):
    return [n for n in needs if getattr(n, "line", None) in (None, index)]


def same_line(a, b) -> bool:
    la, lb = getattr(a, "line", None), getattr(b, "line", None)
    if la is not None and lb is not None:
        return la == lb
    sa, sb = set(toks(getattr(a, "scope", None) or "")), set(toks(getattr(b, "scope", None) or ""))
    return not sa or not sb or bool(sa & sb)
# <<<<<<<< C0017 claude-2: need matcher + reference resolver (END) <<<<<<<<


_SYS_EXTRACT = (
    "You turn a shopper's request into JSON for a store search. Return only JSON with keys: "
    "product (2-4 word search phrase naming the kind of item), search (up to 3 other short search phrases), "
    "needs (list of {attr, value, op} for requirements the shopper STATES; op is eq, in (value is a list), ge or le), "
    "avoid (list of 'attribute value' phrases the shopper excludes), likely_hidden (up to 5 product properties the "
    "shopper probably cares about but did not state, most likely first). Ignore background story, budget, stock, "
    "message and step rules.")
_SYS_QUESTIONS = (
    "A shopper has unstated preferences on a few product properties. Given the request and the properties seen "
    "on candidate products (with example values), pick the properties most likely to hide a preference, most likely "
    "first. Never pick a property the shopper already settled. Return only JSON: {\"ask\": [property names]}.")
_SYS_ANSWER = (
    "Classify a shopper's reply to a question about one product property. Return only JSON: {\"kind\": "
    "\"constraint\"|\"open\"|\"confused\"|\"multi\", \"attr\": property or null, \"value\": required value or null, "
    "\"op\": \"eq\"|\"in\"|\"ge\"|\"le\"}. open = no preference; confused = the shopper did not understand; "
    "multi = the shopper asks for one question per message.")
_SYS_VERIFY = (
    "Decide from the listing data alone whether the product meets each requirement. A property taken from another "
    "listing does not count as met. Return only JSON: {\"needs\": [{\"attr\": name, \"meets\": \"yes\"|\"no\"|\"unknown\"}]}.")
_SYS_COMPARE = (
    "A shopper requires a product property. You get the requirement and the values that store listings "
    "show for that property. List the values that satisfy the requirement: convert units when they differ, "
    "and count a value that says the same thing in other words (a synonym, a spelling variant, a more "
    "specific shade of the named colour). Do not count a value that is only related or partly overlapping. "
    "Return only JSON: {\"satisfy\": [values copied exactly from the list]}.")
_OPS = {"eq", "in", "ge", "le"}


_SYS_READ = (   # R1: our own prompt and our own Ledger field names
    "You turn one shopper's message to an online store into notes for the store's assistant. Answer with one JSON "
    "object only, with these keys: product (the kind of item to buy, copied word for word from the message, 1 to 5 "
    "words); category (a store section or range the message names, copied, else null); quantity (integer); needs "
    "(a list of {attr, op, value}: attr is the property noun the shopper ties the value to, value is copied from the "
    "message, op is one of eq, ne, in, ge, le, with a list value for in); avoid (phrases the shopper rules out, "
    "copied); budget_cents (integer or null); prefer (cheapest, best or any); stock_only (true or false); decline_ok "
    "(true or false); messages_allowed and steps_allowed (integers or null). Story, reasons, people and feelings are "
    "not needs. The price, the budget, stock and the kind of item itself are never needs. Copy words from the "
    "message; never add a value it does not state."
)


class Reasoning:
    """Deterministic reasoning on the language parsers. Every method works with llm=None; with a
    ModelChain it spends at most CONFIG["assist_max_calls"] calls per episode on the few
    decisions where a model helps most."""

    @staticmethod
    def _ask(llm, system, payload, max_tokens=500, purpose="actor", final=False):
        """One model question. Judge questions are counted per question (a failover is not a second
        one): value comparisons and final checks each have their own share of the judge calls, and
        the other calls stop early enough to leave the judge its share."""
        if llm is None or not llm.alive(purpose) or llm.calls >= CONFIG["assist_max_calls"] or llm.spent():
            return None
        tally = llm.__dict__.setdefault("asked", {})
        if purpose == "judge":
            kind = "final" if final else "compare"
            share = CONFIG["final_max_calls"] if final else CONFIG["assist_max_judge_calls"] - CONFIG["final_max_calls"]
            if tally.get(kind, 0) >= share:
                return None
            tally[kind] = tally.get(kind, 0) + 1
        else:
            judge_left = CONFIG["assist_max_judge_calls"] - tally.get("final", 0) - tally.get("compare", 0)
            if llm.calls >= CONFIG["assist_max_calls"] - judge_left:
                return None   # the judge keeps its calls
        try:
            data = llm.ask_json(system, json.dumps(payload, ensure_ascii=False), purpose=purpose, max_tokens=max_tokens)
        except Exception:  # noqa: BLE001 - a model failure only means "no extra evidence"
            return None
        return data if isinstance(data, dict) else None

    def _merge_llm_request(self, led, data):
        product = data.get("product")
        if (isinstance(product, str) and 0 < len(product.split()) <= 6 and words(product)
                and f" {' '.join(words(product))} " in f" {' '.join(words(led.query))} "):   # a phrase of the request
            led.notes["llm_product"] = product.strip().lower()
            if not led.product or led.notes.get("product_weak"):   # a grammar-fallback guess yields to the model
                led.product = led.notes["llm_product"]
        if not CONFIG.get("model_extract_needs"):
            return   # model needs/searches had no measured gain and lost tasks (C0066); the rules parse stands
        led.notes["llm_search"] = [q.strip() for q in data.get("search") or [] if isinstance(q, str) and q.strip()][:3]
        rules_needs = any(n.op != "avoid" for n in led.needs)
        rules_avoid = any(n.op == "avoid" for n in led.needs)
        for n in ([] if rules_needs else data.get("needs") or []):
            if not isinstance(n, dict):
                continue
            attr, value, op = n.get("attr"), n.get("value"), n.get("op") if n.get("op") in _OPS else "eq"
            if isinstance(value, list):
                value, op = [str(v) for v in value if v], "in"
            if not value or not isinstance(value, (str, list)):
                continue
            attr = _clean_attr(str(attr)) if attr else None
            vtoks = set(toks(" ".join(value) if isinstance(value, list) else value))
            if not vtoks <= set(toks(led.query)):
                continue   # grounded needs only: every value word must appear in the request
            if not vtoks or any((attr and n2.prop and _attr_tokens(attr) & _attr_tokens(n2.prop))
                                or vtoks <= set(toks(" ".join(n2.value) if isinstance(n2.value, list) else str(n2.value or "")))
                                for n2 in led.needs):
                continue
            led.needs.append(Need(attr, value, "request", True, op))
            led.notes.setdefault("model_needs", []).append([attr, value, op])
        for phrase in ([] if rules_avoid else data.get("avoid") or []):
            if isinstance(phrase, str) and toks(phrase) and set(toks(phrase)) <= set(toks(led.query)) and not any(n.op == "avoid" and set(toks(n.value)) == set(toks(phrase)) for n in led.needs):
                led.needs.append(Need(None, phrase.lower(), "request", True, "avoid"))
                led.notes.setdefault("model_needs", []).append([None, phrase.lower(), "avoid"])
        led.notes["likely_hidden"] = [h.lower() for h in data.get("likely_hidden") or [] if isinstance(h, str)][:5]

    _FINAL_PROMPT = (
        "You check a shopping cart before checkout. For each product decide, from its listing data and the "
        "shopper's own words only, whether it is the kind of product requested and meets every "
        "requirement they stated or revealed (ignore price, budget and stock). Judge only requirements the "
        "shopper states as requirements; do not infer extra ones from their story (who they are, what it is "
        "for). In \"missing\", quote the shopper's requirement. Return only JSON: "
        "{\"items\": [{\"id\": id, \"fits\": \"yes\"|\"no\"|\"unsure\", \"missing\": [requirement]}]}.")

    def final_check(self, ledger, products, llm, lines=()):
        """One judge-model look at the products the order would carry: {id: True | False | None}.
        `lines` are the per-line ledgers of a multi-line order (they hold its replies and needs)."""
        if not products:
            return {}
        books = [ledger] + [l for l in lines if l is not ledger]
        heard = [text for b in books for source, text in b.heard if source == "reply"][-6:]
        if lines:
            known = [{"line": l.product, "quantity": l.quantity,
                      "needs": [n.as_dict() for n in l.needs if n.op != "ref"]} for l in lines]
        else:
            known = [n.as_dict() for n in ledger.needs if n.op != "ref"]
        payload = {"shopper_request": ledger.query, "shopper_replies": heard, "known_requirements": known,
                   "products": [compact_product(p, full=True) for p in products]}
        data = self._ask(llm, self._FINAL_PROMPT, payload, 400, purpose="judge", final=True)
        # a veto needs evidence: the missing requirement must share a content word with what the shopper said
        said = " ".join([ledger.query] + heard + [f"{n.prop or ''} {n.value}" for b in books for n in b.needs])
        evidence = {w for w in toks(said) if len(w) > 2} - _FUNCTION_WORDS
        out = {}
        for item in (data or {}).get("items") or []:
            if isinstance(item, dict) and item.get("id"):
                fits = str(item.get("fits", "")).lower()
                missing = " ".join(str(m) for m in item.get("missing") or [])
                grounded = bool(({w for w in toks(missing) if len(w) > 2} - _FUNCTION_WORDS) & evidence)
                out[str(item["id"])] = True if fits == "yes" else (False if fits == "no" and grounded else None)
        return out

    def verify(self, ledger, product, llm):
        """Model check of one candidate against the needs: True, False or None (no answer)."""
        needs = [n.as_dict() for n in ledger.needs if n.op not in ("ref",)]
        if not needs:
            return True
        data = self._ask(llm, _SYS_VERIFY, {"requirements": needs, "product": compact_product(product, full=True)}, 300,
                         purpose="judge")
        verdicts = [str(v.get("meets", "")).lower() for v in (data or {}).get("needs") or [] if isinstance(v, dict)]
        if not verdicts:
            return None
        if "no" in verdicts:
            return False
        return True if all(v == "yes" for v in verdicts) and len(verdicts) >= len(needs) else None

    def _model_read(self, led, text, llm) -> bool:
        """R1: one structured model reading that fills what the rules left empty, where it is grounded in the
        request: the product when the rules found none or a weak one, the needs when the rules found none, the
        avoids when the rules found none. It never replaces a rules product, need or quantity; the regex readers
        keep budget, caps, stock and decline. True when a usable reading came back."""
        data = self._ask(llm, _SYS_READ, {"message": text}, 900, purpose="read")
        if not isinstance(data, dict) or not any(k in data for k in ("product", "needs", "avoid")):
            return False
        said = f" {' '.join(words(text))} "
        said_toks = set(toks(text))

        def grounded(phrase):
            return (isinstance(phrase, str) and 0 < len(phrase.split()) <= 6 and bool(words(phrase))
                    and f" {' '.join(words(phrase))} " in said)

        def vtoks(value):
            return set(toks(" ".join(map(str, value)) if isinstance(value, list) else str(value or "")))
        product = data.get("product")
        if (grounded(product) and not _VALUE_JUNK.search(product)
                and not set(words(product)) & (_NARRATIVE | _PRONOUNS | _ATTR_JUNK) and not _order_head(product)):   # A-R5 T1
            mp = product.strip().lower()
            led.notes["llm_product"] = mp
            rules = [w for w in words(led.product or "") if w not in _STOP]
            mine = [w.rstrip("s") for w in words(mp) if w not in _STOP]
            same_head = bool(rules) and bool(mine) and (rules[-1].rstrip("s") == mine[-1]
                                                        or {x.rstrip("s") for x in rules} <= set(mine))
            if not led.product or (led.notes.get("product_weak") and not same_head):   # fills a missing or weak product only
                led.product, led.notes["product_weak"] = mp, False
        category = data.get("category")
        if grounded(category) and not led.category and {w.rstrip("s") for w in words(category)} & {
                w.rstrip("s") for w in words(led.product or "") if w not in _STOP}:   # a section consistent with the product
            led.category = category.strip().lower()
        item = set(toks(led.product or ""))
        found = []
        for n in data.get("needs") or []:
            if not isinstance(n, dict):
                continue
            attr, value, op = n.get("attr"), n.get("value"), str(n.get("op") or "eq").lower()
            op = op if op in ("eq", "ne", "in", "ge", "le") else "eq"
            if isinstance(value, list):
                value = [str(v).strip() for v in value if isinstance(v, (str, int, float)) and str(v).strip()][:6]
                op = "in" if op in ("eq", "in") else op
            elif isinstance(value, (str, int, float)) and str(value).strip():
                value = str(value).strip()
                op = "eq" if op == "in" else op
            else:
                continue
            vt = vtoks(value)
            if not vt or not vt <= said_toks or vt <= item:
                continue   # grounded values only, and the item itself is not a need
            plain = _FUNCTION_WORDS | _STOP | _PRONOUNS | _NARRATIVE
            if set(words(" ".join(map(str, value)) if isinstance(value, list) else value)) <= plain or (
                    isinstance(attr, str) and attr.strip() and set(words(attr)) <= plain):
                continue   # a value or property made of function words only is not a need
            if op in ("ge", "le") and not any(ch.isdigit() for ch in str(value)):
                op = "eq"
            attr = _clean_attr(str(attr)) if isinstance(attr, str) and attr.strip() else None
            if attr and (attr in _ATTR_JUNK or _VALUE_JUNK.search(attr) or set(words(attr)) <= {"category", "stock", "quantity", "brand"}):
                continue
            found.append(Need(attr, value, "request", True, op))
        stated = [n for n in led.needs if n.op not in ("ref", "avoid")]
        avoided = any(n.op == "avoid" for n in led.needs)
        if found and not stated:   # the rules read no need: the model's grounded needs fill the gap
            led.needs = led.needs + found
        for phrase in ([] if avoided else data.get("avoid") or []):
            if isinstance(phrase, str) and set(words(phrase)) & {"stock", "order", "backorder", "sold", "shelf"}:
                continue   # availability is the stock reader's: not a property to avoid
            if (isinstance(phrase, str) and toks(phrase) and set(toks(phrase)) <= said_toks
                    and not any(n.op == "avoid" and set(toks(phrase)) <= vtoks(n.value) for n in led.needs)):
                led.needs.append(Need(None, phrase.strip().lower(), "request", True, "avoid"))
        led.notes["model_read"] = {"product": led.product, "needs": len(found),
                                   "budget_agrees": data.get("budget_cents") in (None, led.budget)}
        return True

    def extract(self, query, opening, llm):
        query, opening = _times_sign(query), _times_sign(opening)   # A-R6 X1: "2 x <item>" is a count
        text = query + ("\n" + opening if isinstance(opening, str) else
                        ("\n" + json.dumps(opening) if opening else ""))
        req = parse_request(text)
        led = Ledger(query, opening)
        led.budget, led.budget_unit = req["budget"], req["budget_unit"]
        led.msg_budget, led.step_budget = req["msg_budget"], req["step_budget"]
        led.quantity, led.prefer = req["quantity"], req["prefer"]
        led.stock_only = req["stock_only"]
        led.notes["decline_stated"] = req["decline_ok"]
        led.product, led.category = req["product"], req["category"]
        led.notes["product_weak"] = bool(req.get("product_weak"))
        for n in req["needs"]:
            led.needs.append(Need(n["attr"], n["value"], "request", True, n["op"]))
        for phrase in req["avoid"]:
            led.needs.append(Need(None, phrase, "request", True, "avoid"))
        led.notes["tentative"] = req["tentative"]
        try:
            refs = parse_request_refs(text)
        except Exception:  # noqa: BLE001 - a reference the parser cannot read is left to the questions
            refs = []
        for r in refs:     # C0017: references the request states
            led.needs = [n for n in led.needs if not (n.prop and _attr_tokens(n.prop) == _attr_tokens(r["attr"]))]
            led.needs.append(Need(r["attr"], None, "request", True, "ref", scope=r["scope"]))
            led.notes.setdefault("refs", []).append({"attr": r["attr"], "ref": r["ref"], "status": "pending", "scope": r["scope"]})
        led.needs = [n for n in led.needs if not re.search(r"\blisting\b|\bsold by\b", str(n.value or ""), re.IGNORECASE)]
        parts = split_lines(text)
        for i in range(len(parts) - 1, 0, -1):   # A-R5 T3: an order-level-only "line" is no line; its sentences join the one before
            if _order_head(parts[i]["product"]):
                parts[i - 1]["sentences"] += parts.pop(i)["sentences"]
        if parts and _order_head(parts[0]["product"]):
            if len(parts) > 1:
                parts[1]["sentences"] = parts[0]["sentences"] + parts[1]["sentences"]
            parts.pop(0)
        if len(parts) >= 2:
            for part in parts:
                body = " ".join(part["sentences"])
                sub = parse_request(body)
                needs = [Need(n["attr"], n["value"], "request", True, n["op"]) for n in sub["needs"]]
                needs += [Need(None, a, "request", True, "avoid") for a in sub["avoid"]]
                try:
                    line_refs = parse_request_refs(body)
                except Exception:  # noqa: BLE001 - a reference the parser cannot read is left to the questions
                    line_refs = []
                for r in line_refs:
                    needs = [n for n in needs if not (n.prop and _attr_tokens(n.prop) == _attr_tokens(r["attr"]))]
                    needs.append(Need(r["attr"], None, "request", True, "ref", scope=r["scope"]))
                needs = [n for n in needs if not re.search(r"\blisting\b|\bsold by\b", str(n.value or ""), re.IGNORECASE)]
                led.lines.append({"product": part["product"], "quantity": part.get("quantity") or line_quantity(
                    _mask_measures(body)[0], _mask_measures(part["product"])[0]),   # A-R7 M1
                                  "needs": needs, "category": sub["category"], "tentative": sub["tentative"], "text": body,
                                  "soft": bool(part.get("soft")),   # R9 P2
                                  "refs": [{"attr": r["attr"], "ref": r["ref"], "status": "pending", "scope": r["scope"]}
                                           for r in line_refs]})
            led.product = led.lines[0]["product"]
        elif led.product:
            led.quantity = max(led.quantity, line_quantity(_mask_measures(text)[0], _mask_measures(led.product)[0]))   # A-R7 M1
            if not _MEASURE_HELD.match(led.product):   # A-R7 M1: a leading measure stays in the product
                led.product = re.sub(r"^(?:\d+|one|two|three|four|five|six)\s+", "", led.product) or led.product
        # rules-first: the model reads the request only when the rules found neither a product nor lines
        read = bool(llm is not None and CONFIG.get("model_read") and CONFIG.get("actor") == "assist" and not led.lines
                    and self._model_read(led, text, llm))   # the model actor reads the raw request itself
        data = (self._ask(llm, _SYS_EXTRACT, {"request": text})   # R1: the older product-only reading when R1 gave nothing
                if not read and not led.lines and not led.product else None)
        if data:
            self._merge_llm_request(led, data)
        if (read or data) and not led.lines and led.product and led.quantity == 1:   # Q2a: a product only the model read
            led.quantity = product_count(_mask_measures(text)[0], _mask_measures(led.product)[0])   # A-R7 M1
        source = led.product or led.category or ""
        led.terms = [w for w in words(source) if w not in _STOP][:5]
        if not led.terms:
            need_words = {w for n in led.needs if isinstance(n.value, str) for w in words(n.value)}
            sentences = [x for x in re.split(r"(?<=[.!?])\s+", query) if need_words & set(words(x))] or [re.split(r"[.!?]", query)[0]]
            led.terms = [w for w in words(sentences[0]) if w not in _STOP and w not in _NARRATIVE and w not in need_words
                         and not w.isdigit() and len(w) > 2][:3]
        return led

    def search_terms(self, ledger, shop, llm):
        product = ledger.product or " ".join(ledger.terms)
        alternatives = [p.strip() for p in re.split(r"\s+(?:or|and|/)\s+|,\s*", product) if p.strip()]
        if len(alternatives) > 1:
            product = alternatives[0]
        unchecked = [m[1] for m in ledger.notes.get("model_needs") or []]   # model values not yet seen listed
        values = [n.value for n in ledger.needs if n.op == "eq" and isinstance(n.value, str) and n.value
                  and n.value not in unchecked]
        n_eq = len(values)   # Q1c: the next-value phrase below draws only on these stated single values
        values += [v for n in ledger.needs if n.op == "in" and isinstance(n.value, (list, tuple))   # Q1: any-of values
                   for v in n.value if isinstance(v, str) and v and v not in unchecked]
        values += tentative_values(ledger)
        if not values:   # OR1: an "X or Y" need names its values too: its first alternative leads the search
            values = [n.value[0] for n in ledger.needs if n.op == "in" and isinstance(n.value, list) and n.value
                      and isinstance(n.value[0], str) and n.value[0].strip()][:1]
        out = []
        if values and product:
            if set(toks(values[0])) <= set(toks(product)):   # "<value>-style <product>" already names the value
                out.append(f"{values[0]} {product.split()[-1]}")
            else:
                out.append(f"{values[0]} {product}")
        if product:
            out.append(product)
        for v in values[1:min(2, n_eq)]:   # Q1: the next stated value with the product, a second route to a fit
            if product and v != values[0] and not set(toks(v)) <= set(toks(product)):
                out.append(f"{v} {product}")
        if ledger.category and ledger.category != product:
            out.append(ledger.category)
        if len(alternatives) > 1:
            out += alternatives[1:]
        out += [q for q in ledger.notes.get("llm_search") or []   # after the rules' own phrases, naming the product
                if set(toks(q)) & set(toks(product))]
        if len(product.split()) > 1:
            out.append(product.split()[-1])
        seen, unique = set(), []
        for phrase in out:
            phrase = _strip_money_clause(phrase)   # A-R7c C1: never search for the budget
            key = phrase.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(phrase.strip())
        kept = [p for p in unique if not _order_head(p) and not _MONEY_TOKEN.search(p)]   # A-R5 T3 / A-R7 M3
        return kept or unique

    def judge(self, ledger, product, llm):
        missing, bad = [], False
        if product.get("in_stock") is False:   # nothing off the shelf can be ordered
            return Verdict(False, ["stock"], "unavailable")
        for need in ledger.needs:
            if need.op == "ref":
                missing.append(need.prop or "reference")
                continue
            status = judged_status(need, product, ledger.notes.get("equivalents"))   # C0017/C0045
            if status is False and need.hard:
                bad = True
                missing.append(need.prop or str(need.value))
            elif status is None:
                missing.append(need.prop or str(need.value))
        if bad:
            return Verdict(False, missing, "violates a need")
        return Verdict(None if missing else True, missing)

    def rank(self, ledger, candidates, llm):
        """Confirmed products first, cheapest first; unconfirmed ones only after them."""
        if ledger.prefer == "best":
            return sorted(candidates, key=lambda p: (p.get("_verdict") is not True, -(p.get("rating") or 0), p.get("price") or 0))
        def blind(p):
            return sum(1 for n in ledger.needs if n.hard and n.prop and n.op not in ("ref", "avoid")
                       and need_status(n, p) is None and not related_facts(p, n.prop))

        def lifted(p):   # confirmed only through the judge model's equivalences, not by the listing's own value
            return any(n.hard and n.op not in ("ref", "avoid") and need_status(n, p) is not True for n in ledger.needs)
        return sorted(candidates, key=lambda p: (p.get("_verdict") is not True,
                                                 lifted(p) if p.get("_verdict") is True else 0,
                                                 blind(p) if p.get("_verdict") is None else 0,
                                                 p.get("price") if p.get("price") is not None else float("inf"), p.get("id") or ""))

    def next_question(self, ledger, shop, messages_left, llm):
        planned = q_plan_next(ledger, shop, messages_left, llm, self)   # section 6c
        if not planned:
            return None
        line, key, text = planned
        ledger.asked.append(key)
        ledger.notes.setdefault("q_log", []).append([line, key, text, None])
        return text

    def compare_values(self, ledger, shop, llm):
        """Settle needs the rules leave unverified or unmet because of another unit or another wording:
        one judge call per need (compare_plan picks at most CONFIG["compare_max_calls"] per episode)
        lists the candidate values that satisfy it; judge() reads ledger.notes["equivalents"]."""
        if llm is None or shop is None:
            return
        done = ledger.notes.setdefault("equivalents", {})
        budget = CONFIG.get("compare_max_calls", 2) - ledger.notes.get("compare_calls", 0)
        for key, need, values in compare_plan(ledger.needs, list(shop.products.values()), done, budget):
            requirement = {"property": need.prop, "op": need.op, "value": need.value}
            data = self._ask(llm, _SYS_COMPARE, {"requirement": requirement, "listing_values": values}, 300, purpose="judge")
            ledger.notes["compare_calls"] = ledger.notes.get("compare_calls", 0) + 1
            opts = need.value if isinstance(need.value, list) else [need.value]
            numeric = need.op in _NUM_OPS or bool(re.search(r"\d", str(need.value)))
            yes = [canon(v) for v in (data or {}).get("satisfy") or [] if isinstance(v, str) and (
                (numeric and measures(v)) or any(set(toks(str(o))) and set(toks(str(o))) <= set(toks(v)) for o in opts))]
            done[key] = {"asked": [canon(v) for v in values], "yes": yes, "answered": data is not None}

    def late_needs(self, ledger, shop):
        """Bind the request's tentative phrases to properties the candidates actually carry: a
        "<value> <property>" group whose property words name a fact key, or a bare value that some
        fact key holds exactly. Each phrase is tried once; nothing comes from a fixed list."""
        todo = ledger.notes.get("tentative") or []
        if not todo or shop is None or not shop.products:
            return
        table = {}
        for p in shop.products.values():
            for key, value in (p.get("props") or {}).items():
                if not _is_options(key) and str(value).strip():
                    for alt in _alternatives(value):
                        table.setdefault(key, set()).add(alt)
        if not table:
            return
        ledger.notes["tentative"] = []
        settled = [n.prop for n in ledger.needs if n.prop]
        dropped = ledger.notes.get("dropped") or []
        product = set(canon(ledger.product or "").split())

        def free(attr):
            return not any(key_score(attr, s) >= 2.0 or key_score(s, attr) >= 2.0 for s in settled + dropped)

        def add(attr, value):
            if (isinstance(value, str) and len(value.split()) == 1 and parse_count(value.strip()) is not None
                    and parse_count(value.strip()) == (ledger.quantity or 1)):
                return   # UNITS: "<count> <key>" restating the order's quantity ("one units") is not a property value
            short = [n for n in ledger.needs if n.source == "request" and n.op == "eq" and n.prop and isinstance(n.value, str)
                     and canon(n.value) == canon(value or "") and set(toks(n.prop)) < set(toks(attr or ""))]
            if short:   # L1: the request parse named this need's property by part of the key ("<word>" for "<word> <word>")
                short[0].prop = attr
                settled.append(attr)
                return
            if attr and value and free(attr):
                ledger.needs.append(Need(attr, value, "request", True, "eq"))
                settled.append(attr)
        for item in todo:
            if item.get("g"):
                words_ = item["g"].split()
                for i in range(1, len(words_)):
                    value, attr = " ".join(words_[:i]), " ".join(words_[i:])
                    if any(key_score(attr, k) >= 2.0 for k in table):
                        vw, aw = words_[:i], words_[i:]   # L1: no function word at the split ("<v> as the <key>")
                        while len(vw) > 1 and vw[-1].lower() in _FUNCTION_WORDS:
                            vw = vw[:-1]
                        while len(aw) > 1 and aw[0].lower() in _FUNCTION_WORDS:
                            aw = aw[1:]
                        add(" ".join(aw), " ".join(vw))
                        break
            elif item.get("a"):
                if any(key_score(item["a"], k) >= 2.0 and canon(item["v"]) in table[k] for k in table):
                    add(item["a"], item["v"])
            elif item.get("v") and canon(item["v"]) not in product:
                holders = sorted((sum(canon(item["v"]) in _alternatives(p_.get("props", {}).get(k)) for p_ in shop.products.values()), k)
                                 for k in table if canon(item["v"]) in table[k])
                if holders:
                    add(plain_key(holders[-1][1]), item["v"])

    def _set_need(self, ledger, need):
        ledger.needs = [n for n in ledger.needs if not (n.prop and need.prop and _attr_tokens(n.prop) == _attr_tokens(need.prop)
                                                         and n.op != "avoid" and same_line(n, need))]
        ledger.needs.append(need)

    def absorb(self, ledger, lines, source, shop, llm):
        tags = set()
        for line in lines or []:
            if not isinstance(line, str):
                continue
            ledger.heard.append((source, line))
            if source == "reply":
                asked = ledger.asked[-1] if ledger.asked else None
                known = self._known_properties(ledger, shop)
                scopes = [ledger.product, ledger.category, ledger.notes.get("q_category")] + [
                    l.get("product") for l in ledger.lines if isinstance(l, dict)] + [
                    p.get("category") for p in (shop.products.values() if shop is not None else ())]
                ans = classify_answer(line, asked, values=shop.values_of(asked) if (asked and shop is not None) else None,
                                      properties=known, scopes=scopes)
                if (ans["kind"] in ("constraint", "reference") and not asked and ans.get("attr")
                        and not any(key_score(ans["attr"], k) >= 2.0 or key_score(k, ans["attr"]) >= 2.0 for k in known)):
                    ans = {"kind": "noinfo"}   # R1: an unasked reply naming no known property is talk, not a need
                kind = ans["kind"]
                if kind == "constraint":
                    attr = ans.get("attr") or asked
                    need = Need(attr, _snap_value(ans["value"], attr, shop, None if ans.get("clause") else line), "shopper", True,
                                ans["op"], scope=ans.get("scope"))
                    need.clause = ans.get("clause")   # RV1: "rv" (the property's own clause) or "as" (the "as / to be" rule)
                    self._set_need(ledger, need)
                    tags.add("need")
                elif kind == "reference":
                    attr = ans.get("attr") or asked
                    self._set_need(ledger, Need(attr, None, "shopper", True, "ref", scope=ans.get("scope")))
                    ledger.notes.setdefault("refs", []).append({"attr": attr, "ref": ans["ref"], "status": "pending",
                                                                "scope": ans.get("scope")})
                    tags.add("reference")
                elif asked and shop is not None and self._value_from_catalog(ledger, asked, line, shop):
                    tags.add("need")
                else:   # no value named: no preference, or not understood; a model may still read it
                    data = (self._ask(llm, _SYS_ANSWER, {"question_about": asked, "reply": line}, 120)
                            if CONFIG.get("model_replies") else None)
                    value = (data or {}).get("value")
                    vt = set(toks(" ".join(map(str, value)) if isinstance(value, list) else str(value or "")))
                    listed = {w for p in (shop.products.values() if shop is not None else ())
                              for k, v in (p.get("props") or {}).items() for w in toks(str(v))}
                    if data and data.get("kind") == "constraint" and value and vt and vt <= set(toks(line)) and vt <= listed:
                        op = data.get("op") if data.get("op") in _OPS else "eq"
                        self._set_need(ledger, Need(_clean_attr(str(data.get("attr") or asked or "")) or asked,
                                                    data["value"], "shopper", True, op))
                        tags.add("need")
                    elif asked:
                        ledger.notes.setdefault("open", []).append(asked)
                continue
            event = classify_event(line, need_props=self._known_properties(ledger, shop))
            kind = event["kind"]
            if kind == "budget":
                if event["cents"] != ledger.budget:
                    ledger.budget, ledger.budget_unit = event["cents"], event["unit"]
                    ledger.notes["budget_event"] = not _R9_PER_ITEM.search(line)   # A-R8b G1
                    tags.add("budget")
            elif kind == "stock":
                tags.add("stock")   # the proposed item can no longer be bought
            elif kind == "item":
                tags.add("price")   # the proposed item changed (price or stock): re-read it before checkout
            elif kind == "relax":
                attr = event["attr"]
                attr_values = set(toks(" ".join(shop.values_of(attr)))) if shop is not None else set()

                def relaxed(n):
                    if n.prop:
                        return bool(_attr_tokens(n.prop) & _attr_tokens(attr))
                    vt = set(toks(" ".join(n.value) if isinstance(n.value, list) else str(n.value or "")))
                    return bool(vt) and vt <= attr_values and n.op != "avoid"
                ledger.needs = [n for n in ledger.needs if not relaxed(n)]
                ledger.notes.setdefault("dropped", []).append(attr)
                tags.add("drop")
        return tags

    @staticmethod
    def _known_properties(ledger, shop):
        """Property names read at runtime: this category's fact keys and the request's needs."""
        names = set(shop._prop_values()) if shop is not None else set()
        names |= {n.prop for n in ledger.needs if n.prop}
        names |= set(ledger.asked)
        return sorted(n for n in names if n and len(n) > 2)

    def _value_from_catalog(self, ledger, asked, line, shop):
        """A reply that names one of the asked property's catalogue values is a constraint."""
        low = " " + " ".join(toks(line)) + " "
        if not set(toks(asked)) <= set(low.split()):
            return False   # a reveal names the property it is about
        values = sorted(shop.values_of(asked), key=len, reverse=True)
        for value in values:
            vt = " ".join(toks(value))
            if vt and not set(vt.split()) <= _FUNCTION_WORDS and f" {vt} " in low:
                self._set_need(ledger, Need(asked, value, "shopper", True, "eq"))
                return True
        bound = re.search(r"\b(at (?:most|least) \d+(?:\.\d+)?\s*[a-z\"]*)", line, re.IGNORECASE)
        if bound:
            op, value = value_op(bound.group(1))
            self._set_need(ledger, Need(asked, value, "shopper", True, op))
            return True
        return False

    def check_order(self, ledger, cart, shop, llm):
        problems = []
        lines = [l for l in (cart or {}).get("lines") or [] if (l.get("qty") or 0) > 0]
        if len(lines) != 1:
            problems.append(f"the cart must hold exactly one product line, it holds {len(lines)}")
        else:
            line = lines[0]
            if line.get("qty") != ledger.quantity:
                problems.append(f"quantity is {line.get('qty')}, the shopper wants {ledger.quantity}")
            product = shop.products.get(line.get("id"))
            if product is None:
                problems.append("the cart item was never inspected")
            else:
                if product.get("in_stock") is False:
                    problems.append("the cart item shows no stock")
                verdict = self.judge(ledger, product, llm)
                if verdict.ok is False:
                    problems.append("the cart item violates: " + ", ".join(map(str, verdict.missing)))
        return not problems, problems

    def should_decline(self, ledger, shop, exhausted, llm):
        return bool(exhausted)

    def decline_note(self, ledger):
        return "None of the listings meets every requirement, so no order is placed."


# ======================================================================================
# 6c. QUESTION PLANNER (claude-3, C0021). Which property to ask about, for which order line,
#     within the message cap, and which line a reveal belongs to. Deterministic; at most one
#     optional model call. Runtime evidence only: property names come from the candidates' fact
#     keys, values from their facts, line names from the ledgers; no name tables, no phrase lists.
#     Race-4 evidence (public feedback) behind the design:
#     - a question that names a hidden property by its table key draws the reveal ~91% of the time;
#     - every hidden property is a key of the line's property table, and candidates differ on it;
#     - a reply without a value of the asked property is never a need (whatever its wording), and
#       re-asking the same property never helped, so every asked property is settled for its line;
#     - open-ended or two-property questions draw "one at a time" refusals (20% vs 2-5%).
# ======================================================================================

import math                              # noqa: E402 - section-local imports keep the patch delimited
from collections import Counter          # noqa: E402

CONFIG.setdefault("q_filter_stated", True)   # rank on candidates that no stated need rules out
CONFIG.setdefault("q_min_pool", 8)            # below this many such candidates, rank on the whole category
CONFIG.setdefault("q_reserve", 0)      # messages kept back when the cap is 4 or more (1 only if a refusal path sends one)
CONFIG.setdefault("q_line_hint", True)   # Q1: a multi-line order's hint words come from the line's own sentences
# score = sum(weight x feature), all features measured on the current candidates' facts and the request:
# h = normalised entropy of the key's values, cov = share of candidates that show the key, top = share of
# the commonest value, dup = the key extends a shorter key of the same table, hint = the request uses the
# key's words or one of its values, model = 1/(1+rank) among the optional model call's picks, num = share of
# the key's values that are measures (a number, at most a short unit): C0081, categorical properties first
CONFIG.setdefault("q_weights", {"h": 0.25, "cov": 2.0, "top": 0.0, "dup": 0.0, "hint": 0.6, "model": 0.0, "num": -1.0,
                                 "split": 2.0})   # split (Q2): what an answer rules out among the candidates
# model = 0 (C0066): in paid p35/p39/p42 the planning call returned nothing usable and the extraction's
# likely-hidden list changed the questions in 8 of 30 tasks with no gain; set it > 0 to use the model's picks again


def _q_key(name) -> str:
    raw = str(name or "")
    w = words(re.sub(r"[._]", " ", raw))
    if len(w) > 1 and len(w[1]) >= 3 and re.match(r"^[A-Za-z]{1,2}[._]", raw):
        w = w[1:]   # a namespace prefix ("fa.pattern", "m_pantshort_fit_type"), not a word of the property
    return " ".join(w)


def _q_core(value) -> str:
    """QK2: a value's identity across key names: its number with its unit ("16gb ram" ~ "16 gb" -> 16gb), else its words."""
    low = str(value).lower()
    found = re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)?", low)
    return " ".join(n + (u or "") for n, u in found) if found else "".join(toks(low))


def _q_alias(a, b) -> bool:
    """QK2: two keys hold largely the same values (two names of one property)."""
    ca, cb = {_q_core(v) for v in a}, {_q_core(v) for v in b}
    return min(len(ca), len(cb)) >= 2 and len(ca & cb) >= 0.6 * min(len(ca), len(cb))


def _q_same_word(a, b) -> bool:
    """One word, or one word's stem inside another ("pack" / "packaging"), four letters at least."""
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))


def q_lines(ledger) -> list:
    """Order lines as [{'i', 'noun', 'category'}]. Reads ledger.lines when the core sets it
    (list of dicts or objects with product/category); otherwise one line from the request."""
    lines = getattr(ledger, "lines", None)
    out = []
    if isinstance(lines, (list, tuple)) and lines:
        for i, line in enumerate(lines):
            get = line.get if isinstance(line, dict) else (lambda k, _l=line: getattr(_l, k, None))
            out.append({"i": i, "noun": str(get("product") or get("category") or ""), "category": get("category")})
        return out
    return [{"i": 0, "noun": ledger.product or ledger.category or "", "category": ledger.notes.get("q_category")}]


def _q_settled(ledger, line) -> set:
    """Property keys this line no longer needs asked: stated, revealed, asked, dropped, selectors."""
    out = set()
    for n in ledger.needs:
        if n.prop and getattr(n, "line", None) in (None, line):
            out.add(_q_key(n.prop))
    for entry in ledger.notes.get("q_log", []):
        if entry[0] == line:
            out.add(entry[1])
    out |= {_q_key(p) for p in ledger.notes.get("dropped", []) if p}
    for item in ledger.notes.get("refs", []):   # qualifiers of a referenced listing are selectors
        quals = (item.get("ref") or {}).get("quals") or ""
        out |= {_q_key(q) for q in re.split(r",|\band\b|\bwith\b", quals) if q.strip()}
    return {k for k in out if k}


def _q_candidates(ledger, shop, line):
    """Inspected products of the line's category that no stated need or the budget rules out."""
    cat = (line.get("category") or "").lower()
    pool = [p for p in shop.products.values() if p.get("props")
            and (not cat or (p.get("category") or "").lower() == cat)]
    if not pool and cat:
        pool = [p for p in shop.products.values() if p.get("props")]
    cap, qty = ledger.notes.get("q_budget_world"), ledger.quantity or 1
    needs = [n for n in ledger.needs if n.hard and n.op != "ref" and getattr(n, "line", None) in (None, line["i"])]
    kept = [p for p in pool
            if not (cap is not None and p.get("price") is not None and p["price"] * qty > cap + 1e-9)
            and not (CONFIG["q_filter_stated"] and any(need_status(n, p) is False for n in needs))]
    return kept if len(kept) >= CONFIG["q_min_pool"] else pool


def _q_numeric(value) -> bool:
    return bool(re.match(r"^\s*-?\d[\d.,]*(?:\s*-\s*\d[\d.,]*)?\s*[a-z\"]{0,8}\s*$", value))   # K3: a range too


def _q_split(forms, shown, key) -> float:
    """Q2: the expected share of the candidates holding the key that a reply naming one of its values rules out,
    read on list items: a value every holder lists rules out nothing, so "x" against "x, y" splits little. q_rank
    counts it for the key's categorical values only: a measure is stated as a bound, not as one exact value."""
    held = []
    for f, c in forms.items():
        items = {" ".join(toks(i)) for i in re.split(r"\s*(?:[,;/|]|\bor\b|\band\b)\s*", shown[(key, f)]) if toks(i)}
        held += [items] * c
    if len(held) < 2:
        return 0.0
    count = Counter(i for items in held for i in items)
    total = sum(count.values())
    return sum(c / total * (1 - c / len(held)) for c in count.values())


def q_rank(ledger, shop, line) -> list:
    """[(score, key, example values)] for one line, best first. Every feature is measured on the
    current candidates' facts and the request text."""
    pool = _q_candidates(ledger, shop, line)
    if len(pool) < 2:
        return []
    settled = _q_settled(ledger, line["i"])
    noun = set(toks(line.get("noun") or "")) | set(toks(line.get("category") or ""))

    def core(key):   # a key's own words, without the line's product words ("fit of jeans" -> fit)
        return (set(toks(key)) - noun) or set(toks(key))
    settled_toks = [core(k) for k in settled]
    fields = {_q_key(k) for p in pool for k in p if k != "props"}   # listing-level fields: identity, price, stock
    table, shown = {}, {}
    for p in pool:
        for k, v in (p.get("props") or {}).items():
            key, val = _q_key(k), " ".join(str(v).lower().split())
            if key and val and val not in ("n/a", "none", "-", "not specified", "unknown"):
                form = "".join(toks(val)) or val   # "16 mb" and "16mb" are one value
                table.setdefault(key, Counter())[form] += 1
                shown.setdefault((key, form), val)
    vals_of = {k: [shown[(k, f)] for f in forms] for k, forms in table.items()}   # QK2
    request = set(toks((CONFIG.get("q_line_hint") and ledger.notes.get("line_text")) or ledger.query))
    picks = [set(toks(p)) for p in list(ledger.notes.get("q_plan") or []) + list(ledger.notes.get("likely_hidden") or [])]
    stated = [(set(toks(n.prop or "")), set(toks(" ".join(n.value) if isinstance(n.value, list) else str(n.value or ""))))
              for n in ledger.needs if n.prop and getattr(n, "line", None) in (None, line["i"])]
    out = []
    weights = CONFIG["q_weights"]
    for key, forms in table.items():
        kt = set(key.split())
        values = Counter({shown[(key, f)]: c for f, c in forms.items()})
        if key in fields or key in settled or _is_options(key):
            continue   # settled or an option list
        if any(s != key and s in vals_of and _q_alias(vals_of[key], vals_of[s]) for s in settled):
            continue   # settled or an option list
        spare = len(values) < 2 or any(
                all(any(_q_same_word(w, u) for u in s) for w in core(key))
                or all(any(_q_same_word(w, u) for u in core(key)) for w in s) for s in settled_toks)
        if spare and not CONFIG.get("q_spare", True):
            continue   # one value among candidates, or a near-duplicate of a settled key ("packaging type" after
            #            "pack type", "jeans fit type" after "fit")
        if sum(len(v) for v in values) / len(values) > 40 or len(kt) > 4 or re.search(r"\d|\be g\b", key):
            continue   # free text or a malformed key, not a property with values
        n = sum(values.values())
        if any(looks_like_id(v) for v in values) or (
                n >= 4 and len(values) == n and not any(_q_numeric(v) for v in values)):
            continue   # identifies a listing (every candidate differs, not a measure): a selector, not a preference
        if sum(c for v, c in values.items() if re.search(r"[a-z]+\d+[a-z0-9]*\d|\d+[a-z]+\d", v.replace(" ", ""))) / n >= 0.5:
            continue   # Q6-H2: most values carry a letter-digit code (a model or part number): a selector as well
        coded = sum(c for v, c in values.items() if any(re.search(r"[a-z]", w) and len(re.findall(r"\d", w)) >= 2
                                                        and len(w) >= 5 for w in v.split()))
        family = Counter(w for v in values for w in set(v.split()) if w.isalpha() and len(w) >= 3 and w not in _FUNCTION_WORDS)
        if coded * 2 > n and not any(c >= 2 for c in family.values()):   # K2F: coded values of one family ("intel ...")
            continue   # K2: mostly codes (letters and digits in one word): names a listing, not a preference
        vt = set(toks(" ".join(values)))
        if any(pt & set(toks(key)) and val & vt for pt, val in stated):
            continue   # a stated need names this key loosely ("<value> on the <part>" -> <part> <value>)
        hint = bool(set(toks(key)) & request) or any(set(toks(v)) and set(toks(v)) <= request for v in values)
        feats = {"h": -sum(c / n * math.log(c / n) for c in values.values()) / math.log(max(2, len(pool))),
                 "cov": n / len(pool), "top": values.most_common(1)[0][1] / n,
                 "dup": float(any(o != key and set(o.split()) < kt for o in table)),
                 "hint": float(hint), "model": max((1.0 / (1 + r) for r, p in enumerate(picks) if p and p <= set(toks(key))), default=0.0),
                 "num": sum(c for v, c in values.items() if _q_numeric(v)) / n,   # a measure: categorical properties first
                 "split": _q_split(forms, shown, key) * (1 - sum(c for v, c in values.items() if _q_numeric(v)) / n)}
        score = sum(w * feats.get(f, 0.0) for f, w in weights.items())
        if spare:   # Q6-H5: asked only once every other key is spent: an unused message reveals nothing
            score -= 10.0
        examples = [v for v, _ in values.most_common() if len(v) <= 30 and not looks_like_id(v)][:2]
        out.append((round(score, 4), key, examples))
    out.sort(key=lambda r: (-r[0], r[1]))
    return out


def q_text(key, examples, noun=None) -> str:
    """One property per message, named by its fact key, with up to two values seen on candidates."""
    if CONFIG.get("q_plain"):   # QT: one sentence; a line name's "and" / "&" becomes "/"; examples joined by "or"
        where = " for the " + re.sub(r"\s+(?:and|&)\s+", "/", noun) if noun else ""
        ex = [e for e in examples if not re.search(r"\s(?:and|&)\s", f" {e} ")][:2]
        tail = f", for example {ex[0]} or {ex[1]}" if len(ex) >= 2 else (f", for example {ex[0]}" if ex else "")
        return f"Which {key} would you prefer{where}{tail}?"
    where = f" for the {noun}" if noun else ""
    seen = f" I can see {examples[0]} and {examples[1]} among the options." if len(examples) >= 2 else (
        f" I can see {examples[0]} among the options." if examples else "")
    return f"Which {key} would you prefer{where}?{seen}"


def q_plan_next(ledger, shop, messages_left, llm=None, reasoning=None):
    """The next (line index, key, question), or None. Spreads questions over the order lines
    (fewest asked first, then best score) and keeps CONFIG['q_reserve'] messages back when the
    cap is 4 or more."""
    cap = ledger.msg_budget if ledger.msg_budget is not None else 2
    reserve = CONFIG["q_reserve"] if cap >= 4 else 0
    if messages_left - reserve <= 0:
        return None
    lines = q_lines(ledger)
    asked = Counter(e[0] for e in ledger.notes.get("q_log", []))
    best = []
    for line in lines:
        ranked = q_rank(ledger, shop, line)
        if ranked:
            best.append((asked[line["i"]], -ranked[0][0], line, ranked))
    if not best:
        return None
    best.sort(key=lambda b: (b[0], b[1], b[2]["i"]))
    _, _, line, ranked = best[0]
    if llm is not None and reasoning is not None and "q_plan" not in ledger.notes and CONFIG["q_weights"].get("model", 0) > 0:
        data = reasoning._ask(llm, _SYS_QUESTIONS, {   # one call per episode; its picks add a bonus, never override
            "request": ledger.query, "already_settled": sorted(_q_settled(ledger, line["i"])),
            "properties": {k: ex for _, k, ex in ranked[:20]}, "max": messages_left}, 200)
        ledger.notes["q_plan"] = [_q_key(p) for p in (data or {}).get("ask") or [] if isinstance(p, str)]
        if ledger.notes["q_plan"]:
            ranked = q_rank(ledger, shop, line) or ranked
    _, key, examples = ranked[0]
    noun = line["noun"] if len(lines) > 1 or q_siblings(ledger) else None
    return line["i"], key, q_text(key, examples, noun)


def _q_line_overlap(text, phrases, key=None) -> int:
    """How many words of the line's product / category phrases the reply uses (the property's own
    words do not count)."""
    have = set(toks(text)) - set(toks(key or ""))
    want = set()
    for phrase in phrases:
        want |= set(toks(phrase or ""))
    return len(have & want)


def q_line_of(ledger, text, asked_line=0, key=None) -> int:
    """The order line a reveal names (the line whose product words it uses most), else the line just asked."""
    lines = q_lines(ledger)
    if len(lines) > 1:
        scored = sorted(((_q_line_overlap(text, [l["noun"], l.get("category")], key), l["i"]) for l in lines), reverse=True)
        if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            return scored[0][1]
    return asked_line


def q_note_reply(ledger, kind):
    """Record the reply kind on the last planned question (every asked key is settled for its line)."""
    log = ledger.notes.get("q_log") or []
    if log and log[-1][3] in (None, "unknown"):
        log[-1][3] = kind


_Q_BOUND = re.compile(r"\b(at\s+(?:most|least)|(?:no|not)\s+(?:more|less)\s+than|up\s+to|(?:maximum|minimum)\s+of)\s+"
                      r"(-?\d[\d.,]*\s*(?:\"|[a-z]+)?)", re.IGNORECASE)


def _q_bound_op(words) -> str:
    """'le' for an upper bound (most, more, up to, maximum), else 'ge'."""
    return "le" if set(words.lower().split()) & {"most", "more", "up", "maximum"} else "ge"


def q_parse_reveal(text, key, values=(), vocab=(), noun=""):
    """(op, value) when a reply carries a value of the asked property, else None. Content only:
    the reply must use the property's words and hold a number with a bound, or one of the values
    the candidates show for that property (the longest match wins), or (v3) a value span after the
    property's words: the longest trailing run of words the property's values / the category's
    listings use, else the span's last word."""
    low = " ".join(str(text or "").lower().split())
    if not key or not set(toks(key)) or not set(toks(key)) <= set(toks(low)):
        return None
    if parse_reference(text):
        return None
    m = _Q_BOUND.search(low)
    if m:
        return _q_bound_op(m.group(1)), m.group(2).strip()
    ws_ = re.findall(r"[a-z0-9][a-z0-9'\-]*", low)
    negated = {w for a, w in zip(ws_, ws_[1:]) if a in ("no", "not", "without", "never")}   # NEGV
    have = set(toks(low)) - set(toks(key)) - set(toks(noun)) - negated   # the line's own product words are not a value
    known = [v for v in values if set(toks(v)) and set(toks(v)) <= have]
    if known:
        return "eq", max(known, key=lambda v: len(toks(v)))
    heads = {toks(v)[0] for v in values if toks(v) and len(toks(v)[0]) >= 3 and not toks(v)[0][0].isdigit()}
    named = sorted(h for h in heads & have)   # a value named without its qualifiers ("<value>" for "<value> <spec>")
    if len(named) == 1:
        return "eq", named[0]
    span = q_value_span(low, key, noun)
    cut = next((i for i, w in enumerate(span or []) if w in ("no", "not", "without", "never")), None)
    if cut is not None:   # RV4: "<v> and no other": the value span ends before the negation
        span = span[:cut]
        while span and span[-1] in _FUNCTION_WORDS:
            span = span[:-1]
    if not span:
        return None
    own = {w for v in values for w in toks(v)}
    runs = []
    for words_ in (own, own | set(vocab)):
        run = []
        for w in reversed(span):
            if not set(toks(w)) or not set(toks(w)) <= words_ or w in negated:   # NEGV: a negated word ends the run
                break
            run.insert(0, w)
        while run and run[0] in _FUNCTION_WORDS:
            run.pop(0)
        runs.append(run)
    best = max(runs, key=len)
    if best and set(toks(" ".join(best))) <= set(toks(key)):
        cut = max((i for i, w in enumerate(span) if w in _FUNCTION_WORDS), default=-1)
        tail = span[cut + 1:]
        if 2 <= len(tail) <= 4 and set(toks(" ".join(tail))) - set(toks(key)):
            return "eq", " ".join(tail)   # the property's own words alone are not a value: the phrase they end
    if best:
        return "eq", " ".join(best)
    last = span[-1]
    if (len(last) >= 2 or last.isdigit()) and last not in _FUNCTION_WORDS and last not in set(toks(noun)):   # "xs", "4g"
        return "eq", last
    return None


def _jv_words(text):
    """JV: every word of a text, lower case, hyphenated and dotted words whole ("all-in-one", "2.5")."""
    return re.findall(r"[a-z0-9]+(?:[\-.'][a-z0-9]+)*", str(text or "").lower())


def _jv_texts(shop, key=None, cat=None):
    """JV: the token strings the observed listings carry, three tiers: values of `key`, values of any key in `cat`, titles."""
    tiers = ([], [], [])
    for p in (shop.products.values() if shop is not None else ()):
        incat = not cat or (p.get("category") or "").lower() == str(cat).lower()
        for k, v in (p.get("props") or {}).items():
            for item in [str(v)] + re.split(r"[,;/|]", str(v)):
                t = " ".join(_jv_words(item))
                if t:
                    (tiers[0] if key and _q_key(k) == _q_key(key) else tiers[1] if incat else tiers[2]).append(t)
        if p.get("title"):
            tiers[2].append(" ".join(_jv_words(p["title"])))
    return [" | ".join(t) for t in tiers]


def jv_seen(value, shop, key=None, cat=None):
    """JV: the best tier (0 the key's values, 1 the category's values, 2 titles) whose listings show the value's words
    whole; None when none does."""
    vt = " ".join(_jv_words(value))
    return next((i for i, t in enumerate(_jv_texts(shop, key, cat)) if vt and f" {vt} " in f" {t} "), None)


def jv_junk(value, key, noun):
    """JV: a value read from template wording, not a value: it opens with a function word ("of the <line>"), holds only
    the key's and the line's own words, or a third of its words are function words ("i couldn't go home")."""
    w = _jv_words(value)
    if not w:
        return True
    own = set(_jv_words(key)) | set(_jv_words(noun))
    fn = sum(x in _FUNCTION_WORDS for x in w)
    return w[0] in _FUNCTION_WORDS or set(w) <= own | _FUNCTION_WORDS or (len(w) >= 3 and fn * 3 >= len(w))


def jv_candidate(text, key, noun, shop, cat=None, want_tier=False):
    """JV: the value a reply gives for `key` when its wording is new: the longest run of 1-6 words AFTER the key's name
    that an observed listing shows (the key's own values first, then the category's, then titles); ties go to the later
    run. None when the listings show none of the reply's words after the key."""
    words_ = _jv_words(text)
    kt = _jv_words(key)
    at = next((i + len(kt) for i in range(len(words_) - len(kt) + 1) if words_[i:i + len(kt)] == kt), None) if kt else None
    if at is None:
        return None
    skip = set(kt) | set(_jv_words(noun))
    best = None
    for tier, hay in enumerate(_jv_texts(shop, key, cat)):
        hay = f" {hay} "
        for n in (6, 5, 4, 3, 2, 1):
            for i in range(at, len(words_) - n + 1):
                run = words_[i:i + n]
                if i > 0 and words_[i - 1] in ("no", "not", "without", "never"):
                    continue   # NEGV: "<v> and no other": a negated run is not the value
                if (set(run) <= skip | _FUNCTION_WORDS or run[0] in skip or (run[0] in _FUNCTION_WORDS and n < 3)
                        or (n == 1 and (len(run[0]) < 2 and not run[0].isdigit()))):
                    continue
                if f" {' '.join(run)} " in hay:
                    cand = (-tier, n, i, " ".join(run))
                    best = max(best, cand) if best else cand
        if best:
            return (-best[0], best[3]) if want_tier else best[3]
    return None


def q_shown_value(text, key, shop):
    """R6: the longest value the opened listings show for this exact key (a list item counts) that the text names as a
    whole phrase outside the key's own name; None when the text does not name the key."""
    kt = " ".join(toks(key))
    low = " " + " ".join(toks(text)) + " "
    if shop is None or not kt or f" {kt} " not in low:
        return None
    best = None
    for p in shop.products.values():
        for k, v in (p.get("props") or {}).items():
            if _q_key(k) != _q_key(key):
                continue
            for item in [str(v)] + re.split(r"[,;/|]", str(v)):
                vt = " ".join(toks(item))
                if (vt and set(words(item)) - _FUNCTION_WORDS and f" {vt} " in low
                        and f" {kt} " in low.replace(f" {vt} ", " | ", 1)   # the key is named apart from the value
                        and (best is None or len(vt) > len(" ".join(toks(best))))):
                    best = item.strip()
    return best


def q_value_span(low, key, noun=""):
    """The words after the property's name up to the clause end, minus a trailing phrase that names
    the order line's product. Words keep their inner hyphens ("<word>-<word>-<word>")."""
    kt = toks(key)
    m = re.search(r"\b" + r"[\s_-]+(?:[a-z]+[\s_-]+)?".join(re.escape(t) for t in kt) + r"\b", low)   # "<key word> of <key word>"
    if not m:
        return []
    rest = re.split(r"[,;:.!?()]| or | but | because | so ", low[m.end():] + " ")[0]
    words_ = [w.strip("'\"") for w in rest.split() if w.strip("'\"")]
    skip = set(toks(noun)) | {"of", "the", "a", "an", "for", "on"}
    if words_ and words_[0] in ("of", "for", "on"):
        while words_ and (words_[0] in skip or set(toks(words_[0])) <= skip):
            words_.pop(0)
    return words_


def q_siblings(ledger) -> list:
    """The other lines' ledgers when the core runs one ledger per order line (drive_lines)."""
    return [led for led in ledger.notes.get("q_siblings") or [] if led is not ledger]


def q_route(ledger, text, key=None):
    """The ledger a reveal belongs to: a sibling line's when the reply uses that line's product
    words more than this line's, else this one."""
    others = q_siblings(ledger)
    if not others:
        return ledger

    def overlap(led):
        return _q_line_overlap(text, [led.product, led.category, led.notes.get("q_category")], key)
    best = max([ledger] + others, key=overlap)
    return best if overlap(best) > overlap(ledger) else ledger


def _q_absorb_reply(core_absorb):
    """Wrap the core's absorb(): after it has read a shopper reply, (1) when it found no need and
    the reply carries a value of the asked property, add that need; (2) move needs and references the
    reply tied to another order line to that line's ledger; (3) record the reply on the question log."""
    def absorb(self, ledger, lines, source, shop, llm):
        if source != "reply":
            return core_absorb(self, ledger, lines, source, shop, llm)
        tags = set()
        for line in lines or []:
            before, refs_before = list(ledger.needs), len(ledger.notes.get("refs", []))
            tags |= core_absorb(self, ledger, [line], source, shop, llm) or set()
            if not isinstance(line, str):
                continue
            asked = ledger.asked[-1] if ledger.asked else None
            log = ledger.notes.get("q_log") or []
            new = [n for n in ledger.needs if n not in before]
            noun = " ".join(x for x in (ledger.product, ledger.category, ledger.notes.get("q_category")) if x)
            junk = [n for n in new if isinstance(n.value, str) and set(toks(n.value))
                    and set(toks(n.value)) <= set(toks(noun)) | (set() if getattr(n, "clause", None) == "rv" else _FUNCTION_WORDS)]
            if junk:   # the core read the line's own product words as the value ("of the <line product>")
                ledger.needs = [n for n in ledger.needs if n not in junk]
                new = [n for n in new if n not in junk]
            cat = ledger.category or ledger.notes.get("q_category")
            vocab = set()
            for p in (shop.products.values() if shop is not None else ()):
                if not cat or (p.get("category") or "").lower() == str(cat).lower():
                    for k, v in (p.get("props") or {}).items():
                        vocab |= set(toks(str(v)))
            for n in list(new):   # the core kept a lead-in with the value ("in the form of stripe"): the value's own words
                lead = words(n.value)[:1] if isinstance(n.value, str) else []
                # a lead-in is a function word, or a first word no listing value of this category uses ("fixed with <v>")
                foreign = bool(lead) and len(words(n.value)) > 1 and bool(vocab) and not set(toks(lead[0])) <= vocab
                if (n.op == "eq" and n.prop and lead and getattr(n, "clause", None) != "rv"
                        and (lead[0] in _FUNCTION_WORDS or (foreign and not getattr(n, "clause", None)))):
                    parsed = q_parse_reveal(line, _q_key(n.prop), shop.values_of(n.prop) if shop is not None else (), vocab, noun)
                    if parsed and parsed[0] == "eq" and set(toks(parsed[1])) < set(toks(n.value)):
                        n.value = parsed[1]
                    elif not parsed and lead[0] in _FUNCTION_WORDS:   # no value of that property in the reply: not a need
                        ledger.needs = [m for m in ledger.needs if m is not n]
                        new.remove(n)
            for n in new:   # R6: the core read a word next to the property ("<key> closed, with <v>"); a shown value wins
                if n.op == "eq" and n.prop and isinstance(n.value, str) and n.source == "shopper" and not getattr(n, "clause", None):
                    hit = q_shown_value(line, n.prop, shop)
                    if hit and " ".join(toks(hit)) != " ".join(toks(n.value)):
                        n.value = hit
            if not new and asked:
                # the asked key first, then any fact key the reply names (the shopper may answer for a
                # nearby property), longest name first
                said = set(toks(line))
                keys = [asked] + sorted({_q_key(k) for p in (shop.products.values() if shop is not None else ())
                                         for k in (p.get("props") or {})
                                         if set(toks(_q_key(k))) and set(toks(_q_key(k))) <= said} - {asked},
                                        key=lambda k: -len(k))
                for key in keys:
                    hit = q_shown_value(line, key, shop)   # R6: a value the listings show for the key, whole
                    parsed = ("eq", hit) if hit else q_parse_reveal(line, key, shop.values_of(key) if shop is not None else (), vocab, noun)
                    if parsed:
                        need = Need(key, parsed[1], "shopper", True, parsed[0])
                        self._set_need(ledger, need)
                        new, tags = [need], tags | {"need"}
                        break
                if not new and CONFIG.get("jv_support") and asked and shop is not None and shop.products:
                    alt = jv_candidate(line, asked, noun, shop, cat, want_tier=True)   # JV: a value the key's listings show
                    alt = alt[1] if alt and alt[0] == 0 else None
                    if alt:
                        need = Need(asked, alt, "shopper", True, "eq")
                        self._set_need(ledger, need)
                        new, tags = [need], tags | {"need"}
                if not new:   # RV2: it may name a property no inspected listing carries yet; read it again later
                    if line not in (ledger.notes.get("q_unread") or []):
                        ledger.notes.setdefault("q_unread", []).append(line)
            if CONFIG.get("jv_support") and shop is not None and shop.products and not _Q_BOUND.search(line):
                for n in list(new):   # JV: a value no observed listing shows is template wording unless nothing better exists
                    if not (n.op in ("eq", "in") and n.prop and isinstance(n.value, str) and n.source == "shopper") or getattr(n, "clause", None):
                        continue
                    got = jv_candidate(line, n.prop, noun, shop, cat, want_tier=True)
                    seen = jv_seen(n.value, shop, n.prop, cat)
                    junk = jv_junk(n.value, n.prop, noun)
                    if seen is not None and not junk:
                        if got and ((got[0] == 0 and seen > 0) or (got[0] <= seen and set(_jv_words(n.value)) < set(_jv_words(got[1])))):
                            n.value = got[1]   # the key's own shown value, or a longer run holding this fragment ("os" of "<word> os")
                        continue
                    if got and (got[0] == 0 or junk):
                        n.value = got[1]
                    elif not junk:
                        continue   # an unseen value that reads like a value ("ruffle"): kept
                    else:   # template wording: read it again once more listings are open (RV2)
                        ledger.needs = [m for m in ledger.needs if m is not n]
                        new.remove(n)
                        if line not in (ledger.notes.get("q_unread") or []):
                            ledger.notes.setdefault("q_unread", []).append(line)
            bound = _Q_BOUND.search(line)
            for n in new:   # a value the reply bounds is a bound ("... at most <n> <unit>" read as "<n>")
                number = re.match(r"-?\d[\d.,]*", bound.group(2)).group(0).rstrip(".,") if bound else None
                lead = re.match(r"-?\d[\d.,]*", str(n.value or "").strip())
                if (number and n.op in ("eq", "ge", "le") and isinstance(n.value, str)   # BND: template words or a cut decimal
                        and (number in n.value or not re.search(r"\d", n.value) or (lead and number.startswith(lead.group(0))))):
                    n.op = _q_bound_op(bound.group(1))
                    n.value = bound.group(2).strip()
            refs = ledger.notes.get("refs", [])[refs_before:]
            q_note_reply(ledger, "value" if new else ("reference" if refs else "none"))
            if len(q_lines(ledger)) > 1:   # one ledger holds every line: tie unscoped needs to the line named
                where = q_line_of(ledger, line, log[-1][0] if log else 0, asked)
                for n in new:
                    if getattr(n, "line", None) is None and not getattr(n, "scope", None):
                        n.line = where
            target = q_route(ledger, line, asked)
            if target is not ledger and (new or refs):   # the reply names another line: its ledger owns it
                ledger.needs = [n for n in ledger.needs if n not in new]
                for n in new:
                    self._set_need(target, n)
                if refs:
                    ledger.notes["refs"] = ledger.notes["refs"][:refs_before]
                    target.notes.setdefault("refs", []).extend(refs)
        return tags
    return absorb


Reasoning.absorb = _q_absorb_reply(Reasoning.absorb)


# ---- generic reply phrasing (claude-3-17): polarity, comparatives, plain-word references, free choice ----------------
_P_WORDS = re.compile(r"[a-z0-9]+(?:[\-.'][a-z0-9]+)*|[.;!?,]", re.IGNORECASE)
_P_NEGATORS = {"not", "no", "never", "without", "avoid", "skip", "exclude", "neither", "nor"}
_P_EXCL = {"but", "except", "besides", "bar"}
_P_UNIVERSAL = {"anything", "any", "whatever"}   # "everything / all" + "but" is no exclusion on its own (a negative verb may lead it)
_P_ONLY = {"nothing", "none", "nowt", "else", "only", "just"}
_P_CMP_AFTER = re.compile(r"(?<![\w.])(-?\d[\d.,]*\s*(?:\"|[a-z]+)?)\s+or\s+(more|above|higher|greater|over|bigger|larger|longer|"
                          r"less|under|below|lower|fewer|smaller|shorter)\b", re.IGNORECASE)
_P_CMP_BEFORE = re.compile(r"\b(over|above|more than|greater than|higher than|bigger than|larger than|longer than|exceeding|"
                           r"under|below|less than|fewer than|lower than|smaller than|shorter than|within)\s+(-?\d[\d.,]*\s*(?:\"|[a-z]+)?)",
                           re.IGNORECASE)
_P_GE = {"more", "above", "higher", "greater", "over", "bigger", "larger", "longer", "exceeding"}
_P_LIKE = re.compile(r"\b(?:same|match(?:es|ing)?|go by|copy|find on|found on|like the|as on|as the)\b", re.IGNORECASE)
_P_DIFF = re.compile(r"\b(?:except|other than|differ(?:s|ent)?|not the same|anything but|unlike|apart from)\b", re.IGNORECASE)
_P_FREE = re.compile(r"\b(?:whichever|whatever|any|either)\b", re.IGNORECASE)
_P_FIRM = re.compile(r"\b(?:only|must|need|needs|has to|have to|required|nothing (?:but|else))\b", re.IGNORECASE)


def p_words(text):
    return [w.lower() for w in _P_WORDS.findall(str(text or ""))]


def p_negated(text, value):
    """NEG: True when the reply negates `value` (an odd number of negators in its clause before it)."""
    ws, vw = p_words(text), p_words(value)
    if not vw:
        return False
    at = next((i for i in range(len(ws) - len(vw) + 1) if ws[i:i + len(vw)] == vw), None)
    if at is None:
        return False
    # the value's own clause; a gerund "being" opens one too (a negator before it belongs to the main verb, not the value)
    start = max((i + 1 for i in range(at) if ws[i] in ".;!?," or ws[i] == "being"), default=0)
    window = ws[max(start, at - 20):at]
    count = 0
    for i, w in enumerate(window):
        if w in _P_NEGATORS or w.endswith("n't"):
            count += 1
        elif w in _P_EXCL or (w == "than" and i and window[i - 1] == "other"):
            gov = next((x for x in reversed(window[:i]) if x in _P_UNIVERSAL | _P_ONLY), None)
            if gov in _P_UNIVERSAL:
                count += 1
    return count % 2 == 1


def p_negated_span(text, key):
    """NEG: the value a negated reply names when no reader found one: the words after the exclusion cue, up to a
    preposition that leads to the key or the clause end ("anything but <v> for the <key>")."""
    low = " ".join(str(text or "").lower().split())
    m = re.search(r"\b(?:anything (?:but|except|other than|besides)|any \w+(?: \w+)? except|other than|except(?: for)?|"
                  r"not|no|avoid|skip anything with|skip|don't want|do not want|without)\s+(?P<v>[a-z0-9][a-z0-9 \-/&]{0,40}?)"
                  r"(?=\s+(?:for|as|in|on|of)\s+(?:the|my|its|their)\b|[,.;!?]|$)", low)
    if not m:
        return None
    v = [w for w in m.group("v").split() if w not in _FUNCTION_WORDS]
    if not v or len(v) > 4 or set(v) & set(toks(key)):
        return None
    return " ".join(v)


def p_bound(text):
    """CMP: (op, "<n> <unit>") of a comparative bound in the reply, or None."""
    m = _P_CMP_AFTER.search(str(text or ""))
    if m:
        return ("ge" if m.group(2).lower() in _P_GE else "le"), m.group(1).strip()
    m = _P_CMP_BEFORE.search(str(text or ""))
    if m:
        return ("ge" if m.group(1).lower().split()[0] in _P_GE else "le"), m.group(2).strip()
    return None


def p_plain_ref(text):
    """GREF: (item, seller, quals, neg) of a reference in plain words, or None."""
    t = " ".join(str(text or "").split())
    if not _P_LIKE.search(t) and not _P_DIFF.search(t):
        return None
    sellers = [m for m in re.finditer(r"(?<=[a-z,] )([A-Z][a-z]{2,})\b", t) if m.group(1) not in _FUNCTION_CAPS]
    for m in sellers:
        s = m.group(1)
        before, after = t[:m.start()], t[m.end():]
        item = None
        mm = re.search(r"\bthe\s+([a-z][a-z \-]{1,30}?)\s+(?:that|which)\s*$", before)   # the <item> that <S> sells
        if mm and re.match(r"\s+(?:sells|lists|offers|has|carries|stocks)\b", after):
            item = mm.group(1)
        mm = re.search(r"\bthe\s+([a-z][a-z \-]{1,30}?)\s+(?:from|by|listed by|sold by|made by)\s*$", before)
        item = item or (mm.group(1) if mm else None)
        mm = re.match(r"\s+([a-z][a-z \-]{1,30}?)(?=\s+with\b|[,.;]|$)", after)   # the <S> <item>
        if not item and mm and re.search(r"\bthe\s*$", before):
            item = mm.group(1)
        if not item:
            continue
        q = re.search(r"\bwith\s+([^.;]{2,80})", after)
        quals = re.sub(r",\s*(?:please|thanks|if that's alright|and that's final)\b.*$", "", q.group(1)).strip(" ,") if q else ""
        return item.strip(), s, quals, bool(_P_DIFF.search(before))
    return None


def _para_absorb(inner):
    """claude-3-17: replies in unseen phrasings: plain-word references become the reference form, negated values become
    exclusions, comparatives become bounds, free choices read no value."""
    def absorb(self, ledger, lines, source, shop, llm):
        if source != "reply":
            return inner(self, ledger, lines, source, shop, llm)
        tags = set()
        for line in lines or []:
            if not isinstance(line, str):
                tags |= inner(self, ledger, [line], source, shop, llm) or set()
                continue
            asked = ledger.asked[-1] if ledger.asked else None
            ref = (p_plain_ref(line) if CONFIG.get("p_gref") and asked and parse_reference(line) is None else None)
            before = list(ledger.needs)
            refs_before = len(ledger.notes.get("refs", []))
            if ref:   # GREF: read it as the reference form the agent follows
                item, seller, quals, neg = ref
                said = f"For the {asked}, go by listing {item} sold by {seller}" + (f", with {quals}." if quals else ".")
                tags |= inner(self, ledger, [said], source, shop, llm) or set()
                for r in ledger.notes.get("refs", [])[refs_before:]:
                    if isinstance(r.get("ref"), dict):
                        r["ref"]["neg"] = neg
                continue
            tags |= inner(self, ledger, [line], source, shop, llm) or set()
            new = [n for n in ledger.needs if n not in before]
            cat = ledger.category or ledger.notes.get("q_category")
            if CONFIG.get("p_cmp") and not _Q_BOUND.search(line):   # the core's bound grammar ("no less than") reads its own
                b = p_bound(line)
                if b:
                    num = re.match(r"-?\d[\d.,]*", b[1]).group(0).rstrip(".,")
                    hit = [n for n in new if n.prop and isinstance(n.value, str) and (num in n.value or not re.search(r"\d", n.value))]
                    for n in hit:   # CMP: "<n> <unit> or more" was read as the value itself
                        n.op, n.value = b
                    if not new and asked and set(toks(asked)) <= set(toks(line)):
                        need = Need(asked, b[1], "shopper", True, b[0])
                        self._set_need(ledger, need)
                        new, tags = [need], tags | {"need"}
            if CONFIG.get("p_neg"):
                for n in new:
                    if n.op in ("eq", "in") and isinstance(n.value, str) and p_negated(line, n.value):
                        n.op = "ne"   # NEG: "anything but <v>", "not <v>", "I don't want <v>"
                if not new and asked and set(toks(asked)) <= set(toks(line)):
                    v = p_negated_span(line, asked)
                    if v and p_negated(line, v) and shop is not None and jv_seen(v, shop, asked, cat) is not None:   # a shown value only
                        need = Need(asked, v, "shopper", True, "ne")
                        self._set_need(ledger, need)
                        new, tags = [need], tags | {"need"}
            if CONFIG.get("p_free") and new and _P_FREE.search(line) and not _P_FIRM.search(line):
                for n in list(new):   # FREE: "pick whichever <key> you like": no value
                    if n.op in ("eq", "in") and isinstance(n.value, str) and shop is not None and jv_seen(n.value, shop, n.prop, cat) != 0:
                        ledger.needs = [m for m in ledger.needs if m is not n]
                        new.remove(n)
        return tags
    return absorb


Reasoning.absorb = _para_absorb(Reasoning.absorb)


REASONING = Reasoning()


def read_request(text, llm=None):
    """R1 entry point: the Ledger this agent reads from one request. `llm` is a ModelChain (None: rules only)."""
    return REASONING.extract(str(text or ""), None, llm)


# ======================================================================================
# 7. Shop: catalogue, pagination state and our model of the cart
# ======================================================================================

_DULL_PROPS = {"options", "model", "id", "sku", "condition", "asin", "upc", "ean", "gtin", "mpn"}


class Shop:
    def __init__(self):
        self.products = {}      # id -> normalised product, merged across every view
        self.views = {}         # id -> {view ("listing", "detail", "spec"): step last seen}
        self.fresh = {}         # id -> step of the latest detail or spec view
        self.queries = {}       # (role, phrase) -> {"pages", "next", "done"}
        self.cart = {}          # id -> units (our model; synced whenever the world shows the cart)
        self.cart_synced = True
        self.cart_version = 0       # bumped on every cart change
        self.cart_reread = -1       # cart_version at the last pre-checkout re-read
        self.stale = set()      # ids whose price or stock a shopper line said may have changed
        self.detail_views = 0
        self.props_seen = False   # has any detail view shown a property table?

    def merge(self, product, view, step):
        pid = product["id"]
        current = self.products.setdefault(pid, {"id": pid, "props": {}})
        for key, value in product.items():
            if key == "props":
                current["props"].update(value or {})
            elif value not in (None, ""):
                current[key] = value
        self.views.setdefault(pid, {})[view] = step
        if view in ("detail", "spec"):
            self.detail_views += 1
            self.props_seen = self.props_seen or bool(product.get("props"))
            current["_textonly"] = not self.props_seen and self.detail_views >= 3
        if view in ("detail", "spec") and (product.get("price") is not None or product.get("in_stock") is not None):
            self.fresh[pid] = step  # only a view that shows price or stock refreshes them
            self.stale.discard(pid)

    def has_view(self, pid, view) -> bool:
        return view in self.views.get(pid, ())

    def is_fresh(self, pid) -> bool:
        """No shopper line since our last look says this price or stock changed."""
        return pid not in self.stale

    def view_fresh(self, pid, view) -> bool:
        """We hold this view and nothing since has made it stale."""
        return view in self.views.get(pid, {}) and pid not in self.stale

    def sync_cart(self, cart, step):
        lines = [l for l in cart.get("lines") or [] if l.get("id")]
        if cart.get("partial"):
            for line in lines:
                self.cart[line["id"]] = line.get("qty") or 0
        else:
            self.cart = {l["id"]: (l.get("qty") or 0) for l in lines}
            self.cart_synced = True
        for line in lines:
            if line.get("price") is not None and line["id"] in self.products:
                self.products[line["id"]]["price"] = line["price"]
                self.fresh[line["id"]] = step
                self.stale.discard(line["id"])

    def note_query(self, key, obs):
        state = self.queries.setdefault(key, {"pages": 0, "next": None, "done": False})
        state["pages"] += 1
        if obs is not None and obs.kind == "listing":
            state["shown"] = state.get("shown", 0) + len(obs.products or [])
            state.setdefault("ids", []).extend(p.get("id") for p in (obs.products or [])   # R9 P3: rank order
                                               if isinstance(p, dict) and p.get("id"))
            if isinstance(obs.total, int):
                state["total"] = obs.total
        state["next"] = obs.next if obs is not None else None
        state["done"] = (obs is None or obs.kind != "listing" or not obs.products or obs.next is None
                         or state["pages"] >= CONFIG["max_pages"])

    def _prop_values(self):
        table = {}
        for p in self.products.values():
            for key, value in (p.get("props") or {}).items():
                name = " ".join(words(key))
                if not name or set(name.split()) & _DULL_PROPS or not str(value).strip():
                    continue
                table.setdefault(name, set()).add(str(value).strip().lower())
        return table

    def question_names(self) -> list:
        """Plain property names worth a question, best first: carried by most detailed candidates,
        with several values that candidates share, words rather than numbers."""
        groups, detailed = {}, 0
        for p in self.products.values():
            props = p.get("props") or {}
            detailed += bool(props)
            for key, value in props.items():
                name = plain_key(key)
                if not name or set(name.split()) & _DULL_PROPS or _is_options(key) or not str(value).strip():
                    continue
                g = groups.setdefault(name, {"ids": set(), "values": []})
                if p["id"] not in g["ids"]:
                    g["ids"].add(p["id"])
                    g["values"].append(canon(value))
        ranked = []
        for name, g in groups.items():
            n, distinct = len(g["ids"]), len(set(g["values"]))
            if distinct < 2 or n < 2:
                continue
            numeric = sum(bool(re.search(r"\d", v)) for v in g["values"]) / n
            score = (n / max(1, detailed)) * (1.0 - (distinct - 1) / n) * (1.0 - 0.7 * numeric) - 0.05 * len(name.split())
            ranked.append((score, name))
        return [name for score, name in sorted(ranked, reverse=True) if score > 0.05]

    def discriminating_props(self, exclude=()):
        """Property names whose values differ across the products seen, most varied first."""
        asked = [set(words(e)) for e in exclude]
        table = self._prop_values()
        ranked = sorted(((len(v), k) for k, v in table.items() if len(v) >= 2), reverse=True)
        return [k for _, k in ranked if not any(set(k.split()) & a for a in asked)]

    def values_of(self, prop):
        target = set(words(prop))
        out = set()
        for name, values in self._prop_values().items():
            if target & set(name.split()):
                out |= values
        return out


# ======================================================================================
# 8. Run (one episode's state and the shared phases) and the two drivers
# ======================================================================================

def _np_candidates(text) -> list:
    """S3-P1: product candidates (phrase, head word) from the request's grammar: "a/an/some/any <np>",
    "this/that/these/those <np>" and a sentence-initial "<Np> is/are". Ordered by how often the head word
    recurs in the request (singular and plural folded), then by position; one per head word."""
    t = " ".join(str(text or "").split())
    low = t.lower()
    found = []
    for kind, rx in ((1, _INDEFINITE),
                     (0, re.compile(r"\b(?:this|that|these|those)\s+(?P<p>[a-z][a-z0-9'\-/&]*(?:\s+[a-z][a-z0-9'\-/&]*){0,2})", re.IGNORECASE)),
                     (0, re.compile(r"(?:^|[.!?]\s+)(?P<p>[A-Z][a-z0-9'\-/&]*(?:\s+[a-z][a-z0-9'\-/&]*){0,2}?)\s+(?:is|are)\b"))):
        for m in rx.finditer(t):
            ws = []
            for w in m.group("p").lower().split():
                w = w.strip(" ,.?!;:()")
                if not w or "'" in w or w in _STOP or w in _FUNCTION_WORDS or w in (
                        "like", "without", "about", "over", "under", "before", "after", "around", "through", "than"):
                    break
                ws.append(w)
            while ws and ws[0] in ("new", "pair"):
                ws = ws[1:]
            phrase = " ".join(ws)
            if (not ws or len(ws) > 3 or _VALUE_JUNK.search(phrase)
                    or set(ws) & (_NARRATIVE | _NOT_PRODUCT | _PRONOUNS | _ATTR_JUNK)):
                continue

            def freq(w):
                stem = w[:-1] if w.endswith("s") else w
                return len(re.findall(r"\b" + re.escape(stem) + r"(?:s|es)?\b", low))
            head = ws[-1]   # the English head noun of the phrase
            if len(head) >= 3:
                found.append((-freq(head), kind, m.start(), phrase, head))
    out, heads = [], set()
    for _, _, _, phrase, head in sorted(found):
        if head not in heads:
            heads.add(head)
            out.append((phrase, head))
    return out


class Run:
    def __init__(self, problem_data, env_client, llm_client=None, clock=time.monotonic, reasoning=None,
                 discovery_client=None):
        problem_data = problem_data if isinstance(problem_data, dict) else {}
        env = problem_data.get("environment") if isinstance(problem_data.get("environment"), dict) else problem_data
        policy = env.get("policy_view") if isinstance(env.get("policy_view"), dict) else {}
        binding = env.get("binding") if isinstance(env.get("binding"), dict) else (
            {"session_id": env["session_id"]} if env.get("session_id") else {})
        query = policy.get("query")
        self.query = query if isinstance(query, str) else ("" if query is None else json.dumps(query, ensure_ascii=False, default=str))  # C0055
        self.opening = policy.get("opening")
        self.adapter = WorldAdapter(policy.get("tools") or [])
        self.env = EnvSession(env_client, binding, policy.get("max_steps") or 30, clock)
        self.env.finished = bool(policy.get("finished"))
        self.R = reasoning or REASONING
        if CONFIG["actor"] == "rules":
            llm_client = None   # rules mode makes no model calls at all
        self.llm = ModelChain(llm_client, clock=clock, deadline=self.env.deadline) if llm_client else None
        if self.llm is not None:
            self.llm.discover(discovery_client)
        self.trace = []
        self.shop = Shop()
        self.sent_messages = 0
        self.status_cap = None    # G4: the message cap a status tool reported
        self.status_read = False
        self.msg_exhausted = False
        self.events = set()
        self.rejected = set()
        self.line_debt = 0          # extra lines of array cart calls (a world may count each as a step)
        self.category = None        # the catalogue category chosen for this request
        self.checkouts = 0          # order attempts so far (the first one meets every pending change)
        self.forced_pick = None     # the decline guard's choice when nothing is confirmed
        self.enumerated = False     # every in-budget item of the category has been listed
        self.line_step_limit = None  # step budget of the current line in a multi-line order
        self.line_ledgers = None     # per-line ledgers in a multi-line order
        self.unorderable = set()    # ids the cart tool or the world refused
        roles = self.adapter.roles   # the view that carries properties (one tool may serve both)
        if "spec" in roles and "detail" in roles and roles["spec"].tool is roles["detail"].tool:
            self.spec_view = "detail"
        else:
            self.spec_view = "spec" if "spec" in roles else ("detail" if "detail" in roles else None)
        self.ledger = self.R.extract(self.query, self.opening, self.llm)
        self.note("bound roles", roles=self.adapter.describe(), ledger=self.ledger.summary())

    # ---- budgets ---------------------------------------------------------------------
    @property
    def step_cap(self) -> int:
        stated = self.ledger.step_budget
        cap = self.env.max_steps
        return min(cap, stated) if isinstance(stated, int) and stated >= 5 else cap

    @property
    def reserve(self) -> int:
        """Steps kept for the final cart fix, re-read and order: CONFIG["reserve_steps"] at 46 steps,
        scaled down for short worlds (at least 3)."""
        return max(3, min(CONFIG["reserve_steps"], round(self.step_cap * CONFIG["reserve_steps"] / 46)))

    @property
    def refusal_reserve(self) -> int:
        return max(3, min(CONFIG["refusal_reserve"], round(self.step_cap * CONFIG["refusal_reserve"] / 46)))

    def steps_left(self) -> int:
        return self.step_cap - self.env.steps - self.line_debt

    def time_left(self) -> float:
        return self.env.time_left()

    def messages_left(self) -> int:
        if self.msg_exhausted or "message" not in self.adapter.roles:
            return 0
        budget = self.ledger.msg_budget if self.ledger.msg_budget is not None else (self.status_cap or 2)
        return max(0, budget - self.sent_messages)

    def read_status(self):
        """G4: with no message cap in the request, read a status tool once before the first question. Its counts
        give the cap when the used count equals the messages sent while other steps have been taken."""
        if (self.status_read or self.ledger.msg_budget is not None or "status" not in self.adapter.roles
                or "message" not in self.adapter.roles or not self.can_spend(2)):
            return
        self.status_read = True
        obs = self.act([self.adapter.status()])
        counts = status_counts(obs[0].raw) if obs else None
        if counts and counts[0] == self.sent_messages < self.env.steps - 1 and 1 <= counts[0] + counts[1] <= 12:
            self.status_cap = self.ledger.msg_budget = counts[0] + counts[1]
            self.note("message cap from status", cap=self.status_cap)

    def can_spend(self, n=1, keep=None) -> bool:
        keep = self.reserve + self.refusal_reserve if keep is None else keep
        if self.line_step_limit is not None and self.env.steps + n > self.line_step_limit:
            return False   # this line of a multi-line order has used its share of the steps
        return (not self.env.finished and self.steps_left() - n >= keep
                and self.time_left() > CONFIG["finish_margin_s"])

    def budget_world(self):
        """The budget in the world's money unit (cents unless the world says dollars)."""
        if self.ledger.budget is None:
            return None
        return self.ledger.budget / 100.0 if self.adapter.money_unit == "dollars" else self.ledger.budget

    def note(self, text, **extra):
        entry = {"step": self.env.steps, "note": text}
        entry.update(extra)
        self.trace.append(entry)

    # ---- acting ----------------------------------------------------------------------
    def act(self, calls) -> list:
        """Send 1-16 calls as one envelope; normalise and ingest each observation."""
        calls = [c for c in calls if c]
        failed = self.__dict__.setdefault("_failed_calls", set())   # R5-S5: a call that failed is never resent unchanged
        calls = [c for c in calls if json.dumps([c[0], c[1]], sort_keys=True, default=str) not in failed]
        if not calls:
            return []
        if self.env.past_guard():   # A-R4: past the margin, only a terminal call goes out
            calls = [c for c in calls if self.adapter.role_of(c[0], c[1]) in ("order", "decline", "finish")]
            if not calls:
                raise _TimeUp()
        raws = self.env.send(calls)
        if raws is None:
            self.note("environment call failed", error=self.env.last_error)
            return []
        out = []
        first = self.env.steps - len(raws) + 1
        for i, ((name, args), raw) in enumerate(zip(calls, raws)):
            role = self.adapter.role_of(name, args)
            obs = self.adapter.normalise(raw, role)
            if obs.kind == "error" and role not in ("order", "decline", "cart_add", "cart_view"):
                failed.add(json.dumps([name, args], sort_keys=True, default=str))   # R5-S5 (finishing and cart calls may be retried)
            self._ingest(first + i, role, obs, args)
            self.trace.append({"step": first + i, "tool": name, "role": role, "args": args,
                               "obs": clip(json.dumps(obs.summary(), ensure_ascii=False), 600)})
            out.append(obs)
        refused = [obs for (name, args), obs in zip(calls, out) if obs.refusal and self.adapter.role_of(name, args) == "order"]
        if refused and not self.env.finished:   # zip: a batch cut by the step cap or a finish returns fewer observations
            try:
                self.named_step(refused[-1])   # P1
            except Exception as exc:  # noqa: BLE001 - P1 is an addition: a failure in it leaves the head's path
                self.note("named step: error", error=f"{type(exc).__name__}: {exc}")
        return out

    def named_step(self, refused):
        """P1: an order refused with text naming a declared tool by its exact name (never a cart, order, decline
        or message tool): call it once for this basket; the checkout loop orders again. The same refusal after the step
        with the basket unchanged sets named_stuck: re-ordering cannot pass."""
        text = " ".join(str(s) for s in refused.refusal)
        def says(name, line):
            return re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(name), line) is not None
        named = [n for n in self.adapter.step_tools() if says(n, text)]
        if not named:
            return None
        basket = sum(1 for e in self.trace if e.get("role") == "cart_add")
        done = self.__dict__.setdefault("named_done", {})   # tool -> basket (cart calls so far) when it was called
        fresh = [n for n in named if done.get(n) != basket]
        if not fresh:
            if text == self.__dict__.get("named_text"):
                self.named_stuck = True
                self.note("named step: same refusal after the step", tools=named)
            return None
        calls = self.__dict__.setdefault("named_calls", [])
        if len(calls) >= 2 or self.steps_left() < 2:
            return None
        call = self.adapter.named_step(fresh[0], self.shop.cart, text)
        if call is None:
            self.note("named step: no value for a required argument", tool=fresh[0])
            return None
        done[fresh[0]] = basket
        self.named_text = text
        calls.append(fresh[0])
        got = self.act([call])
        self.note("named step", tool=fresh[0], ok=got[0].ok if got else None)

    def _ingest(self, step, role, obs, args=None):
        if obs.kind in ("listing", "product"):
            view = "listing" if obs.kind == "listing" else ("spec" if role == "spec" else "detail")
            for product in obs.products:
                self.shop.merge(product, view, step)
                if obs.kind == "product":
                    redirects = parse_redirects(product.get("text"))
                    if redirects:
                        self.shop.products[product["id"]]["redirects"] = redirects
        if obs.cart is not None:
            if role == "cart_add" and not obs.cart.get("partial"):
                self.adapter.learn_cart(args, dict(self.shop.cart), obs.cart)   # G3
            self.shop.sync_cart(obs.cart, step)
        if role == "cart_add":
            self.shop.cart_version += 1
            if obs.kind == "error" or obs.ok is False:
                self.shop.cart_synced = False
            elif obs.cart is None or obs.cart.get("partial"):
                effect = self.adapter.cart_effect(args, lambda pid: self.shop.cart.get(pid, 0))
                if isinstance(effect, dict) and obs.cart is None:
                    self.shop.cart = dict(effect)
                elif effect and obs.cart is None:
                    self.shop.cart[effect[0]] = effect[1]
                self.shop.cart_synced = False  # predicted, not yet seen
        tags = set()
        if role == "message":
            if obs.error:
                if "budget" in obs.error.lower():
                    self.msg_exhausted = True
            else:
                self.sent_messages += 1
                if obs.text:
                    tags |= self.R.absorb(self.ledger, [obs.text], "reply", self.shop, self.llm) or set()
        targets = [self.ledger] + [l for l in (self.line_ledgers or []) if l is not self.ledger]
        if obs.lines:
            for target in targets:
                tags |= self.R.absorb(target, obs.lines, "change", self.shop, self.llm) or set()
        if obs.refusal:
            self.shop.cart_version += 1   # re-read the cart item before the next checkout
            fresh = [s for s in obs.refusal if s not in obs.lines]
            for target in targets:
                tags |= self.R.absorb(target, fresh, "refusal", self.shop, self.llm) or set()
            tags.add("refused")
        carted = [pid for pid, units in self.shop.cart.items() if units > 0]
        if "stock" in tags and len(carted) == 1 and carted[0] in self.shop.products:
            self.shop.products[carted[0]]["in_stock"] = False   # the line is about the item we proposed
            self.shop.fresh[carted[0]] = self.env.steps
            tags.discard("price")
        # with several cart lines the re-read before the next checkout finds which one sold out
        if tags:
            self.events |= tags
            if tags & {"price", "stock", "refused"}:  # the lines are about the item we proposed
                self.shop.stale |= {pid for pid, units in self.shop.cart.items() if units > 0}
            self.note("events", tags=sorted(tags), ledger=self.ledger.summary())
            if "refused" in tags and "drop" in tags and not tags & {"stock", "price", "budget", "need", "quantity", "item"}:
                self.relax_hold = True   # RX: the refusal only relaxed a need: the cart still fits, order it again

    # ---- selection -------------------------------------------------------------------
    def viable(self) -> list:
        """Products we could order now, best first. judge() runs without the LLM here."""
        cap, qty = self.budget_world(), self.ledger.quantity or 1
        before = len(self.ledger.needs)
        self.R.late_needs(self.ledger, self.shop)
        if len(self.ledger.needs) != before:
            self.note("request needs bound to fact keys", needs=[n.as_dict() for n in self.ledger.needs[before:]])
        out = []
        for p in self.shop.products.values():
            if p["id"] in self.rejected or p.get("in_stock") is False or p.get("price") is None:
                continue
            if cap is not None and p["price"] * qty > cap + 1e-9:
                continue
            if self._r9_other_line(p):
                continue   # R9b F1 (F-046 a): another line's product
            verdict = self.R.judge(self.ledger, p, None)
            if verdict.ok is False:
                continue
            p["_verdict"] = None if verdict.ok is True and self.exact_key_gap(p) else verdict.ok   # X1
            out.append(p)
        if self.category:
            inside = [p for p in out if (p.get("category") or "").lower() == self.category.lower()]
            if inside:
                out = inside + [p for p in out if not p.get("category")]
            elif self.ledger.notes.get("decline_stated") and not self.line_ledgers and self.aisle_named():
                # Q6-D1: the request names this aisle and none of its listings fits: another aisle's goods are substitutes
                out = [p for p in out if not p.get("category") or self.aisle_words(p["category"]) & self.aisle_words(self.ledger.product)]
        return self.R.rank(self.ledger, out, self.llm)

    @staticmethod
    def aisle_words(text) -> set:
        return set(toks(text)) - _FUNCTION_WORDS

    def aisle_named(self) -> bool:
        """Q6-D1: every content word of the chosen aisle's name is in the request's product phrase (or its category)."""
        name = self.aisle_words(self.category)
        return bool(name) and name <= self.aisle_words(" ".join(x for x in (self.ledger.product, self.ledger.category) if x))

    def exact_key_gap(self, p):
        """X1: the listing meets a need only through a longer key that holds the attribute's name ("<word> <key>")
        while most opened listings of its category show the key itself. The grader reads the key itself, so
        here the need is unconfirmed, not met. Its own text or the key itself still confirm."""
        props = p.get("props") or {}
        cat = (p.get("category") or "").lower()
        peers = [q for q in self.shop.products.values() if q.get("props") and (q.get("category") or "").lower() == cat]
        if len(peers) < 3:
            return False
        noun = set(toks(self.ledger.product or "")) | set(toks(self.category or "")) | set(toks(cat))
        for n in self.ledger.needs:
            if not n.hard or not n.prop or n.op not in ("eq", "in"):
                continue
            k0 = set(toks(n.prop))

            def own(k):   # the key itself, or the key with the product's own noun ("<noun> <key>")
                return bool(k0) and (set(toks(_q_key(k))) == k0 or set(toks(_q_key(k))) - noun == k0)
            if any(own(k) for k in props):
                continue
            if sum(any(own(k) for k in q["props"]) for q in peers) * 2 < len(peers):
                continue
            bare = dict(p, props={k: v for k, v in props.items() if not (key_score(n.prop, k) > 0 and not own(k))})
            if need_status(n, p) is True and need_status(n, bare) is not True:
                return True
        return False

    def drop_unlisted_model_needs(self):
        """A need or avoid the model read from the request stays only when an inspected listing shows its
        value (its words in order, inside one listing value); otherwise it was not about this store's goods."""
        model = self.ledger.notes.pop("model_needs", None)
        if not model:
            return
        listed = [f" {' '.join(toks(v))} " for values in self.shop._prop_values().values() for v in values]

        def shown(value):
            vt = " ".join(toks(value))
            return bool(vt) and any(f" {vt} " in v for v in listed)

        for attr, value, op in model:
            for n in [n for n in self.ledger.needs if n.prop == attr and n.op == op and n.value == value]:
                kept = [v for v in value if shown(v)] if isinstance(value, list) else (value if shown(value) else None)
                if kept:
                    n.value = kept
                else:
                    self.ledger.needs.remove(n)
                    self.note("model need not listed, dropped", need=[attr, value, op])

    def harvest_request_values(self):
        """Catalogue values of this category named in the request's product sentences ("a <value>
        <product>") become needs, once the listings show which values exist."""
        self.drop_unlisted_model_needs()
        if not self.ledger.product:
            return
        ptoks = set(toks(self.ledger.product))
        sentences = [x for x in re.split(r"(?<=[.!?])\s+", self.ledger.query) if ptoks & set(toks(x))]
        avoid = " ".join(str(n.value) for n in self.ledger.needs if n.op == "avoid")
        have = {(n.prop or "", " ".join(toks(str(n.value)))) for n in self.ledger.needs}
        stated = " ".join(" ".join(toks(str(n.value))) for n in self.ledger.needs)
        for name, values in self.shop._prop_values().items():
            for value in values:
                vt = " ".join(toks(value))
                if (not vt or len(vt) < 3 or set(vt.split()) <= _FUNCTION_WORDS or vt in ptoks
                        or f" {vt} " in f" {' '.join(toks(avoid))} " or f" {vt} " in f" {stated} "):
                    continue
                if any(f" {vt} " in f" {' '.join(toks(x))} " and not re.search(
                        r"\b(?:(?:no|not|nothing|or) (?:\w+ )?)" + re.escape(vt) + r"\b|\b" + re.escape(vt) + r" or\b", " ".join(stem(w) for w in re.findall(r"[a-z0-9]+", x.lower())))
                       for x in sentences) and (name, vt) not in have:   # AN4: not a value after a negation or next to the word or
                    self.ledger.needs.append(Need(name, value, "request", True, "eq"))
                    have.add((name, vt))

    def capture_request_values(self):
        """Needs stated beside a key the inspected listings use (capture_key_values)."""
        table = self.shop._prop_values()
        if not table:
            return
        known = set()   # words of the values already stated: a capture that shares one restates that need
        for n in self.ledger.needs:
            known |= set(toks(" ".join(n.value) if isinstance(n.value, list) else str(n.value or "")))
        for key, value in capture_key_values(self.ledger.query, self.ledger.product, table, self.category or self.infer_category()):
            if not set(toks(value)) & known:
                self.ledger.needs.append(Need(key, value, "request", True, "eq"))
                known |= set(toks(value))
        for n in self.ledger.needs:   # an avoid "<key> <value>" ("anything with <key> <value>"): avoid the value
            if n.op == "avoid" and isinstance(n.value, str):
                vw = n.value.lower().split()
                for key in sorted(table, key=lambda k: -len(k.split())):
                    kw = key.split()
                    if len(vw) > len(kw) and vw[:len(kw)] == kw:
                        n.value = " ".join(vw[len(kw):])
                        n.prop, n.op = key, "ne"
                        break
        self.note("request values", needs=[n.as_dict() for n in self.ledger.needs])

    def reread_replies(self):
        """RV2: replies that gave no need are read again once more listings are open: a property they name that a
        listing now carries, with a value q_parse_reveal finds, becomes a shopper need."""
        unread = self.ledger.notes.get("q_unread") or []
        if not unread:
            return
        noun = " ".join(x for x in (self.ledger.product, self.ledger.category, self.ledger.notes.get("q_category")) if x)
        keys = {_q_key(k) for p in self.shop.products.values() for k in (p.get("props") or {})}
        have = {_q_key(n.prop) for n in self.ledger.needs if n.prop}
        for line in list(unread):
            said = set(toks(line))
            named = sorted((k for k in keys if set(toks(k)) and set(toks(k)) <= said and k not in have), key=lambda k: -len(k))
            for key in named:
                parsed = q_parse_reveal(line, key, self.shop.values_of(key), (), noun)
                if CONFIG.get("jv_support") and (not parsed or (parsed[0] == "eq" and jv_seen(parsed[1], self.shop, key) is None)):
                    alt = jv_candidate(line, key, noun, self.shop, self.ledger.category or self.ledger.notes.get("q_category"), want_tier=True)
                    if alt and (alt[0] == 0 or not parsed or jv_junk(parsed[1], key, noun)):   # JV: the key's own value, or over junk
                        parsed = ("eq", alt[1])
                    elif parsed and parsed[0] == "eq" and jv_junk(parsed[1], key, noun):
                        parsed = None
                if parsed:
                    self.R._set_need(self.ledger, Need(key, parsed[1], "shopper", True, parsed[0]))
                    unread.remove(line)
                    self.note("reply read again", key=key, value=parsed[1])
                    break

    def title_look(self, k=2):
        """T2: after the first look, open up to k listings whose titles no opened listing has: copies of one title are
        variants of one item, so the question pool also sees the keys and values of other items."""
        role = self.spec_view
        if role is None:
            return
        seen = {" ".join(toks(p.get("title") or "")) for p in self.shop.products.values() if self.shop.has_view(p["id"], role)}
        if len(seen) > 1:   # T3: the first look saw more than one item already
            return
        todo = []
        for p in sorted(self.viable(), key=lambda p: (not self.hinted(p), -self.covered(p),
                                                       p.get("price") if p.get("price") is not None else float("inf"))):
            t = " ".join(toks(p.get("title") or ""))
            if t not in seen and not self.shop.has_view(p["id"], role):
                seen.add(t)
                todo.append(p)
        k = min(len(todo), k)
        if k > 0 and self.can_spend(k, keep=self.reserve + min(self.messages_left(), 3) + 4):
            self.act([self.adapter.detail(p["id"], role) for p in todo[:k]])

    def bind_need_props(self):
        """Give an attribute-less need ("party") the property whose catalogue values hold it
        ("occasion"), read from the inspected listings; a later relax of that property can
        then drop it."""
        table = self.shop._prop_values()
        for need in self.ledger.needs:
            if need.prop or need.op not in ("eq", "in") or not need.value:
                continue
            vt = set(toks(" ".join(need.value) if isinstance(need.value, list) else str(need.value)))
            hits = [name for name, values in table.items()
                    if vt and any(vt <= set(toks(v)) for v in values)]
            if len(hits) == 1 or (hits and len({h.split()[-1] for h in hits}) == 1):
                need.prop = min(hits, key=len)

    def infer_category(self, strict=False):
        """The catalogue category the shopper means: the one whose name shares most words with
        the product phrase, ties broken by how often it came up (strict: only one that shares a word)."""
        counts = {}
        for p in self.shop.products.values():
            if p.get("category"):
                counts[p["category"]] = counts.get(p["category"], 0) + 1
        if not counts:
            return None
        target = (set(toks(self.ledger.product or "")) | set(toks(self.ledger.category or ""))
                  | set(toks(" ".join(self.ledger.terms))))
        scored = sorted(((len(set(toks(c)) & target), n, c) for c, n in counts.items()), reverse=True)
        overlap, n, best = scored[0]
        if overlap > 0 or (not strict and n >= 0.6 * sum(counts.values())):
            return best
        return None

    def settle_aisle(self, rounds=2):
        """Search rows that show no category: after the first look, the opened listings' category that names
        the product's head noun is the aisle. When none does and a compare call opens several listings in
        one step, the next cheapest are opened (at most `rounds` more steps)."""
        if self.category or not self.spec_view:
            return
        heads = {toks(alt)[-1] for alt in re.split(r"\s+(?:or|and)\s+|[,/]", self.ledger.product or self.ledger.category or "")
                 if toks(alt)}
        for i in range(rounds + 1):
            labels = {}
            for p in self.shop.products.values():
                if p.get("category") and self.shop.has_view(p["id"], self.spec_view):
                    labels[p["category"]] = labels.get(p["category"], 0) + 1
            fit = sorted(((len(heads & set(toks(c))), n, c) for c, n in labels.items()), reverse=True)
            if fit and fit[0][0] > 0:
                self.category = fit[0][2]
                self.note("aisle from opened listings", category=self.category)
                return
            if not labels or i == rounds or self.compare_size < 2:
                return
            todo = sorted((p for p in self.viable() if not self.shop.has_view(p["id"], self.spec_view)),
                          key=lambda p: p.get("price") if p.get("price") is not None else float("inf"))
            if not todo or not self.can_spend(1, keep=self.reserve + self.refusal_reserve + min(self.messages_left(), 3) + 2):
                return
            self.open_listings([p["id"] for p in todo[:self.compare_size]])

    def pick(self):
        if self.llm is not None:
            self.R.compare_values(self.ledger, self.shop, self.llm)    # C0045: at most 2 judge calls per episode
        ranked = self.viable()
        if ranked:
            return ranked[0]
        forced = self.shop.products.get(self.forced_pick) if self.forced_pick else None
        if forced and forced.get("in_stock") is not False and forced["id"] not in self.rejected:
            return forced
        return None

    def need_values(self) -> list:
        """Token sets of the ledger's stated and revealed values (eq / in needs)."""
        out = []
        for n in self.ledger.needs:
            if n.op in ("eq", "in") and n.value:
                for v in (n.value if isinstance(n.value, list) else [n.value]):
                    if isinstance(v, str) and toks(v):
                        out.append(set(toks(v)))
        return out

    def hinted(self, p) -> bool:
        """The listing text names every stated need value; a query naming such a value also counts."""
        hits = p.get("_vhit")
        if hits and any(v <= hits for v in self.need_values()):
            return True
        values = [n.value for n in self.ledger.needs if n.op == "eq" and isinstance(n.value, str)]
        values += [v for v in tentative_values(self.ledger) if v not in values]
        blob = set(toks(" ".join(str(p.get(k) or "") for k in ("title", "text"))))
        return bool(values) and all(set(toks(v)) <= blob for v in values)

    def covered(self, p) -> int:
        """How many stated or revealed values the listing text or the queries that listed it name."""
        blob = set(toks(" ".join(str(p.get(k) or "") for k in ("title", "text")))) | (p.get("_vhit") or set())
        return sum(1 for v in self.need_values() if v <= blob)

    def noted(self, role, phrase, obs):
        """Record a query page; the hits of a query that names a need value remember its words."""
        self.shop.note_query((role, phrase), obs)
        words_ = set(toks(phrase))
        if obs is not None and any(v <= words_ for v in self.need_values()):
            for p in obs.products or []:
                if p.get("id") in self.shop.products:
                    self.shop.products[p["id"]]["_vhit"] = self.shop.products[p["id"]].get("_vhit", set()) | words_

    def value_requery(self):
        """One "<value> <product>" query per value the shopper revealed; its hits go first."""
        if "search" not in self.adapter.roles:
            return
        product = (self.ledger.product or " ".join(self.ledger.terms)).split(" or ")[0]
        for n in list(self.ledger.needs):
            if n.source != "shopper" or n.op not in ("eq", "in") or not n.value:
                continue
            value = n.value[0] if isinstance(n.value, list) else n.value
            stated = [str(m.value) for m in self.ledger.needs if m.source in ("request", "reference") and m.op == "eq"   # VR3
                      and isinstance(m.value, str) and m.value and not set(toks(m.value)) <= set(toks(str(value)))]
            phrase = " ".join(f"{value} {' '.join(stated[:2])} {product}".split())
            if ("search", phrase) in self.shop.queries or not self.can_spend(1, keep=self.reserve + 2):
                continue
            call = self._query_call("search", phrase, None)
            if call is None:
                continue
            obs = self.act([call])
            self.noted("search", phrase, obs[0] if obs else None)

    def refine_cheapest(self, tries=4, spend=6):
        """Without a price filter a need query whose matches outnumber what its pages show leaves
        cheaper fits unseen. Narrow it with each value the cheapest fitting listing carries
        (fact values are searchable), reading every page of a narrowed query the world shows whole."""
        b = self.adapter.roles.get("browse")
        if (self.ledger.prefer != "cheapest" or not re.search(r"\b(?:cheapest|lowest|(?<!at )least)\b", self.ledger.query.lower())
                or "search" not in self.adapter.roles or len(self.ledger.lines) >= 2
                or (b and b.tool.param("cap")) or not self.need_values()):
            return
        cut = sorted((phrase for (role, phrase), st in self.shop.queries.items()
                      if role == "search" and st.get("total") and st["total"] > st.get("shown", 0)
                      and any(v <= set(toks(phrase)) for v in self.need_values())), key=len)
        pool = [p for p in self.viable() if p.get("props") and p.get("_verdict") is not False
                and (p.get("_verdict") is True or self.hinted(p))]
        if not cut or not pool:
            return
        base, best = cut[0], min(pool, key=lambda p: p["price"])
        have = {w.rstrip("s") for w in toks(base)}
        values = []
        for raw in (best.get("props") or {}).values():
            for piece in re.split(r"[,;/]", str(raw).lower()):
                ws = toks(piece)
                if (1 <= len(ws) <= 2 and all(w.isalpha() and len(w) >= 3 for w in ws)
                        and not {w.rstrip("s") for w in ws} & have and " ".join(ws) not in values):
                    values.append(" ".join(ws))
        shared = lambda v: sum(1 for p in pool if p is not best and v in " ".join(
            str(x).lower() for x in (p.get("props") or {}).values()))
        values.sort(key=shared)   # values the other fits lack narrow the query most
        keep, used = self.reserve + self.refusal_reserve + min(self.messages_left(), 3) + 2, 0
        for value in values[:tries]:
            phrase = f"{base} {value}"
            if ("search", phrase) in self.shop.queries or used >= spend or not self.can_spend(1, keep=keep):
                continue
            call = self._query_call("search", phrase, None)
            if call is None:
                return
            obs = self.act([call])
            used += 1
            self.noted("search", phrase, obs[0] if obs else None)
            st = self.shop.queries.get(("search", phrase)) or {}
            while (not st.get("done") and st.get("next") is not None and (st.get("total") or 99) <= 21
                   and used < spend and self.can_spend(1, keep=keep)):
                more = self.act([self._query_call("search", phrase, st["next"])])
                used += 1
                self.noted("search", phrase, more[0] if more else None)
        if used:
            self.note("cheapest refined", base=base, values=values[:tries], steps=used)

    def explore_more(self) -> bool:
        """More pages of the need-specific queries, and the need values on their own. Returns
        True when anything was fetched."""
        role = "search" if "search" in self.adapter.roles else None
        if role is None:
            return False
        product = (self.ledger.product or " ".join(self.ledger.terms)).split(" or ")[0]
        head = product.split()[-1] if product else ""
        values = [n.value for n in self.ledger.needs if n.op == "eq" and isinstance(n.value, str)]
        phrases = [f"{v} {product}".strip() for v in values] + [f"{v} {head}".strip() for v in values] + values
        fetched = False
        for phrase in dict.fromkeys(p for p in phrases if p):
            if not self.can_spend(1, keep=self.reserve + 2):
                break
            state = self.shop.queries.get((role, phrase))
            if state is None:
                call = self._query_call(role, phrase, None)
            elif not state["done"] and state["next"] is not None:
                call = self._query_call(role, phrase, state["next"])
            else:
                continue
            if call is None:
                continue
            obs = self.act([call])
            self.noted(role, phrase, obs[0] if obs else None)
            fetched = True
            if obs and obs[0].products and len(obs[0].products) >= 3:
                break   # one useful page at a time; the inspection decides what next
        return fetched

    @staticmethod
    def _r9_own(led, p, category=None) -> bool:
        """R9b F1: the listing is this line's own item: its head noun in the title, or the line's own category."""
        head = _r9_stem((words(led.product or "") or [""])[-1])
        if head and head in {_r9_stem(w) for w in words(p.get("title") or "")}:
            return True
        cat = led.notes.get("category") or category
        return bool(cat) and str(p.get("category") or "").lower() == str(cat).lower()

    def _r9_other_line(self, p) -> bool:
        """R9b F1: in a multi-line order, the listing is ANOTHER line's own item and not the current line's.
        A listing that belongs to no line (a title without any line's noun) stays a candidate."""
        lines = getattr(self, "line_ledgers", None) or []
        if len(lines) < 2 or not any(self.ledger is led for led in lines):
            return False
        if self._r9_own(self.ledger, p, self.category):
            return False
        return any(self._r9_own(led, p) for led in lines if led is not self.ledger)

    def _r9_cheapest_own(self, led):
        """R9b F2: the line's cheapest own, in-stock candidate that its needs do not rule out (or None)."""
        saved = self.ledger
        self.ledger = led
        try:
            pool = [p for p in self.shop.products.values()
                    if p.get("in_stock") is not False and p.get("price") is not None and p["id"] not in self.unorderable
                    and p["id"] not in self.rejected and not self._r9_other_line(p)
                    and self.R.judge(led, p, None).ok is not False]
        finally:
            self.ledger = saved
        return min(pool, key=lambda p: p["price"]) if pool else None

    def closest_candidate(self):
        """The in-stock, affordable product (in the chosen category) that breaks the fewest needs:
        what the decline guard orders when no product is confirmed."""
        cap, qty = self.budget_world(), self.ledger.quantity or 1
        pool = []
        for p in self.shop.products.values():
            if p.get("in_stock") is False or p.get("price") is None or p["id"] in self.unorderable:
                continue
            if cap is not None and p["price"] * qty > cap + 1e-9:
                continue
            if self.category and p.get("category") and p["category"].lower() != self.category.lower():
                continue
            if self._r9_other_line(p):
                continue   # R9b F1 (F-046 a): never another line's product
            broken = [n for n in self.ledger.needs if n.hard and n.op != "ref" and need_status(n, p) is False]
            p["_violations"] = len(broken)
            p["_breaks_revealed"] = any(n.source in ("shopper", "reference") for n in broken)
            pool.append(p)
        pool = [p for p in pool if p["id"] not in self.rejected] or pool
        if not pool:
            return None
        best = min(pool, key=lambda p: (p["_breaks_revealed"], p["_violations"], p["price"]))
        certified = (self.enumerated and best["_violations"] > 0
                     and all(self.shop.has_view(p["id"], self.spec_view) for p in pool))
        if certified or best.get("_breaks_revealed"):
            return None   # ordering it cannot pass: it breaks a need the shopper revealed (or all were checked)
        if best["_violations"] and self.ledger.notes.get("decline_stated"):
            self.note("decline audit", id=best["id"], why="breaks a stated need")
            return None   # R5: it breaks a stated need and the request allows a decline: a guess cannot pass
        return best

    # ---- phases ----------------------------------------------------------------------
    def _query_call(self, role, phrase, token):
        if role == "search":
            return self.adapter.search(phrase, token)
        return self.adapter.browse(phrase, None, token)

    def search_phase(self):
        role = "search" if "search" in self.adapter.roles else ("browse" if "browse" in self.adapter.roles else None)
        if role is None:
            return
        phrases = [p for p in self.R.search_terms(self.ledger, self.shop, self.llm) if p][: CONFIG["max_queries"]]
        budget, start = max(2, int(self.step_cap * 0.25)), self.env.steps
        fresh = [p for p in phrases if (role, p) not in self.shop.queries][:2]
        if fresh and self.can_spend(len(fresh)):
            calls = [self._query_call(role, p, None) for p in fresh]
            for phrase, obs in zip(fresh, self.act(calls)):
                self.noted(role, phrase, obs)
        if self.verify_product(role):   # S3-P1: the store did not show the parsed product
            phrases = [p for p in self.R.search_terms(self.ledger, self.shop, self.llm) if p][: CONFIG["max_queries"]]
        self.category = self.category or self.infer_category(strict=True)
        browse_ok = bool((self.category or self.infer_category()) and "browse" in self.adapter.roles
                         and self.adapter.roles["browse"].tool.param("category"))
        while self.env.steps - start < budget and self.can_spend(1):
            pending = None
            for phrase in phrases:
                state = self.shop.queries.get((role, phrase))
                if state is None:
                    pending = (phrase, None)
                elif not state["done"] and state["next"] is not None and (
                        not browse_ok or (state["pages"] < 3 and any(v <= set(toks(phrase)) for v in self.need_values()))):
                    pending = (phrase, state["next"])
                if pending:
                    break
            if pending is None:
                break
            phrase, token = pending
            call = self._query_call(role, phrase, token)
            if call is None:
                self.shop.queries.setdefault((role, phrase), {"pages": 0, "next": None})["done"] = True
                continue
            obs = self.act([call])
            self.noted(role, phrase, obs[0] if obs else None)
            self.category = self.category or self.infer_category(strict=True)
        self.category = self.category or self.infer_category()

    def verify_product(self, role) -> bool:
        """S3-P1: keep a single-line product the store shows (3+ listings seen with its words in their
        titles); otherwise try the request's other noun phrases, at most 3 head-word searches, and take the
        first whose results carry the head word in 3+ titles. True when the product changed."""
        led = self.ledger
        if led.notes.get("product_checked") or len(led.lines or []) >= 2:
            return False
        if led.category and led.product == led.category:
            return False   # the request named the store's category: listing titles need not repeat it
        led.notes["product_checked"] = True

        def titled(word, products):
            w = set(toks(word))
            forms = w | {x + "s" for x in w} | {x[:-1] for x in w if x.endswith("s")}
            return sum(1 for p in products if forms & set(toks(p.get("title") or "")))
        current = [w for w in words(led.product or "") if w not in _STOP and len(w) > 2]
        shown = bool(current) and max(titled(w, list(self.shop.products.values())) for w in current) >= 3
        weak = bool(led.notes.get("product_weak")) or not current   # no product at all: any grounded candidate helps
        if not weak:
            return False   # a product a want-clause names stands, shown or not (titles can be absent from rows)
        tries = 0
        for phrase, head in _np_candidates(led.query):
            if head in current:
                if shown:
                    return False   # a weak product the store shows, ranked before every other candidate: kept
                continue
            if (role, head) in self.shop.queries:
                continue
            if tries >= 3 or not self.can_spend(1, keep=self.reserve + 3):
                break
            call = self._query_call(role, head, None)
            if call is None:
                break
            tries += 1
            obs = self.act([call])
            o = obs[0] if obs else None
            self.noted(role, head, o)
            if o is not None and titled(head, o.products or []) >= 3:
                self.note("product verified", was=led.product, now=phrase, head=head)
                led.product = phrase
                led.quantity = line_quantity(led.query, phrase)   # a count of the old phrase's noun is no quantity
                led.terms = [w for w in words(phrase) if w not in _STOP][:5]
                cat = re.search(r"\b" + re.escape(phrase) + r"\s+(?:in|from)\s+(?:the\s+)?(?P<c>[a-z][a-z &/'\-]{2,40}?)"
                                r"(?=\s*[,.;(]|\s+with\b|\s+and\b|$)", " ".join(str(led.query).split()), re.IGNORECASE)
                if cat and not led.category:   # S3-P1b: "<product> in <category>" names the department
                    led.category = cat.group("c").lower()
                return True
        if self.llm is not None and not shown:   # S3-P1: no candidate the store shows: the grounded model reading
            data = self.R._ask(self.llm, _SYS_EXTRACT, {"request": led.query})
            if data:
                was, led.notes["product_weak"] = led.product, True
                self.R._merge_llm_request(led, data)
                if led.product != was:
                    self.note("product from the model", was=was, now=led.product)
                    led.terms = [w for w in words(led.product) if w not in _STOP][:5]
                    return True
        return False

    def browse_phase(self, rounds=4):
        """Enumerate the cheapest items of the category: filter by category with a price cap,
        lowering the cap until every match fits in the pages the world will show."""
        b = self.adapter.roles.get("browse")
        if not b or not self.category or not b.tool.param("category"):
            return
        cap = self.budget_world()
        has_cap = bool(b.tool.param("cap"))
        page_name = b.tool.param("page")
        limit = (b.tool.props.get(page_name) or {}).get("maximum", CONFIG["max_pages"]) if page_name else 1
        max_pages = max(1, min(CONFIG["max_pages"], int(limit)))
        for _ in range(rounds):
            if not self.can_spend(1):
                return
            obs = self.act([self.adapter.browse(self.category, cap if has_cap else None)])
            o = obs[0] if obs else None
            if o is None or o.kind != "listing" or not o.products:
                return
            capacity = len(o.products) * max_pages
            if (cap is None and has_cap and o.total is not None and o.total > capacity and self.ledger.prefer == "cheapest"
                    and re.search(r"\b(?:cheapest|lowest)\b", self.ledger.query.lower())):
                prices = sorted(p["price"] for p in o.products if isinstance(p.get("price"), (int, float)))
                if prices:   # CHP: no budget, but the cheapest is asked: a cap from this page, narrowed below
                    cap = prices[max(0, int(len(prices) * capacity * 0.7 / o.total) - 1)]
                    continue
            if o.total is None or o.total <= capacity or not has_cap or cap is None:
                self.enumerated = o.total is not None and o.total <= capacity and cap == self.budget_world()
                pages, token = 1, o.next
                while token is not None and pages < max_pages and self.can_spend(1):
                    more = self.act([self.adapter.browse(self.category, cap if has_cap else None, token)])
                    if not more or more[0].kind != "listing":
                        break
                    pages, token = pages + 1, more[0].next
                return
            prices = sorted(p["price"] for p in o.products if isinstance(p.get("price"), (int, float)))
            share = capacity * 0.7 / o.total
            new_cap = prices[max(0, int(len(prices) * share) - 1)] if prices else cap * share
            cap = new_cap if new_cap < cap else cap * share

    def inspect_phase(self, limit=None):
        """Open candidates in price order (batched) until the best confirmed one is cheaper than
        everything still unopened, or the steps set aside for inspection run out."""
        role = self.spec_view
        if role is None:
            return
        opened = 0
        while True:
            room = (self.steps_left() - self.reserve - self.refusal_reserve
                    - min(self.messages_left(), 3) - (1 if self.ledger.notes.get("refs") else 0))
            ranked = self.viable()
            confirmed = [p["price"] for p in ranked if p.get("_verdict") is True and p.get("price") is not None]
            ceiling = min(confirmed) if confirmed and not limit else None   # a first look ignores it
            todo = sorted((p for p in ranked if not self.shop.has_view(p["id"], role)
                           and (ceiling is None or (p.get("price") or 0) < ceiling)),
                          key=lambda p: (not self.hinted(p), -self.covered(p),
                                         p.get("price") if p.get("price") is not None else float("inf")))
            k = min(len(todo), room, CONFIG["inspect_top_k"], (limit - opened) if limit else 99)
            if self.line_step_limit is not None:
                k = min(k, self.line_step_limit - self.env.steps)   # a line's last steps still inspect
            if k <= 0 or not self.can_spend(k):
                return
            self.act([self.adapter.detail(p["id"], role) for p in todo[:k]])
            opened += k

    def question_phase(self, limit=None):
        self.ledger.notes["q_category"] = self.category          # section 6c inputs
        self.ledger.notes["q_budget_world"] = self.budget_world()
        asked = 0
        self.read_status()   # G4
        while self.messages_left() > 0 and self.can_spend(1) and (limit is None or asked < limit):
            asked += 1
            question = self.R.next_question(self.ledger, self.shop, self.messages_left(), self.llm)
            if not question:
                break
            if not self.act([self.adapter.message(question)]):
                break
            if any(r.get("status") == "pending" for r in self.ledger.notes.get("refs", [])):
                self.resolve_references()

    # ---- references to other listings ------------------------------------------------
# >>>>>>>> C0017 claude-2: Run reference resolution (BEGIN) >>>>>>>>
    def resolve_references(self):
        """Turn each pending reference (a listing named by seller and qualifiers) into a hard need:
        resolve_chain follows named listings up to CONFIG["ref_max_hops"] hops and applies pairing
        rules (C0017); values other candidates carry for the attribute help read the last listing."""
        for item in self.ledger.notes.get("refs", []):
            if item.get("status") != "pending":
                continue
            if not self.can_spend(2, keep=CONFIG["reserve_steps"] + 1):
                return
            value, trail = resolve_chain(item["attr"], item["ref"], self.find_listing, CONFIG.get("ref_max_hops", 6),
                                         vocab=self.shop.values_of(item["attr"]))
            item["status"], item["trail"] = ("resolved" if value else "failed"), trail
            self.ledger.needs = [n for n in self.ledger.needs if not (n.op == "ref" and n.prop == item["attr"])]
            if value:   # H3: "anything except that page's value" is a not-equal need
                self.ledger.needs.append(Need(item["attr"], value, "reference", True,
                                              "ne" if (item.get("ref") or {}).get("neg") else "eq", scope=item.get("scope")))
            self.note("reference", attr=item["attr"], ref=item["ref"], value=value, hops=len(trail))

    def find_listing(self, ref, seen=()):
        """The listing a reference (or a redirect) names: that seller's listings (searched for when
        none is known yet), the likeliest few opened, then pick_listing on seller + every selector."""
        seller = str(ref.get("seller") or "").lower()
        if not seller:
            return None

        def mine():
            return [p for p in self.shop.products.values() if p["id"] not in seen
                    and str(p.get("seller") or p.get("brand") or "").lower() == seller]
        if not mine():
            words_ = " ".join(w for w in re.findall(r"[\w.\-]+", " ".join(str(x) for x in (ref.get("item"), ref.get("quals")) if x))
                              if w.lower() not in _SEL_STOP)
            for q in (f"{ref.get('seller')} {words_}"[:60].strip(), str(ref.get("seller"))):
                if not self.can_spend(1, keep=CONFIG["reserve_steps"]):
                    return None
                self.act([self.adapter.search(q)])
                if mine():
                    break
        elif max(selector_score(ref, p) for p in mine()) < 0.75:   # S2-F4: the seller's known listings miss the selectors
            words_ = " ".join(w for w in re.findall(r"[\w.\-]+", " ".join(str(x) for x in (ref.get("item"), ref.get("quals")) if x))
                              if w.lower() not in _SEL_STOP)
            q = f"{ref.get('seller')} {words_}"[:60].strip()
            done = self.ledger.notes.setdefault("ref_queries", [])
            if q not in done and self.can_spend(2, keep=CONFIG["reserve_steps"]):
                done.append(q)
                self.act([self.adapter.search(q)])
        found = sorted(mine(), key=lambda p: -selector_score(ref, p))
        if not found:
            return None
        role = self.spec_view or "detail"
        for p in found[:3]:   # H8: best first, one at a time, until an opened page fits the reference
            if any(self.shop.has_view(q["id"], role) and selector_score(ref, q) >= 0.75 for q in mine()):
                break
            if self.shop.has_view(p["id"], role):
                continue
            if not self.can_spend(1, keep=CONFIG["reserve_steps"]):
                break
            self.act([self.adapter.detail(p["id"], role)])
        pick = pick_listing(ref, mine(), exclude=seen)
        if pick is not None and not self.shop.has_view(pick["id"], role) and self.can_spend(1, keep=3):
            self.act([self.adapter.detail(pick["id"], role)])   # H5: every hop of a chain is opened
            pick = self.shop.products.get(pick["id"], pick)
        return pick

# <<<<<<<< C0017 claude-2: Run reference resolution (END) <<<<<<<<

    # ---- finishing -------------------------------------------------------------------
    def cart_lines(self) -> dict:
        return {"lines": [{"id": pid, "qty": q} for pid, q in self.shop.cart.items() if q > 0]}

    def order_gate(self):
        ok, problems = self.R.check_order(self.ledger, self.cart_lines(), self.shop, self.llm)
        problems = list(problems or [])
        if self.forced_pick and self.shop.cart.get(self.forced_pick, 0) > 0:
            problems = [p for p in problems if "violates" not in p]
            ok = not problems
        cap = self.budget_world()
        total = sum((self.shop.products.get(pid, {}).get("price") or 0) * q for pid, q in self.shop.cart.items())
        if cap is not None and total > cap + 1e-9:
            problems.append(f"cart total {total} is over the budget {cap}")
        return ok and not problems, problems

    def ensure_cart(self, pid) -> bool:
        """Make the cart exactly one line: `pid` x quantity. False when that is not possible."""
        want = self.ledger.quantity or 1
        if self.adapter.cart_mode == "vector":
            if {k: v for k, v in self.shop.cart.items() if v > 0} == {pid: want}:
                return True
            lines = {k: 0 for k, v in self.shop.cart.items() if v > 0}   # supplied lines are set; unlisted ones stay
            lines[pid] = want
            self.line_debt += max(0, len(lines) - 1)
            obs = self.act([self.adapter.cart_vector(lines)])
            return bool(obs) and obs[0].kind != "error" and obs[0].ok is not False
        plan = []
        for other, have in list(self.shop.cart.items()):
            if other != pid and have > 0:
                ops = self.adapter.cart_ops(other, have, 0)
                if ops:
                    plan.append((other, 0, ops))
                else:
                    self.note("cannot remove cart line", id=other)
        ops = self.adapter.cart_ops(pid, self.shop.cart.get(pid, 0), want)
        if ops is None:
            return False
        plan.append((pid, want, ops))
        if self.steps_left() < sum(len(o) for _, _, o in plan) + 1:
            plan = plan[-1:]
            if self.steps_left() < len(ops) + 1:
                return False
        for target, qty, calls in plan:
            for call in calls:
                obs = self.act([call])
                if not obs or obs[0].kind == "error" or obs[0].ok is False:
                    if target == pid:
                        return False
                    break
        return True

    def _r9_not_sticking(self, pid) -> bool:
        """A-R9 loop guard: the write answered, yet the basket does not hold the pick, for the second time."""
        if self.shop.cart.get(pid, 0) >= (self.ledger.quantity or 1):
            return False
        misses = self.__dict__.setdefault("r9_misses", Counter())
        misses[pid] += 1
        if misses[pid] >= 2:
            self.note("basket write does not stick", id=pid)
            return True
        return False

    def cart_cost(self, pid):
        """Cart calls needed to make the cart exactly `pid` x quantity (None if impossible)."""
        want = self.ledger.quantity or 1
        if self.adapter.cart_mode == "vector":
            current = {k: v for k, v in self.shop.cart.items() if v > 0}
            return 0 if current == {pid: want} else len(set(current) | {pid})
        target = self.adapter.cart_ops(pid, self.shop.cart.get(pid, 0), want)
        if target is None:
            return None
        others = [self.adapter.cart_ops(o, h, 0) for o, h in self.shop.cart.items() if o != pid and h > 0]
        return len(target) + sum(len(o or []) for o in others)

    def finish(self):
        """Order the best viable product, or decline. Leaves the episode open only when no
        finishing call is possible at all. Optional checks (spec, price and stock recheck, cart
        view) run only while a refused first checkout would still leave room to re-order."""
        roles = self.adapter.roles
        held_declines, explored, widened = 0, 0, 0
        for _ in range(CONFIG["max_order_attempts"] + 6):
            if self.env.finished or self.steps_left() <= 0:
                return
            if self.__dict__.pop("named_stuck", False):
                self.last_resort()   # P1: the named step is done and the same refusal came back
                return
            carted = [pid for pid, q in self.shop.cart.items() if q > 0]
            if self.__dict__.pop("relax_hold", False) and carted and "order" in roles:
                self.checkouts += 1   # RX: re-order the same cart after a relax refusal
                self.act([self.adapter.order()])
                continue
            if (self.steps_left() <= 4 and not self.checkouts and carted and "order" in roles):
                self.checkouts += 1   # a held first checkout still leaves one step to repeat it
                self.act([self.adapter.order()])
                continue
            if (self.steps_left() <= 2 and carted and "order" in roles
                    and all(pid in self.shop.products and self.R.judge(self.ledger, self.shop.products[pid], None).ok
                            is not False for pid in carted)):   # in stock and not breaking a known need
                self.checkouts += 1   # no room left to change the cart: order what is in it
                self.act([self.adapter.order()])
                continue
            if self.ledger.notes.get("q_unread"):
                self.reread_replies()   # RV4: listings finish() opened may now show a value an unread reply gives
            best = self.pick()
            if best is None and explored < 2 and self.steps_left() > self.reserve + 3 and self.time_left() > CONFIG["finish_margin_s"]:
                explored += 1   # nothing confirmed: look further before guessing or declining
                if self.explore_more():
                    self.inspect_phase()
                    continue
            if best is None and (self.forced_pick is None or self.forced_pick in self.rejected):
                fallback = self.closest_candidate()
                if fallback is not None:
                    # a decline was right in 1 of 90 race-4 tasks: order the closest candidate instead
                    self.forced_pick = fallback["id"]
                    self.note("decline guard", id=fallback["id"], violations=fallback.get("_violations"))
                    best = fallback
            if best is None:
                if self.R.should_decline(self.ledger, self.shop, True, self.llm):
                    if self.decline():
                        return
                    if held_declines < 2 and self.steps_left() > 0:
                        held_declines += 1   # a checkout event held the decline: re-plan, then finish again
                        continue
                break
            pid = best["id"]
            cost = self.cart_cost(pid)
            if cost is None:
                self.rejected.add(pid)
                continue
            keep = 0 if "refused" in self.events else self.refusal_reserve
            reread = 1 if "detail" in roles else 0
            spare = self.steps_left() - cost - 1 - reread - keep   # optional calls we can still afford
            hurry = self.time_left() < CONFIG["finish_margin_s"] / 2
            in_cart = self.shop.cart.get(pid, 0) > 0
            if not self.shop.is_fresh(pid) and not hurry and self.steps_left() >= cost + 2:
                # a change may have moved this price or stock: re-read it before buying
                if "detail" in roles:
                    self.act([self.adapter.detail(pid)])
                elif in_cart and "cart_view" in roles:
                    self.act([self.adapter.cart_view()])
                self.shop.stale.discard(pid)  # read once; never loop on a failed read
                continue
            if (spare >= 2 and not hurry and best.get("_verdict") is None and self.ledger.needs and self.spec_view
                    and not best.get("_hint_round")):
                hinted = [p for p in self.viable() if p["id"] != pid and self.hinted(p)
                          and not self.shop.has_view(p["id"], self.spec_view)][: min(4, spare - 1)]
                best["_hint_round"] = True
                if hinted:   # listings that name every need value may be confirmed: open them first
                    self.act([self.adapter.detail(p["id"], self.spec_view) for p in hinted])
                    continue
            if (spare >= 1 and not hurry and self.ledger.needs
                    and self.spec_view and not self.shop.has_view(pid, self.spec_view)):
                self.act([self.adapter.detail(pid, self.spec_view)])  # confirm the needs before buying
                if not self.shop.has_view(pid, self.spec_view):
                    self.rejected.add(pid)
                continue
            if (best.get("_verdict") is None and self.ledger.needs and widened < 2 and spare >= 2 and not hurry):
                batch = self.widen_for(exclude={pid}, limit=min(3, spare - 1))
                if batch:   # unknown is not met: open more candidates before carting an unconfirmed pick
                    widened += 1
                    self.act([self.adapter.detail(b, self.spec_view) for b in batch])
                    continue
            if best.get("_verdict") is None and self.ledger.notes.get("decline_stated"):
                unseen = self.unshown_needs(best) or self.absent_needs(best)   # Q6-D2
                if unseen and explored < 2 and self.steps_left() > self.reserve + 3 and self.explore_more():
                    explored += 1   # R5: look further first; the value may sit on a listing not yet seen
                    self.inspect_phase()
                    continue
                if unseen:   # R5: a stated value no affordable listing of the aisle shows: nothing here can pass
                    self.note("decline audit", id=pid, why="value shown nowhere", needs=unseen)
                    if self.decline():
                        return
                    self.rejected.add(pid)
                    continue
            if (self.llm is not None and CONFIG.get("model_veto") and not best.get("_checked")
                    and best.get("_verdict") is not True and hasattr(self.R, "final_check")):
                best["_checked"] = True
                ok = self.R.final_check(self.ledger, [best], self.llm).get(pid)
                self.note("model check", id=pid, ok=ok)
                if ok is False:
                    self.rejected.add(pid)
                    continue
            if not self.ensure_cart(pid) or self._r9_not_sticking(pid):
                self.rejected.add(pid)
                self.unorderable.add(pid)
                if pid == self.forced_pick:
                    self.forced_pick = None
                continue
            if not self.shop.is_fresh(pid):
                continue  # a change arrived with the cart reply: recheck first
            if "detail" in roles and self.steps_left() >= 2 and not hurry and self.shop.cart_reread != self.shop.cart_version:
                self.act([self.adapter.detail(pid)])  # a change can sell out the cart item silently
                self.shop.cart_reread = self.shop.cart_version
                if self.shop.products.get(pid, {}).get("in_stock") is False:
                    if pid == self.forced_pick:
                        self.forced_pick = None
                    continue
            elif "cart_view" in roles and not self.shop.cart_synced and spare >= 1 and not hurry:
                self.act([self.adapter.cart_view()])  # the cart shows the price we will pay
            again = self.pick()
            if again is not None and again["id"] != pid:
                held = self.shop.products.get(pid) or {}
                still_ok = (pid not in self.rejected and held.get("in_stock") is not False
                            and self.R.judge(self.ledger, held, None).ok is not False)
                swap = (self.cart_cost(again["id"]) or 1) + 1 + reread   # cart, order, re-read
                fits = self.steps_left() >= (self.cart_cost(again["id"]) or 1) + 1   # the cart change and the order
                if (not still_ok and fits) or self.steps_left() - keep >= swap:
                    continue  # fresher data changed the best choice (and a swap still fits)
                if not still_ok:
                    break     # the cart item breaks a need and no swap fits: never order it (last resort below)
            ok, problems = self.order_gate()
            if not ok and self.steps_left() > 1:
                self.note("order gate", problems=problems)
                if not any(("exactly one" in p or "quantity" in p) for p in problems):
                    self.rejected.add(pid)
                continue
            self.checkouts += 1
            obs = self.act([self.adapter.order()])
            if self.env.finished:
                return
            if obs and obs[0].kind == "error":
                self.shop.cart_synced = False
        if not self.env.finished:
            self.last_resort()

    def unshown_needs(self, pick):
        """R5: the stated word values (eq / in) that `pick` leaves unconfirmed and that no in-stock, affordable listing of
        the chosen aisle shows anywhere (title, text or facts) without its facts contradicting it. Each such need rules out
        every candidate we have."""
        cap, qty = self.budget_world(), self.ledger.quantity or 1
        pool = [p for p in self.shop.products.values() if p.get("in_stock") is not False and p.get("price") is not None
                and not (cap is not None and p["price"] * qty > cap + 1e-9)
                and not (self.category and p.get("category") and p["category"].lower() != self.category.lower())]
        def blob(p):
            return " ".join([str(p.get("title") or ""), str(p.get("text") or "")]
                            + [f"{k} {v}" for k, v in (p.get("props") or {}).items() if not _is_options(k)])
        out = []
        for n in self.ledger.needs:
            if not n.hard or n.source != "request" or n.op not in ("eq", "in") or need_status(n, pick) is not None:
                continue
            values = n.value if isinstance(n.value, list) else [n.value]
            values = [str(v) for v in values if v and not re.search(r"[0-9]", str(v))]
            open_ = [blob(p) for p in pool if need_status(n, p) is not False]   # a listing whose facts contradict it shows nothing
            if values and not any(set(canon(v).split()) <= set(canon(b).split()) for v in values for b in open_):   # its words, any order
                out.append(n.as_dict())
        return out

    def absent_needs(self, pick):
        """Q6-D2: stated needs (eq / in, numbers included) that `pick` leaves unconfirmed with no fact of its own named
        after the attribute, while 3 or more in-stock, affordable listings of the chosen aisle carry that attribute and
        every one contradicts the stated value: the aisle uses the attribute and none of its listings has the value."""
        if not self.category or self.line_ledgers:
            return []
        cap, qty = self.budget_world(), self.ledger.quantity or 1
        pool = [p for p in self.shop.products.values() if p.get("in_stock") is not False and p.get("price") is not None
                and p.get("props") and (p.get("category") or "").lower() == self.category.lower()
                and not (cap is not None and p["price"] * qty > cap + 1e-9)]
        out = []
        for n in self.ledger.needs:
            if (not n.hard or n.source != "request" or n.op not in ("eq", "in") or not n.prop or not n.value
                    or need_status(n, pick) is not None or related_facts(pick, n.prop)):
                continue
            shown = [need_status(n, p) for p in pool if related_facts(p, n.prop)]
            if len(shown) >= 3 and all(x is False for x in shown):
                out.append(n.as_dict())
        return out

    def widen_for(self, exclude=(), limit=3):
        """Unread viable candidates of the current ledger worth opening when its pick is unconfirmed:
        listings hinted by the stated values first, then the cheapest."""
        if not self.spec_view:
            return []
        pool = [p for p in self.viable() if p["id"] not in exclude and not self.shop.has_view(p["id"], self.spec_view)]
        pool.sort(key=lambda p: (not self.hinted(p), p.get("price") if p.get("price") is not None else float("inf")))
        return [p["id"] for p in pool[:limit]]

    def finish_lines(self, subs, chosen):
        """Checkout for a multi-line order (see drive_lines)."""
        base = self.ledger
        widened = 0
        for _ in range(CONFIG["max_order_attempts"] + 8):   # every round spends a step
            if self.env.finished or self.steps_left() <= 0 or self.__dict__.pop("named_stuck", False):
                break   # P1: a stuck named step goes to the last resort below
            for i, led in enumerate(subs):   # re-pick only the lines whose item changed or failed its needs
                p = self.shop.products.get(chosen[i]) if chosen[i] else None
                if (p is None or p.get("in_stock") is False or chosen[i] in self.rejected or not self.shop.is_fresh(chosen[i])
                        or self.R.judge(led, p, None).ok is False):
                    self.ledger, self.category = led, led.notes.get("category")
                    best = self.pick() or self.closest_candidate()
                    chosen[i] = best["id"] if best else None
            self.ledger = base
            keep = 0 if "refused" in self.events else self.refusal_reserve
            fill = len([pid for pid in chosen if pid]) + 2   # cart lines, re-read, order
            unread = [i for i, pid in enumerate(chosen) if pid and self.spec_view and subs[i].needs
                      and not self.shop.has_view(pid, self.spec_view)]
            if unread and self.steps_left() - fill - keep >= len(unread):
                # a title is not the listing's facts: open each unread pick before it is carted
                self.act([self.adapter.detail(chosen[i], self.spec_view) for i in unread])
                for i in unread:
                    if not self.shop.has_view(chosen[i], self.spec_view):
                        self.rejected.add(chosen[i])   # a view the world will not give: never pick it again
                    elif self.R.judge(subs[i], self.shop.products[chosen[i]], None).ok is None:
                        self.ledger, self.category = subs[i], subs[i].notes.get("category")
                        alt = self.pick()   # confirmed candidates rank first: one whose facts meet every need wins
                        if alt is not None and alt.get("_verdict") is True and alt["id"] != chosen[i]:
                            chosen[i] = alt["id"]
                        self.ledger = base
                continue
            unsure = [i for i, pid in enumerate(chosen) if pid and subs[i].needs
                      and self.R.judge(subs[i], self.shop.products[pid], None).ok is not True]
            if unsure and widened < 2:
                batch = []
                for i in unsure:   # unknown is not met: open more candidates of that line before the cart fill
                    self.ledger, self.category = subs[i], subs[i].notes.get("category")
                    batch += [b for b in self.widen_for(exclude=set(chosen)) if b not in batch]
                self.ledger = base
                batch = batch[:max(0, self.steps_left() - fill - keep)]
                if batch:
                    widened += 1
                    self.act([self.adapter.detail(b, self.spec_view) for b in batch])
                    for i in unsure:   # a candidate whose facts now meet every need of the line wins
                        self.ledger, self.category = subs[i], subs[i].notes.get("category")
                        alt = self.pick()
                        if alt is not None and alt.get("_verdict") is True:
                            chosen[i] = alt["id"]
                    self.ledger = base
                    continue
            unchecked = [i for i, pid in enumerate(chosen) if pid and not self.shop.products[pid].get("_checked")
                         and self.R.judge(subs[i], self.shop.products[pid], None).ok is not True]
            if unchecked and self.llm is not None and CONFIG.get("model_veto") and hasattr(self.R, "final_check"):
                for i in unchecked:
                    self.shop.products[chosen[i]]["_checked"] = True
                judged = self.R.final_check(base, [self.shop.products[chosen[i]] for i in unchecked], self.llm,
                                            lines=subs)
                bad = [i for i in unchecked if judged.get(chosen[i]) is False]
                self.note("model check", lines={chosen[i]: judged.get(chosen[i]) for i in unchecked})
                if bad:
                    for i in bad:
                        self.rejected.add(chosen[i])
                    continue
            for i, led in enumerate(subs):
                if _r9_count_only(led.product):
                    chosen[i] = None   # A-R8b G2: a restated count ("3 pcs", "packs") is no line of the basket
            missing = [i for i, pid in enumerate(chosen) if not pid and not _r9_count_only(subs[i].product)]
            if missing and not self.__dict__.get("r9_missing_done"):   # R9b F3: as a single-line order with nothing to order
                self.r9_missing_done = True
                led = subs[missing[0]]
                self.note("multi-line line without a candidate", line=missing[0], product=led.product)
                if self.R.should_decline(led, self.shop, True, self.llm) and self.decline():
                    return
            desired = {}
            for i, pid in enumerate(chosen):
                if pid:
                    desired[pid] = desired.get(pid, 0) + (subs[i].quantity or 1)
            if not desired:
                break
            cap = self.budget_world()   # R9b F2 (F-046 b): the order's total against the total budget
            total = sum((self.shop.products.get(pid, {}).get("price") or 0) * n for pid, n in desired.items())
            if (cap is not None and total > cap + 1e-9 and not self.__dict__.get("r9_budget_done")
                    and _r9_total_budget(base)):   # A-R8b G1
                self.r9_budget_done = True
                cheap = {}
                for i, led in enumerate(subs):
                    if chosen[i]:
                        alt = self._r9_cheapest_own(led)
                        cheap[i] = alt["id"] if alt is not None else chosen[i]
                cheap_total = sum((self.shop.products.get(pid, {}).get("price") or 0) * (subs[i].quantity or 1)
                                  for i, pid in cheap.items())
                self.note("multi-line budget", total=total, cap=cap, cheapest=cheap_total)
                if cheap_total <= cap + 1e-9:
                    for i, pid in cheap.items():
                        chosen[i] = pid
                    continue
                if self.R.should_decline(base, self.shop, True, self.llm) and self.decline():
                    return   # as a single-line order does when nothing is affordable
            if not self.ensure_cart_lines(desired):
                self.note("cart fill failed", desired=desired)
                if self.steps_left() <= 1:
                    break
            if (self.steps_left() > len(desired) + 2 and "detail" in self.adapter.roles
                    and self.shop.cart_reread != self.shop.cart_version):
                self.act([self.adapter.detail(pid) for pid in desired])   # silent stock-outs
                self.shop.cart_reread = self.shop.cart_version
                if any(self.shop.products.get(pid, {}).get("in_stock") is False for pid in desired):
                    continue
            self.checkouts += 1
            self.act([self.adapter.order()])
            if self.__dict__.pop("relax_hold", False) and not self.env.finished and self.steps_left() >= 1:
                self.checkouts += 1   # RXL: a multi-line order held only by a dropped need: the same cart again
                self.act([self.adapter.order()])
        if not self.env.finished:
            self.ledger = subs[0] if subs else base
            self.last_resort()
            self.ledger = base

    def ensure_cart_lines(self, desired) -> bool:
        """Make the cart exactly `desired` ({id: units})."""
        current = {k: v for k, v in self.shop.cart.items() if v > 0}
        if current == desired:
            return True
        if self.adapter.cart_mode == "vector":
            lines = {k: 0 for k in current if k not in desired}
            lines.update(desired)
            self.line_debt += max(0, len(lines) - 1)
            obs = self.act([self.adapter.cart_vector(lines)])
            return bool(obs) and obs[0].kind != "error" and obs[0].ok is not False
        ok = True
        for pid, have in current.items():
            if pid not in desired:
                for call in self.adapter.cart_ops(pid, have, 0) or []:
                    self.act([call])
        for pid, want in desired.items():
            ops = self.adapter.cart_ops(pid, self.shop.cart.get(pid, 0), want)
            if ops is None:
                ok = False
                continue
            for call in ops:
                obs = self.act([call])
                if not obs or obs[0].kind == "error" or obs[0].ok is False:
                    ok = False
        return ok

    def _r9_empty_shelf(self) -> bool:
        """A-R9 G3: True when a fallback search listed something, so the decline waits (once per episode)."""
        if self.__dict__.get("r9_shelf_done") or self.shop.products or len(self.ledger.lines or []) >= 2:
            return False
        self.r9_shelf_done = True
        role = "search" if "search" in self.adapter.roles else None
        if role is None or self.steps_left() < 6 or self.time_left() < CONFIG["finish_margin_s"] + 30:
            return False
        tried = {str(p).lower() for r, p in self.shop.queries if r == role}
        stated = {w for n in self.ledger.needs   # a need's value (wanted or excluded: "not black") is never searched
                  for w in words(" ".join(map(str, n.value)) if isinstance(n.value, list) else str(n.value or ""))}
        words_to_try = []   # the product phrase's own shape only (no word list): head end first, words of 4+ letters
        for w in reversed(words(self.ledger.product or "")):
            if len(w) >= 4 and not w.isdigit() and w not in tried and w not in stated and w not in words_to_try:
                words_to_try.append(w)
        for w in words_to_try[:3]:
            call = self._query_call(role, w, None)
            if call is None or not self.can_spend(1):
                break
            obs = self.act([call])
            self.noted(role, w, obs[0] if obs else None)
            ids = list((self.shop.queries.get((role, w)) or {}).get("ids") or [])
            top = [p for p in (self.shop.products.get(i) for i in ids) if p]   # the first page this word listed
            hits = [p for p in top if _r9_stem(w) in _r9_title_main(p.get("title"))]
            cats = Counter(str(p.get("category") or "").lower() for p in hits)
            if not (len(hits) >= 3 and cats and cats.most_common(1)[0][1] >= 3):
                self.rejected.update(ids)   # the word is not what these listings are (R9 P4's store check)
                continue
            if self.shop.products:
                self.note("empty shelf fallback", word=w)
                self.inspect_phase(limit=4)
                self.question_phase()
                self.inspect_phase()
                return True
        return False

    def decline(self) -> bool:
        if self._r9_empty_shelf():   # A-R9 G3: nothing was ever listed: broader searches before declining
            return False
        call = self.adapter.decline(self.R.decline_note(self.ledger))
        if call is None or self.steps_left() <= 0:
            return False
        self.act([call])
        return self.env.finished

    def time_up_finish(self):
        """A-R4: one terminal call once the time guard tripped: the order when the basket holds something, else the
        decline, else the order. Never another step."""
        self.note("time guard", elapsed=round(self.env.clock() - self.env.started, 1))
        if self.env.finished or self.steps_left() <= 0:
            return
        call = self.adapter.order() if any(q > 0 for q in self.shop.cart.values()) and "order" in self.adapter.roles else None
        call = call or self.adapter.decline(self.R.decline_note(self.ledger))
        call = call or (self.adapter.order() if "order" in self.adapter.roles else None)
        if call is None:
            return
        try:   # the class's own act: the stated-procedure wrapper would send a confirmation message first
            globals().get("_sp_core_act", Run.act)(self, [call])
        except (_TimeUp, Exception) as exc:  # noqa: BLE001 - the last call must never cost the episode
            self.note("time guard error", error=type(exc).__name__)

    def last_resort(self):
        """No viable product and no clean decline: decline if any route exists, else order
        the cheapest in-stock product rather than leave the episode unfinished."""
        if self.env.finished or self.steps_left() <= 0:
            return
        for _ in range(3):   # a held decline (a checkout event) needs the finish repeated
            if self.decline():
                return
            if self.env.finished or self.steps_left() <= 1 or "decline" not in self.adapter.roles:
                break
        if self.env.finished:
            return
        if not self.shop.products and "search" in self.adapter.roles and self.steps_left() > 2:
            phrase = self.ledger.product or " ".join(self.ledger.terms) or self.ledger.category or "item"
            self.act([self.adapter.search(phrase)])
        if not any(q > 0 for q in self.shop.cart.values()):
            stocked = sorted((p for p in self.shop.products.values()
                              if p.get("in_stock") is not False and p.get("price") is not None),
                             key=lambda p: p["price"])
            if not stocked or not self.ensure_cart(stocked[0]["id"]):
                return
        if self.steps_left() > 0 and "order" in self.adapter.roles:
            self.act([self.adapter.order()])


def _route_scoped_needs(subs):
    """A reveal scoped to another line ("<attr> of <another line's item>") moves to that line."""
    for led in subs:
        for need in list(led.needs):
            scope = getattr(need, "scope", None)
            if not scope:
                continue
            target = set(toks(scope))
            best = max(subs, key=lambda l: len(target & (set(toks(l.product or "")) | set(toks(l.notes.get("category") or "")))))
            if best is not led and target & (set(toks(best.product or "")) | set(toks(best.notes.get("category") or ""))):
                led.needs.remove(need)
                best.needs.append(need)


def _r9_stem(w):
    w = str(w).lower()
    for suf in ("es", "s"):
        if len(w) > 3 and w.endswith(suf) and not w.endswith("ss"):
            return w[: -len(suf)]
    return w


def _r9_title_main(title):
    """R9 P4: the main noun of a listing title: the last word of its first segment ("Evening Dress" -> dress)."""
    seg = re.split(r",|\s-\s|\s\|\s|\(|\s(?:with|for|in|of|and)\s", str(title or ""), maxsplit=1)[0]
    return {_r9_stem(w) for w in words(seg)[-1:]}


def _r9_soft_line_kept(run, led, before):
    """R9 P4: a soft line stays when >= 3 of its first search's top-5 results have its head noun as their main noun
    and share one category. No search ran: kept (no evidence to drop it)."""
    head = _r9_stem((words(led.product or "") or [""])[-1])
    ids = []
    for key, state in run.shop.queries.items():
        if key not in before:
            ids += [i for i in state.get("ids", []) if i not in ids]
    top = [run.shop.products.get(i) for i in ids[:5]]
    top = [p for p in top if p]
    if not head or not top:
        return True
    hits = [p for p in top if head in _r9_title_main(p.get("title"))]
    cats = Counter(str(p.get("category") or "").lower() for p in hits)
    return len(hits) >= 3 and bool(cats) and cats.most_common(1)[0][1] >= 3


_R9_NUM_WORD = re.compile(r"\b(\d+(?:\.\d+)?)(\s*-?\s*)([a-z]{1,10})(?![-&'\w])", re.IGNORECASE)


def _r9_learned(products, head):
    """R9 P5: measure words, numeric keys and glued-x values learned from the facts of the listings read."""
    seen, strong, keys, glued_x = Counter(), set(), Counter(), 0
    for p in products:
        for k, v in (p.get("props") or {}).items():
            v = str(v)
            for m in _R9_NUM_WORD.finditer(v):
                w = _r9_stem(m.group(3))
                if w == head or w == "x":
                    continue
                seen[w] += 1
                if "." in m.group(1) or not m.group(2):
                    strong.add(w)
            if re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", v):
                keys[" ".join(words(k))] += 1
            elif re.fullmatch(r"\s*\d+(?:\.\d+)?\s*x\s*", v, re.IGNORECASE):
                glued_x += 1
    return ({w for w, c in seen.items() if c >= 2 and w in strong},
            {k for k, c in keys.items() if c >= 2 and len(k.split()) == 1}, glued_x >= 2)


def _r9_late_quantity(run, text, product, qty):
    """R9 P5: the line's quantity re-read with the learned measures held; lowered only when holding them changes it."""
    head = _r9_stem((words(product or "") or [""])[-1])
    if not head or qty <= 1 or not text:
        return qty
    units, keys, gx = _r9_learned([p for p in run.shop.products.values() if p.get("props")], head)
    base_text = _mask_measures(text)[0]
    held = base_text
    if units:
        held = _R9_NUM_WORD.sub(lambda m: "zqlearnedz" if _r9_stem(m.group(3)) in units else m.group(0), held)
    if keys:
        alt = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
        held = re.sub(r"\b(?:" + alt + r")\s+\d+(?:\.\d+)?\b", "zqlearnedz", held, flags=re.IGNORECASE)
    if gx:
        held = re.sub(r"\b\d+(?:\.\d+)?x\b", "zqlearnedz", held, flags=re.IGNORECASE)
    if held == base_text:
        return qty
    q_orig, q_held = line_quantity(base_text, _mask_measures(product)[0]), line_quantity(held, _mask_measures(product)[0])
    if q_orig == qty and q_held < qty:
        run.note("learned measure", product=product, qty=qty, now=q_held, units=sorted(units)[:6], keys=sorted(keys)[:4])
        return q_held
    return qty


def drive_lines(run):
    """Multi-line orders: run the single-line phases once per line (each line with its own
    ledger and a share of the steps and messages), then fill the cart with every line at its
    quantity and check out; after a refusal only the lines whose item changed are re-picked."""
    base = run.ledger
    subs = []
    for spec in base.lines:
        led = Ledger(base.query, base.opening)
        for field in ("budget", "budget_unit", "msg_budget", "step_budget", "prefer", "stock_only"):
            setattr(led, field, getattr(base, field))
        led.notes["decline_stated"] = base.notes.get("decline_stated")
        led.product, led.quantity, led.category = spec["product"], spec["quantity"], spec["category"]
        led.needs = list(spec["needs"])
        led.notes["tentative"] = list(spec.get("tentative") or [])
        if spec.get("refs"):
            led.notes["refs"] = [dict(r) for r in spec["refs"]]
        led.terms = [w for w in words(led.product) if w not in _STOP][:5]
        led.notes["line_text"] = spec.get("text")   # the line's own sentences: a question hint reads only them
        led.notes["soft"] = bool(spec.get("soft"))   # R9 P4
        subs.append(led)
    run.line_ledgers = subs
    for led in subs:
        led.notes["q_siblings"] = subs   # section 6c: reveals that name another line go to its ledger
    run.note("lines", lines=[[l.product, l.quantity, [n.as_dict() for n in l.needs]] for l in subs])
    chosen, dropped = [], set()
    for i, led in enumerate(subs):
        left = len(subs) - i
        room = run.steps_left() - run.reserve - run.refusal_reserve - len(subs)
        share = max(3, room // left)
        line_end = run.env.steps + share
        run.ledger, run.category, run.enumerated, run.forced_pick = led, None, False, None
        run.line_step_limit = run.env.steps + max(2, share // 3)   # retrieval: at most a third of the line's steps
        before = set(run.shop.queries)
        run.search_phase()
        if led.notes.get("soft") and not _r9_soft_line_kept(run, led, before):   # R9 P4: not a store item
            run.note("soft line dropped", line=i, product=led.product)
            dropped.add(i)
            chosen.append(None)
            continue
        run.browse_phase(rounds=1)
        run.line_step_limit = line_end
        run.inspect_phase(limit=4)
        run.question_phase(limit=max(1, run.messages_left() // left))
        run.resolve_references()
        run.value_requery()
        run.inspect_phase(limit=4)
        led.quantity = _r9_late_quantity(run, led.notes.get("line_text") or base.query, led.product, led.quantity)   # R9 P5
        best = run.pick() or run.closest_candidate()
        led.notes["category"] = run.category
        led.notes["picked_with"] = sorted(need_key(n) for n in led.needs)
        chosen.append(best["id"] if best else None)
        _route_scoped_needs(subs)
    if dropped:   # R9 P4: soft lines the store does not hold leave the order before any pick or decline
        subs = [l for j, l in enumerate(subs) if j not in dropped]
        chosen = [c for j, c in enumerate(chosen) if j not in dropped]
        base.lines = [l for j, l in enumerate(base.lines) if j not in dropped]
        run.line_ledgers = subs
    for i, led in enumerate(subs):   # a later reply changed an earlier line's needs: pick that line again
        if i < len(chosen) and led.notes.get("picked_with") not in (None, sorted(need_key(n) for n in led.needs)):
            run.ledger, run.category, run.enumerated, run.forced_pick = led, led.notes.get("category"), False, None
            best = run.pick()
            if best and best["id"] != chosen[i]:
                run.note("line re-picked after a routed reply", line=i, pick=best["id"])
                chosen[i] = best["id"]
    run.line_step_limit = None
    run.finish_lines(subs, chosen)
    run.ledger = base


def drive_rules(run):
    """The deterministic path: search, inspect, ask, inspect again, finish."""
    if len(run.ledger.lines) >= 2:
        run.note("rules driver, multi-line")
        return drive_lines(run)
    run.note("rules driver")
    run.search_phase()
    run.browse_phase()
    run.inspect_phase(limit=4)
    run.title_look()
    run.capture_request_values()
    run.bind_need_props()
    run.harvest_request_values()
    run.question_phase()
    run.resolve_references()
    run.value_requery()
    run.reread_replies()
    run.refine_cheapest()
    run.inspect_phase()
    run.reread_replies()
    run.ledger.quantity = _r9_late_quantity(run, run.ledger.query, run.ledger.product, run.ledger.quantity)   # R9 P5
    run.finish()


_ACTOR_PROMPT = """You run a store's shopping assistant inside a simulator. Every reply must be exactly one call to one of the tools you are given; each call uses one step of a limited budget.

Your job: place one order for exactly the product the shopper needs, in the quantity they asked for, as a single cart line, within their budget. If no listing meets the needs, decline instead of ordering a substitute.

How to work:
- Search with the product words from the request; look at more pages when the cheapest suitable item may be further down.
- Open product details before choosing, and check every property the shopper cares about.
- The shopper has unstated preferences. Each message asks a single question, naming the property, and the answer is used. Messages are limited, so ask only what tells the candidates apart.
- Unless the request says otherwise, pick the lowest-priced product that meets every need.
- Shopper updates can arrive with any tool result (shopper_update) or when an order is refused (order_refused). Treat them as changes to the requirements, prices, stock or budget, and act on them.
- Never put product ids in a message to the shopper or in a decline note.
- A tool result marked blocked was not executed and cost no step; follow its reason.
"""


class ModelDriver:
    """LLM actor inside a gated harness. The harness repairs and validates arguments, answers
    repeat reads from cache without spending a step, enforces the message budget, refuses an
    order that fails the order gate, narrows the tools in the endgame and hands over to the
    deterministic finish when time or steps run short."""

    def __init__(self, run):
        self.run = run
        self.blocked = 0
        self.decline_blocks = 0

    def drive(self):
        run = self.run
        run.note("model driver", models=run.llm.usable("actor"))
        messages = [{"role": "system", "content": _ACTOR_PROMPT},
                    {"role": "user", "content": self.brief()}]
        idle = 0
        while not run.env.finished:
            if self.must_finish():
                run.note("harness: finishing", steps_left=run.steps_left(), time_left=round(run.time_left(), 1))
                return run.finish()
            reply = run.llm.chat(self.compacted(messages), purpose="actor", tools=self.offered_tools())
            if reply is None:
                run.note("model chain exhausted", error=run.llm.last_error)
                return drive_rules(run)
            calls = [c for c in reply.get("tool_calls") or [] if isinstance(c, dict)]
            if not calls:
                idle += 1
                if idle > 2:
                    return drive_rules(run)
                messages += [{"role": "assistant", "content": reply.get("content") or ""},
                             {"role": "user", "content": "Reply with exactly one tool call."}]
                continue
            idle = 0
            call = calls[0]
            call_id = call.get("id") or f"call_{run.env.steps}_{self.blocked}"
            call = dict(call, id=call_id)
            messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": [call]})
            text, executed = self.gated(call)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            if not executed:
                run.note("free", call=(call.get("function") or {}).get("name"), result=clip(text, 200))
                self.blocked += 1
                if self.blocked > CONFIG["max_blocked"]:
                    run.note("harness: too many blocked calls")
                    return drive_rules(run)

    def brief(self) -> str:
        run = self.run
        return json.dumps({
            "shopper_request": run.query, "opening": run.opening,
            "what_we_know": run.ledger.summary(),
            "tool_roles": {r: b.tool.name for r, b in sorted(run.adapter.roles.items())},
            "limits": {"steps": run.step_cap, "messages": run.messages_left()},
        }, ensure_ascii=False)

    def status(self) -> dict:
        run = self.run
        return {"steps_left": run.steps_left(), "messages_left": run.messages_left(),
                "budget": run.budget_world(), "cart": run.cart_lines()["lines"],
                "needs": [[n.prop, n.value] for n in run.ledger.needs]}

    @staticmethod
    def digest(content) -> str:
        """A short form of an old tool result: ids and prices, and every shopper word kept."""
        try:
            data = json.loads(content)
        except (TypeError, ValueError):
            return clip(content, 240)
        if not isinstance(data, dict):
            return clip(content, 240)
        keep = {k: data[k] for k in ("kind", "error", "blocked", "shopper_reply", "shopper_update",
                                     "order_refused", "ok", "cart") if k in data}
        if data.get("products"):
            keep["products"] = [[p.get("id"), p.get("price"), p.get("in_stock")] for p in data["products"]]
        return json.dumps(keep, ensure_ascii=False)

    def compacted(self, messages) -> list:
        tool_at = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        old = set(tool_at[: -CONFIG["actor_keep_results"]]) if len(tool_at) > CONFIG["actor_keep_results"] else set()
        return [dict(m, content=self.digest(m["content"])) if i in old else m for i, m in enumerate(messages)]

    def must_finish(self) -> bool:
        run = self.run
        return run.time_left() < CONFIG["finish_margin_s"] or run.steps_left() <= 2

    def offered_tools(self) -> list:
        run = self.run
        tools = [t.decl for t in run.adapter.tools]
        if run.steps_left() <= CONFIG["reserve_steps"] + 1:
            keep = {b.tool.name for r, b in run.adapter.roles.items()
                    if r in ("cart_add", "cart_view", "order", "decline")}
            tools = [t.decl for t in run.adapter.tools if t.name in keep] or tools
        if run.messages_left() <= 0 and "message" in run.adapter.roles:
            name = run.adapter.roles["message"].tool.name
            tools = [d for d in tools if (d.get("function") or d).get("name") != name] or tools
        return tools

    def _blocked(self, reason) -> tuple:
        return json.dumps({"blocked": reason, "status": self.status()}, ensure_ascii=False), False

    def gated(self, call) -> tuple:
        """(tool result text, executed?) for one model tool call."""
        run, adapter = self.run, self.run.adapter
        fn = call.get("function") or {}
        name, raw = fn.get("name"), fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
        except ValueError:
            return self._blocked("arguments were not valid JSON")
        tool = adapter.by_name.get(name)
        if tool is None:
            guess = _closest_key(name or "", list(adapter.by_name))
            if guess is None:
                return self._blocked(f"unknown tool {name!r}")
            tool = adapter.by_name[guess]
        args = repair(args if isinstance(args, dict) else {}, tool.schema)
        problems = schema_problems(args, tool.schema)
        if problems:
            return self._blocked("invalid arguments: " + "; ".join(problems[:3]))
        role = adapter.role_of(tool.name, args)
        if role in ("detail", "spec"):
            pid = args.get(tool.param("id"))
            product = run.shop.products.get(pid)
            if product and run.shop.view_fresh(pid, role):
                return json.dumps({"cached": True, "products": [compact_product(product, True)],
                                   "status": self.status()}, ensure_ascii=False), False
        if role == "cart_view" and run.shop.cart_synced:
            return json.dumps({"cached": True, "cart": run.cart_lines(), "status": self.status()}), False
        if role == "message" and run.messages_left() <= 0:
            return self._blocked("no messages left")
        if role == "order":
            ok, problems = run.order_gate()
            if not ok and run.steps_left() > 1:
                return self._blocked("order gate: " + "; ".join(problems))
        if role == "decline":
            viable = run.viable()
            if viable and self.decline_blocks < 2 and run.steps_left() > CONFIG["reserve_steps"] + 1:
                self.decline_blocks += 1
                return self._blocked(f"{len(viable)} products still look viable; check them before declining")
        obs = run.act([(tool.name, args)])
        result = obs[0].summary() if obs else {"error": "environment call failed"}
        result["status"] = self.status()
        return json.dumps(result, ensure_ascii=False), True


# ======================================================================================
# Entry points
# ======================================================================================

def run_episode(problem_data, env_client, llm_client=None, clock=time.monotonic, reasoning=None,
                discovery_client=None):
    """One episode with injectable clients (offline tests use fakes). Returns the Run."""
    run = Run(problem_data, env_client, llm_client, clock, reasoning, discovery_client)
    if run.env.finished:
        return run
    try:
        try:
            if CONFIG["actor"] == "model" and run.llm is not None and run.llm.alive():
                # the actor spends up to CONFIG["max_calls"]; assist mode spends a handful
                ModelDriver(run).drive()
            else:
                drive_rules(run)
        except Exception as exc:  # noqa: BLE001 - always try to close the episode
            run.note("driver error", error=f"{type(exc).__name__}: {exc}")
            try:
                if not run.env.finished:
                    run.finish()
            except Exception as exc2:  # noqa: BLE001
                run.note("finish error", error=f"{type(exc2).__name__}: {exc2}")
    except _TimeUp:   # A-R4: past the margin: one terminal call, then done
        run.time_up_finish()
    run.note("done", finished=run.env.finished, steps=run.env.steps,
             llm_calls=run.llm.calls if run.llm else 0, llm_cost=round(run.llm.cost, 5) if run.llm else 0,
             llm_log=run.llm.log if run.llm else [])
    return run


def agent_main(problem_data):
    """Validator entry point: one episode through the sandbox proxy."""
    if ProxyClient is None:
        raise RuntimeError("the sandbox ProxyClient is not importable")
    env_client = ProxyClient(timeout=CONFIG["env_timeout_s"], max_retries=2)
    llm_client = ProxyClient(timeout=CONFIG["llm_timeout_s"], max_retries=1 + CONFIG["llm_retries"])
    discovery_client = ProxyClient(timeout=CONFIG["discover_timeout_s"], max_retries=1)
    return run_episode(problem_data, env_client, llm_client, discovery_client=discovery_client).trace


# ---- stated procedure (claude-3-18): a step the shopper asks for before the order --------------------------------------
CONFIG.setdefault("sproc", True)
CONFIG.setdefault("sproc_spare", True)   # confirm before any order while a message is left
CONFIG.setdefault("sproc_always", True)   # keep one message for the basket confirmation in every task
CONFIG.setdefault("sproc_always_min_cap", 4)   # ... whose message cap is at least this (3-message worlds keep their questions)
_SP_ORDER = set(_ROLE_NAME["order"]) | set(_E_BUY)
_SP_SPEAKER = {"me", "my", "us", "our", "i", "we"}


def _sp_order_word(w, nxt=None):
    """an order word, or its gerund ("ordering", "buying", "placing", "committing"; "checking out")"""
    if w in _SP_ORDER or (nxt and w + nxt in _SP_ORDER):   # "check out"
        return True
    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        if stem in _SP_ORDER or stem + "e" in _SP_ORDER or stem[:-1] in _SP_ORDER:
            return True
        return stem + nxt in _SP_ORDER if nxt else False
    return False


def p_stated_procedure(text):
    """SPROC: the request sentence asking to hear from the agent before the order, or None."""
    for s in re.split(r"(?<=[.!?;])\s+|\n+", str(text or "")):
        ws = p_words(s)
        if not any(w in _SP_SPEAKER or w.startswith("i'") for w in ws):
            continue
        for k, w in enumerate(ws):
            nxt = ws[k + 1] if k + 1 < len(ws) else None
            if w == "before" or (w in ("prior", "ahead") and nxt in ("to", "of")):
                j = k + (1 if w == "before" else 2)
                end = next((i for i in range(j, len(ws)) if ws[i] in ".;!?,"), len(ws))
                tail = ws[j:end]
                if tail and (tail[0] == "you" or _sp_order_word(tail[0], tail[1] if len(tail) > 1 else None)
                             or any(x in _SP_ORDER for x in tail[:3])):
                    return s.strip()
            elif w in ("until", "till"):   # "don't <order> until I ...": a negated order, released by the speaker
                head, tail = ws[:k], ws[k + 1:k + 4]
                if (any(x in _SP_SPEAKER or x.startswith("i'") for x in tail)
                        and any(x in ("not", "no", "never") or x.endswith("n't") for x in head)
                        and any(_sp_order_word(x, y) for x, y in zip(head, head[1:] + [None]))):
                    return s.strip()
            elif w == "first" and (nxt is None or nxt in ".;!?,"):   # "check with me first", "get my ok first"
                clause = ws[max((i + 1 for i in range(k) if ws[i] in ".;!?,"), default=0):k]
                if any(x in ("my", "our") or (x in ("me", "us") and i and clause[i - 1] in ("with", "by", "past", "from", "to"))
                       for i, x in enumerate(clause)):
                    return s.strip()
    return None


def _sp_state(run):
    st = run.__dict__.get("_sproc")
    if st is None:
        text = None
        if CONFIG.get("sproc"):
            parts = [run.query] + ([run.opening] if isinstance(run.opening, str) else [])
            text = p_stated_procedure(" ".join(str(x) for x in parts if x))
        st = run.__dict__["_sproc"] = {"text": text, "done": False}
        if text:
            run.note("stated procedure", procedure=text)
    return st


def _sp_text(run):
    """the procedure in force now: a stated one, or (ALWAYS) the basket confirmation in a world whose message cap is at
    least CONFIG["sproc_always_min_cap"]. Evaluated on every call: the cap may arrive late (a status read)."""
    st = _sp_state(run)
    if st["text"]:
        return st["text"]
    if CONFIG.get("sproc") and CONFIG.get("sproc_always"):
        cap = run.ledger.msg_budget if run.ledger.msg_budget is not None else run.status_cap
        if cap is not None and cap >= CONFIG.get("sproc_always_min_cap", 0):
            return "(always: the basket is confirmed before every order)"
    return None


def _sp_pending(run):
    """a procedure whose first confirmation is still owed (one message and one step are kept for it)"""
    return bool(_sp_text(run)) and not _sp_state(run)["done"]


def _sp_cart(run):
    return sorted((pid, q) for pid, q in run.shop.cart.items() if q > 0)


def _sp_owed(run):
    """a confirmation is owed before this order: a stated procedure (or, with sproc_spare, any order while a message is
    left) whose last confirmation is missing or named another cart (a refusal, stock-out or budget change moved it)"""
    st = _sp_state(run)
    if not (_sp_text(run) or CONFIG.get("sproc_spare")):
        return False
    return st.get("cart") != _sp_cart(run)


def sp_confirm_text(run):
    items, total = [], 0
    for pid, q in _sp_cart(run):
        p = run.shop.products.get(pid) or {}
        price = p.get("price")
        line = f"{q} x {clip(str(p.get('title') or 'the item I picked'), 60)}"
        if isinstance(price, (int, float)):
            line += f" at {price:g} each"
            total += price * q
        items.append(line)
    cost = f", {total:g} in total" if total else ""
    return f"Here is the basket for your yes: {'; '.join(items) or 'the item I picked'}{cost}. Shall I place the order?"


_sp_core_messages_left = Run.messages_left
_sp_core_reserve = Run.reserve
_sp_core_act = Run.act


def _sp_messages_left(self):
    n = _sp_core_messages_left(self)
    return max(0, n - 1) if _sp_pending(self) and not self.__dict__.get("_sp_sending") else n


def _sp_act(self, calls):
    live = [c for c in calls if c]
    if live and any(self.adapter.role_of(c[0], c[1]) == "order" for c in live) and _sp_owed(self):
        st = self.__dict__["_sproc"]
        st["done"] = True
        self.__dict__["_sp_sending"] = True
        try:
            if (self.messages_left() > 0 and self.steps_left() >= 2 and "message" in self.adapter.roles
                    and _sp_cart(self)):
                st["cart"] = _sp_cart(self)
                asked, self.ledger.asked = self.ledger.asked, []   # the reply answers the confirmation, not a property
                try:
                    obs = _sp_core_act(self, [self.adapter.message(sp_confirm_text(self))])
                finally:
                    self.ledger.asked = asked + [a for a in self.ledger.asked if a not in asked]
                # any reply is the approval: the order follows the confirmation (a decline after it fails the request)
                self.note("stated procedure: confirmed before the order", stated=_sp_text(self),
                          reply=clip(obs[0].text, 200) if obs and obs[0].text else None)
        finally:
            self.__dict__["_sp_sending"] = False
    return _sp_core_act(self, calls)


Run.messages_left = _sp_messages_left
Run.reserve = property(lambda self: _sp_core_reserve.fget(self) + (1 if _sp_pending(self) else 0))
Run.act = _sp_act

# ---- COMPAT: parts of a multi-line order that must fit one model (claude-3-22) -------------------------------
# Acts only when the request ties its lines together (a model-shaped token, or one requested line "with" another) and
# the listings carry the relation in their own text. Tokens (any case, separator or width): a model is a word and a
# number; a part code is letters glued to digits (the code that ends a listing's title). Names compare in one normal
# form, and a token counts only through another listing (CM2).
#   identity   model tokens of a listing outside any list below: the model it IS (and the same model under a second name)
#   claims     "with <model>, <model>": the models this listing fits
#   part fit   "<part code> ...: <model>, ...": the models THAT part fits (the listing whose title ends with the code)
# The anchor line is the one whose listings carry identities; its pick is the cheapest model that every other line can
# fit, and each other line keeps the listings that fit it (when none does, the line is left as it was).
CONFIG.setdefault("compat", True)
CONFIG.setdefault("compat_queries", 2)        # model-name searches per line that has no fitting listing yet
_CM_M = r"[A-Za-z][A-Za-z]{1,15}[ \-]?\d+[A-Za-z]?"
_CM_C = r"[A-Za-z]{1,6}-?\d{2,}"
_CM_LIST = r"(?:{m})(?:\s*(?:,|\band\b|\bor\b|&|/)\s*(?:{m}))*".format(m=_CM_M)
_CM_MODEL = re.compile(r"\b" + _CM_M + r"\b", re.IGNORECASE)
_CM_WITH = re.compile(r"\bwith\s+(" + _CM_LIST + ")", re.IGNORECASE)
_CM_FITS = re.compile(r"\b(" + _CM_C + r")\b[^.:;]{0,40}:\s*(" + _CM_LIST + ")", re.IGNORECASE)
_CM_TAIL = re.compile(r"(" + _CM_C + r")\s*\W*$", re.IGNORECASE)
_CM_SURFACE = {}   # normal form -> the form a listing wrote (for searches)


def _cm_key(token):
    """CM2: one normal form for a model name or part code: lower case, letters and digits only."""
    key = re.sub(r"[^a-z0-9]", "", str(token).lower())
    _CM_SURFACE.setdefault(key, str(token))
    return key


def _cm_keys(tokens):
    """CM2: normal forms of the model tokens whose word is not an ordinary English word ("this 18", "our 20")."""
    return {_cm_key(t) for t in tokens if re.match(r"[A-Za-z]+", t).group(0).lower() not in _STOP | _FUNCTION_WORDS}


def _cm_text(p):
    return " ".join([str(p.get("title") or ""), str(p.get("text") or "")] + [str(v) for v in (p.get("props") or {}).values()])


def _cm_parse(p):
    """{"code", "names" (identity), "claims", "fits" {code: models}} of one listing, cached per listing."""
    text = _cm_text(p)
    hit = p.get("_cm")
    if hit and hit[0] == len(text):
        return hit[1]
    spans, claims, fits = [], set(), {}
    for m in _CM_WITH.finditer(text):
        claims |= _cm_keys(_CM_MODEL.findall(m.group(1)))
        spans.append(m.span(1))
    for m in _CM_FITS.finditer(text):
        fits.setdefault(_cm_key(m.group(1)), set()).update(_cm_keys(_CM_MODEL.findall(m.group(2))))
        spans.append(m.span(2))
    names = _cm_keys(m.group(0) for m in _CM_MODEL.finditer(text) if not any(a <= m.start() < b for a, b in spans))
    tail = _CM_TAIL.search(str(p.get("title") or ""))
    out = {"code": _cm_key(tail.group(1)) if tail else None, "names": names, "claims": claims, "fits": fits}
    p["_cm"] = (len(text), out)
    return out


def _cm_trigger(query, subs):
    """The request ties its lines: "with" followed by another requested line's own word (CM2: no token shape)."""
    if len(subs) < 2 or not CONFIG.get("compat"):
        return False
    heads = {w for led in subs for w in words(led.product or "")[-1:] if len(w) > 2}
    heads |= {w[:-1] for w in heads if w.endswith("s")}
    for m in re.finditer(r"\bwith\s+([^.;:!?]+)", (query or "").lower()):   # CT1: up to the clause end
        if set(words(m.group(1))[:12]) & heads:
            return True
    return False


def _cm_candidates(run, led):
    run.ledger, run.category = led, led.notes.get("category")
    return [p for p in _CM_VIABLE0(run) if p.get("in_stock") is not False]


def _cm_plan(run, subs):
    """{id(ledger): allowed ids} or None. Anchor = the line whose listings carry identities."""
    fits_of = {}
    for p in list(run.shop.products.values()):
        for code, models in _cm_parse(p)["fits"].items():
            fits_of.setdefault(code, set()).update(models)
    fit = lambda p: _cm_parse(p)["claims"] | fits_of.get(_cm_parse(p)["code"], set())
    known = set().union(*fits_of.values()) if fits_of else set()
    for p in list(run.shop.products.values()):
        known |= _cm_parse(p)["claims"]
    cands = [_cm_candidates(run, led) for led in subs]
    ids = lambda p: _cm_parse(p)["names"] & known   # a model name counts once some listing claims to fit it
    ident = [sum(1 for p in c if ids(p) and not fit(p)) for c in cands]
    claim = [sum(1 for p in c if fit(p)) for c in cands]
    if not any(ident) or not any(claim):
        return None
    a = max(range(len(subs)), key=lambda i: (ident[i] - claim[i], ident[i]))
    others = [i for i in range(len(subs)) if i != a]
    price = lambda p: p.get("price") if isinstance(p.get("price"), (int, float)) else float("inf")
    best = None
    # the anchor comes from the top tier of the line's own ranking (needs first): coverage only breaks ties inside it
    tier = [p for p in cands[a] if ids(p) and p.get("_verdict") == next((q.get("_verdict") for q in cands[a] if ids(q)), None)]
    for anchor in tier:
        names = ids(anchor)
        if not names:
            continue
        cover = {i: [p for p in cands[i] if fit(p) & names] for i in others}
        sure = sum(1 for v in cover.values() if any(p.get("_verdict") is True for p in v))   # fitting AND needs confirmed
        key = (sum(1 for v in cover.values() if v), sure, -tier.index(anchor))
        if best is None or key > best[0]:
            best = (key, names, cover)
    if best is None or best[0][0] == 0:
        return None
    _, names, cover = best
    allowed = {id(subs[a]): {p["id"] for p in cands[a] if ids(p) & names}}
    for i, v in cover.items():
        if v:
            allowed[id(subs[i])] = {p["id"] for p in v}
    run.note("compat plan", anchor_line=a, models=sorted(names), lines={i: len(v) for i, v in cover.items()})
    return allowed, a, names


def _cm_widen(run, subs, a, names):
    """Lines without a fitting candidate yet: search the anchor model names (a listing's text states what it fits)."""
    role = "search" if "search" in run.adapter.roles else None
    if role is None:
        return False
    asked = 0
    for name in sorted(_CM_SURFACE.get(n, n) for n in names):   # CM2: the form a listing wrote
        if asked >= CONFIG["compat_queries"] or not run.can_spend(2) or (role, name) in run.shop.queries:
            continue
        call = run._query_call(role, name, None)
        if call is None:
            continue
        asked += 1
        run.noted(role, name, run.act([call]))
    return asked > 0


_CM_VIABLE0 = Run.viable
_CM_FINISH0 = Run.finish_lines


def _cm_viable(self):
    out = _CM_VIABLE0(self)
    allow = (self.__dict__.get("_cm_allow") or {}).get(id(self.ledger))
    if allow:
        keep = [p for p in out if p["id"] in allow]
        if keep:
            return keep
    return out


def _cm_finish_lines(self, subs, chosen):
    base, category = self.ledger, self.category
    try:
        if CONFIG.get("compat") and _cm_trigger(base.query, subs):
            plan = _cm_plan(self, subs)
            if plan and any(id(led) not in plan[0] for i, led in enumerate(subs) if i != plan[1]):
                if _cm_widen(self, subs, plan[1], plan[2]):
                    plan = _cm_plan(self, subs) or plan
            if plan:
                self.__dict__["_cm_allow"] = plan[0]
                for i, led in enumerate(subs):
                    if id(led) in plan[0]:
                        self.ledger, self.category = led, led.notes.get("category")
                        best = self.pick()
                        if best and best["id"] != chosen[i]:
                            self.note("compat re-pick", line=i, pick=best["id"], was=chosen[i])
                            chosen[i] = best["id"]
    except Exception as exc:   # the resolver must never cost the episode
        self.note("compat error", error=repr(exc)[:120])
    finally:
        self.ledger, self.category = base, category
    return _CM_FINISH0(self, subs, chosen)


Run.viable = _cm_viable
Run.finish_lines = _cm_finish_lines

__all__ = ["agent_main"]
