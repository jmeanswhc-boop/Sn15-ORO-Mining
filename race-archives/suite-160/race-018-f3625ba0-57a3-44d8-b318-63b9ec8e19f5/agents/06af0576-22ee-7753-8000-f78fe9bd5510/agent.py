"""Chidori: a code-driven shopping agent for ORO Bench worlds.

The decision flow (search, paging, candidate screening, question planning, reply handling,
reference chains, cart, events, checkout, decline) is Python. Language models are used only
where reading is needed: once to read the request and the world's tools, then in small
calls to judge listings, interpret the shopper and plan the questions.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.agent.proxy_client import ProxyClient

PARSE_MODEL = "anthropic/claude-sonnet-5.5"
PLAN_MODEL = "anthropic/claude-sonnet-5.5"
VERIFY_MODEL = "anthropic/claude-sonnet-5.5"
FAST_MODEL = "anthropic/claude-haiku-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"

TASK_SECONDS = 255.0        # the network gives each task 300 s
WRAP_SECONDS = 45.0         # with this little time left, finish with what is known
FACT_CHARS = 2200           # characters of one listing's facts shown to a judge
ROW_CHARS = 260             # characters of a search row's text shown to a screener
SOFT_SPEND = 0.03           # model spend (USD) of one task after which the Sonnet roles are answered by the fast model
SPEND_CAP = 0.05            # model spend after which no more exploring: the task finishes with what it knows (the
                            # run's key pays every task, about 5.5 cents each)
LINE_RECORDS = True         # dirR: listings sent as text lines ("name: value | ...", "a=b; c=d") are read as records
SINGLE_ASK = True           # dirR: one property of one product per message, a plain product noun, no "x or y" examples
REPLY_ALL = True            # dirR: the reply to a demanded step's or the basket's message is read for needs too
PAIR_HUNT = True            # dirR: a counterpart reference whose page states no pairing: the store's pairing is looked for
REDIRECT_CHECK = True       # dirR: a note that sends a differently named property elsewhere is checked, not set aside
PAD_ONE = True              # dirR: one listing to read in full goes to the side-by-side tool with a second (one step, not two)
PRICE_SHAPE = True          # dirR: a price sent as an object (an amount beside its unit) is read as its number, never as a table
GUARD = True                # dirR: an error inside a dirR lever turns it off for the episode; the base path runs
PAIR_READS = 2              # dirR: most search-and-read rounds one pairing hunt spends
_CODE = re.compile(r"\b[A-Za-z]{2,}-?\d{3,}[A-Za-z]?\b|\b[A-Za-z]{2,}\d*-\d{2,}\b")   # a model or part code
_SERIES = re.compile(r"\b([A-Z]{1,5}\d{1,3})-\d{2,}\b")   # the series head of a part code ("AB3" in "AB3-1234")
# a listing's text sending the reader to another part's page ("details on part QX-210", "see model 4471B")
_POINTER = re.compile(r"\b(?:more|details?|info(?:rmation)?|see|read|look)\b[^.;:!?]{0,24}?\b(?:part|model|item|listing|"
                      r"product|page|code)\s*(?:no\.?|number|#)?\s*([A-Za-z0-9][A-Za-z0-9-]{2,})", re.I)
# a requirement read off another listing's page, and the plain op each relation becomes once that page is read
REF_OPS = {"same as reference": "is", "different from reference": "is not", "counterpart of reference": "is",
           "at least reference": "at least", "at most reference": "at most", "within reference": "at least"}

_llm = ProxyClient(timeout=75, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=90, max_retries=2, retry_delay=1)

# details never asked about: a store's variant-selector field ("options") mixes several choices in one value, and
# shoppers answer a question on it as one they could not follow
_NOT_ASKED = {"option", "variant"}

# learned per world (the sandbox may run several tasks in one process): key -> mapping
_WORLD_MEMORY: dict[str, Any] = {}


# ----------------------------------------------------------------------------- helpers

def _norm(text: Any) -> str:
    text = str(text or "").lower().replace("&", " and ").replace("_", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def _tokens(text: Any) -> set[str]:
    out = set()
    for t in _norm(text).split():
        if len(t) > 1:
            if len(t) > 3 and t.endswith("s"):
                t = t[:-1]
            out.add(t)
    return out


def _words(text: Any) -> set[str]:
    """Like _tokens, keeping one-character words (a "3" of "3 pieces")."""
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in _norm(text).split()}


def _near(a: Any, b: Any) -> bool:
    """Two property names that are the same property in another spelling (colour / colours, max_speed / max speed)."""
    if _norm(a) == _norm(b):
        return True
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta) and ta == tb


def _similar(a: Any, b: Any) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _json_in(text: str) -> Any:
    """The first JSON object or array in a model reply, or None."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        in_str = False
        esc = False
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
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    try:
        return json.loads(text)
    except Exception:
        return None


def _compact(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "..."


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
    return None


_DROP = object()


def _fit_value(spec: dict[str, Any], value: Any) -> Any:
    """A value made to agree with its argument's schema where its meaning stays: numbers and text converted, a single
    value listed, limits and lengths kept, the schema's own spelling of a listed value. _DROP when it cannot agree."""
    kind = spec.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    enum = spec.get("enum")
    if isinstance(enum, list) and enum:
        if value in enum:
            return value
        by = {_norm(e): e for e in enum if isinstance(e, str)}
        if isinstance(value, str) and by:
            best = max(by, key=lambda e: _similar(e, value))
            if _norm(value) in by:
                return by[_norm(value)]
            if _similar(best, value) >= 0.5:
                return by[best]
        return _DROP
    if kind in ("integer", "number"):
        n = _number(value)
        if n is None:
            return _DROP
        lo, hi = spec.get("minimum"), spec.get("maximum")
        n = max(n, lo) if isinstance(lo, (int, float)) else n
        n = min(n, hi) if isinstance(hi, (int, float)) else n
        return int(round(n)) if kind == "integer" else n
    if kind == "string":
        value = value if isinstance(value, str) else str(value[0]) if isinstance(value, list) and value else str(value)
        limit = spec.get("maxLength")
        if isinstance(limit, int) and len(value) > limit:
            value = value[:limit].rsplit(" ", 1)[0] if " " in value[:limit] else value[:limit]
        return _DROP if isinstance(spec.get("minLength"), int) and len(value) < spec["minLength"] else value
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        return {"true": True, "false": False}.get(str(value).strip().lower(), _DROP)
    if kind == "array":
        value = value if isinstance(value, list) else [value]
        return value[:spec["maxItems"]] if isinstance(spec.get("maxItems"), int) else value
    return value


def _arg_default(spec: dict[str, Any]) -> Any:
    """The plainest value an argument's schema allows, for a required one nothing else can fill."""
    kind = spec.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    if isinstance(spec.get("enum"), list) and spec["enum"]:
        return spec["enum"][0]
    if kind in ("integer", "number"):
        return _fit_value(spec, 1)
    if kind == "boolean":
        return False
    if kind == "string":
        return _fit_value(spec, "n/a")
    return [] if kind == "array" and not spec.get("minItems") else _DROP


def _refused_cart(obs: Any) -> bool:
    """A cart tool's answer that is only a line of text saying why not (a success reports a flag or the lines)."""
    if not isinstance(obs, dict) or len(obs) != 1:
        return False
    only = next(iter(obs.values()))
    return isinstance(only, str) and not re.search(r"\b(ok|okay|success\w*|added|done|accepted|applied|updated|in (the |your )?(cart|basket))\b", only, re.I)


def _ref_sig(ref: Any) -> tuple:
    """What identifies a referenced page: its seller, its category and the attribute values named for it."""
    if not isinstance(ref, dict):
        return ()
    attrs = ref.get("attributes") if isinstance(ref.get("attributes"), dict) else {}
    return (_norm(ref.get("seller")), _norm(ref.get("category")), tuple(sorted(_norm(v) for v in attrs.values() if v)))


def _is_sentence(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 24 and " " in value.strip()


def _find_list_of_dicts(obs: Any, depth: int = 0) -> list[dict[str, Any]] | None:
    if isinstance(obs, list) and obs and all(isinstance(r, dict) for r in obs):
        return obs
    if isinstance(obs, dict) and depth < 2:
        best = None
        for v in obs.values():
            found = _find_list_of_dicts(v, depth + 1)
            if found and (best is None or len(found) > len(best)):
                best = found
        return best
    return None


def _unwrap(obs: Any) -> tuple[Any, list[str]]:
    """Split a tool result into the result proper and any shopper sentences riding on it."""
    lines: list[str] = []
    if isinstance(obs, str):
        return obs, lines
    if not isinstance(obs, dict):
        return obs, lines
    inner: dict[str, Any] = {}
    dict_siblings = sum(1 for v in obs.values() if isinstance(v, dict))
    # a record's own fields (an id beside its title and property table) are no wrapper around a result
    record = any(isinstance(v, str) and v.strip() and " " not in v.strip() for v in obs.values())
    for k, v in obs.items():
        if isinstance(v, list) and v and all(_is_sentence(x) for x in v):
            lines.extend(str(x) for x in v)
        elif _is_sentence(v) and dict_siblings == 1 and len(obs) <= 3 and not record:
            # a shopper line riding beside the result proper ({"note": "...", "result": {...}})
            lines.append(str(v))
        else:
            inner[k] = v
    if lines and len(inner) == 1 and isinstance(list(inner.values())[0], dict):
        inner = list(inner.values())[0]
    return inner, lines


# ----------------------------------------------------------------------------- models

def _parallel(jobs: list, workers: int = 3) -> list:
    """Run independent model calls concurrently; results in the jobs' order (None for a job that raised)."""
    if len(jobs) <= 1:
        return [_safe(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(_safe, jobs))


def _safe(job: Any) -> Any:
    try:
        return job()
    except Exception:
        return None


# ----------------------------------------------------------------------------- listings sent as text (dirR)

_FIELD_START = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_ .\-]{0,38}?)\s*:\s+(.*)$", re.S)
_ID_TEXT = re.compile(r"^[A-Za-z]{0,4}[-_]?[0-9a-fA-F]{6,}$|^[A-Za-z]{1,5}-?\d{3,}[A-Za-z]?$")


def _plain_value(text: str) -> Any:
    """A text field's value in its own type: true / false, none, a whole or decimal number; other text as it is."""
    bare = text.strip()
    low = bare.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    if re.fullmatch(r"-?\d+", bare):
        return int(bare)
    if re.fullmatch(r"-?\d+\.\d+", bare):
        return float(bare)
    return bare


def _pairs_text(text: str, least: int = 2) -> dict[str, Any] | None:
    """A property table written as one text ("a=b; c=d"); None unless the text is wholly such pairs (a part without "="
    continues the value before it: a value may hold a semicolon of its own)."""
    out: dict[str, Any] = {}
    last = ""
    for part in [p for p in re.split(r";\s+|;$|\n", text.strip()) if p.strip()]:
        name, eq, value = part.partition("=")
        if eq and 0 < len(name.strip()) <= 40 and ": " not in name and name.strip() not in out:
            last = name.strip()
            out[last] = value.strip()
        elif last:
            out[last] = f"{out[last]}; {part.strip()}"
        else:
            return None
    return out if len(out) >= least else None


def _text_record(text: Any) -> dict[str, Any] | None:
    """A listing sent as one text of "name: value" fields (split by " | " or by lines), or a JSON object sent as text,
    as a record; None for any other text (a shopper's sentence, a description, a note). A part that does not open with a
    field name belongs to the field before it (a title may hold the separator itself), and a field written as "a=b; c=d"
    becomes a nested table. A record has three named fields at least and an identifier among its values."""
    if not isinstance(text, str) or len(text) < 12:
        return None
    bare = text.strip()
    if bare[:1] == "{" and bare[-1:] == "}":
        try:
            parsed = json.loads(bare)
        except Exception:
            parsed = None
        return parsed if isinstance(parsed, dict) else None
    if ": " not in bare:
        return None
    fields: list[list[str]] = []
    for part in re.split(r"\s\|\s|\n", bare):
        m = _FIELD_START.match(part)
        if m and len(m.group(1).split()) <= 4:
            fields.append([m.group(1).strip(), m.group(2).strip()])
        elif fields:
            fields[-1][1] = f"{fields[-1][1]} | {part.strip()}"
        else:
            return None
    if len(fields) < 3 or not any(_ID_TEXT.match(v) for _, v in fields):
        return None
    out: dict[str, Any] = {}
    for name, value in fields:
        if name not in out:
            table = _pairs_text(value, 1) if "=" in value else None
            out[name] = table if table else _plain_value(value)
    return out


def _amount_of(value: Any) -> float | None:
    """A price sent as an object (an amount beside its unit or currency): its one number; None when it holds none or
    several."""
    if not isinstance(value, dict):
        return None
    nums = [n for n in (_number(v) for v in value.values() if not isinstance(v, (dict, list, bool))) if n is not None]
    return nums[0] if len(nums) == 1 else None


def _restore_rows(value: Any, depth: int = 0) -> Any:
    """A store result with its listings sent as text given back as records: text records in a list or under a key
    become objects, a record's "a=b; c=d" field becomes its table, and a result that is one wrapped text record becomes
    that record. Everything else stays as it came (a shopper's sentence stays a sentence)."""
    if depth > 6:
        return value
    if isinstance(value, str):
        found = _text_record(value)
        return found if found is not None else value
    if isinstance(value, list):
        return [_restore_rows(v, depth + 1) for v in value]
    if not isinstance(value, dict):
        return value
    # a record (an identifier among its values) with no table of its own may carry it as one "a=b; c=d" text
    record = (any(isinstance(v, str) and _ID_TEXT.match(v) for v in value.values())
              and not any(isinstance(v, dict) for v in value.values()))
    out: dict[str, Any] = {}
    for k, v in value.items():
        if record and isinstance(v, str) and "=" in v and ";" in v:
            out[k] = _pairs_text(v) or v
        else:
            out[k] = _restore_rows(v, depth + 1)
    if len(out) == 1 and isinstance(next(iter(value.values())), str) and isinstance(next(iter(out.values())), dict):
        return next(iter(out.values()))
    return out


class LLM:
    def __init__(self, episode: "Episode"):
        self.ep = episode
        self.cost = 0.0
        self.calls = 0
        self.failures = 0

    def chat(self, model: str, system: str, user: str, max_tokens: int) -> str | None:
        if self.cost >= SOFT_SPEND and model in (PLAN_MODEL, VERIFY_MODEL, PARSE_MODEL) and model != FAST_MODEL:
            model = FAST_MODEL
        left = self.ep.time_left()
        if left < WRAP_SECONDS - 12:
            return None
        _llm.timeout = int(max(10, min(45, left - WRAP_SECONDS + 20)))
        if model.startswith("anthropic/"):
            sys_content: Any = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        else:
            sys_content = system
        body = {
            "model": model,
            "messages": [{"role": "system", "content": sys_content}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        self.calls += 1
        result = _llm.post_verbose("/inference/chat/completions", body)
        for attempt in range(2):
            # a refused call (rate limit, gateway trouble) is tried again after a short pause, never a timed-out one
            error = result.error if isinstance(result.error, dict) else {}
            if isinstance(result.data, dict) and result.data.get("choices"):
                break
            if error.get("status") not in (429, 500, 502, 503, 504) or self.ep.time_left() < WRAP_SECONDS + 10:
                break
            time.sleep(2.0 * (attempt + 1))
            result = _llm.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
            message = data["choices"][0].get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
            if content:
                return str(content)
        self.failures += 1
        self.ep.note("llm_failure", model=model, error=str(getattr(result, "error", ""))[:200])
        return None

    def json(self, models: list[str], system: str, user: str, max_tokens: int) -> Any:
        for model in models:
            text = self.chat(model, system, user, max_tokens)
            parsed = _json_in(text or "")
            if parsed is not None:
                return parsed
        return None


# ----------------------------------------------------------------------------- prompts

READ_PROMPT = """You prepare a shopping assistant for one task in a simulated store. You get the shopper's request and the store's tool list (OpenAI function schemas). Return ONE JSON object and nothing else.

{
 "items": [  // one entry per different product the shopper wants to buy; usually one
   {"name": "plain noun phrase for the product, as the request names it (for a list, the item's own words)",
    "category": "the store category or kind of product requested, in the request's words",
    "quantity": 1,
    "requirements": [ // every firm requirement on THIS item's properties that the request states; not price, stock, quantity or budget
      {"property": "property name as a listing would name it", "op": "is|is not|at least|at most|one of|contains|same as reference|different from reference|counterpart of reference|at least reference|at most reference|within reference",
       "value": "the value, limit or list exactly as the request gives it (for 'one of': values separated by ' | ')",
       "reference": null  // for the ops ending in 'reference': {"category": "...", "seller": "...", "attributes": {"property": "value", ...}, "property": "the property to read on that page", "count": null or how many listings the description says fit it, "pick": null or the rule for choosing one of several (for example "whichever is wider"), "relation": "the request's own words for how the wanted value relates to that page's value", "tolerance": null or, for 'within reference', the amount with its unit}
      }],
    "price_cap": null,  // the most the request lets ONE unit of this item cost (a cap on this item's own price, not the total), as a plain number in the request's money unit; else null
    "hints": ["properties the story hints the shopper privately cares about, each as a property name (for example a remark about its intended wearer hints at a gender or age group; a remark about what sits inside a computer hints at the processor); empty when none"],
    "search_terms": ["2-4 keyword queries for the store's search box, most specific first. Every word of a query must match, so keep each query short: the product noun plus one or two value words (a colour, a material, a value with its unit such as 5v or 16gb), never property names on their own, never values that only identify a referenced listing; the last query is the product noun alone"]
   }],
 "total_cap": 12345 or null,        // the maximum total price as a plain number in the unit the request uses (cents when it says cents)
 "cheapest": false,                 // true only when the request asks for the cheapest / lowest price / cheapest total among what fits (a rule for ordering near misses when nothing fits does not count)
 "best": null,                      // when the request asks for the best by some measure, say which measure; else null
 "message_budget": null,            // how many messages the shopper allows, as an integer, or null
 "call_budget": null,               // how many tool calls / steps the request allows, as an integer, or null
 "in_stock_only": true,             // the request wants only what is actually available
 "ask_about": ["subjects the request tells the assistant to raise with the shopper, in the request's words; empty when none"],
 "fit_rules": [{"accessory": "item name", "host": "item name"}],   // 'make sure X fits Y' style rules; empty when none
 "conditions": [{"if": {"item": "item name", "property": "...", "values": ["..."]}, "then": {"item": "item name", "property": "...", "op": "is|is not|at least|at most|one of", "value": "..."} or null, "otherwise": the same shape or null}],   // a requirement on one item that depends on what another item turns out to be; empty when none
 "decline_rule": null,              // what the request says to do when nothing fits (e.g. just say so, no substitutes), or null
 "nothing_fits": null,              // "nearest" when, for the case that nothing meets every requirement, the request asks to buy, put forward or recommend the listing that comes closest; "decline" when it says to end without buying then; null when it does not say
 "procedure": [                     // steps the request demands in HOW the task is carried out (not product requirements, budgets, message or step limits, and never the rule for the case that nothing fits, such as saying so or putting forward a near miss: that is decline_rule / nothing_fits): an approval or confirmation to get, something to show or tell the shopper, a tool to use, an order of steps; empty when none
   {"moment": "start|before_order|before_decline|before_finish", "kind": "message|tool", "tool": "tool name or null", "goal": "what must happen, in the request's terms", "messages": 1}  // messages: how many messages to the shopper the step takes (0 for a tool); a check the assistant makes in the store itself (stock, availability, a value) is kind "tool" with 0 messages unless the request asks to be told the result
 ],
 "tools": {
   "search_text": {"name": "tool", "query": "argument for the keywords", "page": "argument that moves to the next results (a page number, a row offset, or a token from the previous result), or null",
                   "page_kind": "number|offset|token or null", "mode": {"arg": "argument that selects keyword search", "value": "its value"} or null},
   "search_fields": {"name": "tool for searching by category/brand/price fields, or the same tool in its fields mode", "category": "arg or null", "brand": "arg or null", "price_max": "arg or null", "page": "arg or null", "page_kind": "number|offset|token or null", "mode": {"arg": "...", "value": "..."} or null} or null,
   "open": {"name": "tool that returns one listing in full", "id": "its id argument"},
   "facts": {"name": "separate tool that returns a listing's property table, if the store has one", "id": "arg"} or null,
   "compare": {"name": "tool returning several listings in full", "ids": "its list argument"} or null,
   "cart_add": {"name": "tool that puts a listing in the cart", "id": "arg", "qty": "quantity arg or null",
                "mode": {"arg": "argument choosing add vs remove", "add": "value that adds", "remove": "value that removes"} or null},
   "cart_set": {"name": "tool that replaces the whole cart with a list of lines, when the store has one", "lines": "its list argument", "id": "the key for the listing id inside one line", "qty": "the key for the quantity inside one line"} or null,
   "cart_view": {"name": "tool that shows the cart"} or null,
   "checkout": {"name": "tool that places the order (final)", "args": {"argument": "value"} or null},
   "decline": {"name": "tool that ends without buying (final)", "remark": "text argument or null", "args": {"argument": "value"} or null},
   "message": {"name": "tool that sends a message to the shopper", "text": "its text argument"},
   "status": {"name": "tool that reports remaining allowances"} or null
 }
}

Rules for reading the request:
- Only what the request firmly requires is a requirement. Wishes (ideally, would be nice) and story details (its intended user, where it will live) are NOT requirements; put what they hint at in "hints". Words that only name the kind of product or its category (men's, women's, kids' in "men's rain capes"; "gaming" in a category name) belong in "name"/"category", never in "requirements". But a value the request gives for a named property ("in the <value> style", "with <value> as its <property>", "a <property> of <value>") IS a requirement on that property, even when the value sounds like a kind of product; the category then stays the product the request names.
- A list like "3 x gadgets where the colour is red and the size is not large" means one item with quantity 3 and two requirements (colour is red; size is not large).
- A size the request gives for what one listing holds (a pack, set or box of N pieces, a pack holding N) is a requirement on that pack or set size, while "quantity" counts the listings to buy.
- A restatement that only spells out what a requirement already implies ("zip front only, so nothing with buttons or toggles") is ONE requirement (closure: zip front), not several; never invent properties for the things it rules out.
- "A or B" for a property is op "one of" with value "A | B". "not X" / "anything but X" is op "is not". "at least N units" is "at least" with value "N units". A value should keep its unit and spelling as written.
- A requirement that points at ANOTHER listing's page (one named by its category, its seller and some attribute values) and wants the same value as that page states is op "same as reference"; one that wants anything other than that page's value is op "different from reference"; one that wants a value that goes with that page's value rather than the value itself (its counterpart: what the page or the store pairs with it, what suits, works or is compatible with it) is op "counterpart of reference"; one that wants a value no lower (or no higher) than that page's is op "at least reference" (or "at most reference"); one that allows some amount above or below that page's figure is op "within reference" with that amount in "tolerance". The page named may only point on to another listing for that property: the assistant follows such pointers itself, so describe the page the request names. Fill "reference" with the category, the seller, the attributes named, and the property to read. When the request says several of that seller's listings fit and gives a rule to choose one (the larger, the cheaper, the one with more of something), put the rule in "pick" and the number in "count"; "attributes" keep only plain identifying values, never the comparison. The attribute values that identify that other listing describe THAT listing only: they are never requirements on the requested item and never search terms for it. Invented example: "for the colour, avoid whatever Fenwick's garden hose page (length 20 m, material rubber) states" gives one requirement {"property": "colour", "op": "different from reference", "value": "", "reference": {"category": "garden hoses", "seller": "Fenwick", "attributes": {"length": "20 m", "material": "rubber"}, "property": "colour"}} and nothing about length or material.
- A requirement on one item that holds only when another item turns out a certain way ("should X's P turn out to be V, I need Y's Q to be W"; "Y's Q must be W, except when X's P is V") goes into "conditions" (the first: "then" holds when X's P is one of the values; the second: "otherwise" holds when it is not), never into "requirements" or "procedure".
- The category is the kind of product; when the request names a store category (for example "the category is garden tools"), use that exact wording.
- Money: if the request states the total in cents, "total_cap" is that number of cents. Every amount (total_cap, price_cap) is in the one unit the request counts money in: when it counts in cents, an amount written in dollars or in words ("twelve dollars forty") is converted to cents (1240).
- Use the exact tool and argument names from the schemas. When ONE tool ends the task either way, with an argument that chooses between buying the cart and leaving without buying, "checkout" and "decline" both name that tool, each with "args" holding the argument value that selects it (exactly as the schema lists it). A tool whose description says it is final and closes without buying is "decline"; the one that checks out the cart is "checkout". If one search tool has an enum argument that switches between keyword search and field search, "search_text" and "search_fields" both name it with the matching mode values."""

SCREEN_PROMPT = """You screen search results for a shopping assistant. For each row (id, title, short text, category, price) say whether the listing COULD meet the requirements once its full property page is read. Judge only from what the row shows: when the row says nothing about a requirement, the listing still could meet it (answer "maybe"); when the row's own words clearly contradict a requirement's value, answer "no"; when the row itself states the required values, answer "yes". The store's category label decides what kind of product a row is: a row in the requested category is the right kind whatever its title suggests, so never answer "no" for the kind of product. The rows are numbered. Return ONE JSON object and nothing else: {"yes": [row numbers], "no": [row numbers]}; a row in neither list counts as "maybe"."""

JUDGE_PROMPT = """You judge whether store listings meet a shopper's requirements, from each listing's own records. Reading rules:
- A requirement is met when the listing's own property for it states the value. Property keys vary in spelling (spaces, underscores, prefixes, abbreviations); any key that names that property counts, and the same property can appear under several keys. Values match by meaning: one element of a comma-separated list is enough, singular and plural are the same, close synonyms and obvious abbreviations count, case does not matter, a store value made of the required value's own words plus qualifiers meets it ('deep teal' for 'teal', 'waxed cork' for 'cork'), and a value with a unit matches after converting units (compare numbers as numbers; a limit includes its edge; a range meets a limit only when the whole range does).
- The listing's property table decides first: when a property covering the requirement exists, its value decides (a different value fails the requirement even if the title says otherwise). When NO property key covers the requirement, a value stated in the seller's notes, the options or variants field, the title or the description counts. When nothing states it at all, the requirement is unaddressed ("unsure"). A value under a different, related property does not count (a handle is not a lock type; a lining material is not the material).
- "is not" and "different from" requirements fail when the listing's property states the excluded value; "one of" requirements need one of the listed values; "contains" needs the value among those stated. A requirement that some feature be absent (a value of none, no or without) is met when no record mentions that feature.
- The store's category label decides the kind of product: a listing whose label is the requested category is the right kind whatever its title says. Only a listing with no label can fail for being an accessory, a spare part or a case rather than the product.
- When a requirement asks for compatibility with another product (named by its model names or codes), it is met only when this listing's own text lists one of those names among what it fits or works with. When the other product's page names a series or family of compatible part codes, a listing whose own part code (often in brackets in its title) starts with that series meets it. A sentence saying that some OTHER part fits a product applies to the listing whose title carries that part's code, not to the page it is written on; a listing's "related_pages" show such pages, and their statements about this listing's own part code count for it. A pointer to another page alone proves nothing.
- A listing's seller notes may send one of its properties to ANOTHER listing (for that property, go by or take the value of another listing, not this one). Then this listing's own value for that property does not count: a requirement on it is "unsure" unless a related page shown with the listing states it. Notes that send away a property no requirement names change nothing.
- STORE FIELDS, when given, list for each required property the keys and values that this store's read listings of the category use. If some of those listings carry the exact required value in that field, a listing whose field holds a different value fails, even a related or broader one (the required words with qualifiers added are not a different value); a value no read listing carries may still be met by a clear synonym.
- Do not invent requirements: judge only the ones listed.
The requirements are numbered R1, R2, ... Return ONE JSON object and nothing else (no commentary): {"verdicts": [{"id": "...", "failed": ["R numbers of requirements the records clearly fail"], "unsure": ["R numbers no record addresses"], "weak": ["R numbers met only by the title, the description or an options field, not by a property or a seller note"], "loose": ["R numbers met by a synonym or a broader term rather than by the value's own words (for example a value 'female' for a required 'girls'); a unit conversion or a series that covers a code is not loose"]}, ...]} covering every listing; leave out a list that would be empty. A listing with no "failed" and no "unsure" meets everything shown."""

PLAN_PROMPT = """You plan the questions a shopping assistant will send to a shopper. The shopper has hidden requirements: each is a value of ONE property of one of the requested products, and the shopper reveals it only when a message asks about that very property (by its name or an obvious synonym). A message is answered only about the FIRST property it names; asking about anything the shopper does not care about just gets "no preference", and the message is spent.
You get the request, and for each requested item: its name, the requirements already known, hints from the story, and a property menu: the properties its candidate listings actually carry (with how many listings carry each and example values).
Choose exactly QUESTION_COUNT questions, best first, as JSON: {"questions": [{"item": "item name exactly as given", "property": "property name exactly as the menu spells it (words with spaces)"}, ...]}.
How to choose:
1. First, properties the story or the hints point to (what the request says the shopper will tell when asked: for example who will use or wear it points to an age group or a gender; "how it holds on" points to a closure; what is inside points to a processor; "how it looks" points to colour, pattern or style). Map each hint to the closest property name in the menu.
2. Then the properties a buyer of that kind of product most often has an opinion about, preferring those that many candidate listings carry (the "listings" count) (what it is made of, how it looks, its size or fit, its type or style, its variant such as a flavour, its intended user, its main technical spec), carried by many of its candidate listings, with values that differ between listings. A property with the same value everywhere, a pseudo-property such as an option or variant selector, an identifier, a model number, a warranty or shipping note, is a poor question.
3. Never ask about a property already fixed by a known requirement, nor the price, stock or quantity, nor the same property twice (two spellings of one property count as the same). Spread questions over the items in proportion to how uncertain each is; an item whose hints are strong gets the first question."""

REPLY_PROMPT = """You interpret what a shopper replied to a shopping assistant's question. The assistant asked about one property of one product (at times two). Return ONE JSON object and nothing else:
{"kind": "need|repeat|free|split|unclear|other",
 "property": "the property the shopper is talking about (their own words, or the property asked about when they mean it)",
 "op": "is|is not|at least|at most|one of|same as reference|different from reference|counterpart of reference|at least reference|at most reference|within reference",
 "value": "the value, limit or list the shopper gave (values of a list separated by ' | ')",
 "reference": null or {"category": "...", "seller": "...", "attributes": {"property": "value"}, "property": "the property to read on that page", "count": null or how many listings it says fit, "pick": null or the rule for choosing one of several, "relation": "the shopper's own words for how the wanted value relates to that page's value", "tolerance": null or the amount allowed above or below that page's figure},
 "budget": null or a number when the shopper states a new total they can spend,
 "answer": "" or, when the shopper asked the assistant something, a short plain answer,
 "steps": [] or the steps the shopper now demands in how the task is carried out (an approval to get, something to show or tell them, a tool to use before ordering), each {"moment": "before_order|before_decline|before_finish", "kind": "message|tool", "tool": null, "goal": "what must happen", "messages": 1}}
- "need": the shopper states a value, limit, exclusion or list they insist on (any phrasing: settled, firm, nothing but, only X will do, anything bar X, X or Y and nothing else). Read "anything bar / other than X" as op "is not" with value X. When the shopper points at ANOTHER listing's page (named by category, seller and attribute values) and wants the same value that page states, use op "same as reference" and fill "reference"; when they want anything other than that page's value, op "different from reference"; the value that page pairs with its own (its counterpart, what goes with it), op "counterpart of reference"; a value no lower or no higher than that page's, op "at least reference" or "at most reference".
- "repeat": the shopper restates a requirement already listed under KNOWN NEEDS (same property and value), instead of answering about the property asked.
- "free": they do not mind, leave it open, any will do, never thought about it.
- "split": they want the message broken up, its subjects asked separately, or just its first part answered (the message touched more than one subject).
- "unclear": they did not follow the message, or ask to repeat or simplify it.
- "other": anything else (a greeting, a question back, a change of budget).
The "property" is the property the shopper names in the reply when they name one (their words), which may differ from what was asked; only when they name none is it the property asked about (of two asked, the one the value belongs to).
When the reply states a limit with explicit words such as "at most N", "at least N", "no more than N" or "under N", those words decide the op and the value even if the sentence around them reads oddly (for example "nothing that falls short of at most 34 inches" is op "at most", value "34 inches").
Without such limit words an insisted value is op "is", whatever figure of speech carries it: a refusal to look beyond a value, or to settle for other than it, wants that value itself, a figure included (op "is", value "50 cm")."""

EVENT_PROMPT = """A shopper's line arrived inside a store tool result while a shopping assistant was working. Say what changed. Return ONE JSON object:
{"changes": [{"kind": "budget|drop|add|price|stock|step|other",
  "budget": null or the new maximum total as a number,
  "property": null or the property a dropped or added requirement is about,
  "op": "is|is not|at least|at most|one of", "value": null or the new requirement's value,
  "item": null or which requested product it concerns, "note": "one short sentence"}]}
- "budget": a new amount the shopper can spend (higher or lower); give the number exactly as the line states it, in the line's own unit (cents stay cents: never convert).
- "drop": the shopper no longer requires some property ("let X go", "drop X", "never mind X").
- "add": a new requirement.
- "price": the shopper says the chosen listing's price changed.
- "stock": the shopper says the chosen listing can no longer be bought.
- "step": the shopper demands a step in how the task is carried out (an approval to get, something to show or tell them, a tool to use before ordering); put it in "step": {"moment": "before_order|before_decline|before_finish", "kind": "message|tool", "tool": null, "goal": "what must happen", "messages": 1}."""

REFER_PROMPT = """You read one store listing page for a shopping assistant that follows a reference. The assistant needs the value of a given property as this page gives it. The page may state the value in its property table or in the seller's notes, or the notes may say to take that property from ANOTHER listing (named by category, seller and some attribute values) instead of this one. The notes may also say which value of the property goes together with which (one pair for this page's value, or the store's whole list of pairs).
Return ONE JSON object: {"value": "this page's own value as written, or null", "paired": "the value the page pairs with that value, or null", "wanted": "the value the shopper's RELATION asks for: the paired value when it asks for what goes with, suits or is compatible with the page's value and the page states a pairing; the page's own value when it asks for the same value", "redirect": null or {"category": "null unless the notes name one", "seller": "...", "attributes": {"property": "value"}, "property": "..."}}
When the notes redirect the WANTED property to another listing, return the redirect and no value; a note that sends some other property elsewhere is not a redirect for this one. When the notes say the value outright, return it. Otherwise take the value from the property table under any key that names the property; when no key names it, from the options field, the title or the description when they state it. A sentence that only lists the store's pairs does not say which value is this page's own. "paired" comes only from a pairing the page states: in a list of pairs, the pair holding this page's own value (pairs read both ways); null when the page states none or its own value is unknown."""

QUERY_PROMPT = """You write keyword queries for a store's search box for a shopping assistant. Every word of a query must appear in a listing's title, brand, category or property values (word stems match; tiny words are ignored), and results come in a fixed order with only a few rows per page, so a query has to be specific enough that most of its matches meet the requirements. You get the product, its requirements, the queries already tried, and some listings seen (with whether they met the requirements). Return ONE JSON object and nothing else: {"queries": ["...", ...]} with up to 5 new queries of 1-3 words each, best first: the product's own noun as the store's titles word it plus a required value as the listings would write it, other words the store uses for that product or value, never a query already tried."""

ARGS_PROMPT = """A shopping assistant is calling a store tool, and the call does not satisfy the tool. You get the tool's name and description, its schema, the arguments the call has, what the tool is used for here, and the store's refusal when there was one. Return ONE JSON object and nothing else: {"args": {the complete arguments for this call}}. Keep the arguments the call has unless the refusal says they are wrong; add what is missing with the value that serves the tool's use (a count of 1 unless the arguments show another, the first page of results, the choice that suits that use). Use only argument names and values the schema allows."""

CHOICE_PROMPT = """A store tool ends a shopping task and needs an argument that says how. You get the tool's description, the arguments it still needs (with the values each allows) and the goal. Return ONE JSON object and nothing else: {"<argument name>": "<the allowed value that reaches the goal>", ...}."""

GATE_PROMPT = """You check a shopping assistant's process against the shopper's request at one moment of a task. You get the request, the steps the request demands in how the task is carried out, the moment (what the assistant is about to do), the basket now, the allowances left, the store's tools (the ones that end the task or change the cart are left out: the assistant runs those itself) and every call made so far with its result. Decide which of the demanded steps are due NOW, before the assistant goes on, and not yet done for the current state: an approval or confirmation counts only for the basket exactly as it is now (one given for an earlier basket does not count); a step already done for this state is not due again; a step for another moment is not due. Return ONE JSON object and nothing else: {"actions": [{"tool": "exact tool name", "args": {"argument": "value", ...}, "why": "the demanded step it carries out"}]} in order, at most 2, or {"actions": []} when nothing is due. A message to the shopper must stand on its own: for a basket, name every line (quantity, title, price) and the total, and say plainly what you ask for. Never put a question on product properties here, and never spend the last message on anything but a demanded step. A check the assistant makes in the store (stock, availability, a value) is done with the store's reading tools, never with a message, unless the request asks to be told; the assistant already reads every listing it buys, so such a check is due only when the basket's listings have not been read."""

RANK_PROMPT = """A shopper wants, among listings that all meet the requirements, the one that is best by a given measure (a value to be as high or as low as possible). Read each listing's value for that measure from its properties, title or description, converting units where they differ so the values compare. Return ONE JSON object and nothing else: {"order": ["id", ...], "values": {"id": "the value as read"}} with every listing id, best first; listings whose value cannot be read come last, and equal values keep the cheaper first."""

AUDIT_PROMPT = """You review a shopping assistant's search for one product before it gives up on it. You get the product, its requirements, the budget, the queries already tried and every listing seen: READ ones (their properties, notes, and the requirements the assistant judged failed or unaddressed) and rows seen only in search results. Return ONE JSON object and nothing else: {"fits_after_all": ["ids of READ listings that do meet every requirement after all, judged from their own records: a property value decides, a different value fails, a property no record states is not met"], "unopened": ["ids of rows not read yet that most likely meet the requirements, best first, at most 4"], "queries": ["at most 2 new keyword queries of 1-3 words likely to list fitting listings, never one already tried"]}. Any list may be empty."""

COND_PROMPT = """A shopping assistant checks one listing against a condition the shopper set. Say whether the listing's value of the given property is one of the given values (same meaning; units converted), judged from its own records. Return ONE JSON object and nothing else: {"holds": true or false, "value": "the listing's value as written, or null"}; when no record states that property, "holds" is false and "value" null."""

MATCH_PROMPT = """A shopping assistant looks for one particular listing described by a category, a seller and some attribute values. Below are candidate listings with their records. Return ONE JSON object and nothing else (no explanation): {"id": "the id of the listing that matches the description, or null if none does"}. The seller has to match, its category label has to name the described kind, and each described attribute has to agree with that listing's records (same meaning; units converted). When a listing's property table is not shown, its seller notes, title and description are its records: a listing whose notes state the described values matches. A listing whose records leave one of the described values unstated, or state another value, does not match: return null rather than the closest listing (the assistant then looks further). When a PICK rule is given, several listings may match: apply the rule to those that match and return the one it chooses."""

VERIFY_PROMPT = """You are the final checker before a shopping assistant orders. Judge the listing below against EVERY requirement listed, from its own records, with these rules: a requirement is met when the listing's own property for it states the value (any key naming the property counts; comma lists, synonyms, abbreviations, plural/singular and unit conversions count); when no property key covers the requirement, the seller's notes, the options field, the title or the description may state it; a requirement that nothing states is not met; a value under a different property does not count; a compatibility requirement is met when this listing names the other product among what it fits, or when the other product's page names a series of compatible part codes and this listing's own part code starts with it; a property for which the notes point to another listing's value is not met by this listing's own value; exclusions and closed lists must be respected; a listing whose category label is the requested category is the right kind of product whatever its title says; price times quantity must be within the budget (same unit as the prices shown); if the records show it cannot be bought, it fails. Judge only the listed requirements, never add your own.
Return ONE JSON object: {"fits": true or false, "problems": ["each requirement that is clearly not met or that no record addresses"]}."""

LABEL_PROMPT = """A store sorts its listings under category labels. A shopper asked for one kind of product. From the labels seen, name every label that IS that kind of product (the label whose meaning matches the request's wording as a whole, including a wider label that contains it such as "sweaters and cardigans" for a cardigan). Leave out neighbouring kinds (accessories, parts, cases for the product, a narrower or different type). Return ONE JSON object and nothing else: {"labels": ["label exactly as given", ...]} (empty when none matches)."""

ANY_REPLY_PROMPT = """A shopping assistant sent the shopper a message that asked about no product property (a check, a notice, a basket to confirm), and the shopper replied. Say whether the reply states requirements on the requested products. Return ONE JSON object and nothing else:
{"needs": [{"item": "the requested item's name exactly as given", "property": "the property the requirement is about", "op": "is|is not|at least|at most|one of|same as reference|different from reference|counterpart of reference|at least reference|at most reference|within reference", "value": "the value, limit or list (values of a list separated by ' | ')", "reference": null or {"category": "...", "seller": "...", "attributes": {"property": "value"}, "property": "the property to read on that page", "count": null, "pick": null, "relation": "the shopper's own words for how the wanted value relates to that page's value", "tolerance": null or the amount allowed above or below that page's figure}}],
 "budget": null or a number when the shopper states a new total they can spend}
A requirement is a value, limit, exclusion or list the shopper insists on, or a pointer to another listing's page for a property: the same value as that page states ("same as reference"), anything but it ("different from reference"), the value that goes with it ("counterpart of reference"), no lower or no higher than it. A plain yes or no, thanks, a request to rephrase, or a reply that only restates a known need gives {"needs": []}."""

PAIR_PROMPT = """A shopper wants, for one property of a product, the value that a shop pairs with a given value, as its own listings state it. Below are notes from shop listings (with each listing's own value of that property when known). Find a statement that pairs the given value with a value of that property: a listing saying which value its own value is paired or matched with, or a list of value pairs the shop keeps (pairs read both ways). Return ONE JSON object and nothing else: {"paired": "the value the statement pairs with the given value, as written, or null", "quote": "the exact words of the statement, copied from the notes", "id": "the id of the listing whose notes say it"}. A statement about another property, or one that pairs other values, does not count: then "paired" is null."""

SAME_PROMPT = """Say whether two names refer to the same property of a product (other wording, a synonym or a more or less specific name for the same characteristic counts; two different characteristics do not). Return ONE JSON object and nothing else: {"same": true or false}."""

MAP_PROMPT = """A store tool returned the JSON record below. Name which of its keys play these roles, using the exact key names, or null when no key does. Return ONE JSON object:
{"id": "the listing's identifier", "title": "its name", "text": "its description text", "price": "its price", "stock": "whether it can be bought", "category": "its category label", "brand": "its brand", "seller": "its seller or shop", "facts": "the object holding its property table", "notes": "the seller's free-text notes about it", "total": "a cart's total price", "lines": "a cart's list of lines", "next": "the token or marker for the next page of results", "count": "the total number of matching results"}"""


# ----------------------------------------------------------------------------- episode state

class Listing:
    def __init__(self, id_: str):
        self.id = id_
        self.title = ""
        self.text = ""
        self.price: float | None = None
        self.stock: bool | None = None
        self.category = ""
        self.brand = ""
        self.seller = ""
        self.facts: dict[str, Any] = {}
        self.notes = ""
        self.opened = False
        self.page_read = False          # a compare row or the listing's own page was seen (its notes are known)
        self.raw: dict[str, Any] = {}
        self.screen = "maybe"

    def record(self, limit: int = FACT_CHARS) -> dict[str, Any]:
        rec: dict[str, Any] = {"id": self.id, "title": self.title[:160]}
        if self.category:
            rec["category"] = self.category
        if self.brand:
            rec["brand"] = self.brand
        if self.seller:
            rec["seller"] = self.seller
        if self.price is not None:
            rec["price"] = self.price
        if self.stock is not None:
            rec["can_buy"] = self.stock
        if self.facts:
            # one key per property and value: stores repeat a value under several spellings of its key
            props, seen = {}, set()
            for k, v in self.facts.items():
                sig = (_norm(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(k))), json.dumps(v, sort_keys=True, default=str))
                if sig not in seen:
                    seen.add(sig)
                    props[k] = v
            rec["properties"] = props
        if self.notes:
            rec["seller_notes"] = self.notes[:600]
        if self.text:
            rec["description"] = self.text[:500] if not self.facts else self.text[:400]
        text = json.dumps(rec, ensure_ascii=False)
        if len(text) > limit and self.facts:
            props = rec["properties"]
            rec["properties"] = dict(list(props.items())[: max(6, int(len(props) * limit / len(text)))])
        if len(json.dumps(rec, ensure_ascii=False)) > limit and rec.get("description"):
            rec["description"] = rec["description"][:200]
        return rec


class Requirement:
    def __init__(self, prop: str, op: str, value: str, reference: dict | None = None, origin: str = "request"):
        self.prop = str(prop or "").strip()
        self.op = str(op or "is").strip().lower()
        self.value = str(value or "").strip()
        self.reference = reference if isinstance(reference, dict) else None
        self.origin = origin

    def text(self) -> str:
        if self.op in REF_OPS and self.reference:
            return f"{self.prop}: {self.op} (not yet resolved)"
        return f"{self.prop}: {self.op} {self.value}"

    def key(self) -> str:
        return _norm(self.prop)


class Item:
    def __init__(self, spec: dict[str, Any], index: int):
        self.index = index
        self.name = str(spec.get("name") or spec.get("category") or "the product").strip()
        self.category = str(spec.get("category") or self.name).strip()
        qty = spec.get("quantity")
        self.quantity = int(qty) if isinstance(qty, (int, float)) and not isinstance(qty, bool) and qty >= 1 else 1
        self.requirements: list[Requirement] = []
        for r in spec.get("requirements") or []:
            if isinstance(r, dict) and r.get("property"):
                self.requirements.append(Requirement(r.get("property"), r.get("op"), r.get("value"), r.get("reference")))
        self.hints = [str(h) for h in (spec.get("hints") or []) if h]
        self.price_cap = _number(spec.get("price_cap"))     # the most one unit of this item may cost
        if self.price_cap is not None and self.price_cap <= 0:
            self.price_cap = None
        self.terms = [str(t) for t in (spec.get("search_terms") or []) if t]
        if not self.terms:
            self.terms = [self.category]
        self.noun = self.terms[-1]      # the request reader's last query is the product noun alone
        self.pool: dict[str, Listing] = {}
        self.verdicts: dict[str, dict[str, set[str]]] = {}   # listing id -> {"failed": set, "unsure": set}
        self.judged_for: dict[str, set[str]] = {}            # listing id -> requirement texts judged
        self.asked: list[str] = []
        self.freed: list[str] = []
        self.search_done = 0
        self.exhausted = False
        self.chosen: Listing | None = None
        self.category_labels: dict[str, int] = {}
        self.labels_ok: set[str] | None = None
        self.cursors: dict[str, dict[str, Any]] = {}      # search signature -> {"page", "token", "done"}
        self.query_ids: dict[str, set[str]] = {}         # keyword search signature -> the ids its rows showed
        self.value_ids: set[str] = set()                  # ids shown by queries made of the required values
        self.value_complete = False                       # a value query showed every listing it matched

    def query_next(self, query: str) -> None:
        """A query the next keyword search of this item runs (explore takes the queries in order, from about as
        many as were run)."""
        if query and not any(_norm(query) == _norm(t) for t in self.terms):
            self.terms.insert(min(self.search_done, len(self.terms)), query[:60])

    def over_cap(self, listing: Listing) -> bool:
        return self.price_cap is not None and listing.price is not None and listing.price > self.price_cap + 1e-6

    def active(self) -> list[Requirement]:
        return [r for r in self.requirements if not (r.reference and r.op in REF_OPS)]

    def req_texts(self) -> list[str]:
        return [r.text() for r in self.active()]

    def wrapped(self, listing: Listing) -> int:
        """How many plain-value requirements the listing's own field for the property carries only inside a longer
        phrase (no element of the field is the value itself)."""
        n = 0
        for r in self.active():
            if r.op not in ("is", "one of") or not r.value:
                continue
            wants = [frozenset(_tokens(w)) for w in r.value.split("|") if _tokens(w)]
            own = [v for k, v in listing.facts.items()
                   if _near(str(k).replace("-", " "), r.prop) or _similar(str(k).replace("-", " ").replace("_", " "), r.prop) >= 0.5]
            if not wants or not own:
                continue
            elements = {frozenset(_tokens(e)) for v in own for e in re.split(r"[,;/|]", str(v)) if _tokens(e)}
            if not any(w in elements for w in wants):
                n += 1
        return n

    def exactness(self, listing: Listing) -> int:
        """How many of the item's plain-value requirements the listing's own properties state word for word: one
        element of a property's listed values (split at commas, slashes, semicolons) is the required value itself,
        not a longer phrase around it."""
        if not listing.facts:
            return 0
        elements: set[frozenset[str]] = set()
        for v in listing.facts.values():
            for e in re.split(r"[,;/|]", str(v)):
                t = frozenset(_tokens(e))
                if t:
                    elements.add(t)
        score = 0
        for r in self.active():
            if r.op not in ("is", "one of") or not r.value:
                continue
            wants = [frozenset(_tokens(w)) for w in r.value.split("|") if _tokens(w)]
            if any(w in elements for w in wants):
                score += 1
        return score

    def fits(self, include_unsure: bool = False, cheapest: bool = False, strong_only: bool = False) -> list[Listing]:
        sure, unsure = [], []
        for lid, listing in self.pool.items():
            if not listing.opened:
                continue
            v = self.verdicts.get(lid)
            if v is None or v["failed"]:
                continue
            if self.over_cap(listing):
                continue
            if strong_only and v.get("weak"):
                continue
            if self.judged_for.get(lid, set()) >= set(self.req_texts()):
                (unsure if v["unsure"] or self.word_loose(lid) else sure).append(listing)
        key = lambda l: (l.price if l.price is not None else 1e18)
        # cheapest first, but a listing whose own field holds a required value only inside a longer phrase ("x (y)",
        # "x back y") comes after those that state it as written (the order is checked on the values as written; a
        # value the title or description states with no field for it counts as stated)
        if cheapest:
            sure.sort(key=lambda l: (self.wrapped(l) > 0, key(l)))
        else:
            # evidence from the property table outranks evidence from the title or description
            sure.sort(key=lambda l: (-self.exactness(l), len(self.verdicts[l.id].get("weak") or ()), key(l)))
        unsure.sort(key=lambda l: (len(self.verdicts[l.id]["unsure"]) + len(self.word_loose(l.id)), key(l)))
        if sure or not include_unsure:
            return sure
        return unsure

    def word_loose(self, lid: str) -> list[str]:
        """Plain word values (no figure, no code) the listing meets only through another word: a synonym or a broader
        term (one word for another). Such a listing is not a sure fit: one that states the value in its own words comes
        first. Figures, codes and unit conversions are left to the judge."""
        v = self.verdicts.get(lid) or {}
        texts = set(v.get("loose") or ())
        if not texts:
            return []
        listing = self.pool.get(lid)
        return [r.text() for r in self.active() if r.text() in texts and r.op in ("is", "one of")
                and r.value and not re.search(r"\d", r.value) and r.origin != "fit"
                and not (listing is not None and self.shares_words(listing, r))]

    @staticmethod
    def shares_words(listing: Listing, req: Requirement) -> bool:
        """The listing's own value for the property shares a word with the required one (the required word with a
        qualifier, or a part of a longer required value): that states the value in its own words, unlike a different word
        for it."""
        own = [str(v) for k, v in listing.facts.items()
               if _near(str(k), req.prop) or _similar(str(k).replace("_", " "), req.prop) >= 0.5]
        words = set().union(*(_tokens(v) for v in own)) if own else set()
        return any(_tokens(alt) & words for alt in req.value.split("|"))


class Episode:
    def __init__(self, problem_data: dict[str, Any]):
        self.started = time.monotonic()
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.request = str(self.policy.get("query") or "")
        self.opening = self.policy.get("opening")
        self.tools = [t for t in self.policy.get("tools") or [] if isinstance(t, dict)]
        self.schemas: dict[str, dict[str, Any]] = {}
        for t in self.tools:
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            if isinstance(fn, dict) and fn.get("name"):
                self.schemas[fn["name"]] = fn.get("parameters") or {"type": "object", "properties": {}}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.trace: list[dict[str, Any]] = []
        self.llm = LLM(self)
        self.roles: dict[str, Any] = {}
        self.items: list[Item] = []
        self.cap: float | None = None
        self.cheapest = False
        self.best: str | None = None
        self.confirm_first = False          # a message to the shopper is due before the order (from the procedure)
        self.procedure: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []     # the episode so far, as the process gate reads it
        self.gate_calls: dict[str, int] = {}
        self.confirmed_sig: tuple | None = None
        self.message_budget: int | None = None
        self.messages_sent = 0
        self.in_stock_only = True
        self.ask_about: list[str] = []
        self.fit_rules: list[dict[str, Any]] = []
        self.pending_lines: list[str] = []
        self.handled_lines: set[str] = set()
        self.cart: dict[str, int] = {}
        self.cart_item: dict[str, Item] = {}
        self.maps: dict[str, dict[str, str]] = {}
        # side-by-side batches start at 4 ids: the stores seen so far refuse more (with an error that names no
        # limit) whatever their schema allows, and each refusal costs a step
        self.compare_limit = 4
        self.page_size = 0
        self.world_key = "|".join(sorted(self.schemas))
        memo = _WORLD_MEMORY.get(self.world_key) or {}
        self.maps = dict(memo.get("maps") or {})
        self.compare_limit = int(memo.get("compare_limit") or 4)
        # the compare tool's rows came back without a property table while a separate facts tool has one
        self.thin_compare = bool(memo.get("thin_compare"))
        self.opened_pages: set[str] = set()
        self.decline_reason = ""
        self.mode = "full"            # "ask": stop after the questions (an engine test)
        # keyword queries sent -> the match count the store reported; from them, whether the store's search wants
        # every word of a query (a longer query counting fewer matches) or ranks by any word (None: not known)
        self.query_counts: dict[str, int] = {}
        self.all_words: bool | None = memo.get("all_words")
        self.levers_down: set[str] = set()       # dirR levers turned off for this episode after an error
        self.reply_changed = False              # dirR: a reply to a non-question message changed the needs or budget

    # -- bookkeeping
    def time_left(self) -> float:
        return TASK_SECONDS - (time.monotonic() - self.started)

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def note(self, kind: str, **fields: Any) -> None:
        entry = {"kind": kind, "step": self.step, "t": round(time.monotonic() - self.started, 1)}
        entry.update(fields)
        self.trace.append(entry)

    def remember(self) -> None:
        _WORLD_MEMORY[self.world_key] = {"maps": self.maps, "compare_limit": self.compare_limit,
                                         "thin_compare": self.thin_compare, "all_words": self.all_words}

    def spent(self) -> bool:
        """The task's model spend reached the cap: no more exploring (the run's key pays for every task)."""
        return self.llm.cost >= SPEND_CAP

    def lever(self, name: str, fn: Any, *args: Any, fallback: Any = None) -> Any:
        """One dirR lever's work under its guard: an error turns that lever off for the rest of the episode (a
        "lever_error" note) and gives `fallback`, so the path without the lever runs."""
        if name in self.levers_down:
            return fallback
        try:
            return fn(*args)
        except Exception as exc:
            if not GUARD:
                raise
            self.levers_down.add(name)
            self.note("lever_error", lever=name, error=repr(exc)[:200])
            return fallback

    def batch_open(self) -> bool:
        """Several listings are read in full with one step (a compare tool whose rows carry the property table)."""
        return bool(self.roles.get("compare")) and not getattr(self, "compare_off", False) and not self.thin_compare

    # -- environment
    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """One tool call: its arguments made to agree with the tool's schema first, and repaired once when the store
        refuses how the call was made. "bad_args" in the result: the refusal stands."""
        args = self.fit_args(name, args)
        out = self.send(name, args)
        if out["error"] and self.args_refused(out):
            fixed = self.repair_args(name, args, out["obs"])
            if fixed is not None and fixed != args:
                out = self.send(name, fixed)
            out["bad_args"] = bool(out["error"]) and self.args_refused(out)
        return out

    @staticmethod
    def args_refused(out: dict[str, Any]) -> bool:
        return bool(re.search(r"invalid|missing|required|argument|parameter|schema|malformed|validation|unexpected|must be",
                              f"{out['error']} {_compact(out['obs'], 400)}", re.I))

    def arg_uses(self, name: str) -> list[str]:
        return [role for role, spec in self.roles.items() if isinstance(spec, dict) and spec.get("name") == name]

    def fit_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The arguments the tool's schema can take: values converted or cut to fit, an argument under a name the schema
        does not list moved to the required one it lacks (else dropped), required ones still missing filled in."""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        if not props:
            return args
        out: dict[str, Any] = {}
        stray: dict[Any, Any] = {}
        for k, v in args.items():
            if v is None:
                continue
            if k not in props:
                stray[k] = v
                continue
            v = _fit_value(props[k], v)
            if v is not _DROP:
                out[k] = v
        missing = [k for k in schema.get("required") or [] if k not in out]
        for k in list(missing):
            kind = {"string": str, "integer": (int, float), "number": (int, float), "array": list, "boolean": bool}.get((props.get(k) or {}).get("type"), object)
            # a value already of the argument's kind first (a count is no listing id), then the likest name
            for other in sorted(stray, key=lambda x: (not isinstance(stray[x], kind), -_similar(str(x), k))):
                v = _fit_value(props.get(k) or {}, stray[other])
                if v is not _DROP:
                    out[k] = v
                    stray.pop(other)
                    missing.remove(k)
                    break
        if missing:
            out.update(self.fill_args(name, out, missing))
        return out

    def fill_args(self, name: str, have: dict[str, Any], missing: list[str], refusal: Any = None) -> dict[str, Any]:
        """Values for required arguments a call lacks: the schema's own default, else one cheap model call, else the
        plainest value allowed (kept for the task's later calls of the tool). A finishing tool's choice is left to
        fixed_args: only it knows which way of ending is meant."""
        cache = self.__dict__.setdefault("arg_fill", {})
        props = (self.schemas.get(name) or {}).get("properties") or {}
        ending = bool({"checkout", "decline"} & set(self.arg_uses(name)))
        out: dict[str, Any] = {}
        unknown = []
        for k in missing:
            spec = props.get(k) or {}
            if (name, k) in cache:
                out[k] = cache[(name, k)]
            elif "default" in spec:
                out[k] = spec["default"]
            elif not (ending and isinstance(spec.get("enum"), list)):
                unknown.append(k)
        if unknown:
            got = (self.ask_args(name, have, refusal) or {}).get("args")
            got = got if isinstance(got, dict) else {}
            for k in unknown:
                spec = props.get(k) or {}
                v = _fit_value(spec, got[k]) if got.get(k) is not None else _DROP
                v = _arg_default(spec) if v is _DROP else v
                if v is not _DROP:
                    out[k] = cache[(name, k)] = v
            self.note("args_filled", tool=name, args=_compact(out, 200))
        return out

    def ask_args(self, name: str, have: dict[str, Any], refusal: Any) -> dict[str, Any] | None:
        user = (f"TOOL: {name}\nDESCRIPTION: {self.tool_description(name)}\nSCHEMA: {json.dumps(self.schemas.get(name) or {}, ensure_ascii=False)}"
                f"\nARGUMENTS OF THE CALL: {json.dumps(have, ensure_ascii=False, default=str)}\nUSED HERE AS: {', '.join(self.arg_uses(name)) or 'unknown'}"
                + (f"\nTHE STORE'S REFUSAL: {_compact(refusal, 400)}" if refusal is not None else ""))
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], ARGS_PROMPT, user, 400)
        return answer if isinstance(answer, dict) else None

    def repair_args(self, name: str, args: dict[str, Any], refusal: Any) -> dict[str, Any] | None:
        """The call corrected after the store refused its arguments (at most twice per tool in a task). None: no retry."""
        tries = self.__dict__.setdefault("arg_repairs", {})
        if tries.get(name, 0) >= 2 or self.steps_left() < 2:
            return None
        tries[name] = tries.get(name, 0) + 1
        props = (self.schemas.get(name) or {}).get("properties") or {}
        text = _compact(refusal, 400)
        named = [k for k in props if k not in args and re.search(r"(?<![A-Za-z0-9_])" + re.escape(k) + r"(?![A-Za-z0-9_])", text)]
        self.note("args_refused", tool=name, args=_compact(args, 200), refusal=text[:200])
        if named:
            return dict(args, **self.fill_args(name, args, named, refusal))
        answer = (self.ask_args(name, args, refusal) or {}).get("args")
        return self.fit_args(name, answer) if isinstance(answer, dict) and answer else None

    def send(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """One environment step. Returns {"obs", "lines", "error", "ok"}."""
        out: dict[str, Any] = {"obs": None, "lines": [], "error": None, "ok": False}
        if self.finished or self.dead or name not in self.schemas:
            out["error"] = "unavailable"
            return out
        finishing = name in {(self.roles.get(r) or {}).get("name") for r in ("checkout", "decline")}
        if self.steps_left() <= 0 or (self.steps_left() <= 1 and not finishing):
            out["error"] = "no steps left"
            return out
        payload: dict[str, Any] = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(3):
            left = self.time_left()
            if left < 3:
                out["error"] = "no time"
                return out
            _env.timeout = int(max(4, min(90, left - 1)))
            result = _env.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict) and ("observation" in data or "finished" in data):
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                observation = data.get("observation")
                if LINE_RECORDS:
                    observation = self.lever("LINE_RECORDS", _restore_rows, observation, fallback=observation)
                obs, lines = _unwrap(observation)
                for line in lines:
                    if line not in self.handled_lines:
                        self.pending_lines.append(line)
                out["obs"] = obs
                out["lines"] = lines
                if isinstance(obs, dict) and isinstance(obs.get("error"), str) and len(obs) <= 2:
                    out["error"] = obs["error"]
                else:
                    out["ok"] = True
                self.note("call", tool=name, args=_compact(args, 160), result=_compact(obs, 200), lines=lines[:3])
                entry = {"step": self.step, "tool": name, "args": _compact(args, 300 if name == (self.roles.get("message") or {}).get("name") else 120),
                         "result": _compact(obs, 160)}
                if lines:
                    entry["shopper_lines"] = [l[:200] for l in lines[:3]]
                if name == (self.roles.get("message") or {}).get("name"):
                    entry["basket_then"] = [list(x) for x in self.basket_sig()]
                self.history.append(entry)
                return out
            error = result.error or {}
            status = error.get("status") if isinstance(error, dict) else None
            body = str(error.get("body") if isinstance(error, dict) else error)[:300]
            self.note("env_failure", status=status, body=body)
            if status == 402:
                self.dead = True
                out["error"] = "payment"
                return out
            if status == 409:
                if "finished" in body:
                    self.finished = True
                    out["error"] = "finished"
                    return out
                m = re.search(r"expected (\d+)", body)
                if m:
                    self.step = int(m.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                out["error"] = "conflict"
                return out
            if status in (403, 422):
                out["error"] = f"refused {status}"
                return out
        out["error"] = "env failure"
        return out

    # -- mapping of observation keys
    def mapping(self, kind: str, sample: dict[str, Any]) -> dict[str, str]:
        key = kind + ":" + ",".join(sorted(sample.keys()))
        known = self.maps.get(key)
        if known is None:
            for known_key, other in self.maps.items():
                if known_key.startswith(kind + ":") and other:
                    present = sum(1 for k in other.values() if k in sample)
                    if present >= max(2, int(0.6 * len(other))):
                        known = self.maps[key] = other
                        break
        if known is not None:
            # a mapping learned from a record whose text fields were empty misses them: a record that fills a key the
            # mapping does not use is mapped again (once per such key)
            done = self.__dict__.setdefault("remapped", set())
            fresh = [k for k, v in sample.items() if k not in known.values() and (k, key) not in done
                     and ((isinstance(v, str) and len(v) > 25) or (isinstance(v, dict) and v))]
            if not fresh or not any(r not in known for r in ("notes", "text", "facts")):
                return known
            done.update((k, key) for k in fresh)
        shown = {k: (v if not isinstance(v, (dict, list)) else _compact(v, 160)) for k, v in sample.items()}
        for k, v in list(shown.items()):
            if isinstance(v, str) and len(v) > 160:
                shown[k] = v[:160] + "..."
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], MAP_PROMPT, json.dumps(shown, ensure_ascii=False), 700)
        result: dict[str, str] = dict(known or {})
        if isinstance(answer, dict):
            for role, k in answer.items():
                if isinstance(k, str) and k in sample and role not in result:
                    result[role] = k
        # structural fallbacks
        if "facts" not in result:
            for k, v in sample.items():
                if isinstance(v, dict) and v and all(isinstance(x, (str, int, float)) or x is None for x in v.values()) \
                        and not (PRICE_SHAPE and k == result.get("price")):
                    result["facts"] = k
                    break
        if "id" not in result:
            for k, v in sample.items():
                if isinstance(v, str) and re.fullmatch(r"[A-Za-z]{0,3}-?[0-9a-f]{8,}", v):
                    result["id"] = k
                    break
        if "price" not in result:
            for k, v in sample.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and "price" in k.lower():
                    result["price"] = k
                    break
        self.maps[key] = result
        self.remember()
        return result

    def listing_from(self, rec: dict[str, Any], kind: str) -> Listing | None:
        m = self.mapping(kind, rec)
        idk = m.get("id")
        if not idk or not isinstance(rec.get(idk), (str, int)):
            return None
        lid = str(rec[idk])
        listing = Listing(lid)
        listing.raw = rec
        if m.get("title") and isinstance(rec.get(m["title"]), str):
            listing.title = rec[m["title"]]
        if m.get("text") and isinstance(rec.get(m["text"]), str):
            listing.text = rec[m["text"]]
        if m.get("price"):
            listing.price = _number(rec.get(m["price"]))
            if PRICE_SHAPE and listing.price is None and isinstance(rec.get(m["price"]), dict):
                listing.price = self.lever("PRICE_SHAPE", _amount_of, rec[m["price"]], fallback=None)
        if m.get("stock") and isinstance(rec.get(m["stock"]), bool):
            listing.stock = rec[m["stock"]]
        if m.get("category") and isinstance(rec.get(m["category"]), str):
            listing.category = rec[m["category"]]
        if m.get("brand") and isinstance(rec.get(m["brand"]), str):
            listing.brand = rec[m["brand"]]
        if m.get("seller") and isinstance(rec.get(m["seller"]), str):
            listing.seller = rec[m["seller"]]
        if m.get("facts") and isinstance(rec.get(m["facts"]), dict):
            listing.facts = {str(k): v for k, v in rec[m["facts"]].items() if v not in (None, "")}
        if m.get("notes") and isinstance(rec.get(m["notes"]), str):
            listing.notes = rec[m["notes"]]
        if LINE_RECORDS and kind == "detail" and not listing.facts and not self.roles.get("facts"):
            listing.facts = self.lever("LINE_RECORDS", self.loose_facts, rec, m, fallback={}) or {}
        return listing

    @staticmethod
    def loose_facts(rec: dict[str, Any], m: dict[str, Any]) -> dict[str, Any]:
        """A listing record with no property table of its own (its properties written as plain fields beside its id,
        title and price, in a store with no separate table tool): the short plain fields no role takes are its table."""
        used = {k for k in m.values() if isinstance(k, str)}
        out: dict[str, Any] = {}
        for k, v in rec.items():
            if k in used or v in (None, "") or isinstance(v, (bool, dict, list)):
                continue
            if isinstance(v, str) and (len(v) > 80 or _ID_TEXT.match(v)):
                continue
            out[str(k)] = v
        return out if len(out) >= 2 else {}

    def merge(self, item: Item, listing: Listing, opened: bool) -> Listing:
        known = item.pool.get(listing.id)
        if known is None:
            item.pool[listing.id] = listing
            known = listing
        else:
            for attr in ("title", "text", "category", "brand", "seller", "notes"):
                if getattr(listing, attr) and (opened or not getattr(known, attr)):
                    setattr(known, attr, getattr(listing, attr))
            if listing.price is not None:
                known.price = listing.price
            if listing.stock is not None:
                known.stock = listing.stock
            if listing.facts:
                if listing.facts != known.facts:
                    item.judged_for.pop(known.id, None)     # new records: the verdict is judged again
                known.facts = listing.facts
            if listing.notes and listing.notes != known.notes and opened:
                item.judged_for.pop(known.id, None)
            known.page_read = known.page_read or listing.page_read
            known.raw = listing.raw or known.raw
        if opened:
            known.opened = True
            self.opened_pages.add(known.id)
        if known.category:
            item.category_labels[known.category] = item.category_labels.get(known.category, 0) + 1
        return known

    # -- setup: read the request and the tools
    def setup(self) -> bool:
        tool_lines = []
        for t in self.tools:
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            tool_lines.append({"name": fn.get("name"), "description": fn.get("description", ""), "parameters": fn.get("parameters")})
        user = "REQUEST:\n" + self.request
        if self.opening is not None:
            user += "\n\nOPENING OBSERVATION:\n" + _compact(self.opening, 1500)
        user += "\n\nTOOLS:\n" + json.dumps(tool_lines, ensure_ascii=False)
        plan = self.llm.json([PARSE_MODEL, BACKUP_MODEL], READ_PROMPT, user, 2500)
        if not isinstance(plan, dict) or not isinstance(plan.get("tools"), dict):
            plan = self.llm.json([PARSE_MODEL, FAST_MODEL], READ_PROMPT, user, 2500)
        if not isinstance(plan, dict):
            self.note("setup_failed")
            return False
        self.plan = plan
        roles = plan.get("tools") if isinstance(plan.get("tools"), dict) else {}
        self.roles = {}
        for role, spec in roles.items():
            if isinstance(spec, dict) and isinstance(spec.get("name"), str) and spec["name"] in self.schemas:
                self.roles[role] = spec
        self.check_roles()
        for role in ("search_text", "open", "checkout", "decline", "message"):
            if role not in self.roles:
                self.note("role_missing", role=role)
        items = plan.get("items") if isinstance(plan.get("items"), list) else []
        self.items = [Item(s, i) for i, s in enumerate(items) if isinstance(s, dict)]
        if not self.items:
            self.items = [Item({"name": "the product", "category": "", "search_terms": [self.request[:60]]}, 0)]
        for item in self.items:
            leaked: set[tuple[str, str]] = set()
            for r in item.requirements:
                if r.reference and isinstance(r.reference.get("attributes"), dict):
                    for k, v in r.reference["attributes"].items():
                        leaked.add((_norm(k), _norm(v)))
            if leaked:
                item.requirements = [r for r in item.requirements
                                     if r.reference or (_norm(r.prop), _norm(r.value)) not in leaked]
                item.terms = [t for t in item.terms if not any(v and v in _norm(t) for _, v in leaked)] or [item.category]
            # a value made only of category words is no requirement on a property (a word the shopper adds to the
            # name, as in "bermuda shorts" for shorts, is one)
            name_words = _tokens(item.category)
            item.requirements = [r for r in item.requirements
                                 if r.reference or r.op not in ("is",) or not (_tokens(r.value) and _tokens(r.value) <= name_words)]
            # each stated value alone is a query too (every word of a query must match, so fewer words find more)
            for r in item.requirements:
                if r.op in ("is", "one of", "contains") and r.value:
                    for v in r.value.split("|"):
                        v = v.strip()
                        forms = [v] + ([v.replace("-", " ")] if "-" in v else [])
                        for form in forms:
                            if form and len(form) <= 40 and not any(_norm(form) == _norm(t) for t in item.terms):
                                item.terms.append(form)
        self.cap = _number(plan.get("total_cap"))
        self.cheapest = bool(plan.get("cheapest"))
        # the request's rule for the case that nothing meets every requirement: "nearest" orders the closest listing
        self.nearest_ok = str(plan.get("nothing_fits") or "").strip().lower() == "nearest"
        # a request that says what to do when nothing fits (end without buying, say so) states its decline
        self.decline_stated = str(plan.get("nothing_fits") or "").strip().lower() == "decline" or bool(plan.get("decline_rule"))
        self.best = plan.get("best") if isinstance(plan.get("best"), str) and plan.get("best") else None
        mb = plan.get("message_budget")
        self.message_budget = int(mb) if isinstance(mb, (int, float)) and not isinstance(mb, bool) and mb >= 0 else None
        cb = plan.get("call_budget")
        if isinstance(cb, (int, float)) and not isinstance(cb, bool) and 0 < cb < self.max_steps:
            self.max_steps = int(cb)
        self.in_stock_only = bool(plan.get("in_stock_only", True))
        self.ask_about = [str(a) for a in (plan.get("ask_about") or []) if a]
        self.fit_rules = [r for r in (plan.get("fit_rules") or []) if isinstance(r, dict)]
        self.conditions = self.read_conditions(plan.get("conditions"))
        self.procedure = [st for st in (plan.get("procedure") or []) if isinstance(st, dict) and st.get("goal")][:6]
        for st in self.procedure:
            st["moment"] = str(st.get("moment") or "before_finish").lower()
            st["kind"] = "message" if str(st.get("kind") or "").lower() == "message" else "tool"
        self.confirm_first = any(st["kind"] == "message" and st["moment"] in ("before_order", "before_finish")
                                 for st in self.procedure)
        self.note("plan", items=[(i.name, i.quantity, i.req_texts(), i.hints) + ((i.price_cap,) if i.price_cap is not None else ())
                                 for i in self.items], cap=self.cap, cheapest=self.cheapest, nearest=self.nearest_ok, messages=self.message_budget, steps=self.max_steps,
                  roles={k: v.get("name") for k, v in self.roles.items()})
        return True

    def check_roles(self) -> None:
        """The cart and finishing roles made to agree with the tools' own schemas: a cart tool whose input is a list
        of lines sets the whole cart; argument names the schema does not have are replaced by its own; a finishing
        tool that needs a choice argument serves both ways of ending."""
        for role in ("cart_add", "cart_set"):
            spec = self.roles.get(role)
            if not spec:
                continue
            props = (self.schemas.get(spec["name"]) or {}).get("properties") or {}
            lists = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array"
                     and isinstance(v.get("items"), dict) and v["items"].get("type") == "object"]
            if lists:
                line = lists[0]
                line_props = props[line]["items"].get("properties") or {}
                idk = spec.get("id") if role == "cart_set" and spec.get("id") in line_props else \
                    next((k for k, v in line_props.items() if isinstance(v, dict) and v.get("type") == "string"), None)
                qk = spec.get("qty") if role == "cart_set" and spec.get("qty") in line_props else \
                    next((k for k, v in line_props.items() if isinstance(v, dict) and v.get("type") == "integer"), None)
                if idk:
                    self.roles["cart_set"] = {"name": spec["name"], "lines": line, "id": idk, "qty": qk}
                if role == "cart_add":
                    self.roles.pop("cart_add", None)
                continue
            if role == "cart_add":
                if spec.get("id") not in props:
                    plain = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and not v.get("enum")]
                    required = (self.schemas.get(spec["name"]) or {}).get("required") or []
                    spec["id"] = next((k for k in required if k in plain), plain[0] if plain else None)
                if spec.get("qty") and spec["qty"] not in props:
                    spec["qty"] = next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "integer"), None)
                if not spec.get("qty"):
                    # a whole-number argument the reader left out is the quantity, whatever its name says
                    mode_arg = (spec.get("mode") or {}).get("arg") if isinstance(spec.get("mode"), dict) else None
                    required = (self.schemas.get(spec["name"]) or {}).get("required") or []
                    spec["qty"] = next((k for k in list(required) + list(props) if k not in (spec.get("id"), mode_arg)
                                        and isinstance(props.get(k), dict) and props[k].get("type") in ("integer", "number")), None)
                if not spec.get("id"):
                    self.roles.pop("cart_add", None)
        # the other roles' argument names: one the schema lacks gives way to its argument of the kind the role sends
        for role, field, kind in (("message", "text", "string"), ("open", "id", "string"), ("facts", "id", "string"),
                                  ("compare", "ids", "array"), ("search_text", "query", "string")):
            spec = self.roles.get(role)
            if not spec:
                continue
            schema = self.schemas.get(spec["name"]) or {}
            props = schema.get("properties") or {}
            if not props or spec.get(field) in props:
                continue
            mode_arg = (spec.get("mode") or {}).get("arg") if isinstance(spec.get("mode"), dict) else None
            pool = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == kind and not v.get("enum") and k != mode_arg]
            spec[field] = next((k for k in schema.get("required") or [] if k in pool), pool[0] if pool else None)
        # a finishing tool with a required choice: it ends the task either way
        for have, other in (("checkout", "decline"), ("decline", "checkout")):
            spec = self.roles.get(have)
            if not spec or other in self.roles:
                continue
            schema = self.schemas.get(spec["name"]) or {}
            props = schema.get("properties") or {}
            if any(isinstance((props.get(k) or {}).get("enum"), list) for k in schema.get("required") or []):
                self.roles[other] = {"name": spec["name"]}

    def tool_description(self, name: str) -> str:
        for t in self.tools:
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            if isinstance(fn, dict) and fn.get("name") == name:
                return str(fn.get("description") or "")
        return ""

    def fixed_args(self, role: str) -> dict[str, Any]:
        """The arguments that select this way of ending on its tool: the reader's values (kept to the values the
        schema lists), and any required choice still missing picked by one cheap model call."""
        spec = self.roles.get(role) or {}
        schema = self._schema(role)
        props = schema.get("properties") or {}
        out: dict[str, Any] = {}
        for k, v in (spec.get("args") or {}).items() if isinstance(spec.get("args"), dict) else []:
            if k not in props or v in (None, ""):
                continue
            enum = props[k].get("enum")
            if isinstance(enum, list) and enum and v not in enum:
                best = max(enum, key=lambda e: _similar(str(e), str(v)))
                if _similar(str(best), str(v)) < 0.5:
                    continue          # no listed value is meant: the choice call below picks one
                v = best
            out[k] = v
        missing = [k for k in schema.get("required") or [] if k not in out and isinstance((props.get(k) or {}).get("enum"), list)]
        if missing:
            cached = getattr(self, "choice_cache", {}).get(role)
            if cached is None:
                goal = "place the order for what is in the cart" if role == "checkout" else "end the task without buying anything"
                user = (f"TOOL: {spec.get('name')}\nDESCRIPTION: {self.tool_description(spec.get('name') or '')}\n"
                        f"ARGUMENTS: {json.dumps({k: props[k] for k in missing}, ensure_ascii=False)}\nGOAL: {goal}")
                answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], CHOICE_PROMPT, user, 300)
                cached = answer if isinstance(answer, dict) else {}
                if cached:            # a failed call is asked again next time
                    self.choice_cache = dict(getattr(self, "choice_cache", {}), **{role: cached})
            for k in missing:
                enum = props[k]["enum"]
                v = cached.get(k)
                if v in enum:
                    out[k] = v
        return out

    # -- argument helpers
    def _schema(self, role: str) -> dict[str, Any]:
        spec = self.roles.get(role) or {}
        return self.schemas.get(spec.get("name") or "", {})

    def _arg_type(self, role: str, arg: str | None) -> str | None:
        if not arg:
            return None
        props = self._schema(role).get("properties") or {}
        return (props.get(arg) or {}).get("type")

    def _page_mode(self, role: str) -> str | None:
        """How the search tool moves on: "index" (a page number), "offset" (a row offset) or "token"."""
        spec = self.roles.get(role) or {}
        page = spec.get("page")
        if not page:
            return None
        if self._arg_type(role, page) == "integer":
            kind = str(spec.get("page_kind") or "").lower()
            name = page.lower()
            if kind == "offset" or (kind != "number" and ("offset" in name or "start" in name or "skip" in name)):
                return "offset"
            return "index"
        return "token"

    def _enum_value(self, role: str, arg: str, value: Any) -> Any:
        """The world's own spelling of a value its schema lists (an enum), None when no listed value is close."""
        props = self._schema(role).get("properties") or {}
        enum = (props.get(arg) or {}).get("enum")
        if not isinstance(enum, list) or not enum or not isinstance(value, str):
            return value
        by = {_norm(e): e for e in enum if isinstance(e, str)}
        if not by:
            return value
        if _norm(value) in by:
            return by[_norm(value)]
        best = max(by, key=lambda e: _similar(e, value))
        return by[best] if _similar(best, value) >= 0.5 else None

    def _fit_query(self, item: Item, role: str, arg: str, query: str) -> str:
        """The query as sent: value words first, category words after them (values decide the matches, and a
        store may read only a query's first words); over the tool's length limit category words are dropped
        before value words, then the query is cut at a word."""
        words = query.split()
        cat = _tokens(item.category) | _tokens(item.name)
        values = [w for w in words if not (_tokens(w) and _tokens(w) <= cat)]
        rest = [w for w in words if w not in values]
        query = " ".join(values + rest)
        props = self._schema(role).get("properties") or {}
        limit = (props.get(arg) or {}).get("maxLength")
        if not isinstance(limit, int) or len(query) <= limit:
            return query
        kept = list(values)
        for w in rest:                        # category words back in, in order, while they fit
            if len(" ".join(kept + [w])) <= limit:
                kept.append(w)
        return self._trim(role, arg, " ".join(kept) if kept else query)

    def _trim(self, role: str, arg: str, value: str) -> str:
        props = self._schema(role).get("properties") or {}
        limit = (props.get(arg) or {}).get("maxLength")
        if isinstance(limit, int) and len(value) > limit:
            value = value[:limit].rsplit(" ", 1)[0] if " " in value[:limit] else value[:limit]
        return value

    # -- searching
    def search(self, item: Item, query: str | None = None, fields: dict[str, Any] | None = None,
               max_pages: int = 1, budget: int | None = None) -> int:
        """Run a search and page through it. Returns the number of new listings merged into the item's pool."""
        role = "search_text" if query is not None else "search_fields"
        spec = self.roles.get(role)
        if not spec:
            return 0
        name = spec["name"]
        new = 0
        page_mode = self._page_mode(role)
        sig = role + ":" + (query if query is not None else json.dumps(fields or {}, sort_keys=True))
        cursor = item.cursors.get(sig)
        if cursor and cursor.get("done"):
            return 0
        token: Any = cursor.get("token") if cursor else None
        first = (cursor.get("page", 0) + 1) if cursor else 1
        if first > 1 and page_mode is None:
            return 0
        done = False
        page = first - 1
        for page in range(first, first + max_pages):
            if self.steps_left() <= (budget if budget is not None else 0) or self.time_left() < WRAP_SECONDS:
                page -= 1
                break
            args: dict[str, Any] = {}
            if spec.get("mode") and isinstance(spec["mode"], dict) and spec["mode"].get("arg"):
                args[spec["mode"]["arg"]] = spec["mode"].get("value")
            if query is not None:
                if not spec.get("query"):
                    return 0
                args[spec["query"]] = self._fit_query(item, role, spec["query"], query)
            else:
                for key, value in (fields or {}).items():
                    arg = spec.get(key)
                    if arg and value not in (None, ""):
                        value = self._enum_value(role, arg, value)
                        if value is None:
                            # a category the store does not list: this field search cannot run
                            item.cursors[sig] = {"page": 0, "token": None, "done": True}
                            return new
                        args[arg] = self._trim(role, arg, value) if isinstance(value, str) else value
            if page > 1:
                if page_mode == "index":
                    args[spec["page"]] = page
                elif page_mode == "offset":
                    args[spec["page"]] = (page - 1) * max(self.page_size, 1)
                elif page_mode == "token" and token:
                    args[spec["page"]] = token
                else:
                    done = True
                    page -= 1
                    break
                top = ((self._schema(role).get("properties") or {}).get(spec["page"]) or {}).get("maximum")
                if isinstance(args[spec["page"]], int) and isinstance(top, (int, float)) and args[spec["page"]] > top:
                    # the store pages no further: the matches it would not show stay hidden
                    item.capped = True
                    done = True
                    page -= 1
                    break
            res = self.call(name, args)
            if not res["ok"]:
                self.note("search_error", error=res["error"])
                done = True
                break
            obs = res["obs"]
            rows = _find_list_of_dicts(obs) or []
            token = None
            total = None
            if isinstance(obs, dict):
                for k, v in obs.items():
                    if isinstance(v, str) and 0 < len(v) <= 20 and " " not in v and not isinstance(v, bool):
                        token = v
                # the match count is the largest number beside the rows (a next-row offset is smaller, and its key
                # may come first)
                numbers = [v for v in obs.values() if isinstance(v, int) and not isinstance(v, bool)]
                total = max(numbers) if numbers else None
            if rows:
                self.page_size = max(self.page_size, len(rows))
            if query is not None and page == 1 and total is not None:
                self.note_count(args[spec["query"]], total if rows else 0)
            elif query is not None and page == 1 and not rows:
                self.note_count(args[spec["query"]], 0)
            shown = []
            for rec in rows:
                listing = self.listing_from(rec, "row")
                if listing is None:
                    continue
                shown.append(listing)
                if listing.id not in item.pool:
                    new += 1
                self.merge(item, listing, opened=False)
                if query is not None:
                    item.query_ids.setdefault(sig, set()).add(listing.id)
            if len(shown) >= 2:
                # rows that show price and stock leave the page little to add when a property table exists
                self.rows_full = all(l.price is not None and l.stock is not None for l in shown)
            item.search_done += 1
            if item.labels_ok is None and item.index != 99:
                self.resolve_labels(item)
            if not rows or (total is not None and page * max(len(rows), 1) >= total) or len(rows) < self.page_size:
                done = True
                break
            if page_mode == "token" and not token:
                if total is not None and total > page * len(rows):
                    item.capped = True        # more matches than the store would show
                done = True
                break
        item.cursors[sig] = {"page": max(page, 0), "token": token, "done": done or page_mode is None}
        return new

    def note_count(self, query: str, count: int) -> None:
        """Keep a keyword query's match count; two queries where one holds the other's words and more tell how the
        store matches: fewer matches for more words means every word must match."""
        words = frozenset(_words(query))
        if not words:
            return
        for other, n in self.query_counts.items():
            ow = frozenset(_words(other))
            small, big, n_small, n_big = (ow, words, n, count) if len(ow) < len(words) else (words, ow, count, n)
            if len(small) == len(big) or not small & big:
                continue
            was = self.all_words
            if small < big and n_big > n_small:
                self.all_words = False
            elif n_big * 10 < n_small * 3 and (small < big or len(big) >= len(small) + 2):
                # more words, far fewer matches: every word must match (a store matching any word counts more; a
                # small dip is a count cut at the store's limit)
                self.all_words = True
            if self.all_words is not was:
                self.note("search_words", all_words=self.all_words, small=n_small, big=n_big)
        self.query_counts[query] = count

    @staticmethod
    def value_groups(item: Item) -> list[list[str]]:
        """The item's plain required values as keyword groups, one per requirement ('one of' gives its
        alternatives), the shopper's own answers first."""
        out: list[tuple[int, list[str]]] = []
        for r in item.active():
            if r.op not in ("is", "one of", "contains") or not r.value:
                continue
            # a model code ("ABC-1234") is matched by the store's fitment wording ("ABC-12xx models"), never by the
            # code itself: it stays out of keyword queries
            alts = [v.strip() for v in r.value.split("|") if _tokens(v) and not _CODE.search(v)]
            if alts and len(alts) <= 4 and all(len(a) <= 24 for a in alts):
                out.append((0 if r.origin == "shopper" else 1, alts))
        return [g for _, g in sorted(out, key=lambda x: x[0])]

    @staticmethod
    def row_hits(listing: Listing, groups: list[list[str]]) -> int:
        """How many value groups a listing's row (or page) carries in its own words."""
        text = _words(" ".join([listing.title, listing.text or "", listing.category, listing.brand, listing.seller]
                               + [f"{k} {v}" for k, v in listing.facts.items()]))
        return sum(1 for g in groups if any(_words(a) <= text for a in g))

    def value_queries(self, item: Item) -> list[str]:
        """Keyword queries made of the item's required values and its noun: one per alternative of a requirement
        with several (each with the other values), the shopper's answers first. A store that wants every word
        returns just the listings that carry them; one that ranks by any word puts them first. Queries already
        covered by a sent one (its words a subset of the sent query's) are left out."""
        out = self.value_query_list(item)
        sent = [frozenset(_words(k.split(":", 1)[1])) for k in item.cursors if k.startswith("search_text:")]
        sent += [frozenset(_words(q)) for q in getattr(item, "value_sent", ())]
        return [q for q in out if not any(frozenset(_words(q)) <= s for s in sent)]

    def value_query_list(self, item: Item) -> list[str]:
        groups = self.value_groups(item)
        if not groups:
            return []
        noun = item.noun if len(item.noun) <= 22 else item.category
        # a store that orders by any word gives the category's own words to every listing of it: the values go alone
        tail = [] if self.all_words is False else [noun]
        heads = [g[0] for g in groups]
        out: list[str] = []
        for i, g in enumerate(groups):
            for alt in g[:3]:
                words = [alt] + [h for j, h in enumerate(heads) if j != i]
                q = " ".join(words[:3] + tail)
                if frozenset(_words(q)) not in {frozenset(_words(x)) for x in out}:
                    out.append(q)
            if len(out) >= 4:
                break
        return out

    def value_search(self, item: Item, floor: int) -> bool:
        """Run the next value query (and, when it matches nothing, a shorter one); True when a search ran."""
        todo = self.value_queries(item)
        if not todo or self.steps_left() <= floor + 1:
            return False
        q = todo[0]
        first = not getattr(item, "value_sent", None)
        item.value_sent = list(getattr(item, "value_sent", [])) + [q]
        self.note("value_query", item=item.name, query=q)
        before = self.step
        # the item's first query pages as deep as its own queries do (a store showing few rows a page hides matches
        # further on); later ones (a heard value) page less, unless the store wants every word (then all of them)
        pages = (6 if len(self.items) == 1 else 3) if first else (3 if self.all_words else 2)
        self.search(item, query=q, max_pages=pages, budget=floor)
        self.value_seen(item, q)
        sent = self._fit_query(item, "search_text", (self.roles.get("search_text") or {}).get("query") or "", q)
        if self.query_counts.get(sent) == 0 and self.steps_left() > floor + 1:
            # nothing carries every word: the values alone, then the first value with the noun
            words = q.split()
            role_q = (self.roles.get("search_text") or {}).get("query") or ""
            for shorter in (" ".join(w for w in words if not (_tokens(w) and _tokens(w) <= _tokens(item.noun))),
                            " ".join([self.value_groups(item)[0][0], item.noun])):
                # (a fallback the store would receive exactly as an earlier query is skipped)
                if shorter and _norm(shorter) != _norm(q) and self.steps_left() > floor + 1 \
                        and self._fit_query(item, "search_text", role_q, shorter) not in self.query_counts:
                    item.value_sent.append(shorter)
                    self.search(item, query=shorter, max_pages=2, budget=floor)
                    self.value_seen(item, shorter)
                    sent = self._fit_query(item, "search_text", (self.roles.get("search_text") or {}).get("query") or "", shorter)
                    if self.query_counts.get(sent):
                        break
        return self.step > before

    def value_seen(self, item: Item, query: str) -> None:
        """A value query's rows join the item's value ids; a query paged to its end (every match shown) whose words
        are all the item's required values makes the set complete."""
        item.value_ids |= item.query_ids.get("search_text:" + query) or set()
        role_q = (self.roles.get("search_text") or {}).get("query") or ""

        def shown_all(q: str) -> bool:
            sig = "search_text:" + q
            count = self.query_counts.get(self._fit_query(item, "search_text", role_q, q))
            return bool(count) and (item.cursors.get(sig) or {}).get("done", False) \
                and len(item.query_ids.get(sig) or ()) >= count
        # complete only when each query of the item's values (one per alternative) was sent with its every word
        # and showed all its matches
        wanted = self.value_query_list(item)
        if wanted and all(shown_all(q) and _words(self._fit_query(item, "search_text", role_q, q)) >= _words(q) - _words(item.noun)
                          for q in wanted):
            item.value_complete = True

    def category_ok(self, item: Item, listing: Listing) -> bool:
        if not listing.category or not item.category:
            return True
        label = _norm(listing.category)
        if item.labels_ok:
            return label in item.labels_ok
        want = _norm(item.category)
        if label == want or label in want or want in label:
            return True
        return _similar(label, want) >= 0.5

    def resolve_labels(self, item: Item) -> None:
        """Which of the category labels seen so far name the requested kind of product (one cheap model call)."""
        labels = sorted({l.category for l in item.pool.values() if l.category})
        if not labels or item.labels_ok is not None:
            return
        by_norm = {_norm(l): l for l in labels}
        if len(labels) == 1:
            item.labels_ok = set(by_norm) if (_similar(labels[0], item.category) >= 0.3 or _norm(item.category) in _norm(labels[0])
                                              or _norm(labels[0]) in _norm(item.category)) else set()
            if not item.labels_ok:
                item.labels_ok = None      # one foreign label: keep the token rule, more labels may come
            return
        user = (f"REQUESTED PRODUCT: {item.name} (category wording: {item.category})\nREQUEST EXCERPT: {self.request[:600]}\n"
                f"STORE CATEGORY LABELS SEEN: {json.dumps(labels, ensure_ascii=False)}")
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], LABEL_PROMPT, user, 500)
        chosen: set[str] = set()
        if isinstance(answer, dict) and isinstance(answer.get("labels"), list):
            for l in answer["labels"]:
                if isinstance(l, str) and _norm(l) in by_norm:
                    chosen.add(_norm(l))
        if chosen:
            item.labels_ok = chosen
            self.note("labels", item=item.name, labels=sorted(chosen))

    def candidates(self, item: Item) -> list[Listing]:
        cap = self.cap
        out = []
        for listing in item.pool.values():
            if listing.stock is False and self.in_stock_only:
                continue
            if cap is not None and listing.price is not None and listing.price * item.quantity > cap + 1e-6:
                continue
            if not self.category_ok(item, listing):
                continue
            out.append(listing)
        out.sort(key=lambda l: (0 if l.screen == "yes" else (2 if l.screen == "no" else 1),
                                l.price if l.price is not None else 1e18))
        return out

    def screen(self, item: Item, listings: list[Listing]) -> None:
        todo = [l for l in listings if not l.opened and l.screen == "maybe" and not getattr(l, "screened", False)]
        if len(todo) < 8 or not item.req_texts():
            return
        rows = []
        batch = todo[:30]
        for n, l in enumerate(batch, 1):
            rows.append({"n": n, "title": l.title[:140], "text": (l.text or "")[:ROW_CHARS], "category": l.category, "price": l.price})
        user = (f"REQUESTED PRODUCT: {item.category}\nREQUIREMENTS:\n- " + "\n- ".join(item.req_texts())
                + "\n\nROWS:\n" + json.dumps(rows, ensure_ascii=False))
        # row numbers in, row numbers out: a short answer is a quick one
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], SCREEN_PROMPT, user, 1200)
        for l in batch:
            l.screened = True
        if isinstance(answer, dict):
            for verdict in ("no", "yes"):
                for n in answer.get(verdict) or []:
                    if isinstance(n, (int, str)) and str(n).strip().isdigit() and 1 <= int(n) <= len(batch):
                        batch[int(n) - 1].screen = verdict

    # -- opening listings
    def sheet_first(self) -> bool:
        """The store has a property-table tool beside its page tool, no side-by-side tool, and its search rows already
        show each listing's price and stock: a listing is judged from its row and its table (one step), and pages are
        read where a page is what counts (cart lines, chain and fit pages, a stock check)."""
        return (bool(self.roles.get("facts")) and bool(self.roles.get("open")) and bool(getattr(self, "rows_full", False))
                and not (self.roles.get("compare") and not getattr(self, "compare_off", False)))

    def sheet_only(self, item: Item | None) -> bool:
        if item is None or not self.sheet_first():
            return False
        if any(item is a or item is h for a, h in self.fit_pairs()):
            return False                    # a fit is read from the pages of both products
        return not any(r.reference for r in item.requirements)

    def open_batch(self, item: Item | None, ids: list[str], floor: int = 0, page: bool = False,
                   spare: list[str] | None = None) -> list[Listing]:
        """Open listings in full, through the compare tool when it exists. Returns the opened listings. page: the
        listing's own page is read even where the property table alone would do. spare: listings that may be read
        beside a single one (PAD_ONE)."""
        ids = [i for i in ids if i]
        opened: list[Listing] = []
        if not ids:
            return opened
        if PAD_ONE and len(ids) == 1 and not page:
            ids = self.lever("PAD_ONE", self.pad_one, item, ids, spare, fallback=ids)
        compare = self.roles.get("compare") if not getattr(self, "compare_off", False) else None
        facts_tool = self.roles.get("facts")
        open_tool = self.roles.get("open")
        if compare and not getattr(self, "compare_checked", False):
            # a limit the tool's schema states on its list argument
            self.compare_checked = True
            most = ((self._schema("compare").get("properties") or {}).get(compare.get("ids") or "") or {}).get("maxItems")
            if isinstance(most, int) and 2 <= most < self.compare_limit:
                self.compare_limit = most
        pos = 0
        requeued: set[str] = set()
        sheets = not page and self.sheet_only(item)
        while pos < len(ids) and self.steps_left() > 0 and self.time_left() > WRAP_SECONDS:
            if (self.thin_compare and facts_tool and item is not None) or sheets:
                # the compare rows carry no property table in this store: each listing's table is read on its own
                if floor and self.steps_left() - 1 < floor:
                    break
                got = self.read_facts(item, ids[pos])
                pos += 1
                if got is not None:
                    opened.append(got)
                continue
            if compare and len(ids) - pos >= 2:
                if floor and self.steps_left() - 1 < floor:
                    break
                chunk = ids[pos: pos + self.compare_limit]
                res = self.call(compare["name"], {compare["ids"]: chunk})
                if res["ok"] and _find_list_of_dicts(res["obs"]):
                    rows = [l for l in (self.listing_from(rec, "detail") for rec in _find_list_of_dicts(res["obs"]) or [])
                            if l is not None]
                    thin = bool(facts_tool) and bool(rows) and not any(l.facts for l in rows)
                    if thin and not self.thin_compare:
                        self.thin_compare = True
                        self.remember()
                        self.note("thin_compare", tool=compare["name"], facts_tool=facts_tool.get("name"))
                    for listing in rows:
                        listing.page_read = True
                        self.opened_pages.add(listing.id)
                        if thin and item is not None:
                            self.merge(item, listing, False)       # its notes are kept; the table is read below
                            continue
                        listing.opened = True
                        opened.append(self.merge(item, listing, True) if item else listing)
                    if thin and item is not None:
                        for listing in rows:
                            if (floor and self.steps_left() - 1 < floor) or self.time_left() < WRAP_SECONDS:
                                break
                            got = self.read_facts(item, listing.id)
                            if got is not None:
                                opened.append(got)
                    pos += len(chunk)
                    back = {l.id for l in rows}
                    dropped = [i for i in chunk if i not in back and i not in requeued]
                    if rows and dropped and len(back) < len(chunk):
                        # fewer rows than ids sent: the tool answers only so many; the rest go round again
                        if 2 <= len(back) < self.compare_limit:
                            self.compare_limit = len(back)
                            self.remember()
                        requeued.update(dropped)
                        ids = ids[:pos] + dropped + ids[pos:]
                    continue
                if res["error"] or (res["ok"] and not _find_list_of_dicts(res["obs"])):
                    # the tool may take fewer listings than were sent, whatever its error says: fewer, then none
                    if self.compare_limit > 2 and len(chunk) > 2:
                        # a declared limit can be larger than the one the tool keeps: half as many next time
                        self.compare_limit = max(2, min(self.compare_limit, len(chunk)) // 2)
                        self.remember()
                        continue
                    self.compare_off = True
                    compare = None    # fall back to single opens for this episode
                    continue
            if not open_tool:
                break
            cost = 2 if facts_tool else 1
            if floor and self.steps_left() - cost < floor:
                break                       # the steps kept for what comes next are not spent on this
            lid = ids[pos]
            pos += 1
            res = self.call(open_tool["name"], {open_tool["id"]: lid})
            if not res["ok"] or not isinstance(res["obs"], dict):
                continue
            rec = res["obs"]
            listing = self.listing_from(rec, "detail")
            if listing is None:
                listing = Listing(lid)
                listing.raw = rec
            listing.page_read = True
            known = item.pool.get(lid) if item is not None else None
            if not listing.facts and facts_tool and self.steps_left() > 0 and not (known is not None and known.facts):
                listing.facts = self.fetch_facts(lid) or {}
            if listing.id != lid:
                listing.id = lid
            listing.opened = True
            opened.append(self.merge(item, listing, True) if item else listing)
            self.opened_pages.add(lid)
        return opened

    def pad_one(self, item: Item | None, ids: list[str], spare: list[str] | None) -> list[str]:
        """A single listing to read where the side-by-side tool shows whole records and the property table has a tool of
        its own: alone it costs its page and then its table (two steps); beside a second listing one step reads both.
        The second is the first unread one the caller names, else the item's next unread candidate."""
        if not (self.roles.get("compare") and self.roles.get("facts") and self.batch_open()) or self.sheet_only(item):
            return ids
        pool = [i for i in spare or [] if i]
        if item is not None and not pool:
            rows = [l for l in item.pool.values() if not l.opened and l.stock is not False and self.category_ok(item, l)]
            rows.sort(key=lambda l: (l.screen == "no", l.screen != "yes", l.price if l.price is not None else 1e18))
            pool = [l.id for l in rows]
        extra = next((i for i in pool if i not in ids), None)
        if extra:
            self.note("pad_one", id=ids[0], beside=extra)
        return ids + [extra] if extra else ids

    def fetch_facts(self, lid: str) -> dict[str, Any] | None:
        """One listing's property table from the store's facts tool (one step); {} when it shows none, None when the
        call failed."""
        tool = self.roles.get("facts")
        if not tool:
            return None
        res = self.call(tool["name"], {tool["id"]: lid})
        if not res["ok"] or not isinstance(res["obs"], dict):
            return None
        m = self.mapping("facts", res["obs"])
        fk = m.get("facts")
        if fk and isinstance(res["obs"].get(fk), dict):
            return {str(k): v for k, v in res["obs"][fk].items() if v not in (None, "")}
        if all(isinstance(v, (str, int, float)) or v is None for v in res["obs"].values()):
            return {str(k): v for k, v in res["obs"].items() if v not in (None, "")}
        return {}

    def read_facts(self, item: Item, lid: str) -> Listing | None:
        """Open a listing through its property table alone: the row already seen gives its title, price and text."""
        known = item.pool.get(lid)
        facts = self.fetch_facts(lid)
        if facts is None or (known is None and not facts):
            return None
        listing = Listing(lid)
        listing.facts = facts
        return self.merge(item, listing, True)

    # -- judging
    def drop_label_restatements(self, item: Item) -> None:
        """A requirement whose value only repeats the store's own category label for the item ("light" of a "light
        jackets" label), on a property no read listing of the item carries, is the category, not a requirement."""
        if not item.labels_ok:
            return
        label_words = set().union(*(_tokens(l) for l in item.labels_ok))
        read = [l for l in item.pool.values() if l.opened and l.facts and self.category_ok(item, l)]
        if len(read) < 3:
            return
        keep = []
        for r in item.requirements:
            words = _tokens(r.value)
            if (r.origin == "request" and r.op == "is" and words and words <= label_words
                    and not any(_near(k, r.prop) or _similar(str(k).replace("_", " "), r.prop) >= 0.5 for l in read for k in l.facts)):
                self.note("label_restatement", item=item.name, req=r.text())
                continue
            keep.append(r)
        item.requirements = keep

    def align_props(self, item: Item) -> None:
        """A required property that no read listing of the item names, while the read listings carry the required
        value under one other key, takes that key's name: the request's word for the property is not the store's
        (a "finish" the store files as its colour). Plain word values only; the requirement is judged again."""
        read = [l for l in item.pool.values() if l.opened and l.facts and self.category_ok(item, l)]
        if len(read) < 3:
            return
        for r in item.active():
            if r.op not in ("is", "one of", "contains") or not r.value or re.search(r"\d", r.value) or getattr(r, "aligned", False):
                continue
            named = lambda k: _near(k, r.prop) or _similar(str(k).replace("_", " ").replace("-", " "), r.prop) >= 0.5
            if any(named(k) for l in read for k in l.facts):
                continue
            wants = [frozenset(_tokens(w)) for w in r.value.split("|") if _tokens(w)]
            if not wants:
                continue
            counts: dict[str, int] = {}
            for l in read:
                keys = set()
                for k, v in l.facts.items():
                    elems = [frozenset(_tokens(e)) for e in re.split(r"[,;/|]", str(v)) if _tokens(e)]
                    if any(w <= e for w in wants for e in elems):
                        keys.add(_norm(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(k))))
                for k in keys:
                    counts[k] = counts.get(k, 0) + 1
            ranked = sorted(counts.items(), key=lambda x: -x[1])
            if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
                self.note("prop_aligned", item=item.name, was=r.prop, now=ranked[0][0], value=r.value)
                r.aligned = True
                r.prop = ranked[0][0]

    def judge(self, item: Item, listings: list[Listing]) -> None:
        self.drop_label_restatements(item)
        self.align_props(item)
        reqs = item.req_texts()
        if not reqs:
            # nothing to judge beyond the kind of product: every opened listing stands
            for l in listings:
                if l.opened and l.id not in item.verdicts:
                    item.verdicts[l.id] = {"failed": set(), "unsure": set()}
                    item.judged_for[l.id] = set()
            return
        todo = [l for l in listings if l.opened and not (item.judged_for.get(l.id, set()) >= set(reqs))]
        for l in list(todo):
            # a listing that fails a requirement still in force fails whatever the new ones say: no call for it
            old = item.verdicts.get(l.id)
            if l.id in item.judged_for and old and set(old.get("failed") or ()) & set(reqs):
                item.judged_for[l.id] = set(reqs)
                todo.remove(l)
        if not todo or self.time_left() < WRAP_SECONDS:
            return
        table = self.field_table(item)
        fields = ("\n\nSTORE FIELDS (from this store's records read so far):\n" + self.field_lines(table)) if table else ""
        batches, prompts = [], []
        for start in range(0, len(todo), 4):
            batch = todo[start:start + 4]
            records = []
            for l in batch:
                rec = l.record()
                related = self.related_pages(item, l)
                if related:
                    rec["related_pages"] = related
                records.append(rec)
            batches.append(batch)
            prompts.append(f"REQUESTED PRODUCT: {item.category}\nREQUIREMENTS (judge each):\n"
                           + "\n".join(f"R{k}: {r}" for k, r in enumerate(reqs, 1)) + fields
                           + "\n\nLISTINGS:\n" + json.dumps(records, ensure_ascii=False))
        # the batches are independent: they are judged concurrently (the model's wait, not the steps, is the cost)
        answers = _parallel([lambda u=u: self.llm.json([FAST_MODEL, BACKUP_MODEL], JUDGE_PROMPT, u, 2000) for u in prompts])
        for batch, answer in zip(batches, answers):
            got = {}
            if isinstance(answer, dict) and isinstance(answer.get("verdicts"), list):
                for v in answer["verdicts"]:
                    if isinstance(v, dict) and v.get("id") in item.pool:
                        got[v["id"]] = v
            for l in batch:
                v = got.get(l.id)
                if v is None:
                    continue
                failed = {str(x) for x in (v.get("failed") or []) if x}
                unsure = {str(x) for x in (v.get("unsure") or []) if x}
                loose = {str(x) for x in (v.get("loose") or []) if x}
                weak = {str(x) for x in (v.get("weak") or []) if x} | loose
                # map free-text requirement names back to the requirement lines
                def canon(names: set[str]) -> set[str]:
                    out = set()
                    for n in names:
                        m = re.fullmatch(r"\s*R?(\d+)\s*", n)
                        if m and 1 <= int(m.group(1)) <= len(reqs):
                            out.add(reqs[int(m.group(1)) - 1])
                            continue
                        best = max(reqs, key=lambda r: _similar(r, n)) if reqs else None
                        if best and _similar(best, n) > 0.2:
                            out.add(best)
                        else:
                            out.add(n)
                    return out
                item.verdicts[l.id] = {"failed": canon(failed) | self.field_fails(item, l, table), "unsure": canon(unsure),
                                       "weak": canon(weak), "loose": canon(loose)}
                item.judged_for[l.id] = set(reqs)
                v = item.verdicts[l.id]
                for r in item.active():
                    # a value met "as written" needs its own words somewhere in the records
                    t = r.text()
                    if t not in v["failed"] and t not in v["unsure"] and t not in v["loose"] and self.unstated(item, l, r):
                        v["unsure"].add(t)

    @staticmethod
    def unstated(item: Item, listing: Listing, req: Requirement) -> bool:
        """No record of the listing carries any word of the required value beyond the product's own name words."""
        if req.op not in ("is", "one of") or req.origin == "fit" or not req.value:
            return False
        # a field that names the property is the judge's to read (other spellings, ranges, ages in years)
        if any(_near(str(k).replace("-", " "), req.prop) or _similar(str(k).replace("-", " ").replace("_", " "), req.prop) >= 0.5
               for k in listing.facts):
            return False
        common = _words(item.name) | _words(item.category)
        # short plain words (articles, joiners) prove nothing; figures of any length do
        alts = [w for w in ({x for x in _words(v) - common if not (x.isalpha() and len(x) <= 3)} for v in req.value.split("|")) if w]
        if not alts:
            return False
        text = {w[:4] for w in _words(" ".join([listing.title or "", listing.text or "", listing.notes or "", listing.category or "",
                                                " ".join(f"{k} {v}" for k, v in listing.facts.items())]))}
        return not any({w[:4] for w in a} & text for a in alts)

    def field_table(self, item: Item) -> list[dict[str, Any]]:
        """How this store writes each plain required value of the item, from the records of its listings read so far:
        the keys named for the property, how many read listings carry one, the values they write there and which
        listings write the required value itself. Nothing fixed: all of it comes from this episode's records."""
        read = [l for l in item.pool.values() if l.opened and l.facts and self.category_ok(item, l)]
        if len(read) < 3:
            return []
        out = []
        for r in item.active():
            if r.op not in ("is", "one of") or not r.value or re.search(r"\d", r.value):
                continue
            wants = [frozenset(_tokens(w)) for w in r.value.split("|") if _tokens(w)]
            if not wants:
                continue
            keys: set[str] = set()
            values: dict[str, int] = {}
            carriers, writers = 0, set()
            for l in read:
                hit = [k for k in l.facts if _near(k, r.prop)]
                if not hit:
                    continue
                carriers += 1
                keys.update(hit)
                for k in hit:
                    for e in re.split(r"[,;/|]", str(l.facts[k])):
                        e = " ".join(e.lower().split())
                        if e:
                            values[e] = values.get(e, 0) + 1
                            if any(w <= frozenset(_tokens(e)) for w in wants):
                                writers.add(l.id)
            if carriers:
                out.append({"req": r.text(), "prop": r.prop, "keys": sorted(keys), "carriers": carriers, "read": len(read),
                            "values": values, "wants": wants, "writers": writers})
        return out

    def field_fails(self, item: Item, listing: Listing, table: list[dict[str, Any]] | None = None) -> set[str]:
        """Requirements the listing plainly fails by the store's own wording: its field for the property has no element
        carrying the value, while other read listings carry the exact value in that field."""
        out = set()
        for row in (self.field_table(item) if table is None else table):
            own = [k for k in listing.facts if _near(k, row["prop"])]
            if not own or not (row["writers"] - {listing.id}):
                continue
            elems = [frozenset(_tokens(e)) for k in own for e in re.split(r"[,;/|]", str(listing.facts[k])) if _tokens(e)]
            # word stems, so another spelling of the value ("colour"/"color" style) is no plain failure
            stem = lambda words: {w[:4] for w in words}
            if elems and not any(stem(w) <= stem(e) for w in row["wants"] for e in elems):
                out.add(row["req"])
        return out

    def field_lines(self, table: list[dict[str, Any]]) -> str:
        lines = []
        for row in table:
            seen = " | ".join(f"{v} ({n})" for v, n in sorted(row["values"].items(), key=lambda x: -x[1])[:8])
            lines.append(f"- {row['req']}: field {', '.join(repr(k) for k in row['keys'][:3])} on {row['carriers']} of "
                         f"{row['read']} read listings; values written: {seen}; {len(row['writers'])} listing(s) write the "
                         "required value itself")
        return "\n".join(lines)

    def judge_all(self) -> None:
        """Judge every item's opened candidates, the items concurrently."""
        _parallel([lambda it=it: self.judge(it, [l for l in self.candidates(it) if l.opened]) for it in self.items])

    def related_pages(self, item: Item, listing: Listing, limit: int = 2) -> list[dict[str, Any]]:
        """Other known listings this one's text names by a code in their title, and those whose text names a code
        in this one's title (a compatibility list can go on another listing's page)."""
        code = re.compile(r"\b[A-Z]{1,6}-?\d{3,}[A-Z0-9]*\b")
        own_text = " ".join([listing.text or "", listing.notes or ""])
        own_codes = set(code.findall(listing.title or ""))
        mentioned = set(code.findall(own_text)) - own_codes
        out = []
        pools = [item.pool] + [it.pool for it in self.items if it is not item]
        seen: set[str] = {listing.id}
        for pool in pools:
            for other in pool.values():
                if other.id in seen or len(out) >= limit:
                    continue
                titled = set(code.findall(other.title or ""))
                names_it = bool(own_codes & set(code.findall(" ".join([other.text or "", other.notes or ""]))))
                if (titled & mentioned) or names_it:
                    seen.add(other.id)
                    out.append({"id": other.id, "title": other.title[:120],
                                "text": " ".join([other.text or "", other.notes or ""])[:400]})
        return out

    def verify(self, item: Item, listing: Listing) -> bool:
        key = (item.index, listing.id, tuple(item.req_texts()), json.dumps(listing.facts, sort_keys=True, default=str))
        cache = self.__dict__.setdefault("verified", {})
        if key not in cache:
            cache[key] = self._verify(item, listing)
        return cache[key]

    def _verify(self, item: Item, listing: Listing, strict: bool = False) -> bool:
        reqs = item.req_texts()
        user = (f"REQUESTED PRODUCT: {item.category} (quantity {item.quantity})\n"
                f"BUDGET, ALL LINES TOGETHER: {self.cap if self.cap is not None else 'none stated'}\n"
                "REQUIREMENTS:\n- " + ("\n- ".join(reqs) if reqs else "(none beyond the product kind, budget and availability)")
                + "\n\nLISTING:\n" + json.dumps(listing.record(3500), ensure_ascii=False))
        answer = self.llm.json([VERIFY_MODEL, FAST_MODEL], VERIFY_PROMPT, user, 800)
        if not isinstance(answer, dict):
            return not strict
        if strict and answer.get("fits") is not True:
            return False
        problems = [str(p) for p in (answer.get("problems") or [])]
        self.note("verify", id=listing.id, fits=answer.get("fits"), problems=problems[:4])
        if answer.get("fits") is False and problems:
            v = item.verdicts.setdefault(listing.id, {"failed": set(), "unsure": set()})
            for p in problems:
                best = max(reqs, key=lambda r: _similar(r, p)) if reqs else None
                v["failed"].add(best if best and _similar(best, p) > 0.2 else p)
            return False
        return True

    # -- the shopper
    def can_message(self) -> bool:
        if "message" not in self.roles:
            return False
        if self.message_budget is not None and self.messages_sent >= self.message_budget:
            return False
        return self.steps_left() > 0

    def held_messages(self) -> int:
        """Messages kept for the steps the request demands at the ending (a confirmation, a notice), not yet spent."""
        if "message" not in self.roles or not self.procedure:
            return 0
        def need(moments: tuple[str, ...]) -> int:
            return sum(max(1, int(st.get("messages") or 1)) for st in self.procedure
                       if st["kind"] == "message" and st["moment"] in moments)
        # a decline needs no message kept: the tool that ends without buying carries its own note
        held = need(("before_order", "before_finish"))
        if self.message_budget is not None:
            held = min(held, max(0, self.message_budget - self.messages_sent))
        return held

    def pending_messages(self) -> int:
        """Messages the ending still owes for the basket as it is now (none once it was shown to the shopper)."""
        return 0 if self.cart and self.confirmed_sig == self.basket_sig() else self.held_messages()

    def can_ask(self) -> bool:
        """A question may go out: a message is left beyond the ones kept for the basket's yes."""
        if not self.can_message():
            return False
        return self.message_budget is None or self.messages_sent < self.message_budget - self.held_messages()

    def gate(self, moment: str, doing: str) -> bool | None:
        """The steps the request demands at this moment and not yet done for the current state, decided by one model
        call over the request, the demanded steps, the tools, the episode so far and the basket, then run as tool
        calls checked against the schemas and the allowances. True when a step ran, False when none was due, None
        when the model gave no answer."""
        if self.gate_calls.get(moment, 0) >= 3 or self.steps_left() < 2 or self.time_left() < WRAP_SECONDS:
            return False
        self.gate_calls[moment] = self.gate_calls.get(moment, 0) + 1
        ending = {(self.roles.get(r) or {}).get("name") for r in ("checkout", "decline", "cart_add", "cart_set")} - {None}
        tools = [{"name": n, "description": self.tool_description(n), "parameters": self.schemas[n]}
                 for n in self.schemas if n not in ending]
        basket = []
        for lid, qty in self.basket_sig():
            item = self.cart_item.get(lid)
            l = item.pool.get(lid) if item else None
            basket.append({"id": lid, "quantity": qty, "title": (l.title if l else "")[:120],
                           "price": l.price if l else None, "for": item.name if item else None})
        total = sum((b["price"] or 0) * b["quantity"] for b in basket)
        left_msgs = (self.message_budget - self.messages_sent) if self.message_budget is not None else None
        user = ("REQUEST:\n" + self.request + "\n\nSTEPS THE REQUEST DEMANDS:\n" + json.dumps(self.procedure, ensure_ascii=False)
                + f"\n\nMOMENT: {moment} (the assistant is about to {doing})"
                + "\n\nBASKET NOW: " + json.dumps({"lines": basket, "total": total}, ensure_ascii=False)
                + f"\nMESSAGES LEFT: {left_msgs if left_msgs is not None else 'no limit'}; STEPS LEFT: {self.steps_left()}"
                + "\n\nTOOLS:\n" + json.dumps(tools, ensure_ascii=False)
                + "\n\nDONE SO FAR (tool calls and results, oldest first):\n" + json.dumps(self.history[-30:], ensure_ascii=False))
        answer = self.llm.json([PLAN_MODEL, BACKUP_MODEL], GATE_PROMPT, user, 900)
        if not isinstance(answer, dict):
            self.note("gate_failed", moment=moment)
            return None
        ran = False
        acts = [a for a in (answer.get("actions") or [])[:2] if isinstance(a, dict)]
        msg_tool = (self.roles.get("message") or {}).get("name")
        for k, act in enumerate(acts):
            if act.get("tool") not in self.schemas or act.get("tool") in ending:
                continue
            name = act["tool"]
            if name != msg_tool:
                # a message still due (in this answer or kept for the ending) and the checkout keep their steps
                due = max(sum(1 for a in acts[k + 1:] if a.get("tool") == msg_tool), self.pending_messages())
                if self.steps_left() < 2 + due:
                    self.note("gate_step_skipped", moment=moment, tool=name, steps=self.steps_left(), due=due)
                    continue
            schema = self.schemas[name]
            props = schema.get("properties") or {}
            args = {k: v for k, v in (act.get("args") or {}).items() if k in props} if isinstance(act.get("args"), dict) else {}
            for k, v in list(args.items()):
                enum = (props.get(k) or {}).get("enum")
                if isinstance(enum, list) and enum and v not in enum:
                    args.pop(k)
                elif isinstance(v, str):
                    args[k] = self._trim_args(props.get(k) or {}, v)
            if any(k not in args for k in schema.get("required") or []):
                continue
            message = name == (self.roles.get("message") or {}).get("name")
            if message and not self.can_message():
                continue
            if self.steps_left() < 2:
                break
            res = self.call(name, args)
            if message:
                self.messages_sent += 1
                self.confirmed_sig = self.basket_sig()
                if REPLY_ALL:
                    self.lever("REPLY_ALL", self.read_any_reply, " ".join(str(v) for v in args.values()), res)
            self.note("gate_step", moment=moment, tool=name, why=str(act.get("why") or "")[:160], ok=res["ok"])
            ran = True
        if not ran:
            self.note("gate_clear", moment=moment)
        return ran

    def _trim_args(self, prop: dict[str, Any], value: str) -> str:
        limit = prop.get("maxLength")
        return value[:limit] if isinstance(limit, int) and len(value) > limit else value

    def basket_sig(self) -> tuple:
        return tuple(sorted((lid, int(q or 0)) for lid, q in self.cart.items() if q))

    def confirm_basket(self) -> bool:
        """Show the shopper the final basket and ask for their yes. Returns True when the message went out."""
        if not self.can_message() or self.steps_left() < 2:
            return False
        spec = self.roles["message"]
        parts, total = [], 0.0
        for lid, qty in self.basket_sig():
            item = self.cart_item.get(lid)
            listing = item.pool.get(lid) if item else None
            title = re.sub(r"\s+", " ", (listing.title if listing else lid) or lid).strip()[:70]
            price = listing.price if listing and listing.price is not None else None
            if price is not None:
                total += price * qty
            parts.append(f"{qty} x {title}" + (f" ({price:g} each)" if price is not None else ""))
        text = ("Your final basket: " + "; ".join(parts) + (f". Total {total:g}." if total else ".")
                + " Shall I check out this basket? Please confirm with a yes.")
        text = self._trim("message", spec["text"], text)
        res = self.call(spec["name"], {spec["text"]: text})
        self.messages_sent += 1
        self.confirmed_sig = self.basket_sig()
        self.note("confirm_asked", basket=self.confirmed_sig, ok=res["ok"], obs=_compact(res["obs"], 160))
        if REPLY_ALL:
            self.lever("REPLY_ALL", self.read_any_reply, text, res)
        return True

    @staticmethod
    def reply_text(res: dict[str, Any]) -> str | None:
        """The shopper's words in a message tool's result."""
        obs = res.get("obs") if isinstance(res, dict) else None
        if isinstance(obs, str):
            return obs
        if isinstance(obs, dict):
            strings = [v for v in obs.values() if isinstance(v, str)]
            return max(strings, key=len) if strings else None
        return None

    def read_any_reply(self, sent: str, res: dict[str, Any]) -> None:
        """The reply to a message that asked about no property (a step the request demands, the basket to confirm) can
        state a need, point at another listing's page or give a new budget: it is read like a question's reply, and each
        need goes to the item it names (a page reference starts its chain at once, as after a question)."""
        reply = self.reply_text(res)
        if not reply or len(reply) < 8 or not self.items:
            return
        known = {it.name: [r.text() for r in it.requirements] for it in self.items}
        user = (f"REQUESTED ITEMS AND THEIR KNOWN NEEDS: {json.dumps(known, ensure_ascii=False)}\n"
                f"MESSAGE SENT: {sent[:500]}\nREPLY: {reply[:700]}")
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], ANY_REPLY_PROMPT, user, 700)
        if not isinstance(answer, dict):
            return
        self.note("any_reply", reply=reply[:200], read=_compact(answer, 300))
        before = (self.cap, [sorted(r.text() for r in it.requirements) for it in self.items])
        budget = _number(answer.get("budget"))
        if budget is not None and budget > 0:
            self.apply_reply(self.items[0], "", {"kind": "other", "budget": budget})
        for need in [x for x in answer.get("needs") or [] if isinstance(x, dict) and x.get("property")][:3]:
            item = self.items[0] if len(self.items) == 1 else self.item_named(need.get("item"))
            if item is not None:
                self.apply_reply(item, str(need["property"]), dict(need, kind="need"))
        after = (self.cap, [sorted(r.text() for r in it.requirements) for it in self.items])
        self.reply_changed = self.reply_changed or after != before

    def take_reply_change(self) -> bool:
        """Whether a reply read since the last look changed the needs or the budget (then cleared)."""
        changed, self.reply_changed = self.reply_changed, False
        return changed

    def ask(self, item: Item, prop: str, narrow: bool = False, also: str | None = None) -> dict[str, Any] | None:
        spec = self.roles["message"]
        noun = self.lever("SINGLE_ASK", self.ask_noun, item, fallback=item.name) if SINGLE_ASK else item.name
        text = f"Which {prop} would you like the {noun} to have?"
        if also:
            text = f"For the {noun}, which {prop} and which {also} would you like?"
        if narrow:
            seen = []
            for l in item.pool.values():
                for k, v in l.facts.items():
                    if _near(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(k)).replace("_", " "), prop) and str(v) not in seen:
                        seen.append(str(v)[:30])
            examples = "" if SINGLE_ASK else " or ".join(seen[:2])
            text = (f"Just this one thing: for the {noun}, what {prop} do you need" + (f", for example {examples}?" if examples else "?"))
        text = self._trim("message", spec["text"], text)
        res = self.call(spec["name"], {spec["text"]: text})
        self.messages_sent += 1
        if not narrow:
            item.asked.append(_norm(prop))
            if also:
                item.asked.append(_norm(also))
        if not res["ok"]:
            return None
        obs = res["obs"]
        reply = None
        if isinstance(obs, str):
            reply = obs
        elif isinstance(obs, dict):
            strings = [v for v in obs.values() if isinstance(v, str)]
            reply = max(strings, key=len) if strings else _compact(obs, 400)
        if not reply:
            return None
        known = [f"{r.prop}: {r.op} {r.value}" for r in item.requirements if r.origin == "shopper" and r.value]
        user = (f"ITEM: {noun}\nPROPERTY ASKED ABOUT: {prop + ' or ' + also if also else prop}\nQUESTION: {text}\nKNOWN NEEDS: {json.dumps(known)}\nREPLY: {reply}")
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL, PLAN_MODEL], REPLY_PROMPT, user, 600)
        if isinstance(answer, dict) and str(answer.get("kind") or "").lower() == "need":
            named = self.property_named(item, reply, [prop, also or "", str(answer.get("property") or "")], str(answer.get("value") or ""))
            if named:
                # the shopper put the need under another property the listings carry: that property is the need's
                answer["property"] = named
        self.note("reply", item=noun, prop=prop, reply=reply[:200], read=_compact(answer, 300))
        if isinstance(answer, dict) and str(answer.get("kind") or "").lower() in ("need", "repeat"):
            # the shopper's own words for a told need name its subject: a later question on it would be a repeat
            item.told_lines = list(getattr(item, "told_lines", [])) + [" " + _norm(reply) + " "]
        return answer if isinstance(answer, dict) else None

    @staticmethod
    def ask_noun(item: Item) -> str:
        """The product's name in a question. A name that lists several things ("x, y and z", "x & y") reads to a shopper
        as several subjects in one message: the reader's own plain product noun is used instead when it lists none."""
        joined = re.compile(r",|&|/|\band\b|\bor\b", re.I)
        if not joined.search(item.name):
            return item.name
        return next((n for n in (item.noun, item.category) if n and len(n) <= 40 and not joined.search(n)), item.name)

    @staticmethod
    def told_already(item: Item, prop: str) -> bool:
        """A property whose full name (two words or more) a reply that told a need already used."""
        name = _norm(prop)
        return len(name.split()) >= 2 and any(f" {name} " in line for line in getattr(item, "told_lines", []))

    def property_named(self, item: Item, reply: str, asked: list[str], value: str) -> str | None:
        """The one listing property a reply names by name when it names none of the asked ones, else None."""
        text = " " + _norm(reply) + " "
        if any(a and _norm(a) and " " + _norm(a) + " " in text for a in asked):
            return None
        keys: set[str] = set()
        for l in item.pool.values():
            for k in l.facts:
                key = _norm(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(k)))
                if len(key) >= 4 and not (_tokens(key) & _tokens(value)) and not any(_near(key, a) for a in asked if a):
                    keys.add(key)
        hits = [k for k in keys if " " + k + " " in text]
        # a longer name that contains a shorter one is the same mention (a two-word name holds its first word)
        hits = [k for k in hits if not any(k != o and " " + k + " " in " " + o + " " for o in hits)]
        return hits[0] if len(hits) == 1 else None

    def apply_reply(self, item: Item, prop: str, answer: dict[str, Any] | None) -> None:
        if not answer:
            return
        for step in answer.get("steps") or []:
            if isinstance(step, dict):
                self.add_step(step)
        kind = str(answer.get("kind") or "").lower()
        budget = _number(answer.get("budget"))
        if budget is not None and budget > 0:
            self.cap = budget
            self.note("budget", cap=budget)
        if kind == "need" and answer.get("value") or (kind == "need" and answer.get("reference")):
            p = str(answer.get("property") or prop)
            req = Requirement(p, answer.get("op") or "is", answer.get("value") or "", answer.get("reference"), origin="shopper")
            if req.reference:
                # the page is read for the property the need is about (the reader sometimes names an attribute instead)
                req.reference["property"] = req.prop
                # the shopper repeating a page reference already taken up (resolved or under way) is a repeat: the
                # requirement read from that page stays
                sig = _ref_sig(req.reference)
                if sig and any(sig in (getattr(r, "ref_sig", None), getattr(r, "ref_sig0", None))
                               or (r.reference and _ref_sig(r.reference) == sig) for r in item.requirements):
                    self.note("reference_repeat", item=item.name, prop=req.prop)
                    return
            # the same value already required from the shopper under another property name is a repeat
            if req.value and any(r.origin == "shopper" and _norm(r.value) == _norm(req.value) and r.op == req.op
                                 for r in item.requirements):
                return
            # a need replaces an earlier requirement on the same property only when it came from the shopper
            item.requirements = [r for r in item.requirements if not (r.key() == req.key() and r.origin == "shopper")]
            item.requirements.append(req)
            if req.op in ("is", "one of") and req.value and not req.reference and len(req.value) <= 40:
                # a heard value narrows the store's search: the item's next query carries it
                item.query_next(f"{req.value.split('|')[0].strip()} {item.noun}")
            if req.reference and req.op in REF_OPS:
                self.resolve_reference(item, req)
        elif kind == "free":
            item.freed.append(_norm(str(answer.get("property") or prop)))

    def apply_reply_kept(self, item: Item, prop: str, answer: dict[str, Any] | None) -> None:
        """apply_reply during the questions: a reference chain a reply starts leaves the remaining questions' steps
        (what it cannot reach now is taken up again after the questions)."""
        left = (self.message_budget - self.messages_sent) if self.message_budget is not None else 2
        self.ref_floor = max(0, left) + self.end_reserve()
        try:
            self.apply_reply(item, prop, answer)
        finally:
            self.ref_floor = 0

    def property_menu(self, item: Item) -> list[dict[str, Any]]:
        """The properties the item's candidate listings carry, most carried first: counted over the opened listings
        that no judgement has ruled out (when there are enough), keys that hold the same values merged into one."""
        opened = [l for l in item.pool.values() if l.opened and self.category_ok(item, l)]
        alive = [l for l in opened if not (item.verdicts.get(l.id) or {}).get("failed")]
        if len(alive) >= 4:
            opened = alive
        per: list[dict[str, str]] = []
        for l in opened:
            f: dict[str, str] = {}
            for k, v in l.facts.items():
                key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(k))
                key = re.sub(r"\s+", " ", key.replace("_", " ").replace(".", " ")).strip().lower()
                if key and key not in f:
                    f[key] = str(v).strip().lower()[:60]
            per.append(f)
        counts: dict[str, int] = {}
        for f in per:
            for k in f:
                counts[k] = counts.get(k, 0) + 1
        # on a tie the plainer (shorter) name speaks for its aliases
        keys = sorted(counts, key=lambda k: (-counts[k], len(k), k))
        alias: dict[str, str] = {}
        for i, a in enumerate(keys):
            if a in alias:
                continue
            for b in keys[i + 1:]:
                if b in alias:
                    continue
                both = [f for f in per if a in f and b in f]
                if len(both) >= 2 and sum(1 for f in both if f[a] == f[b]) >= 0.8 * len(both):
                    alias[b] = a
        fixed = {r.key() for r in item.requirements} | set(item.asked)
        menu = []
        for key in keys:
            if key in alias:
                continue
            names = [key] + [b for b, a in alias.items() if a == key]
            if any(_norm(n) in fixed or any(_similar(n, f) >= 0.67 for f in fixed) for n in names):
                continue
            if _tokens(key) & _NOT_ASKED:
                continue            # a variant selector: the shopper reads a question on it as unclear
            carriers = sum(1 for f in per if any(n in f for n in names))
            values: list[str] = []
            for f in per:
                v = next((f[n] for n in names if n in f), None)
                if v and v not in values:
                    values.append(v)
            # a property every listing gives the same value says little about what the shopper wants
            weight = carriers * (0.5 if len(values) == 1 and carriers >= 3 else 1.0)
            # nor does one whose every listing has its own value (an identifier, a variant selector)
            distinct = len({next((f[n] for n in names if n in f), None) for f in per} - {None})
            if carriers >= 4 and distinct >= carriers:
                weight *= 0.5
            menu.append({"property": key, "listings": carriers, "of": len(opened), "values": values[:3],
                         **({"same_as": names[1:4]} if len(names) > 1 else {}), "_w": weight})
        menu.sort(key=lambda e: (-e["_w"], e["property"]))
        for e in menu:
            e.pop("_w", None)
        return menu[:40]

    def decided_props(self, item: Item) -> list[str]:
        """Properties settled by the request's own rules (a condition's property, what a condition or a fit rule
        requires): never asked about."""
        out = []
        for c in getattr(self, "conditions", []):
            if c["src"] is item:
                out.append(c["prop"])
            out += [str(b["property"]) for _, d, b in c["branches"] if d is item]
        if any(a is item for a, _ in self.fit_pairs()):
            out.append("compatibility")
        return list(dict.fromkeys(out))

    def plan_questions(self, count: int, model: str = PLAN_MODEL) -> list[tuple[Item, str]]:
        """The model orders the questions (hints first, then the properties a shopper states, by how many listings
        carry them); code keeps the order free of junk and of properties already settled, and pads it."""
        if count <= 0:
            return []
        plan: list[tuple[Item, str]] = []

        def taken(item: Item, prop: str) -> bool:
            if any(_near(prop, a) for a in item.asked):
                return True
            if any(_near(prop, d) or _similar(prop, d) >= 0.5 for d in self.decided_props(item)):
                return True
            if any(_near(prop, r.prop) for r in item.requirements) or self.told_already(item, prop):
                return True
            return any(it is item and _near(prop, p) for it, p in plan)

        if not getattr(self, "planned", None):
            blocks = []
            for item in self.items:
                blocks.append({"item": item.name, "category": item.category, "known_requirements": item.req_texts(),
                               "decided_by_the_request": self.decided_props(item),
                               "hints": item.hints, "already_asked": item.asked, "property_menu": self.property_menu(item)})
            total = self.message_budget if self.message_budget is not None else count
            user = ("REQUEST:\n" + self.request + "\n\nSUBJECTS THE REQUEST TELLS THE ASSISTANT TO RAISE: " + json.dumps(self.ask_about)
                    + "\n\nITEMS:\n" + json.dumps(blocks, ensure_ascii=False)
                    + f"\n\nQUESTION_COUNT = {max(total, count)}")
            answer = self.llm.json([model, FAST_MODEL if model != FAST_MODEL else BACKUP_MODEL], PLAN_PROMPT, user, 700)
            ordered: list[tuple[Item, str]] = []
            if isinstance(answer, dict) and isinstance(answer.get("questions"), list):
                for q in answer["questions"]:
                    if not isinstance(q, dict) or not q.get("property"):
                        continue
                    target = max(self.items, key=lambda it: _similar(it.name, q.get("item") or it.name))
                    prop = re.sub(r"\s+", " ", str(q["property"]).replace("_", " ")).strip().lower()
                    if prop and not any(it is target and _near(prop, p) for it, p in ordered):
                        ordered.append((target, prop))
            self.planned = ordered
            self.note("plan_questions", questions=[(it.name, p) for it, p in ordered])
        if len(self.items) > 1:
            return self.multi_plan(count, taken)
        for item, prop in self.planned:
            if len(plan) >= count:
                break
            if not taken(item, prop):
                plan.append((item, prop))
        # the property most of an item's listings carry is a likely hidden need: keep it within the first questions
        if count >= 3:
            top_n = 2 if len(self.items) == 1 else 1
            for item in self.items:
                tops = [e["property"] for e in self.property_menu(item) if e["listings"] >= 3][:top_n]
                for k, prop in enumerate(tops):
                    if taken(item, prop):
                        continue
                    pos = min(len(plan), 1 + 2 * k if len(self.items) == 1 else len(self.items))
                    plan.insert(pos, (item, prop))
            plan = plan[:count]
        # pad from the menus (most-carried properties first, items with fewer questions first)
        if len(plan) < count:
            menus = {it.index: [e for e in self.property_menu(it) if e["listings"] >= 2 or e["of"] <= 2] for it in self.items}
            order = sorted(self.items, key=lambda it: (sum(1 for h, _ in plan if h is it) + len(it.asked), it.index))
            while len(plan) < count:
                added = False
                for item in order:
                    while menus[item.index]:
                        prop = menus[item.index].pop(0)["property"]
                        if taken(item, prop):
                            continue
                        plan.append((item, prop))
                        added = True
                        break
                    if len(plan) >= count:
                        break
                if not added:
                    break
        return plan[:count]

    def multi_plan(self, count: int, taken: Any) -> list[tuple[Item, str]]:
        """Several items: each gets its own queue (the planner's choices for it, its most carried property second,
        then its menu), and the questions take the items in turn (the planner's first item first, items whose need
        the shopper already told last), so no item goes unasked while another gets its third question."""
        queues: dict[int, list[str]] = {it.index: [] for it in self.items}
        first: list[int] = []
        for it, pr in self.planned:
            if it.index in queues and not taken(it, pr) and not any(_near(pr, q) for q in queues[it.index]):
                queues[it.index].append(pr)
                if it.index not in first:
                    first.append(it.index)
        for it in self.items:
            menu = [e for e in self.property_menu(it) if e["listings"] >= 2 or e["of"] <= 2]
            q = queues[it.index]
            top = next((e["property"] for e in menu if e["listings"] >= 3), None)
            if top and not taken(it, top) and not any(_near(top, x) for x in q[:2]):
                q[:] = [x for x in q if not _near(x, top)]
                q.insert(min(1, len(q)), top)
            for e in menu:
                if not taken(it, e["property"]) and not any(_near(e["property"], x) for x in q):
                    q.append(e["property"])
            if it.index not in first:
                first.append(it.index)
        told = {it.index for it in self.items if any(r.origin == "shopper" for r in it.requirements)}
        by_index = {it.index: it for it in self.items}
        # items with fewer questions so far go first (the plan is rebuilt after every need the shopper tells)
        turn = sorted(first, key=lambda i: (i in told, len(by_index[i].asked), first.index(i)))
        plan: list[tuple[Item, str]] = []
        while len(plan) < count and any(queues[i] for i in turn):
            for i in turn:
                if queues[i] and len(plan) < count:
                    plan.append((by_index[i], queues[i].pop(0)))
        return plan

    # -- references ("the same as what is written on the page for ...")
    def resolve_reference(self, item: Item, req: Requirement, hop: int = 0) -> None:
        ref = req.reference or {}
        if not getattr(req, "ref_sig0", None):
            req.ref_sig0 = _ref_sig(ref)       # the page first described (a repeat of it names this one)
        floor = max(4, getattr(self, "ref_floor", 0))
        if hop > 7 or self.steps_left() < floor or self.time_left() < WRAP_SECONDS + 20:
            return
        seller = str(ref.get("seller") or "").strip()
        category = str(ref.get("category") or "").strip()
        attrs = ref.get("attributes") if isinstance(ref.get("attributes"), dict) else {}
        target_prop = str(ref.get("property") or req.prop).strip()
        # one search state per described page: a later attempt pages on and skips the listings already checked
        key = json.dumps([_norm(seller), _norm(category), sorted((_norm(k), _norm(v)) for k, v in attrs.items())])
        scratch, checked = self.__dict__.setdefault("ref_search", {}).setdefault(
            key, (Item({"name": category or "listing", "category": category, "search_terms": []}, 99), set()))
        values = [str(v).strip() for v in attrs.values() if str(v).strip() and len(str(v).split()) <= 3]
        for v in attrs.values():
            # a longer description ("a sturdy, tall outdoor lamp from X") gives its own words
            if len(str(v).split()) > 3:
                values += [w for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", str(v)) if _norm(w) != _norm(seller)
                           and w not in values][:4]
        # words first, figures last: a request may restate a page's figure in other units
        values.sort(key=lambda v: bool(re.search(r"\d", v)))
        attr_words = " ".join(values[:3])
        noun = (category.split()[-1] if category else "")
        # "kettles and tea makers": "kettles" names the kind as well as "makers"
        nouns = list(dict.fromkeys(n for n in [noun, (re.split(r"\s+and\s+", category)[0].split() or [""])[-1] if category else ""] if n))
        queries = []
        if seller:
            queries.append((seller + " " + attr_words).strip())
            queries += [f"{seller} {n}" for n in nouns]
            queries.append(seller)
        if category and attr_words:
            queries.append(f"{attr_words} {noun}".strip())
        queries = list(dict.fromkeys(q for q in queries if q))

        def seller_ok(l: Listing) -> bool:
            if not seller:
                return True
            return (_similar(l.seller, seller) >= 0.5 or _similar(l.brand, seller) >= 0.5
                    or _norm(seller) in _norm(l.title + " " + l.text + " " + l.notes))

        def overlap(l: Listing) -> int:
            words = _tokens(attr_words)
            if not words:
                return 0
            raw = l.title + " " + l.text + " " + l.notes + " " + json.dumps(l.facts)
            text = _tokens(raw)
            # a figure with its unit counts written either way ("0.5 w" / "0.5W")
            squeezed = _norm(raw).replace(" ", "")
            joined = sum(1 for v in values if re.search(r"\d", v) and not (_tokens(v) and _tokens(v) <= text)
                         and len(_norm(v).replace(" ", "")) >= 3 and _norm(v).replace(" ", "") in squeezed)
            return len(words & text) + joined

        def attrs_in_text(l: Listing) -> bool:
            # every described value's words appear in the listing's own text (notes, title, description, table)
            if not attrs:
                return False
            text = _words(" ".join([l.title or "", l.notes or "", l.text or "", " ".join(f"{k} {v}" for k, v in l.facts.items())]))
            return all(_words(v) and _words(v) <= text for v in attrs.values() if str(v).strip())

        def attrs_match(l: Listing) -> bool:
            # the page's own properties state every described value
            if not attrs or not l.opened:
                return False
            facts = [(_norm(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(k))), _norm(v)) for k, v in l.facts.items()]
            for k, v in attrs.items():
                want = _norm(v)
                if not want or not any((_near(fk, k) or _similar(fk, k) >= 0.5) and fv == want for fk, fv in facts):
                    return False
            return True

        page = None

        def candidates_now() -> list[Listing]:
            mine = [l for l in scratch.pool.values() if l.id not in checked and seller_ok(l)]
            got = [l for l in mine if not category or self.category_ok(scratch, l)]
            if not got and seller and len(mine) <= 2 * self.compare_limit:
                got = mine          # a category the notes describe loosely: the seller's own listings decide
            # ties keep the store's own order (its best keyword matches first)
            order = {lid: i for i, lid in enumerate(scratch.pool)}
            return sorted(got, key=lambda l: (-overlap(l), not (_similar(l.seller, seller) >= 0.5), order.get(l.id, 0)))

        for q in queries:
            if page is not None or self.steps_left() < floor:
                break
            # the store ranks by matching words and shows a few rows a page: a page or two more before giving up
            for _ in range(3):
                if self.steps_left() < floor or self.search(scratch, query=q, max_pages=1, budget=3) == 0 or candidates_now():
                    break
            found = candidates_now()
            if not found:
                continue
            checked.update(l.id for l in found)
            page = self._match_page(scratch, found, ref, seller, category, attrs, attrs_match, attrs_in_text)
        if page is None and not checked:
            self.note("reference_unfound", seller=seller, category=category)
            return
        if page is None:
            self.note("reference_unmatched", seller=seller, category=category)
            return
        paired = req.op == "counterpart of reference"

        relation = str(ref.get("relation") or "").strip()

        def read_page() -> Any:
            user = (f"PROPERTY WANTED: {target_prop}\n" + (f"RELATION THE SHOPPER ASKED FOR: {relation}\n" if relation else "")
                    + "\nPAGE:\n" + json.dumps(page.record(3000), ensure_ascii=False))
            return self.llm.json([FAST_MODEL, BACKUP_MODEL], REFER_PROMPT, user, 600)

        # in a store whose rows hold no property table, the notes are read first: a page that only points on costs
        # no read of its table
        thin = self.thin_compare and not page.facts
        if thin and not page.notes and self.steps_left() >= 3:
            page.facts, thin = self.fetch_facts(page.id) or {}, False
        read = read_page()
        if thin and self.steps_left() >= 3 and isinstance(read, dict) and not read.get("redirect") \
                and (not read.get("value") or paired):
            # (a pairing turns on the page's own value, which its property table states)
            page.facts = self.fetch_facts(page.id) or {}
            if page.facts:
                read = read_page()
        self.note("reference", hop=hop, page=page.id, read=_compact(read, 200))
        redirect = read.get("redirect") if isinstance(read, dict) else None
        if isinstance(redirect, dict) and redirect:
            about = str(redirect.get("property") or target_prop)
            if not (_near(about, target_prop) or _similar(about, target_prop) >= 0.5
                    or _norm(target_prop) in _norm(about) or _norm(about) in _norm(target_prop)
                    or (REDIRECT_CHECK and self.lever("REDIRECT_CHECK", self.same_property, about, target_prop, page,
                                                      fallback=False))):
                # the notes send ANOTHER property elsewhere: this page's own value of the wanted one stands
                redirect = None
                if not read.get("value") and thin and not page.facts and self.steps_left() >= 3:
                    page.facts = self.fetch_facts(page.id) or {}
                    again = read_page() if page.facts else None
                    if isinstance(again, dict):
                        read["value"] = again.get("value") or read.get("value")
                        read["paired"] = again.get("paired") or read.get("paired")
                if not read.get("value"):
                    own = next((v for k, v in page.facts.items()
                                if _near(str(k).replace("_", " "), target_prop) or _similar(str(k).replace("_", " "), target_prop) >= 0.5), None)
                    if own not in (None, ""):
                        read["value"] = str(own)
                self.note("reference_other_redirect", about=about, wanted=target_prop, value=read.get("value"))
        if isinstance(redirect, dict) and redirect:
            req.reference = redirect
            req.reference["property"] = target_prop
            if relation:
                req.reference["relation"] = relation
            if ref.get("tolerance"):
                req.reference["tolerance"] = ref["tolerance"]
            self.resolve_reference(item, req, hop + 1)
            return
        value = read.get("paired" if paired else "value") if isinstance(read, dict) else None
        if isinstance(read, dict) and req.op == "same as reference" and read.get("paired") and read.get("wanted") \
                and _norm(read["wanted"]) == _norm(read["paired"]) and _norm(read["paired"]) != _norm(read.get("value")):
            # the shopper's words ask for what goes with the page's value, and the page names it
            value = read["paired"]
            self.note("relation_paired", page=page.id, relation=relation[:80], value=value)
        if value and req.op == "within reference":
            # a figure give or take an amount: two limits around the page's figure, in its unit
            tol = str((ref.get("tolerance") or "")).strip()
            num, amount = _number(value), _number(tol)
            unit = re.sub(r"^[\s\d.,+-]+", "", str(value)).strip()
            if num is not None and amount is not None:
                lo, hi = num - amount, num + amount
                req.ref_sig = _ref_sig(ref)
                req.value, req.op, req.reference = f"{lo:g} {unit}".strip(), "at least", None
                item.requirements.append(Requirement(req.prop, "at most", f"{hi:g} {unit}".strip(), origin=req.origin))
                self.note("reference_range", prop=req.prop, low=req.value, high=f"{hi:g} {unit}".strip())
                value = None
            else:
                req.op = "same as reference"
        if value:
            req.ref_sig = _ref_sig(ref)
            req.value = str(value)
            req.op = REF_OPS.get(req.op, "is")
            req.reference = None
            if req.op == "is" and len(req.value) <= 40 and item.terms and not any(_norm(req.value) in _norm(t) for t in item.terms):
                # the value the page gave is the item's most specific query
                item.query_next(f"{req.value} {item.noun}")
        elif paired and isinstance(read, dict) and read.get("value"):
            found = self.lever("PAIR_HUNT", self.pair_hunt, item, page, str(read["value"]), target_prop) if PAIR_HUNT else None
            req.ref_sig = _ref_sig(ref)
            if found:
                # the store's own pairing statement, read on another listing's notes
                req.value, req.op, req.reference = found, "is", None
                if len(found) <= 40:
                    item.query_next(f"{found} {item.noun}")
                return
            # the page names no pairing: its own value is the closest reading of what the shopper pointed at
            self.note("pair_unstated", page=page.id, value=read.get("value"))
            req.value, req.op, req.reference = str(read["value"]), "is", None

    def _match_page(self, scratch: Item, found: list[Listing], ref: dict, seller: str, category: str, attrs: dict,
                    attrs_match: Any, attrs_in_text: Any) -> Listing | None:
        """The listing a reference describes among the candidates found (read a batch at a time, then matched)."""
        page = None
        tried = 0
        pick = str(ref.get("pick") or "").strip()
        count = ref.get("count") if isinstance(ref.get("count"), int) else None
        if pick or (count or 0) > 1:
            # a choice among several of the seller's listings: every candidate page is read before choosing (a few
            # more than the description says fit)
            reach = min(len(found), 3 * self.compare_limit, max(4, 2 * (count or 2)))
            for start in range(0, reach, self.compare_limit):
                if self.steps_left() < 3:
                    break
                todo = [x.id for x in found[start:min(start + self.compare_limit, reach)]
                        if not (scratch.pool.get(x.id) and scratch.pool[x.id].opened)]
                for l in self.open_batch(None, todo, spare=[x.id for x in scratch.pool.values() if not x.opened]):
                    scratch.pool[l.id] = l
            records = [scratch.pool[l.id].record(1200) for l in found if l.id in scratch.pool and scratch.pool[l.id].opened]
            if records:
                user = ("DESCRIPTION: category " + category + "; seller " + seller + "; attributes " + json.dumps(attrs, ensure_ascii=False)
                        + (f"\nPICK: {pick}" if pick else "") + (f"\nLISTINGS THAT FIT: {count}" if count else "")
                        + "\n\nCANDIDATES:\n" + json.dumps(records, ensure_ascii=False))
                match = self.llm.json([FAST_MODEL, BACKUP_MODEL], MATCH_PROMPT, user, 600)
                if isinstance(match, dict) and match.get("id") in scratch.pool:
                    page = scratch.pool[match["id"]]
            # the pages read and judged no match are not opened again below
            found = [] if page is not None else [l for l in found if not (scratch.pool.get(l.id) and scratch.pool[l.id].opened)]
        single = not self.roles.get("compare") or getattr(self, "compare_off", False)
        size = 1 if single else self.compare_limit
        seen: list[Listing] = []
        while found and tried < (5 if single else 3) and page is None and self.steps_left() >= 3:
            batch = found[:size]
            found = found[size:]
            tried += 1
            opened = self.open_batch(None, [l.id for l in batch], spare=[x.id for x in found if not x.opened]
                                     + [x.id for x in scratch.pool.values() if not x.opened])
            for l in opened:
                scratch.pool[l.id] = l
            now = [scratch.pool[l.id] for l in batch if l.id in scratch.pool and scratch.pool[l.id].opened]
            exact = [l for l in now if attrs_match(l)]
            if exact:
                page = exact[0]
                break
            seen += now
            if single and found and len(seen) < 3:
                continue          # a few single pages before the model compares them
            shown = seen if single else now
            records = [l.record(1600) for l in shown]
            seen = []
            if not records:
                continue
            user = ("DESCRIPTION: category " + category + "; seller " + seller + "; attributes " + json.dumps(attrs, ensure_ascii=False)
                    + "\n\nCANDIDATES:\n" + json.dumps(records, ensure_ascii=False))
            match = self.llm.json([FAST_MODEL, BACKUP_MODEL], MATCH_PROMPT, user, 600)
            worded = [l for l in shown if attrs_in_text(l)]
            if isinstance(match, dict) and match.get("id") in scratch.pool:
                page = scratch.pool[match["id"]]
            elif isinstance(match, dict) and "id" in match:
                continue                  # the matcher read them all and none is the page: look further
            elif len(worded) == 1:
                page = worded[0]          # the only one whose own text carries every described value
            elif len(records) == 1 and not found:
                page = scratch.pool[records[0]["id"]]
        return page

    def same_property(self, a: str, b: str, page: Listing) -> bool:
        """Whether two names call one property of the page's kind of product (the note says "shade" where the shopper
        said "tint"): one fast model call, made only when the names differ; the note's redirect is then followed rather
        than set aside."""
        user = f"PRODUCT KIND: {page.category or 'unknown'}\nNAME A: {a}\nNAME B: {b}"
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], SAME_PROMPT, user, 60)
        same = isinstance(answer, dict) and answer.get("same") is True
        self.note("redirect_check", a=a, b=b, same=same)
        return same

    def known_listings(self) -> list[Listing]:
        """Every listing whose page was read so far, once: the items' own and those looked at while following
        references."""
        out, seen = [], set()
        pools = [it.pool for it in self.items] + [s.pool for s, _ in getattr(self, "ref_search", {}).values()]
        for pool in pools:
            for l in pool.values():
                if l.id not in seen and (l.page_read or l.notes):
                    seen.add(l.id)
                    out.append(l)
        return out

    def pair_hunt(self, item: Item, page: Listing, value: str, prop: str) -> str | None:
        """A counterpart reference (the value that goes with the page's value) whose page states no pairing. A store
        writes which value goes with which on other listings' notes: a listing whose own value it is, or a list of the
        store's pairs. The notes already read are looked through first (no step); then listings found by the value with
        the page's kind, by the page's seller with its kind, and by the value alone are read, PAIR_READS rounds at most,
        keeping the steps the rest of the task needs. None: no pairing found (the page's own value stands)."""
        floor = max(4, getattr(self, "ref_floor", 0)) + 2
        words = _tokens(value)
        if not words:
            return None
        scratch = Item({"name": page.category or "listing", "category": page.category or "", "search_terms": []}, 99)
        kind = (page.category.split() or [""])[-1] if page.category else item.noun
        queries = [q for q in dict.fromkeys([f"{value} {kind}".strip(), f"{page.seller} {kind}".strip() if page.seller else "",
                                             value]) if q]
        looked: set[str] = set()
        for rnd in range(PAIR_READS + 1):
            if rnd:
                if not queries or self.steps_left() < floor + 2 or self.time_left() < WRAP_SECONDS + 25 or self.spent():
                    break
                self.search(scratch, query=queries.pop(0), max_pages=1, budget=floor)
                rows = [l for l in scratch.pool.values() if l.id not in looked and not l.page_read]
                rows.sort(key=lambda l: (not (_tokens(f"{l.title} {l.text}") & words), not self.category_ok(scratch, l)))
                take = rows[: self.compare_limit if self.batch_open() else 2]
                for got in self.open_batch(None, [l.id for l in take], floor=floor, spare=[l.id for l in rows[len(take):]]):
                    scratch.pool[got.id] = got
            notes = []
            for l in self.known_listings() + [l for l in scratch.pool.values() if l.page_read]:
                text = " ".join(x for x in (l.notes, l.text) if x)
                if l.id in looked or not text:
                    continue
                looked.add(l.id)
                own = next((str(v) for k, v in l.facts.items()
                            if _near(k, prop) or _similar(str(k).replace("_", " "), prop) >= 0.5), None)
                if words & _tokens(text) or (own and words & _tokens(own)):
                    notes.append({"id": l.id, "category": l.category, "own_value": own, "notes": text[:600]})
            found = self.read_pairing(notes[:12], value, prop) if notes else None
            if found:
                return found
        return None

    def read_pairing(self, notes: list[dict[str, Any]], value: str, prop: str) -> str | None:
        """The value the notes pair with `value` for `prop`, by one fast model call; taken only when the quote it gives
        stands word for word in that listing's notes, holds the paired value, and holds the given value or comes from a
        listing whose own value it is."""
        user = f"PROPERTY: {prop}\nGIVEN VALUE: {value}\n\nNOTES:\n" + json.dumps(notes, ensure_ascii=False)
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], PAIR_PROMPT, user, 400)
        if not isinstance(answer, dict) or not answer.get("paired") or not answer.get("quote"):
            return None
        quote = " ".join(str(answer["quote"]).lower().split())
        paired = str(answer["paired"]).strip()
        flat = lambda n: " ".join(n["notes"].lower().split())
        where = next((n for n in notes if n["id"] == answer.get("id") and quote in flat(n)), None) \
            or next((n for n in notes if quote and quote in flat(n)), None)
        ok = (where is not None and bool(_tokens(paired)) and _tokens(paired) <= _tokens(quote)
              and bool(_tokens(value) & _tokens(quote) or (where.get("own_value") and _tokens(value) & _tokens(where["own_value"]))))
        self.note("pair_hunt", value=value, prop=prop, paired=paired, quote=quote[:160], ok=ok)
        return paired if ok else None

    # -- events (shopper lines inside tool results)
    def handle_lines(self) -> bool:
        """Interpret pending shopper lines. Returns True when something changed."""
        lines = [l for l in self.pending_lines if l not in self.handled_lines]
        self.pending_lines = []
        if not lines:
            return False
        for l in lines:
            self.handled_lines.add(l)
        user = "LINES:\n- " + "\n- ".join(lines) + "\n\nREQUESTED ITEMS: " + json.dumps([i.name for i in self.items])
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], EVENT_PROMPT, user, 700)
        self.note("event", lines=[l[:160] for l in lines], read=_compact(answer, 300))
        changed = False
        if not isinstance(answer, dict):
            return False
        for ch in answer.get("changes") or []:
            if not isinstance(ch, dict):
                continue
            kind = str(ch.get("kind") or "").lower()
            target = None
            if ch.get("item"):
                target = max(self.items, key=lambda it: _similar(it.name, ch.get("item")))
                if _similar(target.name, ch.get("item")) < 0.2 and len(self.items) > 1:
                    target = None
            if kind == "budget" and _number(ch.get("budget")):
                budget = _number(ch.get("budget"))
                in_lines = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", " ".join(lines))]
                if budget not in in_lines:
                    for alt in (budget * 100, budget / 100):
                        if alt in in_lines:
                            budget = alt
                            break
                    else:
                        if in_lines and self.cap is not None:
                            near = [x for x in in_lines if self.cap / 20 <= x <= self.cap * 20]
                            if near:
                                budget = near[0]
                self.cap = budget
                changed = True
            elif kind == "drop" and ch.get("property"):
                for item in ([target] if target else self.items):
                    before = len(item.requirements)
                    item.requirements = [r for r in item.requirements if _similar(r.prop, ch["property"]) < 0.5]
                    if len(item.requirements) != before:
                        changed = True
                        for v in item.verdicts.values():
                            v["failed"] = {f for f in v["failed"] if _similar(f.split(":")[0], ch["property"]) < 0.5}
            elif kind == "add" and ch.get("property") and ch.get("value"):
                for item in ([target] if target else self.items[:1]):
                    item.requirements.append(Requirement(ch["property"], ch.get("op") or "is", ch["value"], None, origin="shopper"))
                    changed = True
            elif kind in ("price", "stock"):
                changed = True
                self.refresh_cart_items()
            elif kind == "step" and isinstance(ch.get("step"), dict):
                self.add_step(ch["step"])
        return changed

    def add_step(self, step: dict[str, Any]) -> None:
        """A step the shopper demands mid-task joins the procedure the gate checks."""
        if not step.get("goal") or len(self.procedure) >= 8:
            return
        step = dict(step)
        step["moment"] = str(step.get("moment") or "before_finish").lower()
        step["kind"] = "message" if str(step.get("kind") or "").lower() == "message" else "tool"
        if any(_similar(step["goal"], st.get("goal")) >= 0.6 for st in self.procedure):
            return
        self.procedure.append(step)
        self.confirm_first = self.confirm_first or (step["kind"] == "message" and step["moment"] in ("before_order", "before_finish"))
        self.note("step_added", step=step)

    def refresh_cart_items(self) -> None:
        for lid, item in list(self.cart_item.items()):
            if self.steps_left() <= 2:
                return
            opened = self.open_batch(item, [lid], page=True)
            self.note("refreshed", id=lid, price=item.pool[lid].price if lid in item.pool else None,
                      stock=item.pool[lid].stock if lid in item.pool else None)

    # -- cart and finishing
    def set_cart(self, lines: dict[str, int]) -> bool:
        """The cart set to exactly these lines (listing id -> quantity) through a tool that replaces it."""
        spec = self.roles.get("cart_set") or {}
        props = self._schema("cart_set").get("properties") or {}
        line_props = ((props.get(spec.get("lines") or "") or {}).get("items") or {}).get("properties") or {}
        maximum = (line_props.get(spec.get("qty") or "") or {}).get("maximum")
        rows = []
        for lid, qty in lines.items():
            row = {spec["id"]: lid}
            if spec.get("qty"):
                row[spec["qty"]] = min(qty, maximum) if isinstance(maximum, int) else qty
            rows.append(row)
        res = self.call(spec["name"], {spec["lines"]: rows})
        if not res["ok"]:
            self.note("cart_set_failed", error=res["error"], obs=_compact(res["obs"], 200))
        return res["ok"]

    def cart_add(self, item: Item, listing: Listing) -> bool:
        if not self.roles.get("cart_add") and self.roles.get("cart_set"):
            lines = dict(self.cart)
            lines[listing.id] = item.quantity
            if not self.set_cart(lines):
                return False
            self.cart = lines
            self.cart_item[listing.id] = item
            return True
        spec = self.roles.get("cart_add")
        if not spec:
            return False
        args: dict[str, Any] = {spec["id"]: listing.id}
        qty = item.quantity
        if spec.get("qty"):
            props = self._schema("cart_add").get("properties") or {}
            maximum = (props.get(spec["qty"]) or {}).get("maximum")
            if isinstance(maximum, int) and qty > maximum:
                qty = maximum
            args[spec["qty"]] = qty
        mode = spec.get("mode") if isinstance(spec.get("mode"), dict) else None
        if mode and mode.get("arg") and mode.get("add") is not None:
            args[mode["arg"]] = mode["add"]
        res = self.call(spec["name"], args)
        if res["ok"] and not _refused_cart(res["obs"]):
            self.cart[listing.id] = qty
            self.cart_item[listing.id] = item
            return True
        self.cart_faulty = bool(res.get("bad_args"))
        self.note("cart_add_failed", error=res["error"], obs=_compact(res["obs"], 200))
        return False

    def cart_remove(self, listing_id: str) -> bool:
        if not self.roles.get("cart_add") and self.roles.get("cart_set"):
            lines = {k: v for k, v in self.cart.items() if k != listing_id}
            if not self.set_cart(lines):
                return False
            self.cart = lines
            self.cart_item.pop(listing_id, None)
            return True
        spec = self.roles.get("cart_add")
        if not spec:
            return False
        args: dict[str, Any] = {spec["id"]: listing_id}
        mode = spec.get("mode") if isinstance(spec.get("mode"), dict) else None
        if mode and mode.get("arg") and mode.get("remove") is not None:
            args[mode["arg"]] = mode["remove"]
            if spec.get("qty"):
                args[spec["qty"]] = self.cart.get(listing_id, 1)
        elif spec.get("qty"):
            props = self._schema("cart_add").get("properties") or {}
            minimum = (props.get(spec["qty"]) or {}).get("minimum")
            args[spec["qty"]] = 0 if not isinstance(minimum, int) or minimum <= 0 else minimum
        res = self.call(spec["name"], args)
        if res["ok"]:
            self.cart.pop(listing_id, None)
            self.cart_item.pop(listing_id, None)
            return True
        return False

    def finish_order(self) -> bool:
        spec = self.roles.get("checkout")
        if not spec:
            return False
        if self.procedure and self.cart:
            ran = self.gate("before_order", "check out the basket below as the order")
            if ran and self.pending_messages() and self.steps_left() >= 2:
                # the checks ran but a demanded message is still owed for this basket
                ran = self.gate("before_order", "check out the basket below as the order") or ran
            if ran is not False and self.confirm_first and self.basket_sig() != self.confirmed_sig:
                ran = self.confirm_basket() or ran
            if ran and (self.handle_lines() or self.price_or_stock_changed() or self.take_reply_change()):
                # a change rode on the reply: the caller re-chooses, and the gate sees the new basket next time
                self.note("procedure_changed")
                return False
        self.show_pointed_parts()
        res = self.call(spec["name"], self.fixed_args("checkout"))
        if self.finished:
            self.note("ordered", cart=self.cart)
            return True
        self.note("order_refused", obs=_compact(res["obs"], 200), lines=res["lines"][:3])
        return False

    def show_pointed_parts(self) -> None:
        """A cart line whose text sends the reader on to another part's page has that page shown before the order:
        one keyword search for the codes that no listing seen so far carries in its title."""
        spec = self.roles.get("search_text")
        if not spec or not spec.get("query") or not self.cart:
            return
        shown = self.__dict__.setdefault("parts_shown", set())
        titles = " ".join(l.title or "" for it in self.items for l in it.pool.values())
        codes: list[str] = []
        for lid in self.cart:
            l = next((it.pool[lid] for it in self.items if lid in it.pool), None)
            if l is None:
                continue
            for m in _POINTER.finditer(" ".join([l.text or "", l.notes or ""])):
                code = m.group(1).strip("-")
                if len(code) < 4 or not re.search(r"\d", code) or code.lower() in shown or code in codes:
                    continue
                if re.search(r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![A-Za-z0-9])", l.title or "", re.I):
                    continue          # its own code
                if re.search(r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![A-Za-z0-9])", titles, re.I):
                    shown.add(code.lower())
                    continue
                codes.append(code)
        if not codes or self.steps_left() < 2 + self.pending_messages():
            return
        args: dict[str, Any] = {}
        if isinstance(spec.get("mode"), dict) and spec["mode"].get("arg"):
            args[spec["mode"]["arg"]] = spec["mode"].get("value")
        args[spec["query"]] = self._trim("search_text", spec["query"], " ".join(codes[:5]))
        res = self.call(spec["name"], args)
        shown.update(c.lower() for c in codes[:5])
        self.note("pointed_parts", codes=codes[:5], rows=len(_find_list_of_dicts(res["obs"]) or []))

    def nearest_miss(self, item: Item) -> Listing | None:
        """For a request that wants the closest listing when nothing fits: the read listing of the item's kind that
        misses the fewest of the item's requirements (being sold out, over the item's own price cap or failed by the
        store's fields each count as one), then the fewest unaddressed ones, then the cheapest."""
        table = self.field_table(item)
        best, best_key = None, None
        for l in item.pool.values():
            if not l.opened or l.id not in item.verdicts or not self.category_ok(item, l):
                continue
            v = item.verdicts[l.id]
            failed = set(v.get("failed") or ())
            if "cart refused" in failed:
                continue
            short = len(failed | self.field_fails(item, l, table)) + (l.stock is False) + item.over_cap(l)
            if self.cap is not None and l.price is not None and l.price * item.quantity > self.cap + 1e-6:
                short += 1
            key = (short, len(v.get("unsure") or ()), l.price if l.price is not None else 1e18)
            if best_key is None or key < best_key:
                best, best_key = l, key
        return best

    def order_nearest(self) -> bool:
        """Instead of ending without buying, when the request asks for the closest listing in that case: each item's
        pick (its chosen fit, else its best fit, else its nearest miss) is carted and ordered."""
        if getattr(self, "nearest_tried", False) or not self.items or not self.roles.get("checkout"):
            return False
        self.nearest_tried = True
        given = getattr(self, "near_picks", None) or {}
        picks: dict[int, Listing] = {}
        for item in self.items:
            pick = given.get(item.index)
            if pick is None:
                ranked = self.rank_best(item, self.buyable(item))
                pick = ranked[0] if ranked else self.nearest_miss(item)
            if pick is not None:
                picks[item.index] = pick
        if not picks:
            return False
        stale = [lid for lid in self.cart if not any(p.id == lid for p in picks.values())]
        adds = [it for it in self.items if it.index in picks and picks[it.index].id not in self.cart]
        self.note("nearest_order", picks={it.name: picks[it.index].id for it in self.items if it.index in picks})
        for lid in stale:
            if self.steps_left() > len(adds) + 1:
                self.cart_remove(lid)
        for it in adds:
            if self.steps_left() < 2:
                break
            it.chosen = picks[it.index]
            self.cart_add(it, it.chosen)
        if not self.cart or self.steps_left() < 1:
            return False
        if self.finish_order() or self.finished:
            return True
        # refused: a change the checkout brought (a requirement let go, a bigger budget) may leave a fit now
        self.nearest_rounds = getattr(self, "nearest_rounds", 0) + 1
        if self.handle_lines() and self.nearest_rounds <= 2 and self.steps_left() >= len(self.items) + 1:
            self.nearest_tried = False
            self.near_picks = None
            self.choose_and_finish()
            return self.finished
        return False

    def order_unsure(self) -> bool:
        """With no rule for the case that nothing fits, a listing that no requirement plainly fails (only some left
        unaddressed by its records) is ordered rather than ending without buying: every item needs one, within its
        own cap and the budget."""
        if getattr(self, "unsure_tried", False) or getattr(self, "nearest_tried", False) or not self.roles.get("checkout"):
            return False
        self.unsure_tried = True
        picks: dict[int, Listing] = {}
        for item in self.items:
            ranked = self.rank_best(item, self.buyable(item))
            if not ranked:
                table = self.field_table(item)
                ranked = [l for l in item.fits(include_unsure=True, cheapest=self.cheapest)
                          if l.stock is not False and not self.field_fails(item, l, table) and not item.word_loose(l.id)
                          and "cart refused" not in (item.verdicts.get(l.id) or {}).get("failed", ())]
            if not ranked:
                return False
            picks[item.index] = ranked[0]
        total = sum((picks[it.index].price or 0) * it.quantity for it in self.items)
        if self.cap is not None and total > self.cap + 1e-6:
            return False
        self.note("unsure_order", picks={it.name: picks[it.index].id for it in self.items})
        self.near_picks = picks
        return self.order_nearest()

    def decline(self, reason: str = "") -> bool:
        spec = self.roles.get("decline")
        if self.nearest_ok and self.order_nearest():
            return True
        if not self.nearest_ok and not getattr(self, "decline_stated", False) and self.order_unsure():
            return True
        if self.finished:
            return True
        if not spec:
            return False
        if self.procedure and not getattr(self, "decline_gated", False):
            self.decline_gated = True
            self.gate("before_decline", "end the task without buying anything")
        args: dict[str, Any] = self.fixed_args("decline")
        remark = spec.get("remark")
        if remark and remark in (self._schema("decline").get("required") or []):
            args[remark] = self._trim("decline", remark, "Nothing available meets every requirement of the request.")
        res = self.call(spec["name"], args)
        self.note("declined", reason=reason or self.decline_reason)
        if not self.finished and not getattr(self, "decline_refused", False) and self.steps_left() >= 1:
            # refused: a change announced at the finish (a new budget, a new price) may make something fit now
            self.decline_refused = True
            self.note("decline_refused", obs=_compact(res["obs"], 200), lines=res["lines"][:3])
            if self.handle_lines() and self.steps_left() >= len(self.items) + 2:
                self.choose_and_finish()
            if not self.finished:
                # a store may refuse to close with lines in the cart (an early pick held there): they go first
                for lid in list(self.cart):
                    if self.steps_left() > 1:
                        self.cart_remove(lid)
                self.call(spec["name"], args)
        return self.finished

    # ------------------------------------------------------------------ the run
    def run(self) -> list[dict[str, Any]]:
        try:
            if not self.setup():
                self.force_finish()
                return self.trace
            self.work()
        except Exception as exc:   # never leave an episode unfinished
            self.note("exception", error=repr(exc)[:300])
        try:
            if not self.finished:
                self.force_finish()
        except Exception as exc:
            self.note("exception", error=repr(exc)[:300])
        self.note("done", cost=round(self.llm.cost, 4), llm_calls=self.llm.calls, steps=self.step)
        return self.trace

    def end_reserve(self) -> int:
        """Steps that must stay free for the ending: the cart adds, a look at the carted listing, a swap after a
        change, the checkout and one retry."""
        if getattr(self, "reserve_override", None) is not None:
            return self.reserve_override
        looks = min(len(self.items), 3) if self.sheet_first() else 2
        return (len(self.items) + 6 + (looks if self.roles.get("facts") and not self.roles.get("compare") else 0)
                + min(4, sum(1 for st in self.procedure if st["moment"] != "start") + (1 if self.confirm_first else 0))
                + (1 if self.cheapest and not getattr(self, "backups_done", False) else 0))

    def work(self) -> None:
        items = self.items
        n = len(items)
        questions = (self.message_budget if self.message_budget is not None else 3) - self.held_messages()
        if "message" not in self.roles or questions < 0:
            questions = 0

        # references stated in the request itself (the questions' steps and a little exploration are kept apart;
        # a chain left unfinished is taken up again after the questions)
        self.ref_floor = questions + (1 if questions else 0) + self.end_reserve() + 2 * n
        for item in items:
            for req in list(item.requirements):
                if req.reference and req.op in REF_OPS:
                    self.resolve_reference(item, req)
        self.ref_floor = 0

        # steps available for searching and opening before the questions: the questions' steps (and a spare one for
        # a re-asked question) and the ending's are kept apart, so every message the shopper allows gets sent
        free = self.steps_left() - questions - (1 if questions else 0) - self.end_reserve()
        per_item = max(2, free // max(n, 1)) if free > 0 else 0

        if any(st["moment"] == "start" for st in self.procedure):
            self.gate("start", "start the task (search the store)")
            self.handle_lines()
            questions = max(0, min(questions, (self.message_budget - self.messages_sent - self.held_messages())
                                   if self.message_budget is not None else questions))

        # first search and a first look at each item
        for k, item in enumerate(items):
            share = per_item if k < n - 1 else max(per_item, self.steps_left() - questions - 1 - self.end_reserve())
            if share <= 0:
                break
            before = self.steps_left()
            self.explore(item, max(2, share // 2))
            self.warm_up(item, max(0, share - (before - self.steps_left())))

        # questions to the shopper
        if questions > 0 and self.can_ask():
            # several products share few messages: each message names two properties of one product. A need among them
            # is still told most of the time, and a request to take them in turn says both are on the shopper's list
            width = 1 if SINGLE_ASK else (2 if n > 1 else 1)
            plan = self.plan_questions(questions * width)
            alone: list[tuple[Item, str]] = []      # the two of a message the shopper wants asked one at a time

            def partner(item: Item, prop: str, rest: list[tuple[Item, str]]) -> str | None:
                if width < 2:
                    return None
                return next((p for it, p in rest if it is item and not _near(p, prop) and _similar(p, prop) < 0.5), None)

            def put(item: Item, prop: str, also: str | None) -> bool:
                before = len(item.requirements)
                answer = self.ask(item, prop, also=also)
                if isinstance(answer, dict) and str(answer.get("kind") or "").lower() == "split":
                    if also:
                        alone[:0] = [(item, prop), (item, also)]
                    elif self.can_ask() and self.steps_left() > self.end_reserve():
                        # the message read as more than one subject to the shopper: ask again at once, narrowed
                        answer = self.ask(item, prop, narrow=True)
                self.apply_reply_kept(item, prop, answer)
                self.handle_lines()
                return len(item.requirements) != before

            while plan or alone:
                if not self.can_ask() or self.steps_left() <= self.end_reserve():
                    break
                single = bool(alone)
                item, prop = alone.pop(0) if single else plan.pop(0)
                also = None if single else partner(item, prop, plan)
                if also:
                    plan.remove((item, also))
                if also and (any(_similar(r.prop, also) >= 0.67 for r in item.requirements) or self.told_already(item, also)):
                    also = None
                if any(_similar(r.prop, prop) >= 0.67 for r in item.requirements) or self.told_already(item, prop):
                    if not also:
                        continue
                    prop, also = also, None
                if put(item, prop, also) and plan:
                    # a need was learned: the remaining questions skip its property and its other spellings
                    left = min(len(plan), (self.message_budget - self.messages_sent - self.held_messages())
                               if self.message_budget is not None else len(plan))
                    plan = self.plan_questions(left * width)
            # spend any messages left on further properties (every hidden need must be heard)
            while self.can_ask() and self.steps_left() > self.end_reserve() + 1 and self.time_left() > WRAP_SECONDS + 30:
                more = self.plan_questions(width * n if width > 1 else 1)
                if not more:
                    break
                item, prop = more[0]
                put(item, prop, partner(item, prop, more[1:]))

        # reference chains still open (stated in the request or told in a reply)
        for item in items:
            for req in list(item.requirements):
                if req.reference and req.op in REF_OPS:
                    self.ref_floor = self.end_reserve() + 2
                    self.resolve_reference(item, req)
                    self.ref_floor = 0

        if self.mode == "ask":
            return

        self.provisional_due = True
        self.provisional_cart()

        # judge what is open against everything known now, search more where nothing fits (items without a fit first)
        self.judge_all()
        order = sorted(items, key=lambda it: (bool(it.fits()), -len(it.pool)))
        for k, item in enumerate(order):
            share = max(2, (self.steps_left() - self.end_reserve()) // max(len(order) - k, 1))
            self.explore(item, share, final=True)

        self.choose_and_finish()

    def provisional_cart(self) -> None:
        """The best pick known so far goes in the cart before the last searching. A store announces its changes (a sold
        out or repriced pick, a dropped need, a new budget) once something is carted, so they arrive while steps
        remain to act on them; the cart is set to the final picks before the checkout (no extra step when the pick
        holds)."""
        if getattr(self, "provisional_done", False) or self.cart or not (self.roles.get("cart_add") or self.roles.get("cart_set")):
            return
        if self.steps_left() - self.end_reserve() < 3 or self.time_left() < WRAP_SECONDS + 30:
            self.provisional_done = True
            return
        # with no fit known yet the cart stays empty and the first fit found later goes in (explore calls again)
        self.judge_all()
        for item in self.items:
            ranked = self.rank_best(item, self.buyable(item))
            if not ranked:
                continue
            pick = ranked[0]
            if self.cap is not None and pick.price is not None and pick.price * item.quantity > self.cap + 1e-6:
                continue
            self.provisional_done = True
            if self.cart_add(item, pick):
                self.note("provisional_cart", item=item.name, id=pick.id)
            return

    def warm_up(self, item: Item, allowance: int | None = None) -> None:
        """Open more listings of the item's kind so the property menu is representative (about a dozen when the
        store opens several per step), within a step allowance."""
        if self.steps_left() <= self.end_reserve() + 2 or "message" not in self.roles:
            return
        stop_at = self.steps_left() - (allowance if allowance is not None else self.steps_left())
        floor = max(stop_at, self.end_reserve() + 2)
        target = 12 if self.batch_open() else 5

        def spare() -> list[Listing]:
            out = [l for l in item.pool.values() if not l.opened and self.category_ok(item, l) and l.stock is not False]
            out.sort(key=lambda l: (l.screen == "no", l.screen != "yes", l.price if l.price is not None else 1e18))
            return out

        while self.steps_left() > floor and self.time_left() > WRAP_SECONDS + 20 and not self.spent():
            opened = [l for l in item.pool.values() if l.opened and self.category_ok(item, l)]
            want = target - len(opened)
            if want <= 0:
                return
            todo = spare()
            if len(todo) < 2 and item.search_done < len(item.terms) + 1:
                before = len(item.pool)
                self.search(item, query=item.category, max_pages=1, budget=floor)
                todo = spare()
                if len(item.pool) == before and not todo:
                    return
            if not todo:
                return
            chunk = min(want, self.compare_limit if self.batch_open() else 4 if self.thin_compare else 1)
            got = self.open_batch(item, [l.id for l in todo[:chunk]], floor=floor)
            if not got:
                return
            self.judge(item, [l for l in self.candidates(item) if l.opened])

    def explore(self, item: Item, steps: int, final: bool = False) -> None:
        """Search and open listings for an item until fits are known, within a step allowance."""
        stop_at = self.steps_left() - steps
        terms = list(item.terms)
        term_index = min(item.search_done, len(terms))
        want_fits = 2 if not (self.cheapest or self.best) else 3 if self.cheapest else 4
        while self.steps_left() > max(stop_at, self.end_reserve()) and self.time_left() > WRAP_SECONDS + 10 and not self.spent():
            self.handle_lines()
            cands = self.candidates(item)
            self.screen(item, cands)
            cands = self.candidates(item)
            unopened = [l for l in cands if not l.opened]
            self.judge(item, [l for l in cands if l.opened])
            fits = item.fits()
            if final and fits and not self.cart and getattr(self, "provisional_due", False):
                self.provisional_cart()
            # for the cheapest rule every unopened candidate cheaper than the best fit must be seen
            if fits and not self.cheapest and len(fits) >= want_fits:
                break
            groups = self.value_groups(item)
            if repr(groups) != getattr(item, "value_sig", None):
                # the values changed (a need was heard): what the earlier value queries showed no longer covers them
                item.value_sig = repr(groups)
                item.value_ids = set()
                item.value_complete = False
            if groups:
                # rows that carry the required values (in their own words, or shown by a query made of them) open
                # first; when none of the rows left carries them all, a query made of the values comes first
                def hits(l: Listing) -> int:
                    if l.id in item.value_ids:
                        return len(groups)
                    n = self.row_hits(l, groups)
                    if self.all_words and n < len(groups):
                        # a store that wants every word showed the row for a query holding values: it carries them
                        sent = [_words(s.split(":", 1)[1]) for s, ids in item.query_ids.items() if l.id in ids]
                        n = max(n, sum(1 for g in groups if any(_words(a) <= q for a in g for q in sent)))
                    return n
                unopened.sort(key=lambda l: (l.screen == "no", -hits(l), l.screen != "yes",
                                             l.price if l.price is not None else 1e18))
                due = unopened
                if fits and self.cheapest:
                    # for the cheapest only rows cheaper than the best fit would still be opened
                    top = fits[0].price if fits[0].price is not None else 1e18
                    due = [l for l in unopened if l.price is not None and l.price < top]
                full = [l for l in due if hits(l) == len(groups)]
                if not full and self.value_search(item, max(stop_at, self.end_reserve())):
                    continue
            complete = bool(groups) and self.all_words and getattr(item, "value_complete", False)
            if fits and self.cheapest:
                best = fits[0].price if fits[0].price is not None else 1e18
                unopened = [l for l in unopened if (l.price is not None and l.price < best)]
                if complete:
                    # every listing carrying the required values came back from the value query: the others would
                    # not fit, so they need no look
                    unopened = [l for l in unopened if l.id in item.value_ids or l.screen == "yes"]
                if not unopened and self.need_search(item, max(stop_at, self.end_reserve())):
                    continue          # the heard value's own queries before a sweep of everything cheaper
                if not unopened and best < 1e17 and not complete and self.roles.get("search_fields") and (self.roles["search_fields"].get("price_max")) \
                        and getattr(item, "sweep_below", None) != best and self.steps_left() > max(stop_at, self.end_reserve()) + 2:
                    # a field search capped at the best price: only cheaper listings come back
                    item.sweep_below = best
                    fields = {"category": item.category, "price_max": max(0.0, best - 0.01)}
                    self.search(item, fields=fields, max_pages=3, budget=max(stop_at, self.end_reserve()))
                    continue
                if not unopened and not complete and getattr(item, "capped", False) and not getattr(item, "narrowed", False) \
                        and self.steps_left() > max(stop_at, self.end_reserve()) + 2:
                    # the store hid some matches: a query with every required value shows fewer, all of them
                    item.narrowed = True
                    values = [r.value.split("|")[0].strip() for r in item.active() if r.op in ("is", "one of") and r.value]
                    if values:
                        for q in (" ".join(values + [item.category]), " ".join(values)):
                            if self.steps_left() <= max(stop_at, self.end_reserve()) + 1:
                                break
                            self.search(item, query=q, max_pages=2, budget=max(stop_at, self.end_reserve()))
                        continue
                if not unopened:
                    # the cheapest needs every match seen: rows show their prices, so paging on costs a step per
                    # page and only rows cheaper than the best fit get opened
                    main = next((q for q in terms if item.cursors.get("search_text:" + q)
                                 and not item.cursors["search_text:" + q].get("done")), None)
                    if main and self.steps_left() > max(stop_at, self.end_reserve()) + 2:
                        before = self.step
                        self.search(item, query=main, max_pages=1, budget=max(stop_at, self.end_reserve()) + 1)
                        if self.step > before:
                            continue
                if not unopened and item.search_done >= min(len(terms), 2):
                    break
            if unopened:
                # one page a step: on orders of three or more products steps are many and the task's time is short,
                # so more pages go to each judging call
                batch = unopened[: self.compare_limit if self.batch_open() else 4 if self.thin_compare or len(self.items) >= 3 else 2]
                got = self.open_batch(item, [l.id for l in batch], floor=max(stop_at, self.end_reserve()))
                if not got:
                    break
                continue
            # nothing left to open: search more
            # revealed needs as keywords, with the stated values: they narrow the search well
            if self.need_search(item, max(stop_at, self.end_reserve())):
                continue
            if term_index < len(terms):
                query = terms[term_index]
                term_index += 1
                pages = 6 if len(self.items) == 1 else 3
                new = self.search(item, query=query, max_pages=pages, budget=max(stop_at, self.end_reserve()))
                if new == 0 and term_index >= len(terms):
                    item.exhausted = True
                continue
            if self.roles.get("search_fields"):
                fields = {"category": item.category}
                if self.cap is not None:
                    fields["price_max"] = self.cap / max(item.quantity, 1)
                if item.price_cap is not None:
                    fields["price_max"] = min(fields.get("price_max", item.price_cap), item.price_cap)
                sig = "search_fields:" + json.dumps(fields, sort_keys=True)
                if not (item.cursors.get(sig) or {}).get("done"):
                    self.search(item, fields=fields, max_pages=2, budget=max(stop_at, self.end_reserve()))
                    continue
            # keep paging the most specific keyword query while steps remain
            resumed = False
            for q in terms[:2]:
                sig = "search_text:" + q
                if not (item.cursors.get(sig) or {}).get("done"):
                    if self.search(item, query=q, max_pages=2, budget=max(stop_at, self.end_reserve())) > 0:
                        resumed = True
                        break
            if resumed:
                continue
            if not getattr(item, "suggested", False) and self.steps_left() > max(stop_at, self.end_reserve()) + 1:
                item.suggested = True
                extra = self.suggest_queries(item)
                if extra:
                    item.terms += extra
                    terms += extra
                    continue
            item.exhausted = True
            break

    def need_search(self, item: Item, floor: int) -> bool:
        """Once per item, the heard value as keywords: with the category plus a stated value, with the category, and
        alone (a store matching any word ranks a value best alone: the longer queries share their other words with
        every listing of the category; one that wants every word was served by the value queries). True when a
        search ran."""
        needs = [r.value.split("|")[0].strip() for r in item.active() if r.origin == "shopper" and r.value and r.op in ("is", "one of")]
        stated = [r.value.split("|")[0].strip() for r in item.active() if r.origin != "shopper" and r.value and r.op in ("is", "one of")]
        if not needs or getattr(item, "need_search_done", False) or self.all_words is True:
            return False
        item.need_search_done = True
        sent = [frozenset(_words(k.split(":", 1)[1])) for k in item.cursors if k.startswith("search_text:")]
        ran = False
        for q in ([f"{item.category} {stated[0]} {needs[0]}"] if stated else []) + [f"{item.category} {needs[0]}", needs[0]]:
            if self.steps_left() <= floor:
                break
            if frozenset(_words(q)) in sent:
                continue
            ran = True
            self.search(item, query=q, max_pages=3, budget=floor)
        return ran

    def suggest_queries(self, item: Item) -> list[str]:
        """New keyword queries for an item whose own ones found too few fits (one cheap model call)."""
        seen = sorted(item.pool.values(), key=lambda l: (not l.opened, l.id))[:14]
        rows = [{"title": l.title[:110], "fits": (l.id in {f.id for f in item.fits(include_unsure=True)}) if l.opened else None}
                for l in seen]
        tried = sorted({k.split(":", 1)[1] for k in item.cursors if k.startswith("search_text:")})
        limit = (self._schema("search_text").get("properties") or {}).get((self.roles.get("search_text") or {}).get("query") or "", {}).get("maxLength")
        user = (f"PRODUCT: {item.name} (category: {item.category})\nREQUIREMENTS:\n- " + ("\n- ".join(item.req_texts()) or "none")
                + f"\nTRIED: {json.dumps(tried, ensure_ascii=False)}\nLISTINGS SEEN: {json.dumps(rows, ensure_ascii=False)}"
                + (f"\nMAX QUERY LENGTH: {limit} characters" if limit else ""))
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], QUERY_PROMPT, user, 700)
        out: list[str] = []
        if isinstance(answer, dict):
            for q in answer.get("queries") or []:
                q = str(q).strip()
                if q and len(q) <= 60 and _norm(q) not in {_norm(t) for t in tried + item.terms + out}:
                    out.append(q)
        self.note("suggested", item=item.name, queries=out[:5])
        return out[:5]

    def rank_best(self, item: Item, cands: list[Listing]) -> list[Listing]:
        """The fits ordered by the request's own measure of best (a highest or lowest value), one model call;
        unchanged when the request names no such measure."""
        if not self.best or len(cands) < 2:
            return cands
        cached = getattr(self, "best_order", {}).get(item.index)
        ids = [l.id for l in cands]
        if cached is None or cached[0] != ids:
            rows = [{"id": l.id, "title": l.title[:120], "price": l.price, "properties": l.facts,
                     "description": str(l.text or "")[:300]} for l in cands[:12]]
            user = f"MEASURE: {self.best}\nPRODUCT: {item.name}\nLISTINGS: " + _compact(rows, 9000)
            answer = self.llm.json([PLAN_MODEL, BACKUP_MODEL], RANK_PROMPT, user, 600)
            order = [str(x) for x in (answer.get("order") or [])] if isinstance(answer, dict) else []
            self.note("best_rank", measure=self.best, order=order[:12])
            cached = (ids, order)
            if order:
                self.best_order = dict(getattr(self, "best_order", {}))
                self.best_order[item.index] = cached
        order = cached[1]
        pos = {lid: i for i, lid in enumerate(order)}
        return sorted(cands, key=lambda l: pos.get(l.id, len(pos) + ids.index(l.id)))

    def read_backups(self) -> None:
        """Cheapest tasks, before the first cart add: the cheapest rows not read yet, priced up to the second fit known,
        are read in one batch per item (with the step kept for it), so a pick that sells out or changes price when it
        is carted has a checked replacement."""
        self.backups_done = True        # the step kept for this is free from here on
        if not all(self.buyable(it) for it in self.items):
            return                      # an item with no fit yet needs those steps more
        for item in self.items:
            if self.steps_left() - self.end_reserve() < 1 or self.time_left() < WRAP_SECONDS + 20:
                break
            fits = self.buyable(item)
            bound = fits[1].price if len(fits) > 1 else (fits[0].price if fits else None)
            rows = [l for l in self.candidates(item) if not l.opened and l.screen != "no" and l.stock is not False
                    and l.price is not None and (bound is None or l.price <= bound)
                    and not (self.cap is not None and l.price * item.quantity > self.cap + 1e-6)]
            rows.sort(key=lambda l: l.price)
            if not rows:
                continue
            got = self.open_batch(item, [l.id for l in rows[:self.compare_limit]], floor=self.end_reserve())
            self.note("backups", item=item.name, read=[l.id for l in got], bound=bound)
            if got:
                self.judge(item, got)

    def choose_and_finish(self) -> None:
        self.handle_lines()
        if self.cheapest and not getattr(self, "backups_done", False):
            self.read_backups()
        self.judge_all()
        chosen: dict[int, Listing] = {}
        pairs = self.fit_pairs()
        hosts = {h.index for _, h in pairs}
        accessories = {a.index: h for a, h in pairs}
        conditions = getattr(self, "conditions", [])
        sources = {c["src"].index for c in conditions}
        dependents = {d.index for c in conditions for _, d, _ in c["branches"]}
        # the first pick of every item whose choice does not wait for another item is checked concurrently
        tops = [(it, (self.rank_best(it, self.buyable(it)) or [None])[0]) for it in self.items
                if it.index not in accessories and it.index not in dependents]
        _parallel([lambda it=it, c=c: self.verify(it, c) for it, c in tops if c is not None])
        # a host is chosen before the accessories that must fit it, an item a condition looks at before the items
        # whose requirements depend on it
        for item in sorted(self.items, key=lambda it: (it.index in accessories or it.index in dependents,
                                                       it.index not in hosts and it.index not in sources, it.index)):
            if item.index in accessories and accessories[item.index].index in chosen:
                self.bind_fit(item, accessories[item.index], chosen[accessories[item.index].index])
            if item.index in dependents:
                self.apply_conditions(item, chosen)
            tries = 0
            for cand in self.rank_best(item, self.buyable(item)):
                if tries >= 3 or self.time_left() < WRAP_SECONDS:
                    break
                tries += 1
                if self.verify(item, cand):
                    chosen[item.index] = cand
                    break
            if item.index not in chosen:
                rest = self.rank_best(item, self.buyable(item))
                if rest and (tries >= 3 or self.time_left() < WRAP_SECONDS):
                    chosen[item.index] = rest[0]
            if item.index in accessories and item.index not in chosen:
                # no accessory fits that host: another host that some accessory fits, when there is one
                host = accessories[item.index]
                for other in self.buyable(host)[:3]:
                    if chosen.get(host.index) is other or self.time_left() < WRAP_SECONDS:
                        continue
                    self.bind_fit(item, host, other)
                    got = self.buyable(item)
                    if got:
                        chosen[host.index] = other
                        chosen[item.index] = got[0]
                        break
            if item.index in dependents and item.index not in chosen:
                # nothing fits what the condition asks for that pick: another pick of the condition's source with
                # the other outcome, unless an accessory already chosen must fit it
                for cond in [c for c in conditions if c["src"].index in chosen and any(d is item for _, d, _ in c["branches"])]:
                    src = cond["src"]
                    if any(h is src and a.index in chosen for a, h in pairs) or self.time_left() < WRAP_SECONDS:
                        continue
                    now = self.condition_holds(cond, chosen[src.index])
                    for other in self.buyable(src)[:4]:
                        if other is chosen[src.index] or self.condition_holds(cond, other) == now:
                            continue
                        was = chosen[src.index]
                        chosen[src.index] = other
                        self.apply_conditions(item, chosen)
                        got = self.buyable(item)
                        if got and self.verify(src, other):
                            chosen[item.index] = got[0]
                            break
                        chosen[src.index] = was
                        self.apply_conditions(item, chosen)
                    if item.index in chosen:
                        break
        missing = [it for it in self.items if it.index not in chosen]
        if missing and not getattr(self, "last_chance", False) and self.time_left() > WRAP_SECONDS + 20 and not self.spent():
            # nothing to order for these yet: the steps kept for the ending are better spent on one more search,
            # keeping only what a cart, the shopper's yes and the checkout need
            self.last_chance = True
            self.reserve_override = len(self.items) + 3 + max(self.held_messages(), 1 if self.confirm_first else 0)
            try:
                for it in missing:
                    if self.steps_left() - self.end_reserve() >= 3:
                        it.exhausted = False
                        self.explore(it, self.steps_left() - self.end_reserve(), final=True)
            finally:
                self.reserve_override = None
            if any(self.buyable(it) for it in missing):
                self.choose_and_finish()
                return
        # (re-verifying a listing the audit names costs no step; its reads and searches keep the ending's steps)
        if missing and not getattr(self, "audited", False) and self.time_left() > WRAP_SECONDS + 25 and not self.spent():
            self.audited = True
            if self.audit(missing) and all(self.buyable(it) for it in missing):
                self.choose_and_finish()
                return
        self.near_picks = dict(chosen)
        if missing:
            self.decline_reason = "no fitting listing for " + ", ".join(it.name for it in missing)
            self.note("no_fit", items=[it.name for it in missing],
                      pools={it.name: len(it.pool) for it in missing})
            self.decline(self.decline_reason)
            return
        # keep the total within the budget
        if self.cap is not None:
            def total() -> float:
                return sum((chosen[it.index].price or 0) * it.quantity for it in self.items)
            guard = 0
            while total() > self.cap + 1e-6 and guard < 10:
                guard += 1
                moved = False
                for it in sorted(self.items, key=lambda i: -((chosen[i.index].price or 0) * i.quantity)):
                    fits = [f for f in it.fits(cheapest=True) if f.stock is not False]
                    pos = [f.id for f in fits].index(chosen[it.index].id) if chosen[it.index].id in [f.id for f in fits] else -1
                    if 0 <= pos < len(fits) - 1:
                        chosen[it.index] = fits[pos + 1]
                        moved = True
                        break
                if not moved:
                    break
            if total() > self.cap + 1e-6:
                self.decline_reason = "the cheapest fitting combination is over the budget"
                self.note("over_budget", total=total(), cap=self.cap)
                self.decline(self.decline_reason)
                return
        for item in self.items:
            item.chosen = chosen[item.index]
        self.place_order()

    def audit(self, missing: list[Item]) -> bool:
        """Before a decline: one independent look over everything seen for each item still without a fit (read
        listings that may fit after all, rows never read, searches never run), followed up by the code. True when
        something new was learned."""
        did = False
        self.reserve_override = len(self.items) + 3 + max(self.held_messages(), 1 if self.confirm_first else 0)
        try:
            for item in missing:
                if self.time_left() < WRAP_SECONDS + 20:
                    break
                cards = []
                pool = [l for l in item.pool.values() if self.category_ok(item, l) and l.stock is not False
                        and not (self.cap is not None and l.price is not None and l.price * item.quantity > self.cap + 1e-6)]
                for l in sorted(pool, key=lambda l: (not l.opened, l.price if l.price is not None else 1e18))[:40]:
                    card: dict[str, Any] = {"id": l.id, "title": l.title[:110], "price": l.price, "read": l.opened}
                    if l.opened:
                        v = item.verdicts.get(l.id) or {}
                        card["failed"] = sorted(v.get("failed") or [])[:3]
                        card["unsure"] = sorted(v.get("unsure") or [])[:3]
                        card["properties"] = dict(list(l.facts.items())[:16])
                        if l.notes:
                            card["notes"] = l.notes[:200]
                    else:
                        card["text"] = (l.text or "")[:160]
                    cards.append(card)
                if not cards:
                    continue
                tried = sorted({k.split(":", 1)[1] for k in item.cursors if k.startswith("search_text:")})
                user = (f"PRODUCT: {item.name} (category: {item.category}; quantity {item.quantity})\nBUDGET, ALL LINES TOGETHER: "
                        f"{self.cap if self.cap is not None else 'none stated'}\nREQUIREMENTS:\n- " + ("\n- ".join(item.req_texts()) or "none")
                        + f"\nQUERIES TRIED: {json.dumps(tried, ensure_ascii=False)}\n\nLISTINGS SEEN:\n" + _compact(cards, 14000))
                answer = self.llm.json([PLAN_MODEL, FAST_MODEL], AUDIT_PROMPT, user, 1500)
                if not isinstance(answer, dict):
                    continue
                self.note("audit", item=item.name, answer=_compact(answer, 300))
                for lid in [str(x) for x in answer.get("fits_after_all") or []][:2]:
                    l = item.pool.get(lid)
                    if l is None or not l.opened or self.field_fails(item, l):
                        continue
                    if self._verify(item, l, strict=True):
                        # a second, independent reading agrees: the earlier verdict is set aside
                        item.verdicts[l.id] = {"failed": set(), "unsure": set(), "weak": set(), "loose": set()}
                        item.judged_for[l.id] = set(item.req_texts())
                        did = True
                fresh = [str(x) for x in answer.get("unopened") or [] if str(x) in item.pool and not item.pool[str(x)].opened][:4]
                if fresh and self.steps_left() - self.end_reserve() >= 1:
                    if self.open_batch(item, fresh, floor=self.end_reserve()):
                        did = True
                for q in [str(x).strip() for x in answer.get("queries") or [] if str(x).strip()][:2]:
                    if self.steps_left() - self.end_reserve() < 3 or self.time_left() < WRAP_SECONDS + 15:
                        break
                    if _norm(q) in {_norm(t) for t in tried} or self.search(item, query=q[:60], max_pages=1, budget=self.end_reserve()) <= 0:
                        continue
                    rows = [l for l in self.candidates(item) if not l.opened]
                    self.screen(item, rows)
                    rows = [l for l in self.candidates(item) if not l.opened and l.screen != "no"][:2]
                    if rows and self.open_batch(item, [l.id for l in rows], floor=self.end_reserve()):
                        did = True
                self.judge(item, [l for l in self.candidates(item) if l.opened])
        finally:
            self.reserve_override = None
        return did

    def item_named(self, name: Any) -> Item | None:
        name = str(name or "")
        if not name or not self.items:
            return None
        best = max(self.items, key=lambda it: max(_similar(it.name, name), _similar(it.category, name)))
        return best if max(_similar(best.name, name), _similar(best.category, name)) >= 0.2 else None

    def read_conditions(self, raw: Any) -> list[dict[str, Any]]:
        """The request's requirements that hold on one item only when another item turns out a certain way, as
        {"src": item, "prop", "values", "branches": [(holds?, item, requirement spec)]}."""
        out = []
        for c in raw if isinstance(raw, list) else []:
            cond = c.get("if") if isinstance(c, dict) else None
            if not isinstance(cond, dict) or not cond.get("property"):
                continue
            src = self.item_named(cond.get("item"))
            values = cond.get("values")
            values = [str(v) for v in values if v] if isinstance(values, list) else [str(values)] if values else []
            branches = []
            for key, when in (("then", True), ("otherwise", False)):
                b = c.get(key)
                if isinstance(b, dict) and b.get("property") and b.get("value"):
                    dst = self.item_named(b.get("item"))
                    if dst is not None and dst is not src:
                        branches.append((when, dst, b))
            if src is None or not values or not branches:
                continue
            out.append({"src": src, "prop": str(cond["property"]), "values": values, "branches": branches})
            for _, dst, b in branches:
                # the values a branch may require are searched for too, so listings carrying them are at hand
                noun = dst.noun
                for v in str(b["value"]).split("|"):
                    v = v.strip()
                    if v and len(v) <= 40 and not any(_norm(v) in _norm(t) for t in dst.terms):
                        dst.terms.append(f"{v} {noun}"[:60])
        if out:
            self.note("conditions", rules=[(c["src"].name, c["prop"], c["values"], [(w, d.name, b.get("property"), b.get("value"))
                                                                                   for w, d, b in c["branches"]]) for c in out])
        return out

    def condition_holds(self, cond: dict[str, Any], listing: Listing) -> bool | None:
        """Whether the listing's value of the condition's property is one of its values: the property table decides
        when it plainly carries a value or plainly does not; otherwise one fast model call over the records."""
        cache = self.__dict__.setdefault("held_cache", {})
        key = (id(cond), listing.id, json.dumps(listing.facts, sort_keys=True, default=str))
        if key in cache:
            return cache[key]
        own = [str(v) for k, v in listing.facts.items()
               if _near(str(k).replace("_", " "), cond["prop"]) or _similar(str(k).replace("_", " "), cond["prop"]) >= 0.5]
        holds: bool | None = None
        if own:
            text = _words(" ".join(own))
            wants = [_words(v) for v in cond["values"] if _words(v)]
            if any(w <= text for w in wants):
                holds = True
            elif not any(w & text for w in wants):
                holds = False
        if holds is None:
            user = (f"PROPERTY: {cond['prop']}\nVALUES: {json.dumps(cond['values'], ensure_ascii=False)}\n\nLISTING:\n"
                    + json.dumps(listing.record(2500), ensure_ascii=False))
            answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], COND_PROMPT, user, 300)
            if isinstance(answer, dict) and isinstance(answer.get("holds"), bool):
                holds = answer["holds"]
        cache[key] = holds
        self.note("condition", src=cond["src"].name, listing=listing.id, prop=cond["prop"], holds=holds)
        return holds

    def apply_conditions(self, item: Item, chosen: dict[int, Listing]) -> None:
        """The item's requirements from the conditions whose other item is chosen, and its listings judged again."""
        mine = [c for c in self.conditions if any(d is item for _, d, _ in c["branches"])]
        if not mine:
            return
        wanted = []
        for cond in mine:
            pick = chosen.get(cond["src"].index)
            holds = self.condition_holds(cond, pick) if pick is not None else None
            if holds is None:
                continue
            for when, dst, b in cond["branches"]:
                if dst is item and when == holds:
                    wanted.append(Requirement(b["property"], b.get("op") or "is", b["value"], origin="condition"))
        old = sorted(r.text() for r in item.requirements if r.origin == "condition")
        if old == sorted(r.text() for r in wanted):
            return
        item.requirements = [r for r in item.requirements if r.origin != "condition"] + wanted
        self.note("condition_applied", item=item.name, reqs=[r.text() for r in wanted])
        self.judge(item, [l for l in self.candidates(item) if l.opened])
        if not self.buyable(item) and self.steps_left() - self.end_reserve() >= 3 and self.time_left() > WRAP_SECONDS + 20:
            item.exhausted = False
            self.explore(item, self.steps_left() - self.end_reserve(), final=True)

    def fit_pairs(self) -> list[tuple[Item, Item]]:
        """The request's "make sure X fits Y" rules as (accessory item, host item)."""
        out = []
        for rule in self.fit_rules:
            acc, host = str(rule.get("accessory") or ""), str(rule.get("host") or "")
            if not acc or not host or len(self.items) < 2:
                continue
            a = max(self.items, key=lambda it: _similar(it.name, acc))
            h = max(self.items, key=lambda it: _similar(it.name, host))
            if a is not h and _similar(a.name, acc) >= 0.2 and _similar(h.name, host) >= 0.2 and (a, h) not in out:
                out.append((a, h))
        return out

    def bind_fit(self, item: Item, host_item: Item, host: Listing) -> None:
        """The accessory item must work with the host listing chosen: a requirement quoting the host's page, the
        pages its candidates point to for more of their compatibility looked up, and the candidates judged again."""
        page = " ".join(x for x in [host.title[:120], (host.text or "")[:300], (host.notes or "")[:150]] if x)
        item.requirements = [r for r in item.requirements if r.origin != "fit"]
        item.requirements.append(Requirement("compatibility", "is", f"works with the {host_item.name} chosen, whose page reads: {page}",
                                             origin="fit"))
        code = re.compile(r"\b[A-Z]{1,6}-?\d{3,}[A-Z0-9]*\b")
        titled = {c for it in self.items for l in it.pool.values() for c in code.findall(l.title or "")}
        prefixes = {re.match(r"[A-Z]+", c).group(0) for l in item.pool.values() for c in code.findall(l.title or "")}
        wanted: list[str] = []
        for l in self.candidates(item):
            if not l.opened or (item.verdicts.get(l.id) or {}).get("failed"):
                continue
            for c in code.findall(" ".join([l.text or "", l.notes or ""])):
                if c not in titled and c not in wanted and re.match(r"[A-Z]+", c).group(0) in prefixes:
                    wanted.append(c)
        # a short code on the host's page that starts some accessory codes ("AB3" for "AB3-1234") names the series of
        # parts that fit it
        host_text = " ".join([host.text or "", host.notes or "", json.dumps(host.facts, ensure_ascii=False)])
        heads = {h for it in self.items if it is not host_item for l in it.pool.values() for h in _SERIES.findall(l.title or "")}
        series = []
        for c in re.findall(r"\b[A-Z]{1,5}\d{1,3}\b", host_text):
            family = re.match(r"[A-Z]+", c).group(0)
            if c not in series and c not in _SERIES.findall(host.title or "") and \
                    (c in heads or any(re.match(r"[A-Z]+", h).group(0) == family for h in heads)):
                series.append(c)
        for c in series[:2] + [w for w in wanted if w not in series][:3]:
            if self.steps_left() <= self.end_reserve() + 2 or self.time_left() < WRAP_SECONDS + 10:
                break
            query = f"{c} {item.category}" if c in series else c
            self.search(item, query=query, max_pages=1, budget=self.end_reserve())
        item.fit_series = series[:2]
        if series and self.steps_left() > self.end_reserve() + 2:
            # the accessories of that series are read before judging
            mine = [l for l in self.candidates(item) if not l.opened and self.in_series(l, series)]
            if mine:
                self.open_batch(item, [l.id for l in mine[: self.compare_limit if self.batch_open() else 3]],
                                floor=self.end_reserve() + 1)
        self.note("fit_bound", item=item.name, host=host.id, series=series[:2], looked_up=wanted[:3])
        self.judge(item, [l for l in self.candidates(item) if l.opened])
        if series:
            fit_req = item.requirements[-1].text()
            for l in self.candidates(item):
                v = item.verdicts.get(l.id)
                if v and l.opened and self.in_series(l, series):
                    # the listing's own part code is of the series the host's page names: the fit is stated
                    for key in ("unsure", "weak", "failed"):
                        if key in v:
                            v[key].discard(fit_req)

    @staticmethod
    def in_series(listing: Listing, series: list[str]) -> bool:
        return any(h in series for h in _SERIES.findall(listing.title or ""))

    def place_order(self, attempt: int = 0) -> None:
        # a re-choice starts a new round of attempts: the rounds are bounded too
        self.order_rounds = getattr(self, "order_rounds", 0) + 1
        if attempt > 2 or self.finished or self.order_rounds > 8:
            return
        # cart lines = exactly the chosen listings
        for lid in list(self.cart):
            if not any(it.chosen and it.chosen.id == lid for it in self.items):
                self.cart_remove(lid)
        for item in self.items:
            if item.chosen and item.chosen.id not in self.cart:
                if not self.cart_add(item, item.chosen):
                    if getattr(self, "cart_faulty", False):
                        # the tool refuses how it is called: another listing would fare no better
                        self.note("cart_faulty")
                        self.decline("the cart tool refuses its arguments")
                        return
                    item.verdicts.setdefault(item.chosen.id, {"failed": set(), "unsure": set()})["failed"].add("cart refused")
                    item.chosen = None
        if any(it.chosen is None for it in self.items):
            self.choose_and_finish()
            return
        # a look at the carted item after the add: changes announce themselves in the next result
        if self.sheet_first():
            # lines judged from their tables alone get their own pages read (stock, price and the seller's notes)
            for item in self.items:
                l = item.chosen
                if l is not None and not l.page_read and self.steps_left() > 2 + self.pending_messages():
                    self.open_batch(item, [l.id], page=True)
        elif self.steps_left() > 2 + self.pending_messages() and self.items and self.items[0].chosen:
            first = self.items[0].chosen
            self.open_batch(self.items[0], [first.id])
        changed = self.handle_lines()
        for item in self.items:
            # a record read after the add is judged again: a pick it contradicts is swapped before the checkout
            if item.chosen is not None and item.chosen.id not in item.judged_for:
                self.judge(item, [item.chosen])
                if (item.verdicts.get(item.chosen.id) or {}).get("failed"):
                    self.note("chosen_failed", id=item.chosen.id, failed=sorted(item.verdicts[item.chosen.id]["failed"])[:3])
                    changed = True
        if changed or self.price_or_stock_changed():
            self.note("rechoose", cart=self.cart)
            for item in self.items:
                self.judge(item, [l for l in self.candidates(item) if l.opened])
            self.rechoose()
            if self.finished:
                return
            self.place_order(attempt + 1)
            return
        if self.finish_order():
            return
        # refused: read the reasons and try again
        changed = self.handle_lines()
        if not changed:
            self.refresh_cart_items()
            self.handle_lines()
        self.rechoose()
        if not self.finished:
            self.place_order(attempt + 1)

    def price_or_stock_changed(self) -> bool:
        for item in self.items:
            l = item.chosen
            if l is None:
                continue
            if l.stock is False:
                return True
            if self.cap is not None and l.price is not None and sum((it.chosen.price or 0) * it.quantity for it in self.items if it.chosen) > self.cap + 1e-6:
                return True
        return False

    def rechoose(self) -> None:
        """After a change: re-pick each item's listing from the fits (cheapest rule respected), searching again first."""
        for item in self.items:
            self.judge(item, [l for l in self.candidates(item) if l.opened])
            if not [f for f in item.fits() if f.stock is not False] or self.cheapest:
                item.need_search_done = False
                item.sweep_below = None
                share = max(2, self.steps_left() - self.end_reserve())
                if share >= 3 and self.time_left() > WRAP_SECONDS + 20:
                    self.explore(item, share, final=True)
        for item in self.items:
            fits = [f for f in item.fits(cheapest=self.cheapest) if f.stock is not False]
            if not fits:
                item.chosen = None
                continue
            if self.cheapest:
                item.chosen = fits[0]
            elif item.chosen is None or item.chosen.id not in [f.id for f in fits]:
                item.chosen = fits[0]
        picks = {it.index: it.chosen for it in self.items if it.chosen is not None}
        for dst in {d.index: d for c in getattr(self, "conditions", []) for _, d, _ in c["branches"]}.values():
            # a new pick of a condition's source can change what the condition requires
            self.apply_conditions(dst, picks)
            ok = [f.id for f in self.buyable(dst)]
            if dst.chosen is not None and dst.chosen.id not in ok:
                dst.chosen = dst.pool[ok[0]] if ok else None
        if any(it.chosen is None for it in self.items):
            self.choose_and_finish()
            return
        if self.cap is not None:
            total = sum((it.chosen.price or 0) * it.quantity for it in self.items)
            if total > self.cap + 1e-6:
                self.choose_and_finish()

    def force_finish(self) -> None:
        """Out of time, steps or plan: finish with what is known (a fitting, buyable cart within the budget)."""
        if self.finished:
            return
        for item in self.items:
            if item.chosen is None or item.chosen.stock is False:
                fits = self.buyable(item)
                item.chosen = fits[0] if fits else None
        ready = all(it.chosen for it in self.items) and bool(self.items)
        if ready and self.cap is not None:
            ready = sum((it.chosen.price or 0) * it.quantity for it in self.items) <= self.cap + 1e-6
        if ready:
            stale = [lid for lid in self.cart if not any(it.chosen.id == lid for it in self.items)]
            adds = [it for it in self.items if it.chosen.id not in self.cart]
            if self.steps_left() >= len(stale) + len(adds) + 1:
                for lid in stale:
                    self.cart_remove(lid)
                for item in adds:
                    self.cart_add(item, item.chosen)
                if self.finish_order():
                    return
        if not self.finished:
            self.decline("forced")

    def buyable(self, item: Item) -> list[Listing]:
        """The item's fitting listings in choosing order, without those known to be sold out."""
        table = self.field_table(item)
        return [f for f in item.fits(cheapest=self.cheapest) if f.stock is not False and not self.field_fails(item, f, table)]


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    return Episode(problem_data).run()


__all__ = ["agent_main"]
