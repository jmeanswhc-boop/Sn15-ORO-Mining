"""Shopping agent for a simulated-store benchmark. One model call parses the request and tools into items,
requirements, budget, demanded steps and tool roles. It searches by keywords and required values, screens
rows, opens listings for a small model to judge, asks the shopper about likely hidden preferences, follows
requirements that point at other listings' pages, applies conditions and fit rules, and reacts to shopper
notes in results. Models only read text; the flow is plain code. It ends by carting verified picks and
checking out, or ordering the nearest pick or declining.
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
# a limited Sonnet 5.5 call goes to its older kin, once each (providers limit requests per model)
SONNET_KIN = ("anthropic/claude-sonnet-5", "anthropic/claude-sonnet-4.6")

TASK_SECONDS = 255.0
WRAP_SECONDS = 45.0
FACT_CHARS = 2200
ROW_CHARS = 260
SOFT_SPEND = 0.03
SPEND_CAP = 0.05
LIST_WIDTH = 6             # properties named by one list message (one product)
# a reply asking for one question at a time, in the words shoppers use for it
_REFUSED = re.compile(r"\b(?:one (?:thing|question|query|subject|topic|prompt|item|point|matter)s?\b|one at a time|"
                      r"one by one|one per (?:turn|message|go)|separately|separate (?:these|them|the|those)\b|"
                      r"split (?:them|it|these|those|that)\b|smaller (?:bites|pieces|steps|chunks)|"
                      r"(?:the )?(?:first|top) (?:thing|one|question|bit|matter)\b|in turn\b|piece by piece|bit by bit|"
                      r"singly\b|single file|in a (?:queue|trickle|line)\b|a trickle|park the rest|hold the (?:rest|others)|"
                      r"before the rest|too many (?:things|questions|at once)|in one go|pile of them|"
                      r"(?:stacked|piled|bundled|crammed|packed)\b.{0,25}\bquestions|slow (?:down|the pace)|all at once|"
                      r"ask one\b|one, then the next|then the next\b|break (?:that|it|them|these|those) up\b|"
                      r"space (?:them|these|those|it) out|in sequence|dribs and drabs|rather than a flood|first alone|"
                      r"leave the (?:remainder|rest)|first of those|pick (?:a|one) question)", re.I)
_CODE = re.compile(r"\b[A-Za-z]{2,}-?\d{3,}[A-Za-z]?\b|\b[A-Za-z]{2,}\d*-\d{2,}\b")
_SERIES = re.compile(r"\b([A-Z]{1,5}\d{1,3})-\d{2,}\b")
_POINTER = re.compile(r"\b(?:more|details?|info(?:rmation)?|see|read|look)\b[^.;:!?]{0,24}?\b(?:part|model|item|listing|"
                      r"product|page|code)\s*(?:no\.?|number|#)?\s*([A-Za-z0-9][A-Za-z0-9-]{2,})", re.I)
REF_OPS = {"same as reference": "is", "different from reference": "is not", "counterpart of reference": "is",
           "at least reference": "at least", "at most reference": "at most", "within reference": "at least"}

_llm = ProxyClient(timeout=75, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=90, max_retries=2, retry_delay=1)

_NOT_ASKED = {"option", "variant"}

_WORLD_MEMORY: dict[str, Any] = {}



def _norm(text: Any) -> str:
    """Lowercases any value (None gives an empty string), spells '&' as 'and', turns underscores and hyphens into
    spaces and every other run of non-alphanumerics into a space, then strips it. This is the common key used to
    compare names and values throughout.
    """
    text = str(text or "").lower().replace("&", " and ").replace("_", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def _tokens(text: Any) -> set[str]:
    """Splits the normalized text into a set of words of two or more characters. Words longer than three characters
    that end in 's' lose that letter, a crude singular form so plurals compare equal.
    """
    out = set()
    for t in _norm(text).split():
        if len(t) > 1:
            if len(t) > 3 and t.endswith("s"):
                t = t[:-1]
            out.add(t)
    return out


def _words(text: Any) -> set[str]:
    """Word set with the same crude plural folding as _tokens, but one-character words are kept so lone digits and
    letters still count. Used where a single figure such as a piece count matters.
    """
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in _norm(text).split()}


def _stems(text: Any) -> set[str]:
    """Set of four-character prefixes of the text's tokens, a rough stem comparison that tolerates different
    suffixes on the same word.
    """
    return {t[:4] for t in _tokens(text)}


def _near(a: Any, b: Any) -> bool:
    """Decides whether two property names denote the same property: true when their normalized forms are equal or
    when both have the same non-empty token set (order, separators and plural 's' ignored).
    """
    if _norm(a) == _norm(b):
        return True
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta) and ta == tb


def _similar(a: Any, b: Any) -> float:
    """Jaccard overlap of the two values' token sets, from 0.0 to 1.0. Either side having no tokens yields 0.0.
    Used to pick the closest enum member, key or label.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _json_in(text: str) -> Any:
    """Extracts structured data from a model reply. A ``` fenced block is preferred; then the first balanced {...}
    span, else [...] span, is located with string quoting and escapes respected and parsed. As a last resort the
    whole text is parsed. Empty or unparseable input gives None.
    """
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
    """Serializes a value as compact JSON (no spaces, non-ASCII kept) and, when longer than `limit` characters,
    cuts it there and appends an ellipsis character.
    """
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "…"


def _number(value: Any) -> float | None:
    """First numeric value carried by the input, as a float. Ints and floats pass through; strings have commas
    removed and their first signed decimal taken. Booleans, other types and number-free strings give None.
    """
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
    """Coerces a value to match one JSON-schema argument spec. Enums take an exact, normalized or >=0.5-overlap
    member; numbers are parsed, clamped to minimum/maximum and rounded for integers; strings are cut at
    maxLength on a word break and refused under minLength; booleans accept true/false text; arrays are wrapped
    and cut to maxItems. Returns _DROP when no agreeing value exists.
    """
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
    """Fallback value for a required argument when neither cache, schema default nor model supplied one: first enum
    member, 1 for numbers, False for booleans, 'n/a' for strings (passed through _fit_value limits), an empty
    list for arrays without minItems; _DROP for anything else.
    """
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
        return _fit_value(spec, "not given")
    return [] if kind == "array" and not spec.get("minItems") else _DROP


def _refused_cart(obs: Any) -> bool:
    """Decides whether a cart tool's response is a refusal: a dict with exactly one key whose value is a string
    containing none of the success words (ok, added, done, accepted, applied, updated, in cart). Everything else
    counts as not refused.
    """
    if not isinstance(obs, dict) or len(obs) != 1:
        return False
    only = next(iter(obs.values()))
    return isinstance(only, str) and not re.search(r"\b(ok|okay|success\w*|added|done|accepted|applied|updated|in (the |your )?(cart|basket))\b", only, re.I)


def _ref_sig(ref: Any) -> tuple:
    """Tuple key for a reference description built from normalized seller, normalized category and the sorted
    normalized non-empty attribute values; it lets the agent see that a page reference was already taken up.
    Non-dict input gives an empty tuple.
    """
    if not isinstance(ref, dict):
        return ()
    attrs = ref.get("attributes") if isinstance(ref.get("attributes"), dict) else {}
    return (_norm(ref.get("seller")), _norm(ref.get("category")), tuple(sorted(_norm(v) for v in attrs.values() if v)))


def _is_sentence(value: Any) -> bool:
    """True for a string longer than 24 characters containing a space, the test used to tell shopper prose from
    ids, codes and short field values in tool results.
    """
    return isinstance(value, str) and len(value) > 24 and " " in value.strip()


def _find_list_of_dicts(obs: Any, depth: int = 0) -> list[dict[str, Any]] | None:
    """Locates the record list in a tool response: the input itself when it is a non-empty list of dicts, otherwise
    the longest such list found among dict values searched up to two levels down. None when there is none.
    """
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
    """Separates prose a shopper slipped into a tool result from the data. Lists made only of sentences, and a lone
    sentence next to exactly one nested dict in a small wrapper with no id-like string values, become lines;
    other keys stay. When lines were found and one dict remains it is lifted out. Returns (data, lines).
    """
    lines: list[str] = []
    if isinstance(obs, str):
        return obs, lines
    if not isinstance(obs, dict):
        return obs, lines
    inner: dict[str, Any] = {}
    dict_siblings = sum(1 for v in obs.values() if isinstance(v, dict))
    record = any(isinstance(v, str) and v.strip() and " " not in v.strip() for v in obs.values())
    for k, v in obs.items():
        if isinstance(v, list) and v and all(_is_sentence(x) for x in v):
            lines.extend(str(x) for x in v)
        elif _is_sentence(v) and dict_siblings == 1 and len(obs) <= 3 and not record:
            lines.append(str(v))
        else:
            inner[k] = v
    if lines and len(inner) == 1 and isinstance(list(inner.values())[0], dict):
        inner = list(inner.values())[0]
    return inner, lines



def _parallel(jobs: list, workers: int = 3) -> list:
    """Thread-pool map over zero-argument callables, using at most `workers` threads and running a single job
    inline. Output order matches input order, and a callable that raises leaves None in its slot instead of
    failing the batch.
    """
    if len(jobs) <= 1:
        return [_safe(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(_safe, jobs))


def _safe(job: Any) -> Any:
    """Invokes a zero-argument callable and returns its result, swallowing any exception and returning None in that
    case.
    """
    try:
        return job()
    except Exception:
        return None


def _idish(value: Any) -> bool:
    """A text value that can be a listing id: 3-64 characters, no space, at least one digit."""
    return isinstance(value, str) and 3 <= len(value.strip()) <= 64 and " " not in value.strip() and bool(re.search(r"\d", value))


LINE_FIELD = re.compile(r"(?P<key>[A-Za-z_][\w .()/+-]{0,47}?):(?: (?P<value>.*))?", re.S)
PAIR_NAME = re.compile(r"[A-Za-z0-9][\w .()/+-]{0,48}")


def _plain_text(text: str) -> Any:
    """A field value written as text, in the type a JSON store would have sent."""
    value = text.strip()
    if value in ("True", "true") or value in ("False", "false"):
        return value[0] in "Tt"
    if value in ("null", "None"):
        return None
    number = re.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?", value)
    if number and len(value) <= 18:
        return float(value) if number.group(2) else int(value)
    return value


def _pair_map(text: str, least: int = 2) -> dict[str, str] | None:
    """Text of "name=value" parts joined by ";" as a map (a part with no "=" continues the value before it); None when
    fewer than `least` parts carry a name or the text does not start with one."""
    out: dict[str, str] = {}
    last = None
    for part in text.split(";"):
        if not part.strip():
            continue
        name, eq, value = part.partition("=")
        if eq and PAIR_NAME.fullmatch(name.strip()) and len(name.split()) <= 6:
            last = name.strip()
            out[last] = value.strip()
        elif last is None:
            return None
        else:
            out[last] = (out[last] + ";" + part).strip()
    return out if len(out) >= least else None


def _line_map(text: str) -> dict[str, Any] | None:
    """One "key: value | key: value | ..." line as a record map, or None when the text is not such a line."""
    if " | " not in text:
        return None
    fields: list[list[str]] = []
    for piece in text.split(" | "):
        hit = LINE_FIELD.fullmatch(piece.strip())
        if hit:
            fields.append([hit.group("key").strip(), hit.group("value") or ""])
        elif fields:
            fields[-1][1] += " | " + piece
        else:
            return None
    if len(fields) < 4 or not any(_idish(value.strip()) for _, value in fields):
        return None
    record: dict[str, Any] = {}
    for key, value in fields:
        pairs = _pair_map(value) if "=" in value else None
        record[key] = pairs if pairs else _plain_text(value)
    return record


def _text_map(text: str, depth: int) -> Any:
    """What a whole-record string stands for (a line record, decoded JSON text, or name=value parts), or None."""
    found = _line_map(text)
    if found is not None:
        return found
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] in "{[":
        try:
            decoded = json.loads(stripped)
        except ValueError:
            decoded = None
        if isinstance(decoded, (dict, list)) and decoded:
            return _unline(decoded, depth + 1)
    pairs = _pair_map(stripped, 4) if "=" in stripped and " | " not in stripped else None
    if pairs and any(_idish(v) for v in pairs.values()):
        return {k: _plain_text(v) for k, v in pairs.items()}
    return None


def _unline(node: Any, depth: int = 0, whole: bool = True) -> Any:
    """A store result with its text-written listing records turned into maps. Only a string standing for a whole
    record is read: the result itself, an item of a list, or a value of a wrapper of at most two keys; a one-key
    wrapper around a line record becomes that record. Everything else comes back as it was."""
    if depth > 5:
        return node
    if isinstance(node, str):
        found = _text_map(node, depth) if whole else None
        return node if found is None else found
    if isinstance(node, list):
        return [_unline(v, depth + 1) for v in node]
    if isinstance(node, dict):
        small = len(node) <= 2
        out = {k: _unline(v, depth + 1, small) for k, v in node.items()}
        if depth == 0 and len(node) == 1:
            (key, value), = node.items()
            if isinstance(value, str) and isinstance(out[key], dict) and _line_map(value) is not None:
                return out[key]
        return out
    return node


class LLM:
    """Per-episode model client over the proxy chat-completions endpoint. Keeps the running spend, call and failure
    counts, and decides model fallbacks, retries and skips from the episode's clock and spend.
    """
    def __init__(self, episode: "Episode"):
        """Binds the client to its episode, which supplies the clock and the log, and starts the spend, call count and
        failure count at zero.
        """
        self.ep = episode
        self.cost = 0.0
        self.calls = 0
        self.failures = 0

    def chat(self, model: str, system: str, user: str, max_tokens: int) -> str | None:
        """Sends one temperature-0 system+user request and returns the reply text or None. Past SOFT_SPEND the Sonnet
        roles go to FAST_MODEL; near the deadline no call is made, and the timeout follows the time left. 429/5xx
        errors get two paced retries, after which a PLAN_MODEL call still failing that way tries the older Sonnet
        kin. Adds cost; logs 'llm_failure'.
        """
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
            error = result.error if isinstance(result.error, dict) else {}
            if isinstance(result.data, dict) and result.data.get("choices"):
                break
            if error.get("status") not in (429, 500, 502, 503, 504) or self.ep.time_left() < WRAP_SECONDS + 10:
                break
            time.sleep(2.0 * (attempt + 1))
            result = _llm.post_verbose("/inference/chat/completions", body)
        if model == PLAN_MODEL and not (isinstance(result.data, dict) and result.data.get("choices")):
            status = (result.error if isinstance(result.error, dict) else {}).get("status")
            for kin in (SONNET_KIN if status in (429, 500, 502, 503, 504) else ()):
                if self.ep.time_left() < WRAP_SECONDS + 10:
                    break
                self.calls += 1
                result = _llm.post_verbose("/inference/chat/completions", dict(body, model=kin))
                if isinstance(result.data, dict) and result.data.get("choices"):
                    model = kin
                    break
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
        """Asks each listed model in turn via chat and returns the first reply that yields JSON through _json_in. None
        when every model fails or answers without parseable JSON; each attempt spends a call.
        """
        for model in models:
            text = self.chat(model, system, user, max_tokens)
            parsed = _json_in(text or "")
            if parsed is not None:
                return parsed
        return None



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
 "cheapest": false,                 // true only when the request asks for the cheapest / lowest price / cheapest total among what fits (a rule for ranking near misses when nothing fits does not count)
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

SCREEN_PROMPT = (
    "You sort a store's search rows for a shopping assistant before it opens any listing page. Each numbered row shows"
    " an id, a title, a short text, a category and a price. Decide from the row alone whether the listing could still"
    " meet the requirements once its full page is read: \"yes\" when the row itself states the required values, \"no\""
    " when the row's own words plainly contradict a required value, and nothing (it stays possible) when the row is"
    " silent about a requirement. The kind of product is settled by the store's category label: a row filed under the"
    " requested category is that kind of product whatever its title says, so a row is never \"no\" for its kind."
    " Return ONE JSON object and nothing else: {\"yes\": [row numbers], \"no\": [row numbers]}; a row in neither"
    " list counts as \"maybe\"."
)

JUDGE_PROMPT = """You judge whether store listings meet a shopper's requirements, from each listing's own records. Reading rules:
- A requirement is met when the listing's own property for it states the value. Property keys vary in spelling (spaces, underscores, prefixes, abbreviations); any key that names that property counts, and the same property can appear under several keys. Values match by meaning: one element of a comma-separated list is enough, singular and plural are the same, close synonyms and obvious abbreviations count, case does not matter, a store value made of the required value's own words plus qualifiers meets it ('matte black' for 'black', 'faux suede' for 'suede'), and a value with a unit matches after converting units (compare numbers as numbers; a limit includes its edge; a range meets a limit only when the whole range does).
- The listing's property table decides first: when a property covering the requirement exists, its value decides (a different value fails the requirement even if the title says otherwise). When NO property key covers the requirement, a value stated in the seller's notes, the options or variants field, the title or the description counts. When nothing states it at all, the requirement is unaddressed ("unsure"). A value under a different, related property does not count (a handle is not a lock type; a lining material is not the material).
- "is not" and "different from" requirements fail when the listing's property states the excluded value; "one of" requirements need one of the listed values; "contains" needs the value among those stated. A requirement that some feature be absent (a value of none, no or without) is met when no record mentions that feature.
- The store's category label decides the kind of product: a listing whose label is the requested category is the right kind whatever its title says. Only a listing with no label can fail for being an accessory, a spare part or a case rather than the product.
- When a requirement asks for compatibility with another product (named by its model names or codes), it is met only when this listing's own text lists one of those names among what it fits or works with. When the other product's page names a series or family of compatible part codes, a listing whose own part code (often in brackets in its title) starts with that series meets it. A sentence saying that some OTHER part fits a product applies to the listing whose title carries that part's code, not to the page it is written on; a listing's "related_pages" show such pages, and their statements about this listing's own part code count for it. A pointer to another page alone proves nothing.
- A listing's seller notes may send one of its properties to ANOTHER listing (for that property, go by or take the value of another listing, not this one). Then this listing's own value for that property does not count: a requirement on it is "unsure" unless a related page shown with the listing states it. Notes that send away a property no requirement names change nothing.
- STORE FIELDS, when given, list for each required property the keys and values that this store's read listings of the category use. If some of those listings carry the exact required value in that field, a listing whose field holds a different value fails, even a related or broader one (the required words with qualifiers added are not a different value); a value no read listing carries may still be met by a clear synonym.
- Do not invent requirements: judge only the ones listed.
The requirements are numbered R1, R2, ... Return ONE JSON object and nothing else (no commentary): {"verdicts": [{"id": "...", "failed": ["R numbers of requirements the records clearly fail"], "unsure": ["R numbers no record addresses"], "weak": ["R numbers met only by the title, the description or an options field, not by a property or a seller note"], "loose": ["R numbers met by a synonym or a broader term rather than by the value's own words (for example a value 'female' for a required 'girls'); a unit conversion or a series that covers a code is not loose"]}, ...]} covering every listing; leave out a list that would be empty. A listing with no "failed" and no "unsure" meets everything shown."""

PLAN_PROMPT = """You choose what a shopping assistant asks a shopper. The shopper holds a few preferences the request does not state, each one the value of a single property of one of the requested products, and reveals one only to a message that names that property (by its name or a plain synonym). A message may name several properties of ONE product; the shopper then speaks about the first of those that they hold a preference on, and says they do not mind when they hold none.
You get the request, then for each requested item: its name, the requirements already known, the story's hints, properties the request's own rules settle, properties already asked, and a property menu: the attribute keys its opened candidate listings carry, how many of them carry each ("listings" of "of") and some of their values.
Return ONE JSON object and nothing else: {"questions": [{"item": "the item's name exactly as given", "property": "the property, spelled as the menu spells it, with spaces between words"}, ...]} with QUESTION_COUNT entries, the likeliest first.
How to rank, item by item:
1. Properties the story points at: the person who will use or wear it (gender, age group), how it closes or attaches, what goes inside or with it (processor, capacity, a device it serves), how it should look (colour, pattern, style), or a subject the request says to raise. Use the menu's name for it.
2. Then the menu's keys that most of the item's listings carry and whose values differ between listings, a property a shopper of that kind of product has an opinion on: these are where unstated preferences sit. A key carried by few of the opened listings (compared with "of"), an identifier, a model or part number, an option or variant selector, warranty, shipping or return terms come last or not at all.
3. A property is settled only when a known requirement or the request's own rules fix that very property: a key that merely shares a word with a requirement (a "pack type" next to a required "packaging type", a "display size" next to a required "display type") is a different property and stays in the running. Never the price, stock or quantity, a property already asked, or a second spelling of one already chosen.
With several items, give every item its likeliest properties before any item's weaker ones."""

REPLY_PROMPT = (
    "A shopping assistant sent a shopper a message about one product, naming a single property of it or a list of"
    " its properties, and the shopper replied. Read the reply and return ONE JSON object and nothing else:\n"
    '{"kind": "need|repeat|free|split|unclear|other",\n'
    ' "property": "the property the reply is about: the name the shopper uses when they name one, else the asked'
    ' property they answer",\n'
    ' "about": "only when several properties were asked: the one of them, spelled exactly as in the question, that'
    ' the stated value or limit (or the restated known need) is about; otherwise null",\n'
    ' "op": "is|is not|at least|at most|one of|same as reference|different from reference|counterpart of reference|'
    'at least reference|at most reference|within reference",\n'
    ' "value": "the value, limit or set of values the shopper gives (a set written with \' | \' between values)",\n'
    ' "reference": null or {"category": "...", "seller": "...", "attributes": {"property": "value"}, "property":'
    ' "the property to read on that other page", "count": null or how many listings the reply says fit,'
    ' "pick": null or the rule for choosing among several, "relation": "the reply\'s own words for how the wanted'
    ' value relates to the value on that page", "tolerance": null or the margin allowed above or below that page\'s'
    ' figure},\n'
    ' "budget": null or the new total the shopper says they can spend, as a number,\n'
    ' "answer": "" or, when the shopper asks the assistant something, a short plain answer to it,\n'
    ' "steps": [] or the things the shopper now requires in how the task is carried out (an approval to obtain, something'
    ' to show or tell them, a tool to use before the order; never a wish about how questions are put, such as one at a'
    ' time, which is kind split), each {"moment": "before_order|before_decline|'
    'before_finish", "kind": "message|tool", "tool": null, "goal": "what has to happen", "messages": 1}}\n'
    "Kinds:\n"
    "- need: the shopper holds firm on a value, a limit, an exclusion or a set of acceptable values, however it is"
    " phrased (settled on it, nothing but it, only that will do, anything except it, anything bar it, this or that and"
    " nothing else)."
    " \"Anything but X\", \"anything bar X\" or \"other than X\" is op \"is not\" with value X. When the wanted value is to be read"
    " off ANOTHER listing's page (a listing described by its category, its seller and some of its attribute values),"
    " fill \"reference\" and choose the op by the relation: the same value as that page states is \"same as"
    " reference\"; anything but that page's value is \"different from reference\"; the value that page pairs with"
    " its own (its counterpart, what goes with it) is \"counterpart of reference\"; no lower or no"
    " higher than that page's value is \"at least reference\" or \"at most reference\".\n"
    "- repeat: instead of answering about what was asked, the shopper states again a requirement already listed"
    " under KNOWN NEEDS (the same property and value).\n"
    "- free: the shopper has no preference: anything is fine, they leave it to the assistant, they never thought"
    " about it.\n"
    "- split: the shopper wants the message broken up: one question at a time, its subjects asked separately or in"
    " sequence, only its first part answered and the rest held back (ask one, then the next; just give me the first;"
    " space them out; not all at once).\n"
    "- unclear: the shopper did not follow the message or asks for it again or in simpler words.\n"
    "- other: anything else, such as a greeting, a question back or only a change of budget.\n"
    "Naming the property: when the reply names a property, use the shopper's name for it even if the question asked"
    " about a different one; when it names none, use the asked property the answer belongs to (with several asked,"
    " the one the value goes with).\n"
    "Limits: explicit limit words in the reply (such as at most N, at least N, no more than N, under N) decide the op and"
    " the value even inside an odd sentence: \"nothing that falls short of at most 34 inches\" is op \"at most\","
    " value \"34 inches\".\n"
    "Without limit words, an insisted value is op \"is\" whatever figure of speech carries it: refusing to look"
    " past a value, or to settle for anything else, asks for that value itself, a figure or measurement included (op"
    " \"is\", value \"50 cm\")."
)

EVENT_PROMPT = (
    "While a shopping assistant was working, a line from the shopper came inside a store tool's result. Say what it"
    " changes. Return ONE JSON object:\n"
    '{"changes": [{"kind": "budget|drop|add|price|stock|step|other",\n'
    '  "budget": null or the new maximum total as a number,\n'
    '  "property": null or the property of a requirement dropped or added,\n'
    '  "op": "is|is not|at least|at most|one of", "value": null or the added requirement\'s value,\n'
    '  "item": null or the requested product it is about, "note": "one short sentence"}]}\n'
    "- budget: the shopper can now spend a different amount, more or less; copy the number as the line gives it, in"
    " the line's own unit (cents stay cents, never convert).\n"
    "- drop: the shopper gives up a requirement on some property (let it go, drop it, never mind it).\n"
    "- add: the shopper adds a requirement.\n"
    "- price: the shopper says the chosen listing's price has changed.\n"
    "- stock: the shopper says the chosen listing can no longer be bought.\n"
    "- step: the shopper requires a step in how the task is carried out (an approval to obtain, something to show or tell"
    ' them, a tool to use before the order); give it in "step": {"moment": "before_order|before_decline|before_finish",'
    ' "kind": "message|tool", "tool": null, "goal": "what has to happen", "messages": 1}.'
)

REFER_PROMPT = """You read one store listing page for a shopping assistant that follows a reference. The assistant needs the value of a given property as this page gives it. The page may state the value in its property table or in the seller's notes, or the notes may say to take that property from ANOTHER listing (named by category, seller and some attribute values) instead of this one. The notes may also say which value of the property goes together with which (one pair for this page's value, or the store's whole list of pairs).
Return ONE JSON object: {"value": "this page's own value as written, or null", "paired": "the value the page pairs with that value, or null", "wanted": "the value the shopper's RELATION asks for: the paired value when it asks for what goes with, suits or is compatible with the page's value and the page states a pairing; the page's own value when it asks for the same value", "redirect": null or {"category": "null unless the notes name one", "seller": "...", "attributes": {"property": "value"}, "property": "..."}}
When the notes redirect the WANTED property to another listing, return the redirect and no value; a note that sends some other property elsewhere is not a redirect for this one. When the notes say the value outright, return it. Otherwise take the value from the property table under any key that names the property; when no key names it, from the options field, the title or the description when they state it. A sentence that only lists the store's pairs does not say which value is this page's own. "paired" comes only from a pairing the page states: in a list of pairs, the pair holding this page's own value (pairs read both ways); null when the page states none or its own value is unknown."""

QUERY_PROMPT = (
    "You write search-box queries for a shopping assistant. The store lists a row only when every word of the query"
    " appears in its title, brand, category or property values (word stems match, very short words are skipped), and"
    " rows come in a fixed order a few per page, so a good query is narrow enough that most of what it finds meets the"
    " requirements. You get the product, its requirements, the queries already tried and some listings seen, each"
    " marked with whether it met the requirements. Return ONE JSON object and nothing else: {\"queries\": [\"...\","
    " ...]} with up to 5 new queries of 1 to 3 words, best first: the product's noun as the store's titles spell it"
    " with a required value as listings write it, or other words the store uses for that product or value; never a"
    " query already tried."
)

ARGS_PROMPT = (
    "A shopping assistant's call to a store tool does not satisfy the tool. You get the tool's name, description and"
    " schema, the arguments the call carries, what the tool is used for here, and the store's refusal if there was"
    " one. Return ONE JSON object and nothing else: {\"args\": {all the arguments for this call}}. Keep each of the"
    " call's arguments unless the refusal says that one is wrong, and fill in what is missing with the value that serves this use"
    " (a quantity of 1 unless the arguments say otherwise, the first page of results, the option that fits the use)."
    " Use only argument names and values the schema allows."
)

CHOICE_PROMPT = (
    "A store tool that ends a shopping task needs arguments that say how it should end. You get the tool's"
    " description, the arguments still missing with the values each one allows, and the goal. Return ONE JSON object"
    " and nothing else: {\"<argument name>\": \"<the allowed value that reaches the goal>\", ...}."
)

GATE_PROMPT = """You check a shopping assistant's process against the shopper's request at one moment of a task. You get the request, the steps the request demands in how the task is carried out, the moment (what the assistant is about to do), the basket now, the allowances left, the store's tools (the ones that end the task or change the cart are left out: the assistant runs those itself) and every call made so far with its result. Decide which of the demanded steps are due NOW, before the assistant goes on, and not yet done for the current state: an approval or confirmation counts only for the basket exactly as it is now (one given for an earlier basket does not count); a step already done for this state is not due again; a step for another moment is not due. Return ONE JSON object and nothing else: {"actions": [{"tool": "exact tool name", "args": {"argument": "value", ...}, "why": "the demanded step it carries out"}]} in order, at most 2, or {"actions": []} when nothing is due. A message to the shopper must stand on its own: for a basket, name every line (quantity, title, price) and the total, and say plainly what you ask for. Never put a question on product properties here, and never spend the last message on anything but a demanded step. A check the assistant makes in the store (stock, availability, a value) is done with the store's reading tools, never with a message, unless the request asks to be told; the assistant already reads every listing it buys, so such a check is due only when the basket's listings have not been read."""

RANK_PROMPT = (
    "All of the listings below meet a shopper's requirements, and the shopper wants the best one by a given measure"
    " (a value to make as high or as low as possible). Read each listing's value for that measure from its properties,"
    " title or description, converting units so the values can be compared. Return ONE JSON object and nothing else:"
    " {\"order\": [\"id\", ...], \"values\": {\"id\": \"the value as read\"}} listing every id, best first;"
    " listings whose value cannot be read go last, and between equal values the cheaper listing goes first."
)

AUDIT_PROMPT = """You review a shopping assistant's search for one product before it gives up on it. You get the product, its requirements, the budget, the queries already tried and every listing seen: READ ones (their properties, notes, and the requirements the assistant judged failed or unaddressed) and rows seen only in search results. Return ONE JSON object and nothing else: {"fits_after_all": ["ids of READ listings that do meet every requirement after all, judged from their own records: a property value decides, a different value fails, a property no record states is not met"], "unopened": ["ids of rows not read yet that most likely meet the requirements, best first, at most 4"], "queries": ["at most 2 new keyword queries of 1-3 words likely to list fitting listings, never one already tried"]}. Any list may be empty."""

COND_PROMPT = (
    "A shopping assistant tests one listing against a condition the shopper set. From the listing's own records,"
    " decide whether its value for the given property is one of the given values (the same meaning counts, units"
    " converted). Return ONE JSON object and nothing else: {\"holds\": true or false, \"value\": \"the listing\'s"
    " value as written, or null\"}; when no record states the property, \"holds\" is false and \"value\" is null."
)

MATCH_PROMPT = (
    "A shopping assistant is looking for one particular listing, described by a category, a seller and some attribute"
    " values. Candidate listings and their records follow. Return ONE JSON object and nothing else, no explanation:"
    " {\"id\": \"the id of the listing that fits the description, or null when none does\"}. To fit, the seller"
    " must be the described seller, the category label must name the described kind, and every described attribute"
    " must agree with the listing's records (the same meaning counts, units converted). When a listing's property"
    " table is not shown, its seller notes, title and description serve as its records, and a listing whose notes"
    " state the described values fits. A listing whose records leave a described value unstated, or give a different"
    " one, does not fit: answer null rather than the nearest listing, so the assistant searches further. When a PICK"
    " rule is given, several listings may fit: apply the rule to those that fit and answer the one it selects."
)

VERIFY_PROMPT = """You are the final checker before a shopping assistant orders. Judge the listing below against EVERY requirement listed, from its own records, with these rules: a requirement is met when the listing's own property for it states the value (any key naming the property counts; comma lists, synonyms, abbreviations, plural/singular and unit conversions count); when no property key covers the requirement, the seller's notes, the options field, the title or the description may state it; a requirement that nothing states is not met; a value under a different property does not count; a compatibility requirement is met when this listing names the other product among what it fits, or when the other product's page names a series of compatible part codes and this listing's own part code starts with it; a property for which the notes point to another listing's value is not met by this listing's own value; exclusions and closed lists must be respected; a listing whose category label is the requested category is the right kind of product whatever its title says; price times quantity must be within the budget (same unit as the prices shown); if the records show it cannot be bought, it fails. Judge only the listed requirements, never add your own.
Return ONE JSON object: {"fits": true or false, "problems": ["each requirement that is clearly not met or that no record addresses"]}."""

LABEL_PROMPT = (
    "A store files its listings under category labels, and a shopper asked for one kind of product. Among the labels"
    " seen, pick each label that IS that kind of product: a label whose meaning matches the request's wording as a"
    " whole, or a broader label that includes it (\"sweaters and cardigans\" for a cardigan). Leave out labels for"
    " related but different kinds: accessories, spare parts, cases or covers for it, a narrower or another type."
    " Return ONE JSON object and nothing else: {\"labels\": [\"label exactly as given\", ...]}, an empty list when"
    " no label fits."
)

MAP_PROMPT = (
    "Below is one JSON record a store tool returned. Say which of the record's KEYS plays each role listed: answer with"
    " key names exactly as the record spells them, never with the values stored under them, and null when no key plays"
    " the role. Return ONE JSON object whose every value is one of the record's key names or null:\n"
    '{"id": "key holding the listing\'s identifier", "title": "key holding its name", "text": "key holding its'
    ' descriptive text", "price": "key holding its price", "stock": "key saying whether it can be bought now",'
    ' "category": "key holding its category label", "brand": "key holding its brand", "seller": "key holding the shop'
    ' or seller offering it", "facts": "key holding the object of its properties", "notes": "key holding the seller\'s'
    ' free-text notes on it", "total": "key holding the total price of a cart", "lines": "key holding the list of lines'
    ' in a cart", "next": "key holding the token or marker for the next page of results", "count": "key holding how'
    ' many results match in all"}'
)



class Listing:
    """Local view of one store listing: id, title, description, price, stock flag, category, brand, seller,
    property table and seller notes, plus whether it was opened or its page read, the raw record and its
    screening label.
    """
    def __init__(self, id_: str):
        """Creates a blank listing for the given id: empty text fields, unknown price and stock, no properties or
        notes, not opened, page not read, and a screening label of 'maybe'.
        """
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
        self.page_read = False
        self.raw: dict[str, Any] = {}
        self.screen = "maybe"

    def record(self, limit: int = FACT_CHARS) -> dict[str, Any]:
        """Builds the compact dict a model judge sees: id, 160-char title and whichever of category, brand, seller,
        price and can_buy are known, properties deduplicated by (camel-split key, value), notes (600) and
        description (500, or 400 with properties). Over `limit`, properties are cut in proportion (at least six),
        then the description to 200.
        """
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


_PAIR = re.compile(r"([A-Za-z][\w &'/()+-]{1,48}?)\s*:\s*([^;:.\n]{1,60})")


def _fold_phantoms(specs: list[Any], request: str) -> list[Any]:
    """Item specs with any item the reader made out of a "property: value" requirement folded back into the item before
    it (see patch d4); the other specs come back unchanged."""
    pairs = [(m.group(1).strip(), m.group(2).strip(), m.span()) for m in _PAIR.finditer(request or "")]
    out: list[Any] = []
    for spec in specs:
        if out and isinstance(spec, dict) and isinstance(out[-1], dict) and not spec.get("requirements"):
            name = _stems(spec.get("name") or "")
            for prop, value, (a, b) in pairs:
                rest = _stems(request[:a] + " " + request[b:])
                if name and name <= _stems(prop + " " + value) and not (name & rest == name):
                    prev = out[-1]
                    reqs = list(prev.get("requirements") or [])
                    if not any(isinstance(r, dict) and _stems(r.get("property") or "") == _stems(prop) for r in reqs):
                        reqs.append({"property": prop, "op": "is", "value": value, "reference": None})
                    out[-1] = dict(prev, requirements=reqs)
                    spec = None
                    break
        if spec is not None:
            out.append(spec)
    return out


def _host_fits(specs: list[Any], rules: Any) -> list[Any]:
    """Item specs with each fit rule whose host is no item of the request (the shopper's own vehicle, device or model)
    turned into a "fits contains <host>" requirement of its accessory item (see patch d6); the rest unchanged."""
    out = [dict(s) if isinstance(s, dict) else s for s in specs]
    items = [s for s in out if isinstance(s, dict)]
    for rule in rules if isinstance(rules, list) else []:
        if not isinstance(rule, dict) or not items:
            continue
        acc, host = str(rule.get("accessory") or "").strip(), str(rule.get("host") or "").strip()
        if not acc or not host or any(_similar(s.get("name") or "", host) >= 0.2 for s in items):
            continue
        target = max(items, key=lambda s: _similar(s.get("name") or "", acc))
        if _similar(target.get("name") or "", acc) < 0.2:
            continue
        reqs = list(target.get("requirements") or [])
        if not any(isinstance(r, dict) and _norm(host) in _norm(r.get("value") or "") for r in reqs):
            reqs.append({"property": "fits", "op": "contains", "value": host, "reference": None})
        target["requirements"] = reqs
    return out


class Requirement:
    """One condition on an item's property: the property name, an operator (lowercased, 'is' by default), the value
    text, an optional description of another listing for the page-relative operators, and an origin tag such as
    request, reply or fit.
    """
    def __init__(self, prop: str, op: str, value: str, reference: dict | None = None, origin: str = "request"):
        """Stores a requirement with trimmed property and value strings, a lowercased operator that defaults to 'is',
        the reference only when it is a dict, and the given origin tag.
        """
        self.prop = str(prop or "").strip()
        self.op = str(op or "is").strip().lower()
        self.value = str(value or "").strip()
        self.reference = reference if isinstance(reference, dict) else None
        self.origin = origin

    def text(self) -> str:
        """Renders the requirement as 'property: op value'. A page-relative operator still holding its reference is
        shown as not yet resolved instead of with a value. This string doubles as the requirement's identity in
        verdict sets.
        """
        if self.op in REF_OPS and self.reference:
            return f"{self.prop}: {self.op} (not yet resolved)"
        return f"{self.prop}: {self.op} {self.value}"

    def key(self) -> str:
        """Normalized property name of this requirement, used to match requirements, questions and replies that concern
        the same property, for example when a new shopper need replaces an earlier one.
        """
        return _norm(self.prop)


class Item:
    """One product the shopper wants: name, category, quantity, requirements, hints, per-unit price cap and search
    terms, plus its search state (listing pool, judge verdicts, page cursors, ids shown per query), the
    properties asked or left free, and the chosen listing.
    """
    def __init__(self, spec: dict[str, Any], index: int):
        """Builds an item from one request-reader spec at position `index`. Name falls back to category, quantity must
        be a number >= 1, requirements need a property, a non-positive price cap is dropped, and search terms
        default to the category with the last one taken as the bare product noun. Search state starts empty.
        """
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
        self.price_cap = _number(spec.get("price_cap"))
        if self.price_cap is not None and self.price_cap <= 0:
            self.price_cap = None
        self.terms = [str(t) for t in (spec.get("search_terms") or []) if t]
        if not self.terms:
            self.terms = [self.category]
        self.noun = self.terms[-1]
        self.pool: dict[str, Listing] = {}
        self.verdicts: dict[str, dict[str, set[str]]] = {}
        self.judged_for: dict[str, set[str]] = {}
        self.asked: list[str] = []
        self.freed: list[str] = []
        self.search_done = 0
        self.exhausted = False
        self.chosen: Listing | None = None
        self.category_labels: dict[str, int] = {}
        self.labels_ok: set[str] | None = None
        self.cursors: dict[str, dict[str, Any]] = {}
        self.query_ids: dict[str, set[str]] = {}
        self.value_ids: set[str] = set()
        self.value_complete = False

    def query_next(self, query: str) -> None:
        """Queues a query (cut to 60 characters) so this item's next keyword search runs it, by inserting it at the
        index of the next unused term. A query equal after normalization to one already listed is ignored.
        """
        if query and not any(_norm(query) == _norm(t) for t in self.terms):
            self.terms.insert(min(self.search_done, len(self.terms)), query[:60])

    def over_cap(self, listing: Listing) -> bool:
        """True when this item has a per-unit price cap and the listing's known price exceeds it beyond a tiny
        tolerance; unknown prices never count as over.
        """
        return self.price_cap is not None and listing.price is not None and listing.price > self.price_cap + 1e-6

    def active(self) -> list[Requirement]:
        """The requirements that can be judged now: any requirement still holding a reference under a page-relative
        operator is left out until it is resolved into a plain one.
        """
        return [r for r in self.requirements if not (r.reference and r.op in REF_OPS)]

    def req_texts(self) -> list[str]:
        """Text forms of the active requirements; a listing counts as fully judged only when its judged set covers all
        of these.
        """
        return [r.text() for r in self.active()]

    def wrapped(self, listing: Listing) -> int:
        """Counts active 'is'/'one of' requirements for which the listing has a matching property field (near name or
        >=0.5 overlap) yet no element of it, split at , ; / |, has exactly the wanted value's words. Requirements
        with no matching field are not counted. Used to rank such listings later.
        """
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
        """Counts active 'is'/'one of' requirements whose value, as a word set, equals some element (split at , ; / |)
        of any of the listing's property values. Unlike wrapped, any property may supply it. Zero when the listing
        has no property table.
        """
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
        """Returns opened, unfailed, within-cap listings judged on every active requirement. Sure fits (no unsure or
        word-loose result) sort by price with wrapped values last when `cheapest`, else by exactness, weak count,
        then price. `strong_only` skips weak evidence; unsure ones, ranked by doubt, are returned only with
        `include_unsure` and no sure fit.
        """
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
        if cheapest:
            sure.sort(key=lambda l: (self.wrapped(l) > 0, key(l)))
        else:
            sure.sort(key=lambda l: (-self.exactness(l), len(self.verdicts[l.id].get("weak") or ()), key(l)))
        unsure.sort(key=lambda l: (len(self.verdicts[l.id]["unsure"]) + len(self.word_loose(l.id)), key(l)))
        if sure or not include_unsure:
            return sure
        return unsure

    def word_loose(self, lid: str) -> list[str]:
        """Texts of the listing's active plain-word 'is'/'one of' requirements that the judge flagged as loose (matched
        through a different word). Values with digits, fit-rule requirements and cases where a word of the required
        value already appears in the listing's own field are left out, since those count as stated.
        """
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
        """Static test whether the listing's property fields that match the requirement's property (near name or >=0.5
        overlap) share at least one token with any of the required alternatives split at '|'.
        """
        own = [str(v) for k, v in listing.facts.items()
               if _near(str(k), req.prop) or _similar(str(k).replace("_", " "), req.prop) >= 0.5]
        words = set().union(*(_tokens(v) for v in own)) if own else set()
        return any(_tokens(alt) & words for alt in req.value.split("|"))


class Episode:
    """All state and behaviour for one shopping task: store binding, tool schemas and roles, items and budget, step
    and time accounting, message and question bookkeeping, cart, procedure demands, trace and history, plus per-
    world memory of record mappings and store limits.
    """
    def __init__(self, problem_data: dict[str, Any]):
        """Sets up a task from the problem payload: binding, request text, unlined opening observation, tools and their
        parameter schemas, step limit (30 by default) and empty state for items, cart, messages and history.
        Restores record mappings, compare batch size, thin-compare and all-words flags remembered for the same tool
        set.
        """
        self.started = time.monotonic()
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.request = str(self.policy.get("query") or "")
        self.opening = _unline(self.policy.get("opening"))
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
        self.confirm_first = False
        self.procedure: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
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
        self.compare_limit = 4
        self.page_size = 0
        self.world_key = "|".join(sorted(self.schemas))
        memo = _WORLD_MEMORY.get(self.world_key) or {}
        self.maps = dict(memo.get("maps") or {})
        self.compare_limit = int(memo.get("compare_limit") or 4)
        self.thin_compare = bool(memo.get("thin_compare"))
        self.opened_pages: set[str] = set()
        self.decline_reason = ""
        self.planned: list[tuple[Item, str]] | None = None   # the planner's order, once per task
        # whether the shopper takes a message listing several properties: None until a list is answered (True) or
        # refused with a request for one thing at a time (False)
        self.lists: bool | None = None
        self.mode = "full"
        self.query_counts: dict[str, int] = {}
        self.all_words: bool | None = memo.get("all_words")

    def time_left(self) -> float:
        """Seconds remaining of the TASK_SECONDS budget, measured on a monotonic clock from the episode's start and
        negative once overrun. Model and environment calls size their timeouts from it.
        """
        return TASK_SECONDS - (time.monotonic() - self.started)

    def steps_left(self) -> int:
        """Environment steps remaining before the task's step limit, which a call budget in the request may have
        lowered.
        """
        return self.max_steps - self.step

    def note(self, kind: str, **fields: Any) -> None:
        """Appends an entry of the given kind, stamped with the current step and elapsed seconds and carrying any extra
        fields, to the episode's trace log.
        """
        entry = {"kind": kind, "step": self.step, "t": round(time.monotonic() - self.started, 1)}
        entry.update(fields)
        self.trace.append(entry)

    def remember(self) -> None:
        """Saves the learned record mappings, compare batch size, thin-compare flag and all-words flag in process-wide
        memory under this world's tool-set key, so later tasks in the same world start with them.
        """
        _WORLD_MEMORY[self.world_key] = {"maps": self.maps, "compare_limit": self.compare_limit,
                                         "thin_compare": self.thin_compare, "all_words": self.all_words}

    def spent(self) -> bool:
        """Compares this task's running model cost with SPEND_CAP; once reached, the explore and warm-up loops and the
        extra search and audit passes stop.
        """
        return self.llm.cost >= SPEND_CAP

    def batch_open(self) -> bool:
        """True when the compare role is usable for multi-listing reads: it is assigned, it has not been disabled for
        this task, and its rows are not known to lack property tables.
        """
        return bool(self.roles.get("compare")) and not getattr(self, "compare_off", False) and not self.thin_compare

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Wrapper around send: fits arguments to the schema and, on an error that reads like an argument complaint,
        gets corrected arguments from repair_args and resends once if they differ. Sets out['bad_args'] when the
        complaint persists. Each send spends a step.
        """
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
        """Static test whether a failed call's error text, or its compacted observation, complains about arguments:
        words such as invalid, missing, required, parameter, schema, validation or 'must be'.
        """
        return bool(re.search(r"invalid|missing|required|argument|parameter|schema|malformed|validation|unexpected|must be",
                              f"{out['error']} {_compact(out['obs'], 400)}", re.I))

    def arg_uses(self, name: str) -> list[str]:
        """Names of the roles (search, open, cart, checkout and so on) currently assigned to the given tool, used to
        tell the argument helper what a call is for.
        """
        return [role for role, spec in self.roles.items() if isinstance(spec, dict) and spec.get("name") == name]

    def fit_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Makes call arguments acceptable to the tool's schema: drops None values, coerces declared ones with
        _fit_value, and moves arguments under undeclared names into missing required ones, preferring values of the
        right type and then the closest name. Required gaps left over go to fill_args. Schemas without properties
        pass args through.
        """
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
        """Produces values for the missing required arguments of a call: a value cached for this tool and argument, the
        schema default, else one fast-model call with _arg_default as fallback; new values are cached for the task.
        Enum arguments of checkout or decline tools are skipped because fixed_args picks them. Logs 'args_filled'.
        """
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
        """Asks the fast model (backup model second) for a complete argument set, giving it the tool's name,
        description and schema, the call's current arguments, the roles it plays here and the store's refusal when
        there is one. Returns the parsed dict or None.
        """
        user = (f"TOOL: {name}\nDESCRIPTION: {self.tool_description(name)}\nSCHEMA: {json.dumps(self.schemas.get(name) or {}, ensure_ascii=False)}"
                f"\nARGUMENTS OF THE CALL: {json.dumps(have, ensure_ascii=False, default=str)}\nUSED HERE AS: {', '.join(self.arg_uses(name)) or 'unknown'}"
                + (f"\nTHE STORE'S REFUSAL: {_compact(refusal, 400)}" if refusal is not None else ""))
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], ARGS_PROMPT, user, 400)
        return answer if isinstance(answer, dict) else None

    def repair_args(self, name: str, args: dict[str, Any], refusal: Any) -> dict[str, Any] | None:
        """Second attempt at a refused call's arguments, allowed two times per tool per task and only with two or more
        steps left. Schema arguments that the refusal text names but the call lacks go to fill_args; otherwise the
        model proposes a full set, refitted to the schema. Logs 'args_refused'; None means do not resend.
        """
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
        """Posts one action to the environment and returns {obs, lines, error, ok}. Refuses unknown tools, a finished
        task, short time, or the last step for a non-finishing tool. Unlines and unwraps the result, queues new
        shopper lines, updates step and finished, appends trace and history. Handles 402, 409 step resync, 403/422;
        up to three tries.
        """
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
                obs, lines = _unwrap(_unline(data.get("observation")))
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

    def mapping(self, kind: str, sample: dict[str, Any]) -> dict[str, str]:
        """Role-to-key map (id, title, price, facts and so on) for a record of the given kind, cached by kind and key
        set. Reuses an exact or 60%-overlapping known map; asks again once per newly filled long text or dict key
        while notes, text or facts is unmapped. Otherwise the fast model names the keys, with structural fallbacks
        for facts, id and price; stored and remembered.
        """
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
            done = self.__dict__.setdefault("remapped", set())
            fresh = [k for k, v in sample.items() if k not in known.values() and (k, key) not in done
                     and ((isinstance(v, str) and len(v) > 25) or (isinstance(v, dict) and v))]
            if not fresh or not any(r not in known for r in ("notes", "text", "facts")):
                return known
            done.update((k, key) for k in fresh)
        shown = {k: (v if not isinstance(v, (dict, list)) else _compact(v, 160)) for k, v in sample.items()}
        for k, v in list(shown.items()):
            if isinstance(v, str) and len(v) > 160:
                shown[k] = v[:160] + "…"
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], MAP_PROMPT, json.dumps(shown, ensure_ascii=False), 700)
        result: dict[str, str] = dict(known or {})
        if isinstance(answer, dict):
            for role, k in answer.items():
                if isinstance(k, str) and k in sample and role not in result:
                    result[role] = k
        if "facts" not in result:
            for k, v in sample.items():
                if isinstance(v, dict) and v and all(isinstance(x, (str, int, float)) or x is None for x in v.values()):
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
        """Converts one raw store record into a Listing using the learned key map for its kind. Requires an id field
        holding a string or number; then copies title, text, parsed price, boolean stock, category, brand, seller,
        the property table (empty values dropped) and notes when present with the right type. None without an id.
        """
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
        return listing

    def merge(self, item: Item, listing: Listing, opened: bool) -> Listing:
        """Adds a listing to the item's pool or folds it into the known entry: text fields overwrite when the data
        comes from an opened page or the old field is empty, known price, stock and raw replace the old, and a
        changed property table clears that listing's judged set. Opening marks it opened and records the page; its
        category label count is bumped.
        """
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
                    item.judged_for.pop(known.id, None)
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

    def setup(self) -> bool:
        """Reads request, opening and tools with one parse-model call (retried once), keeps roles naming real tools and
        fixes them via check_roles. Builds items, dropping requirements that restate a reference's attributes or
        only the category's words, and adds each required value as a search term. Sets budget, ranking, nothing-
        fits, message/step limits, conditions and procedure; False on no plan.
        """
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
        items = _host_fits(_fold_phantoms(items, self.request), plan.get("fit_rules"))
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
            name_words = _words(item.category)
            item.requirements = [r for r in item.requirements
                                 if r.reference or r.op not in ("is",) or not (_tokens(r.value) and _words(r.value) <= name_words)]
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
        self.nearest_ok = str(plan.get("nothing_fits") or "").strip().lower() == "nearest"
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
        """Reconciles cart and finishing roles with the real schemas. A cart tool taking an array of objects becomes
        cart_set (id/qty keys found by type) and replaces cart_add; otherwise cart_add id and qty names are
        corrected or the role dropped. Other roles get a same-typed argument for a missing name, and a sole ending
        tool with a required enum is registered for both endings.
        """
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
                    mode_arg = (spec.get("mode") or {}).get("arg") if isinstance(spec.get("mode"), dict) else None
                    required = (self.schemas.get(spec["name"]) or {}).get("required") or []
                    spec["qty"] = next((k for k in list(required) + list(props) if k not in (spec.get("id"), mode_arg)
                                        and isinstance(props.get(k), dict) and props[k].get("type") in ("integer", "number")), None)
                if not spec.get("id"):
                    self.roles.pop("cart_add", None)
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
        for have, other in (("checkout", "decline"), ("decline", "checkout")):
            spec = self.roles.get(have)
            if not spec or other in self.roles:
                continue
            schema = self.schemas.get(spec["name"]) or {}
            props = schema.get("properties") or {}
            if any(isinstance((props.get(k) or {}).get("enum"), list) for k in schema.get("required") or []):
                self.roles[other] = {"name": spec["name"]}

    def tool_description(self, name: str) -> str:
        """Description text of the named tool from the world's tool list, or an empty string when no tool by that name
        exists.
        """
        for t in self.tools:
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            if isinstance(fn, dict) and fn.get("name") == name:
                return str(fn.get("description") or "")
        return ""

    def fixed_args(self, role: str) -> dict[str, Any]:
        """Builds the argument set for calling the checkout or decline tool as that particular ending: the reader's
        values for declared arguments, snapped to an enum member at >=0.5 overlap or dropped. Required enum
        arguments still unset are chosen by one fast-model call stating the goal, cached per role only on success.
        """
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
                    continue
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
                if cached:
                    self.choice_cache = dict(getattr(self, "choice_cache", {}), **{role: cached})
            for k in missing:
                enum = props[k]["enum"]
                v = cached.get(k)
                if v in enum:
                    out[k] = v
        return out

    def _schema(self, role: str) -> dict[str, Any]:
        """Parameter schema of the tool assigned to a role, looked up through the role's tool name; an empty dict when
        the role is unassigned or its tool has no schema entry.
        """
        spec = self.roles.get(role) or {}
        return self.schemas.get(spec.get("name") or "", {})

    def _arg_type(self, role: str, arg: str | None) -> str | None:
        """JSON type declared for one argument of a role's tool; None when no argument name is given or the schema does
        not declare it.
        """
        if not arg:
            return None
        props = self._schema(role).get("properties") or {}
        return (props.get(arg) or {}).get("type")

    def _page_mode(self, role: str) -> str | None:
        """Classifies how a search role pages: None without a page argument; for an integer argument 'offset' when the
        reader said offset or, unless it said number, the name has offset/start/skip, else 'index'; any non-integer
        page argument is 'token'.
        """
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
        """Maps a string value onto the enum spelling of a role argument: exact normalized match, else the closest
        member with overlap >= 0.5, else None. Non-enum arguments, non-string values and enums without strings
        return the value unchanged.
        """
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
        """Reorders a query so words outside the item's category and name come first and category words follow. When
        the argument has a maxLength and the query is too long, all value words are kept, category words are re-
        added in order while they fit, and _trim cuts the result at a word.
        """
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
        for w in rest:
            if len(" ".join(kept + [w])) <= limit:
                kept.append(w)
        return self._trim(role, arg, " ".join(kept) if kept else query)

    def _trim(self, role: str, arg: str, value: str) -> str:
        """Cuts a string to the role argument's maxLength at the last space before the limit, or hard when there is
        none; unchanged when there is no limit or the string fits.
        """
        props = self._schema(role).get("properties") or {}
        limit = (props.get(arg) or {}).get("maxLength")
        if isinstance(limit, int) and len(value) > limit:
            value = value[:limit].rsplit(" ", 1)[0] if " " in value[:limit] else value[:limit]
        return value

    def search(self, item: Item, query: str | None = None, fields: dict[str, Any] | None = None,
               max_pages: int = 1, budget: int | None = None) -> int:
        """Keyword (query) or field search for an item, paging up to max_pages from the cursor kept per signature.
        Builds mode, fitted query or enum-snapped fields and index/offset/token paging; stops at the step floor
        `budget`, short time, end of results, or a page cap (sets item.capped). Merges rows, notes match counts,
        records ids per query, resolves labels; returns new listings.
        """
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
                self.rows_full = all(l.price is not None and l.stock is not None for l in shown)
            item.search_done += 1
            if item.labels_ok is None and item.index != 99:
                self.resolve_labels(item)
            if not rows or (total is not None and page * max(len(rows), 1) >= total) or len(rows) < self.page_size:
                done = True
                break
            if page_mode == "token" and not token:
                if total is not None and total > page * len(rows):
                    item.capped = True
                done = True
                break
        item.cursors[sig] = {"page": max(page, 0), "token": token, "done": done or page_mode is None}
        return new

    def note_count(self, query: str, count: int) -> None:
        """Stores the number of matches the store reported for a keyword query and compares it with earlier queries of
        a different size that share words. More matches for a superset query means any word matches; under 30% as
        many for the longer one means every word must match. Sets all_words and logs a change.
        """
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
                self.all_words = True
            if self.all_words is not was:
                self.note("search_words", all_words=self.all_words, small=n_small, big=n_big)
        self.query_counts[query] = count

    @staticmethod
    def value_groups(item: Item) -> list[list[str]]:
        """Static: per-requirement lists of search phrases taken from the item's active is/one of/contains values,
        alternatives split at '|'. Alternatives holding a model code are dropped, and lists with over four
        alternatives or any phrase over 24 characters are skipped. Lists from shopper replies come first.
        """
        out: list[tuple[int, list[str]]] = []
        for r in item.active():
            if r.op not in ("is", "one of", "contains") or not r.value:
                continue
            alts = [v.strip() for v in r.value.split("|") if _tokens(v) and not _CODE.search(v)]
            if alts and len(alts) <= 4 and all(len(a) <= 24 for a in alts):
                out.append((0 if r.origin == "shopper" else 1, alts))
        return [g for _, g in sorted(out, key=lambda x: x[0])]

    @staticmethod
    def row_hits(listing: Listing, groups: list[list[str]]) -> int:
        """Static: number of value groups for which at least one alternative's word set is contained in the words of
        the listing's title, text, category, brand, seller and 'key value' property pairs.
        """
        text = _words(" ".join([listing.title, listing.text or "", listing.category, listing.brand, listing.seller]
                               + [f"{k} {v}" for k, v in listing.facts.items()]))
        return sum(1 for g in groups if any(_words(a) <= text for a in g))

    def value_queries(self, item: Item) -> list[str]:
        """Keyword queries built from the item's required values (value_query_list), minus any whose words are a subset
        of a query already sent for the item, either through a keyword-search cursor or listed in value_sent.
        """
        out = self.value_query_list(item)
        sent = [frozenset(_words(k.split(":", 1)[1])) for k in item.cursors if k.startswith("search_text:")]
        sent += [frozenset(_words(q)) for q in getattr(item, "value_sent", ())]
        return [q for q in out if not any(frozenset(_words(q)) <= s for s in sent)]

    def value_query_list(self, item: Item) -> list[str]:
        """Builds value queries for an item: each alternative (up to three per group) joined with the first value of
        every other group, at most three value phrases, plus the product noun (category when the noun is long)
        unless the store matches any word. Duplicate word sets are skipped; stops after a group once four or more
        exist.
        """
        groups = self.value_groups(item)
        if not groups:
            return []
        noun = item.noun if len(item.noun) <= 22 else item.category
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
        """Runs the next unsent value query, paging deeper for the item's first one (6 alone, 3 with several items) and
        less later (3 if every word must match, else 2), never below `floor`. If it matched nothing, tries the
        values without noun words, then the leading value paired with the noun, skipping forms sent before. True
        when a step was spent.
        """
        todo = self.value_queries(item)
        if not todo or self.steps_left() <= floor + 1:
            return False
        q = todo[0]
        first = not getattr(item, "value_sent", None)
        item.value_sent = list(getattr(item, "value_sent", [])) + [q]
        self.note("value_query", item=item.name, query=q)
        before = self.step
        pages = (6 if len(self.items) == 1 else 3) if first else (3 if self.all_words else 2)
        self.search(item, query=q, max_pages=pages, budget=floor)
        self.value_seen(item, q)
        sent = self._fit_query(item, "search_text", (self.roles.get("search_text") or {}).get("query") or "", q)
        if self.query_counts.get(sent) == 0 and self.steps_left() > floor + 1:
            words = q.split()
            role_q = (self.roles.get("search_text") or {}).get("query") or ""
            for shorter in (" ".join(w for w in words if not (_tokens(w) and _tokens(w) <= _tokens(item.noun))),
                            " ".join([self.value_groups(item)[0][0], item.noun])):
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
        """Unions the ids a value query showed into the item's value-id set. The set is flagged complete once every
        query from value_query_list kept all its value words after fitting, was paged to the end, and showed at
        least as many ids as the store's reported match count.
        """
        item.value_ids |= item.query_ids.get("search_text:" + query) or set()
        role_q = (self.roles.get("search_text") or {}).get("query") or ""

        def shown_all(q: str) -> bool:
            """True when the store reported a non-zero match count for the fitted form of the query, its search cursor is
            finished, and the ids recorded for it number at least that count.
            """
            sig = "search_text:" + q
            count = self.query_counts.get(self._fit_query(item, "search_text", role_q, q))
            return bool(count) and (item.cursors.get(sig) or {}).get("done", False) \
                and len(item.query_ids.get(sig) or ()) >= count
        wanted = self.value_query_list(item)
        if wanted and all(shown_all(q) and _words(self._fit_query(item, "search_text", role_q, q)) >= _words(q) - _words(item.noun)
                          for q in wanted):
            item.value_complete = True

    def category_ok(self, item: Item, listing: Listing) -> bool:
        """Decides whether a listing's category label is the requested kind of product. Missing labels pass; once
        accepted labels are resolved, membership decides; otherwise equal or containing normalized names, or token
        overlap >= 0.5, pass.
        """
        if not listing.category or not item.category:
            return True
        label = _norm(listing.category)
        if label != _norm(item.category) and any(o is not item and _norm(o.category) == label for o in self.items):
            # the listing's shelf is exactly another item's shelf: it belongs to that item
            return False
        if item.labels_ok:
            return label in item.labels_ok
        want = _norm(item.category)
        if label == want or label in want or want in label:
            return True
        return _similar(label, want) >= 0.5

    def resolve_labels(self, item: Item) -> None:
        """Settles which seen category labels name the requested product. A single label is accepted when it overlaps
        the category (>=0.3 or containment) and otherwise left undecided; with several, one fast-model call picks
        them and only labels actually seen are kept. A failed or empty answer leaves the item undecided.
        """
        labels = sorted({l.category for l in item.pool.values() if l.category})
        if not labels or item.labels_ok is not None:
            return
        by_norm = {_norm(l): l for l in labels}
        if len(labels) == 1:
            item.labels_ok = set(by_norm) if (_similar(labels[0], item.category) >= 0.3 or _norm(item.category) in _norm(labels[0])
                                              or _norm(labels[0]) in _norm(item.category)) else set()
            if not item.labels_ok:
                item.labels_ok = None
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
        """The item's pool listings still worth buying: not out of stock when only stocked goods count, within the
        total budget at the item's quantity, and of an accepted category label. Sorted by screening label (yes,
        maybe, no), then price.
        """
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
        """When eight or more unopened, unscreened 'maybe' listings exist and the item has requirements, sends the
        first 30 as numbered rows (title, short text, category, price) to the fast screener in one call. Marks them
        screened and sets each to 'yes' or 'no' from the returned row numbers; the rest stay 'maybe'.
        """
        todo = [l for l in listings if not l.opened and l.screen == "maybe" and not getattr(l, "screened", False)]
        if len(todo) < 8 or not item.req_texts():
            return
        rows = []
        batch = todo[:30]
        for n, l in enumerate(batch, 1):
            rows.append({"n": n, "title": l.title[:140], "text": (l.text or "")[:ROW_CHARS], "category": l.category, "price": l.price})
        user = (f"REQUESTED PRODUCT: {item.category}\nREQUIREMENTS:\n- " + "\n- ".join(item.req_texts())
                + "\n\nROWS:\n" + json.dumps(rows, ensure_ascii=False))
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], SCREEN_PROMPT, user, 1200)
        for l in batch:
            l.screened = True
        if isinstance(answer, dict):
            for verdict in ("no", "yes"):
                for n in answer.get(verdict) or []:
                    if isinstance(n, (int, str)) and str(n).strip().isdigit() and 1 <= int(n) <= len(batch):
                        batch[int(n) - 1].screen = verdict

    def sheet_first(self) -> bool:
        """True for stores where judging can rely on the search row plus a property-table fetch: facts and open roles
        exist, search rows show price and stock, and no usable compare tool is available.
        """
        return (bool(self.roles.get("facts")) and bool(self.roles.get("open")) and bool(getattr(self, "rows_full", False))
                and not (self.roles.get("compare") and not getattr(self, "compare_off", False)))

    def sheet_only(self, item: Item | None) -> bool:
        """True when an item's listings may be opened by property table alone: sheet_first holds, the item is on
        neither side of a fit rule, and none of its requirements points at another listing's page.
        """
        if item is None or not self.sheet_first():
            return False
        if any(item is a or item is h for a, h in self.fit_pairs()):
            return False
        return not any(r.reference for r in item.requirements)

    def open_batch(self, item: Item | None, ids: list[str], floor: int = 0, page: bool = False) -> list[Listing]:
        """Reads the given listing ids completely and returns them, folded into the item's pool if one is passed. Uses
        per-id table reads (table-only or thin-compare stores), compare batches (shrinking compare_limit on short or
        failed answers, requeueing dropped ids, disabling compare on repeated failure) or single opens plus a table
        fetch. Keeps `floor` steps; stops near wrap time.
        """
        ids = [i for i in ids if i]
        opened: list[Listing] = []
        if not ids:
            return opened
        compare = self.roles.get("compare") if not getattr(self, "compare_off", False) else None
        facts_tool = self.roles.get("facts")
        open_tool = self.roles.get("open")
        if compare and not getattr(self, "compare_checked", False):
            self.compare_checked = True
            most = ((self._schema("compare").get("properties") or {}).get(compare.get("ids") or "") or {}).get("maxItems")
            if isinstance(most, int) and 2 <= most < self.compare_limit:
                self.compare_limit = most
        pos = 0
        requeued: set[str] = set()
        sheets = not page and self.sheet_only(item)
        while pos < len(ids) and self.steps_left() > 0 and self.time_left() > WRAP_SECONDS:
            if (self.thin_compare and facts_tool and item is not None) or sheets:
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
                            self.merge(item, listing, False)
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
                        if 2 <= len(back) < self.compare_limit:
                            self.compare_limit = len(back)
                            self.remember()
                        requeued.update(dropped)
                        ids = ids[:pos] + dropped + ids[pos:]
                    continue
                if res["error"] or (res["ok"] and not _find_list_of_dicts(res["obs"])):
                    if self.compare_limit > 2 and len(chunk) > 2:
                        self.compare_limit = max(2, min(self.compare_limit, len(chunk)) // 2)
                        self.remember()
                        continue
                    self.compare_off = True
                    compare = None
                    continue
            if not open_tool:
                break
            cost = 2 if facts_tool else 1
            if floor and self.steps_left() - cost < floor:
                break
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

    def fetch_facts(self, lid: str) -> dict[str, Any] | None:
        """Calls the facts tool for one listing (one step) and extracts a property table, without empty values, via the
        learned mapping or, for a flat record of scalars, from the record itself. An empty dict means nothing
        recognizable came back; None means no facts tool or a failed call.
        """
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
        """Marks a listing opened using only a facts-tool fetch; title, price and description stay as the search row
        left them in the pool. None when the fetch failed, or returned an empty table for a listing not yet in the
        pool.
        """
        known = item.pool.get(lid)
        facts = self.fetch_facts(lid)
        if facts is None or (known is None and not facts):
            return None
        listing = Listing(lid)
        listing.facts = facts
        return self.merge(item, listing, True)

    def drop_label_restatements(self, item: Item) -> None:
        """Removes request-origin 'is' requirements whose value words all belong to the item's accepted category labels
        and whose property no read in-category listing carries (three or more such listings needed), since such a
        value only restates the category. Logs each removal.
        """
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
        """With three or more read in-category listings, renames a plain-word required property (no digits) that none
        of them names to the key that most often holds the required value among them, when that key leads uniquely.
        Each requirement is aligned once; its new text forces a fresh judgement. Logs 'prop_aligned'.
        """
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
        """Judges opened listings not yet judged on the current requirements (after label and property clean-up). Old
        failures on live requirements stand without a call; the rest go four at a time with related pages and store
        fields to the fast judge, in parallel. Answers map back to requirement texts, plus field_fails; a value no
        record states is unsure.
        """
        self.drop_label_restatements(item)
        self.align_props(item)
        reqs = item.req_texts()
        if not reqs:
            for l in listings:
                if l.opened and l.id not in item.verdicts:
                    item.verdicts[l.id] = {"failed": set(), "unsure": set()}
                    item.judged_for[l.id] = set()
            return
        todo = [l for l in listings if l.opened and not (item.judged_for.get(l.id, set()) >= set(reqs))]
        for l in list(todo):
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
                def canon(names: set[str]) -> set[str]:
                    """Maps the judge's requirement references back to requirement texts: 'R<n>' or a bare number in range picks
                    that requirement, other text the most similar one when overlap exceeds 0.2; anything else is kept verbatim.
                    """
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
                    t = r.text()
                    if t not in v["failed"] and t not in v["unsure"] and t not in v["loose"] and self.unstated(item, l, r):
                        v["unsure"].add(t)
                    elif (t in v["failed"] or t in v["unsure"]) and self.list_states(l, r):
                        # the listing's own list for the property holds the value: met, whatever the judge said
                        v["failed"].discard(t)
                        v["unsure"].discard(t)

    @staticmethod
    def list_states(listing: Listing, req: Requirement) -> bool:
        """A fact named for the requirement's property lists several values, one of them exactly a required value."""
        if req.op not in ("is", "one of") or req.origin == "fit" or not req.value:
            return False
        wanted = {_norm(a) for a in req.value.split("|") if _norm(a)}
        for k, value in listing.facts.items():
            key = str(k).replace("-", " ").replace("_", " ")
            if not isinstance(value, str) or not re.search(r"[,;/|]", value):
                continue
            if not (_near(key, req.prop) or _similar(key, req.prop) >= 0.5):
                continue
            if wanted & {_norm(e) for e in re.split(r"[,;/|]", value) if _norm(e)}:
                return True
        return False

    @staticmethod
    def unstated(item: Item, listing: Listing, req: Requirement) -> bool:
        """Static: True when an 'is'/'one of' requirement not from a fit rule has no property field naming it, and none
        of its distinctive value words (product-name words and short alphabetic words left out) shares a four-letter
        stem with the listing's title, text, notes, category or properties.
        """
        if req.op not in ("is", "one of") or req.origin == "fit" or not req.value:
            return False
        if any(_near(str(k).replace("-", " "), req.prop) or _similar(str(k).replace("-", " ").replace("_", " "), req.prop) >= 0.5
               for k in listing.facts):
            return False
        common = _words(item.name) | _words(item.category)
        alts = [w for w in ({x for x in _words(v) - common if not (x.isalpha() and len(x) <= 3)} for v in req.value.split("|")) if w]
        if not alts:
            return False
        text = {w[:4] for w in _words(" ".join([listing.title or "", listing.text or "", listing.notes or "", listing.category or "",
                                                " ".join(f"{k} {v}" for k, v in listing.facts.items())]))}
        return not any({w[:4] for w in a} & text for a in alts)

    def field_table(self, item: Item) -> list[dict[str, Any]]:
        """Summarises how this store fills each digit-free 'is'/'one of' requirement once three in-category listings
        are read with tables: for requirements some listing carries under a near key, the keys, carrier count,
        element value counts and ids of listings whose field holds the wanted words.
        """
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
        """Requirement texts a listing is failed on without a model: it has a field near the property, other read
        listings write the wanted value in that field, and none of its own field elements hold the wanted value's
        four-letter stems. Uses the given field table or builds one.
        """
        out = set()
        for row in (self.field_table(item) if table is None else table):
            own = [k for k in listing.facts if _near(k, row["prop"])]
            if not own or not (row["writers"] - {listing.id}):
                continue
            elems = [frozenset(_tokens(e)) for k in own for e in re.split(r"[,;/|]", str(listing.facts[k])) if _tokens(e)]
            stem = lambda words: {w[:4] for w in words}
            if elems and not any(stem(w) <= stem(e) for w in row["wants"] for e in elems):
                out.add(row["req"])
        return out

    def field_lines(self, table: list[dict[str, Any]]) -> str:
        """Formats the field table for the judge prompt: one line per requirement with up to three key names, carriers
        out of read listings, the eight most common written values with counts, and how many listings write the
        required value.
        """
        lines = []
        for row in table:
            seen = " | ".join(f"{v} ({n})" for v, n in sorted(row["values"].items(), key=lambda x: -x[1])[:8])
            lines.append(f"- {row['req']}: field {', '.join(repr(k) for k in row['keys'][:3])} on {row['carriers']} of "
                         f"{row['read']} read listings; values written: {seen}; {len(row['writers'])} listing(s) write the "
                         "required value itself")
        return "\n".join(lines)

    def judge_all(self) -> None:
        """Fans judge out over all items in parallel threads; each item is judged on those of its current candidates
        that are already opened.
        """
        _parallel([lambda it=it: self.judge(it, [l for l in self.candidates(it) if l.opened]) for it in self.items])

    def related_pages(self, item: Item, listing: Listing, limit: int = 2) -> list[dict[str, Any]]:
        """Up to `limit` other known listings, this item's pool first, tied to this one by part codes: a code its text
        or notes cite is in their title, or their text or notes cite a code from its title. Returns each one's id,
        short title and up to 400 characters of text for the judge.
        """
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
        """Cached final check of a listing for an item: _verify's result is stored per item, listing, current
        requirement texts and property table, so repeating the same question costs no new model call.
        """
        key = (item.index, listing.id, tuple(item.req_texts()), json.dumps(listing.facts, sort_keys=True, default=str))
        cache = self.__dict__.setdefault("verified", {})
        if key not in cache:
            cache[key] = self._verify(item, listing)
        return cache[key]

    def _verify(self, item: Item, listing: Listing, strict: bool = False) -> bool:
        """Asks the verify model (fast model fallback) whether a listing meets every requirement, the quantity and the
        budget from its full record. No answer means True unless strict; strict also needs fits exactly true. A
        negative answer with problems adds them to the listing's failed set, mapped to the closest requirement text,
        and returns False.
        """
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

    def can_message(self) -> bool:
        """True when a message tool exists, the message budget (if the request set one) is not used up, and at least
        one step remains. Messages reserved for the ending are not considered here; can_ask does that.
        """
        if "message" not in self.roles:
            return False
        if self.message_budget is not None and self.messages_sent >= self.message_budget:
            return False
        return self.steps_left() > 0

    def held_messages(self) -> int:
        """Messages to keep in reserve for procedure steps of kind message at the before_order or before_finish
        moments, each counting at least one, capped by the messages still left. Zero without a message tool or
        procedure.
        """
        if "message" not in self.roles or not self.procedure:
            return 0
        def need(moments: tuple[str, ...]) -> int:
            """Sums the message counts, at least one each, of message-kind procedure steps whose moment is one of those
            given.
            """
            return sum(max(1, int(st.get("messages") or 1)) for st in self.procedure
                       if st["kind"] == "message" and st["moment"] in moments)
        held = need(("before_order", "before_finish"))
        if self.message_budget is not None:
            held = min(held, max(0, self.message_budget - self.messages_sent))
        return held

    def pending_messages(self) -> int:
        """Messages still reserved for the ending: zero when the cart is non-empty and this exact basket was already
        shown to the shopper, otherwise the count from held_messages.
        """
        return 0 if self.cart and self.confirmed_sig == self.basket_sig() else self.held_messages()

    def can_ask(self) -> bool:
        """True when a question can go out without eating into the messages reserved for the ending's procedure steps,
        given that a message is possible at all.
        """
        if not self.can_message():
            return False
        return self.message_budget is None or self.messages_sent < self.message_budget - self.held_messages()

    def gate(self, moment: str, intent: str) -> bool | None:
        """Has the plan model decide which procedure steps are due now from request, procedure, intent, basket,
        allowances, non-ending tools and recent history. Runs up to two actions on non-ending tools with schema-
        checked arguments, keeping steps for owed messages. Three tries per moment; True if a step ran, False if
        none due or no room, None on no answer.
        """
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
                + f"\n\nMOMENT: {moment} (the assistant is about to {intent})"
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
            self.note("gate_step", moment=moment, tool=name, why=str(act.get("why") or "")[:160], ok=res["ok"])
            ran = True
        if not ran:
            self.note("gate_clear", moment=moment)
        return ran

    def _trim_args(self, prop: dict[str, Any], value: str) -> str:
        """Cuts a string argument hard at its schema property's maxLength when one is declared; otherwise returns it as
        is.
        """
        limit = prop.get("maxLength")
        return value[:limit] if isinstance(limit, int) and len(value) > limit else value

    def basket_sig(self) -> tuple:
        """Sorted tuple of (listing id, quantity) for cart lines with a non-zero quantity, identifying the basket's
        state for confirmations.
        """
        return tuple(sorted((lid, int(q or 0)) for lid, q in self.cart.items() if q))

    def confirm_basket(self) -> bool:
        """Sends one message listing each cart line (quantity, title up to 70 chars, unit price) and the total, asking
        the shopper to confirm checkout. Needs a message allowance and two steps; counts the message and records the
        basket as confirmed. True once sent, even if the call failed.
        """
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
        return True

    @staticmethod
    def spoken_noun(item: Item) -> str:
        """How a message names the product: the request reader's product noun, never a store phrase such as "the
        listing from <Category>"."""
        for raw in (item.noun, item.category, item.name):
            text = re.sub(r"\s+", " ", str(raw or "")).strip()
            text = re.sub(r"^(?:\d+\s*(?:x\s*)?)?(?:of\s+)?(?:the\s+|a\s+|an\s+)?(?:same\s+)?(?:listings?|items?|products?)\s+"
                          r"(?:from|in|of)\s+(?:the\s+)?", "", text, flags=re.I)
            text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.I).strip(" .,;:")
            if text and "listing" not in text.lower() and len(text.split()) <= 5:
                return text if re.search(r"[A-Z]{2,}|\d", text) else text.lower()
        return "item"

    def ask_set(self, item: Item, props: list[str]) -> tuple[dict[str, Any] | None, str]:
        """One message about one product, naming one property or listing several; returns the reply as read (None
        when it could not be read) and its text."""
        spec = self.roles["message"]
        noun = self.spoken_noun(item)
        if len(props) == 1:
            text = f"Which {props[0]} would you like the {noun} to have?"
        else:
            text = f"What {', '.join(props[:-1])} or {props[-1]} would you like for the {noun}?"
        text = self._trim("message", spec["text"], text)
        res = self.call(spec["name"], {spec["text"]: text})
        self.messages_sent += 1
        if not res["ok"]:
            return None, ""
        obs = res["obs"]
        reply = obs if isinstance(obs, str) else None
        if isinstance(obs, dict):
            strings = [v for v in obs.values() if isinstance(v, str)]
            reply = max(strings, key=len) if strings else _compact(obs, 400)
        if not reply:
            return None, ""
        known = [f"{r.prop}: {r.op} {r.value}" for r in item.requirements if r.origin == "shopper" and r.value]
        user = (f"ITEM: {noun}\nPROPERTY ASKED ABOUT: {'; '.join(props)}\nQUESTION: {text}\n"
                f"KNOWN NEEDS: {json.dumps(known)}\nREPLY: {reply}")
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL, PLAN_MODEL], REPLY_PROMPT, user, 600)
        if not isinstance(answer, dict):
            answer = None
        if answer and str(answer.get("kind") or "").lower() == "need":
            named = self.property_named(item, reply, props + [str(answer.get("property") or "")], str(answer.get("value") or ""))
            if named:
                answer["property"] = named
        self.note("reply", item=item.name, prop=" | ".join(props), reply=reply[:200], read=_compact(answer, 300))
        if answer and str(answer.get("kind") or "").lower() in ("need", "repeat"):
            item.told_lines = list(getattr(item, "told_lines", [])) + [" " + _norm(reply) + " "]
        return answer, reply

    @staticmethod
    def listed_index(props: list[str], answer: dict[str, Any] | None, reply: str) -> int | None:
        """Which listed property a reply's need belongs to: the one the reader names, else the only one the reply
        spells out, else the one most like the property the reader names; None when none fits."""
        if not answer:
            return None
        for key in ("about", "property"):
            said = str(answer.get(key) or "")
            same = [k for k, p in enumerate(props) if said and _near(p, said)]
            if same:
                return same[0]
        text = " " + _norm(reply) + " "
        spelled = [k for k, p in enumerate(props) if _norm(p) and " " + _norm(p) + " " in text]
        if len(spelled) == 1:
            return spelled[0]
        said = str(answer.get("about") or answer.get("property") or "")
        scored = sorted(((_similar(p, said), -k) for k, p in enumerate(props)), reverse=True)
        return -scored[0][1] if scored and scored[0][0] >= 0.5 else None

    def interview(self, questions: int) -> None:
        """Spend the questions. Each message is about ONE product and lists up to LIST_WIDTH of its likeliest
        properties, from the first message on, until the shopper asks for one thing at a time; after that each names
        one property (a strict world loses one message to a list wherever it comes, so it comes first). The products take turns: the one with the fewest
        replies that told or ruled out something goes next, a product whose need was told last."""
        heard_from: dict[int, int] = {it.index: 0 for it in self.items}
        sent: dict[int, int] = {it.index: 0 for it in self.items}
        spent = 0
        while (spent < questions and self.can_ask() and self.steps_left() > self.end_reserve()
               and self.time_left() > WRAP_SECONDS + 8):
            queue = self.plan_questions(LIST_WIDTH * len(self.items) + 2)
            if not queue:
                break
            seen: list[Item] = []
            for it, _ in queue:
                if it not in seen:
                    seen.append(it)
            item = min(seen, key=lambda it: (any(r.origin == "shopper" for r in it.requirements),
                                              heard_from[it.index], sent[it.index], seen.index(it)))
            mine: list[str] = []
            for it, p in queue:
                if it is item and not any(_near(p, q) or _similar(p, q) >= 0.5 for q in mine):
                    mine.append(p)
            width = 1 if self.lists is False or questions == 1 else LIST_WIDTH
            props = mine[:width]
            answer, reply = self.ask_set(item, props)
            spent += 1
            sent[item.index] += 1
            kind = str((answer or {}).get("kind") or "").lower()
            several = len(props) > 1
            if several and kind in ("unclear", "other", "") and _REFUSED.search(reply or ""):
                kind = "split"          # a request to take things one at a time, read as a muddle
            if kind == "split":
                # "one at a time" comes only from a shopper who refuses lists, even when it answers a single
                self.lists = False
            elif several and kind == "need":
                self.lists = True
            about = props[0]
            if kind in ("need", "repeat"):
                # the reply is about one listed property (a repeat names one told before, under another spelling).
                # Only that property counts as asked: which listed need a shopper tells first is not known, so the
                # others stay askable (they cost list places, never a message)
                k = self.listed_index(props, answer, reply) if several else 0
                heard = props[:1] if k is None else [props[k]]
                about = props[k] if k is not None else props[0]
                heard_from[item.index] += 1 if kind == "need" else 0
            elif kind == "split":
                heard = [] if several else props          # a refused list is asked again one property at a time
            elif kind in ("free", "unclear"):
                # a generic reply rules out only the FIRST listed property: live race-14 lists disclosed the earliest
                # hidden property 95% of the time at position 1 but 30-69% at positions 2-6 (less after a decoy), so
                # the next list starts one place further on
                heard = props[:1]
                heard_from[item.index] += 1
            else:
                heard = props[:1]
            for p in heard:
                if _norm(p) not in item.asked:
                    item.asked.append(_norm(p))
            self.note("interview", item=item.name, props=props, read=kind, lists=self.lists, heard=heard)
            if kind == "split" and answer and answer.get("steps"):
                # "one at a time" is how to ask, not a step to carry out before the order: kept as a step it held a
                # message back for the ending and the last question was never sent (a race-16 replica case)
                answer = dict(answer, steps=[])
            self.apply_reply_kept(item, about, answer)
            self.handle_lines()

    @staticmethod
    def told_already(item: Item, prop: str) -> bool:
        """Static: True for a multi-word property whose normalized name already occurred, as a whole phrase, in one of
        the item's recorded need or repeat replies.
        """
        name = _norm(prop)
        return len(name.split()) >= 2 and any(f" {name} " in line for line in getattr(item, "told_lines", []))

    def property_named(self, item: Item, reply: str, asked: list[str], value: str) -> str | None:
        """Detects a store property the reply mentions instead of any asked one: candidate keys come from the item's
        listing tables (four or more characters, sharing no word with the value, not near an asked name). Returns
        the single key spelled in the reply after dropping keys inside longer hits, else None.
        """
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
        hits = [k for k in hits if not any(k != o and " " + k + " " in " " + o + " " for o in hits)]
        return hits[0] if len(hits) == 1 else None

    def apply_reply(self, item: Item, prop: str, answer: dict[str, Any] | None) -> None:
        """Turns an interpreted reply into state: adds demanded steps, sets a new total budget, and for a need with a
        value or reference builds a shopper requirement. A reference already followed or a value already told is
        ignored as a repeat; otherwise it replaces the shopper's earlier one on that property, queues a value search
        and resolves a reference. 'free' marks it open.
        """
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
                req.reference["property"] = req.prop
                sig = _ref_sig(req.reference)
                if sig and any(sig in (getattr(r, "ref_sig", None), getattr(r, "ref_sig0", None))
                               or (r.reference and _ref_sig(r.reference) == sig) for r in item.requirements):
                    self.note("reference_repeat", item=item.name, prop=req.prop)
                    return
            if req.value and any(r.origin == "shopper" and _norm(r.value) == _norm(req.value) and r.op == req.op
                                 for r in item.requirements):
                return
            item.requirements = [r for r in item.requirements if not (r.key() == req.key() and r.origin == "shopper")]
            item.requirements.append(req)
            if req.op in ("is", "one of") and req.value and not req.reference and len(req.value) <= 40:
                item.query_next(f"{req.value.split('|')[0].strip()} {item.noun}")
            if req.reference and req.op in REF_OPS:
                self.resolve_reference(item, req)
        elif kind == "free":
            item.freed.append(_norm(str(answer.get("property") or prop)))

    def apply_reply_kept(self, item: Item, prop: str, answer: dict[str, Any] | None) -> None:
        """Applies a reply during the interview with ref_floor raised to the messages still left (two when unlimited)
        plus the end reserve, so a reference chain the reply starts cannot spend those steps; the floor is reset
        afterwards.
        """
        left = (self.message_budget - self.messages_sent) if self.message_budget is not None else 2
        self.ref_floor = max(0, left) + self.end_reserve()
        try:
            self.apply_reply(item, prop, answer)
        finally:
            self.ref_floor = 0

    def property_menu(self, item: Item) -> list[dict[str, Any]]:
        """Question menu for an item from the keys of its opened in-category listings (only unfailed ones when four or
        more), normalized and merged when two keys agree on 80% of shared listings. Drops keys already required or
        asked and variant selectors; ranks by carriers, halved when all share one value or all differ. Top 40 with
        counts and three sample values.
        """
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
                continue
            carriers = sum(1 for f in per if any(n in f for n in names))
            values: list[str] = []
            for f in per:
                v = next((f[n] for n in names if n in f), None)
                if v and v not in values:
                    values.append(v)
            weight = carriers * (0.5 if len(values) == 1 and carriers >= 3 else 1.0)
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
        """Properties that must not be asked for an item because the request's rules fix them: a condition's source
        property, properties a condition branch sets on the item, and 'compatibility' when it is the accessory side
        of a fit rule.
        """
        out = []
        for c in getattr(self, "conditions", []):
            if c["src"] is item:
                out.append(c["prop"])
            out += [str(b["property"]) for _, d, b in c["branches"] if d is item]
        if any(a is item for a, _ in self.fit_pairs()):
            out.append("compatibility")
        return list(dict.fromkeys(out))

    def plan_questions(self, count: int, model: str = PLAN_MODEL) -> list[tuple[Item, str]]:
        """Up to `count` (item, property) questions. The first call asks the plan model once to rank properties across
        items from the request, hints, decided properties and menus, and caches that order. Several items go to
        multi_plan; one item filters the cached order, puts its two most-carried properties second and fourth when
        count >= 3, and pads from the menu.
        """
        if count <= 0:
            return []
        plan: list[tuple[Item, str]] = []

        def taken(item: Item, prop: str) -> bool:
            """True when a property should not be planned for the item: near one already asked, near or similar to one the
            request decides, near an existing requirement's property, already used in a told reply, or already in this
            plan for that item.
            """
            if any(_near(prop, a) for a in item.asked):
                return True
            if any(_near(prop, d) or _similar(prop, d) >= 0.5 for d in self.decided_props(item)):
                return True
            if any(_near(prop, r.prop) for r in item.requirements) or self.told_already(item, prop):
                return True
            return any(it is item and _near(prop, p) for it, p in plan)

        if getattr(self, "planned", None) is None:
            blocks = []
            for item in self.items:
                blocks.append({"item": item.name, "category": item.category, "known_requirements": item.req_texts(),
                               "decided_by_the_request": self.decided_props(item),
                               "hints": item.hints, "already_asked": item.asked, "property_menu": self.property_menu(item)})
            total = self.message_budget if self.message_budget is not None else count
            user = ("REQUEST:\n" + self.request + "\n\nSUBJECTS THE REQUEST TELLS THE ASSISTANT TO RAISE: " + json.dumps(self.ask_about)
                    + "\n\nITEMS:\n" + json.dumps(blocks, ensure_ascii=False)
                    + f"\n\nQUESTION_COUNT = {max(total, count)}")
            answer = self.llm.json([model, FAST_MODEL if model != FAST_MODEL else BACKUP_MODEL], PLAN_PROMPT, user,
                                   min(2400, 700 + 45 * max(total, count)))
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
        """Plans questions for several items: each gets a queue of the planner's picks, its most-carried menu property
        moved to second place, then the rest of its menu, all filtered by `taken`. Items take turns, those without a
        told need and with fewer questions asked first, until `count` questions or the queues run out.
        """
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
        turn = sorted(first, key=lambda i: (i in told, len(by_index[i].asked), first.index(i)))
        plan: list[tuple[Item, str]] = []
        while len(plan) < count and any(queues[i] for i in turn):
            for i in turn:
                if queues[i] and len(plan) < count:
                    plan.append((by_index[i], queues[i].pop(0)))
        return plan

    def resolve_reference(self, item: Item, req: Requirement, hop: int = 0) -> None:
        """Follows a requirement pointing at another listing's page: searches a scratch item per description with
        seller, attribute and category queries, finds the page via _match_page, and has the fast model read the
        wanted, paired or redirected value. Redirects recurse up to 8 hops; a value makes the requirement plain (two
        limits for 'within'). Keeps max(4, ref_floor) steps.
        """
        ref = req.reference or {}
        if not getattr(req, "ref_sig0", None):
            req.ref_sig0 = _ref_sig(ref)
        floor = max(4, getattr(self, "ref_floor", 0))
        if hop > 7 or self.steps_left() < floor or self.time_left() < WRAP_SECONDS + 20:
            return
        seller = str(ref.get("seller") or "").strip()
        category = str(ref.get("category") or "").strip()
        attrs = ref.get("attributes") if isinstance(ref.get("attributes"), dict) else {}
        target_prop = str(ref.get("property") or req.prop).strip()
        key = json.dumps([_norm(seller), _norm(category), sorted((_norm(k), _norm(v)) for k, v in attrs.items())])
        scratch, checked = self.__dict__.setdefault("ref_search", {}).setdefault(
            key, (Item({"name": category or "listing", "category": category, "search_terms": []}, 99), set()))
        values = [str(v).strip() for v in attrs.values() if str(v).strip() and len(str(v).split()) <= 3]
        for v in attrs.values():
            if len(str(v).split()) > 3:
                values += [w for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", str(v)) if _norm(w) != _norm(seller)
                           and w not in values][:4]
        values.sort(key=lambda v: bool(re.search(r"\d", v)))
        attr_words = " ".join(values[:3])
        noun = (category.split()[-1] if category else "")
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
            """True when no seller was described, or the listing's seller or brand overlaps the described seller by at
            least 0.5, or the seller's name appears in its title, text or notes.
            """
            if not seller:
                return True
            return (_similar(l.seller, seller) >= 0.5 or _similar(l.brand, seller) >= 0.5
                    or _norm(seller) in _norm(l.title + " " + l.text + " " + l.notes))

        def overlap(l: Listing) -> int:
            """Scores a listing against the description: described attribute words found in its title, text, notes and
            properties, plus figures with units that only match once spaces are removed (a unit written right after the
            number).
            """
            words = _tokens(attr_words)
            if not words:
                return 0
            raw = l.title + " " + l.text + " " + l.notes + " " + json.dumps(l.facts)
            text = _tokens(raw)
            squeezed = _norm(raw).replace(" ", "")
            joined = sum(1 for v in values if re.search(r"\d", v) and not (_tokens(v) and _tokens(v) <= text)
                         and len(_norm(v).replace(" ", "")) >= 3 and _norm(v).replace(" ", "") in squeezed)
            return len(words & text) + joined

        def attrs_in_text(l: Listing) -> bool:
            """True when the words of every described attribute value all occur in the listing's title, notes, description
            or property pairs; False when no attributes were described.
            """
            if not attrs:
                return False
            text = _words(" ".join([l.title or "", l.notes or "", l.text or "", " ".join(f"{k} {v}" for k, v in l.facts.items())]))
            return all(_words(v) and _words(v) <= text for v in attrs.values() if str(v).strip())

        def attrs_match(l: Listing) -> bool:
            """True when the listing has been opened and every described attribute has a property whose key is near or
            >=0.5 similar to it and whose normalized value equals the described one exactly.
            """
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
            """Unchecked scratch-pool listings from the described seller, kept to the described category, or all of the
            seller's when none qualify and they are few. Sorted by attribute overlap, then seller likeness, then the
            store's own order.
            """
            mine = [l for l in scratch.pool.values() if l.id not in checked and seller_ok(l)]
            got = [l for l in mine if not category or self.category_ok(scratch, l)]
            if not got and seller and len(mine) <= 2 * self.compare_limit:
                got = mine
            order = {lid: i for i, lid in enumerate(scratch.pool)}
            return sorted(got, key=lambda l: (-overlap(l), not (_similar(l.seller, seller) >= 0.5), order.get(l.id, 0)))

        for q in queries:
            if page is not None or self.steps_left() < floor:
                break
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
            """Asks the fast model to read the matched page's record for the wanted property, including the shopper's
            relation wording when there is one; returns the parsed value, pairing, wanted value and redirect, or None.
            """
            user = (f"PROPERTY WANTED: {target_prop}\n" + (f"RELATION THE SHOPPER ASKED FOR: {relation}\n" if relation else "")
                    + "\nPAGE:\n" + json.dumps(page.record(3000), ensure_ascii=False))
            return self.llm.json([FAST_MODEL, BACKUP_MODEL], REFER_PROMPT, user, 600)

        thin = self.thin_compare and not page.facts
        if thin and not page.notes and self.steps_left() >= 3:
            page.facts, thin = self.fetch_facts(page.id) or {}, False
        read = read_page()
        if thin and self.steps_left() >= 3 and isinstance(read, dict) and not read.get("redirect") \
                and (not read.get("value") or paired):
            page.facts = self.fetch_facts(page.id) or {}
            if page.facts:
                read = read_page()
        self.note("reference", hop=hop, page=page.id, read=_compact(read, 200))
        redirect = read.get("redirect") if isinstance(read, dict) else None
        if isinstance(redirect, dict) and redirect:
            about = str(redirect.get("property") or target_prop)
            if not (_near(about, target_prop) or _similar(about, target_prop) >= 0.5
                    or _norm(target_prop) in _norm(about) or _norm(about) in _norm(target_prop)):
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
            value = read["paired"]
            self.note("relation_paired", page=page.id, relation=relation[:80], value=value)
        if value and req.op == "within reference":
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
                item.query_next(f"{req.value} {item.noun}")
        elif paired and isinstance(read, dict) and read.get("value"):
            self.note("pair_unstated", page=page.id, value=read.get("value"))
            req.ref_sig = _ref_sig(ref)
            req.value, req.op, req.reference = str(read["value"]), "is", None

    def _match_page(self, scratch: Item, found: list[Listing], ref: dict, seller: str, category: str, attrs: dict,
                    attrs_match: Any, attrs_in_text: Any) -> Listing | None:
        """Picks, from search candidates, the page matching a reference's seller, category and attributes. With a pick
        rule or several fitting, reads a few pages and asks the matcher once with the rule. While still unmatched,
        opens batches (single pages up to five rounds, compare up to three), takes an exact attribute match, else
        asks the matcher or keeps the only fully worded one.
        """
        page = None
        tried = 0
        pick = str(ref.get("pick") or "").strip()
        count = ref.get("count") if isinstance(ref.get("count"), int) else None
        if pick or (count or 0) > 1:
            reach = min(len(found), 3 * self.compare_limit, max(4, 2 * (count or 2)))
            for start in range(0, reach, self.compare_limit):
                if self.steps_left() < 3:
                    break
                todo = [x.id for x in found[start:min(start + self.compare_limit, reach)]
                        if not (scratch.pool.get(x.id) and scratch.pool[x.id].opened)]
                for l in self.open_batch(None, todo):
                    scratch.pool[l.id] = l
            records = [scratch.pool[l.id].record(1200) for l in found if l.id in scratch.pool and scratch.pool[l.id].opened]
            if records:
                user = ("DESCRIPTION: category " + category + "; seller " + seller + "; attributes " + json.dumps(attrs, ensure_ascii=False)
                        + (f"\nPICK: {pick}" if pick else "") + (f"\nLISTINGS THAT FIT: {count}" if count else "")
                        + "\n\nCANDIDATES:\n" + json.dumps(records, ensure_ascii=False))
                match = self.llm.json([FAST_MODEL, BACKUP_MODEL], MATCH_PROMPT, user, 600)
                if isinstance(match, dict) and match.get("id") in scratch.pool:
                    page = scratch.pool[match["id"]]
            found = [] if page is not None else [l for l in found if not (scratch.pool.get(l.id) and scratch.pool[l.id].opened)]
        single = not self.roles.get("compare") or getattr(self, "compare_off", False)
        size = 1 if single else self.compare_limit
        seen: list[Listing] = []
        while found and tried < (5 if single else 3) and page is None and self.steps_left() >= 3:
            batch = found[:size]
            found = found[size:]
            tried += 1
            opened = self.open_batch(None, [l.id for l in batch])
            for l in opened:
                scratch.pool[l.id] = l
            now = [scratch.pool[l.id] for l in batch if l.id in scratch.pool and scratch.pool[l.id].opened]
            exact = [l for l in now if attrs_match(l)]
            if exact:
                page = exact[0]
                break
            seen += now
            if single and found and len(seen) < 3:
                continue
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
                continue
            elif len(worded) == 1:
                page = worded[0]
            elif len(records) == 1 and not found:
                page = scratch.pool[records[0]["id"]]
        return page

    def handle_lines(self) -> bool:
        """Interprets unhandled shopper lines with one fast-model call and applies them: a new budget (checked against
        numbers in the lines, x100 or /100, or one near the old cap), dropped requirements and their failures, added
        shopper requirements, price or stock news that re-reads cart pages, and demanded steps. True when something
        changed.
        """
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
        """Appends a shopper-demanded process step to self.procedure for later gate decisions, normalizing its moment
        (before_finish by default) and kind. Skipped without a goal, once eight steps exist, or when a goal with 0.6
        overlap is listed; a message step before the order or finish sets confirm_first.
        """
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
        """Re-reads the page of every listing in the cart, while more than two steps remain, to pick up changed price
        and stock, and logs the values seen. Each read spends steps.
        """
        for lid, item in list(self.cart_item.items()):
            if self.steps_left() <= 2:
                return
            opened = self.open_batch(item, [lid], page=True)
            self.note("refreshed", id=lid, price=item.pool[lid].price if lid in item.pool else None,
                      stock=item.pool[lid].stock if lid in item.pool else None)

    def set_cart(self, lines: dict[str, int]) -> bool:
        """Sends the full desired cart to the cart_set tool as line objects (id key, plus the quantity key capped at
        the schema maximum when there is one). One step; a refusal is logged as 'cart_set_failed' and returns False.
        """
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
        """Puts a listing in the cart at the item's quantity. With only a whole-cart tool, re-sends the cart plus this
        line; otherwise calls the add tool with id, quantity capped at the schema maximum and the add mode value.
        Records line and item on success; on a refusal sets cart_faulty from bad_args, logs it and returns False.
        """
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
        """Takes a listing out of the cart: re-sends the cart without it through a whole-cart tool, or calls the add
        tool with its remove mode (passing the current quantity) or, lacking a mode, a quantity of zero or the
        schema's positive minimum. Updates the local cart on success; one step.
        """
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
        """Checks out the cart. With procedure steps and a non-empty cart, runs the before_order gate (again if a
        message is still owed), sends a basket confirmation when confirm_first holds for an unconfirmed basket and
        the gate did not report nothing due, and returns False if a change arrived. Then surfaces pointed part pages
        and calls checkout; True once finished.
        """
        spec = self.roles.get("checkout")
        if not spec:
            return False
        if self.procedure and self.cart:
            ran = self.gate("before_order", "check out the basket below as the order")
            if ran and self.pending_messages() and self.steps_left() >= 2:
                ran = self.gate("before_order", "check out the basket below as the order") or ran
            if ran is not False and self.confirm_first and self.basket_sig() != self.confirmed_sig:
                ran = self.confirm_basket() or ran
            if ran and (self.handle_lines() or self.price_or_stock_changed()):
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
        """Before ordering, gathers part codes that cart listings' text or notes refer the reader to, skipping their
        own codes and codes already in a known title, and runs one keyword search for up to five so those pages get
        shown. Skipped without codes or when steps are needed for owed messages; codes searched are remembered.
        """
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
                    continue
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
        """Ranks opened, judged, in-category listings (cart-refused ones excluded) by misses: failed and field-failed
        requirements, sold out, over the item cap and over the total budget each add one. Ties go to fewer unsure
        requirements, then lower price; returns the best or None.
        """
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
        """Orders one pick per item instead of declining, once per task: supplied picks, else the best buyable listing,
        else nearest_miss. Removes stale cart lines, adds missing picks and checks out. If checkout is refused and
        shopper lines changed something, re-runs choose_and_finish up to two rounds. True when an order went
        through.
        """
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
        self.nearest_rounds = getattr(self, "nearest_rounds", 0) + 1
        if self.handle_lines() and self.nearest_rounds <= 2 and self.steps_left() >= len(self.items) + 1:
            self.nearest_tried = False
            self.near_picks = None
            self.choose_and_finish()
            return self.finished
        return False

    def order_unsure(self) -> bool:
        """Without a stated nothing-fits rule, orders rather than declines when every item has a buyable fit or an
        unsure-only one (in stock, no field fail, no loose word match, not cart-refused) and the total fits the
        budget. Hands the picks to order_nearest; tried once, and never after a nearest attempt.
        """
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
        """Ends without buying unless the request wants the nearest pick (order_nearest) or, lacking a decline rule, an
        unsure order succeeds. Runs the before_decline gate once and calls the decline tool with fixed args and any
        required remark. On refusal handles lines and may re-choose, then empties the cart and retries. Returns
        finished.
        """
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
            self.decline_refused = True
            self.note("decline_refused", obs=_compact(res["obs"], 200), lines=res["lines"][:3])
            if self.handle_lines() and self.steps_left() >= len(self.items) + 2:
                self.choose_and_finish()
            if not self.finished:
                for lid in list(self.cart):
                    if self.steps_left() > 1:
                        self.cart_remove(lid)
                self.call(spec["name"], args)
        return self.finished

    def run(self) -> list[dict[str, Any]]:
        """Top-level driver: setup, then work, catching and logging any exception; whatever happened, an unfinished
        task is closed by force_finish. Logs the model cost, call count and steps used, and returns the trace.
        """
        try:
            if not self.setup():
                self.force_finish()
                return self.trace
            self.work()
        except Exception as exc:
            self.note("exception", error=repr(exc)[:300])
        try:
            if not self.finished:
                self.force_finish()
        except Exception as exc:
            self.note("exception", error=repr(exc)[:300])
        self.note("done", cost=round(self.llm.cost, 4), llm_calls=self.llm.calls, steps=self.step)
        return self.trace

    def end_reserve(self) -> int:
        """Steps kept free for the ending, unless reserve_override is set: items plus six, extra page looks when a
        facts tool exists without compare, up to four for procedure steps after the start and a confirmation, and
        one for cheapest-mode backup reads still to come.
        """
        if getattr(self, "reserve_override", None) is not None:
            return self.reserve_override
        looks = min(len(self.items), 3) if self.sheet_first() else 2
        return (len(self.items) + 6 + (looks if self.roles.get("facts") and not self.roles.get("compare") else 0)
                + min(4, sum(1 for st in self.procedure if st["moment"] != "start") + (1 if self.confirm_first else 0))
                + (1 if self.cheapest and not getattr(self, "backups_done", False) else 0))

    def work(self) -> None:
        """Main flow: question budget net of reserved messages; request references resolved with steps kept for
        questions and the ending; start gate; an even step share of explore and warm_up per item; the interview;
        leftover references. Unless in ask mode it then carts a provisional pick, judges all, explores items lacking
        a fit first, and calls choose_and_finish.
        """
        items = self.items
        n = len(items)
        questions = (self.message_budget if self.message_budget is not None else 3) - self.held_messages()
        if "message" not in self.roles or questions < 0:
            questions = 0

        self.ref_floor = questions + (1 if questions else 0) + self.end_reserve() + 2 * n
        for item in items:
            for req in list(item.requirements):
                if req.reference and req.op in REF_OPS:
                    self.resolve_reference(item, req)
        self.ref_floor = 0

        free = self.steps_left() - questions - (1 if questions else 0) - self.end_reserve()
        per_item = max(2, free // max(n, 1)) if free > 0 else 0

        if any(st["moment"] == "start" for st in self.procedure):
            self.gate("start", "start the task (search the store)")
            self.handle_lines()
            questions = max(0, min(questions, (self.message_budget - self.messages_sent - self.held_messages())
                                   if self.message_budget is not None else questions))

        for k, item in enumerate(items):
            share = per_item if k < n - 1 else max(per_item, self.steps_left() - questions - 1 - self.end_reserve())
            if share <= 0:
                break
            before = self.steps_left()
            self.explore(item, max(2, share // 2))
            self.warm_up(item, max(0, share - (before - self.steps_left())))

        # questions to the shopper (one product per message; lists until refused, singles after)
        if questions > 0 and self.can_ask():
            self.interview(questions)

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

        self.judge_all()
        order = sorted(items, key=lambda it: (bool(it.fits()), -len(it.pool)))
        for k, item in enumerate(order):
            share = max(2, (self.steps_left() - self.end_reserve()) // max(len(order) - k, 1))
            self.explore(item, share, final=True)

        self.choose_and_finish()

    def provisional_cart(self) -> None:
        """Once per task, with an empty cart and enough steps and time, judges all items and carts the best buyable
        pick of the first item that has one within the total budget, so store change notices arrive while steps
        remain. When no item has such a pick the cart stays empty and a later call may try again.
        """
        if getattr(self, "provisional_done", False) or self.cart or not (self.roles.get("cart_add") or self.roles.get("cart_set")):
            return
        if self.steps_left() - self.end_reserve() < 3 or self.time_left() < WRAP_SECONDS + 30:
            self.provisional_done = True
            return
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
        """Grows the set of opened in-category listings toward 12 (batch reads) or 5 so question planning sees typical
        property keys, within `allowance` steps and above the end reserve. Prefers screened-yes, then cheaper rows,
        runs a category search when few remain, and judges after each read. Skipped without a message tool or past
        the spend cap.
        """
        if self.steps_left() <= self.end_reserve() + 2 or "message" not in self.roles:
            return
        stop_at = self.steps_left() - (allowance if allowance is not None else self.steps_left())
        floor = max(stop_at, self.end_reserve() + 2)
        target = 12 if self.batch_open() else 5

        def spare() -> list[Listing]:
            """Unopened listings of the item in an accepted category and not known to be sold out, ordered by screening
            label (yes, maybe, no) and then price.
            """
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
        """Search-and-open loop for one item within `steps`, the end reserve, time and spend. Each round handles lines,
        screens and judges; it stops at two fits, four with a best measure, while the cheapest rule needs every
        cheaper row seen. Opens rows likeliest to hold the values (value queries first when none do), then need
        queries, terms, fields, pages, suggestions.
        """
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
            if fits and not self.cheapest and len(fits) >= want_fits:
                break
            groups = self.value_groups(item)
            if repr(groups) != getattr(item, "value_sig", None):
                item.value_sig = repr(groups)
                item.value_ids = set()
                item.value_complete = False
            if groups:
                def hits(l: Listing) -> int:
                    """How many value groups an unopened row is taken to carry: all of them if a value query showed it, otherwise
                    those its own words show, raised (when the store needs every word) by groups covered by any query whose
                    results included it.
                    """
                    if l.id in item.value_ids:
                        return len(groups)
                    n = self.row_hits(l, groups)
                    if self.all_words and n < len(groups):
                        sent = [_words(s.split(":", 1)[1]) for s, ids in item.query_ids.items() if l.id in ids]
                        n = max(n, sum(1 for g in groups if any(_words(a) <= q for a in g for q in sent)))
                    return n
                unopened.sort(key=lambda l: (l.screen == "no", -hits(l), l.screen != "yes",
                                             l.price if l.price is not None else 1e18))
                due = unopened
                if fits and self.cheapest:
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
                    unopened = [l for l in unopened if l.id in item.value_ids or l.screen == "yes"]
                if not unopened and self.need_search(item, max(stop_at, self.end_reserve())):
                    continue
                if not unopened and best < 1e17 and not complete and self.roles.get("search_fields") and (self.roles["search_fields"].get("price_max")) \
                        and getattr(item, "sweep_below", None) != best and self.steps_left() > max(stop_at, self.end_reserve()) + 2:
                    item.sweep_below = best
                    fields = {"category": item.category, "price_max": max(0.0, best - 0.01)}
                    self.search(item, fields=fields, max_pages=3, budget=max(stop_at, self.end_reserve()))
                    continue
                if not unopened and not complete and getattr(item, "capped", False) and not getattr(item, "narrowed", False) \
                        and self.steps_left() > max(stop_at, self.end_reserve()) + 2:
                    item.narrowed = True
                    values = [r.value.split("|")[0].strip() for r in item.active() if r.op in ("is", "one of") and r.value]
                    if values:
                        for q in (" ".join(values + [item.category]), " ".join(values)):
                            if self.steps_left() <= max(stop_at, self.end_reserve()) + 1:
                                break
                            self.search(item, query=q, max_pages=2, budget=max(stop_at, self.end_reserve()))
                        continue
                if not unopened:
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
                batch = unopened[: self.compare_limit if self.batch_open() else 4 if self.thin_compare or len(self.items) >= 3 else 2]
                got = self.open_batch(item, [l.id for l in batch], floor=max(stop_at, self.end_reserve()))
                if not got:
                    break
                continue
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
        """Once per item, when the shopper told a plain value and the store is not known to need every word, runs up to
        three searches: category with a stated value and the told value, category with the told value, and the told
        value alone. Skips word sets already sent and stops at `floor`; True when a search ran.
        """
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
        """One fast-model call for fresh keyword queries for an item, given its requirements, the queries tried, any
        query length limit and up to 14 seen titles with fit flags for opened ones. Returns up to five new queries
        of at most 60 characters that were not tried or listed before.
        """
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
        """Reorders candidates by the request's 'best' measure using one plan-model call over up to 12 of them, cached
        per item and candidate id list when an order came back. Listings the model leaves out follow in their
        original order. Unchanged with no measure or fewer than two candidates.
        """
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
        """Cheapest mode, once, only when every item already has a buyable fit: per item, reads one batch of unopened,
        unrejected, in-stock, in-budget rows costing no more than the runner-up fit, cheapest first, and judges
        them, so a checked replacement exists if a pick changes after carting.
        """
        self.backups_done = True
        if not all(self.buyable(it) for it in self.items):
            return
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
        """Final choice: lines, cheapest backups, judging, then up to three verify tries per item (hosts and condition
        sources first), binding fit rules and conditions and trying other hosts or sources when a dependent finds
        none. Missing items get one search pass and an audit, else a decline; over budget, cheaper swaps and one
        hunt; then place_order.
        """
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
        tops = [(it, (self.rank_best(it, self.buyable(it)) or [None])[0]) for it in self.items
                if it.index not in accessories and it.index not in dependents]
        _parallel([lambda it=it, c=c: self.verify(it, c) for it, c in tops if c is not None])
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
        if self.cap is not None:
            def total() -> float:
                """Current basket cost for the budget check: the sum of the chosen listings' prices times their items'
                quantities, with an unknown price counted as zero.
                """
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
            if (total() > self.cap + 1e-6 and not getattr(self, "budget_hunt", False) and not self.spent()
                    and self.time_left() > WRAP_SECONDS + 30 and self.steps_left() - self.end_reserve() >= 4):
                # the known fits are all dear: look for cheaper fitting listings once before declining
                self.budget_hunt = True
                self.note("budget_hunt", total=total(), cap=self.cap)
                was = self.cheapest
                self.cheapest = True
                try:
                    order = sorted(self.items, key=lambda i: -((chosen[i.index].price or 0) * i.quantity))
                    for k, it in enumerate(order):
                        free = self.steps_left() - self.end_reserve()
                        if free < 3:
                            break
                        it.exhausted = False
                        self.explore(it, max(2, free // (len(order) - k)), final=True)
                finally:
                    self.cheapest = was
                self.choose_and_finish()
                return
            if total() > self.cap + 1e-6:
                self.decline_reason = "the cheapest fitting combination is over the budget"
                self.note("over_budget", total=total(), cap=self.cap)
                self.decline(self.decline_reason)
                return
        for item in self.items:
            item.chosen = chosen[item.index]
        self.place_order()

    def audit(self, missing: list[Item]) -> bool:
        """Before declining, asks the plan model to review each fitless item: up to 40 in-budget seen listings
        (verdicts and properties for read ones), requirements and queries tried. Strictly re-verified 'fits after
        all' are cleared; suggested rows are opened and up to two new queries searched, screened and opened. Uses a
        reduced reserve; True if anything new turned up.
        """
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
        """Matches free text to a requested item by token overlap with each item's name and category, returning the
        best one; None for empty text or a best overlap under 0.2.
        """
        name = str(name or "")
        if not name or not self.items:
            return None
        best = max(self.items, key=lambda it: max(_similar(it.name, name), _similar(it.category, name)))
        return best if max(_similar(best.name, name), _similar(best.category, name)) >= 0.2 else None

    def read_conditions(self, raw: Any) -> list[dict[str, Any]]:
        """Parses the reader's conditional rules into {src, prop, values, branches}, each branch being (applies when
        the test holds, dependent item, requirement spec). Rules without a known source item, values or a branch on
        another item are dropped. Branch values are added to the dependent item's search terms; the rules are
        logged.
        """
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
        """Tri-state test of a condition on one listing. A property field matching the condition's property decides:
        True if it holds all words of some value, False if it shares no word with any. Otherwise one fast-model call
        reads the record. Cached per condition, listing and table; None when undecided.
        """
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
        """Rebuilds the item's condition-origin requirements from the conditions whose source item has a pick in
        `chosen` and a known outcome. When they change, re-judges its opened candidates and, if nothing is buyable
        while steps and time allow, explores again.
        """
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
        """Resolves each fit rule's accessory and host names to two distinct items by name overlap (>= 0.2 each) and
        returns the (accessory, host) pairs; empty with fewer than two items.
        """
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
        """Ties an accessory item to the chosen host listing: its fit requirement is rewritten to cite the host page's
        title, text and notes, series codes on that page and codes candidates point to are searched, unread series
        members opened, and after re-judging, listings whose own code is in the series lose doubt or failure on the
        fit.
        """
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
                    for key in ("unsure", "weak", "failed"):
                        if key in v:
                            v[key].discard(fit_req)

    @staticmethod
    def in_series(listing: Listing, series: list[str]) -> bool:
        """Static: True when the series head of any part code in the listing's title (the letters-and-digits part
        before a hyphen) is one of the given series.
        """
        return any(h in series for h in _SERIES.findall(listing.title or ""))

    def place_order(self, attempt: int = 0) -> None:
        """Syncs the cart to the chosen listings and checks out, within three attempts and eight rounds. A refused add
        marks the pick 'cart refused' and re-chooses, or declines when the tool rejects its arguments. Reads carted
        pages, handles lines, re-judges picks; a change, sold-out pick, over-budget total or refused checkout leads
        to rechoose and another attempt.
        """
        self.order_rounds = getattr(self, "order_rounds", 0) + 1
        if attempt > 2 or self.finished or self.order_rounds > 8:
            return
        for lid in list(self.cart):
            if not any(it.chosen and it.chosen.id == lid for it in self.items):
                self.cart_remove(lid)
        for item in self.items:
            if item.chosen and item.chosen.id not in self.cart:
                if not self.cart_add(item, item.chosen):
                    if getattr(self, "cart_faulty", False):
                        self.note("cart_faulty")
                        self.decline("the cart tool refuses its arguments")
                        return
                    item.verdicts.setdefault(item.chosen.id, {"failed": set(), "unsure": set()})["failed"].add("cart refused")
                    item.chosen = None
        if any(it.chosen is None for it in self.items):
            self.choose_and_finish()
            return
        if self.sheet_first():
            for item in self.items:
                l = item.chosen
                if l is not None and not l.page_read and self.steps_left() > 2 + self.pending_messages():
                    self.open_batch(item, [l.id], page=True)
        elif self.steps_left() > 2 + self.pending_messages() and self.items and self.items[0].chosen:
            first = self.items[0].chosen
            self.open_batch(self.items[0], [first.id])
        changed = self.handle_lines()
        for item in self.items:
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
        changed = self.handle_lines()
        if not changed:
            self.refresh_cart_items()
            self.handle_lines()
        self.rechoose()
        if not self.finished:
            self.place_order(attempt + 1)

    def price_or_stock_changed(self) -> bool:
        """True when any chosen listing is now known to be sold out, or the chosen listings' total at their quantities
        exceeds the budget.
        """
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
        """After a change, re-judges each item and explores again when it lacks an in-stock fit or the cheapest rule
        applies (resetting need search and price sweep). Takes each item's first fit (always under the cheapest
        rule, else only if the old pick dropped out), reapplies conditions, and falls back to choose_and_finish on a
        gap or over-budget total.
        """
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
        """Last-resort ending: fills missing or sold-out picks with each item's first buyable fit; if all items have
        one, the total is within budget and steps cover the cart changes plus checkout, syncs the cart and orders.
        Otherwise, or if ordering fails, declines with the reason 'forced'.
        """
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
        """Filters item.fits (ordered cheapest-first when the request wants the cheapest) down to listings not flagged
        sold out and not failed by field_fails against the current field table.
        """
        table = self.field_table(item)
        return [f for f in item.fits(cheapest=self.cheapest) if f.stock is not False and not self.field_fails(item, f, table)]


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Entry point the benchmark calls: builds an Episode from the problem payload, runs it until the task is
    closed one way or another, and returns the episode's trace of logged events.
    """
    return Episode(problem_data).run()


__all__ = ["agent_main"]
