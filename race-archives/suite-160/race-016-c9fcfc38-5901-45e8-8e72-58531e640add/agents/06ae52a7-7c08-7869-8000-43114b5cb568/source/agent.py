from __future__ import annotations

import collections
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

# ------------------------------------------------------------------------------------------------ settings
STRONG = ["anthropic/claude-sonnet-5.5", "openai/gpt-6-sol", "anthropic/claude-haiku-5.5"]
FAST = ["openai/gpt-6-luna", "anthropic/claude-haiku-5.5", "deepseek/deepseek-v4-flash"]
JUDGE = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "deepseek/deepseek-v4-flash"]
CLOCK_LIMIT = 268.0        # seconds of the 300 s task budget we allow ourselves
SPEND_SOFT = 0.03          # USD per task after which strong calls fall back to fast models
SPEND_HARD = 0.05          # after this only the cheapest models are called
SPEND_STOP = 0.08          # after this no more model calls (run caps: $2 qualifying, $5 race)
GAP = 0.25                 # least gap between two proxy requests of this task (proxy cap is shared per client IP)
ID_RX = re.compile(r"^[A-Za-z]{1,3}-[0-9a-f]{6,}$")
WORDS_RX = re.compile(r"[A-Za-z][A-Za-z'\-]+")


# ------------------------------------------------------------------------------------------------ small helpers
def jdump(x: Any, cap: int = 4000) -> str:
    s = json.dumps(x, ensure_ascii=False, separators=(",", ":"), default=str)
    return s if len(s) <= cap else s[:cap] + "…"


def first_json(text: Any) -> Any:
    """First JSON object or array inside a model reply (fences and prose tolerated)."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"```\s*$", "", t)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = t.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(t)):
                ch = t[i]
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
                            return json.loads(t[start:i + 1])
                        except Exception:
                            break
            start = t.find(opener, start + 1)
    return None


def num(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"-?\d+(?:\.\d+)?", x.replace(",", ""))
        if m:
            return float(m.group(0))
    return None


def norm(s: Any) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s or "").lower()).strip()


def words(s: Any) -> list[str]:
    return [w.lower() for w in WORDS_RX.findall(str(s or ""))]


def backoff(attempt: int, status: Any, salt: int) -> float:
    """Retry pause: exponential, longer after 429/503, spread by a value taken from the request (not the clock)."""
    base = 1.5 if status in (429, 503) else 0.6
    spread = 0.75 + 0.5 * ((salt * 7919 + attempt * 104729) % 101) / 100.0
    return min(6.0, base * (2 ** attempt)) * spread


# ------------------------------------------------------------------------------------------------ clock
class Clock:
    def __init__(self, limit: float = CLOCK_LIMIT):
        self.t0 = monotonic()
        self.limit = limit

    def used(self) -> float:
        return monotonic() - self.t0

    def left(self) -> float:
        return self.limit - self.used()


# ------------------------------------------------------------------------------------------------ models
class Brain:
    """All model access: fallback ladders, pacing shared with the world client, spend tracking."""

    def __init__(self, clock: Clock, pacer: "Pacer"):
        self.clock = clock
        self.pacer = pacer
        self.spend = 0.0
        self.calls = 0
        self.dead: set[str] = set()
        self.lock = threading.Lock()
        self.log: list[dict[str, Any]] = []

    def chat(self, models: list[str], messages: list[dict[str, Any]], max_tokens: int = 900,
             want_json: bool = True, cap_s: float = 40.0, tools: list | None = None) -> Any:
        ladder = list(models)
        if self.spend > SPEND_STOP:
            return None
        if self.spend > SPEND_HARD:
            ladder = FAST[:1]
        elif self.spend > SPEND_SOFT:
            ladder = [m for m in ladder if m in FAST or m in JUDGE] or ladder
        for model in ladder:
            if model in self.dead:
                continue
            left = self.clock.left()
            if left < 6:
                break
            client = ProxyClient(timeout=int(max(5, min(cap_s, left - 4))), max_retries=1, retry_delay=0.5)
            body: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            for attempt in range(2):
                self.pacer.wait()
                t = monotonic()
                res = client.post_verbose("/inference/chat/completions", body)
                data = res.data
                with self.lock:
                    self.calls += 1
                if isinstance(data, dict) and data.get("choices"):
                    cost = num((data.get("usage") or {}).get("cost"))
                    with self.lock:
                        self.spend += cost or 0.0
                    msg = data["choices"][0].get("message") or {}
                    self.log.append({"model": model, "s": round(monotonic() - t, 1), "cost": cost})
                    if tools:
                        return msg
                    text = msg.get("content") or ""
                    if not want_json:
                        return text
                    parsed = first_json(text)
                    if parsed is not None:
                        return parsed
                    break  # unparseable: next model
                status = (res.error or {}).get("status")
                self.log.append({"model": model, "err": status})
                if status in (400, 401, 402, 403, 404):
                    self.dead.add(model)
                    break
                if status in (429, 503):
                    self.pacer.slow()
                if self.clock.left() < 10:
                    break
                sleep(backoff(attempt, status, len(jdump(messages, 400))))
        return None

    def many(self, jobs: list[tuple], workers: int = 4) -> list[Any]:
        """Run several chat jobs at once: each job is (models, messages, max_tokens)."""
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            futs = [ex.submit(self.chat, j[0], j[1], j[2] if len(j) > 2 else 900) for j in jobs]
            return [f.result() for f in futs]


class Pacer:
    def __init__(self):
        self.last = 0.0
        self.slow_until = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            gap = 0.8 if monotonic() < self.slow_until else GAP
            w = self.last + gap - monotonic()
            if w > 0:
                sleep(w)
            self.last = monotonic()

    def slow(self):
        self.slow_until = monotonic() + 20


# ------------------------------------------------------------------------------------------------ world
class World:
    """One episode: numbered tool calls, replay-safe retries, a full trace and shopper-line capture."""

    def __init__(self, env: dict[str, Any], clock: Clock, pacer: Pacer):
        self.binding = env["binding"]
        pv = env["policy_view"]
        self.tools = pv.get("tools") or []
        self.tool_names = {t.get("function", t).get("name") for t in self.tools}
        self.query = pv.get("query") or ""
        self.opening = pv.get("opening")
        self.max_steps = int(pv.get("max_steps") or 30)
        self.finished = bool(pv.get("finished"))
        self.step = 0
        self.clock = clock
        self.pacer = pacer
        self.client = ProxyClient(timeout=40, max_retries=1, retry_delay=0.5)
        self.trace: list[dict[str, Any]] = []
        self.heard: list[str] = []        # every shopper/notice line seen, in order
        self.fresh: list[str] = []        # lines not yet consumed by the planner
        self.dialogue: list[dict[str, Any]] = []

    def left(self) -> int:
        return self.max_steps - self.step

    def call(self, name: str, args: dict[str, Any]) -> Any:
        if self.finished or self.step >= self.max_steps:
            return None
        self.step += 1
        body = {**self.binding, "step": self.step, "action": {"name": name, "args": args}}
        result = None
        for attempt in range(4):
            self.pacer.wait()
            self.client.timeout = int(max(3, min(30, self.clock.left() + 25)))
            res = self.client.post_verbose("/environment/call", body)
            if isinstance(res.data, dict) and "observation" in res.data:
                result = res.data
                break
            status = (res.error or {}).get("status")
            if status == 409:
                break
            if status in (429, 503):
                self.pacer.slow()
            if self.clock.left() < 3:
                break
            sleep(backoff(attempt, status, self.step * 31 + len(name)))
        if result is None:
            self.trace.append({"step": self.step, "tool": name, "args": args, "obs": None})
            return None
        obs = result.get("observation")
        self.finished = bool(result.get("finished"))
        self.trace.append({"step": self.step, "tool": name, "args": args, "obs": obs})
        self.dialogue.append({"role": "assistant", "tool_call": {"name": name, "arguments": args},
                              "environment_result": result})
        lines = shopper_lines(obs)
        for ln in lines:
            if ln not in self.heard[-40:]:
                self.heard.append(ln)
                self.fresh.append(ln)
        return obs

    def take_fresh(self) -> list[str]:
        out, self.fresh = self.fresh, []
        return out


# ------------------------------------------------------------------------------------------------ observation parsing
def is_id(v: Any) -> bool:
    return isinstance(v, str) and bool(ID_RX.match(v))


def find_rows(obs: Any, depth: int = 0) -> list[dict[str, Any]]:
    """Every dict (at any depth) that carries a product id value."""
    out: list[dict[str, Any]] = []
    if depth > 6:
        return out
    if isinstance(obs, dict):
        if any(is_id(v) for v in obs.values()):
            out.append(obs)
        for v in obs.values():
            if isinstance(v, (dict, list)):
                out.extend(find_rows(v, depth + 1))
    elif isinstance(obs, list):
        for v in obs:
            out.extend(find_rows(v, depth + 1))
    return out


KEY_HINTS = {
    "title": ("title", "name", "caption", "label", "heading"),
    "price": ("price", "cost", "amount", "unit"),
    "stock": ("stock", "orderable", "buy", "avail", "purchas", "sellable", "on hand", "shelf"),
    "category": ("category", "department", "section", "class", "shelf", "aisle", "range"),
    "brand": ("brand", "make", "maker", "company", "manufacturer"),
    "shop": ("shop", "seller", "store", "vendor", "merchant"),
}


def key_role(k: str, v: Any) -> str | None:
    kl = norm(k)
    if is_id(v):
        return "id"
    if isinstance(v, dict):
        return "facts"
    for role in ("price", "stock", "category", "brand", "shop", "title"):
        if any(h in kl for h in KEY_HINTS[role]):
            if role == "price" and num(v) is None:
                continue
            if role == "stock" and not isinstance(v, (bool, str)) and v is not None:
                continue
            if role in ("category", "brand", "shop", "title") and not isinstance(v, str) and v is not None:
                continue
            if role == "category" and isinstance(v, str) and len(v) > 80:
                continue
            return role
    return None


def stock_value(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.lower()
        if re.search(r"out|sold|unavail|back ?order|no\b|none|0 left|coming", s):
            return False
        if re.search(r"in stock|available|yes|true|on hand|ready", s):
            return True
    if isinstance(v, (int, float)):
        return v > 0
    return None


def parse_row(d: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"raw": d, "facts": {}}
    texts = []
    for k, v in d.items():
        role = key_role(k, v)
        if role == "id" and "id" not in row:
            row["id"] = v
        elif role == "facts":
            facts = {str(a): b for a, b in v.items() if isinstance(b, (str, int, float, bool)) or b is None}
            if len(facts) >= len(row["facts"]):
                row["facts"] = facts
        elif role == "price" and "price" not in row:
            row["price"] = num(v)
        elif role == "stock" and "stock" not in row:
            sv = stock_value(v)
            if sv is not None:
                row["stock"] = sv
        elif role and role not in row and isinstance(v, str):
            row[role] = v
        elif isinstance(v, str) and len(v) > 0:
            texts.append(v)
    if "title" not in row:
        cands = [t for t in texts if 8 <= len(t) <= 300]
        if cands:
            row["title"] = cands[0]
    return row


def merge_row(old: dict[str, Any] | None, new: dict[str, Any], detailed: bool) -> dict[str, Any]:
    if not old:
        new["detailed"] = detailed
        return new
    out = dict(old)
    for k, v in new.items():
        if k == "raw":
            out["raw"] = {**old.get("raw", {}), **v}
        elif k == "facts":
            if v:
                out["facts"] = {**old.get("facts", {}), **v}
        elif v is not None:
            out[k] = v
    out["detailed"] = old.get("detailed") or detailed
    return out


PRODUCT_TEXT_KEYS = ("title", "name", "caption", "about", "description", "desc", "text", "seller", "detail", "summary")


def shopper_lines(obs: Any) -> list[str]:
    """Natural-language lines in a result that are not product data: notices, refusals, shopper speech.
    Shape-agnostic: strings outside product rows with >= 4 words, or a list of such strings."""
    out: list[str] = []

    def visit(x: Any, depth: int, key: str):
        if depth > 6:
            return
        if isinstance(x, dict):
            row = any(is_id(v) for v in x.values())
            if row and depth > 0:
                return  # a product row
            for k, v in x.items():
                if row and (not isinstance(v, list) or any(not isinstance(e, str) for e in v)):
                    continue  # a lone product sheet: its fields are data; only a list of notes beside it is read
                visit(v, depth + 1, str(k))
        elif isinstance(x, list):
            for v in x:
                visit(v, depth + 1, key)
        elif isinstance(x, str):
            kl = key.lower()
            if any(h in kl for h in PRODUCT_TEXT_KEYS) and depth <= 1 and len(x) > 200:
                return
            if len(words(x)) >= 4 and not is_id(x):
                out.append(x.strip())

    visit(obs, 0, "")
    return out


def page_info(obs: Any) -> dict[str, Any]:
    """Next-page cursor and total count from a listing result, learned from structure."""
    info: dict[str, Any] = {"next": None, "total": None, "has_next_key": False}
    if not isinstance(obs, dict):
        return info
    body = obs
    # unwrap a 2-key notice wrapper
    if len(obs) == 2:
        inner = [v for v in obs.values() if isinstance(v, dict)]
        if len(inner) == 1 and not any(is_id(v) for v in obs.values()):
            body = inner[0]
    loose = []
    for k, v in body.items():
        kl = norm(k)
        if any(h in kl for h in ("next", "onward", "continue", "after", "cursor", "more", "token", "follow", "further",
                                 "forward", "resume", "marker", "bookmark", "ahead", "subsequent")):
            info["has_next_key"] = True
            if v is not None and not isinstance(v, (list, dict)):
                info["next"] = v
        elif any(h in kl for h in ("total", "count", "hits", "matched", "found", "number", "num")) and isinstance(v, int):
            info["total"] = v
        elif v is None or (isinstance(v, str) and 0 < len(v) <= 80 and not re.search(r"\s", v) and not is_id(v)):
            loose.append(v)
    # names vary from store to store: a listing reply whose only other scalar is a short token (or null) carries the cursor
    if not info["has_next_key"] and len(loose) == 1 and any(isinstance(v, list) for v in body.values()):
        info["has_next_key"] = True
        info["next"] = loose[0]
    return info


def plain_name(key: Any) -> str:
    """A store attribute key as a shopper would say it: separators to spaces, a leading code-like token dropped
    (one or two letters, or no vowel), a trailing "type" dropped when other words remain."""
    ws = re.sub(r"[._\-]+", " ", str(key or "")).strip().lower().split()
    while len(ws) > 1 and (len(ws[0]) <= 2 or not re.search(r"[aeiouy]", ws[0])):
        ws = ws[1:]
    if len(ws) > 2 and ws[-1] == "type":
        ws = ws[:-1]
    return " ".join(ws)


STOPW = {"of", "the", "a", "an", "for", "with", "to", "in", "on", "and"}


def vtoks(s: Any) -> set:
    """Value tokens: lowercase, parentheses dropped, stop words dropped, stemmed."""
    t = re.sub(r"\(.*?\)", " ", str(s or "").lower())
    return {stem(w) for w in re.findall(r"[a-z0-9]+", t) if w not in STOPW}


LEXICON_PROMPT = """A shopper requirement must be checked against store listings. Below are the attribute fields seen on
listings of this category with sample values. Decide which fields express the requirement's property and which values
satisfy it. Answer ONE JSON object:
{"fields": [exact field names, copied from the list, that express the property - including qualified or alternative
            names for the same property (for "tone" on a lamp, a field named "lamp tone" or "shade_tone" counts)],
 "accept": [values copied from the samples, plus the canonical wording, that SATISFY the requirement],
 "reject": [sample values of those fields that clearly do NOT satisfy it]}
For an is_not requirement, "accept" lists the EXCLUDED values. For at_least/at_most give fields only.
Be literal: a value satisfies only when it states the wanted value or an exact synonym or spelling variant ("grey" and
"gray"; "XL" and "extra large"), or says the same thing in other words (for "frizz control": "anti-frizz", "frizz
prevention"); a merely related value (a neighbouring shade, a broader style family) does not.
REQUIREMENT: """


def stem(w: str) -> str:
    w = w.lower()
    for suf in ("ies", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def one_letter_apart(a: str, b: str) -> bool:
    """Two spellings of one word (one letter dropped from a longer word of five or more letters)."""
    if len(a) < len(b):
        a, b = b, a
    if len(a) - len(b) != 1 or len(b) < 4:
        return False
    return any(a[:i] + a[i + 1:] == b for i in range(len(a)))


def field_matches(prop: str, key: str) -> bool:
    """A fact key carries the property when the property's words all appear in the key (qualifiers allowed); a word
    spelled one letter apart counts as the same word."""
    pw = {stem(w) for w in words(prop)}
    kw = {stem(w) for w in words(key.replace(".", " "))}
    return bool(pw) and all(w in kw or any(one_letter_apart(w, k) for k in kw) for w in pw)


def literal_pass(req: dict[str, Any], facts: dict[str, Any]) -> bool | None:
    """True when a matching field literally states the wanted value; None when undecided by code."""
    op = req.get("op") or "is"
    vals = {stem(w) for w in words(req.get("value"))}
    if not vals:
        return None
    hits = [v for k, v in facts.items() if field_matches(str(req.get("property")), str(k))]
    if not hits:
        return None
    stated = any(vals <= {stem(w) for w in words(v)} for v in hits)
    if op in ("is", "contains", "one_of"):
        return True if stated else None
    if op == "is_not":
        return False if stated else None
    return None


def schema_props(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool.get("function", tool)
    return ((fn.get("parameters") or {}).get("properties")) or {}


def schema_required(tool: dict[str, Any]) -> list[str]:
    fn = tool.get("function", tool)
    return list(((fn.get("parameters") or {}).get("required")) or [])


# ------------------------------------------------------------------------------------------------ prompts
ROLE_PROMPT = """You map the tools of a simulated web store to roles. The tool list (OpenAI function schemas) is below.
Answer with ONE JSON object only, keys exactly as listed; use null when no tool fits. Use exact tool and argument names.
{
 "search": {"tool": str, "query_arg": str, "fixed": {arg: value}} ,      // free-text keyword search over products
 "browse": {"tool": str, "category_arg": str, "fixed": {}} ,            // list products of one category/department/shelf
 "maker": {"tool": str, "arg": str, "fixed": {}} ,                       // filter by brand/maker, if any
 "price_cap": [{"tool": str, "arg": str}],                              // args that cap the max price on a listing tool
 "price_floor": [{"tool": str, "arg": str}],
 "paging": [{"tool": str, "arg": str, "kind": "page"|"offset"|"cursor", "first": int}], // next-page argument
 "details": {"tool": str, "id_arg": str, "fixed": {}} ,                  // the tool returning a product's full attributes/facts
 "view": {"tool": str, "id_arg": str, "fixed": {}} ,                     // another single-product view, if separate
 "batch": {"tool": str, "ids_arg": str, "fixed": {}} ,                   // a tool taking SEVERAL product ids at once
 "cart_add": {"tool": str, "id_arg": str, "qty_arg": str|null, "fixed": {}, "qty_mode": "add"|"set"},
 "cart_remove": {"tool": str, "id_arg": str, "qty_arg": str|null, "fixed": {}},
 "cart_view": {"tool": str, "fixed": {}},
 "checkout": {"tool": str, "fixed": {}},                                // places the order (final)
 "decline": {"tool": str, "remark_arg": str|null, "fixed": {}},          // ends with no purchase (final)
 "message": {"tool": str, "text_arg": str, "fixed": {}},                // sends a message to the shopper
 "status": {"tool": str, "fixed": {}}
}
Rules: "fixed" holds enum/mode argument values that select the role on a shared tool (e.g. a mode enum "typed" for
keyword search, an action enum "add"/"include" for cart_add, "remove"/"discard" for cart_remove, "checkout" vs
"decline" on one finishing tool). qty_mode is "set" when the tool sets the line quantity, "add" when it adds to it.
paging kind is "offset" when the argument is a start index/position, "page" when it is a page number, "cursor" when it
takes an opaque token (string) that a listing reply hands back for the next page.

TOOLS:
"""

SPEC_PROMPT = """You turn a shopper's request to a store assistant into a precise JSON task spec. Read every sentence:
requirements are often scattered among chatty story lines. Answer with ONE JSON object only:
{
 "items": [ {"name": "the product the shopper wants, in the shopper's words",
             "category": "the store department/category phrase the request names, in its exact words",
             "search_words": "1-3 words a listing title would contain: the product noun plus at most one required value word; never attribute names",
             "quantity": int,
             "requirements": [ {"property": "attribute name as the shopper names it",
                                 "op": "is" | "is_not" | "at_least" | "at_most" | "one_of" | "contains",
                                 "value": "the value (string; numbers keep their unit)",
                                 "text": "the exact request fragment"} ] } ],
 "budget": number | null,          // spending cap in the store's price unit as written (cents if the request says cents)
 "budget_scope": "total" | "per_unit",
 "objective": "cheapest" | "best" | "none",
 "best_measure": "what 'best' is measured by, or null",
 "in_stock_only": bool,
 "message_budget": int | null,     // max messages to the shopper, as stated
 "step_budget": int | null,
 "approval_required": bool,        // the shopper wants to confirm/approve the basket before checkout
 "approval_text": "fragment describing that procedure, or null",
 "procedures": ["every other stated rule of procedure, verbatim, e.g. how to decline"],
 "decline_no_names": bool,         // when declining, do not name or suggest products/substitutes
 "story_hints": ["product properties the story implies the shopper may care about but did not state as rules"],
 "multi_item_order": "how the shopper wants several items handled, or null"
}
Requirements hide in many phrasings - catch every one. Invented illustrations:
- "a kettle with a speckled finish" -> finish is speckled; "it's for a picnic" -> occasion is picnic;
  "the grip must be rubber" -> grip is rubber; "it has to be made in Peru" -> country of origin is Peru;
  "lamps of the art deco style" / "art-deco-style lamps" -> style is art deco (a style named next to the product is a
  condition on it, not the category).
- A side remark promising to verify a detail personally marks that detail as a HARD requirement.
- Exclusions ("skip anything whose lid is glass") -> op "is_not". Lower/upper bounds ("12 litres or more") ->
  at_least / at_most. Keep units in values. Use the attribute word the shopper uses as property.
What is NOT a requirement (never put these in "requirements"):
- the product type or the store category/range itself (that goes in name/category);
- the budget, price, stock/availability, message or step limits (they have their own keys);
- story details that are not stated as a condition on the product.
Bounds: "X or less / no more than X / up to X" -> at_most; "X or more / at least X" -> at_least.
Do not invent requirements; soft wishes go to story_hints. approval_required only when the shopper explicitly asks to
confirm/approve the basket or wait for their OK before checkout.

REQUEST:
"""

JUDGE_PROMPT = """You check store listings against a shopper's hard requirements. A listing passes a requirement only
when its record shows the requirement is met:
- Matching fields: attribute fields whose name is the property's name, or that name with extra qualifier words, or a
  usual alternative name for the same property. Ignore separators and case.
- PASS when a matching field states the value (by meaning, spelling variants allowed). FAIL when matching fields exist
  and none of them states it. A broad generic field does not override a specific one that states the value.
- When no matching field exists, the title or option text may show the value; otherwise "unsure".
- is_not: FAIL when a matching field states the excluded value. at_least/at_most: compare numbers in the same unit.
- Qualifiers in the requirement (pure, only, 100%) must be honoured.
Answer ONE JSON object: {"verdicts": [{"id": str, "verdict": "pass"|"fail"|"unsure", "failed": [property...], "why": "≤12 words"}]}

REQUIREMENTS:
"""

REPLY_PROMPT = """A store assistant asked a shopper a question; the shopper replied. Classify the reply and extract any
requirement it states. Answer ONE JSON object:
{"kind": "need" | "no_preference" | "unclear" | "too_many" | "other",
 "requirements": [{"item": "which item it applies to", "property": str, "op": "is"|"is_not"|"at_least"|"at_most"|"one_of"|"contains", "value": str}],
 "reference": null | {"property": str, "relation": "same"|"excluding"|"at_least"|"at_most",
                      "category": "category of the referenced listing", "shop": "seller/shop named or null",
                      "brand": "maker named or null", "filters": {"property": "value", ...}, "text": "verbatim"}}
- "unclear": the shopper did not understand the message. "too_many": the shopper wants one question at a time.
- "no_preference": the shopper does not mind about that property.
- Polarity: a firm positive ("it must be walnut", "only walnut will do") -> op "is"; an exclusion ("anything except
  walnut", "no walnut") -> op "is_not". Never turn a firm positive into is_not.
- "value" holds only the wanted value itself. Words saying the matter is decided ("closed", "fixed", "settled", "locked",
  "final", "no understudies") describe the shopper's resolve, not the product: "the base is locked, walnut and nothing
  else" -> value "walnut".
- If the wanted value has to be read from ANOTHER listing (the shopper points at some other product's page, seller or
  attributes instead of naming the value), fill "reference" and leave requirements empty.
QUESTION: {q}
REPLY: {r}"""

NOTICE_PROMPT = """While shopping for a customer, these new lines appeared in the store's results. Decide what each
changes. Answer ONE JSON object:
{"changes": [{"kind": "drop_requirement"|"new_requirement"|"budget"|"price_change"|"stock_out"|"refusal"|"approval"|"none",
              "property": str|null, "value": str|null, "budget": number|null, "item": str|null, "note": "≤15 words"}]}
- drop_requirement: the shopper no longer needs a stated property. budget: a new spending cap (number in the store's
  unit, e.g. 188 for "188 ¢"). price_change: a listing's price moved. stock_out: an item is no longer available.
- refusal: the store refused the last action (e.g. checkout blocked) – say why in note. approval: the shopper agreed.
CURRENT REQUIREMENTS: {req}
LINES:
{lines}"""


# ================================================================================================ Cartographer
# Code-first planner: the code owns the episode (explore -> verify -> ask -> commit -> repair); models are used as
# pure functions (map tools, compile the request, judge records, read replies and notices).

SORRY = "Sorry, nothing available meets all of your requirements, so I am not placing an order."


def low(s: Any) -> str:
    return str(s or "").lower()


class Cartographer:
    def __init__(self, problem: dict[str, Any]):
        env = problem["environment"]
        self.clock = Clock()
        self.pacer = Pacer()
        self.world = World(env, self.clock, self.pacer)
        self.brain = Brain(self.clock, self.pacer)
        self.tools = {t.get("function", t).get("name"): t for t in self.world.tools}
        self.roles: dict[str, Any] = {}
        self.spec: dict[str, Any] = {}
        self.rows: dict[str, dict[str, Any]] = {}       # id -> merged row
        self.item_of: dict[str, set] = {}               # id -> indexes of items it was found for
        self.verdict: dict[tuple, dict] = {}            # (item_idx, id) -> judge verdict
        self.asked: list[tuple[int, str]] = []          # (item_idx, property)
        self.messages_used = 0
        self.dropped: set[str] = set()
        self.cart: dict[str, int] = {}
        self.notes: list[str] = []
        self.order_tries = 0
        self.searched: set[str] = set()
        self.searched_info: dict[str, Any] = {}
        self.labels: dict[int, Any] = {}
        self.gone: set[str] = set()
        self.window = None
        self.lex: dict[tuple, Any] = {}
        self.qplan: dict[int, list] = {}
        self.audited = False
        self.last_stars: dict[str, Any] = {}

    # ------------------------------------------------------------------ setup
    def setup(self):
        q = self.world.query + ("\n\n" + jdump(self.world.opening) if self.world.opening else "")
        tools_txt = jdump([t.get("function", t) for t in self.world.tools], 12000)
        res = self.brain.many([
            (STRONG, [{"role": "user", "content": ROLE_PROMPT + tools_txt}], 1200),
            (STRONG, [{"role": "user", "content": SPEC_PROMPT + q}], 1500),
        ], workers=2)
        self.roles = self.check_roles(res[0] if isinstance(res[0], dict) else {})
        self.spec = res[1] if isinstance(res[1], dict) else {}
        self.patch_roles()
        items = self.spec.get("items") or []
        if not items:
            items = [{"name": "the product", "category": "", "search_words": " ".join(words(self.world.query)[:4]),
                      "quantity": 1, "requirements": []}]
        for it in items:
            it["quantity"] = int(num(it.get("quantity")) or 1)
            catw = {stem(w) for w in words(it.get("category")) + words(it.get("name"))}
            keep = []
            for r in it.get("requirements") or []:
                if not isinstance(r, dict) or not r.get("property"):
                    continue
                pw = norm(r.get("property"))
                if re.search(r"\b(price|budget|cost|stock|availability|category|department|range)\b", pw):
                    continue
                vw = {stem(w) for w in words(r.get("value"))}
                if pw in ("type", "product type", "kind") and vw and vw <= catw:
                    continue
                keep.append(r)
            it["requirements"] = keep
        self.spec["items"] = items
        mb = num(self.spec.get("message_budget"))
        self.msg_budget = int(mb) if mb is not None else 3
        self.budget = num(self.spec.get("budget"))
        self.cheapest = self.spec.get("objective") == "cheapest"
        self.best = self.spec.get("objective") == "best"
        self.approval = bool(self.spec.get("approval_required"))
        sb = num(self.spec.get("step_budget"))
        if sb and 3 <= sb < self.world.max_steps:
            self.world.max_steps = int(sb)

    def check_roles(self, r: dict[str, Any]) -> dict[str, Any]:
        """Keep only role entries whose tool and argument names exist."""
        out: dict[str, Any] = {}
        for role, v in r.items():
            vs = v if isinstance(v, list) else [v]
            good = []
            for e in vs:
                if not isinstance(e, dict) or e.get("tool") not in self.tools:
                    continue
                props = schema_props(self.tools[e["tool"]])
                ok = all((e.get(a) in props) for a in ("query_arg", "category_arg", "id_arg", "ids_arg", "text_arg",
                                                       "arg") if e.get(a))
                if e.get("qty_arg") and e["qty_arg"] not in props:
                    e["qty_arg"] = None
                if e.get("remark_arg") and e["remark_arg"] not in props:
                    e["remark_arg"] = None
                e["fixed"] = {k: val for k, val in (e.get("fixed") or {}).items() if k in props}
                if ok:
                    good.append(e)
            if good:
                out[role] = good if isinstance(v, list) else good[0]
        return out

    def patch_roles(self):
        """Fill roles the model missed from schema cues."""
        for name, t in self.tools.items():
            fn = t.get("function", t)
            txt = low(name) + " " + low(fn.get("description"))
            props = schema_props(t)
            req = schema_required(t)
            str_args = [a for a, s in props.items() if s.get("type") == "string" and not s.get("enum")]
            id_args = [a for a in str_args if re.search(r"id|code|target|listing|item|product|ref|sku", low(a))]
            if "message" not in self.roles and re.search(r"shopper|customer|ask|say|message|chat|note|write", txt) \
                    and str_args and not id_args and not re.search(r"search|find|query", txt):
                self.roles["message"] = {"tool": name, "text_arg": str_args[0], "fixed": {}}
            if "cart_view" not in self.roles and not props and re.search(r"cart|bag|basket|selection|items|picks", txt) \
                    and not re.search(r"check ?out|purchase|final|settle|order|stand ?down|give ?up|decline", txt):
                self.roles["cart_view"] = {"tool": name, "fixed": {}}
            if "checkout" not in self.roles and not req and re.search(r"check ?out|purchase|buy|settle|finali[sz]e|place|commit", txt) \
                    and not re.search(r"no purchase|give ?up|decline|stand ?down|abandon|cancel", txt):
                self.roles["checkout"] = {"tool": name, "fixed": {}}
            if "decline" not in self.roles and re.search(r"no purchase|give ?up|decline|stand ?down|abandon|without buying", txt):
                self.roles["decline"] = {"tool": name, "remark_arg": str_args[0] if str_args else None, "fixed": {}}
            if "details" not in self.roles and len(req) == 1 and req[0] in id_args \
                    and re.search(r"detail|spec|sheet|inspect|fact|propert|attribute|trait", txt):
                self.roles["details"] = {"tool": name, "id_arg": req[0], "fixed": {}}
        if "details" not in self.roles and "view" in self.roles:
            self.roles["details"] = self.roles["view"]
        add = self.roles.get("cart_add")
        if add and add.get("qty_arg"):
            qs = schema_props(self.tools[add["tool"]]).get(add["qty_arg"], {})
            txt = low(add["tool"]) + " " + low(self.tools[add["tool"]].get("function", {}).get("description"))
            if qs.get("minimum") == 0 or re.search(r"\bset\b|0 removes|zero removes", txt):
                add["qty_mode"] = "set"

    # ------------------------------------------------------------------ calling tools
    def fit(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        props = schema_props(self.tools[tool])
        out: dict[str, Any] = {}
        for k, v in args.items():
            if k not in props or v is None:
                continue
            s = props[k]
            typ = s.get("type")
            if s.get("enum") and v not in s["enum"]:
                cand = [e for e in s["enum"] if low(e) == low(v)]
                if not cand:
                    continue
                v = cand[0]
            if typ == "integer":
                v = int(round(num(v) or 0))
            elif typ == "number":
                v = num(v) or 0
            elif typ == "string":
                v = str(v)
                if s.get("maxLength"):
                    v = v[: int(s["maxLength"])]
            elif typ == "array" and not isinstance(v, list):
                v = [v]
            if typ in ("integer", "number"):
                if s.get("minimum") is not None:
                    v = max(v, s["minimum"])
                if s.get("maximum") is not None:
                    v = min(v, s["maximum"])
            if typ == "array" and s.get("maxItems"):
                v = v[: int(s["maxItems"])]
            out[k] = v
        for k in schema_required(self.tools[tool]):
            if k not in out and props.get(k, {}).get("enum"):
                out[k] = props[k]["enum"][0]
        return out

    def act(self, role: str, extra: dict[str, Any] | None = None, entry: dict | None = None) -> Any:
        e = entry or self.roles.get(role)
        if isinstance(e, list):
            e = e[0] if e else None
        if not e:
            return None
        args = dict(e.get("fixed") or {})
        args.update(extra or {})
        obs = self.world.call(e["tool"], self.fit(e["tool"], args))
        self.absorb(obs, detailed=role in ("details", "view", "batch"))
        return obs

    def absorb(self, obs: Any, detailed: bool, item_idx: int | None = None):
        for d in find_rows(obs):
            r = parse_row(d)
            if not r.get("id"):
                continue
            self.rows[r["id"]] = merge_row(self.rows.get(r["id"]), r, detailed and len(r.get("facts") or {}) > 0)
            if item_idx is not None:
                self.item_of.setdefault(r["id"], set()).add(item_idx)

    # ------------------------------------------------------------------ step plan
    def reserve(self) -> int:
        """Steps kept for the finish: cart lines, a cart view, checkout, and one full repair after a refusal."""
        n = len(self.spec["items"])
        return n + 2 + (n + 2) + (1 if self.approval else 0)

    def explore_left(self) -> int:
        q_left = max(0, self.msg_budget - self.messages_used - (1 if self.approval else 0))
        return self.world.left() - self.reserve() - q_left

    # ------------------------------------------------------------------ exploring
    def unit_cap(self, idx: int) -> float | None:
        if not self.budget:
            return None
        if self.spec.get("budget_scope") == "per_unit":
            return self.budget
        others = 0.0
        for j, it in enumerate(self.spec["items"]):
            if j != idx:
                cheap = [r.get("price") for c, r in self.ranked(j) if r.get("price") is not None]
                others += (min(cheap) if cheap else 0) * it["quantity"]
        return max(0.0, self.budget - others) / max(1, self.spec["items"][idx]["quantity"])

    def query(self, role: str, args: dict[str, Any], idx: int) -> tuple[list[str], dict[str, Any]]:
        obs = self.act(role, args)
        ids = []
        for d in find_rows(obs):
            rid = parse_row(d).get("id")
            if rid:
                ids.append(rid)
                self.item_of.setdefault(rid, set()).add(idx)
        info = page_info(obs)
        info["rows"] = len(ids)
        return ids, info

    def listing(self, role: str, value: str, idx: int, cap: float | None = None, pages: int = 1) -> dict[str, Any]:
        """List one category/search value, optionally capped by price, reading up to `pages` pages."""
        e = self.roles.get(role)
        out = {"ids": [], "total": None, "per_page": 0, "pages": 0, "complete": False}
        if not e or not value:
            return out
        key = f"{role}|{low(value)}|{cap}"
        if key in self.searched:
            return self.searched_info.get(key, out)
        self.searched.add(key)
        argname = e.get("query_arg") or e.get("category_arg") or e.get("arg")
        base = {argname: value}
        if cap is not None:
            for pc in self.roles.get("price_cap") or []:
                if pc.get("tool") == e["tool"]:
                    base[pc["arg"]] = int(cap) if float(cap).is_integer() else round(cap, 2)
        paging = [p for p in (self.roles.get("paging") or []) if p.get("tool") == e["tool"]]
        nxt = None
        for p in range(pages):
            if self.explore_left() <= 0 or self.clock.left() < 60:
                break
            args = dict(base)
            if p > 0:
                if not paging or nxt is None:
                    break
                args[paging[0]["arg"]] = nxt
            ids, info = self.query(role, args, idx)
            out["pages"] += 1
            out["ids"] += ids
            if info["total"] is not None:
                out["total"] = info["total"]
            out["per_page"] = max(out["per_page"], len(ids))
            nxt = info["next"]
            pk = paging[0] if paging else {}
            pschema = schema_props(self.tools[e["tool"]]).get(pk.get("arg"), {}) if pk else {}
            # a cursor argument only takes the token the reply hands back; never make one up
            numeric = pk.get("kind") != "cursor" and pschema.get("type") in ("integer", "number")
            if numeric and isinstance(nxt, str) and not nxt.strip().isdigit():
                nxt = None
            if nxt is None and paging and numeric and info["total"] and ids:
                seen = len(out["ids"])
                if seen < info["total"]:
                    nxt = (p + 2) if pk.get("kind") == "page" else seen + int(pk.get("first") or 0)
                    mx = schema_props(self.tools[e["tool"]]).get(pk["arg"], {}).get("maximum")
                    if mx is not None and nxt > mx:
                        nxt = None
            if not ids or nxt is None:
                out["complete"] = out["total"] is None or len(out["ids"]) >= (out["total"] or 0)
                break
        if out["total"] is not None and len(out["ids"]) >= out["total"]:
            out["complete"] = True
        out.update(role=role, base=base, cursor=None if out["complete"] else nxt, idx=idx)
        self.searched_info[key] = out
        return out

    def page_on(self, idx: int, below: float | None) -> bool:
        """Cheapest rule: keep paging the broadest unfinished listing while it still shows prices under `below`."""
        best = None
        for info in self.searched_info.values():
            if info.get("idx") != idx or info.get("complete") or info.get("cursor") is None or not info.get("ids"):
                continue
            prices = [self.rows.get(i, {}).get("price") for i in info["ids"]]
            prices = [p for p in prices if p is not None]
            if below is not None and prices and min(prices[-3:]) >= below:
                continue
            ontopic = sum(1 for i in info["ids"] if i in self.rows and self.on_topic(idx, self.rows[i]))
            if ontopic * 2 < len(info["ids"]):
                continue
            if best is None or (info.get("total") or 0) > (best.get("total") or 0):
                best = info
        if not best or self.explore_left() <= 1 or self.clock.left() < 75:
            return False
        e = self.roles[best["role"]]
        paging = [p for p in (self.roles.get("paging") or []) if p.get("tool") == e["tool"]]
        if not paging:
            return False
        args = dict(best["base"])
        args[paging[0]["arg"]] = best["cursor"]
        ids, info = self.query(best["role"], args, idx)
        best["ids"] += ids
        best["pages"] += 1
        best["cursor"] = info["next"] if ids else None
        if best["cursor"] is None:
            best["complete"] = True
        return bool(ids)

    def visible_max(self, role: str, per_page: int) -> int:
        e = self.roles.get(role) or {}
        paging = [p for p in (self.roles.get("paging") or []) if p.get("tool") == e.get("tool")]
        if not paging:
            return per_page
        s = schema_props(self.tools[e["tool"]]).get(paging[0]["arg"], {})
        if paging[0].get("kind") == "page" and s.get("maximum"):
            return per_page * int(s["maximum"])
        if s.get("maximum"):
            return int(s["maximum"]) + per_page
        return per_page * 3

    def label_for(self, idx: int) -> str | None:
        it = self.spec["items"][idx]
        want = set(words(it.get("category")) + words(it.get("name")) + words(it.get("search_words")))
        stems = {w[:5] for w in want}
        cats: dict[str, int] = {}
        for rid in self.item_of:
            if idx in self.item_of[rid] and self.rows.get(rid, {}).get("category"):
                c = self.rows[rid]["category"]
                cats[c] = cats.get(c, 0) + 1
        if not cats:
            return None
        named = {w[:5] for w in words(it.get("category")) if len(w) > 2}
        def score(c):
            cw = {w[:5] for w in words(c)}
            # a department the request names outranks any other shelf that merely shares the product noun
            return (len(cw & named), len(cw & stems), cats[c])
        best = max(cats, key=score)
        # a shelf that shares no stem with the product holds look-alikes found by a loose search, not the product
        return best if score(best)[1] > 0 else None

    def on_topic(self, idx: int, r: dict[str, Any]) -> bool:
        """Without a learned shelf label, keep only rows whose shelf shares a word stem with the wanted product."""
        if not r.get("category"):
            return True
        it = self.spec["items"][idx]
        want = {w[:5] for w in words(it.get("category")) + words(it.get("name")) + words(it.get("search_words")) if len(w) > 2}
        have = {w[:5] for w in words(r["category"]) + words(r.get("title")) if len(w) > 2}
        return bool(want & have)

    def head_noun(self, idx: int) -> list[str]:
        it = self.spec["items"][idx]
        label = self.labels.get(idx) or self.label_for(idx)
        for src in (label, it.get("name"), it.get("category")):
            ws = [w for w in words(src) if len(w) >= 4 and w not in ("with", "made", "pair", "some", "range")]
            if ws:
                return ws[-1:] if src != label else ws[:1]
        return words(it.get("search_words"))[:1]

    def short_query(self, idx: int, with_values: bool = False) -> str:
        it = self.spec["items"][idx]
        parts = []
        if with_values:
            for r in self.live_reqs(idx):
                if r.get("op") in ("is", "contains") and len(words(r.get("value"))) <= 3:
                    parts += words(r.get("value"))
                    break
        parts += words(it.get("search_words") or it.get("name") or it.get("category"))
        out, seen = [], set()
        for w in parts:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return " ".join(out[:4])

    def backoff_queries(self, idx: int) -> list[str]:
        it = self.spec["items"][idx]
        noun = self.head_noun(idx)
        out = []
        cat = [w for w in words(it.get("category")) if len(w) > 2 and w not in ("and", "range", "section")]
        if cat:
            out.append(" ".join(cat[:3]))
        for r in self.live_reqs(idx):
            if r.get("op") in ("is", "contains", "one_of") and 0 < len(words(r.get("value"))) <= 2:
                out.append(" ".join(dict.fromkeys(words(r.get("value")) + noun)))
                break
        out.append(" ".join(noun))
        for src in (it.get("name"), it.get("category")):
            ws = [w for w in words(src) if len(w) > 2]
            if ws:
                out += [" ".join(ws[-2:]), ws[-1]]
        if noun and noun[0].endswith("s") and len(noun[0]) > 4:
            out.append(noun[0][:-1])
        seen, res = set(), []
        for q in out:
            if q and q not in seen:
                seen.add(q)
                res.append(q)
        return res[:5]

    def explore(self, idx: int):
        it = self.spec["items"][idx]
        cap = self.unit_cap(idx)
        if self.roles.get("search"):
            res = self.listing("search", self.short_query(idx), idx)
            # keyword search may need every word to match: back off to fewer, more generic words until enough
            # rows from the wanted shelf are in hand (a search-only store has no other way in)
            # (a store with department listing gets there through the shelf walk below instead)
            browse = bool(self.roles.get("browse"))
            for q in self.backoff_queries(idx):
                if res["ids"] and (browse or len(self.pool(idx)) >= 8):
                    break
                res = self.listing("search", q, idx, pages=1 if browse else 2)
        label = self.label_for(idx)
        if self.roles.get("browse"):
            tried = []
            res = None
            for name in [label, it.get("category"), it.get("name")]:
                if not name or low(name) in tried:
                    continue
                tried.append(low(name))
                res = self.listing("browse", name, idx, cap=cap, pages=1)
                if res["ids"]:
                    label = name
                    break
            if res and res["ids"]:
                capped = any(pc.get("tool") == self.roles["browse"]["tool"] for pc in self.roles.get("price_cap") or [])
                if self.cheapest and capped and cap:
                    self.walk(idx, label, cap, res)
                elif not res["complete"]:
                    self.listing("browse", label, idx, cap=cap, pages=2)
        if self.roles.get("search"):
            noun = self.head_noun(idx)
            for q in self.live_reqs(idx):
                if q.get("op") in ("is", "contains", "one_of") and words(q.get("value")):
                    q2 = " ".join(dict.fromkeys([w for w in words(q.get("value")) if len(w) > 1][:3] + noun))
                    self.listing("search", q2, idx, pages=2)
                    break

    def walk(self, idx: int, label: str, cap: float, first: dict[str, Any]):
        """Cheapest rule: find a price cap whose capped category fits the visible pages (aiming near full), read it."""
        per = max(1, first["per_page"])
        vis = self.visible_max("browse", per)
        total = first["total"] if first["total"] is not None else 10 ** 6
        hi_cap, hi_n = cap, total          # smallest cap known to overflow (or the budget cap)
        lo_cap, lo_n = 0.0, 0              # largest cap known to fit
        best = None
        if total <= vis:
            best = (cap, total)
        res = first
        for _ in range(5):
            if best and best[1] >= vis * 0.55:
                break
            if self.explore_left() <= 4:
                break
            if best is None:
                prices = sorted(self.rows[r]["price"] for r in res["ids"] if self.rows.get(r, {}).get("price") is not None)
                frac = max(0.03, min(0.9, vis * 0.8 / max(1, hi_n)))
                guess = prices[int(frac * (len(prices) - 1))] if len(prices) >= 4 else hi_cap * frac
                new = guess if lo_cap < guess < hi_cap else (lo_cap + hi_cap) / 2
            else:
                new = (lo_cap + hi_cap) / 2
            new = float(int(new)) if new > 20 else round(new, 2)
            if new <= lo_cap or new >= hi_cap:
                break
            res = self.listing("browse", label, idx, cap=new, pages=1)
            t = res["total"] if res["total"] is not None else len(res["ids"])
            if t > vis:
                hi_cap, hi_n = new, t
            else:
                lo_cap, lo_n = new, t
                if t > 0:
                    best = (new, t)
        if not best:
            return
        self.window = (idx, best[0])
        key = f"browse|{low(label)}|{best[0]}"
        info = self.searched_info.get(key)
        if not info or not info["complete"]:
            self.searched.discard(key)
            self.listing("browse", label, idx, cap=best[0], pages=6)

    def pool(self, idx: int) -> list[dict[str, Any]]:
        it = self.spec["items"][idx]
        out = []
        label = self.labels.get(idx) or self.label_for(idx)
        self.labels[idx] = label
        for rid, r in self.rows.items():
            if idx not in self.item_of.get(rid, set()):
                continue
            if r.get("stock") is False or rid in self.gone:
                continue
            p = r.get("price")
            if self.budget and p is not None and p * it["quantity"] > self.budget + 1e-9:
                continue
            if label and r.get("category") and norm(r["category"]) != norm(label):
                continue
            if not label and not self.on_topic(idx, r):
                continue
            out.append(r)
        return out

    def reading_order(self, idx: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        wants = set()
        for r in self.live_reqs(idx):
            if r.get("op") in ("is", "contains", "one_of", "at_least", "at_most"):
                wants |= {w for w in words(r.get("value")) if len(w) > 2}
        def score(r):
            t = set(words(r.get("title"))) | set(words(jdump(r.get("facts"), 800))) | set(words((r.get("raw") or {}).get("description")))
            hit = len(wants & t)
            price = r.get("price") if r.get("price") is not None else float("inf")
            # rows whose own text shows the wanted value first; under a cheapest rule, cheapest first within each group
            # under a cheapest rule the cheapest listing must be read whatever its title says
            return (price, hit == 0) if self.cheapest else (-hit, price)
        return sorted(rows, key=score)

    def read_ids(self, ids: list[str]):
        if not ids:
            return
        if self.roles.get("batch"):
            e = self.roles["batch"]
            maxn = int(schema_props(self.tools[e["tool"]]).get(e["ids_arg"], {}).get("maxItems") or 8)
            for i in range(0, len(ids), maxn):
                if self.world.left() <= self.reserve():
                    break
                self.act("batch", {e["ids_arg"]: ids[i:i + maxn]})
            return
        for rid in ids:
            if self.world.left() <= self.reserve() or self.clock.left() < 60:
                break
            self.act("details", {self.roles["details"]["id_arg"]: rid})

    def expand(self, idx: int) -> bool:
        """Find more candidate rows for item idx when every listed one has been read. True if new rows arrived."""
        if self.explore_left() <= 1 or self.clock.left() < 75:
            return False
        it = self.spec["items"][idx]
        before = {r["id"] for r in self.pool(idx)}
        noun = words(it.get("search_words") or it.get("name") or it.get("category"))
        queries = []
        for q in self.live_reqs(idx):
            if q.get("op") in ("is", "contains", "one_of"):
                lx = self.lex.get((idx, norm(q.get("property")), norm(q.get("value")), q.get("op"))) or {}
                for v in [q.get("value")] + (lx.get("accept") or [])[:3]:
                    vw = [w for w in words(v) if len(w) > 1][:3]
                    if vw:
                        queries.append(" ".join(dict.fromkeys(vw + self.head_noun(idx))))
                        queries.append(" ".join(vw))
        label = self.labels.get(idx)
        tries = 0
        if self.roles.get("search"):
            for q in queries:
                if f"search|{low(q)}|None" in self.searched:
                    continue
                self.listing("search", q, idx, pages=2)
                tries += 1
                if {r["id"] for r in self.pool(idx)} - before or tries >= 2:
                    break
        if not ({r["id"] for r in self.pool(idx)} - before) and label and self.roles.get("browse"):
            key = f"browse|{low(label)}|None"
            if key not in self.searched:
                self.listing("browse", label, idx, pages=3)
        return bool({r["id"] for r in self.pool(idx)} - before)

    def verify(self, idx: int, want: int = 3, max_reads: int | None = None):
        """Read records in priority order and judge them in small batches until `want` candidates pass."""
        reads = 0
        while self.clock.left() > 70:
            self.judge(idx)
            passes = [r for c, r in self.ranked(idx) if c == 0]
            if len(passes) >= want:
                break
            cand = [r for r in self.reading_order(idx, self.pool(idx)) if not r.get("detailed")
                    and (idx, r["id"]) not in self.verdict]
            if self.cheapest and passes:
                top = passes[want - 1]["price"] if len(passes) >= want else None
                cand = [r for r in cand if top is None or (r.get("price") or 0) <= top]
            budget = self.explore_left() if max_reads is None else min(self.explore_left(), max_reads - reads)
            if budget <= 0:
                break
            if not cand:
                if self.cheapest and passes and self.page_on(idx, passes[0].get("price")):
                    continue
                if not self.expand(idx):
                    break
                continue
            per_call = 1
            if self.roles.get("batch"):
                e = self.roles["batch"]
                per_call = max(1, min(6, int(schema_props(self.tools[e["tool"]]).get(e["ids_arg"], {}).get("maxItems") or 6)))
            # one batch call reads several sheets for a single step; single-sheet reads cost a step each
            batch = cand[:per_call] if per_call > 1 else cand[:min(3, budget)]
            self.read_ids([r["id"] for r in batch])
            reads += len(batch)
            for r in batch:   # a record that came back without facts still gets judged on what we have
                self.rows[r["id"]]["detailed"] = True
        self.judge(idx)

    # ------------------------------------------------------------------ judging
    def live_reqs(self, idx: int) -> list[dict[str, Any]]:
        return [r for r in self.spec["items"][idx]["requirements"] if norm(r.get("property")) not in self.dropped]

    def compact(self, r: dict[str, Any]) -> dict[str, Any]:
        raw = {k: v for k, v in (r.get("raw") or {}).items() if not isinstance(v, (dict, list))}
        for k, v in list(raw.items()):
            if isinstance(v, str) and len(v) > 220:
                raw[k] = v[:220] + "…"
        return {"id": r["id"], "title": (r.get("title") or "")[:200], "price": r.get("price"), "in_stock": r.get("stock"),
                "category": r.get("category"), "facts": r.get("facts"), "other": raw}

    def lexicon(self, idx: int, req: dict[str, Any]) -> dict[str, Any]:
        """One model call per requirement maps it onto this store's fields and values; code then judges every record."""
        rows = [r for r in self.pool(idx) if r.get("facts")]
        samples: dict[str, set] = {}
        for r in rows:
            for k, v in (r.get("facts") or {}).items():
                if v not in (None, ""):
                    samples.setdefault(str(k), set()).add(str(v)[:60])
        key = (idx, norm(req.get("property")), norm(req.get("value")), req.get("op"))
        sig = sum(len(v) for v in samples.values())
        old = self.lex.get(key)
        if old and (old["sig"] >= sig * 0.7 or old["calls"] >= 3):
            return old
        pw = {stem(w) for w in words(req.get("property"))}
        def rel(k):
            return len(pw & {stem(w) for w in words(k.replace(".", " "))})
        ks = sorted(samples, key=lambda k: (-rel(k), -len(samples[k])))[:36]
        table = {k: sorted(samples[k])[:8] for k in ks}
        res = self.brain.chat(FAST, [{"role": "user", "content": LEXICON_PROMPT + jdump(req, 400) + "\nFIELDS:\n" + jdump(table, 6000)}], 700)
        lx = {"fields": [], "accept": [], "reject": []}
        if isinstance(res, dict):
            for f in ("fields", "accept", "reject"):
                lx[f] = [str(x) for x in (res.get(f) or []) if x not in (None, "")]
        if req.get("op") in ("is", "contains", "one_of", "is_not"):
            lx["accept"].append(str(req.get("value")))
        lx["sig"], lx["calls"] = sig, (old["calls"] + 1 if old else 1)
        self.lex[key] = lx
        return lx

    def star_field(self, idx: int, req: dict[str, Any], lx: dict[str, Any]) -> str | None:
        """The field that decides: the exact-named field when its values ever state the wanted value, else the matching
        field that states it most often in this category."""
        acc = [vtoks(a) for a in lx["accept"] if vtoks(a)]
        fset = {norm(f) for f in lx["fields"]}
        cnt: dict[str, int] = {}
        for r in self.pool(idx):
            for k, v in (r.get("facts") or {}).items():
                nk_ = norm(str(k).replace(".", " "))
                if (nk_ in fset or field_matches(str(req.get("property")), str(k))) and any(g <= vtoks(v) for g in acc):
                    cnt[nk_] = cnt.get(nk_, 0) + 1
        if not cnt:
            return None
        ex = norm(req.get("property"))
        return ex if ex in cnt else max(cnt, key=cnt.get)

    def check_req(self, r: dict[str, Any], req: dict[str, Any], lx: dict[str, Any], star: str | None = None) -> str:
        """'pass' | 'fail' | 'weak' (title evidence only) | 'unknown'."""
        facts = r.get("facts") or {}
        fset = {norm(f) for f in lx["fields"]}
        if star and (req.get("op") or "is") in ("is", "contains", "one_of"):
            sv = [str(v) for k, v in facts.items() if norm(str(k).replace(".", " ")) == star and v not in (None, "")]
            if sv:
                acc0 = [vtoks(a) for a in lx["accept"] if vtoks(a)]
                return "pass" if any(any(g <= vtoks(v) for g in acc0) for v in sv) else "fail"
        hits = [str(v) for k, v in facts.items() if v not in (None, "") and (norm(k) in fset or field_matches(str(req.get("property")), str(k)))]
        acc = [vtoks(a) for a in lx["accept"] if vtoks(a)]
        rej = [vtoks(a) for a in lx.get("reject", []) if vtoks(a)]
        op = req.get("op") or "is"
        def says(v, group):
            vt = vtoks(v)
            return any(g <= vt for g in group)
        if op in ("at_least", "at_most"):
            want = num(req.get("value"))
            vals = [num(v) for v in hits if num(v) is not None]
            if want is None or not vals:
                return "unknown"
            ok = any((x >= want) if op == "at_least" else (x <= want) for x in vals)
            return "pass" if ok else "fail"
        if op == "is_not":
            if any(says(v, acc) for v in hits):
                return "fail"
            return "pass" if hits else ("fail" if says(r.get("title"), acc) else "pass")
        if hits:
            if any(says(v, acc) for v in hits):
                return "pass"
            return "fail"
        # no field addresses the property: fall back to the listing's own title/description wording
        if says(r.get("title"), acc) or says((r.get("raw") or {}).get("description") or (r.get("raw") or {}).get("about"), acc):
            return "pass"
        return "unknown"

    def judge(self, idx: int, force: bool = False):
        reqs = self.live_reqs(idx)
        rows = [r for r in self.pool(idx) if r.get("detailed")]
        lxs = [self.lexicon(idx, q) for q in reqs]
        stars = [self.star_field(idx, q, lx) for q, lx in zip(reqs, lxs)]
        self.last_stars[str(idx)] = stars
        for r in rows:
            res = [self.check_req(r, q, lx, st) for q, lx, st in zip(reqs, lxs, stars)]
            failed = [q["property"] for q, x in zip(reqs, res) if x == "fail"]
            if failed:
                v = "fail"
            elif all(x == "pass" for x in res):
                v = "pass"
            else:
                v = "unsure"
            self.verdict[(idx, r["id"])] = {"verdict": v, "failed": failed,
                                            "weak": sum(1 for x in res if x == "weak"), "detail": res}

    def ranked(self, idx: int) -> list[tuple[int, dict[str, Any]]]:
        """Candidates for item idx with a rank class: 0 pass, 1 unsure; failed ones dropped."""
        out = []
        for r in self.pool(idx):
            v = self.verdict.get((idx, r["id"]))
            if not v:
                continue
            failed = [f for f in (v.get("failed") or []) if norm(f) not in self.dropped]
            if v.get("verdict") == "fail" and failed:
                continue
            if v.get("verdict") == "pass" or (v.get("verdict") == "fail" and not failed):
                cls = 0
            else:
                cls = 1 if v.get("weak") else 2
            out.append((cls, r))
        def key(x):
            cls, r = x
            price = r.get("price") if r.get("price") is not None else float("inf")
            return (cls, price) if self.cheapest else (cls, 0 if self.best else price * 0)
        return sorted(out, key=key)

    # ------------------------------------------------------------------ questions
    def question_menu(self, idx: int) -> list[str]:
        it = self.spec["items"][idx]
        flat = lambda x: norm(re.sub(r"[._\-]+", " ", str(x or "")))
        stated = {flat(r.get("property")) for r in it["requirements"]}
        rows = [r for r in self.pool(idx) if r.get("detailed") and r.get("facts")]
        stats: dict[str, dict[str, Any]] = {}
        for r in rows:
            for k, v in (r.get("facts") or {}).items():
                nk = flat(k)
                if not nk or any(nk == s or nk in s or s in nk for s in stated if s):
                    continue
                st = stats.setdefault(nk, {"n": 0, "vals": set(), "name": k})
                st["n"] += 1
                st["vals"].add(norm(v))
                st.setdefault("count", collections.Counter())[norm(v)] += 1
        for nk in list(stats):
            vals = stats[nk]["vals"]
            avg = sum(len(v) for v in vals) / max(1, len(vals))
            # free text, per-listing identifiers and near-unique values are not preferences a shopper states
            repeats = max(stats[nk]["count"].values()) > 1
            # variant strings and model codes ("navy / l", "st0829") are listing identity, not a shopper preference
            coded = sum(n for v, n in stats[nk]["count"].items() if re.search(r"[/,+]|\d", v)) / max(1, stats[nk]["n"])
            if avg > 40 or (stats[nk]["n"] >= 5 and len(vals) >= 0.9 * stats[nk]["n"]) or (stats[nk]["n"] >= 4 and not repeats) \
                    or (stats[nk]["n"] >= 2 and coded >= 0.5):
                del stats[nk]
        asked = {p for i, p in self.asked if i == idx}
        hints = [norm(h) for h in (self.spec.get("story_hints") or []) if 0 < len(words(h)) <= 3]
        def score(nk):
            # properties nearly every listing of the category carries are the likeliest shopper concerns
            st = stats[nk]
            cover = st["n"] / max(1, len(rows))
            div = 0.15 if len(st["vals"]) > 1 else 0.0
            hint = 0.35 if any(h and (h in nk or nk in h) for h in hints) else 0.0
            numeric = -0.1 if all(re.fullmatch(r"[\d.\s]+[a-z%]*", v or "") for v in st["vals"]) else 0.0
            return cover * 2 + hint + div + numeric
        menu = [nk for nk in stats if stats[nk]["n"] >= max(1, len(rows) // 4)
                and not any(nk == a or nk in a or a in nk for a in asked)]
        menu.sort(key=score, reverse=True)
        # drop keys that are sub-phrases of each other, keep the higher scored one
        out: list[str] = []
        for nk in menu:
            if any(nk in o.split() or o in nk.split() or nk in o or o in nk for o in out):
                continue
            out.append(nk)
        for h in hints:
            if h and h not in out and h not in asked and h not in stated and len(out) < 2:
                out.append(h)
        return [plain_name(stats[n]["name"]) if n in stats else n for n in out]

    def ask(self, idx: int, prop: str, plain: bool = False) -> dict[str, Any] | None:
        e = self.roles.get("message")
        if not e or self.messages_used >= self.msg_budget:
            return None
        name = re.sub(r"^(?:(?:a|an|the|some|something|one|any)\s+(?:from\s+(?:the\s+|your\s+)?)?)+", "",
                      str(self.spec["items"][idx].get("name") or "item").strip(), flags=re.I) or "item"
        text = f"What {prop} should it have?" if plain else f"What {prop} should the {name} have?"
        self.world.take_fresh()
        obs = self.act("message", {e["text_arg"]: text})
        self.messages_used += 1
        self.asked.append((idx, norm(prop)))
        reply = " ".join(shopper_lines(obs)) or jdump(obs, 600)
        self.world.take_fresh()
        res = self.brain.chat(FAST, [{"role": "user", "content": REPLY_PROMPT.replace("{q}", text).replace("{r}", reply)}], 500)
        return res if isinstance(res, dict) else None

    PLAN_PROMPT = """A shopper holds some firm preferences that they reveal only when asked about that exact property.
Given the request (story included) and the attribute properties that listings in the store carry (with how common each
is), order the properties by how likely the shopper holds an unstated firm preference about it. Use the story: what the
shopper says about appearance, who will use the product, where or how it will be used, and what went wrong last time all
point at particular properties. Prefer common properties. Firm preferences mostly concern what tells variants of the
product apart (what it is made of, how it looks, its key specification, who it is for) and seldom fit, size or
measurements. Never list properties the request already states.
Answer ONE JSON object: {"order": [property names copied from the list]}
REQUEST: """

    def plan_questions(self, idx: int) -> list[str]:
        menu = self.question_menu(idx)
        if len(menu) <= 1:
            return menu
        if idx in self.qplan:
            return [p for p in self.qplan[idx] if p in menu] + [p for p in menu if p not in self.qplan[idx]]
        rows = [r for r in self.pool(idx) if r.get("detailed") and r.get("facts")]
        cov = {}
        for p in menu[:10]:
            cov[p] = round(sum(1 for r in rows if any(norm(k) == norm(p) for k in r["facts"])) / max(1, len(rows)), 2)
        it = self.spec["items"][idx]
        prompt = (self.PLAN_PROMPT + self.world.query[:2500] + "\nITEM: " + str(it.get("name")) +
                  "\nSTATED: " + jdump([r.get("property") for r in it["requirements"]], 300) + "\nPROPERTIES (coverage): " + jdump(cov, 800))
        res = self.brain.chat(FAST, [{"role": "user", "content": prompt}], 300)
        order = [str(x) for x in (res.get("order") or [])] if isinstance(res, dict) else []
        order = [p for p in order if p in menu]
        self.qplan[idx] = order
        return order + [p for p in menu if p not in order]

    def ask_round(self):
        budget = self.msg_budget - (1 if self.approval else 0)
        n = len(self.spec["items"])
        order = list(range(n))
        rounds = 0
        while self.messages_used < budget and rounds < budget + 2 and self.clock.left() > 90:
            rounds += 1
            idx = order[(self.messages_used) % n] if n > 1 else 0
            menu = self.plan_questions(idx)
            menu = [p for p in menu if not any(norm(p) == a or norm(p) in a or a in norm(p) for i2, a in self.asked if i2 == idx)]
            if not menu:
                if n > 1 and any(self.question_menu(j) for j in order if j != idx):
                    order = [j for j in order if j != idx] + [idx]
                    continue
                break
            prop = menu[0]
            res = self.ask(idx, prop)
            if not res:
                continue
            self.take_reply(idx, prop, res)

    def take_reply(self, idx: int, prop: str, res: dict[str, Any]):
        for rq in res.get("requirements") or []:
            if isinstance(rq, dict) and rq.get("property") and rq.get("value") not in (None, ""):
                j = self.item_index(rq.get("item"), idx)
                self.spec["items"][j]["requirements"].append(
                    {"property": rq["property"], "op": rq.get("op") or "is", "value": str(rq["value"]), "hidden": True})
                self.reset_verdicts(j)
        ref = res.get("reference")
        if isinstance(ref, dict) and ref.get("property"):
            self.resolve_reference(idx, ref)

    def item_index(self, name: Any, default: int) -> int:
        if not name:
            return default
        best, score = default, 0
        for j, it in enumerate(self.spec["items"]):
            s = len(set(words(name)) & set(words(it.get("name")) + words(it.get("category"))))
            if s > score:
                best, score = j, s
        return best

    def reset_verdicts(self, idx: int):
        for k in [k for k in self.verdict if k[0] == idx]:
            del self.verdict[k]

    def resolve_reference(self, idx: int, ref: dict[str, Any]):
        """A requirement whose value lives on another listing: find that listing (following any listing it points
        on to) and read the value."""
        tmp = 10_000 + idx
        cat = ref.get("category") or ""
        shop = ref.get("shop") or ref.get("brand") or ""
        filt = ref.get("filters") or {}
        if self.roles.get("search") and shop:
            self.listing("search", " ".join([shop] + words(cat)[:2]), tmp)
        if self.roles.get("maker") and shop:
            e = self.roles["maker"]
            self.query("maker", {e["arg"]: shop}, tmp)
        if self.roles.get("browse") and cat:
            self.listing("browse", cat, tmp, pages=2)
        def cands():
            rs = [r for rid, r in self.rows.items() if tmp in self.item_of.get(rid, set())]
            if shop:
                hit = [r for r in rs if low(shop) in low(jdump(r.get("raw"), 4000))]
                rs = hit or rs
            fw = {stem(w) for v in filt.values() for w in words(v)}
            return sorted(rs, key=lambda r: -len(fw & {stem(w) for w in words(jdump(r.get("raw"), 3000) + jdump(r.get("facts"), 2000))}))
        val = None
        for hop in range(3):
            cs = cands()[:5]
            unread = [r["id"] for r in cs if not r.get("detailed")][:3]
            if unread and self.world.left() > self.reserve() + len(unread):
                self.read_ids(unread)
            cs = [self.rows[r["id"]] for r in cs]
            if not cs:
                break
            prompt = ("A shopper's requirement takes its value from another listing: " + jdump(ref, 1200) +
                      "\nCandidate listings (full records):\n" + "\n".join(jdump(self.compact(r), 1800) for r in cs) +
                      "\nFind the referenced listing (its shop/maker and EVERY filter must match). Read the value it states "
                      "for the property (the same-named field, or the number/value the requirement points to). If that "
                      "listing only points on to another listing for that detail, "
                      'give search words for that other listing. Answer ONE JSON object: {"id": str|null, "value": str|null, '
                      '"follow": "search words for the deferred listing" | null}')
            res = self.brain.chat(STRONG, [{"role": "user", "content": prompt}], 300)
            if not isinstance(res, dict):
                break
            if res.get("value"):
                val = res
                break
            if res.get("follow") and self.roles.get("search") and self.world.left() > self.reserve() + 2:
                self.listing("search", " ".join(words(res["follow"])[:4]), tmp)
                filt = {}
                shop = ""
                continue
            break
        if val and val.get("value"):
            rel = ref.get("relation") or "same"
            op = {"same": "is", "excluding": "is_not", "anything_but": "is_not", "at_least": "at_least", "at_most": "at_most"}.get(rel, "is")
            self.spec["items"][idx]["requirements"].append(
                {"property": ref["property"], "op": op, "value": str(val["value"]), "hidden": True, "from_ref": val.get("id")})
            self.reset_verdicts(idx)
            self.notes.append(f"ref {ref.get('property')} {op} {val['value']}")
        for rid in list(self.item_of):
            self.item_of[rid].discard(tmp)

    # ------------------------------------------------------------------ notices
    GONE_RX = re.compile(r"\bgone\b|none left|sold out|out of stock|unavailable|can.?t be (?:ordered|bought)|"
                         r"no longer (?:available|sold|in stock)|not available", re.I)

    def handle_notices(self, after_finish: bool = False) -> list[dict[str, Any]]:
        lines = self.world.take_fresh()
        if not lines:
            return []
        cart_titles = {rid: (self.rows.get(rid, {}).get("title") or "")[:80] for rid in self.cart}
        req = [{"item": it.get("name"), "reqs": self.live_reqs(j)} for j, it in enumerate(self.spec["items"])]
        prompt = (NOTICE_PROMPT.replace("{req}", jdump(req, 2000)).replace("{lines}", "\n".join(lines[-8:])) +
                  "\nCART (id: title): " + jdump(cart_titles, 1500) +
                  '\nFor stock_out/price_change also give "ids": the cart ids it concerns ("your suggestion/pick" = the cart item).')
        res = self.brain.chat(FAST, [{"role": "user", "content": prompt}], 600)
        changes = (res or {}).get("changes") if isinstance(res, dict) else None
        if not isinstance(changes, list):
            changes = []
        if not changes and any(self.GONE_RX.search(l) for l in lines):
            changes = [{"kind": "stock_out"}]
        out = []
        for c in changes:
            if not isinstance(c, dict):
                continue
            k = c.get("kind")
            if k == "drop_requirement" and c.get("property"):
                self.dropped.add(norm(c["property"]))
                for it in self.spec["items"]:
                    for r in it["requirements"]:
                        if norm(r.get("property")) and (norm(r["property"]) in norm(c["property"]) or norm(c["property"]) in norm(r["property"])):
                            self.dropped.add(norm(r["property"]))
                out.append(c)
            elif k == "budget" and num(c.get("budget")):
                self.budget = num(c.get("budget"))
                self.spec["budget_scope"] = "total"
                out.append(c)
            elif k == "new_requirement" and c.get("property") and c.get("value"):
                j = self.item_index(c.get("item"), 0)
                self.spec["items"][j]["requirements"].append({"property": c["property"], "op": "is", "value": c["value"]})
                self.reset_verdicts(j)
                out.append(c)
            elif k == "stock_out":
                ids = [i for i in (c.get("ids") or []) if i in self.cart] or list(self.cart)[:1]
                for i in ids:
                    self.gone.add(i)
                    if i in self.rows:
                        self.rows[i]["stock"] = False
                out.append(c)
            elif k in ("price_change", "refusal"):
                out.append(c)
        if after_finish and not out:
            out = [{"kind": "refusal"}]
        self.notes += [jdump(c, 160) for c in out]
        return out

    # ------------------------------------------------------------------ cart + finish
    def choose(self) -> dict[int, dict[str, Any]] | None:
        picks: dict[int, dict[str, Any]] = {}
        for idx in range(len(self.spec["items"])):
            rk = self.ranked(idx)
            if (not rk or rk[0][0] > 0) and self.explore_left() > 1 and self.clock.left() > 80:
                self.verify(idx, want=1)
                rk = self.ranked(idx)
            if rk and rk[0][0] >= 2 and self.explore_left() > 1 and self.clock.left() > 80 and self.expand(idx):
                self.verify(idx, want=1)
                rk = self.ranked(idx)
            # race worlds almost always hold a valid answer and give-ups are wrong far more often than not: an
            # on-shelf listing with no refuted condition is still the better bet than declining
            if not rk:
                return None
            picks[idx] = rk[0][1]
        if self.budget and len(picks) > 1:
            total = sum((p.get("price") or 0) * self.spec["items"][i]["quantity"] for i, p in picks.items())
            if total > self.budget:  # swap to cheaper candidates where possible
                for idx in picks:
                    alts = sorted([r for c, r in self.ranked(idx) if c == 0 and r.get("price") is not None],
                                  key=lambda r: r["price"])
                    if alts:
                        picks[idx] = alts[0]
        return picks

    def set_cart(self, picks: dict[int, dict[str, Any]]):
        want = {p["id"]: self.spec["items"][i]["quantity"] for i, p in picks.items()}
        add, rem = self.roles.get("cart_add"), self.roles.get("cart_remove")
        for rid, q in list(self.cart.items()):
            if rid not in want or want[rid] != q:
                if add and add.get("qty_mode") == "set" and add.get("qty_arg") and (not rem or rem["tool"] == add["tool"]):
                    self.act("cart_add", {add["id_arg"]: rid, add["qty_arg"]: 0})
                elif rem:
                    extra = {rem["id_arg"]: rid}
                    if rem.get("qty_arg") and rem["qty_arg"] not in (rem.get("fixed") or {}):
                        extra[rem["qty_arg"]] = q
                    self.act("cart_remove", extra)
                elif add and add.get("qty_mode") == "set" and add.get("qty_arg"):
                    self.act("cart_add", {add["id_arg"]: rid, add["qty_arg"]: 0})
                self.cart.pop(rid, None)
        for rid, q in want.items():
            if self.cart.get(rid) == q or not add:
                continue
            extra = {add["id_arg"]: rid}
            if add.get("qty_arg"):
                extra[add["qty_arg"]] = q
            obs = self.act("cart_add", extra)
            if obs is not None:
                self.cart[rid] = q

    def reconcile(self, obs: Any):
        """Make our cart record match what the store's cart view shows (ids and quantities)."""
        if obs is None:
            return
        seen: dict[str, int] = {}
        for d in find_rows(obs):
            rid = parse_row(d).get("id")
            if not rid:
                continue
            q = None
            for k, v in d.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and re.search(r"qty|quant|count|number|pieces|units|amount", low(k)) \
                        and not re.search(r"price|cost|total", low(k)):
                    q = int(v)
            seen[rid] = q if q is not None else self.cart.get(rid, 1)
        if seen or re.search(r"empty|no items|nothing", low(jdump(obs, 400))):
            self.cart = seen

    def summary(self, picks: dict[int, dict[str, Any]]) -> str:
        parts, total = [], 0.0
        for i, p in picks.items():
            q = self.spec["items"][i]["quantity"]
            pr = p.get("price") or 0
            total += pr * q
            parts.append(f"{q} x {(p.get('title') or '')[:90]} at {pr:g} each")
        return "; ".join(parts), total

    def approve(self, picks) -> bool:
        e = self.roles.get("message")
        if not e or self.messages_used >= self.msg_budget:
            return False
        lines, total = self.summary(picks)
        text = f"Here is the final basket for your approval: {lines}. Total {total:g}. Shall I place the order?"
        self.act("message", {e["text_arg"]: text[:780]})
        self.messages_used += 1
        return True

    AUDIT_PROMPT = """You are the final check before a shopping assistant checks out. Compare the planned basket with the
shopper's request (every stated condition and procedure), the shopper's later replies and the store's notices.
Answer ONE JSON object: {"ok": bool, "issue": "what is wrong, or empty", "swap": {"<item index>": "<candidate id>"} | null}
Use swap only with ids from that item's candidate list. Approve when the evidence supports the basket.
"""

    def audit(self, picks: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
        """One review of the basket; may swap a pick for another verified candidate it names."""
        if self.audited or self.clock.left() < 45:
            return picks
        self.audited = True
        cands = {}
        for idx in range(len(self.spec["items"])):
            cands[idx] = [self.compact(r) for c, r in self.ranked(idx)[:5] if c <= 1]
        payload = {"request": self.world.query[:3000], "procedures": self.spec.get("procedures"),
                   "items": [{"index": i, "name": it.get("name"), "quantity": it["quantity"], "requirements": self.live_reqs(i)}
                             for i, it in enumerate(self.spec["items"])],
                   "budget": self.budget, "objective": self.spec.get("objective"),
                   "basket": {i: self.compact(p) for i, p in picks.items()}, "candidates": cands,
                   "shopper_and_store_lines": self.world.heard[-10:]}
        res = self.brain.chat(STRONG, [{"role": "user", "content": self.AUDIT_PROMPT + jdump(payload, 14000)}], 400)
        if not isinstance(res, dict) or res.get("ok", True):
            return picks
        self.notes.append("audit: " + str(res.get("issue"))[:160])
        swap = res.get("swap") if isinstance(res.get("swap"), dict) else {}
        out = dict(picks)
        for k, rid in swap.items():
            try:
                i = int(k)
            except (TypeError, ValueError):
                continue
            if i in out and rid in self.rows and any(r["id"] == rid for _, r in self.ranked(i)):
                out[i] = self.rows[rid]
        return out

    def finish_order(self) -> bool:
        if not self.roles.get("checkout"):
            return False
        self.order_tries += 1
        self.act("checkout")
        return self.world.finished

    def decline(self):
        e = self.roles.get("decline")
        if not e:
            return
        extra = {e["remark_arg"]: SORRY} if e.get("remark_arg") else {}
        self.act("decline", extra)

    # ------------------------------------------------------------------ episode
    def run(self) -> list[dict[str, Any]]:
        self.setup()
        if not self.roles.get("checkout") or not self.roles.get("cart_add") or not (
                self.roles.get("browse") or self.roles.get("search")) or not (
                self.roles.get("details") or self.roles.get("batch")):
            self.notes.append("fallback: roles " + ",".join(self.roles))
            return Fallback(self).run()
        n = len(self.spec["items"])
        for idx in range(n):
            self.explore(idx)
            self.handle_notices()
        share = max(3, self.explore_left() // (2 * n))
        for idx in range(n):
            self.verify(idx, want=2, max_reads=share)
        self.handle_notices()
        self.ask_round()
        self.handle_notices()
        for idx in range(n):
            hidden = [r for r in self.spec["items"][idx]["requirements"] if r.get("hidden")]
            if hidden and not any(c == 0 for c, _ in self.ranked(idx)) and self.roles.get("search"):
                self.listing("search", self.short_query(idx, with_values=True) if False else
                             " ".join((words(hidden[-1]["value"]) + words(self.spec["items"][idx].get("search_words")))[:4]), idx)
            self.verify(idx, want=3)
        return self.commit()

    def commit(self) -> list[dict[str, Any]]:
        approved = False
        for attempt in range(5):
            if self.world.finished or self.world.left() <= 0:
                break
            picks = self.choose()
            if picks is None:
                for idx in range(len(self.spec["items"])):
                    if not self.ranked(idx):
                        self.verify(idx, want=1)
                picks = self.choose()
                if picks is None:
                    for idx in range(len(self.spec["items"])):
                        read = sum(1 for k in self.verdict if k[0] == idx)
                        if read < 4 and self.world.left() > 3 and self.clock.left() > 60:
                            self.expand(idx)
                            self.verify(idx, want=1)
                    picks = self.choose()
                if picks is None:
                    self.notes.append("decline: no candidate")
                    self.decline()
                    break
            if self.budget:
                _, total = self.summary(picks)
                if total > self.budget + 1e-6:
                    self.notes.append(f"over budget {total} > {self.budget}")
                    self.decline()
                    break
            if attempt == 0:
                picks = self.audit(picks)
            self.set_cart(picks)
            ch = self.handle_notices()
            if self.roles.get("cart_view") and self.world.left() > 2 and (attempt == 0 or ch or len(picks) > 1):
                self.reconcile(self.act("cart_view"))
                ch += self.handle_notices()
                if set(self.cart) != {p["id"] for p in picks.values()} and self.world.left() > 3:
                    self.set_cart(picks)
            if ch:
                self.after_change(ch)
                if self.choose() != picks:
                    continue
            if self.approval and not approved:
                approved = self.approve(picks)
                ch = self.handle_notices()
                if ch and any(c.get("kind") not in ("approval", "none") for c in ch):
                    self.after_change(ch)
                    if self.choose() != picks:
                        continue
            if self.finish_order():
                break
            ch = self.handle_notices(after_finish=True)
            self.after_change(ch)
        if not self.world.finished and self.world.left() > 0:
            if self.cart and self.roles.get("checkout") and self.order_tries < 6:
                self.act("checkout")
            if not self.world.finished:
                self.decline()
        return self.world.dialogue

    def after_change(self, changes: list[dict[str, Any]]):
        """Re-derive the fit set after a change notice or refusal."""
        kinds = {c.get("kind") for c in changes}
        if "price_change" in kinds and self.roles.get("view") and self.world.left() > self.reserve() - 2:
            for rid in list(self.cart)[:2]:
                self.act("view", {self.roles["view"]["id_arg"]: rid})
        if "refusal" in kinds and not ({"stock_out", "price_change", "budget", "drop_requirement"} & kinds):
            # unexplained refusal: check the cart lines' stock on the view tool
            e = self.roles.get("view") or self.roles.get("details")
            for rid in list(self.cart)[:2]:
                if e and self.world.left() > 2:
                    self.act("view" if self.roles.get("view") else "details", {e["id_arg"]: rid})
                    if self.rows.get(rid, {}).get("stock") is False:
                        self.gone.add(rid)
        for idx in range(len(self.spec["items"])):
            if "drop_requirement" in kinds or "new_requirement" in kinds:
                self.reset_verdicts(idx)
            self.judge(idx)
            need = 2 if self.cheapest and ("drop_requirement" in kinds) else 1
            if len([1 for c, _ in self.ranked(idx) if c == 0]) < need:
                self.verify(idx, want=need)

    def diagnostics(self) -> dict[str, Any]:
        return {"role": "assistant", "content": "diag " + jdump({
            "roles": {k: (v.get("tool") if isinstance(v, dict) else [x.get("tool") for x in v]) for k, v in self.roles.items()},
            "spec": {k: self.spec.get(k) for k in ("items", "budget", "objective", "approval_required", "message_budget")},
            "dropped": sorted(self.dropped), "gone": sorted(self.gone), "notes": self.notes[-12:],
            "verdicts": {f"{k[0]}:{k[1]}": (v.get("verdict"), v.get("failed"), v.get("detail")) for k, v in list(self.verdict.items())[:30]},
            "lex": {f"{k[0]}|{k[1]}|{k[2]}": {"f": v["fields"][:6], "a": v["accept"][:8]} for k, v in self.lex.items()},
            "stars": self.last_stars,
            "spend": round(self.brain.spend, 4), "llm": self.brain.calls, "t": round(self.clock.used(), 1)}, 6000)}


# ================================================================================================ fallback
class Fallback:
    """When the tool roles cannot be mapped: a compact-history tool-calling loop."""

    def __init__(self, agent: Cartographer):
        self.a = agent

    def run(self) -> list[dict[str, Any]]:
        a, w = self.a, self.a.world
        sys = ("You shop for a customer in a simulated store using only the tools. Each tool call is one step. Read the "
               "request carefully (every requirement, budget, quantity, cheapest rule, message limit, procedures). "
               "Ask the shopper at most the stated number of messages, one property per message. Verify attributes on "
               "the product's own record before ordering. Order exactly the requested quantities. Decline only when "
               "nothing can fit, without naming products.")
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": w.query + (
            "\n\n" + jdump(w.opening) if w.opening else "")}]
        idle = 0
        while not w.finished and w.left() > 0 and a.clock.left() > 15:
            res = a.brain.chat(STRONG, msgs[:2] + msgs[2:][-24:], 600, tools=w.tools, cap_s=35)
            calls = (res or {}).get("tool_calls") or []
            if not calls:
                idle += 1
                if idle > 2:
                    break
                msgs.append({"role": "user", "content": "Call one of the tools."})
                continue
            idle = 0
            c = calls[0]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except Exception:
                args = {}
            obs = w.call(c["function"]["name"], args if isinstance(args, dict) else {})
            msgs.append({"role": "assistant", "content": res.get("content") or "", "tool_calls": [c]})
            msgs.append({"role": "tool", "tool_call_id": c.get("id"), "content": jdump(obs, 5000) +
                         f"\n[steps left {w.left()}]"})
        return w.dialogue


# ================================================================================================ Cartographer 2
# Cartographer with a hard time guard and an ordering procedure:
# - every model and tool call ends before a reserve kept for the finish, and slow replies shrink the model ladder;
# - when the clock runs short, a tool call raises Hurry and the episode is finished from what is already known;
# - the basket is told to the shopper before each order (again after any change), with messages reserved for it;
# - accessories for a product the shopper owns must carry that product's part-series code;
# - the shopper's answer to the basket message is read like a question reply, never as store news;
# - a dropped wish counts only when the news names it; a blocked decline is retried; steps never run out mid-repair.

FINISH_S = 42.0      # seconds kept for the finish (cart lines, basket message, checkout)
HURRY_S = 36.0       # a non-final tool call with less than this left hands over to the finish
SLOW_S = 18.0        # a model reply slower than this marks the proxy as loaded
QUESTION_WIDTH = 3   # properties named in one question
# variant menus (orderable colours and sizes) and stock flags: the listing's own choices, not a shopper's wish
MENU_KEY = re.compile(r"\b(?:options?|variants?|variations?|choices?|selections?|availab\w*)\b")


class Hurry(Exception):
    """The clock says the episode must be finished now."""


class TimedBrain(Brain):
    """Model access whose every call ends before the finish reserve; a loaded proxy gets one fast model per call."""

    def __init__(self, clock: Clock, pacer: Pacer):
        super().__init__(clock, pacer)
        self.finishing = False
        self.slow = 0

    def chat(self, models: list[str], messages: list[dict[str, Any]], max_tokens: int = 900,
             want_json: bool = True, cap_s: float = 40.0, tools: list | None = None) -> Any:
        if self.spend > SPEND_STOP:
            return None
        ladder = list(models)
        if self.spend > SPEND_HARD or self.slow >= 2:
            ladder = [m for m in FAST if m not in self.dead][:1]
        elif self.spend > SPEND_SOFT or self.slow:
            ladder = [m for m in ladder if m in FAST or m in JUDGE] or ladder
        keep = 6.0 if self.finishing else FINISH_S
        for model in ladder[:2]:
            if model in self.dead:
                continue
            room = self.clock.left() - keep
            if room < 8:
                return None
            wait = int(max(5, min(cap_s if not self.slow else min(cap_s, 25), room)))
            body: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            self.pacer.wait()
            t = monotonic()
            res = ProxyClient(timeout=wait, max_retries=1, retry_delay=0.5).post_verbose("/inference/chat/completions", body)
            took = monotonic() - t
            if took > SLOW_S:
                self.slow += 1
            data = res.data
            with self.lock:
                self.calls += 1
            if isinstance(data, dict) and data.get("choices"):
                cost = num((data.get("usage") or {}).get("cost"))
                with self.lock:
                    self.spend += cost or 0.0
                msg = data["choices"][0].get("message") or {}
                self.log.append({"model": model, "s": round(took, 1), "cost": cost})
                if tools:
                    return msg
                text = msg.get("content") or ""
                if not want_json:
                    return text
                parsed = mend_json(text)
                if parsed is None:
                    parsed = first_json(text)
                if parsed is not None:
                    return parsed
                continue
            status = (res.error or {}).get("status")
            self.log.append({"model": model, "err": status, "s": round(took, 1)})
            if status in (400, 401, 402, 403, 404):
                self.dead.add(model)
            elif status in (429, 503):
                self.pacer.slow()
        return None


def mend_json(text: str) -> Any:
    """Decode a reply whose outer object was left open (a list not closed before the outer keys, or a cut-off tail) by
    adding the closers still owed; None when the reply is balanced or cannot be mended that way."""
    at = text.find("{")
    if at < 0:
        return None
    body = text[at:]
    plain = re.sub(r'"(?:\\.|[^"\\])*"', lambda m: "_" * len(m.group(0)), body)   # strings blanked, offsets kept
    if plain.count('"'):
        return None
    owed, cut = [], 0
    pairs = {"{": "}", "[": "]"}
    for pos, c in enumerate(plain):
        if c in pairs:
            owed.append(pairs[c])
        elif c in "}]":
            if not owed or owed.pop() != c or not owed:
                return None
            cut = pos + 1
    if not owed or not cut:
        return None
    try:
        return json.loads(body[:cut].rstrip().rstrip(",") + "".join(reversed(owed)))
    except ValueError:
        return None


class TimedWorld(World):
    """Tool calls bounded by the clock; outside the finish, a call that would eat the finish reserve raises Hurry."""

    def __init__(self, env: dict[str, Any], clock: Clock, pacer: Pacer):
        super().__init__(env, clock, pacer)
        self.finishing = False

    def call(self, name: str, args: dict[str, Any]) -> Any:
        if self.finished or self.step >= self.max_steps:
            return None
        if not self.finishing and self.clock.left() < HURRY_S:
            raise Hurry()
        self.step += 1
        body = {**self.binding, "step": self.step, "action": {"name": name, "args": args}}
        result = None
        for attempt in range(4):
            left = self.clock.left()
            if attempt and left < 4:
                break
            self.pacer.wait()
            self.client.timeout = int(max(3, min(25, left - 1)))
            res = self.client.post_verbose("/environment/call", body)
            if isinstance(res.data, dict) and "observation" in res.data:
                result = res.data
                break
            status = (res.error or {}).get("status")
            if status == 409:
                break
            if status in (429, 503):
                self.pacer.slow()
            pause = backoff(attempt, status, self.step * 31 + len(name))
            if self.clock.left() < pause + 6:
                break
            sleep(pause)
        if result is None:
            self.trace.append({"step": self.step, "tool": name, "args": args, "obs": None})
            return None
        obs = result.get("observation")
        self.finished = bool(result.get("finished"))
        self.trace.append({"step": self.step, "tool": name, "args": args, "obs": obs})
        self.dialogue.append({"role": "assistant", "tool_call": {"name": name, "arguments": args},
                              "environment_result": result})
        for ln in shopper_lines(obs):
            if ln not in self.heard[-40:]:
                self.heard.append(ln)
                self.fresh.append(ln)
        return obs


SERIES_RX = re.compile(r"(?i)\bseries(?:\s+(?:no\.?|number|code))?\s*[:#]?\s*([a-z]{1,4}[0-9]{1,4})(?![a-z0-9])")

LINK_PROMPT = """A shopper's request may say that the products to buy must work with, fit, or match a product the shopper
already owns or is buying (a device, a machine, a vehicle, a system). Answer ONE JSON object:
{"anchor": "that product as the request names it (model words, maker), or null when there is none",
 "dependent_items": [indexes (0-based, in the order listed) of the items that must fit the anchor]}
ITEMS: {items}
REQUEST: {q}"""


def qwords(s: Any) -> list[str]:
    """Search words that keep numbers ("3440 x 1440" -> 3440, 1440)."""
    return [w for w in re.findall(r"[a-z0-9]+", low(s)) if len(w) > 1 or w.isdigit()]


def own_reply(obs: Any) -> set:
    """The shopper's direct answer to our message: the reply itself, or its plain-text top-level values (news lines
    arrive as lists beside it)."""
    if isinstance(obs, str):
        return {obs.strip()}
    if isinstance(obs, dict):
        return {v.strip() for v in obs.values() if isinstance(v, str)}
    return set()


def series_codes(text: Any) -> list[str]:
    """Part-series codes a listing states ("Compatible part series: ZE1", "series no. AB12")."""
    out = []
    for c in SERIES_RX.findall(str(text or "")):
        if re.search(r"\d", c) and c.upper() not in out:
            out.append(c.upper())
    return out


def carries_code(text: Any, code: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![A-Za-z0-9])", str(text or ""), re.I))


class Cartographer2(Cartographer):
    def __init__(self, problem: dict[str, Any]):
        super().__init__(problem)
        env = problem["environment"]
        self.world = TimedWorld(env, self.clock, self.pacer)
        self.brain = TimedBrain(self.clock, self.pacer)
        self.told: tuple | None = None          # cart lines as last told to the shopper
        self.one_at_a_time = False              # set when the shopper asks for one property per question
        self.batch_limit: int | None = None     # ids per sheet call the store accepts, once learned
        self.unread_ok: set[str] = set()        # ids a sheet call returned nothing for
        self.banded: set[tuple[int, float]] = set()
        self.exploring = False
        self.link: dict[str, Any] = {}          # anchor product and the series code its accessories must carry

    # ------------------------------------------------------------------ set-up
    def core_ready(self) -> bool:
        r = self.roles
        return bool(r.get("checkout") and r.get("cart_add") and (r.get("browse") or r.get("search"))
                    and (r.get("details") or r.get("batch")))

    SPEC_TOP = ("budget", "budget_scope", "objective", "best_measure", "in_stock_only", "message_budget", "step_budget",
                "approval_required", "approval_text", "procedures", "decline_no_names", "story_hints", "multi_item_order")

    def setup(self):
        super().setup()
        items = self.spec.get("items") or []
        lost = not items or (len(items) == 1 and items[0].get("name") == "the product")
        stray = items and any(k in items[0] and k not in self.spec for k in self.SPEC_TOP)
        if lost:
            # the task spec never parsed: ask once more, on the fast models, with room to finish
            again = self.brain.chat(FAST, [{"role": "user", "content": SPEC_PROMPT + self.world.query}], 2600)
            if isinstance(again, dict) and again.get("items"):
                self.spec = again
                self.notes.append("spec retried")
                stray = True
        if stray:
            self.adopt_spec()
        self.name_kinds()
        if self.core_ready():
            return
        # the role map came back short (cut off or garbled): map the tools once more on a fast model
        tools_txt = jdump([t.get("function", t) for t in self.world.tools], 12000)
        res = self.brain.chat(FAST, [{"role": "user", "content": ROLE_PROMPT + tools_txt}], 2400)
        if isinstance(res, dict):
            again = self.check_roles(res)
            for k, v in again.items():
                self.roles.setdefault(k, v)
            self.patch_roles()
        self.notes.append("roles retried: " + ",".join(sorted(self.roles)))

    def adopt_spec(self):
        """Lift outer keys that a misplaced bracket left inside the first item, tidy the items and re-read the limits."""
        items = [it for it in (self.spec.get("items") or []) if isinstance(it, dict)]
        if items:
            for k in self.SPEC_TOP:
                if k in items[0] and k not in self.spec:
                    self.spec[k] = items[0].pop(k)
        for it in items:
            it["quantity"] = int(num(it.get("quantity")) or 1)
            own = {stem(w) for w in words(it.get("category")) + words(it.get("name"))}
            reqs = []
            for r in it.get("requirements") or []:
                if not isinstance(r, dict) or not r.get("property"):
                    continue
                pn = norm(r.get("property"))
                if re.search(r"\b(price|budget|cost|stock|availability|category|department|range)\b", pn):
                    continue
                if pn in ("type", "kind", "product type") and {stem(w) for w in words(r.get("value"))} <= own:
                    continue
                reqs.append(r)
            it["requirements"] = reqs
        if items:
            self.spec["items"] = items
        m = num(self.spec.get("message_budget"))
        self.msg_budget = int(m) if m is not None else 3
        self.budget = num(self.spec.get("budget"))
        self.cheapest = self.spec.get("objective") == "cheapest"
        self.best = self.spec.get("objective") == "best"
        self.approval = bool(self.spec.get("approval_required"))
        st = num(self.spec.get("step_budget"))
        if st and 3 <= st < self.world.max_steps:
            self.world.max_steps = int(st)

    def name_kinds(self):
        """A product type narrower than its category lives only in the item name; make it a checked requirement so a
        listing of another type on the same shelf cannot pass."""
        for it in self.spec.get("items") or []:
            base = {stem(w) for w in words(it.get("category"))} | STOPW
            extra = [w for w in words(it.get("name")) if stem(w) not in base and not w.isdigit()
                     and not re.fullmatch(r"items?|products?|things?|ones?|kinds?|pairs?|sets?|some|any|new", w)]
            if not 1 <= len(extra) <= 2 or not {stem(w) for w in extra} <= {stem(w) for w in words(it.get("search_words"))}:
                continue
            if any({stem(w) for w in extra} <= {stem(w) for w in words(r.get("value"))} for r in it["requirements"]):
                continue
            it["requirements"].append({"property": "product type", "op": "contains", "value": " ".join(extra),
                                       "text": it.get("name")})
            self.notes.append("kind from name: " + " ".join(extra))

    # ------------------------------------------------------------------ fact sheets
    def sheet_cap(self) -> int:
        """How many ids one batch call may carry: learned from a refused call, else the schema bound, else 3."""
        if self.batch_limit:
            return self.batch_limit
        e = self.roles.get("batch") or {}
        bound = num(schema_props(self.tools[e["tool"]]).get(e.get("ids_arg"), {}).get("maxItems")) if e.get("tool") else None
        return max(1, min(int(bound), 6)) if bound else 3

    def read_ids(self, ids: list[str]):
        if not ids or not self.roles.get("batch"):
            return super().read_ids(ids)
        e = self.roles["batch"]
        todo = [i for i in ids if i not in self.unread_ok]
        while todo and self.world.left() > self.reserve():
            part = todo[:self.sheet_cap()]
            obs = self.act("batch", {e["ids_arg"]: part})
            came = {parse_row(d).get("id") for d in find_rows(obs)} & set(part)
            if not came and len(part) > 1 and obs is not None:
                self.batch_limit = max(1, len(part) // 2)   # refused as too long: halve and send again
                self.notes.append(f"sheet cap {self.batch_limit}")
                continue
            self.unread_ok |= set(part) - came
            todo = todo[len(part):]

    def verify(self, idx: int, want: int = 3, max_reads: int | None = None):
        """Read records in priority order until `want` candidates pass; only sheets that came back count as read."""
        reads = 0
        while self.clock.left() > 70:
            self.judge(idx)
            passes = [r for c, r in self.ranked(idx) if c == 0]
            cand = [r for r in self.reading_order(idx, self.pool(idx)) if not r.get("detailed")
                    and r["id"] not in self.unread_ok and (idx, r["id"]) not in self.verdict]
            if len(passes) >= want:
                if not self.cheapest:
                    break
                low_p = min((r.get("price") for r in passes if r.get("price") is not None), default=None)
                cheaper = [r for r in cand if low_p is not None and r.get("price") is not None and r["price"] < low_p]
                if not cheaper and (low_p is None or not self.band_below(idx, low_p)):
                    break
                continue_cand = cheaper or [r for r in self.reading_order(idx, self.pool(idx)) if not r.get("detailed")
                                            and r["id"] not in self.unread_ok and (idx, r["id"]) not in self.verdict
                                            and r.get("price") is not None and r["price"] < low_p]
                if not continue_cand:
                    break
                cand = continue_cand
            if self.cheapest and passes:
                top = passes[want - 1]["price"] if len(passes) >= want else None
                cand = [r for r in cand if top is None or (r.get("price") or 0) <= top]
            budget = self.explore_left() if max_reads is None else min(self.explore_left(), max_reads - reads)
            if budget <= 0:
                break
            if not cand:
                if self.cheapest and passes and self.page_on(idx, passes[0].get("price")):
                    reads += 1   # a page is a step, like a read
                    continue
                if not self.expand(idx):
                    break
                continue
            per_call = self.sheet_cap() if self.roles.get("batch") else min(3, budget)
            batch = cand[:max(1, min(per_call, budget if not self.roles.get("batch") else per_call))]
            self.read_ids([r["id"] for r in batch])
            reads += 1 if self.roles.get("batch") else len(batch)
            for r in batch:   # a sheet that came back without facts is judged on what it shows
                if r["id"] not in self.unread_ok:
                    self.rows[r["id"]]["detailed"] = True
        self.judge(idx)

    # ------------------------------------------------------------------ searching
    def listing(self, role: str, value: str, idx: int, cap: float | None = None, pages: int = 1) -> dict[str, Any]:
        """As before, but a repeat request for more pages carries on from the stored cursor instead of stopping."""
        key = f"{role}|{low(value)}|{cap}"
        info = self.searched_info.get(key)
        if key not in self.searched or not info or info.get("complete") or info.get("cursor") is None \
                or pages <= info.get("pages", 0):
            return super().listing(role, value, idx, cap, pages)
        e = self.roles.get(role)
        paging = [p for p in (self.roles.get("paging") or []) if e and p.get("tool") == e["tool"]]
        while paging and info["pages"] < pages and info.get("cursor") is not None:
            if self.explore_left() <= 0 or self.clock.left() < 60:
                break
            args = dict(info["base"])
            args[paging[0]["arg"]] = info["cursor"]
            ids, got = self.query(role, args, idx)
            info["ids"] += ids
            info["pages"] += 1
            info["cursor"] = got["next"] if ids else None
            if not ids or info["cursor"] is None or (info.get("total") and len(info["ids"]) >= info["total"]):
                info["complete"] = True
                info["cursor"] = None
        return info

    def label_terms(self, idx: int) -> list[str]:
        """The learned store category as search words (the head noun until one is known)."""
        label = self.labels.get(idx) or self.label_for(idx)
        ws = [w for w in qwords(label) if len(w) > 1 and w not in STOPW] if label else []
        return ws[:4] or self.head_noun(idx)

    def explore(self, idx: int):
        self.exploring = True
        try:
            super().explore(idx)
        finally:
            self.exploring = False
        code = self.link.get("code")
        if code and idx in self.link.get("deps", []) and self.roles.get("search"):
            self.listing("search", " ".join([code] + self.head_noun(idx)), idx)
        it = self.spec["items"][idx]
        label = self.label_for(idx)
        if label:
            # a requirement the store's own category already states is met by every row on that shelf
            lw = set(qwords(label))
            keep = [r for r in it["requirements"] if r.get("hidden") or r.get("op") not in ("is", "contains")
                    or not qwords(r.get("value")) or not set(qwords(r.get("value"))) <= lw]
            if len(keep) < len(it["requirements"]):
                self.notes.append("met by category: " + ",".join(str(r.get("property")) for r in it["requirements"] if r not in keep))
                it["requirements"] = keep
                self.reset_verdicts(idx)
        if not self.roles.get("search"):
            return
        kind = self.label_terms(idx)
        done = 0
        for r in self.live_reqs(idx):
            if done >= 2 or r.get("op") not in ("is", "contains", "one_of"):
                continue
            vw = [w for w in qwords(r.get("value")) if w not in set(kind)][:4]
            if not vw:
                continue
            done += 1
            res = self.listing("search", " ".join((vw + kind)[:6]), idx, pages=4)
            if not res["ids"]:
                self.listing("search", " ".join(vw + self.head_noun(idx)), idx, pages=4)

    def walk(self, idx: int, label: str, cap: float, first: dict[str, Any]):
        """Cheapest rule: find a price cap whose capped category fits the visible pages, interpolating on the totals
        seen, then read that window to its end (continuing, never re-reading its first page)."""
        if self.exploring and any(r.get("op") in ("is", "contains", "one_of") for r in self.live_reqs(idx)):
            return   # a stated value narrows better than a price window: value searches first, price bands later
        per = max(1, first["per_page"])
        vis = self.visible_max("browse", per)
        total = first["total"] if first["total"] is not None else 10 ** 6
        hi_cap, hi_n = cap, total
        lo_cap, lo_n = 0.0, 0
        best = (cap, total) if total <= vis else None
        res = first
        for _ in range(6):
            if best and best[1] >= vis * 0.5:
                break
            if self.explore_left() <= 4:
                break
            # place the next cap where the listed count should land near the window, assuming counts grow with price
            target = vis * 0.8
            if hi_n > lo_n:
                new = lo_cap + (hi_cap - lo_cap) * max(0.05, min(0.95, (target - lo_n) / (hi_n - lo_n)))
            else:
                new = (lo_cap + hi_cap) / 2
            new = float(int(new)) if new > 20 else round(new, 2)
            if new <= lo_cap or new >= hi_cap:
                break
            res = self.listing("browse", label, idx, cap=new, pages=1)
            t = res["total"] if res["total"] is not None else len(res["ids"])
            if t > vis:
                hi_cap, hi_n = new, t
            else:
                lo_cap, lo_n = new, t
                if t > 0:
                    best = (new, t)
        if not best:
            return
        self.window = (idx, best[0])
        self.listing("browse", label, idx, cap=best[0], pages=6)

    def band_below(self, idx: int, price: float) -> bool:
        """Cheapest rule: the category listed under the best fit's price, once per price. True if new rows came."""
        label = self.labels.get(idx) or self.label_for(idx)
        capped = self.roles.get("browse") and any(pc.get("tool") == self.roles["browse"]["tool"]
                                                  for pc in self.roles.get("price_cap") or [])
        if not label or not capped or (idx, price) in self.banded or self.explore_left() <= 5 or self.short(90):
            return False
        self.banded.add((idx, price))
        before = {r["id"] for r in self.pool(idx)}
        first = self.listing("browse", label, idx, cap=round(price - 0.01, 2), pages=1)
        if first["ids"]:
            self.walk(idx, label, round(price - 0.01, 2), first)
        return bool({r["id"] for r in self.pool(idx)} - before)

    # ------------------------------------------------------------------ questions
    PLAN_PROMPT = """You choose which product property to ask a shopper about. They hold a firm preference on one or two
properties and state it only when the question names that property. Their story is where the hint sits: someone who
studies tags or fine print cares what the thing is made of; talk of pictures, of matching an outfit or a room, means
whichever of its pattern, colour or look the request leaves open; a body part, a space or a device it must fit or sit
on means that shape or dimension; a gift, a purchase for someone else, or how it suits the person wearing it means the
intended wearer or user; comfort against the skin means the fabric; the
occasion or activity means its style or type; frequent opening and closing means how it fastens. Ignore weather, dates
and money in the story. Rank the listed properties by how well the story points at each; when nothing points anywhere,
put first who it is meant for (for things people wear), then the properties whose values vary most across listings. Leave out properties the
request already states, and sizes unless the story is about fit.
Answer with ONE JSON object and nothing before or after it (no reasoning text):
{"order": [property names copied from the list]}
REQUEST: """

    def question_menu(self, idx: int) -> list[str]:
        """Properties worth a question, grouped by the name a shopper would use (store spellings of one property
        merged), scored by how many read listings carry it and how much its values vary; variant pickers, free text,
        codes and near-unique values left out."""
        it = self.spec["items"][idx]
        stated = {plain_name(r.get("property")) for r in it["requirements"]}
        rows = [r for r in self.pool(idx) if r.get("detailed") and r.get("facts")]
        stats: dict[str, dict[str, Any]] = {}
        for r in rows:
            facts = r.get("facts") or {}
            seen_here = set()
            for k, v in facts.items():
                nk = plain_name(k)
                nv = norm(v)
                if not nk or not nv or any(nk == s_ or nk in s_ or s_ in nk for s_ in stated if s_) or MENU_KEY.search(nk):
                    continue
                st = stats.setdefault(nk, {"rows": 0, "count": collections.Counter(), "n": 0})
                if nk not in seen_here:
                    st["rows"] += 1
                    seen_here.add(nk)
                st["n"] += 1
                st["count"][nv] += 1
        asked = {p for i, p in self.asked if i == idx}
        scored = []
        for nk, st in stats.items():
            vals = st["count"]
            avg = sum(len(v) for v in vals) / max(1, len(vals))
            coded = sum(n for v, n in vals.items() if re.search(r"[/+]|\d", v)) / max(1, st["n"])   # lists of words are fine
            if avg > 40 or (st["n"] >= 4 and max(vals.values()) == 1) or (st["n"] >= 2 and coded >= 0.5) \
                    or st["rows"] < max(1, len(rows) // 4):
                continue
            if any(nk == a or nk in a or a in nk for a in asked):
                continue
            spread = min(0.6, 0.12 * (len(vals) - 1))
            scored.append((st["rows"] / max(1, len(rows)) * 2 + spread, nk))
        scored.sort(reverse=True)
        out: list[str] = []
        for _, nk in scored:
            if any(nk in o or o in nk for o in out):
                continue
            out.append(nk)
        hints = [norm(h) for h in (self.spec.get("story_hints") or []) if 0 < len(words(h)) <= 3]
        for h in hints:
            if h and h not in out and h not in asked and h not in stated and len(out) < 2:
                out.append(h)
        return out

    def audit(self, picks: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
        """The basket review, told which wishes were dropped; a rejected pick without a named swap is struck and
        replaced by the next verified candidate."""
        if self.audited or self.clock.left() < 45 or self.short(100):
            return picks
        self.audited = True
        cands = {idx: [self.compact(r) for c, r in self.ranked(idx)[:5] if c <= 1] for idx in range(len(self.spec["items"]))}
        payload = {"request": self.world.query[:3000], "procedures": self.spec.get("procedures"),
                   "items": [{"index": i, "name": it.get("name"), "quantity": it["quantity"], "requirements": self.live_reqs(i)}
                             for i, it in enumerate(self.spec["items"])],
                   "no_longer_required": sorted(self.dropped),
                   "budget": self.budget, "objective": self.spec.get("objective"),
                   "basket": {i: self.compact(p) for i, p in picks.items()}, "candidates": cands,
                   "shopper_and_store_lines": self.world.heard[-10:]}
        res = self.brain.chat(STRONG, [{"role": "user", "content": self.AUDIT_PROMPT +
                                        "Properties under no_longer_required do not count any more.\n" + jdump(payload, 14000)}], 400)
        if not isinstance(res, dict) or res.get("ok", True):
            return picks
        self.notes.append("audit: " + str(res.get("issue"))[:160])
        swap = res.get("swap") if isinstance(res.get("swap"), dict) else {}
        out = dict(picks)
        swapped = set()
        for k, rid in swap.items():
            try:
                i = int(k)
            except (TypeError, ValueError):
                continue
            if i in out and rid in self.rows and any(r["id"] == rid for _, r in self.ranked(i)):
                out[i] = self.rows[rid]
                swapped.add(i)
        if not swapped and len(picks) == 1:
            i, p = next(iter(picks.items()))
            v = self.verdict.get((i, p["id"]))
            if v is not None and v.get("verdict") != "pass":   # an unproven pick the review rejects is struck
                v["verdict"], v["failed"] = "fail", list(v.get("failed") or []) + ["review"]
                again = self.choose()
                if again:
                    return again
        return out

    def check_req(self, r: dict[str, Any], req: dict[str, Any], lx: dict[str, Any], star: str | None = None) -> str:
        """As before, except that (a) one field about the property stating a wanted value is enough, even when another
        field (the usual one included) disagrees, and (b) a wish the shopper revealed must be stated by a fully read
        listing: silence fails it."""
        if (req.get("op") or "is") in ("is", "contains", "one_of"):
            groups = [vtoks(a) for a in lx.get("accept", []) if vtoks(a)]
            named = {norm(f) for f in lx.get("fields", [])} | ({star} if star else set())
            # a field named for the property itself speaks for it: when the listing has one, a look-alike field that
            # agrees ("wifi standard: wifi 4") does not outvote it ("network: 4g")
            prop = plain_name(req.get("property"))
            own = [v for k, v in (r.get("facts") or {}).items() if v not in (None, "") and prop
                   and field_matches(prop, plain_name(k)) and not MENU_KEY.search(plain_name(k))]
            if own and groups:
                return "pass" if any(g <= vtoks(v) or vtoks(v) <= g for v in own for g in groups if vtoks(v)) else "fail"
            for k, v in (r.get("facts") or {}).items():
                if v in (None, "") or not groups:
                    continue
                kn = norm(str(k).replace(".", " "))
                about = kn in named or field_matches(str(req.get("property")), str(k)) or \
                    any(field_matches(f, str(k)) for f in named if f)
                if about and any(g <= vtoks(v) for g in groups):
                    return "pass"
        res = super().check_req(r, req, lx, star)
        if res == "unknown" and req.get("hidden") and r.get("detailed") and r.get("facts"):
            return "fail"
        return res

    # ------------------------------------------------------------------ time
    def short(self, need: float = 90.0) -> bool:
        return self.clock.left() < need

    def lexicon(self, idx: int, req: dict[str, Any]) -> dict[str, Any]:
        key = (idx, norm(req.get("property")), norm(req.get("value")), req.get("op"))
        if self.short(FINISH_S + 20) or self.world.finishing:
            if key in self.lex:
                return self.lex[key]
            lx = {"fields": [], "accept": [str(req.get("value"))] if req.get("op") != "at_least" and req.get("op") != "at_most" else [],
                  "reject": [], "sig": 0, "calls": 0}
            return lx
        return super().lexicon(idx, req)

    def plan_questions(self, idx: int) -> list[str]:
        """Order the question menu with the strong models (the choice of question decides whether a hidden wish surfaces)."""
        menu = self.question_menu(idx)
        if len(menu) <= 1 or (self.short(120) and idx not in self.qplan):
            return menu
        if idx in self.qplan:
            return [p for p in self.qplan[idx] if p in menu] + [p for p in menu if p not in self.qplan[idx]]
        rows = [r for r in self.pool(idx) if r.get("detailed") and r.get("facts")]
        cov = {p: round(sum(1 for r in rows if any(norm(plain_name(k)) == norm(p) for k in r["facts"])) / max(1, len(rows)), 2)
               for p in menu[:10]}
        it = self.spec["items"][idx]
        prompt = (self.PLAN_PROMPT + self.world.query[:2500] + "\nITEM: " + str(it.get("name")) +
                  "\nSTATED: " + jdump([r.get("property") for r in it["requirements"]], 300) + "\nPROPERTIES (coverage): " + jdump(cov, 800))
        res = self.brain.chat(STRONG, [{"role": "user", "content": prompt}], 900)
        said = [str(x) for x in (res.get("order") or [])] if isinstance(res, dict) else []
        by_name = {norm(p): p for p in menu}
        order = [by_name[norm(p)] for p in said if norm(p) in by_name]
        self.qplan[idx] = order
        self.notes.append("menu: " + ",".join(menu[:8]) + " | plan: " + ",".join(said[:5]))
        return order + [p for p in menu if p not in order]

    def handle_notices(self, after_finish: bool = False) -> list[dict[str, Any]]:
        if not (self.short(FINISH_S + 12) or self.world.finishing):
            heard = " ".join(self.world.fresh)
            before = set(self.dropped)
            out = super().handle_notices(after_finish)
            said = {stem(w) for w in words(heard)}
            kept = []
            for c in out:
                if c.get("kind") == "drop_requirement" and not ({stem(w) for w in words(c.get("property"))} & said):
                    continue   # the lines never name that property: a misreading, the stated wish stays
                kept.append(c)
            named = {norm(c.get("property")) for c in kept if c.get("kind") == "drop_requirement"}
            for d in set(self.dropped) - before:
                if not any(d in n or n in d for n in named if n):
                    self.dropped.discard(d)
            return kept
        # no time for a model read: stock-outs from their wording, anything else after a finish is a refusal
        lines = self.world.take_fresh()
        out: list[dict[str, Any]] = []
        if any(self.GONE_RX.search(l) for l in lines):
            for i in list(self.cart)[:1]:
                self.gone.add(i)
                if i in self.rows:
                    self.rows[i]["stock"] = False
            out.append({"kind": "stock_out"})
        if after_finish and not out and lines:
            out.append({"kind": "refusal"})
        return out

    # ------------------------------------------------------------------ messages
    def basket_reserve(self) -> int:
        if not self.roles.get("message"):
            return 0
        return 2 if self.msg_budget >= 5 else 1

    def reserve(self) -> int:
        return super().reserve() + (1 if self.roles.get("message") else 0)

    def explore_left(self) -> int:
        q_left = max(0, self.msg_budget - self.messages_used - self.basket_reserve())
        return self.world.left() - self.reserve() - q_left

    def ask_round(self):
        budget = self.msg_budget - self.basket_reserve()
        n = len(self.spec["items"])
        order = list(range(n))
        rounds = 0
        while self.messages_used < budget and rounds < budget + 2 and not self.short(100):
            rounds += 1
            idx = order[self.messages_used % n] if n > 1 else 0
            menu = [p for p in self.plan_questions(idx)
                    if not any(norm(p) == a or norm(p) in a or a in norm(p) for i2, a in self.asked if i2 == idx)]
            if not menu:
                if n > 1 and any(self.question_menu(j) for j in order if j != idx):
                    order = [j for j in order if j != idx] + [idx]
                    continue
                break
            if self.one_at_a_time:
                res = self.ask(idx, menu[0])
                if res:
                    self.take_reply(idx, menu[0], res)
                continue
            self.ask_several(idx, menu[:QUESTION_WIDTH])

    def ask_several(self, idx: int, props: list[str]):
        """One question naming a few properties. The shopper speaks to the first named property they hold a wish
        about, so properties named before the one in the answer are settled as open, the later ones stay on the menu;
        an answer naming none of them settles them all."""
        e = self.roles.get("message")
        if not e or self.messages_used >= self.msg_budget or not props:
            return
        name = re.sub(r"^(?:(?:a|an|the|some|something|one|any)\s+(?:from\s+(?:the\s+|your\s+)?)?)+", "",
                      str(self.spec["items"][idx].get("name") or "item").strip(), flags=re.I) or "item"
        named = props[0] if len(props) == 1 else ", ".join(props[:-1]) + " or " + props[-1]
        text = f"What {named} should the {name} have?"
        self.world.take_fresh()
        obs = self.act("message", {e["text_arg"]: text})
        self.messages_used += 1
        reply = " ".join(shopper_lines(obs)) or jdump(obs, 600)
        self.world.take_fresh()
        res = self.brain.chat(FAST, [{"role": "user", "content": REPLY_PROMPT.replace("{q}", text).replace("{r}", reply)}], 500)
        res = res if isinstance(res, dict) else {}
        if res.get("kind") == "too_many":
            self.one_at_a_time = True   # this shopper takes one property per question: the rest go singly
            return
        spoken = {stem(w) for w in words(reply)}
        for rq in res.get("requirements") or []:
            if isinstance(rq, dict):
                spoken |= {stem(w) for w in words(rq.get("property"))}
        upto = len(props)
        if res.get("requirements") or res.get("reference"):
            for k, p in enumerate(props):
                if {stem(w) for w in words(p)} & spoken:
                    upto = k + 1
                    break
        for p in props[:upto]:
            self.asked.append((idx, norm(p)))
        if res:
            self.take_reply(idx, props[min(upto, len(props)) - 1], res)

    def lines_now(self) -> tuple:
        return tuple(sorted(self.cart.items()))

    def tell_basket(self, picks: dict[int, dict[str, Any]]) -> bool:
        """Tell the shopper exactly what is about to be ordered (the approval request when one is required)."""
        e = self.roles.get("message")
        if not e or self.messages_used >= self.msg_budget or self.world.left() < 2 or self.lines_now() == self.told:
            return False
        parts, total = [], 0.0
        for i, p in picks.items():
            q = self.spec["items"][i]["quantity"]
            pr = p.get("price") or 0
            total += pr * q
            parts.append(f"{q} x {(p.get('title') or 'item')[:80]} ({p['id']}) at {pr:g}")
        cap = f", within your budget of {self.budget:g}" if self.budget else ""
        tail = "Shall I place the order?" if self.approval else "I will place the order now."
        text = f"Your basket: {'; '.join(parts)}. Total {total:g}{cap}. {tail}"
        self.world.take_fresh()
        obs = self.act("message", {e["text_arg"]: text[:780]})
        self.messages_used += 1
        self.told = self.lines_now()
        own = own_reply(obs)
        self.world.fresh = [ln for ln in self.world.fresh if ln.strip() not in own]
        if own and not self.short(FINISH_S + 15):
            res = self.brain.chat(FAST, [{"role": "user", "content": REPLY_PROMPT.replace("{q}", text[:600])
                                          .replace("{r}", " ".join(own)[:600])}], 500)
            if isinstance(res, dict) and res.get("requirements"):
                self.take_reply(0, "", res)
                for idx in range(len(self.spec["items"])):
                    self.judge(idx)
        return True

    # ------------------------------------------------------------------ linked products
    def find_link(self):
        """Ask once whether the items must fit a product the shopper owns; if so read that product's series code."""
        if len(self.spec["items"]) < 1 or self.short(150):
            return
        if not re.search(r"compatib|fit|fits|work with|works with|goes with|for my|match", low(self.world.query)):
            return
        names = [it.get("name") for it in self.spec["items"]]
        res = self.brain.chat(FAST, [{"role": "user", "content": LINK_PROMPT.replace("{items}", jdump(names, 600))
                                      .replace("{q}", self.world.query[:2500])}], 300)
        if not isinstance(res, dict) or not res.get("anchor"):
            return
        deps = [int(i) for i in (res.get("dependent_items") or []) if isinstance(i, (int, float)) and 0 <= int(i) < len(names)]
        self.link = {"anchor": str(res["anchor"]), "deps": deps or list(range(len(names))), "code": None}
        if not self.roles.get("search"):
            return
        tmp = -1   # pseudo item slot for the anchor search
        self.listing("search", " ".join(words(res["anchor"])[:5]), tmp)
        aw = {stem(w) for w in words(res["anchor"]) if len(w) > 1}
        cands = sorted((r for rid, r in self.rows.items() if tmp in self.item_of.get(rid, set())),
                       key=lambda r: -len(aw & {stem(w) for w in words(r.get("title"))}))[:2]
        for r in cands:
            codes = series_codes(jdump(r.get("raw"), 6000) + " " + jdump(r.get("facts"), 3000))
            if not codes and not r.get("detailed") and self.world.left() > self.reserve() + 3:
                self.read_ids([r["id"]])
                r = self.rows[r["id"]]
                codes = series_codes(jdump(r.get("raw"), 6000) + " " + jdump(r.get("facts"), 3000))
            if codes:
                self.link["code"] = codes[0]
                self.link["anchor_id"] = r["id"]
                self.notes.append(f"link {res['anchor']} series {codes[0]}")
                break
        for rid in list(self.item_of):
            self.item_of[rid].discard(tmp)

    def judge(self, idx: int, force: bool = False):
        super().judge(idx, force)
        code = self.link.get("code")
        if not code or idx not in self.link.get("deps", []):
            return
        # a dependent product fits only when its own record names the anchor's series code
        for (i, rid), v in self.verdict.items():
            if i != idx or rid == self.link.get("anchor_id"):
                continue
            r = self.rows.get(rid) or {}
            text = (r.get("title") or "") + " " + jdump(r.get("raw"), 6000) + " " + jdump(r.get("facts"), 3000)
            if not carries_code(text, code):
                v["verdict"], v["failed"] = "fail", list(v.get("failed") or []) + ["series " + code]

    # ------------------------------------------------------------------ episode
    def run(self) -> list[dict[str, Any]]:
        try:
            self.setup()
            if not self.roles.get("checkout") or not self.roles.get("cart_add") or not (
                    self.roles.get("browse") or self.roles.get("search")) or not (
                    self.roles.get("details") or self.roles.get("batch")):
                self.notes.append("fallback: roles " + ",".join(self.roles))
                self.world.finishing = True
                self.brain.finishing = True
                return Fallback(self).run()
            self.find_link()
            n = len(self.spec["items"])
            for idx in range(n):
                self.explore(idx)
                self.handle_notices()
            share = max(3, self.explore_left() // (2 * n))
            for idx in range(n):
                self.verify(idx, want=2, max_reads=share)
            self.handle_notices()
            self.ask_round()
            self.handle_notices()
            for idx in range(n):
                hidden = [r for r in self.spec["items"][idx]["requirements"] if r.get("hidden")]
                if hidden and not any(c == 0 for c, _ in self.ranked(idx)) and self.roles.get("search"):
                    # every word must match: the revealed value with just the product's name, then with the search words
                    vw = words(hidden[-1]["value"])[:3]
                    tries = [" ".join(vw + words(self.spec["items"][idx].get("name"))[-2:]),
                             " ".join((vw + words(self.spec["items"][idx].get("search_words")))[:4])]
                    for qtext in dict.fromkeys(t for t in tries if t.strip()):
                        if self.listing("search", qtext, idx, pages=3).get("ids"):
                            break
                self.verify(idx, want=3)
            return self.commit()
        except Hurry:
            self.notes.append(f"hurry at step {self.world.step}, {self.clock.left():.0f}s left")
            return self.hurry_finish()

    def quick_picks(self) -> dict[int, dict[str, Any]] | None:
        """Best known candidate per item without new reads: a verified pass, else the best unrefuted one."""
        picks = {}
        for idx in range(len(self.spec["items"])):
            try:
                self.judge(idx)
            except Hurry:
                pass
            rk = self.ranked(idx)
            if rk:
                picks[idx] = rk[0][1]
                continue
            pool = [r for r in self.pool(idx) if (idx, r["id"]) not in self.verdict]
            if not pool:
                return None
            picks[idx] = sorted(pool, key=lambda r: r.get("price") if r.get("price") is not None else float("inf"))[0] \
                if self.cheapest else self.reading_order(idx, pool)[0]
        return picks

    def hurry_finish(self) -> list[dict[str, Any]]:
        self.world.finishing = True
        self.brain.finishing = True
        if self.world.finished or self.world.left() <= 0:
            return self.world.dialogue
        picks = self.quick_picks() if self.spec.get("items") else None
        if picks and self.roles.get("cart_add") and self.roles.get("checkout"):
            if not self.budget or self.summary(picks)[1] <= self.budget + 1e-6:
                self.set_cart(picks)
                if self.clock.left() > 14 and self.world.left() >= 2:
                    self.tell_basket(picks)
                if self.finish_order():
                    return self.world.dialogue
        if not self.world.finished and self.world.left() > 0:
            if self.cart and self.roles.get("checkout") and self.order_tries < 2:
                self.act("checkout")
            if not self.world.finished:
                self.decline()
        return self.world.dialogue

    def last_round(self) -> bool:
        """Too few steps for another repair round: put the best known picks in the cart and order now."""
        if self.world.left() > len(self.spec["items"]) + 1:
            return False
        self.world.finishing = True
        self.brain.finishing = True
        picks = self.quick_picks()
        if picks and (not self.budget or self.summary(picks)[1] <= self.budget + 1e-6):
            missing = sum(1 for i, p in picks.items() if self.cart.get(p["id"]) != self.spec["items"][i]["quantity"])
            if self.world.left() > missing:
                self.set_cart(picks)
        return True

    def decline_or_hear(self) -> bool:
        """Decline; True when the episode closed. A decline blocked by news takes the news in for another look."""
        self.decline()
        if self.world.finished or not self.roles.get("decline"):
            return True
        self.after_change(self.handle_notices(after_finish=True))
        return False

    def commit(self) -> list[dict[str, Any]]:
        for attempt in range(6):
            if self.world.finished or self.world.left() <= 0:
                break
            if self.last_round():
                break
            picks = self.choose()
            if picks is None:
                for idx in range(len(self.spec["items"])):
                    if not self.ranked(idx):
                        self.verify(idx, want=1)
                picks = self.choose()
                if picks is None:
                    for idx in range(len(self.spec["items"])):
                        read = sum(1 for k in self.verdict if k[0] == idx)
                        if read < 4 and self.world.left() > 3 and not self.short(70):
                            self.expand(idx)
                            self.verify(idx, want=1)
                    picks = self.choose()
                if picks is None:
                    self.notes.append("decline: no candidate")
                    if self.decline_or_hear():
                        break
                    continue
            if self.budget and self.summary(picks)[1] > self.budget + 1e-6:
                self.notes.append(f"over budget {self.summary(picks)[1]} > {self.budget}")
                if self.decline_or_hear():
                    break
                continue
            if attempt == 0:
                picks = self.audit(picks)
            self.set_cart(picks)
            ch = self.handle_notices()
            if self.roles.get("cart_view") and self.world.left() > 3 and (attempt == 0 or ch or len(picks) > 1):
                self.reconcile(self.act("cart_view"))
                ch += self.handle_notices()
                if set(self.cart) != {p["id"] for p in picks.values()} and self.world.left() > 3:
                    self.set_cart(picks)
            if ch:
                self.after_change(ch)
                if self.choose() != picks:
                    continue
            # procedure: the shopper hears the exact basket before every order
            if self.tell_basket(picks):
                ch = [c for c in self.handle_notices() if c.get("kind") not in ("approval", "none")]
                if ch:
                    self.after_change(ch)
                if self.choose() != picks:
                    continue
            if self.finish_order():
                break
            ch = self.handle_notices(after_finish=True)
            self.after_change(ch)
        if not self.world.finished and self.world.left() > 0:
            self.world.finishing = True
            if self.cart and self.roles.get("checkout") and self.order_tries < 6:
                self.act("checkout")
            for _ in range(2):   # news can block one decline; the next one closes
                if self.world.finished or self.world.left() <= 0:
                    break
                self.decline()
        return self.world.dialogue

    def diagnostics(self) -> dict[str, Any]:
        d = super().diagnostics()
        d["content"] += " " + jdump({"link": self.link, "told": bool(self.told), "slow": self.brain.slow,
                                     "msgs": self.messages_used, "steps": self.world.step,
                                     "asked": self.asked, "qplan": self.qplan}, 1200)
        return d


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    agent = Cartographer2(problem_data)
    try:
        out = agent.run()
        out.append(agent.diagnostics())
        return out
    except Exception as exc:  # never leave an episode open if we can still finish it
        w = agent.world
        w.finishing = True
        if not w.finished and w.left() > 0:
            try:
                if agent.cart and agent.roles.get("checkout"):
                    agent.act("checkout")
                elif agent.roles.get("decline"):
                    agent.decline()
            except Exception:
                pass
        w.dialogue.append({"role": "assistant", "content": f"error: {type(exc).__name__}"})
        return w.dialogue


__all__ = ["agent_main"]
