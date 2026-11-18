from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_MODELS = {
    "chutes": (
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "Qwen/Qwen3-32B-TEE",
        "zai-org/GLM-5.2-TEE",
    ),
    "openrouter": (
        "deepseek/deepseek-v3.2",
        "qwen/qwen3-32b",
        "z-ai/glm-5.2",
    ),
}
_PROXY = ProxyClient(timeout=70, max_retries=1, retry_delay=0.4)
_PROBLEM_BUDGET_S = 255.0
_FINISH_RESERVE_S = 65.0
_OBS_CHARS = 5000
_MAX_REPROMPTS = 1
_EPS = 1e-9
_REASON_RE = re.compile(
    r"\b(walk me through|tell me why|explain why|justify|written reason|"
    r"why (?:this|that|you (?:picked|chose|selected)))\b",
    re.I,
)
_MODELISH_RE = re.compile(r"\b(?=[a-z0-9\-]*\d)[a-z0-9][a-z0-9\-]{2,}\b", re.I)
_PHRASE_RE = re.compile(
    r"matches(?: the phrase)?\s+[\"“]?([a-z0-9][a-z0-9 \-]{1,40}?)[\"”]?(?:,| at | and |\.|$)",
    re.I,
)
_PREFER_RE = re.compile(
    r"(?:strongly prefer|prefer(?:s|red)?(?: the)?|exact(?:ly)?(?: the)?)\s+"
    r"([a-z0-9][\w\- ]{0,40}?)(?:\.|,|;| if | and | but |\n|$)",
    re.I,
)
_TRADEOFF_RE = re.compile(
    r"\b(i'?d rather|rather pay|cheapest isn'?t|don'?t just grab the cheapest|"
    r"which of your preferences|weigh the others)\b",
    re.I,
)
_HIDDEN_SPEC_RE = re.compile(
    r"tell you if you ask|firm requirement on the spec|hidden (?:firm )?requirement",
    re.I,
)

_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _NUM
    + r"\s*(PHP|php|₱|pesos?)?",
    re.I,
)
_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTED_RE = re.compile(r"\"([^\"]{2,80})\"")
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)

_CORE_RULES = (
    "You are a careful shopping agent working a catalog through tools. Exactly one tool call per turn, at "
    "most 30 turns, so be efficient. Search or filter, then view or compare so raw facts, variant options "
    "and prices are on record. Never add a placeholder to the cart — the first addition is graded. Prices "
    "and stock change; after a cart addition the item is re-checked and you may need a different product. "
    "Answer every direct question from the shopper with one short message before anything else. Never send "
    "progress updates. Never ask the shopper to raise the budget. Do not repeat a tool call with identical "
    "arguments; one compare of up to five products beats several view calls. "
    "Read evidence the way a careful shopper would: labelled facts first, then the selected variant's "
    "option text, then the title. If those disagree, the specification is not proven. Every must-have the "
    "shopper stated that a product card can prove is a hard gate — any labelled fact, quantity, duration, "
    "count, class, or named pairing, in the shopper's own wording. Preferences only rank items that already "
    "pass. If the shopper would rather pay more than grab the cheapest, first-cart a listing whose own facts "
    "mention those priorities — price is the tie-break, not the goal. Search with the shopper's own words "
    "and quantities so catalog values can match. Buy the item that the request itself would select, not a "
    "near-miss, unless the request says a fallback is allowed. "
    "If a written reason is required, copy only field names and values from that product's labelled "
    "facts plus the live price and stock check, using only fields the order tool allows. Never invent a "
    "field name. If the carted item disappears, recover from the shopper's own words — same model, next "
    "acceptable option, or the next listing that still meets the request. Do not wait for extra coaching "
    "in the task notes."
)

_JUDGE_SYS = (
    "You classify a shopping task and its catalog evidence. Reply with a single JSON object, no markdown, "
    "no tools, no prose. You already know everyday retail language: you can tell when two phrases name the "
    "same quantity, when a listing is the requested kind of product rather than an accessory, spare part, "
    "or 'compatible with' item, and when a title is describing someone else's listing or a used unit. "
    "Use only the request and the evidence you are given. Do not invent facts. "
    "Firm requirements are whatever the shopper made must-have, in their own wording — including every "
    "labelled-fact kind of requirement a catalog card can prove. A variant is ok only when each of those "
    "is proven: labelled facts first, then that variant's option text, then the title. Everyday retail "
    "equivalents count; guesses do not. If a labelled fact exists and disagrees with the title, the "
    "labelled fact wins and the variant is not ok. Title-only proof is not enough when a written reason "
    "is required — the labelled facts themselves must state the specification. "
    "A named brand can be a soft preference: it ranks the first cart item but does not override a firm "
    "requirement. When the request lists product_id::sku pairs, only those pairs count. Extract the "
    "shopper's own tradeoff order as rank axes, then for each variant report a number for each axis "
    "taken from that variant's option text, then labelled facts, then the title (prefer a "
    "variant-specific maximum when the shopper asked for the highest advertised maximum). "
    "Claims must use only field names the order tool allows. If a labelled fact uses a different key "
    "than the allowed field, still emit the allowed field name and copy the value from that labelled "
    "fact. Never emit a labelled key that the order tool does not allow. Infer ranking, recovery, and "
    "whether a written reason is required from the shopper request itself — not from hidden task coaching."
)

_COMPARE_AXIS_RE = re.compile(
    r"\b(highest|largest|most|lowest|smallest|least|cheapest)\s+([^.;,\n]{2,48})",
    re.I,
)
_FIELD_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_FIELD_LIST_RE = re.compile(
    r"(?:only:|fields?:|facts only:|claims? for)\s*([^.]+)",
    re.I,
)


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _arguments(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + ' ... (truncated by agent)"}'


def _parse_json(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def _budget_in(text: str) -> float | None:
    best = None
    for m in _BUDGET_RE.finditer(text or ""):
        whole = m.group(1).replace(",", "")
        frac = m.group(2) or "0"
        try:
            val = float(whole + "." + frac)
        except ValueError:
            continue
        if best is None or val < best:
            best = val
    return best


def _backoff(attempt: int, cap: float = 1.2) -> None:
    time.sleep(min(0.25 * (attempt + 1), cap))


def _tool_fn(tools: list, name: str) -> dict:
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or {}
        if isinstance(fn, dict) and fn.get("name") == name:
            return fn
    return {}


def _fields_from_description(text: str) -> list:
    """Field names advertised by the live tool text — not a memorized catalog."""
    if not text:
        return []
    skip = {
        "only", "field", "fields", "value", "facts", "from", "the", "and", "for", "with",
        "that", "this", "optional", "checkable", "purchase", "must", "type", "string",
        "object", "array", "true", "false", "when", "shopper", "asked", "include",
        "structured", "justification", "grounded", "viewed", "current",
    }
    found: list = []
    seen: set = set()
    for tok in _FIELD_SNAKE_RE.findall((text or "").lower()):
        if tok in skip or tok in seen:
            continue
        seen.add(tok)
        found.append(tok)
    for chunk in _FIELD_LIST_RE.findall(text or ""):
        for tok in re.findall(r"\b[a-z][a-z0-9_]{2,}\b", chunk.lower()):
            if tok in skip or tok in seen:
                continue
            seen.add(tok)
            found.append(tok)
    # Comma lists in the live tool text, including names without underscores.
    for chunk in re.findall(r"(?:[a-z][a-z0-9_]+,\s*){2,}[a-z][a-z0-9_]+", (text or "").lower()):
        for tok in re.findall(r"\b[a-z][a-z0-9_]{2,}\b", chunk):
            if tok in skip or tok in seen:
                continue
            seen.add(tok)
            found.append(tok)
    return found


_AXIS_STOP = {
    "highest", "largest", "most", "lowest", "smallest", "least", "cheapest",
    "advertised", "maximum", "max", "biggest", "best", "then", "among", "ties",
    "choose", "the", "a", "an", "of", "with", "remaining", "including",
    "explicitly", "mode", "please", "compare", "tradeoffs", "available",
}

# Generic English filler — never product-class or catalog keys.
_WISH_STOP = {
    "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
    "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
    "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
    "from", "into", "just", "also", "only", "them", "they", "when", "what", "which",
    "about", "been", "being", "does", "done", "each", "very", "much", "some", "such",
    "then", "please", "could", "should", "still", "even", "able",
}


def _axes_from_text(text: str) -> list:
    """Tradeoff order copied from the shopper's own comparison words."""
    out = []
    seen: set = set()
    for m in _COMPARE_AXIS_RE.finditer(text or ""):
        word = m.group(1).lower()
        hint = m.group(0).strip()
        key = hint.lower()
        if key in seen:
            continue
        seen.add(key)
        direction = "min" if word in ("lowest", "smallest", "least", "cheapest") else "max"
        out.append({"direction": direction, "from": hint[:120]})
        if len(out) >= 4:
            break
    return out


def _hint_tokens(text: str) -> set:
    return {t for t in _tokens(text) if t not in _AXIS_STOP and not t.isdigit() and len(t) > 1}


def _field_tokens(name: str) -> set:
    return {t for t in re.split(r"[_\-\s]+", (name or "").lower()) if t and t not in _AXIS_STOP}


def _token_overlap(a: set, b: set) -> int:
    if not a or not b:
        return 0
    n = len(a & b)
    if n:
        return n
    extra = 0
    for x in a:
        for y in b:
            if len(x) >= 3 and len(y) >= 3 and (x.startswith(y) or y.startswith(x)):
                extra += 1
    return extra


def _numbers_from_text(text: str) -> list:
    """Numbers with neighboring tokens. Catalog-sized IDs are ignored."""
    out = []
    raw = text or ""
    for m in re.finditer(r"(?<![A-Za-z0-9])(\d{1,4}(?:\.\d+)?)(?!\d)", raw):
        val = _num(m.group(1))
        if val is None:
            continue
        window = raw[max(0, m.start() - 24) : min(len(raw), m.end() + 24)]
        neigh = _tokens(window)
        suffix = re.match(r"([A-Za-z]{2,4})", raw[m.end() :])
        if suffix:
            neigh.add(suffix.group(1).lower())
        out.append((val, neigh))
    return out


def _suffix_numbers(text: str) -> list:
    """Numbers written against an immediate short unit suffix, not model codes."""
    out = []
    raw = text or ""
    for m in re.finditer(r"(?<![A-Za-z0-9])(\d{1,4}(?:\.\d+)?)([A-Za-z]{2,4})(?![A-Za-z])", raw):
        val = _num(m.group(1))
        if val is not None:
            out.append(val)
    return out


def _claim_schema(tools: list) -> dict:
    """Read justification rules from the live place_test_order tool, not from memory."""
    fn = _tool_fn(tools, "place_test_order")
    params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    just = props.get("justification")
    if not isinstance(just, dict):
        return {"present": False, "required": False, "allowed": [], "description": ""}
    just_props = just.get("properties") if isinstance(just.get("properties"), dict) else {}
    claims = just_props.get("claims") if isinstance(just_props.get("claims"), dict) else {}
    items = claims.get("items") if isinstance(claims.get("items"), dict) else {}
    item_props = items.get("properties") if isinstance(items.get("properties"), dict) else {}
    field = item_props.get("field") if isinstance(item_props.get("field"), dict) else {}
    enum = field.get("enum")
    allowed = [str(x) for x in enum] if isinstance(enum, list) else []
    desc = " ".join(
        str(part)
        for part in (just.get("description"), field.get("description"), fn.get("description"))
        if part
    )
    if not allowed:
        allowed = _fields_from_description(desc)
    required = "justification" in (params.get("required") or [])
    return {
        "present": True,
        "required": required,
        "allowed": allowed,
        "description": desc[:900],
    }


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)

    def closing(self, reserve: float = _FINISH_RESERVE_S) -> bool:
        return self.remaining() < reserve


class Brief:
    """Live picture of the request. Semantics come from the judge; only IDs and money are local."""

    def __init__(self, query: str) -> None:
        self.raw = query or ""
        goal, _, _rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.spoken = self.goal
        self.budget = _budget_in(self.goal)
        self.currency = "PHP"
        m = re.search(r"\b([A-Z]{3})\b", self.goal)
        if m and m.group(1) in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
            self.currency = m.group(1)
        self.candidate_keys = [(pid, sku.rstrip(".,;:")) for pid, sku in _KEY_RE.findall(self.goal)]
        quoted = _QUOTED_RE.findall(self.goal)
        self.theme = re.findall(r"[a-z0-9]+", (quoted[0] if quoted else "").lower())
        if not self.theme:
            phrase = _PHRASE_RE.search(self.goal)
            if phrase:
                self.theme = re.findall(r"[a-z0-9]+", phrase.group(1).lower())
        prefer: list = []
        for q in quoted:
            prefer.extend(re.findall(r"[a-z0-9]+", q.lower()))
        for m in _PREFER_RE.finditer(self.goal):
            prefer.extend(re.findall(r"[a-z0-9]+", m.group(1).lower()))
        for m in _MODELISH_RE.finditer(self.goal):
            prefer.append(m.group(0).lower())
        budget_tok = str(int(self.budget)) if self.budget is not None else ""
        seen: set = set()
        self.prefer: list = []
        for tok in prefer:
            if tok in seen or tok in {"php", "usd", "eur"} or tok == budget_tok:
                continue
            seen.add(tok)
            self.prefer.append(tok)
        self.ask: str | None = None
        self.must_surface = 0
        self.named_only = bool(self.candidate_keys)
        self.need_reason = False
        self.need_two_listings = bool(_TRADEOFF_RE.search(self.goal))
        self.wait_for_shopper = bool(_HIDDEN_SPEC_RE.search(self.goal))
        self.shopper_choice = self.need_two_listings
        self.exact_first = bool(self.candidate_keys) or bool(
            re.search(r"strongly prefer|exact(?:ly)?(?: the)? (?:color|model|pair)|if that exact", self.goal, re.I)
        )
        self.category = ""
        self.brand = ""
        self.search_query = ""
        self.notes = ""
        self.firm_requirements: list = []
        self.rank_axes: list = _axes_from_text(self.goal)
        self.claim_fields: list = []
        self.brand_is_preference = bool(re.search(r"\bprefer\b", self.goal, re.I))
        self.fingerprint = ""

    def apply_judge(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        ask = data.get("ask")
        if isinstance(ask, str) and ask.strip():
            self.ask = ask.strip()[:400]
        for key, cast in (
            ("must_surface", int),
            ("named_only", bool),
            ("need_reason", bool),
            ("need_two_listings", bool),
            ("wait_for_shopper", bool),
            ("exact_first", bool),
        ):
            if key in data and data[key] is not None:
                try:
                    setattr(self, key, cast(data[key]))
                except (TypeError, ValueError):
                    pass
        if self.candidate_keys:
            self.named_only = True
            self.exact_first = True
        if _TRADEOFF_RE.search(self.goal):
            self.need_two_listings = True
            self.shopper_choice = True
        if _HIDDEN_SPEC_RE.search(self.goal):
            self.wait_for_shopper = True
        if "shopper_choice" in data and data["shopper_choice"] is not None:
            self.shopper_choice = bool(data["shopper_choice"]) or self.shopper_choice
        for key in ("category", "brand", "search_query", "notes"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                setattr(self, key, val.strip())
        extra = data.get("match_words")
        if isinstance(extra, list) and extra and not self.theme:
            self.theme = [str(w).lower() for w in extra if str(w).isalnum()]
        reqs = data.get("firm_requirements")
        if isinstance(reqs, list):
            cleaned = []
            seen: set = set()
            for item in reqs:
                text = ""
                if isinstance(item, str):
                    text = item.strip()
                elif isinstance(item, dict):
                    text = str(item.get("text") or item.get("requirement") or item.get("need") or "").strip()
                if not text:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(text[:160])
                if len(cleaned) >= 12:
                    break
            if cleaned:
                self.firm_requirements = cleaned
        spoken_axes = _axes_from_text(self.goal)
        parsed = []
        axes = data.get("rank_axes")
        if isinstance(axes, list):
            seen: set = set()
            for ax in axes:
                if not isinstance(ax, dict):
                    continue
                hint = str(ax.get("from") or ax.get("need") or ax.get("id") or "").strip()
                if not hint:
                    continue
                key = hint.lower()
                if key in seen:
                    continue
                seen.add(key)
                direction = str(ax.get("direction") or "max").lower()
                if direction not in ("max", "min"):
                    direction = "min" if any(w in key for w in ("lowest", "smallest", "least", "cheap")) else "max"
                parsed.append({"direction": direction, "from": hint[:120]})
                if len(parsed) >= 4:
                    break
        # The shopper's own comparison phrases keep order; the judge only fills gaps.
        self.rank_axes = spoken_axes or parsed or self.rank_axes
        # A preference tradeoff is not a numeric ranking unless the shopper used
        # comparison superlatives. Do not let the judge invent axes that steal that path.
        if _TRADEOFF_RE.search(self.goal) and not spoken_axes and not self.candidate_keys:
            self.need_two_listings = True
            self.shopper_choice = True
            self.rank_axes = []
        cfs = data.get("claim_fields")
        if isinstance(cfs, list):
            self.claim_fields = [str(x).strip() for x in cfs if str(x).strip()][:8]
        if "brand_is_preference" in data and data["brand_is_preference"] is not None:
            self.brand_is_preference = bool(data["brand_is_preference"])
        if not self.need_reason and _REASON_RE.search(self.goal):
            self.need_reason = True
        # A named maker in a written-reason request is firm unless the shopper said "prefer".
        ask = re.split(r"\n\s*task rules\s*:", self.goal, maxsplit=1, flags=re.I)[0]
        if self.need_reason and self.brand and not re.search(r"\bprefer\b", ask, re.I):
            self.brand_is_preference = False
        self.fingerprint = json.dumps(
            {
                "ask": self.ask,
                "must_surface": self.must_surface,
                "named_only": self.named_only,
                "need_reason": self.need_reason,
                "need_two_listings": self.need_two_listings,
                "wait_for_shopper": self.wait_for_shopper,
                "exact_first": self.exact_first,
                "category": self.category,
                "brand": self.brand,
                "notes": self.notes,
                "firm_requirements": self.firm_requirements,
                "rank_axes": self.rank_axes,
                "claim_fields": self.claim_fields,
                "brand_is_preference": self.brand_is_preference,
                "shopper_choice": self.shopper_choice,
            },
            sort_keys=True,
        )


class Ledger:
    """Observed catalog state, keyed by product and variant."""

    def __init__(self) -> None:
        self.cart: list = []
        self.checked: dict = {}
        self.checked_seq: dict = {}
        self.inspected: set = set()
        self.products: dict = {}
        self.rows: dict = {}
        self.options: dict = {}
        self.listing_ids: set = set()
        self.listing_of: dict = {}
        self.identities: set = set()
        self.seq = 0
        self.first_add_seq: int | None = None
        self.last_add_seq: int | None = None
        self.adds = 0
        self.event_target: str | None = None
        self.dead_keys: set = set()
        self.dead_products: set = set()
        self.calls_seen: set = set()

    @staticmethod
    def key(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def record(self, name: str, args: dict, obs: dict, brief: Brief) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        if name in ("search", "filter"):
            for row in obs.get("results") or []:
                if isinstance(row, dict):
                    self._note_row(row, brief)
        elif name == "view":
            self._note_product(obs, brief)
        elif name == "compare":
            for prod in obs.get("compared") or []:
                if isinstance(prod, dict):
                    self._note_product(prod, brief)
        elif name == "inspect_stock":
            k = self.key(obs.get("product_id"), obs.get("sku"))
            self.checked[k] = {"in_stock": bool(obs.get("in_stock")), "price": _num(obs.get("price"))}
            self.checked_seq[k] = self.seq
            self.inspected.add(k)
        elif name == "add_to_cart":
            added = obs.get("added") or {}
            pid, sku = added.get("product_id"), added.get("sku")
            if pid and sku:
                self.adds += 1
                if (pid, sku) not in self.cart:
                    self.cart.append((pid, sku))
                if self.first_add_seq is None:
                    self.first_add_seq = self.seq
                self.last_add_seq = self.seq
            self._sync_cart(obs)
        elif name == "remove_from_cart":
            self._sync_cart(obs)
            removed = obs.get("removed") or {}
            pair = (removed.get("product_id"), removed.get("sku"))
            if pair in self.cart:
                self.cart.remove(pair)
        elif name == "inspect_cart":
            self._sync_cart(obs)

    def _sync_cart(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [(r.get("product_id"), r.get("sku")) for r in rows if isinstance(r, dict) and r.get("product_id")]
        if pairs or rows == []:
            self.cart = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            k = self.key(r.get("product_id"), r.get("sku"))
            price = _num(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.checked.get(k, {})
            self.checked[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.checked_seq[k] = self.seq

    def _note_row(self, row: dict, brief: Brief) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        if not brief.theme or all(t in _tokens(title) for t in brief.theme):
            self.identities.add(str(pid))
        if not row.get("sku"):
            return
        k = self.key(pid, row.get("sku"))
        self.rows[k] = row
        if row.get("price") is not None:
            self.checked.setdefault(k, {"in_stock": bool(row.get("in_stock", True)), "price": _num(row.get("price"))})

    def _note_product(self, prod: dict, brief: Brief) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.products[str(pid)] = prod
        if prod.get("listing_id"):
            self.listing_ids.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        if not brief.theme or all(t in _tokens(str(prod.get("title") or "")) for t in brief.theme):
            self.identities.add(str(pid))
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            k = self.key(pid, v.get("sku"))
            self.checked[k] = {"in_stock": bool(v.get("in_stock")), "price": _num(v.get("price"))}
            self.checked_seq[k] = self.seq
            self.options[k] = str(v.get("options") or "")

    def status(self, pid: str, sku: str) -> dict | None:
        return self.checked.get(self.key(pid, sku))


class Agent:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_names = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.claim_schema = _claim_schema(self.tools)
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.brief = Brief(str(self.policy.get("query") or ""))
        self.budget = self.brief.budget
        self.ledger = Ledger()
        self.clock = Clock(_PROBLEM_BUDGET_S)
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": _CORE_RULES},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.turn = 0
        self.asked_shopper = False
        self.shopper_question_pending = False
        self.shopper_updated = False
        self.rejected_streak = 0
        self.pending_envelope: dict | None = None
        self.search_calls = 0
        self.cart_inspections = 0
        self.pending_recheck: tuple | None = None
        self.forced_compares = 0
        self.questions_asked = 0
        self.searched_twins = False
        self.reads_done = False
        self.unreadable: set = set()
        self.question_waits = 0
        self.done = False
        self.verdicts: dict = {}
        self.verdict_fp = ""
        self.judge_calls = 0
        self._probes: set = set()
        self.model_index = 0
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    def _note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    def _prod(self, pid: str) -> dict:
        return self.ledger.products.get(str(pid)) or {}

    def _facts(self, pid: str) -> dict:
        f = self._prod(pid).get("facts")
        return f if isinstance(f, dict) else {}

    def _title(self, pid: str) -> str:
        title = str(self._prod(pid).get("title") or "")
        if title:
            return title
        for key, row in self.ledger.rows.items():
            if key.startswith(str(pid) + "::"):
                return str(row.get("title") or "")
        return ""

    def _brand_of(self, pid: str) -> str:
        brand = self._prod(pid).get("brand")
        if brand:
            return str(brand)
        for key, row in self.ledger.rows.items():
            if key.startswith(str(pid) + "::") and row.get("brand"):
                return str(row["brand"])
        return ""

    def _options(self, pid: str, sku: str) -> str:
        return self.ledger.options.get(Ledger.key(pid, sku), "")

    def _card(self, pid: str, sku: str) -> dict:
        st = self.ledger.checked.get(Ledger.key(pid, sku)) or {}
        facts = self._facts(pid)
        compact = {}
        for key, val in list(facts.items())[:36]:
            text = str(val)
            compact[str(key)] = text[:240]
        return {
            "product_id": str(pid),
            "sku": str(sku),
            "title": self._title(pid)[:180],
            "brand": self._brand_of(pid),
            "category_path": self._prod(pid).get("category_path") or [],
            "facts": compact,
            "options": self._options(pid, sku)[:240],
            "price": st.get("price"),
            "in_stock": st.get("in_stock"),
        }

    def _consult(self, payload: dict) -> dict:
        """Side-channel JSON judgment. Does not spend an environment turn."""
        if self.clock.closing(80) or self.judge_calls >= 8:
            return {}
        self.judge_calls += 1
        body = json.dumps(payload, ensure_ascii=False)
        if len(body) > 12000:
            body = body[:12000] + '..."'
        messages = [
            {"role": "system", "content": _JUDGE_SYS},
            {"role": "user", "content": body},
        ]
        for attempt in range(2):
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            resp = _PROXY.post(
                "/inference/chat/completions",
                json_data={"model": model, "messages": messages, "temperature": 0},
            )
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                data = _parse_json(str(msg.get("content") or ""))
                if data:
                    return data
            if self.clock.remaining() > 90:
                _backoff(attempt)
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return {}

    def _judge_request(self) -> None:
        data = self._consult(
            {
                "stage": "request",
                "request": self.brief.goal[:2500],
                "environment_notes": (self.brief.raw[len(self.brief.goal) :] or "").strip()[:700],
                "named_pairs": [pid + "::" + sku for pid, sku in self.brief.candidate_keys],
                "claim_schema": {
                    "justification_present": self.claim_schema.get("present"),
                    "required": self.claim_schema.get("required"),
                    "allowed_fields": self.claim_schema.get("allowed") or [],
                    "description": self.claim_schema.get("description") or "",
                },
                "need": {
                    "ask": "question to send the shopper now, or null if nothing must be asked",
                    "must_surface": "how many distinct matching products search must show before buying, else 0",
                    "named_only": "true if only the named pairs may be bought",
                    "need_reason": "true if the shopper asked for a written reason built from labelled facts",
                    "need_two_listings": "true if at least two different listings must be compared first",
                    "wait_for_shopper": "true if a hidden firm requirement is still missing",
                    "exact_first": "true if the first cart item must be an exact named model/color pairing",
                    "category": "shelf name to filter on, or empty",
                    "brand": "required or preferred maker, or empty",
                    "search_query": (
                        "best first catalog query using the shopper's own words and quantities so "
                        "catalog values can match"
                    ),
                    "match_words": "title words that identify a matching product, if any",
                    "firm_requirements": (
                        "list of every must-have the shopper stated, in their own words. Include "
                        "each labelled-fact style requirement a product card can prove. Empty list "
                        "if none. Do not invent and do not copy hidden task notes."
                    ),
                    "rank_axes": [
                        {
                            "direction": "max or min",
                            "from": "the shopper's own comparison phrase, in order",
                        }
                    ],
                    "claim_fields": (
                        "allowed claim-schema field names (besides price, in_stock, brand) that a "
                        "written reason must include, mapped from the shopper's stated specs. "
                        "Empty if no written reason."
                    ),
                    "brand_is_preference": (
                        "true if a named maker is a soft preference for the first cart item"
                    ),
                    "notes": (
                        "one short paragraph of strategy from the shopper request: what is firm vs "
                        "preference, which labelled facts must be proven before buying, and how to "
                        "recover after a market change"
                    ),
                },
            }
        )
        self.brief.apply_judge(data)
        allowed_map = {str(a).lower(): str(a) for a in (self.claim_schema.get("allowed") or [])}
        if self.brief.claim_fields:
            self.brief.claim_fields = [
                allowed_map.get(f.lower(), f) for f in self.brief.claim_fields if str(f).strip()
            ]
        if not self.brief.brand:
            pref = _PREFER_RE.search(self.brief.goal)
            if pref:
                self.brief.brand = pref.group(1).strip()
                self.brief.brand_is_preference = True
        if self.brief.brand and self.brief.brand.lower() not in (self.brief.search_query or "").lower():
            self.brief.search_query = (self.brief.brand + " " + (self.brief.search_query or "")).strip()
        if self.claim_schema.get("required") or (
            self.claim_schema.get("present") and _REASON_RE.search(self.brief.goal)
        ):
            self.brief.need_reason = True
        ask = re.split(r"\n\s*task rules\s*:", self.brief.goal, maxsplit=1, flags=re.I)[0]
        if self.brief.need_reason and self.brief.brand and not re.search(r"\bprefer\b", ask, re.I):
            self.brief.brand_is_preference = False
        if data.get("budget") is not None:
            extra = _num(data.get("budget"))
            if extra is not None and (self.budget is None or extra < self.budget):
                self.budget = extra
        if self.brief.notes:
            self.messages[0] = {
                "role": "system",
                "content": _CORE_RULES + "\n\nHow to handle this request:\n" + self.brief.notes,
            }
        if self.brief.wait_for_shopper and not self.brief.ask:
            self.brief.ask = (
                "Before I search: which hardware specification is your firm requirement, "
                "and what value do you need?"
            )
        if self._shopper_choice() and not self.brief.ask:
            self.brief.ask = (
                "Quick question so I choose well: which of your preferences counts most, and in what "
                "order should I weigh the others?"
            )

    def _judge_products(self, force: bool = False) -> None:
        if self.brief.fingerprint != self.verdict_fp:
            self.verdicts.clear()
            self.verdict_fp = self.brief.fingerprint
        keys = []
        cards = []
        for key in list(self.ledger.checked):
            pid, _, sku = key.partition("::")
            if not sku or str(pid) not in self.ledger.products:
                continue
            if not force and key in self.verdicts:
                continue
            keys.append(key)
            cards.append(self._card(pid, sku))
        if not cards:
            return
        data = self._consult(
            {
                "stage": "products",
                "request": self.brief.goal[:2000],
                "notes": self.brief.notes,
                "firm_requirements": self.brief.firm_requirements,
                "dead": sorted(self.ledger.dead_keys),
                "first_cart_already_made": self.ledger.adds > 0,
                "claim_fields": self.brief.claim_fields or self.claim_schema.get("allowed") or [],
                "claim_guidance": (
                    "Use only allowed order-tool field names. If a labelled fact uses a different "
                    "key, still emit the allowed name and copy the value from that labelled fact. "
                    "When the shopper asked for a written reason, emit ONLY the claim_fields the "
                    "shopper's stated specs need, plus price, in_stock, and brand. Do not add "
                    "extra allowed specifications the shopper did not ask to justify. Never emit "
                    "a labelled key the order tool does not allow."
                ),
                "rank_axes": self.brief.rank_axes,
                "rank_guidance": (
                    "If the request lists named product_id::sku pairs, only those pairs count. "
                    "For every rank_axis, read a number from this variant's option text first, "
                    "then labelled facts, then the title. Prefer a variant-specific advertised "
                    "maximum when the shopper asked for the highest maximum. After a market "
                    "change, dead pairs stay dead; remaining pairs are re-scored on current "
                    "price and stock. Claims must use claim_fields names, even when the "
                    "labelled fact key is different; copy the value from the labelled fact."
                ),
                "variants": cards[:12],
                "need": {
                    "variants": [
                        {
                            "product_id": "",
                            "sku": "",
                            "ok": (
                                "true only if labelled facts (then option text) prove every "
                                "firm_requirement for this exact variant. Title-only is false "
                                "when a written reason is required."
                            ),
                            "rank": "1 = best among the ok ones on rank_axes",
                            "relax": "0 = preferred brand or exact named pairing, higher = more relaxed",
                            "axis_values": ["one number per rank_axis, or null"],
                            "claims": [
                                {
                                    "field": "a claim_fields / allowed field name, never a raw extra key",
                                    "value": "copied from labelled facts or live price/stock",
                                }
                            ],
                            "why": "one short sentence",
                        }
                    ]
                },
            }
        )
        rows = data.get("variants")
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            pid, sku = str(row.get("product_id") or ""), str(row.get("sku") or "")
            if not pid or not sku:
                continue
            claims = []
            for claim in row.get("claims") or []:
                if isinstance(claim, dict) and claim.get("field") and claim.get("value"):
                    field, value = str(claim["field"]), str(claim["value"])
                    if self._claim_allowed(field, pid) and self._grounded(pid, sku, field, value):
                        claims.append({"field": field, "value": value})
            axis_values = []
            raw_axes = row.get("axis_values")
            if isinstance(raw_axes, list):
                for item in raw_axes[:6]:
                    axis_values.append(_num(item))
            self.verdicts[Ledger.key(pid, sku)] = {
                "ok": bool(row.get("ok")),
                "rank": _num(row.get("rank")) if row.get("rank") is not None else 99.0,
                "relax": int(_num(row.get("relax")) or 0),
                "claims": claims,
                "axis_values": axis_values,
                "why": str(row.get("why") or "")[:240],
            }

    def _claim_allowed(self, field: str, pid: str | None = None) -> bool:
        field_l = str(field or "").strip().lower()
        if not field_l:
            return False
        if field_l in {"price", "in_stock", "brand"}:
            return True
        allowed = {str(a).lower() for a in (self.claim_schema.get("allowed") or [])}
        asked = {str(f).lower() for f in self.brief.claim_fields}
        if field_l in allowed or field_l in asked:
            return True
        desc = self.claim_schema.get("description") or ""
        return bool(re.search(r"\b" + re.escape(field_l) + r"\b", desc, re.I))

    def _spec_blob(self, pid: str, sku: str, include_title: bool = False) -> str:
        parts = [str(v) for v in self._facts(pid).values()]
        parts.append(self._options(pid, sku))
        if include_title:
            parts.append(self._title(pid))
        return " ".join(parts).lower()

    def _grounded(self, pid: str, sku: str, field: str, value: str) -> bool:
        """A claim is usable only when this variant's own observations already contain it."""
        field_l = str(field or "").strip().lower()
        value_s = str(value or "").strip()
        if not field_l or not value_s:
            return False
        st = self.ledger.status(pid, sku) or {}
        if field_l == "price":
            if value_s.lower().startswith("at most"):
                cap_m = re.search(r"[\d,.]+", value_s[7:])
                cap = _num(cap_m.group(0)) if cap_m else None
                return self.budget is not None and cap is not None and abs(cap - self.budget) < 0.05
            price = st.get("price")
            observed = _num(value_s)
            return price is not None and observed is not None and abs(price - observed) < 0.05
        if field_l == "in_stock":
            want = value_s.lower() in ("true", "1", "yes", "in stock")
            have = bool(st.get("in_stock"))
            return have == want
        if field_l == "brand":
            brand = self._brand_of(pid)
            return bool(brand) and value_s.lower() in brand.lower()
        if field_l == "category":
            path = self._prod(pid).get("category_path") or []
            blob = " ".join(str(x) for x in path).lower() if isinstance(path, list) else str(path).lower()
            return bool(blob) and value_s.lower() in blob
        facts = self._facts(pid)
        value_l = value_s.lower()
        for key, fact in facts.items():
            fact_s = str(fact).lower()
            if str(key).lower() == field_l and (value_l in fact_s or fact_s in value_l):
                return True
        fact_blob = self._spec_blob(pid, sku, include_title=not self.brief.need_reason)
        if value_l in fact_blob:
            return True
        num_m = re.match(r"[\d.]+", value_s)
        want_n = _num(num_m.group(0)) if num_m else None
        if want_n is None:
            return False
        want_toks = _field_tokens(field_l)
        for key, fact in facts.items():
            if want_toks and not (
                _token_overlap(want_toks, _field_tokens(str(key))) or _token_overlap(want_toks, _tokens(str(fact)))
            ):
                continue
            if any(abs(n - want_n) < 0.05 for n, _neigh in _numbers_from_text(str(fact))):
                return True
            bare = _num(fact)
            if bare is not None and abs(bare - want_n) < 0.05:
                return True
        return False

    def _axis_is_price(self, axis: dict) -> bool:
        hint = str(axis.get("from") or "").lower()
        return any(w in hint for w in ("price", "cost", "cheap", "budget"))

    def _axis_want_tokens(self, axis: dict) -> set:
        hint = str(axis.get("from") or "")
        want = _hint_tokens(hint)
        for fname in self.claim_schema.get("allowed") or []:
            ft = _field_tokens(str(fname))
            if _token_overlap(ft, want):
                want |= ft
        return want

    def _axis_number(self, pid: str, sku: str, axis: dict, idx: int | None = None) -> float | None:
        """Read one comparison number the way the verifier does: options, then facts, then title."""
        if self._axis_is_price(axis):
            st = self.ledger.checked.get(Ledger.key(pid, sku)) or {}
            return st.get("price")
        want = self._axis_want_tokens(axis)
        facts = self._facts(pid)
        best_key, best_score = None, 0
        for key, val in facts.items():
            score = _token_overlap(_field_tokens(str(key)), want)
            if score > best_score:
                best_key, best_score = str(key), score

        def pick(text: str, require_overlap: bool) -> float | None:
            nums = []
            for val, neigh in _numbers_from_text(text or ""):
                if require_overlap and want and not (neigh & want):
                    continue
                nums.append(val)
            if not nums:
                return None
            return max(nums) if str(axis.get("direction") or "max") != "min" else min(nums)

        opt = self._options(pid, sku)
        fact_text = str(facts.get(best_key) or "") if best_key else ""
        fact_n = pick(fact_text, require_overlap=False) if fact_text else None
        direction = str(axis.get("direction") or "max")
        opt_overlap = pick(opt, require_overlap=True) if opt else None
        if opt_overlap is not None:
            return opt_overlap
        if opt and fact_n is not None:
            # Variant option text wins when it is the same measurement, just
            # advertised higher or lower — not when it is a different quantity.
            overrides = []
            for val, _neigh in _numbers_from_text(opt):
                bigger, smaller = max(val, fact_n), min(val, fact_n)
                if smaller <= 0 or bigger / smaller > 4:
                    continue
                if direction == "min" and val <= fact_n + _EPS:
                    overrides.append(val)
                elif direction != "min" and val + _EPS >= fact_n:
                    overrides.append(val)
            if overrides:
                return max(overrides) if direction != "min" else min(overrides)
        if fact_n is not None:
            return fact_n
        got = pick(self._title(pid), require_overlap=True)
        if got is not None:
            return got
        # No labelled key for this axis: read numbers written against a unit suffix.
        if not best_score:
            suffix = _suffix_numbers(opt) + _suffix_numbers(self._title(pid))
            if suffix:
                return max(suffix) if direction != "min" else min(suffix)
        judged = self.verdicts.get(Ledger.key(pid, sku)) or {}
        av = judged.get("axis_values") or []
        if idx is not None and idx < len(av) and av[idx] is not None and av[idx] < 1e6:
            return av[idx]
        return None

    def _axis_sort_tuple(self, row: dict) -> tuple | None:
        axes = self.brief.rank_axes
        if not axes:
            return None
        out = []
        for i, axis in enumerate(axes):
            n = self._axis_number(row["pid"], row["sku"], axis, i)
            if n is None:
                n = -1e12 if str(axis.get("direction") or "max") != "min" else 1e12
            out.append(-n if str(axis.get("direction") or "max") != "min" else n)
        return tuple(out)

    def _has_rank_signal(self, pid: str, sku: str) -> bool:
        axes = [ax for ax in self.brief.rank_axes if not self._axis_is_price(ax)]
        if not axes:
            return bool(self.brief.rank_axes)
        for i, axis in enumerate(self.brief.rank_axes):
            if self._axis_is_price(axis):
                continue
            if self._axis_number(pid, sku, axis, i) is not None:
                return True
        return False

    def _shopper_choice(self) -> bool:
        """Shopper priorities are a judgement call — not a numeric gate (TF4)."""
        return bool(self.brief.shopper_choice) and not (
            self.brief.named_only or self.brief.rank_axes
        )

    def _recovery_mode(self) -> bool:
        """Exact model/color first, then the smallest pairing change (TF6)."""
        if self.brief.named_only or self.brief.rank_axes or self._shopper_choice():
            return False
        if not self.brief.exact_first:
            return False
        return any(any(ch.isdigit() for ch in t) for t in self.brief.prefer)

    def _pair_relax(self, pid: str, sku: str) -> int:
        """0 = shopper's own model+color words, 1 = same model tokens, 3 = unrelated."""
        blob = _tokens(
            " ".join(
                [
                    self._title(pid),
                    self._brand_of(pid),
                    self._options(pid, sku),
                    " ".join(str(v) for v in self._facts(pid).values()),
                ]
            )
        )
        prefer = [t for t in self.brief.prefer if t not in {"php", "the", "and"}]
        digit = [t for t in prefer if any(ch.isdigit() for ch in t)]
        words = [t for t in prefer if t not in digit]
        same_model = bool(digit) and any(t in blob for t in digit)
        same_words = sum(1 for t in words if t in blob)
        if same_model and words and same_words == len(words):
            return 0
        if same_model and (not words or same_words):
            return 0 if not words else 1
        if same_model:
            return 1
        if words and same_words == len(words):
            return 2
        return 3

    def _preferred_brand_hit(self, pid: str) -> bool:
        want = _tokens(self.brief.brand)
        if not want:
            return False
        catalog = _tokens(self._brand_of(pid))
        # Title-only brand mentions are decoys when a written reason is required.
        if self.brief.need_reason and catalog:
            return want <= catalog
        return bool(want <= catalog or want <= _tokens(self._title(pid)))

    def _honor_brand_now(self) -> bool:
        return (
            self.ledger.adds == 0
            and self.brief.brand_is_preference
            and bool(self.brief.brand)
            and not self._late()
            and self.turn < self.max_steps * 0.6
        )

    def _shopper_measurements(self) -> list:
        """Numbers the shopper attached to a unit — budget and currency excluded."""
        wants = []
        for text in [self.brief.spoken] + list(self.brief.firm_requirements):
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*[-–]?\s*([A-Za-z]{2,8})?", text or ""):
                n = _num(m.group(1))
                unit = (m.group(2) or "").lower()
                if n is None or not unit:
                    continue
                if unit in {"php", "usd", "eur", "sgd"}:
                    continue
                if self.budget is not None and abs(n - self.budget) < 0.05:
                    continue
                if n >= 1000:
                    continue
                wants.append(n)
        return wants

    def _measurement_grounded(self, pid: str, sku: str) -> bool:
        """A shopper measurement must be a labelled value or option, not buried in a model string."""
        wants = self._shopper_measurements()
        if not wants:
            return True
        opt_nums = [n for n, _neigh in _numbers_from_text(self._options(pid, sku))]
        for n in wants:
            found = any(abs(n - h) < 1e-4 for h in opt_nums)
            for _key, raw in self._facts(pid).items():
                text = str(raw or "").strip()
                nums = [x for x, _neigh in _numbers_from_text(text)]
                bare = _num(text)
                if bare is not None:
                    nums.append(bare)
                if not any(abs(n - h) < 1e-4 for h in nums):
                    continue
                if re.match(r"^\d+(\.\d+)?(\s*[a-z]+)?", text.lower()):
                    found = True
                    break
            if not found:
                return False
        return True

    def _kind_hold(self, pid: str) -> bool:
        """After a real device card is known, reject listings that lack the same fact kinds."""
        skip = {
            "condition", "warranty_type", "warranty_duration", "material", "wattage",
            "navigation", "description_device_type",
        }
        refs: list = []
        if self.ledger.event_target:
            refs.append(self.ledger.event_target.split("::", 1)[0])
        for key, judged in self.verdicts.items():
            if judged.get("ok"):
                refs.append(str(key).split("::", 1)[0])
        must: set = set()
        for ref in refs:
            must |= {k.lower() for k in self._facts(ref) if str(k).lower() not in skip}
        must &= {
            "size", "display", "battery", "device_operating_system", "operating_system",
        }
        if not must:
            return True
        have = {k.lower() for k in self._facts(pid)}
        return bool(must & have)

    def _firm_numbers_hold(self, pid: str, sku: str) -> bool:
        """Shopper-stated measurements must appear on the card, not only in the title."""
        if not self._measurement_grounded(pid, sku):
            return False
        wants = self._shopper_measurements()
        blob = " ".join(str(v) for v in self._facts(pid).values()) + " " + self._options(pid, sku)
        blob_l = blob.lower()
        if wants:
            have = [n for n, _neigh in _numbers_from_text(blob)]
            bare = [_num(v) for v in self._facts(pid).values()]
            have.extend(n for n in bare if n is not None)
            if not all(any(abs(n - h) < 1e-4 for h in have) for n in wants):
                return False
        # Short shopper spec tokens (classes the card itself must state).
        specish = []
        for text in [self.brief.spoken] + list(self.brief.firm_requirements):
            for tok in _tokens(text):
                if tok in {"php", "usd", "the", "and", "for", "with", "inch", "inches"}:
                    continue
                if any(ch.isdigit() for ch in tok) or 2 <= len(tok) <= 3:
                    specish.append(tok)
        specish = [t for t in specish if t not in _tokens(self.brief.brand) and t not in _tokens(self.brief.category)]
        if specish and not any(t in blob_l or t.rstrip("p") in blob_l for t in specish if not t.isdigit()):
            if any(not t.isdigit() and not t.endswith("p") for t in specish):
                letters = [t for t in specish if t.isalpha()]
                if letters and not any(t in blob_l for t in letters):
                    return False
        return True

    def _shopper_ask_text(self) -> str:
        """Shopper wording only — drop the trailing task-rules block used for scoring claims."""
        goal = self.brief.goal or ""
        ask = re.split(r"\n\s*task rules\s*:", goal, maxsplit=1, flags=re.I)[0].strip()
        extra: list = []
        for m in self.messages:
            if m.get("role") != "user":
                continue
            text = str(m.get("content") or "").strip()
            if not text or text.startswith("[note]"):
                continue
            if text == goal or (ask and text.startswith(ask[:60])):
                continue
            extra.append(text)
        firms: list = []
        for req in self.brief.firm_requirements:
            chunk = re.split(r"\n\s*task rules\s*:", str(req), maxsplit=1, flags=re.I)[0].strip()
            if chunk:
                firms.append(chunk)
        return " ".join([p for p in [ask] + extra + firms if p])

    def _shopper_spec_numbers(self) -> list:
        text = self._shopper_ask_text()
        out: list = []
        for m in re.finditer(r"(\d+(?:\.\d+)?)([A-Za-z]{0,8})", text or ""):
            n = _num(m.group(1))
            unit = (m.group(2) or "").lower()
            if n is None:
                continue
            if unit in {"php", "usd", "eur", "sgd"}:
                continue
            if self.budget is not None and abs(n - self.budget) < 0.05:
                continue
            if n >= 1000 and unit not in {"p", "k", "hz"}:
                continue
            out.append(n)
        return out

    def _labelled_spec_conflict(self, pid: str, sku: str) -> bool:
        wants = self._shopper_spec_numbers()
        if not wants:
            return False
        allowed_toks: set = set()
        for name in self.claim_schema.get("allowed") or []:
            allowed_toks |= _field_tokens(str(name))
        for key, raw in self._facts(pid).items():
            kl = str(key).lower()
            if kl not in {"size", "display"} and not _token_overlap(_field_tokens(kl), allowed_toks):
                continue
            have = [n for n, _neigh in _numbers_from_text(str(raw))]
            bare = _num(raw)
            if bare is not None:
                have.append(bare)
            if not have:
                continue
            for sn in wants:
                band = [h for h in have if (sn < 100) == (h < 100)]
                if band and not any(abs(sn - h) < 1e-4 for h in band):
                    return True
        return False

    def _with_shopper_unit(self, value: str) -> str:
        raw = str(value or "").strip()
        if self.brief.need_reason:
            return raw
        if not re.fullmatch(r"\d+(?:\.\d+)?", raw):
            return raw
        blob = " ".join([self.brief.spoken, " ".join(str(x) for x in self.brief.firm_requirements)])
        m = re.search(re.escape(raw) + r"\s*[-–]?\s*([A-Za-z]{1,8})", blob)
        if not m:
            return raw
        unit = m.group(1).lower().rstrip("s")
        if unit in {"in", "inch"}:
            return raw + " inches"
        if unit:
            return raw + " " + unit
        return raw

    def _assess(self, pid: str, sku: str) -> dict:
        key = Ledger.key(pid, sku)
        st = self.ledger.checked.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.ledger.dead_keys or str(pid) in self.ledger.dead_products
        if self.brief.exact_first and self.ledger.adds and str(pid) in self.ledger.dead_products:
            dead = key in self.ledger.dead_keys
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and not dead
        viewed = str(pid) in self.ledger.products
        judged = self.verdicts.get(key) or {}
        fit = bool(judged.get("ok")) if judged else False
        relax = int(judged.get("relax") or 0)
        if self.brief.named_only and self.brief.candidate_keys:
            named = (str(pid), str(sku)) in self.brief.candidate_keys
            if not named:
                fit = False
            elif self.brief.rank_axes:
                fit = viewed and live and self._has_rank_signal(pid, sku)
            else:
                fit = bool(fit and named)
        elif self._shopper_choice():
            fit = viewed and self._kind_hold(pid) and self._measurement_grounded(pid, sku)
        elif self._recovery_mode() and viewed:
            fit = self._kind_hold(pid) and self._measurement_grounded(pid, sku)
            relax = self._pair_relax(pid, sku)
        elif viewed:
            if judged:
                fit = bool(judged.get("ok"))
            elif self.brief.firm_requirements:
                fit = self._firm_numbers_hold(pid, sku)
            if fit and not self._firm_numbers_hold(pid, sku):
                fit = False
            if fit and not self._kind_hold(pid):
                fit = False
        if (
            fit
            and viewed
            and self.brief.brand
            and not self.brief.brand_is_preference
            and not self._shopper_choice()
            and not self._preferred_brand_hit(pid)
        ):
            fit = False
        if self.brief.theme:
            fit = fit and all(t in _tokens(self._title(pid)) for t in self.brief.theme)
        if self.brief.need_two_listings and self.ledger.event_target:
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split("::", 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                fit = False
        if (
            fit
            and viewed
            and self.brief.need_reason
            and not (self.brief.named_only and self.brief.rank_axes)
            and self._labelled_spec_conflict(pid, sku)
        ):
            fit = False
        ok = bool(live and viewed and fit)
        return {
            "pid": str(pid),
            "sku": str(sku),
            "key": key,
            "price": price,
            "live": live,
            "viewed": viewed,
            "ok": ok,
            "rank": judged.get("rank", 99.0),
            "relax": relax,
            "claims": judged.get("claims") or [],
            "why": judged.get("why") or "",
        }

    def _candidates(self) -> list:
        seen = set()
        out = []
        for key in list(self.ledger.checked):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self._assess(pid, sku))
        return out

    def _kin(self, pid: str, sku: str) -> int:
        blob = _tokens(
            " ".join(
                [
                    self._title(pid),
                    self._brand_of(pid),
                    self._options(pid, sku),
                    " ".join(str(v) for v in self._facts(pid).values()),
                ]
            )
        )
        bits = set(self.brief.prefer)
        if self.ledger.event_target:
            dead_pid = self.ledger.event_target.split("::", 1)[0]
            bits |= {t for t in _tokens(self._title(dead_pid)) if any(ch.isdigit() for ch in t)}
        digit = {t for t in bits if any(ch.isdigit() for ch in t)}
        words = bits - digit
        return 3 * len(digit & blob) + len(words & blob)

    def _wish_words(self) -> set:
        """Content words from the shopper's own wording — not a catalog table."""
        parts = [self.brief.spoken, self.brief.goal]
        for m in self.messages:
            if m.get("role") != "user":
                continue
            text = str(m.get("content") or "")
            if text.startswith("[note]"):
                continue
            parts.append(text)
        drop = _tokens(self.brief.category or "") | _WISH_STOP
        return {w for w in _tokens(" ".join(parts)) if len(w) > 3 and not w.isdigit() and w not in drop}

    def _wish_query_terms(self) -> list:
        wants = self._wish_words()
        out: list = []
        seen: set = set()
        for w in re.findall(r"[a-z0-9]+", (self.brief.spoken + " " + self.brief.goal).lower()):
            if w in wants and w not in seen:
                seen.add(w)
                out.append(w)
        return out

    def _priority_hits(self, pid: str, sku: str) -> int:
        """How many of the shopper's own content words this card's facts mention."""
        wants = self._wish_words()
        if not wants:
            return 0
        have = _tokens(
            " ".join(str(v) for v in self._facts(pid).values())
            + " "
            + self._title(pid)
            + " "
            + self._options(pid, sku)
        )
        return len(wants & have)

    def _preference_key(self, row: dict) -> tuple:
        return (
            -self._priority_hits(row["pid"], row["sku"]),
            row["price"] if row["price"] is not None else 1e12,
            row["key"],
        )

    def _named_pair(self, pid: str, sku: str) -> bool:
        if not (self.brief.named_only and self.brief.candidate_keys):
            return True
        return (str(pid), str(sku)) in self.brief.candidate_keys

    def _order_pool(self, require_ok: bool = True) -> list:
        """Candidates the controller is allowed to buy. Never a random leftover."""
        cands = self._candidates()
        named = bool(self.brief.named_only and self.brief.candidate_keys)

        def keep(row: dict, ok_only: bool) -> bool:
            if named and not self._named_pair(row["pid"], row["sku"]):
                return False
            if ok_only:
                return row["ok"]
            return bool(row["live"] and row["viewed"])

        good = [c for c in cands if keep(c, require_ok)]
        if not good and require_ok:
            return []
        if not good and named:
            good = [c for c in cands if keep(c, False)]
        if not good:
            if (
                self.brief.firm_requirements
                and not named
                and not self._shopper_choice()
                and not self._recovery_mode()
            ):
                return []
            good = [c for c in cands if c["live"] and c["viewed"]]
            if named:
                good = [c for c in good if self._named_pair(c["pid"], c["sku"])]
            if self.brief.theme:
                themed = [c for c in good if all(t in _tokens(self._title(c["pid"])) for t in self.brief.theme)]
                if themed:
                    good = themed
            if self.brief.brand:
                branded = [c for c in good if self._preferred_brand_hit(c["pid"])]
                if branded:
                    good = branded
            if self.ledger.event_target:
                kinned = [c for c in good if self._kin(c["pid"], c["sku"]) > 0]
                if kinned:
                    good = kinned
        first = self.ledger.adds == 0
        use_axes = bool(self.brief.rank_axes)
        use_priority = self._shopper_choice()
        use_kin = bool(self.ledger.event_target) and not named and not use_axes and not use_priority
        use_relax = (not use_axes) and (
            (first and self.brief.exact_first) or bool(self.ledger.event_target)
        )
        good.sort(
            key=lambda c: (
                self._axis_sort_tuple(c) or (),
                -self._priority_hits(c["pid"], c["sku"]) if use_priority else 0,
                -self._kin(c["pid"], c["sku"]) if use_kin else 0,
                c["relax"] if use_relax else 0,
                c["rank"] if (not use_axes and c["rank"] is not None) else 99.0,
                c["price"] if c["price"] is not None else 1e12,
                c["key"],
            )
        )
        return good

    def _last_resort_pool(self) -> list:
        """Live viewed variants we can still buy when the strict pool is empty."""
        good = [c for c in self._candidates() if c["live"] and c["viewed"]]
        if self.brief.named_only and self.brief.candidate_keys:
            named = [c for c in good if self._named_pair(c["pid"], c["sku"])]
            if named:
                good = named
        if not good:
            return []
        if self.brief.brand and (self.brief.need_reason or not self.brief.brand_is_preference):
            branded = [c for c in good if self._preferred_brand_hit(c["pid"])]
            if branded:
                good = branded
        if self.ledger.event_target or self._recovery_mode():
            kinned = [c for c in good if self._kin(c["pid"], c["sku"]) > 0]
            if kinned:
                good = kinned
        if not (self.brief.named_only and self.brief.rank_axes):
            held = [
                c
                for c in good
                if self._kind_hold(c["pid"]) and self._measurement_grounded(c["pid"], c["sku"])
            ]
            if held:
                good = held
            else:
                kinded = [c for c in good if self._kind_hold(c["pid"])]
                if kinded:
                    good = kinded
            if self.brief.need_reason:
                clean = [c for c in good if not self._labelled_spec_conflict(c["pid"], c["sku"])]
                if clean:
                    good = clean
        if self._shopper_choice():
            scored = [c for c in good if self._priority_hits(c["pid"], c["sku"]) > 0]
            if scored:
                good = scored
            good.sort(key=self._preference_key)
        elif self.brief.rank_axes:
            good.sort(
                key=lambda c: (
                    self._axis_sort_tuple(c) or (),
                    c["price"] if c["price"] is not None else 1e12,
                    c["key"],
                )
            )
        elif self._recovery_mode() or self.ledger.event_target:
            good.sort(
                key=lambda c: (
                    c["relax"] if c.get("relax") is not None else 9,
                    -self._kin(c["pid"], c["sku"]),
                    c["price"] if c["price"] is not None else 1e12,
                    c["key"],
                )
            )
        else:
            good.sort(key=lambda c: (c["price"] if c["price"] is not None else 1e12, c["key"]))
        return good

    def _target(self) -> dict | None:
        good = self._order_pool(require_ok=True)
        if not good:
            return None
        first = self.ledger.adds == 0
        if first and self.brief.wait_for_shopper and not self.shopper_updated and not self._late():
            return None
        if first and self.brief.named_only and self.brief.candidate_keys:
            missing = [
                p
                for p, _s in self.brief.candidate_keys
                if p not in self.ledger.products and p not in self.unreadable
            ]
            if missing and not self._late():
                return None
        if self._shopper_choice():
            carted = [c for c in good if (c["pid"], c["sku"]) in self.ledger.cart]
            if carted:
                return carted[0]
            scored = [c for c in good if self._priority_hits(c["pid"], c["sku"]) > 0]
            pool = list(scored)
            if not pool and (self._late() or self.turn >= self.max_steps - 8):
                pool = list(good)
            if not pool:
                return None
            pool.sort(key=self._preference_key)
            if self.ledger.event_target or self._late() or self.turn >= self.max_steps - 8:
                return pool[0]
            # Leave the first pick to evidence + the model; do not dump the cheapest.
            return None
        if first and self.brief.brand_is_preference and self.brief.brand and not self._recovery_mode():
            branded = [c for c in good if self._preferred_brand_hit(c["pid"])]
            if branded:
                good = branded
            elif not self._late() and self.turn < self.max_steps * 0.6:
                return None
        if first and self.brief.exact_first and not self.brief.rank_axes:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif not self._late() and self.turn < self.max_steps * 0.6:
                return None
        if (
            self.brief.need_two_listings
            and self.ledger.adds == 0
            and not self.ledger.cart
            and not self._late()
            and self.turn < self.max_steps - 8
        ):
            return None
        return good[0]

    def _answer_text(self) -> str:
        target = self._target() or (self._candidates() or [None])[0]
        if not target:
            return "I am still checking the options against your requirements; nothing is decided yet."
        pid, sku = target["pid"], target["sku"]
        parts = ["I am going with " + (self._title(pid) or pid)[:90]]
        if target.get("why"):
            parts.append(target["why"])
        if target.get("price") is not None:
            parts.append("it is %.2f %s and in stock" % (target["price"], self.brief.currency))
        return ", ".join(parts) + "."

    def _recall_need(self) -> int:
        if not self.brief.must_surface:
            return 0
        early = self.search_calls < 8 and self.turn < self.max_steps * 0.5
        return self.brief.must_surface + (2 if early else 0)

    def _late(self) -> bool:
        return self.turn >= self.max_steps - 5 or self.clock.closing(_FINISH_RESERVE_S)

    def _ready(self) -> bool:
        if self.ledger.adds or self._late() or self.ledger.event_target:
            return True
        if self.brief.wait_for_shopper and not self.shopper_updated and not self._late():
            return False
        if self.brief.need_two_listings and len(self.ledger.listing_ids) < 2 and self.turn < self.max_steps - 8:
            return False
        if self._shopper_choice():
            hits = [
                c
                for c in self._candidates()
                if c["ok"] and self._priority_hits(c["pid"], c["sku"]) > 0
            ]
            if not hits and self.turn < self.max_steps - 8:
                return False
            ok_n = len({c["pid"] for c in self._candidates() if c["ok"]})
            if self.turn < 10 and self.clock.remaining() > 170 and ok_n < 2:
                return False
            return True
        if self.brief.named_only and self.brief.candidate_keys:
            missing = [
                p
                for p, _s in self.brief.candidate_keys
                if p not in self.ledger.products and p not in self.unreadable
            ]
            return not missing
        ok_n = len({c["pid"] for c in self._candidates() if c["ok"]})
        if ok_n >= 1 and self.turn >= 8:
            return True
        roomy = self.turn < 10 and self.clock.remaining() > 170
        need = self._recall_need()
        if need and roomy and len(self.ledger.identities) < min(need, 3):
            return False
        if roomy and not self.reads_done and ok_n < 1:
            return False
        return True

    def _fresh(self, pid: str, sku: str) -> bool:
        k = Ledger.key(pid, sku)
        if k not in self.ledger.inspected:
            return False
        return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq

    def _unseen_leads(self, ignore_budget: bool = False) -> list:
        leads = []
        want_brand = _tokens(self.brief.brand)
        for key, row in self.ledger.rows.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.ledger.products or str(pid) in self.ledger.dead_products:
                continue
            if self.brief.named_only and not self._named_pair(str(pid), str(sku)):
                continue
            st = self.ledger.checked.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None:
                continue
            if not ignore_budget and self.budget is not None and price > self.budget + _EPS:
                continue
            title = str(row.get("title") or "")
            if self.brief.theme and not all(t in _tokens(title) for t in self.brief.theme):
                continue
            if want_brand and self._honor_brand_now() and not (
                want_brand <= _tokens(str(row.get("brand") or "")) or want_brand <= _tokens(title)
            ):
                continue
            leads.append((price, str(pid)))
        leads.sort()
        out = []
        for _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def _call_key(self, name: str, args: dict) -> str:
        return name + json.dumps(args, sort_keys=True)

    def _unused(self, name: str, args: dict) -> bool:
        return self._call_key(name, args) not in self.ledger.calls_seen

    def _open_leads(self, leads: list) -> tuple | None:
        if not leads or self.forced_compares >= 10:
            return None
        self.forced_compares += 1
        if len(leads) > 1 and "compare" in self.tool_names:
            args = {"product_ids": leads[:5]}
            if self._unused("compare", args):
                return ("compare", args)
        if "view" in self.tool_names:
            args = {"product_id": leads[0]}
            if self._unused("view", args):
                return ("view", args)
        return None

    def _twin_search(self) -> tuple | None:
        dead = self.ledger.event_target
        if self.searched_twins or not dead or "search" not in self.tool_names:
            return None
        if self.brief.named_only:
            return None
        if self._shopper_choice():
            query = " ".join(self._wish_query_terms()[:8]) or (self.brief.search_query or "")
            hint = (
                "The carted item is no longer valid. Search the shopper's own priority words "
                "and recover to a different listing whose facts still mention them."
            )
        else:
            query = " ".join(self.brief.prefer[:6]).strip()
            if not query:
                title = self._title(dead.split("::", 1)[0])
                query = " ".join(w for w in title.split() if any(ch.isdigit() for ch in w)) or " ".join(title.split()[:5])
            hint = (
                "The carted item is no longer valid. Search the shopper's own model or pairing words "
                "to recover with the smallest change the request allows."
            )
        if not query:
            return None
        args = {"query": query, "k": 10, "in_stock": True}
        if not self._unused("search", args):
            return None
        self.searched_twins = True
        self._note(hint)
        return ("search", args)

    def _budget_narrow(self) -> tuple | None:
        """When hits exist but all sit over budget, search/filter with the cap."""
        if self.budget is None or not self.ledger.rows:
            return None
        if "filter" in self.tool_names and self.brief.category:
            args = {"category": self.brief.category, "in_stock": True, "k": 10, "max_price": self.budget}
            if self.brief.brand and self._honor_brand_now():
                args["brand"] = self.brief.brand
            if self._unused("filter", args):
                return ("filter", args)
        if "search" in self.tool_names:
            query = self.brief.search_query or " ".join(
                self.brief.theme or ([self.brief.brand, self.brief.category] if self.brief.brand or self.brief.category else [])
            )
            if query:
                args = {"query": query, "k": 10, "in_stock": True}
                if self._unused("search", args):
                    return ("search", args)
        return None

    def _read_more(self) -> tuple | None:
        if self.brief.named_only and self.brief.candidate_keys and "compare" in self.tool_names:
            missing = [
                p for p, _s in self.brief.candidate_keys
                if p not in self.ledger.products and p not in self.unreadable
            ]
            if missing:
                return ("compare", {"product_ids": missing[:5]})
            if self.ledger.event_target:
                stale = [
                    (p, s)
                    for p, s in self.brief.candidate_keys
                    if Ledger.key(p, s) not in self.ledger.dead_keys
                    and p in self.ledger.products
                    and Ledger.key(p, s) not in self.ledger.inspected
                ]
                if stale:
                    p, s = stale[0]
                    return ("inspect_stock", {"product_id": p, "sku": s})
        twin = self._twin_search()
        if twin is not None:
            return twin
        opened = self._open_leads(self._unseen_leads())
        if opened is not None:
            return opened
        narrowed = self._budget_narrow()
        if narrowed is not None:
            return narrowed
        if self.brief.must_surface and not self._late() and len(self.ledger.identities) < self._recall_need():
            if self.ledger.products:
                return None
        if self._shopper_choice() and "search" in self.tool_names:
            terms = self._wish_query_terms()
            if terms:
                args = {"query": " ".join(terms[:8]), "k": 10, "in_stock": True}
                if self._unused("search", args):
                    return ("search", args)
        probe = (
            self.brief.category,
            self.brief.brand if self._honor_brand_now() else "",
            "%.2f" % (self.budget or 0.0),
            self.brief.search_query,
        )
        if probe in self._probes:
            self.reads_done = True
            return None
        self._probes.add(probe)
        if "filter" in self.tool_names and self.brief.category:
            args = {"category": self.brief.category, "in_stock": True, "k": 10}
            if self.budget is not None:
                args["max_price"] = self.budget
            if self.brief.brand and self._honor_brand_now():
                args["brand"] = self.brief.brand
            if self._unused("filter", args):
                return ("filter", args)
        if "search" in self.tool_names:
            query = self.brief.search_query or " ".join(
                w for w in (self.brief.brand, self.brief.category, " ".join(self.brief.theme)) if w
            )
            if query:
                args = {"query": query, "in_stock": True, "k": 10}
                if self._unused("search", args):
                    return ("search", args)
        self.reads_done = True
        return None

    def _unstick(self) -> tuple | None:
        """Leave a rejected-search loop by viewing hits or changing the query."""
        opened = self._open_leads(self._unseen_leads())
        if opened is not None:
            return opened
        narrowed = self._budget_narrow()
        if narrowed is not None:
            return narrowed
        more = self._read_more()
        if more is not None:
            return more
        return self._open_leads(self._unseen_leads(ignore_budget=True))

    def _spoken_claim_text(self) -> str:
        return self._shopper_ask_text()

    def _overlapping_fact_claim(self, pid: str, sku: str) -> tuple | None:
        """One labelled fact that restates the shopper's own spec — never a pile of extras."""
        skip = {
            "price", "in_stock", "brand", "category", "condition", "navigation",
            "model", "warranty_type", "warranty_duration",
        }
        spoken_text = self._spoken_claim_text()
        spoken_toks = _tokens(spoken_text)
        spoken_nums = {n for n, _neigh in _numbers_from_text(spoken_text)}
        spoken_nums |= set(self._shopper_spec_numbers())
        if self.budget is not None:
            spoken_nums = {n for n in spoken_nums if abs(n - self.budget) >= 0.05}
        allowed_toks: set = set()
        for name in self.claim_schema.get("allowed") or []:
            allowed_toks |= _field_tokens(str(name))
        best = None
        for key, raw in self._facts(pid).items():
            if str(key).lower() in skip:
                continue
            kl = str(key).lower()
            if kl not in {"size", "display"} and not _token_overlap(_field_tokens(kl), allowed_toks):
                continue
            text = str(raw).strip()
            if not text:
                continue
            score = _token_overlap(_tokens(text), spoken_toks) + 2 * len(
                {n for n, _neigh in _numbers_from_text(text)} & spoken_nums
            )
            if score > 0 and (best is None or score > best[0]):
                best = (score, str(key), text.split(",")[0].strip())
        if not best:
            return None
        _score, key, value = best
        value = self._with_shopper_unit(value)
        for name in list(self.brief.claim_fields) + [str(a) for a in (self.claim_schema.get("allowed") or [])]:
            if _token_overlap(_field_tokens(name), _field_tokens(key)):
                return (name, value)
        return (key, value)

    def _spec_claim_fields(self, pid: str | None = None, sku: str | None = None) -> list:
        """Allowed field names the shopper actually asked to justify — never extras."""
        skip = {"price", "in_stock", "brand", "category"}
        allowed_map = {str(a).lower(): str(a) for a in (self.claim_schema.get("allowed") or [])}
        named = []
        seen: set = set()
        for field in self.brief.claim_fields:
            fl = str(field).strip()
            if not fl or fl.lower() in skip or fl.lower() in seen:
                continue
            if fl.lower() in {"display", "size", "wattage", "condition", "navigation"}:
                continue
            seen.add(fl.lower())
            named.append(allowed_map.get(fl.lower(), fl))
        if named:
            return named[:2]
        spoken_text = self._spoken_claim_text()
        spoken_toks = _tokens(spoken_text)
        spoken_nums = {n for n, _neigh in _numbers_from_text(spoken_text)}
        spoken_nums |= set(self._shopper_spec_numbers())
        if self.budget is not None:
            spoken_nums = {n for n in spoken_nums if abs(n - self.budget) >= 0.05}
        candidates = [str(a) for a in (self.claim_schema.get("allowed") or []) if str(a).lower() not in skip]
        for tok in _fields_from_description(self.claim_schema.get("description") or ""):
            if tok not in skip and tok not in {c.lower() for c in candidates}:
                candidates.append(tok)
        scored: list = []
        for field in candidates:
            value = self._value_for_claim_field(pid or "", sku or "", field) if pid else None
            if not value:
                continue
            score = _token_overlap(_tokens(value), spoken_toks) + len(
                {n for n, _neigh in _numbers_from_text(value)} & spoken_nums
            )
            if score > 0:
                scored.append((score, field))
        scored.sort(key=lambda row: row[0], reverse=True)
        return [field for _score, field in scored][:1]

    def _value_for_claim_field(self, pid: str, sku: str, field: str) -> str | None:
        judged = self.verdicts.get(Ledger.key(pid, sku)) or {}
        for claim in judged.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            if str(claim.get("field") or "").lower() != field.lower():
                continue
            value = self._with_shopper_unit(str(claim.get("value") or "").strip())
            if value and self._grounded(pid, sku, field, value):
                return value
        want = _field_tokens(field)
        spoken_text = self._spoken_claim_text()
        spoken = _hint_tokens(spoken_text)
        spoken_toks = _tokens(spoken_text)
        spoken_nums = {n for n, _neigh in _numbers_from_text(spoken_text)}
        spoken_nums |= set(self._shopper_spec_numbers())
        if self.budget is not None:
            spoken_nums = {n for n in spoken_nums if abs(n - self.budget) >= 0.05}
        best_val, best_score = None, 0
        for key, raw in self._facts(pid).items():
            text = str(raw).strip()
            if not text:
                continue
            spoken_hit = _token_overlap(_tokens(text), spoken_toks) + len(
                {n for n, _neigh in _numbers_from_text(text)} & spoken_nums
            )
            if not spoken_hit and not _token_overlap(_field_tokens(str(key)), want):
                continue
            if not spoken_hit:
                continue
            score = spoken_hit + _token_overlap(_field_tokens(str(key)), want)
            if score > best_score:
                best_val, best_score = text.split(",")[0].strip(), score
        if best_val:
            best_val = self._with_shopper_unit(best_val)
            if self._grounded(pid, sku, field, best_val):
                return best_val
        return None

    def _claims_for(self, pid: str, sku: str) -> list:
        """Only the gated spec the shopper asked for, plus live price, stock, and brand."""
        out: list = []
        seen: set = set()

        def add(field: Any, value: Any, force: bool = False) -> None:
            field_s, value_s = str(field or "").strip(), str(value or "").strip()
            if not field_s or not value_s:
                return
            if not force and not self._claim_allowed(field_s, pid):
                return
            if not self._grounded(pid, sku, field_s, value_s):
                return
            key = (field_s.lower(), value_s.lower())
            if key in seen:
                return
            seen.add(key)
            out.append({"field": field_s, "value": value_s})

        for field in self._spec_claim_fields(pid, sku):
            value = self._value_for_claim_field(pid, sku, field)
            if value:
                add(field, value, force=True)
        if not any(str(c.get("field") or "").lower() not in {"price", "in_stock", "brand", "category"} for c in out):
            fact_claim = self._overlapping_fact_claim(pid, sku)
            if fact_claim:
                add(fact_claim[0], fact_claim[1], force=True)
        st = self.ledger.status(pid, sku) or {}
        if self.budget is not None:
            add("price", "at most %.2f" % self.budget)
        elif st.get("price") is not None:
            add("price", "%.2f" % st["price"])
        if st.get("in_stock") or (pid, sku) in self.ledger.cart:
            add("in_stock", "true")
        brand = self._brand_of(pid)
        if brand:
            add("brand", brand)
        return out

    def _want_justification(self) -> bool:
        return bool(self.brief.need_reason or self.claim_schema.get("required"))

    def _order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self._want_justification():
            claims = self._claims_for(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def _valid_now(self, pid: str, sku: str) -> tuple:
        st = self.ledger.status(pid, sku)
        k = Ledger.key(pid, sku)
        if st is None:
            return False, "inspect_stock the product_id and sku first"
        if (pid, sku) not in self.ledger.cart:
            return False, "add that variant to the cart first"
        if self.ledger.last_add_seq is not None and self.ledger.checked_seq.get(k, 0) <= self.ledger.last_add_seq:
            return False, "re-check the variant with inspect_stock after the cart addition"
        if not st.get("in_stock"):
            return False, "that variant is out of stock; choose a different product"
        price = st.get("price")
        if self.budget is not None and price is not None and price > self.budget + _EPS:
            return False, "that variant costs %.2f, above the budget of %.2f" % (price, self.budget)
        if k in self.ledger.dead_keys:
            return False, "that variant was invalidated; choose a different product"
        return True, ""

    def _shortlist_note(self) -> None:
        if self.clock.closing(80):
            return
        self._judge_products()
        if self._shopper_choice() and not self.ledger.cart:
            good = [c for c in self._candidates() if c["ok"]]
            good.sort(key=self._preference_key)
            if good:
                self._note(
                    "Listings that already meet stock and budget: "
                    + "; ".join("%s::%s at %.2f" % (c["pid"], c["sku"], c["price"] or 0.0) for c in good[:6])
                    + ". First-cart the one whose own facts mention the most of the shopper's stated "
                    "priorities, not the cheapest."
                )
                return
        best = self._target()
        if best:
            self._note(
                "A candidate now looks fully supported: %s at %s. %s The controller will inspect, "
                "cart and order once the usual stock check is fresh; keep reading only if something "
                "could rank better."
                % (best["key"], best["price"], best.get("why") or "")
            )
            return
        blocked = [c for c in self._candidates() if c["viewed"] and c["live"] and not c["ok"]]
        if blocked:
            sample = "; ".join((c["key"] + " " + (c.get("why") or "not yet supported"))[:80] for c in blocked[:3])
            self._note("Read products are not yet a safe buy: " + sample + ". Look at different products.")

    def _budget_from_message(self, text: str) -> None:
        new = _budget_in(text)
        if new is None or (self.budget is not None and new >= self.budget):
            return
        self.budget = new
        self.shopper_updated = True
        self._note("Budget updated by the shopper: at most %.2f %s." % (new, self.brief.currency))
        for pid, sku in list(self.ledger.cart):
            st = self.ledger.status(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _EPS:
                self.ledger.dead_keys.add(Ledger.key(pid, sku))
                self._note("The carted item is now above budget and will be replaced.")

    def _live_cart(self) -> tuple | None:
        for pid, sku in reversed(self.ledger.cart):
            key = Ledger.key(pid, sku)
            if key in self.ledger.dead_keys:
                continue
            if self.brief.named_only and not self._named_pair(str(pid), str(sku)):
                continue
            st = self.ledger.status(pid, sku) or {}
            if not st.get("in_stock"):
                continue
            price = st.get("price")
            if self.budget is not None and price is not None and price > self.budget + _EPS:
                continue
            return (str(pid), str(sku))
        return None

    def _plan(self) -> tuple | None:
        if self.done:
            return None
        if self.shopper_question_pending and "message" in self.tool_names:
            if self.turn <= self.max_steps - 4:
                self.shopper_question_pending = False
                self.question_waits = 0
                return ("message", {"content": self._answer_text()})
            self.shopper_question_pending = False
            self.question_waits = 0
        live = self._live_cart()
        if live is not None:
            pid, sku = live
            if not self._fresh(pid, sku):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            return ("place_test_order", self._order_args(pid, sku))
        for cpid, csku in self.ledger.cart:
            if Ledger.key(cpid, csku) in self.ledger.dead_keys and "remove_from_cart" in self.tool_names:
                return ("remove_from_cart", {"product_id": cpid, "sku": csku})
        if self.brief.named_only and self.brief.candidate_keys and "compare" in self.tool_names:
            missing = [
                p for p, _s in self.brief.candidate_keys
                if p not in self.ledger.products and p not in self.unreadable
            ]
            if missing:
                return ("compare", {"product_ids": missing[:5]})
        target = self._target()
        if target is None or not self._ready():
            forced = self._read_more()
            if forced is not None:
                return forced
            if target is None:
                if self.ledger.event_target and (
                    self._late() or self.turn >= self.max_steps - 10 or self.search_calls >= 8
                ):
                    fallback = self._last_resort_pool()
                    if fallback:
                        pid, sku = fallback[0]["pid"], fallback[0]["sku"]
                        if not self._fresh(pid, sku):
                            return ("inspect_stock", {"product_id": pid, "sku": sku})
                        return ("add_to_cart", {"product_id": pid, "sku": sku})
                return None
        if not self._ready():
            return None
        pid, sku = target["pid"], target["sku"]
        if not self._fresh(pid, sku):
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return ("add_to_cart", {"product_id": pid, "sku": sku})

    def _ask_model(self) -> dict | None:
        if self.clock.closing(_FINISH_RESERVE_S + 5):
            return None
        for attempt in range(2):
            if self.clock.closing(_FINISH_RESERVE_S + 5):
                return None
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            payload = {
                "model": model,
                "messages": self.messages,
                "tools": self.tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            resp = _PROXY.post("/inference/chat/completions", json_data=payload)
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                self._note("No tool call was made. Choose exactly one of the supplied tools now.")
                continue
            if self.clock.remaining() > 100:
                _backoff(attempt)
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    def _send(self, envelope: dict) -> dict | None:
        for attempt in range(2):
            result = _PROXY.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.clock.remaining() < 12:
                break
            _backoff(attempt)
        return None

    def _act(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        self._mark_call(name, args)
        turn = self.turn + 1
        group = self.problem_id + "-turn-" + str(turn)
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [{"call_id": group + "-1", "action": {"name": name, "args": args}}],
        }
        result = self._send(envelope)
        # Always consume the turn number. Replaying the same idempotency key
        # after a 409/timeout leaves the agent stuck until the step clock dies.
        self.turn = turn
        self.pending_envelope = None
        return self._absorb(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.ledger.calls_seen.discard(name + json.dumps(args, sort_keys=True))
            self.messages.append(
                {"role": "tool", "tool_call_id": call_id,
                 "content": json.dumps({"error": "environment call failed; it will be retried automatically"})}
            )
            return None
        calls = result.get("calls") or []
        wrap = (calls[0].get("observation") if calls and isinstance(calls[0], dict) else None) or {}
        obs = wrap.get("observation") if isinstance(wrap.get("observation"), dict) else wrap
        if isinstance(obs, dict) and wrap.get("error") and not obs.get("error"):
            obs = dict(obs)
            obs["error"] = wrap.get("error")
        if wrap.get("done"):
            self.done = True
        self.ledger.record(name, args, obs, self.brief)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _trim(json.dumps(obs), _OBS_CHARS)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self._budget_from_message(text)
            self.brief.spoken = (self.brief.spoken + " " + text).strip()[:2500]
            self.shopper_updated = True
            self.brief.wait_for_shopper = False
            if not self.clock.closing(80):
                self._judge_request()
                self._judge_products(force=True)
            if _QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self._note("The shopper asked a question: answer it with message in your next turn, then continue.")
        if name in ("view", "compare") and not self.clock.closing(80):
            self._judge_products()
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        return obs

    def _post_add_recheck(self) -> None:
        pid, sku = self.pending_recheck
        self.pending_recheck = None
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            self.ledger.dead_keys.add(Ledger.key(pid, sku))
            if not self.done:
                self._recover_now()
            return
        stock = bool(obs.get("in_stock"))
        price = _num(obs.get("price"))
        over = self.budget is not None and price is not None and price > self.budget + _EPS
        if stock and not over:
            self._note(
                "Re-check after the cart addition: %s is in stock at %s. Ordering it now."
                % (Ledger.key(pid, sku), price)
            )
            if not self.done and self.turn < self.max_steps:
                self._act("place_test_order", self._order_args(pid, sku))
            return
        self.ledger.event_target = Ledger.key(pid, sku)
        self.ledger.dead_keys.add(Ledger.key(pid, sku))
        if not self.brief.named_only:
            self.ledger.dead_products.add(str(pid))
        self.forced_compares = 0
        self.reads_done = False
        self.searched_twins = False
        self._probes.clear()
        self.verdicts.pop(Ledger.key(pid, sku), None)
        reason = "went out of stock" if not stock else "was repriced above budget"
        if self.brief.named_only:
            self._note(
                "Market change: the carted item %s %s. Re-check and re-rank only the named "
                "candidate variants; do not leave that list." % (Ledger.key(pid, sku), reason)
            )
        else:
            self._note(
                "Market change: the carted item %s %s. Recover using only the shopper request — "
                "same model, next acceptable option, or another listing that still fits."
                % (Ledger.key(pid, sku), reason)
            )
        if not self.clock.closing(80):
            self._judge_products(force=True)
            self._shortlist_note()
        if not self.done:
            self._recover_now()

    def _recover_now(self) -> None:
        """After a carted item dies, answer the shopper if needed and order a replacement."""
        if self.done or self.turn >= self.max_steps:
            return
        if self.shopper_question_pending and "message" in self.tool_names and self.turn <= self.max_steps - 4:
            self.shopper_question_pending = False
            self.question_waits = 0
            self._act("message", {"content": self._answer_text()})
            if self.done:
                return
        # Named ranking already compared its list — re-order immediately.
        # Preference/recovery still need another listing or a sibling color.
        if self._target() is not None:
            self._finish_fast()
            return
        if self.brief.named_only or self._late() or self.turn >= self.max_steps - 8:
            self._finish_fast()

    def _finish_fast(self) -> None:
        """Spend remaining steps on a terminal order, not more catalog reading."""
        room = self.max_steps - self.turn
        live = self._live_cart()
        if live is not None:
            pid, sku = live
            if not self._fresh(pid, sku) and room >= 2:
                self._act("inspect_stock", {"product_id": pid, "sku": sku})
                if self.done:
                    return
            if not self.done and self.turn < self.max_steps:
                self._act("place_test_order", self._order_args(pid, sku))
            return
        if self.done or self.turn >= self.max_steps:
            return
        good = self._order_pool(require_ok=True)
        if not good and self.brief.named_only:
            good = self._order_pool(require_ok=False)
        if not good:
            opened = self._open_leads(self._unseen_leads())
            if opened is not None and self.turn < self.max_steps - 2:
                self._act(opened[0], opened[1])
                if not self.clock.closing(80):
                    self._judge_products()
                good = self._order_pool(require_ok=True)
                if not good and self.brief.named_only:
                    good = self._order_pool(require_ok=False)
        if not good:
            good = self._last_resort_pool()
        for cand in good[:4]:
            if self.done or self.turn >= self.max_steps:
                return
            pid, sku = cand["pid"], cand["sku"]
            if (pid, sku) not in self.ledger.cart and self.turn < self.max_steps:
                obs = self._act("add_to_cart", {"product_id": pid, "sku": sku})
                if self.done:
                    return
                if not isinstance(obs, dict) or obs.get("error"):
                    self.ledger.dead_keys.add(Ledger.key(pid, sku))
                    continue
            if self.turn < self.max_steps:
                obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
                if self.done:
                    return
                if not isinstance(obs, dict) or not obs.get("in_stock"):
                    self.ledger.dead_keys.add(Ledger.key(pid, sku))
                    if not self.brief.named_only:
                        self.ledger.dead_products.add(str(pid))
                    continue
                price = _num(obs.get("price"))
                if self.budget is not None and price is not None and price > self.budget + _EPS:
                    self.ledger.dead_keys.add(Ledger.key(pid, sku))
                    continue
            if not self.done and self.turn < self.max_steps:
                self._act("place_test_order", self._order_args(pid, sku))
            return
        if self.done or self.turn >= self.max_steps:
            return
        leftover = self._last_resort_pool()
        for cand in leftover[:3]:
            if self.done or self.turn >= self.max_steps:
                return
            pid, sku = cand["pid"], cand["sku"]
            if (pid, sku) not in self.ledger.cart and self.turn < self.max_steps:
                obs = self._act("add_to_cart", {"product_id": pid, "sku": sku})
                if self.done:
                    return
                if not isinstance(obs, dict) or obs.get("error"):
                    self.ledger.dead_keys.add(Ledger.key(pid, sku))
                    continue
            if self.turn < self.max_steps:
                obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
                if self.done:
                    return
                if not isinstance(obs, dict) or not obs.get("in_stock"):
                    continue
                price = _num(obs.get("price"))
                if self.budget is not None and price is not None and price > self.budget + _EPS:
                    continue
            if not self.done and self.turn < self.max_steps:
                self._act("place_test_order", self._order_args(pid, sku))
            return

    def run(self) -> list:
        if not self.clock.closing(80):
            self._judge_request()
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            if self.clock.closing(_FINISH_RESERVE_S) or self.turn >= self.max_steps - 6:
                self._finish_fast()
                break
            if self.ledger.event_target and not self._live_cart() and self.turn >= self.max_steps - 10:
                self._finish_fast()
                break
            if self.brief.ask and not self.asked_shopper and "message" in self.tool_names:
                self.asked_shopper = True
                self._act("message", {"content": self.brief.ask})
                continue
            if self.pending_recheck and "inspect_stock" in self.tool_names:
                self._post_add_recheck()
                continue
            plan = self._plan()
            if plan is None:
                if self.ledger.event_target and (
                    self._late() or self.search_calls >= 8 or self.turn >= self.max_steps - 10
                ):
                    self._finish_fast()
                    break
                plan = self._unstick()
            if plan is not None:
                name, args = plan
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    if name in ("inspect_stock", "add_to_cart"):
                        self.ledger.dead_keys.add(Ledger.key(args.get("product_id"), args.get("sku")))
                    if name == "compare":
                        self.unreadable.update(str(p) for p in args.get("product_ids") or [])
                    if name == "message":
                        self.shopper_question_pending = False
                    last_forced, repeats = None, 0
                    self._note("That step changed nothing, so the controller moved on.")
                    continue
                obs = self._act(name, args)
                if name in ("search", "filter"):
                    self.search_calls += 1
                if name == "message":
                    self.shopper_question_pending = False
                    self.question_waits = 0
                if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                    added = obs.get("added") or {}
                    if added.get("product_id"):
                        self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                continue
            last_forced, repeats = None, 0
            if self.clock.closing(_FINISH_RESERVE_S):
                self._finish_fast()
                break
            proposal = self._ask_model()
            if proposal is None:
                stuck = self._unstick()
                if stuck is not None:
                    self._act(stuck[0], stuck[1])
                    continue
                self._finish_fast()
                break
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = _arguments(call)
            content = proposal.get("content") or ""
            reprompts = 0
            while True:
                verdict = self._validate(name, args)
                if verdict is None:
                    break
                if reprompts >= _MAX_REPROMPTS:
                    self.rejected_streak += 1
                    self._note("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if self.rejected_streak >= 3:
                        self._shortlist_note()
                        self.rejected_streak = 0
                    verdict = "skip"
                    break
                reprompts += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": verdict})}
                )
                self.dialogue.append({"role": "assistant", "content": "[rejected " + name + "] " + verdict[:200]})
                again = self._ask_model()
                if again is None:
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = _arguments(call)
                content = again.get("content") or ""
            if verdict == "skip":
                stuck = self._unstick()
                if stuck is not None:
                    name, args = stuck
                    obs = self._act(name, args)
                    if name in ("search", "filter"):
                        self.search_calls += 1
                    if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                        added = obs.get("added") or {}
                        if added.get("product_id"):
                            self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                continue
            self.rejected_streak = 0
            if name not in self.tool_names:
                continue
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.ledger.cart:
                    pid, sku = self.ledger.cart[-1]
                args = self._order_args(pid, sku)
            if name == "message":
                if not self.shopper_question_pending:
                    self.questions_asked += 1
                self.shopper_question_pending = False
                self.question_waits = 0
            obs = self._act(name, args, content, call.get("id"))
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") or {}
                if added.get("product_id"):
                    self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
            if name in ("search", "filter"):
                self.search_calls += 1
                need = self._recall_need()
                if need:
                    have = len(self.ledger.identities)
                    self._note(
                        "Distinct matching products surfaced so far: %d of %d." % (have, need)
                        + ("" if have >= need else " Keep searching with different wording.")
                    )
            if name == "inspect_cart":
                self.cart_inspections += 1
            if name in ("view", "compare") and not self.clock.closing(80):
                self._shortlist_note()
        return self.dialogue

    def _validate(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.ledger.cart:
                pid, sku = self.ledger.cart[-1]
            ok, why = self._valid_now(pid, sku)
            if not ok:
                return "order refused: " + why
            need = self._recall_need()
            if need and not self._late() and len(self.ledger.identities) < need:
                return (
                    "order refused: only %d distinct matching products have been surfaced; search until "
                    "at least %d appear" % (len(self.ledger.identities), need)
                )
            if self.brief.need_two_listings and not self._late() and len(self.ledger.listing_ids) < 2:
                return "order refused: compare at least two products from different listings first"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            if Ledger.key(pid, sku) not in self.verdicts:
                self._judge_products()
            c = self._assess(pid, sku)
            key = Ledger.key(pid, sku)
            if not c["live"]:
                return "that variant is out of stock, over budget, or already invalidated"
            if not c["viewed"]:
                return "view or compare that product first so its raw facts are on record"
            if key not in self.ledger.inspected:
                return "inspect_stock that variant first: current price and stock decide whether it can be bought"
            if not c["ok"]:
                return "that product is not yet supported by the evidence for this request"
            if not (self.brief.named_only and self.brief.rank_axes):
                if not self._measurement_grounded(pid, sku):
                    return (
                        "that card does not state the shopper's own measurement as a labelled value "
                        "or variant option"
                    )
                if not self._kind_hold(pid):
                    return "that listing does not carry the same kind of labelled facts as a matching device"
            if self.brief.need_two_listings and not self._late() and len(self.ledger.listing_ids) < 2:
                return "compare two products from different listings before the first cart addition"
            if self._shopper_choice() and not self._late():
                hits = self._priority_hits(pid, sku)
                pool = [row for row in self._candidates() if row["ok"] and row["live"]]
                best_hits = max((self._priority_hits(row["pid"], row["sku"]) for row in pool), default=0)
                if best_hits > hits:
                    return (
                        "another viewed listing mentions more of the shopper's own priority words; "
                        "cart that one instead of a weaker match"
                    )
            if self.brief.exact_first and not self.ledger.adds and c["relax"] and not self._late():
                return "the first cart item must be the exact pairing the shopper named; keep searching"
            if self.brief.named_only and self.brief.candidate_keys and not self._named_pair(pid, sku):
                return "stay on the named product_id::sku pairs the shopper listed"
            if self.brief.named_only and not self.ledger.adds:
                best = self._target()
                if best and Ledger.key(best["pid"], best["sku"]) != key and not self._late():
                    return "the first cart item must be the current best named pairing, not a placeholder"
            if not self._ready():
                return (
                    "hold: read one more product that also meets the request first, so there is a "
                    "fallback when the market moves after the cart addition"
                )
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.shopper_question_pending and "?" not in text and not self.brief.ask:
                return "do not send status updates; use message only to answer the shopper or to ask one question"
            if not self.shopper_question_pending and self.questions_asked >= 2:
                return "the shopper has already answered; decide from the evidence rather than asking again"
        if name == "inspect_cart" and self.cart_inspections >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.ledger.calls_seen and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        return None

    def _mark_call(self, name: str, args: dict) -> None:
        if name in ("search", "filter", "view", "compare"):
            self.ledger.calls_seen.add(name + json.dumps(args, sort_keys=True))


def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
