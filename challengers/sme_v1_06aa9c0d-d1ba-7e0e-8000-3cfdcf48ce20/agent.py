"""General shopping agent.

The public request is compiled by the model. Cards are judged by the
model against that request. Code owns only the environment protocol:
turns, tools, live price and stock, and commit or recover.
"""
from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient


_PROXY = ProxyClient(timeout=70, max_retries=1, retry_delay=0.4)
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
_BUDGET_S = 250.0
_RESERVE_S = 60.0
_EPS = 1e-9

_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|"
    r"max(?:imum)? of|within|budget (?:is|of)?|only go up to|can spend up to|"
    r"can only go up to|at most|my limit is)\s*" + _NUM,
    re.I,
)
_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTE_RE = re.compile(r"\"([^\"]{2,80})\"")
_FROM_TO_RE = re.compile(r"from\s*" + _NUM + r"\s*(?:[A-Za-z]{3}|₱)?\s*to\s*" + _NUM, re.I)
_SPAN_RE = re.compile(
    r"\b(\d+(?:\.\d+)?-?[A-Za-z]{1,12}|\d+(?:\.\d+)?\s+[A-Za-z]{1,12})\b"
)
_CODE_RE = re.compile(r"\b([A-Za-z]{1,6}\d{2,}[A-Za-z0-9]*)\b")
_WITH_RE = re.compile(
    r"\b(?:with|has|have|that has)\s+((?:[A-Za-z]{2,}[ \-]?){1,4})"
    r"(?=\s*(?:,|\.|at or|under|up to|budget|my |in stock|;|$))",
    re.I,
)
_AXIS_RE = re.compile(
    r"\b(highest|largest|most|lowest|smallest|cheapest)\s+"
    r"([a-z][a-z ]{1,40}?)(?=\s*(?:wins|first|second|third|then|,|\.|;|$))",
    re.I,
)
_DIM_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_TRADEOFF_RE = re.compile(
    r"\b(i'?d rather|rather pay|cheapest isn'?t|don'?t just grab the cheapest|"
    r"which of your preferences|weigh the others)\b",
    re.I,
)
_REASON_RE = re.compile(
    r"\b(written reason|justify|justification|tell me why|explain why|reason for|"
    r"walk me through|give me a reason|reason i can trust|why the one you pick|"
    r"why your choice)\b",
    re.I,
)
_LEAN_RE = re.compile(
    r"\b(?:i lean toward|lean toward)\s+([a-z0-9]{2,20})\b|"
    r"\b([a-z0-9]{2,20})\s+would be my pick\b",
    re.I,
)
_PAIR_NEAR_RE = re.compile(
    r"\b(?:the\s+)?([a-z]{3,12})\s+version of\b|"
    r"\bideally in\s+([a-z]{3,12})\b|"
    r"\bprefer(?:s|red)?(?: the)?\s+([a-z]{3,12})\s+(?=[A-Za-z]{1,6}\d)",
    re.I,
)
_CLASS_LEAD_RE = re.compile(
    r"\b(?:looking for|need|want|get me|find me|buy)\s+"
    r"(?:an?\s+|the\s+)?(?:in[-\s]?stock\s+)?"
    r"([a-z0-9][a-z0-9.&\- ]{1,40}?)"
    r"(?=\s*(?:,|with\b|that has\b|at or|up to|under|in stock|my limit|and it has))",
    re.I,
)
_MEASURE = {
    "hd",
    "fhd",
    "uhd",
    "inch",
    "inches",
    "hz",
    "ghz",
    "mhz",
    "gb",
    "tb",
    "mb",
    "ram",
    "p",
    "full",
}
_FIELD_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_AXIS_STOP = {
    "highest",
    "largest",
    "most",
    "lowest",
    "smallest",
    "cheapest",
    "the",
    "a",
    "an",
    "of",
    "then",
    "wins",
    "first",
    "second",
    "third",
    "and",
    "or",
}
_WISH_STOP = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "find",
    "need",
    "want",
    "under",
    "over",
    "stock",
    "order",
    "please",
    "would",
    "like",
    "have",
    "from",
    "just",
    "also",
    "only",
    "then",
    "than",
    "more",
    "rather",
    "into",
}
_MONEY = {
    "under",
    "below",
    "budget",
    "limit",
    "spend",
    "price",
    "at",
    "or",
    "up",
    "to",
    "max",
    "than",
    "only",
    "go",
    "can",
    "my",
    "is",
    "of",
    "most",
    "within",
}
_LIVE_NEED = {"stock", "in", "available", "availability", "php", "peso", "pesos"}
_ITEM_NOISE = {
    "something",
    "listing",
    "item",
    "items",
    "product",
    "products",
    "one",
    "stock",
}

_EXTRACT_SYS = (
    "You compile a shopping request into JSON. Use only words the shopper "
    "said. Do not invent attributes, brands, or values. Do not assume a "
    "product family. Return JSON only."
)
_JUDGE_SYS = (
    "You judge whether one catalog card satisfies a shopper request. "
    "Use labelled facts and the selected option text before the title. "
    "If those layers disagree, the requirement is not proven. "
    "Preferences only rank cards that already pass every firm need. "
    "Return JSON only."
)
_CHOOSE_SYS = (
    "You pick one catalog card for this shopper. "
    "A firm measurement must appear on labelled facts or the selected option. "
    "If this is recovery, prefer the same model or listing in another color. "
    "Pick only from the cards you were given. Return JSON only."
)
_CLAIM_SYS = (
    "You write short order justification claims. "
    "Each field must be an allowed name or a labelled fact key on this card. "
    "Each value must appear on this card. Do not invent. Return JSON only."
)
_REWRITE_SYS = (
    "You write the next catalog search query. "
    "Use at most eight of the shopper's own words: class, maker, model code, "
    "color, and measured values. Do not write a sentence. Return JSON only."
)
_REFINE_SYS = (
    "You map the shopper's firm measurements onto labelled fact keys seen "
    "on catalog cards. Use only keys that appear on the cards. "
    "Do not invent values. Return JSON only."
)


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        found = re.search(r"(\d+(?:\.\d+)?)", text)
        return float(found.group(1)) if found else None


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _fold_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for tok in _tokens(text):
        if len(tok) <= 1:
            continue
        out.add(tok)
        glued = re.match(r"^(\d+(?:\.\d+)?)([a-z]{1,12})$", tok)
        if glued:
            out.add(glued.group(1))
            unit = glued.group(2)
            if len(unit) > 1:
                out.add(unit)
    return out


def _parse_json(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {}
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if fence:
        raw = fence.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _budget_in(text: str) -> float | None:
    best = None
    for match in _BUDGET_RE.finditer(text or ""):
        whole = match.group(1).replace(",", "")
        frac = match.group(2) or "0"
        try:
            value = float(whole + "." + frac)
        except ValueError:
            continue
        if best is None or value < best:
            best = value
    return best


def _tool_name(tool: dict) -> str:
    return str((tool.get("function") or {}).get("name") or "")


def _field_tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[_\-\s]+", (name or "").lower()) if t and t not in _AXIS_STOP}


def _token_overlap(a: set[str], b: set[str]) -> int:
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


_AXIS_UNITS = {
    "refresh": {"hz", "refresh", "rate"},
    "rate": {"hz", "refresh", "rate"},
    "ram": {"gb", "ram", "memory", "ddr"},
    "memory": {"gb", "ram", "memory", "ddr"},
    "resolution": {"p", "px", "k", "res", "resolution", "display", "fhd", "uhd", "qhd", "hd"},
    "display": {"p", "px", "k", "display", "resolution", "fhd", "uhd", "qhd", "hd"},
    "screen": {"in", "inch", "inches", "screen", "size"},
    "size": {"in", "inch", "inches", "size", "screen"},
}


def _hint_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _AXIS_STOP and not t.isdigit() and len(t) > 1}


def _axis_units(hint: str) -> set[str]:
    toks = _hint_tokens(hint)
    extra: set[str] = set()
    for tok in toks:
        extra |= _AXIS_UNITS.get(tok, set())
    return toks | extra


def _numbers_near(text: str) -> list[tuple[float, set[str]]]:
    out: list[tuple[float, set[str]]] = []
    raw = text or ""
    for match in re.finditer(r"(?<![A-Za-z0-9])(\d{1,5}(?:\.\d+)?)(?!\d)", raw):
        val = _num(match.group(1))
        if val is None:
            continue
        window = raw[max(0, match.start() - 24) : min(len(raw), match.end() + 24)]
        neigh = _tokens(window)
        suffix = re.match(r"([A-Za-z]{1,8})", raw[match.end() :])
        if suffix:
            neigh.add(suffix.group(1).lower())
        out.append((val, neigh))
    return out


def _fields_from_description(text: str) -> list[str]:
    skip = {
        "only",
        "field",
        "fields",
        "value",
        "facts",
        "from",
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "optional",
        "string",
        "object",
        "array",
        "true",
        "false",
        "justification",
        "claims",
    }
    found: list[str] = []
    seen: set[str] = set()
    for tok in _FIELD_SNAKE_RE.findall((text or "").lower()):
        if tok in skip or tok in seen:
            continue
        seen.add(tok)
        found.append(tok)
    return found[:16]


def _claim_schema(tools: list) -> dict:
    schema = {"present": False, "allowed": [], "required": False, "description": ""}
    for tool in tools:
        if _tool_name(tool) != "place_test_order":
            continue
        props = ((tool.get("function") or {}).get("parameters") or {}).get("properties") or {}
        just = props.get("justification") or {}
        claims = ((just.get("properties") or {}).get("claims") or {}).get("items") or {}
        field = ((claims.get("properties") or {}).get("field") or {})
        allowed = [str(x) for x in (field.get("enum") or []) if str(x).strip()]
        desc = " ".join(
            str(part)
            for part in (just.get("description"), field.get("description"), (tool.get("function") or {}).get("description"))
            if part
        )
        if not allowed:
            allowed = _fields_from_description(desc)
        schema.update(
            {
                "present": True,
                "allowed": allowed,
                "required": "justification"
                in ((tool.get("function") or {}).get("parameters") or {}).get("required", []),
                "description": desc[:900],
            }
        )
    return schema


def _is_live_need(text: str, budget: float | None) -> bool:
    nums = _numbers(text)
    leftover = {
        t
        for t in _fold_tokens(text)
        if not re.fullmatch(r"\d+(?:\.\d+)?", t) and t not in _MONEY
    }
    leftover -= _LIVE_NEED
    if leftover:
        return False
    if not nums:
        return True
    if budget is None:
        return False
    return all(abs(n - budget) < 0.05 or n == 0 for n in nums)


def _spoken_ok(text: str, spoken: set[str]) -> bool:
    toks = {t for t in _fold_tokens(text) if len(t) > 2 and t not in _ITEM_NOISE}
    have = set(spoken)
    for tok in spoken:
        have |= _fold_tokens(tok)
        have.add(_stem_word(tok))
    extra = {t for t in toks if t not in have and _stem_word(t) not in have}
    return not extra


def _numbers(text: str) -> list[float]:
    found = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)", text or ""):
        found.append(float(match.group(1)))
    return found


_QUERY_TAIL = re.compile(
    r"\bbudget\b|\bcategory\s*:|\bfind something\b|\bbuy one\b|"
    r"\bto be in stock\b|\bin stock\b|\bgenuine\b|\bnot a\b|"
    r"\bstrongly prefer\b|\bproper\b|\bideally\b|\bmatches\b|"
    r"\bat or under\b|\bat most\b|\bup to\b|\bunder\b|\blimit is\b|"
    r"\bcombination\b|\bunavailable\b|\bnot\b",
    re.I,
)
_QUERY_DROP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "with",
    "that",
    "this",
    "listing",
    "item",
    "something",
    "please",
    "should",
    "also",
    "then",
    "than",
    "just",
    "only",
    "from",
    "into",
    "your",
    "my",
    "one",
    "strongly",
    "prefer",
    "accessory",
    "genuine",
    "php",
    "peso",
    "pesos",
    "buy",
    "looking",
    "get",
}


def _class_phrase(text: str) -> str:
    positive = re.sub(r"\bnot\s+(?:a\s+|an\s+)?[^.;,]{2,48}", " ", (text or "").lower())
    match = _CLASS_LEAD_RE.search(positive)
    if not match:
        return ""
    raw = re.sub(r"\s+", " ", match.group(1)).strip(" ,.")
    raw = re.sub(r"\b(?:in[-\s]?stock|available)\b", " ", raw)
    words = [
        w
        for w in re.findall(r"[a-z0-9]+", raw)
        if w not in _QUERY_DROP
        and w not in _ITEM_NOISE
        and w not in _MEASURE
        and not w.isdigit()
        and not re.match(r"^\d", w)
    ]
    if words and words[0] in {"in", "to", "on"}:
        words = words[1:]
    if not words or len(words) > 4:
        return ""
    return " ".join(words)[:40]


def _clean_query(text: str) -> str:
    cut = _QUERY_TAIL.split(text or "", maxsplit=1)[0]
    seen: set[str] = set()
    words: list[str] = []
    for word in cut.split():
        key = word.lower().strip(".,;:()")
        if key in _QUERY_DROP or key in seen:
            continue
        if re.fullmatch(r"\d+\.\d{2}", key):
            continue
        seen.add(key)
        words.append(word)
        if len(words) >= 8:
            break
    return " ".join(words)[:80]


def _stem_word(tok: str) -> str:
    text = (tok or "").lower()
    if text.endswith("ies") and len(text) > 4:
        return text[:-3] + "y"
    if text.endswith("es") and text[:-2] in _MEASURE:
        return text[:-2]
    if text.endswith("s") and len(text) > 3:
        stem = text[:-1]
        if stem in _MEASURE:
            return stem
        return stem
    return text


def _measure_units(tokens: set[str]) -> set[str]:
    out: set[str] = set()
    for tok in tokens:
        stem = _stem_word(tok)
        if tok in _MEASURE or stem in _MEASURE:
            out.add(tok)
            out.add(stem)
    return out


def _sku_stem(sku: str) -> str:
    text = str(sku or "").strip()
    return text.split("-")[0] if text else ""


def _firm_need(text: str) -> bool:
    piece = (text or "").strip()
    if not piece:
        return False
    if _numbers(piece):
        return True
    toks = _fold_tokens(piece)
    return bool(toks) and toks <= _MEASURE


# --------------------------------------------------------------------------- spec / memory


class Spec:
    def __init__(self, query: str) -> None:
        goal, _, rules = (query or "").partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.strip()
        self.spoken = self.goal
        self.budget = _budget_in(self.goal)
        self.hard: list[dict] = []
        self.soft: list[dict] = []
        self.universe = "open"
        self.closed: list[tuple[str, str]] = []
        self.phrase: list[str] = []
        self.category = ""
        self.item = ""
        self.survey = 0
        self.need_reason = False
        self.ask = ""
        self.axes: list[dict] = []
        self.search_query = ""
        self.soft_brand = ""
        self.exact_first = False
        self.prefer_not_cheap = bool(_TRADEOFF_RE.search(self.goal))
        self.codes: list[str] = []
        self.near: list[str] = []
        self.claim_fields: list[str] = []
        self.fact_hints: list[str] = []
        self.locked = False
        self._seed()

    def _seed(self) -> None:
        self.closed = [(pid, sku.rstrip(".,;:")) for pid, sku in _KEY_RE.findall(self.goal)]
        if self.closed:
            self.universe = "closed"
        quoted = _QUOTE_RE.findall(self.goal)
        self.phrase = re.findall(r"[a-z0-9]+", (quoted[0] if quoted else "").lower())
        if self.phrase and self.universe != "closed":
            self.universe = "phrase"
        if _REASON_RE.search(self.goal):
            self.need_reason = True
        lean = _LEAN_RE.search(self.goal)
        if lean:
            self.soft_brand = (lean.group(1) or lean.group(2) or "").strip()
        for span in _SPAN_RE.findall(self.goal):
            value = _num(span)
            if value is not None and self.budget is not None and abs(value - self.budget) < _EPS:
                continue
            text = span.strip()
            if text and not any(h.get("text") == text for h in self.hard):
                self.hard.append({"text": text[:40]})
        for match in _WITH_RE.finditer(self.goal):
            piece = " ".join(match.group(1).split())
            if piece and not any(str(h.get("text") or "").lower() == piece.lower() for h in self.hard):
                self.hard.append({"text": piece[:40]})
        seen = set()
        for match in _AXIS_RE.finditer(self.goal):
            hint = match.group(2).strip()
            if not hint or hint in seen:
                continue
            seen.add(hint)
            word = match.group(1).lower()
            direction = "min" if word in {"lowest", "smallest", "cheapest"} else "max"
            self.axes.append({"from": hint[:80], "direction": direction})
        if self.axes and not any("price" in a["from"].lower() for a in self.axes):
            if re.search(r"\bprice\b", self.goal, re.I):
                self.axes.append({"from": "price", "direction": "min"})
        self.codes = [m.group(1) for m in _CODE_RE.finditer(self.goal)]
        near: list[str] = []
        skip_near = _WISH_STOP | {
            "version",
            "should",
            "vanish",
            "keep",
            "prefer",
            "another",
            "different",
            "model",
            "color",
            "same",
            "listing",
            "from",
            "into",
            "strongly",
            "one",
            "ear",
            "earphones",
            "earbuds",
            "over",
            "buy",
            "looking",
        }
        for match in _PAIR_NEAR_RE.finditer(self.goal):
            tok = next((g for g in match.groups() if g), "")
            tok = (tok or "").strip()
            if tok and tok.lower() not in skip_near and tok not in near and tok not in self.codes:
                near.append(tok)
        for match in _CODE_RE.finditer(self.goal):
            prevs = [
                t
                for t in re.findall(r"[A-Za-z]{3,}", self.goal[max(0, match.start() - 48) : match.start()])
                if t.lower() not in skip_near
            ]
            for tok in prevs[-2:]:
                if tok not in near and tok not in self.codes:
                    near.append(tok)
        self.near = near[:4]
        spoken = self.goal.lower()
        self.item = _class_phrase(spoken) or self.item
        if self.item and not self.category:
            joined = re.search(
                rf"((?:[a-z0-9]+\s*&\s*|[a-z0-9]+\s+and\s+)+[a-z0-9]+)\s+{re.escape(self.item)}",
                spoken,
            )
            if joined:
                self.category = f"{joined.group(1)} {self.item}"[:80]
        self.search_query = self._fallback_query()

    def _fallback_query(self) -> str:
        parts = [
            self.item,
            " ".join(self.phrase),
            " ".join(self.codes),
            " ".join(self.near),
            self.soft_brand,
        ]
        parts.extend(str(h.get("text") or "") for h in self.hard)
        text = " ".join(p for p in parts if p).strip()
        if text:
            return _clean_query(text)
        lead = re.split(r"[.!?]", self.goal, maxsplit=1)[0]
        return _clean_query(" ".join(lead.split()[:12]))

    def apply_model(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        if self.locked:
            ask = data.get("ask")
            if isinstance(ask, str) and ask.strip():
                self.ask = ask.strip()[:200]
            if data.get("need_reason") and _REASON_RE.search(self.goal):
                self.need_reason = True
            self._take_fields(data)
            return
        spoken = _tokens(self.spoken + " " + self.goal)
        for key in ("category", "item", "ask", "search_query"):
            val = data.get(key)
            if isinstance(val, str) and val.strip() and _spoken_ok(val, spoken | _tokens(self.rules)):
                if key == "search_query":
                    cleaned = _clean_query(val)
                    short = len(cleaned.split()) <= 8
                    named = any(c.lower() in cleaned.lower() for c in self.codes)
                    if cleaned and (short or named):
                        self.search_query = cleaned
                else:
                    setattr(self, key, val.strip()[:200])
        hard = data.get("hard")
        if isinstance(hard, list):
            cleaned = []
            for item in hard:
                if isinstance(item, str):
                    item = {"text": item}
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or item.get("need") or "").strip()
                if not text or not _spoken_ok(text, spoken):
                    continue
                if _is_live_need(text, self.budget):
                    continue
                if not _firm_need(text):
                    if not any(str(s.get("text") or "").lower() == text.lower() for s in self.soft):
                        self.soft.append({"text": text[:80], "role": "pref"})
                    continue
                cleaned.append({"text": text[:80]})
            if cleaned:
                have = {str(h.get("text") or "").lower() for h in self.hard}
                for row in cleaned:
                    if row["text"].lower() not in have:
                        self.hard.append(row)
                        have.add(row["text"].lower())
                self.hard = self.hard[:12]
        soft = data.get("soft")
        if isinstance(soft, list):
            self.soft = []
            for item in soft:
                if isinstance(item, str):
                    item = {"text": item}
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if not text or not _spoken_ok(text, spoken):
                    continue
                role = str(item.get("role") or "pref")[:24]
                self.soft.append({"text": text[:80], "role": role})
                if role == "brand" and not self.soft_brand:
                    self.soft_brand = text
        brand = data.get("soft_brand")
        if isinstance(brand, str) and brand.strip() and _spoken_ok(brand, spoken) and not self.soft_brand:
            self.soft_brand = brand.strip()
            if not any(s.get("role") == "brand" for s in self.soft):
                self.soft.append({"text": self.soft_brand, "role": "brand"})
        if self.soft_brand and not any(s.get("role") == "brand" for s in self.soft):
            self.soft.append({"text": self.soft_brand, "role": "brand"})
        if isinstance(data.get("survey"), int):
            self.survey = max(0, min(8, int(data["survey"])))
        if data.get("need_reason") and _REASON_RE.search(self.goal):
            self.need_reason = True
        if data.get("exact_first"):
            self.exact_first = True
        axes = data.get("rank_axes")
        if isinstance(axes, list):
            parsed = []
            for ax in axes:
                if not isinstance(ax, dict):
                    continue
                hint = str(ax.get("from") or ax.get("need") or "").strip()
                if not hint:
                    continue
                direction = str(ax.get("direction") or "max").lower()
                if direction not in {"max", "min"}:
                    direction = "min" if any(w in hint.lower() for w in ("low", "small", "cheap")) else "max"
                parsed.append({"from": hint[:80], "direction": direction})
            if parsed and not self.axes:
                self.axes = parsed[:4]
        if self.search_query:
            self.search_query = _clean_query(self.search_query)
        if not self.search_query:
            self.search_query = self._fallback_query()
        if data.get("prefer_not_cheap"):
            self.prefer_not_cheap = True
        self._take_fields(data)
        if (self.hard or self.item or self.axes) and not self.ask:
            self.locked = True

    def _take_fields(self, data: dict) -> None:
        for key, dest in (
            ("claim_fields", self.claim_fields),
            ("fact_keys", self.fact_hints),
            ("fact_hints", self.fact_hints),
        ):
            raw = data.get(key)
            if not isinstance(raw, list):
                continue
            for item in raw:
                text = str(item or "").strip()
                if text and text not in dest:
                    dest.append(text[:40])
        self.claim_fields = self.claim_fields[:8]
        self.fact_hints = self.fact_hints[:8]

    def revise(self, text: str) -> None:
        self.spoken = (self.spoken + " " + (text or "")).strip()[:2500]
        cut = _budget_in(text or "")
        if cut is not None:
            self.budget = cut if self.budget is None else min(self.budget, cut)
        self.ask = ""
        self.locked = False
        for span in _SPAN_RE.findall(text or ""):
            value = _num(span)
            if value is not None and self.budget is not None and abs(value - self.budget) < _EPS:
                continue
            piece = span.strip()
            if piece and not any(h.get("text") == piece for h in self.hard):
                self.hard.append({"text": piece[:40]})
        seen = {a["from"] for a in self.axes}
        for match in _AXIS_RE.finditer(text or ""):
            hint = match.group(2).strip()
            if not hint or hint in seen:
                continue
            seen.add(hint)
            word = match.group(1).lower()
            direction = "min" if word in {"lowest", "smallest", "cheapest"} else "max"
            self.axes.append({"from": hint[:80], "direction": direction})


class Memory:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.products: dict[str, dict] = {}
        self.options: dict[str, str] = {}
        self.live: dict[str, dict] = {}
        self.live_seq: dict[str, int] = {}
        self.inspected: set[str] = set()
        self.cart: list[tuple[str, str]] = []
        self.dead: set[str] = set()
        self.dead_products: set[str] = set()
        self.identities: set[str] = set()
        self.listings: set[str] = set()
        self.listing_of: dict[str, str] = {}
        self.recent_ids: list[str] = []
        self.lead_titles: dict[str, str] = {}
        self.lead_rank: dict[str, int] = {}
        self.seq = 0
        self.adds = 0
        self.last_add_seq: int | None = None
        self.event_key: str | None = None
        self.event_kind: str | None = None

    @staticmethod
    def key(pid: Any, sku: Any) -> str:
        return f"{pid}::{sku or ''}"

    def record(self, name: str, args: dict, obs: dict, spec: Spec) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        if name in {"search", "filter"}:
            if name == "search":
                self.recent_ids = []
            for i, row in enumerate(obs.get("results") or []):
                if isinstance(row, dict):
                    self._row(row, spec, rank=i if name == "search" else None)
        elif name == "view":
            self._product(obs, spec)
        elif name == "compare":
            for prod in obs.get("compared") or []:
                if isinstance(prod, dict):
                    self._product(prod, spec)
        elif name == "inspect_stock":
            self._live(obs.get("product_id"), obs.get("sku"), obs.get("in_stock"), obs.get("price"))
            self.inspected.add(self.key(obs.get("product_id"), obs.get("sku")))
        elif name == "add_to_cart":
            added = obs.get("added") or {}
            pid, sku = added.get("product_id"), added.get("sku")
            if pid and sku:
                self.adds += 1
                pair = (str(pid), str(sku))
                if pair not in self.cart:
                    self.cart.append(pair)
                self.last_add_seq = self.seq
            event = obs.get("market_event")
            if isinstance(event, dict):
                self._event(event)
            self._cart(obs)
        elif name in {"remove_from_cart", "inspect_cart"}:
            self._cart(obs)

    def _row(self, row: dict, spec: Spec, rank: int | None = None) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        if str(pid) not in self.recent_ids:
            self.recent_ids.append(str(pid))
        if rank is not None:
            prev = self.lead_rank.get(str(pid), 99)
            self.lead_rank[str(pid)] = min(prev, int(rank))
        title = str(row.get("title") or "")
        if title:
            self.lead_titles[str(pid)] = title
        if self._theme_hit(title, spec):
            self.identities.add(str(pid))
        sku = row.get("sku")
        if not sku:
            return
        key = self.key(pid, sku)
        self.rows[key] = row
        if row.get("price") is not None:
            self.live.setdefault(key, {"in_stock": bool(row.get("in_stock", True)), "price": _num(row.get("price"))})

    def _product(self, prod: dict, spec: Spec) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.products[str(pid)] = prod
        if prod.get("listing_id"):
            lid = str(prod["listing_id"])
            self.listings.add(lid)
            self.listing_of[str(pid)] = lid
        if self._theme_hit(str(prod.get("title") or ""), spec):
            self.identities.add(str(pid))
        for variant in prod.get("variants") or []:
            if not isinstance(variant, dict) or not variant.get("sku"):
                continue
            key = self.key(pid, variant.get("sku"))
            self.options[key] = str(variant.get("options") or "")
            self._live(pid, variant.get("sku"), variant.get("in_stock"), variant.get("price"))

    def _live(self, pid: Any, sku: Any, stock: Any, price: Any) -> None:
        if not pid or not sku:
            return
        key = self.key(pid, sku)
        prev = self.live.get(key, {})
        self.live[key] = {
            "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
            "price": _num(price) if price is not None else prev.get("price"),
        }
        self.live_seq[key] = self.seq

    def _cart(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [
            (str(r.get("product_id")), str(r.get("sku")))
            for r in rows
            if isinstance(r, dict) and r.get("product_id")
        ]
        if pairs or rows == []:
            self.cart = pairs
        for row in rows:
            if isinstance(row, dict) and row.get("product_id"):
                self._live(
                    row.get("product_id"),
                    row.get("sku"),
                    row.get("current_in_stock", row.get("in_stock")),
                    row.get("current_price"),
                )

    def _event(self, event: dict) -> None:
        target = event.get("target") or {}
        key = self.key(target.get("product_id"), target.get("sku"))
        self.event_key = key
        self.event_kind = str(event.get("kind") or "")
        if event.get("kind") == "stockout" or event.get("in_stock_after") is False:
            self.dead.add(key)
            self._live(target.get("product_id"), target.get("sku"), False, event.get("new_price"))
        if event.get("kind") == "price_change" and event.get("new_price") is not None:
            self._live(
                target.get("product_id"),
                target.get("sku"),
                event.get("in_stock_after", True),
                event.get("new_price"),
            )

    def _theme_hit(self, title: str, spec: Spec) -> bool:
        if not spec.phrase:
            return True
        have = _tokens(title)
        return all(tok in have for tok in spec.phrase)

    def status(self, pid: str, sku: str) -> dict | None:
        return self.live.get(self.key(pid, sku))


# --------------------------------------------------------------------------- prover


class Prover:
    def __init__(self, spec: Spec, memory: Memory, consult: Callable[[str, dict], dict]) -> None:
        self.spec = spec
        self.memory = memory
        self.consult = consult
        self.cache: dict[str, dict] = {}

    def facts(self, pid: str) -> dict:
        raw = (self.memory.products.get(str(pid)) or {}).get("facts")
        return raw if isinstance(raw, dict) else {}

    def product(self, pid: str) -> dict:
        return self.memory.products.get(str(pid)) or {}

    def title(self, pid: str) -> str:
        title = str(self.product(pid).get("title") or "")
        if title:
            return title
        for key, row in self.memory.rows.items():
            if key.startswith(str(pid) + "::"):
                return str(row.get("title") or "")
        return ""

    def brand(self, pid: str) -> str:
        brand = self.product(pid).get("brand")
        if brand:
            return str(brand)
        for key, row in self.memory.rows.items():
            if key.startswith(str(pid) + "::") and row.get("brand"):
                return str(row["brand"])
        return ""

    def viewed(self, pid: str) -> bool:
        return str(pid) in self.memory.products

    def card(self, pid: str, sku: str) -> dict:
        facts = {}
        for key, val in list(self.facts(pid).items())[:36]:
            facts[str(key)] = str(val)[:240]
        live = self.memory.status(pid, sku) or {}
        return {
            "product_id": str(pid),
            "sku": str(sku),
            "title": self.title(pid)[:180],
            "brand": self.brand(pid),
            "category_path": self.product(pid).get("category_path") or [],
            "facts": facts,
            "options": self.memory.options.get(Memory.key(pid, sku), "")[:240],
            "price": live.get("price"),
            "in_stock": live.get("in_stock"),
        }

    def layers(self, pid: str, sku: str) -> str:
        options = self.memory.options.get(Memory.key(pid, sku), "")
        facts = " ".join(f"{k} {v}" for k, v in self.facts(pid).items())
        return f"{options} {facts}"

    def judge_new(self) -> None:
        pending = []
        for pid in self.memory.products:
            for key in self.memory.options:
                if not key.startswith(str(pid) + "::"):
                    continue
                if key in self.cache:
                    continue
                _, _, sku = key.partition("::")
                live = self.memory.status(pid, sku) or {}
                if live.get("in_stock") is False:
                    continue
                if self.spec.budget is not None and live.get("price") is not None and live["price"] > self.spec.budget + _EPS:
                    continue
                pending.append((str(pid), sku))
        if not pending:
            return
        rounds = 0
        limit = 2
        while pending and rounds < limit:
            batch = pending[:8]
            pending = pending[8:]
            rounds += 1
            data = self.consult(
                _JUDGE_SYS,
                {
                    "request": self.spec.goal[:1600],
                    "hard": [h.get("text") for h in self.spec.hard],
                    "soft": self.spec.soft,
                    "item": self.spec.item,
                    "category": self.spec.category,
                    "axes": self.spec.axes,
                    "cards": [self.card(pid, sku) for pid, sku in batch],
                    "need": {
                        "verdicts": [
                            {
                                "product_id": "id",
                                "sku": "sku",
                                "ok": "true only if every firm need is proven on this card",
                                "relax": "0 if every preference is present, else how many are missing",
                                "scores": "numeric values for each rank axis, from this card only",
                            }
                        ]
                    },
                },
            )
            rows = data.get("verdicts") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                rows = []
            by_key = {}
            for row in rows:
                if isinstance(row, dict) and row.get("product_id") and row.get("sku"):
                    by_key[Memory.key(row["product_id"], row["sku"])] = row
            for pid, sku in batch:
                key = Memory.key(pid, sku)
                row = by_key.get(key)
                if isinstance(row, dict):
                    self.cache[key] = {
                        "ok": bool(row.get("ok")),
                        "relax": int(row.get("relax") or 0),
                        "scores": row.get("scores") if isinstance(row.get("scores"), dict) else {},
                    }
                else:
                    self.cache[key] = self._fallback(pid, sku)

    def _need_holds(self, pid: str, sku: str, text: str) -> bool:
        status = self._need_status(pid, sku, text)
        if _numbers(text) and not _is_live_need(text, self.spec.budget):
            return status == "match"
        return status != "conflict"

    def _need_status(self, pid: str, sku: str, text: str) -> str:
        if _is_live_need(text, self.spec.budget):
            return "match"
        primary = self.layers(pid, sku)
        title = self.title(pid)
        need = _fold_tokens(text)
        nums = _numbers(text)
        leftover = {t for t in need if not re.fullmatch(r"\d+(?:\.\d+)?", t) and t not in _MONEY}
        leftover -= _LIVE_NEED
        glued = {t for t in leftover if re.match(r"^\d+[a-z]+$", t)}
        units = _measure_units({t for t in leftover if len(t) >= 2} - glued)
        if units & {"inch", "inches"}:
            units |= {"inch", "inches", "in", "screen", "size", "display"}
        if nums:
            matched = False
            conflict = False

            def around_of(neigh: set[str]) -> set[str]:
                around = set(neigh)
                for tok in list(neigh):
                    around |= _fold_tokens(tok)
                    around.add(_stem_word(tok))
                return around

            for val_n, neigh in _numbers_near(primary):
                around = around_of(neigh)
                if units and not (units & around):
                    continue
                if any(abs(val_n - n) < 1e-6 for n in nums):
                    matched = True
                else:
                    conflict = True
            if matched:
                return "match"
            if conflict:
                return "conflict"
            if not self.spec.need_reason and title:
                for val_n, neigh in _numbers_near(title):
                    around = around_of(neigh)
                    if units and not (units & around):
                        continue
                    if any(abs(val_n - n) < 1e-6 for n in nums):
                        return "match"
            return "unknown"
        source = primary.strip() or title
        have = _fold_tokens(source)
        if {"full", "hd"} & leftover and ({"fhd", "1080", "1080p"} & have or "1080" in source.lower()):
            leftover -= {"full", "hd"}
        if leftover and not (leftover & have):
            return "unknown"
        return "match"

    def code_ok(self, pid: str, sku: str) -> bool:
        if self.spec.item or self.spec.category:
            want = {
                t
                for t in _fold_tokens(self.spec.item + " " + self.spec.category)
                if len(t) > 2 and t not in _ITEM_NOISE
            }
            path = " ".join(str(x) for x in (self.product(pid).get("category_path") or []))
            typed = f"{path} {self.layers(pid, sku)} {self.brand(pid)}"
            if not path.strip():
                typed = f"{typed} {self.title(pid)}"
            have = _fold_tokens(typed)
            have |= {t[:-1] for t in have if t.endswith("s") and len(t) > 3}
            want |= {t[:-1] for t in want if t.endswith("s") and len(t) > 3}
            if want and not (want & have):
                return False
        for hard in self.spec.hard:
            piece = str(hard.get("text") or "")
            if piece and not self._need_holds(pid, sku, piece):
                return False
        return True

    def _fallback(self, pid: str, sku: str) -> dict:
        blob = f"{self.layers(pid, sku)} {self.title(pid)} {self.brand(pid)}"
        relax = 0
        for pref in self.spec.soft:
            text = str(pref.get("text") or "")
            if text and text.lower() not in blob.lower():
                relax += 1
        scores = {axis["from"]: self._axis_number(pid, sku, axis) for axis in self.spec.axes}
        return {"ok": self.code_ok(pid, sku), "relax": relax, "scores": scores}

    def _axis_number(self, pid: str, sku: str, axis: dict) -> float:
        hint_text = str(axis.get("from") or "")
        want = _axis_units(hint_text)
        live = self.memory.status(pid, sku) or {}
        if "price" in want or "cheap" in hint_text.lower() or "cost" in hint_text.lower():
            return live.get("price") if live.get("price") is not None else 1e18
        facts = self.facts(pid)
        best_key, best_score = None, 0
        for key, val in facts.items():
            score = _token_overlap(_field_tokens(str(key)), want) or _token_overlap(_tokens(str(val)), want)
            if score > best_score:
                best_key, best_score = str(key), score
        direction = str(axis.get("direction") or "max")
        res_axis = any(w in hint_text.lower() for w in ("resolution", "display")) and not any(
            w in hint_text.lower() for w in ("size", "screen", "inch")
        )

        def _alias_res(text: str, nums: list[float]) -> None:
            if not res_axis:
                return
            low = (text or "").lower()
            if re.search(r"\b4k\b|\buhd\b", low):
                nums.append(2160.0)
            if re.search(r"\bqhd\b|\b1440p\b", low):
                nums.append(1440.0)
            if re.search(r"\bfhd\b|\b1080p\b|\bfull\s*hd\b", low):
                nums.append(1080.0)

        def pick(text: str, require_overlap: bool) -> float | None:
            nums: list[float] = []
            for val, neigh in _numbers_near(text or ""):
                if require_overlap and want and not (neigh & want):
                    continue
                nums.append(val)
            for a, b in _DIM_RE.findall(text or ""):
                if res_axis:
                    nums.append(float(a) * float(b))
            _alias_res(text, nums)
            if not nums:
                return None
            return max(nums) if direction != "min" else min(nums)

        options = self.memory.options.get(Memory.key(pid, sku), "")
        fact_n = pick(str(facts.get(best_key) or ""), False) if best_key else None
        opt_n = pick(options, True) if options else None
        if opt_n is not None:
            return opt_n
        if options and fact_n is not None:
            overrides = []
            for val, _neigh in _numbers_near(options):
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
        title_n = pick(self.title(pid), True)
        if title_n is not None:
            return title_n
        return -1.0 if direction != "min" else 1e18

    def grounded(self, pid: str, sku: str, field: str, value: str) -> bool:
        field_l = str(field or "").strip().lower()
        value_s = str(value or "").strip()
        if not field_l or not value_s:
            return False
        live = self.memory.status(pid, sku) or {}
        if field_l == "price":
            if value_s.lower().startswith("at most"):
                found = re.search(r"[\d,.]+", value_s[7:])
                cap = _num(found.group(0) if found else None)
                return self.spec.budget is not None and cap is not None and abs(cap - self.spec.budget) < 0.05
            price = live.get("price")
            observed = _num(value_s)
            return price is not None and observed is not None and abs(price - observed) < 0.05
        if field_l == "in_stock":
            want = value_s.lower() in {"true", "1", "yes", "in stock"}
            return bool(live.get("in_stock")) == want
        if field_l == "brand":
            brand = self.brand(pid)
            return bool(brand) and value_s.lower() in brand.lower()
        if field_l == "category":
            path = self.product(pid).get("category_path") or []
            blob = " ".join(str(x) for x in path).lower()
            return bool(blob) and value_s.lower() in blob
        facts = self.facts(pid)
        value_l = value_s.lower()
        for key, fact in facts.items():
            fact_s = str(fact).lower()
            if str(key).lower() == field_l and (value_l in fact_s or fact_s in value_l):
                return True
        options = self.memory.options.get(Memory.key(pid, sku), "")
        blob = f"{self.layers(pid, sku)} {options}".lower()
        if not self.spec.need_reason:
            blob = f"{blob} {self.title(pid).lower()}"
        if value_l in blob:
            return True
        want_n = _num(value_s)
        if want_n is None:
            return False
        want_toks = _field_tokens(field_l)
        for key, fact in facts.items():
            if want_toks and not (
                _token_overlap(want_toks, _field_tokens(str(key))) or _token_overlap(want_toks, _tokens(str(fact)))
            ):
                continue
            if any(abs(n - want_n) < 0.05 for n, _neigh in _numbers_near(str(fact))):
                return True
        return False

    def priority_hits(self, pid: str, sku: str) -> int:
        wants = {
            w
            for w in _tokens(self.spec.spoken + " " + self.spec.goal)
            if len(w) > 3 and not w.isdigit() and w not in _WISH_STOP
        }
        wants -= _tokens(self.spec.category + " " + self.spec.item)
        if not wants:
            return 0
        have = _tokens(
            f"{self.layers(pid, sku)} {self.title(pid)} {self.brand(pid)} {self.memory.options.get(Memory.key(pid, sku), '')}"
        )
        return len(wants & have)

    def identity(self, pid: str, sku: str) -> int:
        blob = (
            f"{self.title(pid)} {self.brand(pid)} "
            f"{self.memory.options.get(Memory.key(pid, sku), '')} {self.layers(pid, sku)}"
        ).lower()
        score = 0
        for code in self.spec.codes:
            if code.lower() in blob:
                score += 3
        for tok in self.spec.near:
            if tok.lower() in blob:
                score += 2
        for tok in self.spec.phrase:
            if tok and tok in blob:
                score += 1
        for hard in self.spec.hard:
            piece = str(hard.get("text") or "").lower()
            if piece and piece in blob:
                score += 2
        return score

    def kin(self, pid: str, sku: str) -> int:
        if not self.memory.event_key:
            return 0
        dead_pid = self.memory.event_key.split("::", 1)[0]
        bits = {t for t in _tokens(self.title(dead_pid)) if any(ch.isdigit() for ch in t)}
        bits |= {t for t in _tokens(self.brand(dead_pid)) if len(t) > 2}
        bits |= {c.lower() for c in self.spec.codes}
        have = _tokens(f"{self.title(pid)} {self.brand(pid)} {self.layers(pid, sku)}")
        digit = {t for t in bits if any(ch.isdigit() for ch in t)}
        return 3 * len(digit & have) + len((bits - digit) & have)

    def prove(self, pid: str, sku: str) -> dict:
        key = Memory.key(pid, sku)
        live = self.memory.status(pid, sku) or {}
        if key in self.memory.dead:
            return {"ok": False, "live": False, "why": "dead", "score": (), "relax": 9, "pid": str(pid), "sku": str(sku), "price": live.get("price"), "key": key}
        if self.spec.universe == "closed" and (str(pid), str(sku)) not in self.spec.closed:
            return {"ok": False, "live": False, "why": "outside", "score": (), "relax": 9, "pid": str(pid), "sku": str(sku), "price": live.get("price"), "key": key}
        if self.spec.phrase and not self.memory._theme_hit(self.title(pid), self.spec):
            return {"ok": False, "live": True, "why": "phrase", "score": (), "relax": 9, "pid": str(pid), "sku": str(sku), "price": live.get("price"), "key": key, "viewed": self.viewed(pid)}
        price = live.get("price")
        stock = live.get("in_stock")
        live_ok = stock is not False
        if self.spec.budget is not None and price is not None and price > self.spec.budget + _EPS:
            live_ok = False
        if not self.viewed(pid):
            return {"ok": False, "live": live_ok, "why": "unviewed", "score": (), "relax": 9, "pid": str(pid), "sku": str(sku), "price": price, "key": key, "viewed": False}
        if key not in self.cache:
            self.judge_new()
        judged = self.cache.get(key) or self._fallback(pid, sku)
        score = []
        for axis in self.spec.axes:
            value = None
            if isinstance(judged.get("scores"), dict):
                value = _num(judged["scores"].get(axis["from"]))
            if value is None:
                value = self._axis_number(pid, sku, axis)
            score.append(-value if axis.get("direction") == "max" else value)
        if not self.spec.prefer_not_cheap:
            score.append(price if price is not None else 1e18)
        if self.spec.universe == "phrase":
            measured = True
        elif self.spec.hard:
            measured = self.code_ok(pid, sku)
        elif self.spec.item or self.spec.category:
            measured = self.code_ok(pid, sku)
        else:
            measured = bool(judged.get("ok")) or live_ok
        return {
            "ok": measured and live_ok,
            "live": live_ok,
            "viewed": True,
            "why": "ok" if measured else "spec",
            "score": tuple(score),
            "relax": int(judged.get("relax") or 0),
            "hits": self.priority_hits(pid, sku),
            "kin": self.kin(pid, sku),
            "ident": self.identity(pid, sku),
            "pid": str(pid),
            "sku": str(sku),
            "price": price,
            "key": key,
        }

    def dominated(self, row: dict, pool: list[dict]) -> bool:
        if not self.spec.axes:
            return False
        for other in pool:
            if other["key"] == row["key"]:
                continue
            better = False
            worse = False
            for i, axis in enumerate(self.spec.axes):
                a = row["score"][i] if i < len(row["score"]) else 0
                b = other["score"][i] if i < len(other["score"]) else 0
                if b + _EPS < a:
                    better = True
                elif a + _EPS < b:
                    worse = True
            if better and not worse:
                return True
        return False


# --------------------------------------------------------------------------- planner


class Planner:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.names = {_tool_name(t) for t in self.tools if _tool_name(t)}
        self.schema = _claim_schema(self.tools)
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.spec = Spec(str(self.policy.get("query") or ""))
        self.memory = Memory()
        self.prover = Prover(self.spec, self.memory, self._consult)
        self.start = time.monotonic()
        self.turn = 0
        self.done = False
        self.dialogue: list = []
        self.pending_question = False
        self.asked = False
        self.pending_recheck: tuple[str, str] | None = None
        self.searches = 0
        self.queries_used: list[str] = []
        self._seen_calls: set[str] = set()
        self.shelf = ""
        self.page_reads = 0
        self.searched_twins = False
        self.model_index = 0
        self.llm_calls = 0
        self._choice_fp = ""
        self._choice_row: dict | None = None
        self._refined = False
        self._asked_query = False
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)
        if self.schema.get("required"):
            self.spec.need_reason = True

    def remaining(self) -> float:
        return _BUDGET_S - (time.monotonic() - self.start)

    def late(self) -> bool:
        return self.turn >= self.max_steps - 5 or self.remaining() < _RESERVE_S

    def _consult(self, system: str, payload: dict, floor: float = 35.0) -> dict:
        if self.remaining() < floor or self.llm_calls >= 12:
            return {}
        self.llm_calls += 1
        body = json.dumps(payload, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": body[:12000]},
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
            if self.remaining() > 90:
                time.sleep(min(0.4 * (attempt + 1), 1.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return {}

    def interpret(self) -> None:
        data = self._consult(
            _EXTRACT_SYS,
            {
                "request": self.spec.goal[:2500],
                "rules": self.spec.rules[:900],
                "need": {
                    "hard": "short firm measurements only, each {text}. Long wishes are not hard",
                    "soft": "preferences and exclusions only, each {text, role}",
                    "soft_brand": "preferred maker if the shopper marked it as a preference, else empty",
                    "category": "the shopper's shelf or class phrase, copied exactly",
                    "item": "product class in the shopper's words",
                    "search_query": "at most eight catalog words: class, maker, model code, and measured values only",
                    "survey": "how many distinct matching products to see before the first commit, else 0",
                    "need_reason": "true if the shopper asked for a written reason",
                    "ask": "one question if a firm need is still unknown and the rules say to ask, else empty",
                    "exact_first": "true if the first commit must be an exact named pairing",
                    "prefer_not_cheap": "true if the shopper said not to grab the cheapest",
                    "rank_axes": [{"direction": "max or min", "from": "the shopper's comparison phrase"}],
                    "claim_fields": "allowed justification field names the shopper's specs need, besides price and stock",
                    "fact_keys": "labelled fact keys that would prove the firm measurements",
                },
            },
        )
        self.spec.apply_model(data)
        spoken = self.spec.goal.lower()
        not_spans = [
            " ".join(m.group(1).split()).lower()
            for m in re.finditer(r"\bnot\s+(?:a\s+|an\s+)?([^.;,]{2,48})", spoken)
        ]
        if self.spec.item and any(self.spec.item.lower() in span or span in self.spec.item.lower() for span in not_spans):
            self.spec.item = ""
        phrase = _class_phrase(spoken)
        if phrase:
            self.spec.item = phrase
        q = (self.spec.search_query or "").lower()
        need_brand = bool(self.spec.soft_brand and self.spec.soft_brand.lower() not in q)
        need_item = bool(self.spec.item and not any(t in q for t in self.spec.item.lower().split() if len(t) > 2))
        if need_brand or need_item or not self.spec.search_query:
            self.spec.search_query = self.spec._fallback_query()
        if self.spec.item and not self.spec.category:
            joined = re.search(
                rf"((?:[a-z0-9]+\s*&\s*|[a-z0-9]+\s+and\s+)+[a-z0-9]+)\s+{re.escape(self.spec.item.lower())}",
                spoken,
            )
            if joined:
                self.spec.category = f"{joined.group(1)} {self.spec.item}"[:80]
        excl = set()
        for span in not_spans:
            excl |= {t for t in _fold_tokens(span) if len(t) > 2}
        if excl and self.spec.search_query:
            keep = {t.lower() for t in (*self.spec.near, *self.spec.codes) if t}
            kept = [
                w
                for w in self.spec.search_query.split()
                if w.lower().strip(".,;:()") not in excl or w.lower().strip(".,;:()") in keep
            ]
            if kept:
                self.spec.search_query = " ".join(kept)
        extra = _num(data.get("budget")) if isinstance(data, dict) else None
        if extra is not None and self.spec.budget is None:
            self.spec.budget = extra
        if self.schema.get("required") or _REASON_RE.search(self.spec.goal):
            self.spec.need_reason = True

    def _brief_card(self, row: dict) -> dict:
        pid, sku = row["pid"], row["sku"]
        return {
            "product_id": pid,
            "sku": sku,
            "title": self.prover.title(pid)[:90],
            "brand": self.prover.brand(pid),
            "facts": self.prover.facts(pid),
            "options": self.memory.options.get(Memory.key(pid, sku), "")[:160],
            "listing_id": self.memory.listing_of.get(str(pid), ""),
            "price": row.get("price"),
            "ident": int(row.get("ident") or 0),
            "kin": int(row.get("kin") or 0),
            "ok": bool(row.get("ok")),
        }

    def model_pick(self, rows: list[dict], phase: str) -> dict | None:
        if not rows:
            return None
        if len(rows) == 1 or self.spec.universe == "closed" or self.late():
            return None
        if phase == "first_add" and self.spec.codes:
            coded = [r for r in rows if int(r.get("ident") or 0) > 0]
            if len(coded) == 1:
                return coded[0]
            if not coded:
                return None
            rows = coded
        elif phase == "first_add" and self.spec.hard:
            measured = [r for r in rows if r.get("ok")]
            if len(measured) == 1:
                return measured[0]
            if measured:
                rows = measured
        elif phase == "recover" and self.spec.codes:
            close = [r for r in rows if int(r.get("ident") or 0) > 0 or int(r.get("kin") or 0) > 0]
            if len(close) == 1:
                return close[0]
            if close:
                rows = close
        if len(rows) < 2:
            return rows[0] if rows else None
        fp = phase + "|" + ",".join(r["key"] for r in rows[:6]) + "|" + str(self.memory.event_key)
        if fp == self._choice_fp:
            return self._choice_row
        data = self._consult(
            _CHOOSE_SYS,
            {
                "phase": phase,
                "request": self.spec.goal[:1400],
                "hard": [h.get("text") for h in self.spec.hard],
                "codes": self.spec.codes,
                "near": self.spec.near,
                "event": self.memory.event_key,
                "cards": [self._brief_card(r) for r in rows[:6]],
                "need": {"product_id": "id", "sku": "sku", "why": "one short reason"},
            },
            floor=45.0,
        )
        self._choice_fp = fp
        self._choice_row = None
        if not isinstance(data, dict):
            return None
        want_pid = str(data.get("product_id") or "")
        want_sku = str(data.get("sku") or "")
        for row in rows:
            if row["pid"] == want_pid and (not want_sku or row["sku"] == want_sku):
                if phase == "first_add" and self.spec.hard and not row.get("ok"):
                    break
                self._choice_row = row
                return row
        return None

    def model_query(self) -> str | None:
        if self._asked_query or self.late() or self.searches < 1:
            return None
        titles = []
        for pid in list(self.memory.lead_titles)[:8]:
            titles.append(self.memory.lead_titles.get(pid, "")[:80])
        data = self._consult(
            _REWRITE_SYS,
            {
                "request": self.spec.goal[:1200],
                "tried": self.queries_used[:8],
                "seen_titles": titles,
                "codes": self.spec.codes,
                "near": self.spec.near,
                "item": self.spec.item,
                "hard": [h.get("text") for h in self.spec.hard],
                "need": {"query": "at most eight catalog words"},
            },
            floor=50.0,
        )
        self._asked_query = True
        if not isinstance(data, dict):
            return None
        query = _clean_query(str(data.get("query") or ""))
        if query and query not in self.queries_used:
            return query
        return None

    def refine_from_cards(self) -> None:
        if self._refined or self.late() or not self.memory.products:
            return
        cards = []
        for pid, prod in list(self.memory.products.items())[:5]:
            cards.append(
                {
                    "title": str(prod.get("title") or "")[:80],
                    "facts": prod.get("facts") if isinstance(prod.get("facts"), dict) else {},
                    "category_path": prod.get("category_path") or [],
                }
            )
        data = self._consult(
            _REFINE_SYS,
            {
                "request": self.spec.goal[:1200],
                "hard": [h.get("text") for h in self.spec.hard],
                "allowed_claim_fields": self.schema.get("allowed") or [],
                "cards": cards,
                "need": {
                    "fact_keys": ["labelled keys that prove the firm measurements"],
                    "claim_fields": ["allowed justification names to emit"],
                    "item": "product class if still empty",
                    "search_query": "next short query if the first page missed the need",
                },
            },
            floor=50.0,
        )
        self._refined = True
        if isinstance(data, dict):
            self.spec._take_fields(data)

    def model_claims(self, pid: str, sku: str) -> list[dict]:
        if not (self.spec.need_reason or self.schema.get("required")):
            return []
        data = self._consult(
            _CLAIM_SYS,
            {
                "request": self.spec.goal[:1000],
                "hard": [h.get("text") for h in self.spec.hard],
                "allowed": self.schema.get("allowed") or [],
                "claim_fields": self.spec.claim_fields,
                "description": (self.schema.get("description") or "")[:400],
                "card": self.prover.card(pid, sku),
                "need": {"claims": [{"field": "name", "value": "shown on this card"}]},
            },
            floor=40.0,
        )
        rows = data.get("claims") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return []
        out: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            field, value = str(row.get("field") or "").strip(), str(row.get("value") or "").strip()
            if field and value:
                out.append({"field": field, "value": value})
        return out[:6]

    def _unwrap(self, raw: Any) -> dict:
        if not isinstance(raw, dict):
            return {}
        inner = raw.get("observation")
        obs = inner if isinstance(inner, dict) else raw
        if raw.get("error") and isinstance(obs, dict) and not obs.get("error"):
            obs = dict(obs)
            obs["error"] = raw.get("error")
        return obs if isinstance(obs, dict) else {}

    def send(self, calls: list[tuple[str, dict]], content: str = "") -> list[dict]:
        self.turn += 1
        group = f"{self.problem_id}-turn-{self.turn}"
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": self.turn,
            "calls": [
                {"call_id": f"{group}-{i}", "action": {"name": name, "args": args}}
                for i, (name, args) in enumerate(calls, start=1)
            ],
        }
        result = None
        for attempt in range(2):
            result = _PROXY.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                break
            if self.remaining() < 12:
                break
            time.sleep(0.35 * (attempt + 1))
        tool_calls = [
            {
                "id": f"agent-{self.turn}-{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(calls, start=1)
        ]
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        observations: list[dict] = []
        if not isinstance(result, dict):
            return observations
        returned = result.get("calls") or []
        for (name, args), rec in zip(calls, returned):
            wrap = rec.get("observation") if isinstance(rec, dict) else {}
            if not isinstance(wrap, dict):
                wrap = {}
            if wrap.get("done"):
                self.done = True
            obs = self._unwrap(wrap)
            self.memory.record(name, args, obs, self.spec)
            observations.append(obs)
        if any(name in {"view", "compare"} for name, _args in calls):
            self.prover.judge_new()
            if self.spec.need_reason:
                self.refine_from_cards()
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.spec.revise(text)
            hike = _FROM_TO_RE.search(text)
            if hike and self.memory.event_key:
                new_price = _num((hike.group(3) or "") + "." + (hike.group(4) or "0"))
                if new_price is not None:
                    pid, _, sku = self.memory.event_key.partition("::")
                    self.memory._live(pid, sku, True, new_price)
                    if self.spec.budget is not None and new_price > self.spec.budget + _EPS:
                        self.memory.dead.add(self.memory.event_key)
            if "?" in text:
                self.pending_question = True
            if self.remaining() > 90 and not self.memory.event_key:
                self.interpret()
                self.prover.cache.clear()
                self.prover.judge_new()
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in returned):
            self.done = True
        return observations

    def act(self, name: str, args: dict, content: str = "") -> dict:
        if name not in self.names or self.done or self.turn >= self.max_steps:
            return {}
        obs = self.send([(name, args)], content)
        return obs[0] if obs else {}

    def legal(self, require_viewed: bool = True) -> list[dict]:
        rows: list[dict] = []
        keys = set(self.memory.live) | set(self.memory.options) | set(self.memory.rows)
        if self.spec.universe == "closed":
            keys |= {Memory.key(pid, sku) for pid, sku in self.spec.closed}
        rank_pool = bool(self.spec.axes and self.spec.universe == "closed")
        for key in keys:
            pid, _, sku = key.partition("::")
            if not sku:
                continue
            if key in self.memory.dead or key == self.memory.event_key:
                continue
            if require_viewed and str(pid) not in self.memory.products:
                continue
            row = self.prover.prove(pid, sku)
            if rank_pool:
                if not row.get("live"):
                    continue
            elif not row["ok"]:
                continue
            rows.append(row)
        rows.sort(key=self._row_key)
        if self.spec.axes:
            rows = [r for r in rows if not self.prover.dominated(r, rows)]
            rows.sort(key=self._row_key)
        return rows

    def _row_key(self, row: dict) -> tuple:
        hits = -int(row.get("hits") or 0) if self.spec.prefer_not_cheap else 0
        kin = -int(row.get("kin") or 0) if self.memory.event_key and not self.spec.axes else 0
        ident = -int(row.get("ident") or 0)
        if self.spec.axes:
            return (row["score"], ident, hits, row["relax"], row["key"])
        if self.spec.prefer_not_cheap:
            return (ident, hits, row["relax"], row.get("price") if row.get("price") is not None else 1e18, row["key"])
        return (row["relax"], ident, kin, row["score"], row["key"])

    def best_effort(self) -> dict | None:
        rows: list[dict] = []
        keys = set(self.memory.live) | set(self.memory.options)
        for key in keys:
            pid, _, sku = key.partition("::")
            if not sku or str(pid) not in self.memory.products:
                continue
            if key in self.memory.dead or key == self.memory.event_key:
                continue
            row = self.prover.prove(pid, sku)
            if not row.get("live"):
                continue
            if row.get("price") is None:
                continue
            if self.spec.budget is not None and row["price"] > self.spec.budget + _EPS:
                continue
            rows.append(row)
        if self.spec.need_reason and self.spec.hard:
            proven = self._labelled(rows)
            tight = [r for r in proven if self._class_fit(r["pid"]) >= 2]
            if tight:
                rows = tight
            else:
                mid = [r for r in proven if self._class_fit(r["pid"]) > 0]
                if mid:
                    rows = mid
                elif proven:
                    rows = proven
                else:
                    classy = [r for r in rows if self._class_fit(r["pid"]) >= 2]
                    if classy:
                        rows = classy
        elif self.memory.event_key and self.spec.codes:
            kin = self._stem_kin(rows)
            if kin:
                rows = kin
            else:
                event_pid, _event_sku = self._event_pid_sku()
                coded = [
                    r
                    for r in rows
                    if r["pid"] != event_pid and int(r.get("ident") or 0) > 0
                ]
                if coded:
                    rows = coded
        elif not self.memory.adds and (self.spec.item or self.spec.category):
            tight = [r for r in rows if self._class_fit(r["pid"]) >= 2]
            if tight:
                rows = tight
        if not self.memory.adds:
            rows.sort(
                key=lambda r: (
                    0 if r["ok"] else 1,
                    -int(r.get("ident") or 0),
                    -self._lead_score(r["pid"]),
                    r.get("price") if r.get("price") is not None else 1e18,
                    r["key"],
                )
            )
        else:
            event_pid, event_sku = self._event_pid_sku()
            stem = _sku_stem(event_sku)
            lid = self.memory.listing_of.get(event_pid) if event_pid else None

            def _recover_key(r: dict) -> tuple:
                same_stem = 0 if stem and r["pid"] != event_pid and _sku_stem(r.get("sku") or "") == stem else 1
                same = 0 if lid and self.memory.listing_of.get(r["pid"]) == lid else 1
                return (
                    same_stem,
                    0 if r["ok"] else 1,
                    -self._family_score(r),
                    self.memory.lead_rank.get(str(r["pid"]), 99),
                    same,
                    -int(r.get("ident") or 0),
                    -int(r.get("kin") or 0),
                    r.get("price") if r.get("price") is not None else 1e18,
                    r["key"],
                )

            rows.sort(key=_recover_key)
        return rows[0] if rows else None

    def pick(self) -> dict | None:
        rows = self.legal()
        if not rows:
            return None
        first = self.memory.adds == 0
        if first and self.spec.universe == "closed":
            rows.sort(key=self._row_key)
            return rows[0] if rows else None
        if first and self.spec.codes:
            coded = [r for r in rows if int(r.get("ident") or 0) > 0]
            if coded:
                rows = coded
            elif not self.late():
                return None
        if first and self.spec.hard and not self.late():
            if self.spec.need_reason:
                proven = self._labelled(rows)
                if proven:
                    rows = proven
                else:
                    return None
            else:
                proven = self._labelled(rows)
                if proven:
                    rows = proven
                else:
                    measured = [r for r in rows if r.get("ok")]
                    if not measured:
                        return None
        if first and self.spec.soft_brand:
            want = self.spec.soft_brand.lower()
            branded = [
                r
                for r in rows
                if want in self.prover.brand(r["pid"]).lower() or want in self.prover.title(r["pid"]).lower()
            ]
            if branded:
                rows = branded
        if first and (self.spec.item or self.spec.category) and not self.spec.codes:
            tight = [r for r in rows if self._class_fit(r["pid"]) >= 2]
            if tight:
                rows = tight
            elif (self.spec.need_reason or self.spec.hard) and not self.late():
                return None
        if first and self.spec.codes and self.spec.near and not self.late():
            colored = [r for r in rows if self._near_hits(r) > 0]
            if colored:
                rows = colored
            else:
                return None
        if first and self.spec.exact_first and self.turn < 8 and not self.late() and not (
            self.spec.need_reason and self._labelled(rows)
        ):
            exact = [r for r in rows if r["relax"] == 0]
            if exact:
                rows = exact
            else:
                return None
        if self.memory.event_key:
            rows = [r for r in rows if r["key"] != self.memory.event_key]
            if not self.spec.axes:
                kin = self._stem_kin(rows)
                if kin:
                    rows = kin
                elif self.spec.codes and not self.late():
                    return None
                elif self.spec.codes:
                    event_pid, _event_sku = self._event_pid_sku()
                    coded = [
                        r
                        for r in rows
                        if r["pid"] != event_pid and int(r.get("ident") or 0) > 0
                    ]
                    if coded:
                        rows = coded
                    else:
                        return None
        if first and self.spec.universe == "phrase":
            rows.sort(
                key=lambda r: (
                    -self._lead_score(r["pid"]),
                    r.get("price") if r.get("price") is not None else 1e18,
                    r["key"],
                )
            )
            return rows[0]
        if not first and self.spec.need_reason and self.spec.hard:
            proven = self._labelled(rows)
            tight = [r for r in proven if self._class_fit(r["pid"]) >= 2]
            if tight:
                rows = tight
            else:
                mid = [r for r in proven if self._class_fit(r["pid"]) > 0]
                if mid:
                    rows = mid
                elif not self.late():
                    return None
                elif proven:
                    rows = proven
            if not rows and not self.late():
                return None
        if first and self.spec.need_reason:
            rows.sort(
                key=lambda r: (
                    0 if r.get("price") is not None else 1,
                    r.get("price") if r.get("price") is not None else 1e18,
                    -int(r.get("ident") or 0),
                    -self._lead_score(r["pid"]),
                    r["key"],
                )
            )
            return rows[0]
        if first:
            labelled = {r["key"] for r in self._labelled(rows)}
            want = (self.spec.soft_brand or "").lower()

            def _first_key(r: dict) -> tuple:
                branded = 0
                if want and (
                    want in self.prover.brand(r["pid"]).lower() or want in self.prover.title(r["pid"]).lower()
                ):
                    branded = 1
                return (
                    0 if r["key"] in labelled else 1,
                    0 if branded else 1,
                    -int(r.get("ident") or 0),
                    -self._near_option_hits(r),
                    -self._near_hits(r),
                    -self._lead_score(r["pid"]),
                    r["relax"],
                    r.get("price") if r.get("price") is not None else 1e18,
                    r["key"],
                )

            rows.sort(key=_first_key)
            return rows[0]
        if self.memory.event_key and not self.spec.axes:
            event_pid, event_sku = self._event_pid_sku()
            stem = _sku_stem(event_sku)
            rows.sort(
                key=lambda r: (
                    0 if stem and _sku_stem(r.get("sku") or "") == stem and r["pid"] != event_pid else 1,
                    0 if r["ok"] else 1,
                    -self._family_score(r),
                    self.memory.lead_rank.get(str(r["pid"]), 99),
                    -int(r.get("ident") or 0),
                    -int(r.get("kin") or 0),
                    r.get("price") if r.get("price") is not None else 1e18,
                    r["key"],
                )
            )
            return rows[0] if rows else None
        rows.sort(key=self._row_key)
        return rows[0] if rows else None

    def surveyed(self) -> bool:
        if self.late() or self.memory.adds or self.memory.event_key:
            return True
        if self.spec.universe == "closed":
            missing = [p for p, _s in self.spec.closed if p not in self.memory.products]
            return not missing
        if self.spec.survey and len(self.memory.identities) < min(self.spec.survey, 6) and self.searches < 3 and self.page_reads < 2:
            return False
        rows = self.legal()
        unread_left = any(str(pid) not in self.memory.products for pid in (self.memory.recent_ids or []))
        if self.spec.need_reason and self._labelled(rows):
            if not (self.spec.item or self.spec.category) or any(self._class_fit(r["pid"]) >= 2 for r in self._labelled(rows)):
                return True
        if self.spec.codes and not any(int(r.get("ident") or 0) > 0 for r in rows) and self.searches < 4:
            return False
        ready = [
            r
            for r in rows
            if not (self.spec.item or self.spec.category) or self._class_fit(r["pid"]) >= 2
        ]
        branded_ready = []
        if self.spec.soft_brand:
            want = self.spec.soft_brand.lower()
            branded_ready = [
                r
                for r in ready
                if want in self.prover.brand(r["pid"]).lower() or want in self.prover.title(r["pid"]).lower()
            ]
        if branded_ready:
            return True
        if ready and not self.spec.soft_brand and (self.spec.hard or self.spec.item):
            return True
        if unread_left:
            return False
        if self.spec.soft_brand and not self.spec.codes and not branded_ready and self.searches < 3:
            return False
        if (self.spec.item or self.spec.category) and not any(self._class_fit(r["pid"]) >= 2 for r in rows):
            if not self.late():
                return False
        if self.spec.universe == "phrase" and self.searches < 2 and not self._labelled(rows):
            return False
        if self.spec.phrase and not rows and self.searches < 4:
            return False
        if (self.spec.hard or self.spec.codes) and not rows and self.searches < 6:
            return False
        if not rows and self.searches < 2:
            return False
        return True

    def _labelled(self, rows: list[dict] | None = None) -> list[dict]:
        pool = rows if rows is not None else self.legal()
        hard = [str(h.get("text") or "") for h in self.spec.hard if str(h.get("text") or "").strip()]
        if not hard:
            return []
        return [
            r
            for r in pool
            if all(self.prover._need_status(r["pid"], r["sku"], text) == "match" for text in hard)
        ]

    def _near_option_hits(self, row: dict) -> int:
        blob = str(self.memory.options.get(row["key"], "") or "").lower()
        score = 0
        for tok in self.spec.near:
            if tok and tok.lower() in blob:
                score += 1
        return score

    def _near_hits(self, row: dict) -> int:
        blob = (
            f"{self.prover.title(row['pid'])} {self.prover.brand(row['pid'])} "
            f"{self.memory.options.get(row['key'], '')} {self.prover.layers(row['pid'], row['sku'])}"
        ).lower()
        score = 0
        for tok in self.spec.near:
            if tok and tok.lower() in blob:
                score += 2
        return score

    def _event_pid_sku(self) -> tuple[str, str]:
        key = self.memory.event_key or ""
        pid, _, sku = key.partition("::")
        return pid, sku

    def _stem_kin(self, rows: list[dict]) -> list[dict]:
        event_pid, event_sku = self._event_pid_sku()
        stem = _sku_stem(event_sku)
        if not stem:
            return []
        return [
            r
            for r in rows
            if r.get("pid") != event_pid and _sku_stem(r.get("sku") or "") == stem
        ]

    def _stem_search_ids(self) -> list[str]:
        event_pid, event_sku = self._event_pid_sku()
        stem = _sku_stem(event_sku)
        if not stem:
            return []
        ids: list[str] = []
        for _key, row in self.memory.rows.items():
            pid = str(row.get("product_id") or "")
            sku = str(row.get("sku") or "")
            if not pid or pid == event_pid or pid in self.memory.products:
                continue
            if _sku_stem(sku) == stem and pid not in ids:
                ids.append(pid)
        return ids

    def _compare_stem_hits(self) -> list[tuple[str, dict]] | None:
        ids = self._stem_search_ids()
        if not ids or "compare" not in self.names:
            return None
        return [("compare", {"product_ids": ids[:5]})]

    def _family_score(self, row: dict) -> int:
        if not self.memory.event_key:
            return 0
        event_pid, _, event_sku = self.memory.event_key.partition("::")
        score = 0
        stem = _sku_stem(event_sku)
        if stem and _sku_stem(row.get("sku") or "") == stem:
            score += 4
        lid = self.memory.listing_of.get(event_pid)
        if lid and self.memory.listing_of.get(row["pid"]) == lid:
            score += 4
        dead_brand = self.prover.brand(event_pid).lower()
        brand = self.prover.brand(row["pid"]).lower()
        if dead_brand and brand and dead_brand == brand:
            score += 2
        elif not dead_brand or not brand:
            score += 0
        kin = int(row.get("kin") or 0)
        score += kin
        if kin >= 3 and (not dead_brand or not brand or dead_brand == brand):
            score = max(score, 4)
        if dead_brand and brand and dead_brand != brand and _sku_stem(row.get("sku") or "") != stem:
            score = min(score, 3)
        return score

    def _item_toks(self) -> set[str]:
        raw = _fold_tokens(f"{self.spec.item} {self.spec.category}")
        keep = {t for t in raw if len(t) > 2 and t not in _ITEM_NOISE}
        tail = re.findall(r"[a-z0-9]+", (self.spec.item or "").lower())
        if tail and tail[-1] not in _ITEM_NOISE:
            keep.add(tail[-1])
            keep.add(_stem_word(tail[-1]))
        return keep

    def _class_fit(self, pid: str) -> int:
        want = self._item_toks()
        if not want:
            return 1
        path = self.prover.product(pid).get("category_path") or []
        if not path:
            return 0 if self.spec.need_reason else 1
        leaf_words = [t for t in re.findall(r"[a-z0-9]+", str(path[-1]).lower()) if len(t) > 2]
        if not leaf_words:
            return 0
        last = _stem_word(leaf_words[-1])
        want_stem = {_stem_word(t) for t in want} | want
        if last in want_stem:
            return 2
        if _token_overlap(want, set(leaf_words)):
            return 1
        whole = _fold_tokens(" ".join(str(x) for x in path))
        if _token_overlap(want, whole):
            return 1
        return 0

    def _learn_shelf(self) -> None:
        source = self._labelled()
        for row in source:
            if self._class_fit(row["pid"]) < 2:
                continue
            path = self.prover.product(row["pid"]).get("category_path") or []
            if path:
                self.shelf = str(path[-1])
                return

    def _short_hard(self) -> str:
        bits: list[str] = []
        for hard in self.spec.hard:
            text = str(hard.get("text") or "").strip()
            if text and _firm_need(text):
                bits.append(text)
        return " ".join(bits[:3])

    def _code_queries(self, recover: bool = False) -> list[str]:
        codes = list(self.spec.codes)
        near = list(self.spec.near[:2])
        item = self.spec.item
        if recover:
            hyphens = [
                h
                for h in re.findall(r"\b[a-z]+-[a-z]+\b", (self.spec.goal or "").lower())
                if h not in {"in-stock", "at-or", "to-haves"}
            ][:3]
            raw = [
                " ".join(p for p in (*codes, item) if p),
                " ".join(p for p in (*codes, *hyphens, item) if p),
                " ".join(codes),
                " ".join(p for p in (*codes, *near, item) if p),
                " ".join(p for p in (*codes, *near) if p),
            ]
        else:
            raw = [
                " ".join(p for p in (*codes, *near, item) if p),
                " ".join(p for p in (*codes, item) if p),
                " ".join(p for p in (*codes, *near) if p),
                " ".join(codes),
            ]
        out: list[str] = []
        for text in raw:
            query = _clean_query(text)
            if query and query not in out:
                out.append(query)
        return out

    def _next_query(self) -> str | None:
        phrase = " ".join(self.spec.phrase)
        hard = self._short_hard()
        named = " ".join(p for p in (*self.spec.codes, *self.spec.near, self.spec.soft_brand, self.spec.item, hard) if p)
        measured = " ".join(p for p in (hard, self.spec.item) if p)
        shelf = self.spec.category or self.shelf
        if self.spec.codes:
            for query in (*self._code_queries(), self.spec.search_query, named):
                query = _clean_query(query or "")
                if query and query not in self.queries_used:
                    return query
            return None
        if self.spec.universe == "phrase" and phrase:
            candidates = [
                phrase,
                " ".join(p for p in (shelf, phrase) if p),
                named,
                self.spec.search_query,
            ]
        else:
            lead = measured if self.spec.need_reason and measured else named
            candidates = [
                lead,
                named,
                measured,
                self.spec.search_query,
                phrase,
                " ".join(p for p in (shelf, phrase) if p),
                " ".join(p for p in (self.spec.item, hard) if p),
                self.spec.item,
                shelf,
            ]
            if not self.spec.item:
                candidates.append(hard)
        for query in candidates:
            query = _clean_query(query or "")
            if query and query not in self.queries_used:
                return query
        return self._rewrite_query()

    def _rewrite_query(self) -> str | None:
        bits = [self.spec.item, " ".join(self.spec.codes), self.spec.soft_brand]
        bits.append(self._short_hard())
        source = list(self.memory.identities) or list(self.memory.products)
        for pid in source:
            words = self.prover.title(pid).split()
            digits = [w for w in words if any(ch.isdigit() for ch in w) and len(w) >= 2]
            if digits:
                bits.extend(digits[:2])
                break
        text = _clean_query(" ".join(p for p in bits if p))
        if text and text not in self.queries_used:
            return text
        return None

    def _lead_score(self, pid: str) -> int:
        title = (self.memory.lead_titles.get(str(pid)) or self.prover.title(pid) or "").lower()
        if not title:
            return 0
        score = 0
        for code in self.spec.codes:
            if code.lower() in title:
                score += 6
        for tok in self.spec.near:
            if tok.lower() in title:
                score += 3
        if self.spec.soft_brand and self.spec.soft_brand.lower() in title:
            score += 3
        rank = self.memory.lead_rank.get(str(pid), 99)
        if rank < 10:
            score += 10 - rank
        for hard in self.spec.hard:
            piece = str(hard.get("text") or "").lower()
            if piece and piece in title:
                score += 4
            for num in _numbers(piece):
                token = ("%g" % num) if abs(num - int(num)) > 1e-9 else str(int(num))
                if re.search(rf"(?<!\d){re.escape(token)}(?!\d)", title):
                    score += 2
        for tok in self.spec.phrase:
            if tok and tok in title:
                score += 2
        return score

    def gather(self) -> list[tuple[str, dict]] | None:
        self._learn_shelf()
        if self.spec.universe == "closed":
            missing = [p for p, _s in self.spec.closed if p not in self.memory.products]
            if missing and "compare" in self.names:
                return [("compare", {"product_ids": missing[:5]})]
            return None
        unread = []
        source = self.memory.recent_ids or [
            str(row.get("product_id") or key.split("::", 1)[0])
            for key, row in self.memory.rows.items()
        ]
        for pid in source:
            if pid in self.memory.products or pid in self.memory.dead_products:
                continue
            if pid not in unread:
                unread.append(pid)
        unread.sort(key=lambda pid: -self._lead_score(pid))
        view_cap = 6 if (self.spec.hard and (self.spec.item or self.spec.category)) else 4
        if unread and self.page_reads < view_cap:
            if "compare" in self.names:
                calls = [("compare", {"product_ids": unread[:5]})]
                extra = unread[5:10]
                if extra and (self.spec.hard or self.spec.soft_brand or self.spec.codes):
                    calls.append(("compare", {"product_ids": extra}))
                return calls
            if "view" in self.names:
                return [("view", {"product_id": unread[0]})]
        calls: list[tuple[str, dict]] = []

        def _filter_call() -> tuple[str, dict] | None:
            if "filter" not in self.names or self.searches >= 8:
                return None
            filt: dict[str, Any] = {"in_stock": True, "k": 10}
            shelf = self.spec.category or self.spec.item
            if self.shelf:
                leaf_words = [t for t in re.findall(r"[a-z0-9]+", self.shelf.lower()) if len(t) > 2]
                last = _stem_word(leaf_words[-1]) if leaf_words else ""
                want = {_stem_word(t) for t in self._item_toks()} | self._item_toks()
                if last and last in want:
                    shelf = self.shelf
            shelf_toks = _fold_tokens(shelf) if shelf else set()
            if shelf and shelf.lower() not in _ITEM_NOISE and not shelf_toks <= _ITEM_NOISE:
                filt["category"] = shelf
            if self.spec.budget is not None:
                filt["max_price"] = self.spec.budget
            if self.spec.soft_brand:
                filt["brand"] = self.spec.soft_brand
            signature = json.dumps(filt, sort_keys=True)
            if (
                filt.get("category") or filt.get("max_price") is not None or filt.get("brand")
            ) and "filter" + signature not in self._seen_calls:
                return ("filter", filt)
            return None

        have_class = any(self._class_fit(r["pid"]) >= 2 for r in self.legal())
        branded_legal = False
        if self.spec.soft_brand:
            want = self.spec.soft_brand.lower()
            branded_legal = any(
                want in self.prover.brand(r["pid"]).lower() or want in self.prover.title(r["pid"]).lower()
                for r in self.legal()
            )
        unread_left = any(str(pid) not in self.memory.products for pid in (self.memory.recent_ids or []))
        need_shelf = self.page_reads > 0 and not unread_left and (
            (bool(self.spec.need_reason and self.spec.hard) and not self._labelled())
            or (bool((self.spec.need_reason or self.spec.hard) and self.spec.item) and not have_class)
            or bool(
                self.memory.event_key
                and (self.spec.need_reason or self.spec.item)
                and self.shelf
                and not self.spec.codes
            )
        )
        if need_shelf:
            filt = _filter_call()
            if filt:
                return [filt]
        if self.spec.soft_brand and self.searches >= 1 and not branded_legal:
            filt = _filter_call()
            if filt:
                return [filt]
        query = self._next_query()
        if query and "search" in self.names and self.searches < 8:
            calls.append(("search", {"query": _clean_query(query), "k": 10, "in_stock": True}))
            return calls[: self.max_calls]
        filt = _filter_call()
        if filt:
            calls.append(filt)
        return calls or None

    def fresh(self, pid: str, sku: str) -> bool:
        key = Memory.key(pid, sku)
        if key not in self.memory.inspected:
            return False
        return self.memory.last_add_seq is None or self.memory.live_seq.get(key, 0) > self.memory.last_add_seq

    def claims(self, pid: str, sku: str) -> list[dict]:
        allowed = [str(a) for a in self.schema.get("allowed") or []]
        live = self.memory.status(pid, sku) or {}
        facts = self.prover.facts(pid)
        built: list[dict] = []
        used: set[str] = set()

        allowed_l = {a.lower(): a for a in allowed}
        extra_fields = [f for f in (*self.spec.claim_fields, *self.spec.fact_hints) if f]
        need_all = bool(self.spec.need_reason or self.schema.get("required"))
        drafted = self.model_claims(pid, sku)

        def add(field: Any, value: Any, force: bool = False) -> None:
            field_s, value_s = str(field or "").strip(), str(value or "").strip()
            if not field_s or not value_s:
                return
            key = field_s.lower()
            if allowed and key not in allowed_l and key not in {"price", "in_stock", "brand"} and not force:
                best, hit = None, 0
                extra = _field_tokens(key) | _fold_tokens(key)
                for al, orig in allowed_l.items():
                    n = _token_overlap(extra, _field_tokens(al) | {al})
                    if n > hit:
                        best, hit = orig, n
                if best and hit:
                    field_s, key = best, best.lower()
                elif not force:
                    return
            if key in used:
                return
            if not self.prover.grounded(pid, sku, field_s, value_s):
                return
            used.add(key)
            built.append({"field": field_s, "value": value_s})

        spec_keys = {"display", "resolution", "size", "screen_size", "phone_type", "refresh_rate"}
        for row in drafted:
            field_s = str(row.get("field") or "").strip().lower()
            if field_s not in spec_keys | {"price", "in_stock", "brand"}:
                continue
            add(row.get("field"), row.get("value"), force=True)
        if need_all and self.spec.budget is not None:
            add("price", "at most %.2f" % self.spec.budget)
        if need_all:
            add("in_stock", "true" if live.get("in_stock") else "false")

        asked = _fold_tokens(
            " ".join(
                [
                    self.spec.goal,
                    self.spec.spoken,
                    " ".join(str(h.get("text") or "") for h in self.spec.hard),
                    " ".join(str(s.get("text") or "") for s in self.spec.soft),
                    " ".join(str(a.get("from") or "") for a in self.spec.axes),
                ]
            )
        )
        core = {"price", "in_stock", "brand"}
        hard_nums: list[float] = []
        hard_units: set[str] = set()
        for hard in self.spec.hard:
            piece = str(hard.get("text") or "")
            hard_nums.extend(_numbers(piece))
            hard_units |= _fold_tokens(piece)
        for field in list(allowed) + extra_fields:
            key = field.lower()
            asked_hit = bool(_field_tokens(field) & asked) or bool(_fold_tokens(field) & asked)
            if key not in core and not need_all and not asked_hit:
                continue
            value = None
            if key == "brand":
                value = self.prover.brand(pid)
            elif key == "category":
                path = self.prover.product(pid).get("category_path") or []
                value = path[-1] if path else None
            elif key == "in_stock":
                value = "true" if live.get("in_stock") else "false"
            elif key == "price":
                if self.spec.budget is not None:
                    value = "at most %.2f" % self.spec.budget
                elif live.get("price") is not None:
                    value = "%.2f" % live["price"]
            else:
                want = _field_tokens(field)
                best, score = None, 0
                for fact_key, fact_val in facts.items():
                    n = _token_overlap(want, _field_tokens(str(fact_key)))
                    if hard_nums and any(
                        abs(x - w) < 1e-6 for x, _neigh in _numbers_near(str(fact_val)) for w in hard_nums
                    ):
                        n += 2
                    if _token_overlap(_fold_tokens(str(fact_key)), hard_units):
                        n += 2
                    if n > score:
                        best, score = fact_val, n
                value = best if score else None
            if value is not None:
                add(field, value)
        if need_all or self.spec.hard:
            for fact_key, fact_val in facts.items():
                fk = str(fact_key)
                val_s = str(fact_val)
                hit = bool(_token_overlap(_fold_tokens(fk), hard_units | asked))
                shown = val_s
                for want_n in hard_nums:
                    if any(abs(x - want_n) < 1e-6 for x, _neigh in _numbers_near(val_s)):
                        hit = True
                        shown = ("%g" % want_n) if abs(want_n - int(want_n)) < 1e-9 else ("%.1f" % want_n)
                        break
                if not hit:
                    continue
                add(fk, shown, force=True)
                for al, orig in allowed_l.items():
                    if _token_overlap(_field_tokens(fk), _field_tokens(al)):
                        add(orig, shown)
        if need_all:
            skip_val = {"", "unknown", "absent", "false", "n/a", "none", "no"}
            for key in ("phone_type", "display", "resolution", "size", "screen_size"):
                for fact_key, fact_val in facts.items():
                    if str(fact_key).lower() != key:
                        continue
                    val_s = str(fact_val).strip()
                    if val_s.lower() in skip_val:
                        continue
                    blob = _fold_tokens(f"{fact_key} {val_s}")
                    hard_hit = bool(hard_nums) and any(
                        abs(x - w) < 1e-6 for x, _n in _numbers_near(val_s) for w in hard_nums
                    )
                    token_hit = bool(_token_overlap(blob, hard_units | asked))
                    if key == "phone_type" or hard_hit or token_hit:
                        add(fact_key, val_s, force=True)
        if "price" not in used and live.get("price") is not None:
            add("price", "%.2f" % live["price"])
        if "in_stock" not in used:
            add("in_stock", "true" if live.get("in_stock") else "false")
        brand = self.prover.brand(pid)
        if brand and "brand" not in used:
            add("brand", brand)
        return built[:8]

    def order_args(self, pid: str, sku: str) -> dict:
        args: dict[str, Any] = {"product_id": pid, "sku": sku}
        if self.spec.need_reason or self.schema.get("required"):
            claims = self.claims(pid, sku)
            if claims:
                args["justification"] = {"claims": claims}
        return args

    def speak(self) -> str:
        target = self.pick()
        if not target:
            return "I am still matching the catalog to your requirements and will re-check live price and stock before ordering."
        title = self.prover.title(target["pid"]) or target["pid"]
        bits = [f"I am going with {title[:90]}"]
        if target.get("price") is not None:
            bits.append(f"live price {target['price']}")
        bits.append("I will verify stock and price before placing the order.")
        return ", ".join(bits) + "."

    def plan(self) -> list[tuple[str, dict]] | None:
        if self.done:
            return None
        if self.pending_question and "message" in self.names and self.turn <= self.max_steps - 3:
            self.pending_question = False
            return [("message", {"content": self.speak()})]
        if self.spec.ask and not self.asked and "message" in self.names:
            self.asked = True
            return [("message", {"content": self.spec.ask})]
        if self.late():
            return None
        live_cart = None
        for pid, sku in self.memory.cart:
            key = Memory.key(pid, sku)
            if key == self.memory.event_key or key in self.memory.dead:
                continue
            row = self.prover.prove(pid, sku)
            if row["ok"]:
                live_cart = (pid, sku)
                break
        if live_cart and self.memory.adds and not self.pending_recheck:
            pid, sku = live_cart
            if not self.fresh(pid, sku) and "inspect_stock" in self.names:
                return [("inspect_stock", {"product_id": pid, "sku": sku})]
            return [("place_test_order", self.order_args(pid, sku))]
        if self.memory.adds and self.memory.event_key:
            target = self.pick()
            if target is None:
                hits = self._compare_stem_hits()
                if hits:
                    return hits[: self.max_calls]
                extra = self.dead_search()
                if extra:
                    more = self._compare_stem_hits()
                    if more:
                        extra = extra + more
                    return extra[: self.max_calls]
            if target is None and not self.spec.codes:
                extra = self.gather()
                if extra and not self.late():
                    return extra[: self.max_calls]
            if target is None:
                target = self.best_effort()
            if target is None:
                return None
            pid, sku = target["pid"], target["sku"]
            if not self.fresh(pid, sku) and "inspect_stock" in self.names:
                return [("inspect_stock", {"product_id": pid, "sku": sku})]
            return [("add_to_cart", {"product_id": pid, "sku": sku})]
        for pid, sku in list(self.memory.cart):
            if Memory.key(pid, sku) in self.memory.dead and "remove_from_cart" in self.names:
                return [("remove_from_cart", {"product_id": pid, "sku": sku})]
        if not self.surveyed() or self.pick() is None:
            gathered = self.gather()
            if gathered:
                return gathered[: self.max_calls]
        target = self.pick()
        if target is None:
            gathered = self.gather()
            return gathered[: self.max_calls] if gathered else None
        pid, sku = target["pid"], target["sku"]
        if not self.fresh(pid, sku) and "inspect_stock" in self.names:
            return [("inspect_stock", {"product_id": pid, "sku": sku})]
        return [("add_to_cart", {"product_id": pid, "sku": sku})]

    def survivors(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        if self.spec.universe == "closed":
            pairs = [(str(p), str(s)) for p, s in self.spec.closed]
        else:
            for key in set(self.memory.options) | set(self.memory.live):
                pid, _, sku = key.partition("::")
                if sku:
                    pairs.append((str(pid), str(sku)))
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for pid, sku in pairs:
            key = Memory.key(pid, sku)
            if key in seen or key == self.memory.event_key or key in self.memory.dead:
                continue
            if str(pid) not in self.memory.products:
                continue
            seen.add(key)
            out.append((pid, sku))
        event_pid, event_sku = self._event_pid_sku()
        stem = _sku_stem(event_sku)
        if event_pid and stem and not self.spec.axes:
            out.sort(
                key=lambda pair: (
                    0 if pair[0] != event_pid and _sku_stem(pair[1]) == stem else 1,
                    pair[0],
                    pair[1],
                )
            )
            return out[:12]
        return out[:6]

    def refresh_survivors(self) -> None:
        if "inspect_stock" not in self.names:
            return
        event_pid, event_sku = self._event_pid_sku()
        stem = _sku_stem(event_sku)
        prefer_stem = bool(event_pid and stem and not self.spec.axes)
        twins = []
        rest = []
        for r in self.legal():
            pair = (r["pid"], r["sku"])
            if prefer_stem and r["pid"] != event_pid and _sku_stem(r.get("sku") or "") == stem:
                twins.append(pair)
            else:
                rest.append(pair)
        ordered = twins + rest
        extra = [pair for pair in self.survivors() if pair not in ordered]
        if prefer_stem:
            extra.sort(
                key=lambda pair: (
                    0 if pair[0] != event_pid and _sku_stem(pair[1]) == stem else 1,
                    pair[0],
                    pair[1],
                )
            )
        if prefer_stem:
            extra = [
                pair
                for pair in extra
                if pair[0] != event_pid and _sku_stem(pair[1]) == stem
            ]
            if not twins and not extra:
                return
            ordered = twins
        cap = 6 if self.spec.axes and self.spec.universe == "closed" else (4 if prefer_stem else 2)
        checked = 0
        for pid, sku in ordered + extra:
            if self.done or self.turn >= self.max_steps or checked >= cap:
                return
            if not self.fresh(pid, sku):
                self.act("inspect_stock", {"product_id": pid, "sku": sku})
                checked += 1

    def recheck(self) -> None:
        pid, sku = self.pending_recheck or ("", "")
        self.pending_recheck = None
        if not pid:
            return
        obs = self.act("inspect_stock", {"product_id": pid, "sku": sku})
        live = self.memory.status(pid, sku) or {}
        price = live.get("price")
        stock = bool(live.get("in_stock")) if live else False
        if isinstance(obs, dict) and obs.get("error"):
            stock = False
        over = self.spec.budget is not None and price is not None and price > self.spec.budget + _EPS
        key = Memory.key(pid, sku)
        first = self.memory.adds <= 1
        if first:
            self.memory.event_key = self.memory.event_key or key
        if not stock or over:
            self.memory.dead.add(key)
            if self.spec.universe != "closed":
                self.memory.dead_products.add(str(pid))
            if not self.done:
                self.recover()
            return
        if first:
            if not self.done:
                self.recover()
            return
        if self.pending_question:
            return
        self.act("place_test_order", self.order_args(pid, sku))

    def _dead_query(self) -> str:
        if not self.memory.event_key:
            return ""
        if self.spec.codes:
            for query in self._code_queries(recover=True):
                if query not in self.queries_used:
                    return query
            if self.spec.item:
                query = _clean_query(self.spec.item)
                if query and query not in self.queries_used:
                    return query
            return ""
        title = self.prover.title(self.memory.event_key.split("::", 1)[0])
        words = title.split()
        bits: list[str] = []
        for i, word in enumerate(words):
            if any(ch.isdigit() for ch in word) and len(word) >= 2:
                prev = words[i - 1] if i else ""
                bits.append(" ".join(x for x in (prev, word) if x))
                break
        if not bits and words:
            bits.append(" ".join(words[:4]))
        hard = " ".join(str(h.get("text") or "") for h in self.spec.hard[:2])
        return _clean_query(" ".join(p for p in (*bits, hard, self.spec.item) if p))

    def dead_search(self) -> list[tuple[str, dict]] | None:
        if self.spec.universe == "closed":
            return None
        if self.searched_twins and not self.spec.codes:
            return None
        query = self._dead_query()
        if not query or "search" not in self.names:
            return None
        args = {"query": query, "k": 10, "in_stock": True}
        signature = "search" + json.dumps(args, sort_keys=True)
        if signature in self._seen_calls:
            return None
        if not self.spec.codes:
            self.searched_twins = True
        return [("search", args)]

    def recover(self) -> None:
        if self.done or self.turn >= self.max_steps:
            return
        self.pending_question = False
        self.refresh_survivors()
        target = self.pick()
        if target is None:
            hits = self._compare_stem_hits()
            extra = hits or self.dead_search()
            if extra:
                self.send(extra[: self.max_calls])
                hits = self._compare_stem_hits()
                if hits:
                    self.send(hits[: self.max_calls])
                self.refresh_survivors()
                target = self.pick()
        if target is None and not self.spec.codes:
            extra = self.gather()
            if extra:
                self.send(extra[: self.max_calls])
                self.refresh_survivors()
                target = self.pick()
        if target:
            self.finish(target)
        elif self.late() and not self.done:
            self.finish(self.best_effort())

    def _next_after(self, key: str) -> dict | None:
        nxt = self.pick()
        if nxt is None and self.memory.adds:
            nxt = self.best_effort()
        if nxt and nxt["key"] != key:
            return nxt
        return None

    def _live_illegal(self, pid: str, sku: str, obs: dict | None = None) -> bool:
        live = self.memory.status(pid, sku) or {}
        if isinstance(obs, dict) and obs.get("error"):
            return True
        if not live.get("in_stock"):
            return True
        if self.spec.budget is not None and live.get("price") is not None and live["price"] > self.spec.budget + _EPS:
            return True
        return False

    def finish(self, target: dict | None = None) -> None:
        if self.done or self.turn >= self.max_steps:
            return
        chosen = self.pick() or target
        first_open = self.memory.adds == 0 and self.spec.universe != "closed"
        if chosen is None or (
            first_open
            and not chosen.get("ok")
            and (self.spec.codes or self.spec.hard or self.spec.phrase)
            and not self.late()
        ):
            if first_open and not self.late() and self.pick() is None:
                fallback = self.best_effort()
                classy = (not (self.spec.item or self.spec.category)) or self._class_fit(fallback["pid"]) >= 2
                if fallback and int(fallback.get("ident") or 0) > 0 and classy:
                    chosen = fallback
                elif fallback and self.spec.need_reason and self._class_fit(fallback["pid"]) >= 2:
                    chosen = fallback
                elif (
                    fallback
                    and classy
                    and self._lead_score(fallback["pid"]) >= 4
                    and not (self.spec.need_reason and self.spec.hard)
                ):
                    chosen = fallback
                else:
                    return
            else:
                chosen = chosen or self.best_effort()
        target = chosen
        if target is None:
            return
        pid, sku = target["pid"], target["sku"]
        key = Memory.key(pid, sku)
        if key in self.memory.dead or key == self.memory.event_key:
            return
        if not self.fresh(pid, sku) and "inspect_stock" in self.names and self.turn < self.max_steps:
            obs = self.act("inspect_stock", {"product_id": pid, "sku": sku})
            if self._live_illegal(pid, sku, obs):
                self.memory.dead.add(key)
                nxt = self._next_after(key)
                if nxt:
                    self.finish(nxt)
                return
        if (pid, sku) not in self.memory.cart and "add_to_cart" in self.names and self.turn < self.max_steps:
            self.act("add_to_cart", {"product_id": pid, "sku": sku})
            if self.done:
                return
            key = Memory.key(pid, sku)
            first = self.memory.adds <= 1
            if first:
                self.memory.event_key = self.memory.event_key or key
            if first or key in self.memory.dead or key == self.memory.event_key:
                if "inspect_stock" in self.names and self.turn < self.max_steps:
                    obs = self.act("inspect_stock", {"product_id": pid, "sku": sku})
                    if self._live_illegal(pid, sku, obs):
                        self.memory.dead.add(key)
                        if self.spec.universe != "closed":
                            self.memory.dead_products.add(str(pid))
                self.refresh_survivors()
                nxt = self._next_after(key)
                if nxt:
                    self.finish(nxt)
                return
        if self.turn < self.max_steps and Memory.key(pid, sku) != self.memory.event_key:
            self.act("place_test_order", self.order_args(pid, sku))

    def run(self) -> list:
        if self.remaining() > 50:
            self.interpret()
        while not self.done and self.turn < self.max_steps:
            if self.late():
                self.finish(self.pick() or self.best_effort())
                break
            if self.pending_recheck and "inspect_stock" in self.names:
                self.recheck()
                continue
            planned = self.plan()
            if planned:
                signature = planned[0][0] + json.dumps(planned[0][1], sort_keys=True)
                if signature in self._seen_calls:
                    planned = None
            if not planned:
                self.finish(self.pick() or self.best_effort())
                break
            name = planned[0][0]
            result = self.send(planned)
            for n, args in planned:
                self._seen_calls.add(n + json.dumps(args, sort_keys=True))
            if name in {"search", "filter"}:
                self.searches += 1
                self.page_reads = 0
                for n, args in planned:
                    if n == "search":
                        self.queries_used.append(str(args.get("query") or ""))
            if name in {"view", "compare"}:
                self.page_reads += 1
            if name == "message":
                self.pending_question = False
            if name == "add_to_cart" and result and not result[0].get("error"):
                added = result[0].get("added") or {}
                if added.get("product_id"):
                    self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                if result[0].get("market_event"):
                    self.memory._event(result[0]["market_event"])
        if not self.done and self.turn < self.max_steps:
            self.finish(self.pick() or self.best_effort())
        return self.dialogue


def agent_main(problem_data: dict) -> list:
    return Planner(problem_data).run()


__all__ = ["agent_main"]
