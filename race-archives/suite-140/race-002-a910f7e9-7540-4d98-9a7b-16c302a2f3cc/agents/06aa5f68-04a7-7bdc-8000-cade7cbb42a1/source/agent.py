"""ORO Bench supervisor: query-driven planner; the model writes shopper text."""

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_ALT_MODELS = {
    "chutes": ("Qwen/Qwen3-235B-A22B-Instruct-2507",),
    "openrouter": ("qwen/qwen3-235b-a22b-2507", "openai/gpt-4.1-mini"),
}
_INFER = ProxyClient(timeout=120, max_retries=2)
_ENV = ProxyClient(timeout=120, max_retries=2)

_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET = re.compile(
    r"(?:at or under|at or below|up to|no more than|not more than|max(?:imum)? of|"
    r"budget (?:is|of)|can spend up to|spend up to|at most)\s*"
    + _NUM,
    re.I,
)
_CURRENCY = re.compile(r"\b([A-Z]{3})\b")
_KEY = re.compile(r"\b([A-Za-z0-9]{3,})::([A-Za-z0-9._\-]+)")
_QUOTE = re.compile(r"\"([^\"]{2,80})\"")
_PHRASE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_HIDDEN = re.compile(
    r"tell you if you ask|if you ask|hidden (?:firm )?requirement|"
    r"one firm requirement|ask(?:s|ed)? (?:the )?(?:shopper|user)",
    re.I,
)
_PREF_ITEM = re.compile(
    r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b",
    re.I,
)
_PREF_BRAND = re.compile(
    r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)"
    r"\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)"
)
_CAT = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
_REQ_CAT = re.compile(r"required catalog category is ([A-Za-z][\w&' \-]{1,60})", re.I)
_REQ_BRAND = re.compile(
    r"required brand is ([A-Za-z][\w&']{1,24}(?:\s+[A-Z][\w&']{1,24}){0,3})",
    re.I,
)
_NEED = re.compile(
    r"\b(?:need|want|find|get|buy|order|looking for)\s+(?:an?\s+|the\s+)?"
    r"([A-Za-z][\w&\- ]{1,40}?)(?:\s+with|\s+at|\s+for|\s+that|\s+in\b|,|\.|$)",
    re.I,
)
_INTENT = re.compile(
    r"ideally\b.{0,48}\bprefer|firm requirements? matter|requirements? matter more",
    re.I,
)
_QUALITY = re.compile(
    r"(?:not|isn't|is not|don'?t)\s+(?:just\s+)?(?:grab |take )?(?:the\s+)?cheap|"
    r"cheap(?:est)? isn't|rather pay|flimsy|matters more|care(?:s)? more|"
    r"what i actually care",
    re.I,
)
_REASON = re.compile(
    r"\b(?:why|tell me why|walk me through|reason i can trust|give me a reason)\b",
    re.I,
)
_REVISE = re.compile(
    r"\b(?:instead|actually|change[ds]?|update[d]?|now (?:need|want|require)|"
    r"switch(?:ed)? to|no longer)\b",
    re.I,
)
_ASK = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|which (?:one|option)|"
    r"confirm whether|before you buy|still available|out of stock|more expensive)\b",
    re.I,
)
_MEASURE = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(inches|inch|in|hz|gb|tb|mm|p)\b",
    re.I,
)
_AXIS = re.compile(
    r"\b(highest|largest|most|lowest|smallest|cheapest)\s+([a-z][a-z \-]{0,24}?)(?:\.|,|;|$)",
    re.I,
)
_RES_NUM = re.compile(r"\b(\d{3,4}p|\d+k)\b", re.I)
_RES_HD = re.compile(r"\b([a-z]hd|hd)\b", re.I)
_RES_PHRASE = re.compile(r"\b(full|quad|ultra)[\s-]+hd\b", re.I)
_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)")
_REJECT = re.compile(
    r"\bnot (?:an? )([a-z][a-z \-]{2,48}?)(?:\s+listing\b|,|\.|$)",
    re.I,
)
_PREF_SPAN = re.compile(
    r"(?:rather (?:pay[^.]*?for|than)|care(?:s)?(?: more)? about|"
    r"what i actually care about)\s+"
    r"([^.;]{3,80}?)(?:\s+than|\s+that|\.|$)",
    re.I,
)
_STOP = {
    "a", "an", "the", "and", "or", "for", "with", "of", "to", "on", "at", "in",
    "is", "it", "this", "that", "i", "me", "my", "need", "want", "find", "get",
    "buy", "one", "can", "up", "under", "below", "stock", "in-stock",
}
_GENERIC_NOUNS = {
    "order", "listing", "listings", "item", "items", "product", "products",
    "one", "test", "stock", "screen", "spec", "requirement", "budget", "color",
    "model", "option", "options", "variant", "variants", "place",
    "grab", "cheap", "cheapest", "just", "rather", "settle", "flimsy",
    "saving", "dollar", "dollars",
}
_LEAD_CUT = re.compile(
    r"\b(?:walk me through|tell me why|give me a reason|reason i can trust|"
    r"for a \w+, the title|category labels|reject conflicting)\b",
    re.I,
)
_ITEM_LIMIT = re.compile(
    r"\b([A-Za-z][A-Za-z&'\-]{2,40})\s+(?:at or under|at or below|at or\b|in stock)",
    re.I,
)
_PREF_FOR = re.compile(r"rather pay[^.]*?\bfor\s+([^.;]{3,80}?)\s+than", re.I)
_PURPOSE = re.compile(
    r"\b(?:for|during)\s+([^.;]{6,70}?)(?:\s+-\s+|\s+i'd\b|\s+i would\b|\.|$)",
    re.I,
)
_PREF_IS = re.compile(r"([^.;]{8,80})\s+is what i actually care", re.I)
_PREF_MATTERS = re.compile(
    r"(?:^|[.!?;]\s+|-\s+)([^.;]{8,80}?)\s+matters more",
    re.I,
)
_EPS = 1e-9


def _model() -> str:
    return _model_stack()[0]


def _model_stack() -> list[str]:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    primary = getenv("SANDBOX_MODEL") or _MODELS.get(provider, _MODELS["chutes"])
    found: list[str] = [primary]
    alt = getenv("SANDBOX_MODEL_ALT") or ""
    if alt:
        found.append(alt)
    for name in _ALT_MODELS.get(provider, ()):
        if name not in found:
            found.append(name)
    return found


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _vid(pid: Any, sku: Any) -> str:
    return "%s::%s" % (pid, sku or "")


def _sku_family(sku: Any) -> str:
    text = str(sku or "")
    if "-" not in text:
        return text
    head, tail = text.rsplit("-", 1)
    if head and (tail.isdigit() or tail.lower() == "default" or len(tail) <= 8):
        return head
    return text


def _pref_hit(blob: str, term: str) -> bool:
    term = (term or "").lower()
    text = (blob or "").lower()
    if not term:
        return False
    if term in text:
        return True
    if len(term) < 4:
        return False
    return any(word.startswith(term) for word in re.findall(r"[a-z0-9]+", text))


def _pref_word_hit(blob: str, term: str) -> bool:
    term = (term or "").lower()
    if not term:
        return False
    return bool(
        re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", (blob or "").lower())
    )


def _has_token(blob: str, token: str) -> bool:
    if not token:
        return False
    return bool(
        re.search(
            r"(?<![a-z0-9])" + re.escape(str(token).lower()) + r"(?![a-z0-9])",
            (blob or "").lower(),
        )
    )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _content_tokens(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", (text or "").lower()) if tok not in _STOP]


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _clip(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 16] + "...[truncated]"


def _tool_name(schema: dict[str, Any]) -> str:
    fn = schema.get("function") if isinstance(schema, dict) else None
    if isinstance(fn, dict):
        return str(fn.get("name") or "")
    return str(schema.get("name") or "") if isinstance(schema, dict) else ""


def _tool_fn(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    fn = schema.get("function")
    return fn if isinstance(fn, dict) else schema


def _tool_params(schema: dict[str, Any]) -> dict[str, Any]:
    params = _tool_fn(schema).get("parameters")
    return params if isinstance(params, dict) else {}


def _tool_props(schema: dict[str, Any]) -> set[str]:
    props = _tool_params(schema).get("properties")
    return set(props) if isinstance(props, dict) else set()


def _clip_args(args: dict[str, Any], allowed: set[str] | None) -> dict[str, Any]:
    if allowed is None:
        return dict(args)
    return {key: value for key, value in args.items() if key in allowed}


def _unit(raw: str) -> str:
    unit = (raw or "").lower()
    if unit in {"inch", "inches", "in"}:
        return "in"
    return unit


def _normalize_catalog(text: str) -> str:
    return re.sub(r'(\d+(?:\.\d+)?)\s*"', r"\1 inch", text or "")


def _res_tokens(text: str) -> list[str]:
    raw = text or ""
    found: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = (token or "").lower()
        if token and token not in seen:
            seen.add(token)
            found.append(token)

    for match in _RES_NUM.finditer(raw):
        add(match.group(1))
    covered: list[tuple[int, int]] = []
    for match in _RES_PHRASE.finditer(raw):
        add(match.group(1)[0] + "hd")
        covered.append((match.start(), match.end()))
    for match in _RES_HD.finditer(raw):
        if any(start <= match.start() and match.end() <= end for start, end in covered):
            continue
        add(match.group(1))
    return found


def _measures(text: str) -> list[tuple[float, str, str]]:
    found: list[tuple[float, str, str]] = []
    for match in _MEASURE.finditer(_normalize_catalog(text)):
        value = float(match.group(1))
        unit = _unit(match.group(2))
        found.append((value, unit, match.group(0).lower()))
    return found


def _measure_forms(item: tuple[float, str, str]) -> list[str]:
    value, unit, raw = item
    compact = "%g" % value
    forms = [raw, raw.replace("-", " "), raw.replace(" ", "")]
    if unit == "in":
        forms.extend([compact + "in", compact + " inch", compact + "-inch"])
    elif unit == "hz":
        forms.extend([compact + "Hz", compact + "hz", compact + " Hz"])
    elif unit == "p":
        forms.extend([compact + "p"])
    else:
        forms.append(compact + unit)
    out: list[str] = []
    seen: set[str] = set()
    for form in forms:
        text = form.strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _measure_match(want: tuple[float, str, str], blob: str) -> bool:
    value, unit, raw = want
    have = _measures(blob)
    for got_value, got_unit, _got_raw in have:
        if got_unit == unit and round(got_value, 2) == round(value, 2):
            return True
    compact = raw.replace(" ", "")
    return compact in re.sub(r"\s+", "", (blob or "").lower())


def _measure_match_exact(want: tuple[float, str, str], blob: str) -> bool:
    if not _measure_match(want, blob):
        return False
    stripped = _RANGE.sub(" ", blob or "")
    if _measure_match(want, stripped):
        return True
    return False


def _axis_number(blob: str, axis: str) -> float:
    text = _normalize_catalog(blob or "")
    axis_toks = _content_tokens(axis)
    measures = list(_MEASURE.finditer(text))
    if not measures:
        return 0.0
    first = float(measures[0].group(1))
    axis_l = (axis or "").lower()
    prefer = None
    if any(word in axis_l for word in ("refresh", "hz", "rate")):
        prefer = "hz"
    elif any(word in axis_l for word in ("screen", "size", "inch")):
        prefer = "in"
    unit_hits = [
        float(match.group(1))
        for match in measures
        if prefer and _unit(match.group(2)) == prefer
    ]
    if not axis_toks:
        return unit_hits[0] if unit_hits else first
    best: tuple[int, int, float] | None = None
    low = text.lower()
    for tok in axis_toks:
        for token in re.finditer(r"(?<![a-z0-9])" + re.escape(tok) + r"(?![a-z0-9])", low):
            for match in measures:
                before = 0 if match.start() >= token.start() else 1
                dist = min(
                    abs(match.start() - token.end()),
                    abs(match.end() - token.start()),
                    abs(match.start() - token.start()),
                )
                cand = (dist, before, float(match.group(1)))
                if best is None or cand[:2] < best[:2]:
                    best = cand
    if best is not None:
        return best[2]
    if prefer:
        return unit_hits[0] if unit_hits else 0.0
    return unit_hits[0] if unit_hits else first


def _unwrap(step: Any) -> tuple[dict[str, Any], bool, str | None]:
    if not isinstance(step, dict):
        return {}, False, None
    done = bool(step.get("done"))
    error = step.get("error")
    body = step.get("observation")
    if isinstance(body, dict):
        done = done or bool(body.get("done"))
        error = error or body.get("error")
        if any(
            key in body
            for key in ("results", "cart", "added", "compared", "product_id", "variants", "facts")
        ):
            return body, done, error if isinstance(error, str) else None
        nested = body.get("observation")
        if isinstance(nested, dict):
            return nested, done or bool(nested.get("done")), error if isinstance(error, str) else None
        return body, done, error if isinstance(error, str) else None
    return step, done, error if isinstance(error, str) else None


def _walk_dicts(obj: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    stack = [obj]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        ident = id(cur)
        if ident in seen:
            continue
        seen.add(ident)
        if isinstance(cur, dict):
            found.append(cur)
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return found


# --------------------------------------------------------------------------- goal


class Goal:
    """Public query only. Constraints are extracted from the text, not a catalog."""

    def __init__(self, raw: str) -> None:
        text = raw or ""
        goal, _, rules = text.partition("\n\nTask rules:")
        self.raw = text
        self.text = goal.strip()
        self.rules = rules.lower()
        self.budget = None
        for match in _BUDGET.finditer(self.text):
            whole = match.group(1).replace(",", "")
            frac = match.group(2) or "0"
            try:
                value = float(whole + "." + frac)
            except ValueError:
                continue
            if self.budget is None or value < self.budget:
                self.budget = value
        self.opening_budget = self.budget
        self.currency = ""
        cur = _CURRENCY.search(self.text)
        if cur:
            self.currency = cur.group(1)
        self.keys = [(pid, sku.rstrip(".,;:")) for pid, sku in _KEY.findall(self.text)]
        phrase = _PHRASE.search(self.text)
        quoted = _QUOTE.findall(self.text)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        self.theme = re.findall(r"[a-z0-9]+", source.lower())
        pref = _PREF_ITEM.search(self.text)
        self.color = pref.group(1).lower() if pref else None
        self.model = pref.group(2) if pref else None
        cat = _CAT.search(self.text)
        self.category = cat.group(1).strip() if cat else None
        self.item = _parse_category(self.text)
        self.brand = _parse_brand(self.text, self.item or self.category)
        self.pref_brand = None if pref else _parse_pref_brand(self.text)
        self.res_tokens = _res_tokens(self.text)
        self.measures = [item for item in _measures(self.text) if item[1] != "p"]
        self.hidden = not self.measures and not self.res_tokens and bool(_HIDDEN.search(text))
        self.wants_reason = bool(_REASON.search(self.text))
        self.intent_cue = bool(_INTENT.search(self.text))
        self.pref_spans = _pref_spans(self.text)
        self.pref_terms = _pref_terms(self.text)
        self.purpose_terms = _purpose_terms(self.text)
        self.soft_pref = bool(_QUALITY.search(self.text) or self.pref_terms) and not self.intent_cue
        self.reject_terms = _reject_terms(self.text)
        self.rank_axes = _rank_axes(self.text + " " + self.rules)
        self.family = infer_family(self)

    def theme_tokens_in(self, *texts: str) -> bool:
        if not self.theme:
            return True
        have: set[str] = set()
        for text in texts:
            have |= _tokens(text)
        return all(token in have for token in self.theme)

    def adopt(self, text: str) -> str | None:
        changed: list[str] = []
        for match in _BUDGET.finditer(text or ""):
            whole = match.group(1).replace(",", "")
            frac = match.group(2) or "0"
            try:
                value = float(whole + "." + frac)
            except ValueError:
                continue
            if self.budget is None or value + _EPS < self.budget:
                if self.opening_budget is None:
                    self.opening_budget = value
                self.budget = value
                changed.append("%.2f" % value)
        incoming = [item for item in _measures(text or "") if item[1] != "p"]
        have = {(value, unit) for value, unit, _raw in self.measures}
        for item in incoming:
            key = (item[0], item[1])
            if key in have:
                continue
            self.measures.append(item)
            have.add(key)
            changed.append(item[2])
        for token in _res_tokens(text or ""):
            if token not in self.res_tokens:
                self.res_tokens.append(token)
                changed.append(token)
        if incoming or self.res_tokens:
            self.hidden = False
        cat = _REQ_CAT.search(text or "") or _CAT.search(text or "")
        if cat:
            name = cat.group(1).strip()
            if name:
                self.category = name
                changed.append(name)
        brand = _REQ_BRAND.search(text or "")
        if brand:
            name = brand.group(1).strip(" .,")
            if name:
                self.brand = name
                changed.append(name)
        pref = _PREF_ITEM.search(text or "")
        if pref:
            color = pref.group(1).lower()
            model = pref.group(2)
            if color and color != self.color:
                self.color = color
                changed.append(color)
            if model and model != self.model:
                self.model = model
                changed.append(model)
        for term in _reject_terms(text or ""):
            if term not in self.reject_terms:
                self.reject_terms.append(term)
                changed.append(term)
        return ", ".join(changed) or None


def _usable_noun(word: str | None) -> str | None:
    if not word:
        return None
    clean = word.strip(".,;:")
    if not clean or clean.lower() in _STOP or clean.lower() in _GENERIC_NOUNS:
        return None
    return clean


def _parse_category(goal: str) -> str | None:
    found: list[str] = []
    for match in _NEED.finditer(goal or ""):
        words = [w for w in match.group(1).split() if w.lower() not in _STOP]
        if words:
            found.append(words[-1].strip(".,"))
    for match in _ITEM_LIMIT.finditer(goal or ""):
        found.append(match.group(1))
    for word in reversed(found):
        usable = _usable_noun(word)
        if usable:
            return usable
    return None


def _parse_brand(goal: str, category: str | None) -> str | None:
    if not category:
        return None
    match = re.search(
        r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category),
        goal or "",
        re.I,
    )
    if not match:
        match = re.search(
            r"\b([A-Z][\w&.]{1,24})\s+" + re.escape(category),
            goal or "",
            re.I,
        )
    if not match:
        return None
    words = [w for w in re.findall(r"[\w&.'\-]+", match.group(1))]
    keep: list[str] = []
    for word in words:
        if any(ch.isdigit() for ch in word):
            break
        if word.isupper() and len(word) <= 3:
            break
        if word[:1].isupper() and word.lower() not in _STOP:
            keep.append(word)
        elif keep:
            break
    return " ".join(keep) or None


def _parse_pref_brand(goal: str) -> str | None:
    match = _PREF_BRAND.search(goal or "")
    if not match:
        return None
    name = match.group(1).strip(" .,")
    words = name.split()
    if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
        return None
    return name


def _pref_spans(text: str) -> list[str]:
    spans: list[str] = []
    seen: set[str] = set()
    for pattern in (_PREF_FOR, _PREF_IS, _PREF_MATTERS, _PREF_SPAN):
        for match in pattern.finditer(text or ""):
            span = re.sub(r"\s+", " ", match.group(1).strip())
            key = span.lower()
            if 3 <= len(span) <= 80 and key not in seen:
                seen.add(key)
                spans.append(span)
    return spans


def _pref_terms(text: str) -> list[str]:
    spans: list[str] = list(_pref_spans(text))
    terms: list[str] = []
    seen: set[str] = set()
    for span in spans:
        for sep in (" - ", " — ", ": ", " -"):
            if sep in span:
                span = span.split(sep)[-1]
        for tok in _content_tokens(span):
            if tok not in seen and tok not in _GENERIC_NOUNS:
                seen.add(tok)
                terms.append(tok)
    return terms


def _purpose_terms(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for match in _PURPOSE.finditer(text or ""):
        for tok in _content_tokens(match.group(1)):
            if tok in _STOP or tok in _GENERIC_NOUNS or len(tok) < 4:
                continue
            if tok not in seen:
                seen.add(tok)
                found.append(tok)
    return found


def _shopper_lead(text: str) -> str:
    goal, _, _rules = (text or "").partition("\n\nTask rules:")
    cut = _LEAD_CUT.search(goal) or _REASON.search(goal)
    head = goal[: cut.start()] if cut else goal
    parts = re.split(r"(?<=[.!?])\s+", head.strip(), maxsplit=1)
    return parts[0] if parts else head


def _reject_terms(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for match in _REJECT.finditer(_shopper_lead(text)):
        span = match.group(1).strip()
        for chunk in re.split(r"\s+or\s+", span):
            chunk = re.sub(r"\s+listing$", "", chunk.strip().lower())
            if len(chunk) < 3 or chunk in seen:
                continue
            seen.add(chunk)
            terms.append(chunk)
    return terms


def _rank_axes(text: str) -> list[tuple[str, int]]:
    axes: list[tuple[str, int]] = []
    down = {"lowest", "smallest", "cheapest"}
    for match in _AXIS.finditer(text or ""):
        name = match.group(2).strip()
        direction = -1 if match.group(1).lower() in down else 1
        if name:
            axes.append((name, direction))
    return axes


def infer_family(goal: Goal) -> str:
    """Optional label for logs. The planner does not dispatch on this."""

    if goal.keys:
        return "ranking"
    if goal.model:
        return "recovery"
    if goal.theme:
        return "retrieval"
    if goal.hidden:
        return "constraint"
    if goal.soft_pref and not goal.intent_cue:
        return "preference"
    if goal.intent_cue or (goal.pref_brand and goal.measures):
        return "intent"
    if goal.wants_reason:
        return "justification"
    if goal.measures or goal.brand:
        return "constraint"
    return "fallback"


def parse_goal(text: str) -> Goal:
    return Goal(text)


# --------------------------------------------------------------------------- observation index


class ObservationIndex:
    def __init__(self) -> None:
        self.hits: dict[str, dict[str, Any]] = {}
        self.pages: dict[str, dict[str, Any]] = {}
        self.quotes: dict[str, dict[str, Any]] = {}
        self.options: dict[str, str] = {}
        self.listings: dict[str, str] = {}
        self.clusters: set[str] = set()
        self.inspected: set[str] = set()
        self.viewed: set[str] = set()
        self.cart: list[tuple[str, str]] = []
        self.queries: list[str] = []
        self.filters: list[dict[str, Any]] = []
        self.messages: list[str] = []
        self.first_add: tuple[str, str] | None = None
        self.events: list[str] = []
        self.dead: set[str] = set()

    def ingest(self, name: str, args: dict[str, Any], body: dict[str, Any], goal: Goal) -> None:
        if not isinstance(body, dict) or body.get("error"):
            return
        if name in ("search", "filter"):
            query = str(args.get("query") or "")
            if query:
                self.queries.append(query)
            if name == "filter":
                self.filters.append(dict(args or {}))
            for row in body.get("results") or []:
                if isinstance(row, dict):
                    self._hit(row, goal)
        elif name == "view":
            self._page(body, goal)
        elif name == "compare":
            compared = body.get("compared")
            if not isinstance(compared, list):
                compared = [
                    item
                    for item in _walk_dicts(body)
                    if item.get("product_id") and (item.get("facts") or item.get("variants"))
                ]
            for product in compared or []:
                if isinstance(product, dict):
                    self._page(product, goal)
        elif name == "inspect_stock":
            pid = str(body.get("product_id") or args.get("product_id") or "")
            sku = str(body.get("sku") or args.get("sku") or "")
            key = _vid(pid, sku)
            self.quotes[key] = {
                "in_stock": bool(body.get("in_stock")),
                "price": _num(body.get("price")),
            }
            self.inspected.add(key)
            if pid and pid not in self.pages:
                self.pages[pid] = {"product_id": pid}
        elif name == "add_to_cart":
            added = body.get("added") or {}
            pid, sku = str(added.get("product_id") or ""), str(added.get("sku") or "")
            if pid and sku:
                if self.first_add is None:
                    self.first_add = (pid, sku)
                if (pid, sku) not in self.cart:
                    self.cart.append((pid, sku))
                key = _vid(pid, sku)
                if added.get("in_stock") is False:
                    self.dead.add(key)
                    self.quotes[key] = {
                        "in_stock": False,
                        "price": _num(added.get("price") or added.get("current_price")),
                    }
            self._cart(body)
            event = body.get("market_event")
            if event:
                self.events.append(str(event))
            for row in body.get("cart") or []:
                if not isinstance(row, dict):
                    continue
                key = _vid(row.get("product_id"), row.get("sku"))
                if row.get("in_stock") is False or row.get("current_in_stock") is False:
                    self.dead.add(key)
                price = _num(row.get("current_price") or row.get("price"))
                if goal.budget is not None and price is not None and price > goal.budget + _EPS:
                    self.dead.add(key)
        elif name == "inspect_cart":
            self._cart(body)
        elif name == "remove_from_cart":
            self._cart(body)
        elif name == "message":
            content = str(args.get("content") or "")
            if content:
                self.messages.append(content)

    def _hit(self, row: dict[str, Any], goal: Goal) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        pid = str(pid)
        sku = str(row.get("sku") or "")
        key = _vid(pid, sku)
        self.hits[key] = row
        if row.get("listing_id"):
            self.listings[pid] = str(row["listing_id"])
        title = str(row.get("title") or "")
        if not goal.theme or goal.theme_tokens_in(title, str(row.get("brand") or "")):
            self.clusters.add(pid)
        price = _num(row.get("price"))
        if sku and price is not None:
            self.quotes.setdefault(key, {"in_stock": bool(row.get("in_stock", True)), "price": price})

    def _page(self, product: dict[str, Any], goal: Goal) -> None:
        if not product.get("product_id") and isinstance(product.get("product"), dict):
            product = product["product"]
        pid = product.get("product_id")
        if not pid:
            return
        pid = str(pid)
        self.pages[pid] = product
        self.viewed.add(pid)
        if product.get("listing_id"):
            self.listings[pid] = str(product["listing_id"])
        facts = product.get("facts") or {}
        fact_blob = " ".join(str(v) for v in facts.values()) if isinstance(facts, dict) else ""
        if not goal.theme or goal.theme_tokens_in(str(product.get("title") or ""), fact_blob):
            self.clusters.add(pid)
        for variant in product.get("variants") or []:
            if not isinstance(variant, dict) or not variant.get("sku"):
                continue
            key = _vid(pid, variant.get("sku"))
            self.quotes[key] = {
                "in_stock": bool(variant.get("in_stock", True)),
                "price": _num(variant.get("price")),
            }
            self.options[key] = str(variant.get("options") or "")

    def _cart(self, body: dict[str, Any]) -> None:
        rows = body.get("cart")
        if not isinstance(rows, list):
            return
        pairs = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("product_id"):
                continue
            pid, sku = str(row["product_id"]), str(row.get("sku") or "")
            pairs.append((pid, sku))
            key = _vid(pid, sku)
            price = _num(row.get("current_price") or row.get("price"))
            stock = row.get("current_in_stock")
            if stock is None:
                stock = row.get("in_stock")
            prev = self.quotes.get(key, {})
            live_stock = bool(stock) if stock is not None else prev.get("in_stock", True)
            live_price = price if price is not None else prev.get("price")
            if key in self.dead:
                live_stock = False
            if prev.get("in_stock") and not live_stock:
                self.dead.add(key)
                self.events.append("stockout:%s" % key)
            if prev.get("price") is not None and live_price is not None and live_price > prev["price"] + _EPS:
                self.events.append("price:%s" % key)
            self.quotes[key] = {"in_stock": live_stock, "price": live_price}
        self.cart = pairs

    def title(self, pid: str) -> str:
        page = self.pages.get(str(pid)) or {}
        if page.get("title"):
            return str(page["title"])
        for key, row in self.hits.items():
            if key.startswith(str(pid) + "::") or str((row or {}).get("product_id")) == str(pid):
                return str(row.get("title") or "")
        return ""

    def facts(self, pid: str) -> dict[str, Any]:
        page = self.pages.get(str(pid)) or {}
        facts = page.get("facts")
        return facts if isinstance(facts, dict) else {}

    def path_text(self, pid: str) -> str:
        page = self.pages.get(str(pid)) or {}
        path = page.get("category_path") or page.get("category") or ""
        if isinstance(path, list):
            return " ".join(str(part) for part in path if part)
        hit = {}
        for key, row in self.hits.items():
            if key.startswith(str(pid) + "::") or str((row or {}).get("product_id")) == str(pid):
                hit = row or {}
                break
        extra = hit.get("category_path") or hit.get("category") or ""
        if isinstance(extra, list):
            extra = " ".join(str(part) for part in extra if part)
        return ("%s %s" % (path, extra)).strip()

    def brand_of(self, pid: str) -> str:
        page = self.pages.get(str(pid)) or {}
        if page.get("brand"):
            return str(page["brand"])
        for key, row in self.hits.items():
            if key.startswith(str(pid) + "::") or str((row or {}).get("product_id")) == str(pid):
                if row.get("brand"):
                    return str(row["brand"])
        return ""

    def blob(self, pid: str, sku: str = "") -> str:
        page = self.pages.get(str(pid)) or {}
        parts = [
            self.title(pid),
            self.path_text(pid),
            self.brand_of(pid),
            str(self.listings.get(str(pid)) or ""),
            sku,
            " ".join(self._fact_text(key, value) for key, value in self.facts(pid).items() if value),
            self.options.get(_vid(pid, sku), ""),
            str(page.get("brand") or ""),
        ]
        hit = self.hits.get(_vid(pid, sku)) or {}
        parts.append(str(hit.get("title") or ""))
        parts.append(str(hit.get("brand") or ""))
        return " ".join(parts)

    def _fact_text(self, key: str, value: Any) -> str:
        raw = str(value).strip()
        if not raw:
            return ""
        name = (key or "").lower().replace("_", " ")
        if _MEASURE.search(_normalize_catalog(raw)):
            return "%s %s" % (name, raw)
        if any(word in name for word in ("size", "screen", "inch")):
            return "%s %s inch" % (name, raw)
        if any(word in name for word in ("refresh", "hz", "rate")):
            return "%s %s Hz" % (name, raw)
        return "%s %s" % (name, raw)

    def quote(self, pid: str, sku: str) -> dict[str, Any] | None:
        return self.quotes.get(_vid(pid, sku))

    def live(self, pid: str, sku: str, budget: float | None) -> bool:
        quote = self.quote(pid, sku)
        if not quote or not quote.get("in_stock"):
            return False
        price = quote.get("price")
        if price is None:
            return False
        if budget is not None and price > budget + _EPS:
            return False
        return _vid(pid, sku) not in self.dead


# --------------------------------------------------------------------------- protocol kernel


class ProtocolKernel:
    def __init__(self, problem: dict[str, Any]) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        policy = env["policy_view"]
        self.policy = policy
        self.tools_schema = list(policy.get("tools") or [])
        self.tools = {_tool_name(item) for item in self.tools_schema}
        self.tool_props = {
            _tool_name(item): _tool_props(item)
            for item in self.tools_schema
            if _tool_name(item)
        }
        self.tool_params = {
            _tool_name(item): _tool_params(item)
            for item in self.tools_schema
            if _tool_name(item)
        }
        self.max_steps = int(policy.get("max_steps") or 20)
        self.max_calls = max(1, int(policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.turn = 0
        self.dialogue: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Help the shopper using only facts already observed. "
                    "Do not invent product IDs. Write concise shopper messages."
                ),
            },
            {"role": "user", "content": policy.get("query") or ""},
        ]

    def has(self, name: str) -> bool:
        return name in self.tools

    def allows(self, name: str, field: str) -> bool:
        props = self.tool_props.get(name)
        return props is None or field in props

    def param(self, name: str) -> dict[str, Any]:
        return self.tool_params.get(name) or {}

    def prop(self, name: str, field: str) -> dict[str, Any]:
        props = self.param(name).get("properties") or {}
        spec = props.get(field)
        return spec if isinstance(spec, dict) else {}

    def cap(self, name: str, field: str, want: int) -> int | None:
        if not self.allows(name, field):
            return None
        spec = self.prop(name, field)
        lo = spec.get("minimum")
        hi = spec.get("maximum") or spec.get("maxItems")
        default = spec.get("default")
        value = int(default) if default is not None else want
        if lo is not None:
            value = max(value, int(lo))
        if hi is not None:
            value = min(value, int(hi))
        return max(1, value)

    def clip(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return _clip_args(args, self.tool_props.get(name))

    def infer(self, messages: list[dict[str, Any]], tools: bool = False) -> dict[str, Any]:
        last_error = None
        for model in _model_stack():
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": 0,
            }
            if tools:
                payload["tools"] = self.tools_schema
                payload["tool_choice"] = "none"
            result = _INFER.post("/inference/chat/completions", json_data=payload)
            if result is not None and (result.get("choices") or result.get("error") is None):
                if result.get("choices"):
                    return result
            last_error = result
        if last_error is None:
            raise RuntimeError("inference request failed")
        return last_error

    def call(self, actions: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        actions = [
            (name, self.clip(name, args))
            for name, args in actions
            if self.has(name)
        ][: self.max_calls]
        if not actions:
            raise RuntimeError("no available environment tools for this turn")
        self.turn += 1
        group_id = "%s-turn-%d" % (self.problem_id, self.turn)
        envelope = {
            **self.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": self.turn,
            "calls": [
                {
                    "call_id": "%s-%d" % (group_id, index),
                    "action": {"name": name, "args": args},
                }
                for index, (name, args) in enumerate(actions, start=1)
            ],
        }
        result = _ENV.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        tool_calls = [
            {
                "id": "call-%s-%d" % (group_id, index),
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for index, (name, args) in enumerate(actions, start=1)
        ]
        self.messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        self.dialogue.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )
        for tool_call, wrap in zip(tool_calls, result.get("calls") or [], strict=False):
            obs = wrap.get("observation") if isinstance(wrap, dict) else {}
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(obs),
                }
            )
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            self.messages.append({"role": "user", "content": user["content"]})
        return result


class HelperLLM:
    def __init__(self, kernel: ProtocolKernel) -> None:
        self.kernel = kernel

    def text(self, instruction: str, facts: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Reply with a short shopping message. Use only the supplied facts. "
                    "Do not invent IDs, prices, or attributes."
                ),
            },
            {
                "role": "user",
                "content": instruction + "\n\nFacts:\n" + _clip(facts, 4000),
            },
        ]
        try:
            result = self.kernel.infer(messages)
            content = ((result.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            return str(content).strip()
        except Exception:
            return ""

    def search_hints(self, goal: Goal) -> list[str]:
        drafted = self.text(
            "Write 3 short catalog search queries. Use only words from the shopper text. "
            "No product IDs, no rating codes, no invented brands.",
            goal.text,
        )
        if not drafted:
            return []
        allowed = set(_content_tokens(goal.text))
        out: list[str] = []
        for line in drafted.splitlines():
            clean = re.sub(r"^[\d\-\.\*\)]+\s*", "", line).strip().strip('"')
            words = [tok for tok in _content_tokens(clean) if tok in allowed or tok.isdigit()]
            if len(words) >= 2:
                out.append(" ".join(words[:8]))
        return out[:3]


# --------------------------------------------------------------------------- planner


class TimeBox:
    """Reserve the last turns to re-check the cart and place one order.

    Compare accepts several product_ids in one call, so survey width is not
    tied to max_calls_per_turn (live packs often use 1).
    """

    def __init__(self, kernel: ProtocolKernel, goal: Goal | None = None) -> None:
        self.kernel = kernel
        self.goal = goal

    def left(self) -> int:
        return max(0, self.kernel.max_steps - self.kernel.turn)

    def reserve(self) -> int:
        need = 4 if self.kernel.max_steps <= 22 else 5
        if self.goal and (self.goal.wants_reason or self.goal.keys):
            need += 1
        return min(need, max(3, self.kernel.max_steps // 3))

    def closing(self) -> bool:
        return self.left() <= self.reserve()

    def urgent(self) -> bool:
        return self.left() <= 2

    def can_search(self) -> bool:
        return self.left() > self.reserve()

    def can_llm(self) -> bool:
        return self.left() > self.reserve() + 1

    def survey_k(self) -> int:
        width = 5
        capped = self.kernel.cap("compare", "product_ids", width)
        if capped is not None:
            width = capped
        if self.urgent():
            return 1
        if self.closing():
            return min(2, width)
        return width

    def pref_open_searches(self) -> int:
        return 2 if self.kernel.max_steps <= 22 else 3


class Supervisor:
    def __init__(self, problem: dict[str, Any]) -> None:
        self.kernel = ProtocolKernel(problem)
        self.goal = parse_goal(str(self.kernel.policy.get("query") or ""))
        self.index = ObservationIndex()
        self.helper = HelperLLM(self.kernel)
        self.clock = TimeBox(self.kernel, self.goal)
        self.pending_q = False
        self.asked_hidden = False
        self.justified = False
        self.committed_turn: int | None = None
        self.rechecked = False
        self.post_event_surveyed = False
        self.done = False
        self.last_error: tuple[str, str, dict[str, Any]] | None = None
        self.failed_orders: set[str] = set()
        self.need_order_claims = self._order_needs_claims()
        self.idle_inspects = 0

    def _order_needs_claims(self) -> bool:
        if self.goal.wants_reason:
            return True
        required = self.kernel.param("place_test_order").get("required") or []
        return "justification" in required or "claims" in required

    def run(self) -> list[dict[str, Any]]:
        while self.kernel.turn < self.kernel.max_steps and not self.done:
            actions = self.next_actions()
            if not actions:
                target = self._choose_target()
                if (
                    not target
                    and self.clock.closing()
                    and self.index.first_add is None
                ):
                    target = self.opening_bait()
                if target and target not in self.index.cart:
                    actions = self.commit(target)
                elif target and self.can_order(target):
                    actions = self.close_actions(target)
            if not actions:
                search = self.next_search()
                if search:
                    unread = []
                    if self.index.first_add is not None:
                        unread = self.inspect_unread(
                            [pid for pid in self.survey_pids() if pid not in self.index.viewed][
                                : self.clock.survey_k()
                            ]
                        )
                    actions = (unread + [search])[: self.kernel.max_calls]
                else:
                    actions = self.idle_actions()
                if not actions and self.index.first_add is None:
                    fallback = self.opening_bait()
                    if fallback:
                        actions = self.commit(fallback)
                if not actions:
                    break
            if actions[0][0] == "inspect_cart":
                self.idle_inspects += 1
            else:
                self.idle_inspects = 0
            result = self.kernel.call(actions)
            self._ingest_result(actions, result)
            if self._episode_done(result):
                self.done = True
                break
        return self.kernel.dialogue

    def _ingest_result(
        self, actions: list[tuple[str, dict[str, Any]]], result: dict[str, Any]
    ) -> None:
        wraps = result.get("calls") or []
        for (name, args), wrap in zip(actions, wraps):
            body, done, error = _unwrap(wrap.get("observation") if isinstance(wrap, dict) else wrap)
            if error:
                self.last_error = (name, str(error), args)
                if name == "place_test_order":
                    self.failed_orders.add(_vid(args.get("product_id"), args.get("sku")))
                continue
            self.index.ingest(name, args, body, self.goal)
            if name == "add_to_cart":
                self.rechecked = False
                self.committed_turn = self.kernel.turn
                self.failed_orders.discard(_vid(args.get("product_id"), args.get("sku")))
            if name == "remove_from_cart":
                self.rechecked = False
            if done:
                self.done = True
            if isinstance(body, dict) and body.get("market_event"):
                self.index.events.append(str(body["market_event"]))
                self.rechecked = False
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            adopted = self.goal.adopt(text)
            asked = bool(_ASK.search(text) or _REVISE.search(text))
            if self.index.first_add is not None and asked:
                self.pending_q = True
            if adopted or _REVISE.search(text):
                self.rechecked = False
            if self.index.first_add is not None and re.search(r"\bcompar", text, re.I):
                self.post_event_surveyed = False

    def _episode_done(self, result: dict[str, Any]) -> bool:
        for wrap in result.get("calls") or []:
            _body, done, _error = _unwrap(wrap.get("observation") if isinstance(wrap, dict) else wrap)
            if done:
                return True
        return bool(result.get("done"))

    def next_actions(self) -> list[tuple[str, dict[str, Any]]]:
        if self.pending_q:
            return self.reply_actions()
        if self.goal.hidden and not self.asked_hidden:
            return self.ask_hidden()
        repair = self.repair_order()
        if repair:
            return repair
        target = self.pick_target()
        held = self.after_commit(target)
        if held:
            return held
        ready = bool(target and self.ready_to_commit(target))
        if self.need_search() and not self.index.hits:
            search = self.next_search()
            if search:
                return [search]
        if (
            self.need_search()
            and self.index.first_add is None
            and not ready
            and (
                (self.goal.theme and len(self.index.clusters) < 3)
                or (
                    self.goal.soft_pref
                    and len(self.index.queries) < self.clock.pref_open_searches()
                    and len(set(self.index.listings.values())) < 2
                )
            )
        ):
            search = self.next_search()
            if search:
                return [search]
        if (
            self.index.first_add is None
            and self.kernel.has("filter")
            and not self.index.filters
            and self.index.queries
            and not (self.goal.model or self.goal.theme)
            and (
                self.goal.brand
                or self.goal.pref_brand
                or (self.goal.measures and self.index.queries)
            )
        ):
            return [("filter", self.filter_args())]
        if self._budget_filter_stale():
            return [("filter", self.filter_args())]
        if not ready:
            unread = self.inspect_unread(self.survey_pids()[: self.clock.survey_k()])
            if unread:
                return unread
        if (
            self.need_search()
            and self.index.first_add is None
            and not ready
            and self._want_pair()
            and len(self.eligible_pairs()) < 2
        ):
            search = self.next_search()
            if search:
                return [search]
        if (
            self.index.first_add is None
            and not self.goal.keys
            and not self.goal.theme
            and not self.goal.soft_pref
            and not self._item_pids()
            and self.need_search()
        ):
            search = self.next_search()
            if search:
                return [search]
        for pid, sku in self.goal.keys:
            stock = self.inspect_variant(pid, sku)
            if stock:
                return stock
        if target:
            pending = self.inspect_unread([target[0]]) + self.inspect_variant(*target)
            if pending:
                return pending[: self.kernel.max_calls]
            return self.commit(target)
        search = self.next_search()
        return [search] if search else []

    def reply_actions(self) -> list[tuple[str, dict[str, Any]]]:
        self.pending_q = False
        facts = self.fact_sheet(self.best_target())
        last = ""
        for msg in reversed(self.kernel.messages):
            if msg.get("role") == "user":
                last = str(msg.get("content") or "")
                break
        drafted = ""
        if self.clock.can_llm():
            drafted = self.helper.text(
                "Answer the shopper in one or two sentences. If they revealed a spec, acknowledge it.",
                "Shopper: %s\n%s" % (last, facts),
            )
        if not drafted:
            drafted = self.default_reply(last)
        return [("message", self._message_args(drafted))]

    def default_reply(self, last: str) -> str:
        adopted = self.goal.adopt(last)
        if adopted:
            return "Understood, I will keep %s as a firm requirement." % adopted
        if self.index.first_add is None:
            return "I am still checking the options against your requirements; nothing is decided yet."
        if "stock" in last.lower() or "expensive" in last.lower() or "price" in last.lower() or "changed" in last.lower():
            return "I noticed the change. I will re-check the cart and move to the next valid option."
        target = self.pick_target()
        if target and ("which" in last.lower() or "picking" in last.lower()):
            return "I am choosing product %s, SKU %s after checking stock and the listed specs." % (
                target[0],
                target[1],
            )
        return "Thanks — I will stay inside the budget and stock constraints."

    def ask_hidden(self) -> list[tuple[str, dict[str, Any]]]:
        self.asked_hidden = True
        drafted = ""
        if self.clock.can_llm():
            drafted = self.helper.text(
                "Ask the shopper for the hidden firm requirement in one short question.",
                self.goal.text,
            )
        text = drafted or "What firm requirement should I match on the exact item?"
        return [("message", self._message_args(text))]

    def _message_args(self, text: str) -> dict[str, Any]:
        spec = self.kernel.prop("message", "content")
        limit = spec.get("maxLength")
        clipped = str(text or "").strip()
        if limit is not None:
            clipped = clipped[: int(limit)]
        else:
            clipped = clipped[:800]
        return {"content": clipped}

    def search_args(self, query: str) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if self.kernel.allows("search", "query"):
            args["query"] = query
        count = self.kernel.cap("search", "k", 10)
        if count is not None:
            args["k"] = count
        if self.kernel.allows("search", "in_stock"):
            args["in_stock"] = True
        return args

    def _inspect_more(self) -> list[tuple[str, dict[str, Any]]]:
        unread = self.inspect_unread(
            [pid for pid in self.survey_pids() if pid not in self.index.viewed][: self.clock.survey_k()]
        )
        if unread:
            return unread[: self.kernel.max_calls]
        for pid in self.survey_pids()[:8]:
            if str(pid) in self.index.viewed and not any(
                self.eligible(*pair) for pair in self.variants_of(pid)
            ):
                continue
            for pair in self.variants_of(pid):
                if str(pid) in self.index.viewed and not self.eligible(*pair):
                    continue
                stock = self.inspect_variant(*pair)
                if stock:
                    return stock
        return []

    def idle_actions(self) -> list[tuple[str, dict[str, Any]]]:
        if self.index.first_add is not None:
            target = self._choose_target()
            if target and target not in self.index.cart:
                pending = self.commit(target)
                if pending:
                    return pending
            other = [pair for pair in self.cart_live() if pair != self.index.first_add]
            if other and self.can_order(other[-1]):
                return self.close_actions(other[-1])
        pending = self._inspect_more()
        if pending:
            return pending
        target = self._choose_target()
        if target and target not in self.index.cart:
            pending = self.commit(target)
            if pending:
                return pending
        if target and self.can_order(target):
            return self.close_actions(target)
        search = self.next_search()
        if search:
            unread = self.inspect_unread(
                [pid for pid in self.survey_pids() if pid not in self.index.viewed][
                    : self.clock.survey_k()
                ]
            )
            return (unread + [search])[: self.kernel.max_calls]
        if (
            self.index.first_add is not None
            and self.kernel.has("filter")
            and not self.index.filters
            and (self.goal.category or self.goal.item)
        ):
            return [("filter", self.filter_args())]
        if self.index.first_add is not None and self.clock.closing():
            blocked = {_vid(*self.index.first_add)}
            nxt = self.best_target(exclude=blocked)
            if nxt and nxt not in self.index.cart:
                pending = self.commit(nxt)
                if pending:
                    return pending
            other = [pair for pair in self.cart_live() if pair != self.index.first_add]
            if other:
                return self.close_actions(other[-1])
        if self.index.first_add is None:
            unread = self.inspect_unread(
                [pid for pid in self.all_pids() if pid not in self.index.viewed][
                    : self.clock.survey_k()
                ]
            )
            if unread:
                return unread[: self.kernel.max_calls]
            bait = self.opening_bait()
            if bait:
                pending = self.commit(bait)
                if pending:
                    return pending
        if self.idle_inspects < 2 and self.kernel.has("inspect_cart"):
            return [("inspect_cart", {})]
        return []

    def _kind(self) -> str:
        return self.goal.item or self.goal.category or ""

    def _looks_like_item(self, pid: str) -> bool:
        title = self.index.title(pid).lower()
        brand = (self.goal.brand or self.goal.pref_brand or "").lower()
        if brand and brand in title:
            return True
        for tok in self._kind_terms():
            if tok and tok.lower() in title:
                return True
        return False

    def _item_pids(self) -> list[str]:
        return [pid for pid in self.survey_pids() if self._looks_like_item(pid)]

    def _viewed_item_pids(self) -> list[str]:
        return [pid for pid in self._item_pids() if pid in self.index.viewed]

    def _catalog_breadth(self, pid: str) -> int:
        title = self.index.title(pid).lower()
        sizes = re.findall(r"(\d+(?:\.\d+)?)\s*(?:inch|in|''|\")", title)
        slash = re.findall(r"(\d+)\s*/\s*(\d+)", title)
        for left, right in slash:
            sizes.extend([left, right])
        return len({str(item) for item in sizes})

    def _kind_terms(self) -> list[str]:
        kind = self._kind()
        out: list[str] = []
        if kind:
            out.append(kind)
            low = kind.lower()
            if low.endswith("s") and len(low) > 4 and not low.endswith("ss"):
                out.append(low[:-1])
            if " " not in low and len(low) >= 10:
                for width in range(6, 4, -1):
                    tail = low[-width:]
                    head = low[:-width]
                    if len(head) >= 5 and head.isalpha() and tail.isalpha():
                        out.append(tail)
                        break
        return list(dict.fromkeys(out))

    def _want_pair(self) -> bool:
        return bool(
            self.goal.wants_reason
            or self.goal.hidden
            or self.goal.intent_cue
            or (
                (self.goal.brand or self.goal.pref_brand)
                and (self.goal.measures or self.goal.res_tokens)
            )
        )

    def _recover_premium(self) -> bool:
        return bool(
            self.goal.wants_reason
            or self.goal.intent_cue
            or (self.goal.brand and (self.goal.measures or self.goal.res_tokens))
        )

    def _thin_recovery(self, target: tuple[str, str] | None) -> bool:
        if self.index.first_add is None or not target or not self._deep_budget_cut():
            return False
        if self.clock.closing() or self.goal.budget is None or not self.need_search():
            return False
        return self._price_of(*target) < 0.5 * self.goal.budget - _EPS

    def _weak_res(self) -> bool:
        if not self.goal.wants_reason or not self.goal.res_tokens:
            return False
        return not any(
            token.endswith("k") or (token.endswith("hd") and token != "hd")
            for token in self.goal.res_tokens
        )

    def _relax_res(self) -> bool:
        return self._weak_res()

    def _opening_premium(self) -> bool:
        return not self._weak_res()

    def _local_text(self, pid: str, sku: str = "") -> str:
        options = self.index.options.get(_vid(pid, sku), "")
        facts = self.index.facts(pid)
        bits = [options, sku]
        for key, value in facts.items():
            if not value or _RANGE.search(str(value)):
                continue
            bits.append(self.index._fact_text(key, value))
        return " ".join(bits)

    def _spec_strength(self, pid: str, sku: str = "") -> int:
        local = self._local_text(pid, sku)
        title = self.index.title(pid)
        score = 0
        if self.goal.brand and self.goal.brand.lower() in (
            self.index.brand_of(pid) + " " + local
        ).lower():
            score += 2
        elif self.goal.brand and self.goal.brand.lower() in title.lower():
            score += 1
        for item in self.goal.measures:
            if _measure_match_exact(item, local):
                score += 3
            elif _measure_match_exact(item, title):
                score += 1
        if self.goal.res_tokens:
            if self._res_match(local):
                score += 3
            elif self._res_match(title):
                score += 1
        if self.goal.soft_pref:
            score += self.pref_score(pid, sku)
        return score

    def _query_overlap(self, pid: str, sku: str = "") -> int:
        blob = (self.index.title(pid) + " " + self.index.blob(pid, sku)).lower()
        bits = _content_tokens(self.goal.text)
        extra = []
        if self.goal.brand:
            extra.extend(_content_tokens(self.goal.brand))
        if self.goal.item:
            extra.extend(_content_tokens(self.goal.item))
        want = [tok for tok in bits + extra if tok not in _STOP and len(tok) > 2]
        return sum(1 for tok in dict.fromkeys(want) if tok in blob)

    def _need_is_side(self, need: str | None) -> bool:
        return bool(
            re.search(
                r"\b(accessories|accessory|cases|case|covers|cover)\b",
                (need or "").lower(),
            )
        )

    def _is_side_listing(self, pid: str, need: str | None) -> bool:
        if not need or self._need_is_side(need):
            return False
        path = self.index.path_text(pid).lower()
        title = self.index.title(pid).lower()
        if re.search(r"\b(accessories|accessory|cases)\b", path):
            return True
        noun = re.escape((need or "").lower().rstrip("s"))
        if noun and re.search(
            r"\b%s[s]?\b.{0,24}\b(case|cases|cover|covers|accessor)" % noun,
            title,
        ):
            return True
        return False

    def _path_matches(self, pid: str, need: str | None) -> bool:
        if not need:
            return True
        if self._is_side_listing(pid, need):
            return False
        path = self.index.path_text(pid).lower()
        title = self.index.title(pid).lower()
        if not path:
            tokens = _content_tokens(need)
            blob = title + " " + self.index.blob(pid).lower()
            return any(tok in blob for tok in tokens) if tokens else True
        tokens = _content_tokens(need)
        have = _tokens(path)
        for tok in tokens:
            if tok in have or (tok + "s") in have or tok.rstrip("s") in have:
                return True
            if len(tok) > 3 and tok in path:
                return True
        return False

    def fallback_query(self) -> str:
        bits = [self.goal.brand or self.goal.pref_brand or "", self._kind()]
        if self.goal.theme:
            bits.extend(self.goal.theme)
        if self.goal.model:
            bits.append(self.goal.model)
        if self.goal.pref_terms:
            bits.extend(self.goal.pref_terms[:3])
        if self.goal.measures:
            forms = _measure_forms(self.goal.measures[0])
            bits.insert(1, forms[0] if self.goal.measures[0][1] != "in" else ("%gin" % self.goal.measures[0][0]))
        if self.goal.res_tokens:
            named = [tok for tok in self.goal.res_tokens if not tok[:1].isdigit()]
            bits.insert(1, (named or self.goal.res_tokens)[0])
        return " ".join(bit for bit in bits if bit).strip() or self.goal.text[:80]

    def need_search(self) -> bool:
        if self.goal.keys:
            return False
        used = len(self.index.queries) + len(self.index.filters)
        if self.kernel.has("search") or self.kernel.has("filter"):
            return used < self.search_budget()
        return False

    def _deep_budget_cut(self) -> bool:
        high = self.goal.opening_budget
        low = self.goal.budget
        return high is not None and low is not None and low < 0.9 * high - _EPS

    def _filter_price(self) -> float | None:
        if self.goal.budget is None:
            return None
        if self._deep_budget_cut() and self.index.first_add is None:
            return self.goal.opening_budget
        return self.goal.budget

    def _filter_brand(self) -> str | None:
        if self.index.first_add is None:
            return self.goal.pref_brand or self.goal.brand
        return self.goal.brand

    def _budget_filter_stale(self) -> bool:
        if not self.kernel.has("filter") or not self.index.filters:
            return False
        prev = self.index.filters[-1]
        wanted_price = self._filter_price()
        if wanted_price is not None:
            prev_price = prev.get("max_price")
            if prev_price is None:
                return True
            try:
                if float(prev_price) > wanted_price + _EPS:
                    return True
            except (TypeError, ValueError):
                return True
        wanted_brand = self._filter_brand()
        prev_brand = str(prev.get("brand") or "").strip()
        if wanted_brand and prev_brand.lower() != wanted_brand.lower():
            return True
        wanted_cat = self.goal.item or self.goal.category
        prev_cat = str(prev.get("category") or "").strip()
        if wanted_cat and prev_cat and prev_cat.lower() != wanted_cat.lower():
            return True
        return False

    def search_budget(self) -> int:
        if self.goal.theme:
            base = 4
        elif self.goal.model:
            base = 5
        elif self.goal.soft_pref:
            base = 8
        elif self.goal.measures or self.goal.brand or self.goal.pref_brand or self.goal.res_tokens:
            base = 8
        else:
            base = 6
        if self.index.first_add is None:
            return base
        other = [pair for pair in self.cart_live() if pair != self.index.first_add]
        if other:
            return base
        used = len(self.index.queries) + len(self.index.filters)
        if not self.clock.can_search():
            return used
        return max(base, used + 1)

    def filter_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {}
        count = self.kernel.cap("filter", "k", 10)
        if count is not None:
            args["k"] = count
        if self.kernel.allows("filter", "in_stock"):
            args["in_stock"] = True
        if self.kernel.allows("filter", "category"):
            if self.goal.item:
                args["category"] = self.goal.item
            elif self.goal.category:
                args["category"] = self.goal.category
        price = self._filter_price()
        if price is not None and self.kernel.allows("filter", "max_price"):
            args["max_price"] = price
        brand = self._filter_brand()
        if brand and self.kernel.allows("filter", "brand"):
            args["brand"] = brand
        return args

    def search_candidates(self) -> list[str]:
        if self.goal.theme:
            phrase = " ".join(self.goal.theme)
            return [
                phrase,
                ("%s %s" % (phrase, self._kind())).strip(),
                ("%s %s" % (self._kind(), phrase)).strip(),
                "%s in stock" % phrase,
            ]
        if self.goal.model:
            return [
                ("%s %s" % (self.goal.color or "", self.goal.model or "")).strip(),
                str(self.goal.model),
                ("%s %s" % (self.goal.model or "", self._kind())).strip(),
                ("%s %s" % (self._kind(), self.goal.model or "")).strip(),
            ]
        if self.goal.soft_pref:
            return self._pref_search_queries()
        kinds = self._kind_terms() or [""]
        kind = kinds[0]
        brand = self.goal.pref_brand or self.goal.brand or ""
        out: list[str] = []
        if (
            self.index.first_add is not None
            and self._deep_budget_cut()
            and self.goal.measures
            and kind
        ):
            for form in _measure_forms(self.goal.measures[0])[:4]:
                out.append(("%s %s" % (form, kind)).strip())
        if self.goal.measures and not self.goal.res_tokens:
            if brand and kind:
                out.append(("%s %s" % (brand, kind)).strip())
            elif kind:
                out.append(kind)
        out.append(self.fallback_query())
        if self.goal.budget is not None:
            money = ("%.2f" % self.goal.budget).rstrip("0").rstrip(".")
            cur = self.goal.currency or ""
            out.append(("%s %s under %s %s" % (brand, kind, money, cur)).strip())
            out.append(("%s under %s %s" % (kind, money, cur)).strip())
        named = [tok for tok in self.goal.res_tokens if not tok[:1].isdigit()]
        for tok in named + [tok for tok in self.goal.res_tokens if tok[:1].isdigit()]:
            for noun in kinds:
                out.append(("%s %s %s" % (brand, tok, noun)).strip())
                out.append(("%s %s" % (tok, noun)).strip())
            if brand:
                out.append(("%s %s" % (brand, tok)).strip())
        if self.goal.measures:
            value, unit, raw = self.goal.measures[0]
            forms = _measure_forms((value, unit, raw))
            if unit == "in":
                forms.extend(["%g inch" % value, "%g-inch" % value, "%gin" % value])
            for form in list(dict.fromkeys(forms))[:5]:
                for noun in kinds:
                    out.append(("%s %s %s" % (brand, form, noun)).strip())
                    out.append(("%s %s" % (form, noun)).strip())
        out.append(("%s %s" % (brand, kind)).strip())
        out.append((kind or "item") + " in stock")
        out.extend(self._extra_search_queries())
        return out

    def _extra_search_queries(self) -> list[str]:
        kind = self._kind()
        brand = self.goal.brand or self.goal.pref_brand or ""
        out: list[str] = []
        for item in self.goal.measures:
            for form in _measure_forms(item):
                out.append(form)
                if kind:
                    out.append(("%s %s" % (form, kind)).strip())
                    out.append(("%s %s" % (kind, form)).strip())
                if brand:
                    out.append(("%s %s" % (brand, form)).strip())
        if kind:
            out.append("%s available" % kind)
            out.append("buy %s" % kind)
        if self.index.first_add is not None:
            for pid in list(self.index.viewed)[:6]:
                seen = self.index.brand_of(pid)
                if not seen or not kind:
                    continue
                if brand and seen.lower() == brand.lower():
                    continue
                out.append(("%s %s" % (seen, kind)).strip())
                if self.goal.measures:
                    out.append(
                        ("%s %s %s" % (seen, _measure_forms(self.goal.measures[0])[0], kind)).strip()
                    )
        return out

    def _pref_search_queries(self) -> list[str]:
        item = self._kind()
        out: list[str] = []
        money = ""
        if self.goal.budget is not None:
            money = ("%.2f" % self.goal.budget).rstrip("0").rstrip(".")
        for span in self.goal.pref_spans:
            for sep in (" - ", " — ", ": "):
                if sep in span:
                    span = span.split(sep)[-1]
            words = [w for w in _content_tokens(span) if w not in _GENERIC_NOUNS and len(w) > 3][:3]
            for index in range(len(words) - 1):
                gram = "%s %s" % (words[index], words[index + 1])
                out.append(("%s %s" % (gram, item)).strip())
                out.append(gram)
            for word in words[:3]:
                out.append(("%s %s" % (word, item)).strip())
        extra = " ".join(self.goal.pref_terms[:3])
        if extra:
            out.append(("%s %s" % (extra, item)).strip())
        if item:
            out.append(item)
        if self.index.first_add is not None:
            extra_q: list[str] = []
            purpose = self.goal.purpose_terms[:3]
            if purpose and item:
                extra_q.append((" ".join(purpose + [item])).strip())
                extra_q.append((" ".join(purpose[:2] + [item])).strip())
                if money:
                    extra_q.append(
                        ("%s %s under %s" % (" ".join(purpose[:2]), item, money)).strip()
                    )
            for term in purpose:
                extra_q.append(("%s %s" % (term, item)).strip())
                for pref in self.goal.pref_terms[:2]:
                    extra_q.append(("%s %s %s" % (term, pref, item)).strip())
            out[0:0] = extra_q
        out.extend(self._extra_search_queries())
        return [query for query in out if query]

    def _filter_ready(self) -> bool:
        specific = bool(
            self.goal.model
            or self.goal.theme
            or self.goal.measures
            or self.goal.brand
            or self.goal.pref_brand
            or self.goal.pref_terms
            or self.goal.res_tokens
        )
        return (
            self.kernel.has("filter")
            and not self.index.filters
            and (self.goal.category or self.goal.item or self.goal.budget is not None)
            and not (self.goal.model or self.goal.theme)
            and (not specific or self.index.queries)
        )

    def next_search(self) -> tuple[str, dict[str, Any]] | None:
        if self._budget_filter_stale():
            return ("filter", self.filter_args())
        if not self.need_search():
            return None
        used = {q.lower() for q in self.index.queries}
        if self.kernel.has("search"):
            for query in self.search_candidates():
                if query and query.lower() not in used:
                    return ("search", self.search_args(query))
        if self._filter_ready():
            return ("filter", self.filter_args())
        if (
            self.index.first_add is not None
            and self.kernel.has("filter")
            and not self.index.filters
            and (self.goal.category or self.goal.item)
        ):
            return ("filter", self.filter_args())
        return None

    def inspect_unread(self, pids: list[str]) -> list[tuple[str, dict[str, Any]]]:
        unseen = [pid for pid in pids if pid not in self.index.viewed]
        if not unseen:
            return []
        if self.kernel.has("compare") and len(unseen) > 1:
            return [("compare", {"product_ids": unseen[: self.clock.survey_k()]})]
        if self.kernel.has("view"):
            return [("view", {"product_id": unseen[0]})]
        return []

    def inspect_variant(self, pid: str, sku: str) -> list[tuple[str, dict[str, Any]]]:
        key = _vid(pid, sku)
        if key in self.index.inspected or not self.kernel.has("inspect_stock"):
            return []
        return [("inspect_stock", {"product_id": pid, "sku": sku})]

    def cart_live(self) -> list[tuple[str, str]]:
        return [
            pair
            for pair in self.index.cart
            if self.index.live(pair[0], pair[1], self.goal.budget)
        ]

    def cart_stale(self) -> list[tuple[str, str]]:
        return [
            pair
            for pair in self.index.cart
            if not self.index.live(pair[0], pair[1], self.goal.budget)
        ]

    def repair_order(self) -> list[tuple[str, dict[str, Any]]]:
        if not self.last_error or self.last_error[0] != "place_test_order":
            return []
        _name, code, args = self.last_error
        self.last_error = None
        wanted = (str(args.get("product_id") or ""), str(args.get("sku") or ""))
        if code == "not_in_cart" and wanted[0] and wanted[1]:
            if wanted in self.index.cart and self.can_order(wanted):
                return self.close_actions(wanted)
            return self.commit(wanted)
        return []

    def after_commit(self, target: tuple[str, str] | None) -> list[tuple[str, dict[str, Any]]]:
        if self.index.first_add is None:
            return []
        if self.committed_turn is None:
            self.committed_turn = self.kernel.turn
        if not self.rechecked:
            self.rechecked = True
            if self.kernel.has("inspect_cart"):
                return [("inspect_cart", {})]
        if self._budget_filter_stale():
            return [("filter", self.filter_args())]
        if not self.post_event_surveyed:
            self.post_event_surveyed = True
            surveyed = self._post_event_survey()
            if surveyed:
                return surveyed
        evented = self.index.first_add
        other_live = [pair for pair in self.cart_live() if pair != evented]
        stale = self.cart_stale()
        nxt = target if target and target != evented else self.pick_target()
        if nxt == evented:
            nxt = None
        if nxt and nxt not in self.index.cart:
            pending = self.commit(nxt)
            if pending:
                return pending
        if other_live:
            if evented in self.index.cart and self.kernel.has("remove_from_cart"):
                return [("remove_from_cart", {"product_id": evented[0], "sku": evented[1]})]
            if stale and self.kernel.has("remove_from_cart"):
                dead = stale[0]
                return [("remove_from_cart", {"product_id": dead[0], "sku": dead[1]})]
            current = nxt if nxt in other_live else other_live[-1]
            if self.can_order(current):
                return self.close_actions(current)
            return []
        if nxt:
            return self.commit(nxt)
        return self.recover_actions(evented)

    def _post_event_survey(self) -> list[tuple[str, dict[str, Any]]]:
        evented = self.index.first_add
        skip = evented[0] if evented else ""
        if self.goal.keys:
            rest = [pid for pid, _sku in self.goal.keys if pid != skip]
            rest = list(dict.fromkeys(rest))
            if rest:
                if self.kernel.has("compare") and len(rest) > 1:
                    return [("compare", {"product_ids": rest[: self.clock.survey_k()]})]
                unread = self.inspect_unread(rest[: self.clock.survey_k()])
                if unread:
                    return unread
                if self.goal.keys:
                    for pid, sku in self.goal.keys:
                        if (pid, sku) != evented:
                            stock = self.inspect_variant(pid, sku)
                            if stock:
                                return stock
        unread = self.inspect_unread(
            [pid for pid in self.survey_pids() if pid != skip][: self.clock.survey_k()]
        )
        if unread:
            return unread[: self.kernel.max_calls]
        search = self.next_search()
        if search:
            return [search]
        return []

    def recover_actions(self, dead: tuple[str, str]) -> list[tuple[str, dict[str, Any]]]:
        blocked = set(self.index.dead)
        blocked.add(_vid(*dead))
        nxt = None
        if self.goal.keys:
            nxt = self.rank_target()
        elif self.goal.soft_pref:
            nxt = self.preference_target()
        elif not self.goal.wants_reason:
            nxt = self.sibling_target(dead)
        if not nxt and self.goal.model:
            nxt = self.recovery_target(dead)
        if not nxt:
            nxt = self.best_target(exclude=blocked, premium=self._recover_premium())
        if nxt and self.index.first_add and nxt == self.index.first_add:
            nxt = None
        if self._thin_recovery(nxt):
            nxt = None
        if self._budget_filter_stale():
            return [("filter", self.filter_args())]
        unread = self.inspect_unread(
            [pid for pid in self.survey_pids() if not dead or pid != dead[0]][
                : self.clock.survey_k()
            ]
        )
        if not nxt:
            search = self.next_search()
            packed = []
            if unread:
                packed.extend(unread)
            if search:
                packed.append(search)
            return packed[: self.kernel.max_calls]
        pending = unread[:1] if unread and unread[0][0] == "compare" else []
        pending.extend(self.inspect_unread([nxt[0]]) + self.inspect_variant(*nxt))
        if pending:
            return pending[: self.kernel.max_calls]
        if self.kernel.has("add_to_cart"):
            return [("add_to_cart", {"product_id": nxt[0], "sku": nxt[1]})]
        return []

    def close_actions(self, target: tuple[str, str]) -> list[tuple[str, dict[str, Any]]]:
        key = _vid(*target)
        if key in self.failed_orders and target not in self.index.cart:
            return self.commit(target)
        self.failed_orders.discard(key)
        return [("place_test_order", self.order_args(target))]

    def can_order(self, target: tuple[str, str]) -> bool:
        if self.index.first_add is None:
            return False
        if target == self.index.first_add:
            return False
        if target not in self.index.cart:
            return False
        if self.committed_turn is not None and self.kernel.turn <= self.committed_turn:
            return False
        if not self.index.live(target[0], target[1], self.goal.budget):
            return False
        if self.pending_q:
            return False
        if (
            self.goal.soft_pref
            and len(set(self.index.listings.values())) < 2
            and len(self.cart_live()) < 2
            and not self.clock.closing()
        ):
            return False
        if self.goal.theme and len(self.index.clusters) < 3:
            used = len(self.index.queries) + len(self.index.filters)
            if used < self.search_budget():
                return False
        return True

    def commit(self, target: tuple[str, str]) -> list[tuple[str, dict[str, Any]]]:
        if _vid(*target) in self.index.dead:
            return []
        if self.index.first_add is not None and target == self.index.first_add:
            return []
        if not self.ready_to_commit(target):
            pending = []
            if str(target[0]) not in self.index.viewed:
                pending.extend(self.inspect_unread([target[0]]))
            pending.extend(self.inspect_variant(*target))
            if pending:
                return pending[: self.kernel.max_calls]
            search = self.next_search()
            return [search] if search else []
        pending = []
        if target[0] not in self.index.viewed:
            pending.extend(self.inspect_unread([target[0]]))
        pending.extend(self.inspect_variant(*target))
        if pending:
            return pending[: self.kernel.max_calls]
        if not self.eligible(*target) and not self._opening_override(target):
            return []
        return [("add_to_cart", {"product_id": target[0], "sku": target[1]})]

    def ready_to_commit(self, target: tuple[str, str]) -> bool:
        if not self.eligible(*target) and not self._opening_override(target):
            return False
        if (self.goal.measures or self.goal.res_tokens) and not self.sku_has_spec(*target):
            if not self._relax_res():
                return False
        if self.goal.model and not self.model_pids():
            return False
        if str(target[0]) not in self.index.viewed and (
            self.goal.measures
            or self.goal.res_tokens
            or self.goal.item
            or self.goal.category
            or self.goal.soft_pref
        ):
            return False
        return True

    def survey_pids(self) -> list[str]:
        if self.goal.keys:
            seen: list[str] = []
            for pid, _sku in self.goal.keys:
                if pid not in seen:
                    seen.append(pid)
            return seen
        if self.goal.model:
            pids = self.model_pids()
        elif self.goal.soft_pref:
            pids = self.pref_pids() or self.theme_pids() or self.all_pids()
        elif self.goal.theme:
            pids = self.theme_pids() or self.all_pids()
        elif self.index.first_add is not None and (self.goal.measures or self.goal.brand):
            spec = self.spec_pids()
            itemish = [pid for pid in self.all_pids() if self._looks_like_item(pid)]
            pids = []
            for pid in spec + itemish:
                if pid not in pids:
                    pids.append(pid)
            if not pids:
                pids = self.all_pids()
        elif self.goal.measures or self.goal.res_tokens:
            pids = self.spec_pids() or self.all_pids()
        else:
            pids = self.all_pids()
        brand = self.goal.brand or self.goal.pref_brand
        if self.index.first_add is None and (brand or self.goal.wants_reason or self.goal.measures):
            pids = sorted(pids, key=self._pid_max_price, reverse=True)
        elif (
            self.index.first_add is not None
            and self._deep_budget_cut()
            and self.goal.brand
        ):
            pids = sorted(pids, key=self._pid_max_price, reverse=True)
        if brand:
            branded = [pid for pid in pids if brand.lower() in self.index.blob(pid).lower()]
            if branded:
                rest = [pid for pid in pids if pid not in set(branded)]
                pids = branded + rest
        itemish = [pid for pid in pids if self._looks_like_item(pid)]
        if itemish and not self.goal.keys and not self.goal.theme and not self.goal.soft_pref:
            pids = itemish
        if (
            self.goal.budget is not None
            and (self.goal.wants_reason or self.goal.measures or brand)
        ):
            if self._deep_budget_cut() and self.index.first_add is None:
                band = self._band_pids()
                if band:
                    rest = [pid for pid in pids if pid not in set(band)]
                    pids = band + rest
            else:
                affordable = [
                    pid
                    for pid in pids
                    if self._pid_max_price(pid) <= self.goal.budget + _EPS
                ]
                if affordable:
                    rest = [pid for pid in pids if pid not in set(affordable)]
                    pids = affordable + rest
        if self.index.viewed and not self.goal.soft_pref:
            unread = [pid for pid in pids if pid not in self.index.viewed]
            if unread:
                if brand and not self.eligible_pairs():
                    off = [
                        pid
                        for pid in unread
                        if brand.lower() not in self.index.title(pid).lower()
                    ]
                    if off:
                        unread = off + [pid for pid in unread if pid not in set(off)]
                pids = unread + [pid for pid in pids if pid not in set(unread)]
        return pids

    def _pid_max_price(self, pid: str) -> float:
        best = 0.0
        for pair in self.variants_of(pid):
            price = self._price_of(*pair)
            if price > best:
                best = price
        hit = self.index.hits.get(_vid(pid, "")) or {}
        for key, row in self.index.hits.items():
            if key.startswith(str(pid) + "::"):
                price = _num((row or {}).get("price"))
                if price is not None and price > best:
                    best = price
        price = _num(hit.get("price")) if hit else None
        if price is not None and price > best:
            best = price
        return best

    def _choose_target(self) -> tuple[str, str] | None:
        target = self.pick_target()
        if self._thin_recovery(target):
            target = None
        if target:
            return target
        if self.goal.brand:
            return self.best_target(
                brand=self.goal.brand,
                premium=self.index.first_add is None or self._recover_premium(),
            ) or self.opening_bait()
        return self.best_target() or self.opening_bait()

    def opening_bait(self) -> tuple[str, str] | None:
        if self.index.first_add is not None:
            return None
        if self.goal.wants_reason or self.goal.soft_pref or self.goal.theme or self.goal.keys:
            return None
        brand = self.goal.brand or self.goal.pref_brand
        viewed_ok = False
        for pid in self.index.viewed:
            if not self._looks_like_item(pid):
                continue
            blob = (self.index.title(pid) + " " + self.index.blob(pid)).lower()
            if brand and brand.lower() not in blob:
                continue
            viewed_ok = True
            break
        if not viewed_ok:
            return None
        need = self.goal.category or self.goal.item
        found: list[tuple[float, tuple[str, str]]] = []
        for pid in self.all_pids():
            if str(pid) not in self.index.viewed:
                continue
            if not self._looks_like_item(pid):
                continue
            if need and self._is_side_listing(pid, need):
                continue
            for pair in self.variants_of(pid):
                quote = self.index.quote(*pair) or {}
                if quote.get("in_stock") is False:
                    continue
                price = quote.get("price")
                if price is None:
                    continue
                blob = (self.index.title(pid) + " " + self.index.blob(*pair)).lower()
                if brand and brand.lower() not in blob:
                    continue
                if self.goal.measures and not self.sku_has_spec(*pair):
                    continue
                if self.goal.res_tokens and not self._res_match(blob):
                    if not self._relax_res():
                        continue
                found.append((float(price), pair))
        found.sort()
        return found[0][1] if found else None

    def _band_pairs(self, viewed_only: bool = False) -> list[tuple[float, tuple[str, str]]]:
        if not self._deep_budget_cut() or self.index.first_add is not None:
            return []
        high = self.goal.opening_budget
        low = self.goal.budget
        if high is None or low is None:
            return []
        brand = self.goal.brand or self.goal.pref_brand
        need = self.goal.category or self.goal.item
        found: list[tuple[float, tuple[str, str]]] = []
        for pid in self.all_pids():
            if viewed_only and str(pid) not in self.index.viewed:
                continue
            if viewed_only and not self._looks_like_item(pid):
                continue
            if need and str(pid) in self.index.viewed and self._is_side_listing(pid, need):
                continue
            for pair in self.variants_of(pid):
                quote = self.index.quote(*pair) or {}
                if quote.get("in_stock") is False:
                    continue
                price = quote.get("price")
                if price is None:
                    price = _num((self.index.hits.get(_vid(*pair)) or {}).get("price"))
                if price is None or price <= low + _EPS or price > high + _EPS:
                    continue
                blob = (self.index.title(pid) + " " + self.index.blob(*pair)).lower()
                if brand and brand.lower() not in blob:
                    continue
                if viewed_only and self.goal.measures and not self.sku_has_spec(*pair):
                    continue
                found.append((float(price), pair))
        found.sort()
        return found

    def _band_pids(self) -> list[str]:
        seen: list[str] = []
        for _price, pair in self._band_pairs(viewed_only=False):
            if pair[0] not in seen:
                seen.append(pair[0])
        return seen

    def _band_unread(self) -> bool:
        return any(pid not in self.index.viewed for pid in self._band_pids())

    def _cap_viewed(self) -> tuple[str, str] | None:
        if self.goal.budget is None:
            return None
        brand = self.goal.brand or self.goal.pref_brand
        need = self.goal.category or self.goal.item
        found: list[tuple[float, tuple[str, str]]] = []
        for pid in self.index.viewed:
            if not self._looks_like_item(pid):
                continue
            if need and self._is_side_listing(pid, need):
                continue
            for pair in self.variants_of(pid):
                quote = self.index.quote(*pair) or {}
                if quote.get("in_stock") is False:
                    continue
                price = quote.get("price")
                if price is None or price > self.goal.budget + _EPS:
                    continue
                blob = (self.index.title(pid) + " " + self.index.blob(*pair)).lower()
                if brand and brand.lower() not in blob:
                    continue
                if self.goal.measures and not self.sku_has_spec(*pair):
                    continue
                found.append((float(price), pair))
        found.sort(reverse=True)
        return found[0][1] if found else None

    def band_event(self) -> tuple[str, str] | None:
        found = self._band_pairs(viewed_only=True)
        return found[0][1] if found else None

    def _deep_cut_unread(self) -> bool:
        if (
            self.index.first_add is None
            or not self._deep_budget_cut()
            or not self.goal.brand
            or self.clock.closing()
        ):
            return False
        brand = self.goal.brand.lower()
        top = [
            pid
            for pid in self.survey_pids()
            if brand in self.index.blob(pid).lower()
            and (
                self.goal.budget is None
                or self._pid_max_price(pid) <= self.goal.budget + _EPS
            )
        ][: self.clock.survey_k()]
        return any(pid not in self.index.viewed for pid in top)

    def _opening_override(self, target: tuple[str, str] | None) -> bool:
        if self.index.first_add is not None or not target:
            return False
        return target == self.opening_bait() or target == self.band_event()

    def pick_target(self) -> tuple[str, str] | None:
        opening = self.index.first_add is None
        if self.goal.keys:
            return self.rank_target()
        if not opening and not self.goal.wants_reason:
            sibling = self.sibling_target(self.index.first_add)
            if sibling:
                return sibling
        if self.goal.model:
            return self.recovery_preferred() or self.best_target(premium=opening)
        if self.goal.soft_pref:
            return self.preference_target()
        recover_premium = (opening and self._opening_premium()) or (
            not opening and self._recover_premium()
        )
        brand = self.goal.pref_brand if opening else (self.goal.pref_brand or self.goal.brand)
        if opening and self.goal.pref_brand:
            return self.best_target(brand=self.goal.pref_brand, premium=True) or self.best_target(
                premium=True
            )
        if self.goal.brand:
            if opening and self._deep_budget_cut():
                bait = self.band_event()
                if bait:
                    return bait
                if self._band_unread():
                    return None
            if not opening and self._deep_cut_unread():
                return None
            live = self.best_target(brand=self.goal.brand, premium=recover_premium)
            if not opening and self._deep_budget_cut():
                cap = self._cap_viewed()
                if cap and cap != self.index.first_add and not self._thin_recovery(cap):
                    live = cap
            if not opening and self._thin_recovery(live):
                return None
            return live or self.opening_bait()
        if brand:
            return self.best_target(brand=brand, premium=recover_premium) or self.best_target(
                premium=recover_premium
            )
        return self.best_target(premium=recover_premium) or self.opening_bait()

    def all_pids(self) -> list[str]:
        pids: list[str] = []
        seen: set[str] = set()
        for key, row in self.index.hits.items():
            pid = str((row or {}).get("product_id") or key.split("::", 1)[0])
            if pid not in seen:
                seen.add(pid)
                pids.append(pid)
        for pid in self.index.pages:
            if pid not in seen:
                pids.append(pid)
        return pids

    def theme_pids(self) -> list[str]:
        return [pid for pid in self.all_pids() if pid in self.index.clusters]

    def pref_pids(self) -> list[str]:
        scored = []
        for pid in self.all_pids():
            score = self.pref_score(pid)
            if score > 0:
                scored.append((score, pid))
        scored.sort(reverse=True)
        return [pid for _score, pid in scored]

    def model_pids(self) -> list[str]:
        if not self.goal.model:
            return self.all_pids()
        return [
            pid
            for pid in self.all_pids()
            if _has_token(self.index.blob(pid), self.goal.model)
        ]

    def spec_pids(self) -> list[str]:
        if not self.goal.measures:
            return self.all_pids()
        out = []
        for pid in self.all_pids():
            if self.product_has_spec(pid):
                out.append(pid)
        return out

    def variants_of(self, pid: str) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        page = self.index.pages.get(str(pid)) or {}
        for variant in page.get("variants") or []:
            if isinstance(variant, dict) and variant.get("sku"):
                found.append((str(pid), str(variant["sku"])))
        for key in self.index.hits:
            if key.startswith(str(pid) + "::"):
                _pid, sku = key.split("::", 1)
                pair = (str(pid), sku)
                if sku and pair not in found:
                    found.append(pair)
        if not found:
            for key in self.index.quotes:
                if key.startswith(str(pid) + "::"):
                    _pid, sku = key.split("::", 1)
                    if sku:
                        found.append((str(pid), sku))
        return found

    def eligible(self, pid: str, sku: str, *, brand: str | None = None) -> bool:
        if _vid(pid, sku) in self.index.dead:
            return False
        if not self.index.live(pid, sku, self.goal.budget):
            return False
        title = self.index.title(pid)
        options = self.index.options.get(_vid(pid, sku), "")
        blob = self.index.blob(pid, sku)
        low = (title + " " + options + " " + blob).lower()
        if self.goal.reject_terms and any(
            re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", low)
            for term in self.goal.reject_terms
        ):
            return False
        if brand and brand.lower() not in low:
            return False
        if self.goal.measures and not self.sku_has_spec(pid, sku):
            return False
        if self.goal.res_tokens and not self._res_match(low):
            if not self._relax_res():
                return False
        if self.goal.theme and not self.goal.theme_tokens_in(title, options, blob):
            return False
        need = self.goal.category or self.goal.item
        if need and not self.goal.keys and self._is_side_listing(pid, need):
            return False
        if str(pid) in self.index.viewed:
            if need and not self.goal.keys and not self._path_matches(pid, need):
                return False
        return True

    def product_has_spec(self, pid: str) -> bool:
        blob = self.index.blob(pid)
        if self.goal.measures and not all(_measure_match_exact(item, blob) for item in self.goal.measures):
            return False
        if self.goal.res_tokens and not self._res_match(blob):
            if not self._relax_res():
                return False
        return True

    def sku_has_spec(self, pid: str, sku: str) -> bool:
        options = self.index.options.get(_vid(pid, sku), "")
        facts = self.index.facts(pid)
        fact_bits = []
        for key, value in facts.items():
            if not value:
                continue
            if _RANGE.search(str(value)):
                continue
            fact_bits.append(self.index._fact_text(key, value))
        local = " ".join([options, sku] + fact_bits)
        title = self.index.title(pid)
        blob = self.index.blob(pid, sku)
        if self.goal.measures:
            viewed = str(pid) in self.index.viewed
            for item in self.goal.measures:
                local_has_unit = any(unit == item[1] for _value, unit, _raw in _measures(local))
                if viewed and (options.strip() or facts):
                    source = local
                elif local_has_unit:
                    source = local
                else:
                    source = (local + " " + title).strip() or blob
                if not _measure_match_exact(item, source):
                    return False
        if self.goal.res_tokens and not self._res_match(local + " " + title + " " + blob):
            if not self._relax_res():
                return False
        return True

    def _res_match(self, blob: str) -> bool:
        if not self.goal.res_tokens:
            return True
        have = set(_res_tokens(blob))
        want = set(self.goal.res_tokens)
        return bool(have & want)

    def pref_score(self, pid: str, sku: str = "") -> int:
        if not self.goal.pref_terms:
            return 0
        blob = self.index.blob(pid, sku).lower()
        return sum(1 for term in self.goal.pref_terms if _pref_hit(blob, term))

    def pref_score_strict(self, pid: str, sku: str = "") -> int:
        terms = list(self.goal.pref_terms) + list(self.goal.purpose_terms)
        if not terms:
            return 0
        blob = self.index.blob(pid, sku).lower()
        return sum(1 for term in dict.fromkeys(terms) if _pref_word_hit(blob, term))

    def rank_score(self, pid: str, sku: str) -> tuple:
        quote = self.index.quote(pid, sku) or {}
        local = (self.index.options.get(_vid(pid, sku), "") + " " + sku).strip()
        blob = self.index.blob(pid, sku)
        parts: list[float] = []
        for axis, direction in self.goal.rank_axes:
            if "price" in axis.lower():
                price = quote.get("price")
                value = -(price if price is not None else 10**12)
            else:
                local_val = _axis_number(local, axis) if local else 0.0
                value = direction * (local_val or _axis_number(blob, axis))
            parts.append(value)
        if not parts:
            price = quote.get("price")
            parts.append(-(price if price is not None else 10**12))
        return tuple(parts)

    def rank_target(self) -> tuple[str, str] | None:
        ranked: list[tuple[tuple, tuple[str, str]]] = []
        evented = _vid(*self.index.first_add) if self.index.first_add else ""
        for pid, sku in self.goal.keys:
            key = _vid(pid, sku)
            if key in self.index.dead or key == evented:
                continue
            quote = self.index.quote(pid, sku)
            if quote and self.goal.budget is not None and (quote.get("price") or 0) > self.goal.budget + _EPS:
                continue
            if quote and not quote.get("in_stock"):
                continue
            if self.index.first_add is None and not self.eligible(pid, sku):
                if not (quote and quote.get("in_stock")):
                    continue
            ranked.append((self.rank_score(pid, sku), (pid, sku)))
        ranked.sort(reverse=True)
        return ranked[0][1] if ranked else None

    def recovery_preferred(self) -> tuple[str, str] | None:
        return self.recovery_target(None)

    def recovery_target(self, dead: tuple[str, str] | None) -> tuple[str, str] | None:
        candidates: list[tuple[int, tuple[str, str]]] = []
        pids = self.model_pids()
        if not pids and self.index.first_add is not None:
            pids = self.all_pids()
        skip = {dead, self.index.first_add} if self.index.first_add else {dead}
        for pid in pids:
            for sku in [pair[1] for pair in self.variants_of(pid)] or [""]:
                if not sku:
                    continue
                if (pid, sku) in skip:
                    continue
                if not self.eligible(pid, sku):
                    continue
                level = self.relax_level(pid, sku)
                candidates.append((level, (pid, sku)))
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1] if candidates else None

    def relax_level(self, pid: str, sku: str) -> int:
        blob = self.index.blob(pid, sku)
        same_model = bool(self.goal.model) and _has_token(blob, self.goal.model)
        shade = (
            self.index.options.get(_vid(pid, sku), "")
            + " "
            + str(self.index.facts(pid).get("color") or "")
            + " "
            + self.index.title(pid)
        ).lower()
        same_color = bool(self.goal.color) and re.search(
            r"(?<![a-z])" + re.escape(self.goal.color) + r"(?![a-z])",
            shade,
        )
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        if same_color:
            return 2
        return 3

    def preference_target(self) -> tuple[str, str] | None:
        scored: list[tuple[tuple, tuple[str, str]]] = []
        evented = self.index.first_add
        opening = evented is None
        for pid in self.pref_pids() or self.all_pids():
            for pid2, sku in self.variants_of(pid) or [(pid, "")]:
                if not sku or not self.eligible(pid2, sku):
                    continue
                if evented and (pid2, sku) == evented:
                    continue
                quote = self.index.quote(pid2, sku) or {}
                price = quote.get("price") or 0.0
                scored.append(
                    (
                        (
                            1 if self._looks_like_item(pid2) else 0,
                            self.pref_score(pid2, sku),
                            self._spec_strength(pid2, sku),
                            price if opening else -price,
                        ),
                        (pid2, sku),
                    )
                )
        scored.sort(reverse=True)
        if opening and self.index.viewed:
            scored = [item for item in scored if item[1][0] in self.index.viewed]
        if scored:
            liked = [item for item in scored if self._looks_like_item(item[1][0])]
            if liked:
                scored = liked
            elif not opening and self.need_search():
                return None
            return scored[0][1]
        if self.index.first_add is not None:
            return self.best_target(brand=self.goal.brand) if self.goal.brand else self.best_target()
        return None

    def best_target(
        self,
        brand: str | None = None,
        exclude: set[str] | None = None,
        premium: bool = False,
    ) -> tuple[str, str] | None:
        blocked = set(exclude or set())
        blocked.update(self.index.dead)
        if self.index.first_add is not None:
            blocked.add(_vid(*self.index.first_add))
        found: list[tuple[str, str]] = []
        for pid in self.all_pids():
            for pair in self.variants_of(pid):
                key = _vid(*pair)
                if key in blocked:
                    continue
                if self.eligible(pair[0], pair[1], brand=brand):
                    found.append(pair)
        if not found:
            return None
        if self.index.first_add is None and self.index.viewed:
            viewed_found = [pair for pair in found if pair[0] in self.index.viewed]
            if viewed_found:
                found = viewed_found
        if self.goal.soft_pref:
            found.sort(
                key=lambda pair: (
                    self._kinship(pair),
                    self.pref_score(*pair),
                    self._price_of(*pair) if premium else -self._price_of(*pair),
                    self._query_overlap(*pair),
                ),
                reverse=True,
            )
            return found[0]
        if self.goal.keys:
            found.sort(key=lambda pair: self.rank_score(*pair), reverse=True)
            return found[0]
        found.sort(
            key=lambda pair: (
                1 if self._looks_like_item(pair[0]) else 0,
                self._kinship(pair),
                self._price_of(*pair) if premium else -self._price_of(*pair),
                self._catalog_breadth(pair[0]) if (self.goal.wants_reason and not self.goal.measures) else 0,
                self._spec_strength(*pair),
                self._query_overlap(*pair),
            ),
            reverse=True,
        )
        return found[0]

    def inspected_eligible(self, brand: str | None = None) -> list[tuple[str, str]]:
        return [
            pair
            for pair in self.eligible_pairs(brand)
            if _vid(*pair) in self.index.inspected
        ]

    def eligible_pairs(self, brand: str | None = None) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        blocked = set(self.index.dead)
        if self.index.first_add is not None:
            blocked.add(_vid(*self.index.first_add))
        for pid in self.all_pids():
            for pair in self.variants_of(pid):
                if _vid(*pair) in blocked:
                    continue
                if self.eligible(pair[0], pair[1], brand=brand):
                    found.append(pair)
        return found

    def sibling_target(self, dead: tuple[str, str] | None) -> tuple[str, str] | None:
        if not dead:
            return None
        found: list[tuple[tuple, tuple[str, str]]] = []
        for pid in self.all_pids():
            for pair in self.variants_of(pid):
                if pair == dead or pair == self.index.first_add:
                    continue
                if not self.eligible(pair[0], pair[1]):
                    continue
                kin = self._kinship(pair)
                if kin <= 0:
                    continue
                found.append(((kin, self._price_of(*pair)), pair))
        found.sort(reverse=True)
        return found[0][1] if found else None

    def _kinship(self, pair: tuple[str, str]) -> int:
        evented = self.index.first_add
        if not evented:
            return 0
        score = 0
        if _sku_family(pair[1]) and _sku_family(pair[1]) == _sku_family(evented[1]):
            score += 2
        if pair[0] == evented[0]:
            score += 1
        listing = self.index.listings.get(str(pair[0]))
        held = self.index.listings.get(str(evented[0]))
        if listing and held and listing == held:
            score += 1
        return score

    def _price_of(self, pid: str, sku: str) -> float:
        price = (self.index.quote(pid, sku) or {}).get("price")
        if price is None:
            return 0.0
        return float(price)

    def best_live(self, exclude: set[str] | None = None) -> tuple[str, str] | None:
        exclude = exclude or set()
        found: list[tuple[str, str]] = []
        for pid in self.all_pids():
            for pair in self.variants_of(pid):
                if _vid(*pair) in exclude:
                    continue
                if self.index.live(pair[0], pair[1], self.goal.budget):
                    found.append(pair)
        if not found:
            return None
        found.sort(
            key=lambda pair: (
                -self.pref_score(*pair),
                (self.index.quote(*pair) or {}).get("price")
                if (self.index.quote(*pair) or {}).get("price") is not None
                else 10**12,
            )
        )
        return found[0]

    def justification(self, target: tuple[str, str]) -> str:
        return "; ".join(
            "%s=%s" % (item["field"], item["value"]) for item in self.structured_claims(target)
        )

    def fact_claims(self, target: tuple[str, str] | None) -> list[str]:
        if not target:
            return ["in_stock=unknown"]
        return ["%s=%s" % (item["field"], item["value"]) for item in self.structured_claims(target)]

    def fact_sheet(self, target: tuple[str, str] | None) -> str:
        return " | ".join(self.fact_claims(target))

    def _order_schema(self) -> dict[str, Any]:
        return self.kernel.param("place_test_order")

    def _claim_fields(self) -> list[str]:
        props = (self._order_schema().get("properties") or {}).get("justification") or {}
        claims = (props.get("properties") or {}).get("claims") or {}
        field = ((claims.get("items") or {}).get("properties") or {}).get("field") or {}
        enum = field.get("enum")
        if isinstance(enum, list) and enum:
            return [str(item) for item in enum]
        return []

    def structured_claims(self, target: tuple[str, str] | None) -> list[dict[str, str]]:
        if not target:
            return []
        pid, sku = target
        fields = self._claim_fields()
        allowed = set(fields) if fields else None
        quote = self.index.quote(pid, sku) or {}
        facts = self.index.facts(pid)
        blob = self.index.blob(pid, sku)
        low = blob.lower()
        out: list[dict[str, str]] = []

        def add(field: str, value: Any) -> None:
            if allowed is not None and field not in allowed:
                return
            if value in (None, ""):
                return
            text = str(value).strip()
            if not text:
                return
            if any(item["field"] == field for item in out):
                return
            out.append({"field": field, "value": text})

        if quote.get("price") is not None:
            add("price", "%.2f" % quote["price"])
        if quote.get("in_stock") is not None:
            add("in_stock", "true" if quote.get("in_stock") else "false")
        wanted_brand = self.goal.brand or self.goal.pref_brand
        seen_brand = self.index.brand_of(pid)
        if wanted_brand and (
            wanted_brand.lower() in low
            or (seen_brand and wanted_brand.lower() in seen_brand.lower())
        ):
            add("brand", wanted_brand)
        for value, unit, raw in self.goal.measures:
            if not _measure_match_exact((value, unit, raw), blob):
                continue
            hints = ("size", "screen", "inch") if unit == "in" else ("refresh", "hz", "rate")
            field = None
            pool = list(allowed) if allowed is not None else [str(key) for key in facts]
            for name in pool:
                low_name = name.lower().replace("_", " ")
                if any(word in low_name for word in hints):
                    field = name
                    break
            if field is None and allowed is None:
                field = "screen_size" if unit == "in" else ("refresh_rate" if unit == "hz" else None)
            if field:
                add(field, ("%gin" % value) if unit == "in" else raw)
                break
        observed = str(facts.get("display") or facts.get("resolution") or "") + " " + blob
        have = set(_res_tokens(observed))
        overlap = have & set(self.goal.res_tokens)
        for token in self.goal.res_tokens:
            if token in overlap:
                add("resolution", token)
                break
        return out

    def order_args(self, target: tuple[str, str]) -> dict[str, Any]:
        args: dict[str, Any] = {}
        schema = self._order_schema()
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        if "product_id" in props:
            args["product_id"] = target[0]
        if "sku" in props:
            args["sku"] = target[1]
        just = props.get("justification")
        just_is_object = isinstance(just, dict) and (
            "claims" in ((just.get("properties") or {}))
            or (just.get("type") == "object")
        )
        must_claim = self.need_order_claims or "justification" in required
        if just_is_object and must_claim:
            args["justification"] = {"claims": self.structured_claims(target)}
        elif "claims" in props and must_claim:
            args["claims"] = self.structured_claims(target)
        return args


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    return Supervisor(problem_data).run()


__all__ = ["agent_main", "Goal", "infer_family", "parse_goal"]
