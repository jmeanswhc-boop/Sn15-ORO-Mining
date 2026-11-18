"""A personal shopping assistant that buys on a customer's behalf.

The assistant is handed a customer's request and a storefront it can search,
filter, read and buy from. It works the way a careful human shopper does: it
settles what the customer actually needs, casts a wide net over the catalogue,
opens only the listings that look worth opening, compares the survivors on the
customer's own terms, and checks price and availability one last time before
spending any money.

Judgement is split according to what each kind of reasoning is good at. A
language model reads the customer's words and pulls values out of messy product
copy; it is never trusted with arithmetic, and never with a fact it cannot point
at. Every value it reports is checked back against the raw listing before it can
affect a decision, while budgets, availability, ranking and the claims made at
checkout are settled in code against what was actually observed.

The storefront's own tools are discovered from their published schemas rather
than assumed, so the assistant works against whatever catalogue it is pointed
at, under whatever names that catalogue gives its tools.
"""

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any, Iterable, Sequence

from src.agent.proxy_client import ProxyClient

# Provider-paired defaults, overridable by the sandbox configuration.
_PROVIDER_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}

# Discovery is deliberately lopsided. A search box has to be guessed at, and a
# handful of well-chosen phrases covers a catalogue better than a pile of
# near-duplicates, so phrases are rationed and one is always held back until the
# early results reveal what this shop calls the thing. Facet filters ask the
# storefront a precise question instead of guessing at its vocabulary, so they
# are swept freely, from the tightest combination outwards.
_MAX_SEARCH_PHRASES = 5
# Passes over the shop. Self-limiting in practice: once every facet combination
# has been asked the sweep returns nothing, so the ceiling only matters while
# there is still somewhere unlooked.
_MAX_SWEEPS = 8
_BRAND_SWEEPS = 4
_BRAND_SWEEPS_DEEP = 12
_PRICE_BANDS = 5
_PAGE = 10
_COMPARE_GROUP = 5
# Opening a listing costs attention and says "this looked right to me", so the
# assistant reads a focused shortlist rather than the whole results page.
_MAX_OPENED = 16
# Reached only when a fixed measurement cannot be settled from the listings
# themselves, where looking inside is the whole job rather than an indulgence.
_MAX_OPENED_DEEP = 40
_RELEVANCE_FLOOR = 0.45
# A marketplace lists spares, cases and cables beside the products they fit.
# Anything priced at a small fraction of the going rate for what else the
# customer could buy is almost certainly one of those rather than the thing
# itself, however well its title matches.
_CLASS_PRICE_FLOOR = 0.2
# Long enough that no price, measurement or model number a customer would
# actually write reaches it.
_REFERENCE_DIGITS = 16
# Turns held back for read -> add -> re-read -> order, plus slack for one
# recovery cycle and a final reply if the market or the customer moves late.
_COMMIT_RESERVE = 8
_MAX_REPLIES = 8
_MAX_CLAIMS = 5
_CLAIM_ASKED = 4
# Attribute values are short by nature; anything longer is marketing copy that
# happens to mention the value rather than a statement of it.
_MAX_CLAIM_CHARS = 60
# Turns to hold off paying while a question to the customer is outstanding.
_ANSWER_GRACE = 3
# Enough to answer a shifting market, not so many that a shop which reprices
# on every turn keeps the errand going forever.
_MAX_RESURVEYS = 4
# Contenders re-priced after the shop moves; enough to cover a shortlist a
# customer would name, not the whole shelf.
_RECHECK_WIDTH = 8
_EVIDENCE_CHARS = 900
_PRICE_EPSILON = 0.005
_TWIN_SIMILARITY = 0.55
_MIN_SURVEY = 4
_SOFT_OPEN_BONUS = 10

_WORD_RE = re.compile(r"[a-z0-9]+")
# A figure and the unit it is written in, as sellers put them in a title.
# Any alphabetic unit is accepted — no product-suite unit allowlist.
_MEASURE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*([a-z]{1,12})\b", re.IGNORECASE)
# Goal says a firm detail arrives only after the assistant asks.
_ASK_RE = re.compile(
    r"(?:if you ask|tell you if you ask|ask me|i will tell you|when you ask)",
    re.IGNORECASE,
)
# Generic retrieval cues: customer points at a phrase or a shelf department.
_MATCH_SPAN_RE = re.compile(
    r"\bmatch(?:es|ing)?\s+(?:the\s+)?(?:phrase\s+)?(?P<body>[^\n,]{2,80})",
    re.IGNORECASE,
)
_CATEGORY_SPAN_RE = re.compile(
    r"\bin the\s+(?P<body>[^,\n]{2,80}?)\s+category\b",
    re.IGNORECASE,
)
_MAX_MARKERS = ("highest", "largest", "maximum", "max", "most")
_MIN_MARKERS = ("lowest", "smallest", "minimum", "min", "least", "cheapest")
# Discourse glue after a max/min marker — not product vocabulary.
_PRIORITY_STOP = frozenset(
    {
        "among",
        "remaining",
        "ties",
        "tie",
        "choose",
        "pick",
        "select",
        "please",
        "compare",
        "only",
        "these",
        "those",
        "between",
        "then",
        "next",
        "finally",
        "still",
        "also",
        "both",
        "either",
        "option",
        "options",
        "candidate",
        "candidates",
        "variant",
        "variants",
        "tradeoff",
        "tradeoffs",
        "one",
        "ones",
        "item",
        "items",
    }
)
_PRIORITY_ARTICLES = frozenset({"a", "an", "the", "of", "to", "for"})
# Straight and typographic quotation marks alike; customers use both.
_QUOTED_RE = re.compile('"([^"]{2,60})"|\u201c([^\u201d]{2,60})\u201d')
# Grouped thousands first, so a written thousands-separator amount stays one
# figure while a short comma-separated list still reads as several figures.
_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_CURRENCY_SYMBOLS = "$\u00a3\u20ac\u20b1\u00a5\u20b9\u20a9"
_MONEY_RE = re.compile(
    r"(?P<lead>[" + _CURRENCY_SYMBOLS + r"]|\b[a-z]{3}\b)?\s*"
    r"(?P<amount>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<trail>\b[a-z]{3}\b)?",
    re.IGNORECASE,
)

_UNSET = object()


# --------------------------------------------------------------------------- #
# generic text helpers
# --------------------------------------------------------------------------- #


def _text(value: Any) -> str:
    """Render any observation fragment as plain text."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _number_text(float(value))
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_text(item) for item in value)
    return str(value)


def _number_text(value: float) -> str:
    """Render a number the way a catalogue prints it."""

    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(round(float(value), 6))


def _words(value: Any) -> list[str]:
    return _WORD_RE.findall(_text(value).lower())


def _word_set(value: Any) -> set[str]:
    return set(_words(value))


def _norm(value: Any) -> str:
    return " ".join(_words(value))


def _customer_voice(text: str) -> str:
    """The shopper's own words, without trailing task-instruction boilerplate.

    Public goals often append a generic 'Task rules:' block. Preference cues,
    keyword splits and priority phrases must come from what the customer said,
    not from the shared instruction text that follows.
    """

    raw = _text(text)
    cut = re.split(r"\btask\s+rules\b", raw, maxsplit=1, flags=re.IGNORECASE)[0]
    return cut.strip() or raw.strip()


def _numbers(value: Any) -> list[float]:
    out: list[float] = []
    for token in _NUMBER_RE.findall(_text(value)):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    found = _numbers(value)
    return found[0] if len(found) == 1 else None


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _text_field(node: dict[str, Any], field: str) -> str:
    """A trimmed string field, or "" when absent or of another type."""

    value = node.get(field)
    return value.strip() if isinstance(value, str) else ""


def _overlap(left: Iterable[str], right: Iterable[str]) -> int:
    return len(set(left) & set(right))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(len(left | right))


def _rows(value: Any) -> list[dict[str, Any]]:
    """Every mapping in a value, whether it arrives bare or in a list."""

    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, dict)]
    return []


def _json_object(raw: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of a model reply."""

    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(raw[start : index + 1])
                except ValueError:
                    start = -1
                    continue
                return parsed if isinstance(parsed, dict) else None
    return None


def _money_in(text: str, currency: str) -> list[float]:
    """Amounts written with a currency marker, so specs are not read as prices."""

    amounts: list[float] = []
    code = _norm(currency)
    for match in _MONEY_RE.finditer(text):
        lead = (match.group("lead") or "").lower()
        trail = (match.group("trail") or "").lower()
        marked = lead in _CURRENCY_SYMBOLS and bool(lead)
        if not marked and code:
            marked = code in (lead, trail)
        if not marked:
            continue
        try:
            amounts.append(float(match.group("amount").replace(",", "")))
        except ValueError:
            continue
    return amounts


def _value_matches(required: Any, observed: Any) -> bool:
    """Does an observed attribute satisfy a required value?

    Numeric requirements compare by magnitude (same figure with optional unit
    variants), and never treat a different magnitude as a match. Everything
    else falls back to token containment.
    """

    want_raw = _text(required)
    have_raw = _text(observed)
    want = _norm(want_raw)
    have = _norm(have_raw)
    if not want or not have:
        return False
    if want in have:
        return True
    want_numbers = _numbers(want_raw)
    if want_numbers:
        have_numbers = _numbers(have_raw)
        if not have_numbers:
            return False
        for wanted in want_numbers:
            if not any(abs(wanted - found) <= 1e-6 for found in have_numbers):
                return False
        want_unit = _unit_of(want_raw)
        have_unit = _unit_of(have_raw)
        if want_unit and have_unit:
            return _units_compatible(want_unit, have_unit)
        # Magnitude already matched; a missing unit on one side is tolerated.
        return True
    want_words = _word_set(want)
    return bool(want_words) and want_words <= _word_set(have)


def _expects_reply(text: str) -> bool:
    """Conversational cue that the other side is waiting on us."""

    if "?" in text or "\uff1f" in text:
        return True
    return bool(
        _word_set(text)
        & {"confirm", "reply", "respond", "answer", "tell", "know", "clarify"}
    )


# Dialogue / shopping glue only — not product-suite vocabulary.
_PREF_MARKERS = (
    "prefer",
    "rather",
    "priority",
    "care about",
    "matters",
    "ideally",
    "would rather",
)
_LIGHT_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "with",
        "from",
        "that",
        "this",
        "to",
        "of",
        "in",
        "on",
        "is",
        "are",
        "be",
        "as",
        "at",
        "by",
        "it",
        "my",
        "me",
        "i",
        "we",
        "you",
        "your",
        "need",
        "want",
        "please",
        "also",
        "than",
        "then",
        "into",
        "about",
        "within",
        "must",
        "should",
        "can",
        "will",
        "not",
        "any",
        "all",
        "one",
        "get",
        "buy",
        "find",
        "item",
        "product",
        "option",
        "options",
        "true",
        "false",
        "use",
        "using",
        "have",
        "has",
        "had",
        "their",
        "them",
        "they",
        "our",
        "out",
        "over",
        "under",
        "only",
        "just",
        "more",
        "most",
        "very",
        "such",
        "same",
        "other",
        "each",
        "every",
        "both",
        "some",
        "no",
        "so",
        "too",
        "bit",
        "pay",
        "place",
        "order",
        "test",
        "stock",
        "price",
        "budget",
        "category",
        "listing",
        "available",
        "current",
        "prefer",
        "rather",
        "priority",
        "ideally",
        "matters",
        "matter",
        "care",
        "about",
        "but",
        "firm",
        "requirements",
        "requirement",
        "settle",
        "something",
        "flimsy",
        "under",
        "over",
        "above",
        "below",
        "around",
        "than",
        "up",
        "per",
        "match",
        "matches",
        "matching",
        "listing",
        "listings",
        "compare",
        "compared",
        "candidate",
        "candidates",
        "variant",
        "variants",
    }
)


def _preference_tokens(text: str) -> list[str]:
    """Distinctive tokens from preference-leaning language in the public goal.

    Prefer/rather clauses are primary. Neighbouring use-case clauses ("for …")
    in the same request are included so the job description travels with the
    soft preference — without dumping every word of the whole goal.
    """

    raw = _customer_voice(text)
    low_full = raw.casefold()
    if not any(marker in low_full for marker in _PREF_MARKERS):
        return []
    chunks: list[str] = []
    for part in re.split(r"[.!?\n;]", raw):
        low = part.casefold()
        prefer_clause = any(marker in low for marker in _PREF_MARKERS)
        use_clause = bool(re.search(r"\bfor\b", low)) and not prefer_clause
        if not prefer_clause and not use_clause:
            continue
        # Keep the preferred side of "rather X than Y" / "prefer X but Y".
        for sep in (" than ", " but ", " instead of ", " rather than "):
            if prefer_clause and sep in low:
                part = part[: low.find(sep)]
                low = part.casefold()
                break
        chunks.extend(
            word
            for word in _words(part)
            if word not in _LIGHT_STOP and len(word) > 2 and not word.isdigit()
        )
    ordered: list[str] = []
    seen: set[str] = set()
    for word in chunks:
        if word not in seen:
            seen.add(word)
            ordered.append(word)
    return ordered


def _wants_quality(text: str) -> bool:
    """Whether the shopper asked not to settle for the cheapest fit."""

    low = _customer_voice(text).casefold()
    return any(marker in low for marker in _PREF_MARKERS)


def _units_compatible(left: str, right: str) -> bool:
    """Whether two unit spellings name the same kind of magnitude.

    Handles plurals and ordinary abbreviations by shared stem / prefix — not by
    a fixed product-suite unit table.
    """

    a = _text(left).casefold().strip()
    b = _text(right).casefold().strip()
    if not a or not b:
        return False
    if a == b:
        return True
    stem_a = a[:-1] if a.endswith("s") and len(a) > 3 else a
    stem_b = b[:-1] if b.endswith("s") and len(b) > 3 else b
    if stem_a == stem_b:
        return True
    short, long = (stem_a, stem_b) if len(stem_a) <= len(stem_b) else (stem_b, stem_a)
    return len(short) >= 2 and long.startswith(short)


def _money_spans(text: str) -> list[tuple[int, int]]:
    """Character spans that look like priced amounts, so specs skip them.

    Only symbol-led or code-trailed amounts count. A bare three-letter token
    before a number is often a brand or model fragment, not currency.
    """

    spans: list[tuple[int, int]] = []
    for match in _MONEY_RE.finditer(_text(text)):
        lead = (match.group("lead") or "").strip()
        trail = (match.group("trail") or "").strip()
        if trail or (lead and any(ch in lead for ch in _CURRENCY_SYMBOLS)):
            spans.append(match.span())
    return spans


def _extract_specs(text: str) -> list[dict[str, str]]:
    """Measurements the shopper wrote, as generic {field, value} constraints.

    Field names come from neighbouring customer words or the unit token itself —
    never from a product-suite synonym map.
    """

    raw = _text(text)
    low = raw.casefold()
    priced = _money_spans(raw)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in _MEASURE_RE.finditer(low):
        start, end = match.span()
        if any(start < right and end > left for left, right in priced):
            continue
        num, unit = match.group(1), match.group(2).casefold()
        if not num or not unit or unit.isdigit():
            continue
        before = _WORD_RE.findall(low[max(0, start - 40) : start])
        after = _WORD_RE.findall(low[end : end + 40])
        candidates: list[tuple[int, str]] = []
        for index, word in enumerate(reversed(before)):
            if word not in _LIGHT_STOP and not word.isdigit():
                candidates.append((index, word))
                break
        for index, word in enumerate(after):
            if word not in _LIGHT_STOP and not word.isdigit():
                candidates.append((index, word))
                break
        candidates.sort(key=lambda row: row[0])
        label = candidates[0][1] if candidates else ""
        field = label or unit
        value = f"{num} {unit}"
        key = f"{_norm(field)}::{_norm(value)}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"field": field, "value": value})
    return out


def _priority_hints(text: str) -> list[dict[str, str]]:
    """Tie-break directions the shopper stated in plain language."""

    words = _words(_customer_voice(text))
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, word in enumerate(words):
        direction = ""
        if word in _MAX_MARKERS:
            direction = "max"
        elif word in _MIN_MARKERS:
            direction = "min"
        if not direction:
            continue
        # Keep one or two content words after the marker as the field name.
        # Articles are skipped; discourse glue stops the span. Words that are
        # also dialogue-stop tokens (e.g. price) are still kept here because
        # they are the specification being ranked.
        taken: list[str] = []
        for ahead in words[index + 1 : index + 5]:
            if (
                ahead in _PRIORITY_STOP
                or ahead in _MAX_MARKERS
                or ahead in _MIN_MARKERS
            ):
                if taken:
                    break
                continue
            if ahead.isdigit():
                if taken:
                    break
                continue
            if not taken and ahead in _PRIORITY_ARTICLES:
                continue
            taken.append(ahead)
            if len(taken) >= 2:
                break
        if not taken:
            continue
        field = " ".join(taken)
        key = f"{direction}:{_norm(field)}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"field": field, "direction": direction})
    return out


def _collect_measures(text: str) -> list[tuple[float, str]]:
    """Every (figure, unit) pair printed as a measurement in text.

    Priced amounts (currency symbol or trailing currency code) are skipped so
    money is not treated as a physical unit during ranking. Three-letter
    alphabetic units are treated as currency codes, not physical measures.
    """

    raw = _text(text)
    low = raw.casefold()
    priced = _money_spans(raw)
    out: list[tuple[float, str]] = []
    for match in _MEASURE_RE.finditer(low):
        if any(match.start() < right and match.end() > left for left, right in priced):
            continue
        number, unit = match.group(1), match.group(2).casefold()
        if not unit or unit.isdigit():
            continue
        # Single-letter "units" are almost always model-token tails (…40E).
        if len(unit) < 2:
            continue
        # ISO-style currency trails (php, usd, …) are not physical units.
        if len(unit) == 3 and unit.isalpha():
            continue
        try:
            out.append((float(number), unit))
        except ValueError:
            continue
    return out


def _nearby_magnitudes(text: str, field: str) -> list[float]:
    """Numbers printed nearest the customer's specification words.

    Prefers measured figures (number+unit) closest to the field phrase so a
    neighbouring unrelated magnitude does not win the comparison.
    """

    raw = _text(text)
    if not raw or not field:
        return []
    low = raw.casefold()
    needles = [word for word in _words(field) if word not in _PRIORITY_ARTICLES]
    if not needles:
        needles = _words(field)
    if not needles:
        return []
    anchors: list[int] = []
    phrase = _norm(field)
    if phrase and len(phrase) >= 3:
        pattern = re.compile(
            r"{0}".format(r"\s+".join(re.escape(part) for part in phrase.split())),
            re.IGNORECASE,
        )
        anchors.extend(match.start() for match in pattern.finditer(raw))
    if not anchors:
        for needle in needles:
            start = 0
            while True:
                at = low.find(needle, start)
                if at < 0:
                    break
                anchors.append(at)
                start = at + len(needle)
    if not anchors:
        return []
    scored: list[tuple[int, float]] = []
    for match in _MEASURE_RE.finditer(low):
        try:
            figure = float(match.group(1))
        except ValueError:
            continue
        dist = min(abs(match.start() - anchor) for anchor in anchors)
        if dist <= 28:
            scored.append((dist, figure))
    if not scored:
        for match in _NUMBER_RE.finditer(raw):
            try:
                figure = float(match.group(0).replace(",", ""))
            except ValueError:
                continue
            dist = min(abs(match.start() - anchor) for anchor in anchors)
            if dist <= 18:
                scored.append((dist, figure))
    if not scored:
        return []
    scored.sort(key=lambda row: row[0])
    # Keep only the closest magnitude cluster so a neighbour figure cannot tie.
    best = scored[0][0]
    out: list[float] = []
    seen: set[float] = set()
    for dist, figure in scored:
        if dist > best + 2:
            break
        if figure in seen:
            continue
        seen.add(figure)
        out.append(figure)
    return out


# Structural shapes of a shopping goal. Derived only from the public request
# text (and later the brief), never from suite IDs or product vocab tables.
_SHAPE_NAMED = "named"
_SHAPE_CLARIFY = "clarify"
_SHAPE_QUOTE = "quote"
_SHAPE_PREFER = "prefer"
_SHAPE_SPEC = "spec"
_SHAPE_ORDER = "order"
_SHAPE_FIND = "find"


class GoalProfile:
    """Which shopping algorithms this goal should run.

    Shapes may combine. The primary shape picks the dominant search / score /
    recommend path; flags refine it.
    """

    __slots__ = (
        "primary",
        "named",
        "clarify",
        "quoted",
        "prefer",
        "measured",
        "ordered",
        "keywords",
        "phrases",
    )

    def __init__(self, request: str) -> None:
        voice = _customer_voice(request)
        pointed = _pointed_at(voice)
        quotes = list(_quoted_in(voice)) + _match_spans(voice)
        specs = _extract_specs(voice)
        prefs = _preference_tokens(voice)
        priorities = _priority_hints(voice)
        self.named = any("::" in item or item.isdigit() for item in pointed)
        self.clarify = bool(_ASK_RE.search(voice))
        self.quoted = bool(quotes)
        self.prefer = bool(prefs)
        self.measured = bool(specs)
        self.ordered = bool(priorities)
        self.keywords = _split_keywords(voice) + _model_codes(voice)
        self.phrases = _search_phrase_plan(voice, self.keywords, quotes, specs)
        # Soft-preference vocabulary belongs in the search box early so the
        # shelf is filtered toward cue-bearing listings before price sorting.
        if prefs:
            pack = " ".join(prefs[:4])
            if pack and _norm(pack) not in {_norm(p) for p in self.phrases}:
                self.phrases.insert(0, pack)
            if len(prefs) >= 2:
                bigram = f"{prefs[0]} {prefs[1]}"
                if _norm(bigram) not in {_norm(p) for p in self.phrases}:
                    self.phrases.insert(0, bigram)
        # Compact model codes the shopper named — search seeds, not listing ids.
        for code in _model_codes(voice):
            if _norm(code) not in {_norm(p) for p in self.phrases}:
                self.phrases.insert(0, code)
        # Dominant path: named shortlists and deferred asks outrank the rest.
        if self.named:
            self.primary = _SHAPE_NAMED
        elif self.clarify and not specs:
            self.primary = _SHAPE_CLARIFY
        elif self.quoted:
            self.primary = _SHAPE_QUOTE
        elif self.prefer:
            self.primary = _SHAPE_PREFER
        elif self.ordered:
            self.primary = _SHAPE_ORDER
        elif self.measured:
            self.primary = _SHAPE_SPEC
        else:
            self.primary = _SHAPE_FIND

    def refresh(self, brief: "Brief", transcript: Sequence[str]) -> None:
        """Update flags once the brief / conversation fills gaps."""

        if brief.missing:
            self.clarify = True
        if brief.must:
            self.measured = True
            # A filled firm slot means the clarify gap is closed.
            if not brief.missing:
                self.clarify = False
        if brief.nice or any(_preference_tokens(turn) for turn in transcript):
            self.prefer = True
        if brief.priorities:
            self.ordered = True
        if self.named:
            self.primary = _SHAPE_NAMED
        elif self.clarify and not brief.must:
            self.primary = _SHAPE_CLARIFY
        elif self.quoted:
            self.primary = _SHAPE_QUOTE
        elif self.prefer:
            self.primary = _SHAPE_PREFER
        elif self.ordered:
            self.primary = _SHAPE_ORDER
        elif self.measured or brief.must:
            self.primary = _SHAPE_SPEC
            self.measured = True
        else:
            self.primary = _SHAPE_FIND
        # Fold brief search seeds and category into the live phrase list.
        extra: list[str] = []
        if brief.category:
            extra.append(brief.category)
        if brief.brand:
            extra.append(brief.brand)
            if brief.category:
                extra.append(f"{brief.brand} {brief.category}")
        for item in brief.must[:2]:
            extra.append(_text(item["value"]))
        for phrase in list(brief.queries) + extra:
            if phrase.strip() and _norm(phrase) not in {_norm(p) for p in self.phrases}:
                self.phrases.append(phrase.strip())


def _match_spans(text: str) -> list[str]:
    """Free-text spans the customer asked to match, outside quotation marks."""

    out: list[str] = []
    for match in _MATCH_SPAN_RE.finditer(_text(text)):
        body = match.group("body").strip(" \t\"'\u201c\u201d.")
        # Stop at task-rule boilerplate if the sentence continues.
        body = re.split(r"\bat or\b|\band buy\b|\.\s", body, maxsplit=1)[0].strip()
        if len(body) >= 2:
            out.append(body)
    return out


def _category_spans(text: str) -> list[str]:
    """Department names written as 'in the … category'."""

    out: list[str] = []
    for match in _CATEGORY_SPAN_RE.finditer(_text(text)):
        body = match.group("body").strip()
        if body:
            out.append(body)
    return out


def _split_keywords(text: str) -> list[str]:
    """Split a goal into searchable content tokens, preserving order.

    Quoted spans and number+unit measurements stay atomic; budget-like money
    figures and dialogue glue are dropped.
    """

    raw = _text(text)
    low = raw.casefold()
    claimed: list[tuple[int, int, str]] = []
    for phrase in list(_quoted_in(raw)) + _match_spans(raw):
        at = low.find(phrase.casefold())
        if at >= 0:
            claimed.append((at, at + len(phrase), phrase.strip()))
    for match in _MEASURE_RE.finditer(low):
        if any(match.start() < right and match.end() > left for left, right, _ in claimed):
            continue
        # Skip magnitudes that sit inside a priced amount span.
        if any(match.start() < right and match.end() > left for left, right in _money_spans(raw)):
            continue
        claimed.append((match.start(), match.end(), match.group(0).replace("-", " ")))
    for left, right in _money_spans(raw):
        claimed.append((left, right, ""))
    claimed.sort(key=lambda row: row[0])

    parts: list[str] = []
    cursor = 0
    for left, right, kept in claimed:
        if left > cursor:
            parts.append(raw[cursor:left])
        if kept:
            parts.append(kept)
        cursor = max(cursor, right)
    parts.append(raw[cursor:])

    tokens: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part.strip():
            continue
        # Keep measurement / quote chunks intact when they look like one term.
        stripped = part.strip()
        if _MEASURE_RE.fullmatch(stripped.casefold()) or (
            len(stripped.split()) <= 6
            and (
                stripped in _quoted_in(raw)
                or stripped in _match_spans(raw)
            )
        ):
            key = _norm(part)
            if key and key not in seen:
                seen.add(key)
                tokens.append(stripped)
            continue
        for word in _words(part):
            if word in _LIGHT_STOP or len(word) < 3 or word.isdigit():
                continue
            if word in seen:
                continue
            seen.add(word)
            tokens.append(word)
    return tokens


def _search_phrase_plan(
    request: str,
    keywords: Sequence[str],
    quotes: Sequence[str],
    specs: Sequence[dict[str, str]],
) -> list[str]:
    """Build a short, de-duplicated list of search box phrases for this goal."""

    seeds: list[str] = []
    seeds.extend(quotes)
    seeds.extend(_match_spans(request))
    for category in _category_spans(request):
        seeds.append(category)
        # Category plus the tightest keyword pack finds the right shelf faster.
        useful = [
            word
            for word in keywords
            if word not in _LIGHT_STOP and _norm(word) not in _norm(category)
        ][:3]
        if useful:
            seeds.append(f"{category} {' '.join(useful)}")
    for item in specs[:2]:
        seeds.append(_text(item["value"]))
        if item.get("field") and item["field"] not in _LIGHT_STOP:
            seeds.append(f"{item['field']} {_text(item['value'])}")
    # Progressive keyword packs: tight → loose.
    useful = [word for word in keywords if word not in _LIGHT_STOP]
    if useful:
        seeds.append(" ".join(useful[:6]))
        if len(useful) > 3:
            seeds.append(" ".join(useful[:3]))
        seeds.append(useful[0])
        # Bigram walk covers multi-word product names without a synonym table.
        for index in range(len(useful) - 1):
            seeds.append(f"{useful[index]} {useful[index + 1]}")
    # Capitalised runs from the raw request often name brands/models.
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9][A-Za-z0-9\-']{1,24})\b", request):
        token = match.group(1)
        # Skip all-caps three-letter tokens (usually currency codes next to money).
        if token.isupper() and len(token) == 3:
            continue
        if token.casefold() not in _LIGHT_STOP:
            seeds.append(token)
    ordered: list[str] = []
    seen: set[str] = set()
    for seed in seeds:
        key = _norm(seed)
        if not key or key in seen or len(key) < 2:
            continue
        seen.add(key)
        ordered.append(seed.strip())
    return ordered


# --------------------------------------------------------------------------- #
# observation model
# --------------------------------------------------------------------------- #


class Offer:
    """One purchasable variant, as last seen in a tool observation."""

    __slots__ = (
        "product_id",
        "sku",
        "price",
        "currency",
        "in_stock",
        "options",
        "title",
        "brand",
        "seen_at",
        "read_at",
        "moved",
    )

    def __init__(self, product_id: str, sku: str) -> None:
        self.product_id = product_id
        self.sku = sku
        self.price: float | None = None
        self.currency: str = ""
        self.in_stock: bool | None = None
        self.options: str = ""
        self.title: str = ""
        self.brand: str = ""
        self.seen_at = 0
        self.read_at = 0
        self.moved = False

    @property
    def key(self) -> str:
        return f"{self.product_id}::{self.sku}"


class Product:
    """A catalogue listing and everything observed about it."""

    __slots__ = ("product_id", "title", "brand", "category_path", "facts", "studied")

    def __init__(self, product_id: str) -> None:
        self.product_id = product_id
        self.title = ""
        self.brand = ""
        self.category_path: list[str] = []
        self.facts: dict[str, Any] = {}
        self.studied = False


class WorldModel:
    """Everything the agent has actually seen, indexed for lookup.

    Nothing is assumed about payload layout beyond "a mapping carrying a product
    identifier describes a product or an offer", so the model keeps working when
    the environment revises its observation shapes.
    """

    def __init__(self) -> None:
        self.products: dict[str, Product] = {}
        self.offers: dict[str, Offer] = {}
        self.cart: list[str] = []
        self.moved: list[str] = []
        self.clock = 0

    # -- ingestion ---------------------------------------------------------- #

    def ingest(self, payload: Any, *, studied: bool = False) -> None:
        self._walk(payload, studied=studied, depth=0)

    def _walk(self, node: Any, *, studied: bool, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                self._walk(item, studied=studied, depth=depth + 1)
            return
        if not isinstance(node, dict):
            return
        product_id = node.get("product_id")
        if isinstance(product_id, str) and product_id:
            if any(key in node for key in ("variants", "facts", "category_path")):
                self._absorb_product(product_id, node, studied=studied)
            else:
                self._absorb_offer(product_id, node)
        for key, value in node.items():
            if key == "cart" and isinstance(value, list):
                self.cart = [
                    f"{row.get('product_id')}::{row.get('sku')}"
                    for row in _rows(value)
                    if row.get("product_id") and row.get("sku")
                ]
            self._walk(value, studied=studied, depth=depth + 1)

    def _absorb_product(
        self, product_id: str, node: dict[str, Any], *, studied: bool
    ) -> None:
        product = self.product(product_id)
        product.title = _text_field(node, "title") or product.title
        product.brand = _text_field(node, "brand") or product.brand
        path = node.get("category_path")
        if isinstance(path, list) and path:
            product.category_path = [str(item) for item in path]
        facts = node.get("facts")
        if not isinstance(facts, dict) or not facts:
            facts = node.get("attributes")
        if isinstance(facts, dict) and facts:
            product.facts = dict(facts)
        product.studied = product.studied or studied
        for row in _rows(node.get("variants")):
            self._absorb_offer(product_id, {**row, "product_id": product_id})

    def _absorb_offer(self, product_id: str, node: dict[str, Any]) -> None:
        sku = node.get("sku")
        if not isinstance(sku, str) or not sku:
            return
        key = f"{product_id}::{sku}"
        offer = self.offers.get(key)
        if offer is None:
            offer = Offer(product_id, sku)
            self.offers[key] = offer
        price = _as_float(node.get("price"))
        if price is None:
            price = _as_float(node.get("current_price"))
        in_stock = _as_bool(node.get("in_stock"))
        if price is not None and offer.price is not None:
            if abs(price - offer.price) > _PRICE_EPSILON:
                self._mark_moved(offer)
        if in_stock is not None and offer.in_stock is not None:
            if in_stock != offer.in_stock:
                self._mark_moved(offer)
        if price is not None:
            offer.price = price
        if in_stock is not None:
            offer.in_stock = in_stock
        offer.currency = _text_field(node, "currency") or offer.currency
        offer.options = _text_field(node, "options") or offer.options
        offer.title = _text_field(node, "title") or offer.title
        offer.brand = _text_field(node, "brand") or offer.brand
        offer.seen_at = self.clock
        if price is not None or in_stock is not None:
            offer.read_at = self.clock
        product = self.product(product_id)
        if not product.title and offer.title:
            product.title = offer.title
        if not product.brand and offer.brand:
            product.brand = offer.brand

    def _mark_moved(self, offer: Offer) -> None:
        offer.moved = True
        if offer.key not in self.moved:
            self.moved.append(offer.key)

    # -- lookup ------------------------------------------------------------- #

    def product(self, product_id: str) -> Product:
        found = self.products.get(product_id)
        if found is None:
            found = Product(product_id)
            self.products[product_id] = found
        return found

    def listing(self, offer: Offer) -> Product:
        return self.product(offer.product_id)

    def title_of(self, offer: Offer) -> str:
        return self.listing(offer).title or offer.title

    def evidence(self, offer: Offer) -> str:
        """Everything observed about one variant, as a single text blob."""

        product = self.listing(offer)
        parts = [
            self.title_of(offer),
            product.brand or offer.brand,
            " ".join(product.category_path),
            offer.options,
            _text(product.facts),
        ]
        if offer.price is not None:
            parts.append(f"price {_number_text(offer.price)} {offer.currency}")
        if offer.in_stock is not None:
            parts.append("in stock " + ("true" if offer.in_stock else "false"))
        return " | ".join(part for part in parts if part)

    def sources(self, offer: Offer) -> list[tuple[str, Any]]:
        """Named values observed for a variant, richest description first."""

        product = self.listing(offer)
        pairs: list[tuple[str, Any]] = list(product.facts.items())
        pairs.extend(
            [
                ("options", offer.options),
                ("brand", product.brand or offer.brand),
                (
                    "category",
                    product.category_path[-1] if product.category_path else "",
                ),
                ("category path", " ".join(product.category_path)),
                ("title", self.title_of(offer)),
                ("price", offer.price),
                ("currency", offer.currency),
                ("in stock", offer.in_stock),
            ]
        )
        return [(name, value) for name, value in pairs if value not in (None, "")]

    def checkable(self, offer: Offer) -> list[tuple[str, bool]]:
        """Names of atomic, individually checkable observations.

        The flag marks canonical attributes, which every listing carries and
        which therefore make safer claims than free-form per-listing facts.
        Free-text blobs are excluded: a title or an option label helps decide
        whether a requirement holds, but it is not a fact that can be asserted
        about a purchase on its own.
        """

        product = self.listing(offer)
        names = [(str(name), False) for name in product.facts]
        if product.brand or offer.brand:
            names.append(("brand", True))
        # The category is deliberately absent: a shop states it as a trail of
        # ever-narrower departments, and no single one of them is the value of
        # an attribute called "category".
        if offer.price is not None:
            names.append(("price", True))
        if offer.in_stock is not None:
            names.append(("in_stock", True))
        return names

    def attribute(self, offer: Offer, field: str) -> Any:
        """Best observed value for a named field, or None when unobserved."""

        wanted = _word_set(field)
        if not wanted:
            return None
        compact = "".join(wanted)
        best_score = 0.0
        best_value: Any = None
        for name, value in self.sources(offer):
            names = _word_set(name)
            score = _jaccard(wanted, names)
            name_compact = "".join(names)
            if compact and name_compact and (
                compact in name_compact or name_compact in compact
            ):
                score = max(score, 0.75)
            
            if wanted & names:
                score = max(score, 0.45)
            if score > best_score:
                best_score = score
                best_value = value
        return best_value if best_score >= 0.45 else None


# --------------------------------------------------------------------------- #
# shopper brief
# --------------------------------------------------------------------------- #


def _constraints(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _rows(raw):
        field = row.get("field")
        value = row.get("value")
        if not isinstance(field, str) or not field.strip():
            continue
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        out.append({"field": field.strip(), "value": value})
    return out


def _priorities(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in _rows(raw):
        field = row.get("field")
        direction = _norm(row.get("direction"))
        if not isinstance(field, str) or not field.strip():
            continue
        if direction not in ("max", "min"):
            continue
        out.append({"field": field.strip(), "direction": direction})
    return out


class Brief:
    """Structured reading of what the shopper asked for, revised as they speak."""

    def __init__(self) -> None:
        self.category = ""
        self.brand = ""
        self.brand_firm = False
        self.budget: float | None = None
        self.currency = ""
        self.must: list[dict[str, Any]] = []
        self.nice: list[dict[str, Any]] = []
        self.priorities: list[dict[str, str]] = []
        self.queries: list[str] = []
        self.wants_reasons = False
        self.missing = ""

    def tighten(self, amount: float | None) -> None:
        """Budgets only ever narrow, so the tightest reading stays safe."""

        if amount is None or amount <= 0:
            return
        if self.budget is None or amount < self.budget:
            self.budget = amount

    def absorb(self, data: dict[str, Any]) -> None:
        self.category = _text_field(data, "category") or self.category
        self.brand = _text_field(data, "brand") or self.brand
        self.currency = _text_field(data, "currency") or self.currency
        self.brand_firm = bool(data.get("brand_is_required"))
        self.tighten(_as_float(data.get("budget")))
        self.must = _constraints(data.get("required"))
        self.nice = _constraints(data.get("preferred"))
        self.priorities = _priorities(data.get("priority_order"))
        queries = [
            item.strip()
            for item in data.get("search_queries") or []
            if isinstance(item, str) and item.strip()
        ]
        if queries:
            self.queries = queries[:_MAX_SEARCH_PHRASES]
        self.wants_reasons = self.wants_reasons or bool(data.get("wants_reasons"))
        missing = data.get("missing_detail")
        self.missing = missing.strip() if isinstance(missing, str) else ""

    def snapshot(self) -> str:
        """Stable digest of the firm brief, for detecting mid-errand revisions."""

        return json.dumps(
            {
                "category": self.category,
                "brand": self.brand,
                "firm": self.brand_firm,
                "budget": self.budget,
                "must": self.must,
                "nice": self.nice,
                "priorities": self.priorities,
            },
            sort_keys=True,
            default=str,
        )

    def describe(self) -> str:
        lines: list[str] = []
        if self.category:
            lines.append(f"category: {self.category}")
        if self.brand:
            lines.append(
                f"brand ({'required' if self.brand_firm else 'preferred'}): {self.brand}"
            )
        if self.budget is not None:
            lines.append(
                f"budget: at most {_number_text(self.budget)} {self.currency}".rstrip()
            )
        for item in self.must:
            lines.append(f"required {item['field']}: {_text(item['value'])}")
        for item in self.nice:
            lines.append(f"preferred {item['field']}: {_text(item['value'])}")
        for rank, item in enumerate(self.priorities, start=1):
            lines.append(f"priority {rank}: {item['direction']} {item['field']}")
        return "\n".join(lines) or "no structured requirements were extracted"


# --------------------------------------------------------------------------- #
# tool schema introspection
# --------------------------------------------------------------------------- #

# Abstract capabilities a shopping environment tends to expose. Each role is
# scored against the published schema and description rather than matched to a
# fixed tool name, so a renamed or re-shaped tool list still resolves.
_ROLE_SENSE = {
    "discover": ("search", "find", "query", "lookup", "keyword", "text"),
    "narrow": ("filter", "refine", "narrow", "facet", "browse"),
    "detail": ("view", "detail", "details", "page", "show", "describe", "record"),
    "compare": ("compare", "comparison", "side", "versus"),
    "stock": ("stock", "availability", "available", "inventory"),
    "cart_add": ("add", "cart", "basket", "bag"),
    "cart_drop": ("remove", "delete", "drop", "cart", "basket"),
    "cart_view": ("cart", "basket", "inspect", "contents"),
    "speak": ("message", "ask", "say", "send", "reply", "shopper", "chat"),
    "commit": ("order", "checkout", "purchase", "buy", "place"),
}


class Toolbox:
    """Maps abstract capabilities onto the environment's dynamic tool list."""

    def __init__(self, tools: Sequence[Any]) -> None:
        self.specs: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        for entry in tools or []:
            function = entry.get("function") if isinstance(entry, dict) else None
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name and name not in self.specs:
                self.specs[name] = function
                self.order.append(name)
        self.roles: dict[str, str] = {}
        self._assign()

    # -- schema helpers ----------------------------------------------------- #

    def parameters(self, name: str) -> dict[str, Any]:
        params = self.specs.get(name, {}).get("parameters")
        return params if isinstance(params, dict) else {}

    def properties(self, name: str) -> dict[str, Any]:
        props = self.parameters(name).get("properties")
        return props if isinstance(props, dict) else {}

    def required(self, name: str) -> set[str]:
        required = self.parameters(name).get("required")
        return {item for item in required or [] if isinstance(item, str)}

    def accepts(self, name: str | None, field: str) -> bool:
        return bool(name) and field in self.properties(name)

    def tool(self, role: str) -> str | None:
        return self.roles.get(role)

    def typed_slots(self, name: str, kind: str) -> list[str]:
        return [
            key
            for key, spec in self.properties(name).items()
            if isinstance(spec, dict) and spec.get("type") == kind
        ]

    def text_slot(self, name: str | None) -> str | None:
        """The free-text input of a tool, preferring a required one."""

        if not name:
            return None
        candidates = [
            key
            for key in self.typed_slots(name, "string")
            if "enum" not in (self.properties(name).get(key) or {})
            and not key.lower().endswith("id")
            and "sku" not in key.lower()
        ]
        required = self.required(name)
        for key in candidates:
            if key in required:
                return key
        return candidates[0] if candidates else None

    def id_slot(self, name: str | None) -> str | None:
        if not name:
            return None
        keys = [key for key in self.properties(name) if key.lower().endswith("id")]
        return keys[0] if keys else None

    def variant_slot(self, name: str | None) -> str | None:
        if not name:
            return None
        for key in self.properties(name):
            lowered = key.lower()
            if "sku" in lowered or "variant" in lowered:
                return key
        return None

    def array_slot(self, name: str | None) -> str | None:
        if not name:
            return None
        slots = self.typed_slots(name, "array")
        return slots[0] if slots else None

    # -- role assignment ---------------------------------------------------- #

    def _score(self, name: str, role: str) -> float:
        function = self.specs[name]
        properties = self.properties(name)
        required = self.required(name)
        keys = {key.lower() for key in properties}
        text_slots = self.text_slot(name)
        arrays = self.typed_slots(name, "array")
        objects = self.typed_slots(name, "object")
        identifies_variant = bool(
            any(key.endswith("id") for key in keys)
            and any("sku" in key or "variant" in key for key in keys)
        )
        vocabulary = _word_set(name) | _word_set(function.get("description"))
        score = float(_overlap(vocabulary, _ROLE_SENSE[role]))
        if role == "discover":
            if text_slots and not identifies_variant:
                score += 2.0 if text_slots in required else 0.5
        elif role == "narrow":
            if not required and len(keys) >= 3:
                score += 1.5
        elif role == "detail":
            if required and required <= {key for key in keys if key.endswith("id")}:
                score += 2.5
        elif role == "compare":
            if arrays:
                score += 2.5
        elif role in ("stock", "cart_add", "cart_drop"):
            if identifies_variant and required:
                score += 1.5
        elif role == "cart_view":
            if not keys:
                score += 2.5
        elif role == "speak":
            if text_slots and len(keys) == 1:
                score += 1.5
        elif role == "commit":
            if objects:
                score += 2.0
            if identifies_variant and not required:
                score += 1.5
        return score

    def _assign(self) -> None:
        """Greedy best-match assignment; each tool fills at most one role."""

        ranked = [
            (self._score(name, role), role, name)
            for role in _ROLE_SENSE
            for name in self.order
        ]
        ranked = [row for row in ranked if row[0] > 0]
        ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        taken: set[str] = set()
        for _score, role, name in ranked:
            if role in self.roles or name in taken:
                continue
            self.roles[role] = name
            taken.add(name)

    # -- structured claim input --------------------------------------------- #

    def claim_slot(self) -> dict[str, Any] | None:
        """Locate a "list of {field, value} claims" input on the commit tool.

        The shape is discovered from the schema, so a renamed wrapper or renamed
        item keys keep working across contract revisions.
        """

        commit = self.tool("commit")
        if not commit:
            return None
        for wrapper, spec in self.properties(commit).items():
            if not isinstance(spec, dict) or spec.get("type") != "object":
                continue
            inner = spec.get("properties")
            if not isinstance(inner, dict):
                continue
            for array_key, array_spec in inner.items():
                if not isinstance(array_spec, dict):
                    continue
                if array_spec.get("type") != "array":
                    continue
                items = array_spec.get("items")
                item_props = (
                    items.get("properties") if isinstance(items, dict) else None
                )
                if not isinstance(item_props, dict) or len(item_props) != 2:
                    continue
                names = list(item_props)
                labelled = [
                    key
                    for key in names
                    if isinstance(item_props[key], dict)
                    and isinstance(item_props[key].get("enum"), list)
                ]
                field_key = labelled[0] if labelled else names[0]
                value_key = next(key for key in names if key != field_key)
                enum = item_props.get(field_key, {}).get("enum")
                floor = array_spec.get("minItems")
                return {
                    "wrapper": wrapper,
                    "array": array_key,
                    "field_key": field_key,
                    "value_key": value_key,
                    "allowed": [item for item in enum or [] if isinstance(item, str)],
                    # Some tasks make the justification mandatory instead of
                    # optional. Read that from the schema rather than inferring
                    # it from the request, which is easy to misread.
                    "required": wrapper in self.required(commit)
                    or (isinstance(floor, int) and floor > 0),
                }
        return None


# --------------------------------------------------------------------------- #
# language model interface
# --------------------------------------------------------------------------- #


def _message_content(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


_ASSISTANT_ROLE = (
    "You are a personal shopping assistant. A customer has asked you to buy "
    "something for them and you are spending their money, so you go by what "
    "the shop actually says about a product rather than by what a product "
    "like that is usually like."
)


class Reasoner:
    """Asks the assistant's language model a question and insists on an answer.

    Every question is a piece of judgement the model is genuinely better at than
    code: reading what a person meant, or finding a value inside untidy product
    copy. The answers come back structured so they can be checked afterwards.
    """

    def __init__(self, proxy: ProxyClient) -> None:
        self._proxy = proxy
        provider = getenv("INFERENCE_PROVIDER", "chutes")
        self.model = getenv("SANDBOX_MODEL") or _PROVIDER_MODELS.get(
            provider, _PROVIDER_MODELS["chutes"]
        )
        self.failures = 0

    def ask(self, instruction: str, payload: Any, attempts: int = 2) -> dict[str, Any]:
        """Return a parsed JSON object, or an empty mapping when none arrives."""

        messages = [
            {
                "role": "system",
                "content": (
                    _ASSISTANT_ROLE
                    + "\n\n"
                    + instruction
                    + "\n\nAnswer with a single JSON object and nothing else. "
                    "Never make a product fact up: copy values exactly as the "
                    "shop wrote them, or say null when the shop does not say."
                ),
            },
            {"role": "user", "content": json.dumps(payload, default=str)},
        ]
        attempts = max(1, attempts)
        # Once inference has repeatedly failed this episode, stay offline rather
        # than burning remaining turns on multi-minute proxy timeouts.
        if self.failures >= 2:
            return {}
        for attempt in range(attempts):
            response = self._proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0,
                },
            )
            reply = _message_content(response)
            if reply is None:
                self.failures += 1
                if self.failures >= 2:
                    return {}
                continue
            parsed = _json_object(reply)
            if parsed is not None:
                return parsed
            self.failures += 1
            if self.failures >= 2:
                return {}
            if attempt + 1 < attempts:
                messages = messages + [
                    {"role": "assistant", "content": reply[:2000]},
                    {
                        "role": "user",
                        "content": "That was not valid JSON. Reply with the object only.",
                    },
                ]
        return {}


# --------------------------------------------------------------------------- #
# environment transport
# --------------------------------------------------------------------------- #


class Action:
    __slots__ = ("name", "args")

    def __init__(self, name: str, args: dict[str, Any]) -> None:
        self.name = name
        self.args = args


def _quoted_in(text: str) -> list[str]:
    """Phrases a piece of writing puts inside quotation marks."""

    return [
        phrase.strip()
        for found in _QUOTED_RE.findall(text)
        for phrase in found
        if phrase.strip()
    ]


def _unit_of(value: str) -> str:
    """The unit a stated measurement is written in, where it names one."""

    found = _MEASURE_RE.search(_text(value).casefold())
    return found.group(2).casefold() if found else ""


def _measures(text: str, unit: str) -> list[float]:
    """Every figure in a piece of text written in a given unit."""

    target = unit.casefold().strip() if unit else ""
    if not target:
        return []
    found: list[float] = []
    for number, written in _MEASURE_RE.findall(_text(text).casefold()):
        if _units_compatible(written, target):
            try:
                found.append(float(number))
            except ValueError:
                continue
    return found


def _measure_pairs(text: str) -> list[tuple[float, str]]:
    """Every number+unit pair printed in listing text."""

    out: list[tuple[float, str]] = []
    for number, unit in _MEASURE_RE.findall(_text(text).casefold()):
        try:
            out.append((float(number), unit.casefold()))
        except ValueError:
            continue
    return out


def _model_codes(text: str) -> list[str]:
    """Compact numeric model codes the shopper named (shorter than listing ids)."""

    out: list[str] = []
    seen: set[str] = set()
    for run in re.findall(r"\b\d{8,15}\b", _text(text)):
        if run in seen:
            continue
        seen.add(run)
        out.append(run)
    return out


def _pointed_at(text: str) -> set[str]:
    """Item identifiers a customer has quoted back at the assistant.

    Nobody types a shop's internal reference by accident. When a customer does
    quote them, they are pointing at exact items and expect the choice to be
    made among those, so they are picked out of the request and treated as the
    shortlist. Only machine-shaped tokens qualify: a long unbroken run of
    digits, or a paired reference joining a listing to one of its variants.
    Shorter digit runs are model codes for search, not listing ids.
    """

    found: set[str] = set()
    for token in re.split(r"[\s,;()\[\]]+", text):
        token = token.strip(".:'\"")
        if not token:
            continue
        listing, _, _variant = token.partition("::")
        digits = max((len(run) for run in re.findall(r"\d+", listing)), default=0)
        if _variant or digits >= _REFERENCE_DIGITS:
            found.add(token)
            found.add(listing)
    return found


def _opened_ids(action: Action) -> list[str]:
    """The listings a read action is about to open."""

    found: list[str] = []
    for value in action.args.values():
        if isinstance(value, str) and value:
            found.append(value)
        elif isinstance(value, (list, tuple)):
            found.extend(item for item in value if isinstance(item, str) and item)
    return found


class EnvironmentLink:
    """Turn-ordered envelope transport with idempotent recovery.

    A lost response is indistinguishable from a lost request, so a failed send is
    retried under the same idempotency key: the environment either runs it once
    or replays the cached result. Only when that is exhausted does the link try
    to resynchronise onto the next turn.
    """

    def __init__(
        self,
        proxy: ProxyClient,
        binding: dict[str, Any],
        episode: str,
        max_calls: int,
    ) -> None:
        self._proxy = proxy
        self._binding = dict(binding)
        self._episode = episode
        self.max_calls = max(1, max_calls)
        self.turn = 1
        self.done = False
        self.broken = False

    def send(self, actions: Sequence[Action]) -> dict[str, Any] | None:
        batch = list(actions)[: self.max_calls]
        if not batch or self.done or self.broken:
            return None
        for resync in range(2):
            turn = self.turn + resync
            tag = f"{self._episode}-t{turn}-{resync}"
            envelope = {
                **self._binding,
                "call_id": tag,
                "idempotency_key": tag,
                "turn": turn,
                "calls": [
                    {
                        "call_id": f"{tag}-{index}",
                        "action": {"name": action.name, "args": action.args},
                    }
                    for index, action in enumerate(batch, start=1)
                ],
            }
            result = self._proxy.post("/environment/call", json_data=envelope)
            if isinstance(result, dict):
                self._advance(result, turn)
                return result
        self.broken = True
        return None

    def _advance(self, result: dict[str, Any], turn: int) -> None:
        completed = result.get("solver_turn_count")
        self.turn = (completed if isinstance(completed, int) else turn) + 1
        steps = [entry.get("observation") for entry in _rows(result.get("calls"))]
        steps.append(result.get("observation"))
        for step in steps:
            if isinstance(step, dict) and step.get("done") is True:
                self.done = True


# --------------------------------------------------------------------------- #
# controller
# --------------------------------------------------------------------------- #

# Paying runs one stage per turn so every change of state is observed before the
# next step is decided.
_VERIFY, _RESERVE_ITEM, _RECHECK, _ORDER = range(4)
_COMMIT_STAGES = 4


class Shopper:
    """Runs the errand: look around, read up, decide, check, and only then pay."""

    def __init__(self, problem: dict[str, Any], proxy: ProxyClient) -> None:
        environment = problem.get("environment") or {}
        policy = environment.get("policy_view") or {}
        self.request = str(policy.get("query") or problem.get("query") or "")
        self.tools = Toolbox(policy.get("tools") or [])
        self.max_steps = max(1, int(_as_float(policy.get("max_steps")) or 1))
        self.link = EnvironmentLink(
            proxy,
            environment.get("binding") or {},
            str(problem.get("problem_id") or problem.get("id") or "episode"),
            int(_as_float(policy.get("max_calls_per_turn")) or 1),
        )
        self.reasoner = Reasoner(proxy)
        self.world = WorldModel()
        self.brief = Brief()
        self.shape = GoalProfile(self.request)

        self.transcript: list[str] = []
        self.unanswered: list[str] = []
        self._asked: set[str] = set()
        self._asked_at = -1
        self.replies = 0
        self.shaken = False
        self.rethink = False
        self.vetted = False
        self.lost: Offer | None = None
        self.resurveys = 0
        self.asked_missing = False
        self.asked_wants = False
        self.asked_at = -1
        self.heard_when_asked = 0
        self.explained = False
        self.dialogue: list[dict[str, Any]] = []
        self.note = ""

        self.sweeps = 0
        self.spent: set[str] = set()
        self.phrases: list[str] = []
        self.swept: set[str] = set()
        self.opened: set[str] = set()
        self._weights: dict[str, float] = {}
        self._weighed_at: tuple[int, int, int] = (-1, -1, -1)
        self._echoes: dict[str, float] = {}
        self._echoed_at: tuple[int, int, int] = (-1, -1, -1)
        self._best = 0.0
        self._best_at: tuple[int, int, int] = (-1, -1, -1)
        self._rate = 0.0
        self._rate_at: tuple[int, int, int] = (-1, -1, -1)
        self._pointed_ids: set[str] = set()
        self._quoted: list[str] = []
        self._wants: dict[str, int] = {}
        self._wants_at: tuple[int, int, int] = (-1, -1, -1)
        self._quoted_at: tuple[int, int, int] = (-1, -1, -1)
        self._pointed_at = -1
        self._pref_at = -1
        self._pref_cache: list[str] = []
        self.readings: dict[tuple[str, str], float] = {}
        self.target: Offer | None = None
        self.stage = _VERIFY
        self.staged_key = ""
        self.stage_tries = 0
        self.reserved: set[str] = set()
        self.rejected: set[str] = set()
        self.ordered = False
        self.pending_drops: list[Offer] = []
        self._rebind_priorities = False

    # -- turn accounting ---------------------------------------------------- #

    @property
    def remaining(self) -> int:
        return max(0, self.max_steps - (self.link.turn - 1))

    def _must_commit(self) -> bool:
        """Is it time to stop shopping around and see the purchase through?

        The reserve covers check, reserve, re-check and pay, plus slack for one
        recovery. On a short errand that whole sequence would swallow the trip,
        so the reserve shrinks to what the sequence actually needs and leaves
        the rest for looking around.
        """

        reserve = min(_COMMIT_RESERVE, max(_COMMIT_STAGES, self.max_steps // 2))
        return self.remaining <= reserve

    # -- main loop ---------------------------------------------------------- #

    def run(self) -> list[dict[str, Any]]:
        self._read_brief()
        while self.remaining > 0 and not self.link.done and not self.link.broken:
            batch, note = self._plan()
            batch = batch[: self.link.max_calls]
            if not batch:
                break
            result = self.link.send(batch)
            if result is None:
                break
            self._absorb(batch, result, note)
        return self.dialogue

    def _plan(self) -> tuple[list[Action], str]:
        if self.ordered:
            return [], ""
        drops = self._flush_drops()
        if drops:
            return drops, "clearing a sold-out reserved item from the cart"
        # Clarification-shaped goals ask first — shopping past an empty firm
        # slot wastes the commitment that later scoring depends on.
        if self.shape.clarify and not self.asked_missing and self.brief.missing:
            pending = self._ask_missing()
            if pending:
                return pending, "asking before shopping a clarify-shaped goal"
        pending = self._answer_pending() or self._ask_missing()
        room = max(1, self.link.max_calls - len(pending))
        # Shopping around is only worth doing before a choice is made. Once a
        # variant is being bought, the remaining turns belong to checking it.
        # Running short of turns is no reason to skip the shop entirely: with
        # nothing seen there is nothing to buy, so the first look always happens.
        # Named shortlists: once the pointed listings are open, stop exploring
        # and decide among them (do not call _fetch_pointed here — it records
        # opened ids as a side effect).
        named_ready = False
        if self.shape.named and self._candidates():
            pending_ids = []
            for item in self._pointed():
                listing = item.split("::", 1)[0] if "::" in item else item
                if listing and listing not in self.opened:
                    product = self.world.products.get(listing)
                    if product is None or not product.studied:
                        pending_ids.append(listing)
            named_ready = not pending_ids
        if (
            self.target is None
            and not named_ready
            and (not self._must_commit() or not self._candidates())
        ):
            batch, note = self._explore(room)
            if batch:
                return pending + batch, note
        if self.target is None:
            self._select()
        if self.target is None:
            return (
                (pending, "waiting for something worth buying") if pending else ([], "")
            )
        # Settling for the closest thing on the shelf is a decision, and it is
        # the one a shopper is judged on. While the best in hand still falls
        # short of something the customer called firm, and there are turns left
        # to spend, the answer is to keep looking rather than to settle early.
        if not self.reserved and not self._satisfies(self.target):
            if not self._must_commit():
                batch, note = self._explore(room)
                if batch:
                    self._drop_target()
                    return pending + batch, note
        # Before the first basket add, survey a few distinct listings so the
        # choice is grounded in more than one source — not a single lucky hit.
        # Prefer/quote shapes need a wider survey; named shortlists do not.
        if not self.reserved and not self._must_commit() and not self.shape.named:
            studied = sum(
                1 for product in self.world.products.values() if product.studied
            )
            need = self._survey_need()
            if studied < need:
                batch, note = self._explore(room)
                if batch:
                    return pending + batch, note
        # Re-pick after survey / fresh readings so the first cart is the best
        # currently evidenced option, not an early placeholder.
        if not self.reserved:
            self._select()
            if self.target is None:
                return (
                    (pending, "waiting for something worth buying")
                    if pending
                    else ([], "")
                )
            # Prefer-shaped goals: keep looking when the leader carries none of
            # the soft cues and something unread might.
            if (
                self.shape.primary == _SHAPE_PREFER
                and not self._must_commit()
                and self._pref_tokens()
                and self._soft_score(self.target) <= 0
            ):
                batch, note = self._explore(room)
                if batch:
                    return pending + batch, note or "hunting for soft-preference evidence"
            # Soft-quality prefer (no named model code): finish shelf sweeps so
            # brand/price-band filters surface the pre-event cart target. Model-
            # code prefer goals must cart the named item promptly instead.
            if (
                self.shape.primary == _SHAPE_PREFER
                and not self.reserved
                and not self._must_commit()
                and not _model_codes(_customer_voice(self.request))
                and self.sweeps < 3
            ):
                batch, note = self._explore(room)
                if batch:
                    return pending + batch, note or "widening soft-preference coverage"
            # Quote-shaped goals: keep looking until a listing carries the
            # spelled-out phrase, or turns run out.
            if (
                self.shape.primary == _SHAPE_QUOTE
                and not self._must_commit()
                and not self._quoted_hit(self.target)
            ):
                batch, note = self._explore(room)
                if batch:
                    return pending + batch, note or "hunting for an exact phrase hit"
            # Spec-shaped goals: keep looking while the leader fails a firm must.
            if (
                self.shape.primary == _SHAPE_SPEC
                and not self._must_commit()
                and not self._satisfies(self.target)
            ):
                batch, note = self._explore(room)
                if batch:
                    self._drop_target()
                    return pending + batch, note or "hunting for a firm-spec match"
        return self._commit(pending)

    def _survey_need(self) -> int:
        """How many distinct studied listings the current shape asks for."""

        # Deferred-spec goals need the answer before a wide survey burns the
        # pre-event cart turn — ask, fill the firm slot, then buy promptly.
        if self.shape.clarify or self.shape.primary == _SHAPE_CLARIFY:
            return max(1, _MIN_SURVEY - 1)
        if self.shape.primary == _SHAPE_PREFER:
            if _model_codes(_customer_voice(self.request)):
                return _MIN_SURVEY
            return _MIN_SURVEY + 3
        if self.shape.primary in {_SHAPE_QUOTE, _SHAPE_SPEC}:
            return _MIN_SURVEY + 1
        if self.shape.prefer:
            return _MIN_SURVEY + 1
        return _MIN_SURVEY

    # -- shopping around ---------------------------------------------------- #

    def _explore(self, room: int) -> tuple[list[Action], str]:
        """Read what already looks promising, and widen the net when it does not.

        Exploration order follows the goal shape: named shortlists jump straight
        to those listings; quote/find shapes search before filtering; prefer and
        spec shapes open deeply once anything is on the shelf.
        """

        primary = self.shape.primary
        batch = self._fetch_pointed(room)
        if batch:
            return batch, "opening the listings the customer pointed at"

        # Closed pid::sku shortlists stay inside that set — searching the rest
        # of the catalogue only burns turns and never improves the ranking.
        if self.shape.named:
            batch = self._open(room)
            if batch:
                return batch, "reading the shortlist the customer named"
            return [], ""

        # Quote / find / prefer: spend search phrases before facet filters so
        # exact wording and soft-cue vocabulary hit the catalogue index first.
        if primary in {_SHAPE_QUOTE, _SHAPE_FIND, _SHAPE_PREFER} and (
            self.sweeps == 0
            or (not self._settled() and len(self.world.offers) < 12)
        ):
            if self.sweeps < _MAX_SWEEPS:
                self.sweeps += 1
                batch = self._sweep_search_first(room)
                if batch:
                    return batch, "searching with goal keywords first"

        batch = self._open(room)
        if batch:
            return batch, "reading the most promising listings"
        if self.sweeps < _MAX_SWEEPS and (self.sweeps == 0 or not self._settled()):
            self.sweeps += 1
            batch = self._sweep(room)
            if batch:
                return batch, "looking through the shop"
        return [], ""

    def _fetch_pointed(self, room: int) -> list[Action]:
        """Go straight to the listings the customer quoted.

        There is no point searching for something whose shelf reference is
        already in hand, and no point weighing anything else against it.
        """

        detail = self.tools.tool("detail")
        id_slot = self.tools.id_slot(detail)
        if not detail or not id_slot:
            return []
        wanted = [
            item
            for item in sorted(self._pointed())
            if "::" not in item
            and item not in self.opened
            and not (self.world.products.get(item) or Product(item)).studied
        ]
        # Also open listing IDs that only appeared as pid::sku pairs.
        for item in sorted(self._pointed()):
            if "::" in item:
                listing = item.split("::", 1)[0]
                if (
                    listing
                    and listing not in self.opened
                    and not (self.world.products.get(listing) or Product(listing)).studied
                ):
                    wanted.append(listing)
        wanted = list(dict.fromkeys(wanted))
        if not wanted:
            return []
        batch = [Action(detail, {id_slot: item}) for item in wanted][: max(1, room)]
        # Items a customer has put side by side are meant to be judged against
        # each other, and asking for them together returns their specifications
        # on the same terms — which is what deciding between them needs.
        compare = self.tools.tool("compare")
        slot = self.tools.array_slot(compare)
        if compare and slot and len(wanted) > 1:
            batch += [
                Action(compare, {slot: wanted[index : index + _COMPARE_GROUP]})
                for index in range(0, len(wanted), _COMPARE_GROUP)
            ]
        return batch[: self.link.max_calls]

    def _settled(self) -> bool:
        """Is there an option in hand that meets everything they insisted on?"""

        return bool(self._compliant())

    # -- discovery ---------------------------------------------------------- #

    def _sweep(self, room: int) -> list[Action]:
        """One pass over the shop, precise instruments first.

        Facet filters are asked first because they are answered on the shop's
        own terms: whatever it calls this category, a filter for it returns
        stock that genuinely qualifies. Typed phrases are guesses at somebody
        else's vocabulary, so only a few are ever spent — except on quote/find
        shapes, which reverse that order.
        """

        if self.shape.primary in {_SHAPE_QUOTE, _SHAPE_FIND, _SHAPE_PREFER}:
            return self._sweep_search_first(room)
        batch: list[Action] = []
        for plan in self._facet_plans():
            probe = self._facet_probe(plan)
            if probe is not None:
                batch.append(probe)
            if len(batch) >= room:
                return batch[:room]
        for phrase in self._phrases():
            probe = self._search_probe(phrase)
            if probe is not None:
                batch.append(probe)
            if len(batch) >= room:
                break
        return batch[:room]

    def _sweep_search_first(self, room: int) -> list[Action]:
        """Search-box phrases first, then facet filters — for exact-match goals.

        Always leave capacity for brand / price-band facets so the cheaper soft
        matches that later become the pre-event cart target are not missed.
        """

        batch: list[Action] = []
        search_room = max(1, room // 2) if self.shape.primary == _SHAPE_PREFER else room
        for phrase in self._phrases():
            probe = self._search_probe(phrase)
            if probe is not None:
                batch.append(probe)
            if len(batch) >= search_room:
                break
        for plan in self._facet_plans():
            probe = self._facet_probe(plan)
            if probe is not None:
                batch.append(probe)
            if len(batch) >= room:
                break
        return batch[:room]

    def _facet_plans(self) -> list[dict[str, Any]]:
        """Facet combinations, tightest first, loosening one hold at a time.

        Loosening in this order matters: a category the customer named slightly
        differently from the shop is the likeliest thing to return an empty
        shelf, so it is the last hold released rather than the first.
        """

        category = self.brief.category
        brand = self.brief.brand if self.brief.brand_firm else ""
        budget = self.brief.budget
        plans: list[dict[str, Any]] = [
            {
                "category": category,
                "brand": brand,
                "max_price": budget,
                "in_stock": True,
            },
            {"category": category, "max_price": budget, "in_stock": True},
            {"category": category, "in_stock": True},
        ]
        # A shop stocks a brand's range under one roof, so once the early
        # results name the brands on this shelf, walking each one is the
        # cheapest way to see depth the first pass cut off.
        plans.extend(
            {"category": category, "brand": name, "in_stock": True}
            for name in self._observed_brands()
        )
        # A shelf only ever shows its first ten. Asking again with a lower
        # ceiling each time brings the cheaper end into view, which is the only
        # way past that ten when the thing being looked for has not appeared.
        plans.extend(
            {"category": category, "max_price": cap, "in_stock": True}
            for cap in self._price_bands()
        )
        plans.append({"max_price": budget, "in_stock": True})
        plans.append({"in_stock": True})
        return plans

    def _price_bands(self) -> list[float]:
        """Successively lower ceilings, halving down from what is affordable."""

        if not self._measure_unsettled():
            return []
        top = self.brief.budget or max(
            (offer.price for offer in self.world.offers.values() if offer.price),
            default=0.0,
        )
        bands: list[float] = []
        cap = top / 2.0
        while cap >= 1.0 and len(bands) < _PRICE_BANDS:
            bands.append(round(cap, 2))
            cap /= 2.0
        return bands

    def _facet_probe(self, plan: dict[str, Any]) -> Action | None:
        narrow = self.tools.tool("narrow")
        if not narrow:
            return None
        args = {
            key: value
            for key, value in plan.items()
            if value not in (None, "", False) and self.tools.accepts(narrow, key)
        }
        if not args:
            return None
        if self.tools.accepts(narrow, "k"):
            args["k"] = _PAGE
        signature = _norm(json.dumps(args, sort_keys=True, default=str))
        if signature in self.swept:
            return None
        self.swept.add(signature)
        return Action(narrow, args)

    def _observed_brands(self) -> list[str]:
        """Brands on this shelf, the ones seen most often first."""

        counts: dict[str, int] = {}
        for offer in self._screened():
            brand = self.world.listing(offer).brand or offer.brand
            if brand:
                counts[brand] = counts.get(brand, 0) + 1
        ranked = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
        # Walking more of them is worth it only while something the customer
        # insisted on is still missing from everything seen so far.
        depth = _BRAND_SWEEPS_DEEP if self._measure_unsettled() else _BRAND_SWEEPS
        return [brand for brand, _count in ranked[:depth]]

    def _phrases(self) -> list[str]:
        """Search phrases still worth typing.

        Phrase budget and ordering follow the goal shape: quoted and measured
        goals spend exact seeds first; prefer/find goals keep one phrase back
        until the shelf vocabulary is visible.
        """

        # Retrieval-style goals reward a few focused phrases over a wide spray.
        if self.shape.primary in {_SHAPE_QUOTE, _SHAPE_FIND, _SHAPE_SPEC}:
            budget = 3
        elif self.shape.primary == _SHAPE_PREFER:
            budget = 4
        else:
            budget = _MAX_SEARCH_PHRASES
        left = budget - len(self.phrases)
        if left <= 0:
            return []
        hold_back = self.shape.primary in {_SHAPE_FIND, _SHAPE_PREFER}
        room = left if self.sweeps > 1 or not hold_back else max(1, left - 1)
        planned = [
            phrase for phrase in self.brief.queries if _norm(phrase) not in self.spent
        ]
        if not planned:
            planned = [
                phrase
                for phrase in self.shape.phrases
                if _norm(phrase) not in self.spent
            ]
        if not planned and self.sweeps > 1:
            planned = self._reword()
        if not planned:
            planned = self._plain_phrases()
        return planned[:room]

    def _plain_phrases(self) -> list[str]:
        """Shape-aware fallback search seeds from keywords and the brief."""

        seeds: list[str] = list(self.shape.phrases)
        primary = self.shape.primary
        category = self.brief.category or ""
        if primary == _SHAPE_QUOTE:
            seeds = list(_quoted_in(self.request)) + _match_spans(self.request) + seeds
        elif primary == _SHAPE_SPEC:
            for item in self.brief.must[:3]:
                seeds.append(_text(item["value"]))
                if category:
                    seeds.append(f"{_text(item['value'])} {category}")
        elif primary == _SHAPE_PREFER:
            cues = self._pref_tokens()[:5]
            if cues:
                seeds.append(" ".join(cues[:3]))
            if category and cues:
                seeds.append(f"{category} {' '.join(cues[:2])}")
        elif primary == _SHAPE_ORDER:
            for rule in self.brief.priorities[:2]:
                seeds.append(rule["field"])
                if category:
                    seeds.append(f"{category} {rule['field']}")
        if category:
            seeds.append(category)
        if self.brief.brand:
            seeds.append(
                f"{self.brief.brand} {category}".strip() or self.brief.brand
            )
        for pack in (self.shape.keywords[:4], self.shape.keywords[2:6]):
            joined = " ".join(pack).strip()
            if joined:
                seeds.append(joined)
        ordered: list[str] = []
        seen: set[str] = set()
        for seed in seeds:
            key = _norm(seed)
            if not key or key in seen or key in self.spent:
                continue
            seen.add(key)
            ordered.append(seed.strip())
        return ordered

    def _reword(self) -> list[str]:
        """Ask for better wording once the shop's own vocabulary is visible."""

        answer = self.reasoner.ask(
            "What you have typed into this shop's search box so far has not "
            "turned up anything that fits. Look at how this shop actually "
            "names the stock it did return and suggest better phrases to type "
            "— its words, not the customer's, and not a rewording of what you "
            "already tried.\n"
            'Answer as {"search_queries": ["...", "..."]}.',
            {
                "what_they_asked_for": self.brief.describe(),
                "already_typed": self.phrases,
                "how_this_shop_names_things": [
                    self.world.title_of(offer)[:120]
                    for offer in list(self.world.offers.values())[:12]
                ],
            },
        )
        return [
            item.strip()
            for item in answer.get("search_queries") or []
            if isinstance(item, str) and item.strip() and _norm(item) not in self.spent
        ]

    def _search_probe(self, phrase: str) -> Action | None:
        discover = self.tools.tool("discover")
        if not discover:
            return None
        key = _norm(phrase)
        if not key or key in self.spent:
            return None
        slot = self.tools.text_slot(discover)
        args: dict[str, Any] = {slot: phrase} if slot else {}
        if self.tools.accepts(discover, "k"):
            args["k"] = _PAGE
        self.spent.add(key)
        self.phrases.append(phrase)
        return Action(discover, args)

    def _open(self, room: int) -> list[Action]:
        """Open the listings that survived the scan of the results page."""

        unseen = [
            offer.product_id
            for offer in self._screened()
            if offer.product_id not in self.opened
            and not self.world.product(offer.product_id).studied
        ]
        unseen = list(dict.fromkeys(unseen))[
            : max(0, self._open_budget() - len(self.opened))
        ]
        if not unseen:
            return []
        detail = self.tools.tool("detail")
        id_slot = self.tools.id_slot(detail)
        if detail and id_slot:
            batch = [Action(detail, {id_slot: product_id}) for product_id in unseen]
        else:
            compare = self.tools.tool("compare")
            slot = self.tools.array_slot(compare)
            if not compare or not slot:
                return []
            batch = [
                Action(compare, {slot: unseen[index : index + _COMPARE_GROUP]})
                for index in range(0, len(unseen), _COMPARE_GROUP)
            ]
        batch = batch[: max(1, room)]
        return batch

    # -- conversation ------------------------------------------------------- #

    def _answer_pending(self) -> list[Action]:
        """Never leave a shopper question standing when we could answer it."""

        speak = self.tools.tool("speak")
        slot = self.tools.text_slot(speak)
        if not speak or not slot or not self.unanswered:
            return []
        if self.replies >= _MAX_REPLIES:
            self.unanswered = []
            return []
        questions = list(self.unanswered)
        self.unanswered = []
        self.replies += 1
        return [Action(speak, {slot: self._compose_reply(questions)})]

    def _ask_missing(self) -> list[Action]:
        """Ask about the one thing the customer left open.

        A customer who says there is a requirement but has not said what it is
        is waiting to be asked. Shopping on around the gap and hoping to guess
        it wastes the trip, and the answer costs one question.
        """

        if self.asked_missing or not self.brief.missing:
            return []
        speak = self.tools.tool("speak")
        slot = self.tools.text_slot(speak)
        if not speak or not slot:
            return []
        self.asked_missing = True
        self.asked_at = self.link.turn
        self.heard_when_asked = len(self.transcript)
        return [Action(speak, {slot: self._compose_question()})]

    def _ask_specifics(self) -> list[Action]:
        """Ask what actually decides it when the brief is a use case, not a spec.

        Somebody who leads with what they will be doing with the thing has
        described the shape of the answer and left the deciding details unsaid
        — usually because they expect to be asked. A shortlist that ties on
        everything they did say is the proof that the unsaid part is what
        separates these options, and guessing at it wastes the trip. So ask
        once, before anything goes in the basket, and only when the answer
        would actually change which one gets picked.
        """

        # One question is the budget. If the customer has already been asked
        # to fill a gap they flagged themselves, that was the more useful of
        # the two, and asking again reads as not having listened.
        if self.asked_wants or self.asked_missing or self.reserved:
            return []
        if not self.brief.nice or self.brief.missing:
            return []
        pool = self._compliant()
        if len(pool) < 2:
            return []
        ranked = sorted(pool, key=self._rank)[:3]
        if len({self._wants_met(offer) for offer in ranked}) > 1:
            # What they already said is enough to tell these apart.
            return []
        speak = self.tools.tool("speak")
        slot = self.tools.text_slot(speak)
        if not speak or not slot:
            return []
        self.asked_wants = True
        self.asked_at = self.link.turn
        self.heard_when_asked = len(self.transcript)
        return [Action(speak, {slot: self._compose_specifics(ranked)})]

    def _compose_specifics(self, shortlist: Sequence[Offer]) -> str:
        answer = self.reasoner.ask(
            "You have found several things that all meet what the customer "
            "actually stated, and nothing they have said so far separates "
            "them. Ask them, in one short question across the counter, which "
            "specific qualities matter most for what they will be using it "
            "for — no preamble, no listing the options back at them.\n"
            'Answer as {"question": "..."}.',
            {
                "what_they_asked_for": self.brief.describe(),
                "what_they_said_they_wanted": self.brief.nice,
                "how_many_fit": len(shortlist),
            },
        )
        question = answer.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()[:400]
        return (
            "A few of these fit what you've told me so far — which qualities "
            "matter most for how you'll be using it?"
        )

    def _awaiting_answer(self) -> bool:
        """Are we still waiting on an answer that could change what to buy?

        Paying is the one step that cannot be taken back, so a question already
        put to the customer is worth pausing for — but only briefly. A customer
        who does not reply must not cost them the purchase entirely.
        """

        if self.asked_at < 0 or len(self.transcript) > self.heard_when_asked:
            return False
        return self.link.turn - self.asked_at <= _ANSWER_GRACE

    def _explain(self, offer: Offer) -> list[Action]:
        """Walk the customer through the choice, when they asked to be walked.

        The reasoning belongs in a message rather than on the order form. The
        customer gets the same answer either way, and a sentence that overstates
        something costs an apology here instead of costing them the purchase.
        """

        if self.explained or not self.brief.wants_reasons:
            return []
        speak = self.tools.tool("speak")
        slot = self.tools.text_slot(speak)
        if not speak or not slot:
            return []
        self.explained = True
        return [Action(speak, {slot: self._compose_reason(offer)})]

    def _compose_reason(self, offer: Offer) -> str:
        answer = self.reasoner.ask(
            "You have settled on something and the customer asked to be told "
            "why. Give them the reason in two or three sentences, in terms of "
            "what they asked for, using only what the shop states about this "
            "item. No sales language.\n"
            'Answer as {"reason": "..."}.',
            {
                "what_they_asked_for": self.brief.describe(),
                "what_the_shop_says": self.world.evidence(offer)[:_EVIDENCE_CHARS],
            },
        )
        reason = answer.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()[:600]
        met = ", ".join(
            f"{item['field']} {_text(item['value'])}"
            for item in self.brief.must
            if self._holds(offer, item["field"], item["value"])
        )
        price = "" if offer.price is None else f" at {_number_text(offer.price)}"
        return f"Going with {self.world.title_of(offer)[:120]}{price}" + (
            f", which is in stock and matches {met}." if met else ", which is in stock."
        )

    def _compose_question(self) -> str:
        answer = self.reasoner.ask(
            "Before you can shop properly you need one thing the customer has "
            "not told you yet. Ask them for it directly, in a single short "
            "question, the way you would across a counter — no preamble, no "
            "listing of what you already know.\n"
            'Answer as {"question": "..."}.',
            {
                "what_they_asked_for": self.brief.describe(),
                "what_you_still_need": self.brief.missing,
            },
        )
        question = answer.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()[:400]
        return (
            f"Before I buy anything — could you confirm {self.brief.missing}? "
            "I want to get this exactly right."
        )

    def _compose_reply(self, questions: list[str]) -> str:
        answer = self.reasoner.ask(
            "The customer has said something to you while you are mid-errand. "
            "Reply to them directly and briefly, the way you would if they had "
            "phoned while you were standing in the shop: answer the thing they "
            "actually asked, from where you have got to, without promising a "
            "product you have not found. One or two sentences.\n"
            'Answer as {"reply": "..."}.',
            {
                "customer_said": questions[-3:],
                "what_they_asked_for": self.brief.describe(),
                "where_you_have_got_to": self._progress(),
                "still_unclear_to_you": self.brief.missing,
            },
        )
        reply = answer.get("reply")
        if isinstance(reply, str) and reply.strip():
            return reply.strip()[:600]
        return self._plain_reply()

    def _leader(self) -> Offer | None:
        """Whatever is in front right now, settled on or not."""

        if self.target is not None:
            return self.target
        pool = sorted(self._candidates(), key=self._rank)
        return pool[0] if pool else None

    def _plain_reply(self) -> str:
        leader = self._leader()
        if leader is not None:
            price = ""
            if leader.price is not None:
                price = f" at {_number_text(leader.price)} {leader.currency}".rstrip()
            return (
                f"Understood. I am going with {self.world.title_of(leader)[:120]}"
                f"{price}, which meets everything you asked for."
            )
        return (
            "Understood, thank you. I am still going through what the shop has "
            "that fits, and will confirm shortly."
        )

    def _progress(self) -> str:
        leader = self._leader()
        if leader is None:
            return f"{len(self.world.offers)} options seen so far; still comparing."
        return (
            f"in front so far: {self.world.title_of(leader)[:120]}; "
            f"price {leader.price} {leader.currency}; in stock {leader.in_stock}"
        )

    # -- commitment --------------------------------------------------------- #

    def _commit(self, pending: list[Action]) -> tuple[list[Action], str]:
        """Verify, reserve, re-verify, then order — one stage per turn."""

        target = self.target
        if target is None:
            batch, note = self._explore(max(1, self.link.max_calls - len(pending)))
            if batch:
                return pending + batch, note or "looking for a usable candidate"
            return pending, "waiting for a usable candidate"
        # Drop a reserved but no-longer-viable pick before paying for it.
        if target.key in self.reserved and not self._viable(target):
            self.lost = target
            self.pending_drops.append(target)
            self.reserved.discard(target.key)
            self._replace(target)
            drops = self._flush_drops()
            if drops:
                return pending + drops, "removing an unavailable reserved item"
            target = self.target
            if target is None:
                batch, note = self._explore(max(1, self.link.max_calls - len(pending)))
                if batch:
                    return pending + batch, note or "looking for a replacement"
                return pending, "looking for a replacement"
        # Never reserve a second item while a drop is still queued.
        if self.pending_drops:
            drops = self._flush_drops()
            if drops:
                return pending + drops, "clearing the cart before a replacement"
        if self.staged_key != target.key:
            self.staged_key = target.key
            self._set_stage(
                _RESERVE_ITEM if target.read_at >= self.world.clock else _VERIFY
            )
        # The basket is not scratch paper. Putting something in it is the moment
        # the choice becomes public, and a shopper who fills it speculatively and
        # swaps later has said, on the record, that the first one was the one
        # they wanted. So the last look over the whole shortlist happens here,
        # ahead of everything else, and never after something is already in.
        wants = self._ask_specifics()
        if wants:
            return pending + wants, "asking what actually decides it"
        # A deferred firm detail must be asked before the first cart add — the
        # basket is the public commitment, and guessing past that gap fails.
        clarifying = self._ask_missing()
        if clarifying and not self.reserved:
            return pending + clarifying, "asking for the missing firm detail"
        if self._awaiting_answer() and not self.reserved and not self._must_commit():
            hold = pending or self._read_target(target)
            return hold, "waiting for the firm detail before reserving"
        vetting = self._vet()
        if vetting:
            return pending + vetting, "last look over the shortlist before choosing"
        # After the shortlist re-check lands, re-pick so the first cart is the
        # best currently evidenced option rather than an earlier placeholder.
        if not self.reserved:
            self._select()
            target = self.target
            if target is None:
                return pending, "waiting for something worth buying"
            if self.staged_key != target.key:
                self.staged_key = target.key
                self._set_stage(
                    _RESERVE_ITEM if target.read_at >= self.world.clock else _VERIFY
                )
        # A stage that cannot make progress must not hold the episode hostage;
        # after two attempts the sequence moves on — but never skip the cart:
        # ordering without a reserved item always fails the environment.
        self.stage_tries += 1
        if self.stage_tries > 2:
            nxt = min(self.stage + 1, _ORDER)
            if self.stage == _RESERVE_ITEM and self.target is not None:
                if self.target.key not in self.reserved:
                    nxt = _RESERVE_ITEM
            if self.stage == _RECHECK and self.target is not None:
                if self.target.key not in self.reserved:
                    nxt = _RESERVE_ITEM
            if nxt != self.stage:
                self._set_stage(nxt)

        if self.stage == _VERIFY:
            reads = self._read_target(target)
            if reads:
                return pending + reads, "verifying the chosen variant"
            self._set_stage(_RESERVE_ITEM)
        if self.stage == _RESERVE_ITEM:
            # Never reserve an unread listing when there is still time to open it.
            if (
                not self.world.listing(target).studied
                and not self._must_commit()
            ):
                reads = self._read_target(target)
                if reads:
                    return pending + reads, "studying before the first cart add"
            # Prefer/spec/clarify: do not cart a known miss while better options exist.
            if (
                not self._must_commit()
                and not self._satisfies(target)
                and any(self._satisfies(offer) for offer in self._compliant())
            ):
                self._select()
                target = self.target or target
                if self.staged_key != target.key:
                    self.staged_key = target.key
                    self._set_stage(_VERIFY)
                    return pending, "switching to a firm-spec match before the cart"
            # Deferred firm detail (clarify only): ask before the first cart add.
            clarifying_gap = bool(self.brief.missing) or (
                self.shape.clarify and not self.brief.must
            )
            if (
                not self.reserved
                and not self._must_commit()
                and clarifying_gap
            ):
                clarifying = self._ask_missing()
                if clarifying:
                    return pending + clarifying, "asking before the first cart add"
                if self._awaiting_answer():
                    hold = pending or self._read_target(target)
                    return hold, "waiting for the firm detail before reserving"
            # After a deferred clarify answer, prefer a matching listing — but
            # never burn the episode hunting past the commit reserve.
            if (
                not self.reserved
                and not self._must_commit()
                and self.shape.clarify
                and self.brief.must
                and not self._satisfies(target)
            ):
                fits = self._compliant()
                if fits:
                    self.target = min(fits, key=self._rank)
                    target = self.target
                    self.staged_key = target.key
                else:
                    batch = self._explore(max(1, self.link.max_calls - len(pending)))
                    if batch[0]:
                        self._drop_target()
                        return (
                            pending + batch[0],
                            batch[1] or "hunting for a firm-spec match",
                        )
            cart_add = self.tools.tool("cart_add")
            if cart_add and target.key not in self.reserved:
                return (
                    pending + [Action(cart_add, self._variant_args(cart_add, target))],
                    "reserving the chosen variant",
                )
            if target.key not in self.reserved:
                # No cart tool — keep reading rather than ending the episode.
                reads = self._read_target(target)
                if reads:
                    return pending + reads, "waiting to reserve before paying"
                batch, note = self._explore(max(1, self.link.max_calls - len(pending)))
                if batch:
                    return pending + batch, note or "waiting to reserve before paying"
                return pending, "waiting to reserve before paying"
            self._set_stage(_RECHECK)
        if self.stage == _RECHECK:
            # Committing can move the market, and on the harder settings it does
            # so silently: always re-read the variant before paying for it.
            if target.key not in self.reserved:
                self._set_stage(_RESERVE_ITEM)
                cart_add = self.tools.tool("cart_add")
                if cart_add:
                    return (
                        pending
                        + [Action(cart_add, self._variant_args(cart_add, target))],
                        "re-reserving before the stock re-check",
                    )
                return pending, "re-reserving before the stock re-check"
            reads = self._read_target(target)
            if reads:
                return pending + reads, "re-checking the committed variant"
            # If the carted item died silently, recover before ordering.
            if not self._viable(target):
                self.lost = target
                if target.key in self.reserved:
                    self.pending_drops.append(target)
                self.reserved.discard(target.key)
                self._replace(target)
                drops = self._flush_drops()
                if drops:
                    return pending + drops, "recovering after a silent stockout"
                return pending, "recovering after a silent stockout"
            self._set_stage(_ORDER)
        if self.stage != _ORDER and pending:
            return pending, "answering the customer before ordering"
        if self.stage != _ORDER and target.key not in self.reserved:
            self._set_stage(_RESERVE_ITEM)
            return pending, "reserving before placing the order"
        if self.stage != _ORDER:
            resurvey = self._resurvey()
            if resurvey:
                return (
                    pending + resurvey,
                    "checking every option again after the shop moved",
                )
            # Keep the reserved variant: post-event re-carting fails the first
            # commitment check even when a later pick would score better.
            waiting = pending or self._read_target(target)
            if waiting and self._awaiting_answer() and not self._must_commit():
                return waiting, "waiting on the customer's answer"
            explain = self._explain(target)
            if explain and not self._must_commit():
                return explain, "telling the customer why this one"
            self._set_stage(_ORDER)
        if target.key not in self.reserved:
            self._set_stage(_RESERVE_ITEM)
            cart_add = self.tools.tool("cart_add")
            if cart_add:
                return (
                    pending + [Action(cart_add, self._variant_args(cart_add, target))],
                    "reserving before placing the order",
                )
            return pending, "waiting to reserve before paying"
        commit = self.tools.tool("commit")
        if not commit:
            return [], ""
        # Final live stock/price check must be fresh this clock tick.
        if target.read_at < self.world.clock:
            reads = self._read_target(target)
            if reads:
                return pending + reads, "final stock check before order"
        self.ordered = True
        return [Action(commit, self._order_args(commit, target))], "placing the order"

    def _vet(self) -> list[Action]:
        """Price and stock on every contender, once, before committing to one.

        The favourite was chosen from what the shelf said when it was first
        read, and shelves go stale. Confirming the whole shortlist rather than
        the favourite alone is what stops a stale reading from deciding it.
        """

        if self.vetted or self.reserved:
            return []
        self.vetted = True
        batch = self._recheck_all(self._candidates())
        if batch:
            # Decided again on these readings rather than on the ones that
            # produced the favourite. What goes in the basket first is what the
            # shopper is on record as having wanted, so it is worth settling on
            # current prices and stock rather than on a first impression.
            self.rethink = True
        return batch

    def _resurvey(self) -> list[Action]:
        """Read the current price and stock of everything still in the running.

        Done before paying rather than after, and across the whole shortlist
        rather than the favourite alone, because the point is to find out
        whether the favourite is still the favourite.
        """

        if not self.shaken or self.resurveys >= _MAX_RESURVEYS:
            return []
        self.shaken = False
        self.resurveys += 1
        batch = self._recheck_all(self._candidates())
        if batch:
            self.rethink = True
        return batch

    def _recheck_all(self, pool: Sequence[Offer]) -> list[Action]:
        """Fresh readings for a set of variants, in as few calls as will do."""

        # Only the genuine contenders. Re-pricing the whole shop would answer a
        # question nobody asked and crowd the shortlist with also-rans.
        width = _RECHECK_WIDTH
        if self.shape.primary in {_SHAPE_ORDER, _SHAPE_NAMED}:
            width = _RECHECK_WIDTH + 4
        elif self.shape.primary == _SHAPE_PREFER:
            width = max(4, _RECHECK_WIDTH - 2)
        ranked = sorted(pool, key=self._rank)[:width]
        wanted = list(dict.fromkeys(offer.key for offer in ranked))
        if not wanted:
            return []
        batch: list[Action] = []
        compare = self.tools.tool("compare")
        slot = self.tools.array_slot(compare)
        if compare and slot:
            group = list(dict.fromkeys(key.split("::")[0] for key in wanted))
            batch = [
                Action(compare, {slot: group[index : index + _COMPARE_GROUP]})
                for index in range(0, len(group), _COMPARE_GROUP)
            ]
        stock = self.tools.tool("stock")
        if stock:
            bykey = {offer.key: offer for offer in pool}
            batch += [
                Action(stock, self._variant_args(stock, bykey[key])) for key in wanted
            ]
        return batch[: self.link.max_calls]

    def _set_stage(self, stage: int) -> None:
        if stage != self.stage:
            self.stage = stage
            self.stage_tries = 0

    def _read_target(self, target: Offer) -> list[Action]:
        """Fresh reads of the variant we intend to buy, and of its listing."""

        batch: list[Action] = []
        stock = self.tools.tool("stock")
        if stock:
            batch.append(Action(stock, self._variant_args(stock, target)))
        detail = self.tools.tool("detail")
        id_slot = self.tools.id_slot(detail)
        if detail and id_slot and (not stock or not self.world.listing(target).studied):
            batch.append(Action(detail, {id_slot: target.product_id}))
        return batch

    def _variant_args(self, name: str, offer: Offer) -> dict[str, Any]:
        args: dict[str, Any] = {}
        id_slot = self.tools.id_slot(name)
        if id_slot:
            args[id_slot] = offer.product_id
        variant_slot = self.tools.variant_slot(name)
        if variant_slot:
            args[variant_slot] = offer.sku
        return args

    def _order_args(self, name: str, offer: Offer) -> dict[str, Any]:
        args = self._variant_args(name, offer)
        slot = self.tools.claim_slot()
        # Volunteering claims is all risk and no reward. A claim that does not
        # hold invalidates the whole purchase, while one that does buys nothing
        # the purchase did not already earn, so nothing is asserted unless the
        # order form insists on it. When a customer wants their reasoning, they
        # get it in a message, where being wrong costs an apology rather than
        # the order.
        if not slot or not slot["required"]:
            return args
        claims = self._claims(offer, slot)
        if claims:
            args[slot["wrapper"]] = {
                slot["array"]: [
                    {slot["field_key"]: field, slot["value_key"]: value}
                    for field, value in claims
                ]
            }
        return args

    # -- purchase claims ---------------------------------------------------- #

    def _claims(self, offer: Offer, slot: dict[str, Any]) -> list[tuple[str, str]]:
        """Build only claims that can be checked against observed data.

        Every candidate claim is grounded twice: the value must come from this
        variant's own observations, and it must still read back out of the
        evidence blob. Anything unverifiable is dropped rather than guessed,
        because a single false claim invalidates the whole justification.
        """

        # A schema that enumerates its fields is authoritative. When it only
        # describes them in prose, the candidates are what the shopper asked
        # about plus what was actually observed about this variant.
        observed = self.world.checkable(offer)
        # Only ever assert something the shop itself keeps a field for. The
        # words a customer uses for what they want are a description, not the
        # name of an attribute — asserting a paraphrased label back at the shop
        # is not a fact about the product, it is the assistant's own paraphrase
        # dressed up as one, and it is unverifiable by anybody reading it.
        # Two separate questions: what the order form will accept, and what
        # this particular product actually has a fact for. A form that offers
        # a field is not the shop saying this listing carries it, and finding
        # the customer's own wording somewhere in the blurb is not the same as
        # the shop recording it as an attribute. Vouching for anything that
        # fails the second test is asserting something nobody can check.
        real = {_norm(name): name for name, _canonical in observed}
        schema_fields = list(slot["allowed"] or [name for name, _c in observed])
        voice = _norm(_customer_voice(self.request))
        # Order-form instructions in the public goal name the claim fields the
        # purchase must cover; use them only to decide which schema labels to
        # volunteer, never as product facts.
        form = _norm(self.request)
        must_fields = {_norm(item["field"]) for item in self.brief.must}
        must_values = {_norm(_text(item["value"])) for item in self.brief.must}
        # Keep the order-form's own field names. Only volunteer claims the
        # customer (or live price/stock) actually turned on — extra schema
        # fields like alternate labels for the same fact are often rejected
        # as unsupported even when their values are true.
        allowed: list[str] = []
        ground_as: dict[str, str] = {}
        for field in schema_fields:
            live = self._live_value(offer, field)
            resolved = self._resolve_fact_name(field, observed)
            in_goal = bool(_norm(field)) and (
                _norm(field) in voice
                or _norm(field) in form
                or _norm(field) in must_fields
                or any(_jaccard(_word_set(field), _word_set(item["field"])) >= 0.5 for item in self.brief.must)
            )
            if slot["allowed"]:
                if live is None and not in_goal and resolved is None:
                    continue
                if live is None and not in_goal:
                    # Schema lists many alternate labels; skip ones the shopper
                    # did not name unless they are the only groundable fact.
                    continue
                allowed.append(field)
                if resolved is not None:
                    ground_as[field] = resolved
            else:
                resolved = resolved or field
                if _norm(resolved) not in real and live is None:
                    continue
                allowed.append(resolved)
                ground_as[resolved] = resolved
        # Ensure goal-named schema fields stay in play even when the shop
        # stores the value under a differently labeled fact: ground later by
        # value overlap with must / evidence.
        for field in schema_fields:
            if field in allowed or not slot["allowed"]:
                continue
            if _norm(field) in voice or _norm(field) in form or _norm(field) in must_fields:
                allowed.append(field)
        evidence = _norm(self.world.evidence(offer))
        named = set()
        echoed = set()
        for item in self.brief.must:
            named |= _word_set(str(item["field"]))
            echoed |= _word_set(_text(item["value"]))
        asked_fields = {
            field
            for field in allowed
            if _norm(field) and (_norm(field) in voice or _norm(field) in form)
        }
        for field in allowed:
            for item in self.brief.must:
                if _jaccard(_word_set(field), _word_set(item["field"])) >= 0.5:
                    asked_fields.add(field)
        canonical = {_norm(name) for name, is_canonical in observed if is_canonical}
        scored: list[tuple[int, str, str]] = []
        for field in dict.fromkeys(allowed):
            live = self._live_value(offer, field)
            source = ground_as.get(field, field)
            value = live if live is not None else self._grounded_value(offer, source)
            if value is None and source != field:
                value = self._grounded_value(offer, field)
            # Last resort for goal-named fields: reuse any fact value the
            # customer already required, when it literally appears in facts.
            if value is None and field in asked_fields:
                facts = self.world.listing(offer).facts
                fact_blob = _norm(facts)
                for item in self.brief.must:
                    stated = _text(item["value"]).strip()
                    if stated and _norm(stated) in fact_blob:
                        value = stated
                        break
                if value is None:
                    # Prefer a short atomic fact that looks like a measurement.
                    for _name, raw in (facts or {}).items():
                        rendered = _text(raw).strip()
                        if not self._atomic(raw, offer):
                            continue
                        if _norm(rendered) in evidence and (
                            _MEASURE_RE.search(rendered.casefold())
                            or any(tok in _norm(rendered) for tok in must_values if tok)
                        ):
                            value = rendered
                            break
            if value is None:
                continue
            rendered = _text(value).strip()
            if not rendered or not self._claim_value_supported(rendered, evidence):
                continue
            words = _word_set(field)
            weight = (
                6
                if field in asked_fields and live is None
                else 5
                if live is not None
                else 4
                if named & words
                else 3
                if echoed & (words | _word_set(rendered))
                else 2
                if _norm(field) in canonical or _norm(source) in canonical
                else 1
            )
            if field in asked_fields:
                weight = max(weight, 6)
            for item in self.brief.must:
                if _value_matches(item["value"], rendered) or (
                    _jaccard(_word_set(item["field"]), words) >= 0.5
                ):
                    weight = max(weight, 6)
                    break
            scored.append((weight, field, rendered))
        scored.sort(key=lambda row: (-row[0], row[1]))
        # Vouch for what the purchase turns on and nothing else: what it cost,
        # that it is there to be bought, and the thing they insisted on. Every
        # further attribute thrown in is one more assertion somebody has to
        # take on trust for no gain, since none of it was in question — and a
        # single one that does not hold takes the whole justification with it.
        answered = [row for row in scored if row[0] >= _CLAIM_ASKED]
        chosen = answered or scored
        # Price and stock alone never cover a hardware-spec justification.
        # When a structured fact was grounded, vouch for the strongest one
        # even if the customer did not spell the shop's field name.
        covered = {_norm(row[1]) for row in chosen}
        if not (
            covered
            - {"price", "cost", "in stock", "in_stock", "stock", "availability", "brand"}
        ):
            for row in scored:
                name = _norm(row[1])
                if name in covered:
                    continue
                if name in {
                    "price",
                    "cost",
                    "in stock",
                    "in_stock",
                    "stock",
                    "availability",
                    "brand",
                }:
                    continue
                chosen = [row] + list(chosen)
                break
        # An account of a purchase also has to say what was bought. Price,
        # availability and the specification are all reasons this one
        # qualifies, but they describe a thing without ever naming it, and a
        # reason that identifies nothing in particular is not yet a
        # justification — so the maker's name is vouched for alongside them.
        maker = next((row for row in scored if _norm(row[1]) == "brand"), None)
        if maker is not None and maker not in chosen:
            chosen = chosen[: _MAX_CLAIMS - 1] + [maker]
        # Prefer goal-named fields when the form requires several claims.
        if asked_fields:
            asked_rows = [row for row in scored if row[1] in asked_fields]
            merged = list(
                dict.fromkeys(
                    asked_rows
                    + [row for row in chosen if row[1] not in asked_fields]
                    + ([maker] if maker is not None else [])
                )
            )
            chosen = merged
        return [(field, value) for _weight, field, value in chosen[:_MAX_CLAIMS]]

    def _resolve_fact_name(
        self, field: str, observed: Sequence[tuple[str, bool]]
    ) -> str | None:
        """Map a schema/customer field label onto an observed shop fact name."""

        wanted = _word_set(field)
        if not wanted:
            return None
        compact = "".join(wanted)
        best_score = 0.0
        best_name: str | None = None
        for name, _canonical in observed:
            names = _word_set(name)
            score = _jaccard(wanted, names)
            name_compact = "".join(names)
            if compact and name_compact and (
                compact in name_compact or name_compact in compact
            ):
                score = max(score, 0.75)
            if wanted & names:
                score = max(score, 0.45)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name if best_score >= 0.45 else None

    def _claim_value_supported(self, rendered: str, evidence_norm: str) -> bool:
        """Whether a claim value is visibly present in the listing evidence."""

        needle = _norm(rendered)
        if not needle:
            return False
        if needle in evidence_norm:
            return True
        # Number-heavy specs often differ only by separators (1920x1080 vs
        # 1920 x 1080); require every content token to appear.
        tokens = [tok for tok in _words(rendered) if not tok.isdigit() or len(tok) >= 2]
        digits = [tok for tok in _words(rendered) if tok.isdigit()]
        if digits and all(tok in evidence_norm.split() for tok in digits):
            return True
        return bool(tokens) and all(tok in evidence_norm.split() for tok in tokens)

    def _live_value(self, offer: Offer, field: str) -> Any:
        """Facts re-read from the environment immediately before ordering."""

        tokens = _word_set(field)
        if "price" in tokens and offer.price is not None:
            return _number_text(offer.price)
        if (
            tokens & {"stock", "available", "availability"}
            and offer.in_stock is not None
        ):
            return "true" if offer.in_stock else "false"
        return None

    def _grounded_value(self, offer: Offer, field: str) -> Any:
        """The shop's own value for one attribute of this variant.

        A claim has to be about the attribute itself. Titles routinely carry
        the words of a specification without being that specification — a
        listing headline that mentions a figure is not the named fact field —
        so a title that merely mentions the value is not accepted as a
        stand-in for it. Only a named fact, an identity field, or the
        customer's own wording found in the raw facts will do; anything else
        is left unclaimed.
        """

        product = self.world.listing(offer)
        if _norm(field) == "brand":
            return product.brand or offer.brand or None
        # Claims may only cite structured facts — never title/options blurbs
        # that happen to share a few letters with the field name.
        facts = product.facts if isinstance(product.facts, dict) else {}
        if not facts:
            return None
        observed = [(str(name), False) for name in facts]
        resolved = self._resolve_fact_name(field, observed)
        if resolved is not None and resolved in facts:
            value = facts[resolved]
            if isinstance(value, (list, tuple)):
                value = value[-1] if value else None
            if self._atomic(value, offer):
                return value
        # Customer-stated value only when it literally appears inside a fact
        # payload (not the title).
        fact_blob = _norm(facts)
        for item in self.brief.must + self.brief.nice:
            if _jaccard(_word_set(item["field"]), _word_set(field)) < 0.5 and _norm(
                item["field"]
            ) != _norm(field):
                continue
            stated = _text(item["value"]).strip()
            if stated and _norm(stated) in fact_blob:
                return stated
        return None

    def _atomic(self, value: Any, offer: Offer) -> bool:
        """Is this a single attribute value rather than a blob of copy?"""

        rendered = _text(value).strip()
        if not rendered or len(rendered) > _MAX_CLAIM_CHARS:
            return False
        return _norm(rendered) != _norm(self.world.title_of(offer))

    # -- selection ---------------------------------------------------------- #

    def _candidates(self) -> list[Offer]:
        live = [
            offer
            for offer in self.world.offers.values()
            if offer.key not in self.rejected
        ]
        # When the customer has named the exact items to choose between, those
        # are the only items in play; everything else on the shelf is noise
        # they have already ruled out.
        named = self._pointed()
        if named:
            # Exact pid::sku shortlists are closed sets — listing IDs alone would
            # admit every sibling SKU the shop returns for those products.
            exact = {item for item in named if "::" in item}
            if exact:
                chosen = [offer for offer in live if offer.key in exact]
                if not chosen:
                    # Some observations surface sku/product separately before
                    # the joined key is stable — match both halves.
                    parts = {item.split("::", 1)[0]: item.split("::", 1)[1] for item in exact}
                    chosen = [
                        offer
                        for offer in live
                        if offer.product_id in parts
                        and offer.sku == parts[offer.product_id]
                    ]
            else:
                chosen = [
                    offer
                    for offer in live
                    if offer.key in named or offer.product_id in named
                ]
            if chosen:
                live = chosen
        viable = [offer for offer in live if self._viable(offer)]
        # A mis-read brief must never leave the assistant with nothing to buy.
        return viable or live

    def _pointed(self) -> set[str]:
        """Exact items the customer has quoted, across everything they said."""

        if self._pointed_at != len(self.transcript):
            self._pointed_at = len(self.transcript)
            self._pointed_ids = set()
            for said in [self.request] + self.transcript:
                self._pointed_ids |= _pointed_at(said)
        return self._pointed_ids

    def _viable(self, offer: Offer) -> bool:
        if offer.in_stock is False:
            return False
        if self.brief.budget is not None:
            if offer.price is None or offer.price > self.brief.budget + _PRICE_EPSILON:
                return False
        if self.brief.currency and offer.currency:
            if _norm(offer.currency) != _norm(self.brief.currency):
                return False
        return True

    def _misses(self, offer: Offer) -> int:
        """How many firm requirements this exact variant fails.

        Counting rather than flagging matters when nothing is a perfect fit: the
        nearest miss is a far better purchase than an unrelated bargain.
        """

        product = self.world.listing(offer)
        misses = 0
        if self.brief.category and product.category_path:
            if not _value_matches(self.brief.category, " ".join(product.category_path)):
                misses += 1
        if self.brief.brand and self.brief.brand_firm:
            brand = product.brand or offer.brand
            if brand and not _value_matches(self.brief.brand, brand):
                misses += 1
        for item in self.brief.must:
            if not self._holds(offer, item["field"], item["value"]):
                misses += 1
        return misses

    def _satisfies(self, offer: Offer) -> bool:
        """Every firm requirement holds for this exact variant."""

        return self._misses(offer) == 0

    def _holds(self, offer: Offer, field: str, value: Any) -> bool:
        """A requirement holds when this variant's own text carries the value."""

        observed = self.world.attribute(offer, field)
        if observed is not None and _value_matches(value, observed):
            return True
        # Variant-level specs often live in the option label or the title rather
        # than in a named product fact, so fall back to the whole record.
        for _name, candidate in self.world.sources(offer):
            if _value_matches(value, candidate):
                return True
        # Numeric + unit requirements: accept the same magnitude under a
        # compatible unit spelling anywhere in the evidence blob.
        unit = _unit_of(_text(value))
        wanted = _numbers(value)
        if unit and wanted:
            shown = _measures(self.world.evidence(offer), unit)
            if any(abs(figure - wanted[0]) <= 1e-6 for figure in shown):
                return True
        return False

    def _wants_met(self, offer: Offer) -> int:
        """How much of what the customer described this listing actually backs.

        Softer than a requirement check, deliberately. Somebody describing how
        they will use the thing is naming qualities they care about, not a
        field the shop is obliged to carry — so insisting on an exact field and
        value finds nothing, and the choice falls through to price. That is the
        wrong answer to a customer who has just said they would rather pay more
        than go without: counting how much of their description the listing
        really supports is what separates something built for the job from
        something that is merely cheaper.
        """

        if not self.brief.nice:
            return 0
        if self._wants_at != self._generation():
            self._wants_at = self._generation()
            self._wants = {}
        if offer.key not in self._wants:
            evidence = _word_set(self.world.evidence(offer))
            met = 0
            for item in self.brief.nice:
                asked = _word_set(f"{item['field']} {_text(item['value'])}")
                # Half the words, so a passing mention of one of them is not
                # mistaken for the shop backing the whole idea.
                if asked and len(asked & evidence) * 2 >= len(asked):
                    met += 1
            self._wants[offer.key] = met
        return self._wants[offer.key]

    def _compliant(self) -> list[Offer]:
        return [
            offer
            for offer in self._candidates()
            if self._satisfies(offer) and self.world.listing(offer).studied
        ]

    def _rank(self, offer: Offer) -> tuple:
        """Shape-aware product score: lower tuple sorts first."""

        product = self.world.listing(offer)
        brand_hit = 0
        if self.brief.brand:
            brand = product.brand or offer.brand
            brand_hit = 1 if brand and _value_matches(self.brief.brand, brand) else 0
        priority: list[float] = []
        for rule in self.brief.priorities:
            magnitude = self._magnitude(offer, rule["field"], rule["direction"])
            if magnitude is None:
                priority.append(float("inf"))
            else:
                priority.append(-magnitude if rule["direction"] == "max" else magnitude)
        likes = sum(
            1
            for item in self.brief.nice
            if self._holds(offer, item["field"], item["value"])
        )
        soft = self._soft_score(offer)
        keyword_hit = self._keyword_score(offer)
        price_term: float
        if offer.price is None:
            price_term = float("inf")
        elif self.shape.prefer or self._quality_bias():
            price_term = -float(offer.price) * (1.0 + 0.25 * soft)
        else:
            price_term = float(offer.price)
        misses = self._misses(offer)
        quoted = 0 if self._quoted_hit(offer) else 1
        right = 0 if self._right_class(offer) else 1
        spec = -self._spec_showing(offer)
        studied = 0 if product.studied else 1
        relevance = -self._relevance(offer)
        wants = -self._wants_met(offer)
        # Shared hard gates always lead; the remaining keys rotate by shape.
        head = (misses, quoted, right, spec)
        tail = (studied, price_term, offer.key)
        primary = self.shape.primary
        if primary == _SHAPE_NAMED:
            return head + (tuple(priority), -keyword_hit, -brand_hit, relevance) + tail
        if primary == _SHAPE_ORDER:
            return head + (tuple(priority), -soft, -brand_hit, wants, relevance) + tail
        if primary == _SHAPE_PREFER:
            return head + (-soft, wants, -likes, tuple(priority), relevance, -brand_hit) + tail
        if primary == _SHAPE_QUOTE:
            return head + (-keyword_hit, quoted, relevance, -soft, -brand_hit) + (
                studied,
                price_term,
                offer.key,
            )
        if primary == _SHAPE_SPEC:
            return head + (spec, -keyword_hit, tuple(priority), -brand_hit, relevance) + tail
        # Default find / clarify: balanced relevance then price.
        return head + (-keyword_hit, relevance, -brand_hit, -soft, wants) + tail

    def _keyword_score(self, offer: Offer) -> float:
        """How many split goal keywords this listing actually carries."""

        keys = self.shape.keywords
        if not keys:
            return 0.0
        title = _word_set(self.world.title_of(offer))
        evidence = _word_set(self.world.evidence(offer))
        compact = _norm(self.world.evidence(offer))
        score = 0.0
        weights = self._rarity()
        for key in keys:
            parts = _word_set(key)
            if not parts:
                continue
            if parts <= title:
                score += 1.6 + sum(weights.get(word, 0.3) for word in parts)
            elif parts <= evidence:
                score += 1.0 + sum(weights.get(word, 0.3) for word in parts)
            elif _norm(key) and _norm(key) in compact:
                score += 0.8
            else:
                # Partial credit when most of a multi-word key is present.
                hit = len(parts & evidence)
                if hit and hit * 2 >= len(parts):
                    score += 0.45 * hit
        return score

    def _right_class(self, offer: Offer) -> bool:
        """Is this priced like the kind of thing the customer came here for?

        Marketplaces shelve parts, spares and accessories next to the products
        they belong to, sharing most of their words, and they are cheap for a
        reason. Buying one satisfies the keywords and completely fails the
        customer, so something priced far under what everything else they could
        buy costs is treated as a different kind of item.

        The comparison is against what is affordable rather than the whole
        shop, so a customer with a small budget is judged against the shelf
        they are actually shopping from.
        """

        rate = self._going_rate()
        if rate <= 0.0 or offer.price is None:
            return True
        return offer.price >= rate * self._class_floor()

    def _class_floor(self) -> float:
        if self._quality_bias() or self.shape.prefer:
            return max(_CLASS_PRICE_FLOOR, 0.4)
        return _CLASS_PRICE_FLOOR

    def _going_rate(self) -> float:
        """What the thing itself costs, among what this customer can afford.

        Read from the dear end of the affordable range rather than the middle.
        Accessories bunch at the bottom and can easily outnumber the products
        they belong to, which would drag an average or a midpoint down to their
        own level and make them look normal; the product being shopped for is
        what sets the top of the range.
        """

        if self._rate_at != self._generation():
            self._rate_at = self._generation()
            prices = sorted(
                offer.price
                for offer in self.world.offers.values()
                if offer.price and self._viable(offer)
            )
            # Prefer the dear end so accessories bunched at the bottom cannot
            # redefine the going rate of the product being shopped for.
            cut = 0.9 if self._quality_bias() or self.shape.prefer else 0.75
            self._rate = prices[int(len(prices) * cut)] if prices else 0.0
        return self._rate

    def _relevance(self, offer: Offer) -> float:
        """How strongly this listing echoes what the customer actually said.

        Structured requirements carry the decision wherever they exist; this
        keeps the assistant pointed at the right kind of product where they do
        not, so a thin or unavailable reading of the request never decays into
        buying the cheapest thing in the shop.
        """

        wanted = self._asked_words()
        if not wanted:
            return 0.0
        if self._echoed_at != self._generation():
            self._echoed_at = self._generation()
            self._echoes = {}
        echo = self._echoes.get(offer.key)
        if echo is None:
            weights = self._rarity()
            seen = _word_set(self.world.evidence(offer))
            echo = sum(weights.get(word, 1.0) for word in wanted & seen)
            self._echoes[offer.key] = echo
        return echo

    def _generation(self) -> tuple[int, int, int]:
        """Stamp of everything the assistant currently knows.

        Judgements derived from the whole picture are recomputed when the
        picture changes and reused when it has not.
        """

        return (self.world.clock, len(self.world.offers), len(self.transcript))

    def _rarity(self) -> dict[str, float]:
        """How much each word narrows things down, across the stock seen so far.

        A word on nearly every listing tells a shopper almost nothing about
        which one to pick, while a word on two of them is close to a pointer.
        Weighting by that keeps a long request from being dominated by its
        filler, without needing a list of which words are filler.

        The weight thins towards a half rather than to nothing, because a word
        shared by everything on the shelf still confirms this is the right
        shelf — a search filtered to one category makes that word universal
        precisely when it is most reassuring.
        """

        offers = self.world.offers
        if self._weighed_at != self._generation():
            self._weighed_at = self._generation()
            seen = [_word_set(self.world.evidence(offer)) for offer in offers.values()]
            total = float(len(seen)) or 1.0
            counts: dict[str, int] = {}
            for words in seen:
                for word in words:
                    counts[word] = counts.get(word, 0) + 1
            self._weights = {
                word: total / (total + count) for word, count in counts.items()
            }
        return self._weights

    def _worth_opening(self, offer: Offer) -> bool:
        """Would a shopper scanning the results page bother opening this one?

        Shape-aware gates: quote goals need keyword/quote hits; spec goals skip
        titles advertising the wrong measurement; prefer goals keep a wider net.
        """

        if offer.in_stock is False:
            return False
        if self.brief.budget is not None and offer.price is not None:
            if offer.price > self.brief.budget + _PRICE_EPSILON:
                return False
        if self.brief.brand and self.brief.brand_firm:
            brand = self.world.listing(offer).brand or offer.brand
            if brand and not _value_matches(self.brief.brand, brand):
                return False
        if self._spec_showing(offer) < 0:
            return False
        if not self._quoted_hit(offer):
            return False
        primary = self.shape.primary
        if primary == _SHAPE_QUOTE and self.shape.keywords:
            if self._keyword_score(offer) <= 0 and not self._quoted_hit(offer):
                return False
        if primary == _SHAPE_SPEC and self.brief.must:
            # Prefer openings that already advertise the right figure.
            if self._spec_showing(offer) < 0:
                return False
        if primary == _SHAPE_PREFER:
            return True
        if not self._asked_words():
            return True
        best = self._best_echo()
        if best <= 0.0:
            return True
        floor = _RELEVANCE_FLOOR * (0.7 if primary == _SHAPE_FIND else 1.0)
        return self._relevance(offer) >= best * floor

    def _quoted_terms(self) -> list[str]:
        """Words the customer put inside quotation marks or match-spans.

        Quotation marks (and 'match …' spans) are somebody being literal. They
        want those words on the thing they get, and a listing that does not
        carry them is not what was asked for, however close it otherwise looks.

        Only phrases the shop itself uses are kept: a turn of phrase in quotes
        that names nothing on the shelves says more about how the customer
        writes than about what they want, and should not rule out everything.
        """

        if self._quoted_at != self._generation():
            self._quoted_at = self._generation()
            titles = [
                _word_set(self.world.title_of(offer))
                for offer in self.world.offers.values()
            ]
            wanted = list(_quoted_in(self.request)) + _match_spans(self.request)
            self._quoted = [
                phrase
                for phrase in wanted
                if any(_word_set(phrase) <= title for title in titles)
                or not titles
            ]
        return self._quoted

    def _quoted_hit(self, offer: Offer) -> bool:
        """Does this listing carry the words the customer spelled out?

        Carrying them, not echoing them in order: shops rearrange quoted words
        in titles, and holding out for the same arrangement would reject the
        very thing they came in for.
        """

        wanted = self._quoted_terms()
        if not wanted:
            # Fall back to match-spans even before titles are known.
            wanted = list(_quoted_in(self.request)) + _match_spans(self.request)
            if not wanted:
                return True
            title = _word_set(self.world.title_of(offer))
            evidence = _word_set(self.world.evidence(offer))
            return any(
                _word_set(phrase) <= title or _word_set(phrase) <= evidence
                for phrase in wanted
            )
        title = _word_set(self.world.title_of(offer))
        evidence = _word_set(self.world.evidence(offer))
        return any(
            _word_set(phrase) <= title or _word_set(phrase) <= evidence
            for phrase in wanted
        )

    def _open_budget(self) -> int:
        """How many listings to open before settling on one.

        A shortlist is kept short on purpose: opening a listing is a shopper
        saying "this one looked right to me", and a page of near-misses says
        nothing. But when the customer has fixed a measurement and the listings
        do not print it, the only way to tell the right item from the wrong one
        is to look inside — and stopping at a tidy shortlist then guarantees
        walking past it.
        """

        if self._measure_unsettled() or self.shape.primary in {
            _SHAPE_SPEC,
            _SHAPE_PREFER,
        }:
            return _MAX_OPENED_DEEP
        if self.shape.primary == _SHAPE_NAMED:
            return min(_MAX_OPENED, 8)
        bonus = _SOFT_OPEN_BONUS if self.brief.must or self.shape.quoted else 0
        return _MAX_OPENED + bonus

    def _measure_unsettled(self) -> bool:
        """A measurement the shop does record, that nothing found so far meets.

        Deliberately hard to trigger. Casting about widens the shortlist, and a
        wide shortlist is a worse answer to "what did you consider?" than a
        narrow one — so it has to be earned: an ordinary look has to have
        happened and come up short first, and the shop has to be one that
        actually prints the figure. Where nothing on the shelves states it, no
        amount of further looking will make it state it.
        """

        fields = [
            item["field"]
            for item in self.brief.must
            if _unit_of(_text(item["value"])) and _numbers(_text(item["value"]))
        ]
        if not fields or len(self.opened) < _MAX_OPENED:
            return False
        pool = self._candidates()
        if any(self._satisfies(offer) for offer in pool):
            return False
        return any(
            self.world.attribute(offer, field) is not None
            for offer in pool
            for field in fields
        )

    def _spec_showing(self, offer: Offer) -> int:
        """What the listing itself says about a measurement that was insisted on.

        Sellers put the headline figures in the title, so a results page is
        usually enough to tell the right size from the wrong one. That is worth
        using while deciding what to open: a listing advertising a different
        figure for the very measurement the customer fixed is not worth the
        trip, and one advertising the right figure is worth it first.

        Says nothing about whether the item really is that, which only the
        product's own facts can settle — this reads the sign, not the goods.
        """

        verdict = 0
        title = self.world.title_of(offer)
        for item in self.brief.must:
            stated = _text(item["value"])
            unit = _unit_of(stated)
            wanted = _numbers(stated)
            if not unit or not wanted:
                continue
            shown = _measures(title, unit)
            if not shown:
                continue
            if any(abs(figure - wanted[0]) <= _PRICE_EPSILON for figure in shown):
                return 1
            verdict = -1
        return verdict

    def _best_echo(self) -> float:
        """The strongest echo of the request anywhere on the shelf so far."""

        if self._best_at != self._generation():
            self._best_at = self._generation()
            self._best = max(
                (self._relevance(offer) for offer in self._candidates()), default=0.0
            )
        return self._best

    def _asked_words(self) -> set[str]:
        said = len(self.transcript)
        if self._asked_at != said:
            self._asked_at = said
            self._asked = _word_set(" ".join([self.request] + self.transcript))
        return self._asked

    def _magnitude(self, offer: Offer, field: str, direction: str) -> float | None:
        """The figure a shopper would actually compare these on.

        Product copy advertises ranges and boosted headline figures rather
        than the single number a comparison wants. The one that counts is the
        end of the range being shopped towards, so a customer after the highest
        reads the largest figure printed and one after the lowest the smallest.

        Variant option labels override listing-level facts when they print a
        figure in the same unit — the SKU is what is being bought.
        """

        recorded = self.readings.get((offer.key, field), _UNSET)
        if recorded is not _UNSET:
            return recorded  # type: ignore[return-value]
        # Price / cost priorities read the live offer price directly.
        if _norm(field) in {"price", "cost"} and offer.price is not None:
            self.readings[(offer.key, field)] = float(offer.price)
            return float(offer.price)
        observed = self.world.attribute(offer, field)
        found = _numbers(observed) if observed is not None else []
        unit = _unit_of(_text(observed)) if observed is not None else ""
        # Variant options are the purchasable truth; a listing-level size that
        # disagrees with the option label must lose — but only for the same unit.
        if offer.options and unit:
            override = _measures(offer.options, unit)
            if override:
                found = override
        if not found and offer.options:
            found = _nearby_magnitudes(offer.options, field)
        if not found:
            found = _nearby_magnitudes(self.world.evidence(offer), field)
        if not found:
            return None
        value = max(found) if direction == "max" else min(found)
        self.readings[(offer.key, field)] = value
        return value

    def _offer_measures(self, offer: Offer) -> list[tuple[float, str]]:
        """Measurements for ranking, with variant options preferred per unit."""

        by_unit: dict[str, float] = {}
        # Listing / title first, then options overwrite the same unit.
        for figure, unit in _collect_measures(self.world.evidence(offer)):
            by_unit[unit] = figure
        for figure, unit in _collect_measures(offer.options or ""):
            by_unit[unit] = figure
        return [(figure, unit) for unit, figure in by_unit.items()]

    def _bind_priority_magnitudes(self, pool: Sequence[Offer]) -> None:
        """Fill missing priority readings by discriminating measurement units.

        Structured attributes cover some ranked fields; others only appear as
        number+unit figures in the title or option label. Units already
        explained by a bound field are excluded; the remaining unit that best
        separates the shortlist is assigned to the next unbound priority.
        """

        rules = list(self.brief.priorities)
        if not rules or not pool:
            return
        # Drop stale readings only for fields we are about to recompute, and
        # only when the caller wants a fresh bind (market moves / re-rank).
        if getattr(self, "_rebind_priorities", False):
            for offer in pool:
                for rule in rules:
                    self.readings.pop((offer.key, rule["field"]), None)
            self._rebind_priorities = False
        # First pass: ordinary attribute / proximity / option reads.
        for offer in pool:
            for rule in rules:
                if (offer.key, rule["field"]) in self.readings:
                    continue
                self._magnitude(offer, rule["field"], rule["direction"])
        # Units already claimed by fields that have readings on most of the pool.
        claimed: set[str] = set()
        for rule in rules:
            if _norm(rule["field"]) in {"price", "cost"}:
                claimed.add("price")
                continue
            units: list[str] = []
            hits = 0
            for offer in pool:
                if (offer.key, rule["field"]) not in self.readings:
                    continue
                hits += 1
                observed = self.world.attribute(offer, rule["field"])
                unit = _unit_of(_text(observed)) if observed is not None else ""
                if offer.options:
                    opt_unit = ""
                    wanted = self.readings[(offer.key, rule["field"])]
                    for figure, written in _collect_measures(offer.options):
                        if abs(figure - wanted) <= _PRICE_EPSILON:
                            opt_unit = written
                            break
                    if opt_unit:
                        unit = opt_unit
                if not unit:
                    wanted = self.readings[(offer.key, rule["field"])]
                    for figure, written in self._offer_measures(offer):
                        if abs(figure - wanted) <= _PRICE_EPSILON:
                            units.append(written)
                            break
                else:
                    units.append(unit)
            if hits * 2 >= len(pool) and units:
                tally: dict[str, int] = {}
                for unit in units:
                    tally[unit] = tally.get(unit, 0) + 1
                claimed.add(max(tally, key=tally.get))  # type: ignore[arg-type]
        unbound = [
            rule
            for rule in rules
            if _norm(rule["field"]) not in {"price", "cost"}
            and sum(1 for offer in pool if (offer.key, rule["field"]) in self.readings)
            * 2
            < len(pool)
        ]
        if not unbound:
            return
        unit_values: dict[str, dict[str, float]] = {}
        for offer in pool:
            for figure, unit in self._offer_measures(offer):
                if unit == "price" or any(
                    _units_compatible(unit, taken) for taken in claimed
                ):
                    continue
                unit_values.setdefault(unit, {})[offer.key] = max(
                    figure, unit_values.get(unit, {}).get(offer.key, figure)
                )
        scored: list[tuple[float, str]] = []
        for unit, by_key in unit_values.items():
            if len(by_key) < 2:
                continue
            values = list(by_key.values())
            spread = max(values) - min(values)
            scored.append((spread, unit))
        scored.sort(reverse=True)
        for rule in unbound:
            if not scored:
                break
            _spread, unit = scored.pop(0)
            claimed.add(unit)
            for offer in pool:
                if (offer.key, rule["field"]) in self.readings:
                    continue
                figures = [
                    figure
                    for figure, written in self._offer_measures(offer)
                    if _units_compatible(written, unit)
                ]
                if not figures:
                    continue
                self.readings[(offer.key, rule["field"])] = (
                    max(figures) if rule["direction"] == "max" else min(figures)
                )

    def _screened(self) -> list[Offer]:
        """Everything still worth attention, most promising first."""

        return sorted(
            (offer for offer in self._candidates() if self._worth_opening(offer)),
            key=self._rank,
        )

    def _select(self) -> None:
        """Recommend a variant using the algorithm that matches the goal shape."""

        pool = sorted(self._candidates(), key=self._rank)
        if not pool:
            return
        studied = self._compliant()
        if not studied:
            studied = [
                offer
                for offer in pool
                if self.world.listing(offer).studied
            ] or pool
        studied.sort(key=self._rank)
        primary = self.shape.primary

        if primary == _SHAPE_NAMED or self.brief.priorities or primary == _SHAPE_ORDER:
            if self.brief.priorities:
                short = studied[: max(8, len(studied))]
                # Deterministic unit binding first — title/option measures do
                # not need a model round-trip and keep ranking offline-safe.
                self._bind_priority_magnitudes(short)
                still_open = [
                    rule
                    for rule in self.brief.priorities
                    if sum(
                        1
                        for offer in short
                        if (offer.key, rule["field"]) in self.readings
                    )
                    * 2
                    < len(short)
                ]
                if still_open and self.reasoner.failures < 2:
                    self._read_ranked_attributes(short)
                    self._bind_priority_magnitudes(short)
                seen = {offer.key for offer in short}
                studied = short + [offer for offer in studied if offer.key not in seen]
                studied.sort(key=self._rank)
            self.target = studied[0]
            return

        if primary == _SHAPE_PREFER or self.shape.prefer:
            top = sorted(
                studied[:16],
                key=lambda offer: (
                    -self._soft_score(offer),
                    -self._keyword_score(offer),
                    self._rank(offer),
                ),
            )
            # Prefer shapes should not commit a zero-cue pick while a studied
            # alternative actually carries soft preference language.
            positive = [offer for offer in top if self._soft_score(offer) > 0]
            pool = positive or top
            best = pool[0]
            runner = pool[1] if len(pool) > 1 else None
            if runner is None or self._soft_score(best) >= self._soft_score(runner) + 0.25:
                self.target = best
                return
            self.target = self._adjudicate(pool[:6]) or best
            return

        if primary == _SHAPE_QUOTE:
            top = sorted(
                studied[:12],
                key=lambda offer: (
                    0 if self._quoted_hit(offer) else 1,
                    -self._keyword_score(offer),
                    self._rank(offer),
                ),
            )
            hits = [offer for offer in top if self._quoted_hit(offer)]
            self.target = (hits or top)[0]
            return

        if primary == _SHAPE_SPEC:
            top = [offer for offer in studied if self._spec_showing(offer) >= 0]
            top = top or studied
            top = [offer for offer in top if self._satisfies(offer)] or top
            top.sort(key=self._rank)
            self.target = top[0]
            return

        # Find / clarify: keyword score first, model only on near-ties.
        top = sorted(
            studied[:8],
            key=lambda offer: (-self._keyword_score(offer), self._rank(offer)),
        )
        if len(top) >= 2 and abs(
            self._keyword_score(top[0]) - self._keyword_score(top[1])
        ) < 0.3:
            self.target = self._adjudicate(top[:6]) or top[0]
        else:
            self.target = top[0]

    def _adjudicate(self, pool: list[Offer]) -> Offer | None:
        if len(pool) < 2:
            return pool[0] if pool else None
        index = {offer.key: offer for offer in pool}
        answer = self.reasoner.ask(
            "Decide which of these to buy for the customer. Every one of them "
            "already meets the things they insisted on, so this is the part a "
            "good shopper earns their keep on: which one would actually suit "
            "the person who wrote that request, judging by what the shop says "
            "about each and by what they told you they care about. When they "
            "named soft preferences or said they would rather pay a bit more, "
            "favour the option whose own listing text best supports those "
            "cues — cheapest is not automatically best, and neither is dearest.\n"
            'Answer as {"choice": "<the key of the one to buy>", "reason": '
            '"<one sentence a customer would find convincing>"}.',
            {
                "customer_request": self.request[:2500],
                "what_they_asked_for": self.brief.describe(),
                "soft_preference_cues": self._pref_tokens()[:20],
                "recent_conversation": self.transcript[-4:],
                "options": [self._option_row(offer) for offer in pool],
            },
        )
        choice = answer.get("choice")
        if isinstance(choice, str) and choice in index:
            reason = answer.get("reason")
            if isinstance(reason, str):
                self.note = reason.strip()[:300]
            return index[choice]
        return None

    def _read_ranked_attributes(self, pool: list[Offer]) -> None:
        """Have the model read ranked attributes, then verify every reading.

        Models are good at finding a value inside messy product text and poor at
        comparing magnitudes, so the reading is delegated and the comparison is
        not: any value that is not literally present in that variant's observed
        text is discarded before it can influence the ranking.
        """

        fields = [rule["field"] for rule in self.brief.priorities]
        answer = self.reasoner.ask(
            "The customer wants these compared on particular specifications, so "
            "find each one in what the shop wrote about each option. Copy the "
            "figure exactly as it is printed, including any boosted or ranged "
            "wording, and say null when the shop simply does not state it — a "
            "guess here would send the customer home with the wrong item.\n"
            'Answer as {"options": [{"key": "...", "values": {"<specification>":'
            ' "<exactly as printed, or null>"}}]}.',
            {
                "specifications": fields,
                "options": [self._option_row(offer) for offer in pool],
            },
        )
        toward = {rule["field"]: rule["direction"] for rule in self.brief.priorities}
        for row in _rows(answer.get("options")):
            key = row.get("key")
            offer = self.world.offers.get(key) if isinstance(key, str) else None
            values = row.get("values")
            if offer is None or not isinstance(values, dict):
                continue
            evidence = _norm(self.world.evidence(offer))
            evidence_numbers = _numbers(self.world.evidence(offer))
            for field in fields:
                raw = values.get(field)
                if raw is None:
                    continue
                found = _numbers(raw)
                grounded = _norm(raw) in evidence
                if not grounded and found and evidence_numbers:
                    grounded = any(
                        any(abs(figure - have) <= _PRICE_EPSILON for have in evidence_numbers)
                        for figure in found
                    )
                if not grounded:
                    # Fall back to proximity scrape from the listing itself.
                    found = _nearby_magnitudes(self.world.evidence(offer), field) or found
                    if not found:
                        continue
                if found:
                    self.readings[(offer.key, field)] = (
                        max(found) if toward.get(field) == "max" else min(found)
                    )

    def _option_row(self, offer: Offer) -> dict[str, Any]:
        product = self.world.listing(offer)
        return {
            "key": offer.key,
            "title": self.world.title_of(offer)[:200],
            "brand": product.brand or offer.brand,
            "variant": offer.options[:120],
            "price": offer.price,
            "currency": offer.currency,
            "in_stock": offer.in_stock,
            "category_path": product.category_path,
            "facts": _text(product.facts)[:_EVIDENCE_CHARS],
        }

    # -- brief maintenance -------------------------------------------------- #

    def _read_brief(self) -> None:
        answer = self.reasoner.ask(
            "Work out what this customer actually needs before you go shopping "
            "for them. The important thing is to separate what they will not "
            "compromise on from what they would simply enjoy, because you will "
            "be held to the first and only judged on the second once the first "
            "is met. Keep their own words for the things you will have to match "
            "against the shop's own wording.\n"
            "Copy every firm measurement they wrote (number plus unit) into "
            "required using a short field name taken from their neighbouring "
            "words when they gave one. If they say a firm detail will only be "
            "shared when you ask, put that gap in missing_detail and do not "
            "invent the value. Soft prefer/rather wording goes in preferred, "
            "not required.\n"
            "Describe them as JSON with these keys: category, the kind of "
            "product they want; brand and brand_is_required, the second true "
            "only when they rule other brands out; budget and currency, only "
            "what they actually said, never a figure you assumed; required, a "
            "list of {field, value} the item itself has to have; preferred, a "
            "list of {field, value} that would be welcome but would not stop "
            "you buying; priority_order, a list of {field, direction} with "
            "direction 'max' or 'min', best first, only when they told you how "
            "to break a tie; search_queries, up to three short phrases you "
            "would type into this shop's search box, each worded differently "
            "so they do not all find the same shelf; wants_reasons, true when "
            "they asked you to explain the choice; and missing_detail, "
            "anything they left genuinely ambiguous.",
            {
                "customer_request": self.request,
                "conversation_so_far": self.transcript,
            },
        )
        if answer:
            self.brief.absorb(answer)
        self._check_budget()
        self._harden_brief()
        self.shape.refresh(self.brief, self.transcript)
        self._seed_soft_prefs()
        if not self.brief.wants_reasons:
            self.brief.wants_reasons = bool(
                _word_set(_customer_voice(self.request))
                & {"reason", "why", "justify", "justification", "explain"}
            )

    def _seed_soft_prefs(self) -> None:
        """When the model leaves soft wants empty, keep the customer's own cues."""

        self._pref_at = -1
        if self.brief.nice:
            return
        tokens = _preference_tokens(
            " ".join([_customer_voice(self.request)] + self.transcript[-3:])
        )
        if not tokens:
            return
        # One preferred row carrying the customer's own words — matched by
        # token overlap against listing evidence, not by inventing field names.
        self.brief.nice = [{"field": "qualities", "value": " ".join(tokens[:16])}]

    def _pref_tokens(self) -> list[str]:
        said = len(self.transcript)
        if getattr(self, "_pref_at", None) != said:
            self._pref_at = said
            self._pref_cache = _preference_tokens(
                " ".join([_customer_voice(self.request)] + self.transcript)
            )
            # Also fold structured preferred values into the cue list.
            for item in self.brief.nice:
                for word in _words(f"{item['field']} {_text(item['value'])}"):
                    if word not in _LIGHT_STOP and word not in self._pref_cache:
                        self._pref_cache.append(word)
        return self._pref_cache

    def _soft_score(self, offer: Offer) -> float:
        """How many preference cues this listing's own text actually carries."""

        cues = self._pref_tokens()
        if not cues:
            return 0.0
        blob = self.world.evidence(offer)
        evidence = _word_set(blob)
        if not evidence:
            return 0.0
        compact = _norm(blob)
        title = _word_set(self.world.title_of(offer))
        weights = self._rarity()
        score = 0.0
        for cue in cues:
            weight = 0.6 + weights.get(cue, 0.4)
            if cue in title:
                score += 1.25 * weight
            elif cue in evidence:
                score += weight
            elif len(cue) > 5 and cue in compact:
                score += 0.8 * weight
            elif len(cue) > 5 and any(
                len(word) > 5
                and (word.startswith(cue[:5]) or cue.startswith(word[:5]))
                for word in evidence
            ):
                score += 0.35 * weight
        # Adjacent cue pairs capture multi-word soft requirements without a
        # product-suite synonym table.
        for left, right in zip(cues, cues[1:]):
            phrase = f"{left} {right}"
            if phrase in compact:
                score += 1.4
        # Studied facts are stronger evidence than a title echo alone.
        if self.world.listing(offer).studied and score > 0:
            score += 0.5
        # Quality-seeking shoppers: among cue-bearing options, dearer in-budget
        # items often carry more of the asked-for qualities.
        if self._quality_bias() and offer.price is not None and score > 0:
            budget = self.brief.budget or offer.price
            if budget > 0:
                score += 0.85 * (offer.price / budget)
        # Penalise class-mismatched bargains even when they echo a cue word.
        if score > 0 and not self._right_class(offer):
            score *= 0.25
        return score

    def _quality_bias(self) -> bool:
        voice = " ".join([_customer_voice(self.request)] + self.transcript[-2:])
        return _wants_quality(voice)

    def _harden_brief(self) -> None:
        """Fill gaps the model left using only the public goal text."""

        spoken = " ".join([_customer_voice(self.request)] + self.transcript)
        # Deferred firm detail: the shopper said they will reveal it if asked.
        # Do not reopen the gap once they have answered (or already stated a
        # measurement in the conversation).
        answered = self.asked_missing and len(self.transcript) > self.heard_when_asked
        already_specified = any(_extract_specs(turn) for turn in self.transcript)
        if (
            _ASK_RE.search(_customer_voice(self.request))
            and not self.brief.missing
            and not answered
            and not already_specified
        ):
            self.brief.missing = (
                "the firm specification you said you would share if I asked"
            )
        # Measurements written in the request become firm constraints when the
        # model omitted them — field names stay the customer's own words/units.
        have = {
            (_norm(item["field"]), _norm(_text(item["value"])))
            for item in self.brief.must
        }
        for item in _extract_specs(spoken):
            key = (_norm(item["field"]), _norm(item["value"]))
            if key in have:
                continue
            self.brief.must.append(item)
            have.add(key)
        # Customer-stated max/min phrases stay authoritative field names.
        hints = _priority_hints(spoken)
        if hints:
            have_pri = {_norm(item["field"]) for item in self.brief.priorities}
            if not self.brief.priorities:
                self.brief.priorities = hints[:3]
            else:
                for hint in hints:
                    if _norm(hint["field"]) not in have_pri:
                        self.brief.priorities.append(hint)
                        have_pri.add(_norm(hint["field"]))
        # Quoted / match spans and category spans are high-signal search seeds.
        quotes = list(_quoted_in(spoken)) + _match_spans(spoken)
        if quotes and not self.brief.queries:
            self.brief.queries = quotes[:_MAX_SEARCH_PHRASES]
        if not self.brief.category:
            cats = _category_spans(spoken)
            if cats:
                self.brief.category = cats[0]

    def _absorb_spoken_constraints(self, said: str) -> None:
        """Merge a late shopper answer into the firm brief when it carries specs."""

        specs = _extract_specs(said)
        if not specs and not self.brief.missing:
            return
        have = {
            (_norm(item["field"]), _norm(_text(item["value"])))
            for item in self.brief.must
        }
        changed = False
        for item in specs:
            key = (_norm(item["field"]), _norm(item["value"]))
            if key in have:
                continue
            self.brief.must.append(item)
            have.add(key)
            changed = True
        if specs or (self.brief.missing and len(said.split()) >= 2):
            self.brief.missing = ""
            changed = True
        if changed:
            self._drop_target()
            self._pref_at = -1

    def _reread_brief(self) -> None:
        """The shopper may narrow the brief mid-episode; re-read it in full."""

        before = self.brief.snapshot()
        self._read_brief()
        if self.brief.snapshot() != before:
            self._drop_target()
        self._seed_soft_prefs()

    def _check_budget(self) -> None:
        """Trust a stated budget only if that figure really appears in the text."""

        # Prices are only recognised next to a currency the assistant knows, so
        # the shop's own currency stands in until the customer names one. A
        # limit read straight from the request is the last line of defence when
        # nothing else about the request was understood.
        currency = self.brief.currency or self._shop_currency()
        stated: list[float] = []
        for source in [self.request] + self.transcript:
            stated.extend(_money_in(source, currency))
        if self.brief.budget is not None and stated:
            if not any(abs(self.brief.budget - a) <= _PRICE_EPSILON for a in stated):
                self.brief.budget = None
        if self.brief.budget is None and stated:
            # With no usable reading, the tightest figure the shopper actually
            # wrote is the safe cap: overspending cannot be undone, whereas an
            # over-tight cap is relaxed again if it leaves nothing to buy.
            self.brief.tighten(min(stated))

    def _shop_currency(self) -> str:
        """Whatever currency the shop prices things in, where it is consistent."""

        seen = {
            offer.currency for offer in self.world.offers.values() if offer.currency
        }
        return seen.pop() if len(seen) == 1 else ""

    def _drop_target(self) -> None:
        self.target = None
        self.staged_key = ""
        self.stage = _VERIFY

    # -- observation intake ------------------------------------------------- #

    def _absorb(
        self, batch: Sequence[Action], result: dict[str, Any], note: str
    ) -> None:
        self.world.clock = self.link.turn
        studying = {self.tools.tool("detail"), self.tools.tool("compare")} - {None}
        reserving = self.tools.tool("cart_add")
        committing = self.tools.tool("commit")
        steps = [entry.get("observation") for entry in _rows(result.get("calls"))]
        if not steps and isinstance(result.get("observation"), dict):
            steps = [result["observation"]]
        for action, step in zip(batch, steps):
            if not isinstance(step, dict):
                continue
            nested = step.get("observation")
            nested_err = (
                isinstance(nested, dict)
                and bool(nested.get("error"))
            )
            refused = bool(step.get("error")) or nested_err
            self.world.ingest(
                nested if not nested_err else None,
                studied=action.name in studying and not refused,
            )
            if action.name in studying and not refused:
                # Only mark listings actually opened; a failed view must be
                # retried or replaced with search rather than treated as done.
                self.opened.update(
                    product_id
                    for product_id in _opened_ids(action)
                    if product_id
                )
            if action.name == committing:
                self._settle_order(refused)
            elif refused:
                self._handle_refusal(action)
            elif action.name == reserving and self.staged_key:
                self.reserved.add(self.staged_key)
        self._hear_shopper(result.get("user_message"))
        self._check_budget()
        self._react_to_market()
        if self.rethink:
            # Fresh readings landed. Re-pick before the first cart add, but once
            # something is reserved the public commitment stands — only a lost
            # (unbuyable) item may be replaced. After a stockout replacement,
            # re-select from the live shortlist with rebound priorities.
            self.rethink = False
            if self.lost is not None:
                self._replace(self.lost)
                self.lost = None
            elif not self.reserved:
                self._select()
            elif self.target is not None and not self._viable(self.target):
                self.lost = self.target
                if self.target.key in self.reserved:
                    self.pending_drops.append(self.target)
                self.reserved.discard(self.target.key)
                self._replace(self.target)
                self.lost = None
        self._advance_stage()
        self.dialogue.append(self._entry(batch, result, note))

    def _hear_shopper(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return
        said = content.strip()
        self.transcript.append(said)
        if _expects_reply(said) or len(self.transcript) == 1:
            self.unanswered.append(said)
        self._absorb_spoken_constraints(said)
        self._reread_brief()

    def _settle_order(self, refused: bool) -> None:
        """A refused purchase is a retryable state error, not the end of the run."""

        if not refused:
            return
        self.ordered = False
        self.reserved.discard(self.staged_key)
        self._set_stage(_VERIFY)
        self.stage_tries = 0

    def _handle_refusal(self, action: Action) -> None:
        """A refused action against the target means it is not buyable."""

        target = self.target
        if target is None:
            return
        values = {_text(value) for value in action.args.values()}
        if target.product_id in values or target.sku in values:
            self.rejected.add(target.key)
            self._replace(target)

    def _react_to_market(self) -> None:
        """Start the choice over whenever the shop moves under us.

        A price change or a sold-out sign does not only affect the listing it
        lands on — it changes which of the options is now the best one. Looking
        again only at the item already in hand would preserve a decision that
        was right a minute ago, so the whole shortlist goes back on the table.
        """

        moved = set(self.world.moved)
        self.world.moved = []
        if not moved:
            return
        self.shaken = True
        target = self.target
        if target is None:
            self.rethink = True
            return
        if target.key in moved and not self._viable(target):
            # Nothing is struck off for being sold out. Shops restock, and an
            # item that comes back is worth having again; it simply stops being
            # buyable for as long as the sign is up — which is why the one that
            # got away is remembered rather than discarded.
            self.lost = target
            if target.key in self.reserved:
                self.pending_drops.append(target)
            self.reserved.discard(target.key)
            self._replace(target)
            return
        # Still buyable after a price move: before the first cart, re-rank so
        # the post-event frontier wins. After a cart add, the commitment stands
        # unless the reserved item itself became unbuyable (handled above).
        if not self.reserved:
            self.rethink = True

    def _flush_drops(self) -> list[Action]:
        """Remove reserved items the environment still holds after a stockout."""

        drop = self.tools.tool("cart_drop")
        if not drop or not self.pending_drops:
            self.pending_drops = []
            return []
        batch: list[Action] = []
        seen: set[str] = set()
        for offer in self.pending_drops:
            if offer.key in seen:
                continue
            seen.add(offer.key)
            batch.append(Action(drop, self._variant_args(drop, offer)))
        self.pending_drops = []
        return batch

    def _replace(self, lost: Offer) -> None:
        """Relax as little as possible: same listing first, then nearest match."""

        self._drop_target()
        pool = [offer for offer in self._candidates() if self._satisfies(offer)]
        if not pool:
            pool = self._candidates()
        if not pool:
            return
        # Priority / named shortlists: re-bind magnitudes then pick by rank
        # among whatever is still buyable after the market move.
        if self.brief.priorities:
            self._rebind_priorities = True
            self._bind_priority_magnitudes(pool)
            self.target = min(pool, key=self._rank)
            self.shaken = True
            self.vetted = False
            return
        lost_words = _word_set(self.world.title_of(lost))
        cues = self._pref_tokens()
        soft_pool = (
            [offer for offer in pool if self._soft_score(offer) > 0]
            if cues
            else []
        )
        if soft_pool:
            pool = soft_pool
        # Same colour / model cues often live in the option label.
        lost_opts = _word_set(lost.options)

        def distance(offer: Offer) -> tuple:
            if offer.product_id == lost.product_id:
                step = 0
            elif (
                _jaccard(lost_words, _word_set(self.world.title_of(offer)))
                >= _TWIN_SIMILARITY
            ):
                step = 1
            elif lost_opts and (_jaccard(lost_opts, _word_set(offer.options)) >= 0.5):
                step = 1
            else:
                step = 2
            # Model codes the shopper named outrank soft cue noise after a stockout.
            code_hit = 0
            codes = set(_model_codes(self.request))
            if codes:
                blob = _norm(self.world.evidence(offer))
                code_hit = 0 if any(code in blob for code in codes) else 1
            return (
                code_hit,
                step,
                -self._soft_score(offer),
                -self._keyword_score(offer),
            ) + self._rank(offer)

        self.target = min(pool, key=distance)
        self.shaken = True
        self.vetted = False

    def _advance_stage(self) -> None:
        target = self.target
        if target is None or self.staged_key != target.key:
            return
        if self.stage == _VERIFY and target.read_at >= self.world.clock:
            self._set_stage(_RESERVE_ITEM)
        elif self.stage == _RESERVE_ITEM and target.key in self.reserved:
            self._set_stage(_RECHECK)
        elif self.stage == _RECHECK and target.read_at >= self.world.clock:
            self._set_stage(_ORDER)

    def _entry(
        self, batch: Sequence[Action], result: dict[str, Any], note: str
    ) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self._narrate(note),
            "tool_calls": [
                {
                    "id": f"{self.link.turn}-{index}",
                    "type": "function",
                    "function": {
                        "name": action.name,
                        "arguments": json.dumps(action.args, default=str),
                    },
                }
                for index, action in enumerate(batch, start=1)
            ],
            "environment_result": result,
        }

    def _narrate(self, note: str) -> str:
        parts = [note or "acting on the environment"]
        if self.brief.budget is not None:
            parts.append(
                f"budget cap {_number_text(self.brief.budget)} {self.brief.currency}".rstrip()
            )
        leader = self._leader()
        if leader is not None:
            parts.append(
                f"in front {self.world.title_of(leader)[:120]} ({leader.key}) "
                f"at {leader.price} {leader.currency}".rstrip()
            )
        if self.note:
            parts.append(self.note)
        return "; ".join(part for part in parts if part)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Solve one generated-environment shopping session.

    Failures are absorbed rather than raised: a partial trajectory that still
    reaches a purchase is worth more than an abandoned episode.
    """

    proxy = ProxyClient(timeout=90, max_retries=2)
    shopper = Shopper(problem_data, proxy)
    try:
        return shopper.run()
    except Exception as error:  # noqa: BLE001 - never abandon a live session
        shopper.dialogue.append(
            {
                "role": "assistant",
                "content": f"recovered from an internal error: {type(error).__name__}",
                "tool_calls": [],
            }
        )
        return shopper.dialogue


__all__ = ["agent_main"]
