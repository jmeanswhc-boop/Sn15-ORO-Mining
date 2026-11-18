from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from typing import Any, Iterable, Sequence

from src.agent.proxy_client import ProxyClient

_LADDER = (
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-chat-v3.1",
    "openai/gpt-oss-120b",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-chat-v3-0324",
    "z-ai/glm-5.1",
    "google/gemma-4-31b-it",
    "qwen/qwen3.6-27b",
    "xiaomi/mimo-v2-flash",
    "qwen/qwen3-32b",
)
_ROLE_MODELS = {
    "act": ("deepseek/deepseek-v3.2", "deepseek/deepseek-chat-v3.1", "openai/gpt-oss-120b"),
    "judge": ("deepseek/deepseek-v3.2", "deepseek/deepseek-chat-v3.1", "openai/gpt-oss-120b"),
    "reason": ("deepseek/deepseek-v3.2", "deepseek/deepseek-chat-v3.1", "openai/gpt-oss-120b"),
}
_INFERENCE = ProxyClient(timeout=30, max_retries=1)
_SESSION = ProxyClient(timeout=70, max_retries=1)

_WALL = 300.0
_EPISODE_BUDGET = _WALL - 35.0
_HARD_BUDGET = _WALL - 12.0
_ASK_SHARE = 0.55
_MIN_CALL_SECS = 8.0
_MAX_CALL_SECS = 45.0
_FINISH_RESERVE_SECS = 35.0
_INSTANT_REJECT_SECS = 0.5
_FAST_FAIL_SECS = 2.0
_FAST_RETRIES = 2
_BACKOFF_BASE = 1.5
_BACKOFF_MAX = 8.0
_REJECT_BACKOFF_BASE = 3.0
_REJECT_BACKOFF_MAX = 12.0
_ATTEMPT_CAP = 10
_ENV_ATTEMPTS = 3
_ENV_BACKOFF_BASE = 1.0
_ENV_BACKOFF_MAX = 6.0
_TURN_PROBES = 2
_STALL_LIMIT = 5
_CLOSE_ACTIONS = 4
_ENV_CALL_FLOOR = 65.0

_MAX_SEARCHES = 3
_FINALIZE_RESERVE = 6
_DESPERATE_AT = 3
_IDLE_LIMIT = 4
_MAX_REPLIES = 3
_MAX_QUESTIONS = 2
_MAX_FACT_READS = 3
_AUTO_READS = 8
_VERIFY_EXTRA = 3
_COMMIT_BY = 12
_SPEC_PROBES = 3
_REPLACEMENT_WORDS = 8
_COMPARE_CAP = 5
_FALLBACK_PRODUCTS = 2
_COMMIT_WAIT_SHARE = 0.45
_CLAIM_LIMIT = 6
_SHORTLIST = 8
_STATED_CAP = 10
_MAX_ASK_SECS = 70.0
_RETURN_CHARS = 56000
_HISTORY = 40
_OBS_CHARS = 6000
_DIALOGUE_CHARS = 48000
_READ_TOOLS = ("view", "compare", "inspect_stock", "inspect_cart", "search", "filter")
_WRITE_TOOLS = ("add_to_cart", "remove_from_cart", "place_test_order")

_FUNCTION_WORDS = frozenset(
    """a an and are as at be been being but by for from in into is it its of on onto
    or over per than that the their then there these they this those to toward under
    until up upon was were what when where which while who with within without""".split()
)
_NEGATIVE = (
    "no",
    "not",
    "none",
    "without",
    "excluding",
    "excludes",
    "lacking",
    "missing",
    "unavailable",
    "unsupported",
    "false",
    "replacement",
    "compatible with",
)
_UNIT_ALIASES = (
    ("inch", "in"),
    ("inches", "in"),
    ("hertz", "hz"),
    ("pieces", "pcs"),
    ("piece", "pcs"),
    ("watt", "w"),
    ("watts", "w"),
    ("liter", "l"),
    ("liters", "l"),
    ("litre", "l"),
    ("litres", "l"),
)
_AMOUNT = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SYMBOLS = "".join(
    chr(point)
    for point in range(sys.maxunicode + 1)
    if unicodedata.category(chr(point)) == "Sc"
)
_CURRENCY = re.compile(r"\b[A-Z]{3}\b|[" + re.escape(_SYMBOLS) + r"]")
_UNITS_LOOSE = (
    "inch", "inches", "cm", "mm", "gb", "tb", "mb", "ghz", "mhz", "mah", "kg", "ml",
    "hertz", "pieces", "piece", "pcs", "watt", "watts", "liter", "liters", "litre",
    "litres", "count",
)
_UNITS_TIGHT = ("in", "hz", "mp", "kw", "w", "kg", "ml", "l", "p", "k", "x", "ct")
_SPEC_LOOSE = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(" + "|".join(sorted(_UNITS_LOOSE, key=len, reverse=True)) + r")\b",
    re.I,
)
_SPEC_LIST = re.compile(
    r"((?:\d+(?:\.\d+)?\s*[;,/]\s*)+\d+(?:\.\d+)?)\s*-?\s*("
    + "|".join(sorted(_UNITS_LOOSE, key=len, reverse=True))
    + r")\b",
    re.I,
)
_SPEC_TIGHT = re.compile(
    r"(\d+(?:\.\d+)?)-?(" + "|".join(sorted(_UNITS_TIGHT, key=len, reverse=True)) + r")\b",
    re.I,
)
_SPLIT_DIGIT_ALPHA = re.compile(r"(\d)([a-z])", re.I)
_SPLIT_ALPHA_DIGIT = re.compile(r"([a-z])(\d)", re.I)
_PUNCT = re.compile(r"[^\w.]+")
_WHITESPACE = re.compile(r"\s+")
_FENCE = re.compile(r"^```[a-zA-Z]*|```$", re.M)
_IDENT = re.compile(r"\b(?=[a-z0-9-]*\d)[a-z0-9][a-z0-9-]{2,}\b", re.I)
_QUOTED = re.compile(r"[\"“”]([^\"“”]{2,80})[\"“”]")
_FIELD_NAME = re.compile(r"[a-z][a-z0-9_]{1,39}")
_ACRONYMS = re.compile(r"(?:\s+[A-Za-z]*[A-Z][A-Za-z]*){1,2}")
_MEASURE = re.compile(r"\d+(?:\.\d+)?-?[a-z]+", re.I)
_STATED = re.compile(r"\d+(?:\.\d+)?\s*\+?\s*[a-z]{1,6}\b", re.I)
_DIMENSIONS = re.compile(r"\b(\d{3,4})\s*[x*×]\s*(\d{3,4})\b")
_ASKS = re.compile(r"\?|？|\b(let me know|tell me|confirm|which one|what do you|can you)\b", re.I)
_CANDIDATE_KEY = re.compile(r"\b(\d{6,})::([A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)")
_CEILING = (
    "up to", "under", "below", "at most", "no more than", "not more than", "max",
    "maximum", "budget", "spend", "afford", "or less", "or under", "at or",
)

_FAMILIES = (
    "intent_decomposition",
    "retrieval_recall",
    "constraint_satisfaction",
    "preference_reasoning",
    "ranking",
    "recovery",
    "justification",
)
_PROCEDURE = {
    "intent_decomposition": (
        "Procedure: separate what is firm from what is merely liked. Treat the category, any "
        "stated measurement, the budget and stock as firm; treat a named brand as a preference "
        "to satisfy when a fully compliant item of that brand exists and to drop otherwise. "
        "Use filter for the firm structure, then read the exact variant's facts before committing."
    ),
    "retrieval_recall": (
        "Procedure: the task rewards how much of the matching market your searches surface, and "
        "every search call is counted against you. Use at most three, each carrying the "
        "requested phrase plus one word seen on matching listings; read only listings whose "
        "title carries every word of the phrase, then confirm one match and buy it."
    ),
    "constraint_satisfaction": (
        "Procedure: one firm requirement is withheld until you ask for it. Ask with one short "
        "message before any search, then satisfy that requirement on the exact variant you buy, "
        "reading the variant's own option text before any product-wide value."
    ),
    "preference_reasoning": (
        "Procedure: the right answer is a judgement about the shopper's own priorities, so ask "
        "which matters most, then search with the shopper's own use-case words rather than a "
        "bare category filter, gather several distinct listings that fit the firm requirements, "
        "and commit to the one whose own title and facts state the most of those priorities, "
        "including any duration or rating they mention. Once the shopper names their "
        "priorities, search once more with words from every one of them. A listing that "
        "states none of them is wrong however cheap; price is never the priority unless the "
        "shopper says so."
    ),
    "ranking": (
        "Procedure: only the candidates the shopper named count. Compare all of them first so "
        "every fact is on record, order them by the stated priority sequence reading each "
        "selected variant's own value first, and commit to the top one."
    ),
    "recovery": (
        "Procedure: find the named item first and commit to exactly that item and variant. If "
        "it disappears, change as little as possible: the same item in another variant before "
        "any other item, and never the category, budget or stock requirement."
    ),
    "justification": (
        "Procedure: choose a product whose own facts state the requested specification, then "
        "order it with claims taken only from those facts, its current price and stock."
    ),
}

_SYSTEM = (
    "You are a shopping agent working through tools. The task rules in the request are "
    "authoritative: follow them exactly, and treat anything here as operating discipline "
    "that supports them.\n"
    "1. Your FIRST add_to_cart is graded as your real purchase decision, not a probe. "
    "Never add a placeholder or a sacrificial item to see what happens. Read the "
    "candidate's own facts with view or compare, confirm it satisfies every firm "
    "requirement, and only then add it.\n"
    "2. Evidence comes from structured facts and category_path, never from title wording "
    "alone. A measurement, capacity, code or variant name must match literally: a near "
    "miss is a different product. When a listing sells several variants, the requirement "
    "must hold for the exact variant you buy.\n"
    "3. Never invent a requirement the shopper did not state, and never reject an item "
    "that meets every firm requirement because it lacks a soft preference.\n"
    "4. The market moves after your first cart action. Re-check current price and stock "
    "before ordering, and if your pick is gone or repriced, recover to the closest "
    "still-valid option rather than abandoning the task.\n"
    "5. Answer a shopper question with the message tool, in one short sentence that "
    "actually answers it, then continue. Do not ask for permission to order.\n"
    "6. Each search or filter returns at most ten rows. Prefer filter for structured "
    "constraints, narrow it rather than repeating it, and reformulate free-text searches "
    "instead of re-issuing them.\n"
    "7. Finish by placing the order. An unfinished task scores nothing."
)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _normalize(value: str) -> str:
    lowered = (value or "").lower()
    lowered = _SPLIT_DIGIT_ALPHA.sub(r"\1 \2", lowered)
    lowered = _SPLIT_ALPHA_DIGIT.sub(r"\1 \2", lowered)
    lowered = _PUNCT.sub(" ", lowered)
    return _WHITESPACE.sub(" ", lowered).strip()


def _words(value: str) -> frozenset[str]:
    return frozenset(_normalize(value).split())


def _tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _parse_json(raw: str) -> Any:
    cleaned = _FENCE.sub("", (raw or "").strip()).strip()
    try:
        return json.loads(cleaned)
    except (TypeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    for start in sorted(p for p in (cleaned.find("{"), cleaned.find("[")) if p >= 0):
        try:
            return decoder.raw_decode(cleaned[start:])[0]
        except ValueError:
            continue
    return None


def _spec_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    found = _SPEC_LOOSE.findall(value or "") + _SPEC_TIGHT.findall(value or "")
    for group, unit in _SPEC_LIST.findall(value or ""):
        found.extend((part.strip(), unit) for part in re.split(r"[;,/]", group))
    for amount, unit in found:
        unit = unit.lower()
        for source, target in _UNIT_ALIASES:
            if unit == source:
                unit = target
        number = _number(amount)
        if number is None:
            continue
        rendered = f"{number:g}"
        tokens.add(f"{rendered} {unit}")
    return tokens


def _spec_phrases(value: str) -> set[str]:
    phrases: set[str] = set()
    for match in _SPEC_LOOSE.finditer(value or ""):
        phrases.add(match.group(0).strip())
    for match in _SPEC_TIGHT.finditer(value or ""):
        phrases.add(match.group(0).strip())
    return phrases


def _spec_aliases(value: str) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for pattern in (_SPEC_LOOSE, _SPEC_TIGHT):
        for match in pattern.finditer(value or ""):
            tokens = _spec_tokens(match.group(0))
            if len(tokens) != 1:
                continue
            tail = _ACRONYMS.match(value[match.end() :])
            if tail:
                phrase = _normalize(tail.group(0))
                if phrase:
                    aliases.setdefault(next(iter(tokens)), set()).add(phrase)
    return aliases


def _money_caps(value: str) -> list[float]:
    caps: list[float] = []
    lowered = (value or "").lower()
    for match in _AMOUNT.finditer(value or ""):
        amount = _number(match.group(0).replace(",", ""))
        if amount is None or amount <= 0:
            continue
        if value[match.end() : match.end() + 1].isalnum():
            continue
        before = lowered[max(0, match.start() - 26) : match.start()]
        after = value[match.end() : match.end() + 6]
        near = bool(
            _CURRENCY.search(after)
            or _CURRENCY.search(value[max(0, match.start() - 3) : match.start()])
        )
        if near and any(word in before for word in _CEILING):
            caps.append(amount)
    return caps


def _negated(haystack: str, needle: str) -> bool:
    lowered = (haystack or "").lower()
    position = lowered.find((needle or "").lower())
    if position < 0:
        return False
    window = lowered[max(0, position - 40) : position]
    return any(marker in window for marker in _NEGATIVE)


def _contains(needle: str, haystack: str) -> bool:
    if not needle:
        return False
    target = _normalize(haystack)
    probe = _normalize(needle)
    if not probe:
        return False
    if f" {probe} " in f" {target} ":
        return True
    probe_words = frozenset(probe.split()) - _FUNCTION_WORDS
    if not probe_words:
        return False
    return probe_words <= frozenset(target.split())


def _ground_spec(
    spec: str, facts: dict[str, Any], spec_field: str | None, rules: _Constraints
) -> tuple[str, str] | None:
    amount, unit = spec.split()
    target = _number(amount)
    bare: list[tuple[str, str]] = []
    for key, value in facts.items():
        rendered = str(value).strip()
        if not rendered:
            continue
        if spec in _spec_tokens(rendered):
            return str(key)[:60], rendered[:60]
        if unit == "p" and amount.isdigit():
            pairs = [(int(a), int(b)) for a, b in _DIMENSIONS.findall(rendered)]
            if any(int(amount) in pair for pair in pairs):
                return str(key)[:60], rendered[:60]
        if rules.aliased(spec, rendered):
            return str(key)[:60], rendered[:60]
        if target is not None and _number(rendered) == target:
            bare.append((str(key)[:60], rendered[:60]))
    if spec_field:
        named = [pair for pair in bare if _contains(spec_field.replace("_", " "), pair[0])]
        return named[0] if named else None
    return bare[0] if len(bare) == 1 else None


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    parsed = _parse_json(raw) if isinstance(raw, str) else raw
    return parsed if isinstance(parsed, dict) else {}


def _key(product_id: Any, sku: Any) -> tuple[str, str]:
    return (str(product_id or ""), str(sku or ""))


def _unwrap(call_result: Any) -> tuple[dict[str, Any] | None, bool]:
    step = call_result.get("observation") if isinstance(call_result, dict) else None
    done = bool(
        (isinstance(step, dict) and step.get("done") is True)
        or (isinstance(call_result, dict) and call_result.get("done") is True)
    )
    if isinstance(step, dict) and isinstance(step.get("observation"), dict):
        return step["observation"], done
    return (step if isinstance(step, dict) else None), done


class _Clock:
    def __init__(self) -> None:
        self.started = time.monotonic()

    def left(self) -> float:
        return _EPISODE_BUDGET - (time.monotonic() - self.started)

    def hard(self) -> float:
        return _HARD_BUDGET - (time.monotonic() - self.started)

    def spent(self) -> float:
        return time.monotonic() - self.started


class _Constraints:
    def __init__(self, goal: str) -> None:
        caps = _money_caps(goal.split("\n\n")[0])
        self.cap: float | None = min(caps) if caps else None
        head = goal.split("Task rules:")[0].split("\n\n")[0]
        self.head = head
        self.specs = _spec_tokens(head)
        self.phrases = _spec_phrases(head)
        self.aliases = _spec_aliases(head)
        spec_parts = {part for token in self.specs for part in token.split()}
        money = {int(value) for value in _money_caps(head)}
        amount_spans = [match.span() for match in _AMOUNT.finditer(head)]
        listed = {part.lower() for pair in _CANDIDATE_KEY.findall(head) for part in pair}
        self.idents = {
            match.group(0).lower()
            for match in _IDENT.finditer(head)
            if match.group(0).lower() not in spec_parts
            and match.group(0).lower() not in listed
            and not _MEASURE.fullmatch(match.group(0))
            and not (match.group(0).isdigit() and int(match.group(0)) in money)
            and not any(
                start <= match.start() and match.end() <= end and (start, end) != match.span()
                for start, end in amount_spans
            )
        }
        self.ident_context: list[str] = []
        tokens = re.findall(r"[a-z0-9]+", head.lower())
        for index, token in enumerate(tokens):
            if token in self.idents:
                for neighbour in tokens[max(0, index - 2) : index]:
                    if (
                        neighbour not in _FUNCTION_WORDS
                        and len(neighbour) > 2
                        and neighbour not in self.ident_context
                        and not any(char.isdigit() for char in neighbour)
                    ):
                        self.ident_context.append(neighbour)
        self.wanted = {
            word
            for word in _words(head)
            if len(word) > 3
            and word not in _FUNCTION_WORDS
            and word not in spec_parts
            and word not in self.idents
            and not any(char.isdigit() for char in word)
        }
        self.requires_stock = "in stock" in goal.lower() or "in-stock" in goal.lower()
        self.theme: frozenset[str] = frozenset()
        self.theme_text = ""
        for quoted in _QUOTED.findall(head):
            self.theme |= {word for word in _tokens(quoted) if word not in _FUNCTION_WORDS}
            self.theme_text = self.theme_text or " ".join(quoted.lower().split())
        self.brand: str | None = None
        self.brand_firm = True
        self.phrase = ""
        self.category: str | None = None
        self.family: str | None = None
        self.spec_field: str | None = None
        self.withheld = False

    def tighten(self, message: str) -> None:
        caps = _money_caps(message)
        if caps:
            lowest = min(caps)
            self.cap = lowest if self.cap is None else min(self.cap, lowest)
        self.specs |= _spec_tokens(message)
        self.phrases |= _spec_phrases(message)
        for spec, names in _spec_aliases(message).items():
            self.aliases.setdefault(spec, set()).update(names)

    def adopt(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        brand = _text(payload.get("brand")).strip()
        category = _text(payload.get("category")).strip()
        if brand and len(brand) < 40:
            self.brand = brand
        if category and len(category) < 60:
            self.category = category
        stated = _spec_tokens(self.head)
        fields: list[str] = [_text(payload.get("spec_field"))]
        for spec in payload.get("specs") or []:
            if isinstance(spec, dict):
                phrase = _text(spec.get("phrase") or spec.get("spec") or spec.get("value"))
                if _spec_tokens(phrase) & stated:
                    fields.append(_text(spec.get("spec_field") or spec.get("field")))
                spec = phrase
            self.specs |= _spec_tokens(_text(spec)) & stated
        family = _text(payload.get("family")).strip().lower().replace(" ", "_")
        if family in _FAMILIES:
            self.family = family
        for field in fields:
            field = field.strip().lower().replace(" ", "_")
            if self.specs and _FIELD_NAME.fullmatch(field):
                self.spec_field = field
                break
        self.withheld = payload.get("withheld") is True
        phrase = " ".join(_text(payload.get("phrase")).lower().split())
        if phrase and len(phrase) < 80 and _contains(phrase, self.head.lower()):
            self.phrase = phrase

    def branding(self, row: dict[str, Any], facts: dict[str, Any] | None) -> float:
        if not self.brand:
            return 0.0
        stated = _text(row.get("brand")) or _text((facts or {}).get("brand"))
        if not stated:
            return -4.0 if self.brand_firm else 0.0
        if _contains(self.brand, stated):
            return 2.0
        return -4.0 if self.brand_firm else -0.5

    def classification(self, facts: dict[str, Any] | None) -> int:
        if not self.category or not isinstance(facts, dict):
            return 0
        path = facts.get("category_path")
        if not isinstance(path, list) or not path:
            return 0
        rendered = " ".join(str(part) for part in path)
        if _contains(self.category, rendered):
            return 1
        asked = {word.rstrip("s") for word in _words(self.category) if len(word) > 2}
        shown = {word.rstrip("s") for word in _words(rendered) if len(word) > 2}
        return 0 if asked & shown else -1

    def spec_status(self, evidence: str) -> str:
        if not self.specs or not evidence:
            return "unknown"
        present = _spec_tokens(evidence)
        verdict = "match"
        for spec in self.specs:
            amount, unit = spec.split()
            rival = {token for token in present if token.split()[-1] == unit}
            if len(rival) > 1:
                verdict = "unknown"
                continue
            if spec in present and not _negated(evidence, spec.split()[0]):
                continue
            if rival:
                return "conflict"
            if unit == "p" and amount.isdigit():
                pairs = [(int(a), int(b)) for a, b in _DIMENSIONS.findall(evidence)]
                if any(int(amount) in pair for pair in pairs):
                    continue
                if pairs:
                    return "conflict"
            if self.aliased(spec, evidence):
                continue
            verdict = "unknown"
        return verdict

    def ambiguous(self, facts: dict[str, Any] | None, title: str) -> bool:
        if not self.specs:
            return False
        for spec in self.specs:
            unit = spec.split()[-1]
            stated = 0
            for value in (facts or {}).values():
                found = {
                    token
                    for token in _spec_tokens(_text(value))
                    if token.split()[-1] == unit
                }
                if len(found) > 1:
                    return True
                stated += len(found)
            if not stated:
                titled = {
                    token for token in _spec_tokens(title) if token.split()[-1] == unit
                }
                if len(titled) > 1:
                    return True
        return False

    def aliased(self, spec: str, evidence: str) -> bool:
        words = _words(evidence)
        for alias in self.aliases.get(spec, ()):
            if set(alias.split()) <= words and not _negated(evidence, alias):
                return True
        return False

    def affordable(self, price: float | None) -> bool:
        if price is None:
            return False
        return self.cap is None or price <= self.cap

    def score(
        self,
        row: dict[str, Any],
        facts: dict[str, Any] | None,
        evidence: str = "",
    ) -> float:
        price = _number(row.get("price"))
        if not self.affordable(price):
            return -1.0
        if self.requires_stock and row.get("in_stock") is not True:
            return -1.0
        blob = " ".join(
            [evidence or json.dumps(facts or {}, sort_keys=True), _text(row.get("title"))]
        )
        points = 1.0
        if self.brand:
            points += self.branding(row, facts)
        points += {1: 2.0, 0: -1.0, -1: -4.0}[self.classification(facts)] if self.category else 0.0
        present = _spec_tokens(blob)
        for spec in self.specs:
            if spec in present or self.aliased(spec, blob):
                points += 3.0
            elif facts and spec.split()[-1] in {token.split()[-1] for token in present}:
                points -= 2.0
        for ident in self.idents:
            if _contains(ident, blob):
                points += 4.0
        if self.family == "retrieval_recall" and self.theme:
            points += 3.0 if self.theme <= _tokens(_text(row.get("title"))) else -1.0
        return points


class _Gateway:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock

    @staticmethod
    def _jitter(seed: int) -> float:
        return 0.65 + 0.7 * ((seed % 1009) / 1009.0)

    def _pause(self, attempt: int, seed: int, instant: bool) -> float:
        base = _REJECT_BACKOFF_BASE if instant else _BACKOFF_BASE
        ceiling = _REJECT_BACKOFF_MAX if instant else _BACKOFF_MAX
        return min(ceiling, base * (2.0**attempt)) * self._jitter(seed)

    def _models(self, role: str) -> tuple[str, ...]:
        preferred = _ROLE_MODELS.get(role, ())
        rest = tuple(model for model in _LADDER if model not in preferred)
        return preferred + rest

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        role: str = "act",
        tool_choice: str | None = None,
        share: float | None = None,
    ) -> dict[str, Any] | None:
        seed = sum(len(_text(message.get("content"))) for message in messages) + 7
        started = time.monotonic()
        allowance = share if share is not None else max(
            _MIN_CALL_SECS, min(_MAX_ASK_SECS, self.clock.left() * _ASK_SHARE)
        )
        attempts = 0
        for model in self._models(role):
            retries = 0
            while True:
                left = self.clock.left()
                if left <= _FINISH_RESERVE_SECS or attempts >= _ATTEMPT_CAP:
                    return None
                if time.monotonic() - started >= allowance:
                    return None
                attempts += 1
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 4096,
                }
                if tools:
                    payload["tools"] = tools
                    if tool_choice:
                        payload["tool_choice"] = tool_choice
                room = allowance - (time.monotonic() - started)
                held = _INFERENCE.timeout
                _INFERENCE.timeout = max(
                    _MIN_CALL_SECS,
                    min(_MAX_CALL_SECS, left - _FINISH_RESERVE_SECS, room),
                )
                tick = time.monotonic()
                try:
                    response = _INFERENCE.post(
                        "/inference/chat/completions", json_data=payload
                    )
                except Exception:
                    response = None
                finally:
                    _INFERENCE.timeout = held
                spent = time.monotonic() - tick
                message = None
                if isinstance(response, dict):
                    choices = response.get("choices") or []
                    if choices:
                        message = choices[0].get("message")
                if isinstance(message, dict):
                    return message
                if spent >= _FAST_FAIL_SECS or retries >= _FAST_RETRIES:
                    break
                wait = self._pause(retries, seed, spent < _INSTANT_REJECT_SECS)
                if self.clock.left() - wait <= _FINISH_RESERVE_SECS:
                    return None
                retries += 1
                time.sleep(wait)
        return None


class _Episode:
    def __init__(self, problem_data: dict[str, Any]) -> None:
        environment = problem_data.get("environment") or {}
        self.binding = environment.get("binding") or {}
        policy = environment.get("policy_view") or {}
        self.problem_id = str(
            problem_data.get("problem_id") or problem_data.get("id") or "problem"
        )
        self.goal = _text(policy.get("query"))
        self.tools = [tool for tool in policy.get("tools") or [] if isinstance(tool, dict)]
        self.names = {_text((tool.get("function") or {}).get("name")) for tool in self.tools}
        self.max_steps = max(1, int(policy.get("max_steps") or 1))
        self.max_calls = max(1, int(policy.get("max_calls_per_turn") or 1))
        self.claims_required = self._claims_required()
        self.clock = _Clock()
        self.gateway = _Gateway(self.clock)
        self.rules = _Constraints(self.goal)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": self.goal},
        ]
        self.dialogue: list[dict[str, Any]] = []
        self.cart: list[tuple[str, str]] = []
        self.stock: dict[tuple[str, str], dict[str, Any]] = {}
        self.market: dict[tuple[str, str], dict[str, Any]] = {}
        self.products: dict[str, dict[str, Any]] = {}
        self.verified: set[tuple[str, str]] = set()
        self.forced: list[dict[str, Any]] = []
        self.pending: list[tuple[str, dict[str, Any] | None]] = []
        self.attempted: set[str] = set()
        self.searches = 0
        self.looked_up = False
        self.spec_probes = 0
        self.fresh: list[str] = []
        self.auto_reads = 0
        self.filtered = 0
        self.asked = 0
        self.ranking: list[tuple[str, str]] | None = None
        self.ranked = False
        self.rank_tries = 0
        self.pref_pool: frozenset[tuple[str, str]] = frozenset()
        self.pref_top: tuple[str, str] | None = None
        self.pref_tries = 0
        self.seeded = False
        self.seed_tries = 0
        self.labelled = False
        self.revealed = ""
        self.dead_listings: set[str] = set()
        self.dead_title = ""
        self.replaced = False
        self.reply_terms: list[str] = []
        self.expansions: list[str] = []
        self.step = 0
        self.turn = 0
        self.rejected: set[tuple[str, str]] = set()
        self.notes: list[str] = []
        self.inspected: set[str] = set()
        self.event_target: tuple[str, str] | None = None
        self.event_seen = False
        self.replies = 0
        self.reads = 0
        self.allowed: list[tuple[str, str]] = [
            _key(pid, sku)
            for pid, sku in _CANDIDATE_KEY.findall(self.goal.split("Task rules:")[0])
        ]
        self.ordered = False
        self.done = False
        self.awaiting_reply = False
        self.focus: tuple[str, str] | None = None

    def _claims_required(self) -> bool:
        for tool in self.tools:
            function = tool.get("function") or {}
            if function.get("name") != "place_test_order":
                continue
            if "justification" in ((function.get("parameters") or {}).get("required") or []):
                return True
        return False

    def _facts(self, ref: tuple[str, str]) -> dict[str, Any] | None:
        return self.products.get(ref[0])

    def _variant(self, ref: tuple[str, str]) -> dict[str, Any]:
        for variant in (self._facts(ref) or {}).get("variants") or []:
            if str(variant.get("sku")) == ref[1]:
                return variant
        return {}

    def _evidence(self, ref: tuple[str, str]) -> str:
        facts = self._facts(ref) or {}
        parts = [
            json.dumps(facts.get("facts") or {}, sort_keys=True),
            " ".join(str(part) for part in facts.get("category_path") or []),
            _text(facts.get("title")),
            _text(facts.get("brand")),
            _text(self._variant(ref).get("options")),
        ]
        return " ".join(part for part in parts if part)

    def _spec_evidence(self, ref: tuple[str, str]) -> str:
        facts = self._facts(ref) or {}
        parts = [
            json.dumps(facts.get("facts") or {}, sort_keys=True),
            _text(self._variant(ref).get("options")),
        ]
        return " ".join(part for part in parts if part)

    def _option_vocab(self) -> frozenset[str]:
        vocab: set[str] = set()
        for payload in self.products.values():
            for variant in payload.get("variants") or []:
                vocab |= _words(_text(variant.get("options")))
        for row in self.market.values():
            vocab |= _words(_text(row.get("options")))
        return frozenset(vocab)

    def _option_fit(self, ref: tuple[str, str]) -> float:
        asked = self.rules.wanted & self._option_vocab()
        if not asked:
            return 0.0
        mine = _words(_text(self._variant(ref).get("options")))
        if not mine:
            mine = _words(_text((self.market.get(ref) or {}).get("options")))
        if asked <= mine:
            return 1.0
        return -1.0 if not (asked & mine) else 0.0

    def _rank(self, refs: Iterable[tuple[str, str]]) -> list[tuple[float, tuple[str, str]]]:
        scored: list[tuple[float, tuple[str, str]]] = []
        order = {ref: index for index, ref in enumerate(self._known_refs())}
        for ref in refs:
            row = self.stock.get(ref) or self.market.get(ref)
            if row is None:
                facts = self._facts(ref)
                for variant in (facts or {}).get("variants") or []:
                    if str(variant.get("sku")) == ref[1]:
                        row = dict(variant, title=_text((facts or {}).get("title")))
                        break
            if row is None:
                continue
            if not _text(row.get("title")):
                title = self._title(ref)
                if title:
                    row = dict(row, title=title)
            points = self.rules.score(row, self._facts(ref), self._evidence(ref))
            if points >= 0:
                scored.append((points + self._option_fit(ref), ref))
        scored.sort(key=lambda pair: (-pair[0], order.get(pair[1], len(order)), pair[1]))
        return scored

    def _known_refs(self) -> list[tuple[str, str]]:
        if self.allowed:
            return [ref for ref in self.allowed if ref not in self.rejected]
        refs: list[tuple[str, str]] = []
        for ref in list(self.stock) + list(self.market):
            if all(ref) and ref not in refs:
                refs.append(ref)
        for product_id, payload in self.products.items():
            for variant in payload.get("variants") or []:
                ref = _key(product_id, variant.get("sku"))
                if all(ref) and ref not in refs:
                    refs.append(ref)
        return refs

    def _note(self, line: str) -> None:
        if len(self.notes) < 240:
            self.notes.append(line[:160])

    def _reject(self, ref: tuple[str, str], why: str) -> None:
        if all(ref) and ref not in self.rejected:
            self.rejected.add(ref)
            self.filtered = 0
            self.spec_probes = 0
            self.auto_reads = 0
            self.looked_up = False
            self.dead_title = " ".join(
                part
                for part in (self._title(ref), _text(self._variant(ref).get("options")))
                if part
            )
            listing = _text((self._facts(ref) or {}).get("listing_id"))
            if listing:
                self.dead_listings.add(listing)
            if ref in self.allowed:
                self.ranked = False
                self.ranking = None
            if ref == self.pref_top:
                self.pref_pool = frozenset()
                self.pref_top = None
                if self.revealed:
                    self._search_reply(self.revealed)
            self._note(f"reject {ref[0]}::{ref[1]} {why}")
            drop = {"name": "remove_from_cart", "args": {"product_id": ref[0], "sku": ref[1]}}
            if "remove_from_cart" in self.names and ref in self.cart and drop not in self.forced:
                self.forced.append(drop)
        if self.focus == ref:
            self.focus = None

    def _viable(self, ref: tuple[str, str]) -> bool:
        if ref in self.rejected:
            return False
        state = self.stock.get(ref)
        if not state or state.get("in_stock") is not True:
            return False
        return self.rules.affordable(_number(state.get("price")))

    def _qualified(self) -> set[str]:
        out: set[str] = set()
        floor = 0 if self.rules.specs else 1
        for ref in self._candidates():
            if ref in self.rejected or self._facts(ref) is None:
                continue
            if self._spec_tier(ref) > floor:
                continue
            state = self.stock.get(ref) or self.market.get(ref) or {}
            if state.get("in_stock") is False:
                continue
            if not self.rules.affordable(_number(state.get("price"))):
                continue
            out.add(ref[0])
        return out

    def _commit_ready(self) -> bool:
        if self.cart or self.ordered:
            return True
        if self.allowed or self.rules.idents:
            return True
        if self.step >= self.max_steps * _COMMIT_WAIT_SHARE:
            return True
        if self.clock.left() <= _FINISH_RESERVE_SECS * 2:
            return True
        if len(self._qualified()) >= _FALLBACK_PRODUCTS:
            return True
        return not any(
            self._facts(ref) is None and ref[0] not in self.inspected
            for ref in self._candidates()
        )

    def _spec_tier(self, ref: tuple[str, str]) -> int:
        facts = self._facts(ref)
        if facts is not None and self.rules.classification(facts) < 0 and any(
            self.rules.classification(self._facts(other)) > 0 for other in self._known_refs()
        ):
            return 2
        loose = self.rules.spec_status(self._evidence(ref))
        structured = self.rules.spec_status(self._spec_evidence(ref))
        if "conflict" in (loose, structured):
            return 2
        if facts is not None and self.rules.ambiguous(
            facts.get("facts") if isinstance(facts.get("facts"), dict) else {},
            self._title(ref),
        ):
            return 2
        row = self.stock.get(ref) or self.market.get(ref) or {}
        branded = self.rules.branding(row, facts)
        if branded < 0 and self.rules.brand_firm:
            return 2
        if self.rules.brand and self.rules.brand_firm and facts is not None and branded == 0:
            return 1
        return 0 if structured == "match" else 1

    def _same_source(self, ref: tuple[str, str]) -> bool:
        if self.rules.family != "preference_reasoning":
            return False
        listing = _text((self._facts(ref) or {}).get("listing_id"))
        return bool(listing) and listing in self.dead_listings

    def _candidates(self) -> list[tuple[str, str]]:
        ranked = [
            ref
            for _points, ref in self._rank(self._known_refs())
            if ref not in self.rejected and self._spec_tier(ref) < 2 and not self._same_source(ref)
        ]
        ranked.sort(
            key=lambda ref: (
                self._spec_tier(ref),
                1 if ref in self.verified and not self._viable(ref) else 0,
                0 if self._covers_theme(ref) else 1,
            )
        )
        return ranked

    def _shortlist_note(self) -> str | None:
        if self.rules.family == "preference_reasoning":
            return None
        ranked = self._rank(self._known_refs())[:_SHORTLIST]
        if not ranked:
            return None
        rows = []
        for points, ref in ranked:
            row = self.stock.get(ref) or self.market.get(ref) or {}
            rows.append(
                {
                    "product_id": ref[0],
                    "sku": ref[1],
                    "title": _text(row.get("title"))[:70],
                    "price": row.get("price"),
                    "in_stock": row.get("in_stock"),
                    "fit": round(points, 2),
                    "verified": ref in self.verified,
                }
            )
        return json.dumps(
            {
                "best_candidates_so_far": rows,
                "price_ceiling": self.rules.cap,
                "required_specs": sorted(self.rules.specs),
            }
        )

    def _label_field(self) -> None:
        if self.rules.spec_field or not self.rules.phrases or self.labelled:
            return
        self.labelled = True
        phrase = sorted(self.rules.phrases, key=len)[-1]
        message = self.gateway.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "A shopper requires a product attribute to have a given value. Name "
                        "that attribute with its shortest common label: a single word whenever "
                        "a one-word label is in common use, otherwise two words joined by an "
                        "underscore; never prefix the device part when the single word already "
                        'names the attribute. Return JSON {"field": string}. JSON only.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Product: {self.rules.category or 'item'}. Required value: {phrase}",
                },
            ],
            None,
            role="judge",
            share=min(25.0, self.clock.left() * 0.1),
        )
        parsed = _parse_json(_text((message or {}).get("content")))
        field = _text(parsed.get("field")).strip().lower().replace(" ", "_") if isinstance(parsed, dict) else ""
        if _FIELD_NAME.fullmatch(field):
            self.rules.spec_field = field
            self._note(f"specification labelled {field}")

    def _spec_claims(self, ref: tuple[str, str]) -> list[dict[str, str]]:
        if self.rules.spec_status(self._evidence(ref)) == "conflict":
            return []
        self._label_field()
        facts = (self._facts(ref) or {}).get("facts")
        if not isinstance(facts, dict):
            return []
        out: list[dict[str, str]] = []
        if self.rules.spec_field:
            for spec in sorted(self.rules.specs):
                if not _ground_spec(spec, facts, self.rules.spec_field, self.rules):
                    continue
                phrase = next(
                    (p for p in sorted(self.rules.phrases) if _spec_tokens(p) == {spec}), None
                )
                if phrase:
                    out.append({"field": self.rules.spec_field, "value": phrase[:60]})
            if out:
                return out
        for spec in sorted(self.rules.specs):
            grounded = _ground_spec(spec, facts, self.rules.spec_field, self.rules)
            if not grounded:
                continue
            claim = {"field": grounded[0], "value": grounded[1]}
            if claim not in out:
                out.append(claim)
        return out

    def _claims(self, ref: tuple[str, str], proposed: Any) -> list[dict[str, str]]:
        state = self.stock.get(ref) or {}
        evidence = self._evidence(ref)
        claims: list[dict[str, str]] = []
        seen: set[str] = set()

        def add(claim: dict[str, str] | None) -> None:
            if claim and claim["field"] not in seen and claim["value"]:
                claims.append(claim)
                seen.add(claim["field"])

        price = _number(state.get("price"))
        if price is not None:
            add({"field": "price", "value": f"{price:.2f}"})
        if state.get("in_stock") is True:
            add({"field": "in_stock", "value": "true"})
        for claim in self._spec_claims(ref):
            add(claim)
        facts = (self._facts(ref) or {}).get("facts")
        for claim in proposed if isinstance(proposed, list) else []:
            if not isinstance(claim, dict) or not isinstance(facts, dict):
                continue
            field = _text(claim.get("field")).strip()
            value = _text(claim.get("value")).strip()
            if not field or not value or field in seen:
                continue
            stated = next(
                (
                    str(fact)
                    for key, fact in facts.items()
                    if _normalize(str(key)) == _normalize(field)
                ),
                None,
            )
            if stated and _contains(value, stated) and not _negated(evidence, value):
                add({"field": field, "value": value})
        brand = _text((self._facts(ref) or {}).get("brand")).strip()
        if brand:
            add({"field": "brand", "value": brand[:60]})
        return claims[:_CLAIM_LIMIT]

    def _order_action(self, ref: tuple[str, str], proposed: Any) -> dict[str, Any]:
        args: dict[str, Any] = {"product_id": ref[0], "sku": ref[1]}
        if self.claims_required:
            args["justification"] = {"claims": self._claims(ref, proposed)}
        return {"name": "place_test_order", "args": args}

    def _guard(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        if name not in self.names:
            return None, "that tool is not available on this task"
        if name == "search":
            allowance = _MAX_SEARCHES if self.rules.family == "retrieval_recall" else _MAX_SEARCHES * 2
            if self.searches >= allowance:
                return None, (
                    "the free-text search allowance is spent; use filter, view or compare"
                )
            if self.searches == 0 and not self.rules.theme and self.rules.family == "retrieval_recall":
                self.rules.theme = frozenset(
                    word for word in _tokens(_text(args.get("query"))) if word not in _FUNCTION_WORDS
                )
                self.rules.theme_text = " ".join(_text(args.get("query")).lower().split())
            self.searches += 1
            return {"name": name, "args": args}, ""
        if name == "filter" and self.theme:
            return None, (
                "filter cannot target the requested phrase; search with the phrase, "
                "or read the listings whose title already carries it"
            )
        if name == "compare" and self.theme:
            ids = [_text(product_id) for product_id in args.get("product_ids") or []]
            kept = [product_id for product_id in ids if product_id and self._on_theme(product_id)]
            if len(kept) < len(ids):
                for ref in self._candidates():
                    if len(kept) >= _COMPARE_CAP:
                        break
                    if ref[0] not in kept and self._on_theme(ref[0]) and self._facts(ref) is None:
                        kept.append(ref[0])
                if not kept:
                    return None, (
                        "none of those titles carries every word of the requested phrase; "
                        "read the listings that do"
                    )
                args = {"product_ids": kept}
        if name == "view" and self.theme:
            product_id = _text(args.get("product_id"))
            if product_id and not self._on_theme(product_id) and any(
                self._on_theme(ref[0]) for ref in self._candidates()
            ):
                return None, (
                    "that title does not carry every word of the requested phrase; "
                    "read a listing that does"
                )
        if name == "message":
            content = _text(args.get("content")).strip()
            if not content:
                return None, "message needs content"
            if not self.awaiting_reply:
                if not _ASKS.search(content):
                    return None, "no shopper message is due; use a catalogue tool instead"
                if self.asked >= _MAX_QUESTIONS:
                    return None, "the shopper has answered enough; decide from the evidence"
                self.asked += 1
            return {"name": name, "args": {"content": content}}, ""
        if name == "add_to_cart":
            ref = _key(args.get("product_id"), args.get("sku"))
            if not all(ref):
                return None, "add_to_cart needs both product_id and sku"
            if ref in self.rejected:
                return None, "that variant was already ruled out; choose a different one"
            if self._same_source(ref):
                self._reject(ref, "same source listing as the item the market removed")
                return None, (
                    "that is the same source listing as the one the market just removed; "
                    "recover to a different listing"
                )
            if self.allowed and ref not in self.allowed:
                self._reject(ref, "outside the shopper's named candidate list")
                return None, "the shopper named which candidates to compare; use those"
            if self.cart and ref not in self.cart:
                held = self.cart[-1]
                return None, (
                    f"the cart already holds {held[0]}::{held[1]}; order it, or remove it "
                    "first if it no longer qualifies"
                )
            if not self.cart:
                tier = self._spec_tier(ref)
                if tier == 2:
                    self._reject(ref, "disqualified by its own facts or category")
                    return None, (
                        "that listing's facts or category_path contradict a stated "
                        "requirement; choose one whose own facts confirm it"
                    )
                if tier == 1 and any(self._spec_tier(o) == 0 for o in self._candidates()):
                    return None, (
                        "nothing on that listing confirms the stated specification while "
                        "another candidate does; commit to the confirmed one"
                    )
                if self.theme and not self._covers_theme(ref) and any(
                    self._covers_theme(other) for other in self._candidates()
                ):
                    self._reject(ref, "title does not carry the requested phrase")
                    return None, (
                        "that listing's title does not carry every word of the requested "
                        "phrase while another candidate's does; commit to one that does"
                    )
                if self.allowed:
                    best = self._ranked_top()
                elif self.rules.family == "preference_reasoning":
                    best = self._preference_top()
                    if ref in self.rejected:
                        return None, (
                            "that listing states none of the shopper's priorities; choose "
                            "one whose own title or facts support them"
                        )
                elif self.rules.family == "retrieval_recall":
                    best = None
                else:
                    best = self._best()
                if best is not None and best != ref:
                    mine = next(
                        (points for points, other in self._rank([ref]) if other == ref), -1.0
                    )
                    theirs = next(
                        (points for points, other in self._rank([best]) if other == best), -1.0
                    )
                    judged = best == self.pref_top or bool(self.ranking and best in self.ranking)
                    if theirs > mine or judged:
                        action = self._advance(best)
                        if action is not None:
                            self._note(
                                f"first commit steered to {best[0]}::{best[1]} "
                                f"({theirs:.2f} over {mine:.2f})"
                            )
                            return action, ""
            if not self._commit_ready():
                return None, (
                    "the market moves the moment anything is carted, so a second "
                    "qualified product must be on the record before committing; "
                    "read another candidate first"
                )
            unseen = self._facts(ref) is None and ref[0] not in self.inspected
            if unseen and "view" in self.names and ref not in self.cart:
                self.inspected.add(ref[0])
                return {"name": "view", "args": {"product_id": ref[0]}}, ""
            if ref in self.cart:
                self.focus = ref
                if ref not in self.verified and "inspect_stock" in self.names:
                    return (
                        {"name": "inspect_stock", "args": {"product_id": ref[0], "sku": ref[1]}},
                        "",
                    )
                return None, "that variant is already in the cart; verify it and order"
            self.focus = ref
            return {"name": name, "args": args}, ""
        if name == "view" and "compare" in self.names:
            product_id = _text(args.get("product_id"))
            batch = [product_id] if product_id else []
            for ref in self._candidates():
                if len(batch) >= _COMPARE_CAP:
                    break
                if self._facts(ref) is None and ref[0] not in batch and ref[0] not in self.inspected:
                    batch.append(ref[0])
            if len(batch) > 1:
                self.inspected.update(batch)
                return {"name": "compare", "args": {"product_ids": batch}}, ""
        if name != "place_test_order":
            return {"name": name, "args": args}, ""

        ref = _key(args.get("product_id"), args.get("sku"))
        if not all(ref):
            ref = self.focus or (self.cart[-1] if self.cart else ("", ""))
        if not all(ref):
            return None, "choose a variant and add it to the cart before ordering"
        duty = self._pending_duty(ref)
        if duty is not None:
            return duty, ""
        if ref not in self.cart:
            if self._facts(ref) is None and ref[0] not in self.inspected and "view" in self.names:
                self.inspected.add(ref[0])
                return {"name": "view", "args": {"product_id": ref[0]}}, ""
            return self._guard("add_to_cart", {"product_id": ref[0], "sku": ref[1]})
        if ref not in self.verified and "inspect_stock" in self.names:
            return {"name": "inspect_stock", "args": {"product_id": ref[0], "sku": ref[1]}}, ""
        if self.allowed and ref not in self.allowed:
            self._reject(ref, "outside the shopper's named candidate list")
            return None, "the shopper named the candidates to compare; order one of those"
        tier = self._spec_tier(ref)
        if tier == 2:
            self._reject(ref, "disqualified by its own facts or category")
            return None, (
                "that listing's facts or category_path contradict a stated requirement; "
                "order one whose own facts confirm it"
            )
        if tier == 1 and any(self._spec_tier(other) == 0 for other in self._candidates()):
            return None, (
                "no fact on that listing confirms the stated specification, and another "
                "candidate does confirm it; pick the confirmed one"
            )
        if not self._viable(ref):
            state = self.stock.get(ref) or {}
            self._reject(ref, "unviable at order time")
            return None, (
                f"that variant is not orderable at price={state.get('price')} "
                f"in_stock={state.get('in_stock')} against a ceiling of {self.rules.cap}; "
                "pick another that meets every firm requirement"
            )
        justification = args.get("justification")
        claims = justification.get("claims") if isinstance(justification, dict) else None
        return self._order_action(ref, claims), ""

    def _pending_duty(self, ref: tuple[str, str] | None = None) -> dict[str, Any] | None:
        if self.awaiting_reply and "message" in self.names:
            best = ref if ref and self._viable(ref) else next(iter(self._candidates()), None)
            state = (self.stock.get(best) or self.market.get(best) or {}) if best else {}
            title = _text(state.get("title"))[:60]
            price = _number(state.get("price"))
            if title and price is not None:
                answer = (
                    f"Going with {title} at {price:.2f}, in stock and within"
                    f"{' ' + str(self.rules.cap) if self.rules.cap else ' your limit'}."
                )
            elif price is not None:
                answer = f"Going with a verified in-stock option at {price:.2f}."
            else:
                answer = (
                    "That one is no longer available, so I am checking the remaining "
                    "options against your requirements now."
                )
            return {"name": "message", "args": {"content": answer}}
        if (
            self.event_target
            and not self.event_seen
            and "inspect_stock" in self.names
        ):
            return {
                "name": "inspect_stock",
                "args": {
                    "product_id": self.event_target[0],
                    "sku": self.event_target[1],
                },
            }
        return None

    def _advance(self, ref: tuple[str, str]) -> dict[str, Any] | None:
        if self._facts(ref) is None and ref[0] not in self.inspected and "view" in self.names:
            self.inspected.add(ref[0])
            return {"name": "view", "args": {"product_id": ref[0]}}
        if ref not in self.verified and "inspect_stock" in self.names:
            return {"name": "inspect_stock", "args": {"product_id": ref[0], "sku": ref[1]}}
        if not self._viable(ref):
            return None
        if ref not in self.cart:
            if not self._commit_ready():
                return None
            action, _note = self._guard("add_to_cart", {"product_id": ref[0], "sku": ref[1]})
            return action
        return self._order_action(ref, None)

    def _best(self) -> tuple[str, str] | None:
        return next(iter(self._candidates()), None)

    @property
    def theme(self) -> frozenset[str]:
        return self.rules.theme if self.rules.family == "retrieval_recall" else frozenset()

    def _title(self, ref: tuple[str, str]) -> str:
        facts = self._facts(ref) or {}
        row = self.stock.get(ref) or self.market.get(ref) or {}
        return _text(facts.get("title")) or _text(row.get("title"))

    def _covers_theme(self, ref: tuple[str, str]) -> bool:
        return self.theme <= _tokens(self._title(ref))

    def _on_theme(self, product_id: str) -> bool:
        if not self.theme:
            return True
        titles = [_text((self.products.get(product_id) or {}).get("title"))]
        titles.extend(
            _text(row.get("title")) for ref, row in self.market.items() if ref[0] == product_id
        )
        return any(self.theme <= _tokens(title) for title in titles if title)

    def _theme_search(self) -> dict[str, Any] | None:
        phrase = self.rules.theme_text
        if not phrase or "search" not in self.names or self.cart or self.ordered:
            return None
        if self.searches >= _MAX_SEARCHES or self.max_steps - self.step <= _FINALIZE_RESERVE:
            return None
        if self.searches and not self.theme:
            return None
        query = phrase
        if self.searches:
            covering: dict[str, list[str]] = {}
            for ref, row in self.market.items():
                title = _text(row.get("title"))
                if self.theme <= _tokens(title) and self.rules.affordable(_number(row.get("price"))):
                    covering.setdefault(ref[0], re.findall(r"[a-z0-9]+", title.lower()))
            skip = _tokens(phrase) | set(self.theme) | _FUNCTION_WORDS
            skip |= set(self.expansions)
            rows = [tokens for tokens in covering.values() if not set(tokens) & set(self.expansions)]
            counts: dict[str, int] = {}
            for tokens in rows or list(covering.values()):
                for token in set(tokens):
                    if len(token) > 2 and token.isalpha() and token not in skip:
                        counts[token] = counts.get(token, 0) + 1
            if not counts:
                return None
            term = min(counts, key=lambda token: (-counts[token], token))
            self.expansions.append(term)
            query = f"{phrase} {term}"
        args: dict[str, Any] = {"query": query, "in_stock": True, "k": 10}
        if self.rules.cap is not None:
            args["max_price"] = self.rules.cap
        action = {"name": "search", "args": args}
        signature = json.dumps(action, sort_keys=True)
        if signature in self.attempted:
            return None
        self.attempted.add(signature)
        self.searches += 1
        self._note(f"turn {self.step}: searching the requested phrase: {query}")
        return action

    def _ranked_top(self) -> tuple[str, str] | None:
        if self.ranking is None:
            self._rank_allowed()
        for ref in self.ranking or []:
            if ref in self.rejected:
                continue
            state = self.stock.get(ref)
            if state is not None and not self._viable(ref):
                continue
            return ref
        return self._best()

    def _sheet(self, refs: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
        sheet = []
        for ref in refs:
            facts = self._facts(ref) or {}
            state = self.stock.get(ref) or self.market.get(ref) or self._variant(ref)
            sheet.append(
                {
                    "key": f"{ref[0]}::{ref[1]}",
                    "title": _text(facts.get("title"))[:160],
                    "brand": facts.get("brand"),
                    "category_path": facts.get("category_path"),
                    "facts": facts.get("facts"),
                    "variant_options": self._variant(ref).get("options"),
                    "values_stated_with_a_unit": self._stated_values(ref),
                    "price": state.get("price"),
                    "in_stock": state.get("in_stock"),
                }
            )
        return sheet

    def _judge_order(
        self, system: str, payload: dict[str, Any], share: float, role: str = "judge"
    ) -> tuple[
        list[tuple[str, str]],
        list[tuple[str, str]],
        dict[tuple[str, str], dict[str, str]],
    ]:
        message = self.gateway.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)[:14000]},
            ],
            None,
            role=role,
            share=share,
        )
        parsed = _parse_json(_text((message or {}).get("content")))
        if not isinstance(parsed, dict):
            self._note(f"judge gave no usable reply: {_text((message or {}).get('content'))[:120]}")
        out: list[list[tuple[str, str]]] = []
        for name in ("order", "unsupported"):
            items = parsed.get(name) if isinstance(parsed, dict) else None
            refs: list[tuple[str, str]] = []
            for item in items if isinstance(items, list) else []:
                product_id, _sep, sku = str(item).partition("::")
                ref = _key(product_id, sku)
                if all(ref) and ref not in refs:
                    refs.append(ref)
            out.append(refs)
        stated: dict[tuple[str, str], dict[str, str]] = {}
        reading = parsed.get("stated") if isinstance(parsed, dict) else None
        if isinstance(reading, dict):
            for item, quotes in reading.items():
                product_id, _sep, sku = str(item).partition("::")
                ref = _key(product_id, sku)
                if not all(ref):
                    continue
                pairs = quotes if isinstance(quotes, dict) else {}
                if isinstance(quotes, list):
                    pairs = {_text(name): _text(name) for name in quotes}
                stated[ref] = {
                    _text(name)[:40]: _text(quote)[:80]
                    for name, quote in pairs.items()
                    if _text(name) and _text(quote)
                }
        return out[0], out[1], stated

    def _stated_values(self, ref: tuple[str, str]) -> list[str]:
        out: list[str] = []
        for match in _STATED.finditer(self._evidence(ref)):
            phrase = " ".join(match.group(0).lower().split())
            if phrase not in out:
                out.append(phrase)
            if len(out) >= _STATED_CAP:
                break
        return out

    def _shopper_words(self) -> frozenset[str]:
        words: set[str] = set()
        for entry in self.dialogue:
            if entry.get("role") == "user":
                words |= _words(_text(entry.get("content")))
        return frozenset(word for word in words if len(word) > 3 and word not in _FUNCTION_WORDS)

    def _search_reply(self, content: str) -> None:
        if "search" not in self.names:
            return
        words = list(self.reply_terms)
        if not words:
            words = self._reply_terms(content)
            self.reply_terms = words
        if len(words) < 2:
            return
        category = [self.rules.category] if self.rules.category else []
        for chosen in (category + words[:12], words[:12], category + words[:6], words[:4]):
            query = " ".join(chosen)
            args: dict[str, Any] = {"query": query, "in_stock": True, "k": 10}
            if self.rules.cap is not None:
                args["max_price"] = self.rules.cap
            action = {"name": "search", "args": args}
            signature = json.dumps(action, sort_keys=True)
            if signature in self.attempted:
                continue
            self.attempted.add(signature)
            self.searches += 1
            self.forced.append(action)
            self._note(f"searching with the shopper's own words: {query}")
            return

    def _reply_terms(self, content: str) -> list[str]:
        message = self.gateway.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "A shopper describes what matters to them about a product. List the "
                        "product features they name, as short search terms in the shopper's "
                        'own words, most important first. Return JSON {"terms": [string]}. '
                        "JSON only."
                    ),
                },
                {"role": "user", "content": content[:1500]},
            ],
            None,
            role="judge",
            share=min(20.0, self.clock.left() * 0.1),
        )
        parsed = _parse_json(_text((message or {}).get("content")))
        terms = parsed.get("terms") if isinstance(parsed, dict) else None
        words: list[str] = []
        for term in terms if isinstance(terms, list) else []:
            for word in _words(_text(term)):
                if word not in words and word not in _FUNCTION_WORDS and word.isalpha():
                    words.append(word)
        return words

    def _mentions(self, ref: tuple[str, str]) -> list[str]:
        text = _normalize(self._evidence(ref))
        terms = self.reply_terms or sorted(self._shopper_words() | self.rules.wanted)
        return [term for term in terms if len(term) > 3 and term in text]

    def _preference_top(self) -> tuple[str, str] | None:
        pool = [ref for ref in self._candidates() if self._facts(ref) is not None]
        lead = self.reply_terms[:1]
        pool.sort(
            key=lambda ref: (
                0 if lead and lead[0] in self._mentions(ref) else 1,
                0 if self._stated_values(ref) else 1,
                -len(self._mentions(ref)),
            )
        )
        pool = pool[:12]
        if not pool:
            return None
        fallback = next((ref for ref in pool if ref not in self.rejected), None)
        if len(pool) < 2:
            return fallback
        key = frozenset(pool)
        if key == self.pref_pool or self.pref_tries >= 4:
            return self.pref_top if self.pref_top and self.pref_top not in self.rejected else fallback
        self.pref_tries += 1
        self.pref_pool = key
        shopper = [self.rules.head] + [
            _text(entry.get("content"))
            for entry in self.dialogue
            if entry.get("role") == "user" and _text(entry.get("content"))
        ]
        order, unsupported, stated = self._judge_order(
            (
                "You judge which listing best serves a shopper's stated use case and "
                "priorities. Evidence counts only when the listing's own title, facts or "
                "option text states it; never assume a capability the listing does not "
                "mention. A listing that states none of the shopper's priorities is "
                "unsupported however cheap, and price is not a priority unless the shopper "
                "says so. Rank the supported listings by how many of the priorities they "
                "state: a listing that states the shopper's first-named priority ranks above "
                "every listing that does not, then the more named priorities stated the "
                "better. A listing states a priority when its own text carries a value for "
                "it, and a value stated with a unit counts even when the priority's own "
                "words are absent: read values_stated_with_a_unit and decide which "
                "priority each one measures. A stated value outranks the priority named "
                "without one. Return JSON with keys order (candidate keys, best first), "
                "unsupported (candidate keys) and stated (candidate key -> an object "
                "mapping each priority that listing states to the exact words copied from "
                "that listing which state it). Copy the words; never paraphrase them or "
                "supply words the listing does not carry. JSON only."
            ),
            {
                "shopper": shopper,
                "candidates": [
                    dict(entry, mentions_of_shopper_terms=self._mentions(ref))
                    for entry, ref in zip(self._sheet(pool), pool)
                ],
            },
            min(45.0, self.clock.left() * 0.2),
        )
        for ref in unsupported:
            if ref in key and ref not in order[:1]:
                self._reject(ref, "states none of the shopper's priorities")
        ranked = [
            ref
            for ref in order
            if ref in key
            and ref not in self.rejected
            and any(
                _contains(quote, self._evidence(ref))
                for quote in (stated.get(ref) or {}).values()
            )
        ] or [ref for ref in order if ref in key and ref not in self.rejected]
        self.pref_top = next(iter(ranked), None)
        if self.pref_top is None:
            self.pref_top = fallback
        if self.pref_top:
            self._note(f"preference judge prefers {self.pref_top[0]}::{self.pref_top[1]}")
        return self.pref_top

    def _rank_allowed(self) -> None:
        if self.ranked or not self.allowed:
            return
        live = [ref for ref in self.allowed if ref not in self.rejected]
        unread = [ref for ref in live if self._facts(ref) is None]
        if unread or self.rank_tries >= 3 or not live:
            return
        self.rank_tries += 1
        message = self.gateway.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "The shopper ranks candidates by a stated priority order. First list "
                        "those priorities in order, each with direction max or min. Then, for "
                        "every candidate, read the value of each priority attribute from its "
                        "own evidence: the exact variant's option text first, then the facts, "
                        "then the title; a title listing several models describes the range, "
                        "not this variant, so never take a value from it when options or facts "
                        "carry one. Give each reading as a plain number with its unit, or null "
                        'if the evidence does not state it. Return JSON {"priorities": '
                        '[{"name": string, "direction": "max"|"min"}], "readings": {candidate '
                        "key: {priority name: string|null}}}. JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"request": self.rules.head[:1500], "candidates": self._sheet(live)}
                    )[:14000],
                },
            ],
            None,
            role="judge",
            share=min(45.0, self.clock.left() * 0.2),
        )
        parsed = _parse_json(_text((message or {}).get("content")))
        if not isinstance(parsed, dict):
            self._note(f"ranking judge gave no usable reply: {_text((message or {}).get('content'))[:80]}")
            return
        priorities = [
            (_text(item.get("name")), _text(item.get("direction")).lower() == "min")
            for item in (parsed.get("priorities") or [])
            if isinstance(item, dict) and _text(item.get("name"))
        ]
        readings = parsed.get("readings") if isinstance(parsed.get("readings"), dict) else {}
        keyed = {f"{ref[0]}::{ref[1]}": ref for ref in live}
        scored: list[tuple[list[float], tuple[str, str]]] = []
        for key, ref in keyed.items():
            row = readings.get(key) if isinstance(readings.get(key), dict) else {}
            vector: list[float] = []
            for name, ascending in priorities:
                raw = row.get(name)
                text = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
                match = _AMOUNT.search(text)
                value = _number(match.group(0).replace(",", "")) if match else None
                if value is None and "price" in name.lower():
                    state = self.stock.get(ref) or self.market.get(ref) or self._variant(ref)
                    value = _number(state.get("price"))
                if value is None:
                    vector.append(float("inf"))
                else:
                    vector.append(value if ascending else -value)
            scored.append((vector, ref))
        if priorities and scored:
            scored.sort(key=lambda pair: (pair[0], pair[1]))
            ranking = [ref for _vector, ref in scored]
            self.ranked = True
            self.ranking = ranking
            self._note(
                "priority ranking "
                + " > ".join(f"{r[0]}::{r[1]}" for r in ranking)
                + f" on {[name for name, _ in priorities]}"
            )

    def _finalize(self, desperate: bool = False) -> dict[str, Any] | None:
        duty = self._pending_duty(self.focus)
        if duty is not None:
            return duty
        candidates = self._candidates()
        if (
            not desperate
            and self.rules.specs
            and self.reads < _MAX_FACT_READS
            and self.max_steps - self.step > _FINALIZE_RESERVE
            and not any(self._spec_tier(ref) == 0 for ref in candidates)
        ):
            unread = [
                ref[0]
                for ref in candidates
                if self._facts(ref) is None and ref[0] not in self.inspected
            ]
            batch: list[str] = []
            for product_id in unread:
                if product_id not in batch:
                    batch.append(product_id)
                if len(batch) >= _COMPARE_CAP:
                    break
            if batch:
                self.reads += 1
                self.inspected.update(batch)
                if len(batch) > 1 and "compare" in self.names:
                    self._note(f"comparing {len(batch)} candidates for the specification")
                    return {"name": "compare", "args": {"product_ids": batch}}
                if "view" in self.names:
                    self._note(f"reading facts of {batch[0]} for the specification")
                    return {"name": "view", "args": {"product_id": batch[0]}}
        if not desperate and self.max_steps - self.step > _FINALIZE_RESERVE:
            probe = self._probe()
            if probe is not None:
                return probe
        for ref in candidates:
            if not self._viable(ref) and ref in self.verified:
                continue
            action = self._advance(ref)
            if action is not None:
                return action
        if desperate:
            for ref in reversed(self.cart):
                if self._viable(ref) and self._spec_tier(ref) < 2:
                    return self._order_action(ref, None)
            for ref, row in self.market.items():
                if not all(ref) or ref in self.cart or ref in self.rejected:
                    continue
                if self.allowed and ref not in self.allowed:
                    continue
                if self.rules.score(row, self._facts(ref), self._evidence(ref)) < 0:
                    continue
                if self._spec_tier(ref) < 2:
                    return {"name": "add_to_cart", "args": {"product_id": ref[0], "sku": ref[1]}}
        return None

    def _explore(self) -> dict[str, Any] | None:
        options: list[dict[str, Any]] = []
        if "inspect_cart" in self.names:
            options.append({"name": "inspect_cart", "args": {}})
        for _points, ref in self._rank(self._known_refs()):
            if "inspect_stock" in self.names:
                options.append(
                    {"name": "inspect_stock", "args": {"product_id": ref[0], "sku": ref[1]}}
                )
            if "view" in self.names and ref[0] not in self.products:
                options.append({"name": "view", "args": {"product_id": ref[0]}})
        for option in options:
            signature = json.dumps(option, sort_keys=True)
            if signature not in self.attempted:
                self.attempted.add(signature)
                return option
        return None

    def _absorb(self, name: str, args: dict[str, Any], observation: Any) -> None:
        if not isinstance(observation, dict):
            return
        if name == "view":
            product_id = _text(observation.get("product_id"))
            if product_id:
                self.products[product_id] = observation
        elif name == "compare":
            for entry in observation.get("compared") or []:
                if isinstance(entry, dict) and _text(entry.get("product_id")):
                    self.products[_text(entry["product_id"])] = entry
        elif name in ("search", "filter"):
            self.fresh = []
            for row in observation.get("results") or []:
                if isinstance(row, dict):
                    ref = _key(row.get("product_id"), row.get("sku"))
                    if all(ref):
                        self.market[ref] = row
                        if self.rules.affordable(_number(row.get("price"))) and (
                            row.get("in_stock") is not False
                        ):
                            self.fresh.append(ref[0])
        elif name == "inspect_stock":
            ref = _key(observation.get("product_id"), observation.get("sku"))
            if all(ref):
                self.stock[ref] = observation
                self.verified.add(ref)
        elif name == "add_to_cart":
            self.verified.discard(_key(args.get("product_id"), args.get("sku")))
        if name == "inspect_stock":
            ref = _key(observation.get("product_id"), observation.get("sku"))
            if all(ref) and not self._viable(ref) and ref not in self.rejected:
                self._reject(ref, "verified out of stock or over ceiling")
        if name in ("add_to_cart", "remove_from_cart", "inspect_cart"):
            cart = observation.get("cart")
            if isinstance(cart, list):
                self.cart = [
                    _key(item.get("product_id"), item.get("sku"))
                    for item in cart
                    if isinstance(item, dict)
                ]
                for item in cart:
                    if not isinstance(item, dict):
                        continue
                    ref = _key(item.get("product_id"), item.get("sku"))
                    price = _number(item.get("current_price"))
                    if all(ref) and price is not None:
                        self.stock[ref] = {
                            "product_id": ref[0],
                            "sku": ref[1],
                            "price": price,
                            "in_stock": item.get("in_stock"),
                        }
        event = observation.get("market_event")
        if isinstance(event, dict):
            target = event.get("target") or {}
            ref = _key(target.get("product_id"), target.get("sku"))
            if all(ref):
                self.event_target = ref
                self.event_seen = False
            self.verified.clear()
            self._note(f"market event on {ref[0]}::{ref[1]} {event.get('kind')}")
        if name == "inspect_stock" and self.event_target == _key(
            observation.get("product_id"), observation.get("sku")
        ):
            self.event_seen = True
            self._note("event target re-inspected")
        if name == "place_test_order" and not observation.get("error"):
            self.ordered = True

    def _seed_constraints(self) -> None:
        self.seed_tries += 1
        message = self.gateway.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Read the shopper's request and its Task rules. Return exactly this "
                        'JSON shape: {"brand": string|null, "category": string|null, '
                        '"specs": [string], "spec_field": string|null, "phrase": string|null, '
                        '"family": string, "withheld": boolean}. specs lists short measurable '
                        "product requirement phrases copied from the request (not price, not "
                        "stock); spec_field is the shortest standard snake_case label a "
                        "specification table would give the product attribute those phrases "
                        "constrain; phrase is the free-text wording, copied verbatim, that the "
                        "request says a listing must match, null when the request states "
                        "measurable requirements instead; family is which one of "
                        + ", ".join(_FAMILIES)
                        + " the Task rules describe: retrieval_recall rewards surveying several "
                        "distinct listings that match a phrase, constraint_satisfaction withholds "
                        "one firm requirement until asked, preference_reasoning lets the "
                        "shopper's stated priorities decide, intent_decomposition mixes a firm "
                        "specification with a soft brand, recovery names one item that may "
                        "become unavailable, ranking orders named candidates, justification "
                        "requires claims on the order; withheld is true only if the rules say a "
                        "firm requirement is not stated and will be given when asked. "
                        "Invent nothing. JSON only."
                    ),
                },
                {"role": "user", "content": self.goal[:4000]},
            ],
            None,
            role="reason",
            share=min(40.0, self.clock.left() * 0.15),
        )
        if isinstance(message, dict):
            self.seeded = True
            self.rules.adopt(_parse_json(_text(message.get("content"))))
            self._note(f"seed reply {_text(message.get('content'))[:200]}")
        else:
            self._note("seed judge gave no reply")
        self.rules.family = self._family()
        self.rules.brand_firm = self.rules.family != "intent_decomposition"
        if self.rules.family == "retrieval_recall" and not self.rules.theme and self.rules.phrase:
            self.rules.theme = frozenset(
                word for word in _tokens(self.rules.phrase) if word not in _FUNCTION_WORDS
            )
            self.rules.theme_text = self.rules.phrase
        if self.claims_required:
            self.rules.spec_field = None
            self._label_field()
        procedure = _PROCEDURE.get(self.rules.family or "")
        if procedure:
            self.messages[0] = {"role": "system", "content": _SYSTEM + "\n" + procedure}
            self._note(f"family {self.rules.family}, withheld={self.rules.withheld}")
        if self.rules.family == "constraint_satisfaction" and self.rules.withheld and "message" in self.names:
            self.forced.append(
                {
                    "name": "message",
                    "args": {
                        "content": (
                            "Before I search: which requirement is the firm one you have "
                            "not stated yet, and what value does it need?"
                        )
                    },
                }
            )
        elif self.rules.family == "preference_reasoning" and "message" in self.names and not self.asked:
            self.asked += 1
            self.forced.append(
                {
                    "name": "message",
                    "args": {
                        "content": (
                            "Before I search: which of your priorities matters most, and "
                            "how should I weigh the others?"
                        )
                    },
                }
            )

    def _family(self) -> str | None:
        if self.claims_required:
            return "justification"
        if self.allowed:
            return "ranking"
        if self.rules.theme:
            return "retrieval_recall"
        if self.rules.idents:
            return "recovery"
        if self.rules.brand and self.rules.specs:
            return "intent_decomposition"
        if self.rules.withheld:
            return "constraint_satisfaction"
        guess = self.rules.family
        if guess in ("preference_reasoning", "intent_decomposition", "retrieval_recall"):
            return guess
        if self.rules.phrase and not self.rules.specs and not self.rules.brand:
            return "retrieval_recall"
        return None

    def _ask(self) -> list[dict[str, Any]]:
        self._trim()
        note = self._shortlist_note()
        for attempt in range(2):
            messages = list(self.messages)
            if note:
                messages.append({"role": "user", "content": note})
            message = self.gateway.chat(
                messages, self.tools, role="act", tool_choice="required"
            )
            if not isinstance(message, dict):
                self._note(f"step {self.step}: no reply from any model")
                return []
            content = _text(message.get("content"))
            calls = [
                call
                for call in message.get("tool_calls") or []
                if isinstance(call, dict) and call.get("function")
            ]
            if calls:
                self.messages.append(
                    {"role": "assistant", "content": content, "tool_calls": calls}
                )
                return calls[: self.max_calls]
            self._note(f"step {self.step}: reply carried no tool call ({content[:60]})")
            if attempt == 0:
                self.messages.append({"role": "assistant", "content": content})
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The task is not finished. Call one of the supplied tools now; "
                            "do not answer in prose."
                        ),
                    }
                )
        return []

    def _lookup_named(self) -> dict[str, Any] | None:
        if self.allowed or not self.rules.idents or "search" not in self.names:
            return None
        if self.looked_up or self.cart:
            return None
        if self.rules.theme and self.rules.idents <= set(
            re.findall(r"[a-z0-9-]+", self.rules.theme_text)
        ):
            return None
        self.looked_up = True
        idents = sorted(self.rules.idents)
        context = self.rules.ident_context[:2]
        category = [self.rules.category] if self.rules.category else []
        for parts in (idents + context + category, idents + category, idents):
            query = " ".join(parts).strip()
            action = {"name": "search", "args": {"query": query, "in_stock": True, "k": 10}}
            signature = json.dumps(action, sort_keys=True)
            if signature in self.attempted:
                continue
            self.attempted.add(signature)
            self.searches += 1
            self._note(f"looking up the named identifier {query}")
            return action
        return None

    def _replace_dead(self) -> dict[str, Any] | None:
        if self.replaced or not self.dead_title or self.cart or self.ordered:
            return None
        if self.rules.idents or self.allowed:
            return None
        if "search" not in self.names or self.max_steps - self.step <= _FINALIZE_RESERVE:
            return None
        allowance = _MAX_SEARCHES if self.rules.family == "retrieval_recall" else _MAX_SEARCHES * 2
        if self.searches >= allowance:
            return None
        words: list[str] = []
        for word in re.findall(r"[a-z0-9]+", self.dead_title.lower()):
            if len(word) > 1 and word not in _FUNCTION_WORDS and word not in words:
                words.append(word)
            if len(words) >= _REPLACEMENT_WORDS:
                break
        if not words:
            return None
        self.replaced = True
        args: dict[str, Any] = {"query": " ".join(words), "in_stock": True, "k": 10}
        if self.rules.cap is not None:
            args["max_price"] = self.rules.cap
        action = {"name": "search", "args": args}
        signature = json.dumps(action, sort_keys=True)
        if signature in self.attempted:
            return None
        self.attempted.add(signature)
        self.searches += 1
        self._note(f"the market removed that listing; looking for another of {' '.join(words)}")
        return action

    def _mechanics(self) -> dict[str, Any] | None:
        if not self.ordered:
            duty = self._pending_duty(self.focus)
            if duty is not None:
                return duty
        if self.cart and not self.ordered:
            live = [ref for ref in self.cart if ref not in self.rejected and self._spec_tier(ref) < 2]
            ref = live[-1] if live else self.cart[-1]
            if ref not in self.rejected and self._spec_tier(ref) < 2:
                if ref not in self.verified or self._viable(ref):
                    action = self._advance(ref)
                    if action is not None:
                        self._note(f"turn {self.step}: cart mechanics {action['name']}")
                        return action
            if "remove_from_cart" in self.names and (
                ref in self.rejected or self._spec_tier(ref) >= 2 or (
                    ref in self.verified and not self._viable(ref)
                )
            ):
                self._note(f"turn {self.step}: clearing {ref[0]}::{ref[1]} from the cart")
                return {"name": "remove_from_cart", "args": {"product_id": ref[0], "sku": ref[1]}}
            self._note(
                f"turn {self.step}: carted {ref[0]}::{ref[1]} not advanced "
                f"(tier={self._spec_tier(ref)} verified={ref in self.verified} viable={self._viable(ref)})"
            )
        if (
            not self.cart
            and not self.ordered
            and self.step >= _COMMIT_BY
            and not self.awaiting_reply
            and self.rules.family != "retrieval_recall"
            and self._commit_ready()
        ):
            if self.allowed:
                best = self._ranked_top()
            elif self.rules.family == "preference_reasoning":
                best = self._preference_top() or self._best()
            else:
                best = self._best()
            settled = best is not None and (not self.rules.specs or self._spec_tier(best) == 0)
            if best is not None and not settled and self.step >= self.max_steps * 0.6:
                unread = [
                    product_id
                    for product_id in self.fresh
                    if product_id not in self.products and product_id not in self.inspected
                ]
                if not unread or self.auto_reads >= _AUTO_READS:
                    probe = self._probe()
                    if probe is not None:
                        return probe
                    settled = True
            if settled:
                action = self._advance(best)
                if action is not None:
                    self._note(f"turn {self.step}: committing to {best[0]}::{best[1]}")
                    return action
        themed = self._theme_search()
        if themed is not None:
            return themed
        unread: list[str] = []
        if self.allowed:
            unread = [ref[0] for ref in self.allowed if self._facts(ref) is None]
        elif self.fresh:
            unread = [
                product_id
                for product_id in self.fresh
                if product_id not in self.products
                and product_id not in self.inspected
                and self._on_theme(product_id)
            ]
        batch: list[str] = []
        for product_id in unread:
            if product_id not in batch:
                batch.append(product_id)
            if len(batch) >= _COMPARE_CAP:
                break
        self.fresh = [product_id for product_id in unread if product_id not in batch]
        if not batch or self.auto_reads >= _AUTO_READS:
            self.fresh = []
            return self._probe()
        if len(batch) > 1 and "compare" in self.names:
            action = {"name": "compare", "args": {"product_ids": batch}}
        elif "view" in self.names:
            action = {"name": "view", "args": {"product_id": batch[0]}}
        else:
            return None
        signature = json.dumps(action, sort_keys=True)
        if signature in self.attempted:
            return None
        self.attempted.add(signature)
        self.auto_reads += 1
        self.inspected.update(batch)
        self._note(f"turn {self.step}: reading {len(batch)} candidates")
        return action

    def _probe(self) -> dict[str, Any] | None:
        if self.cart or self.ordered or not self.products or not self.rules.specs:
            return None
        if self.rules.family in ("preference_reasoning", "ranking", "retrieval_recall"):
            return None
        if self.max_steps - self.step <= _FINALIZE_RESERVE:
            return None
        confirmed = any(self._spec_tier(ref) == 0 for ref in self._candidates())
        preferred = bool(self.rules.brand) and not self.rules.brand_firm and self.filtered == 0
        if confirmed and not preferred:
            return None
        brands = [self.rules.brand] if self.rules.brand else [""]
        if self.rules.brand and not self.rules.brand_firm:
            brands.append("")
        for brand in brands:
            if self.filtered >= 2 or (confirmed and brand != self.rules.brand):
                break
            structured = {
                key: value
                for key, value in (
                    ("category", self.rules.category),
                    ("brand", brand),
                    ("max_price", self.rules.cap),
                )
                if value
            }
            if not structured or "filter" not in self.names:
                continue
            action = {"name": "filter", "args": {**structured, "in_stock": True, "k": 10}}
            signature = json.dumps(action, sort_keys=True)
            if signature in self.attempted:
                continue
            self.filtered += 1
            self.attempted.add(signature)
            self._note("no confirmed match yet; filtering on the structured requirements")
            return action
        if confirmed:
            return None
        if self.spec_probes < _SPEC_PROBES and self.rules.phrases and "search" in self.names:
            phrases = sorted(phrase.replace("-", " ") for phrase in self.rules.phrases)
            aliases = sorted({alias for names in self.rules.aliases.values() for alias in names})
            brand = self.rules.brand if self.rules.brand_firm else ""
            category = self.rules.category or ""
            variants = [(brand, category, *phrases), (category, *phrases), tuple(phrases)]
            if aliases:
                variants[1:1] = [(brand, category, *aliases), (category, *aliases)]
            for parts in variants:
                query = " ".join(part for part in parts if part)
                args: dict[str, Any] = {"query": query, "in_stock": True, "k": 10}
                if self.rules.cap is not None:
                    args["max_price"] = self.rules.cap
                action = {"name": "search", "args": args}
                signature = json.dumps(action, sort_keys=True)
                if signature in self.attempted:
                    continue
                self.attempted.add(signature)
                self.searches += 1
                self.spec_probes += 1
                self._note(f"no confirmed match yet; searching by the stated requirement: {query}")
                return action
        return None

    def _plan(self) -> list[dict[str, Any]]:
        remaining = self.max_steps - self.step
        desperate = remaining <= _DESPERATE_AT or self.clock.left() <= _FINISH_RESERVE_SECS
        if (remaining <= _FINALIZE_RESERVE or desperate) and not self.ordered:
            forced = self._finalize(desperate)
            if forced:
                return [forced]
        if self.forced:
            return [self.forced.pop(0)]
        lookup = self._lookup_named()
        if lookup is not None:
            return [lookup]
        replacement = self._replace_dead()
        if replacement is not None:
            return [replacement]
        auto = self._mechanics()
        if auto is not None:
            return [auto]
        if not self.seeded and self.seed_tries < 2 and not self.cart:
            self._seed_constraints()

        tool_calls = self._ask()
        if not tool_calls:
            forced = self._finalize(desperate)
            return [forced] if forced else []
        actions: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            name = _text((tool_call.get("function") or {}).get("name"))
            call_id = _text(tool_call.get("id"))
            if any(action["name"] in _WRITE_TOOLS for action in actions):
                self.pending.append(
                    (call_id, {"rejected": "one cart action per turn; the earlier one is running"})
                )
                continue
            action, note = self._guard(name, _arguments(tool_call))
            if action is None:
                self.pending.append((call_id, {"rejected": note}))
                self._note(f"turn {self.step}: rejected {name}: {note[:70]}")
                continue
            signature = json.dumps(action, sort_keys=True)
            if signature in self.attempted and action["name"] in _READ_TOOLS:
                self.pending.append(
                    (call_id, {"rejected": "already done; try something different"})
                )
                continue
            self.attempted.add(signature)
            self.pending.append((call_id, None))
            actions.append(action)
        if not actions and not self.forced:
            forced = self._finalize(desperate)
            if forced:
                return [forced]
        if len(actions) == 1 and actions[0]["name"] == "inspect_stock" and not self.cart:
            seen = {json.dumps(actions[0], sort_keys=True)}
            for ref in self._candidates():
                if len(actions) > _VERIFY_EXTRA:
                    break
                if ref in self.verified or ref in self.rejected:
                    continue
                extra = {"name": "inspect_stock", "args": {"product_id": ref[0], "sku": ref[1]}}
                signature = json.dumps(extra, sort_keys=True)
                if signature in seen or signature in self.attempted:
                    continue
                seen.add(signature)
                self.attempted.add(signature)
                actions.append(extra)
        return actions[: self.max_calls]

    def _trim(self) -> None:
        if len(self.messages) <= _HISTORY:
            return
        head = self.messages[:2]
        tail = self.messages[-(_HISTORY - 2) :]
        while tail and tail[0].get("role") == "tool":
            tail = tail[1:]
        self.messages = head + tail

    def _flush(self, observations: Sequence[Any]) -> None:
        queue = list(observations)
        if not self.pending:
            for observation in queue:
                self.messages.append(
                    {"role": "user", "content": json.dumps(observation)[:_OBS_CHARS]}
                )
            return
        for call_id, note in self.pending:
            payload = note if note is not None else (
                queue.pop(0) if queue else {"error": "no observation returned"}
            )
            self.messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload)[:_OBS_CHARS]}
            )
        self.pending = []
        for observation in queue:
            self.messages.append(
                {"role": "user", "content": json.dumps(observation)[:_OBS_CHARS]}
            )

    def _submit(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        for offset in range(_TURN_PROBES):
            turn = self.turn + 1 + offset
            group = f"{self.problem_id}-turn-{turn}"
            envelope = {
                **self.binding,
                "call_id": group,
                "idempotency_key": group,
                "turn": turn,
                "calls": [
                    {"call_id": f"{group}-{index}", "action": action}
                    for index, action in enumerate(actions, start=1)
                ],
            }
            attempts = _ENV_ATTEMPTS if offset == 0 else 1
            for attempt in range(attempts):
                room = self.clock.hard()
                if room <= 1.0:
                    return None
                held = _SESSION.timeout
                _SESSION.timeout = max(_ENV_CALL_FLOOR, min(held, room))
                try:
                    result = _SESSION.post("/environment/call", json_data=envelope)
                finally:
                    _SESSION.timeout = held
                if isinstance(result, dict):
                    if offset:
                        self._note(f"turn resynced to {turn}")
                    self.turn = turn
                    return result
                if attempt + 1 >= attempts:
                    break
                wait = min(_ENV_BACKOFF_MAX, _ENV_BACKOFF_BASE * (2.0**attempt))
                if self.clock.hard() - wait <= 1.0:
                    return None
                time.sleep(wait)
        return None

    def _deliver(self, actions: list[dict[str, Any]]) -> None:
        snapshot = (
            self.searches,
            self.asked,
            self.filtered,
            self.spec_probes,
            self.looked_up,
            self.replaced,
        )
        result = self._submit(actions)
        if not isinstance(result, dict):
            self._note(f"step {self.step}: environment call returned nothing")
            (
                self.searches,
                self.asked,
                self.filtered,
                self.spec_probes,
                self.looked_up,
                self.replaced,
            ) = snapshot
            for action in actions:
                self.attempted.discard(json.dumps(action, sort_keys=True))
            self._flush([])
            return
        self.dialogue.append({"role": "assistant", "environment_result": result})
        observations: list[Any] = []
        for action, call_result in zip(actions, result.get("calls") or []):
            observation, done = _unwrap(call_result)
            try:
                self._absorb(action["name"], action.get("args") or {}, observation)
            except Exception as error:
                self._note(f"step {self.step}: {action['name']} observation unreadable {error}")
            observations.append({action["name"]: observation})
            if done:
                self.done = True
        self._flush(observations)
        user_message = result.get("user_message")
        if isinstance(user_message, dict):
            content = _text(user_message.get("content"))
            if content:
                before = self.rules.cap
                before_specs = set(self.rules.specs)
                self.rules.tighten(content)
                if self.rules.cap != before or self.rules.specs - before_specs:
                    self.filtered = 0
                    self.spec_probes = 0
                asked = any(
                    action["name"] == "message" and "?" in _text(action["args"].get("content"))
                    for action in actions
                )
                if asked and self.rules.family == "preference_reasoning" and not self.cart:
                    self.revealed = content
                    self._search_reply(content)
                if _ASKS.search(content) and self.replies < _MAX_REPLIES:
                    self.awaiting_reply = True
                if _money_caps(content) or _spec_tokens(content):
                    self.verified.discard(self.focus) if self.focus else None
                self.messages.append({"role": "user", "content": content})
                self.dialogue.append({"role": "user", "content": content})
        for action in actions:
            if action["name"] == "message":
                self.awaiting_reply = False
                self.replies += 1

    def _close(self, limit: int) -> None:
        action: dict[str, Any] | None = None
        for _ in range(limit):
            if self.ordered or self.done or self.turn >= self.max_steps:
                return
            if self.clock.hard() <= 2.0:
                return
            if action is None:
                action = self._finalize(True)
            if action is None:
                return
            before = self.turn
            self._deliver([action])
            if self.turn == before:
                continue
            action = None

    def run(self) -> list[dict[str, Any]]:
        self._seed_constraints()
        idle = 0
        stalled = 0
        pending: list[dict[str, Any]] | None = None
        while self.turn < self.max_steps and not self.done:
            if self.clock.left() <= 0:
                break
            self.step = self.turn + 1
            actions = pending if pending is not None else self._plan()
            if not actions:
                self._flush([])
                self._note(
                    f"step {self.step}: no action; cands={len(self._candidates())} "
                    f"cart={len(self.cart)} left={self.clock.left():.0f}"
                )
                idle += 1
                if idle > _IDLE_LIMIT or self.clock.left() <= _FINISH_RESERVE_SECS:
                    fallback = self._finalize(True) or self._explore()
                    if not fallback:
                        self._note("give up: nothing left to try")
                        break
                    actions = [fallback]
                else:
                    continue
            else:
                idle = 0
            self.dialogue.append({"role": "assistant", "actions": actions})
            before = self.turn
            self._deliver(actions)
            if self.turn == before:
                stalled += 1
                pending = actions
                if stalled >= _STALL_LIMIT:
                    self._note("give up: the environment stopped accepting turns")
                    break
            else:
                stalled = 0
                pending = None
            self.step = self.turn
        self._close(_CLOSE_ACTIONS)
        self._note(
            f"end step={self.step} turn={self.turn} ordered={self.ordered} "
            f"left={self.clock.left():.0f} cart={len(self.cart)} known={len(self._known_refs())} "
            f"cands={len(self._candidates())} rejected={len(self.rejected)} cap={self.rules.cap}"
        )
        return self._bounded()

    def _bounded(self) -> list[dict[str, Any]]:
        trace = {"role": "assistant", "content": "trace", "extra_info": {"trace": self.notes}}
        room = _RETURN_CHARS - len(json.dumps(trace, default=str))
        out: list[dict[str, Any]] = []
        size = 0
        for entry in self.dialogue:
            blob = json.dumps(entry, default=str)
            if size + len(blob) > min(_DIALOGUE_CHARS, room):
                break
            size += len(blob)
            out.append(entry)
        out.append(trace)
        return out


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = _Episode(problem_data)
    except Exception as error:
        return [
            {
                "role": "assistant",
                "content": "trace",
                "extra_info": {"trace": [f"agent_main: episode setup failed: {error}"]},
            }
        ]
    try:
        return episode.run()
    except Exception:
        try:
            episode._close(_CLOSE_ACTIONS)
        except Exception:
            pass
        return episode._bounded()


__all__ = ["agent_main"]
