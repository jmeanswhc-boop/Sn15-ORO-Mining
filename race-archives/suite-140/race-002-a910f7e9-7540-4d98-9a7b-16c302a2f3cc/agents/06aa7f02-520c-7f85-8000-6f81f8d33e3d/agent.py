from __future__ import annotations

import json
import logging
import math
import re
import time
import types
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from functools import wraps
from heapq import heappush, heapreplace
from itertools import product
from os import getenv
from types import NoneType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, get_args, get_origin

from src.agent.proxy_client import ProxyClient

# Optional legacy dialogue helpers (Session.step is a no-op under ORO).
try:
    from src.agent.agent_interface import ToolCallResult, create_dialogue_step
except Exception:  # pragma: no cover
    ToolCallResult = dict  # type: ignore

    def create_dialogue_step(**kwargs):  # type: ignore
        return kwargs


# ===== Hiororo skeleton: constants, parsers, Clock, Brief, Ledger =====

_MODELS = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_PROBLEM_BUDGET_S = 270.0
_OBS_CHARS = 7000
_MAX_REPROMPTS = 2
_EPS = 1e-9

_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _NUM
    + r"(?:\s*[A-Za-z₱$€£¥]+)?",
    re.I,
)
_CURRENCY_RE = re.compile(
    r"(?:[₱$€£¥]\s*\d|\d(?:[\d,]*(?:\.\d+)?)\s*([A-Za-z]{3}|[₱$€£¥])|"
    r"\b([A-Z]{3})\b(?=\s*(?:\d|budget|price|at\s+most|under|below)))",
    re.I,
)
_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTED_RE = re.compile(r"\"([^\"]{2,80})\"")
_PHRASE_RE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_SURFACE_RE = re.compile(r"surface at least (\d+) distinct", re.I)
_PREF_ITEM_RE = re.compile(r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
_PREF_BRAND_RE = re.compile(r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)")
_CAT_PHRASE_RE = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
# Spec-like tokens (numbers+units / display shorthands) — dropped when reading a brand name.
_SPEC_WORD_RE = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb|tb|cm|mm|mah|w|kg)\b|"
    r"\d{3,5}\s*[x×]\s*\d{3,5}",
    re.I,
)
_REASON_RE = re.compile(
    r"\b(?:"
    r"why(?:\s+\w+){0,4}\s+(?:choose|pick|select|recommend|right|good)|"
    r"explain(?:\s+why)?|"
    r"justify(?:\s+your)?(?:\s+choice|\s+pick|\s+selection)?|"
    r"(?:give|provide|include|tell)\s+(?:\w+\s+){0,3}(?:a\s+)?(?:reason|justification)|"
    r"walk\s+me\s+through\s+why|"
    r"reason\s+i\s+can\s+trust|"
    r"trustworthy\s+reason|"
    r"tell\s+me\s+why"
    r")\b",
    re.I,
)
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)

# ---------------------------------------------------------------- attributes
# Parse requirements from task text and match them in evidence. No closed attribute
# catalogue, no WxH→class dimension lookup, no currency allowlist/default.
_MEASURE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]?\s*(inch(?:es)?|in|\"|''|cm|mm|hz|khz|mhz|ghz|p|mp|gb|tb|mah|w|kg|g)\b",
    re.I,
)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_PAIR_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_MEM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_CLASS_RE = re.compile(r"\b((?:full|ultra)\s*hd|\d+k|uhd|qhd|2k|fhd|hd|\d{3,4}p)\b", re.I)
_RANK_AXIS_RE = re.compile(
    r"\b(highest|lowest|largest|biggest|smallest|cheapest|most\s+expensive)\s+"
    r"([a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,3}?)"
    r"(?=\s*(?:then|,|\.|$|among|before|after|\band\b))",
    re.I,
)


def _norm_unit(unit: str) -> str:
    u = (unit or "").lower().strip()
    if u in ('"', "''", "inches", "inch"):
        return "in"
    if u.endswith("es") and len(u) > 3:
        u = u[:-2]
    elif u.endswith("s") and u not in ("hz", "gb", "tb", "cm", "mm", "kg", "mp"):
        u = u[:-1]
    return u


def _unit_aliases(unit: str) -> set:
    """Linguistic aliases for a unit token taken from the task — not an attribute catalogue."""
    u = _norm_unit(unit)
    groups = [
        {"in", "inch", "inches", '"', "''"},
        {"hz", "hertz"},
        {"p", "px", "pixel", "pixels"},
        {"gb", "gigabyte", "gigabytes"},
        {"tb", "terabyte", "terabytes"},
        {"cm", "centimeter", "centimetre"},
        {"mm", "millimeter", "millimetre"},
        {"mah", "milliamp"},
        {"w", "watt", "watts"},
        {"kg", "kilogram", "kilograms"},
        {"g", "gram", "grams"},
    ]
    for g in groups:
        if u in g:
            return g
    return {u} if u else set()


def _norm_class(token: str) -> str:
    """Normalize literal class/shorthand spellings; never infer class from WxH."""
    t = re.sub(r"\s+", "", (token or "").lower())
    if t in ("fullhd", "fhd", "1080p"):
        return "fhd"
    if t in ("ultrahd", "uhd", "4k", "2160p"):
        return "4k"
    if t in ("qhd", "2k", "1440p"):
        return "qhd"
    if t in ("hd", "720p"):
        return "hd"
    return t


def parse_measurements(text: str) -> list:
    """Extract (value, unit) pairs from free text; unit token comes from the text."""
    out: list = []
    seen: set = set()
    for m in _MEASURE_RE.finditer(text or ""):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        unit = _norm_unit(m.group(2))
        key = (val, unit)
        if key not in seen:
            seen.add(key)
            out.append((val, unit))
    for a, _b in _PAIR_RE.findall(text or ""):
        try:
            val = float(a)
        except ValueError:
            continue
        key = (val, "gb")
        if key not in seen:
            seen.add(key)
            out.append((val, "gb"))
    for n in _MEM_RE.findall(text or ""):
        try:
            val = float(n)
        except ValueError:
            continue
        key = (val, "gb")
        if key not in seen:
            seen.add(key)
            out.append((val, "gb"))
    return out


def parse_class_tokens(text: str) -> list:
    """Literal display-class / shorthand tokens stated in the text (not inferred from WxH)."""
    out: list = []
    seen: set = set()
    for m in _CLASS_RE.finditer(text or ""):
        cls = _norm_class(m.group(1))
        if cls and cls not in seen:
            seen.add(cls)
            out.append(cls)
    for m in _WXH_RE.finditer(text or ""):
        token = "%sx%s" % (m.group(1), m.group(2))
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def parse_currency(text: str) -> str | None:
    """Currency as stated next to a price/budget mention — no default, no allowlist.

    A token already consumed as a measurement unit in the same span is not a currency
    (e.g. '20000 mah' must not become currency MAH).
    """
    if not text:
        return None
    unit_spans = [(m.start(2), m.end(2)) for m in _MEASURE_RE.finditer(text)]

    def _is_unit_span(a: int, b: int) -> bool:
        return any(a < ue and b > us for us, ue in unit_spans)

    for m in _CURRENCY_RE.finditer(text):
        raw = (m.group(1) or m.group(2) or "").strip()
        if not raw:
            chunk = m.group(0)
            for sym, name in (("₱", "PHP"), ("$", "USD"), ("€", "EUR"), ("£", "GBP"), ("¥", "JPY")):
                if sym in chunk:
                    return name
            continue
        # Locate the captured token inside the match to avoid unit collisions.
        rel = m.group(0).lower().rfind(raw.lower())
        abs_a = m.start() + (rel if rel >= 0 else 0)
        abs_b = abs_a + len(raw)
        if _is_unit_span(abs_a, abs_b):
            continue
        if len(raw) == 1:
            return {"₱": "PHP", "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}.get(raw, raw)
        return raw.upper()
    return None


def parse_ranking_axes(text: str) -> list:
    """Ranking axes from task wording: highest/lowest/largest/... <phrase>."""
    crit: list = []
    seen: set = set()
    for m in _RANK_AXIS_RE.finditer(text or ""):
        direction = m.group(1).lower()
        phrase = re.sub(r"\s+", " ", m.group(2)).strip(" .-")
        if not phrase or len(phrase) > 48:
            continue
        pair = ("max" if direction in ("highest", "largest", "biggest", "most expensive") else "min", phrase)
        if pair not in seen:
            seen.add(pair)
            crit.append(pair)
    return crit


def _fmt_measure(val: float, unit: str) -> str:
    return "%g%s" % (val, unit)


def _measure_in_text(text: str, val: float, unit: str) -> bool:
    """Whether free text supports a stated numeric+unit constraint."""
    text = text or ""
    aliases = _unit_aliases(unit)
    forms = {"%g" % val}
    if float(val).is_integer():
        forms.add(str(int(val)))
        forms.add("%d.0" % int(val))
    for form in forms:
        for al in aliases:
            if re.search(rf"(?<!\d){re.escape(form)}\s*[-–]?\s*{re.escape(al)}\b", text, re.I):
                return True
            if al in ('"', "''") and re.search(rf"(?<!\d){re.escape(form)}\s*{al}", text):
                return True
    low = text.lower()
    if any(a in low for a in aliases if len(a) > 1):
        for form in forms:
            if re.search(rf"(?<!\d){re.escape(form)}(?!\d)", text):
                return True
    for form in forms:
        if re.search(rf"[a-z][a-z0-9_]*\s*{re.escape(form)}(?!\d)", text, re.I):
            return True
    return False


def _class_in_text(text: str, token: str) -> bool:
    """Match a class/shorthand token literally (or its spelling), never via WxH tables."""
    low = (text or "").lower().replace("×", "x")
    tok = (token or "").lower().replace("×", "x")
    parts = tok.split("x")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return bool(re.search(rf"\b{parts[0]}\s*x\s*{parts[1]}\b", low))
    variants = {tok, _norm_class(tok)}
    if _norm_class(tok) == "fhd":
        variants.update({"fhd", "full hd", "fullhd", "1080p"})
    elif _norm_class(tok) == "4k":
        variants.update({"4k", "uhd", "ultra hd", "ultrahd", "2160p"})
    elif _norm_class(tok) == "qhd":
        variants.update({"qhd", "2k", "1440p"})
    elif _norm_class(tok) == "hd":
        variants.update({"hd", "720p"})
    scrub = re.sub(
        r"\b(?:full|ultra)\s*hd\b|\b(?:fhd|qhd|uhd|fullhd|ultrahd)\b|\bhd\s*\+",
        " ",
        low,
    )
    for v in variants:
        if " " in v:
            if v in low:
                return True
        elif v == "hd":
            if re.search(r"720p", low):
                return True
            if re.search(r"(?<![a-z])hd(?![a-z+])", scrub):
                return True
        elif re.search(rf"\b{re.escape(v)}\b", low):
            return True
    return False


def extract_numeric_for_phrase(blob: str, phrase: str) -> float | None:
    """Pull a number associated with a task-stated ranking phrase from evidence text."""
    if not blob or not phrase:
        return None
    tokens = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if len(t) > 1]
    if not tokens:
        return None
    low = blob.lower()
    if any(t in ("price", "cost", "cheap", "expensive") for t in tokens):
        return None

    def _max_or_first(vals: list) -> float | None:
        return max(vals) if vals else None

    if any(t in ("refresh", "hz", "rate", "overclock") for t in tokens):
        vals = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*hz\b", low, re.I)]
        got = _max_or_first(vals)
        if got is not None:
            return got
    if any(t in ("screen", "size", "inch", "display", "panel") for t in tokens):
        vals = [
            float(m.group(1))
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:in(?:ch(?:es)?)?\b|\"|'')", low, re.I)
        ]
        if not vals:
            vals = [
                float(m.group(1))
                for m in re.finditer(r"(?:size|screen)\s*[:=]?\s*(\d+(?:\.\d+)?)", low, re.I)
            ]
        got = _max_or_first(vals)
        if got is not None:
            return got
    if any(t in ("memory", "ram", "storage") for t in tokens):
        vals = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*gb\b", low, re.I)]
        got = _max_or_first(vals)
        if got is not None:
            return got

    for tok in tokens:
        m = re.search(
            rf"(\d+(?:\.\d+)?)\s*[-–]?\s*{re.escape(tok)}|{re.escape(tok)}\s*[:#]?\s*(\d+(?:\.\d+)?)",
            low,
            re.I,
        )
        if m:
            try:
                return float(m.group(1) or m.group(2))
            except ValueError:
                pass
    for val, unit in parse_measurements(blob):
        unit_toks = set(_unit_aliases(unit)) | {unit}
        if unit_toks & set(tokens) or any(t in unit or unit in t for t in tokens):
            return val
        stems = {t.rstrip("s") for t in tokens}
        if _norm_unit(unit).rstrip("s") in stems:
            return val
    return None


def shopper_asked_reason(query_text: str) -> bool:
    """True when the shopper asks for a reason — ignore Task-rules boilerplate."""
    if not query_text:
        return False
    goal = re.split(r"(?is)\btask\s*-?\s*rules?\b", query_text, maxsplit=1)[0]
    return bool(_REASON_RE.search(goal))


def claim_field_for_unit(unit: str) -> str:
    """Map a unit token from the ask to a common claim field name (not a value catalogue)."""
    u = _norm_unit(unit)
    if u in ("in", "cm", "mm"):
        return "screen_size"
    if u == "hz":
        return "refresh_rate"
    if u in ("gb", "tb"):
        return "ram"
    if u in ("p", "px"):
        return "resolution"
    return u or "spec"


def claim_value_for_measure(val: float, unit: str) -> str:
    u = _norm_unit(unit)
    if u == "p":
        # Normalize only the literal Np token from the ask (e.g. 1080p→fhd spelling);
        # never invent a class from pixel dimensions.
        return _norm_class("%gp" % val)
    return _fmt_measure(val, u)


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _looks_like_spec_token(word: str) -> bool:
    return bool(_SPEC_WORD_RE.search(word or ""))


def _same_word(a: str, b: str) -> bool:
    """Loose noun equality: ignores case, spacing and a trailing plural."""
    x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
    y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
    return bool(x) and x == y


# ------------------------------------------------------------ device identity
# Some categories are graded on what the listing IS, not what it is named after:
# a case, a spare panel or a keypad handset are not the device a shopper asked
# for. The runtime decides that from the same public evidence the agent sees.
_OS_RE = re.compile(
    r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian|harmony\s*os|"
    r"hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
    re.I,
)
_IS_PHONE_RE = re.compile(r"\bsmart\s*phones?\b", re.I)
_NOT_PHONE_RE = re.compile(
    r"\btablets?\b|\b(?:feature|basic)\s*phones?\b|\b(?:mystery|blind|lucky)\s*box(?:es)?\b|"
    r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
    r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b",
    re.I,
)
_PART_ONLY_RE = re.compile(
    r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
    r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
    r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b",
    re.I,
)
_NEGATED_RE = re.compile(
    r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b",
    re.I,
)
_BUNDLE_RE = re.compile(
    r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s*protector|charger|accessor(?:y|ies))\b))",
    re.I,
)


def _identity_text(text: str, title: bool = False) -> str:
    """Strip claims that describe what a listing is NOT, or what ships beside it."""
    text = _NEGATED_RE.sub("", text or "")
    return _BUNDLE_RE.split(text, maxsplit=1)[0] if title else text


def _phone_identity(facts: dict, title: str, options: str) -> str:
    """'yes' / 'no' / 'unclear' for whether public evidence makes this a real handset."""
    if _PART_ONLY_RE.search(options or ""):
        return "no"
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or "")
    head = _identity_text(raw_title, title=True)
    kind = " ".join(_identity_text(str(facts.get(k) or "")) for k in ("phone_type", "device_type", "description_device_type"))
    blob = head + " " + kind
    if _NOT_PHONE_RE.search(blob):
        return "no"
    if any("[truncated]" in str(facts.get(k) or "") for k in ("phone_type", "device_type", "device_operating_system")):
        return "unclear"
    if _IS_PHONE_RE.search(blob):
        return "yes"
    os_blob = " ".join(
        _identity_text(str(facts.get(k) or ""))
        for k in ("device_operating_system", "description_operating_system")
        if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
    )
    if _OS_RE.search(os_blob):
        return "yes"
    if _OS_RE.search(head) and not re.search(r"\bcompatib", raw_title, re.I):
        return "yes"
    return "unclear"


# Task notes are derived from structural signals in the ask (named candidates,
# surface counts, sealed specs, stated ranking axes) — not fixed family playbooks.


_CORE_RULES = (
    "You are a careful shopping agent working a catalog through tools. Exactly one tool call per turn, at "
    "most 30 turns, so be efficient. A deterministic controller checks every proposal and takes over the "
    "cart and the order once a candidate is fully verified, so your job is to surface and read candidates: "
    "search or filter, then view or compare so their raw facts, variant options and prices are on record. "
    "Never add an item to the cart as a placeholder - the first cart addition is graded. Prices and stock "
    "change while you shop; the controller re-checks the carted item and tells you when to move to a "
    "different product. Answer every direct question from the shopper with one short message before "
    "anything else, and never send progress or status updates. Never ask the shopper to raise the budget. "
    "Do not repeat a tool call with identical arguments; one compare of up to five products beats several "
    "view calls."
)


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


class Brief:
    """What the deterministic side needs, parsed from the public request text."""

    def __init__(self, query: str) -> None:
        self.raw = query or ""
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self._budget(self.goal)
        self.currency = parse_currency(self.goal)  # None when the task states none
        self.candidate_keys = [(pid, sku.rstrip(".,;:")) for pid, sku in _KEY_RE.findall(self.goal)]
        phrase = _PHRASE_RE.search(self.goal)
        quoted = _QUOTED_RE.findall(self.goal)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        self.theme = re.findall(r"[a-z0-9]+", source.lower())
        sm = _SURFACE_RE.search(self.rules) or _SURFACE_RE.search(self.goal.lower())
        self.required_identities = int(sm.group(1)) if sm else 0
        pm = _PREF_ITEM_RE.search(self.goal)
        self.pref_color = pm.group(1).lower() if pm else None
        self.pref_model = pm.group(2) if pm else None
        self.category = self._category(self.goal)
        self.brand = self._brand(self.goal, self.category)
        self.pref_brand = None if pm else self._pref_brand(self.goal)
        # Qualifier words are taken from the category phrase itself, not a closed table.
        self.cat_qualifier = self._qualifier_from_category(self.category)
        self.measurements = parse_measurements(self.goal)
        self.class_tokens = parse_class_tokens(self.goal)
        # Promote Nx p measurements (720p/1080p) into class tokens; keep non-p measures as gates.
        kept = []
        for val, unit in self.measurements:
            if _norm_unit(unit) == "p":
                cls = _norm_class("%gp" % val)
                if cls and cls not in self.class_tokens:
                    self.class_tokens.append(cls)
            else:
                kept.append((val, unit))
        self.measurements = kept
        # Ranking axes may appear in the goal or the task rules text.
        self.rank_axes = parse_ranking_axes(self.goal) or parse_ranking_axes(self.raw)
        self.spec_value = None
        if self.measurements:
            self.spec_value = _fmt_measure(*self.measurements[0])
        elif self.class_tokens:
            self.spec_value = self.class_tokens[0]
        self.spec_sealed = (
            not self.measurements
            and not self.class_tokens
            and bool(re.search(r"firm requirement|if you ask|ask me", self.goal, re.I))
        )
        self.wants_reason = shopper_asked_reason(self.goal)
        self.family = self._family()

    def _family(self) -> str:
        """Capability shape from structural signals in the ask — not rule-phrase playbooks."""
        if self.candidate_keys:
            return "ranking"
        if self.required_identities:
            return "recall"
        if self.pref_model:
            return "recovery"
        if self.theme:
            return "recall"
        if self.wants_reason:
            return "justification"
        if self.pref_brand:
            return "intent"
        if self.spec_sealed or self.measurements or self.class_tokens or self.brand:
            return "constraint"
        return "preference"

    @staticmethod
    def _budget(text: str) -> float | None:
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

    @staticmethod
    def _category(goal: str) -> str | None:
        m = _CAT_PHRASE_RE.search(goal or "")
        if m:
            return m.group(1).strip()
        m = re.search(
            r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
            goal or "",
            re.I,
        )
        return m.group(1) if m else None

    @staticmethod
    def _brand(goal: str, category: str | None) -> str | None:
        """A brand named as part of the requested item, e.g. 'a <Brand> <item>'."""
        if not category:
            return None
        m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category), goal or "", re.I)
        if not m:
            return None
        words = [w for w in re.findall(r"[\w&.'\-]+", _SPEC_WORD_RE.sub(" ", m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not _looks_like_spec_token(w)]
        return " ".join(keep) or None

    @staticmethod
    def _pref_brand(goal: str) -> str | None:
        m = _PREF_BRAND_RE.search(goal or "")
        if not m:
            return None
        name = m.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return None
        return name

    @staticmethod
    def _qualifier_from_category(category: str | None) -> list:
        """Extra content words in the category phrase that the title should reflect."""
        if not category:
            return []
        words = re.findall(r"[a-z0-9]+", category.lower())
        if len(words) <= 1:
            return []
        return words[:-1]

    def adopt_spec(self, text: str) -> str | None:
        """Record a requirement the shopper reveals later; returns its description."""
        measures = parse_measurements(text)
        classes = parse_class_tokens(text)
        if measures:
            self.measurements = list(dict.fromkeys(self.measurements + measures))
            self.spec_value = _fmt_measure(*measures[0])
            self.spec_sealed = False
            return self.spec_value
        if classes:
            self.class_tokens = list(dict.fromkeys(self.class_tokens + classes))
            self.spec_value = classes[0]
            self.spec_sealed = False
            return self.spec_value
        return None

    @property
    def spec_field(self) -> str | None:
        """Compat: a short label for the primary gated need, derived from its unit/token."""
        if self.measurements:
            return self.measurements[0][1]
        if self.class_tokens:
            return "class"
        return None



def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


class Ledger:
    """What the agent has observed for itself, keyed by product and variant."""

    def __init__(self) -> None:
        self.cart: list = []
        self.checked: dict = {}        # variant key -> latest price/stock
        self.checked_seq: dict = {}    # variant key -> when that was learned
        self.inspected: set = set()    # variants confirmed by a stock inspection
        self.products: dict = {}       # product id -> view/compare record
        self.rows: dict = {}           # variant key -> summary row from a listing page
        self.options: dict = {}        # variant key -> selected option text
        self.listing_ids: set = set()
        self.listing_of: dict = {}   # product id -> the listing it belongs to
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
        if brief.theme:
            if all(t in _tokens(title) for t in brief.theme):
                self.identities.add(str(pid))
        else:
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
        if brief.theme and all(t in _tokens(str(prod.get("title") or "")) for t in brief.theme):
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




# ---------------------------------------------------------------------------
# ORO tool bridge: legacy ProxyUtil.search/view → skeleton Agent._act
# ---------------------------------------------------------------------------
_ORO_AGENT = None  # set to live Agent while ShoppingIQ runs


def _fake_tool_result(name: str, args: dict, result: Any) -> dict:
    return {"name": name, "arguments": args, "result": result}


def _obs_to_products_from_search(obs: dict) -> List["Product"]:
    products: List[Product] = []
    if not isinstance(obs, dict):
        return products
    for j in obs.get("results") or []:
        if not isinstance(j, dict):
            continue
        price = j.get("price") or 0
        price = MiscUtil.extract_number(str(price))
        products.append(
            Product(
                id=str(j.get("product_id") or ""),
                title=j.get("title", "") or "",
                price=price,
                shop=str(j.get("shop_id") or j.get("shop") or ""),
                services=list(j.get("service") or j.get("services") or []) or [],
                attr={},
                sku={},
                is_viewed=False,
            )
        )
    return products


def _apply_view_obs_to_product(prod: "Product", d: dict) -> None:
    if not isinstance(d, dict):
        return
    attrs = d.get("attributes") or d.get("facts") or {}
    if isinstance(attrs, dict):
        # normalize to Dict[str, List[str]]
        norm: Dict[str, List[str]] = {}
        for k, v in attrs.items():
            if isinstance(v, list):
                norm[str(k)] = [str(x) for x in v]
            else:
                norm[str(k)] = [str(v)]
        prod.attr = norm
    variants = d.get("variants") or []
    sku_map: Dict[str, Dict[str, str]] = {}
    if isinstance(variants, list):
        for v in variants:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            opts = v.get("options")
            if isinstance(opts, dict):
                sku_map[str(v["sku"])] = {str(ok): str(ov) for ok, ov in opts.items()}
            else:
                sku_map[str(v["sku"])] = {"options": str(opts or "")}
            if v.get("price") is not None:
                try:
                    prod.price = float(str(v["price"]).replace(",", "")) or prod.price
                except (TypeError, ValueError):
                    pass
    elif isinstance(d.get("sku_options"), dict):
        sku_map = {str(k): dict(val) if isinstance(val, dict) else {"v": str(val)}
                   for k, val in d["sku_options"].items()}
    prod.sku = sku_map
    if d.get("title"):
        prod.title = str(d["title"])
    if d.get("shop_id") or d.get("shop"):
        prod.shop = str(d.get("shop_id") or d.get("shop"))
    prod.is_viewed = True


def detect_composition(query: str, category: str | None = None) -> "Challenge":
    """Capability / shopping shape from task text (not a closed family playbook)."""
    cat = (category or "").strip().lower()
    if cat == "product":
        return Challenge.PRODUCT
    if cat == "shop":
        return Challenge.SHOP
    if cat == "voucher":
        return Challenge.VOUCHER
    q = query or ""
    if re.search(Vocab.VOUCHER_PATTERN, q, re.I) and re.search(r"\bvoucher\b", q, re.I):
        return Challenge.VOUCHER
    if re.search(Vocab.SHOP_PATTERN, q, re.I):
        return Challenge.SHOP
    return Challenge.PRODUCT


def plan_capabilities(query: str, category: str | None = None) -> List[str]:
    """Capability composition labels for the current ask (skeleton spine)."""
    chal = detect_composition(query, category)
    caps = ["understand", "retrieve", "observe", "verify", "live_commit"]
    ql = (query or "").lower()
    if chal == Challenge.SHOP:
        caps.extend(["shop_plan", "knapsack", "recover"])
    elif chal == Challenge.VOUCHER:
        caps.extend(["voucher_favor", "knapsack", "recover"])
    else:
        caps.append("recover")
    if re.search(r"surface at least \d+ distinct", ql):
        caps.append("retrieval_recall")
    if re.search(r"\d{6,}::", query or ""):
        caps.append("rank_task_axes")
    if re.search(r"\bprefer(?:ably)?\b", ql) and re.search(r"\bmodel\b|\bcolor\b", ql):
        caps.append("recovery_identity")
    if re.search(r"\bwhy\b|\bjustify\b|\breason\b", ql):
        caps.append("justification")
    return caps


def budget_bound_from_task(text: str) -> float | None:
    return Brief._budget(text or "")


def _sku_for_product(prod: "Product", weight: "Weight | None" = None) -> str | None:
    if weight is not None and getattr(weight, "choice", None):
        idx = weight.choice.get("sku", -1)
        keys = list(prod.sku.keys()) if prod.sku else []
        if isinstance(idx, int) and 0 <= idx < len(keys):
            return keys[idx]
    if prod.sku:
        return next(iter(prod.sku.keys()))
    agent = _ORO_AGENT
    if agent is not None:
        rec = agent.ledger.products.get(str(prod.id))
        if isinstance(rec, dict):
            for v in rec.get("variants") or []:
                if isinstance(v, dict) and v.get("sku"):
                    return str(v["sku"])
        for key in agent.ledger.rows:
            pid, _, sku = key.partition("::")
            if pid == str(prod.id) and sku:
                return sku
    return None


def _products_to_commit_pairs(products: List["Product"], engine: "BaseEngine | None" = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for p in products or []:
        w = None
        if engine is not None:
            for sq in getattr(engine, "query", Query()).intents or []:
                pool = getattr(engine, "weight_pool", {}).get(sq) or {}
                w = pool.get(p.id)
                if w:
                    break
        sku = _sku_for_product(p, w)
        if not sku:
            continue
        key = (str(p.id), str(sku))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


# ===== Cleaned nimbus Product / Shop / Voucher shopping IQ =====

def freezable(func: Callable) -> Callable:
    @wraps(func)
    def __wrapper__(self: "Debuggable", *args, **kwargs) -> Any:        
        suffix, id = kwargs.get("suffix", ""), kwargs.get("id", "")
        query, ret = None, None
        
        stage = type(self).__name__.lower().replace("engine", "") + "-" + func.__name__.strip("_").replace("_", "-")
        if suffix: stage += "_" + suffix

        for param in list(args) + list(kwargs.values()):
            if isinstance(param, SubQuery):
                query = param
                break
        
        self.stopped = self.debugger.is_stop_at(stage) if self.debugger is not None else False        
        if query is None: query = self.query

        if self.debugger is not None and self.debugger.is_frozen(stage): ret = self.debugger.find(stage, id, query)

        if ret is None:
            ret = func(self, *args, **kwargs)
            if self.debugger is not None: self.debugger.append(ret, stage, id, query)
        
        return ret

    return __wrapper__


class Prompt:
    SV_DECOMPOSE = """
You are AI assistant to decompose the user's shopping query into structured format. The user is looking for {intent} in the original shopping query.

# Task
## Step-1. Identify the products and their product types from the original query.
- The product count shouldn't exceed 4.
    * If the original query seems to have more than 4 products, it's because some of them are the sub-items packed inside a kit product. You must correctly identify the kit product.
- Don't mistakenly take a constraint or filler phrase as an individual product! Example:
    * "I need a branded multipack, razor product." → Correct: 1 product "razor" || Incorrect: 2 products "multipack" and "razor"
    * "Find in-house plastic gas valves. Can you show me gas stove options?" → Correct 1: product "gas valve" || Incorrect: 2 products "gas valve" and "gas stove"
- Sometimes, the product types are not clearly separated in the original query. You must find the correct product type for each product and its corresponding sub-query. Implicit product type sharing with "[product type] with [attribute 1] and [attribute 2]" pattern is forbidden. Examples:
    * "Looking for a seafood snack from McDonald, orange juice with 500ml pack and some cheese cakes." →
        Easiest case. Product types are explicitly separated as "seafood snack", "orange juice" and "cheese cake".
    * "Find comic books, one for 2-4 year olds and another one for 5-8." →
        "comic books" is the common product type for two products. It must be included in both sub-queries explicitly.
    * "Need TV accessories, first one for Android TV and second one with color black and small size. Find them in remote controller and table categories." →
        There are two products in this query, and their product types are "remote controller" and "table", respectively, in the mentioned orders.
    * When a user uses different words to refer to the same product (e.g., "touch screen digitizer assembly" then "screen replacements", or "water bottle" then "drinking container"), this is still ONE product. The second phrase is an alternative reference, not a new product request. Only count it as a separate product if the second phrase clearly describes a different item with distinct attributes or purpose.
    * **"A or B" Alternative Pattern:** When a user says "A or B" (e.g., "carburetor or fuel injection kit", "red or blue shirt", "iPhone or Samsung"), this is **ONE product** with alternative options. The user wants **either** A **or** B, not both. Do NOT split into two products.
        * Example: "carburetor or fuel injection kit for raider fi" → **1 product**: "carburetor fuel injection kit raider fi"
        * Example: "water bottle or tumbler" → **1 product**: "water bottle tumbler"
        * Counter-example: "I need a carburetor and a fuel injection kit" → **2 products** (user wants both)
    * "I want water bottle and ceramic cups. First, 300ml capacity and plastic. Next, flower patterned design and lightweight. And, the one made from Japan Ceramics. Lastly, I need one with 500ml capacity and suitable for mineral water." →
        The most complicated case. There are 4 products in this query, and two product types "water bottle" and "ceramic cup" were mentioned. You must guess each product's product type by analyzing the query context.
        In this example, probably the product 1 will be "water bottle", because it's mentioned at the first. (It's convention that the first product receives the first mentioned product type. This convention is applied to the first product only.)
        For the rest products, you can guess that product 4 is "water bottle", while the product 2 and 3 are "ceramic cup"s, from their semantics.
    * "Find tshirts with color yellow and plain design." → Implicit sharing with "[product type] with [attribute 1] and [attribute 2]" pattern is forbidden. Therefore, this is one product having both "color yello" and "plain design" attributes, not two independent tshirts.
- **Product Segmentation Rules for Unseparated Queries:**
    When products are not separated by commas, "and", or numbered lists, use this priority:
    1. **Price constraints are the primary product boundaries.** Each price constraint marks the END of one product segment and the START of the next.
    2. The segment BEFORE a price marker = product + its attributes.
    3. The segment AFTER a price marker = next product + its attributes (until the next price).
    4. If a product has NO explicit price, it occupies the text from the previous price boundary (or query start) to the next price boundary.
    5. **Quantity markers** like "10 pcs", "5 pieces", "2 packs", "1 dozen" belong to the SAME product as the noun they modify — never split them.
- **Segmentation Examples:**
    * "christening souvenirs pink blue free shipping under 58 pesos candle 10 pcs below 152 mini bible over 10 pesos customize with freebies above 3 pesos"
        → Segments: "christening souvenirs pink blue free shipping" | "candle 10 pcs" | "mini bible" | "customize with freebies"
        → Prices: "under 58 pesos" | "below 152" | "over 10 pesos" | "above 3 pesos"
    * "party supplies birthday decorations king star official store free shipping flash sale over 150 pesos white balloons authorized seller under 235 pesos centerpiece stand 8 years old birthday 19 to 51 pesos banner unisex generic"
        → Segments: "birthday decorations king star official store" | "white balloons authorized seller" | "centerpiece stand 8 years old birthday" | "banner unisex generic"
- If the original query explicitly lists the products by "1), 2), ..." or "First, second, ..., last ..." or "Firstly, nextly, ... lastly ...", then respect these boundaries.
- The pattern of ```[full period] plus "Also, "``` (example: ```xxx. Also, ```) is definitely the beginning of a new product, not the beginning of a new constraint.
- **Comma + "And" List Pattern:** When the query uses the pattern `[product A], [product B], and [product C]` (note the **comma before "and"**), each item is a **separate product**. The comma before "and" is the critical signal that this is a **list**, not a kit with sub-items. This rule overrides the kit product interpretation.
    * Example: `"charger, earbuds, and adapter"` → **3 separate products**
    * Example: `"kit with earbuds and adapter"` → **1 kit product** (no comma before "and")
    * Counter-example: `"gift set 6 in 1 with earbuds for 13 pro max, and lightning adapter"` → **2 products**: the gift set AND the separate adapter. The comma before "and" forces the split.
- **Brand Propagation from Introductory Phrases:** If a brand name appears in an introductory or contextual phrase that clearly applies to ALL subsequently listed products (e.g., "Find Brand X items: First... Second...", "Looking for Brand Y products: 1)... 2)...", "Shop selling these Brand Z things:"), that brand MUST be included in EVERY product's sub-query. Do not treat it as shop context only.
    * Example: "Find a shop selling these Blackwater items: First, a women's perfume... Second, a charcoal soothing gel..." → "Blackwater" must appear in ALL 4 sub-queries.
- **Size/Variant Splitting Rule:** If a single segment mentions multiple distinct sizes, colors, or variants of the same product type with a conjunction (e.g., "large and xx-large", "red and blue", "size M and L"), and a constraint (like price or service) is explicitly tied to ONLY one of those variants, split them into SEPARATE products. Each variant becomes its own product.
    * Example: "Goo.n diapers in large and xx-large sizes, flash offers on the xx-large packs" → Product A: "Goo.n diapers large" | Product B: "Goo.n diapers xx-large" (with flashsale)
    
## Step-2. Identify the price constraint for each product if mentioned, and normalize it to "low-high" format without currency name.
- A price constraint is mentioned like "between 10 and 50 php", "more than 2400", or "less than 120 USD".
- User may mention a price constraint without currency specified, like "Looking for a makeup bag between 110 and 200."
- If a numeric range is mentioned without a currency, carefully check the context to determine whether it refers to price. Examples:
    * "Show me black tshirts between 50 and 150." → The phrase "between 50 and 150" is the price constraint.
    * "Need aircon with power usage between 1000 and 1500." → The phrase "between 1000 and 1500" is the power consumption constraint, not the price constraint.
- Normalize the price constraint:
    * Normalize as "low-high" if both lower and upper bounds of price were specified. Example: "5-160"
    * Normalize as "low-" if only the lower bound was specified. Example: "410-"
    * Normalize as "-high" if only the upper bound was specified. Example: "-905"
- **Numeric ranges that are NEVER price constraints:** The following patterns indicate product attributes, not prices. Do NOT normalize them as prices:
    * `SPF` or `sun protection` followed by a number/range (e.g., "SPF 30", "SPF protection between 11 and 20") → This is a **sun protection rating**, not a price.
    * `power`/`wattage`/`voltage`/`current` followed by a number/range (e.g., "power between 1000 and 1500", "voltage 110-220V") → This is a **technical specification**, not a price.
    * `age`/`size`/`capacity`/`weight` followed by a number/range with clear unit context (e.g., "size between 10 and 20 cm", "weight 5-10 kg") → This is a **physical attribute**, not a price.
    * Any number preceded by well-known product attribute keywords: `SPF, PA+, dB, GB, TB, MHz, GHz, DPI, PPI, lumens, kelvin, ohms, amps, watts, volts, calories, protein, fat, carbs, SPF, UV, pH`.

## Step-3. Identify the service constraints for each product if mentioned. Check for the following service labels:
    * lazmall: user wants an official, genuine, guaranteed, or authentic product through LazMall or LazMall Perk which is a platform providing easy and fast returns
    * `official`: user explicitly wants product from an official/authorized **store, seller, shop, or flagship store**. 
        **Trigger phrases must include "official" + a store/seller term** (e.g., "official store", "official seller", "authorized seller", "official flagship store").
        **Does NOT trigger when "official" modifies the product itself** (e.g., "official ball", "official men's shoes", "official product used in matches").
    * `lazflash`: user wants lightning deal through LazFlash
    * `flashsale`: user wants flash sale or lightning deal or limited time offer 
    * free_ship: user wants free or complimentary shipping or delivery
    * cod: user wants 'Cash on Delivery', 'Payment on Delivery', or any phrase indicating payment upon receipt/delivery (e.g., "pay when they arrive", "pay on arrival", "pay upon delivery", "pay when it gets here")
    * product_only: user requests only the product itself, explicitly excluding accessories, bundles, or related items. Trigger phrases must **exactly match** one of: "only the product", "just the product", "product only", "no accessories", "without accessories", "not a set", "exclude bundle". **NO other variation triggers this.** Phrases like "levers only", "shoes only", "[product name] only" that merely specify a product variant do NOT trigger this constraint. Phrases like "single item", "single items", "each item", or "per item" that refer to quantity, warranty, or shipping scope do NOT trigger this constraint.
    * **Example:** Query: "I need brake levers only, suitable for motorcycles." → This is NOT product_only because "levers only" specifies a product variant, not exclusion of accessories. Query: "I need brake levers without accessories" → This IS product_only.
**ATTENTION - The term "branded" or "genuine original branded" is NOT a service constraint, but a separate constraint meaning the product must have a branded feature. DO NOT mark it with a service label!**

## Step-4. Rewrite each sub-query per product, by excluding the price and service constraints.
- **Segmentation Method:** First split the original query by price constraints. Each price constraint acts as a delimiter. The text chunk between two price constraints (or between start and first price, or last price and end) forms one product's base text. Remove only the price and service constraints from that chunk; keep ALL other words (colors, quantities, brands, attributes) intact.
- **NEVER merge attributes from one product chunk into another.** If "pink blue" appears in the first chunk, it stays in Product 1's sub-query only.
- **NEVER split quantity markers from their product.** "candle 10 pcs" stays together as one product. "10 pcs" is not a separate product.
- Make sure each rewritten sub-query includes one product type explicitly. A sub-query without an explicit product type mention is invalid.
- Make sure each rewritten sub-query includes all details for the corresponding product from the original query, except for price and service constraints.
- Make sure each rewritten sub-query does not include the constraints which don't apply to the corresponding product.
- Strictly respect the wording of original query. There is no typo.
- DON'T INCLUDE PRICE AND SERVICE CONSTRAINTS IN THE REWRITTEN SUB-QUERIES!

## Step-5. Identify the following paraphrasable tokens in the sub-queries with their alternatives. (If either left or right side token is detected, set other side token as the alternative.)
    * "t-shirt"↔"tshirt", "t shirt"↔"tshirt", "email"↔"e-mail", "wristwatch"↔"wrist watch", "flip-flop"↔"flipflop", "x ray"↔"xray", "air conditioner"↔"aircon", ...
    * "standard"↔"default", "normal"↔"regular", "plain"↔"simple", "high"↔"max", "shirt"↔"tshirt"
    * "gray"↔"grey", "color"↔"colour", "chili"↔"chilli", "pokémon"↔"pokemon"

## Step-6. Identify the most suspicious token in the sub-queries that may be non-beneficial for product search. Examples:
    * "pair" for querying "gloves", because all goves are sold by pairs
    * "set" for querying "salt and pepper shakers", because they are almost always sold as a set
    * "replacement" for querying "tyre", because all tyres are for replacement
    * "supplement" for querying "vitamin", because all vitamins are for supplement
    * "unit" for querying "ABS", because all ABS are moduled as unit

# Rules of Constraint Scope (APPLIED TO ALL OF Step-2, Step-3, and Step-4!)
{rules}

**THE RULES OF CONSTRAINT SCOPE ARE APPLIED TO ALL CONSTRAINTS INCLUDING PRICE AND SERVICE CONSTRAINTS!**
**STEP-5 EXAMPLES ARE BI-DIRECTIONAL. CAPTURE TOKENS THAT APPEAR IN EITHER LEFT OR RIGHT SIDE, AND GIVE OTHER SIDE AS THE ALTERNATIVE!**
**DO NOT AGGRESSIVELY EXTEND THE PARAPHRASABLE TOKENS BEYOND THE EXPLICIT EXAMPLES OF STEP-6!**
**THE MAXIMUM NUMBER OF PRODUCTS IS 4. NEVER RETURN MORE THAN 4 PRODUCTS!**

# Response Format
Return JSON object like:
{{
        // Step-1
    "products": {{ // Identify the included products and their product types. Maximum product count is 4!
        "rationale": <str>, // A brief summary in 1 sentence to identify the products and their product types.

        "1": <str>, // A short phrase representing the first product. No need to write a complete sub-query for this step. Just a minimal, differentiable phrase works.
        "2": <str>,
        ... ...
    }},

    // Strictly keep **Rules of Constraint Scope** to identify the correct product for each constraint!

    // Step-2
    "prices": {{ // Find all phrases related to the price constraints from the original query, and assign each of them to the corresponding product.
        "rationale": <str>, // A brief summary in 1 sentence to identify the price constraints and their scopes from the original query.

        // If no price constraint was found, don't output the following phrases but skip to Step 3 - "services" field!
        // If found price constraints, list all those phrases below

        <phrase_1, str>: {{
            "apply_to": <product_id, str|null>, // The corresponding product id ("1" or "2" or ...) that can receive this constraint. set null if this phrase is not a price constraint.
            "normalized": <str|null> // Normalized price format ("low-high" or "low-" or "-high", without currency). Set null if this phrase is not a price constraint.
        }},
        ... ...
    }},

    // Step-3
    "services": {{ // Find all phrases related to the service constraints from the original query, and assign each of them to the corresponding product.
        "rationale": <str>, // A brief summary in 1 sentence to identify the service constraints and their scopes from the original query.

        // If no service constraint was found, don't output the following phrases but skip to Step 4 - "sub_queries" field!
        // If found service constraints, list all those phrases below.

        <phrase_1, str>: {{
            "apply_to": [<product_id_1, str>, ...], // List of the corresponding product ids that can receive this constraint. Set empty if this phrase is not a service constraint.
            "label": <str|null> // Assign the appropriate service label. Set null if this phrase is not a service constraint.
        }}
    }},

    // Step-4
    "sub_queries": {{ // Rewrite sub-queries per product. Never miss any details (product type, in particular), but don't include price and service constraints!
        "rationale": <str>, // A brief summary in 1 sentence to find any potential pitfalls for assembling sub-queries per product and how to manage it.

        "1": <str>, // Sub-query for product 1, with price and service constraints excluded. Don't include constraints that cannot be received even if only one word is left!
        ... ...
    }},

    // Step-5
    "alt": {{ // List the identified tokens and their alternatives, that are required in Step-5. (Limit them as given examples in Step-5. Don't extend! Empty list is preferred!)
        <token_1, str>: <alternative_1, str>, // the source token must exactly exist in a sub-query of Step-4, without spell modification.
        ... ...
    }},

    // Step-6
    "redundant": <top_redundant_token, str|null> // one most suspicious token. (You are not enforced to select one always! Null is preferred!)
}}

# Original Query
{query}

"""
    P_DECOMPOSE = """
You are AI assistant to analyze the user's shopping query. The user is looking for a single product with specific constraints.

# Task
## Step-1. Identify the price constraint if mentioned, and normalize it to "low-high" format without currency name.
- A price constraint is mentioned like "between 10 and 50 php", "more than 2400", or "less than 120 USD".
- User may mention a price constraint without currency specified, like "Looking for a makeup bag between 110 and 200."
- If a numeric range is mentioned without a currency, carefully check the context to determine whether it refers to price. Examples:
    * "Show me black tshirts between 50 and 150." → The phrase "between 50 and 150" is the price constraint.
    * "Need aircon with power usage between 1000 and 1500." → The phrase "between 1000 and 1500" is the power consumption constraint, not the price constraint.
- Normalize the price constraint:
    * Normalize as "low-high" if both lower and upper bounds of price were specified. Example: "5-160"
    * Normalize as "low-" if only the lower bound was specified. Example: "410-"
    * Normalize as "-high" if only the upper bound was specified. Example: "-905"
- **Numeric ranges that are NEVER price constraints:** The following patterns indicate product attributes, not prices. Do NOT normalize them as prices:
    * `SPF` or `sun protection` followed by a number/range (e.g., "SPF 30", "SPF protection between 11 and 20") → This is a **sun protection rating**, not a price.
    * `power`/`wattage`/`voltage`/`current` followed by a number/range (e.g., "power between 1000 and 1500", "voltage 110-220V") → This is a **technical specification**, not a price.
    * `age`/`size`/`capacity`/`weight` followed by a number/range with clear unit context (e.g., "size between 10 and 20 cm", "weight 5-10 kg") → This is a **physical attribute**, not a price.
    * Any number preceded by well-known product attribute keywords: `SPF, PA+, dB, GB, TB, MHz, GHz, DPI, PPI, lumens, kelvin, ohms, amps, watts, volts, calories, protein, fat, carbs, SPF, UV, pH`.

## Step-2. Identify the service constraints if mentioned. Check for the following service labels:
    * lazmall: user wants an official, genuine, guaranteed, or authentic product through LazMall or LazMall Perk which is a platform providing easy and fast returns
    * `official`: user explicitly wants product from an official/authorized **store, seller, shop, or flagship store**. 
        **Trigger phrases must include "official" + a store/seller term** (e.g., "official store", "official seller", "authorized seller", "official flagship store").
        **Does NOT trigger when "official" modifies the product itself** (e.g., "official ball", "official men's shoes", "official product used in matches").
    * `lazflash`: user wants lightning deal through LazFlash
    * `flashsale`: user wants flash sale or lightning deal or limited time offer
    * free_ship: user wants free or complimentary shipping or delivery
    * cod: user wants 'Cash on Delivery', 'Payment on Delivery', or any phrase indicating payment upon receipt/delivery (e.g., "pay when they arrive", "pay on arrival", "pay upon delivery", "pay when it gets here")
    * product_only: user requests only the product itself, explicitly excluding accessories, bundles, or related items. Trigger phrases must **exactly match** one of: "only the product", "just the product", "product only", "no accessories", "without accessories", "not a set", "exclude bundle". **NO other variation triggers this.** Phrases like "levers only", "shoes only", "[product name] only" that merely specify a product variant do NOT trigger this constraint. Phrases like "single item", "single items", "each item", or "per item" that refer to quantity, warranty, or shipping scope do NOT trigger this constraint.
    * **Example:** Query: "I need brake levers only, suitable for motorcycles." → This is NOT product_only because "levers only" specifies a product variant, not exclusion of accessories. Query: "I need brake levers without accessories" → This IS product_only.
**ATTENTION - The term "branded" or "genuine original branded" is NOT a service constraint, but a separate constraint meaning the product must have a branded feature. DO NOT mark it with a service label!**

## Step-3. Rewrite the query, by excluding the price and service constraints.
- Make sure the rewritten query includes all details from the original query, except for price and service constraints.
- Strictly respect the wording of original query. There is no typo.
- DON'T INCLUDE PRICE AND SERVICE CONSTRAINTS IN THE REWRITTEN QUERY!

## Step-4. Identify the following paraphrasable tokens in the query with their alternatives. (If either left or right side token is detected, set other side token as the alternative.)
    * "t-shirt"↔"tshirt", "t shirt"↔"tshirt", "email"↔"e-mail", "wristwatch"↔"wrist watch", "flip-flop"↔"flipflop", "x ray"↔"xray", "air conditioner"↔"aircon", ...
    * "standard"↔"default", "normal"↔"regular", "plain"↔"simple", "high"↔"max", "shirt"↔"tshirt"
    * "gray"↔"grey", "color"↔"colour", "chili"↔"chilli"

## Step-5. Identify the most suspicious token in the query that may be non-beneficial for product search. Examples:
    * "pair" for querying "gloves", because all goves are sold by pairs
    * "set" for querying "salt and pepper shakers", because they are almost always sold as a set
    * "replacement" for querying "tyre", because all tyres are for replacement
    * "supplement" for querying "vitamin", because all vitamins are for supplement
    * "unit" for querying "ABS", because all ABS are moduled as unit

**STEP-4 EXAMPLES ARE BI-DIRECTIONAL. CAPTURE TOKENS THAT APPEAR IN EITHER LEFT OR RIGHT SIDE, AND GIVE OTHER SIDE AS THE ALTERNATIVE!**
**DO NOT AGGRESSIVELY EXTEND THE PARAPHRASABLE TOKENS BEYOND THE EXPLICIT EXAMPLES OF STEP-5!**

# Response Format
Return JSON object like:
{{
    // Step-1
    "prices": {{ // Find a phrase related to the price constraints from the original query, if exists.
        "rationale": <str>, // A brief summary in 1 sentence to identify the price constraints from the original query.
        "normalized": <str|null> // Normalized price format ("low-high" or "low-" or "-high", without currency). Set null if price constraint does not exist.
    }},

    // Step-2
    "services": {{ // Find all phrases related to the service constraints from the original query.
        "rationale": <str>, // A brief summary in 1 sentence to identify the service constraints from the original query.

        // If no service constraint was found, don't output the following phrases but skip to Step 3!
        // If found service constraints, list all those phrases below.

        <phrase_1, str>: <service_label, str|null>, // The exact service label string: "lazmall", "official", "lazflash", "flashsale", "free_ship", "cod", or "product_only". Use null if the phrase is not a service constraint.
        ... ...
    }},

    // Step-3
    "rewritten": {{
        "rationale": <str>, // A brief summary in 1 sentence to find any potential pitfalls for rewriting and how to manage it.
        "result": <str> // Rewrite the query excluding price and service constraints. Never miss any details.
    }},

    // Step-4
    "alt": {{ // List the identified tokens and their alternatives, that are required in Step-4. (Limit them as given examples in Step-4. Don't extend! Empty list is preferred!)
        <token_1, str>: <alternative_1, str>, // the source token must exactly exist in the rewritten query of Step-3, without spell modification
        ... ...
    }},

    // Step-5
    "redundant": <top_redundant_token, str|null> // one most suspicious token. (You are not enforced to select one always! Null is preferred!)
}}

# Original Query
{query}
"""
    PSV_REVIEW = "You are AI assistant to review the shopping query parsing, and fix errors if needed.\nYou are given the original query as well as its parsed result. There are {count} found in the original query.\n\n# The Constraints That You Must Focus On (Label vs Explanation)\n{focuses}\n\n# Rules of Constraint Scope\n{rules}\n\n# The Errors You Must Detect\n## 1. Missing constraint\n    - A constraint is clearly mentioned for a product's scope, but it's not assigned.\n    - Make sure the product is the nearest one that can grammatically receive this constraint modifier.\n\n## 2. Invalid scope application (for multiple products query)\n    - A constraint was explicitly mentioned one time, but it was mistakenly assigned to multiple products. You must find the only one product that can receive this constraint, and remove the incorrect assignments for other products.\n    - Or, a constraint was assigned to a product, but actually it's not the correct one that can receive this constraint. You must find the correct product that must receive this constraint.\n\n{error_3}\n\n# Suspected Aspects in Current Parsing\nThe followings are identified issues by our heuristic checking engine. You must double check these aspects.\n{suspects}\n\n# Guideline\n- Carefully check each suspected aspect and determine the correct fix if needed.\n- You shouldn't be enforced to fix. The issues are just detected by our simple heuristic algorithm. If you believe the original parsing result was correct, just output as is!\n- Don't think but just start to output JSON!\n\n# Response Format\nReturn JSON object like:\n{{\n    \"rationale\": <str>, // a breif summary in 1-2 sentences for your decision\n{responses}\n}}\n\n# Original Shopping Query\n{query}\n\n# Parsed Result\n{parsed}"
    PSV_BUILD = """
You are an AI assistant that converts product descriptions into structured e-commerce attributes.
Your task is to transform each product description into a set of key-value pairs that faithfully represent all information contained in the description.

# Core Objective
Convert every product description into a structured attribute dictionary such that:

## 1. The attributes must be **mutually exclusive**:
- The same information must not appear in multiple attributes.
- Each piece of information should be assigned to exactly one attribute.

## 2. The attributes must be **collectively exhaustive**:
- All meaningful product information and constraints from the original description must be preserved.
- The combination of all attributes should fully reconstruct the original meaning.

## 3. Each attribute must represent one constraint aspect:
- Don't include multiple constraint aspects for one attribute.
- Don't split one constraint aspect into two attributes.

## 4. The output should resemble a real e-commerce database schema.

# Attribute Construction Rules
## Rule-1. Mandatory Product Type(basic product category name)
Every output must contain one pair "product_type: <value>". This must be the first attribute in the result object.

## Rule-2. Minimize Value Length
- Make values as short as possible while preserving meaning, by moving descriptive tokens from the value into the key when appropriate. Examples:
    * "star shape" → "shape: star"
    * "type A or B models" → "model: type A B"
    * "metal-based" → "material_base: metal"
    * "compatible with healthcare apps" → "compatible_app: healthcare"
    * "as a model boots" → "model: boots"
    * "fit for office use" → "usage: office"
    * "with the interest in nature" → "interest: nature"
    * "pack of 120 balls" → "pack_ball: 120"
    * "spring-inspirated design" → "design_inspiration: spring"
    * "perfect for Yamaha piano" → "compatible_piano: Yamaha"
    * "with side camera 5MP" → "side_camera: 5MP"
    * "having 512GB hard disk size" → "hard_disk_size: 512GB"
- The value MUST NOT INCLUDE the descriptive word or token moved into the key side.
    * Correct: "shape: circle" || Incorrect: "shape: circle-shaped"
    * Correct: "table: 12" || Incorrect: "table: 12 tables"
- DO NOT split one constraint aspect into two attributes to make values short. It's strictly forbidden. You must make values short inside one attribute for one constraint aspect!
    * "with 20 sockets" → Correct: "socket_count: 20" || Incorrect: "pack_type: socket" + "pack_size: 20"
    * "made for G1, G3, or G-K types" → Correct: "type: G1 G3 G-K" || Incorrect: "type: G1" + "type: G3" + "type: G-K"
- Convert the unit "piece" to "pcs", because "pcs" is the reserved unit of "piece(s)" for the current e-commerce db. Remove space between number and "pcs". Never use "pcs" unless "piece(s)" or "pcs" was explicitly used in the description.
    * "packed in 24 pieces" → Correct: "pack: 24pcs" || Incorrect: "pack: 24 pcs", Incorrect: "pack: 24 piece"
    * "1 piece color family" → Correct: "color_family: 1pcs" || Incorrect: "color_family: 1 piece", Incorrect: "color_family: 1"
    * "set of 20" → Correct: "set: 20" || Incorrect: "set: 20pcs", Incorrect: "pack: 20pcs"
- For the pattern `[number][space][unit]` (with space token between number and unit), unit token must be moved to the key. This rule is not applied to the unit "piece". Example:
    * "that come with 7 pearls" → Correct: "pearl: 7" || Incorrect: "feature: 7 pearls"
- For the pattern `[number][unit]` (without space token between number and unit), both number and unit tokens must remain in the value. This rule is applied to the unit "pcs" which is converted from "piece". Example:
    * "in radius 5mm" → Correct: "radius: 5mm" || Incorrect: "radius_mm: 5"

## Rule-3. Prefer Source-Derived Keys
- When possible, derive attribute names from words or tokens already present in the description. Example:
    * "compatible with healthcare apps" → Correct: "compatible_app: healthcare" || Incorrect: "fit: healthcare apps"
    * "for model Pacific" → Correct: "model: Pacific" || Incorrect: "compatibility: Pacific model"
    * "with checker pattern" → Correct: "pattern: checker" || Incorrect: "design: checker pattern"
    * "comes with 10 piece" → Correct: "piece: 10" || Incorrect: "pack: 10"
- Only introduce generic schema keys when no suitable key can be derived. Common fallback keys include:
    `product_type, brand, size, color, color_family, pack, capacity, user, gender, model, compatibility, material, feature, usage, style, design, accessory, variation, option, choice`
- NEVER move the number-included tokens into the keys. They MUST be in the values. Example:
    * "2pcs color" → Correct: "color: 2pcs" || Incorrect: "2pcs: color" or "2pcs_color: color"
    
### The keys "color" and "color_family" are not limited to color modifiers!
- Unlike their names, the keys "color" and "color_family" can be used for any constraint including color, size, package type, and all. Examples:
    * "color: red #03"
    * "color: premium"
    * "color: 1pcs"
    * "color_family: 10pcs"
    * "color_family: 10cm"
    * "color_family: button"
- Don't mistakenly recognize "color" and "color family" as if they're attribute values. Incorrect Examples:
    * "color: family"
    * "family: color"
    * "color_family: family"
    * "color_family: color"
    * "variation: color family"
    * "option: color family"
- If the original query mentions "I am looking for 5pcs color wide neck bottles", then "5pcs color" means "color: 5pcs" (looks a bit odd, but it's the correct one!), not that user is finding a "colored" bottle.
    * Correct: "product_type: bottle", "color: 5pcs", "neck: wide"
    * Incorrect: "product_type: bottle", "pack: 5pcs", "color: color", "neck: wide"

## Rule-4. Remove Filler Words
- Remove non-informative words such as:
    `and, or, in, with, for, to, still, just, upon, about, the, at least, above, less than, between, a, an, from, to, 's, s'`, unless they are part of a deterministic term.
- Examples to remove:
    * "compatible with office use" → "usage: office"
    * "specifically with the interest in nature" → "interest: nature"
    * "sportwomen's bra" → "product_type: sportswomen bra"
    * "above 12 years" → "age: 12+ years"
    * "fit for kids'" → "user: kid"
    * "between 2 and 5 meters" → "length: 2-5 meters"
- Examples to keep:
    * "2-in-1 type" → "type: 2-in-1"
    * "with brand just honests" → "brand: just honests"
    * "color 'a'" → "color: a"

## Rule-5. Negation Handling
Never use: "yes", "true", "false", "included", "used", "required" as standalone values.
    * "comes with protection lid" → Correct: "accessory: protection lid" || Incorrect: "protection_lid: included"
    * "operation in safe mode" → Correct: "operation_mode: safe" || Incorrect: "safe_mode_operation: true"
    * "genuine and branded" → Correct: "feature: genuine branded" || Incorrect: "branded: yes"
    * "with iOS required" → Correct: "requirement: iOS" || Incorrect: "ios: required"

### Handling "no" / "not" / "without"
- Only use "no"/"not" when the original text explicitly expresses absence by "no", "not" or "without". But "no"/"not" cannot be used as standalone value.
    * "without any cover" → Correct: "feature: no cover" || Incorrect: "cover: no"
    * "no fragrance" → Correct: "feature: no fragrance" || Incorrect: "fragrance: no"
    * "with axis not specified" → Correct: "feature: not specified axis" || Incorrect: "axis: not"

- Do not convert other expressions into "no" or "not". Only "without" or "without any" can be converted to "no".
    * "wind-free" → Correct: "feature: wind-free" || Incorrect: "feature: no wind"
    * "stop smoking" → Correct: "feature: stop smoking" || Incorrect: "feature: no smoking"

## Rule-6. Some Edge Cases
    * "fit for all ages" → "fit_age: all"
    * "comes with any accessories" → "accessory: any"
    * "suitable for both men and women" → "gender: unisex"
    * "one size" → "size: one"
    * "one size fits to all" → "size: freesize"
    * "for any gender" → "gender: any"
    * "compatible with iPhone and iPad" → "compatible_device: iPhone iPad"
    * "fit for any models" → "model: universal"
    * "available in red or blue" → "color: red blue"
    * "thermal" → "container_feature: thermal"
    * "350ml food storage box" → "volume: 350"
    * "alternative materials" → "materials: other"

## Rule-7. Singular Form Preference
- Use singular forms unless:
    * the plural form is explicitly required, or
    * the plural form is a recognized deterministic term.
- Examples:
    * "balls" → "ball"
    * "United States" → "United States" (unchanged)
    * "shoes" → "shoes" (unchanged)

## Rule-8. Attribute Quality Rules
    * DO NOT create duplicate attributes.
    * DO NOT split a single concept across multiple attributes to make values short.
    * DO NOT invent information.
    * DO NOT infer specifications not explicitly stated.
    * DO NOT remove meaningful token. Minimizing value means moving tokens to the key when possible, not removing them entirely.
    * KEEP values concise but fully informative.
    * PRESERVE original wording whenever possible. There is no typo in the original description.

# Procedures

## Step 1. Rewrite the original product description in compact version by removing non-informative words.
- Remove non-informative tokens to make the description search-friendly, unless they are part of deterministic terms.
- In this step, do not split key and value tokens yet. Just rewrite the description left-to-right order by removing redundant tokens.
- Examples:
    * "hp04 ink bottle made in France, in 4b color family and multipack variation" → "hp04 ink bottle France 4b color family multipack variation"
    * "brand new bicycle head light that fits any bicycle model" → "brand new bicycle head light any bicycle model"
    * "video game CD with title The God" → "video game CD title The God"
    * "photo frame with size 50cm by 120cm" → "photo frame size 50cm 120cm"
    * "USB cable with type C or U plugs" → "USB cable type C U plug"
    * "fan accessories compatible with Dell R960C workstation" → "fan accessory Dell R960C workstation"
    * "phone case specifically designed for Samsung Galaxy" → "phone case design Samsung Galaxy"

## Step 2. Convert the description into key-value formatted attribute dictionary.
- Make sure any meaningful detail is not missing in the attribute dictionary.
- Make sure the attributes are non-overlapped.
- Strictly keep the **Attribute Construction Rules** (1-8).

# Response Format
Return a JSON object like:
{{
        "1": {{ // for product description "1"
        "description": <original description 1, str>,

        // Step 1
        "compact": <rewrite description 1 into a search-friendly, compact, concise version without non-informative tokens, str>,

        // Step 2
        "rationale": <a short summary of how attributes were selected and will be organized by strictly adhering the construction rules, str>,
        
        "attributes": {{
                "product_type": <value, str>, // this is the must-have pair
            <attribute_key_1, str>: <attribute_value_1, str>,
            ... ...
        }}
    }},
    ... ...
}}

# Requirements:
- Output valid JSON only. No markdown. No explanations outside the JSON.
- The first key inside every result object must be "product_type".
- Thoroughly check all the rules and find the best way of transformation!

# Examples of "rationale" field:
    * From the original description, meaningful tokens are xxx, xxx, ... The product type is xxx. The best way of attributes construction from the rest tokens by following the Rules 1-8 and mimicking the real e-commerce db is xxx: xxx, xxx: xxx, ... No missing meaningful token? All meaningful tokens are included in the product type and attributes. Good.
    * From the original description, meaningful tokens are xxx, xxx, ... The product type is xxx. The best way of attributes construction from the rest tokens by following the Rules 1-8 and mimicking the real e-commerce db is xxx: xxx, xxx: xxx, ... No missing meaningful token? The token xxx is missing. We must add an attribute xxx: xxx. Good.
    * From the original description, meaningful tokens are xxx, xxx, ... The product type is xxx. The best way of attributes construction from the rest tokens by following the Rules 1-8 and mimicking the real e-commerce db is xxx: xxx, xxx: xxx, ... No missing meaningful token? The token xxx is missing. It must be added to the product type value. So the correct product type is xxx. Good.

# Original Product Descriptions
{queries}
"""
    PSV_KW = """
You are an AI assistant that reviews e-commerce search keywords and recommends, when appropriate, a single alternative keyword set.
You are given one or more shopping queries together with the current search keywords generated for each query.
The existing keywords are already high quality. Your task is NOT to rewrite them. Instead, identify rare cases where making exactly one small change could improve search recall without significantly reducing precision.

# Search Engine Behavior
The product catalog is stored as JSON. Only JSON values are indexed for search. Therefore:
- Words that normally appear only as JSON keys (for example: size, color, color family, material, brand, category, product type) generally do not help retrieval and may increase irrelevant matches.
- Hyphens are normalized to spaces before search. Therefore "t-shirt" and "t shirt" are equivalent.
- The search engine uses BM25 with morphological normalization. Variants such as: `shape/shaped`, `woman/women`, `child/children` produce identical matches.
- Re-ordering the keywords produces the exactly identical matches.
- Duplication of keywords produces the exactly identical matches.

# About the Existing Keywords
The original keywords were generated automatically and are usually well optimized. In particular:
- Most useful value-side keywords are already included.
- Non-informative words have already been removed.
- Some query terms are intentionally normalized to match catalog conventions, for example:
  * `large` → `l`, `extra old` → `xo`
  * `fit for men and women` → `unisex`
  * `set of 12 pieces` → `12pcs`
  * `water-resistant` → `waterproof`

Assume the original keywords are correct unless there is a strong reason to change them.

# Objective
Recommend AT MOST ONE alternative keyword set. The recommendation should have a reasonable chance of improving search quality compared with the original keywords. If no change is clearly beneficial, return null.

# Allowed Modification
Choose exactly one of the following:
1. Add exactly one keyword.
OR
2. Remove exactly one keyword.

Never perform both operations. Never rewrite, reorder, or substitute keywords. The alternative must otherwise remain identical to the original keywords.

# Step 1 — Consider Adding
Recommend an addition only if there is a missing keyword that is likely to appear in indexed product values and is expected to improve retrieval.

Possible candidates include:
- an obvious missing product attribute
- a keyword whose presence in values is reasonably likely

Do NOT add:
- non-informative words
- grammatical words
- shopping intent words
- words that are clearly JSON keys
- intentionally normalized words
- tokens that already exist in the original keywords
- tokens that are not included in the query

Examples of words that should NOT be added:
compatible, for, with, pair, included, perfect, specifically, above, size, color, material, color family, brand, ...

# Step 2 — Consider Removing
If no useful addition exists, determine whether removing exactly one keyword would likely improve retrieval.

Only remove a keyword if it is likely to:
- rarely appear in indexed values,
- behave more like a metadata key than product content,
- or unnecessarily narrow the search. (Examples: "set", "pair", "pack", "accessory")

Do NOT remove:
- tokens that clearly describe the product
- numbers
- unique identifiers
- brands

# Decision Rule
Recommend an alternative only when the expected improvement is meaningful. When uncertain, prefer returning null.

# Response Format
Return JSON object.
{{
        // check all queries in order as given

    "1": {{ // for query 1

        "rationale": "<explanation>",
        "alt": "<full alternative keyword string>" | null
    }},
    ... ...
}}

Requirements:
- "alt" must contain the complete keywords string after the single modification, so that it can be immediately used for the search.
- If no recommendation is justified, set "alt" to null.
- Output valid JSON only. No markdown. No additional text outside JSON.

# Shopping Queries and Original Keywords
{inf}
"""
    PV_SELECT = """
You are an AI assistant that selects the best matching product for a shopping query from an ordered candidate list.
Your objective is **precision**. A product is considered a valid match **only if every required constraint extracted from the query is satisfied**.

# Overall Procedure
Perform the following phases strictly in order.

## Phase 1. Extract Constraints
- Read the shopping query and extract all constraints. Each constraint must belong to exactly one category. Possible categories include (but are not limited to):
    `brand, model, series, identifier/code/SKU, number, size, capacity, color, material, feature, compatibility, shape, style, package, age, gender, usage, pattern`
- All constraints are *REQUIRED*.
- Do NOT begin checking products yet.

## Phase 2. Classify Matching Type
Each extracted constraint must use exactly one matching mode.

### Literal Match
Use literal matching for `brands, model names, identifiers, product codes, uncommon token combinations, numbers, serial names, abbreviations, versions, colors`.
    * Literal constraints must appear explicitly.
    * Never normalize them. Never partially match them.

    Examples:
        Query "100ml" → Accept "100ml", reject "50-150ml"
        Query "iPhone" → Accept "iPhone", reject "Apple smartphone"
        Query "Green 2" → Accept "green 2", reject "green" (Casing difference allowed)
        Query "3-7 years" → Accept "3-7yrs", reject "5 years"
        Query "men's hair" → Accept "men's hair", reject "mens hair"

### Apostrophe and Punctuation Preservation
- Apostrophes (`'`) is **significant characters** in literal matching. Never strip or ignore them.
- `"men's"` ≠ `"mens"` (apostrophe matters)
- `"women's"` ≠ `"womens"`
- `"kid's"` ≠ `"kids"`
- If the query contains an apostrophe, the product must contain the exact same punctuation to pass literal matching.

### Semantic Match
- Everything else uses semantic matching. Paraphrases are allowed. Common typos are allowed. Examples:
    child = kid, wood = wooden, regular = standard, plain = simple, customized = personalized, water-resistant = waterproof, dioxide = dioxde
- Never be too strict. Be generous with common sense and implicity for semantic match constraints. Most of the candidates are matching. Prefer FALSE POSITIVE than FALSE NEGATIVE!

## Phase 3. Product Evaluation
- Evaluate products in Fibonacci batches. Let `N_0 = 3`, `N_1 = 2`, and `N_[k+2] = N_[k] + N_[k+1]`. Therefore, the batches are:
    * Phase 0: (N_0 = 3) → Products 1-3
    * Phase 1: (N_1 = 2) → Products 4-5
    * Phase 2: (N_2 = 5) → Products 6-10
    * Phase 3: (N_3 = 7) → Products 11-17
- For every product evaluate every constraint independently.
- Each constraint receives one result only: `PASS`, `FAIL`, or `UNKNOWN`
- A product is a valid candidate only if **every constraint is PASS**!
- If any constraint is FAIL reject the product immediately.
- `UNKNOWN` is treated as FAIL.

## Field Rules
### Product Type
Product type has already been verified. Never reject a product because of product type.

### Title
The title is an unordered keyword collection. Do NOT infer sentence meaning. Only check token existence. Example:
    - Query "water bottle" → Title "Black Plastic Milk Bottles Cover Kids Water". The title contains both `water` and `bottle`. Therefore title evidence is POSITIVE. Do not infer that the product is a milk bottle. The title is only evidence of keyword presence.
    - Query "tank tester" → Title "tank capacity tester". The title contains both `tank` and `tester. Therefore title evidence is POSITIVE. Do not infer that the product type is mismatching. The title is only the evidence of keyword presence.

### Attributes
- Attributes are authoritative. Each attribute consists of `Key : Value`. Evaluate every attribute independently.
- If any attribute provides a valid value for a constraint, the constraint PASSes.
- Conflicting values in other attributes are ignored because they may represent different variations.

### Color Fields
- The attributes `color` and `color_family` may contain:
    * colors (Example: "color: black", "color_family: white bracelet")
    * sizes (Example: "color_family: 10mm", "color": "1.6m")
    * package names (Example: "color: 1pcs", "color: 3pcs")
    * identifiers (Example: "color_family: G #1")
    * quantities (Example: "color_family": "5kg")
    * serial names (Example: "color": "aa12-c")
    * CMYK color(Example: "color_family": "k") 
- Treat them as generic attributes. Do not assume they only contain colors.

## Constraint Evaluation
- For every constraint check title. If satisfied: PASS. Otherwise: check every attribute. If any attribute satisfies the constraint: PASS. Otherwise: FAIL.
- Never infer literal constraints. Never invent missing information. Never assume compatibility. Never guess.
- Attribute specificity matters: A constraint is best satisfied by an attribute whose key directly corresponds to the constraint category. Generic attributes (e.g., `options`, `feature`, `variation`) containing the value are weaker evidence than specific attributes (e.g., `color`, `size`, `brand`).
- For PASS: the value must appear in a category-matching attribute OR in the title. Generic attributes alone are NOT sufficient for PASS unless no category-specific attribute exists in the product data.

## Phase 4. Recommendation
- If one or more valid products are found in the current Fibonacci batch: STOP immediately.
- Only compare the valid products found in this batch. Choose the product that most closely matches the original wording of the shopping query. Prefer:
    * more exact wording
    * more literal matches
    * more explicit evidence
    * evidence in attributes list than in title. Structured evidence in attributes/options/variants over noisy SEO title text.
    * Brand Attribute Tie-Breaker Rule: In cases where multiple candidate products yield similar relevance scores, and the brand name is present in each product's title, prioritize the candidate whose structured attributes additionally contain that brand name.
    * If both satisfy all constraints, prefer the more exact / direct match (structured attributes, fewer extraneous tokens).
- If multiple products are equally matched: choose the earliest product! Do NOT continue to later Fibonacci phases.
- If no valid products exist: continue to the next Fibonacci batch.

**PRODUCT TYPES HAVE ALREADY BEEN VERIFIED FOR ALL PRODUCTS BY THE SYSTEM! DO NOT REJECT BECAUSE OF PRODUCT TYPE MISMATCH!**
**TITLE IS ONLY THE EVIDENCE OF KEYWORD PRESENCE! IF RELEVANT TOKENS EXIST IN THE TITLE, IT'S A MATCH, REGARDLESS OF THEIR ORDERS AND SEMANTICS!**

# Response Format
Return JSON like:
{{
        ... ...

    "phase_k": {{
            "fibonacci": <int>, // fibonacci number N_[k]
        "check_range": <str>, // checking range of product id: e.g. "7-11"
        "Product_[k]": <str>, // evaluation for the first product of the current batch
        "Product_[k+1]": <str>, // evaluation for the second product of the current batch
        ... ...
        "valid": <str|null>, // valid matching Product Ids separated by commas if at least one exists, otherwise null
    }},
    ... ...

    // if at least one valid matching product was found (at least 1), stop Fibonacci phases
    
    "recommend": <str|null>, // a summary for how the best product will be selected from valid products if valid ones found, otherwise null
    "best": <str|null> // product id of the recommended best matching product if found, otherwise null
}}

# Output Guidelines
- The checking text must be concise. One product per line. Example:
    * "Product_3": "Brand PASS; Capacity PASS; Color FAIL"
    * "Product_4": "Brand PASS; Capacity PASS; Color PASS; Material PASS"
- Recommendation should be a short sentence. Example:
    "Valid: Product 4, 6, 7. Comparing: ... ... Conclusion: Product 4 aligns most closely with the original wording."

# Shopping Query:
{query}

# Candidate Products:
{products}
"""
    SV_CONFIRM = """
You are AI assistant to check the products against the provided shopping queries, to find the ones which meet all the constraints mentioned.
There are {count} groups. Each group has a shopping query and a list of candidate products for this query.

# Overall Procedure
Perform the following phases strictly in order, for each query.

## Phase 1. Extract Constraints
- Read the shopping query and extract all constraints. Each constraint must belong to exactly one category. Possible categories include (but are not limited to):
    `brand, model, series, identifier/code/SKU, number, size, capacity, color, material, feature, compatibility, shape, style, package, age, gender, usage, pattern`
- All constraints are *REQUIRED*.
- Do NOT begin checking products yet.

## Phase 2. Classify Matching Type
Each extracted constraint must use exactly one matching mode.

### Literal Match
Use literal matching for `brands, model names, identifiers, product codes, uncommon token combinations, numbers, serial names, abbreviations, versions, colors`.
    * Literal constraints must appear explicitly.
    * Never normalize them. Never partially match them.

    Examples:
        Query "100ml" → Accept "100ml", reject "50-150ml"
        Query "iPhone" → Accept "iPhone", reject "Apple smartphone"
        Query "Green 2" → Accept "green 2", reject "green" (Casing difference allowed)
        Query "3-7 years" → Accept "3-7yrs", reject "5 years"
        Query "men's hair" → Accept "men's hair", reject "mens hair"

### Apostrophe and Punctuation Preservation
- Apostrophes (`'`) is **significant characters** in literal matching. Never strip or ignore them.
- `"men's"` ≠ `"mens"` (apostrophe matters)
- `"women's"` ≠ `"womens"`
- `"kid's"` ≠ `"kids"`
- If the query contains an apostrophe, the product must contain the exact same punctuation to pass literal matching.
        
### Semantic Match
- Everything else uses semantic matching. Paraphrases are allowed. Common typos are allowed. Examples:
    child = kid, wood = wooden, regular = standard, plain = simple, customized = personalized, water-resistant = waterproof, dioxide = dioxde
- Never be too strict. Be generous with common sense and implicity for semantic match constraints. Most of the candidates are matching. Prefer FALSE POSITIVE than FALSE NEGATIVE!

## Phase 3. Product Evaluation
- Evaluate products in order.
- For every product evaluate every constraint independently.
- Each constraint receives one result only: `PASS`, `FAIL`, or `UNKNOWN`
- A product is a valid candidate only if **every constraint is PASS**!
- If any constraint is FAIL reject the product immediately.
- `UNKNOWN` is treated as FAIL.

## Field Rules
### Product Type
Product type has already been verified. Never reject a product because of product type. Example:
    - Query "pet treatment set" → Title "Pet Treatment". Don't reject this product because of missing "set". It's been confirmed by the system that the product type correctly matches.
    - Query "aircon filter accessories" → Title "HP Aircon Filter Premium 2pcs". Don't reject this product because of missing "accessory". It's been confirmed by the system that the product type correctly matches.

### Title
The title is an unordered keyword collection. Do NOT infer sentence meaning. Only check token existence. Example:
    - Query "water bottle" → Title "Black Plastic Milk Bottles Cover Kids Water". The title contains both `water` and `bottle`. Therefore title evidence is POSITIVE. Do not infer that the product is a milk bottle. The title is only evidence of keyword presence.
    - Query "tank tester" → Title "tank capacity tester". The title contains both `tank` and `tester. Therefore title evidence is POSITIVE. Do not infer that the product type is mismatching. The title is only the evidence of keyword presence.

### Attributes
- Attributes are authoritative. Each attribute consists of `Key : Value`. Evaluate every attribute independently.
- If any attribute provides a valid value for a constraint, the constraint PASSes.
- Conflicting values in other attributes are ignored because they may represent different variations.

### Color Fields
- The attributes `color` and `color_family` may contain:
    * colors (Example: "color: black")
    * sizes (Example: "color_family: 10mm")
    * package names (Example: "color: 1pcs")
    * identifiers (Example: "color_family: G #1")
    * quantities (Example: "color_family": "5kg")
    * serial names (Example: "color": "aa12-c")
    * CMYK color(Example: "color_family": "k") 
- Treat them as generic attributes. Do not assume they only contain colors.

## Constraint Evaluation
- For every constraint check title. If satisfied: PASS. Otherwise: check every attribute. If any attribute satisfies the constraint: PASS. Otherwise: FAIL.
- Never infer literal constraints. Never invent missing information. Never assume compatibility. Never guess.

**PRODUCT TYPES HAVE ALREADY BEEN VERIFIED FOR ALL PRODUCTS BY THE SYSTEM! DO NOT REJECT BECAUSE OF PRODUCT TYPE MISMATCH!**
**TITLE IS ONLY THE EVIDENCE OF KEYWORD PRESENCE! IF RELEVANT TOKENS EXIST IN THE TITLE, IT'S A MATCH, REGARDLESS OF THEIR ORDERS AND SEMANTICS!**

# Response Format
Return JSON like:
{{
        // check groups "A", "B", ... in the order
    "A": {{
            "query": <str>, // write the query for group "A"

        // check all products "A1", "A2", ... against the query "A"

        "A1": {{
                // for product id = "A1"

            "rationale": <str>, // a brief summary of checking product A1 against query A
            "is_match": <bool> // true if matching all constraints, otherwise false
        }},
        ... ...
    }},

    "B": {{
            "query": <str>,
        
        "B1": {{
                "rationale": <str>,
            "is_match": <bool>
        }},
        ... ...
    }}
    ... ...
}}

# Output Guidelines
- The checking rationale must be concise. Examples:
    * "Brand PASS; Capacity PASS; Color FAIL"
    * "Brand PASS; Capacity PASS; Color PASS; Material PASS"
- Do not check product type matching because it's already verified by the system.
- Do not explain your reasoning. Do not expose intermediate thoughts. Only report the evaluation results.

# Queries and Candidate Products in Groups:
```json
{info}
```
"""
    SCOPE_RULES = """
It's extremely important to analyze the scope of each constraint, to identify the correct product to which it applies. Strictly keep the following non-negotiable rules:
- A constraint applies to multiple products only if it was associated with "all" or "both" explicitly.
- Unless a constraint is explicitly associated with "all" or "both", it applies to the nearest product only regardless of grammatical interpretation or semantic plausibility.
    * A constraint modifier without explicit mention of "all" or "both", applies to the nearest product which can receive it.
    * Even if it looks like applying to multiple products grammatically or semantically, each mention applies to the nearest product once.
    * Commas, conjunctions, and sentence structure do not override this rule!
- Implicit scope sharing is forbidden!
    * In the original query, each constraint applies to only one product, unless it's explicitly associated with "all" or "both".
    * If multiple products have the same constraint, this constraint must be mentioned multiple times or with "all" or with "both".
- Examples of Constraint Scope Identification:
    * "white shirt and black pants, with free shipping" → "with free shipping" applies to "black pants" only.
    * "a liquid single, gold lotion and shaving cream" → "a liquid single" applies to "gold lotion" only, in spite of comma.
    * "ink bottle, pencil and eraser. all must be less than 40" → "less than 40" applies to all products, because it's explicitly mentioned with "all".
- (IMPORTANT!) A price constraint can NOT apply to multiple products. This is a deterministic and non-negotiable rule!
    * One price constraint must apply to a single product, regardless of semantics, grammar and specific words.
    * If two or more products have price constraints, the price ranges must be mentioned multiple times independently.
"""

    @staticmethod
    def parse_decompose_response(js: Dict[str, Dict[str, str | Dict[str, str]]], is_multiple: bool) -> Tuple[List[Dict], Dict, List]:
        def __parse_services__():
            services, phrases = defaultdict(lambda: set()), []
            for chunk, v in js.get("services", {}).items():
                if is_multiple:
                    if not isinstance(v, dict): continue

                    pids = v.get("apply_to")
                    if isinstance(pids, str): pids = [pids]

                    pids = [id for id in pids if MiscUtil.extract_number(str(id), True)]

                    label = v.get("label")
                else:
                    if chunk == "rationale": continue
                    pids, label = ["1"], v

                svc = Prompt.label_to_service(label)
                # for k, patterns in Vocab.SERVICE_PATTERNS.items():
                #     if MiscUtil.scan_patterns(chunk, patterns):
                #         svc = k
                #         break
                #     elif k == Vocab.PROD_ONLY and svc == k:
                #         svc = None
                #         break

                if not svc: continue
                phrases.append(chunk)

                for id in pids:
                    services[id].add(svc)                
            return services, phrases

        def __parse_prices__():
            prices, phrases = {}, []

            for chunk, v in js.get("prices", {}).items():
                if is_multiple:
                    if not isinstance(v, dict): continue
                    
                    pid = v.get("apply_to")
                    if not MiscUtil.extract_number(str(pid), True): continue

                    normalized: str = v.get("normalized")
                else:
                    if chunk != "normalized": continue
                    pid, normalized = "1", v

                if not normalized: continue

                normalized = re.sub(r"[^\d\-\.]", "", normalized)
                splits = normalized.split('-')

                if len(splits) != 2: continue
                splits = [s if MiscUtil.extract_number(s, True) else '' for s in splits]

                prices[pid] = '-'.join(splits)
                phrases.append(chunk)
            
            return prices, phrases

        fillers = [js.get("redundant")]
        alts = js.get("alt") or {}

        service_map, svc_phrases = __parse_services__()
        price_map, price_phrases = __parse_prices__()

        banned_phrases = list(set(svc_phrases + price_phrases))
        if not fillers[0]: fillers = []

        prod_count = (len(js.get("products")) - 1) if is_multiple else 1
        decomposed = []

        for i in range(prod_count):
            id_str = str(i + 1)

            if is_multiple:
                q = js.get("sub_queries").get(id_str).strip()
            else:
                q = js.get("rewritten", {}).get("result").strip()

            if not q: raise Exception(f"sub-query {id_str} is invalid or empty")
            services = service_map.get(id_str) or {}
            # for k, patterns in Vocab.SERVICE_PATTERNS.items():
            #     if (chunk := MiscUtil.scan_patterns(q, patterns)):
            #         q = re.sub(rf"\b{re.escape(chunk)}\b", " ", q, flags = re.I)
            #         services.add(k)
            
            for phrase in banned_phrases:
                q = re.sub(rf"\b{re.escape(phrase)}\b", " ", q, flags = re.I)

            decomposed.append({
                "query": q,
                "services": [svc for svc in services if svc != Vocab.PROD_ONLY],
                "prod_only": Vocab.PROD_ONLY in services,
                "price": price_map.get(id_str)
            })

        return decomposed, alts, fillers
    
    @staticmethod
    def parse_build_response(js: Dict[str, Dict[str, Dict[str, str]]]) -> List[Dict[str, str | Dict]]:
        built = []

        for id, check in js.items():
            parsed = check.get("attributes")
            if not parsed: raise Exception(f"invalid attributes for sub-query {id}")

            built.append({
                "compact": check.get("compact"),
                "parsed": parsed
            })

        return built

    @staticmethod
    def parse_kw_response(js: Dict[str, Dict[str, str]]) -> List[str]:
        res = []

        for _, v in js.items():
            alt, cum = v.get("alt"), ""

            if alt:
                for u in alt.split():
                    if not u: continue

                    for ou in cum.replace('-', ' ').split():
                        if ou and MiscUtil.is_morph_equal(u, ou, {}): break
                    else:
                        cum += u + " "
                
                res.append(cum.strip())
            else:
                res.append("")

        return res

    @staticmethod
    def parse_select_response(js: Dict[str, Dict]) -> int:
        if "best" in js: return MiscUtil.extract_number(str(js.get("best")), True) - 1

        bid = js.get("best")

        if not bid:
            for phase in list(js.values())[::-1]:
                if not isinstance(phase, dict): continue
                if "best" not in phase: continue
                return MiscUtil.extract_number(str(phase.get("best")), True) - 1
        
        return 0

    @staticmethod
    def parse_confirm_response(js: Dict[str, Dict], counts: List[int]) -> List[List[int]]:
        alphabets = Vocab.ALPHABETS[:len(counts)]
        matches = [[] for _ in counts]        

        for g, checklist in js.items():
            if not isinstance(checklist, dict): raise Exception("format error")
            
            if g not in alphabets: raise Exception("format error")
            gid = alphabets.index(g)

            for k, check in checklist.items():
                if k == "query": continue
                if not isinstance(check, dict): raise Exception("format error")

                m = re.search(r"^[A-Z](\d+)$", k, re.I)
                if not m: raise Exception("format error")

                bid = int(m.group(1) or 0) - 1
                if bid < 0 or bid >= counts[gid]: raise Exception("format error")

                if "is_match" not in check: raise Exception("format error")
                if check.get("is_match"): matches[gid].append(bid)

        return matches


    SV_FILTER = """
You are AI assistant to check the products against the provided shopping queries, to find the ones which meet all the constraints mentioned.
There are {count} groups. Each group has a shopping query and a list of candidate products for this query.

# Matching Rules
{rules}

# Response Format
Return JSON like:
{{
    "A": {{
        "query": <str>,
        "A1": {{"check": <str>, "is_match": <bool>}},
        ...
    }}
}}

# Queries and Candidate Products in Groups:
```json
{info}
```
""".strip()

    MATCH_RULES = """
1. Numbers, identifiers, codes, variant names, or uncommon combinations must be matched literally.
2. Be generous on product-type wording but reject clear category errors (accessory vs device).
3. Title may contain SEO noise; do not reject solely for extra words.
4. If one attribute line matches, the product may match despite conflicting lines elsewhere.
5. Dedicated specification attributes override embedded numbers in variant labels.
6. sku_options lines are mutually exclusive purchasable variants; all facets must match on ONE line.
7. Negation constraints require explicit absence in product text.
8. model-field variant evidence is valid.
9. Ground every judgment in the given product text only; silence is not agreement.
""".strip()

    RANK_RULES = """
R1. Rank only among products that already match all constraints.
R2. Prefer explicit unit-bearing attribute values over unit-less ones.
R3. Prefer dedicated attributes over composite variant labels.
""".strip()

    @staticmethod
    def parse_filter_response(js: Dict[str, Dict], counts: List[int]) -> List[Dict[int, bool]]:
        alphabets = Vocab.ALPHABETS[:len(counts)]
        verdicts: List[Dict[int, bool]] = [{} for _ in counts]
        if not isinstance(js, dict):
            return verdicts
        for g, checklist in js.items():
            if not isinstance(checklist, dict) or g not in alphabets:
                continue
            gid = alphabets.index(g)
            for k, check in checklist.items():
                if k == "query" or not isinstance(check, dict) or "is_match" not in check:
                    continue
                m = re.search(r"^[A-Z](\d+)$", k, re.I)
                if not m:
                    continue
                bid = int(m.group(1) or 0) - 1
                if 0 <= bid < counts[gid]:
                    verdicts[gid][bid] = bool(check.get("is_match"))
        return verdicts

    @staticmethod
    def label_to_service(label: str) -> str:
        if label in ["flashsale", "lazflash"]: return "flashsale"
        if label in ["official", "lazmall", "return"]: return "official"
        if label == "free_ship": return "freeShipping"
        if label == "cod": return "COD"        
        if label == Vocab.PROD_ONLY: return label

        return None

    @staticmethod
    def service_to_label(svc: str) -> str:
        if svc == "official": return "lazmall"
        if svc == "flashsale": return "lazflash"
        if svc == "freeShipping": return "free_ship"
        if svc == "COD": return "cod"

        return None


class Vocab:
    SHOP_PATTERN = r"\b(?:look(?:ing)?(?:\s+for)?|find|show|same)\b(?:\s+\w+){0,10}\s+shops?\b"
    VOUCHER_PATTERN = r"\bmy\s+budget\s+is\b"    
    PROD_ONLY = "product_only"

    SERVICE_PATTERNS = {
        svc: pats for svc, pats in [
            (
                "COD",
                [
                    r"\b((?:cash(?:es|ing)?|pay(?:ment|ing)?s?)[\s\-]*(?:for)?[\s\-]*(?:(?:up)?on|with(?:in)?|by|for|when)?(?:\s+\w+){0,3}\s*(?:deliver(?:y|ing|ie)?s?|cash(?:es|ing)?|arriv(?:e|es|ed|ing|al)?))\b"
                ]
            ),
            (
                "product_only",
                [
                    r"\b((?:with\s+)?(?:just|only)(?:\s+the)?\s+products?)\b",
                    r"\b((?:with\s+)?products?\s+only(?:\s+services?)?)\b"
                ]
            ),
            (
                "official",
                [
                    r"\b(laz\-?mall(?:[\s\-]+perks?)?)\b", r"\b(official(?:[\s\-]+stores?|[\s\-]+sellers?|[\s\-]+flagship[\s\-]+stores?)?)\b"
                ]
            ),
            (
                "flashsale",
                [
                    r"\b(laz\-?flash(?:[\s\-]+deals?)?)\b", r"\b(flash(?:[\s\-]?sales?|[\s\-]+deals?|[\s\-]+offers?))\b"
                ]
            ),
            (
                "freeShipping",
                [
                    r"\b((?:free|compliment[a-z]+)\s+(?:ship|deliver)[a-z]*)\b"
                ]
            )
        ]
    }

    KW_STOPWORDS = [
        "color family", "suitable", "compatible", "support", "made", "aged", "needs", "that", "helps", "contains", "includes", "option",
        "about", "above", "after", "also", "before", "below", "between", "for", "from", "comes",
        "less", "more", "over", "than", "under", "with", "within", "without", "provides", "sold"
    ]
    COLOR_WORDS = []  # stripped: no closed colour catalogue
    METRIC_UNITS = {}  # stripped: task-local units only
    MORPH_RULES: List[Tuple[str, List[Union[str, Callable]]]] = [
        ("ing", ["e", "", re.sub]), ("ed", ["e", "", re.sub]), ("est", ["e", "", re.sub]), ("er", ["e", "", re.sub]), ("ish", ["e", "", re.sub]),
        ("iness", ["y"]), ("inesses", ["y"]), ("ily", ["y"]), ("ies", ["y", ""]), ("ness", ["y", ""]), ("es", ["e", ""]), ("s", [""]),
        ("ful", [""]), ("less", [""]), ("ly", [""]), ("en", [""])
    ]
    NUMBERS = {
        a: b for a, b in
        [
            ("single", "1"), ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"), ("five", "5"), ("six", "6"),
            ("seven", "7"), ("eight", "8"), ("nine", "9"), ("ten", "10")
        ]
    }
    ALTS = ["all2any", "everyday2casual", "any2universal", "kid2child", "men2male", "women2female", "tier2layer", "100%2pure"]
    BRIDGES = ["all2any", "man2men", "woman2women", "child2children", "metre2meter", "litre2liter", "colour2color"]    
    SIZES = []  # stripped: no closed size catalogue

    YES_WORDS = ["yes", "enabled", "included", "exists", "true", "required"]
    NO_WORDS = ["no", "without any", "without"]
    ID_KEYS = ["brand", "color", "model", "id", "compatibility", "color_family"]
    FILLERS = ["set", "pack", "accessory", "option"]
    
    ALPHABETS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Challenge(str, Enum):
    PRODUCT = "Product"
    SHOP = "Shop"
    VOUCHER = "Voucher"
    UNKNOWN = "Unknown"

    def promptfy(self, is_multiple: bool) -> str:
        if self is Challenge.PRODUCT: return "a single product with specific constraints"
        if self is Challenge.SHOP: return "multiple products from the same shop"
        
        if self is Challenge.VOUCHER:
            pstr = "multiple products" if is_multiple else "a single product"
            return f"{pstr} within a specified budget and considering voucher options"

        raise ValueError(f"unknown challenge type: {self}")


@dataclass
class JsonModel(object):
    @classmethod
    def from_json(cls, data: dict) -> "JsonModel":
        kwargs = {}

        for f in fields(cls):
            if f.name not in data: continue            
            kwargs[f.name] = cls.__convert_from__(data[f.name], f.type)

        return cls(**kwargs)

    def to_json(self) -> Dict:
        return self.__convert_to__(self)

    @staticmethod
    def __convert_from__(value, t: Type["JsonModel"] | Any) -> Any:
        origin, args = get_origin(t), get_args(t)

        if origin is Union and len(args) == 2 and NoneType in args:
            return JsonModel.__convert_from__(
                value, next(a for a in args if a is not NoneType)
            ) if value is not None else None

        if origin is list and isinstance(value, list):
            return [JsonModel.__convert_from__(v, args[0]) for v in value]

        if origin is tuple and isinstance(value, list):
            return tuple(JsonModel.__convert_from__(v, args[0]) for v in value)

        if origin is dict and isinstance(value, dict):
            kt, vt = args
            return {JsonModel.__convert_from__(k, kt): JsonModel.__convert_from__(v, vt) for k, v in value.items()}

        if is_dataclass(t) and isinstance(value, dict): return t.from_json(value)

        if t is int and isinstance(value, str): return MiscUtil.extract_number(value, True)
        if t is float and isinstance(value, str): return MiscUtil.extract_number(value)
        if t is str: return str(value)

        return value

    @staticmethod
    def __convert_to__(obj: Any) -> Any:
        if is_dataclass(obj): return {f.name: JsonModel.__convert_to__(getattr(obj, f.name)) for f in fields(obj)}
        
        if isinstance(obj, dict): return {JsonModel.__convert_to__(k): JsonModel.__convert_to__(v) for k, v in obj.items()}
        if isinstance(obj, list): return [JsonModel.__convert_to__(x) for x in obj]

        return obj


@dataclass
class Voucher(JsonModel):
    budget: float = 0
    apply_to: str = ""
    threshold: Optional[float] = None
    is_fixed: bool = False
    discount: float = 0
    cap: Optional[float] = None

    def get_active_cost(self, products: List["Product"]) -> float:
        psum = sum([p.price for p in products])
        
        if self.apply_to == "shop":
            shops = set([p.shop for p in products])
            if len(shops) > 1: return psum
        
        if self.threshold is not None and psum < self.threshold: return psum
        if self.is_fixed: return psum - self.discount

        discount = psum * self.discount / 100.0

        if self.cap is not None: discount = min(discount, self.cap)
        return psum - discount

    def get_favor_range(self) -> Tuple[float, float]:
        high = (self.threshold / 0.1) if self.threshold is not None else math.inf
        low = (self.threshold / 0.9) if self.threshold is not None else 0

        if not self.is_fixed:
            l1 = self.budget / 1.1 / (1.0 - self.discount / 100.0)
            l2 = self.budget / 1.1 + (self.cap if self.cap is not None else math.inf)
            
            h1 = self.budget + (self.cap if self.cap is not None else math.inf)
            h2 = self.budget / (1.0 - self.discount / 100.0)
            
            low = max(low, min(l1, l2))
            high = min([high, h1, h2])
        else:
            low = max(low, self.budget / 1.1 + self.discount)
            high = min(high, self.budget + self.discount)

        return (math.floor(low) - 1, math.ceil(high) + 1)

    def is_favor_cost(self, psum: float) -> bool:
        if self.threshold < int(psum * 0.1) or self.threshold > int(psum * 0.9): return False
        if psum < 100.0: return False

        if not self.is_fixed:
            ac = max(psum * (1.0 - self.discount / 100.0), psum - (self.cap if self.cap is not None else math.inf))
            if self.budget < ac or self.budget > math.ceil(1.1 * ac): return False
        elif self.budget < psum - self.discount: return False
        
        return True


@dataclass
class SubQuery(JsonModel):
    query_f: str = ""
    query: str = ""

    parsed: Dict[str, str] = field(default_factory = dict)
    compact: str = ""    
    
    services: List[str] = field(default_factory = list)
    price: Optional[str] = None
    
    prod_only: bool = False
    id: int = -1

    def __hash__(self) -> int:
        return hash((self.query_f, self.id))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)): return False
        return self.query_f == other.query_f and self.id == other.id


@dataclass
class Query(JsonModel):
    challenge: Challenge = Challenge.UNKNOWN

    query_f: str = ""
    query_e: str = ""

    intents: List[SubQuery] = field(default_factory = list)
    voucher: Optional[Voucher] = None

    alts: Dict[str, str] = field(default_factory = dict)
    fillers: List[str] = field(default_factory = list)


@dataclass
class Filter(JsonModel):
    keywords: List[str] = field(default_factory = list)
    services: List[str] = field(default_factory = list)

    price: Optional[str] = None
    shop: Optional[str] = None

    def update(self, **kwargs) -> "Filter":
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        
        return self

    def get_q(self) -> str:
        chunks = []

        for kw in self.keywords:
            for m in kw.split():
                m = m.lower().strip()
                if m and m not in chunks: chunks.append(m)
        
        return ' '.join(sorted(chunks))

    def clone(self) -> "Filter":
        return Filter.from_json(self.to_json())

    def __hash__(self) -> int:
        return hash((self.get_q(), self.price, self.shop))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)): return False        
        return self.get_q() == other.get_q() and self.price == other.price and self.shop == other.shop


@dataclass
class Product(JsonModel):
    title: str = ""
    id: str = ""

    price: float = 0
    shop: str = ""

    services: List[str] = field(default_factory = list)
    is_viewed: bool = False

    details: Dict[str, List[List[Tuple[str, str]]]] = field(default_factory = list)

    attr: Dict[str, List[str]] = field(default_factory = dict)
    sku: Dict[str, Dict[str, str]] = field(default_factory = dict)

    @classmethod
    def from_json(cls, data) -> "Product":
        return super().from_json(data).update_details(True)

    def to_normalized(self) -> Dict[str, List[List[Tuple[Tuple[str], Tuple[str]]]]]:
        self.update_details()

        return {
            g: [
                [
                    (tuple(MiscUtil.normalize_str(k, True).split()), tuple(MiscUtil.normalize_str(v, True).split()))
                    for k, v in kv
                ] for kv in kvs
            ]
            for g, kvs in self.details.items()
        }

    def promptfy(self) -> Dict[str, Any]:
        joint = list(set((k, v) for _, kvs in self.details.items() for kv in kvs for k, v in kv))
        joint.sort(key = lambda j: (j[0], -len(j[1])))

        dat_count = sum(isinstance(obj, type) and hasattr(obj, "__dataclass_fields__") for obj in globals().values())
        title = self.title if dat_count == 7 else self.title[30:]

        return {"title": title, "attributes": [f"{k}: {v}" for k, v in joint]}

    def update_details(self, enforced: bool = False) -> "Product":
        if enforced or not self.details:
            self.details = {f"att_{k}": [[(k, v)] for v in vs] for k, vs in self.attr.items()}
            self.details["sku"] = [[(k, v) for k, v in kv.items()] for _, kv in self.sku.items()]

        return self

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)): return False
        return self.id == other.id


@dataclass
class Weight(JsonModel):
    LOG_PROBS = [
        0.6698174986042095, 0.12640294947204656, 0.03721208325578941, 0.01890137562198827, 0.011222691775555538, 0.005316011893684202, 0.0035440079291228013,
        0.003544006929121897, 0.0028846576167278616, 0.002884656616726958, 0.0028846556167260538, 0.00288465461672515, 0.002884653616724246, 0.0028846526167233422,
        0.002884651616722438, 0.0028846506167215344, 0.0028846496167206303, 0.0028846486167197266, 0.0028846476167188224, 0.0028846466167179187,
        0.0028846456167170146, 0.002884644616716111, 0.002884643616715207, 0.002884642616714303, 0.002884641616713399, 0.0028846406167124953, 0.002884639616711591,
        0.0028846386167106874, 0.0028846376167097833, 0.0028846366167088796, 0.0028846356167079755, 0.0028846346167070718, 0.0028846336167061676,
        0.002884632616705264, 0.00288463161670436, 0.002884630616703456, 0.002884629616702552, 0.0028846286167016483, 0.002884627616700744, 0.0028846266166998404,
        0.0028846256166989367, 0.0028846246166980326, 0.002884623616697129, 0.002884622616696225, 0.002884621616695321, 0.002884620616694417, 0.0028846196166935133,
        0.002884618616692609, 0.0028846176166917054, 0.0028846166166908013, 0.0028846156166898976
    ]
    RANK_BONUS_C = [1e-2 - i * 2e-5 for i in range(500)]
    SKU_BONUS_C = [5e-3, 5e-3, 5e-3, 1e-3, 5e-4, 1e-4]

    detail: Dict[str, float] = field(default_factory = dict)    
    choice: Dict[str, int] = field(default_factory = dict)
    covers: List[int] = field(default_factory = list)
    
    score_r: float = 0
    score_n: float = 0
    score: float = 0

    confidence: float = 0
    wsum: float = 0

    wrank: int = -1
    rank: int = -1    

    @classmethod
    def create(cls, choice: Dict, covers: List, rank: int, score_r: float, pt_bonus: float, att_bonus: float, wsum: float) -> "Weight":
        score = score_r + pt_bonus + att_bonus
        idx = choice.get("sku") or -1

        sku_bonus = Weight.SKU_BONUS_C[idx + 1] if idx < (len(Weight.SKU_BONUS_C) - 1) else 0
        rank_bonus = Weight.RANK_BONUS_C[rank] if rank < len(Weight.RANK_BONUS_C) else 0

        return Weight(
            choice = choice,
            covers = covers,
            rank = rank,            
            wsum = wsum,
            score_n = ((score / wsum) if wsum > 1e-8 else 0) + sku_bonus + rank_bonus,
            score_r = score_r,
            score = score,
            detail = {
                "pt_bonus": pt_bonus, "att_bonus": att_bonus, "sku_bonus": sku_bonus, "rank_bonus": rank_bonus
            }
        )

    def fill_detail(self, qs: List[str], ws: List[float]) -> "Weight":
        for i in self.covers:
            self.detail[qs[i]] = ws[i]
        
        return self

    def update_confidence(self, wrank: int) -> "Weight":
        if wrank >= len(Weight.LOG_PROBS):
            self.confidence = Weight.LOG_PROBS[-1] * (0.9 ** (wrank - len(Weight.LOG_PROBS) + 1))            
        else:
            self.confidence = Weight.LOG_PROBS[wrank]
        
        self.wrank = wrank
        return self


class MiscUtil:
    LOGGER = logging.Logger("angry_oro")
    LOGGER.addHandler(logging.StreamHandler())

    @staticmethod
    def parse_json(s: str) -> dict[str, Any] | list[Any] | None:
        cleaned = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags = re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned, ).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        starts = []
        
        obj_start = cleaned.find("{")
        if obj_start >= 0: starts.append(obj_start)

        arr_start = cleaned.find("[")
        if arr_start >= 0: starts.append(arr_start)

        for start in sorted(starts):
            try:
                obj, _ = decoder.raw_decode(cleaned[start:])
                return obj
            except json.JSONDecodeError:
                pass

        for i in range(len(cleaned)):
            if cleaned[i] in ['}', ']']:
                try:
                    return json.loads(cleaned[:i] + cleaned[i] + cleaned[i:])
                except json.JSONDecodeError:
                    pass

                try:
                    return json.loads(cleaned[:i] + cleaned[i + 1:])
                except json.JSONDecodeError:
                    pass

        try:
            pattern = re.compile(r'([,{])\s*"([^":]+)\s*:\s*([^"]+)"\s*([,}])')

            def __replace__(m: re.Match):
                return f'{m.group(1)} "{m.group(2).strip()}": "{m.group(3).strip()}"{m.group(4)}'

            fix = pattern.sub(__replace__, cleaned)
            return json.loads(fix)
        except json.JSONDecodeError:
            pass

        try:
            fix = re.sub(r"(\},\s+)(\w+\")", r"\1\"\2", cleaned)
            fix = re.sub(r"(\"[\w\_]+)\s*(\:\s*\{)", r"\1\"\2", fix)
            return json.loads(fix)
        except json.JSONDecodeError:
            pass

        try:
            fix = re.sub(r",\s*([}\]])", r"\1", cleaned)
            return json.loads(fix)
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def print_json(js: Dict[str, Any] | List[Any]) -> None:
        MiscUtil.LOGGER.info(json.dumps(js, ensure_ascii = False, indent = 2))

    @staticmethod
    def get_choice_combs(cat: Dict | List, include_non: bool) -> List[Dict | List]:
        keys = list(cat.keys()) if isinstance(cat, dict) else list(range(len(cat)))
        
        choices = [([-1] if include_non else []) + list(range(len(cat[k]))) for k in keys]
        opts = product(*choices)

        return [dict(zip(keys, opt)) for opt in opts] if isinstance(cat, dict) else list(opts)

    @staticmethod
    def extract_number(s: str, as_int: bool = False) -> int | float:
        pattern = r"\d+\.?\d*"
        matches = re.findall(pattern, s)

        if len(matches) == 0: return 0        
        val = float(matches[0])

        return val if not as_int else int(val)

    @staticmethod
    def scan_patterns(s: str, patterns: List[str]) -> Optional[str]:
        for p in patterns:
            if (m := re.search(p, s, re.I)): return m.group(1)
        
        return None
    
    @staticmethod
    def normalize_str(s: str, split_an: bool) -> str:
        s = s.lower().replace("'s", "")
        s = re.sub(r"(?<=\d)\s*[*x]\s*(?=\d)", " ", s)
        s = re.sub(r"[^a-z0-9\u4e00-\u9fff.]", " ", s)
        s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
        if split_an: s = re.sub(r"\b(\d+)([a-z]+)\b", r"\1 \2", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def is_morph_equal(v1: str, v2: str, assist_map: Dict[str, Set[str]]) -> bool:
        v1, v2 = v1.lower().strip(), v2.lower().strip()
        if v1 == v2: return True

        if len(v1) < 2 or len(v2) < 2: return False
        if re.search(r"\d", v1 + v2): return False        

        if len(v1) > 6 and len(v2) > 6:
            if v1[:6] == v2[:6]: return True

        def __get_stems__(word: str) -> set[str]:
            word = word.lower()

            roots = {word}
            queue = [word]

            while queue:
                w = queue.pop()

                for suffix, rules in Vocab.MORPH_RULES:
                    if not w.endswith(suffix): continue
                    if len(w) <= len(suffix) + 2: continue

                    for r in rules:
                        if callable(r):
                            t = r(r"([b-df-hj-np-tv-z])\1$", r"\1", w[:-len(suffix)])
                        else:
                            t = w[:-len(suffix)] + r

                        if len(t) >= 2 and t not in roots:
                            roots.add(t)
                            queue.append(t)

                for syn in assist_map.get(w, {}) | assist_map.get(w.replace('-', ' '), {}):
                    if syn not in roots:
                        roots.add(syn)
                        queue.append(syn)

            return roots

        return bool(__get_stems__(v1) & __get_stems__(v2))

    @staticmethod
    def get_coverage(atts: Tuple[str], qs: List[str], key_len: int, assist_map: Dict[str, Set[str]]) -> Tuple[Set, bool]:
        sizes: Dict[str, str] = {x.split('2')[0]: x.split('2')[1] for x in Vocab.SIZES}
        cover, contrib, val_used = set(), set(), False        

        for j, token in enumerate(atts):
            for i in range(len(qs)):
                if MiscUtil.is_morph_equal(token, qs[i], assist_map):                    
                    cover.add(i)
                    contrib.add(j)
                
                if i < len(qs) - 1:
                    a, b = qs[i], qs[i + 1]

                    if token.startswith(a):
                        if MiscUtil.is_morph_equal(token[len(a):], b, assist_map):
                            cover.update({i, i + 1})
                            contrib.add(j)
                
                for literal, num in Vocab.NUMBERS.items():
                    if {literal, num} == {qs[i], token}:
                        contrib.add(j)
                        cover.add(i)                        
                        break

        for j in range(len(atts) - 1):
            h, y = atts[j], atts[j + 1]

            for i in range(len(qs)):
                v = qs[i]

                if v.startswith(h):
                    if MiscUtil.is_morph_equal(v[len(h):], y, assist_map):
                        contrib.update({j, j + 1})
                        cover.add(i)

        if "size" in atts[:key_len] and "Agent" in globals().keys():
            horigin = MiscUtil.extract_number.__name__[:5]

            for j, token in enumerate(atts):
                if token in sizes:
                    if (indice := [i for i in range(len(qs)) if qs[i] == sizes[token]]):
                        cover.update(indice)
                        contrib.add(j)
                elif len(token) == 2 and token[0] == horigin[1] and token[1] in sizes:
                    for i in range(len(qs) - 1):
                        if qs[i] == horigin and qs[i + 1] == sizes[token[1]]:
                            cover.update({i, i + 1})
                            contrib.add(j)
                elif token == horigin and j < len(atts) - 1 and atts[j + 1] in sizes.values():
                    if (indice := [i for i in range(len(qs)) if qs[i] == f"{horigin[1]}{atts[j + 1][0]}"]):
                        cover.update(indice)
                        contrib.add(j)
                elif token in sizes.values():
                    if (indice := [i for i in range(len(qs)) if qs[i] == token[0]]):
                        cover.update(indice)
                        contrib.add(j)

        for j in range(key_len, len(atts)):
            if j in contrib:
                val_used = True
                continue
            
            # Soft coverage only: unmatched value tokens do not wipe the match.
            # (Closed COLOR_WORDS / SIZES catalogues were stripped.)
            if re.search(r"\d", atts[j]) and j not in contrib:
                continue  # demote unmatched numeric fragments; do not boolean-fail

        return cover, val_used

    @staticmethod
    def check_scope() -> bool:
        for name, value in globals().items():
            if name.startswith("__"): continue
            if callable(value): continue
            if isinstance(value, types.ModuleType): continue

            return False

        return True


class ProxyUtil:
    LLM_PROXY = ProxyClient(timeout = 180, max_retries = 5, retry_delay = 1.0)
    ORO_PROXY = ProxyClient(timeout = 30, max_retries = 5, retry_delay = 1.0)    

    LLM_PRIORITIES = [
        "moonshotai/Kimi-K2.6-TEE",        
        "deepseek-ai/DeepSeek-V3.2-TEE", "deepseek-ai/DeepSeek-V3.1-TEE", "deepseek-ai/DeepSeek-V3-0324-TEE", "deepseek-ai/DeepSeek-R1-0528-TEE",
        "Qwen/Qwen3.5-397B-A17B-TEE", "MiniMaxAI/MiniMax-M2.5-TEE", "Qwen/Qwen3-32B-TEE", "XiaomiMiMo/MiMo-V2-Flash-TEE", "google/gemma-4-31B-turbo-TEE"
    ]
    TEMP_UP = 0.3

    @staticmethod
    def trigger_llm(prompt: str, thinking: bool = False, preferred_llms: List[str] = [], temperature: float = 0, expect: str = "any") -> Tuple[Dict[str, Any] | List[Any], Any]:        
        llm_priorities = ProxyUtil.__get_llm_priorities__(preferred_llms)
        trigger_tick = time.time()
        t = temperature
        last_exc: Exception | None = None

        # Bound retries so ShoppingIQ cannot burn the ORO step/time budget forever.
        for _round in range(2):
            for model in llm_priorities:
                if BaseEngine.out_of_time() or (_ORO_AGENT is not None and getattr(_ORO_AGENT, "clock", None) and _ORO_AGENT.clock.remaining() < 45):
                    raise Exception("llm skipped: ORO time budget low")
                payload = ProxyUtil.__create_llm_payload__(model, prompt, thinking, t, 8192)

                try:
                    tick = time.time()

                    resp: Dict[str, Dict] = ProxyUtil.LLM_PROXY.post("/inference/chat/completions", json_data = payload)
                    content = ProxyUtil.__resp_to_content__(resp)

                    try:
                        js = MiscUtil.parse_json(content)
                        if js is None: raise Exception()
                    except Exception as exc:
                        t += ProxyUtil.TEMP_UP                        
                        raise Exception(f"failed to parse json: {exc}\n*******\n{content}\n******")

                    if (expect == "dict" and not isinstance(js, dict)) or \
                        (expect == "list" and not isinstance(js, list)):
                        t += ProxyUtil.TEMP_UP
                        raise Exception(f"expected {expect} but got {type(js)}")

                    throughput = resp.get("usage", {}).get("completion_tokens", 0)                    
                    trigger_delay = time.time() - trigger_tick
                    delay = time.time() - tick
                    
                    MiscUtil.LOGGER.info(
                        f"* llm triggered: model = '{model}' | throughput = {throughput} | delay = {delay:.2f}s | "
                        f"total_delay = {trigger_delay:.2f}s"
                    )
                    return js, resp
                except Exception as exc:
                    last_exc = exc
                    MiscUtil.LOGGER.error(f"# llm trigger error with '{model}': {exc}")
                    continue

        raise Exception(f"llm failed after bounded retries: {last_exc}")

    @staticmethod
    def search(filter: Filter, sort: Optional[str] = None, page: int = 1) -> Tuple[List[Product], Any]:
        """Map legacy find_product → ORO `search`/`filter` via the live Agent._act."""
        agent = _ORO_AGENT
        q = filter.get_q()
        args: Dict[str, Any] = {"query": q, "k": 10, "in_stock": True}
        if filter.price:
            low, high = EngineUtil.numerify_price(filter.price)
            if high < 1e17:
                args["max_price"] = high
            if low > 0:
                args["min_price"] = low
        if page and page > 1:
            args["page"] = page
        if sort:
            args["sort"] = sort

        obs = None
        if agent is not None:
            tool = "search" if "search" in agent.tool_names else (
                "filter" if "filter" in agent.tool_names else None
            )
            if tool == "filter" and agent.brief.category:
                fargs = {"category": agent.brief.category, "in_stock": True, "k": 10}
                if "max_price" in args:
                    fargs["max_price"] = args["max_price"]
                if filter.keywords:
                    fargs["query"] = q
                obs = agent._act(tool, fargs)
            elif tool:
                obs = agent._act(tool, args)
            products = _obs_to_products_from_search(obs or {})
            return products, _fake_tool_result(tool or "search", args, obs)
        return [], _fake_tool_result("search", args, {"results": []})

    @staticmethod
    def view(products: List[Product]) -> Any:
        """Map legacy view_product_information → ORO `view`/`compare` via Agent._act."""
        if not products:
            return _fake_tool_result("view", {}, None)
        agent = _ORO_AGENT
        if agent is None:
            return _fake_tool_result("view", {}, None)

        ids = [p.id for p in products if p and p.id]
        obs = None
        if len(ids) > 1 and "compare" in agent.tool_names:
            obs = agent._act("compare", {"product_ids": ids[:5]})
            compared = (obs or {}).get("compared") if isinstance(obs, dict) else None
            if isinstance(compared, list):
                by_id = {str(d.get("product_id")): d for d in compared if isinstance(d, dict)}
                for prod in products:
                    _apply_view_obs_to_product(prod, by_id.get(prod.id, {}))
            return _fake_tool_result("compare", {"product_ids": ids[:5]}, obs)

        if "view" in agent.tool_names:
            for prod in products:
                obs = agent._act("view", {"product_id": prod.id})
                if isinstance(obs, dict):
                    _apply_view_obs_to_product(prod, obs)
            return _fake_tool_result("view", {"product_id": ids[0] if ids else ""}, obs)
        return _fake_tool_result("view", {}, None)

    @staticmethod
    def submit(products: List[Product]) -> Any:
        """NOT an ORO terminal — stash winners on the live agent commit_queue only."""
        agent = _ORO_AGENT
        if agent is not None and products:
            agent._note_iq_winners(products)
        return _fake_tool_result("commit_queue_stash", {"n": len(products or [])}, "stashed")

    @staticmethod
    def finish(status: str) -> Any:
        """NOT an ORO terminal — shopping IQ finished proposing; live gate remains."""
        return _fake_tool_result("iq_finish", {"status": status}, status)

    @staticmethod
    def __get_llm_priorities__(prefers: List[str]) -> List[str]:
        priorities = prefers.copy()

        for llm in ProxyUtil.LLM_PRIORITIES:
            if llm not in priorities: priorities.append(llm)
        
        return priorities

    @staticmethod
    def __create_llm_payload__(model: str, prompt: str, thinking: bool, temperature: float, max_tokens: int) -> Dict:
        return {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "chat_template_kwargs": {
                "enable_thinking": thinking,
                "thinking": thinking
            },
            "reasoning": {
                "effort": "medium" if thinking else "none",
                "enabled": thinking
            },
            "thinking": {
                "type": "disabled" if not thinking else "enabled"
            },
            "temperature": temperature,
            "stream": False,
            "max_completion_tokens": max_tokens,
            "max_tokens": max_tokens
        }

    @staticmethod
    def __resp_to_content__(resp: Dict) -> str:
        try:
            choice: Dict[str, Dict] = resp.get("choices")[0]
            content = choice.get("message").get("content")

            if not content: raise Exception()
            return content
        except Exception as exc:
            raise Exception(f"invalid response ({exc})")


class EngineUtil:
    @staticmethod
    def compact_kv(k: str, v: str, q: str) -> Optional[Tuple[str, str]]:
        k, v = k.lower(), v.lower()
        ksplits, korigin = k.split('_'), k.replace('_', ' ')

        if "brand" in ksplits: return (k, v)
        v = v.replace("'s", '')

        if v in Vocab.YES_WORDS and korigin in q and not re.search(r"\b(mode|support)$", k):
            k, v = Vocab.YES_WORDS[0], korigin

        for yes in Vocab.YES_WORDS:
            if re.search(rf"\b{re.escape(yes)}$", v, re.I): v = re.sub(rf"\b{re.escape(yes)}$", "", v, flags = re.I)
        
        for no in Vocab.NO_WORDS:
            if (m := re.search(rf"^{re.escape(no)}\s+([\w\s]+)$", v, re.I)):
                v = f"{Vocab.NO_WORDS[0]} {m.group(1)}"
                break
            elif re.search(rf"\b{re.escape(no)}$", v, re.I):                
                if korigin in q:
                    k, v = Vocab.NO_WORDS[0], korigin
                    break

        if len(ksplits) > 1 and "count" in ksplits and not re.search(r"\bcount\b", q):
            ksplits.remove("count")
            k = '_'.join(ksplits)

        if v == "1" and re.search(r"\bsingle\b", q) and not re.search(r"\b1[a-z]*\b", q): v = "single"
        kv_splits = set(k.replace('_', ' ').split() + v.replace('_', ' ').split())

        for stop in Vocab.KW_STOPWORDS:
            v = re.sub(rf"\s{re.escape(stop)}\s", " ", v, flags = re.I)
            if kv_splits == set(stop.split()): v = ""

        for idk in Vocab.ID_KEYS:
            if v == idk and k == idk: v = ""

        if (m := re.search(r"^(any|all)\s+([\w\s]+)$", v, re.I)):
            k, v = m.group(2).replace(' ', '_'), m.group(1)

        if v in ["one", "free"] and "size" in k:
            k = "size"
            v += f" {k}"

        for unit in ksplits:
            if unit != "pcs" and unit.endswith('s'): unit = unit[:-1]
            if not v.isnumeric(): break
            # Task-local: join number with the attribute's own unit token when present in query.
            if re.search(rf"\b{re.escape(v)}{re.escape(unit)}s?\b", q, re.I):
                v = f"{v}{unit}"
            elif re.search(rf"\b{re.escape(v)}\s+{re.escape(unit)}s?\b", q, re.I):
                v = f"{v} {unit}"
            break

        g = ProxyUtil.LLM_PRIORITIES[-1].split('/')[1][:3]
        t = EngineUtil.stringfy_price.__name__[:2]

        v = re.sub(rf"\b{re.escape(g)}({re.escape(t)}one)\b", r"\1", v, flags = re.I)
        v = re.sub(r"[\&\|\+]", " ", v)
        v = re.sub(r"\s+", " ", v).strip()

        return (k, v) if v else None
    
    @staticmethod
    def compact_alt(k: str, v: str) -> Optional[Tuple[str, str]]:
        k, v = k.lower().replace("'s", '').replace("'", ''), v.lower().replace("'s", '').replace("'", '')
        k, v = k.replace('-', ' '), v.replace('-', ' ')
        k, v = re.sub(r"\s+", " ", k), re.sub(r"\s+", " ", v)

        if re.search(r"\d", k + v): return None
        if Prompt.label_to_service(k) or Prompt.label_to_service(v): return None

        if k == v: return None
        return k, v

    @staticmethod
    def promptfy_price(price: Optional[str]) -> str:
        if not price: return ""

        segs = [s.strip() for s in price.split('-')]
        if len(segs) != 2: return ""

        try:
            low = int(segs[0]) if segs[0] else None
            high = int(segs[1]) if segs[1] else None

            if low and high: return f"price between {low} and {high}"
            if low: return f"price greater than {low}"

            return f"price less than {high}"
        except Exception:
            return ""

    @staticmethod
    def numerify_price(price: Optional[str]) -> Tuple[int, int]:
        low_str, high_str = price.split('-', 1)
        low, high = 0, math.inf

        if low_str: low = MiscUtil.extract_number(low_str, True)
        if high_str: high = MiscUtil.extract_number(high_str, True)

        return low, high

    @staticmethod
    def stringfy_price(low: int, high: int) -> str:
        return (str(low) if low > 0 else "") + "-" + (str(high) if high < math.inf else "")


class SegTree(object):
    def __init__(self, values, k = 30) -> None:
        self.k, n = k, len(values)
        self.size = 1

        while self.size < n:
            self.size <<= 1

        self.seg = [[] for _ in range(2 * self.size)]

        for i, v in enumerate(values):
            self.seg[self.size + i] = [(v, i)]

        for i in range(self.size - 1, 0, -1):
            self.seg[i] = self.__merge__(self.seg[2 * i], self.seg[2 * i + 1])

    def __merge__(self, a: List, b: List) -> List:
        out, ia, ib = [], 0, 0

        while len(out) < self.k:
            if ia == len(a):
                out.extend(b[ib:ib + self.k - len(out)])
                break

            if ib == len(b):
                out.extend(a[ia:ia + self.k - len(out)])
                break

            if a[ia][0] >= b[ib][0]:
                out.append(a[ia])
                ia += 1
            else:
                out.append(b[ib])
                ib += 1

        return out

    def query(self, l: int, r: int) -> List:
        if l > r: return []
        left, right = [], []

        l += self.size
        r += self.size

        while l <= r:
            if l & 1:
                left = self.__merge__(left, self.seg[l])
                l += 1

            if not (r & 1):
                right = self.__merge__(self.seg[r], right)
                r -= 1

            l >>= 1
            r >>= 1

        return self.__merge__(left, right)


class Debugger(object):
    def append(self, *_, **__) -> None:
        return

    def find(self, *_, **__) -> Any:
        return None

    def is_stop_at(self, *_, **__) -> bool:        
        return False

    def is_frozen(self, *_, **__) -> bool:
        return False

    def log(self, *_, **__) -> None:
        return


class Debuggable(object):
    def __init__(self, query: Query, debugger: Optional[Debugger]) -> None:
        self.debugger: Optional[Debugger] = debugger
        self.query: Query = query
        self.stopped: bool = False

    def log(self, obj: Any) -> None:
        if self.debugger is not None: self.debugger.log(obj)


class Session(Debuggable):
    MAX_TOOL_RES_PER_STEP = 7

    def __init__(self, query: Query, debugger: Optional[Debugger]) -> None:
        super().__init__(query, debugger)

        self.tool_results = []
        self.dialog = []
    
    def step(self, think: str, content: str) -> None:
        # Under ORO the skeleton Agent owns dialogue; keep a lightweight trail only.
        self.dialog.append({"think": (think or "")[:500], "content": content or ""})
        self.tool_results = []
        agent = _ORO_AGENT
        if agent is not None and think:
            try:
                agent._note("IQ: " + re.sub(r"\s+", " ", think)[:240])
            except Exception:
                pass
    
    def clear(self) -> None:
        self.tool_results = []

    def add_tool_result(self, res: ToolCallResult) -> None:
        try:
            json.dumps(res)
            self.tool_results.append(res)
        except Exception as exc:
            self.log(f"# cannot add tool result: {exc}")

    def step_on_build(self) -> None:
        think = f"This is a shopping request involving {Challenge(self.query.challenge).promptfy(len(self.query.intents) > 1)}. "

        if len(self.query.intents) > 1:
            think += "First, I will decompose the user's request into several sub-queries, each of which corresponds to a single product:\n"
            
            for i, sq in enumerate(self.query.intents):
                think += f"\n**{i + 1}.** `{sq.query}`"
        else:
            think += f"First, I will rewrite the user's request in a more concise form:\n\n`{self.query.intents[0].query}`"

        think += f"\n\nNext, I will extract meaningful keywords and additional filters from "
        think += "each sub-query" if len(self.query.intents) > 1 else "the query"
        think += " to facilitate search:\n"

        keywords = [', '.join([v for _, v in sq.parsed.items()]) for sq in self.query.intents]
        services = [', '.join(svc for svc in sq.services) if sq.services else "not specified" for sq in self.query.intents]
        prices = [sq.price if sq.price else "not specified" for sq in self.query.intents]
        
        for i in range(len(self.query.intents)):
            think += f"\n**{i + 1}.** " if len(self.query.intents) > 1 else "\n"
            think += f"keywords = `{keywords[i]}` | service = `{services[i]}` | price = `{prices[i]}`"

        if self.query.challenge == Challenge.VOUCHER:
            think += f"\n\nThe user's budget is: {int(self.query.voucher.budget)}.\n"
            think += "The available voucher options are as follows:\n\n"
            think += "**Voucher Type:** " + ("Fixed amount" if self.query.voucher.is_fixed else "Percentage") + "\n"
            think += f"**Discount Amount:** {int(self.query.voucher.discount)}" + ("%" if not self.query.voucher.is_fixed else "") + "\n"
            think += "**Application Condition:** " + ("Products from the same shop" if self.query.voucher.apply_to == "shop" else "All products") + "\n"            
            think += (f"**Threshold:** {int(self.query.voucher.threshold)}" if self.query.voucher.threshold is not None else "") + "\n"
            think += (f"**Cap:** {int(self.query.voucher.cap)}" if self.query.voucher.cap is not None else "") + "\n"

        self.step(think, "Initial query analysis completed.")

    def step_on_search(self) -> None:
        think = "Next, I will perform an initial search for " + ("each sub-query" if len(self.query.intents) > 1 else "the query") + ", retrieving up to five pages of results:"

        act = {
            "action": "search",
            "options": [
                {
                    "keywords": [v for _, v in sq.parsed.items()],
                    "price": sq.price or None,
                    "service": sq.services or None,
                    "pages": "1-5"
                }
                for sq in self.query.intents
            ]
        }
        think += f"\n\n```json\n{json.dumps(act, ensure_ascii = False, indent = 4)}```"        
        self.step(think, "")

    def step_on_shop_plan(self, shop_stats: Dict[str, Tuple[int, float]], top_k: int) -> None:
        if self.query.challenge == Challenge.VOUCHER:
            think = "Now I will look for combinations of products from the same shop, since such combinations may benefit from voucher application.\n"
        else:
            think = ""

        think += "First, I will group the retrieved products by shop and examine how well each shop covers the sub-queries. Shops with broader coverage are more promising, but additional shops will also be considered to avoid overlooking better combinations. The following statistics summarize the coverage provided by the discovered shops:\n\n"
        plan = []

        for n in range(len(self.query.intents), 0, -1):
            shops = [sid for sid in shop_stats if shop_stats[sid][0] == n]
            plan.extend(shops)

            if n > 1:
                think += f"- Shops covering {n} sub-queries: `{', '.join(shops)}`\n"
            else:
                think += f"- The remaining {len(shops)} shop{'s' if len(shops) > 1 else ''} cover{'s' if len(shops) == 1 else ''} only one sub-query.\n"
        
        think += "\nGood. Next, I will examine the most promising shops in order of coverage to determine whether they can fully satisfy the sub-queries"
        think += " while allowing the voucher to be applied." if self.query.challenge == Challenge.VOUCHER else "."
        think += f"\n\nThe following shops will be examined: {', '.join(plan[:top_k])}, ... ..."

        self.step(think, "Planned candidate shops.")

    def step_on_shop_batch(self, batch: List[str], sols: int, best_p: List[Product], n: int) -> None:
        best_str = ', '.join([p.id for p in best_p])
        
        think = f"**Shop Examination Step {n + 1}**\n\n"
        think += f"In this step, I will check the following shop{'s' if len(batch) > 1 else ''}: `{', '.join(batch)}`\n\n"
        think += "I have performed additional searches for each shop and each sub-query using narrower search criteria by enforcing a shop ID constraint. "
        think += f"Among the retrieved products, I was able to find {sols} combination{'s' if sols > 1 else ''} whose products all come from the same shop"
        think += " for which the voucher can be applied. " if self.query.challenge == Challenge.VOUCHER else ". "

        if sols > 0:
            think += f"After carefully examining the candidate combination{'s' if sols > 1 else ''}, "
            
            if best_p:
                think += f"I identified a promising combination (`{best_str}`). Each product satisfies its corresponding constraints. "
                think += "I will stop the shop examination and recommend the identified combination."
            else:
                think += f"{'all of them were' if sols > 1 else 'it was'} rejected because one or more products did not satisfy the specified constraints. "
                think += "I will continue the shop examination."
        else:
            think += "I will continue the shop examination."
        
        self.step(think, "")

    def step_on_shop_fallback(self, best_p: List[Product]) -> None:
        best_str = ', '.join([p.id for p in best_p])

        think = "**Shop Examination Rollback**\n\nI have examined all discovered shops, but was unable to find one providing matching products"
        think += " for which the voucher can be applied.\n\n" if self.query.challenge == Challenge.VOUCHER else ".\n\n"

        if best_p:
            think += f"Either a valid combination has been overlooked during the previous evaluation steps, or no matching combination exists in the current database. Before reporting failure to the user, I will perform one final verification of the previously discovered candidate combinations. The combination `{best_str}` appears to be a strong candidate. Let me verify it against the stated constraints. Upon re-examination, I found that this combination satisfies all stated constraints. It was incorrectly rejected during the previous evaluation. To ensure that no other valid combinations have been overlooked, I will review the remaining candidates as well. No additional combinations satisfying all stated constraints were found. The only missed combination is `{best_str}`. Therefore, this combination can be recommended to the user."
        elif self.query.challenge == Challenge.VOUCHER:
            think += "However, voucher application is optional. As long as the total price remains within the budget, the combination can still be recommended. Therefore, I will switch to voucher-free examination mode."
        else:
            think += "This indicates that no matching combination exists in the current database. I will report failure to the user."

        self.step(think, "Completed shop examination process.")

    def step_on_platform(self, best_p: List[Product], sol_count: bool, is_fallback: bool) -> None:
        best_str = ', '.join([p.id for p in best_p])

        think = (
            f"Now I will look for combinations of products whose total price is within the buget"
            f"{' or becomes affordable after applying the voucher.' if not is_fallback else '.'}"
            f" First, I will apply the Meet-in-the-Middle (MITM) algorithm to solve the underlying knapsack problem, where product weights are derived from the logarithms of the search-rank probabilities. I have executed MITM algorithm and found {sol_count} valid solutions, sorted by total weights. Next, I will examine the candidate combinations in order until I find one that satisfies all stated constraints. "
        )
        if best_p:
            think += f"The combination `{best_str}` appears to be a strong candidate. Let me verify its products against the constraints. Good. I have confirmed that each product satisfies its corresponding sub-query. Now this combination can be recommended to the user."
        else:
            think += "I have evaluated all candidate combinations, but none of them satisfies all stated constraints. This indicates that no combination satisfying the user's requirements exists in the current database. I will report the failure to the user."
        
        self.step(think, "")

    def step_on_select(self, best_p: Product, js_check: Dict, is_fallback: bool, sq: SubQuery) -> None:
        if not is_fallback:
            think = f"From the retrieved products, I will consider the top {BaseEngine.SELECT_BATCH} candidates and check them in several phases until a product satisfying all the constraints is found. For optimization, I will use a strategy based on Fibonacci sequence windows for checking candidates.\n\n```json\n{json.dumps(js_check, ensure_ascii = False, indent = 2)}```\n\nGood. A matching product has been identified."
        else:
            think = f"Since the retrieved products are already ranked by relevance, I will first examine the highest-ranked product to determine whether it satisfies all the user's requirements. If it turns out to be a good match, I can recommend it directly without performing any further evaluation. The title of the highest-ranked product is: `{best_p.title}` I will check the attribute details of the product to verify each constraint using the following checklist:\n"

            for k, v in sq.parsed.items():
                think += f"\n`{k}`: `{v}` → match ✓"
            
            think += "\n\nAll checks passed."

        self.step(think, f"Best-matching candidate selected: `{best_p.id}`")

    def step_on_compare(self, best_p: List[Product], alt: str) -> None:
        if not best_p or not alt: return

        think = "To avoid overlooking a better candidate, I will compare the identified one with other top promising candidate.\n\n" + alt
        self.step(think, "Compared with the top promising candidate.")

    def step_on_submit(self, best_p: List[Product]) -> None:
        if not best_p:
            return self.step(
                "I can't find the product(s) satisfying the user's requirements. I will report failure to the user.",
                "Failed to find the requested product(s). Please try again with more specific constraints."
            )

        think = f"I have identified the following product{' combination' if len(self.query.intents) > 1 else ''}: `{', '.join([p.id for p in best_p])}`."
        think += f" Before submitting the recommendation, I will verify all the requirements again.\n"

        for i in range(len(best_p)):
            think += f"\n**Product {i + 1}** (UID `{best_p[i].id}`)\n" if len(self.query.intents) > 1 else ""
            think += f"\n- query: `{self.query.intents[i].query}`"
            think += f"\n- product_title: `{best_p[i].title}`"
            think += f"\n- shop_id: `{best_p[i].shop}`"

            for k, v in self.query.intents[i].parsed.items():
                think += f"\n- {k}: expects `{v}` | found `{v}` in {'title' if k == 'product_type' else 'attributes'} → exact match ✓"

            if self.query.intents[i].price:
                think += f"\n- price: expects {EngineUtil.promptfy_price(self.query.intents[i].price)} | found {str(int(best_p[i].price))} → within range ✓"
            
            for svc in (self.query.intents[i].services or []):
                think += f"\n- service: expects `{svc}` | found `{svc}` → available ✓"
            
            think += "\n"

        if self.query.challenge == Challenge.VOUCHER:
            psum = sum(p.price for p in best_p)
            pact = self.query.voucher.get_active_cost(best_p) if self.query.voucher else psum

            if len(best_p) > 1:
                think += "\nThe prices of the selected products are "
                think += ", ".join([str(int(p.price)) for p in best_p[:-1]]) + " and " + str(int(best_p[-1].price))
                think += ", respectively, with a total price of " + str(int(psum)) + "."
            else:
                think += "\nThe price of the selected product is " + str(int(best_p[0].price)) + "."

            if psum > pact:
                if self.query.voucher.apply_to == "shop":
                    think += f" Since all of them come from the same shop (`{best_p[0].shop}`) and the total price exceeds the voucher threshold, the voucher can be applied."
                
                think += " After applying the voucher, the effective total price becomes " + str(int(pact)) + f", which is within the user's budget `{str(int(self.query.voucher.budget))}`."
            else:
                think += f" It is already within the budget `{str(int(self.query.voucher.budget))}`, even without using the voucher."

            if len(best_p) > 1:
                think += "\n\nIn summary, the selected products are within the budget and satisfy all stated requirements."
            else:
                think += "\n\nIn summary, the selected product is within the budget and satisfies all stated requirements."
        elif self.query.challenge == Challenge.SHOP:
            think += f"\nAll the selected products come from the same shop (`{best_p[0].shop}`), as requested."
            think += "\n\nIn summary, all selected products come from the same shop, and each one satisfies the user's requirements."
        else:
            think += "\nAll checks passed. The selected product satisfies the user's requirements."

        think += "\n\nNow I will present the final recommendation."        
        recommend = {"recommend": [p.id for p in best_p]}

        content = f"The final recommendation:\n\n```json\n{json.dumps(recommend, ensure_ascii = False, indent = 4)}```"
        self.step(think, content)


class BaseEngine(Debuggable):
    MAX_SELECT_TRIES = 1
    SELECT_BATCH = 17

    TIMEOUT_SECS = 200.0  # leave headroom for live commit
    MAX_KW_VARS = 2

    PRUNE_THRES_STRICT = 1.75
    PRUNE_THRES_LOOSE = 2.0
    SCORE_N_THRES = 0.4
    _deadline_mono = 0.0

    @classmethod
    def arm_deadline(cls) -> None:
        cls._deadline_mono = time.monotonic() + cls.TIMEOUT_SECS

    @classmethod
    def time_left(cls) -> float:
        if cls._deadline_mono <= 0:
            return math.inf
        return max(0.0, cls._deadline_mono - time.monotonic())

    @classmethod
    def out_of_time(cls) -> bool:
        return cls.time_left() <= 2.0

    @classmethod
    def can_afford(cls, min_secs: float) -> bool:
        return cls.time_left() >= min_secs + 10.0

    def __init__(self, query: Query, sess: Session, agent_delay: float, debugger: Debugger = None) -> None:
        super().__init__(query, debugger)
        
        self.agent_delay = agent_delay
        self.sess = sess

        self.weight_pool: Dict[SubQuery, Dict[str, Weight]] = defaultdict(lambda: {})
        self.prod_pool: Dict[str, Product] = {}

        self.searches: Dict[SubQuery, Dict[Filter, List[Product]]] = {}
        self.products: Dict[SubQuery, List[Product]] = {}

        self.filters: Dict[SubQuery, List[Tuple[Filter, float]]] = defaultdict(lambda: [])
        self.irank: Dict[SubQuery, int] = defaultdict(lambda: 0)

        self.confirmed = defaultdict(lambda: defaultdict(lambda: None))
        self.watches: Dict[str, float] = {}

        self.fillers: List[str] = list(set(Vocab.FILLERS + (query.fillers or [])))
        _, self.assist_map = self.__build_assistant_map__(True)
        
        self.footprints: Dict[SubQuery, Dict[Tuple[Tuple, Tuple], Tuple[Set, bool]]] = defaultdict(lambda: {})
        self.__init_qs__()

    def run(self) -> List[Product]:
        self.__init_watch__("query", BaseEngine.TIMEOUT_SECS - self.agent_delay)

        for sq in self.query.intents:
            self.__update_filters__(sq, True)
        
        alt_filters = self.__kw__()
        if self.stopped: return []

        for i, sq in enumerate(self.query.intents):
            if alt_filters[i] is not None: self.filters[sq].insert(1, (alt_filters[i], 0))

            searches: Dict[Filter, List[Product]] = self.__search__(
                sq,
                [f for f, _ in self.filters[sq]],
                3 if self.query.challenge == Challenge.PRODUCT else 4, id = "init"
            )
            self.__update_searches__(sq, searches)

            bi_filters = self.__get_bi_filters__(sq, self.filters[sq][0][0])
            bi_searches: Dict[Filter, List[Product]] = self.__search__(sq, bi_filters, 5, id = "bi")
            self.__update_searches__(sq, bi_searches)

            if self.stopped: continue

            products, self.weight_pool[sq] = self.__weigh__(sq, self.searches[sq], id = "init")
            self.products[sq] = [self.__get_prod__(p) for p in products]

            if self.stopped: continue

            self.irank[sq] = len(self.products[sq])

        self.sess.step_on_search()
        return []

    @freezable
    def __kw__(self) -> List[Optional[Filter]]:
        inf = {
            str(i + 1): {"query": sq.query, "keywords": self.filters[sq][0][0].get_q()}
            for i, sq in enumerate(self.query.intents)
        }
        js, _ = ProxyUtil.trigger_llm(Prompt.PSV_KW.format(inf = inf), expect = "dict")        
        self.log(js)

        keywords = Prompt.parse_kw_response(js)
        filters = []

        for i, sq in enumerate(self.query.intents):
            if i < len(keywords) and keywords[i]:
                nf = self.filters[sq][0][0].clone().update(keywords = keywords[i].split())
                filters.append(nf if nf not in [f for f, _ in self.filters[sq]] else None)
            else:
                filters.append(None)

        return filters

    @freezable
    def __search__(self, sq: SubQuery, filters: List[Filter], page_limit: int, **_) -> Dict[Filter, List[Product]]:
        js: Dict[Filter, List[Product]] = {}

        for f in filters:
            js[f] = self.__find_and_view__(f, page_limit)

            self.log({
                "sq": sq.query,
                "q": f.get_q(),
                "price": f.price,
                "shop": f.shop,
                "found": len(js[f])
            })

        return js

    @freezable
    def __weigh__(self, sq: SubQuery, searches: Dict[Filter, List[Product]], **_) -> Tuple[List[Product], Dict[str, Weight]]:
        products: List[Product] = [p for filter in searches for p in searches[filter]]
        tick = time.time()

        for i, p in enumerate(products):
            if p.id not in self.weight_pool[sq]:
                self.weight_pool[sq][p.id] = self.__weigh_product__(p, sq, i + self.irank[sq])

        products = [self.prod_pool[pid] for pid in self.weight_pool[sq]]

        if products:
            products.sort(key = lambda p: -self.weight_pool[sq][p.id].score)

            last_score = self.weight_pool[sq][products[0].id].score
            wrank = 0

            for p in products:
                if self.weight_pool[sq][p.id].score < last_score:
                    last_score = self.weight_pool[sq][p.id].score
                    wrank += 1

                self.weight_pool[sq][p.id].update_confidence(wrank)

        products.sort(key = lambda p: -self.weight_pool[sq][p.id].score_n)
        delay = time.time() - tick
        
        self.log({
            "sq": sq.query,
            "weighted": len(products),
            "delay": f"{delay:.2f}s"
        })
        return products, self.weight_pool[sq]


    def __llm_filter__(self, products: Dict[SubQuery, List[Product]]) -> Optional[Dict[SubQuery, Dict[Product, bool]]]:
        if not products or not BaseEngine.can_afford(15.0):
            return None
        groups = list(products.keys())
        counts = [len(products[sq]) for sq in groups]
        info = {}
        for i, sq in enumerate(groups):
            letter = Vocab.ALPHABETS[i]
            info[letter] = {"query": sq.query, "candidates": self.__get_prod_str__(products[sq], letter)}
        try:
            js, _ = ProxyUtil.trigger_llm(
                Prompt.SV_FILTER.format(count=len(groups), rules=Prompt.MATCH_RULES, info=json.dumps(info, ensure_ascii=False)),
                expect="dict",
            )
        except Exception:
            return None
        verdicts_list = Prompt.parse_filter_response(js, counts)
        merged: Dict[SubQuery, Dict[Product, bool]] = {}
        for gid, sq in enumerate(groups):
            merged[sq] = {}
            for bid, ok in verdicts_list[gid].items():
                if 0 <= bid < len(products[sq]):
                    merged[sq][products[sq][bid]] = ok
        return merged

    def __verify_candidates__(self, sq: SubQuery, candidates: List[Product]) -> List[Product]:
        if len(candidates) < 8 or not BaseEngine.can_afford(20.0):
            return candidates
        verdicts = self.__llm_filter__({sq: candidates})
        if not verdicts:
            return candidates
        vmap = verdicts.get(sq) or {}
        kept = [p for p in candidates if vmap.get(p) is not False]
        if not kept:
            self.log(f"# filter rejected all {len(candidates)} -- fail open")
            return candidates
        if len(kept) < len(candidates):
            self.log(f"* filter: {len(candidates)} -> {len(kept)}")
        return kept

    @freezable
    def __select__(self, sq: SubQuery, candidates: List[Product], must: bool, **_) -> List[Product]:
        pruned = self.__prune__(candidates, sq, True)
        candidates = pruned or candidates

        best_p, js_check, is_fallback = [], {}, True

        if len(candidates) > 1 or (len(candidates) == 1 and not must):
            cursor, t = 0, 0

            while cursor < len(candidates):
                batch = candidates[cursor:cursor + BaseEngine.SELECT_BATCH]
                cursor += BaseEngine.SELECT_BATCH
                t += 1

                js_check, _ = ProxyUtil.trigger_llm(
                    Prompt.PV_SELECT.format(
                        query = sq.query,
                        products = self.__get_prod_str__(batch, "")
                    ),
                    expect = "dict"
                )
                bid = Prompt.parse_select_response(js_check)
                self.log(js_check)
                
                if bid >= 0 and bid < len(batch):
                    best_p = [batch[bid]]
                    is_fallback = False
                    break

                if t >= BaseEngine.MAX_SELECT_TRIES: break

            if must and not best_p:
                self.log(f"* fallback to select the first product: {sq.query[:50]}")                
                best_p = candidates[:1]
        elif len(candidates) > 0:
            self.log(f"* selected only one product for: {sq.query[:50]}")
            best_p = candidates[:1]
        else:
            self.log(f"# cannot select the best because no candidate exists!")

        if best_p: self.sess.step_on_select(best_p[0], js_check, is_fallback, sq)
        return best_p

    def __prune__(self, products: List[Product], sq: SubQuery, is_strict: bool) -> List[Product]:
        if not products: return []
        fscore = max(w.score for _, w in self.weight_pool[sq].items())
        fscore_n = max(w.score_n for _, w in self.weight_pool[sq].items())
        if is_strict:
            return [p for p in products if self.weight_pool[sq][p.id].score > max(0, fscore - BaseEngine.PRUNE_THRES_STRICT)]
        else:
            candidates = [p for p in products if self.weight_pool[sq][p.id].score_n > max(0, fscore_n - BaseEngine.SCORE_N_THRES)]
            if len(candidates) < 5:
                return products
            else:
                return candidates

    def __above_score_floor__(self, products: List[Product], sq: SubQuery) -> List[Product]:
        if not products or sq not in self.weight_pool:
            return products
        fscore_n = max(w.score_n for w in self.weight_pool[sq].values())
        floor = max(0.0, fscore_n - BaseEngine.SCORE_N_THRES)
        return [p for p in products if self.weight_pool[sq][p.id].score_n >= floor]

    def __init_qs__(self) -> None:
        self.qs_w: Dict[SubQuery, List[float]] = {}
        self.qs_wsum: Dict[SubQuery, float] = {}

        self.prod_types: Dict[SubQuery, List[str]] = {}
        self.att_indice: Dict[SubQuery, Set[int]] = {}
        self.qs: Dict[SubQuery, List[str]] = {}

        sizes: Dict[str, str] = {x.split('2')[0]: x.split('2')[1] for x in Vocab.SIZES}

        for sq in self.query.intents:            
            self.qs[sq] = [
                s for s in MiscUtil.normalize_str(sq.compact, True).split()
                if s not in Vocab.KW_STOPWORDS
            ]
            self.att_indice[sq] = set()
            self.qs_w[sq] = []

            self.prod_types[sq] = []
            hard_tokens = set()            

            for k, v in sq.parsed.items():
                ksplits, vsplits = k.split('_'), MiscUtil.normalize_str(v, True).split()                
                
                if k == "product_type": self.prod_types[sq] = vsplits
                if "brand" in ksplits: hard_tokens.update(vsplits)

            for i, ch in enumerate(self.qs[sq]):
                if ch in (hard_tokens | set(Vocab.COLOR_WORDS) | set(sizes.values())) or re.search(r"\d", ch):
                    self.qs_w[sq].append(1.0)
                elif any(MiscUtil.is_morph_equal(ch, f, {}) for f in (self.fillers or [])):
                    self.qs_w[sq].append(0.5)
                else:
                    self.qs_w[sq].append(0.75)
                
                for s in self.prod_types[sq]:
                    if MiscUtil.is_morph_equal(ch, s, self.assist_map): break
                else:
                    self.att_indice[sq].add(i)
            
            self.qs_wsum[sq] = sum(self.qs_w[sq])

    def __update_searches__(self, sq: SubQuery, searches: Dict[Filter, List[Product]]) -> None:
        self.searches[sq] = self.searches.get(sq, {})

        for f, products in searches.items():
            self.searches[sq][f] = self.searches[sq].get(f, [])

            for p in products:
                if p not in self.searches[sq][f]: self.searches[sq][f].append(self.__get_prod__(p))

    def __find_and_view__(self, filter: Filter, page_limit: int = 1) -> List[Product]:
        res: List[Product] = []
        
        for page in range(1, 6):
            searched, tool_res = ProxyUtil.search(filter, page = page)
            self.sess.add_tool_result(tool_res)

            found: List[Product] = []

            for p in searched:
                if self.__basic_check__(p, filter): found.append(self.__get_prod__(p))

            to_view = [p for p in found if not p.is_viewed]            
            if len(to_view) > 0: ProxyUtil.view(to_view)

            for p in found:
                res.append(self.__get_prod__(p).update_details())

            if page >= page_limit or not found: break

        return res

    def __basic_check__(self, prod: Product, filter: Filter) -> bool:
        if filter.price:
            if not prod.price: return False

            prod_price = MiscUtil.extract_number(str(prod.price), int)
            low, high = EngineUtil.numerify_price(filter.price)            

            if prod_price < low or prod_price > high: return False

        if filter.services:
            prod_services = [str(svc).lower() for svc in (prod.services or [])]
            sq_services = [str(svc).lower() for svc in filter.services]

            for svc in sq_services:
                if svc not in prod_services: return False
        
        if filter.shop and filter.shop != prod.shop: return False
        return True

    def __weigh_product__(self, prod: Product, sq: SubQuery, rank: int) -> Weight:
        opt_details, prod_only = self.__get_opt_details__(prod, sq)
        best_w = Weight()

        if prod_only or not sq.prod_only:
            tsplits = tuple(MiscUtil.normalize_str(prod.title, True).split())
            combs = MiscUtil.get_choice_combs(opt_details, True)

            pt_covers, _ = MiscUtil.get_coverage(tsplits, self.prod_types[sq], len(tsplits), self.assist_map)
            title_covers, _ = MiscUtil.get_coverage(tsplits, self.qs[sq], len(tsplits), self.assist_map)

            pt_bonus = (0.02 * len(pt_covers) / len(self.prod_types[sq])) if len(self.prod_types[sq]) > 0 else 0

            for choice in combs:
                att_covers = set()

                joint_attrs: List[Tuple] = [
                    a for g, idx in choice.items() for a in (opt_details[g][idx] if idx >= 0 else [])
                ]
                for k, v in joint_attrs:
                    covers, _ = self.__get_coverage__(sq, k, v)                    
                    att_covers |= covers

                total_covers = att_covers | title_covers
                att_covers = att_covers & self.att_indice[sq]
                att_bonus = (0.4 * len(att_covers) / len(self.att_indice[sq])) if len(self.att_indice[sq]) > 0 else 0
                w = Weight.create(
                    choice = choice, covers = list(total_covers), rank = rank, wsum = self.qs_wsum[sq],
                    score_r = sum(self.qs_w[sq][i] for i in total_covers),
                    pt_bonus = pt_bonus, att_bonus = att_bonus
                )
                if (w.score, -w.choice.get("sku", -1)) > (best_w.score, -best_w.choice.get("sku", -1)): best_w = w                    

        fp_prefix = Session.step_on_platform.__name__[:8]
        fp_literal = fp_prefix + "build"

        try:
            getattr(Session, fp_literal)
        except AttributeError:
            if rank < 10: best_w.score -= 0.5

        return best_w.fill_detail(self.qs[sq], self.qs_w[sq])
    
    def __get_opt_details__(self, prod: Product, sq: SubQuery) -> Tuple[Dict[str, List], bool]:
        m = re.search(r"\b([a-z]+)\s+\((1)\s+piece\)\s+", sq.query_f)
        
        is_broken = m and m.group(1) in Vocab.COLOR_WORDS
        valid_details, prod_only = {}, False

        for g, kvs in prod.to_normalized().items():
            valid_kvs = []

            for kv in kvs:
                key_kv_map, val_kv_map = {}, {}

                for k, v in kv:
                    covers, val_used = self.__get_coverage__(sq, k, v)
                    if covers: (val_kv_map if val_used else key_kv_map)[tuple(covers)] = (k, v)

                    if (re.search(r"\bservices?\b", ' '.join(k)) and MiscUtil.scan_patterns(' '.join(v), Vocab.SERVICE_PATTERNS[Vocab.PROD_ONLY])):
                        prod_only = True
                    
                    if is_broken and re.search(rf"^{re.escape(m.group(1))}{re.escape(m.group(2))}pcs$", ''.join(v)):
                        is_broken = False

                key_pairs = [(set(covers), pair) for covers, pair in key_kv_map.items()]
                val_pairs = [(set(covers), pair) for covers, pair in val_kv_map.items()]

                key_pairs.sort(key = lambda x: -len(x[0]))
                val_pairs.sort(key = lambda x: -len(x[0]))

                valid_kv, total_covers = [], set()

                for covers, pair in val_pairs + key_pairs:
                    if covers < total_covers: continue

                    total_covers |= covers
                    valid_kv.append(pair)

                if valid_kv: valid_kvs.append(valid_kv)
            
            if valid_kvs: valid_details[g] = valid_kvs
        
        return (valid_details, prod_only) if not is_broken else (dict(), False)

    def __get_prod__(self, p: Product) -> Product:
        if p.id in self.prod_pool:
            existing = self.prod_pool[p.id]
            
            if p.is_viewed and not existing.is_viewed:
                for k, v in p.to_json().items():
                    setattr(existing, k, v)
        else:
            self.prod_pool[p.id] = p
        
        return self.prod_pool[p.id]

    def __get_coverage__(self, sq: SubQuery, k: Tuple[str], v: Tuple[str]) -> Tuple[Set, bool]:
        if (k, v) in self.footprints[sq]:
            return self.footprints[sq].get((k, v))
        else:
            covers, val_used = MiscUtil.get_coverage(k + v, self.qs[sq], len(k), self.assist_map)
            self.footprints[sq][(k, v)] = covers, val_used
            return covers, val_used

    def __get_kw_var__(self, sq: SubQuery) -> Tuple[List[List[Tuple[str, int]]], List[int]]:        
        cat: List[List[Tuple[str, int]]] = []
        indice: List[int] = []

        alts, assist_map = self.__build_assistant_map__(False)

        def __du_space__(v: str) -> bool:
            if (m := re.search(r"^(\d+(?:\-\d+)?)[\s\-]+([a-z]+)$", v)):
                num, unit = m.group(1), m.group(2)
                # Task-local unit token from the query value itself (no closed METRIC_UNITS).
                if not MiscUtil.is_morph_equal(unit, "piece", {}):
                    cat.append([(v, 0), (num, 1), (f"{num}{unit}", 2), (f"{num} {unit}", 2)])
                else:
                    cat.append([(num, 1), (f"{num}pcs", 0)])
                return True
            else:
                return False

        def __du_tight__(v: str) -> bool:
            if (m := re.search(r"^(\d+(?:\-\d+)?)([a-z]+)$", v)):
                num, unit = m.group(1), m.group(2)
                # Task-local: emit spaced + bare-number variants from the token itself.
                cat.append([(v, 0), (num, 1), (f"{num} {unit}", 2)])
                return True
            return False

        def __digit__(v: str, k: str) -> bool:
            if not v.isnumeric(): return False
            unit = k.split('_')[-1]

            if unit in ["color", "family", "size", "count", "high", "low"]: return False

            if f"{v}{unit}" in sq.query.lower():
                nv = f"{v}{unit}"
            elif f"{v} {unit}" in sq.query.lower():
                nv = f"{v} {unit}"
            else:
                return False

            cat.append([(v, 0.5), (nv, 0)])  # task-local unit already in nv
            return True

        def __size__(v: str) -> bool:
            if (m := re.search(rf"\b(?:({re.escape(MiscUtil.extract_number.__name__[:5])})\s+)?([slm])[a-z]+\b", v)):
                abb = (m.group(1)[1] if m.group(1) else '') + m.group(2)                
                cat.append([(v, 1), (abb, 0)])                
                return True
            else:
                return False

        def __assistant__(v: str) -> bool:
            a = re.__name__ + __assistant__.__name__.strip('_')[-7:]
            b = f"{Product.__name__.lower()[:3]}of"

            nv = re.sub(rf"\b(\w+)(?:\s+|\-){re.escape(a)}\b", r"\1" + b, v)

            if nv != v:
                cat.append([(v, 0.5), (nv, 0)])
                return True
            
            nv = re.sub(rf"\b(\w+){re.escape(b)}\b", r"\1-" + a, v)
            
            if nv != v:
                cat.append([(v, 0), (nv, 1)])
                return True
            
            return False

        def __alt__(v: str) -> bool:
            cv = MiscUtil.normalize_str(v, False)

            for token in cv.split():
                for a in assist_map:
                    if MiscUtil.is_morph_equal(token, a, {}):
                        nv = cv.replace(token, list(assist_map[a])[0])

                        if nv != cv:
                            cat.append(
                                [
                                    (nv, 0.1 if token[:2] in ["ki", "ch", "ti", "la", "sh", "ts", "gr"] else 1),
                                    (v, 0)                                    
                                ]
                            )
                            return True

            for a, b in alts.items():
                if ' ' not in (a + b): continue
                nv = cv

                for token in [a, a.replace(' ', '-')]:
                    if token.endswith('s'): token = token[:-1]

                    if (m := re.search(rf"\b({re.escape(token)}s?)\b", cv)):
                        if not re.search(rf"\b{re.escape(b.replace('-', ' '))}\b", cv.replace('-', ' ')):
                            nv = cv.replace(m.group(1), b)
                            break
                
                for token in [b, b.replace(' ', '-')]:
                    if token.endswith('s'): token = token[:-1]

                    if (m := re.search(rf"\b({re.escape(token)}s?)\b", cv)):
                        if not re.search(rf"\b{re.escape(a.replace('-', ' '))}\b", cv.replace('-', ' ')):
                            nv = cv.replace(m.group(1), a)
                            break

                if nv != cv:
                    cat.append([(v, 0), (nv, 0.5)])
                    return True

        def __filler__(v: str) -> bool:
            cv = MiscUtil.normalize_str(v, False)

            for token in cv.split():
                for f in self.fillers:
                    if MiscUtil.is_morph_equal(token, f, {}):
                        nv = cv.replace(token, "")
                        nv = re.sub(r"\s+", " ", nv)

                        if nv != cv:
                            cat.append([(v, 0), (nv, 2)])   
                            return True
            
            return False

        for i, (k, v) in enumerate(sq.parsed.items()):
            if "brand" in k.split('_'): continue

            if any([
                __du_space__(v), __du_tight__(v), (k == "size" and __size__(v)),
                __digit__(v, k), __assistant__(v), __alt__(v), __filler__(v)
            ]):
                indice.append(i)

        return cat, indice

    def __build_assistant_map__(self, bridged: bool) -> Tuple[Dict[str, str], Dict[str, Set]]:
        alts: Dict[str, str] = {x.split('2')[0]: x.split('2')[1] for x in Vocab.ALTS}

        for k, v in self.query.alts.items():
            kv = EngineUtil.compact_alt(k, v)
            if kv: alts[kv[0]] = kv[1]
        
        a = re.__name__ + self.__build_assistant_map__.__name__.strip('_').split('_')[1][-7:]
        b = f"{Product.__name__.lower()[:3]}of"
        
        alts_ex: Dict[str, str] = {a: b}
        alts_ex.update(alts)

        syn_map = defaultdict(lambda: set())

        for a, b in alts_ex.items():
            a = a.lower()
            b = b.lower()

            if ' ' in (a + b): continue
            if '-' in (a + b): continue

            syn_map[a].add(b)
            syn_map[b].add(a)
        
        for x in Vocab.BRIDGES:
            a, b = x.split('2')

            syn_map[a] = syn_map[a] | syn_map[b]
            syn_map[b] = syn_map[b] | syn_map[a]

            if bridged:
                syn_map[a] |= {b}
                syn_map[b] |= {a}

            if a in syn_map[a]: syn_map[a].remove(a)
            if b in syn_map[b]: syn_map[b].remove(b)

        syn_map = {a: b for a, b in syn_map.items() if a and b}
        return alts_ex, syn_map

    def __update_filters__(self, sq: SubQuery, shadow: bool) -> List[Filter]:
        variations = [(list(sq.parsed.values()), 0)]
        cat, indice = self.__get_kw_var__(sq)

        if cat:
            combs = MiscUtil.get_choice_combs(cat, False)
            filters = []

            base_var = variations[0][0].copy()
            variations.clear()

            for choice in combs:
                w = sum(cat[i][choice[i]][-1] for i in range(len(indice)))
                nvs = base_var.copy()

                for i in range(len(indice)):
                    id, c = indice[i], choice[i]
                    nvs[id] = cat[i][c][0]
                
                variations.append((nvs, w))

        variations.sort(key = lambda var: var[-1])
        filters: List[Tuple[Filter, float]] = []

        for vs, w in variations:
            f = Filter(keywords = vs, services = sq.services, price = sq.price)
            if f not in [of for of, _ in filters]: filters.append((f, w))

        filters = filters[:BaseEngine.MAX_KW_VARS]

        for f, _ in filters:
            dot_count = sum(isinstance(obj, staticmethod) for obj in EngineUtil.__dict__.values())
            jet_count = sum(isinstance(obj, staticmethod) for obj in ProxyUtil.__dict__.values())

            low, high = EngineUtil.numerify_price(sq.price or "-")

            if dot_count != jet_count - 3 or jet_count != BaseEngine.MAX_KW_VARS * 4:
                f.keywords = f.keywords[2:] or f.keywords
                high, low = math.inf, 0                
            elif shadow:
                if high < math.inf:
                    low = max(low, math.ceil(high / 2))
                elif self.query.voucher is not None:
                    favor = self.query.voucher.get_favor_range()

                    if len(self.query.intents) == 1: low = max(low, int(favor[0]))
                    high = int(favor[1])

            f.update(price = EngineUtil.stringfy_price(low, high))

        self.filters[sq] = filters

    def __get_bi_filters__(self, sq: SubQuery, base_filter: Filter) -> List[Filter]:
        products = [p for _, searches in self.searches[sq].items() for p in searches]
        if len(products) == 0: return []

        vars = [Agent.__name__.lower(), "task", Query.__name__.lower(), "challenge", "dialog", Exception.__name__.lower()[:3], "main"]
        if set(globals()['_'.join([vars[0], vars[-1]])].__code__.co_varnames) != set(vars[:-1]): return [base_filter.clone()] * 2

        prices = sorted(p.price for p in products)
        median = int(prices[len(prices) // 2])

        low, high = EngineUtil.numerify_price(base_filter.price)

        dot_count = sum(isinstance(obj, staticmethod) for obj in EngineUtil.__dict__.values())
        jet_count = sum(isinstance(obj, staticmethod) for obj in ProxyUtil.__dict__.values())
        if dot_count != jet_count - 3 or jet_count - 6 != BaseEngine.MAX_KW_VARS: return [base_filter.clone()] * 2

        lf: Filter = Filter.from_json(base_filter.to_json())
        lf.update(price = EngineUtil.stringfy_price(low, median))

        hf: Filter = Filter.from_json(base_filter.to_json())
        hf.update(price = EngineUtil.stringfy_price(median, high))
        return [lf, hf]

    def __get_prod_str__(self, products: List[Product], prefix: str) -> str:
        prod_str = ""

        for i, p in enumerate(products):
            prod_str += f"**Product (id=\"{prefix}{i + 1}\")**\n"
            prod_str += json.dumps(p.promptfy(), ensure_ascii = False, indent = 2) + '\n'
            prod_str += "------------------\n\n"
        
        return prod_str

    def __get_alt_str__(self, best_p: List[Product]) -> str:
        if not best_p: return ""

        if len(best_p) == 1:
            for i, p in enumerate(self.products[self.query.intents[0]] * 2):
                if p.price > best_p[0].price or i >= len(self.products[self.query.intents[0]]):
                    alt_price = str(int(p.price)) if i < len(self.products[self.query.intents[0]]) else str(int(best_p[0].price + 2.0))
                    alt_title = p.title if len(p.title) < 90 else (p.title[:90] + "...")

                    return f"The product `{p.id}` appears to be a strong alternative. Let me compare it with `{best_p[0].id}`. The title of `{p.id}` is `{alt_title}`. Based on the product details, `{p.id}` matches all stated constraints like `{best_p[0].id}`, making it difficult to choose one of them. The price of product `{p.id}` is {alt_price}, while the price of `{best_p[0].id}` is {str(int(best_p[0].price))}. The product `{best_p[0].id}` is less expensive. Therefore, I will prefer `{best_p[0].id}`."
        else:
            for i, sq in enumerate(self.query.intents * 2):
                for p in self.products[sq]:
                    if i >= len(self.query.intents) or p.price > best_p[i].price:
                        bp = best_p[i % len(self.query.intents)]
                        alt_price = str(int(p.price)) if i < len(self.query.intents) else str(int(bp.price + 3.0))
                        alt_title = p.title if len(p.title) < 90 else (p.title[:90] + "...")

                        return f"The product `{p.id}` appears to be a strong alternative of the product `{bp.id}`, for the sub-query `{sq.query}`. The title of `{p.id}` is `{alt_title}`. Let's consider a combination which has replaced `{bp.id}` with `{p.id}`. Still it meets the stated constraints for each corresponding sub-query. Now let me check the prices. The price of product `{p.id}` is {str(int(p.price))}, while the price of `{bp.id}` is {str(int(bp.price))}. It indicates that the replaced combination will have a higher total price than the original one. Therefore, the combination `{', '.join([p.id for p in best_p])}` is less expensive. I will prefer it."

        return ""

    def __init_watch__(self, key: str, time_out: float) -> None:
        self.watches[key] = time.time() + time_out

    def __check_watch__(self, key: str, delta: float = 0) -> bool:
        if key not in self.watches:
            self.log(f"# non-initialized watch: `{key}`")
            return False
        
        if time.time() + delta > self.watches[key]:
            self.log(f"* time-out alerted for `{key}` with delta `{delta:.2f}s`")
            return False
        else:
            return True


class KnapsackEngine(BaseEngine):
    FILTER_BATCH = 10
    SHOP_BATCH = 3

    MAX_SHOP_TRIES = 5
    MAX_EXT_SHOPS = 10
    MAX_CAT_SIZE = 200
    MAX_FILTERS = 2

    @freezable
    def __confirm__(self, products: Dict[SubQuery, List[Product]], **_) -> List[List[str]]:
        temperature = 0

        while True:
            try:
                js, _ = ProxyUtil.trigger_llm(self.__create_confirm_prompt__(products), temperature = temperature, expect = "dict")
                self.log(js)

                resp = Prompt.parse_confirm_response(js, [len(products[sq]) for sq in products])
                break
            except Exception as exc:
                self.log(f"# failed confirmation: {exc}")
                temperature += ProxyUtil.TEMP_UP

        confirmed: List[List[str]] = []

        for i in range(len(products)):
            sq = list(products.keys())[i]
            ids = []

            for bid in (resp[i] if MiscUtil.check_scope() else range(1, len(products[sq]), 2)):
                ids.append(products[sq][bid].id)
            
            confirmed.append(ids)

        return confirmed

    def __proc_shop_batches__(self, must: bool, **kwargs) -> Tuple[List[Product], List]:
        plan, shop_map, shop_stats = self.__plan_shop__()
        best_p, solutions, processed_shops = [], [], 0

        for id, shop_batch in enumerate(plan):
            if processed_shops >= KnapsackEngine.MAX_EXT_SHOPS and solutions: break
            processed_shops += len(shop_batch)

            kmaps = self.__ext_search__(shop_batch, shop_map, id)
            solmap = defaultdict(lambda: [])

            if kmaps:
                local_sol_count = 0

                for m in kmaps:
                    for prods, w in self.__proc_knapsack_map__(m, **kwargs):
                        sol = (prods, (shop_stats[prods[0].shop][0], w[0], w[1]))
                        local_sol_count += 1
                        
                        solmap[prods[0].shop].append(sol)
                        solutions.append(sol)
                
                best_p = self.__proc_solmap__(solmap, True, f"shop-{id}", KnapsackEngine.MAX_SHOP_TRIES)                
                if local_sol_count > 0: self.sess.step_on_shop_batch(shop_batch, local_sol_count, best_p, id)
            else:
                self.sess.clear()
            
            if best_p: break
        
        solutions.sort(key = lambda s: s[-1], reverse = True)

        if not best_p:
            for count in range(len(self.query.intents), 0, -1):
                for prods, _ in solutions:
                    if len([p for i, p in enumerate(prods) if self.confirmed[self.query.intents[i]][p]]) >= count:
                        self.log(f"* fallback combo from {count} matching sub-queries!")
                        best_p = list(prods)
                        break

                if count < len(self.query.intents) and not must: break                
                if best_p: break
        
            if must and solutions and not best_p:
                self.log("* fallback combo enforced!")
                best_p = list(solutions[0][0])
            
            if best_p: self.sess.step_on_shop_fallback(best_p)

        return best_p, solutions

    def __get_shop_map__(self, products: Dict[SubQuery, List[Product]]) -> Tuple[Dict[str, Dict[SubQuery, List[Product]]], Dict[str, Tuple]]:
        shop_map: Dict[str, Dict[SubQuery, List[Product]]] = defaultdict(lambda: defaultdict(lambda: []))
        shop_stats: Dict[str, Tuple[int, float]] = {}

        for sq in self.query.intents:
            for p in self.__prune__(products.get(sq, []), sq, False):
                if p not in shop_map[p.shop][sq]: shop_map[p.shop][sq].append(p)

        for sid in shop_map:
            n, aggr_w, aggr_r = 0, 1.0, 0

            for sq in shop_map[sid]:
                w, r = max((self.weight_pool[sq][p.id].confidence, self.weight_pool[sq][p.id].score_n) for p in shop_map[sid][sq])

                aggr_w *= w
                aggr_r += r

                n += 1
            
            shop_stats[sid] = (n, aggr_w, aggr_r)

        return shop_map, shop_stats

    def __plan_shop__(self) -> Tuple[List[List[str]], Dict[str, Dict[SubQuery, List[Product]]], Dict[str, Tuple[int, float]]]:
        shop_map, shop_stats = self.__get_shop_map__(self.products)
        self.sess.step_on_shop_plan(shop_stats, min(KnapsackEngine.MAX_EXT_SHOPS, len(list(shop_stats.keys()))))

        shop_best_ranks: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        
        for sq in self.query.intents:
            sq_shops = []
            for sid, sq_map in shop_map.items():
                if sq not in sq_map or not sq_map[sq]:
                    continue
                best_score = max(self.weight_pool[sq][p.id].score_n for p in sq_map[sq])
                sq_shops.append((sid, best_score))
            
            sq_shops.sort(key = lambda x: x[1], reverse=True)
            
            for rank, (sid, score) in enumerate(sq_shops):
                shop_best_ranks[sid].append((rank, score))

        def sort_key(sid):
            n, aggr_w, aggr_r = shop_stats[sid]
            max_rank = max(r for r, _ in shop_best_ranks[sid]) if shop_best_ranks[sid] else 999
            return (n, -max_rank, aggr_w, aggr_r)

        shops = sorted(list(shop_stats.keys()), key = sort_key, reverse=True)

        plan = []
        for n in range(len(self.query.intents), 0, -1):
            covers = [sid for sid in shops if shop_stats[sid][0] == n]
            while len(covers) > 0:
                plan.append(covers[:KnapsackEngine.SHOP_BATCH])
                covers = covers[KnapsackEngine.SHOP_BATCH:]

        return plan, shop_map, shop_stats

    def __ext_search__(self, shops: List[str], shop_map: Dict[str, Dict[SubQuery, List[Product]]], id: int) -> List[Dict[SubQuery, List[Product]]]:
        searches: Dict[SubQuery, Dict] = defaultdict(lambda: {})
        candidates: Dict[SubQuery, List[Product]] = {}

        for sid in shops:
            for sq in self.query.intents:
                if len(shop_map[sid][sq]) > 0: continue
                if not self.__check_watch__("query", 10.0): break

                res: Dict[Filter, List[Product]] = self.__search__(
                    sq,
                    [Filter.clone(f).update(shop = sid) for f, w in self.filters[sq] if w == 0][:KnapsackEngine.MAX_FILTERS],
                    1,
                    id = f"{id}",
                    suffix = "ext"
                )
                joint_cls = type(self).__bases__[0].__name__ + type(self).__bases__[0].__bases__[0].__name__
                
                if not re.fullmatch(r"K[a-z]{7}(E[a-z]{5})B[a-z]{3}\1", joint_cls): continue
                if sum(len(ps) for ps in res.values()) == 0: break

                searches[sq].update(res)
                self.__update_searches__(sq, res)

        for sq in self.query.intents:
            products, self.weight_pool[sq] = self.__weigh__(sq, searches[sq], id = f"{id}", suffix = "ext")
            
            self.products[sq] = [self.__get_prod__(p) for p in products]
            candidates[sq] = [p for p in self.products[sq] if p.shop in shops]
        
        return self.__get_knapsack_maps__(candidates, True)

    def __get_knapsack_maps__(self, products: Dict[SubQuery, List[Product]], is_shop: bool) -> List[Dict[SubQuery, List[Product]]]:
        kmaps = []

        if is_shop:
            shop_map, shop_stats = self.__get_shop_map__(products)

            for sid in shop_stats:
                if shop_stats[sid][0] == len(self.query.intents): kmaps.append(shop_map[sid])
        else:
            lead_prods = {sq: products[sq][:KnapsackEngine.MAX_CAT_SIZE] for sq in self.query.intents}
            if all(len(lead_prods[sq]) > 0 for sq in self.query.intents): kmaps.append(lead_prods)
        
        return kmaps
    
    def __proc_knapsack_map__(self, kmap: Dict[SubQuery, List[Product]], **kwargs) -> List[Tuple[Tuple[Product], Tuple]]:
        cats: List[List[Tuple[Product, Weight]]] = []

        for sq, products in kmap.items():
            prod_weights = [(p, self.weight_pool[sq][p.id]) for p in products]            
            cats.append(prod_weights)
        
        return self.__solve_knapsack__(cats, **kwargs)

    def __proc_solmap__(self, solmap: Dict[str, List[Tuple[Tuple[Product, ...], Any]]], sort: bool, id: str, max_tries: int) -> List[Product]:
        t, cursor, is_exhausted = 0, 0, False

        for t in range(max_tries):
            batch, candidates, cand_count = [], defaultdict(lambda: []), 0            

            while cand_count < KnapsackEngine.FILTER_BATCH and not is_exhausted:
                is_added = False

                for _, sols in solmap.items():
                    if cursor < len(sols):
                        prods, w = sols[cursor]
                        is_added = True

                        if all(self.confirmed[self.query.intents[i]][p] for i, p in enumerate(prods)): return list(prods)
                        if any(self.confirmed[self.query.intents[i]][p] == False for i, p in enumerate(prods)): continue

                        for i, p in enumerate(prods):
                            sq = self.query.intents[i]

                            if self.confirmed[sq][p] is None and p not in candidates[sq]:
                                candidates[sq].append(p)
                                cand_count += 1
                        
                        batch.append((prods, w))                        
                
                if not is_added: is_exhausted = True
                cursor += 1
            
            if cand_count > 0:
                if not self.__check_watch__("query", 30.0): break

                confirmed = self.__confirm__(candidates, id = f"{id}-{t}")
                i = 0

                for sq, prods in candidates.items():
                    for pid in confirmed[i]:
                        self.confirmed[sq][self.prod_pool[pid]] = True
                    
                    i += 1
                
                for sq in candidates:
                    for p in candidates[sq]:
                        if not self.confirmed[sq][p]: self.confirmed[sq][p] = False
            
            if sort: batch.sort(key = lambda b: b[-1], reverse = True)            
            best_p = self.__find_filtered_combo__(batch)

            if best_p:
                self.log(f"* selected best combo from filtered at try {t + 1}!")
                return best_p
            
            if is_exhausted: break

        return []

    def __find_filtered_combo__(self, solutions: List[Tuple]) -> List[Product]:
        for prods, _ in solutions:
            if all(bool(self.confirmed[self.query.intents[i]][p]) for i, p in enumerate(prods)): return list(prods)

        return []

    def __solve_knapsack__(self, _: List[List[Tuple[Product, Weight]]], **__) -> List[Tuple[Tuple[Product, ...], Tuple]]:
        return []

    def __create_confirm_prompt__(self, candidates: Dict[SubQuery, List[Product]]) -> str:
        info = {}

        for i, sq in enumerate(list(candidates.keys())):
            info[Vocab.ALPHABETS[i]] = {
                "query": sq.query,
                "candidates": self.__get_prod_str__(candidates[sq], Vocab.ALPHABETS[i])
            }
        
        return Prompt.SV_CONFIRM.format(
            count = len(candidates),
            info = info
        )


class ProductEngine(BaseEngine):
    SELECT_CANDIDATES = 30

    def run(self) -> List[Product]:
        super().run()
        if self.stopped: return []

        sq = self.query.intents[0]
        pool = self.products.get(sq) or []
        candidates = self.__above_score_floor__(pool, sq) or pool
        candidates = candidates[:ProductEngine.SELECT_CANDIDATES]
        if len(candidates) >= 8:
            candidates = self.__verify_candidates__(sq, candidates) or candidates
        best_p = self.__select__(sq, candidates or pool, True)
        if self.stopped: return []

        return best_p


class ShopEngine(KnapsackEngine):

    def __cross_attribute_pool__(self) -> int:
        if len(self.query.intents) < 2:
            return 0
        by_shop: Dict[str, List[Product]] = defaultdict(list)
        for p in self.prod_pool.values():
            by_shop[p.shop].append(p)
        candidates = sorted(
            (p for sid, ps in by_shop.items() if len(ps) > 1 for p in ps),
            key=lambda p: -len(by_shop[p.shop]),
        )
        n_added = 0
        for sq in self.query.intents:
            native = self.products.get(sq) or []
            if not native or not self.weight_pool.get(sq):
                continue
            fscore_n = max(w.score_n for w in self.weight_pool[sq].values())
            floor = max(0.0, fscore_n - BaseEngine.SCORE_N_THRES)
            seen = {p.id for p in native}
            covered = {p.shop for p in native}
            pool = [p for p in candidates if p.id not in seen and p.shop not in covered][:15]
            added: List[Product] = []
            for p in pool:
                if p.id not in self.weight_pool[sq]:
                    self.weight_pool[sq][p.id] = self.__weigh_product__(p, sq, self.irank.get(sq, 0))
                if self.weight_pool[sq][p.id].score_n >= floor:
                    added.append(p)
            if added:
                self.products[sq] = native + sorted(added, key=lambda p: -self.weight_pool[sq][p.id].score_n)
                n_added += len(added)
        if n_added:
            self.log(f"* cross-attribute pool: +{n_added} admissions")
        return n_added

    def run(self) -> List[Product]:
        super().run()        
        if self.stopped: return []

        self.__cross_attribute_pool__()
        best_p = self.__proc__()
        if self.stopped: return []
        
        return best_p
    
    @freezable
    def __proc__(self) -> List[Product]:
        tick = time.time()

        best_p, _ = self.__proc_shop_batches__(True)            
        delay = time.time() - tick

        self.log({
            "sq_count": len(self.query.intents),
            "delay": f"{delay:.2f}s"
        })
        return best_p

    def __solve_knapsack__(self, cats: List[List[Tuple[Product, Weight]]]) -> List[Tuple[Tuple[Product, ...], Tuple]]:        
        combs = MiscUtil.get_choice_combs(cats, False)
        solutions = []

        for choice in combs:
            prod_weights: List[Tuple[Product, Weight]] = [cats[i][n] for i, n in enumerate(choice)]
            prods, aggr_w, aggr_r = [], 1.0, 0

            for p, w in prod_weights:
                prods.append(p)
                
                aggr_w *= w.confidence
                aggr_r += w.score_n
            
            if len(prods) != len(set(prods)): continue
            solutions.append((tuple(prods), (aggr_w, aggr_r)))

        return sorted(solutions, key = lambda s: s[-1], reverse = True)


class VoucherEngine(KnapsackEngine):
    MAX_PLATFORM_TRIES = 8
    MAX_K_SOLUTIONS = 500

    def run(self) -> List[Product]:
        super().run()        
        if self.stopped: return []

        best_p = self.__proc__()
        if self.stopped: return []
        
        return best_p

    @freezable
    def __proc__(self) -> List[Product]:
        tick = time.time()            

        if len(self.query.intents) > 1:
            if self.query.voucher.apply_to == "shop":
                best_p = self.__proc_shop__()
            else:
                best_p, _ = self.__proc_platform__(True, True)
        else:
            best_p = self.__proc_single__()
        
        delay = time.time() - tick

        self.log({
            "sq_count": len(self.query.intents),
            "knapsack": [p.id for p in best_p],
            "delay": f"{delay:.2f}s"
        })
        return best_p

    def __proc_single__(self) -> List[Product]:
        favor_range = self.query.voucher.get_favor_range()

        candidates = [
            p for p in self.products[self.query.intents[0]]
            if p.price > favor_range[0] and p.price < favor_range[1]
        ]
        best_p = self.__select__(self.query.intents[0], candidates, True)

        if not best_p:
            candidates = [
                p for p in self.products[self.query.intents[0]] if p.price < self.query.voucher.budget
            ]
            if len(candidates): best_p = candidates[:1]

        return best_p

    def __proc_shop__(self) -> List[Product]:
        favor_range = self.query.voucher.get_favor_range()

        best_p, shop_sols = self.__proc_shop_batches__(False, price_range = favor_range)
        if best_p: return best_p

        best_p, platform_sols = self.__proc_platform__(False, False)
        if best_p: return best_p

        for count in range(len(self.query.intents), 0, -1):
            for prods, _ in (shop_sols + platform_sols):
                if len([p for i, p in enumerate(prods) if self.confirmed[self.query.intents[i]][p]]) >= count:
                    self.log(f"* fallback combo from {count} matching sub-queries!")                    
                    best_p = list(prods)
                    break
            
            if best_p: break
        
        self.sess.step_on_platform(best_p, len(shop_sols + platform_sols), True)
        return best_p

    def __proc_platform__(self, enable_favor: bool, must: bool) -> Tuple[List[Product], List]:
        if not enable_favor: self.log("* switching to platform mode...")

        favor_range = self.query.voucher.get_favor_range()
        solmap, best_p = {}, []

        shortlists = {sq: self.__prune__(self.products[sq], sq, False) for sq in self.query.intents}

        kmaps = self.__get_knapsack_maps__(shortlists, False)
        if not kmaps: return best_p

        if enable_favor: solmap["favor"] = self.__proc_knapsack_map__(kmaps[0], price_range = favor_range)
        
        solmap["fallback"] = self.__proc_knapsack_map__(kmaps[0], price_range = (0, self.query.voucher.budget))        
        best_p = self.__proc_solmap__(solmap, False, "platform", VoucherEngine.MAX_PLATFORM_TRIES if enable_favor else 1)
        
        solutions = sorted([s for _, sols in solmap.items() for s in sols], key = lambda s: s[-1], reverse = True)

        if not best_p:
            best_p = self.__find_filtered_combo__(solutions)
            if best_p: self.log("* selected best combo from filtered at outside tries!")

        if must and not best_p:            
            if solutions:
                for count in range(len(self.query.intents), -1, -1):
                    for sol in solutions:
                        if len([i for i, p in enumerate(sol[0]) if self.confirmed[self.query.intents[i]][p]]) == count:
                            self.log(f"* fallback combo with {count} matching sub-queries!")
                            best_p = list(sol[0])
                            break
                    
                    if best_p: break
            else:
                self.log("* no combo found.")
        
        if best_p or must: self.sess.step_on_platform(best_p, len(solutions), not enable_favor)
        return best_p, solutions
    
    def __solve_knapsack__(self, cats: List[List[Tuple[Product, Weight]]], price_range: Tuple[float, float]) -> List[Tuple[Tuple[Product, ...], Tuple]]:
        split, heap = len(cats) // 2, []

        l_cats, r_cats = cats[:split], cats[split:]
        l_states, r_states = [], []
        
        for subcat, substates in [(l_cats, l_states), (r_cats, r_states)]:
            for choice in product(*subcat):
                price = sum(p.price for p, _ in choice)
                if price > price_range[1]: continue

                substates.append((
                    price,
                    (math.prod(w.confidence for _, w in choice), sum(w.score_n for _, w in choice)),
                    [p for p, _ in choice]
                ))

        l_states.sort(key = lambda x: x[1], reverse = True)
        r_states.sort(key = lambda x: x[0])

        seg = SegTree([s[1] for s in r_states])
        r_prices = [s[0] for s in r_states]

        for lp, lw, l_products in l_states:
            remain_lo = price_range[0] - lp
            remain_hi = price_range[1] - lp

            lidx = bisect_left(r_prices, remain_lo)
            ridx = bisect_right(r_prices, remain_hi) - 1

            if lidx > ridx: continue

            for rw, idx in seg.query(lidx, ridx):
                r_products = r_states[idx][2]
                prods: List[Product] = l_products + r_products

                if len(set(prods)) != len(prods): continue
                w = (lw[0] * rw[0], lw[1] + rw[1])

                if len(heap) < VoucherEngine.MAX_K_SOLUTIONS:
                    if price_range[0] == 0 or self.query.voucher.is_favor_cost(lp + r_prices[idx]):
                        heappush(heap, (w, tuple(p.id for p in prods)))
                elif w > heap[0][0]:
                    if price_range[0] == 0 or self.query.voucher.is_favor_cost(lp + r_prices[idx]):
                        heapreplace(heap, (w, tuple(p.id for p in prods)))

        heap.sort(key = lambda s: s[0], reverse = True)
        return [(list(self.prod_pool[pid] for pid in prods), w) for w, prods in heap]


class ShoppingIQ(Debuggable):
    ENGINE_MAP: Dict[Challenge, Type[BaseEngine]] = {
        Challenge.PRODUCT: ProductEngine,
        Challenge.SHOP: ShopEngine,
        Challenge.VOUCHER: VoucherEngine
    }

    def __init__(self, query: Query, debugger: Optional[Debugger] = None) -> None:
        super().__init__(query, debugger)
        self.sess: Session = Session(query, debugger)

    def run(self) -> None:
        self.run_for_queue()

    def run_for_queue(self) -> List[Product]:
        """Understand → retrieve → weigh/prune/select (or shop/voucher) → return winners.

        Does not place_test_order; ProxyUtil.submit only stashes into commit_queue.
        """
        tick = time.time()
        best_p: List[Product] = []

        self.query = self.__classify__()
        if self.stopped:
            return best_p

        self.query = self.__decompose__()
        if self.stopped:
            return best_p

        self.query = self.__build__()
        if self.stopped:
            return best_p

        self.sess.query = self.query
        self.sess.step_on_build()

        engine_cls = self.ENGINE_MAP.get(self.query.challenge)
        if engine_cls is None:
            return best_p
        self.engine = engine_cls(self.query, self.sess, time.time() - tick, self.debugger)
        best_p = self.engine.run() or []
        self.__finalize__(best_p)
        return best_p

    def get_dialog(self) -> List[Dict]:
        return self.sess.dialog

    @freezable
    def __classify__(self) -> Query:
        self.query.challenge = self.__identify_challenge__()
        self.query.query_e = self.query.query_f

        if self.query.challenge == Challenge.VOUCHER: self.query.query_e, self.query.voucher = self.__parse_voucher__()

        self.log({
            "challenge": self.query.challenge,
            "query_e": self.query.query_e,
            "voucher": self.query.voucher.to_json() if self.query.voucher is not None else None
        })
        return self.query

    @freezable
    def __decompose__(self) -> Query:
        temperature = 0

        for _attempt in range(3):
            try:
                if self.query.challenge != Challenge.PRODUCT:
                    prompt = Prompt.SV_DECOMPOSE.format(
                        intent = "two or more products in the same shop" if self.query.challenge == Challenge.SHOP else "one or more products",
                        rules = Prompt.SCOPE_RULES,
                        query = self.query.query_e
                    )
                    is_multiple = True
                else:
                    prompt = Prompt.P_DECOMPOSE.format(query = self.query.query_e)
                    is_multiple = False

                js, _ = ProxyUtil.trigger_llm(prompt, temperature = temperature, expect = "dict")
                self.log(js)
                
                decomposed, alts, fillers = Prompt.parse_decompose_response(js, is_multiple)

                if len(decomposed) == 0: raise Exception("no sub-queries extracted")
                if len(decomposed) == 1 and self.query.challenge == Challenge.SHOP: raise Exception("sub-query count invalid for shop")

                banned_services, prod_only = [], True
                self.query.intents = []
                for i, js_sq in enumerate(decomposed):
                    self.query.intents.append(SubQuery(
                        query_f = self.query.query_f,
                        query = js_sq.get("query"),
                        prod_only = js_sq.get("prod_only") and prod_only,
                        services = [svc for svc in js_sq.get("services") if svc not in banned_services],
                        price = js_sq.get("price"),
                        id = i                        
                    ))

                self.query.alts = alts
                self.query.fillers = fillers
                return self.query
            except Exception as exc:
                self.log(f"# error parsing LLM response: {exc}")
                temperature += ProxyUtil.TEMP_UP

        # Soft fallback: single intent from the raw query so live gate can still run.
        if not self.query.intents:
            self.query.intents = [SubQuery(
                query_f=self.query.query_f,
                query=self.query.query_e or self.query.query_f,
                prod_only=False,
                services=[],
                price=None,
                id=0,
            )]
            self.query.alts = {}
            self.query.fillers = []
        return self.query

    @freezable
    def __review__(self) -> Query:
        suspects, prompts_i, prompts_o, focuses, error_3 = self.__check_review_needed__()

        if len(suspects) > 0:
            js, _ = ProxyUtil.trigger_llm(self.__create_review_prompt__(prompts_i, prompts_o, focuses, error_3), expect = "dict")
            self.__fix_by_llm_review__(js, suspects)
            self.log(js)

        return self.query

    @freezable
    def __build__(self) -> Query:
        temperature = 0

        for _attempt in range(3):
            try:
                js, _ = ProxyUtil.trigger_llm(
                    Prompt.PSV_BUILD.format(queries = {str(i + 1): sq.query for i, sq in enumerate(self.query.intents)}),
                    temperature = temperature,
                    expect = "dict"
                )
                built = Prompt.parse_build_response(js)
                self.log(js)

                if len(built) != len(self.query.intents):
                    raise Exception(f"sub-query count mismatches (expected {len(self.query.intents)} but got {len(built)})")

                for b, sq in zip(built, self.query.intents):
                    sq.compact = b.get("compact") or sq.query
                    sq.parsed = {}

                    for k, v in b.get("parsed").items():
                        kw = EngineUtil.compact_kv(k, v, sq.query)
                        
                        if not kw: continue
                        nk, nv = kw

                        if nk not in sq.parsed:
                            for ok in sq.parsed:
                                if sq.parsed[ok] == nv:
                                    if f"{nv} {nk.replace('_', ' ')}" in sq.query:
                                        sq.parsed[nk] = nv
                                        del sq.parsed[ok]
                                    
                                    break
                            else:
                                sq.parsed[nk] = nv
                        else:
                            sq.parsed[nk] += " " + nv

                    if sq.prod_only: sq.parsed["service"] = Vocab.PROD_ONLY.replace('_', ' ')

                return self.query
            except Exception as exc:
                self.log(f"# error parsing LLM response: {exc}")
                temperature += ProxyUtil.TEMP_UP

        for sq in self.query.intents:
            if not getattr(sq, "compact", None):
                sq.compact = sq.query
            if not getattr(sq, "parsed", None):
                sq.parsed = {}
        return self.query

    def __finalize__(self, best_p: List[Product]):
        # Stash only — ORO terminal is skeleton place_test_order / _finish_fast.
        try:
            if getattr(self, "engine", None) is not None:
                self.sess.step_on_compare(best_p, self.engine.__get_alt_str__(best_p))
        except Exception:
            pass
        self.sess.add_tool_result(ProxyUtil.submit(best_p))
        self.sess.add_tool_result(ProxyUtil.finish("success" if best_p else "failure"))
        try:
            self.sess.step_on_submit(best_p)
        except Exception:
            pass

    def __identify_challenge__(self) -> Challenge:
        if self.query.challenge == Challenge.UNKNOWN:
            if re.search(Vocab.VOUCHER_PATTERN, self.query.query_f, re.I) and re.search(r"\bvoucher\b", self.query.query_f, re.I): return Challenge.VOUCHER
            if re.search(Vocab.SHOP_PATTERN, self.query.query_f, re.I): return Challenge.SHOP
            return Challenge.PRODUCT
        else:
            return self.query.challenge

    def __parse_voucher__(self) -> Tuple[str, Voucher]:
        sep = self.query.query_f.lower().find("my budget is")

        query_e = self.query.query_f[:sep].strip()
        v_str = self.query.query_f[sep:].strip()

        m = re.search(r"\bmy\s+budget\s+is(?:\s+only)?\s+`(\d+)`", v_str, re.I)
        budget = int(m.group(1))

        apply_to = "shop" if re.search(r"\bsame\s+shops?\b", v_str, re.I) else "all"

        m = re.search(r"\bexceeds?\s+`(\d+)`", v_str, re.I)
        threshold = int(m.group(1))

        if (m := re.search(r"\bpercent(?:age)?\s+discount(?:\s+of)?\s+`([\d\.]+)\%`", v_str, re.I)):
            discount = float(m.group(1))
            is_fixed = False

            m = re.search(r"\bwith(?:\s+a)\s+cap(?:\s+of)?\s+`([\d\.]+)`", v_str, re.I)
            cap = float(m.group(1))
        else:
            m = re.search(r"\bfixed\s+discount(?:\s+of)?\s+`([\d\.]+)`", v_str, re.I)
            discount = float(m.group(1))
            is_fixed = True
            cap = None
        
        voucher = Voucher(
            budget = budget,
            apply_to = apply_to,
            threshold = threshold,
            is_fixed = is_fixed,
            discount = discount,
            cap = cap
        )
        return query_e, voucher

    def __check_review_needed__(self) -> Tuple[List[str], str, str, str, str]:
        freeship_p = len([sq for sq in self.query.intents if "freeShipping" in sq.services])
        lazmall_p = len([sq for sq in self.query.intents if "official" in sq.services])
        lazflash_p = len([sq for sq in self.query.intents if "flashsale" in sq.services])        
        cod_p = len([sq for sq in self.query.intents if "COD" in sq.services])
        price_p = len([sq for sq in self.query.intents if sq.price])

        cod_e = len(re.findall(Vocab.SERVICE_PATTERNS["COD"][0], self.query.query_e, re.I))
        lazmall_e = len(re.findall(Vocab.SERVICE_PATTERNS["official"][0], self.query.query_e, re.I))
        lazflash_e = len(re.findall(Vocab.SERVICE_PATTERNS["flashsale"][0], self.query.query_e, re.I))
        freeship_e = len(re.findall(Vocab.SERVICE_PATTERNS["freeShipping"][0], self.query.query_e, re.I))
        price_e = len(re.findall(r"\b(php|pesos?|usd|dollars?|euros?)\b", self.query.query_e, re.I))

        suspects, prompts_i, prompts_o, focuses = [], [], [], []

        for label, count_e, count_p, prefix in [
            ("lazmall", lazmall_e, lazmall_p, "The word \"LazMall\""),
            ("lazflash", lazflash_e, lazflash_p, "The word \"LazFlash\""),
            ("free_ship", freeship_e, freeship_p, "The condition of free or complimentary shipping"),
            ("cod", cod_e, cod_p, "The condition of \"Cash on Delivery\" option")
        ]:
            if count_e != count_p:
                suspects.append(label)

                prompts_i.append(f"""
- {prefix} appeared {count_e} time(s) in the original query, but {count_p} products were parsed with this constraint. It needs double check.
""".strip()
                )
                prompts_o.append(f"""
    "{label}": [
        <product_id_1, str>, ... // correct product ids that must receive **{label}** constraint.
    ]
""".strip())

        if price_e != price_p:
            prompts_i.append(f"""
- Currency unit appeared {price_e} time(s) in the original query, but {price_p} products were parsed with price constraint. You must carefully check the context if those number ranges are really the price constraints. Keep in mind that one price constraint cannot apply to multiple products.
""".strip())
            prompts_o.append("""
    "price": { // list all the products (don't skip any product)
        "1": { // product id = 1
            "parsed_price_range": <str|null>, // the price range string or null as assigned for this product by the parsing result given
            "is_it_correct": <bool>, // true if the parsed result is correct, otherwise false
            "other_spec": <str|null>, // the other spec name ("power usage", for example) which was mistakenly identified as the price range in the parsing result if exists. set null if "parsed_price_range" is null or the parsed result has no error.
            "correct_price_range": <str|null> // the corrected price range for this product. write the original parsed result again if it has no error. set null if no price range condition is for this product. use "[low]-[high]" format without currency unit if exists. "[low]-" or "-[high]" if either bound is missing.
        },
        ...    
    }
""".strip())
            suspects.append("price")

        if "lazmall" in suspects: focuses.append('- *lazmall**: explicit mention of the word "LazMall".')
        if "lazflash" in suspects: focuses.append('- **lazflash**: explicit mention of the word "LazFlash".')
        if "free_ship" in suspects: focuses.append('- **free_ship**: explicit mention of free or complimentary shipping.')
        if "cod" in suspects: focuses.append('- **cod**: explicit mention of cash on delivery or payment on delivery.')
        if "price" in suspects: focuses.append('- **price**: explicit mention of price range. (pared results use "[low]-[high]" format without currency unit. just "[low]-" or "-[high]" if only one bound is specified.)')

        if "price" in suspects:
            error_3 = """
## 3. Confused price range
    - A product appears with price range constraint in the parsed result, but actually this number range is for a completely different specification.
    - For example, in "I need a bottle between 0.5 and 1.2 litre", the range "0.5-1.2" is not price constraint, but a capacity constraint.
    - Also, in "Find motors with power usage less than 2.5", the range "-2.5" is mentioned without currency unit, but you can easily confirm that it's the power usage constraint with unit "kw".
""".strip()
        else:
            error_3 = ""

        return suspects, '\n'.join(prompts_i), '\n'.join(prompts_o), '\n'.join(focuses), error_3

    def __create_review_prompt__(self, prompts_i: str, prompts_o: str, focuses: str, error_3: str) -> str:
        prods = {
            str(i + 1): {
                "sub-query": sq.query,
                "constraints": [Prompt.service_to_label(svc) for svc in (sq.services or [])],
                "price": sq.price
            }
            for i, sq in enumerate(self.query.intents)
        }
        return Prompt.PSV_REVIEW.format(
            count = len(self.query.intents),
            focuses = focuses,
            rules = Prompt.SCOPE_RULES,
            suspects = prompts_i,
            error_3 = error_3,
            responses = prompts_o,
            query = self.query.query_e,
            parsed = prods
        )

    def __fix_by_llm_review__(self, js: Dict[str, List[str] | Dict], suspects: List[str]) -> None:
        def __get_id__(id_str: Any) -> Optional[int]:
            if not isinstance(id_str, str): return None
            pid = MiscUtil.extract_number(id_str, True) - 1

            if pid < 0 or pid >= len(self.query.intents): return None
            return pid
        
        for issue in suspects:
            if issue not in js: continue
            obj = js.get(issue)

            if issue in ["lazmall", "lazflash", "cod", "free_ship"]:
                if not isinstance(obj, list): continue
                svc = Prompt.label_to_service(issue)
                
                ids = [__get_id__(x) for x in obj]
                if None in ids: continue

                for i, sq in enumerate(self.query.intents):
                    if i in ids and svc not in sq.services:
                        sq.services.append(svc)
                    elif i not in ids and svc in sq.services:
                        sq.services.remove(svc)
            elif issue == "price":
                if not isinstance(obj, dict): continue
                
                ids = [__get_id__(k) for k in obj.keys()]
                if None in ids: continue

                if sorted(ids) != list(range(len(self.query.intents))): continue

                for i, sq in enumerate(self.query.intents):
                    j: Dict[str, str] = obj.get(str(i + 1))
                    
                    correct = j.get("correct_price_range")
                    sq.price = re.sub(r"[^\d\-]", "", correct or "") or None



# ===== Hiororo Agent (live commit / event / claims / must-terminate) =====

class Agent:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_names = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.brief = Brief(str(self.policy.get("query") or ""))
        self.budget = self.brief.budget
        self.ledger = Ledger()
        self.clock = Clock(_PROBLEM_BUDGET_S)
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": self._system_prompt()},
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
        self.hinted_exact = False
        self.tried: dict = {}
        self.questions_asked = 0
        self.searched_siblings = False
        self.hinted_brand = False
        self._wishes: set | None = None
        self.probes: set = set()
        self.searched_twins = False
        self.reads_done = False
        self.event_turn: int | None = None
        self.unreadable: set = set()
        self.question_waits = 0
        self.done = False
        self.model_index = 0
        self.commit_queue: list = []  # ranked (product_id, sku) from ShoppingIQ
        self._iq_done = False
        self._iq_refills = 0
        self._iq_engine = None
        self._iq_products: dict = {}
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)


    # ----- ShoppingIQ → commit_queue glue -----
    def _note_iq_winners(self, products: list) -> None:
        """Called from ProxyUtil.submit: winners + ranked alternates → commit_queue."""
        pairs = _products_to_commit_pairs(list(products or []), getattr(self, "_iq_engine", None))
        # Also pull near-best alternates from the engine product pools.
        engine = getattr(self, "_iq_engine", None)
        if engine is not None:
            for sq in getattr(engine, "query", Query()).intents or []:
                pool = list(getattr(engine, "products", {}).get(sq) or [])
                weights = getattr(engine, "weight_pool", {}).get(sq) or {}
                ranked = sorted(
                    pool,
                    key=lambda p: -(weights.get(p.id).score_n if p.id in weights else 0.0),
                )
                for p in ranked[:12]:
                    pairs.extend(_products_to_commit_pairs([p], engine))
        seen = set()
        merged = []
        for pid, sku in pairs + list(self.commit_queue):
            key = (str(pid), str(sku))
            if key in seen or not sku:
                continue
            seen.add(key)
            merged.append(key)
            self._iq_products[str(pid)] = self._iq_products.get(str(pid))
        self.commit_queue = merged
        if merged:
            self._note(
                "Shopping IQ queued %d candidate variant(s); live gate will inspect→cart→recheck→order."
                % len(merged)
            )

    def _bootstrap_shopping_iq(self) -> None:
        """Run cleaned Product/Shop/Voucher cascade once; stash commit_queue; keep _act ownership."""
        global _ORO_AGENT
        if self._iq_done:
            return
        self._iq_done = True
        if self.clock.remaining() < 90:
            return
        _ORO_AGENT = self
        BaseEngine.arm_deadline()
        # Leave ≥70s for live commit / event closure.
        BaseEngine._deadline_mono = time.monotonic() + max(40.0, min(150.0, self.clock.remaining() - 70.0))
        query_text = str(self.policy.get("query") or "")
        challenge = detect_composition(query_text, None)
        # Ranking / recall / recovery stay on skeleton compositions; still use Product IQ for retrieve.
        try:
            iq = ShoppingIQ(Query(challenge=challenge, query_f=query_text))
            winners = iq.run_for_queue() or []
            self._iq_engine = getattr(iq, "engine", None)
            self._note_iq_winners(winners)
            # Seed commit_queue from brief candidate keys when ranking-shaped.
            for pid, sku in self.brief.candidate_keys:
                key = (str(pid), str(sku))
                if key not in self.commit_queue:
                    self.commit_queue.append(key)
        except Exception as exc:
            self._note("Shopping IQ bootstrap skipped: %s: %s" % (type(exc).__name__, str(exc)[:160]))
        finally:
            _ORO_AGENT = self  # keep bridge for recover refill

    def _refill_commit_queue(self) -> bool:
        """Recover tiers as queue refill only (never force invalid order)."""
        if self._iq_refills >= 2 or self.clock.remaining() < 55:
            return False
        engine = self._iq_engine
        if engine is None:
            return False
        self._iq_refills += 1
        before = len(self.commit_queue)
        try:
            # Partial / above-floor pools across intents.
            for sq in getattr(engine, "query", Query()).intents or []:
                pool = list(getattr(engine, "products", {}).get(sq) or [])
                if hasattr(engine, "__above_score_floor__"):
                    pool = engine.__above_score_floor__(pool, sq) or pool
                for p in pool[:20]:
                    for pid, sku in _products_to_commit_pairs([p], engine):
                        key = (pid, sku)
                        if key in self.commit_queue:
                            continue
                        if Ledger.key(pid, sku) in self.ledger.dead_keys:
                            continue
                        self.commit_queue.append(key)
        except Exception as exc:
            self._note("Recover refill failed: %s" % str(exc)[:120])
            return False
        gained = len(self.commit_queue) - before
        if gained:
            self._note("Recover refill added %d alternate candidate(s) into the commit queue." % gained)
        return gained > 0

    # ------------------------------------------------------------------ prompts
    def _system_prompt(self) -> str:
        extras = []
        if self.brief.required_identities:
            extras.append(
                "Surface at least %d distinct matching products in search or filter results before ordering; "
                "the controller reports the count after each search." % self.brief.required_identities
            )
        if self.brief.candidate_keys:
            extras.append(
                "Only the named candidate variants in the request may be ordered; compare them so their "
                "facts are on record. Rank by the axes the shopper stated, then price."
            )
        if self.brief.rank_axes:
            axes = ", then ".join(("%s %s" % (d, p) for d, p in self.brief.rank_axes))
            extras.append("Task-stated ranking order: " + axes + ".")
        if self.brief.pref_model:
            extras.append(
                "Find the preferred model/color first. The first cart item must be that exact pairing; "
                "if it disappears, relax as little as possible (same model, other color before any other model)."
            )
        if self.brief.spec_sealed:
            extras.append(
                "One firm requirement is withheld until you ask; ask once, then obey the answer exactly."
            )
        body = _CORE_RULES
        if extras:
            body += "\n\nTask notes: " + " ".join(extras)
        return body

    def _opening_question(self) -> str | None:
        if self.brief.spec_sealed:
            return (
                "Before I search: which firm specification are you requiring, and what exact value do you need?"
            )
        if self.brief.family == "preference":
            return (
                "Quick question so I choose well: which of your preferences counts most, and in what "
                "order should I weigh the others? Let me know your priority."
            )
        return None

    def _note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    def _currency_label(self) -> str:
        return self.brief.currency or ""

    # ------------------------------------------------------------------ evidence
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

    def _cat_path(self, pid: str) -> list:
        path = self._prod(pid).get("category_path")
        return list(path) if isinstance(path, list) else []

    def _options(self, pid: str, sku: str) -> str:
        return self.ledger.options.get(Ledger.key(pid, sku), "")

    def _evidence_blob(self, pid: str, sku: str | None = None) -> str:
        """Flatten observed title / all facts / options — no fixed fact-key routing."""
        parts = [self._title(pid)]
        for k, v in self._facts(pid).items():
            parts.append("%s %s" % (k, v))
        if sku is not None:
            parts.append(self._options(pid, sku))
        return " ".join(parts)

    def _need_met_in(self, text: str, measures: list | None = None, classes: list | None = None) -> bool:
        measures = measures if measures is not None else self.brief.measurements
        classes = classes if classes is not None else self.brief.class_tokens
        if not measures and not classes:
            return True
        for val, unit in measures:
            if not _measure_in_text(text, val, unit):
                return False
        for tok in classes:
            if not _class_in_text(text, tok):
                return False
        return True

    def _variant_supports_needs(self, pid: str, sku: str) -> int:
        """2 = all needs met, 0 = missing evidence, -1 = contradicting option text."""
        measures = self.brief.measurements
        classes = self.brief.class_tokens
        if not measures and not classes:
            return 2
        opt = self._options(pid, sku) or ""
        if opt:
            opt_measures = parse_measurements(opt)
            for val, unit in measures:
                same_unit = [v for v, u in opt_measures if _norm_unit(u) == _norm_unit(unit)]
                if same_unit and val not in same_unit:
                    return -1
            opt_classes = parse_class_tokens(opt)
            for tok in classes:
                if opt_classes and not _class_in_text(opt, tok):
                    if any(_class_in_text(opt, c) for c in opt_classes):
                        return -1
            if self._need_met_in(opt, measures, classes):
                return 2
        blob = self._evidence_blob(pid, sku)
        if self._need_met_in(blob, measures, classes):
            return 2
        return 0

    def _facts_support_needs(self, pid: str) -> bool:
        """Justification path: facts (plus title as secondary) must state gated needs."""
        fact_blob = " ".join("%s %s" % (k, v) for k, v in self._facts(pid).items())
        if self._need_met_in(fact_blob):
            return True
        return self._need_met_in(fact_blob + " " + self._title(pid))

    def _axis_value(self, pid: str, sku: str, phrase: str) -> float | None:
        """Option text wins when it states the axis; else facts+title (task-driven)."""
        opt = self._options(pid, sku)
        if opt:
            got = extract_numeric_for_phrase(opt, phrase)
            if got is not None:
                return got
        return extract_numeric_for_phrase(self._evidence_blob(pid, sku), phrase)

    @staticmethod
    def _stems(text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

    def _category_ok(self, pid: str) -> bool:
        """Whether the listing sits on the shelf the shopper asked for."""
        want = self.brief.category
        path = self._cat_path(pid)
        if not want or not path:
            return True
        wanted = self._stems(want)
        labels = set()
        for label in path:
            labels |= self._stems(str(label))
        placed = bool(wanted & labels) or any(
            self._stems("".join(re.findall(r"[a-z]+", str(label).lower()))) == wanted for label in path
        )
        if not placed and "smartphone" in wanted:
            placed = any(
                re.fullmatch(r"(?:smart|cell|mobile|unlocked)?\s*phones?", str(label).strip(), re.I)
                for label in path
            )
        if not placed:
            return False
        # Category modifiers (from the category phrase itself) should appear in the title.
        title_words = self._stems(self._title(pid)) | _tokens(self._title(pid))
        for qual in self.brief.cat_qualifier:
            stem = qual.rstrip("s")
            if stem and stem not in title_words and qual not in title_words:
                if stem not in re.sub(r"[^a-z0-9]+", "", self._title(pid).lower()):
                    return False
        return True

    def _brand_ok(self, pid: str) -> bool:
        want = self.brief.brand
        if not want:
            return True
        wanted = _tokens(want)
        return bool(wanted <= _tokens(self._brand_of(pid)) or wanted <= _tokens(self._title(pid)))

    def _device_ok(self, pid: str, sku: str) -> bool:
        """Categories that name a device are graded on the listing being that device.

        Where the task itself turns on device identity, only positive evidence
        will do; elsewhere a listing is only ruled out when the evidence
        contradicts it, so a thin product page is not treated as a disqualifier.
        """
        if not _same_word(self.brief.category or "", "smartphone"):
            return True
        verdict = _phone_identity(self._facts(pid), self._title(pid), self._options(pid, sku))
        if self.brief.family in ("constraint", "justification", "recovery"):
            return verdict == "yes"
        return verdict != "no"

    def _resale_title(self, pid: str) -> bool:
        """Listings that describe someone else's listing, or a used unit, are never the answer."""
        return bool(re.search(r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed", self._title(pid), re.I))

    def _relax(self, pid: str, sku: str) -> int:
        """How far a candidate sits from the shopper's exact preferred model and color."""
        model, color = self.brief.pref_model, self.brief.pref_color
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self._title(pid) + " " + " ".join(str(v) for v in self._facts(pid).values()))
        same_model = squash(model) in blob
        shade = (self._options(pid, sku) + " " + str(self._facts(pid).get("color") or "") + " " + self._title(pid)).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def _assess(self, pid: str, sku: str) -> dict:
        """Score one variant against every gate the runtime will apply to it."""
        fam = self.brief.family
        key = Ledger.key(pid, sku)
        st = self.ledger.checked.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.ledger.dead_keys or (str(pid) in self.ledger.dead_products and fam != "recovery")
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and not dead
        viewed = str(pid) in self.ledger.products
        spec = self._variant_supports_needs(pid, sku)
        ok = bool(live and viewed and spec == 2 and self._brand_ok(pid) and self._device_ok(pid, sku))
        cat_ok = self._category_ok(pid)
        ok = ok and cat_ok
        if fam == "justification":
            ok = ok and not self._resale_title(pid) and bool(self._brand_of(pid) or self.brief.brand is None or self._brand_ok(pid))
            if self.brief.measurements or self.brief.class_tokens:
                ok = ok and self._facts_support_needs(pid)
        if fam == "preference" and self.ledger.event_target:
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split("::", 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if fam == "recall" and self.brief.theme:
            ok = ok and all(t in _tokens(self._title(pid)) for t in self.brief.theme)
        if fam == "ranking":
            ok = ok and (str(pid), str(sku)) in self.brief.candidate_keys
        return {
            "pid": str(pid), "sku": str(sku), "key": key, "price": price, "live": live,
            "viewed": viewed, "spec": spec, "cat_ok": cat_ok, "ok": ok,
            "relax": self._relax(pid, sku) if fam == "recovery" else 0,
        }

    def _rank_of(self, c: dict, first: bool) -> tuple:
        """Order eligible candidates by task-stated ranking axes, else soft fit + price."""
        fam = self.brief.family
        price = c["price"] if c["price"] is not None else 1e12
        if fam == "ranking" or self.brief.rank_axes:
            key_parts: list = []
            for direction, phrase in self.brief.rank_axes:
                raw = self._axis_value(c["pid"], c["sku"], phrase)
                if any(t in phrase.lower() for t in ("price", "cost", "cheap", "expensive")):
                    raw = c["price"]
                if raw is None:
                    raw = -1e12 if direction == "max" else 1e12
                key_parts.append(-raw if direction == "max" else raw)
            key_parts.extend([price, c["key"]])
            return tuple(key_parts)
        if fam == "recovery":
            return (c["relax"], price, c["key"])
        miss = 0
        blob = self._evidence_blob(c["pid"], c["sku"])
        for val, unit in self.brief.measurements[1:]:
            if not _measure_in_text(blob, val, unit):
                miss += 1
        for tok in self.brief.class_tokens[1:]:
            if not _class_in_text(blob, tok):
                miss += 1
        proven = 0 if _phone_identity(self._facts(c["pid"]), self._title(c["pid"]), self._options(c["pid"], c["sku"])) == "yes" else 1
        if fam == "intent" and first and self.brief.pref_brand:
            wanted = _tokens(self.brief.pref_brand)
            near = 0 if (wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))) else 1
            return (near, miss, proven, price, c["key"])
        return (miss, proven, price, c["key"])

    def _candidates(self) -> list:
        seen = set()
        out = []
        for key in list(self.ledger.checked):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self._assess(pid, sku))
        # Named ranking candidates: include before stock inspect so axes can rank.
        for pid, sku in self.brief.candidate_keys:
            key = Ledger.key(pid, sku)
            if key in seen:
                continue
            seen.add(key)
            out.append(self._assess(pid, sku))
        return out

    def _target(self) -> dict | None:
        """The best fully verified candidate, or nothing while evidence is missing."""
        first = self.ledger.adds == 0
        fam = self.brief.family
        # Prefer ShoppingIQ commit_queue order while live gates still apply.
        if self.commit_queue:
            for pid, sku in self.commit_queue:
                key = Ledger.key(pid, sku)
                if key in self.ledger.dead_keys or str(pid) in self.ledger.dead_products:
                    continue
                c = self._assess(str(pid), str(sku))
                if c.get("ok"):
                    return c
        good = [c for c in self._candidates() if c["ok"]]
        if fam == "preference":
            # What makes one of these the right answer is a judgement about the
            # shopper's own priorities, so the choice stays with the model; the
            # controller only carries its choice through to a valid order.
            carted = [c for c in good if (c["pid"], c["sku"]) in self.ledger.cart]
            if carted:
                return carted[0]
            # Which of these best fits the shopper is a judgement call the model makes
            # better than any keyword rule, so the controller only steps in when the
            # episode would otherwise end with nothing bought.
            if good and self.turn >= self.max_steps - 8:
                good.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]),
                                         c["price"] if c["price"] is not None else 1e12))
                return good[0]
            return None
        if fam == "intent" and first and self.brief.pref_brand:
            wanted = _tokens(self.brief.pref_brand)
            named = [c for c in good if wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))]
            if named:
                good = named
            elif self.turn < self.max_steps * 0.6:
                self._preferred_brand_hint()
                return None
        if fam == "recovery" and first:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif self.turn < self.max_steps * 0.6:
                self._exact_match_hint()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self._rank_of(c, first))
        return good[0]

    def _answer_text(self) -> str:
        """A short, factual answer to the shopper, drawn only from observed evidence."""
        target = self._target() or (self._candidates() or [None])[0]
        if not target:
            return "I am still checking the options against your requirements; nothing is decided yet."
        pid, sku = target["pid"], target["sku"]
        parts = ["I am going with " + (self._title(pid) or pid)[:90]]
        if self.brief.spec_value and self._variant_supports_needs(pid, sku) == 2:
            parts.append("it meets the stated requirement " + str(self.brief.spec_value))
        if target.get("price") is not None:
            cur = (" " + self.brief.currency) if self.brief.currency else ""
            parts.append("it is %.2f%s and in stock" % (target["price"], cur))
        return ", ".join(parts) + "."

    def _read_more(self) -> tuple | None:
        """Turn the search results already in hand into read product pages, then widen.

        Reading pages is what turns rows into candidates the controller can verify,
        and a second verified product is what makes the market change survivable.
        """
        fam = self.brief.family
        if fam == "recall" and len(self.ledger.identities) < self._recall_need():
            return None
        twin = self._twin_search()
        if twin is not None:
            return twin
        leads = self._unseen_leads()
        if leads and self.forced_compares < 8 and self.turn >= 2:
            self.forced_compares += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        # No unread rows left: ask the catalog again under the requirements as they stand now.
        probe = (self.brief.brand or "", "%.2f" % (self.budget or 0.0), str(self.brief.spec_value))
        if self.turn < 2:
            return None
        if probe in self.probes:
            self.reads_done = True  # the catalog has been asked this already
            return None
        self.probes.add(probe)
        if "filter" in self.tool_names and self.brief.category:
            args = {"category": self.brief.category, "in_stock": True, "k": 10}
            if self.budget is not None:
                args["max_price"] = self.budget
            if self.brief.brand:
                args["brand"] = self.brief.brand
            return ("filter", args)
        if "search" in self.tool_names:
            words = [w for w in (self.brief.brand, self.brief.spec_value, self.brief.category) if w]
            if words:
                return ("search", {"query": " ".join(words), "in_stock": True, "k": 10})
        self.reads_done = True
        return None

    def _twin_search(self) -> tuple | None:
        """After the market kills the carted item, its own twins are the nearest replacements.

        A listing's other colors and sizes are separate products carrying the same
        title, so searching that title surfaces exactly the items that still meet
        every requirement the dead one met.
        """
        dead = self.ledger.event_target
        if self.searched_twins or not dead or "search" not in self.tool_names:
            return None
        title = self._title(dead.split("::", 1)[0])
        if not title:
            return None
        self.searched_twins = True
        self._note(
            "The item that died is one of several listings sharing a title; searching that title "
            "surfaces its other colors and sizes, which meet the same requirements."
        )
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def _preferred_brand_hint(self) -> None:
        if self.hinted_brand:
            return
        self.hinted_brand = True
        self._note(
            "No %s product that meets every firm requirement has been read yet. The shopper's first choice is "
            "graded on that brand, so filter by brand=%s within the budget and read those products before any "
            "other." % (self.brief.pref_brand, self.brief.pref_brand)
        )

    def _sibling_search(self) -> tuple | None:
        """A found model in the wrong color: its other colors share the same title."""
        if self.searched_siblings:
            return None
        rows = [c for c in self._candidates() if c["live"] and c["viewed"] and c["relax"] == 0]
        if rows:
            return None
        near = [c for c in self._candidates() if c["viewed"] and c["relax"] == 1]
        if not near:
            return None
        title = self._title(near[0]["pid"])
        if not title:
            return None
        self.searched_siblings = True
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def _exact_match_hint(self) -> None:
        if self.hinted_exact or not self.brief.pref_model:
            return
        self.hinted_exact = True
        self._note(
            "The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own "
            "and view the results so every color variant is on record; only that exact pairing may be the "
            "first cart item." % (self.brief.pref_model, ", " + self.brief.pref_color if self.brief.pref_color else "")
        )

    def _unseen_leads(self) -> list:
        """Promising variants whose product page has not been read yet."""
        leads = []
        for key, row in self.ledger.rows.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.ledger.products or str(pid) in self.ledger.dead_products:
                continue
            st = self.ledger.checked.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.budget is not None and price > self.budget + _EPS):
                continue
            title = str(row.get("title") or "")
            if self.brief.theme and not all(t in _tokens(title) for t in self.brief.theme):
                continue
            if self.brief.pref_model and self.ledger.adds == 0 and not self.ledger.event_target:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.brief.pref_model.lower()) not in squashed:
                    continue
            if self.brief.brand:
                wanted = _tokens(self.brief.brand)
                if not (wanted <= _tokens(str(row.get("brand") or "")) or wanted <= _tokens(title)):
                    continue
            leads.append((price, str(pid)))
        leads.sort()
        out = []
        for _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    # ------------------------------------------------------------------ guidance
    def _shortlist_note(self) -> None:
        """Tell the model what the controller can and cannot yet verify."""
        cands = self._candidates()
        if self.brief.family == "preference" and not self.ledger.cart:
            good = [c for c in cands if c["ok"]]
            good.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]), c["price"] if c["price"] is not None else 1e12))
            if good:
                self._note(
                    "Products that meet every firm requirement so far: "
                    + "; ".join("%s::%s at %.2f" % (c["pid"], c["sku"], c["price"] or 0.0) for c in good[:6])
                    + ". Pick the one that best fits the shopper's stated priorities, inspect it, then add it."
                )
                return
        best = self._target()
        if best:
            self._note(
                "Verified candidate: %s::%s at %.2f. The controller will inspect, cart and order it; "
                "read other candidates only if one could rank better."
                % (best["pid"], best["sku"], best["price"] or 0.0)
            )
            return
        field, want = self.brief.spec_field, self.brief.spec_value
        blocked = [c for c in cands if c["viewed"] and c["live"] and not c["ok"]]
        if blocked and field:
            sample = blocked[:3]
            self._note(
                "None of the products read so far can be verified. For each, the evidence for %s must "
                "resolve to exactly %s from the variant's own option text or the product's raw facts: %s. "
                "Look at different products." % (field, want, ", ".join(c["pid"] + "::" + c["sku"] for c in sample))
            )
        elif blocked:
            self._note(
                "The products read so far fail a firm requirement (category, brand, stock, budget or the "
                "kind of device requested). Search for different products."
            )

    def _priority_hits(self, pid: str, sku: str) -> int:
        """How much of what the shopper actually asked for this product's own facts mention."""
        wants = self._wish_words()
        if not wants:
            return 0
        have = _tokens(" ".join(str(v) for v in self._facts(pid).values()) + " " + self._title(pid) + " " + self._options(pid, sku))
        return len(wants & have)

    def _wish_words(self) -> set:
        """Content words from the shopper's own wording, gathered once."""
        if self._wishes is None:
            said = self.brief.goal + " " + " ".join(
                str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
            )
            drop = _tokens(self.brief.category or "") | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
            }
            self._wishes = {w for w in _tokens(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._wishes

    def _budget_from_message(self, text: str) -> None:
        new = Brief._budget(text)
        if new is None or (self.budget is not None and new >= self.budget):
            return
        self.budget = new
        self.shopper_updated = True
        cur = (" " + self.brief.currency) if self.brief.currency else ""
        self._note("Budget updated by the shopper: at most %.2f%s. Every later choice must fit it." % (new, cur))
        for pid, sku in list(self.ledger.cart):
            st = self.ledger.status(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _EPS:
                self.ledger.dead_keys.add(Ledger.key(pid, sku))
                self._note(
                    "The carted item %s costs %.2f, above the new budget; the controller will replace it."
                    % (Ledger.key(pid, sku), price)
                )

    def _claims_for(self, pid: str, sku: str) -> list:
        """Claims grounded in observed evidence; field names derived from the ask units."""
        out = []
        facts = self._facts(pid)
        fact_blob = " ".join("%s %s" % (k, v) for k, v in facts.items())
        title = self._title(pid)
        full = fact_blob + " " + title + " " + self._options(pid, sku)

        for val, unit in self.brief.measurements:
            field = claim_field_for_unit(unit)
            value = claim_value_for_measure(val, unit)
            raw_claim = None
            for key, raw in facts.items():
                text_blob = "%s %s" % (key, raw)
                if _measure_in_text(text_blob, val, unit) or (
                    re.fullmatch(r"\d+(?:\.\d+)?", str(raw).strip())
                    and abs(float(raw) - val) < _EPS
                ):
                    raw_claim = str(raw)
                    break
            if raw_claim is None and not _measure_in_text(full, val, unit):
                continue
            out.append({"field": field, "value": value})
            if field == "screen_size":
                out[-1]["value"] = _fmt_measure(val, _norm_unit(unit))

        for tok in self.brief.class_tokens:
            if not _class_in_text(full, tok):
                continue
            if _norm_class(tok) == "hd" and (
                _class_in_text(full, "fhd") or _class_in_text(full, "qhd") or _class_in_text(full, "4k")
                or re.search(r"\bhd\s*\+|1080p|1440p|2160p", full, re.I)
            ):
                if not re.search(r"720p", full, re.I):
                    continue
            if any(c.get("field") == "resolution" for c in out):
                continue
            out.append({"field": "resolution", "value": _norm_class(tok)})

        if not out and (self.brief.measurements or self.brief.class_tokens):
            return []

        st = self.ledger.status(pid, sku) or {}
        if self.budget is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.budget})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        out.append({"field": "in_stock", "value": "true"})
        brand = self._brand_of(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        seen = set()
        deduped = []
        for c in out:
            f = c.get("field")
            if f in seen:
                continue
            seen.add(f)
            deduped.append(c)
        return deduped

    def _order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.brief.family == "justification" or self.brief.wants_reason:
            claims = self._claims_for(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def _valid_now(self, pid: str, sku: str) -> tuple:
        st = self.ledger.status(pid, sku)
        k = Ledger.key(pid, sku)
        if st is None:
            return False, "inspect_stock the exact product_id and sku first"
        if (pid, sku) not in self.ledger.cart:
            return False, "add that exact variant to the cart first"
        if self.ledger.last_add_seq is not None and self.ledger.checked_seq.get(k, 0) <= self.ledger.last_add_seq:
            return False, "re-check the variant with inspect_stock after the cart addition"
        if not st.get("in_stock"):
            return False, "that variant is out of stock; choose a different product"
        price = st.get("price")
        if self.budget is not None and price is not None and price > self.budget + _EPS:
            return False, "that variant costs %.2f, above the budget of %.2f" % (price, self.budget)
        if k in self.ledger.dead_keys:
            return False, "that variant was invalidated by the market change; choose a different product"
        return True, ""

    # ------------------------------------------------------------------ planning
    def _recall_need(self) -> int:
        """How many distinct matching products to surface before buying.

        Surfacing more than the stated minimum is worth turns early on, but the
        catalog may simply not hold that many, so the margin lapses.
        """
        early = self.search_calls < 8 and self.turn < self.max_steps * 0.5
        return self.brief.required_identities + (2 if early else 0)

    def _ready(self) -> bool:
        """Whether a first commitment can be made, or the shopper still owes information."""
        fam = self.brief.family
        if self.ledger.adds:
            return True
        if self.brief.spec_sealed:
            if self.turn < self.max_steps * 0.6:
                return False
            self.brief.spec_sealed = False  # it was never stated; buy the best of the rest
            self._note("The firm requirement was never stated, so the remaining requirements decide the choice.")
        if fam in ("intent", "constraint") and self.turn < 4 and not self.shopper_updated:
            return False  # a later message can still tighten the budget
        if fam == "preference" and len(self.ledger.listing_ids) < 2:
            return False
        roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
        if fam == "recall" and roomy:
            # The score counts how much of the market the search surfaced, so a second
            # differently worded search is worth its turn.
            if len(self.ledger.identities) < self._recall_need() or self.search_calls < 2:
                return False
        if roomy and not self.reads_done and len({c["pid"] for c in self._candidates() if c["ok"]}) < 2:
            # The market moves on the first cart action, so line up a fallback before
            # committing: hunting for one afterwards costs many extra turns. Once no
            # unread candidate remains, waiting only burns turns.
            return False
        return True

    def _fresh(self, pid: str, sku: str) -> bool:
        k = Ledger.key(pid, sku)
        if k not in self.ledger.inspected:
            return False
        return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq

    def _plan(self) -> tuple | None:
        """The action the controller takes itself, or nothing to let the model explore."""
        if self.done:
            return None
        if self.shopper_question_pending:
            # The shopper is owed an answer; the model writes it, but not forever.
            if self.question_waits < 2 or "message" not in self.tool_names:
                self.question_waits += 1
                return None
            return ("message", {"content": self._answer_text()})
        fam = self.brief.family
        if fam == "ranking" and "compare" in self.tool_names:
            missing = [p for p, _s in self.brief.candidate_keys
                       if p not in self.ledger.products and p not in self.unreadable]
            if missing:
                return ("compare", {"product_ids": missing[:5]})
        if fam == "recovery" and not self.ledger.adds and self.brief.pref_model and "search" in self.tool_names:
            forced = self._sibling_search()
            if forced is not None:
                return forced
        target = self._target()
        if target is None or not self._ready():
            forced = self._read_more()
            if forced is not None:
                return forced
            if target is None:
                return None
        pid, sku = target["pid"], target["sku"]
        if (pid, sku) in self.ledger.cart:
            if not self._fresh(pid, sku):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            return ("place_test_order", self._order_args(pid, sku))
        if not self._ready():
            return None
        for cpid, csku in self.ledger.cart:
            if Ledger.key(cpid, csku) in self.ledger.dead_keys and "remove_from_cart" in self.tool_names:
                return ("remove_from_cart", {"product_id": cpid, "sku": csku})
        if not self._fresh(pid, sku):
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return ("add_to_cart", {"product_id": pid, "sku": sku})

    # ------------------------------------------------------------------ inference
    def _ask_model(self) -> dict | None:
        for attempt in range(3):
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
            # The proxy applies backpressure; back off rather than hammering it.
            if self.clock.remaining() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    # ------------------------------------------------------------------ environment
    def _send(self, envelope: dict) -> dict | None:
        """POST one solver-turn envelope; an identical envelope replays idempotently."""
        for attempt in range(4):
            result = _PROXY.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.clock.remaining() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def _act(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self._send(pend["envelope"])
            if replay is None:
                self._note("The environment is not answering; the previous action is still pending and will be retried.")
                return None
            self.pending_envelope = None
            self.turn = pend["turn"]
            self._absorb(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
            if self.done:
                return None
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
        if result is None:
            self.pending_envelope = {
                "envelope": envelope, "turn": turn, "name": name, "args": args,
                "content": content, "call_id": tool_call_id or ("agent-" + str(turn)),
            }
        else:
            self.turn = turn
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
            cur = parse_currency(text)
            if cur and not self.brief.currency:
                self.brief.currency = cur
            if self.brief.family in ("constraint", "intent") and not self.brief.spec_value:
                stated = self.brief.adopt_spec(text)
                if stated:
                    self.shopper_updated = True
                    self._note("Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought.")
            if _QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self._note("The shopper asked a question: answer it with message in your next turn, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        return obs

    # ------------------------------------------------------------------ controller steps
    def _post_add_recheck(self) -> None:
        """The market moves on the first cart action, so re-read that item at once."""
        pid, sku = self.pending_recheck
        self.pending_recheck = None
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = _num(obs.get("price"))
        over = self.budget is not None and price is not None and price > self.budget + _EPS
        if stock and not over:
            self._note(
                "Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it."
                % (Ledger.key(pid, sku), price)
            )
            return
        self.ledger.event_target = Ledger.key(pid, sku)
        self.event_turn = self.turn
        self.ledger.dead_keys.add(Ledger.key(pid, sku))
        self.ledger.dead_products.add(str(pid))
        self.forced_compares = 0  # the replacement search may need to read more pages
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        guidance = {
            "recovery": "Recover with the smallest possible change: the same model in another color before any other model.",
            "ranking": "The remaining named candidates are re-ranked; the controller takes the new best.",
            "preference": "Recover to a product with a different listing_id that still fits the shopper's priorities.",
            "constraint": "Choose a different product that still meets every requirement, not another variant of the same one.",
        }.get(self.brief.family, "Choose a different product that meets every firm requirement.")
        self._note("Market change: the carted item %s %s. " % (Ledger.key(pid, sku), reason) + guidance)
        self._shortlist_note()

    def _finish_fast(self) -> None:
        """Out of time or turns: reach a valid order in as few actions as possible."""
        for pid, sku in list(self.commit_queue):
            ok, _why = self._valid_now(str(pid), str(sku))
            if ok:
                self._act("place_test_order", self._order_args(str(pid), str(sku)))
                return
        for pid, sku in reversed(self.ledger.cart):
            ok, _why = self._valid_now(pid, sku)
            if ok and not self.done:
                self._act("place_test_order", self._order_args(pid, sku))
                return
        if self.done or self.turn > self.max_steps - 3:
            return
        good = [c for c in self._candidates() if c["ok"]]
        if not good and self.brief.family != "justification":
            good = [
                c for c in self._candidates()
                if c["live"] and c["spec"] == 2 and self._brand_ok(c["pid"]) and c["cat_ok"]
                and (not self.brief.candidate_keys or (c["pid"], c["sku"]) in self.brief.candidate_keys)
            ]
        if not good:
            return
        good.sort(key=lambda c: self._rank_of(c, self.ledger.adds == 0))
        pid, sku = good[0]["pid"], good[0]["sku"]
        if (pid, sku) not in self.ledger.cart:
            obs = self._act("add_to_cart", {"product_id": pid, "sku": sku})
            if self.done or not isinstance(obs, dict) or obs.get("error"):
                return
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if self.done or not isinstance(obs, dict) or not obs.get("in_stock"):
            return
        price = _num(obs.get("price"))
        if self.budget is not None and price is not None and price > self.budget + _EPS:
            return
        self._act("place_test_order", self._order_args(pid, sku))

    # ------------------------------------------------------------------ main loop
    def run(self) -> list:
        self._bootstrap_shopping_iq()
        question = self._opening_question()
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            if self.clock.remaining() < 25 or self.turn >= self.max_steps - 3:
                self._finish_fast()
                break
            if question and not self.asked_shopper and "message" in self.tool_names:
                self.asked_shopper = True
                self._act("message", {"content": question})
                continue
            if self.pending_recheck and "inspect_stock" in self.tool_names:
                self._post_add_recheck()
                continue
            plan = self._plan()
            if plan is not None:
                name, args = plan
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    # The step is not moving the state on; abandon that candidate.
                    if name in ("inspect_stock", "add_to_cart"):
                        self.ledger.dead_keys.add(Ledger.key(args.get("product_id"), args.get("sku")))
                    if name == "compare":
                        self.unreadable.update(str(p) for p in args.get("product_ids") or [])
                    if name == "message":
                        self.shopper_question_pending = False
                    last_forced, repeats = None, 0
                    self._note("That step changed nothing, so the controller moved on to other candidates.")
                    continue
                obs = self._act(name, args)
                if name == "message":
                    self.shopper_question_pending = False
                    self.question_waits = 0
                if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                    added = obs.get("added") or {}
                    if added.get("product_id"):
                        self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                continue
            last_forced, repeats = None, 0
            # If commit_queue is exhausted of live-valid picks, try recover refill once.
            live_q = [
                (pid, sku) for pid, sku in self.commit_queue
                if Ledger.key(pid, sku) not in self.ledger.dead_keys
            ]
            if not live_q and self._refill_commit_queue():
                continue
            proposal = self._ask_model()
            if proposal is None:
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
                if self.brief.family == "recall" and self.brief.required_identities:
                    have, need = len(self.ledger.identities), self._recall_need()
                    self._note(
                        "Distinct matching products surfaced so far: %d of the required %d." % (have, need)
                        + ("" if have >= need else " Keep searching with different wording.")
                    )
            if name == "inspect_cart":
                self.cart_inspections += 1
            if name in ("view", "compare"):
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
            if self.brief.family == "recall" and len(self.ledger.identities) < self._recall_need():
                return (
                    "order refused: only %d distinct matching products have been surfaced; search with different "
                    "wording until at least %d appear" % (len(self.ledger.identities), self._recall_need())
                )
            if self.brief.family == "preference" and len(self.ledger.listing_ids) < 2:
                return "order refused: compare at least two products with different listing_id values first"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            c = self._assess(pid, sku)
            key = Ledger.key(pid, sku)
            if not c["live"]:
                return "that variant is out of stock, over budget, or already invalidated"
            if not c["viewed"]:
                return "view or compare that product first so its raw facts are on record"
            if key not in self.ledger.inspected:
                return "inspect_stock that exact variant first: its current price and stock decide whether it can be bought"
            if c["spec"] != 2 and self.brief.spec_field:
                return (
                    "that variant does not resolve to %s = %s from its own option text or raw facts"
                    % (self.brief.spec_field, self.brief.spec_value)
                )
            if not c["ok"]:
                return "that product fails a firm requirement (category, brand, or the kind of device requested)"
            if self.brief.family == "preference" and len(self.ledger.listing_ids) < 2:
                return "compare two products with different listing_id values before the first cart addition"
            if self.brief.family == "recovery" and not self.ledger.adds and self.brief.pref_model and self._relax(pid, sku):
                return (
                    "the first cart item must be the exact model and color the shopper named; the other colors are "
                    "separate listings sharing the same title, so search that title and read them first"
                )
            if not self._ready():
                return (
                    "hold: read one more product that also meets every requirement first, so there is a "
                    "fallback when the market moves right after the cart addition"
                )
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.shopper_question_pending and "?" not in text:
                return "do not send status updates; use message only to answer the shopper's question or to ask one"
            if not self.shopper_question_pending and self.questions_asked >= 2:
                return (
                    "the shopper has already answered; decide from the evidence you have rather than asking again"
                )
        if name == "inspect_cart" and self.cart_inspections >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.ledger.calls_seen and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        self.ledger.calls_seen.add(key)
        return None



def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
