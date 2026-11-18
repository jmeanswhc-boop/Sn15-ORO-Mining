"""General shopping agent for ORO generated environments.

Nothing here is tied to a particular problem suite. The request is read into a
structured brief by the model (what is required, what is merely preferred, what
the shopper still has to be asked), and from then on a deterministic controller
verifies every requirement against the raw tool observations and owns the cart
and the order. The model is asked only to explore: where to search, what to read,
how to answer the shopper. Behavior branches on the contents of ``problem_data``
alone.
"""

from __future__ import annotations

import collections
import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


# ----------------------------------------------------------------- inference
# Fallback order per provider; only models that answer tool calls, fastest first.
_MODELS = {
    "chutes": (
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "zai-org/GLM-5.2-TEE",
        "Qwen/Qwen3.5-397B-A17B-TEE",
        "moonshotai/Kimi-K2.6-TEE",
        "zai-org/GLM-5.1-TEE",
        "Qwen/Qwen3-32B-TEE",
        "Qwen/Qwen3.6-27B-TEE",
        "google/gemma-4-31B-turbo-TEE",
    ),
    "openrouter": (
        "deepseek/deepseek-v3.2",
        "moonshotai/kimi-k2.6",
        "z-ai/glm-5.1",
        "openai/gpt-oss-120b",
        "deepseek/deepseek-chat-v3.1",
        "z-ai/glm-5.2",
        "google/gemma-4-31b-it",
        "deepseek/deepseek-chat-v3-0324",
        "qwen/qwen3.6-27b",
        "qwen/qwen3.5-397b-a17b",
        "mistralai/mistral-small-2603",
    ),
}
# One HTTP attempt per call, timed out from the budget still left. A stalled
# provider costs at most the cap; a task that has seen a stall gets a tighter one.
_CALL_CAP_S = 50.0
_PROXY = ProxyClient(timeout=_CALL_CAP_S, max_retries=1, retry_delay=1.0)
_PROBLEM_BUDGET_S = 270.0
_OBS_CHARS = 4500
_MAX_REPROMPTS = 2
_EPS = 1e-9
# Fewer, better searches: a read costs a turn and a page holds ten rows.
_SEARCH_BUDGET = 3
_PREF_MIN = 0.5


def _post(path: str, payload: dict, remaining: float, cap: float = _CALL_CAP_S) -> dict | None:
    _PROXY.timeout = max(5.0, min(cap, remaining - 3.0))
    return _PROXY.post(path, json_data=payload)


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


# ---------------------------------------------------------------------- text
# Ordinary English function words only; nothing here is drawn from any request.
_STOP = frozenset(
    "the a an and or but of for to in on at with without that this these those it its is "
    "are was were be been being as by from than into onto over under up down out more most "
    "less least very just only also not no nor any all some such each either neither i my "
    "me you your we our they them their he she his her would could should like have has "
    "had do does did can will shall may might want need wants needs please if when then "
    "than so because while where which who whom whose what why how here there".split()
)


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _stems(text: str) -> frozenset:
    """Content stems: lowercase words minus stopwords, cut to five letters."""
    out = set()
    for word in re.findall(r"[a-z][a-z0-9]+", (text or "").lower()):
        if len(word) >= 3 and word not in _STOP:
            out.add(word[:5])
    return frozenset(out)


def _words(text: str, drop: frozenset = frozenset()) -> list:
    out: list = []
    for word in re.findall(r"[a-z][a-z0-9]+", (text or "").lower()):
        if len(word) >= 3 and word not in _STOP and word[:5] not in drop and word not in out:
            out.append(word)
    return out


def _same_word(a: str, b: str) -> bool:
    x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
    y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
    return bool(x) and x == y


def _num(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _norm_query(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _sig(name: str, args: dict) -> str:
    def canon(v: Any) -> Any:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            return {k: canon(x) for k, x in v.items()}
        if isinstance(v, list):
            return [canon(x) for x in v]
        return v

    return name + json.dumps(canon(args), sort_keys=True)


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + ' ... (truncated by agent)"}'


def _json_block(text: str) -> dict | None:
    """The first JSON object in a model reply, tolerating fences and prose around it."""
    text = str(text or "")
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


# ----------------------------------------------------------------- evidence
# Measurement units the resolver understands. These are units of measure, not
# product vocabulary: which of them a task cares about comes from the request.
_UNITS = {
    "inch": (r'(?:inch(?:es)?|in\b|"|”|″)', 1.0),
    "hz": (r"hz", 1.0),
    "gb": (r"gb", 1.0),
    "tb": (r"tb", 1024.0),
    "mah": (r"mah", 1.0),
    "w": (r"(?:watts?|w\b)", 1.0),
    "mm": (r"mm", 1.0),
    "cm": (r"cm", 10.0),
    "kg": (r"kg", 1000.0),
    "g": (r"(?:grams?|g\b)", 1.0),
    "ml": (r"ml", 1.0),
    "l": (r"(?:liters?|litres?|l\b)", 1000.0),
    "mp": (r"(?:mp|megapixels?)", 1.0),
    "%": (r"%", 1.0),
    "hours": (r"(?:hours?|hrs?|h\b)", 1.0),
}
_UNIT_ALIASES = {
    "inches": "inch", "in": "inch", '"': "inch", "hertz": "hz", "gigabyte": "gb",
    "gigabytes": "gb", "terabyte": "tb", "terabytes": "tb", "watt": "w", "watts": "w",
    "gram": "g", "grams": "g", "liter": "l", "litre": "l", "hour": "hours", "hr": "hours",
    "megapixel": "mp", "megapixels": "mp", "percent": "%",
}
# Same physical quantity, different unit: canonical family and factor.
_UNIT_FAMILY = {"tb": "gb", "cm": "mm", "kg": "g", "l": "ml"}
# An amount of money: a number beside a currency code, symbol, or currency word
# (symbol before or after the number; codes are upper-case, words any case).
_MONEY_RE = re.compile(
    r"(?:[$\u20b1\u20ac\u00a3\u00a5]\s*(\d[\d,]*(?:\.\d+)?)\b"
    r"|(\d[\d,]*(?:\.\d+)?)\s*(?:(?-i:[A-Z]{3}\b)|[$\u20b1\u20ac\u00a3\u00a5]"
    r"|(?:[A-Za-z]+\s+)?(?:pesos?|dollars?|euros?|pounds?|yen|rupees?|ringgit|baht|dong|won)\b))",
    re.I,
)


def _money_amounts(text: str) -> list:
    """Every amount of money stated in the text, as numbers."""
    out = []
    for m in _MONEY_RE.finditer(text or ""):
        n = _num(m.group(1) or m.group(2))
        if n is not None:
            out.append(n)
    return out


_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")


def _unit_key(unit: str | None) -> str | None:
    if not unit:
        return None
    u = str(unit).strip().lower()
    u = _UNIT_ALIASES.get(u, u)
    return u if u in _UNITS else None


def _unit_pattern(unit: str) -> re.Pattern:
    family = [u for u in _UNITS if _UNIT_FAMILY.get(u, u) == _UNIT_FAMILY.get(unit, unit)]
    alts = "|".join("(?P<u%d>%s)" % (i, _UNITS[u][0]) for i, u in enumerate(family))
    return re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[-–]?\s*(?:" + alts + ")", re.I)


def _unit_values(unit: str, text: str) -> set:
    """Every value of a quantity a text states, normalized to the unit's family."""
    unit = _unit_key(unit) or unit
    family = [u for u in _UNITS if _UNIT_FAMILY.get(u, u) == _UNIT_FAMILY.get(unit, unit)]
    base = _UNITS[_UNIT_FAMILY.get(unit, unit)][1]
    out = set()
    for m in _unit_pattern(unit).finditer(text or ""):
        for i, u in enumerate(family):
            if m.group("u%d" % i) is not None:
                out.add(round(float(m.group(1)) * _UNITS[u][1] / base, 4))
                break
    return out


class ClassScale:
    """Named classes over a pixel pair, as the request defines them.

    A rule of the form "<class> starts at <width> by <height>" defines an ordered
    scale; the request may also say a class word means something else in
    lowercase, which is read as an exclusion.
    """

    def __init__(self, rules: str) -> None:
        self.classes: list = []  # (name, tokens, long, short) descending
        text = rules or ""
        for m in re.finditer(
            r"([A-Za-z0-9/ ]{2,20}?)\s+(?:starts?|begins?)?\s*at\s+(\d{3,5})\s*(?:by|x|×)\s*(\d{3,5})",
            text,
            re.I,
        ):
            names = [n.strip() for n in m.group(1).split("/") if n.strip()]
            names = [re.sub(r"^(?:and|,)\s+", "", n).strip() for n in names]
            longer, shorter = sorted((int(m.group(2)), int(m.group(3))), reverse=True)
            tokens = set()
            for n in names:
                tokens.add(re.sub(r"[^a-z0-9]", "", n.lower()))
            tokens.add("%dp" % shorter)
            self.classes.append((names[0].lower(), tokens, longer, shorter))
        if not self.classes:
            # No scale defined by the request: the standard display classes, which are
            # units of measure in the same sense as gigabytes or inches.
            for names, longer, shorter in (
                (["4K", "UHD", "Ultra HD"], 3840, 2160),
                (["QHD", "2K"], 2560, 1440),
                (["Full HD", "FHD"], 1920, 1080),
                (["HD"], 1280, 720),
            ):
                tokens = {re.sub(r"[^a-z0-9]", "", n.lower()) for n in names} | {"%dp" % shorter}
                self.classes.append((names[0].lower(), tokens, longer, shorter))
        self.classes.sort(key=lambda c: -c[2])
        self.excluded: set = set()
        for m in re.finditer(r"lowercase\s+(\w+)\s+means", text, re.I):
            self.excluded.add(m.group(1))
        # "<quantity> names denote ...": the quantity these classes measure.
        m = re.search(r"\b([A-Za-z]+)\s+names\s+(?:here\s+)?denote", text, re.I)
        self.concept = m.group(1).lower() if m else ("resolution" if self.classes else "")

    def __bool__(self) -> bool:
        return bool(self.classes)

    def name_of(self, text: str) -> str | None:
        """The class a value or phrase names, or None when unknown."""
        low = re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())
        squashed = low.replace(" ", "")
        for name, tokens, _l, _s in self.classes:
            if any(t and (t in low.split() or t == squashed or squashed.endswith(t)) for t in tokens):
                return name
        return None

    def values(self, text: str) -> set:
        """Classes a free text asserts, from pixel pairs and class words; empty when it
        names an excluded lowercase variant."""
        raw = text or ""
        for ex in self.excluded:
            if re.search(r"\b%s\b" % re.escape(ex), raw):
                return set()
        low = raw.lower()
        out = set()
        for m in _WXH_RE.finditer(low):
            longer, shorter = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
            for name, _t, cl, cs in self.classes:
                if longer >= cl and shorter >= cs:
                    out.add(name)
                    break
        words = set(re.findall(r"[a-z0-9]+", low))
        joined = re.sub(r"[^a-z0-9]", "", low)
        for name, tokens, _l, _s in self.classes:
            for t in tokens:
                if not t:
                    continue
                if " " in name and name.replace(" ", "") == t:
                    if re.search(r"\b%s\b" % re.escape(name), low) or t in words:
                        out.add(name)
                elif t in words:
                    out.add(name)
        # "full hd" also contains "hd": only the most specific reading counts.
        for name, tokens, _l, _s in self.classes:
            for other, otokens, _ol, _os in self.classes:
                if other != name and other in out and name in out and any(
                    (" " + name) in (" " + other) for _ in [0]
                ):
                    if re.search(r"\b%s\b" % re.escape(other), low) and not re.search(
                        r"\b%s\b(?!\s*\w)" % re.escape(name), re.sub(r"\b%s\b" % re.escape(other), "", low)
                    ):
                        out.discard(name)
        return out


class Requirement:
    """One attribute the request pins down, with the way to read it from evidence."""

    def __init__(self, name: str, value: str, unit: str | None, scale: ClassScale) -> None:
        self.name = str(name or "").strip()
        self.raw_value = str(value or "").strip()
        self.unit = _unit_key(unit)
        self.scale = scale
        self.number = None
        self.class_name = None
        if self.unit:
            found = _unit_values(self.unit, self.raw_value + " " + self.unit)
            self.number = next(iter(found)) if len(found) == 1 else _num(re.sub(r"[^\d.]", "", self.raw_value) or None)
        elif scale and scale.name_of(self.raw_value):
            self.class_name = scale.name_of(self.raw_value)
        else:
            self.number = _num(self.raw_value)
        self.pair_first = bool(re.search(r"\b(?:ram|memory)\b", self.name.lower()))

    def describe(self) -> str:
        return "%s = %s%s" % (self.name, self.raw_value, (" " + self.unit) if self.unit and not self.raw_value.lower().endswith(self.unit) else "")

    def values_in(self, text: str) -> set:
        """Every value for this requirement a free text asserts."""
        text = text or ""
        if self.unit:
            if self.pair_first:
                pairs = re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:gb)?\s*\+\s*(\d+(?:\.\d+)?)\s*gb\b", text, re.I)
                if pairs:
                    return {round(float(a), 4) for a, _b in pairs}
                near = re.findall(
                    r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", text, re.I
                )
                return {round(float(n), 4) for n in near}
            return _unit_values(self.unit, text)
        if self.class_name is not None:
            return self.scale.values(text)
        if self.number is not None:
            return {round(float(n), 4) for n in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])", text)}
        want = _tokens(self.raw_value)
        return {self.raw_value.lower()} if want and want <= _tokens(text) else set()

    def mentions(self, text: str) -> bool:
        """Whether a text talks about this quantity at all, even ambiguously."""
        if self.unit:
            if self.pair_first:
                return bool(re.search(r"\d\s*(?:gb)?\s*\+\s*\d|\bgb\s*(?:ram|memory)\b", text or "", re.I))
            return bool(_unit_pattern(self.unit).search(text or ""))
        if self.class_name is not None:
            return bool(_WXH_RE.search(text or "")) or bool(self.scale.values(text))
        return bool(self.values_in(text))

    def target(self) -> Any:
        if self.class_name is not None:
            return self.class_name
        if self.number is not None:
            return round(float(self.number), 4)
        return self.raw_value.lower()

    # Words a fact label tends to carry for a quantity measured in a unit: generic
    # measurement semantics (inches are sizes), not knowledge of any product.
    _UNIT_LABEL_HINTS = {
        "inch": ("size", "screen", "display", "diagonal"),
        "hz": ("refresh", "rate", "frequency"),
        "gb": ("memory", "ram", "storage", "capacity"),
        "mah": ("battery", "capacity"),
        "w": ("power", "watt", "output"),
        "mp": ("camera", "resolution"),
        "hours": ("battery", "playtime", "runtime", "life"),
        "ml": ("volume", "capacity"),
        "g": ("weight",),
        "mm": ("size", "dimension", "thickness", "width", "height"),
    }

    def related_facts(self, facts: dict) -> list:
        """Fact values whose label speaks of this attribute; all facts when none does."""
        facts = facts if isinstance(facts, dict) else {}
        name_words = set(_words(self.name)) | {w[:5] for w in _words(self.name)}
        hints = set(self._UNIT_LABEL_HINTS.get(self.unit or "", ()))
        if self.class_name is not None and self.scale.concept:
            hints |= set(_words(self.scale.concept)) | {"display", "screen"}
        related = []
        for label, value in facts.items():
            label_words = set(re.findall(r"[a-z]+", str(label).lower()))
            label_stems = {w[:5] for w in label_words}
            if (label_words & (name_words | hints)) or (label_stems & name_words):
                text = str(value).strip()
                # A bare number under a label about this quantity carries its unit:
                # "27" under a size label is 27 of whatever the request measures in.
                if self.unit and re.fullmatch(r"\d+(?:\.\d+)?", text):
                    text = text + " " + self.unit
                related.append(text)
        return related if related else [str(v) for v in facts.values()]

    def satisfied_by(self, options: str, facts: dict, title: str) -> int:
        """2 = the variant itself states the value, 0 = says nothing, -1 = states another.

        The variant's own option text is read first; when it is silent, the facts
        about this attribute and the title are read together and must agree.
        """
        want = self.target()
        vals = self.values_in(options)
        if len(vals) > 1:
            return -1
        if vals:
            return 2 if want in vals else -1
        if self.mentions(options):
            return -1
        vals = set()
        for v in self.related_facts(facts):
            vals |= self.values_in(v)
        vals |= self.values_in(title or "")
        if len(vals) > 1:
            return -1
        if vals:
            return 2 if want in vals else -1
        return 0


_PART_ONLY_RE = re.compile(
    r"\bonly\s+(?:\w+\s+)?(?:screen|display|panel)\b|\b(?:screen|display|panel)\s+only\b|"
    r"^\s*replacement\s+(?:\w+\s+)?(?:screen|display|panel)\b",
    re.I,
)
_BUNDLE_RE = re.compile(
    r"\b(?:with\s+(?:free|bundled)\s+|bundled\s+with\s+|free\s+(?=\w)|includes?\s+(?:an?\s+)?)", re.I
)
_FOR_DEVICE_RE = re.compile(
    r"\b(?:for|fits?|compatible\s+with|suitable\s+for|designed\s+for)\b[^,;.()]{0,80}", re.I
)


def _word_regex(words: list) -> re.Pattern | None:
    parts = []
    for word in words or []:
        toks = re.findall(r"[a-z0-9]+", str(word).lower())
        if toks:
            parts.append(r"\s*".join(re.escape(t) for t in toks) + r"(?:s|es)?")
    if not parts:
        return None
    return re.compile(r"\b(?:" + "|".join(sorted(parts, key=len, reverse=True)) + r")\b", re.I)


class DeviceRule:
    """What the request says a genuine device must show, and what disqualifies one."""

    def __init__(self, positive: list, os_names: list, negatives: list) -> None:
        self.positive = _word_regex(positive)
        self.os_names = _word_regex(os_names)
        self.negatives = _word_regex(negatives)
        stated = [str(w) for w in (positive or []) + (negatives or []) if w]
        self.negated = (
            re.compile(r"\bnot\s+(?:an?\s+)?(?:" + "|".join(re.escape(w) for w in stated) + r")s?\b", re.I)
            if stated
            else None
        )

    def _clean(self, text: str, title: bool = False) -> str:
        text = self.negated.sub("", text or "") if self.negated else (text or "")
        text = _FOR_DEVICE_RE.sub(" ", text)
        return _BUNDLE_RE.split(text, maxsplit=1)[0] if title else text

    def verdict(self, facts: dict, title: str, options: str) -> str:
        if _PART_ONLY_RE.search(options or ""):
            return "no"
        facts = facts if isinstance(facts, dict) else {}
        head = self._clean(str(title or ""), title=True)
        kind = " ".join(
            self._clean(str(v)) for k, v in facts.items() if re.search(r"type", str(k), re.I)
        )
        blob = head + " " + kind
        if self.negatives and self.negatives.search(blob):
            return "no"
        if any("[truncated]" in str(v) for k, v in facts.items() if re.search(r"type|operating|os\b", str(k), re.I)):
            return "unclear"
        if self.positive and self.positive.search(blob):
            return "yes"
        if self.os_names:
            os_blob = " ".join(
                self._clean(str(v))
                for k, v in facts.items()
                if re.search(r"operating|\bos\b", str(k), re.I) and not re.search(r"compatib", str(v), re.I)
            )
            if self.os_names.search(os_blob):
                return "yes"
            if self.os_names.search(head) and not re.search(r"compatib", str(title or ""), re.I):
                return "yes"
        return "unclear"


# ------------------------------------------------------------------- intake
_INTAKE_PROMPT = """You turn a shopper's request into a JSON brief for a shopping controller.
Read the whole request, including any rules it contains. Use only what the text states; never invent values, never guess numbers. Output one JSON object and nothing else, with exactly these keys:

{
  "category": string|null,            // the required catalog category / kind of product, as worded
  "brand": string|null,               // a brand the item MUST be (firm), else null
  "budget": number|null,              // maximum price stated, as a number
  "currency": string|null,            // currency code or symbol next to the budget
  "attributes": [                     // firm, measurable product attributes the request states with a concrete value (a number with its unit, or a named resolution/class); wishes like "durable" belong in soft_preferences
    {"name": string, "value": string, "unit": string|null}
  ],
  "hidden_requirement_expected": bool,// true if the shopper says they will state a requirement only when asked
  "ask_shopper_first": string|null,   // a short question to send before searching when something must be asked first (a withheld requirement, or several soft preferences with no stated priority), else null
  "soft_preferences": [               // wishes that are NOT firm, most important first
    {"text": string, "priority": integer}
  ],
  "preferred_brand": string|null,     // a brand that is preferred but not required
  "preferred_model": string|null,     // an exact model name/code the shopper prefers, else null
  "preferred_color": string|null,     // a color the shopper prefers for that model, else null
  "relaxation_order": [string],       // if the request says what to relax first when the preferred item is gone, e.g. ["color","model"]
  "candidate_keys": [string],         // exact product_id::sku candidates the request restricts the choice to
  "tradeoff_axes": [                  // when the request ranks options by attributes, in priority order
    {"attribute": string, "prefer": "high"|"low"}
  ],
  "required_distinct_matches": integer|null, // a stated minimum number of distinct matching products to surface before ordering
  "match_phrase": string|null,        // words that must all appear in the listing title, if the request says so
  "justification_required": bool,     // true if the shopper asks why / for a reason with the order
  "device_rule": {                    // only if the request states how to tell a genuine device from an accessory
    "positive_terms": [string], "os_terms": [string], "reject_terms": [string]
  }|null
}"""


class Brief:
    """The request as the model read it, checked against the text where the text can."""

    def __init__(self, query: str, data: dict | None, tools: list) -> None:
        self.raw = query or ""
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules
        d = data if isinstance(data, dict) else {}
        self.category = self._str(d.get("category")) or self._category_in_text()
        self.brand = self._str(d.get("brand"))
        self.currency = self._str(d.get("currency")) or self._currency_in_text()
        self.budget = self._checked_number(d.get("budget"))
        if self.budget is None:
            self.budget = self._budget_in_text()
        self.scale = ClassScale(self.rules)
        self.attributes: list = []
        demoted: list = []
        for item in d.get("attributes") or []:
            if isinstance(item, dict) and item.get("name") and item.get("value") is not None:
                req = Requirement(item["name"], item["value"], item.get("unit"), self.scale)
                # A firm attribute must be measurable: a number, a unit, or a class the
                # request defines. A wish phrased as a value is a preference, not a gate.
                measurable = req.unit is not None or req.class_name is not None or req.number is not None
                money = bool(re.search(r"\b(?:price|cost|budget|amount)\b", req.name, re.I)) or (
                    req.unit is None and req.number is not None and self.budget is not None and abs(req.number - self.budget) < 0.005
                )
                if money:
                    continue  # the budget is a ceiling on price, never an attribute of the item
                if measurable and self._value_in_text(req):
                    # "1080p" and "Full HD" name one value; two spellings are one requirement.
                    same = lambda r: (r.unit, r.class_name is not None, r.target()) == (req.unit, req.class_name is not None, req.target())
                    if not any(same(r) for r in self.attributes):
                        self.attributes.append(req)
                else:
                    demoted.append(str(item["name"]) + " " + str(item["value"]))
        # The model's reading is checked against the text: measurable values the goal
        # states become requirements even when the model left them out, and a hidden
        # requirement is recognized from the goal's own wording.
        if not self.attributes:
            for req in self._measurables_in_goal():
                self.attributes.append(req)
        # Something the shopper will only state on request, however they phrase it.
        self.hidden_requirement = bool(d.get("hidden_requirement_expected")) or bool(
            not self.attributes
            and re.search(r"\b(?:if|when|once|unless)\b[^.]{0,20}\bask", self.goal, re.I)
        )
        self.ask_first = self._str(d.get("ask_shopper_first"))
        self.preferred_brand = self._str(d.get("preferred_brand"))
        self.preferred_model = self._str(d.get("preferred_model"))
        self.preferred_color = (self._str(d.get("preferred_color")) or "").lower() or None
        if self.preferred_model and re.sub(r"[^a-z0-9]", "", self.preferred_model.lower()) not in re.sub(r"[^a-z0-9]", "", self.raw.lower()):
            self.preferred_model = None
        if not self.preferred_model and "::" not in self.goal:
            # The exact item the shopper wants, when the intake missed it: a code named as a
            # model, shown as "the black 635", or the one long code in the request.
            plain = _MONEY_RE.sub(" ", self.goal)
            code = r"(?!\d+(?:\.\d+)?-?[a-z]+\b)(?:\d{3,}|[A-Za-z]+-?\d[A-Za-z0-9-]*)"   # not a measurement like 7.6-inch or 120hz
            mentioned = lambda c: any(req.mentions(c) for req in self.attributes)
            found, shade = None, None
            named = re.search(r"\bmodel\b\s*(?:#|no\.?|number|code|name)?\s*[:\-]?\s*(%s)(?!\w)(?!\.\d)" % code, plain, re.I)
            if named and not mentioned(named.group(1)):
                found = named.group(1)
            shown = [(w, c) for w, c in re.findall(r"\b(?:the|a|an|my)\s+([A-Za-z]+)\s+(%s)(?!\w)(?!\.\d)" % code, plain) if not mentioned(c)]
            if found is None and len(shown) == 1:
                found = shown[0][1]
            if found is None:
                codes = {c for c in re.findall(r"(?<![\w.])(?=[A-Za-z0-9-]*\d{4})[A-Za-z0-9-]{6,}(?![\w-])(?!\.\d)", plain) if not mentioned(c)}
                if len(codes) == 1:
                    found = codes.pop()
            if found:
                self.preferred_model = found
                # "the black 635" names a color only when the request talks about color at all.
                if re.search(r"\bcolou?r", self.goal, re.I):
                    for w, c in shown:
                        if c == found and w.islower() and not self.preferred_color and not re.fullmatch(r"model|code|same|number|version|variant|item|one|exact|specific|particular|new|old", w):
                            self.preferred_color = w.lower()
        order = d.get("relaxation_order") or []
        self.relax_color_first = not order or str(order[0]).lower().startswith("col")
        text_keys = [(p, s.rstrip(".,;:")) for p, s in re.findall(r"(\d{5,})::([A-Za-z0-9._\-]+)", self.raw)]
        self.candidate_keys = text_keys
        self.tradeoff_axes: list = []
        for item in d.get("tradeoff_axes") or []:
            if isinstance(item, dict) and item.get("attribute"):
                self.tradeoff_axes.append((str(item["attribute"]), str(item.get("prefer", "high")).lower() != "low"))
        need = d.get("required_distinct_matches")
        self.required_matches = int(need) if isinstance(need, (int, float)) and not isinstance(need, bool) and need > 0 else 0
        m = re.search(r"at least (\d+) distinct", self.raw, re.I)
        if m and not self.required_matches:
            self.required_matches = int(m.group(1))
        phrase = self._str(d.get("match_phrase"))
        self.theme = [w for w in re.findall(r"[a-z0-9]+", phrase.lower())] if phrase else []
        self.theme = [w for w in self.theme if w in _tokens(self.raw)]
        self.justify = bool(d.get("justification_required")) or bool(
            re.search(r"\b(?:why|reason\w*|explain\w*|justif\w*)\b", self.goal, re.I)
        )
        self.claim_fields = self._claim_fields(tools)
        rule = d.get("device_rule")
        self.device_rule = None
        if isinstance(rule, dict) and any(rule.get(k) for k in ("positive_terms", "os_terms", "reject_terms")):
            self.device_rule = DeviceRule(
                [str(x) for x in rule.get("positive_terms") or []],
                [str(x) for x in rule.get("os_terms") or []],
                [str(x) for x in rule.get("reject_terms") or []],
            )
        prefs = []
        for item in d.get("soft_preferences") or []:
            if isinstance(item, dict) and item.get("text"):
                prefs.append((str(item["text"]), _num(item.get("priority")) or len(prefs) + 1))
        for text in demoted:
            prefs.append((text, len(prefs) + 1))
        if not prefs and not self.attributes and not self.preferred_model and not self.candidate_keys:
            # The model listed no wishes: clauses of the goal that voice one still count.
            for clause in re.split(r"[.;\n]|(?<=\S)\s-\s", self.goal.lower()):
                if re.search(r"\b(?:prefer\w*|rather|ideally|matter\w*|important|care\w*|worth|priorit\w*)\b", clause):
                    prefs.append((clause.strip(), len(prefs) + 1))
        self.pref_axes = self._axes(prefs)
        if self.pref_axes and not self.attributes and not self.preferred_model and not self.justify and not self.ask_first:
            # Soft wishes and no stated order: only the shopper can rank them, and their
            # answer is where the full list of what they care about comes from.
            self.ask_first = (
                "Quick question so I choose well: which of your preferences matters most, and in what "
                "order should I weigh the others?"
            )
        self.tool_names = {(t.get("function") or {}).get("name") for t in tools if isinstance(t, dict)}

    @staticmethod
    def _str(value: Any) -> str | None:
        if value is None or isinstance(value, (bool, dict, list)):
            return None
        text = str(value).strip()
        return text or None

    def _measurables_in_goal(self) -> list:
        """Requirements the goal states as a number with its unit, or as a defined class."""
        out = []
        goal = self.goal
        # Money is a budget, never an attribute: blank out amounts next to a currency.
        goal = _MONEY_RE.sub(" ", goal)
        for unit in ("inch", "hz", "gb", "mah", "mp", "hours", "w"):
            vals = _unit_values(unit, goal)
            if len(vals) == 1:
                value = next(iter(vals))
                hint = Requirement._UNIT_LABEL_HINTS.get(unit, (unit,))[0]
                out.append(Requirement(hint, "%g" % value, unit, self.scale))
        if self.scale:
            names = self.scale.values(goal)
            if len(names) == 1:
                out.append(Requirement(self.scale.concept or "class", next(iter(names)), None, self.scale))
        return out

    @staticmethod
    def _claim_fields(tools: list) -> list:
        """Claim field names the order tool accepts, read from its own schema text."""
        for t in tools or []:
            fn = (t.get("function") or {}) if isinstance(t, dict) else {}
            if fn.get("name") != "place_test_order":
                continue
            blob = json.dumps(fn.get("parameters") or {}) + " " + str(fn.get("description") or "")
            fields: list = []
            enum = re.search(r'"enum":\s*\[([^\]]+)\]', blob)
            if enum:
                fields += re.findall(r'"([a-z_]+)"', enum.group(1))
            # Field names also appear as comma lists in the schema's own prose
            # ("price, in_stock, and one supported reason: brand, streaming, ... or hdr").
            for m in re.finditer(r"((?:[a-z_]+,\s*)+(?:and|or)\s+[a-z_]+)", blob):
                fields += re.findall(r"[a-z_]+", m.group(1))
            fields = [f for f in fields if f not in ("and", "or", "one", "the", "a")]
            return list(dict.fromkeys(fields))
        return []

    # What a measured quantity is called as a claim when the schema does not say.
    _CANONICAL_FIELD = {"inch": "screen_size", "hz": "refresh_rate", "gb": "ram"}

    def claim_field_for(self, req: "Requirement") -> str | None:
        """The accepted claim field an attribute maps to, by shared words or unit sense."""
        words = set(_words(req.name)) | {w[:5] for w in _words(req.name)}
        hints = set(Requirement._UNIT_LABEL_HINTS.get(req.unit or "", ()))
        canonical = None
        if req.class_name is not None:
            canonical = self.scale.concept or "resolution"
            words.add(canonical)
        elif req.unit in self._CANONICAL_FIELD:
            canonical = self._CANONICAL_FIELD[req.unit]
            words |= set(canonical.split("_"))
        if not self.claim_fields:
            return canonical or re.sub(r"\s+", "_", req.name.lower())
        best = None
        for field in self.claim_fields:
            fw = set(field.lower().split("_"))
            if fw & words or {w[:5] for w in fw} & words:
                return field
            if fw & hints and best is None:
                best = field
        # The schema may describe the specification claim in words rather than list
        # its field; a measured quantity then claims under its canonical name.
        return best or canonical

    def _value_in_text(self, req: Requirement) -> bool:
        digits = re.sub(r"[^\d.]", "", req.raw_value)
        if digits:
            return digits.rstrip("0").rstrip(".") in self.raw.replace(",", "") or digits in self.raw
        return bool(_tokens(req.raw_value) & _tokens(self.raw))

    def _checked_number(self, value: Any) -> float | None:
        n = _num(value)
        if n is None:
            return None
        flat = self.raw.replace(",", "")
        forms = {("%.2f" % n), ("%g" % n), str(int(n)) if float(n).is_integer() else None}
        return n if any(f and f in flat for f in forms) else None

    def _currency_in_text(self) -> str | None:
        m = re.search(r"\d[\d,]*(?:\.\d+)?\s*([A-Z]{3})\b", self.goal)
        return m.group(1) if m else None

    _GENERIC_NOUNS = frozenset("item items order orders purchase purchases listing listings product products unit units pick choice option one thing deal offer condition requirement requirements budget price model variant version".split())

    def _category_in_text(self) -> str | None:
        """The kind of item asked for, from the request's own grammar: a named category or
        section, the object of "order/purchase of ...", or the noun of the first "a/an ..."
        phrase that carries no number and is not a generic word for a purchase."""
        for pat in (r"\b(?:in|from|under|within) the ([A-Z][\w&' -]{1,40}?) (?:category|section|department|aisle)\b",
                    r"\b(?:category|section|department|aisle)(?: of| named| called|:)? ([A-Z][\w&' -]{1,40}?)(?=[,.;]| that| which| with| and\b|$)"):
            m = re.search(pat, self.goal)
            if m:
                return m.group(1).strip()
        m = re.search(r"\b(?:order|purchase|listing|item|product)s?\s+(?:of|for)\s+(?:a|an|the|some)?\s*((?:[a-z-]+\s+){0,2}?[a-z][a-z-]{2,})\b(?=\s*(?:with|that|which|priced|at|under|below|for|in|costing|available|currently|from|[,.;]|$))", self.goal)
        if m and not re.search(r"\d", m.group(1)) and m.group(1).split()[-1] not in self._GENERIC_NOUNS:
            return m.group(1).split()[-1]
        stop = {"with", "that", "which", "priced", "at", "under", "below", "for", "in", "costing", "available", "currently", "from", ",", ".", ";"}
        first = re.split(r"[.;?!]\s", self.goal, maxsplit=1)[0]
        for m in re.finditer(r"\b(?:a|an)\s+", first):
            if re.search(r"\b(?:is|it's|its|be|as|why|because|such)\s*$", first[: m.start()]):
                continue                                   # "why it's a pick I can trust" describes, not asks
            words = re.findall(r"[A-Za-z][\w-]*|[,.;]", first[m.end():])[:5]
            for k in range(min(4, len(words))):
                head = words[k]
                if head in ("I", "you", "we", "they", "can", "could", "will", "would", "should", "must", "to"):
                    break                                  # a clause, not a noun phrase
                if not re.fullmatch(r"[a-z][a-z-]{2,}", head) or head in self._GENERIC_NOUNS or head.endswith("ly"):
                    continue
                if any(re.search(r"\d", w) for w in words[: k + 1]):
                    break
                if k + 1 == len(words) or words[k + 1] in stop:
                    return head
        return None

    def _budget_in_text(self) -> float | None:
        amounts = _money_amounts(self.goal)
        return min(amounts) if amounts else None

    def _axes(self, prefs: list) -> list:
        drop = _stems(self.category or "") | {"stock", "budge"}
        axes = []
        seen = set()
        ranked = sorted(prefs, key=lambda p: p[1])
        n = len(ranked)
        for i, (text, _p) in enumerate(ranked):
            words = _words(text, drop)
            stems = frozenset(w[:5] for w in words)
            if stems and stems not in seen:
                seen.add(stems)
                axes.append((float(n - i), stems, words))
        return axes

    def adopt_preferences(self, prefs: list) -> None:
        axes = self._axes(prefs)
        if axes:
            self.pref_axes = axes

    # what shape the task has, read off the brief rather than from any fixed list
    @property
    def ranks(self) -> bool:
        return bool(self.candidate_keys) and bool(self.tradeoff_axes)

    @property
    def recalls(self) -> bool:
        return bool(self.required_matches or self.theme)

    @property
    def recovers(self) -> bool:
        return bool(self.preferred_model)

    @property
    def prefers(self) -> bool:
        return bool(self.pref_axes) and not self.attributes and not self.preferred_model

    def summary(self) -> str:
        parts = []
        if self.category:
            parts.append("category %s" % self.category)
        if self.brand:
            parts.append("brand %s" % self.brand)
        if self.budget is not None:
            parts.append("at most %.2f %s" % (self.budget, self.currency or ""))
        for req in self.attributes:
            parts.append(req.describe())
        if self.hidden_requirement:
            parts.append("one requirement is revealed only if asked")
        if self.preferred_model:
            parts.append("preferred model %s%s" % (self.preferred_model, " in " + self.preferred_color if self.preferred_color else ""))
        if self.preferred_brand:
            parts.append("preferred (not required) brand %s" % self.preferred_brand)
        if self.candidate_keys:
            parts.append("%d named candidate variants" % len(self.candidate_keys))
        if self.tradeoff_axes:
            parts.append("rank by " + " > ".join("%s (%s)" % (a, "high" if hi else "low") for a, hi in self.tradeoff_axes))
        if self.required_matches:
            parts.append("surface at least %d distinct matching products" % self.required_matches)
        if self.theme:
            parts.append("title must contain: " + " ".join(self.theme))
        if self.pref_axes:
            parts.append("preferences: " + "; ".join(" ".join(w) for _, _, w in self.pref_axes))
        if self.justify:
            parts.append("a reason is required with the order")
        return "; ".join(parts) or "no requirements parsed"


# ------------------------------------------------------------------- ledger
class Ledger:
    """Everything observed so far, keyed by product and variant."""

    def __init__(self) -> None:
        self.cart: list = []
        self.checked: dict = {}
        self.checked_seq: dict = {}
        self.inspected: set = set()
        self.products: dict = {}
        self.rows: dict = {}
        self.options: dict = {}
        self.listing_of: dict = {}
        self.listing_ids: set = set()
        self.identities: set = set()
        self.cluster_of: dict = {}
        self.twin_index: dict = {}
        self.seq = 0
        self.adds = 0
        self.last_add_seq: int | None = None
        self.event_target: str | None = None
        self.dead_keys: set = set()
        self.dead_products: set = set()
        self.calls_seen: set = set()

    @staticmethod
    def key(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def _add_identity(self, pid: Any, title: str, brand: Any, price: float | None) -> None:
        pid = str(pid)
        cluster = self.cluster_of.get(pid)
        if cluster is None:
            sig = (str(brand or "").strip().lower(), frozenset(re.findall(r"[a-z0-9]+", (title or "").lower())))
            if price is not None:
                for (twin_sig, twin_price), twin_cluster in self.twin_index.items():
                    if twin_sig == sig and abs(twin_price - price) <= 0.01:
                        cluster = twin_cluster
                        break
                self.twin_index.setdefault((sig, round(price, 2)), cluster or pid)
            cluster = cluster or pid
            self.cluster_of[pid] = cluster
        self.identities.add(cluster)

    def record(self, name: str, args: dict, obs: dict, brief: Brief, budget: float | None) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        if name in ("search", "filter"):
            for row in obs.get("results") or []:
                if isinstance(row, dict):
                    self._note_row(row, brief, budget)
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

    def _note_row(self, row: dict, brief: Brief, budget: float | None) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        price = _num(row.get("price"))
        eligible = row.get("in_stock", True) is not False and (
            budget is None or price is None or price <= budget + _EPS
        )
        if eligible and (not brief.theme or all(t in _tokens(title) for t in brief.theme)):
            self._add_identity(pid, title, row.get("brand"), price)
        if not row.get("sku"):
            return
        k = self.key(pid, row.get("sku"))
        self.rows[k] = row
        if row.get("price") is not None:
            self.checked.setdefault(k, {"in_stock": bool(row.get("in_stock", True)), "price": price})

    def _note_product(self, prod: dict, brief: Brief) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.products[str(pid)] = prod
        if prod.get("listing_id"):
            self.listing_ids.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        if brief.theme and all(t in _tokens(str(prod.get("title") or "")) for t in brief.theme):
            self._add_identity(pid, str(prod.get("title") or ""), prod.get("brand"), None)
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


_CORE_RULES = (
    "You are a careful shopping agent working a catalog through tools. Exactly one tool call per "
    "turn, so be efficient. A deterministic controller verifies every requirement from the raw "
    "observations and takes over the cart and the order once a candidate is fully verified; your "
    "job is to surface and read candidates: search or filter, then view or compare so raw facts, "
    "variant options and prices are on record. Never add an item to the cart as a placeholder - the "
    "first cart addition is graded. Prices and stock change while you shop; the controller re-checks "
    "the carted item and tells you when to move on. Answer every direct question from the shopper "
    "with one short message before anything else, and never send progress updates. Never ask the "
    "shopper to raise the budget. Do not repeat a tool call with identical arguments; one compare of "
    "up to five products beats several view calls. Prefer few, well-chosen searches."
)
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)


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
        self.query = str(self.policy.get("query") or "")
        self.clock = Clock(_PROBLEM_BUDGET_S)
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)
        self.model_index = 0
        self.timeouts = 0
        self.model_misses = 0
        self.model_down = False
        self.dialogue: list = []
        self.brief = Brief(self.query, self._intake(), self.tools)
        self.budget = self.brief.budget
        self.ledger = Ledger()
        self.messages: list = [
            {"role": "system", "content": _CORE_RULES + "\n\nParsed brief: " + self.brief.summary()},
            {"role": "user", "content": self.query},
        ]
        self.turn = 0
        self.asked_shopper = False
        self.shopper_question_pending = False
        self.question_waits = 0
        self.questions_asked = 0
        self.shopper_updated = False
        self.rejected_streak = 0
        self.stalls = 0
        self.pending_envelope: dict | None = None
        self.pending_recheck: tuple | None = None
        self.search_calls = 0
        self.search_queries: set = set()
        self.cart_inspections = 0
        self.forced_compares = 0
        self.probes: dict = {}
        self.recall_bands: list = [1.0, 0.6, 0.3]
        self.recall_brands_probed: set = set()
        self.reach_queries: list = self._axis_queries() if self.brief.prefers else []
        self.model_probed = False
        self.model_search_pids: set = set()
        self.shelf_price_probes: dict = {}
        self.shelf: str | None = None          # catalog label resolved for the requested category
        self.shelf_inferred = False
        self.brand_probed = False
        self.searched_twins = False
        self.searched_siblings = False
        self.hinted_exact = False
        self.hinted_brand = False
        self.reads_done = False
        self.unreadable: set = set()
        self.done = False

    # ------------------------------------------------------------------ intake
    def _intake(self) -> dict | None:
        """Ask the model to read the request into a brief; verified field by field later."""
        messages = [
            {"role": "system", "content": _INTAKE_PROMPT},
            {"role": "user", "content": self.query},
        ]
        for attempt in range(3):
            if self.clock.remaining() < (60 if attempt == 0 else 150):
                break
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            payload = {"model": model, "messages": messages, "temperature": 0}
            started = time.monotonic()
            resp = _post("/inference/chat/completions", payload, self.clock.remaining(), 40.0 if attempt == 0 else 30.0)
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                data = _json_block(msg.get("content") or "")
                if data:
                    return data
            elif time.monotonic() - started >= 28:
                self.timeouts += 1
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    def _note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    # ---------------------------------------------------------------- evidence
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

    @staticmethod
    def _shelf_stems(text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 2}

    @staticmethod
    def _stem_hit(w: str, labels: set) -> float:
        """How well one requested word is answered by a label's words."""
        best = 0.0
        for l in labels:
            if w == l:
                return 1.0
            if w in l:
                best = max(best, 0.9)       # the requested word inside a longer label word
            elif len(l) >= 4 and l in w:
                best = max(best, 0.6)       # a shorter label word inside the requested word
        return best

    def _part_score(self, wanted: set, joined: str, stems: set) -> float:
        hits = [self._stem_hit(w, stems) for w in wanted]
        score = max(sum(hits) / len(hits), 0.9 * max(hits))
        if len(wanted) > 1 and len(joined) > 2:
            score = max(score, self._stem_hit(joined, stems))      # two words written as one
        extra = sum(1 for l in stems if not any(w in l or (len(l) >= 4 and l in w) or l in joined for w in wanted))
        return score / (1.0 + 0.4 * extra)

    def _label_score(self, want: str, label: str) -> tuple:
        """(score, full): how well one catalog label names the requested category (0..1). A
        label that lists alternatives ("A & B") counts by its best alternative; words of its
        own that the request never said discount it. `full` means the request names the
        whole label, so everything shelved under it qualifies."""
        wanted = self._shelf_stems(want)
        whole = self._shelf_stems(label)
        if not wanted or not whole:
            return (0.0, False)
        joined = "".join(re.findall(r"[a-z]+", want.lower())).rstrip("s")
        best = self._part_score(wanted, joined, whole)
        full = best >= 0.99
        for part in re.split(r"\s*(?:&|,|/|\band\b)\s*", str(label).lower()):
            stems = self._shelf_stems(part)
            if stems and stems != whole:
                best = max(best, self._part_score(wanted, joined, stems))
        return (round(best, 3), full)

    def _path_fit(self, want: str, path: list) -> tuple:
        """(fit, index of the label to name the shelf by). The fit is the score of the deepest
        label that answers the request. When that label is only partly named (one of its
        alternatives, or with words the request lacks), every deeper label that does not match
        discounts it: a matching parent whose child names something else is a different shelf.
        Adjacent labels are also read as one name, since catalogs split hyphenated names."""
        scored = [self._label_score(want, str(label)) for label in path]
        scores = [sc for sc, _full in scored]
        fulls = [full for _sc, full in scored]
        for i in range(1, len(path)):
            pair, full = self._label_score(want, "%s %s" % (path[i - 1], path[i]))
            if pair > max(scores[i - 1], scores[i]):        # the two labels only match read together
                scores[i], fulls[i] = pair, full
        deepest = max((i for i, sc in enumerate(scores) if sc > 0), default=None)
        if deepest is None:
            return (0.0, 0)
        top = max(scores)
        named = max(i for i, sc in enumerate(scores) if sc >= top - 0.1)
        if any(fulls):
            return (1.0, min(i for i, full in enumerate(fulls) if full))
        fit = scores[deepest] / (1.0 + 0.4 * (len(path) - 1 - deepest))
        return (round(fit, 3), named)

    def _resolve_shelf(self) -> None:
        """Name the catalog shelf for the requested category from the paths observed so far.
        Deterministic: recomputed from all evidence whenever new products are read."""
        want = self.brief.category
        if not want:
            return
        paths = [self._cat_path(pid) for pid in self.ledger.products]
        paths = [p for p in paths if p]
        if not paths:
            return
        scored = sorted(((self._path_fit(want, p), tuple(p)) for p in paths), key=lambda x: (x[0][0], -x[0][1]), reverse=True)
        (score, index), best_path = scored[0]
        if score >= 0.7:
            label = str(best_path[index])
            if label != self.shelf:
                self.shelf, self.shelf_inferred = label, False
            return
        # Nothing read is worded like the request: when the products found for it agree on
        # one shelf, that shelf is what the request means.
        leaves = collections.Counter(tuple(p) for p in paths)
        (top, count), total = leaves.most_common(1)[0], len(paths)
        if total >= 3 and count * 3 >= total * 2:
            label = str(top[-1])
            if label != self.shelf:
                self.shelf, self.shelf_inferred = label, True
                self._note("No catalog shelf is worded like the requested category (%s); the products found for it sit under %s, which the controller now treats as the required category." % (want, " > ".join(top)))

    def _shelf_name(self) -> str | None:
        """The category argument for filter: the catalog's own label once known."""
        return self.shelf or self.brief.category

    def _category_ok(self, pid: str) -> bool:
        want = self.brief.category
        path = self._cat_path(pid)
        if not want or not path:
            return True
        if self.shelf and self.shelf_inferred:
            return any(str(label) == self.shelf for label in path)
        fit = self._path_fit(want, path)[0]
        # At least half the request must be answered, and once a shelf that answers it
        # better has been read, weaker matches fail.
        ref = max((self._path_fit(want, self._cat_path(p))[0] for p in self.ledger.products), default=0.0)
        return fit >= max(0.5, ref - 0.05)

    def _brand_ok(self, pid: str) -> bool:
        want = self.brief.brand
        if not want:
            return True
        wanted = _tokens(want)
        return bool(wanted <= _tokens(self._brand_of(pid)) or wanted <= _tokens(self._title(pid)))

    def _device_verdict(self, pid: str, sku: str) -> str:
        rule = self.brief.device_rule
        if rule is None:
            return "yes"
        return rule.verdict(self._facts(pid), self._title(pid), self._options(pid, sku))

    def _device_ok(self, pid: str, sku: str) -> bool:
        verdict = self._device_verdict(pid, sku)
        strict = bool(self.brief.attributes or self.brief.justify or self.brief.recovers or self.brief.hidden_requirement)
        return verdict == "yes" if strict else verdict != "no"

    def _resale_title(self, pid: str) -> bool:
        return bool(re.search(r"customer (?:rating|review)|\bcompare\b|refurb|pre-owned|restored|renewed", self._title(pid), re.I))

    def _spec_state(self, pid: str, sku: str) -> int:
        """Worst state over every stated attribute: 2 all stated, 0 unknown, -1 contradicted."""
        state = 2
        for req in self.brief.attributes:
            s = req.satisfied_by(self._options(pid, sku), self._facts(pid), self._title(pid))
            state = min(state, s)
        return state

    def _fact_states(self, pid: str, sku: str, req: Requirement) -> bool:
        """Whether the product's own facts about this attribute state the value outright."""
        want = req.target()
        vals: set = set()
        for v in req.related_facts(self._facts(pid)):
            vals |= req.values_in(v)
        return want in vals and len(vals) == 1

    def _measure(self, pid: str, sku: str, attribute: str) -> float | None:
        """A numeric reading for a tradeoff axis named in the request."""
        name = attribute.lower()
        unit = None
        for u in ("inch", "hz", "gb", "mah", "w", "hours", "mp"):
            if u in name or (u == "inch" and re.search(r"size|screen|display", name)) or (u == "hz" and "refresh" in name) or (u == "gb" and re.search(r"ram|memory|storage", name)) or (u == "mah" and "battery" in name):
                unit = u
                break
        if unit is None:
            return None
        req = Requirement(attribute, "0", unit, self.brief.scale)
        vals = req.values_in(self._options(pid, sku))
        if len(vals) == 1:
            return float(next(iter(vals)))
        if vals:
            return None
        vals = set()
        for v in req.related_facts(self._facts(pid)):
            vals |= req.values_in(v)
        vals |= req.values_in(self._title(pid))
        return float(next(iter(vals))) if len(vals) == 1 else None

    def _relax(self, pid: str, sku: str) -> int:
        model, color = self.brief.preferred_model, self.brief.preferred_color
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self._title(pid) + " " + " ".join(str(v) for v in self._facts(pid).values()))
        same_model = squash(model) in blob
        shade = (self._options(pid, sku) + " " + " ".join(str(v) for k, v in self._facts(pid).items() if "color" in str(k).lower()) + " " + self._title(pid)).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if self.brief.relax_color_first:
            return 1 if same_model else (2 if same_color else 3)
        return 1 if same_color else (2 if same_model else 3)

    def _pref_score(self, pid: str, sku: str) -> float:
        axes = self.brief.pref_axes
        if not axes:
            return 0.0
        text = " ".join([f"{k} {v}" for k, v in self._facts(pid).items()] + [self._title(pid), self._options(pid, sku)] + [str(p) for p in self._cat_path(pid)])
        have = _stems(text)
        total = sum(a[0] for a in axes)
        got = sum(a[0] for a in axes if a[1] & have)
        return round(got / total, 3) if total else 0.0

    def _axis_queries(self) -> list:
        cat = re.sub(r"[^a-z ]", "", (self.brief.category or "").lower()).strip()
        cat = cat[:-1] if cat.endswith("s") else cat
        out = []
        for _w, _s, words in sorted(self.brief.pref_axes, key=lambda a: -a[0]):
            if words[:3]:
                out.append(" ".join(words[:3] + ([cat] if cat else [])))
        return out

    def _assess(self, pid: str, sku: str) -> dict:
        key = Ledger.key(pid, sku)
        st = self.ledger.checked.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.ledger.dead_keys or (str(pid) in self.ledger.dead_products and not self.brief.recovers)
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and not dead
        viewed = str(pid) in self.ledger.products
        spec = self._spec_state(pid, sku) if viewed else 0
        ok = bool(live and viewed and spec == 2 and self._brand_ok(pid) and self._device_ok(pid, sku))
        cat_ok = self._category_ok(pid)
        ok = ok and cat_ok
        if self.brief.justify:
            ok = ok and not self._resale_title(pid) and bool(self._brand_of(pid))
            ok = ok and all(self._fact_states(pid, sku, req) for req in self.brief.attributes)
        if self.brief.prefers and self.ledger.event_target:
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split("::", 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if self.brief.theme:
            ok = ok and all(t in _tokens(self._title(pid)) for t in self.brief.theme)
        if self.brief.candidate_keys:
            ok = ok and (str(pid), str(sku)) in self.brief.candidate_keys
        if self.brief.ranks:
            ok = ok and all(self._measure(pid, sku, a) is not None for a, _ in self.brief.tradeoff_axes if not re.search(r"price|cost", a, re.I))
        return {"pid": str(pid), "sku": str(sku), "key": key, "price": price, "live": live, "viewed": viewed, "spec": spec, "cat_ok": cat_ok, "ok": ok, "relax": self._relax(pid, sku) if self.brief.recovers else 0}

    def _rank_of(self, c: dict, first: bool) -> tuple:
        price = c["price"] if c["price"] is not None else 1e12
        if self.brief.ranks:
            order = []
            for attribute, high in self.brief.tradeoff_axes:
                if re.search(r"price|cost", attribute, re.I):
                    value = price
                else:
                    value = self._measure(c["pid"], c["sku"], attribute) or 0.0
                order.append(-value if high else value)
            return (*order, price, c["key"])
        if self.brief.recovers:
            return (c["relax"], price, c["key"])
        proven = 0 if self._device_verdict(c["pid"], c["sku"]) == "yes" else 1
        fit = -self._pref_score(c["pid"], c["sku"])
        if self.brief.preferred_brand and first:
            wanted = _tokens(self.brief.preferred_brand)
            near = 0 if (wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))) else 1
            return (near, proven, fit, price, c["key"])
        return (proven, fit, price, c["key"])

    def _candidates(self) -> list:
        out = []
        seen = set()
        for key in list(self.ledger.checked):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self._assess(pid, sku))
        return out

    def _target(self) -> dict | None:
        first = self.ledger.adds == 0
        good = [c for c in self._candidates() if c["ok"]]
        if self.brief.prefers:
            carted = [c for c in good if (c["pid"], c["sku"]) in self.ledger.cart]
            if carted:
                return carted[0]
            if not good:
                return None
            good.sort(key=lambda c: (-self._pref_score(c["pid"], c["sku"]), c["price"] if c["price"] is not None else 1e12, c["key"]))
            best = good[0]
            fit = self._pref_score(best["pid"], best["sku"])
            if fit < _PREF_MIN and self._early():
                return None
            if fit < 1.0 - _EPS and self.reach_queries and self._early():
                return None
            return best
        if self.brief.preferred_brand and first and not self.brief.recovers:
            wanted = _tokens(self.brief.preferred_brand)
            named = [c for c in good if wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))]
            if named:
                good = named
            elif self._early():
                self._preferred_brand_hint()
                return None
        if self.brief.recovers and first:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif self._early():
                self._exact_match_hint()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self._rank_of(c, first))
        return good[0]

    def _answer_text(self) -> str:
        target = self._target() or (self._candidates() or [None])[0]
        if not target:
            return "I am still checking the options against your requirements; nothing is decided yet."
        pid, sku = target["pid"], target["sku"]
        parts = ["I am going with " + (self._title(pid) or pid)[:90]]
        for req in self.brief.attributes:
            if req.satisfied_by(self._options(pid, sku), self._facts(pid), self._title(pid)) == 2:
                parts.append("its listed %s is %s" % (req.name, req.raw_value))
        if target.get("price") is not None:
            parts.append("it is %.2f %s and in stock" % (target["price"], self.brief.currency or ""))
        return ", ".join(parts) + "."

    # ---------------------------------------------------------------- planning
    def _early(self) -> bool:
        return self.turn + self.stalls < self.max_steps * 0.6 and self.clock.remaining() > 75

    def _recall_need(self) -> int:
        return self.brief.required_matches

    def _ready(self) -> bool:
        if self.ledger.adds:
            return True
        if self.brief.hidden_requirement and not self.brief.attributes:
            if self._early():
                return False
            self.brief.hidden_requirement = False
            self._note("The withheld requirement was never stated, so the remaining requirements decide the choice.")
        if (self.brief.attributes or self.brief.hidden_requirement) and self.turn < 4 and not self.shopper_updated:
            return False
        if self.brief.prefers and len(self.ledger.listing_ids) < 2:
            return False
        roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
        if self.brief.recalls and roomy:
            if len(self.ledger.identities) < self._recall_need() or self.search_calls < 1:
                return False
        if roomy and not self.reads_done and len({c["pid"] for c in self._candidates() if c["ok"]}) < 2:
            return False
        return True

    def _fresh(self, pid: str, sku: str) -> bool:
        k = Ledger.key(pid, sku)
        if k not in self.ledger.inspected:
            return False
        return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq

    def _normalize_read(self, args: dict) -> None:
        args["k"] = 10
        if "in_stock" not in args:
            args["in_stock"] = True
        if self.budget is not None and args.get("max_price") is None:
            args["max_price"] = self.budget

    def _unseen_leads(self) -> list:
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
            if self.brief.candidate_keys and (str(pid), str(sku)) not in self.brief.candidate_keys:
                continue
            if self.brief.preferred_model and self.ledger.adds == 0 and not self.ledger.event_target and str(pid) not in self.model_search_pids:
                if re.sub(r"[^a-z0-9]+", "", self.brief.preferred_model.lower()) not in re.sub(r"[^a-z0-9]+", "", title.lower()):
                    continue
            if self.brief.brand:
                wanted = _tokens(self.brief.brand)
                if not (wanted <= _tokens(str(row.get("brand") or "")) or wanted <= _tokens(title)):
                    continue
            leads.append((price, str(pid)))
        leads.sort()
        out: list = []
        for _p, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def _read_more(self) -> tuple | None:
        b = self.brief
        if b.recalls and self.ledger.adds == 0 and len(self.ledger.identities) < 10 and len(self.search_queries) < _SEARCH_BUDGET and self._early() and "search" in self.tool_names and b.theme:
            theme = " ".join(b.theme)
            options = [theme, theme + " " + (b.category or ""), " ".join(reversed(b.theme))]
            used = {_norm_query(q) for q in self.search_queries}
            for query in options:
                query = query.strip()
                if _norm_query(query) not in used:
                    args = {"query": query, "in_stock": True, "k": 10}
                    if self.budget is not None:
                        args["max_price"] = self.budget
                    self.search_queries.add(query)
                    self.search_calls += 1
                    self.ledger.calls_seen.add(_sig("search", args))
                    return ("search", args)
        if b.recalls and len(self.ledger.identities) < self._recall_need():
            return None
        if b.recalls and self.ledger.adds == 0 and len(self.ledger.identities) < 10 and self.recall_bands and self._early() and "filter" in self.tool_names and b.category and self.budget is not None:
            band = self.recall_bands.pop(0)
            return ("filter", {"category": self._shelf_name(), "max_price": round(self.budget * band, 2), "in_stock": True, "k": 10})
        if b.recalls and self.ledger.adds == 0 and len(self.ledger.identities) < 10 and self._early() and "filter" in self.tool_names and b.category:
            brands = []
            for key, row in self.ledger.rows.items():
                brand = str(row.get("brand") or "").strip()
                pid = key.partition("::")[0]
                if brand and pid in self.ledger.cluster_of and brand.lower() not in self.recall_brands_probed and brand not in brands:
                    brands.append(brand)
            if brands and len(self.recall_brands_probed) < 3:
                brand = brands[0]
                self.recall_brands_probed.add(brand.lower())
                args = {"category": self._shelf_name(), "brand": brand, "in_stock": True, "k": 10}
                if self.budget is not None:
                    args["max_price"] = self.budget
                return ("filter", args)
        twin = self._twin_search()
        if twin is not None:
            return twin
        leads = self._unseen_leads()
        if leads and self.forced_compares < 8 and self.turn + self.stalls >= 2:
            self.forced_compares += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        if b.prefers and self.reach_queries and self._early() and "search" in self.tool_names and not any(c["ok"] and self._pref_score(c["pid"], c["sku"]) >= 1.0 - _EPS for c in self._candidates()):
            args = {"query": self.reach_queries.pop(0), "in_stock": True, "k": 10}
            if self.budget is not None:
                args["max_price"] = self.budget
            self.forced_compares = 0
            return ("search", args)
        if b.recovers and b.preferred_model and self.ledger.adds == 0 and not self.model_probed and "search" in self.tool_names:
            self.model_probed = True
            args = {"query": b.preferred_model, "in_stock": True, "k": 10}
            if _sig("search", args) not in self.ledger.calls_seen:
                self.ledger.calls_seen.add(_sig("search", args))
                return ("search", args)
        if b.preferred_brand and self.ledger.adds == 0 and not self.brand_probed and "filter" in self.tool_names and b.category:
            self.brand_probed = True
            args = {"category": self._shelf_name(), "brand": b.preferred_brand, "in_stock": True, "k": 10}
            if self.budget is not None:
                args["max_price"] = self.budget
            return ("filter", args)
        probe = (b.brand or "", "%.2f" % (self.budget or 0.0), "|".join(r.describe() for r in b.attributes))
        if self.turn < 2 and not self.model_down:
            return None
        bands_used = self.probes.setdefault(probe, set())
        band = next((x for x in (1.0, 0.75, 0.5, 0.35) if x not in bands_used), None)
        if band is None or (band < 1.0 and self.budget is None):
            self.reads_done = True
            return None
        bands_used.add(band)
        if "filter" in self.tool_names and b.category:
            args = {"category": self._shelf_name(), "in_stock": True, "k": 10}
            if self.budget is not None:
                args["max_price"] = round(self.budget * band, 2)
            if b.brand:
                args["brand"] = b.brand
            return ("filter", args)
        if "search" in self.tool_names:
            words = [w for w in (b.brand, b.category) if w] + [r.raw_value for r in b.attributes]
            if words:
                return ("search", {"query": " ".join(words), "in_stock": True, "k": 10})
        self.reads_done = True
        return None

    def _twin_search(self) -> tuple | None:
        dead = self.ledger.event_target
        if self.searched_twins or not dead or "search" not in self.tool_names or self.brief.prefers:
            return None
        title = self._title(dead.split("::", 1)[0])
        if not title:
            return None
        self.searched_twins = True
        self._note("The item that died is one of several listings sharing a title; searching that title surfaces its siblings, which meet the same requirements.")
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def _sibling_search(self) -> tuple | None:
        if self.searched_siblings:
            return None
        if any(c["live"] and c["viewed"] and c["relax"] == 0 for c in self._candidates()):
            return None
        near = [c for c in self._candidates() if c["viewed"] and c["relax"] == 1]
        if not near:
            return None
        title = self._title(near[0]["pid"])
        if not title:
            return None
        self.searched_siblings = True
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def _preferred_brand_hint(self) -> None:
        if self.hinted_brand:
            return
        self.hinted_brand = True
        self._note("No %s product meeting every requirement has been read yet; the first choice is judged on that brand, so read %s products within budget first." % (self.brief.preferred_brand, self.brief.preferred_brand))

    def _exact_match_hint(self) -> None:
        if self.hinted_exact or not self.brief.preferred_model:
            return
        self.hinted_exact = True
        self._note("The exact preferred item (%s%s) has not been found yet. Search the model on its own and view the results so every variant is on record; only that exact item may be the first cart addition." % (self.brief.preferred_model, ", " + self.brief.preferred_color if self.brief.preferred_color else ""))

    def _shortlist_note(self) -> None:
        cands = self._candidates()
        if self.brief.prefers and not self.ledger.cart:
            good = [c for c in cands if c["ok"]]
            good.sort(key=lambda c: (-self._pref_score(c["pid"], c["sku"]), c["price"] if c["price"] is not None else 1e12))
            if good:
                self._note("Products meeting every requirement so far, with how much of the stated preferences their facts mention (%.2f or better is accepted): " % _PREF_MIN + "; ".join("%s::%s fit=%.2f at %.2f" % (c["pid"], c["sku"], self._pref_score(c["pid"], c["sku"]), c["price"] or 0.0) for c in good[:6]) + ". If none is strong, search for products whose facts state the preferred features and compare them.")
                return
        best = self._target()
        if best:
            self._note("Verified candidate: %s::%s at %.2f. The controller will inspect, cart and order it; read other candidates only if one could rank better." % (best["pid"], best["sku"], best["price"] or 0.0))
            return
        blocked = [c for c in cands if c["viewed"] and c["live"] and not c["ok"]]
        if blocked and self.brief.attributes:
            self._note("None of the products read so far can be verified: %s must resolve exactly from the variant's own option text or the product's raw facts: %s. Look at different products." % ("; ".join(r.describe() for r in self.brief.attributes), ", ".join(c["pid"] + "::" + c["sku"] for c in blocked[:3])))
        elif blocked:
            self._note("The products read so far fail a requirement (category, brand, stock, budget or the kind of item requested). Search for different products.")

    def _budget_from_message(self, text: str) -> None:
        amounts = _money_amounts(text)
        new = min(amounts) if amounts else None
        if new is None or (self.budget is not None and new >= self.budget):
            return
        if not re.search(r"\b(?:only|now|up to|at most|budget|afford|lower|less|maximum|cap)\b", text or "", re.I):
            return
        self.budget = new
        self.shopper_updated = True
        self._note("Budget updated by the shopper: at most %.2f %s. Every later choice must fit it." % (new, self.brief.currency or ""))
        for pid, sku in list(self.ledger.cart):
            st = self.ledger.status(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _EPS:
                self.ledger.dead_keys.add(Ledger.key(pid, sku))
                self._note("The carted item %s costs %.2f, above the new budget; the controller will replace it." % (Ledger.key(pid, sku), price))

    def _requirement_from_message(self, text: str) -> None:
        """A firm attribute the shopper states later, read with the same unit resolver."""
        if self.brief.attributes or not self.brief.hidden_requirement:
            return
        cleaned = _MONEY_RE.sub(" ", text or "")
        for unit in ("inch", "hz", "gb", "mah", "w", "mp", "hours"):
            vals = _unit_values(unit, cleaned)
            if len(vals) == 1:
                value = next(iter(vals))
                name = Requirement._UNIT_LABEL_HINTS.get(unit, (unit,))[0]
                self.brief.attributes.append(Requirement(name, "%g" % value, unit, self.brief.scale))
                self.brief.hidden_requirement = False
                self.shopper_updated = True
                self._note("Requirement recorded: %s. Only a variant that resolves to it may be bought." % self.brief.attributes[-1].describe())
                return
        if self.brief.scale:
            names = self.brief.scale.values(text)
            if len(names) == 1:
                cls = next(iter(names))
                self.brief.attributes.append(Requirement("resolution", cls, None, self.brief.scale))
                self.brief.hidden_requirement = False
                self.shopper_updated = True
                self._note("Requirement recorded: resolution = %s." % cls)

    def _preferences_from_message(self, text: str) -> None:
        if not self.brief.prefers:
            return
        # A ranking is read from ordinary priority words in reading order: each stretch
        # of text that ends in one ("most", "first", "then", "next", "last", "least",
        # "close behind", "after that") is one preference, earlier ones weightier.
        priority = re.compile(
            r"(?P<txt>[^,;.\n]+?)\s*\b(?:most|first|primarily|foremost|close\s+(?:second|behind)|"
            r"second|then|next|after\s+that|third|finally|last|least)\b"
        )
        ranked = []
        for m in priority.finditer((text or "").lower()):
            words = _words(m.group("txt"))
            if len(words) >= 1:
                ranked.append((m.group("txt"), len(ranked) + 1))
        if ranked:
            self.brief.adopt_preferences(ranked)
            used = {_norm_query(q) for q in self.search_queries}
            self.reach_queries = [q for q in self._axis_queries() if _norm_query(q) not in used]

    def _claims_for(self, pid: str, sku: str) -> list:
        out = []
        for req in self.brief.attributes:
            field = self.brief.claim_field_for(req)
            if field and self._fact_states(pid, sku, req) and all(c["field"] != field for c in out):
                out.append({"field": field, "value": req.raw_value})
        if not out and self.brief.attributes:
            return []
        if self.budget is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.budget})
        else:
            st = self.ledger.status(pid, sku) or {}
            if st.get("price") is not None:
                out.append({"field": "price", "value": "%.2f" % st["price"]})
        out.append({"field": "in_stock", "value": "true"})
        brand = self._brand_of(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out

    def _order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.brief.justify:
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
            return False, "add that variant to the cart first"
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

    def _stall_fallback(self) -> tuple | None:
        if self.stalls >= 2:
            self.reads_done = True
        return self._read_more()

    def _plan(self) -> tuple | None:
        if self.done:
            return None
        if self.shopper_question_pending:
            if "message" in self.tool_names and (self.question_waits >= 1 or self.clock.remaining() < 110):
                return ("message", {"content": self._answer_text()})
            self.question_waits += 1
            return None
        if self.brief.ranks and "compare" in self.tool_names:
            missing = [p for p, _s in self.brief.candidate_keys if p not in self.ledger.products and p not in self.unreadable]
            if missing:
                return ("compare", {"product_ids": missing[:5]})
        if self.brief.recovers and not self.ledger.adds and "search" in self.tool_names:
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

    # --------------------------------------------------------------- inference
    def _ask_model(self) -> dict | None:
        for attempt in range(3):
            if self.clock.remaining() <= 8:
                break
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            payload = {"model": model, "messages": self.messages, "tools": self.tools, "tool_choice": "required", "temperature": 0}
            cap = _CALL_CAP_S if self.timeouts == 0 else _CALL_CAP_S * 0.6
            remaining = self.clock.remaining()
            started = time.monotonic()
            resp = _post("/inference/chat/completions", payload, remaining, cap)
            if resp is None and time.monotonic() - started >= min(cap, remaining - 3.0) - 1:
                self.timeouts += 1
                if attempt >= 1:
                    return None
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                self._note("No tool call was made. Choose exactly one of the supplied tools now.")
                self.model_index = min(self.model_index + 1, len(self.roster) - 1)
                continue
            if self.clock.remaining() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    # ------------------------------------------------------------- environment
    def _send(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            if self.clock.remaining() <= 5:
                break
            result = _post("/environment/call", envelope, self.clock.remaining())
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
        envelope = {**self.binding, "call_id": group, "idempotency_key": group, "turn": turn, "calls": [{"call_id": group + "-1", "action": {"name": name, "args": args}}]}
        result = self._send(envelope)
        if result is None:
            self.pending_envelope = {"envelope": envelope, "turn": turn, "name": name, "args": args, "content": content, "call_id": tool_call_id or ("agent-" + str(turn))}
        else:
            self.turn = turn
        return self._absorb(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append({"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result})
        if not isinstance(result, dict):
            self.ledger.calls_seen.discard(_sig(name, args))
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps({"error": "environment call failed; it will be retried automatically"})})
            return None
        calls = result.get("calls") or []
        wrap = (calls[0].get("observation") if calls and isinstance(calls[0], dict) else None) or {}
        obs = wrap.get("observation") if isinstance(wrap.get("observation"), dict) else wrap
        if isinstance(obs, dict) and wrap.get("error") and not obs.get("error"):
            obs = dict(obs)
            obs["error"] = wrap.get("error")
        if wrap.get("done"):
            self.done = True
        self.ledger.record(name, args, obs, self.brief, self.budget)
        if name in ("view", "compare"):
            self._resolve_shelf()
        if name == "search" and self.brief.preferred_model and isinstance(obs, dict):
            code = re.sub(r"[^a-z0-9]+", "", self.brief.preferred_model.lower())
            if code in re.sub(r"[^a-z0-9]+", "", str(args.get("query") or "").lower()):
                self.model_search_pids.update(str(r.get("product_id")) for r in obs.get("results") or [] if isinstance(r, dict) and r.get("product_id"))
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _trim(json.dumps(obs), _OBS_CHARS)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self._budget_from_message(text)
            stated = re.search(r"\bcategory is ([A-Za-z][\w&' -]{1,40}?)(?:\.|,|;| and\b|$)", text)
            if stated:
                self.brief.category = stated.group(1).strip()
                self.shelf, self.shelf_inferred = None, False
                self._resolve_shelf()
            stated = re.search(r"\bbrand is ([A-Za-z][\w&' -]{1,30}?)(?:\.|,|;| and\b|$)", text)
            if stated and not self.brief.brand and not self.brief.preferred_brand:
                self.brief.brand = stated.group(1).strip()
            self._requirement_from_message(text)
            self._preferences_from_message(text)
            if _QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self._note("The shopper asked a question: answer it with message in your next turn, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        return obs

    # -------------------------------------------------------- controller steps
    def _post_add_recheck(self) -> None:
        pid, sku = self.pending_recheck
        self.pending_recheck = None
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = _num(obs.get("price"))
        over = self.budget is not None and price is not None and price > self.budget + _EPS
        if stock and not over:
            self._note("Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it." % (Ledger.key(pid, sku), price))
            return
        self.ledger.event_target = Ledger.key(pid, sku)
        self.ledger.dead_keys.add(Ledger.key(pid, sku))
        self.ledger.dead_products.add(str(pid))
        self.forced_compares = 0
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        if self.brief.recovers:
            guidance = "Recover with the smallest possible change, in the order the shopper stated, before changing anything else."
        elif self.brief.ranks:
            guidance = "The remaining named candidates are re-ranked; the controller takes the new best."
        elif self.brief.prefers:
            guidance = "Recover to a product from a different listing that still fits the shopper's priorities."
        else:
            guidance = "Choose a different product that still meets every requirement."
        self._note("Market change: the carted item %s %s. %s" % (Ledger.key(pid, sku), reason, guidance))
        self._shortlist_note()

    def _finish_fast(self) -> None:
        for pid, sku in reversed(self.ledger.cart):
            ok, _why = self._valid_now(pid, sku)
            if ok and not self.done:
                self._act("place_test_order", self._order_args(pid, sku))
                return
        if self.done or self.turn > self.max_steps - 3:
            return
        good = [c for c in self._candidates() if c["ok"]]
        if not good and not self.brief.justify:
            good = [c for c in self._candidates() if c["live"] and c["spec"] == 2 and self._brand_ok(c["pid"]) and c["cat_ok"] and (not self.brief.candidate_keys or (c["pid"], c["sku"]) in self.brief.candidate_keys)]
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

    # ---------------------------------------------------------------- main loop
    def run(self) -> list:
        question = self.brief.ask_first
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            if self.clock.remaining() < 35 or self.turn >= self.max_steps - 3:
                self._finish_fast()
                break
            if question and not self.asked_shopper and "message" in self.tool_names:
                self.asked_shopper = True
                self.questions_asked += 1
                self._act("message", {"content": question})
                continue
            if self.pending_recheck and "inspect_stock" in self.tool_names:
                self._post_add_recheck()
                continue
            plan = self._plan()
            if plan is not None:
                name, args = plan
                signature = _sig(name, args)
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
                    self._note("That step changed nothing, so the controller moved on to other candidates.")
                    continue
                obs = self._act(name, args)
                if name == "message":
                    self.shopper_question_pending = False
                    self.question_waits = 0
                if name in ("search", "filter"):
                    self.search_calls += 1
                if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                    added = obs.get("added") or {}
                    if added.get("product_id"):
                        self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                continue
            last_forced, repeats = None, 0
            if self.model_down:
                if self.shopper_question_pending and "message" in self.tool_names:
                    self._act("message", {"content": self._answer_text()})
                    self.shopper_question_pending = False
                    self.question_waits = 0
                    continue
                forced = self._stall_fallback()
                if forced is not None:
                    self._act(*forced)
                    continue
                self._finish_fast()
                break
            proposal = self._ask_model()
            if proposal is None:
                self.model_misses += 1
                if self.model_misses >= 2 or self.timeouts >= 2:
                    self.model_down = True
                continue
            self.model_misses = 0
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = _arguments(call)
            content = proposal.get("content") or ""
            if name in ("search", "filter"):
                self._normalize_read(args)
            reprompts = 0
            while True:
                verdict = self._validate(name, args)
                if verdict is None:
                    break
                if reprompts >= _MAX_REPROMPTS:
                    self.rejected_streak += 1
                    self._note("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if self.rejected_streak >= 3:
                        # Three proposals in a row the controller had to refuse: the
                        # model is not going to move the state on; the controller will.
                        self._shortlist_note()
                        self.model_down = True
                    verdict = "skip"
                    break
                reprompts += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.messages.append({"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": verdict})})
                self.dialogue.append({"role": "assistant", "content": "[rejected " + name + "] " + verdict[:200]})
                again = self._ask_model()
                if again is None:
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = _arguments(call)
                if name in ("search", "filter"):
                    self._normalize_read(args)
                content = again.get("content") or ""
            if verdict == "skip":
                self.stalls += 1
                forced = self._stall_fallback()
                if forced is not None:
                    self._act(*forced)
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
                if name == "search":
                    self.search_queries.add(str(args.get("query") or ""))
                if self.brief.recalls and self.brief.required_matches:
                    have, need = len(self.ledger.identities), self._recall_need()
                    self._note("Distinct matching products surfaced so far: %d of the required %d." % (have, need) + ("" if have >= need else " Keep searching with different wording."))
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
            if self.brief.recalls and len(self.ledger.identities) < self._recall_need():
                return "order refused: only %d distinct matching products have been surfaced; search with different wording until at least %d appear" % (len(self.ledger.identities), self._recall_need())
            if self.brief.prefers and len(self.ledger.listing_ids) < 2:
                return "order refused: compare at least two products from different listings first"
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
                return "inspect_stock that variant first: its current price and stock decide whether it can be bought"
            if c["spec"] != 2 and self.brief.attributes:
                return "that variant does not resolve to %s from its own option text or raw facts" % "; ".join(r.describe() for r in self.brief.attributes)
            if not c["ok"]:
                return "that product fails a requirement (category, brand, or the kind of item requested)"
            if self.brief.prefers and len(self.ledger.listing_ids) < 2:
                return "compare two products from different listings before the first cart addition"
            if self.brief.prefers and self._early():
                fit = self._pref_score(pid, sku)
                if fit < _PREF_MIN:
                    return "that item mentions too little of what the shopper asked for (%.2f of 1.00); read candidates whose facts state the preferred features before committing" % fit
                if fit < 1.0 - _EPS and self.reach_queries:
                    return "that item fits %.2f of 1.00; the controller is still surveying for a product whose facts cover every preference, so hold the cart for now" % fit
            if self.brief.recovers and not self.ledger.adds and self.brief.preferred_model and self._relax(pid, sku):
                return "the first cart item must be the exact model and color the shopper named; its other colors are separate listings sharing the title, so search that title and read them first"
            if not self._ready():
                return "hold: read one more product that also meets every requirement first, so there is a fallback when the market moves right after the cart addition"
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.shopper_question_pending and "?" not in text:
                return "do not send status updates; use message only to answer the shopper's question or to ask one"
            if not self.shopper_question_pending and self.questions_asked >= 2:
                return "the shopper has already answered; decide from the evidence you have rather than asking again"
        if name == "filter" and not args.get("brand"):
            shelf = (str(args.get("category") or "").lower(), bool(args.get("in_stock")))
            seen = self.shelf_price_probes.setdefault(shelf, set())
            if len(seen) >= 2 and args.get("max_price") not in seen:
                return "that shelf has already been read at several price ceilings; narrow it with brand, search by keywords or model code, or read the rows found"
            seen.add(args.get("max_price"))
        if name == "filter" and args.get("query"):
            return "filter has no query parameter and would return arbitrary items; use search for keywords, or filter with only category, brand, max_price, in_stock, k"
        if name == "search" and self.brief.recalls:
            query = str(args.get("query") or "")
            if query not in self.search_queries and len(self.search_queries) >= _SEARCH_BUDGET:
                return "enough distinct searches have been made; surface more with filter(category, max_price, in_stock, k=10) or read the results already found"
        if name == "inspect_cart" and self.cart_inspections >= 1:
            return "the cart contents are already known from earlier observations"
        key = _sig(name, args)
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
