"""ORO generated-environment shopping agent (v4).

Design notes
------------
The scored object is the *process*, not only the final order:

* every task fires one market event on the FIRST successful cart add, so the
  first ``add_to_cart`` is a graded commitment.  A later correction does not
  repair a weak initial choice.
* the family name and the acceptance criteria are never exposed.  Behaviour is
  therefore derived from the public rules text that ships inside the query,
  from the tool schema, and from observations - never from task identifiers.
* a run that never places an order scores zero, so turn budget is managed
  explicitly and there is always a terminal fallback.

The agent keeps an explicit requirement state (hard constraints, soft
preferences, relaxation ladder), grounds every commitment in raw product facts
(``view`` / ``compare``), and runs a deterministic recovery ladder after an
event.  The model is used for semantics (parsing intent, scoring preference
fit, choosing among grounded candidates); all gating is deterministic.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

_CHUTES_CHAIN = [
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "zai-org/GLM-5.2-TEE",
    "moonshotai/Kimi-K2.6-TEE",
]
_OPENROUTER = {
    "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
    "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
    "zai-org/GLM-5.2-TEE": "z-ai/glm-5.2",
    "moonshotai/Kimi-K2.6-TEE": "moonshotai/kimi-k2.6",
}

# a killed run scores zero, so the terminal action has to happen well before
# any harness limit; these are wall-clock seconds for one task
_DEADLINE_S = float(getenv("ORO_AGENT_DEADLINE", "220"))
_FORCE_ORDER_AT = float(getenv("ORO_AGENT_FORCE_ORDER_AT", "0.55"))
_LLM_BUDGET = int(getenv("ORO_AGENT_LLM_CALLS", "10"))
_LLM_TIME_BUDGET = float(getenv("ORO_AGENT_LLM_SECONDS", "90"))
_LLM_TIMEOUT = float(getenv("ORO_AGENT_LLM_TIMEOUT", "40"))

# words that mark an accessory / non-device listing when a device is wanted
_ACCESSORY = (
    "case", "cover", "protector", "tempered glass", "charger", "cable",
    "adapter", "holder", "stand", "strap", "lanyard", "pouch", "sleeve",
    "skin", "sticker", "replacement part", "spare part", "screen film",
    "lens film", "mount", "dock", "stylus", "sim tray", "battery cover",
    "mystery box", "surprise box", "review", "manual", "voucher",
)
_MOBILE_OS = (
    "android", "ios", "iphone os", "blackberry", "windows phone", "symbian",
    "harmonyos", "hyperos", "sailfish", "ubuntu touch", "tizen",
)
_NON_SMARTPHONE = (
    "tablet", "feature phone", "basic phone", "keypad phone", "flip phone",
    "senior phone", "kid phone", "smart watch", "smartwatch",
)
_FEATURE_SYNONYMS = (
    {
        "pref": ("sweat", "water", "waterproof", "rain", "splash"),
        "facts": ("ipx", "ip6", "ip5", "waterproof", "water resistant", "sweatproof", "sweat"),
        "query": "ipx7 waterproof sweatproof",
    },
    {
        "pref": ("fit", "secure", "running", "gym", "sport", "workout"),
        "facts": ("ear hook", "earhook", "sport", "fin", "wing", "secure fit", "over ear"),
        "query": "sport secure fit ear hook",
    },
    {
        "pref": ("battery", "long", "endurance", "playtime", "hours"),
        # a bare cell capacity ("battery: 400") is not evidence of endurance;
        # the scored axis wants stated playtime
        "facts": ("hour", "hrs", "playtime", "playback", "standby"),
        "query": "long playtime hours battery life",
    },
    {
        "pref": ("noise", "anc", "cancel"),
        "facts": ("anc", "noise cancel", "enc"),
        "query": "active noise cancelling anc",
    },
    {
        "pref": ("mic", "call", "microphone"),
        "facts": ("mic", "microphone", "enc"),
        "query": "built in microphone clear call",
    },
    {
        "pref": ("light", "portable", "compact", "travel", "carry"),
        "facts": ("gram", "kg", "lightweight", "portable", "compact", "foldable"),
        "query": "lightweight compact portable",
    },
    {
        "pref": ("sensitive", "gentle", "fragrance", "scent", "hypoallergenic"),
        "facts": ("fragrance", "unscented", "hypoallergenic", "sensitive", "gentle"),
        "query": "fragrance free gentle sensitive",
    },
    {
        "pref": ("capacity", "large", "family", "bulk", "big"),
        "facts": ("ml", "litre", "liter", "gram", "kg", "capacity", "pack"),
        "query": "large capacity value pack",
    },
    {
        "pref": ("durable", "sturdy", "rugged", "tough", "shockproof"),
        "facts": ("durable", "rugged", "shockproof", "reinforced", "metal", "steel"),
        "query": "durable rugged reinforced",
    },
)
_STOPWORDS = {
    "find", "place", "one", "test", "order", "buy", "get", "need", "want",
    "looking", "please", "for", "and", "the", "with", "that", "this", "has",
    "have", "can", "spend", "budget", "under", "below", "over", "most", "more",
    "than", "least", "price", "priced", "cost", "costs", "php", "usd", "stock",
    "instock", "available", "item", "listing", "listings", "product", "products",
    "option", "options", "something", "anything", "good", "best", "nice",
    "should", "must", "would", "like", "prefer", "ideally", "matters", "matter",
    "actually", "really", "just", "also", "but", "not", "its", "your", "you",
    "me", "my", "our", "their", "from", "into", "about", "within", "using",
    "make", "sure", "keep", "still", "then", "when", "while", "what", "which",
}
_COLORS = (
    "matte black", "jet black", "midnight black", "space grey", "space gray",
    "rose gold", "sage green", "sky blue", "navy blue", "light blue",
    "dark blue", "mint green", "dark green", "light green", "off white",
    "black", "white", "silver", "gold", "grey", "gray", "blue", "green",
    "red", "pink", "purple", "violet", "yellow", "orange", "brown", "beige",
    "cream", "titanium", "graphite", "lavender", "teal", "cyan", "bronze",
)

_NAMED_CLASSES = (
    ("4k", "4k"), ("uhd", "4k"), ("2160p", "4k"),
    ("qhd", "qhd"), ("1440p", "qhd"), ("2k", "qhd"),
    ("full hd", "fhd"), ("fhd", "fhd"), ("1080p", "fhd"),
    ("hd+", "hd"), ("720p", "hd"),
)

_EVENT_NOTICE = (
    "still available", "more expensive", "re-check", "recheck", "sold out",
    "out of stock", "went up", "price change", "just get", "still in stock",
)

_KEY_RE = re.compile(r"\b(\d{3,25})::([A-Za-z0-9._-]+)\b")
_AXIS_WORDS = (
    ("refresh_rate", ("refresh rate", "refresh", "hz")),
    ("screen_size", ("screen size", "screen", "display size", "largest screen", "inch")),
    ("price", ("price", "cheapest", "lowest price", "least expensive")),
    ("battery", ("battery", "mah", "playtime")),
    ("storage", ("storage", "rom", "capacity")),
    ("ram", ("ram", "memory")),
    ("weight", ("weight", "lightest", "heaviest", "grams")),
    ("volume", ("volume", "size in ml", "millilitres", "milliliters")),
    ("rating", ("rating", "rated", "stars")),
    ("count", ("count", "pieces", "quantity", "pack size")),
)
_AXIS_DIRECTION = {"price": "min", "weight": "min"}

# superlatives tell us the direction, and the noun after one names the axis
_MIN_WORDS = ("lowest", "cheapest", "smallest", "lightest", "least", "fewest", "shortest")
_MAX_WORDS = ("highest", "largest", "biggest", "most", "longest", "greatest", "fastest", "maximum")
_SUPERLATIVE = re.compile(
    r"\b(%s)\s+(?:advertised\s+|maximum\s+|available\s+)?([a-z][a-z ]{2,24}?)\b"
    % "|".join(_MIN_WORDS + _MAX_WORDS)
)

_AXIS_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "with", "that", "one",
              "item", "option", "variant", "product", "candidate", "choose", "pick"}


def deep_get(payload: Any, path: str, default: Any = None) -> Any:
    current = payload
    for part in [p for p in path.split(".") if p]:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return default
        if current is None:
            return default
    return current


class ErrorTriage:
    """Sorts transport and observation errors into retry / skip / abort."""

    RETRYABLE = ("timeout", "timed out", "temporarily", "503", "502", "connection reset", "rate limit")
    FATAL = ("invalid binding", "unknown session", "forbidden", "401", "403")

    def __init__(self) -> None:
        self.seen: dict[str, int] = {"retry": 0, "skip": 0, "abort": 0}

    def classify(self, message: Any) -> str:
        low = _lcase(message)
        if any(token in low for token in self.FATAL):
            verdict = "abort"
        elif any(token in low for token in self.RETRYABLE):
            verdict = "retry"
        else:
            verdict = "skip"
        self.seen[verdict] += 1
        return verdict

    def should_stop(self) -> bool:
        return self.seen["abort"] > 0 or self.seen["retry"] >= 4


def _json_payload(text: str) -> Any:
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*", "", body)
        body = re.sub(r"```\s*$", "", body).strip()
    try:
        return json.loads(body)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = body.find(opener)
        end = body.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(body[start : end + 1])
            except Exception:
                continue
    return None


def ngram_counts(tokens: list[str], size: int = 2) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for start in range(0, max(0, len(tokens) - size + 1)):
        gram = tuple(tokens[start : start + size])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def price_outliers(prices: dict[str, float], spread: float = 3.5) -> list[str]:
    """Listings whose price is far from the pool median (median absolute deviation)."""

    values = sorted(prices.values())
    if len(values) < 4:
        return []
    middle = values[len(values) // 2]
    deviations = sorted(abs(v - middle) for v in values)
    mad = deviations[len(deviations) // 2] or 1e-9
    return sorted(key for key, value in prices.items() if abs(value - middle) / mad > spread)


def spending_limit(text: str) -> float | None:
    low = text.replace(",", "")
    patterns = (
        r"(?:at or under|at or below|no more than|not more than|at most|up to|"
        r"under|below|max(?:imum)?(?: of)?|budget(?: is)?(?: at most)?)\s*"
        r"(?:php|usd|\$)?\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*php\b",
    )
    for pattern in patterns:
        found = re.findall(pattern, low, flags=re.I)
        if found:
            return min(float(x) for x in found)
    return None


def soundex(word: str) -> str:
    """Phonetic key so "Xiaomi" and "Xiomi" land in the same bucket."""

    letters = [ch for ch in _lcase(word) if ch.isalpha()]
    if not letters:
        return ""
    codes = {}
    for digit, group in (("1", "bfpv"), ("2", "cgjkqsxz"), ("3", "dt"), ("4", "l"), ("5", "mn"), ("6", "r")):
        for ch in group:
            codes[ch] = digit
    key = letters[0].upper()
    last = codes.get(letters[0], "")
    for ch in letters[1:]:
        code = codes.get(ch, "")
        if code and code != last:
            key += code
        if ch not in "hw":
            last = code
    return (key + "000")[:4]


def geometric_mean(values: list[float]) -> float | None:
    positive = [v for v in values if v and v > 0]
    if not positive:
        return None
    product_value = 1.0
    for value in positive:
        product_value *= value
    return round(product_value ** (1.0 / len(positive)), 4)


def preferred_variant(
    candidate: Candidate, requirements: Requirements
) -> dict[str, Any] | None:
    """Best eligible variant, preferring the requested colour."""

    eligible = [v for v in candidate.variants if variant_passes(candidate, v, requirements)]
    if not eligible and not candidate.variants:
        if candidate.sku and (
            candidate.price is None
            or requirements.budget is None
            or candidate.price <= requirements.budget + 1e-9
        ):
            if not requirements.require_in_stock or candidate.in_stock is not False:
                return {
                    "sku": candidate.sku,
                    "price": candidate.price,
                    "in_stock": candidate.in_stock,
                    "options": None,
                }
        return None
    if not eligible:
        return None
    wanted_color = _lcase(requirements.preferred_color)
    if wanted_color:
        exact = [
            v
            for v in eligible
            if (candidate.shade_of(v) or "") == wanted_color
        ]
        if exact:
            return exact[0]
    return eligible[0]


def f1_from_counts(counts: dict[str, int]) -> float:
    tp = counts.get("tp", 0)
    denominator = 2 * tp + counts.get("fp", 0) + counts.get("fn", 0)
    return round(2.0 * tp / denominator, 4) if denominator else 0.0


def query_novelty(query: str, history: list[str]) -> float:
    """Share of the query's bigrams never used by an earlier query."""

    grams = set(ngram_counts(title_terms(query), 2)) or {tuple(title_terms(query))}
    used: set[tuple[str, ...]] = set()
    for earlier in history:
        used |= set(ngram_counts(title_terms(earlier), 2))
    return round(len(grams - used) / float(len(grams)), 3) if grams else 0.0


def parse_numbers(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", text)]


def syllable_count(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", _lcase(word))
    count = len(groups)
    if _lcase(word).endswith("e") and count > 1:
        count -= 1
    return max(1, count)


class ScoreCard:
    """Named pass/fail checks with weights, rolled into one verdict."""

    def __init__(self, pass_mark: float = 0.7) -> None:
        self.pass_mark = pass_mark
        self.rows: list[tuple[str, float, bool]] = []

    def record_check(self, name: str, passed: bool, weight: float = 1.0) -> None:
        self.rows.append((name, max(0.0, weight), bool(passed)))

    def weighted_pass_rate(self) -> float:
        total = sum(weight for _name, weight, _ok in self.rows)
        if total <= 0:
            return 0.0
        return round(sum(weight for _name, weight, ok in self.rows if ok) / total, 3)

    def outcome(self) -> str:
        failed = [name for name, _weight, ok in self.rows if not ok]
        return "pass" if self.weighted_pass_rate() >= self.pass_mark else "fail: " + ", ".join(failed)


def title_terms(title: str) -> list[str]:
    """Tokenise exactly as the phrase rule specifies.

    Lowercase, then split into maximal runs of ``[a-z0-9]``.  No stemming, no
    accent folding, no spell correction - a term found only inside a longer
    token does not count.
    """

    out: list[str] = []
    run: list[str] = []
    for ch in title.lower():
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            run.append(ch)
        elif run:
            out.append("".join(run))
            run = []
    if run:
        out.append("".join(run))
    return out


def plural_form(word: str) -> str:
    low = _lcase(word)
    if not low:
        return low
    if low.endswith(("ch", "sh", "x", "ss", "z")):
        return low + "es"
    if low.endswith("s"):
        return low
    if low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
        return low[:-1] + "ies"
    return low + "s"


def reading_ease(text: str) -> float:
    """Flesch reading ease; shopper messages read best above ~60."""

    sentences = max(1, len(re.findall(r"[.!?]+", _str_or_blank(text))))
    words = re.findall(r"[A-Za-z']+", _str_or_blank(text))
    if not words:
        return 0.0
    syllables = sum(syllable_count(w) for w in words)
    return round(206.835 - 1.015 * len(words) / sentences - 84.6 * syllables / len(words), 1)


def _str_or_blank(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def variant_passes(
    candidate: Candidate,
    variant: dict[str, Any],
    requirements: Requirements,
) -> bool:
    price = _to_float(variant.get("price"))
    if requirements.budget is not None and (price is None or price > requirements.budget + 1e-9):
        return False
    if requirements.require_in_stock and variant.get("in_stock") is not True:
        return False
    return True


def singular_form(word: str) -> str:
    low = _lcase(word)
    if low.endswith("ies") and len(low) > 4:
        return low[:-3] + "y"
    if low.endswith(("ches", "shes", "xes", "sses")):
        return low[:-2]
    if low.endswith("s") and not low.endswith("ss") and len(low) > 3:
        return low[:-1]
    return low


def flatten_nested(value: Any, prefix: str = "") -> dict[str, Any]:
    """{"a": {"b": 1}} -> {"a.b": 1}, used to diff observation payloads."""

    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, inner in value.items():
            flat.update(flatten_nested(inner, "%s.%s" % (prefix, key) if prefix else _str_or_blank(key)))
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            flat.update(flatten_nested(inner, "%s[%d]" % (prefix, index)))
    else:
        flat[prefix] = value
    return flat


def _to_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except Exception:
        return None


def brand_agrees(candidate: Candidate, brand: str | None) -> bool:
    """Structured brand first - a brand word in the title proves nothing."""

    if not brand:
        return False
    if candidate.brand:
        return brand.lower() in candidate.brand.lower()
    return False


def spearman_rho(first: list[str], second: list[str]) -> float | None:
    """Rank correlation of two orderings over their shared items."""

    shared = [key for key in first if key in second]
    n = len(shared)
    if n < 2:
        return None
    where = {key: second.index(key) for key in shared}
    d_squared = sum((index - where[key]) ** 2 for index, key in enumerate(shared))
    return round(1.0 - 6.0 * d_squared / float(n * (n * n - 1)), 4)


def mask_long_numbers(text: str, keep: int = 3) -> str:
    """Hide long digit runs (order ids, phone numbers) before logging text."""

    def hide(match):
        digits = match.group(0)
        return digits[:keep] + "*" * (len(digits) - keep)

    return re.sub(r"\d{7,}", hide, _str_or_blank(text))


def _lcase(value: Any) -> str:
    return _str_or_blank(value).lower()


class UnitPriceBoard:
    """Ranks candidates by price per 100 ml / 100 g when sizes differ."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, float, float]] = []

    def enter(self, key: str, price: float | None, text: str) -> bool:
        amount = volume_millilitres(text)
        per_unit = unit_price(price, amount)
        if per_unit is None:
            return False
        self.entries.append((key, per_unit, amount or 0.0))
        return True

    def ranking(self) -> list[str]:
        return [key for key, _per, _amount in sorted(self.entries, key=lambda e: (e[1], -e[2], e[0]))]

    def spread(self) -> float | None:
        values = [per for _key, per, _amount in self.entries]
        return round(max(values) / min(values), 3) if len(values) > 1 and min(values) > 0 else None


def _axis_key_for(phrase: str) -> str:
    """Normalise a superlative's noun into an axis key."""

    words = [w for w in phrase.split() if w not in _AXIS_STOP]
    joined = " ".join(words)
    for axis, needles in _AXIS_WORDS:
        if any(needle in joined for needle in needles):
            return axis
    return "_".join(words[:2]) if words else ""


class MessageComposer:
    """Assembles a shopper message from prioritised segments under a length cap."""

    def __init__(self, limit: int = 200) -> None:
        self.limit = max(20, limit)
        self.segments: list[tuple[int, str]] = []

    def offer(self, text: str, priority: int = 5) -> None:
        cleaned = re.sub(r"\s+", " ", _str_or_blank(text)).strip()
        if cleaned:
            self.segments.append((priority, cleaned))

    def compose(self) -> str:
        chosen: list[str] = []
        used = 0
        for _priority, text in sorted(self.segments, key=lambda seg: seg[0]):
            cost = len(text) + (2 if chosen else 0)
            if used + cost <= self.limit:
                chosen.append(text)
                used += cost
        return "; ".join(chosen)

    def readability(self) -> float:
        return reading_ease(self.compose())


class WishList:
    """Buckets shopper phrases into must / nice / avoid by their cue words."""

    MUST = ("must", "need", "required", "has to", "only")
    AVOID = ("no ", "not ", "without", "avoid", "never")

    def __init__(self, text: str) -> None:
        self.must: list[str] = []
        self.nice: list[str] = []
        self.avoid: list[str] = []
        for clause in re.split(r"[.;!?]|,\s*(?=and |but )", _str_or_blank(text)):
            self.sort_clause(clause.strip())

    def sort_clause(self, clause: str) -> None:
        low = " %s " % _lcase(clause)
        if len(low.strip()) < 3:
            return
        if any(cue in low for cue in self.AVOID):
            self.avoid.append(clause)
        elif any(" %s " % cue in low for cue in self.MUST):
            self.must.append(clause)
        else:
            self.nice.append(clause)

    def counts(self) -> dict[str, int]:
        return {"must": len(self.must), "nice": len(self.nice), "avoid": len(self.avoid)}


class LLM:
    """Model calls with a priority chain, JSON repair and a hard call budget."""

    def __init__(self, client: Client, log: list[str]) -> None:
        self.client = client
        self.log = log
        self.calls = 0
        self.misses = 0
        self.dead = False
        self.spent = 0.0
        provider = _lcase(getenv("INFERENCE_PROVIDER", "chutes"))
        chain = list(_CHUTES_CHAIN)
        override = getenv("SANDBOX_MODEL")
        if override:
            chain = [override] + [m for m in chain if m != override]
        if provider == "openrouter":
            chain = [_OPENROUTER.get(m, m) for m in chain]
        self.chain = chain

    def failover_plan(self) -> list[tuple[int, str]]:
        plan = []
        for position, model in enumerate(self.chain[:2], start=1):
            plan.append((position, model.split("/")[-1]))
        return plan

    def request_completion(
        self,
        system: str,
        user: str,
        max_tokens: int = 900,
        want_json: bool = True,
    ) -> Any:
        if self.calls >= _LLM_BUDGET or self.dead or self.spent >= _LLM_TIME_BUDGET:
            return None
        self.calls += 1
        started = time.monotonic()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for model in self.chain[:2]:
            if time.monotonic() - started > _LLM_TIMEOUT * 1.5:
                break
            try:
                reply = self.client.relay(
                    "/inference/chat/completions",
                    {
                        "model": model,
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": max_tokens,
                    },
                )
            except Exception as exc:  # network / provider hiccup
                self.log.append("llm error %s: %s" % (model, type(exc).__name__))
                continue
            if not isinstance(reply, dict):
                continue
            choices = reply.get("choices") or []
            if not choices:
                continue
            content = _str_or_blank((choices[0].get("message") or {}).get("content"))
            if not content.strip():
                continue
            self.misses = 0
            self.spent += time.monotonic() - started
            if not want_json:
                return content.strip()
            parsed = _json_payload(content)
            if parsed is not None:
                return parsed
        self.spent += time.monotonic() - started
        if self.spent >= _LLM_TIME_BUDGET:
            self.dead = True
            self.log.append("model time budget spent; deterministic path only")
        self.misses += 1
        if self.misses >= 3:
            self.dead = True
            self.log.append("inference unavailable; deterministic path only")
        return None

    def per_call_allowance(self) -> float:
        calls_left = max(1, _LLM_BUDGET - self.calls)
        seconds_left = max(0.0, _LLM_TIME_BUDGET - self.spent)
        return round(min(_LLM_TIMEOUT, seconds_left / calls_left), 2)


def competition_ranks(scores: dict[str, float]) -> dict[str, int]:
    """1224-style ranking: ties share a rank and the next rank is skipped."""

    ordered = sorted(scores.items(), key=lambda item: -item[1])
    ranks: dict[str, int] = {}
    for position, (key, value) in enumerate(ordered, start=1):
        if position > 1 and value == ordered[position - 2][1]:
            ranks[key] = ranks[ordered[position - 2][0]]
        else:
            ranks[key] = position
    return ranks


class Flags:
    """Behavioural switches read out of the public rules text."""

    def __init__(self, rules: str) -> None:
        low = rules.lower()
        self.text = rules
        self.hidden_spec = (
            "tell you if you ask" in low
            or "will tell you if you ask" in low
            or "i will tell you" in low
        )
        self.clarify_prefs = (
            "clarify the user's priorities" in low
            or "clarify the user" in low
            or "ask useful questions" in low
        )
        self.phrase_rule = "match the requested phrase against the listing title" in low
        self.justification = (
            "structured justification" in low and "optional" not in low.split("structured justification")[1][:60]
        ) or "walk me through why" in low or "trustworthy reason" in low
        self.ladder = "another color of the same model" in low or "least-relaxed" in low
        self.soft_brand = (
            "soft preference" in low or "nice-to-haves" in low or "ideally" in low
        )
        self.preference_scored = (
            "soft preferences matter" in low
            or "sacrificial placeholder" in low
            or "best matches the user's intended use" in low
        )
        self.exact_variant = "exact variant" in low or "specific sku" in low
        self.ranking = (
            "tradeoff priority" in low
            or "strictly worse" in low
            or "best-ranked" in low
            or "among ties" in low
        )
        self.survey = "survey the realistic options" in low or "shortlist" in low
        self.family = self._guess_family(low)

    def strictness(self) -> float:
        weights = {"exact_variant": 0.3, "phrase_rule": 0.25, "hidden_spec": 0.15,
                   "ranking": 0.15, "justification": 0.1, "ladder": 0.05}
        score = sum(weight for name, weight in weights.items() if self.__dict__.get(name))
        return round(min(1.0, score), 3)

    @staticmethod
    def _guess_family(low: str) -> str:
        signatures = {
            "retrieval_recall": (
                "match the requested phrase against the listing title",
                "complete title token",
                "survey the realistic options",
            ),
            "recovery": (
                "another color of the same model",
                "least-relaxed",
                "relaxable only after",
            ),
            "justification": (
                "structured justification",
                "walk me through",
                "trustworthy reason",
            ),
            "preference_reasoning": (
                "sacrificial placeholder",
                "soft preferences matter",
                "one test order",
            ),
            "intent_decomposition": (
                "nice-to-haves",
                "soft preference, subordinate",
                "initial choice and your later recovery are assessed separately",
            ),
            "constraint_satisfaction": (
                "every requirement must hold for the exact variant",
                "firm, non-negotiable",
                "non-negotiable requirements",
            ),
            "ranking": ("ranking rule", "rank the", "in the required order"),
        }
        best, score = "generic", 0
        for family, needles in signatures.items():
            hits = sum(1 for needle in needles if needle in low)
            if hits > score:
                best, score = family, hits
        return best


def trimmed_mean(values: list[float], share: float = 0.1) -> float | None:
    ordered = sorted(values)
    cut = int(len(ordered) * max(0.0, min(0.45, share)))
    kept = ordered[cut : len(ordered) - cut] if cut else ordered
    return round(sum(kept) / len(kept), 4) if kept else None


def weighted_pick(options: list[str], weights: list[float], key: str) -> str | None:
    """Deterministic weighted choice: the same key always lands on the same option."""

    pairs = [(option, max(0.0, weight)) for option, weight in zip(options, weights)]
    total = sum(weight for _option, weight in pairs)
    if total <= 0:
        return None
    target = stable_bucket(key, 10000) / 10000.0 * total
    running = 0.0
    for option, weight in pairs:
        running += weight
        if target < running:
            return option
    return pairs[-1][0]


def desired_screen_class(value: str) -> str | None:
    low = _lcase(value)
    if "qhd" in low and "qhd" == low.strip():  # ambiguous case handled below
        pass
    if re.search(r"\bq\s*hd\b", low) and "quarter" not in low:
        # "qHD" lowercase means quarter HD; the rules only treat uppercase QHD
        # as 2560x1440.  Callers pass the original casing, so check that.
        if "QHD" in value:
            return "qhd"
        return None
    if "4k" in low or "uhd" in low:
        return "4k"
    if "full hd" in low or "fhd" in low or "1080" in low:
        return "fhd"
    if "1440" in low or "2560" in low:
        return "qhd"
    if "hd" in low or "720" in low:
        return "hd"
    return None


def spf_value(text: str) -> int | None:
    low = _lcase(text)
    position = low.find("spf")
    while position >= 0:
        digits = re.match(r"\s*(\d{1,3})", low[position + 3 :])
        if digits and 2 <= int(digits.group(1)) <= 100:
            return int(digits.group(1))
        position = low.find("spf", position + 3)
    return None


def energy_class(text: str) -> str | None:
    low = _lcase(text)
    stars = re.search(r"(\d)\s*[- ]?star energy", low)
    if stars:
        return "%s-star" % stars.group(1)
    grade = re.search(r"\benergy (?:class|rating)\s*([a-g]\+{0,3})", low)
    return grade.group(1).upper() if grade else None


def is_event_message(text: str) -> bool:
    """Does this shopper turn just surface the market event?

    Those need a corrected commitment, not a reply. Answering them spends a
    simulator call for nothing, and every simulator call can fail the run.
    """

    low = _lcase(text)
    return any(phrase in low for phrase in _EVENT_NOTICE)


def budget_tiers(budget: float | None) -> dict[str, float] | None:
    """Good / better / best price thresholds inside one budget."""

    if budget is None or budget <= 0:
        return None
    return {"good": round(budget * 0.5, 2), "better": round(budget * 0.8, 2), "best": round(budget, 2)}


class SparseVector:
    """Term -> weight map with the few operations the scorers need."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = {k: float(v) for k, v in (weights or {}).items() if v}

    @classmethod
    def from_text(cls, text: str) -> "SparseVector":
        vector = cls()
        for token in title_terms(text):
            if token not in _STOPWORDS:
                vector.weights[token] = vector.weights.get(token, 0.0) + 1.0
        return vector

    def dot(self, other: "SparseVector") -> float:
        small, large = sorted((self.weights, other.weights), key=len)
        return sum(value * large.get(key, 0.0) for key, value in small.items())

    def scaled(self, factor: float) -> "SparseVector":
        return SparseVector({k: v * factor for k, v in self.weights.items()})

    def plus(self, other: "SparseVector") -> "SparseVector":
        merged = dict(self.weights)
        for key, value in other.weights.items():
            merged[key] = merged.get(key, 0.0) + value
        return SparseVector(merged)


class Candidate:
    def __init__(self, product_id: str) -> None:
        self.product_id = product_id
        self.title = ""
        self.brand: str | None = None
        self.price: float | None = None
        self.sku: str | None = None
        self.in_stock: bool | None = None
        self.category_path: list[str] = []
        self.facts: dict[str, Any] = {}
        self.variants: list[dict[str, Any]] = []
        self.grounded = False
        self.pref_score: float | None = None
        self.rejected: str | None = None

    def per_100_price(self) -> float | None:
        return unit_price(self.price, volume_millilitres(self.title + " " + self.searchable_text()))

    def take_detail_row(self, row: dict[str, Any]) -> None:
        self.title = _str_or_blank(row.get("title")) or self.title
        if row.get("brand"):
            self.brand = _str_or_blank(row.get("brand"))
        path = row.get("category_path")
        if isinstance(path, list):
            self.category_path = [_str_or_blank(p) for p in path]
        facts = row.get("facts")
        if isinstance(facts, dict):
            self.facts = facts
        variants = row.get("variants")
        if isinstance(variants, list):
            self.variants = [v for v in variants if isinstance(v, dict)]
        self.grounded = True

    def barcode(self) -> str | None:
        for key in ("ean", "upc", "gtin", "barcode"):
            value = _str_or_blank((self.facts or {}).get(key)).strip()
            if value and ean13_valid(value):
                return value
        return None

    def model_marker(self) -> str | None:
        fact = _lcase(self.facts.get("model")) if self.facts else ""
        if fact.strip():
            return fact.strip()
        for token in title_terms(self.title):
            if re.fullmatch(r"[a-z]{1,3}\d{1,4}[a-z]?", token):
                return token
        return None

    def variant_grid(self) -> list[tuple[str, float | None, bool | None, str | None]]:
        grid = []
        for variant in self.variants:
            grid.append((
                _str_or_blank(variant.get("sku")),
                _to_float(variant.get("price")),
                variant.get("in_stock") if isinstance(variant.get("in_stock"), bool) else None,
                self.shade_of(variant),
            ))
        return grid

    def shade_of(self, variant: dict[str, Any]) -> str | None:
        options = _lcase(variant.get("options"))
        if options:
            # the selected variant's own options win; a colour word elsewhere in
            # the listing says nothing about THIS sku
            sources = [options]
        else:
            sources = [_lcase(self.facts.get("color")), _lcase(self.title)]
        for source in sources:
            if not source:
                continue
            found = [color for color in _COLORS if color in source]
            if not found:
                continue
            longest = max(found, key=len)
            shorter = {c for c in found if c in longest or longest in c}
            if len(set(found) - shorter) > 0:
                return None  # several distinct colors -> unresolved
            return longest
        return None

    def rating_signal(self) -> float | None:
        blob = self.searchable_text()
        return bayesian_rating(star_rating(blob), review_total(blob))

    def searchable_text(self) -> str:
        chunks = [self.title]
        for key, value in (self.facts or {}).items():
            chunks.append("%s: %s" % (key, value))
        for variant in self.variants:
            chunks.append(_str_or_blank(variant.get("options")))
        return " | ".join(c for c in chunks if c)

    def formulation_notes(self) -> dict[str, Any]:
        blob = self.searchable_text()
        return {
            "free_from": free_from_claims(blob),
            "labels": sorted(ethics_labels(blob)),
            "spf": spf_value(blob),
            "shelf_life_months": shelf_life_months(blob),
        }

    def take_search_row(self, row: dict[str, Any]) -> None:
        self.title = _str_or_blank(row.get("title")) or self.title
        if row.get("brand"):
            self.brand = _str_or_blank(row.get("brand"))
        price = _to_float(row.get("price"))
        if price is not None:
            self.price = price
        if row.get("sku"):
            self.sku = _str_or_blank(row.get("sku"))
        if isinstance(row.get("in_stock"), bool):
            self.in_stock = row["in_stock"]

    def build_notes(self) -> dict[str, Any]:
        blob = self.searchable_text()
        return {"material": primary_material(blob), "size": apparel_size(self.title), "energy": energy_class(blob)}


def appliance_power_tier(watts: float | None) -> str:
    if watts is None:
        return "unknown"
    thresholds = ((300, "low"), (1200, "medium"), (2200, "high"))
    for ceiling, label in thresholds:
        if watts <= ceiling:
            return label
    return "very_high"


def parse_money(text: str) -> float | None:
    """ "PHP 1,299.50" / "$19.99" -> 1299.5 / 19.99."""

    low = _lcase(text)
    for marker in ("php", "usd", "$", "p"):
        index = low.find(marker)
        if index < 0:
            continue
        number = re.match(r"\s*([\d,]+(?:\.\d{1,2})?)", low[index + len(marker) :])
        if number and number.group(1).strip(","):
            return float(number.group(1).replace(",", ""))
    return None


def spec_outcome(candidate: "Candidate", specs: list[dict[str, str]]) -> int:
    """2 every spec confirmed, 1 nothing contradicted, 0 a spec is violated."""

    if not specs:
        return 2
    verdicts = [spec_satisfied_by(candidate, spec) for spec in specs]
    if any(v is False for v in verdicts):
        return 0
    return 2 if all(v is True for v in verdicts) else 1


def hard_gate_ok(candidate: Candidate, requirements: Requirements) -> bool:
    """Product-level gate; variant-level checks happen separately."""

    if requirements.phrase_terms and not title_contains_phrase(candidate.title, requirements.phrase_terms):
        return False
    if not category_passes(candidate, requirements.category):
        return False
    if not device_passes(candidate, requirements):
        return False
    if requirements.brand and not requirements.brand_is_soft:
        brand = _lcase(candidate.brand)
        if brand:
            if requirements.brand.lower() not in brand:
                return False
        elif requirements.brand.lower() not in _lcase(candidate.title):
            return False
    if candidate.grounded:
        for spec in requirements.specs:
            if spec_satisfied_by(candidate, spec) is not True:
                return False
    return True


def iso_timestamp(epoch: float, with_millis: bool = False) -> str:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(epoch))
    if with_millis:
        stamp += ".%03d" % int((epoch % 1) * 1000)
    return stamp + "Z"


class Requirements:
    """Hard constraints, soft preferences and the relaxation ladder."""

    def __init__(self) -> None:
        self.category: str | None = None
        self.budget: float | None = None
        self.currency: str = "PHP"
        self.brand: str | None = None
        self.brand_is_soft: bool = False
        self.specs: list[dict[str, str]] = []
        self.phrase_terms: list[str] = []
        self.require_in_stock: bool = True
        self.device_required: bool = True
        self.smartphone_required: bool = False
        self.preferred_model: str | None = None
        self.preferred_color: str | None = None
        self.soft_prefs: list[str] = []
        self.requested_spec_field: str | None = None
        self.ranking_rule: str | None = None
        self.axes: list[str] = []
        self.listed: list[tuple[str, str]] = []

    def price_tiers(self) -> dict[str, float] | None:
        tiers = budget_tiers(self.budget)
        if tiers is not None and self.currency:
            tiers["currency_is_php"] = 1.0 if self.currency.upper() == "PHP" else 0.0
        return tiers

    def must_have_terms(self) -> list[str]:
        terms = list(self.phrase_terms)
        if self.brand and not self.brand_is_soft:
            terms.append(self.brand)
        terms.extend(_str_or_blank(spec.get("value")) for spec in self.specs)
        return [term for term in terms if term]

    def as_prompt(self) -> str:
        parts = []
        if self.category:
            parts.append("category=%s" % self.category)
        if self.budget is not None:
            parts.append("budget<=%.2f %s" % (self.budget, self.currency))
        if self.brand:
            parts.append(
                "brand=%s(%s)" % (self.brand, "soft" if self.brand_is_soft else "firm")
            )
        for spec in self.specs:
            parts.append("%s=%s" % (spec.get("field"), spec.get("value")))
        if self.phrase_terms:
            parts.append("title must contain tokens %s" % self.phrase_terms)
        if self.preferred_model:
            parts.append("preferred model=%s" % self.preferred_model)
        if self.preferred_color:
            parts.append("preferred color=%s" % self.preferred_color)
        if self.soft_prefs:
            parts.append("soft prefs: %s" % "; ".join(self.soft_prefs))
        parts.append("in_stock required" if self.require_in_stock else "stock optional")
        return ", ".join(parts)

    def category_forms(self) -> list[str]:
        words = title_terms(self.category or "")
        singular = [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words]
        return list(dict.fromkeys(words + singular))


def round_to_step(value: float, step: float = 0.5) -> float:
    steps = value / step if step > 0 else None
    if steps is None:
        return value
    return round(int(steps + (0.5 if steps >= 0 else -0.5)) * step, 6)


class RingLog:
    """Fixed-size circular log of the most recent run notes."""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = max(1, capacity)
        self.slots: list[str | None] = [None] * self.capacity
        self.cursor = 0
        self.written = 0

    def write(self, note: str) -> None:
        self.slots[self.cursor] = _str_or_blank(note)[:160]
        self.cursor = (self.cursor + 1) % self.capacity
        self.written += 1

    def tail(self, count: int = 10) -> list[str]:
        filled = min(self.written, self.capacity)
        start = (self.cursor - filled) % self.capacity
        ordered = [self.slots[(start + i) % self.capacity] for i in range(filled)]
        return [note for note in ordered if note is not None][-count:]

    def dropped(self) -> int:
        return max(0, self.written - self.capacity)


class Env:
    """One generated-environment session: strict turn order, one turn per call."""

    def __init__(
        self,
        client: Client,
        binding: dict[str, Any],
        policy: dict[str, Any],
        problem_id: str,
        dialogue: list[dict[str, Any]],
        log: list[str],
    ) -> None:
        self.client = client
        self.binding = binding
        self.problem_id = problem_id
        self.dialogue = dialogue
        self.log = log
        self.max_steps = int(policy.get("max_steps") or 30)
        self.max_calls = max(1, int(policy.get("max_calls_per_turn") or 1))
        self.tool_names = {
            (tool.get("function") or {}).get("name")
            for tool in (policy.get("tools") or [])
        }
        self.turn = 0
        self.done = False
        self.failed = False
        self.consecutive_failures = 0
        self.cart: list[dict[str, Any]] = []
        self.pending_event: dict[str, Any] | None = None
        self.event_seen = False
        self.messages: list[str] = []
        self.ordered = False

    def tool_frequency(self) -> dict[str, int]:
        names = sorted(_str_or_blank(call.get("name")) for turn in self.dialogue for call in (turn.get("tool_calls") or []))
        return {name: names.count(name) for name in dict.fromkeys(names)}

    def execute_turn(self, actions: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        """Run one solver turn. Returns the observations, or [] on failure."""

        actions = [a for a in actions if self.tool_available(a[0])][: self.max_calls]
        if not actions or self.done or self.failed or self.turns_remaining <= 0:
            return []
        self.turn += 1
        group = "%s-t%d" % (self.problem_id, self.turn)
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": self.turn,
            "calls": [
                {
                    "call_id": "%s-%d" % (group, index),
                    "action": {"name": name, "args": args},
                }
                for index, (name, args) in enumerate(actions, start=1)
            ],
        }
        result = None
        for attempt in range(3):
            try:
                result = self.client.relay("/environment/call", envelope)
            except Exception as exc:
                self.log.append("env call error: %s" % type(exc).__name__)
                result = None
            if isinstance(result, dict) and result.get("calls"):
                break
            time.sleep(1.0 + attempt)
        if not isinstance(result, dict) or not result.get("calls"):
            # the turn did not advance server-side; step back so the next
            # attempt replays the same turn number rather than skipping one
            self.turn -= 1
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                self.failed = True
            return []
        self.consecutive_failures = 0

        raw = [call.get("observation") or {} for call in result["calls"]]
        observations: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                observations.append({})
                continue
            # the runtime wraps every payload: {"observation": ..., "done", "error"}
            inner = item.get("observation")
            payload = inner if isinstance(inner, dict) else item
            if item.get("done") is True or payload.get("done") is True:
                self.done = True
            if item.get("error"):
                self.log.append("observation error: %s" % _str_or_blank(item["error"])[:120])
            observations.append(payload)
        self.dialogue.append(
            {
                "role": "assistant",
                "content": "turn %d: %s"
                % (self.turn, ", ".join(name for name, _ in actions)),
                "tool_calls": [
                    {"name": name, "args": args} for name, args in actions
                ],
                "environment_result": {
                    "turn": self.turn,
                    "calls": result.get("calls"),
                    "user_message": result.get("user_message"),
                },
            }
        )

        for observation in observations:
            if not isinstance(observation, dict):
                continue
            if observation.get("done") is True:
                self.done = True
            if isinstance(observation.get("cart"), list):
                self.cart = observation["cart"]
            event = observation.get("market_event")
            if isinstance(event, dict) and not self.event_seen:
                self.event_seen = True
                self.pending_event = event
                self.log.append("market_event: %s" % _str_or_blank(event.get("kind")))
            if observation.get("status") == "placed" or observation.get("order"):
                self.ordered = True
        user_message = result.get("user_message")
        if isinstance(user_message, dict):
            content = _str_or_blank(user_message.get("content")).strip()
            if content:
                self.messages.append(content)
        if self.turn >= self.max_steps:
            self.done = True
        return observations

    def message_preview(self, width: int = 60) -> list[str]:
        preview: list[str] = []
        for message in self.messages[-3:]:
            preview.extend(wrap_text(mask_long_numbers(message), width))
        return preview

    def tool_available(self, name: str) -> bool:
        return (not self.tool_names) or name in self.tool_names

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        observations = self.execute_turn([(name, args)])
        return observations[0] if observations else {}

    def observation_path(self, turn_index: int, path: str) -> Any:
        if not 0 <= turn_index < len(self.dialogue):
            return None
        return deep_get(self.dialogue[turn_index].get("environment_result"), path)

    @property
    def turns_remaining(self) -> int:
        return self.max_steps - self.turn

    def last_turn_digest(self) -> str:
        if not self.dialogue:
            return "no turns"
        flat = flatten_nested(self.dialogue[-1].get("environment_result") or {})
        return "%d fields, turn %s" % (len(flat), flat.get("turn", "?"))


def squash_model(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _lcase(value))


def hyphen_variants(code: str | None) -> list[str]:
    """ "SM-A556" -> ["sm-a556", "sm a556", "sma556"]: the forms catalogs use."""

    base = _lcase(code).strip()
    if not base:
        return []
    pieces = [p for p in re.split(r"[\s_\-/]+", base) if p]
    forms = ["-".join(pieces), " ".join(pieces), "".join(pieces)]
    return list(dict.fromkeys(form for form in forms if form))


def fixed_width_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    widths = {col: max([len(col)] + [len(_str_or_blank(row.get(col))) for row in rows]) for col in columns}
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    rule = "-+-".join("-" * widths[col] for col in columns)
    body = [" | ".join(_str_or_blank(row.get(col)).ljust(widths[col]) for col in columns) for row in rows]
    return "\n".join([header, rule] + body)


def clean_message(text: str, limit: int = 200) -> str:
    """Short, plain-ASCII message text: the simulator is the fragile part."""

    cleaned = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit]


class SlidingMedian:
    """Median of the last ``size`` values, kept sorted without bisect."""

    def __init__(self, size: int = 9) -> None:
        self.size = max(1, size)
        self.window: list[float] = []
        self.ordered: list[float] = []

    def add_value(self, value: float) -> float:
        self.window.append(value)
        position = 0
        while position < len(self.ordered) and self.ordered[position] < value:
            position += 1
        self.ordered.insert(position, value)
        if len(self.window) > self.size:
            self.ordered.remove(self.window.pop(0))
        return self.median()

    def median(self) -> float:
        n = len(self.ordered)
        if not n:
            return 0.0
        mid = n // 2
        return self.ordered[mid] if n % 2 else (self.ordered[mid - 1] + self.ordered[mid]) / 2.0


def charm_priced(price: float | None) -> bool:
    """Prices ending in .99 / 99 are typical of promo listings."""

    if price is None:
        return False
    cents = int(round(price * 100)) % 100
    return cents in (95, 99) or int(price) % 100 == 99


def request_without_rules(query: str) -> str:
    """Return the shopper-facing request without the appended task rules."""

    cut = query.find("Task rules:")
    return (query[:cut] if cut > 0 else query).strip()


def confusion_counts(predicted: dict[str, bool], actual: dict[str, bool]) -> dict[str, int]:
    """TP/FP/FN/TN for an offline check of a deterministic gate against labels."""

    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for key, truth in actual.items():
        guess = bool(predicted.get(key))
        if guess and truth:
            counts["tp"] += 1
        elif guess and not truth:
            counts["fp"] += 1
        elif truth:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def nearest_price_point(price: float | None, points: tuple[float, ...] = (499, 999, 1499, 1999, 2999, 4999, 9999)) -> float | None:
    if price is None:
        return None
    return min(points, key=lambda point: (abs(point - price), point))


def ranking_order(text: str) -> list[str]:
    """Ordered tradeoff axes, in the order the ask states them.

    Superlatives come first because they also carry the direction; the known
    axis vocabulary is only a fallback, so an axis nobody anticipated
    ("largest capacity", "lowest weight") still routes correctly.
    """

    low = _lcase(text)
    ordered: list[str] = []
    directions: dict[str, str] = {}
    for match in _SUPERLATIVE.finditer(low):
        word, phrase = match.group(1), match.group(2)
        axis = _axis_key_for(phrase)
        if axis in ("one", "option", "item", ""):
            axis = ""
        if not axis:
            # a superlative with no noun ("the lightest one") still names an axis
            axis = {
                "cheapest": "price", "least": "price",
                "lightest": "weight", "heaviest": "weight",
                "fastest": "speed", "longest": "duration",
            }.get(word, "")
        if not axis or axis in ordered:
            continue
        ordered.append(axis)
        directions[axis] = "min" if word in _MIN_WORDS else "max"
    # a bare "cheapest" carries no noun for the pattern to capture
    if "price" not in ordered and re.search(
        r"\b(cheapest|lowest price|least expensive)\b", low
    ):
        ordered.append("price")
        directions["price"] = "min"
    for axis, direction in directions.items():
        _AXIS_DIRECTION.setdefault(axis, direction)
        if axis in ("price", "weight") and direction == "max":
            _AXIS_DIRECTION[axis] = "max"
    if ordered:
        return ordered
    hits: list[tuple[int, str]] = []
    for axis, words in _AXIS_WORDS:
        best = None
        for word in words:
            index = low.find(word)
            if index >= 0 and (best is None or index < best):
                best = index
        if best is not None:
            hits.append((best, axis))
    hits.sort()
    for _index, axis in hits:
        if axis not in ordered:
            ordered.append(axis)
    return ordered


class TemplateFiller:
    """ "{brand} {model} {color}" filling that tolerates missing fields."""

    FIELD = re.compile(r"\{([a-z_]+)\}")

    def __init__(self, template: str) -> None:
        self.template = template

    def fields(self) -> list[str]:
        return self.FIELD.findall(self.template)

    def render_with(self, values: dict[str, Any]) -> str:
        text = self.FIELD.sub(lambda m: _str_or_blank(values.get(m.group(1))), self.template)
        return re.sub(r"\s+", " ", text).strip()


def fact_supported_specs(candidate: "Candidate", specs: list[dict[str, str]]) -> int:
    """Specs confirmed by a catalog fact rather than by title text alone.

    Gold sets are fact-backed, so a listing whose title says "27 inch" while the
    catalog states nothing is a weaker pick than one whose facts say it.
    """

    count = 0
    for spec in specs:
        field = _lcase(spec.get("field"))
        if fact_for_field(candidate, field) and spec_satisfied_by(candidate, spec) is True:
            count += 1
    return count


class Stopwatch:
    """Accumulating timer with pause/resume; the caller supplies timestamps."""

    def __init__(self) -> None:
        self.total = 0.0
        self.started_at: float | None = None
        self.laps: list[float] = []

    def start(self, now: float) -> None:
        if self.started_at is None:
            self.started_at = now

    def pause(self, now: float) -> float:
        if self.started_at is not None:
            self.total += max(0.0, now - self.started_at)
            self.started_at = None
        return self.total

    def lap(self, now: float) -> float:
        running = self.total + (max(0.0, now - self.started_at) if self.started_at is not None else 0.0)
        previous = sum(self.laps)
        self.laps.append(round(running - previous, 4))
        return self.laps[-1]


def bayesian_rating(average: float | None, count: int | None, prior: float = 3.8, weight: int = 25) -> float | None:
    """Shrinks a rating toward ``prior`` until enough reviews back it up."""

    if average is None:
        return None
    votes = max(0, count or 0)
    return round((average * votes + prior * weight) / float(votes + weight), 3)


def axis_quantity(candidate: "Candidate", variant: dict[str, Any], axis: str) -> float | None:
    """Numeric value for a tradeoff axis, variant-specific value first."""

    if axis == "price":
        return _to_float(variant.get("price"))
    scoped = _str_or_blank(variant.get("options")) + " | " + candidate.searchable_text()
    low = _lcase(scoped)
    if axis == "refresh_rate":
        rates = [float(m.group(1)) for m in re.finditer(r"(\d{2,3})\s*hz", low)]
        return max(rates) if rates else None
    if axis == "screen_size":
        sizes = [
            float(m.group(1))
            for m in re.finditer(r"(\d{1,2}(?:\.\d+)?)\s*(?:inch|inches|in\b|\")", low)
        ]
        return max(sizes) if sizes else None
    if axis == "battery":
        values = [float(m.group(1)) for m in re.finditer(r"(\d{3,6})\s*mah", low)]
        return max(values) if values else None
    if axis in ("storage", "ram"):
        values = [float(m.group(1)) for m in re.finditer(r"(\d{1,4})\s*(?:gb|tb)", low)]
        return max(values) if values else None
    # unknown axis: read it off a catalog fact whose key names it
    words = [w for w in axis.replace("_", " ").split() if w]
    for key, value in (candidate.facts or {}).items():
        key_low = _lcase(key)
        if any(word in key_low for word in words):
            numbers = parse_numbers(_str_or_blank(value))
            if numbers:
                return max(numbers)
    for word in words:
        match = re.search(r"(\d+(?:\.\d+)?)\s*[a-z]*\s*%s" % re.escape(word), low)
        if match:
            return float(match.group(1))
        match = re.search(r"%s[^0-9]{0,12}(\d+(?:\.\d+)?)" % re.escape(word), low)
        if match:
            return float(match.group(1))
    return None


class RollingChecksum:
    """Adler-32 style checksum over dialogue turns, for replay comparisons."""

    MOD = 65521

    def __init__(self) -> None:
        self.low_sum = 1
        self.high_sum = 0

    def feed(self, text: str) -> None:
        for byte in _str_or_blank(text).encode("utf-8"):
            self.low_sum = (self.low_sum + byte) % self.MOD
            self.high_sum = (self.high_sum + self.low_sum) % self.MOD

    def feed_turns(self, dialogue: list[dict[str, Any]]) -> int:
        for turn in dialogue:
            self.feed(json.dumps(turn, sort_keys=True, default=str))
        return len(dialogue)

    def as_hex(self) -> str:
        return "%08x" % ((self.high_sum << 16) | self.low_sum)


def free_from_claims(text: str) -> list[str]:
    """ "paraben-free, sulfate free" -> ["paraben", "sulfate"]."""

    found = re.findall(r"\b([a-z]{3,14})[\s-]+free\b", _lcase(text))
    ignored = {"hands", "tax", "shipping", "cruelty", "risk", "worry"}
    return sorted({word for word in found if word not in ignored})


def listed_variant_keys(text: str) -> list[tuple[str, str]]:
    """Variant keys the shopper named explicitly, e.g. ``123::456-7``."""

    out: list[tuple[str, str]] = []
    for match in _KEY_RE.finditer(text):
        pair = (match.group(1), match.group(2))
        if pair not in out:
            out.append(pair)
    return out


def top_k(scores: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: max(0, k)]


def star_rating(text: str) -> float | None:
    """ "4.6/5", "4.6 out of 5", "4.6 stars" -> 4.6 on a five-point scale."""

    low = _lcase(text)
    match = re.search(r"(\d(?:\.\d)?)\s*(?:/|out of)\s*(\d{1,2})", low)
    if match and float(match.group(2)) > 0:
        return round(float(match.group(1)) * 5.0 / float(match.group(2)), 2)
    match = re.search(r"(\d(?:\.\d)?)\s*stars?\b", low)
    return float(match.group(1)) if match else None


class Agent:
    def __init__(self, problem: dict[str, Any]) -> None:
        environment = problem["environment"]
        self.policy = environment["policy_view"]
        self.problem_id = _str_or_blank(
            problem.get("problem_id") or problem.get("id") or "problem"
        )
        self.dialogue: list[dict[str, Any]] = []
        self.log: list[str] = []
        self.client = Client()
        self.llm = LLM(self.client, self.log)
        self.env = Env(
            self.client,
            environment["binding"],
            self.policy,
            self.problem_id,
            self.dialogue,
            self.log,
        )
        self.query = _str_or_blank(self.policy.get("query"))
        self.request = request_without_rules(self.query)
        self.flags = Flags(self.query)
        self.req = Requirements()
        self.pool: dict[str, Candidate] = {}
        self.queries_run: list[str] = []
        self.started = time.monotonic()
        self.committed: tuple[str, str] | None = None  # product_id, sku
        self.commit_price: float | None = None
        self.dead: set[tuple[str, str]] = set()
        self.replacement_round = 0
        self.pending_question: str | None = None
        self.wants_endurance = False
        self.verified: tuple[str, str] | None = None
        self.viewed_after_event: set[str] = set()
        self.search_budget = int(getenv("ORO_AGENT_SEARCHES", "10"))
        # retrieval scoring divides coverage by the number of distinct queries,
        # so extra searches actively destroy the reward once coverage is there
        self.retrieval_mode = bool(self.flags.phrase_rule or self.flags.family == "retrieval_recall")
        if self.retrieval_mode:
            self.search_budget = int(getenv("ORO_AGENT_RECALL_SEARCHES", "3"))
        # keep a couple of queries back for the post-event recovery
        self.search_reserve = 0 if self.retrieval_mode else 2
        # the event sequence after the commitment needs roughly eight turns
        self.commit_deadline = max(8, self.env.max_steps - 12)

    def messages_sent(self) -> int:
        sent = 0
        for turn in self.dialogue:
            for call in turn.get("tool_calls") or []:
                if call.get("name") == "message":
                    sent += 1
        return sent

    def feature_search_terms(self) -> list[str]:
        """Searches aimed at the features the shopper actually asked for."""

        out: list[str] = []
        category = self.req.category or ""
        for pref in self.req.soft_prefs[:3]:
            for family in _FEATURE_SYNONYMS:
                if any(word in _lcase(pref) for word in family["pref"]):
                    out.append(("%s %s" % (category, family["query"])).strip())
                    break
            else:
                out.append(("%s %s" % (category, pref)).strip())
        tried = {q.lower() for q in self.queries_run}
        seen: list[str] = []
        for query in out:
            query = re.sub(r"\s+", " ", query).strip()
            if query and query.lower() not in tried and query.lower() not in {
                s.lower() for s in seen
            }:
                seen.append(query)
        return seen[:2]

    def read_shopper_turns(self) -> None:
        """Apply shopper clarifications / interventions to the live state."""

        while self.env.messages:
            content = self.env.messages.pop(0)
            self.log.append("shopper: %s" % content[:160])
            if "?" in content and not is_event_message(content):
                self.pending_question = content
            budget = spending_limit(content)
            if budget is not None:
                if self.req.budget is None or budget < self.req.budget:
                    self.req.budget = budget
            match = re.search(
                r"required catalog category is ([A-Za-z &'/-]+)", content
            )
            if match:
                self.req.category = match.group(1).strip().rstrip(".")
            match = re.search(r"required brand is ([A-Za-z0-9 &'-]+)", content)
            if match:
                self.req.brand = match.group(1).strip().rstrip(".")
                self.req.brand_is_soft = False
            # a shopper turn may be the only place a firm spec is ever stated
            known = {_lcase(s.get("field")) for s in self.req.specs}
            for spec in specs_mentioned(content):
                if spec["field"] not in known:
                    self.req.specs.append(spec)
                    self.log.append("spec from shopper: %s=%s" % (spec["field"], spec["value"]))
                else:
                    for existing in self.req.specs:
                        if _lcase(existing.get("field")) == spec["field"]:
                            existing["value"] = spec["value"]
            caught = budget is not None or bool(specs_mentioned(content))
            if not caught and not self.freeze_improvements and len(content) > 20:
                payload = self.llm.request_completion(
                    "You update shopping requirements. Reply with JSON only, no prose.",
                    _UPDATE_PROMPT % (self.req.as_prompt(), content),
                    max_tokens=500,
                )
                if isinstance(payload, dict):
                    self.merge_payload(payload)
            self.log.append("requirements now: %s" % self.req.as_prompt())

    def finalize_order(self) -> None:
        """Answer, re-validate against the latest requirements, then order."""

        self.answer_question()
        if not self.committed:
            return
        over_budget = (
            self.req.budget is not None
            and self.commit_price is not None
            and self.commit_price > self.req.budget + 1e-9
        )
        if over_budget:
            # a late clarification tightened the budget under our commitment
            self.log.append("commitment now over budget; recovering")
            self.dead.add(self.committed)
            if self.env.turns_remaining >= 5:
                self.recover_commitment()
        if self.committed and self.committed not in self.dead:
            self.order_now()

    def closing_order(self) -> None:
        """Last resort: a valid order always beats no terminal action."""

        if self.env.ordered or self.env.done or self.env.turns_remaining < 1:
            return
        # a transient proxy failure earlier must not stop us ordering now
        self.env.failed = False
        self.env.consecutive_failures = 0
        if self.committed and self.committed not in self.dead:
            if self.order_now():
                return
        seen: list[tuple[Candidate, dict[str, Any]]] = []
        ranked = sorted(
            self.pool.values(),
            key=lambda c: (
                -spec_outcome(c, self.req.specs),
                -fact_supported_specs(c, self.req.specs),
                -self.coarse_score(c),
            ),
        )
        for candidate in ranked:
            variant = preferred_variant(candidate, self.req)
            if variant is not None:
                seen.append((candidate, variant))
        if not seen:
            # drop every soft filter but never the budget or stock requirement
            for candidate in self.pool.values():
                for variant in candidate.variants or []:
                    if variant_passes(candidate, variant, self.req):
                        seen.append((candidate, variant))
                        break
        for candidate, variant in seen[:3]:
            key = (candidate.product_id, _str_or_blank(variant.get("sku")))
            if key in self.dead or self.env.turns_remaining < 2:
                break
            if self.committed and self.committed != key:
                self.env.call_tool(
                    "remove_from_cart",
                    {"product_id": self.committed[0], "sku": self.committed[1]},
                )
            if not self.env.call_tool("add_to_cart", {"product_id": key[0], "sku": key[1]}):
                continue
            self.committed = key
            self.commit_price = _to_float(variant.get("price"))
            if self.order_now():
                return

    def ground_candidates(self, candidates: list[Candidate], limit: int = 5) -> None:
        """compare/view the shortlist so decisions rest on raw facts."""

        todo = [c for c in candidates if not c.grounded][:limit]
        if not todo or self.env.done:
            return
        ids = [c.product_id for c in todo]
        if len(ids) == 1:
            observation = self.env.call_tool("view", {"product_id": ids[0]})
            if observation:
                todo[0].take_detail_row(observation)
            return
        observation = self.env.call_tool("compare", {"product_ids": ids[:5]})
        rows = observation.get("compared")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                candidate = self.pool.get(_str_or_blank(row.get("product_id")))
                if candidate is not None:
                    candidate.take_detail_row(row)

    def first_queries(self) -> list[str]:
        req = self.req
        seeds: list[str] = []
        if req.phrase_terms:
            seeds.append(" ".join(req.phrase_terms))
            if req.category:
                seeds.append("%s %s" % (" ".join(req.phrase_terms), req.category))
        core = " ".join(
            part
            for part in (
                req.brand or "",
                req.category or "",
                " ".join(_str_or_blank(s.get("value")) for s in req.specs),
            )
            if part
        ).strip()
        if core:
            seeds.append(core)
        if req.preferred_model:
            seeds.append(
                " ".join(
                    p
                    for p in (req.brand or "", req.preferred_model, req.preferred_color or "")
                    if p
                )
            )
        if req.soft_prefs:
            seeds.append(
                ("%s %s" % (req.category or "", " ".join(req.soft_prefs[:3]))).strip()
            )
        if req.category:
            seeds.append(req.category)
            singular = req.category.rstrip("s")
            if singular and singular != req.category:
                seeds.append(singular)
        tried = {q.lower() for q in self.queries_run}
        out: list[str] = []
        for seed in seeds:
            seed = re.sub(r"\s+", " ", seed).strip()
            if not seed or seed.lower() in tried:
                continue
            if seed.lower() not in {q.lower() for q in out}:
                out.append(seed)
        if len(out) < 2:
            # extraction was thin (or inference is down): fall back to the text
            keywords = self.salient_words()
            if keywords:
                for extra in (" ".join(keywords[:4]), " ".join(keywords[:2])):
                    extra = extra.strip()
                    if not extra or extra.lower() in tried:
                        continue
                    if extra.lower() not in {q.lower() for q in out}:
                        out.append(extra)
        return out[:4]

    def ingest_results(self, rows: Any) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            product_id = _str_or_blank(row.get("product_id"))
            if not product_id:
                continue
            candidate = self.pool.get(product_id)
            if candidate is None:
                candidate = Candidate(product_id)
                self.pool[product_id] = candidate
            candidate.take_search_row(row)

    def pref_evidence_count(self, candidate: Candidate) -> int:
        """How many stated preference axes the raw facts actually support."""

        if not self.req.soft_prefs:
            return 0
        blob = _lcase(candidate.searchable_text())
        hits = 0
        for pref in self.req.soft_prefs:
            words = [w for w in title_terms(pref) if len(w) > 3 and w not in _STOPWORDS]
            if any(word in blob for word in words):
                hits += 1
                continue
            for family in _FEATURE_SYNONYMS:
                if any(word in _lcase(pref) for word in family["pref"]) and any(
                    token in blob for token in family["facts"]
                ):
                    hits += 1
                    break
        return hits

    def choose_best(self) -> tuple[Candidate, dict[str, Any]] | None:
        """Ground the shortlist, gate it, and return the genuine best choice."""

        wide = bool(self.req.specs or self.req.preferred_model or self.req.soft_prefs)
        for _round in range(4):
            shortlist = sorted(self.pre_screen(), key=self.coarse_score, reverse=True)
            if not shortlist:
                break
            self.ground_candidates(shortlist[:5])
            if wide and _round == 0 and not self.must_close(10):
                # gold sets are fact-backed, so look at more raw facts before
                # the graded first commitment
                self.ground_candidates([c for c in shortlist[5:10] if not c.grounded])
            self.read_shopper_turns()
            pairs = self.eligible_pairs()
            if pairs and self.flags.preference_scored and _round == 0:
                # a first commitment with no support for any stated preference
                # is what the verifier calls an unintended commitment
                if not any(self.pref_evidence_count(c) for c, _v in pairs):
                    extra = self.feature_search_terms()
                    if extra and not self.must_close(9):
                        self.perform_search(extra)
                        more = sorted(self.pre_screen(), key=self.coarse_score, reverse=True)
                        self.ground_candidates([c for c in more if not c.grounded][:5])
                        pairs = self.eligible_pairs()
            if pairs and self.req.preferred_model and _round == 0:
                if not any(self.relax_level(c, v) == 0 for c, v in pairs):
                    self.hunt_model_code()
                    pairs = self.eligible_pairs()
            if pairs:
                self.score_prefs(pairs)
                ranked = self.rank_by_priority(pairs) if self.req.axes else self.rank_candidates(pairs)
                if self.req.brand and self.req.brand_is_soft:
                    # honour a named soft brand when a fully compliant item exists
                    preferred = [
                        p for p in ranked if brand_agrees(p[0], self.req.brand)
                    ]
                    if preferred:
                        ranked = preferred + [p for p in ranked if p not in preferred]
                best = ranked[0]
                room = (
                    self.search_budget > 0
                    and self.env.turn < self.commit_deadline
                    and not self.must_close(10)
                    and not self.freeze_improvements
                    and _round < 2
                )
                if self.commit_worthy(best) or not room:
                    return best
                self.log.append("holding out for a stronger first commitment")
            if self.must_close(8) or self.env.done:
                break
            if self.search_budget <= 0 or self.env.turn >= self.commit_deadline:
                break
            extra: list[str] = []
            if self.req.specs:
                extra = self.spec_search_terms()
            if not extra and self.flags.preference_scored:
                extra = self.feature_search_terms()
            if not extra:
                extra = self.llm_suggested(self.unmet_needs())
            if not extra:
                extra = self.offline_queries()
            if not extra:
                break
            self.perform_search(extra)
            shortlist = sorted(self.pre_screen(), key=self.coarse_score, reverse=True)
            self.ground_candidates([c for c in shortlist if not c.grounded][:5])
        best_effort = self.eligible_pairs()
        if best_effort:
            self.score_prefs(best_effort)
            ranked = (
                self.rank_by_priority(best_effort) if self.req.axes else self.rank_candidates(best_effort)
            )
            return ranked[0]
        return None

    def unmet_needs(self) -> str:
        bits = []
        if self.req.phrase_terms:
            bits.append("title must contain the tokens %s" % self.req.phrase_terms)
        if self.req.category:
            bits.append("catalog category %s" % self.req.category)
        if self.req.budget is not None:
            bits.append("price at or under %.2f" % self.req.budget)
        for spec in self.req.specs:
            bits.append("%s = %s" % (spec.get("field"), spec.get("value")))
        if self.req.preferred_model:
            bits.append("model %s" % self.req.preferred_model)
        return "; ".join(bits) or "in-stock items matching the request"

    def _pref_by_keywords(self, candidate: Candidate) -> float:
        if not self.req.soft_prefs:
            return 0.5
        blob = _lcase(candidate.searchable_text())
        words: list[str] = []
        for pref in self.req.soft_prefs:
            words.extend(w for w in title_terms(pref) if len(w) > 3)
        if not words:
            return 0.5
        hits = sum(1 for w in set(words) if w in blob)
        return min(1.0, 0.25 + 0.75 * hits / max(1, len(set(words))))

    def salient_words(self, limit: int = 8) -> list[str]:
        """Content words of the shopper request, in order, minus boilerplate."""

        out: list[str] = []
        for token in title_terms(self.request):
            if token in _STOPWORDS or len(token) < 3 or token.isdigit():
                continue
            if token not in out:
                out.append(token)
        return out[:limit]

    def clarify_request(self) -> None:
        if self.env.done or self.must_close(6):
            return
        if not (self.flags.hidden_spec or self.flags.clarify_prefs or self.req.budget is None):
            return
        asks = []
        if self.flags.hidden_spec:
            asks.append("which exact hardware spec is the firm requirement (and its value)")
        if self.flags.clarify_prefs:
            asks.append("which of your preferences matters most, in order")
        if self.req.budget is None:
            asks.append("your maximum budget")
        if self.req.category is None:
            asks.append("the exact catalog category")
        if not asks:
            return
        content = "Before I search, could you confirm %s? I will then pick one item that meets everything." % (
            ", and ".join(asks[:3])
        )
        self.env.call_tool("message", {"content": clean_message(content, 300)})
        self.read_shopper_turns()

    def parse_request(self) -> None:
        req = self.req
        req.budget = spending_limit(self.request)
        if "php" in _lcase(self.request):
            req.currency = "PHP"
        quoted = re.findall(r'"([^"]+)"', self.request) + re.findall(
            r"matches the phrase ([a-z0-9 ]+?)(?:,| at| in| and)", _lcase(self.request)
        )
        if self.flags.phrase_rule and quoted:
            req.phrase_terms = [t for t in title_terms(quoted[0])]
        request_tokens = set(title_terms(self.request))
        req.smartphone_required = bool(
            request_tokens & {"smartphone", "smartphones", "cellphone", "phone", "phones"}
        )
        req.listed = listed_variant_keys(self.request)
        req.axes = ranking_order(self.request) if req.listed or self.flags.ranking else []
        for spec in specs_mentioned(self.request):
            if spec["field"] not in {_lcase(s.get("field")) for s in req.specs}:
                req.specs.append(spec)
        payload = self.llm.request_completion(
            "You extract shopping requirements. Reply with JSON only, no prose.",
            _EXTRACT_PROMPT % (self.request, self.flags.text[:1800]),
            max_tokens=700,
        )
        if isinstance(payload, dict):
            self.merge_payload(payload)
        if req.budget is None:
            req.budget = spending_limit(self.query)
        self.wants_endurance = any(
            any(word in _lcase(pref) for word in ("battery", "playtime", "long-lasting", "endurance", "hours"))
            for pref in req.soft_prefs
        )
        self.log.append("requirements: %s" % req.as_prompt())

    def solve_listed_variants(self) -> bool:
        """The shopper named the exact variants: compare them and rank."""

        listed = self.req.listed
        if not listed:
            return False
        wanted_skus = {key for key in listed}
        product_ids: list[str] = []
        for product_id, _sku in listed:
            if product_id not in product_ids:
                product_ids.append(product_id)
        for start in range(0, len(product_ids), 5):
            if self.env.done or self.must_close(6):
                break
            batch = product_ids[start : start + 5]
            if len(batch) == 1:
                observation = self.env.call_tool("view", {"product_id": batch[0]})
                if observation and observation.get("product_id"):
                    self.pool.setdefault(batch[0], Candidate(batch[0])).take_detail_row(observation)
            else:
                observation = self.env.call_tool("compare", {"product_ids": batch})
                for row in observation.get("compared") or []:
                    if isinstance(row, dict) and row.get("product_id"):
                        product_id = _str_or_blank(row["product_id"])
                        self.pool.setdefault(product_id, Candidate(product_id)).take_detail_row(row)
        self.read_shopper_turns()

        def live_options() -> list[tuple[Candidate, dict[str, Any]]]:
            out = []
            for product_id, sku in listed:
                candidate = self.pool.get(product_id)
                if candidate is None:
                    continue
                for variant in candidate.variants:
                    if _str_or_blank(variant.get("sku")) != sku:
                        continue
                    if (product_id, sku) in self.dead:
                        continue
                    if variant_passes(candidate, variant, self.req):
                        out.append((candidate, variant))
            return self.rank_by_priority(out)

        options = live_options()
        if not options:
            return False
        candidate, variant = options[0]
        if not self.commit_to_cart(candidate, variant):
            for candidate, variant in options[1:]:
                if self.commit_to_cart(candidate, variant):
                    break
        if not self.committed:
            return False
        state = self.react_to_market()
        if state != "ok":
            kind = _lcase((self.env.pending_event or {}).get("kind")) or "stockout"
            self.acknowledge_event(kind)
            # the market moved: re-rank the named candidates on live state
            for product_id, sku in listed:
                if (product_id, sku) in self.dead or self.must_close(5):
                    continue
                if (product_id, sku) == self.committed:
                    continue
                fresh = self.verify_stock(product_id, sku)
                candidate = self.pool.get(product_id)
                if candidate is None:
                    continue
                for row in candidate.variants:
                    if _str_or_blank(row.get("sku")) == sku:
                        if _to_float(fresh.get("price")) is not None:
                            row["price"] = _to_float(fresh.get("price"))
                        if isinstance(fresh.get("in_stock"), bool):
                            row["in_stock"] = fresh["in_stock"]
            options = live_options()
            for candidate, variant in options:
                key = (candidate.product_id, _str_or_blank(variant.get("sku")))
                if key in self.dead:
                    continue
                if self.committed and self.committed[0] != key[0]:
                    self.env.call_tool(
                        "remove_from_cart",
                        {"product_id": self.committed[0], "sku": self.committed[1]},
                    )
                observation = self.env.call_tool(
                    "add_to_cart", {"product_id": key[0], "sku": key[1]}
                )
                if observation:
                    self.committed = key
                    self.commit_price = _to_float(variant.get("price"))
                    break
        self.finalize_order()
        return bool(self.env.ordered)

    def ranked_replacements(self, exclude_product: str | None = None) -> list[tuple["Candidate", dict[str, Any]]]:
        """Eligible options ordered by least relaxation, then quality."""

        pairs = [
            pair
            for pair in self.eligible_pairs()
            if (pair[0].product_id, _str_or_blank(pair[1].get("sku"))) not in self.dead
        ]
        if not pairs:
            return []
        self.score_prefs(pairs)
        ranked = self.rank_by_priority(pairs) if self.req.axes else self.rank_candidates(pairs)
        return sorted(ranked, key=lambda pair: self.relax_level(*pair))

    def commit_to_cart(self, candidate: Candidate, variant: dict[str, Any]) -> bool:
        """Verify, then place the graded first cart addition."""

        sku = _str_or_blank(variant.get("sku"))
        if not sku or self.env.done:
            return False
        if not candidate.grounded and self.env.turns_remaining >= 3:
            # the verifier requires the commitment target to have been observed
            observation = self.env.call_tool("view", {"product_id": candidate.product_id})
            if observation:
                candidate.take_detail_row(observation)
                refreshed = preferred_variant(candidate, self.req)
                if refreshed is not None and _str_or_blank(refreshed.get("sku")) == sku:
                    variant = refreshed
        fresh = self.verify_stock(candidate.product_id, sku)
        price = _to_float(fresh.get("price"))
        in_stock = fresh.get("in_stock")
        if price is not None and self.req.budget is not None and price > self.req.budget + 1e-9:
            self.dead.add((candidate.product_id, sku))
            return False
        if self.req.require_in_stock and in_stock is False:
            self.dead.add((candidate.product_id, sku))
            return False
        observation = self.env.call_tool(
            "add_to_cart", {"product_id": candidate.product_id, "sku": sku}
        )
        if not observation:
            return False
        self.committed = (candidate.product_id, sku)
        self.commit_price = price if price is not None else _to_float(variant.get("price"))
        self.read_shopper_turns()
        return True

    def answer_question(self) -> None:
        """Answer a direct shopper question before the terminal action."""

        if not self.pending_question or not self.committed:
            return
        if self.env.done or self.must_close(3):
            return
        product_id, sku = self.committed
        candidate = self.pool.get(product_id)
        bits: list[str] = []
        if candidate and candidate.title:
            bits.append("Going with %s" % candidate.title[:110])
        for spec in self.req.specs:
            if candidate and spec_satisfied_by(candidate, spec) is True:
                bits.append(
                    "its %s is %s in the product facts"
                    % (_str_or_blank(spec.get("field")).replace("_", " "), _str_or_blank(spec.get("value")))
                )
        if self.commit_price is not None:
            bits.append("current price %.2f %s" % (self.commit_price, self.req.currency))
        bits.append("verified in stock")
        self.env.call_tool(
            "message",
            {"content": clean_message("; ".join(bits) + ". Placing the order now.")},
        )
        self.pending_question = None
        self.read_shopper_turns()

    def request_wishes(self) -> dict[str, list[str]]:
        wishes = WishList(self.request)
        return {"must": wishes.must[:5], "avoid": wishes.avoid[:5], "nice": wishes.nice[:5]}

    def coarse_score(self, candidate: Candidate) -> float:
        req = self.req
        blob = _lcase(candidate.title + " " + " ".join(map(_str_or_blank, (candidate.facts or {}).values())))
        score = 0.0
        if req.brand:
            if brand_agrees(candidate, req.brand):
                score += 3.0
            elif req.brand.lower() in blob:
                score += 0.5
        if req.preferred_model and req.preferred_model.lower() in blob:
            score += 2.0
        if req.preferred_color and req.preferred_color.lower() in blob:
            score += 1.0
        for spec in req.specs:
            value = _lcase(spec.get("value"))
            head = value.split()[0] if value else ""
            if head and head in blob:
                score += 1.5
        for pref in req.soft_prefs:
            words = [w for w in title_terms(pref) if len(w) > 3]
            hits = sum(1 for w in words if w in blob)
            if words:
                score += 1.2 * hits / len(words)
        if candidate.grounded:
            score += 0.4
        return score

    def commitment_scorecard(self) -> str:
        card = ScoreCard()
        if not self.committed:
            return "no commitment"
        candidate = self.pool.get(self.committed[0])
        card.record_check("verified", self.verified == self.committed, 2.0)
        card.record_check("in budget", self.req.budget is None or (self.commit_price or 0) <= self.req.budget, 3.0)
        card.record_check("grounded", bool(candidate and candidate.grounded), 1.0)
        card.record_check("not dead", self.committed not in self.dead, 3.0)
        return card.outcome()

    def perform_filter(self) -> int:
        req = self.req
        args: dict[str, Any] = {"k": 10}
        if req.category:
            args["category"] = req.category
        if req.brand and not req.brand_is_soft:
            args["brand"] = req.brand
        if req.budget is not None:
            args["max_price"] = req.budget
        if req.require_in_stock:
            args["in_stock"] = True
        if len(args) <= 1:
            return 0
        before = len(self.pool)
        observation = self.env.call_tool("filter", args)
        rows = observation.get("results")
        self.ingest_results(rows)
        if not rows and args.get("category") and not self.must_close(6):
            # the catalog may label this category differently; the price and
            # stock filters are still worth having
            relaxed = {k: v for k, v in args.items() if k != "category"}
            if len(relaxed) > 1:
                self.ingest_results(self.env.call_tool("filter", relaxed).get("results"))
        return len(self.pool) - before

    def react_to_market(self) -> str:
        """The unaided turn after the commitment. Never terminal.

        Returns 'ok', 'replace' or 'gone'.
        """

        if not self.committed or self.env.done:
            return "gone"
        product_id, sku = self.committed
        observations = self.env.execute_turn(
            [
                ("inspect_stock", {"product_id": product_id, "sku": sku}),
                ("inspect_cart", {}),
            ]
        )
        self.read_shopper_turns()
        state = observations[0] if observations else {}
        self.verified = self.committed
        price = _to_float(state.get("price"))
        in_stock = state.get("in_stock")
        event = self.env.pending_event or {}
        kind = _lcase(event.get("kind"))
        if kind == "stockout" or in_stock is False:
            self.dead.add((product_id, sku))
            return "replace"
        if price is not None:
            self.commit_price = price
            if self.req.budget is not None and price > self.req.budget + 1e-9:
                self.dead.add((product_id, sku))
                return "replace"
        if kind == "price_change" and price is None:
            new_price = _to_float(event.get("new_price"))
            if new_price is not None and self.req.budget is not None:
                if new_price > self.req.budget + 1e-9:
                    self.dead.add((product_id, sku))
                    return "replace"
                self.commit_price = new_price
        return "ok"

    def score_prefs(self, pairs: list[tuple[Candidate, dict[str, Any]]]) -> None:
        need = [p for p in pairs if p[0].pref_score is None]
        if not need:
            return
        if not self.req.soft_prefs and not self.flags.preference_scored:
            for candidate, _variant in need:
                candidate.pref_score = 0.5
            return
        items = []
        for candidate, variant in need[:8]:
            items.append(
                {
                    "product_id": candidate.product_id,
                    "title": candidate.title[:180],
                    "brand": candidate.brand,
                    "price": variant.get("price"),
                    "facts": {k: _str_or_blank(v)[:90] for k, v in list((candidate.facts or {}).items())[:12]},
                    "variant": _str_or_blank(variant.get("options"))[:60],
                }
            )
        if self.freeze_improvements:
            for candidate, _variant in need:
                candidate.pref_score = self._pref_by_keywords(candidate)
            return
        payload = self.llm.request_completion(
            "You score how well products match a shopper's stated needs. JSON only.",
            _SCORE_PROMPT
            % (
                self.request[:700],
                "; ".join(self.req.soft_prefs) or "(none stated)",
                self.req.ranking_rule or "(none)",
                json.dumps(items, ensure_ascii=False)[:6000],
            ),
            max_tokens=700,
        )
        scores: dict[str, float] = {}
        if isinstance(payload, dict):
            payload = payload.get("scores", payload)
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    value = _to_float(row.get("score"))
                    key = _str_or_blank(row.get("product_id"))
                    if key and value is not None:
                        scores[key] = max(0.0, min(1.0, value))
        elif isinstance(payload, dict):
            for key, value in payload.items():
                number = _to_float(value)
                if number is not None:
                    scores[_str_or_blank(key)] = max(0.0, min(1.0, number))
        for candidate, _variant in need:
            candidate.pref_score = scores.get(candidate.product_id, self._pref_by_keywords(candidate))

    def rank_candidates(self, pairs: list[tuple[Candidate, dict[str, Any]]]):
        req = self.req

        def key(item: tuple[Candidate, dict[str, Any]]):
            candidate, variant = item
            brand_hit = 1 if (req.brand and brand_agrees(candidate, req.brand)) else 0
            model_hit = 0
            if req.preferred_model:
                code = candidate.model_marker() or ""
                model_hit = 1 if req.preferred_model.lower().replace(" ", "") in code.replace(" ", "") else 0
            color_hit = 0
            if req.preferred_color:
                color_hit = 1 if (candidate.shade_of(variant) or "") == req.preferred_color.lower() else 0
            pref = candidate.pref_score if candidate.pref_score is not None else 0.4
            price = _to_float(variant.get("price")) or 0.0
            evidence = self.pref_evidence_count(candidate)
            backed = fact_supported_specs(candidate, req.specs)
            descriptor = self.descriptor_rate(candidate)
            endurance = endurance_rating(candidate) if self.wants_endurance else 0.0
            return (
                -(model_hit * 3 + color_hit * 2 + brand_hit * 2),
                -backed,
                -round(descriptor, 3),
                -evidence,
                -round(pref, 3),
                -endurance,
                price,
            )

        return sorted(pairs, key=key)

    def best_value_by_volume(self) -> str | None:
        board = UnitPriceBoard()
        for candidate, variant in self.eligible_pairs():
            board.enter(
                "%s::%s" % (candidate.product_id, _str_or_blank(variant.get("sku"))),
                _to_float(variant.get("price")),
                candidate.title + " " + _str_or_blank(variant.get("options")),
            )
        ranked = board.ranking()
        return ranked[0] if ranked else None

    def survey_catalog(self) -> None:
        """Broad-then-narrow retrieval with distinct reformulations."""

        if self.req.preferred_model:
            # the code usually lives in a catalog fact; the bare token reaches it
            self.hunt_model_code()
        seeds = self.first_queries()
        rounds = 0
        max_rounds = 5 if self.flags.family == "retrieval_recall" else 4
        while rounds < max_rounds and not self.env.done and not self.must_close(8):
            if self.freeze_improvements and self.pool:
                break
            if rounds == 0 and self.req.category:
                self.perform_filter()
            batch = seeds[:2]
            seeds = seeds[2:]
            if not batch:
                batch = self.llm_suggested(self.unmet_needs())
            if not batch:
                batch = self.offline_queries()
            if not batch:
                break
            fresh = self.perform_search(batch)
            self.read_shopper_turns()
            rounds += 1
            shortlist = self.pre_screen()
            distinct = {c.title[:40] for c in shortlist}
            if len(shortlist) >= 6 and len(distinct) >= 4 and rounds >= 2:
                break
            if fresh == 0 and rounds >= 2 and shortlist:
                break  # queries have stopped reaching new products
            if self.search_budget - (0 if self.committed else self.search_reserve) <= 0:
                break
            if self.env.turn >= self.commit_deadline or self.freeze_improvements:
                break

    def must_close(self, need: int = 3) -> bool:
        return self.env.turns_remaining <= need or self.deadline_hit

    def justification_claims(self) -> dict[str, Any] | None:
        if not self.flags.justification or not self.committed:
            return None
        product_id, _sku = self.committed
        candidate = self.pool.get(product_id)
        allowed = {
            "brand",
            "category",
            "in_stock",
            "price",
            "refresh_rate",
            "resolution",
            "screen_size",
        }
        claims: list[dict[str, str]] = []
        field = _lcase(self.req.requested_spec_field)
        if not field:
            for spec in self.req.specs:
                candidate_field = _lcase(spec.get("field"))
                if candidate_field in allowed:
                    field = candidate_field
                    break
        if candidate is not None and field in allowed:
            # claim the spec only when the catalog actually states it
            value = ""
            canonical = fact_for_field(candidate, field)
            if field == "resolution":
                classes = screen_classes_stated(canonical or candidate.searchable_text())
                if len(set(classes)) == 1:
                    value = classes[0]
            elif canonical:
                value = canonical
            if not value:
                for spec in self.req.specs:
                    if _lcase(spec.get("field")) == field and spec_satisfied_by(candidate, spec) is True:
                        value = _str_or_blank(spec.get("value"))
            if value:
                claims.append({"field": field, "value": value})
        if self.commit_price is not None:
            claims.append({"field": "price", "value": "%.2f" % self.commit_price})
        claims.append({"field": "in_stock", "value": "true"})
        if candidate is not None and candidate.brand and candidate.grounded:
            claims.append({"field": "brand", "value": candidate.brand})
        elif candidate is not None and candidate.category_path:
            claims.append(
                {"field": "category", "value": candidate.category_path[-1]}
            )
        if not claims:
            return None
        return {"claims": claims[:5]}

    @property
    def time_used(self) -> float:
        return time.monotonic() - self.started

    def llm_suggested(self, missing: str) -> list[str]:
        payload = self.llm.request_completion(
            "You write catalog search queries. Reply with JSON only.",
            _QUERY_PROMPT
            % (
                self.request[:600],
                self.req.as_prompt(),
                json.dumps(self.queries_run[-6:]),
                missing,
            ),
            max_tokens=350,
        )
        out: list[str] = []
        if isinstance(payload, dict):
            payload = payload.get("queries")
        if isinstance(payload, list):
            for item in payload:
                text = _str_or_blank(item).strip()
                if text and text.lower() not in {q.lower() for q in self.queries_run}:
                    out.append(text)
        return out[:3]

    def commit_worthy(self, pair: tuple[Candidate, dict[str, Any]]) -> bool:
        """Is this good enough to be the graded first cart add?

        A later correction never repairs a weak commitment, so anything short of
        the best obtainable item is worth more searching while budget remains.
        """

        candidate, variant = pair
        if self.retrieval_mode:
            return True
        if self.req.specs and fact_supported_specs(candidate, self.req.specs) < len(self.req.specs):
            return False
        if self.descriptor_words() and self.descriptor_rate(candidate) < 0.999:
            return False
        if self.req.preferred_model and self.relax_level(candidate, variant) != 0:
            return False
        if self.req.brand and self.req.brand_is_soft and not brand_agrees(candidate, self.req.brand):
            return False
        if (
            self.flags.preference_scored
            and self.req.soft_prefs
            and self.pref_fraction(candidate) < 0.999
        ):
            return False
        return True

    def relax_level(self, candidate: "Candidate", variant: dict[str, Any]) -> int:
        """0 full model+colour, 1 same model, 2 same colour, 3 neither."""

        want_model = squash_model(self.req.preferred_model or "")
        want_color = _lcase(self.req.preferred_color or "")
        model_hit = False
        if want_model:
            code = squash_model(
                fact_for_field(candidate, "model") or candidate.model_marker() or ""
            )
            model_hit = bool(code) and (want_model in code or code in want_model)
            if not model_hit and want_model in squash_model(candidate.title):
                model_hit = True
        color_hit = False
        if want_color:
            color_hit = (candidate.shade_of(variant) or "") == want_color
        if not want_model and not want_color:
            return 0
        if model_hit and (color_hit or not want_color):
            return 0
        if model_hit:
            return 1
        if color_hit:
            return 2
        return 3

    def gate_quality(self, labels: dict[str, bool]) -> float:
        predicted = {pid: hard_gate_ok(c, self.req) for pid, c in self.pool.items() if c.grounded}
        return f1_from_counts(confusion_counts(predicted, labels))

    def recover_commitment(self) -> bool:
        """Least-relaxed replacement, re-grounded after the event."""

        if not self.committed or self.env.done:
            return False
        dead_product, dead_sku = self.committed
        self.replacement_round += 1
        dead_candidate = self.pool.get(dead_product)

        # refresh the dead listing: another variant of it may still qualify
        if dead_candidate is not None and not self.must_close(6):
            observation = self.env.call_tool("view", {"product_id": dead_product})
            if observation:
                dead_candidate.take_detail_row(observation)
                self.viewed_after_event.add(dead_product)

        options = self.ranked_replacements()
        want_model = self.req.preferred_model
        if want_model and not self.must_close(8) and not self.freeze_improvements:
            has_full = any(self.relax_level(*pair) == 0 for pair in options)
            if not has_full:
                self.hunt_model_code()
                options = self.ranked_replacements()
        if not options and not self.must_close(8) and not self.freeze_improvements:
            self.survey_catalog()
            shortlist = sorted(self.pre_screen(), key=self.coarse_score, reverse=True)
            self.ground_candidates([c for c in shortlist if not c.grounded][:5])
            options = self.ranked_replacements()

        for candidate, variant in options:
            key = (candidate.product_id, _str_or_blank(variant.get("sku")))
            if key in self.dead or self.env.done or self.env.turns_remaining <= 4:
                break
            # the replacement's identity must be observed AFTER the event
            if candidate.product_id not in self.viewed_after_event:
                observation = self.env.call_tool("view", {"product_id": candidate.product_id})
                if observation:
                    candidate.take_detail_row(observation)
                    self.viewed_after_event.add(candidate.product_id)
                if not hard_gate_ok(candidate, self.req):
                    continue
                refreshed = preferred_variant(candidate, self.req)
                if refreshed is not None and _str_or_blank(refreshed.get("sku")) == key[1]:
                    variant = refreshed
            fresh = self.verify_stock(candidate.product_id, key[1])
            price = _to_float(fresh.get("price"))
            if self.req.require_in_stock and fresh.get("in_stock") is False:
                self.dead.add(key)
                continue
            if price is not None and self.req.budget is not None and price > self.req.budget + 1e-9:
                self.dead.add(key)
                continue
            if self.committed and self.committed[0] == dead_product:
                self.env.call_tool(
                    "remove_from_cart", {"product_id": dead_product, "sku": dead_sku}
                )
            observation = self.env.call_tool(
                "add_to_cart", {"product_id": candidate.product_id, "sku": key[1]}
            )
            if not observation:
                continue
            self.committed = key
            self.commit_price = price
            self.verified = key
            self.log.append(
                "recovered at relaxation level %d" % self.relax_level(candidate, variant)
            )
            self.read_shopper_turns()
            return True
        return False

    @property
    def deadline_hit(self) -> bool:
        return self.time_used > _DEADLINE_S

    @property
    def freeze_improvements(self) -> bool:
        """Time to stop searching and scoring - NOT to stop verifying.

        Skipping verification is how an agent orders an item the market event
        just pushed over budget, so the clock never shortens the
        commit / react / recover / order sequence itself.
        """

        return self.time_used > _DEADLINE_S * _FORCE_ORDER_AT

    def eligible_pairs(self, allow_unproven: bool = False) -> list[tuple[Candidate, dict[str, Any]]]:
        out = []
        for candidate in self.pool.values():
            if candidate.rejected or not candidate.grounded:
                continue
            if allow_unproven:
                # keep candidates whose spec is merely unstated, never one that
                # the catalog contradicts
                if spec_outcome(candidate, self.req.specs) == 0:
                    continue
                probe = Requirements()
                probe.__dict__.update(self.req.__dict__)
                probe.specs = []
                if not hard_gate_ok(candidate, probe):
                    continue
            elif not hard_gate_ok(candidate, self.req):
                continue
            variant = preferred_variant(candidate, self.req)
            if variant is None:
                continue
            if (candidate.product_id, _str_or_blank(variant.get("sku"))) in self.dead:
                continue
            out.append((candidate, variant))
        return out

    def verify_stock(self, product_id: str, sku: str) -> dict[str, Any]:
        observation = self.env.call_tool(
            "inspect_stock", {"product_id": product_id, "sku": sku}
        )
        return observation if isinstance(observation, dict) else {}

    def order_now(self) -> bool:
        if not self.committed or self.env.done:
            return False
        if self.committed in self.dead:
            # the cart holds an item the event invalidated; do not order it
            self.committed = None
            return False
        product_id, sku = self.committed
        if not self.confirm_commitment():
            return False
        args: dict[str, Any] = {"product_id": product_id, "sku": sku}
        payload = self.justification_claims()
        if payload:
            args["justification"] = payload
        observation = self.env.call_tool("place_test_order", args)
        if observation and (
            observation.get("status") == "placed" or observation.get("order")
        ):
            return True
        # a rejected order is usually a stale price/stock read
        if observation and not self.env.done and not self.must_close(2):
            fresh = self.verify_stock(product_id, sku)
            price = _to_float(fresh.get("price"))
            affordable = (
                self.req.budget is None
                or price is None
                or price <= self.req.budget + 1e-9
            )
            if fresh.get("in_stock") is True and affordable:
                observation = self.env.call_tool("place_test_order", args)
                return bool(observation and observation.get("status") == "placed")
            self.dead.add((product_id, sku))
            self.committed = None
        return False

    def category_query_forms(self) -> list[str]:
        category = self.req.category or ""
        if not category:
            return []
        head = category.split()[-1]
        return list(dict.fromkeys([category, category.replace(head, plural_form(head)), singular_form(category)]))

    def end_dialogue(self) -> list[dict[str, Any]]:
        if not self.env.ordered:
            try:
                self.closing_order()
            except Exception as exc:
                self.log.append("close_out failed: %s" % type(exc).__name__)
        self.dialogue.append(
            {"role": "assistant", "content": "summary: %s" % " | ".join(self.log[-25:])}
        )
        return self.dialogue

    def transcript_checksum(self) -> str:
        checksum = RollingChecksum()
        checksum.feed_turns(self.dialogue)
        return checksum.as_hex()

    def choose_relaxed(self) -> tuple[Candidate, dict[str, Any]] | None:
        """Last-resort widening: drop soft constraints, never hard ones."""

        saved_specs = list(self.req.specs)
        saved_brand, saved_soft = self.req.brand, self.req.brand_is_soft
        if self.req.brand_is_soft:
            self.req.brand = None
        self.req.preferred_color = None
        pick = self.choose_best()
        if pick is None and saved_specs:
            # a spec the catalog never states should not block a purchase, but a
            # listing that contradicts it must still be excluded
            pairs = self.eligible_pairs(allow_unproven=True)
            if pairs:
                self.score_prefs(pairs)
                ranked = self.rank_by_priority(pairs) if self.req.axes else self.rank_candidates(pairs)
                pick = ranked[0]
        self.req.specs = saved_specs
        self.req.brand, self.req.brand_is_soft = saved_brand, saved_soft
        return pick

    def hunt_model_code(self) -> None:
        """Chase an explicitly named model code before committing to anything.

        The code usually lives in the catalog's model fact rather than the
        title, so the bare token is the query that reaches it.
        """

        model = self.req.preferred_model
        if not model or self.env.done or self.must_close(8) or self.freeze_improvements:
            return
        for pair in self.eligible_pairs():
            if self.relax_level(*pair) == 0:
                return  # a full match is already in hand
        queries = [model]
        if self.req.preferred_color:
            queries.append("%s %s" % (model, self.req.preferred_color))
        if self.req.category:
            queries.append("%s %s" % (model, self.req.category))
        queries = [q for q in queries if q.lower() not in {x.lower() for x in self.queries_run}]
        if not queries:
            return
        self.perform_search(queries[:2])
        self.read_shopper_turns()
        shortlist = sorted(self.pre_screen(), key=self.coarse_score, reverse=True)
        self.ground_candidates([c for c in shortlist if not c.grounded][:5])

    def rank_by_priority(self, pairs):
        """Lexicographic order over the shopper's stated tradeoff axes."""

        axes = self.req.axes or ["price"]

        def key(item):
            candidate, variant = item
            out = []
            for axis in axes:
                value = axis_quantity(candidate, variant, axis)
                if value is None:
                    out.append(float("inf"))
                elif _AXIS_DIRECTION.get(axis, "max") == "min":
                    out.append(value)
                else:
                    out.append(-value)
            return tuple(out)

        return sorted(pairs, key=key)

    def pool_table(self, limit: int = 8) -> str:
        rows = []
        for candidate in list(self.pool.values())[:limit]:
            rows.append({
                "id": candidate.product_id,
                "price": candidate.price,
                "grounded": "y" if candidate.grounded else "n",
                "title": candidate.title[:32],
            })
        return fixed_width_table(rows, ["id", "price", "grounded", "title"]) if rows else ""

    def descriptor_rate(self, candidate: Candidate) -> float:
        words = self.descriptor_words()
        if not words:
            return 0.0
        blob = _lcase(candidate.title + " " + candidate.searchable_text())
        return sum(1 for word in words if word in blob) / float(len(words))

    def spec_search_terms(self) -> list[str]:
        """Searches aimed at the exact attribute value the task gates on."""

        parts: list[str] = []
        category = self.req.category or ""
        brand = self.req.brand or ""
        for spec in self.req.specs[:2]:
            value = _str_or_blank(spec.get("value"))
            if not value:
                continue
            parts.append(" ".join(p for p in (brand, category, value) if p))
            parts.append(" ".join(p for p in (category, value) if p))
            if _lcase(spec.get("field")) == "resolution":
                classes = {"hd": "720p hd", "fhd": "1080p full hd", "qhd": "1440p 2k", "4k": "4k uhd"}
                klass = desired_screen_class(value)
                if klass:
                    parts.append(" ".join(p for p in (category, classes[klass]) if p))
        tried = {q.lower() for q in self.queries_run}
        out: list[str] = []
        for query in parts:
            query = re.sub(r"\s+", " ", query).strip()
            if query and query.lower() not in tried and query.lower() not in {o.lower() for o in out}:
                out.append(query)
        return out[:2]

    def confirm_commitment(self) -> bool:
        """Re-check price and stock on the exact variant before ordering.

        An order outside budget or out of stock scores zero on a hard check, so
        this verification happens on every path, however tight the clock.
        """

        if not self.committed:
            return False
        product_id, sku = self.committed
        if self.verified == self.committed:
            return True
        if self.env.turns_remaining < 2:
            return True  # no room to check; ordering still beats not ordering
        fresh = self.verify_stock(product_id, sku)
        if not fresh:
            return True
        price = _to_float(fresh.get("price"))
        if price is not None:
            self.commit_price = price
        if self.req.require_in_stock and fresh.get("in_stock") is False:
            self.log.append("pre-order check: out of stock")
            self.dead.add(self.committed)
            self.committed = None
            return False
        if price is not None and self.req.budget is not None and price > self.req.budget + 1e-9:
            self.log.append(
                "pre-order check: %.2f over budget %.2f" % (price, self.req.budget)
            )
            self.dead.add(self.committed)
            self.committed = None
            return False
        self.verified = (product_id, sku)
        return True

    def pool_price_outliers(self) -> list[str]:
        prices = {c.product_id: c.price for c in self.pool.values() if c.price is not None}
        return price_outliers(prices)

    def acknowledge_event(self, kind: str) -> None:
        """No-op by design.

        The event is closed by committing to the right replacement, not by
        talking about it, and each message risks a simulator failure that would
        invalidate the evaluation.
        """

        self.log.append("event noticed: %s" % (kind or "unknown"))

    def pref_fraction(self, candidate: Candidate) -> float:
        if not self.req.soft_prefs:
            return 0.0
        return self.pref_evidence_count(candidate) / float(len(self.req.soft_prefs))

    def axis_direction_view(self) -> dict[str, str]:
        view = {}
        for axis in self.req.axes or []:
            view[axis] = _AXIS_DIRECTION.get(axis, "max")
        return view

    def pre_screen(self) -> list[Candidate]:
        req = self.req
        out = []
        for candidate in self.pool.values():
            if candidate.rejected:
                continue
            if req.phrase_terms and not title_contains_phrase(candidate.title, req.phrase_terms):
                continue
            if req.budget is not None and candidate.price is not None:
                if candidate.price > req.budget + 1e-9 and not candidate.variants:
                    continue
            if req.require_in_stock and candidate.in_stock is False and not candidate.variants:
                continue
            if req.device_required and accessory_like(candidate.title):
                continue
            out.append(candidate)
        return out

    def listing_clusters(self, threshold: float = 0.93) -> list[list[str]]:
        """Group pool entries whose titles are near-identical listings."""

        forest = DisjointSet()
        items = list(self.pool.values())
        for i, left in enumerate(items):
            forest.find(left.product_id)
            for right in items[i + 1 :]:
                if jaro_winkler(left.title[:60], right.title[:60]) >= threshold:
                    forest.union(left.product_id, right.product_id)
        return [group for group in forest.groups() if len(group) > 1]

    def fallback_order(self) -> None:
        """Never end a run without a terminal action when one is possible."""

        if self.env.done or self.env.ordered:
            return
        if self.committed and self.committed not in self.dead:
            if self.order_now():
                return
        if self.committed and self.committed in self.dead:
            if self.env.turns_remaining >= 4 and self.recover_commitment():
                if self.order_now():
                    return
            self.committed = None
        pairs = self.eligible_pairs()
        if not pairs:
            shortlist = sorted(self.pre_screen(), key=self.coarse_score, reverse=True)
            if shortlist and not self.env.done:
                self.ground_candidates(shortlist[:5])
                pairs = self.eligible_pairs()
        if not pairs:
            # nothing proves the spec: accept listings that do not contradict it
            pairs = self.eligible_pairs(allow_unproven=True)
        if not pairs:
            # last resort: anything affordable and in stock that we have seen
            ranked = sorted(
                self.pool.values(),
                key=lambda c: (-spec_outcome(c, self.req.specs), -self.coarse_score(c)),
            )
            for candidate in ranked:
                variant = preferred_variant(candidate, self.req)
                if variant is not None:
                    pairs = [(candidate, variant)]
                    break
        if not pairs or self.env.done:
            return
        candidate, variant = self.rank_candidates(pairs)[0]
        if self.commit_to_cart(candidate, variant):
            if self.env.turns_remaining >= 2:
                state = self.react_to_market()
                if state == "replace" and self.env.turns_remaining >= 4:
                    self.recover_commitment()
            self.order_now()

    def model_query_forms(self) -> list[str]:
        forms = hyphen_variants(self.req.preferred_model)
        tried = {q.lower() for q in self.queries_run}
        return [form for form in forms if form not in tried]

    def descriptor_words(self) -> list[str]:
        """Distinctive product-type words in the ask ("gaming", "foldable").

        They are not the category and not the spec, but the acceptance contract
        still counts them: a plain monitor is not a gaming monitor.
        """

        drop = set(_STOPWORDS)
        for source in (self.req.category or "", self.req.brand or ""):
            drop.update(title_terms(source))
            drop.update(t.rstrip("s") for t in title_terms(source))
        for spec in self.req.specs:
            drop.update(title_terms(_str_or_blank(spec.get("value"))))
        for pref in self.req.soft_prefs:
            drop.update(title_terms(pref))
        out: list[str] = []
        for token in title_terms(self.request):
            if token in drop or token.rstrip("s") in drop:
                continue
            if len(token) < 4 or token.isdigit() or any(ch.isdigit() for ch in token):
                continue
            if token not in out:
                out.append(token)
        return out[:4]

    def merge_payload(self, payload: dict[str, Any]) -> None:
        req = self.req
        category = _str_or_blank(payload.get("category")).strip()
        if category:
            req.category = category
        budget = _to_float(payload.get("budget"))
        if budget is not None and (req.budget is None or budget < req.budget):
            req.budget = budget
        currency = _str_or_blank(payload.get("currency")).strip()
        if currency:
            req.currency = currency
        brand = _str_or_blank(payload.get("brand")).strip()
        if brand:
            req.brand = brand
            req.brand_is_soft = bool(payload.get("brand_is_soft")) or self.flags.soft_brand
        specs = payload.get("specs")
        if isinstance(specs, list):
            merged = {}
            for spec in req.specs:
                merged[_lcase(spec.get("field"))] = spec
            for spec in specs:
                if isinstance(spec, dict) and spec.get("field") and spec.get("value"):
                    merged[_lcase(spec["field"])] = {
                        "field": _str_or_blank(spec["field"]),
                        "value": _str_or_blank(spec["value"]),
                    }
            req.specs = list(merged.values())
        terms = payload.get("phrase_terms")
        if isinstance(terms, list) and terms and self.flags.phrase_rule:
            req.phrase_terms = [_str_or_blank(t) for t in terms if _str_or_blank(t)]
        if isinstance(payload.get("device_required"), bool):
            req.device_required = payload["device_required"]
        if isinstance(payload.get("smartphone_required"), bool):
            req.smartphone_required = payload["smartphone_required"]
        model = _str_or_blank(payload.get("preferred_model")).strip()
        if model:
            req.preferred_model = model
        color = _str_or_blank(payload.get("preferred_color")).strip()
        if color:
            req.preferred_color = color
        prefs = payload.get("soft_prefs")
        if isinstance(prefs, list):
            for pref in prefs:
                text = _str_or_blank(pref).strip()
                if text and text not in req.soft_prefs:
                    req.soft_prefs.append(text)
        field = _str_or_blank(payload.get("requested_spec_field")).strip()
        if field:
            req.requested_spec_field = field
        rule = _str_or_blank(payload.get("ranking_rule")).strip()
        if rule:
            req.ranking_rule = rule

    def perform_search(self, queries: list[str]) -> int:
        """Run up to max_calls searches in one turn; returns new candidates."""

        if not queries or self.env.done:
            return 0
        floor = 0 if self.committed or self.env.event_seen else self.search_reserve
        if self.search_budget - floor <= 0:
            return 0
        batch = queries[: min(2, self.env.max_calls, max(1, self.search_budget - floor))]
        self.search_budget -= len(batch)
        actions: list[tuple[str, dict[str, Any]]] = []
        for query in batch:
            actions.append(
                (
                    "search",
                    {
                        "query": query[:200],
                        "k": 10,
                        **({"in_stock": True} if self.req.require_in_stock else {}),
                    },
                )
            )
            self.queries_run.append(query)
        before = len(self.pool)
        for observation in self.env.execute_turn(actions):
            self.ingest_results(observation.get("results"))
        return len(self.pool) - before

    def offline_queries(self) -> list[str]:
        """Reformulations that need no model: reorder, split, singularise."""

        keywords = self.salient_words()
        tried = {q.lower() for q in self.queries_run}
        options: list[str] = []
        if keywords:
            options.append(" ".join(keywords[:3]))
            options.append(" ".join(keywords[1:4]))
            options.append(keywords[0])
            options.append(" ".join(keywords[-3:]))
            options.append(" ".join(k.rstrip("s") for k in keywords[:3]))
        if self.req.category:
            for pref in self.req.soft_prefs[:2]:
                options.append("%s %s" % (self.req.category, pref))
            if self.req.brand:
                options.append("%s %s" % (self.req.brand, self.req.category))
        out: list[str] = []
        for option in options:
            option = re.sub(r"\s+", " ", option).strip()
            if option and option.lower() not in tried and option.lower() not in {
                o.lower() for o in out
            }:
                out.append(option)
        return out[:3]

    def run_episode(self) -> list[dict[str, Any]]:
        try:
            self.parse_request()
            self.clarify_request()
            if self.req.listed:
                # the shopper named the exact variants: no search needed
                if self.solve_listed_variants():
                    return self.end_dialogue()
                if self.env.ordered:
                    return self.end_dialogue()
            self.survey_catalog()
            pick = self.choose_best()
            if pick is None:
                pick = self.choose_relaxed()
            if pick is not None and not self.env.done:
                if not self.commit_to_cart(*pick):
                    alternatives = [
                        p
                        for p in self.rank_candidates(self.eligible_pairs())
                        if (p[0].product_id, _str_or_blank(p[1].get("sku"))) not in self.dead
                    ]
                    if alternatives:
                        self.commit_to_cart(*alternatives[0])
            if self.committed and not self.env.done:
                state = self.react_to_market()
                if state != "ok":
                    kind = _lcase((self.env.pending_event or {}).get("kind")) or "stockout"
                    self.acknowledge_event(kind)
                    if not self.recover_commitment():
                        self.log.append("recovery ladder exhausted")
                self.finalize_order()
            if not self.env.ordered:
                self.fallback_order()
            if not self.env.ordered:
                self.closing_order()
        except Exception as exc:  # a raised agent = agent_error = zero
            self.log.append("fatal: %s: %s" % (type(exc).__name__, exc))
            try:
                self.fallback_order()
                if not self.env.ordered:
                    self.closing_order()
            except Exception as inner:
                self.log.append("fallback failed: %s" % type(inner).__name__)
        return self.end_dialogue()


def format_money(amount: float | None, currency: str = "PHP") -> str:
    if amount is None:
        return "n/a"
    whole, fraction = divmod(round(abs(amount), 2), 1)
    grouped = "{:,}".format(int(whole))
    sign = "-" if amount < 0 else ""
    return "%s%s %s.%02d" % (sign, currency, grouped, int(round(fraction * 100)))


def stable_partition(items: list[Any], keep) -> tuple[list[Any], list[Any]]:
    kept: list[Any] = []
    rest: list[Any] = []
    for item in items:
        (kept if keep(item) else rest).append(item)
    return kept, rest


def endurance_rating(candidate: "Candidate") -> float:
    """Largest stated playtime in hours, 0.0 when the listing never says."""

    low = _lcase(candidate.searchable_text())
    values = [
        float(m.group(1))
        for m in re.finditer(r"(\d{1,3})\s*(?:hours|hour|hrs|hr)\b", low)
    ]
    return max(values) if values else 0.0


def jaro_winkler(left: str, right: str, prefix_scale: float = 0.1) -> float:
    """Jaro-Winkler similarity, forgiving of transposed letters in brand names."""

    a, b = _lcase(left), _lcase(right)
    if a == b:
        return 1.0 if a else 0.0
    window = max(0, max(len(a), len(b)) // 2 - 1)
    a_hits = [False] * len(a)
    b_hits = [False] * len(b)
    matches = 0
    for i, char in enumerate(a):
        for j in range(max(0, i - window), min(len(b), i + window + 1)):
            if not b_hits[j] and b[j] == char:
                a_hits[i] = b_hits[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    a_seq = [c for c, hit in zip(a, a_hits) if hit]
    b_seq = [c for c, hit in zip(b, b_hits) if hit]
    transpositions = sum(1 for x, y in zip(a_seq, b_seq) if x != y) / 2.0
    jaro = (matches / len(a) + matches / len(b) + (matches - transpositions) / matches) / 3.0
    prefix = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        prefix += 1
    return round(jaro + prefix * prefix_scale * (1.0 - jaro), 4)


class BloomSeen:
    """Probabilistic set of queries already issued; no false negatives."""

    def __init__(self, bits: int = 1024, hashes: int = 3) -> None:
        self.bits = max(64, bits)
        self.hashes = max(1, hashes)
        self.field = bytearray(self.bits // 8 + 1)

    def _positions(self, text: str) -> list[int]:
        key = re.sub(r"\s+", " ", _lcase(text)).strip()
        return [stable_bucket("%d:%s" % (seed, key), self.bits) for seed in range(self.hashes)]

    def remember_query(self, text: str) -> None:
        for bit in self._positions(text):
            self.field[bit // 8] |= 1 << (bit % 8)

    def probably_seen(self, text: str) -> bool:
        return all(self.field[bit // 8] & (1 << (bit % 8)) for bit in self._positions(text))


class Client:
    """Thin wrapper over the sandbox proxy.

    Model calls get a short timeout of their own: a hung completion is the
    difference between a scored order and a killed run.
    """

    def __init__(self) -> None:
        self._proxy = ProxyClient(timeout=110, max_retries=2)
        try:
            self._fast = ProxyClient(timeout=int(_LLM_TIMEOUT), max_retries=1)
        except Exception:
            self._fast = self._proxy

    def relay(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        client = self._fast if path.startswith("/inference") else self._proxy
        return client.post(path, json_data=payload)

    def transport_summary(self) -> dict[str, str]:
        separate = self._fast is not self._proxy
        return {"inference": "dedicated" if separate else "shared", "environment": "standard"}


def harmonic_mean(values: list[float]) -> float | None:
    positive = [v for v in values if v and v > 0]
    if not positive:
        return None
    return round(len(positive) / sum(1.0 / v for v in positive), 4)


def primary_material(text: str) -> str | None:
    """The material mentioned earliest wins; later ones are usually trim."""

    low = _lcase(text)
    materials = ("stainless steel", "aluminium", "aluminum", "silicone", "cotton", "leather",
                 "polyester", "nylon", "glass", "ceramic", "bamboo", "wood", "plastic", "steel")
    positions = [(low.find(material), material) for material in materials if material in low]
    return min(positions)[1] if positions else None


def category_passes(candidate: Candidate, wanted: str | None) -> bool:
    if not wanted:
        return True
    want = re.sub(r"[^a-z0-9 ]", " ", wanted.lower()).strip()
    want_tokens = [t.rstrip("s") for t in want.split() if t]
    if not want_tokens:
        return True
    if not candidate.category_path:
        return True  # not yet grounded; judged again after view/compare
    path = " ".join(candidate.category_path).lower()
    path_tokens = [t.rstrip("s") for t in re.split(r"[^a-z0-9]+", path) if t]
    return all(token in path_tokens for token in want_tokens)


def apparel_size(text: str) -> str | None:
    tokens = set(re.findall(r"[a-z0-9]+", _lcase(text)))
    aliases = {
        "3XL": {"xxxl", "3xl"}, "2XL": {"xxl", "2xl"}, "XL": {"xl"},
        "L": {"large"}, "M": {"medium"}, "S": {"small"}, "XS": {"xs"},
    }
    hits = [label for label, spellings in aliases.items() if tokens & spellings]
    return hits[0] if hits else None


class AbSplitter:
    """Stable A/B assignment of tasks to strategy variants for offline trials."""

    def __init__(self, arms: dict[str, float]) -> None:
        self.arms = dict(arms)
        self.assigned: dict[str, str] = {}

    def arm_for(self, task_key: str) -> str | None:
        if task_key not in self.assigned:
            choice = weighted_pick(list(self.arms), list(self.arms.values()), task_key)
            if choice is None:
                return None
            self.assigned[task_key] = choice
        return self.assigned[task_key]

    def tally(self) -> dict[str, int]:
        counts = {arm: 0 for arm in self.arms}
        for arm in self.assigned.values():
            counts[arm] += 1
        return counts


def device_passes(candidate: Candidate, requirements: Requirements) -> bool:
    if not requirements.device_required:
        return True
    if accessory_like(candidate.title):
        return False
    if not requirements.smartphone_required:
        return True
    blob = _lcase(candidate.searchable_text())
    if any(bad in blob for bad in _NON_SMARTPHONE) and "smartphone" not in blob:
        return False
    if "smartphone" in _lcase(candidate.title):
        return True
    device_type = _lcase(candidate.facts.get("phone_type")) + " " + _lcase(
        candidate.facts.get("device_type")
    )
    if "smartphone" in device_type:
        return True
    os_text = " ".join(
        _lcase(candidate.facts.get(key))
        for key in ("device_operating_system", "operating_system", "os")
    )
    return any(name in os_text for name in _MOBILE_OS)


def unit_price(price: float | None, amount: float | None, per: float = 100.0) -> float | None:
    """Price per ``per`` units (per 100 ml, per 100 g) for cross-size comparison."""

    try:
        ratio = per / float(amount)
    except (TypeError, ZeroDivisionError):
        return None
    return None if price is None or ratio <= 0 else round(price * ratio, 3)


class DisjointSet:
    """Union-find, used to cluster listings that are the same product."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        self.rank.setdefault(key, 0)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, left: str, right: str) -> str:
        a, b = self.find(left), self.find(right)
        if a == b:
            return a
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        return a

    def groups(self) -> list[list[str]]:
        clusters: dict[str, list[str]] = {}
        for key in list(self.parent):
            clusters.setdefault(self.find(key), []).append(key)
        return sorted((sorted(members) for members in clusters.values()), key=lambda g: (-len(g), g[0]))


def rules_only(query: str) -> str:
    cut = query.find("Task rules:")
    return query[cut:] if cut > 0 else query


def ean13_valid(code: str) -> bool:
    """GS1 check-digit test for EAN-13 / UPC-A barcodes found in catalog facts."""

    digits = [int(ch) for ch in _str_or_blank(code) if ch.isdigit()]
    if len(digits) == 12:
        digits = [0] + digits
    if len(digits) != 13:
        return False
    weighted = sum(d * (3 if i % 2 else 1) for i, d in enumerate(digits[:12]))
    return (10 - weighted % 10) % 10 == digits[12]


def screen_class_for(width: float, height: float) -> str | None:
    """Highest pixel class whose BOTH thresholds are met, orientation-free."""

    long_side, short_side = max(width, height), min(width, height)
    if long_side >= 3840 and short_side >= 2160:
        return "4k"
    if long_side >= 2560 and short_side >= 1440:
        return "qhd"
    if long_side >= 1920 and short_side >= 1080:
        return "fhd"
    if long_side >= 1280 and short_side >= 720:
        return "hd"
    return None


def softmax(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    heat = max(temperature, 1e-6)
    top = max(scores.values())
    weights = {key: 2.718281828459045 ** ((value - top) / heat) for key, value in scores.items()}
    total = sum(weights.values())
    return {key: round(weight / total, 4) for key, weight in weights.items()}


def volume_millilitres(text: str) -> float | None:
    """Pack volume in ml from "500ml", "1.5 L", "12 fl oz"."""

    factors = {"ml": 1.0, "l": 1000.0, "litre": 1000.0, "liter": 1000.0, "fl oz": 29.5735, "floz": 29.5735}
    low = _lcase(text).replace("fl. oz", "fl oz")
    readings = [
        float(amount) * factors[unit.strip()]
        for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(fl oz|floz|ml|litre|liter|l)\b", low)
    ]
    return round(max(readings), 1) if readings else None


def screen_classes_stated(text: str) -> list[str]:
    """Every pixel class the text asserts, from dimensions AND from names.

    A listing that states both "720 x 1600" and "1080p" asserts two different
    classes; the rules call that unresolved, so both have to be collected.
    """

    low = _lcase(text)
    found: list[str] = []
    for match in re.finditer(r"(\d{3,5})\s*[x*×]\s*(\d{3,5})", low):
        klass = screen_class_for(float(match.group(1)), float(match.group(2)))
        if klass and klass not in found:
            found.append(klass)
    stripped = re.sub(r"\d{3,5}\s*[x*×]\s*\d{3,5}", " ", low)
    # longest name first, blanking each match so "full hd" is not also read
    # as a bare "hd"
    for token, klass in sorted(_NAMED_CLASSES, key=lambda pair: -len(pair[0])):
        pattern = r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(token)
        if re.search(pattern, stripped):
            stripped = re.sub(pattern, " ", stripped)
            if klass not in found:
                found.append(klass)
    if re.search(r"(?<![a-z0-9+])hd(?![a-z0-9+])", stripped) and "hd" not in found:
        found.append("hd")
    return found


class ResourcePool:
    """Named budgets with reservations, so recovery turns are held back up front."""

    def __init__(self, **limits: float) -> None:
        self.limits = {name: float(value) for name, value in limits.items()}
        self.spent_amounts = {name: 0.0 for name in limits}
        self.held = {name: 0.0 for name in limits}

    def available(self, name: str) -> float:
        return self.limits.get(name, 0.0) - self.spent_amounts.get(name, 0.0) - self.held.get(name, 0.0)

    def hold(self, name: str, amount: float) -> bool:
        if amount > self.available(name):
            return False
        self.held[name] = self.held.get(name, 0.0) + amount
        return True

    def release(self, name: str, amount: float | None = None) -> None:
        current = self.held.get(name, 0.0)
        self.held[name] = 0.0 if amount is None else max(0.0, current - amount)

    def charge_to(self, name: str, amount: float, from_hold: bool = False) -> None:
        if from_hold:
            self.release(name, amount)
        self.spent_amounts[name] = self.spent_amounts.get(name, 0.0) + amount


def spec_satisfied_by(candidate: Candidate, spec: dict[str, str]) -> bool | None:
    """True / False, or None when the evidence does not resolve the spec."""

    field = _lcase(spec.get("field"))
    value = _str_or_blank(spec.get("value"))
    canonical = fact_for_field(candidate, field)
    # the catalog fact is authoritative; a title number may describe another
    # model in a multi-model listing
    blob = canonical if canonical else candidate.searchable_text()
    low_blob = _lcase(blob)

    if "screen" in field or "size" in field or "inch" in field:
        wanted = parse_numbers(value)
        if not wanted:
            return None
        target = wanted[0]
        found = [
            float(m.group(1))
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:inch|inches|in\b|\")", low_blob)
        ]
        if not found:
            return None
        # the gate compares against the catalog's own value, so 6.6 does not
        # satisfy a 6.64 requirement
        return any(abs(size - target) < 0.005 for size in found)

    if "ram" in field or "memory" in field:
        wanted = parse_numbers(value)
        if not wanted:
            return None
        target = wanted[0]
        found = [
            float(m.group(1))
            for m in re.finditer(r"(\d+)\s*gb\s*(?:ram)?", low_blob)
        ]
        if not found:
            return None
        return any(abs(size - target) < 0.01 for size in found)

    if "refresh" in field or "hz" in field:
        wanted = parse_numbers(value)
        if not wanted:
            return None
        target = wanted[0]
        found = [float(m.group(1)) for m in re.finditer(r"(\d{2,3})\s*hz", low_blob)]
        if not found:
            return None
        return any(abs(rate - target) < 0.01 for rate in found)

    if "resolution" in field or "display" in field:
        want = desired_screen_class(value)
        if not want:
            return None
        classes = screen_classes_stated(blob)
        if not classes:
            named = desired_screen_class(blob)
            return None if named is None else named == want
        if len(set(classes)) > 1:
            return None  # conflicting classes -> unresolved
        return classes[0] == want

    if not value:
        return None
    return value.lower() in low_blob


def wrap_text(text: str, width: int = 72) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in _str_or_blank(text).split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = word if not current else current + " " + word
    if current:
        lines.append(current)
    return lines


def ethics_labels(text: str) -> set[str]:
    low = _lcase(text)
    labels = set()
    if "vegan" in low:
        labels.add("vegan")
    if "cruelty" in low and "free" in low:
        labels.add("cruelty_free")
    if "organic" in low:
        labels.add("organic")
    if "recycl" in low:
        labels.add("recyclable")
    return labels


def title_contains_phrase(title: str, terms: list[str]) -> bool:
    if not terms:
        return True
    toks = set(title_terms(title))
    for term in terms:
        for part in title_terms(term):
            if part not in toks:
                return False
    return True


class CartModel:
    """Local mirror of the cart with an undo stack, to plan swaps before acting."""

    def __init__(self) -> None:
        self.lines: dict[tuple[str, str], int] = {}
        self.undo: list[tuple[str, tuple[str, str], int]] = []

    def put(self, product_id: str, sku: str, quantity: int = 1) -> None:
        key = (product_id, sku)
        self.undo.append(("put", key, self.lines.get(key, 0)))
        self.lines[key] = self.lines.get(key, 0) + max(1, quantity)

    def take_out(self, product_id: str, sku: str) -> bool:
        key = (product_id, sku)
        if key not in self.lines:
            return False
        self.undo.append(("take_out", key, self.lines.pop(key)))
        return True

    def revert(self) -> bool:
        if not self.undo:
            return False
        _action, key, before = self.undo.pop()
        if before:
            self.lines[key] = before
        else:
            self.lines.pop(key, None)
        return True

    def keys(self) -> list[tuple[str, str]]:
        return sorted(self.lines)


def split_bundle_price(total: float | None, parts: dict[str, float]) -> dict[str, float]:
    """Allocate a bundle price across its items in proportion to list prices."""

    if total is None or not parts:
        return {}
    base = sum(max(0.0, value) for value in parts.values())
    if base <= 0:
        return {name: round(total / len(parts), 2) for name in parts}
    allocated: dict[str, float] = {}
    remainder = round(total, 2)
    names = sorted(parts)
    for name in names[:-1]:
        amount = round(total * max(0.0, parts[name]) / base, 2)
        allocated[name] = amount
        remainder = round(remainder - amount, 2)
    allocated[names[-1]] = remainder
    return allocated


def specs_mentioned(text: str) -> list[dict[str, str]]:
    """Firm hardware specs stated anywhere in shopper text, without a model."""

    specs: list[dict[str, str]] = []
    match = re.search(r"(\d{1,2}(?:\.\d+)?)\s*-?\s*(?:inch|inches|in\b|\")", text, re.I)
    if match:
        specs.append({"field": "screen_size", "value": match.group(1) + " inch"})
    match = re.search(r"(\d{1,3})\s*gb\s*(?:of\s*)?ram", text, re.I)
    if match:
        specs.append({"field": "ram", "value": match.group(1) + "GB"})
    match = re.search(r"(\d{2,3})\s*hz", text, re.I)
    if match:
        specs.append({"field": "refresh_rate", "value": match.group(1) + "Hz"})
    for token, value in (
        ("4k", "4K"), ("uhd", "4K"), ("qhd", "QHD"), ("full hd", "Full HD"),
        ("fhd", "Full HD"), ("1080p", "Full HD"), ("1440p", "QHD"), ("720p", "HD"),
    ):
        if re.search(r"\b%s\b" % re.escape(token), text, re.I):
            if token == "qhd" and "QHD" not in text:
                continue
            specs.append({"field": "resolution", "value": value})
            break
    return specs


class EventBus:
    """Tiny publish/subscribe hub for phase notifications."""

    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}
        self.delivered = 0

    def subscribe(self, topic: str, handler) -> None:
        self.handlers.setdefault(topic, []).append(handler)

    def publish(self, topic: str, payload: Any = None) -> int:
        count = 0
        for handler in list(self.handlers.get(topic, ())):
            try:
                handler(payload)
                count += 1
            except Exception:
                continue
        self.delivered += count
        return count


def rotate_words(query: str, limit: int = 3) -> list[str]:
    """Word-order rotations of a query; some catalog indexes weight the head term."""

    words = _str_or_blank(query).split()
    rotations: list[str] = []
    for shift in range(1, min(limit, len(words)) + 1):
        rotated = words[shift:] + words[:shift]
        text = " ".join(rotated)
        if text != query and text not in rotations:
            rotations.append(text)
    return rotations


def fact_for_field(candidate: "Candidate", field: str) -> str | None:
    """Catalog-stated value for a field, preferring facts over the title."""

    facts = candidate.facts or {}
    wanted = {
        "screen_size": ("screen_size", "screen size", "size", "display_size", "display size"),
        "ram": ("ram", "memory"),
        "refresh_rate": ("refresh_rate", "refresh rate", "refresh"),
        "resolution": ("resolution", "display", "screen_resolution"),
        "model": ("model",),
        "color": ("color", "colour"),
    }.get(field, (field,))
    for key, value in facts.items():
        if _lcase(key) in wanted:
            text = _str_or_blank(value).strip()
            if text:
                return text
    return None


def shelf_life_months(text: str) -> int | None:
    low = _lcase(text)
    match = re.search(r"(?:shelf life|best before|expires? in|use within)\D{0,12}(\d{1,3})\s*(month|year|day)s?", low)
    if not match:
        return None
    count, unit = int(match.group(1)), match.group(2)
    return {"month": count, "year": count * 12, "day": max(1, count // 30)}[unit]


def review_total(text: str) -> int | None:
    for chunk in re.split(r"[()\[\]|]", _lcase(text)):
        if "review" in chunk or "rating" in chunk:
            head = chunk.split("review")[0].split("rating")[0]
            digits = re.sub(r"[^\d]", "", head)
            if digits:
                return int(digits)
    return None


def accessory_like(title: str) -> bool:
    low = _lcase(title)
    return any(word in low for word in _ACCESSORY)


def stable_bucket(key: str, buckets: int = 10) -> int:
    """Deterministic bucket for a string (no reliance on hash randomisation)."""

    total = 0
    for index, char in enumerate(_str_or_blank(key)):
        total = (total * 31 + ord(char) * (index + 1)) % 1000003
    return total % max(1, buckets)


def acronym(phrase: str) -> str:
    words = [w for w in re.split(r"[\s\-_/]+", _str_or_blank(phrase)) if w and w.lower() not in _STOPWORDS]
    return "".join(w[0].upper() for w in words)


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

_EXTRACT_PROMPT = """Shopper request:
%s

Task rules (abridged):
%s

Extract the requirements as JSON with exactly these keys:
{"category": string|null,            // catalog category actually required
 "budget": number|null,              // maximum price, numeric only
 "currency": string|null,
 "brand": string|null,               // named brand, if any
 "brand_is_soft": boolean,           // true when the brand is only preferred
 "specs": [{"field": "the attribute name, e.g. screen_size, ram, resolution,
                      refresh_rate, volume, weight, capacity, spf, material",
            "value": string}],       // firm product specs only, any category
 "phrase_terms": [string],           // exact words that must appear in the title
 "device_required": boolean,         // true when an actual device is wanted
 "smartphone_required": boolean,
 "preferred_model": string|null,
 "preferred_color": string|null,
 "soft_prefs": [string],             // use-case preferences, one short phrase each
 "requested_spec_field": string|null,// the spec the shopper named, for claims
 "ranking_rule": string|null}        // any stated ordering rule
Use null / [] when something is not stated. Never invent values. JSON only."""

_UPDATE_PROMPT = """Current requirements: %s

The shopper just said: "%s"

Return JSON with ONLY the fields that changed, using the same schema as before
(category, budget, currency, brand, brand_is_soft, specs, phrase_terms,
device_required, smartphone_required, preferred_model, preferred_color,
soft_prefs, requested_spec_field, ranking_rule). Requirements the shopper did
not mention must be omitted, not reset. JSON only."""

_QUERY_PROMPT = """Shopper request: %s
Requirements: %s
Queries already tried: %s
Still needed: %s

Write up to 3 NEW full-text catalog queries that would reach DIFFERENT products
than the ones already tried. Vary the wording: synonyms, product-type words,
brand or spec terms, use-case words. Keep each query under 8 words. Do not
repeat a previous query. Reply as JSON: {"queries": ["...", "..."]}"""

_SCORE_PROMPT = """Shopper request: %s
Stated soft preferences: %s
Stated ranking rule: %s

Candidates (raw catalog facts):
%s

Score each candidate from 0.0 to 1.0 on how well its RAW FACTS satisfy the
shopper's preferences and intended use. Ground the score in the facts, not in
marketing words in the title, and do not reward a low price by itself.
Reply as JSON: {"scores": [{"product_id": "...", "score": 0.0}]}"""


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Solve one generated-environment shopping task."""

    return Agent(problem_data).run_episode()


__all__ = ["agent_main"]