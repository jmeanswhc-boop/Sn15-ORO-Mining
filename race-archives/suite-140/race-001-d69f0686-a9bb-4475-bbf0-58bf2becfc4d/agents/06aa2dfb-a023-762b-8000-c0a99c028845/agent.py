"""ORO Bench v1 shopping agent.

The environment mutates the moment the agent commits: the first successful
``add_to_cart`` either stocks the item out or reprices it above budget. Reward is
therefore not paid for finding a good product -- it is paid for committing to a
genuine best choice, *observing* the resulting market change, and recovering from
it. A run that skips the observation step still passes the terminal checks but is
scored as a blind recovery and pays nothing.

So the protocol is owned by code, not by the model:

    parse -> retrieve -> shortlist -> commit -> observe the delta -> recover -> order

The model is used only where judgement beats rules (constraint extraction,
candidate screening, soft-preference ranking, terminal claims), and every one of
those calls has a deterministic fallback so an inference outage degrades the run
instead of ending it.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------- environment

READ_CAP = 10          # search/filter row cap enforced by the tools
COMPARE_CAP = 5        # compare product_ids cap
CALLS_PER_TURN = 16    # actions we may batch into one solver turn
MAX_SEARCH_QUERIES = 3 # distinct `search` query strings; more dilutes TF2 quality
SHORTLIST_CAP = 6      # views/compares are precision-scored, so keep this tight
FALLBACK_TURN_RESERVE = 6

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}

_proxy = ProxyClient(timeout=120, max_retries=2)


def _model() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(
        provider, _DEFAULT_MODELS["chutes"]
    )


# ---------------------------------------------------------------- text helpers

_WORD = re.compile(r"[a-z0-9]+")
_RULES_SPLIT = re.compile(r"\n\s*Task rules:\s*\n")

# "at or under 693.00 PHP", "up to 61000.00", "at or below 25899.00 PHP"
_BUDGET = re.compile(
    r"(?:at\s+or\s+under|at\s+or\s+below|no\s+more\s+than|at\s+most|up\s+to|under|"
    r"below|within|budget\s+of|maximum\s+of|max)\s*"
    r"(?:php|₱|\$|usd)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.I,
)
_TRAILING_CCY = re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(php|usd)\b", re.I)
_CANDIDATE_KEY = re.compile(r"\b(\d{4,})::([A-Za-z0-9][\w.\-]*)")
_QUOTED = re.compile(r"[\"“]([^\"”]{2,60})[\"”]")
_INCH = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(?:inch|in\b|\")", re.I)
_HZ = re.compile(r"(\d{2,4}(?:\.\d+)?)\s*-?\s*hz", re.I)
_RESOLUTION = re.compile(
    r"\b(4k|uhd|1440p|qhd|1080p|fhd|full\s*hd|720p|hd|8k)\b", re.I
)
_GB = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_MAH = re.compile(r"(\d{3,6})\s*mah\b", re.I)

_STOP = frozenset(
    """a an the and or for with from that this these those to of in on is are be as at by it
    its my me i we you your please looking find show need want get buy purchase one order place
    test in-stock instock stock item listing product please can could would like new""".split()
)


def _tokens(text: Any) -> list[str]:
    return _WORD.findall(str(text or "").lower())


def _dec(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value).replace(",", "").strip().lstrip("$₱"))
    except (InvalidOperation, ValueError, ArithmeticError):
        return None


def _num(text: Any) -> float | None:
    d = _dec(text)
    return float(d) if d is not None else None


def _split_query(query: str) -> tuple[str, str]:
    """Split the policy query into the shopper's goal and the public task rules.

    ``public_task_query`` joins them with a literal ``Task rules:`` heading, so the
    goal is everything before it. Parsing the goal alone keeps the boilerplate's
    own numbers and quoted words out of constraint extraction.
    """
    parts = _RULES_SPLIT.split(query, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return query.strip(), ""


# ------------------------------------------------------------------- goal spec


class Spec:
    """Constraints recovered from the shopper goal."""

    def __init__(self) -> None:
        self.goal = ""
        self.rules = ""
        self.budget: Decimal | None = None
        self.currency = "PHP"
        self.category: str | None = None
        self.brand: str | None = None
        self.attributes: dict[str, str] = {}
        self.soft: list[str] = []
        self.queries: list[str] = []
        self.candidates: list[tuple[str, str]] = []   # explicit product_id::sku
        self.rank_rule: list[tuple[str, str]] = []    # [(axis, "max"|"min")]
        self.theme_terms: list[str] = []
        self.min_identities = 0
        self.needs_claims = False
        self.signature = "general"
        self.device_rule: dict[str, Any] | None = None
        self.preferred: dict[str, str] = {}

    def budget_f(self) -> float:
        return float(self.budget) if self.budget is not None else float("inf")

    def as_prompt(self) -> str:
        bits = [f"budget: <= {self.budget} {self.currency}" if self.budget else "budget: none"]
        if self.category:
            bits.append(f"category: {self.category}")
        if self.brand:
            bits.append(f"brand: {self.brand}")
        for key, value in self.attributes.items():
            bits.append(f"{key}: {value}")
        bits.append("must be in stock")
        if self.soft:
            bits.append("soft preferences: " + "; ".join(self.soft))
        return "\n".join(f"- {b}" for b in bits)


def _parse_budget(goal: str) -> Decimal | None:
    best: Decimal | None = None
    for match in _BUDGET.finditer(goal):
        value = _dec(match.group(1))
        if value is not None and (best is None or value < best):
            best = value
    if best is None:
        match = _TRAILING_CCY.search(goal)
        if match:
            best = _dec(match.group(1))
    return best


def _parse_attributes(goal: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    inch = _INCH.search(goal)
    if inch:
        attributes["screen_size"] = f"{_num(inch.group(1)):g}in"
    hz = _HZ.search(goal)
    if hz:
        attributes["refresh_rate"] = f"{_num(hz.group(1)):g}hz"
    res = _RESOLUTION.search(goal)
    if res:
        attributes["resolution"] = re.sub(r"\s+", " ", res.group(1)).strip().lower()
    return attributes


_RANK_AXIS = {
    "refresh": "refresh_rate",
    "refresh rate": "refresh_rate",
    "screen": "screen_size",
    "screen size": "screen_size",
    "display": "screen_size",
    "price": "price",
    "cost": "price",
    "resolution": "resolution",
    "storage": "storage",
    "battery": "battery",
    "memory": "storage",
    "ram": "storage",
}
_RANK_PHRASE = re.compile(
    r"\b(highest|largest|biggest|greatest|most|fastest|lowest|smallest|cheapest|least)\s+"
    r"([a-z ]{3,20}?)\b",
    re.I,
)
_MAXWORDS = {"highest", "largest", "biggest", "greatest", "most", "fastest"}


def _parse_rank_rule(goal: str) -> list[tuple[str, str]]:
    """Read a stated lexicographic tie-break rule out of the goal, in order."""
    rule: list[tuple[str, str]] = []
    for match in _RANK_PHRASE.finditer(goal):
        word = match.group(1).lower()
        axis_text = match.group(2).strip().lower()
        axis = None
        for key in sorted(_RANK_AXIS, key=len, reverse=True):
            if axis_text.startswith(key):
                axis = _RANK_AXIS[key]
                break
        if axis is None:
            continue
        direction = "max" if word in _MAXWORDS else "min"
        if not any(existing == axis for existing, _ in rule):
            rule.append((axis, direction))
    return rule


_PREFERRED = re.compile(
    r"(?:strongly\s+)?prefers?\s+the\s+([A-Za-z]+)\s+([A-Za-z0-9][A-Za-z0-9\-]{2,})",
    re.I,
)


def _parse_preferred(goal: str) -> dict[str, str]:
    """The exact item the shopper named, e.g. "the black 431915030732".

    Tasks that state a fallback ladder ("another colour of the same model before
    changing models") are scored on how little the final choice gives up, measured
    from this item -- so the run has to start by committing to it.
    """
    match = _PREFERRED.search(goal or "")
    if not match:
        return {}
    return {"color": match.group(1).lower(), "model": match.group(2).lower()}


_THEME = re.compile(
    r"matches\s+(?:the\s+phrase\s+)?"
    r"(?:[\"\u201c]([^\"\u201d]{2,60})[\"\u201d]|([A-Za-z0-9][\w\- ]{1,58}?))"
    r"\s*(?:,|\.|$)",
    re.I,
)


def _parse_theme(goal: str) -> list[str]:
    """Terms that must appear as whole title tokens.

    The phrase is sometimes quoted and sometimes written bare ("matches the phrase
    roto pvc"), so both spellings have to be read or the title gate never applies.
    """
    match = _THEME.search(goal)
    if match:
        phrase = match.group(1) or match.group(2) or ""
        terms = [t for t in _tokens(phrase) if t]
        if terms:
            return terms
    quoted = _QUOTED.search(goal)
    return [t for t in _tokens(quoted.group(1))] if quoted else []


def _parse_min_identities(rules: str) -> int:
    match = re.search(r"at least (\d+) distinct matching product identities", rules, re.I)
    return int(match.group(1)) if match else 0


_CURRENCY_WORDS = frozenset({"php", "usd", "eur", "gbp", "peso", "pesos", "dollar",
                             "dollars", "budget", "price", "spend", "cost"})


def _attr_terms(key: str, value: str) -> list[str]:
    """Render a constraint value the way a product title would spell it.

    "7.6in" tokenizes to ["7", "6in"], which matches nothing. The catalog writes the
    unit as a word, so emit the number and the unit separately.
    """
    text = str(value).strip().lower()
    if key == "screen_size":
        found = _INCH.search(text) or re.fullmatch(r"(\d+(?:\.\d+)?)", text)
        if found:
            return [f"{_num(found.group(1)):g}", "inch"]
    if key == "refresh_rate":
        found = _HZ.search(text) or re.fullmatch(r"(\d+(?:\.\d+)?)", text)
        if found:
            return [f"{_num(found.group(1)):g}hz"]
    return [t for t in _WORD.findall(text) if len(t) > 1 or t.isdigit()]


def _content_terms(goal: str) -> list[str]:
    """Contentful goal words: no function words, bare numbers, or money vocabulary."""
    out: list[str] = []
    for token in _tokens(goal):
        if token in _STOP or token in _CURRENCY_WORDS or token.isdigit() or len(token) < 3:
            continue
        if token not in out:
            out.append(token)
    return out


def _keyword_queries(spec: Spec) -> list[str]:
    """Up to three BM25 queries. Search is bag-of-words, so word choice is the lever."""
    queries: list[str] = []

    def add(*groups: list[str]) -> None:
        terms: list[str] = []
        for group in groups:
            for term in group:
                if term and term not in terms:
                    terms.append(term)
        text = " ".join(terms).strip()
        if text and text not in queries and len(queries) < MAX_SEARCH_QUERIES:
            queries.append(text)

    brand = _tokens(spec.brand) if spec.brand else []
    category = _tokens(spec.category) if spec.category else []
    attrs: list[str] = []
    for key, value in spec.attributes.items():
        attrs.extend(_attr_terms(key, value))
    content = _content_terms(spec.goal)

    if spec.preferred.get("model"):
        # A named model is the one number worth querying: the run is scored on how
        # close the final pick stays to it, so it has to be found first.
        model = [spec.preferred["model"]]
        colour = [spec.preferred["color"]] if spec.preferred.get("color") else []
        add(model, colour)
        add(model, category)
        add(model)
    if spec.theme_terms:
        # The requested phrase must appear as whole title tokens, so lead with it alone.
        add(spec.theme_terms)
        add(spec.theme_terms, category)
        add(spec.theme_terms, content[:4])
    add(brand, category, attrs)
    add(category, attrs)
    add(brand, category, content[:5])
    add(content[:7])
    return queries[:MAX_SEARCH_QUERIES]


def _signature(spec: Spec, tools: list[dict]) -> str:
    """Name the task shape from public evidence only; it tunes tactics, not the protocol."""
    for tool in tools:
        function = tool.get("function") or {}
        if function.get("name") != "place_test_order":
            continue
        required = (function.get("parameters") or {}).get("required") or []
        if "justification" in required:
            return "justification"
    if spec.candidates:
        return "ranking"
    if re.search(r"disappear|sold out|no longer available|out of stock.*(?:try|another)",
                 spec.goal, re.I):
        return "recovery"
    if spec.theme_terms or spec.min_identities:
        return "retrieval"
    if re.search(r"listing_id", spec.rules, re.I):
        return "preference"
    if re.search(r"prefer|ideally|firm requirements matter more", spec.goal, re.I):
        return "intent"
    return "general"


def parse_goal(query: str, tools: list[dict]) -> Spec:
    spec = Spec()
    spec.goal, spec.rules = _split_query(query)
    goal = spec.goal
    spec.budget = _parse_budget(goal)
    currency = re.search(r"\b(php|usd)\b", goal, re.I)
    if currency:
        spec.currency = currency.group(1).upper()
    spec.attributes = _parse_attributes(goal)
    spec.candidates = [(pid, sku) for pid, sku in _CANDIDATE_KEY.findall(goal)]
    spec.rank_rule = _parse_rank_rule(goal)
    spec.theme_terms = _parse_theme(goal)
    spec.min_identities = _parse_min_identities(spec.rules)
    spec.device_rule = _parse_device_rule(spec.rules)
    spec.preferred = _parse_preferred(goal)
    spec.needs_claims = False
    spec.signature = _signature(spec, tools)
    spec.needs_claims = spec.signature == "justification"
    return spec


# ------------------------------------------------------------------- candidates


class Cand:
    """One purchasable variant, merged from every observation that mentioned it."""

    __slots__ = ("pid", "sku", "title", "brand", "price", "in_stock", "currency",
                 "options", "facts", "category_path", "listing_id", "viewed",
                 "state_seen", "from_filter")

    def __init__(self, pid: str, sku: str) -> None:
        self.pid = str(pid)
        self.sku = str(sku)
        self.title = ""
        self.brand = ""
        self.price: Decimal | None = None
        self.in_stock: bool | None = None
        self.currency = ""
        self.options = ""
        self.facts: dict[str, Any] = {}
        self.category_path: list[str] = []
        self.listing_id = ""
        self.viewed = False       # seen through view/compare (identity evidence)
        self.state_seen = False   # price/stock re-read after the market delta
        self.from_filter = False  # surfaced by `filter`, so category-screened already

    @property
    def key(self) -> str:
        return f"{self.pid}::{self.sku}"

    def text(self) -> str:
        parts = [self.title, self.brand, self.options]
        parts.extend(str(v) for v in self.facts.values())
        return " ".join(p for p in parts if p)

    def brief(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title[:150],
            "brand": self.brand,
            "price": str(self.price) if self.price is not None else None,
            "in_stock": self.in_stock,
            "options": self.options[:80],
            "category_path": self.category_path[-2:],
            "facts": {k: str(v)[:110] for k, v in list(self.facts.items())[:10]},
        }


class Pool:
    """Every candidate the episode has surfaced, keyed by product_id::sku."""

    def __init__(self) -> None:
        self._items: dict[str, Cand] = {}

    def get(self, pid: str, sku: str) -> Cand:
        key = f"{pid}::{sku}"
        if key not in self._items:
            self._items[key] = Cand(pid, sku)
        return self._items[key]

    def find(self, key: str) -> Cand | None:
        return self._items.get(key)

    def all(self) -> list[Cand]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def absorb(self, payload: dict[str, Any], *, post_event: bool) -> None:
        """Merge one observation payload into the pool."""
        if not isinstance(payload, dict):
            return

        from_filter = payload.get("action") == "filter"
        for row in payload.get("results") or []:
            if not isinstance(row, dict) or not row.get("product_id"):
                continue
            cand = self.get(row["product_id"], row.get("sku") or "")
            self._row(cand, row, post_event=post_event)
            # `filter` screens on category server-side; `search` does not, so only a
            # filter hit carries any category evidence before the listing is viewed.
            cand.from_filter = cand.from_filter or from_filter

        for block in [payload] if payload.get("variants") else (payload.get("compared") or []):
            if not isinstance(block, dict) or not block.get("product_id"):
                continue
            self._listing(block, post_event=post_event)

        if payload.get("product_id") and "in_stock" in payload and "variants" not in payload:
            cand = self.get(payload["product_id"], payload.get("sku") or "")
            self._row(cand, payload, post_event=post_event)
            if post_event:
                cand.state_seen = True

        for row in payload.get("cart") or []:
            if not isinstance(row, dict) or not row.get("product_id"):
                continue
            cand = self.get(row["product_id"], row.get("sku") or "")
            # Only `current_price` is live. `price_at_add` is a historical receipt, and
            # letting it win would paper over the very reprice the episode turns on.
            price = _dec(row.get("current_price"))
            if price is not None:
                cand.price = price
            if row.get("in_stock") is not None:
                cand.in_stock = bool(row["in_stock"])

    def _listing(self, block: dict[str, Any], *, post_event: bool) -> None:
        pid = str(block["product_id"])
        for variant in block.get("variants") or []:
            if not isinstance(variant, dict):
                continue
            cand = self.get(pid, variant.get("sku") or "")
            cand.title = block.get("title") or cand.title
            cand.brand = block.get("brand") or cand.brand
            cand.category_path = list(block.get("category_path") or cand.category_path)
            cand.listing_id = block.get("listing_id") or cand.listing_id
            facts = block.get("facts")
            if isinstance(facts, dict):
                cand.facts = dict(facts)
            cand.options = str(variant.get("options") or cand.options)
            price = _dec(variant.get("price"))
            if price is not None:
                cand.price = price
            if variant.get("in_stock") is not None:
                cand.in_stock = bool(variant["in_stock"])
            cand.currency = variant.get("currency") or cand.currency
            cand.viewed = True
            if post_event:
                cand.state_seen = True

    @staticmethod
    def _row(cand: Cand, row: dict[str, Any], *, post_event: bool) -> None:
        cand.title = row.get("title") or cand.title
        cand.brand = row.get("brand") or cand.brand
        price = _dec(row.get("price"))
        if price is not None:
            cand.price = price
        if row.get("in_stock") is not None:
            cand.in_stock = bool(row["in_stock"])
        cand.currency = row.get("currency") or cand.currency
        if post_event:
            cand.state_seen = True


# ---------------------------------------------------------- numeric attributes


def _attr_value(cand: Cand, axis: str) -> float | None:
    """Best-effort numeric reading of one ranking axis from observed facts."""
    if axis == "price":
        return float(cand.price) if cand.price is not None else None
    text = cand.text()
    for key, value in cand.facts.items():
        if axis.split("_")[0] in str(key).lower():
            text = f"{value} {text}"
            break
    if axis == "refresh_rate":
        found = _HZ.search(text)
        return _num(found.group(1)) if found else None
    if axis == "screen_size":
        found = _INCH.search(text)
        return _num(found.group(1)) if found else None
    if axis == "resolution":
        found = _RESOLUTION.search(text)
        if not found:
            return None
        ladder = {"hd": 1.0, "720p": 1.0, "fhd": 2.0, "full hd": 2.0, "1080p": 2.0,
                  "qhd": 3.0, "1440p": 3.0, "4k": 4.0, "uhd": 4.0, "8k": 5.0}
        return ladder.get(re.sub(r"\s+", " ", found.group(1).lower()))
    if axis == "storage":
        found = _GB.search(text)
        if not found:
            return None
        scale = 1024.0 if found.group(2).lower() == "tb" else 1.0
        value = _num(found.group(1))
        return value * scale if value is not None else None
    if axis == "battery":
        found = _MAH.search(text)
        return _num(found.group(1)) if found else None
    return None


def _rank_key(cand: Cand, rule: list[tuple[str, str]]) -> tuple:
    """Lexicographic sort key; a missing reading sorts last on that axis."""
    key: list[float] = []
    for axis, direction in rule:
        value = _attr_value(cand, axis)
        if value is None:
            key.append(float("inf"))
        else:
            key.append(-value if direction == "max" else value)
    return tuple(key)


# ----------------------------------------------------------------- hard screen


# Generic retail category synonyms. The catalog and the shopper rarely spell a
# department the same way, and a mismatch here reads as "wrong product".
_CATEGORY_SYNONYMS = {
    "smartphone": {"smartphone", "smartphones", "mobile phone", "mobile phones",
                   "cell phone", "cell phones", "cellphone", "cellphones", "phone",
                   "phones", "mobiles"},
    "headphone": {"headphone", "headphones", "headset", "headsets", "earphone",
                  "earphones", "earbud", "earbuds"},
    "monitor": {"monitor", "monitors", "computer monitor", "computer monitors",
                "display", "displays"},
    "laptop": {"laptop", "laptops", "notebook", "notebooks"},
    "tablet": {"tablet", "tablets"},
    "graphics card": {"graphics card", "graphics cards", "video card", "video cards",
                      "gpu", "gpus"},
    "smartwatch": {"smartwatch", "smartwatches", "smart watch", "smart watches"},
}


_DEVICE_RULE = re.compile(
    r"the title or raw [\w/ ]*type must explicitly say (\w+)"
    r"(?:, or its device OS must identify ([^.]+))?",
    re.I,
)
_DEVICE_CONFLICTS = re.compile(r"Reject conflicting ([^.]+?) evidence", re.I)
_NEGATED = re.compile(
    r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|"
    r"tablets?|accessor(?:y|ies))\b", re.I)
_BUNDLE = re.compile(
    r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|"
    r"screen\s*protector|charger|accessor(?:y|ies))\b))", re.I)

# Conflict vocabulary, keyed by the words the rule text uses to name each class.
_CONFLICT_PATTERNS = {
    "tablet": r"\btablets?\b",
    "feature": r"\b(?:feature|basic)\s*phones?\b",
    "basic": r"\b(?:feature|basic)\s*phones?\b",
    "mystery": r"\b(?:mystery|blind|lucky)\s*box(?:es)?\b",
    "accessory": r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|"
                 r"case\s*(?:cover\s*)?for|replacement\s*(?:screen|battery))\b",
}


def _parse_device_rule(rules: str) -> dict[str, Any] | None:
    """Read the task's stated device-identity rule instead of assuming one.

    The environment publishes the exact test ("must explicitly say smartphone, or its
    device OS must identify Android, iOS, ..."), so parsing it keeps this generic:
    a different device class with a different rule is handled by the same code.
    """
    match = _DEVICE_RULE.search(rules or "")
    if not match:
        return None
    keyword = match.group(1).lower()
    os_terms: list[str] = []
    if match.group(2):
        for part in re.split(r",| or ", match.group(2)):
            term = " ".join(_tokens(part))
            if term:
                os_terms.append(term)
    conflicts: list[str] = []
    found = _DEVICE_CONFLICTS.search(rules or "")
    if found:
        for part in re.split(r",| or ", found.group(1)):
            for word in _tokens(part):
                pattern = _CONFLICT_PATTERNS.get(word)
                if pattern and pattern not in conflicts:
                    conflicts.append(pattern)
    return {"keyword": keyword, "os": os_terms, "conflicts": conflicts}


def _device_ok(cand: Cand, spec: Spec) -> bool | None:
    """True/False/None against the published device rule. None means unresolved.

    Category is explicitly not proof: a "Mobile Phones" path holds feature phones and
    screen protectors alike, so identity has to come from the title and raw facts.
    """
    rule = spec.device_rule
    if not rule:
        return True
    if not cand.viewed and not cand.title:
        return None

    title = _BUNDLE.split(cand.title or "", maxsplit=1)[0]
    facts = " ".join(f"{k} {v}" for k, v in cand.facts.items())
    identity = _NEGATED.sub("", f"{title} {facts} {cand.options}")

    for pattern in rule["conflicts"]:
        if re.search(pattern, identity, re.I):
            return False
    # An option that sells only a screen or a replacement part is not the device.
    if re.search(r"\b(?:screen\s*only|replacement\s*screen|lcd\s*only)\b",
                 cand.options or "", re.I):
        return False

    if re.search(rf"\b{re.escape(rule['keyword'])}s?\b", identity, re.I):
        return True
    if re.search(rf"\bsmart\s*{re.escape(rule['keyword'])}s?\b", identity, re.I):
        return True
    for term in rule["os"]:
        if re.search(rf"\b{re.escape(term)}\b", identity, re.I):
            return True
    return None if not cand.viewed else None


def _category_group(label: str) -> set[str]:
    """Every spelling that names the same department as `label`."""
    text = " ".join(_tokens(label))
    if not text:
        return set()
    for group in _CATEGORY_SYNONYMS.values():
        if text in group:
            return set(group)
    singular = text[:-1] if text.endswith("s") and len(text) > 3 else text
    return {text, singular, singular + "s"}


def _category_ok(cand: Cand, spec: Spec) -> bool | None:
    """True/False once the listing's category_path is known, else None (unconfirmed).

    A device request must not be satisfied by an accessory for that device, and the
    title alone cannot establish the department -- the catalog path is the evidence.
    """
    if not spec.category:
        return True
    if not cand.category_path:
        return None
    wanted = _category_group(spec.category)
    if not wanted:
        return None
    labels = {" ".join(_tokens(part)) for part in cand.category_path}
    if labels & wanted:
        return True

    # A qualified request like "Gaming Monitors" names no catalog department of its
    # own: it is the head noun's department, narrowed by the modifier. Match the head
    # against the path and look for the modifier in what the listing says about itself.
    words = _tokens(spec.category)
    if len(words) > 1:
        head = _category_group(words[-1])
        if head & labels:
            modifiers = words[:-1]
            text = " ".join(_tokens(f"{cand.title} {cand.options} "
                                    + " ".join(str(v) for v in cand.facts.values())))
            if all(m in text.split() for m in modifiers):
                return True
            return None  # right department, modifier unproven

    # A leaf like "Screen Protectors" under "Mobile & Tablet Accessories" shares the
    # word "mobile" with the request but is not the device.
    return False


def _spec_evidence(cand: Cand, key: str) -> str | None:
    """The listing's own statement of one spec, or None if it never states it.

    Deliberately excludes the title. Tasks plant decoys whose title advertises a
    spec the raw facts contradict, and the verifier reads the facts -- so a title
    match is not evidence, it is the trap.
    """
    matcher = {"screen_size": _INCH, "refresh_rate": _HZ, "resolution": _RESOLUTION}.get(key)
    if matcher is None:
        return None
    options = cand.options or ""
    if matcher.search(options):
        # A variant-level spec gate is decided by the SKU's own options.
        return options
    stated = [f"{k} {v}" for k, v in cand.facts.items() if matcher.search(str(v))]
    return " ".join(stated) if stated else None


def _attrs_confirmed(cand: Cand, spec: Spec) -> bool | None:
    """True when every stated attribute is positively matched in the raw facts.

    Absence is not agreement: a listing that never states the spec returns None so
    the caller goes and looks, rather than being silently accepted.
    """
    if not spec.attributes:
        return True
    if not cand.viewed:
        return None
    verdict: bool | None = True
    for key, wanted in spec.attributes.items():
        text = _spec_evidence(cand, key)
        if text is None:
            verdict = None
            continue
        if key == "screen_size":
            want = _num(re.sub(r"[^\d.]", "", wanted))
            if want is not None and want not in {_num(v) for v in _INCH.findall(text)}:
                return False
        elif key == "refresh_rate":
            want = _num(re.sub(r"[^\d.]", "", wanted))
            if want is not None and want not in {_num(v) for v in _HZ.findall(text)}:
                return False
        elif key == "resolution":
            ladder = {"hd": 1, "720p": 1, "fhd": 2, "full hd": 2, "1080p": 2,
                      "qhd": 3, "1440p": 3, "4k": 4, "uhd": 4, "8k": 5}
            want = ladder.get(re.sub(r"\s+", " ", wanted.lower()))
            got = {ladder.get(re.sub(r"\s+", " ", m.lower()))
                   for m in _RESOLUTION.findall(text)}
            if want is not None and got and want not in got:
                return False
    return verdict


def _hard_ok(cand: Cand, spec: Spec, budget: Decimal | None) -> bool:
    """Cheap, conservative pre-screen. Only rejects on evidence, never on absence."""
    if cand.in_stock is False:
        return False
    if budget is not None:
        # Unknown is not affordable. `within_budget` is a hard gate, so a listing
        # whose price was never observed cannot be bought on the hope that it fits.
        if cand.price is None or cand.price > budget:
            return False
    if spec.brand and cand.brand:
        wanted = set(_tokens(spec.brand))
        if wanted and not wanted <= set(_tokens(cand.brand) + _tokens(cand.title)):
            return False
    if spec.theme_terms and cand.title:
        title_tokens = set(_tokens(cand.title))
        if not set(spec.theme_terms) <= title_tokens:
            return False
    return True


def _relevance(cand: Cand, spec: Spec) -> float:
    """Token overlap with the goal, used to order the shortlist before viewing."""
    goal_tokens = {t for t in _tokens(spec.goal) if t not in _STOP}
    if not goal_tokens:
        return 0.0
    hit = goal_tokens & set(_tokens(cand.title) + _tokens(cand.brand))
    score = len(hit) / len(goal_tokens)
    if spec.theme_terms and set(spec.theme_terms) <= set(_tokens(cand.title)):
        score += 1.0
    return score


# ---------------------------------------------------------------- model oracle

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.S)


class Oracle:
    """Strict-JSON model calls with one repair round and a silent give-up."""

    def __init__(self, trace: list[dict[str, Any]]) -> None:
        self._trace = trace
        self.calls = 0

    def _chat(self, messages: list[dict[str, Any]]) -> str | None:
        response = _proxy.post(
            "/inference/chat/completions",
            json_data={
                "model": _model(),
                "messages": messages,
                "temperature": 0,
                "max_tokens": 1600,
            },
        )
        self.calls += 1
        if not isinstance(response, dict):
            return None
        choices = response.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        return message.get("content") or ""

    def json(self, system: str, user: str) -> Any:
        """Ask for JSON; on unparsable output, show the model its own text and retry once.

        The repair round is the one technique the v3 corpus agreed on most strongly:
        a second pass that audits the first is worth more than a longer first prompt.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for attempt in range(2):
            raw = self._chat(messages)
            if raw is None:
                return None
            parsed = self._parse(raw)
            if parsed is not None:
                return parsed
            if attempt == 0:
                messages.extend([
                    {"role": "assistant", "content": raw[:2000]},
                    {"role": "user", "content":
                        "That was not valid JSON. Reply with the JSON value only: "
                        "no prose, no code fence, no trailing commentary."},
                ])
        return None

    @staticmethod
    def _parse(raw: str) -> Any:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            pass
        match = _JSON_BLOCK.search(text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except (ValueError, TypeError):
            return None


_SPEC_SYSTEM = (
    "You extract purchase constraints from a shopper's request. Separate what the "
    "shopper requires from what they merely prefer: a firm requirement disqualifies a "
    "product, a preference only ranks it. Report only what the text states."
)

_SCREEN_SYSTEM = (
    "You check candidate products against a shopper's firm requirements. "
    "For each candidate, write the check first and decide afterwards. "
    "Judge only from the fields given; if a required fact is absent, the candidate "
    "does not satisfy it. Titles can lie about specifications -- prefer the facts and "
    "the category path over wording in the title."
)

_RANK_SYSTEM = (
    "You order products by how well they serve a shopper's stated preferences. "
    "Every candidate already meets the firm requirements, so rank on the preferences "
    "alone, best first."
)

_CLAIMS_SYSTEM = (
    "You state checkable facts about a product a shopper is about to buy. "
    "Every claim must be literally true of the product data given. "
    "Never guess and never restate the shopper's wording; copy values from the data."
)


def llm_spec(oracle: Oracle, spec: Spec) -> None:
    """Let the model fill the gaps regex cannot reach (category, brand, soft prefs)."""
    payload = oracle.json(
        _SPEC_SYSTEM,
        "Shopper request:\n"
        f"{spec.goal[:2000]}\n\n"
        "Return JSON exactly like:\n"
        '{"category": <string or null>, "brand": <string or null>, '
        '"attributes": {<spec field>: <value>}, '
        '"soft_preferences": [<short phrase>, ...]}\n'
        "category is the product type the shopper wants (singular or plural noun). "
        "brand is only set when the shopper names a manufacturer as a requirement. "
        "attributes holds firm technical requirements such as screen_size, resolution "
        "or refresh_rate, with the value written as the shopper wrote it. "
        "soft_preferences holds wishes that should not disqualify a product.",
    )
    if not isinstance(payload, dict):
        return
    category = payload.get("category")
    if isinstance(category, str) and category.strip():
        spec.category = category.strip()
    brand = payload.get("brand")
    if isinstance(brand, str) and brand.strip():
        spec.brand = brand.strip()
    attributes = payload.get("attributes")
    if isinstance(attributes, dict):
        for key, value in attributes.items():
            key = re.sub(r"[^a-z_]", "_", str(key).lower().replace(" ", "_"))
            if key in spec.attributes or not isinstance(value, (str, int, float)):
                continue
            if key in {"screen_size", "resolution", "refresh_rate"}:
                spec.attributes[key] = str(value).strip().lower()
    soft = payload.get("soft_preferences")
    if isinstance(soft, list):
        spec.soft = [str(s).strip() for s in soft if str(s).strip()][:6]


def llm_screen(oracle: Oracle, spec: Spec, cands: list[Cand]) -> set[str]:
    """Return the keys the model believes satisfy every firm requirement."""
    if not cands:
        return set()
    labelled = {f"C{i}": c for i, c in enumerate(cands, start=1)}
    payload = oracle.json(
        _SCREEN_SYSTEM,
        f"Shopper request:\n{spec.goal[:1200]}\n\n"
        f"Firm requirements:\n{spec.as_prompt()}\n\n"
        "Candidates:\n"
        + json.dumps({k: c.brief() for k, c in labelled.items()}, ensure_ascii=False)[:9000]
        + "\n\nReturn JSON exactly like:\n"
        '{"C1": {"check": "<one sentence>", "ok": true|false}, ...}\n'
        "Include every candidate label exactly once.",
    )
    if not isinstance(payload, dict):
        return {c.key for c in cands}
    keep = set()
    for label, cand in labelled.items():
        verdict = payload.get(label)
        if isinstance(verdict, dict) and verdict.get("ok") is True:
            keep.add(cand.key)
    return keep or {c.key for c in cands}


def llm_rank(oracle: Oracle, spec: Spec, cands: list[Cand]) -> list[str]:
    # A task can carry its preferences entirely in prose, so the absence of an
    # extracted list is not a reason to skip ranking on the request itself.
    if len(cands) < 2 or not (spec.soft or spec.signature == "preference"):
        return []
    labelled = {f"C{i}": c for i, c in enumerate(cands[:12], start=1)}
    stated = ("\n".join(f"- {s}" for s in spec.soft)
              if spec.soft else "- as described in the request above")
    payload = oracle.json(
        _RANK_SYSTEM,
        f"Shopper request:\n{spec.goal[:1400]}\n\n"
        f"Preferences, most important first:\n{stated}"
        + "\n\nCandidates:\n"
        + json.dumps({k: c.brief() for k, c in labelled.items()}, ensure_ascii=False)[:11000]
        + "\n\nJudge from the product facts, not the price: the shopper said what they "
          "care about, and the cheapest item is rarely the one that serves it best. "
          'Return JSON exactly like {"order": ["C2", "C1", ...]}, best first, '
          "every label exactly once.",
    )
    order = payload.get("order") if isinstance(payload, dict) else None
    if not isinstance(order, list):
        return []
    return [labelled[label].key for label in order
            if isinstance(label, str) and label in labelled]


def llm_claims(oracle: Oracle, spec: Spec, cand: Cand, price: Decimal | None,
               required: list[str]) -> list[dict[str, str]]:
    payload = oracle.json(
        _CLAIMS_SYSTEM,
        f"Shopper request:\n{spec.goal[:1000]}\n\n"
        f"Product about to be ordered:\n"
        + json.dumps(cand.brief(), ensure_ascii=False)[:4000]
        + f"\n\nIts price at order is {price} {spec.currency} and it is in stock.\n"
        f"Required claim fields: {', '.join(required)}\n\n"
        'Return JSON exactly like {"claims": [{"field": "<field>", "value": "<value>"}, ...]} '
        "with one entry per required field and nothing else. "
        "For price, give the numeric amount only. For in_stock, give \"true\". "
        "For a specification field, copy the value from the product facts verbatim.",
    )
    claims = payload.get("claims") if isinstance(payload, dict) else None
    out: list[dict[str, str]] = []
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            field = str(claim.get("field") or "").strip().lower().replace(" ", "_")
            value = str(claim.get("value") or "").strip()
            if field and value:
                out.append({"field": field, "value": value})
    return out


# --------------------------------------------------------------- protocol layer


class Env:
    """One session's call channel. Owns turn numbering and the observation pool."""

    def __init__(self, problem: dict[str, Any], trace: list[dict[str, Any]]) -> None:
        environment = problem["environment"]
        self.binding = environment["binding"]
        self.policy = environment["policy_view"]
        self.problem_id = str(problem.get("problem_id", problem.get("id", "problem")))
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, min(int(self.policy.get("max_calls_per_turn", 1) or 1),
                                    CALLS_PER_TURN))
        self.turn = 0
        self.done = False
        self.pool = Pool()
        self.event_seen = False
        self.event_target: str | None = None
        self.pending_messages: list[str] = []
        self.budget_updates: list[Decimal] = []
        self._trace = trace

    @property
    def turns_left(self) -> int:
        return self.max_steps - self.turn

    def act(self, actions: list[dict[str, Any]], *, post_event: bool = False
            ) -> list[dict[str, Any]]:
        """Run one solver turn. Returns the observation payloads, in order."""
        if self.done or not actions:
            return []
        actions = actions[: self.max_calls]
        self.turn += 1
        group = f"{self.problem_id}-t{self.turn}"
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": self.turn,
            "calls": [
                {"call_id": f"{group}-{i}", "action": action}
                for i, action in enumerate(actions, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if not isinstance(result, dict):
            raise RuntimeError("environment call failed")

        observations: list[dict[str, Any]] = []
        for action, call in zip(actions, result.get("calls") or []):
            # calls[i].observation is the public *step result*; the tool payload sits
            # one level inside it, alongside its own done/error flags.
            step = call.get("observation")
            if not isinstance(step, dict):
                step = {}
            payload = step.get("observation")
            if not isinstance(payload, dict):
                payload = {}
            observations.append(payload)
            self.pool.absorb(payload, post_event=post_event or self.event_seen)
            event = payload.get("market_event")
            if isinstance(event, dict):
                target = event.get("target") or {}
                pid, sku = target.get("product_id"), target.get("sku")
                if pid:
                    self.event_target = f"{pid}::{sku or ''}"
                self.event_seen = True
            if step.get("done") is True:
                self.done = True
            self._trace.append({
                "turn": self.turn,
                "action": action.get("name"),
                "args": action.get("args"),
                "error": step.get("error"),
                "observation": _compact(payload),
            })

        message = result.get("user_message")
        if isinstance(message, dict) and message.get("content"):
            content = str(message["content"])
            self.pending_messages.append(content)
            self._trace.append({"turn": self.turn, "user_message": content[:600]})
            tightened = _parse_budget(content)
            if tightened is not None:
                self.budget_updates.append(tightened)
        return observations


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    """Shrink an observation for the returned trajectory; the ledger keeps the original."""
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in ("results", "variants", "compared", "cart"):
            out[key] = len(value) if isinstance(value, list) else value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value if not isinstance(value, str) else value[:200]
        else:
            out[key] = str(value)[:200]
    return out


def _act(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": {k: v for k, v in args.items() if v is not None}}


# ------------------------------------------------------------------- selection


def effective_budget(spec: Spec, env: Env) -> Decimal | None:
    """The shopper may tighten the budget mid-episode; the tightest value wins."""
    values = [b for b in ([spec.budget] if spec.budget is not None else []) + env.budget_updates]
    return min(values) if values else None


def viable(env: Env, spec: Spec, exclude: set[str],
           exclude_listings: set[str] | None = None,
           *, strict: bool = True) -> list[Cand]:
    """Candidates still worth buying.

    `strict` demands positive confirmation of category and attributes, so an
    unexamined listing is excluded rather than assumed good. Relaxing it admits
    listings that merely have not been contradicted, which is a last resort.
    """
    budget = effective_budget(spec, env)
    out = []
    for cand in env.pool.all():
        if cand.key in exclude:
            continue
        if exclude_listings and cand.listing_id and cand.listing_id in exclude_listings:
            continue
        if spec.candidates and (cand.pid, cand.sku) not in spec.candidates:
            continue
        if not _hard_ok(cand, spec, budget):
            continue
        category = _category_ok(cand, spec)
        attrs = _attrs_confirmed(cand, spec)
        device = _device_ok(cand, spec)
        if category is False or attrs is False or device is False:
            continue
        if strict and (category is not True or attrs is not True or device is not True):
            continue
        out.append(cand)
    return out


def order_candidates(env: Env, spec: Spec, cands: list[Cand],
                     llm_order: list[str]) -> list[Cand]:
    """Stated tie-break rule first, then model preference order, then cheapest."""
    if spec.rank_rule:
        return sorted(cands, key=lambda c: _rank_key(c, spec.rank_rule))
    if llm_order:
        position = {key: i for i, key in enumerate(llm_order)}
        return sorted(cands, key=lambda c: (position.get(c.key, len(position)),
                                            float(c.price or 0)))
    # An unresolved device identity is not eligible under the published rule, so a
    # confirmed listing outranks an unproven one whatever else is equal.
    def proven(cand: Cand) -> int:
        return 0 if _device_ok(cand, spec) is True else 1

    if spec.soft:
        # "I would rather pay a bit more than settle" -- cheapest-first is the wrong
        # default here, and committing to it fails the family's own commitment gate.
        return sorted(
            cands,
            key=lambda c: (proven(c), -_soft_score(c, spec), -float(c.price or 0)),
        )
    return sorted(
        cands,
        key=lambda c: (proven(c), -_relevance(c, spec),
                       float(c.price) if c.price is not None else 0.0),
    )


def _soft_score(cand: Cand, spec: Spec) -> float:
    """How much observable evidence supports the shopper's stated preferences."""
    text = set(_tokens(cand.text()))
    if not text:
        return 0.0
    score = 0.0
    for phrase in spec.soft:
        terms = {t for t in _tokens(phrase) if t not in _STOP and len(t) > 2}
        if terms:
            score += len(terms & text) / len(terms)
    return score


def _preferred_match(cand: Cand, spec: Spec) -> int:
    """2 when this variant is the exact item named, 1 for the model, 0 otherwise."""
    if not spec.preferred:
        return 0
    text = " ".join(_tokens(f"{cand.title} {cand.options}"))
    model = spec.preferred.get("model", "")
    color = spec.preferred.get("color", "")
    has_model = bool(model) and model in text
    has_color = bool(color) and color in text
    return 2 if (has_model and has_color) else 1 if has_model else 0


def relaxation_rank(cand: Cand, spec: Spec, target: Cand | None) -> tuple:
    """Order recovery options by how little they give up, per the stated order.

    The goals state their own ladder ("try another color of the same model before
    changing models"), so model identity is preserved ahead of colour.
    """
    if target is None:
        return (0,)
    model_kept = cand.pid == target.pid
    target_options = set(_tokens(target.options))
    options_kept = bool(target_options) and target_options <= set(_tokens(cand.options))
    colour_first = bool(re.search(r"another colou?r of the same model", spec.goal, re.I))
    if colour_first:
        return (0 if model_kept else 1, 0 if options_kept else 1,
                float(cand.price or 0))
    return (0 if options_kept else 1, 0 if model_kept else 1, float(cand.price or 0))


def _shortlist_cap(spec: Spec) -> int:
    """How many listings are worth opening for this task shape.

    A spec gate is decided per variant and is usually invisible from the title, so
    those tasks need a wide sweep. Retrieval tasks are scored on the precision of
    what gets opened, so they stay narrow.
    """
    if spec.theme_terms:
        return SHORTLIST_CAP
    if spec.attributes or spec.soft:
        return SHORTLIST_CAP * 2
    return SHORTLIST_CAP


def _plausible(env: Env, spec: Spec, exclude: set[str]) -> list[Cand]:
    """Candidates worth spending a look on: not ruled out, not yet confirmed."""
    budget = effective_budget(spec, env)
    out = []
    for cand in env.pool.all():
        if cand.key in exclude:
            continue
        if spec.candidates and (cand.pid, cand.sku) not in spec.candidates:
            continue
        if not _hard_ok(cand, spec, budget):
            continue
        if (_category_ok(cand, spec) is False or _attrs_confirmed(cand, spec) is False
                or _device_ok(cand, spec) is False):
            continue
        out.append(cand)
    return out


def _expand(env: Env, spec: Spec, seen: set[str]) -> bool:
    """Surface more candidates without spending a search query.

    Only distinct `search` query strings are counted against retrieval quality, so
    `filter` is the free way to widen the pool -- and it screens category server-side.
    """
    budget = effective_budget(spec, env)
    ceiling = float(budget) if budget is not None else None
    plans: list[dict[str, Any]] = []
    if spec.brand:
        plans.append({"category": spec.category, "brand": spec.brand,
                      "max_price": ceiling, "in_stock": True, "k": READ_CAP})
    plans.append({"category": spec.category, "max_price": ceiling,
                  "in_stock": True, "k": READ_CAP})
    if ceiling is not None:
        # Cheaper bands reach rows the first page never showed.
        for share in (0.6, 0.35, 0.15):
            plans.append({"category": spec.category, "max_price": round(ceiling * share, 2),
                          "in_stock": True, "k": READ_CAP})
    actions = []
    for plan in plans:
        marker = json.dumps(plan, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        actions.append(_act("filter", **plan))
    if not actions:
        return False
    env.act(actions[: env.max_calls], post_event=env.event_seen)
    return True


def _resolve(env: Env, spec: Spec, exclude: set[str], *, post_event: bool,
             want: int = 2, rounds: int = 6) -> list[Cand]:
    """Look until enough candidates are positively confirmed, or the budget runs out.

    Buying an unconfirmed listing is how a smartphone request ends up ordering a
    screen protector, so the loop spends turns on evidence before it spends one on
    a commitment.
    """
    seen_filters: set[str] = set()
    for _ in range(rounds):
        confirmed = viable(env, spec, exclude, strict=True)
        if len(confirmed) >= want:
            return confirmed
        if env.turns_left <= FALLBACK_TURN_RESERVE:
            break
        todo = [c for c in _plausible(env, spec, exclude)
                if not c.viewed or (post_event and not c.state_seen)]
        if todo:
            _inspect(env, spec, order_candidates(env, spec, todo, []),
                     post_event=post_event)
            continue
        if spec.candidates or not _expand(env, spec, seen_filters):
            break
    strict = viable(env, spec, exclude, strict=True)
    if strict:
        return strict
    # Nothing could be confirmed. Prefer a merely-uncontradicted listing over no
    # order at all, but never one already shown to violate the request.
    return viable(env, spec, exclude, strict=False)


# ------------------------------------------------------------------ main phases


def _reply_actions(env: Env, spec: Spec, oracle: Oracle) -> list[dict[str, Any]]:
    """Answer a shopper question. An unanswered clarification fails the family gate."""
    if not env.pending_messages:
        return []
    questions = env.pending_messages[:]
    env.pending_messages.clear()
    text = " ".join(questions)[-800:]
    reply = None
    if "?" in text or re.search(r"confirm|reply|respond|answer|tell me|let me know",
                                text, re.I):
        payload = oracle.json(
            "You reply to a shopper on behalf of a shopping agent. Be brief, concrete, "
            "and answer exactly what was asked.",
            f"The shopper's original request was:\n{spec.goal[:800]}\n\n"
            f"They now say:\n{text}\n\n"
            'Return JSON exactly like {"reply": "<one or two sentences>"}.',
        )
        if isinstance(payload, dict) and str(payload.get("reply") or "").strip():
            reply = str(payload["reply"]).strip()[:400]
    if reply is None:
        reply = (
            "Understood - I am keeping your firm requirements and budget, confirming "
            "current price and stock, and I will place the order once it checks out."
        )
    return [_act("message", content=reply)]


def _gather(env: Env, spec: Spec) -> None:
    """One turn of retrieval: every planned search plus a structured filter."""
    actions: list[dict[str, Any]] = []
    if spec.candidates:
        # The goal named the exact variants; read them directly instead of searching.
        for pid in list(dict.fromkeys(pid for pid, _ in spec.candidates))[:COMPARE_CAP]:
            actions.append(_act("view", product_id=pid))
        env.act(actions)
        return

    for query in spec.queries:
        actions.append(_act("search", query=query, k=READ_CAP, in_stock=True))
    budget = effective_budget(spec, env)
    actions.append(_act(
        "filter",
        category=spec.category,
        brand=spec.brand,
        max_price=float(budget) if budget is not None else None,
        in_stock=True,
        k=READ_CAP,
    ))
    env.act(actions)


def _distinct_listing_required(spec: Spec) -> bool:
    """True when the task's public rules speak in terms of listing identity."""
    return bool(re.search(r"listing_id", spec.rules, re.I))


def _spread_listings(cands: list[Cand], limit: int) -> list[Cand]:
    """One candidate per listing first, so a shortlist cannot be all one listing."""
    first: list[Cand] = []
    rest: list[Cand] = []
    seen: set[str] = set()
    for cand in cands:
        marker = cand.listing_id or cand.pid
        (rest if marker in seen else first).append(cand)
        seen.add(marker)
    return (first + rest)[:limit]


def _inspect(env: Env, spec: Spec, cands: list[Cand], *, post_event: bool) -> None:
    """View or compare a shortlist. Every look is precision-scored, so keep it small."""
    fresh = [c for c in cands if not c.viewed or (post_event and not c.state_seen)]
    if not fresh:
        return
    cap = _shortlist_cap(spec)
    fresh = (_spread_listings(fresh, cap)
             if _distinct_listing_required(spec) else fresh[:cap])
    actions: list[dict[str, Any]] = []
    pids = list(dict.fromkeys(c.pid for c in fresh))
    for start in range(0, len(pids), COMPARE_CAP):
        chunk = pids[start:start + COMPARE_CAP]
        if len(chunk) == 1:
            actions.append(_act("view", product_id=chunk[0]))
        else:
            actions.append(_act("compare", product_ids=chunk))
    if post_event:
        for cand in fresh:
            actions.append(_act("inspect_stock", product_id=cand.pid, sku=cand.sku))
    env.act(actions, post_event=post_event)


def _finalize(env: Env, spec: Spec, oracle: Oracle, choice: Cand) -> bool:
    """Add the final pick and order it, with claims only when the shopper asked."""
    env.act([_act("add_to_cart", product_id=choice.pid, sku=choice.sku)],
            post_event=True)
    if env.done:
        return True

    fresh = env.pool.find(choice.key) or choice
    args: dict[str, Any] = {"product_id": choice.pid, "sku": choice.sku}
    if spec.needs_claims:
        required = ["price", "in_stock"]
        gate = next(iter(spec.attributes), None)
        if gate:
            required.insert(0, gate)
        claims = llm_claims(oracle, spec, fresh, fresh.price, required)
        claims = _repair_claims(claims, spec, fresh, required)
        if claims:
            args["justification"] = {"claims": claims}
    env.act([{"name": "place_test_order", "args": args}], post_event=True)
    return env.done


_REASON_FIELDS = frozenset({"streaming", "microphone", "privacy_cover",
                            "autofocus", "hdr"})


def _repair_claims(claims: list[dict[str, str]], spec: Spec, cand: Cand,
                   required: list[str]) -> list[dict[str, str]]:
    """Force every required field present and every value checkable against observation.

    A false claim is a hard gate failure, and a claim is only checked when it is
    submitted -- so anything that cannot be grounded is dropped rather than guessed.
    """
    allowed = {"category", "resolution", "refresh_rate", "screen_size", "price",
               "in_stock", "brand"}
    if spec.needs_claims:
        allowed = allowed | _REASON_FIELDS
    by_field = {c["field"]: c["value"] for c in claims if c["field"] in allowed}

    if cand.price is not None:
        by_field["price"] = f"{cand.price}"
    elif "price" in by_field and _dec(by_field["price"]) is None:
        by_field.pop("price")
    by_field["in_stock"] = "true"

    for field in list(by_field):
        if field in ("price", "in_stock"):
            continue
        if field == "brand":
            if not cand.brand or _tokens(by_field[field]) != _tokens(cand.brand):
                by_field[field] = cand.brand or by_field[field]
            if not cand.brand:
                by_field.pop(field)
        elif field in ("resolution", "refresh_rate", "screen_size"):
            observed = _observed_attr(cand, field)
            if observed:
                by_field[field] = observed
            elif field not in spec.attributes:
                by_field.pop(field)

    for field in required:
        if field in by_field or field not in allowed:
            continue
        if field in spec.attributes:
            observed = _observed_attr(cand, field) or spec.attributes[field]
            by_field[field] = observed

    # A justification is only faithful when it also gives a *reason*: one claim from
    # the supported-reason set, not just the spec, the price and the stock state.
    # `brand` is the one reason that is always checkable from an observed listing.
    if spec.needs_claims and not ({"brand"} | _REASON_FIELDS) & set(by_field):
        if cand.brand:
            by_field["brand"] = cand.brand
    return [{"field": f, "value": v} for f, v in by_field.items() if str(v).strip()]


def _observed_attr(cand: Cand, field: str) -> str | None:
    """Read one specification value back out of the product's own facts."""
    for key, value in cand.facts.items():
        if field.split("_")[0] not in str(key).lower():
            continue
        text = str(value)
        if field == "refresh_rate":
            found = _HZ.search(text)
            if found:
                return f"{_num(found.group(1)):g}Hz"
        elif field == "screen_size":
            found = _INCH.search(text)
            if found:
                return f"{_num(found.group(1)):g} inch"
        elif field == "resolution":
            found = _RESOLUTION.search(text)
            if found:
                return found.group(1)
        return text.strip()[:60] or None
    text = cand.text()
    if field == "refresh_rate":
        found = _HZ.search(text)
        return f"{_num(found.group(1)):g}Hz" if found else None
    if field == "screen_size":
        found = _INCH.search(text)
        return f"{_num(found.group(1)):g} inch" if found else None
    if field == "resolution":
        found = _RESOLUTION.search(text)
        return found.group(1) if found else None
    return None


def _last_resort(env: Env, spec: Spec, oracle: Oracle, exclude: set[str]) -> None:
    """Place the best still-legal order rather than ending the episode with none.

    An episode that never orders fails every gate at once, so even a doubtful
    order strictly dominates giving up.
    """
    if env.done or env.turns_left < 2:
        return
    budget = effective_budget(spec, env)
    pool = [
        cand for cand in env.pool.all()
        if cand.key not in exclude
        and cand.in_stock is not False
        and _device_ok(cand, spec) is not False
        and _category_ok(cand, spec) is not False
        and (budget is None or (cand.price is not None and cand.price <= budget))
    ]
    if not pool:
        # Relax what is unproven, never what is checked: an order over budget or of
        # unknown price scores exactly zero, so it buys nothing that no order does.
        pool = [cand for cand in env.pool.all()
                if cand.key not in exclude and cand.in_stock is not False
                and (budget is None or (cand.price is not None and cand.price <= budget))]
    for choice in order_candidates(env, spec, pool, [])[:2]:
        if env.done or env.turns_left < 2:
            return
        if _finalize(env, spec, oracle, choice):
            return


# ----------------------------------------------------------------- entry point


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Run one generated-environment episode and return its trajectory."""
    trace: list[dict[str, Any]] = []
    env = Env(problem_data, trace)
    oracle = Oracle(trace)
    spec = parse_goal(str(env.policy.get("query") or ""), env.policy.get("tools") or [])
    llm_spec(oracle, spec)
    spec.queries = _keyword_queries(spec)
    trace.append({"spec": {
        "signature": spec.signature,
        "budget": str(spec.budget),
        "category": spec.category,
        "brand": spec.brand,
        "attributes": spec.attributes,
        "soft": spec.soft,
        "queries": spec.queries,
        "rank_rule": spec.rank_rule,
        "candidates": len(spec.candidates),
    }})

    # -- retrieve, then confirm before committing -----------------------------
    _gather(env, spec)
    pool = _resolve(env, spec, exclude=set(), post_event=False, want=3)
    if env.pending_messages:
        env.act(_reply_actions(env, spec, oracle))
        pool = _resolve(env, spec, exclude=set(), post_event=False, want=3)
    if not pool:
        pool = [c for c in env.pool.all()
                if c.in_stock is not False and _device_ok(c, spec) is not False]
    if not pool:
        return trace

    keep = llm_screen(oracle, spec, order_candidates(env, spec, pool, [])[:SHORTLIST_CAP])
    screened = [c for c in pool if c.key in keep] or pool
    ordered = order_candidates(env, spec, screened, llm_rank(oracle, spec, screened))
    if spec.preferred:
        # The stated ladder is measured from the named item, so it has to be the
        # first commitment even when something else looks like a better buy.
        ordered = sorted(ordered, key=lambda c: -_preferred_match(c, spec))
    if not ordered:
        return trace

    # -- commit: the honest first choice, which the market will then break -----
    # Several families score the initial commitment on evidence gathered *before* the
    # add, and an unviewed listing has none -- so look first, then commit.
    first = None
    for candidate in ordered[:3]:
        if not candidate.viewed and env.turns_left > FALLBACK_TURN_RESERVE:
            env.act([_act("view", product_id=candidate.pid)])
        fresh = env.pool.find(candidate.key) or candidate
        if (_category_ok(fresh, spec) is not False
                and _attrs_confirmed(fresh, spec) is not False
                and _device_ok(fresh, spec) is not False):
            first = fresh
            break
    if first is None:
        first = env.pool.find(ordered[0].key) or ordered[0]
    env.act([_act("add_to_cart", product_id=first.pid, sku=first.sku)])
    if env.done:
        return trace

    # -- observe the delta before choosing again ------------------------------
    # Skipping this is scored as a blind recovery and pays nothing, so it is
    # unconditional even when the add already echoed a market_event.
    probe = [_act("inspect_cart"),
             _act("inspect_stock", product_id=first.pid, sku=first.sku)]
    env.act(probe, post_event=True)
    if env.pending_messages:
        env.act(_reply_actions(env, spec, oracle))

    dead = env.pool.find(env.event_target) if env.event_target else None
    broken = {first.key}
    if dead is not None:
        broken.add(dead.key)
    current = env.pool.find(first.key)
    if current is not None and current.in_stock is not False:
        budget = effective_budget(spec, env)
        if budget is None or current.price is None or current.price <= budget:
            broken.discard(first.key)   # nothing actually moved; keep the choice

    # -- recover --------------------------------------------------------------
    # Some tasks state that a replacement must come from a different underlying
    # listing, not a sibling variant of the one that just broke.
    dead_listings: set[str] = set()
    if _distinct_listing_required(spec):
        for key in broken:
            spoiled = env.pool.find(key)
            if spoiled is not None and spoiled.listing_id:
                dead_listings.add(spoiled.listing_id)

    survivors = [c for c in _resolve(env, spec, broken, post_event=True, want=2)
                 if not (dead_listings and c.listing_id in dead_listings)]
    if not survivors:
        survivors = _resolve(env, spec, broken, post_event=True, want=1)
    if not survivors:
        _last_resort(env, spec, oracle, broken)
        return trace

    ranked = order_candidates(env, spec, survivors, llm_rank(oracle, spec, survivors))
    if spec.signature == "recovery":
        # Giving up less is only worth anything among items that are still eligible.
        ranked = sorted(ranked, key=lambda c: (
            0 if _device_ok(c, spec) is True else 1,) + relaxation_rank(c, spec, first))

    # Identity and current state must both be observed after the delta.
    _inspect(env, spec, ranked[:SHORTLIST_CAP], post_event=True)
    if env.pending_messages:
        env.act(_reply_actions(env, spec, oracle))

    final = (viable(env, spec, exclude=broken, exclude_listings=dead_listings)
             or viable(env, spec, exclude=broken)
             or viable(env, spec, exclude=broken, strict=False))
    final = order_candidates(env, spec, final, []) if final else ranked
    if spec.signature == "recovery":
        final = sorted(final, key=lambda c: (
            0 if _device_ok(c, spec) is True else 1,) + relaxation_rank(c, spec, first))

    for choice in final[:3]:
        if env.done or env.turns_left < 2:
            break
        if _finalize(env, spec, oracle, choice):
            break
    if not env.done:
        _last_resort(env, spec, oracle, broken)

    trace.append({"summary": {
        "turns": env.turn,
        "inference_calls": oracle.calls,
        "pool": len(env.pool),
        "event_seen": env.event_seen,
        "event_target": env.event_target,
        "done": env.done,
    }})
    return trace


__all__ = ["agent_main"]
