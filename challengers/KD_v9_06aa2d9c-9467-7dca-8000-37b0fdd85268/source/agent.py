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
_MAX_SEARCH_PHRASES = 3
_MAX_SWEEPS = 4
_BRAND_SWEEPS = 3
_PAGE = 10
_COMPARE_GROUP = 5
# Opening a listing costs attention and says "this looked right to me", so the
# assistant reads a focused shortlist rather than the whole results page.
_MAX_OPENED = 12
_RELEVANCE_FLOOR = 0.5
# A marketplace lists spares, cases and cables beside the products they fit.
# Anything priced at a small fraction of the going rate for what else the
# customer could buy is almost certainly one of those rather than the thing
# itself, however well its title matches.
_CLASS_PRICE_FLOOR = 0.2
# Long enough that no price, measurement or model number a customer would
# actually write reaches it.
_REFERENCE_DIGITS = 10
# Turns held back for read -> add -> re-read -> order, plus slack for one
# recovery cycle and a final reply if the market or the customer moves late.
_COMMIT_RESERVE = 7
_MAX_REPLIES = 8
_MAX_CLAIMS = 4
# Attribute values are short by nature; anything longer is marketing copy that
# happens to mention the value rather than a statement of it.
_MAX_CLAIM_CHARS = 60
# Turns to hold off paying while a question to the customer is outstanding.
_ANSWER_GRACE = 3
_EVIDENCE_CHARS = 700
_PRICE_EPSILON = 0.005
_TWIN_SIMILARITY = 0.6

_WORD_RE = re.compile(r"[a-z0-9]+")
# Grouped thousands first, so "18,000" reads as one price while a written list
# like "24,27,32" still reads as three separate figures.
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

    Numeric requirements compare by magnitude, so "6.9 inches" satisfies "6.9in"
    while "6.9" never satisfies "7.6". Everything else falls back to token
    containment.
    """

    want = _norm(required)
    have = _norm(observed)
    if not want or not have:
        return False
    if want in have:
        return True
    want_numbers = _numbers(want)
    if want_numbers:
        have_numbers = _numbers(have)
        if not have_numbers:
            return False
        for wanted in want_numbers:
            if not any(abs(wanted - found) <= 1e-6 for found in have_numbers):
                return False
        want_units = _word_set(want) - {_number_text(n) for n in want_numbers}
        have_units = _word_set(have) - {_number_text(n) for n in have_numbers}
        return not want_units or not have_units or bool(want_units & have_units)
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
        best_score = 0.0
        best_value: Any = None
        for name, value in self.sources(offer):
            score = _jaccard(wanted, _word_set(name))
            if score > best_score:
                best_score = score
                best_value = value
        return best_value if best_score > 0 else None


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

    def fingerprint(self) -> str:
        return json.dumps(
            {
                "category": self.category,
                "brand": self.brand,
                "firm": self.brand_firm,
                "budget": self.budget,
                "must": self.must,
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
                continue
            parsed = _json_object(reply)
            if parsed is not None:
                return parsed
            self.failures += 1
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


def _pointed_at(text: str) -> set[str]:
    """Item identifiers a customer has quoted back at the assistant.

    Nobody types a shop's internal reference by accident. When a customer does
    quote them, they are pointing at exact items and expect the choice to be
    made among those, so they are picked out of the request and treated as the
    shortlist. Only machine-shaped tokens qualify: a long unbroken run of
    digits, or a paired reference joining a listing to one of its variants.
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

        self.transcript: list[str] = []
        self.unanswered: list[str] = []
        self._asked: set[str] = set()
        self._asked_at = -1
        self.replies = 0
        self.asked_missing = False
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
        self._pointed_at = -1
        self.readings: dict[tuple[str, str], float] = {}
        self.target: Offer | None = None
        self.stage = _VERIFY
        self.staged_key = ""
        self.stage_tries = 0
        self.reserved: set[str] = set()
        self.rejected: set[str] = set()
        self.ordered = False

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
        pending = self._answer_pending() or self._ask_missing()
        room = max(1, self.link.max_calls - len(pending))
        # Shopping around is only worth doing before a choice is made. Once a
        # variant is being bought, the remaining turns belong to checking it.
        # Running short of turns is no reason to skip the shop entirely: with
        # nothing seen there is nothing to buy, so the first look always happens.
        if self.target is None and (not self._must_commit() or not self._candidates()):
            batch, note = self._explore(room)
            if batch:
                return pending + batch, note
        if self.target is None:
            self._select()
        if self.target is None:
            return (
                (pending, "waiting for something worth buying") if pending else ([], "")
            )
        return self._commit(pending)

    # -- shopping around ---------------------------------------------------- #

    def _explore(self, room: int) -> tuple[list[Action], str]:
        """Read what already looks promising, and widen the net when it does not.

        The two halves feed each other: a sweep turns up listings worth opening,
        and opening them reveals whether the sweep found the right shelf at all.
        The loop ends when something meets every firm requirement, or when the
        catalogue has been asked in every way worth asking.
        """

        batch = self._fetch_pointed(room)
        if batch:
            return batch, "opening the listings the customer pointed at"
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
        if not wanted:
            return []
        batch = [Action(detail, {id_slot: item}) for item in wanted][: max(1, room)]
        self.opened.update(
            product_id for action in batch for product_id in _opened_ids(action)
        )
        return batch

    def _settled(self) -> bool:
        """Is there an option in hand that meets everything they insisted on?"""

        return bool(self._compliant())

    # -- discovery ---------------------------------------------------------- #

    def _sweep(self, room: int) -> list[Action]:
        """One pass over the shop, precise instruments first.

        Facet filters are asked first because they are answered on the shop's
        own terms: whatever it calls this category, a filter for it returns
        stock that genuinely qualifies. Typed phrases are guesses at somebody
        else's vocabulary, so only a few are ever spent.
        """

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
        plans.append({"max_price": budget, "in_stock": True})
        plans.append({"in_stock": True})
        return plans

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
        return [brand for brand, _count in ranked[:_BRAND_SWEEPS]]

    def _phrases(self) -> list[str]:
        """Search phrases still worth typing.

        One phrase is always kept back on the first pass. Until some results
        come in there is no way to know what this shop calls the thing, and a
        phrase spent guessing is a phrase not available to spend knowing.
        """

        left = _MAX_SEARCH_PHRASES - len(self.phrases)
        if left <= 0:
            return []
        room = left if self.sweeps > 1 else max(1, left - 1)
        planned = [
            phrase for phrase in self.brief.queries if _norm(phrase) not in self.spent
        ]
        if not planned and self.sweeps > 1:
            planned = self._reword()
        if not planned:
            planned = self._plain_phrases()
        return planned[:room]

    def _plain_phrases(self) -> list[str]:
        """What to type when nothing better has been suggested."""

        category = self.brief.category or " ".join(_words(self.request)[:8])
        seeds = [category]
        if self.brief.brand:
            seeds.append(f"{self.brief.brand} {category}")
        for item in self.brief.must[:1]:
            seeds.append(f"{_text(item['value'])} {category}")
        return [seed.strip() for seed in seeds if seed.strip()]

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
        unseen = list(dict.fromkeys(unseen))[: max(0, _MAX_OPENED - len(self.opened))]
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
        self.opened.update(
            product_id
            for action in batch
            for product_id in _opened_ids(action)
            if product_id
        )
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
            return pending, "waiting for a usable candidate"
        if self.staged_key != target.key:
            self.staged_key = target.key
            self._set_stage(
                _RESERVE_ITEM if target.read_at >= self.world.clock else _VERIFY
            )
        # A stage that cannot make progress must not hold the episode hostage;
        # after two attempts the sequence moves on regardless.
        self.stage_tries += 1
        if self.stage_tries > 2:
            self._set_stage(min(self.stage + 1, _ORDER))

        if self.stage == _VERIFY:
            reads = self._read_target(target)
            if reads:
                return pending + reads, "verifying the chosen variant"
            self._set_stage(_RESERVE_ITEM)
        if self.stage == _RESERVE_ITEM:
            cart_add = self.tools.tool("cart_add")
            if cart_add and target.key not in self.reserved:
                return (
                    pending + [Action(cart_add, self._variant_args(cart_add, target))],
                    "reserving the chosen variant",
                )
            self._set_stage(_RECHECK)
        if self.stage == _RECHECK:
            # Committing can move the market, and on the harder settings it does
            # so silently: always re-read the variant before paying for it.
            reads = self._read_target(target)
            if reads:
                return pending + reads, "re-checking the committed variant"
            self._set_stage(_ORDER)
        if pending:
            return pending, "answering the customer before ordering"
        waiting = pending or self._read_target(target)
        if waiting and self._awaiting_answer():
            return waiting, "waiting on the customer's answer"
        explain = self._explain(target)
        if explain:
            return explain, "telling the customer why this one"
        commit = self.tools.tool("commit")
        if not commit:
            return [], ""
        self.ordered = True
        return [Action(commit, self._order_args(commit, target))], "placing the order"

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
        allowed = slot["allowed"] or (
            [item["field"] for item in self.brief.must + self.brief.nice]
            + [name for name, _canonical in observed]
        )
        evidence = _norm(self.world.evidence(offer))
        required_fields = {_norm(item["field"]) for item in self.brief.must}
        canonical = {_norm(name) for name, is_canonical in observed if is_canonical}
        scored: list[tuple[int, str, str]] = []
        for field in dict.fromkeys(allowed):
            live = self._live_value(offer, field)
            value = live if live is not None else self._grounded_value(offer, field)
            if value is None:
                continue
            rendered = _text(value).strip()
            if not rendered or _norm(rendered) not in evidence:
                continue
            weight = (
                4
                if live is not None
                else 3
                if _norm(field) in required_fields
                else 2
                if _norm(field) in canonical
                else 1
            )
            scored.append((weight, field, rendered))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [(field, value) for _weight, field, value in scored[:_MAX_CLAIMS]]

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
        listing headed "27 Inch 1080P 165Hz Gaming Monitor" says plenty, but
        none of it is the resolution field — so a title that merely mentions
        the value is not accepted as a stand-in for it. Only a named fact, an
        identity field, or the customer's own wording found in the raw facts
        will do; anything else is left unclaimed.
        """

        product = self.world.listing(offer)
        if _norm(field) == "brand":
            return product.brand or offer.brand or None
        value = self.world.attribute(offer, field)
        if isinstance(value, (list, tuple)):
            value = value[-1] if value else None
        if self._atomic(value, offer):
            return value
        facts = _norm(product.facts)
        for item in self.brief.must + self.brief.nice:
            if _norm(item["field"]) != _norm(field):
                continue
            stated = _text(item["value"]).strip()
            if stated and _norm(stated) in facts:
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
        return False

    def _compliant(self) -> list[Offer]:
        return [
            offer
            for offer in self._candidates()
            if self._satisfies(offer) and self.world.listing(offer).studied
        ]

    def _rank(self, offer: Offer) -> tuple:
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
        return (
            self._misses(offer),
            # Ranked below the customer's stated requirements deliberately: an
            # item they asked for is theirs to buy however oddly it is priced,
            # and this only settles which of two equally-qualified listings is
            # really the product rather than something that fits it.
            0 if self._right_class(offer) else 1,
            tuple(priority),
            -brand_hit,
            -likes,
            -self._relevance(offer),
            0 if product.studied else 1,
            offer.price if offer.price is not None else float("inf"),
            offer.key,
        )

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
        return offer.price >= rate * _CLASS_PRICE_FLOOR

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
            self._rate = prices[int(len(prices) * 0.75)] if prices else 0.0
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

        A results page gives a title, a brand, a price and whether it is in
        stock, and that is enough to skip what is plainly out of stock, plainly
        too dear, plainly the wrong make, or plainly not the kind of thing being
        shopped for. Opening a listing costs attention and, for anyone reviewing
        the trip afterwards, says "this looked right to me" — so a short list of
        real candidates beats a page of near-misses.
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
        if not self._asked_words():
            return True
        # Judged against the best on offer rather than a fixed bar: when every
        # title is a weak match there is nothing better to hold out for, and
        # when several are strong the stragglers are not worth the trip.
        best = self._best_echo()
        if best <= 0.0:
            return True
        return self._relevance(offer) >= best * _RELEVANCE_FLOOR

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

        Product copy advertises ranges and headline-plus-boost figures rather
        than the single number a comparison wants. The one that counts is the
        end of the range being shopped towards, so a customer after the highest
        reads the largest figure printed and one after the lowest the smallest.
        """

        recorded = self.readings.get((offer.key, field), _UNSET)
        if recorded is not _UNSET:
            return recorded  # type: ignore[return-value]
        observed = self.world.attribute(offer, field)
        found = _numbers(observed) if observed is not None else []
        if not found:
            return None
        return max(found) if direction == "max" else min(found)

    def _screened(self) -> list[Offer]:
        """Everything still worth attention, most promising first."""

        return sorted(
            (offer for offer in self._candidates() if self._worth_opening(offer)),
            key=self._rank,
        )

    def _select(self) -> None:
        """Choose the variant to buy: code ranks, the model breaks near-ties."""

        pool = sorted(self._candidates(), key=self._rank)
        if not pool:
            return
        studied = self._compliant()
        if not studied:
            # Nothing is fully inspected yet; hold the best guess so the commit
            # reserve always has something to fall back on.
            self.target = pool[0]
            return
        studied.sort(key=self._rank)
        if self.brief.priorities:
            self._read_ranked_attributes(studied[:8])
            studied.sort(key=self._rank)
            self.target = studied[0]
            return
        self.target = self._adjudicate(studied[:6]) or studied[0]

    def _adjudicate(self, pool: list[Offer]) -> Offer | None:
        if len(pool) < 2:
            return pool[0] if pool else None
        index = {offer.key: offer for offer in pool}
        answer = self.reasoner.ask(
            "Decide which of these to buy for the customer. Every one of them "
            "already meets the things they insisted on, so this is the part a "
            "good shopper earns their keep on: which one would actually suit "
            "the person who wrote that request, judging by what the shop says "
            "about each and by what they told you they care about. Cheapest is "
            "not automatically best, and neither is dearest.\n"
            'Answer as {"choice": "<the key of the one to buy>", "reason": '
            '"<one sentence a customer would find convincing>"}.',
            {
                "customer_request": self.request[:2500],
                "what_they_asked_for": self.brief.describe(),
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
            "figure exactly as it is printed, including any headline-plus-boost "
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
            for field in fields:
                raw = values.get(field)
                if raw is None or _norm(raw) not in evidence:
                    continue
                found = _numbers(raw)
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
        if not self.brief.wants_reasons:
            self.brief.wants_reasons = bool(
                _word_set(self.request)
                & {"reason", "why", "justify", "justification", "explain"}
            )

    def _reread_brief(self) -> None:
        """The shopper may narrow the brief mid-episode; re-read it in full."""

        before = self.brief.fingerprint()
        self._read_brief()
        if self.brief.fingerprint() != before:
            self._drop_target()

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
            self.world.ingest(step.get("observation"), studied=action.name in studying)
            refused = bool(step.get("error"))
            if action.name == committing:
                self._settle_order(refused)
            elif refused:
                self._handle_refusal(action)
            elif action.name == reserving and self.staged_key:
                self.reserved.add(self.staged_key)
        self._hear_shopper(result.get("user_message"))
        self._check_budget()
        self._react_to_market()
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
        """Re-plan whenever a variant we care about moves under us."""

        moved = set(self.world.moved)
        self.world.moved = []
        target = self.target
        if target is None or target.key not in moved:
            return
        if not self._viable(target):
            self.rejected.add(target.key)
            self._replace(target)

    def _replace(self, lost: Offer) -> None:
        """Relax as little as possible: same listing first, then nearest match."""

        self._drop_target()
        pool = [offer for offer in self._candidates() if self._satisfies(offer)]
        if not pool:
            pool = self._candidates()
        if not pool:
            return
        lost_words = _word_set(self.world.title_of(lost))

        def distance(offer: Offer) -> tuple:
            if offer.product_id == lost.product_id:
                step = 0
            elif (
                _jaccard(lost_words, _word_set(self.world.title_of(offer)))
                >= _TWIN_SIMILARITY
            ):
                step = 1
            else:
                step = 2
            return (step,) + self._rank(offer)

        self.target = min(pool, key=distance)

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

    proxy = ProxyClient(timeout=120, max_retries=3)
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
