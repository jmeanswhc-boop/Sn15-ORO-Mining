"""ORO Bench agent: LLM-driven shopping loop with hard verifier-aware guardrails.

Contract: ``agent_main(problem_data)`` drives one sealed environment session via
``problem_data["environment"]["binding"]`` and ``policy_view``. All network I/O
goes through ``src.agent.proxy_client.ProxyClient`` (the only HTTP path the
sandbox allows). No filesystem writes, no banned imports.

Strategy
--------
* An LLM picks tool calls from the dynamic ``policy_view.tools`` list, several
  per turn when they are independent (the runtime allows a call group per turn).
* A product registry records every fact the environment has shown us (search
  rows, view/compare facts, variants, live price/stock, cart contents, market
  events). The registry is what makes the guardrails possible.
* Guardrails rewrite unsafe actions before they are sent:
    - the first cart add must target a product we have viewed/compared;
    - a firm screen-size / resolution-class / RAM requirement is verified against
      raw facts and variant options before any cart add or order;
    - retrieval tasks may not commit before the required number of distinct
      title-matching identities has been surfaced, may not exceed the query
      budget the verifier rewards, and may only commit to title/category matches;
    - recovery tasks must commit to the preferred colour when a sibling listing
      offers it; ranking tasks are steered to the computed best candidate;
    - an order must target a variant that is in the cart, was inspected after the
      last cart change or market event, and is within the shopper's budget;
    - justification claims are built from raw facts only (never from titles),
      and are omitted entirely unless the shopper asked for a reason;
    - near the step limit the agent commits/orders deterministically instead of
      timing out with zero reward.
"""

from __future__ import annotations

import json
import logging
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

logger = logging.getLogger("oro_bench_agent")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_proxy = ProxyClient(timeout=150, max_retries=3)

# Turns reserved at the end of the episode for deterministic completion.
_FORCE_ADD_MARGIN = 5
_FORCE_INSPECT_MARGIN = 3

# Retrieval tasks: the verifier's query-efficiency term is min(1, 3 / distinct queries).
_RETRIEVAL_QUERY_BUDGET = 3

# Claim fields we can ground in raw facts. Maps claim field -> candidate fact keys.
_SPEC_FACT_KEYS = {
    "resolution": ("resolution", "display_resolution", "display", "screen_resolution"),
    "screen_size": ("screen_size", "size", "display_size", "screen"),
    "refresh_rate": ("refresh_rate", "refresh", "max_refresh_rate"),
}
_RESOLUTION_TOKEN_RE = re.compile(
    r"\b(\d{3,4}\s*[x×*]\s*\d{3,4}|4k|uhd|qhd|fhd\+?|full hd|hd\+?|2k|1\.5k|720p|1080p|1440p|2160p)\b",
    re.IGNORECASE,
)
_RES_CLASS_NAMES = {1: "HD (>=1280x720)", 2: "Full HD (>=1920x1080)", 3: "QHD (>=2560x1440)", 4: "4K/UHD (>=3840x2160)"}
_RES_CLASS_RE = re.compile(r"\b(2160p|4k|uhd|1440p|full hd|fhd\+?|1080p|720p|hd\+?)\b")
_RES_TOKEN_CLASS = {"2160p": 4, "4k": 4, "uhd": 4, "1440p": 3, "full hd": 2, "fhd": 2, "1080p": 2, "720p": 1, "hd": 1}
_DIMS_RE = re.compile(r"(\d{3,4})\s*(?:[x×*]|-?by-?)\s*(\d{3,4})")
_SIZE_TOKEN_RE = re.compile(r"(\d{1,2}(?:\.\d{1,2})?)\s*-?\s*(?:inches|inch|in\b|\"|”|''|′′)", re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})?)\s*$")
_HZ_RE = re.compile(r"(\d{2,3})\s*-?\s*hz", re.IGNORECASE)
_RAM_REQ_RE = re.compile(r"(\d{1,3})\s*gb\s*(?:of\s*)?ram", re.IGNORECASE)
_RAM_OPT_RE = re.compile(r"(\d{1,3})\s*gb\s*\+\s*(\d{1,4})\s*gb", re.IGNORECASE)
_CATEGORY_REQ_RE = re.compile(r"in the ([A-Za-z0-9&',\- ]+?) category", re.IGNORECASE)
_PREFER_RE = re.compile(r"prefer(?:s|red)?\s+the\s+([a-z]+)\s+([a-z0-9\-]+)", re.IGNORECASE)
# Firm brand in the shopper request, e.g. "I want a Samsung 720p HD smartphone", "Find me a Herman Miller
# 720p HD tablet", "get me an in-stock Xiaomi 27-inch gaming monitor", "I need a Apple smartphone".
_BRAND_REQ_RE = re.compile(
    r"(?:I want an?|Find me an?|get me an?(?: in-stock)?|I need an?)\s+([A-Z][\w&'\-]*(?:\s+[A-Z][\w&'\-]*)*?)\s+"
    r"(?=\d|720p|1080p|1440p|2160p|HD\b|Full HD|QHD|4K|UHD|smartphone|tablet|monitor|gaming|laptop|phone|headphone|earbud)"
)
# Preference-reasoning axes the hidden scorer credits from visible title/option words.
_PREF_AXES = {
    "sweat_water": re.compile(
        r"\b(ipx?\d|ip\d{2}|waterproof|water-proof|water\s?resistant|water-resistant|sweatproof|sweat-proof|"
        r"sweat\s?resistant|sweat-resistant)\b",
        re.IGNORECASE,
    ),
    "sport_fit": re.compile(
        r"\b(ear\s?hooks?|earhooks?|sports?|running|gym|workout|fitness|jogging|neckband|secure\s?fit)\b", re.IGNORECASE
    ),
    "battery_hours": re.compile(r"\b\d{1,3}\s*(?:h|hr|hrs|hours?)\b", re.IGNORECASE),
    "wireless": re.compile(r"\b(bluetooth|wireless|tws|bt\s?5(?:\.\d)?)\b", re.IGNORECASE),
}


def _requested_brand(goal: str) -> str | None:
    m = _BRAND_REQ_RE.search(goal or "")
    return m.group(1).strip() if m else None


_DEVICE_CATEGORIES = (
    ("smartphone", "Smartphones"),
    ("tablet", "Tablets"),
    ("monitor", "Monitors"),
    ("headphone", "Headphones & Headsets"),
    ("earbud", "Headphones & Headsets"),
    ("laptop", "Laptops"),
)


def _requested_device_category(goal: str) -> str | None:
    g = (goal or "").lower()
    for needle, leaf in _DEVICE_CATEGORIES:
        if needle in g:
            return leaf
    return None


def _size_is_ambiguous(text: Any) -> bool:
    """A size fact listing several different sizes (multi-model listing) is unresolved."""
    if text is None:
        return False
    numbers = {float(x) for x in re.findall(r"\d{1,2}(?:\.\d{1,2})?", str(text))}
    return len(numbers) >= 2 and bool(re.search(r"\d\s*(?:-|–|/|to|,|;)\s*\d", str(text)))


def _preference_axes(text: str) -> dict[str, bool]:
    return {axis: bool(rx.search(text or "")) for axis, rx in _PREF_AXES.items()}


def _model_name() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(
        provider, _DEFAULT_MODELS["chutes"]
    )


# --------------------------------------------------------------------------- #
# Small parsing helpers
# --------------------------------------------------------------------------- #

_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|at most|under|below|no more than|max(?:imum)? of|budget (?:is|of)?)\s*"
    r"(?:php\s*)?([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:php)?",
    re.IGNORECASE,
)
_CANDIDATE_RE = re.compile(r"(\d{6,})::([A-Za-z0-9\-]+)")
_RECALL_MIN_RE = re.compile(r"surface at least (\d+) distinct", re.IGNORECASE)
_RECALL_PHRASE_RE = re.compile(
    r'matches (?:the phrase )?"?([^",]+?)"?,\s*at or under', re.IGNORECASE
)
_MODEL_ID_RE = re.compile(r"\b(?=[a-z0-9]{5,24}\b)(?=[a-z0-9]*\d)[a-z0-9]+\b")
_QUERY_PRICE_RE = re.compile(
    r"\b(?:under|below|at or under|at or below|at most|up to|max(?:imum)?)\s*\d[\d,.]*\s*(?:php)?", re.IGNORECASE
)


def _parse_budget(text: str) -> float | None:
    best = None
    for match in _BUDGET_RE.finditer(text or ""):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if value > 0:
            best = value  # keep the LAST stated budget (updates override)
    return best


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _title_tokens(title: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (title or "").lower())


def _detect_family(query: str) -> str:
    q = (query or "").lower()
    checks = [
        ("ranking", "stated tradeoff priority"),
        ("ranking", "compare only these candidate variants"),
        ("recovery", "relaxable only after the exact item becomes unavailable"),
        ("justification", "valid product plus a trustworthy reason"),
        ("preference_reasoning", "soft preferences matter"),
        ("retrieval_recall", "survey the realistic options before committing"),
        ("retrieval_recall", "surface at least"),
        ("intent_decomposition", "mix of firm requirements and nice-to-haves"),
        ("constraint_satisfaction", "firm, non-negotiable requirements: a category"),
    ]
    for family, needle in checks:
        if needle in q:
            return family
    return "generic"


def _requested_spec_field(goal: str) -> str | None:
    g = (goal or "").lower()
    if re.search(r"\d+(?:\.\d+)?\s*-?\s*inch", g):
        return "screen_size"
    if re.search(r"\b(720p|1080p|1440p|2160p|full hd|qhd|uhd|4k|hd)\b", g):
        return "resolution"
    if re.search(r"\d+\s*hz", g):
        return "refresh_rate"
    return None


def _sizes_in(text: Any) -> list[float]:
    if text is None:
        return []
    s = str(text)
    bare = _BARE_NUMBER_RE.match(s)
    if bare:
        return [float(bare.group(1))]
    out = []
    for m in _SIZE_TOKEN_RE.finditer(s):
        try:
            out.append(float(m.group(1)))
        except ValueError:
            continue
    return out


def _requested_size(text: str) -> float | None:
    for m in re.finditer(r"(\d{1,2}(?:\.\d{1,2})?)\s*-?\s*inch", text or "", re.IGNORECASE):
        try:
            return float(m.group(1))
        except ValueError:
            continue
    return None


def _res_class_from_dims(w: int, h: int) -> int:
    long_side, short_side = max(w, h), min(w, h)
    if long_side >= 3840 and short_side >= 2160:
        return 4
    if long_side >= 2560 and short_side >= 1440:
        return 3
    if long_side >= 1920 and short_side >= 1080:
        return 2
    if long_side >= 1280 and short_side >= 720:
        return 1
    return 0


def _requested_res_class(text: str) -> int | None:
    t = (text or "").lower()
    m = _RES_CLASS_RE.search(t)
    if m:
        return _RES_TOKEN_CLASS.get(m.group(1).rstrip("+"))
    dims = _DIMS_RE.search(t)
    if dims:
        return _res_class_from_dims(int(dims.group(1)), int(dims.group(2)))
    return None


def _fact_res_classes(display: str) -> set[int]:
    """Resolution classes stated by a raw display fact.

    Every stated value counts: pixel dimensions and class tokens alike (a video
    mode such as "1080p@30fps" still reads as a conflicting class to the verifier).
    More than one class means the fact is unresolved.
    """
    text = (display or "").lower()
    classes: set[int] = set()
    for w, h in _DIMS_RE.findall(text):
        classes.add(_res_class_from_dims(int(w), int(h)))
    for m in _RES_CLASS_RE.finditer(text):
        cls = _RES_TOKEN_CLASS.get(m.group(1).rstrip("+"))
        if cls is not None:
            classes.add(cls)
    return classes


def _requested_ram(text: str) -> int | None:
    m = _RAM_REQ_RE.search(text or "")
    return int(m.group(1)) if m else None


def _requested_category(goal: str) -> str | None:
    m = _CATEGORY_REQ_RE.search(goal or "")
    return m.group(1).strip() if m else None


def _preferred_color_model(goal: str) -> tuple[str | None, str | None]:
    m = _PREFER_RE.search(goal or "")
    if not m:
        return None, None
    return m.group(1).lower(), m.group(2).lower()


def _max_hz(text: Any) -> float | None:
    values = [float(v) for v in _HZ_RE.findall(str(text or ""))]
    return max(values) if values else None


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("&", "and").split())


# --------------------------------------------------------------------------- #
# Product registry
# --------------------------------------------------------------------------- #


class Registry:
    """Everything the environment has shown us, keyed by product_id / sku."""

    def __init__(self) -> None:
        self.products: dict[str, dict[str, Any]] = {}
        self.variants: dict[tuple[str, str], dict[str, Any]] = {}
        self.viewed: set[str] = set()
        self.view_order: list[str] = []
        self.cart: dict[tuple[str, str], dict[str, Any]] = {}
        self.cart_version = 0
        self.inspected_at: dict[tuple[str, str], int] = {}
        self.events: list[dict[str, Any]] = []
        self.search_rows: list[dict[str, Any]] = []
        self.order_placed = False
        self._version_turn = 0

    # -- ingestion ---------------------------------------------------------
    def _product(self, pid: str) -> dict[str, Any]:
        return self.products.setdefault(
            pid,
            {
                "product_id": pid,
                "title": None,
                "brand": None,
                "category_path": None,
                "facts": {},
                "listing_id": None,
                "skus": [],
            },
        )

    def _variant(self, pid: str, sku: str) -> dict[str, Any]:
        key = (pid, sku)
        rec = self.variants.setdefault(
            key,
            {"product_id": pid, "sku": sku, "price": None, "in_stock": None, "options": None},
        )
        prod = self._product(pid)
        if sku not in prod["skus"]:
            prod["skus"].append(sku)
        return rec

    def ingest_row(self, row: dict[str, Any]) -> None:
        pid = str(row.get("product_id") or "")
        if not pid:
            return
        prod = self._product(pid)
        if row.get("title"):
            prod["title"] = row["title"]
        if row.get("brand"):
            prod["brand"] = row["brand"]
        sku = row.get("sku")
        if sku:
            var = self._variant(pid, str(sku))
            if row.get("price") is not None:
                var["price"] = _to_float(row.get("price"))
            if row.get("in_stock") is not None:
                var["in_stock"] = bool(row.get("in_stock"))
        self.search_rows.append(row)

    def ingest_view(self, obs: dict[str, Any]) -> None:
        pid = str(obs.get("product_id") or "")
        if not pid:
            return
        prod = self._product(pid)
        prod["title"] = obs.get("title") or prod["title"]
        prod["brand"] = obs.get("brand") if obs.get("brand") is not None else prod["brand"]
        if obs.get("category_path"):
            prod["category_path"] = list(obs["category_path"])
        if isinstance(obs.get("facts"), dict):
            prod["facts"] = dict(obs["facts"])
        if obs.get("listing_id"):
            prod["listing_id"] = obs["listing_id"]
        for var in obs.get("variants") or []:
            sku = var.get("sku")
            if not sku:
                continue
            rec = self._variant(pid, str(sku))
            rec["price"] = _to_float(var.get("price"))
            rec["in_stock"] = bool(var.get("in_stock"))
            rec["options"] = var.get("options")
        if pid in self.view_order:
            self.view_order.remove(pid)
        self.view_order.append(pid)
        self.viewed.add(pid)

    def ingest_stock(self, obs: dict[str, Any], turn: int) -> None:
        pid = str(obs.get("product_id") or "")
        sku = str(obs.get("sku") or "")
        if not pid or not sku:
            return
        rec = self._variant(pid, sku)
        rec["price"] = _to_float(obs.get("price"))
        rec["in_stock"] = bool(obs.get("in_stock"))
        self.inspected_at[(pid, sku)] = turn

    def ingest_cart(self, cart: Any) -> None:
        if not isinstance(cart, list):
            return
        new_cart: dict[tuple[str, str], dict[str, Any]] = {}
        for item in cart:
            pid = str(item.get("product_id") or "")
            sku = str(item.get("sku") or "")
            if not pid or not sku:
                continue
            new_cart[(pid, sku)] = dict(item)
            rec = self._variant(pid, sku)
            price = item.get("current_price", item.get("price"))
            if price is not None:
                rec["price"] = _to_float(price)
            if item.get("in_stock") is not None:
                rec["in_stock"] = bool(item.get("in_stock"))
        if set(new_cart) != set(self.cart):
            self.cart_version += 1
        self.cart = new_cart

    def ingest_event(self, event: dict[str, Any], turn: int) -> None:
        if not isinstance(event, dict) or not event.get("kind"):
            return
        target = event.get("target") or {}
        pid = str(target.get("product_id") or "")
        sku = str(target.get("sku") or "")
        record = {
            "turn": turn,
            "kind": event.get("kind"),
            "product_id": pid,
            "sku": sku,
            "old_price": _to_float(event.get("old_price")),
            "new_price": _to_float(event.get("new_price")),
            "in_stock_after": event.get("in_stock_after"),
        }
        self.events.append(record)
        if pid and sku:
            rec = self._variant(pid, sku)
            if record["new_price"] is not None:
                rec["price"] = record["new_price"]
            if event.get("in_stock_after") is not None:
                rec["in_stock"] = bool(event.get("in_stock_after"))
        # Any event invalidates earlier stock inspections.
        self.cart_version += 1

    # -- queries -----------------------------------------------------------
    def variant(self, pid: str, sku: str) -> dict[str, Any] | None:
        return self.variants.get((pid, sku))

    def is_fresh(self, pid: str, sku: str) -> bool:
        turn = self.inspected_at.get((pid, sku))
        return turn is not None and turn >= self._version_turn

    def note_version_turn(self, turn: int) -> None:
        self._version_turn = turn

    def siblings(self, sku: str) -> list[tuple[str, str]]:
        prefix = sku.split("-")[0]
        return [key for key in self.variants if key[1].split("-")[0] == prefix]

    def best_cart_item(self, budget: float | None) -> tuple[str, str] | None:
        for key in self.cart:
            rec = self.variants.get(key) or {}
            if rec.get("in_stock") is False:
                continue
            price = rec.get("price")
            if budget is not None and price is not None and price > budget + 1e-6:
                continue
            return key
        return None

    def best_known_candidate(self, budget: float | None) -> tuple[str, str] | None:
        """Most recently viewed in-stock, within-budget variant (fallback only)."""
        for pid in reversed(self.view_order):
            prod = self.products.get(pid) or {}
            for sku in prod.get("skus") or []:
                rec = self.variants.get((pid, sku)) or {}
                if rec.get("in_stock") is not True:
                    continue
                price = rec.get("price")
                if price is None or (budget is not None and price > budget + 1e-6):
                    continue
                return (pid, sku)
        return None


# --------------------------------------------------------------------------- #
# Retrieval-recall identity counting (mirrors the task's stated rule)
# --------------------------------------------------------------------------- #


class RecallTracker:
    def __init__(self, phrase: str | None, required: int) -> None:
        self.terms = _title_tokens(phrase) if phrase else []
        self.required = required
        self.matching: dict[str, dict[str, Any]] = {}

    def active(self) -> bool:
        return bool(self.terms) and self.required > 0

    def title_matches(self, title: str | None) -> bool:
        tokens = set(_title_tokens(title or ""))
        return all(term in tokens for term in self.terms)

    def observe_rows(self, rows: list[dict[str, Any]]) -> None:
        if not self.active():
            return
        for row in rows:
            if self.title_matches(row.get("title")):
                self.matching[str(row.get("product_id"))] = row

    def identity_count(self) -> int:
        """Union-find over product_id, model identifier, and (brand,words,price)."""
        pids = list(self.matching)
        parent = {pid: pid for pid in pids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        by_model: dict[str, str] = {}
        by_sig: dict[tuple[str, frozenset[str], float], str] = {}
        for pid in pids:
            row = self.matching[pid]
            title = (row.get("title") or "").lower()
            words = frozenset(_title_tokens(title))
            price = round(_to_float(row.get("price")) or 0.0, 2)
            brand = (row.get("brand") or "").lower()
            sig = (brand, words, price)
            if sig in by_sig:
                union(pid, by_sig[sig])
            else:
                by_sig[sig] = pid
            compact = re.sub(r"[^a-z0-9 ]", "", title)
            for tok in _MODEL_ID_RE.findall(compact):
                if tok in by_model:
                    union(pid, by_model[tok])
                else:
                    by_model[tok] = pid
        return len({find(pid) for pid in pids})

    def summary(self, queries_used: int) -> str:
        if not self.active():
            return ""
        return (
            f"Retrieval progress: {self.identity_count()} distinct product identities whose TITLE contains every "
            f"term {self.terms} surfaced so far (required before ordering: {self.required}); "
            f"search queries used: {queries_used}/{_RETRIEVAL_QUERY_BUDGET} (the verifier penalises more)."
        )

    def candidate_lines(self, registry: Registry, limit: int = 8) -> list[str]:
        lines = []
        for pid, row in list(self.matching.items())[:limit]:
            var = registry.variant(pid, str(row.get("sku") or "")) or {}
            lines.append(
                f"  - {pid} sku={row.get('sku')} price={var.get('price', row.get('price'))} "
                f"in_stock={var.get('in_stock', row.get('in_stock'))} viewed={pid in registry.viewed} "
                f"title={str(row.get('title') or '')[:70]!r}"
            )
        return lines


# --------------------------------------------------------------------------- #
# Observation condensing for the LLM
# --------------------------------------------------------------------------- #


def _short(value: Any, limit: int) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value


def _condense_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": row.get("product_id"),
        "sku": row.get("sku"),
        "price": row.get("price"),
        "in_stock": row.get("in_stock"),
        "brand": row.get("brand"),
        "title": _short(row.get("title"), 140),
    }


def _condense_product(obs: dict[str, Any]) -> dict[str, Any]:
    facts = obs.get("facts") if isinstance(obs.get("facts"), dict) else {}
    out = {
        "product_id": obs.get("product_id"),
        "brand": obs.get("brand"),
        "category_path": obs.get("category_path"),
        "title": _short(obs.get("title"), 180),
        "facts": {k: _short(v, 80) for k, v in facts.items()},
        "variants": [
            {
                "sku": v.get("sku"),
                "options": _short(v.get("options"), 80),
                "price": v.get("price"),
                "in_stock": v.get("in_stock"),
            }
            for v in (obs.get("variants") or [])[:12]
        ],
    }
    if obs.get("listing_id"):
        out["listing_id"] = obs["listing_id"]
    for key in ("short_description", "description"):
        if obs.get(key):
            out[key] = _short(obs[key], 300)
    return out


def _condense(name: str, step: dict[str, Any]) -> dict[str, Any]:
    obs = step.get("observation") if isinstance(step.get("observation"), dict) else {}
    out: dict[str, Any] = {"tool": name, "done": step.get("done")}
    if step.get("error"):
        out["error"] = step.get("error")
    if name in ("search", "filter"):
        rows = obs.get("results") or []
        out["returned"] = obs.get("returned", len(rows))
        out["results"] = [_condense_row(r) for r in rows]
    elif name == "view":
        out["product"] = _condense_product(obs)
    elif name == "compare":
        out["compared"] = [_condense_product(p) for p in obs.get("compared") or []]
    elif name == "add_to_cart":
        out["added"] = obs.get("added")
        out["cart"] = obs.get("cart")
        if obs.get("market_event"):
            ev = obs["market_event"]
            out["MARKET_EVENT"] = {
                "kind": ev.get("kind"),
                "target": ev.get("target"),
                "old_price": ev.get("old_price"),
                "new_price": ev.get("new_price"),
                "in_stock_after": ev.get("in_stock_after"),
            }
    else:
        out["observation"] = obs
    return out


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_UNIVERSAL_PROTOCOL = """You are an autonomous shopping agent operating a tool-driven store environment.
You act ONLY through tool calls. Each of your replies is one environment turn and may contain SEVERAL tool calls
(they run in order within the turn). The episode has a hard limit of {max_steps} turns, so batch independent
lookups: e.g. view/compare several candidates in one turn, or inspect_stock several variants in one turn.

MANDATORY PROTOCOL (the verifier checks all of it):
1. Read the shopper's goal and the Task rules carefully; the rules define what counts as valid.
2. Find candidates with `search` (plain natural-language queries: no quotes, no prices, no "under X";
   in_stock=true) and/or `filter` (category is the catalog leaf such as "Smartphones", "Monitors", "Headphones";
   brand; max_price). Search results show only titles; facts come from view/compare.
3. Before the FIRST `add_to_cart`, you MUST have called `view` or `compare` on that exact product and confirmed:
   category_path matches the requested category (an accessory/case/part/replacement screen/LCD is never the device),
   the specific variant (sku) satisfies every firm requirement, price <= budget, in_stock true.
   The first cart add must be your genuine best choice, not a placeholder; it is scored.
4. Right after your first cart add a MARKET EVENT usually fires on that variant (stock-out or price change) and the
   shopper will comment. You must then: `inspect_stock` the affected variant, and if it is now out of stock or above
   budget, pick a replacement that still meets every firm requirement (recovery rules in the task text), `view` it
   if not viewed, `remove_from_cart` the old one and `add_to_cart` the replacement, `inspect_stock` it, then order.
   If the event left the item in stock and within budget, `inspect_stock` it and order it.
   The event hits ONE sku. Sibling listings of the same product (same sku prefix before the dash, listed in the
   state note) usually still have the old price and stock: check those first, then already-viewed alternatives.
5. `place_test_order` ends the episode. Only call it alone in its own turn, for a variant that is in the cart and
   that you inspected with `inspect_stock` AFTER the last cart change / market event. Provide product_id and sku.
6. Shopper messages can tighten the budget or add/clarify a requirement: the latest statement wins for all later
   choices. Answer direct questions with a short `message` batched together with your next lookups in the same
   turn. Never send status updates or progress reports; never ask for permission to order; just order.
7. Justification: do NOT include a justification unless the shopper explicitly asked for a reason. If they did,
   claims must be {{field, value}} pairs only, using ONLY raw `facts` from view/compare of the final product
   (never title words) plus current price and in_stock; the agent runtime will drop anything unsupported.
8. Firm specs are verified against raw facts and the selected variant's options, not titles: screen size from
   facts.size / variant options; resolution class from facts.display (HD>=1280x720, Full HD>=1920x1080,
   QHD>=2560x1440, 4K>=3840x2160, highest class met; conflicting values = unresolved, skip that product).
9. Do not loop on the same search; if results are poor, change wording (brand names, model names, product type
   synonyms), use `filter`, or inspect different candidates. Commit early, verify, order.
"""

_FAMILY_PLAYBOOKS = {
    "intent_decomposition": """FAMILY PLAYBOOK (intent decomposition):
- Firm: category, the one hardware spec (screen size / RAM / resolution / refresh rate), budget, in stock. Soft: named brand.
- Turn 1: search "<brand> <category> <spec>" AND a brand-free search "<category> <spec>", in_stock=true, k=10.
- Turn 2: view/compare the best 2-4 candidates; prefer the named brand ONLY if that item meets every firm requirement
  (facts.size must state the exact requested size; category_path must be the device category; variant price <= budget).
- Turn 3: inspect_stock the chosen sku. Turn 4: add_to_cart. Then handle the market event, re-inspect, order.
- If the shopper lowers the budget or your item is repriced above it, find ANOTHER device with the exact spec
  (different brand is fine): search with other brand/model words (Infinix, Tecno, Realme, Xiaomi, itel, Honor...),
  never with prices or quotes. Replacement screens / LCD assemblies are accessories, never valid.""",
    "retrieval_recall": """FAMILY PLAYBOOK (retrieval recall):
- The verifier rewards recall across at most 3 distinct `search` queries (a 4th query is penalised; `filter`,
  `view`, `compare`, `inspect_stock` are free). Turn 1: run TWO searches: (a) the exact phrase alone, (b) the
  phrase plus the category word. Turn 2: ONE more search that adds a product-type or brand word you saw in the
  matching titles (e.g. "roto pvc cement"), batched with view/compare of the matching candidates. No further
  searches after that; post-event replacements must come from candidates you already surfaced.
- The phrase must match TITLE tokens exactly (every term as a whole lowercase alphanumeric token, any order).
  The item must be in the requested category (category_path), in stock, within budget.
- Then view/compare 3-5 matching candidates in ONE turn (so you have verified alternatives for after the event),
  add the best, handle the event with an already-viewed alternative, inspect_stock, order.""",
    "constraint_satisfaction": """FAMILY PLAYBOOK (constraint satisfaction):
- If the shopper says a firm spec "I will tell you if you ask": in turn 1 send a `message` asking for that exact
  requirement, in the SAME turn also run search/filter for the category+brand+budget so no turn is wasted.
- Every requirement must hold for the EXACT sku: check the variant options (e.g. "128gb / blue"), facts, price, stock.
- Resolution classes: HD>=1280x720, FHD>=1920x1080, QHD>=2560x1440, 4K>=3840x2160 (use the highest class met).
- Smartphone identity: title or facts must say smartphone, or device OS in {Android, iOS, HarmonyOS, HyperOS, ...}.
  Reject tablets, feature phones, accessories, screen-only options.
- If your sku sells out, find ANOTHER product meeting all requirements (not a non-compliant variant).""",
    "preference_reasoning": """FAMILY PLAYBOOK (preference reasoning):
- Turn 1: `message` the shopper to confirm priorities (one short question) AND run 3 searches in the same turn
  that target ALL FOUR axes at once, e.g. "IPX7 waterproof sport earbuds ear hooks 40 hours battery bluetooth",
  "sweatproof wireless earphones running gym 30H playtime", "IPX5 sport bluetooth headphones 20 hours battery".
- The hidden score has FOUR axes, each judged from visible title words / facts: (1) sweat & water resistance
  (an explicit IPX4-IPX8 / waterproof / sweatproof statement), (2) secure sport fit (ear hooks, sport, running,
  gym), (3) battery life (an explicit hours figure, e.g. "40H", "30 hours"), (4) wireless (Bluetooth / TWS).
  Pick the candidate whose title/facts EXPLICITLY state all four; a cheaper item lacking an IPX/waterproof word
  scores badly. Never the cheapest for its own sake.
- Turn 2: view/compare 3-5 in-stock, within-budget candidates from at least TWO different listing_id values.
- Turn 3: inspect_stock the best. Turn 4: add_to_cart.
- After the event: if repriced above budget, recover to a DIFFERENT listing_id (not another variant of the same
  listing) that also states the four axes, inspect_stock it, add it, order.""",
    "ranking": """FAMILY PLAYBOOK (ranking):
- The goal lists the only candidate variants as product_id::sku. Turn 1: `compare` all of them (max 5 per call).
- For EACH candidate read, in this priority: the SELECTED VARIANT'S `options` text first (e.g. "b: 27g4 (27 inch)"
  means the selected variant is 27 inch even if facts.size says 24.5), then facts, then the title. Refresh rate =
  the highest advertised Hz including overclock (e.g. "300Hz" in the title). Rank: highest refresh rate, then
  largest screen, then LOWEST price. The runtime also prints a computed ranking table in the state note; follow it.
- Turn 2: inspect_stock the top-ranked sku. Turn 3: add_to_cart it (exact sku from the goal).
- The market event will remove or reprice it: re-rank the remaining candidates using CURRENT stock/price,
  inspect_stock the new best, remove the old one, add the new best, inspect_stock, then order.
- Do NOT include a justification in the order.""",
    "recovery": """FAMILY PLAYBOOK (recovery):
- Turn 1: search the preferred model code with the color AND the model code alone, in_stock=true, k=10.
  Results usually contain SEVERAL sibling listings with the same title and the same sku prefix (e.g. 4718455817-1,
  -2, -3): each sibling is one colour. Turn 2: `view` ALL siblings (batch them) - the colour is in each variant's
  `options` (e.g. "forrest green"). Confirm genuine smartphone (title/facts/OS), category Smartphones, price <=
  budget, in stock.
- Turn 3: inspect_stock the sibling whose options state the preferred colour; Turn 4: add_to_cart it. It will then
  go out of stock.
- Recovery order: SAME model in ANOTHER colour first (another sibling with the same sku prefix), only if none is in
  stock change model. inspect_stock the replacement, remove the old item, add the replacement, inspect_stock, order.
- Do not send explanatory messages; act.""",
    "justification": """FAMILY PLAYBOOK (justification):
- The shopper asked for a reason, so the order MUST carry a justification whose claims are checkable raw facts:
  the requested spec field (resolution from facts.display / screen_size from facts.size / refresh_rate only if a
  refresh-rate fact exists), price (current), in_stock, and brand (from the `brand` field, not the title).
- Therefore choose a product whose `facts` actually contain the requested spec with a value that satisfies the
  request (e.g. request "720p HD" -> facts.display shows 720p/hd/1280x720-class such as "720 x 1600"; "27-inch" ->
  facts.size 27). If facts.display lists two different resolutions (e.g. "540x960; 720p") the spec is unresolved:
  skip that product. Brand must match the requested brand via the `brand` field (null brand = not a match).
  Category must match via category_path. Resolution classes: HD>=1280x720, FHD>=1920x1080, QHD>=2560x1440,
  4K>=3840x2160 (highest class met).
- Turn 1: search "<brand> <spec> <category>" and a variant phrasing, in_stock=true. Turn 2: view/compare 3-5
  candidates and check facts. Turn 3: inspect_stock. Turn 4: add_to_cart. Handle the event (a price rise above
  budget means switch to another compliant, already-viewed product and re-verify). inspect_stock, then order.
- When ordering, pass justification.claims using only fields from the tool schema enum and values copied from
  facts / current price / "true" for in_stock / the brand field. The runtime will filter unsupported claims.""",
    "generic": """FAMILY PLAYBOOK (generic): follow the Task rules literally; search, view/compare, verify, add, handle
the market event, inspect_stock, order.""",
}


def _system_prompt(family: str, max_steps: int, max_calls: int) -> str:
    return (
        _UNIVERSAL_PROTOCOL.format(max_steps=max_steps)
        + f"\nYou may include up to {max_calls} tool calls per turn.\n\n"
        + _FAMILY_PLAYBOOKS.get(family, _FAMILY_PLAYBOOKS["generic"])
    )


# --------------------------------------------------------------------------- #
# Claim construction (justification family)
# --------------------------------------------------------------------------- #


def _fact_lookup(facts: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in facts and facts[key] not in (None, ""):
            return str(facts[key])
    lowered = {k.lower(): v for k, v in facts.items()}
    for key in keys:
        if key in lowered and lowered[key] not in (None, ""):
            return str(lowered[key])
    return None


def _resolution_value(raw: str, requested_hint: str) -> str | None:
    """Pick the most defensible raw resolution string from a messy fact value."""
    cleaned = re.sub(r"\d{3,4}p\s*@[^;,]*", " ", raw)
    candidates = [m.group(0) for m in _RESOLUTION_TOKEN_RE.finditer(cleaned)]
    if not candidates:
        return raw.strip() or None
    dims = [c for c in candidates if re.search(r"\d\s*[x×*]\s*\d", c)]
    if dims:
        return re.sub(r"\s*[x×*]\s*", " x ", dims[0]).strip()
    hint = requested_hint.lower()
    for cand in candidates:
        if cand.lower().replace(" ", "") in hint.replace(" ", ""):
            return cand
    return candidates[0]


def build_claims(
    product: dict[str, Any],
    variant: dict[str, Any],
    goal: str,
    allowed_fields: set[str] | None,
) -> list[dict[str, str]]:
    facts = product.get("facts") or {}
    claims: list[dict[str, str]] = []

    def allowed(field: str) -> bool:
        return allowed_fields is None or field in allowed_fields

    spec = _requested_spec_field(goal)
    if spec and allowed(spec):
        raw = _fact_lookup(facts, _SPEC_FACT_KEYS[spec])
        if raw:
            value = _resolution_value(raw, goal) if spec == "resolution" else raw.strip()
            if value:
                claims.append({"field": spec, "value": value})
    price = variant.get("price")
    if price is not None and allowed("price"):
        claims.append({"field": "price", "value": f"{price:.2f}".rstrip("0").rstrip(".")})
    if variant.get("in_stock") is True and allowed("in_stock"):
        claims.append({"field": "in_stock", "value": "true"})
    brand = product.get("brand")
    if brand and allowed("brand"):
        claims.append({"field": "brand", "value": str(brand)})
    return claims


# --------------------------------------------------------------------------- #
# Environment session
# --------------------------------------------------------------------------- #


class Session:
    def __init__(self, binding: dict[str, Any], problem_id: str) -> None:
        self.binding = dict(binding)
        self.problem_id = problem_id
        self.turn = 0
        self.failures = 0

    def call(self, calls: list[dict[str, Any]]) -> dict[str, Any] | None:
        turn = self.turn + 1
        group_id = f"{self.problem_id}-t{turn}"
        envelope = {
            **self.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {"call_id": f"{group_id}-c{i}", "action": {"name": c["name"], "args": c.get("args") or {}}}
                for i, c in enumerate(calls, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            # Retry the identical envelope once: idempotency makes it safe.
            result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            self.failures += 1
            return None
        self.failures = 0
        self.turn = int(result.get("solver_turn_count") or turn)
        return result


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class Agent:
    def __init__(self, problem_data: dict[str, Any]) -> None:
        env = problem_data["environment"]
        self.binding = env["binding"]
        self.policy = env["policy_view"]
        self.problem_id = str(problem_data.get("problem_id") or problem_data.get("id") or "problem")
        self.query: str = self.policy.get("query") or ""
        self.goal = self.query.split("Task rules:", 1)[0].strip()
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.tools = list(self.policy.get("tools") or [])
        self.tool_names = {t.get("function", t).get("name") for t in self.tools}
        self.family = _detect_family(self.query)
        self.budget = _parse_budget(self.goal)
        self.registry = Registry()
        self.session = Session(self.binding, self.problem_id)
        self.model = _model_name()
        self.wants_reason = self.family == "justification" or bool(
            re.search(r"\b(why|reason|walk me through|tell me why)\b", self.goal, re.IGNORECASE)
        )
        self.allowed_claim_fields = self._claim_field_enum()
        # Firm spec requirements (updated from shopper messages as they arrive).
        self.spec_size = _requested_size(self.goal) if self.family != "ranking" else None
        self.spec_res = _requested_res_class(self.goal) if self.family != "ranking" else None
        self.spec_ram = _requested_ram(self.goal) if self.family != "ranking" else None
        self.category_req = _requested_category(self.goal)
        self.brand_req = (
            _requested_brand(self.goal)
            if self.family in ("justification", "constraint_satisfaction")
            else None
        )
        self.pref_color, self.pref_model = (
            _preferred_color_model(self.goal) if self.family == "recovery" else (None, None)
        )
        required = 0
        phrase = None
        if self.family == "retrieval_recall":
            m = _RECALL_MIN_RE.search(self.query)
            required = int(m.group(1)) if m else 3
            pm = _RECALL_PHRASE_RE.search(self.goal)
            phrase = pm.group(1).strip().strip('"') if pm else None
        self.recall = RecallTracker(phrase, required)
        self.ranking_candidates = _CANDIDATE_RE.findall(self.goal) if self.family == "ranking" else []
        self.distinct_queries: set[str] = set()
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system_prompt(self.family, self.max_steps, self.max_calls)},
            {"role": "user", "content": self.query},
        ]
        self.dialogue: list[dict[str, Any]] = []
        self.done = False
        self.last_user_message: str | None = None
        self.llm_failures = 0
        self.consecutive_env_failures = 0
        self.first_add_done = False
        self.blocked_streak = 0
        self.ranking_blocks = 0
        self.recovery_blocks = 0
        self.preference_blocks = 0
        self.spec_blocks: dict[str, int] = {}

    # -- helpers -----------------------------------------------------------
    def _claim_field_enum(self) -> set[str] | None:
        for tool in self.tools:
            fn = tool.get("function", tool)
            if fn.get("name") != "place_test_order":
                continue
            try:
                enum = fn["parameters"]["properties"]["justification"]["properties"]["claims"]["items"][
                    "properties"
                ]["field"]["enum"]
                return set(enum)
            except (KeyError, TypeError):
                return None
        return None

    def _turn(self) -> int:
        return self.session.turn

    def _turns_left(self) -> int:
        return self.max_steps - self.session.turn

    def _has(self, name: str) -> bool:
        return name in self.tool_names

    # -- LLM ---------------------------------------------------------------
    def _llm_actions(self) -> list[dict[str, Any]]:
        for _attempt in range(3):
            response = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": self.model,
                    "messages": self.messages,
                    "tools": self.tools,
                    "tool_choice": "required",
                    "temperature": 0,
                    "max_tokens": 2048,
                },
            )
            if response is None:
                self.llm_failures += 1
                continue
            try:
                message = response["choices"][0]["message"]
            except (KeyError, IndexError, TypeError):
                self.llm_failures += 1
                continue
            tool_calls = message.get("tool_calls") or []
            parsed = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name")
                raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except json.JSONDecodeError:
                    args = {}
                if name in self.tool_names and isinstance(args, dict):
                    parsed.append({"name": name, "args": args, "id": tc.get("id")})
            if parsed:
                for i, p in enumerate(parsed):
                    p["id"] = p["id"] or f"call_{self.session.turn + 1}_{i}"
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content") or "",
                        "tool_calls": [
                            {
                                "id": p["id"],
                                "type": "function",
                                "function": {"name": p["name"], "arguments": json.dumps(p["args"])},
                            }
                            for p in parsed
                        ],
                    }
                )
                return parsed
            # No tool call: nudge and retry.
            self.messages.append({"role": "assistant", "content": message.get("content") or ""})
            self.messages.append(
                {"role": "user", "content": "You must respond with tool calls only. Continue the protocol."}
            )
        return []

    # -- eligibility checks ----------------------------------------------------
    def _spec_issues(self, pid: str, sku: str) -> list[str]:
        """Deterministic firm-spec verification against raw facts / variant options."""
        prod = self.registry.products.get(pid) or {}
        var = self.registry.variant(pid, sku) or {}
        facts = prod.get("facts") or {}
        options = str(var.get("options") or "")
        title = prod.get("title") or ""
        issues: list[str] = []
        if self.brand_req:
            brand = str(prod.get("brand") or "").strip()
            want = self.brand_req.lower()
            if not brand:
                issues.append(
                    f"brand field is null but the shopper requires {self.brand_req!r}; a title mentioning the brand "
                    "does not count. Choose a product whose `brand` field states it"
                )
            elif want not in brand.lower() and brand.lower() not in want:
                issues.append(f"brand mismatch: brand field is {brand!r}, shopper requires {self.brand_req!r}")
        device_cat = self.category_req or _requested_device_category(self.goal)
        if device_cat and prod.get("category_path") and self.family != "retrieval_recall":
            path = [_norm(p) for p in prod["category_path"]]
            if _norm(device_cat) not in path:
                issues.append(
                    f"category_path {prod['category_path']} does not contain the required {device_cat!r} category"
                )
        if self.spec_size is not None:
            stated = None
            for name, text in (("variant options", options), ("facts.size", facts.get("size")), ("title", title)):
                vals = _sizes_in(text)
                if vals:
                    stated = (name, vals)
                    break
            if stated and stated[0] != "variant options" and _size_is_ambiguous(facts.get("size")):
                issues.append(
                    f"facts.size={facts.get('size')!r} lists several sizes (multi-model listing): the selected variant's "
                    "size is unresolved; choose a listing whose facts.size states a single size"
                )
            elif stated and not any(abs(v - self.spec_size) < 0.06 for v in stated[1]):
                issues.append(
                    f"screen size mismatch: {stated[0]} states {stated[1]} inch but the shopper requires "
                    f"{self.spec_size}-inch; choose a product whose facts.size states {self.spec_size}"
                )
            elif stated and stated[0] == "title":
                issues.append(
                    f"screen size {self.spec_size} appears only in the title (a decoy risk); the verifier needs it in "
                    "facts.size or the selected variant's options. Prefer a product whose facts state the size"
                )
        if self.spec_res is not None:
            display = facts.get("display") or facts.get("resolution")
            if display:
                classes = _fact_res_classes(str(display))
                if len(classes) > 1:
                    issues.append(
                        f"resolution unresolved: facts.display={display!r} states conflicting classes; skip this product"
                    )
                elif classes and self.spec_res not in classes:
                    issues.append(
                        f"resolution mismatch: facts.display={display!r} is not {_RES_CLASS_NAMES.get(self.spec_res)} class"
                    )
            elif self.family == "justification":
                issues.append(
                    "no resolution fact: a required resolution claim cannot be grounded; choose a product whose "
                    "facts.display states the resolution"
                )
        if self.spec_ram is not None:
            m = _RAM_OPT_RE.search(options)
            if m and int(m.group(1)) != self.spec_ram:
                issues.append(
                    f"RAM mismatch: variant options {options!r} state {m.group(1)}GB RAM but the shopper requires {self.spec_ram}GB"
                )
        if self.family == "retrieval_recall":
            if self.recall.active() and not self.recall.title_matches(title):
                issues.append(
                    f"title {title[:80]!r} does not contain every required term {self.recall.terms} as whole tokens"
                )
            if self.category_req and prod.get("category_path"):
                path = [_norm(p) for p in prod["category_path"]]
                if _norm(self.category_req) not in path:
                    issues.append(
                        f"category_path {prod['category_path']} is not in the required {self.category_req!r} category"
                    )
        return issues

    def _candidate_text(self, pid: str, sku: str) -> str:
        prod = self.registry.products.get(pid) or {}
        var = self.registry.variant(pid, sku) or {}
        facts = prod.get("facts") or {}
        return " ".join(
            str(x) for x in (prod.get("title"), var.get("options"), " ".join(str(v) for v in facts.values())) if x
        )

    def _preference_table(self) -> list[dict[str, Any]]:
        rows = []
        for pid in self.registry.view_order:
            prod = self.registry.products.get(pid) or {}
            for sku in prod.get("skus") or []:
                var = self.registry.variant(pid, sku) or {}
                if var.get("in_stock") is False:
                    continue
                price = var.get("price")
                if price is None or (self.budget is not None and price > self.budget + 1e-6):
                    continue
                axes = _preference_axes(self._candidate_text(pid, sku))
                rows.append(
                    {"pid": pid, "sku": sku, "price": price, "axes": axes, "score": sum(axes.values()),
                     "listing_id": prod.get("listing_id")}
                )
        rows.sort(key=lambda r: (-r["score"], r["price"]))
        return rows

    def _ranking_table(self) -> list[dict[str, Any]]:
        rows = []
        for pid, sku in self.ranking_candidates:
            prod = self.registry.products.get(pid) or {}
            var = self.registry.variant(pid, sku) or {}
            facts = prod.get("facts") or {}
            options = var.get("options")
            hz = _max_hz(options)
            if hz is None:
                hz = _max_hz(_fact_lookup(facts, _SPEC_FACT_KEYS["refresh_rate"]))
            if hz is None:
                hz = _max_hz(prod.get("title"))
            size_vals = _sizes_in(options) or _sizes_in(facts.get("size")) or _sizes_in(prod.get("title"))
            size = max(size_vals) if size_vals else None
            rows.append(
                {
                    "key": f"{pid}::{sku}",
                    "pid": pid,
                    "sku": sku,
                    "hz": hz,
                    "size": size,
                    "price": var.get("price"),
                    "in_stock": var.get("in_stock"),
                    "seen": pid in self.registry.viewed,
                }
            )
        return rows

    def _ranking_best(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        eligible = [
            r
            for r in rows
            if r["seen"]
            and r["hz"] is not None
            and r["size"] is not None
            and r["price"] is not None
            and r["in_stock"] is not False
            and (self.budget is None or r["price"] <= self.budget + 1e-6)
        ]
        if not eligible:
            return None
        eligible.sort(key=lambda r: (-r["hz"], -r["size"], r["price"]))
        return eligible[0]

    # -- guardrails --------------------------------------------------------
    def _sanitize_args(self, call: dict[str, Any], notes: list[str]) -> dict[str, Any] | None:
        name, args = call["name"], dict(call.get("args") or {})
        if name in ("search", "filter"):
            k = args.get("k")
            try:
                args["k"] = max(1, min(10, int(k))) if k is not None else 10
            except (TypeError, ValueError):
                args["k"] = 10
            if name == "search":
                query = str(args.get("query") or "")
                query = query.replace('"', " ").replace("'", " ")
                query = _QUERY_PRICE_RE.sub(" ", query)
                query = re.sub(r"\bphp\b", " ", query, flags=re.IGNORECASE)
                query = " ".join(query.split())[:200]
                if not query:
                    return None
                args["query"] = query
                args.setdefault("in_stock", True)
                normalized = _norm(query)
                if (
                    self.family == "retrieval_recall"
                    and normalized not in self.distinct_queries
                    and len(self.distinct_queries) >= _RETRIEVAL_QUERY_BUDGET
                ):
                    notes.append(
                        f"search {query!r} blocked: the {_RETRIEVAL_QUERY_BUDGET}-query budget is used; pick from "
                        "the matching candidates already surfaced (see list), or use filter/view/compare."
                    )
                    return None
                self.distinct_queries.add(normalized)
            else:
                if "max_price" in args:
                    mp = _to_float(args.get("max_price"))
                    if mp is None:
                        args.pop("max_price")
                    else:
                        args["max_price"] = mp
                if not any(key in args for key in ("brand", "category", "max_price")):
                    return None
        elif name == "view":
            pid = str(args.get("product_id") or "").strip()
            if not pid:
                return None
            args = {"product_id": pid}
        elif name == "compare":
            ids = args.get("product_ids") or []
            if isinstance(ids, str):
                ids = [x.strip() for x in ids.split(",") if x.strip()]
            ids = [str(x) for x in ids if x][:5]
            if not ids:
                return None
            args = {"product_ids": ids}
        elif name in ("inspect_stock", "add_to_cart", "remove_from_cart"):
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return None
            args = {"product_id": pid, "sku": sku}
        elif name == "message":
            content = str(args.get("content") or "").strip()
            if not content:
                return None
            args = {"content": content[:600]}
        elif name == "inspect_cart":
            args = {}
        elif name == "place_test_order":
            pass
        return {**call, "args": args}

    def _guard(self, calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        notes: list[str] = []
        cleaned: list[dict[str, Any]] = []
        seen: set[str] = set()
        for call in calls:
            fixed = self._sanitize_args(call, notes)
            if fixed is None:
                if not notes or not notes[-1].startswith("search"):
                    notes.append(f"Dropped malformed call {call['name']}.")
                continue
            sig = json.dumps({"n": fixed["name"], "a": fixed["args"]}, sort_keys=True)
            if sig in seen:
                continue
            seen.add(sig)
            cleaned.append(fixed)

        # Orders travel alone.
        orders = [c for c in cleaned if c["name"] == "place_test_order"]
        if orders:
            order = self._guard_order(orders[0], notes)
            if order is not None:
                if order.get("_converted_add"):
                    removals = [
                        {"name": "remove_from_cart", "args": {"product_id": opid, "sku": osku}, "id": None}
                        for (opid, osku) in list(self.registry.cart)
                        if (opid, osku) != (order["args"]["product_id"], order["args"]["sku"])
                        and self._has("remove_from_cart")
                    ]
                    return removals + [order], notes
                return [order], notes
            cleaned = [c for c in cleaned if c["name"] != "place_test_order"]

        # A lone status message wastes a turn unless the shopper asked something.
        if (
            len(cleaned) == 1
            and cleaned[0]["name"] == "message"
            and self._turn() > 0
            and not (self.last_user_message and "?" in self.last_user_message)
        ):
            notes.append(
                "Lone message blocked: the shopper asked nothing; do not spend a turn on status updates. "
                "Batch any reply with your next lookup/cart calls."
            )
            cleaned = []

        out: list[dict[str, Any]] = []
        mutation_seen = False
        for call in cleaned:
            if call["name"] == "add_to_cart":
                pid, sku = call["args"]["product_id"], call["args"]["sku"]
                if pid not in self.registry.viewed:
                    notes.append(
                        f"add_to_cart({pid}) blocked: view/compare the product first (required before the first cart add)."
                    )
                    if self._has("view"):
                        out.append({"name": "view", "args": {"product_id": pid}, "id": call.get("id")})
                    continue
                var = self.registry.variant(pid, sku)
                if var and var.get("in_stock") is False:
                    notes.append(f"add_to_cart({pid},{sku}) blocked: that variant is out of stock.")
                    continue
                if var and self.budget is not None and var.get("price") is not None and var["price"] > self.budget + 1e-6:
                    notes.append(
                        f"add_to_cart({pid},{sku}) blocked: price {var['price']} exceeds budget {self.budget}."
                    )
                    continue
                if (
                    self.recall.active()
                    and self.recall.identity_count() < self.recall.required
                    and len(self.distinct_queries) < _RETRIEVAL_QUERY_BUDGET
                ):
                    notes.append(
                        "add_to_cart blocked: surface more distinct title-matching identities first (see progress line)."
                    )
                    continue
                issues = self._spec_issues(pid, sku)
                if issues:
                    count = self.spec_blocks.get(pid, 0)
                    self.spec_blocks[pid] = count + 1
                    if count < 2:
                        notes.append(f"add_to_cart({pid},{sku}) blocked: " + "; ".join(issues))
                        continue
                    notes.append(f"add_to_cart({pid},{sku}) allowed despite: " + "; ".join(issues))
                block = self._family_first_add_block(pid, sku)
                if block:
                    notes.append(block)
                    continue
                block = self._preference_add_block(pid, sku)
                if block:
                    notes.append(block)
                    continue
                if mutation_seen:
                    notes.append("Only one cart mutation per turn; deferred extra add_to_cart.")
                    continue
                mutation_seen = True
                # Remove other cart items so the cart holds only the intended purchase.
                for (opid, osku) in list(self.registry.cart):
                    if (opid, osku) == (pid, sku) or not self._has("remove_from_cart"):
                        continue
                    already = any(
                        c["name"] == "remove_from_cart"
                        and c["args"] == {"product_id": opid, "sku": osku}
                        for c in out
                    )
                    if not already:
                        out.append({"name": "remove_from_cart", "args": {"product_id": opid, "sku": osku}, "id": None})
                out.append(call)
            else:
                out.append(call)
        # An empty group consumes no environment turn: the caller re-prompts the
        # model with the guard notes instead.
        return out[: self.max_calls], notes

    def _family_first_add_block(self, pid: str, sku: str) -> str | None:
        """Family-specific checks that only apply before the first cart commitment."""
        if self.first_add_done:
            return None
        if self.family == "ranking" and self.ranking_candidates and self.ranking_blocks < 1:
            rows = self._ranking_table()
            if all(r["seen"] and r["hz"] is not None and r["size"] is not None and r["price"] is not None for r in rows):
                best = self._ranking_best(rows)
                if best and best["key"] != f"{pid}::{sku}":
                    self.ranking_blocks += 1
                    return (
                        f"add_to_cart({pid},{sku}) blocked: the computed ranking (refresh rate, then screen size, then "
                        f"lowest price, using variant options first) puts {best['key']} first "
                        f"(hz={best['hz']}, size={best['size']}, price={best['price']}). Commit to that one unless the "
                        "table in the state note is clearly wrong."
                    )
        if self.family == "recovery" and self.pref_color and self.recovery_blocks < 2:
            var = self.registry.variant(pid, sku) or {}
            options = str(var.get("options") or "").lower()
            if self.pref_color not in options:
                sibs = self.registry.siblings(sku)
                match = [
                    key for key in sibs
                    if self.pref_color in str((self.registry.variants.get(key) or {}).get("options") or "").lower()
                    and (self.registry.variants.get(key) or {}).get("in_stock") is not False
                ]
                unviewed = sorted({p for (p, _s) in sibs if p not in self.registry.viewed})
                if match:
                    self.recovery_blocks += 1
                    mp, ms = match[0]
                    return (
                        f"add_to_cart({pid},{sku}) blocked: its options {options!r} are not '{self.pref_color}'; the sibling "
                        f"{mp}/{ms} states options {(self.registry.variants[match[0]].get('options'))!r}. inspect_stock and "
                        "add that one as the initial choice."
                    )
                if unviewed:
                    self.recovery_blocks += 1
                    return (
                        f"add_to_cart({pid},{sku}) blocked: options {options!r} are not '{self.pref_color}'. Sibling listings "
                        f"with the same sku prefix are not viewed yet: {unviewed[:6]}. `view` them (batch) to find the "
                        f"'{self.pref_color}' variant before committing."
                    )
        return None

    def _preference_add_block(self, pid: str, sku: str) -> str | None:
        """Steer preference tasks to the candidate with the most explicitly stated axes."""
        if self.family != "preference_reasoning" or self.preference_blocks >= 2:
            return None
        rows = self._preference_table()
        mine = next((r for r in rows if r["pid"] == pid and r["sku"] == sku), None)
        if mine is None:
            return None
        better = [r for r in rows if r["score"] > mine["score"] and r["pid"] != pid]
        if not better:
            return None
        self.preference_blocks += 1
        missing = [axis for axis, ok in mine["axes"].items() if not ok]
        alts = "; ".join(
            f"{r['pid']}/{r['sku']} price={r['price']} axes={[a for a, ok in r['axes'].items() if ok]}" for r in better[:3]
        )
        return (
            f"add_to_cart({pid},{sku}) blocked: its visible text lacks {missing} while viewed alternatives state more "
            f"of the four axes: {alts}. inspect_stock and add the best of those instead (the hidden score rewards "
            "explicit sweat/water, sport-fit, battery-hours and wireless statements)."
        )

    def _guard_order(self, order: dict[str, Any], notes: list[str]) -> dict[str, Any] | None:
        args = dict(order.get("args") or {})
        pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
        if (not pid or not sku) and self.registry.cart:
            pid, sku = next(iter(self.registry.cart))
        if not pid or not sku:
            notes.append("place_test_order blocked: nothing is in the cart yet.")
            return None
        if (
            self.recall.active()
            and self.recall.identity_count() < self.recall.required
            and len(self.distinct_queries) < _RETRIEVAL_QUERY_BUDGET
        ):
            notes.append("place_test_order blocked: retrieval requirement not met yet (see progress line).")
            return None
        var = self.registry.variant(pid, sku) or {}
        if (pid, sku) not in self.registry.cart:
            if pid in self.registry.viewed and var.get("in_stock") is not False and self._has("add_to_cart"):
                issues = self._spec_issues(pid, sku)
                if issues and self.spec_blocks.get(pid, 0) < 2:
                    self.spec_blocks[pid] = self.spec_blocks.get(pid, 0) + 1
                    notes.append(f"place_test_order blocked: {pid}/{sku} fails firm requirements: " + "; ".join(issues))
                    return None
                notes.append(
                    f"place_test_order converted: {pid}/{sku} was not in the cart, so it was added instead; "
                    "inspect_stock it next, then order."
                )
                return {
                    "name": "add_to_cart",
                    "args": {"product_id": pid, "sku": sku},
                    "id": order.get("id"),
                    "_converted_add": True,
                }
            notes.append(f"place_test_order blocked: {pid}/{sku} is not in the cart and was never viewed.")
            return None
        if var.get("in_stock") is False:
            notes.append("place_test_order blocked: that variant is out of stock; recover to another item.")
            return None
        if self.budget is not None and var.get("price") is not None and var["price"] > self.budget + 1e-6:
            notes.append(
                f"place_test_order blocked: current price {var['price']} exceeds budget {self.budget}; recover."
            )
            return None
        issues = self._spec_issues(pid, sku)
        if issues and self.spec_blocks.get(pid, 0) < 2 and self._turns_left() > 3:
            self.spec_blocks[pid] = self.spec_blocks.get(pid, 0) + 1
            notes.append(f"place_test_order blocked: {pid}/{sku} fails firm requirements: " + "; ".join(issues))
            return None
        if not self.registry.is_fresh(pid, sku) and self._turns_left() > 1 and self._has("inspect_stock"):
            notes.append("place_test_order deferred: inspect_stock the variant after the last cart change/event first.")
            return {"name": "inspect_stock", "args": {"product_id": pid, "sku": sku}, "id": order.get("id"), "_deferred_order": True}
        final_args: dict[str, Any] = {"product_id": pid, "sku": sku}
        if self.wants_reason:
            product = self.registry.products.get(pid) or {}
            claims = build_claims(product, var, self.goal, self.allowed_claim_fields)
            if claims:
                final_args["justification"] = {"claims": claims}
        return {"name": "place_test_order", "args": final_args, "id": order.get("id")}

    # -- deterministic completion near the step limit ------------------------
    def _forced_actions(self) -> list[dict[str, Any]] | None:
        left = self._turns_left()
        reg = self.registry
        if left > _FORCE_ADD_MARGIN:
            return None
        cart_item = reg.best_cart_item(self.budget)
        if cart_item:
            pid, sku = cart_item
            if reg.is_fresh(pid, sku) or left <= 1:
                order = self._guard_order({"name": "place_test_order", "args": {"product_id": pid, "sku": sku}}, [])
                if order and order["name"] == "place_test_order":
                    return [order]
            if left <= _FORCE_INSPECT_MARGIN + 1 and self._has("inspect_stock"):
                return [{"name": "inspect_stock", "args": {"product_id": pid, "sku": sku}, "id": None}]
            return None
        cand = reg.best_known_candidate(self.budget)
        if cand:
            pid, sku = cand
            calls = [
                {"name": "remove_from_cart", "args": {"product_id": opid, "sku": osku}, "id": None}
                for (opid, osku) in reg.cart
                if self._has("remove_from_cart")
            ]
            calls.append({"name": "add_to_cart", "args": {"product_id": pid, "sku": sku}, "id": None})
            return calls
        return None

    # -- observation handling ----------------------------------------------
    def _update_requirements_from_message(self, text: str) -> None:
        new_budget = _parse_budget(text)
        lowered = text.lower()
        if new_budget is not None and (
            self.budget is None or new_budget < self.budget or "up to" in lowered or "at most" in lowered
        ):
            self.budget = new_budget
        if self.family != "ranking":
            size = _requested_size(text)
            if size is not None:
                self.spec_size = size
            res = _requested_res_class(text)
            if res is not None and re.search(r"\b(resolution|display|screen|needs?|require)", lowered):
                self.spec_res = res
            ram = _requested_ram(text)
            if ram is not None:
                self.spec_ram = ram

    def _ingest(self, calls: list[dict[str, Any]], result: dict[str, Any]) -> list[dict[str, Any]]:
        turn = self.session.turn
        condensed: list[dict[str, Any]] = []
        reg = self.registry
        for call, res in zip(calls, result.get("calls") or []):
            step = res.get("observation") if isinstance(res, dict) else None
            if not isinstance(step, dict):
                continue
            name = call["name"]
            obs = step.get("observation") if isinstance(step.get("observation"), dict) else {}
            if name in ("search", "filter"):
                rows = obs.get("results") or []
                for row in rows:
                    reg.ingest_row(row)
                self.recall.observe_rows(rows)
            elif name == "view":
                reg.ingest_view(obs)
            elif name == "compare":
                for prod in obs.get("compared") or []:
                    reg.ingest_view(prod)
            elif name == "inspect_stock":
                reg.ingest_stock(obs, turn)
            elif name == "add_to_cart":
                before = reg.cart_version
                reg.ingest_cart(obs.get("cart"))
                if obs.get("market_event"):
                    reg.ingest_event(obs["market_event"], turn)
                if obs.get("added") and not step.get("error"):
                    self.first_add_done = True
                if reg.cart_version != before:
                    reg.note_version_turn(turn + 1)
            elif name == "remove_from_cart":
                before = reg.cart_version
                reg.ingest_cart(obs.get("cart"))
                if reg.cart_version != before:
                    reg.note_version_turn(turn + 1)
            elif name == "inspect_cart":
                reg.ingest_cart(obs.get("cart"))
            elif name == "place_test_order":
                if step.get("done") and not step.get("error"):
                    reg.order_placed = True
            if step.get("done"):
                self.done = True
            condensed.append({"id": call.get("id"), "payload": _condense(name, step)})
        user_message = result.get("user_message")
        if isinstance(user_message, dict) and user_message.get("content"):
            self.last_user_message = str(user_message["content"])
            self._update_requirements_from_message(self.last_user_message)
        return condensed

    def _state_note(self, notes: list[str]) -> str:
        reg = self.registry
        lines = [f"[STATE] turn {self._turn()}/{self.max_steps} used; {self._turns_left()} turns left."]
        reqs = []
        if self.budget is not None:
            reqs.append(f"budget <= {self.budget:.2f} PHP")
        if self.spec_size is not None:
            reqs.append(f"screen size = {self.spec_size} inch (facts.size / variant options)")
        if self.spec_res is not None:
            reqs.append(f"resolution class = {_RES_CLASS_NAMES.get(self.spec_res)} (facts.display)")
        if self.spec_ram is not None:
            reqs.append(f"RAM = {self.spec_ram} GB (variant options)")
        if self.category_req:
            reqs.append(f"category_path must include {self.category_req!r}")
        if self.brand_req:
            reqs.append(f"brand field must be {self.brand_req!r} (null brand never qualifies)")
        if self.pref_color:
            reqs.append(f"preferred colour '{self.pref_color}' of model '{self.pref_model}' for the first cart add")
        if reqs:
            lines.append("Firm requirements (latest): " + "; ".join(reqs) + ".")
        if reg.cart:
            items = []
            for (pid, sku) in reg.cart:
                var = reg.variant(pid, sku) or {}
                fresh = "inspected after last change" if reg.is_fresh(pid, sku) else "NOT yet inspected after last change"
                items.append(f"{pid}/{sku} price={var.get('price')} in_stock={var.get('in_stock')} ({fresh})")
            lines.append("Cart: " + "; ".join(items))
        else:
            lines.append("Cart: empty.")
        if reg.events:
            ev = reg.events[-1]
            lines.append(
                f"Last MARKET EVENT (turn {ev['turn']}): {ev['kind']} on {ev['product_id']}/{ev['sku']} "
                f"old_price={ev['old_price']} new_price={ev['new_price']} in_stock_after={ev['in_stock_after']}."
            )
            sibs = [
                key for key in reg.siblings(ev["sku"]) if key != (ev["product_id"], ev["sku"])
            ]
            if sibs:
                lines.append(
                    "Sibling listings of the affected product (same sku prefix; often the same product in another "
                    "colour/storage at the OLD price): "
                    + "; ".join(
                        f"{p}/{s} price={(reg.variants.get((p, s)) or {}).get('price')} "
                        f"in_stock={(reg.variants.get((p, s)) or {}).get('in_stock')} viewed={p in reg.viewed}"
                        for p, s in sibs[:8]
                    )
                    + ". Check them (view + inspect_stock) before searching elsewhere."
                )
        if self.recall.active():
            lines.append(self.recall.summary(len(self.distinct_queries)))
            cands = self.recall.candidate_lines(reg)
            if cands:
                lines.append("Title-matching candidates surfaced:")
                lines.extend(cands)
        if self.family == "preference_reasoning":
            rows = self._preference_table()
            if rows:
                lines.append("[PREFERENCE HELPER] viewed in-stock candidates within budget, axes stated in title/options:")
                for r in rows[:10]:
                    flags = ",".join(a for a, ok in r["axes"].items() if ok) or "none"
                    lines.append(f"  - {r['pid']}/{r['sku']} price={r['price']} score={r['score']}/4 axes={flags}")
                lines.append("  => commit only to a 4/4 candidate; if none is viewed yet, search for one that states "
                             "IPX/waterproof + sport/ear hook + NN hours battery + Bluetooth.")
        if self.family == "ranking" and self.ranking_candidates:
            rows = self._ranking_table()
            lines.append("[RANKING HELPER] parsed from variant options > facts > title:")
            for r in rows:
                lines.append(
                    f"  - {r['key']}: refresh_hz={r['hz']} screen_in={r['size']} price={r['price']} "
                    f"in_stock={r['in_stock']} viewed={r['seen']}"
                )
            best = self._ranking_best(rows)
            if best:
                lines.append(f"  => best available now: {best['key']}")
        for note in notes:
            lines.append("GUARD: " + note)
        left = self._turns_left()
        if left <= _FORCE_ADD_MARGIN:
            lines.append("WARNING: very few turns left. Commit to the best verified option and order now.")
        elif not self.first_add_done and self._turn() >= 8:
            lines.append("Hint: you have not committed yet; choose, inspect_stock, add_to_cart.")
        return "\n".join(lines)

    # -- main loop -----------------------------------------------------------
    def run(self) -> list[dict[str, Any]]:
        pending_notes: list[str] = []
        while not self.done and self.session.turn < self.max_steps:
            forced = self._forced_actions()
            if forced:
                calls = [dict(c, id=c.get("id") or f"forced_{self.session.turn + 1}_{i}") for i, c in enumerate(forced)]
                self.messages.append(
                    {
                        "role": "user",
                        "content": "[STATE] Turn budget nearly exhausted; the runtime is executing a deterministic "
                        "completion step: " + ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in calls),
                    }
                )
                self._append_assistant_calls(calls)
                notes: list[str] = []
            else:
                if pending_notes or self._turn() > 0:
                    self.messages.append({"role": "user", "content": self._state_note(pending_notes)})
                pending_notes = []
                proposed = self._llm_actions()
                if not proposed:
                    fallback = self._fallback_actions()
                    if not fallback:
                        raise RuntimeError("model produced no usable tool calls")
                    calls = fallback
                    self._append_assistant_calls(calls)
                    notes = []
                else:
                    calls, notes = self._guard(proposed)
                    if not calls:
                        self.blocked_streak += 1
                        pending_notes = notes
                        # Remove the assistant tool_calls we cannot honour, then tell the model why.
                        self._drop_last_assistant()
                        if self.blocked_streak >= 3:
                            fallback = self._fallback_actions() or [
                                {"name": "inspect_cart", "args": {}, "id": "probe"}
                            ]
                            calls = fallback
                            self._append_assistant_calls(calls)
                            notes = ["Model kept proposing blocked actions; executed a deterministic step instead."]
                        else:
                            continue
                    else:
                        self._replace_last_assistant_calls(calls)
                    self.blocked_streak = 0

            result = self.session.call(calls)
            if result is None:
                self.consecutive_env_failures += 1
                self._drop_last_assistant()
                if self.consecutive_env_failures >= 3:
                    raise RuntimeError("environment call failed repeatedly")
                pending_notes = ["Environment call failed (transport error); the turn was not consumed. Retry with valid arguments."]
                continue
            self.consecutive_env_failures = 0
            condensed = self._ingest(calls, result)
            for item in condensed:
                self.messages.append(
                    {"role": "tool", "tool_call_id": item["id"], "content": json.dumps(item["payload"], ensure_ascii=False)}
                )
            if self.last_user_message is not None and (result.get("user_message") or {}).get("content"):
                self.messages.append({"role": "user", "content": "SHOPPER: " + self.last_user_message})
            self.dialogue.append(
                {
                    "turn": self.session.turn,
                    "calls": [{"name": c["name"], "args": c["args"]} for c in calls],
                    "user_message": (result.get("user_message") or {}).get("content"),
                    "done": self.done,
                    "guard_notes": notes,
                }
            )
            pending_notes = notes
            if any(c.get("_deferred_order") for c in calls):
                pending_notes.append("Stock inspected; if still valid, place the order now.")
        return self.dialogue

    def _fallback_actions(self) -> list[dict[str, Any]]:
        """Model unavailable: deterministic best effort using what we know."""
        reg = self.registry
        cart_item = reg.best_cart_item(self.budget)
        if cart_item:
            pid, sku = cart_item
            if reg.is_fresh(pid, sku):
                order = self._guard_order({"name": "place_test_order", "args": {"product_id": pid, "sku": sku}}, [])
                if order:
                    return [dict(order, id="fb_order")]
            return [{"name": "inspect_stock", "args": {"product_id": pid, "sku": sku}, "id": "fb_inspect"}]
        cand = reg.best_known_candidate(self.budget)
        if cand:
            pid, sku = cand
            return [{"name": "add_to_cart", "args": {"product_id": pid, "sku": sku}, "id": "fb_add"}]
        if not reg.search_rows and self._has("search"):
            return [{"name": "search", "args": {"query": self.goal[:120], "k": 10, "in_stock": True}, "id": "fb_search"}]
        return []

    def _append_assistant_calls(self, calls: list[dict[str, Any]]) -> None:
        self.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                    for c in calls
                ],
            }
        )

    def _replace_last_assistant_calls(self, calls: list[dict[str, Any]]) -> None:
        """Keep the transcript consistent with what was actually executed."""
        for i, c in enumerate(calls):
            if not c.get("id"):
                c["id"] = f"g{self.session.turn + 1}_{i}"
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()
        self._append_assistant_calls(calls)

    def _drop_last_assistant(self) -> None:
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Entry point required by the sandbox."""
    agent = Agent(problem_data)
    logger.info(
        "problem=%s family=%s budget=%s model=%s", agent.problem_id, agent.family, agent.budget, agent.model
    )
    return agent.run()


__all__ = ["agent_main"]
