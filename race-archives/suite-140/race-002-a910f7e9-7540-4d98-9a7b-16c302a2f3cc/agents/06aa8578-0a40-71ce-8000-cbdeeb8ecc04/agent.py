"""firstking chassis + race-general finish control (EnvPack).

Seed: references/Newstyle agent/firstking.py (0828 closed-set + dialogue hygiene).
Adds: stock ledger, post-cart inspect gate, recovery/finish nudges, claim grounding.
No suite fingerprints, no product-ID tables, no ShoppingBench engines.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_OPENROUTER_MODELS = [
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "deepseek/deepseek-chat-v3.1",
]
_CHUTES_MODELS = [
    "moonshotai/Kimi-K2.6-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
]

_DEFAULT_CLAIM_FIELDS = frozenset(
    {
        "brand",
        "category",
        "in_stock",
        "price",
        "refresh_rate",
        "resolution",
        "screen_size",
    }
)
_FIELD_ALIASES = {
    "display": "resolution",
    "screen": "screen_size",
    "size": "screen_size",
    "screen size": "screen_size",
    "refresh": "refresh_rate",
    "hz": "refresh_rate",
    "category_path": "category",
}

_CAND_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9][A-Za-z0-9._-]*)")
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|"
    r"max(?:imum)? of|within|budget (?:is|of)|can spend up to|at most)\s*"
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?",
    re.I,
)
_ASK_RE = re.compile(
    r"ask me|when you ask|until you ask|if you ask|withheld until",
    re.I,
)
_PREFER_RE = re.compile(
    r"(?:strongly\s+)?prefer(?:red)?\s+(?:the\s+)?(?P<color>[A-Za-z]+)\s+"
    r"(?P<model>[A-Za-z0-9][A-Za-z0-9._-]*)",
    re.I,
)
_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(?:inches|inch|in|\"|'')", re.I)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_EPS = 1e-6

_proxy = ProxyClient(timeout=120, max_retries=2)

_SYSTEM = """You are an ORO Bench shopping agent on EnvPack tools.

Obey this episode's shopper text + Task rules only. Race tasks are hidden and differ
from any local suite — never memorize product IDs or suite ranks. Use only supplied tools.
Continue until an observation has done=true. One tool call per turn.

## Dialogue
If the shopper asks a question or changes budget/spec, reply once with `message`, then continue.
Do not loop on empty acknowledgements. Empty shopper pings are not new questions.
If a firm spec is withheld until you ask, first call is `message`, then shop with the answer.

## Firm vs soft
Firm: category, named hardware spec, budget, in-stock, brands marked required
("I need/want Brand …" without ideally). Soft: ideally/prefer brand — honor on FIRST cart
when a fully compliant preferred item exists; after shock, firm constraints win.
Never put budget/currency words in the search query. Never loosen a firm size.

## Closed candidate ranking (critical)
When the shopper lists explicit variant keys like `product_id::sku` ("compare only these"):
  - Those keys ARE the entire eligible set. Do NOT open-catalog search for substitutes.
  - view each listed product_id (and compare), read refresh/size/price/stock for the listed sku.
  - Rank by the stated priority (often highest refresh → largest screen → lowest price).
  - First add_to_cart must be the current best listed option; after market shocks, re-rank listed
    options only, inspect_stock, then place_test_order on a listed key.
  - Never order a product_id that was not in the listed set.

## Retrieval / constraints / prefs / recovery / justification
- Retrieval: title-token match; if THIS text has a match phrase, keep those tokens, run ≥2
  distinct searches, and view several product_ids before first cart.
- Constraints: verify exact SKU options; category_path from view/compare.
- Prefs: budget+stock hard; match use-case soft attrs without defaulting to cheapest.
- Recovery: preferred color+model first; then same model other color; then other models.
  COMMIT with message if asked — never ask permission.
- Justification: only if shopper asked why/reason. Claims from view facts / variant options /
  inspect only (schema fields). Omit otherwise.

## Finish path
view/compare → inspect_stock → add_to_cart → inspect_stock again → place_test_order.
Once a carted sku is freshly inspected live and in-cap, order it — do not keep searching.
"""


def _provider() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw in ("chutes", "openrouter"):
        return raw
    if getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "chutes"


def _model_list() -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    return list(_OPENROUTER_MODELS if _provider() == "openrouter" else _CHUTES_MODELS)


def _shopper_text(query: str) -> str:
    return query.split("Task rules:", 1)[0].strip()


def _parse_budget(text: str) -> float | None:
    best = None
    for m in _BUDGET_RE.finditer(text or ""):
        whole = m.group(1).replace(",", "")
        frac = m.group(2) or "0"
        try:
            val = float(f"{whole}.{frac}" if m.group(2) is not None else whole)
        except ValueError:
            continue
        best = val if best is None else min(best, val)
    return best


def _parse_candidate_variants(query: str) -> list[tuple[str, str]]:
    """Extract product_id::sku pairs from shopper text (closed ranking sets)."""
    head = _shopper_text(query)
    low = head.lower()
    if not any(
        s in low
        for s in (
            "compare only",
            "candidate variant",
            "candidate variants",
            "these candidate",
            "only these",
        )
    ):
        pairs = _CAND_KEY_RE.findall(head)
        if len(pairs) < 2:
            return []
    else:
        pairs = _CAND_KEY_RE.findall(head)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pid, sku in pairs:
        sku = sku.rstrip(".,;:)")
        key = f"{pid}::{sku}"
        if key in seen:
            continue
        seen.add(key)
        out.append((pid, sku))
    return out


def _justification_required(query: str) -> bool:
    head = _shopper_text(query).lower()
    needles = (
        "why",
        "reason",
        "walk me through",
        "tell me why",
        "justif",
        "good pick",
    )
    return any(n in head for n in needles)


def _needs_ask(query: str) -> bool:
    return bool(_ASK_RE.search(_shopper_text(query)))


def _match_phrase(query: str) -> str | None:
    head = _shopper_text(query)
    m = re.search(r'matches\s+"([^"]+)"', head, re.I)
    if m:
        return m.group(1).strip() or None
    m = re.search(
        r"matches\s+the\s+phrase\s+(.+?)(?:,|\s+at\s+or|\s+and\s+buy|$)",
        head,
        re.I,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip(" .") or None
    return None


def _soft_brand(query: str) -> str | None:
    m = re.search(
        r"ideally,?\s+i\s+prefer\s+([A-Za-z0-9][A-Za-z0-9 ._-]{0,40}?)(?:,|\.|but|$)",
        _shopper_text(query),
        re.I,
    )
    if not m:
        return None
    brand = re.sub(r"\s+", " ", m.group(1)).strip(" .,")
    return brand or None


def _firm_brand(query: str) -> str | None:
    head = _shopper_text(query)
    if re.search(r"\bideally\b", head, re.I):
        return None
    m = re.search(
        r"\b(?:i\s+need|i\s+want|find\s+me)\s+(?:a|an)\s+"
        r"([A-Za-z][A-Za-z0-9][A-Za-z0-9._-]{0,30})\s+"
        r"(?:smartphone|tablet|monitor|headphones?|earbuds?|laptop)\b",
        head,
        re.I,
    )
    if not m:
        return None
    brand = m.group(1).strip()
    if brand.lower() in {"the", "new", "good", "best", "cheap"}:
        return None
    return brand


def _prefer_color_model(query: str) -> dict[str, str]:
    """Recovery-style color+model prefer — not soft 'Ideally prefer Brand'."""
    head = _shopper_text(query)
    if re.search(r"\bideally\b", head, re.I) and not re.search(
        r"strongly\s+prefer", head, re.I
    ):
        return {}
    recoveryish = bool(
        re.search(
            r"strongly\s+prefer|prefer(?:red)?\s+the\s+|another color|other colour|same model",
            head,
            re.I,
        )
    )
    if not recoveryish:
        return {}
    m = _PREFER_RE.search(head)
    if not m:
        return {}
    color = (m.group("color") or "").strip().rstrip(".,;:)")
    model = (m.group("model") or "").strip().rstrip(".,;:)")
    if color.lower() in {"the", "a", "an", "my", "our"}:
        return {"model": model} if model else {}
    out: dict[str, str] = {}
    if color:
        out["color"] = color
    if model:
        out["model"] = model
    return out


def _color_model_lock(query: str) -> bool:
    pref = _prefer_color_model(query)
    return bool(pref.get("model") or pref.get("color")) and (
        bool(re.search(r"another color|other colour|same model", query or "", re.I))
        or bool(re.search(r"strongly\s+prefer", query or "", re.I))
    )


def _asked_inches(query: str) -> float | None:
    m = _INCH_RE.search(_shopper_text(query))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _asked_resolution(query: str) -> str | None:
    head = _shopper_text(query).lower()
    if re.search(r"\buhd\b|\b4k\b|2160p", head):
        return "uhd"
    if re.search(r"\bqhd\b|1440p|\b2k\b", head):
        return "qhd"
    if re.search(r"\bfhd\b|1080p|full\s*hd", head):
        return "fhd"
    if re.search(r"720p|\bhd\b", head):
        return "hd"
    return None


def _asked_fields(query: str) -> list[str]:
    head = _shopper_text(query).lower()
    fields: list[str] = []
    if _INCH_RE.search(head):
        fields.append("screen_size")
    if re.search(
        r"720p|1080p|1440p|2160p|\bfhd\b|\bqhd\b|\buhd\b|\b4k\b|full\s*hd|\bhd\b|resolution",
        head,
    ):
        fields.append("resolution")
    if re.search(r"\d+\s*hz|refresh", head):
        fields.append("refresh_rate")
    return fields


def _claim_field_allowlist(tools: list[dict[str, Any]]) -> frozenset[str]:
    for tool in tools:
        fn = tool.get("function") or tool
        if (fn.get("name") or "") != "place_test_order":
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        claims = (
            ((props.get("justification") or {}).get("properties") or {})
            .get("claims", {})
            .get("items", {})
            .get("properties", {})
            .get("field", {})
        )
        enum = claims.get("enum")
        if isinstance(enum, list) and enum:
            return frozenset(str(x) for x in enum)
    return _DEFAULT_CLAIM_FIELDS


def _normalize_claim_field(field: str, allowed: frozenset[str]) -> str | None:
    raw = str(field or "").strip()
    if not raw:
        return None
    key = raw.lower()
    if raw in allowed:
        return raw
    if key in allowed:
        return key
    mapped = _FIELD_ALIASES.get(key)
    if mapped and mapped in allowed:
        return mapped
    return None


def _normalize_claim_value(field: str, value: Any) -> str:
    text = str(value).strip()
    if field == "in_stock":
        low = text.lower()
        if low in ("true", "1", "yes", "in stock", "instock"):
            return "true"
        if low in ("false", "0", "no", "out of stock"):
            return "false"
    if field == "price":
        cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
        if cleaned:
            try:
                return str(float(cleaned))
            except ValueError:
                return text
    if field == "refresh_rate":
        m = re.search(r"(\d+(?:\.\d+)?)\s*hz", text, flags=re.I)
        if m:
            return f"{m.group(1)}Hz"
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            return f"{text}Hz"
    if field == "screen_size":
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:inch|in|\"|'')?", text, flags=re.I)
        if m:
            return f"{m.group(1)} inches"
    return text


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _stock_flag(value: Any) -> bool | None:
    if value is True:
        return True
    if value is False:
        return False
    if value is None:
        return None
    low = str(value).strip().lower()
    if low in ("true", "1", "yes", "in stock"):
        return True
    if low in ("false", "0", "no", "out of stock"):
        return False
    return None


def _inner_obs(obs: Any) -> dict[str, Any]:
    """EnvPack wraps tool results as {observation:{...}, done, error}."""
    payload = obs if isinstance(obs, dict) else {}
    node = payload.get("observation")
    if isinstance(node, dict) and (
        "done" in payload or "error" in payload or "observation" in payload
    ):
        return node
    return payload


def _res_from_display(text: str) -> str | None:
    low = (text or "").lower()
    if not low:
        return None
    if "uhd" in low or "4k" in low:
        return "uhd"
    if "qhd" in low or re.search(r"\b2k\b", low):
        return "qhd"
    if "fhd" in low or "1080p" in low or "full hd" in low:
        return "fhd"
    if "720p" in low or "hd+" in low or re.search(r"\bhd\b", low) or "hd;" in low:
        return "hd"
    m = re.search(r"(\d{3,5})\s*[x×*]\s*(\d{3,5})", low)
    if m:
        a, b = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        if a >= 3840 and b >= 2160:
            return "uhd"
        if a >= 2560 and b >= 1440:
            return "qhd"
        if a >= 1920 and b >= 1080:
            return "fhd"
        if a >= 1280 and b >= 720:
            return "hd"
    return None


class _StockBook:
    """Race-general finish ledger: views, inspects, cart, shocks."""

    def __init__(self) -> None:
        self.viewed: set[str] = set()
        self.seen: set[str] = set()
        self.searches: list[str] = []
        self.views: dict[str, dict[str, Any]] = {}
        self.inspects: dict[str, dict[str, Any]] = {}
        self.inspect_seq: dict[str, int] = {}
        self.cart: list[tuple[str, str]] = []
        self.cart_seq: int | None = None
        self.shocked: dict[str, str] | None = None
        self.color_relaxed = False
        self.seq = 0

    def save_listing(self, row: dict[str, Any]) -> None:
        pid = str(row.get("product_id") or "")
        if not pid:
            return
        self.seen.add(pid)
        facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
        variants = [v for v in (row.get("variants") or []) if isinstance(v, dict)]
        path = row.get("category_path") if isinstance(row.get("category_path"), list) else []
        self.views[pid] = {
            "brand": row.get("brand"),
            "facts": facts,
            "variants": variants,
            "category_path": [str(x) for x in path],
            "title": str(row.get("title") or ""),
        }

    def sku_row(self, pid: str, sku: str) -> dict[str, Any]:
        for row in (self.views.get(pid) or {}).get("variants") or []:
            if str(row.get("sku") or "") == sku:
                return row
        return {}

    def option_color(self, pid: str, sku: str) -> str:
        options = str(self.sku_row(pid, sku).get("options") or "")
        return options.split("/")[0].strip().lower()

    def listing_inches(self, pid: str, sku: str) -> float | None:
        options = str(self.sku_row(pid, sku).get("options") or "")
        facts = (self.views.get(pid) or {}).get("facts") or {}
        for text in (options, str(facts.get("size") or "")):
            m = _INCH_RE.search(text)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
            raw = str(text).strip()
            if re.fullmatch(r"\d+(?:\.\d+)?", raw):
                try:
                    return float(raw)
                except ValueError:
                    pass
        return None

    def ground_value(self, field: str, pid: str, sku: str) -> str | None:
        snap = self.inspects.get(f"{pid}::{sku}") or {}
        rec = self.views.get(pid) or {}
        facts = rec.get("facts") if isinstance(rec.get("facts"), dict) else {}
        options = str(self.sku_row(pid, sku).get("options") or "")
        if field == "in_stock":
            flag = snap.get("in_stock")
            if flag is True:
                return "true"
            if flag is False:
                return "false"
            return None
        if field == "price":
            price = snap.get("price")
            return None if price is None else str(float(price))
        if field == "brand":
            brand = rec.get("brand")
            return str(brand).strip() if brand else None
        if field == "screen_size":
            for text in (options, str(facts.get("size") or "")):
                m = _INCH_RE.search(text)
                if m:
                    return f"{float(m.group(1)):g} inches"
                raw = str(text).strip()
                if re.fullmatch(r"\d+(?:\.\d+)?", raw):
                    return f"{float(raw):g} inches"
            return None
        if field == "resolution":
            return _res_from_display(
                str(facts.get("display") or facts.get("resolution") or "")
            )
        if field == "refresh_rate":
            m = _HZ_RE.search(options) or _HZ_RE.search(
                str(facts.get("refresh") or facts.get("refresh_rate") or "")
            )
            return f"{m.group(1)}Hz" if m else None
        if field == "category":
            path = rec.get("category_path") or []
            return " > ".join(path) if path else None
        return None

    def ground_claims(
        self, pid: str, sku: str, *, query: str, allowed: frozenset[str]
    ) -> list[dict[str, str]]:
        claims: list[dict[str, str]] = []
        seen: set[str] = set()
        for field in ("in_stock", "price", *_asked_fields(query), "brand"):
            if field not in allowed or field in seen:
                continue
            value = self.ground_value(field, pid, sku)
            if not value:
                continue
            claims.append({"field": field, "value": value})
            seen.add(field)
        return claims

    def live_pairs(self, cap: float | None) -> list[tuple[str, str]]:
        live: list[tuple[str, str]] = []
        for pid, sku in self.cart:
            key = f"{pid}::{sku}"
            snap = self.inspects.get(key) or {}
            if snap.get("in_stock") is False:
                continue
            price = snap.get("price")
            if cap is not None and price is not None and price > cap + _EPS:
                continue
            seq = self.inspect_seq.get(key, 0)
            if self.cart_seq is not None and seq <= self.cart_seq:
                continue
            live.append((pid, sku))
        return live

    def ingest(
        self,
        name: str,
        args: dict[str, Any],
        obs: Any,
        *,
        prefer: dict[str, str],
        query: str,
        cap: float | None,
    ) -> None:
        body = _inner_obs(obs)
        self.seq += 1
        if name == "search":
            q = args.get("query")
            if q:
                self.searches.append(str(q))
            for row in body.get("results") or []:
                if isinstance(row, dict) and row.get("product_id"):
                    self.seen.add(str(row["product_id"]))
        elif name in ("view", "compare"):
            pid = str(args.get("product_id") or "")
            if pid:
                self.viewed.add(pid)
                self.seen.add(pid)
            for extra in args.get("product_ids") or []:
                if extra:
                    self.viewed.add(str(extra))
                    self.seen.add(str(extra))
            if body.get("product_id"):
                self.save_listing(body)
                self.viewed.add(str(body["product_id"]))
            for row in body.get("compared") or []:
                if isinstance(row, dict) and row.get("product_id"):
                    self.save_listing(row)
                    self.viewed.add(str(row["product_id"]))
        elif name == "inspect_stock":
            pid = str(body.get("product_id") or args.get("product_id") or "")
            sku = str(body.get("sku") or args.get("sku") or "")
            if pid:
                key = f"{pid}::{sku}"
                prev = self.inspects.get(key) or {}
                flag = _stock_flag(body.get("in_stock"))
                price = _as_float(body.get("price") or body.get("current_price"))
                self.inspects[key] = {
                    "in_stock": prev.get("in_stock") if flag is None else flag,
                    "price": price if price is not None else prev.get("price"),
                }
                self.inspect_seq[key] = self.seq
        elif name == "add_to_cart":
            added = body.get("added") if isinstance(body.get("added"), dict) else {}
            pid = str(added.get("product_id") or args.get("product_id") or "")
            sku = str(added.get("sku") or args.get("sku") or "")
            if pid and sku:
                pair = (pid, sku)
                if pair not in self.cart:
                    self.cart.append(pair)
                self.cart_seq = self.seq
                flag = _stock_flag(added.get("in_stock"))
                price = _as_float(added.get("price") or added.get("current_price"))
                key = f"{pid}::{sku}"
                prev = self.inspects.get(key) or {}
                if flag is not None or price is not None:
                    self.inspects[key] = {
                        "in_stock": prev.get("in_stock") if flag is None else flag,
                        "price": price if price is not None else prev.get("price"),
                    }
                if flag is False or (
                    cap is not None and price is not None and price > cap + _EPS
                ):
                    self.shocked = {
                        "pid": pid,
                        "sku": sku,
                        "color": self.option_color(pid, sku),
                        "model": prefer.get("model") or "",
                    }
                    if _color_model_lock(query):
                        self.color_relaxed = True
                elif flag is not False and (
                    cap is None or price is None or price <= cap + _EPS
                ):
                    self.shocked = None
            rows = body.get("cart")
            if isinstance(rows, list):
                self.cart = [
                    (str(r.get("product_id")), str(r.get("sku") or ""))
                    for r in rows
                    if isinstance(r, dict) and r.get("product_id")
                ]
        elif name == "remove_from_cart":
            gone = body.get("removed") if isinstance(body.get("removed"), dict) else {}
            pair = (
                str(gone.get("product_id") or args.get("product_id") or ""),
                str(gone.get("sku") or args.get("sku") or ""),
            )
            if pair in self.cart:
                self.cart.remove(pair)
            rows = body.get("cart")
            if isinstance(rows, list):
                self.cart = [
                    (str(r.get("product_id")), str(r.get("sku") or ""))
                    for r in rows
                    if isinstance(r, dict) and r.get("product_id")
                ]
        elif name == "inspect_cart":
            rows = body.get("cart")
            if isinstance(rows, list):
                self.cart = [
                    (str(r.get("product_id")), str(r.get("sku") or ""))
                    for r in rows
                    if isinstance(r, dict) and r.get("product_id")
                ]


def _sanitize_place_order_args(
    args: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
    book: _StockBook | None = None,
) -> dict[str, Any]:
    out = dict(args)
    if not _justification_required(query):
        out.pop("justification", None)
        return out
    pid = str(out.get("product_id") or "")
    sku = str(out.get("sku") or "")
    if book is not None and pid and sku:
        grounded = book.ground_claims(
            pid, sku, query=query, allowed=allowed_fields
        )
        if grounded:
            out["justification"] = {"claims": grounded}
            return out
    just = out.get("justification")
    if not isinstance(just, dict):
        return out
    claims_in = just.get("claims")
    if not isinstance(claims_in, list):
        out.pop("justification", None)
        return out
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    for claim in claims_in:
        if not isinstance(claim, dict):
            continue
        field = _normalize_claim_field(str(claim.get("field", "")), allowed_fields)
        if field is None or field in seen:
            continue
        value = _normalize_claim_value(field, claim.get("value", ""))
        if value == "":
            continue
        cleaned.append({"field": field, "value": value})
        seen.add(field)
    if cleaned:
        out["justification"] = {"claims": cleaned}
    else:
        out.pop("justification", None)
    return out


def _arguments(
    tool_call: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
    book: _StockBook | None = None,
) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    if (tool_call["function"].get("name") or "") == "place_test_order":
        return _sanitize_place_order_args(
            parsed, query=query, allowed_fields=allowed_fields, book=book
        )
    return parsed


def _looks_like_shopper_intervention(text: str) -> bool:
    low = (text or "").lower().strip()
    if len(low) < 8:
        return False
    cues = (
        "?",
        "which one",
        "before you buy",
        "hold on",
        "wait,",
        "still available",
        "i can only go",
        "non-negotiable",
        "re-check",
        "tell you if you ask",
        "firm requirement",
        "please keep to",
        "compare the current",
    )
    return any(c in low for c in cues)


def _chat(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    last_err: Exception | None = None
    for model in _model_list():
        try:
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if inference is None:
                last_err = RuntimeError(f"inference failed for {model}")
                continue
            return inference
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(f"inference unavailable: {last_err}")


def _env_call(envelope: dict[str, Any]) -> dict[str, Any]:
    base_id = str(envelope.get("call_id") or "call")
    for attempt in range(3):
        payload = dict(envelope)
        if attempt:
            suffix = f"r{attempt}-{int(time.time() * 1000) % 1_000_000}"
            payload["call_id"] = f"{base_id}-{suffix}"
            payload["idempotency_key"] = payload["call_id"]
            calls = []
            for index, call in enumerate(payload.get("calls") or [], start=1):
                c = dict(call)
                c["call_id"] = f"{payload['call_id']}-{index}"
                calls.append(c)
            payload["calls"] = calls
        result = _proxy.post("/environment/call", json_data=payload)
        if result is not None:
            return result
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"environment call failed after retries ({base_id})")


def _tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [str(c.get("function", {}).get("name") or "") for c in tool_calls]


def _call_args(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = str((call.get("function") or {}).get("name") or "")
    raw = (call.get("function") or {}).get("arguments", "{}")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def _closed_set_violation(
    tool_calls: list[dict[str, Any]],
    *,
    candidates: list[tuple[str, str]],
    viewed: set[str],
) -> str | None:
    """Return rejection reason if calls leave the closed candidate set / skip views."""
    if not candidates:
        return None
    cand_pids = {pid for pid, _ in candidates}
    cand_keys = {f"{pid}::{sku}" for pid, sku in candidates}
    names = _tool_names(tool_calls)
    if any(n in ("search", "filter") for n in names) and len(viewed & cand_pids) < min(
        2, len(cand_pids)
    ):
        return (
            "Closed candidate set: do not search/filter. "
            f"Call view on listed product_ids first: {sorted(cand_pids)}"
        )
    for call in tool_calls:
        name, args = _call_args(call)
        if name == "view":
            pid = str(args.get("product_id") or "")
            if pid and pid not in cand_pids:
                return (
                    f"view product_id {pid} is outside the listed candidates. "
                    f"View only: {sorted(cand_pids)}"
                )
        if name in ("add_to_cart", "inspect_stock", "place_test_order"):
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and pid not in cand_pids:
                return (
                    f"{name} product_id {pid} is outside listed candidates "
                    f"{sorted(cand_pids)}"
                )
            if pid and sku and f"{pid}::{sku}" not in cand_keys and name != "inspect_stock":
                if name in ("add_to_cart", "place_test_order"):
                    return (
                        f"{name} {pid}::{sku} is not a listed variant. "
                        f"Use one of: {sorted(cand_keys)}"
                    )
    return None


def _finish_violation(
    tool_calls: list[dict[str, Any]],
    *,
    book: _StockBook,
    query: str,
    cap: float | None,
    allowed_fields: frozenset[str],
) -> str | None:
    """Reject unsafe finish / search / recovery moves (race-general)."""
    prefer = _prefer_color_model(query)
    phrase = _match_phrase(query)
    firm = _firm_brand(query)
    need_inch = _asked_inches(query)
    need_res = _asked_resolution(query)
    lock = _color_model_lock(query)
    failed_color = (book.shocked or {}).get("color") or (
        prefer.get("color") if book.color_relaxed else ""
    )

    for call in tool_calls:
        name, args = _call_args(call)
        if name == "search":
            q = str(args.get("query") or "")
            low = q.lower()
            if re.search(
                r"\b(?:under|below|within|budget|cheap|php|usd|eur|at\s+most|upto|up\s+to)\b",
                low,
            ) or re.search(r"\d+(?:\.\d+)?\s*(?:php|usd|eur)\b", low):
                return (
                    "leave budget/currency words out of the search query; "
                    "filter by price after view/inspect"
                )
            if need_inch is not None:
                found = _INCH_RE.search(q)
                if found:
                    try:
                        got = float(found.group(1))
                    except ValueError:
                        got = None
                    if got is not None and abs(got - need_inch) > 0.001:
                        return (
                            f"keep firm size {need_inch:g} inch in search; "
                            "do not loosen to a nearby size"
                        )
            if phrase:
                toks = [
                    t
                    for t in re.findall(r"[A-Za-z0-9]+", phrase.lower())
                    if len(t) > 1
                ]
                if any(t not in low for t in toks):
                    return f"keep match phrase tokens in search: `{phrase}`"
            if firm and firm.lower() not in low:
                prior = " ".join(book.searches).lower()
                if firm.lower() not in prior:
                    return f"include firm brand `{firm}` in the search query"

        if name == "add_to_cart" and phrase:
            distinct = {s.strip().lower() for s in book.searches if s.strip()}
            if len(distinct) < 2:
                return (
                    "retrieval: run a second distinct search reformulation of the "
                    "match phrase before carting"
                )
            if len(book.viewed) < 3 and len(book.seen) < 8:
                return (
                    "retrieval: view more distinct product_ids from search hits "
                    "before carting"
                )

        if name in ("add_to_cart", "place_test_order"):
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if not pid:
                continue
            if pid not in book.views and name == "add_to_cart":
                return "view this product_id before carting"
            brand = str((book.views.get(pid) or {}).get("brand") or "").strip()
            if firm and brand and brand.lower() != firm.lower():
                return f"firm brand is `{firm}`; cart a listing whose view.brand matches"
            if need_inch is not None:
                got_inch = book.listing_inches(pid, sku)
                if got_inch is not None and abs(got_inch - need_inch) > 0.001:
                    return (
                        f"facts/options size {got_inch:g} inch != firm "
                        f"{need_inch:g} inch"
                    )
            if need_res:
                got_res = book.ground_value("resolution", pid, sku)
                if got_res and got_res != need_res:
                    return (
                        f"facts display resolves to {got_res}, shopper asked "
                        f"{need_res}"
                    )
            if book.shocked and pid == book.shocked.get("pid") and sku == book.shocked.get(
                "sku"
            ):
                return "do not reuse the shocked sku; recover on another live option"
            color = book.option_color(pid, sku)
            if lock and (book.shocked or book.color_relaxed):
                need = (prefer.get("model") or "").strip()
                if need and need.lower() not in (
                    str((book.views.get(pid) or {}).get("title") or "")
                    + " "
                    + str(
                        ((book.views.get(pid) or {}).get("facts") or {}).get("model")
                        or ""
                    )
                ).lower():
                    return (
                        f"recover on the same model token ({need}), other color — "
                        "search that token again before changing models"
                    )
                if failed_color and color and (
                    failed_color == color
                    or failed_color in color
                    or color in failed_color
                ):
                    return (
                        "preferred/shocked color already failed; "
                        "cart another color of the same model and place_test_order"
                    )

        if name == "place_test_order":
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if not pid or not sku:
                return "order needs product_id and sku"
            if (pid, sku) not in book.cart:
                return "cart that variant before ordering"
            key = f"{pid}::{sku}"
            snap = book.inspects.get(key) or {}
            seq = book.inspect_seq.get(key, 0)
            if book.cart_seq is not None and seq <= book.cart_seq:
                return "inspect_stock that sku after the latest cart add"
            if snap.get("in_stock") is False:
                return (
                    "latest inspect is out of stock. remove_from_cart, pick another "
                    "live option, inspect, then order"
                )
            price = snap.get("price")
            if cap is not None and price is not None and price > cap + _EPS:
                return (
                    f"price {price:.2f} exceeds cap {cap:.2f}. remove_from_cart, "
                    "pick a cheaper live option, inspect, then order"
                )
            if "inspect_cart" in _tool_names(tool_calls) and book.cart_seq is not None:
                return "inspect_stock the just-carted sku first (not inspect_cart alone)"
            if _justification_required(query):
                claims = book.ground_claims(
                    pid, sku, query=query, allowed=allowed_fields
                )
                have = {c["field"] for c in claims}
                asked = _asked_fields(query)
                missing = [f for f in asked if f in allowed_fields and f not in have]
                if missing:
                    return (
                        f"viewed facts/options for this sku have no {missing[0]}; "
                        "view a listing whose structured fields state that spec"
                    )
                if "price" not in have or "in_stock" not in have:
                    return "inspect_stock this sku so price and stock claims are grounded"
    return None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """EnvPack loop: firstking closed-set spine + finish ledger."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    max_steps = int(policy["max_steps"])
    tools = policy["tools"]
    query = str(policy.get("query") or "")
    allowed_fields = _claim_field_allowlist(tools)
    candidates = _parse_candidate_variants(query)
    cand_pids = {pid for pid, _ in candidates}
    cap = _parse_budget(_shopper_text(query))
    prefer = _prefer_color_model(query)
    book = _StockBook()
    book.seen |= cand_pids

    boot = (
        f"{query}\n\n"
        "Follow Task rules. Reply to real shopper questions with message once, then continue. "
        "Leave budget out of search strings. inspect_stock after every cart. "
        "Finish with place_test_order on a live in-cap sku."
    )
    if candidates:
        listed = ", ".join(f"{pid}::{sku}" for pid, sku in candidates)
        boot += (
            "\n\nCLOSED CANDIDATE SET (mandatory): "
            f"{listed}. "
            "view each of these product_ids (compare if helpful). Rank by the shopper's "
            "stated priority. Do not search the open catalog for other products. "
            "First cart and final order must use a listed product_id::sku."
        )
    if _needs_ask(query):
        boot += "\n\nFirst call: `message` asking for the withheld spec, then shop."
    soft = _soft_brand(query)
    if soft and not _color_model_lock(query):
        boot += (
            f"\n\nSoft brand prefer `{soft}` only when a fully firm-compliant listing "
            "exists; firm size/budget/stock win after a market shock."
        )
    firm = _firm_brand(query)
    if firm:
        boot += (
            f"\n\nFirm brand `{firm}` from THIS text: keep it in searches and carts "
            "even after market shocks."
        )
    phrase = _match_phrase(query)
    if phrase:
        boot += (
            f"\n\nMatch phrase from THIS text: `{phrase}`. Use ≥2 distinct searches "
            "that keep those tokens; view several hits before cart."
        )
    if prefer.get("model") and _color_model_lock(query):
        hint = prefer["model"]
        if prefer.get("color"):
            hint = f"{prefer['model']} {prefer['color']}"
        boot += (
            f"\n\nPreferred token from THIS text: search `{hint}` first. "
            "First cart the preferred color when a variant matches."
        )
    if _justification_required(query):
        spec = ", ".join(_asked_fields(query)) or "the named spec"
        boot += (
            f"\n\nJustification: only in_stock, price, {spec}, and brand if "
            "view.brand is set. Read view facts / selected variant / inspect."
        )
    if cap is not None:
        boot += f"\n\nCAP: {cap:.2f}."

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": boot},
    ]
    dialogue: list[dict[str, Any]] = []
    must_message = bool(_needs_ask(query))

    for turn in range(1, max_steps + 1):
        live_now = book.live_pairs(cap)
        if (
            not must_message
            and live_now
            and (
                turn >= max_steps - 8
                or _justification_required(query)
                or book.color_relaxed
            )
        ):
            pid, sku = live_now[0]
            tool_calls = [
                {
                    "id": f"forced-order-{turn}",
                    "type": "function",
                    "function": {
                        "name": "place_test_order",
                        "arguments": json.dumps({"product_id": pid, "sku": sku}),
                    },
                }
            ]
            assistant_content = f"Ordering verified live in-cap sku {pid}::{sku}."
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                }
            )
            group_id = f"{problem_id}-turn-{turn}"
            envelope = {
                **binding,
                "call_id": group_id,
                "idempotency_key": group_id,
                "turn": turn,
                "calls": [
                    {
                        "call_id": f"{group_id}-1",
                        "action": {
                            "name": "place_test_order",
                            "args": _arguments(
                                tool_calls[0],
                                query=query,
                                allowed_fields=allowed_fields,
                                book=book,
                            ),
                        },
                    }
                ],
            }
            result = _env_call(envelope)
            dialogue.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                    "environment_result": result,
                }
            )
            break

        if must_message:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "REQUIRED this turn: call `message` once to answer the shopper "
                        "(COMMIT the next buy if stock broke; do not ask permission), "
                        "then next turns resume shopping/ranking."
                    ),
                }
            )
        elif candidates and len(book.viewed & cand_pids) < len(cand_pids):
            missing = sorted(cand_pids - book.viewed)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Still missing views for closed-set product_ids: "
                        f"{missing}. Call view on the next missing id (or compare several)."
                    ),
                }
            )

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for _attempt in range(4):
            inference = _chat(messages=messages, tools=tools)
            assistant = inference["choices"][0]["message"]
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if not tool_calls:
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {
                            "role": "user",
                            "content": "No tool call. Call a supplied tool now.",
                        },
                    ]
                )
                continue
            if must_message and "message" not in _tool_names(tool_calls):
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": "Rejected. Call only `message` to answer the shopper.",
                        },
                    ]
                )
                continue
            if not must_message:
                violation = _closed_set_violation(
                    tool_calls, candidates=candidates, viewed=book.viewed
                )
                if not violation:
                    violation = _finish_violation(
                        tool_calls,
                        book=book,
                        query=query,
                        cap=cap,
                        allowed_fields=allowed_fields,
                    )
                if violation:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": f"Rejected: {violation}"},
                        ]
                    )
                    continue
            break
        else:
            if candidates and len(book.viewed & cand_pids) < len(cand_pids):
                miss = sorted(cand_pids - book.viewed)[0]
                tool_calls = [
                    {
                        "id": f"forced-view-{turn}",
                        "type": "function",
                        "function": {
                            "name": "view",
                            "arguments": json.dumps({"product_id": miss}),
                        },
                    }
                ]
                assistant_content = f"Viewing closed-set candidate {miss}."
            else:
                raise RuntimeError("model returned no valid tool call before completion")

        if must_message:
            msg_only = [
                c
                for c in tool_calls
                if c.get("function", {}).get("name") == "message"
            ]
            if msg_only:
                tool_calls = msg_only[:1]

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        group_id = f"{problem_id}-turn-{turn}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {
                        "name": call["function"]["name"],
                        "args": _arguments(
                            call,
                            query=query,
                            allowed_fields=allowed_fields,
                            book=book,
                        ),
                    },
                }
                for index, call in enumerate(tool_calls, start=1)
            ],
        }
        result = _env_call(envelope)
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for call, row in zip(tool_calls, result.get("calls") or [], strict=False):
            name, parsed = _call_args(call)
            # Re-parse through sanitizer for place_test_order consistency.
            try:
                parsed = _arguments(
                    call, query=query, allowed_fields=allowed_fields, book=book
                )
            except ValueError:
                pass
            obs = row.get("observation") if isinstance(row, dict) else {}
            book.ingest(
                name,
                parsed,
                obs,
                prefer=prefer,
                query=query,
                cap=cap,
            )

        if "message" in _tool_names(tool_calls):
            must_message = False

        for tool_call, call_result in zip(
            tool_calls, result.get("calls") or [], strict=False
        ):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(call_result.get("observation")),
                }
            )
        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            messages.append({"role": "user", "content": content})
            tighter = _parse_budget(str(content))
            if tighter is not None and (cap is None or tighter < cap):
                cap = tighter
            if _looks_like_shopper_intervention(str(content)):
                must_message = True
        if any(
            isinstance(row, dict) and (row.get("observation") or {}).get("done")
            for row in (result.get("calls") or [])
        ):
            break

        if must_message:
            continue
        if candidates and len(book.viewed & cand_pids) >= len(cand_pids):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "All listed candidates viewed. Compare/rank by stated priority, "
                        "inspect_stock the winner, add_to_cart that listed sku, "
                        "inspect_stock again, place_test_order (no open-catalog substitutes)."
                    ),
                }
            )
        elif book.shocked:
            if _color_model_lock(query):
                token = prefer.get("model") or "the preferred model"
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Market shock on the last cart. Search `{token}` again, "
                            "view a different color of that model, inspect_stock, cart "
                            "that live in-cap sku, then place_test_order. Do not reuse "
                            "the shocked sku or the failed color. COMMIT; do not ask permission."
                        ),
                    }
                )
            else:
                bits = []
                inch = _asked_inches(query)
                if inch is not None:
                    bits.append(f"size {inch:g} inch")
                need_res = _asked_resolution(query)
                if need_res:
                    bits.append(f"resolution {need_res}")
                if firm:
                    bits.append(f"brand {firm}")
                keep = (", ".join(bits) + ". ") if bits else ""
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Market shock on the last cart. remove_from_cart if needed, "
                            f"then view another live in-cap listing. Keep firm: {keep}"
                            "Soft brand is optional unless THIS text made brand firm. "
                            "inspect_stock, cart, inspect_stock, place_test_order."
                        ),
                    }
                )
        elif book.color_relaxed and book.cart:
            live = book.live_pairs(cap)
            if live:
                keys = ", ".join(f"{p}::{s}" for p, s in live)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Recovery cart is live and in-cap: {keys}. "
                            "Call place_test_order now (inspect_stock first only if "
                            "you have not inspected after the latest cart)."
                        ),
                    }
                )
        else:
            extra = ""
            if _justification_required(query):
                extra = " Ground claims from facts/options/inspect only."
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Episode not done. Honor Task rules; inspect_stock after cart; "
                        "place_test_order on a fresh in-stock in-cap sku. "
                        "Omit justification unless the shopper asked for a reason."
                        f"{extra}"
                    ),
                }
            )

    return dialogue


__all__ = ["agent_main"]
