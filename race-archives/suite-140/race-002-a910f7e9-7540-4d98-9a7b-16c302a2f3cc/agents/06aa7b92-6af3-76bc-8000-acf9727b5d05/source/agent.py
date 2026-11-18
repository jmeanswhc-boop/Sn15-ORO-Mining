
from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_OR = {
    "kimi": "moonshotai/kimi-k2.6",
    "deepseek": "deepseek/deepseek-v3.2",
    "qwen": "qwen/qwen3.5-397b-a17b",
    "glm": "z-ai/glm-5.2",
    "deepseek_chat": "deepseek/deepseek-chat-v3.1",
}
_TEE = {
    "kimi": "moonshotai/Kimi-K2.6-TEE",
    "deepseek": "deepseek-ai/DeepSeek-V3.2-TEE",
    "qwen": "Qwen/Qwen3.5-397B-A17B-TEE",
    "glm": "zai-org/GLM-5.2-TEE",
    "deepseek_chat": "deepseek-ai/DeepSeek-V3.1-TEE",
}
# Content-driven ladders. Justification never glm-first.
_LANE_ROSTER = {
    "intent": ("kimi", "deepseek", "qwen", "glm", "deepseek_chat"),
    "retrieval": ("kimi", "qwen", "deepseek", "deepseek_chat", "glm"),
    "constraint": ("deepseek", "kimi", "qwen", "glm", "deepseek_chat"),
    "preference": ("kimi", "deepseek", "qwen", "deepseek_chat", "glm"),
    "ranking": ("deepseek", "qwen", "kimi", "deepseek_chat", "glm"),
    "recovery": ("deepseek", "kimi", "qwen", "glm", "deepseek_chat"),
    "justification": ("deepseek", "kimi", "deepseek_chat", "qwen", "glm"),
    "generic": ("deepseek", "kimi", "qwen", "glm", "deepseek_chat"),
}

# Fallback only if this episode's tool schema has no claim enum.
# Justification EnvPack schemas often omit the enum and only describe allowed
# fields in prose — still permit the checkable spec fields the grader requires.
_DEFAULT_CLAIM_FIELDS = frozenset(
    {"brand", "category", "in_stock", "price", "refresh_rate", "resolution", "screen_size"}
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
    r"max(?:imum)? of|within|budget (?:is|of)|only go up to|can spend up to|"
    r"at most)\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?",
    re.I,
)
_PREF_PAIR_RE = re.compile(
    r"prefer(?:ably)?(?:\s+the)?\s+([A-Za-z]+)\s+"
    r"([A-Za-z0-9][A-Za-z0-9.\-]*\d[A-Za-z0-9.\-]*)\b",
    re.I,
)
_ASK_FIRST_RE = re.compile(
    r"if you ask|ask me|withheld until|when you ask|until you ask",
    re.I,
)
_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(?:inch(?:es)?|in\b|''|\")", re.I)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×*]\s*(\d{3,5})")
_EPS = 1e-6

_proxy = ProxyClient(timeout=140, max_retries=2)

_SYSTEM = """You are an ORO Bench shopping agent on EnvPack tools.

Obey this episode's shopper text + Task rules only. Hidden races differ from any
local pack — never memorize product IDs. Use only supplied tools. Continue until an
observation has done=true. Exactly one tool call per turn.

## 0828 shopping habits
- Split the request by price/spec boundaries; each budget marker ends one item.
- Firm: category, named hardware, budget, in-stock, brands marked required.
- Soft: ideally/prefer brand — honor on FIRST cart when a fully compliant preferred
  item exists; never let a soft pref override a failed firm spec.
- Structured facts (view/compare options, labelled attrs, exact sku) beat SEO titles.
- Title is unordered token evidence, not a sentence to interpret.
- Recovery: preferred color+model first; then same model other color; then other models.
  When stock dies, message a COMMIT (what you will buy next), never "should I?".

## Dialogue
If the shopper asks a question, lowers budget, or reveals a withheld spec, call
`message` once, then continue shopping. Empty pings are not new questions.
If a firm spec is withheld until you ask: first call is `message` asking for it.

## Closed candidate ranking
When shopper text lists two or more `product_id::sku` keys, that list IS the catalog.
Do not search/filter for substitutes. view (or compare) each listed product_id.
Rank by the priority THIS shopper stated, in the order the text names it.
First cart and final order must be a listed key. After a market shock, re-rank
remaining listed keys.

## Retrieval / constraints / prefs / justification
- Retrieval: several differently worded searches; survey distinct product_ids.
- Constraints: the exact sku's options; category_path from view/compare.
- Prefs: budget+stock hard; match use-case without defaulting to cheapest.
- Justification: only if shopper asked why/reason. Claims = schema fields whose
  values you observed. Omit the justification object otherwise.

## Finish
view/compare → inspect_stock → add_to_cart → inspect_stock again (market may move
the moment you cart) → place_test_order. Never order out-of-stock or over budget.
Never order without a fresh inspect after the latest cart add.
"""


def _provider() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw in ("chutes", "openrouter"):
        return raw
    if getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "chutes"


def _shopper_text(query: str) -> str:
    return query.split("Task rules:", 1)[0].strip()


def _parse_budget(text: str) -> float | None:
    best = None
    for m in _BUDGET_RE.finditer(text or ""):
        whole = m.group(1).replace(",", "")
        frac = m.group(2) or "0"
        try:
            val = float(f"{whole}.{frac}")
        except ValueError:
            continue
        if best is None or val < best:
            best = val
    return best


def _parse_candidate_variants(query: str) -> list[tuple[str, str]]:
    """Closed set iff this episode's shopper text lists ≥2 pid::sku tokens."""
    head = _shopper_text(query)
    pairs = _CAND_KEY_RE.findall(head)
    if len(pairs) < 2:
        return []
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
    return any(n in head for n in ("why", "reason", "justif"))


def _ask_first(query: str) -> bool:
    return bool(_ASK_FIRST_RE.search(_shopper_text(query)))


def _rank_axes(query: str) -> list[tuple[str, int]]:
    """Sort keys from THIS shopper text: (field, sign) with sign -1 = higher wins."""
    head = _shopper_text(query).lower()
    hits: list[tuple[int, str, int]] = []
    patterns = (
        (r"highest\s+refresh|max(?:imum)?\s+refresh|refresh\s+rate", "refresh_rate", -1),
        (r"largest\s+screen|largest\s+size|biggest\s+screen|largest\s+display", "screen_size", -1),
        (r"lowest\s+price|lowest\s+cost|cheapest|lowest\s+priced", "price", 1),
    )
    for pat, field, sign in patterns:
        m = re.search(pat, head)
        if m:
            hits.append((m.start(), field, sign))
    hits.sort()
    if hits:
        return [(field, sign) for _pos, field, sign in hits]
    return [("price", 1)]


def _pref_pair(query: str) -> tuple[str | None, str | None]:
    m = _PREF_PAIR_RE.search(_shopper_text(query))
    if not m:
        return None, None
    return m.group(1).lower(), m.group(2)


def _lane_kind(query: str) -> str:
    head = _shopper_text(query).lower()
    if _parse_candidate_variants(query):
        return "ranking"
    if _justification_required(query):
        return "justification"
    color, model = _pref_pair(query)
    if model:
        return "recovery"
    if _ask_first(query):
        return "constraint"
    if re.search(r"\bmatches\b", head):
        return "retrieval"
    if any(w in head for w in ("prefer", "preferably", "ideally", "rather than", "tradeoff", "trade-off")):
        if any(w in head for w in ("among", "options", "between", "cheapest", "rather")):
            return "preference"
        return "intent"
    return "generic"


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


def _pixel_class(long_side: int, short_side: int) -> str | None:
    if long_side >= 3840 and short_side >= 2160:
        return "uhd"
    if long_side >= 2560 and short_side >= 1440:
        return "qhd"
    if long_side >= 1920 and short_side >= 1080:
        return "fhd"
    if long_side >= 1280 and short_side >= 720:
        return "hd"
    return None


def _res_class(text: str) -> str | None:
    low = (text or "").lower()
    m = _WXH_RE.search(low)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        pair = sorted((a, b), reverse=True)
        cls = _pixel_class(pair[0], pair[1])
        if cls:
            return cls
    if re.search(r"\b(?:4k|uhd|ultra\s*hd|2160p)\b", low):
        return "uhd"
    if re.search(r"\b(?:qhd|1440p|2k)\b", low):
        return "qhd"
    if re.search(r"\b(?:fhd|1080p|full\s*hd)\b", low):
        return "fhd"
    if re.search(r"\b(?:720p|hd)\b", re.sub(r"\b(?:full|ultra)\s*hd\b", "", low)):
        return "hd"
    return None


def _query_wants_resolution_class(text: str) -> str | None:
    """Shopper-named resolution class: hd|fhd|qhd|uhd (first match by specificity)."""
    low = (text or "").lower()
    if re.search(r"\b(?:4k|uhd|ultra\s*hd)\b", low):
        return "uhd"
    if re.search(r"\b(?:qhd|1440p|2k)\b", low):
        return "qhd"
    if re.search(r"\b(?:fhd|1080p|full\s*hd)\b", low):
        return "fhd"
    if re.search(r"\b(?:720p|hd)\b", low):
        return "hd"
    return None


def _query_wants_screen_size(text: str) -> str | None:
    """Shopper-named screen size as canonical `Nin` (e.g. 27in)."""
    hits = list(_INCH_RE.finditer(text or ""))
    if not hits:
        return None
    # Prefer the first explicit size in the shopper request.
    return f"{float(hits[0].group(1)):g}in"


def _query_named_brand(text: str) -> str | None:
    head = text or ""
    m = re.search(
        r"(?:want a|find me a|get me an?|need an?|i want a)\s+(.+?)"
        r"(?=\s+(?:\d+\s*p\b|720p|1080p|1440p|4k|uhd|qhd|fhd|full\s*hd|\bhd\b|"
        r"\d+(?:\.\d+)?-?inch|gaming|smartphone|tablet|monitor|headphone))",
        head,
        flags=re.I,
    )
    if not m:
        return None
    brand = re.sub(r"\s+", " ", m.group(1)).strip(" .,")
    brand = re.sub(r"^(?:an?|the|in-stock|in stock)\s+", "", brand, flags=re.I).strip()
    if not brand or brand.lower() in {"in-stock", "in stock", "an", "a", "the"}:
        return None
    if len(brand) > 40:
        return None
    return brand


def _display_supports_class(display: str, cls: str) -> bool:
    """Whether facts.display can faithfully back a class claim.

    Live TF7: bare `720p` does NOT support claiming `hd`. Need explicit `hd`
    or 1280-class dims. Pixel strings support only their matching class.
    """
    raw = str(display or "").strip()
    if not raw or not cls:
        return False
    low = raw.casefold().replace("×", "x").replace("*", "x")
    parts = [p.strip() for p in re.split(r"[,;/|]", low) if p.strip()] or [low]
    for part in parts:
        if cls == "uhd" and any(x in part for x in ("3840", "2160", "4k", "uhd")):
            return True
        if cls == "qhd" and any(x in part for x in ("2560", "1440", "qhd", "2k")):
            return True
        if cls == "fhd" and any(x in part for x in ("1920", "1080", "fhd", "full hd")):
            return True
        if cls == "hd" and (
            re.search(r"(?<![a-z])hd(?![a-z])", part) or "1280" in part
        ):
            return True
    return False


def _size_vals(text: str) -> set[str]:
    return {f"{float(m.group(1)):g}in" for m in _INCH_RE.finditer(text or "")}


def _one_val(values: set[str]) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


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
        # Accept both `27in` canonical and free text → claim as "N inches".
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:inch|in|\"|'')?", text, flags=re.I)
        if m:
            return f"{float(m.group(1)):g} inches"
    if field == "resolution":
        low = text.lower()
        if low in ("hd", "fhd", "qhd", "uhd", "4k"):
            return "uhd" if low == "4k" else low
        cls = _res_class(text)
        if cls:
            return cls
    return text


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _stock_flag(value: Any) -> bool | None:
    if value is None:
        return None
    if value is True:
        return True
    if value is False:
        return False
    low = str(value).strip().lower()
    if low in ("true", "1", "yes", "in stock", "instock"):
        return True
    if low in ("false", "0", "no", "out of stock", "oos"):
        return False
    return None


def _measure_text(field: str, text: str) -> float | None:
    blob = text or ""
    if field == "refresh_rate":
        m = _HZ_RE.search(blob)
        if m:
            return float(m.group(1))
        if re.fullmatch(r"\d+(?:\.\d+)?", blob.strip()):
            return float(blob.strip())
        return None
    if field == "screen_size":
        m = _INCH_RE.search(blob)
        if m:
            return float(m.group(1))
        if re.fullmatch(r"\d+(?:\.\d+)?", blob.strip()):
            return float(blob.strip())
        return None
    if field == "price":
        return _to_float(blob)
    return None


def _tool_names_of(tools: list[dict[str, Any]]) -> set[str]:
    names = set()
    for tool in tools:
        fn = tool.get("function") or tool
        n = fn.get("name") if isinstance(fn, dict) else None
        if n:
            names.add(str(n))
    return names


class EpisodeNeed:
    """Firm/soft facts parsed from this episode's public query only."""

    def __init__(self, query: str, tools: list[dict[str, Any]]) -> None:
        self.query = query
        self.shopper = _shopper_text(query)
        self.budget = _parse_budget(self.shopper)
        self.candidates = _parse_candidate_variants(query)
        self.cand_pids = {pid for pid, _ in self.candidates}
        self.cand_keys = {f"{pid}::{sku}" for pid, sku in self.candidates}
        self.lane = _lane_kind(query)
        self.ask_first = _ask_first(query)
        self.want_reason = _justification_required(query)
        self.rank_axes = _rank_axes(query)
        self.pref_color, self.pref_model = _pref_pair(query)
        self.allowed_fields = _claim_field_allowlist(tools)
        self.tool_names = _tool_names_of(tools)
        self.named_brand = _query_named_brand(self.shopper)
        self.specs: dict[str, str] = {}
        size = _query_wants_screen_size(self.shopper)
        if size:
            self.specs["screen_size"] = size
        res = _query_wants_resolution_class(self.shopper)
        if res:
            self.specs["resolution"] = res
        # Gate is the first spec the shopper stated.
        self.spec_field: str | None = None
        self.spec_value: str | None = None
        for field, value in self.specs.items():
            self.spec_field, self.spec_value = field, value
            break

    def adopt_budget(self, text: str) -> None:
        new = _parse_budget(text)
        if new is None:
            return
        if self.budget is None or new < self.budget:
            self.budget = new


class StockBook:
    """Observation ledger for this episode — never a memorized catalog."""

    def __init__(self) -> None:
        self.seq = 0
        self.viewed: set[str] = set()
        self.pages: dict[str, dict[str, Any]] = {}
        self.facts: dict[str, dict[str, Any]] = {}
        self.titles: dict[str, str] = {}
        self.brands: dict[str, str] = {}
        self.options: dict[str, str] = {}
        self.variant: dict[str, dict[str, Any]] = {}
        self.inspect: dict[str, dict[str, Any]] = {}
        self.inspect_seq: dict[str, int] = {}
        self.cart: list[tuple[str, str]] = []
        self.last_cart_seq: int | None = None
        self.adds = 0
        self.dead: set[str] = set()
        self.shock_told = False
        self.asked = False
        self.event_seen = False
        self.needs_rerank = False
        self.cart_uptake_done = False
        self.post_event_checked: set[str] = set()
        self.recovery_searches = 0
        self.find_queue: list[str] = []
        self.pref_anchor_title = ""
        self.pref_anchor_brand = ""
        self.pref_anchor_pid = ""

    @staticmethod
    def vk(pid: Any, sku: Any) -> str:
        return f"{pid}::{sku or ''}"

    def note_view_args(self, pid: str) -> None:
        if pid:
            self.viewed.add(str(pid))

    def ingest(self, name: str, args: dict[str, Any], obs: Any) -> None:
        self.seq += 1
        # EnvPack wraps every tool result as {observation: {...}, done, error}.
        # Ledger must read the INNER payload or stock/price shocks are invisible.
        payload = obs if isinstance(obs, dict) else {}
        if payload.get("error"):
            return
        if isinstance(payload.get("observation"), dict) and (
            "done" in payload or "error" in payload or "observation" in payload
        ):
            # Prefer inner body when present; keep outer error already checked.
            inner = payload.get("observation")
            if isinstance(inner, dict):
                payload = inner
        if name in ("view", "compare"):
            pid_arg = str(args.get("product_id") or "")
            if pid_arg:
                self.viewed.add(pid_arg)
            for pid in args.get("product_ids") or []:
                if pid:
                    self.viewed.add(str(pid))
            for prod in _iter_products(payload):
                self._page(prod)
        elif name in ("search", "filter"):
            for row in payload.get("results") or []:
                if isinstance(row, dict) and row.get("product_id") and row.get("sku"):
                    self._variant_row(row)
                    pid = str(row.get("product_id") or "")
                    if pid and pid not in self.viewed and pid not in self.find_queue:
                        self.find_queue.append(pid)
                    title = str(row.get("title") or "")
                    if pid and title and pid not in self.titles:
                        self.titles[pid] = title
                    brand = row.get("brand")
                    if pid and brand and pid not in self.brands:
                        self.brands[pid] = str(brand)
        elif name == "inspect_stock":
            pid = str(payload.get("product_id") or args.get("product_id") or "")
            sku = str(payload.get("sku") or args.get("sku") or "")
            if pid:
                flag = _stock_flag(payload.get("in_stock"))
                price = _to_float(payload.get("price") or payload.get("current_price"))
                key = self.vk(pid, sku)
                prev = self.inspect.get(key) or {}
                self.inspect[key] = {
                    "in_stock": True if flag is None else flag,
                    "price": price if price is not None else prev.get("price"),
                    "pid": pid,
                    "sku": sku,
                }
                self.inspect_seq[key] = self.seq
                if flag is False:
                    self.dead.add(key)
                if self.event_seen:
                    self.post_event_checked.add(key)
        elif name == "add_to_cart":
            added = payload.get("added") if isinstance(payload.get("added"), dict) else {}
            pid = str(added.get("product_id") or args.get("product_id") or "")
            sku = str(added.get("sku") or args.get("sku") or "")
            if pid and sku:
                self.adds += 1
                pair = (pid, sku)
                if pair not in self.cart:
                    self.cart.append(pair)
                self.last_cart_seq = self.seq
                self.cart_uptake_done = False
                # Remember the preferred listing identity for same-model recovery.
                if not self.pref_anchor_title:
                    self.pref_anchor_title = self.titles.get(pid, "")
                    self.pref_anchor_brand = self.brands.get(pid, "")
                    self.pref_anchor_pid = pid
                # Cart-time stock/price is authoritative for the market event.
                flag = _stock_flag(added.get("in_stock"))
                price = _to_float(added.get("price") or added.get("current_price"))
                key = self.vk(pid, sku)
                prev = self.inspect.get(key) or {}
                self.inspect[key] = {
                    "in_stock": True if flag is None else flag,
                    "price": price if price is not None else prev.get("price"),
                    "pid": pid,
                    "sku": sku,
                }
                self.inspect_seq[key] = self.seq
                if flag is False:
                    self.dead.add(key)
            self._sync_cart(payload)
            self._note_market_event(payload.get("market_event"))
        elif name == "remove_from_cart":
            self._sync_cart(payload)
            removed = payload.get("removed") if isinstance(payload.get("removed"), dict) else {}
            pair = (
                str(removed.get("product_id") or args.get("product_id") or ""),
                str(removed.get("sku") or args.get("sku") or ""),
            )
            if pair[0] and pair in self.cart:
                self.cart.remove(pair)
            elif args.get("product_id") and args.get("sku"):
                pair2 = (str(args["product_id"]), str(args["sku"]))
                self.cart = [c for c in self.cart if c != pair2]
            self.cart_uptake_done = False
        elif name == "inspect_cart":
            self._sync_cart(payload)
            if self.cart:
                self.cart_uptake_done = True
        elif name == "message":
            self.asked = True

    def _note_market_event(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        self.event_seen = True
        self.needs_rerank = True
        self.post_event_checked = set()
        target = event.get("target") if isinstance(event.get("target"), dict) else {}
        pid = str(target.get("product_id") or "")
        sku = str(target.get("sku") or "")
        if not pid:
            return
        key = self.vk(pid, sku)
        after = _stock_flag(event.get("in_stock_after"))
        new_price = _to_float(event.get("new_price"))
        prev = self.inspect.get(key) or {}
        flag = after if after is not None else prev.get("in_stock")
        price = new_price if new_price is not None else prev.get("price")
        self.inspect[key] = {
            "in_stock": flag if flag is not None else True,
            "price": price,
            "pid": pid,
            "sku": sku,
        }
        self.inspect_seq[key] = self.seq
        if after is False or str(event.get("kind") or "").lower() in (
            "stockout",
            "out_of_stock",
            "oos",
        ):
            self.dead.add(key)

    def _page(self, prod: dict[str, Any]) -> None:
        pid = str(prod.get("product_id") or "")
        if not pid:
            return
        self.viewed.add(pid)
        self.pages[pid] = prod
        facts = prod.get("facts")
        if isinstance(facts, dict):
            self.facts[pid] = facts
        title = str(prod.get("title") or "")
        if title:
            self.titles[pid] = title
        brand = prod.get("brand")
        if brand:
            self.brands[pid] = str(brand)
        for v in prod.get("variants") or []:
            if isinstance(v, dict):
                self._variant_row(v, default_pid=pid)

    def _variant_row(self, row: dict[str, Any], default_pid: str = "") -> None:
        pid = str(row.get("product_id") or default_pid or "")
        sku = str(row.get("sku") or "")
        if not pid or not sku:
            return
        key = self.vk(pid, sku)
        flag = _stock_flag(row.get("in_stock"))
        price = _to_float(row.get("price") or row.get("current_price"))
        prev = self.variant.get(key) or {}
        self.variant[key] = {
            "in_stock": prev.get("in_stock") if flag is None else flag,
            "price": price if price is not None else prev.get("price"),
        }
        opt = row.get("options")
        if opt:
            self.options[key] = str(opt)

    def _sync_cart(self, payload: dict[str, Any]) -> None:
        rows = payload.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [
            (str(r.get("product_id")), str(r.get("sku") or ""))
            for r in rows
            if isinstance(r, dict) and r.get("product_id")
        ]
        if pairs or rows == []:
            self.cart = [(p, s) for p, s in pairs if p]
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            key = self.vk(r.get("product_id"), r.get("sku"))
            price = _to_float(r.get("current_price") or r.get("price"))
            flag = _stock_flag(r.get("current_in_stock") if "current_in_stock" in r else r.get("in_stock"))
            prev = self.inspect.get(key) or {}
            if price is None and flag is None:
                continue
            self.inspect[key] = {
                "in_stock": prev.get("in_stock", True) if flag is None else flag,
                "price": price if price is not None else prev.get("price"),
                "pid": str(r.get("product_id")),
                "sku": str(r.get("sku") or ""),
            }
            self.inspect_seq[key] = self.seq
            if flag is False:
                self.dead.add(key)

    def snapshot(self, pid: str, sku: str) -> dict[str, Any]:
        key = self.vk(pid, sku)
        out = dict(self.variant.get(key) or {})
        if key in self.inspect:
            out.update(self.inspect[key])
            out["inspected"] = True
        else:
            out["inspected"] = False
        return out

    def live(self, pid: str, sku: str, budget: float | None) -> bool:
        key = self.vk(pid, sku)
        if key in self.dead:
            return False
        snap = self.snapshot(pid, sku)
        if snap.get("in_stock") is False:
            return False
        price = snap.get("price")
        if budget is not None and price is not None and price > budget + _EPS:
            return False
        return True

    def fresh(self, pid: str, sku: str) -> bool:
        key = self.vk(pid, sku)
        if key not in self.inspect:
            return False
        if self.last_cart_seq is None:
            return True
        return self.inspect_seq.get(key, 0) > self.last_cart_seq

    def blob(self, pid: str, sku: str) -> str:
        facts = self.facts.get(str(pid)) or {}
        bits = [
            self.titles.get(str(pid), ""),
            self.options.get(self.vk(pid, sku), ""),
            " ".join(str(v) for v in facts.values()),
        ]
        for label in ("size", "refresh_rate", "display", "storage"):
            if facts.get(label) is not None:
                bits.append(str(facts[label]))
        return " ".join(bits)

    def measure(self, pid: str, sku: str, field: str) -> float | None:
        if field == "price":
            return _to_float(self.snapshot(pid, sku).get("price"))
        facts = self.facts.get(str(pid)) or {}
        label = {"screen_size": "size", "refresh_rate": "refresh_rate", "resolution": "display"}.get(
            field, field
        )
        # Variant option text wins over parent facts (multi-SKU parent listings).
        sources: list[Any] = [
            self.options.get(self.vk(pid, sku), ""),
            facts.get(label),
            facts.get(field),
            self.titles.get(str(pid), ""),
        ]
        for source in sources:
            if source is None:
                continue
            got = _measure_text(field, str(source))
            if got is not None:
                return got
        return _measure_text(field, self.blob(pid, sku))

    def fact_str(self, pid: str, field: str) -> str | None:
        facts = self.facts.get(str(pid)) or {}
        label = {"screen_size": "size", "refresh_rate": "refresh_rate", "resolution": "display"}.get(
            field, field
        )
        raw = facts.get(label) or facts.get(field)
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        if field == "resolution":
            return _res_class(text) or text
        if field == "refresh_rate":
            n = _measure_text("refresh_rate", text)
            return f"{n:g}Hz" if n is not None else text
        if field == "screen_size":
            n = _measure_text("screen_size", text)
            return f"{n:g} inches" if n is not None else text
        return text


def _fact_spec(book: StockBook, pid: str, field: str) -> str | None:
    """Value the product's labelled facts state for a gated spec, or None."""
    facts = book.facts.get(str(pid)) or {}
    if field == "resolution":
        raw = str(facts.get("display") or facts.get("resolution") or "").strip()
        if not raw:
            return None
        hits = [cls for cls in ("uhd", "qhd", "fhd", "hd") if _display_supports_class(raw, cls)]
        return hits[0] if len(hits) == 1 else None
    if field == "screen_size":
        raw = str(facts.get("size") or facts.get("screen_size") or "").strip()
        if not raw:
            return None
        found = _size_vals(raw)
        if not found and re.fullmatch(r"\d+(?:\.\d+)?", raw):
            return f"{float(raw):g}in"
        return _one_val(found)
    return None


def _option_mentions(field: str, text: str) -> bool:
    blob = text or ""
    if field == "screen_size":
        return bool(_INCH_RE.search(blob))
    if field == "refresh_rate":
        return bool(_HZ_RE.search(blob))
    if field == "resolution":
        return bool(
            _WXH_RE.search(blob)
            or re.search(
                r"\b(?:qhd|fhd|full\s*hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra\s*hd|2160p)\b",
                blob,
                re.I,
            )
        )
    return False


def _variant_spec(book: StockBook, pid: str, sku: str, field: str) -> str | None:
    """Variant-level spec: option text wins when it mentions the concept."""
    opt = book.options.get(book.vk(pid, sku), "")
    if field == "screen_size":
        found = _size_vals(opt)
        if len(found) > 1:
            return None
        if found:
            return next(iter(found))
        if _option_mentions(field, opt):
            return None
        return _fact_spec(book, pid, field)
    if field == "resolution":
        if _option_mentions(field, opt):
            for cls in ("uhd", "qhd", "fhd", "hd"):
                if _display_supports_class(opt, cls):
                    return cls
            return None
        return _fact_spec(book, pid, field)
    return _fact_spec(book, pid, field)


def _justification_ok(book: StockBook, pid: str, sku: str, need: EpisodeNeed) -> bool:
    """Justification gold: labelled facts (and the SKU) must state the gated spec."""
    if not need.want_reason:
        return True
    if need.named_brand:
        brand = book.brands.get(str(pid), "")
        title = book.titles.get(str(pid), "")
        want = need.named_brand.lower()
        if want not in brand.lower() and want not in title.lower():
            return False
    for field, want in need.specs.items():
        if _fact_spec(book, pid, field) != want:
            return False
        if _variant_spec(book, pid, sku, field) != want:
            return False
    return True


def _iter_products(obs: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if obs.get("product_id") and (
        obs.get("facts") or obs.get("variants") or obs.get("title") or obs.get("brand")
    ):
        out.append(obs)
    inner = obs.get("product")
    if isinstance(inner, dict) and inner.get("product_id"):
        out.append(inner)
    for key in ("compared", "products", "items"):
        block = obs.get(key)
        if isinstance(block, list):
            for row in block:
                if isinstance(row, dict) and row.get("product_id"):
                    out.append(row)
    return out


def _sanitize_place_order_args(
    args: dict[str, Any],
    *,
    need: EpisodeNeed,
) -> dict[str, Any]:
    out = dict(args)
    if not need.want_reason:
        out.pop("justification", None)
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
        field = _normalize_claim_field(str(claim.get("field", "")), need.allowed_fields)
        if field is None or field in seen or field == "category":
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


def _claims_from_book(
    book: StockBook, pid: str, sku: str, need: EpisodeNeed
) -> list[dict[str, str]]:
    if not need.want_reason:
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    # Claim only gated specs the labelled facts actually back. Never invent a
    # class from title pixels or from a bare 720p string when the shopper said hd.
    for field, want in need.specs.items():
        if field not in need.allowed_fields or field in seen:
            continue
        if _fact_spec(book, pid, field) != want:
            continue
        if _variant_spec(book, pid, sku, field) != want:
            continue
        value = _normalize_claim_value(field, want)
        out.append({"field": field, "value": value})
        seen.add(field)
    snap = book.snapshot(pid, sku)
    if "price" in need.allowed_fields and snap.get("price") is not None and "price" not in seen:
        out.append({"field": "price", "value": _normalize_claim_value("price", snap["price"])})
    if "in_stock" in need.allowed_fields and "in_stock" not in seen:
        out.append({"field": "in_stock", "value": "true" if snap.get("in_stock") is not False else "false"})
    brand = book.brands.get(str(pid))
    if brand and "brand" in need.allowed_fields and "brand" not in seen:
        out.append({"field": "brand", "value": brand})
    return out


def _order_args(book: StockBook, pid: str, sku: str, need: EpisodeNeed) -> dict[str, Any]:
    args: dict[str, Any] = {"product_id": pid, "sku": sku}
    claims = _claims_from_book(book, pid, sku, need)
    if claims:
        args["justification"] = {"claims": claims}
    return _sanitize_place_order_args(args, need=need)


def _arguments(
    tool_call: dict[str, Any],
    *,
    need: EpisodeNeed,
    book: StockBook | None = None,
) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    name = tool_call["function"].get("name") or ""
    if name == "place_test_order":
        pid = str(parsed.get("product_id") or "")
        sku = str(parsed.get("sku") or "")
        if book is not None and need.want_reason and pid and sku:
            return _order_args(book, pid, sku, need)
        return _sanitize_place_order_args(parsed, need=need)
    return parsed


def _looks_like_shopper_intervention(text: str) -> bool:
    low = (text or "").lower().strip()
    if len(low) < 8:
        return False
    cues = (
        "?",
        "which one",
        "which option",
        "hold on",
        "wait,",
        "still available",
        "re-check",
        "change the budget",
        "new budget",
    )
    return any(c in low for c in cues)


def _looks_like_shock(text: str) -> bool:
    low = (text or "").lower()
    return any(
        s in low
        for s in (
            "out of stock",
            "sold out",
            "no longer available",
            "price increased",
            "price went",
            "now costs",
        )
    )


def _model_list(lane: str) -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    table = _TEE if _provider() == "chutes" else _OR
    order = _LANE_ROSTER.get(lane) or _LANE_ROSTER["generic"]
    return [table[k] for k in order]


def _chat(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    lane: str,
) -> dict[str, Any]:
    last_err: Exception | None = None
    for model in _model_list(lane):
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


def _parse_args(call: dict[str, Any]) -> dict[str, Any]:
    raw = (call.get("function") or {}).get("arguments", "{}")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        args = {}
    return args if isinstance(args, dict) else {}


def _closed_set_violation(
    tool_calls: list[dict[str, Any]],
    *,
    need: EpisodeNeed,
    book: StockBook,
) -> str | None:
    if not need.candidates:
        return None
    names = _tool_names(tool_calls)
    viewed = book.viewed
    if any(n in ("search", "filter") for n in names) and len(viewed & need.cand_pids) < min(
        2, len(need.cand_pids)
    ):
        return (
            "Closed candidate set: do not search/filter. "
            f"Call view on listed product_ids first: {sorted(need.cand_pids)}"
        )
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        args = _parse_args(call)
        if name == "view":
            pid = str(args.get("product_id") or "")
            if pid and pid not in need.cand_pids:
                return (
                    f"view product_id {pid} is outside the listed candidates. "
                    f"View only: {sorted(need.cand_pids)}"
                )
        if name == "compare":
            ids = args.get("product_ids") or []
            if isinstance(ids, list):
                bad = [str(x) for x in ids if str(x) not in need.cand_pids]
                if bad:
                    return (
                        f"compare includes ids outside the listed set: {bad}. "
                        f"Compare only: {sorted(need.cand_pids)}"
                    )
        if name in ("add_to_cart", "inspect_stock", "place_test_order"):
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and pid not in need.cand_pids:
                return (
                    f"{name} product_id {pid} is outside listed candidates "
                    f"{sorted(need.cand_pids)}"
                )
            if pid and sku and f"{pid}::{sku}" not in need.cand_keys and name != "inspect_stock":
                if name in ("add_to_cart", "place_test_order"):
                    return (
                        f"{name} {pid}::{sku} is not a listed variant. "
                        f"Use one of: {sorted(need.cand_keys)}"
                    )
    return None


def _commit_block(
    tool_calls: list[dict[str, Any]],
    *,
    need: EpisodeNeed,
    book: StockBook,
) -> str | None:
    """Reject an order/cart the observations cannot support yet."""
    names = _tool_names(tool_calls)
    if "place_test_order" in names:
        for call in tool_calls:
            if (call.get("function") or {}).get("name") != "place_test_order":
                continue
            args = _parse_args(call)
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if not pid or not sku:
                return "place_test_order needs product_id and sku from a viewed variant."
            if (pid, sku) not in book.cart:
                return "add_to_cart that exact variant before place_test_order."
            if not book.fresh(pid, sku):
                return "inspect_stock that exact sku AFTER the latest add_to_cart, then order."
            snap = book.snapshot(pid, sku)
            if snap.get("in_stock") is False:
                return "latest inspect shows out of stock; recover to another in-stock sku."
            price = snap.get("price")
            if need.budget is not None and price is not None and price > need.budget + _EPS:
                return (
                    f"latest price {price:.2f} exceeds firm budget {need.budget:.2f}; "
                    "remove_from_cart and pick an in-budget in-stock sku."
                )
            if need.want_reason and not _justification_ok(book, pid, sku, need):
                return (
                    "place_test_order refused: labelled facts must state the shopper's "
                    "gated spec (and brand); keep searching/viewing a matching item."
                )
    if "add_to_cart" in names:
        for call in tool_calls:
            if (call.get("function") or {}).get("name") != "add_to_cart":
                continue
            args = _parse_args(call)
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if need.want_reason and pid and sku:
                if pid not in book.viewed:
                    return "view the product so labelled facts are known before add_to_cart."
                if not _justification_ok(book, pid, sku, need):
                    return (
                        "do not cart an item whose labelled facts miss the gated spec "
                        "or named brand; view another listing that states it."
                    )
    if "add_to_cart" in names and book.adds == 0:
        for call in tool_calls:
            if (call.get("function") or {}).get("name") != "add_to_cart":
                continue
            args = _parse_args(call)
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and sku and not book.live(pid, sku, need.budget) and book.snapshot(pid, sku).get("inspected"):
                return "do not cart an out-of-stock or over-budget sku; pick a live one."
            if need.pref_model and pid and sku and pid in book.viewed:
                if _relax_level(book, pid, sku, need) != 0:
                    return (
                        f"first cart must be the preferred {need.pref_color or ''} "
                        f"{need.pref_model}".strip()
                        + " exact color/model; keep searching/viewing that pair."
                    )
    if "add_to_cart" in names and book.adds > 0 and need.pref_model:
        for call in tool_calls:
            if (call.get("function") or {}).get("name") != "add_to_cart":
                continue
            args = _parse_args(call)
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if not pid or not sku or pid not in book.viewed:
                continue
            lvl = _relax_level(book, pid, sku, need)
            if lvl > 1:
                # Prefer same-model recovery; reject accessory-level substitutes while early.
                return (
                    f"recover on the same model ({need.pref_model}) with another color before "
                    "changing models; view another listing of that model first."
                )
    return None


def _sort_tuple(book: StockBook, pid: str, sku: str, need: EpisodeNeed) -> tuple:
    row: list[float] = []
    for field, sign in need.rank_axes:
        val = book.measure(pid, sku, field)
        if val is None:
            val = 0.0 if sign < 0 else 1e18
        row.append(sign * val)
    relax = _relax_level(book, pid, sku, need) if need.pref_model else 0
    price = book.measure(pid, sku, "price")
    if price is None:
        price = 1e18
    return (relax, *row, price, f"{pid}::{sku}")


def _listed_pick(book: StockBook, need: EpisodeNeed, *, require_inspect: bool) -> tuple[str, str] | None:
    scored: list[tuple[tuple, str, str]] = []
    for pid, sku in need.candidates:
        key = book.vk(pid, sku)
        if key in book.dead:
            continue
        if pid not in book.viewed:
            continue
        snap = book.snapshot(pid, sku)
        if require_inspect and not snap.get("inspected"):
            continue
        if not book.live(pid, sku, need.budget):
            continue
        if need.want_reason and not _justification_ok(book, pid, sku, need):
            continue
        scored.append((_sort_tuple(book, pid, sku, need), pid, sku))
    if not scored:
        return None
    scored.sort()
    return scored[0][1], scored[0][2]


def _open_pick(book: StockBook, need: EpisodeNeed, *, require_inspect: bool) -> tuple[str, str] | None:
    scored: list[tuple[tuple, str, str]] = []
    keys = set(book.inspect) | set(book.variant)
    for key in keys:
        pid, _, sku = key.partition("::")
        if not sku or key in book.dead:
            continue
        if pid not in book.viewed:
            continue
        snap = book.snapshot(pid, sku)
        if require_inspect and not snap.get("inspected"):
            continue
        if not book.live(pid, sku, need.budget):
            continue
        if need.want_reason and not _justification_ok(book, pid, sku, need):
            continue
        scored.append((_sort_tuple(book, pid, sku, need), pid, sku))
    if not scored:
        return None
    scored.sort()
    return scored[0][1], scored[0][2]


def _winner(book: StockBook, need: EpisodeNeed, *, require_inspect: bool) -> tuple[str, str] | None:
    if need.candidates:
        return _listed_pick(book, need, require_inspect=require_inspect)
    carted = [
        (p, s)
        for p, s in book.cart
        if book.live(p, s, need.budget)
        and (not need.want_reason or _justification_ok(book, p, s, need))
    ]
    if carted:
        if require_inspect and not book.snapshot(*carted[-1]).get("inspected"):
            pass
        else:
            return carted[-1]
    return _open_pick(book, need, require_inspect=require_inspect)


def _commit_text(need: EpisodeNeed, nxt: tuple[str, str] | None) -> str:
    if need.pref_model:
        color = f" (keeping {need.pref_model}, changing color" + (
            f" from {need.pref_color}" if need.pref_color else ""
        ) + ")"
        return (
            f"The previous option is unavailable. I will stay on {need.pref_model}"
            f"{color} and order the next in-stock variant that still meets the firm specs."
        )
    if nxt:
        return (
            "The previous option is unavailable. I will take the next surviving "
            f"option {nxt[0]}::{nxt[1]} that still meets the firm specs."
        )
    return (
        "The previous option is unavailable. I will switch to the next in-stock, "
        "in-budget option that still meets the firm specs."
    )


def _ask_text() -> str:
    return (
        "Before I search: which hardware specification is non-negotiable, "
        "and what exact value do you need?"
    )


def _force(name: str, args: dict[str, Any], turn: int, tag: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"pilot-{tag}-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
    ]


_ACCESSORY_TITLE_RE = re.compile(
    r"\b(?:battery|bagpack|backpack|jacket|shorts|refill|basin|motor|holder|mount|"
    r"clamp|charger|cable|case|cover|light\s*bar|underwear|dress|denim|handlebar|"
    r"cotton|cartoon|utility)\b",
    re.I,
)
_PHONE_TITLE_RE = re.compile(
    r"\b(?:smartphone|smart\s*phone|mobile\s*phone|cellphone|cell\s*phone|unlocked)\b",
    re.I,
)


def _model_match(book: StockBook, pid: str, need: EpisodeNeed) -> bool:
    """True when this listing is the preferred model as a phone, not an accessory decoy."""
    if not need.pref_model:
        return True
    model = re.sub(r"[^a-z0-9]+", "", need.pref_model.lower())
    if not model:
        return False
    title = book.titles.get(pid, "")
    facts = book.facts.get(pid) or {}
    fact_model = re.sub(r"[^a-z0-9]+", "", str(facts.get("model") or "").lower())
    if _ACCESSORY_TITLE_RE.search(title) and not _PHONE_TITLE_RE.search(title):
        return False
    if fact_model and model in fact_model:
        # facts.model hit is strong when the title is phone-like or not an accessory.
        if _PHONE_TITLE_RE.search(title) or not _ACCESSORY_TITLE_RE.search(title):
            return True
    if _PHONE_TITLE_RE.search(title) and re.search(
        r"(?<![a-z0-9])" + re.escape(need.pref_model.lower()) + r"(?![a-z0-9])",
        title.lower(),
    ):
        return True
    # Anchor-title twin listings (other color of the same carted phone).
    if book.pref_anchor_title and title == book.pref_anchor_title and pid != book.pref_anchor_pid:
        return True
    return False


def _relax_level(book: StockBook, pid: str, sku: str, need: EpisodeNeed) -> int:
    if not need.pref_model:
        return 0
    shade = (
        book.options.get(book.vk(pid, sku), "")
        + " "
        + book.titles.get(pid, "")
        + " "
        + str((book.facts.get(pid) or {}).get("color") or "")
    ).lower()
    same_model = _model_match(book, pid, need)
    same_color = bool(need.pref_color) and re.search(
        r"(?<![a-z])" + re.escape(need.pref_color) + r"(?![a-z])", shade
    )
    if same_model and same_color:
        return 0
    if same_model:
        return 1
    return 2 if same_color else 3


def _anchor_title_hit(book: StockBook, pid: str) -> bool:
    title = book.titles.get(pid, "")
    if book.pref_anchor_title and title == book.pref_anchor_title:
        return True
    return False


def _endgame_move(
    *,
    need: EpisodeNeed,
    book: StockBook,
    turn: int,
    max_steps: int,
    must_message: bool,
    msg_waits: int,
) -> tuple[str, dict[str, Any]] | None:
    """Controller owns cart / re-inspect / order when evidence is enough.

    Returns None so the LLM keeps exploring.
    """
    remain = max_steps - turn + 1
    tight = remain <= 8
    names = need.tool_names

    if must_message:
        # Broken cart after a market shock: commit immediately (do not wait for LLM).
        cart_broken = any(
            book.vk(p, s) in book.dead or not book.live(p, s, need.budget)
            for p, s in book.cart
        )
        if msg_waits < 2 and not cart_broken:
            return None
        if "message" in names:
            nxt = _winner(book, need, require_inspect=False)
            if need.ask_first and not book.asked and not cart_broken:
                return ("message", {"content": _ask_text()})
            return ("message", {"content": _commit_text(need, nxt)})
        return None

    if need.ask_first and not book.asked and turn <= 2 and "message" in names:
        return ("message", {"content": _ask_text()})

    if need.candidates:
        missing = [p for p in sorted(need.cand_pids - book.viewed)]
        if missing:
            if "compare" in names and len(missing) > 1:
                return ("compare", {"product_ids": missing[:5]})
            if "view" in names:
                return ("view", {"product_id": missing[0]})

    for pid, sku in list(book.cart):
        key = book.vk(pid, sku)
        snap = book.snapshot(pid, sku)
        dead = key in book.dead or snap.get("in_stock") is False
        over = (
            need.budget is not None
            and snap.get("price") is not None
            and snap["price"] > need.budget + _EPS
        )
        mismatch = bool(
            need.want_reason
            and pid in book.viewed
            and not _justification_ok(book, pid, sku, need)
        )
        if dead or over or mismatch:
            # Uptake cart state after a market shock before committing recovery.
            if not book.cart_uptake_done and "inspect_cart" in names:
                return ("inspect_cart", {})
            nxt = _winner(book, need, require_inspect=False)
            if not book.shock_told and "message" in names:
                return ("message", {"content": _commit_text(need, nxt)})
            if "remove_from_cart" in names:
                return ("remove_from_cart", {"product_id": pid, "sku": sku})
            return None

    # Closed-set ranking: after a shock, re-inspect surviving listed keys before re-carting.
    if need.candidates and book.needs_rerank and not book.cart and "inspect_stock" in names:
        pending = False
        for pid, sku in need.candidates:
            key = book.vk(pid, sku)
            if key in book.dead:
                continue
            if key not in book.post_event_checked:
                pending = True
                return ("inspect_stock", {"product_id": pid, "sku": sku})
        if not pending:
            book.needs_rerank = False

    if book.cart:
        pid, sku = book.cart[-1]
        key = book.vk(pid, sku)
        if key in book.dead:
            return None
        if not book.cart_uptake_done and "inspect_cart" in names:
            return ("inspect_cart", {})
        if not book.fresh(pid, sku) and "inspect_stock" in names:
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        if book.fresh(pid, sku) and book.live(pid, sku, need.budget):
            if need.want_reason and not _justification_ok(book, pid, sku, need):
                if "remove_from_cart" in names:
                    return ("remove_from_cart", {"product_id": pid, "sku": sku})
            elif "place_test_order" in names:
                return ("place_test_order", _order_args(book, pid, sku, need))
        if book.fresh(pid, sku) and not book.live(pid, sku, need.budget):
            book.dead.add(key)
            book.needs_rerank = True
            nxt = _winner(book, need, require_inspect=False)
            if not book.shock_told and "message" in names:
                return ("message", {"content": _commit_text(need, nxt)})
            if "remove_from_cart" in names:
                return ("remove_from_cart", {"product_id": pid, "sku": sku})
            return None

    pick = _winner(book, need, require_inspect=False)
    ranking_ready = bool(need.candidates) and not (need.cand_pids - book.viewed)
    take_over = ranking_ready or tight or (book.adds > 0 and pick is not None)

    if need.lane in ("preference", "intent") and book.adds == 0 and not ranking_ready and not tight:
        return None

    if need.want_reason:
        if pick is None:
            unseen = [p for p in book.find_queue if p not in book.viewed]
            if unseen and "view" in names:
                return ("view", {"product_id": unseen[0]})
            if "search" in names and book.recovery_searches < 5 and not tight:
                book.recovery_searches += 1
                kind = ""
                low = need.shopper.lower()
                if "tablet" in low:
                    kind = "tablet"
                elif "monitor" in low:
                    kind = "gaming monitor" if "gaming" in low else "monitor"
                elif "smartphone" in low or "phone" in low:
                    kind = "smartphone"
                bits = [need.named_brand or ""]
                if need.specs.get("resolution") == "hd":
                    bits.append("hd")
                elif need.specs.get("resolution") == "fhd":
                    bits.append("1080p")
                elif need.specs.get("resolution"):
                    bits.append(need.specs["resolution"])
                if need.specs.get("screen_size"):
                    bits.append(need.specs["screen_size"].replace("in", " inch"))
                bits.append(kind)
                q = " ".join(b for b in bits if b)
                return ("search", {"query": q or need.shopper.split(".")[0], "in_stock": True, "k": 10})
            if not tight:
                return None
        else:
            take_over = True

    if need.lane == "recovery" and need.pref_model and not tight:
        def _title_hit(pid: str) -> bool:
            return _model_match(book, pid, need)

        # Always view search hits that look like the preferred model before carting.
        if "view" in names:
            # Same listing title as the first preferred cart (other color / sku).
            for pid in list(book.find_queue) + list(book.titles):
                if pid in book.viewed:
                    continue
                if _anchor_title_hit(book, pid) or _title_hit(pid):
                    return ("view", {"product_id": pid})
            # Prefer phone-like titles over accessory decoys.
            unseen_phone = [
                p
                for p in book.find_queue
                if p not in book.viewed
                and _PHONE_TITLE_RE.search(book.titles.get(p, ""))
            ]
            if unseen_phone:
                return ("view", {"product_id": unseen_phone[0]})
            unseen = [p for p in book.find_queue if p not in book.viewed]
            if unseen and len(book.viewed) < (6 if book.adds else 4):
                return ("view", {"product_id": unseen[0]})

        exact = None
        same = None
        keys = set(book.inspect) | set(book.variant)
        scored_exact: list[tuple[tuple, str, str]] = []
        scored_same: list[tuple[tuple, str, str]] = []
        for key in keys:
            pid, _, sku = key.partition("::")
            if not sku or key in book.dead or pid not in book.viewed:
                continue
            if not book.live(pid, sku, need.budget):
                continue
            lvl = _relax_level(book, pid, sku, need)
            row = (_sort_tuple(book, pid, sku, need), pid, sku)
            if lvl == 0:
                scored_exact.append(row)
            elif lvl == 1:
                scored_same.append(row)
        if scored_exact:
            scored_exact.sort()
            exact = (scored_exact[0][1], scored_exact[0][2])
        if scored_same:
            scored_same.sort()
            same = (scored_same[0][1], scored_same[0][2])

        if book.adds == 0:
            if exact is None and book.recovery_searches < 2 and "search" in names:
                book.recovery_searches += 1
                q = f"{need.pref_model} smartphone"
                if need.pref_color:
                    q = f"{need.pref_color} {need.pref_model} smartphone"
                return ("search", {"query": q, "in_stock": True, "k": 10})
            if exact is None and turn < max(6, int(max_steps * 0.35)) and book.find_queue:
                miss = next((p for p in book.find_queue if p not in book.viewed), None)
                if miss and "view" in names:
                    return ("view", {"product_id": miss})
                return None
            if exact:
                pick = exact
                take_over = True
            elif turn < max(8, int(max_steps * 0.45)):
                return None
        else:
            # Post-shock: same model other color before other models.
            if exact:
                pick = exact
                take_over = True
            elif same:
                pick = same
                take_over = True
            else:
                # Search then view anchor-title hits; do not cart unrelated accessories.
                unseen_anchor = [
                    p
                    for p in book.find_queue
                    if p not in book.viewed and (_anchor_title_hit(book, p) or _title_hit(p))
                ]
                if unseen_anchor and "view" in names:
                    return ("view", {"product_id": unseen_anchor[0]})
                if book.recovery_searches < 6 and "search" in names:
                    book.recovery_searches += 1
                    q = book.pref_anchor_title or f"{need.pref_model} smartphone"
                    return ("search", {"query": q, "in_stock": True, "k": 10})
                # Last resort only when tight on steps.
                if not tight:
                    return None

    if pick is None:
        if ranking_ready and "inspect_stock" in names:
            for pid, sku in need.candidates:
                if book.vk(pid, sku) not in book.dead and pid in book.viewed:
                    if not book.snapshot(pid, sku).get("inspected"):
                        return ("inspect_stock", {"product_id": pid, "sku": sku})
        return None

    if not take_over:
        return None

    # Ranking: inspect every listed survivor before the first commitment.
    if need.candidates and book.adds == 0 and "inspect_stock" in names:
        for pid, sku in need.candidates:
            key = book.vk(pid, sku)
            if key in book.dead:
                continue
            if not book.snapshot(pid, sku).get("inspected"):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
        # Recompute winner after full inspect coverage.
        pick = _winner(book, need, require_inspect=True) or pick

    pid, sku = pick
    if not book.fresh(pid, sku) and "inspect_stock" in names:
        return ("inspect_stock", {"product_id": pid, "sku": sku})
    if (pid, sku) not in book.cart and "add_to_cart" in names:
        if book.live(pid, sku, need.budget):
            return ("add_to_cart", {"product_id": pid, "sku": sku})
    return None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """EnvPack loop: firstking hygiene + StockBook finish pilot."""
    try:
        return _run_episode(problem_data)
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "role": "assistant",
                "content": f"episode error: {type(exc).__name__}: {str(exc)[:240]}",
            }
        ]


def _run_episode(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    # Serial turns: parallel dumps leave market events unclosed.
    max_calls = 1
    tools = policy["tools"]
    query = str(policy.get("query") or "")
    max_steps = int(policy["max_steps"])
    need = EpisodeNeed(query, tools)
    book = StockBook()

    boot = (
        f"{query}\n\n"
        "Follow Task rules. One tool per turn. Reply to real shopper questions with "
        "message once, then continue. After add_to_cart, inspect_stock again before "
        "place_test_order. Never order out-of-stock or over the firm budget."
    )
    if need.ask_first:
        boot += (
            "\n\nASK-FIRST: first call is `message` asking for the withheld firm spec; "
            "then search/view with the revealed value."
        )
    if need.pref_model:
        boot += (
            f"\n\nPREFERRED: model token `{need.pref_model}`"
            + (f", color {need.pref_color}" if need.pref_color else "")
            + ". First cart that exact pair when it exists; on stockout keep the model "
            "and change color before changing models. COMMIT in message — do not ask permission."
        )
    if need.want_reason and need.specs:
        spec_bits = ", ".join(f"{k}={v}" for k, v in need.specs.items())
        boot += (
            "\n\nJUSTIFICATION: view labelled facts before the first cart. Only cart a "
            "variant whose facts.display/size state "
            f"{spec_bits}"
            + (f" and brand {need.named_brand}" if need.named_brand else "")
            + ". Bare 720p is not an hd claim. After a shock, recover to another "
            "fact-backed match — never a conflicting size/resolution listing."
        )
    if need.candidates:
        listed = ", ".join(f"{pid}::{sku}" for pid, sku in need.candidates)
        boot += (
            "\n\nCLOSED CANDIDATE SET (mandatory): "
            f"{listed}. "
            "compare/view each product_id. Rank by the shopper's stated priority using "
            "variant option text over conflicting parent facts. Inspect_stock every "
            "listed survivor before the first add_to_cart. After a market shock: "
            "inspect_cart, COMMIT message, remove_from_cart, re-inspect_stock every "
            "remaining listed key, then re-rank and order. Do not search the open catalog."
        )
    if need.budget is not None:
        boot += f"\n\nFIRM BUDGET: {need.budget:.2f}. Never place_test_order above this."

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": boot},
    ]
    dialogue: list[dict[str, Any]] = []
    must_message = bool(need.ask_first)
    msg_waits = 0

    for turn in range(1, max_steps + 1):
        if must_message:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "REQUIRED this turn: call `message` once to answer the shopper "
                        "(COMMIT the next buy if stock/budget broke; do not ask permission), "
                        "then next turns resume shopping/ranking."
                    ),
                }
            )
        elif need.candidates and len(book.viewed & need.cand_pids) < len(need.cand_pids):
            missing = sorted(need.cand_pids - book.viewed)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Still missing views for closed-set product_ids: "
                        f"{missing}. Call view on the next missing id (or compare several)."
                    ),
                }
            )

        move = _endgame_move(
            need=need,
            book=book,
            turn=turn,
            max_steps=max_steps,
            must_message=must_message,
            msg_waits=msg_waits,
        )
        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []

        if move is not None:
            name, args = move
            assistant_content = f"Finish pilot: {name}."
            tool_calls = _force(name, args, turn, name.replace("_", ""))[:max_calls]
            # Forced orders must still pass the same stock/budget gate.
            block = _commit_block(tool_calls, need=need, book=book)
            if block:
                messages.append({"role": "user", "content": f"Pilot blocked: {block}"})
                move = None
                tool_calls = []
                assistant_content = ""
        if move is None:
            for _attempt in range(4):
                inference = _chat(messages=messages, tools=tools, lane=need.lane)
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
                    msg_waits += 1
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
                        tool_calls, need=need, book=book
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
                # Always gate orders — even if a message was also required this turn.
                block = _commit_block(tool_calls, need=need, book=book)
                if block:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": f"Rejected: {block}"},
                        ]
                    )
                    continue
                break
            else:
                fallback = _endgame_move(
                    need=need,
                    book=book,
                    turn=turn,
                    max_steps=max_steps,
                    must_message=must_message,
                    msg_waits=99,
                )
                if fallback:
                    name, args = fallback
                    assistant_content = f"Finish pilot fallback: {name}."
                    tool_calls = _force(name, args, turn, "fb")
                elif need.candidates and len(book.viewed & need.cand_pids) < len(need.cand_pids):
                    miss = sorted(need.cand_pids - book.viewed)[0]
                    tool_calls = _force("view", {"product_id": miss}, turn, "view")
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
                        "args": _arguments(call, need=need, book=book),
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

        for call, call_result in zip(tool_calls, result.get("calls") or [], strict=False):
            name = str((call.get("function") or {}).get("name") or "")
            args = _parse_args(call)
            obs = call_result.get("observation") if isinstance(call_result, dict) else {}
            book.ingest(name, args, obs)
            if name == "message":
                book.asked = True
                if any(
                    book.vk(p, s) in book.dead or not book.live(p, s, need.budget)
                    for p, s in book.cart
                ):
                    book.shock_told = True

        if "message" in _tool_names(tool_calls):
            must_message = False
            msg_waits = 0

        for tool_call, call_result in zip(tool_calls, result.get("calls") or [], strict=False):
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
            need.adopt_budget(str(content))
            text = str(content or "").strip()
            if text and _looks_like_shopper_intervention(text):
                # Already committed a shock recovery — do not stall again on message;
                # finish remove / re-rank / reorder.
                cart_broken = any(
                    book.vk(p, s) in book.dead or not book.live(p, s, need.budget)
                    for p, s in book.cart
                )
                if book.shock_told and cart_broken:
                    must_message = False
                else:
                    must_message = True
                    msg_waits = 0
            elif text and _looks_like_shock(text):
                for pid, sku in list(book.cart):
                    book.dead.add(book.vk(pid, sku))
                book.needs_rerank = True
                book.event_seen = True
                book.post_event_checked = set()
        if any(
            isinstance(call, dict) and (call.get("observation") or {}).get("done")
            for call in (result.get("calls") or [])
        ):
            break

        if not must_message and not need.candidates:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Episode not done. Honor Task rules; inspect_stock after cart; "
                        "place_test_order only on a fresh in-stock in-budget sku. "
                        "Omit justification unless the shopper asked for a reason. "
                        "On stockout: COMMIT message, same model other color, then reorder."
                    ),
                }
            )
        elif not must_message and need.candidates and len(book.viewed & need.cand_pids) >= len(
            need.cand_pids
        ):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "All listed candidates viewed. Rank by stated priority from viewed facts, "
                        "inspect_stock the winner, add_to_cart that listed sku, re-inspect, "
                        "place_test_order (no open-catalog substitutes)."
                    ),
                }
            )

    return dialogue


__all__ = ["agent_main"]
