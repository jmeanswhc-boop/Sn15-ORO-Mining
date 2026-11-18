
from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_CHUTES = {
    "k2": "moonshotai/Kimi-K2.6-TEE",
    "ds": "deepseek-ai/DeepSeek-V3.2-TEE",
    "qw": "Qwen/Qwen3.5-397B-A17B-TEE",
    "gl": "zai-org/GLM-5.2-TEE",
    "ds1": "deepseek-ai/DeepSeek-V3.1-TEE",
}
_OR = {
    "k2": "moonshotai/kimi-k2.6",
    "ds": "deepseek/deepseek-v3.2",
    "qw": "qwen/qwen3.5-397b-a17b",
    "gl": "z-ai/glm-5.2",
    "ds1": "deepseek/deepseek-chat-v3.1",
}
# Three rosters only. explain never starts on glm.
_ROSTERS = {
    "find": ("k2", "qw", "ds", "ds1", "gl"),
    "match": ("ds", "qw", "k2", "ds1", "gl"),
    "explain": ("ds", "k2", "ds1", "qw", "gl"),
}

# Used only when this episode's place_test_order schema has no field enum.
# Checkable spec names, not catalog answers. Never claim `category`.
_CLAIM_FALLBACK = frozenset(
    {"brand", "in_stock", "price", "refresh_rate", "resolution", "screen_size"}
)
_ALIAS = {
    "display": "resolution",
    "screen": "screen_size",
    "size": "screen_size",
    "refresh": "refresh_rate",
    "hz": "refresh_rate",
}

_PAIR = re.compile(r"(\d{6,})::([A-Za-z0-9][A-Za-z0-9._-]*)")
_CAP = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|"
    r"max(?:imum)? of|within|budget (?:is|of)|can spend up to|at most|"
    r"only go up to)\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?",
    re.I,
)
_ASK = re.compile(r"if you ask|ask me|when you ask|until you ask|withheld until", re.I)
_PREF = re.compile(
    r"prefer(?:ably)?(?:\s+the)?\s+([A-Za-z]+)\s+"
    r"([A-Za-z0-9][A-Za-z0-9.\-]*\d[A-Za-z0-9.\-]*)\b",
    re.I,
)
_INCH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:inch(?:es)?|in\b|''|\")", re.I)
_HZ = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_XY = re.compile(r"(\d{3,5})\s*[x×*]\s*(\d{3,5})")
_SLACK = 1e-6

_proxy = ProxyClient(timeout=130, max_retries=2)

_VOICE = """You are an EnvPack shopping agent. Use only supplied tools. Stop when
an observation has done=true. One tool call per turn. Obey this episode's shopper
text and Task rules. Do not memorize catalog IDs.

## bno8 retrieval (title then attrs)
Search/filter with the shopper's own content words. After hits come back, view
several distinct product_ids. A listing is promising when:
  1. title tokens match the request (whole words first; numbers/codes extra),
  2. labelled attributes / sku option text overlap those same words,
  3. price sits inside the stated cap.
Do not cart the first search row. Shortlist, view, then pick. Soft brand/color
only after every firm spec, budget, and stock pass.

## Closed lists
If the shopper named two or more product_id::sku keys, that list is the whole
catalog. view/compare those ids. Do not search substitutes. Rank by whatever
priority THIS text stated; if it stated none, prefer the listed variant whose
viewed title+attrs overlap the request, then lower price.

## Dialogue and recovery
A real shopper question or budget change: one `message`, then resume.
If a spec is withheld until you ask: first call is `message`.
Stock dies: COMMIT the next buy (same model, other color before a new model).
Never ask permission.

## Grounded claims
Only if the shopper asked why/reason. Every claim value must appear in viewed
facts, sku options, or the latest inspect. If a term is missing from evidence,
omit that claim — do not invent. Drop justification entirely when not asked.

## Finish
view/compare → inspect_stock → add_to_cart → inspect_cart → inspect_stock again
→ place_test_order. After a cart shock, COMMIT, remove the dead/over-cap sku,
re-rank remaining live options from fresh inspects, then cart a different sku.
Never order the shocked listing. Never order out of stock or over the cap.
"""


def _which_stack() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw in ("chutes", "openrouter"):
        return raw
    if getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "chutes"


def _goal_head(query: str) -> str:
    return query.split("Task rules:", 1)[0].strip()


def _money_cap(text: str) -> float | None:
    best = None
    for m in _CAP.finditer(text or ""):
        whole = m.group(1).replace(",", "")
        frac = m.group(2) or "0"
        try:
            val = float(f"{whole}.{frac}")
        except ValueError:
            continue
        if best is None or val < best:
            best = val
    return best


def _pair_keys(query: str) -> list[tuple[str, str]]:
    found = _PAIR.findall(_goal_head(query))
    if len(found) < 2:
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pid, sku in found:
        sku = sku.rstrip(".,;:)")
        key = f"{pid}::{sku}"
        if key in seen:
            continue
        seen.add(key)
        out.append((pid, sku))
    return out


def _wants_why(query: str) -> bool:
    head = _goal_head(query).lower()
    return any(n in head for n in ("why", "reason", "justif"))


def _needs_ask(query: str) -> bool:
    return bool(_ASK.search(_goal_head(query)))


def _pref_pair(query: str) -> tuple[str | None, str | None]:
    m = _PREF.search(_goal_head(query))
    if not m:
        return None, None
    return m.group(1).lower(), m.group(2)


def _cue_kind(query: str) -> str:
    if _pair_keys(query):
        return "match"
    if _wants_why(query):
        return "explain"
    return "find"


def _word_bits(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) > 2 or any(ch.isdigit() for ch in t)]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _flag_stock(value: Any) -> bool | None:
    if value is True:
        return True
    if value is False:
        return False
    if value is None:
        return None
    low = str(value).strip().lower()
    if low in ("true", "1", "yes", "in stock", "instock"):
        return True
    if low in ("false", "0", "no", "out of stock", "oos"):
        return False
    return None


def _hz_num(text: str) -> float | None:
    found = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*hz", text or "", flags=re.I)]
    if found:
        return max(found)
    if re.fullmatch(r"\d+(?:\.\d+)?", (text or "").strip()):
        return float(text.strip())
    return None


def _unwrap(obs: Any) -> dict[str, Any]:
    """EnvPack wraps every tool result as {observation: {...}, done, error}."""
    payload = obs if isinstance(obs, dict) else {}
    inner = payload.get("observation")
    if isinstance(inner, dict) and (
        "done" in payload or "error" in payload or "observation" in payload
    ):
        return inner
    return payload


def _inch_num(text: str) -> float | None:
    m = _INCH.search(text or "")
    if m:
        return float(m.group(1))
    if re.fullmatch(r"\d+(?:\.\d+)?", (text or "").strip()):
        return float(text.strip())
    return None


def _panel_class(text: str) -> str | None:
    low = (text or "").lower()
    m = _XY.search(low)
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
    if re.search(r"\b(?:4k|uhd|ultra\s*hd|2160p)\b", low):
        return "uhd"
    if re.search(r"\b(?:qhd|1440p|2k)\b", low):
        return "qhd"
    if re.search(r"\b(?:fhd|1080p|full\s*hd)\b", low):
        return "fhd"
    if re.search(r"\b(?:720p|hd)\b", re.sub(r"\b(?:full|ultra)\s*hd\b", "", low)):
        return "hd"
    return None


def _schema_fields(tools: list[dict[str, Any]]) -> frozenset[str]:
    for tool in tools:
        fn = tool.get("function") or tool
        if (fn.get("name") or "") != "place_test_order":
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        field = (
            ((props.get("justification") or {}).get("properties") or {})
            .get("claims", {})
            .get("items", {})
            .get("properties", {})
            .get("field", {})
        )
        enum = field.get("enum")
        if isinstance(enum, list) and enum:
            return frozenset(str(x) for x in enum)
    return _CLAIM_FALLBACK


def _toolset(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        fn = tool.get("function") or tool
        n = fn.get("name") if isinstance(fn, dict) else None
        if n:
            names.add(str(n))
    return names


def _clean_field(name: str, allowed: frozenset[str]) -> str | None:
    raw = str(name or "").strip()
    if not raw:
        return None
    if raw in allowed:
        return raw
    low = raw.lower()
    if low in allowed:
        return low
    mapped = _ALIAS.get(low)
    if mapped and mapped in allowed:
        return mapped
    return None


def _clean_value(field: str, value: Any) -> str:
    text = str(value).strip()
    if field == "in_stock":
        flag = _flag_stock(text)
        if flag is True:
            return "true"
        if flag is False:
            return "false"
    if field == "price":
        n = _as_float(text)
        if n is not None:
            return str(n)
    if field == "refresh_rate":
        n = _hz_num(text)
        if n is not None:
            return f"{n:g}Hz"
    if field == "screen_size":
        n = _inch_num(text)
        if n is not None:
            return f"{n:g} inches"
    if field == "resolution":
        cls = _panel_class(text)
        if cls:
            return cls
    return text


def _axis_list(query: str) -> list[tuple[str, int]]:
    head = _goal_head(query).lower()
    hits: list[tuple[int, str, int]] = []
    specs = (
        (r"highest\s+refresh|max(?:imum)?\s+refresh|refresh\s+rate", "refresh_rate", -1),
        (r"largest\s+screen|largest\s+size|biggest\s+screen", "screen_size", -1),
        (r"lowest\s+price|lowest\s+cost|cheapest", "price", 1),
    )
    for pat, field, sign in specs:
        m = re.search(pat, head)
        if m:
            hits.append((m.start(), field, sign))
    hits.sort()
    return [(field, sign) for _i, field, sign in hits]


def _call_args(call: dict[str, Any]) -> dict[str, Any]:
    raw = (call.get("function") or {}).get("arguments", "{}")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _names(calls: list[dict[str, Any]]) -> list[str]:
    return [str((c.get("function") or {}).get("name") or "") for c in calls]


def _products_in(obs: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if obs.get("product_id") and (
        obs.get("title") or obs.get("facts") or obs.get("variants") or obs.get("brand")
    ):
        rows.append(obs)
    inner = obs.get("product")
    if isinstance(inner, dict) and inner.get("product_id"):
        rows.append(inner)
    for key in ("compared", "products", "items"):
        block = obs.get(key)
        if isinstance(block, list):
            for row in block:
                if isinstance(row, dict) and row.get("product_id"):
                    rows.append(row)
    return rows


def _shopper_ping(text: str) -> bool:
    low = (text or "").lower().strip()
    if len(low) < 8:
        return False
    return any(
        cue in low
        for cue in ("?", "which one", "which option", "hold on", "wait,", "still available", "re-check")
    )


def _market_ping(text: str) -> bool:
    low = (text or "").lower()
    return any(
        s in low
        for s in ("out of stock", "sold out", "no longer available", "price increased", "now costs")
    )


def _roster(kind: str) -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    table = _CHUTES if _which_stack() == "chutes" else _OR
    order = _ROSTERS.get(kind) or _ROSTERS["find"]
    return [table[k] for k in order]


def _infer(messages: list[dict[str, Any]], tools: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    last: Exception | None = None
    for model in _roster(kind):
        try:
            out = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if out is None:
                last = RuntimeError(f"inference failed for {model}")
                continue
            return out
        except Exception as exc:  # noqa: BLE001
            last = exc
            continue
    raise RuntimeError(f"inference unavailable: {last}")


def _env(envelope: dict[str, Any]) -> dict[str, Any]:
    base = str(envelope.get("call_id") or "call")
    for attempt in range(3):
        payload = dict(envelope)
        if attempt:
            tag = f"r{attempt}-{int(time.time() * 1000) % 1_000_000}"
            payload["call_id"] = f"{base}-{tag}"
            payload["idempotency_key"] = payload["call_id"]
            packed = []
            for i, call in enumerate(payload.get("calls") or [], start=1):
                c = dict(call)
                c["call_id"] = f"{payload['call_id']}-{i}"
                packed.append(c)
            payload["calls"] = packed
        result = _proxy.post("/environment/call", json_data=payload)
        if result is not None:
            return result
        time.sleep(0.35 * (attempt + 1))
    raise RuntimeError(f"environment call failed ({base})")


def _forced(name: str, args: dict[str, Any], turn: int, tag: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"bno-{tag}-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
    ]


class Cue:
    """Public request facts for this episode only."""

    def __init__(self, query: str, tools: list[dict[str, Any]]) -> None:
        self.query = query
        self.head = _goal_head(query)
        self.cap = _money_cap(self.head)
        self.pairs = _pair_keys(query)
        self.pids = {p for p, _ in self.pairs}
        self.keys = {f"{p}::{s}" for p, s in self.pairs}
        self.kind = _cue_kind(query)
        self.ask = _needs_ask(query)
        self.why = _wants_why(query)
        self.axes = _axis_list(query)
        self.color, self.model = _pref_pair(query)
        self.words = _word_bits(self.head)
        self.allowed = _schema_fields(tools)
        self.tools = _toolset(tools)
        self.panel = _panel_class(self.head)

    def tighten_cap(self, text: str) -> None:
        new = _money_cap(text)
        if new is None:
            return
        if self.cap is None or new < self.cap:
            self.cap = new


class Sight:
    """What this episode has actually observed."""

    def __init__(self) -> None:
        self.n = 0
        self.seen: set[str] = set()
        self.title: dict[str, str] = {}
        self.brand: dict[str, str] = {}
        self.facts: dict[str, dict[str, Any]] = {}
        self.opt: dict[str, str] = {}
        self.row: dict[str, dict[str, Any]] = {}
        self.check: dict[str, dict[str, Any]] = {}
        self.check_n: dict[str, int] = {}
        self.cart: list[tuple[str, str]] = []
        self.cart_n: int | None = None
        self.adds = 0
        self.dead: set[str] = set()
        self.said_ask = False
        self.said_shock = False
        self.uptake = False
        self.shocked = False
        self.rerank = False

    @staticmethod
    def k(pid: Any, sku: Any) -> str:
        return f"{pid}::{sku or ''}"

    def note(self, name: str, args: dict[str, Any], obs: Any) -> None:
        self.n += 1
        outer = obs if isinstance(obs, dict) else {}
        if outer.get("error"):
            return
        payload = _unwrap(outer)
        if name in ("view", "compare"):
            pid = str(args.get("product_id") or "")
            if pid:
                self.seen.add(pid)
            for extra in args.get("product_ids") or []:
                if extra:
                    self.seen.add(str(extra))
            for prod in _products_in(payload):
                self._page(prod)
        elif name in ("search", "filter"):
            for hit in payload.get("results") or []:
                if isinstance(hit, dict) and hit.get("product_id") and hit.get("sku"):
                    self._sku_row(hit)
                if isinstance(hit, dict) and hit.get("product_id") and hit.get("title"):
                    self.title[str(hit["product_id"])] = str(hit["title"])
        elif name == "inspect_stock":
            pid = str(payload.get("product_id") or args.get("product_id") or "")
            sku = str(payload.get("sku") or args.get("sku") or "")
            if not pid:
                return
            key = self.k(pid, sku)
            prev = self.check.get(key) or {}
            flag = _flag_stock(payload.get("in_stock"))
            price = _as_float(payload.get("price") or payload.get("current_price"))
            merged_flag = prev.get("in_stock") if flag is None else flag
            self.check[key] = {
                "in_stock": merged_flag,
                "price": price if price is not None else prev.get("price"),
            }
            self.check_n[key] = self.n
            if flag is False:
                self.dead.add(key)
            if self.shocked:
                self.rerank = True
        elif name == "add_to_cart":
            added = payload.get("added") if isinstance(payload.get("added"), dict) else {}
            pid = str(added.get("product_id") or args.get("product_id") or "")
            sku = str(added.get("sku") or args.get("sku") or "")
            if pid and sku:
                self.adds += 1
                pair = (pid, sku)
                if pair not in self.cart:
                    self.cart.append(pair)
                self.cart_n = self.n
                self.uptake = False
            if payload.get("market_event"):
                self.shocked = True
                self.rerank = True
            flag = _flag_stock(added.get("in_stock"))
            price = _as_float(added.get("price") or added.get("current_price"))
            if pid and sku:
                key = self.k(pid, sku)
                prev = self.check.get(key) or {}
                if flag is not None or price is not None:
                    self.check[key] = {
                        "in_stock": prev.get("in_stock") if flag is None else flag,
                        "price": price if price is not None else prev.get("price"),
                    }
                if flag is False:
                    self.dead.add(key)
                    self.shocked = True
                    self.rerank = True
            self._cart_sync(payload)
        elif name == "remove_from_cart":
            self._cart_sync(payload)
            gone = payload.get("removed") if isinstance(payload.get("removed"), dict) else {}
            pair = (
                str(gone.get("product_id") or args.get("product_id") or ""),
                str(gone.get("sku") or args.get("sku") or ""),
            )
            if pair in self.cart:
                self.cart.remove(pair)
            elif args.get("product_id"):
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                self.cart = [(p, s) for p, s in self.cart if not (p == pid and (not sku or s == sku))]
            self.uptake = False
            self.rerank = True
        elif name == "inspect_cart":
            self._cart_sync(payload)
            self.uptake = True
        elif name == "message":
            self.said_ask = True

    def _page(self, prod: dict[str, Any]) -> None:
        pid = str(prod.get("product_id") or "")
        if not pid:
            return
        self.seen.add(pid)
        if prod.get("title"):
            self.title[pid] = str(prod["title"])
        if prod.get("brand"):
            self.brand[pid] = str(prod["brand"])
        facts = prod.get("facts")
        if isinstance(facts, dict):
            self.facts[pid] = facts
        for v in prod.get("variants") or []:
            if isinstance(v, dict):
                self._sku_row(v, pid)

    def _sku_row(self, row: dict[str, Any], default_pid: str = "") -> None:
        pid = str(row.get("product_id") or default_pid or "")
        sku = str(row.get("sku") or "")
        if not pid or not sku:
            return
        key = self.k(pid, sku)
        flag = _flag_stock(row.get("in_stock"))
        price = _as_float(row.get("price") or row.get("current_price"))
        prev = self.row.get(key) or {}
        self.row[key] = {
            "in_stock": prev.get("in_stock") if flag is None else flag,
            "price": price if price is not None else prev.get("price"),
        }
        if row.get("options"):
            self.opt[key] = str(row["options"])

    def _cart_sync(self, payload: dict[str, Any]) -> None:
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
            key = self.k(r.get("product_id"), r.get("sku"))
            price = _as_float(r.get("current_price") or r.get("price"))
            flag = _flag_stock(
                r.get("current_in_stock") if "current_in_stock" in r else r.get("in_stock")
            )
            prev = self.check.get(key) or {}
            if price is None and flag is None:
                continue
            self.check[key] = {
                "in_stock": prev.get("in_stock", True) if flag is None else flag,
                "price": price if price is not None else prev.get("price"),
            }
            if flag is False:
                self.dead.add(key)
                self.shocked = True

    def snap(self, pid: str, sku: str) -> dict[str, Any]:
        key = self.k(pid, sku)
        out = dict(self.row.get(key) or {})
        if key in self.check:
            out.update(self.check[key])
            out["checked"] = True
        else:
            out["checked"] = False
        return out

    def alive(self, pid: str, sku: str, cap: float | None, *, strict: bool = False) -> bool:
        key = self.k(pid, sku)
        if key in self.dead:
            return False
        snap = self.snap(pid, sku)
        flag = snap.get("in_stock")
        if flag is False:
            return False
        if strict and flag is not True:
            return False
        price = snap.get("price")
        if cap is not None:
            if price is None:
                return not strict
            if price > cap + _SLACK:
                return False
        return True

    def after_cart(self, pid: str, sku: str) -> bool:
        key = self.k(pid, sku)
        if key not in self.check:
            return False
        if self.cart_n is None:
            return True
        return self.check_n.get(key, 0) > self.cart_n

    def cart_sick(self, pid: str, sku: str, cap: float | None) -> bool:
        key = self.k(pid, sku)
        snap = self.snap(pid, sku)
        if key in self.dead or snap.get("in_stock") is False:
            return True
        price = snap.get("price")
        if cap is not None and price is not None and price > cap + _SLACK:
            return True
        return False

    def mix(self, pid: str, sku: str) -> str:
        facts = self.facts.get(str(pid)) or {}
        return " ".join(
            [
                self.title.get(str(pid), ""),
                self.opt.get(self.k(pid, sku), ""),
                " ".join(str(v) for v in facts.values()),
            ]
        )

    def read_num(self, pid: str, sku: str, field: str) -> float | None:
        if field == "price":
            return _as_float(self.snap(pid, sku).get("price"))
        facts = self.facts.get(str(pid)) or {}
        label = {"screen_size": "size", "refresh_rate": "refresh_rate", "resolution": "display"}.get(
            field, field
        )
        # Selected-variant option text beats parent listing facts.
        for src in (
            self.opt.get(self.k(pid, sku), ""),
            facts.get(label),
            facts.get(field),
            self.title.get(str(pid), ""),
        ):
            if src is None or src == "":
                continue
            if field == "refresh_rate":
                n = _hz_num(str(src))
            elif field == "screen_size":
                n = _inch_num(str(src))
            else:
                n = _as_float(src)
            if n is not None:
                return n
        return None

    def fact_text(self, pid: str, field: str) -> str | None:
        facts = self.facts.get(str(pid)) or {}
        label = {"screen_size": "size", "refresh_rate": "refresh_rate", "resolution": "display"}.get(
            field, field
        )
        raw = facts.get(label) or facts.get(field)
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None


def _title_hits(cue: Cue, sight: Sight, pid: str) -> float:
    title = (sight.title.get(str(pid)) or "").lower()
    bag = set(re.findall(r"[a-z0-9]+", title))
    score = 0.0
    for w in cue.words:
        if w in bag:
            score += 2.0
        elif any(t.startswith(w) or w.startswith(t) for t in bag if len(t) > 2):
            score += 1.0
        if any(ch.isdigit() for ch in w) and w in title:
            score += 2.0
    return score


def _attr_hits(cue: Cue, sight: Sight, pid: str, sku: str) -> float:
    blob = sight.mix(pid, sku).lower()
    bag = set(re.findall(r"[a-z0-9]+", blob))
    score = 0.0
    for w in cue.words:
        if w in bag:
            score += 3.0
        elif w in blob:
            score += 1.0
    return score


def _bno_rank(cue: Cue, sight: Sight, pid: str, sku: str) -> tuple:
    """bno8 composite: stated axes (if any), then title hits, attrs, price."""
    axis_row: list[float] = []
    for field, sign in cue.axes:
        val = sight.read_num(pid, sku, field)
        if val is None:
            val = 0.0 if sign < 0 else 1e18
        axis_row.append(sign * val)
    relax = 0
    if cue.model:
        blob = re.sub(r"[^a-z0-9]+", "", sight.mix(pid, sku).lower())
        model = re.sub(r"[^a-z0-9]+", "", cue.model.lower())
        shade = (sight.opt.get(sight.k(pid, sku), "") + " " + sight.title.get(pid, "")).lower()
        same_model = bool(model) and model in blob
        same_color = bool(cue.color) and re.search(
            r"(?<![a-z])" + re.escape(cue.color) + r"(?![a-z])", shade
        )
        if same_model and same_color:
            relax = 0
        elif same_model:
            relax = 1
        else:
            relax = 2 if same_color else 3
    price = sight.read_num(pid, sku, "price")
    if price is None:
        price = 1e18
    return (
        relax,
        *axis_row,
        -_title_hits(cue, sight, pid),
        -_attr_hits(cue, sight, pid, sku),
        price,
        f"{pid}::{sku}",
    )


class ShopRun:
    """EnvPack episode: LLM explores, Sight records, bno8 score picks, finish gates."""

    def __init__(self, problem: dict[str, Any]) -> None:
        env = problem["environment"]
        self.binding = env["binding"]
        self.policy = env["policy_view"]
        self.tools = self.policy["tools"]
        self.query = str(self.policy.get("query") or "")
        self.pid = str(problem.get("problem_id", problem.get("id", "problem")))
        self.max_steps = int(self.policy["max_steps"])
        self.cue = Cue(self.query, self.tools)
        self.sight = Sight()
        self.talk = False
        self.talk_waits = 0
        self.messages: list[dict[str, Any]] = []
        self.dialogue: list[dict[str, Any]] = []

    def go(self) -> list[dict[str, Any]]:
        boot = (
            f"{self.query}\n\n"
            "One tool per turn. Search with content words, view a shortlist, then "
            "inspect_stock → cart → inspect again → place_test_order. Title-token "
            "and sku/attr overlap beat SEO wording. Never order OOS or over cap."
        )
        if self.cue.ask:
            boot += "\n\nFirst call: `message` asking for the withheld spec, then shop with it."
        if self.cue.model:
            boot += (
                f"\n\nKeep model token `{self.cue.model}`"
                + (f" (color {self.cue.color})" if self.cue.color else "")
                + ". On stockout stay on that model and change color. COMMIT, do not ask."
            )
        if self.cue.pairs:
            listed = ", ".join(f"{p}::{s}" for p, s in self.cue.pairs)
            boot += (
                f"\n\nCLOSED SET: {listed}. view those product_ids only. Rank by this "
                "shopper's stated priority; if none, title+attr overlap then lower price."
            )
        if self.cue.cap is not None:
            boot += f"\n\nCAP: {self.cue.cap:.2f}. Do not order above it."
        self.messages = [
            {"role": "system", "content": _VOICE},
            {"role": "user", "content": boot},
        ]
        self.talk = bool(self.cue.ask)
        for turn in range(1, self.max_steps + 1):
            self._nudge()
            calls, text = self._choose(turn)
            done = self._send(turn, calls, text)
            if done:
                break
            self._after()
        return self.dialogue

    def _nudge(self) -> None:
        if self.talk:
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "This turn: call `message` once (COMMIT the next buy if stock "
                        "broke; do not ask permission), then resume."
                    ),
                }
            )
            return
        if self.cue.pairs and len(self.sight.seen & self.cue.pids) < len(self.cue.pids):
            miss = sorted(self.cue.pids - self.sight.seen)
            self.messages.append(
                {
                    "role": "user",
                    "content": f"Still unread closed-set ids: {miss}. view or compare them.",
                }
            )

    def _choose(self, turn: int) -> tuple[list[dict[str, Any]], str]:
        forced = self._pilot(turn)
        if forced is not None:
            name, args = forced
            return _forced(name, args, turn, name[:8]), f"bno8 pilot: {name}"
        for _try in range(4):
            inf = _infer(self.messages, self.tools, self.cue.kind)
            msg = inf["choices"][0]["message"]
            text = msg.get("content") or ""
            calls = (msg.get("tool_calls") or [])[:1]
            if not calls:
                self.messages.extend(
                    [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": "No tool call. Call a supplied tool now."},
                    ]
                )
                continue
            if self.talk and "message" not in _names(calls):
                self.talk_waits += 1
                self.messages.extend(
                    [
                        {"role": "assistant", "content": text, "tool_calls": calls},
                        {"role": "user", "content": "Rejected. Call only `message`."},
                    ]
                )
                continue
            if not self.talk:
                why = self._ban(calls) or self._unready(calls)
                if why:
                    self.messages.extend(
                        [
                            {"role": "assistant", "content": text, "tool_calls": calls},
                            {"role": "user", "content": f"Rejected: {why}"},
                        ]
                    )
                    continue
            return calls, text
        backup = self._pilot(turn, insist=True)
        if backup:
            name, args = backup
            return _forced(name, args, turn, "fb"), f"bno8 fallback: {name}"
        if self.cue.pairs and (self.cue.pids - self.sight.seen):
            miss = sorted(self.cue.pids - self.sight.seen)[0]
            return _forced("view", {"product_id": miss}, turn, "see"), f"view listed {miss}"
        raise RuntimeError("no valid tool call")

    def _ban(self, calls: list[dict[str, Any]]) -> str | None:
        if not self.cue.pairs:
            return None
        names = _names(calls)
        if any(n in ("search", "filter") for n in names) and len(
            self.sight.seen & self.cue.pids
        ) < min(2, len(self.cue.pids)):
            return f"closed set: view listed ids first {sorted(self.cue.pids)}"
        for call in calls:
            name = (call.get("function") or {}).get("name") or ""
            args = _call_args(call)
            if name == "view":
                pid = str(args.get("product_id") or "")
                if pid and pid not in self.cue.pids:
                    return f"view {pid} is outside {sorted(self.cue.pids)}"
            if name == "compare":
                ids = args.get("product_ids") or []
                if isinstance(ids, list):
                    bad = [str(x) for x in ids if str(x) not in self.cue.pids]
                    if bad:
                        return f"compare outside closed set: {bad}"
            if name in ("add_to_cart", "place_test_order"):
                pid = str(args.get("product_id") or "")
                sku = str(args.get("sku") or "")
                if pid and pid not in self.cue.pids:
                    return f"{name} {pid} outside listed ids"
                if pid and sku and f"{pid}::{sku}" not in self.cue.keys:
                    return f"{name} {pid}::{sku} is not a listed variant"
        return None

    def _unready(self, calls: list[dict[str, Any]]) -> str | None:
        if "place_test_order" in _names(calls):
            for call in calls:
                if (call.get("function") or {}).get("name") != "place_test_order":
                    continue
                args = _call_args(call)
                pid = str(args.get("product_id") or "")
                sku = str(args.get("sku") or "")
                if not pid or not sku:
                    return "order needs product_id and sku"
                if (pid, sku) not in self.sight.cart:
                    return "cart that variant before ordering"
                if not self.sight.after_cart(pid, sku):
                    return "inspect_stock that sku after the latest cart add"
                snap = self.sight.snap(pid, sku)
                if snap.get("in_stock") is not True:
                    return "latest inspect is not in_stock=true; recover to another sku"
                price = snap.get("price")
                if self.cue.cap is not None:
                    if price is None:
                        return "inspect current price before ordering"
                    if price > self.cue.cap + _SLACK:
                        return f"price {price:.2f} exceeds cap {self.cue.cap:.2f}"
                if self.cue.why and not self._claims_ready(pid, sku):
                    return "justification needs observed price plus the named spec from view/inspect"
        if "add_to_cart" in _names(calls) and self.sight.adds == 0 and not self.cue.pairs:
            for call in calls:
                if (call.get("function") or {}).get("name") != "add_to_cart":
                    continue
                args = _call_args(call)
                pid = str(args.get("product_id") or "")
                sku = str(args.get("sku") or "")
                if not pid:
                    continue
                if pid in self.sight.title and _title_hits(self.cue, self.sight, pid) <= 0:
                    if self.sight.mix(pid, sku) and _attr_hits(self.cue, self.sight, pid, sku) <= 0:
                        return (
                            "title and attrs share no request tokens; view a better "
                            "shortlist hit before the first cart"
                        )
                snap = self.sight.snap(pid, sku)
                if snap.get("checked") and not self.sight.alive(pid, sku, self.cue.cap, strict=True):
                    return "do not cart an OOS or over-cap sku"
        return None

    def _pick(self, *, need_check: bool) -> tuple[str, str] | None:
        scored: list[tuple[tuple, str, str]] = []
        if self.cue.pairs:
            pool = list(self.cue.pairs)
        else:
            pool = []
            for key in set(self.sight.check) | set(self.sight.row):
                pid, _, sku = key.partition("::")
                if sku:
                    pool.append((pid, sku))
            for pid, sku in self.sight.cart:
                if (pid, sku) not in pool:
                    pool.append((pid, sku))
        for pid, sku in pool:
            if self.sight.k(pid, sku) in self.sight.dead:
                continue
            if pid not in self.sight.seen:
                continue
            if self.cue.panel:
                observed = _panel_class(self.sight.mix(pid, sku))
                if observed and observed != self.cue.panel:
                    continue
            snap = self.sight.snap(pid, sku)
            if need_check and not snap.get("checked"):
                continue
            if not self.sight.alive(pid, sku, self.cue.cap, strict=bool(snap.get("checked"))):
                continue
            scored.append((_bno_rank(self.cue, self.sight, pid, sku), pid, sku))
        if not scored:
            return None
        scored.sort()
        return scored[0][1], scored[0][2]

    def _commit_line(self, nxt: tuple[str, str] | None) -> str:
        if self.cue.model:
            extra = f" from {self.cue.color}" if self.cue.color else ""
            return (
                f"That option is gone. I will keep {self.cue.model}{extra} and take "
                "the next in-stock color of the same model."
            )
        if nxt:
            return (
                "That option is gone. I will take the next surviving in-stock option "
                f"{nxt[0]}::{nxt[1]} that still meets the firm specs."
            )
        return "That option is gone. I will take the next in-stock in-budget option that still fits."

    def _ask_line(self) -> str:
        return "Before I search: which hardware spec is non-negotiable, and what value do you need?"

    def _need_fields(self) -> list[str]:
        if not self.cue.why:
            return []
        low = self.cue.head.lower()
        want = ["in_stock", "price"]
        if any(w in low for w in ("refresh", "hz")):
            want.append("refresh_rate")
        if any(w in low for w in ("screen", "inch")):
            want.append("screen_size")
        if any(
            w in low
            for w in ("resolution", "720p", "1080p", "1440p", "2160p", "4k", "uhd", "qhd", "fhd")
        ) or re.search(r"\bhd\b", low):
            want.append("resolution")
        return [f for f in want if f in self.cue.allowed]

    def _claims(self, pid: str, sku: str) -> list[dict[str, str]]:
        if not self.cue.why:
            return []
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        blob = self.sight.mix(pid, sku)
        named_panel = self.cue.panel
        for field in self._need_fields():
            if field in ("price", "in_stock") or field in seen:
                continue
            raw = self.sight.fact_text(pid, field)
            if field == "resolution":
                observed = _panel_class(str(raw or "")) or _panel_class(blob)
                if named_panel:
                    if observed != named_panel:
                        continue
                    value = named_panel
                else:
                    if not observed:
                        continue
                    value = observed
                out.append({"field": field, "value": value})
                seen.add(field)
                continue
            if not raw:
                raw = blob
            if field == "screen_size" and _inch_num(str(raw or "")) is None:
                raw = blob
            if field == "refresh_rate" and _hz_num(str(raw or "")) is None:
                raw = blob
            value = _clean_value(field, raw)
            if not value:
                continue
            if field == "screen_size" and _inch_num(str(raw)) is None:
                continue
            if field == "refresh_rate" and _hz_num(str(raw)) is None:
                continue
            out.append({"field": field, "value": value})
            seen.add(field)
        snap = self.sight.snap(pid, sku)
        if "price" in self.cue.allowed and snap.get("price") is not None and "price" not in seen:
            out.append({"field": "price", "value": _clean_value("price", snap["price"])})
        if "in_stock" in self.cue.allowed and "in_stock" not in seen:
            out.append(
                {
                    "field": "in_stock",
                    "value": "true" if snap.get("in_stock") is True else "false",
                }
            )
        brand = self.sight.brand.get(str(pid))
        if brand and "brand" in self.cue.allowed and "brand" not in seen:
            out.append({"field": "brand", "value": brand})
        return out

    def _claims_ready(self, pid: str, sku: str) -> bool:
        if not self.cue.why:
            return True
        have = {c["field"] for c in self._claims(pid, sku)}
        return all(f in have for f in self._need_fields())

    def _order_args(self, pid: str, sku: str) -> dict[str, Any]:
        args: dict[str, Any] = {"product_id": pid, "sku": sku}
        claims = self._claims(pid, sku)
        if claims:
            args["justification"] = {"claims": claims}
        return self._scrub_order(args)

    def _scrub_order(self, args: dict[str, Any]) -> dict[str, Any]:
        out = dict(args)
        if not self.cue.why:
            out.pop("justification", None)
            return out
        just = out.get("justification")
        if not isinstance(just, dict):
            return out
        raw = just.get("claims")
        if not isinstance(raw, list):
            out.pop("justification", None)
            return out
        cleaned: list[dict[str, str]] = []
        seen: set[str] = set()
        for claim in raw:
            if not isinstance(claim, dict):
                continue
            field = _clean_field(str(claim.get("field", "")), self.cue.allowed)
            if field is None or field in seen or field == "category":
                continue
            value = _clean_value(field, claim.get("value", ""))
            if not value:
                continue
            cleaned.append({"field": field, "value": value})
            seen.add(field)
        if cleaned:
            out["justification"] = {"claims": cleaned}
        else:
            out.pop("justification", None)
        return out

    def _prep_args(self, call: dict[str, Any]) -> dict[str, Any]:
        parsed = _call_args(call)
        if (call.get("function") or {}).get("name") == "place_test_order":
            pid = str(parsed.get("product_id") or "")
            sku = str(parsed.get("sku") or "")
            if self.cue.why and pid and sku:
                return self._order_args(pid, sku)
            return self._scrub_order(parsed)
        return parsed

    def _pilot(
        self, turn: int, insist: bool = False
    ) -> tuple[str, dict[str, Any]] | None:
        remain = self.max_steps - turn + 1
        late = remain <= 8 or insist
        names = self.cue.tools
        if self.talk:
            if self.sight.cart:
                pid, sku = self.sight.cart[-1]
                if not self.sight.uptake and "inspect_cart" in names:
                    return ("inspect_cart", {})
                if not self.sight.after_cart(pid, sku) and "inspect_stock" in names:
                    return ("inspect_stock", {"product_id": pid, "sku": sku})
            if self.talk_waits < 2 and not insist:
                return None
            if "message" in names:
                nxt = self._pick(need_check=False)
                if self.cue.ask and not self.sight.said_ask:
                    return ("message", {"content": self._ask_line()})
                return ("message", {"content": self._commit_line(nxt)})
            return None
        if self.cue.ask and not self.sight.said_ask and turn <= 2 and "message" in names:
            return ("message", {"content": self._ask_line()})
        if self.cue.pairs:
            miss = sorted(self.cue.pids - self.sight.seen)
            if miss:
                if "compare" in names and len(miss) > 1:
                    return ("compare", {"product_ids": miss[:5]})
                if "view" in names:
                    return ("view", {"product_id": miss[0]})
        if self.sight.cart:
            pid, sku = self.sight.cart[-1]
            if not self.sight.uptake and "inspect_cart" in names:
                return ("inspect_cart", {})
            if not self.sight.after_cart(pid, sku) and "inspect_stock" in names:
                return ("inspect_stock", {"product_id": pid, "sku": sku})
        for pid, sku in list(self.sight.cart):
            if not self.sight.cart_sick(pid, sku, self.cue.cap):
                continue
            nxt = self._pick(need_check=False)
            if not self.sight.said_shock and "message" in names:
                return ("message", {"content": self._commit_line(nxt)})
            if "remove_from_cart" in names:
                return ("remove_from_cart", {"product_id": pid, "sku": sku})
        if self.sight.cart:
            pid, sku = self.sight.cart[-1]
            if self.sight.cart_sick(pid, sku, self.cue.cap):
                return None
            if self.sight.after_cart(pid, sku) and self.sight.alive(
                pid, sku, self.cue.cap, strict=True
            ):
                if self.cue.why and not self._claims_ready(pid, sku):
                    if "view" in names and (
                        pid not in self.sight.facts or pid not in self.sight.seen
                    ):
                        return ("view", {"product_id": pid})
                    # Still order if we have price+stock; missing spec means this
                    # listing is a bad pick only when a named class mismatches.
                    if self.cue.panel:
                        observed = _panel_class(self.sight.mix(pid, sku))
                        if observed and observed != self.cue.panel and "remove_from_cart" in names:
                            return ("remove_from_cart", {"product_id": pid, "sku": sku})
                    if "place_test_order" in names and self._claims(pid, sku):
                        return ("place_test_order", self._order_args(pid, sku))
                    return None
                if "place_test_order" in names:
                    return ("place_test_order", self._order_args(pid, sku))
        if self.cue.pairs and self.sight.rerank and "inspect_stock" in names:
            pending = False
            for pid, sku in self.cue.pairs:
                if self.sight.k(pid, sku) in self.sight.dead:
                    continue
                if not self.sight.after_cart(pid, sku):
                    pending = True
                    return ("inspect_stock", {"product_id": pid, "sku": sku})
            if not pending:
                self.sight.rerank = False
        listed_ready = bool(self.cue.pairs) and not (self.cue.pids - self.sight.seen)
        take = listed_ready or late or self.sight.adds > 0
        if not take and not listed_ready:
            return None
        pick = self._pick(need_check=False)
        if pick is None:
            if listed_ready and "inspect_stock" in names:
                for pid, sku in self.cue.pairs:
                    if self.sight.k(pid, sku) not in self.sight.dead and pid in self.sight.seen:
                        if not self.sight.snap(pid, sku).get("checked"):
                            return ("inspect_stock", {"product_id": pid, "sku": sku})
            return None
        pid, sku = pick
        if not self.sight.snap(pid, sku).get("checked") and "inspect_stock" in names:
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        if self.sight.cart_n is not None and not self.sight.after_cart(pid, sku):
            if "inspect_stock" in names:
                return ("inspect_stock", {"product_id": pid, "sku": sku})
        if (pid, sku) not in self.sight.cart and "add_to_cart" in names:
            if self.sight.alive(pid, sku, self.cue.cap):
                return ("add_to_cart", {"product_id": pid, "sku": sku})
        return None

    def _send(self, turn: int, calls: list[dict[str, Any]], text: str) -> bool:
        if self.talk:
            only = [c for c in calls if (c.get("function") or {}).get("name") == "message"]
            if only:
                calls = only[:1]
        self.messages.append({"role": "assistant", "content": text, "tool_calls": calls})
        group = f"{self.pid}-t{turn}"
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group}-{i}",
                    "action": {"name": c["function"]["name"], "args": self._prep_args(c)},
                }
                for i, c in enumerate(calls, start=1)
            ],
        }
        result = _env(envelope)
        self.dialogue.append(
            {
                "role": "assistant",
                "content": text,
                "tool_calls": calls,
                "environment_result": result,
            }
        )
        for call, row in zip(calls, result.get("calls") or [], strict=False):
            name = str((call.get("function") or {}).get("name") or "")
            args = _call_args(call)
            obs = row.get("observation") if isinstance(row, dict) else {}
            self.sight.note(name, args, obs)
            if name == "message":
                self.sight.said_ask = True
                if any(self.sight.cart_sick(p, s, self.cue.cap) for p, s in self.sight.cart):
                    self.sight.said_shock = True
        if "message" in _names(calls):
            self.talk = False
            self.talk_waits = 0
        for call, row in zip(calls, result.get("calls") or [], strict=False):
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(row.get("observation")),
                }
            )
        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            self.messages.append({"role": "user", "content": content})
            self.cue.tighten_cap(str(content))
            if _shopper_ping(str(content)):
                self.talk = True
                self.talk_waits = 0
            if _market_ping(str(content)):
                self.sight.shocked = True
                self.sight.rerank = True
        return any(
            isinstance(row, dict) and (row.get("observation") or {}).get("done")
            for row in (result.get("calls") or [])
        )

    def _after(self) -> None:
        if self.talk:
            return
        if self.cue.pairs and len(self.sight.seen & self.cue.pids) >= len(self.cue.pids):
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "Listed ids are viewed. Rank by stated priority or by title+attr "
                        "overlap, inspect the winner, cart that listed sku, re-inspect, order."
                    ),
                }
            )
            return
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "Not done. Keep using title-token + attr overlap on viewed pages. "
                    "inspect after cart. If stock died or the price jumped over the cap, "
                    "COMMIT, remove that sku, inspect a different live option, then order. "
                    "Omit justification unless asked why; if asked, claim observed price "
                    "and the named spec from view/inspect, never invent."
                ),
            }
        )


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return ShopRun(problem_data).go()
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "role": "assistant",
                "content": f"run error: {type(exc).__name__}: {str(exc)[:240]}",
            }
        ]


__all__ = ["agent_main"]
