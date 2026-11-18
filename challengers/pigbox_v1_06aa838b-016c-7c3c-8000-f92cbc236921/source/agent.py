
from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# 0727 default ladder starts on Kimi, then Qwen.
_CHUTES = [
    "moonshotai/Kimi-K2.6-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
]
_OPENROUTER = [
    "moonshotai/kimi-k2.6",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-chat-v3.1",
]

_CLAIM_FIELDS = frozenset(
    {"brand", "in_stock", "price", "refresh_rate", "resolution", "screen_size"}
)

_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9][A-Za-z0-9._-]*)")
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|"
    r"max(?:imum)? of|within|budget (?:is|of)|can spend up to|at most)\s*"
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?",
    re.I,
)
_ASK_RE = re.compile(r"ask me|when you ask|until you ask|withheld until", re.I)
_PREFER_RE = re.compile(
    r"(?:strongly\s+)?prefer(?:red)?\s+(?:the\s+)?(?P<color>[A-Za-z]+)\s+"
    r"(?P<model>[A-Za-z0-9][A-Za-z0-9._-]*)",
    re.I,
)
_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(?:inches|inch|in|\"|'')", re.I)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_TYPES = (
    "smartphone",
    "tablet",
    "monitor",
    "headphones",
    "headphone",
    "earbuds",
    "laptop",
)
_EPS = 1e-6

_proxy = ProxyClient(timeout=120, max_retries=2)

_VOICE = """You are an EnvPack shopping agent. Use only supplied tools. Stop when
an observation has done=true. One tool call per turn. Obey this episode's shopper
text and Task rules. Never memorize catalog IDs.

## 0727 retrieve (aspects, then structured fields)
Decompose THIS request into mutually exclusive aspects (product type, required
brand if named, one distinctive hardware/use spec). Search 2–8 words. If THIS
text names a preferred model token, that token is the first search — never a
generic type-only query, and never treat the model token as a product_id.
Leave budget and shipping words out of the query string.

After hits, view several distinct product_ids. Judge a listing in this order:
  R1  matching sku_option / attribute value has the fewest extra tokens
  R2  evidence sits on a dedicated key (size, display, brand), not a blob
  R3  structured fields beat SEO title padding
  R4  a narrow stated value beats a range
  R5  more of the shopper's named aspects covered
  R6  original hit order only after R1–R5 tie
If a firm aspect fails on the viewed sku, reformulate and view again.
Do not cart a near-miss. Soft prefer only after every firm aspect, stock, and cap.

## Closed lists
If this shopper text names two or more product_id::sku keys, that list is the
whole catalog. view/compare those ids. Do not search substitutes. Rank by the
priority THIS text stated; if it stated none, apply R1–R6 on listed variants.

## Dialogue
A real shopper question or budget change: one `message`, then resume.
If a spec is withheld until you ask: first call is `message`.
Empty pings are not new questions.

## Recovery
If THIS text strongly prefers a color+model and says to try another color first:
first cart that pair when a viewed variant matches. After stock dies or price jumps
over the cap: COMMIT same model, other color before a new model.
If THIS text only soft-prefers a brand ("ideally") with firm size/budget/stock:
after shock, keep those firm constraints and cart any live in-cap match — soft
brand is optional. Never ask permission. inspect_stock after every cart add.

## Claims
Only if the shopper asked why/reason. Use the view `facts` dict, the selected
variant options, and the latest inspect. Never title words. Never extra spec
fields the shopper did not name. Omit a field with no structured evidence.

## Finish
view/compare → inspect_stock → add_to_cart → inspect_stock again → place_test_order.
Never order out of stock or over the cap. Never order the shocked sku.
"""


def _provider() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw in ("chutes", "openrouter"):
        return raw
    if getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "chutes"


def _models() -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    return list(_OPENROUTER if _provider() == "openrouter" else _CHUTES)


def _shopper(query: str) -> str:
    return query.split("Task rules:", 1)[0].strip()


def _budget(text: str) -> float | None:
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


def _closed_keys(query: str) -> list[tuple[str, str]]:
    head = _shopper(query)
    found = _KEY_RE.findall(head)
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
    return any(n in _shopper(query).lower() for n in ("why", "reason", "justif"))


def _needs_ask(query: str) -> bool:
    return bool(_ASK_RE.search(_shopper(query)))


def _wants_other_color(query: str) -> bool:
    return bool(
        re.search(r"another color|other color|other colour|same model", query or "", re.I)
    )


def _prefer(query: str) -> dict[str, str]:
    """Firm color+model prefer (recovery-style). Soft 'Ideally prefer Brand' is NOT this."""
    head = _shopper(query)
    # Soft brand preference must not become a color+model lock.
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
    """True only when THIS text asks color-before-model recovery (not soft brand)."""
    pref = _prefer(query)
    return bool(pref.get("model") or pref.get("color")) and (
        _wants_other_color(query) or bool(re.search(r"strongly\s+prefer", query or "", re.I))
    )


def _soft_brand(query: str) -> str | None:
    """Optional brand from 'Ideally, I prefer X' — search hint only, never a lock."""
    m = re.search(
        r"ideally,?\s+i\s+prefer\s+([A-Za-z0-9][A-Za-z0-9 ._-]{0,40}?)(?:,|\.|but|$)",
        _shopper(query),
        re.I,
    )
    if not m:
        return None
    brand = re.sub(r"\s+", " ", m.group(1)).strip(" .,")
    return brand or None


def _asked_fields(query: str) -> list[str]:
    """Spec fields THIS shopper named — not a suite table."""
    head = _shopper(query).lower()
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


def _asked_resolution(query: str) -> str | None:
    head = _shopper(query).lower()
    if re.search(r"\buhd\b|\b4k\b|2160p", head):
        return "uhd"
    if re.search(r"\bqhd\b|1440p|\b2k\b", head):
        return "qhd"
    if re.search(r"\bfhd\b|1080p|full\s*hd", head):
        return "fhd"
    if re.search(r"720p|\bhd\b", head):
        return "hd"
    return None


def _asked_inches(query: str) -> float | None:
    m = _INCH_RE.search(_shopper(query))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _aspect_line(query: str) -> str:
    """0727: type + firm spec first; soft brand last; budget never a search token."""
    head = _BUDGET_RE.sub(" ", _shopper(query))
    bits: list[str] = []
    low_head = head.lower()
    for kind in _TYPES:
        if re.search(rf"\b{re.escape(kind)}\b", low_head):
            bits.append(kind)
            break
    inch = _INCH_RE.search(head)
    if inch:
        bits.append(f"{inch.group(1)} inch")
    asked = _asked_resolution(query)
    if asked == "fhd":
        bits.append("1080p")
    elif asked == "hd":
        bits.append("720p")
    elif asked == "qhd":
        bits.append("1440p")
    elif asked == "uhd":
        bits.append("4k")
    pref = _prefer(query)
    if pref.get("model"):
        bits.append(pref["model"])
    if pref.get("color"):
        bits.append(pref["color"])
    soft = _soft_brand(query)
    if soft and soft.lower() not in {b.lower() for b in bits}:
        bits.append(soft)
    # Distinctive tokens that contain digits (model codes), not bare prices / sizes.
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]{0,24}", head):
        if not any(ch.isdigit() for ch in tok):
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?-?inch(?:es)?", tok, flags=re.I):
            continue
        if tok.lower() in {b.lower() for b in bits}:
            continue
        bits.append(tok)
        if len(bits) >= 8:
            break
    # De-dupe preserve order
    out: list[str] = []
    seen: set[str] = set()
    for b in bits:
        key = b.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return " ".join(out[:8]) if out else "in stock"


def _color_overlap(left: str, right: str) -> bool:
    a = (left or "").strip().lower()
    b = (right or "").strip().lower()
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _option_color(options: str) -> str:
    return (options or "").split("/")[0].strip().lower()


def _sku_row(views: dict[str, dict[str, Any]], pid: str, sku: str) -> dict[str, Any]:
    rec = views.get(pid) or {}
    for row in rec.get("variants") or []:
        if str(row.get("sku") or "") == sku:
            return row
    return {}


def _sku_color(views: dict[str, dict[str, Any]], pid: str, sku: str) -> str:
    return _option_color(str(_sku_row(views, pid, sku).get("options") or ""))


def _sku_model(views: dict[str, dict[str, Any]], pid: str) -> str:
    rec = views.get(pid) or {}
    facts = rec.get("facts") if isinstance(rec.get("facts"), dict) else {}
    return str(facts.get("model") or "").strip()


def _listing_mentions(views: dict[str, dict[str, Any]], pid: str, token: str) -> bool:
    if not token:
        return True
    rec = views.get(pid) or {}
    facts = rec.get("facts") if isinstance(rec.get("facts"), dict) else {}
    blob = " ".join(
        [
            str(facts.get("model") or ""),
            str(rec.get("title") or ""),
        ]
    ).lower()
    return token.lower() in blob


def _inch_token(text: str) -> str | None:
    m = _INCH_RE.search(text or "")
    if not m:
        return None
    return f"{float(m.group(1)):g} inches"


def _res_from_display(text: str) -> str | None:
    """Only tokens that appear in structured display facts — no title inference."""
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


def _listing_inches(
    views: dict[str, dict[str, Any]], pid: str, sku: str
) -> float | None:
    options = str(_sku_row(views, pid, sku).get("options") or "")
    facts = (views.get(pid) or {}).get("facts") or {}
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


def _model_token_ok(token: str) -> bool:
    """Reject garbage facts.model values like 'smart phone' as recovery locks."""
    t = (token or "").strip()
    if not t or " " in t:
        return False
    return any(ch.isdigit() for ch in t) or len(t) >= 5


def _save_listing(
    row: dict[str, Any],
    *,
    views: dict[str, dict[str, Any]],
    seen: set[str],
) -> None:
    pid = str(row.get("product_id") or "")
    if not pid:
        return
    seen.add(pid)
    facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
    variants = [v for v in (row.get("variants") or []) if isinstance(v, dict)]
    path = row.get("category_path") if isinstance(row.get("category_path"), list) else []
    views[pid] = {
        "brand": row.get("brand"),
        "facts": facts,
        "variants": variants,
        "category_path": [str(x) for x in path],
        "title": str(row.get("title") or ""),
    }


def _ground_value(
    field: str,
    *,
    pid: str,
    sku: str,
    views: dict[str, dict[str, Any]],
    inspects: dict[str, dict[str, Any]],
) -> str | None:
    snap = inspects.get(f"{pid}::{sku}") or {}
    rec = views.get(pid) or {}
    facts = rec.get("facts") if isinstance(rec.get("facts"), dict) else {}
    options = str(_sku_row(views, pid, sku).get("options") or "")
    if field == "in_stock":
        flag = snap.get("in_stock")
        if flag is True:
            return "true"
        if flag is False:
            return "false"
        return None
    if field == "price":
        price = snap.get("price")
        if price is None:
            return None
        return str(float(price))
    if field == "brand":
        brand = rec.get("brand")
        if brand:
            return str(brand).strip()
        return None
    if field == "screen_size":
        from_opt = _inch_token(options)
        from_fact = _inch_token(str(facts.get("size") or ""))
        if not from_fact:
            raw = str(facts.get("size") or "").strip()
            if re.fullmatch(r"\d+(?:\.\d+)?", raw):
                from_fact = f"{float(raw):g} inches"
        if from_opt and from_fact:
            opt_n = float(_INCH_RE.search(from_opt).group(1))  # type: ignore[union-attr]
            fact_n = float(_INCH_RE.search(from_fact).group(1))  # type: ignore[union-attr]
            if abs(opt_n - fact_n) > 0.15:
                return from_opt
        return from_opt or from_fact
    if field == "resolution":
        return _res_from_display(str(facts.get("display") or facts.get("resolution") or ""))
    if field == "refresh_rate":
        m = _HZ_RE.search(options) or _HZ_RE.search(str(facts.get("refresh") or facts.get("refresh_rate") or ""))
        if m:
            return f"{m.group(1)}Hz"
        return None
    return None


def _ground_claims(
    pid: str,
    sku: str,
    *,
    query: str,
    allowed: frozenset[str],
    views: dict[str, dict[str, Any]],
    inspects: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    for field in ("in_stock", "price", *_asked_fields(query), "brand"):
        if field not in allowed or field in seen:
            continue
        value = _ground_value(
            field, pid=pid, sku=sku, views=views, inspects=inspects
        )
        if not value:
            continue
        claims.append({"field": field, "value": value})
        seen.add(field)
    return claims


def _allowlist(tools: list[dict[str, Any]]) -> frozenset[str]:
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
    return _CLAIM_FIELDS


def _scrub_order(
    args: dict[str, Any],
    *,
    query: str,
    allowed: frozenset[str],
    views: dict[str, dict[str, Any]],
    inspects: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    out = dict(args)
    if not _wants_why(query):
        out.pop("justification", None)
        return out
    pid = str(out.get("product_id") or "")
    sku = str(out.get("sku") or "")
    claims = _ground_claims(
        pid, sku, query=query, allowed=allowed, views=views, inspects=inspects
    )
    if claims:
        out["justification"] = {"claims": claims}
    else:
        out.pop("justification", None)
    return out


def _args(
    call: dict[str, Any],
    *,
    query: str,
    allowed: frozenset[str],
    views: dict[str, dict[str, Any]] | None = None,
    inspects: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    raw = (call.get("function") or {}).get("arguments", "{}")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    if (call.get("function") or {}).get("name") == "place_test_order":
        return _scrub_order(
            parsed,
            query=query,
            allowed=allowed,
            views=views or {},
            inspects=inspects or {},
        )
    return parsed


def _buyer_interrupt(text: str) -> bool:
    low = (text or "").lower().strip()
    if len(low) < 8:
        return False
    return any(
        c in low
        for c in (
            "?",
            "which one",
            "hold on",
            "wait,",
            "still available",
            "re-check",
            "before you buy",
        )
    )


def _inner(obs: Any) -> dict[str, Any]:
    """EnvPack wraps tool results as {observation:{...}, done, error}."""
    payload = obs if isinstance(obs, dict) else {}
    node = payload.get("observation")
    if isinstance(node, dict) and (
        "done" in payload or "error" in payload or "observation" in payload
    ):
        return node
    return payload


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


def _talk(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    last: Exception | None = None
    for model in _models():
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


def _world(envelope: dict[str, Any]) -> dict[str, Any]:
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
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"environment call failed ({base})")


def _names(calls: list[dict[str, Any]]) -> list[str]:
    return [str((c.get("function") or {}).get("name") or "") for c in calls]


def _outside_list(
    calls: list[dict[str, Any]],
    *,
    pairs: list[tuple[str, str]],
    viewed: set[str],
) -> str | None:
    if not pairs:
        return None
    pids = {p for p, _ in pairs}
    keys = {f"{p}::{s}" for p, s in pairs}
    names = _names(calls)
    if any(n in ("search", "filter") for n in names) and len(viewed & pids) < min(2, len(pids)):
        return f"closed set: view listed product_ids first {sorted(pids)}"
    for call in calls:
        name = (call.get("function") or {}).get("name") or ""
        raw = (call.get("function") or {}).get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        if name == "view":
            pid = str(args.get("product_id") or "")
            if pid and pid not in pids:
                return f"view {pid} is outside listed ids {sorted(pids)}"
        if name == "compare":
            ids = args.get("product_ids") or []
            if isinstance(ids, list):
                bad = [str(x) for x in ids if str(x) not in pids]
                if bad:
                    return f"compare outside closed set: {bad}"
        if name in ("add_to_cart", "place_test_order"):
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and pid not in pids:
                return f"{name} {pid} is outside listed ids"
            if pid and sku and f"{pid}::{sku}" not in keys:
                return f"{name} {pid}::{sku} is not a listed variant"
    return None


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


def _id_reject(
    calls: list[dict[str, Any]],
    *,
    seen: set[str],
    pairs: list[tuple[str, str]],
) -> str | None:
    allowed = {p for p, _ in pairs} | seen
    for call in calls:
        name, args = _call_args(call)
        if name == "view":
            pid = str(args.get("product_id") or "")
            if pid and pid not in allowed:
                return (
                    "view only product_ids from this episode's search/compare "
                    "(a model token is not a product_id)"
                )
        if name == "compare":
            ids = args.get("product_ids") or []
            if isinstance(ids, list):
                bad = [str(x) for x in ids if str(x) not in allowed]
                if bad:
                    return "compare only product_ids observed in this episode"
    return None


def _feature_blocked(query: str, pid: str, views: dict[str, dict[str, Any]]) -> bool:
    blob = (query or "").lower()
    if "smartphone" not in blob:
        return False
    if "feature" not in blob and "basic-phone" not in blob:
        return False
    path = " ".join((views.get(pid) or {}).get("category_path") or []).lower()
    return "feature" in path


def _any_color(views: dict[str, dict[str, Any]], color: str) -> bool:
    for rec in views.values():
        for row in rec.get("variants") or []:
            if _color_overlap(color, _option_color(str(row.get("options") or ""))):
                return True
    return False


def _prefer_reject(
    calls: list[dict[str, Any]],
    *,
    query: str,
    prefer: dict[str, str],
    searches: list[str],
    views: dict[str, dict[str, Any]],
    shocked: dict[str, str] | None,
    color_relaxed: bool,
) -> str | None:
    pref_m = (prefer.get("model") or "").strip()
    pref_c = (prefer.get("color") or "").strip()
    blob = " ".join(searches).lower()
    lock = _color_model_lock(query)
    need_inch = _asked_inches(query)
    need_res = _asked_resolution(query)
    failed_color = (shocked or {}).get("color") or (pref_c if color_relaxed else "")
    for call in calls:
        name, args = _call_args(call)
        if name not in ("add_to_cart", "place_test_order"):
            continue
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        if not pid:
            continue
        if _feature_blocked(query, pid, views):
            return "this listing's category conflicts with the shopper's smartphone constraint"
        color = _sku_color(views, pid, sku)
        if need_inch is not None:
            got_inch = _listing_inches(views, pid, sku)
            if got_inch is not None and abs(got_inch - need_inch) > 0.05:
                return (
                    f"facts/options size {got_inch:g} inch != firm {need_inch:g} inch; "
                    "view a listing whose structured size matches"
                )
        if need_res:
            got_res = _ground_value(
                "resolution", pid=pid, sku=sku, views=views, inspects={}
            )
            if got_res and got_res != need_res:
                return (
                    f"facts display resolves to {got_res}, shopper asked {need_res}; "
                    "pick a listing whose structured display matches"
                )
        if shocked and pid == shocked.get("pid") and sku == shocked.get("sku"):
            return "do not reuse the shocked sku; recover on another live option"
        # After preferred color fails (OOS/price shock), stay on same model + other color.
        if lock and (shocked or color_relaxed):
            need = pref_m if _model_token_ok(pref_m) else ""
            if not need and shocked:
                sm = (shocked.get("model") or "").strip()
                need = sm if _model_token_ok(sm) else ""
            if need and not _listing_mentions(views, pid, need):
                return (
                    f"recover on the same model token ({need}), other color — "
                    "search that token again before changing models"
                )
            if failed_color and color and _color_overlap(failed_color, color):
                return (
                    "preferred/shocked color already failed; "
                    "cart another color of the same model and place_test_order"
                )
            continue
        if lock and pref_m and pref_m.lower() not in blob:
            return f"search the preferred model token `{pref_m}` before carting"
        if lock and pref_c and color and not _color_overlap(pref_c, color):
            if _any_color(views, pref_c):
                return "first cart the preferred color of the preferred model"
            if pref_c.lower() not in blob:
                token = f"{pref_m} {pref_c}".strip() if pref_m else pref_c
                return f"search `{token}` before carting another color"
    return None


def _claim_reject(
    calls: list[dict[str, Any]],
    *,
    query: str,
    allowed: frozenset[str],
    views: dict[str, dict[str, Any]],
    inspects: dict[str, dict[str, Any]],
) -> str | None:
    if not _wants_why(query) or "place_test_order" not in _names(calls):
        return None
    asked = _asked_fields(query)
    need_res = _asked_resolution(query)
    for call in calls:
        name, args = _call_args(call)
        if name != "place_test_order":
            continue
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        claims = _ground_claims(
            pid, sku, query=query, allowed=allowed, views=views, inspects=inspects
        )
        have = {c["field"] for c in claims}
        by_field = {c["field"]: c["value"] for c in claims}
        if need_res and "resolution" in by_field and by_field["resolution"] != need_res:
            return (
                f"do not claim resolution={by_field['resolution']} when shopper asked "
                f"{need_res}; choose a listing whose facts.display matches"
            )
        missing = [f for f in asked if f in allowed and f not in have]
        if missing:
            return (
                f"viewed facts/options for this sku have no {missing[0]}; "
                "view a listing whose structured fields state that spec"
            )
        if "price" not in have or "in_stock" not in have:
            return "inspect_stock this sku so price and stock claims are grounded"
    return None


def _finish_reject(
    calls: list[dict[str, Any]],
    *,
    cart: list[tuple[str, str]],
    inspects: dict[str, dict[str, Any]],
    inspect_seq: dict[str, int],
    cart_seq: int | None,
    cap: float | None,
) -> str | None:
    """firstking-style reject: do not order what this episode's inspect forbids."""
    if "inspect_cart" in _names(calls) and cart_seq is not None:
        fresh = any(inspect_seq.get(f"{p}::{s}", 0) > cart_seq for p, s in cart)
        if not fresh:
            return "inspect_stock the just-carted sku first (not inspect_cart alone)"
    if "place_test_order" not in _names(calls):
        return None
    for call in calls:
        if (call.get("function") or {}).get("name") != "place_test_order":
            continue
        raw = (call.get("function") or {}).get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        if not pid or not sku:
            return "order needs product_id and sku"
        if (pid, sku) not in cart:
            return "cart that variant before ordering"
        key = f"{pid}::{sku}"
        snap = inspects.get(key) or {}
        seq = inspect_seq.get(key, 0)
        if cart_seq is not None and seq <= cart_seq:
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
    return None


def _ingest(
    name: str,
    args: dict[str, Any],
    obs: Any,
    *,
    seq: int,
    viewed: set[str],
    seen: set[str],
    searches: list[str],
    views: dict[str, dict[str, Any]],
    inspects: dict[str, dict[str, Any]],
    inspect_seq: dict[str, int],
    cart: list[tuple[str, str]],
) -> int | None:
    """Update local notes from the INNER EnvPack payload. Returns new cart_seq or None."""
    body = _inner(obs)
    cart_seq = None
    if name == "search":
        q = args.get("query")
        if q:
            searches.append(str(q))
        for row in body.get("results") or []:
            if isinstance(row, dict) and row.get("product_id"):
                seen.add(str(row["product_id"]))
    elif name in ("view", "compare"):
        pid = str(args.get("product_id") or "")
        if pid:
            viewed.add(pid)
            seen.add(pid)
        for extra in args.get("product_ids") or []:
            if extra:
                viewed.add(str(extra))
                seen.add(str(extra))
        if body.get("product_id"):
            _save_listing(body, views=views, seen=seen)
            viewed.add(str(body["product_id"]))
        for row in body.get("compared") or []:
            if isinstance(row, dict) and row.get("product_id"):
                _save_listing(row, views=views, seen=seen)
                viewed.add(str(row["product_id"]))
    elif name == "inspect_stock":
        pid = str(body.get("product_id") or args.get("product_id") or "")
        sku = str(body.get("sku") or args.get("sku") or "")
        if pid:
            key = f"{pid}::{sku}"
            prev = inspects.get(key) or {}
            flag = _stock_flag(body.get("in_stock"))
            price = _as_float(body.get("price") or body.get("current_price"))
            inspects[key] = {
                "in_stock": prev.get("in_stock") if flag is None else flag,
                "price": price if price is not None else prev.get("price"),
            }
            inspect_seq[key] = seq
    elif name == "add_to_cart":
        added = body.get("added") if isinstance(body.get("added"), dict) else {}
        pid = str(added.get("product_id") or args.get("product_id") or "")
        sku = str(added.get("sku") or args.get("sku") or "")
        if pid and sku:
            pair = (pid, sku)
            if pair not in cart:
                cart.append(pair)
            cart_seq = seq
            flag = _stock_flag(added.get("in_stock"))
            price = _as_float(added.get("price") or added.get("current_price"))
            key = f"{pid}::{sku}"
            prev = inspects.get(key) or {}
            if flag is not None or price is not None:
                inspects[key] = {
                    "in_stock": prev.get("in_stock") if flag is None else flag,
                    "price": price if price is not None else prev.get("price"),
                }
        rows = body.get("cart")
        if isinstance(rows, list):
            cart[:] = [
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
        if pair in cart:
            cart.remove(pair)
        rows = body.get("cart")
        if isinstance(rows, list):
            cart[:] = [
                (str(r.get("product_id")), str(r.get("sku") or ""))
                for r in rows
                if isinstance(r, dict) and r.get("product_id")
            ]
    elif name == "inspect_cart":
        rows = body.get("cart")
        if isinstance(rows, list):
            cart[:] = [
                (str(r.get("product_id")), str(r.get("sku") or ""))
                for r in rows
                if isinstance(r, dict) and r.get("product_id")
            ]
            for r in rows:
                if not isinstance(r, dict) or not r.get("product_id"):
                    continue
                key = f"{r.get('product_id')}::{r.get('sku') or ''}"
                price = _as_float(r.get("current_price") or r.get("price"))
                flag = _stock_flag(r.get("current_in_stock") if "current_in_stock" in r else r.get("in_stock"))
                prev = inspects.get(key) or {}
                if price is None and flag is None:
                    continue
                inspects[key] = {
                    "in_stock": prev.get("in_stock") if flag is None else flag,
                    "price": price if price is not None else prev.get("price"),
                }
    return cart_seq


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    tools = policy["tools"]
    query = str(policy.get("query") or "")
    allowed = _allowlist(tools)
    pairs = _closed_keys(query)
    pids = {p for p, _ in pairs}
    cap = _budget(_shopper(query))
    line = _aspect_line(query)
    prefer = _prefer(query)
    asked = _asked_fields(query)

    boot = (
        f"{query}\n\n"
        "Follow Task rules. Search with aspect words "
        f"`{line}` (budget is a cap, not a keyword). "
        "Reply to real shopper questions with message once, then continue. "
        "inspect_stock after cart. Finish with place_test_order on a live in-cap sku."
    )
    if prefer.get("model") and _color_model_lock(query):
        hint = prefer["model"]
        if prefer.get("color"):
            hint = f"{prefer['model']} {prefer['color']}"
        boot += (
            f"\n\nPreferred token from THIS text: search `{hint}` first. "
            "view listings; first cart the preferred color when a variant matches. "
            "A model token is not a product_id."
        )
    soft = _soft_brand(query)
    if soft and not _color_model_lock(query):
        boot += (
            f"\n\nSoft brand prefer `{soft}` only when a fully firm-compliant "
            "listing exists; firm size/budget/stock always win after a market shock."
        )
    if _wants_why(query):
        spec = ", ".join(asked) if asked else "the named spec"
        boot += (
            f"\n\nJustification: only in_stock, price, {spec}, and brand if "
            "view.brand is set. Read view facts / selected variant / inspect. "
            "Never title decoys. Never extra spec fields."
        )
    if _needs_ask(query):
        boot += "\n\nFirst call: `message` asking for the withheld spec, then shop with the answer."
    if pairs:
        listed = ", ".join(f"{pid}::{sku}" for pid, sku in pairs)
        boot += (
            "\n\nCLOSED CANDIDATE SET: "
            f"{listed}. "
            "view each listed product_id. Rank by this shopper's stated priority, "
            "else R1–R6 on structured fields. Do not search substitutes. "
            "First cart and final order must use a listed product_id::sku."
        )
    if cap is not None:
        boot += f"\n\nCAP: {cap:.2f}."

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _VOICE},
        {"role": "user", "content": boot},
    ]
    dialogue: list[dict[str, Any]] = []
    must_message = bool(_needs_ask(query))
    viewed: set[str] = set()
    seen: set[str] = set(pids)
    searches: list[str] = []
    views: dict[str, dict[str, Any]] = {}
    inspects: dict[str, dict[str, Any]] = {}
    inspect_seq: dict[str, int] = {}
    cart: list[tuple[str, str]] = []
    cart_seq: int | None = None
    shocked: dict[str, str] | None = None
    color_relaxed = False
    seq = 0

    for turn in range(1, int(policy["max_steps"]) + 1):
        if must_message:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "REQUIRED this turn: call `message` once to answer the shopper "
                        "(COMMIT the next buy if stock broke; do not ask permission), "
                        "then next turns resume."
                    ),
                }
            )
        elif pairs and len(viewed & pids) < len(pids):
            missing = sorted(pids - viewed)
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
            inference = _talk(messages, tools)
            assistant = inference["choices"][0]["message"]
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:1]
            if not tool_calls:
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {"role": "user", "content": "No tool call. Call a supplied tool now."},
                    ]
                )
                continue
            if must_message and "message" not in _names(tool_calls):
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
                why = _outside_list(tool_calls, pairs=pairs, viewed=viewed)
                if not why:
                    why = _id_reject(tool_calls, seen=seen, pairs=pairs)
                if not why:
                    for _call in tool_calls:
                        _n, _a = _call_args(_call)
                        if _n == "add_to_cart":
                            _pid = str(_a.get("product_id") or "")
                            if _pid and _pid not in views:
                                why = "view this product_id before carting"
                                break
                if not why:
                    why = _prefer_reject(
                        tool_calls,
                        query=query,
                        prefer=prefer,
                        searches=searches,
                        views=views,
                        shocked=shocked,
                        color_relaxed=color_relaxed,
                    )
                if not why:
                    why = _finish_reject(
                        tool_calls,
                        cart=cart,
                        inspects=inspects,
                        inspect_seq=inspect_seq,
                        cart_seq=cart_seq,
                        cap=cap,
                    )
                if not why:
                    why = _claim_reject(
                        tool_calls,
                        query=query,
                        allowed=allowed,
                        views=views,
                        inspects=inspects,
                    )
                if why:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": f"Rejected: {why}"},
                        ]
                    )
                    continue
            break
        else:
            if pairs and len(viewed & pids) < len(pids):
                miss = sorted(pids - viewed)[0]
                tool_calls = [
                    {
                        "id": f"fn-view-{turn}",
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
            only = [c for c in tool_calls if (c.get("function") or {}).get("name") == "message"]
            if only:
                tool_calls = only[:1]

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        group = f"{problem_id}-turn-{turn}"
        envelope = {
            **binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group}-{index}",
                    "action": {
                        "name": call["function"]["name"],
                        "args": _args(
                            call,
                            query=query,
                            allowed=allowed,
                            views=views,
                            inspects=inspects,
                        ),
                    },
                }
                for index, call in enumerate(tool_calls, start=1)
            ],
        }
        result = _world(envelope)
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for call, row in zip(tool_calls, result.get("calls") or [], strict=False):
            seq += 1
            name = str((call.get("function") or {}).get("name") or "")
            parsed = _args(
                call,
                query=query,
                allowed=allowed,
                views=views,
                inspects=inspects,
            )
            obs = row.get("observation") if isinstance(row, dict) else {}
            new_cart_seq = _ingest(
                name,
                parsed,
                obs,
                seq=seq,
                viewed=viewed,
                seen=seen,
                searches=searches,
                views=views,
                inspects=inspects,
                inspect_seq=inspect_seq,
                cart=cart,
            )
            if new_cart_seq is not None:
                cart_seq = new_cart_seq
            if name == "add_to_cart":
                body = _inner(obs)
                added = body.get("added") if isinstance(body.get("added"), dict) else {}
                apid = str(added.get("product_id") or parsed.get("product_id") or "")
                asku = str(added.get("sku") or parsed.get("sku") or "")
                flag = _stock_flag(added.get("in_stock"))
                price = _as_float(added.get("price") or added.get("current_price"))
                if apid and asku and (
                    flag is False
                    or (cap is not None and price is not None and price > cap + _EPS)
                ):
                    shocked = {
                        "pid": apid,
                        "sku": asku,
                        "color": _sku_color(views, apid, asku),
                        "model": _sku_model(views, apid) or prefer.get("model") or "",
                    }
                    if _color_model_lock(query):
                        color_relaxed = True
                elif (
                    apid
                    and asku
                    and flag is not False
                    and (cap is None or price is None or price <= cap + _EPS)
                ):
                    # Live in-cap cart clears the shocked sku lock, but color_relaxed
                    # stays so preferred-color re-demand cannot block place_test_order.
                    shocked = None

        if "message" in _names(tool_calls):
            must_message = False

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
            tighter = _budget(str(content))
            if tighter is not None and (cap is None or tighter < cap):
                cap = tighter
            if _buyer_interrupt(str(content)):
                must_message = True
        if any(
            isinstance(row, dict) and (row.get("observation") or {}).get("done")
            for row in (result.get("calls") or [])
        ):
            break

        if must_message:
            continue
        if pairs and len(viewed & pids) >= len(pids):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "All listed candidates viewed. Rank by stated priority or R1–R6, "
                        "inspect_stock the winner, add_to_cart that listed sku, "
                        "inspect_stock again, place_test_order."
                    ),
                }
            )
        elif shocked:
            if _color_model_lock(query):
                token = prefer.get("model") or ""
                if not _model_token_ok(token):
                    token = shocked.get("model") or ""
                if not _model_token_ok(token):
                    token = "the preferred model"
                shock_msg = (
                    f"Market shock on the last cart. Search `{token}` again, view a "
                    "different color of that model, inspect_stock, cart that live "
                    "in-cap sku, then place_test_order. Do not reuse the shocked sku "
                    "or the failed color. COMMIT with message if asked; do not ask permission."
                )
            else:
                size_bit = ""
                inch = _asked_inches(query)
                if inch is not None:
                    size_bit = f" Keep firm size {inch:g} inch."
                res_bit = ""
                need_res = _asked_resolution(query)
                if need_res:
                    res_bit = f" Keep firm resolution class {need_res}."
                shock_msg = (
                    "Market shock on the last cart. remove_from_cart if needed, then "
                    "view another live in-cap listing that still meets firm constraints."
                    f"{size_bit}{res_bit} Soft brand is optional. Do not reuse the "
                    "shocked sku. inspect_stock, cart, inspect_stock, place_test_order. "
                    "COMMIT with message if asked; do not ask permission."
                )
            messages.append({"role": "user", "content": shock_msg})
        elif color_relaxed and cart:
            live = []
            for pid, sku in cart:
                key = f"{pid}::{sku}"
                snap = inspects.get(key) or {}
                if snap.get("in_stock") is False:
                    continue
                price = snap.get("price")
                if cap is not None and price is not None and price > cap + _EPS:
                    continue
                live.append(f"{pid}::{sku}")
            if live:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Recovery cart is live and in-cap: "
                            f"{', '.join(live)}. "
                            "Call place_test_order on that sku now. "
                            "Do not chase the failed preferred color. "
                            "inspect_stock first only if you have not inspected after the latest cart."
                        ),
                    }
                )
            else:
                token = prefer.get("model") or "the preferred model"
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Preferred color already failed. Search `{token}`, view another "
                            "color of that model, inspect_stock, cart, place_test_order."
                        ),
                    }
                )
        else:
            extra = ""
            if _wants_why(query):
                extra = (
                    " Claims only from view facts / selected variant / inspect; "
                    "omit extra spec fields."
                )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Episode not done. Honor Task rules. Prefer structured facts "
                        "and sku options over title. inspect_stock after cart. "
                        "place_test_order only on a fresh in-stock in-cap sku. "
                        "Omit justification unless asked why."
                        f"{extra}"
                    ),
                }
            )

    return dialogue


__all__ = ["agent_main"]
