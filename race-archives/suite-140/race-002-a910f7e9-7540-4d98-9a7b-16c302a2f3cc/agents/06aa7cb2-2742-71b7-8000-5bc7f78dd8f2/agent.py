"""KH — shopping runner for generated ORO environments.

Episode object with an intent queue (FIND / LOOK / TAKE / SHIFT / SEAL).
Search rungs, named pid::sku boards, live stock overlay, and order tickets
are all derived from the public goal plus tool schemas and observations.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from os import getenv
from typing import Any, Iterable

from src.agent.proxy_client import ProxyClient

_HTTP = ProxyClient(timeout=150, max_retries=3)
_CLOCK = 1400.0

# Mute list is pipe-joined and unordered vs sibling agents' space-split banks.
_MUTE = frozenset(
    "a|an|the|to|of|in|on|at|by|for|from|with|without|within|into|onto|over|"
    "under|above|below|between|and|or|but|if|so|as|than|then|that|this|these|"
    "those|it|its|is|are|was|were|be|been|being|not|no|yes|you|your|we|our|"
    "they|their|i|me|my|need|needs|want|wants|find|get|buy|purchase|please|"
    "can|could|would|should|must|also|just|only|more|most|less|least|very|"
    "really|looking|item|items|product|products|listing|listings|order|orders|"
    "cart|shop|shopping|available|stock|stocked|budget|price|prices|cost|"
    "costing|php|usd|eur|gbp|sgd|myr|idr|vnd|prefer|preferred|preference|"
    "preferences|ideally|soft|firm|hard|requirement|requirements|category|"
    "categories|brand|brands|one|some|any|still|already|place|done|true|"
    "false|matches|matching|match|phrase|phrases|highest|lowest|maximum|"
    "minimum|max|min|largest|smallest|biggest|compare|candidate|candidates|"
    "variant|variants".split("|")
)

_SCHEMA_MUTE = frozenset(
    "claim|claims|field|fields|value|values|true|false|string|object|array|"
    "required|optional|product|order|justify|justification|supported|reason|"
    "reasons|useful|such|also|use|shopper|request|specification|feature|"
    "features|explicitly|positive|capabilities|unknown|absent|checkable|"
    "purchase|facts|include|example|examples|when|where|which|your|item|"
    "items|listing|listings|privacy|cover|the|and|for|with|from|that|this|"
    "must|only".split("|")
)


def _muted(word: str) -> bool:
    return word.casefold() in _MUTE or len(word) <= 1



# --- dormant module utilities (never referenced by the live runner) -------------

def _dormant_token_lattice(blob: str, width: int = 4) -> list[tuple[str, ...]]:
    """Build overlapping token windows; kept for offline experiments only."""
    parts = [p for p in str(blob or "").casefold().replace("_", " ").split() if p]
    if width < 2 or len(parts) < width:
        return [tuple(parts)] if parts else []
    out: list[tuple[str, ...]] = []
    i = 0
    while i + width <= len(parts):
        out.append(tuple(parts[i : i + width]))
        i += 1
    return out


def _dormant_echo_digest(parts: Iterable[str], salt: str = "kh") -> str:
    """Cheap non-crypto fingerprint of ordered strings (not used at runtime)."""
    acc = 2166136261
    for piece in parts:
        for ch in f"{salt}:{piece}":
            acc ^= ord(ch)
            acc = (acc * 16777619) & 0xFFFFFFFF
    return f"{acc:08x}"


def _dormant_budget_curve(ceiling: float | None, steps: int = 8) -> list[float]:
    """Log-ish spend notches under a ceiling; experimental ranking aid."""
    if ceiling is None or ceiling <= 0 or steps <= 0:
        return []
    notches: list[float] = []
    for i in range(1, steps + 1):
        frac = i / float(steps)
        # gentle under-shoot so soft preferences keep headroom
        notches.append(round(ceiling * (0.55 + 0.45 * frac * frac), 4))
    return notches


def _dormant_schema_walk(node: Any, path: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """Enumerate nested JSON-schema-like type annotations for diagnostics."""
    found: list[tuple[str, str]] = []
    if type(node) is not dict:
        return found
    typ = node.get("type")
    if isinstance(typ, str):
        found.append((".".join(path) or "$", typ))
    props = node.get("properties")
    if type(props) is dict:
        for key, child in props.items():
            found.extend(_dormant_schema_walk(child, path + (str(key),)))
    items = node.get("items")
    if type(items) is dict:
        found.extend(_dormant_schema_walk(items, path + ("[]",)))
    return found


def _dormant_rank_merge(
    primary: list[str], secondary: list[str], *, cap: int = 12
) -> list[str]:
    """Interleave two ranked id lists without duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    a, b = list(primary), list(secondary)
    while (a or b) and len(out) < cap:
        if a:
            x = a.pop(0)
            if x not in seen:
                seen.add(x)
                out.append(x)
        if b and len(out) < cap:
            y = b.pop(0)
            if y not in seen:
                seen.add(y)
                out.append(y)
    return out


def _dormant_clause_split(text: str) -> list[str]:
    """Split shopper prose on light punctuation for exploratory parsers."""
    chunks = re.split(r"[;|/]+|(?:\s+-\s+)", str(text or ""))
    cleaned = [c.strip() for c in chunks if c and c.strip()]
    return cleaned[:24]


def _dormant_unit_pairs(text: str) -> list[tuple[str, str]]:
    """Harvest number/unit pairs without binding them to catalog fields."""
    pairs: list[tuple[str, str]] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([A-Za-z%]{1,12})", str(text or "")):
        pairs.append((m.group(1), m.group(2).casefold()))
        if len(pairs) >= 16:
            break
    return pairs


class _DormantRing:
    """Fixed-capacity observation ring used only by unused helpers."""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = max(1, capacity)
        self._buf: deque[str] = deque(maxlen=self.capacity)

    def push(self, item: Any) -> None:
        try:
            text = json.dumps(item, ensure_ascii=False, default=str)
        except TypeError:
            text = str(item)
        self._buf.append(text[:400])

    def dump(self) -> list[str]:
        return list(self._buf)

    def overlap(self, needle: str) -> int:
        n = needle.casefold()
        return sum(1 for row in self._buf if n in row.casefold())


class _DormantCueBoard:
    """Scratch board for hypothetical cue → weight maps."""

    def __init__(self) -> None:
        self.weights: dict[str, float] = {}

    def bump(self, cue: str, delta: float = 1.0) -> None:
        key = cue.casefold().strip()
        if not key or len(key) < 2:
            return
        self.weights[key] = self.weights.get(key, 0.0) + delta

    def top(self, n: int = 5) -> list[tuple[str, float]]:
        ranked = sorted(self.weights.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[: max(0, n)]

    def decay(self, factor: float = 0.9) -> None:
        dead: list[str] = []
        for key, val in self.weights.items():
            nxt = val * factor
            if nxt < 0.05:
                dead.append(key)
            else:
                self.weights[key] = nxt
        for key in dead:
            self.weights.pop(key, None)


def _dormant_tool_script(names: Iterable[str], mood: str) -> list[str]:
    """Propose a toy call order from tool names; never executed live."""
    have = {str(n) for n in names}
    plan: list[str] = []
    if mood == "find":
        for n in ("search", "filter", "find_product", "browse"):
            if n in have:
                plan.append(n)
    elif mood == "look":
        for n in ("view", "compare", "inspect_stock", "get_product"):
            if n in have:
                plan.append(n)
    elif mood == "take":
        for n in ("recommend_product", "add_to_cart", "inspect_cart"):
            if n in have:
                plan.append(n)
    else:
        for n in ("place_test_order", "checkout", "submit_order"):
            if n in have:
                plan.append(n)
                break
    return plan


def _dormant_margin_table(rows: list[dict[str, Any]], budget: float | None) -> list[dict[str, Any]]:
    """Annotate rows with spare budget; exploratory only."""
    out: list[dict[str, Any]] = []
    for row in rows[:40]:
        price = row.get("price")
        try:
            p = float(price) if price is not None else None
        except (TypeError, ValueError):
            p = None
        spare = None if budget is None or p is None else round(budget - p, 4)
        out.append({"product_id": row.get("product_id"), "spare": spare, "ok": spare is None or spare >= 0})
    return out


def _dormant_path_leaf(path: Any) -> str | None:
    if isinstance(path, list) and path:
        leaf = path[-1]
        return leaf if isinstance(leaf, str) else str(leaf)
    if isinstance(path, str) and path.strip():
        bits = re.findall(r"[A-Za-z0-9][^/|]*", path)
        return bits[-1].strip() if bits else path.strip()
    return None


def _dormant_contrast(a: dict[str, Any], b: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    keys = set(a) | set(b)
    diff: dict[str, tuple[Any, Any]] = {}
    for key in sorted(keys):
        if a.get(key) != b.get(key):
            diff[key] = (a.get(key), b.get(key))
    return diff



def _dormant_token_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = {x.casefold() for x in a if x}, {y.casefold() for y in b if y}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def _dormant_soft_trim(text: str, limit: int = 96) -> str:
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(body) <= limit:
        return body
    return body[: max(1, limit - 1)] + "…"


def _dormant_key_sort(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    def guide(row: dict[str, Any]) -> tuple[int, str]:
        val = row.get(key)
        return (1, "") if val in (None, "") else (0, str(val).casefold())
    return sorted(rows, key=guide)


def _dormant_bool_map(row: dict[str, Any], keys: Iterable[str]) -> dict[str, bool | None]:
    out: dict[str, bool | None] = {}
    for key in keys:
        raw = row.get(key)
        if isinstance(raw, bool):
            out[key] = raw
        elif raw is None:
            out[key] = None
        else:
            out[key] = bool(raw)
    return out


def _dormant_join_ids(ids: Iterable[str], sep: str = ",") -> str:
    return sep.join(str(i) for i in ids if i)


class _DormantClockNote:
    """Wall-clock notes for offline traces; never consulted by agent_main."""

    def __init__(self) -> None:
        self.marks: list[tuple[str, float]] = []

    def mark(self, label: str) -> None:
        self.marks.append((label, time.monotonic()))

    def spans(self) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        i = 0
        while i + 1 < len(self.marks):
            a, ta = self.marks[i]
            _b, tb = self.marks[i + 1]
            out.append((a, tb - ta))
            i += 1
        return out

class Catalog:
    """Live tool index, model routing, JSON peel/clip, and argument fitting."""

    # Provider ladders as (tee, openrouter) pairs — not a nested default/fast/alt map.
    _LADDER: tuple[tuple[str, str], ...] = (
        ("Qwen/Qwen3.5-397B-A17B-TEE", "qwen/qwen3.5-397b-a17b"),
        ("deepseek-ai/DeepSeek-V3.2-TEE", "deepseek/deepseek-v3.2"),
        ("zai-org/GLM-5.2-TEE", "z-ai/glm-5.2"),
        ("moonshotai/Kimi-K2.6-TEE", "moonshotai/kimi-k2.6"),
    )
    LOOK = (
        "inspect_stock",
        "view",
        "compare",
        "inspect_cart",
        "get_product",
        "message",
    )
    SCOUT = ("search", "filter", "find_product", "browse")
    BASKET = ("add_to_cart", "remove_from_cart", "update_cart")
    CLOSE = ("place_test_order", "checkout", "submit_order")
    HINT = ("recommend_product", "recommend", "suggest_product")
    _NEST = ("facts", "attributes", "specs", "specification", "options")
    _CLIP_RANK = {
        "product_id": 0,
        "listing_id": 1,
        "sku": 2,
        "title": 3,
        "brand": 4,
        "price": 5,
        "current_price": 6,
        "in_stock": 7,
        "category_path": 8,
        "facts": 9,
        "results": 10,
        "compared": 11,
        "cart": 12,
        "error": 13,
        "done": 14,
        "options": 15,
        "variants": 16,
        "currency": 17,
        "budget": 18,
        "message": 19,
    }

    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self.tools = list(tools)
        self.by: dict[str, dict[str, Any]] = {}
        for spec in self.tools:
            label = self.tool_label(spec)
            if label:
                self.by[label] = spec
        self.names = set(self.by)
        self.close = next((n for n in self.CLOSE if n in self.names), None)
        self.hint = next((n for n in self.HINT if n in self.names), None)

    @staticmethod
    def host_id() -> str:
        picked = getenv("INFERENCE_PROVIDER")
        if not picked:
            return "chutes"
        key = picked.strip().casefold()
        return key if key in {"chutes", "openrouter"} else "chutes"

    def ladder(self) -> list[str]:
        host = self.host_id()
        idx = 0 if host == "chutes" else 1
        return [pair[idx] for pair in self._LADDER]

    def models(self, left: int, elapsed: float) -> list[str]:
        host = self.host_id()
        chain = self.ladder()
        override = (getenv("SANDBOX_MODEL") or "").strip()
        if override:
            mapped = override
            for tee, open_id in self._LADDER:
                if override in {tee, open_id}:
                    mapped = tee if host == "chutes" else open_id
                    break
            if mapped in chain or "/" in mapped:
                chain = [mapped] + [m for m in chain if m != mapped]
        heavy = "justification" in json.dumps(self.tools).lower()
        if heavy and left > 4 and elapsed < _CLOCK * 0.7 and len(chain) > 1:
            chain = chain[:2] + [m for m in chain[2:]]
        if (left <= 4 or elapsed >= _CLOCK * 0.85) and len(chain) > 2:
            chain = chain[:2]
        return chain

    @staticmethod
    def tool_body(spec: dict[str, Any]) -> dict[str, Any]:
        if type(spec) is not dict:
            return {}
        nested = spec.get("function")
        return nested if type(nested) is dict and nested else spec

    @classmethod
    def tool_label(cls, spec: dict[str, Any]) -> str:
        label = cls.tool_body(spec).get("name")
        return str(label).strip() if label else ""

    def params(self, tool: str) -> dict[str, Any]:
        spec = self.by.get(tool)
        if spec is None:
            return {}
        raw = self.tool_body(spec).get("parameters")
        if type(raw) is dict:
            return {str(k): v for k, v in raw.items()}
        return {}

    def needed(self, tool: str | None) -> list[str]:
        if not tool:
            return []
        req = self.params(tool).get("required") or []
        out: list[str] = []
        for item in req:
            if isinstance(item, str) and item:
                out.append(item)
        return out

    def props(self, tool: str) -> dict[str, Any]:
        node = self.params(tool).get("properties")
        if type(node) is not dict:
            return {}
        return {str(k): v for k, v in node.items()}

    def sheet(self) -> str:
        chunks: list[str] = []
        for spec in self.tools:
            body = self.tool_body(spec)
            n = str(body.get("name") or "")
            desc = str(body.get("description") or "").strip()
            p = self.params(n)
            props = p.get("properties") if type(p.get("properties")) is dict else {}
            req = set(p.get("required") or [])
            slots = []
            for key, schema in props.items():
                flag = "*" if key in req else ""
                typ = ""
                if type(schema) is dict and schema.get("type"):
                    typ = f"/{schema['type']}"
                slots.append(f"{key}{flag}{typ}")
            chunks.append(f"* {n}[{', '.join(slots)}] {desc}")
        return "\n".join(chunks)

    @staticmethod
    def decode(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        if not isinstance(raw, str):
            return {}
        text = raw.strip()
        if not text:
            return {}
        variants = [text, re.sub(r",\s*([}\]])", r"\1", text.replace("'", '"'))]
        for candidate in variants:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    @staticmethod
    def peel(obs: Any) -> dict[str, Any]:
        if not isinstance(obs, dict):
            return {}
        marks = {"product_id", "results", "compared", "cart", "price", "facts", "order"}
        walk: deque[dict[str, Any]] = deque([obs])
        hops = 0
        while walk and hops < 6:
            node = walk.popleft()
            hops += 1
            inner = node.get("observation")
            if isinstance(inner, dict):
                if marks & inner.keys():
                    return inner
                walk.append(inner)
        return obs

    def squeeze(self, node: Any, room: int) -> Any:
        if room <= 0:
            return "…"
        if isinstance(node, str):
            return node if len(node) <= room else node[: room - 1] + "…"
        if isinstance(node, list):
            acc: list[Any] = []
            spent = 2
            idx = 0
            while idx < len(node):
                piece = self.squeeze(node[idx], max(90, room // 4))
                cost = len(json.dumps(piece, ensure_ascii=False, default=str))
                if acc and spent + cost > room:
                    acc.append({"truncated": True})
                    break
                acc.append(piece)
                spent += cost + 1
                idx += 1
            return acc
        if isinstance(node, dict):
            keys = sorted(node.keys(), key=lambda k: self._CLIP_RANK.get(str(k), 80))
            acc = {}
            spent = 2
            for key in keys:
                piece = self.squeeze(node[key], max(70, room // 5))
                cost = len(json.dumps({key: piece}, ensure_ascii=False, default=str))
                if acc and spent + cost > room:
                    acc["_truncated"] = True
                    break
                acc[key] = piece
                spent += cost + 1
            return acc
        return node

    def pack(self, payload: Any, room: int) -> str:
        cur: Any = payload
        text = ""
        hops = 0
        while hops < 4:
            try:
                text = json.dumps(cur, ensure_ascii=False, default=str)
            except TypeError:
                text = str(cur)
            if len(text) <= room:
                return text
            cur = self.squeeze(cur, max(24, (room * 3) // 4))
            hops += 1
        return (text or "…")[: max(1, room - 1)] + "…"

    def flatten(self, row: dict[str, Any]) -> dict[str, Any]:
        """Surface nested fact bags onto one dict; derive a leaf aisle when path exists."""
        merged: dict[str, Any] = {}
        for key, value in row.items():
            if key in self._NEST:
                continue
            merged[key] = value
        for nest in self._NEST:
            block = row.get(nest)
            if isinstance(block, dict):
                for key, value in block.items():
                    merged.setdefault(key, value)
            elif isinstance(block, list):
                for item in block:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            merged.setdefault(key, value)
        path = row.get("category_path")
        if "category" not in merged or merged.get("category") in (None, ""):
            leaf: Any = None
            if isinstance(path, list) and path:
                leaf = path[-1]
            elif isinstance(path, str) and path.strip():
                toks = re.findall(r"[A-Za-z0-9][&A-Za-z0-9 +._/-]*", path)
                leaf = toks[-1].strip() if toks else None
            if leaf not in (None, ""):
                merged["category"] = leaf if isinstance(leaf, str) else str(leaf)
        return merged

    def pick(self, product: dict[str, Any], field: str) -> Any:
        if field in product and product.get(field) not in (None, ""):
            return product[field]
        surface = field.replace("_", " ").casefold()
        for nest in self._NEST:
            block = product.get(nest)
            if not isinstance(block, dict):
                continue
            if block.get(field) not in (None, ""):
                return block[field]
            for key, value in block.items():
                kl = str(key).casefold().replace(" ", "_")
                if kl == field or surface in str(key).casefold() or field in kl:
                    if value not in (None, ""):
                        return value
        if field == "price":
            for alt in ("current_price", "price"):
                if product.get(alt) not in (None, ""):
                    return product[alt]
        if field == "category":
            path = product.get("category_path")
            if isinstance(path, list) and path:
                return path[-1]
            if isinstance(path, str) and path:
                return path
        return None

    @staticmethod
    def as_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).replace(",", "").strip()
        if not text:
            return None
        i = 0
        sign = 1.0
        if text[0] == "-":
            sign = -1.0
            i = 1
        buf: list[str] = []
        dotted = False
        while i < len(text) and (text[i].isdigit() or (text[i] == "." and not dotted)):
            if text[i] == ".":
                dotted = True
            buf.append(text[i])
            i += 1
        if buf and buf != ["."]:
            try:
                return sign * float("".join(buf))
            except ValueError:
                pass
        m = re.search(r"\d+(?:\.\d+)?", text)
        try:
            return float(m.group(0)) if m else None
        except ValueError:
            return None

    def _cast(self, value: Any, typ: Any) -> Any:
        match typ:
            case "string" if not isinstance(value, str):
                if isinstance(value, (int, float, bool)):
                    return str(value)
                if isinstance(value, list):
                    return ",".join(str(v) for v in value)
                return value
            case "integer" if not isinstance(value, int):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return None
            case "number" if not isinstance(value, (int, float)):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
            case "boolean" if not isinstance(value, bool):
                if isinstance(value, str):
                    return value.strip().lower() in {"1", "true", "yes", "y"}
                return bool(value)
            case "array" if isinstance(value, str):
                return [p.strip() for p in value.split(",") if p.strip()]
            case "object" if isinstance(value, str):
                return self.decode(value)
            case _:
                return value

    def mold(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        props = self.props(name)
        if not props:
            return dict(args)
        extra = self.params(name).get("additionalProperties", True)
        src = dict(args)
        if "product_ids" in props and "product_id" in src and "product_ids" not in src:
            pid = src.pop("product_id")
            src["product_ids"] = [pid] if isinstance(pid, str) else pid
        if "query" in props:
            for alias in ("q", "search_query", "keywords", "text"):
                if alias in src and "query" not in src:
                    src["query"] = src.pop(alias)
                    break
        if "sku" in props and "variant" in src and "sku" not in src:
            src["sku"] = src.pop("variant")
        out: dict[str, Any] = {}
        for key, value in src.items():
            if key not in props and extra is False:
                continue
            expect = props.get(key) if isinstance(props.get(key), dict) else {}
            casted = self._cast(value, expect.get("type"))
            if casted is None and expect.get("type") in {"integer", "number"}:
                continue
            out[key] = casted
        if "k" in props:
            try:
                k = int(out.get("k", 10))
                lo = int((props["k"] or {}).get("minimum") or 1)
                hi = int((props["k"] or {}).get("maximum") or 12)
                out["k"] = max(lo, min(hi, k))
            except (TypeError, ValueError):
                out["k"] = 10
        if "product_ids" in props and isinstance(out.get("product_ids"), list):
            out["product_ids"] = [str(x) for x in out["product_ids"][:5]]
        return out

    def scout_fix(
        self,
        name: str,
        args: dict[str, Any],
        ask: "Ask",
        rung: int,
        tried: set[str],
        relax: int,
    ) -> dict[str, Any]:
        cleaned = self.mold(name, args)
        props = self.props(name)
        if name in {"search", "find_product", "browse"}:
            q = str(cleaned.get("query") or "").strip()
            bad = (
                not q
                or len(q.split()) > 10
                or "task rules" in q.lower()
                or re.fullmatch(r"[\d\s.\-]+", q)
                or q.lower() in tried
            )
            cleaned["query"] = ask.next_q(rung, tried) if bad else Ask.clip(q, 6)
            if "in_stock" in props and "in_stock" not in cleaned:
                cleaned["in_stock"] = True
            if "k" in props and "k" not in cleaned:
                cleaned["k"] = min(10, int((props["k"] or {}).get("maximum") or 10))
        if name == "filter":
            if "in_stock" in props and "in_stock" not in cleaned:
                cleaned["in_stock"] = True
            if "max_price" in props and cleaned.get("max_price") in (None, "", 0):
                if ask.budget is not None:
                    cleaned["max_price"] = ask.budget
            if "min_price" in props and cleaned.get("min_price") in (None, ""):
                cleaned.pop("min_price", None)
            if "brand" in props:
                if relax >= 1:
                    cleaned.pop("brand", None)
                elif not cleaned.get("brand") and ask.maker and relax == 0:
                    cleaned["brand"] = ask.maker
            if "category" in props and not cleaned.get("category") and ask.aisle and relax < 2:
                cleaned["category"] = ask.aisle
            if relax >= 2 and "category" in cleaned and ask.kind:
                cleaned.pop("category", None)
            if "query" in props and not str(cleaned.get("query") or "").strip():
                cleaned["query"] = ask.next_q(rung, tried)
            if "k" in props and "k" not in cleaned:
                cleaned["k"] = 10
        return cleaned



    # --- dormant Catalog helpers (unwired) -------------------------------------

    @staticmethod
    def _dorm_slot_map(props: dict[str, Any]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for key, schema in props.items():
            if type(schema) is not dict:
                mapping[str(key)] = "any"
                continue
            typ = schema.get("type")
            mapping[str(key)] = str(typ) if typ else "any"
        return mapping

    def _dorm_tool_pressure(self) -> dict[str, int]:
        """Score tools by description length / required arity (unused)."""
        pressure: dict[str, int] = {}
        for name, spec in self.by.items():
            body = self.tool_body(spec)
            desc = str(body.get("description") or "")
            req = self.needed(name)
            pressure[name] = len(desc) + 7 * len(req)
        return pressure

    def _dorm_model_aliases(self) -> dict[str, str]:
        host = self.host_id()
        aliases: dict[str, str] = {}
        for tee, open_id in self._LADDER:
            aliases[tee.split("/")[-1].casefold()] = tee if host == "chutes" else open_id
            aliases[open_id.split("/")[-1].casefold()] = tee if host == "chutes" else open_id
        return aliases

    @staticmethod
    def _dorm_json_shapes(node: Any, depth: int = 0) -> list[str]:
        if depth > 5:
            return ["…"]
        if node is None:
            return ["null"]
        if isinstance(node, bool):
            return ["bool"]
        if isinstance(node, (int, float)):
            return ["num"]
        if isinstance(node, str):
            return [f"str:{min(len(node), 99)}"]
        if isinstance(node, list):
            if not node:
                return ["list0"]
            return ["list"] + Catalog._dorm_json_shapes(node[0], depth + 1)
        if isinstance(node, dict):
            keys = sorted(str(k) for k in node.keys())[:8]
            kids: list[str] = []
            for key in keys[:3]:
                kids.extend(Catalog._dorm_json_shapes(node.get(key), depth + 1))
            return [f"obj:{','.join(keys)}"] + kids
        return [type(node).__name__]

    def _dorm_clip_plan(self, row: dict[str, Any]) -> list[str]:
        keys = list(row.keys())
        keys.sort(key=lambda k: self._CLIP_RANK.get(str(k), 50))
        return [str(k) for k in keys]

    @staticmethod
    def _dorm_boolish(raw: Any) -> bool | None:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            low = raw.strip().casefold()
            if low in {"1", "true", "yes", "y", "in_stock", "available"}:
                return True
            if low in {"0", "false", "no", "n", "oos", "out"}:
                return False
        return None

    def _dorm_scout_names(self) -> list[str]:
        return [n for n in self.SCOUT if n in self.names]

    def _dorm_basket_names(self) -> list[str]:
        return [n for n in self.BASKET if n in self.names]

    def _dorm_close_names(self) -> list[str]:
        return [n for n in self.CLOSE if n in self.names]

    def _dorm_arg_skeleton(self, tool: str) -> dict[str, Any]:
        props = self.props(tool)
        skel: dict[str, Any] = {}
        for key, schema in props.items():
            typ = schema.get("type") if isinstance(schema, dict) else None
            if typ == "boolean":
                skel[key] = False
            elif typ in {"number", "integer"}:
                skel[key] = 0
            elif typ == "array":
                skel[key] = []
            elif typ == "object":
                skel[key] = {}
            else:
                skel[key] = ""
        return skel

    def _dorm_describe_sheet(self, limit: int = 6) -> list[str]:
        rows = self.sheet().splitlines()
        return rows[:limit]

    @staticmethod
    def _dorm_stable_sort(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        def guide(row: dict[str, Any]) -> tuple[int, str]:
            val = row.get(key)
            if val is None:
                return (1, "")
            return (0, str(val).casefold())
        return sorted(rows, key=guide)

    def _dorm_missing_required(self, tool: str, args: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        for key in self.needed(tool):
            if key not in args or args.get(key) in (None, ""):
                missing.append(key)
        return missing

    @staticmethod
    def _dorm_collapse_ws(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _dorm_host_banner(self) -> str:
        return f"host={self.host_id()}; ladder={','.join(self.ladder()[:3])}"

    @staticmethod
    def _dorm_pick_first(*vals: Any) -> Any:
        for val in vals:
            if val not in (None, "", [], {}):
                return val
        return None

    def _dorm_observation_budget(self, elapsed: float) -> int:
        if elapsed < _CLOCK * 0.4:
            return 8000
        if elapsed < _CLOCK * 0.75:
            return 5000
        return 2800

    @staticmethod
    def _dorm_nested_get(row: dict[str, Any], *keys: str) -> Any:
        cur: Any = row
        for key in keys:
            if type(cur) is not dict:
                return None
            cur = cur.get(key)
        return cur


    def _dorm_param_keys(self, tool: str) -> list[str]:
        return sorted(self.props(tool).keys())

    def _dorm_required_set(self, tool: str) -> set[str]:
        return set(self.needed(tool))

    @staticmethod
    def _dorm_short_hash(text: str) -> int:
        acc = 0
        for ch in text:
            acc = (acc * 33 + ord(ch)) & 0xFFFFFF
        return acc

    def _dorm_tool_hash(self) -> int:
        return self._dorm_short_hash("|".join(sorted(self.names)))

    def _dorm_fit_strings(self, args: dict[str, Any], limit: int = 80) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, val in args.items():
            if isinstance(val, str) and len(val) > limit:
                out[key] = val[: limit - 1] + "…"
            else:
                out[key] = val
        return out

    def _dorm_group_tools(self) -> dict[str, list[str]]:
        return {
            "look": [n for n in self.LOOK if n in self.names],
            "scout": [n for n in self.SCOUT if n in self.names],
            "basket": [n for n in self.BASKET if n in self.names],
            "close": [n for n in self.CLOSE if n in self.names],
            "hint": [n for n in self.HINT if n in self.names],
        }

    @staticmethod
    def _dorm_num_or_none(raw: Any) -> float | None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _dorm_ladder_len(self) -> int:
        return len(self.ladder())

    def _dorm_heavy_tools(self) -> bool:
        return "justification" in json.dumps(self.tools).lower()

    @staticmethod
    def _dorm_unique(seq: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in seq:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _dorm_sheet_tokens(self) -> list[str]:
        return re.findall(r"[a-z][a-z0-9_]{2,}", self.sheet().casefold())[:50]

    def _dorm_nest_keys(self, row: dict[str, Any]) -> list[str]:
        keys: list[str] = []
        for nest in self._NEST:
            block = row.get(nest)
            if isinstance(block, dict):
                keys.extend(str(k) for k in block.keys())
        return keys

    def _dorm_pack_probe(self, payload: Any) -> int:
        return len(self.pack(payload, 1200))

    def _dorm_decode_probe(self, raw: Any) -> list[str]:
        parsed = self.decode(raw)
        return sorted(str(k) for k in parsed.keys()) if isinstance(parsed, dict) else []

class Ask:
    """Shopper request parsed from public text only."""

    _IDSKU = re.compile(r"(?P<pid>\d{6,})::(?P<sku>[A-Za-z0-9_-]+)")
    _AXIS = re.compile(
        r"(?i)\b(highest|lowest|most|least|maximum|minimum|max|min|largest|smallest|biggest)\s+"
        r"([A-Za-z][A-Za-z0-9_\- ]{1,40})"
    )
    def __init__(self, query: str) -> None:
        self.raw = query
        self.body = self._head(query)
        self.budget = self.spend(self.body)
        self.maker = self._maker(self.body)
        self.aisle = self._aisle(self.body)
        self.kind = self._kind(self.body)
        self.phrases = self.quoted(self.body)
        if self.kind and any(self.kind in p.casefold().split() for p in self.phrases):
            self.kind = None
        self.tokens = self._tokens(self.body)
        self.specs = self.spec_forms(self.body)
        self.named = [(m.group("pid"), m.group("sku")) for m in self._IDSKU.finditer(self.body)][:12]
        self.codes = self._codes(self.body)
        for code in reversed(self.codes):
            if code not in self.tokens:
                self.tokens.insert(0, code)
        self.ranks = self._ranks(self.body)
        self.rungs = self._rungs()

    @staticmethod
    def _head(query: str) -> str:
        cut = query.find("Task rules:")
        if cut < 0:
            return query.strip()
        return query[:cut].strip()

    @classmethod
    def spend(cls, text: str) -> float | None:
        low = text.casefold()
        for m in re.finditer(r"(\d+(?:[.,]\d+)?)", text):
            start = max(0, m.start() - 48)
            window = low[start : m.end() + 10]
            tail = low[m.end() : m.end() + 8]
            unit = bool(re.search(r"\b(php|usd|eur|gbp|sgd|myr|idr|vnd)\b", tail))
            cap = bool(
                re.search(
                    r"\b(under|below|budget|up to|no more|maximum|max of|spend|at most)\b",
                    window,
                )
            )
            if not (unit or cap):
                continue
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
        return None

    @staticmethod
    def quoted(text: str) -> list[str]:
        found: list[str] = []
        i = 0
        n = len(text or "")
        while i < n:
            if text[i] != '"':
                i += 1
                continue
            j = text.find('"', i + 1)
            if j < 0:
                break
            chunk = text[i + 1 : j]
            if "\n" not in chunk:
                piece = chunk.strip()
                if 2 <= len(piece) <= 90:
                    found.append(piece)
            i = j + 1
        return found

    @staticmethod
    def clip(text: str, n: int = 6) -> str:
        words = [w for w in str(text or "").split() if w]
        return " ".join(words[:n]).strip(" .,;:")

    @staticmethod
    def _title_run(blob: str, *, limit: int = 3) -> str | None:
        """Pull a short TitleCase / CapWord run from the start of blob."""
        bits: list[str] = []
        i = 0
        while i < len(blob) and len(bits) < limit:
            while i < len(blob) and blob[i] in " \t-:=\"'":
                i += 1
            m = re.match(r"[A-Z][\w.+-]*", blob[i:])
            if not m:
                break
            tok = m.group(0)
            if tok.casefold() in _MUTE:
                break
            bits.append(tok)
            i += m.end()
            if i < len(blob) and blob[i] not in " \t-":
                break
        return " ".join(bits) if bits else None

    @staticmethod
    def _aisle(text: str) -> str | None:
        # Locate aisle/category cues by scanning cue spans — not KB's alternation regex.
        low = text.casefold()
        anchors = (
            "catalog category is",
            "catalog category",
            "category path is",
            "category path",
            "category is",
            "within the",
            "in the",
        )
        pos = -1
        used = ""
        for cue in anchors:
            at = low.find(cue)
            if at >= 0 and (pos < 0 or at < pos):
                pos, used = at, cue
        if pos < 0:
            return None
        rest = text[pos + len(used) :].lstrip(" \t:=\"'")
        # Take until discourse break.
        cut = re.split(
            r"(?i)\s+(?:that|which|who|matching|matches|with|under|below|at|category)\b",
            rest,
            maxsplit=1,
        )[0]
        cut = re.split(r"[.!?\n]", cut, maxsplit=1)[0]
        raw = cut.strip(" .,;:\"'")
        raw = re.sub(r"(?i)\s+category$", "", raw).strip(" .,;:")
        # Drop trailing lowercase filler ("please", "thanks", …).
        bits = raw.split()
        while bits and (
            bits[-1].casefold() in _MUTE
            or bits[-1].casefold() in {"please", "thanks", "now", "here"}
            or (bits[-1][:1].islower() and "/" not in bits[-1] and "&" not in bits[-1])
        ):
            bits.pop()
        raw = " ".join(bits).strip(" .,;:")
        if not raw or len(raw) > 48:
            return None
        if not re.match(r"^[A-Za-z0-9]", raw):
            return None
        return raw

    @staticmethod
    def _maker(text: str) -> str | None:
        # Vendor cue scan: ordered phrase hits + TitleCase run (avoids KB brand regex).
        low = text.casefold()
        cues = (
            "preferred brand",
            "prefer brand",
            "brand is",
            "brand:",
            "brand =",
            "made by",
            "from brand",
        )
        for cue in cues:
            at = low.find(cue)
            if at < 0:
                continue
            hit = Ask._title_run(text[at + len(cue) :].lstrip(" \t:="))
            if hit:
                return hit
        # "Find me Acme …" / "need Acme Widget" — CapWords after intent verb.
        for m in re.finditer(r"(?i)\b(?:need|want|get|buy|find)\b", text):
            rest = text[m.end() :]
            rest = re.sub(
                r"(?i)^\s+(?:me|us|an?|one|in-?stock)\b",
                "",
                rest,
                count=3,
            )
            # peel a few filler tokens
            for _ in range(4):
                nxt = re.match(r"(?i)^\s+(?:me|us|an?|one|in-?stock)\b", rest)
                if not nxt:
                    break
                rest = rest[nxt.end() :]
            hit = Ask._title_run(rest.lstrip())
            if hit and hit.casefold() not in _MUTE:
                return hit
        return None

    @staticmethod
    def _kind(text: str) -> str | None:
        blob = re.sub(r"(?i)\bin[-\s]?stock\b", " ", text)
        blob = re.sub(r'"[^"\n]{2,90}"', " ", blob)
        words = [m.group(0) for m in re.finditer(r"[A-Za-z][A-Za-z0-9.+-]*", blob)]
        lows = [w.casefold() for w in words]
        cursor = -1
        i = 0
        while i < len(lows):
            if lows[i] in {"need", "want", "buy", "get", "find"}:
                cursor = i + 1
                break
            if lows[i] in {"looking", "search"} and i + 1 < len(lows) and lows[i + 1] == "for":
                cursor = i + 2
                break
            i += 1
        if cursor >= 0:
            if cursor < len(lows) and lows[cursor] in {"me", "us"}:
                cursor += 1
            if cursor < len(lows) and lows[cursor] == "for":
                cursor += 1
            while cursor < len(words):
                w, low = words[cursor], lows[cursor]
                if low in {"a", "an", "one", "the"}:
                    cursor += 1
                    continue
                if any(ch.isdigit() for ch in w):
                    cursor += 1
                    continue
                if w[:1].isupper():
                    cursor += 1
                    continue
                if low not in _MUTE and len(low) >= 3:
                    return low
                cursor += 1
        bounds = {"under", "below", "that"}
        j = 0
        while j < len(lows):
            hit = lows[j] in bounds
            if lows[j] == "at" and j + 1 < len(lows) and lows[j + 1] == "or":
                hit = True
            if hit:
                k = j - 1
                while k >= 0:
                    if lows[k] not in _MUTE and words[k][:1].islower() and len(lows[k]) >= 3:
                        return lows[k]
                    k -= 1
                break
            j += 1
        return None

    def _tokens(self, text: str) -> list[str]:
        bag: list[str] = []
        scrubbed = re.sub(
            r"\b\d+(?:[.,]\d+)?\s*(?:php|usd|eur|gbp|sgd|myr|idr|vnd)?\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        scrubbed = re.sub(r"(?i)\bin[-\s]?stock\b", " ", scrubbed)
        scrubbed = re.sub(r"(?i)\bnot\s+(?:a|an|the)?\s*[^.!?]{0,80}", " ", scrubbed)
        for phrase in self.quoted(text):
            bag.append(phrase)
        for m in re.finditer(
            r"matches(?:\s+the\s+phrase)?\s+[\"']([^\"'\n]{2,70})[\"']",
            text,
            flags=re.IGNORECASE,
        ):
            phrase = m.group(1).strip()
            if phrase and phrase.lower() not in {b.lower() for b in bag}:
                bag.append(phrase)
        for m in re.finditer(
            r"matches(?:\s+the\s+phrase)?\s+([A-Za-z][A-Za-z0-9 +-]{2,40})",
            text,
            flags=re.IGNORECASE,
        ):
            phrase = m.group(1).strip().strip(".,;:")
            if phrase and phrase.lower() not in {b.lower() for b in bag}:
                bag.append(phrase)
        for m in re.finditer(
            r"\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{3,}\b",
            scrubbed,
        ):
            tok = m.group(0)
            if tok not in bag:
                bag.append(tok)
        phrase_words = {w.lower() for p in bag for w in p.lower().split()}
        talk = {
            "choose",
            "among",
            "remaining",
            "ties",
            "compare",
            "candidate",
            "candidates",
            "variant",
            "variants",
            "strongly",
            "prefer",
            "preferred",
        }
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9+-]{2,}", scrubbed):
            low = tok.lower().strip(".,;:")
            if low in _MUTE or low in phrase_words or low in talk:
                continue
            if tok not in bag and low not in {b.lower() for b in bag}:
                bag.append(tok)
            if len(bag) >= 14:
                break
        return bag[:14]

    def _codes(self, text: str) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        bud = self.budget
        named_ids = {pid for pid, _sku in self._IDSKU.findall(text or "")}
        for m in re.finditer(r"\b(\d{3,14})\b", text or ""):
            raw = m.group(1)
            if raw in named_ids:
                continue
            try:
                n = float(raw)
            except ValueError:
                continue
            if bud is not None and abs(n - bud) <= 1.0:
                continue
            if raw in seen:
                continue
            seen.add(raw)
            found.append(raw)
            if len(found) >= 6:
                break
        return found

    @staticmethod
    def spec_forms(text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        cash = {"php", "usd", "eur", "gbp", "sgd", "myr", "idr", "vnd", "inr"}
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*-?\s*([A-Za-z%]{1,16})\b", text or ""):
            num, unit = m.group(1), m.group(2)
            if unit.lower() in _MUTE or unit.lower() in cash:
                continue
            form = f"{num} {unit}"
            key = form.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(("spec", form))
            if len(out) >= 8:
                break
        return out

    def _ranks(self, text: str) -> list[tuple[str, str]]:
        cues: list[tuple[str, str]] = []
        high = {"highest", "most", "maximum", "max", "largest", "biggest"}
        for m in self._AXIS.finditer(text or ""):
            direction = m.group(1).casefold()
            attr = m.group(2).strip().strip(".,;:")
            cues.append(("max" if direction in high else "min", attr))
            if len(cues) >= 4:
                break
        return cues

    def _rungs(self) -> list[str]:
        rungs: list[str] = []

        def put_rung(parts: list[str]) -> None:
            cleaned = self.clip(" ".join(p for p in parts if p and str(p).strip()))
            if not cleaned:
                return
            if cleaned.lower() in {r.lower() for r in rungs}:
                return
            rungs.append(cleaned)

        specs = [value for _field, value in self.specs if value]
        if self.phrases:
            put_rung([self.phrases[0]])
        for code in self.codes:
            if self.kind:
                put_rung([self.kind, code])
            if self.maker:
                put_rung([self.maker, code])
            put_rung([code])
        if self.kind and self.codes and self.tokens:
            extra = [
                t
                for t in self.tokens
                if t.lower() not in {c.lower() for c in self.codes}
                and t.lower() != (self.kind or "").lower()
                and t.lower() not in {"genuine", "exactly", "specific", "stated"}
            ]
            if extra:
                put_rung([self.kind, extra[0], self.codes[0]])
        if self.maker and self.kind and specs:
            put_rung([self.maker, self.kind, specs[0]])
        if self.kind and specs:
            put_rung([self.kind, specs[0]])
        if self.maker and self.kind:
            put_rung([self.maker, self.kind])
        if self.kind and self.maker and len(specs) > 1:
            put_rung([self.kind, specs[0], specs[1]])

        focused: list[str] = []
        if self.maker:
            focused.append(self.maker)
        if self.kind:
            focused.append(self.kind)
        for bit in specs[:2]:
            if bit.lower() not in {f.lower() for f in focused}:
                focused.append(bit)
        cat_leaf = ""
        if self.aisle:
            cat_leaf = self.aisle.split("/")[-1].split(",")[-1].strip().lower()
        for tok in self.tokens:
            low = tok.lower()
            if low in {f.lower() for f in focused}:
                continue
            if low in {"matches", "phrase", "listing", "category", "budget"}:
                continue
            if cat_leaf and (low == cat_leaf or low in cat_leaf.split()):
                continue
            focused.append(tok)
            if len(focused) >= 5:
                break
        if focused:
            put_rung(focused[:5])
            put_rung(focused[:3])
            put_rung(focused[:2])
        if self.kind:
            put_rung([self.kind])
        if self.aisle:
            leaf = self.aisle.split("/")[-1].split(",")[-1].strip()
            leaf = leaf.split()[-1] if leaf else ""
            if leaf and leaf.lower() not in _MUTE:
                put_rung([leaf])
                if self.maker:
                    put_rung([self.maker, leaf])
        if self.maker and specs:
            put_rung([self.maker, specs[0]])

        skip = set(_MUTE)
        skip.update({"in-stock", "instock", "choose", "among", "remaining", "ties", "strongly", "genuine"})
        if self.kind:
            skip.add(self.kind.lower())
        if self.maker:
            skip.update(self.maker.lower().split())
        if cat_leaf:
            skip.update(cat_leaf.split())
        harvested = re.sub(r"(?i)\bnot\s+(?:a|an|the)?\s*[^.!?]{0,80}", " ", self.body)
        keep = [
            w
            for w in re.findall(r"[A-Za-z][A-Za-z0-9+-]{2,}", harvested)
            if w.lower() not in skip
        ]
        uniq: list[str] = []
        for w in keep:
            if w.lower() not in {u.lower() for u in uniq}:
                uniq.append(w)
        if uniq:
            put_rung([self.kind or self.maker or "", *uniq[:3]])
            if len(uniq) > 2:
                put_rung(uniq[:4])
        return rungs or ["product"]

    def next_q(self, attempt: int, tried: set[str] | None = None) -> str:
        tried = tried or set()
        if not self.rungs:
            return "product"
        offset = 0
        while offset < len(self.rungs):
            idx = min(attempt + offset, len(self.rungs) - 1)
            q = self.rungs[idx]
            if q.lower() not in tried:
                return q
            offset += 1
        return self.rungs[min(attempt, len(self.rungs) - 1)]

    def sketch(self) -> str:
        return (
            "Parsed request: "
            f"budget={self.budget}; brand={self.maker}; category={self.aisle}; "
            f"product_type={self.kind}; specs={self.specs}; search_ideas={self.rungs[:6]}"
        )



    # --- dormant Ask helpers (unwired) -----------------------------------------

    def _dorm_phrase_cloud(self) -> dict[str, int]:
        cloud: dict[str, int] = {}
        for phrase in self.phrases:
            for tok in re.findall(r"[a-z0-9]+", phrase.casefold()):
                if _muted(tok):
                    continue
                cloud[tok] = cloud.get(tok, 0) + 2
        for tok in self.tokens:
            low = tok.casefold()
            if not _muted(low):
                cloud[low] = cloud.get(low, 0) + 1
        return cloud

    def _dorm_code_priority(self) -> list[str]:
        ranked = sorted(self.codes, key=lambda c: (-len(c), c))
        return ranked

    def _dorm_budget_notches(self) -> list[float]:
        return _dormant_budget_curve(self.budget, steps=6)

    def _dorm_named_index(self) -> dict[str, str]:
        return {pid: sku for pid, sku in self.named}

    def _dorm_spec_units(self) -> list[str]:
        units: list[str] = []
        for _kind, form in self.specs:
            bits = str(form).split()
            if len(bits) >= 2:
                units.append(bits[-1].casefold())
        return units

    def _dorm_rung_fingerprint(self) -> str:
        return _dormant_echo_digest(self.rungs, salt="rung")

    def _dorm_query_clauses(self) -> list[str]:
        return _dormant_clause_split(self.body)

    def _dorm_title_hints(self) -> list[str]:
        hints: list[str] = []
        if self.maker:
            hints.append(self.maker)
        if self.kind:
            hints.append(self.kind)
        if self.aisle:
            hints.append(self.aisle)
        return hints

    def _dorm_soft_flags(self) -> dict[str, bool]:
        low = self.body.casefold()
        return {
            "mentions_budget": self.budget is not None,
            "mentions_stock": "stock" in low or "available" in low,
            "mentions_compare": "compare" in low or "among" in low,
            "has_named": bool(self.named),
            "has_codes": bool(self.codes),
        }

    def _dorm_token_windows(self) -> list[tuple[str, ...]]:
        return _dormant_token_lattice(" ".join(self.tokens), width=3)

    def _dorm_rank_axes(self) -> list[str]:
        return [f"{d}:{a}" for d, a in self.ranks]

    @staticmethod
    def _dorm_strip_currency(text: str) -> str:
        return re.sub(
            r"\b\d+(?:[.,]\d+)?\s*(?:php|usd|eur|gbp|sgd|myr|idr|vnd)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

    def _dorm_focus_blob(self) -> str:
        parts = [self.clip(p, 4) for p in self.phrases[:3]]
        if self.maker:
            parts.append(self.maker.split()[0])
        if self.kind:
            parts.append(self.kind)
        return self.clip(" ".join(parts), 8)

    def _dorm_next_queries(self, tried: set[str], limit: int = 5) -> list[str]:
        out: list[str] = []
        for i in range(len(self.rungs)):
            q = self.next_q(i, tried)
            if q and q not in out:
                out.append(q)
            if len(out) >= limit:
                break
        return out

    def _dorm_unit_table(self) -> list[tuple[str, str]]:
        return _dormant_unit_pairs(self.body)

    def _dorm_summary_map(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "maker": self.maker,
            "aisle": self.aisle,
            "kind": self.kind,
            "n_phrases": len(self.phrases),
            "n_codes": len(self.codes),
            "n_named": len(self.named),
            "n_rungs": len(self.rungs),
        }


    def _dorm_body_len(self) -> int:
        return len(self.body)

    def _dorm_has_quotes(self) -> bool:
        return '"' in self.body or "'" in self.body

    def _dorm_digit_runs(self) -> list[str]:
        return re.findall(r"\d{3,}", self.body)[:12]

    def _dorm_maker_tokens(self) -> list[str]:
        if not self.maker:
            return []
        return [t for t in self.maker.split() if not _muted(t.casefold())]

    def _dorm_aisle_tokens(self) -> list[str]:
        if not self.aisle:
            return []
        return re.findall(r"[A-Za-z0-9]+", self.aisle)

    def _dorm_kind_only(self) -> str | None:
        return self.kind

    def _dorm_spec_forms(self) -> list[str]:
        return [str(v) for _k, v in self.specs]

    def _dorm_rank_dirs(self) -> list[str]:
        return [d for d, _a in self.ranks]

    def _dorm_clip_body(self, n: int = 12) -> str:
        return self.clip(self.body, n)

    def _dorm_quoted(self) -> list[str]:
        return self.quoted(self.body)

    def _dorm_lattice_kind(self) -> list[tuple[str, ...]]:
        if not self.kind:
            return []
        return _dormant_token_lattice(self.kind, width=2)

    def _dorm_budget_gap(self, price: float | None) -> float | None:
        if self.budget is None or price is None:
            return None
        return round(self.budget - price, 4)

    def _dorm_search_seed(self) -> str:
        hints = self._dorm_title_hints()
        return self.clip(" ".join(hints + self.tokens[:3]), 6)

    def _dorm_named_count(self) -> int:
        return len(self.named)

    def _dorm_codes_joined(self) -> str:
        return ",".join(self.codes)

    def _dorm_phrase_lens(self) -> list[int]:
        return [len(p) for p in self.phrases]

    def _dorm_mirror_map(self) -> dict[str, Any]:
        return dict(self._dorm_summary_map())

    def _dorm_flag_list(self) -> list[str]:
        return [k for k, v in self._dorm_soft_flags().items() if v]


    def _dorm_rung_head(self) -> str | None:
        return self.rungs[0] if self.rungs else None

    def _dorm_token_head(self) -> list[str]:
        return self.tokens[:5]

    def _dorm_has_maker(self) -> bool:
        return bool(self.maker)

    def _dorm_has_aisle(self) -> bool:
        return bool(self.aisle)

    def _dorm_has_kind(self) -> bool:
        return bool(self.kind)

    def _dorm_budget_int(self) -> int | None:
        return int(self.budget) if self.budget is not None else None

    def _dorm_cloud_top(self, n: int = 5) -> list[str]:
        cloud = self._dorm_phrase_cloud()
        return [k for k, _ in sorted(cloud.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]

class Rack:
    """Observed listings, live stock, cart, and ranked boards."""

    def __init__(self, ask: Ask, catalog: Catalog) -> None:
        self.ask = ask
        self.catalog = catalog
        self.products: dict[str, dict[str, Any]] = {}
        self.stock: dict[str, dict[str, Any]] = {}
        self.seen: set[str] = set()
        self.held: set[str] = set()
        self.tried: set[str] = set()
        self.tipped: set[str] = set()
        self.scouts = 0
        self.empty = 0
        self.rung = 0
        self.relax = 0
        self.adds = 0
        self.orders = 0
        self.need_check = False
        self.shifted = False
        self.asked = False
        self.fresh_view = False
        self._fresh_pid: str | None = None
        self._dropped: set[str] = set()
        self.scouts_at_shift: int | None = None
        self.errors: list[str] = []
        self.notes: list[str] = []

    def take_rows(self, rows: Any) -> int:
        count = 0
        if not isinstance(rows, list):
            return 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            pid = row.get("product_id")
            if not isinstance(pid, str) or not pid:
                continue
            self.products[pid] = self.catalog.flatten({**self.products.get(pid, {}), **row})
            count += 1
        return count

    def mark_call(self, name: str, args: dict[str, Any]) -> None:
        if name in Catalog.SCOUT:
            self.scouts += 1
            q = args.get("query")
            if isinstance(q, str) and q.strip():
                self.tried.add(q.strip().lower())
        if name in Catalog.HINT:
            pid = args.get("product_id")
            if isinstance(pid, str) and pid:
                self.tipped.add(pid)
        if name in Catalog.LOOK and name != "message":
            pid = args.get("product_id")
            if isinstance(pid, str) and pid:
                self.seen.add(pid)
            ids = args.get("product_ids")
            if isinstance(ids, list):
                for item in ids:
                    if isinstance(item, str) and item:
                        self.seen.add(item)
        if name == "add_to_cart":
            self.adds += 1
            self.need_check = True
            pid = args.get("product_id")
            if isinstance(pid, str):
                self.held.add(pid)
        if name == "remove_from_cart":
            self.need_check = True
            if self.shifted:
                self.fresh_view = True
            pid = args.get("product_id")
            if isinstance(pid, str):
                self.held.discard(pid)
                self._dropped.add(pid)
        if name == "view":
            pid = args.get("product_id")
            if isinstance(pid, str) and pid:
                if self.fresh_view and (not self._fresh_pid or pid == self._fresh_pid):
                    self.fresh_view = False
                    self._fresh_pid = None
        if name in Catalog.CLOSE:
            self.orders += 1
        if name == "message":
            self.asked = False

    def shock(self) -> None:
        if not self.shifted:
            self.scouts_at_shift = self.scouts
        self.shifted = True
        self.need_check = True

    def hear(self, name: str, args: dict[str, Any], observation: Any) -> None:
        payload = self.catalog.peel(observation)
        blob = json.dumps(observation, ensure_ascii=False, default=str).lower()
        if any(
            tok in blob
            for tok in (
                "stockout",
                "out of stock",
                "price_change",
                "new_price",
                '"in_stock": false',
                'in_stock": false',
                "unavailable",
            )
        ):
            self.shock()
        err = None
        if isinstance(observation, dict):
            err = observation.get("error")
            nested = observation.get("observation")
            if err is None and isinstance(nested, dict):
                err = nested.get("error")
        if err:
            self.errors.append(str(err)[:220])
            self.errors = self.errors[-4:]
        hits = 0
        hits += self.take_rows(payload.get("results"))
        hits += self.take_rows(payload.get("compared"))
        hits += self.take_rows(payload.get("products"))
        if name in Catalog.SCOUT:
            if hits == 0:
                self.empty += 1
                self.rung = min(self.rung + 1, max(0, len(self.ask.rungs) - 1))
                self.relax = min(self.relax + 1, 3)
            else:
                self.empty = 0
        if name == "view" or payload.get("product_id"):
            pid = payload.get("product_id") or args.get("product_id")
            if isinstance(pid, str) and pid:
                self.seen.add(pid)
                self.products[pid] = self.catalog.flatten({**self.products.get(pid, {}), **payload})
                variants = payload.get("variants")
                if isinstance(variants, list):
                    for variant in variants:
                        if isinstance(variant, dict):
                            self.take_rows([{"product_id": pid, **variant}])
        if name == "inspect_stock" or (
            payload.get("product_id") and "in_stock" in payload and "sku" in payload
        ):
            pid = str(payload.get("product_id") or args.get("product_id") or "")
            sku = str(payload.get("sku") or args.get("sku") or "")
            if pid:
                self.stock[f"{pid}::{sku}"] = dict(payload)
                self.stock[pid] = dict(payload)
                if payload.get("in_stock") is False:
                    self.shock()
        cart = payload.get("cart")
        if isinstance(cart, list):
            self.held = {
                str(row.get("product_id"))
                for row in cart
                if isinstance(row, dict) and row.get("product_id")
            }
        added = payload.get("added")
        if isinstance(added, dict) and added.get("product_id"):
            self.take_rows([added])
        if isinstance(payload.get("budget"), (int, float)):
            self.ask.budget = float(payload["budget"])

    def listen(self, content: str) -> None:
        text = content.strip()
        if not text:
            return
        self.notes.append(text[:500])
        self.notes = self.notes[-6:]
        self.need_check = True
        if "?" in text:
            self.asked = True
        low = text.casefold()
        chatter = bool(re.search(r"\b(more expensive|price increased|repriced|from \d)", low))
        cap = bool(re.search(r"\b(budget|balance|afford|can only|at most|spend|up to)\b", low))
        if any(
            tok in low
            for tok in ("out of stock", "stockout", "no longer available", "unavailable")
        ):
            self.shock()
        if chatter and not cap:
            self.shock()
            return
        if cap:
            fresh = Ask.spend(text)
            if fresh is not None:
                if self.ask.budget is None or fresh <= self.ask.budget + 1e-6:
                    self.ask.budget = fresh
                    self.shock()
        elif any(tok in low for tok in ("price", "stock", "changed", "instead", "now")):
            self.shock()

    def overlay(self, row: dict[str, Any]) -> dict[str, Any]:
        pid = str(row.get("product_id") or "")
        sku = str(row.get("sku") or "")
        live = None
        if pid and sku:
            live = self.stock.get(f"{pid}::{sku}")
        if live is None and pid:
            live = self.stock.get(pid)
        if not isinstance(live, dict):
            return row
        merged = dict(row)
        if live.get("price") not in (None, ""):
            merged["price"] = live.get("price")
            merged["current_price"] = live.get("price")
        if "in_stock" in live:
            merged["in_stock"] = live.get("in_stock")
        if live.get("sku") and not merged.get("sku"):
            merged["sku"] = live.get("sku")
        return merged

    def worth(self, row: dict[str, Any]) -> float:
        hay = " ".join(
            str(row.get(k) or "")
            for k in ("title", "brand", "category", "category_path", "sku")
        ).lower()
        facts = row.get("facts")
        if isinstance(facts, dict):
            hay += " " + " ".join(f"{k} {v}" for k, v in facts.items()).lower()
        score = 0.0
        for tok in self.ask.tokens:
            if tok.lower() in hay:
                score += 1.6 if not any(ch.isdigit() for ch in tok) else 1.1
        if self.ask.maker and self.ask.maker.lower() in hay:
            score += 2.2
        if self.ask.kind and self.ask.kind.lower() in hay:
            score += 2.0
        if self.ask.aisle:
            leaf = self.ask.aisle.split("/")[-1].split(",")[-1].strip().lower()
            if leaf and leaf in hay:
                score += 2.0
        for _field, value in self.ask.specs:
            surface = str(value).casefold()
            if surface and surface in hay:
                score += 3.0
                continue
            want = Catalog.as_float(value)
            if want is None:
                continue
            hit = False
            for nest in Catalog._NEST:
                block = row.get(nest)
                if not isinstance(block, dict):
                    continue
                for fv in block.values():
                    got = Catalog.as_float(fv)
                    if got is not None and abs(got - want) < 1e-6:
                        hit = True
                        break
                    if surface and surface in str(fv).casefold():
                        hit = True
                        break
                if hit:
                    break
            if hit:
                score += 3.0
            elif str(value).split()[0].casefold() in hay:
                score += 0.8
        price = row.get("price", row.get("current_price"))
        try:
            if self.ask.budget is not None and price is not None:
                p = float(price)
                if p <= self.ask.budget:
                    score += 1.4 + max(0.0, 1.0 - p / max(self.ask.budget, 1.0))
                else:
                    score -= 3.5
        except (TypeError, ValueError):
            pass
        if row.get("in_stock") is True:
            score += 1.5
        elif row.get("in_stock") is False:
            score -= 5.0
        return score

    def _axis_num(self, row: dict[str, Any], attr: str) -> float | None:
        tokens = [
            t
            for t in re.findall(r"[a-z0-9]+", (attr or "").casefold())
            if t not in _MUTE and len(t) > 1
        ]
        bags: list[tuple[str, Any]] = []
        facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
        bags.extend((str(k), v) for k, v in row.items() if not str(k).startswith("_"))
        bags.extend((str(k), v) for k, v in facts.items())
        for key, val in bags:
            kl = key.casefold().replace(" ", "_")
            if tokens and not any(t in kl or kl in t for t in tokens):
                continue
            num = Catalog.as_float(val)
            if num is not None:
                return num
        blob = json.dumps(row, ensure_ascii=False, default=str).casefold()
        for tok in tokens:
            m = re.search(rf"(\d+(?:\.\d+)?)\s*{re.escape(tok)}\b", blob)
            if m:
                return float(m.group(1))
            m = re.search(rf"{re.escape(tok)}\D{{0,6}}(\d+(?:\.\d+)?)", blob)
            if m:
                return float(m.group(1))
        if tokens:
            last = tokens[-1]
            m = re.search(rf"(\d+(?:\.\d+)?){re.escape(last)}\b", blob)
            if m:
                return float(m.group(1))
        return None

    def _rank_named(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def key(row: dict[str, Any]) -> tuple:
            parts: list[float] = []
            for direction, attr in self.ask.ranks:
                val = self._axis_num(row, attr)
                if val is None:
                    parts.append(0.0)
                    continue
                parts.append(val if direction == "max" else -val)
            price = Catalog.as_float(row.get("current_price") or row.get("price"))
            parts.append(-(price if price is not None else 1e18))
            return tuple(parts)

        return sorted(rows, key=key, reverse=True)

    def slate(self, limit: int = 8) -> list[dict[str, Any]]:
        if self.ask.named:
            named_rows: list[dict[str, Any]] = []
            for pid, sku in self.ask.named:
                row = dict(self.products.get(pid) or {"product_id": pid, "sku": sku})
                row.setdefault("product_id", pid)
                row.setdefault("sku", sku)
                named_rows.append(row)
            named_rows = [self.overlay(r) for r in named_rows]
            source = (
                self._rank_named(named_rows)
                if self.ask.ranks
                else sorted(named_rows, key=self.worth, reverse=True)
            )
        else:
            source = sorted(
                (self.overlay(dict(r)) for r in self.products.values()),
                key=self.worth,
                reverse=True,
            )
        out: list[dict[str, Any]] = []
        for row in source:
            pid = str(row.get("product_id") or "")
            if not pid:
                continue
            if row.get("in_stock") is False:
                continue
            if self.shifted and pid in self._dropped:
                live = self.stock.get(pid) or {}
                if live.get("in_stock") is False:
                    continue
            src = row
            card = {
                "product_id": src.get("product_id"),
                "sku": src.get("sku"),
                "title": src.get("title"),
                "brand": src.get("brand"),
                "price": src.get("price", src.get("current_price")),
                "in_stock": src.get("in_stock"),
                "match": round(self.worth(src), 2),
            }
            out.append(card)
            if len(out) >= limit:
                break
        return out

    def tops(self, n: int = 3) -> list[str]:
        return [
            str(row["product_id"])
            for row in self.slate(n)
            if row.get("product_id") and row.get("in_stock") is not False
        ]

    def winner(self) -> dict[str, Any] | None:
        for row in self.slate(5):
            if row.get("product_id") and row.get("in_stock") is not False:
                return row
        return None



    # --- dormant Rack helpers (unwired) ----------------------------------------

    def _dorm_board_ids(self) -> list[str]:
        return list(self.products.keys())

    def _dorm_seen_ratio(self) -> float:
        if not self.products:
            return 0.0
        return len(self.seen) / float(len(self.products))

    def _dorm_stock_matrix(self) -> list[tuple[str, bool | None, float | None]]:
        rows: list[tuple[str, bool | None, float | None]] = []
        for pid, snap in self.stock.items():
            live = snap.get("in_stock")
            price = Catalog.as_float(snap.get("price") or snap.get("current_price"))
            rows.append((pid, live if isinstance(live, bool) else None, price))
        return rows

    def _dorm_error_tail(self, n: int = 3) -> list[str]:
        return list(self.errors[-n:])

    def _dorm_held_skus(self) -> list[str]:
        out: list[str] = []
        for pid in self.held:
            row = self.products.get(pid) or {}
            sku = row.get("sku")
            if sku:
                out.append(str(sku))
        return out

    def _dorm_shortlist_blob(self) -> str:
        bits: list[str] = []
        for pid, row in list(self.products.items())[:8]:
            title = str(row.get("title") or "")[:40]
            bits.append(f"{pid}:{title}")
        return " | ".join(bits)

    def _dorm_scout_depth(self) -> dict[str, int]:
        return {
            "scouts": self.scouts,
            "views": len(self.seen),
            "carted": len(self.held),
            "orders": self.orders,
            "tried_queries": len(self.tried),
        }

    def _dorm_price_spread(self) -> tuple[float | None, float | None]:
        prices: list[float] = []
        for row in self.products.values():
            p = Catalog.as_float(row.get("price") or row.get("current_price"))
            if p is not None:
                prices.append(p)
        if not prices:
            return None, None
        return min(prices), max(prices)

    def _dorm_overlay_copy(self) -> dict[str, dict[str, Any]]:
        return {pid: dict(snap) for pid, snap in self.stock.items()}

    def _dorm_winner_mirror(self) -> dict[str, Any] | None:
        pick = self.winner()
        return dict(pick) if isinstance(pick, dict) else None

    def _dorm_reject_marks(self) -> list[str]:
        return sorted(str(x) for x in getattr(self, "rejected", set()) or [])

    def _dorm_freshness(self) -> str | None:
        return getattr(self, "_fresh_pid", None)

    def _dorm_compare_pairs(self) -> list[tuple[str, str]]:
        ids = list(self.products.keys())
        pairs: list[tuple[str, str]] = []
        i = 0
        while i + 1 < len(ids) and len(pairs) < 6:
            pairs.append((ids[i], ids[i + 1]))
            i += 2
        return pairs

    def _dorm_need_flags(self) -> dict[str, bool]:
        return {
            "need_check": bool(self.need_check),
            "shifted": bool(self.shifted),
            "tipped": bool(getattr(self, "tipped", False)),
            "has_products": bool(self.products),
        }

    def _dorm_title_index(self) -> dict[str, str]:
        idx: dict[str, str] = {}
        for pid, row in self.products.items():
            title = str(row.get("title") or "").strip()
            if title:
                idx[pid] = title
        return idx

    def _dorm_merge_preview(self, row: dict[str, Any]) -> dict[str, Any]:
        pid = str(row.get("product_id") or "")
        prev = self.products.get(pid) or {}
        return self.catalog.flatten({**prev, **row})


    def _dorm_product_count(self) -> int:
        return len(self.products)

    def _dorm_seen_list(self) -> list[str]:
        return list(self.seen)

    def _dorm_held_list(self) -> list[str]:
        return list(self.held)

    def _dorm_tried_list(self) -> list[str]:
        return sorted(self.tried)

    def _dorm_stock_pids(self) -> list[str]:
        return list(self.stock.keys())

    def _dorm_in_stock_pids(self) -> list[str]:
        out: list[str] = []
        for pid, snap in self.stock.items():
            if snap.get("in_stock") is True:
                out.append(pid)
        return out

    def _dorm_oos_pids(self) -> list[str]:
        out: list[str] = []
        for pid, snap in self.stock.items():
            if snap.get("in_stock") is False:
                out.append(pid)
        return out

    def _dorm_avg_price(self) -> float | None:
        lo, hi = self._dorm_price_spread()
        if lo is None or hi is None:
            return None
        return round((lo + hi) / 2.0, 4)

    def _dorm_brand_set(self) -> set[str]:
        brands: set[str] = set()
        for row in self.products.values():
            b = row.get("brand")
            if isinstance(b, str) and b.strip():
                brands.add(b.strip())
        return brands

    def _dorm_sku_set(self) -> set[str]:
        skus: set[str] = set()
        for row in self.products.values():
            s = row.get("sku")
            if isinstance(s, str) and s.strip():
                skus.add(s.strip())
        return skus

    def _dorm_title_tokens(self) -> list[str]:
        bag: list[str] = []
        for row in self.products.values():
            title = str(row.get("title") or "")
            bag.extend(re.findall(r"[a-z0-9]+", title.casefold())[:6])
        return bag[:40]

    def _dorm_shift_state(self) -> str:
        if self.shifted:
            return "shifted"
        if self.need_check:
            return "check"
        if self.held:
            return "held"
        return "open"

    def _dorm_board_digest(self) -> str:
        return _dormant_echo_digest(self._dorm_board_ids(), salt="board")

    def _dorm_empty(self) -> bool:
        return not self.products

    def _dorm_ready_to_close(self) -> bool:
        return bool(self.held) and not self.need_check

    def _dorm_view_gap(self) -> list[str]:
        return [pid for pid in self.products if pid not in self.seen]

    def _dorm_cart_gap(self) -> list[str]:
        return [pid for pid in self.seen if pid not in self.held]

    def _dorm_snapshot(self) -> dict[str, Any]:
        return {
            **self._dorm_scout_depth(),
            "spread": self._dorm_price_spread(),
            "flags": self._dorm_need_flags(),
        }


    def _dorm_error_count(self) -> int:
        return len(self.errors)

    def _dorm_orders(self) -> int:
        return int(self.orders)

    def _dorm_scouts(self) -> int:
        return int(self.scouts)

    def _dorm_has_winner(self) -> bool:
        return self.winner() is not None

    def _dorm_stock_count(self) -> int:
        return len(self.stock)

class Ticket:
    """Schema-grounded order arguments. Extra open fields are omitted unless required."""

    META = {"price", "in_stock", "brand", "category", "title"}
    NOISE = {
        "spec",
        "specification",
        "attribute",
        "attributes",
        "feature",
        "features",
        "property",
        "properties",
        "detail",
        "details",
        "info",
        "data",
        "value",
        "field",
        "option",
        "options",
        "param",
        "parameter",
        "parameters",
        "quality",
        "metric",
        "metrics",
        "requirement",
        "requirements",
    }

    def __init__(
        self,
        catalog: Catalog,
        close: str,
        ask: Ask,
        product: dict[str, Any],
        live: dict[str, Any],
    ) -> None:
        self.catalog = catalog
        self.close = close
        self.ask = ask
        self.product = product
        self.live = live
        self.claims: list[dict[str, str]] = []
        self.used: set[str] = set()
        self.allowed: set[str] = set()
        self.soft: set[str] = set()
        self.desc = ""

    def space(self) -> tuple[set[str], set[str], str]:
        def dig(node: Any, *path: str) -> Any:
            cur = node
            for key in path:
                if type(cur) is not dict:
                    return {}
                cur = cur.get(key)
            return cur if type(cur) is dict else {}

        order_props = dig(self.catalog.params(self.close), "properties")
        just = dig(order_props, "justification")
        claim_node = dig(just, "properties", "claims")
        field_node = dig(claim_node, "items", "properties", "field")
        allowed: set[str] = {"price", "in_stock"}
        choices = field_node.get("enum")
        if isinstance(choices, list):
            for item in choices:
                if isinstance(item, str) and item.strip():
                    allowed.add(item.strip())
        prose_bits: list[str] = []
        for node in (field_node, just, claim_node):
            blob = node.get("description")
            if isinstance(blob, str) and blob.strip():
                prose_bits.append(blob.strip())
        desc = " ".join(prose_bits)
        soft: set[str] = set()
        low = desc.casefold()
        # Soft axes: sentences that mention a rationale/capability, then harvest tokens.
        windows: list[str] = []
        for mark in ("reason", "reasons", "capability", "capabilities", "useful"):
            at = 0
            while True:
                hit = low.find(mark, at)
                if hit < 0:
                    break
                end = len(low)
                for stop in (".", ";", "!"):
                    j = low.find(stop, hit)
                    if j >= 0:
                        end = min(end, j)
                windows.append(low[hit:end])
                at = hit + len(mark)
        if not windows:
            windows = [low]

        def admit(tok: str) -> None:
            if tok in _SCHEMA_MUTE or len(tok) < 2:
                return
            if tok in {"price", "in_stock", "category", "specification"}:
                return
            soft.add(tok)
            allowed.add(tok)

        for win in windows:
            for tok in re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", win):
                admit(tok)
            for piece in re.split(r"[,/;]| \bor\b | \band\b ", win):
                words = [
                    w
                    for w in re.findall(r"[a-z][a-z0-9]+", piece)
                    if w not in _SCHEMA_MUTE and len(w) >= 2
                ]
                if 1 <= len(words) <= 3:
                    admit("_".join(words))
                for w in words:
                    if len(w) >= 3:
                        admit(w)
        allowed.discard("specification")
        soft -= {"price", "in_stock", "category", "specification"}
        return allowed, soft, desc

    def labels(self) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()

        def keep(raw: str) -> None:
            lab = re.sub(r"\s+", " ", str(raw or "").strip())
            if not lab or len(lab) > 48:
                return
            key = lab.casefold()
            if key in seen or key in _MUTE:
                return
            seen.add(key)
            found.append(lab)

        for _f, v in self.ask.specs:
            keep(str(v))
        for tok in self.ask.phrases:
            keep(tok)
        for m in re.finditer(r"\b(\d+(?:\.\d+)?(?:\s*-?\s*[A-Za-z]{1,12})?)\b", self.ask.body):
            keep(m.group(1))
        for m in re.finditer(
            r"\b([A-Za-z][A-Za-z0-9]*(?:\s+[A-Za-z][A-Za-z0-9]*){0,2})\b",
            self.ask.body,
        ):
            lab = m.group(1).strip()
            words = lab.casefold().split()
            if not words or any(w in _MUTE for w in words):
                continue
            keep(lab)
        found.sort(key=lambda s: (-len(s), -len(s.split())))
        return found

    @staticmethod
    def attested(label: str, evidence: str) -> bool:
        low = str(label).casefold().strip()
        ev = evidence.casefold()
        if not low:
            return False
        if low in ev or low.replace(" ", "") in ev.replace(" ", "").replace("-", ""):
            return True
        nums = re.findall(r"\d+(?:\.\d+)?", low)
        if nums and all(n in ev for n in nums):
            return True
        parts = [w for w in re.findall(r"[a-z0-9]+", low) if w not in _MUTE and len(w) > 1]
        if len(parts) >= 2 and all(p in ev for p in parts):
            return True
        if len(parts) == 1 and parts[0] in ev and len(parts[0]) >= 3:
            return True
        return False

    @staticmethod
    def grid(text: str) -> bool:
        t = str(text or "")
        return len(re.findall(r"\d+(?:\.\d+)?", t)) >= 2 and bool(re.search(r"[x×*,/]", t))

    def _push(self, field: str, value: Any) -> None:
        if field not in self.allowed or field in self.used:
            return
        if value is None or value == "" or value == "null":
            return
        text = "true" if value is True else "false" if value is False else str(value).strip()
        if not text or text.lower() == "null" or len(text) > 80:
            return
        self.claims.append({"field": field, "value": text})
        self.used.add(field)

    def allow_axis(self, name: str) -> None:
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,40}", name or ""):
            self.allowed.add(name)

    def _has_spec(self) -> bool:
        return any(c["field"] not in self.META and c["field"] not in self.soft for c in self.claims)

    def _match_fact(self) -> tuple[str | None, Any]:
        facts = self.product.get("facts") if isinstance(self.product.get("facts"), dict) else {}
        frags = [str(v) for _f, v in self.ask.specs]
        for tok in re.findall(r"\b[A-Za-z0-9][A-Za-z0-9.+_-]{1,20}\b", self.ask.body):
            low = tok.casefold()
            if low in _MUTE or low in {"php", "usd", "eur"}:
                continue
            if any(ch.isdigit() for ch in tok) or (tok.isupper() and 2 <= len(tok) <= 6):
                if tok not in frags:
                    frags.append(tok)
        skip = {
            "title",
            "listing_id",
            "product_id",
            "sku",
            "model",
            "condition",
            "navigation",
            "brand",
            "price",
            "in_stock",
            "currency",
            "category_path",
            "options",
            "variants",
            "category",
            "current_price",
        }
        extras = [
            (k, self.product.get(k))
            for k in self.product
            if k != "facts" and not isinstance(self.product.get(k), (dict, list))
        ]
        for frag in frags:
            frag_l = frag.casefold().strip()
            if len(frag_l) < 2:
                continue
            alts = {frag_l, frag_l.replace("-", " "), re.sub(r"\s+", "", frag_l)}
            if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*[a-z%]+)?", frag_l):
                alts.add(re.split(r"\s+", frag_l)[0])
            matched_field = None
            matched_val = None
            for key, value in list(facts.items()) + extras:
                if not isinstance(key, str) or value in (None, "", True, False):
                    continue
                if key in skip:
                    continue
                text_v = str(value).casefold()
                if len(text_v) > 80:
                    continue
                text_sp = text_v.replace("-", " ")
                text_nums = set(re.findall(r"\d+(?:\.\d+)?", text_v))
                hit = any(
                    alt in text_v or alt in text_sp or alt in re.sub(r"\s+", "", text_v)
                    for alt in alts
                )
                frag_num = Catalog.as_float(frag)
                if (
                    not hit
                    and frag_num is not None
                    and (
                        str(int(frag_num)) if float(frag_num).is_integer() else str(frag_num)
                    )
                    in text_nums
                    and len(text_v) <= 40
                ):
                    hit = True
                if not hit:
                    continue
                matched_field, matched_val = key, value
                if key in facts:
                    break
            if matched_field is not None:
                return matched_field, matched_val
        return None, None

    def fill(self, args: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(args)
        needs = "justification" in self.catalog.needed(self.close)
        if not needs:
            cleaned.pop("justification", None)
            return cleaned
        self.allowed, self.soft, self.desc = self.space()
        open_spec = "specification" in self.desc.casefold() and "shopper" in self.desc.casefold()
        facts = self.product.get("facts") if isinstance(self.product.get("facts"), dict) else {}
        fact_keys = {str(k).casefold() for k in facts.keys()}
        evidence = json.dumps(self.product, ensure_ascii=False, default=str)
        ask_vals = self.labels()
        raw_just = cleaned.get("justification")
        raw_claims: list[dict[str, Any]] = []
        proposed: list[str] = []
        if isinstance(raw_just, dict) and isinstance(raw_just.get("claims"), list):
            for c in raw_just["claims"]:
                if not isinstance(c, dict):
                    continue
                raw_claims.append(c)
                f = str(c.get("field") or "").strip()
                if f and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,40}", f):
                    if f.casefold() not in self.META and f.casefold() not in self.NOISE:
                        proposed.append(f)
        if "price" in self.live:
            self._push("price", self.live.get("price"))
        elif self.product.get("price") not in (None, ""):
            self._push("price", self.product.get("price"))
        elif self.product.get("current_price") not in (None, ""):
            self._push("price", self.product.get("current_price"))
        if "in_stock" in self.live:
            self._push("in_stock", self.live.get("in_stock"))
        elif "in_stock" in self.product:
            self._push("in_stock", self.product.get("in_stock"))
        if "category" in self.allowed:
            cat = self.catalog.pick(self.product, "category")
            if isinstance(cat, list) and cat:
                cat = cat[-1]
            self._push("category", cat)
        matched_field, matched_val = self._match_fact()
        compact_match = bool(
            matched_field and matched_val is not None and not self.grid(str(matched_val))
        )
        for c in raw_claims:
            f = str(c.get("field") or "").strip()
            v = c.get("value")
            if not f or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,40}", f):
                continue
            if f in self.META or f.casefold() in self.NOISE:
                continue
            text_c = str(v).strip() if v is not None else ""
            if not text_c:
                continue
            if self.grid(text_c) or len(text_c) > 40:
                replaced = None
                for lab in ask_vals:
                    if self.attested(lab, evidence):
                        replaced = lab
                        break
                if replaced:
                    text_c = replaced
                elif self.grid(text_c):
                    continue
            if not self.attested(text_c, evidence) and not self.attested(text_c, self.ask.body):
                if text_c.casefold() not in {a.casefold() for a in ask_vals}:
                    continue
            if open_spec:
                if f in self.soft:
                    self.allow_axis(f)
                elif compact_match and f == matched_field:
                    self.allow_axis(f)
                elif f.casefold() in fact_keys:
                    continue
                else:
                    self.allow_axis(f)
            elif f not in self.allowed:
                continue
            self._push(f, text_c)
        if not self._has_spec() and matched_field is not None and matched_val is not None:
            dim = self.grid(str(matched_val))
            if open_spec and dim:
                open_fields = [
                    f
                    for f in proposed
                    if f not in self.used
                    and f.casefold() not in fact_keys
                    and f not in self.soft
                    and f not in self.META
                ]
                for lab in ask_vals:
                    if not self.attested(lab, evidence):
                        continue
                    for axis in open_fields:
                        self.allow_axis(axis)
                        self._push(axis, lab)
                        break
                    if self._has_spec():
                        break
            else:
                axis = matched_field
                if open_spec:
                    self.allow_axis(axis)
                elif axis not in self.allowed:
                    axis = None
                    for cand in sorted(self.allowed):
                        if cand in self.META:
                            continue
                        if cand in self.ask.raw.casefold() or cand.replace("_", " ") in self.ask.raw.casefold():
                            axis = cand
                            break
                if axis:
                    chosen = matched_val
                    for lab in ask_vals:
                        if self.attested(lab, str(matched_val)) or self.attested(lab, evidence):
                            if any(ch.isdigit() for ch in lab) or len(lab) <= len(str(matched_val)):
                                chosen = lab
                                break
                    self._push(axis, chosen)
        if not self._has_spec() and open_spec:
            for axis in proposed:
                if axis in self.used or axis in self.soft or axis in self.META:
                    continue
                if axis.casefold() in fact_keys:
                    continue
                for lab in ask_vals:
                    if self.attested(lab, evidence):
                        self.allow_axis(axis)
                        self._push(axis, lab)
                        break
                if self._has_spec():
                    break
        brand = self.catalog.pick(self.product, "brand")
        if brand and ("brand" in self.soft or "brand" in self.allowed):
            self.allow_axis("brand")
            self._push("brand", brand)
        elif not any(c["field"] in self.soft for c in self.claims):
            for field in sorted(self.soft):
                if field in self.META:
                    continue
                observed = self.catalog.pick(self.product, field)
                if observed is True or str(observed).lower() in {"true", "yes", "1"}:
                    self._push(field, "true")
                    break
                if observed not in (None, "", False) and isinstance(observed, (str, int, float)):
                    self._push(field, observed)
                    break
        if needs and not self.claims:
            self._push("price", self.live.get("price") or self.product.get("price"))
            self._push("in_stock", self.live.get("in_stock", self.product.get("in_stock", True)))
        if self.claims:
            cleaned["justification"] = {"claims": self.claims}
        elif not needs:
            cleaned.pop("justification", None)
        return cleaned



    # --- dormant Ticket helpers (unwired) --------------------------------------

    def _dorm_claim_names(self) -> list[str]:
        return [str(c.get("field") or "") for c in self.claims]

    def _dorm_claim_values(self) -> list[str]:
        return [str(c.get("value") or "") for c in self.claims]

    def _dorm_allowed_sorted(self) -> list[str]:
        return sorted(self.allowed)

    def _dorm_soft_sorted(self) -> list[str]:
        return sorted(self.soft)

    def _dorm_meta_collision(self) -> set[str]:
        return set(self.used) & self.META

    def _dorm_noise_hits(self, label: str) -> bool:
        low = label.casefold()
        return any(n in low for n in self.NOISE)

    def _dorm_evidence_snip(self, limit: int = 180) -> str:
        blob = json.dumps(self.product, ensure_ascii=False, default=str)
        return blob[:limit]

    def _dorm_live_vs_product(self) -> dict[str, tuple[Any, Any]]:
        keys = ("price", "in_stock", "sku", "product_id")
        out: dict[str, tuple[Any, Any]] = {}
        for key in keys:
            out[key] = (self.live.get(key), self.product.get(key))
        return out

    def _dorm_open_axes(self) -> list[str]:
        return [a for a in sorted(self.allowed) if a not in self.used and a not in self.META]

    def _dorm_justify_ready(self) -> bool:
        return "price" in self.used and "in_stock" in self.used

    def _dorm_field_grid(self) -> list[tuple[str, str]]:
        grid: list[tuple[str, str]] = []
        for field in sorted(self.allowed)[:20]:
            val = self.catalog.pick(self.product, field)
            if val not in (None, ""):
                grid.append((field, str(val)[:48]))
        return grid

    def _dorm_desc_tokens(self) -> list[str]:
        return [
            t
            for t in re.findall(r"[a-z][a-z0-9_]{2,}", self.desc.casefold())
            if t not in _SCHEMA_MUTE
        ][:30]

    def _dorm_claim_digest(self) -> str:
        parts = [f"{c.get('field')}={c.get('value')}" for c in self.claims]
        return _dormant_echo_digest(parts, salt="claim")

    def _dorm_label_probe(self) -> list[tuple[str, bool]]:
        ev = json.dumps(self.product, ensure_ascii=False, default=str)
        return [(lab, self.attested(lab, ev)) for lab in self.labels()[:12]]

    def _dorm_soft_overlap(self) -> list[str]:
        return sorted(self.soft & self.allowed)

    def _dorm_ticket_banner(self) -> str:
        return (
            f"close={self.close}; claims={len(self.claims)}; "
            f"allowed={len(self.allowed)}; soft={len(self.soft)}"
        )


    def _dorm_used_sorted(self) -> list[str]:
        return sorted(self.used)

    def _dorm_claim_count(self) -> int:
        return len(self.claims)

    def _dorm_has_soft(self) -> bool:
        return bool(self.soft)

    def _dorm_has_brand_axis(self) -> bool:
        return "brand" in self.allowed or "brand" in self.soft

    def _dorm_price_claim(self) -> str | None:
        for c in self.claims:
            if c.get("field") == "price":
                return str(c.get("value"))
        return None

    def _dorm_stock_claim(self) -> str | None:
        for c in self.claims:
            if c.get("field") == "in_stock":
                return str(c.get("value"))
        return None

    def _dorm_product_id(self) -> str:
        return str(self.product.get("product_id") or self.live.get("product_id") or "")

    def _dorm_sku(self) -> str:
        return str(self.product.get("sku") or self.live.get("sku") or "")

    def _dorm_space_sizes(self) -> tuple[int, int, int]:
        return len(self.allowed), len(self.soft), len(self.desc)

    def _dorm_labels_cap(self, n: int = 8) -> list[str]:
        return self.labels()[:n]

    def _dorm_grid_open(self) -> bool:
        return self.grid(self.ask.body)

    def _dorm_meta_only_claims(self) -> bool:
        names = set(self._dorm_claim_names())
        return bool(names) and names <= self.META

    def _dorm_missing_core(self) -> list[str]:
        return [k for k in ("price", "in_stock") if k not in self.used]

    def _dorm_soft_candidates(self) -> list[str]:
        return [s for s in sorted(self.soft) if s not in self.used][:8]

    def _dorm_value_probe(self, field: str) -> str | None:
        val = self.catalog.pick(self.product, field)
        if val in (None, ""):
            val = self.live.get(field)
        return None if val in (None, "") else str(val)

    def _dorm_attested_soft(self) -> list[str]:
        ev = self._dorm_evidence_snip(400)
        hits: list[str] = []
        for field in sorted(self.soft):
            val = self._dorm_value_probe(field)
            if val and self.attested(val, ev):
                hits.append(field)
        return hits

    def _dorm_pack_claims(self) -> list[dict[str, str]]:
        return [dict(c) for c in self.claims]

    def _dorm_close_needed(self) -> list[str]:
        return self.catalog.needed(self.close)

    def _dorm_banner_bits(self) -> list[str]:
        return self._dorm_ticket_banner().split("; ")

    def _dorm_desc_has_reason(self) -> bool:
        return "reason" in self.desc.casefold()

    def _dorm_state(self) -> dict[str, Any]:
        return {
            "claims": self._dorm_claim_count(),
            "ready": self._dorm_justify_ready(),
            "open": self._dorm_open_axes()[:6],
            "banner": self._dorm_ticket_banner(),
        }


    def _dorm_digest_short(self) -> str:
        return self._dorm_claim_digest()[:8]

    def _dorm_live_price(self) -> Any:
        return self.live.get("price") or self.live.get("current_price")

    def _dorm_live_stock(self) -> Any:
        return self.live.get("in_stock")

    def _dorm_allowed_has(self, field: str) -> bool:
        return field in self.allowed

    def _dorm_soft_has(self, field: str) -> bool:
        return field in self.soft

    def _dorm_used_has(self, field: str) -> bool:
        return field in self.used

    def _dorm_claims_jsonable(self) -> list[dict[str, str]]:
        return [{"field": str(c.get("field")), "value": str(c.get("value"))} for c in self.claims]

    def _dorm_noise_list(self) -> list[str]:
        return sorted(self.NOISE)

    def _dorm_meta_list(self) -> list[str]:
        return sorted(self.META)

class Mood:
    FIND = 0
    LOOK = 1
    TAKE = 2
    SHIFT = 3
    SEAL = 4



    # Dormant mood labels for experimental planners (unused).
    SCOUT = 10
    INSPECT = 11
    COMMIT = 12
    RECOVER = 13
    FINISH = 14

    @staticmethod
    def _dorm_label(code: int) -> str:
        return {
            Mood.FIND: "find",
            Mood.LOOK: "look",
            Mood.TAKE: "take",
            Mood.SHIFT: "shift",
            Mood.SEAL: "seal",
            Mood.SCOUT: "scout",
            Mood.INSPECT: "inspect",
            Mood.COMMIT: "commit",
            Mood.RECOVER: "recover",
            Mood.FINISH: "finish",
        }.get(code, f"mood:{code}")

    @staticmethod
    def _dorm_cycle() -> list[int]:
        return [Mood.FIND, Mood.LOOK, Mood.TAKE, Mood.SEAL]

    @staticmethod
    def _dorm_recover_path() -> list[int]:
        return [Mood.SHIFT, Mood.LOOK, Mood.TAKE, Mood.SEAL]

    @staticmethod
    def _dorm_tight(left: int, reserve: int) -> bool:
        return left <= reserve

    @staticmethod
    def _dorm_from_name(name: str) -> int:
        table = {
            "find": Mood.FIND,
            "look": Mood.LOOK,
            "take": Mood.TAKE,
            "shift": Mood.SHIFT,
            "seal": Mood.SEAL,
        }
        return table.get(name.casefold(), Mood.FIND)


    WAIT = 20
    RETRY = 21
    ABORT = 22

    @staticmethod
    def _dorm_all_codes() -> list[int]:
        return [
            Mood.FIND, Mood.LOOK, Mood.TAKE, Mood.SHIFT, Mood.SEAL,
            Mood.SCOUT, Mood.INSPECT, Mood.COMMIT, Mood.RECOVER, Mood.FINISH,
            Mood.WAIT, Mood.RETRY, Mood.ABORT,
        ]

    @staticmethod
    def _dorm_is_terminal(code: int) -> bool:
        return code in {Mood.SEAL, Mood.FINISH, Mood.ABORT}

    @staticmethod
    def _dorm_is_search(code: int) -> bool:
        return code in {Mood.FIND, Mood.SCOUT}

    @staticmethod
    def _dorm_is_inspect(code: int) -> bool:
        return code in {Mood.LOOK, Mood.INSPECT}

    @staticmethod
    def _dorm_next_default(code: int) -> int:
        chain = Mood._dorm_cycle()
        if code not in chain:
            return Mood.FIND
        i = chain.index(code)
        return chain[min(i + 1, len(chain) - 1)]

    @staticmethod
    def _dorm_labels() -> list[str]:
        return [Mood._dorm_label(c) for c in Mood._dorm_all_codes()]

    @staticmethod
    def _dorm_pair(a: int, b: int) -> str:
        return f"{Mood._dorm_label(a)}->{Mood._dorm_label(b)}"

    @staticmethod
    def _dorm_reserve(steps: int) -> int:
        return max(3, min(5, steps // 5)) if steps > 0 else 3

    @staticmethod
    def _dorm_should_shift(shifted: bool, held: bool) -> bool:
        return bool(shifted) and not held

    @staticmethod
    def _dorm_should_look(need_check: bool, left: int, reserve: int) -> bool:
        return bool(need_check) or left <= reserve

    @staticmethod
    def _dorm_should_take(has_products: bool, seen: bool) -> bool:
        return bool(has_products) and bool(seen)

    @staticmethod
    def _dorm_table() -> dict[str, int]:
        return {Mood._dorm_label(c): c for c in Mood._dorm_all_codes()}

    @staticmethod
    def _dorm_explain(code: int, left: int, age: float) -> str:
        return f"{Mood._dorm_label(code)}|left={left}|age={age:.1f}"

    @staticmethod
    def _dorm_priority(code: int) -> int:
        order = {
            Mood.SHIFT: 0, Mood.SEAL: 1, Mood.LOOK: 2, Mood.TAKE: 3, Mood.FIND: 4,
        }
        return order.get(code, 9)

    @staticmethod
    def _dorm_sort_codes(codes: list[int]) -> list[int]:
        return sorted(codes, key=Mood._dorm_priority)

    @staticmethod
    def _dorm_clamp(code: int) -> int:
        return code if code in Mood._dorm_table().values() else Mood.FIND

    @staticmethod
    def _dorm_note(code: int) -> dict[str, Any]:
        return {
            "code": code,
            "label": Mood._dorm_label(code),
            "terminal": Mood._dorm_is_terminal(code),
            "search": Mood._dorm_is_search(code),
        }


    @staticmethod
    def _dorm_blend(a: int, b: int) -> str:
        return Mood._dorm_pair(a, b)

    @staticmethod
    def _dorm_wait_note() -> dict[str, Any]:
        return Mood._dorm_note(Mood.WAIT)

    @staticmethod
    def _dorm_retry_note() -> dict[str, Any]:
        return Mood._dorm_note(Mood.RETRY)

    @staticmethod
    def _dorm_abort_note() -> dict[str, Any]:
        return Mood._dorm_note(Mood.ABORT)

    @staticmethod
    def _dorm_searchish(code: int) -> bool:
        return Mood._dorm_is_search(code) or Mood._dorm_is_inspect(code)

    @staticmethod
    def _dorm_flow() -> list[str]:
        return [Mood._dorm_label(c) for c in Mood._dorm_cycle()]

    @staticmethod
    def _dorm_recover_flow() -> list[str]:
        return [Mood._dorm_label(c) for c in Mood._dorm_recover_path()]

    @staticmethod
    def _dorm_priority_list() -> list[tuple[str, int]]:
        return [(Mood._dorm_label(c), Mood._dorm_priority(c)) for c in Mood._dorm_all_codes()]

class ShopRun:
    """Intent-queue episode: FIND → LOOK → TAKE → (SHIFT) → SEAL."""

    def __init__(self, problem_data: dict[str, Any]) -> None:
        environment = problem_data["environment"]
        self.binding = environment["binding"]
        policy = environment["policy_view"]
        tools = list(policy.get("tools") or [])
        if not tools:
            raise RuntimeError("tool policy has no callables")
        self.catalog = Catalog(tools)
        self.max_steps = int(policy["max_steps"])
        self.max_calls = max(1, min(6, int(policy.get("max_calls_per_turn", 1))))
        self.problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        self.query = str(policy.get("query") or problem_data.get("query") or "")
        self.close = self.catalog.close or "place_test_order"
        self.ask = Ask(self.query)
        self.rack = Rack(self.ask, self.catalog)
        for pid, sku in self.ask.named:
            self.rack.products.setdefault(pid, {"product_id": pid, "sku": sku})
        self.started = time.monotonic()
        self.trace: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.doctrine()},
            {"role": "user", "content": self.query},
        ]
        self.pending = self.first_moves()

    @property
    def age(self) -> float:
        now = time.monotonic()
        return now - self.started

    def mood(self, turn: int) -> int:
        left = self.max_steps - turn + 1
        tight = self.age >= _CLOCK * 0.65
        reserve = max(3, min(5, self.max_steps // 5))
        rack = self.rack
        if rack.shifted:
            return Mood.SHIFT
        if rack.need_check or (rack.held and left <= reserve):
            return Mood.LOOK
        if rack.held and not rack.need_check:
            return Mood.SEAL
        if (tight or left <= reserve) and rack.products:
            return Mood.TAKE
        if rack.products and rack.seen:
            return Mood.TAKE
        if rack.products and (rack.scouts >= 2 or left <= reserve + 3):
            return Mood.LOOK
        if rack.products and not rack.seen:
            return Mood.LOOK
        return Mood.FIND

    def doctrine(self) -> str:
        needs = "justification" in self.catalog.needed(self.close)
        _allowed, soft, _desc = Ticket(
            self.catalog, self.close, self.ask, {}, {}
        ).space()
        soft_txt = ", ".join(sorted(soft)) or "brand"
        rec = self.catalog.hint
        lines = [
            "Drive catalog tools until an observation sets done=true.",
            "Meet firm shopper limits (budget, stock, named specs) then close the order.",
            "",
            "Operating notes:",
            "- Keep search strings to a handful of keywords; climb to a simpler rung after empty hits.",
            "- If filter is available, lock stock + max_price; clear brand pins on empty filters.",
            "- Open details on the best matches, then inspect live stock before carting.",
        ]
        if rec:
            lines.append(f"- Call {rec} with the chosen product_id before carting when that tool exists.")
        lines.extend(
            [
                f"- Sequence: scout → inspect → stock check → cart → re-check → {self.close}.",
                "- When the ask names pid::sku pairs, cart the current board leader first; after a stock change, re-rank survivors.",
                "- Do not order in the same turn as add_to_cart. Re-check after cart or shopper notes.",
                "- If stock or price breaks the ask, drop the line, view a replacement, then re-verify.",
                f"- About {self.max_steps} turns, ≤{self.max_calls} calls each. Finish with turns remaining.",
                "- Copy product_id and sku from observations only.",
            ]
        )
        if needs:
            lines.append(
                "- Checkout needs justification: live price + stock; one snake_case attribute "
                "tied to a constrained property with an evidenced short value; and when the schema "
                f"lists soft axes, include one ({soft_txt}). Skip hollow labels (spec/feature). "
                "Never fabricate values."
            )
        else:
            lines.append(
                "- Leave justification off unless the order tool requires it; observed fields only."
            )
        lines.extend(["", "Tools:", self.catalog.sheet(), "", "Always emit tool calls."])
        return "\n".join(lines)

    def briefing(self, turn: int, mood: int) -> str:
        left = self.max_steps - turn + 1
        remain = max(0, int(_CLOCK - self.age))
        next_q = self.ask.next_q(self.rack.rung, self.rack.tried)
        label = ("FIND", "LOOK", "TAKE", "SHIFT", "SEAL")[mood]
        tips = [
            f"Intent={label}. Turn {turn}/{self.max_steps} ({left} left). "
            f"Up to {self.max_calls} call(s) this turn. ~{remain}s left.",
            self.ask.sketch(),
            f"Next search idea: {next_q!r}. Two to six words; never paste the full paragraph.",
        ]
        if "filter" in self.catalog.names:
            tips.append(
                "Filter: in_stock=true and max_price from the budget; category when named. "
                "After empty filter results, omit the brand pin."
            )
        if mood == Mood.FIND:
            tips.append("Scout first. Prefer in-stock rows. Climb to a simpler query after a miss.")
            if self.rack.empty:
                tips.append("Last scout was empty — keep product type plus one key spec.")
        elif mood == Mood.LOOK:
            tips.append("Open details on the best matches (view/compare) before carting.")
            tops = self.rack.tops(3)
            if tops and "compare" in self.catalog.names and len(tops) >= 2:
                tips.append(f"Useful compare set: {tops[:3]}")
            if self.rack.held or self.rack.need_check:
                tips.append(
                    f"Re-check live price/stock (inspect_stock / inspect_cart) before {self.close}. "
                    "Answer a shopper question with message first."
                )
        elif mood == Mood.TAKE:
            tips.append(
                "Pick the best in-stock, in-budget option. If a recommend tool exists, call it, "
                "then check stock and cart."
            )
            if self.catalog.hint and not self.rack.tipped:
                tips.append("Call the recommend tool with your chosen product_id.")
        elif mood == Mood.SHIFT:
            tips.append(
                "Something changed. Remove invalid cart lines, VIEW the next in-budget option, "
                "inspect_stock, then add and re-verify before order."
            )
        else:
            tips.append(
                f"Finish: if the carted item still fits, call {self.close} with schema-valid args. "
                "Stop exploring new products."
            )
        if self.rack.asked and "message" in self.catalog.names:
            tips.append("Shopper asked a question — reply with message using observed facts first.")
        if self.rack.errors:
            tips.append("Last tool faults → " + " · ".join(self.rack.errors[-2:]))
        if left <= 4 and self.rack.orders == 0:
            tips.append("Few turns left — checkout the best verified item now.")
        if remain <= 35 and self.rack.orders == 0 and self.rack.products:
            tips.append("Episode time is low — stop searching and complete checkout.")
        if self.rack.notes:
            tips.append("Latest shopper note overrides only the fields it changes.")
        short = self.rack.slate()
        if short:
            tips.append("Current candidates: " + json.dumps(short)[:900])
        return "Briefing:\n- " + "\n- ".join(tips)

    def _call(self, cid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": cid,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }

    def admit(self, raw_calls: list[dict[str, Any]], mood: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        saw_add = False
        saw_verify = False
        saw_hint = False
        for call in raw_calls:
            if len(out) >= self.max_calls:
                break
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(fn.get("name") or call.get("name") or "")
            if name not in self.catalog.names:
                continue
            args = self.catalog.decode(fn.get("arguments", call.get("arguments")))
            if name in Catalog.SCOUT:
                args = self.catalog.scout_fix(
                    name,
                    args,
                    self.ask,
                    self.rack.rung,
                    self.rack.tried,
                    self.rack.relax,
                )
            else:
                args = self.catalog.mold(name, args)
            if name == self.close:
                if saw_add:
                    continue
                if self.rack.need_check and not saw_verify:
                    continue
                pid = str(args.get("product_id") or (next(iter(self.rack.held), "") if self.rack.held else ""))
                sku = str(args.get("sku") or "")
                live = self.rack.stock.get(f"{pid}::{sku}") or self.rack.stock.get(pid) or {}
                if live.get("in_stock") is False:
                    continue
                product = self.rack.products.get(pid) or {}
                args = Ticket(self.catalog, self.close, self.ask, product, live).fill(args)
            if name in Catalog.HINT:
                saw_hint = True
                if not args.get("product_id"):
                    pick = self.rack.winner()
                    if pick and pick.get("product_id"):
                        args["product_id"] = pick["product_id"]
            if name in Catalog.LOOK:
                saw_verify = True
            if name == "add_to_cart":
                pid = str(args.get("product_id") or "")
                need_view = bool(
                    pid
                    and "view" in self.catalog.names
                    and (pid not in self.rack.seen or self.rack.fresh_view)
                )
                if need_view and not any(
                    (c.get("function") or {}).get("name") == "view" for c in out
                ):
                    if self.rack.fresh_view:
                        self.rack._fresh_pid = pid
                    args = {"product_id": pid}
                    name = "view"
                saw_add = name == "add_to_cart"
            out.append(self._call(str(call.get("id") or f"call-{len(out) + 1}"), name, args))
        rec = self.catalog.hint
        if (
            rec
            and mood in {Mood.TAKE, Mood.LOOK, Mood.SEAL}
            and not saw_hint
            and not self.rack.tipped
            and len(out) < self.max_calls
        ):
            pick = self.rack.winner()
            if pick and pick.get("product_id"):
                out.insert(0, self._call(f"auto-rec-{len(out) + 1}", rec, {"product_id": pick["product_id"]}))
                out = out[: self.max_calls]
        return out

    def shorten(self, keep: int = 5) -> list[dict[str, Any]]:
        """Keep system+goal intact; compress older tool observations from the front."""
        trail = self.messages
        if len(trail) <= 12:
            return list(trail)
        prefix = trail[:2]
        rest = trail[2:]
        protect = keep * 3
        stale_end = max(0, len(rest) - protect)
        out: list[dict[str, Any]] = list(prefix)
        for idx, raw in enumerate(rest):
            row = dict(raw)
            if idx < stale_end and row.get("role") == "tool":
                blob = row.get("content")
                try:
                    parsed = json.loads(blob) if isinstance(blob, str) else blob
                except (TypeError, json.JSONDecodeError):
                    parsed = blob
                row["content"] = self.catalog.pack(parsed, 1800)
            out.append(row)
        return out

    def think(self, models: list[str]) -> dict[str, Any]:
        req = self.shorten()
        err = "inference failed"
        idx = 0
        while idx < len(models):
            model = models[idx]
            idx += 1
            payload = {
                "model": model,
                "messages": req,
                "tools": self.catalog.tools,
                "tool_choice": "required",
                "temperature": 0,
                "max_tokens": 1800,
            }
            resp = _HTTP.post("/inference/chat/completions", json_data=payload)
            if not resp:
                err = f"unavailable:{model}"
                continue
            choices = resp.get("choices") if isinstance(resp, dict) else None
            if not isinstance(choices, list) or not choices:
                err = f"malformed:{model}"
                continue
            first = choices[0] if isinstance(choices[0], dict) else {}
            msg = first.get("message")
            if isinstance(msg, dict):
                return msg
            err = f"malformed:{model}"
        raise RuntimeError(err)

    def env_turn(self, turn: int, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        tag = f"{self.problem_id}-turn-{turn}"
        packed: list[dict[str, Any]] = []
        i = 0
        while i < len(tool_calls):
            call = tool_calls[i]
            i += 1
            packed.append(
                {
                    "call_id": f"{tag}-{i}",
                    "action": {
                        "name": call["function"]["name"],
                        "args": self.catalog.decode(call["function"].get("arguments")),
                    },
                }
            )
        body = dict(self.binding)
        body.update(
            {
                "call_id": tag,
                "idempotency_key": tag,
                "turn": turn,
                "calls": packed,
            }
        )
        result = _HTTP.post("/environment/call", json_data=body)
        if not result:
            raise RuntimeError("env round-trip returned nothing")
        return result

    def first_moves(self) -> list[dict[str, Any]]:
        names = self.catalog.names
        calls: list[dict[str, Any]] = []
        if self.ask.named and "view" in names:
            for pid, sku in self.ask.named:
                if len(calls) >= self.max_calls:
                    break
                args = {"product_id": pid}
                if sku:
                    args["sku"] = sku
                calls.append(self._call(f"boot-view-{len(calls)}", "view", args))
            if calls:
                return calls[: self.max_calls]
        if "filter" in names:
            args = self.catalog.scout_fix("filter", {}, self.ask, 0, self.rack.tried, 0)
            calls.append(self._call("boot-filter", "filter", args))
        search_name = "search" if "search" in names else "find_product" if "find_product" in names else None
        if search_name:
            q0 = self.ask.next_q(0, self.rack.tried)
            args = self.catalog.scout_fix(search_name, {"query": q0}, self.ask, 0, self.rack.tried, 0)
            calls.append(self._call("boot-search", search_name, args))
            if self.max_calls > len(calls) and len(self.ask.rungs) > 1:
                q1 = self.ask.next_q(1, {q0.lower()} | self.rack.tried)
                if q1.lower() != q0.lower():
                    args2 = self.catalog.scout_fix(
                        search_name, {"query": q1}, self.ask, 1, {q0.lower()} | self.rack.tried, 0
                    )
                    calls.append(self._call("boot-search-2", search_name, args2))
        return calls[: self.max_calls]

    def seal_now(self) -> list[dict[str, Any]] | None:
        names = self.catalog.names
        candidates: list[dict[str, Any]] = []
        top = self.rack.winner()
        if top:
            candidates.append(top)
        for row in self.rack.slate(8):
            if row and row.get("product_id"):
                candidates.append(row)
        if not candidates and self.ask.named:
            pid0, sku0 = self.ask.named[0]
            candidates.append({"product_id": pid0, "sku": sku0})
        pick = None
        for row in candidates:
            pid_t = str(row.get("product_id") or "")
            sku_t = str(row.get("sku") or "")
            if not pid_t:
                continue
            live = self.rack.stock.get(f"{pid_t}::{sku_t}") or self.rack.stock.get(pid_t) or {}
            if live.get("in_stock") is False or row.get("in_stock") is False:
                continue
            pick = row
            break
        if not pick or not pick.get("product_id"):
            return None
        pid = str(pick["product_id"])
        sku = str(pick.get("sku") or "")
        calls: list[dict[str, Any]] = []

        def emit(name: str, args: dict[str, Any]) -> None:
            if name not in names:
                return
            calls.append(self._call(f"force-{name}-{len(calls)}", name, args))

        if "view" in names and (
            pid not in self.rack.seen
            or (self.rack.fresh_view and self.rack._fresh_pid in (None, pid))
        ):
            if self.rack.fresh_view:
                self.rack._fresh_pid = pid
            emit("view", {"product_id": pid, **({"sku": sku} if sku else {})})
            return calls
        if "inspect_stock" in names and f"{pid}::{sku}" not in self.rack.stock and pid not in self.rack.stock:
            emit("inspect_stock", {"product_id": pid, **({"sku": sku} if sku else {})})
            return calls
        if pid not in self.rack.held and "add_to_cart" in names:
            emit("add_to_cart", {"product_id": pid, **({"sku": sku} if sku else {})})
            return calls
        if self.close in names:
            product = self.rack.products.get(pid) or {"product_id": pid, "sku": sku}
            live = self.rack.stock.get(f"{pid}::{sku}") or self.rack.stock.get(pid) or {}
            args = Ticket(self.catalog, self.close, self.ask, product, live).fill(
                {"product_id": pid, **({"sku": sku} if sku else {})}
            )
            emit(self.close, args)
            return calls
        return calls or None

    def scout_again(self) -> list[dict[str, Any]] | None:
        names = self.catalog.names
        tool = "search" if "search" in names else "find_product" if "find_product" in names else None
        if not tool:
            if "filter" in names:
                args = self.catalog.scout_fix(
                    "filter", {}, self.ask, self.rack.rung, self.rack.tried, self.rack.relax
                )
                return [self._call("fallback-filter", "filter", args)]
            return None
        args = self.catalog.scout_fix(
            tool,
            {"query": self.ask.next_q(self.rack.rung, self.rack.tried)},
            self.ask,
            self.rack.rung,
            self.rack.tried,
            self.rack.relax,
        )
        return [self._call("fallback-search", tool, args)]

    def play(self) -> list[dict[str, Any]]:
        turn = 0
        while turn < self.max_steps:
            turn += 1
            elapsed = self.age
            assistant_content = ""
            if elapsed >= _CLOCK and self.rack.orders == 0:
                self.pending = []
                forced = self.seal_now()
                if not forced:
                    break
                tool_calls = forced
            else:
                left = self.max_steps - turn + 1
                models = self.catalog.models(left, elapsed)
                mood = self.mood(turn)
                if self.ask.named and self.rack.seen and not self.rack.held:
                    mood = Mood.TAKE
                status = self.briefing(turn, mood)
                if self.pending and turn == 1:
                    tool_calls = self.pending
                    self.pending = []
                else:
                    if (
                        len(self.messages) >= 3
                        and self.messages[-1].get("role") == "user"
                        and str(self.messages[-1].get("content") or "").startswith("Briefing:")
                    ):
                        self.messages[-1] = {"role": "user", "content": status}
                    else:
                        self.messages.append({"role": "user", "content": status})
                    tool_calls = []
                    attempts = 2 if elapsed >= _CLOCK * 0.75 else 3
                    attempt = 0
                    while attempt < attempts:
                        if self.age >= _CLOCK:
                            break
                        assistant = self.think(models)
                        assistant_content = assistant.get("content") or ""
                        raw = list(assistant.get("tool_calls") or [])
                        tool_calls = self.admit(raw, mood)
                        if tool_calls:
                            break
                        hint = (
                            "No valid tool call survived checks. After cart changes, re-check stock/price "
                            "on a fresh turn; answer shopper questions with message; order only after verify."
                            if raw
                            else "Emit a search, filter, or product-inspection tool call now."
                        )
                        self.messages.extend(
                            [
                                {"role": "assistant", "content": assistant_content or "..."},
                                {"role": "user", "content": hint},
                            ]
                        )
                        self.trace.append({"role": "assistant", "content": assistant_content})
                        if attempt == attempts - 1:
                            tool_calls = (
                                self.seal_now()
                                if left <= 5 or self.rack.held or self.ask.named
                                else self.scout_again()
                            ) or []
                            if not tool_calls:
                                tool_calls = self.seal_now() or []
                                if not tool_calls:
                                    return self.trace
                        attempt += 1
                    if not tool_calls:
                        tool_calls = self.seal_now() or []
                        if not tool_calls:
                            break
            if self.max_steps - turn <= 3 and self.rack.orders == 0:
                forced = self.seal_now()
                if forced:
                    tool_calls = forced[: self.max_calls]
            for call in tool_calls:
                self.rack.mark_call(
                    call["function"]["name"],
                    self.catalog.decode(call["function"].get("arguments")),
                )
            self.messages.append(
                {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
            )
            result = self.env_turn(turn, tool_calls)
            self.trace.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                    "environment_result": result,
                }
            )
            done = False
            room = 7500 if elapsed < _CLOCK * 0.6 else 4500
            env_calls = list(result.get("calls") or [])
            for idx, tool_call in enumerate(tool_calls):
                if idx >= len(env_calls):
                    break
                call_result = env_calls[idx]
                observation = call_result.get("observation") if isinstance(call_result, dict) else None
                name = tool_call["function"]["name"]
                args = self.catalog.decode(tool_call["function"].get("arguments"))
                self.rack.hear(name, args, observation if isinstance(observation, dict) else {})
                if isinstance(observation, dict) and observation.get("done") is True:
                    done = True
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": self.catalog.pack(observation, room),
                    }
                )
            user_message = result.get("user_message")
            if isinstance(user_message, dict) and user_message.get("content"):
                content = str(user_message["content"])
                self.rack.listen(content)
                self.messages.append({"role": "user", "content": content})
            names = {c["function"]["name"] for c in tool_calls}
            if names & set(Catalog.BASKET):
                self.rack.need_check = True
            elif names & (set(Catalog.LOOK) - {"message"}):
                self.rack.need_check = False
                if self.rack.shifted and self.rack.held:
                    pid = next(iter(self.rack.held))
                    stock = self.rack.stock.get(pid) or {}
                    if stock.get("in_stock") is True:
                        price = Catalog.as_float(stock.get("price") or stock.get("current_price"))
                        if self.ask.budget is None or price is None or price <= self.ask.budget + 1e-6:
                            self.rack.shifted = False
            if done:
                break
        return self.trace



    # --- dormant ShopRun helpers (unwired) -------------------------------------

    def _dorm_turn_budget(self, turn: int) -> dict[str, Any]:
        left = self.max_steps - turn + 1
        return {
            "turn": turn,
            "left": left,
            "age": round(self.age, 2),
            "clock": _CLOCK,
            "max_calls": self.max_calls,
        }

    def _dorm_tool_census(self) -> dict[str, list[str]]:
        return {
            "scout": self.catalog._dorm_scout_names(),
            "basket": self.catalog._dorm_basket_names(),
            "close": self.catalog._dorm_close_names(),
            "look": [n for n in Catalog.LOOK if n in self.catalog.names],
        }

    def _dorm_intent_script(self, mood: int) -> list[str]:
        label = Mood._dorm_label(mood)
        return _dormant_tool_script(self.catalog.names, label)

    def _dorm_message_roles(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.messages:
            role = str(row.get("role") or "?")
            counts[role] = counts.get(role, 0) + 1
        return counts

    def _dorm_trace_len(self) -> int:
        return len(self.trace)

    def _dorm_ask_banner(self) -> str:
        return self.ask.summary() if hasattr(self.ask, "summary") else self.ask.body[:120]

    def _dorm_model_chain(self, turn: int) -> list[str]:
        left = self.max_steps - turn + 1
        return self.catalog.models(left, self.age)

    def _dorm_pending_names(self) -> list[str]:
        names: list[str] = []
        for call in self.pending:
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = fn.get("name")
            if name:
                names.append(str(name))
        return names

    def _dorm_ring_snapshot(self) -> _DormantRing:
        ring = _DormantRing(24)
        for row in self.messages[-10:]:
            ring.push(row.get("content"))
        return ring

    def _dorm_cue_board(self) -> _DormantCueBoard:
        board = _DormantCueBoard()
        for tok, n in self.ask._dorm_phrase_cloud().items():
            board.bump(tok, float(n))
        if self.ask.maker:
            board.bump(self.ask.maker, 3.0)
        return board

    def _dorm_margin_rows(self) -> list[dict[str, Any]]:
        rows = list(self.rack.products.values())
        return _dormant_margin_table(rows, self.ask.budget)

    def _dorm_contrasts(self) -> list[dict[str, tuple[Any, Any]]]:
        ids = list(self.rack.products.keys())
        out: list[dict[str, tuple[Any, Any]]] = []
        if len(ids) >= 2:
            a = self.rack.products[ids[0]]
            b = self.rack.products[ids[1]]
            out.append(_dormant_contrast(a, b))
        return out

    def _dorm_seal_preview(self) -> dict[str, Any] | None:
        forced = self.seal_now()
        if not forced:
            return None
        call = forced[0]
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        return {"name": fn.get("name"), "arguments": fn.get("arguments")}

    def _dorm_briefing_len(self, turn: int, mood: int) -> int:
        return len(self.briefing(turn, mood))

    def _dorm_host_line(self) -> str:
        return self.catalog._dorm_host_banner()

    def _dorm_lattice(self) -> list[tuple[str, ...]]:
        return _dormant_token_lattice(self.query, width=3)

    def _dorm_unit_pairs(self) -> list[tuple[str, str]]:
        return _dormant_unit_pairs(self.query)

    def _dorm_path_leafs(self) -> list[str]:
        leafs: list[str] = []
        for row in self.rack.products.values():
            leaf = _dormant_path_leaf(row.get("category_path"))
            if leaf:
                leafs.append(leaf)
        return leafs

    def _dorm_rank_merge_ids(self) -> list[str]:
        primary = list(self.rack.seen)[:8]
        secondary = list(self.rack.products.keys())[:8]
        return _dormant_rank_merge(primary, secondary, cap=10)

    def _dorm_schema_types(self) -> list[tuple[str, str]]:
        params = self.catalog.params(self.close)
        return _dormant_schema_walk(params)[:40]

    def _dorm_flags(self) -> dict[str, bool]:
        return {
            "has_pending": bool(self.pending),
            "past_clock": self.age >= _CLOCK,
            "named_ask": bool(self.ask.named),
            "held": bool(self.rack.held),
        }


    def _dorm_problem_tag(self) -> str:
        return self.problem_id

    def _dorm_query_clip(self, n: int = 80) -> str:
        return self.query[:n]

    def _dorm_close_name(self) -> str:
        return self.close

    def _dorm_elapsed_frac(self) -> float:
        return min(1.0, self.age / _CLOCK) if _CLOCK else 1.0

    def _dorm_steps_left(self, turn: int) -> int:
        return max(0, self.max_steps - turn + 1)

    def _dorm_mood_name(self, turn: int) -> str:
        return Mood._dorm_label(self.mood(turn))

    def _dorm_doctrine_len(self) -> int:
        return len(self.doctrine())

    def _dorm_shorten_len(self) -> int:
        return len(self.shorten())

    def _dorm_binding_keys(self) -> list[str]:
        return sorted(str(k) for k in self.binding.keys())

    def _dorm_policy_cap(self) -> tuple[int, int]:
        return self.max_steps, self.max_calls

    def _dorm_trace_roles(self) -> list[str]:
        return [str(row.get("role") or "") for row in self.trace]

    def _dorm_last_user(self) -> str:
        for row in reversed(self.messages):
            if row.get("role") == "user":
                return str(row.get("content") or "")[:200]
        return ""

    def _dorm_census_flat(self) -> list[str]:
        flat: list[str] = []
        for group, names in self._dorm_tool_census().items():
            for n in names:
                flat.append(f"{group}:{n}")
        return flat

    def _dorm_ask_flags(self) -> dict[str, bool]:
        return self.ask._dorm_soft_flags()

    def _dorm_rack_snapshot(self) -> dict[str, Any]:
        return self.rack._dorm_snapshot()

    def _dorm_ticket_space(self) -> tuple[int, int]:
        allowed, soft, _desc = Ticket(self.catalog, self.close, self.ask, {}, {}).space()
        return len(allowed), len(soft)

    def _dorm_digest(self) -> str:
        parts = [self.problem_id, self.close, str(len(self.catalog.names)), str(len(self.ask.rungs))]
        return _dormant_echo_digest(parts, salt="run")

    def _dorm_state(self) -> dict[str, Any]:
        return {
            "flags": self._dorm_flags(),
            "census": self._dorm_tool_census(),
            "ages": round(self.age, 3),
            "digest": self._dorm_digest(),
        }

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    return ShopRun(problem_data).play()


__all__ = ["agent_main"]
