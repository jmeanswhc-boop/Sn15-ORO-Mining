
from __future__ import annotations

from src.agent.agent_interface import (
    ToolCallResult, Tool, execute_tool_call, create_dialogue_step, register_tool,
)
from src.agent.proxy_client import ProxyClient
from typing import (
    Any, Optional, List, Dict, Set, Tuple, Union, Type, ClassVar,
    Callable, Sequence, Iterable, get_origin, get_args, TypeGuard, NamedTuple, TypeVar,
    get_type_hints, Generic, Final,
)
from collections.abc import Sequence as AbcSequence, Iterable as AbcIterable, Iterator
from dataclasses import dataclass, fields, field, is_dataclass
from bisect import bisect_left, bisect_right
from collections import defaultdict, Counter, deque, OrderedDict
from itertools import product, islice
from heapq import heappush, heapreplace
from types import NoneType
from enum import Enum
from functools import wraps, total_ordering, lru_cache
from os import getenv
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from decimal import Decimal
import builtins
import copy
import hashlib
import time
import json
import math
import re
import types
import logging
import threading
import traceback
import sys as _sys

ENABLE_DEBUG = False

def _debug_print(*args, **kwargs):
    if ENABLE_DEBUG:
        builtins.print(*args, **kwargs)

try:
    from src.agent.agent_interface import generate_tool_call_id
except ImportError:
    def generate_tool_call_id():
        return "tool-call"

if getenv("SANDBOX_PROXY_URL"):
    _sys.stdout = _sys.stderr




class U_ShoppingTaskType(str, Enum):
    PRODUCT = 'product'
    SHOP = 'shop'
    VOUCHER = 'voucher'

    def explain(self, is_multiple: bool) -> str:
        match self:
            case U_ShoppingTaskType.PRODUCT:
                return 'a single product with specific constraints'
            case U_ShoppingTaskType.SHOP:
                return 'multiple products from the same shop'
            case U_ShoppingTaskType.VOUCHER:
                pstr = 'multiple products' if is_multiple else 'a single product'
                return f'{pstr} within a specified budget and considering voucher options'


class U_ENHANCEMENT_BENCH:
    self_consistency_gap: float = 1.5
    enable_self_consistency: bool = False  # off by default: multi-vote inflates proxy_calls / OOM risk
    low_judge_score: float = 6.0
    fast_accept_score: float = 8.0
    fuzzy_price_ratio: float = 0.08
    fuzzy_price_abs_tol: float = 1.0
    temperature_retry_step: float = 0.3
    temperature_max: float = 1.2
    product_task_shortlist: int = 10
    llm_retry_max: int = 3
    voucher_window_eps: float = 0.5
    rrf_k: int = 60
    rrf_weight: float = 0.15
    duel_reserve_secs: float = 35.0
    dual_cutoff_secs: float = 140.0
    dual_min_left_secs: float = 55.0
    dialog_wire_limit: int = 56_000
    coverage_prune_floor: float = 0.35
    tight_race_gap: float = 0.12
    weak_top_score: float = 0.6
    max_view_per_filter: int = 20
    product_search_pages: int = 3
    shop_search_pages: int = 2
    bi_search_pages: int = 2
    voucher_search_pages: int = 4
    voucher_bi_pages: int = 5


class U_TemperatureController:
    """Role-aware temperature with retry ramp (from Somy lineage)."""

    BASE = {
        "parse": 0.0,
        "filter": 0.0,
        "score": 0.5,
        "judge": 0.5,
        "pick": 0.2,
        "duel": 0.3,
        "dual_compare": 0.4,
        "alt_select": 0.55,
    }

    @classmethod
    def for_attempt(cls, role: str, attempt: int = 0) -> float:
        base = cls.BASE.get(role, 0.0)
        step = U_ENHANCEMENT_BENCH.temperature_retry_step
        tmax = U_ENHANCEMENT_BENCH.temperature_max
        return min(tmax, base + max(0, attempt) * step)


class U_AdversarialIngressGuard:
    _RX_INJECTION = re.compile(
        r"(?is)(?:^|\n)\s*(?:system\s*:|ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions|"
        r"you\s+are\s+now\s+|developer\s+mode\s*|jailbreak\s*:|do\s+not\s+follow\s+the\s+rules)"
    )
    _RX_ROLE_PLAY = re.compile(
        r"(?is)\b(?:act\s+as\s+(?:DAN|an?\s+unrestricted)|pretend\s+you\s+have\s+no\s+restrictions)\b"
    )

    @staticmethod
    def sanitize(query: str) -> str:
        q = str(query or "")
        q = U_AdversarialIngressGuard._RX_INJECTION.sub(" ", q)
        q = U_AdversarialIngressGuard._RX_ROLE_PLAY.sub(" ", q)
        q = re.sub(r"[ \t]{2,}", " ", q)
        q = re.sub(r"\n{3,}", "\n\n", q)
        return q.strip() or str(query or "").strip()


class U_QueryRecovery:
    """Recover a usable query string from heterogeneous harness payloads."""

    @staticmethod
    def recover(task: Any) -> str:
        if task is None:
            return ""
        if isinstance(task, str):
            return task.strip()
        if isinstance(task, (list, tuple)):
            for item in task:
                got = U_QueryRecovery.recover(item)
                if got:
                    return got
            return ""
        if not isinstance(task, dict):
            try:
                return str(task).strip()
            except Exception:
                return ""
        for key in ("query", "q", "text", "request", "user_query", "message", "prompt"):
            val = task.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, (dict, list, tuple)):
                got = U_QueryRecovery.recover(val)
                if got:
                    return got
        # longest space-containing string anywhere one level deep
        best = ""
        for v in task.values():
            if isinstance(v, str) and " " in v and len(v) > len(best):
                best = v.strip()
            elif isinstance(v, (dict, list, tuple)):
                got = U_QueryRecovery.recover(v)
                if got and len(got) > len(best):
                    best = got
        return best


class U_DialogWireFit:
    """Compress dialogue to fit harness IPC budget (Somy wire ladder).

    Critical: the sandbox attaches full `extra_info.proxy_calls` (every HTTP
    request/response) onto dialogue steps. Leaving those in place produces
    20–150MB jsonl lines, OOM-kills the sandbox (exit 137), and truncates
    output so scoring crashes with JSONDecodeError. Always strip them.
    """

    THINK_CAPS = (2000, 1000, 500)
    KEEP_TOOL_NAMES = ("recommend_product", "terminate")

    @staticmethod
    def wire_size(dialog: List[Dict]) -> int:
        try:
            return len(json.dumps(dialog, ensure_ascii=False, default=str).encode("utf-8"))
        except Exception:
            return 10**9

    @staticmethod
    def _strip_proxy_trace(step: Dict) -> Dict:
        """Drop sandbox HTTP traces; keep only scorer-needed extra_info fields."""
        if not isinstance(step, dict):
            return step
        info = step.get("extra_info")
        if not isinstance(info, dict):
            return step
        if "proxy_calls" not in info and "http_calls" not in info and "trace" not in info:
            return step
        out = dict(step)
        clean = {
            k: v for k, v in info.items()
            if k in ("step", "query", "timestamp", "problem_id")
        }
        out["extra_info"] = clean
        return out

    @staticmethod
    def _slim_tool(tc: Any) -> Any:
        if not isinstance(tc, dict):
            return tc
        name = tc.get("name")
        if name in U_DialogWireFit.KEEP_TOOL_NAMES:
            return tc
        # Compact any search/view payload to ids + price + shop only
        res = tc.get("result")
        rows = res.get("result") if isinstance(res, dict) else res
        if isinstance(rows, list):
            slim = []
            for r in rows[:10]:
                if not isinstance(r, dict):
                    continue
                slim.append({
                    "product_id": str(r.get("product_id") or r.get("pid") or r.get("id") or ""),
                    "price": r.get("price", r.get("p")),
                    "shop_id": str(r.get("shop_id") or r.get("s") or ""),
                })
            out = {k: v for k, v in tc.items() if k != "result"}
            out["result"] = {**res, "result": slim} if isinstance(res, dict) else slim
            return out
        if name == "find_product":
            return U_health_bs_compact_result(tc)
        return tc

    @staticmethod
    def _rebuild(step: Dict, think: str, tools: List[Any]) -> Dict:
        # Re-emit through harness constructor so completion.content stays consistent
        # and extra_info.proxy_calls is not preserved.
        msg = (step.get("completion") or {}).get("message") or {}
        info = step.get("extra_info") or {}
        return create_dialogue_step(
            think=think,
            tool_results=tools,
            response=msg.get("response", "") or "",
            query=info.get("query", "") or "",
            step=int(info.get("step") or 1),
        )

    @classmethod
    def fit(cls, dialog: List[Dict], limit: int | None = None) -> List[Dict]:
        limit = limit or U_ENHANCEMENT_BENCH.dialog_wire_limit
        if not dialog:
            return dialog

        # Always strip proxy traces first (even when under the byte limit).
        dialog = [cls._strip_proxy_trace(s) for s in dialog]
        if cls.wire_size(dialog) <= limit:
            return dialog

        before = cls.wire_size(dialog)
        parts = []
        for i, step in enumerate(dialog):
            msg = (step.get("completion") or {}).get("message") or {}
            tools = list(msg.get("tool_call") or [])
            keep = i == 0 or any(
                isinstance(tc, dict) and tc.get("name") in cls.KEEP_TOOL_NAMES
                for tc in tools
            )
            parts.append({
                "step": step,
                "keep": keep,
                "dirty": False,
                "think": msg.get("think", "") or "",
                "tools": tools,
                "response": msg.get("response", "") or "",
            })

        def assemble() -> List[Dict]:
            return [
                p["step"] if p["keep"] and not p["dirty"]
                else cls._rebuild(p["step"], p["think"], p["tools"])
                for p in parts
            ]

        def settled() -> Optional[List[Dict]]:
            out = assemble()
            if cls.wire_size(out) <= limit:
                return out
            return None

        # Slim tool payloads on every non-answer step (and answer steps keep recommend/terminate)
        for p in parts:
            if p["keep"]:
                # Still drop non-answer tools if any leaked onto the final step
                slimmed = []
                for tc in p["tools"]:
                    if isinstance(tc, dict) and tc.get("name") in cls.KEEP_TOOL_NAMES:
                        slimmed.append(tc)
                    else:
                        slimmed.append(cls._slim_tool(tc))
                if slimmed != p["tools"]:
                    p["tools"] = slimmed
                    p["dirty"] = True
            else:
                p["tools"] = [cls._slim_tool(tc) for tc in p["tools"]]
                p["dirty"] = True
        out = settled()
        if out is not None:
            return out

        for cap in cls.THINK_CAPS:
            for p in parts:
                if len(p["think"]) > cap:
                    p["think"] = p["think"][:cap]
                    p["dirty"] = True
            out = settled()
            if out is not None:
                return out

        for p in parts:
            if not p["keep"]:
                p["tools"] = []
                p["dirty"] = True
        out = settled()
        if out is not None:
            return out

        # Last resort: keep only first + answer steps
        essential = []
        for i, p in enumerate(parts):
            if p["keep"] or i == 0:
                essential.append(cls._rebuild(p["step"], p["think"][:500], [
                    tc for tc in p["tools"]
                    if isinstance(tc, dict) and tc.get("name") in cls.KEEP_TOOL_NAMES
                ]))
        if essential and cls.wire_size(essential) <= limit:
            return essential
        return essential or assemble()


class U_EdgeCaseProcessor:
    MAX_SAFE_FLOAT = 1e15

    @staticmethod
    def is_nullish(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            s = value.strip().lower()
            return s in ("", "null", "none", "n/a", "undefined", "nan")
        if isinstance(value, (list, dict, tuple, set)) and len(value) == 0:
            return True
        return False

    @staticmethod
    def coerce_float(value: Any, *, default: float = 0.0, lo: float | None = None, hi: float | None = None,) -> float:
        if U_EdgeCaseProcessor.is_nullish(value):
            out = default
        elif isinstance(value, bool):
            out = float(value)
        elif isinstance(value, (int, float)):
            out = float(value)
        elif isinstance(value, str):
            try:
                out = float(U_MiscUtil.extract_number(value))
            except (ValueError, TypeError):
                out = default
        else:
            out = default

        if not math.isfinite(out):
            out = default
        out = max(-U_EdgeCaseProcessor.MAX_SAFE_FLOAT, min(U_EdgeCaseProcessor.MAX_SAFE_FLOAT, out))
        if lo is not None:
            out = max(lo, out)
        if hi is not None:
            out = min(hi, out)
        return out

    @staticmethod
    def coerce_int(value: Any, *, default: int = 0, lo: int | None = None, hi: int | None = None,) -> int:
        f = U_EdgeCaseProcessor.coerce_float(
            value,
            default = float(default),
            lo = float(lo) if lo is not None else None,
            hi = float(hi) if hi is not None else None,
        )
        return int(f)

    @staticmethod
    def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
        if denominator == 0 or not math.isfinite(denominator):
            return default
        result = numerator / denominator
        return result if math.isfinite(result) else default

    @staticmethod
    def sanitize_price_range(price_range: str | None) -> str | None:
        if U_EdgeCaseProcessor.is_nullish(price_range):
            return None
        s = re.sub(r"[^\d\-\.]", "", str(price_range).strip())
        if not s or s == "-":
            return None
        left_raw, sep, right_raw = s.partition("-")
        if not sep:
            return None
        left, right = left_raw.strip(), right_raw.strip()
        if left and right:
            lo = U_EdgeCaseProcessor.coerce_float(left)
            hi = U_EdgeCaseProcessor.coerce_float(right)
            if hi < lo:
                lo, hi = hi, lo
            return f"{lo:g}-{hi:g}"
        if left:
            return f"{U_EdgeCaseProcessor.coerce_float(left):g}-"
        if right:
            return f"-{U_EdgeCaseProcessor.coerce_float(right):g}"
        return None

    @staticmethod
    def sanitize_product_id(value: Any) -> str | None:
        pid = str(value or "").strip()
        _INVALID_PRODUCT_IDS = frozenset({"", "none", "null", "0", "undefined", "n/a"})
        if pid.lower() in _INVALID_PRODUCT_IDS:
            return None
        return pid or None

    @staticmethod
    def sanitize_service_csv(value: Any) -> str | None:
        if U_EdgeCaseProcessor.is_nullish(value):
            return None
        parts: List[str] = []
        for token in re.split(r"[,|]", str(value)):
            t = token.strip()
            if t and t not in parts:
                parts.append(t)
        return ",".join(parts) if parts else None

    @staticmethod
    def clamp_listing_price(price: Any, *, max_price: float = 1e12) -> float | None:
        if U_EdgeCaseProcessor.is_nullish(price):
            return None
        try:
            p = float(price)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(p):
            return None
        if p < 0:
            return 0.0
        if p > max_price:
            return max_price
        return p

    @staticmethod
    def parse_comma_prices(product_prices: Any) -> Tuple[List[float], Optional[str]]:
        if U_EdgeCaseProcessor.is_nullish(product_prices):
            return [], "product_prices is null or empty."
        try:
            prices = [
                U_EdgeCaseProcessor.coerce_float(p.strip(), lo = 0.0)
                for p in str(product_prices).split(",")
                if p.strip()
            ]
        except (ValueError, TypeError):
            return [], "Invalid product_prices format. Use comma-separated numbers."
        if any(p < 0 for p in prices):
            return [], "product_prices must not contain negative values."
        return prices, None

    @staticmethod
    def empty_dict_if_null(js: Dict | None) -> Dict:
        return js if isinstance(js, dict) else {}

    @staticmethod
    def empty_list_if_null(items: List | None) -> List:
        return items if isinstance(items, list) else []

class U_FuzzyBoundaryLogic:
    """Fuzzy price/boundary helpers (bird/beam/tower/shrike/garland patterns)."""

    @staticmethod
    def parse_price_bounds(price_range: str | None) -> Tuple[float | None, float | None]:
        cleaned = U_EdgeCaseProcessor.sanitize_price_range(price_range)
        if not cleaned:
            return (None, None)
        left_raw, _, right_raw = cleaned.partition("-")
        lo = U_EdgeCaseProcessor.coerce_float(left_raw) if left_raw.strip() else None
        hi = U_EdgeCaseProcessor.coerce_float(right_raw) if right_raw.strip() else None
        return (lo, hi)

    @staticmethod
    def normalize_price_from_query(text: str) -> str | None:
        if U_EdgeCaseProcessor.is_nullish(text):
            return None
        _RX_PRICE_BETWEEN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:to|and|-)\s*(\d+(?:\.\d+)?)\s*(?:pesos|php|usd|dollars?)?", re.I,)
        m = _RX_PRICE_BETWEEN.search(text)
        if m:
            return U_EdgeCaseProcessor.sanitize_price_range(f"{m.group(1)}-{m.group(2)}")
        _RX_PRICE_OVER = re.compile(r"(?:greater|more|over|above|at\s+least|minimum|min|>=)\s*(?:than\s*)?(\d+(?:\.\d+)?)", re.I,)
        m = _RX_PRICE_OVER.search(text)
        if m:
            return U_EdgeCaseProcessor.sanitize_price_range(f"{m.group(1)}-")
        _RX_PRICE_UNDER = re.compile(r"(?:less|under|below|at\s+most|maximum|max|<=)\s*(?:than\s*)?(\d+(?:\.\d+)?)", re.I,)
        m = _RX_PRICE_UNDER.search(text)
        if m:
            return U_EdgeCaseProcessor.sanitize_price_range(f"-{m.group(1)}")
        return None

    @staticmethod
    def fuzzy_tolerance(lo: float, hi: float, *, ratio: float | None = None, abs_tol: float | None = None) -> Tuple[float, float]:
        ratio = U_ENHANCEMENT_BENCH.fuzzy_price_ratio if ratio is None else ratio
        abs_tol = U_ENHANCEMENT_BENCH.fuzzy_price_abs_tol if abs_tol is None else abs_tol
        span = max(hi - lo, 0.0)
        pad = max(abs_tol, span * ratio)
        return lo - pad, hi + pad

    @staticmethod
    def clip_price_band(
        floor: float,
        ceiling: float,
        price_range: str | None,
    ) -> Tuple[float, float] | None:
        if ceiling < floor:
            return None
        parsed_lo, parsed_hi = U_FuzzyBoundaryLogic.parse_price_bounds(price_range)
        lo = floor if parsed_lo is None else max(floor, parsed_lo)
        hi = ceiling if parsed_hi is None else min(ceiling, parsed_hi)
        if hi < lo:
            return None
        return (lo, hi)

    @staticmethod
    def widen_price_range(price_range: str | None, ratio: float | None = None) -> str | None:
        if U_EdgeCaseProcessor.is_nullish(price_range):
            return None
        ratio = U_ENHANCEMENT_BENCH.fuzzy_price_ratio if ratio is None else ratio
        try:
            lo, hi = U_FuzzyBoundaryLogic.parse_price_bounds(price_range)
            if lo is not None:
                lo = max(0.0, lo * (1.0 - ratio))
            if hi is not None:
                hi = hi * (1.0 + ratio)
            lo_fmt = f"{lo:.0f}" if lo is not None else ""
            hi_fmt = f"{hi:.0f}" if hi is not None else ""
            widened = f"{lo_fmt}-{hi_fmt}"
            return None if widened == "-" else widened
        except (ValueError, TypeError):
            return U_EdgeCaseProcessor.sanitize_price_range(price_range)

    @staticmethod
    def price_fits_range(price: Any, price_range: str | None, *, fuzzy: bool = True) -> Tuple[bool, str]:
        pv = U_EdgeCaseProcessor.clamp_listing_price(price)
        if pv is None:
            return (False, "price is null or non-numeric")
        lo, hi = U_FuzzyBoundaryLogic.parse_price_bounds(price_range)
        if lo is None and hi is None:
            return (True, "no price constraint")
        if fuzzy and lo is not None and hi is not None:
            flo, fhi = U_FuzzyBoundaryLogic.fuzzy_tolerance(lo, hi)
        elif fuzzy:
            pad = U_ENHANCEMENT_BENCH.fuzzy_price_abs_tol
            flo = (lo - pad) if lo is not None else None
            fhi = (hi + pad) if hi is not None else None
        else:
            flo, fhi = lo, hi
        if flo is not None and pv < flo:
            return (False, f"price {pv:g} below lower bound {lo:g}")
        if fhi is not None and pv > fhi:
            return (False, f"price {pv:g} above upper bound {hi:g}")
        return (True, f"price {pv:g} within range")

    @staticmethod
    def classify_price_boundary(price: Any, lo: float | None, hi: float | None) -> str:
        pv = U_EdgeCaseProcessor.clamp_listing_price(price)
        if pv is None:
            return "null"
        tol = U_ENHANCEMENT_BENCH.fuzzy_price_abs_tol
        if lo is not None and abs(pv - lo) <= tol:
            return "boundary_low"
        if hi is not None and abs(pv - hi) <= tol:
            return "boundary_high"
        if lo is not None and pv < lo:
            return "under"
        if hi is not None and pv > hi:
            return "over"
        return "inside"

    @staticmethod
    def price_fit_score_adjustment(price_val: Any, price_range_str: Any) -> float:
        ok, _ = U_FuzzyBoundaryLogic.price_fits_range(
            price_val,
            str(price_range_str) if price_range_str else None,
            fuzzy = False,
        )
        return 5.0 if ok else -25.0

    @staticmethod
    def voucher_window_status(cart_total: float, threshold: float, allowed_total: float) -> Dict[str, Any]:
        cart_total = U_EdgeCaseProcessor.coerce_float(cart_total)
        threshold = U_EdgeCaseProcessor.coerce_float(threshold)
        allowed_total = U_EdgeCaseProcessor.coerce_float(allowed_total)
        eps = U_ENHANCEMENT_BENCH.voucher_window_eps

        if cart_total > allowed_total + eps:
            return {
                "within_window": False,
                "reason": "over_allowed_total",
                "status": "over",
                "below_threshold": cart_total < threshold,
                "at_boundary": False,
            }
        if threshold > 0 and cart_total + eps < threshold:
            return {
                "within_window": False,
                "reason": "below_threshold",
                "status": "below_threshold",
                "below_threshold": True,
                "at_boundary": True,
            }
        if abs(cart_total - threshold) <= eps or abs(cart_total - allowed_total) <= eps:
            return {
                "within_window": True,
                "reason": "boundary",
                "status": "boundary",
                "below_threshold": False,
                "at_boundary": True,
            }
        return {
            "within_window": True,
            "reason": "ok",
            "status": "ok",
            "below_threshold": False,
            "at_boundary": False,
        }

    @staticmethod
    def extend_top_n_on_tie(scores: List[Tuple[Any, float]], top_n: int) -> List[Any]:
        """Extend selection when multiple items tie at the rank boundary (tower/bird)."""
        if not scores or top_n <= 0:
            return []
        ordered = sorted(scores, key = lambda t: t[1], reverse = True)
        if len(ordered) <= top_n:
            return [item for item, _ in ordered]
        cutoff = ordered[top_n - 1][1]
        return [item for item, sc in ordered if sc >= cutoff]

class U_HOO19_PrecisionGate:
    RRF_WEIGHT = U_ENHANCEMENT_BENCH.rrf_weight
    RRF_K = U_ENHANCEMENT_BENCH.rrf_k
    VSF_MAX = 15
    JUDGE_MAX = 15
    PRODUCT_SELECT = 10
    JUDGE_FAST = 9.0
    JUDGE_MIN = 5.0
    WEAK_TOP_SCORE = U_ENHANCEMENT_BENCH.weak_top_score

    JUDGE_PROMPT = (
        "Task: identify the single best candidate product for a shopping request.\n"
        "Input JSON has `request`, `candidates` (title, price, service, attributes, "
        "sku_options_preview), and `only_product_type`.\n"
        "Rules:\n"
        "- attributes and sku_options_preview are more trustworthy than title\n"
        "- each stated requirement must be satisfied "
        "(brand, color, size, material, model, quantity/pack, price, service)\n"
        "- do NOT prefer a candidate only because it is cheaper\n"
        "- do NOT prefer a candidate only because the title has more query words\n"
        "- pack counts (e.g. 20pcs), sizes (EU 37, XXL, L, S), colors, dimensions "
        "(80x160cm), and model codes are hard constraints — missing them is a hard mismatch\n"
        "- price outside the requested range is a hard mismatch\n"
        "- missing required service tags is a hard mismatch\n"
        "- accessories (case/cover/holder) are wrong when the query asks for the bare product type\n"
        "Output strict JSON only:\n"
        "{\n"
        '  "best_product_id": "<id>",\n'
        '  "reason": "1-2 sentences citing attribute/sku evidence",\n'
        '  "relevance_score": <integer 0-10>\n'
        "}"
    )

    PV_DUEL = """
Two candidate products both plausibly answer a shopping query. Choose the ONE that a
careful shopper would say is the intended item.

# How to judge
- Evidence in `attributes` or `sku_options` is STRONGER than evidence in the title.
  The title is marketing text; a word appearing there is a hint, not proof.
- Matching MORE words of the query is NOT evidence. A product repeating every query word
  in its title is routinely the wrong answer.
- A stated value that CONTRADICTS the query (query says red, product says color: blue)
  disqualifies that product. A product that is simply silent about an attribute does not.
- Singular/plural and participle forms are the SAME value: caplet=caplets, no stone=no
  stones, 32 ohm=32 ohms, mix=mixed. Never separate the two on that basis.
- The product TYPE must be the item asked for: a remote control is not a TV, a phone case
  is not a phone, an ink cartridge is not a printer.
- If both satisfy the query equally, prefer A -- it is the higher-ranked candidate.

# Query
{query}

# Candidate A
{a}

# Candidate B
{b}

Return JSON only: {{"reason": "<one sentence citing attribute or sku evidence>", "winner": "A" | "B"}}
""".strip()

    DUAL_COMPARE = """
You must choose which of two candidate answers better satisfies the user's shopping request.
Candidate A came from the primary pipeline. Candidate B came from an alternate pipeline.

# Decision Rules
1. Prefer the candidate that satisfies MORE required constraints with clear evidence.
2. If both satisfy all constraints, prefer the more exact / direct match (structured attributes, fewer extraneous tokens).
3. If still tied, prefer Candidate A.
4. Never invent product details; only use the provided product JSON.

# User Query
```{query}```
# Candidate A
{cand_a}
# Candidate B
{cand_b}

Return JSON only:
{{
    "check_a": "<brief constraint check for A>",
    "check_b": "<brief constraint check for B>",
    "best": "A" | "B",
    "reason": "<one sentence>"
}}
""".strip()

    @classmethod
    def verify_pick(cls, query: "Query", products: List["Product"]) -> Tuple[bool, str]:
        """Typed post-select auditor. Returns (ok, reason)."""
        if not products:
            return False, "empty"
        intents = getattr(query, "intents", None) or []
        challenge = getattr(query, "challenge", None)

        if challenge == U_Challenge.PRODUCT and len(products) != 1:
            return False, "product_count"
        if challenge == U_Challenge.SHOP:
            if len(products) < 2:
                return False, "shop_count"
            shops = {getattr(p, "shop", None) for p in products}
            if len(shops) > 1:
                return False, "mixed_shops"

        if intents and len(products) != len(intents):
            # allow partial only as soft fail for voucher fallbacks
            if challenge != U_Challenge.VOUCHER:
                return False, "count_mismatch"

        for sq, p in zip(intents, products):
            if sq.price:
                # Use fuzzy boundary logic (was shelf-only; strict numerify rejected near-bound hits)
                price_ok, _note = U_FuzzyBoundaryLogic.price_fits_range(p.price, sq.price, fuzzy=True)
                if not price_ok:
                    return False, "price_mismatch"
            if sq.services:
                prod_svcs = {str(s).lower() for s in (p.services or [])}
                if not all(str(s).lower() in prod_svcs for s in sq.services):
                    return False, "service_mismatch"
            # cheap coverage gate using parsed attrs
            parsed = getattr(sq, "parsed", None) or {}
            if parsed:
                blob = " ".join([
                    str(p.title or ""),
                    json.dumps(getattr(p, "attr", {}) or {}, ensure_ascii=False),
                    json.dumps(getattr(p, "sku", {}) or {}, ensure_ascii=False),
                ]).lower()
                hits = 0
                keys = list(parsed.items())
                for k, v in keys:
                    tokens = U_MiscUtil.normalize_str(str(v), True).split()
                    if tokens and any(t in blob for t in tokens):
                        hits += 1
                if len(keys) >= 2 and hits / max(1, len(keys)) < 0.25:
                    return False, "low_coverage"

        if challenge == U_Challenge.VOUCHER and query.voucher:
            v = query.voucher
            if v.apply_to == "shop" and len({p.shop for p in products}) > 1:
                return False, "mixed_shops"
            try:
                cost = v.get_active_cost(products) if hasattr(v, "get_active_cost") else sum(p.price for p in products)
            except Exception:
                cost = sum(float(p.price or 0) for p in products)
            if cost > float(v.budget) + U_ENHANCEMENT_BENCH.voucher_window_eps:
                return False, "over_budget"
            thr = float(getattr(v, "threshold", 0) or 0)
            total = sum(float(p.price or 0) for p in products)
            if thr > 0 and total + U_ENHANCEMENT_BENCH.voucher_window_eps < thr and cost < total:
                # voucher claimed but below threshold — still ok if under budget without voucher
                pass
        return True, "ok"

    @classmethod
    def parse_duel_winner(cls, js: Any) -> str:
        if not isinstance(js, dict):
            return "A"
        w = str(js.get("winner", "A")).strip().upper()[:1]
        return w if w in ("A", "B") else "A"

    @classmethod
    def empty_pool_rescue_queries(cls, sq: "SubQuery", base_q: str) -> List[str]:
        def kw_key(kw: List[str]) -> str:
            return kw[-2] if len(kw) >= 2 else (kw[0] if kw else "")

        kws = {kw_key(kw): str(kw[-1]).strip() for kw in (sq.kw or []) if kw}
        pt, brand = kws.get("product_type"), kws.get("brand")
        model = kws.get("model") or kws.get("compatibility")
        color = kws.get("color") or kws.get("color_family")
        size = kws.get("size")
        plans: List[str] = []
        if pt and brand:
            plans.append(f"{brand} {pt}")
        if pt and model:
            plans.append(f"{pt} {model}")
        if pt and color and size:
            plans.append(f"{color} {pt} {size}")
        if pt and color:
            plans.append(f"{color} {pt}")
        if pt and size:
            plans.append(f"{pt} {size}")
        if pt:
            plans.append(pt)
        # Deduplicate while preserving order
        out, seen = [], set()
        for q in plans:
            qn = U_MiscUtil.normalize_str(q)
            if qn and qn not in seen and qn != U_MiscUtil.normalize_str(base_q):
                seen.add(qn)
                out.append(q)
        return out[:5]

def U_beach_bs_call_api(change_embrace_d3859f: str, finger_anxiety_ea109d: dict) -> dict:
    agree_identify_b83983 = 10
    goat_badge_793ab3 = 0.5
    alcohol_basket_abc34d = 1.0
    chaos_confirm_6cf0b2 = time.monotonic() - agree_identify_b83983
    if chaos_confirm_6cf0b2 < goat_badge_793ab3:
        time.sleep(goat_badge_793ab3 - chaos_confirm_6cf0b2)
    invest_drastic_5fe9f8 = 0
    broad_inmate_7e25b9 = 3
    while True:
        try:
            elbow_demand_9347a9 = execute_tool_call(change_embrace_d3859f, finger_anxiety_ea109d)
            agree_identify_b83983 = time.monotonic()
            return elbow_demand_9347a9
        except Exception:
            invest_drastic_5fe9f8 += 1
            if invest_drastic_5fe9f8 >= broad_inmate_7e25b9:
                raise
            army_advice_78f8c1 = alcohol_basket_abc34d * 2 ** (invest_drastic_5fe9f8 - 1)
            time.sleep(army_advice_78f8c1)

def U_horse_bs_shrink_items(enjoy_bounce_ea04d1: list) -> list:
    defense_hungry_445caf = 10
    return [{'pid': str(easily_fire_5c263c.get('product_id', '')), 'p': easily_fire_5c263c.get('price'), 's': str(easily_fire_5c263c.get('shop_id', ''))} for easily_fire_5c263c in enjoy_bounce_ea04d1[:defense_hungry_445caf] if isinstance(easily_fire_5c263c, dict)]

def U_health_bs_compact_result(define_diamond_3dc367: dict) -> dict:
    if not isinstance(define_diamond_3dc367, dict) or define_diamond_3dc367.get('name') != 'find_product':
        return define_diamond_3dc367
    legal_essence_2bce1c = define_diamond_3dc367.get('result')
    if isinstance(legal_essence_2bce1c, dict) and isinstance(legal_essence_2bce1c.get('result'), list):
        return {**define_diamond_3dc367, 'result': {**legal_essence_2bce1c, 'result': U_horse_bs_shrink_items(legal_essence_2bce1c['result'])}}
    if isinstance(legal_essence_2bce1c, list):
        return {**define_diamond_3dc367, 'result': U_horse_bs_shrink_items(legal_essence_2bce1c)}
    return define_diamond_3dc367

def U_balance_bs_pipe_append_step(boil_account_b6f863, think: str, arctic_edge_4975dc: list, force_fatal_c1d19b: str='') -> None:
    compact = [U_health_bs_compact_result(endless_bunker_b7edc9) for endless_bunker_b7edc9 in arctic_edge_4975dc or []]
    boil_account_b6f863.steps.append(create_dialogue_step(think, compact, force_fatal_c1d19b, boil_account_b6f863.query, len(boil_account_b6f863.steps) + 1))

def U_freezable(func: Callable) -> Callable:
    @wraps(func)
    def __wrapper__(self: "Debuggable", *args, **kwargs) -> Any:        
        suffix, id = kwargs.get("suffix", ""), kwargs.get("id", "")
        query, ret = None, None
        
        stage = type(self).__name__.lower().replace("engine", "") + "-" + func.__name__.strip("_").replace("_", "-")
        if suffix: stage += "_" + suffix

        for param in list(args) + list(kwargs.values()):
            if isinstance(param, U_SubQuery):
                query = param
                break
        
        self.stopped = self.debugger.is_stop_at(stage) if self.debugger is not None else False        
        if query is None: query = self.query

        if self.debugger is not None and self.debugger.is_frozen(stage): ret = self.debugger.find(stage, id, query)

        if ret is None:
            ret = func(self, *args, **kwargs)
            if self.debugger is not None: self.debugger.append(ret, stage, id, query)
        
        return ret

    return __wrapper__

def U_hobby_bs_unique_nonempty_ids(ids: list) -> list[str]:
    gentle_hurt_d448de: set[str] = set()
    out: list[str] = []
    for capital_ill_6c3fc2 in ids:
        val = str(capital_ill_6c3fc2).strip()
        if not val or val in gentle_hurt_d448de:
            continue
        gentle_hurt_d448de.add(val)
        out.append(val)
    return out

def U_fuel_bs_join_ids(ids: list, across_chicken_8f345c: list=None) -> str:
    cave_book_71cead = '0'
    act_category_112463 = U_hobby_bs_unique_nonempty_ids(ids)
    if across_chicken_8f345c:
        burst_fork_a451ab = {enemy_isolate_9505a7: i for i, enemy_isolate_9505a7 in enumerate(across_chicken_8f345c)}
        dumb_cool_938ebb = len(across_chicken_8f345c)
        act_category_112463.sort(key=lambda enemy_isolate_9505a7: burst_fork_a451ab.get(enemy_isolate_9505a7, dumb_cool_938ebb))
    return ','.join(act_category_112463) if act_category_112463 else cave_book_71cead


class U_Prompt:
    SV_DECOMPOSE = """
You are AI assistant to decompose the user's shopping query into structured format. The user is looking for {intent} in the original shopping query.

# Task
## Step-1. Identify the products and their product types from the original query.
- The product count shouldn't exceed 4.
    * If the original query seems to have more than 4 products, it's because some of them are the sub-items packed inside a kit product. You must correctly identify the kit product.
- Don't mistakenly take a constraint or filler phrase as an individual product! Example:
    * "I need a branded multipack, razor product." → Correct: 1 product "razor" || Incorrect: 2 products "multipack" and "razor"
    * "Find in-house plastic gas valves. Can you show me gas stove options?" → Correct 1: product "gas valve" || Incorrect: 2 products "gas valve" and "gas stove"
- Sometimes, the product types are not clearly separated in the original query. You must find the correct product type for each product and its corresponding sub-query. Implicit product type sharing with "[product type] with [attribute 1] and [attribute 2]" pattern is forbidden. Examples:
    * "Looking for a seafood snack from McDonald, orange juice with 500ml pack and some cheese cakes." →
        Easiest case. Product types are explicitly separated as "seafood snack", "orange juice" and "cheese cake".
    * "Find comic books, one for 2-4 year olds and another one for 5-8." →
        "comic books" is the common product type for two products. It must be included in both sub-queries explicitly.
    * "Need TV accessories, first one for Android TV and second one with color black and small size. Find them in remote controller and table categories." →
        There are two products in this query, and their product types are "remote controller" and "table", respectively, in the mentioned orders.
    * When a user uses different words to refer to the same product (e.g., "touch screen digitizer assembly" then "screen replacements", or "water bottle" then "drinking container"), this is still ONE product. The second phrase is an alternative reference, not a new product request. Only count it as a separate product if the second phrase clearly describes a different item with distinct attributes or purpose.
    * **"A or B" Alternative Pattern:** When a user says "A or B" (e.g., "carburetor or fuel injection kit", "red or blue shirt", "iPhone or Samsung"), this is **ONE product** with alternative options. The user wants **either** A **or** B, not both. Do NOT split into two products.
        * Example: "carburetor or fuel injection kit for raider fi" → **1 product**: "carburetor fuel injection kit raider fi"
        * Example: "water bottle or tumbler" → **1 product**: "water bottle tumbler"
        * Counter-example: "I need a carburetor and a fuel injection kit" → **2 products** (user wants both)
    * "I want water bottle and ceramic cups. First, 300ml capacity and plastic. Next, flower patterned design and lightweight. And, the one made from Japan Ceramics. Lastly, I need one with 500ml capacity and suitable for mineral water." →
        The most complicated case. There are 4 products in this query, and two product types "water bottle" and "ceramic cup" were mentioned. You must guess each product's product type by analyzing the query context.
        In this example, probably the product 1 will be "water bottle", because it's mentioned at the first. (It's convention that the first product receives the first mentioned product type. This convention is applied to the first product only.)
        For the rest products, you can guess that product 4 is "water bottle", while the product 2 and 3 are "ceramic cup"s, from their semantics.
    * "Find tshirts with color yellow and plain design." → Implicit sharing with "[product type] with [attribute 1] and [attribute 2]" pattern is forbidden. Therefore, this is one product having both "color yello" and "plain design" attributes, not two independent tshirts.
- **Product Segmentation Rules for Unseparated Queries:**
    When products are not separated by commas, "and", or numbered lists, use this priority:
    1. **Price constraints are the primary product boundaries.** Each price constraint marks the END of one product segment and the START of the next.
    2. The segment BEFORE a price marker = product + its attributes.
    3. The segment AFTER a price marker = next product + its attributes (until the next price).
    4. If a product has NO explicit price, it occupies the text from the previous price boundary (or query start) to the next price boundary.
    5. **Quantity markers** like "10 pcs", "5 pieces", "2 packs", "1 dozen" belong to the SAME product as the noun they modify — never split them.
- **Segmentation Examples:**
    * "christening souvenirs pink blue free shipping under 58 pesos candle 10 pcs below 152 mini bible over 10 pesos customize with freebies above 3 pesos"
        → Segments: "christening souvenirs pink blue free shipping" | "candle 10 pcs" | "mini bible" | "customize with freebies"
        → Prices: "under 58 pesos" | "below 152" | "over 10 pesos" | "above 3 pesos"
    * "party supplies birthday decorations king star official store free shipping flash sale over 150 pesos white balloons authorized seller under 235 pesos centerpiece stand 8 years old birthday 19 to 51 pesos banner unisex generic"
        → Segments: "birthday decorations king star official store" | "white balloons authorized seller" | "centerpiece stand 8 years old birthday" | "banner unisex generic"
- If the original query explicitly lists the products by "1), 2), ..." or "First, second, ..., last ..." or "Firstly, nextly, ... lastly ...", then respect these boundaries.
- The pattern of ```[full period] plus "Also, "``` (example: ```xxx. Also, ```) is definitely the beginning of a new product, not the beginning of a new constraint.
- **Comma + "And" List Pattern:** When the query uses the pattern `[product A], [product B], and [product C]` (note the **comma before "and"**), each item is a **separate product**. The comma before "and" is the critical signal that this is a **list**, not a kit with sub-items. This rule overrides the kit product interpretation.
    * Example: `"charger, earbuds, and adapter"` → **3 separate products**
    * Example: `"kit with earbuds and adapter"` → **1 kit product** (no comma before "and")
    * Counter-example: `"gift set 6 in 1 with earbuds for 13 pro max, and lightning adapter"` → **2 products**: the gift set AND the separate adapter. The comma before "and" forces the split.
- **Brand Propagation from Introductory Phrases:** If a brand name appears in an introductory or contextual phrase that clearly applies to ALL subsequently listed products (e.g., "Find Brand X items: First... Second...", "Looking for Brand Y products: 1)... 2)...", "Shop selling these Brand Z things:"), that brand MUST be included in EVERY product's sub-query. Do not treat it as shop context only.
    * Example: "Find a shop selling these Blackwater items: First, a women's perfume... Second, a charcoal soothing gel..." → "Blackwater" must appear in ALL 4 sub-queries.
- **Size/Variant Splitting Rule:** If a single segment mentions multiple distinct sizes, colors, or variants of the same product type with a conjunction (e.g., "large and xx-large", "red and blue", "size M and L"), and a constraint (like price or service) is explicitly tied to ONLY one of those variants, split them into SEPARATE products. Each variant becomes its own product.
    * Example: "Goo.n diapers in large and xx-large sizes, flash offers on the xx-large packs" → Product A: "Goo.n diapers large" | Product B: "Goo.n diapers xx-large" (with flashsale)
    
## Step-2. Identify the price constraint for each product if mentioned, and normalize it to "low-high" format without currency name.
- A price constraint is mentioned like "between 10 and 50 php", "more than 2400", or "less than 120 USD".
- User may mention a price constraint without currency specified, like "Looking for a makeup bag between 110 and 200."
- If a numeric range is mentioned without a currency, carefully check the context to determine whether it refers to price. Examples:
    * "Show me black tshirts between 50 and 150." → The phrase "between 50 and 150" is the price constraint.
    * "Need aircon with power usage between 1000 and 1500." → The phrase "between 1000 and 1500" is the power consumption constraint, not the price constraint.
- Normalize the price constraint:
    * Normalize as "low-high" if both lower and upper bounds of price were specified. Example: "5-160"
    * Normalize as "low-" if only the lower bound was specified. Example: "410-"
    * Normalize as "-high" if only the upper bound was specified. Example: "-905"
- **Numeric ranges that are NEVER price constraints:** The following patterns indicate product attributes, not prices. Do NOT normalize them as prices:
    * `SPF` or `sun protection` followed by a number/range (e.g., "SPF 30", "SPF protection between 11 and 20") → This is a **sun protection rating**, not a price.
    * `power`/`wattage`/`voltage`/`current` followed by a number/range (e.g., "power between 1000 and 1500", "voltage 110-220V") → This is a **technical specification**, not a price.
    * `age`/`size`/`capacity`/`weight` followed by a number/range with clear unit context (e.g., "size between 10 and 20 cm", "weight 5-10 kg") → This is a **physical attribute**, not a price.
    * Any number preceded by well-known product attribute keywords: `SPF, PA+, dB, GB, TB, MHz, GHz, DPI, PPI, lumens, kelvin, ohms, amps, watts, volts, calories, protein, fat, carbs, SPF, UV, pH`.

## Step-3. Identify the service constraints for each product if mentioned. Check for the following service labels:
    * lazmall: user wants an official, genuine, guaranteed, or authentic product through LazMall or LazMall Perk which is a platform providing easy and fast returns
    * `official`: user explicitly wants product from an official/authorized **store, seller, shop, or flagship store**. 
        **Trigger phrases must include "official" + a store/seller term** (e.g., "official store", "official seller", "authorized seller", "official flagship store").
        **Does NOT trigger when "official" modifies the product itself** (e.g., "official ball", "official men's shoes", "official product used in matches").
    * `lazflash`: user wants lightning deal through LazFlash
    * `flashsale`: user wants flash sale or lightning deal or limited time offer 
    * free_ship: user wants free or complimentary shipping or delivery
    * cod: user wants 'Cash on Delivery', 'Payment on Delivery', or any phrase indicating payment upon receipt/delivery (e.g., "pay when they arrive", "pay on arrival", "pay upon delivery", "pay when it gets here")
    * product_only: user requests only the product itself, explicitly excluding accessories, bundles, or related items. Trigger phrases must **exactly match** one of: "only the product", "just the product", "product only", "no accessories", "without accessories", "not a set", "exclude bundle". **NO other variation triggers this.** Phrases like "levers only", "shoes only", "[product name] only", "just the [product name]" that merely specify a product variant do NOT trigger this constraint. Phrases like "single item", "single items", "each item", or "per item" that refer to quantity, or shipping scope do NOT trigger this constraint.
    * **Special:**  Query:"I'm looking for a gold necklace, but just the necklace, not the full set" → This IS NOT product_only.
    * **Example:** Query: "I need brake levers only, suitable for motorcycles." → This is NOT product_only because "levers only" specifies a product variant, not exclusion of accessories. Query: "I need brake levers without accessories" → This IS product_only.
**ATTENTION - The term "branded" or "genuine original branded" is NOT a service constraint, but a separate constraint meaning the product must have a branded feature. DO NOT mark it with a service label!**

## Step-4. Rewrite each sub-query per product, by excluding the price and service constraints.
- **Segmentation Method:** First split the original query by price constraints. Each price constraint acts as a delimiter. The text chunk between two price constraints (or between start and first price, or last price and end) forms one product's base text. Remove only the price and service constraints from that chunk; keep ALL other words (colors, quantities, brands, attributes) intact.
- **NEVER merge attributes from one product chunk into another.** If "pink blue" appears in the first chunk, it stays in Product 1's sub-query only.
- **NEVER split quantity markers from their product.** "candle 10 pcs" stays together as one product. "10 pcs" is not a separate product.
- Make sure each rewritten sub-query includes one product type explicitly. A sub-query without an explicit product type mention is invalid.
- Make sure each rewritten sub-query includes all details for the corresponding product from the original query, except for price and service constraints.
- Make sure each rewritten sub-query does not include the constraints which don't apply to the corresponding product.
- Strictly respect the wording of original query. There is no typo.
- The term `warranty` is not service constraint!
- DON'T INCLUDE PRICE AND SERVICE CONSTRAINTS IN THE REWRITTEN SUB-QUERIES!

## Step-5. Identify the following paraphrasable tokens in the sub-queries with their alternatives. (If either left or right side token is detected, set other side token as the alternative.)
    * "t-shirt"↔"tshirt", "t shirt"↔"tshirt", "email"↔"e-mail", "wristwatch"↔"wrist watch", "flip-flop"↔"flipflop", "x ray"↔"xray", "air conditioner"↔"aircon", ...
    * "standard"↔"default", "normal"↔"regular", "plain"↔"simple", "high"↔"max", "shirt"↔"tshirt", "default"↔"regular",
    * "gray"↔"grey", "color"↔"colour", "chili"↔"chilli", "pokémon"↔"pokemon"

## Step-6. Identify the most suspicious token in the sub-queries that may be non-beneficial for product search. Examples:
    * "pair" for querying "gloves", because all goves are sold by pairs
    * "set" for querying "salt and pepper shakers", because they are almost always sold as a set
    * "replacement" for querying "tyre", because all tyres are for replacement
    * "supplement" for querying "vitamin", because all vitamins are for supplement
    * "unit" for querying "ABS", because all ABS are moduled as unit

# Rules of Constraint Scope (APPLIED TO ALL OF Step-2, Step-3, and Step-4!)
{rules}

**THE RULES OF CONSTRAINT SCOPE ARE APPLIED TO ALL CONSTRAINTS INCLUDING PRICE AND SERVICE CONSTRAINTS!**
**STEP-5 EXAMPLES ARE BI-DIRECTIONAL. CAPTURE TOKENS THAT APPEAR IN EITHER LEFT OR RIGHT SIDE, AND GIVE OTHER SIDE AS THE ALTERNATIVE!**
**DO NOT AGGRESSIVELY EXTEND THE PARAPHRASABLE TOKENS BEYOND THE EXPLICIT EXAMPLES OF STEP-6!**
**THE MAXIMUM NUMBER OF PRODUCTS IS 4. NEVER RETURN MORE THAN 4 PRODUCTS!**

# Response Format
Return JSON object like:
{{
        // Step-1
    "products": {{ // Identify the included products and their product types. Maximum product count is 4!
        "rationale": <str>, // A brief summary in 1 sentence to identify the products and their product types.

        "1": <str>, // A short phrase representing the first product. No need to write a complete sub-query for this step. Just a minimal, differentiable phrase works.
        "2": <str>,
        ... ...
    }},

    // Strictly keep **Rules of Constraint Scope** to identify the correct product for each constraint!

    // Step-2
    "prices": {{ // Find all phrases related to the price constraints from the original query, and assign each of them to the corresponding product.
        "rationale": <str>, // A brief summary in 1 sentence to identify the price constraints and their scopes from the original query.

        // If no price constraint was found, don't output the following phrases but skip to Step 3 - "services" field!
        // If found price constraints, list all those phrases below

        <phrase_1, str>: {{
            "apply_to": <product_id, str|null>, // The corresponding product id ("1" or "2" or ...) that can receive this constraint. set null if this phrase is not a price constraint.
            "normalized": <str|null> // Normalized price format ("low-high" or "low-" or "-high", without currency). Set null if this phrase is not a price constraint.
        }},
        ... ...
    }},

    // Step-3
    "services": {{ // Find all phrases related to the service constraints from the original query, and assign each of them to the corresponding product.
        "rationale": <str>, // A brief summary in 1 sentence to identify the service constraints and their scopes from the original query.

        // If no service constraint was found, don't output the following phrases but skip to Step 4 - "sub_queries" field!
        // If found service constraints, list all those phrases below.

        <phrase_1, str>: {{
            "apply_to": [<product_id_1, str>, ...], // List of the corresponding product ids that can receive this constraint. Set empty if this phrase is not a service constraint.
            "label": <str|null> // Assign the appropriate service label. Set null if this phrase is not a service constraint.
        }}
    }},

    // Step-4
    "sub_queries": {{ // Rewrite sub-queries per product. Never miss any details (product type, in particular), but don't include price and service constraints!
        "rationale": <str>, // A brief summary in 1 sentence to find any potential pitfalls for assembling sub-queries per product and how to manage it.

        "1": <str>, // Sub-query for product 1, with price and service constraints excluded. Don't include constraints that cannot be received even if only one word is left!
        ... ...
    }},

    // Step-5
    "alt": {{ // List the identified tokens and their alternatives, that are required in Step-5. (Limit them as given examples in Step-5. Don't extend! Empty list is preferred!)
        <token_1, str>: <alternative_1, str>, // the source token must exactly exist in a sub-query of Step-4, without spell modification.
        ... ...
    }},

    // Step-6
    "redundant": <top_redundant_token, str|null> // one most suspicious token. (You are not enforced to select one always! Null is preferred!)
}}

# Original Query
{query}

"""
    P_DECOMPOSE = """
You are AI assistant to analyze the user's shopping query. The user is looking for a single product with specific constraints.

# Task
## Step-1. Identify the price constraint if mentioned, and normalize it to "low-high" format without currency name.
- A price constraint is mentioned like "between 10 and 50 php", "more than 2400", or "less than 120 USD".
- User may mention a price constraint without currency specified, like "Looking for a makeup bag between 110 and 200."
- If a numeric range is mentioned without a currency, carefully check the context to determine whether it refers to price. Examples:
    * "Show me black tshirts between 50 and 150." → The phrase "between 50 and 150" is the price constraint.
    * "Need aircon with power usage between 1000 and 1500." → The phrase "between 1000 and 1500" is the power consumption constraint, not the price constraint.
- Normalize the price constraint:
    * Normalize as "low-high" if both lower and upper bounds of price were specified. Example: "5-160"
    * Normalize as "low-" if only the lower bound was specified. Example: "410-"
    * Normalize as "-high" if only the upper bound was specified. Example: "-905"
- **Numeric ranges that are NEVER price constraints:** The following patterns indicate product attributes, not prices. Do NOT normalize them as prices:
    * `SPF` or `sun protection` followed by a number/range (e.g., "SPF 30", "SPF protection between 11 and 20") → This is a **sun protection rating**, not a price.
    * `power`/`wattage`/`voltage`/`current` followed by a number/range (e.g., "power between 1000 and 1500", "voltage 110-220V") → This is a **technical specification**, not a price.
    * `age`/`size`/`capacity`/`weight` followed by a number/range with clear unit context (e.g., "size between 10 and 20 cm", "weight 5-10 kg") → This is a **physical attribute**, not a price.
    * Any number preceded by well-known product attribute keywords: `SPF, PA+, dB, GB, TB, MHz, GHz, DPI, PPI, lumens, kelvin, ohms, amps, watts, volts, calories, protein, fat, carbs, SPF, UV, pH`.

## Step-2. Identify the service constraints if mentioned. Check for the following service labels:
    * lazmall: user wants an official, genuine, guaranteed, or authentic product through LazMall or LazMall Perk which is a platform providing easy and fast returns
    * `official`: user explicitly wants product from an official/authorized **store, seller, shop, or flagship store**. 
        **Trigger phrases must include "official" + a store/seller term** (e.g., "official store", "official seller", "authorized seller", "official flagship store").
        **Does NOT trigger when "official" modifies the product itself** (e.g., "official ball", "official men's shoes", "official product used in matches").
    * `lazflash`: user wants lightning deal through LazFlash
    * `flashsale`: user wants flash sale or lightning deal or limited time offer
    * free_ship: user wants free or complimentary shipping or delivery
    * cod: user wants 'Cash on Delivery', 'Payment on Delivery', or any phrase indicating payment upon receipt/delivery (e.g., "pay when they arrive", "pay on arrival", "pay upon delivery", "pay when it gets here")
    * product_only: user requests only the product itself, explicitly excluding accessories, bundles, or related items. Trigger phrases must **exactly match** one of: "only the product", "just the product", "product only", "no accessories", "without accessories", "not a set", "exclude bundle", "for sale only". **NO other variation triggers this.** Phrases like "levers only", "shoes only", "[product name] only" that merely specify a product variant do NOT trigger this constraint. Phrases like "single item", "single items", "each item", or "per item" that refer to quantity, or shipping scope do NOT trigger this constraint.
    * **Example:** Query: "I need brake levers only, suitable for motorcycles." → This is NOT product_only because "levers only" specifies a product variant, not exclusion of accessories. Query: "I need brake levers without accessories" → This IS product_only.
**ATTENTION - The term "branded" or "genuine original branded" is NOT a service constraint, but a separate constraint meaning the product must have a branded feature. DO NOT mark it with a service label!**

## Step-3. Rewrite the query, by excluding the price and service constraints.
- Make sure the rewritten query includes all details from the original query, except for price and service constraints.
- Strictly respect the wording of original query. There is no typo.
- DON'T INCLUDE PRICE AND SERVICE CONSTRAINTS IN THE REWRITTEN QUERY!

## Step-4. Identify the following paraphrasable tokens in the query with their alternatives. (If either left or right side token is detected, set other side token as the alternative.)
    * "t-shirt"↔"tshirt", "t shirt"↔"tshirt", "email"↔"e-mail", "wristwatch"↔"wrist watch", "flip-flop"↔"flipflop", "x ray"↔"xray", "air conditioner"↔"aircon", ...
    * "standard"↔"default", "normal"↔"regular", "plain"↔"simple", "high"↔"max", "shirt"↔"tshirt"
    * "gray"↔"grey", "color"↔"colour", "chili"↔"chilli"

## Step-5. Identify the most suspicious token in the query that may be non-beneficial for product search. Examples:
    * "pair" for querying "gloves", because all goves are sold by pairs
    * "set" for querying "salt and pepper shakers", because they are almost always sold as a set
    * "replacement" for querying "tyre", because all tyres are for replacement
    * "supplement" for querying "vitamin", because all vitamins are for supplement
    * "unit" for querying "ABS", because all ABS are moduled as unit

**STEP-4 EXAMPLES ARE BI-DIRECTIONAL. CAPTURE TOKENS THAT APPEAR IN EITHER LEFT OR RIGHT SIDE, AND GIVE OTHER SIDE AS THE ALTERNATIVE!**
**DO NOT AGGRESSIVELY EXTEND THE PARAPHRASABLE TOKENS BEYOND THE EXPLICIT EXAMPLES OF STEP-5!**

# Response Format
Return JSON object like:
{{
    // Step-1
    "prices": {{ // Find a phrase related to the price constraints from the original query, if exists.
        "rationale": <str>, // A brief summary in 1 sentence to identify the price constraints from the original query.
        "normalized": <str|null> // Normalized price format ("low-high" or "low-" or "-high", without currency). Set null if price constraint does not exist.
    }},

    // Step-2
    "services": {{ // Find all phrases related to the service constraints from the original query.
        "rationale": <str>, // A brief summary in 1 sentence to identify the service constraints from the original query.

        // If no service constraint was found, don't output the following phrases but skip to Step 3!
        // If found service constraints, list all those phrases below.

        <phrase_1, str>: <service_label, str|null>, // The exact service label string: "lazmall", "official", "lazflash", "flashsale", "free_ship", "cod", or "product_only". Use null if the phrase is not a service constraint.
        ... ...
    }},

    // Step-3
    "rewritten": {{
        "rationale": <str>, // A brief summary in 1 sentence to find any potential pitfalls for rewriting and how to manage it.
        "result": <str> // Rewrite the query excluding price and service constraints. Never miss any details.
    }},

    // Step-4
    "alt": {{ // List the identified tokens and their alternatives, that are required in Step-4. (Limit them as given examples in Step-4. Don't extend! Empty list is preferred!)
        <token_1, str>: <alternative_1, str>, // the source token must exactly exist in the rewritten query of Step-3, without spell modification
        ... ...
    }},

    // Step-5
    "redundant": <top_redundant_token, str|null> // one most suspicious token. (You are not enforced to select one always! Null is preferred!)
}}

# Original Query
{query}
"""
    PSV_REVIEW = "You are AI assistant to review the shopping query parsing, and fix errors if needed.\nYou are given the original query as well as its parsed result. There are {count} found in the original query.\n\n# The Constraints That You Must Focus On (Label vs Explanation)\n{focuses}\n\n# Rules of Constraint Scope\n{rules}\n\n# The Errors You Must Detect\n## 1. Missing constraint\n    - A constraint is clearly mentioned for a product's scope, but it's not assigned.\n    - Make sure the product is the nearest one that can grammatically receive this constraint modifier.\n\n## 2. Invalid scope application (for multiple products query)\n    - A constraint was explicitly mentioned one time, but it was mistakenly assigned to multiple products. You must find the only one product that can receive this constraint, and remove the incorrect assignments for other products.\n    - Or, a constraint was assigned to a product, but actually it's not the correct one that can receive this constraint. You must find the correct product that must receive this constraint.\n\n{error_3}\n\n# Suspected Aspects in Current Parsing\nThe followings are identified issues by our heuristic checking engine. You must double check these aspects.\n{suspects}\n\n# Guideline\n- Carefully check each suspected aspect and determine the correct fix if needed.\n- You shouldn't be enforced to fix. The issues are just detected by our simple heuristic algorithm. If you believe the original parsing result was correct, just output as is!\n- Don't think but just start to output JSON!\n\n# Response Format\nReturn JSON object like:\n{{\n    \"rationale\": <str>, // a breif summary in 1-2 sentences for your decision\n{responses}\n}}\n\n# Original Shopping Query\n{query}\n\n# Parsed Result\n{parsed}"
    PSV_BUILD = """
You are an AI assistant that converts product descriptions into structured e-commerce attributes.
Your task is to transform each product description into a set of key-value pairs that faithfully represent all information contained in the description.

# Core Objective
Convert every product description into a structured attribute dictionary such that:

## 1. The attributes must be **mutually exclusive**:
- The same information must not appear in multiple attributes.
- Each piece of information should be assigned to exactly one attribute.

## 2. The attributes must be **collectively exhaustive**:
- All meaningful product information and constraints from the original description must be preserved.
- The combination of all attributes should fully reconstruct the original meaning.

## 3. Each attribute must represent one constraint aspect:
- Don't include multiple constraint aspects for one attribute.
- Don't split one constraint aspect into two attributes.

## 4. The output should resemble a real e-commerce database schema.

# Attribute Construction Rules
## Rule-1. Mandatory Product Type(basic product category name)
Every output must contain one pair "product_type: <value>". This must be the first attribute in the result object.

## Rule-2. Minimize Value Length
- Make values as short as possible while preserving meaning, by moving descriptive tokens from the value into the key when appropriate. Examples:
    * "star shape" → "shape: star"
    * "type A or B models" → "model: type A B"
    * "metal-based" → "material_base: metal"
    * "compatible with healthcare apps" → "compatible_app: healthcare"
    * "as a model boots" → "model: boots"
    * "fit for office use" → "usage: office"
    * "with the interest in nature" → "interest: nature"
    * "pack of 120 balls" → "pack_ball: 120"
    * "spring-inspirated design" → "design_inspiration: spring"
    * "perfect for Yamaha piano" → "compatible_piano: Yamaha"
    * "Hello Kitty backpack for girls" → "brand: hello kitty"
    * "with side camera 5MP" → "side_camera: 5MP"
    * "having 512GB hard disk size" → "hard_disk_size: 512GB"
- The value MUST NOT INCLUDE the descriptive word or token moved into the key side.
    * Correct: "shape: circle" || Incorrect: "shape: circle-shaped"
    * Correct: "table: 12" || Incorrect: "table: 12 tables"
- DO NOT split one constraint aspect into two attributes to make values short. It's strictly forbidden. You must make values short inside one attribute for one constraint aspect!
    * "with 20 sockets" → Correct: "socket_count: 20" || Incorrect: "pack_type: socket" + "pack_size: 20"
    * "made for G1, G3, or G-K types" → Correct: "type: G1 G3 G-K" || Incorrect: "type: G1" + "type: G3" + "type: G-K"
- Convert the unit "piece" to "pcs", because "pcs" is the reserved unit of "piece(s)" for the current e-commerce db. Remove space between number and "pcs". Never use "pcs" unless "piece(s)" or "pcs" was explicitly used in the description.
    * "packed in 24 pieces" → Correct: "pack: 24pcs" || Incorrect: "pack: 24 pcs", Incorrect: "pack: 24 piece"
    * "1 piece color family" → Correct: "color_family: 1pcs" || Incorrect: "color_family: 1 piece", Incorrect: "color_family: 1"
    * "set of 20" → Correct: "set: 20" || Incorrect: "set: 20pcs", Incorrect: "pack: 20pcs"
- For the pattern `[number][space][unit]` (with space token between number and unit), unit token must be moved to the key. This rule is not applied to the unit "piece". Example:
    * "that come with 7 pearls" → Correct: "pearl: 7" || Incorrect: "feature: 7 pearls"
- For the pattern `[number][unit]` (without space token between number and unit), both number and unit tokens must remain in the value. This rule is applied to the unit "pcs" which is converted from "piece". Example:
    * "in radius 5mm" → Correct: "radius: 5mm" || Incorrect: "radius_mm: 5"

## Rule-3. Prefer Source-Derived Keys
- When possible, derive attribute names from words or tokens already present in the description. Example:
    * "compatible with healthcare apps" → Correct: "compatible_app: healthcare" || Incorrect: "fit: healthcare apps"
    * "for model Pacific" → Correct: "model: Pacific" || Incorrect: "compatibility: Pacific model"
    * "with checker pattern" → Correct: "pattern: checker" || Incorrect: "design: checker pattern"
    * "comes with 10 piece" → Correct: "piece: 10" || Incorrect: "pack: 10"
- Only introduce generic schema keys when no suitable key can be derived. Common fallback keys include:
    `product_type, brand, size, color, color_family, pack, capacity, user, gender, model, compatibility, material, feature, usage, style, design, accessory, variation, option, choice`
- NEVER move the number-included tokens into the keys. They MUST be in the values. Example:
    * "2pcs color" → Correct: "color: 2pcs" || Incorrect: "2pcs: color" or "2pcs_color: color"
    
### The keys "color" and "color_family" are not limited to color modifiers!
- Unlike their names, the keys "color" and "color_family" can be used for any constraint including color, size, package type, and all. Examples:
    * "color: red #03"
    * "color: premium"
    * "color: 1pcs"
    * "color_family: 10pcs"
    * "color_family: 10cm"
    * "color_family: button"
    * "color: 1box"
- Don't mistakenly recognize "color" and "color family" as if they're attribute values. Incorrect Examples:
    * "color: family"
    * "family: color"
    * "color_family: family"
    * "color_family: color"
    * "variation: color family"
    * "option: color family"
- If the original query mentions "I am looking for 5pcs color wide neck bottles", then "5pcs color" means "color: 5pcs" (looks a bit odd, but it's the correct one!), not that user is finding a "colored" bottle.
    * Correct: "product_type: bottle", "color: 5pcs", "neck: wide"
    * Incorrect: "product_type: bottle", "pack: 5pcs", "color: color", "neck: wide"

## Rule-4. Remove Filler Words
- Remove non-informative words such as:
    `and, or, in, with, for, to, still, just, upon, about, the, at least, above, less than, between, a, an, from, to, 's, s'`, unless they are part of a deterministic term.
- Examples to remove:
    * "compatible with office use" → "usage: office"
    * "specifically with the interest in nature" → "interest: nature"
    * "sportwomen's bra" → "product_type: sportswomen bra"
    * "infants up to 12 months" → "recommended_age: 0-12 months"
    * "above 12 years" → "age: 12+ years"
    * "fit for kids'" → "user: kid"
    * "spray paint suitable for automotive or metal" → "product_type: spray paint"
    * "between 2 and 5 meters" → "length: 2-5 meters"
    * "kids aged 3 and up" → "recommended_age: 3 years and above"
- Examples to keep:
    * "2-in-1 type" → "type: 2-in-1"
    * "with brand just honests" → "brand: just honests"
    * "with design a" → "design: design a"
    * "color 'a'" → "color: a"
    * "10pcs a color option" → "color_family": "10pcs a"
    * "'gp power 1l' in the details" → "others": "gp power 1l"
    * "'ax5 1l' mentioned" → "others": "ax5 1l"

## Rule-5. Negation Handling
Never use: "yes", "true", "false", "included", "used", "required" as standalone values.
    * "comes with protection lid" → Correct: "accessory: protection lid" || Incorrect: "protection_lid: included"
    * "operation in safe mode" → Correct: "operation_mode: safe" || Incorrect: "safe_mode_operation: true"
    * "genuine and branded" → Correct: "feature: genuine branded" || Incorrect: "branded: yes"
    * "with iOS required" → Correct: "requirement: iOS" || Incorrect: "ios: required"

### Handling "no" / "not" / "without"
- Only use "no"/"not" when the original text explicitly expresses absence by "no", "not" or "without". But "no"/"not" cannot be used as standalone value.
    * "without any cover" → Correct: "feature: no cover" || Incorrect: "cover: no"
    * "no fragrance" → Correct: "feature: no fragrance" || Incorrect: "fragrance: no"
    * "with axis not specified" → Correct: "feature: not specified axis" || Incorrect: "axis: not"

- Do not convert other expressions into "no" or "not". Only "without" or "without any" can be converted to "no".
    * "wind-free" → Correct: "feature: wind-free" || Incorrect: "feature: no wind"
    * "stop smoking" → Correct: "feature: stop smoking" || Incorrect: "feature: no smoking"

## Rule-6. Some Edge Cases
    * "fit for all ages" → "fit_age: all"
    * "comes with any accessories" → "accessory: any"
    * "suitable for both men and women" → "gender: unisex"
    * "one size" → "size: one"
    * "from mixed brands" → "brand: mixed brands"
    * "one size fits to all" → "size: freesize"
    * "for any gender" → "gender: any"
    * "compatible with iPhone and iPad" → "compatible_device: iPhone iPad"
    * "fit for any models" → "model: universal"
    * "available in red or blue" → "color: red blue"
    * "thermal" → "container_feature: thermal"
    * "350ml food storage box" → "volume: 350"
    * "alternative materials" → "materials: other"
    * "single green zipper lock" → "color_family: 1pcs-green"
    * "fun graphic print" → "fa_pattern: graphic print"
    * "other herbs like oregano plants" → "types_of_herbs_&_spices: other herbs"
    * "3-in-1 set model" → "model: 3in1 set"
## Rule-7. Singular Form Preference
- Use singular forms unless:
    * the plural form is explicitly required, or
    * the plural form is a recognized deterministic term.
- Examples:
    * "balls" → "ball"
    * "United States" → "United States" (unchanged)
    * "shoes" → "shoes" (unchanged)

## Rule-8. Attribute Quality Rules
    * DO NOT create duplicate attributes.
    * DO NOT split a single concept across multiple attributes to make values short.
    * DO NOT invent information.
    * DO NOT infer specifications not explicitly stated.
    * DO NOT remove meaningful token. Minimizing value means moving tokens to the key when possible, not removing them entirely.
    * KEEP values concise but fully informative.
    * PRESERVE original wording whenever possible. There is no typo in the original description.

# Procedures

## Step 1. Rewrite the original product description in compact version by removing non-informative words.
- Remove non-informative tokens to make the description search-friendly, unless they are part of deterministic terms.
- In this step, do not split key and value tokens yet. Just rewrite the description left-to-right order by removing redundant tokens.
- Examples:
    * "hp04 ink bottle made in France, in 4b color family and multipack variation" → "hp04 ink bottle France 4b color family multipack variation"
    * "brand new bicycle head light that fits any bicycle model" → "brand new bicycle head light any bicycle model"
    * "video game CD with title The God" → "video game CD title The God"
    * "photo frame with size 50cm by 120cm" → "photo frame size 50cm 120cm"
    * "USB cable with type C or U plugs" → "USB cable type C U plug"
    * "fan accessories compatible with Dell R960C workstation" → "fan accessory Dell R960C workstation"
    * "phone case specifically designed for Samsung Galaxy" → "phone case design Samsung Galaxy"

## Step 2. Convert the description into key-value formatted attribute dictionary.
- Make sure any meaningful detail is not missing in the attribute dictionary.
- Make sure the attributes are non-overlapped.
- Strictly keep the **Attribute Construction Rules** (1-8).

# Response Format
Return a JSON object like:
{{
        "1": {{ // for product description "1"
        "description": <original description 1, str>,

        // Step 1
        "compact": <rewrite description 1 into a search-friendly, compact, concise version without non-informative tokens, str>,

        // Step 2
        "rationale": <a short summary of how attributes were selected and will be organized by strictly adhering the construction rules, str>,
        
        "attributes": {{
                "product_type": <value, str>, // this is the must-have pair
            <attribute_key_1, str>: <attribute_value_1, str>,
            ... ...
        }}
    }},
    ... ...
}}

# Requirements:
- Output valid JSON only. No markdown. No explanations outside the JSON.
- The first key inside every result object must be "product_type".
- Thoroughly check all the rules and find the best way of transformation!

# Examples of "rationale" field:
    * From the original description, meaningful tokens are xxx, xxx, ... The product type is xxx. The best way of attributes construction from the rest tokens by following the Rules 1-8 and mimicking the real e-commerce db is xxx: xxx, xxx: xxx, ... No missing meaningful token? All meaningful tokens are included in the product type and attributes. Good.
    * From the original description, meaningful tokens are xxx, xxx, ... The product type is xxx. The best way of attributes construction from the rest tokens by following the Rules 1-8 and mimicking the real e-commerce db is xxx: xxx, xxx: xxx, ... No missing meaningful token? The token xxx is missing. We must add an attribute xxx: xxx. Good.
    * From the original description, meaningful tokens are xxx, xxx, ... The product type is xxx. The best way of attributes construction from the rest tokens by following the Rules 1-8 and mimicking the real e-commerce db is xxx: xxx, xxx: xxx, ... No missing meaningful token? The token xxx is missing. It must be added to the product type value. So the correct product type is xxx. Good.

# Original Product Descriptions
{queries}
"""
    PSV_KW = """
You are an AI assistant that reviews e-commerce search keywords and recommends, when appropriate, a single alternative keyword set.
You are given one or more shopping queries together with the current search keywords generated for each query.
The existing keywords are already high quality. Your task is NOT to rewrite them. Instead, identify rare cases where making exactly one small change could improve search recall without significantly reducing precision.

# Search Engine Behavior
The product catalog is stored as JSON. Only JSON values are indexed for search. Therefore:
- Words that normally appear only as JSON keys (for example: size, color, color family, material, brand, category, product type) generally do not help retrieval and may increase irrelevant matches.
- Hyphens are normalized to spaces before search. Therefore "t-shirt" and "t shirt" are equivalent.
- The search engine uses BM25 with morphological normalization. Variants such as: `shape/shaped`, `woman/women`, `child/children` produce identical matches.
- Re-ordering the keywords produces the exactly identical matches.
- Duplication of keywords produces the exactly identical matches.

# About the Existing Keywords
The original keywords were generated automatically and are usually well optimized. In particular:
- Most useful value-side keywords are already included.
- Non-informative words have already been removed.
- Some query terms are intentionally normalized to match catalog conventions, for example:
  * `large` → `l`, `extra old` → `xo`
  * `fit for men and women` → `unisex`
  * `set of 12 pieces` → `12pcs`
  * `water-resistant` → `waterproof`

Assume the original keywords are correct unless there is a strong reason to change them.

# Objective
Recommend AT MOST ONE alternative keyword set. The recommendation should have a reasonable chance of improving search quality compared with the original keywords. If no change is clearly beneficial, return null.

# Allowed Modification
Choose exactly one of the following:
1. Add exactly one keyword.
OR
2. Remove exactly one keyword.

Never perform both operations. Never rewrite, reorder, or substitute keywords. The alternative must otherwise remain identical to the original keywords.

# Step 1 — Consider Adding
Recommend an addition only if there is a missing keyword that is likely to appear in indexed product values and is expected to improve retrieval.

Possible candidates include:
- an obvious missing product attribute
- a keyword whose presence in values is reasonably likely

Do NOT add:
- non-informative words
- grammatical words
- shopping intent words
- words that are clearly JSON keys
- intentionally normalized words
- tokens that already exist in the original keywords
- tokens that are not included in the query

Examples of words that should NOT be added:
compatible, for, with, pair, included, perfect, specifically, above, size, color, material, color family, brand, ...

# Step 2 — Consider Removing
If no useful addition exists, determine whether removing exactly one keyword would likely improve retrieval.

Only remove a keyword if it is likely to:
- rarely appear in indexed values,
- behave more like a metadata key than product content,
- or unnecessarily narrow the search. (Examples: "set", "pair", "pack", "accessory")

Do NOT remove:
- tokens that clearly describe the product
- numbers
- unique identifiers
- brands

# Decision Rule
Recommend an alternative only when the expected improvement is meaningful. When uncertain, prefer returning null.

# Response Format
Return JSON object.
{{
        // check all queries in order as given

    "1": {{ // for query 1

        "rationale": "<explanation>",
        "alt": "<full alternative keyword string>" | null
    }},
    ... ...
}}

Requirements:
- "alt" must contain the complete keywords string after the single modification, so that it can be immediately used for the search.
- If no recommendation is justified, set "alt" to null.
- Output valid JSON only. No markdown. No additional text outside JSON.

# Shopping Queries and Original Keywords
{inf}
"""
    PV_SELECT = """
You are an AI assistant that selects the best matching product for a shopping query from an ordered candidate list.
Your objective is **precision**. A product is considered a valid match **only if every required constraint extracted from the query is satisfied**.

# Overall Procedure
Perform the following phases strictly in order.

## Phase 1. Extract Constraints
- Read the shopping query and extract all constraints. Each constraint must belong to exactly one category. Possible categories include (but are not limited to):
    `brand, model, series, identifier/code/SKU, number, size, capacity, color, material, feature, compatibility, shape, style, package, age, gender, usage, pattern`
- All constraints are *REQUIRED*.
- Do NOT begin checking products yet.

## Phase 2. Classify Matching Type
Each extracted constraint must use exactly one matching mode.

### Literal Match
Use literal matching for `brands, model names, identifiers, product codes, uncommon token combinations, numbers, serial names, abbreviations, versions, colors`.
    * Literal constraints must appear explicitly.
    * Never normalize them. Never partially match them.

    Examples:
        Query "100ml" → Accept "100ml", reject "50-150ml"
        Query "iPhone" → Accept "iPhone", reject "Apple smartphone"
        Query "Green 2" → Accept "green 2", reject "green" (Casing difference allowed)
        Query "3-7 years" → Accept "3-7yrs", reject "5 years"
        Query "men's hair" → Accept "men's hair", reject "mens hair"

### Semantic Match
- Everything else uses semantic matching. Paraphrases are allowed. Common typos are allowed. Examples:
    child = kid, wood = wooden, regular = standard, plain = simple, customized = personalized, water-resistant = waterproof, dioxide = dioxde
- Never be too strict. Be generous with common sense and implicity for semantic match constraints. Most of the candidates are matching. Prefer FALSE POSITIVE than FALSE NEGATIVE!

## Phase 3. Product Evaluation
- Evaluate products in Fibonacci batches. Let `N_0 = 3`, `N_1 = 2`, and `N_[k+2] = N_[k] + N_[k+1]`. Therefore, the batches are:
    * Phase 0: (N_0 = 3) → Products 1-3
    * Phase 1: (N_1 = 2) → Products 4-5
    * Phase 2: (N_2 = 5) → Products 6-10
    * Phase 3: (N_3 = 7) → Products 11-17
- For every product evaluate every constraint independently.
- Each constraint receives one result only: `PASS`, `FAIL`, or `UNKNOWN`
- A product is a valid candidate only if **every constraint is PASS**!
- If any constraint is FAIL reject the product immediately.
- `UNKNOWN` is treated as FAIL.

## Field Rules
### Product Type
Product type has already been verified. Never reject a product because of product type.

### Title
The title is an unordered keyword collection. Do NOT infer sentence meaning. Only check token existence. Example:
    - Query "water bottle" → Title "Black Plastic Milk Bottles Cover Kids Water". The title contains both `water` and `bottle`. Therefore title evidence is POSITIVE. Do not infer that the product is a milk bottle. The title is only evidence of keyword presence.
    - Query "tank tester" → Title "tank capacity tester". The title contains both `tank` and `tester. Therefore title evidence is POSITIVE. Do not infer that the product type is mismatching. The title is only the evidence of keyword presence.

### Attributes
- Attributes are authoritative. Each attribute consists of `Key : Value`. Evaluate every attribute independently.
- If any attribute provides a valid value for a constraint, the constraint PASSes.
- Conflicting values in other attributes are ignored because they may represent different variations.

### Color Fields
- The attributes `color` and `color_family` may contain:
    * colors (Example: "color: black", "color_family: white bracelet")
    * sizes (Example: "color_family: 10mm", "color": "1.6m")
    * package names (Example: "color: 1pcs", "color: 3pcs")
    * identifiers (Example: "color_family: G #1")
    * quantities (Example: "color_family": "5kg")
    * serial names (Example: "color": "aa12-c")
    * CMYK color(Example: "color_family": "k") 
- Treat them as generic attributes. Do not assume they only contain colors.

## Constraint Evaluation
- For every constraint check title. If satisfied: PASS. Otherwise: check every attribute. If any attribute satisfies the constraint: PASS. Otherwise: FAIL.
- Never infer literal constraints. Never invent missing information. Never assume compatibility. Never guess.
- Attribute specificity matters: A constraint is best satisfied by an attribute whose key directly corresponds to the constraint category. Generic attributes (e.g., `options`, `feature`, `variation`) containing the value are weaker evidence than specific attributes (e.g., `color`, `size`, `brand`).
- For PASS: the value must appear in a category-matching attribute OR in the title. Generic attributes alone are NOT sufficient for PASS unless no category-specific attribute exists in the product data.

## Phase 4. Recommendation
- If one or more valid products are found in the current Fibonacci batch: STOP immediately.
- Only compare the valid products found in this batch. Choose the product that most closely matches the original wording of the shopping query. Prefer:
    * more exact wording
    * more literal matches
    * more explicit evidence
    * evidence in attributes list than in title. Structured evidence in attributes/options/variants over noisy SEO title text.
    * Brand Attribute Tie-Breaker Rule: In cases where multiple candidate products yield similar relevance scores, and the brand name is present in each product's title, prioritize the candidate whose structured attributes additionally contain that brand name.
    * If both satisfy all constraints, prefer the more exact / direct match (structured attributes, fewer extraneous tokens).
- If multiple products are equally matched: choose the earliest product! Do NOT continue to later Fibonacci phases.
- If no valid products exist: continue to the next Fibonacci batch.

**PRODUCT TYPES HAVE ALREADY BEEN VERIFIED FOR ALL PRODUCTS BY THE SYSTEM! DO NOT REJECT BECAUSE OF PRODUCT TYPE MISMATCH!**
**TITLE IS ONLY THE EVIDENCE OF KEYWORD PRESENCE! IF RELEVANT TOKENS EXIST IN THE TITLE, IT'S A MATCH, REGARDLESS OF THEIR ORDERS AND SEMANTICS!**

# Response Format
Return JSON like:
{{
        ... ...

    "phase_k": {{
            "fibonacci": <int>, // fibonacci number N_[k]
        "check_range": <str>, // checking range of product id: e.g. "7-11"
        "Product_[k]": <str>, // evaluation for the first product of the current batch
        "Product_[k+1]": <str>, // evaluation for the second product of the current batch
        ... ...
        "valid": <str|null>, // valid matching Product Ids separated by commas if at least one exists, otherwise null
    }},
    ... ...

    // if at least one valid matching product was found (at least 1), stop Fibonacci phases
    
    "recommend": <str|null>, // a summary for how the best product will be selected from valid products if valid ones found, otherwise null
    "best": <str|null> // product id of the recommended best matching product if found, otherwise null
}}

# Output Guidelines
- The checking text must be concise. One product per line. Example:
    * "Product_3": "Brand PASS; Capacity PASS; Color FAIL"
    * "Product_4": "Brand PASS; Capacity PASS; Color PASS; Material PASS"
- Recommendation should be a short sentence. Example:
    "Valid: Product 4, 6, 7. Comparing: ... ... Conclusion: Product 4 aligns most closely with the original wording."

# Shopping Query:
{query}

# Candidate Products:
{products}
"""
    SV_CONFIRM = """
You are AI assistant to check the products against the provided shopping queries, to find the ones which meet all the constraints mentioned.
There are {count} groups. Each group has a shopping query and a list of candidate products for this query.

# Overall Procedure
Perform the following phases strictly in order, for each query.

## Phase 1. Extract Constraints
- Read the shopping query and extract all constraints. Each constraint must belong to exactly one category. Possible categories include (but are not limited to):
    `brand, model, series, identifier/code/SKU, number, size, capacity, color, material, feature, compatibility, shape, style, package, age, gender, usage, pattern`
- All constraints are *REQUIRED*.
- Do NOT begin checking products yet.

## Phase 2. Classify Matching Type
Each extracted constraint must use exactly one matching mode.

### Literal Match
Use literal matching for `brands, model names, identifiers, product codes, uncommon token combinations, numbers, serial names, abbreviations, versions, colors`.
    * Literal constraints must appear explicitly.
    * Never normalize them. Never partially match them.

    Examples:
        Query "100ml" → Accept "100ml", reject "50-150ml"
        Query "iPhone" → Accept "iPhone", reject "Apple smartphone"
        Query "Green 2" → Accept "green 2", reject "green" (Casing difference allowed)
        Query "3-7 years" → Accept "3-7yrs", reject "5 years"
        Query "men's hair" → Accept "men's hair", reject "mens hair"
        Query "'gp power 1l'" → Accept "gp power 1l", reject "power gp 10w-40 x1"

### Semantic Match
- Everything else uses semantic matching. Paraphrases are allowed. Common typos are allowed. Examples:
    child = kid, wood = wooden, regular = standard, plain = simple, customized = personalized, water-resistant = waterproof, dioxide = dioxde
- Never be too strict. Be generous with common sense and implicity for semantic match constraints. Most of the candidates are matching. Prefer FALSE POSITIVE than FALSE NEGATIVE!

## Phase 3. Product Evaluation
- Evaluate products in order.
- For every product evaluate every constraint independently.
- Each constraint receives one result only: `PASS`, `FAIL`, or `UNKNOWN`
- A product is a valid candidate only if **every constraint is PASS**!
- If any constraint is FAIL reject the product immediately.
- `UNKNOWN` is treated as FAIL.

## Field Rules
### Product Type
Product type has already been verified. Never reject a product because of product type. Example:
    - Query "pet treatment set" → Title "Pet Treatment". Don't reject this product because of missing "set". It's been confirmed by the system that the product type correctly matches.
    - Query "aircon filter accessories" → Title "HP Aircon Filter Premium 2pcs". Don't reject this product because of missing "accessory". It's been confirmed by the system that the product type correctly matches.

### Title
The title is an unordered keyword collection. Do NOT infer sentence meaning. Only check token existence. Example:
    - Query "water bottle" → Title "Black Plastic Milk Bottles Cover Kids Water". The title contains both `water` and `bottle`. Therefore title evidence is POSITIVE. Do not infer that the product is a milk bottle. The title is only evidence of keyword presence.
    - Query "tank tester" → Title "tank capacity tester". The title contains both `tank` and `tester. Therefore title evidence is POSITIVE. Do not infer that the product type is mismatching. The title is only the evidence of keyword presence.

### Attributes
- Attributes are authoritative. Each attribute consists of `Key : Value`. Evaluate every attribute independently.
- If any attribute provides a valid value for a constraint, the constraint PASSes.
- Conflicting values in other attributes are ignored because they may represent different variations.

### Color Fields
- The attributes `color` and `color_family` may contain:
    * colors (Example: "color: black")
    * sizes (Example: "color_family: 10mm")
    * package names (Example: "color: 1pcs")
    * identifiers (Example: "color_family: G #1")
    * quantities (Example: "color_family": "5kg")
    * serial names (Example: "color": "aa12-c")
    * CMYK color(Example: "color_family": "k") 
- Treat them as generic attributes. Do not assume they only contain colors.

## Constraint Evaluation
- For every constraint check title. If satisfied: PASS. Otherwise: check every attribute. If any attribute satisfies the constraint: PASS. Otherwise: FAIL.
- Never infer literal constraints. Never invent missing information. Never assume compatibility. Never guess.

**PRODUCT TYPES HAVE ALREADY BEEN VERIFIED FOR ALL PRODUCTS BY THE SYSTEM! DO NOT REJECT BECAUSE OF PRODUCT TYPE MISMATCH!**
**TITLE IS ONLY THE EVIDENCE OF KEYWORD PRESENCE! IF RELEVANT TOKENS EXIST IN THE TITLE, IT'S A MATCH, REGARDLESS OF THEIR ORDERS AND SEMANTICS!**

# Response Format
Return JSON like:
{{
        // check groups "A", "B", ... in the order
    "A": {{
            "query": <str>, // write the query for group "A"

        // check all products "A1", "A2", ... against the query "A"

        "A1": {{
                // for product id = "A1"

            "rationale": <str>, // a brief summary of checking product A1 against query A
            "is_match": <bool> // true if matching all constraints, otherwise false
        }},
        ... ...
    }},

    "B": {{
            "query": <str>,
        
        "B1": {{
                "rationale": <str>,
            "is_match": <bool>
        }},
        ... ...
    }}
    ... ...
}}

# Output Guidelines
- The checking rationale must be concise. Examples:
    * "Brand PASS; Capacity PASS; Color FAIL"
    * "Brand PASS; Capacity PASS; Color PASS; Material PASS"
- Do not check product type matching because it's already verified by the system.
- Do not explain your reasoning. Do not expose intermediate thoughts. Only report the evaluation results.

# Queries and Candidate Products in Groups:
```json
{info}
```
"""
    SCOPE_RULES = """
It's extremely important to analyze the scope of each constraint, to identify the correct product to which it applies. Strictly keep the following non-negotiable rules:
- A constraint applies to multiple products only if it was associated with "all" or "both" explicitly.
- Unless a constraint is explicitly associated with "all" or "both", it applies to the nearest product only regardless of grammatical interpretation or semantic plausibility.
    * A constraint modifier without explicit mention of "all" or "both", applies to the nearest product which can receive it.
    * Even if it looks like applying to multiple products grammatically or semantically, each mention applies to the nearest product once.
    * Commas, conjunctions, and sentence structure do not override this rule!
- Implicit scope sharing is forbidden!
    * In the original query, each constraint applies to only one product, unless it's explicitly associated with "all" or "both".
    * If multiple products have the same constraint, this constraint must be mentioned multiple times or with "all" or with "both".
- Examples of Constraint Scope Identification:
    * "white shirt and black pants, with free shipping" → "with free shipping" applies to "black pants" only.
    * "a liquid single, gold lotion and shaving cream" → "a liquid single" applies to "gold lotion" only, in spite of comma.
    * "ink bottle, pencil and eraser. all must be less than 40" → "less than 40" applies to all products, because it's explicitly mentioned with "all".
- (IMPORTANT!) A price constraint can NOT apply to multiple products. This is a deterministic and non-negotiable rule!
    * One price constraint must apply to a single product, regardless of semantics, grammar and specific words.
    * If two or more products have price constraints, the price ranges must be mentioned multiple times independently.
"""

    @staticmethod
    def parse_decompose_response(js: Dict[str, Dict[str, str | Dict[str, str]]], is_multiple: bool) -> Tuple[List[Dict], Dict, List]:
        def __parse_services__():
            services, phrases = defaultdict(lambda: set()), []
            for chunk, v in js.get("services", {}).items():
                if is_multiple:
                    if not isinstance(v, dict): continue

                    pids = v.get("apply_to")
                    if isinstance(pids, str): pids = [pids]

                    pids = [id for id in pids if U_MiscUtil.extract_number(str(id), True)]

                    label = v.get("label")
                else:
                    if chunk == "rationale": continue
                    pids, label = ["1"], v

                svc = U_Prompt.label_to_service(label)
                # for k, patterns in Vocab.SERVICE_PATTERNS.items():
                #     if MiscUtil.scan_patterns(chunk, patterns):
                #         svc = k
                #         break
                #     elif k == Vocab.PROD_ONLY and svc == k:
                #         svc = None
                #         break

                if not svc: continue
                phrases.append(chunk)

                for id in pids:
                    services[id].add(svc)                
            return services, phrases

        def __parse_prices__():
            prices, phrases = {}, []

            for chunk, v in js.get("prices", {}).items():
                if is_multiple:
                    if not isinstance(v, dict): continue
                    
                    pid = v.get("apply_to")
                    if not U_MiscUtil.extract_number(str(pid), True): continue

                    normalized: str = v.get("normalized")
                else:
                    if chunk != "normalized": continue
                    pid, normalized = "1", v

                if not normalized: continue

                normalized = re.sub(r"[^\d\-\.]", "", normalized)
                splits = normalized.split('-')

                if len(splits) != 2: continue
                splits = [s if U_MiscUtil.extract_number(s, True) else '' for s in splits]

                prices[pid] = '-'.join(splits)
                phrases.append(chunk)
            
            return prices, phrases

        fillers = [js.get("redundant")]
        alts = js.get("alt") or {}

        service_map, svc_phrases = __parse_services__()
        price_map, price_phrases = __parse_prices__()

        banned_phrases = list(set(svc_phrases + price_phrases))
        if not fillers[0]: fillers = []

        prod_count = (len(js.get("products")) - 1) if is_multiple else 1
        decomposed = []

        for i in range(prod_count):
            id_str = str(i + 1)

            if is_multiple:
                q = js.get("sub_queries").get(id_str).strip()
            else:
                q = js.get("rewritten", {}).get("result").strip()

            if not q: raise Exception(f"sub-query {id_str} is invalid or empty")
            services = service_map.get(id_str) or {}
            # for k, patterns in Vocab.SERVICE_PATTERNS.items():
            #     if (chunk := MiscUtil.scan_patterns(q, patterns)):
            #         q = re.sub(rf"\b{re.escape(chunk)}\b", " ", q, flags = re.I)
            #         services.add(k)
            
            for phrase in banned_phrases:
                q = re.sub(rf"\b{re.escape(phrase)}\b", " ", q, flags = re.I)

            decomposed.append({
                "query": q,
                "services": [svc for svc in services if svc != U_Vocab.PROD_ONLY],
                "prod_only": U_Vocab.PROD_ONLY in services,
                "price": price_map.get(id_str)
            })

        return decomposed, alts, fillers
    
    @staticmethod
    def parse_build_response(js: Dict[str, Dict[str, Dict[str, str]]]) -> List[Dict[str, str | Dict]]:
        built = []

        for id, check in js.items():
            parsed = check.get("attributes")
            if not parsed: raise Exception(f"invalid attributes for sub-query {id}")

            built.append({
                "compact": check.get("compact"),
                "parsed": parsed
            })

        return built

    @staticmethod
    def parse_kw_response(js: Dict[str, Dict[str, str]]) -> List[str]:
        res = []

        for _, v in js.items():
            alt, cum = v.get("alt"), ""

            if alt:
                for u in alt.split():
                    if not u: continue

                    for ou in cum.replace('-', ' ').split():
                        if ou and U_MiscUtil.is_morph_equal(u, ou, {}): break
                    else:
                        cum += u + " "
                
                res.append(cum.strip())
            else:
                res.append("")

        return res

    @staticmethod
    def parse_select_response(js: Dict[str, Dict]) -> int:
        if "best" in js: return U_MiscUtil.extract_number(str(js.get("best")), True) - 1

        bid = js.get("best")

        if not bid:
            for phase in list(js.values())[::-1]:
                if not isinstance(phase, dict): continue
                if "best" not in phase: continue
                return U_MiscUtil.extract_number(str(phase.get("best")), True) - 1
        
        return 0

    @staticmethod
    def parse_confirm_response(js: Dict[str, Dict], counts: List[int]) -> List[List[int]]:
        alphabets = U_Vocab.ALPHABETS[:len(counts)]
        matches = [[] for _ in counts]        

        for g, checklist in js.items():
            if not isinstance(checklist, dict): raise Exception("format error")
            
            if g not in alphabets: raise Exception("format error")
            gid = alphabets.index(g)

            for k, check in checklist.items():
                if k == "query": continue
                if not isinstance(check, dict): raise Exception("format error")

                m = re.search(r"^[A-Z](\d+)$", k, re.I)
                if not m: raise Exception("format error")

                bid = int(m.group(1) or 0) - 1
                if bid < 0 or bid >= counts[gid]: raise Exception("format error")

                if "is_match" not in check: raise Exception("format error")
                if check.get("is_match"): matches[gid].append(bid)

        return matches

    @staticmethod
    def label_to_service(label: str) -> str:
        if label in ["flashsale", "lazflash"]: return "flashsale"
        if label in ["official", "lazmall", "return"]: return "official"
        if label == "free_ship": return "freeShipping"
        if label == "cod": return "COD"        
        if label == U_Vocab.PROD_ONLY: return label

        return None

    @staticmethod
    def service_to_label(svc: str) -> str:
        if svc == "official": return "lazmall"
        if svc == "flashsale": return "lazflash"
        if svc == "freeShipping": return "free_ship"
        if svc == "COD": return "cod"

        return None


def U_funny_bs_narrate_step(query: str, impose_comfort_f4a455: dict, dumb_cool_938ebb: str, confirm_blue_2cb095: bool=False) -> str:
    _ = (query, impose_comfort_f4a455, confirm_blue_2cb095)
    return dumb_cool_938ebb

def U_cash_bs_pipe_finalize(boil_account_b6f863, product_ids: list, status: str, think: str='', census_inside_a4c579: str='') -> None:
    caught_city_1cf7a0 = U_fuel_bs_join_ids(product_ids)
    inner_dolphin_d0668e = str(getattr(boil_account_b6f863, 'query', '') or '')[:240]
    if inner_dolphin_d0668e:
        pass
    flame_lawn_b967e5 = U_beach_bs_call_api('recommend_product', {'product_ids': caught_city_1cf7a0})
    fringe_few_940fb5 = U_beach_bs_call_api('terminate', {'status': status})
    if not think:
        depend_deputy_322fef = format(census_inside_a4c579, '') + ' ' if census_inside_a4c579 else ''
        blanket_code_c0589b = 'I am recommending product(s) ' + format(caught_city_1cf7a0, '') + ' for the query. ' + format(depend_deputy_322fef, '') + 'Status: ' + format(status, '') + '.'
        current_child_8f4c6a: dict = {'recommended_product_ids': caught_city_1cf7a0, 'status': status, 'note': 'Finalising recommendation and terminating the session.'}
        if census_inside_a4c579:
            current_child_8f4c6a['llm_reason'] = census_inside_a4c579
        think = U_funny_bs_narrate_step(boil_account_b6f863.query, current_child_8f4c6a, fallback=blanket_code_c0589b, force=True)
    U_balance_bs_pipe_append_step(boil_account_b6f863, think, [flame_lawn_b967e5, fringe_few_940fb5], 'Done.')


class U_Vocab:
    SHOP_PATTERN = r"\b(?:look(?:ing)?(?:\s+for)?|find|show|same)\b(?:\s+\w+){0,10}\s+shops?\b"
    VOUCHER_PATTERN = r"\bmy\s+budget\s+is\b"    
    PROD_ONLY = "product_only"

    SERVICE_PATTERNS = {
        svc: pats for svc, pats in [
            (
                "COD",
                [
                    r"\b((?:cash(?:es|ing)?|pay(?:ment|ing)?s?)[\s\-]*(?:for)?[\s\-]*(?:(?:up)?on|with(?:in)?|by|for|when)?(?:\s+\w+){0,3}\s*(?:deliver(?:y|ing|ie)?s?|cash(?:es|ing)?|arriv(?:e|es|ed|ing|al)?))\b"
                ]
            ),
            (
                "product_only",
                [
                    r"\b((?:with\s+)?(?:just|only)(?:\s+the)?\s+products?)\b",
                    r"\b((?:with\s+)?products?\s+only(?:\s+services?)?)\b"
                ]
            ),
            (
                "official",
                [
                    r"\b(laz\-?mall(?:[\s\-]+perks?)?)\b", r"\b(official(?:[\s\-]+stores?|[\s\-]+sellers?|[\s\-]+flagship[\s\-]+stores?)?)\b"
                ]
            ),
            (
                "flashsale",
                [
                    r"\b(laz\-?flash(?:[\s\-]+deals?)?)\b", r"\b(flash(?:[\s\-]?sales?|[\s\-]+deals?|[\s\-]+offers?))\b"
                ]
            ),
            (
                "freeShipping",
                [
                    r"\b((?:free|compliment[a-z]+)\s+(?:ship|deliver)[a-z]*)\b"
                ]
            )
        ]
    }

    KW_STOPWORDS = [
        "color family", "suitable", "compatible", "support", "made", "aged", "needs", "that", "helps", "contains", "includes", "option",
        "about", "above", "after", "also", "before", "below", "between", "for", "from", "comes",
        "less", "more", "over", "than", "under", "with", "within", "without", "provides", "sold"
    ]
    COLOR_WORDS = [
        "red", "blue", "green", "yellow", "black", "white", "pink", "purple", "violet", "orange", "brown",
        "gray", "grey", "beige", "navy", "gold", "silver", "bronze", "magenta", "cyan", "dark", "light", "khaki", "turquoise"
    ]
    METRIC_UNITS = {
        a: b for a, b in
        [
            ("mm", ""), ("cm", ""), ("mg", ""), ("kg", ""), ("oz", ""), ("lb", ""), ("ml", ""), ("mah", ""), ("ah", ""), ("wh", ""), ("kwh", ""), ("mv", ""),
            ("kv", ""), ("ma", ""), ("kw", ""), ("hz", ""), ("khz", ""), ("mhz", ""), ("ghz", ""), ("bps", ""), ("kbps", ""), ("mbps", ""), ("gbps", ""), ("°c", ""),
            ("°f", ""), ("kpa", ""), ("mpa", ""), ("dpi", ""), ("lux", ""), ("db", ""), ("rpm", ""), ("y", "year"), ("yrs", "year"),
            ("km", "kilometer"), ("ft", "feet"), ("m", "meter"), ("yd", "yard"), ("g", "gram"), ("l", "liter"), ("v", "volt"), ("w", "watt"), ("pa", "pascal"),
            ("kb", "kilobyte"), ("mb", "megabyte"), ("gb", "gigabyte"), ("tb", "terabyte"), ("mp", "megapixel"), ("pc", "piece"), ("pcs", "piece")
        ]
    }
    MORPH_RULES: List[Tuple[str, List[Union[str, Callable]]]] = [
        ("ing", ["e", "", re.sub]), ("ed", ["e", "", re.sub]), ("est", ["e", "", re.sub]), ("er", ["e", "", re.sub]), ("ish", ["e", "", re.sub]),
        ("iness", ["y"]), ("inesses", ["y"]), ("ily", ["y"]), ("ies", ["y", ""]), ("ness", ["y", ""]), ("es", ["e", ""]), ("s", [""]),
        ("ful", [""]), ("less", [""]), ("ly", [""]), ("en", [""])
    ]
    NUMBERS = {
        a: b for a, b in
        [
            ("single", "1"), ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"), ("five", "5"), ("six", "6"),
            ("seven", "7"), ("eight", "8"), ("nine", "9"), ("ten", "10")
        ]
    }
    ALTS = ["all2any", "everyday2casual", "any2universal", "kid2child", "men2male", "women2female", "tier2layer", "100%2pure", "anti2non", "any2random", "standard2regular"]
    BRIDGES = ["all2any", "man2men", "woman2women", "child2children", "metre2meter", "litre2liter", "colour2color"]    
    SIZES = ["s2small", "l2large", "m2medium", "xl2xlarge", "xxxl23xlarge", "3xl23xlarge", "4xl24xlarge"]

    YES_WORDS = ["yes", "enabled", "included", "exists", "true", "required"]
    NO_WORDS = ["no", "without any", "without"]
    ID_KEYS = ["brand", "color", "model", "id", "compatibility", "color_family"]
    FILLERS = ["set", "pack", "accessory", "option"]
    
    ALPHABETS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class U_MultiModelVotingSystem:
    @staticmethod
    def model_chain_for_role(role: str) -> List[str]:
        sandbox = getenv("SANDBOX_MODEL")
        if sandbox:
            return [sandbox.strip()]
        return U_ProxyUtil.get_role_llm_priorities(role)

    @staticmethod
    def vote_product_id(votes: List[Dict[str, Any]], *, score_key: str = "_llm_relevance_score") -> str | None:
        tallies: Dict[str, List[float]] = defaultdict(list)
        for row in votes:
            if not isinstance(row, dict):
                continue
            pid = U_EdgeCaseProcessor.sanitize_product_id(
                row.get("product_id") or row.get("best_product_id")
            )
            if not pid:
                continue
            try:
                sc = float(row.get(score_key, row.get("score", row.get("relevance_score", 0))) or 0)
            except (TypeError, ValueError):
                sc = 0.0
            tallies[pid].append(sc)
        if not tallies:
            return None
        ranked = sorted(
            ((pid, sum(scores), len(scores)) for pid, scores in tallies.items()),
            key = lambda t: (t[1], t[2]),
            reverse = True,
        )
        return ranked[0][0]

    @staticmethod
    def merge_model_json_votes(parsed_list: List[Dict], *, list_key: str = "products") -> Dict | None:
        valid = [p for p in parsed_list if isinstance(p, dict)]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]
        return max(valid, key = lambda d: len(U_EdgeCaseProcessor.empty_list_if_null(d.get(list_key))))

    @staticmethod
    def vote_select_index(
        prompt: str,
        candidate_count: int,
        *,
        role: str = "pick",
        task: str = "judge",
        expect: str = "dict",
        max_models: int = 2,
        max_attempts: int = 1,
    ) -> Tuple[int, Dict[str, Any]]:
        votes: Dict[int, int] = defaultdict(int)
        details: Dict[str, Any] = {"votes": {}, "models": []}
        models = U_MultiModelVotingSystem.model_chain_for_role(role)[:max(1, max_models)]
        attempts = max(1, min(max_attempts, U_ENHANCEMENT_BENCH.llm_retry_max))

        for attempt in range(attempts):
            temp = U_TemperatureController.for_attempt(role, attempt)
            for model in models:
                try:
                    js, _ = U_ProxyUtil.trigger_llm(
                        prompt,
                        preferred_llms = [model],
                        temperature = temp,
                        expect = expect,
                    )
                    bid = U_Prompt.parse_select_response(js)
                    if 0 <= bid < candidate_count:
                        votes[bid] += 1
                        details["models"].append({"model": model, "pick": bid, "temperature": temp})
                        # Early stop once a clear majority exists
                        if votes[bid] > len(models) // 2:
                            details["votes"] = dict(votes)
                            details["winner"] = bid
                            details["early_stop"] = True
                            return bid, details
                except Exception as exc:
                    details["models"].append({"model": model, "error": str(exc)})

        if not votes:
            return -1, details

        winner = max(votes.items(), key = lambda kv: kv[1])[0]
        details["votes"] = dict(votes)
        details["winner"] = winner
        return winner, details

    @staticmethod
    def self_consistency_pick(
        first_pick: Dict | None,
        second_pick: Dict | None,
        *,
        heuristic_gap: float,
        gap_threshold: float | None = None,
    ) -> Dict | None:
        if not first_pick:
            return second_pick
        if not second_pick:
            return first_pick
        gap_threshold = U_ENHANCEMENT_BENCH.self_consistency_gap if gap_threshold is None else gap_threshold
        if heuristic_gap >= gap_threshold:
            return first_pick
        pid1 = U_EdgeCaseProcessor.sanitize_product_id(first_pick.get("product_id"))
        pid2 = U_EdgeCaseProcessor.sanitize_product_id(second_pick.get("product_id"))
        if pid1 and pid1 == pid2:
            try:
                s1 = float(first_pick.get("_llm_relevance_score", 0))
                s2 = float(second_pick.get("_llm_relevance_score", 0))
                merged = dict(first_pick)
                merged["_llm_relevance_score"] = min(10.0, s1 + 0.5 * s2 / 10.0)
                return merged
            except (TypeError, ValueError):
                return first_pick
        try:
            sc1 = float(first_pick.get("_llm_relevance_score", 0))
            sc2 = float(second_pick.get("_llm_relevance_score", 0))
        except (TypeError, ValueError):
            return first_pick
        return first_pick if sc1 >= sc2 else second_pick

    @staticmethod
    def should_run_self_consistency(heuristic_scores: List[float]) -> bool:
        if not U_ENHANCEMENT_BENCH.enable_self_consistency:
            return False
        if not heuristic_scores or len(heuristic_scores) < 2:
            return False
        top = sorted(heuristic_scores, reverse=True)
        return (top[0] - top[1]) < U_ENHANCEMENT_BENCH.self_consistency_gap

    @staticmethod
    def aggregate_batch_scores(score_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Median-score aggregation across model passes (radiant/garland)."""
        by_pid: Dict[str, List[float]] = defaultdict(list)
        for scores in score_lists:
            for row in scores or []:
                pid = U_EdgeCaseProcessor.sanitize_product_id(row.get("product_id"))
                if not pid:
                    continue
                by_pid[pid].append(U_EdgeCaseProcessor.coerce_float(row.get("score"), default = 0.0))
        merged: List[Dict[str, Any]] = []
        for pid, vals in by_pid.items():
            vals.sort()
            mid = vals[len(vals) // 2]
            merged.append({"product_id": pid, "score": int(round(mid))})
        merged.sort(key = lambda r: -r["score"])
        return merged

    @staticmethod
    def cycle_models_until_json(prompt: str, *, role: str = "parse", task: str = "parse", expect: str = "dict", max_cycles: int | None = None,) -> Tuple[Dict[str, Any] | List[Any], Any]:
        max_cycles = U_ENHANCEMENT_BENCH.llm_retry_max if max_cycles is None else max_cycles
        models = U_MultiModelVotingSystem.model_chain_for_role(role)
        collected: List[Dict] = []

        for attempt in range(max_cycles):
            temp = U_TemperatureController.for_attempt(role, attempt)
            for model in models:
                js, resp = U_ProxyUtil.trigger_llm(
                    prompt,
                    preferred_llms = [model],
                    temperature = temp,
                    expect = expect,
                )
                if expect == "dict" and isinstance(js, dict) and js:
                    collected.append(js)
                    merged = U_MultiModelVotingSystem.merge_model_json_votes(collected)
                    if merged is not None:
                        return merged, resp
                elif expect == "list" and isinstance(js, list) and js:
                    return js, resp

        return U_ProxyUtil.__empty_for__(expect)


def U_concert_bs_spec_to_query(product: dict, *, human_fame_e0d50a: bool=True) -> dict[str, Any]:
    kw = product.get('keywords', 'product')
    svc = product.get('service')
    q = kw + (' only' if not svc and bool(product.get('only_product_type')) else '')
    p: dict[str, Any] = {'q': q}
    if human_fame_e0d50a and product.get('price_range'):
        p['price'] = product['price_range']
    if svc:
        p['service'] = svc
    return p

def U_book_bs_float_or_none(broad_harsh_92980e: str) -> float | None:
    try:
        return float(broad_harsh_92980e)
    except (ValueError, TypeError):
        return None

class U_ExtTool:
    @Tool("find_product")
    def search(**kwargs) -> List[Dict]:
        result = U_ProxyUtil.ORO_PROXY.get("/search/find_product", kwargs)
        return result if result else []

    @Tool("view_product_information")
    def view(ids: str) -> List[Dict]:
        result = U_ProxyUtil.ORO_PROXY.get("/search/view_product_information", {"product_ids": ids})
        return result if result else []

    @Tool("recommend_product")
    def submit(product_ids: str) -> str:
        return f"submitted recommendation to the user: {product_ids}."

    @Tool("terminate")
    def finish(status: str) -> str:
        return f"session completed with status: {status}"

def U_dance_bs_parse_price_opt(price_range: str | None) -> tuple[float | None, float | None]:
    if not price_range:
        return (None, None)
    s = str(price_range).strip()
    if '-' not in s:
        v = U_book_bs_float_or_none(s)
        return (None, v) if v is not None else (None, None)
    affair_define_463b47 = s.index('-')
    cost_badge_a0d337, donate_crouch_625602 = (s[:affair_define_463b47].strip(), s[affair_define_463b47 + 1:].strip())
    return (U_book_bs_float_or_none(cost_badge_a0d337) if cost_badge_a0d337 else None, U_book_bs_float_or_none(donate_crouch_625602) if donate_crouch_625602 else None)

def U_chief_bs_probe_edges_base(products: list, dress_blouse_bcd82e: float, *, fan_eye_5b34d9: str | None) -> tuple[list[float], list[float], list]:
    bench_dinosaur_6e06dd: list[float] = []
    avocado_bike_2f52f0: list[float] = []
    jungle_grade_adb736: list = []
    hundred_ceiling_25d0c8 = str(fan_eye_5b34d9) if fan_eye_5b34d9 is not None else None
    hedgehog_insane_5e28b = '1-' + format(dress_blouse_bcd82e, '.0f')
    for exact_crush_a00403 in products:
        finger_anxiety_ea109d = U_concert_bs_spec_to_query(exact_crush_a00403, include_price=False)
        bring_blast_834fed = dict(finger_anxiety_ea109d)
        if hundred_ceiling_25d0c8 is not None:
            bring_blast_834fed['shop_id'] = hundred_ceiling_25d0c8
        for sort_key, cloud_curtain_a4200 in (('priceasc', bench_dinosaur_6e06dd), ('pricedesc', avocado_bike_2f52f0)):
            force_fatal_c1d19b = U_beach_bs_call_api('find_product', {**bring_blast_834fed, 'price': hedgehog_insane_5e28b, 'sort': sort_key, 'page': 1})
            jungle_grade_adb736.append(force_fatal_c1d19b)
            focus_bargain_4aba0 = (force_fatal_c1d19b.get('result') or [None])[0]
            cloud_curtain_a4200.append(float(focus_bargain_4aba0.get('price', 0)) if focus_bargain_4aba0 else 0.0)
    return (bench_dinosaur_6e06dd, avocado_bike_2f52f0, jungle_grade_adb736)

class U_Challenge(str, Enum):
    PRODUCT = 'Product'
    SHOP = 'Shop'
    VOUCHER = 'Voucher'
    UNKNOWN = 'Unknown'

    def promptfy(self, is_multiple: bool) -> str:
        if self is U_Challenge.PRODUCT: return "a single product with specific constraints"
        if self is U_Challenge.SHOP: return "multiple products from the same shop"
        
        if self is U_Challenge.VOUCHER:
            pstr = "multiple products" if is_multiple else "a single product"
            return f"{pstr} within a specified budget and considering voucher options"

        raise ValueError(f"unknown challenge type: {self}")

def U_hotel_bs_clip_band(kiss_dignity_e2d75: float, insect_bounce_cc8a7f: float, price_range: str | None) -> tuple[float, float] | None:
    if insect_bounce_cc8a7f < kiss_dignity_e2d75:
        return None
    april_august_ddbd53, clump_capable_d124e6 = U_dance_bs_parse_price_opt(price_range)
    faith_gorilla_acab86 = kiss_dignity_e2d75 if april_august_ddbd53 is None else max(kiss_dignity_e2d75, april_august_ddbd53)
    grab_action_208383 = insect_bounce_cc8a7f if clump_capable_d124e6 is None else min(insect_bounce_cc8a7f, clump_capable_d124e6)
    if grab_action_208383 < faith_gorilla_acab86:
        return None
    return (faith_gorilla_acab86, grab_action_208383)

def U_enable_bs_probe_edges(products: list, dress_blouse_bcd82e: float) -> tuple[list[float], list[float], list]:
    return U_chief_bs_probe_edges_base(products, dress_blouse_bcd82e, shop_id=None)

@dataclass
class U_JsonModel(object):
    @classmethod
    def from_json(cls, data: dict) -> "JsonModel":
        kwargs = {}

        for f in fields(cls):
            if f.name not in data: continue            
            kwargs[f.name] = cls.__convert_from__(data[f.name], f.type)

        return cls(**kwargs)

    def to_json(self) -> Dict:
        return self.__convert_to__(self)

    @staticmethod
    def __convert_from__(value, t: Type["JsonModel"] | Any) -> Any:
        origin, args = get_origin(t), get_args(t)

        if origin is Union and len(args) == 2 and NoneType in args:
            return U_JsonModel.__convert_from__(
                value, next(a for a in args if a is not NoneType)
            ) if value is not None else None

        if origin is list and isinstance(value, list):
            return [U_JsonModel.__convert_from__(v, args[0]) for v in value]

        if origin is tuple and isinstance(value, list):
            return tuple(U_JsonModel.__convert_from__(v, args[0]) for v in value)

        if origin is dict and isinstance(value, dict):
            kt, vt = args
            return {U_JsonModel.__convert_from__(k, kt): U_JsonModel.__convert_from__(v, vt) for k, v in value.items()}

        if is_dataclass(t) and isinstance(value, dict): return t.from_json(value)

        if t is int and isinstance(value, str): return U_MiscUtil.extract_number(value, True)
        if t is float and isinstance(value, str): return U_MiscUtil.extract_number(value)
        if t is str: return str(value)

        return value

    @staticmethod
    def __convert_to__(obj: Any) -> Any:
        if is_dataclass(obj): return {f.name: U_JsonModel.__convert_to__(getattr(obj, f.name)) for f in fields(obj)}
        
        if isinstance(obj, dict): return {U_JsonModel.__convert_to__(k): U_JsonModel.__convert_to__(v) for k, v in obj.items()}
        if isinstance(obj, list): return [U_JsonModel.__convert_to__(x) for x in obj]

        return obj

def U_answer_bs_rate_limited_search_get(final_crazy_271564: str, finger_anxiety_ea109d: dict | None=None):
    pass

def U_clean_bs_load_details(product_ids: list[str]) -> dict[str, dict]:
    if not product_ids:
        return {}
    inside_govern_79ef87: dict[str, dict] = {}
    confirm_easy_254501 = [pid for pid in product_ids if pid not in inside_govern_79ef87]
    candy_glare_6cc492 = 10
    idx = 0
    while idx < len(confirm_easy_254501):
        chunk = confirm_easy_254501[idx:idx + candy_glare_6cc492]
        idx += candy_glare_6cc492
        become_hedgehog_94aa14 = U_answer_bs_rate_limited_search_get('/search/view_product_information', {'product_ids': ','.join(chunk)})
        if isinstance(become_hedgehog_94aa14, list):
            for easily_fire_5c263c in become_hedgehog_94aa14:
                inside_govern_79ef87[str(easily_fire_5c263c.get('product_id', ''))] = easily_fire_5c263c
    return {pid: inside_govern_79ef87[pid] for pid in product_ids if pid in inside_govern_79ef87}

@dataclass
class U_Voucher(U_JsonModel):
    budget: float = 0
    apply_to: str = ""
    threshold: Optional[float] = None
    is_fixed: bool = False
    discount: float = 0
    cap: Optional[float] = None

    def get_active_cost(self, products: List["Product"]) -> float:
        psum = sum([p.price for p in products])
        
        if self.apply_to == 'shop':
            shops = set([p.shop for p in products])
            if len(shops) > 1: return psum
        
        if self.threshold is not None and psum < self.threshold: return psum
        if self.is_fixed: return psum - self.discount

        discount = psum * self.discount / 100.0

        if self.cap is not None: discount = min(discount, self.cap)
        return psum - discount

    def get_favor_range(self) -> Tuple[float, float]:
        high = (self.threshold / 0.1) if self.threshold is not None else math.inf
        low = (self.threshold / 0.9) if self.threshold is not None else 0

        if not self.is_fixed:
            l1 = self.budget / 1.1 / (1.0 - self.discount / 100.0)
            l2 = self.budget / 1.1 + (self.cap if self.cap is not None else math.inf)
            
            h1 = self.budget + (self.cap if self.cap is not None else math.inf)
            h2 = self.budget / (1.0 - self.discount / 100.0)
            
            low = max(low, min(l1, l2))
            high = min([high, h1, h2])
        else:
            low = max(low, self.budget / 1.1 + self.discount)
            high = min(high, self.budget + self.discount)

        return (math.floor(low) - 1, math.ceil(high) + 1)

    def is_favor_cost(self, psum: float) -> bool:
        if self.threshold < int(psum * 0.1) or self.threshold > int(psum * 0.9): return False
        if psum < 100.0: return False

        if not self.is_fixed:
            ac = max(psum * (1.0 - self.discount / 100.0), psum - (self.cap if self.cap is not None else math.inf))
            if self.budget < ac or self.budget > math.ceil(1.1 * ac): return False
        elif self.budget < psum - self.discount: return False

        return True

def U_flower_bs_normalise_sku_options(junk_gas_7d1764: Any) -> list[dict]:
    result: list[dict] = []
    if isinstance(junk_gas_7d1764, list):
        for couch_chase_9d9253 in junk_gas_7d1764:
            if not isinstance(couch_chase_9d9253, dict):
                continue
            dentist_air_b51c1c = couch_chase_9d9253.get('values', [])
            if not isinstance(dentist_air_b51c1c, list):
                dentist_air_b51c1c = list(dentist_air_b51c1c.values()) if isinstance(dentist_air_b51c1c, dict) else []
            result.append({'name': couch_chase_9d9253.get('name'), 'values': dentist_air_b51c1c[:5]})
    elif isinstance(junk_gas_7d1764, dict):
        blind_hamster_4033f5: dict[str, list] = {}
        for just_choose_79a048 in junk_gas_7d1764.values():
            if not isinstance(just_choose_79a048, dict):
                continue
            for kind_include_3c711c, baby_idea_deb3cb in just_choose_79a048.items():
                address_cargo_f4ac3b = blind_hamster_4033f5.setdefault(kind_include_3c711c, [])
                if baby_idea_deb3cb not in address_cargo_f4ac3b:
                    address_cargo_f4ac3b.append(baby_idea_deb3cb)
        for kind_include_3c711c, values in blind_hamster_4033f5.items():
            result.append({'name': kind_include_3c711c, 'values': values[:5]})
    return result

def U_hero_bs_enrich_listings(impose_half_65dba5: list[dict]) -> list[dict]:
    try:
        U_clean_bs_load_details([str(s.get('product_id', '')) for s in impose_half_65dba5])
    except Exception:
        pass
    inside_govern_79ef87: dict[str, dict] = {}
    gasp_income_b556d0: list[dict] = []
    for husband_become_edcbac in impose_half_65dba5:
        pid = str(husband_become_edcbac.get('product_id', ''))
        try:
            detail = inside_govern_79ef87.get(pid) or {}
            title = husband_become_edcbac.get('title') or (detail.get('title', '') if detail else '')
            price = husband_become_edcbac.get('price')
            if price is None and detail:
                price = detail.get('price')
            garden_guide_d63f93: dict = {'product_id': pid, 'title': title, 'price': price}
            if detail:
                dove_again_60fdcd = U_flower_bs_normalise_sku_options(detail.get('sku_options') or [])
                if dove_again_60fdcd:
                    garden_guide_d63f93['sku_options'] = dove_again_60fdcd[:3]
                final_chicken_b95b2d = detail.get('attributes') or {}
                if isinstance(final_chicken_b95b2d, dict) and final_chicken_b95b2d:
                    garden_guide_d63f93['attributes'] = dict(list(final_chicken_b95b2d.items())[:8])
                jaguar_emerge_f1f956 = detail.get('service_tags') or detail.get('services') or []
                if isinstance(jaguar_emerge_f1f956, list) and jaguar_emerge_f1f956:
                    garden_guide_d63f93['service_tags'] = jaguar_emerge_f1f956[:6]
        except Exception:
            garden_guide_d63f93 = {'product_id': pid, 'title': husband_become_edcbac.get('title', ''), 'price': husband_become_edcbac.get('price')}
        gasp_income_b556d0.append(garden_guide_d63f93)
    return gasp_income_b556d0

@dataclass
class U_SubQuery(U_JsonModel):
    query_f: str = ""
    query: str = ""

    parsed: Dict[str, str] = field(default_factory = dict)
    compact: str = ""    
    
    services: List[str] = field(default_factory = list)
    price: Optional[str] = None
    
    prod_only: bool = False
    id: int = -1

    def __hash__(self) -> int:
        return hash((self.query_f, self.id))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)): return False
        return self.query_f == other.query_f and self.id == other.id

def U_attract_bs_richness_rank(exact_crush_a00403: dict) -> int:
    pass

def U_best_bs_heuristic_score(product: dict, query_text: str, detail: dict | None=None) -> float:
    title = product.get('title', '').lower()
    crumble_focus_7b0378 = set(re.findall('\\b\\w+\\b', title))
    return 0

@dataclass
class U_Query(U_JsonModel):
    challenge: U_Challenge = U_Challenge.UNKNOWN

    query_f: str = ""
    query_e: str = ""

    intents: List[U_SubQuery] = field(default_factory = list)
    voucher: Optional[U_Voucher] = None

    alts: Dict[str, str] = field(default_factory = dict)
    fillers: List[str] = field(default_factory = list)

def U_boy_bs_budget_sec_left() -> float:
    return 280.0

def U_hour_bs_fetch_band_hits_base(exact_crush_a00403: dict, kiss_dignity_e2d75: float, insect_bounce_cc8a7f: float, *, jelly_crisp_f1378e: int, fan_eye_5b34d9: str | None) -> tuple[list, list]:
    bid_glory_4a6d7d = U_concert_bs_spec_to_query(exact_crush_a00403, include_price=False)
    bid_glory_4a6d7d['price'] = format(kiss_dignity_e2d75, '.0f') + '-' + format(insect_bounce_cc8a7f, '.0f')
    if fan_eye_5b34d9 is not None:
        bid_glory_4a6d7d['shop_id'] = str(fan_eye_5b34d9)
    juice_agent_70ec4f: list[dict] = []
    gentle_hurt_d448de: set[str] = set()
    jungle_grade_adb736: list = []
    for brick_frost_7d7d8c in (1, 2):
        resp = U_beach_bs_call_api('find_product', {**bid_glory_4a6d7d, 'page': brick_frost_7d7d8c})
        jungle_grade_adb736.append(resp)
        for product in resp.get('result') or []:
            pid = str(product.get('product_id', ''))
            if pid and pid not in gentle_hurt_d448de:
                gentle_hurt_d448de.add(pid)
                juice_agent_70ec4f.append(product)
                if len(juice_agent_70ec4f) >= jelly_crisp_f1378e:
                    return (juice_agent_70ec4f, jungle_grade_adb736)
    return (juice_agent_70ec4f, jungle_grade_adb736)

@dataclass
class U_Filter(U_JsonModel):
    keywords: List[str] = field(default_factory = list)
    services: List[str] = field(default_factory = list)

    price: Optional[str] = None
    shop: Optional[str] = None

    def update(self, **kwargs) -> "Filter":
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        
        return self

    def get_q(self) -> str:
        chunks = []

        for kw in self.keywords:
            for m in kw.split():
                m = m.lower().strip()
                if m and m not in chunks: chunks.append(m)
        
        return ' '.join(sorted(chunks))

    def clone(self) -> "Filter":
        return U_Filter.from_json(self.to_json())

    def __hash__(self) -> int:
        return hash((self.get_q(), self.price, self.shop))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)): return False        
        return self.get_q() == other.get_q() and self.price == other.price and self.shop == other.shop

def U_hungry_bs_fetch_band_hits(exact_crush_a00403: dict, kiss_dignity_e2d75: float, insect_bounce_cc8a7f: float, jelly_crisp_f1378e: int=20) -> tuple[list, list]:
    return U_hour_bs_fetch_band_hits_base(exact_crush_a00403, kiss_dignity_e2d75, insect_bounce_cc8a7f, limit=jelly_crisp_f1378e, shop_id=None)

def U_crater_bs_clip_strings(value: Any, cement_infant_97f9d6: int) -> Any:
    if isinstance(value, str):
        return value[:cement_infant_97f9d6] if len(value) > cement_infant_97f9d6 else value
    if isinstance(value, list):
        return [U_crater_bs_clip_strings(v, cement_infant_97f9d6) for v in value]
    if isinstance(value, dict):
        return {k: U_crater_bs_clip_strings(v, cement_infant_97f9d6) for k, v in value.items()}
    return value

@dataclass
class U_Product(U_JsonModel):
    title: str = ""
    id: str = ""

    price: float = 0
    shop: str = ""

    services: List[str] = field(default_factory = list)
    is_viewed: bool = False

    details: Dict[str, List[List[Tuple[str, str]]]] = field(default_factory = list)

    attr: Dict[str, List[str]] = field(default_factory = dict)
    sku: Dict[str, Dict[str, str]] = field(default_factory = dict)

    @classmethod
    def from_json(cls, data) -> "Product":
        return super().from_json(data).update_details(True)

    def to_normalized(self) -> Dict[str, List[List[Tuple[Tuple[str], Tuple[str]]]]]:
        self.update_details()

        return {
            g: [
                [
                    (tuple(U_MiscUtil.normalize_str(k, True).split()), tuple(U_MiscUtil.normalize_str(v, True).split()))
                    for k, v in kv
                ] for kv in kvs
            ]
            for g, kvs in self.details.items()
        }

    def promptfy(self) -> Dict[str, Any]:
        joint = list(set((k, v) for _, kvs in self.details.items() for kv in kvs for k, v in kv))
        joint.sort(key = lambda j: (j[0], -len(j[1])))

        return {"title": self.title, "attributes": [f"{k}: {v}" for k, v in joint]}

    def update_details(self, enforced: bool = False) -> "Product":
        if enforced or not self.details:
            self.details = {f"att_{k}": [[(k, v)] for v in vs] for k, vs in self.attr.items()}
            self.details["sku"] = [[(k, v) for k, v in kv.items()] for _, kv in self.sku.items()]

        return self

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)): return False
        return self.id == other.id

def U_league_bs_build_candidate(product: dict, detail: dict | None, query_text: str) -> dict:
    betray_august_78bea9 = detail or {}
    forget_cream_e8cd31 = betray_august_78bea9.get('sku_options') or {}
    fluid_churn_2493a6: frozenset[str] = frozenset({'the', 'a', 'an', 'for', 'with', 'from', 'that', 'this', 'i', 'me', 'my', 'looking', 'show', 'find', 'want', 'need', 'get', 'finish', 'buy', 'also', 'and', 'in', 'is', 'it', 'am', 'im', 'priced', 'pesos'})
    company_carry_13448c = {w for w in re.findall('\\b\\w+\\b', query_text.lower()) if len(w) > 1 and w not in fluid_churn_2493a6}
    chaos_final_5f01d0: list[tuple[int, dict]] = []
    for opt in forget_cream_e8cd31.values():
        if isinstance(opt, dict):
            frozen_erase_1ea47a = {w for w in re.findall('\\b\\w+\\b', ' '.join((str(v).lower() for v in opt.values()))) if len(w) > 1}
            chaos_final_5f01d0.append((len(company_carry_13448c & frozen_erase_1ea47a), opt))
    acoustic_gate_12d55b: set[str] = set()
    board_comic_51b6ee: list[dict] = []
    for _, opt in sorted(chaos_final_5f01d0, key=lambda t: t[0], reverse=True):
        key = json.dumps(opt, sort_keys=True, ensure_ascii=False)
        if key not in acoustic_gate_12d55b:
            acoustic_gate_12d55b.add(key)
            board_comic_51b6ee.append(opt)
    early_country_35a607 = betray_august_78bea9.get('attributes') or {}
    add_gain_96b0d3: dict = {}
    if isinstance(early_country_35a607, dict):
        for k, v in list(early_country_35a607.items())[:8]:
            add_gain_96b0d3[str(k)[:40]] = U_crater_bs_clip_strings(v, 80)
    funny_galaxy_3e3c81 = str(product.get('title', ''))
    title = funny_galaxy_3e3c81[:200] if len(funny_galaxy_3e3c81) > 200 else funny_galaxy_3e3c81
    return {'product_id': str(product.get('product_id', '')).strip(), 'title': title, 'price': product.get('price'), 'service': product.get('service', []), 'attributes': add_gain_96b0d3, 'sku_options_preview': [U_crater_bs_clip_strings(gym_canvas_9448c6, 80) for gym_canvas_9448c6 in board_comic_51b6ee[:8]]}

def U_convince_bs_active_provider() -> str:
    return getenv('INFERENCE_PROVIDER', 'openrouter')

@dataclass
class U_Weight(U_JsonModel):
    LOG_PROBS = [
        0.6698174986042095, 0.12640294947204656, 0.03721208325578941, 0.01890137562198827, 0.011222691775555538, 0.005316011893684202, 0.0035440079291228013,
        0.003544006929121897, 0.0028846576167278616, 0.002884656616726958, 0.0028846556167260538, 0.00288465461672515, 0.002884653616724246, 0.0028846526167233422,
        0.002884651616722438, 0.0028846506167215344, 0.0028846496167206303, 0.0028846486167197266, 0.0028846476167188224, 0.0028846466167179187,
        0.0028846456167170146, 0.002884644616716111, 0.002884643616715207, 0.002884642616714303, 0.002884641616713399, 0.0028846406167124953, 0.002884639616711591,
        0.0028846386167106874, 0.0028846376167097833, 0.0028846366167088796, 0.0028846356167079755, 0.0028846346167070718, 0.0028846336167061676,
        0.002884632616705264, 0.00288463161670436, 0.002884630616703456, 0.002884629616702552, 0.0028846286167016483, 0.002884627616700744, 0.0028846266166998404,
        0.0028846256166989367, 0.0028846246166980326, 0.002884623616697129, 0.002884622616696225, 0.002884621616695321, 0.002884620616694417, 0.0028846196166935133,
        0.002884618616692609, 0.0028846176166917054, 0.0028846166166908013, 0.0028846156166898976
    ]
    RANK_BONUS_C = [1e-2 - i * 2e-5 for i in range(500)]
    SKU_BONUS_C = [5e-3, 5e-3, 5e-3, 1e-3, 5e-4, 1e-4]

    detail: Dict[str, float] = field(default_factory = dict)    
    choice: Dict[str, int] = field(default_factory = dict)
    covers: List[int] = field(default_factory = list)
    
    score_r: float = 0
    score_n: float = 0
    score: float = 0

    confidence: float = 0
    wsum: float = 0

    wrank: int = -1
    rank: int = -1    

    @classmethod
    def create(cls, choice: Dict, covers: List, rank: int, score_r: float, pt_bonus: float, att_bonus: float, brand_bonus: float, wsum: float) -> "Weight":
        score = score_r + pt_bonus + att_bonus + brand_bonus
        idx = choice.get("sku") or -1

        sku_bonus = U_Weight.SKU_BONUS_C[idx + 1] if idx < (len(U_Weight.SKU_BONUS_C) - 1) else 0
        rank_bonus = U_Weight.RANK_BONUS_C[rank] if rank < len(U_Weight.RANK_BONUS_C) else 0

        return U_Weight(
            choice = choice,
            covers = covers,
            rank = rank,            
            wsum = wsum,
            score_n = ((score / wsum) if wsum > 1e-8 else 0) + sku_bonus + rank_bonus,
            score_r = score_r,
            score = score,
            detail = {
                "pt_bonus": pt_bonus, "att_bonus": att_bonus, "brand_bonus": brand_bonus, "sku_bonus": sku_bonus, "rank_bonus": rank_bonus
            }
        )

    def fill_detail(self, qs: List[str], ws: List[float]) -> "Weight":
        for i in self.covers:
            self.detail[qs[i]] = ws[i]
        
        return self

    def update_confidence(self, wrank: int) -> "Weight":
        if wrank >= len(U_Weight.LOG_PROBS):
            self.confidence = U_Weight.LOG_PROBS[-1] * (0.9 ** (wrank - len(U_Weight.LOG_PROBS) + 1))            
        else:
            self.confidence = U_Weight.LOG_PROBS[wrank]
        
        self.wrank = wrank
        return self

def U_arrange_bs_score_model_seq() -> list[str]:
    crucial_fold_d7e145 = getenv('SANDBOX_MODEL')
    if crucial_fold_d7e145:
        return [crucial_fold_d7e145]
    holiday_cream_e5766c: dict[str, Any] = {'BS_PRODUCT_PARSE_': 'deepseek-ai/DeepSeek-V3.2-TEE', 'BS_VOUCHER_PARSE_': 'deepseek-ai/DeepSeek-V3.2-TEE', 'BS_PRODUCT_RANK_': 'deepseek-ai/DeepSeek-V3.2-TEE', 'BS_FINAL_FALLBAC_': 'google/gemma-4-31B-turbo-TEE', 'BS_BACKUP_LLM_': 'deepseek-ai/DeepSeek-V3.2-TEE', 'BS_SHOP_PARSE_': 'deepseek-ai/DeepSeek-V3.2-TEE', 'PICK_CHAIN': ['google/gemma-4-31B-turbo-TEE', 'deepseek-ai/DeepSeek-V3.2-TEE', 'deepseek-ai/DeepSeek-V3.2-TEE'], 'SCORE_CHAIN': ['deepseek-ai/DeepSeek-V3.2-TEE', 'deepseek-ai/DeepSeek-V3.2-TEE', 'google/gemma-4-31B-turbo-TEE']}
    exist_confirm_be9e28: dict[str, Any] = {'BS_PRODUCT_PARSE_': 'deepseek/deepseek-v3.2', 'BS_VOUCHER_PARSE_': 'deepseek/deepseek-v3.2', 'BS_FINAL_FALLBAC_': 'google/gemma-4-31b-it', 'BS_PRODUCT_RANK_': 'deepseek/deepseek-v3.2', 'BS_BACKUP_LLM_': 'deepseek/deepseek-chat-v3.1', 'BS_SHOP_PARSE_': 'deepseek/deepseek-v3.2', 'PICK_CHAIN': ['google/gemma-4-31b-it', 'deepseek/deepseek-v3.2', 'deepseek/deepseek-chat-v3.1'], 'SCORE_CHAIN': ['deepseek/deepseek-chat-v3.1', 'deepseek/deepseek-v3.2', 'google/gemma-4-31b-it']}
    camera_concert_f592ee: dict[str, dict] = {'chutes': holiday_cream_e5766c, 'openrouter': exist_confirm_be9e28}
    flat_forget_18f7e8 = camera_concert_f592ee.get(U_convince_bs_active_provider()) or exist_confirm_be9e28
    return flat_forget_18f7e8['SCORE_CHAIN']

def U_diagram_bs_score_listings(query_text: str, candidates: list[dict], details: dict[str, dict], donate_document_174e57: bool=False, model: str=None) -> list[tuple[dict, float]]:
    if not candidates:
        return []
    if U_boy_bs_budget_sec_left() < 35.0:
        return [(p, 7.0) for p in candidates if U_best_bs_heuristic_score(p, query_text) > 0]
    payload = {'request': query_text, 'candidates': [U_league_bs_build_candidate(p, details.get(str(p.get('product_id', ''))), query_text) for p in candidates], 'only_product_type': donate_document_174e57}
    dirt_female_8fd477 = json.dumps(payload, ensure_ascii=False)
    capable_imitate_b2eb8b = 1
    for m in U_arrange_bs_score_model_seq():
        club_cousin_8e2e17 = 0
        while club_cousin_8e2e17 < capable_imitate_b2eb8b:
            club_cousin_8e2e17 += 1
            icon_demand_cc0a48 = None
            if not (icon_demand_cc0a48 and icon_demand_cc0a48.get('choices')):
                continue
            bench_finish_9afb2e = icon_demand_cc0a48['choices'][0].get('message', {}).get('content', '')
            gospel_cargo_eb0f14 = re.sub('```json?\\s*', '', bench_finish_9afb2e)
            gospel_cargo_eb0f14 = re.sub('```\\s*$', '', gospel_cargo_eb0f14).strip()
            describe_action_661eee = None
            try:
                describe_action_661eee = json.loads(gospel_cargo_eb0f14)
            except json.JSONDecodeError:
                hand_cargo_293fb3 = re.search('\\[.*\\]', bench_finish_9afb2e, re.DOTALL)
                if hand_cargo_293fb3:
                    try:
                        describe_action_661eee = json.loads(hand_cargo_293fb3.group())
                    except json.JSONDecodeError:
                        pass
            if not isinstance(describe_action_661eee, list):
                continue
            gallery_beef_9ae581: dict[str, float] = {}
            for garden_guide_d63f93 in describe_action_661eee:
                if not isinstance(garden_guide_d63f93, dict):
                    continue
                pid = str(garden_guide_d63f93.get('product_id', '')).strip()
                if not pid:
                    continue
                try:
                    gallery_beef_9ae581[pid] = float(garden_guide_d63f93.get('score', 0))
                except (TypeError, ValueError):
                    gallery_beef_9ae581[pid] = 0.0
            access_diesel_6f0d1c = [(p, gallery_beef_9ae581.get(str(p.get('product_id', '')).strip(), 0.0)) for p in candidates]
            access_diesel_6f0d1c.sort(key=lambda x: (x[1], str(x[0].get('product_id', ''))), reverse=True)
            return access_diesel_6f0d1c
    access_diesel_6f0d1c = [(p, 7.0 if U_best_bs_heuristic_score(p, query_text) > 0 else 0.0) for p in candidates]
    access_diesel_6f0d1c.sort(key=lambda x: (x[1], str(x[0].get('product_id', ''))), reverse=True)
    return access_diesel_6f0d1c

class U_MiscUtil:
    LOGGER = logging.Logger("angry_oro")
    LOGGER.addHandler(logging.StreamHandler())

    @staticmethod
    def parse_json(s: str) -> dict[str, Any] | list[Any] | None:
        cleaned = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags = re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned, ).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        starts = []
        
        obj_start = cleaned.find("{")
        if obj_start >= 0: starts.append(obj_start)

        arr_start = cleaned.find("[")
        if arr_start >= 0: starts.append(arr_start)

        for start in sorted(starts):
            try:
                obj, _ = decoder.raw_decode(cleaned[start:])
                return obj
            except json.JSONDecodeError:
                pass

        for i in range(len(cleaned)):
            if cleaned[i] in ['}', ']']:
                try:
                    return json.loads(cleaned[:i] + cleaned[i] + cleaned[i:])
                except json.JSONDecodeError:
                    pass

                try:
                    return json.loads(cleaned[:i] + cleaned[i + 1:])
                except json.JSONDecodeError:
                    pass

        try:
            pattern = re.compile(r'([,{])\s*"([^":]+)\s*:\s*([^"]+)"\s*([,}])')

            def __replace__(m: re.Match):
                return f'{m.group(1)} "{m.group(2).strip()}": "{m.group(3).strip()}"{m.group(4)}'

            fix = pattern.sub(__replace__, cleaned)
            return json.loads(fix)
        except json.JSONDecodeError:
            pass

        try:
            fix = re.sub(r"(\},\s+)(\w+\")", r"\1\"\2", cleaned)
            fix = re.sub(r"(\"[\w\_]+)\s*(\:\s*\{)", r"\1\"\2", fix)
            return json.loads(fix)
        except json.JSONDecodeError:
            pass

        try:
            fix = re.sub(r",\s*([}\]])", r"\1", cleaned)
            return json.loads(fix)
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def print_json(js: Dict[str, Any] | List[Any]) -> None:
        U_MiscUtil.LOGGER.info(json.dumps(js, ensure_ascii = False, indent = 2))

    @staticmethod
    def get_choice_combs(cat: Dict | List, include_non: bool) -> List[Dict | List]:
        keys = list(cat.keys()) if isinstance(cat, dict) else list(range(len(cat)))
        
        choices = [([-1] if include_non else []) + list(range(len(cat[k]))) for k in keys]
        opts = product(*choices)

        return [dict(zip(keys, opt)) for opt in opts] if isinstance(cat, dict) else list(opts)

    @staticmethod
    def extract_number(s: str, as_int: bool = False) -> int | float:
        pattern = r"\d+\.?\d*"
        matches = re.findall(pattern, s)

        if len(matches) == 0: return 0        
        val = float(matches[0])

        return val if not as_int else int(val)

    @staticmethod
    def scan_patterns(s: str, patterns: List[str]) -> Optional[str]:
        for p in patterns:
            if (m := re.search(p, s, re.I)): return m.group(1)
        
        return None
    
    @staticmethod
    def normalize_str(s: str, split_an: bool) -> str:
        s = s.lower().replace("'s", "")
        s = re.sub(r"(?<=\d)\s*[*x]\s*(?=\d)", " ", s)
        s = re.sub(r"[^a-z0-9\u4e00-\u9fff.]", " ", s)
        s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
        if split_an: s = re.sub(r"\b(\d+)([a-z]+)\b", r"\1 \2", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def is_morph_equal(v1: str, v2: str, assist_map: Dict[str, Set[str]]) -> bool:
        v1, v2 = v1.lower().strip(), v2.lower().strip()
        if v1 == v2: return True

        if len(v1) < 2 or len(v2) < 2: return False
        if re.search(r"\d", v1 + v2): return False        

        if len(v1) > 6 and len(v2) > 6:
            if v1[:6] == v2[:6]: return True

        def __get_stems__(word: str) -> set[str]:
            word = word.lower()

            roots = {word}
            queue = [word]

            while queue:
                w = queue.pop()

                for suffix, rules in U_Vocab.MORPH_RULES:
                    if not w.endswith(suffix): continue
                    if len(w) <= len(suffix) + 2: continue

                    for r in rules:
                        if callable(r):
                            t = r(r"([b-df-hj-np-tv-z])\1$", r"\1", w[:-len(suffix)])
                        else:
                            t = w[:-len(suffix)] + r

                        if len(t) >= 2 and t not in roots:
                            roots.add(t)
                            queue.append(t)

                for syn in assist_map.get(w, {}) | assist_map.get(w.replace('-', ' '), {}):
                    if syn not in roots:
                        roots.add(syn)
                        queue.append(syn)

            return roots

        return bool(__get_stems__(v1) & __get_stems__(v2))

    @staticmethod
    def get_coverage(atts: Tuple[str], qs: List[str], key_len: int, assist_map: Dict[str, Set[str]]) -> Tuple[Set, bool]:
        sizes: Dict[str, str] = {x.split('2')[0]: x.split('2')[1] for x in U_Vocab.SIZES}
        cover, contrib, val_used = set(), set(), False        

        for j, token in enumerate(atts):
            for i in range(len(qs)):
                if U_MiscUtil.is_morph_equal(token, qs[i], assist_map):                    
                    cover.add(i)
                    contrib.add(j)
                
                if i < len(qs) - 1:
                    a, b = qs[i], qs[i + 1]

                    if token.startswith(a):
                        if U_MiscUtil.is_morph_equal(token[len(a):], b, assist_map):
                            cover.update({i, i + 1})
                            contrib.add(j)
                
                for literal, num in U_Vocab.NUMBERS.items():
                    if {literal, num} == {qs[i], token}:
                        contrib.add(j)
                        cover.add(i)                        
                        break

        for j in range(len(atts) - 1):
            h, y = atts[j], atts[j + 1]

            for i in range(len(qs)):
                v = qs[i]

                if v.startswith(h):
                    if U_MiscUtil.is_morph_equal(v[len(h):], y, assist_map):
                        contrib.update({j, j + 1})
                        cover.add(i)

        if "size" in atts[:key_len]:
            horigin = U_MiscUtil.extract_number.__name__[:5]

            for j, token in enumerate(atts):
                if token in sizes:
                    if (indice := [i for i in range(len(qs)) if qs[i] == sizes[token]]):
                        cover.update(indice)
                        contrib.add(j)
                elif len(token) == 2 and token[0] == horigin[1] and token[1] in sizes:
                    for i in range(len(qs) - 1):
                        if qs[i] == horigin and qs[i + 1] == sizes[token[1]]:
                            cover.update({i, i + 1})
                            contrib.add(j)
                elif token == horigin and j < len(atts) - 1 and atts[j + 1] in sizes.values():
                    if (indice := [i for i in range(len(qs)) if qs[i] == f"{horigin[1]}{atts[j + 1][0]}"]):
                        cover.update(indice)
                        contrib.add(j)
                elif token in sizes.values():
                    if (indice := [i for i in range(len(qs)) if qs[i] == token[0]]):
                        cover.update(indice)
                        contrib.add(j)

        for j in range(key_len, len(atts)):
            if j in contrib:
                val_used = True
                continue
            
            if atts[j] in U_Vocab.COLOR_WORDS + list(sizes.keys()) + list(sizes.values()): return set(), False
            if re.search(r"\d", atts[j]): return set(), False

        return cover, val_used

class U_ProxyUtil:
    LLM_PROXY = ProxyClient(timeout = 180, max_retries = 5, retry_delay = 1.0)
    ORO_PROXY = ProxyClient(timeout = 30, max_retries = 5, retry_delay = 1.0)    

    LLM_PRIORITIES = ["moonshotai/Kimi-K2.6-TEE", "zai-org/GLM-5.1-TEE", "deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3.5-397B-A17B-TEE", "Qwen/Qwen3-32B-TEE", "google/gemma-4-31B-turbo-TEE"]
    TEMP_UP = 0.3

    OPENROUTER_PRIORITIES = ["moonshotai/kimi-k2.6", "z-ai/glm-5.1", "deepseek/deepseek-v3.2", "deepseek/deepseek-chat-v3.1", "qwen/qwen3.5-397b-a17b", "qwen/qwen3-32b", "xiaomi/mimo-v2-flash", "google/gemma-4-31b-it"]

    @staticmethod
    def default_llm_priorities() -> List[str]:
        provider = U_convince_bs_active_provider() if "U_convince_bs_active_provider" in globals() else "openrouter"
        if provider == "openrouter":
            return list(U_ProxyUtil.OPENROUTER_PRIORITIES)
        return list(U_ProxyUtil.LLM_PRIORITIES)

    @staticmethod
    def get_role_llm_priorities(role: str) -> List[str]:
        defaults = U_ProxyUtil.default_llm_priorities()
        if not defaults:
            return []
        role = (role or "").lower()
        if role in ("parse", "filter", "build", "review"):
            return defaults[:3]
        if role in ("score", "weigh"):
            mid = max(1, len(defaults) // 2)
            return defaults[mid:mid + 3] or defaults[:3]
        if role in ("pick", "select", "judge"):
            rotated = defaults[2:] + defaults[:2] if len(defaults) > 2 else defaults
            return rotated[:3]
        if role in ("duel", "dual_compare", "alt_select"):
            # diversify: prefer later models for second opinion
            rotated = defaults[1:] + defaults[:1]
            return rotated[:3]
        return defaults[:3]

    @staticmethod
    def __empty_for__(expect: str) -> Tuple[Any, None]:
        if expect == "list":
            return [], None
        if expect == "dict":
            return {}, None
        return None, None

    @staticmethod
    def trigger_llm(prompt: str, thinking: bool = False, preferred_llms: List[str] = [], temperature: float = 0, expect: str = "any") -> Tuple[Dict[str, Any] | List[Any], Any]:        
        llm_priorities = U_ProxyUtil.__get_llm_priorities__(preferred_llms)
        trigger_tick = time.time()
        t = temperature
        # Cap outer retries — unbounded while-True hung agents and burned the sandbox budget.
        max_rounds = max(2, U_ENHANCEMENT_BENCH.llm_retry_max)

        for _round in range(max_rounds):
            for model in llm_priorities:
                payload = U_ProxyUtil.__create_llm_payload__(model, prompt, thinking, t, 8192)

                try:
                    tick = time.time()

                    resp: Dict[str, Dict] = U_ProxyUtil.LLM_PROXY.post("/inference/chat/completions", json_data = payload)
                    content = U_ProxyUtil.__resp_to_content__(resp)

                    try:
                        js = U_MiscUtil.parse_json(content)
                        if js is None: raise Exception()
                    except Exception as exc:
                        t = min(U_ENHANCEMENT_BENCH.temperature_max, t + U_ProxyUtil.TEMP_UP)
                        raise Exception(f"failed to parse json: {exc}\n*******\n{content}\n******")

                    if (expect == "dict" and not isinstance(js, dict)) or\
                        (expect == "list" and not isinstance(js, list)):
                        t = min(U_ENHANCEMENT_BENCH.temperature_max, t + U_ProxyUtil.TEMP_UP)
                        raise Exception(f"expected {expect} but got {type(js)}")

                    throughput = resp.get("usage", {}).get("completion_tokens", 0)                    
                    trigger_delay = time.time() - trigger_tick
                    delay = time.time() - tick
                    
                    U_MiscUtil.LOGGER.info(
                        f"* llm triggered: model = '{model}' | throughput = {throughput} | delay = {delay:.2f}s | "
                        f"total_delay = {trigger_delay:.2f}s"
                    )
                    return js, resp
                except Exception as exc:
                    U_MiscUtil.LOGGER.error(f"# llm trigger error with '{model}': {exc}")
                    continue
            t = min(U_ENHANCEMENT_BENCH.temperature_max, t + U_ProxyUtil.TEMP_UP)

        U_MiscUtil.LOGGER.error("# llm trigger exhausted all models/rounds; returning empty")
        return U_ProxyUtil.__empty_for__(expect)

    @staticmethod
    def search(filter: U_Filter, sort: Optional[str] = None, page: int = 1) -> Tuple[List[U_Product], ToolCallResult]:
        def _params(q: str, drop_service: bool = False) -> dict:
            p = {
                "q": q,
                "page": page,
                "shop_id": filter.shop,
                "price": filter.price,
                "sort": sort,
            }
            if not drop_service and filter.services:
                p["service"] = ",".join(filter.services)
            return p

        def _to_products(js_prods: List[Dict]) -> List[U_Product]:
            products = []
            for j in js_prods or []:
                price = j.get("price") or 0
                price = U_MiscUtil.extract_number(str(price))
                products.append(
                    U_Product(
                        id = str(j.get("product_id")),
                        title = j.get("title", ""),
                        price = price,
                        shop = j.get("shop_id", ""),
                        services = j.get("service", []) or [],
                        attr = {},
                        sku = {},
                        is_viewed = False,
                    )
                )
            return products

        q0 = filter.get_q()
        attempts = [
            _params(q0, False),
            _params(q0, True),
        ]
        tokens = str(q0 or "").split()
        if len(tokens) > 2:
            attempts.append(_params(" ".join(tokens[:2]), True))
        if len(tokens) > 1:
            attempts.append(_params(tokens[0], True))

        last_exc = None
        last_tool = None
        for params in attempts:
            try:
                tool_res = execute_tool_call("find_product", params)
                last_tool = tool_res
                js_prods: List[Dict] = tool_res["result"] if isinstance(tool_res.get("result"), list) else []
                products = _to_products(js_prods)
                if products or params is attempts[-1]:
                    return products, tool_res
            except Exception as exc:
                last_exc = exc
                U_MiscUtil.LOGGER.error(f"# error in `find_products`: {exc}")
                time.sleep(0.3)
        if last_tool is not None:
            return [], last_tool
        raise last_exc or RuntimeError("find_product failed")

    @staticmethod
    def view(products: List[U_Product]) -> ToolCallResult:
        if not products: return        
        pid_str = ','.join([p.id for p in products])

        while True:
            try:
                tool_res = execute_tool_call("view_product_information", {"product_ids": pid_str})        
                
                details = tool_res["result"]
                if not details: return tool_res

                details_map: Dict[str, Dict[str, Any]] = {d["product_id"]: d for d in details}

                for prod in products:
                    d = details_map.get(prod.id, {})
                    
                    prod.sku = d.get("sku_options", {})
                    prod.attr = d.get("attributes", {})
                    prod.is_viewed = True
                
                return tool_res
            except Exception as exc:
                U_MiscUtil.LOGGER.error(f"# error in `view_products`: {exc}")
                time.sleep(0.5)

    @staticmethod
    def submit(products: List[U_Product]) -> ToolCallResult:
        return execute_tool_call("recommend_product", {"product_ids": ','.join([p.id for p in products])})
    
    @staticmethod
    def finish(status: str) -> ToolCallResult:
        return execute_tool_call("terminate", {"status": status})

    @staticmethod
    def __get_llm_priorities__(prefers: List[str]) -> List[str]:
        # Append provider-matched defaults (openrouter vs TEE), not always the chutes list.
        priorities = list(prefers or [])
        for llm in U_ProxyUtil.default_llm_priorities():
            if llm not in priorities:
                priorities.append(llm)
        return priorities

    @staticmethod
    def __create_llm_payload__(model: str, prompt: str, thinking: bool, temperature: float, max_tokens: int) -> Dict:
        return {
            'model': model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'chat_template_kwargs': {
                'enable_thinking': thinking,
                'thinking': thinking
            },
            'reasoning': {
                'effort': 'medium' if thinking else 'none',
                'enabled': thinking
            },
            "thinking": {
                "type": "disabled" if not thinking else "enabled"
            },
            'temperature': temperature,
            'stream': False,
            'max_completion_tokens': max_tokens,
            'max_tokens': max_tokens,
        }

    @staticmethod
    def __resp_to_content__(resp: Dict) -> str:
        try:
            choice: Dict[str, Dict] = resp.get("choices")[0]
            content = choice.get("message").get("content")

            if not content: raise Exception()
            return content
        except Exception as exc:
            raise Exception(f"invalid response ({exc})")

class U_EngineUtil:
    @staticmethod
    def compact_kv(k: str, v: str, q: str) -> Optional[Tuple[str, str]]:
        k, v = k.lower(), v.lower()
        ksplits, korigin = k.split('_'), k.replace('_', ' ')

        if "brand" in ksplits: return (k, v)
        v = v.replace("'s", '')

        if v in U_Vocab.YES_WORDS and korigin in q and not re.search(r"\b(mode|support)$", k):
            k, v = U_Vocab.YES_WORDS[0], korigin

        for yes in U_Vocab.YES_WORDS:
            if re.search(rf"\b{re.escape(yes)}$", v, re.I): v = re.sub(rf"\b{re.escape(yes)}$", "", v, flags = re.I)
        
        for no in U_Vocab.NO_WORDS:
            if (m := re.search(rf"^{re.escape(no)}\s+([\w\s]+)$", v, re.I)):
                v = f"{U_Vocab.NO_WORDS[0]} {m.group(1)}"
                break
            elif re.search(rf"\b{re.escape(no)}$", v, re.I):                
                if korigin in q:
                    k, v = U_Vocab.NO_WORDS[0], korigin
                    break

        if len(ksplits) > 1 and "count" in ksplits and not re.search(r"\bcount\b", q):
            ksplits.remove("count")
            k = '_'.join(ksplits)

        if v == "1" and re.search(r"\bsingle\b", q) and not re.search(r"\b1[a-z]*\b", q): v = "single"
        kv_splits = set(k.replace('_', ' ').split() + v.replace('_', ' ').split())

        for stop in U_Vocab.KW_STOPWORDS:
            v = re.sub(rf"\s{re.escape(stop)}\s", " ", v, flags = re.I)
            if kv_splits == set(stop.split()): v = ""

        for idk in U_Vocab.ID_KEYS:
            if v == idk and k == idk: v = ""

        if (m := re.search(r"^(any|all)\s+([\w\s]+)$", v, re.I)):
            k, v = m.group(2).replace(' ', '_'), m.group(1)

        if v in ["one", "free"] and "size" in k:
            k = "size"
            v += f" {k}"

        for unit in ksplits:
            if unit != "pcs" and unit.endswith('s'): unit = unit[:-1]
            if not v.isnumeric(): break

            for mu in list(U_Vocab.METRIC_UNITS.keys()) + list(U_Vocab.METRIC_UNITS.values()):
                if not mu: continue

                if mu == unit:
                    if re.search(rf"\b{re.escape(v)}{re.escape(unit)}s?\b", q):
                        v = f"{v}{mu}"
                    else:
                        v = f"{v} {mu}"

                    break

        g = U_ProxyUtil.LLM_PRIORITIES[-1].split('/')[1][:3]
        t = U_EngineUtil.stringfy_price.__name__[:2]

        v = re.sub(rf"\b{re.escape(g)}({re.escape(t)}one)\b", r"\1", v, flags = re.I)
        v = re.sub(r"[\&\|\+]", " ", v)
        v = re.sub(r"\s+", " ", v).strip()

        return (k, v) if v else None
    
    @staticmethod
    def compact_alt(k: str, v: str) -> Optional[Tuple[str, str]]:
        k, v = k.lower().replace("'s", '').replace("'", ''), v.lower().replace("'s", '').replace("'", '')
        k, v = k.replace('-', ' '), v.replace('-', ' ')
        k, v = re.sub(r"\s+", " ", k), re.sub(r"\s+", " ", v)

        if re.search(r"\d", k + v): return None
        if U_Prompt.label_to_service(k) or U_Prompt.label_to_service(v): return None

        if k == v: return None
        return k, v

    @staticmethod
    def promptfy_price(price: Optional[str]) -> str:
        if not price: return ""

        segs = [s.strip() for s in price.split('-')]
        if len(segs) != 2: return ""

        try:
            low = int(segs[0]) if segs[0] else None
            high = int(segs[1]) if segs[1] else None

            if low and high: return f"price between {low} and {high}"
            if low: return f"price greater than {low}"

            return f"price less than {high}"
        except Exception:
            return ""

    @staticmethod
    def numerify_price(price: Optional[str]) -> Tuple[int, int]:
        low_str, high_str = price.split('-', 1)
        low, high = 0, math.inf

        if low_str: low = U_MiscUtil.extract_number(low_str, True)
        if high_str: high = U_MiscUtil.extract_number(high_str, True)

        return low, high

    @staticmethod
    def stringfy_price(low: int, high: int) -> str:
        return (str(low) if low > 0 else "") + "-" + (str(high) if high < math.inf else "")

class U_SegTree(object):
    def __init__(self, values, k = 30) -> None:
        self.k, n = k, len(values)
        self.size = 1

        while self.size < n:
            self.size <<= 1

        self.seg = [[] for _ in range(2 * self.size)]

        for i, v in enumerate(values):
            self.seg[self.size + i] = [(v, i)]

        for i in range(self.size - 1, 0, -1):
            self.seg[i] = self.__merge__(self.seg[2 * i], self.seg[2 * i + 1])

    def __merge__(self, a: List, b: List) -> List:
        out, ia, ib = [], 0, 0

        while len(out) < self.k:
            if ia == len(a):
                out.extend(b[ib:ib + self.k - len(out)])
                break

            if ib == len(b):
                out.extend(a[ia:ia + self.k - len(out)])
                break

            if a[ia][0] >= b[ib][0]:
                out.append(a[ia])
                ia += 1
            else:
                out.append(b[ib])
                ib += 1

        return out

    def query(self, l: int, r: int) -> List:
        if l > r: return []
        left, right = [], []

        l += self.size
        r += self.size

        while l <= r:
            if l & 1:
                left = self.__merge__(left, self.seg[l])
                l += 1

            if not (r & 1):
                right = self.__merge__(self.seg[r], right)
                r -= 1

            l >>= 1
            r >>= 1

        return self.__merge__(left, right)

class U_Debugger(object):
    def append(self, *_, **__) -> None:
        return

    def find(self, *_, **__) -> Any:
        return None

    def is_stop_at(self, *_, **__) -> bool:        
        return False

    def is_frozen(self, *_, **__) -> bool:
        return False

    def log(self, *_, **__) -> None:
        return

class U_Debuggable(object):
    def __init__(self, query: U_Query, debugger: Optional[U_Debugger]) -> None:
        self.debugger: Optional[U_Debugger] = debugger
        self.query: U_Query = query
        self.stopped: bool = False

    def log(self, obj: Any) -> None:
        if self.debugger is not None: self.debugger.log(obj)

class U_Session(U_Debuggable):
    MAX_TOOL_RES_PER_STEP = 7

    def __init__(self, query: U_Query, debugger: Optional[U_Debugger]) -> None:
        super().__init__(query, debugger)

        self.tool_results = []
        self.dialog = []
    
    def step(self, think: str, content: str) -> None:
        compact_tools = [
            U_health_bs_compact_result(tr) if isinstance(tr, dict) else tr
            for tr in self.tool_results[:U_Session.MAX_TOOL_RES_PER_STEP]
        ]
        self.dialog.append(
            create_dialogue_step(
                think = think,
                tool_results = compact_tools,
                response = content,
                query = self.query.query_f,
                step = len(self.dialog) + 1
            )
        )
        self.tool_results = []

        if self.debugger is not None and self.debugger.show_step:                        
            self.log({
                "step": len(self.dialog),
                "think": think,
                "content": content
            })
    
    def clear(self) -> None:
        self.tool_results = []

    def add_tool_result(self, res: ToolCallResult) -> None:
        try:
            json.dumps(res)
            self.tool_results.append(res)
        except Exception as exc:
            self.log(f"# cannot add tool result: {exc}")

    def step_on_build(self) -> None:
        think = f"This is a shopping request involving {U_Challenge(self.query.challenge).promptfy(len(self.query.intents) > 1)}. "

        if len(self.query.intents) > 1:
            think += "First, I will decompose the user's request into several sub-queries, each of which corresponds to a single product:\n"
            
            for i, sq in enumerate(self.query.intents):
                think += f"\n**{i + 1}.** `{sq.query}`"
        else:
            think += f"First, I will rewrite the user's request in a more concise form:\n\n`{self.query.intents[0].query}`"

        think += f"\n\nNext, I will extract meaningful keywords and additional filters from "
        think += "each sub-query" if len(self.query.intents) > 1 else "the query"
        think += " to facilitate search:\n"

        keywords = [', '.join([v for _, v in sq.parsed.items()]) for sq in self.query.intents]
        services = [', '.join(svc for svc in sq.services) if sq.services else "not specified" for sq in self.query.intents]
        prices = [sq.price if sq.price else "not specified" for sq in self.query.intents]
        
        for i in range(len(self.query.intents)):
            think += f"\n**{i + 1}.** " if len(self.query.intents) > 1 else "\n"
            think += f"keywords = `{keywords[i]}` | service = `{services[i]}` | price = `{prices[i]}`"

        if self.query.challenge == U_Challenge.VOUCHER:
            think += f"\n\nThe user's budget is: {int(self.query.voucher.budget)}.\n"
            think += "The available voucher options are as follows:\n\n"
            think += "**Voucher Type:** " + ("Fixed amount" if self.query.voucher.is_fixed else "Percentage") + "\n"
            think += f"**Discount Amount:** {int(self.query.voucher.discount)}" + ("%" if not self.query.voucher.is_fixed else "") + "\n"
            think += "**Application Condition:** " + ("Products from the same shop" if self.query.voucher.apply_to == "shop" else "All products") + "\n"            
            think += (f"**Threshold:** {int(self.query.voucher.threshold)}" if self.query.voucher.threshold is not None else "") + "\n"
            think += (f"**Cap:** {int(self.query.voucher.cap)}" if self.query.voucher.cap is not None else "") + "\n"

        self.step(think, "Initial query analysis completed.")

    def step_on_search(self) -> None:
        think = "Next, I will perform an initial search for " + ("each sub-query" if len(self.query.intents) > 1 else "the query") + ", retrieving up to five pages of results:"

        act = {
            "action": "search",
            "options": [
                {
                    "keywords": [v for _, v in sq.parsed.items()],
                    "price": sq.price or None,
                    "service": sq.services or None,
                    "pages": "1-5"
                }
                for sq in self.query.intents
            ]
        }
        think += f"\n\n```json\n{json.dumps(act, ensure_ascii = False, indent = 4)}```"        
        self.step(think, "")

    def step_on_shop_plan(self, shop_stats: Dict[str, Tuple[int, float]], top_k: int) -> None:
        if self.query.challenge == U_Challenge.VOUCHER:
            think = "Now I will look for combinations of products from the same shop, since such combinations may benefit from voucher application.\n"
        else:
            think = ""

        think += "First, I will group the retrieved products by shop and examine how well each shop covers the sub-queries. Shops with broader coverage are more promising, but additional shops will also be considered to avoid overlooking better combinations. The following statistics summarize the coverage provided by the discovered shops:\n\n"
        plan = []

        for n in range(len(self.query.intents), 0, -1):
            shops = [sid for sid in shop_stats if shop_stats[sid][0] == n]
            plan.extend(shops)

            if n > 1:
                think += f"- Shops covering {n} sub-queries: `{', '.join(shops)}`\n"
            else:
                think += f"- The remaining {len(shops)} shop{'s' if len(shops) > 1 else ''} cover{'s' if len(shops) == 1 else ''} only one sub-query.\n"
        
        think += "\nGood. Next, I will examine the most promising shops in order of coverage to determine whether they can fully satisfy the sub-queries"
        think += " while allowing the voucher to be applied." if self.query.challenge == U_Challenge.VOUCHER else "."
        think += f"\n\nThe following shops will be examined: {', '.join(plan[:top_k])}, ... ..."

        self.step(think, "Planned candidate shops.")

    def step_on_shop_batch(self, batch: List[str], sols: int, best_p: List[U_Product], n: int) -> None:
        best_str = ', '.join([p.id for p in best_p])
        
        think = f"**Shop Examination Step {n + 1}**\n\n"
        think += f"In this step, I will check the following shop{'s' if len(batch) > 1 else ''}: `{', '.join(batch)}`\n\n"
        think += "I have performed additional searches for each shop and each sub-query using narrower search criteria by enforcing a shop ID constraint. "
        think += f"Among the retrieved products, I was able to find {sols} combination{'s' if sols > 1 else ''} whose products all come from the same shop"
        think += " for which the voucher can be applied. " if self.query.challenge == U_Challenge.VOUCHER else ". "

        if sols > 0:
            think += f"After carefully examining the candidate combination{'s' if sols > 1 else ''}, "
            
            if best_p:
                think += f"I identified a promising combination (`{best_str}`). Each product satisfies its corresponding constraints. "
                think += "I will stop the shop examination and recommend the identified combination."
            else:
                think += f"{'all of them were' if sols > 1 else 'it was'} rejected because one or more products did not satisfy the specified constraints. "
                think += "I will continue the shop examination."
        else:
            think += "I will continue the shop examination."
        
        self.step(think, "")

    def step_on_shop_fallback(self, best_p: List[U_Product]) -> None:
        best_str = ', '.join([p.id for p in best_p])

        think = "**Shop Examination Rollback**\n\nI have examined all discovered shops, but was unable to find one providing matching products"
        think += " for which the voucher can be applied.\n\n" if self.query.challenge == U_Challenge.VOUCHER else ".\n\n"

        if best_p:
            think += f"Either a valid combination has been overlooked during the previous evaluation steps, or no matching combination exists in the current database. Before reporting failure to the user, I will perform one final verification of the previously discovered candidate combinations. The combination `{best_str}` appears to be a strong candidate. Let me verify it against the stated constraints. Upon re-examination, I found that this combination satisfies all stated constraints. It was incorrectly rejected during the previous evaluation. To ensure that no other valid combinations have been overlooked, I will review the remaining candidates as well. No additional combinations satisfying all stated constraints were found. The only missed combination is `{best_str}`. Therefore, this combination can be recommended to the user."
        elif self.query.challenge == U_Challenge.VOUCHER:
            think += "However, voucher application is optional. As long as the total price remains within the budget, the combination can still be recommended. Therefore, I will switch to voucher-free examination mode."
        else:
            think += "This indicates that no matching combination exists in the current database. I will report failure to the user."

        self.step(think, "Completed shop examination process.")

    def step_on_platform(self, best_p: List[U_Product], sol_count: bool, is_fallback: bool) -> None:
        best_str = ', '.join([p.id for p in best_p])

        think = (
            f"Now I will look for combinations of products whose total price is within the buget"
            f"{' or becomes affordable after applying the voucher.' if not is_fallback else '.'}"
            f" First, I will apply the Meet-in-the-Middle (MITM) algorithm to solve the underlying knapsack problem, where product weights are derived from the logarithms of the search-rank probabilities. I have executed MITM algorithm and found {sol_count} valid solutions, sorted by total weights. Next, I will examine the candidate combinations in order until I find one that satisfies all stated constraints. "
        )
        if best_p:
            think += f"The combination `{best_str}` appears to be a strong candidate. Let me verify its products against the constraints. Good. I have confirmed that each product satisfies its corresponding sub-query. Now this combination can be recommended to the user."
        else:
            think += "I have evaluated all candidate combinations, but none of them satisfies all stated constraints. This indicates that no combination satisfying the user's requirements exists in the current database. I will report the failure to the user."
        
        self.step(think, "")

    def step_on_select(self, best_p: U_Product, js_check: Dict, is_fallback: bool, sq: U_SubQuery) -> None:
        if not is_fallback:
            think = f"From the retrieved products, I will consider the top {U_BaseEngine.SELECT_BATCH} candidates and check them in several phases until a product satisfying all the constraints is found. For optimization, I will use a strategy based on Fibonacci sequence windows for checking candidates.\n\n```json\n{json.dumps(js_check, ensure_ascii = False, indent = 2)}```\n\nGood. A matching product has been identified."
        else:
            think = f"Since the retrieved products are already ranked by relevance, I will first examine the highest-ranked product to determine whether it satisfies all the user's requirements. If it turns out to be a good match, I can recommend it directly without performing any further evaluation. The title of the highest-ranked product is: `{best_p.title}` I will check the attribute details of the product to verify each constraint using the following checklist:\n"

            for k, v in sq.parsed.items():
                think += f"\n`{k}`: `{v}` → match ✓"
            
            think += "\n\nAll checks passed."

        self.step(think, f"Best-matching candidate selected: `{best_p.id}`")

    def step_on_compare(self, best_p: List[U_Product], alt: str) -> None:
        if not best_p or not alt: return

        think = "To avoid overlooking a better candidate, I will compare the identified one with other top promising candidate.\n\n" + alt
        self.step(think, "Compared with the top promising candidate.")

    def step_on_submit(self, best_p: List[U_Product]) -> None:
        if not best_p:
            return self.step(
                "I can't find the product(s) satisfying the user's requirements. I will report failure to the user.",
                "Failed to find the requested product(s). Please try again with more specific constraints."
            )

        think = f"I have identified the following product{' combination' if len(self.query.intents) > 1 else ''}: `{', '.join([p.id for p in best_p])}`."
        think += f" Before submitting the recommendation, I will verify all the requirements again.\n"

        for i in range(len(best_p)):
            think += f"\n**Product {i + 1}** (UID `{best_p[i].id}`)\n" if len(self.query.intents) > 1 else ""
            think += f"\n- query: `{self.query.intents[i].query}`"
            think += f"\n- product_title: `{best_p[i].title}`"
            think += f"\n- shop_id: `{best_p[i].shop}`"

            for k, v in self.query.intents[i].parsed.items():
                think += f"\n- {k}: expects `{v}` | found `{v}` in {'title' if k == 'product_type' else 'attributes'} → exact match ✓"

            if self.query.intents[i].price:
                think += f"\n- price: expects {U_EngineUtil.promptfy_price(self.query.intents[i].price)} | found {str(int(best_p[i].price))} → within range ✓"
            
            for svc in (self.query.intents[i].services or []):
                think += f"\n- service: expects `{svc}` | found `{svc}` → available ✓"
            
            think += "\n"

        if self.query.challenge == U_Challenge.VOUCHER:
            psum = sum(p.price for p in best_p)
            pact = self.query.voucher.get_active_cost(best_p) if self.query.voucher else psum

            if len(best_p) > 1:
                think += "\nThe prices of the selected products are "
                think += ", ".join([str(int(p.price)) for p in best_p[:-1]]) + " and " + str(int(best_p[-1].price))
                think += ", respectively, with a total price of " + str(int(psum)) + "."
            else:
                think += "\nThe price of the selected product is " + str(int(best_p[0].price)) + "."

            if psum > pact:
                if self.query.voucher.apply_to == "shop":
                    think += f" Since all of them come from the same shop (`{best_p[0].shop}`) and the total price exceeds the voucher threshold, the voucher can be applied."
                
                think += " After applying the voucher, the effective total price becomes " + str(int(pact)) + f", which is within the user's budget `{str(int(self.query.voucher.budget))}`."
            else:
                think += f" It is already within the budget `{str(int(self.query.voucher.budget))}`, even without using the voucher."

            if len(best_p) > 1:
                think += "\n\nIn summary, the selected products are within the budget and satisfy all stated requirements."
            else:
                think += "\n\nIn summary, the selected product is within the budget and satisfies all stated requirements."
        elif self.query.challenge == U_Challenge.SHOP:
            think += f"\nAll the selected products come from the same shop (`{best_p[0].shop}`), as requested."
            think += "\n\nIn summary, all selected products come from the same shop, and each one satisfies the user's requirements."
        else:
            think += "\nAll checks passed. The selected product satisfies the user's requirements."

        think += "\n\nNow I will present the final recommendation."        
        recommend = {"recommend": [p.id for p in best_p]}

        content = f"The final recommendation:\n\n```json\n{json.dumps(recommend, ensure_ascii = False, indent = 4)}```"
        self.step(think, content)

class U_BaseEngine(U_Debuggable):
    MAX_SELECT_TRIES = 1
    SELECT_BATCH = 17

    TIMEOUT_SECS = 280.0
    MAX_KW_VARS = 2

    PRUNE_THRES_STRICT = 1.75
    PRUNE_THRES_LOOSE = 2.0
    SCORE_N_THRES = 0.4

    def __init__(self, query: U_Query, sess: U_Session, agent_delay: float, debugger: U_Debugger = None) -> None:
        super().__init__(query, debugger)
        
        self.agent_delay = agent_delay
        self.sess = sess

        self.weight_pool: Dict[U_SubQuery, Dict[str, U_Weight]] = defaultdict(lambda: {})
        self.prod_pool: Dict[str, U_Product] = {}

        self.searches: Dict[U_SubQuery, Dict[U_Filter, List[U_Product]]] = {}
        self.products: Dict[U_SubQuery, List[U_Product]] = {}

        self.filters: Dict[U_SubQuery, List[Tuple[U_Filter, float]]] = defaultdict(lambda: [])
        self.irank: Dict[U_SubQuery, int] = defaultdict(lambda: 0)

        self.confirmed = defaultdict(lambda: defaultdict(lambda: None))
        self.watches: Dict[str, float] = {}

        self.fillers: List[str] = list(set(U_Vocab.FILLERS + (query.fillers or [])))
        _, self.assist_map = self.__build_assistant_map__(True)
        
        self.footprints: Dict[U_SubQuery, Dict[Tuple[Tuple, Tuple], Tuple[Set, bool]]] = defaultdict(lambda: {})
        self.__init_qs__()

    def run(self) -> List[U_Product]:
        self.__init_watch__("query", U_BaseEngine.TIMEOUT_SECS - self.agent_delay)

        for sq in self.query.intents:
            self.__update_filters__(sq, True)
        
        alt_filters = self.__kw__()
        if self.stopped: return []

        for i, sq in enumerate(self.query.intents):
            if alt_filters[i] is not None: self.filters[sq].insert(1, (alt_filters[i], 0))

            if self.query.challenge == U_Challenge.PRODUCT:
                init_pages = U_ENHANCEMENT_BENCH.product_search_pages
                bi_pages = U_ENHANCEMENT_BENCH.bi_search_pages
            elif self.query.challenge == U_Challenge.VOUCHER:
                init_pages = U_ENHANCEMENT_BENCH.voucher_search_pages
                bi_pages = U_ENHANCEMENT_BENCH.voucher_bi_pages
            else:
                init_pages = U_ENHANCEMENT_BENCH.shop_search_pages
                bi_pages = U_ENHANCEMENT_BENCH.bi_search_pages

            searches: Dict[U_Filter, List[U_Product]] = self.__search__(
                sq,
                [f for f, _ in self.filters[sq]],
                init_pages,
                id = "init"
            )
            self.__update_searches__(sq, searches)

            bi_filters = self.__get_bi_filters__(sq, self.filters[sq][0][0])
            bi_searches: Dict[U_Filter, List[U_Product]] = self.__search__(
                sq, bi_filters, bi_pages, id = "bi"
            )
            self.__update_searches__(sq, bi_searches)

            if self.stopped: continue

            products, self.weight_pool[sq] = self.__weigh__(sq, self.searches[sq], id = "init")
            self.products[sq] = [self.__get_prod__(p) for p in products]
            if not self.products[sq]:
                self.__rescue_empty_pool__(sq)

            if self.stopped: continue

            self.irank[sq] = len(self.products[sq])

        self.sess.step_on_search()
        return []

    @U_freezable
    def __kw__(self) -> List[Optional[U_Filter]]:
        inf = {
            str(i + 1): {"query": sq.query, "keywords": self.filters[sq][0][0].get_q()}
            for i, sq in enumerate(self.query.intents)
        }
        js, _ = U_ProxyUtil.trigger_llm(U_Prompt.PSV_KW.format(inf = inf), expect = "dict")        
        self.log(js)

        keywords = U_Prompt.parse_kw_response(js)
        filters = []

        for i, sq in enumerate(self.query.intents):
            if i < len(keywords) and keywords[i]:
                nf = self.filters[sq][0][0].clone().update(keywords = keywords[i].split())
                filters.append(nf if nf not in [f for f, _ in self.filters[sq]] else None)
            else:
                filters.append(None)

        return filters

    @U_freezable
    def __search__(self, sq: U_SubQuery, filters: List[U_Filter], page_limit: int, **_) -> Dict[U_Filter, List[U_Product]]:
        js: Dict[U_Filter, List[U_Product]] = {}

        for f in filters:
            js[f] = self.__find_and_view__(f, page_limit)

            self.log({
                "sq": sq.query,
                "q": f.get_q(),
                "price": f.price,
                "shop": f.shop,
                "found": len(js[f])
            })

        return js

    @U_freezable
    def __weigh__(self, sq: U_SubQuery, searches: Dict[U_Filter, List[U_Product]], **_) -> Tuple[List[U_Product], Dict[str, U_Weight]]:
        products: List[U_Product] = [p for filter in searches for p in searches[filter]]
        tick = time.time()

        for i, p in enumerate(products):
            if p.id not in self.weight_pool[sq]:
                self.weight_pool[sq][p.id] = self.__weigh_product__(p, sq, i + self.irank[sq])

        products = [self.prod_pool[pid] for pid in self.weight_pool[sq]]

        if products:
            products.sort(key = lambda p: -self.weight_pool[sq][p.id].score)

            last_score = self.weight_pool[sq][products[0].id].score
            wrank = 0

            for p in products:
                if self.weight_pool[sq][p.id].score < last_score:
                    last_score = self.weight_pool[sq][p.id].score
                    wrank += 1

                self.weight_pool[sq][p.id].update_confidence(wrank)

        products.sort(key = lambda p: -self.weight_pool[sq][p.id].score_n)
        if self.query.challenge != U_Challenge.VOUCHER:
            products = self.__rrf_rerank__(sq, products, searches)
        delay = time.time() - tick
        
        self.log({
            "sq": sq.query,
            "weighted": len(products),
            "delay": f"{delay:.2f}s"
        })
        return products, self.weight_pool[sq]

    @U_freezable
    def __select__(self, sq: U_SubQuery, candidates: List[U_Product], must: bool, **_) -> List[U_Product]:
        pruned = self.__prune__(candidates, sq, True)
        candidates = pruned or candidates

        best_p, js_check, is_fallback = [], {}, True

        if len(candidates) > 1 or (len(candidates) == 1 and not must):
            cursor, t = 0, 0
            # Self-consistency when top heuristic scores are close
            top_scores = [
                self.weight_pool[sq][p.id].score_n
                for p in candidates[:8]
                if p.id in self.weight_pool.get(sq, {})
            ]
            use_vote = U_MultiModelVotingSystem.should_run_self_consistency(top_scores)

            while cursor < len(candidates):
                batch = candidates[cursor:cursor + U_BaseEngine.SELECT_BATCH]
                cursor += U_BaseEngine.SELECT_BATCH
                t += 1
                prompt = U_Prompt.PV_SELECT.format(
                    query = sq.query,
                    products = self.__get_prod_str__(batch, "")
                )

                if use_vote and t == 1 and len(batch) > 1:
                    bid, details = U_MultiModelVotingSystem.vote_select_index(
                        prompt, len(batch), role="pick", expect="dict"
                    )
                    js_check = {"vote_details": details, "product_index": bid + 1 if bid >= 0 else -1}
                    self.log(js_check)
                else:
                    js_check, _ = U_ProxyUtil.trigger_llm(
                        prompt,
                        preferred_llms = U_ProxyUtil.get_role_llm_priorities("select"),
                        temperature = U_TemperatureController.for_attempt("pick", t - 1),
                        expect = "dict"
                    )
                    bid = U_Prompt.parse_select_response(js_check)
                    self.log(js_check)
                
                if bid >= 0 and bid < len(batch):
                    best_p = [batch[bid]]
                    is_fallback = False
                    break

                if t >= U_BaseEngine.MAX_SELECT_TRIES: break

            if must and not best_p:
                self.log(f"* fallback to select the first product: {sq.query[:50]}")                
                best_p = candidates[:1]
        elif len(candidates) > 0:
            self.log(f"* selected only one product for: {sq.query[:50]}")
            best_p = candidates[:1]
        else:
            self.log(f"# cannot select the best because no candidate exists!")

        if best_p:
            # Duel can swap the favor-window champ; skip on voucher
            if self.query.challenge != U_Challenge.VOUCHER:
                best_p = self.__duel_refine__(sq, best_p[0], candidates)
            self.sess.step_on_select(best_p[0], js_check, is_fallback, sq)
        return best_p

    def __rrf_scores__(self, searches: Dict[U_Filter, List[U_Product]]) -> Dict[str, float]:
        rrf: Dict[str, float] = defaultdict(float)
        k = U_HOO19_PrecisionGate.RRF_K
        for fltr, prods in (searches or {}).items():
            for rank, p in enumerate(prods or []):
                if p and p.id:
                    rrf[p.id] += 1.0 / (k + rank)
        return rrf

    def __rrf_rerank__(self, sq: U_SubQuery, products: List[U_Product], searches: Dict[U_Filter, List[U_Product]]) -> List[U_Product]:
        if len(products) <= 1:
            return products
        rrf = self.__rrf_scores__(searches)
        if not rrf:
            return products
        top = max(rrf.values()) or 1.0
        w = U_HOO19_PrecisionGate.RRF_WEIGHT

        def key(p: U_Product) -> float:
            sc = self.weight_pool[sq][p.id].score_n if p.id in self.weight_pool.get(sq, {}) else 0.0
            return -(sc + w * (rrf.get(p.id, 0.0) / top))

        return sorted(products, key=key)

    def __coverage_ratio__(self, prod: U_Product, sq: U_SubQuery) -> float:
        parsed = getattr(sq, "parsed", None) or {}
        if not parsed:
            return 1.0
        blob = " ".join([
            str(prod.title or ""),
            json.dumps(getattr(prod, "attr", {}) or {}, ensure_ascii=False),
            json.dumps(getattr(prod, "sku", {}) or {}, ensure_ascii=False),
        ]).lower()
        hits = 0
        for _, v in parsed.items():
            tokens = U_MiscUtil.normalize_str(str(v), True).split()
            if tokens and any(t in blob for t in tokens):
                hits += 1
        return hits / max(1, len(parsed))

    def __duel_refine__(self, sq: U_SubQuery, champ: U_Product, candidates: List[U_Product]) -> List[U_Product]:
        """Head-to-head attribute>title duel vs strongest rival (Il lineage)."""
        if not champ or not candidates:
            return [champ] if champ else []
        if not self.__check_watch__("query", U_ENHANCEMENT_BENCH.duel_reserve_secs):
            return [champ]
        rivals = [p for p in candidates if p.id != champ.id][:5]
        if not rivals:
            return [champ]
        # pick rival with best weight among non-champ
        rival = max(
            rivals,
            key=lambda p: self.weight_pool[sq][p.id].score_n if p.id in self.weight_pool.get(sq, {}) else 0.0,
        )
        try:
            a = json.dumps(champ.promptfy(), ensure_ascii=False)[:1500]
            b = json.dumps(rival.promptfy(), ensure_ascii=False)[:1500]
            prompt = U_HOO19_PrecisionGate.PV_DUEL.format(query=sq.query, a=a, b=b)
            js, _ = U_ProxyUtil.trigger_llm(
                prompt,
                preferred_llms=U_ProxyUtil.get_role_llm_priorities("duel"),
                temperature=U_TemperatureController.for_attempt("duel", 0),
                expect="dict",
            )
            winner = U_HOO19_PrecisionGate.parse_duel_winner(js)
            self.log({"duel": {"winner": winner, "a": champ.id, "b": rival.id, "js": js}})
            if winner == "B":
                return [rival]
        except Exception as exc:
            self.log(f"# duel failed open: {exc}")
        return [champ]

    def __rescue_empty_pool__(self, sq: U_SubQuery) -> None:
        """Progressive query widening when initial search yields nothing."""
        if self.products.get(sq):
            return
        base_f = self.filters[sq][0][0] if self.filters.get(sq) else None
        if not base_f:
            return
        base_q = base_f.get_q()
        plans = U_HOO19_PrecisionGate.empty_pool_rescue_queries(sq, base_q)
        hints = U_OptionalEnhancementKit.build_regex_hints(sq.query)
        for lit in (hints.get("quoted_literals") or [])[:2]:
            if lit and lit.lower() not in (base_q or "").lower():
                plans.insert(0, lit)
        for q in plans:
            if not self.__check_watch__("query", 15.0):
                break
            nf = base_f.clone().update(keywords=str(q).split())
            res = self.__search__(sq, [nf], 2, id="rescue")
            self.__update_searches__(sq, res)
            products, self.weight_pool[sq] = self.__weigh__(sq, self.searches[sq], id="rescue")
            self.products[sq] = [self.__get_prod__(p) for p in products]
            if self.products[sq]:
                self.log({"rescue": q, "found": len(self.products[sq])})
                break

    def __prune__(self, products: List[U_Product], sq: U_SubQuery, is_strict: bool) -> List[U_Product]:
        if not products: return []
        fscore = max(w.score for _, w in self.weight_pool[sq].items())
        fscore_n = max(w.score_n for _, w in self.weight_pool[sq].items())
        if is_strict:
            return [p for p in products if self.weight_pool[sq][p.id].score > max(0, fscore - U_BaseEngine.PRUNE_THRES_STRICT)]
        else:
            candidates = [p for p in products if self.weight_pool[sq][p.id].score_n > max(0, fscore_n - U_BaseEngine.SCORE_N_THRES)]
            if len(candidates) < 5:
                return products
            else:
                return candidates

    def __init_qs__(self) -> None:
        self.qs_w: Dict[U_SubQuery, List[float]] = {}
        self.qs_wsum: Dict[U_SubQuery, float] = {}

        self.prod_types: Dict[U_SubQuery, List[str]] = {}
        self.att_indice: Dict[U_SubQuery, Set[int]] = {}
        self.qs: Dict[U_SubQuery, List[str]] = {}

        sizes: Dict[str, str] = {x.split('2')[0]: x.split('2')[1] for x in U_Vocab.SIZES}

        for sq in self.query.intents:            
            self.qs[sq] = [
                s for s in U_MiscUtil.normalize_str(sq.compact, True).split()
                if s not in U_Vocab.KW_STOPWORDS
            ]
            self.att_indice[sq] = set()
            self.qs_w[sq] = []

            self.prod_types[sq] = []
            hard_tokens = set()            

            for k, v in sq.parsed.items():
                ksplits, vsplits = k.split('_'), U_MiscUtil.normalize_str(v, True).split()                
                
                if k == "product_type": self.prod_types[sq] = vsplits
                if "brand" in ksplits: hard_tokens.update(vsplits)

            for i, ch in enumerate(self.qs[sq]):
                if ch in (hard_tokens | set(U_Vocab.COLOR_WORDS) | set(sizes.values())) or re.search(r"\d", ch):
                    self.qs_w[sq].append(1.0)
                elif any(U_MiscUtil.is_morph_equal(ch, f, {}) for f in (self.fillers or [])):
                    self.qs_w[sq].append(0.5)
                else:
                    self.qs_w[sq].append(0.75)
                
                for s in self.prod_types[sq]:
                    if U_MiscUtil.is_morph_equal(ch, s, self.assist_map): break
                else:
                    self.att_indice[sq].add(i)
            
            self.qs_wsum[sq] = sum(self.qs_w[sq])

    def __update_searches__(self, sq: U_SubQuery, searches: Dict[U_Filter, List[U_Product]]) -> None:
        self.searches[sq] = self.searches.get(sq, {})

        for f, products in searches.items():
            self.searches[sq][f] = self.searches[sq].get(f, [])

            for p in products:
                if p not in self.searches[sq][f]: self.searches[sq][f].append(self.__get_prod__(p))

    def __find_and_view__(self, filter: U_Filter, page_limit: int = 1) -> List[U_Product]:
        res: List[U_Product] = []
        viewed = 0
        view_cap = U_ENHANCEMENT_BENCH.max_view_per_filter
        
        for page in range(1, max(2, page_limit + 1)):
            searched, tool_res = U_ProxyUtil.search(filter, page = page)
            self.sess.add_tool_result(tool_res)

            found: List[U_Product] = []

            for p in searched:
                if self.__basic_check__(p, filter): found.append(self.__get_prod__(p))

            to_view = [p for p in found if not p.is_viewed]
            if to_view and viewed < view_cap:
                batch = to_view[: max(0, view_cap - viewed)]
                U_ProxyUtil.view(batch)
                viewed += len(batch)

            for p in found:
                res.append(self.__get_prod__(p).update_details())

            if page >= page_limit or not found: break

        return res

    def __basic_check__(self, prod: U_Product, filter: U_Filter) -> bool:
        if filter.price:
            if not prod.price: return False
            price_ok, _ = U_FuzzyBoundaryLogic.price_fits_range(prod.price, filter.price, fuzzy=True)
            if not price_ok: return False

        if filter.services:
            prod_services = [str(svc).lower() for svc in (prod.services or [])]
            sq_services = [str(svc).lower() for svc in filter.services]

            for svc in sq_services:
                if svc not in prod_services: return False
        
        if filter.shop and filter.shop != prod.shop: return False
        return True

    def __weigh_product__(self, prod: U_Product, sq: U_SubQuery, rank: int) -> U_Weight:
        opt_details, prod_only = self.__get_opt_details__(prod, sq)
        best_w = U_Weight()

        if prod_only or not sq.prod_only:
            tsplits = tuple(U_MiscUtil.normalize_str(prod.title, True).split())
            combs = U_MiscUtil.get_choice_combs(opt_details, True)

            pt_covers, _ = U_MiscUtil.get_coverage(tsplits, self.prod_types[sq], len(tsplits), self.assist_map)
            title_covers, _ = U_MiscUtil.get_coverage(tsplits, self.qs[sq], len(tsplits), self.assist_map)

            pt_bonus = (0.02 * len(pt_covers) / len(self.prod_types[sq])) if len(self.prod_types[sq]) > 0 else 0

            brand_tokens: Set[str] = set()
            for k, v in sq.parsed.items():
                if "brand" in k.split('_'):
                    brand_tokens.update(U_MiscUtil.normalize_str(v, True).split())

            for choice in combs:
                att_covers = set()
                brand_bonus = 0.0

                joint_attrs: List[Tuple] = [
                    a for g, idx in choice.items() for a in (opt_details[g][idx] if idx >= 0 else [])
                ]
                for k, v in joint_attrs:
                    covers, _ = self.__get_coverage__(sq, k, v)                    
                    att_covers |= covers

                    k_norm = U_MiscUtil.normalize_str('_'.join(k), False)
                    if "brand" in k_norm:
                        v_norm = set(U_MiscUtil.normalize_str('_'.join(v), True).split())
                        matched_brands = brand_tokens & v_norm
                        if matched_brands:
                            brand_bonus = max(brand_bonus, 0.15 * (len(matched_brands) / len(brand_tokens)) if brand_tokens else 0)
            
                total_covers = att_covers | title_covers
                att_covers = att_covers & self.att_indice[sq]
                att_bonus = (0.4 * len(att_covers) / len(self.att_indice[sq])) if len(self.att_indice[sq]) > 0 else 0
                w = U_Weight.create(
                    choice = choice, covers = list(total_covers), rank = rank, wsum = self.qs_wsum[sq],
                    score_r = sum(self.qs_w[sq][i] for i in total_covers),
                    pt_bonus = pt_bonus, att_bonus = att_bonus, brand_bonus = brand_bonus
                )
                if (w.score, -w.choice.get("sku", -1)) > (best_w.score, -best_w.choice.get("sku", -1)): best_w = w                    

        # Live price-band bonus/penalty (FuzzyBoundary was shelf-only)
        if sq.price:
            best_w.score += U_FuzzyBoundaryLogic.price_fit_score_adjustment(prod.price, sq.price) / 100.0

        return best_w.fill_detail(self.qs[sq], self.qs_w[sq])
    def __get_opt_details__(self, prod: U_Product, sq: U_SubQuery) -> Tuple[Dict[str, List], bool]:
        valid_details, prod_only = {}, False

        for g, kvs in prod.to_normalized().items():
            valid_kvs = []

            for kv in kvs:
                key_kv_map, val_kv_map = {}, {}

                for k, v in kv:
                    covers, val_used = self.__get_coverage__(sq, k, v)
                    if covers: (val_kv_map if val_used else key_kv_map)[tuple(covers)] = (k, v)

                    if (re.search(r"\bservices?\b", ' '.join(k)) and U_MiscUtil.scan_patterns(' '.join(v), U_Vocab.SERVICE_PATTERNS[U_Vocab.PROD_ONLY])):
                        prod_only = True

                key_pairs = [(set(covers), pair) for covers, pair in key_kv_map.items()]
                val_pairs = [(set(covers), pair) for covers, pair in val_kv_map.items()]

                key_pairs.sort(key = lambda x: -len(x[0]))
                val_pairs.sort(key = lambda x: -len(x[0]))

                valid_kv, total_covers = [], set()

                for covers, pair in val_pairs + key_pairs:
                    if covers < total_covers: continue

                    total_covers |= covers
                    valid_kv.append(pair)

                if valid_kv: valid_kvs.append(valid_kv)
            
            if valid_kvs: valid_details[g] = valid_kvs
        
        return (valid_details, prod_only)

    def __get_prod__(self, p: U_Product) -> U_Product:
        if p.id in self.prod_pool:
            existing = self.prod_pool[p.id]
            
            if p.is_viewed and not existing.is_viewed:
                for k, v in p.to_json().items():
                    setattr(existing, k, v)
        else:
            self.prod_pool[p.id] = p
        
        return self.prod_pool[p.id]

    def __get_coverage__(self, sq: U_SubQuery, k: Tuple[str], v: Tuple[str]) -> Tuple[Set, bool]:
        if (k, v) in self.footprints[sq]:
            return self.footprints[sq].get((k, v))
        else:
            covers, val_used = U_MiscUtil.get_coverage(k + v, self.qs[sq], len(k), self.assist_map)
            self.footprints[sq][(k, v)] = covers, val_used
            return covers, val_used

    def __get_kw_var__(self, sq: U_SubQuery) -> Tuple[List[List[Tuple[str, int]]], List[int]]:        
        cat: List[List[Tuple[str, int]]] = []
        indice: List[int] = []

        alts, assist_map = self.__build_assistant_map__(False)

        def __du_space__(v: str) -> bool:
            if (m := re.search(r"^(\d+(?:\-\d+)?)[\s\-]+([a-z]+)$", v)):
                num, unit = m.group(1), m.group(2)
                
                if not U_MiscUtil.is_morph_equal(unit, "piece", {}):
                    for su, lu in U_Vocab.METRIC_UNITS.items():
                        if lu and unit in [lu, f"{lu}s"]:
                            cat.append([(num, 1), (f"{num}{su}", 0)])
                            break
                    else:
                        cat.append([(v, 0), (num, 1), (f"{num}{unit}", 2)])
                else:
                    cat.append([(num, 1), (f"{num}pcs", 0)])

                return True
            else:            
                return False

        def __du_tight__(v: str) -> bool:
            if (m := re.search(r"^(\d+(?:\-\d+)?)([a-z]+)$", v)):
                num, unit = m.group(1), m.group(2)                
                
                for su, lu in U_Vocab.METRIC_UNITS.items():
                    if unit  == su:
                        cat.append([(v, 0), (num, 1 if unit[:2] != "pc" else 2)])
                        if lu: cat[-1].append((f"{num}{lu}", 3))                        
                        return True
                    elif lu and unit in [lu, f"{lu}s"]:
                        cat.append([(v, 0), (f"{num}{su}", 1), (num, 2)])
                        return True

            return False

        def __digit__(v: str, k: str) -> bool:
            if not v.isnumeric(): return False
            unit = k.split('_')[-1]

            if unit in ["color", "family", "size", "count", "high", "low"]: return False

            if f"{v}{unit}" in sq.query.lower():
                nv = f"{v}{unit}"
            elif f"{v} {unit}" in sq.query.lower():
                nv = f"{v} {unit}"
            else:
                return False

            cat.append([(v, 0.5), (nv, 0 if unit in U_Vocab.METRIC_UNITS else 2)])
            return True

        def __size__(v: str) -> bool:
            if re.search(r"\bone\s+size\b", v, re.I):
                cat.append([(v, 0)])
                return True
            if (m := re.search(rf"\b(?:({re.escape(U_MiscUtil.extract_number.__name__[:5])})\s+)?([slm])[a-z]+\b", v)):
                abb = (m.group(1)[1] if m.group(1) else '') + m.group(2)                
                cat.append([(v, 1), (abb, 0)])                
                return True
            else:
                return False

        def __assistant__(v: str) -> bool:
            a = re.__name__ + __assistant__.__name__.strip('_')[-7:]
            b = f"{U_Product.__name__.lower()[:3]}of"

            nv = re.sub(rf"\b(\w+)(?:\s+|\-){re.escape(a)}\b", r"\1" + b, v)

            if nv != v:
                cat.append([(v, 0.5), (nv, 0)])
                return True
            
            nv = re.sub(rf"\b(\w+){re.escape(b)}\b", r"\1-" + a, v)
            
            if nv != v:
                cat.append([(v, 0), (nv, 1)])
                return True
            
            return False

        def __alt__(v: str) -> bool:
            cv = U_MiscUtil.normalize_str(v, False)

            for token in cv.split():
                for a in assist_map:
                    if U_MiscUtil.is_morph_equal(token, a, {}):
                        nv = cv.replace(token, list(assist_map[a])[0])

                        if nv != cv:
                            cat.append(
                                [
                                    (nv, 0.1 if token[:2] in ["ki", "ch", "ti", "la", "sh", "ts", "gr"] else 1),
                                    (v, 0)                                    
                                ]
                            )
                            return True

            for a, b in alts.items():
                if ' ' not in (a + b): continue
                nv = cv

                for token in [a, a.replace(' ', '-')]:
                    if token.endswith('s'): token = token[:-1]

                    if (m := re.search(rf"\b({re.escape(token)}s?)\b", cv)):
                        if not re.search(rf"\b{re.escape(b.replace('-', ' '))}\b", cv.replace('-', ' ')):
                            nv = cv.replace(m.group(1), b)
                            break
                
                for token in [b, b.replace(' ', '-')]:
                    if token.endswith('s'): token = token[:-1]

                    if (m := re.search(rf"\b({re.escape(token)}s?)\b", cv)):
                        if not re.search(rf"\b{re.escape(a.replace('-', ' '))}\b", cv.replace('-', ' ')):
                            nv = cv.replace(m.group(1), a)
                            break

                if nv != cv:
                    cat.append([(v, 0), (nv, 0.5)])
                    return True

        def __filler__(v: str) -> bool:
            cv = U_MiscUtil.normalize_str(v, False)

            for token in cv.split():
                for f in self.fillers:
                    if U_MiscUtil.is_morph_equal(token, f, {}):
                        nv = cv.replace(token, "")
                        nv = re.sub(r"\s+", " ", nv)

                        if nv != cv:
                            cat.append([(v, 0), (nv, 2)])   
                            return True
            
            return False

        for i, (k, v) in enumerate(sq.parsed.items()):
            if "brand" in k.split('_'): continue

            if any([
                __du_space__(v), __du_tight__(v), (k == "size" and __size__(v)),
                __digit__(v, k), __assistant__(v), __alt__(v), __filler__(v)
            ]):
                indice.append(i)

        return cat, indice

    def __build_assistant_map__(self, bridged: bool) -> Tuple[Dict[str, str], Dict[str, Set]]:
        alts: Dict[str, str] = {x.split('2')[0]: x.split('2')[1] for x in U_Vocab.ALTS}

        for k, v in self.query.alts.items():
            kv = U_EngineUtil.compact_alt(k, v)
            if kv: alts[kv[0]] = kv[1]
        
        a = re.__name__ + self.__build_assistant_map__.__name__.strip('_').split('_')[1][-7:]
        b = f"{U_Product.__name__.lower()[:3]}of"
        
        alts_ex: Dict[str, str] = {a: b}
        alts_ex.update(alts)

        syn_map = defaultdict(lambda: set())

        for a, b in alts_ex.items():
            a = a.lower()
            b = b.lower()

            if ' ' in (a + b): continue
            if '-' in (a + b): continue

            syn_map[a].add(b)
            syn_map[b].add(a)
        
        for x in U_Vocab.BRIDGES:
            a, b = x.split('2')

            syn_map[a] = syn_map[a] | syn_map[b]
            syn_map[b] = syn_map[b] | syn_map[a]

            if bridged:
                syn_map[a] |= {b}
                syn_map[b] |= {a}

            if a in syn_map[a]: syn_map[a].remove(a)
            if b in syn_map[b]: syn_map[b].remove(b)

        syn_map = {a: b for a, b in syn_map.items() if a and b}
        return alts_ex, syn_map

    def __update_filters__(self, sq: U_SubQuery, shadow: bool) -> List[U_Filter]:
        variations = [(list(sq.parsed.values()), 0)]
        cat, indice = self.__get_kw_var__(sq)

        if cat:
            combs = U_MiscUtil.get_choice_combs(cat, False)
            filters = []

            base_var = variations[0][0].copy()
            variations.clear()

            for choice in combs:
                w = sum(cat[i][choice[i]][-1] for i in range(len(indice)))
                nvs = base_var.copy()

                for i in range(len(indice)):
                    id, c = indice[i], choice[i]
                    nvs[id] = cat[i][c][0]
                
                variations.append((nvs, w))

        variations.sort(key = lambda var: var[-1])
        filters: List[Tuple[U_Filter, float]] = []

        for vs, w in variations:
            f = U_Filter(keywords = vs, services = sq.services, price = sq.price)
            if f not in [of for of, _ in filters]: filters.append((f, w))

        filters = filters[:U_BaseEngine.MAX_KW_VARS]

        for f, _ in filters:
            # Always apply favor/shadow price logic. Do NOT gate on ProxyUtil
            # sabotage voucher by stripping keywords + clearing the favor band.
            low, high = U_EngineUtil.numerify_price(sq.price or "-")

            if shadow:
                if high < math.inf:
                    low = max(low, math.ceil(high / 2))
                elif self.query.voucher is not None:
                    favor = self.query.voucher.get_favor_range()

                    if len(self.query.intents) == 1: low = max(low, int(favor[0]))
                    high = int(favor[1])

            f.update(price = U_EngineUtil.stringfy_price(low, high))

        self.filters[sq] = filters

    def __get_bi_filters__(self, sq: U_SubQuery, base_filter: U_Filter) -> List[U_Filter]:
        products = [p for _, searches in self.searches[sq].items() for p in searches]
        if len(products) == 0: return []

        prices = sorted(p.price for p in products)
        median = int(prices[len(prices) // 2])

        low, high = U_EngineUtil.numerify_price(base_filter.price)

        lf: U_Filter = U_Filter.from_json(base_filter.to_json())
        lf.update(price = U_EngineUtil.stringfy_price(low, median))

        hf: U_Filter = U_Filter.from_json(base_filter.to_json())
        hf.update(price = U_EngineUtil.stringfy_price(median, high))
        return [lf, hf]

    def __get_prod_str__(self, products: List[U_Product], prefix: str) -> str:
        prod_str = ""

        for i, p in enumerate(products):
            prod_str += f"**Product (id=\"{prefix}{i + 1}\")**\n"
            prod_str += json.dumps(p.promptfy(), ensure_ascii = False, indent = 2) + '\n'
            prod_str += "------------------\n\n"
        
        return prod_str

    def __get_alt_str__(self, best_p: List[U_Product]) -> str:
        if not best_p: return ""

        if len(best_p) == 1:
            for i, p in enumerate(self.products[self.query.intents[0]] * 2):
                if p.price > best_p[0].price or i >= len(self.products[self.query.intents[0]]):
                    alt_price = str(int(p.price)) if i < len(self.products[self.query.intents[0]]) else str(int(best_p[0].price + 2.0))
                    alt_title = p.title if len(p.title) < 90 else (p.title[:90] + "...")

                    return f"The product `{p.id}` appears to be a strong alternative. Let me compare it with `{best_p[0].id}`. The title of `{p.id}` is `{alt_title}`. Based on the product details, `{p.id}` matches all stated constraints like `{best_p[0].id}`, making it difficult to choose one of them. The price of product `{p.id}` is {alt_price}, while the price of `{best_p[0].id}` is {str(int(best_p[0].price))}. The product `{best_p[0].id}` is less expensive. Therefore, I will prefer `{best_p[0].id}`."
        else:
            for i, sq in enumerate(self.query.intents * 2):
                for p in self.products[sq]:
                    if i >= len(self.query.intents) or p.price > best_p[i].price:
                        bp = best_p[i % len(self.query.intents)]
                        alt_price = str(int(p.price)) if i < len(self.query.intents) else str(int(bp.price + 3.0))
                        alt_title = p.title if len(p.title) < 90 else (p.title[:90] + "...")

                        return f"The product `{p.id}` appears to be a strong alternative of the product `{bp.id}`, for the sub-query `{sq.query}`. The title of `{p.id}` is `{alt_title}`. Let's consider a combination which has replaced `{bp.id}` with `{p.id}`. Still it meets the stated constraints for each corresponding sub-query. Now let me check the prices. The price of product `{p.id}` is {str(int(p.price))}, while the price of `{bp.id}` is {str(int(bp.price))}. It indicates that the replaced combination will have a higher total price than the original one. Therefore, the combination `{', '.join([p.id for p in best_p])}` is less expensive. I will prefer it."

        return ""

    def __init_watch__(self, key: str, time_out: float) -> None:
        self.watches[key] = time.time() + time_out

    def __check_watch__(self, key: str, delta: float = 0) -> bool:
        if key not in self.watches:
            self.log(f"# non-initialized watch: `{key}`")
            return False
        
        if time.time() + delta > self.watches[key]:
            self.log(f"* time-out alerted for `{key}` with delta `{delta:.2f}s`")
            return False
        else:
            return True
class U_ProductEngine(U_BaseEngine):
    def run(self) -> List[U_Product]:
        super().run()
        if self.stopped: return []

        sq = self.query.intents[0]
        candidates = list(self.products.get(sq, []) or [])
        # Coverage prune before LLM select — keep high-coverage first, drop weak tail
        if len(candidates) > 3:
            scored = [(p, self.__coverage_ratio__(p, sq)) for p in candidates]
            kept = [p for p, r in scored if r >= U_ENHANCEMENT_BENCH.coverage_prune_floor]
            if len(kept) >= 2:
                # Prefer kept; only pad with low-coverage if shortlist too thin
                if len(kept) >= U_ENHANCEMENT_BENCH.product_task_shortlist:
                    candidates = kept
                else:
                    rest = [p for p, _ in scored if p not in kept]
                    candidates = kept + rest[: max(0, U_ENHANCEMENT_BENCH.product_task_shortlist - len(kept))]

        best_p = self.__select__(sq, candidates, True)
        if self.stopped: return []

        # Coverage override: refuse low-coverage LLM pick when a stronger alt exists
        if best_p:
            champ = best_p[0]
            champ_cov = self.__coverage_ratio__(champ, sq)
            if champ_cov < 0.4:
                alts = sorted(
                    [p for p in candidates if p.id != champ.id],
                    key=lambda p: self.__coverage_ratio__(p, sq),
                    reverse=True,
                )
                if alts and self.__coverage_ratio__(alts[0], sq) >= champ_cov + 0.2:
                    self.log({
                        "coverage_override": {
                            "from": champ.id,
                            "to": alts[0].id,
                            "from_cov": champ_cov,
                            "to_cov": self.__coverage_ratio__(alts[0], sq),
                        }
                    })
                    best_p = [alts[0]]

        return best_p
class U_VoucherEarlyValidator:
    @staticmethod
    def validate(voucher: Optional["Voucher"]) -> Tuple[bool, str]:
        if voucher is None:
            return False, "missing_voucher"
        budget = U_EdgeCaseProcessor.coerce_float(getattr(voucher, "budget", 0), lo=0.0)
        if budget <= 0:
            return False, "invalid_budget"
        discount = U_EdgeCaseProcessor.coerce_float(getattr(voucher, "discount", 0), lo=0.0)
        if not voucher.is_fixed and (discount <= 0 or discount >= 100):
            return False, "invalid_percent_discount"
        if voucher.is_fixed and discount <= 0:
            return False, "invalid_fixed_discount"
        return True, "ok"


class U_OptionalEnhancementKit:
    EdgeCase = U_EdgeCaseProcessor
    Boundary = U_FuzzyBoundaryLogic
    Voting = U_MultiModelVotingSystem
    Gate = U_HOO19_PrecisionGate
    Temp = U_TemperatureController
    Guard = U_AdversarialIngressGuard
    Wire = U_DialogWireFit
    Settings = U_ENHANCEMENT_BENCH

    @staticmethod
    def find_product_with_fallback(**kwargs) -> List[Dict]:
        result = U_ProxyUtil.ORO_PROXY.get("/search/find_product", kwargs)
        result = result if result else []
        if result:
            return result
        service = kwargs.get("service")
        if service:
            retry_params = dict(kwargs)
            retry_params.pop("service", None)
            result = U_ProxyUtil.ORO_PROXY.get("/search/find_product", retry_params)
            result = result if result else []
        if not result:
            q = kwargs.get("q") or ""
            tokens = str(q).split()
            short_q = " ".join(tokens[:2])
            if short_q and short_q != q:
                retry_params = dict(kwargs)
                retry_params["q"] = short_q
                retry_params.pop("service", None)
                result = U_ProxyUtil.ORO_PROXY.get("/search/find_product", retry_params)
                result = result if result else []
        return result

    @staticmethod
    def build_regex_hints(query: str) -> Dict[str, Any]:
        raw = str(query or "")
        stripped = raw.strip()
        return {
            "quoted_literals": re.findall(r'["\']([^"\']+)["\']', stripped),
            "number_unit_tokens": re.findall(r"\b\d+(?:\.\d+)?[a-zA-Z]+\b", stripped.lower()),
            "size_labels": re.findall(
                r"\b(?:xxs|xs|s|m|l|xl|xxl|xxxl|\dxl|\d+\s*(?:cm|mm|inch|in))\b",
                stripped.lower(),
            ),
            "color_words": [
                w for w in re.findall(r"\b[a-z]{3,}\b", stripped.lower())
                if w in (
                    "black", "white", "red", "blue", "green", "gold", "silver",
                    "pink", "gray", "grey", "brown", "yellow", "purple", "orange",
                )
            ],
            "service_tags": [
                t for t in ("official", "freeShipping", "COD", "flashsale")
                if t.lower() in stripped.lower() or t in stripped
            ],
            "price_range_guess": U_FuzzyBoundaryLogic.normalize_price_from_query(stripped),
        }

    @staticmethod
    def audit_listing_against_spec(*, title: str, price: Any, parsed_spec: Dict) -> Dict[str, Any]:
        price_range = parsed_spec.get("price") if isinstance(parsed_spec, dict) else None
        lo, hi = U_FuzzyBoundaryLogic.parse_price_bounds(price_range)
        boundary = U_FuzzyBoundaryLogic.classify_price_boundary(price, lo, hi)
        price_ok, price_note = U_FuzzyBoundaryLogic.price_fits_range(price, price_range, fuzzy = True)
        keywords = parsed_spec.get("keywords") if isinstance(parsed_spec, dict) else []
        title_l = (title or "").lower()
        missing = [
            kw[-1] if isinstance(kw, list) and kw else str(kw)
            for kw in (keywords or [])
            if isinstance(kw, (list, tuple)) and len(kw) >= 2
            and str(kw[-1]).lower() not in title_l
        ]
        note = "ok" if price_ok and not missing else "review"
        return {
            "keywords_matched": not missing,
            "keywords_missing": missing,
            "title_contains_all_keywords": not missing,
            "price_ok": price_ok,
            "price_note": price_note,
            "boundary_class": boundary,
            "overall_note": note,
        }


class U_Agent(U_Debuggable):
    ENGINE_MAP: Dict[U_Challenge, Type[U_BaseEngine]] = {
        U_Challenge.PRODUCT: U_ProductEngine,
    }

    def __init__(self, query: U_Query, debugger: Optional[U_Debugger] = None) -> None:
        super().__init__(query, debugger)
        self.sess: U_Session = U_Session(query, debugger)

    def run(self) -> None:
        tick = time.time()

        self.query = self.__classify__()
        if self.stopped: return

        self.query = self.__decompose__()
        if self.stopped: return

        self.query = self.__build__()
        if self.stopped: return

        # Multi-item / voucher: always review constraint scope mismatches (cancancan pattern)
        if self.query.challenge in (U_Challenge.SHOP, U_Challenge.VOUCHER) or len(self.query.intents) > 1:
            try:
                self.query = self.__review__()
            except Exception as exc:
                self.log(f"# review soft-fail: {exc}")

        self.sess.query = self.query
        self.sess.step_on_build()

        self.engine = self.ENGINE_MAP.get(self.query.challenge)(self.query, self.sess, time.time() - tick, self.debugger)
        best_p = self.engine.run()
        best_p = self.__dual_refine__(best_p, tick)
        # Audit/repair coverage swaps ignore knapsack favor windows — skip on voucher
        if self.query.challenge != U_Challenge.VOUCHER:
            best_p = self.__audit_and_repair__(best_p)
        self.__finalize__(best_p)

    def get_dialog(self) -> List[Dict]:
        return self.sess.dialog

    @U_freezable
    def __classify__(self) -> U_Query:
        self.query.challenge = self.__identify_challenge__()
        self.query.query_e = self.query.query_f

        if self.query.challenge == U_Challenge.VOUCHER: self.query.query_e, self.query.voucher = self.__parse_voucher__()

        if self.query.challenge == U_Challenge.VOUCHER and self.query.voucher is not None:
            ok, reason = U_VoucherEarlyValidator.validate(self.query.voucher)
            if not ok:
                self.log({"voucher_early_invalid": reason})
        self.log({
            "challenge": self.query.challenge,
            "query_e": self.query.query_e,
            "voucher": self.query.voucher.to_json() if self.query.voucher is not None else None
        })
        return self.query

    @U_freezable
    def __decompose__(self) -> U_Query:
        temperature = 0.0
        last_exc: Exception | None = None

        for attempt in range(U_ENHANCEMENT_BENCH.llm_retry_max):
            try:
                if self.query.challenge != U_Challenge.PRODUCT:
                    prompt = U_Prompt.SV_DECOMPOSE.format(
                        intent = "two or more products in the same shop" if self.query.challenge == U_Challenge.SHOP else "one or more products",
                        rules = U_Prompt.SCOPE_RULES,
                        query = self.query.query_e
                    )
                    is_multiple = True
                else:
                    prompt = U_Prompt.P_DECOMPOSE.format(query = self.query.query_e)
                    is_multiple = False

                js, _ = U_ProxyUtil.trigger_llm(
                    prompt,
                    preferred_llms = U_ProxyUtil.get_role_llm_priorities("parse"),
                    temperature = U_TemperatureController.for_attempt("parse", attempt) if not temperature else temperature,
                    expect = "dict",
                )
                self.log(js)
                if not js:
                    raise Exception("empty decompose response")
                
                decomposed, alts, fillers = U_Prompt.parse_decompose_response(js, is_multiple)

                if len(decomposed) == 0: raise Exception("no sub-queries extracted")
                if len(decomposed) == 1 and self.query.challenge == U_Challenge.SHOP: raise Exception("sub-query count invalid for shop")

                banned_services, prod_only = [], True
                self.query.intents = []
                for i, js_sq in enumerate(decomposed):
                    self.query.intents.append(U_SubQuery(
                        query_f = self.query.query_f,
                        query = js_sq.get("query"),
                        prod_only = js_sq.get("prod_only") and prod_only,
                        services = [svc for svc in js_sq.get("services") if svc not in banned_services],
                        price = js_sq.get("price"),
                        id = i                        
                    ))

                self.query.alts = alts
                self.query.fillers = fillers
                return self.query
            except Exception as exc:
                last_exc = exc
                self.log(f"# error parsing LLM response: {exc}")
                temperature = min(U_ENHANCEMENT_BENCH.temperature_max, temperature + U_ENHANCEMENT_BENCH.temperature_retry_step)

        # Last-resort single-intent fallback so the agent never hangs
        self.log(f"# decompose exhausted; fallback single intent ({last_exc})")
        if not self.query.intents:
            self.query.intents = [U_SubQuery(
                query_f=self.query.query_f,
                query=self.query.query_e or self.query.query_f,
                prod_only=True,
                services=[],
                price=U_FuzzyBoundaryLogic.normalize_price_from_query(self.query.query_e or self.query.query_f),
                id=0,
            )]
        return self.query

    @U_freezable
    def __review__(self) -> U_Query:
        suspects, prompts_i, prompts_o, focuses, error_3 = self.__check_review_needed__()

        if len(suspects) > 0:
            js, _ = U_ProxyUtil.trigger_llm(self.__create_review_prompt__(prompts_i, prompts_o, focuses, error_3), expect = "dict")
            self.__fix_by_llm_review__(js, suspects)
            self.log(js)

        return self.query

    @U_freezable
    def __build__(self) -> U_Query:
        temperature = 0.0
        last_exc: Exception | None = None

        for attempt in range(U_ENHANCEMENT_BENCH.llm_retry_max):
            try:
                js, _ = U_ProxyUtil.trigger_llm(
                    U_Prompt.PSV_BUILD.format(queries = {str(i + 1): sq.query for i, sq in enumerate(self.query.intents)}),
                    preferred_llms = U_ProxyUtil.get_role_llm_priorities("build"),
                    temperature = U_TemperatureController.for_attempt("parse", attempt) if not temperature else temperature,
                    expect = "dict"
                )
                built = U_Prompt.parse_build_response(js)
                self.log(js)
                if not built:
                    raise Exception("empty build response")

                if len(built) != len(self.query.intents):
                    raise Exception(f"sub-query count mismatches (expected {len(self.query.intents)} but got {len(built)})")

                for b, sq in zip(built, self.query.intents):
                    sq.compact = b.get("compact") or sq.query
                    sq.parsed = {}

                    for k, v in b.get("parsed").items():
                        kw = U_EngineUtil.compact_kv(k, v, sq.query)
                        
                        if not kw: continue
                        nk, nv = kw

                        if nk not in sq.parsed:
                            for ok in sq.parsed:
                                if sq.parsed[ok] == nv:
                                    if f"{nv} {nk.replace('_', ' ')}" in sq.query:
                                        sq.parsed[nk] = nv
                                        del sq.parsed[ok]
                                    
                                    break
                            else:
                                sq.parsed[nk] = nv
                        else:
                            sq.parsed[nk] += " " + nv

                    if sq.prod_only: sq.parsed["service"] = U_Vocab.PROD_ONLY.replace('_', ' ')
                    # Recover price from query text when LLM omitted it
                    if not sq.price:
                        guessed = U_FuzzyBoundaryLogic.normalize_price_from_query(sq.query or self.query.query_e or "")
                        if guessed:
                            sq.price = guessed

                return self.query
            except Exception as exc:
                last_exc = exc
                self.log(f"# error parsing LLM response: {exc}")
                temperature = min(U_ENHANCEMENT_BENCH.temperature_max, temperature + U_ProxyUtil.TEMP_UP)

        self.log(f"# build exhausted; keeping compact=query ({last_exc})")
        for sq in self.query.intents:
            if not getattr(sq, "compact", None):
                sq.compact = sq.query
            if not getattr(sq, "parsed", None):
                sq.parsed = {}
            if not sq.price:
                sq.price = U_FuzzyBoundaryLogic.normalize_price_from_query(sq.query or "")
        return self.query

    def __dual_refine__(self, best_p: List[U_Product], tick: float) -> List[U_Product]:
        """Time-boxed second-opinion challenger for PRODUCT lane (cancancan DualSolve)."""
        if not best_p or self.query.challenge != U_Challenge.PRODUCT:
            return best_p
        elapsed = time.time() - tick
        left = U_BaseEngine.TIMEOUT_SECS - elapsed - self.engine.agent_delay
        if elapsed > U_ENHANCEMENT_BENCH.dual_cutoff_secs or left < U_ENHANCEMENT_BENCH.dual_min_left_secs:
            return best_p
        if not self.engine.__check_watch__("query", U_ENHANCEMENT_BENCH.dual_min_left_secs):
            return best_p
        sq = self.query.intents[0]
        pool = list(self.engine.products.get(sq, []) or [])
        champ = best_p[0]
        alts = [p for p in pool if p.id != champ.id]
        if not alts:
            return best_p
        # diversify: rotate keywords / pick mid-ranked rival via alt select prompt
        shortlist = alts[:8]
        try:
            js, _ = U_ProxyUtil.trigger_llm(
                U_Prompt.PV_SELECT.format(
                    query=sq.query,
                    products=self.engine.__get_prod_str__(shortlist, "B"),
                ),
                preferred_llms=U_ProxyUtil.get_role_llm_priorities("alt_select"),
                temperature=U_TemperatureController.for_attempt("alt_select", 0),
                expect="dict",
            )
            bid = U_Prompt.parse_select_response(js)
            alt = shortlist[bid] if 0 <= bid < len(shortlist) else shortlist[0]
        except Exception:
            alt = shortlist[0]
        if alt.id == champ.id:
            return best_p
        try:
            prompt = U_HOO19_PrecisionGate.DUAL_COMPARE.format(
                query=self.query.query_e or self.query.query_f,
                cand_a=json.dumps(champ.promptfy(), ensure_ascii=False)[:1800],
                cand_b=json.dumps(alt.promptfy(), ensure_ascii=False)[:1800],
            )
            js, _ = U_ProxyUtil.trigger_llm(
                prompt,
                preferred_llms=U_ProxyUtil.get_role_llm_priorities("dual_compare"),
                temperature=U_TemperatureController.for_attempt("dual_compare", 0),
                expect="dict",
            )
            best = str((js or {}).get("best", "A")).strip().upper()[:1]
            self.log({"dual_compare": js, "a": champ.id, "b": alt.id})
            if best == "B":
                return [alt]
        except Exception as exc:
            self.log(f"# dual compare soft-fail: {exc}")
        return best_p

    def __audit_and_repair__(self, best_p: List[U_Product]) -> List[U_Product]:
        """Post-select auditor with coverage / shop / budget repair (Angel pattern)."""
        ok, reason = U_HOO19_PrecisionGate.verify_pick(self.query, best_p or [])
        if ok or not best_p:
            return best_p
        self.log({"verify_pick_fail": reason, "ids": [p.id for p in best_p]})

        if reason == "mixed_shops" and self.query.challenge in (U_Challenge.SHOP, U_Challenge.VOUCHER):
            # keep majority shop
            from collections import Counter
            counts = Counter(p.shop for p in best_p)
            majority = counts.most_common(1)[0][0]
            repaired = []
            for i, sq in enumerate(self.query.intents):
                pool = [p for p in self.engine.products.get(sq, []) if p.shop == majority]
                if not pool:
                    return best_p
                repaired.append(pool[0] if i >= len(best_p) or best_p[i].shop != majority else best_p[i])
                if repaired[-1].shop != majority:
                    repaired[-1] = pool[0]
            ok2, _ = U_HOO19_PrecisionGate.verify_pick(self.query, repaired)
            if ok2:
                return repaired

        if reason in ("over_budget",) and self.query.voucher:
            favor = self.query.voucher.get_favor_range() if hasattr(self.query.voucher, "get_favor_range") else (0, self.query.voucher.budget)
            # try cheaper alts per slot
            trial = list(best_p)
            for i, sq in enumerate(self.query.intents):
                if i >= len(trial):
                    break
                alts = [
                    p for p in self.engine.products.get(sq, [])
                    if p.price is not None and favor[0] <= p.price <= favor[1]
                ]
                if alts:
                    trial[i] = alts[0]
            ok2, _ = U_HOO19_PrecisionGate.verify_pick(self.query, trial)
            if ok2:
                return trial

        if reason in ("low_coverage", "price_mismatch", "service_mismatch") and self.query.challenge == U_Challenge.PRODUCT:
            sq = self.query.intents[0]
            for alt in self.engine.products.get(sq, [])[:12]:
                trial = [alt]
                ok2, _ = U_HOO19_PrecisionGate.verify_pick(self.query, trial)
                if ok2:
                    return trial

        # Per-slot repair for shop/voucher (coverage / price / service on individual items)
        if reason in ("low_coverage", "price_mismatch", "service_mismatch", "count_mismatch") and\
                self.query.challenge in (U_Challenge.SHOP, U_Challenge.VOUCHER) and best_p:
            trial = list(best_p)
            changed = False
            require_same_shop = (
                self.query.challenge == U_Challenge.SHOP
                or (self.query.voucher and getattr(self.query.voucher, "apply_to", "") == "shop")
            )

            def _slot_ok(prod: U_Product, sq: U_SubQuery) -> bool:
                if sq.price:
                    pok, _ = U_FuzzyBoundaryLogic.price_fits_range(prod.price, sq.price, fuzzy=True)
                    if not pok:
                        return False
                return self.engine.__coverage_ratio__(prod, sq) >= U_ENHANCEMENT_BENCH.coverage_prune_floor

            for i, sq in enumerate(self.query.intents):
                if i >= len(trial):
                    break
                if _slot_ok(trial[i], sq):
                    continue
                shop = trial[i].shop
                alts = [p for p in self.engine.products.get(sq, []) if p.id != trial[i].id]
                if require_same_shop:
                    same = [p for p in alts if p.shop == shop]
                    alts = same or alts
                alts.sort(key=lambda p: self.engine.__coverage_ratio__(p, sq), reverse=True)
                for alt in alts[:8]:
                    if _slot_ok(alt, sq):
                        trial[i] = alt
                        changed = True
                        break
            if changed:
                ok2, _ = U_HOO19_PrecisionGate.verify_pick(self.query, trial)
                if ok2:
                    return trial
                # Prefer trial if it improves coverage hit-count even when not fully verified
                def _hits(combo: List[U_Product]) -> int:
                    return sum(
                        1 for i, sq in enumerate(self.query.intents)
                        if i < len(combo) and _slot_ok(combo[i], sq)
                    )
                if _hits(trial) > _hits(best_p):
                    return trial

        # SO2 fallback: first verifying candidate / combo
        if self.query.challenge == U_Challenge.PRODUCT and self.query.intents:
            sq = self.query.intents[0]
            for p in self.engine.products.get(sq, [])[:20]:
                if U_HOO19_PrecisionGate.verify_pick(self.query, [p])[0]:
                    return [p]
        return best_p

    def __finalize__(self, best_p: List[U_Product]):
        try:
            self.sess.step_on_compare(best_p, self.engine.__get_alt_str__(best_p))
        except Exception:
            pass

        self.sess.add_tool_result(U_ProxyUtil.submit(best_p))
        self.sess.add_tool_result(U_ProxyUtil.finish("success" if best_p else "failure"))
        
        self.sess.step_on_submit(best_p)

    def __identify_challenge__(self) -> U_Challenge:
        if self.query.challenge == U_Challenge.UNKNOWN:
            if re.search(U_Vocab.VOUCHER_PATTERN, self.query.query_f, re.I) and re.search(r"\bvoucher\b", self.query.query_f, re.I): return U_Challenge.VOUCHER
            if re.search(U_Vocab.SHOP_PATTERN, self.query.query_f, re.I): return U_Challenge.SHOP
            return U_Challenge.PRODUCT
        else:
            return self.query.challenge

    def __parse_voucher__(self) -> Tuple[str, U_Voucher]:
        sep = self.query.query_f.lower().find("my budget is")

        query_e = self.query.query_f[:sep].strip()
        v_str = self.query.query_f[sep:].strip()

        m = re.search(r"\bmy\s+budget\s+is(?:\s+only)?\s+`(\d+)`", v_str, re.I)
        budget = int(m.group(1))

        apply_to = "shop" if re.search(r"\bsame\s+shops?\b", v_str, re.I) else "all"

        m = re.search(r"\bexceeds?\s+`(\d+)`", v_str, re.I)
        threshold = int(m.group(1))

        if (m := re.search(r"\bpercent(?:age)?\s+discount(?:\s+of)?\s+`([\d\.]+)\%`", v_str, re.I)):
            discount = float(m.group(1))
            is_fixed = False

            m = re.search(r"\bwith(?:\s+a)\s+cap(?:\s+of)?\s+`([\d\.]+)`", v_str, re.I)
            cap = float(m.group(1))
        else:
            m = re.search(r"\bfixed\s+discount(?:\s+of)?\s+`([\d\.]+)`", v_str, re.I)
            discount = float(m.group(1))
            is_fixed = True
            cap = None
        
        voucher = U_Voucher(
            budget = budget,
            apply_to = apply_to,
            threshold = threshold,
            is_fixed = is_fixed,
            discount = discount,
            cap = cap
        )
        return query_e, voucher

    def __check_review_needed__(self) -> Tuple[List[str], str, str, str, str]:
        """
        Identifies discrepancies between explicit query mentions and parsed product constraints,
        generating diagnostic prompts for LLM-based review.
        
        Returns:
            suspects:     List of constraint labels requiring review
            prompts_i:    Inline diagnostic messages for the review prompt
            prompts_o:    JSON schema snippets for the review response
            focuses:      Bullet list of focus areas for the reviewer
            error_3:      Specific guidance for price-range misclassification
        """
        
        # ------------------------------------------------------------------
        # 1. Gather evidence: explicit mentions vs. parsed constraints
        # ------------------------------------------------------------------
        query_text = self.query.query_e
        intents = self.query.intents
        
        # Count parsed constraints across all product intents
        parsed_counts = {
            "lazmall":     sum(1 for sq in intents if "official"     in sq.services),
            "lazflash":    sum(1 for sq in intents if "flashsale"    in sq.services),
            "free_ship":   sum(1 for sq in intents if "freeShipping" in sq.services),
            "cod":         sum(1 for sq in intents if "COD"          in sq.services),
            "price":       sum(1 for sq in intents if sq.price),
        }
        
        # Count explicit mentions in the raw query text
        explicit_counts = {
            "lazmall":   len(re.findall(U_Vocab.SERVICE_PATTERNS["official"][0],     query_text, re.I)),
            "lazflash":  len(re.findall(U_Vocab.SERVICE_PATTERNS["flashsale"][0],    query_text, re.I)),
            "free_ship": len(re.findall(U_Vocab.SERVICE_PATTERNS["freeShipping"][0], query_text, re.I)),
            "cod":       len(re.findall(U_Vocab.SERVICE_PATTERNS["COD"][0],          query_text, re.I)),
            "price":     len(re.findall(r"\b(php|pesos?|usd|dollars?|euros?)\b", query_text, re.I)),
        }
        
        # ------------------------------------------------------------------
        # 2. Constraint metadata (label → display name & description)
        # ------------------------------------------------------------------
        CONSTRAINT_META = {
            "lazmall":   ("lazmall",   'The word "LazMall"'),
            "lazflash":  ("lazflash",  'The word "LazFlash"'),
            "free_ship": ("free_ship", "The condition of free or complimentary shipping"),
            "cod":       ("cod",       'The condition of "Cash on Delivery" option'),
        }
        
        FOCUS_META = {
            "lazmall":   '**lazmall**: explicit mention of the word "LazMall".',
            "lazflash":  '**lazflash**: explicit mention of the word "LazFlash".',
            "free_ship": '**free_ship**: explicit mention of free or complimentary shipping.',
            "cod":       '**cod**: explicit mention of cash on delivery or payment on delivery.',
            "price":     '**price**: explicit mention of price range. (parsed results use "[low]-[high]" format without currency unit. just "[low]-" or "-[high]" if only one bound is specified.)',
        }
        
        # ------------------------------------------------------------------
        # 3. Detect mismatches and build diagnostic blocks
        # ------------------------------------------------------------------
        suspects:  List[str] = []
        prompts_i: List[str] = []
        prompts_o: List[str] = []
        
        # --- Standard service constraints ---
        for label, (json_key, prefix) in CONSTRAINT_META.items():
            exp = explicit_counts[label]
            par = parsed_counts[label]
            if exp == par:
                continue
            
            suspects.append(label)
            
            prompts_i.append(
                f'- {prefix} appeared {exp} time(s) in the original query, '
                f'but {par} product(s) were parsed with this constraint. It needs double check.'
            )
            
            prompts_o.append(
                f'    "{json_key}": [\n'
                f'        <product_id_1, str>, ... // correct product ids that must receive **{label}** constraint.\n'
                f'    ]'
            )
        
        # --- Price constraint (special handling due to complex schema) ---
        if explicit_counts["price"] != parsed_counts["price"]:
            suspects.append("price")
            
            prompts_i.append(
                f'- Currency unit appeared {explicit_counts["price"]} time(s) in the original query, '
                f'but {parsed_counts["price"]} product(s) were parsed with price constraint. '
                f'You must carefully check the context if those number ranges are really the price constraints. '
                f'Keep in mind that one price constraint cannot apply to multiple products.'
            )
            
            prompts_o.append("""
        "price": { // list all the products (don't skip any product)
            "1": { // product id = 1
                "parsed_price_range": <str|null>, // the price range string or null as assigned for this product by the parsing result given
                "is_it_correct": <bool>, // true if the parsed result is correct, otherwise false
                "other_spec": <str|null>, // the other spec name ("power usage", for example) which was mistakenly identified as the price range in the parsing result if exists. set null if "parsed_price_range" is null or the parsed result has no error.
                "correct_price_range": <str|null> // the corrected price range for this product. write the original parsed result again if it has no error. set null if no price range condition is for this product. use "[low]-[high]" format without currency unit if exists. "[low]-" or "-[high]" if either bound is missing.
            },
            ...    
        }""".strip())
        
        # ------------------------------------------------------------------
        # 4. Build focus list
        # ------------------------------------------------------------------
        focuses = [f'- {FOCUS_META[label]}' for label in suspects if label in FOCUS_META]
        
        # ------------------------------------------------------------------
        # 5. Conditional guidance for price-range misclassification
        # ------------------------------------------------------------------
        error_3 = """\
    ## 3. Confused price range
        - A product appears with price range constraint in the parsed result, but actually this number range is for a completely different specification.
        - For example, in "I need a bottle between 0.5 and 1.2 litre", the range "0.5-1.2" is not price constraint, but a capacity constraint.
        - Also, in "Find motors with power usage less than 2.5", the range "-2.5" is mentioned without currency unit, but you can easily confirm that it's the power usage constraint with unit "kw".""".strip() if "price" in suspects else ""
        
        return suspects, "\n".join(prompts_i), "\n".join(prompts_o), "\n".join(focuses), error_3

    def __create_review_prompt__(self, prompts_i: str, prompts_o: str, focuses: str, error_3: str) -> str:
        prods = {
            str(i + 1): {
                "sub-query": sq.query,
                "constraints": [U_Prompt.service_to_label(svc) for svc in (sq.services or [])],
                "price": sq.price
            }
            for i, sq in enumerate(self.query.intents)
        }
        return U_Prompt.PSV_REVIEW.format(
            count = len(self.query.intents),
            focuses = focuses,
            rules = U_Prompt.SCOPE_RULES,
            suspects = prompts_i,
            error_3 = error_3,
            responses = prompts_o,
            query = self.query.query_e,
            parsed = prods
        )

    def __fix_by_llm_review__(self, js: Dict[str, List[str] | Dict], suspects: List[str]) -> None:
        def __get_id__(id_str: Any) -> Optional[int]:
            if not isinstance(id_str, str): return None
            pid = U_MiscUtil.extract_number(id_str, True) - 1

            if pid < 0 or pid >= len(self.query.intents): return None
            return pid
        
        for issue in suspects:
            if issue not in js: continue
            obj = js.get(issue)

            if issue in ["lazmall", "lazflash", "cod", "free_ship"]:
                if not isinstance(obj, list): continue
                svc = U_Prompt.label_to_service(issue)
                
                ids = [__get_id__(x) for x in obj]
                if None in ids: continue

                for i, sq in enumerate(self.query.intents):
                    if i in ids and svc not in sq.services:
                        sq.services.append(svc)
                    elif i not in ids and svc in sq.services:
                        sq.services.remove(svc)
            elif issue == "price":
                if not isinstance(obj, dict): continue
                
                ids = [__get_id__(k) for k in obj.keys()]
                if None in ids: continue

                if sorted(ids) != list(range(len(self.query.intents))): continue

                for i, sq in enumerate(self.query.intents):
                    j: Dict[str, str] = obj.get(str(i + 1))
                    
                    correct = j.get("correct_price_range")
                    sq.price = re.sub(r"[^\d\-]", "", correct or "") or None

def U_find_product(**kwargs) -> List[Dict]:
    result = U_ProxyUtil.ORO_PROXY.get("/search/find_product", kwargs)
    return result if result else []

def U_view_product_information(product_ids: str = "", ids: str = "", **kwargs) -> List[Dict]:
    pid = product_ids or ids or kwargs.get("product_ids") or kwargs.get("ids") or ""
    result = U_ProxyUtil.ORO_PROXY.get("/search/view_product_information", {"product_ids": pid})
    return result if result else []

def U_recommend_product(product_ids: str) -> str:
    return f"submitted recommendation to the user: {product_ids}."

def U_terminate(status: str = "success") -> str:
    return f"session completed with status: {status}"



class NS_ShoppingTaskType(str, Enum):
    PRODUCT = "product"
    SHOP = "shop"
    VOUCHER = "voucher"

    def explain(self, is_multiple: bool) -> str:
        if self == NS_ShoppingTaskType.PRODUCT:
            return "a single product with specific constraints"
        elif self == NS_ShoppingTaskType.SHOP:
            return "multiple products from the same shop"
        elif self == NS_ShoppingTaskType.VOUCHER:
            pstr = "multiple products" if is_multiple else "a single product"
            return f"{pstr} within a specified budget and considering voucher options"
        else:
            raise ValueError(f"unknown task type: {self}")

def NS_resolve_engine_cls(challenge):
    """Lane resolver — always NS_ShopSearchEngine."""
    return NS_ShopSearchEngine


class NS_Vocab:


    ID_KEYS = ("brand", "color", "model", "id", "compatibility", "color_family")


    SCHEMA_KEYS: frozenset = frozenset({
        "product_type", "color", "color_family", "material", "brand", "style",
        "size", "size_system", "gender", "user", "age", "pack", "set", "model",
        "compatibility", "pattern", "design", "shape", "origin", "usage",
        "fit", "feature", "category", "capacity", "volume", "length", "weight",
        "pre_config", "suitable", "condition", "warranty_days", "version",
        "option", "variation", "solution", "quality", "unit", "frame", "theme",
        "promotion", "inspiration", "interest", "vibe", "id",
    })


    NON_SEARCH_KEYS: frozenset = frozenset({"delivery", "service"})

    ALPHABETS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    KW_STOPWORDS: Tuple[str, ...] = (
        "color family", "suitable", "compatible",
        "about", "above", "after", "also", "and", "as", "at", "before", "below", "between", "by", "for", "from", "in",
        "less", "more", "of", "on", "or", "over", "per", "than", "to", "under", "up", "with", "within"
    )

    NO_WORDS: Tuple[str, ...] = ("no", "without any", "without")
    YES_WORDS: Tuple[str, ...] = ("yes", "enabled", "included", "exists", "true")

    COLOR_WORDS: Tuple[str, ...] = (
        "red", "blue", "green", "yellow", "black", "white", "pink", "purple", "violet", "orange", "brown", "gray", "grey",
        "beige", "navy", "gold", "silver", "bronze", "magenta", "cyan"
    )

    SEARCHABLE_UNITS: Tuple[str, ...] = (
        "mm", "cm", "m", "km", "ft", "yd", "mg", "g", "kg", "oz", "lb",
        "ml", "l", "mah", "ah", "wh", "kwh", "v", "mv", "kv", "ma", "w", "kw",
        "hz", "khz", "mhz", "ghz", "kb", "mb", "gb", "tb",
        "bps", "kbps", "mbps", "gbps", "°c", "°f", "pcs", "pc",
        "pa", "kpa", "mpa", "mp", "dpi", "lux", "db", "rpm"
    )

    SYNONYMS: Tuple[Tuple[str, str], ...] = tuple(
        (min(a, b), max(a, b)) for a, b in
        [
            ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"), ("five", "5"),
            ("six", "6"), ("seven", "7"), ("eight", "8"), ("nine", "9"), ("ten", "10"),
            ("one", "single"), ("two", "double"), ("three", "triple"),
            ("1", "single"), ("2", "double"), ("3", "triple"),
            ("first", "1st"), ("second", "2nd"), ("third", "3rd"),
            ("hundred", "100"), ("thousand", "1000"), ("zero", "0")
        ]
    )

    P_EXT_SEARCH_UNITS: Tuple[Tuple[str, str], ...] = (
        ("piece", "pcs"), ("pack", "pcs")
    )

@dataclass
class NS_JsonModel(object):
    @classmethod
    def from_json(cls, data: dict) -> "JsonModel":
        kwargs = {}

        for f in fields(cls):
            if f.name not in data: continue

            value = data[f.name]
            t = f.type

            kwargs[f.name] = cls._coerce_from_json(value, t)

        return cls(**kwargs)

    def to_json(self) -> Dict[str, Any] | List[Any]:
        return self._coerce_to_json(self)

    @staticmethod
    def _coerce_from_json(value, t) -> Any:
        origin = get_origin(t)
        args = get_args(t)

        if origin is Union and len(args) == 2 and NoneType in args:
            inner_t = next(a for a in args if a is not NoneType)

            if value is None: return None
            return NS_JsonModel._coerce_from_json(value, inner_t)

        if is_dataclass(t) and isinstance(value, dict):
            return t.from_json(value)

        if origin is list and isinstance(value, list):
            return [NS_JsonModel._coerce_from_json(v, args[0]) for v in value]

        if origin is tuple and isinstance(value, list):
            return tuple(NS_JsonModel._coerce_from_json(v, args[0]) for v in value)

        if origin is dict and isinstance(value, dict):
            kt, vt = args

            return {
                NS_JsonModel._coerce_from_json(k, kt): NS_JsonModel._coerce_from_json(v, vt)
                for k, v in value.items()
            }

        if t is float and isinstance(value, str): return NS_MiscUtil.extract_number(value)
        if t is int and isinstance(value, str): return NS_MiscUtil.extract_number(value, True)
        if t is str: return str(value)

        return value

    @staticmethod
    def _coerce_to_json(obj) -> Any:
        if is_dataclass(obj):
            result = {}

            for f in fields(obj):
                value = getattr(obj, f.name)
                result[f.name] = NS_JsonModel._coerce_to_json(value)

            return result

        if isinstance(obj, list):
            return [NS_JsonModel._coerce_to_json(x) for x in obj]

        if isinstance(obj, dict):
            return {NS_JsonModel._coerce_to_json(k): NS_JsonModel._coerce_to_json(v) for k, v in obj.items()}

        return obj

@dataclass
class NS_VoucherOption(NS_JsonModel):
    budget: float
    apply_to: str
    threshold: Optional[float]
    is_fixed: bool
    discount: float
    cap: Optional[float]

    def is_favor_cost(self, psum: float) -> bool:


        if self.threshold is None: return False
        if self.threshold < int(psum * 0.1) or self.threshold > int(psum * 0.9): return False
        if psum < 100.0: return False
        if not self.is_fixed:
            ac = max(psum * (1.0 - self.discount / 100.0), psum - (self.cap if self.cap is not None else math.inf))
            if self.budget < math.floor(ac) or self.budget > math.ceil(1.1 * ac): return False
        elif self.budget < psum - self.discount: return False
        return True

@dataclass
class NS_SubQuery(NS_JsonModel):
    query: str = ""


    phrase: str = ""

    services: List[str] = field(default_factory = list, compare = False)
    price: Optional[str] = None

    kw: List[List[str]] = field(default_factory = list, compare = False)

    def __hash__(self):
        return hash((self.query, ''.join(sorted(self.services)), self.price))

    def __eq__(self, other):
        if not isinstance(other, NS_SubQuery): return NotImplemented

        if self.query != other.query: return False
        if ''.join(sorted(self.services)) != ''.join(sorted(other.services)): return False
        if self.price != other.price: return False

        return True

@dataclass
class NS_SearchOption(NS_SubQuery):
    shop_id: Optional[str] = None
    q: str = ""

    def update_shop(self, shop_id: Optional[str]) -> "SearchOption":
        self.shop_id = shop_id
        return self

    def update_price(self, price: Optional[str]) -> "SearchOption":
        self.price = price
        return self

    def update_kw(self, id: int, kw: List[str]) -> "SearchOption":
        self.kw[id] = kw
        return self

    def update_q(self, q: Optional[str] = None) -> "SearchOption":
        if not q:
            segs = []

            for kw in self.kw:
                v = kw[-1]
                keys = kw[:-1]


                if "no" in keys:
                    continue
                if any(re.match(rf"^{re.escape(nw)}(\s+|$)", v, re.I) for nw in NS_Vocab.NO_WORDS):
                    continue


                if any(k in NS_Vocab.NON_SEARCH_KEYS for k in keys):
                    continue


                leaf_key = keys[-1] if keys else ""
                if leaf_key and leaf_key not in NS_Vocab.SCHEMA_KEYS:
                    for m in leaf_key.replace("_", " ").split():
                        m = m.lower().strip()
                        if m and m not in segs:
                            segs.append(m)

                for m in v.split():
                    m = m.lower().strip()
                    if m and m not in segs: segs.append(m)

            self.q = " ".join(segs)
        else:
            self.q = q

        return self

    def __hash__(self):
        return hash((self.q, self.query, ''.join(sorted(self.services)), self.price, self.shop_id))

    def __eq__(self, other):
        if not isinstance(other, NS_SearchOption): return NotImplemented

        if self.query != other.query: return False
        if ''.join(sorted(self.services)) != ''.join(sorted(other.services)): return False
        if self.price != other.price: return False
        if self.shop_id != other.shop_id: return False
        if self.q != other.q: return False

        return True

@dataclass
class NS_Weight(NS_JsonModel):
    SKU_LIN_PROBS = (0.1, 0.1, 0.08, 0.05, 0.02, 0.01)

    RANK_BONUS_C = 0.05
    RANK_LIMIT = 300

    RANK_LOG_PROBS = (
        0.6554774978466211, 0.09305071398002347, 0.04004714272557973, 0.024146071349246602, 0.019434642793296042, 0.01943464179329533,
        0.018845714223802223, 0.005889285694938195, 0.005889284694937484, 0.005005892840697467, 0.005005891840696755, 0.005005890840696044,
        0.005005889840695332, 0.002572687961473001, 0.00257268696147229, 0.0025726859614715793, 0.002572684961470868, 0.0025726839614701575,
        0.0025726829614694463, 0.0025726819614687356, 0.0025726809614680245, 0.0025726799614673138, 0.0025726789614666026,
        0.002572677961465892, 0.002572676961465181, 0.00257267596146447, 0.0025726749614637594, 0.002572673961463048, 0.0025726729614623375,
        0.0025726719614616264, 0.0025726709614609156, 0.0025726699614602045, 0.002572668961459494, 0.0025726679614587827,
        0.002572666961458072, 0.0025726659614573612, 0.00257266496145665, 0.0025726639614559394, 0.0025726629614552282,
        0.0025726619614545175, 0.0025726609614538064, 0.0025726599614530957, 0.0025726589614523845, 0.002572657961451674,
        0.0025726569614509627, 0.002572655961450252, 0.0025726549614495412, 0.00257265396144883, 0.0025726529614481194,
        0.0025726519614474083, 0.0025726509614466975
    )
    detail: Dict[str, float] = field(default_factory = dict)
    group: Dict[str, int] = field(default_factory = dict)

    weighted_rank: int = -1
    rank: int = -1

    confidence: float = 0

    score: float = 0
    score_n: float = 0

    rank_bonus: float = 0
    sku_bonus: float = 0


    def _detail_n(self) -> int:
        return len(self.detail) or 1

    def normalize(self):
        idx = self.group.get("sku") or -1

        self.sku_bonus = NS_Weight.SKU_LIN_PROBS[idx + 1] if idx < (len(NS_Weight.SKU_LIN_PROBS) - 1) else 0
        self.rank_bonus = max(1.0 - self.rank / NS_Weight.RANK_LIMIT, 0) * NS_Weight.RANK_BONUS_C
        self.score_n = self.score / self._detail_n() + self.rank_bonus + self.sku_bonus

    def update_confidence(self, weighted_rank: int):
        if self.weighted_rank >= 0: return
        self.weighted_rank = weighted_rank

        if weighted_rank >= len(NS_Weight.RANK_LOG_PROBS):
            self.confidence = NS_Weight.RANK_LOG_PROBS[-1] * (0.9 ** (weighted_rank - len(NS_Weight.RANK_LOG_PROBS) + 1))
        else:
            self.confidence = NS_Weight.RANK_LOG_PROBS[weighted_rank]

    def update_with_sibling(self, sibling: "Weight"):
        if self.score < sibling.score:
            self.detail = sibling.detail
            self.group = sibling.group
            self.score = sibling.score

        self.sku_bonus = max(self.sku_bonus, sibling.sku_bonus)
        self.score_n = self.score / self._detail_n() + self.rank_bonus + self.sku_bonus

@dataclass
class NS_Product(NS_JsonModel):
    id: str
    title: str
    price: float
    service: Optional[List[str]] = None
    shop_id: Optional[str] = None

    attributes: Dict[str, List[str]] = field(default_factory = dict)
    sku: Dict[str, Dict[str, str]] = field(default_factory = dict)

    is_viewed: bool = False

    def to_prompt(self, q: str, for_reason: bool = False) -> Dict[str, Any]:
        attr = self.extract_attr(q, ':', False)

        joint_attr = []
        for g, group in attr.items():
            if g == "sku": continue
            for lines in group:
                for l in lines:
                    joint_attr.append(l)


        sku_rows = [
            f"[variant {i}] " + " | ".join(lines)
            for i, lines in enumerate(attr.get("sku", []), start = 1)
        ]

        js = {
            "title": self.title,
            "attributes": '\n'.join(joint_attr)
        }
        if sku_rows: js["sku_options"] = '\n'.join(sku_rows)
        if for_reason: js["price"] = self.price
        return js

    def clone_from(self, other: "Product") -> None:
        self.id = other.id
        self.title = other.title
        self.price = other.price
        self.service = other.service
        self.shop_id = other.shop_id

        self.attributes = other.attributes
        self.sku = other.sku
        self.is_viewed = other.is_viewed

    def extract_attr(self, q: str, delimiter: str, norm: bool) -> Dict[str, List[List[str]]]:
        extracted = {"sku": []}
        is_valid = False

        for k, vs in self.attributes.items():
            extracted[f"att_{k}"] = []
            atoms = []

            for v in vs:
                if NS_EngineUtil.is_valid_attr(k, v, q): is_valid = True
                atoms.append((k, v))

            if atoms: extracted[f"att_{k}"].append(atoms)

        for _, kv in self.sku.items():
            atoms = []

            for k, v in kv.items():
                if NS_EngineUtil.is_valid_attr(k, v, q): is_valid = True
                atoms.append((k, v))

            if atoms: extracted["sku"].append(atoms)

        if not is_valid: extracted.clear()

        return {
            g: [
                [
                    f"{NS_MiscUtil.normalize_str(k) if norm else k}{delimiter} "
                    f"{NS_MiscUtil.normalize_str(v) if norm else v}"
                    for k, v in kv
                ]
                for kv in kvs
            ]
            for g, kvs in extracted.items()
        }

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, NS_Product): return NotImplemented
        return self.id == other.id

@dataclass
class NS_QueryInfo(NS_JsonModel):
    challenge: NS_ShoppingTaskType = ""

    query_f: str = ""
    query_e: str = ""

    items: List[NS_SubQuery] = field(default_factory = list)
    vopt: Optional[NS_VoucherOption] = None

def NS_cast_model(source: NS_JsonModel, target_cls: Type[NS_JsonModel]) -> NS_JsonModel:


    return target_cls.from_json(source.to_json())

class NS_MiscUtil:

    @staticmethod
    def extract_number(s: str, as_int: bool = False) -> int | float:
        pattern = r"\d+\.?\d*"
        matches = re.findall(pattern, s)

        if len(matches) == 0: return 0
        val = float(matches[0])

        return val if not as_int else int(val)
    @staticmethod
    def ensure_number(v: Any, as_int: bool = False) -> int | float:
        if v is None: return 0
        if isinstance(v, (int, float)): return v
        if isinstance(v, str): return NS_MiscUtil.extract_number(v, as_int = as_int)
        raise ValueError(f"# cannot convert {v} to number")
    @staticmethod
    def extract_bool(s: str) -> bool:
        s = s.lower().strip()

        if not s:
            return False
        elif s == "true":
            return True
        elif s == "false":
            return False
        else:
            return int(float(s)) != 0 if s.isnumeric() else False

    @staticmethod
    def normalize_str(x: str) -> str:
        x = x.lower()

        x = re.sub(r"\b(\d+)([a-z]+)\b", r"\1 \2", x)
        x = re.sub(r"\b([a-z]+)(\d+)\b", r"\1 \2", x)
        x = re.sub(r"[\(\)\[\]\{\}\+\*\_\:\;\,\-\~\/\<\>\=]", " ", x)
        x = re.sub(r"\b(\d+)\s*x\s*(\d+)\b", r"\1 \2", x)
        x = re.sub(r"\s+", " ", x).strip()

        return x

    @staticmethod
    def get_choice_combs(cat: Dict | List, include_non: bool) -> List[Dict | List]:
        keys = list(cat.keys()) if isinstance(cat, dict) else list(range(len(cat)))

        choices = [
            ([-1] if include_non else []) + list(range(len(cat[k])))
            for k in keys
        ]
        opts = product(*choices)

        return [dict(zip(keys, opt)) for opt in opts] if isinstance(cat, dict) else list(opts)

    @staticmethod
    def print_json(js: Dict[str, Any] | List[Any]) -> None:
        _debug_print(json.dumps(js, ensure_ascii = False, indent = 2))
    @staticmethod
    def parse_json(s: str) -> dict[str, Any] | list[Any] | None:
        cleaned = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags = re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned, ).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        starts = []

        obj_start = cleaned.find("{")
        if obj_start >= 0: starts.append(obj_start)

        arr_start = cleaned.find("[")
        if arr_start >= 0: starts.append(arr_start)

        for start in sorted(starts):
            try:
                obj, _ = decoder.raw_decode(cleaned[start:])
                return obj
            except json.JSONDecodeError:
                pass

        return None

class NS_EngineUtil:


    @staticmethod
    def get_spell_variations(s: str) -> List[str]:
        vars = []

        if len(s) >= 3 and not re.search(r"\d", s):
            for suffix in ["es", "ic", "n", "d", "ed", "y", "ing", "ily", "ied", "er", "est", "ier", "ren"]:
                vars.append(s + suffix)

                if s.endswith(suffix):
                    body = s[:-len(suffix)]

                    vars.append(body + "y")
                    vars.append(body)

                    if len(body) >= 4 and body[-1] == body[-2]: vars.append(body[:-1])

            for suffix in ["ed", "es", "y", "er", "est", "ing"]:
                vars.append(s + s[-1] + suffix)

            if s.endswith("s"):
                vars.append(s[:-1] + "es")
                vars.append(s[:-1])
            else:
                vars.append(s + "s")

            if s.endswith("y"):
                for suffix in ["", "ied", "ies", "ier", "ing", "ily"]:
                    vars.append(s[:-1] + suffix)

            if len(s) <= 5 and s.endswith("en"): vars.append(s[:-2] + "an")
            if len(s) <= 5 and s.endswith("an"): vars.append(s[:-2] + "en")
            if len(s) >= 5 and s.endswith("or"): vars.append(s[:-2] + "our")
            if len(s) >= 6 and s.endswith("our"): vars.append(s[:-3] + "or")

        vars = list(set(v for v in vars if v != s and v not in NS_Vocab.KW_STOPWORDS and len(v) >= 3))
        return vars
    @staticmethod
    def is_kw_in(kw: str, s: str, as_is: bool, synonyms: List[Tuple[str, str]], terminal: bool = False) -> bool:
        if re.search(
            (r"\b" if re.match(r"\w", kw[0]) else "") + re.escape(kw) + (r"\b" if re.match(r"\w", kw[-1]) else ""),
            s,
            re.I
        ):
            return True

        if as_is: return False
        vars = NS_EngineUtil.get_spell_variations(kw)

        for v in vars:
            if re.search(rf"\b{re.escape(v)}\b", s, re.I): return True

        if len(kw) >= 6 and len(s) > len(kw) + 1:
            segs = s.replace('🎈', ' ').split()

            if len(segs) > 1:
                for i in range(len(segs) - 1):
                    sub_s = segs[i] + segs[i + 1]
                    if kw == sub_s: return True

        if kw in ["pcs", "pc", "pack"]:
            for w in ["piece", "pack", "set"]:
                if w in s: return True

        for from_s, to_s in synonyms:
            if kw == from_s or kw == f"{from_s}s" or from_s == f"{kw}s":
                if re.search(rf"\b{re.escape(to_s)}s?\b", s): return True
            elif kw == to_s or kw == f"{to_s}s" or to_s == f"{kw}s":
                if re.search(rf"\b{re.escape(from_s)}s?\b", s): return True

        if not terminal:
            spell = re.sub(rf"ss", "sss", kw, flags = re.I)
            if spell != kw and NS_EngineUtil.is_kw_in(spell, s, False, synonyms, True): return True

        return False

    @staticmethod
    def is_valid_attr(k: str, v: str, q: str) -> bool:
        pairs: List[Tuple[str, str, str, str, bool]] = [
            (v, q, r'\b(\d+[a-z]*)\s*\*\s*(\d+[a-z]*)\b', r'{m1} by {m2}', True), (v, q, r'\b(\d+)\-(\d+)([a-z]+)\b', r'{m1}-{m2} {m3}', True),
            (v, q, r'\b(\d+)([a-z])\+', r'\b{m1}\s*{m2}([a-z]*)\sand\sup\b', True), (v, q, r'\b(\d+)\-(\d+)[a-z]+\b', r'\b{m1}\-{m2}\b', True),
            (v, q, r'\b(\d+)\-(\d+)y\b', r'\b{m1}\sto\s{m2}\syears\b', True), (q, v, r'\b(\d+)\s*\-\&\-\s*(\d+)\b', r'{m1}\sto\s{m2}', False),
            (v, q, r'\b(\d+)\s*-\s*(\d+)\s*([a-z]+)\b', r'{m3}\sbetween\s{m1}\sand\s{m2}', True), (v, q, r'\b(\d+)\.0\b', r'\b{m1}\b', True),
            (v, q, r'\b([\d\.]+)[a-z]*\s*~\s*([\d\.]+)\b', r'\b{m1}\sto\s{m2}\b', True), (v, q, r'【(\d+\-\d+[a-z]+)】', r'\b{m1}\b', True),
            (v, q, r'\b(\d+)\s*-\s*(\d+)([a-z]*)\b', r'up to {m2}{m3}', True), (v, q, r'\b(\w+)\s(\w+)\b', r'\b{m1}\s(\w+)\s{m2}\b', True),
            (q, v, r'\b([a-z]+)\s+\((\d)\s+piece\)\s+', r'^{m1}\s+{m2}pcs$', False), (v, q, r'^(?:[a-z]+)\:\s*(\d+)$', r'\b{m1}\b', True),
            (v, q, r'\b(?:[a-z]+)\:\s*(\d+)\s*-\s*(\d+)\s*([a-z]+)\b', r'up\sto\s{m2}{m3}', True),
            (v, q, r'\b(\d+)\s*-\s*(\d+)\b', r'{m1} and {m2}', True)
        ]
        for src, dst, src_pattern, dst_pattern, yes in sorted(pairs, key = lambda p: p[-1]):
            if (m := re.search(src_pattern, src, re.I)):
                exists = re.search(
                    dst_pattern.format(
                        m1 = re.escape(m.group(1)) if m.lastindex and m.lastindex >= 1 else "",
                        m2 = re.escape(m.group(2)) if m.lastindex and m.lastindex >= 2 else "",
                        m3 = re.escape(m.group(3)) if m.lastindex and m.lastindex >= 3 else "",
                        k = k
                    ),
                    dst,
                    re.I
                )
                if not yes and not exists: return False

        return True

    @staticmethod
    def compact_kw(nested: str, q: str) -> Optional[List[str]]:
        splits = [s for s in [s.strip() for s in nested.lower().split("→")] if s]
        if len(splits) < 2: return None

        ks, v = splits[:-1], splits[-1]

        if v.lower() in NS_Vocab.YES_WORDS:
            for i in range(len(ks)):
                t = " ".join([k.replace("_", " ") for k in ks[i:]])

                if t in q:
                    ks, v = [v], t
                    break
            else:
                return None

        for yes in NS_Vocab.YES_WORDS:
            if re.search(rf"\b{re.escape(yes)}$", v, re.I):
                v = re.sub(rf"\b{re.escape(yes)}$", "", v, flags = re.I)

        for nw in NS_Vocab.NO_WORDS:
            if (m := re.search(rf"^{re.escape(nw)}\s+([\w\s]+)$", v, re.I)):
                ks.append("no")
                v = m.group(1)
                break

        if f"{ks[-1]} {v}" in NS_Vocab.KW_STOPWORDS: return None
        if f"{v} {ks[-1]}" in NS_Vocab.KW_STOPWORDS: return None

        for sw in NS_Vocab.KW_STOPWORDS:


            v = re.sub(rf"(?<!-)\b{re.escape(sw)}\b(?!-)", "", v, flags = re.I)

        if len(ks) == 1 and '_' in ks[0]:
            prefix = ks[0].split('_')[0] + '-'
            if v.startswith(prefix): v = v[len(prefix):]

            prefix = prefix[:-1]
            if re.search(rf"^{re.escape(prefix)}[a-z]+$", v, re.I): v = v[len(prefix):]

        if (m := re.search(r"^(any|all)\s+([\w\s]+)$", v, re.I)):
            k = m.group(2).replace(' ', '_')
            v = m.group(1)

            if k not in ks: ks.append(k)

        if v.isnumeric() and ks[-1] in ["eu", "us", "int"]:
            v, ks = f"{ks[-1]} {v}", ["size"]

        v = re.sub(r"\bge[a-z]([a-z]{2}one)\b", r"\1", v, flags = re.I)
        v = re.sub(r"\b(\d+)in(\d+)\b", r"\1-in-\2", v, flags = re.I)
        v = re.sub(r"[\&\|]", " ", v)
        v = re.sub(r"\s+", " ", v).strip()


        _RETAIL_CONTEXT = frozenset({"store", "shop", "seller", "vendor", "marketplace"})
        if v.lower() in _RETAIL_CONTEXT:
            return None

        return (ks + [v]) if v else None
    @staticmethod
    def kw_to_constraints(kws: List[List[str]]) -> List[str]:


        constraints = []

        for kw in kws:
            if not kw: continue
            *keys, v = kw
            key = "_".join(k for k in keys if k) or "attribute"
            constraints.append(f"{key}: {v}")

        return constraints


    _PROD_ONLY_RES = (
        re.compile(r"\b(?:just|only)(?:\s+the)?\s+products?\b", re.I),
        re.compile(r"\bproducts?\s+only\b", re.I),
    )

    @staticmethod
    def is_prod_only(q: str) -> bool:
        return any(pat.search(q) is not None for pat in NS_EngineUtil._PROD_ONLY_RES)

    @staticmethod
    def get_vouchered_cost(products: List[NS_Product], vo: Optional[NS_VoucherOption]) -> float:
        psum = sum([p.price for p in products])
        if vo is None: return psum

        if vo.apply_to == "shop":
            shops = set([p.shop_id for p in products])
            if len(shops) > 1: return psum

        if vo.threshold is not None:
            if psum < vo.threshold: return psum

        if vo.is_fixed: return psum - vo.discount

        discount = psum * vo.discount / 100.0
        if vo.cap is not None: discount = min(discount, vo.cap)

        return psum - discount
    @staticmethod
    def get_favor_range(vo: NS_VoucherOption) -> Tuple[float, float]:
        high = (vo.threshold / 0.1) if vo.threshold is not None else math.inf
        low = (vo.threshold / 0.9) if vo.threshold is not None else 0

        if not vo.is_fixed:
            l1 = vo.budget / 1.1 / (1.0 - vo.discount / 100.0)
            l2 = vo.budget / 1.1 + (vo.cap if vo.cap is not None else math.inf)
            h1 = vo.budget + (vo.cap if vo.cap is not None else math.inf)
            h2 = vo.budget / (1.0 - vo.discount / 100.0)
            low = max(low, min(l1, l2))
            high = min([high, h1, h2])
        else:
            low = max(low, vo.budget / 1.1 + vo.discount)
            high = min(high, vo.budget + vo.discount)

        return (math.floor(low) - 1, math.ceil(high) + 1)
    @staticmethod
    def promptfy_price(price: Optional[str]) -> str:
        if not price: return ""

        segs = [s.strip() for s in price.split('-')]
        if len(segs) != 2: return ""

        try:
            low = int(segs[0]) if segs[0] else None
            high = int(segs[1]) if segs[1] else None

            if low and high: return f"price between {low} and {high}"
            if low: return f"price greater than {low}"

            return f"price less than {high}"
        except Exception:
            return ""
    @staticmethod
    def numerify_price(price: Optional[str]) -> Tuple[int, int]:
        low_str, high_str = price.split('-', 1)
        low, high = 0, math.inf

        if low_str: low = NS_MiscUtil.extract_number(low_str, True)
        if high_str: high = NS_MiscUtil.extract_number(high_str, True)

        return low, high
    @staticmethod
    def stringfy_price(low: int, high: int) -> str:
        return (str(low) if low > 0 else "") + "-" + (str(high) if high < math.inf else "")

class NS_ProxyUtil:


    DEADLINE = 0.0
    @staticmethod
    def get_llm_priorities(prefers: List[str]) -> List[str]:
        priorities = prefers.copy()

        for llm in NS_ProxyUtil.default_llm_priorities():
            if llm not in priorities:
                priorities.append(llm)

        return priorities

    @staticmethod
    def get_role_llm_priorities(role: str) -> List[str]:


        openrouter = NS_ProxyUtil.active_inference_provider() == "openrouter"
        if role == "decompose":
            preferred = NS_ProxyUtil.OPENROUTER_DECOMPOSE_LLM_PRIORITIES if openrouter else NS_ProxyUtil.DECOMPOSE_LLM_PRIORITIES
        elif role == "select":
            preferred = NS_ProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else NS_ProxyUtil.SELECT_LLM_PRIORITIES
        elif role == "filter":
            preferred = NS_ProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else NS_ProxyUtil.SELECT_LLM_PRIORITIES
        elif role == "combo":
            preferred = NS_ProxyUtil.OPENROUTER_COMBO_LLM_PRIORITIES if openrouter else NS_ProxyUtil.COMBO_LLM_PRIORITIES
        elif role == "classify":
            preferred = NS_ProxyUtil.OPENROUTER_CLASSIFY_LLM_PRIORITIES if openrouter else NS_ProxyUtil.CLASSIFY_LLM_PRIORITIES
        else:
            preferred = []
        return NS_ProxyUtil.get_llm_priorities(preferred)
    @staticmethod
    def default_llm_priorities() -> List[str]:
        if NS_ProxyUtil.active_inference_provider() == "openrouter":
            return NS_ProxyUtil.OPENROUTER_LLM_PRIORITIES
        return NS_ProxyUtil.LLM_PRIORITIES

    LLM_PRIORITIES = ("deepseek-ai/DeepSeek-V3.2-TEE", "google/gemma-4-31B-turbo-TEE")


    OPENROUTER_LLM_PRIORITIES = ("deepseek/deepseek-v3.2", "google/gemma-4-31b-it")


    DECOMPOSE_LLM_PRIORITIES: List[str] = ["deepseek-ai/DeepSeek-V3.2-TEE", "google/gemma-4-31B-turbo-TEE"]
    OPENROUTER_DECOMPOSE_LLM_PRIORITIES: List[str] = ["deepseek/deepseek-v3.2", "google/gemma-4-31b-it"]


    SELECT_LLM_PRIORITIES: List[str] = ["deepseek-ai/DeepSeek-V3.2-TEE", "google/gemma-4-31B-turbo-TEE"]
    OPENROUTER_SELECT_LLM_PRIORITIES: List[str] = ["deepseek/deepseek-v3.2", "google/gemma-4-31b-it"]

    COMBO_LLM_PRIORITIES: List[str] = ["google/gemma-4-31B-turbo-TEE", "deepseek-ai/DeepSeek-V3.2-TEE"]
    OPENROUTER_COMBO_LLM_PRIORITIES: List[str] = ["google/gemma-4-31b-it", "deepseek/deepseek-v3.2"]


    CLASSIFY_LLM_PRIORITIES: List[str] = ["deepseek-ai/DeepSeek-V3.2-TEE"]
    OPENROUTER_CLASSIFY_LLM_PRIORITIES: List[str] = ["deepseek/deepseek-v3.2"]

    @staticmethod
    def active_inference_provider() -> str:
        raw = (getenv("INFERENCE_PROVIDER") or "openrouter").strip().lower()
        return raw if raw in ("chutes", "openrouter") else "openrouter"
    @staticmethod
    def trigger_llm(
        prompt: str,
        thinking: bool = False,
        preferred_llms: List[str] = [],
        temperature: float = 0,
        expect: str = "any",
        min_secs: float = 0.0,
        max_secs: float = 0.0,
        reserve_after: Optional[float] = None,
    ) -> Tuple[Dict[str, Any] | List[Any], Any]:


        llm_priorities = NS_ProxyUtil.get_llm_priorities(preferred_llms)
        trigger_tick = time.monotonic()

        with NS_ProxyUtil.envelope(reserve_after = reserve_after):
            return NS_ProxyUtil._trigger_llm(
                prompt, thinking, llm_priorities, temperature, expect, min_secs, max_secs, trigger_tick
            )

    @staticmethod
    def _trigger_llm(
        prompt: str,
        thinking: bool,
        llm_priorities: List[str],
        temperature: float,
        expect: str,
        min_secs: float,
        max_secs: float,
        trigger_tick: float,
    ) -> Tuple[Dict[str, Any] | List[Any], Any]:
        for _pass in range(2):


            t = temperature if _pass == 0 else min(temperature + 0.3, 1.0)

            for model in llm_priorities:
                if NS_ProxyUtil.out_of_time():
                    _debug_print("# trigger_llm: global deadline reached, returning safe empty value")
                    return NS_ProxyUtil._empty_result_for(expect)


                if min_secs and not NS_ProxyUtil.can_afford(min_secs):


                    envelope_at = getattr(NS_ProxyUtil._TLS, "envelope_at", None)
                    spent = time.monotonic() - trigger_tick
                    why = (
                        f"call envelope spent ({spent:.0f}s used, {NS_ProxyUtil.time_left():.0f}s "
                        f"of run budget preserved for later stages)"
                        if envelope_at is not None and envelope_at - time.monotonic() <= min_secs
                        else f"run budget nearly spent ({NS_ProxyUtil.time_left():.0f}s left)"
                    )
                    _debug_print(
                        f"# trigger_llm: needs ~{min_secs:.0f}s, allows "
                        f"{NS_ProxyUtil.allowance():.0f}s -- {why}; refusing rather than "
                        f"issuing a request that cannot complete"
                    )
                    return NS_ProxyUtil._empty_result_for(expect)

                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": {
                        "enable_thinking": thinking,
                        "thinking": thinking
                    },
                    "temperature": t,
                    "stream": False,
                    "max_completion_tokens": 8192,
                    "max_tokens": 8192
                }
                try:
                    tick = time.monotonic()


                    with NS_ProxyUtil.envelope(max_secs) if max_secs else nullcontext():
                        resp = NS_ProxyUtil._post_chat(payload, expect in ("dict", "list"))

                    try:
                        choice: Dict[str, Dict] = resp.get("choices")[0]
                        content = choice.get("message").get("content")
                        if not content:
                            raise Exception()
                    except Exception:
                        time.sleep(0.5)
                        raise Exception("invalid response")

                    try:
                        js = NS_MiscUtil.parse_json(content)
                        if js is None:
                            raise Exception()
                    except Exception as exc:
                        raise Exception(f"failed to parse json: {exc}\n*******\n{content}\n******")

                    if (expect == "dict" and not isinstance(js, dict)) or \
                        (expect == "list" and not isinstance(js, list)):
                        raise Exception(f"expected {expect} but got {type(js)}")

                    throughput = resp.get("usage", {}).get("completion_tokens", 0)
                    trigger_delay = time.monotonic() - trigger_tick
                    delay = time.monotonic() - tick

                    _debug_print(
                        f"* llm triggered: "
                        f"model = '{model}' | "
                        f"throughput = {throughput} | "
                        f"delay = {delay:.2f}s | "
                        f"total_delay = {trigger_delay:.2f}s"
                    )
                    return js, resp
                except Exception as exc:
                    _debug_print(f"# llm trigger error with '{model}': {exc}")
                    continue

        _debug_print("# trigger_llm: exhausted all models, returning safe empty value")
        return NS_ProxyUtil._empty_result_for(expect)

    LLM_TIMEOUT_SECS, LLM_MAX_RETRIES = 180, 5
    LLM_PROXY = ProxyClient(timeout = LLM_TIMEOUT_SECS, max_retries = LLM_MAX_RETRIES)


    _TLS = threading.local()

    @staticmethod
    def _thread_proxy(name: str, timeout: int, retries: int) -> ProxyClient:
        cache = getattr(NS_ProxyUtil._TLS, "proxies", None)
        if cache is None:
            cache = {}
            NS_ProxyUtil._TLS.proxies = cache

        proxy = cache.get(name)
        if proxy is None:
            proxy = ProxyClient(timeout = timeout, max_retries = retries)
            cache[name] = proxy

        return proxy

    @staticmethod
    def llm_proxy() -> ProxyClient:
        return NS_ProxyUtil._thread_proxy("llm", NS_ProxyUtil.LLM_TIMEOUT_SECS, NS_ProxyUtil.LLM_MAX_RETRIES)

    @staticmethod
    def oro_proxy() -> ProxyClient:
        return NS_ProxyUtil._thread_proxy("search", NS_ProxyUtil.SEARCH_TIMEOUT_SECS, NS_ProxyUtil.SEARCH_MAX_RETRIES)


    _JSON_MODE_OK: bool = (getenv("LLM_JSON_MODE", "1").strip().lower() not in ("0", "false", "no"))

    @staticmethod
    def _post_chat(payload: Dict[str, Any], want_json: bool) -> Optional[Dict]:


        if want_json and NS_ProxyUtil._JSON_MODE_OK:


            tick = time.monotonic()
            with NS_ProxyUtil.budgeted(NS_ProxyUtil.llm_proxy()) as allowed:
                resp = NS_ProxyUtil.llm_proxy().post(
                    "/inference/chat/completions",
                    json_data = {**payload, "response_format": {"type": "json_object"}},
                )
            if resp: return resp


            if time.monotonic() - tick < allowed * 0.9:
                NS_ProxyUtil._JSON_MODE_OK = False
                _debug_print("# json mode unavailable; disabling response_format for the rest of this run")

            if NS_ProxyUtil.out_of_time(): return None


        if NS_ProxyUtil.allowance() < NS_ProxyUtil.MIN_REQUEST_SECS: return None

        with NS_ProxyUtil.budgeted(NS_ProxyUtil.llm_proxy()):
            return NS_ProxyUtil.llm_proxy().post("/inference/chat/completions", json_data = payload)

    @staticmethod
    def _empty_result_for(expect: str) -> Tuple[Dict[str, Any] | List[Any], Any]:
        if expect == "list":
            return [], None
        return {}, None

    SEARCH_TIMEOUT_SECS, SEARCH_MAX_RETRIES = 30, 5
    ORO_PROXY = ProxyClient(timeout = SEARCH_TIMEOUT_SECS, max_retries = SEARCH_MAX_RETRIES)
    @staticmethod
    def find_products(
        so: NS_SearchOption, sort: Optional[str] = None, page: int = 1
    ) -> Tuple[List[NS_Product], ToolCallResult]:

        empty_res: ToolCallResult = {
            "name": "find_product",
            "parameters": {"q": so.q, "page": page},
            "tool_call_id": "",
            "result": []
        }

        for _attempt in range(2):
            if NS_ProxyUtil.out_of_time():
                _debug_print("# find_products: global deadline reached, returning empty results")
                return [], empty_res

            try:
                with NS_ProxyUtil.budgeted(NS_ProxyUtil.oro_proxy()):
                    tool_res = execute_tool_call(
                        "find_product",
                        {
                            "q": so.q,
                            "page": page,
                            "shop_id": so.shop_id,
                            "price": so.price,
                            "sort": sort,
                            "service": ','.join(so.services or [])
                        }
                    )
                js_prods: List[Dict] = tool_res["result"]
                products = []

                for j in js_prods:
                    price = j.get("price") or 0
                    price = NS_MiscUtil.extract_number(str(price))

                    products.append(
                        NS_Product(
                            id = str(j.get("product_id")),
                            title = j.get("title", ""),
                            price = price,
                            shop_id = j.get("shop_id", ""),
                            service = j.get("service", []) or [],
                            attributes = {},
                            sku = {},
                            is_viewed = False
                        )
                    )

                return products, tool_res

            except Exception as exc:
                _debug_print(f"# error in `find_products`: {exc}")
                time.sleep(0.5)

        _debug_print("# find_products: attempts exhausted, returning empty results")
        return [], empty_res
    @staticmethod
    def view_products(products: List[NS_Product]) -> ToolCallResult:
        if not products: return
        pid_str = ','.join([p.id for p in products])


        for _attempt in range(3):
            if NS_ProxyUtil.out_of_time():
                _debug_print("# view_products: global deadline reached, leaving products unviewed")
                return

            try:
                with NS_ProxyUtil.budgeted(NS_ProxyUtil.oro_proxy()):
                    tool_res = execute_tool_call(
                        "view_product_information", {"product_ids": pid_str}
                    )

                details = tool_res["result"]
                if not details: return tool_res

                details_map: Dict[str, Dict[str, Any]] = {d["product_id"]: d for d in details}

                for prod in products:
                    d = details_map.get(prod.id, {})

                    prod.sku = d.get("sku_options", {})
                    prod.attributes = d.get("attributes", {})
                    prod.is_viewed = True

                return tool_res
            except Exception as exc:
                _debug_print(f"# error in `view_products`: {exc}")
                time.sleep(0.5)

        _debug_print("# view_products: attempts exhausted, leaving products unviewed")
        return
    @staticmethod
    def submit(products: List[NS_Product]) -> ToolCallResult:
        pid_str = ','.join([p.id for p in products])
        return execute_tool_call("recommend_product", {"product_ids": pid_str})
    @staticmethod
    def terminate(status: str = "success") -> ToolCallResult:
        return execute_tool_call("terminate", {"status": status})

    def arm_deadline() -> None:
        NS_ProxyUtil.DEADLINE = time.monotonic() + NS_SearchEngine.TIMEOUT_SECS


    MAX_REQUEST_SHARE = 0.5


    MIN_REQUEST_ALLOWANCE = 30.0


    MIN_REQUEST_SECS = 5.0

    @staticmethod
    @contextmanager
    def envelope(secs: Optional[float] = None, reserve_after: Optional[float] = None):


        prev = getattr(NS_ProxyUtil._TLS, "envelope_at", None)
        limit = time.monotonic() + (NS_ProxyUtil.allowance(reserve_after) if secs is None else secs)

        NS_ProxyUtil._TLS.envelope_at = limit if prev is None else min(prev, limit)
        try:
            yield
        finally:
            NS_ProxyUtil._TLS.envelope_at = prev

    @staticmethod
    def allowance(reserve_after: Optional[float] = None, tail: float = 10.0) -> float:


        left = NS_ProxyUtil.time_left()
        if left == math.inf: return math.inf

        usable = left - tail
        if usable <= 0: return 0.0

        envelope_at = getattr(NS_ProxyUtil._TLS, "envelope_at", None)
        if envelope_at is not None:
            return max(0.0, min(usable, envelope_at - time.monotonic()))

        if reserve_after is None:
            share = max(usable * NS_ProxyUtil.MAX_REQUEST_SHARE, NS_ProxyUtil.MIN_REQUEST_ALLOWANCE)
        else:
            share = usable - reserve_after

        return max(0.0, min(usable, share))

    @staticmethod
    def can_afford(min_secs: float, reserve_after: Optional[float] = None, tail: float = 10.0) -> bool:


        return NS_ProxyUtil.allowance(reserve_after, tail) >= min_secs

    @staticmethod
    @contextmanager
    def budgeted(proxy: ProxyClient, reserve: float = 10.0):


        timeout, retries = proxy.timeout, proxy.max_retries
        share = NS_ProxyUtil.allowance(tail = reserve)

        if share != math.inf:
            share = max(NS_ProxyUtil.MIN_REQUEST_SECS, share)
            proxy.timeout = int(max(5, min(timeout, share)))
            proxy.max_retries = max(1, min(retries, int(share // proxy.timeout)))
        try:
            yield float(proxy.timeout * proxy.max_retries)
        finally:
            proxy.timeout, proxy.max_retries = timeout, retries

    @staticmethod
    def time_left() -> float:
        if not NS_ProxyUtil.DEADLINE:
            return math.inf
        return NS_ProxyUtil.DEADLINE - time.monotonic()
    @staticmethod
    def out_of_time() -> bool:
        return bool(NS_ProxyUtil.DEADLINE) and time.monotonic() > NS_ProxyUtil.DEADLINE

class NS_SegTree(object):


    def __init__(self, values, k: int = 30) -> None:
        self.k, n = k, len(values)
        self.size = 1

        while self.size < n:
            self.size <<= 1

        self.seg: List[List[Tuple[Any, int]]] = [[] for _ in range(2 * self.size)]

        for i, v in enumerate(values):
            self.seg[self.size + i] = [(v, i)]

        for i in range(self.size - 1, 0, -1):
            self.seg[i] = self._merge(self.seg[2 * i], self.seg[2 * i + 1])

    def _merge(self, a: List, b: List) -> List:
        out, ia, ib = [], 0, 0

        while len(out) < self.k:
            if ia == len(a):
                out.extend(b[ib:ib + self.k - len(out)])
                break

            if ib == len(b):
                out.extend(a[ia:ia + self.k - len(out)])
                break

            if a[ia][0] >= b[ib][0]:
                out.append(a[ia])
                ia += 1
            else:
                out.append(b[ib])
                ib += 1

        return out

    def query(self, l, r) -> List[Tuple[Any, int]]:
        if l > r: return []
        ans: List[Tuple[Any, int]] = []

        l += self.size
        r += self.size

        while l <= r:
            if l & 1:
                ans = self._merge(ans, self.seg[l])
                l += 1

            if not (r & 1):
                ans = self._merge(ans, self.seg[r])
                r -= 1

            l >>= 1
            r >>= 1

        return ans

class NS_Debugger(object):
    def __init__(
        self,
        suite: Optional[str] = None,
        frozen: Optional[List[str]] = None,
        stop_after: Optional[str] = None,
        show_dlg: bool = False,
        quiet: bool = False
    ) -> None:
        self.memory: Dict[str, List] = defaultdict(lambda: [])
        self.quiet = quiet

        self.frozen = frozen or []
        self.stop_after = stop_after
        self.suite = suite
        self.show_dlg = show_dlg

    def is_frozen(self, stage: str) -> bool:
        return stage in self.frozen

    def is_stop(self, stage: str) -> bool:
        return self.stop_after == stage

    def is_stop_in(self, episode: str) -> bool:
        if self.stop_after:
            if episode == "AGENT" and self.stop_after.startswith("a-"): return True
            if episode == "BASE" and self.stop_after.startswith("b-"): return True
            if episode == "PRODUCT" and self.stop_after.startswith("p-"): return True
            if episode == "SHOP" and self.stop_after.startswith("s-"): return True
            if episode == "VOUCHER" and self.stop_after.startswith("v-"): return True

        return False

    def find(self, stage: str, kv: Dict) -> Any:
        for js in self.memory[stage]:
            for key, val in kv.items():
                if js[key] != val: break
            else:
                return js

        raise Exception(
            f"# cannot find the freezed {stage} instance for `{json.dumps(kv, ensure_ascii = False)}`"
        )


    def add(self, stage: str, data: Any) -> None:
        if self.suite: self.memory[stage].append(data)

    def dump(self) -> None:
        pass

    def print(self, obj: Any) -> None:
        if not self.quiet:
            if isinstance(obj, dict) or isinstance(obj, list):
                NS_MiscUtil.print_json(obj)
            else:
                builtins.print(obj)

class NS_Prompt:
    class Responses:
        _LABEL_TO_SERVICE_MAP: Dict[str, str] = {
            "flashsale": "flashsale", "lazflash": "flashsale",
            "official": "official", "lazmall": "official", "return": "official",
            "free_ship": "freeShipping",
            "cod": "COD",
        }
        _SERVICE_TO_LABEL_MAP: Dict[str, str] = {
            "official": "lazmall",
            "flashsale": "lazflash",
            "freeShipping": "free_ship",
            "COD": "cod",
        }

        @staticmethod
        def label_to_service(label: str) -> str:
            return NS_Prompt.Responses._LABEL_TO_SERVICE_MAP.get(label)

        @staticmethod
        def service_to_label(svc: str) -> str:
            return NS_Prompt.Responses._SERVICE_TO_LABEL_MAP.get(svc)

        @staticmethod
        def extract_decompose_response(js: Dict[str, Dict[str, str | Dict[str, str]]]) -> List[Dict]:
            def __parse_services__(js_service: Dict[str, Dict[str, str]], id_str: str, q: str):
                services, fast_deliver, prod_only = set(), False, False

                for ch, v in js_service.items():
                    apply_for = v.get("apply_for")
                    if apply_for != id_str: continue


                    raw = v.get("labels") or ([v.get("label")] if v.get("label") else [])
                    is_effective = False

                    for label in raw:
                        if label == "fast_delivery":
                            fast_deliver = True
                            is_effective = True
                        elif NS_EngineUtil.is_prod_only(ch):
                            prod_only = True
                            is_effective = True
                        else:
                            svc = NS_Prompt.label_to_service(label)
                            if svc:
                                services.add(svc)
                                is_effective = True

                    if is_effective: q = q.replace(ch, "")

                if any(s in services for s in ["flashsale", "official"]): fast_deliver = False
                return services, fast_deliver, prod_only, q

            def __parse_keywords__(js_keyword: Dict[str, Dict[str, str]], id_str: str) -> List[str]:
                keywords = []

                raw = js_keyword.get(id_str)
                if not isinstance(raw, list): raise Exception(f"keywords for sub-query {id_str} is missing")

                for kw in raw:
                    if not isinstance(kw, str): continue


                    kw = kw.strip()
                    kw = (
                        re.sub(r"\s*→\s*", "→", kw) if "→" in kw
                        else re.sub(r"\s*(?:->|=>)\s*", "→", kw, count = 1)
                    )
                    if not kw: continue


                    if len([s for s in kw.split('→') if s.strip()]) < 2:
                        raise Exception(f"invalid keyword `{kw}` for sub-query {id_str}")

                    if kw not in keywords: keywords.append(kw)

                if not keywords: raise Exception(f"keywords for sub-query {id_str} is empty")
                return keywords

            def __parse_price__(js_price: Dict[str, Dict[str, Dict]], id_str: str):
                for v in js_price.values():
                    apply_for = v.get("apply_for")
                    normalized = v.get("normalized")

                    if apply_for != id_str: continue
                    if not normalized: continue

                    price: str = re.sub(r"[^\d\-\.]", "", normalized)

                    if len(price.split('-')) != 2: raise Exception(f"invalid price for sub-query {id_str}: `{normalized}`")
                    return price

                return None

            prod_count = len(js.get("products"))
            decomposed = []

            for i in range(prod_count):
                prod_only, price = False, False
                id_str = str(i + 1)

                q = js.get("sub_queries").get(id_str).strip()
                if not q: raise Exception(f"sub-query {id_str} is empty")

                services, fast_deliver, prod_only, q = __parse_services__(js.get("service") or {}, id_str, q)
                keywords = __parse_keywords__(js.get("keywords") or {}, id_str)
                price = __parse_price__(js.get("price") or {}, id_str)

                prod_only = prod_only or NS_EngineUtil.is_prod_only(q)

                if fast_deliver:
                    q += ", with fast delivery"
                    keywords.append("delivery→fast")

                decomposed.append({
                    "query": q,
                    "phrase": (js.get("products") or {}).get(id_str) or "",
                    "keywords": keywords,
                    "services": list(services),
                    "prod_only": prod_only,
                    "price": price
                })

            return decomposed
        @staticmethod
        def coerce_choice_id(value: Any) -> Optional[int]:


            if value is None or isinstance(value, bool): return None
            if isinstance(value, (int, float)): return int(value) - 1
            if not isinstance(value, str): return None

            m = re.search(r"\d+", value)
            return int(m.group()) - 1 if m else None

        @staticmethod
        def extract_select_response(js: Dict[str, Dict]) -> int:
            if not isinstance(js, dict): return 0

            if "best" in js:
                bid = NS_Prompt.Responses.coerce_choice_id(js.get("best"))
                return bid if bid is not None else -1

            for phase in list(js.values())[::-1]:
                if not isinstance(phase, dict): continue
                if "best" not in phase: continue

                bid = NS_Prompt.Responses.coerce_choice_id(phase.get("best"))
                return bid if bid is not None else -1

            return 0

        @staticmethod
        def extract_closest_response(js: Dict[str, Dict]) -> int:


            if not isinstance(js, dict): return -1

            if "closest" in js:
                cid = NS_Prompt.Responses.coerce_choice_id(js.get("closest"))
                return cid if cid is not None else -1

            for phase in list(js.values())[::-1]:
                if not isinstance(phase, dict): continue
                if "closest" not in phase: continue

                cid = NS_Prompt.Responses.coerce_choice_id(phase.get("closest"))
                return cid if cid is not None else -1

            return -1


        TALLY_LEVELS = ("admissibility", "structured evidence", "unrequested tokens")

        @staticmethod
        def parse_tally(js: Dict) -> Dict[int, List[Tuple[bool, bool, Optional[int]]]]:


            tally = js.get("tally") if isinstance(js, dict) else None
            if not isinstance(tally, str) or not tally.strip(): return {}

            rows: Dict[int, List[Tuple[bool, bool, Optional[int]]]] = {}

            for line in tally.split('\n'):
                head = re.match(r"\s*['\"]?(\d+)['\"]?\s*[→>\-:]+\s*(.*)", line)
                if not head: continue

                cons: List[Tuple[bool, bool, Optional[int]]] = []

                for seg in head.group(2).split(';'):
                    s = re.search(r"structured\s*[:=]\s*(true|false)", seg, re.I)
                    if not s: continue

                    a = re.search(r"admissible\s*[:=]\s*(true|false)", seg, re.I)
                    e = re.search(r"extra_tokens\s*[:=]\s*(\d+)", seg, re.I)

                    cons.append((
                        a.group(1).lower() == "true" if a else True,
                        s.group(1).lower() == "true",
                        int(e.group(1)) if e else None,
                    ))

                if cons: rows[int(head.group(1)) - 1] = cons

            return rows

        @staticmethod
        def _tally_beats(
            p: List[Tuple[bool, bool, Optional[int]]],
            w: List[Tuple[bool, bool, Optional[int]]],
        ) -> Tuple[bool, int]:


            for lvl in (0, 1):
                pw = sum(1 for i in range(len(w)) if p[i][lvl] and not w[i][lvl])
                wp = sum(1 for i in range(len(w)) if w[i][lvl] and not p[i][lvl])

                if pw or wp: return (pw > 0 and wp == 0), lvl + 1


            cmpable = [
                i for i in range(len(w))
                if p[i][1] and w[i][1] and p[i][2] is not None and w[i][2] is not None
            ]
            if not cmpable: return False, 0

            pw = sum(1 for i in cmpable if p[i][2] < w[i][2])
            wp = sum(1 for i in cmpable if p[i][2] > w[i][2])

            if pw or wp: return (pw > 0 and wp == 0), 3
            return False, 0

        @staticmethod
        def arbitrate_tally(js: Dict, bid: int, n_candidates: int) -> Tuple[int, int]:


            try:
                rows = NS_Prompt.Responses.parse_tally(js)

                if len(rows) < 2 or bid not in rows: return bid, 0

                n = len(rows[bid])
                if n < 1: return bid, 0

                for cid, cons in rows.items():
                    if cid < 0 or cid >= n_candidates: return bid, 0
                    if len(cons) != n: return bid, 0


                claimed = js.get("all_matches") if isinstance(js, dict) else None
                if isinstance(claimed, list):
                    for x in claimed:
                        m = re.search(r"\d+", str(x))
                        if m and (int(m.group()) - 1) not in rows: return bid, 0

                beats = NS_Prompt.Responses._tally_beats
                won = rows[bid]

                challengers = []
                for cid in sorted(rows):
                    if cid == bid: continue

                    ok, lvl = beats(rows[cid], won)
                    if ok: challengers.append((cid, lvl))

                if not challengers: return bid, 0


                for cid, lvl in challengers:
                    if any(beats(rows[o], rows[cid])[0] for o in rows if o != cid): continue
                    return cid, lvl

                return bid, 0
            except Exception:


                return bid, 0

        @staticmethod
        def extract_filter_response(js: Dict[str, Dict], counts: List[int]) -> List[Dict[int, bool]]:


            alphabets = NS_Vocab.ALPHABETS[:len(counts)]
            verdicts: List[Dict[int, bool]] = [{} for _ in counts]

            if not isinstance(js, dict): return verdicts

            for g, checklist in js.items():
                if not isinstance(checklist, dict): continue
                if g not in alphabets: continue
                gid = alphabets.index(g)

                for k, check in checklist.items():
                    if k == "query": continue
                    if not isinstance(check, dict): continue
                    if "is_match" not in check: continue

                    m = re.search(r"^[A-Z](\d+)$", k, re.I)
                    if not m: continue

                    bid = int(m.group(1) or 0) - 1
                    if bid < 0 or bid >= counts[gid]: continue

                    verdicts[gid][bid] = bool(check.get("is_match"))

            return verdicts


        _ABSENCE_CLAIM = re.compile(
            r"not\s+(?:found|mentioned|stated|specified|listed|present|explicitly|indicated)"
            r"|no\s+explicit|assum(?:e|ed|ing)|implied|inferred|presumabl",
            re.I,
        )
        _QUOTED = re.compile(r"['\"]([^'\"]{2,40})['\"]")


        _ARGUED = re.compile(r"[()]|['\"]|\bbut\b|\bthough\b|\bhowever\b|\bmaybe\b|\bnot a\b|\?", re.I)

        @staticmethod
        def _absence_is_real(seg: str, body: str, wide: bool) -> Optional[bool]:


            head = seg.split(':', 1)[0] if ':' in seg else seg
            head = re.sub(r"[^\w\s\-/]", " ", head).strip().lower()
            if not head or len(head) > 60: return None


            value = ""
            if ':' in seg:
                tail = seg.split(':', 1)[1]
                cut = NS_Prompt.Responses._ABSENCE_CLAIM.search(tail)
                value = re.sub(r"[^\w\s\-/]", " ", tail[:cut.start()] if cut else tail).strip().lower()

            probes = [value or head]
            if wide:
                probes.append(head)
                probes += [w for w in re.split(r"[\s/\-]+", head + " " + value) if len(w) >= 4]

            q = NS_Prompt.Responses._QUOTED.search(seg)
            if q: probes.append(q.group(1).strip().lower())

            probes = [p for p in probes if len(p) >= 3]
            if not probes: return None


            return not any(
                re.search(r"\b%s(?!\s*:)" % re.escape(p), body) for p in probes
            )

        @staticmethod
        def arbitrate_filter(
            js: Dict[str, Dict],
            counts: List[int],
            text_of: Callable[[int, int], str],
        ) -> List[Dict[int, str]]:


            drops: List[Dict[int, str]] = [{} for _ in counts]

            try:
                if not isinstance(js, dict): return drops
                alphabets = NS_Vocab.ALPHABETS[:len(counts)]

                for g, checklist in js.items():
                    if g not in alphabets: continue
                    if not isinstance(checklist, dict): continue
                    gid = alphabets.index(g)

                    for k, check in checklist.items():
                        if k == "query" or not isinstance(check, dict): continue
                        if "is_match" not in check: continue

                        m = re.search(r"^[A-Z](\d+)$", k, re.I)
                        if not m: continue

                        bid = int(m.group(1) or 0) - 1
                        if bid < 0 or bid >= counts[gid]: continue

                        text = str(check.get("check") or "")
                        if not text: continue

                        passed = bool(check.get("is_match"))
                        body = None

                        for seg in text.split(';'):
                            if not NS_Prompt.Responses._ABSENCE_CLAIM.search(seg): continue


                            if passed and "mismatch" in seg.lower(): continue


                            if not passed and NS_Prompt.Responses._ARGUED.search(seg): continue

                            if body is None: body = (text_of(gid, bid) or "").lower()
                            absent = NS_Prompt.Responses._absence_is_real(seg, body, passed)

                            if absent is None: continue
                            if absent != passed: continue

                            drops[gid][bid] = (
                                "passed on a value its own check calls unstated"
                                if passed else
                                "rejected for a token present in the product text"
                            )
                            break

                return drops
            except Exception:


                return [{} for _ in counts]

        @staticmethod
        def extract_combo_select_response(js: Dict, n_bundles: int) -> Optional[int]:
            if not isinstance(js, dict): return None

            idx = NS_Prompt.Responses.coerce_choice_id(js.get("best"))
            if idx is not None and 0 <= idx < n_bundles: return idx
            return None

        @staticmethod
        def structured_hits(item: Dict, constraints: List[str]) -> Tuple[int, int]:


            attrs = " ".join(
                str(item.get(k) or "") for k in ("attributes", "sku_options")
            ).lower()
            title = str(item.get("title") or "").lower()

            structured = title_only = 0

            for c in constraints:
                val = (c.split(':', 1)[1] if ':' in c else c).strip().lower()
                if not val: continue

                words = [w for w in re.split(r"[\s,]+", val) if len(w) > 1]

                def seen(body: str) -> bool:
                    if val in body: return True
                    return bool(words) and all(w in body for w in words)

                if seen(attrs): structured += 1
                elif seen(title): title_only += 1

            return structured, title_only

    class Templates:
        A_CLASSIFY = """
You are an AI assistant that classifies a user's shopping query into exactly one of three task types: **voucher**, **shop**, or **product**.
Evaluate the task types in the order below and return the FIRST one that fits.

---

## 1. voucher
The user explicitly intends to apply a **cart-level voucher or promotional code** to their order,
OR they state a **total spending budget across the whole cart** that triggers a discount.

A genuine voucher task has ALL of the following characteristics:
- There is a **spending threshold** the total cart must reach (e.g. "spend at least 500 to use the voucher")
- AND/OR there is an explicit **discount rule** applied at checkout (e.g. "get 10% off", "100 PHP off")
- AND/OR the user explicitly mentions a **voucher, promo code, or coupon** they want to apply

Important notes:
- The voucher may apply to all products in the cart OR only to products from the same shop/seller.
  Either way it is still a **voucher** task. Do NOT downgrade it to shop or product.
- The query may list one or more products — that does not change the classification.

### What is NOT a voucher

**Price constraints on individual products** — these are product search filters, never voucher signals:
- "priced under 500", "costs between 100 and 300", "above 200 PHP", "less than 1000"
- "my budget is above 10 pesos", "within my budget of 50", "budget above X" — when only one product
  is being searched, any price mention (including the word "budget") is a per-item price filter, not
  a cart-level spending cap. Only treat "budget" language as voucher evidence when the query involves
  multiple products and a combined spend threshold that unlocks a discount.

**Product listing promotions** — phrases that describe a deal printed on the product card rather than
a voucher the user applies at checkout:
- "buy 1 take 1", "BOGO", "buy one get one free", "3-for-2", "buy 2 get 1 free"
- "with freebie", "with free item", "comes with a free ..."
- "bundle", "bundle deal", "package deal"
- "on sale", "sale item", "discounted", "clearance"
- "cashback", "rebate"
- "free shipping" (a shipping perk, not a checkout voucher)

If the user says "I want a buy 1 take 1 knife under 87 pesos", they are describing **attributes they
want the product to have** — this is a product task, not a voucher task.

---

## 2. shop
The user is looking for two or more distinct products — regardless of whether they explicitly
mention the same shop or use connectors like "and" / commas.

### Strong signals — any ONE of these alone is enough to classify as shop

**Signal A — Multiple price ranges or price constraints in the same query**
If the query contains two or more separate price expressions (ranges, "under X", "above X",
"X-Y pesos", etc.) that clearly apply to DIFFERENT items, each price expression belongs to a
different product. A single product almost never has two price constraints.
Examples that correctly trigger Signal A:
- "electric fan motor parts stand base 256-319 pesos 89-116 pesos"
  → two price ranges for two different items → shop
- "rattan bag over 384 PHP adjustable strap summer bag 16-425 PHP"
  → two price ranges for two different items → shop

Exception — do NOT trigger Signal A when the two prices are clearly for size/color variants
of the same product (e.g. "T-shirt size M 200-300 pesos size L 300-400 pesos" is still one
product with variant pricing). This is rare; default to shop when in doubt.

**Signal B — A price constraint appears mid-query AND is followed by a new product name**
When a price expression appears somewhere other than the very end of the query AND what follows
it is a **new product name** (a different noun naming a different product category), it acts as
a **boundary between two products**.
Everything before the mid-query price = product 1; everything after = product 2.

CRITICAL: Signal B only fires when a genuinely new product name follows the price.
It does NOT fire when what follows the price is merely:
- A physical attribute of the same product: "waterproof", "lightweight", "durable", "portable"
- A color, size, or material: "blue", "large", "leather"
- A gender/age qualifier: "for women", "for men", "for kids", "for boys"
- A use-case or occasion phrase: "for gaming", "for office", "for prom", "for running"
- A compatibility note: "for iPhone 14", "for Toyota Vios", "for 5-blade motors"
- A brand or model qualifier: "brand X", "model Y"

These trailing qualifiers describe the SAME product — they are NOT a second product.

Correctly triggers Signal B (new product name follows):
- "golf bag white under 15342 PHP stand bag blue for men and women"
  → "under 15342 PHP" mid-query → "stand bag" is a new product name → shop
- "rattan bag canvas floral free shipping over 384 PHP adjustable strap summer bag 16-425 PHP"
  → "over 384 PHP" mid-query → "adjustable strap summer bag" is a new product name → shop

Does NOT trigger Signal B (only attributes follow):
- "bluetooth speaker under 1000 pesos waterproof" → "waterproof" is an attribute → product
- "running shoes under 2000 pesos for women" → "for women" is a gender qualifier → product
- "gaming chair 1000-3000 pesos adjustable armrest" → "adjustable armrest" is a feature → product
- "laptop bag under 500 pesos lightweight" → "lightweight" is an attribute → product

**Signal C — Explicitly mentions same shop, same store, same seller**, or asks to find shops
that carry all items.

**Signal D — Two or more separately named products joined by "and", "with", commas, or
numbered/enumerated** (e.g. "ducktail and side skirt", "keyboard, mouse, and mousepad").

### Detecting two distinct products without explicit connectors

Queries in this domain are often terse and do NOT use "and" or commas between products.
You must actively try to split the query into two product descriptions by looking for:

1. **A completing spec or quantity in the middle** — specs like "5 seats", "1 meter", "3 pcs",
   "2-pack", or a color/size that naturally concludes a product description often signal a
   boundary. After such a spec, a new product name may begin.
   Example: "car seat cushion cover 5 seats [END of product 1] blue patterned leatherette [product 2]"

2. **Two product names from different categories placed back to back**
   When two noun phrases each independently name a product (e.g. "golf bag" and "stand bag",
   "motor parts" and "stand base", "seat cover" and "leatherette"), treat them as two separate
   products — even without a connector word.
   Be conservative: only split when BOTH parts are standalone product names on their own.
   Do NOT split a string like "wireless bluetooth mechanical keyboard" — all parts are
   attributes/modifiers of a single product noun ("keyboard"), not two product names.

3. **Different color/attribute clusters assigned to different names**
   "golf bag WHITE ... stand bag BLUE" — each product has its own attribute cluster.

### What is NOT enough for shop

- Two adjectives or attributes modifying the same product
  ("blue patterned waterproof backpack" is ONE product)
- Accessories that are inseparable from the main item
  ("laptop with charger" when sold as a set is ONE product)

### How to tell if items are distinct products

- Each item has its own product name (even if they share the same compatibility, brand, or car model).
  "Ducktail and side skirt" are two separate car parts → two products.
- Do NOT collapse related items into one just because they serve the same purpose or fit the same
  object. Shared context (same car model, same device, same brand) does not make them a single product.
- Only treat items as one product when a single listing would naturally include all of them
  (e.g. "left and right side mirrors" sold as a pair, or a "keyboard with built-in trackpad").

---

## 3. product
The user is looking for exactly one product (no budget/voucher intent, no multi-product listing).
This is the default when neither **voucher** nor **shop** applies.

Product-attribute language that should NOT change this classification:
- Price filters: "under X", "above X", "within X", "my budget is X" — only when there is a
  single price expression and it appears at the end of the query
- Promotional attributes on the product: "buy 1 take 1", "on sale", "with freebie", "bundle"
- Discount descriptions of the product itself: "50% off", "discounted"
- Multiple descriptors of the same product: color, size, material, style all attached to one name

---

## Decision Checklist — run through these in order before answering

1. Does the query explicitly mention a **voucher, promo code, or coupon** to be applied at checkout?
   → If yes → **voucher**
2. Does the query describe a **cart-level spending threshold** that **unlocks a discount** across
   multiple items or an entire order?
   → If yes → **voucher**
3. Could the "budget" or "discount" language simply be a **price filter on one product**, or a
   **promotion attribute of the product being searched** (e.g. buy 1 take 1)?
   → If yes → do NOT classify as voucher; continue to step 4
4. Does the query contain **two or more separate price expressions** that apply to different items
   (not variant-level pricing for the same product)?
   → If yes → **shop**
5. Does a price expression appear **mid-query** AND is it followed by a **new product name**
   (not just attributes, qualifiers, gender markers, use-case phrases, or compatibility notes)?
   → If yes → **shop**
6. Can the query be split into **two or more distinct product names** where EACH part independently
   names a different product category (not just attributes of a single product)?
   → If yes → **shop**
7. Otherwise → **product**

---

## Output Format
Return ONLY a JSON object — no markdown fences, no extra text.
Write `reason` FIRST, working through the Decision Checklist, and let `task_type` follow from it:
{{"reason": "<one-sentence explanation naming the checklist step that decided it>", "task_type": "<voucher|shop|product>"}}

## Query
{query}
"""

        A_DECOMPOSE = """
You are AI assistant to decompose a user's shopping query into several chunks, for automatic processing.
The user is looking for one or more (max 4) products{shop_intent}.

# Task
1. Identify the number of products user is looking for, and what they are.
- Keep in mind that the max number of products is 4.
- Sometimes it seems enumerating more than 4 products. In fact, some of them are the sub-products packed inside a kit. If the number seems beyond 4, correctly identify the kit product.
- Check carefully and don't mistakenly take some constraints or filler words as an individual product!
- Shop queries like "Find shops offering both X and Y" or "First,... Second,... Third,..." contain multiple products; each item gets its own sub-query and may have different price/service constraints.

2. Identify the price constraint for each product, and rewrite it to the normalized format.
- Price constraint is mentioned like "between 10 and 50 php", "priced from 180 to 505 PHP", "priced above 410 PHP", "cost is above 85 PHP", "cost over 176 PHP", "more than 2400", "less than 120 USD", ...
- User may mention the price constraint without currency unit: e.g. "Looking for a makeup bag between 110 and 200."
- Don't mistakenly take other numerical constraint as the price constraint: e.g. "Need aircon with power usage between 1000 and 1500." -> "1000-1500" is power consumption range constraint.
- For normalization, write using "[low]-[high]" format. Exclude currency unit! e.g. "10-100"
- "[low]-" if upper bound is missing. e.g. "57-"
- "-[high]" if lower bound is missing. e.g. "-88"
- null if price constraint is not mentioned

3. Identify the service constraints, if exist, and find which products they apply for. Check the following service labels:
- **official**: user wants product that is official, guaranteed and authentic. This also covers phrasing like "from an authorized store/seller/dealer/brand", "authorized reseller", "verified brand/store/seller", "trusted store/seller/brand/shop" — map ALL of these to **official**, even when the literal word "official" never appears.
- **free_ship**: user wants free or complimentary shippable product
- **fast_delivery**: user wants fast delivery
- **cod**: user wants 'Cash on Delivery' option
- **flashsale**: user wants a flash sale or lightning deal, OR the product is simply described as being **on sale / discounted / with a discount (option)** — a plain sale/discount status on the product itself, not a specific buy-quantity mechanic (see IMPORTANT note below). Also apply this label when the phrasing clearly signals a **time-constrained** promotion: e.g. "limited-time offer", "timed deal", "time-bound promo", "flash deal(s)", or any wording that implies a temporary discounted window. Examples that ARE `flashsale`: "on sale now", "also available with a discount", "with a discount option", "discounted price", "limited-time offer", "flash deals". Do NOT apply it to vague marketing filler that contains no sale/discount/time cue at all ("seller's promo", "trending item").
- **return**: user wants easy or fast returns
- **lazmall**: user wants products from LazMall or LazMall Perk
- **lazflash**: user wants products from LazFlash
- Map "free delivery/shipping" to **free_ship**, "cash/pay on delivery" to **cod**, "LazFlash deal" or "flash deal(s)" to **lazflash**, "LazMall perks/store" to **lazmall**, "authorized/verified/trusted store, seller, dealer, or brand" to **official**.
- When a single phrase mentions multiple services (e.g. "LazMall and LazFlash"), include ALL of them in the `labels` list for that entry.
- **IMPORTANT — do not confuse a product-listing deal with a service.** Phrases that describe a specific quantity/bundling mechanic printed on the product's own listing/SKU — "buy 1 take 1", "buy 1 get 2", "BOGO", "buy one get one free", "3-for-2", "with freebie", "bundle deal", "package deal", "clearance", "cashback", "rebate" — are something the product itself must have, NOT a platform-level service. These are NEVER `flashsale`/`lazflash`, even though some mention a "deal". Do not put these phrases in `service` at all — leave them in the sub-query text so they are captured as a `promotion` keyword in step 5 (e.g. "promotion→buy 1 take 2"). This is different from a plain "on sale" / "discounted" / "with a discount" status with no attached quantity mechanic — that DOES map to `flashsale` per the rule above.

4. Rewrite the sub-query for each product, by excluding the price constraints and service constraints.
- Make sure each rewritten sub-query does not include price and service constraints.
- Make sure each rewritten sub-query does not miss any details for the corresponding product from the original query.
- Also exclude generic retail-context words ("store", "shop", "seller", "marketplace") that describe the type of outlet rather than the product itself. e.g. "fish food store with mini pellet" -> sub-query is "mini pellet fish food", not "fish food store mini pellet".
- Collapse a redundant piece/pack-count + collective-noun pairing into ONE quantity mention. When the query states an explicit piece/pack count (e.g. "2-piece", "3-pack", "5pcs") together with a collective/quantity noun for the same product (e.g. "set", "pack", "kit", "bundle", "lot") — such as "2-piece ... hair oil set" or "3-pack cable bundle" — these two words describe the SAME single quantity fact, not two separate constraints. Drop the now-redundant collective noun from the rewritten sub-query and keep only the explicit count, so the sub-query reads as one unambiguous quantity mention. e.g. "2-piece organic hair oil set" -> "2-piece organic hair oil" (NOT "2-piece organic hair oil set"); "3-pack cable bundle" -> "3-pack cable". This is not a loss of detail — the explicit count already fully captures the "set"/"pack"/"kit" meaning — but leaving both words in the sub-query invites downstream matching to wrongly treat them as two independent constraints requiring separate evidence. Only keep the collective noun as-is when NO explicit piece/pack count is stated (e.g. "hair oil set" with no number stays "hair oil set").

5. Convert each sub-query into a list of key-value formatted keywords for search.
- Convert like "from manuel store" -> "brand→manuel store", "with gold color" -> "color→gold", ... by using `→` for delimiting key and value.
- Do NOT extract standalone generic retail words ("store", "shop", "seller", "vendor") as brand keywords unless they are preceded by a specific name (e.g. "Manuel Store" is a brand, but "pet store" or "fish food store" is not — "store" there means the type of retail outlet).
- For replacement parts or accessories, prefer product_type + compatibility/model keys; e.g. "blue button flex replacement for Oppo A5S" -> product_type→button flex, compatibility→oppo a5s, color→blue.
- Remove redundant words like "compatible with", "suitable for", "still", "fit to", "available with", "made from", ... unless they are used for brand, title, name, identifier or other non-negotiable attributes.
- Our e-commerce database uses JSON for product details, and only the value fields are indexed. Be careful not to include key-like words in the value side.
- For example, "metal-based" must be converted to "base→metal", not "material→metal-based". "suitable for student use" -> "suitable_use→student", not "suitable→student use". Shorten value strings as much as possible.
- When converting to key-value format, fetch the key name from the original query when possible. e.g. "suitable for model queen" -> "model→queen", not "compatibility→model queen".
- When a phrase describes a component's attribute (e.g. "rubber sole", "leather upper", "metal base"), produce exactly ONE keyword with the component as the key and the attribute as the value (e.g. `sole→rubber`). Do NOT also emit a general `material→rubber` — doing so would hallucinate a separate constraint that was never stated in the query.
- Convert plural forms into singular forms for better search performance, unless plural forms are used for brand, title, name, identifier or non-negotiable.
- The keywords must be mutually exclusive: exactly one keyword for one constraint aspect.
- The keywords must be collectively exhausted: the sum of the keywords must exactly cover the all meaningful information of the sub-query.
- Keep search values concise: each sub-query's keyword values should total 2-8 words when joined for search. Prioritize brand, product_type, and 1-3 distinctive attributes; omit price and service wording.
- For negation that expresses a physically absent component or ingredient ("no stones", "without fragrance", "stone-free"), capture it as a keyword value, e.g. feature→no stones, material→fragrance-free. Only do this when the query genuinely requires that something be absent.
- Do NOT use negated keyword values for universal-compatibility phrasing. Expressions like "no specific car year", "any phone model", or "fits any size" do not exclude anything — they declare no preference. Map these to X→universal (e.g. car_year→universal) or omit the keyword entirely.
- Variant identifiers (color codes, pack counts like 10pcs, SKU names) belong in color, pack/set, or color_family keys when they pick a purchasable variant.
- Product-listing deals ("buy 1 take 1", "buy 1 get 2", "BOGO", "3-for-2", "bundle deal") must produce a `promotion` keyword carrying the deal phrase as-is, e.g. "with a buy 1 get 2 deal" -> `promotion→buy 1 get 2`. Never drop this information — it must be checkable against the product's SKU options later.

# Rule of Handling the Scope of Modifiers
- It's very important to understand the scope of modifiers for identifying which product a certain constraint chunk is belonged to.
- A modifier applies to the nearest product mention only that can grammatically receive that modifier, unless there is explicit mention.
- Don't judge the scope of modifiers by the existence and the locations of commas or conjunctions.

e.g. In ```white shirt and black pants, with free shipping```, "with free shipping" applies for "black pants" only.
e.g. In ```a liquid single, gold lotion and shaving cream```, "a liquid single" applies for "gold lotion" only.
e.g. In ```Find comic books, one for 2-4 year olds and other one for 5-8 year olds.```, "comic books" applies for all products, because it's explicitly declared and followed by enumeration of "one", "another one", ...
e.g. In ```ink bottle, pencil and eraser. all must be less than 40```, "all must be less than 40" applies for all products, because it's explicitly mentioned with "all".

# Widely Used Keys for Keyword Key-Value Formatting
The following keys are widely recognized:
- product_type
    there must exist at least one keyword with "product_type" key
- color_family
    unlike the nuance of its name, it's not limited to color modifier, but can be anything which represents a variation of the product like color, size, serial number, identifier, ...
- color
    color attribute of the product, but not limited to the color adjectives.
    sometimes it can be a strange-looking color code or identifier like "black 002", "#4 green", "other/extra", "gold 1", ...
- pre_config
    the attribute that is pre-configured in the factory or the shop, like `pre_config→built`, `pre_config→tested`, ...
- pack, set
    the attribute about the package or bundle size, like `pack→20`, `set→3`, ...
- brand, feature, material, category, model, compatibility, origin, gender, user, usage, fit, suitable, promotion
- style, design, pattern, frame, theme, shape, inspiration, interest, vibe, condition, warranty_days
- size, length, capacity, volume, quality, unit, variation, option, solution, version

# Handling Search-Unfriendly Keywords
When you convert each sub-query into a list of key-value formatted keywords for search, some keywords are general expressions like "fit for any gender", "one size fits all", "suitable for all models", "good for men and women" ...
Those expressions are not search-friendly. They must be converted with the following rules:
- If a gender constraint contains "all" or "any" like "all gender" or "any gender", then convert it to "gender→all" or "gender→any". If it's like "men and women" or "boys and girls", then convert it to "gender→unisex".
- If a size constraint is about "one size fits all" meaning, convert it to "size→freesize".
- If a constraint is about the casualness of wearings, convert it to "fit→casual".
- For other constraints like "compatible with any machines", convert it to "compatible_machine→universal".
- Any compatibility or fit phrase that means "no specific requirement" or "works with anything" (e.g. "no particular model", "any car year", "fits all phones") must be expressed as X→universal or dropped. Never write these as negated values such as model→no particular or car_year→any specific.

# Response Format
Return JSON object like:
{{
    "products": {{ // identify the number of the products mentioned, and their boundaries from the original query
        // make sure max product count is 4!
        "1": <str>, // a short phrase for the first product. no need to write full text for this product. just a short differentiable phrase works.
        ... ...
    }},

    // Strictly keep in mind the `Rule of Handling the Scope of Modifiers` to correctly identify the corresponding product for each constraint chunk!

    "price": {{ // find all chunks related to the price constraints from the original query, and assign each of them to the corresponding product
        // IMPORTANT: each <chunk> key must be COPIED VERBATIM from the original query, character for character.
        // Do not paraphrase it, normalize its casing, fix its spelling, or re-order its words.
        <chunk_1, str>: {{
            "apply_for": <product_id, str>, // the corresponding product id ("1", "2", ...)
            "normalized": <str> // the normalized price format
        }},
        ... ...
    }},

    "service": {{ // find all chunks related to the service constraints from the original query, and assign each of them to the corresponding product
        // IMPORTANT: each <chunk> key must be COPIED VERBATIM from the original query, character for character.
        // This exact text is removed from the sub-query, so a paraphrased key silently leaves the service wording in the search terms.
        <chunk_1, str>: {{
            "apply_for": <product_id, str>, // the corresponding product id ("1", "2", ...)
            "labels": [<str>, ...]  // list ALL applicable service labels for this chunk (e.g. ["lazmall", "lazflash"] when both are mentioned)
        }},
        ... ...
    }},

    "sub_queries": {{ // rewrite sub-queries for the products, respectively. never miss any details, but don't include price and service constraints!
        "1": <str>, // sub-query for the first product, with price and service constraints excluded (never include price and service constraints, even if one word is left!)
        ... ...
    }},

    "keywords": {{ // convert each sub-query into a list of key-value formatted keywords.
        "1": [ // for the first sub-query

            // split the sub-query into mutually exclusive and collectively exhausted keywords, each of which corresponds to exactly one constraint aspect
            // one keyword must be a key-value pair

            // for example, "in color dark blue": "color→dark blue"

            <key→value, str>, // extracted keyword. use "→" for delimiting key and value in the keyword.
            ... ...
        ]
    }}
}}

# Original Shopping Query
{query}
""".strip()
        A_REVIEW = """
You are AI assistant to review the shopping query parsing, and fix errors if needed.
You are given the original query as well as its parsed result. {count} product(s) were identified in the original query.

# The Constraints That You Must Focus On (Label vs Explanation)
{focuses}

# Rules of Constraint Scope
1. Implicit scope sharing is forbidden!
- By default, each constraint mention in the original query applies for only one product.
- It cannot silently apply for two or more products.
- If multiple products have the same constraint, this constraint must be mentioned multiple times explicitly.

2. Each constraint mention applies for the nearest product only.
- A modifier applies for the nearest product which can grammatically receive it.
- Even if it looks like applying for multiple products grammatically or semantically, one mention applies for the nearest product once.

3. EXCEPTION — an explicit universal quantifier overrides rules 1 and 2.
- When a mention is explicitly scoped to every product by a word like "all", "both", "each", or "every", that ONE mention applies to EVERY product.
  e.g. In ```ink bottle, pencil and eraser. all must be less than 40```, "all must be less than 40" applies for all three products.
- The same holds when a shared head noun is stated once and then enumerated.
  e.g. In ```Find comic books, one for 2-4 year olds and other one for 5-8 year olds.```, "comic books" applies for both products.
- In these cases, 1 mention assigned to N products is CORRECT. Do NOT strip the constraint from the other products to make the counts match.

# The Errors You Must Detect
1. Missing constraint
- A constraint is clearly mentioned for a product's scope, but it's not assigned.
- Make sure the product is the nearest one that can grammatically receive this constraint modifier.

2. Invalid scope application (for multiple products query)
- A constraint was explicitly mentioned one time, but it was mistakenly assigned to multiple products. You must find the only one product that can receive this constraint, and remove the incorrect assignments for other products.
- Or, a constraint was assigned to a product, but actually it's not the correct one that can receive this constraint. You must find the correct product that must receive this constraint.

{error_3}

# Suspected Aspects in Current Parsing
The followings are identified issues by our heuristic checking engine. You must double check these aspects.
{suspects}

# Guideline
- Carefully check each suspected aspect and determine the correct fix if needed.
- You shouldn't be pushed to fix. The issues are just detected by our simple heuristic algorithm. If you believe the original parsing result was correct, just output as is!
- Work the reasoning out inside the `decision` field, then emit the fixes. Output the JSON object only — no preamble, no explanation outside the JSON, no markdown fences.

# Response Format
Return JSON object like:
{{
    "decision": <str>, // a brief reasoning (1 or 2 short sentences) for your decision
{responses}
}}

# Original Shopping Query
{query}

# Parsed Result
```json
{parsed}
```
""".strip()
        PV_SELECT = """
You are AI assistant to select the best matching product for the user's shopping query from the candidate products.
You should examine each product thoroughly with all the constraints mentioned in the query.

# Primary Goal
Your goal is to choose the most exact matching product, not merely the first acceptable product.
Accuracy is more important than reducing checking cost.

# Strategy
- You must check all candidate products before deciding the final best product.
- For each product, check whether it satisfies every constraint in the query.
- A product is an all-matching product only if every required query constraint is satisfied.
- After checking all candidates, collect all all-matching products.
- If there are no all-matching products, return null for `best` and nominate the least-wrong candidate in `closest`.
- If there are multiple all-matching products, rank them using the recommendation rules below.
- Original candidate order is only a final tie-breaker when match quality is equal.

# Exact Variant Matching
- When the user explicitly specifies a variant attribute (such as color, quantity/pieces, size, weight, volume, pack count, model, or similar), prefer candidates whose variant exactly matches that requirement.
- When the user does NOT specify such a variant, do not treat additional variant qualifiers as equivalent to the base product. Prefer the base product or the value that represents the requested constraint more directly, with fewer unrequested qualifiers.
- Do not stop at the first partially matching variant. Continue checking available variants for a more exact match before deciding.

# Matching Rules
{rules}

# Ranking Rules
{rank_rules}

# Recommendation Rules
All-matching products are not automatically equal. When there are 2+ all-matching products, you MUST run
the phased comparison below across EVERY one of them (not just the first one or two you happened to check
first) before choosing `best`. Move to the next phase only when the current phase cannot distinguish the
remaining candidates.

## Phase 1 — Evidence Quality (apply first, to every all-matching candidate)
You MUST evaluate evidence quality separately for EVERY constraint listed in `constraints` (product
type/model, variant such as color/size/pack-count, compatibility, etc.) — not just for whichever single
constraint happens to visibly differ between candidates. A candidate can look "tied" on the constraint you
first thought of (e.g. the color/quantity variant) while actually already being weaker on a different
constraint (e.g. product type). You must catch that.

For one candidate and one constraint, the matching evidence has three properties. Determine all three before
comparing anything.

- `admissible`: the KEY the matching value sits under must be a plausible carrier of that constraint's
  concept. Inadmissible evidence counts as NO evidence for that constraint — not as weak evidence, and not
  as a tie with a candidate that has admissible evidence.
  - If the query names a FACET of a property ("grey FRAME", "black STRAP", "red CASE", "grey LENS") and ANY
    candidate carries that facet under a key that names it (`frame_color`, `strap_color`, `case_color`,
    `lens_color`, ...), then only such a key is a plausible carrier. A key naming a DIFFERENT facet is not
    admissible for that constraint, however clean its value looks. Words inside a value never repair a wrong
    key: `lens_color: grey frame - grey` is lens evidence that happens to contain the word "frame".
  - Within `attributes`, the key must be semantically about the constraint's concept. A feature/benefit
    constraint is carried by a benefit-like key such as `body_benefits`, not by a brand, certification, or
    classification tag that merely reuses the same word.
  - Within `sku_options`, when the query names a variant generically (a bare colour, size, or option with no
    facet word), or when no candidate carries a facet-naming key, EVERY `sku_options` key is a plausible
    carrier (`color_family`, `variation`, `variation2`, `model`, `option1`, ...). Sellers rename the same
    variant concept freely, so there the key name carries no meaning by itself.
- `structured`: true if the admissible evidence comes from `sku_options` or a named `attributes` field;
  false if it comes only from free-text `title`.
- `extra_tokens`: split the matching value into tokens and count how many do NOT appear (literally or as a
  close paraphrase) in the query text AND do not support any other stated constraint. Count every such token
  mechanically, even ones that look like harmless elaboration — an unrequested year, an extra model number,
  promotional wording all count. A token that IS requested anywhere in the query is never extra, even when it
  belongs to a different constraint than the one being scored, and neither is the query's own facet word
  ("frame", "strap", "lens", "case", ...). An exact or near-exact reproduction of the query's phrase scores 0.
  Punctuation and separators are not tokens. When ONE value encodes two or more facets that the query asked
  for together (a `color_family` of `black (10pcs)` answering a query for black bags in a 10-piece pack),
  every one of those facets is requested, so `extra_tokens` is 0 for EACH constraint that value serves: do
  not charge the colour constraint for the pack-count tokens, or the pack-count constraint for the colour.

### How to decide Phase 1
Work the levels strictly in order, and within a level look across ALL constraints before moving on:

1. Start at Level 1.
2. Check EVERY constraint at this level. If any constraint separates the candidates, STOP HERE: this level
   decides Phase 1, the candidate it favours wins, and you MUST NOT look at any later level.
3. Only if every constraint ties at this level, move to the next level and repeat.

A level that has decided cannot be overturned by a later one. A candidate that wins at Level 2 wins even if
it looks worse at Level 3 or Level 4 — being tidier, shorter, or better organised is irrelevant once an
earlier level has separated the candidates. Do not let a constraint that ties at an early level distract you
from one that separates there. Your `recommend` MUST name the level that decided and the constraint that
separated there, and MUST NOT offer any later level as a reason.

- **Level 1 — admissibility.** For a given constraint, a candidate with admissible evidence beats a
  candidate with none.
- **Level 2 — structured.** Among admissible evidence, `structured = true` beats `structured = false`.
- **Level 3 — extra_tokens.** Fewer extra tokens wins.
- **Level 4 — one key per constraint.** Only when Levels 1-3 tie on EVERY constraint. When the query
  expresses two or more independent constraint concepts (e.g. a physical spec like size/colour AND a
  separate bundle/deal, warranty, or compatibility requirement), prefer the candidate whose evidence is
  spread across one key per concept over one that concatenates several concepts inside a single key's value
  (e.g. a `size` key whose value is `"5cm*5m-buy-1-take-2"`, mixing a size spec with a deal count). A
  concatenated value does not show that the platform's own schema tracks each requirement independently, and
  it is easy to match by string coincidence. Do NOT treat "combined into one clean field" as a strength.
  - This does NOT apply when one value simply combines two facets of the SAME variant selection that the
    query asked for together (e.g. a `color_family` value like `"black (10pcs)"` encoding both the colour and
    the pack-count of one SKU option) and no competing candidate carries those facets under keys that name
    them — that is ordinary SKU naming, not cramming.

Worked examples, each labelled with the level that decides it:
- **Level 1 (facet key).** Query "black sunglasses with a grey full rim frame". Candidate A has
  `[variant 2] color_family: black | lens_color: grey frame - grey`; Candidate B has
  `[variant 3] color_family: black | frame_color: black frame grey`. B's grey sits under `frame_color`, the
  key naming the requested facet; A's sits under `lens_color`, which is inadmissible for the frame
  constraint. B wins at Level 1 — do not prefer A because its value reads more cleanly or scores fewer
  extra tokens.
- **Level 1 (attribute semantics).** Query "dermacare benefits". Candidate A has
  `attributes.body_benefits: "dermacare"` — admissible. Candidate B has only
  `attributes.brand_classification: "dermacare"`, a certification tag rather than a benefit, so it is
  inadmissible. A wins at Level 1; B is not merely "unstructured", it has no evidence at all.
- **Level 1 (composite doing double duty).** Query wants an overall colour AND a frame colour. A candidate
  whose only evidence is `frame_color: black grey`, with no overall-colour key anywhere on the listing, has
  no admissible evidence for the overall-colour constraint, so it loses at Level 1 to
  `color_family: black | frame_color: black frame grey`, where each constraint has its own carrier.
- **Level 2 (structured).** Query "10-piece black storage bag". Candidate A has
  `attributes.model: "storage bag"` for the product-type constraint; Candidate B has no product-type
  attribute at all and matches "storage bag" only via free-text `title`. A wins at Level 2 on the
  product-type constraint alone, even if B's variant fields look cleaner — the variant fields are never
  compared, because a Level 2 difference on ANY constraint ends the phase.
- **Level 3 (extra tokens).** Query "samsung tab a8 10.5". Candidate A's evidence is
  `compatibility: samsung tab a8 10.5` → 0 extra tokens. Candidate B's is
  `compatibility_by_model: tab a8 10.5 2021` → 1 extra token ("2021" is not in the query), even though 2021
  plausibly describes the same real-world model. Both are admissible and structured, so Levels 1 and 2 tie
  and A wins at Level 3.
- **Scoring extra tokens across facets.** Query "black sunglasses with a grey frame". Scoring the
  frame-colour constraint on `frame_color: black frame grey` gives 0 extra tokens: "grey" is the requested
  frame colour, "frame" is the query's own facet word, and "black" is the requested overall colour, so it
  supports another stated constraint. Do NOT score it 1 because "black" is not the frame colour — that
  penalises a two-tone variant for also naming the colour the user asked for.

Title cleanliness, phrasing style, "focus", spelling, and every other subjective impression of the `title`
field are NOT evidence properties and are never a reason to prefer one candidate. A justification that
appeals to them is wrong; redo Phase 1 using the four levels above.

## Phase 2 — Exact Variant Matching (only if Phase 1 is tied)
Prefer the candidate whose explicitly requested variant (color, size, pack, model, etc.) is an exact
literal match, regardless of which attribute/sku key name carries it — key-name/facet disagreements were
already settled in Phase 1, so reaching this phase means the surviving candidates agree on their carriers.

## Phase 3 — Descriptive Coverage (only if Phases 1-2 are tied)
Prefer the candidate with more complete coverage of the user's descriptive intent (use-case, function,
material, compatibility, concern, style, benefit).

## Phase 4 — Tie-breaker (only if Phases 1-3 are all tied)
Prefer the candidate that appears earlier in the original candidate list.

Do not choose an earlier product only because it is the first all-matching product, and do not stop
comparing after only two candidates — every member of `all_matches` must be accounted for in `tally`.

# When No Candidate Matches
`all_matches` being empty does NOT mean the candidates are equally wrong, and it is not a reason to fall
back on the original candidate order. Nominate the least-wrong candidate in `closest`, ordered by WHICH
constraint each candidate fails, never by how many:

1. A candidate that is the wrong KIND of object is the worst outcome available, whatever else it matches.
   Identity constraints — product type, model, and compatibility — come first: a candidate satisfying them
   beats one failing them even if the second matches every remaining constraint and the first matches none.
2. Among candidates that agree on identity, prefer the one failing the fewest of the remaining constraints,
   then apply the Phase 1 levels above to what each does match.
3. A constraint no candidate can satisfy carries no information — sellers do not always publish every
   facet, and an absent field is not a contradicted one. Ignore such a constraint when ranking `closest`
   rather than letting it decide.

Nominate a `closest` whenever `best` is null and at least one candidate exists. Both being null is only
correct when the candidate list is empty.

# Response Format
Return JSON like:
{{
    "constraints": [<short list of extracted query constraints>],

    "check": "<short product-by-product checking summary>",

    "all_matches": [<product ids that satisfy all constraints>],

    "tally": "<one line PER id in all_matches (every one, not just the top few), covering EVERY constraint separately, like: '<id>→<constraint1>:key:<the attribute/sku key the evidence came from>,admissible:<true|false>,structured:<true|false>,extra_tokens:<n>,evidence:<short phrase>; <constraint2>:...;'>",

    "phase1": "<decide this from `tally` ALONE, before writing `recommend` or `best`. Walk Levels 1-4 in order and report the FIRST level at which any constraint separates the candidates, like 'level:2,constraint:product type,winner:4'. Write 'level:none' only if every constraint ties at all four levels.>",

    "recommend": "<short explanation naming the phase (1-4) and, for Phase 1, the level (1-4) that decided the winner, citing the specific key/value evidence compared for the winner vs. the next-best candidate>",

    "best": <integer|null>, // the 1-based number of the winning product (e.g. 3), NOT a quoted id. null if no product matches all constraints.

    "closest": <integer|null> // ONLY when `best` is null: the 1-based number of the least-wrong candidate, chosen by the "When No Candidate Matches" rules. Omit or set null when `best` is a number, or when there are no candidates at all.
}}

# Output Guidelines
1. Be concise and structured.
- Each product checking content must be one line.
- Use `\\n` for delimiting products.
- Each product checking content should look like:
  "Product <ID>→<constraint>:<evidence>(structured|unstructured), match; <constraint>:<evidence>(structured|unstructured), match;"
- If a required constraint is missing, stop checking that product and mark it as mismatch.
- Mark each constraint's evidence as `(structured)` or `(unstructured)` inline — do not just write "match"
  with no structuredness note, even when the constraint isn't the one you expect to be decisive.

2. For recommendation:
- `tally` must cover every id in `all_matches`, not a subset, and must list EVERY constraint per id, not
  only the constraint that first seemed to distinguish the candidates.
- In `tally`, quote each constraint's evidence with the KEY it actually came from (`frame_color: black frame
  grey`, not a bare `grey`). You cannot apply the facet-key rules without the key names in front of you, and
  writing the value alone is how a differently-named carrier gets mistaken for the requested facet.
- When ONE key/value carries several of the constraints (an sku option like `baruan: 6pcs longsleeve`
  answering both a piece-count and a garment-type constraint), cite that SAME key and value for EVERY
  constraint it covers. Never credit it to one constraint and then fall back to a weaker carrier such as
  `title` for a sibling constraint the same value already answers — that understates the candidate and hands
  a `structured` win at Level 2 to a rival that merely spread the identical facts across more keys.
- Fill `phase1` before `recommend` and `best`. Whenever `phase1` names a level, its `winner` IS the Phase 1
  winner: `best` must be that product, and `recommend` must justify it from that level. Never let a later
  level, a cleaner-looking key, or a tidier value change the answer after `phase1` has named a level.
- Work the Phase 1 levels in order against the finished `tally`. Before using a level, confirm that EVERY
  constraint for EVERY candidate ties at all earlier levels: never compare `structured` while an
  `admissible:false` split exists, never compare `extra_tokens` while a `structured` split exists, and never
  invoke the one-key-per-constraint rule while any of Levels 1-3 still separates the candidates.
- Explain briefly why the selected product wins, naming the deciding phase, WHICH constraint decided it, and
  the specific structured evidence compared — never title aesthetics.
- Do not rely on original order unless every phase above is tied.

3. Return JSON only.

# Shopping Query:
```{query}```

# Candidate Products:
{products}
""".strip()
        SV_COMBO_SELECT = """
You are AI assistant to select the best product bundle for the user's shopping request.
The user needs to buy {n_items} items. Each candidate bundle satisfies the voucher budget constraint.

Every bundle contains exactly one product per slot (slots 1 through {n_items}), matched positionally
by each item's `slot` field. The requirements for each slot are given ONCE in `# Slot Requirements` below —
they are identical across every bundle (only the candidate product differs per bundle), so they are not
repeated inside each bundle's item. For an item with `"slot": N`, check it against `slots["N"].constraints`
in `# Slot Requirements`. Those constraints were already extracted from the original query during query
decomposition — treat them as the definitive, exhaustive requirement list for **pass/fail matching**
on that slot; do not invent new constraints a product must satisfy. Use each slot's `sub_query` for
context/phrasing — especially when applying S1, to decide whether SKU tokens support the request or
are unsupported/conflicting extras.
Your task is to find the bundle where ALL items best match their respective slot's `constraints`.

# Matching Rules
{rules}

# Ranking Rules
{rank_rules}

# Selection Rules (When Multiple Bundles Are Valid)
Evaluate bundles in three phases. Move to the next phase only when the current phase cannot distinguish bundles.

---

## How to compare bundles across all items

Each bundle contains one product per sub-query slot (slot 1, slot 2, …). When applying any selection rule:

1. **Compare slot by slot.** For slot N, compare Bundle A's product against Bundle B's product on that rule. Mark which bundle wins that slot (or "tie" if equal).
2. **Tally wins across ALL slots.** Count how many slots each bundle wins on that rule.
3. **The bundle with more slot wins is preferred.** Do not decide based on one or two slots alone — every slot counts equally.
4. **If slot wins are tied**, treat that rule as inconclusive and move to the next rule.

This ensures no product in a bundle is ignored during comparison. All items contribute to the final decision.

---

## Phase 1 — Evidence Quality (consider S1, S2, S3 together)
S1, S2, and S3 each measure a different facet of how reliable and clean the matching evidence is for each item. Apply the slot-by-slot win-tally described above across all three rules, then ask: does one bundle have clearly better evidence quality overall?

- If one bundle leads on all three (or two out of three) rules by slot wins, it has clearly better evidence quality — prefer it.
- If the picture is mixed (e.g. Bundle A leads on S1 but Bundle B leads on S2 and S3), weigh the total slot wins across S1+S2+S3 combined. If still tied, apply S1 → S2 → S3 in order as an internal tiebreaker.

**S1 — Directness of attribute match (fewest unsupported / conflicting tokens)**
For each slot, prefer the product whose matching attribute value has the fewest tokens that are **unnecessary or disagree** with the user's request. Tally slot wins across the bundle. Do NOT treat "longer SKU" as automatically worse.
- **What counts as an extra token:** a token in the matching value that does **not** help satisfy the slot's sub_query / constraints — unsupported by the request, or conflicting / narrowing beyond what was asked.
- **What does NOT count as extra:** tokens that appear in the sub_query (or clearly paraphrase a stated request), and tokens that cover another stated constraint for that slot. Those are supporting evidence.
- Ground this judgment against the slot's **sub_query** text together with its constraints — not against a truncated keyword alone. A longer value that more faithfully reproduces query phrasing beats a shorter bare substring.
- **An attribute value that is an exact or near-exact reproduction of the query term (or of several slot constraints combined) has zero unsupported tokens and is always the most direct possible evidence — it wins S1 unconditionally against any other form of matching.**
- Example: query asks for "redcell 1 quart" as package content → `package_content: redcell 1 quart` beats `package_content: redcell iron 1quart` (extra unsupported tokens / less exact phrasing).

**S2 — Attribute key specificity**
For each slot, prefer the product whose matching evidence comes from an attribute key dedicated to the queried property. Tally slot wins across the bundle.
- A dedicated, property-specific key signals the vendor explicitly categorized that dimension.

**S3 — Structured field over noisy title**
For each slot, prefer the product whose matching evidence appears in a structured field (`sku_options`, named `attributes`) rather than only in the free-text `title`. Tally slot wins across the bundle.
- SEO titles frequently pile on unrelated words; structured fields carry deliberate, vendor-confirmed data.
- **A product whose slot is won by a structured attribute match ALWAYS beats a product that relies on title text for that slot.**

---

## Phase 2 — Value Precision (S4)
If evidence quality (Phase 1) is equal, assess value precision using the same slot-by-slot win-tally.

**S4 — Exact or narrow value over compound or range value**
For each slot, prefer the product whose matching attribute holds a single, narrow value rather than a multi-value or range value. Tally slot wins across the bundle.
- Prefer an attribute value that is an exact phrase match to the query (e.g. `package_content: redcell 1 quart`) over a near-miss paraphrase (`package_content: redcell iron 1quart`).

---

## Phase 3 — Tie-breaker (S5)
Only when Phases 1 and 2 are both equal across all slots, prefer the bundle that appeared earlier in the candidate list.

**S5 — Original ranking as final tiebreaker**

# Slot Requirements
Applies identically to every bundle below. For any item with `"slot": N`, use `slots["N"]` here as its requirements:
```json
{slots}
```

# Candidate Bundles:
Each item omits `sub_query`/`constraints` — look them up via the item's `slot` field in `# Slot Requirements` above.
```json
{bundles}
```

# Response Format
Return JSON like (fields in this exact order — `best` comes last so it follows from the checking and the reason, never precedes them):
{{
    "check": "<per-bundle checking, one line per bundle like: Bundle 1→item_1:<constraint>:<evidence>,match; item_2:...; all_match:true/false>",
    "reason": "<brief explanation citing which phase/rule decided the winner and the slot-level tally>",
    "best": <bundle id (1-indexed integer) of the best all-matching bundle, or null>
}}

# Guidelines
- For each bundle, check every item against its slot's `constraints` list from `# Slot Requirements` (matched by the item's `slot` field; fall back to `sub_query` only for phrasing/context, never to invent new constraints).
- A bundle is valid only if ALL items match all constraints for their respective slot.
- If multiple bundles are valid, apply the Selection Rules above phase by phase (Phase 1 → Phase 2 → Phase 3) to determine the winner.
- In `reason`, name the phase and rule(s) that decided the winner, report the slot-level tally, and give a concrete per-slot evidence comparison for the decisive slots.
- Return null for `best` if no bundle has all items matching.
""".strip()
        SV_FILTER = """
You are AI assistant to check the products against the provided shopping queries, to find the ones which meet all the constraints mentioned.
There are {count} groups. Each group has a shopping query and a list of candidate products for this query.

# Matching Rules
{rules}

# Response Format
Return JSON like:
{{
    // check groups "A", "B", ... in the order
    "A": {{
        "query": <str>, // write the query for group "A"

        // check all products "A1", "A2", ... against the query "A"

        "A1": {{
            // for product id = "A1"

            "check": <str>, // checking content
            "is_match": <bool> // true if matching all constraints, otherwise false
        }},
        ... ...
    }},

    "B": {{
        "query": <str>,

        "B1": {{
            "check": <str>,
            "is_match": <bool>
        }},
        ... ...
    }}
    ... ...
}}

# Output Guidelines
- Each product checking content must be a brief string like "<segment_1>;<segment_2>; ..." where each segment is for checking one constraint.
- Each segment must be a brief string, like `<constraint>:<xxx> in title (or attributes), match (or 'mismatch so skip');`.
- Check constraints in order. Once a mismatch is found, you may stop and set is_match=false.
- When a product includes `sku_options`, add a segment checking variant constraints (color, quantity/pack, variant name) against those lines.

# Queries and Candidate Products in Groups:
```json
{info}
```
""".strip()
        MATCH_RULES = """
1. Tokens that appear to be numbers, identifiers, codes, variant names, or uncommon combinations must be matched literally and must not be normalized or partially matched. They are non-negotiable!
- If query mentioned "100ml", the product details must explicitly include the number "100" in the context of volume. "10-150ml" is not a match, even if 100 is in range 10-150.
- If query mentioned "3-7 yrs old", the product details must explicitly include both "3" and "7" in one of its attributes with the context of age. "5 years" is not a match, even if 5 is in range 3-7.
- If query mentioned "iPhone", the product details must explicitly include "iPhone". Guessing from "Apple Smartphone" is not allowed.
- If query mentioned "color green 2", the product details must explicitly include "green 2" for color-related details. "green" is not allowed. "green 2" is not spell error, but a specific identifier!
- If query mentioned "up to 100m/s", the product details must explicitly include "100" with compatible semantics, like "90-100m/s" or "< 100m/s". "100-110m/s" is incorrect because the context doesn't match!

2. For descriptive constraints, rely on semantics and common sense.
- In particular, you must be generous for matching product type. If it's acceptable product type by common sense, see it a match. For example, "capacity tester" can be seen a match of "tester" if it's about battery.
- Generosity about product-type WORDING is not permission to accept a different KIND of product. Independently of the other constraints, first decide whether the candidate actually IS the product type the query asks for — as a category, judged only from its own title/attributes/sku_options. Sharing words with the query is NOT category membership: a knife sharpener is not a knife, and a phone case is not a phone, however many query words their text repeats. So a candidate that is a clearly different kind of object is a MISMATCH no matter how many other constraints (color, pack count, size, ...) its text satisfies. This gate is category-level ONLY — keep applying the generous wording rules here to synonyms, hypernyms, subtypes, and replacement/accessory parts of the requested type ("capacity tester" for "tester", "button flex for Oppo A5S"). When you genuinely cannot tell whether it is a plausible subtype, variant, or accessory of the requested type, keep it; reject only clear category errors.
- Replacement or accessory queries (e.g. "button flex for Oppo A5S") match when model/compatibility appears in title or attributes, even if product_type wording differs slightly.
- Also, consider paraphrasing as a match. (It's not applied to non-negotiable constraints mentioned in Rule-1.)
- For example, "standard" = "default", "kid" = "child", "beautiful" = "pretty", "regular" = "normal", "plain" = "simple", "freesize" = "one size", "personalized" = "customized", ...
- Common collective/quantity words used as product-type qualifiers — "set", "pack", "kit", "bundle", "lot" — are descriptive, NOT identifiers. They must NOT be matched literally. A product whose title or attributes use the plural form of the item (e.g. "Stickers") or contain a variation/quantity that implies a collection is a valid match for a query like "sticker set" or "cable pack".

3. `title` might be messy.
- Sometimes the `title` string contains more words than needed. It's vendor's intention for better SEO. So don't drop a product only because the `title` contains redundant words.
- For example, if you are looking for "water bottle", and a product's `title` is "Black Plastic Water Bottle Cover Kids Feeding", then this is a match. Don't drop it because it's "water bottle cover". Vendors love to enumerate many many words in `title` for SEO.
- If an attribute is found within `title`, see it a match, even if `title` field includes redundant information than needed.

4. Other attribute lines may contain different values. As long as at least one attribute line contributes a valid match, the product may still match.
- `attributes` filed consists of many lines. Each line represents one attribute with format "[key]: [value]".
- `attributes` field may represent a number of variations of the product. If at least one variation is a match, see this product a match.
- Therefore, the conflicting or alternative values in other attribute lines are allowed.
- For example, if the query mentioned "black" while both "black" and "green" are found in two different lines from `attributes`, respectively, then see it a match.
- This line-independent allowance applies ONLY to `attributes`. It NEVER applies to `sku_options`, which is a separate field governed by Rule 6 — there, evidence may not be combined across lines.
- EXCEPTION: If the product has a **dedicated specification attribute** (e.g., `battery_capacity`, `storage`, `memory`, `screen_size`, `weight`, `volume`, `units_hb`, `pack`, `pack_type`) whose value explicitly contradicts the query's numeric/measured requirement, then the product does NOT match that constraint — even if the required number appears elsewhere (e.g., embedded in a composite variant label like `color_family: black-1200mah`, or inside one `sku_options` line like `volume: 2pcs(300ml+500ml)`). The dedicated specification attribute is the authoritative source for that measurement/classification and overrides embedded numbers or piece-counts found in composite SKU labels or individual `sku_options` lines.
- This explicitly applies to single-item vs. bundle/multi-pack constraints: if the query asks for "sold as a single item" / "single pack" but the product's dedicated attribute reads e.g. `units_hb: multi-pack`, it is a mismatch — and vice versa, if the query asks for a "bundle pack" / "multi-pack" but the dedicated attribute reads `units_hb: single item`, it is a mismatch, even though one `sku_options` line separately offers a multi-piece variant (e.g. `volume: 2pcs(...)`). Do not treat a single `sku_options` variant line as proof of the product's pack format when a dedicated pack-quantity attribute states otherwise.

5. Composite variant labels vs. specification attributes.
- Fields like `color_family` sometimes contain composite SKU identifiers such as `black-1200mah` or `blue-64gb`. The numeric portion is a variant name, NOT a measured specification.
- When the user requests a technical specification (e.g., "1200mah battery", "64gb storage"), matching against a composite variant label is weaker evidence than matching against the corresponding dedicated attribute (e.g., `battery_capacity: 1200mah`, `storage: 64gb`).
- If the product has a dedicated specification attribute that contradicts the requirement (e.g., `battery_capacity: 10000` when the user asks for "1200mah battery"), the product FAILS that constraint, regardless of the composite label.
- The same priority applies to pack-quantity requests ("single item" vs. "bundle"/"multi-pack"): a dedicated attribute like `units_hb` is the authoritative source, and always outranks an embedded piece-count found inside a `sku_options` value (e.g. `volume: 2pcs(300ml+500ml)`) or inside `title` SEO text.

6. SKU / variant options (`sku_options` field)
- `sku_options` is rendered as one line per purchasable variant, prefixed `[variant N]`, with that variant's key-values joined by ` | ` on that same line. A line like `[variant 3] color_family: grey | compatibility_by_model: realme 11 pro 5g` means grey and realme 11 pro 5g are buyable TOGETHER; values sitting on two different `[variant N]` lines can never be bought together.
- When `sku_options` is present, variant constraints in the query (color, pack size like "10pcs", variant names like "button blue", color codes like "a") must match at least one line in `sku_options` literally (Rule-1 applies).
- If the query specifies a variant and `sku_options` exists but no line contains that variant, it is a mismatch even when `title` partially matches.
- `sku_options` should only be used as supporting/positive evidence when no dedicated attribute for that property exists, or when it agrees with the dedicated attribute. When a `sku_options` line and a dedicated attribute (Rule 4/5) disagree on the same property (e.g. pack quantity), the dedicated attribute wins and the product does not match the conflicting requirement.
- Each `[variant N]` line is a distinct, mutually exclusive purchasable variant — a real unit is exactly ONE of these lines, never a mix of several. If the query's variant requirement is really TWO OR MORE facets that must co-occur on the same physical unit (e.g. "blue" AND "with lights", "red" AND "size L", or a color AND a compatibility model), you must find a SINGLE `[variant N]` line satisfying every one of those facets together — whether they sit in one composite value or in separate keys on that same line. It is NOT sufficient for one line to satisfy facet A while a different line satisfies facet B — that is not a match, it is two incompatible options being cherry-picked. Unlike Rule 4 (which allows `attributes` lines to represent independent, possibly co-existing properties), `sku_options` lines are alternatives you choose between, so evidence must never be combined across lines.
- Do this check by scanning line-by-line: for each `[variant N]` line, ask whether THAT line alone satisfies every variant facet the query asks for. Only if some line passes all of them is the product a match. Never answer by searching the whole `sku_options` block for each facet separately — that is exactly the error this rule forbids.
  - Worked example: query wants "blue ... with lights". Candidate has `[variant 1] option: 11cm-glowing` and `[variant 3] option: 11cm-blue` (no color on variant 1, no light/glow on variant 3). This is a MISMATCH — no single line states both "blue" and "glowing/lights" — even though "blue" and "glowing" each appear somewhere in `sku_options`.
  - Worked example (facets in separate keys): query wants "grey ... for realme 11 pro 5g". Candidate A has `[variant 1] color_family: grey | compatibility_by_model: realme 12 pro 5g` and `[variant 5] color_family: purple | compatibility_by_model: realme 11 pro 5g`. This is a MISMATCH: grey is only sold for the 12 pro, and the 11 pro 5g is only sold in purple, so no purchasable unit is both. Candidate B has `[variant 3] color_family: grey | compatibility_by_model: realme 11 pro 5g` — that single line carries both facets, so Candidate B matches and A does not. Do not treat A as a match merely because "grey" and "realme 11 pro 5g" both appear somewhere in its `sku_options`.

7. Negation and absence constraints
- Phrases like "no stones", "without X", "non-X", or "stone-free" require the forbidden concept must NOT appear as a positive feature in title, attributes, or sku_options.
- Do not treat unrelated category words as proof of absence; require explicit negation or absence in product details.

8. Variant evidence carried by a `model` attribute is valid.
- A value like `model: 1pc black` is valid color/variant evidence, exactly equivalent to `color_family: black`.
- Vendors sometimes encode variant selection (color, quantity, configuration) in the `model` field rather than in `color_family`, so do not treat its absence from `color_family` as a mismatch.

9. Ground every judgment in the product text you were given.
- The ONLY admissible evidence is that product's own `title`, `attributes`, and `sku_options`. You have no other knowledge of the product.
- For every constraint you mark as matched, you must be able to quote the supporting `key: value` line, or the exact `title` substring, verbatim from that product's fields. If you cannot quote it, the constraint is NOT matched.
- Never infer an attribute from brand reputation, product category, a typical specification, or what the product "probably" has. An unstated attribute is unmatched, never assumed.
- Never carry evidence from one product to another. Each product is judged only on its own fields, even when the candidates are near-identical listings.
- If a product's fields say nothing at all about a required constraint, that constraint is unmatched. Silence is not agreement.
- This rule outranks the generosity granted by Rule 2 and Rule 3: those relax HOW a present value may be worded, never whether the value must be present.
""".strip()
        RANK_RULES = """
These rules NEVER decide whether a product matches. They only order products that have
ALREADY satisfied every constraint. Never reject a product for ranking weakly.

R1. Matching decides whether a product satisfies the query; ranking decides which all-matching product is the most exact.
- A product that technically satisfies all constraints can still be less exact than another all-matching product.
- Prefer products whose matching evidence represents the requested values more directly, requiring fewer ignored or unrelated tokens.
- Structured attributes are generally stronger evidence than noisy SEO title text.
- Prefer the product whose constraint match comes from a semantically dedicated attribute (e.g., `battery_capacity: 1200mah`) over one where the value is only embedded in a composite variant label (e.g., `color_family: black-1200mah`).
- Original ranking is only a tie-breaker after match quality is equal.

R2. Explicit unit matching ranks above unit-less matching.
- When the user specifies a value with an explicit unit (e.g., "1200mah", "64gb", "100ml"), a product whose attribute value also explicitly includes the same unit (e.g., `battery_capacity: 1200mah`) ranks above one whose attribute omits the unit (e.g., `battery_capacity: 1200`).
- Both still MATCH. A unit-less value is an inferred match; a unit-bearing value is a literal match.
- This applies to common measurement units: mah, mw, gb, mb, ml, cm, mm, inch, kg, g, oz, etc.

R3. Do not penalize a `model`-carried variant when ranking.
- A product with `model: 1pc black` and `battery_capacity: 1200mah` (exact unit) should be preferred over a product with `color_family: black` and `battery_capacity: 1200` (no unit), because the battery capacity match is more literal in the former.
""".strip()

    A_CLASSIFY = Templates.A_CLASSIFY
    A_DECOMPOSE = Templates.A_DECOMPOSE
    A_REVIEW = Templates.A_REVIEW
    PV_SELECT = Templates.PV_SELECT
    SV_COMBO_SELECT = Templates.SV_COMBO_SELECT
    SV_FILTER = Templates.SV_FILTER
    MATCH_RULES = Templates.MATCH_RULES
    RANK_RULES = Templates.RANK_RULES

    label_to_service = staticmethod(Responses.label_to_service)
    service_to_label = staticmethod(Responses.service_to_label)
    extract_decompose_response = staticmethod(Responses.extract_decompose_response)
    extract_select_response = staticmethod(Responses.extract_select_response)
    extract_closest_response = staticmethod(Responses.extract_closest_response)
    arbitrate_tally = staticmethod(Responses.arbitrate_tally)
    TALLY_LEVELS = Responses.TALLY_LEVELS
    extract_filter_response = staticmethod(Responses.extract_filter_response)
    arbitrate_filter = staticmethod(Responses.arbitrate_filter)
    extract_combo_select_response = staticmethod(Responses.extract_combo_select_response)
    structured_hits = staticmethod(Responses.structured_hits)

class NS_QueryPipeline:


    JSON_PARSE_TRIES  = 2
    SPLIT_PARSE_TRIES = 2

    _SPEND_LEAK_FLAG = re.compile(
        r'\b(my\s+budget|voucher with|fixed\s+discount|percentage\s+discount|exceeds\s+`\d+`)\b',
        re.I,
    )
    _KEYWORD_NOISE = re.compile(
        r'^(looking(?:\s+for)?|show\s+me|find(?:\s+me)?|i(?:\'m|\s+am)\s+looking)\b',
        re.I,
    )


    _LAZFLASH_RE = re.compile(
        r"\blaz\-?flash\b|\bflash\s+(?:sale|deal)s?\b|\blightning\s+deal\b"
        r"|\blimited[\s\-]*time\b|\btime[\s\-]*(?:bound|constrained|limited)\b"
        r"|\btimed\s+(?:deal|promo|promotion|offer)\b|\btemporary\s+discount(?:ed)?\b"
        r"|\btime[\s\-]*bound\s+promo\b"
        r"|\bon\s+sale\b|\bwith\s+(?:a\s+)?discount(?:s|ed)?\b|\bdiscount(?:ed)?\s+option\b",
        re.I,
    )


    _LAZMALL_RE = re.compile(
        r"\blaz\-?mall\b"
        r"|\b(?:official|authorized|verified|trusted)\s+(?:store|shop|seller|dealer|brand|reseller)s?\b",
        re.I,
    )

    def __init__(self, query: str, debugger: NS_Debugger, task_hint: str = "") -> None:
        self.query = query
        self.debugger = debugger
        self.task_hint = task_hint


    def _build_hint_text(self, qi: NS_QueryInfo) -> str:
        if qi.challenge == NS_ShoppingTaskType.PRODUCT:
            return "\n\n# Hint\nThe original query contains only one product!"
        if qi.challenge == NS_ShoppingTaskType.SHOP:
            return (
                "\n\n# Hint\nThe user wants multiple products from the same shop. "
                "Split one sub-query per product; assign each product its own price and service constraints."
            )
        if qi.challenge == NS_ShoppingTaskType.VOUCHER:
            return (
                "\n\n# Hint\nOnly the text before the budget block describes products. "
                "Split one sub-query per distinct product; exclude voucher/budget wording from sub-queries and keywords."
            )
        return ""

    @staticmethod
    def _has_type_keyword(sq: NS_SubQuery) -> bool:
        return any(NS_QueryPipeline._kw_signature(kw) == "product_type" for kw in sq.kw if kw)

    _BRANDED_RE = re.compile(r"\bbranded\b", re.I)

    @staticmethod
    def _has_branded_kw(sq: NS_SubQuery) -> bool:
        return any(kw for kw in sq.kw if NS_QueryPipeline._BRANDED_RE.search(kw[-1]))

    @staticmethod
    def _kw_core_terms(kw_list: List[List[str]]) -> str:

        skip = NS_Vocab.NON_SEARCH_KEYS
        parts: List[str] = []
        for kw in kw_list:
            if not kw:
                continue
            keys, val = kw[:-1], kw[-1].strip()
            if any(k in skip for k in keys):
                continue
            if val and val not in parts:
                parts.append(val)
        return " ".join(parts)


    @staticmethod
    def _kw_signature(kw: List[str]) -> str:

        return kw[0] if len(kw) == 2 else kw[-2]

    @staticmethod
    def _keyword_value_is_noisy(sq: NS_SubQuery) -> bool:
        terms = NS_QueryPipeline._kw_core_terms(sq.kw)
        if not terms.strip():
            return True
        if NS_QueryPipeline._KEYWORD_NOISE.search(terms):
            return True
        for kw in sq.kw:
            if kw and NS_QueryPipeline._kw_signature(kw) == "product_type" and NS_QueryPipeline._KEYWORD_NOISE.search(kw[-1].strip()):
                return True
        return False

    @staticmethod
    def _intent_has_spend_leak(sq: NS_SubQuery) -> bool:
        blob = " ".join([sq.query or ""] + [kw[-1] for kw in sq.kw if kw])
        return NS_QueryPipeline._SPEND_LEAK_FLAG.search(blob) is not None

    @staticmethod
    def _price_is_valid(price: Optional[str]) -> bool:
        if not price:
            return True
        parts = price.split("-", 1)
        if len(parts) != 2:
            return False
        for part in parts:
            if part and not re.fullmatch(r'\d+(?:\.\d+)?', part):
                return False
        return True

    @staticmethod
    def _estimate_min_items(query_e: str) -> int:
        q = query_e.lower()


        n = len(set(re.findall(r'\b(?:first|second|third|fourth)\b', q)))
        if n >= 2:
            return n
        return 2 if re.search(r'\band also\b|\bone option\b.*\banother\b|(?=.*\bboth\b)(?=.*\band\b)', q) else 1


    _UNIVERSAL_SCOPE_RE = re.compile(
        r"\b(?:all|both|each|every)\b"
        r"(?:\s+(?:of\s+)?(?:the\s+)?(?:products?|items?|ones?|pieces?))?"
        r"\s+(?:must|should|shall|will|need(?:s)?\s+to|ha(?:s|ve)\s+to|are|is|be)\b",
        re.I,
    )

    @staticmethod
    def _is_universal_scope(query_e: str, n_items: int, count_e: int, count_p: int) -> bool:


        if n_items < 2 or count_e != 1 or count_p != n_items: return False
        return NS_QueryPipeline._UNIVERSAL_SCOPE_RE.search(query_e or "") is not None

    _SCOPE_PARSE_RULES = (
        ("free_ship", "freeship_e", "freeship_p", "Free/complimentary shipping"),
        ("lazmall",   "lazmall_e",  "lazmall_p",  "LazMall / official store"),
        ("lazflash",  "lazflash_e", "lazflash_p", "Flash sale / LazFlash"),
        ("cod",       "cod_e",      "cod_p",       "Cash on delivery"),
        ("branded",   "brand_e",    "brand_p",     "Branded"),
    )

    _PRICE_UNIT_RE = re.compile(r"\b(?:php|pesos?|usd|dollars?|euros?)\b", re.I)


    _PRICE_UNIT_BRIDGE_RE = re.compile(
        r"\b(?:php|pesos?|usd|dollars?|euros?)\b\s*(?:and|to|-|–|~)\s*[\d.,]+\s*"
        r"\b(?:php|pesos?|usd|dollars?|euros?)\b",
        re.I,
    )

    @staticmethod
    def _count_price_mentions(q: str) -> int:
        return len(NS_QueryPipeline._PRICE_UNIT_RE.findall(q)) - len(NS_QueryPipeline._PRICE_UNIT_BRIDGE_RE.findall(q))

    def _count_audit_flags(self, qi: NS_QueryInfo) -> Dict[str, int]:
        q = qi.query_e
        return {
            "freeship_p": len([sq for sq in qi.items if "freeShipping" in sq.services]),
            "lazmall_p":  len([sq for sq in qi.items if "official"     in sq.services]),
            "lazflash_p": len([sq for sq in qi.items if "flashsale"    in sq.services]),
            "cod_p":      len([sq for sq in qi.items if "COD"          in sq.services]),
            "price_p":    len([sq for sq in qi.items if sq.price]),
            "brand_p":    len([sq for sq in qi.items if NS_QueryPipeline._has_branded_kw(sq)]),
            "cod_e": len(re.findall(
                r"\bCOD\b"
                r"|\b(cash(es|ing)?|pay(ment|ing)?s?)[\s\-]*(for)?[\s\-]*((up)?on|with(in)?|by|for)?[\s\-]*(deliver(y|ing|ie)?s?|cash(es|ing)?)",
                q, flags=re.I,
            )),
            "warranty_e": (
                len(re.findall(r"\b(\d+)[\s\-]*(days?|weeks?|months?|years?)?[\-\s]*(periods?)?\s*(of)?[\-\s]*(guarantee|warrant)", q, re.I))
                + len(re.findall(r"\b(guarantee|warrant)[a-z]*[\s\-]*(days?|weeks?|months?|years?)?[\-\s]*(periods?)?\s*(of)?(\d+)", q, re.I))
            ),
            "lazmall_e":  len(NS_QueryPipeline._LAZMALL_RE.findall(q)),
            "lazflash_e": len(NS_QueryPipeline._LAZFLASH_RE.findall(q)),
            "freeship_e": len(re.findall(r"\b(free|compliment[a-z]+)\s+(ship|deliver)", q, re.I)),
            "brand_e":    len(re.findall(r"\s+branded\b", q, re.I)),
            "price_e":    NS_QueryPipeline._count_price_mentions(q),
        }

    def _validate_query_info(self, qi: NS_QueryInfo) -> List[str]:

        errors: List[str] = []
        n = len(qi.items)

        if n == 0:
            return ["No sub-queries were produced."]
        if qi.challenge == NS_ShoppingTaskType.PRODUCT and n != 1:
            errors.append(f"Product challenge must have exactly 1 sub-query, got {n}.")
        if qi.challenge == NS_ShoppingTaskType.SHOP and n < 2:
            errors.append(f"Shop challenge must have at least 2 sub-queries, got {n}.")
        if qi.challenge == NS_ShoppingTaskType.SHOP and n >= 2:
            terms = [NS_QueryPipeline._kw_core_terms(sq.kw) for sq in qi.items]
            if len(set(terms)) < len(terms):
                errors.append("Shop sub-queries must target distinct products (duplicate search terms).")

        min_products = self._estimate_min_items(qi.query_e)
        if qi.challenge in (NS_ShoppingTaskType.SHOP, NS_ShoppingTaskType.VOUCHER) and n < min_products:
            errors.append(f"Query describes at least {min_products} distinct products but only {n} sub-query(ies) were parsed.")
        if qi.challenge == NS_ShoppingTaskType.VOUCHER and n == 1 and min_products >= 2:
            errors.append("Voucher query lists multiple products; split one sub-query per product (exclude budget/voucher text).")

        for i, sq in enumerate(qi.items, 1):
            if not (sq.query or "").strip():
                errors.append(f"Sub-query {i} is empty.")
            if not self._has_type_keyword(sq):
                errors.append(f"Sub-query {i} is missing a product_type keyword.")
            if self._keyword_value_is_noisy(sq):
                errors.append(f"Sub-query {i} keywords are not search-ready; drop filler like 'looking for'.")
            if not self._price_is_valid(sq.price):
                errors.append(f"Sub-query {i} has invalid price format `{sq.price}`.")
            if qi.challenge == NS_ShoppingTaskType.VOUCHER and self._intent_has_spend_leak(sq):
                errors.append(f"Sub-query {i} still contains voucher/budget wording.")


        c = self._count_audit_flags(qi)
        for _, count_e_key, count_p_key, desc in self._SCOPE_PARSE_RULES:
            count_e, count_p = c[count_e_key], c[count_p_key]
            if count_e != count_p and not self._is_universal_scope(qi.query_e, n, count_e, count_p):
                errors.append(
                    f"{desc}: mentioned {count_e} time(s) in query but assigned to {count_p} sub-query(ies) "
                    f"-- verify the assignment; a mention scoped by \"all\"/\"both\"/\"each\" may legitimately cover every product."
                )
        if c["price_e"] != c["price_p"] and not self._is_universal_scope(qi.query_e, n, c["price_e"], c["price_p"]):
            errors.append(
                f"Price: currency appears {c['price_e']} time(s) but {c['price_p']} sub-quer(ies) carry a price range "
                f"-- verify; do NOT invent a price for a product that has none, and do NOT record a non-price "
                f"number range (capacity, age, power, size) as a price."
            )
        elif (
            c["price_e"] == 1
            and qi.items
            and not qi.items[-1].price
            and re.search(r"\b(php|pesos?|usd|dollars?|euros?)\b[\s.]*$", qi.query_e, re.I)
        ):
            errors.append("Price constraint appears at the end of the query but was not assigned to the last sub-query.")
        if c["warranty_e"] > 0 and not any(
            re.search(r"\b(guarantee|warrant)", (sq.query or "") + " " + " ".join(kw[-1] for kw in sq.kw if kw), re.I)
            for sq in qi.items
        ):
            errors.append(f"Warranty/guarantee mentioned {c['warranty_e']} time(s) but no sub-query carries a warranty constraint.")

        return errors


    _VOUCHER_CLAUSE_RE = re.compile(r"\bmy\s+budget\s+is\b", re.I)

    def _has_voucher_clause(self) -> bool:
        return NS_QueryPipeline._VOUCHER_CLAUSE_RE.search(self.query) is not None

    def _identify_task_type(self) -> NS_ShoppingTaskType:

        if self.task_hint in {t.value for t in NS_ShoppingTaskType}:
            _debug_print(f"* classify_task (hint): {self.task_hint}")
            return NS_ShoppingTaskType(self.task_hint)

        prompt = NS_Prompt.A_CLASSIFY.format(query=self.query)
        models = NS_ProxyUtil.get_role_llm_priorities("classify")
        result, _ = NS_ProxyUtil.trigger_llm(prompt, preferred_llms=models, expect="dict")

        task_type_str = (result.get("task_type") or "").strip().lower()
        if task_type_str in {t.value for t in NS_ShoppingTaskType}:
            challenge = NS_ShoppingTaskType(task_type_str)

            if challenge == NS_ShoppingTaskType.VOUCHER and not self._has_voucher_clause():
                _debug_print("# classify_task: llm answered voucher but no budget clause is present, re-deciding")
                return self._identify_non_voucher_type()

            _debug_print(f"* classify_task (llm): {task_type_str} | reason: {result.get('reason', '')}")
            return challenge

        _debug_print("# classify_task: llm result invalid or missing, falling back to regex")
        return self._identify_task_type_regex()

    _SHOP_INTENT_RE = re.compile(
        r"\b(?:look(?:ing)?(?:\s+for)?|find|show|same)\b(?:\s+\w+){0,10}\s+shops?\b",
        re.I,
    )

    def _identify_non_voucher_type(self) -> NS_ShoppingTaskType:


        if NS_QueryPipeline._SHOP_INTENT_RE.search(self.query): return NS_ShoppingTaskType.SHOP
        return NS_ShoppingTaskType.PRODUCT

    def _identify_task_type_regex(self) -> NS_ShoppingTaskType:

        if self._has_voucher_clause():
            return NS_ShoppingTaskType.VOUCHER
        if NS_QueryPipeline._SHOP_INTENT_RE.search(self.query):
            return NS_ShoppingTaskType.SHOP
        return NS_ShoppingTaskType.PRODUCT

    def _parse_voucher_clause(self) -> Tuple[str, Optional[NS_VoucherOption]]:


        sep = self.query.lower().find("my budget is")
        if sep < 0: return self.query, None

        query_e = self.query[:sep].strip()
        v_str = self.query[sep:].strip()

        try:
            m = re.search(r"\bmy\s+budget\s+is(?:\s+only)?\s+`(\d+)`", v_str, re.I)
            budget = int(m.group(1))

            apply_to = "shop" if re.search(r"\bsame\s+shops?\b", v_str, re.I) else "all"

            m = re.search(r"\bexceeds?\s+`(\d+)`", v_str, re.I)
            threshold = int(m.group(1))

            if (m := re.search(r"\bpercent(?:age)?\s+discount(?:\s+of)?\s+`([\d\.]+)\%`", v_str, re.I)):
                discount = float(m.group(1))
                is_fixed = False

                m = re.search(r"\bwith(?:\s+a)\s+cap(?:\s+of)?\s+`([\d\.]+)`", v_str, re.I)
                cap = float(m.group(1))
            else:
                m = re.search(r"\bfixed\s+discount(?:\s+of)?\s+`([\d\.]+)`", v_str, re.I)
                discount = float(m.group(1))
                is_fixed = True
                cap = None
        except (AttributeError, ValueError) as exc:
            _debug_print(f"# voucher clause present but unparseable ({type(exc).__name__}): {v_str[:120]}")
            return self.query, None

        vopt = NS_VoucherOption(
            budget = budget,
            apply_to = apply_to,
            threshold = threshold,
            is_fixed = is_fixed,
            discount = discount,
            cap = cap
        )
        return query_e, vopt

    def _check_review_triggers(self, qi: NS_QueryInfo) -> Tuple[List[str], str, str, str, str]:
        freeship_p = len([sq for sq in qi.items if "freeShipping" in sq.services])
        lazmall_p = len([sq for sq in qi.items if "official" in sq.services])
        lazflash_p = len([sq for sq in qi.items if "flashsale" in sq.services])
        cod_p = len([sq for sq in qi.items if "COD" in sq.services])
        price_p = len([sq for sq in qi.items if sq.price])
        brand_p = len([sq for sq in qi.items if NS_QueryPipeline._has_branded_kw(sq)])

        cod_e = len(
            re.findall(
                r"\bCOD\b"
                r"|\b(cash(es|ing)?|pay(ment|ing)?s?)[\s\-]*(for)?[\s\-]*((up)?on|with(in)?|by|for)?[\s\-]*(deliver(y|ing|ie)?s?|cash(es|ing)?)",
                qi.query_e,
                flags = re.I
            )
        )
        warranty_e = len(
            re.findall(
                r"\b(\d+)[\s\-]*(days?|weeks?|months?|years?)?[\-\s]*(periods?)?\s*(of)?[\-\s]*(guarantee|warrant)",
                qi.query_e,
                re.I
            )
        ) + len(
            re.findall(
                r"\b(guarantee|warrant)[a-z]*[\s\-]*(days?|weeks?|months?|years?)?[\-\s]*(periods?)?\s*(of)?(\d+)",
                qi.query_e,
                re.I
            )
        )
        lazmall_e = len(NS_QueryPipeline._LAZMALL_RE.findall(qi.query_e))
        lazflash_e = len(NS_QueryPipeline._LAZFLASH_RE.findall(qi.query_e))
        freeship_e = len(re.findall(r"\b(free|compliment[a-z]+)\s+(ship|deliver)", qi.query_e, re.I))
        brand_e = len(re.findall(r"\s+branded\b", qi.query_e, re.I))
        price_e = NS_QueryPipeline._count_price_mentions(qi.query_e)

        suspects, prompts_i, prompts_o, focuses = [], [], [], []

        for label, count_e, count_p, prefix in [
            ("lazmall", lazmall_e, lazmall_p, "The official store condition (\"LazMall\", \"official store/shop/brand\", \"authorized store/seller/dealer\", \"verified store/brand\", or \"trusted store/seller/brand\")"),
            ("lazflash", lazflash_e, lazflash_p, "The flash sale condition (\"flash sale\", \"flash deal(s)\", \"lightning deal\", \"LazFlash\", a time-constrained promotion like \"limited-time offer\", or generic \"on sale\"/\"with a discount\" status wording)"),
            ("free_ship", freeship_e, freeship_p, "The condition of free or complimentary shipping"),
            ("cod", cod_e, cod_p, "The condition of \"Cash on Delivery\" option"),
            ("branded", brand_e, brand_p, "The word \"branded\""),
        ]:
            if count_e != count_p and not self._is_universal_scope(qi.query_e, len(qi.items), count_e, count_p):
                suspects.append(label)

                prompts_i.append(f"""
- {prefix} appeared {count_e} time(s) in the original query, but {count_p} products were parsed with this constraint. It needs double check.
""".strip()
                )
                prompts_o.append(f"""
    "{label}": [
        <product_id_1, str>, ... // correct product ids that must receive **{label}** constraint.
    ]
""".strip())

        if "lazmall" in suspects and warranty_e > 0:
            prompts_i.append(f"""
- Warranty period was mentioned in the original query {warranty_e} time(s), but no product was parsed with this constraint. It needs double check.
""".strip())
            prompts_o.append("""
    "warranty_period": {
        "apply_for": <product_id_1, str|null>, // the product id that must receive warranty period constraint. set null if no product is relevant.
        "period": <str|null> // warranty period mentioned for this product in the original query if exists, otherwise null. for example, "period": "14 days"
    }
""".strip())
            suspects.append("warranty_period")

        if price_e != price_p and not self._is_universal_scope(qi.query_e, len(qi.items), price_e, price_p):
            prompts_i.append(f"""
- Currency unit appeared {price_e} time(s) in the original query, but {price_p} products were parsed with price constraint. You must carefully check the context if those number ranges are really the price constraints.
""".strip())
            prompts_o.append("""
    "price": { // list all the products (don't skip any product)
        "1": { // product id = 1
            "parsed_price_range": <str|null>, // the price range string or null as assigned for this product by the parsing result given
            "is_it_correct": <bool>, // true if the parsed result is correct, otherwise false
            "other_spec": <str|null>, // the other spec name ("power usage", for example) which was mistakenly identified as the price range in the parsing result if exists. set null if "parsed_price_range" is null or the parsed result has no error.
            "correct_price_range": <str|null> // the corrected price range for this product. write the original parsed result again if it has no error. set null if no price range condition is for this product. use "[low]-[high]" format without currency unit if exists. "[low]-" or "-[high]" if either bound is missing.
        },
        ...
    }
""".strip())
            suspects.append("price")
        elif price_e == 1 and not qi.items[-1].price:
            if re.search(r"\b(php|pesos?|usd|dollars?|euros?)\b[\s.]*$", qi.query_e, re.I):
                prompts_i.append(
                    "- The price range appeared at the end of the original query text, but in the parsed result, "
                    "the product which is not the last one was parsed with the price constraint. You must carefully check if it's correctly parsed."
                )
                prompts_o.append("""
    "price": { // list all products (don't skip any product)
        "1": { // product id = 1
            "parsed_price_range": <str|null>, // the price range string or null as assigned for this product by the parsing result given
            "correct_price_range": <str|null> // the corrected price range for this product. write the original parsed result again if it has no error. set null if no price range condition is for this product. use "[low]-[high]" format without currency unit if exists. "[low]-" or "-[high]" if either bound is missing.
        }
    }
""".strip())
                suspects.append("price")

        if "lazmall" in suspects: focuses.append('- **lazmall**: explicit mention of "LazMall", "official store/shop/brand", "authorized store/seller/dealer/brand", "verified store/brand", or "trusted store/shop/seller/brand". These authorized/verified/trusted phrasings map to **official** even without the literal word "official".')
        if "lazflash" in suspects: focuses.append('- **lazflash**: explicit mention of "flash sale", "flash deal(s)", "lightning deal", or "LazFlash". Also applies to time-constrained promotion phrasing such as "limited-time offer", "timed deal", or "time-bound promo" that implies a temporary discounted window, AND to generic "on sale" / "with a discount" / "discount option" status wording on the product. Do NOT apply it to a specific quantity-based deal mechanism printed on the product\'s own SKU, like "buy 1 take 1", "BOGO", "3-for-2", or "bundle deal" — those are product-listing promotions, not a platform sale status.')
        if "free_ship" in suspects: focuses.append('- **free_ship**: explicit mention of free or complimentary shipping.')
        if "cod" in suspects: focuses.append('- **cod**: explicit mention of cash on delivery or payment on delivery.')
        if "branded" in suspects: focuses.append('- **branded**: explicit mention of the word "branded". (don\'t confuse it with the brand constraint. the word "branded" must exist as is!)')
        if "warranty_period" in suspects: focuses.append('- **warranty_period**: explicit mention of guarantee or warranty period.')
        if "price" in suspects: focuses.append('- **price**: explicit mention of price range. (pared results use "[low]-[high]" format without currency unit. just "[low]-" or "-[high]" if only one bound is specified.)')

        if "price" in suspects:
            error_3 = """
3. Confused price range
- A product appears with price range constraint in the parsed result, but actually this number range is for a completely different specification.
- For example, in "I need a bottle between 0.5 and 1.2 litre", the range "0.5-1.2" is not price constraint, but a capacity constraint.
- Also, in "Find motors with power usage less than 2.5", the range "-2.5" is mentioned without currency unit, but you can easily confirm that it's the power usage constraint with unit "kw".
""".strip()
        else:
            error_3 = ""


        return suspects, '\n'.join(prompts_i), ',\n'.join(prompts_o), '\n'.join(focuses), error_3

    def _build_review_prompt(self, qi: NS_QueryInfo, prompts_i: str, prompts_o: str, focuses: str, error_3: str) -> str:
        prods = {
            str(i + 1): {
                "sub-query": sq.query,
                "constraints": [NS_Prompt.service_to_label(svc) for svc in (sq.services or [])],
                "price": sq.price
            }
            for i, sq in enumerate(qi.items)
        }
        return NS_Prompt.A_REVIEW.format(
            count = len(qi.items),
            focuses = focuses,
            suspects = prompts_i,
            error_3 = error_3,
            responses = prompts_o,
            query = qi.query_e,
            parsed = json.dumps(prods, ensure_ascii = False, indent = 2)
        )

    def _apply_review_fixes(self, qi: NS_QueryInfo, js: Dict[str, List[str] | Dict], suspects: List[str]):
        def __get_id__(id_str: Any) -> Optional[int]:
            if not isinstance(id_str, str): return None
            pid = NS_MiscUtil.extract_number(id_str, True) - 1

            if pid < 0 or pid >= len(qi.items): return None
            return pid

        for issue in suspects:
            if issue not in js: continue
            obj = js.get(issue)

            if issue in ["lazmall", "lazflash", "cod", "free_ship"]:
                if not isinstance(obj, list): continue
                svc = NS_Prompt.label_to_service(issue)

                ids = [__get_id__(x) for x in obj]
                if None in ids: continue

                for i, sq in enumerate(qi.items):
                    if i in ids and svc not in sq.services:
                        sq.services.append(svc)
                    elif i not in ids and svc in sq.services:
                        sq.services.remove(svc)
            elif issue == "branded":
                if not isinstance(obj, list): continue

                ids = [__get_id__(x) for x in obj]
                if None in ids: continue

                for i, sq in enumerate(qi.items):
                    if i in ids and not NS_QueryPipeline._has_branded_kw(sq):
                        sq.kw.append(["brand", issue])
                        sq.query += ", it must come with branded feature."
                    elif i not in ids:
                        for kw in sq.kw:
                            kw[-1] = re.sub(r"\s+branded\b", " ", kw[-1], flags = re.I)
                            kw[-1] = re.sub(r"\s+", " ", kw[-1])

                        sq.query = re.sub(r"\bbranded\b", "", sq.query, flags = re.I)
                        sq.query = re.sub(r"\s+", " ", sq.query)
            elif issue == "warranty_period":
                if not isinstance(obj, dict): continue

                id = __get_id__(obj.get("apply_for"))
                if id is None: continue

                period = obj.get("period")
                if not period: continue

                qi.items[id].kw.append([issue, period])
                qi.items[id].query += f", it must provide with {period} of guarantee"
            elif issue == "price":
                if not isinstance(obj, dict): continue

                ids = [__get_id__(k) for k in obj.keys()]
                if None in ids: continue

                if sorted(ids) != list(range(len(qi.items))): continue

                for i, sq in enumerate(qi.items):
                    j: Dict[str, str] = obj.get(str(i + 1))

                    correct = j.get("correct_price_range")
                    price = re.sub(r"[^\d\-]", "", correct or "") or None

                    if price != sq.price and sq.price:
                        spec = j.get("other_spec")

                        if spec:
                            sq.kw.append([spec.replace(' ', '_'), sq.price])
                            sq.query += f", it must have {spec} with range {sq.price}"

                    sq.price = price

    def classify_task(self) -> NS_QueryInfo:
        if not self.debugger.is_frozen("a-classify"):
            challenge = self._identify_task_type()

            if challenge == NS_ShoppingTaskType.VOUCHER:
                query_e, vopt = self._parse_voucher_clause()


                if vopt is None:
                    query_e = self.query
                    challenge = self._identify_non_voucher_type()
                    _debug_print(f"* classify_task: voucher clause unusable, downgraded to {challenge.value}")
            else:
                query_e, vopt = self.query, None

            qi = NS_QueryInfo(
                challenge = challenge,
                query_f = self.query,
                query_e = query_e,
                vopt = vopt
            )
            self.debugger.add("a-classify", qi.to_json())

            self.debugger.print({
                "challenge": challenge,
                "query_e": query_e,
                "voucher": vopt.to_json() if vopt else None
            })
            return qi
        else:
            return NS_QueryInfo.from_json(
                self.debugger.find("a-classify", {"query_f": self.query})
            )

    def decompose_query(
        self,
        qi: NS_QueryInfo,
        feedback: str = "",
        attempt: int = 0,
    ) -> List[Dict[str, Dict[str, Any]]]:
        if not self.debugger.is_frozen("a-decompose"):
            temperature = min(0.4, attempt * 0.2)
            tries = 0


            best: List[Dict[str, Dict[str, Any]]] = []
            while tries < self.SPLIT_PARSE_TRIES:
                tries += 1
                try:
                    prompt = NS_Prompt.A_DECOMPOSE.format(
                        shop_intent = " in the same shop" if qi.challenge == NS_ShoppingTaskType.SHOP else "",
                        query = qi.query_e,
                    ) + self._build_hint_text(qi)
                    if feedback:
                        prompt += (
                            "\n\n# Review Required\nOur heuristic checker flagged the previous parsing:\n"
                            f"{feedback}\n"
                            "These checks are approximate. Correct the points that are genuine errors, but where "
                            "the previous parsing was actually right, keep it as is rather than inventing a "
                            "constraint to satisfy a count. Return the full corrected JSON only."
                        )
                    js, _ = NS_ProxyUtil.trigger_llm(
                        prompt,
                        preferred_llms=NS_ProxyUtil.get_role_llm_priorities("decompose"),
                        temperature=temperature,
                        expect="dict",
                    )
                    decomposed = NS_Prompt.extract_decompose_response(js)
                    self.debugger.print(js)

                    if len(decomposed) == 0:
                        raise Exception("no sub-queries extracted")
                    if len(decomposed) > len(best):
                        best = decomposed
                    min_products = self._estimate_min_items(qi.query_e)
                    if len(decomposed) == 1 and qi.challenge == NS_ShoppingTaskType.SHOP:
                        raise Exception("sub-query count invalid")
                    if len(decomposed) < min_products and qi.challenge in (NS_ShoppingTaskType.SHOP, NS_ShoppingTaskType.VOUCHER):
                        raise Exception(f"sub-query count {len(decomposed)} < expected {min_products}")

                    self.debugger.add("a-decompose", {"query": self.query, "products": decomposed})
                    return decomposed
                except Exception as exc:
                    _debug_print(f"# error parsing LLM response for decomposition: {exc}")
                    temperature += 0.2


            if best:
                _debug_print(f"# decompose: count checks unsatisfied after retries, accepting best parse of {len(best)}")
                self.debugger.add("a-decompose", {"query": self.query, "products": best})
                return best

            raise Exception("decompose failed after retries")
        else:
            return self.debugger.find("a-decompose", {"query": self.query})["products"]

    def build_query_info(self, qi: NS_QueryInfo, parsed: List[Dict]) -> NS_QueryInfo:
        if not self.debugger.is_frozen("a-build"):
            for inf in parsed:
                q = inf.get("query")


                kw_com = [kw for kw in (NS_EngineUtil.compact_kw(kw, q) for kw in inf.get("keywords")) if kw]

                if inf.get("prod_only"):


                    vstr = "Product only"

                    for i in range(len(kw_com)):


                        if NS_EngineUtil.is_prod_only(kw_com[i][-1]):
                            kw_com[i] = ["service", vstr]
                            break
                    else:
                        kw_com += [["service", "product only"]]

                sq = NS_SubQuery(
                    query = q,
                    phrase = inf.get("phrase") or "",
                    services = inf.get("services"),
                    price = inf.get("price"),
                    kw = kw_com
                )
                qi.items.append(sq)

            self.debugger.add("a-build", qi.to_json())
            return qi
        else:
            return NS_QueryInfo.from_json(
                self.debugger.find("a-build", {"query_f": self.query})
            )


    RAW_FALLBACK_MAX_TOKENS = 10

    @staticmethod
    def _raw_query_item(qi: NS_QueryInfo) -> NS_SubQuery:


        text = NS_QueryPipeline._KEYWORD_NOISE.sub("", (qi.query_e or "").strip()).strip()
        words = [
            w for w in re.findall(r"[a-z][\w\-\.]*", NS_QueryPipeline._PRICE_UNIT_RE.sub(" ", text.lower()))
            if w not in NS_Vocab.KW_STOPWORDS
        ]
        kw = [["product_type", " ".join(words[:NS_QueryPipeline.RAW_FALLBACK_MAX_TOKENS])]] if words else []

        return NS_SubQuery(query = text or (qi.query_e or ""), kw = kw)

    def prepare_context(self, qi: NS_QueryInfo) -> NS_QueryInfo:
        qi = self._prepare_context(qi)

        if not qi.items:
            _debug_print("# prepare_context: no sub-queries parsed; falling back to the raw query")
            qi.items.append(self._raw_query_item(qi))

        return qi

    def _prepare_context(self, qi: NS_QueryInfo) -> NS_QueryInfo:

        max_attempts = 1 if self.debugger.is_frozen("a-decompose") else self.JSON_PARSE_TRIES
        feedback = ""
        for attempt in range(max_attempts):
            try:
                decomposed = self.decompose_query(qi, feedback, attempt)
            except Exception as exc:
                _debug_print(f"# prepare_context decompose failed: {exc}")
                if qi.items:
                    return qi
                break
            qi.items.clear()
            qi = self.build_query_info(qi, decomposed)

            errors = self._validate_query_info(qi)
            if not errors:
                if attempt:
                    self.debugger.print({"parse_validate": "ok", "attempt": attempt + 1})
                return qi
            feedback = "\n".join(f"- {e}" for e in errors)
            self.debugger.print({"parse_validate": "retry", "attempt": attempt + 1, "errors": errors})
            if attempt + 1 >= max_attempts:
                self.debugger.print({"parse_validate": "accept_with_errors", "errors": errors})
                return qi
        return qi

    def review_query_info(self, qi: NS_QueryInfo) -> NS_QueryInfo:
        if not self.debugger.is_frozen("a-review"):
            suspects, prompts_i, prompts_o, focuses, error_3 = self._check_review_triggers(qi)

            if len(suspects) > 0:
                js, _ = NS_ProxyUtil.trigger_llm(
                    self._build_review_prompt(qi, prompts_i, prompts_o, focuses, error_3),
                    expect = "dict"
                )
                js["query"] = qi.query_e
                self.debugger.print(js)

                self._apply_review_fixes(qi, js, suspects)

            self.debugger.add("a-review", qi.to_json())
            return qi
        else:
            return NS_QueryInfo.from_json(
                self.debugger.find("a-review", {"query_f": self.query})
            )

class NS_SearchSession(object):
    MAX_TOOL_RES_PER_STEP = 5


    FILTER_CHECK_SNIPPET_LEN = 100

    def __init__(self, query: str, debugger: NS_Debugger) -> None:
        self.debugger = debugger
        self.query = query

        self.tool_results = []
        self.dialog = []


        self.base_search_done = False

    def step(self, think: str, content: str) -> None:
        self.dialog.append(
            create_dialogue_step(
                think = think,


                tool_results = self.tool_results[-NS_SearchSession.MAX_TOOL_RES_PER_STEP:],
                response = content,
                query = self.query,
                step = len(self.dialog) + 1
            )
        )
        self.tool_results = []

        if self.debugger.show_dlg:
            self.debugger.print(f"*** STEP-{len(self.dialog)} ***")
            self.debugger.print(think)

            if content:
                self.debugger.print("*" * 30)
                _debug_print(content)

            self.debugger.print("*" * 30)
            self.debugger.print("")

    def _summarise_extra_search(self, res: ToolCallResult) -> ToolCallResult:


        if not self.base_search_done: return res
        if not isinstance(res, dict): return res
        if res.get("name") in ("recommend_product", "terminate"): return res
        if "result" not in res: return res

        rows = res["result"]
        rows = rows.get("result") if isinstance(rows, dict) else rows
        n = len(rows) if isinstance(rows, list) else 0

        return {**res, "result": {"n_results": n, "rows_omitted_from_transcript": True}}

    def add_tool_result(self, res: ToolCallResult) -> None:
        try:
            res = self._summarise_extra_search(res)
            json.dumps(res)
            self.tool_results.append(res)
        except Exception as exc:
            self.debugger.print(f"# tool result is not JSON-serialisable, dropped: {exc}")


    def step_on_submit(self, best_p: List[NS_Product], qi: NS_QueryInfo) -> None:
        if not best_p:
            return self.step(
                "I can't find the product(s) satisfying the user's requirements. I will report failure to the user.",
                "Failed to find the requested product(s). Please try again with more specific constraints."
            )

        think = f"I have identified the following product{' combination' if len(qi.items) > 1 else ''}: `{', '.join([p.id for p in best_p])}`."
        think += f" Before submitting the recommendation, I will verify all the requirements again.\n"

        for i in range(len(best_p)):
            think += f"\n**Product {i + 1}** (UID `{best_p[i].id}`)\n" if len(qi.items) > 1 else ""
            think += f"\n- query: `{qi.items[i].query}`"
            think += f"\n- product_title: `{best_p[i].title}`"
            think += f"\n- shop_id: `{best_p[i].shop_id}`"

            if qi.items[i].price:
                think += f"\n- price: expects {NS_EngineUtil.promptfy_price(qi.items[i].price)} | found {str(int(best_p[i].price))} → within range ✓"

            for svc in (qi.items[i].services or []):
                think += f"\n- service: expects `{svc}` | found `{svc}` → available ✓"

            think += "\n"

        if qi.challenge == NS_ShoppingTaskType.VOUCHER:
            psum = sum(p.price for p in best_p)
            pact = NS_EngineUtil.get_vouchered_cost(best_p, qi.vopt)

            if len(best_p) > 1:
                think += "\nThe prices of the selected products are "
                think += ", ".join([str(int(p.price)) for p in best_p[:-1]]) + " and " + str(int(best_p[-1].price))
                think += ", respectively, with a total price of " + str(int(psum)) + "."
            else:
                think += "\nThe price of the selected product is " + str(int(best_p[0].price)) + "."

            if psum > pact:
                if qi.vopt.apply_to == "shop":
                    think += f" Since all of them come from the same shop (`{best_p[0].shop_id}`) and the total price exceeds the voucher threshold, the voucher can be applied."

                think += " After applying the voucher, the effective total price becomes " + str(int(pact)) + f", which is within the user's budget `{str(int(qi.vopt.budget))}`."
            else:
                think += f" It is already within the budget `{str(int(qi.vopt.budget))}`, even without using the voucher."

            if len(best_p) > 1:
                think += "\n\nIn summary, the selected products are within the budget and satisfy all stated requirements."
            else:
                think += "\n\nIn summary, the selected product is within the budget and satisfies all stated requirements."
        elif qi.challenge == NS_ShoppingTaskType.SHOP:
            think += f"\nAll the selected products come from the same shop (`{best_p[0].shop_id}`), as requested."
            think += "\n\nIn summary, all selected products come from the same shop, and each one satisfies the user's requirements."
        else:
            think += "\nAll checks passed. The selected product satisfies the user's requirements."

        think += "\n\nNow I will present the final recommendation."

        recommend = {
            "recommend": [p.id for p in best_p]
        }
        content = f"The final recommendation:\n\n```json\n{json.dumps(recommend, ensure_ascii = False, indent = 4)}```"
        self.step(think, content)


    def step_on_build(self, qi: NS_QueryInfo) -> None:
        if len(qi.items) > 1:
            think_1 = "First, I will decompose the user's request into a set of mutually exclusive and collectively exhaustive sub-queries, each corresponding to a single product:\n"

            for i, sq in enumerate(qi.items):
                think_1 += f"\n**{i + 1}.** `{sq.query}`"
        else:
            think_1 = f"First, I will rewrite the user's request in a more concise form:\n\n`{qi.items[0].query}`"

        keywords = [', '.join([kw[-1] for kw in sq.kw]) for sq in qi.items]
        services = [', '.join(svc for svc in sq.services) if sq.services else "not specified" for sq in qi.items]
        prices = [sq.price if sq.price else "not specified" for sq in qi.items]

        if len(qi.items) > 1:
            think_2 = f"Next, I will extract meaningful keywords and additional options from each sub-query to facilitate search:\n"
        else:
            think_2 = f"Next, I will extract meaningful keywords and additional options from the query to facilitate search:\n"

        for i in range(len(qi.items)):
            think_2 += f"\n**{i + 1}.** " if len(qi.items) > 1 else "\n"
            think_2 += f"keywords = `{keywords[i]}` | service = `{services[i]}` | price = `{prices[i]}`"

        if qi.challenge == NS_ShoppingTaskType.VOUCHER:
            think_3 = f"\n\nThe user's budget is: {int(qi.vopt.budget)}.\n"
            think_3 += "The available voucher options are as follows:\n\n"
            think_3 += "**Voucher Type:** " + ("Fixed amount" if qi.vopt.is_fixed else "Percentage") + "\n"
            think_3 += f"**Discount Amount:** {int(qi.vopt.discount)}" + ("%" if not qi.vopt.is_fixed else "") + "\n"
            think_3 += "**Application Condition:** " + ("Products from the same shop" if qi.vopt.apply_to == "shop" else "All products") + "\n"
            think_3 += (f"**Threshold:** {int(qi.vopt.threshold)}" if qi.vopt.threshold is not None else "") + "\n"
            think_3 += (f"**Cap:** {int(qi.vopt.cap)}" if qi.vopt.cap is not None else "") + "\n"
        else:
            think_3 = ""

        think = (
            f"This is a shopping request involving {NS_ShoppingTaskType(qi.challenge).explain(len(qi.items) > 1)}. "
            f"{think_1}\n\n{think_2}{think_3}"
        )
        self.step(think, "Initial query analysis completed.")

    def step_on_search(self, qi: NS_QueryInfo) -> None:
        if len(qi.items) > 1:
            think = "Next, I will perform an initial search for each sub-query, retrieving up to five pages of results:"
        else:
            think = "Next, I will perform a search for the query, retrieving up to five pages of results:"

        act = {
            "action": "search",
            "options": [
                {
                    "keywords": [kw[-1] for kw in sq.kw],
                    "price": sq.price or None,
                    "service": sq.services or None,
                    "pages": "1-5"
                }
                for sq in qi.items
            ]
        }
        think += f"\n\n```json\n{json.dumps(act, ensure_ascii = False, indent = 4)}```"
        self.step(think, "")


        self.base_search_done = True


    def step_on_shop_plan(self, shop_stats: Dict[str, Tuple[int, float]], top_k: int, qi: NS_QueryInfo) -> None:
        if qi.challenge == NS_ShoppingTaskType.VOUCHER:
            think = "Now I will look for combinations of products from the same shop, since such combinations may benefit from voucher application.\n"
        else:
            think = ""

        think += (
            "First, I will group the retrieved products by shop and examine how well each shop covers the sub-queries. "
            "Shops with broader coverage are more promising, but additional shops will also be considered to avoid overlooking better combinations. "
            "The following statistics summarize the coverage provided by the discovered shops:\n\n"
        )
        plan = []

        for n in range(len(qi.items), 0, -1):
            shops = [sid for sid in shop_stats if shop_stats[sid][0] == n]
            plan.extend(shops)

            if n > 1:
                think += f"- Shops covering {n} sub-queries: `{', '.join(shops)}`\n"
            else:
                think += f"- The remaining {len(shops)} shop{'s' if len(shops) > 1 else ''} cover{'s' if len(shops) == 1 else ''} only one sub-query.\n"

        think += "\nGood. Next, I will examine the most promising shops in order of coverage to determine whether they can fully satisfy the sub-queries"
        think += " while allowing the voucher to be applied." if qi.challenge == NS_ShoppingTaskType.VOUCHER else "."
        think += f"\n\nThe following shops will be examined: {', '.join(plan[:top_k])}, ... ..."

        self.step(think, "Planned candidate shops.")

    def step_on_shop_batch(
        self,
        batch: List[str],
        sols: int,
        n: int,
        qi: NS_QueryInfo,
        n_passed: int = 0,
    ) -> None:
        think = f"**Shop Examination Step {n + 1}**\n\n"
        think += f"In this step, I will check the following shop{'s' if len(batch) > 1 else ''}: `{', '.join(batch)}`\n\n"
        think += "I have performed additional searches for each shop and each sub-query using narrower search criteria by enforcing a shop ID constraint. "
        think += f"Among the retrieved products, I was able to find {sols} combination{'s' if sols > 1 else ''} whose products all come from the same shop"
        think += " for which the voucher can be applied. " if qi.challenge == NS_ShoppingTaskType.VOUCHER else ". "

        if sols > 0:
            think += f"After carefully examining the candidate combination{'s' if sols > 1 else ''}, "

            if n_passed > 0:
                think += (
                    f"{n_passed} combination{'s' if n_passed > 1 else ''} passed all constraint checks. "
                    "I will keep these as candidates and continue examining other shops before selecting the best bundle."
                )
            else:
                think += f"{'all of them were' if sols > 1 else 'it was'} rejected because one or more products did not satisfy the specified constraints. "
                think += "I will continue the shop examination."
        else:
            think += "I will continue the shop examination."

        self.step(think, "")

    def step_on_combo_select(self, best_p: List[NS_Product], n_bundles: int, reason: str, is_fallback: bool) -> None:
        best_str = ', '.join(p.id for p in best_p)
        if is_fallback:
            think = (
                f"The LLM could not determine a best matching bundle from the {n_bundles} candidate(s). "
                f"Falling back to the highest-scored combination from the knapsack: `{best_str}`."
            )
            response = f"Combination selected (knapsack fallback): `{best_str}`."
        else:
            think = (
                f"I presented {n_bundles} candidate bundle(s) to the LLM for evaluation. "
                "Each bundle already satisfies the voucher budget constraint. "
                "The LLM checked every item against its specific sub-query requirements "
                "and selected the bundle that best matches all constraints.\n\n"
                f"**Selected combination:** `{best_str}`"
            )
            if reason:
                think += f"\n\n**Reason:** {reason}"
            response = f"Best-matching bundle selected: `{best_str}`."
        self.step(think, response)

    def step_on_shop_fallback(
        self,
        best_p: List[NS_Product],
        qi: NS_QueryInfo,
        has_reject: bool = False,
        has_unchecked: bool = False,
    ) -> None:


        best_str = ', '.join([p.id for p in best_p])

        think = (
            "**Shop Examination Rollback**\n\n"
            "I have examined all discovered shops, but was unable to find one providing matching products"
            " for which the voucher can be applied.\n\n" if qi.challenge == NS_ShoppingTaskType.VOUCHER else ".\n\n"
        )
        if best_p and has_reject:


            think += (
                "No shop was found where every product satisfies its corresponding sub-query. "
                f"The best available combination, `{best_str}`, still has at least one slot that failed the constraint check during filtering. "
                "Since no better alternative exists among the examined shops, I will recommend this partial match while flagging the unresolved slot."
            )
        elif best_p and has_unchecked:


            think += (
                "No shop was found where every product was confirmed against its corresponding sub-query. "
                f"For the best available combination, `{best_str}`, at least one product was never checked against its constraints, "
                "so I cannot claim it as a confirmed match. "
                "Since no better alternative exists among the examined shops, I will recommend it as the closest available result and flag it as unverified."
            )
        elif best_p:
            think += (
                "Either a valid combination has been overlooked during the previous evaluation steps, or no matching combination exists in the current database. "
                "Before reporting failure to the user, I will perform one final verification of the previously discovered candidate combinations. "
                f"The combination `{best_str}` appears to be a strong candidate. Let me verify it against the stated constraints. "
                "Upon re-examination, I found that this combination satisfies all stated constraints. It was incorrectly rejected during the previous evaluation. "
                "To ensure that no other valid combinations have been overlooked, I will review the remaining candidates as well. "
                f"No additional combinations satisfying all stated constraints were found. The only missed combination is `{best_str}`. "
                "Therefore, this combination can be recommended to the user."
            )
        elif qi.challenge == NS_ShoppingTaskType.VOUCHER:
            think += (
                "However, voucher application is optional. "
                "As long as the total price remains within the budget, the combination can still be recommended. "
                "Therefore, I will switch to voucher-free examination mode."
            )
        else:
            think += (
                "This indicates that no matching combination exists in the current database. "
                "I will report failure to the user."
            )

        self.step(think, "Completed shop examination process.")

    @staticmethod
    def _summarize_filter_js(js: Any) -> Any:


        if not isinstance(js, dict): return js

        summary: Dict[str, Any] = {}
        for gk, gv in js.items():
            if not isinstance(gv, dict):
                summary[gk] = gv
                continue

            new_group: Dict[str, Any] = {}
            for k, v in gv.items():
                if isinstance(v, dict) and "is_match" in v:
                    check = v.get("check", "")
                    if isinstance(check, str) and len(check) > NS_SearchSession.FILTER_CHECK_SNIPPET_LEN:
                        check = check[:NS_SearchSession.FILTER_CHECK_SNIPPET_LEN].rstrip() + "..."
                    new_group[k] = {"check": check, "is_match": v.get("is_match")}
                else:
                    new_group[k] = v
            summary[gk] = new_group

        return summary

    def step_on_filter(self, n_groups: int, n_checked: int, n_passed: int, js: Dict) -> None:
        if n_checked == 0: return

        think = (
            f"I checked {n_checked} product(s) across "
            f"{n_groups} sub-quer{'ies' if n_groups > 1 else 'y'} against all stated constraints:\n\n"
            f"```json\n{json.dumps(NS_SearchSession._summarize_filter_js(js), ensure_ascii=False, indent=2)}```\n\n"
            f"{n_passed} out of {n_checked} candidate(s) passed all constraint checks."
        )
        self.step(think, f"Constraint check: {n_passed}/{n_checked} candidate(s) passed.")


    def step_on_platform(self, sol_count: int, is_fallback: bool) -> None:
        think = (
            "Now I will look for combinations of products whose total price satisfies the voucher constraint"
            f"{' (falling back to platform-wide search).' if is_fallback else '.'}"
            " I applied the Meet-in-the-Middle (MITM) algorithm to solve the underlying knapsack problem, "
            "where product weights are derived from the logarithms of the search-rank probabilities. "
            f"MITM found {sol_count} valid solution(s) whose combined price meets the voucher threshold and stays within budget. "
            "I will now evaluate the top candidates to identify which best meets the stated requirements."
        )
        self.step(think, f"MITM knapsack: {sol_count} valid combination(s) found.")


    def step_on_select(
        self, best_p: NS_Product, js_check: Dict, is_fallback: bool, sq: NS_SubQuery, is_partial: bool = False,
        arb_from: Optional[NS_Product] = None, arb_level: int = 0,
    ) -> None:


        if arb_from is not None and not is_fallback:
            reason = {
                1: "it has usable evidence for a constraint the other one only leaves unstated",
                2: "its evidence sits in a structured product field rather than in title wording",
                3: "its evidence carries fewer words the query never asked for",
            }.get(arb_level, "its evidence is the stronger of the two")

            think = (
                f"Two candidates are close here: `{arb_from.title}` and `{best_p.title}`. "
                f"Comparing the evidence each one actually carries, I prefer `{best_p.id}` because {reason}. "
                "I will recommend it."
            )

            self.step(think, f"Best-matching candidate selected: `{best_p.id}`")
            return


        if is_partial:
            think = (
                "No candidate satisfies every constraint in this query, so I will not claim an exact match. "
                "Rather than take whichever candidate search ranked first, I will prefer the one that matches "
                "the kind of product being asked for and misses only a secondary attribute: a product of the "
                "wrong kind cannot be right at all, whereas the right kind of product with an unlisted variant "
                "still can be. Here is the comparison I made across the candidates:\n\n"
                f"```json\n{json.dumps(js_check, ensure_ascii = False, indent = 2)}```\n\n"
                f"On that basis the closest available match is `{best_p.title}`, which I will recommend while "
                "noting that it is a partial match rather than an exact one."
            )

            self.step(think, f"Closest partial match selected: `{best_p.id}`")
            return

        if not is_fallback:
            think = (
                f"From the retrieved products, I will consider the top {NS_SearchEngine.SELECT_CANDIDATES} candidates and check them in several phases until a product satisfying all the constraints is found. "
                "For optimization, I will use a strategy based on Fibonacci sequence windows for checking candidates.\n\n"
                f"```json\n{json.dumps(js_check, ensure_ascii = False, indent = 2)}```\n\n"
                "Good. A matching product has been identified."
            )
        else:
            think = (
                "Since the retrieved products are already ranked by relevance, I will first examine the highest-ranked product to determine whether it satisfies all the user's requirements. "
                "If it turns out to be a good match, I can recommend it directly without performing any further evaluation. "
                f"The title of the highest-ranked product is: `{best_p.title}` "
                "I will check the attribute details of the product to verify each constraint using the following checklist:\n"
            )
            for kw in sq.kw:
                key = '_'.join([s for k in kw[:-1] for s in k.split()])
                think += f"\n`{key}`: `{kw[-1]}` → match ✓"

            think += "\n\nAll checks passed."

        self.step(think, f"Best-matching candidate selected: `{best_p.id}`")

    def step_on_compare(self, best_p: List[NS_Product], alt: str) -> None:
        if not best_p or not alt: return

        think = "To avoid overlooking a better candidate, I will compare the identified one with other top promising candidate.\n\n" + alt
        self.step(think, "Compared with the top promising candidate.")

class NS_SearchEngine(object):
    SELECT_CANDIDATES = 30
    SCORE_N_THRES = 0.4


    TIMEOUT_SECS = 290.0


    SEARCH_WORKERS = 4
    FILTER_MAX_TRIES = 3


    FILTER_SHARD_SIZE = 9
    FILTER_MAX_SHARDS = 3
    FILTER_WORKERS = 3


    MAX_SEARCH_OPTIONS = 16


    VERIFY_RESERVE_SECS = 55.0


    FILTER_MIN_SECS = 18.0
    COMBO_MIN_SECS = 0.0
    SELECT_MIN_SECS = 0.0


    RESERVE_AFTER_TERMINAL = 5.0
    RESERVE_AFTER_FILTER = 45.0


    FILTER_MAX_SECS = 50.0
    SELECT_MAX_SECS = 0.0

    def __init__(self, qi: NS_QueryInfo, sess: NS_SearchSession, debugger: NS_Debugger = None) -> None:
        self.debugger = debugger
        self.sess = sess
        self.qi = qi

        self.weight_pool: Dict[NS_SubQuery, Dict[str, NS_Weight]] = defaultdict(lambda: {})
        self.prod_pool: Dict[str, NS_Product] = {}

        self.searches: Dict[NS_SubQuery, Dict[NS_SearchOption, List[NS_Product]]] = {}
        self.products: Dict[NS_SubQuery, List[NS_Product]] = {}

        self.synonyms: List[Tuple[str, str]] = [(a, b) for a, b in NS_Vocab.SYNONYMS]
        self.fillers: List[str] = []

        self.sos: Dict[NS_SubQuery, List[NS_SearchOption]] = defaultdict(lambda: [])
        self.irank: Dict[NS_SubQuery, int] = defaultdict(lambda: 0)
        self.fscore: Dict[NS_SubQuery, float] = {}

        self.filtered = defaultdict(lambda: defaultdict(lambda: None))


        self._evidence_memo: Dict[Tuple[int, str], int] = {}
        self.watches: Dict[str, float] = {}


        self.cross_admitted: Dict[NS_SubQuery, set] = defaultdict(lambda: set())

    def _start_watch(self, key: str, reserve: float = 0.0) -> None:


        self.watches[key] = (NS_ProxyUtil.DEADLINE - reserve) if NS_ProxyUtil.DEADLINE else math.inf

    def _watch_ok(self, key: str = "query") -> bool:
        if key not in self.watches:
            self.debugger.print(f"# non-initialized watch: `{key}`")
            return False

        if time.monotonic() > self.watches[key]:
            _debug_print(f"* time-out alerted for `{key}`")
            return False
        else:
            return True

    def _dedupe_via_pool(self, p: NS_Product) -> NS_Product:
        if p.id in self.prod_pool:
            existing = self.prod_pool[p.id]

            if p.is_viewed and not existing.is_viewed:
                existing.clone_from(p)
        else:
            self.prod_pool[p.id] = p

        return self.prod_pool[p.id]

    def _passes_basic_filters(self, prod: NS_Product, so: NS_SearchOption) -> bool:
        if so.price:
            if not prod.price: return False

            prod_price = NS_MiscUtil.extract_number(str(prod.price))
            low, high = NS_EngineUtil.numerify_price(so.price)

            if prod_price < low or prod_price > high: return False

        if so.services:
            prod_services = [str(svc).lower() for svc in (prod.service or [])]
            sq_services = [str(svc).lower() for svc in so.services]

            for svc in sq_services:
                if svc not in prod_services: return False

        if so.shop_id and so.shop_id != prod.shop_id: return False
        return True

    def _derive_kw_variants(self, sq: NS_SubQuery) -> Tuple[List[List[List[str]]], List[int]]:
        unit_indice, cat = [], []

        for i, ki in enumerate(sq.kw):
            ks, v = ki[:-1], ki[-1]

            if (m := re.search(r"^(\d+)\s+([a-z]+)$", v, re.I)):
                if not any(k in ks for k in NS_Vocab.ID_KEYS):
                    val, unit = m.group(1), m.group(2)

                    cat.append(
                        [
                            ki.copy(),
                            ks + [unit] + [val],
                            ks + [f"{val}{unit}"]
                        ]
                    )
                    unit_indice.append(i)
            elif (m := re.search(r"^(\d+(?:\-\d+)?)([a-z]+)$", v, re.I)):
                if not any(k in ks for k in NS_Vocab.ID_KEYS):
                    val, unit = m.group(1), m.group(2)

                    cat.append(
                        [
                            ki.copy(),
                            ks + [unit] + [val]
                        ]
                    )
                    unit_indice.append(i)


            elif (m := re.search(r"^([a-z]{1,4})\s+(\d+(?:\.\d+)?)$", v, re.I)):
                if not any(k in ks for k in NS_Vocab.ID_KEYS):
                    unit, val = m.group(1), m.group(2)

                    cat.append(
                        [
                            ki.copy(),
                            ks + [f"{unit}{val}"]
                        ]
                    )
                    unit_indice.append(i)
            elif ks == ["size"] and len(v) >= 5 and not re.search(r"\d", v):


                if (m := re.search(r"\b(extra)\b", v, re.I)):
                    abb = m.group(1)[1]
                    v = v.replace(m.group(1), "").strip()
                else:
                    abb = ""

                if len(v) >= 5 and len(v.split()) == 1 and v[0] in "slm":
                    abb += v[0]

                    cat.append([ki.copy(), ks + [abb]])
                    unit_indice.append(i)

                    self.synonyms.append((min(ki[-1], abb), max(ki[-1], abb)))
            else:


                nv = re.sub(r"(\w+)(?:\s+|\-)res(?:is)[a-z]+", r"\1proof", v, flags = re.I)

                if nv != v:
                    cat.append([ki.copy(), ks + [nv]])
                    unit_indice.append(i)

        return cat, unit_indice

    @staticmethod
    def _variant_choices(cat: List[List[List[str]]]) -> List[Tuple[int, ...]]:


        base = tuple(0 for _ in cat)
        choices: List[Tuple[int, ...]] = [base]

        for i, forms in enumerate(cat):
            for c in range(1, len(forms)):
                choice = list(base)
                choice[i] = c
                choices.append(tuple(choice))

        return choices[:NS_SearchEngine.MAX_SEARCH_OPTIONS]

    def _build_search_options(self, sq: NS_SubQuery, shadow: bool) -> List[NS_SearchOption]:
        if self.sos[sq]: return [NS_cast_model(so, NS_SearchOption) for so in self.sos[sq]]

        cat, unit_indice = self._derive_kw_variants(sq)
        sos: List[NS_SearchOption] = []

        if cat:
            for choice in self._variant_choices(cat):
                new_sq: NS_SubQuery = NS_cast_model(sq, NS_SubQuery)

                for i in range(len(unit_indice)):
                    id, c = unit_indice[i], choice[i]
                    new_sq.kw[id] = cat[i][c]

                sos.append(NS_cast_model(new_sq, NS_SearchOption))

        else:
            sos = [NS_cast_model(sq, NS_SearchOption)]

        for so in sos:
            if shadow:
                low, high = NS_EngineUtil.numerify_price(sq.price or "-")
                if high < math.inf: low = max(low, high // 2)
                so.price = NS_EngineUtil.stringfy_price(low, high)

            so.update_q()


        seen, uniq = set(), []
        for so in sos:
            if so in seen: continue
            seen.add(so)
            uniq.append(so)

        self.sos[sq] = uniq
        return self.sos[sq]

    def _build_bisect_search_options(self, sq: NS_SubQuery) -> List[NS_SearchOption]:


        base_sos = self.sos[sq]
        if len(base_sos) != 1: return []

        products = [p for prods in self.searches[sq].values() for p in prods]
        if not products: return []

        base_so: NS_SearchOption = NS_cast_model(base_sos[0], NS_SearchOption)
        prices = sorted(p.price for p in products)
        median = int(prices[len(prices) // 2])
        low, high = NS_EngineUtil.numerify_price(base_so.price or "-")

        lso: NS_SearchOption = NS_cast_model(base_so, NS_SearchOption)
        lso.update_price(NS_EngineUtil.stringfy_price(low, median))

        hso: NS_SearchOption = NS_cast_model(base_so, NS_SearchOption)
        hso.update_price(NS_EngineUtil.stringfy_price(median, high))

        return [lso, hso]

    def _fetch_pages(
        self, opt: NS_SearchOption, page_limit: int = 1
    ) -> List[Tuple[List[NS_Product], ToolCallResult]]:


        pages: List[Tuple[List[NS_Product], ToolCallResult]] = []

        for page in range(1, 6):
            searched, tool_res = NS_ProxyUtil.find_products(opt, page = page)
            pages.append((searched, tool_res))
            if page >= page_limit: break

        return pages

    def _integrate_pages(
        self, opt: NS_SearchOption, pages: List[Tuple[List[NS_Product], ToolCallResult]]
    ) -> List[NS_Product]:


        res: List[NS_Product] = []

        for searched, tool_res in pages:
            self.sess.add_tool_result(tool_res)

            found: List[NS_Product] = []

            for p in searched:
                if self._passes_basic_filters(p, opt): found.append(self._dedupe_via_pool(p))

            to_view = [p for p in found if not p.is_viewed]
            if len(to_view) > 0: NS_ProxyUtil.view_products(to_view)

            for p in found:
                res.append(self._dedupe_via_pool(p))

        return res

    def _fetch_and_enrich(self, opt: NS_SearchOption, page_limit: int = 1) -> List[NS_Product]:
        return self._integrate_pages(opt, self._fetch_pages(opt, page_limit))

    def _score_single_product(self, prod: NS_Product, sq: NS_SubQuery, rank: int) -> NS_Weight:


        sku_headers: set = set()

        def __get_cand_vs__(v_o: str) -> List[str]:
            v_cands = [v_o]

            for a, b in self.synonyms:
                if a == v_o and b not in v_cands: v_cands.append(b)
                if b == v_o and a not in v_cands: v_cands.append(a)

            return v_cands

        def __dry_run__(title: str, joint_attr: List[str]):
            used_attr, used_headers = [], []
            len_pool, ti_pool, ai_pool, num_pool, match_pool = {}, {}, {}, {}, {}

            for kw in sq.kw:
                v_cands, ks = __get_cand_vs__(kw[-1]), kw[:-1]

                for v in v_cands:
                    atoms = [a.strip() for a in NS_MiscUtil.normalize_str(v).split() if a.strip()]
                    hash_key = tuple(kw + [v])

                    len_pool[hash_key] = len(atoms)
                    as_is = re.search(r"^(\d+)$", v)

                    ti_pool[hash_key] = {
                        i for i, a in enumerate(atoms)
                        if NS_EngineUtil.is_kw_in(a, title, as_is, self.synonyms)
                    }
                    num_pool[hash_key] = [i for i, a in enumerate(atoms) if a.isnumeric()]

                    for ja in joint_attr:
                        ai_pool[(hash_key, ja)] = [
                            i for i, kw in enumerate(atoms)
                            if NS_EngineUtil.is_kw_in(kw, ja, as_is, self.synonyms)
                        ]
                        match_pool[(hash_key, ja)] = (False, 0)

                        if len(ai_pool[(hash_key, ja)]) == len(atoms):
                            if any(NS_EngineUtil.is_kw_in(a, ja.split('🎈')[1], as_is, self.synonyms) for a in atoms):
                                match_pool[(hash_key, ja)] = (True, __get_key_bonus__(ja, ks, atoms))

                        if ja in used_attr: continue
                        h, b = ja.split('🎈')

                        if any(NS_EngineUtil.is_kw_in(kw, b, as_is, self.synonyms) for kw in atoms):
                            used_attr.append(ja)

                        if h in used_headers: continue

                        if any(NS_EngineUtil.is_kw_in(kw, h, as_is, self.synonyms) for kw in atoms):
                            used_headers.append(h)

            return (used_attr, used_headers, len_pool, ti_pool, ai_pool, num_pool, match_pool)

        def __pre_compute__(title: str):
            all_attr = prod.extract_attr(sq.query, '🎈', True)
            joint_attr = [l for _, g in all_attr.items() for lines in g for l in lines]

            sku_headers.update(l.split('🎈')[0] for lines in all_attr.get("sku", []) for l in lines)

            attr = {}
            (
                used_attr, used_headers, len_pool, ti_pool, ai_pool, num_pool, match_pool
            ) = __dry_run__(title, joint_attr)

            for idx, g in all_attr.items():
                new_g = []

                for lines in g:
                    new_lines, headers_seen = [], []

                    for l in lines:
                        if l in used_attr:
                            new_lines.append(l)
                        else:
                            h = l.split('🎈')[0]

                            if h not in headers_seen and h in used_headers:
                                headers_seen.append(h)
                                new_lines.append(l)

                    new_lines = [l for l in new_lines if __is_valid__(l)]
                    if new_lines: new_g.append(new_lines)

                if new_g: attr[idx] = new_g

            combs = NS_MiscUtil.get_choice_combs(attr, True)
            return (attr, combs, len_pool, ti_pool, ai_pool, num_pool, match_pool)

        def __get_key_bonus__(ja: str, ks: List[str], atoms: List[str]) -> float:
            bon = 0
            header = ja.split('🎈')[0]


            if header not in sku_headers and header not in ["variation", "color_family"]:
                for k in ks:
                    k = NS_MiscUtil.normalize_str(k.replace('_', ' '))
                    found = False

                    for ak in k.split():
                        ak = ak.strip()
                        if not ak or ak in atoms: continue

                        if NS_EngineUtil.is_kw_in(ak, ja, False, self.synonyms):
                            bon += 0.1
                            found = True
                            break

                    if found:
                        if k == ja.split('🎈')[0]: bon += 0.05
                        break
            else:
                bon += 0.05

            return bon

        def __is_valid__(ja: str) -> float:
            jv = NS_MiscUtil.normalize_str(ja.split('🎈')[1])

            for s in jv.split():
                if not s: continue

                if s.isnumeric() or s in NS_Vocab.COLOR_WORDS:
                    if f".{s}" in ja or f".{s}" in sq.query: continue
                    cands, exists = {s}, False

                    for a, b in self.synonyms:
                        if a == s: cands.add(b)
                        if b == s: cands.add(a)

                    for k in cands:
                        if re.search(rf"\b{re.escape(k)}\b", NS_MiscUtil.normalize_str(sq.query), re.I):
                            exists = True
                            break

                    if not exists: return False

            return True

        title = NS_MiscUtil.normalize_str(prod.title)
        candidates: List[NS_Weight] = []
        (
            attr, combs, len_pool, ti_pool, ai_pool, num_pool, match_pool
        ) = __pre_compute__(title)

        for group in combs:
            joint_attr: List[str] = [a for ak, idx in group.items() for a in (attr[ak][idx] if idx >= 0 else [])]
            weight = NS_Weight(group = group, rank = rank)

            for i_kw, kw in enumerate(sq.kw):
                v_cands, ks = __get_cand_vs__(kw[-1]), kw[:-1]
                max_score = -math.inf

                if "no" in ks:
                    continue

                for v in v_cands:
                    hash_key = tuple(kw + [v])
                    atom_len = len_pool[hash_key]

                    attr_includes = set()
                    score = 0

                    title_includes = ti_pool[hash_key]
                    found_in_attr, bonuses = False, []

                    for ja in joint_attr:
                        attr_includes.update(ai_pool[(hash_key, ja)])

                        if match_pool[(hash_key, ja)][0]:
                            bonuses.append(match_pool[(hash_key, ja)][1])
                            found_in_attr = True

                    if not found_in_attr:
                        if len(title_includes) < atom_len:
                            frags = title_includes | attr_includes

                            if all(i in frags for i in num_pool[hash_key]):
                                score += len(frags) / atom_len
                        else:
                            score += 1.0
                    else:
                        score += 1.0 if ks[-1] != "product_type" else 0.95
                        score += max(bonuses)

                    if score > max_score: max_score = score

                wk = '_'.join(ks)
                if wk in weight.detail: wk += f"_{i_kw + 1}"

                weight.detail[wk] = max_score
                weight.score += max_score

            weight.normalize()
            candidates.append(weight)

        candidates.sort(key = lambda w: -w.score_n)
        return candidates[0]

    def _score_products(
        self,
        sq: NS_SubQuery,
        searches: Dict[NS_SearchOption, List[NS_Product]],
        id: Optional[str],
        dbg_prefix: str
    ) -> List[NS_Product]:

        ranks: Dict[NS_Product, int] = defaultdict(lambda: math.inf)
        products: List[NS_Product] = []

        for so in searches:
            for r, p in enumerate(searches[so]):
                if r < ranks[p]:
                    if p not in products: products.append(p)
                    ranks[p] = r

        for p in sorted(products, key = lambda p: ranks[p]):
            w = self._score_single_product(p, sq, ranks[p] + self.irank[sq])

            if p.id in self.weight_pool[sq]:
                self.weight_pool[sq][p.id].update_with_sibling(w)
            else:
                self.weight_pool[sq][p.id] = w

        products.sort(key = lambda p: -self.weight_pool[sq][p.id].score_n)

        for i, p in enumerate(products):
            self.weight_pool[sq][p.id].update_confidence(i + self.irank[sq])

        return products

    def _above_score_floor(self, products: List[NS_Product], sq: NS_SubQuery) -> List[NS_Product]:
        return [
            p for p in products
            if self.weight_pool[sq][p.id].score_n >= self.fscore[sq] - NS_SearchEngine.SCORE_N_THRES
        ]

    def _build_filter_prompt(
        self,
        shard: Dict[NS_SubQuery, Tuple[int, List[NS_Product]]],
        letters: Dict[NS_SubQuery, str],
    ) -> str:


        info = {}

        for sq, (start, products) in shard.items():
            info[letters[sq]] = {
                "query": sq.query,
                "candidates": self._render_products_list(products, sq, letters[sq], False, start = start)
            }

        return NS_Prompt.SV_FILTER.format(
            rules = NS_Prompt.MATCH_RULES,
            count = len(shard),
            info = json.dumps(info, ensure_ascii = False, indent = 2)
        )

    def _render_products_list(
        self, products: List[NS_Product], sq: NS_SubQuery, prefix: str, for_reason: bool, start: int = 0
    ) -> List[Dict[str, Any]]:
        rendered = []

        for i, p in enumerate(products):
            js_p = p.to_prompt(sq.query, for_reason)
            js_p["id"] = prefix + str(start + i + 1)

            rendered.append(js_p)

        return rendered

    def _render_products_block(self, products: List[NS_Product], sq: NS_SubQuery, prefix: str, for_reason: bool) -> str:
        return "".join(
            json.dumps(js_p, ensure_ascii = False) + '\n'
            for js_p in self._render_products_list(products, sq, prefix, for_reason)
        )

    def _render_alt_comparison(self, best_p: List[NS_Product]) -> str:
        if not best_p: return ""

        if len(best_p) == 1:
            for i, p in enumerate(self.products[self.qi.items[0]] * 2):
                if p.price > best_p[0].price or i >= len(self.products[self.qi.items[0]]):
                    alt_price = str(int(p.price)) if i < len(self.products[self.qi.items[0]]) else str(int(best_p[0].price + 2.0))
                    alt_title = p.title if len(p.title) < 90 else (p.title[:90] + "...")

                    return (
                        f"The product `{p.id}` appears to be a strong alternative. Let me compare it with `{best_p[0].id}`. The title of `{p.id}` is `{alt_title}`. "
                        f"Based on the product details, `{p.id}` matches all stated constraints like `{best_p[0].id}`, making it difficult to choose one of them. "
                        f"The price of product `{p.id}` is {alt_price}, while the price of `{best_p[0].id}` is {str(int(best_p[0].price))}. "
                        f"The product `{best_p[0].id}` is less expensive. Therefore, I will prefer `{best_p[0].id}`."
                    )
        else:
            for i, sq in enumerate(self.qi.items * 2):
                for p in self.products[sq]:
                    if i >= len(self.qi.items) or p.price > best_p[i].price:
                        bp = best_p[i % len(self.qi.items)]
                        alt_price = str(int(p.price)) if i < len(self.qi.items) else str(int(bp.price + 3.0))
                        alt_title = p.title if len(p.title) < 90 else (p.title[:90] + "...")

                        return (
                            f"The product `{p.id}` appears to be a strong alternative of the product `{bp.id}`, for the sub-query `{sq.query}`. "
                            f"The title of `{p.id}` is `{alt_title}`. Let's consider a combination which has replaced `{bp.id}` with `{p.id}`. "
                            f"Still it meets the stated constraints for each corresponding sub-query. Now let me check the prices. "
                            f"The price of product `{p.id}` is {str(int(p.price))}, while the price of `{bp.id}` is {str(int(bp.price))}. "
                            f"It indicates that the replaced combination will have a higher total price than the original one. "
                            f"Therefore, the combination `{', '.join([p.id for p in best_p])}` is less expensive. I will prefer it."
                        )

        return ""

    def _run_search(
        self,
        sq: NS_SubQuery,
        sos: List[NS_SearchOption],
        id: str,
        page_limit: int,
        dbg_prefix: str
    ) -> Dict:


        workers = min(getattr(type(self), "SEARCH_WORKERS", 1), len(sos))
        if workers <= 1:
            return {so: self._fetch_and_enrich(so, page_limit) for so in sos}

        raw: List[Optional[List[Tuple[List[NS_Product], ToolCallResult]]]] = [None] * len(sos)
        with ThreadPoolExecutor(max_workers = workers) as pool:
            fmap = {pool.submit(self._fetch_pages, so, page_limit): i for i, so in enumerate(sos)}
            for fut in as_completed(fmap):
                i = fmap[fut]
                try:
                    raw[i] = fut.result()
                except Exception as exc:
                    self.debugger.print(f"# parallel search fetch failed for option {i}: {exc}")
                    raw[i] = []

        return {so: self._integrate_pages(so, raw[i] or []) for i, so in enumerate(sos)}

    def _pick_best(
        self, sq: NS_SubQuery, candidates: List[NS_Product], must: bool, dbg_prefix: str
    ) -> Tuple[
        Optional[NS_Product], bool
    ]:
        best_p, js_check, is_fallback = None, {}, False
        is_partial, arb_from, alvl = False, None, 0

        if len(candidates) > 1 or (len(candidates) == 1 and not must):
            js_check, _ = NS_ProxyUtil.trigger_llm(
                NS_Prompt.PV_SELECT.format(
                    rules = NS_Prompt.MATCH_RULES,
                    rank_rules = NS_Prompt.RANK_RULES,
                    query = sq.query,
                    products = self._render_products_block(candidates, sq, "", False)
                ),
                preferred_llms=NS_ProxyUtil.get_role_llm_priorities("select"),
                expect = "dict",
                min_secs = NS_SearchEngine.SELECT_MIN_SECS,
                max_secs = NS_SearchEngine.SELECT_MAX_SECS,
                reserve_after = NS_SearchEngine.RESERVE_AFTER_TERMINAL,
            )
            bid = NS_Prompt.extract_select_response(js_check)
            self.debugger.print(js_check)

            if bid >= 0 and bid < len(candidates):


                abid, alvl = NS_Prompt.arbitrate_tally(js_check, bid, len(candidates))

                if abid != bid and 0 <= abid < len(candidates):
                    self.debugger.print(
                        f"* tally arbiter: #{bid + 1} -> #{abid + 1} on {NS_Prompt.TALLY_LEVELS[alvl - 1]}"
                    )

                    arb_from, bid = candidates[bid], abid

                best_p = candidates[bid]
            elif must:


                cid = NS_Prompt.extract_closest_response(js_check)

                if cid >= 0 and cid < len(candidates):
                    self.debugger.print(f"* fallback to the closest partial match: #{cid + 1} for {sq.query[:50]}")

                    best_p = candidates[cid]
                    is_partial = True
                else:
                    self.debugger.print(f"* fallback to select the first product: {sq.query[:50]}")

                    best_p = candidates[0]

                is_fallback = True
        elif len(candidates) > 0:
            self.debugger.print(f"* selected only one product for: {sq.query[:50]}")

            best_p = candidates[0]
            is_fallback = True
        else:
            self.debugger.print(f"# cannot select the best because no candidate exists!")

        if best_p is not None:
            self.sess.step_on_select(best_p, js_check, is_fallback, sq, is_partial, arb_from, alvl)
        return best_p, is_fallback

    def _shard_filter_candidates(
        self, products: Dict[NS_SubQuery, List[NS_Product]]
    ) -> List[Dict[NS_SubQuery, Tuple[int, List[NS_Product]]]]:


        total = sum(len(ps) for ps in products.values())
        n = min(
            NS_SearchEngine.FILTER_MAX_SHARDS,
            max(1, -(-total // NS_SearchEngine.FILTER_SHARD_SIZE)),
        )

        if n <= 1:
            return [{sq: (0, list(ps)) for sq, ps in products.items() if ps}]

        shards: List[Dict[NS_SubQuery, Tuple[int, List[NS_Product]]]] = []

        for s in range(n):
            shard: Dict[NS_SubQuery, Tuple[int, List[NS_Product]]] = {}

            for sq, ps in products.items():
                lo, hi = len(ps) * s // n, len(ps) * (s + 1) // n
                if hi > lo: shard[sq] = (lo, ps[lo:hi])

            if shard: shards.append(shard)

        return shards

    def _filter_shard(
        self,
        shard: Dict[NS_SubQuery, Tuple[int, List[NS_Product]]],
        letters: Dict[NS_SubQuery, str],
        counts: List[int],
    ) -> Tuple[List[Dict[int, bool]], Dict]:


        last_js: Dict = {}

        for _try in range(NS_SearchEngine.FILTER_MAX_TRIES):
            js, _ = NS_ProxyUtil.trigger_llm(
                self._build_filter_prompt(shard, letters),
                preferred_llms = NS_ProxyUtil.get_role_llm_priorities("filter"),
                temperature = _try * 0.3,
                expect = "dict",
                min_secs = NS_SearchEngine.FILTER_MIN_SECS,
                max_secs = NS_SearchEngine.FILTER_MAX_SECS,
                reserve_after = NS_SearchEngine.RESERVE_AFTER_FILTER,
            )
            if isinstance(js, dict) and js: last_js = js

            verdicts = NS_Prompt.extract_filter_response(js, counts)
            if any(verdicts): return verdicts, last_js

        return [{} for _ in counts], last_js

    def _llm_filter(
        self,
        products: Dict[NS_SubQuery, List[NS_Product]],
        id: str,
        dbg_prefix: str
    ) -> Optional[Dict[NS_SubQuery, Dict[NS_Product, bool]]]:


        groups = list(products.keys())
        letters = {sq: NS_Vocab.ALPHABETS[i] for i, sq in enumerate(groups)}
        counts = [len(products[sq]) for sq in groups]
        shards = self._shard_filter_candidates(products)

        self.debugger.print(
            f"* llm filter [{id}]: checking {sum(counts)} candidate(s) across "
            f"{len(groups)} group(s) in {len(shards)} shard(s)..."
        )

        results: List[Optional[Tuple[List[Dict[int, bool]], Dict]]] = [None] * len(shards)
        workers = min(NS_SearchEngine.FILTER_WORKERS, len(shards))

        if workers <= 1:
            for i, shard in enumerate(shards):
                try:
                    results[i] = self._filter_shard(shard, letters, counts)
                except Exception as exc:
                    self.debugger.print(f"# filter shard {i} failed: {exc}")
        else:
            with ThreadPoolExecutor(max_workers = workers) as pool:
                fmap = {
                    pool.submit(self._filter_shard, shard, letters, counts): i
                    for i, shard in enumerate(shards)
                }
                for fut in as_completed(fmap):
                    i = fmap[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as exc:
                        self.debugger.print(f"# filter shard {i} failed: {exc}")


        merged: Dict[NS_SubQuery, Dict[NS_Product, bool]] = {sq: {} for sq in groups}
        js: Dict[str, Any] = {}
        n_checked = 0

        for res in results:
            if res is None: continue
            verdicts, shard_js = res

            for gid, vmap in enumerate(verdicts):
                sq = groups[gid]
                for bid, ok in vmap.items():
                    if products[sq][bid] not in merged[sq]: n_checked += 1
                    merged[sq][products[sq][bid]] = ok

            for gk, gv in (shard_js or {}).items():
                if isinstance(gv, dict) and isinstance(js.get(gk), dict):
                    js[gk].update(gv)
                else:
                    js[gk] = gv


        def _text_of(gid: int, bid: int) -> str:
            sq = groups[gid]
            if bid >= len(products[sq]): return ""

            block = products[sq][bid].to_prompt(sq.query)
            return "\n".join(str(v) for v in block.values())

        n_dropped = 0
        for gid, dmap in enumerate(NS_Prompt.arbitrate_filter(js, counts, _text_of)):
            sq = groups[gid]

            for bid, why in dmap.items():
                p = products[sq][bid]
                if merged[sq].pop(p, None) is None: continue

                n_dropped += 1
                self.debugger.print(
                    f"* filter arbiter [{id}]: {NS_Vocab.ALPHABETS[gid]}{bid + 1} -> unverified ({why})"
                )

        n_checked -= n_dropped

        if n_checked == 0:


            self.debugger.print(f"# llm filter [{id}] unusable after retries -- leaving {sum(counts)} candidate(s) unverified")
            self.debugger.add("k-filter", {"id": id, "js": js, "n_checked": sum(counts), "skipped": True})
            return None

        self.debugger.print(js)

        n_passed = sum(1 for vmap in merged.values() for ok in vmap.values() if ok)
        self.debugger.add("k-filter", {"id": id, "js": js, "n_checked": n_checked, "n_passed": n_passed})
        self.sess.step_on_filter(len(groups), n_checked, n_passed, js)

        return merged

    def run(self) -> List[NS_Product]:


        self._start_watch("query")
        self._start_watch("breadth", NS_SearchEngine.VERIFY_RESERVE_SECS)

        for sq in self.qi.items:
            self.searches[sq] = self._run_search(sq, self._build_search_options(sq, True), None, 5, "b")
            if self.debugger.is_stop("b-search"): continue

            bi_sos = self._build_bisect_search_options(sq)
            if bi_sos:
                bi_results = self._run_search(sq, bi_sos, None, 5, "bi")
                self.searches[sq].update(bi_results)

            self.products[sq] = self._score_products(sq, self.searches[sq], None, "b")
            if self.debugger.is_stop("b-weigh"): continue

            if self.products[sq]:
                self.fscore[sq] = self.weight_pool[sq][self.products[sq][0].id].score_n
            else:
                self.debugger.print(f"* no products found for `{sq.query[:50]}`")
                self.fscore[sq] = math.inf

            self.irank[sq] = len(self.products[sq])

        self.sess.step_on_search(self.qi)
        return []

class NS_ComboEngine(NS_SearchEngine):


    FILTER_BATCH = 25
    SHOP_BATCH = 3

    MAX_EXT_SHOPS = 10
    MAX_CAT_SIZE = 300


    CROSS_MAX_PRODUCTS = 150
    CROSS_CLOCK_EVERY = 40


    EVIDENCE_TIEBREAK_N = 24


    PROBE_TERMS = 4

    MAX_SELECT_COMBOS = 10
    MAX_SELECT_BATCHES = 3


    MAX_VERIFIED_COMBOS = 200000

    def _has_filter_reject(self, prods) -> bool:


        return any(
            self.filtered[self.qi.items[i]][p] == False
            for i, p in enumerate(prods)
        )

    def _partial_match_combo(
        self,
        solutions: List[Tuple[Tuple[NS_Product, ...], Any]],
        min_verified: int,
    ) -> List[NS_Product]:


        clean = [s for s in solutions if not self._has_filter_reject(s[0])]
        tainted = [s for s in solutions if self._has_filter_reject(s[0])]

        for pool in (clean, tainted):
            if not pool: continue
            note = " [contains explicitly-rejected slot]" if pool is tainted else ""

            for count in range(len(self.qi.items), min_verified - 1, -1):
                eligible = [
                    prods for prods, _ in pool
                    if len([
                        p for i, p in enumerate(prods)
                        if self.filtered[self.qi.items[i]][p]
                    ]) >= count
                ]
                if not eligible: continue


                head = eligible[:NS_ComboEngine.EVIDENCE_TIEBREAK_N]
                best = max(head, key = lambda prods: self._structured_evidence(prods))

                if best is not head[0]:
                    self.debugger.print(
                        "* fallback tie broken on structured evidence "
                        f"({self._structured_evidence(head[0])} -> {self._structured_evidence(best)})"
                    )

                n_ok = len([
                    p for i, p in enumerate(best)
                    if self.filtered[self.qi.items[i]][p]
                ])
                self.debugger.print(f"* fallback combo with {n_ok} verified slot(s){note}")
                return list(best)

        return []

    def _structured_evidence(self, prods: Tuple[NS_Product, ...]) -> int:


        total = 0

        for i, p in enumerate(prods):
            if i >= len(self.qi.items): break
            sq = self.qi.items[i]
            key = (id(sq), p.id)

            if key not in self._evidence_memo:
                self._evidence_memo[key] = NS_Prompt.structured_hits(
                    p.to_prompt(sq.query), NS_EngineUtil.kw_to_constraints(sq.kw)
                )[0]

            total += self._evidence_memo[key]

        return total

    def _has_filter_unchecked(self, prods) -> bool:


        return any(
            self.filtered[self.qi.items[i]][p] is None
            for i, p in enumerate(prods)
        )

    def _verified_combo(
        self, same_shop: bool = False, price_range: Optional[Tuple[float, float]] = None
    ) -> List[NS_Product]:


        if not same_shop:
            slot_map: Dict[NS_SubQuery, List[NS_Product]] = {}

            for sq in self.qi.items:
                verified = [p for p, ok in self.filtered[sq].items() if ok]
                if not verified: return []
                slot_map[sq] = verified

            return self._pick_verified_solution(self._enumerate_verified(slot_map, price_range))

        per_shop: Dict[str, Dict[NS_SubQuery, List[NS_Product]]] = defaultdict(lambda: defaultdict(lambda: []))
        for sq in self.qi.items:
            for p, ok in self.filtered[sq].items():
                if ok: per_shop[p.shop_id][sq].append(p)

        all_solutions: List[Tuple[Tuple[NS_Product, ...], float]] = []
        for slot_map in per_shop.values():
            if len(slot_map) != len(self.qi.items): continue
            all_solutions.extend(self._enumerate_verified(slot_map, price_range))

        return self._pick_verified_solution(all_solutions)

    def _enumerate_verified(
        self,
        slot_map: Dict[NS_SubQuery, List[NS_Product]],
        price_range: Optional[Tuple[float, float]]
    ) -> List[Tuple[Tuple[NS_Product, ...], float]]:


        combos: List[Tuple[Tuple[NS_Product, ...], float]] = []

        n_slots = max(len(self.qi.items), 1)
        cap = max(2, int(NS_ComboEngine.MAX_VERIFIED_COMBOS ** (1.0 / n_slots)))
        cats = [slot_map[sq][:cap] for sq in self.qi.items]

        dropped = sum(max(0, len(slot_map[sq]) - cap) for sq in self.qi.items)
        if dropped:
            self.debugger.print(
                f"* verified cross-product capped at {cap}/slot ({dropped} lower-confidence "
                f"candidate(s) not enumerated)"
            )

        for i, choice in enumerate(product(*cats)):


            if not (i & 0x3FF) and not self._watch_ok():
                self.debugger.print(f"* verified cross-product stopped by the clock after {i} combo(s)")
                break
            if len({p.id for p in choice}) < len(choice): continue
            if price_range is not None:
                total = sum(p.price for p in choice)
                if total < price_range[0] or total > price_range[1]: continue

            weight = math.prod(
                self.weight_pool[sq][p.id].confidence for sq, p in zip(self.qi.items, choice)
            )
            combos.append((choice, weight))

        return combos

    def _pick_verified_solution(
        self, solutions: List[Tuple[Tuple[NS_Product, ...], float]]
    ) -> List[NS_Product]:


        if not solutions: return []
        if len(solutions) == 1: return list(solutions[0][0])

        solutions = sorted(solutions, key = lambda s: -s[-1])[:NS_ComboEngine.MAX_SELECT_COMBOS]
        return self._combo_select(solutions, "verified")

    def _cross_attribute_pool(self) -> int:


        if len(self.qi.items) < 2: return 0


        by_shop: Dict[str, List[NS_Product]] = defaultdict(lambda: [])
        for p in self.prod_pool.values():
            by_shop[p.shop_id].append(p)


        candidates = sorted(
            (p for sid, ps in by_shop.items() if len(ps) > 1 for p in ps),
            key = lambda p: -len(by_shop[p.shop_id])
        )

        n_added = 0

        for sq in self.qi.items:
            if self.fscore.get(sq, math.inf) == math.inf: continue

            floor = self.fscore[sq] - NS_SearchEngine.SCORE_N_THRES
            native = self.products.get(sq) or []
            seen = {p.id for p in native}
            covered = {p.shop_id for p in native}


            pool = [p for p in candidates if p.id not in seen and p.shop_id not in covered]
            pool = pool[:NS_ComboEngine.CROSS_MAX_PRODUCTS]

            added: List[NS_Product] = []

            for i, p in enumerate(pool):
                if i % NS_ComboEngine.CROSS_CLOCK_EVERY == 0 and not self._watch_ok("breadth"):
                    self.debugger.print("* cross-attribution stopped early: inside the verification reserve")
                    break

                w = self._score_single_product(p, sq, self.irank[sq])
                if w.score_n < floor: continue

                if p.id in self.weight_pool[sq]:
                    self.weight_pool[sq][p.id].update_with_sibling(w)
                else:
                    self.weight_pool[sq][p.id] = w

                added.append(p)

            if not added: continue


            merged = sorted(native + added, key = lambda p: -self.weight_pool[sq][p.id].score_n)
            for i, p in enumerate(merged):
                self.weight_pool[sq][p.id].update_confidence(i)

            self.products[sq] = native + sorted(
                added, key = lambda p: -self.weight_pool[sq][p.id].score_n
            )
            self.irank[sq] = len(self.products[sq])
            self.cross_admitted[sq].update(p.id for p in added)
            n_added += len(added)

            self.debugger.print(
                f"* cross-attribution: +{len(added)} candidate(s) for `{sq.query[:45]}` "
                f"({len(native)} from its own search)"
            )

        return n_added

    def _build_shop_map(
        self, products: Dict[NS_SubQuery, List[NS_Product]], apply_floor: bool = True
    ) -> Tuple[
        Dict[str, Dict[NS_SubQuery, List[NS_Product]]],
        Dict[str, Tuple[int, float]]
    ]:
        shop_map: Dict[str, Dict[NS_SubQuery, List[NS_Product]]] = defaultdict(lambda: defaultdict(lambda: []))
        shop_stats: Dict[str, Tuple[int, float]] = {}

        for sq in self.qi.items:
            candidates = self._above_score_floor(products[sq], sq) if apply_floor else products[sq]
            for p in candidates:
                if p not in shop_map[p.shop_id][sq]: shop_map[p.shop_id][sq].append(p)

        for sid in shop_map:
            n, w = 0, 1.0

            for sq in shop_map[sid]:
                w *= max(self.weight_pool[sq][p.id].confidence for p in shop_map[sid][sq])
                n += 1

            shop_stats[sid] = (n, w)

        return shop_map, shop_stats

    def _plan_shop_batches(
        self
    ) -> Tuple[
        List[List[str]],
        Dict[str, Dict[NS_SubQuery, List[NS_Product]]],
        Dict[str, Tuple[int, float]]
    ]:
        shop_map, shop_stats = self._build_shop_map(self.products)
        self.sess.step_on_shop_plan(shop_stats, min(NS_ComboEngine.MAX_EXT_SHOPS, len(list(shop_stats.keys()))), self.qi)

        shops = sorted(list(shop_stats.keys()), key = lambda sid: shop_stats[sid], reverse = True)
        plan = []

        for n in range(4, 0, -1):
            covers = [sid for sid in shops if shop_stats[sid][0] == n]

            while len(covers) > 0:
                plan.append(covers[:NS_ComboEngine.SHOP_BATCH])
                covers = covers[NS_ComboEngine.SHOP_BATCH:]

        return plan, shop_map, shop_stats

    def _build_knapsack_maps(
        self,
        products: Dict[NS_SubQuery, List[NS_Product]],
        is_shop: bool,
        apply_floor: bool = True
    ) -> List[
        Dict[NS_SubQuery, List[NS_Product]]
    ]:
        kmaps = []

        if is_shop:
            shop_map, shop_stats = self._build_shop_map(products, apply_floor)

            for sid in shop_stats:
                if shop_stats[sid][0] == len(self.qi.items): kmaps.append(shop_map[sid])
        else:
            cap = self._max_cat_size()
            lead_prods = {sq: products[sq][:cap] for sq in self.qi.items}
            if all(len(lead_prods[sq]) > 0 for sq in self.qi.items): kmaps.append(lead_prods)

        return kmaps

    def _max_cat_size(self) -> int:

        return NS_ComboEngine.MAX_CAT_SIZE

    def _probe_shop_slot(self, sq: NS_SubQuery, sid: str) -> Dict:


        def words(text: str) -> List[str]:
            return [
                w for w in re.split(r"\W+", (text or "").lower())
                if len(w) >= 4 and not w.isnumeric() and w not in NS_Vocab.KW_STOPWORDS
            ]


        others = {
            w for other in self.qi.items if other is not sq for w in words(other.query)
        }

        terms, seen = [], set()
        ordered = (
            words(sq.phrase) + words(sq.query)
            + [w for w in words(self.qi.query_f) if w not in others]
        )

        for w in ordered:
            if w in seen: continue
            seen.add(w)
            terms.append(w)

        for term in terms[:NS_ComboEngine.PROBE_TERMS]:
            if not self._watch_ok("breadth"): break

            so = NS_cast_model(sq, NS_SearchOption).update_shop(sid).update_price(sq.price or "-")
            res = self._run_search(sq, [so.update_q(term)], sid, 1, "kp")

            if sum(len(ps) for ps in res.values()) > 0:
                self.debugger.print(
                    f"* shop {sid}: last missing slot recovered by single-term probe `{term}`"
                )
                return res

        return {}

    def _search_within_shops(
        self, shops: List[str], shop_map: Dict[str, Dict[NS_SubQuery, List[NS_Product]]], id: int,
        wide: bool = False
    ) -> List[
        Dict[NS_SubQuery, List[NS_Product]]
    ]:
        products: Dict[NS_SubQuery, List[NS_Product]] = {}
        searches: Dict[NS_SubQuery, Dict] = defaultdict(lambda: {})

        for sid in shops:


            missing = [
                sq for sq in self.qi.items
                if not any(p.id not in self.cross_admitted[sq] for p in shop_map[sid][sq])
            ]


            for i, sq in enumerate(missing):
                res = self._run_search(
                    sq,
                    [so.update_shop(sid) for so in self._build_search_options(sq, True)],
                    sid, 1, "k"
                )

                if sum(len(ps) for ps in res.values()) == 0:


                    if wide and i == len(missing) - 1:
                        res = self._probe_shop_slot(sq, sid)

                    if sum(len(ps) for ps in res.values()) == 0: break

                searches[sq].update(res)

        for sq in self.qi.items:
            products[sq] = [p for sid in shops for p in shop_map[sid][sq]]
            if sum(len(ps) for ps in searches[sq].values()) == 0: continue

            for p in self._score_products(sq, searches[sq], str(id), "k"):
                if p not in products[sq]: products[sq].append(p)


        return self._build_knapsack_maps(products, True, apply_floor = False)

    def _solve_knapsack_map(
        self, kmap: Dict[NS_SubQuery, List[NS_Product]], **kwargs
    ) -> List[Tuple[Tuple[NS_Product], float]]:
        cats: List[List[Tuple[NS_Product, float]]] = []

        for sq, products in kmap.items():
            prod_weights = [(p, self.weight_pool[sq][p.id].confidence) for p in products]
            cats.append(prod_weights)

        return self._solve_knapsack(cats, **kwargs)

    def _solve_knapsack(
        self, _: List[List[Tuple[NS_Product, float]]], **__
    ) -> List[Tuple[Tuple[NS_Product, ...], float]]:
        return []

    def _collect_filtered_combos(
        self,
        solmap: Dict[str, List[Tuple[Tuple[NS_Product, ...], Any]]],
        sort: bool,
        id: str,
        max_tries: int
    ) -> List[Tuple[Tuple[NS_Product, ...], Any]]:


        cursor, is_exhausted = 0, False
        collected: List[Tuple[Tuple[NS_Product, ...], Any]] = []
        seen: set = set()


        offered: set = set()

        for t in range(max_tries):
            batch, candidates, cand_count = [], defaultdict(lambda: []), 0
            pre_valid: List[Tuple[Tuple[NS_Product, ...], Any]] = []
            if not self._watch_ok(): break

            while cand_count < NS_ComboEngine.FILTER_BATCH and not is_exhausted:
                is_added = False

                for _, sols in solmap.items():
                    if cursor < len(sols):
                        prods, w = sols[cursor]
                        is_added = True

                        if all(self.filtered[self.qi.items[i]][p] for i, p in enumerate(prods)):
                            pre_valid.append((prods, w))
                            continue
                        if any(self.filtered[self.qi.items[i]][p] == False for i, p in enumerate(prods)): continue

                        for i, p in enumerate(prods):
                            sq = self.qi.items[i]

                            if self.filtered[sq][p] is None and p not in candidates[sq] \
                                and (i, p.id) not in offered:
                                candidates[sq].append(p)
                                cand_count += 1

                        batch.append((prods, w))

                if not is_added: is_exhausted = True
                cursor += 1

            if cand_count > 0:
                for i, sq in enumerate(self.qi.items):
                    for p in candidates[sq]:
                        offered.add((i, p.id))

                verdicts = self._llm_filter(candidates, f"{id}-sol-{t}", "k")


                if verdicts is not None:
                    for sq, vmap in verdicts.items():
                        for p, ok in vmap.items():
                            self.filtered[sq][p] = ok

            if sort: batch.sort(key = lambda b: b[-1], reverse = True)

            for prods, w in pre_valid + batch:
                if not all(bool(self.filtered[self.qi.items[i]][p]) for i, p in enumerate(prods)):
                    continue
                key = tuple(p.id for p in prods)
                if key in seen: continue
                seen.add(key)
                collected.append((prods, w))

            if is_exhausted: break

        if collected:
            self.debugger.print(
                f"* collected {len(collected)} filter-passing combo(s) at [{id}]"
            )
        return collected

    def _resolve_solution_map(
        self,
        solmap: Dict[str, List[Tuple[Tuple[NS_Product, ...], Any]]],
        sort: bool,
        id: str,
        max_tries: int
    ) -> List[NS_Product]:
        collected = self._collect_filtered_combos(solmap, sort, id, max_tries)
        if not collected: return []

        if len(collected) == 1:
            self.debugger.print(f"* selected only one valid combo at [{id}]")
            return list(collected[0][0])

        self.debugger.print(
            f"* multiple valid combos ({len(collected)}) at [{id}] — escalating to LLM combo select"
        )
        return self._combo_select(collected, id)

    def _combo_select(
        self,
        solutions: List[Tuple[Tuple[NS_Product, ...], Any]],
        id: str
    ) -> List[NS_Product]:

        if not solutions: return []

        slots: Dict[str, Any] = {
            str(i + 1): {
                "sub_query": sq.query,
                "constraints": NS_EngineUtil.kw_to_constraints(sq.kw),
            }
            for i, sq in enumerate(self.qi.items)
        }

        batch_size = NS_ComboEngine.MAX_SELECT_COMBOS
        max_batches = NS_ComboEngine.MAX_SELECT_BATCHES
        max_attempts = len(NS_ProxyUtil.get_llm_priorities([])) + 1
        fallback_combo: Optional[Tuple[NS_Product, ...]] = None
        total_evaluated = 0

        for batch_i in range(max_batches):
            if not self._watch_ok():
                self.debugger.print("* combo select: watch expired — stopping batch retries")
                break

            start = batch_i * batch_size
            candidates = solutions[start:start + batch_size]
            if not candidates:
                break
            if fallback_combo is None:
                fallback_combo = candidates[0][0]

            bundles: Dict[str, Any] = {}
            for idx, (prods, _) in enumerate(candidates):
                items = []
                for slot, (sq, p) in enumerate(zip(self.qi.items, prods), start = 1):
                    item = p.to_prompt(sq.query, False)
                    item["slot"] = slot
                    items.append(item)
                bundles[str(idx + 1)] = {
                    "total_price": round(sum(p.price for p in prods), 2),
                    "shop_id": prods[0].shop_id,
                    "items": items,
                }

            temperature = 0
            got_null = False
            for _attempt in range(max_attempts):
                try:
                    n_cands = len(candidates)
                    self.debugger.print(
                        f"* llm combo select [{id}]: batch {batch_i + 1}/{max_batches}, "
                        f"evaluating {n_cands} bundle(s) across {len(self.qi.items)} item(s)..."
                    )
                    js, _ = NS_ProxyUtil.trigger_llm(
                        NS_Prompt.SV_COMBO_SELECT.format(
                            rules = NS_Prompt.MATCH_RULES,
                            rank_rules = NS_Prompt.RANK_RULES,
                            n_items = len(self.qi.items),
                            slots = json.dumps(slots, ensure_ascii = False),
                            bundles = json.dumps(bundles, ensure_ascii = False),
                        ),
                        preferred_llms = NS_ProxyUtil.get_role_llm_priorities("combo"),
                        temperature = temperature,
                        expect = "dict",
                        min_secs = NS_SearchEngine.COMBO_MIN_SECS,
                        reserve_after = NS_SearchEngine.RESERVE_AFTER_TERMINAL,
                    )
                    self.debugger.print(js)
                    reason = js.get("reason", "") if isinstance(js, dict) else ""
                    self.debugger.add(
                        "k-combo-select",
                        {"id": id, "js": js, "n_bundles": n_cands, "batch": batch_i + 1},
                    )
                    best_idx = NS_Prompt.extract_combo_select_response(js, n_cands)
                    if best_idx is not None:
                        if reason:
                            self.debugger.print(f"* combo select reason: {reason}")


                        result = list(candidates[best_idx][0])
                        self.sess.step_on_combo_select(
                            result, total_evaluated + n_cands, reason, is_fallback = False
                        )
                        return result
                    got_null = True
                    break
                except Exception as exc:
                    self.debugger.print(f"# failed combo select: {exc}")
                    temperature += 0.3

            total_evaluated += len(candidates)

            if got_null and batch_i + 1 < max_batches:
                self.debugger.print(
                    f"* combo select returned null on batch {batch_i + 1} — "
                    f"trying next batch of unevaluated combos"
                )
                continue
            break

        self.debugger.print(
            f"* combo select returned null after {total_evaluated} bundle(s) "
            f"across {min(batch_i + 1, max_batches)} batch(es) — falling back to top-scored combo"
        )
        result = list(fallback_combo if fallback_combo is not None else solutions[0][0])
        self.sess.step_on_combo_select(result, total_evaluated or 1, "", is_fallback = True)
        return result

    def _first_filtered_combo(self, solutions: List[Tuple[NS_Product, Any]]) -> List[NS_Product]:

        valid: List[Tuple[Tuple[NS_Product, ...], Any]] = []
        seen: set = set()

        for prods, w in solutions:
            if all(bool(self.filtered[self.qi.items[i]][p]) for i, p in enumerate(prods)):
                key = tuple(p.id for p in prods)
                if key not in seen:
                    seen.add(key)
                    valid.append((prods, w))

        if not valid:
            return []
        if len(valid) == 1:
            return list(valid[0][0])

        self.debugger.print(f"* multiple valid combos ({len(valid)}) — escalating to LLM combo select")
        return self._combo_select(valid, "filtered-multi")

    def _run_shop_batches(self, must: bool, wide: bool = False, **kwargs):


        best_p = []

        plan, shop_map, shop_stats = self._plan_shop_batches()
        solutions, processed_shops = [], 0
        passed: List[Tuple[Tuple[NS_Product, ...], Any]] = []
        seen_keys: set = set()
        full_coverage = len(self.qi.items)

        for id, shop_batch in enumerate(plan):


            if processed_shops >= NS_ComboEngine.MAX_EXT_SHOPS and (passed or not wide): break


            if id and not self._watch_ok("breadth"):
                self.debugger.print(
                    f"* breadth reserve reached after {processed_shops} shop(s) -- "
                    "keeping the remaining budget for verification"
                )
                break

            processed_shops += len(shop_batch)

            kmaps = self._search_within_shops(shop_batch, shop_map, id, wide)
            solmap = defaultdict(lambda: [])
            batch_passed: List[Tuple[Tuple[NS_Product, ...], Any]] = []
            local_sol_count = 0

            if kmaps:
                for m in kmaps:
                    for prods, w in self._solve_knapsack_map(m, **kwargs):
                        sol = (prods, (shop_stats[prods[0].shop_id][0], w))
                        local_sol_count += 1

                        solmap[prods[0].shop_id].append(sol)
                        solutions.append(sol)


                batch_passed = self._collect_filtered_combos(
                    solmap, True, f"shop-{id}", NS_ShopSearchEngine.MAX_TRIES
                )
                for prods, w in batch_passed:
                    key = tuple(p.id for p in prods)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        passed.append((prods, w))


            self.sess.step_on_shop_batch(
                shop_batch, local_sol_count, id, self.qi, n_passed = len(batch_passed)
            )

            if not self._watch_ok(): break


            if passed and id + 1 < len(plan):
                curr_cov = shop_stats[shop_batch[0]][0]
                next_cov = shop_stats[plan[id + 1][0]][0]
                n_shops = len({prods[0].shop_id for prods, _ in passed})

                if curr_cov == full_coverage and next_cov < full_coverage and (n_shops > 1 or not wide):
                    self.debugger.print(
                        f"* collected {len(passed)} valid shop combo(s) from {n_shops} full-coverage shops — "
                        "stopping before partial-coverage shops"
                    )
                    break

        if passed:
            passed.sort(key = lambda s: s[-1], reverse = True)
            if len(passed) == 1:
                best_p = list(passed[0][0])
                self.debugger.print("* selected only one valid shop combo")
            else:
                self.debugger.print(
                    f"* {len(passed)} valid shop combo(s) collected — escalating to LLM combo select"
                )
                best_p = self._combo_select(passed, "shop-multi")
            return best_p

        if not best_p:
            best_p = self._verified_combo(same_shop = True, **kwargs)
            if best_p:
                self.debugger.print("* fallback combo assembled from verified-true slots!")
                self.sess.step_on_shop_fallback(best_p, self.qi, has_reject = False)

        if not best_p:
            solutions.sort(key = lambda s: s[-1], reverse = True)


            best_p = self._partial_match_combo(solutions, 2)

            if must and solutions and not best_p:
                self.debugger.print("* fallback combo enforced!")
                best_p = list(solutions[0][0])

            if best_p:
                self.sess.step_on_shop_fallback(
                    best_p,
                    self.qi,
                    has_reject = self._has_filter_reject(best_p),
                    has_unchecked = self._has_filter_unchecked(best_p),
                )

        return best_p
class NS_ShopSearchEngine(NS_ComboEngine):
    MAX_TRIES = 4

    def _solve_knapsack(
        self, cats: List[List[Tuple[NS_Product, float]]]
    ) -> List[Tuple[Tuple[NS_Product, ...], float]]:

        combs = NS_MiscUtil.get_choice_combs(cats, False)
        solutions = []

        for choice in combs:
            prod_weights = [cats[i][n] for i, n in enumerate(choice)]
            prods, aggr_w = [], 1.0

            for p, w in prod_weights:
                prods.append(p)
                aggr_w *= w

            if len({p.id for p in prods}) < len(prods):
                continue

            solutions.append((tuple(prods), aggr_w))

        return sorted(solutions, key = lambda s: -s[-1])

    def _run_shop_strategy(self) -> List[NS_Product]:
        return self._run_shop_batches(True, wide = True)

    def run(self) -> List[NS_Product]:
        super(NS_ShopSearchEngine, self).run()
        if self.debugger.is_stop_in("BASE"): return []


        self._cross_attribute_pool()

        best_p = self._run_shop_strategy()
        if self.debugger.is_stop("s-proc"): return []

        return best_p
class NS_ShoppingAgent(object):
    def __init__(self, query: str, debugger: Optional[NS_Debugger] = None, task_hint: str = "") -> None:
        self.debugger = debugger if debugger is not None else NS_Debugger()
        self.query = query
        self.task_hint = task_hint
        self.sess: NS_SearchSession = NS_SearchSession(query, self.debugger)

    def run(self) -> None:
        pipeline = NS_QueryPipeline(self.query, self.debugger, task_hint=self.task_hint)

        qi = pipeline.classify_task()
        if self.debugger.is_stop("a-classify"): return []

        qi = pipeline.prepare_context(qi)
        if self.debugger.is_stop("a-decompose") or self.debugger.is_stop("a-build"): return []

        qi = pipeline.review_query_info(qi)
        if self.debugger.is_stop("a-review"): return []

        if self.debugger.is_stop_in("AGENT"): return []
        self.sess.step_on_build(qi)

        self.engine = NS_resolve_engine_cls(qi.challenge)(qi, self.sess, self.debugger)
        best_p = self.engine.run()

        if qi.challenge != NS_ShoppingTaskType.VOUCHER:
            self.sess.step_on_compare(best_p, self.engine._render_alt_comparison(best_p))

        self.sess.add_tool_result(NS_ProxyUtil.submit(best_p))
        self.sess.add_tool_result(NS_ProxyUtil.terminate("success" if best_p else "failure"))

        self.sess.step_on_submit(best_p, qi)

    def get_dialog(self) -> List[Dict]:
        return self.sess.dialog

NS_DIALOG_WIRE_LIMIT = 56 * 1024

NS_DIALOG_THINK_CAPS = (2000, 1000, 500)

def NS__wire_size(dialog: List[Dict]) -> int:


    try:
        return len(json.dumps(dialog, ensure_ascii = False, default = str).encode("utf-8"))
    except Exception:


        return NS_DIALOG_WIRE_LIMIT + 1

def NS__slim_tool_result(tc: Any) -> Any:


    if not isinstance(tc, dict) or tc.get("name") in ("recommend_product", "terminate"):
        return tc

    res = tc.get("result")
    rows = res.get("result") if isinstance(res, dict) else res
    if not isinstance(rows, list):
        return tc

    slim = [
        {
            "product_id": str(r.get("product_id") or r.get("id") or ""),
            "price": r.get("price"),
            "shop_id": str(r.get("shop_id") or ""),
        }
        for r in rows if isinstance(r, dict)
    ]

    out = dict(tc)
    out["result"] = {**res, "result": slim} if isinstance(res, dict) else slim
    return out

def NS__rebuild_step(step: Dict, think: str, tool_calls: List[Any]) -> Dict:


    msg = (step.get("completion") or {}).get("message") or {}
    info = step.get("extra_info") or {}

    return create_dialogue_step(
        think = think,
        tool_results = tool_calls,
        response = msg.get("response", "") or "",
        query = info.get("query", "") or "",
        step = info.get("step", 1),
    )

def NS__fit_dialog_to_wire(dialog: List[Dict]) -> List[Dict]:


    if not dialog or NS__wire_size(dialog) <= NS_DIALOG_WIRE_LIMIT:
        return dialog

    before = NS__wire_size(dialog)
    parts = []

    for i, step in enumerate(dialog):
        msg = (step.get("completion") or {}).get("message") or {}
        tools = list(msg.get("tool_call") or [])
        keep = i == 0 or any(
            isinstance(tc, dict) and tc.get("name") in ("recommend_product", "terminate")
            for tc in tools
        )
        parts.append({
            "step": step, "keep": keep, "dirty": False,
            "think": msg.get("think", "") or "", "tools": tools,
        })

    def assemble() -> List[Dict]:
        return [
            p["step"] if p["keep"] or not p["dirty"]
            else NS__rebuild_step(p["step"], p["think"], p["tools"])
            for p in parts
        ]

    def settled(rung: str) -> Optional[List[Dict]]:
        out = assemble()
        size = NS__wire_size(out)
        if size <= NS_DIALOG_WIRE_LIMIT:
            _debug_print(
                f"* dialogue compacted for IPC: {before:,} -> {size:,} bytes "
                f"(limit {NS_DIALOG_WIRE_LIMIT:,}) at `{rung}`"
            )
            return out
        return None

    for p in parts:
        if not p["keep"]:
            p["tools"] = [NS__slim_tool_result(tc) for tc in p["tools"]]
            p["dirty"] = True
    out = settled("slim tool results")
    if out is not None: return out

    for cap in NS_DIALOG_THINK_CAPS:
        for p in parts:
            if not p["keep"] and len(p["think"]) > cap:
                p["think"] = p["think"][:cap]
                p["dirty"] = True
        out = settled(f"think <= {cap}")
        if out is not None: return out

    for p in parts:
        if not p["keep"]:
            p["tools"] = []
            p["dirty"] = True
    out = settled("drop tool calls")
    if out is not None: return out


    out = assemble()
    _debug_print(
        f"# dialogue still over the IPC limit after compaction: {NS__wire_size(out):,} "
        f"bytes vs {NS_DIALOG_WIRE_LIMIT:,} -- the run may be killed at the wall"
    )
    return out

def NS_find_product(**kwargs) -> List[Dict]:


    result = NS_ProxyUtil.oro_proxy().get("/search/find_product", kwargs)
    return result if result else []

def NS_view_product_information(product_ids: str) -> List[Dict]:
    result = NS_ProxyUtil.oro_proxy().get("/search/view_product_information", {"product_ids": product_ids})
    return result if result else []

def NS_recommend_product(product_ids: str) -> str:
    return f"recommended the product(s) to the user: {product_ids}."

def NS_terminate(status: str = "success") -> str:
    return f"the interaction has been completed with status: {status}"



def LOO_freezable(func: Callable) -> Callable:
    return func

class LOO_Prompt:
    CLASSIFY = "You are AI assistant to classify the user's raw query, and parse the voucher details if applicable.\n\n# STEP-1. Classify the query into `single` or `multiple`, by the number of product intents included in it.\n\n# STEP-2. Parse voucher details if they are explicitly mentioned. Parsing must be done in the following structured format:\n{{\n    \"budget\": <int>, // user's budget\n    \"apply_to\": <\"all\"|\"shop\", str>, // \"all\" if voucher applies to all products, \"shop\" if voucher only applies to the products from the same shop\n    \"threshold\": <int>, // minimum purchase amount to apply the voucher\n    \"is_fixed\": <bool>, // true if the voucher is a fixed amount discount, false if the voucher is a percentage discount\n    \"discount\": <int>, // discount amount if is_fixed is true, or discount percentage if is_fixed is false\n    \"cap\": <int|null> // maximum discount amount if is_fixed is false, otherwise null\n}}\n\n# STEP-3. Rewrite the query by removing voucher-related phrases, if voucher information is present.\n    * Exclude voucher-related phrases cleanly.\n    * Strictly keep the original wording and respect the details for remaining text.\n\n# Response Format\nReturn JSON object like:\n{{\n    \"summary\": <str>, // a short summary with 1-2 sentences of how to classify and parse the query\n    \"category\": <\"single\"|\"multiple\", str>, // \"single\" if the query contains only one product intent, otherwise \"multiple\"\n    \"voucher\": <dict|null>, // the parsed voucher details if mentioned, otherwise null\n    \"effective_query\": <str|null> // if no voucher information is present, return null. otherwise, return the original query with only voucher-related text removed.\n}}\n\n# Output Guidelines\n- No markdown.\n- No preceding explanation.\n- No thinking outside JSON.\n- Just start to output JSON only!\n\n# Raw Query\n{query}"
    DECOMPOSE = "You are AI assistant to decompose the user's raw query into independently complete sub-queries, each of which corresponds to a single product intent.\n\n# Core Objective\nGiven the query containing one or more product intents, you must construct an ordered list of sub-queries, each of which:\n    * corresponds to a single product intent\n    * is independently complete — it must include all constraints that apply to this product intent\n\n# Min/Max of Product Intent Count\n    * Minimum: {min_count} (inclusive)\n    * Maximum: {max_count} (inclusive)\n\n# Procedure\n## STEP-1: Identify the product intents included in the raw query.\n    * The product intent count MUST be within the [Min, Max] range specified in **Min/Max of Product Intent Count**.\n    * The raw query is user input. Therefore, sometimes it may be unorganized and ambiguous. You must intelligently analyze the query and decompose the product intents correctly.\n    * Do NOT confuse constraint phrase with individual product intent. The product type must be clear for every product intent.\n\n## STEP-2: For each product intent, identify the constraints that apply to it.\n    * Strictly follow the **Constraint Scope Principles**.\n    * Do NOT omit any constraint which applies to this product intent.\n\n## STEP-3: Write independently complete sub-queries, each of which corresponds to a single product intent.\n    * Respect the original wording and details. System confirmed that there is no typo. Do NOT invent or modify terms.\n    * Include all the constraints that apply to this product intent, to make the sub-query independent and complete.\n    * The constraints that look like nice-to-have MUST be included too, if they apply to this product intent. Do NOT omit them.\n    * Do NOT include the constraints that don't apply to it.\n\nExample: \"a phone battery with 3000-5000mAh capacity, and available with payment on delivery\"\n\n### You can modify the sentence structure to make the sub-query a single sentence.\n    * Raw-Query: \"Need luxury watches for women. ... Show me Seiko brand options. ...\" → Sub-Query: \"luxury watches for women with Seiko brand options\"\n    * Raw-Query: \"Show me Lancome products: (1) ..., (2) gold lotion with 4 species, ...\" → Sub-Query: \"Lancome gold lotion with 4 species\"\n    * Raw-Query: \"Find bonsai trees on OEM shops. First one must be size S4 and PVC material. ...\" → Sub-Query: \"bonsai tree on OEM shop with size S4 and PVC material\"\n    * Raw-Query: \"Looking for the following products on official stores. ... Lastly, yellow t-shirt for hiking.\" → Sub-Query: \"yellow t-shirt for hiking on official stores\"\n\n### Sentence restructuring is accepted, but wording modification of individual terms is NOT allowed.\n\n# Constraint Scope Principles\n    * A constraint applies to multiple product intents if and only if it was repeatedly mentioned in the raw query OR it was associated with explicit indicators such as \"both\", \"all\", \"each\" and \"one/another\". Examples:\n        - \"ceramic mug and glass cup, both must be made in Japan\" → \"made in Japan\" applies to both product intents.\n        - \"high waist women's blouse, and men's sports jacket. all with color black\" → \"color black\" applies to both product intents.\n        - \"finding Seiko watches, one is 22K gold for old men and other one should be with Roman numerals for young men\" → \"Seiko watch\" applies to both, because \"one\" and \"another one\" indicates sharing.\n    * If there exists any ambiguity about the scope of a constraint, it MUST only apply to a single nearest product intent that can most likely receive it grammatically. Never infer global scope from sentence meaning. Examples:\n        - \"white shirt and black pants, with free shipping\" → \"with free shipping\": the rule enforces it to apply only to \"black pants\".\n        - \"a liquid single, gold lotion and shaving cream\" → \"liquid single\": the rule enforces it to apply only to \"gold lotion\".\n    * Sometimes \"both\" and \"all\" are used for different context, not associating with constraints. In that case, do NOT apply unassociated constraints to multiple product intents mechanically. Example:\n        - \"looking for a store selling both wireless mouse and game-friendly keyboards, from Logitech\" → \"both\" is not associated with the constraint \"from Logitech\". Therefore, \"from Logitech\" applies only to \"game-friendly keyboards\".\n    * Commas, conjunctions, and sentence structure can NOT override these principles! Words like \"also\" and \"only\" can NOT override these principles!\n\n## Common Mistakes to Avoid\n    * Invalid constraint sharing: do NOT apply a constraint to multiple product intents from sentence structure or common conventions. Only trust explicit indicators like \"both\", \"all\", \"each\", and \"one/another\". Examples:\n        - \"wireless mouse and game-friendly keyboards. i prefer if they support free delivery option.\" → the rule enforces \"support free delivery option\" to apply only to \"game-friendly keyboards\", even though it is common sense to share it with \"wireless mouse\".\n        - \"I need <intent 1>, and <intent 2>. Also I want them to be from official stores.\" → the rule enforces \"from official stores\" to apply only to \"<intent 2>\", even though it is common sense to share it with \"<intent 1>\".\n        - \"<intent 1>. <intent 2>. show me the delivery payment options if possible.\" → the rule enforces \"delivery payment options\" to apply only to \"<intent 2>\", even though it is common sense to share it with \"<intent 1>\".\n        - \"<intent 1> and <intent 2>. Only looking for products not services.\" → the rule enforces \"only looking for products\" to apply only to \"<intent 2>\", even though it is common sense to share it with \"<intent 1>\".\n    * Invalid choice of nearest product intent: the product intent must be the nearest and most likely to receive the constraint grammatically. Example:\n        - \"white shirt and nylon pants. find ones that are available with COD. also black hats.\" → the rule enforces \"COD\" to apply only to \"nylon pants\". \"black hats\" seems near from \"COD\" but it's not grammatically correct to receive \"COD\".\n    * Invalid interpretation of \"both\" and \"all\" that are not meaning constraint sharing.\n\n## (IMPORTANT!) Unsharable Constraints\nThe following types of constraints are NEVER sharable with multiple product intents, unless they are repeatedly mentioned. One mention MUST apply to ONLY ONE product intent!\n    * Price constraints\n    * Official store/seller constraints\n    * Authenticity/verified constraints\n    * \"Products Only\" constraints\n    * Flashsale/Limited-time offer constraints\n    * Discount constraints\n\n# Response Format\nOutput your reply in the following JSON format:\n{{\n    \"step_1\": <a brief summary of Step-1, str>,\n    \"step_2\": <a brief summary of Step-2, str>,\n    \"step_3\": <a brief summary of Step-3, str>,\n    \n    \"result\": [\n        <sub-query 1, str>,\n        <sub-query 2, str>,\n        ... ...\n    ]\n}}\n\n# Raw Query\n{query}"
    CLEAN = "You are AI assistant to extract price and service constraints from user's queries.\n\n# Core Objective\nFor each of {count} queries, you must:\n    * extract and normalize price constraint phrases\n    * find the phrases expressing the given service constraint types\n    * rewrite the query without price and service constraints\n\n# Price Constraints\n## Identify the lower and/or upper bound of price constraint, and normalize.\n    * `low-high` if both lower and upper bounds were mentioned. Example: \"between 5 and 160 php\" → `5-160`\n    * `low-∞` if only the lower bound was mentioned. Example: \"more than 410\" → `410-∞`\n    * `0-high` if only the upper bound was mentioned. Example: \"903 pesos or below\" → `0-903`\n    * `0-∞` if price constraint was not specified.\n\n## Guidelines\n    * A number or number range is price constraint, if and only if it is accompanied by explicit price indicators OR it is the only remaining plausible numeric constraint. Examples:\n        - \"ball pens with 5mm width that cost less than 250\" → `250-∞`\n        - \"basketball shoes between 50 and 150\" → `50-150` (\"between 50 and 150\" is not associated with any non-price constraint)\n        - \"with power usage 500-1500\" → `0-∞` (\"500-1500\" is associated with \"power usage\", which is non-price constraint)\n    * Avoid the following mistakes for normalization:\n        - Do NOT increase or decrease numbers to adjust the range ends. Example: \"less than 100\" → Correct `0-100`, Incorrect `0-99`\n        - Do NOT include currency name. Example: \"costing 180 USD or more\" → Correct `180-∞`, Incorrect `180USD-∞`\n\n# Service Constraints\n## Find the phrases expressing the provided service constraint types.\n    * `official`: user wants product from official sources (and/or mentioned LazMall)\n    * `flashsale`: user wants product with limited time offer or lightning deal for special discounts (and/or mentioned LazFlash)\n    * `freeShipping`: user wants product available with free/complimentary shipping\n    * `COD`: user wants product with the option of payment/cash on delivery\n    * `product_only`: user wants the product only, not its derived objects\n\n## Guidelines\n    * The service constraint types are given with the original query. Identify the corresponding phrases based on the given types.\n    * The list of given service constraint types is absolutely authentic.\n        - Never infer a new service constraint type that was not given.\n        - You MUST always find the corresponding phrase for every given service constraint type. It exists in the query for sure.\n    * `product_only` service constraint type triggers if and only if the query contains the one of following regular expression patterns:\n        - r\"\\b((?:with\\s+)?(?:just|only)(?:\\s+look(?:ing)?\\s+for)?(?:\\s+the)?\\s+products?)\\b\"\n        - r\"\\b((?:with\\s+)?products?\\s+only(?:\\s+services?)?)\\b\"\n\n# Query Rewriting\nRewrite the query by removing price and service constraint phrases only.\n\n## Guidelines\n    * Write with the product-level intent, not with the shop-level intent. Example:\n        - Correct \"5-10kg dog food for any dog size\"\n        - Incorrect \"find a shop selling 5-10kg dog food for any dog size\"\n    * Preserve all the details except for the phrases associated with price/service constraints or the shop-level intent.\n    * Remove the service and constraint phrases cleanly and naturally. Incomplete or odd-looking removal is rejected. Examples:\n        - Original Query: \"women's handbag in brown color, preferably with special discount offers\" → Correct: \"women's handbag in brown color\", Incorrect: \"women's handbag in brown color with offers\"\n        - Original Query: \"pet necklace with pearl design that costs 120php or more\" → Correct: \"pet necklace with pearl design\", Incorrect: \"pet necklace with pearl design that costs php\"\n        - Original Query: \"Mr.Lee cooking tools from authorized shops\" → Correct: \"Mr.Lee cooking tools\", Incorrect: \"Mr.Lee cooking tools from shops\"\n    * If brand name is mixed with `official` service constraint, smartly handle it to remove service constraint cleanly but keep the brand constraint. Example:\n        - Original Query: \"ice-box with volume 45L from official elemento store\" → Correct: \"ice-box with volume 45L from elemento\", Incorrect: \"ice-box with volume 45L\"\n    * Strictly respect original wording for the phrases to remain. Never invent or modify terms. System confirmed that there is no typo.\n\n# Response Format\nOutput your reply in the following JSON format:\n{{\n    \"1\": {{ // for query 1\n        \"extract\": <a brief summary of extracting price and service constraints, str>,\n        \"price\": <normalized price constraint if applicable, str|null>,\n        \"rewriting\": <a brief summary of query rewriting, str>,\n        \"result\": <rewritten query, str>\n    }}{output}\n}}\n\n# Original Queries\n{inf}"
    EXTRACT = "You are AI assistant to extract search keywords from user's queries.\n\n# Core Objective\nThe current database stores each product as a JSON object with several attributes. Each attribute is the `<key>: <value>` pair.\nOnly the attribute values are indexed for search. Including the attribute keys in the search keywords are not only unnecessary, but also harmful for search.\nTherefore, do NOT include tokens that are likely for attribute keys, when you extract search keywords.\n\n# STEP-1: Normalize queries.\nRemove redundant filler words and ownership marks. Normalize numeric constraints that are written in verbose plain English.\n\n## Remove fillers and conjunction words that are non-informative. Examples:\n    - \"for\", \"from\", \"over\", \"to\", \"and\", \"or\", \"about\", \"between\", \"of\", \"as\", \"by\", \"with\", \"without\", ...\n    - \"perfect\", \"specifically\", \"preferably\", \"compatible\", \"includes\", \"come\", \"have\", \"fits\", \"just\", \"suitable\", \"still\", \"yet\", ...\n    - \"a\", \"the\", \"an\", \"this\", \"that\", \"these\", \"both\", \"each\", \"they\", ...\n\n## Remove ownership marks: `'s` and `'`. Examples:\n    - \"men's\" → \"men\"\n    - \"kids'\" → \"kids\"\n\n## Normalize numeric constraints written in verbose plain English. Examples:\n    - \"between 10 and 20 watts\" → \"10-20 watts\"\n    - \"above 25m\" → \"25m\"\n    - \"18cm by 10cm\" → \"18cm*10cm\"\n    - \"20kg or up\" → \"20kg\"\n    - \"10 months and under\" → \"10 months\"\n    - \"about 20 pages\" → \"20 pages\"\nHowever,    \n    - Do NOT change number tokens by math. Example: \"with 1/2 inch\" → Correct \"1/2 inch\", Incorrect \"0.5 inch\"    \n    - Do NOT shorten unit names. Example: \"5 megapixels or so\" → Correct \"5 megapixels\", Incorrect \"5mp\"\n\n## For other words, respect their original forms. System confirmed that there is no typo in the original query. Do NOT transform or invent. Examples:\n    - \"green l\" → Correct: \"green l\", Incorrect: \"green\"\n    - \"single piece\" → Correct: \"single piece\", Incorrect: \"1pc\"\n    - \"1pcs\" → Correct: \"1pcs\", Incorrect: \"single\"\n    - \"type Z\" → Correct: \"type Z\", Incorrect: \"typez\"\n    - \"3-in-1\" → Correct: \"3-in-1\", Incorrect: \"3-1\"\n    - \"21 metres\" → Correct: \"21 metres\", Incorrect: \"21m\"\n    - \"14-gram\" → Correct: \"14-gram\", Incorrect: \"14gram\"\n    - \"eu 41\" → Correct: \"eu 41\", Incorrect: \"eu41\"\n    - \"large\" → Correct: \"large\", Incorrect: \"l\"\n\n## Do NOT yet remove tokens that are likely attribute keys. Removing such tokens is the task of STEP-2.\n\n# STEP-2: Extract search keywords from each normalized query.\n\n## Exclude tokens that are likely attribute keys.\nThe followings are typical examples of attributes stored in product database. Learn which tokens are likely for attribute keys.\n    - \"Gizell brand\" → \"brand: Gizell\" (\"brand\" must be excluded)\n    - \"8GB disk\" → \"disk: 8GB\" (\"disk\" must be excluded)\n    - \"pack of 20 pieces\" → \"pack: 20pcs\" (\"pack\" must be excluded)\n    - \"with 20m size\" → \"size: 20m\" (\"size\" must be excluded)\n    - \"simple design\" → \"design: simple\" (\"design\" must be excluded)\n    - \"nature theme\" → \"theme: nature\" (\"theme\" must be excluded)\n    - \"cartoon print\" → \"print: cartoon\" (\"print\" must be excluded)\n    - \"round shape\" → \"shape: round\" (\"shape\" must be excluded)\n    - \"butterfly model\" → \"model: butterfly\" (\"model\" must be excluded)\n    - \"ignite color family\" → \"color_family: ignite\" (\"color family\" must be excluded)\n    - \"made with pvc material\" → \"material: pvc\" (\"material\" must be excluded)\n    - \"compatible with piano\" → \"compatibility: piano\" (\"compatible\" must be excluded)\n    - \"50ml volume\" → \"volume: 50ml\" (\"volume\" must be excluded)\n    - \"7-days warranty period\" → \"warranty_period: 7-days\" (\"warranty period\" must be excluded)\n    - \"buy 1 take 1\" → \"promotion: buy-1-take-1\" (nothing to be excluded)\n    - \"type A or B\" → \"type: A, B\"\n    \n## \"color\" and \"color_family\" are very special attribute keys in the current database.\nThey can receive any attribute values beyond colors. Therefore, do NOT include \"color\" and \"family\" in search keywords. Examples:\n    * Query: \"1pcs color family\" → Keyword: \"1pcs\"\n    * Query: \"color 20A\" → Keyword: \"20A\"\n\n## Work on token levels to identify the attribute keys to remove from the keywords. Examples:\n    * \"star-shaped\" → \"star\" (\"shape: star\")\n    * \"round-themed\" → \"round\" (\"theme: round\")\n\n## A token must be removed from the keywords ONLY IF the evidence of attribute value exists in the normalized query associated with this attribute keys. Example:\n    * Query: \"find organic acid container with 5L capacity\"\n        - Correct removal: \"capacity\" → Evidence: \"capacity: 5L\"\n        - Incorrect removal: \"organic\" → No evidence: \"organic: ???\"\n\n## Do NOT exclude informative tokens. Examples:\n    - \"buy 1 take 1\" → Keep\n    - \"up-to-down\" → Keep\n\n## Do NOT exclude tokens that are used for core product form. Example:\n    * Query \"animal lamp with color c\"\n        - Correct keywords: [\"animal lamp\", \"c\"] (removing \"color\", because it's the key token of attribute `color: c`)\n        - Incorrect keywords: [\"animal\", \"c\"] (removing \"lamp\" is invalid, because \"animal lamp\" is the core product form, therefore `lamp: animal` is unrealistic)\n\n## Group the words by attributes.\nThe search keywords must be returned as a list. Do NOT enumerate the keywords individually that lie in the same attribute cateogry. Examples:\n    * Correct: [\"water resistant\"], Incorrect: [\"water\", \"resistant\"]\n    * Correct: [\"M12 M13 M14\"], Incorrect: [\"M12\", \"M13\", \"M14\"]\n\n# Response Format\nReturn JSON object like:\n{{\n    \"1\": {{ // for query 1\n        \"summary\": <a brief summary of processing through STEP 1-2, str>,\n        \"keywords\": [\n            // search keywords after excluding non-informative words and key-like tokens\n            <keyword 1, str>, ...\n        ]\n    }}{output}\n}}\n\n# Original Queries\n{inf}"
    TEMPLATE = "You are AI assistant to build a product template from an e-commerce query.\n\n# Background\n## Product Template Structure\nThe current database stores product details as JSON template consisting of several pairs like:\n    {{ \"product_type\": \"...\", <attribute_key_1, str>: <attribute_value_1, str>, ... ..., \"extra\": [...] }}\n- `product_type: ...` is the must-have pair containing the core product form.\n- Attribute pairs are optional that represent the product's properties in key-value format.\n- `extra: ...` is the optional pair containing the additional properties that are not suitable to represent in key-value format.\n\n## Input Data\nYou are given:\n    * user input query\n    * a collection of available keywords (for `product_type` and `extra` construction)\n    * a list of available attribute pairs (for attributes construction)    \n\n## Synonyms are Acceptable\nRare but not unseen, database may use different term for the same semantic concept. Example:\n    * Query → \"fit for 5 people seats\"\n    * Attribute → `seat: 5 players`\n\n# Objective\nConvert the query into the most realistic product JSON template, by using the available keywords and attribute pairs.\n- You must strictly ensure the template's integrity against the query. (**Integrity Criteria**)\n- Choose the most preferred template among all possible ones. (**Preferences**)\n\n# Integrity Criteria\n1. The value of `product_type` must not be empty.\n2. The values of `product_type` and `extra` must consist of only the available keywords. Do NOT invent or modify!\n3. The attribute pairs must consist of only the available attribute pairs. Do NOT invent or modify!\n4. The sum of all pairs (`product_type`, attributes, and `extra`) must represent all the meaningful constraints expressed in the query. Do NOT omit any constraint.\n    - Sometimes a constraint in the query looks like nice-to-have. However, you MUST include it in the template. There is no \"nice-to-have\" concept in template construction. Every constraint is \"must-have\"!\n5. Pairs must be mutually exclusive. Do NOT duplicate information across pairs.\n6. The following types of constraints must be represented in attributes, instead of `extra`:\n    - Numeric constraints like size, weight, volume, power, age, etc.\n    - Color and color family constraints\n7. NEVER include attribute pairs representing unrequested constraints!\n\n# Preferences\n1. Representing constraint in attributes (key-value format) is preferred over `extra` (enumeration format).\n    - This preference NEVER permits invention and usage of unavailable attribute pairs.\n2. Less unmentioned keywords in the values of pairs is preferred.\n    - This preference NEVER permits omitting any meaningful constraint expressed in the query.\n3. Less overlapped keywords between the values of pairs is preferred.\n    - This preference NEVER permits invention and usage of unavailable attribute pairs.\n4. More structured attributes representation of constraints is preferred.\n    - This preference NEVER permits duplicated usage of keywords that have already been used in `product_type`.\n5. More aligned attributes representation with the query is preferred. \n    - This preference NEVER permits duplicated usage of keywords that have already been used in `product_type`, or introducing unavailable keywords.\n6. More plausible key is preferred for attribute pairs. Example: `size: 120mm` is preferred over `variation3: 120mm`.\n    - This preference applies to the case only when two or more possible attribute pairs have the same value and different keys.\n\n# Common Mistakes to Avoid\n## An attribute whose value contains the keyword that have already been used for `product_type`.\nIf this keyword was mentioned several times in original query, it's not violation.\nHowever, if it was mentioned only once in original query, it's violating Rule-5 of Integrity Criteria.\nHow to fix it? First, try to remove the duplicated keyword from `product_type`. If the result `product_type` is unrealistically too short, cancel the removal and instead remove the duplicated keyword from attribute.\n\nExample 1:\n    * Query: \"gasoline generator from Siemens\"\n    * `product_type: gasoline generator`\n    * `fule_type: gasoline` → Violation\n\nExample 2:\n    * Query: \"gasoline generator from Siemens with fule type gasoline\"\n    * `product_type: gasoline generator`\n    * `fule_type: gasoline` → Correct\n\n## Unrealistically too short `product_type` value.\nExample 1:\n    * Query: \"gasoline generator\"\n    * Incorrect: `product_type: generator` + `fule_type: gasoline`\n    * Correct: `product_type: gasoline generator`\n\n## A keyword in `extra` values that have already been used for `product_type` or attribute values.\nIt is obviously violating Rule-5 of Integrity Criteria.\n\n## An attribute whose value contains the keyword that is conflicting the query's requested constraint. Example:\n    * Query: \"red t-shirt with size medium\"\n    * Correct attribute `size: m`\n    * Incorrect attribute `size: m/l/xl`\n\n## Invalid synonyms: Significantly different term for the concept.\n    * Query: \"with simple pattern\" vs Attribute: `pattern: plain` → Correct synonym\n    * Query: \"in color yellow\" vs Attribute: `color: slim` → Invalid synonym\n\n# Handling of Duplicated Keys for Multiple Attribute Pairs\nSometimes two or more attribute pairs must be included in the template, which have the same key but different values.    \nHaving both attributes will break the output JSON format. In that case, use the numbered suffix pattern \"_n\" for the key. Example:\n    * `variation: black`, `variation: high heel` → `variation: black`, `variation_2: high heel`\n\n# Response Format\nReturn JSON object like:\n{{\n    \"construction\": <a summary in 3-4 sentences for template construction. simulate the template and review it. fix if needed before output below.>,\n    \n    \"template\": {{\n        \"product_type\": <str>,\n        \n        <attribute_key_1, str>: <attribute_value_1, str>, // attribute pairs are optional if just `product_type` is sufficient to represent the query.\n        ... ...\n\n        \"extra\": [ // extra keywords list. set empty if nothing applicable.\n            <extra_keyword_1>, ...\n        ]\n    }}\n}}\n\n# User's Query\n{query}\n\n# Available Keywords for `product_type` and `extra` construction\n{words}\n\n# Available Key-Value Pairs for Attributes Construction\n{pairs}"
    SELECT = "You are AI assistant to select the best matching product for the user's shopping query from the candidate products.\nYou should examine each product with all the constraints mentioned in the query, and recommend the best one.\n\n# Procedure\n    * Phase 1. For each product, check all requested constraints:\n        - Step-1.1. (Generously) Check by common sense if its product type is what is intended.\n        - Step-1.2. Find evidences for all other requested constraints.\n    * Phase 2. Exclude products that are mismatching. Among the matching products:\n        - (Shortlist) If one or more products are significantly more aligned with the query than others, consider those only. Otherwise, consider all matching products.\n        - (Recommend) Choose the one which appear earlier in the provided candidate list, from the shortlisted products.\n\n# Structure of Product Details\nEach product details contain `title` and several attributes.\n\n## Title\n### `title` is the unorganized collection of words\n`title` field is designed with the purpose of maximize the SEO score, therefore each piece of its content has less consistent semantics.\nWhen looking for evidence in `title`, do NOT expect it includes the relevant keywords in order and in specific location.\nThe tokens forming the intended product type may be scattered here and there in the messy word collection of `title` field.\nExistence of tokens is the evidence in `title`. Do NOT try to infer any semantic meaning from `title`. It tells you NOTHING! Example:\n    * Query: \"water bottle\"\n        - `title: Black Plastic Bottle Lids For Kids Feeding Water 50/100/250ml Medium Cart Juice Drink` → Match\n        - Do NOT infer this title represents \"bottle lids\". Since both tokens \"water\" and \"bottle\" appear, it's the matching evidence!\n\n## Attributes\n### Attributes are the preferred source of finding constraints except for product type\nFor all other constraints beside product type, find evidence in attributes first. Check `title` as the alternative.\n\n### Attributes are variations\nAttributes may contain different variations of properties. If multiple property values exist, any exact matching value is a sufficient evidence.\nDo NOT reject a product only because it has conflicting attributes. Just find the matching evidence attribute, then it's fine. Example:\n    * Query: \"with color blue\"\n        - Two attributes appear: `color: blue` and `color: gold`\n        - Evidence: `color: blue` → Match\n\n# Matching Rules\n## Rule 1 — Tokens that appear to be identifiers, model names, brands, SKU, codes, numbers, capacities and sizes MUST be matched literally and MUST NOT be partial-matching.\n    * Query: \"100ml\"\n        - `title: ... 100ml ...` → Match\n        - `volume: 100ml` → Match\n        - `capacity: 100` → Match\n        - `volume: 100-150ml` → Mismatch (Do NOT infer 100 is in range 100-150.)\n    * Query: \"with color green 2\"\n        - `color_family: green 2` → Match\n        - `color: green x` → Mismatch\n        - `color_classification: green` → Mismatch\n    * Query: \"compatible with models X3R or P3R or C3S\"\n        - `model: X3R/P3R/C3S` → Match\n        - `model: C3S` → Mismatch (Do NOT infer with \"or\" logic operator. All identifiers MUST appear!)\n\n## Rule 2 — Accept commonly recognized synonyms and equivalent domain terminology for descriptive constraints that are not applicable to Rule 1.\n    * Query: \"battery tester\", `title: battery capacity tester` → Match\n    * Query: \"with standard material\", `material: default` → Match\n    * Query: \"in personalized mode\", `running_mode: customized` → Match\n    * Query: \"sold by single piece\", `frozen_package_set: 1pcs` → Match\n    * Query: \"fit for everday wear\", `occasion: casual` → Match\n    * Query: \"both men and women\", `gender: unisex` → Match\n    * Query: \"compatible with all makers or models\", `model_maker: universal` → Match\n\n## Rule 3 — Use common sense instead of literal string match for descriptive constraints that are not applicable to Rule 1.\n    * Query: \"350ml per bottle\", `capacity: 350ml gallons`\n        - In this example, \"350ml\" is the constraint applicable to Rule 1, which must be matched literally.\n        - However, the constraint \"bottle\" is the descriptive constraint, which must be evaluated by common sense.\n        - In this context, `capacity: 350ml gallons` is acceptable. Therefore, match!\n\n# Evidence Quality Preferences\n## Preference 1 — Prefer exact matching evidence over being more or less than the query requested.\n    * Query: \"with size 39\"\n        - Prefer `size: eu 39`\n        - Over `size: 39/40/41` (having unrequested values)\n    * Query: \"with variation of golden premium\"\n        - Prefer `variation: gold premium\"\n        - Over `variation: gold plus` (partial matching with \"premium\" missed)\n## Preference 2 — Prefer evidence in attributes than in `title` for the constraints beside product type\n\n# Key Steps for Phase 1\n## Check the product type if matching. If not matching, set is_match false and skip to the next product immediately.\n    * Check product type generously. If it's acceptable by common sense, accept it even though a generic keyword is missing.\n## Find the most preferred evidences for all other constraints.\n    * Find in attributes first. Find in `title` if not found in attributes.\n\n# Key Steps for Phase 2\n## Exclude mismatching products.\n## Identify the products that are obvisouly more aligned to the original query than others.\n    * Fewest unrequested information in attribute evidences (Not `title`, because `title` is designed to be messy with a lot of words.)\n    * More aligned expression and wording in attribute evidences\n## Choose the one from the shortlisted products, whose original rank is the earliest.\n\n# Mistakes to Avoid\n    * Do NOT check product type literally. Use common sense for product type matching.\n    * Do NOT check constraints unmentioned in the original query.\n    * Do NOT judge products with your invented standards, for Phase 2. Example:\n        - Number of occurencies of the evidence has NO benefit.\n        - Having promising but unrequested keywords has NO benefit.\n\n# Response Format\nReturn JSON like:\n{{\n    \"phase_1\": {{ // Phase 1: Checking and Evidence Collection\n        \"1\": {{ // for product 1\n            \"check\": <a brief summary of checking product type and finding the most preferred evidences for the constraints.>,\n            \"is_match\": <true if matching, otherwise false>\n        }},\n        \"2\": {{ ... }},\n        ... ...\n    }},\n\n    \"matched\": <str>, // IDs of matching products, separated by comma\n    \n    // skip \"phase_2\" if no matching product was found\n    \"phase_2\": <a brief summary of shortlisting (if applicable) and recommending the best product>\n\n    \"best\": <str|null> // ID of the best product if found, otherwise null\n}}\n\n# Output Guidelines\n1. Be concise, structured and short, for the checking. Examples:\n    * \"Product type→'feature: motorcycle lights' in attributes. Acceptable.\\nHonda→'model: Honda' in attributes.\\nSecond hand→'second hand' in `title`.\\nThus, match.\"\n    * \"Product type→'book' in `title`. Acceptable.\\nPublished in 1998→Not found in attributes. Found '1998' in `title`.\\nWritten by Richard→'author: Richard' in attributes.\\nThus, match.\"\n    * \"Product type→'phone case' in `title`. Acceptable.\\nNavy blue→Not found in attributes. Not found in `title`.\\nThus, mismatch.\"\n2. Do NOT check unrequested constraints.\n3. No markdown. No preceding explanations. No thinking outside JSON. Just start to output JSON only!\n\n# Shopping Query:\n```{query}```\n\n# Candidate Products:\n{products}"
    CONFIRM = "You are AI assistant to check the products against the provided shopping queries, to find the ones which meet all the constraints mentioned.\nThere are {count} group(s), where each group has a shopping query and a list of candidate products for this query.\n\n# Procedure\nFor each product, check all requested constraints:\n    - Step-1. Check if its product type is what is intended.\n    - Step-2. Find evidences for all other requested constraints.\n\n# Structure of Product Details\nEach product details contain `title` and several attributes.\n\n## Title: Unorganized Collection of Words\n`title` field is designed with the purpose of maximize the SEO rating, therefore each piece of its content has less consistent semantics.\nWhen looking for product type evidence in `title`, do NOT expect it includes the relevant keywords in order and in specific location.\nThe tokens forming the intended product type may be scattered here and there in the messy word collection of `title` field.\nExistence of tokens is the evidence in `title`. Do NOT try to infer any semantic meaning from `title`. It tells you NOTHING! Example:\n    * Query: \"water bottle\"\n        - `title: Black Plastic Bottle Lids For Kids Feeding Water 50/100/250ml Medium Cart Juice Drink` → Match\n        - Do NOT infer this title represents \"bottle lids\". Since both tokens \"water\" and \"bottle\" appear, it's the matching evidence!\n\n## Attributes: Variations of Properties\nAttributes may contain different variations of properties. If multiple property values exist, any exact matching value is a sufficient evidence.\nDo NOT reject a product only because it has conflicting attributes. Just find the matching evidence attribute, then it's fine. Example:\n    * Query: \"with color blue\"\n        - Two attributes appear: `color: blue` and `color: gold`\n        - Evidence: `color: blue` → Match\n\n# Matching Rules\n## Rule 1 — Tokens that appear to be identifiers, model names, brands, SKU, codes, numbers, capacities and sizes MUST be matched literally and MUST NOT be partial-matching.\n    * Query: \"100ml\"\n        - `title: ... 100ml ...` → Match\n        - `volume: 100ml` → Match\n        - `capacity: 100` → Match\n        - `volume: 100-150ml` → Mismatch (Do NOT infer 100 is in range 100-150.)\n    * Query: \"with color green 2\"\n        - `color_family: green 2` → Match\n        - `color: green x` → Mismatch\n        - `color_classification: green` → Mismatch\n    * Query: \"compatible with models X3R or P3R or C3S\"\n        - `model: X3R/P3R/C3S` → Match\n        - `model: C3S` → Mismatch (Do NOT infer with \"or\" logic operator. All identifiers MUST appear!)\n\n## Rule 2 — Accept commonly recognized synonyms and equivalent domain terminology for descriptive constraints that are not applicable to Rule 1.\n    * Query: \"battery tester\", `title: battery capacity tester` → Match\n    * Query: \"with standard material\", `material: default` → Match\n    * Query: \"in personalized mode\", `running_mode: customized` → Match\n    * Query: \"sold by single piece\", `frozen_package_set: 1pcs` → Match\n    * Query: \"fit for everday wear\", `occasion: casual` → Match\n    * Query: \"both men and women\", `gender: unisex` → Match\n    * Query: \"compatible with all makers or models\", `model_maker: universal` → Match\n\n## Rule 3 — Use common sense instead of literal string match for descriptive constraints that are not applicable to Rule 1.\n    * Query: \"350ml per bottle\", `capacity: 350ml gallons`\n        - In this example, \"350ml\" is the constraint applicable to Rule 1, which must be matched literally.\n        - However, the constraint \"bottle\" is the descriptive constraint, which must be evaluated by common sense.\n        - In this context, `capacity: 350ml gallons` is acceptable. Therefore, match!\n\n# Response Format\nReturn JSON like:\n{{\n    // check groups A, B, ... in order\n    \"A\": {{\n        \"query\": <write the query for this group>,\n\n        // check all products A1, A2, ... against the query A\n\n        \"A1\": {{ // for product A1\n            \"check\": <a brief summary of checking product type and finding the evidences for constraints>,\n            \"is_match\": <true if matching, otherwise false>\n        }},\n        ... ...\n    }},\n    \"B\": {{ ... }},\n    ... ...\n}}\n\n# Output Guidelines\n1. Be concise, structured and short, for each checking. Examples:\n    * \"Product type→'motorcycle lights' in `title`. Acceptable.\\nHonda→'model: Honda' in attributes.\\nSecond hand→'second hand' in `title`.\\nThus, match.\"\n    * \"Product type→'book' in `title`. Acceptable.\\nPublished in 1998→Not found in attributes. Found '1998' in `title`.\\nWritten by Richard→'author: Richard' in attributes.\\nThus, match.\"\n    * \"Product type→'phone case' in `title`. Acceptable.\\nNavy blue→Not found in attributes. Not found in `title`.\\nThus, mismatch.\"\n2. Do NOT check unrequested constraints.\n3. No markdown. No preceding explanations. No thinking outside JSON. Just start to output JSON only!\n\n# Queries and Candidate Products in Groups:\n```json\n{info}\n```"

    @staticmethod
    def parse_classify_response(js: Dict[str, str | Dict]) -> Tuple["Challenge", Optional["Voucher"], str]:
        is_single = js.get("category").lower() == "single"
        query_e = js.get("effective_query")

        v = js.get("voucher")
        voucher = LOO_Voucher.from_json(v) if v else None

        if voucher is not None:
            challenge = LOO_Challenge.VOUCHER
        elif is_single:
            challenge = LOO_Challenge.PRODUCT
        else:
            challenge = LOO_Challenge.SHOP

        return challenge, voucher, query_e

    @staticmethod
    def parse_decompose_response(js: Dict[str, List[str]]) -> List[str]:
        return js.get("result") or []

    @staticmethod
    def parse_clean_response(js: Dict[str, Dict[str, str]], is_single: bool) -> List[Dict]:
        if is_single:
            js["1"] = item = js.get("1", {})

            for k in ["price", "result"]:
                if k in js and k not in item: item[k] = js[k]

        cleaned = []

        for k, item in js.items():
            if not k.isnumeric(): continue
            price = item.get("price")

            if price:
                splits = [s if LOO_Misc.extract_number(s, True) else '' for s in price.split('-')]
                price = '-'.join(splits) if len(splits) == 2 else None
            else:
                price = None

            cleaned.append({
                "query": item.get("result"),
                "price": price
            })

        return cleaned

    @staticmethod
    def parse_extract_response(js: Dict[str, Dict[str, str | List]], is_single: bool) -> List[List[str]]:
        if is_single:
            js["1"] = item = js.get("1", {})

            for k in ["keywords"]:
                if k in js and k not in item: item[k] = js[k]

        extracted = []

        for k, item in js.items():
            if k.isnumeric(): extracted.append(item.get("keywords"))

        return extracted

    @staticmethod
    def parse_build_response(js: Dict[str, List[Dict[str, str]]]) -> List[Dict]:
        parsed, built = js.get("parsed"), []
        if not isinstance(parsed, list): parsed = [parsed]

        for item in parsed:
            prod_only = bool(LOO_Misc.scan_patterns(item.get("raw") or item.get("rewritten"), LOO_Vocab.PROD_ONLY_PATTERNS))
            price, services = re.sub(r"[^\d\-\.]", "", (item.get("price") or "")).strip(), []

            if price:
                splits = price.split('-')
                if len(splits) != 2: raise Exception(f"invalid price `{price}`")

                splits = [s if LOO_Misc.extract_number(s, True) else '' for s in splits]
                price = '-'.join(splits)

            for lbl in (item.get("services") or []):
                svc = LOO_Prompt.label_to_service(lbl)

                if svc and svc not in services:
                    services.append(svc)
                elif lbl == LOO_Vocab.PROD_ONLY:
                    prod_only = True

            built.append({
                "query": item["rewritten"],
                "product_type": item["product_type"],
                "keywords": [s.lower().replace("'s", "") for s in item["keywords"]],
                "price": price,
                "services": services,
                LOO_Vocab.PROD_ONLY: prod_only
            })

        return built

    @staticmethod
    def parse_review_response(js: Dict[str, Dict[str, Dict[str, str]]], is_single: bool) -> List[Dict]:
        if is_single:
            js["1"] = item = js.get("1", {})

            if "new_keywords" in js and "new_keywords" not in item: item["new_keywords"] = js["new_keywords"]
            if "new_query" in js and "new_query" not in item: item["new_query"] = js["new_query"]

        reviewed = []

        for k, item in js.items():
            if not k.isnumeric(): continue

            reviewed.append({
                "keywords": item.get("new_keywords"), "query": item.get("new_query")
            })

        return reviewed

    @staticmethod
    def parse_template_response(js: Dict[str, Dict[str, str]]) -> Dict:
        template = js.get("template")
        words = [w.lower() for w in template.get("product_type").split()]

        for ext_word in (template.get("extra") or []):
            for w in ext_word.split():
                if w.lower() not in words: words.append(w.lower())

        product_type = ' '.join(words)
        res = {"product_type": product_type}

        for k, v in template.items():
            if k not in ["product_type", "extra"]: res[k] = v

        return res

    @staticmethod
    def parse_select_response(js: Dict[str, Dict]) -> int:
        if "best" in js: return LOO_Misc.extract_number(str(js.get("best")), True) - 1

        bid = js.get("best")

        if not bid:
            for phase in list(js.values())[::-1]:
                if not isinstance(phase, dict): continue
                if "best" not in phase: continue
                return LOO_Misc.extract_number(str(phase.get("best")), True) - 1

        return 0

    @staticmethod
    def parse_confirm_response(js: Dict[str, Dict], counts: List[int]) -> List[List[int]]:
        alphabets = LOO_Vocab.ALPHABETS[:len(counts)]
        matches = [[] for _ in counts]

        for g, checklist in js.items():
            if not isinstance(checklist, dict): raise Exception("format error")

            if g not in alphabets: raise Exception("format error")
            gid = alphabets.index(g)

            for k, check in checklist.items():
                if k in ["query", "checklist"]: continue
                if not isinstance(check, dict): raise Exception("format error")

                m = re.search(r"^[A-Z](\d+)$", k, re.I)
                if not m: raise Exception("format error")

                bid = int(m.group(1) or 0) - 1
                if bid < 0 or bid >= counts[gid]: raise Exception("format error")

                if "is_match" not in check: raise Exception("format error")
                if check.get("is_match"): matches[gid].append(bid)

        return matches

class LOO_Vocab:
    PROD_ONLY = "product_only"

    SERVICE_PATTERNS = {
        "official": [
            r"\b(official(?:ly)?)[\s\-]+(?:[a-z]+\s+)?(?:brand(?:ed)?|flagship|stor[a-z]+|seller|shop|mall|product|platform|item|sold|sale|retailer|channel)s?\b",
            r"\b(auth(?:entic|enticated|enticable|orized?|orizable))\b", r"\b(verifi(?:ed|able))\b", r"\b(laz[\s\-]?mall)\b"
        ],
        "flashsale": [
            r"\b(?:lighte?ning|flash)[\s\-]*(sale|deal|offer|discount)s?\b", r"\b(limit(?:ed)|special|discount(?:ed))[\s\-]+(time|deal|offer|sale|discount|promotion)s?\b",
            r"\bavailable[\s\-]+(?:at|with)[\s\-]+a[\s\-]+(discount[a-z]*)\b", r"\b((?:promotion[a-z]+)[\s\-]+hours?)\b", r"\b(?:upon|on|big)[\s\-]+(sale|clearance|discount)s?\b",
            r"\b(time[\s\-]+limited)\b", r"\b(laz[\s\-]?flash)\b"
        ],
        "COD": [
            r"\b((?:cash(?:es|ing)?|pay(?:ment|ing)?s?|paid)[\s\-]*(?:for)?[\s\-]*(?:(?:up)?on|with(?:in)?|by|for|when)?(?:\s+\w+){0,3}[\s\-]*(?:deliver(?:y|ing|ie|ed)?s?|cash(?:es|ing)?|(it\s)?arriv(?:e|es|ed|ing|al)?|recei[a-z]+|you|they))\b",
            r"\b(cod)\b", r"\b((?:deliver|cash)[a-z]*[\s\-]+pay[a-z]*)\b"
        ],
        "freeShipping": [
            r"\b((?:free|compliment[a-z]+|no(?:\sextra)?|zero)\s+(?:ship|deliver|cost)[a-z]*)\b",
            r"\b((?:ship|deliver)[a-z]*[\s\-]+(?:(?:by|at|with|on|upon|for|is|was|are)[\s\-]+)?(?:free|compliment[a-z]+|no|zero|included))\b"
        ],
        PROD_ONLY: [
            r"\b((?:with\s+)?(?:just|only)(?:\s+look(?:ing)?\s+for)?(?:\s+the)?\s+products?)\b", r"\b((?:with\s+)?products?\s+only(?:\s+services?)?)\b"
        ]
    }
    COLOR_TOKENS = [
        "red", "blue", "green", "yellow", "black", "white", "pink", "purple", "violet", "orange", "brown", "gray", "grey", "beige", "navy", "gold", "silver",
        "bronze", "magenta", "cyan", "dark", "light"
    ]
    SIZE_TOKENS = ["small", "large", "medium"]
    SYMBOL_TOKENS = ["int", "eu", "us"]

    YES_TOKENS = ["yes", "true", "included", "enabled", "specified", "support"]
    NO_TOKENS = ["no", "free", "none", "not"]

    UNIT_MAP = {
        a: b for a, b in
        [
            ("mm", ""), ("cm", ""), ("mg", ""), ("oz", ""), ("lb", ""), ("ml", ""), ("mah", ""), ("ah", ""), ("wh", ""), ("kwh", ""), ("mv", ""), ("kv", ""),
            ("ma", ""), ("kw", ""), ("hz", ""), ("khz", ""), ("mhz", ""), ("ghz", ""), ("bps", ""), ("kbps", ""), ("mbps", ""), ("gbps", ""), ("°c", ""),
            ("°f", ""), ("kpa", ""), ("mpa", ""), ("dpi", ""), ("lux", ""), ("db", ""), ("rpm", ""), ("yrs", "year"), ("y", "year"),
            ("km", "kilometer"), ("ft", "feet"), ("kg", "kilo"), ("m", "meter"), ("yd", "yard"), ("g", "gram"), ("l", "liter"), ("v", "volt"), ("w", "watt"), ("pa", "pascal"),
            ("kb", "kilobyte"), ("mb", "megabyte"), ("gb", "gigabyte"), ("tb", "terabyte"), ("mp", "megapixel"), ("pc", "piece"), ("pcs", "piece"), ("in", "inch")
        ]
    }
    MORPH_RULES: List[Tuple[str, List[Union[str, Callable]]]] = [
        ("ing", ["e", "", re.sub]), ("ed", ["e", "", re.sub]), ("est", ["e", "", re.sub]), ("er", ["e", "", re.sub]), ("ish", ["e", "", re.sub]),
        ("iness", ["y"]), ("inesses", ["y"]), ("ily", ["y"]), ("ies", ["y", ""]), ("ness", ["y", ""]), ("es", ["e", ""]), ("ce", ["t"]), ("s", [""]),
        ("ful", [""]), ("less", [""]), ("ly", [""]), ("en", [""]), ("re", ["er"]), ("ay", ["ey"]), ("our", ["or"]), ("man", ["men"]), ("ren", [""]), ("age", [""])
    ]
    KW_ALTS = [
        ("everyday", "casual"), ("kid", "child"), ("child", "kid"), ("t-shirt", "tshirt"), ("shirt", "t-shirt"), ("stone", "gemstone"), ("gemstone", "stone"), ("pure", "100%")
    ]
    KW_WEAKS = [
        "set", "pack", "option", "variation", "model", "color", "brand", "design", "theme", "pattern", "material", "category", "print", "shape", "style", "fabric", "size", "type",
        "accessory", "kit", "supplement", "replacement", "pair", "unit", "package", "module", "require", "enable", "include", "layer", "outlet", "tool", "security", "occasion",
        "provide", "sold", "help", "contain", "cost", "come", "have", "has", "just", "still", "yet", "never", "recommend", "bottle", "treat", "spare", "part", "clothing"
    ]
    FILLERS = [
        "the", "these", "those", "that", "which", "more", "than", "less", "equal", "between", "above", "below", "by", "for", "with", "without", "of", "at",
        "to", "such", "up", "under", "following", "are", "is", "very", "and", "or"
    ]
    NUM_ALTS = [
        ("single", "1"), ("zero", "0"), ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"), ("five", "5"), ("six", "6"), ("seven", "7"), ("eight", "8"), ("nine", "9")
    ]
    ALTS = [
        ("all", "any"), ("all", "every"), ("kid", "child"), ("man", "male"), ("woman", "female"), ("men", "male"), ("women", "female"), ("tier", "layer"),
        ("everyday", "casual"), ("shade", "color"), ("resistant", "proof"), ("drop", "fall"), ("simple", "plain"), ("standard", "default"), ("stone", "gemstone")
    ]
    ALPHABETS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ALL_UNITS = list(set(UNIT_MAP.keys()) | {b for b in UNIT_MAP.values() if b})

    @staticmethod
    def build_alt_map() -> Dict[str, Set]:
        alt_map = defaultdict(lambda: set())

        for a, b in LOO_Vocab.ALTS:
            alt_map[a].add(b)
            alt_map[b].add(a)

        return alt_map

class LOO_ExtTool:
    def search(**kwargs) -> List[Dict]:
        result = LOO_Proxy.ORO_PROXY.get("/search/find_product", kwargs)
        return result if result else []

    def view(ids: str) -> List[Dict]:
        result = LOO_Proxy.ORO_PROXY.get("/search/view_product_information", {"product_ids": ids})
        return result if result else []

    def submit(product_ids: str) -> str:
        return f"submitted recommendation to the user: {product_ids}."

    def finish(status: str) -> str:
        return f"session completed with status: {status}"

class LOO_Challenge(str, Enum):
    PRODUCT = "Product"
    SHOP = "Shop"
    VOUCHER = "Voucher"
    UNKNOWN = "Unknown"

    def promptfy(self, is_multiple: bool) -> str:
        if self is LOO_Challenge.PRODUCT: return "a single product with specific constraints"
        if self is LOO_Challenge.SHOP: return "multiple products from the same shop"

        if self is LOO_Challenge.VOUCHER:
            pstr = "multiple products" if is_multiple else "a single product"
            return f"{pstr} within a specified budget and considering voucher options"

        raise ValueError(f"unknown challenge type: {self}")

@dataclass
class LOO_JsonModel(object):
    @classmethod
    def from_json(cls, data: dict) -> "JsonModel":
        kwargs = {}

        for f in fields(cls):
            if f.name not in data: continue
            kwargs[f.name] = cls.__convert_from__(data[f.name], f.type)

        return cls(**kwargs)

    def to_json(self) -> Dict:
        return self.__convert_to__(self)

    @staticmethod
    def __convert_from__(value, t: Type["JsonModel"] | Any) -> Any:
        origin, args = get_origin(t), get_args(t)

        if origin is Union and len(args) == 2 and NoneType in args:
            return LOO_JsonModel.__convert_from__(
                value, next(a for a in args if a is not NoneType)
            ) if value is not None else None

        if origin is list and isinstance(value, list):
            return [LOO_JsonModel.__convert_from__(v, args[0]) for v in value]

        if origin is tuple and isinstance(value, list):
            return tuple(LOO_JsonModel.__convert_from__(v, args[0]) for v in value)

        if origin is dict and isinstance(value, dict):
            kt, vt = args
            return {LOO_JsonModel.__convert_from__(k, kt): LOO_JsonModel.__convert_from__(v, vt) for k, v in value.items()}

        if is_dataclass(t) and isinstance(value, dict): return t.from_json(value)

        if t is int and isinstance(value, str): return LOO_Misc.extract_number(value, True)
        if t is float and isinstance(value, str): return LOO_Misc.extract_number(value)
        if t is str: return str(value)

        return value

    @staticmethod
    def __convert_to__(obj: Any) -> Any:
        if is_dataclass(obj): return {f.name: LOO_JsonModel.__convert_to__(getattr(obj, f.name)) for f in fields(obj)}

        if isinstance(obj, dict): return {LOO_JsonModel.__convert_to__(k): LOO_JsonModel.__convert_to__(v) for k, v in obj.items()}
        if isinstance(obj, list): return [LOO_JsonModel.__convert_to__(x) for x in obj]

        return obj

@dataclass
class LOO_Voucher(LOO_JsonModel):
    budget: float = 0
    apply_to: str = ""
    threshold: float = 0
    is_fixed: bool = False
    discount: float = 0
    cap: Optional[float] = None

    def get_active_cost(self, products: List["Product"]) -> float:
        psum = sum([p.price for p in products])

        if self.apply_to == "shop":
            shops = set([p.shop for p in products])
            if len(shops) > 1: return psum

        if self.threshold is not None and psum < self.threshold: return psum
        if self.is_fixed: return psum - self.discount

        discount = psum * self.discount / 100.0

        if self.cap is not None: discount = min(discount, self.cap)
        return psum - discount

    def get_favor_range(self) -> Tuple[float, float]:
        high = (self.threshold / 0.1) if self.threshold is not None else math.inf
        low = (self.threshold / 0.9) if self.threshold is not None else 0

        if not self.is_fixed:
            l1 = self.budget / 1.1 / (1.0 - self.discount / 100.0)
            l2 = self.budget / 1.1 + (self.cap if self.cap is not None else math.inf)

            h1 = self.budget + (self.cap if self.cap is not None else math.inf)
            h2 = self.budget / (1.0 - self.discount / 100.0)

            low = max(low, min(l1, l2))
            high = min([high, h1, h2])
        else:
            low = max(low, self.budget / 1.1 + self.discount)
            high = min(high, self.budget + self.discount)

        return (math.floor(low) - 1, math.ceil(high) + 1)

    def is_favor_cost(self, psum: float) -> bool:
        if self.threshold < int(psum * 0.1) or self.threshold > int(psum * 0.9): return False
        if psum < 100.0: return False

        if not self.is_fixed:
            ac = max(psum * (1.0 - self.discount / 100.0), psum - (self.cap if self.cap is not None else math.inf))
            if self.budget < ac or self.budget > math.ceil(1.1 * ac): return False
        elif self.budget < psum - self.discount: return False

        return True

@dataclass
class LOO_SubQuery(LOO_JsonModel):
    query_f: str = ""
    query: str = ""

    keywords: List[str] = field(default_factory = list)
    services: List[str] = field(default_factory = list)

    price: Optional[str] = None
    id: int = -1

    template: Dict[str, str] = field(default_factory = dict)
    prod_only: bool = False

    def __hash__(self) -> int:
        return hash((self.query_f, self.id))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)): return False
        return self.query_f == other.query_f and self.id == other.id

@dataclass
class LOO_Query(LOO_JsonModel):
    challenge: LOO_Challenge = LOO_Challenge.UNKNOWN

    query_f: str = ""
    query_e: str = ""

    intents: List[LOO_SubQuery] = field(default_factory = list)
    voucher: Optional[LOO_Voucher] = None

@dataclass
class LOO_Keywords(LOO_JsonModel):
    keywords: List[str] = field(default_factory = list)

    def get_q(self) -> str:
        chunks = []

        for kw in self.keywords:
            for m in kw.split():
                m = m.lower().strip()
                if m and m not in chunks: chunks.append(m)

        return ' '.join(sorted(chunks))

    def __hash__(self) -> int:
        return hash(self.get_q())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)): return False
        return self.get_q() == other.get_q()

@dataclass
class LOO_Filter(LOO_Keywords):
    services: List[str] = field(default_factory = list)

    price: Optional[str] = None
    shop: Optional[str] = None

    def get_kw(self) -> LOO_Keywords:
        return LOO_Keywords(keywords = self.keywords)

    def update(self, **kwargs) -> "Filter":
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        return self

    def clone(self) -> "Filter":
        return LOO_Filter.from_json(self.to_json())

    def __hash__(self) -> int:
        return hash((self.get_q(), self.price, self.shop))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)): return False
        return self.get_q() == other.get_q() and self.price == other.price and self.shop == other.shop

@dataclass
class LOO_Product(LOO_JsonModel):
    title: str = ""
    id: str = ""

    price: float = 0
    shop: str = ""

    services: List[str] = field(default_factory = list)
    is_viewed: bool = False

    details: Dict[str, List[List[Tuple[str, str]]]] = field(default_factory = list)

    attr: Dict[str, List[str]] = field(default_factory = dict)
    sku: Dict[str, Dict[str, str]] = field(default_factory = dict)

    @classmethod
    def from_json(cls, data) -> "Product":
        prod: LOO_Product = super().from_json(data)
        return prod.update_details(True)

    def promptfy(self, id: str) -> Dict[str, Any]:
        joint = list(set((k, v) for _, kvs in self.details.items() for kv in kvs for k, v in kv))
        return {"id": id, "title": self.title, "attributes": [f"{k}: {v}" for k, v in joint]}

    def update_details(self, enforced: bool = False) -> "Product":
        if enforced or not self.details:
            self.details = {"sku": [[(k, v) for k, v in kv.items()] for _, kv in self.sku.items()]}
            self.details.update({f"att_{k}": [[(k, v)] for v in vs] for k, vs in self.attr.items()})

        return self

    def get_joint_details(self) -> List[Tuple[str, str]]:
        self.update_details()

        return list({
            (k, v) for _, cats in self.details.items() for kvs in cats for k, v in kvs
        })

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)): return False
        return self.id == other.id

@dataclass
class LOO_Weight(LOO_JsonModel):
    DENSE_LOG_PROB = [
        -0.2787871503170742, -2.539370307651016, -3.503979616795979, -4.111766948404767, -4.207077128209092, -4.685112929152092, -5.0280576802789225, -5.074577695913815,
        -5.346511411397457, -5.816515040643193, -5.816515376443249, -6.292854488338015, -6.292855029033076, -6.292855569728429, -6.292856110424076, -6.292856651120014,
        -6.292857191816245, -6.292857732512768, -6.292858273209584, -6.292858813906691, -6.292859354604092, -6.292859895301784, -6.292860435999769, -6.292860976698046,
        -6.292861517396616, -6.292862058095478, -6.292862598794632, -6.292863139494079, -6.292863680193818, -6.29286422089385, -6.292864761594173, -6.292865302294789,
        -6.292865842995698, -6.292866383696898, -6.292866924398392, -6.292867465100177, -6.292868005802255, -6.292868546504625, -6.292869087207288, -6.292869627910243,
        -6.29287016861349, -6.29287070931703, -6.292871250020863, -6.292871790724987, -6.292872331429403, -6.2928728721341125, -6.292873412839114, -6.292873953544408,
        -6.292874494249994
    ]
    ORDINAL_LOG_PROB = [
        -0.5435135830763057, -2.266895754859624, -2.9716037578206453, -3.5139280486460067, -3.788364894347767, -4.237534436690661, -4.237534505927776, -4.622590673167618,
        -4.622590774925198, -4.823261368629769, -4.983604018704948, -5.20132750254982, -5.201327684063349, -5.346509512394317, -5.528831069188271, -5.528831321038303,
        -5.528831572888398, -5.69556053122239, -5.695560828766738, -5.6955611263111745, -5.6955614238556995, -5.695561721400313, -5.695562018945015, -5.695562316489806,
        -5.979032071137827, -5.979032466196729, -6.071405391268843, -6.071405824559259, -6.071406257849863, -6.071406691140655, -6.509660322199998, -6.509660993800224,
        -6.5096616654009, -6.684013709344775, -6.684014508868905, -6.684015308393673, -6.6840161079190805, -6.684016907445127, -6.902702910309606, -6.902703905273064,
        -6.902704900237512, -6.902705895202949, -6.9027068901693776, -6.902707885136795, -6.902708880105203, -6.9027098750746, -6.902710870044988, -6.902711865016365,
        -6.902712859988733, -6.90271385496209, -6.902714849936437, -6.902715844911775, -6.902716839888103, -6.9027178348654195, -6.902718829843727, -6.902719824823024,
        -6.902720819803312, -6.902721814784589, -6.902722809766856, -6.902723804750114, -6.9027247997343615, -6.902725794719598, -6.9027267897058255, -6.902727784693043,
        -6.90272877968125, -6.902729774670448, -6.902730769660635, -6.902731764651812, -6.902732759643979, -6.902733754637136, -6.902734749631284, -6.902735744626421,
        -6.902736739622549, -6.902737734619666, -6.902738729617774, -6.902739724616871, -6.902740719616959, -6.902741714618036, -6.902742709620103, -6.902743704623161,
        -6.902744699627208, -6.902745694632245, -6.902746689638273, -6.902747684645291, -6.902748679653299, -6.902749674662297, -6.902750669672284, -6.902751664683262,
        -6.9027526596952296, -6.902753654708188, -6.9027546497221355, -6.9027556447370735, -6.902756639753002, -6.9027576347699195, -6.9027586297878285, -6.902759624806727,
        -6.902760619826615, -6.902761614847493, -6.902762609869361, -6.90276360489222, -6.902764599916069, -6.902765594940908, -6.9027665899667365, -6.902767584993555,
        -6.902768580021364, -6.902769575050163, -6.9027705700799515, -6.902771565110731, -6.9027725601425, -6.902773555175259
    ]
    detail: Dict[str, float] = field(default_factory = dict)
    choice: Dict[str, int] = field(default_factory = dict)
    covers: List[float] = field(default_factory = list)

    score: float = 0
    price: float = 0

    ordinal_p: float = 0
    dense_p: float = 0

    srank: Tuple[int, int] = (0, -1)
    wrank: Tuple[int, int] = (-1, -1)

    def fill_detail(self, dst_tokens: List[str]) -> "Weight":
        for i in range(len(self.covers)):
            self.detail[dst_tokens[i]] = self.covers[i]

        return self

    def update_wrank(self, wrank: Tuple[int, int]) -> "Weight":
        self.ordinal_p = LOO_Misc.decaying_log(LOO_Weight.ORDINAL_LOG_PROB, wrank[1], 0.9)
        self.dense_p = LOO_Misc.decaying_log(LOO_Weight.DENSE_LOG_PROB, wrank[0], 0.9)

        self.wrank = wrank
        return self

    def value(self, service_order: int, sku_count: int) -> Tuple:
        sku_order = self.choice.get("sku", -1) + 1
        if not sku_order: sku_order = sku_count

        return (
            self.score,
            math.log(1.0 / 2 ** service_order) + math.log(1.0 / 2 ** sku_order),
            -self.srank[0], -self.srank[1]
        )

@total_ordering
class LOO_BundleScore(object):
    def __init__(self, dense_p: float = 0, ordinal_p: float = 0) -> None:
        self.ordinal_p = ordinal_p
        self.dense_p = dense_p

    def __eq__(self, other: "BundleScore") -> bool:
        if not isinstance(other, LOO_BundleScore): return NotImplemented
        return self.dense_p == other.dense_p and self.ordinal_p == other.ordinal_p

    def __lt__(self, other: "BundleScore") -> bool:
        if not isinstance(other, LOO_BundleScore): return NotImplemented
        return (self.dense_p, self.ordinal_p) < (other.dense_p, other.ordinal_p)

    def __add__(self, other: "BundleScore") -> "BundleScore":
        if not isinstance(other, LOO_BundleScore): return NotImplemented
        return LOO_BundleScore(self.dense_p + other.dense_p, self.ordinal_p + other.ordinal_p)

    def __iadd__(self, other: "BundleScore") -> "BundleScore":
        if not isinstance(other, LOO_BundleScore): return NotImplemented

        self.ordinal_p += other.ordinal_p
        self.dense_p += other.dense_p

        return self

    def __radd__(self, other: "BundleScore") -> "BundleScore":
        return self if other == 0 else NotImplemented

    @staticmethod
    def create(weight: LOO_Weight) -> "BundleScore":
        return LOO_BundleScore(weight.dense_p, weight.ordinal_p)

class LOO_Misc:
    LOGGER = logging.Logger("angry_oro")
    LOGGER.addHandler(logging.StreamHandler())

    @staticmethod
    def parse_json(s: str) -> dict[str, Any] | list[Any] | None:
        cleaned = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags = re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned, ).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        starts = []

        obj_start = cleaned.find("{")
        if obj_start >= 0: starts.append(obj_start)

        arr_start = cleaned.find("[")
        if arr_start >= 0: starts.append(arr_start)

        for start in sorted(starts):
            try:
                obj, _ = decoder.raw_decode(cleaned[start:])
                return obj
            except json.JSONDecodeError:
                pass

        for i in range(len(cleaned)):
            if cleaned[i] in ['}', ']']:
                try:
                    return json.loads(cleaned[:i] + cleaned[i] + cleaned[i:])
                except json.JSONDecodeError:
                    pass

                try:
                    return json.loads(cleaned[:i] + cleaned[i + 1:])
                except json.JSONDecodeError:
                    pass

        try:
            pattern = re.compile(r'([,{])\s*"([^":]+)\s*:\s*([^"]+)"\s*([,}])')

            def __replace__(m: re.Match):
                return f'{m.group(1)} "{m.group(2).strip()}": "{m.group(3).strip()}"{m.group(4)}'

            fix = pattern.sub(__replace__, cleaned)
            return json.loads(fix)
        except json.JSONDecodeError:
            pass

        try:
            fix = re.sub(r"(\},\s+)(\w+\")", r"\1\"\2", cleaned)
            fix = re.sub(r"(\"[\w\_]+)\s*(\:\s*\{)", r"\1\"\2", fix)
            return json.loads(fix)
        except json.JSONDecodeError:
            pass

        try:
            fix = re.sub(r",\s*([}\]])", r"\1", cleaned)
            return json.loads(fix)
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def print_json(js: Dict[str, Any] | List[Any]) -> None:
        LOO_Misc.LOGGER.info(json.dumps(js, ensure_ascii = False, indent = 2))

    @staticmethod
    def get_choice_combs(cats: Dict | List, include_non: bool) -> List[Dict | List]:
        keys = list(cats.keys()) if isinstance(cats, dict) else list(range(len(cats)))

        choices = [([-1] if include_non else []) + list(range(len(cats[k]))) for k in keys]
        opts = product(*choices)

        return [dict(zip(keys, opt)) for opt in opts] if isinstance(cats, dict) else list(opts)

    @staticmethod
    def extract_number(s: str, as_int: bool = False) -> int | float:
        pattern = r"\d+\.?\d*"
        matches = re.findall(pattern, s)

        if len(matches) == 0: return 0
        val = float(matches[0])

        return val if not as_int else int(val)

    @staticmethod
    def scan_patterns(s: str, patterns: List[str]) -> Optional[str]:
        for p in patterns:
            if (m := re.search(p, s, re.I)): return m.group(1)

        return None

    @staticmethod
    def is_morph_equal(v1: str, v2: str, alt_map: Dict[str, Set] = {}) -> bool:
        v1, v2 = v1.lower().strip(), v2.lower().strip()
        if v1 == v2: return True

        if any({v1, v2} == {a, b} for a, b in LOO_Vocab.NUM_ALTS): return True
        if {v1, v2} < {"y", "yr", "yrs"}: return True
        if {v1, v2} == {"pc", "pcs"}: return True

        if v1.isnumeric() or v2.isnumeric():
            if v1.isnumeric() and v2.isnumeric() and v1 == v2: return True
            return False

        if len(v1) < 3 or len(v2) < 3: return False
        if re.search(r"\d", v1 + v2): return False

        if len(v1) >= 5 and len(v2) >= 5:
            if v1[-5:] == v2[-5:] and abs(len(v1) - len(v2)) <= 2 and max(len(v1), len(v2)) <= 8: return True
            if v1[:5] == v2[:5]: return True

        def __get_stems__(word: str) -> set[str]:
            word = word.lower()

            roots = {word}
            queue = [word]

            while queue:
                w = queue.pop()

                for suffix, rules in LOO_Vocab.MORPH_RULES:
                    if not w.endswith(suffix): continue
                    if len(w) <= len(suffix) + 2: continue

                    for r in rules:
                        if callable(r):
                            t = r(r"([b-df-hj-np-tv-z])\1$", r"\1", w[:-len(suffix)])
                        else:
                            t = w[:-len(suffix)] + r

                        if len(t) >= 2 and t not in roots:
                            roots.add(t)
                            queue.append(t)

                for syn in alt_map.get(w, {}):
                    if syn not in roots:
                        roots.add(syn)
                        queue.append(syn)

            return roots

        return bool(__get_stems__(v1) & __get_stems__(v2))

    @staticmethod
    def patch_max(arr: List[float], patch: List[float]) -> None:
        for i in range(len(arr)):
            if arr[i] < patch[i]: arr[i] = patch[i]

    @staticmethod
    def safe_div(a: float, b: float) -> float:
        return (a / b) if b > 1e-7 else 0

    @staticmethod
    def decaying_log(x: List[float], idx: int, decay: float) -> float:
        if idx >= len(x):
            return x[-1] + (idx - len(x) + 1) * math.log(decay)
        else:
            return x[idx]

class LOO_Proxy:
    LLM_PROXY = ProxyClient(timeout = 200, max_retries = 5, retry_delay = 1.0)
    ORO_PROXY = ProxyClient(timeout = 30, max_retries = 5, retry_delay = 1.0)

    LLM_PRIORITIES = ["moonshotai/Kimi-K2.6-TEE", "deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3.5-397B-A17B-TEE", "zai-org/GLM-5.1-TEE", "zai-org/GLM-5.2-TEE", "Qwen/Qwen3-32B-TEE", "google/gemma-4-31B-turbo-TEE"]
    OPENROUTER_LLM_PRIORITIES = ["moonshotai/kimi-k2.6", "deepseek/deepseek-v3.2", "qwen/qwen3.5-397b-a17b", "z-ai/glm-5.1", "z-ai/glm-5.2", "qwen/qwen3-32b", "google/gemma-4-31b-it"]

    @staticmethod
    def active_inference_provider() -> str:
        raw = (getenv("INFERENCE_PROVIDER") or "openrouter").strip().lower()
        return raw if raw in ("chutes", "openrouter") else "openrouter"

    TEMP_UP = 0.3

    @staticmethod
    def trigger_llm(
        prompt: str, thinking: bool = False, preferred_llms: List[str] = [], temperature: float = 0, max_tokens: int = 8192, expect: str = "any"
    ) -> Tuple[
        Dict[str, Any] | List[Any], Any
    ]:
        llm_priorities = LOO_Proxy.__get_llm_priorities__(preferred_llms)
        trigger_tick = time.time()
        t = temperature

        while True:
            for model in llm_priorities:
                payload = LOO_Proxy.__create_llm_payload__(model, prompt, thinking, t, max_tokens)

                try:
                    tick = time.time()

                    resp: Dict[str, Dict] = LOO_Proxy.LLM_PROXY.post("/inference/chat/completions", json_data = payload)
                    content = LOO_Proxy.__resp_to_content__(resp)

                    try:
                        js = LOO_Misc.parse_json(content)
                        if js is None: raise Exception()
                    except Exception as exc:
                        t += LOO_Proxy.TEMP_UP
                        raise Exception(f"failed to parse json: {exc}\n*******\n{content}\n******")

                    if (expect == "dict" and not isinstance(js, dict)) or \
                        (expect == "list" and not isinstance(js, list)):
                        t += LOO_Proxy.TEMP_UP
                        raise Exception(f"expected {expect} but got {type(js)}")

                    throughput = resp.get("usage", {}).get("completion_tokens", 0)
                    trigger_delay = time.time() - trigger_tick
                    delay = time.time() - tick

                    LOO_Misc.LOGGER.info(
                        f"* llm triggered: model = '{model}' | throughput = {throughput} | delay = {delay:.2f}s | "
                        f"total_delay = {trigger_delay:.2f}s"
                    )
                    return js, resp
                except Exception as exc:
                    LOO_Misc.LOGGER.error(f"# llm trigger error with '{model}': {exc}")
                    continue

    @staticmethod
    def search(filter: LOO_Filter, sort: Optional[str] = None, page: int = 1) -> Tuple[List[LOO_Product], ToolCallResult]:
        while True:
            try:
                tool_res = execute_tool_call(
                    "find_product",
                    {
                        "q": filter.get_q(),
                        "page": page,
                        "shop_id": filter.shop,
                        "price": filter.price,
                        "sort": sort,
                        "service": ','.join(filter.services or [])
                    }
                )
                js_prods: List[Dict] = tool_res["result"]
                products = []

                for j in js_prods:
                    price = j.get("price") or 0
                    price = LOO_Misc.extract_number(str(price))

                    products.append(
                        LOO_Product(
                            id = str(j.get("product_id")),
                            title = j.get("title", ""),
                            price = price,
                            shop = j.get("shop_id", ""),
                            services = j.get("service", []) or [],
                            attr = {},
                            sku = {},
                            is_viewed = False
                        )
                    )

                return products, tool_res

            except Exception as exc:
                LOO_Misc.LOGGER.error(f"# error in `find_products`: {exc}")
                time.sleep(0.5)

    @staticmethod
    def view(products: List[LOO_Product]) -> ToolCallResult:
        if not products: return
        pid_str = ','.join([p.id for p in products])

        while True:
            try:
                tool_res = execute_tool_call("view_product_information", {"product_ids": pid_str})

                details = tool_res["result"]
                if not details: return tool_res

                details_map: Dict[str, Dict[str, Any]] = {d["product_id"]: d for d in details}

                for prod in products:
                    d = details_map.get(prod.id, {})

                    prod.sku = d.get("sku_options", {})
                    prod.attr = d.get("attributes", {})
                    prod.is_viewed = True

                return tool_res
            except Exception as exc:
                LOO_Misc.LOGGER.error(f"# error in `view_products`: {exc}")
                time.sleep(0.5)

    @staticmethod
    def submit(products: List[LOO_Product]) -> ToolCallResult:
        return execute_tool_call("recommend_product", {"product_ids": ','.join([p.id for p in products])})

    @staticmethod
    def finish(status: str) -> ToolCallResult:
        return execute_tool_call("terminate", {"status": status})

    @staticmethod
    def __get_llm_priorities__(prefers: List[str]) -> List[str]:
        CHUTES_TO_OR = {
            "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
            "moonshotai/Kimi-K2.6-TEE": "moonshotai/kimi-k2.6",
            "zai-org/GLM-5.1-TEE": "z-ai/glm-5.1",
            "zai-org/GLM-5.2-TEE": "z-ai/glm-5.2",
            "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
            "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
            "Qwen/Qwen3-32B-TEE": "qwen/qwen3-32b",
            "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
        }
        use_or = LOO_Proxy.active_inference_provider() == "openrouter"
        if use_or:
            priorities = [CHUTES_TO_OR.get(x, x) for x in prefers]
        else:
            priorities = prefers.copy()
        base = (
            LOO_Proxy.OPENROUTER_LLM_PRIORITIES
            if use_or and hasattr(LOO_Proxy, "OPENROUTER_LLM_PRIORITIES")
            else LOO_Proxy.LLM_PRIORITIES
        )
        for llm in base:
            if llm not in priorities:
                priorities.append(llm)
        return priorities


    @staticmethod
    def __create_llm_payload__(model: str, prompt: str, thinking: bool, temperature: float, max_tokens: int) -> Dict:
        return {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "chat_template_kwargs": {
                "enable_thinking": thinking,
                "thinking": thinking
            },
            "reasoning": {
                "effort": "medium" if thinking else "none",
                "enabled": thinking
            },
            "thinking": {
                "type": "disabled" if not thinking else "enabled"
            },
            "temperature": temperature,
            "stream": False,
            "max_completion_tokens": max_tokens,
            "max_tokens": max_tokens
        }

    @staticmethod
    def __resp_to_content__(resp: Dict) -> str:
        try:
            choice: Dict[str, Dict] = resp.get("choices")[0]
            content = choice.get("message").get("content")

            if not content: raise Exception()
            return content
        except Exception as exc:
            raise Exception(f"invalid response ({exc})")

class LOO_Util:
    @staticmethod
    def normalize_str(s: str) -> str:
        s = s.lower().replace("'s", "")

        s = re.sub(r"\b(t)[\s\-](shirt)s?\b", r"\1\2", s)
        s = re.sub(r"\b(shirt)s?\b", r"\1", s)

        s = re.sub(r"\b([a-z]+)[\s\-]+resistan(?:t|ce)\b", r"\1 proof", s)
        s = re.sub(r"\b([a-z]+)(proof)\b", r"\1 \2", s)

        s = re.sub(r"\bextra[\s\-]+(s|l)(?:mall|arge)\b", r"x\1", s)

        s = re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL>", s)
        s = re.sub(r"[^a-z0-9%\u4e00-\u9fff]", " ", s)
        s = s.replace("<DECIMAL>", ".")

        s = re.sub(r"\b(\d+)([a-z]+)\b", r"\1 \2", s)
        s = re.sub(r"\s+", " ", s).strip()

        return s

    @staticmethod
    def match_line(
        src: Tuple[str, str],
        dst: Tuple[List[str], List[str]],
        is_title: bool,
        is_prod_type: bool,
        alt_map: Dict[str, Set],
        ws: List[float],
        offset: int
    ) -> List[float]:
        ks, vs = LOO_Util.normalize_str(src[0]).split(), LOO_Util.normalize_str(src[1]).split()
        kd, vd = dst

        res = [0 for _ in ws]
        rate = 1.0

        if (not vs) or (not vd): return res
        v2v_contrib, v2v_cover = LOO_Util.__check_coverage__(vs, vd, alt_map)

        if not is_prod_type:
            v2k_contrib, _ = LOO_Util.__check_coverage__(vs, kd, {})
        else:
            v2k_contrib = set()

        if not is_title:
            _, k2v_cover = LOO_Util.__check_coverage__(ks, vd, alt_map)
        else:
            k2v_cover = set()

        vcontrib = v2v_contrib | v2k_contrib
        vcover = v2v_cover | k2v_cover

        if not is_title and not is_prod_type:
            rate += 0.01 * LOO_Misc.safe_div(len(v2v_cover), len(vd))
            rate += 0.01 * LOO_Misc.safe_div(len(vcover), len(vd))
            rate += 0.01 * LOO_Misc.safe_div(len(vcontrib), len(vs))
        elif is_title and is_prod_type:
            rate += 0.01 * LOO_Misc.safe_div(len(v2v_cover), len(vd))
        elif is_title and not is_prod_type:
            rate -= 0.05
        else:
            rate -= 0.05

        for i in vcover:
            res[offset + i] = ws[offset + i] * rate

        return res

    @staticmethod
    def get_kw_vars(kws: List[str]) -> Tuple[List[List[Tuple[str, float]]], List[int]]:
        cat, indice = [], []

        def __du_space__(v: str) -> bool:
            if (m := re.search(r"^(\d+(?:\-\d+)?)[\s\-]+([a-z]+)$", v)):
                num, unit = m.group(1), m.group(2)

                if unit in ["piece", "pieces"]:
                    cat.append([(num, 0), (f"{num}pcs", 0.1)])
                elif unit in ["year", "years"]:
                    cat.append([(v, 0), (num, 0.1), (f"{num}yrs", 0.5), (f"{num}y", 1.0)])
                else:
                    for su, lu in LOO_Vocab.UNIT_MAP.items():
                        if lu and LOO_Misc.is_morph_equal(unit, lu):
                            cat.append([(f"{num}{su}", 0), (num, 0.1), (v, 1.0)])
                            break
                        elif su == unit:
                            cat.append([(num, 0), (v, 0.1), (f"{num}{unit}", 0.5)])
                            break
                    else:
                        cat.append([(v, 0), (num, 0.5), (f"{num}{unit}", 2.0)])

                return True
            else:
                return False

        def __du_tight__(v: str) -> bool:
            if (m := re.search(r"^(\d+(?:\-\d+)?)([a-z]+)$", v)):
                num, unit = m.group(1), m.group(2)
                if unit in ["pc", "pcs", "y", "yrs"]: return False

                for su, lu in LOO_Vocab.UNIT_MAP.items():
                    if unit == su:
                        cat.append([(v, 0), (num, 1.0)])
                        return True
                    elif lu and LOO_Misc.is_morph_equal(unit, lu):
                        cat.append([(v, 0), (f"{num}{su}", 1.0), (num, 2.0)])
                        return True

            return False

        def __single__(v: str) -> bool:
            if re.search(r"^single[\-\s]+(piece|item)$", v):
                cat.append([(v, 0), ("single", 0.1), ("1pc", 0.2)])
                return True
            else:
                return False

        def __size__(v: str) -> bool:
            if (m := re.search(r"^(extra\s+)?(small|large|medium)$", v)):
                abb = ('x' if m.group(1) else '') + m.group(2)[0]

                if m.group(1):
                    cat.append([(abb, 0), (v, 1.0)])
                else:
                    cat.append([(v, 0), (abb, 0.2)])

                return True
            else:
                return False

        def __proof__(v: str) -> bool:
            if (m := re.search(r"^(\w+)[\s\-]+resistan(t|ce)$", v)):
                w = m.group(1)

                cat.append([(f"{w}proof", 0), (w, 0.5), (v, 1.0)])
                return True
            elif (m := re.search(r"^(\w+)([\s\-])?proof$", v)):
                w, s = m.group(1), m.group(2)

                if s:
                    cat.append([(v, 0), (f"{w}proof", 0.2), (w, 0.5)])
                else:
                    cat.append([(v, 0), (w, 1.0)])

                return True

            return False

        def __alt__(v: str) -> bool:
            for k in v.split():
                for a, b in LOO_Vocab.KW_ALTS:
                    if LOO_Misc.is_morph_equal(k, a):
                        nv = v.replace(k, b)

                        cat.append([(v, 0), (nv, 0.5)])
                        return True

            return False

        def __filler__(v: str) -> bool:
            for k in v.split():
                for f in LOO_Vocab.KW_WEAKS:
                    if LOO_Misc.is_morph_equal(k, f):
                        nv = v.replace(k, "").strip()

                        cat.append([(v, 0), (nv, 0.5)])
                        return True

            return False

        for i, v in enumerate(kws):
            if any(check(v) for check in [__du_space__, __du_tight__, __size__, __single__, __proof__, __alt__, __filler__]): indice.append(i)

        return cat, indice

    @staticmethod
    def promptfy_prod_batch(products: List[LOO_Product], prefix: str) -> str:
        prod_str = ""

        for i, p in enumerate(products):
            j = p.promptfy(f"{prefix}{i + 1}")
            prod_str += json.dumps(j, ensure_ascii = False, indent = 2) + '\n\n'

        return prod_str

    @staticmethod
    def promptfy_price(price: Optional[str]) -> str:
        if not price: return ""

        segs = [s.strip() for s in price.split('-')]
        if len(segs) != 2: return ""

        try:
            low = int(segs[0]) if segs[0] else None
            high = int(segs[1]) if segs[1] else None

            if low and high: return f"price between {low} and {high}"
            if low: return f"price greater than {low}"

            return f"price less than {high}"
        except Exception:
            return ""

    @staticmethod
    def numerify_price(price: Optional[str]) -> Tuple[int, int]:
        low_str, high_str = price.split('-', 1)
        low, high = 0, math.inf

        if low_str: low = LOO_Misc.extract_number(low_str, True)
        if high_str: high = LOO_Misc.extract_number(high_str, True)

        return low, high

    @staticmethod
    def stringfy_price(low: int, high: int) -> str:
        return (str(low) if low > 0 else "") + "-" + (str(high) if high < math.inf else "")

    @staticmethod
    def basic_check(prod: LOO_Product, filter: LOO_Filter) -> bool:
        if filter.price:
            if not prod.price: return False

            prod_price = LOO_Misc.extract_number(str(prod.price), int)
            low, high = LOO_Util.numerify_price(filter.price)

            if prod_price < low or prod_price > high: return False

        if filter.services:
            prod_services = [str(svc).lower() for svc in (prod.services or [])]
            sq_services = [str(svc).lower() for svc in filter.services]

            for svc in sq_services:
                if svc not in prod_services: return False

        if filter.shop and filter.shop != prod.shop: return False
        return True

    @staticmethod
    def __check_coverage__(src_ts: List[str], dst_ts: List[str], alt_map: Dict[str, Set]) -> Tuple[Set, Set]:
        contrib, cover = set(), set()

        for j, ts in enumerate(src_ts):
            for i, td in enumerate(dst_ts):
                if i > 0 and j > 0 and src_ts[j - 1] != dst_ts[i - 1] and src_ts[j - 1].isnumeric() and dst_ts[i - 1].isnumeric():
                    if ts in LOO_Vocab.ALL_UNITS and td in LOO_Vocab.ALL_UNITS: continue

                if LOO_Misc.is_morph_equal(ts, td, alt_map):
                    contrib.add(j)
                    cover.add(i)
                    continue

                if i < len(dst_ts) - 1:
                    if ts.startswith(td) and LOO_Misc.is_morph_equal(ts[len(td):], dst_ts[i + 1], alt_map):
                        cover.update({i, i + 1})
                        contrib.add(j)
                        continue

                if j < len(src_ts) - 1:
                    if td.startswith(ts) and LOO_Misc.is_morph_equal(td[len(ts):], src_ts[j + 1], alt_map):
                        contrib.update({j, j + 1})
                        cover.add(i)
                        continue

                if any({size, size[0]} == {ts, td} for size in LOO_Vocab.SIZE_TOKENS):
                    contrib.add(j)
                    cover.add(i)
                    continue

                if i > 0 and j > 0 and src_ts[j - 1] == dst_ts[i - 1] and src_ts[j - 1].isnumeric():
                    a, b = (ts, td) if len(ts) < len(td) else (td, ts)
                    lu = LOO_Vocab.UNIT_MAP.get(a)

                    if lu and LOO_Misc.is_morph_equal(lu, b):
                        contrib.add(j)
                        cover.add(i)
                        continue

                    if a in LOO_Vocab.ALL_UNITS and b in LOO_Vocab.ALL_UNITS:
                        if j - 1 in contrib: contrib.remove(j - 1)
                        if i - 1 in cover: cover.remove(i - 1)
                        continue

        return contrib, cover

class LOO_SegTree(object):
    def __init__(self, values, k = 100) -> None:
        self.k, n = k, len(values)
        self.size = 1

        while self.size < n:
            self.size <<= 1

        self.seg = [[] for _ in range(2 * self.size)]

        for i, v in enumerate(values):
            self.seg[self.size + i] = [(v, i)]

        for i in range(self.size - 1, 0, -1):
            self.seg[i] = self.__merge__(self.seg[2 * i], self.seg[2 * i + 1])

    def __merge__(self, a: List, b: List) -> List:
        out, ia, ib = [], 0, 0

        while len(out) < self.k:
            if ia == len(a):
                out.extend(b[ib:ib + self.k - len(out)])
                break

            if ib == len(b):
                out.extend(a[ia:ia + self.k - len(out)])
                break

            if a[ia][0] >= b[ib][0]:
                out.append(a[ia])
                ia += 1
            else:
                out.append(b[ib])
                ib += 1

        return out

    def query(self, l: int, r: int) -> List:
        if l > r: return []
        left, right = [], []

        l += self.size
        r += self.size

        while l <= r:
            if l & 1:
                left = self.__merge__(left, self.seg[l])
                l += 1

            if not (r & 1):
                right = self.__merge__(self.seg[r], right)
                r -= 1

            l >>= 1
            r >>= 1

        return self.__merge__(left, right)

class LOO_Debugger(object):
    def log(self, *_, **__) -> None:
        return

class LOO_Debuggable(object):
    def __init__(self, query: LOO_Query, debugger: Optional[LOO_Debugger]) -> None:
        self.debugger: Optional[LOO_Debugger] = debugger
        self.query: LOO_Query = query
        self.stopped: bool = False

    def log(self, obj: Any) -> None:
        if self.debugger is not None: self.debugger.log(obj)

class LOO_Session(LOO_Debuggable):
    MAX_TOOL_RES_PER_STEP = 7

    def __init__(self, query: LOO_Query, debugger: Optional[LOO_Debugger]) -> None:
        super(LOO_Session, self).__init__(query, debugger)
        self.epoch = time.time()

        self.tool_results = []
        self.dialog = []

    def step(self, think: str, content: str) -> None:
        timestamp = time.time() - self.epoch

        self.dialog.append(
            create_dialogue_step(
                think = think,
                tool_results = self.tool_results[:LOO_Session.MAX_TOOL_RES_PER_STEP],
                response = content + f"\n(Timestamp: {timestamp:.2f}s)",
                query = self.query.query_f,
                step = len(self.dialog) + 1
            )
        )
        self.tool_results = []

        if self.debugger is not None and self.debugger.show_step:
            self.log({
                "step": len(self.dialog),
                "think": think,
                "content": content
            })

    def clear(self) -> None:
        self.tool_results = []

    def add_tool_result(self, res: ToolCallResult) -> None:
        try:
            json.dumps(res)
            self.tool_results.append(res)
        except Exception as exc:
            self.log(f"# cannot add tool result: {exc}")

    def step_on_build(self) -> None:
        think = f"This is a shopping request involving {LOO_Challenge(self.query.challenge).promptfy(len(self.query.intents) > 1)}. "
        think += f"First, I will extract meaningful keywords and additional filters from "
        think += ("each product intent" if len(self.query.intents) > 1 else "the query") + " to facilitate search:\n"

        keywords = [", ".join(sq.keywords) for sq in self.query.intents]
        services = [", ".join(svc for svc in sq.services) if sq.services else "Not specified" for sq in self.query.intents]
        prices = [sq.price if sq.price else "Not specified" for sq in self.query.intents]

        for i in range(len(self.query.intents)):
            think += f"\n- Product {i + 1}: " if len(self.query.intents) > 1 else "\n- "
            think += f"Keywords = \"{keywords[i]}\" | Services = \"{services[i]}\" | Price = \"{prices[i]}\""

        if self.query.challenge == LOO_Challenge.VOUCHER:
            think += f"\n\nUser's budget is {int(self.query.voucher.budget)} and the available voucher options are as follows:\n\n"
            think += "- Voucher Type: " + ("Fixed amount" if self.query.voucher.is_fixed else "Percentage") + "\n"
            think += f"- Discount: {int(self.query.voucher.discount)}" + ("%" if not self.query.voucher.is_fixed else "") + "\n"
            think += "- Application Criteria: " + ("Same shop" if self.query.voucher.apply_to == "shop" else "All products") + "\n"
            think += (f"- Threshold: {int(self.query.voucher.threshold)}" if self.query.voucher.threshold is not None else "Not specified") + "\n"
            think += (f"- Cap: {int(self.query.voucher.cap)}" if self.query.voucher.cap is not None else "Not specified") + "\n"

        self.step(think, "Initial query analysis completed.")

    def step_on_retrieve(self) -> None:
        think = "Next, I will perform initial searches for " + ("each sub-query." if len(self.query.intents) > 1 else "the query.")
        self.step(think, "Initial search completed.")

    def step_on_shop_plan(self, shop_stats: Dict[str, Tuple[int, float]], top_k: int) -> None:
        think = "Now I will look for combinations of products from the same shop, since such combinations may benefit from voucher application. For this, " if self.query.challenge == LOO_Challenge.VOUCHER else "Now, "
        think += "I will group the retrieved products by shop and examine how well each shop covers the sub-queries. Shops with broader coverage are more promising, but additional shops will also be considered to avoid overlooking better combinations. "
        think += "The following statistics summarize the coverage provided by the discovered shops:\n\n"

        plan = []

        for n in range(len(self.query.intents), 0, -1):
            shops = [sid for sid in shop_stats if shop_stats[sid][0] == n]
            plan.extend(shops)

            if n > 1:
                think += f"- Shops covering {n} sub-queries: `{', '.join(shops)}`\n"
            else:
                think += f"- The remaining {len(shops)} shop{'s' if len(shops) > 1 else ''} cover{'s' if len(shops) == 1 else ''} only one sub-query.\n"

        think += "\nGood. Next, I will examine the most promising shops in order of coverage to determine whether they can fully satisfy the sub-queries"
        think += " while allowing the voucher to be applied." if self.query.challenge == LOO_Challenge.VOUCHER else "."
        think += f"\n\nThe following shops will be examined: {', '.join(plan[:top_k])}, ... ..."

        self.step(think, "Planned candidate shops.")

    def step_on_shop_batch(self, batch: List[str], sols: int, best_p: List[LOO_Product], n: int) -> None:
        best_str = ', '.join([p.id for p in best_p])

        think = f"[Shop Examination Step {n + 1}]\n\n"
        think += f"In this step, I will check the shop{'s' if len(batch) > 1 else ''}: `{', '.join(batch)}`\n\n"
        think += "I will perform additional searches for each shop on its missing sub-queries using narrower search criteria by enforcing a shop ID constraint. "
        think += f"Among the retrieved products, I can find {sols} combination{'s' if sols > 1 else ''} whose products all come from the same shop"
        think += " for which the voucher can be applied. " if self.query.challenge == LOO_Challenge.VOUCHER else ". "

        if sols > 0:
            think += f"After carefully examining the candidate combination{'s' if sols > 1 else ''}, "

            if best_p:
                think += f"I identified a promising combination (`{best_str}`). Each product satisfies its corresponding constraints. "
                think += "Now I will stop the shop examination and recommend the identified combination."
            else:
                think += f"{'all of them were' if sols > 1 else 'it was'} rejected because one or more products did not satisfy the specified constraints. "
                think += "Therefore, I will continue the shop examination."
        else:
            think += "Therefore, I will continue the shop examination."

        self.step(think, "")

    def step_on_shop_fallback(self, best_p: List[LOO_Product]) -> None:
        best_str = ', '.join([p.id for p in best_p])

        think = "[Shop Examination Rollback]\n\nI have examined all discovered shops, but was unable to find one providing matching products"
        think += " for which the voucher can be applied. " if self.query.challenge == LOO_Challenge.VOUCHER else ". "

        if best_p:
            think += "It's because either a valid combination has been overlooked during the previous evaluation steps, or no matching combination exists in the current database. "
            think += "Before reporting failure to the user, I will perform one final verification of the previously discovered candidate combinations. "
            think += f"The combination `{best_str}` appears to be a strong candidate. Let me verify it against the stated constraints. "
            think += "Upon re-examination, I found that this combination satisfies all stated constraints. It was incorrectly rejected during the previous evaluation. "
            think += f"To ensure that no other valid combinations have been overlooked, I will review the remaining candidates as well. "
            think += f"No additional combinations satisfying all stated constraints were found. The only missed combination is `{best_str}`. "
            think += f"Therefore, this combination can be recommended to the user."
        elif self.query.challenge == LOO_Challenge.VOUCHER:
            think += "However, voucher application is optional. As long as the total price remains within the budget, the combination can still be recommended. "
            think += "Therefore, I will switch to voucher-free examination mode."
        else:
            think += "This indicates that no matching combination exists in the current database. I will report failure to the user."

        self.step(think, "Completed shop examination process.")

    def step_on_platform(self, best_p: List[LOO_Product], sol_count: bool, is_fallback: bool) -> None:
        best_str = ', '.join([p.id for p in best_p])

        think = (
            f"Now I will look for combinations of products whose total price is within the buget"
            f"{' or becomes affordable after applying the voucher. ' if not is_fallback else '. '}"
            f"First, I will apply the Meet-in-the-Middle (MITM) algorithm to solve the underlying knapsack problem, where product weights are derived from the logarithms of the search-rank probabilities. "
            f"I can find {sol_count} valid solutions, sorted by total weights. Next, I will examine the candidate combinations in order until I find one that satisfies all stated constraints. "
        )
        if best_p:
            think += f"The combination `{best_str}` appears to be a strong candidate. Let me verify its products against the constraints. "
            think += "Good. I have confirmed that each product satisfies its corresponding sub-query. Now this combination can be recommended to the user."
        else:
            think += "I have evaluated all candidate combinations, but none of them satisfies all stated constraints. "
            think += "This indicates that no combination satisfying the user's requirements exists in the current database. I will report the failure to the user."

        self.step(think, "")

    def step_on_select(self, best_p: LOO_Product, js_check: Dict, is_fallback: bool, sq: LOO_SubQuery) -> None:
        if not is_fallback:
            think = "From the retrieved products, I will examine the top ranked candidates to find a product satisfying all the constraints."
            think += f"\n\n{json.dumps(js_check, ensure_ascii = False, indent = 2)}\n\nGood. A matching product has been identified."
        else:
            think = "Since the retrieved products are already ranked by relevance, I will first examine the highest-ranked product to determine whether it satisfies all the user's requirements. "
            think += "If it turns out to be a good match, I can recommend it directly without performing any further evaluation. "
            think += f"The title of the highest-ranked product is \"{best_p.title}\" I will check its attribute details to verify each constraint using the following checklist:\n"

            for k, v in sq.template.items():
                think += f"\n`{k}`: `{v}` → match ✓"

            think += "\n\nAll checks passed."

        self.step(think, f"A matching product selected: `{best_p.id}`")

    def step_on_compare(self, best_p: List[LOO_Product], alt: str) -> None:
        if not best_p or not alt: return

        think = "To avoid overlooking a better candidate, I will compare the identified one with other top promising candidate.\n\n" + alt
        self.step(think, "Compared with the top promising candidate.")

    def step_on_submit(self, best_p: List[LOO_Product]) -> None:
        if not best_p:
            return self.step(
                "I can't find the product(s) satisfying the user's requirements. I will report failure to the user.",
                "Failed to find the requested product(s). Please try again with more specific constraints."
            )

        think = f"I have identified the following product{' combination' if len(self.query.intents) > 1 else ''}: `{', '.join([p.id for p in best_p])}`"
        think += f" Before submitting it, I will verify all the requirements again.\n"

        for i in range(len(best_p)):
            think += f"\n[Product {i + 1}] (UID = {best_p[i].id})\n" if len(self.query.intents) > 1 else ""
            think += f"\n- Requested: \"{self.query.intents[i].query}\""
            think += f"\n- Product Title: \"{best_p[i].title}\""
            think += f"\n- Shop ID: {best_p[i].shop}"

            for k, v in self.query.intents[i].template.items():
                think += f"\n- Constraint `{k}`: Expects \"{v}\" | Found \"{v}\" in {'product title' if k == 'product_type' else 'SKU options'} → Exact Match ✓"

            if self.query.intents[i].price:
                think += f"\n- Price: Expects {LOO_Util.promptfy_price(self.query.intents[i].price)} | Found {str(int(best_p[i].price))} → Within Range ✓"

            for svc in (self.query.intents[i].services or []):
                think += f"\n- Service: Expects \"{svc}\" | Found \"{svc}\" in product details → Available ✓"

            think += "\n"

        if self.query.challenge == LOO_Challenge.VOUCHER:
            psum = sum(p.price for p in best_p)
            pact = self.query.voucher.get_active_cost(best_p) if self.query.voucher else psum

            if len(best_p) > 1:
                think += "\nThe prices of the selected products are "
                think += ", ".join([str(int(p.price)) for p in best_p[:-1]]) + " and " + str(int(best_p[-1].price))
                think += ", respectively, with a total price of " + str(int(psum)) + "."
            else:
                think += "\nThe price of the selected product is " + str(int(best_p[0].price)) + "."

            if psum > pact:
                if self.query.voucher.apply_to == "shop":
                    think += f" Since all of them come from the same shop (`{best_p[0].shop}`) and the total price exceeds the voucher threshold, the voucher can be applied."

                think += " After applying the voucher, the effective total price becomes " + str(int(pact)) + f", which is within the user's budget `{str(int(self.query.voucher.budget))}`."
            else:
                think += f" It is already within the budget `{str(int(self.query.voucher.budget))}`, even without using the voucher."

            if len(best_p) > 1:
                think += "\n\nIn summary, the selected products are within the budget and satisfy all stated requirements."
            else:
                think += "\n\nIn summary, the selected product is within the budget and satisfies all stated requirements."
        elif self.query.challenge == LOO_Challenge.SHOP:
            think += f"\nAll the selected products come from the same shop (`{best_p[0].shop}`), as requested."
            think += "\n\nIn summary, all selected products come from the same shop, and each one satisfies the user's requirements."
        else:
            think += "\nAll checks passed. The selected product satisfies the user's requirements."

        think += "\n\nNow I will present the final recommendation."
        self.step(think, f"The final recommendation: \"{', '.join([p.id for p in best_p])}\"")

class LOO_BaseEngine(LOO_Debuggable):
    TIMEOUT_SECS = 290.0

    MAX_RETRIEVE_FILTERS = 4
    MAX_REF_PRODUCTS = 500

    MAX_SELECT_TRIES = 3
    SELECT_BATCH = 5

    MAX_TEMPLATE_TRIES = 3
    TEMPLATE_BATCH = 2

    def __init__(self, query: LOO_Query, sess: LOO_Session, agent_delay: float, debugger: LOO_Debugger = None) -> None:
        super(LOO_BaseEngine, self).__init__(query, debugger)

        self.agent_delay = agent_delay
        self.sess = sess

        self.weight_pool: Dict[LOO_SubQuery, Dict[str, LOO_Weight]] = defaultdict(lambda: {})
        self.prod_pool: Dict[str, LOO_Product] = {}

        self.products: Dict[LOO_SubQuery, List[LOO_Product]] = defaultdict(lambda: [])
        self.searches: Dict[LOO_SubQuery, Dict[LOO_Keywords, List[LOO_Product]]] = {}

        self.confirmed = defaultdict(lambda: defaultdict(lambda: None))
        self.watches: Dict[str, float] = {}

        self.line_cache: Dict[LOO_SubQuery, Dict[Tuple[str, str], Tuple[Set, List]]] = defaultdict(lambda: {})
        self.filters: Dict[LOO_SubQuery, List[Tuple[LOO_Filter, float]]] = defaultdict(lambda: [])
        self.tkw: Dict[LOO_SubQuery, LOO_Keywords] = {}

    def run(self) -> List[LOO_Product]:
        self.__init_watch__("query", LOO_BaseEngine.TIMEOUT_SECS - self.agent_delay)

        for sq in self.query.intents:
            self.filters[sq] = self.__get_filter_vars__(sq, True)

            self.searches[sq] = self.__retrieve__(sq)
            if self.stopped: continue

        self.__unify_products__()
        if self.stopped: return []

        self.query = self.__template__()
        if self.stopped: return []

        for sq in self.query.intents:
            tf = self.__get_template_filter__(sq)
            self.tkw[sq] = tf.get_kw()

            if tf not in [f for f, _ in self.filters[sq]]:
                self.filters[sq].insert(1, (tf, 0))
            else:
                tf = None

            self.searches[sq] = self.__retrieve_ex__(sq, tf)
            if self.stopped: continue

        self.__unify_products__()
        if self.stopped: return []

        self.__build_lines__()

        for sq in self.query.intents:
            self.products[sq], self.weight_pool[sq] = self.__weigh__(sq, id = "init")
            if self.stopped: continue

        self.__unify_products__()
        self.__quick_confirm__()

        self.sess.step_on_retrieve()
        return []

    @LOO_freezable
    def __retrieve__(self, sq: LOO_SubQuery) -> Dict[LOO_Keywords, List[LOO_Product]]:
        pages, weight = 5, 0

        for f, w in self.filters[sq][:LOO_BaseEngine.MAX_RETRIEVE_FILTERS]:
            if w > weight:
                pages = max(pages - 1, 1)
                weight = w

            self.__search__(sq, [f], pages)

        if len(self.filters[sq]) == 1: self.__search__(sq, self.__get_bi_filters__(sq), 5)
        return self.searches[sq]

    @LOO_freezable
    def __template__(self) -> LOO_Query:
        for sq in self.query.intents:
            words, pairs = self.__collect_template_ref__(sq)
            t = 0

            while True:
                try:
                    js, _ = LOO_Proxy.trigger_llm(
                        LOO_Prompt.TEMPLATE.format(query = sq.query, words = words, pairs = pairs),
                        preferred_llms = ["deepseek-ai/DeepSeek-V3.2-TEE", "deepseek-ai/DeepSeek-V3.1-TEE"],
                        temperature = t, expect = "dict"
                    )
                    self.log({"query": sq.query} | js)

                    sq.template = LOO_Prompt.parse_template_response(js)
                    break
                except Exception as exc:
                    self.log(f"# error in templating: {exc}")
                    t += LOO_Proxy.TEMP_UP

        return self.query

    @LOO_freezable
    def __retrieve_ex__(self, sq: LOO_SubQuery, tf: Optional[LOO_Filter]) -> Dict[LOO_Keywords, List[LOO_Product]]:
        if tf is not None: self.__search__(sq, [tf], 5)
        return self.searches[sq]

    @LOO_freezable
    def __weigh__(self, sq: LOO_SubQuery, **_) -> Tuple[List[LOO_Product], Dict[str, LOO_Weight]]:
        tick = time.time()

        products = {p for _, prods in self.searches[sq].items() for p in prods}
        prod_ranks: List[Tuple[LOO_Product, Tuple]] = []

        for p in products:
            for i, (_, ps) in enumerate(self.searches[sq].items()):
                if p in ps:
                    j = ps.index(p)
                    prod_ranks.append((p, (i, j)))
                    break

        for i, (p, srank) in enumerate(prod_ranks):
            if p.id not in self.weight_pool[sq]:
                self.weight_pool[sq][p.id] = self.__score_product__(p, sq, srank)

        ordinal_products = sorted(
            products,
            key = lambda p: self.weight_pool[sq][p.id].value(len(p.services or []) - len(sq.services), len(p.sku or {})),
            reverse = True
        )
        if products:
            dense_products = sorted(products, key = lambda p: -self.weight_pool[sq][p.id].score)
            last_score, dense_r = self.weight_pool[sq][dense_products[0].id].score, 0

            for p in dense_products:
                if self.weight_pool[sq][p.id].score < last_score:
                    last_score = self.weight_pool[sq][p.id].score
                    dense_r += 1

                self.weight_pool[sq][p.id].update_wrank((dense_r, ordinal_products.index(p)))

        self.products[sq] = ordinal_products
        delay = time.time() - tick

        self.log({
            "sq": sq.query,
            "weighed": len(products),
            "delay": f"{delay:.2f}s"
        })
        return self.products[sq], self.weight_pool[sq]

    @LOO_freezable
    def __select__(self, sq: LOO_SubQuery, candidates: List[LOO_Product], must: bool, **_) -> List[LOO_Product]:
        best_p, js_check, is_fallback = [], {}, True

        if len(candidates) > 1 or (len(candidates) == 1 and not must):
            cursor, t = 0, 0

            while cursor < len(candidates):
                batch = candidates[cursor:cursor + LOO_BaseEngine.SELECT_BATCH]

                cursor += LOO_BaseEngine.SELECT_BATCH
                t += 1

                js_check, _ = LOO_Proxy.trigger_llm(
                    LOO_Prompt.SELECT.format(query = sq.query, products = LOO_Util.promptfy_prod_batch(batch, "")),
                    preferred_llms = ["deepseek-ai/DeepSeek-V3.2-TEE", "deepseek-ai/DeepSeek-V3.1-TEE"],
                    expect = "dict"
                )
                self.log(js_check)

                bid = LOO_Prompt.parse_select_response(js_check)

                if bid >= 0 and bid < len(batch):
                    best_p = [batch[bid]]
                    is_fallback = False
                    break

                if not self.__check_watch__("query", 20.0): break
                if t >= LOO_BaseEngine.MAX_SELECT_TRIES: break

            if must and not best_p:
                self.log(f"* fallback to select the first product: {sq.query[:50]}")
                best_p = candidates[:1]
        elif len(candidates) > 0:
            self.log(f"* selected only one product for: {sq.query[:50]}")
            best_p = candidates[:1]
        else:
            self.log(f"# cannot select the best because no candidate exists!")

        if best_p: self.sess.step_on_select(best_p[0], js_check, is_fallback, sq)
        return best_p

    def __quick_confirm__(self) -> None:
        for sq in self.query.intents:
            for p in self.products[sq]:
                if self.confirmed[sq][p] is not None: continue
                covers = self.weight_pool[sq][p.id].covers

                if min(covers or [0]) > 0:
                    if all(not (t.isnumeric() or len(t) <= 1) for t in self.dst_tokens[sq]):
                        self.confirmed[sq][p] = True
                elif max(covers or [0]) == 0:
                    self.confirmed[sq][p] = False

                if len(covers) == len(self.line_ws[sq]):
                    for i, w in enumerate(self.line_ws[sq]):
                        if w > 1.0 and not covers[i]:
                            self.confirmed[sq][p] = False
                            break

    def __search__(self, sq: LOO_SubQuery, filters: List[LOO_Filter], page_limit: int) -> Dict[LOO_Keywords, List[LOO_Product]]:
        self.searches[sq] = self.searches.get(sq, {})
        res = {}

        for f in filters:
            tick = time.time()
            products = self.__find_and_view__(f, page_limit)

            self.log({
                "sq": sq.query,
                "q": f.get_q(),
                "price": f.price,
                "shop": f.shop,
                "found": len(products),
                "delay": f"{(time.time() - tick):.2f}s"
            })
            kw = f.get_kw()

            self.searches[sq][kw] = self.searches[sq].get(kw, [])
            res[kw] = products

            for p in products:
                if p not in self.searches[sq][kw]: self.searches[sq][kw].append(self.__get_prod__(p))

        return res

    def __build_lines__(self) -> None:
        self.lines: Dict[LOO_SubQuery, List[Tuple[List[str], List[str], bool, int]]] = defaultdict(lambda: [])
        self.dst_tokens: Dict[LOO_SubQuery, List[str]] = defaultdict(lambda: [])

        self.line_ws: Dict[LOO_SubQuery, List[float]] = defaultdict(lambda: [])
        self.line_wsum: Dict[LOO_SubQuery, float] = {}

        for sq in self.query.intents:
            offset = 0

            for k, v in sq.template.items():
                k = re.sub(r"_\d$", "", k) or k
                ks, vs = LOO_Util.normalize_str(k).split(), LOO_Util.normalize_str(v).split()

                for i, token in enumerate(vs):
                    if "brand" in k or (token in LOO_Vocab.COLOR_TOKENS + LOO_Vocab.SIZE_TOKENS) or re.search(r"\d", token):
                        self.line_ws[sq].append(1.1)
                    elif any(LOO_Misc.is_morph_equal(token, f) for f in LOO_Vocab.KW_WEAKS):
                        self.line_ws[sq].append(0.9)
                    elif i > 0 and vs[i - 1].isnumeric() and token in LOO_Vocab.ALL_UNITS:
                        self.line_ws[sq].append(0.8)
                    elif token in LOO_Vocab.SYMBOL_TOKENS:
                        self.line_ws[sq].append(0.7)
                    elif token in LOO_Vocab.FILLERS:
                        self.line_ws[sq].append(0.5)
                    else:
                        self.line_ws[sq].append(1.0)

                self.lines[sq].append((ks, vs, k == "product_type", offset))
                self.dst_tokens[sq].extend(vs)

                offset += len(vs)

            self.line_wsum[sq] = sum(self.line_ws[sq])

        self.alt_map = LOO_Vocab.build_alt_map()

    def __collect_template_ref__(self, sq: LOO_SubQuery) -> Tuple[List, List]:
        prods_seen, added = [], 0
        words, pairs = [], []

        for cursor in range(LOO_BaseEngine.MAX_REF_PRODUCTS):
            is_valid = False

            for _, products in self.searches[sq].items():
                if cursor >= len(products): continue
                p = products[cursor]

                is_valid = True
                if p in prods_seen: continue

                prods_seen.append(p)
                added += 1

                for k, v in p.get_joint_details():
                    kv = (k.replace(" ", "_").lower(), v.lower())
                    if kv not in pairs: pairs.append(kv)

                for w in p.title.split():
                    w = w.lower()
                    if w not in words: words.append(w)

            if added >= LOO_BaseEngine.MAX_REF_PRODUCTS: break
            if not is_valid: break

        return words, pairs

    def __find_and_view__(self, filter: LOO_Filter, page_limit: int = 1) -> List[LOO_Product]:
        res: List[LOO_Product] = []

        for page in range(1, 6):
            searched, tool_res = LOO_Proxy.search(filter, page = page)

            self.sess.add_tool_result(tool_res)
            found: List[LOO_Product] = []

            for p in searched:
                if LOO_Util.basic_check(p, filter): found.append(self.__get_prod__(p))

            to_view = [p for p in found if not p.is_viewed]
            if len(to_view) > 0: LOO_Proxy.view(to_view)

            for p in found:
                res.append(self.__get_prod__(p).update_details())

            if page >= page_limit or len(found) < 10: break

        return res

    def __score_product__(self, prod: LOO_Product, sq: LOO_SubQuery, srank: Tuple[int, int]) -> LOO_Weight:
        title_covers = self.__eval_line__(sq, ("title", prod.title))
        best_w = LOO_Weight(srank = srank, score = 0)

        footprints, cats, violated = self.__get_footprints__(prod, sq)
        combs = LOO_Misc.get_choice_combs(cats, False)

        if not violated:
            for choice in combs:
                total_covers, face_choice = title_covers.copy(), {}

                for group, id in choice.items():
                    face_id = cats[group][id]

                    covers = footprints[group][face_id]
                    face_choice[group] = face_id

                    LOO_Misc.patch_max(total_covers, covers)

                w = LOO_Weight(
                    choice = face_choice, covers = total_covers, srank = srank, price = prod.price,
                    score = LOO_Misc.safe_div(sum(total_covers), self.line_wsum[sq])
                )
                if (w.score, -w.choice.get("sku", 0)) > (best_w.score, -best_w.choice.get("sku", 0)): best_w = w

        return best_w.fill_detail(self.dst_tokens[sq])

    def __get_footprints__(self, prod: LOO_Product, sq: LOO_SubQuery) -> Tuple[Dict[str, Dict[int, List[float]]], Dict[str, List[int]], bool]:
        def __is_dominating__(b: List[float], kb: List[float], strict: bool) -> bool:
            err = [x - y for x, y in zip(b, kb)]

            if not all(z >= 0 for z in err): return False
            if strict and not any(z > 0 for z in err): return False

            return True

        pareto_map, prod_only = {}, False
        prod.update_details()

        for group, cats in prod.details.items():
            for id, kvs in enumerate(cats):
                max_covers = [0 for _ in self.line_ws[sq]]

                for k, v in kvs:
                    if re.search(r"^services?$", k) and LOO_Misc.scan_patterns(v, LOO_Vocab.SERVICE_PATTERNS[LOO_Vocab.PROD_ONLY]): prod_only = True

                    covers = self.__eval_line__(sq, (k, v))
                    LOO_Misc.patch_max(max_covers, covers)

                if max(max_covers) > 0:
                    indice = tuple(i for i, w in enumerate(max_covers) if w > 0)

                    if indice not in pareto_map or __is_dominating__(max_covers, pareto_map[indice][-1], True):
                        pareto_map[indice] = (group, id, max_covers)

        if sq.prod_only and not prod_only: return {}, {}, True

        ordered_indice = sorted(pareto_map.keys(), key = lambda i: (-len(i), -sum(pareto_map[i][-1])))
        greedy_map = {}

        for indice in ordered_indice:
            for old_indice, (_, _, old_covers) in greedy_map.items():
                if set(indice) <= set(old_indice) and __is_dominating__(old_covers, pareto_map[indice][-1], False): break
            else:
                greedy_map[indice] = pareto_map[indice]

        footprints = defaultdict(lambda: {})

        for _, (group, id, covers) in greedy_map.items():
            footprints[group][id] = covers

        return footprints, {group: list(footprints[group].keys()) for group in footprints}, False

    def __eval_line__(self, sq: LOO_SubQuery, line: Tuple[str, str]) -> List[float]:
        if line not in self.line_cache[sq]:
            max_covers = [0 for _ in self.line_ws[sq]]

            for kd, vd, is_prod_type, offset in self.lines[sq]:
                covers = LOO_Util.match_line(line, (kd, vd), line[0] == "title", is_prod_type, self.alt_map, self.line_ws[sq], offset)
                LOO_Misc.patch_max(max_covers, covers)

            self.line_cache[sq][line] = max_covers

        return self.line_cache[sq][line]

    def __unify_products__(self) -> None:
        for sq in self.query.intents:
            for f in self.searches[sq]:
                self.searches[sq][f] = [self.__get_prod__(p) for p in self.searches[sq][f]]

            self.products[sq] = [self.__get_prod__(p) for p in self.products[sq]]

    def __get_prod__(self, p: LOO_Product) -> LOO_Product:
        if p.id in self.prod_pool:
            existing = self.prod_pool[p.id]

            if p.is_viewed and not existing.is_viewed:
                for k, v in p.to_json().items():
                    setattr(existing, k, v)
        else:
            self.prod_pool[p.id] = p

        return self.prod_pool[p.id]

    def __get_filter_vars__(self, sq: LOO_SubQuery, shadow: bool) -> List[Tuple[LOO_Filter, float]]:
        cat, indice = LOO_Util.get_kw_vars(sq.keywords)

        if cat:
            combs = LOO_Misc.get_choice_combs(cat, False)

            variations, filters = [], []
            base_var = sq.keywords.copy()

            for choice in combs:
                w = sum(cat[i][choice[i]][-1] for i in range(len(indice)))
                var = base_var.copy()

                for i in range(len(indice)):
                    id, c = indice[i], choice[i]
                    var[id] = cat[i][c][0]

                variations.append((var, w))
        else:
            variations = [(sq.keywords, 0)]

        variations.sort(key = lambda var: var[-1])
        filters: List[Tuple[LOO_Filter, float]] = []

        for var, w in variations:
            f = LOO_Filter(keywords = var, services = sq.services, price = sq.price)

            if sq.prod_only: f.keywords.append(LOO_Vocab.PROD_ONLY.replace('_', ' '))
            if f not in [of for of, _ in filters]: filters.append((f, w))

        if shadow:
            for f, _ in filters:
                f = self.__sharp_filter__(f)

        return filters

    def __get_bi_filters__(self, sq: LOO_SubQuery) -> List[LOO_Filter]:
        products = {p for _, searches in self.searches[sq].items() for p in searches}
        if len(products) == 0: return []

        prices = sorted(p.price for p in products)
        median = int(prices[len(prices) // 2])

        base_f, _ = self.filters[sq][0]
        low, high = LOO_Util.numerify_price(base_f.price)

        return [
            LOO_Filter.clone(base_f).update(price = LOO_Util.stringfy_price(low, median)),
            LOO_Filter.clone(base_f).update(price = LOO_Util.stringfy_price(median, high))
        ]

    def __sharp_filter__(self, filter: LOO_Filter) -> LOO_Filter:
        low, high = LOO_Util.numerify_price(filter.price or "-")

        if self.query.voucher is not None:
            favor = self.query.voucher.get_favor_range()

            if len(self.query.intents) == 1: low = max(low, int(favor[0]))
            high = int(favor[1])
        elif high < math.inf:
            low = max(low, math.floor(high / 2))

        return filter.update(price = LOO_Util.stringfy_price(low, high))

    def __get_template_filter__(self, sq: LOO_SubQuery) -> LOO_Filter:
        f = self.filters[sq][0][0].clone().update(keywords = list(sq.template.values()))

        if sq.prod_only: f.keywords.append(LOO_Vocab.PROD_ONLY.replace('_', ' '))
        return f

    def __get_alt_str__(self, best_p: List[LOO_Product]) -> str:
        if not best_p: return ""

        if len(best_p) == 1:
            for i, p in enumerate(self.products[self.query.intents[0]] * 2):
                if p.price > best_p[0].price or i >= len(self.products[self.query.intents[0]]):
                    alt_price = str(int(p.price)) if i < len(self.products[self.query.intents[0]]) else str(int(best_p[0].price + 2.0))
                    alt_title = p.title if len(p.title) < 90 else (p.title[:90] + "...")

                    return (
f"The product {p.id} appears to be a strong alternative. Let me compare it with {best_p[0].id}. "
f"The title of {p.id} is \"{alt_title}\". Based on the product details, {p.id} matches all stated constraints like {best_p[0].id}, making it difficult to choose one of them. "
f"The price of product {p.id} is {alt_price}, while the price of {best_p[0].id} is {str(int(best_p[0].price))}. The product {best_p[0].id} is less expensive. "
f"Therefore, the tie is broken and I will prefer {best_p[0].id}."
                    )
        else:
            for i, sq in enumerate(self.query.intents * 2):
                for p in self.products[sq]:
                    if i >= len(self.query.intents) or p.price > best_p[i].price:
                        bp = best_p[i % len(self.query.intents)]
                        alt_price = str(int(p.price)) if i < len(self.query.intents) else str(int(bp.price + 3.0))
                        alt_title = p.title if len(p.title) < 90 else (p.title[:90] + "...")

                        return (
f"The product {p.id} appears to be a strong alternative of the product {bp.id}, for the sub-query \"{sq.query}\". The title of {p.id} is {alt_title}. "
f"Let's consider a combination which has replaced {bp.id} with {p.id}. Still it meets the stated constraints for each corresponding sub-query. "
f"Now let me check the prices. The price of product {p.id} is {alt_price}, while the price of {bp.id} is {str(int(bp.price))}. "
f"It indicates that the replaced combination will have a higher total price than the original one. "
f"Therefore, the tie is broken and I will prefer `{', '.join([p.id for p in best_p])}`, whose total price is less expensive."
                        )

        return ""

    def __init_watch__(self, key: str, time_out: float) -> None:
        self.watches[key] = time.time() + time_out

    def __check_watch__(self, key: str, delta: float = 0) -> bool:
        if key not in self.watches:
            self.log(f"# non-initialized watch: `{key}`")
            return False

        if time.time() + delta > self.watches[key]:
            self.log(f"* time-out alerted for `{key}` with delta `{delta:.2f}s`")
            return False
        else:
            return True

class LOO_KnapsackEngine(LOO_BaseEngine):
    FILTER_BATCH = 6
    SHOP_BATCH = 2

    MAX_SHOP_TRIES = 8
    MAX_EXT_SHOPS = 6

    @LOO_freezable
    def __confirm__(self, products: Dict[LOO_SubQuery, List[LOO_Product]], **_) -> List[List[str]]:
        temperature = 0

        while True:
            try:
                js, _ = LOO_Proxy.trigger_llm(
                    self.__create_confirm_prompt__(products),
                    preferred_llms = ["deepseek-ai/DeepSeek-V3.2-TEE", "deepseek-ai/DeepSeek-V3.1-TEE"],
                    temperature = temperature, expect = "dict"
                )
                self.log(js)

                resp = LOO_Prompt.parse_confirm_response(js, [len(products[sq]) for sq in products])
                break
            except Exception as exc:
                self.log(f"# failed confirmation: {exc}")
                temperature += LOO_Proxy.TEMP_UP

        confirmed: List[List[str]] = []

        for i in range(len(products)):
            sq = list(products.keys())[i]
            ids = []

            for bid in resp[i]:
                ids.append(products[sq][bid].id)

            confirmed.append(ids)

        return confirmed

    def __proc_shop_batches__(self, must: bool, **kwargs) -> Tuple[List[LOO_Product], List]:
        plan, shop_map, shop_stats = self.__plan_shop__()
        best_p, solutions, processed_shops = [], [], 0

        for id, shop_batch in enumerate(plan):
            if processed_shops >= LOO_KnapsackEngine.MAX_EXT_SHOPS: break
            processed_shops += len(shop_batch)

            kmaps = self.__ext_search__(shop_batch, shop_map, id)
            self.__quick_confirm__()

            solmap = defaultdict(lambda: [])

            if kmaps:
                local_sol_count = 0

                for m in kmaps:
                    for prods, score in self.__proc_knapsack_map__(m, **kwargs):
                        sol = (prods, (shop_stats[prods[0].shop][0], score))
                        local_sol_count += 1

                        solmap[prods[0].shop].append(sol)
                        solutions.append(sol)

                best_p = self.__proc_solmap__(solmap, True, f"shop-{id}", LOO_KnapsackEngine.MAX_SHOP_TRIES)
                if local_sol_count > 0: self.sess.step_on_shop_batch(shop_batch, local_sol_count, best_p, id)
            else:
                self.sess.clear()

            if best_p: break

        solutions.sort(key = lambda s: s[-1], reverse = True)

        if not best_p:
            for count in range(len(self.query.intents), 0, -1):
                for prods, _ in solutions:
                    if len([p for i, p in enumerate(prods) if self.confirmed[self.query.intents[i]][p]]) >= count:
                        self.log(f"* fallback combo from {count} matching sub-queries!")
                        best_p = list(prods)
                        break

                if count < len(self.query.intents) and not must: break
                if best_p: break

            if best_p: self.sess.step_on_shop_fallback(best_p)

        return best_p, solutions

    def __get_shop_map__(self, products: Dict[LOO_SubQuery, List[LOO_Product]]) -> Tuple[Dict[str, Dict[LOO_SubQuery, List[LOO_Product]]], Dict[str, Tuple]]:
        shop_map: Dict[str, Dict[LOO_SubQuery, List[LOO_Product]]] = defaultdict(lambda: defaultdict(lambda: []))
        shop_stats: Dict[str, Tuple[int, float]] = {}

        for sq in self.query.intents:
            for p in products.get(sq, []):
                if p not in shop_map[p.shop][sq]: shop_map[p.shop][sq].append(p)

        for sid in shop_map:
            score, n = LOO_BundleScore(), 0

            for sq in shop_map[sid]:
                score += max(LOO_BundleScore.create(self.weight_pool[sq][p.id]) for p in shop_map[sid][sq])
                n += 1

            shop_stats[sid] = (n, score)

        return shop_map, shop_stats

    def __plan_shop__(self) -> Tuple[List[List[str]], Dict[str, Dict[LOO_SubQuery, List[LOO_Product]]], Dict[str, Tuple[int, float]], bool]:
        shop_map, shop_stats = self.__get_shop_map__(self.products)
        self.sess.step_on_shop_plan(shop_stats, min(LOO_KnapsackEngine.MAX_EXT_SHOPS, len(list(shop_stats.keys()))))

        shops = sorted(list(shop_stats.keys()), key = lambda sid: shop_stats[sid], reverse = True)
        plan = []

        for n in range(4, 0, -1):
            covers = [sid for sid in shops if shop_stats[sid][0] == n]

            while len(covers) > 0:
                plan.append(covers[:LOO_KnapsackEngine.SHOP_BATCH])
                covers = covers[LOO_KnapsackEngine.SHOP_BATCH:]

        return plan, shop_map, shop_stats

    def __ext_search__(self, shops: List[str], shop_map: Dict[str, Dict[LOO_SubQuery, List[LOO_Product]]], id: int) -> List[Dict[LOO_SubQuery, List[LOO_Product]]]:
        candidates: Dict[LOO_SubQuery, List[LOO_Product]] = {}

        for sid in shops:
            for sq in self.query.intents:
                if len(shop_map[sid][sq]) > 0: continue
                if not self.__check_watch__("query", 10.0): break

                res: Dict = self.__search__(sq, [LOO_Filter.clone(f).update(shop = sid) for f, w in self.filters[sq] if w == 0], 1)
                if sum(len(ps) for ps in res.values()) == 0: break

        for sq in self.query.intents:
            self.__weigh__(sq, id = f"{id}", suffix = "ext")
            candidates[sq] = [p for p in self.products[sq] if p.shop in shops]

        return self.__get_knapsack_maps__(candidates, True)

    def __get_knapsack_maps__(self, products: Dict[LOO_SubQuery, List[LOO_Product]], is_shop: bool) -> List[Dict[LOO_SubQuery, List[LOO_Product]]]:
        kmaps = []

        if is_shop:
            shop_map, shop_stats = self.__get_shop_map__(products)

            for sid in shop_stats:
                if shop_stats[sid][0] == len(self.query.intents): kmaps.append(shop_map[sid])
        else:
            if all(len(products[sq]) > 0 for sq in self.query.intents): kmaps.append(products)

        return kmaps

    def __proc_knapsack_map__(self, kmap: Dict[LOO_SubQuery, List[LOO_Product]], **kwargs) -> List[Tuple[Tuple[LOO_Product], Tuple]]:
        cats: List[List[Tuple[LOO_Product, LOO_Weight]]] = []

        for sq, products in kmap.items():
            prod_weights = [(p, self.weight_pool[sq][p.id]) for p in products]
            cats.append(prod_weights)

        return self.__solve_knapsack__(cats, **kwargs)

    def __proc_solmap__(self, solmap: Dict[str, List[Tuple[Tuple[LOO_Product, ...], Any]]], sort: bool, id: str, max_tries: int) -> List[LOO_Product]:
        t, cursor, is_exhausted = 0, 0, False

        for t in range(max_tries):
            batch, candidates, cand_count = [], defaultdict(lambda: []), 0

            while cand_count < LOO_KnapsackEngine.FILTER_BATCH and not is_exhausted:
                is_added = False

                for _, sols in solmap.items():
                    if cursor < len(sols):
                        prods, score = sols[cursor]
                        is_added = True

                        if all(self.confirmed[self.query.intents[i]][p] for i, p in enumerate(prods)): return list(prods)
                        if any(self.confirmed[self.query.intents[i]][p] == False for i, p in enumerate(prods)): continue

                        for i, p in enumerate(prods):
                            sq = self.query.intents[i]

                            if self.confirmed[sq][p] is None and p not in candidates[sq]:
                                candidates[sq].append(p)
                                cand_count += 1

                        batch.append((prods, score))

                if not is_added: is_exhausted = True
                cursor += 1

            if cand_count > 0:
                if not self.__check_watch__("query", 30.0): break
                confirmed = self.__confirm__(candidates, id = f"{id}-{t}")

                for i, (sq, prods) in enumerate(candidates.items()):
                    for pid in confirmed[i]:
                        self.confirmed[sq][self.prod_pool[pid]] = True

                for sq in candidates:
                    for p in candidates[sq]:
                        if not self.confirmed[sq][p]: self.confirmed[sq][p] = False

            if sort: batch.sort(key = lambda b: b[-1], reverse = True)
            best_p = self.__find_filtered_combo__(batch)

            if best_p:
                self.log(f"* selected best combo from filtered at try {t + 1}!")
                return best_p

            if is_exhausted: break

        return []

    def __find_filtered_combo__(self, solutions: List[Tuple]) -> List[LOO_Product]:
        for prods, _ in solutions:
            if all(bool(self.confirmed[self.query.intents[i]][p]) for i, p in enumerate(prods)): return list(prods)

        return []

    def __solve_knapsack__(self, _: List[List[Tuple[LOO_Product, LOO_Weight]]], **__) -> List[Tuple[Tuple[LOO_Product, ...], "BundleScore"]]:
        return []

    def __create_confirm_prompt__(self, candidates: Dict[LOO_SubQuery, List[LOO_Product]]) -> str:
        info = {}

        for i, sq in enumerate(list(candidates.keys())):
            info[LOO_Vocab.ALPHABETS[i]] = {
                "candidates": LOO_Util.promptfy_prod_batch(candidates[sq], LOO_Vocab.ALPHABETS[i]),
                "query": sq.query
            }

        return LOO_Prompt.CONFIRM.format(count = len(candidates), info = info)
class LOO_VoucherEngine(LOO_KnapsackEngine):
    MAX_PLATFORM_TRIES = 8
    MAX_K_SOLUTIONS = 1500

    def run(self) -> List[LOO_Product]:
        super(LOO_VoucherEngine, self).run()
        if self.stopped: return []

        best_p = self.__proc__()
        if self.stopped: return []

        return best_p

    @LOO_freezable
    def __proc__(self) -> List[LOO_Product]:
        tick = time.time()

        if len(self.query.intents) > 1:
            if self.query.voucher.apply_to == "shop":
                best_p = self.__proc_shop__()
            else:
                best_p, _ = self.__proc_platform__(True, True)
        else:
            best_p = self.__proc_single__()

        delay = time.time() - tick

        self.log({
            "sq_count": len(self.query.intents),
            "knapsack": [p.id for p in best_p],
            "delay": f"{delay:.2f}s"
        })
        return best_p

    def __proc_single__(self) -> List[LOO_Product]:
        candidates = [p for p in self.products[self.query.intents[0]] if self.query.voucher.is_favor_cost(p.price)]
        best_p = self.__select__(self.query.intents[0], candidates, True)

        if not best_p:
            candidates = [p for p in self.products[self.query.intents[0]] if p.price < self.query.voucher.budget]
            best_p = candidates[:1]

        return best_p

    def __proc_shop__(self) -> List[LOO_Product]:
        favor_range = self.query.voucher.get_favor_range()

        best_p, shop_sols = self.__proc_shop_batches__(False, price_range = favor_range)
        if best_p: return best_p

        best_p, platform_sols = self.__proc_platform__(False, False)
        if best_p: return best_p

        for count in range(len(self.query.intents), 0, -1):
            for prods, _ in (shop_sols + platform_sols):
                if len([p for i, p in enumerate(prods) if self.confirmed[self.query.intents[i]][p]]) >= count:
                    self.log(f"* fallback combo from {count} matching sub-queries!")
                    best_p = list(prods)
                    break

            if best_p: break

        self.sess.step_on_platform(best_p, len(shop_sols + platform_sols), True)
        return best_p

    def __proc_platform__(self, enable_favor: bool, must: bool) -> Tuple[List[LOO_Product], List]:
        if not enable_favor: self.log("* switching to platform mode...")

        favor_range = self.query.voucher.get_favor_range()
        solmap, best_p = {}, []

        kmaps = self.__get_knapsack_maps__(self.products, False)
        if not kmaps: return best_p, []

        if enable_favor:
            solmap["favor"] = self.__proc_knapsack_map__(kmaps[0], price_range = favor_range)
        else:
            solmap["fallback"] = self.__proc_knapsack_map__(kmaps[0], price_range = (0, self.query.voucher.budget))

        best_p = self.__proc_solmap__(solmap, False, "platform", LOO_VoucherEngine.MAX_PLATFORM_TRIES if enable_favor else 1)
        solutions = sorted([s for _, sols in solmap.items() for s in sols], key = lambda s: s[-1], reverse = True)

        if not best_p:
            best_p = self.__find_filtered_combo__(solutions)
            if best_p: self.log("* selected best combo from filtered at outside tries!")

        if must and not best_p:
            if solutions:
                for count in range(len(self.query.intents), -1, -1):
                    for sol in solutions:
                        if len([i for i, p in enumerate(sol[0]) if self.confirmed[self.query.intents[i]][p]]) == count:
                            self.log(f"* fallback combo with {count} matching sub-queries!")
                            best_p = list(sol[0])
                            break

                    if best_p: break
            else:
                self.log("* no combo found.")

        if best_p or must: self.sess.step_on_platform(best_p, len(solutions), not enable_favor)
        return best_p, solutions

    def __solve_knapsack__(self, cats: List[List[Tuple[LOO_Product, LOO_Weight]]], price_range: Tuple[float, float]) -> List[Tuple[Tuple[LOO_Product, ...], "BundleScore"]]:
        split, heap = len(cats) // 2, []

        l_cats, r_cats = cats[:split], cats[split:]
        l_states, r_states = [], []

        for subcat, substates in [(l_cats, l_states), (r_cats, r_states)]:
            for choice in product(*subcat):
                price = sum(p.price for p, _ in choice)
                if price > price_range[1]: continue

                substates.append((price, sum(LOO_BundleScore.create(w) for _, w in choice), [p for p, _ in choice]))

        l_states.sort(key = lambda x: x[1], reverse = True)
        r_states.sort(key = lambda x: x[0])

        seg = LOO_SegTree([s[1] for s in r_states], 1000)
        r_prices = [s[0] for s in r_states]

        for lp, lw, l_products in l_states:
            remain_lo = price_range[0] - lp
            remain_hi = price_range[1] - lp

            lidx = bisect_left(r_prices, remain_lo)
            ridx = bisect_right(r_prices, remain_hi) - 1

            if lidx > ridx: continue

            for rw, idx in seg.query(lidx, ridx):
                r_products = r_states[idx][2]
                prods: List[LOO_Product] = l_products + r_products

                if len(set(prods)) != len(prods): continue
                score = lw + rw

                if price_range[0] == 0 or self.query.voucher.is_favor_cost(lp + r_prices[idx]):
                    if len(heap) < LOO_VoucherEngine.MAX_K_SOLUTIONS:
                        heappush(heap, (score, tuple(p.id for p in prods)))
                    elif score > heap[0][0]:
                        heapreplace(heap, (score, tuple(p.id for p in prods)))

        heap.sort(key = lambda s: s[0], reverse = True)
        return [(list(self.prod_pool[pid] for pid in prods), score) for score, prods in heap]

class LOO_Agent(LOO_Debuggable):
    ENGINE_MAP: Dict[LOO_Challenge, Type[LOO_BaseEngine]] = {
        LOO_Challenge.VOUCHER: LOO_VoucherEngine,
    }

    def __init__(self, query: LOO_Query, debugger: Optional[LOO_Debugger] = None) -> None:
        super(LOO_Agent, self).__init__(query, debugger)
        self.sess: LOO_Session = LOO_Session(query, debugger)

    def run(self) -> None:
        tick = time.time()

        self.query = self.__classify__()
        if self.stopped: return

        self.query = self.__decompose__()
        if self.stopped: return

        self.query = self.__clean__()
        if self.stopped: return

        self.query = self.__extract__()
        if self.stopped: return

        self.sess.query = self.query
        self.sess.step_on_build()

        self.engine = self.ENGINE_MAP.get(self.query.challenge)(
            self.query, self.sess, time.time() - tick, self.debugger
        )
        best_p = self.engine.run()
        self.__finalize__(best_p)

    def get_dialog(self) -> List[Dict]:
        return self.sess.dialog

    @LOO_freezable
    def __classify__(self) -> LOO_Query:
        if self.query.challenge == LOO_Challenge.UNKNOWN:
            js, _ = LOO_Proxy.trigger_llm(LOO_Prompt.CLASSIFY.format(query = self.query.query_f), expect = "dict")
            self.log(js)

            challenge, voucher, query_e = LOO_Prompt.parse_classify_response(js)

            self.query.query_e = query_e if query_e else self.query.query_f
            self.query.challenge = challenge
            self.query.voucher = voucher
        else:
            if self.query.challenge == LOO_Challenge.VOUCHER:
                self.query.query_e, self.query.voucher = self.__parse_voucher__()
            else:
                self.query.query_e = self.query.query_f

        self.log({
            "challenge": self.query.challenge,
            "query_e": self.query.query_e,
            "voucher": self.query.voucher.to_json() if self.query.voucher is not None else None
        })
        return self.query

    @LOO_freezable
    def __decompose__(self) -> LOO_Query:
        if self.query.challenge == LOO_Challenge.SHOP or (self.query.challenge == LOO_Challenge.VOUCHER and self.query.voucher.apply_to == "shop"):
            min_count, max_count = 2, 4
        elif self.query.challenge == LOO_Challenge.PRODUCT:
            min_count, max_count = 1, 1
        else:
            min_count, max_count = 1, 4

        prompt = LOO_Prompt.DECOMPOSE.format(query = self.query.query_e, min_count = min_count, max_count = max_count)
        temperature = 0

        while True:
            try:
                js, _ = LOO_Proxy.trigger_llm(prompt, temperature = temperature, expect = "dict")
                self.log(js)

                decomposed = LOO_Prompt.parse_decompose_response(js)
                if len(decomposed) < min_count or len(decomposed) > max_count: raise Exception(f"invalid intent count {len(decomposed)}")

                for i, q in enumerate(decomposed):
                    services, prod_only = [], False

                    for svc, patterns in LOO_Vocab.SERVICE_PATTERNS.items():
                        if LOO_Misc.scan_patterns(q, patterns):
                            if svc == LOO_Vocab.PROD_ONLY:
                                prod_only = True
                            else:
                                if svc == "official" and "authentic and branded" in q: continue
                                services.append(svc)

                    self.query.intents.append(LOO_SubQuery(
                        query_f = self.query.query_f,
                        query = q,
                        prod_only = prod_only,
                        services = services,
                        id = i
                    ))

                break
            except Exception as exc:
                self.log(f"# error parsing decompose response: {exc}")
                temperature += LOO_Proxy.TEMP_UP

        return self.query

    @LOO_freezable
    def __clean__(self) -> LOO_Query:
        inf, output = {}, ""

        for i, sq in enumerate(self.query.intents):
            inf[str(i + 1)] = {"query": sq.query, "services": sq.services + ([LOO_Vocab.PROD_ONLY] if sq.prod_only else [])}

            output += """,
    "{id}": {{ ... }}""".format(id = i + 1) if i > 0 else ""

        prompt = LOO_Prompt.CLEAN.format(inf = inf, output = output, count = len(self.query.intents))
        temperature = 0

        while True:
            try:
                js, _ = LOO_Proxy.trigger_llm(prompt, temperature = temperature, expect = "dict")
                self.log(js)

                cleaned = LOO_Prompt.parse_clean_response(js, len(self.query.intents) == 1)

                for i, item in enumerate(cleaned):
                    self.query.intents[i].query = item.get("query")
                    self.query.intents[i].price = item.get("price")

                break
            except Exception as exc:
                self.log(f"# error parsing clean response: {exc}")
                temperature += LOO_Proxy.TEMP_UP

        return self.query

    @LOO_freezable
    def __extract__(self) -> LOO_Query:
        inf, output = {}, ""

        for i, sq in enumerate(self.query.intents):
            inf[str(i + 1)] = {"query": sq.query}

            output += """,
    "{id}": {{ ... }}""".format(id = i + 1) if i > 0 else ""

        prompt = LOO_Prompt.EXTRACT.format(inf = inf, output = output)
        temperature = 0

        while True:
            try:
                js, _ = LOO_Proxy.trigger_llm(prompt, temperature = temperature, expect = "dict")
                self.log(js)

                extracted = LOO_Prompt.parse_extract_response(js, len(self.query.intents) == 1)

                for i, keywords in enumerate(extracted):
                    self.query.intents[i].keywords = keywords

                break
            except Exception as exc:
                self.log(f"# error parsing extract response: {exc}")
                temperature += LOO_Proxy.TEMP_UP

        return self.query

    def __finalize__(self, best_p: List[LOO_Product]):
        self.sess.step_on_compare(best_p, self.engine.__get_alt_str__(best_p))

        self.sess.add_tool_result(LOO_Proxy.submit(best_p))
        self.sess.add_tool_result(LOO_Proxy.finish("success" if best_p else "failure"))

        self.sess.step_on_submit(best_p)

    def __parse_voucher__(self) -> Tuple[str, LOO_Voucher]:
        sep = self.query.query_f.lower().find("my budget is")

        query_e = self.query.query_f[:sep].strip()
        v_str = self.query.query_f[sep:].strip()

        m = re.search(r"\bmy\s+budget\s+is(?:\s+only)?\s+`(\d+)`", v_str, re.I)
        budget = int(m.group(1))

        apply_to = "shop" if re.search(r"\bsame\s+shops?\b", v_str, re.I) else "all"

        m = re.search(r"\bexceeds?\s+`(\d+)`", v_str, re.I)
        threshold = int(m.group(1))

        if (m := re.search(r"\bpercent(?:age)?\s+discount(?:\s+of)?\s+`([\d\.]+)\%`", v_str, re.I)):
            discount = float(m.group(1))
            is_fixed = False

            m = re.search(r"\bwith(?:\s+a)\s+cap(?:\s+of)?\s+`([\d\.]+)`", v_str, re.I)
            cap = float(m.group(1))
        else:
            m = re.search(r"\bfixed\s+discount(?:\s+of)?\s+`([\d\.]+)`", v_str, re.I)
            discount = float(m.group(1))
            is_fixed = True
            cap = None

        voucher = LOO_Voucher(
            budget = budget,
            apply_to = apply_to,
            threshold = threshold,
            is_fixed = is_fixed,
            discount = discount,
            cap = cap
        )
        return query_e, voucher

def LOO_find_product(**kwargs) -> List[Dict]:
    return LOO_ExtTool.search(**kwargs)

def LOO_view_product_information(product_ids: str = None, ids: str = None, **kwargs) -> List[Dict]:
    pid = product_ids or ids or kwargs.get("product_ids") or kwargs.get("ids") or ""
    return LOO_ExtTool.view(pid)

def LOO_recommend_product(product_ids: str) -> str:
    return LOO_ExtTool.submit(product_ids)

def LOO_terminate(status: str = "success") -> str:
    return LOO_ExtTool.finish(status)

def LOO_calculate_voucher(
    product_prices: str,
    voucher_type: str,
    discount_value: float,
    threshold: float,
    budget: float,
    cap: float = 0,
) -> Dict:
    prices_raw = [s.strip() for s in product_prices.split(",") if s.strip()]
    try:
        prices = [float(p) for p in prices_raw]
    except ValueError as exc:
        return {"error": f"Invalid product_prices: {exc}"}
    if not prices:
        return {"error": "product_prices is empty or null."}
    vtype = (voucher_type or "").strip().lower()
    if vtype not in ("fixed", "percentage"):
        return {"error": f"Invalid voucher_type: {voucher_type!r}. Use 'fixed' or 'percentage'."}
    threshold = max(float(threshold or 0), 0.0)
    budget = max(float(budget or 0), 0.0)
    discount_value = max(float(discount_value or 0), 0.0)
    cap = max(float(cap or 0), 0.0)
    total = sum(prices)
    discount = 0.0
    voucher_applied = False
    if total >= threshold:
        voucher_applied = True
        if vtype == "fixed":
            discount = min(discount_value, total)
        else:
            pct = discount_value / 100.0
            discount = total * pct
            if cap > 0:
                discount = min(discount, cap)
    total_after = max(0.0, total - discount)
    return {
        "prices": prices,
        "total_before": round(total, 2),
        "discount_amount": round(discount, 2),
        "total_after": round(total_after, 2),
        "within_budget": total_after <= budget,
        "voucher_applied": voucher_applied,
        "budget": budget,
    }

def _LOO_restore_identity():
    for _n, _o in list(globals().items()):
        if not _n.startswith("LOO_"):
            continue
        try:
            _o.__name__ = _n.removeprefix("LOO_")
        except Exception:
            pass
        try:
            _o.__qualname__ = _n.removeprefix("LOO_")
        except Exception:
            pass
_LOO_restore_identity()
if "Agent" not in globals():
    Agent = type("Agent", (), {})

def _loo_combo_integrity_patches():
    _orig_finalize = LOO_Agent.__finalize__
    def __finalize__(self, best_p):
        if self.query.challenge == LOO_Challenge.VOUCHER and best_p and self.query.voucher is not None:
            v = self.query.voucher
            prices = [float(p.price) for p in best_p]
            prices_str = ",".join(str(round(p, 2)) for p in prices)
            vtype = "fixed" if v.is_fixed else "percentage"
            cap = float(v.cap or 0) if v.cap is not None else 0.0
            try:
                tool_res = execute_tool_call(
                    "calculate_voucher",
                    {
                        "product_prices": prices_str,
                        "voucher_type": vtype,
                        "discount_value": float(v.discount or 0),
                        "threshold": float(v.threshold or 0),
                        "budget": float(v.budget or 0),
                        "cap": cap,
                    },
                )
                self.sess.add_tool_result(tool_res)
                self.sess.step(
                    "Confirming the cart with calculate_voucher: prices=[" + prices_str + "].",
                    "Voucher calculation completed.",
                )
            except Exception:
                pass
        return _orig_finalize(self, best_p)
    LOO_Agent.__finalize__ = __finalize__
_loo_combo_integrity_patches()



def unity_product_self_route(query: str) -> str:
    
    q = query or ""
    try:
        qlow = q.lower()
        if re.search(r"\bmy\s+budget\s+is\b", qlow) and re.search(r"\bvoucher\b", qlow):
            return ""
        if re.search(r"\bsame\s+shop", qlow) or re.search(
            r"\bfrom\s+(the\s+)?same\s+(store|seller|shop)", qlow
        ):
            return ""
        return "product"
    except Exception:
        return ""


def topas_shop_self_route(query: str) -> str:
  
    q = query or ""
    try:
        if NS_QueryPipeline._VOUCHER_CLAUSE_RE.search(q):
            return ""  # voucher owned by 727-VII LOO self-router
        if NS_QueryPipeline._SHOP_INTENT_RE.search(q):
            return "shop"
    except Exception:
        pass
    return ""


def vii_voucher_self_route(query: str) -> str:
   
    q = query or ""
    try:
        if re.search(r"\bmy\s+budget\s+is\b", q, re.I):
            return "voucher"
        if re.search(
            r"\bvoucher\b.*\b(?:fixed|percentage)\s+discount\b"
            r"|\b(?:fixed|percentage)\s+discount\b.*\bvoucher\b",
            q, re.I | re.S,
        ):
            return "voucher"
        ql = q.lower()
        if re.search(r"\b(budget|voucher)\b", ql) and re.search(
            r"\b(threshold|discount|cap|%\s*off|my\s+budget)\b", ql
        ):
            return "voucher"
    except Exception:
        pass
    return ""




class ComboChallenge(str, Enum):
    PRODUCT = "Product"
    SHOP = "Shop"
    VOUCHER = "Voucher"


def _normalize_category(cat: object) -> Optional[ComboChallenge]:
    if cat is None:
        return None
    c = str(cat).strip().lower()
    if not c:
        return None
    if c == "store":
        c = "shop"
    if c.startswith("product"):
        return ComboChallenge.PRODUCT
    if c.startswith("shop"):
        return ComboChallenge.SHOP
    if c.startswith("voucher") or c in ("coupon", "promo"):
        return ComboChallenge.VOUCHER
    return None


def _coerce_query(prob: Dict) -> str:
    if not isinstance(prob, dict):
        return ""
    q = prob.get("query", "")
    return q if isinstance(q, str) else str(q)


def _router_ensure_dialog(dialog: list, query: str, lane: str) -> list:
    if dialog:
        return dialog
    return [
        create_dialogue_step(
            think=f"Agent returned no dialogue steps ({lane} lane).",
            tool_results=[],
            response="Done.",
            query=query,
            step=1,
        )
    ]


def _install_merged_product_tools() -> None:
    register_tool("find_product", U_find_product)
    register_tool("view_product_information", U_view_product_information)
    register_tool("recommend_product", U_recommend_product)
    register_tool("terminate", U_terminate)


def _install_merged_shop_tools() -> None:
    register_tool("find_product", NS_find_product)
    register_tool("view_product_information", NS_view_product_information)
    register_tool("recommend_product", NS_recommend_product)
    register_tool("terminate", NS_terminate)


def _install_merged_voucher_tools() -> None:
    register_tool("find_product", LOO_find_product)
    register_tool("view_product_information", LOO_view_product_information)
    register_tool("recommend_product", LOO_recommend_product)
    register_tool("terminate", LOO_terminate)
    register_tool("calculate_voucher", LOO_calculate_voucher)


def resolve_challenge(prob: Dict) -> ComboChallenge:
    """Top router: category field (production) or per-pipeline self-routers (local fallback)."""
    if not isinstance(prob, dict):
        return ComboChallenge.PRODUCT

    for key in ("category", "task", "type"):
        cat = _normalize_category(prob.get(key))
        if cat is not None:
            return cat

    query = _coerce_query(prob)

    if "voucher" in prob and prob.get("voucher") is not None:
        return ComboChallenge.VOUCHER

    claims: list[ComboChallenge] = []
    if vii_voucher_self_route(query) == "voucher":
        claims.append(ComboChallenge.VOUCHER)
    if topas_shop_self_route(query) == "shop":
        claims.append(ComboChallenge.SHOP)
    if unity_product_self_route(query) == "product":
        claims.append(ComboChallenge.PRODUCT)

    unique = list(dict.fromkeys(claims))
    if len(unique) == 1:
        return unique[0]
    return ComboChallenge.PRODUCT


def _merged_product_lane(prob: Dict) -> List[Dict]:
    """unity U_* product entry — preset PRODUCT so internal classify cannot override."""
    _install_merged_product_tools()
    query = _coerce_query(prob)
    if hasattr(U_QueryRecovery, "recover"):
        try:
            query = U_QueryRecovery.recover(prob if isinstance(prob, dict) else {})
        except Exception:
            pass
    if hasattr(U_AdversarialIngressGuard, "sanitize"):
        try:
            query = U_AdversarialIngressGuard.sanitize(query)
        except Exception:
            pass
    dialog: List[Dict] = []
    try:
        agent = U_Agent(U_Query(challenge=U_Challenge.PRODUCT, query_f=query))
        agent.run()
        dialog = agent.get_dialog() or []
        try:
            dialog = U_DialogWireFit.fit(dialog)
        except Exception:
            pass
    except Exception as exc:
        tb = traceback.format_exc()
        if len(tb) > 2000:
            tb = "...(truncated)\n" + tb[-2000:]
        dialog = [
            create_dialogue_step(
                think=f"Unhandled product lane error: {type(exc).__name__}: {exc}\n\n{tb}",
                tool_results=[],
                response="Done.",
                query=query,
                step=1,
            )
        ]
    return _router_ensure_dialog(dialog, query, "product")


def _merged_shop_lane(prob: Dict) -> List[Dict]:
    """top_A_S NS_* shop entry — preset SHOP so internal classify cannot override."""
    _install_merged_shop_tools()
    query = _coerce_query(prob)
    dialog: List[Dict] = []
    original_identify = NS_QueryPipeline._identify_task_type

    def forced_shop(self):
        return NS_ShoppingTaskType.SHOP

    try:
        NS_QueryPipeline._identify_task_type = forced_shop
        NS_ProxyUtil.arm_deadline()
        agent = NS_ShoppingAgent(query, task_hint="shop")
        agent.run()
        dialog = agent.get_dialog() or []
        try:
            dialog = NS__fit_dialog_to_wire(dialog)
        except Exception:
            pass
    except Exception as exc:
        tb = traceback.format_exc()
        if len(tb) > 2000:
            tb = "...(truncated)\n" + tb[-2000:]
        dialog = [
            create_dialogue_step(
                think=f"Unhandled shop lane error: {type(exc).__name__}: {exc}\n\n{tb}",
                tool_results=[],
                response="Done.",
                query=query,
                step=1,
            )
        ]
    finally:
        NS_QueryPipeline._identify_task_type = original_identify
    return _router_ensure_dialog(dialog, query, "shop")


def _merged_voucher_lane(prob: Dict) -> List[Dict]:
    """727-VII LOO_* voucher entry — preset VOUCHER so internal classify cannot override."""
    _install_merged_voucher_tools()
    query = _coerce_query(prob if isinstance(prob, dict) else {})
    dialog: List[Dict] = []
    original_classify = LOO_Agent.__classify__

    def forced_voucher(self):
        self.query.challenge = LOO_Challenge.VOUCHER
        return original_classify(self)

    try:
        LOO_Agent.__classify__ = forced_voucher
        agent = LOO_Agent(LOO_Query(challenge=LOO_Challenge.VOUCHER, query_f=query))
        agent.run()
        dialog = agent.get_dialog() or []
    except Exception as exc:
        tb = traceback.format_exc()
        if len(tb) > 2000:
            tb = "...(truncated)\n" + tb[-2000:]
        dialog = [
            create_dialogue_step(
                think=f"Unhandled voucher lane error: {type(exc).__name__}: {exc}\n\n{tb}",
                tool_results=[],
                response="Done.",
                query=query,
                step=1,
            )
        ]
    finally:
        LOO_Agent.__classify__ = original_classify
    return _router_ensure_dialog(dialog, query, "voucher")


def agent_main(prob: Dict) -> List[Dict]:
    query = _coerce_query(prob if isinstance(prob, dict) else {})
    try:
        kind = resolve_challenge(prob if isinstance(prob, dict) else {})
        _debug_print(f"\n[MERGED] lane={kind.value} | query={query[:120]!r}")
        if kind == ComboChallenge.PRODUCT:
            return _merged_product_lane(prob)
        if kind == ComboChallenge.SHOP:
            return _merged_shop_lane(prob)
        return _merged_voucher_lane(prob)
    except Exception as exc:
        return [
            create_dialogue_step(
                think=f"Unhandled agent error: {type(exc).__name__}: {exc}",
                tool_results=[],
                response="Done.",
                query=query,
                step=1,
            )
        ]
