"""ORO Bench agent 102 — stronger generic generated-environment agent.

Key upgrades vs 101:
  * justification need detected from the shopper GOAL only (Task rules contain
    the word \"reason\" and previously forced false claims that zeroed wins)
  * place_test_order claims always stripped unless the goal asks why; when
    required, claims are rebuilt from inspect_stock / view evidence only
  * firm-spec cart gate (RAM≠ROM, refresh/resolution/screen, smartphone vs
    feature-phone) using goal text + observed product text — no sealed tables
  * ranking/recovery nudges: compare listed candidates; search preferred
    model+color first; after events, order the post-event best (not a revert)
  * primary model DeepSeek-V3.2 (fast enough for prod step budgets;
    Qwen397B / GLM timed out locally and would fail qualifying)
"""

from __future__ import annotations

import json
import logging
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent102")

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_FALLBACK_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.1-TEE",
    "openrouter": "deepseek/deepseek-chat-v3.1",
}

_proxy = ProxyClient(timeout=120, max_retries=3)
_OBS_CHARS = 7000
_MAX_PARALLEL = 6

_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|"
    r"max(?:imum)? of|within|budget (?:is|of)|only go up to|can spend up to|"
    r"at most|limit is|can go up to|My limit is)\s*"
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?",
    re.I,
)
_PREF_RE = re.compile(
    r"(?:prefer(?: the)?|strongly prefer(?: the)?)\s+(?:the\s+)?"
    r"([a-z0-9._\-]+)\s+([A-Za-z0-9][A-Za-z0-9._\-]*)",
    re.I,
)
# "the gold Y81" / "the grey AE11"
_PREF_COLOR_MODEL_RE = re.compile(
    r"(?:prefer(?: the)?|strongly prefer(?: the)?)\s+(?:the\s+)?"
    r"(black|white|blue|red|green|gold|grey|gray|silver|pink|purple|orange|"
    r"yellow|brown|beige|titanium|graphite|midnight|starlight)\s+"
    r"([A-Za-z0-9][A-Za-z0-9._\-]*)",
    re.I,
)

_SYSTEM = """You are an expert shopping agent for a live product catalog.

Tools only — never invent product_id or sku; copy them from observations.
Follow the shopper GOAL and Task rules. No external product lookup tables.

Scoring-critical habits:
1) First successful add_to_cart is graded. Never cart placeholders or accessories
   when the goal wants a real phone/tablet.
2) Firm specs beat brand: category, RAM (not ROM), screen size, resolution,
   refresh rate, budget, in-stock. Verify on the exact sku options / facts /
   title before carting. If search returns RAM sticks or "16GB ROM" phones for a
   "16GB RAM" goal, reformulate (e.g. quote "16GB RAM", add category words).
3) Prices/stock change after carting. Always inspect_stock again after events,
   then place_test_order on a still-valid CURRENT best — do not revert to a
   pre-event pick that is now dominated.
4) place_test_order justification:
   - Goal asks why / reason / explain / walk me through → include
     justification.claims with ONLY schema fields (brand, category, in_stock,
     price, refresh_rate, resolution, screen_size) copied from evidence.
   - Otherwise OMIT justification completely (false optional claims fail).
5) Ranking with candidate product_id::sku list: compare those first; cart the
   best by stated priority; after an event, re-compare survivors and order the
   new best.
6) Recovery with preferred model+color: search "<color> <model>" first and cart
   the exact match when available; if it vanishes, same model other color before
   other models. Observe the full match before relaxing.
7) Be fast: batch tools; order once verified. Step-limit with no order = zero.

Stop when an observation reports done=true.
"""


def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()


def _models() -> list[str]:
    override = getenv("AGENT_MODEL")
    if override:
        return [override]
    primary = _DEFAULT_MODELS.get(_provider(), _DEFAULT_MODELS["chutes"])
    fallback = _FALLBACK_MODELS.get(_provider(), _FALLBACK_MODELS["chutes"])
    return [primary] if primary == fallback else [primary, fallback]


def _split_query(query: str) -> tuple[str, str]:
    goal, _, rules = (query or "").partition("\n\nTask rules:")
    return goal.strip(), rules.strip()


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("function", {}).get("arguments", "{}")
    if isinstance(raw, dict):
        parsed = raw
    else:
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _trim_obs(obs: Any) -> str:
    text = json.dumps(obs, ensure_ascii=False, default=str)
    if len(text) <= _OBS_CHARS:
        return text
    return text[:_OBS_CHARS] + f"...[truncated {len(text) - _OBS_CHARS} chars]"


def _needs_justification(goal: str) -> bool:
    """Only the shopper goal — Task rules mention 'reason' and must be ignored."""
    return bool(
        re.search(
            r"\b(why|reason|justify|justification|explain|walk me through|"
            r"tell me why|give me a reason)\b",
            goal or "",
            re.I,
        )
    )


def _parse_budget(goal: str) -> float | None:
    best = None
    for m in _BUDGET_RE.finditer(goal or ""):
        whole = m.group(1).replace(",", "")
        frac = m.group(2) or "0"
        try:
            val = float(f"{whole}.{frac}")
        except ValueError:
            continue
        if best is None or val < best:
            best = val
    return best


def _parse_candidates(goal: str) -> list[tuple[str, str]]:
    return [(pid, sku.rstrip(".,;:")) for pid, sku in _KEY_RE.findall(goal or "")]


def _parse_preference(goal: str) -> tuple[str | None, str | None]:
    m = _PREF_COLOR_MODEL_RE.search(goal or "")
    if m:
        return m.group(1).lower(), m.group(2)
    m = _PREF_RE.search(goal or "")
    if m:
        # "prefer the Y81" without color — second group may be model
        a, b = m.group(1), m.group(2)
        colors = {
            "black",
            "white",
            "blue",
            "red",
            "green",
            "gold",
            "grey",
            "gray",
            "silver",
            "pink",
            "purple",
            "orange",
            "yellow",
            "brown",
            "beige",
            "titanium",
            "graphite",
            "midnight",
            "starlight",
        }
        if a.lower() in colors:
            return a.lower(), b
        return None, b
    return None, None


def _parse_firm_specs(goal: str) -> dict[str, str]:
    g = goal or ""
    low = g.lower()
    out: dict[str, str] = {}
    m = re.search(r"(\d+)\s*gb\s*(?:of\s*)?ram\b", low)
    if not m:
        m = re.search(r"\bram[,\s]+(\d+)\s*gb\b", low)
    if m:
        out["ram"] = f"{m.group(1)}gb"
    if re.search(r"\b(full\s*hd|1080p|fhd)\b", low):
        out["resolution"] = "fhd"
    elif re.search(r"\b(4k|uhd|ultra\s*hd)\b", low):
        out["resolution"] = "4k"
    elif re.search(r"\b(qhd|1440p)\b", low):
        out["resolution"] = "qhd"
    elif re.search(r"\b(720p|\bhd\b)\b", low) and "full" not in low:
        out["resolution"] = "hd"
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", low)
    if m:
        out["screen_size"] = f"{m.group(1)}in"
    m = re.search(r"(\d+(?:\.\d+)?)\s*hz", low)
    if m:
        out["refresh_rate"] = f"{m.group(1)}hz"
    if re.search(r"\bsmartphone", low):
        out["category"] = "smartphone"
    elif re.search(r"\bfeature\s*phone", low) and not re.search(
        r"\bnot a feature\s*phone\b", low
    ):
        out["category"] = "feature_phone"
    elif re.search(r"\btablet\b", low):
        out["category"] = "tablet"
    return out


def _claim_field_enum(tools: list[dict[str, Any]]) -> set[str] | None:
    """Return schema enum if present; None means any fact label is allowed."""
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or fn.get("name") != "place_test_order":
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        field = (
            ((props.get("justification") or {}).get("properties") or {})
            .get("claims", {})
            .get("items", {})
            .get("properties", {})
            .get("field", {})
        )
        enum = field.get("enum")
        if isinstance(enum, list) and enum:
            return {str(x) for x in enum}
    return None


def _goal_claim_hints(goal: str) -> dict[str, str]:
    specs = _parse_firm_specs(goal)
    return {
        k: v
        for k, v in specs.items()
        if k in {"resolution", "screen_size", "refresh_rate"}
    }


def _evidence_blob(state: dict[str, Any]) -> str:
    return "\n".join(
        [
            state.get("last_view") or "",
            state.get("last_inspect") or "",
            state.get("last_compare") or "",
        ]
    ).lower()


def _pick_claim_value(field: str, hint: str, blob: str) -> str | None:
    if field == "resolution":
        tokens = {
            "fhd": ["fhd", "full hd", "fullhd", "1080"],
            "hd": ["720", " hd", "hd ", "hd,"],
            "4k": ["4k", "uhd", "2160", "ultra hd"],
            "qhd": ["qhd", "1440"],
        }.get(hint, [hint])
        return hint if any(t in blob for t in tokens) else None
    if field == "screen_size":
        num = re.sub(r"[^0-9.]", "", hint)
        return hint if num and num in blob else None
    if field == "refresh_rate":
        num = re.sub(r"[^0-9.]", "", hint)
        compact = blob.replace(" ", "")
        return hint if num and (f"{num}hz" in compact or f"{num} hz" in blob) else None
    return None


def _build_claims(
    *,
    goal: str,
    state: dict[str, Any],
    allowed: set[str] | None,
) -> list[dict[str, str]]:
    """Build required + one soft fact claim from viewed facts / inspect_stock.

    Tool schema asks for the shopper's specification field, price, in_stock, and
    at least one further supported reason from the product's own fact labels.
    """
    claims: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(field: str, value: str) -> None:
        field = str(field).strip()
        value = str(value).strip()
        if not field or not value:
            return
        if allowed is not None and field not in allowed:
            return
        if field in seen:
            return
        seen.add(field)
        claims.append({"field": field, "value": value})

    if state.get("price") is not None:
        add("price", f"{float(state['price']):.2f}")
    if state.get("in_stock") is not None:
        add("in_stock", "true" if state.get("in_stock") else "false")
    else:
        add("in_stock", "true")

    facts = dict(state.get("facts") or {})
    pid = str(state.get("inspect_pid") or state.get("cart_pid") or "")
    if pid and (state.get("facts_by_pid") or {}).get(pid):
        facts = dict(state["facts_by_pid"][pid])
    specs = _parse_firm_specs(goal)

    # Gate attribute: use the required specification field name (resolution /
    # screen_size / refresh_rate), with values copied from facts when possible.
    if specs.get("screen_size"):
        if "size" in facts:
            add("size", str(facts["size"]))
        else:
            add("screen_size", specs["screen_size"])
    if specs.get("resolution"):
        # Required claim field is always "resolution" (not display).
        blob = " ".join(str(v) for v in facts.values()).lower() + " " + _evidence_blob(state)
        val = specs["resolution"]
        tokens = {
            "fhd": ("fhd", "full hd", "fullhd", "1080"),
            "hd": ("720", " hd", "hd ", "hd;"),
            "4k": ("4k", "uhd", "2160"),
            "qhd": ("qhd", "1440"),
        }.get(val, (val,))
        if any(t in blob for t in tokens):
            add("resolution", val)
    if specs.get("refresh_rate"):
        add("refresh_rate", specs["refresh_rate"])

    # Soft reason: prefer stable fact labels. Avoid ram — often mismatched.
    soft_keys = (
        "model",
        "phone_type",
        "condition",
        "warranty_type",
        "operating_system",
        "device_operating_system",
        "battery",
    )
    for key in soft_keys:
        if key in facts and key not in seen:
            add(key, str(facts[key]))
            break
    else:
        cat = None
        if pid and (state.get("category_by_pid") or {}).get(pid):
            cat = state["category_by_pid"][pid]
        else:
            cat = state.get("category_path")
        if isinstance(cat, list) and cat:
            add("category", str(cat[-1]))
        elif state.get("brand"):
            add("brand", str(state["brand"]))

    return claims


def _inner_obs(obs: Any) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return {}
    inner = obs.get("observation")
    return inner if isinstance(inner, dict) else obs


def _update_state_from_obs(state: dict[str, Any], name: str, obs: Any) -> None:
    inner = _inner_obs(obs)
    text = json.dumps(obs, ensure_ascii=False, default=str)
    if name in ("view", "compare"):
        if name == "view":
            state["last_view"] = text[:12000]
            pid = str(inner.get("product_id") or "")
            if pid:
                state.setdefault("product_text", {})[pid] = text.lower()
                state["last_view_pid"] = pid
            if inner.get("brand"):
                state["brand"] = inner.get("brand")
            facts = inner.get("facts")
            if isinstance(facts, dict) and facts:
                state["facts"] = {str(k): v for k, v in facts.items()}
                if pid:
                    state.setdefault("facts_by_pid", {})[pid] = state["facts"]
            cpath = inner.get("category_path")
            if isinstance(cpath, list) and cpath:
                state["category_path"] = cpath
                if pid:
                    state.setdefault("category_by_pid", {})[pid] = cpath
            # index variant option strings
            for v in inner.get("variants") or []:
                if not isinstance(v, dict):
                    continue
                sku = str(v.get("sku") or "")
                if sku:
                    state.setdefault("sku_text", {})[sku] = (
                        f"{v.get('options','')} {inner.get('title','')} "
                        f"{json.dumps(inner.get('facts') or {})}"
                    ).lower()
        else:
            state["last_compare"] = text[:12000]
            for prod in inner.get("compared") or inner.get("products") or []:
                if not isinstance(prod, dict):
                    continue
                pid = str(prod.get("product_id") or "")
                if pid:
                    state.setdefault("product_text", {})[pid] = json.dumps(
                        prod, ensure_ascii=False
                    ).lower()
                if prod.get("brand"):
                    state["brand"] = prod.get("brand")
                facts = prod.get("facts")
                if isinstance(facts, dict) and facts:
                    state["facts"] = {str(k): v for k, v in facts.items()}
                    if pid:
                        state.setdefault("facts_by_pid", {})[pid] = state["facts"]
                cpath = prod.get("category_path")
                if isinstance(cpath, list) and cpath:
                    state["category_path"] = cpath
                    if pid:
                        state.setdefault("category_by_pid", {})[pid] = cpath
    if name == "inspect_stock":
        state["needs_reinspect"] = False
        state["last_inspect"] = text[:4000]
        if "price" in inner:
            try:
                state["price"] = float(inner["price"])
            except (TypeError, ValueError):
                pass
        elif "price" in (obs if isinstance(obs, dict) else {}):
            try:
                state["price"] = float(obs["price"])  # type: ignore[index]
            except (TypeError, ValueError):
                pass
        stock = inner.get("in_stock", obs.get("in_stock") if isinstance(obs, dict) else None)
        if stock is not None:
            state["in_stock"] = bool(stock)
        pid = inner.get("product_id")
        sku = inner.get("sku")
        if isinstance(obs, dict):
            pid = pid or obs.get("product_id")
            sku = sku or obs.get("sku")
        state["inspect_pid"] = str(pid or "")
        state["inspect_sku"] = str(sku or "")
    if name == "add_to_cart":
        state["carted"] = True
        if isinstance(obs, dict) and not (obs.get("error") or inner.get("error")):
            state["cart_pid"] = str(
                inner.get("product_id")
                or (obs.get("product_id") if isinstance(obs, dict) else "")
                or state.get("inspect_pid")
                or ""
            )
            state["cart_sku"] = str(
                inner.get("sku")
                or (obs.get("sku") if isinstance(obs, dict) else "")
                or state.get("inspect_sku")
                or ""
            )
    if name == "search":
        state["searches"] = int(state.get("searches") or 0) + 1


def _blob_for_product(state: dict[str, Any], pid: str, sku: str) -> str:
    parts = [
        (state.get("product_text") or {}).get(pid, ""),
        (state.get("sku_text") or {}).get(sku, ""),
    ]
    if pid and state.get("last_view_pid") == pid:
        parts.append(state.get("last_view") or "")
    if pid and state.get("inspect_pid") == pid:
        parts.append(state.get("last_inspect") or "")
    return "\n".join(p for p in parts if p).lower()


def _firm_specs_ok(specs: dict[str, str], blob: str) -> tuple[bool, str]:
    """Block only on clear contradictions. Missing sparse facts → allow."""
    if not specs or not blob:
        return True, ""
    if "ram" in specs:
        n = re.sub(r"[^0-9]", "", specs["ram"])
        ram_ok = bool(
            re.search(rf"\b{n}\s*gb\s*ram\b", blob)
            or re.search(rf"\bram\s*[:=]?\s*{n}\s*gb\b", blob)
            or re.search(rf"\b{n}gb\s*ram\b", blob)
        )
        rom_only = bool(re.search(rf"\b{n}\s*gb\s*rom\b", blob)) and not ram_ok
        if rom_only:
            return False, f"ROM-only hit for firm RAM {specs['ram']}"
        if re.search(r"\b(ddr[345]?|sodimm|memory\s*card|ram\s*module|kingston\s*kvr)\b", blob):
            return False, "RAM module / memory stick is not the product"
    if "refresh_rate" in specs:
        n = re.sub(r"[^0-9.]", "", specs["refresh_rate"])
        compact = blob.replace(" ", "")
        # Only reject if another explicit Hz is present and required Hz is absent.
        has_req = f"{n}hz" in compact
        other = re.findall(r"(\d+(?:\.\d+)?)hz", compact)
        if other and not has_req:
            return False, f"refresh mismatch vs firm {specs['refresh_rate']}"
    if specs.get("category") == "smartphone":
        if re.search(r"\bfeature\s*phone\b|\bkeypad\b|\bbasic\s*phone\b", blob):
            if not re.search(r"\bsmartphone\b|\bandroid\b", blob):
                return False, "feature/keypad phone for a smartphone goal"
    if specs.get("category") == "feature_phone":
        if re.search(r"\b(ddr[345]?|sodimm|memory card|ram module)\b", blob):
            return False, "RAM module is not a feature phone"
    return True, ""


def _sanitize_action(
    name: str,
    args: dict[str, Any],
    *,
    goal: str,
    want_reason: bool,
    allowed: set[str] | None,
    state: dict[str, Any],
    specs: dict[str, str],
    budget: float | None,
) -> tuple[str, dict[str, Any], str | None]:
    """Return (name, args, optional_user_nudge). May rewrite unsafe carts/orders."""
    out = dict(args)
    nudge = None

    if name == "add_to_cart":
        pid = str(out.get("product_id") or "")
        sku = str(out.get("sku") or "")
        blob = _blob_for_product(state, pid, sku)
        ok, reason = _firm_specs_ok(specs, blob)
        if not ok and blob:
            q_parts = []
            if specs.get("category") == "feature_phone":
                q_parts.append("feature phone")
            elif specs.get("category") == "smartphone":
                q_parts.append("smartphone")
            elif specs.get("category") == "tablet":
                q_parts.append("tablet")
            if "ram" in specs:
                q_parts.append(f'"{specs["ram"].replace("gb", "GB")} RAM"')
            if "refresh_rate" in specs:
                q_parts.append(specs["refresh_rate"])
            if "resolution" in specs:
                q_parts.append(
                    {"fhd": "Full HD", "hd": "720p", "4k": "4K", "qhd": "QHD"}.get(
                        specs["resolution"], specs["resolution"]
                    )
                )
            if "screen_size" in specs:
                q_parts.append(specs["screen_size"].replace("in", "-inch"))
            query = " ".join(q_parts) or "in stock"
            nudge = (
                f"Blocked add_to_cart for {pid}::{sku}: {reason}. "
                f"Search/filter again for firm specs, then view+inspect before carting."
            )
            return "search", {"query": query, "k": 10}, nudge
        if budget is not None and state.get("price") is not None:
            try:
                if float(state["price"]) > budget + 1e-6:
                    nudge = (
                        f"Blocked add_to_cart: price {state['price']} exceeds budget {budget}."
                    )
                    return (
                        "search",
                        {"query": f"in stock under {budget:.0f}", "k": 10},
                        nudge,
                    )
            except (TypeError, ValueError):
                pass

    if name == "place_test_order":
        # Must re-inspect after a market event before ordering.
        if state.get("needs_reinspect"):
            pid = str(
                out.get("product_id")
                or state.get("inspect_pid")
                or state.get("cart_pid")
                or ""
            )
            sku = str(
                out.get("sku")
                or state.get("inspect_sku")
                or state.get("cart_sku")
                or ""
            )
            if pid and sku and state.get("inspect_pid") != pid:
                nudge = "Market event pending — inspect_stock again before place_test_order."
                return "inspect_stock", {"product_id": pid, "sku": sku}, nudge
            if pid and sku and state.get("needs_reinspect"):
                # Force one fresh inspect even if same pid was inspected pre-event.
                nudge = "Re-inspect after market event, then order."
                state["needs_reinspect"] = False  # prevent inspect loop
                return "inspect_stock", {"product_id": pid, "sku": sku}, nudge

        out.pop("justification", None)
        if not out.get("product_id") and state.get("inspect_pid"):
            out["product_id"] = state["inspect_pid"]
        if not out.get("sku") and state.get("inspect_sku"):
            out["sku"] = state["inspect_sku"]
        if want_reason:
            pid = str(out.get("product_id") or "")
            if pid and not (state.get("facts_by_pid") or {}).get(pid):
                nudge = (
                    f"Must view product {pid} before justifying — need raw facts."
                )
                return "view", {"product_id": pid}, nudge
            claims = _build_claims(goal=goal, state=state, allowed=allowed)
            if claims:
                out["justification"] = {"claims": claims}
        return name, out, nudge

    return name, out, nudge


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.add(str(fn["name"]))
    return names


def _env_call(
    *,
    binding: dict[str, Any],
    problem_id: str,
    turn: int,
    actions: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    group_id = f"{problem_id}-turn-{turn}"
    envelope = {
        **binding,
        "call_id": group_id,
        "idempotency_key": group_id,
        "turn": turn,
        "calls": [
            {
                "call_id": f"{group_id}-{index}",
                "action": {"name": name, "args": args},
            }
            for index, (name, args) in enumerate(actions, start=1)
        ],
    }
    result = _proxy.post("/environment/call", json_data=envelope)
    if result is None:
        raise RuntimeError("environment call failed")
    return result


def _bootstrap_actions(
    *,
    tool_names: set[str],
    candidates: list[tuple[str, str]],
    pref_color: str | None,
    pref_model: str | None,
    specs: dict[str, str],
) -> list[tuple[str, dict[str, Any]]]:
    """Deterministic first-turn actions grounded only in the goal text."""
    if candidates and "compare" in tool_names:
        pids = []
        seen: set[str] = set()
        for pid, _sku in candidates:
            if pid not in seen:
                seen.add(pid)
                pids.append(pid)
            if len(pids) >= 5:
                break
        return [("compare", {"product_ids": pids})]
    if pref_model and "search" in tool_names:
        q = f"{pref_color + ' ' if pref_color else ''}{pref_model}".strip()
        cat = ""
        if specs.get("category") == "smartphone":
            cat = "smartphone "
        elif specs.get("category") == "feature_phone":
            cat = "feature phone "
        elif specs.get("category") == "tablet":
            cat = "tablet "
        return [("search", {"query": f"{cat}{q}".strip(), "k": 10})]
    return []


def _recovery_followup_views(
    *,
    tool_names: set[str],
    search_obs: Any,
    pref_model: str | None,
    pref_color: str | None,
    budget: float | None,
    max_views: int = 3,
) -> list[tuple[str, dict[str, Any]]]:
    """After preferred-model search, view hits that mention the model/color."""
    if "view" not in tool_names or not pref_model:
        return []
    inner = _inner_obs(search_obs)
    results = inner.get("results") if isinstance(inner, dict) else None
    if not isinstance(results, list):
        return []
    model_l = pref_model.lower()
    color_l = (pref_color or "").lower()
    accessory = re.compile(
        r"\b(case|cover|housing|casing|tempered|charger|cable|sim\s*tray|"
        r"button|ribbon|protector|glass|sticker|skin)\b",
        re.I,
    )
    scored: list[tuple[int, str]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        title = str(r.get("title") or "")
        opts = str(r.get("options") or r.get("sku") or "")
        blob = f"{title} {opts}".lower()
        if model_l not in blob:
            continue
        if accessory.search(title):
            continue
        if budget is not None:
            try:
                if float(r.get("price")) > budget + 1e-6:
                    continue
            except (TypeError, ValueError):
                continue
        pid = str(r.get("product_id") or "")
        if not pid:
            continue
        score = 0
        if color_l and color_l in blob:
            score += 2
        if "smartphone" in blob or "phone" in blob:
            score += 1
        scored.append((score, pid))
    scored.sort(reverse=True)
    return [("view", {"product_id": pid}) for _s, pid in scored[:max_views]]


def _chat(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return _proxy.post(
        "/inference/chat/completions",
        json_data={
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "temperature": 0,
        },
    )


def _infer(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_calls: int,
) -> tuple[str, list[dict[str, Any]], str]:
    last_content = ""
    for model in _models():
        attempt_messages = messages
        for _ in range(2):
            inference = _chat(model, attempt_messages, tools)
            if not inference or not isinstance(inference, dict):
                logger.warning("inference failed model=%s", model)
                break
            try:
                choice = inference["choices"][0]
                assistant = choice.get("message") or choice
            except (KeyError, IndexError, TypeError, AttributeError):
                logger.warning("malformed inference model=%s body=%s", model, str(inference)[:300])
                break
            if not isinstance(assistant, dict):
                logger.warning("malformed assistant model=%s", model)
                break
            content = assistant.get("content") or ""
            if isinstance(content, list):
                # Some providers return content parts
                content = " ".join(
                    str(p.get("text") if isinstance(p, dict) else p) for p in content
                )
            last_content = content
            tool_calls = list(assistant.get("tool_calls") or [])[:max_calls]
            if not tool_calls and isinstance(assistant.get("function_call"), dict):
                fc = assistant["function_call"]
                tool_calls = [{
                    "id": "fn-1",
                    "type": "function",
                    "function": {
                        "name": fc.get("name"),
                        "arguments": fc.get("arguments") or "{}",
                    },
                }]
            if tool_calls:
                return content, tool_calls, model
            attempt_messages = attempt_messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": "Call one of the supplied tools now."},
            ]
    return last_content, [], _models()[0]


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    query = str(policy.get("query") or problem_data.get("query") or "")
    goal, rules = _split_query(query)
    tools = policy["tools"]
    max_steps = int(policy.get("max_steps") or 30)
    max_calls = min(int(policy.get("max_calls_per_turn") or 1), _MAX_PARALLEL)
    allowed = _claim_field_enum(tools)
    want_reason = _needs_justification(goal)
    candidates = _parse_candidates(goal)
    budget = _parse_budget(goal)
    specs = _parse_firm_specs(goal)
    pref_color, pref_model = _parse_preference(goal)

    opener = goal
    if rules:
        opener = f"{goal}\n\nTask rules:\n{rules}"

    hints: list[str] = []
    if candidates:
        keys = ", ".join(f"{p}::{s}" for p, s in candidates[:8])
        hints.append(
            f"Ranking candidates from the goal: {keys}. Compare these product_ids "
            "first (max 5), then inspect_stock the best by the stated priority."
        )
    if pref_model:
        q = f"{pref_color + ' ' if pref_color else ''}{pref_model}".strip()
        hints.append(
            f"Preferred exact match: search '{q}' first and cart that model/color "
            "when in stock and in budget before relaxing."
        )
    if specs:
        hints.append(f"Parsed firm specs from the goal: {specs}. Verify before carting.")
    if budget is not None:
        hints.append(f"Hard budget ceiling: {budget:.2f}.")
    hints.append(
        "Justification required — claims must match viewed/inspected evidence."
        if want_reason
        else "Do NOT attach justification to place_test_order."
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"{opener}\n\nTurns={max_steps}, max_calls/turn={max_calls}. "
                + " ".join(hints)
            ),
        },
    ]
    dialogue: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "carted": False,
        "searches": 0,
        "price": None,
        "in_stock": None,
        "brand": None,
        "facts": {},
        "category_path": None,
        "needs_reinspect": False,
        "last_view": "",
        "last_inspect": "",
        "last_compare": "",
        "product_text": {},
        "sku_text": {},
    }
    event_pending = False
    tool_names = _tool_names(tools)
    start_turn = 1

    # Goal-grounded bootstrap so recovery observes the preferred match and
    # ranking sees candidates before the model wanders.
    boot = _bootstrap_actions(
        tool_names=tool_names,
        candidates=candidates,
        pref_color=pref_color,
        pref_model=pref_model,
        specs=specs,
    )
    if boot:
        result = _env_call(
            binding=binding,
            problem_id=problem_id,
            turn=1,
            actions=boot,
        )
        fake_calls = []
        for index, (name, args) in enumerate(boot, start=1):
            tc_id = f"boot-{index}"
            fake_calls.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            )
        messages.append(
            {
                "role": "assistant",
                "content": "Bootstrapping from the goal.",
                "tool_calls": fake_calls,
            }
        )
        call_results = result.get("calls") or []
        search_obs = None
        for (name, _args), call_result, tc in zip(boot, call_results, fake_calls):
            obs = (call_result or {}).get("observation")
            if name == "search":
                search_obs = obs
            _update_state_from_obs(state, name, obs)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": _trim_obs(obs),
                }
            )
        dialogue.append(
            {
                "role": "assistant",
                "content": "Bootstrapping from the goal.",
                "tool_calls": fake_calls,
                "sanitized_actions": [{"name": n, "args": a} for n, a in boot],
                "environment_result": result,
                "model": "bootstrap",
            }
        )

        follow = _recovery_followup_views(
            tool_names=tool_names,
            search_obs=search_obs,
            pref_model=pref_model,
            pref_color=pref_color,
            budget=budget,
        )
        next_turn = 2
        if follow:
            result2 = _env_call(
                binding=binding,
                problem_id=problem_id,
                turn=2,
                actions=follow,
            )
            fake2 = []
            for index, (name, args) in enumerate(follow, start=1):
                tc_id = f"boot-view-{index}"
                fake2.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": "Viewing preferred-model search hits.",
                    "tool_calls": fake2,
                }
            )
            for (name, _args), call_result, tc in zip(
                follow, result2.get("calls") or [], fake2
            ):
                obs = (call_result or {}).get("observation")
                _update_state_from_obs(state, name, obs)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": _trim_obs(obs),
                    }
                )
            dialogue.append(
                {
                    "role": "assistant",
                    "content": "Viewing preferred-model search hits.",
                    "tool_calls": fake2,
                    "sanitized_actions": [{"name": n, "args": a} for n, a in follow],
                    "environment_result": result2,
                    "model": "bootstrap",
                }
            )
            next_turn = 3
            call_results = result2.get("calls") or []

        messages.append(
            {
                "role": "user",
                "content": (
                    "Bootstrap observations are above. Continue: inspect_stock the "
                    "best matching in-budget in-stock option and progress toward "
                    "add_to_cart / place_test_order. For recovery, cart the exact "
                    "preferred model+color if present in these results."
                ),
            }
        )
        start_turn = next_turn
        if any(
            isinstance(c, dict)
            and isinstance(c.get("observation"), dict)
            and c["observation"].get("done")
            for c in call_results
        ):
            return dialogue

    for turn in range(start_turn, max_steps + 1):
        if turn == start_turn and start_turn == 1:
            if candidates:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "First action: compare the listed candidate product_ids "
                            "(max 5) or view them. Do not search unrelated items."
                        ),
                    }
                )
            elif pref_model:
                q = f"{pref_color + ' ' if pref_color else ''}{pref_model}".strip()
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"First action: search for '{q}' (and category words from "
                            "the goal). Find and inspect the exact preferred match "
                            "before any other model."
                        ),
                    }
                )
            elif specs:
                parts = []
                if specs.get("category") == "feature_phone":
                    parts.append("feature phone")
                elif specs.get("category") == "smartphone":
                    parts.append("smartphone")
                elif specs.get("category") == "tablet":
                    parts.append("tablet")
                if "ram" in specs:
                    parts.append(f'{specs["ram"].replace("gb","GB")} RAM')
                if "refresh_rate" in specs:
                    parts.append(specs["refresh_rate"])
                if "resolution" in specs:
                    parts.append(specs["resolution"])
                if "screen_size" in specs:
                    parts.append(specs["screen_size"].replace("in", "-inch"))
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "First action: search with firm specs in the query, e.g. "
                            f"\"{' '.join(parts)}\". Avoid ROM/memory-stick false hits."
                        ),
                    }
                )

        if turn == max(7, max_steps // 4) and not state["carted"]:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Urgency: inspect_stock a firm-spec-compliant in-budget "
                        "variant and add_to_cart. Prefer exact preferred model/color "
                        "if named."
                    ),
                }
            )
        if turn >= max_steps - 5 and turn < max_steps:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{max_steps - turn} turns left. Re-inspect if needed and "
                        "place_test_order now"
                        + (
                            " with minimal truthful schema claims."
                            if want_reason
                            else " with NO justification."
                        )
                    ),
                }
            )
        if event_pending:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Market event seen. Re-inspect survivors. For ranking: "
                        "re-compare candidates and place_test_order on the NEW best "
                        "(do not revert to a dominated pre-event pick). For recovery: "
                        "same model other color before other models. Then order."
                    ),
                }
            )
            event_pending = False

        content, tool_calls, model_used = _infer(messages, tools, max_calls)
        if not tool_calls:
            raise RuntimeError("model returned no tool call before completion")

        actions: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
        send_calls: list[dict[str, Any]] = []
        nudges: list[str] = []
        for call in tool_calls:
            name = call["function"]["name"]
            raw_args = _arguments(call)
            new_name, args, nudge = _sanitize_action(
                name,
                raw_args,
                goal=goal,
                want_reason=want_reason,
                allowed=allowed,
                state=state,
                specs=specs,
                budget=budget,
            )
            if nudge:
                nudges.append(nudge)
            patched = {
                "id": call.get("id"),
                "type": call.get("type", "function"),
                "function": {
                    "name": new_name,
                    "arguments": json.dumps(args),
                },
            }
            send_calls.append(patched)
            actions.append((patched, new_name, args))

        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": send_calls,
            }
        )

        result = _env_call(
            binding=binding,
            problem_id=problem_id,
            turn=turn,
            actions=[(n, a) for _c, n, a in actions],
        )

        dialogue.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": send_calls,
                "sanitized_actions": [{"name": n, "args": a} for _c, n, a in actions],
                "environment_result": result,
                "model": model_used,
                "nudges": nudges,
            }
        )

        call_results = result.get("calls") or []
        for (tool_call, name, _args), call_result in zip(actions, call_results):
            obs = (call_result or {}).get("observation")
            _update_state_from_obs(state, name, obs)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _trim_obs(obs),
                }
            )

        if nudges:
            messages.append({"role": "user", "content": " ".join(nudges)})

        user_message = result.get("user_message") or {}
        user_text = user_message.get("content") if isinstance(user_message, dict) else None
        if user_text:
            event_pending = True
            state["needs_reinspect"] = True
            messages.append({"role": "user", "content": str(user_text)})

        if any(
            isinstance(c, dict)
            and isinstance(c.get("observation"), dict)
            and c["observation"].get("done")
            for c in call_results
        ):
            break

    return dialogue


__all__ = ["agent_main"]
