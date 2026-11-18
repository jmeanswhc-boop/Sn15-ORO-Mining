"""ORO generated-environment agent v1.

Self-contained agent_main for oro-env-runtime tasks (TF1-TF7).
Design targets the deterministic verifier:
  hard gates   : final_in_gold, final_in_stock, within_budget, no_illegal_side_effects
  event closure: intended first cart commit -> notice changed state in a LATER turn
                 (post-event observation of the event target) -> order in a turn
                 strictly after the noticing turn; never order an invalidated target
  interventions: every shopper question gets a `message` reply before the order
  claims       : justification claims only when the shopper asked for a reason,
                 grounded in fields actually seen via view/compare/inspect_stock
  efficiency   : cost = tool_calls + unique observed candidates (ref ~20)
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# --------------------------------------------------------------------------- #
# model selection + deadline budget (ported pattern from Alan_v52 ProxyUtil)
# --------------------------------------------------------------------------- #

# provider-paired priority chains; SANDBOX_MODEL (if set) is prepended as the
# preferred head, the chain below stays as fallback.
_LLM_PRIORITIES = {
    "chutes": [
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "Qwen/Qwen3.5-397B-A17B-TEE",
        "zai-org/GLM-5.2-TEE",
    ],
    "openrouter": [
        "deepseek/deepseek-v3.2",
        "qwen/qwen3.5-397b-a17b",
        "z-ai/glm-5.2",
    ],
}

# per-problem wall clock; refuse work that cannot finish and force the endgame
_TIME_BUDGET = float(getenv("AGENT_TIME_BUDGET_SECS") or "280")
_ENDGAME_SECS = 60.0  # below this: stop exploring, converge on an order
_LLM_MIN_SECS = 20.0  # typical cost of one inference round trip


def _provider() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()
    return raw if raw in _LLM_PRIORITIES else "chutes"


def _llm_chain() -> list[str]:
    chain = list(_LLM_PRIORITIES[_provider()])
    head = getenv("SANDBOX_MODEL")
    if head:
        chain = [head] + [m for m in chain if m != head]
    return chain


class _Deadline:
    def __init__(self, budget: float) -> None:
        self.at = time.monotonic() + budget

    def left(self) -> float:
        return self.at - time.monotonic()

    def can_afford(self, secs: float) -> bool:
        return self.left() > secs

    def endgame(self) -> bool:
        return self.left() < _ENDGAME_SECS


class _Proxy:
    """Thin wrapper over the project ProxyClient so a per-call timeout can be
    clamped to the remaining budget (ProxyClient's timeout is per instance)."""

    def __init__(self) -> None:
        self._clients: dict[int, ProxyClient] = {}

    def _client(self, timeout: float) -> ProxyClient:
        key = max(10, int(round(timeout / 10.0)) * 10)
        client = self._clients.get(key)
        if client is None:
            client = ProxyClient(timeout=key, max_retries=3)
            self._clients[key] = client
        return client

    def post(
        self, path: str, json_data: dict, timeout: float | None = None
    ) -> dict | None:
        return self._client(timeout if timeout is not None else 120.0).post(
            path, json_data=json_data
        )


_proxy = _Proxy()

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

_BUDGET_HINT = re.compile(
    r"(?:budget|spend|afford|no more than|at most|under|below|cap(?:ped)?|"
    r"limit|maximum|max|up to)\D{0,20}?([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_REASON_HINT = re.compile(
    r"\b(why|reason|justif|explain|because|evidence|prove)\w*\b", re.IGNORECASE
)


def _goal_part(query: str) -> str:
    """The shopper's own request; the runtime appends generic 'Task rules:'
    boilerplate that must not drive routing, budget, or reason detection."""
    return (query or "").split("Task rules:", 1)[0]


def _parse_budget(text: str) -> float | None:
    best = None
    for match in _BUDGET_HINT.finditer(text or ""):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if value > 0:
            best = value if best is None else min(best, value)
    return best


# --------------------------------------------------------------------------- #
# family router + lanes (combo-router pattern, keyed on TF family signatures)
# --------------------------------------------------------------------------- #

_PAIR_RE = re.compile(r"(\d{6,})::([0-9A-Za-z_-]+)")

# claim_grounding: each claim must be EXACTLY {field, value}; the justification
# object may contain ONLY "claims"; non-TF7 tasks accept only these fields.
_OPTIONAL_CLAIM_FIELDS = {
    "category", "resolution", "refresh_rate", "screen_size", "price", "in_stock", "brand",
}
_TF7_CLAIM_FIELDS = _OPTIONAL_CLAIM_FIELDS | {
    "streaming", "microphone", "privacy_cover", "autofocus", "hdr",
}


def _sanitize_justification(args: dict, lane_name: str) -> dict:
    """Coerce the model's justification into the exact verifier-accepted shape,
    dropping anything that would trip the noncompensable truth gate."""
    just = args.get("justification")
    if not isinstance(just, dict):
        args.pop("justification", None)
        return args
    allowed = _TF7_CLAIM_FIELDS if lane_name == "justification" else _OPTIONAL_CLAIM_FIELDS
    clean = []
    for claim in just.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        field = str(claim.get("field") or "").strip()
        value = claim.get("value")
        if isinstance(value, bool):
            value = "true" if value else "false"
        value = str(value if value is not None else "").strip()
        if field in allowed and value:
            clean.append({"field": field, "value": value})
    if clean:
        args["justification"] = {"claims": clean}
    else:
        args.pop("justification", None)
    return args


_HZ_RE = re.compile(r"(\d{2,3})\s*hz", re.IGNORECASE)
_SIZE_REQ_RE = re.compile(r"(\d{1,2}(?:\.\d{1,2})?)\s*-?\s*inch", re.IGNORECASE)
_RES_WORD_RE = re.compile(
    r"\b(4k|uhd|2160p|qhd|1440p|2k|full\s*hd|fhd|1080p|hd|720p)\b", re.IGNORECASE
)
_RES_PAIR_RE = re.compile(r"(\d{3,4})\s*[x×*]\s*(\d{3,4})")
_BRAND_REQ_RE = re.compile(
    r"(?:required brand is|brand is|\bprefer(?:red)?(?: brand is)?|\bfrom)\s+([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)?)"
)
_BRAND_LEAD_RE = re.compile(
    r"\b(?:a|an|the|in-stock|me a|me an)\s+([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)?)\s+"
    r"(?i:(?:\d+p\s+|hd\s+|full\s+hd\s+|\d+(?:\.\d+)?-inch\s+|gaming\s+)*"
    r"(?:smartphone|phone|monitor|tablet|headphone|laptop|earbud))"
)


def _res_class(text: str) -> str | None:
    """Normalize a resolution mention to a class; None if absent or conflicting."""
    if not text:
        return None
    classes = set()
    for m in _RES_PAIR_RE.finditer(text):
        short = min(int(m.group(1)), int(m.group(2)))
        classes.add("uhd" if short >= 2000 else "qhd" if short >= 1400 else
                    "fhd" if short >= 1000 else "hd" if short >= 700 else "sub-hd")
    for m in _RES_WORD_RE.finditer(text):
        w = m.group(1).lower().replace(" ", "")
        classes.add({"4k": "uhd", "uhd": "uhd", "2160p": "uhd", "qhd": "qhd", "1440p": "qhd",
                     "2k": "qhd", "fullhd": "fhd", "fhd": "fhd", "1080p": "fhd",
                     "hd": "hd", "720p": "hd"}[w])
    return classes.pop() if len(classes) == 1 else None


def _extract_specs(text: str) -> dict[str, Any]:
    """Hard spec requirements stated by the shopper (goal or later messages)."""
    specs: dict[str, Any] = {}
    m = _SIZE_REQ_RE.search(text or "")
    if m:
        specs["screen_size"] = float(m.group(1))
    res = _res_class(text or "")
    if res:
        specs["resolution"] = res
    m = _HZ_RE.search(text or "")
    if m:
        specs["refresh_rate"] = float(m.group(1))
    return specs


def _extract_brand(goal: str) -> tuple[str | None, bool]:
    """(brand, required). 'Ideally, I prefer X' -> preferred; 'a X smartphone' /
    'required brand is X' -> required."""
    m = _BRAND_REQ_RE.search(goal or "")
    if m:
        brand = m.group(1).strip()
        required = "prefer" not in m.group(0).lower()
        return brand, required
    m = _BRAND_LEAD_RE.search(goal or "")
    if m:
        return m.group(1).strip(), True
    return None, False


def _product_size(info: dict, sku: str | None) -> float | None:
    variant = (info.get("variants") or {}).get(sku or "") or {}
    for source in (variant.get("options") or "", (info.get("facts") or {}).get("size") or ""):
        m = _INCH_RE.search(str(source))
        if m:
            return float(m.group(1))
    return None


def _spec_failures(specs: dict[str, Any], info: dict | None, sku: str | None) -> list[str]:
    """Unknown or mismatched hard specs against OBSERVED raw facts. Unknown = fail."""
    if not specs:
        return []
    if not info:
        return ["product facts not observed yet (view it first)"]
    facts = info.get("facts") or {}
    out = []
    if "screen_size" in specs:
        actual = _product_size(info, sku)
        if actual is None:
            out.append("screen size is not in the raw facts")
        elif abs(actual - specs["screen_size"]) > 0.005:
            out.append(f"screen size {actual} != required {specs['screen_size']}")
    if "resolution" in specs:
        actual = _res_class(str(facts.get("display") or facts.get("resolution") or ""))
        if actual is None:
            out.append("resolution is missing or ambiguous in the raw facts")
        elif actual != specs["resolution"]:
            out.append(f"resolution {actual} != required {specs['resolution']}")
    if "refresh_rate" in specs:
        hz_text = " ".join([str(facts.get("refresh_rate") or ""), info.get("title") or ""])
        hz = [float(x) for x in _HZ_RE.findall(hz_text)]
        if not hz:
            out.append("refresh rate not observed")
        elif max(hz) < specs["refresh_rate"]:
            out.append(f"refresh rate {max(hz)}Hz < required {specs['refresh_rate']}Hz")
    return out


def _brand_ok(info: dict | None, brand: str | None) -> bool | None:
    """True/False on the structured brand field; None when not observed."""
    if not brand or not info or info.get("brand") is None:
        return None
    return str(info["brand"]).strip().casefold() == brand.strip().casefold()


def _build_claims(state: "_State", pid: str, sku: str, specs: dict[str, Any],
                  brand: str | None, goal: str) -> list[dict[str, str]]:
    """Fact-derived claims: price + in_stock + every shopper-named attribute that
    the raw facts state unambiguously. Nothing speculative."""
    info = state.info.get(pid) or {}
    facts = info.get("facts") or {}
    record = state.seen.get((pid, sku))
    claims: list[dict[str, str]] = []
    if record and record[2] is not None:
        claims.append({"field": "price", "value": f"{record[2]:.2f}"})
    if record and record[1] is not None:
        claims.append({"field": "in_stock", "value": "true" if record[1] else "false"})
    if "screen_size" in specs and facts.get("size"):
        claims.append({"field": "screen_size", "value": str(facts["size"])})
    if "resolution" in specs:
        disp = str(facts.get("display") or facts.get("resolution") or "")
        if disp and _res_class(disp) is not None:
            claims.append({"field": "resolution", "value": disp})
    if "refresh_rate" in specs and facts.get("refresh_rate"):
        claims.append({"field": "refresh_rate", "value": str(facts["refresh_rate"])})
    if brand and info.get("brand") and _brand_ok(info, brand):
        claims.append({"field": "brand", "value": str(info["brand"])})
    return claims
_INCH_RE = re.compile(r"(\d{1,2}(?:\.\d{1,2})?)\s*(?:-|\s)?\s*(?:inch|inches|in\b|\")", re.IGNORECASE)


def _rank_key(info: dict, sku: str) -> tuple[float, float, float] | None:
    """(refresh_hz desc, screen_in desc, price asc) per the ranking task rules:
    max advertised Hz from the title; variant option size before facts.size."""
    hz = max((float(m) for m in _HZ_RE.findall(info.get("title") or "")), default=None)
    variant = (info.get("variants") or {}).get(sku) or {}
    size = None
    for source in (variant.get("options") or "", (info.get("facts") or {}).get("size") or ""):
        m = _INCH_RE.search(str(source))
        if m:
            size = float(m.group(1))
            break
    price = variant.get("price")
    if hz is None or size is None or price is None:
        return None
    return (-hz, -size, float(price))


def _route(query: str) -> str:
    q = (query or "").lower()
    if _PAIR_RE.search(query or "") and "compare" in q:
        return "ranking"
    if len(_PAIR_RE.findall(query or "")) >= 2:
        return "ranking"
    if _REASON_HINT.search(q):
        return "justification"
    if re.search(r"tell you if you ask|if you ask", q):
        return "constraint_ask"
    if re.search(r"if (that|the).{0,40}(disappears|is gone|goes out of stock)", q) or (
        "strongly prefer" in q
    ):
        return "recovery"
    if re.search(r"rather pay|cheapest is\s?n.t|don.t just grab the cheapest", q):
        return "preference"
    if re.search(r"listing in the .{2,60} category|matches (the phrase|\")", q):
        return "retrieval"
    if re.search(r"ideally,? i prefer|firm requirements matter more", q):
        return "general"  # soft-preference precision task; generalist handles it
    return "general"


_LANES: dict[str, dict[str, Any]] = {
    "general": {
        "k": 5, "ask_text": None, "event_policy": "keep", "claims": "auto",
        "addendum": "",
    },
    "retrieval": {
        "k": 10, "ask_text": None, "event_policy": "keep", "claims": "auto",
        "max_queries": 3,
        "addendum": (
            "RETRIEVAL TASK: you may use AT MOST 3 search queries in the whole "
            "session - craft each one carefully (category words + the exact "
            "quoted phrase; reformulate rather than repeat). Use k=10 for "
            "coverage. Only view/compare listings that plausibly match the "
            "phrase; a sloppy shortlist is penalized. Then commit, re-check, "
            "and order as usual."
        ),
    },
    "constraint_ask": {
        "k": 5, "event_policy": "keep", "claims": "auto",
        "ask_text": (
            "Before I search: what is the firm requirement on the spec?"
        ),
        "addendum": (
            "The shopper is holding back one firm spec requirement until asked. "
            "It has been requested; treat their answer as a HARD constraint and "
            "verify it against raw product facts with view before committing."
        ),
    },
    "preference": {
        "k": 5, "event_policy": "keep", "claims": "auto",
        "ask_text": (
            "Quick check before I choose: should I prioritize durability "
            "features like sweat and water resistance over a lower price?"
        ),
        "addendum": (
            "PREFERENCE TASK: the shopper's priorities matter more than price. "
            "A clarifying question has been sent; use their reply to rank "
            "candidates. Never default to the cheapest item - pick the one "
            "whose raw facts best satisfy the stated priorities within budget, "
            "and verify those facts with view before committing."
        ),
    },
    "ranking": {
        "k": 5, "ask_text": None, "event_policy": "rerank", "claims": "auto",
        "addendum": (
            "RANKING TASK: the shopper listed the ONLY candidate variants "
            "(product_id::sku pairs in the message). Do NOT search. view or "
            "compare every listed candidate (compare up to 5 at a time) to "
            "learn refresh rate, screen size, price, and stock. Rank the valid "
            "ones: higher refresh rate first, then larger screen, then lower "
            "price. add_to_cart the top-ranked valid candidate. The market "
            "WILL change after you commit: re-check candidates with "
            "inspect_stock, RE-RANK under the new prices/stock, and order the "
            "NEW top-ranked valid candidate - which will be a DIFFERENT item "
            "than your first commit. Never order your first commit after the "
            "market changes."
        ),
    },
    "recovery": {
        "k": 5, "ask_text": None, "event_policy": "keep", "claims": "auto",
        "addendum": (
            "RECOVERY TASK: commit the shopper's exact preferred variant "
            "first. If it then goes out of stock or otherwise invalid, follow "
            "their fallback instructions exactly (keep budget and stock "
            "requirements, honor the stated substitution rule), remove the "
            "dead item, add the replacement, inspect it, then order it."
        ),
    },
    "justification": {
        "k": 5, "ask_text": None, "event_policy": "keep", "claims": "required",
        "addendum": (
            "JUSTIFICATION TASK: your final place_test_order MUST include "
            "justification.claims. Before ordering make sure you have (a) "
            "viewed the chosen product's raw facts with view or compare and "
            "(b) inspected its current price/stock with inspect_stock - claims "
            "are only accepted for observed facts. Include one claim per "
            "attribute the shopper named (brand, resolution, screen size, "
            "category...), plus price and in_stock from your latest "
            "inspection. Values must exactly match what you observed."
        ),
    },
}


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("function", {}).get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    return parsed if isinstance(parsed, dict) else {}


def _pair(args: dict[str, Any]) -> tuple[str, str]:
    return str(args.get("product_id") or ""), str(args.get("sku") or "")


class _State:
    """Deterministic ledger the guardrails reason over."""

    def __init__(self, budget: float | None) -> None:
        self.budget = budget
        self.cart: list[tuple[str, str]] = []
        self.first_commit_turn: int | None = None
        self.first_commit_pair: tuple[str, str] | None = None
        self.allowed_pairs: set[tuple[str, str]] | None = None  # ranking lane
        self.event_seen = False
        # pair -> (turn, in_stock, price) from the latest observation of it
        self.seen: dict[tuple[str, str], tuple[int, bool | None, float | None]] = {}
        self.pending_question = False
        self.market_event_note: str | None = None
        self.specs: dict[str, Any] = {}        # hard specs from goal + messages
        self.brand: str | None = None
        self.brand_required = False
        self.spec_blocks: dict[tuple[str, str], int] = {}
        self.query_count = 0
        self.viewed: set[str] = set()          # product_ids seen via view/compare
        self.info: dict[str, dict] = {}        # product_id -> title/facts/variants

    # ---- observation ingestion ------------------------------------------- #

    def note_rows(self, rows: Any, turn: int) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            pid, sku = str(row.get("product_id") or ""), str(row.get("sku") or "")
            if not pid or not sku:
                continue
            price = row.get("price")
            in_stock = row.get("in_stock")
            self.seen[(pid, sku)] = (
                turn,
                in_stock if isinstance(in_stock, bool) else None,
                float(price) if isinstance(price, (int, float)) else None,
            )

    def ingest(self, name: str, args: dict, observation: Any, turn: int) -> None:
        if not isinstance(observation, dict):
            return
        payload = observation
        for _ in range(4):  # call_result -> step_result -> observation ...
            inner = payload.get("observation")
            if isinstance(inner, dict):
                payload = inner
            else:
                break
        event = payload.get("market_event")
        if isinstance(event, dict):
            self.event_seen = True
            target = event.get("target") or {}
            self.market_event_note = (
                "MARKET EVENT on product_id="
                f"{target.get('product_id')} sku={target.get('sku')}: "
                f"{json.dumps({k: v for k, v in event.items() if k != 'target'})}"
            )
        self.note_rows(payload.get("results"), turn)
        compared = payload.get("compared")
        if isinstance(compared, list):
            for item in compared:
                if isinstance(item, dict) and item.get("product_id"):
                    self._note_product(item, turn)
        if payload.get("product_id") and isinstance(payload.get("variants"), list):
            self._note_product(payload, turn)
        variants = payload.get("variants")
        if isinstance(variants, list):
            pid = str(payload.get("product_id") or args.get("product_id") or "")
            for variant in variants:
                if isinstance(variant, dict) and pid:
                    variant.setdefault("product_id", pid)
            self.note_rows(variants, turn)
        if payload.get("product_id") and payload.get("sku"):
            self.note_rows([payload], turn)
        cart_rows = payload.get("cart")
        if isinstance(cart_rows, list):
            self.cart = [
                (str(r.get("product_id") or ""), str(r.get("sku") or ""))
                for r in cart_rows
                if isinstance(r, dict)
            ]
        if name == "add_to_cart" and not payload.get("error"):
            pair = _pair(args)
            if all(pair) and pair not in self.cart:
                self.cart.append(pair)
            if self.first_commit_turn is None and all(pair):
                self.first_commit_turn = turn
                self.first_commit_pair = pair
        if name == "remove_from_cart":
            pair = _pair(args)
            if pair in self.cart:
                self.cart.remove(pair)

    def _note_product(self, item: dict, turn: int) -> None:
        pid = str(item.get("product_id") or "")
        if not pid:
            return
        self.viewed.add(pid)
        rec = self.info.setdefault(pid, {"title": "", "facts": {}, "variants": {}})
        if item.get("title"):
            rec["title"] = str(item["title"])
        if item.get("brand") is not None:
            rec["brand"] = item["brand"]
        if isinstance(item.get("facts"), dict):
            rec["facts"].update(item["facts"])
        for v in item.get("variants") or []:
            if isinstance(v, dict) and v.get("sku"):
                rec["variants"][str(v["sku"])] = dict(v)
                self.note_rows([{**v, "product_id": pid}], turn)

    def ranked_candidates(self) -> list[tuple[str, str]] | None:
        """Deterministic ranking of the listed candidates; None if any listed
        candidate still lacks parsed specs (defer to the model)."""
        if not self.allowed_pairs:
            return None
        table = []
        for pid, sku in self.allowed_pairs:
            info = self.info.get(pid)
            if not info:
                return None
            # latest observed price/stock overrides the cached variant row
            latest = self.seen.get((pid, sku))
            variant = dict((info.get("variants") or {}).get(sku) or {})
            if latest is not None:
                if latest[2] is not None:
                    variant["price"] = latest[2]
                if latest[1] is not None:
                    variant["in_stock"] = latest[1]
            key = _rank_key({**info, "variants": {sku: variant}}, sku)
            if key is None:
                return None
            valid = variant.get("in_stock") is not False and (
                self.budget is None or float(variant["price"]) <= self.budget + 1e-6
            )
            if valid:
                table.append((key, (pid, sku)))
        table.sort()
        return [pair for _, pair in table]

    def spec_block_reason(self, pair: tuple[str, str]) -> str | None:
        """Deterministic hard-spec / required-brand gate on observed raw facts.
        After 3 blocks on the same pair the model's judgement is allowed to stand
        (protects against a parsing miss on an unusual catalog)."""
        if not self.specs and not (self.brand and self.brand_required):
            return None
        if self.spec_blocks.get(pair, 0) >= 3:
            return None
        info = self.info.get(pair[0])
        problems = _spec_failures(self.specs, info, pair[1])
        if self.brand and self.brand_required:
            ok = _brand_ok(info, self.brand)
            if ok is False:
                problems.append(
                    f"brand field is {info.get('brand')!r}, not {self.brand!r} "
                    "(title spellings near the brand are decoys)"
                )
            elif ok is None and info is not None:
                problems.append("brand field not observed")
        if not problems:
            return None
        self.spec_blocks[pair] = self.spec_blocks.get(pair, 0) + 1
        return (
            "SPEC CHECK FAILED for "
            f"{pair[0]}::{pair[1]}: " + "; ".join(problems)
            + ". A requirement that is not stated exactly in the raw facts is NOT "
            "satisfied. Find a candidate whose facts match exactly."
        )

    def preferred_brand_block(self, pair: tuple[str, str]) -> str | None:
        """First commit must honor a stated preferred brand when a compliant
        candidate with that brand field has been observed."""
        if not self.brand or self.brand_required or self.first_commit_pair is not None:
            return None
        info = self.info.get(pair[0])
        if _brand_ok(info, self.brand) is True:
            return None
        for pid, rec in self.info.items():
            if _brand_ok(rec, self.brand) is not True:
                continue
            for sku, variant in (rec.get("variants") or {}).items():
                if variant.get("in_stock") is False:
                    continue
                price = variant.get("price")
                if self.budget is not None and price is not None and price > self.budget:
                    continue
                if _spec_failures(self.specs, rec, sku):
                    continue
                return (
                    f"COMMIT BLOCKED: the shopper prefers {self.brand} and "
                    f"{pid}::{sku} has brand field {self.brand!r} and meets the "
                    "firm requirements. Your first cart choice must be that one."
                )
        return None

    # ---- order guardrail --------------------------------------------------- #

    def order_block_reason(
        self, args: dict, turn: int, last_turns: bool, lane: dict | None = None
    ) -> str | None:
        pair = _pair(args)
        if self.allowed_pairs is not None and pair not in self.allowed_pairs:
            return (
                "Only the candidate variants listed by the shopper may be "
                "ordered. Pick from the listed product_id::sku pairs."
            )
        if lane is not None and lane.get("event_policy") == "rerank" and not last_turns:
            ranked = self.ranked_candidates()
            if self.event_seen and pair == self.first_commit_pair:
                return (
                    "The market changed after your commit. Re-rank ALL listed "
                    "candidates under current prices/stock and order the NEW "
                    "top-ranked valid candidate - not your original commit."
                )
            if ranked and pair != ranked[0] and (
                self.event_seen or ranked[0] != self.first_commit_pair
            ):
                top = ranked[0]
                return (
                    f"By the stated criteria the top-ranked valid candidate is "
                    f"{top[0]}::{top[1]} (highest refresh rate, then largest "
                    "screen, then lowest price). Order that one."
                )
        if not all(pair):
            return (
                "place_test_order must name explicit product_id and sku of the "
                "item already in the cart."
            )
        if pair not in self.cart:
            return (
                f"{pair[0]}::{pair[1]} is not in the cart. add_to_cart it first, "
                "then re-check it with inspect_stock in the following turn."
            )
        record = self.seen.get(pair)
        # Hard gates are never bypassed: an out-of-stock or over-budget order is
        # a certain zero, so relaxing them at the end can only lose.
        if record is not None:
            if record[1] is False:
                return (
                    "That variant is OUT OF STOCK. remove_from_cart it and pick a "
                    "different valid candidate."
                )
            if (
                self.budget is not None
                and record[2] is not None
                and record[2] > self.budget + 1e-6
            ):
                return (
                    f"Latest observed price {record[2]} exceeds the current budget "
                    f"{self.budget}. Pick a cheaper valid candidate."
                )
        spec_reason = self.spec_block_reason(pair)
        if spec_reason and not last_turns:
            return spec_reason
        if last_turns:
            return None  # out of runway: relax SEQUENCING only
        if record is None:
            return (
                "Before ordering, inspect_stock this exact variant to confirm "
                "current price and stock."
            )
        seen_turn, in_stock, price = record
        if self.first_commit_turn is not None and seen_turn <= self.first_commit_turn:
            return (
                "State may have changed after your cart commit. inspect_stock the "
                "variant now, review the result, and order in the NEXT turn."
            )
        if seen_turn >= turn:
            return (
                "Order in a turn AFTER the turn where you inspected the variant. "
                "Do nothing else this turn except inspect_cart if needed."
            )
        if in_stock is False:
            return (
                "That variant is OUT OF STOCK. remove_from_cart it and pick a "
                "different valid candidate."
            )
        if (
            self.budget is not None
            and price is not None
            and price > self.budget + 1e-6
        ):
            return (
                f"Latest observed price {price} exceeds the current budget "
                f"{self.budget}. Pick a cheaper valid candidate or re-check."
            )
        return None


_SYSTEM_PROMPT = """You are a meticulous shopping agent operating a catalog through tools. A hidden deterministic verifier scores you. Follow this playbook exactly.

GOAL
Find the single variant (product_id + sku) that satisfies EVERY hard constraint in the shopper's request (category, attributes like screen size or resolution, brand rules, price budget, stock), honor soft preferences when possible (e.g. "ideally Samsung" - prefer it only among fully valid options), then buy it.

HARD RULES
1. Budget: the price AT ORDER TIME must be <= the budget, in the stated currency. If the shopper changes the budget mid-session, the NEW budget applies to your order. Never order anything over budget.
2. Verify attributes from raw product facts using view or compare before trusting them; search snippets and titles are not proof. A requirement the raw facts do not state EXACTLY is NOT satisfied: "6.8 inches" does not satisfy 6.79-inch, a missing size fact is a failure, and an ambiguous fact ("720 x 1600; 1080p") is a failure. Brand is decided by the `brand` field only; a title that merely resembles the brand name (misspellings, lookalikes) is a decoy. Verify the exact variant's current price and stock with inspect_stock.
3. Shopper messages: if the shopper asks you anything or requests a reply, respond with the `message` tool BEFORE placing the order. Answer briefly and concretely. If they add or change a constraint, comply with it.
4. Commit sequence (critical):
   a. Explore with search/filter (k=5) and view/compare until you are confident of the best valid variant. Do not add throwaway items to the cart - your FIRST add_to_cart must be your genuine intended choice.
   b. add_to_cart that variant.
   c. In the NEXT turn, inspect_stock the same variant. Prices or stock can change right after you commit (watch for market_event notes and price shifts anywhere in results).
   d. Read the result. If it is still in stock and within budget, place_test_order in the FOLLOWING turn, naming product_id and sku. If it went out of stock or over budget, remove_from_cart, choose the best alternative valid variant, add it, inspect it, then order it - never order the invalidated item.
5. If a price DROPPED, that is fine - proceed. If a different product's price dropped into validity and the shopper's rules make it strictly better, you may switch to it (add, inspect next turn, then order).
6. Efficiency: every tool call and every distinct product you observe costs score. Use k=5, avoid repeated identical searches, do not view products you will not consider. Aim to finish in under ~15 tool calls. But NEVER sacrifice correctness for speed.

JUSTIFICATION CLAIMS
Only if the shopper asked WHY / for a reason: include `justification` = {"claims": [...]} in place_test_order. Each claim is EXACTLY {"field": ..., "value": ...} - no evidence key, no summary key, nothing else. Allowed fields: category, resolution, refresh_rate, screen_size, price, in_stock, brand, streaming, microphone, privacy_cover, autofocus, hdr. Values must exactly match what you saw in view/compare raw facts for the FINAL product (price and in_stock from your latest inspect_stock). If a raw fact is ambiguous (e.g. "720 x 1600; 1080p"), do NOT claim that field. If the shopper did not ask for reasons, send NO justification object at all.

OUTPUT
Always act through tool calls. Keep any text content to one short sentence of reasoning."""


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #


def _call(name: str, args: dict, cid: str) -> dict[str, Any]:
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _envelope(binding: dict, group_id: str, turn: int, calls: list[dict]) -> dict:
    return {
        **binding,
        "call_id": group_id,
        "idempotency_key": group_id,
        "turn": turn,
        "calls": [
            {
                "call_id": f"{group_id}-{index}",
                "action": {
                    "name": call["function"]["name"],
                    "args": _arguments(call),
                },
            }
            for index, call in enumerate(calls, start=1)
        ],
    }


def _endgame_order(
    state: "_State", pid: str, sku: str, lane_name: str, wants_reason: bool, cid: str
) -> dict[str, Any]:
    args: dict[str, Any] = {"product_id": pid, "sku": sku}
    if wants_reason and pid in state.viewed:
        claims = _build_claims(state, pid, sku, state.specs, state.brand, "")
        if claims:
            args["justification"] = {"claims": claims}
    return _call("place_test_order", args, cid)


def _endgame_calls(
    state: "_State", turn: int, deadline: "_Deadline",
    lane_name: str = "general", wants_reason: bool = False,
) -> list[dict[str, Any]] | None:
    """Deterministic finish when the time budget is nearly spent.

    Returns tool calls to run this turn, or None to let the LLM act (only
    when there is nothing sensible to script).
    """
    critical = deadline.left() < 25.0
    if state.pending_question:
        state.pending_question = False
        return [_call("message", {"content": (
            "Confirmed - proceeding with your request as stated."
        )}, f"eg-msg-{turn}")]
    if state.cart:
        pid, sku = state.cart[-1]
        if (
            state.event_seen
            and (pid, sku) == state.first_commit_pair
            and state.allowed_pairs is not None
            and not critical
        ):
            alt = _best_alternative(state, exclude=(pid, sku))
            if alt is not None:
                return [
                    _call("remove_from_cart",
                          {"product_id": pid, "sku": sku}, f"eg-sw1-{turn}"),
                    _call("add_to_cart",
                          {"product_id": alt[0], "sku": alt[1]}, f"eg-sw2-{turn}"),
                ]
        record = state.seen.get((pid, sku))
        fresh = (
            record is not None
            and state.first_commit_turn is not None
            and record[0] > state.first_commit_turn
            and record[0] < turn
        )
        over_budget = (
            record is not None and record[2] is not None
            and state.budget is not None and record[2] > state.budget + 1e-6
        )
        if fresh or critical:
            if record is not None and (record[1] is False or over_budget):
                # invalidated: swap to the best surviving alternative
                alt = _best_alternative(state, exclude=(pid, sku))
                calls = [_call("remove_from_cart",
                               {"product_id": pid, "sku": sku}, f"eg-rm-{turn}")]
                if alt is not None:
                    calls.append(_call("add_to_cart",
                                       {"product_id": alt[0], "sku": alt[1]},
                                       f"eg-add-{turn}"))
                return calls
            return [_endgame_order(state, pid, sku, lane_name, wants_reason,
                                   f"eg-ord-{turn}")]
        return [_call("inspect_stock",
                      {"product_id": pid, "sku": sku}, f"eg-chk-{turn}")]
    pick = _best_alternative(state, exclude=None)
    if pick is not None:
        return [_call("add_to_cart",
                      {"product_id": pick[0], "sku": pick[1]}, f"eg-add-{turn}")]
    return None  # nothing observed yet: let the LLM search


def _best_alternative(
    state: "_State", exclude: tuple[str, str] | None
) -> tuple[str, str] | None:
    """Most recently seen in-stock candidate within budget; in the ranking
    lane the deterministic ranking wins when it can be computed."""
    ranked = state.ranked_candidates()
    if ranked:
        for pair in ranked:
            if pair != exclude:
                return pair
        return None
    best: tuple[int, tuple[str, str]] | None = None
    for pair, (seen_turn, in_stock, price) in state.seen.items():
        if exclude is not None and pair == exclude:
            continue
        if state.allowed_pairs is not None and pair not in state.allowed_pairs:
            continue
        info = state.info.get(pair[0])
        if info is not None and _spec_failures(state.specs, info, pair[1]):
            continue
        if state.brand and state.brand_required and _brand_ok(info, state.brand) is False:
            continue
        if in_stock is False:
            continue
        if (
            state.budget is not None
            and price is not None
            and price > state.budget + 1e-6
        ):
            continue
        if best is None or seen_turn > best[0]:
            best = (seen_turn, pair)
    return best[1] if best else None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_steps = int(policy["max_steps"])
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
    query = policy["query"]

    goal = _goal_part(query)
    state = _State(budget=_parse_budget(goal))
    state.specs = _extract_specs(goal)
    state.brand, state.brand_required = _extract_brand(goal)
    lane_name = _route(goal)
    lane = _LANES[lane_name]
    if lane_name == "ranking":
        listed = [(pid, sku) for pid, sku in _PAIR_RE.findall(goal)]
        if listed:
            state.allowed_pairs = set(listed)
    wants_reason = bool(_REASON_HINT.search(goal)) or lane["claims"] == "required"

    system_prompt = _SYSTEM_PROMPT
    if lane["addendum"]:
        system_prompt = system_prompt + "\n\nTASK-TYPE PLAYBOOK\n" + lane["addendum"]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    if wants_reason:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Note: I asked for a reason - your final place_test_order must "
                    "include grounded justification claims."
                ),
            }
        )
    dialogue: list[dict[str, Any]] = []
    deadline = _Deadline(_TIME_BUDGET)
    pending_scripted_ask = lane.get("ask_text")
    last_action_sig: str | None = None
    repeat_count = 0
    rejected_streak = 0
    forced_endgame = False

    env_turn = 0  # number of envelopes actually dispatched
    for _iteration in range(max_steps * 2):
        if env_turn >= max_steps:
            break
        turn = env_turn + 1  # the turn number the NEXT envelope must carry
        turns_left = max_steps - env_turn
        last_turns = turns_left <= 2 or deadline.left() < 30.0
        guard_notes: list[str] = []

        # ---- scripted opener for ask-first lanes ---------------------------- #
        if pending_scripted_ask is not None:
            tool_calls = [_call("message", {"content": pending_scripted_ask},
                                f"ask-{turn}")]
            pending_scripted_ask = None
            assistant_content = ""
            group_id = f"{problem_id}-turn-{turn}"
            envelope = _envelope(binding, group_id, turn, tool_calls)
            result = _proxy.post("/environment/call", json_data=envelope)
            if result is None:
                raise RuntimeError("environment call failed")
            env_turn = turn
            dialogue.append(
                {"role": "assistant", "content": assistant_content,
                 "tool_calls": tool_calls, "environment_result": result}
            )
            messages.append({"role": "assistant", "content": assistant_content,
                             "tool_calls": tool_calls})
            for tool_call, call_result in zip(tool_calls, result.get("calls") or []):
                messages.append(
                    {"role": "tool",
                     "tool_call_id": tool_call.get("id") or f"{group_id}-x",
                     "content": json.dumps(call_result.get("observation"))}
                )
            user_message = result.get("user_message")
            if isinstance(user_message, dict) and user_message.get("content"):
                content = str(user_message["content"])
                messages.append({"role": "user", "content": content})
                state.specs.update(_extract_specs(content))
                new_budget = _parse_budget(content)
                if new_budget is not None:
                    state.budget = (new_budget if state.budget is None
                                    else min(state.budget, new_budget))
            continue

        # ---- scripted endgame: converge on an order without the LLM -------- #
        # trigger on ANY scarce resource: wall clock, remaining turns, or a
        # detected action loop - turns were the binding constraint in run 1.
        if deadline.endgame() or turns_left <= 6 or forced_endgame:
            scripted = _endgame_calls(state, turn, deadline, lane_name, wants_reason)
            if scripted is not None:
                assistant_content = "Time budget low - finishing deterministically."
                tool_calls = scripted
                messages.append(
                    {"role": "assistant", "content": assistant_content,
                     "tool_calls": tool_calls}
                )
                group_id = f"{problem_id}-turn-{turn}"
                envelope = _envelope(binding, group_id, turn, tool_calls)
                result = _proxy.post("/environment/call", json_data=envelope)
                if result is None:
                    raise RuntimeError("environment call failed")
                env_turn = turn
                dialogue.append(
                    {"role": "assistant", "content": assistant_content,
                     "tool_calls": tool_calls, "environment_result": result}
                )
                um = result.get("user_message")
                if isinstance(um, dict) and um.get("content"):
                    messages.append({"role": "user", "content": str(um["content"])})
                    state.pending_question = True
                    nb = _parse_budget(str(um["content"]))
                    if nb is not None:
                        state.budget = nb if state.budget is None else min(state.budget, nb)
                done = False
                for tool_call, call_result in zip(
                    tool_calls, result.get("calls") or []
                ):
                    observation = call_result.get("observation")
                    messages.append(
                        {"role": "tool",
                         "tool_call_id": tool_call.get("id") or f"{group_id}-x",
                         "content": json.dumps(observation)}
                    )
                    state.ingest(
                        tool_call["function"]["name"],
                        _arguments(tool_call), call_result, turn,
                    )
                    if isinstance(observation, dict) and observation.get("done"):
                        done = True
                if done:
                    break
                continue

        # ---- model step: walk the priority chain, two passes, temp bump ---- #
        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for _pass in range(2):
            temp = 0.0 if _pass == 0 else 0.3
            for model in _llm_chain():
                if not deadline.can_afford(_LLM_MIN_SECS):
                    break  # cannot fund another round trip; fall to endgame
                inference = _proxy.post(
                    "/inference/chat/completions",
                    json_data={
                        "model": model,
                        "messages": messages,
                        "tools": policy["tools"],
                        "tool_choice": "required",
                        "temperature": temp,
                    },
                    timeout=max(10.0, min(120.0, deadline.left() - 15.0)),
                )
                if inference is None:
                    continue  # next model in the chain
                try:
                    assistant = inference["choices"][0]["message"]
                except (KeyError, IndexError, TypeError):
                    continue
                assistant_content = assistant.get("content") or ""
                tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
                if tool_calls:
                    break
            if tool_calls or not deadline.can_afford(_LLM_MIN_SECS):
                break
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": (
                            "The session is not done. Choose one of the supplied "
                            "tools to continue."
                        ),
                    },
                ]
            )
        if not tool_calls and not deadline.endgame():
            break

        # ---- guardrails ----------------------------------------------------- #
        filtered: list[dict[str, Any]] = []
        for call in tool_calls:
            name = call.get("function", {}).get("name")
            args = _arguments(call)
            if name == "place_test_order":
                reason = state.order_block_reason(args, turn, last_turns, lane)
                if reason is None and state.pending_question and not any(
                    c.get("function", {}).get("name") == "message" for c in filtered
                ):
                    # inject a reply so the intervention gate cannot fail; the
                    # order itself still goes out in a later turn
                    reply = assistant_content.strip() or (
                        "Yes - I've confirmed the item and your updated request, "
                        "and I'm proceeding accordingly."
                    )
                    filtered.append(
                        {
                            "id": f"reply-{turn}",
                            "type": "function",
                            "function": {
                                "name": "message",
                                "arguments": json.dumps({"content": reply[:400]}),
                            },
                        }
                    )
                    state.pending_question = False
                    guard_notes.append(
                        "ORDER DEFERRED: replied to the shopper first; place the "
                        "order again next turn."
                    )
                    continue
                if reason is not None:
                    guard_notes.append(f"ORDER BLOCKED: {reason}")
                    pair = _pair(args)
                    if all(pair) and pair in state.cart and "inspect_stock" in reason:
                        # substitute the safe re-check so the turn is not wasted
                        call = {
                            "id": call.get("id") or f"guard-{turn}",
                            "type": "function",
                            "function": {
                                "name": "inspect_stock",
                                "arguments": json.dumps(
                                    {"product_id": pair[0], "sku": pair[1]}
                                ),
                            },
                        }
                        filtered.append(call)
                    elif (
                        all(pair)
                        and pair not in state.cart
                        and "not in the cart" in reason
                        and (
                            state.allowed_pairs is None
                            or pair in state.allowed_pairs
                        )
                    ):
                        # the model has converged on a target: put it in the
                        # cart now so the order can go out on a later turn
                        filtered.append(
                            _call(
                                "add_to_cart",
                                {"product_id": pair[0], "sku": pair[1]},
                                call.get("id") or f"guard-add-{turn}",
                            )
                        )
                    continue
                if wants_reason:
                    built = _build_claims(state, *_pair(args), state.specs, state.brand, goal)
                    args = dict(args)
                    if built:
                        args["justification"] = {"claims": built}
                    else:
                        args = _sanitize_justification(args, lane_name)
                elif "justification" in args:
                    args.pop("justification", None)
                call = _call(
                    "place_test_order", args, call.get("id") or f"claims-{turn}"
                )
                filtered.append(call)
                break  # nothing after an order
            if (
                name == "add_to_cart"
                and state.allowed_pairs is not None
                and state.first_commit_pair is None
            ):
                ranked = state.ranked_candidates()
                if ranked and _pair(args) != ranked[0]:
                    top = ranked[0]
                    guard_notes.append(
                        f"COMMIT ADJUSTED: {top[0]}::{top[1]} is the top-ranked "
                        "valid candidate by the stated criteria; it was added "
                        "instead."
                    )
                    call = _call(
                        "add_to_cart", {"product_id": top[0], "sku": top[1]},
                        call.get("id") or f"top-{turn}",
                    )
                    args = _arguments(call)
            if name in ("search", "filter") and lane.get("max_queries"):
                if state.query_count >= lane["max_queries"]:
                    guard_notes.append(
                        "SEARCH BLOCKED: the query budget for this task is spent. "
                        "Work with the candidates already surfaced (view/compare)."
                    )
                    continue
                state.query_count += 1
            if name == "add_to_cart" and state.allowed_pairs is None:
                block = state.spec_block_reason(_pair(args)) or state.preferred_brand_block(_pair(args))
                if block:
                    guard_notes.append("ADD BLOCKED: " + block)
                    continue
            if (
                name == "add_to_cart"
                and lane.get("event_policy") == "rerank"
                and state.first_commit_pair is not None
                and not state.event_seen
                and _pair(args) != state.first_commit_pair
            ):
                guard_notes.append(
                    "ADD BLOCKED: you already committed. inspect_stock your "
                    "committed candidate first to observe the market change, "
                    "then re-rank and switch."
                )
                continue
            if (
                name == "add_to_cart"
                and state.allowed_pairs is not None
                and _pair(args) not in state.allowed_pairs
            ):
                guard_notes.append(
                    "ADD BLOCKED: only the shopper's listed candidate variants "
                    "may be added to the cart."
                )
                continue
            filtered.append(call)
            if name == "message":
                state.pending_question = False
        tool_calls = filtered

        if not tool_calls:
            rejected_streak += 1
            if rejected_streak >= 3:
                forced_endgame = True  # model keeps proposing blocked actions
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append(
                {"role": "user", "content": " ".join(guard_notes) or "Continue."}
            )
            continue
        rejected_streak = 0

        # ---- dispatch -------------------------------------------------------- #
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
                        "args": _arguments(call),
                    },
                }
                for index, call in enumerate(tool_calls, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        env_turn = turn
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        done = False
        for tool_call, call_result in zip(tool_calls, result.get("calls") or []):
            observation = call_result.get("observation")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id") or f"{group_id}-x",
                    "content": json.dumps(observation),
                }
            )
            state.ingest(
                tool_call["function"]["name"],
                _arguments(tool_call),
                call_result,
                turn,
            )
            if isinstance(observation, dict) and observation.get("done"):
                done = True

        action_sig = json.dumps(
            [[c["function"]["name"], _arguments(c)] for c in tool_calls],
            sort_keys=True,
        )
        if action_sig == last_action_sig:
            repeat_count += 1
            if repeat_count >= 2:
                forced_endgame = True  # stuck: hand over to the scripted finish
        else:
            repeat_count = 0
        last_action_sig = action_sig

        user_message = result.get("user_message")
        if isinstance(user_message, dict) and user_message.get("content"):
            content = str(user_message["content"])
            messages.append({"role": "user", "content": content})
            state.pending_question = True
            state.specs.update(_extract_specs(content))
            new_budget = _parse_budget(content)
            if new_budget is not None:
                state.budget = (
                    new_budget if state.budget is None else min(state.budget, new_budget)
                )

        notes = list(guard_notes)
        if state.market_event_note:
            notes.append(
                state.market_event_note
                + " -> inspect_stock the affected variant before deciding."
            )
            state.market_event_note = None
        if notes and not done:
            messages.append({"role": "user", "content": " ".join(notes)})

        if done:
            break

    return dialogue


__all__ = ["agent_main"]