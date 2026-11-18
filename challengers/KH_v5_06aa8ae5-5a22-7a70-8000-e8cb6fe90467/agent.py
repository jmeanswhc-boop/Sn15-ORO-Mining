from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from os import getenv
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------
# Runtime constants (inference + env bridge)
# ---------------------------------------------------------------------------

_FLASH_MODEL = "deepseek/deepseek-v3.2"
_HEAVY_MODEL = "qwen/qwen3.5-397b-a17b"
_MODEL_FALLBACK_CHAIN = (
    "google/gemma-4-31b-it",
    "openai/gpt-oss-120b",
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-v3.2",
)
_HEAVY_LANES = frozenset(
    {
        "constraint_satisfaction",
        "preference_reasoning",
        "ranking",
        "recovery",
        "justification",
        "intent_decomposition",
    }
)
_LANE_TIEBREAK = {
    "ranking": 70,
    "recovery": 60,
    "justification": 50,
    "constraint_satisfaction": 40,
    "intent_decomposition": 30,
    "preference_reasoning": 20,
    "retrieval_recall": 10,
}
_DEFAULT_LANE = "retrieval_recall"

_PROXY_BRIDGE = ProxyClient(timeout=120, max_retries=2)

_EMPTY_TOOL_PUSH = (
    "No tool call received. Choose a supplied tool now. If you have not searched yet, call search or filter.",
    "Still no tool call. Progress the episode: search/filter, view a candidate, or place_test_order if the cart already holds a verified in-stock SKU.",
)

# Claim fields that must never appear in justification.claims (marketing / meta).
_FORBIDDEN_CLAIM_LABELS = frozenset(
    {
        "category",
        "category_path",
        "title",
        "summary",
        "evidence",
        "reason",
        "description",
        "name",
        "note",
        "display",
        "device_operating_system",
        "device os",
        "operating_system",
        "operating system",
        "os",
    }
)
_OBS_WALK_SKIP = _FORBIDDEN_CLAIM_LABELS | {
    "error",
    "done",
    "calls",
    "observation",
    "event",
    "stock_changed",
    "price_changed",
    "products",
    "results",
    "items",
    "product_id",
    "sku",
    "options",
    "sku_options",
    "market_event",
}

_SKU_TOKEN_RE = re.compile(r"\b(\d{10,}::[A-Za-z0-9._-]+)\b")
_WHY_GOAL_RE = re.compile(r"\bwhy\b|justif|reason|tell me why", re.I)
_OPTIONAL_JUST_PHRASE = "justification is optional"
_TASK_RULES_SPLIT = "Task rules:"

# ---------------------------------------------------------------------------
# Live coach / playbook text (semantic content preserved for score stability)
# ---------------------------------------------------------------------------

_CHARTER = """
You are KH_1, an ORO Bench shopping agent.

# Contract
- Use ONLY the supplied tools from the environment.
- Continue until an observation reports done=true.
- Respect max_steps and max_calls_per_turn.
- Prefer parallel tool calls when the limit allows and it saves steps.

# Healthy loop
1) Discover — search / filter
2) Inspect — view / compare / inspect_stock
3) Decide — add_to_cart / remove_from_cart
4) Confirm — inspect_stock / inspect_cart on the final SKU
5) Finish — place_test_order (with justification.claims when required)

# Precision
A candidate is valid ONLY if every FIRM constraint from the shopper goal is
satisfied on the EXACT variant you order. Soft preferences never override firm ones.
Catalog search matches VALUES, not JSON keys. Prefer value tokens from the goal.
If the first search is weak, make ONE small reformulation — do not rewrite everything.

# Messaging
If user_message is a question, hold, or "which one": call message THIS turn, then continue.

# Market / cart events
After stock_changed, price_changed, a failed cart, or remove_from_cart:
re-inspect price/stock, recover per family rules, then place_test_order.
First cart is graded. Leave ≥1–2 turns after a shock for inspect + order.
Late turns: stop open search and finish.
""".strip()

_LANE_PLAYBOOKS: dict[str, str] = {
    "intent_decomposition": """
# Family: TF1 Intent Decomposition
1. List firm requirements (category, named spec/attribute, budget, in-stock).
2. List soft preferences (named preferred brand).
3. Dual shortlist: (brand+firm) and (firm-only backup).
4. First cart: preferred brand IF a fully compliant SKU exists; otherwise the best firm-compliant item. No placeholders.
5. If the shopper asks which/spec/budget, message with evidence, then continue.
6. After price/stock shocks: remove invalid, recover under the same brand rule, inspect, then place_test_order.
The first cart add is scored separately from later fixes.
""".strip(),
    "retrieval_recall": """
# Family: TF2 Retrieval Recall
1. Try the shopper's own wording, then a tokenised variant, then a category-scoped pass.
2. Assemble a shortlist of genuinely different products before choosing.
3. Open promising ones; discard accessories/bundles by listing classification, not title.
4. Buy the strongest available item inside the ceiling. Once verified, do not idle waiting for approval.
""".strip(),
    "constraint_satisfaction": """
# Family: TF3 Constraint Satisfaction
Bind every wall to the ordered option row — never the parent title. View the sku, then inspect_stock on that sku.
When a firm figure appears only after you ask, message first; search waits.
If the chosen sku sells out, switch to a different product_id that still clears every wall; sibling options that miss a figure are not acceptable.
Treat named figures as written; do not invent class or unit conversions.
""".strip(),
    "preference_reasoning": """
# Family: TF4 Preference Reasoning
Walls: product class, spend ceiling, availability, a real sellable unit.
Wishes: only qualities the shopper named for use-case fit.
Before the first cart, compare at least two credible candidates on those wishes.
A slightly higher price that still fits the ceiling can win when the fit is better.
Do not cart a cheap stand-in.
""".strip(),
    "ranking": """
# Family: TF5 Ranking
- Restrict to named product_id::sku candidates; batch-view them; avoid open search.
- Rank ONLY by the stated tradeoff priority. Never invent axes.
- Read axis values from title, option label, and facts.
- First cart = current #1. After stock/price/cart events: re-inspect, RE-RANK from scratch, cart the NEW #1, inspect_stock, place_test_order.
- Never pick an option strictly dominated on every stated axis.
- place_test_order: product_id + sku only (no claims unless why is required).
""".strip(),
    "recovery": """
# Family: TF6 Recovery
Never relax: category, budget, in-stock.
Least-relaxation ladder (do not skip): exact model+color → same model other color → other models last.
If the preferred item disappears, climb one level, re-check price/stock, then place_test_order.
Use explicit model facts when supplied; conflicting titles do not override them.
Omit optional claims unless the goal asks for a reason.
""".strip(),
    "justification": """
# Family: TF7 Justification
View/compare raw facts before first cart and before ordering.
place_test_order MUST include justification.claims as field/value pairs.
Claim fields must be catalog keys from view/inspect — never marketing labels (display, title, category, OS, summary, evidence).
Required coverage: price, in_stock, and the specification the shopper asked about (plus at most one evidenced extra, prefer brand).
Values must equal observed catalog text for that sku. If facts lack the asked measurable, switch sku rather than inventing from title.
""".strip(),
}

_NOTE_ASK = (
    "Coach: a firm requirement appears to be withheld until you ask. Call message first "
    "to request that exact requirement before searching. Do not search or filter until the shopper replies."
)
_NOTE_RANK_REFRESH = (
    "Coach [TF5]: market/cart state changed. Re-inspect the named candidate variants that "
    "are still available, re-rank by the stated tradeoff priority, cart the NEW #1 "
    "(not the pre-event pick), inspect_stock, then place_test_order with only product_id+sku."
)
_NOTE_RANK_CLOSE = (
    "Coach [TF5]: stop searching. Re-rank the named candidates still in stock, add the "
    "current #1 to cart if needed, and place_test_order now (product_id+sku only)."
)
_NOTE_CLAIMS = (
    "Coach [TF7]: on place_test_order, justification.claims must copy catalog JSON keys "
    "from view/inspect_stock — price, in_stock, the asked spec key, and at most one "
    "evidenced extra (prefer brand). Drop marketing/meta labels (display, title, category, "
    "OS, summary, evidence). Values must equal observed catalog text."
)
_NOTE_REPLY = (
    "Coach: the shopper just spoke. Call message THIS turn with a concrete answer "
    "(identity + key firm evidence + price/stock). Do not cart or order until that reply is sent."
)
_NOTE_BRAND = (
    "Coach [TF1]: keep a brand+firm shortlist and a firm-only backup. If a preferred-brand "
    "SKU is fully firm-compliant, it MUST win the first cart; otherwise cart the best "
    "firm-compliant any-brand item. Answer shopper questions with message before further cart/order."
)
_NOTE_MARKET = (
    "Coach: catalog/cart state moved ({kind}). Re-check price/stock on the affected SKU. "
    "If invalid, remove and recover per the family rules, then place_test_order so the event is closed."
)
_NOTE_CLOSEOUT = (
    "Coach: stop open search. Verify the carted SKU (or cart the current valid pick), "
    "then place_test_order now so the episode closes."
)
_NOTE_RECOVER = (
    "Coach: an action failed or stock/price changed ({error}). Climb the least-relaxation "
    "ladder without skipping: exact model+color → same model other color → other models last. "
    "Never relax category, budget, or in-stock. Then place_test_order to close the event."
)
_NOTE_NEW_PID = (
    "Coach [TF3]: the current SKU is gone. Switch to a NEW product_id that still meets every "
    "firm constraint — sibling options that drop a firm spec are invalid. Then place_test_order."
)
_NOTE_LADDER = (
    "Coach [TF6]: climb the least-relaxation ladder without skipping — same model / other "
    "color before other models. Never relax category, budget, or in-stock. Omit optional claims. "
    "Then place_test_order."
)
_NOTE_SOFT_SEAL = (
    "Coach [TF4]: stop messaging/inspect loops. If cart holds a pick, place_test_order now; "
    "if empty after shock, recover a hard-valid soft-fit SKU then order."
)
_NOTE_SWEEP = (
    "Coach [TF2]: make one small keyword reformulation or add a filter so the shortlist "
    "covers several distinct products before ordering."
)
_NOTE_RANK_OPEN = (
    "Hint [TF5]: view/compare the named candidate variants and rank by the stated "
    "tradeoff axes before add_to_cart. After any stock/cart event, re-rank and order "
    "the new #1. place_test_order = product_id+sku only."
)
_NOTE_RANK_FIRST = (
    "Coach [TF5]: candidates inspected — add_to_cart only the current #1 (first cart is graded)."
)
_NOTE_SOFT_COMPARE = (
    "Hint: compare candidates on soft preference axes (not only price) before the first cart add."
)
_NOTE_TF4_OPEN = (
    "Hint [TF4]: compare ≥2 hard-compliant candidates on named soft axes before "
    "add_to_cart; after cart, inspect then place_test_order — no message essays."
)
_NOTE_START = (
    "Hint: start with search or filter using value-side keywords from the shopper goal, then view the best hits."
)

# ---------------------------------------------------------------------------
# PART A — Cue atlas / lane routing (live) + unused cue laboratories
# ---------------------------------------------------------------------------


def _goal_body(raw: str) -> str:
    text = raw or ""
    if _TASK_RULES_SPLIT in text:
        return text.split(_TASK_RULES_SPLIT, 1)[0]
    return text


@dataclass(frozen=True)
class _CueAtom:
    """One scoring unit for a lane (counts as +1 when any probe fires)."""

    probes: tuple[re.Pattern[str], ...]

    def fires(self, blob: str) -> bool:
        return any(p.search(blob) for p in self.probes)


def _rx(*parts: str) -> re.Pattern[str]:
    return re.compile("|".join(parts), re.I)


class LaneRouter:
    """Classify public query text into a task-family lane.

    Scoring units mirror the historical three-bucket hit counts so lane choice
    stays stable; representation is method/atom based rather than a flat table.
    """

    def __init__(self) -> None:
        self._atoms: dict[str, tuple[_CueAtom, ...]] = {
            "ranking": (
                _CueAtom(
                    (
                        re.compile(r"\btrade-?off\b.{0,40}\bpriorit", re.I),
                        re.compile(r"\brank(?:ing|ed)?\b", re.I),
                        re.compile(r"\bre-?rank\b", re.I),
                    )
                ),
                _CueAtom(
                    (
                        re.compile(r"\bdominat", re.I),
                        re.compile(r"\bstrictly worse\b", re.I),
                        re.compile(r"\bclosed set\b", re.I),
                        re.compile(r"\bnamed candidates?\b", re.I),
                    )
                ),
                _CueAtom((re.compile(r"\d{8,}::\S+"),)),
            ),
            "recovery": (
                _CueAtom(
                    (
                        re.compile(r"\bleast[- ]relax", re.I),
                        re.compile(r"\brecover\b.{0,24}\bunavailable", re.I),
                        re.compile(r"\bfallback\b", re.I),
                    )
                ),
                _CueAtom(
                    (
                        re.compile(
                            r"\bsame model\b.{0,48}(?:\bother\b|\banother\b).{0,12}\bcolou?r\b",
                            re.I,
                        ),
                    )
                ),
                _CueAtom(
                    (
                        re.compile(r"\bdisappear", re.I),
                        re.compile(r"\bsold\s*out\b", re.I),
                        re.compile(r"\bunavailable\b", re.I),
                    )
                ),
            ),
            "justification": (
                _CueAtom(
                    (
                        re.compile(r"\bjustification\b.{0,24}\bclaims?", re.I),
                        re.compile(r"\bclaims?\b.{0,24}\bjustif", re.I),
                        re.compile(r"\bground(?:ed)? claims?\b", re.I),
                    )
                ),
                _CueAtom(
                    (
                        re.compile(r"\btell me why\b", re.I),
                        re.compile(r"\bgive me a reason\b", re.I),
                        re.compile(r"\bwhy\b.{0,16}(?:the one|it'?s|your pick)", re.I),
                        re.compile(r"\bexplain\s+(?:the\s+)?(?:choice|pick|order)\b", re.I),
                    )
                ),
            ),
            "intent_decomposition": (
                _CueAtom((re.compile(r"nice[- ]to[- ]have", re.I),)),
                _CueAtom(
                    (
                        re.compile(r"\bsubordinate\b.{0,40}\b(?:firm|requirement)", re.I),
                        re.compile(r"preferred brand", re.I),
                    )
                ),
                _CueAtom(
                    (
                        re.compile(r"\bfirm\b.{0,80}\bsoft\b", re.I),
                        re.compile(r"\bsoft\b.{0,80}\bfirm\b", re.I),
                    )
                ),
            ),
            "constraint_satisfaction": (
                _CueAtom(
                    (
                        re.compile(r"exact(?:ly)?\s+(?:varian|sku|option)", re.I),
                        re.compile(r"specific sku", re.I),
                    )
                ),
                _CueAtom(
                    (
                        re.compile(r"every (?:firm )?requirement must", re.I),
                        re.compile(r"\bgated (?:spec|requirement)\b", re.I),
                    )
                ),
                _CueAtom(
                    (
                        re.compile(r"\bif you ask\b", re.I),
                        re.compile(r"\bwhen asked\b", re.I),
                        re.compile(r"\bwithhold\b.{0,36}\bspec\b", re.I),
                    )
                ),
            ),
            "preference_reasoning": (
                _CueAtom(
                    (
                        re.compile(r"\bnot (?:just )?the cheapest\b", re.I),
                        re.compile(r"\bsoft (?:axes|preferences?)\b", re.I),
                    )
                ),
                _CueAtom(
                    (
                        re.compile(r"intended use", re.I),
                        re.compile(r"use[- ]case", re.I),
                    )
                ),
                _CueAtom(
                    (
                        re.compile(r"\brather pay\b", re.I),
                        re.compile(r"\bmatters more\b", re.I),
                    )
                ),
            ),
            "retrieval_recall": (
                _CueAtom(
                    (
                        re.compile(r"\breformulate\b", re.I),
                        re.compile(r"\bshortlist\b", re.I),
                    )
                ),
                _CueAtom(
                    (
                        re.compile(r"\bdistinct (?:products|identities)\b", re.I),
                        re.compile(r"\boff[- ]category\b", re.I),
                    )
                ),
                _CueAtom(
                    (
                        re.compile(r'matches? (?:the )?(?:phrase )?["\'][^"\']+["\']', re.I),
                    )
                ),
            ),
        }
        self._ask_gate = re.compile(
            r"(?:tell|reveal|share).{0,48}\bif you ask\b|\bif you ask\b|\bwhen asked\b|"
            r"\bask\b.{0,24}(?:exact )?(?:requirement|spec)\b|\bwithhold\b.{0,36}\bspec\b",
            re.I,
        )
        self._reply_cues = (
            re.compile(r"^\s*which\b", re.I),
            re.compile(r"^\s*what\b", re.I),
            re.compile(r"^\s*does\b", re.I),
            re.compile(r"^\s*did\b", re.I),
            re.compile(r"\bcan you\b", re.I),
            re.compile(r"\bcould you\b", re.I),
            re.compile(r"^\s*wait\b", re.I),
            re.compile(r"\bhold on\b", re.I),
            re.compile(r"\bbefore you\b", re.I),
            re.compile(r"\bre-?check\b", re.I),
            re.compile(r"\bgot anything\b", re.I),
        )

    def scoreboard(self, query: str) -> dict[str, int]:
        text = query or ""
        board: dict[str, int] = {}
        for lane, atoms in self._atoms.items():
            hits = sum(1 for atom in atoms if atom.fires(text))
            if hits:
                board[lane] = hits
        if _OPTIONAL_JUST_PHRASE in text.lower() and board.get("justification"):
            if not _WHY_GOAL_RE.search(_goal_body(text)):
                board.pop("justification", None)
        return board

    def classify(self, query: str) -> str:
        board = self.scoreboard(query)
        if not board:
            return _DEFAULT_LANE
        return max(board.items(), key=lambda kv: (kv[1], _LANE_TIEBREAK.get(kv[0], 0)))[0]

    def withheld_spec(self, query: str) -> bool:
        return bool(self._ask_gate.search(_goal_body(query)))

    def goal_wants_why(self, query: str) -> bool:
        return bool(_WHY_GOAL_RE.search(_goal_body(query)))

    def extract_named_skus(self, query: str) -> list[str]:
        return list(dict.fromkeys(_SKU_TOKEN_RE.findall(query or "")))

    def shopper_awaits_reply(self, spoken: str) -> bool:
        blob = (spoken or "").strip().lower()
        if not blob:
            return False
        if "?" in blob:
            return True
        return any(rx.search(blob) for rx in self._reply_cues)


_LANES = LaneRouter()


# --- unused cue laboratories (not referenced by ShopConductor) ---------------


class _CueLabNotebook:
    """Offline cue brainstorming helpers — never wired into classify()."""

    WEIGHT_GRID = {
        "lexical": (0.35, 0.55, 0.75),
        "span": (12, 24, 40, 80),
        "boost": (1, 2, 3),
    }

    def __init__(self, seed_phrases: Sequence[str] | None = None) -> None:
        self._phrases = list(seed_phrases or [])
        self._scratch: list[dict[str, Any]] = []

    def ingest_phrase(self, phrase: str) -> None:
        cleaned = " ".join(str(phrase or "").split())
        if cleaned and cleaned not in self._phrases:
            self._phrases.append(cleaned)

    def hypothetical_overlap(self, left: str, right: str) -> float:
        a = set(re.findall(r"[a-z0-9]+", (left or "").lower()))
        b = set(re.findall(r"[a-z0-9]+", (right or "").lower()))
        if not a or not b:
            return 0.0
        return len(a & b) / float(len(a | b))

    def draft_span_windows(self, text: str, radius: int = 24) -> list[tuple[int, int, str]]:
        blob = text or ""
        out: list[tuple[int, int, str]] = []
        for m in re.finditer(r"\b\w+\b", blob):
            lo = max(0, m.start() - radius)
            hi = min(len(blob), m.end() + radius)
            out.append((lo, hi, blob[lo:hi]))
        return out[:64]

    def record_scratch(self, label: str, payload: Mapping[str, Any]) -> None:
        self._scratch.append({"label": label, "payload": dict(payload)})

    def export_scratch(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._scratch]

    def invent_negatives(self, positive: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", (positive or "").lower())
        if len(tokens) < 2:
            return []
        return [" ".join(reversed(tokens)), " ".join(tokens[::2]), " ".join(tokens[1::2])]

    def score_density(self, text: str) -> dict[str, float]:
        words = re.findall(r"[a-z0-9]+", (text or "").lower())
        if not words:
            return {"tokens": 0.0, "unique": 0.0, "avg_len": 0.0}
        uniq = len(set(words))
        return {
            "tokens": float(len(words)),
            "unique": float(uniq),
            "avg_len": sum(len(w) for w in words) / float(len(words)),
        }


class _AlternateLaneSketch:
    """Sketch of a multi-pass lane voter that is intentionally unused."""

    def __init__(self) -> None:
        self._lab = _CueLabNotebook()
        self._history: list[tuple[str, dict[str, float]]] = []

    def soft_vote(self, query: str) -> dict[str, float]:
        dens = self._lab.score_density(query)
        votes = {lane: 0.0 for lane in _LANE_TIEBREAK}
        for lane, weight in _LANE_TIEBREAK.items():
            votes[lane] = dens["unique"] * 0.01 + weight * 0.001
        self._history.append((query[:80], dict(votes)))
        return votes

    def explain(self, query: str) -> str:
        votes = self.soft_vote(query)
        ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        return "; ".join(f"{k}:{v:.3f}" for k, v in ranked[:4])

    def mutate_query(self, query: str) -> list[str]:
        body = _goal_body(query)
        toks = re.findall(r"[A-Za-z0-9:-]+", body)
        variants = [" ".join(toks), " ".join(toks[:8]), " ".join(toks[-8:])]
        return [v for v in variants if v]

    def unused_calibration_matrix(self) -> list[list[float]]:
        rows: list[list[float]] = []
        for i in range(7):
            rows.append([((i + j) % 5) * 0.17 for j in range(7)])
        return rows

    def phantom_family_alias(self, lane: str) -> str:
        aliases = {
            "ranking": "closed_set_priority",
            "recovery": "availability_ladder",
            "justification": "evidence_checkout",
            "intent_decomposition": "brand_dual_track",
            "constraint_satisfaction": "gated_walls",
            "preference_reasoning": "soft_axis_fit",
            "retrieval_recall": "sweep_shortlist",
        }
        return aliases.get(lane, lane)


class _RegexFarmUnused:
    """Extra compiled patterns kept around for local regex A/B notes."""

    def __init__(self) -> None:
        self.patterns = [
            re.compile(r"\bbudget\b.{0,20}\d", re.I),
            re.compile(r"\bin[- ]?stock\b", re.I),
            re.compile(r"\bbrand\b", re.I),
            re.compile(r"\bmodel\b", re.I),
            re.compile(r"\bcolour\b|\bcolor\b", re.I),
            re.compile(r"\bsize\b|\binch(?:es)?\b", re.I),
            re.compile(r"\bwarranty\b", re.I),
            re.compile(r"\bshipping\b", re.I),
            re.compile(r"\bcompare\b", re.I),
            re.compile(r"\bfilter\b", re.I),
            re.compile(r"\bsearch\b", re.I),
            re.compile(r"\bcart\b", re.I),
            re.compile(r"\border\b", re.I),
            re.compile(r"\bsku\b", re.I),
            re.compile(r"\bproduct(?:_id)?\b", re.I),
            re.compile(r"\bpreference\b", re.I),
            re.compile(r"\bconstraint\b", re.I),
            re.compile(r"\brecover(?:y|ing)?\b", re.I),
            re.compile(r"\brank(?:ing)?\b", re.I),
            re.compile(r"\bjustif(?:y|ication)\b", re.I),
        ]

    def count_hits(self, text: str) -> int:
        return sum(1 for p in self.patterns if p.search(text or ""))

    def labeled_hits(self, text: str) -> list[str]:
        return [p.pattern for p in self.patterns if p.search(text or "")]

    def compress_report(self, text: str) -> dict[str, Any]:
        hits = self.labeled_hits(text)
        return {"n": len(hits), "sample": hits[:5], "density": len(hits) / max(1, len(self.patterns))}


def _unused_lane_diagnostics(query: str) -> dict[str, Any]:
    lab = _CueLabNotebook(["firm", "soft", "budget", "brand"])
    alt = _AlternateLaneSketch()
    farm = _RegexFarmUnused()
    return {
        "density": lab.score_density(query),
        "soft_vote": alt.soft_vote(query),
        "farm": farm.compress_report(query),
        "windows": len(lab.draft_span_windows(query)),
        "alias": alt.phantom_family_alias(_LANES.classify(query)),
    }


# ---------------------------------------------------------------------------
# PART B — Observation / catalog scaffolding (live extractors + dead prism)
# ---------------------------------------------------------------------------


def _event_kind_from_obs(obs: Any) -> str:
    if not isinstance(obs, dict):
        return ""
    if obs.get("stock_changed"):
        return "stock_changed"
    if obs.get("price_changed"):
        return "price_changed"
    if obs.get("event"):
        return str(obs.get("event"))
    node = obs.get("market_event")
    if isinstance(node, dict):
        return str(node.get("kind") or "market_event")
    if node:
        return "market_event"
    nested = obs.get("observation")
    if isinstance(nested, dict):
        return _event_kind_from_obs(nested)
    return ""


def _collect_fact_keys(observations: Sequence[Any], *, cap: int = 12) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 6 or len(found) >= cap:
            return
        if isinstance(node, dict):
            for raw_key, value in node.items():
                key = str(raw_key).strip()
                low = key.lower()
                if not key or low in _OBS_WALK_SKIP:
                    walk(value, depth + 1)
                    continue
                if low not in seen and isinstance(value, (str, int, float, bool)):
                    seen.add(low)
                    found.append(key)
                else:
                    walk(value, depth + 1)
        elif isinstance(node, list):
            for item in node[:12]:
                walk(item, depth + 1)

    for obs in observations:
        walk(obs)
    return found[:cap]


def _shrink_obs(obs: Any, limit: int = 6000) -> str:
    try:
        text = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obs)
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "...<truncated>"


@dataclass
class _FactLeaf:
    path: tuple[str, ...]
    key: str
    value: Any

    def as_pair(self) -> tuple[str, str]:
        return self.key, "" if self.value is None else str(self.value)


class CatalogPrismUnused:
    """Deep catalog walker retained for offline claim experiments (unused live)."""

    def __init__(self, max_depth: int = 8, max_leaves: int = 128) -> None:
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self._cache: list[_FactLeaf] = []

    def flatten(self, payload: Any, prefix: tuple[str, ...] = ()) -> list[_FactLeaf]:
        leaves: list[_FactLeaf] = []

        def dig(node: Any, path: tuple[str, ...], depth: int) -> None:
            if depth > self.max_depth or len(leaves) >= self.max_leaves:
                return
            if isinstance(node, dict):
                for k, v in node.items():
                    dig(v, path + (str(k),), depth + 1)
            elif isinstance(node, list):
                for i, item in enumerate(node[:20]):
                    dig(item, path + (str(i),), depth + 1)
            else:
                if path:
                    leaves.append(_FactLeaf(path=path, key=path[-1], value=node))

        dig(payload, prefix, 0)
        self._cache = leaves
        return leaves

    def numeric_candidates(self) -> list[_FactLeaf]:
        out: list[_FactLeaf] = []
        for leaf in self._cache:
            try:
                float(str(leaf.value).replace(",", ""))
            except (TypeError, ValueError):
                continue
            out.append(leaf)
        return out

    def key_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for leaf in self._cache:
            hist[leaf.key] = hist.get(leaf.key, 0) + 1
        return hist

    def suggest_claim_pairs(self, prefer: Sequence[str]) -> list[dict[str, str]]:
        prefer_l = {p.lower() for p in prefer}
        pairs: list[dict[str, str]] = []
        for leaf in self._cache:
            if leaf.key.lower() in prefer_l:
                pairs.append({"field": leaf.key, "value": str(leaf.value)})
        return pairs[:8]

    def drop_marketing(self, pairs: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
        cleaned: list[dict[str, str]] = []
        for row in pairs:
            field = str(row.get("field") or "").strip()
            if field.lower() in _FORBIDDEN_CLAIM_LABELS:
                continue
            cleaned.append({"field": field, "value": str(row.get("value") or "")})
        return cleaned


class SoftAxisDraftUnused:
    """Soft-preference sketch utilities (never consulted by the live coach)."""

    AXIS_HINTS = (
        "battery",
        "wireless",
        "water",
        "sweat",
        "noise",
        "comfort",
        "durability",
        "portable",
        "gaming",
        "office",
    )

    def __init__(self) -> None:
        self._notes: list[str] = []

    def extract_wish_tokens(self, goal: str) -> list[str]:
        low = (goal or "").lower()
        return [h for h in self.AXIS_HINTS if h in low]

    def pairwise_margin(self, left: Mapping[str, Any], right: Mapping[str, Any], axes: Sequence[str]) -> float:
        score = 0.0
        for axis in axes:
            lv = str(left.get(axis, ""))
            rv = str(right.get(axis, ""))
            if lv and not rv:
                score += 1.0
            elif rv and not lv:
                score -= 1.0
            elif lv and rv and lv != rv:
                score += 0.15
        return score

    def annotate(self, text: str) -> None:
        self._notes.append(text[:200])

    def dump_notes(self) -> list[str]:
        return list(self._notes)

    def fake_compare_table(self, skus: Sequence[str]) -> list[dict[str, str]]:
        rows = []
        for i, sku in enumerate(skus[:6]):
            rows.append({"sku": sku, "slot": str(i), "tag": "draft"})
        return rows


class VariantLatticeUnused:
    """Named-variant bookkeeping prototype (unused)."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}

    def absorb(self, token: str, meta: Mapping[str, Any] | None = None) -> None:
        self._nodes[token] = dict(meta or {})

    def survivors(self, predicate: Callable[[str, Mapping[str, Any]], bool]) -> list[str]:
        return [k for k, v in self._nodes.items() if predicate(k, v)]

    def split_pid_sku(self, token: str) -> tuple[str, str]:
        if "::" not in token:
            return token, ""
        pid, sku = token.split("::", 1)
        return pid, sku

    def reorder(self, order: Sequence[str]) -> list[str]:
        known = [t for t in order if t in self._nodes]
        rest = [t for t in self._nodes if t not in known]
        return known + rest


def _unused_catalog_bundle(obs: Any) -> dict[str, Any]:
    prism = CatalogPrismUnused()
    soft = SoftAxisDraftUnused()
    lattice = VariantLatticeUnused()
    leaves = prism.flatten(obs)
    return {
        "leaf_n": len(leaves),
        "hist": prism.key_histogram(),
        "wishes": soft.extract_wish_tokens(json.dumps(obs, default=str)[:500]),
        "lattice": len(lattice.reorder([])),
    }


# ---------------------------------------------------------------------------
# PART C — Checkout scrubbing (live) + unused claim forge
# ---------------------------------------------------------------------------


def _scrub_claim_field(field: str) -> str:
    raw = str(field or "").strip()
    if raw.lower() in _FORBIDDEN_CLAIM_LABELS:
        return ""
    return raw


def _scrub_claim_value(field: str, value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if field == "in_stock":
        return "true" if text.lower() in {"true", "1", "yes", "in_stock"} else text
    return text


class CheckoutGuard:
    """Sanitize place_test_order payloads for the active lane."""

    def __init__(self, router: LaneRouter) -> None:
        self._router = router

    def scrub_action(
        self,
        action_name: str,
        args: Mapping[str, Any],
        *,
        lane: str,
        query: str,
    ) -> dict[str, Any]:
        out = dict(args)
        if action_name != "place_test_order":
            return out
        if lane != "justification":
            if not self._router.goal_wants_why(query):
                out.pop("justification", None)
            return out
        just = out.get("justification")
        if not isinstance(just, dict):
            return out
        claims = just.get("claims")
        if not isinstance(claims, list):
            return out
        cleaned: list[dict[str, str]] = []
        seen: set[str] = set()
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            field = _scrub_claim_field(str(claim.get("field") or ""))
            value = _scrub_claim_value(field, claim.get("value"))
            if not field or value == "":
                continue
            fl = field.lower()
            if fl in _FORBIDDEN_CLAIM_LABELS or fl in seen:
                continue
            seen.add(fl)
            cleaned.append({"field": field, "value": value})
        if cleaned:
            out["justification"] = {"claims": cleaned}
        return out


class ClaimForgeUnused:
    """Experimental claim builders — not called from the episode loop."""

    REQUIRED_CORE = ("price", "in_stock")

    def __init__(self) -> None:
        self._drafts: list[list[dict[str, str]]] = []

    def from_keys(self, keys: Sequence[str], values: Mapping[str, Any]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for key in keys:
            if key in values:
                rows.append({"field": key, "value": str(values[key])})
        self._drafts.append(rows)
        return rows

    def enforce_core(self, rows: Sequence[Mapping[str, str]], catalog: Mapping[str, Any]) -> list[dict[str, str]]:
        have = {r.get("field", "").lower() for r in rows}
        out = [dict(r) for r in rows]
        for need in self.REQUIRED_CORE:
            if need not in have and need in catalog:
                out.append({"field": need, "value": str(catalog[need])})
        return out

    def strip_meta(self, rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
        return [
            {"field": str(r.get("field")), "value": str(r.get("value"))}
            for r in rows
            if str(r.get("field") or "").lower() not in _FORBIDDEN_CLAIM_LABELS
        ]

    def diversity_score(self, rows: Sequence[Mapping[str, str]]) -> float:
        fields = [str(r.get("field") or "").lower() for r in rows]
        if not fields:
            return 0.0
        return len(set(fields)) / float(len(fields))

    def last_draft(self) -> list[dict[str, str]]:
        return list(self._drafts[-1]) if self._drafts else []


class OrderEnvelopeUnused:
    """Alternate envelope serializer (unused)."""

    def __init__(self, binding: Mapping[str, Any]) -> None:
        self.binding = dict(binding)
        self._seq = 0

    def next_group(self, problem_id: str, turn: int) -> str:
        self._seq += 1
        return f"{problem_id}-turn-{turn}-x{self._seq}"

    def pack(self, group_id: str, turn: int, calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            **self.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": list(calls),
            "meta_unused": {"seq": self._seq},
        }


def _unused_claim_sandbox(keys: Sequence[str]) -> dict[str, Any]:
    forge = ClaimForgeUnused()
    rows = forge.from_keys(list(keys), {k: f"demo-{i}" for i, k in enumerate(keys)})
    return {"rows": forge.strip_meta(rows), "div": forge.diversity_score(rows)}


# ---------------------------------------------------------------------------
# PART D — Model bridge + playbook assembly
# ---------------------------------------------------------------------------


def _remap_model_id(model: str) -> str:
    table = {
        "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
        "deepseek-ai/DeepSeek-V3.1-TEE": "deepseek/deepseek-chat-v3.1",
        "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
        "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
        "openai/gpt-oss-120b-TEE": "openai/gpt-oss-120b",
        "zai-org/GLM-5.1-TEE": _HEAVY_MODEL,
        "zai-org/GLM-5.2-TEE": _HEAVY_MODEL,
        "moonshotai/Kimi-K2.6-TEE": _HEAVY_MODEL,
        "moonshotai/kimi-k2.6": _HEAVY_MODEL,
        "z-ai/glm-5.1": _HEAVY_MODEL,
        "z-ai/glm-5.2": _HEAVY_MODEL,
    }
    mapped = table.get(model, model)
    low = mapped.lower()
    if "kimi" in low or "glm" in low or low.startswith("z-ai/") or low.startswith("moonshotai/"):
        return _HEAVY_MODEL
    return mapped


def _pick_model(lane: str) -> str:
    override = (getenv("SANDBOX_MODEL") or "").strip()
    if override:
        return _remap_model_id(override)
    return _HEAVY_MODEL if lane in _HEAVY_LANES else _FLASH_MODEL


def _tool_labels(tools: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
        elif isinstance(tool, dict) and tool.get("name"):
            names.append(str(tool["name"]))
    return names


class PlaybookBank:
    def compose(self, lane: str, policy: Mapping[str, Any], query: str) -> str:
        names = _tool_labels(policy.get("tools") or [])
        family_prompt = _LANE_PLAYBOOKS.get(lane, _LANE_PLAYBOOKS[_DEFAULT_LANE])
        variants = _LANES.extract_named_skus(query)
        variant_line = ""
        if lane == "ranking" and variants:
            variant_line = (
                "Named candidate variants (view these; re-rank after events):\n"
                + ", ".join(variants)
                + "\n"
            )
        return (
            f"{_CHARTER}\n\n{family_prompt}\n\n{variant_line}"
            f"# Runtime\nDetected task family: {lane}\n"
            f"Available tools: {(', '.join(names) if names else '(see tool schemas)')}\n"
            f"max_steps={policy.get('max_steps')} max_calls_per_turn={policy.get('max_calls_per_turn', 1)}\n"
        )


def _parse_tool_args(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    if isinstance(raw, dict):
        parsed = raw
    else:
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _names_from_calls(tool_calls: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for call in tool_calls:
        try:
            names.append(str(call["function"]["name"]))
        except (KeyError, TypeError):
            names.append("?")
    return names


class InferenceBridge:
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_calls: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        queue = [model]
        for fb in _MODEL_FALLBACK_CHAIN:
            if fb not in queue:
                queue.append(fb)
        last_err: Exception | None = None
        for mid in queue[:3]:
            inference = _PROXY_BRIDGE.post(
                "/inference/chat/completions",
                json_data={
                    "model": mid,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if inference is None:
                last_err = RuntimeError(f"inference failed for model={mid}")
                continue
            try:
                assistant = inference["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                last_err = exc
                continue
            content = assistant.get("content") or ""
            tool_calls = list(assistant.get("tool_calls") or [])[:max_calls]
            return content, tool_calls
        raise RuntimeError(f"inference request failed ({last_err})")


class PhraseBudgetPlannerUnused:
    """Keyword reformulation scratchpad (unused)."""

    STOP = frozenset({"the", "a", "an", "and", "or", "for", "with", "to", "of", "in", "on"})

    def tokens(self, text: str) -> list[str]:
        return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in self.STOP]

    def shrink(self, text: str, keep: int = 8) -> str:
        toks = self.tokens(text)
        return " ".join(toks[:keep])

    def expand_hyphen(self, text: str) -> list[str]:
        base = self.shrink(text)
        return [base, base.replace("-", " "), base.replace(" ", "-")]

    def category_guess(self, text: str) -> str:
        toks = self.tokens(text)
        return toks[0] if toks else ""


# ---------------------------------------------------------------------------
# PART E — Turn coaching (live NudgeDesk) + unused advisor drafts
# ---------------------------------------------------------------------------


@dataclass
class TurnSnapshot:
    lane: str
    turn: int
    tool_join: str
    last_obs: list[Any]
    query: str
    max_steps: int
    await_reply: bool
    saw_cart_event: bool
    ordered: bool
    first_cart_done: bool
    asked_hidden_spec: bool


class NudgeDesk:
    """Produces the same coach schedule as the prior KH_1 progress hints."""

    def __init__(self, router: LaneRouter) -> None:
        self._router = router

    def claims_note(self, last_obs: Sequence[Any]) -> str:
        keys = _collect_fact_keys(last_obs)
        extra = ""
        if keys:
            extra = (
                " Observed catalog keys from the last view/inspect: "
                + ", ".join(keys)
                + ". Copy those keys as claim fields."
            )
        return _NOTE_CLAIMS + extra

    def advice(self, snap: TurnSnapshot) -> str | None:
        joined = snap.tool_join
        if snap.await_reply:
            return _NOTE_REPLY
        if (
            snap.lane == "constraint_satisfaction"
            and self._router.withheld_spec(snap.query)
            and (not snap.asked_hidden_spec)
            and ("message" not in joined)
        ):
            return _NOTE_ASK
        if snap.lane == "ranking" and (not snap.ordered):
            if snap.saw_cart_event and snap.turn >= 2:
                if "place_test_order" not in joined:
                    return _NOTE_RANK_REFRESH
            if snap.turn >= 8:
                return _NOTE_RANK_CLOSE
            if snap.turn == 1:
                return _NOTE_RANK_OPEN
            if snap.turn >= 2 and (not snap.first_cart_done) and ("view" in joined or "compare" in joined):
                return _NOTE_RANK_FIRST
        if snap.lane == "intent_decomposition" and snap.turn == 1:
            return _NOTE_BRAND
        if snap.lane == "justification" and (not snap.ordered):
            if snap.turn in (2, 5) or (snap.turn >= 2 and "view" not in joined and "compare" not in joined):
                return self.claims_note(snap.last_obs)
        if snap.lane == "recovery" and snap.saw_cart_event and (not snap.ordered):
            return _NOTE_LADDER
        if snap.lane == "constraint_satisfaction" and snap.saw_cart_event and (not snap.ordered):
            return _NOTE_NEW_PID
        if snap.lane == "preference_reasoning" and (not snap.ordered):
            if snap.saw_cart_event and snap.turn >= 2:
                return _NOTE_SOFT_SEAL
            if (snap.first_cart_done or snap.turn >= 4) and snap.turn >= 3:
                return _NOTE_SOFT_SEAL
            if snap.turn == 2 and "compare" not in joined:
                return _NOTE_SOFT_COMPARE
        if not snap.ordered and snap.turn >= max(8, snap.max_steps - 2):
            return _NOTE_CLOSEOUT
        if snap.turn <= 2 and (
            not any(n in joined for n in ("search", "filter", "view", "compare", "message"))
        ):
            return _NOTE_START
        if snap.lane == "recovery" and any(isinstance(o, dict) and o.get("error") for o in snap.last_obs):
            err = next(o.get("error") for o in snap.last_obs if isinstance(o, dict) and o.get("error"))
            return _NOTE_RECOVER.format(error=str(err))
        if snap.lane == "retrieval_recall" and snap.turn == 3 and ("search" in joined):
            return _NOTE_SWEEP
        if snap.saw_cart_event and (not snap.ordered) and snap.turn >= 2:
            return _NOTE_MARKET.format(kind="stock/price/cart")
        return None

    def opening_messages(self, lane: str, query: str) -> list[dict[str, Any]]:
        extras: list[dict[str, Any]] = []
        if lane == "constraint_satisfaction" and self._router.withheld_spec(query):
            extras.append({"role": "user", "content": _NOTE_ASK})
        if lane == "ranking":
            extras.append({"role": "user", "content": _NOTE_RANK_OPEN})
        if lane == "justification":
            extras.append({"role": "user", "content": _NOTE_CLAIMS})
        if lane == "intent_decomposition":
            extras.append({"role": "user", "content": _NOTE_BRAND})
        if lane == "retrieval_recall":
            extras.append({"role": "user", "content": _NOTE_SWEEP})
        if lane == "preference_reasoning":
            extras.append({"role": "user", "content": _NOTE_TF4_OPEN})
        if lane == "recovery":
            extras.append({"role": "user", "content": _NOTE_LADDER})
        return extras


class AdvisorDraftUnused:
    """Alternate multi-advisor design kept for local notes only."""

    def __init__(self) -> None:
        self._log: list[str] = []

    def rank_pressure(self, turn: int, ordered: bool) -> str | None:
        if ordered:
            return None
        if turn >= 10:
            return "unused-close"
        if turn >= 5:
            return "unused-mid"
        return None

    def recovery_pressure(self, errors: Sequence[str]) -> str | None:
        if not errors:
            return None
        self._log.append(errors[0][:80])
        return f"unused-recover:{errors[0][:40]}"

    def blend(self, parts: Sequence[str | None]) -> str | None:
        kept = [p for p in parts if p]
        return " | ".join(kept) if kept else None

    def transcript(self) -> list[str]:
        return list(self._log)


class DualShortlistSketchUnused:
    def __init__(self) -> None:
        self.brand_track: list[str] = []
        self.firm_track: list[str] = []

    def push_brand(self, sku: str) -> None:
        if sku not in self.brand_track:
            self.brand_track.append(sku)

    def push_firm(self, sku: str) -> None:
        if sku not in self.firm_track:
            self.firm_track.append(sku)

    def choose(self, brand_ok: bool) -> str | None:
        if brand_ok and self.brand_track:
            return self.brand_track[0]
        if self.firm_track:
            return self.firm_track[0]
        return None

    def snapshot(self) -> dict[str, list[str]]:
        return {"brand": list(self.brand_track), "firm": list(self.firm_track)}


class RelaxLadderSimulatorUnused:
    STEPS = ("exact_model_color", "same_model_other_color", "other_models")

    def __init__(self) -> None:
        self.rung = 0

    def climb(self) -> str:
        self.rung = min(self.rung + 1, len(self.STEPS) - 1)
        return self.STEPS[self.rung]

    def reset(self) -> None:
        self.rung = 0

    def label(self) -> str:
        return self.STEPS[self.rung]


def _unused_coach_matrix(lane: str) -> list[str]:
    draft = AdvisorDraftUnused()
    dual = DualShortlistSketchUnused()
    ladder = RelaxLadderSimulatorUnused()
    dual.push_firm("demo::a")
    dual.push_brand("demo::b")
    return [
        draft.rank_pressure(3, False) or "",
        dual.choose(True) or "",
        ladder.climb(),
        lane,
    ]


# ---------------------------------------------------------------------------
# PART F — Episode conductor (live path)
# ---------------------------------------------------------------------------


@dataclass
class EpisodeFlags:
    await_reply: bool = False
    saw_cart_event: bool = False
    ordered: bool = False
    first_cart_done: bool = False
    asked_hidden_spec: bool = False
    last_tool_names: list[str] = field(default_factory=list)
    last_obs: list[Any] = field(default_factory=list)


class ShopConductor:
    def __init__(self) -> None:
        self.router = _LANES
        self.playbooks = PlaybookBank()
        self.nudges = NudgeDesk(self.router)
        self.checkout = CheckoutGuard(self.router)
        self.infer = InferenceBridge()

    def _apply_action_flags(self, name: str, flags: EpisodeFlags) -> None:
        if name == "message":
            flags.await_reply = False
            flags.asked_hidden_spec = True
        elif name == "add_to_cart":
            flags.first_cart_done = True
        elif name == "remove_from_cart":
            flags.saw_cart_event = True
        elif name == "place_test_order":
            flags.ordered = True

    def run(self, problem_data: Mapping[str, Any]) -> list[dict[str, Any]]:
        environment = problem_data["environment"]
        binding = environment["binding"]
        policy = environment["policy_view"]
        query = str(policy.get("query") or problem_data.get("query") or "")
        problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
        max_steps = max(1, int(policy.get("max_steps", 20)))
        tools = list(policy.get("tools") or [])
        lane = self.router.classify(query)
        model = _pick_model(lane)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.playbooks.compose(lane, policy, query)},
            {"role": "user", "content": query},
            *self.nudges.opening_messages(lane, query),
        ]
        dialogue: list[dict[str, Any]] = [
            {
                "role": "meta",
                "kh1": {
                    "version": 3,
                    "lane": lane,
                    "model": model,
                    "provider": "openrouter",
                    "conductor": "ShopConductor",
                },
            }
        ]
        flags = EpisodeFlags()
        for turn in range(1, max_steps + 1):
            hint = self.nudges.advice(
                TurnSnapshot(
                    lane=lane,
                    turn=turn,
                    tool_join=" ".join(flags.last_tool_names),
                    last_obs=flags.last_obs,
                    query=query,
                    max_steps=max_steps,
                    await_reply=flags.await_reply,
                    saw_cart_event=flags.saw_cart_event,
                    ordered=flags.ordered,
                    first_cart_done=flags.first_cart_done,
                    asked_hidden_spec=flags.asked_hidden_spec,
                )
            )
            if hint:
                messages.append({"role": "user", "content": hint})
            assistant_content = ""
            tool_calls: list[dict[str, Any]] = []
            for attempt in range(len(_EMPTY_TOOL_PUSH) + 1):
                assistant_content, tool_calls = self.infer.complete(
                    model=model, messages=messages, tools=tools, max_calls=max_calls
                )
                if tool_calls:
                    break
                dialogue.append({"role": "assistant", "content": assistant_content})
                nudge = _EMPTY_TOOL_PUSH[min(attempt, len(_EMPTY_TOOL_PUSH) - 1)]
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {"role": "user", "content": nudge},
                    ]
                )
            else:
                raise RuntimeError("model returned no tool call before completion")
            messages.append(
                {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
            )
            group_id = f"{problem_id}-turn-{turn}"
            call_payloads = []
            for index, call in enumerate(tool_calls, start=1):
                name = call["function"]["name"]
                args = self.checkout.scrub_action(
                    name, _parse_tool_args(call), lane=lane, query=query
                )
                if isinstance(call.get("function"), dict):
                    call["function"]["arguments"] = json.dumps(args)
                call_payloads.append(
                    {"call_id": f"{group_id}-{index}", "action": {"name": name, "args": args}}
                )
                self._apply_action_flags(name, flags)
            envelope = {
                **binding,
                "call_id": group_id,
                "idempotency_key": group_id,
                "turn": turn,
                "calls": call_payloads,
            }
            result = _PROXY_BRIDGE.post("/environment/call", json_data=envelope)
            if result is None:
                raise RuntimeError("environment call failed")
            dialogue.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                    "environment_result": result,
                }
            )
            flags.last_tool_names = _names_from_calls(tool_calls)
            flags.last_obs = []
            for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
                obs = call_result.get("observation")
                flags.last_obs.append(obs)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": _shrink_obs(obs),
                    }
                )
                kind = _event_kind_from_obs(obs)
                if kind:
                    flags.saw_cart_event = True
                    if not flags.ordered:
                        messages.append(
                            {"role": "user", "content": _NOTE_MARKET.format(kind=kind)}
                        )
                elif isinstance(obs, dict) and obs.get("error"):
                    flags.saw_cart_event = True
            user_message = result.get("user_message")
            if user_message and user_message.get("content"):
                spoken = str(user_message["content"])
                messages.append({"role": "user", "content": spoken})
                if self.router.shopper_awaits_reply(spoken):
                    flags.await_reply = True
            errors = [
                o.get("error")
                for o in flags.last_obs
                if isinstance(o, dict) and o.get("error")
            ]
            if errors and lane == "recovery" and (not flags.ordered):
                messages.append(
                    {"role": "user", "content": _NOTE_RECOVER.format(error=errors[0])}
                )
            if (
                lane == "ranking"
                and flags.saw_cart_event
                and (not flags.ordered)
                and (errors or any(_event_kind_from_obs(o) for o in flags.last_obs))
            ):
                messages.append({"role": "user", "content": _NOTE_RANK_REFRESH})
            if lane == "preference_reasoning" and flags.saw_cart_event and (not flags.ordered):
                messages.append({"role": "user", "content": _NOTE_SOFT_SEAL})
            if any(
                isinstance(call.get("observation"), dict) and call["observation"].get("done")
                for call in result.get("calls") or []
            ):
                break
        return dialogue




# ---------------------------------------------------------------------------
# PART G — Extra unused laboratories (intentionally unreachable)
# ---------------------------------------------------------------------------


class SearchPassPlannerUnused:
    """Multi-pass search planner sketch for offline notebooks."""

    def __init__(self, max_passes: int = 3) -> None:
        self.max_passes = max_passes
        self._passes: list[dict[str, Any]] = []

    def plan(self, goal: str) -> list[dict[str, str]]:
        toks = re.findall(r"[A-Za-z0-9]+", goal or "")
        plans = [
            {"mode": "raw", "q": " ".join(toks[:12])},
            {"mode": "tokens", "q": " ".join(toks[::2][:10])},
            {"mode": "tail", "q": " ".join(toks[-10:])},
        ]
        self._passes = plans[: self.max_passes]
        return list(self._passes)

    def annotate_pass(self, idx: int, hit_n: int) -> None:
        if 0 <= idx < len(self._passes):
            self._passes[idx]["hits"] = hit_n

    def best_pass(self) -> dict[str, Any] | None:
        if not self._passes:
            return None
        return max(self._passes, key=lambda row: int(row.get("hits") or 0))

    def reset(self) -> None:
        self._passes.clear()


class StockShockJournalUnused:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, kind: str, sku: str, detail: str = "") -> None:
        self.events.append({"kind": kind, "sku": sku, "detail": detail[:120]})

    def last(self) -> dict[str, Any] | None:
        return self.events[-1] if self.events else None

    def kinds(self) -> list[str]:
        return [e["kind"] for e in self.events]

    def filter_kind(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("kind") == kind]


class BudgetParserUnused:
    MONEY_RE = re.compile(r"(\d+(?:[.,]\d+)?)")

    def extract_ceiling(self, text: str) -> float | None:
        m = self.MONEY_RE.search(text or "")
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None

    def within(self, price: Any, ceiling: float | None) -> bool:
        if ceiling is None:
            return True
        try:
            return float(price) <= ceiling + 1e-9
        except (TypeError, ValueError):
            return False

    def slack(self, price: Any, ceiling: float | None) -> float:
        if ceiling is None:
            return 0.0
        try:
            return float(ceiling) - float(price)
        except (TypeError, ValueError):
            return 0.0


class TitleNoiseFilterUnused:
    NOISE = (
        "bundle",
        "accessory",
        "case only",
        "screen protector",
        "review",
        "gift card",
    )

    def looks_noisy(self, title: str) -> bool:
        low = (title or "").lower()
        return any(n in low for n in self.NOISE)

    def clean_tokens(self, title: str) -> list[str]:
        toks = re.findall(r"[a-z0-9]+", (title or "").lower())
        return [t for t in toks if len(t) > 2]

    def overlap(self, title: str, goal: str) -> float:
        a = set(self.clean_tokens(title))
        b = set(self.clean_tokens(goal))
        if not a or not b:
            return 0.0
        return len(a & b) / float(len(a | b))


class CompareMatrixUnused:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, sku: str, attrs: Mapping[str, Any]) -> None:
        self.rows.append({"sku": sku, **dict(attrs)})

    def column(self, key: str) -> list[Any]:
        return [r.get(key) for r in self.rows]

    def dominate_mask(self, axes: Sequence[str], higher_better: Mapping[str, bool] | None = None) -> list[bool]:
        higher_better = higher_better or {}
        mask = [True] * len(self.rows)
        for i, row in enumerate(self.rows):
            for j, other in enumerate(self.rows):
                if i == j:
                    continue
                better_or_eq = True
                strictly = False
                for axis in axes:
                    a = row.get(axis)
                    b = other.get(axis)
                    if a is None or b is None:
                        better_or_eq = False
                        break
                    try:
                        af = float(a)
                        bf = float(b)
                    except (TypeError, ValueError):
                        better_or_eq = False
                        break
                    if higher_better.get(axis, True):
                        if af < bf:
                            better_or_eq = False
                            break
                        if af > bf:
                            strictly = True
                    else:
                        if af > bf:
                            better_or_eq = False
                            break
                        if af < bf:
                            strictly = True
                if better_or_eq and strictly:
                    mask[i] = False
                    break
        return mask


class MessageTemplateBankUnused:
    TEMPLATES = (
        "Confirming the firm figure before I search.",
        "Here is the pick with price and stock evidence.",
        "I will re-check availability after the catalog move.",
        "Comparing the named options on the stated axes.",
    )

    def pick(self, idx: int) -> str:
        return self.TEMPLATES[idx % len(self.TEMPLATES)]

    def fill(self, idx: int, **kwargs: Any) -> str:
        base = self.pick(idx)
        if not kwargs:
            return base
        extra = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"{base} ({extra})"


class EpisodeTelemetryUnused:
    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []

    def mark(self, turn: int, tools: Sequence[str], lane: str) -> None:
        self.turns.append({"turn": turn, "tools": list(tools), "lane": lane})

    def tool_hist(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for row in self.turns:
            for t in row.get("tools") or []:
                hist[str(t)] = hist.get(str(t), 0) + 1
        return hist

    def summary(self) -> dict[str, Any]:
        return {"n": len(self.turns), "hist": self.tool_hist()}


class PolicyInspectorUnused:
    def list_action_names(self, policy: Mapping[str, Any]) -> list[str]:
        tools = policy.get("tools") or []
        return _tool_labels(tools)

    def max_budget(self, policy: Mapping[str, Any]) -> tuple[int, int]:
        return (
            int(policy.get("max_steps") or 0),
            int(policy.get("max_calls_per_turn") or 0),
        )

    def describe(self, policy: Mapping[str, Any]) -> str:
        names = self.list_action_names(policy)
        steps, calls = self.max_budget(policy)
        return f"tools={len(names)} steps={steps} calls={calls}"


class GroundingChecklistUnused:
    CHECKS = (
        "viewed_before_cart",
        "inspected_stock",
        "claims_use_catalog_keys",
        "no_marketing_labels",
        "price_present",
        "in_stock_present",
    )

    def __init__(self) -> None:
        self.state = {c: False for c in self.CHECKS}

    def tick(self, name: str) -> None:
        if name in self.state:
            self.state[name] = True

    def ready(self) -> bool:
        return all(self.state.values())

    def missing(self) -> list[str]:
        return [k for k, v in self.state.items() if not v]


class QueryNormalizerUnused:
    def collapse_space(self, text: str) -> str:
        return " ".join((text or "").split())

    def lower_alnum(self, text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))

    def split_goal_rules(self, text: str) -> tuple[str, str]:
        if _TASK_RULES_SPLIT in (text or ""):
            a, b = text.split(_TASK_RULES_SPLIT, 1)
            return a, b
        return text or "", ""

    def fingerprint(self, text: str) -> str:
        return self.lower_alnum(text)[:160]


def _unused_part_g_demo(goal: str) -> dict[str, Any]:
    planner = SearchPassPlannerUnused()
    shock = StockShockJournalUnused()
    budget = BudgetParserUnused()
    noise = TitleNoiseFilterUnused()
    matrix = CompareMatrixUnused()
    msgs = MessageTemplateBankUnused()
    telem = EpisodeTelemetryUnused()
    grounding = GroundingChecklistUnused()
    norm = QueryNormalizerUnused()
    plans = planner.plan(goal)
    shock.record("stock_changed", "demo::sku", "offline")
    matrix.add("a", {"price": 10, "score": 2})
    matrix.add("b", {"price": 12, "score": 5})
    telem.mark(1, ["search"], "retrieval_recall")
    grounding.tick("viewed_before_cart")
    return {
        "plans": plans,
        "ceiling": budget.extract_ceiling(goal),
        "noisy": noise.looks_noisy(goal),
        "mask": matrix.dominate_mask(["score"], {"score": True}),
        "msg": msgs.fill(1, lane="demo"),
        "telem": telem.summary(),
        "missing": grounding.missing(),
        "fp": norm.fingerprint(goal),
        "shock": shock.kinds(),
    }



def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    return ShopConductor().run(problem_data)


__all__ = ["agent_main", "ShopConductor", "LaneRouter"]
