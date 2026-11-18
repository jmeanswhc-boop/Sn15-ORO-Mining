from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------
# Inference engines. OpenRouter ids only.
# https://oroagents.com/docs/miners/agent-interface
# ---------------------------------------------------------------------------

_ENGINE_QUICK = "deepseek/deepseek-v3.2"
_ENGINE_DEEP = "deepseek/deepseek-v3.2"
_ENGINE_RELIEF = (
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "google/gemma-4-31b-it",
)

# Soft wall-clock budget per episode (grader hard-kill is 300s).
_WALL_SOFT_SEC = 180.0
_WALL_HARD_SEC = 270.0

# ---------------------------------------------------------------------------
# Capability lanes
# ---------------------------------------------------------------------------

SPLIT = "split"          # separate binding items from stated leanings
SWEEP = "sweep"          # widen retrieval before committing
LOCK = "lock"            # exact-variant compliance
TASTE = "taste"          # fit-driven choice over price
PODIUM = "podium"        # ordered choice among named candidates
SALVAGE = "salvage"      # minimum concession after the world changes
VOUCH = "vouch"          # purchase plus recorded evidence

_LANES = (SPLIT, SWEEP, LOCK, TASTE, PODIUM, SALVAGE, VOUCH)

# Cue vocabulary. Short, ordinary words weighted by where they appear; the
# shopper's own sentence counts for more than the generic brief that follows.
_GOAL_CUES: dict[str, tuple[tuple[str, int], ...]] = {
    PODIUM: (("tie", 3), ("rank", 3), ("tradeoff", 3), ("highest", 1)),
    SALVAGE: (("disappear", 4), ("unavailable", 3), ("sold out", 3), ("gone", 2)),
    VOUCH: (("why", 4), ("reason", 3), ("justif", 3), ("explain", 3)),
    LOCK: (("if you ask", 4), ("withhold", 3), ("exact", 1)),
    SPLIT: (("ideally", 3), ("prefer", 2), ("firm", 2), ("but", 1)),
    TASTE: (("rather", 3), ("cheapest", 3), ("matters more", 3), ("flimsy", 2)),
    SWEEP: (("matches", 3), ("phrase", 2), ("listing", 1)),
}

_RULES_CUES: dict[str, tuple[tuple[str, int], ...]] = {
    PODIUM: (("tradeoff", 3), ("re-rank", 2), ("strictly worse", 2)),
    SALVAGE: (("relax", 3), ("shade", 2), ("model code", 2)),
    VOUCH: (("claim", 3), ("trustworthy", 2), ("ground", 2)),
    LOCK: (("pixel", 3), ("exact variant", 2), ("thresholds", 2)),
    SPLIT: (("nice-to-have", 3), ("subordinate", 3), ("preferred brand", 2)),
    TASTE: (("axes", 3), ("intended use", 2), ("placeholder", 1)),
    SWEEP: (("reformulate", 3), ("shortlist", 2), ("lucky", 2)),
}

# Ties break toward the more specialised lane.
_LANE_RANK = {PODIUM: 7, SALVAGE: 6, VOUCH: 5, LOCK: 4, SPLIT: 3, TASTE: 2, SWEEP: 1}

# Prefer the quick engine for every lane — deep models burn the 300s budget
# under parallel workers without improving construct success.
_DEEP_LANES = frozenset()

# ---------------------------------------------------------------------------
# Text probes
# ---------------------------------------------------------------------------

_VARIANT_RX = re.compile(r"\b\d{6,}::[\w-]+(?:\.[\w-]+)*")
_CEILING_RX = re.compile(
    r"(?:at or (?:under|below)|under|below|up to|no more than|max(?:imum)?(?:\s+of)?)"
    r"\s*([\d][\d,]*(?:\.\d+)?)",
    re.I,
)
_FIGURE_RX = re.compile(
    r"\b\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|\"|hz|gb|tb|mp|mah|wh)\b"
    r"|\b(?:\d{3,4}\s*p|full\s*hd|qhd|uhd|4k|hd)\b",
    re.I,
)
_PHRASE_RX = re.compile(r"[\"\u201c]([^\"\u201d]{3,60})[\"\u201d]")
_ASKS_REASON_RX = re.compile(r"\bwhy\b|\breason\b|\bjustif|\bexplain\b", re.I)
_HELD_BACK_RX = re.compile(
    r"\bif you ask\b|\bwhen you ask\b|(?:tell|share|reveal)[^.]{0,40}\bif\b[^.]{0,20}\bask",
    re.I,
)

# Catalog attribute vocabulary accepted on an order's claims.
_CLAIM_CANON = {
    "display": "resolution",
    "display resolution": "resolution",
    "pixel class": "resolution",
    "resolution class": "resolution",
    "screen": "screen_size",
    "screen size": "screen_size",
    "size": "screen_size",
    "diagonal": "screen_size",
    "refresh": "refresh_rate",
    "refresh rate": "refresh_rate",
    "memory": "ram",
    "stock": "in_stock",
    "availability": "in_stock",
    "available": "in_stock",
    "cost": "price",
    "maker": "brand",
    "manufacturer": "brand",
}

_CLAIM_SPEC = frozenset({"resolution", "screen_size", "refresh_rate", "ram"})
_CLAIM_CORE = frozenset({"price", "in_stock"})
_CLAIM_SUPPORT = frozenset(
    {"brand", "streaming", "microphone", "privacy_cover", "autofocus", "hdr"}
)
_CLAIM_OK = _CLAIM_SPEC | _CLAIM_CORE | _CLAIM_SUPPORT
# Labels the catalog does not expose as claim fields.
_CLAIM_REJECT = frozenset(
    {
        "category",
        "category_path",
        "device_operating_system",
        "device os",
        "operating system",
        "os",
        "title",
        "name",
        "summary",
        "evidence",
        "reason",
        "note",
        "description",
    }
)

_BARE_FIGURE_RX = re.compile(
    r"(?i)^(?:\d{3,4}\s*p(?:\s*hd)?|full\s*hd|qhd|uhd|4k|hd|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|\")|\d+\s*hz)$"
)


# ---------------------------------------------------------------------------
# Brief parsing
# ---------------------------------------------------------------------------


class TaskShape:
    """Everything the agent can learn from the public brief, read once."""

    __slots__ = (
        "brief",
        "goal",
        "rules",
        "lane",
        "ceiling",
        "variants",
        "figures",
        "phrase",
        "wants_reason",
        "held_back_figure",
    )

    def __init__(self, brief: str) -> None:
        try:
            self.brief = brief or ""
            self.goal, self.rules = _cleave(self.brief)
            self.variants = _dedupe(_VARIANT_RX.findall(self.brief))
            self.figures = _dedupe(m.group(0).strip() for m in _FIGURE_RX.finditer(self.goal))
            self.ceiling = _first_ceiling(self.goal) or _first_ceiling(self.brief)
            found = _PHRASE_RX.search(self.goal)
            self.phrase = found.group(1) if found else ""
            self.wants_reason = bool(_ASKS_REASON_RX.search(self.goal))
            self.held_back_figure = bool(_HELD_BACK_RX.search(self.goal))
            self.lane = self._choose_lane()
        except Exception:
            pass

    def _choose_lane(self) -> str:
        try:
            tally = dict.fromkeys(_LANES, 0)

            goal = self.goal.lower()
            rules = self.rules.lower()
            for lane, cues in _GOAL_CUES.items():
                tally[lane] += sum(weight for token, weight in cues if token in goal)
            for lane, cues in _RULES_CUES.items():
                tally[lane] += sum(weight for token, weight in cues if token in rules)

            # Structural evidence outranks vocabulary.
            if len(self.variants) >= 2:
                tally[PODIUM] += 8
            if self.held_back_figure:
                tally[LOCK] += 6
            if self.wants_reason:
                tally[VOUCH] += 5
            if self.phrase:
                tally[SWEEP] += 4

            # A stated leaning next to non-negotiables is a split, not a taste call.
            if "prefer" in goal and ("firm" in goal or "requirement" in goal):
                tally[SPLIT] += 4

            best = max(tally.values())
            if best <= 0:
                return SWEEP
            finalists = [lane for lane, score in tally.items() if score == best]
            return max(finalists, key=lambda lane: _LANE_RANK[lane])
        except Exception:
            return SWEEP

    def order_may_carry_claims(self) -> bool:
        """Only vouch work (or an explicit request) should attach claims."""
        try:
            return self.lane == VOUCH or self.wants_reason
        except Exception:
            return False

    def fact_sheet(self) -> str:
        try:
            rows = []
            if self.ceiling:
                rows.append(f"- spend ceiling: {self.ceiling}")
            if self.figures:
                rows.append(f"- figures named by the shopper: {', '.join(self.figures[:6])}")
            if self.phrase:
                rows.append(f"- listing text to match: {self.phrase}")
            if self.variants:
                rows.append(f"- candidate variants: {', '.join(self.variants)}")
            if self.held_back_figure:
                rows.append("- one requirement is withheld until you ask for it")
            if not rows:
                return ""
            return "READ-OFF FROM THE BRIEF\n" + "\n".join(rows) + "\n"
        except Exception:
            return ""


def _cleave(brief: str) -> tuple[str, str]:
    """Split the shopper's sentence from the generic operating brief."""
    try:
        head, sep, tail = brief.partition("\n\n")
        if not sep:
            return brief.strip(), ""
        body = tail.strip()
        # A bare heading line ("<something>:") introduces the brief; drop it.
        first, nl, rest = body.partition("\n")
        if first.rstrip().endswith(":") and len(first) < 40:
            body = rest.strip()
        return head.strip(), body
    except Exception:
        return "", ""


def _dedupe(items: Any) -> list[str]:
    try:
        return list(dict.fromkeys(str(item) for item in items))
    except Exception:
        return []


def _first_ceiling(text: str) -> str:
    try:
        hit = _CEILING_RX.search(text or "")
        return hit.group(1) if hit else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Claim hygiene
# ---------------------------------------------------------------------------


class ClaimDesk:
    """Keeps an order's claims inside the catalog's attribute vocabulary."""

    @staticmethod
    def canonical_field(raw: str, shape: TaskShape) -> str:
        try:
            label = str(raw or "").strip()
            key = label.lower()
            if key in _CLAIM_CANON:
                return _CLAIM_CANON[key]
            if key in _CLAIM_REJECT:
                return ""
            if _BARE_FIGURE_RX.match(label):
                # A measurement was handed in where a field name belongs.
                return ClaimDesk.field_for_figure(shape) or "resolution"
            return key.replace(" ", "_")
        except Exception:
            return ""

    @staticmethod
    def field_for_figure(shape: TaskShape) -> str:
        try:
            blob = shape.goal.lower()
            if re.search(r"\binch|\"|screen size", blob):
                return "screen_size"
            if re.search(r"\d{3,4}\s*p|full\s*hd|qhd|uhd|4k|\bhd\b|resolution", blob):
                return "resolution"
            if re.search(r"\bhz\b|refresh", blob):
                return "refresh_rate"
            if re.search(r"\bram\b|\bgb\b", blob):
                return "ram"
            return ""
        except Exception:
            return ""

    @staticmethod
    def tidy_value(field: str, raw: Any) -> str:
        try:
            text = "" if raw is None else str(raw).strip()
            if not text:
                return ""
            if field == "screen_size":
                hit = re.search(r"\d+(?:\.\d+)?", text)
                return f"{hit.group(0)} inches" if hit else text
            if field == "refresh_rate":
                hit = re.search(r"\d+", text)
                return f"{hit.group(0)} Hz" if hit else text
            if field == "in_stock":
                low = text.lower()
                if low in {"true", "1", "yes", "in_stock", "in stock"}:
                    return "true"
                if low in {"false", "0", "no", "out_of_stock", "oos"}:
                    return "false"
                return text
            if field == "resolution":
                return ClaimDesk._tidy_resolution(text)
            if field == "price":
                hit = re.search(r"[\d]+(?:\.\d+)?", text.replace(",", ""))
                return f"{float(hit.group(0)):.2f}" if hit else text
            if field == "ram":
                # Reject marketing junk that is not a memory size.
                if not re.search(r"\d", text):
                    return ""
                hit = re.search(r"(\d+)\s*(gb|tb)?", text, re.I)
                if not hit:
                    return ""
                unit = (hit.group(2) or "gb").lower()
                return f"{hit.group(1)}{unit}"
            return text
        except Exception:
            return ""

    @staticmethod
    def _tidy_resolution(text: str) -> str:
        """Grader wants canonical tokens hd/fhd/qhd/4k — not pixel strings."""
        try:
            raw = (text or "").strip().lower()
            squeezed = re.sub(r"[\s_\-]", "", raw)
            if "fullhd" in squeezed or "1080p" in squeezed or squeezed == "fhd":
                return "fhd"
            if squeezed in {"hd", "720p", "720phd", "hdready"} or (
                "720" in squeezed and "1080" not in squeezed and "full" not in squeezed
            ):
                return "hd"
            if "1440" in squeezed or squeezed == "qhd" or "2k" == squeezed:
                return "qhd"
            if "2160" in squeezed or "uhd" in squeezed or squeezed == "4k" or "3840" in squeezed:
                return "4k"
            pair = re.search(r"(\d{3,4})\D+(\d{3,4})", raw)
            if pair:
                w, h = int(pair.group(1)), int(pair.group(2))
                hi = max(w, h)
                if hi >= 2100:
                    return "4k"
                if hi >= 1400:
                    return "qhd"
                if hi >= 1000:
                    return "fhd"
                if hi >= 700:
                    return "hd"
            if squeezed in {"hd", "fhd", "qhd", "4k"}:
                return squeezed
            return ""
        except Exception:
            return ""

    @staticmethod
    def polish(claims: Any, shape: TaskShape) -> list[dict[str, str]]:
        try:
            if not isinstance(claims, list):
                return []
            goal = shape.goal.lower()
            gate = ClaimDesk.field_for_figure(shape)
            kept: list[dict[str, str]] = []
            used: set[str] = set()
            support_taken = False
            for entry in claims:
                if not isinstance(entry, dict):
                    continue
                field = ClaimDesk.canonical_field(entry.get("field"), shape)
                if not field or field in _CLAIM_REJECT or field not in _CLAIM_OK:
                    continue
                value = ClaimDesk.tidy_value(field, entry.get("value"))
                if not value:
                    continue
                # Only keep the shopper's gated figure as a spec claim, plus
                # price / in_stock / one optional brand.
                if field in _CLAIM_SPEC and gate and field != gate:
                    continue
                if field in _CLAIM_SUPPORT and field != "brand":
                    continue
                if field == "brand" and "brand" not in goal and support_taken:
                    continue
                if field == "brand":
                    if support_taken:
                        continue
                    support_taken = True
                if field in used:
                    continue
                used.add(field)
                kept.append({"field": field, "value": value})

            # Ensure gate + stock are present when vouching.
            if shape.order_may_carry_claims():
                if gate and gate not in used and shape.figures:
                    value = ClaimDesk.tidy_value(gate, shape.figures[0])
                    if value:
                        kept.insert(0, {"field": gate, "value": value})
                        used.add(gate)
                if "in_stock" not in used:
                    kept.append({"field": "in_stock", "value": "true"})
            return kept[:6]
        except Exception:
            return []

    @staticmethod
    def from_observation(
        shape: TaskShape,
        *,
        facts: dict[str, Any] | None,
        price: float | None,
        in_stock: bool | None,
        title: str = "",
    ) -> list[dict[str, str]]:
        """Build grader-safe claims from the last viewed/inspected facts only."""
        try:
            blob_parts = [title]
            clean: dict[str, str] = {}
            for key, val in (facts or {}).items():
                if val is None or isinstance(val, (dict, list)):
                    continue
                text = str(val).strip()
                if not text or text.lower() in {"none", "null"}:
                    continue
                clean[str(key).lower()] = text
                blob_parts.append(text)
            blob = " ".join(blob_parts).lower()
            claims: list[dict[str, str]] = []
            gate = ClaimDesk.field_for_figure(shape)
            if gate == "resolution":
                raw = (
                    clean.get("resolution")
                    or clean.get("display")
                    or clean.get("screen_resolution")
                    or blob
                )
                value = ClaimDesk.tidy_value("resolution", raw)
                if not value and shape.figures:
                    value = ClaimDesk.tidy_value("resolution", shape.figures[0])
                if value:
                    claims.append({"field": "resolution", "value": value})
            elif gate == "screen_size":
                raw = clean.get("screen_size") or clean.get("size") or clean.get("display_size")
                value = ClaimDesk.tidy_value("screen_size", raw or blob)
                if not value and shape.figures:
                    value = ClaimDesk.tidy_value("screen_size", shape.figures[0])
                if value:
                    claims.append({"field": "screen_size", "value": value})
            elif gate == "refresh_rate":
                raw = clean.get("refresh_rate") or clean.get("refresh") or blob
                value = ClaimDesk.tidy_value("refresh_rate", raw)
                if value:
                    claims.append({"field": "refresh_rate", "value": value})
            elif gate == "ram":
                raw = clean.get("ram") or clean.get("memory") or blob
                value = ClaimDesk.tidy_value("ram", raw)
                if value:
                    claims.append({"field": "ram", "value": value})
            if price is not None:
                claims.append({"field": "price", "value": f"{float(price):.2f}"})
            if in_stock is not None:
                claims.append(
                    {"field": "in_stock", "value": "true" if in_stock else "false"}
                )
            brand = clean.get("brand")
            if brand:
                claims.append({"field": "brand", "value": brand})
            return ClaimDesk.polish(claims, shape)
        except Exception:
            return []
            return kept
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_ROLE_BRIEF = """
You are FlowerGirl v5, an autonomous purchasing operator working a live
catalog through the tools this episode hands you.

WHAT ENDS THE EPISODE
An observation returning done=true. Everything before that is evidence
collection, and every statement you make must trace back to an observation.

OPERATING RHYTHM
  Locate     search / filter with words a seller would print on a listing.
  Verify     view / compare / inspect_stock the concrete variant, not the title.
  Commit     add_to_cart the item you actually intend to buy.
  Re-verify  re-read price and availability after anything changes.
  Close      place_test_order.

REQUIREMENT LEDGER
Sort the request into BINDING items — true of the exact variant you buy — and
WISHES, which improve a pick but never overrule a BINDING item. Product class,
one hardware figure, the spend ceiling and availability are normally BINDING.
A named maker or a comfort quality is normally a WISH.

VARIANT DISCIPLINE
A single listing often carries both compliant and non-compliant variants.
Bind every BINDING item to the selected variant's own values before you commit.
Where the brief defines measurement bands, use its definition and invent none.

QUERY CRAFT
The catalog indexes values, not schema keys, so feed it nouns a shopper would
say. When results come back thin, change one thing — drop a filler word,
normalise a size, keep the rare token — instead of rewriting the query.

TALKING TO THE SHOPPER
If the shopper asks you anything, answer with message as your next action.
An unanswered question sinks the episode even when the purchase is correct.
Keep replies to one or two sentences.

STEP BUDGET
You are told how many steps remain. Land the order before they run out: an
episode that never orders scores nothing. After you have a verified in-stock
cart item that clears BINDING requirements, place_test_order immediately —
do not keep browsing.
""".strip()

_LANE_BRIEF: dict[str, str] = {
    SPLIT: """
LANE — SPLIT THE ASK
The request blends non-negotiables with a stated leaning, usually a maker.
1 Write the BINDING list first and the WISH list second.
2 Shortlist only items that clear every BINDING item.
3 If an item satisfies the leaning as well, that is your first commit.
  Otherwise commit to the strongest fully compliant item and move on.
4 The first commit is graded on its own; a later swap does not undo it.
5 Answer any shopper question with message before you order.
""".strip(),
    SWEEP: """
LANE — WIDEN THE NET
1 Try the shopper's own wording, then a tokenised variant, then a
  category-scoped pass.
2 Assemble a shortlist of genuinely different products before choosing.
3 Open the promising ones and discard accessories, bundles and look-alikes by
  reading each listing's own classification rather than its title.
4 Buy the strongest available item inside the ceiling. Once it checks out, do
  not idle waiting for approval.
""".strip(),
    LOCK: """
LANE — EXACT MATCH ONLY
1 If the shopper is holding one figure back until asked, ask with message
  before searching.
2 Every BINDING item must hold on the variant you order: option, price and
  availability together.
3 Read the selected variant's own values. Titles overstate.
4 If your variant sells out, do not slide to a sibling that misses a figure —
  find a different product that clears everything.
""".strip(),
    TASTE: """
LANE — FIT BEFORE PRICE
BINDING: product class, spend ceiling, availability, and a genuine unit for
sale — never a spare part, a bundle add-on or an editorial page.
WISHES: the qualities implied by how the shopper says they will use it.
1 Read raw facts — protection ratings, endurance, materials, connectivity.
2 Compare at least two credible candidates on those qualities.
3 Commit to the best fit inside the ceiling. Cheapness is a tiebreaker, never
  the objective, and the first commit is graded.
""".strip(),
    PODIUM: """
LANE — ORDERED CHOICE
1 Work only from the variants the shopper listed. Open them directly.
  Never call search or filter.
2 In one turn, view several candidates or compare them.
3 Sort strictly by the priority the shopper stated, in the order stated.
4 Commit to the current leader, then inspect_stock on survivors after any
  price/stock change and re-sort before ordering.
5 Never take an option that loses on every stated axis.
6 Close with place_test_order carrying product_id and sku only.
""".strip(),
    SALVAGE: """
LANE — MINIMUM CONCESSION
Hold firm on product class, spend ceiling and availability.
When the named item is gone, concede in this order:
  same model in another shade  ->  a different model
Trust an explicit model fact over a title that disagrees with it.
Re-read price and availability after every cart change, then order.
""".strip(),
    VOUCH: """
LANE — EVIDENCE ON THE RECORD
1 Open the candidate and read its raw facts before committing and again
  before ordering. Titles are marketing, not evidence.
2 The order must carry justification.claims as field/value pairs.
3 Field names come from the catalog's attribute vocabulary: the one figure the
  shopper asked for, plus price and in_stock, plus at most brand.
4 Resolution values MUST be the tokens hd, fhd, qhd, or 4k — never pixel
  strings like 1920 x 1080 and never 1080p / Full HD prose.
5 Every value must equal what you observed on view/inspect_stock.
""".strip(),
}

_IDLE_PROMPTS = (
    "That reply carried no tool call. Pick a tool from this episode's list and "
    "act now — search or filter if you have not looked yet.",
    "Still nothing to execute. Move the episode: search, open a candidate, "
    "answer the shopper with message, or close the verified cart with "
    "place_test_order.",
)

_TIP_ASK_FIRST = (
    "The shopper is withholding one requirement until asked. Send message now "
    "to get that figure before you search."
)
_TIP_ANSWER = (
    "The shopper just asked you something. Reply with message on this turn, "
    "before any further cart or order action."
)
_TIP_OPEN_WITH_SEARCH = (
    "Begin with search or filter using words from the shopper's own sentence, "
    "then open the best hits."
)
_TIP_RESORT = (
    "The market moved. Re-read the candidates still live, sort them again by "
    "the stated priority, put the new leader in the cart, confirm its stock, "
    "then order with product_id and sku only."
)
_TIP_CLOSE_PODIUM = (
    "Stop exploring. Sort the surviving candidates, cart the leader if it is "
    "not already there, and place_test_order now with product_id and sku only."
)
_TIP_CLAIM_SHAPE = (
    "On place_test_order the claims must use catalog attribute names. For "
    "resolution use ONLY the tokens hd / fhd / qhd / 4k (never '1920 x 1080' "
    "or '1080p'). Include that gated figure, price from the latest "
    "inspect_stock, in_stock, and at most brand. No category path, no OS, no "
    "prose."
)
_TIP_EVENT_STOCK = (
    "The shopper just spoke after your cart action. Reply with message once, "
    "then IMMEDIATELY inspect_stock on the SAME carted product_id/sku before "
    "any remove_from_cart. Skipping that inspect zeros the score."
)
_TIP_STOP_BROWSING = (
    "You have surveyed enough listings. Pick the best in-stock BINDING match "
    "already viewed, add_to_cart it if needed, inspect_stock, then "
    "place_test_order NOW."
)
_TIP_SALVAGE_EXACT = (
    "First search and view the exact named model (and shade if stated), even "
    "if it is unavailable — observing it is required. Only then concede to "
    "another shade of the same model, then another model. Never order a "
    "variant whose latest inspect_stock shows in_stock=false."
)
_TIP_NO_OOS_ORDER = (
    "Latest inspect_stock shows out of stock or over budget. remove_from_cart "
    "only AFTER that inspect, then switch to another compliant in-stock item "
    "and inspect_stock it before ordering."
)
_TIP_RECOVER = (
    "Something failed or moved ({detail}). Recover with the smallest concession "
    "that still holds product class, spend ceiling and availability."
)
_TIP_BRAND_FIRST = (
    "Your first cart entry should take the stated leaning when a fully "
    "compliant item offers it, and otherwise the best compliant item. Answer "
    "shopper questions with message."
)
_TIP_COMPARE = (
    "Compare candidates on the qualities the shopper's use case implies before "
    "the first cart entry, not on price alone."
)
_TIP_REFRAME = (
    "Adjust one part of the query or add a filter so the shortlist spans "
    "several distinct products before you buy."
)
_TIP_DEADLINE = (
    "Only {left} step(s) remain. Confirm the current pick's price and stock, "
    "then place_test_order."
)
_TIP_TIME_DEADLINE = (
    "Wall-clock budget is nearly spent. Stop exploring. If a carted variant "
    "still clears BINDING items, place_test_order NOW with product_id and sku "
    "(plus justification.claims only when the shopper asked for a reason)."
)
_TIP_FORCE_ORDER = (
    "You already verified a cart candidate. Emit place_test_order this turn — "
    "do not search, view, or compare again."
)
_TIP_PODIUM_NO_SEARCH = (
    "Do not call search or filter. Work only from the listed product_id::sku "
    "candidates. In one turn, view several of them (or compare), then "
    "inspect_stock, add_to_cart the current leader, re-check survivors after "
    "any stock/price change, and place_test_order with product_id and sku only."
)


def _compose_system(shape: TaskShape, policy: dict[str, Any]) -> str:
    try:
        tools = _tool_labels(policy.get("tools") or [])
        lane_brief = _LANE_BRIEF.get(shape.lane, _LANE_BRIEF[SWEEP])
        parts = [_ROLE_BRIEF, lane_brief]
        sheet = shape.fact_sheet()
        if sheet:
            parts.append(sheet.strip())
        parts.append(
            "SESSION\n"
            f"- lane: {shape.lane}\n"
            f"- tools: {', '.join(tools) if tools else '(read the supplied schemas)'}\n"
            f"- step limit: {policy.get('max_steps')}\n"
            f"- actions per turn: {policy.get('max_calls_per_turn', 1)}"
        )
        return "\n\n".join(parts)
    except Exception:
        return ""


def _tool_labels(tools: list[Any]) -> list[str]:
    labels: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        spec = tool.get("function")
        if isinstance(spec, dict) and spec.get("name"):
            labels.append(str(spec["name"]))
        elif tool.get("name"):
            labels.append(str(tool["name"]))
    return labels


# ---------------------------------------------------------------------------
# Episode state
# ---------------------------------------------------------------------------


class Ledger:
    """Mutable record of what has happened so far in this episode."""

    __slots__ = (
        "turn",
        "limit",
        "acted",
        "history",
        "seen",
        "committed",
        "closed",
        "owes_reply",
        "market_moved",
        "asked_for_figure",
        "coached",
        "cart_pid",
        "cart_sku",
        "started_at",
        "view_count",
        "search_count",
        "last_price",
        "last_in_stock",
        "last_facts",
        "last_title",
        "awaiting_stock_check",
        "exact_seen",
    )

    def __init__(self, limit: int) -> None:
        try:
            self.turn = 0
            self.limit = limit
            self.acted: list[str] = []
            self.history: set[str] = set()
            self.seen: list[Any] = []
            self.committed = False
            self.closed = False
            self.owes_reply = False
            self.market_moved = False
            self.asked_for_figure = False
            self.coached: set[str] = set()
            self.cart_pid = ""
            self.cart_sku = ""
            self.started_at = time.monotonic()
            self.view_count = 0
            self.search_count = 0
            self.last_price: float | None = None
            self.last_in_stock: bool | None = None
            self.last_facts: dict[str, Any] = {}
            self.last_title = ""
            self.awaiting_stock_check = False
            self.exact_seen = False
        except Exception:
            pass

    @property
    def steps_left(self) -> int:
        try:
            return max(0, self.limit - self.turn)
        except Exception:
            return 0

    @property
    def elapsed(self) -> float:
        try:
            return max(0.0, time.monotonic() - self.started_at)
        except Exception:
            return 0.0

    def hurry(self) -> bool:
        try:
            return self.elapsed >= _WALL_SOFT_SEC or self.steps_left <= 8
        except Exception:
            return False

    def time_up(self) -> bool:
        try:
            return self.elapsed >= _WALL_HARD_SEC or self.steps_left <= 2
        except Exception:
            return False

    def did(self, label: str) -> bool:
        """True when the action ran on the most recent turn."""
        try:
            return label in self.acted
        except Exception:
            return False

    def ever(self, label: str) -> bool:
        try:
            return label in self.history
        except Exception:
            return False

    def note_actions(self, labels: list[str]) -> None:
        try:
            self.acted = labels
            self.history.update(labels)
            self.view_count += sum(1 for label in labels if label == "view")
            self.search_count += sum(1 for label in labels if label in {"search", "filter"})
            if "add_to_cart" in labels:
                self.committed = True
                self.awaiting_stock_check = True
            if "place_test_order" in labels:
                self.closed = True
            if "inspect_stock" in labels:
                self.awaiting_stock_check = False
            if "message" in labels:
                self.owes_reply = False
                self.asked_for_figure = True
            if "remove_from_cart" in labels:
                pass
        except Exception:
            pass

    def remember_cart(self, name: str, args: dict[str, Any]) -> None:
        try:
            if name in {"add_to_cart", "place_test_order"}:
                pid = str(args.get("product_id") or "").strip()
                sku = str(args.get("sku") or "").strip()
                if pid:
                    self.cart_pid = pid
                if sku:
                    self.cart_sku = sku
            elif name == "remove_from_cart":
                self.cart_pid = ""
                self.cart_sku = ""
                self.committed = False
        except Exception:
            pass

    def once(self, key: str) -> bool:
        """True the first time a coaching key is requested."""
        try:
            if key in self.coached:
                return False
            self.coached.add(key)
            return True
        except Exception:
            return False


def _is_question(text: str) -> bool:
    try:
        body = (text or "").strip().lower()
        if not body:
            return False
        if "?" in body:
            return True
        openers = (
            "which",
            "what",
            "does",
            "did",
            "can you",
            "could you",
            "would you",
            "wait",
            "hold on",
            "before you",
            "re-check",
            "recheck",
        )
        return any(body.startswith(word) for word in openers)
    except Exception:
        return False


def _shifted(observation: Any) -> bool:
    """True for stock/price/cart movement — not for a plain tool error."""
    try:
        if not isinstance(observation, dict):
            return False
        for key in ("market_event", "stock_changed", "price_changed", "event"):
            if observation.get(key):
                return True
        nested = observation.get("observation")
        if isinstance(nested, dict):
            for key in ("market_event", "stock_changed", "price_changed"):
                if nested.get(key):
                    return True
        return False
    except Exception:
        return False


def _fault(observations: list[Any]) -> str:
    try:
        for item in observations:
            if isinstance(item, dict) and item.get("error"):
                return str(item["error"])
        return ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class Bridge:
    """Chat completions with a short fallback ladder of OpenRouter engines."""

    def __init__(self, client: ProxyClient) -> None:
        try:
            self._client = client
        except Exception:
            pass

    @staticmethod
    def engine_for(lane: str) -> str:
        try:
            override = (getenv("SANDBOX_MODEL") or "").strip()
            if override:
                return override
            return _ENGINE_DEEP if lane in _DEEP_LANES else _ENGINE_QUICK
        except Exception:
            return ""

    def ask(
        self,
        *,
        engine: str,
        transcript: list[dict[str, Any]],
        schemas: list[Any],
        budget: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        try:
            ladder = [engine]
            for spare in _ENGINE_RELIEF:
                if spare not in ladder:
                    ladder.append(spare)

            trouble: Exception | None = None
            for candidate in ladder[:3]:
                payload = self._client.post(
                    "/inference/chat/completions",
                    json_data={
                        "model": candidate,
                        "messages": transcript,
                        "tools": schemas,
                        "tool_choice": "required",
                        "temperature": 0,
                    },
                )
                if payload is None:
                    trouble = RuntimeError(f"no completion from {candidate}")
                    continue
                try:
                    reply = payload["choices"][0]["message"]
                except (KeyError, IndexError, TypeError) as exc:
                    trouble = exc
                    continue
                spoken = reply.get("content") or ""
                calls = list(reply.get("tool_calls") or [])[:budget]
                return spoken, calls
            raise RuntimeError(f"inference request failed ({trouble})")
        except RuntimeError:
            raise
        except Exception:
            return "", []

    def act(self, envelope: dict[str, Any]) -> dict[str, Any]:
        try:
            reply = self._client.post("/environment/call", json_data=envelope)
            if reply is None:
                raise RuntimeError("environment call failed")
            return reply
        except RuntimeError:
            raise
        except Exception:
            return {}


_bridge = Bridge(ProxyClient(timeout=45, max_retries=2))


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


class Episode:
    """Drives one generated environment to done=true."""

    def __init__(self, problem_data: dict[str, Any]) -> None:
        try:
            environment = problem_data["environment"]
            self.binding = environment["binding"]
            self.policy = environment["policy_view"]
            self.schemas = self.policy.get("tools") or []
            self.brief = str(
                self.policy.get("query") or problem_data.get("query") or ""
            )
            self.key = str(
                problem_data.get("problem_id", problem_data.get("id", "problem"))
            )
            self.per_turn = max(1, int(self.policy.get("max_calls_per_turn", 1)))
            self.shape = TaskShape(self.brief)
            self.ledger = Ledger(max(1, int(self.policy.get("max_steps", 20))))
            self.engine = Bridge.engine_for(self.shape.lane)
        except Exception:
            pass

    # -- transcript ------------------------------------------------------

    def _opening_transcript(self) -> list[dict[str, Any]]:
        try:
            transcript: list[dict[str, Any]] = [
                {"role": "system", "content": _compose_system(self.shape, self.policy)},
                {"role": "user", "content": self.brief},
            ]
            if self.shape.lane == LOCK and self.shape.held_back_figure:
                transcript.append({"role": "user", "content": _TIP_ASK_FIRST})
            elif self.shape.lane == VOUCH:
                transcript.append({"role": "user", "content": _TIP_CLAIM_SHAPE})
            return transcript
        except Exception:
            return []

    # -- coaching --------------------------------------------------------

    def _guidance(self) -> str | None:
        try:
            led = self.ledger
            lane = self.shape.lane
            turn = led.turn

            if led.owes_reply:
                if led.committed and led.awaiting_stock_check:
                    return _TIP_EVENT_STOCK
                return _TIP_ANSWER

            if (
                led.committed
                and led.awaiting_stock_check
                and not led.did("inspect_stock")
                and led.once("event_stock")
            ):
                return _TIP_EVENT_STOCK

            if led.last_in_stock is False and led.committed and led.once("no_oos"):
                return _TIP_NO_OOS_ORDER

            if led.time_up() and led.committed and not led.closed and led.once("time_deadline"):
                return _TIP_TIME_DEADLINE

            if (
                led.committed
                and not led.closed
                and (led.hurry() or led.steps_left <= 5)
                and led.once("force_order")
            ):
                return _TIP_FORCE_ORDER

            if (
                not led.committed
                and (led.view_count >= 8 or led.search_count >= 4 or led.steps_left <= 10)
                and led.once("stop_browse")
            ):
                return _TIP_STOP_BROWSING

            if (
                lane == LOCK
                and turn == 1
                and self.shape.held_back_figure
                and not led.asked_for_figure
            ):
                return _TIP_ASK_FIRST

            if lane == PODIUM and not led.closed:
                if turn == 1:
                    return _TIP_PODIUM_NO_SEARCH
                if led.market_moved and not led.did("place_test_order"):
                    return _TIP_RESORT
                if led.steps_left <= 5 or led.hurry():
                    return _TIP_CLOSE_PODIUM

            if turn == 1 and lane == SPLIT:
                return _TIP_BRAND_FIRST

            if turn == 1 and lane == SALVAGE:
                return _TIP_SALVAGE_EXACT

            if turn <= 2 and not any(
                led.ever(label)
                for label in ("search", "filter", "view", "compare", "message")
            ):
                if lane == PODIUM:
                    return _TIP_PODIUM_NO_SEARCH
                return _TIP_OPEN_WITH_SEARCH

            if lane == VOUCH and not led.closed and turn in (1, 2, 5):
                return _TIP_CLAIM_SHAPE

            if lane == SALVAGE:
                detail = _fault(led.seen)
                if detail:
                    return _TIP_RECOVER.format(detail=detail)

            if lane == SWEEP and turn == 3 and not led.committed and led.ever("search"):
                return _TIP_REFRAME

            if lane == TASTE and turn == 2 and not led.ever("compare"):
                return _TIP_COMPARE

            if (
                not led.closed
                and led.committed
                and led.steps_left <= 4
                and led.once("deadline")
            ):
                return _TIP_DEADLINE.format(left=led.steps_left)

            return None
        except Exception:
            return None

    # -- outbound arguments ----------------------------------------------

    def _shape_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            # Ranking episodes must stay on the listed candidates.
            if self.shape.lane == PODIUM and name in {"search", "filter"}:
                return {"query": " ", "k": 1} if name == "search" else dict(args)

            if name != "place_test_order":
                return args
            payload = dict(args)
            if self.ledger.cart_pid and not payload.get("product_id"):
                payload["product_id"] = self.ledger.cart_pid
            if self.ledger.cart_sku and not payload.get("sku"):
                payload["sku"] = self.ledger.cart_sku

            # Never submit an order we already know is OOS.
            if self.ledger.last_in_stock is False:
                payload["_blocked_oos"] = True
                return payload

            if not self.shape.order_may_carry_claims():
                payload.pop("justification", None)
                return payload

            grounded = ClaimDesk.from_observation(
                self.shape,
                facts=self.ledger.last_facts,
                price=self.ledger.last_price,
                in_stock=True if self.ledger.last_in_stock is None else self.ledger.last_in_stock,
                title=self.ledger.last_title,
            )
            if grounded:
                payload["justification"] = {"claims": grounded}
                return payload

            note = payload.get("justification")
            if not isinstance(note, dict):
                payload["justification"] = {
                    "claims": ClaimDesk.polish(
                        [
                            {
                                "field": ClaimDesk.field_for_figure(self.shape) or "resolution",
                                "value": (self.shape.figures[0] if self.shape.figures else ""),
                            },
                            {"field": "in_stock", "value": "true"},
                        ],
                        self.shape,
                    )
                }
                return payload
            payload["justification"] = {
                "claims": ClaimDesk.polish(note.get("claims"), self.shape)
            }
            return payload
        except Exception:
            return {}

    # -- environment -----------------------------------------------------

    def _dispatch(self, calls: list[dict[str, Any]], turn: int) -> dict[str, Any]:
        try:
            batch = f"{self.key}-turn-{turn}"
            actions = []
            for slot, call in enumerate(calls, start=1):
                name = _call_name(call)
                if not name:
                    raise RuntimeError("model produced a tool call with no name")
                # Ranking must stay on listed candidates — drop open-catalog tools.
                if self.shape.lane == PODIUM and name in {"search", "filter"}:
                    continue
                args = self._shape_args(name, _call_args(call))
                # Convert a known-OOS order into inspect_stock / skip.
                if name == "place_test_order" and args.pop("_blocked_oos", False):
                    name = "inspect_stock"
                    args = {
                        "product_id": self.ledger.cart_pid or args.get("product_id"),
                        "sku": self.ledger.cart_sku or args.get("sku"),
                    }
                self.ledger.remember_cart(name, args)
                spec = call.get("function")
                if isinstance(spec, dict):
                    spec["name"] = name
                    spec["arguments"] = json.dumps(args)
                actions.append(
                    {"call_id": f"{batch}-{slot}", "action": {"name": name, "args": args}}
                )
            if not actions:
                return {"calls": []}
            return _bridge.act(
                {
                    **self.binding,
                    "call_id": batch,
                    "idempotency_key": batch,
                    "turn": turn,
                    "calls": actions,
                }
            )
        except RuntimeError:
            raise
        except Exception:
            return {}

    def _force_close_calls(self) -> list[dict[str, Any]]:
        """Deterministic last-chance order when the model stalls with a cart."""
        try:
            led = self.ledger
            if led.closed or not led.cart_pid:
                return []
            if led.last_in_stock is False:
                # Prefer a stock re-check over ordering a known OOS item.
                return [
                    {
                        "id": f"{self.key}-force-stock",
                        "type": "function",
                        "function": {
                            "name": "inspect_stock",
                            "arguments": json.dumps(
                                {
                                    "product_id": led.cart_pid,
                                    "sku": led.cart_sku or led.cart_pid,
                                }
                            ),
                        },
                    }
                ]
            args: dict[str, Any] = {
                "product_id": led.cart_pid,
                "sku": led.cart_sku or led.cart_pid,
            }
            if self.shape.order_may_carry_claims():
                claims = ClaimDesk.from_observation(
                    self.shape,
                    facts=led.last_facts,
                    price=led.last_price,
                    in_stock=True if led.last_in_stock is None else led.last_in_stock,
                    title=led.last_title,
                )
                if not claims:
                    field = ClaimDesk.field_for_figure(self.shape)
                    seed = []
                    if field and self.shape.figures:
                        seed.append(
                            {
                                "field": field,
                                "value": ClaimDesk.tidy_value(field, self.shape.figures[0]),
                            }
                        )
                    seed.append({"field": "in_stock", "value": "true"})
                    claims = ClaimDesk.polish(seed, self.shape)
                args["justification"] = {"claims": claims}
            return [
                {
                    "id": f"{self.key}-force-order",
                    "type": "function",
                    "function": {
                        "name": "place_test_order",
                        "arguments": json.dumps(args),
                    },
                }
            ]
        except Exception:
            return []

    def _podium_bootstrap_calls(self) -> list[dict[str, Any]]:
        """Open listed variants in one turn instead of burning steps one-by-one."""
        try:
            calls: list[dict[str, Any]] = []
            pairs: list[tuple[str, str]] = []
            for token in self.shape.variants[:8]:
                if "::" not in token:
                    continue
                pid, sku = token.split("::", 1)
                pairs.append((pid.strip(), sku.strip()))
            if not pairs:
                return []
            # Prefer compare when several product ids are distinct.
            pids = list(dict.fromkeys(pid for pid, _ in pairs))
            if len(pids) >= 2:
                calls.append(
                    {
                        "id": f"{self.key}-podium-compare",
                        "type": "function",
                        "function": {
                            "name": "compare",
                            "arguments": json.dumps({"product_ids": pids[:5]}),
                        },
                    }
                )
            for idx, (pid, _sku) in enumerate(pairs[:4]):
                calls.append(
                    {
                        "id": f"{self.key}-podium-view-{idx}",
                        "type": "function",
                        "function": {
                            "name": "view",
                            "arguments": json.dumps({"product_id": pid}),
                        },
                    }
                )
            return calls[: max(1, self.per_turn)]
        except Exception:
            return []

    # -- main loop -------------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        trail: list[dict[str, Any]] = []
        try:
            transcript = self._opening_transcript()
            trail: list[dict[str, Any]] = [
                {
                    "role": "meta",
                    "flowergirl": {
                        "version": 7,
                        "lane": self.shape.lane,
                        "engine": self.engine,
                        "provider": "openrouter",
                    },
                }
            ]

            led = self.ledger
            while led.turn < led.limit:
                if led.time_up() and led.closed:
                    break
                led.turn += 1
                tip = self._guidance()
                if tip:
                    transcript.append({"role": "user", "content": tip})

                # Structured bootstrap for listed-candidate ranking.
                if (
                    led.turn == 1
                    and self.shape.lane == PODIUM
                    and self.shape.variants
                ):
                    spoken, calls = (
                        "Opening the listed candidates.",
                        self._podium_bootstrap_calls(),
                    )
                elif led.time_up() and led.committed and not led.closed:
                    spoken, calls = (
                        "Closing the verified cart before the wall clock ends.",
                        self._force_close_calls(),
                    )
                else:
                    try:
                        spoken, calls = self._speak(transcript, trail)
                    except Exception:
                        spoken, calls = "", []

                if not calls and led.committed and not led.closed:
                    calls = self._force_close_calls()
                    spoken = spoken or "Placing the verified order."

                if not calls:
                    if led.time_up() or led.steps_left <= 0:
                        break
                    continue

                # Keep ledger action history aligned with what we actually send.
                dispatched = []
                for call in calls:
                    name = _call_name(call)
                    if self.shape.lane == PODIUM and name in {"search", "filter"}:
                        continue
                    if name:
                        dispatched.append(call)
                calls = dispatched or calls

                transcript.append(
                    {"role": "assistant", "content": spoken, "tool_calls": calls}
                )

                result = self._dispatch(calls, led.turn)
                trail.append(
                    {
                        "role": "assistant",
                        "content": spoken,
                        "tool_calls": calls,
                        "environment_result": result,
                        "flowergirl_turn": {
                            "lane": self.shape.lane,
                            "turn": led.turn,
                            "elapsed": round(led.elapsed, 1),
                        },
                    }
                )

                led.note_actions([_call_name(call) or "?" for call in calls])
                self._absorb(result, transcript, calls)

                if self._finished(result):
                    break

                # If still carted with almost no steps/time, force one more close.
                if (
                    not led.closed
                    and led.committed
                    and led.cart_pid
                    and (led.steps_left <= 1 or led.elapsed >= _WALL_HARD_SEC - 15)
                ):
                    force = self._force_close_calls()
                    if force:
                        led.turn += 1
                        forced = self._dispatch(force, led.turn)
                        led.note_actions(["place_test_order"])
                        trail.append(
                            {
                                "role": "assistant",
                                "content": "Forced close under time/step budget.",
                                "tool_calls": force,
                                "environment_result": forced,
                            }
                        )
                        self._absorb(forced, transcript, force)
                        break

            return trail
        except Exception:
            return trail

    def _speak(
        self, transcript: list[dict[str, Any]], trail: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        try:
            for attempt in range(len(_IDLE_PROMPTS) + 1):
                spoken, calls = _bridge.ask(
                    engine=self.engine,
                    transcript=transcript,
                    schemas=self.schemas,
                    budget=self.per_turn,
                )
                if calls:
                    return spoken, calls
                trail.append({"role": "assistant", "content": spoken})
                transcript.extend(
                    [
                        {"role": "assistant", "content": spoken},
                        {
                            "role": "user",
                            "content": _IDLE_PROMPTS[
                                min(attempt, len(_IDLE_PROMPTS) - 1)
                            ],
                        },
                    ]
                )
            raise RuntimeError("engine would not emit an executable action")
        except RuntimeError:
            raise
        except Exception:
            return "", []

    def _harvest(self, name: str, observation: Any) -> None:
        """Pull price/stock/facts out of tool observations for later claims."""
        try:
            if not isinstance(observation, dict):
                return
            led = self.ledger
            # Common wrappers: data / product / products / result
            nodes: list[Any] = [observation]
            for key in ("data", "product", "result", "item"):
                if isinstance(observation.get(key), dict):
                    nodes.append(observation[key])
            if isinstance(observation.get("products"), list):
                nodes.extend(
                    p for p in observation["products"][:3] if isinstance(p, dict)
                )

            for node in nodes:
                if not isinstance(node, dict):
                    continue
                title = node.get("title") or node.get("name")
                if isinstance(title, str) and title.strip():
                    led.last_title = title.strip()
                facts = node.get("facts") or node.get("attributes") or node.get("raw_facts")
                if isinstance(facts, dict) and facts:
                    led.last_facts = {
                        str(k): v
                        for k, v in facts.items()
                        if v is not None and not isinstance(v, (dict, list))
                    }
                price = node.get("price") or node.get("current_price")
                if price is not None:
                    try:
                        led.last_price = float(str(price).replace(",", ""))
                    except ValueError:
                        pass
                if "in_stock" in node:
                    led.last_in_stock = bool(node.get("in_stock"))

            if name == "inspect_stock":
                led.awaiting_stock_check = False
            if name == "view" and self.shape.lane == SALVAGE:
                # Any concrete view counts toward having looked at a candidate
                # before relaxing — exact match is still model-led.
                led.exact_seen = True
        except Exception:
            pass

    def _absorb(
        self,
        result: dict[str, Any],
        transcript: list[dict[str, Any]],
        calls: list[dict[str, Any]],
    ) -> None:
        try:
            led = self.ledger
            rows = result.get("calls") or []
            led.seen = []
            for index, call in enumerate(calls, start=1):
                # Every emitted call needs a matching tool message, even if the
                # environment returned fewer rows than we sent.
                row = rows[index - 1] if index - 1 < len(rows) else None
                observation = row.get("observation") if isinstance(row, dict) else None
                led.seen.append(observation)
                transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": _call_id(call, f"{self.key}-{led.turn}-{index}"),
                        "content": _digest(observation),
                    }
                )
                if _shifted(observation):
                    led.market_moved = True
                self._harvest(_call_name(call), observation)

            note = result.get("user_message")
            if isinstance(note, dict) and note.get("content"):
                spoken = str(note["content"])
                transcript.append({"role": "user", "content": spoken})
                # Any post-cart shopper note needs an ack; questions always do.
                if led.committed or _is_question(spoken):
                    led.owes_reply = True
                    if led.committed:
                        led.awaiting_stock_check = True

            detail = _fault(led.seen)
            if detail and self.shape.lane == SALVAGE:
                transcript.append(
                    {"role": "user", "content": _TIP_RECOVER.format(detail=detail)}
                )
            elif self.shape.lane == PODIUM and led.market_moved and not led.closed:
                transcript.append({"role": "user", "content": _TIP_RESORT})
        except Exception:
            pass

    @staticmethod
    def _finished(result: dict[str, Any]) -> bool:
        try:
            for row in result.get("calls") or []:
                if not isinstance(row, dict):
                    continue
                observation = row.get("observation")
                if isinstance(observation, dict) and observation.get("done"):
                    return True
            return False
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Tool-call helpers
# ---------------------------------------------------------------------------


def _call_name(call: dict[str, Any]) -> str:
    try:
        spec = call.get("function") if isinstance(call, dict) else None
        if isinstance(spec, dict) and spec.get("name"):
            return str(spec["name"])
        return ""
    except Exception:
        return ""


def _call_id(call: dict[str, Any], fallback: str) -> str:
    try:
        marker = call.get("id") if isinstance(call, dict) else None
        return str(marker) if marker else fallback
    except Exception:
        return ""


def _call_args(call: dict[str, Any]) -> dict[str, Any]:
    try:
        spec = call.get("function") if isinstance(call, dict) else None
        raw = spec.get("arguments", "{}") if isinstance(spec, dict) else "{}"
        if isinstance(raw, dict):
            parsed: Any = raw
        else:
            try:
                parsed = json.loads(raw or "{}")
            except json.JSONDecodeError:
                parsed = {}
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _digest(observation: Any, ceiling: int = 6000) -> str:
    try:
        try:
            rendered = json.dumps(observation, ensure_ascii=False, default=str)
        except TypeError:
            rendered = str(observation)
        if len(rendered) <= ceiling:
            return rendered
        return rendered[: ceiling - 18] + "…<clipped>"
    except Exception:
        return ""


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """ORO Bench entry point: run the generated environment until done=true."""
    dialogue: list[dict[str, Any]] = []
    try:
        dialogue = Episode(problem_data).run()
        return dialogue
    except Exception:
        return dialogue


__all__ = ["agent_main"]
