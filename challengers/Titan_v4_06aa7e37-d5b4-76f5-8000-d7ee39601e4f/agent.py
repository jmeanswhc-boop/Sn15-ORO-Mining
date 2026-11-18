from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------
# Inference engines. OpenRouter ids only.
# https://oroagents.com/docs/miners/agent-interface
# ---------------------------------------------------------------------------

_ENGINE_QUICK = "deepseek/deepseek-v3.2"
_ENGINE_DEEP = "moonshotai/kimi-k2.6"
_ENGINE_RELIEF = (
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "google/gemma-4-31b-it",
)

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

# Lanes that reward slower, more careful reasoning.
_DEEP_LANES = frozenset({SPLIT, LOCK, TASTE, PODIUM, SALVAGE, VOUCH})

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
                return "true" if text.lower() in {"true", "1", "yes", "in_stock"} else text
            if field == "resolution":
                return ClaimDesk._tidy_resolution(text)
            return text
        except Exception:
            return ""

    @staticmethod
    def _tidy_resolution(text: str) -> str:
        try:
            squeezed = text.lower().replace(" ", "")
            pair = re.search(r"(\d{3,4})\D+(\d{3,4})", text)
            if pair:
                return f"{pair.group(1)} x {pair.group(2)}"
            if squeezed in {"hd", "720p", "720phd"}:
                return "hd"
            if squeezed in {"fullhd", "fhd", "1080p", "1080pfullhd"}:
                return "1920 x 1080"
            if squeezed in {"qhd", "1440p"}:
                return "2560 x 1440"
            if squeezed in {"uhd", "4k", "2160p"}:
                return "3840 x 2160"
            return text
        except Exception:
            return ""

    @staticmethod
    def polish(claims: Any, shape: TaskShape) -> list[dict[str, str]]:
        try:
            if not isinstance(claims, list):
                return []
            goal = shape.goal.lower()
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
                if field in _CLAIM_SUPPORT and field != "brand" and field not in goal:
                    # A capability the shopper never raised is not evidence.
                    continue
                if field in _CLAIM_SUPPORT:
                    if support_taken:
                        continue
                    support_taken = True
                if field in used:
                    continue
                used.add(field)
                kept.append({"field": field, "value": value})
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
episode that never orders scores nothing.
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
1 Work only from the variants the shopper listed. Open them directly instead
  of spending steps on open search.
2 Sort strictly by the priority the shopper stated, in the order stated. Add
  no criteria of your own.
3 Read the facts before the first commit, then commit to the current leader.
4 If availability or price shifts, sort again from scratch across whatever is
  still live and commit to the new leader.
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
  shopper asked for, plus price and in_stock, plus at most one supporting
  quality the facts actually show.
4 Never send a label the catalog does not use as a field — no category path,
  no operating-system tag, no marketing string, no prose.
5 Every value must equal what you observed.
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
    "On place_test_order the claims must use catalog attribute names — the "
    "shopper's figure plus price and in_stock, and at most one supporting "
    "quality you actually observed. No category path, no OS tag, no prose."
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
        except Exception:
            pass

    @property
    def steps_left(self) -> int:
        try:
            return max(0, self.limit - self.turn)
        except Exception:
            return 0

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
            if "add_to_cart" in labels:
                self.committed = True
            if "place_test_order" in labels:
                self.closed = True
            if "message" in labels:
                self.owes_reply = False
                self.asked_for_figure = True
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


_bridge = Bridge(ProxyClient(timeout=120, max_retries=2))


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
                return _TIP_ANSWER

            if (
                lane == LOCK
                and turn == 1
                and self.shape.held_back_figure
                and not led.asked_for_figure
            ):
                return _TIP_ASK_FIRST

            if lane == PODIUM and not led.closed:
                if led.market_moved and not led.did("place_test_order"):
                    return _TIP_RESORT
                if led.steps_left <= 3:
                    return _TIP_CLOSE_PODIUM
                if turn == 1:
                    return (
                        "Open the listed variants, sort them by the stated priority, "
                        "and only then commit. Re-sort after any market move. The "
                        "order carries product_id and sku only."
                    )

            if turn == 1 and lane == SPLIT:
                return _TIP_BRAND_FIRST

            if turn <= 2 and not any(
                led.ever(label)
                for label in ("search", "filter", "view", "compare", "message")
            ):
                return _TIP_OPEN_WITH_SEARCH

            if lane == VOUCH and not led.closed and turn in (2, 5):
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
                and led.steps_left <= 2
                and led.once("deadline")
            ):
                return _TIP_DEADLINE.format(left=led.steps_left)

            return None
        except Exception:
            return None

    # -- outbound arguments ----------------------------------------------

    def _shape_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if name != "place_test_order":
                return args
            payload = dict(args)
            if not self.shape.order_may_carry_claims():
                # Optional claims that cannot be verified only create risk.
                payload.pop("justification", None)
                return payload
            note = payload.get("justification")
            if not isinstance(note, dict):
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
                args = self._shape_args(name, _call_args(call))
                spec = call.get("function")
                if isinstance(spec, dict):
                    spec["arguments"] = json.dumps(args)
                actions.append(
                    {"call_id": f"{batch}-{slot}", "action": {"name": name, "args": args}}
                )
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

    # -- main loop -------------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        trail: list[dict[str, Any]] = []
        try:
            transcript = self._opening_transcript()
            trail: list[dict[str, Any]] = [
                {
                    "role": "meta",
                    "flowergirl": {
                        "version": 5,
                        "lane": self.shape.lane,
                        "engine": self.engine,
                        "provider": "openrouter",
                    },
                }
            ]

            led = self.ledger
            while led.turn < led.limit:
                led.turn += 1
                tip = self._guidance()
                if tip:
                    transcript.append({"role": "user", "content": tip})

                spoken, calls = self._speak(transcript, trail)
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
                        "flowergirl_turn": {"lane": self.shape.lane, "turn": led.turn},
                    }
                )

                led.note_actions([_call_name(call) or "?" for call in calls])
                self._absorb(result, transcript, calls)

                if self._finished(result):
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

            note = result.get("user_message")
            if isinstance(note, dict) and note.get("content"):
                spoken = str(note["content"])
                transcript.append({"role": "user", "content": spoken})
                if _is_question(spoken):
                    led.owes_reply = True

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
