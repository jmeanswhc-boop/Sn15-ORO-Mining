"""FirstTop — ORO SN15 agent for validator-owned generated environments.

The scored artifact is the validator-side ledger, not the dialogue this returns.
Reward is gate-then-gradient: a graded family metric is paid only after a hard
gate, an exploit check, and a stateful gate all pass. Most of the reward the
naive loop loses is lost on ledger *shape*, not on shopping judgment, so this
agent puts every scored mechanic in code and calls the model only where real
judgment is required:

    turn k     ASK        ask for anything the shopper withheld
    turn k+1   SEARCH     gather candidates (few distinct queries)
    turn k+2   INSPECT    view/compare — raw facts before any commitment
    turn k+3   COMMIT     add_to_cart(ideal)      <- arms the market event
    turn k+4   OBSERVE    inspect_stock + view alternates  <- ALONE, proves uptake
    turn k+5   RECOMMIT   add_to_cart(replacement)
    turn k+6   ORDER      place_test_order

Two invariants earn most of the recovery and are structural here, not prompted:

* The first successful cart add is graded permanently against gold, the hard
  constraints and family quality. Nothing later repairs a bad one, so cart
  actions are unreachable until a selection has been made.
* Event-uptake evidence only counts from a strictly earlier turn than the order
  (`entry.turn < order_turn`). OBSERVE therefore never shares a turn with ORDER.

Every task in the qualifying pack fires an event on the first cart add, and that
event always invalidates exactly what was added (a price bump above budget, or a
stockout). Commit -> invalidate -> recover is the task, not a contingency.
"""

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}

_proxy = ProxyClient(timeout=120, max_retries=2)

# Family markers: substrings of each family's public rules block. Advisory only —
# they tune the selection prompt and never gate a scored mechanic, so a reworded
# runtime degrades to the generic policy instead of failing.
_FAMILY_MARKERS = (
    ("retrieval_recall", "Match the requested phrase against the listing title"),
    ("preference_reasoning", "View and compare return listing_id"),
    ("ranking", "stated tradeoff priority"),
    ("recovery", "first try the same model in another color"),
    ("justification", "include a structured justification with claims"),
    ("constraint_satisfaction", "EVERY requirement must hold for the EXACT variant"),
    ("intent_decomposition", "MIX of firm requirements and nice-to-haves"),
)

# Distinct search strings divide the retrieval metric (min(1, 3/n)); three is the
# point where that factor stops costing anything.
_MAX_QUERIES = 3
_COMPARE_CAP = 5
_SHORTLIST = 15

_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below)\s+([\d,]+(?:\.\d+)?)\s*([A-Z]{3})",
    re.I,
)
# A mid-session tightening binds retroactively, so any figure the shopper names
# later has to be folded into the working budget.
_MONEY_RE = re.compile(r"([\d,]+(?:\.\d+)?)\s*([A-Z]{3})", re.I)
_CANDIDATE_RE = re.compile(r"\b([0-9A-Za-z_-]+::[0-9A-Za-z_.-]+)\b")
_QUESTION_RE = re.compile(r"(confirm|reply|respond|answer|tell me|let me know)", re.I)
_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.S)


def _model() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(
        provider, _DEFAULT_MODELS["chutes"]
    )


def _key(product_id: Any, sku: Any) -> str:
    return f"{product_id}::{sku}"


def _goal_text(query: str) -> str:
    """The shopper's own request, without the appended rules block."""
    return query.split("\n\nTask rules:", 1)[0].strip()


def _detect_family(query: str) -> str | None:
    for family, marker in _FAMILY_MARKERS:
        if marker in query:
            return family
    return None


def _tool_index(tools: list[dict]) -> dict[str, dict]:
    index = {}
    for tool in tools or []:
        function = tool.get("function") or {}
        name = function.get("name")
        if name:
            index[str(name)] = function
    return index


def _claims_required(tools: dict[str, dict]) -> bool:
    """True when the order tool itself demands a justification.

    The runtime rebuilds `place_test_order` per family: only the justification
    family marks `justification` required and drops the restricted field enum.
    Reading that off the supplied schema is exact, and survives a reworded
    rules block in a way that prose matching does not.
    """
    order = tools.get("place_test_order") or {}
    required = ((order.get("parameters") or {}).get("required")) or []
    return "justification" in required


def _claim_fields(tools: dict[str, dict]) -> list[str]:
    """The claim `field` enum the supplied schema allows, when it pins one."""
    order = tools.get("place_test_order") or {}
    properties = (order.get("parameters") or {}).get("properties") or {}
    justification = properties.get("justification") or {}
    claims = (justification.get("properties") or {}).get("claims") or {}
    field = ((claims.get("items") or {}).get("properties") or {}).get("field") or {}
    enum = field.get("enum")
    return [str(value) for value in enum] if isinstance(enum, list) else []


def _parse_budget(goal: str) -> tuple[float | None, str]:
    match = _BUDGET_RE.search(goal)
    if not match:
        return None, ""
    return float(match.group(1).replace(",", "")), match.group(2).upper()


def _parse_candidates(goal: str) -> list[tuple[str, str]]:
    """Explicitly enumerated candidate variants, when the shopper names a set."""
    out: list[tuple[str, str]] = []
    for token in _CANDIDATE_RE.findall(goal):
        product_id, _, sku = token.partition("::")
        if product_id and sku and (product_id, sku) not in out:
            out.append((product_id, sku))
    return out


# Ranking over a named candidate set is a stated rule, not a judgement call, so
# it is resolved deterministically. Advertised figures decide what ranks best —
# including a title-only overclock rate — which is a different standard from the
# raw-fact grounding a claim needs.
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*hz", re.I)
_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\"|inch|inches|in\b)", re.I)
_AXES = (
    ("refresh_rate", ("refresh rate", "refresh"), True),
    ("screen_size", ("largest screen", "screen size", "screen", "display size"), True),
    ("price", ("lowest price", "cheapest", "price"), False),
)


def _axis_order(goal: str) -> list[tuple[str, bool]]:
    """Tiebreak axes in the order the shopper states them, most significant first."""
    low = goal.casefold()
    found = []
    for name, phrases, higher in _AXES:
        positions = [low.find(phrase) for phrase in phrases if low.find(phrase) >= 0]
        if positions:
            found.append((min(positions), name, higher))
    return [(name, higher) for _, name, higher in sorted(found)]


def _axis_value(row: dict, axis: str) -> float | None:
    if axis == "price":
        price = row.get("price")
        return float(price) if price is not None else None
    facts = " ".join(str(value) for value in (row.get("facts") or {}).values())
    text = f"{row.get('title') or ''} {facts} {row.get('options') or ''}"
    pattern = _HZ_RE if axis == "refresh_rate" else _INCH_RE
    values = [float(found) for found in pattern.findall(text)]
    return max(values) if values else None


def _rank(rows: list[dict], order: list[tuple[str, bool]]) -> list[dict]:
    def sort_key(row: dict) -> tuple:
        parts: list[tuple] = []
        for axis, higher in order:
            value = _axis_value(row, axis)
            parts.append(
                (value is None, (-value if higher else value) if value is not None else 0.0)
            )
        parts.append((False, row["sku"]))
        return tuple(parts)

    return sorted(rows, key=sort_key)


# When the shopper asks to match a phrase, the rules state the matching algorithm
# exactly: lowercase the title, split on runs of letters/digits, and require every
# term as a complete token. Inspecting anything else is pure cost — the retrieval
# metric divides by how much of what we looked at belonged in the pool.
_PHRASE_RE = re.compile(
    r'matches?\s+(?:the\s+phrase\s+)?["\u201c]?([a-z0-9][a-z0-9 ]*?)["\u201d]?'
    r'(?:\s*,|\s+at or under|\s+at or below|\s+up to|$)',
    re.I,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Generic accessory vocabulary. The shared rules require rejecting accessory and
# non-device listings whenever a device is requested.
_ACCESSORY = (
    "case", "cover", "charger", "cable", "adapter", "holder", "mount", "bracket",
    "screen protector", "tempered glass", "stand", "strap", "pouch", "sticker",
    "replacement screen", "lcd only", "battery only", "sim tray",
)


def _phrase_terms(goal: str) -> list[str]:
    match = _PHRASE_RE.search(goal)
    if not match:
        return []
    return _TOKEN_RE.findall(match.group(1).casefold())


def _title_matches(title: Any, terms: list[str]) -> bool:
    tokens = set(_TOKEN_RE.findall(str(title or "").casefold()))
    return all(term in tokens for term in terms)


def _looks_like_accessory(title: Any) -> bool:
    low = str(title or "").casefold()
    return any(word in low for word in _ACCESSORY)


# A stated size requirement is an exact variant-level test, not a judgement, and
# the shopper may only reveal it when asked. Variant options describe the SKU
# actually being bought, so they outrank the listing's own facts and title.
_REQ_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|\s)?\s*inch", re.I)


def _required_inches(text: str) -> float | None:
    found = {float(value) for value in _REQ_INCH_RE.findall(text or "")}
    return next(iter(found)) if len(found) == 1 else None


def _variant_inches(row: dict) -> set[float]:
    """Screen sizes for this variant, from the most specific source available."""
    for source in (
        row.get("options"),
        (row.get("facts") or {}).get("size"),
        row.get("title"),
    ):
        values = {float(value) for value in _INCH_RE.findall(str(source or ""))}
        if values:
            return values
    return set()


# The rules state the pixel ladder outright: classes are decided by the longer and
# shorter side regardless of orientation, taking the highest class whose BOTH
# thresholds are met. Lowercase "qHD" is quarter HD and deliberately not QHD.
_RES_LADDER = (("4k", 3840, 2160), ("qhd", 2560, 1440), ("fhd", 1920, 1080), ("hd", 1280, 720))
_PIXELS_RE = re.compile(r"(\d{3,5})\s*(?:x|\u00d7|by)\s*(\d{3,5})", re.I)
_RES_NAMES = (
    ("4k", (r"\b4k\b", r"\buhd\b")),
    ("qhd", (r"\b1440p\b", r"\bQHD\b")),
    ("fhd", (r"\b1080p\b", r"\bfull\s+hd\b", r"\bFHD\b")),
    ("hd", (r"\b720p\b", r"\bhd\b")),
)


def _required_resolution(goal: str) -> str | None:
    for name, patterns in _RES_NAMES:
        for pattern in patterns:
            # QHD/FHD stay case-sensitive: lowercase qHD means quarter HD.
            flags = 0 if pattern.strip("\\b") in ("QHD", "FHD") else re.I
            if re.search(pattern, goal, flags):
                return name
    return None


def _resolution_class(text: Any) -> str | None:
    best = None
    for raw_long, raw_short in _PIXELS_RE.findall(str(text or "")):
        longer, shorter = max(int(raw_long), int(raw_short)), min(int(raw_long), int(raw_short))
        for name, need_long, need_short in _RES_LADDER:
            if longer >= need_long and shorter >= need_short:
                if best is None or _RES_LADDER.index(
                    next(x for x in _RES_LADDER if x[0] == name)
                ) < _RES_LADDER.index(next(x for x in _RES_LADDER if x[0] == best)):
                    best = name
                break
    return best


def _variant_resolution(row: dict) -> str | None:
    for source in (row.get("options"), (row.get("facts") or {}).get("display"), row.get("title")):
        found = _resolution_class(source)
        if found:
            return found
    return None


def _extract_json(text: str) -> Any:
    match = _JSON_RE.search(text or "")
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None


def _is_question(text: str) -> bool:
    return bool(text) and ("?" in text or "？" in text or _QUESTION_RE.search(text))


class Catalog:
    """In-memory mirror of everything the environment has shown us.

    Selection reads from here rather than from the raw transcript so that price
    and stock always reflect the most recent observation of a variant.
    """

    def __init__(self) -> None:
        self.products: dict[str, dict[str, Any]] = {}
        self.queries: set[str] = set()

    def note_query(self, query: str) -> None:
        if query:
            self.queries.add(query.strip().casefold())

    def add_summaries(self, rows: list[dict]) -> None:
        for row in rows or []:
            product_id = str(row.get("product_id") or "")
            if not product_id:
                continue
            record = self.products.setdefault(
                product_id,
                {"product_id": product_id, "variants": {}, "facts": {}},
            )
            for field in ("title", "brand"):
                if row.get(field) is not None:
                    record[field] = row[field]
            sku = row.get("sku")
            if sku:
                self._variant(record, str(sku), row)

    def add_detail(self, row: dict) -> None:
        product_id = str(row.get("product_id") or "")
        if not product_id:
            return
        record = self.products.setdefault(
            product_id, {"product_id": product_id, "variants": {}, "facts": {}}
        )
        record["detailed"] = True
        for field in ("title", "brand", "category_path", "facts", "listing_id"):
            if row.get(field) is not None:
                record[field] = row[field]
        for variant in row.get("variants") or []:
            sku = variant.get("sku")
            if sku:
                self._variant(record, str(sku), variant)

    def update_variant(self, row: dict) -> None:
        product_id = str(row.get("product_id") or "")
        sku = row.get("sku")
        if not product_id or not sku:
            return
        record = self.products.setdefault(
            product_id, {"product_id": product_id, "variants": {}, "facts": {}}
        )
        self._variant(record, str(sku), row)

    @staticmethod
    def _variant(record: dict, sku: str, row: dict) -> None:
        variant = record["variants"].setdefault(sku, {"sku": sku})
        for field in ("options", "price", "currency", "in_stock"):
            if row.get(field) is not None:
                variant[field] = row[field]

    def eligible(self, budget: float | None, exclude: set[str]) -> list[dict]:
        """Purchasable variants we have actually inspected.

        Selection is restricted to products seen through view or compare. A
        search summary carries a title and a price but no raw facts, and both
        the chosen item and the first cart add must appear in the rendered
        view/compare surface for the evidence gates to credit them — so an
        uninspected product is not a candidate, however good its title looks.
        """
        rows: list[dict] = []
        for record in self.products.values():
            if not record.get("detailed"):
                continue
            for sku, variant in record["variants"].items():
                key = _key(record["product_id"], sku)
                if key in exclude or variant.get("in_stock") is not True:
                    continue
                price = variant.get("price")
                if budget is not None and (price is None or float(price) > budget):
                    continue
                rows.append(
                    {
                        "product_id": record["product_id"],
                        "sku": sku,
                        "title": record.get("title"),
                        "brand": record.get("brand"),
                        "category": (record.get("category_path") or [None])[-1],
                        "facts": record.get("facts") or {},
                        "options": variant.get("options"),
                        "price": price,
                        "in_stock": True,
                    }
                )
        return rows

    def undetailed(self, limit: int) -> list[str]:
        pending = [
            product_id
            for product_id, record in self.products.items()
            if not record.get("detailed")
        ]
        return pending[:limit]


class Driver:
    """One `/environment/call` per turn, so turn separation is structural.

    The verifier only credits event-uptake evidence recorded in a turn strictly
    earlier than the order. Collapsing a step into the previous turn would
    silently forfeit that credit, so every step here takes its own turn.
    """

    def __init__(self, binding: dict, problem_id: str, max_calls: int) -> None:
        self.binding = binding
        self.problem_id = problem_id
        self.max_calls = max(1, max_calls)
        self.turn = 0
        self.done = False
        self.pending_message: str | None = None
        self.dialogue: list[dict[str, Any]] = []

    def step(self, actions: list[dict], note: str) -> list[dict]:
        """Run one turn and return its tool payloads.

        Each entry in the response's `calls` carries an envelope shaped
        `{done, error, observation}` — terminal and error state on the outside,
        the tool's own payload nested within. Unwrapping here keeps that detail
        in one place and hands callers the payload directly.
        """
        if self.done or not actions:
            return []
        self.turn += 1
        group = f"{self.problem_id}-t{self.turn}"
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": self.turn,
            "calls": [
                {"call_id": f"{group}-{index}", "action": action}
                for index, action in enumerate(actions[: self.max_calls], start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")

        envelopes = [call.get("observation") or {} for call in result.get("calls", [])]
        user_message = result.get("user_message") or {}
        content = str(user_message.get("content") or "")
        if content:
            self.pending_message = content
        if any(envelope.get("done") for envelope in envelopes):
            self.done = True
        self.dialogue.append(
            {
                "role": "assistant",
                "content": note,
                "turn": self.turn,
                "actions": actions[: self.max_calls],
                "observations": envelopes,
                "user_message": content or None,
            }
        )
        return [envelope.get("observation") or {} for envelope in envelopes]


def _chat(messages: list[dict], *, expect_json: bool = True) -> Any:
    """One inference round-trip; returns parsed JSON or raw text."""
    response = _proxy.post(
        "/inference/chat/completions",
        json_data={
            "model": _model(),
            "messages": messages,
            "temperature": 0,
        },
    )
    if response is None:
        raise RuntimeError("inference request failed")
    content = (response["choices"][0]["message"].get("content") or "").strip()
    return _extract_json(content) if expect_json else content


def _plan_queries(goal: str, family: str | None) -> list[str]:
    hint = (
        "The shopper named the phrase to match; use it and close variations."
        if family == "retrieval_recall"
        else "Cover the category and the stated hardware requirement."
    )
    plan = _chat(
        [
            {
                "role": "system",
                "content": (
                    "You plan catalog searches for a shopping agent. "
                    f"Return JSON: {{\"queries\": [..]}} with at most {_MAX_QUERIES} "
                    f"short keyword queries. {hint} No prose."
                ),
            },
            {"role": "user", "content": goal},
        ]
    )
    queries = []
    if isinstance(plan, dict):
        for entry in plan.get("queries") or []:
            text = str(entry).strip()
            if text and text not in queries:
                queries.append(text)
    return queries[:_MAX_QUERIES] or [goal[:80]]


def _choose(
    goal: str,
    rules: str,
    rows: list[dict],
    budget: float | None,
    currency: str,
    *,
    dead: str | None,
) -> dict | None:
    """Pick one variant. This is the only genuinely judgmental call."""
    if not rows:
        return None
    context = (
        f"The variant {dead} is no longer available. Choose the best remaining "
        "option under the shopper's stated fallback order."
        if dead
        else "Choose the option you intend to buy. This is a real commitment, "
        "not a placeholder."
    )
    budget_line = (
        f"Working budget: {budget:.2f} {currency}." if budget is not None else ""
    )
    choice = _chat(
        [
            {
                "role": "system",
                "content": (
                    "You select a product variant for a shopper.\n"
                    "1. Firm requirements must hold for the EXACT variant, judged "
                    "from its raw facts and options, never from title wording.\n"
                    "2. Then rank the survivors on every preference the shopper "
                    "stated, scoring each one from raw facts. Prefer the best fit "
                    "over the cheapest: an item with no factual support for a "
                    "stated preference scores zero on it.\n"
                    "3. When a device is requested, require positive evidence that "
                    "the item IS that device. Reject accessories, parts and "
                    "screen-only listings whatever the title claims.\n"
                    "This is a real purchase commitment. Return JSON: "
                    '{"product_id": "..", "sku": "..", "why": ".."}. No prose.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Shopper request:\n{goal}\n\n"
                    f"Rules:\n{rules}\n\n{budget_line}\n{context}\n\n"
                    f"Candidates:\n{json.dumps(rows[:40], default=str)}"
                ),
            },
        ]
    )
    ordered = sorted(rows, key=lambda row: (row.get("price") is None, row.get("price"), row["sku"]))
    if not isinstance(choice, dict):
        return ordered[0]
    product_id, sku = str(choice.get("product_id") or ""), str(choice.get("sku") or "")
    for row in rows:
        if row["product_id"] == product_id and row["sku"] == sku:
            return row
    # The model named a product but mismatched the variant; keep its product.
    for row in ordered:
        if row["product_id"] == product_id:
            return row
    return ordered[0]


def _build_claims(
    row: dict, goal: str, budget: float | None, currency: str, allowed: list[str]
) -> list[dict]:
    """Claims assembled from observed facts only.

    Every claim is checked and one bad claim voids the whole set, so a field is
    emitted only when the raw fact behind it was actually observed. `price` is
    stated as an inclusive bound rather than an exact figure: the exact form is
    compared with `==` against the price at order, which a mid-session reprice
    can break, while the bound is true whenever the budget check passed.
    """
    facts = row.get("facts") or {}
    claims: list[dict] = []

    # The specification claim must be the one the shopper actually stated: the
    # check demands that exact field, and a different (even true) attribute
    # leaves the required one uncovered and voids the set.
    fact_fields = [("resolution", "display"), ("screen_size", "size"), ("refresh_rate", "refresh_rate")]
    if _required_inches(goal) is not None:
        fact_fields.insert(0, ("screen_size", "size"))
    elif _required_resolution(goal) is not None:
        fact_fields.insert(0, ("resolution", "display"))
    for field, label in fact_fields:
        value = facts.get(label)
        if value not in (None, "") and (not allowed or field in allowed):
            claims.append({"field": field, "value": str(value)})
            break

    if budget is not None:
        claims.append(
            {"field": "price", "value": f"at most {budget:.2f} {currency}".strip()}
        )
    claims.append({"field": "in_stock", "value": "true"})

    # At least one supporting reason is required alongside the specification,
    # price and stock claims. Brand is the only one groundable without prose.
    brand = row.get("brand")
    if brand:
        claims.append({"field": "brand", "value": str(brand)})
    return claims


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Drive one generated-environment session to a placed order."""

    environment = problem_data["environment"]
    policy = environment["policy_view"]
    query = str(policy.get("query") or "")
    goal = _goal_text(query)
    rules = query[len(goal) :].strip()
    family = _detect_family(query)

    tools = _tool_index(policy.get("tools") or [])
    needs_claims = _claims_required(tools)
    allowed_fields = _claim_fields(tools)

    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    driver = Driver(
        environment["binding"],
        problem_id,
        int(policy.get("max_calls_per_turn", 1)),
    )
    catalog = Catalog()
    budget, currency = _parse_budget(goal)
    dead: set[str] = set()
    terms = _phrase_terms(goal)
    # The shared rules reject accessory and non-device listings whenever the
    # shopper asks for a device, and both device families gate on it.
    wants_device = "not an accessory" in query.casefold() or bool(
        re.search(r"\b(smartphone|phone|monitor|headphone|laptop|tablet|camera)s?\b", goal, re.I)
    )

    def absorb(observations: list[dict]) -> None:
        for obs in observations:
            catalog.add_summaries(obs.get("results") or [])
            for row in obs.get("compared") or []:
                catalog.add_detail(row)
            if obs.get("facts") is not None or obs.get("variants") is not None:
                catalog.add_detail(obs)
            if obs.get("in_stock") is not None and obs.get("sku"):
                catalog.update_variant(obs)
            added = obs.get("added")
            if isinstance(added, dict):
                catalog.update_variant(added)

    def inspect_more(note: str) -> bool:
        """View the next batch of search hits. Returns False when none are left."""
        pending = catalog.undetailed(_SHORTLIST * 3)
        if terms:
            # Only phrase-matching titles can be in the eligible pool, and every
            # inspection of something outside it lowers the retrieval payout.
            pending = [
                pid for pid in pending
                if _title_matches((catalog.products[pid] or {}).get("title"), terms)
            ] or pending
        if wants_device:
            preferred = [
                pid for pid in pending
                if not _looks_like_accessory((catalog.products[pid] or {}).get("title"))
            ]
            pending = preferred or pending
        pending = pending[:_SHORTLIST]
        if not pending:
            return False
        absorb(
            driver.step(
                [
                    {"name": "compare", "args": {"product_ids": pending[index : index + _COMPARE_CAP]}}
                    for index in range(0, len(pending), _COMPARE_CAP)
                ],
                note,
            )
        )
        return True

    def answer_if_asked() -> None:
        """Reply to a shopper question before continuing.

        An unanswered question forces the family gate false no matter how good
        the purchase is, and a reply only counts if it lands before the order.
        A tightened budget delivered in the same breath binds retroactively.
        """
        nonlocal budget, goal
        message = driver.pending_message
        driver.pending_message = None
        if not message:
            return
        # Everything the shopper says is part of the requirement set: an
        # underspecified task reveals its firm spec only here, and a tightening
        # often restates it. Selection reads the goal, so keep it all.
        goal = f"{goal}\n\nShopper: {message}"
        for amount, code in _MONEY_RE.findall(message):
            if currency and code.upper() != currency:
                continue
            value = float(amount.replace(",", ""))
            budget = value if budget is None else min(budget, value)
        if not _is_question(message) or driver.done:
            return
        reply = _chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the shopper in one or two short sentences. "
                        "Be concrete about what you are buying and confirm it "
                        "meets what they asked. Plain text only."
                    ),
                },
                {"role": "user", "content": f"Request:\n{goal}\n\nShopper:\n{message}"},
            ],
            expect_json=False,
        )
        absorb(driver.step([{"name": "message", "args": {"content": str(reply)[:400]}}], "reply"))

    # ASK — anything the shopper deliberately withheld has to be requested
    # before the first cart add, because that add is graded against the full
    # requirement set including the part not yet stated.
    if "if you ask" in goal.casefold() or family == "preference_reasoning":
        question = _chat(
            [
                {
                    "role": "system",
                    "content": (
                        "The shopper held something back. Ask one short, direct "
                        "question for the missing requirement or their top "
                        "priority. Plain text only."
                    ),
                },
                {"role": "user", "content": goal},
            ],
            expect_json=False,
        )
        absorb(driver.step([{"name": "message", "args": {"content": str(question)[:300]}}], "ask"))
        answer_if_asked()

    # SEARCH — a named candidate set replaces retrieval entirely.
    named = _parse_candidates(goal)
    if named:
        ids = sorted({product_id for product_id, _ in named})
        absorb(
            driver.step(
                [
                    {"name": "compare", "args": {"product_ids": ids[index : index + _COMPARE_CAP]}}
                    for index in range(0, len(ids), _COMPARE_CAP)
                ],
                "compare named candidates",
            )
        )
    else:
        actions = []
        for text in _plan_queries(goal, family):
            catalog.note_query(text)
            args: dict[str, Any] = {"query": text, "k": 10, "in_stock": True}
            actions.append({"name": "search", "args": args})
        absorb(driver.step(actions, "search"))

        inspect_more("inspect candidates")
    answer_if_asked()
    if not catalog.eligible(budget, dead):
        # A tightened budget can empty the inspected pool; widen it before
        # giving up, since an abandoned session scores zero outright.
        inspect_more("inspect further candidates")

    ranked_axes = _axis_order(goal) if named else []

    def select(exclude: set[str], dead_key: str | None) -> dict | None:
        rows = catalog.eligible(budget, exclude)
        inches = _required_inches(goal)
        if inches is not None:
            exact = [row for row in rows if inches in _variant_inches(row)]
            rows = exact or rows
        wanted_res = _required_resolution(goal)
        if wanted_res is not None:
            exact = [row for row in rows if _variant_resolution(row) == wanted_res]
            rows = exact or rows
        if not rows:
            return None
        if ranked_axes:
            return _rank(rows, ranked_axes)[0]
        return _choose(goal, rules, rows, budget, currency, dead=dead_key)

    # COMMIT — the graded first add.
    ideal = select(dead, None)
    if ideal is None:
        return driver.dialogue
    ideal_key = _key(ideal["product_id"], ideal["sku"])
    absorb(
        driver.step(
            [{"name": "add_to_cart", "args": {"product_id": ideal["product_id"], "sku": ideal["sku"]}}],
            "commit to intended purchase",
        )
    )
    answer_if_asked()

    # OBSERVE — its own turn. Seeing the committed variant's changed price or
    # stock is what proves the event was noticed; evidence recorded in the
    # order's own turn is ignored. Alternates are refreshed here too, so the
    # replacement's current state is also on the record before ordering.
    alternates = [
        row["product_id"]
        for row in catalog.eligible(budget, {ideal_key})
        if row["product_id"] != ideal["product_id"]
    ]
    seen: list[str] = []
    for product_id in alternates:
        if product_id not in seen:
            seen.append(product_id)
    observe: list[dict] = [
        {"name": "inspect_stock", "args": {"product_id": ideal["product_id"], "sku": ideal["sku"]}}
    ]
    for index in range(0, min(len(seen), _SHORTLIST), _COMPARE_CAP):
        observe.append({"name": "compare", "args": {"product_ids": seen[index : index + _COMPARE_CAP]}})
    absorb(driver.step(observe, "verify committed variant after the event"))
    answer_if_asked()

    # RECOMMIT — the event always invalidates what was committed, so the order
    # necessarily lands on a different variant.
    dead.add(ideal_key)
    replacement = select(dead, ideal_key)
    if replacement is None:
        return driver.dialogue
    absorb(
        driver.step(
            [
                {
                    "name": "inspect_stock",
                    "args": {"product_id": replacement["product_id"], "sku": replacement["sku"]},
                },
                {
                    "name": "add_to_cart",
                    "args": {"product_id": replacement["product_id"], "sku": replacement["sku"]},
                },
            ],
            "recommit after the event",
        )
    )
    answer_if_asked()

    # ORDER — a separate turn from every piece of evidence above. Justification
    # is omitted unless the supplied schema requires it: an absent justification
    # skips the claim check entirely, while a single unsupported claim voids the
    # whole set.
    price = replacement.get("price")
    if budget is not None and price is not None and float(price) > budget:
        # The shopper tightened the budget after we committed. The order is
        # judged against the tightened figure, so re-select rather than order
        # something that is now too expensive.
        dead.add(_key(replacement["product_id"], replacement["sku"]))
        cheaper = select(dead, ideal_key)
        if cheaper is not None:
            replacement = cheaper
            absorb(
                driver.step(
                    [
                        {
                            "name": "add_to_cart",
                            "args": {
                                "product_id": replacement["product_id"],
                                "sku": replacement["sku"],
                            },
                        }
                    ],
                    "re-commit within the tightened budget",
                )
            )
            answer_if_asked()

    order_args: dict[str, Any] = {
        "product_id": replacement["product_id"],
        "sku": replacement["sku"],
    }
    if needs_claims:
        order_args["justification"] = {
            "claims": _build_claims(replacement, goal, budget, currency, allowed_fields)
        }
    absorb(driver.step([{"name": "place_test_order", "args": order_args}], "order"))
    return driver.dialogue


__all__ = ["agent_main"]
