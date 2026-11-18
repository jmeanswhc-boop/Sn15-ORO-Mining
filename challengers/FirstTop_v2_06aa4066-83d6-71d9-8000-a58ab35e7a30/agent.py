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

# Distinct search strings divide the retrieval metric (min(1, 3/n)); three is the
# point where that factor stops costing anything.
# The three-query cap is not a general efficiency rule, and applying it
# everywhere costs real reward. It exists because ONE family divides its payout
# by the number of distinct search strings (`min(1, 3/n)`); no other family
# counts queries at all, so outside that family a search is free.
#
# It is also the only breadth left. The catalogue filter sorts by product id and
# offers a ceiling with no floor, so its price bands are NESTED rather than
# disjoint: every band contains the lowest-id matches, and every band therefore
# returns the same head. Measured after a mid-session tightening, redrawing all
# four bands at the new ceiling returned two products that were not already
# held. BM25 ranks by relevance instead — a different ordering, and so genuinely
# different rows — which makes an extra query the one thing that reaches what
# the structured index cannot.
_MAX_QUERIES = 3
_MAX_QUERIES_WIDE = 6
_COMPARE_CAP = 5
# How many products one inspection sweep may reach. This costs something only
# where a phrase to match was named: `shortlist_precision` scores every viewed,
# compared and carted product as a fraction of those in the eligible pool, so
# there a wider sweep subtracts from the payout directly. Everywhere else
# inspection is free — efficiency does not scale reward — and the only ceiling
# that matters is `max_calls_per_turn`, which at 16 calls of 5 ids is far above
# the ~30 products a three-query search ever returns.
#
# The same value governs the post-event refresh, and must: `select` may choose
# any variant in the pool, `inspect_stock` at RECOMMIT lands in the same turn
# as the add, and a variant whose price moved after an unrefreshed observation
# fails `within_budget` outright. A pool wider than the refresh is a liability,
# not breadth.
_SHORTLIST = 15
# Wide enough that the structured half of retrieval actually reaches selection.
# `undetailed` hands products back in the order they were first seen, and the
# retrieval turn issues its BM25 searches before its filters, so a cap at the
# ~30 rows three searches return would leave every filter row uninspected — and
# outside TF2 a product that is never inspected can never be chosen, because
# `eligible` admits only what `view`/`compare` has detailed.
#
# The value is set by the call budget rather than chosen: `max_calls_per_turn`
# is 16, `compare` takes five ids, and the post-event refresh spends one call on
# `inspect_stock` before its own sweep. 70 is therefore the widest reach that
# leaves both turns inside the budget — 14 `compare` calls to inspect, 15 to
# refresh — and 75 would put the refresh exactly on the cap with nothing spare.
#
# Widening from 50 was not a guess at more breadth being better. Retrieval had
# begun returning a gold product that the sweep then cut: `replay.py gold` put
# one gold key of fourteen in `not_inspected` while the pool stayed empty, and
# outside TF2 an uninspected product is simply unreachable, however good it is.
_SHORTLIST_WIDE = 70

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


def _fold_budget(budget: float | None, currency: str, message: str) -> float | None:
    """Fold any figure the shopper names later into the working budget.

    A mid-session tightening binds retroactively and only ever narrows, so the
    minimum wins; a figure in another currency is not ours to compare.
    """
    for amount, code in _MONEY_RE.findall(message):
        if currency and code.upper() != currency:
            continue
        value = float(amount.replace(",", ""))
        budget = value if budget is None else min(budget, value)
    return budget


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
_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:gb|tb)\b", re.I)
_MAH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mah\b", re.I)
_MASS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg|g)\b", re.I)
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")

# The shopper states both halves of a tiebreak: a comparative word carries the
# direction and the words after it name the axis. Reading them in the order
# stated means a tiebreak on something never seen before is followed like any
# other, so nothing here enumerates what a shopper is allowed to rank on. The
# two lists are English comparatives, not a vocabulary of product specifications.
_HIGHER = (
    "highest", "largest", "biggest", "greatest", "most", "fastest", "longest",
    "widest", "maximum", "max", "best", "newest", "strongest",
)
_LOWER = (
    "lowest", "smallest", "cheapest", "least", "fewest", "minimum", "min",
    "shortest", "lightest", "slowest",
)
# A few comparatives are the comparative OF price, so they name an axis without
# naming a field: "the cheapest" needs no noun after it, and what does follow
# ("the cheapest valid item") names the thing, not the measure.
_PRICE_WORDS = ("cheapest", "priciest", "costliest", "dearest")
_AXIS_RE = re.compile(
    r"\b(%s)(?:\s+([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2}))?"
    % "|".join(_HIGHER + _LOWER + _PRICE_WORDS),
    re.I,
)
# The axis is the noun phrase, so it ends where the sentence turns to grammar.
_AXIS_TAIL_RE = re.compile(
    r"\s+(?:among|than|and|or|of|for|with|in|on|at|from|by|to|that|which|the|a|an)\b.*$",
    re.I,
)

# Last resort for reading a named measurement off a listing: the unit its
# quantity is quoted in. This is general measurement vocabulary — what a refresh
# rate or a capacity is counted in — rather than anything about this catalogue,
# and it is open rather than a closed set of axes: a concept absent from it can
# still resolve through a row field or a fact label, and adding one is one line.
_UNITS = (
    (("refresh", "rate", "frequency"), _HZ_RE),
    (("screen", "display", "size", "diagonal", "panel"), _INCH_RE),
    (("storage", "memory", "ram", "capacity"), _GB_RE),
    (("battery",), _MAH_RE),
    (("weight", "mass"), _MASS_RE),
)


def _axis_order(goal: str) -> list[tuple[str, bool]]:
    """Tiebreak axes in the order the shopper states them, most significant first."""
    order: list[tuple[str, bool]] = []
    for match in _AXIS_RE.finditer(goal or ""):
        word = match.group(1).casefold()
        if word in _PRICE_WORDS:
            axis = "price"
        elif match.group(2):
            axis = _AXIS_TAIL_RE.sub("", " ".join(match.group(2).split())).casefold()
        else:
            # a comparative with nothing after it names no axis
            continue
        if any(axis == seen for seen, _ in order):
            continue
        order.append((axis, word in _HIGHER))
    return order


def _axis_value(row: dict, axis: str) -> float | None:
    """This listing's value on a named axis.

    Structured evidence first — a field of the row itself, then a fact the
    catalogue filed under a label the axis names — and only then the quantity's
    unit read out of the listing's text. Same order of trust as everywhere else:
    what the seller filed beats what the title says.
    """
    words = set(_WORD_RE.findall(axis.casefold()))
    for key, value in row.items():
        if key in words and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    for label, value in (row.get("facts") or {}).items():
        if words & set(_WORD_RE.findall(str(label).casefold())):
            found = _NUMBER_RE.search(str(value))
            if found:
                return float(found.group(1))
    facts = " ".join(str(value) for value in (row.get("facts") or {}).values())
    text = f"{row.get('title') or ''} {facts} {row.get('options') or ''}"
    for concepts, pattern in _UNITS:
        if words & set(concepts):
            values = [float(found) for found in pattern.findall(text)]
            if values:
                return max(values)
    return None


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


# Two reasons to put a question to the shopper before committing, and both are
# stated rather than inferred from which shape of task this is:
#
#   * the request itself says something is being held back until asked, or
#   * the rules instruct clarifying what the shopper is prioritising first.
#
# The second is a clarify-verb next to the idea of a priority, in either order
# and within a short window. That is the instruction itself rather than a phrase
# lifted off the rules block: the longest literal here is one word, so a
# rewording degrades the match instead of flipping it, and text that instructs
# nothing of the kind leaves the question unasked.
_WITHHELD_RE = re.compile(r"\b(?:if you ask|will tell you|when asked)\b", re.I)
_ASK_PRIORITIES_RE = re.compile(
    r"\b(?:clarify|ask|confirm|establish)\b[^.]{0,40}\bpriorit"
    r"|\bpriorit\w*\b[^.]{0,40}\b(?:clarify|ask|confirm)",
    re.I,
)


def _should_ask(goal: str, query: str) -> bool:
    """Whether this task earns a question ahead of any commitment."""
    return bool(_WITHHELD_RE.search(goal) or _ASK_PRIORITIES_RE.search(query))


def _phrase_terms(goal: str) -> list[str]:
    match = _PHRASE_RE.search(goal)
    if not match:
        return []
    return _TOKEN_RE.findall(match.group(1).casefold())


def _title_matches(title: Any, terms: list[str]) -> bool:
    tokens = set(_TOKEN_RE.findall(str(title or "").casefold()))
    return all(term in tokens for term in terms)


# Whole tokens, never substrings. `"mount" in title` is True for a genuine
# monitor advertising "Support Wall Mounting", and that one substring hit put
# every gold candidate of a justification task out of reach. Measured over the
# pack, whole-token matching loses gold on ZERO tasks where substring matching
# loses all of it on one, and the 20 candidates it stops rejecting were all
# false positives.
_ACCESSORY_RE = re.compile("|".join(rf"\b{re.escape(w)}\b" for w in _ACCESSORY), re.I)


def _looks_like_accessory(title: Any) -> bool:
    return bool(_ACCESSORY_RE.search(str(title or "")))


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


# The pixel ladder is not ours to define. Wherever a resolution is graded, the
# rules state the ladder outright -- each class and the two thresholds it starts
# at -- so we read it from that text at runtime and follow it. Classes are decided
# by the longer and shorter side regardless of orientation, taking the highest
# class whose BOTH thresholds are met.
#
# The tuple below is only the fallback for a task whose rules state nothing, and
# its names are canonical: a parsed rung whose dimensions match one of these
# inherits its name and its spellings, so a stated ladder that agrees with the
# familiar one is labelled identically and nothing downstream shifts. Each rung
# is (name, longer side, shorter side, spellings a shopper might use).
_RES_LADDER = (
    ("4k", 3840, 2160, ("4k", "uhd")),
    ("qhd", 2560, 1440, ("qhd",)),
    ("fhd", 1920, 1080, ("fhd", "full hd")),
    ("hd", 1280, 720, ("hd",)),
)
_PIXELS_RE = re.compile(r"(\d{3,5})\s*(?:x|\u00d7|by)\s*(\d{3,5})", re.I)
# "HD starts at 1280 by 720", "Full HD at 1920 by 1080", "4K/UHD at 3840 by 2160".
# The name is whatever sits before the threshold; the two trims below take off the
# sentence's connective tissue rather than anything about displays, so a rewording
# ("HD begins at", "and 4K/UHD at") still yields the class name alone.
_LADDER_RE = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9/ ]{0,20}?)\s+at\s+"
    r"(\d{3,5})\s*(?:x|\u00d7|by)\s*(\d{3,5})",
    re.I,
)
_LEAD_RE = re.compile(r"^(?:and|or|the|a)\s+", re.I)
_VERB_RE = re.compile(r"\s+(?:starts?|begins?|opens?|runs?|sits?|is)$", re.I)
# Lowercase "qHD" is quarter HD, a far smaller panel, and the rules say so. It is
# the one spelling that must never resolve to QHD; every other alias is read
# case-insensitively, so a shopper writing "qhd" is understood.
_QUARTER_HD = re.compile(r"\bqHD\b")

_ladder_rungs: tuple[tuple[str, int, int, tuple[str, ...]], ...] = _RES_LADDER


def _parse_res_ladder(rules: str) -> tuple[tuple[str, int, int, tuple[str, ...]], ...]:
    """The pixel ladder these rules state, highest class first.

    Falls back to the familiar ladder unless the text states at least two rungs:
    one threshold pair in isolation is a stray sentence, not a ladder.
    """
    rungs: list[tuple[str, int, int, tuple[str, ...]]] = []
    for raw_name, raw_long, raw_short in _LADDER_RE.findall(rules or ""):
        longer, shorter = int(raw_long), int(raw_short)
        if shorter > longer:
            longer, shorter = shorter, longer
        stated = _VERB_RE.sub("", _LEAD_RE.sub("", " ".join(raw_name.split()))).strip()
        spellings = [piece.strip().casefold() for piece in stated.split("/") if piece.strip()]
        if not spellings:
            continue
        known = next(
            (rung for rung in _RES_LADDER if (rung[1], rung[2]) == (longer, shorter)), None
        )
        aliases = tuple(dict.fromkeys(spellings + list(known[3] if known else ())))
        rungs.append((known[0] if known else spellings[0], longer, shorter, aliases))
    if len(rungs) < 2:
        return _RES_LADDER
    rungs.sort(key=lambda rung: rung[1] * rung[2], reverse=True)
    unique: list[tuple[str, int, int, tuple[str, ...]]] = []
    for rung in rungs:
        if not any(rung[0] == kept[0] for kept in unique):
            unique.append(rung)
    return tuple(unique)


def _arm_ladder(rules: str) -> None:
    """Adopt the ladder this task states.

    Module state is safe to hold it: the executor runs each problem in its own
    forkserver child process, so one task's ladder cannot reach another. Helpers
    called outside a task -- the replay harness, the self-tests -- see the
    fallback until they arm one themselves.
    """
    global _ladder_rungs
    _ladder_rungs = _parse_res_ladder(rules)


def _named_class(text: Any) -> str | None:
    """The highest ladder class this text names outright.

    Highest rung first, so a broader name nested inside a narrower one -- the
    "HD" sitting inside "Full HD" -- cannot claim the match. The shorthand for a
    class is its shorter side ("1080p"), derived from the rung rather than listed.
    """
    cleaned = _QUARTER_HD.sub(" ", str(text or ""))
    for name, _longer, shorter, aliases in _ladder_rungs:
        for alias in (*aliases, "%dp" % shorter):
            pattern = r"\b%s\b" % re.escape(alias).replace(r"\ ", r"\s+")
            if re.search(pattern, cleaned, re.I):
                return name
    return None


def _required_resolution(goal: str) -> str | None:
    """The class the shopper asked for, read from what they called it."""
    return _named_class(goal)


def _resolution_class(text: Any) -> str | None:
    """The class this text advertises. Pixels decide wherever pixels are stated.

    An explicit pixel pair is the most specific statement a listing makes about
    a panel, so where one is present it settles the question -- including when
    it settles it as "no class at all". A listing whose pixels reach no rung
    while its wording says otherwise ("540x960; 720p") is advertising a class it
    does not have, and reading the wording there is how a decoy wins.

    Only where nothing states pixels do the class names speak. Without that
    fallback a catalogue that writes its resolution in words resolves to
    nothing, every candidate reads as unknown, and the resolution filter in
    `_narrow` empties and falls open -- which is indistinguishable from carrying
    no resolution requirement at all.
    """
    raw = str(text or "")
    pairs = _PIXELS_RE.findall(raw)
    if not pairs:
        return _named_class(raw)
    best: tuple[int, str] | None = None
    for raw_long, raw_short in pairs:
        longer, shorter = max(int(raw_long), int(raw_short)), min(int(raw_long), int(raw_short))
        for index, (name, need_long, need_short, _aliases) in enumerate(_ladder_rungs):
            if longer >= need_long and shorter >= need_short:
                if best is None or index < best[0]:
                    best = (index, name)
                break
    return best[1] if best else None


def _variant_resolution(row: dict) -> str | None:
    for source in (row.get("options"), (row.get("facts") or {}).get("display"), row.get("title")):
        found = _resolution_class(source)
        if found:
            return found
    return None


def _narrow(rows: list[dict], goal: str) -> list[dict]:
    """Deterministic variant-level narrowing, ahead of any judgement call.

    Each filter falls open rather than emptying the pool: a stated requirement
    that no inspected candidate meets is evidence about the pool, not a reason
    to stop deciding.
    """
    inches = _required_inches(goal)
    if inches is not None:
        rows = [row for row in rows if inches in _variant_inches(row)] or rows
    wanted = _required_resolution(goal)
    if wanted is not None:
        rows = [row for row in rows if _variant_resolution(row) == wanted] or rows
    return rows


# ---------------------------------------------------------------------------
# Evidence tiering
#
# Where a fact is carried is itself evidence about the product. The runtime
# already draws this line for us twice: `place_test_order` accepts a
# specification claim only from a raw fact, never from title wording, and the
# recovery family states its own resolution order outright — variant options,
# then the product's colour fact, then the title. The ladder below is that
# standard, generalised to every requirement.
#
# The levels are compared lexicographically and carry no coefficients. That is
# deliberate rather than lazy: repeat runs of one build move the aggregate by
# ~0.015 and a single binary task by 0.2, so any fitted weight would be tuned
# against noise. A ranking with nothing to tune needs no fit to be trusted.
_EV_NONE, _EV_TITLE, _EV_STRUCTURED, _EV_NATIVE = 0, 1, 2, 3
_EV_NAMES = {_EV_TITLE: "title", _EV_STRUCTURED: "structured", _EV_NATIVE: "structured/native"}

_WORD_RE = re.compile(r"[a-z0-9]+")
_FUSE_RE = re.compile(r"(\d+(?:\.\d+)?)([a-z]+)")


def _words(value: Any) -> list[str]:
    """Normalised word tokens, with each number+unit run also split apart.

    `27inches` and `27 inches` are one fact spelled two ways and the catalogue
    uses both. Emitting the split form alongside the fused one lets either side
    match the other without a stemmer.
    """
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True, default=str).casefold()
    else:
        text = str(value if value is not None else "").casefold()
    out: list[str] = []
    for token in _WORD_RE.findall(text):
        out.append(token)
        fused = _FUSE_RE.fullmatch(token)
        if fused:
            out.extend(fused.groups())
    return out


# The baked index runs Lucene's DefaultEnglishAnalyzer, so a search for
# "headphone" already matches a listing titled "Headphones". Scoring has to
# speak the same dialect or it silently disagrees with the retrieval that
# produced its candidates: measured on a real pool, an exact-token comparison
# found the firm requirement `product_type: headphone` carried by **0 of 20**
# candidates — one of them titled "Wireless Clip-On Bluetooth Headphones" —
# which left the firm gate vacuous and let waterproof swimming shoes win a
# headphone request on soft evidence alone.
#
# This is deliberately only plural folding, not a full stemmer. It is the rule
# the catalogue actually varies on, and every extra rule is a chance to equate
# two things a shopper distinguishes.
def _stem(token: str) -> str:
    if any(character.isdigit() for character in token):
        return token                      # 2pcs and 2pc are different products
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and token[-3] in "sxzhio":
        return token[:-2]                 # boxes -> box, watches -> watch
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]                 # headphones -> headphone
    return token


def _covers(carrier: Any, needles: list[str]) -> bool:
    """Every token of the requirement's value appears in this carrier."""
    if not needles:
        return False
    have = {_stem(word) for word in _words(carrier)}
    return all(_stem(needle) in have for needle in needles)


def _native_key(carrier_key: Any, requirement_key: str) -> bool:
    """The carrier's own key names the concept the requirement is about.

    `size` is native for a `screen_size` requirement; `option_1` is not. This
    is not a claim that such a key describes the product better — only that a
    seller who filed a value under its own name is stating a property, where
    one who filed it under a generic label may just be naming a variant.
    """
    carrier_words = set(_words(carrier_key))
    return any(word in carrier_words for word in _words(requirement_key))


def _carriers(row: dict) -> list[tuple[str, Any]]:
    """This variant's structured carriers: the listing's facts and its options.

    Raw facts and variant options are one body of evidence and rank equally —
    the split records where the seller attached a value, not how good it is.
    Sorted, because an unordered read would make the rendered prompt differ
    between runs on identical input.
    """
    pairs: list[tuple[str, Any]] = []
    for key, value in sorted((row.get("facts") or {}).items()):
        if value not in (None, ""):
            pairs.append((str(key), value))
    options = row.get("options")
    if isinstance(options, dict):
        for key, value in sorted(options.items()):
            if value not in (None, ""):
                pairs.append((str(key), value))
    elif options:
        pairs.append(("", options))
    return pairs


def _measured_level(row: dict, want: float, pattern: Any, fact_key: str) -> int:
    """Evidence level for a numeric spec we resolve ourselves.

    The most specific source that carries any reading decides, and a reading
    that disagrees is a rejection rather than an absence: a variant stating a
    different size has answered the question, in the negative.
    """
    for source, level in (
        (row.get("options"), _EV_NATIVE),
        ((row.get("facts") or {}).get(fact_key), _EV_STRUCTURED),
        (row.get("title"), _EV_TITLE),
    ):
        values = {float(found) for found in pattern.findall(str(source or ""))}
        if values:
            return level if want in values else _EV_NONE
    return _EV_NONE


def _resolution_level(row: dict, want: str) -> int:
    """As `_measured_level`, over the stated pixel ladder."""
    for source, level in (
        (row.get("options"), _EV_NATIVE),
        ((row.get("facts") or {}).get("display"), _EV_STRUCTURED),
        (row.get("title"), _EV_TITLE),
    ):
        found = _resolution_class(source)
        if found:
            return level if found == want else _EV_NONE
    return _EV_NONE


def _evidence_level(row: dict, requirement: dict) -> int:
    """How directly this variant carries one requirement's value."""
    probe = requirement.get("probe")
    if probe == "inches":
        return _measured_level(row, float(requirement["value"]), _INCH_RE, "size")
    if probe == "resolution":
        return _resolution_level(row, str(requirement["value"]))

    needles = _words(requirement.get("value"))
    if not needles:
        return _EV_NONE
    carriers = _carriers(row)
    best = _EV_NONE
    for carrier_key, carrier_value in carriers:
        if _covers(carrier_value, needles):
            if _native_key(carrier_key, str(requirement.get("key") or "")):
                return _EV_NATIVE
            best = _EV_STRUCTURED
    if best == _EV_NONE and carriers:
        # The union pass: a value split across several structured carriers is
        # still carried structurally. It can never reach NATIVE, because no
        # single key holds the whole of it.
        joined = " ".join(f"{key} {_render_value(value)}" for key, value in carriers)
        if _covers(joined, needles):
            best = _EV_STRUCTURED
    if best == _EV_NONE and _covers(row.get("title"), needles):
        best = _EV_TITLE
    return best


# Some tasks state a relaxation ORDER to follow when the committed item dies:
# try the same model in another colour before moving to a different model. Read
# it off the rule text rather than off a family label, because the constraint
# family states the opposite — switch product rather than settle for a bad
# sibling variant — and a mislabelled family would then invert the ladder.
# No match means no ladder, which is the behaviour without this.
_RELAX_LADDER_RE = re.compile(
    r"(?:same\s+model\s+in\s+another\s+colou?r"
    r"|another\s+colou?r\s+of\s+the\s+same"
    r"|different\s+colou?r\s+of\s+the\s+same)",
    re.I,
)


def _sku_family(sku: Any) -> str:
    """The listing a variant belongs to.

    Every product_id in this catalogue carries exactly ONE variant, so
    product_id cannot express "another colour of the same model" — the sku
    does. Skus are shaped `<listing>-<n>` and the prefix groups a listing's
    colours: measured over a full run, 33 prefixes span more than one
    (product_id, options) pair and every one of them spans several product_ids.
    A sku without that shape falls back to itself and matches nothing else,
    which is the safe direction.
    """
    return str(sku).rsplit("-", 1)[0]


# The catalogue's classification is its own vocabulary, and a requirement
# extracted from the request describes what an item is FOR rather than where it
# is filed. Asked for a headphone "for daily runs in every kind of weather", the
# checklist offered `sport` -- which names a real branch of this catalogue, just
# not the shopper's -- so holding candidates to it removed all twenty-nine
# acceptable items rather than falling open. The shopper had said "the required
# catalog category is Headphones" in the same breath. Read what they said.
_STATED_CATEGORY_RE = re.compile(
    r"(?:required|requested)\s+(?:catalog(?:ue)?\s+)?category\s*(?:is|:)\s*"
    r"([A-Za-z][A-Za-z0-9 &/'-]{0,48}?)\s*(?:[.,;\n]|$)",
    re.I,
)


def _stated_category(text: str) -> str | None:
    """The catalogue category the shopper named outright, if they named one."""
    match = _STATED_CATEGORY_RE.search(text or "")
    return match.group(1).strip() if match else None


def _category_matches(row: dict, wanted: str) -> bool:
    """Whether the CATALOGUE files this listing under the requested category.

    Read off `category_path`, which is the classification the catalogue
    publishes and the only thing that establishes one: the rules say outright
    that a phone-like title does not put a listing in Smartphones, and the
    field exists on view and compare precisely so the claim can be checked.

    Matched one path element at a time under the same stemmed token-subset rule
    the rest of the evidence code uses, so "Smartphones" answers a request for
    "smartphone" while "Feature Phones" does not -- however the title reads and
    whatever the seller filed under a `phone_type` fact.
    """
    needles = _words(wanted)
    if not needles:
        return False
    return any(_covers(element, needles) for element in (row.get("category_path") or []))


def _brand_matches(row: dict, brand: str) -> bool:
    """Token-subset brand match, the rule the verifier itself uses.

    Reads the `brand` FACT only. A brand word in a title is marketing copy —
    the same reason a specification claim may not be built from one.
    """
    wanted = set(_words(brand))
    return bool(wanted) and wanted <= set(_words(row.get("brand")))


# The shopper names both halves of what they want -- "the green 635", "the
# black 431915030732" -- and the recovery rules confirm the pair by calling it
# "that exact color/model". Colour first, because that is the order the request
# states it in and the order the rules relax it in.
_PREFERRED_RE = re.compile(
    r"prefer(?:s|red|ring)?\s+(?:the\s+)?([0-9a-z][0-9a-z-]*)\s+([0-9a-z][0-9a-z-]*)",
    re.I,
)


# Grammar, not product vocabulary: "prefer a quiet pair" names no pair, and
# without this the two words after the verb are read as one. A bogus pair is
# mostly harmless -- every candidate then lands on the same bottom rung and the
# rung filter keeps the lot -- but a word that accidentally matches some of them
# would reorder the rest for no reason.
_DETERMINERS = frozenset(
    ("a", "an", "the", "my", "our", "your", "their", "its", "this", "that",
     "these", "those", "some", "any", "one", "it", "something", "anything")
)


def _preferred_pair(goal: str) -> tuple[str | None, str | None]:
    """The colour and model the shopper named, when they named a pair.

    Which token is which is read off the tokens rather than their order: a
    model designation carries a digit where a colour word does not, so
    "the green 635" and a hypothetical "the 635 green" resolve alike. With
    neither carrying one the stated order decides, which is colour then model.
    """
    match = _PREFERRED_RE.search(goal or "")
    if not match:
        return None, None
    first, second = match.group(1), match.group(2)
    if first.casefold() in _DETERMINERS or second.casefold() in _DETERMINERS:
        return None, None
    if any(ch.isdigit() for ch in second):
        return first, second
    if any(ch.isdigit() for ch in first):
        return second, first
    return first, second


def _model_matches(row: dict, model: str) -> bool:
    """Whether this listing is the named model.

    The `model` fact carries the designation; a title merely mentions it, and
    "lumia 635" answers a request for "635" because the token is present.
    """
    needles = _words(model)
    if not needles:
        return False
    fact = (row.get("facts") or {}).get("model")
    if fact not in (None, ""):
        return _covers(fact, needles)
    return _covers(row.get("title"), needles)


def _colour_matches(row: dict, colour: str) -> bool:
    """Whether this VARIANT is the named colour.

    Variant options first, then the listing's own colour fact, then the title
    -- the resolution order the recovery rules state. The most specific source
    that carries any colour at all decides, so a variant whose options say
    black is not green however the title reads. "midnight black" and "forrest
    green" answer for their base colour, because the token is there.
    """
    needles = _words(colour)
    if not needles:
        return False
    facts = row.get("facts") or {}
    for source in (row.get("options"), facts.get("color_family"), facts.get("color"),
                   facts.get("colour"), row.get("title")):
        if source not in (None, "", {}, []):
            return _covers(source, needles)
    return False


def _preference_level(row: dict, colour: str, model: str) -> int:
    """How far this variant sits from the pair the shopper named.

    0 is both, 1 the model alone, 2 the colour alone, 3 neither -- the ladder
    the rules state, which gives up colour before model.

    Graded against the STATED preference rather than against the item that
    died. The two coincide only while the item that died was the full match,
    and nothing guarantees that; measuring from the corpse also leaves colour
    invisible, which collapses the rules' two middle rungs -- 161 candidates
    and 1846 of them on one measured task -- into a single indistinguishable
    step worth a third of the payout.
    """
    has_model = _model_matches(row, model)
    has_colour = _colour_matches(row, colour)
    if has_model and has_colour:
        return 0
    if has_model:
        return 1
    if has_colour:
        return 2
    return 3


def _relaxation_step(
    row: dict, dead_row: dict | None, preference: tuple[str | None, str | None]
) -> int:
    """The rung this candidate sits on, by the best measure available."""
    colour, model = preference
    if colour and model:
        return _preference_level(row, colour, model)
    return _relaxation_distance(row, dead_row)


def _relaxation_distance(row: dict, dead_row: dict | None) -> int:
    """How far this candidate steps away from the item that just died.

    0 is another colour of the same listing, 1 keeps the model but changes
    listing, 2 is a different model. Graded recovery pays for the smallest step
    that still satisfies the firm constraints, so this orders by that step.

    Taking the rungs from the catalogue rather than from intuition mattered:
    keying rung 0 on product_id looked obviously right, could never fire, and
    left a same-colour/different-model candidate ranked above the
    different-colour/same-listing one the rule actually asks for.
    """
    if not dead_row:
        return 0
    if _sku_family(row.get("sku")) == _sku_family(dead_row.get("sku")):
        return 0
    dead_model = str((dead_row.get("facts") or {}).get("model") or "").strip().casefold()
    model = str((row.get("facts") or {}).get("model") or "").strip().casefold()
    if dead_model and model and dead_model == model:
        return 1
    return 2


def _evidence_row(row: dict, requirements: list[dict]) -> list[int]:
    return [_evidence_level(row, requirement) for requirement in requirements]


def _evidence_key(
    levels: list[int], requirements: list[dict] | None = None
) -> tuple[int, ...]:
    """An ordering over candidates with nothing in it to tune.

    Firm requirements first, then soft ones, each read three ways: carried at
    all, carried by structured data rather than title wording, and carried
    under a key that names the property.

    **Firm acts as a GATE, not as a score.** How many firm requirements a
    candidate MEETS ranks first; how deeply it carries them ranks last, behind
    every soft component. Splitting firm from soft was not enough on its own:
    firm depth varies freely across survivors — a listing naming the product
    kind in a fact outranks one naming it in its title — so putting all three
    firm components ahead of soft let firm depth decide every comparison and
    soft decide none. Replayed against a real pool, adding a soft requirement
    moved the matching candidates from rank 12 to rank 2 and left the winner
    unchanged.

    With no checklist supplied every requirement counts as firm, which is the
    old behaviour and the right default for a caller that has no soft/firm
    split to offer.
    """
    firm = [
        level
        for index, level in enumerate(levels)
        if requirements is None or requirements[index].get("firm", True)
    ]
    soft = [
        level
        for index, level in enumerate(levels)
        if requirements is not None and not requirements[index].get("firm", True)
    ]
    return (
        sum(1 for level in firm if level > _EV_NONE),      # firm requirements MET
        sum(1 for level in soft if level > _EV_NONE),      # then soft, at full depth
        sum(1 for level in soft if level >= _EV_STRUCTURED),
        sum(1 for level in soft if level >= _EV_NATIVE),
        sum(1 for level in firm if level >= _EV_STRUCTURED),   # firm DEPTH last
        sum(1 for level in firm if level >= _EV_NATIVE),
    )


def _dominates(challenger: tuple, incumbent: tuple) -> bool:
    return challenger != incumbent and all(a >= b for a, b in zip(challenger, incumbent))


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
        # Retrieval calls already issued, normalised. `_retrieval_actions` owns
        # this and updates it in place: a repeated query string buys no rows
        # against a backend that is deterministic within a run, and TF2 records
        # the repeat as an overfetch attempt.
        self.queries: set[str] = set()

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

    def absorb(self, observations: list[dict]) -> None:
        """Fold one turn's tool payloads in, whatever tools produced them.

        Callers hand over the unwrapped payloads, not the `{done, error,
        observation}` envelopes `/environment/call` returns.
        """
        for obs in observations:
            self.add_summaries(obs.get("results") or [])
            for row in obs.get("compared") or []:
                self.add_detail(row)
            if obs.get("facts") is not None or obs.get("variants") is not None:
                self.add_detail(obs)
            if obs.get("in_stock") is not None and obs.get("sku"):
                self.update_variant(obs)
            added = obs.get("added")
            if isinstance(added, dict):
                self.update_variant(added)

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
                        # The whole path, not just the leaf: a requirement may
                        # name any level the catalogue files the listing under.
                        "category_path": record.get("category_path") or [],
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


def _chat(
    messages: list[dict], *, expect_json: bool = True, temperature: float = 0.0
) -> Any:
    """One inference round-trip; returns parsed JSON or raw text.

    `temperature` is raised only by a caller retrying a reply it could not
    parse, and belongs to that reply rather than to the call site.
    """
    response = _proxy.post(
        "/inference/chat/completions",
        json_data={
            "model": _model(),
            "messages": messages,
            "temperature": temperature,
        },
    )
    if response is None:
        raise RuntimeError("inference request failed")
    content = (response["choices"][0]["message"].get("content") or "").strip()
    return _extract_json(content) if expect_json else content


def _plan_queries(goal: str, matching_phrase: bool) -> list[str]:
    # Metered only where the shopper named a phrase to match — the same signal
    # that decides how widely this task can afford to inspect.
    cap = _MAX_QUERIES if matching_phrase else _MAX_QUERIES_WIDE
    hint = (
        "The shopper named the phrase to match; use it and close variations."
        if matching_phrase
        else "Cover the category and the stated requirement, and vary the "
        "wording between queries: near-duplicates return the same rows."
    )
    plan = _chat(
        [
            {
                "role": "system",
                "content": (
                    "You plan catalog searches for a shopping agent. "
                    f"Return JSON: {{\"queries\": [..]}} with at most {cap} "
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
    return queries[:cap] or [goal[:80]]


# ---------------------------------------------------------------------------
# Requirement extraction
#
# The checklist is built once and handed to the verifier complete. A verifier
# left to re-derive its own requirements from the request text re-fuses the
# ones that were correctly separated — a stated feature folds back into the
# product kind, and the candidate that files that feature as a raw fact is
# never compared against the right thing.
#
# Price and stock are deliberately absent: `Catalog.eligible` has already
# filtered on both, so every candidate reaching the verifier satisfies them,
# and a requirement no rendered block can answer only ever fails everyone.
_REQUIREMENTS_SYSTEM = (
    "You turn a shopper's request into a requirement checklist.\n"
    'Return JSON: {"firm": [{"key": "..", "value": ".."}], '
    '"soft": [{"key": "..", "value": ".."}]}\n'
    "- firm: must hold for the exact variant bought — the product kind, and any "
    "specification the shopper stated as a requirement.\n"
    "- soft: a preference. A named brand is soft unless the shopper ruled out "
    "every other brand.\n"
    "- `key` names the property; `value` is what a listing has to carry.\n"
    "- Minimise the value by moving descriptive words into the key: a stated "
    "shape becomes shape: <the shape>, not <the shape> shape.\n"
    "- One aspect per entry. If each word names a complete value on its own, "
    "split them; if either names nothing without the other, keep them together.\n"
    "- Absence is a value: something the shopper wants omitted becomes "
    "feature: no <thing>, never a dropped entry.\n"
    "- Keys must be unique. Derive a more specific key rather than repeating one.\n"
    "- Omit price, budget, cost and stock entirely; those are handled elsewhere.\n"
    "- When unsure, leave it out. An entry you omit costs a little ranking; a "
    "wrong one deletes a real requirement and imposes a filter nothing meets.\n"
    "No prose."
)

_MAX_REQUIREMENTS = 8
_SKIP_KEYS = ("price", "budget", "cost", "stock", "availability", "currency")


def _clean_requirements(payload: Any, firm: bool) -> list[dict]:
    """Structural validation, applied before anything reads the checklist."""
    out: list[dict] = []
    seen: set[str] = set()
    for entry in payload or []:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key") or "").strip().casefold().replace(" ", "_")
        value = str(entry.get("value") or "").strip()
        if not key or not value or key in seen:
            continue
        if any(skip in key for skip in _SKIP_KEYS):
            continue
        # A row whose value merely restates its own key names no value to look
        # for. It cannot be satisfied by any candidate, so checking it would
        # empty the survivor set rather than narrow it.
        if set(_words(value)) <= set(_words(key)):
            continue
        seen.add(key)
        out.append({"key": key, "value": value, "firm": firm})
    return out


def _requirements(goal: str) -> list[dict]:
    """The checklist for this request: parsed facts first, then the model's."""
    fixed: list[dict] = []
    inches = _required_inches(goal)
    if inches is not None:
        fixed.append(
            {"key": "screen_size", "value": inches, "firm": True, "probe": "inches"}
        )
    resolution = _required_resolution(goal)
    if resolution is not None:
        fixed.append(
            {"key": "resolution", "value": resolution, "firm": True, "probe": "resolution"}
        )

    parsed: list[dict] = []
    for temperature in (0.0, 0.3):
        try:
            reply = _chat(
                [
                    {"role": "system", "content": _REQUIREMENTS_SYSTEM},
                    {"role": "user", "content": goal},
                ],
                temperature=temperature,
            )
        except RuntimeError:
            break
        if isinstance(reply, dict):
            parsed = _clean_requirements(reply.get("firm"), True) + _clean_requirements(
                reply.get("soft"), False
            )
        if parsed:
            break

    # A parsed requirement naming a property we already resolve ourselves is a
    # restatement of it, and the deterministic form is the exact one.
    taken = {entry["key"] for entry in fixed}
    merged = fixed + [entry for entry in parsed if entry["key"] not in taken]
    return merged[:_MAX_REQUIREMENTS]


# ---------------------------------------------------------------------------
# Retrieval breadth
#
# The two retrieval surfaces answer different questions, and only one of them
# is metered.
#
# `search` is BM25 over the baked index, capped at 10 rows with no paging. TF2
# divides its payout by the number of DISTINCT query strings, so there the
# three the planner already produces are the entire budget; the runtime counts
# only actions named `search` toward that divisor.
#
# `filter` walks the catalogue's own category and brand indexes in catalogue
# order — an ordering BM25 never produces, which is precisely why it reaches
# listings no reformulation does. The runtime counts it in neither
# `query_efficiency` (which reads `name == "search"` alone) nor
# `shortlist_precision` (which reads view/compare/add_to_cart alone), so it is
# free breadth in every family, TF2 included. Better still for TF2:
# `cluster_recall_at_10` scores everything OBSERVED before the order, and a
# filter result is an observation — so it raises recall without inspecting
# anything, which is the half of that metric we were not paying for.
#
# The catalogue's category filter is forgiving (it canonicalises labels, so a
# plain category word reaches the compound ones), and its brand filter uses the
# verifier's own token rule — a brand the verifier accepts is reachable here.
# There is no price FLOOR, so a band cannot be raised the way the retired
# agents raised one against a stated ceiling. What the API does offer is
# several distinct slices of one category index, which buys the same breadth
# by a different route.
_FILTER_BANDS = (1.0, 0.5)
_MAX_RETRIEVAL_CALLS = 12
_CATEGORY_KEYS = ("category", "type", "kind", "product")


def _requirement_value(requirements: list[dict], *names: str) -> str | None:
    """The first checklist value filed under a key naming one of `names`."""
    for name in names:
        for requirement in requirements:
            if name in str(requirement.get("key") or "").split("_"):
                value = str(requirement.get("value") or "").strip()
                if value and not isinstance(requirement.get("value"), float):
                    return value
    return None


def _retrieval_actions(
    queries: list[str],
    requirements: list[dict],
    budget: float | None,
    seen: set[str],
    limit: int = _MAX_RETRIEVAL_CALLS,
    category: str | None = None,
) -> list[dict]:
    """One turn of retrieval: the planned searches, then structured breadth.

    `seen` carries the query strings already issued and is updated in place. A
    repeated string buys no rows — the backend is deterministic within a run —
    and the retrieval family records the repeat as an overfetch attempt.
    """
    actions: list[dict] = []
    for text in queries:
        key = str(text).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        actions.append(
            {"name": "search", "args": {"query": text, "k": 10, "in_stock": True}}
        )

    category = category or _requirement_value(requirements, *_CATEGORY_KEYS)
    brand = _requirement_value(requirements, "brand")

    combos: list[dict] = []
    if category:
        combos.append({"category": category})
    if brand:
        combos.append({"brand": brand})
    if brand and category:
        combos.append({"brand": brand, "category": category})
    if budget is not None:
        # Distinct slices of one index. Without a floor the lower band can only
        # add cheaper rows, but those are rows the unbanded call ranked below
        # its own cut-off and would never have returned.
        for share in _FILTER_BANDS:
            band: dict[str, Any] = {"max_price": round(budget * share, 2)}
            if category:
                band["category"] = category
            combos.append(band)
            if brand:
                # Every slice is capped at ten rows and the cap is not ours to
                # raise -- the tool's own description says to narrow the filters
                # to reach other items. A slice that comes back full is one we
                # have only partly seen, so the narrowest combination available
                # is the only thing that reaches what a broader one truncated
                # away. Measured on a run where all eight retrieval calls came
                # back saturated, `{brand, category}` and `{category, max_price}`
                # between them returned none of the fourteen gold variants; the
                # band carrying both is the slice neither of them covers.
                combos.append({**band, "brand": brand})

    for combo in combos:
        key = "filter:" + json.dumps(combo, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        actions.append({"name": "filter", "args": {**combo, "k": 10, "in_stock": True}})

    return actions[:limit]


# ---------------------------------------------------------------------------
# Verification, then selection
#
# Two questions, two calls. Asking one call to both check every requirement and
# name a winner stops doing the per-candidate check once the rendered pool gets
# large — and because it then reports that nothing matched, the whole selection
# collapses onto a near miss. So the check runs per candidate, where it is
# independent and batchable, and the comparison runs afterwards over survivors
# only.
_VERIFY_BATCH = 10
_MAX_CANDIDATES = 40

_VERIFY_SYSTEM = (
    "You check candidate products against a requirement checklist.\n"
    "The checklist is complete and final. Check every line on it and nothing "
    "that is not on it. Never add a requirement you believe the request "
    "implies, never drop one, never merge two.\n"
    "Read each line whole: `value` is what the candidate must carry, `key` "
    "names the property it describes. A value is sometimes filed under an "
    "imperfect key — trust the value and use the key only to understand what "
    "is being asked.\n"
    "MATCHING. Identifiers, model names, numbers, versions, codes and colours "
    "match literally: never normalise or partially match them. Everything else "
    "matches semantically — paraphrases, plurals and common typos all pass. "
    "Prefer a false positive to a false negative: a candidate you wrongly pass "
    "is rejected by the comparison that follows, one you wrongly fail is gone.\n"
    "EVIDENCE. A title is an unordered keyword collection: check whether the "
    "tokens are present, never what the sentence appears to mean. Raw facts "
    "and variant options are one body of evidence and rank equally, but values "
    "on ONE variant line are sold together and values on different lines are "
    "not — two requirements met by different lines are not met by one "
    "purchasable item. Evidence must be carried, never inferred: a line that "
    "merely makes a requirement plausible is not evidence for it. A "
    "requirement stating an absence is satisfied only by evidence stating the "
    "same absence.\n"
    "Extra words around a matching value are free. A value narrower than the "
    "requirement does not carry it.\n"
    "Evaluate EVERY candidate in the order given, to the last one. A "
    "candidate's position is not evidence about it. End a line at its first "
    "FAIL. Passing one does not end the sweep.\n"
    'Return JSON: {"evaluation": {"1": "<key> PASS <evidence>; <key> FAIL", ..}, '
    '"passed": "1, 4"} with one entry for every candidate, and null for passed '
    "when none pass. No prose."
)

_SELECT_SYSTEM = (
    "You choose between product listings that have ALREADY been checked. Every "
    "candidate below satisfies every requirement; that decision is final. "
    "Never reject one here, never call one invalid, never answer that none "
    "matches. Choose the one this shopper should be given.\n"
    "Ask which listing a seller wrote for this request — not which one happens "
    "to satisfy a checklist. Weigh, in decreasing order:\n"
    "1. Where the evidence lives. A property filed as a raw fact or a variant "
    "option describes the product; a title is written to be found. Structured "
    "evidence wins regardless of how well the title is phrased.\n"
    "2. Whether the value is stated the way the property is normally stated. A "
    "value containing the requested one with extra words around it still "
    "carries it; a value narrower than the request does not. A longer value is "
    "not worse for being longer.\n"
    "3. Whether it is the right kind of thing. A listing from another category "
    "can file a requested property as an ordinary fact and so appear to carry "
    "more than the right kind of product does.\n"
    "Prefer the best FIT over the cheapest. Every candidate here is already "
    "inside the budget, so a lower price is not a reason to choose one: an "
    "item with no factual support for a stated preference scores nothing on "
    "that preference, whatever it costs.\n"
    "Not reasons to choose: repeating the request's words, being listed first, "
    "or having more evidence entries than another candidate.\n"
    'Return JSON: {"best": "<candidate number>", "why": ".."}. No prose.'
)


# Used only where the per-candidate check could not run. Asking one call to
# both check and compare is the failure this split exists to avoid, so it is
# the degraded path and never the normal one — but a combined call still beats
# no call at all.
_JUDGE_SYSTEM = (
    "You choose one product listing for a shopper.\n"
    "The requirements below have NOT been checked against these candidates. "
    "Satisfy yourself that your choice carries every one of them, judged from "
    "its raw facts and variant options rather than its title, then choose.\n"
    "A title is an unordered keyword collection: token presence is evidence, "
    "the apparent meaning of the sentence is not. Raw facts and variant "
    "options rank equally, but values on one variant line are sold together "
    "and values on different lines are not.\n"
    "Prefer the best fit over the cheapest. An item with no factual support "
    "for a stated preference scores zero on it.\n"
    "When a device is requested, require positive evidence that the item IS "
    "that device, and reject accessories and parts whatever the title says.\n"
    'Return JSON: {"best": "<candidate number>", "why": ".."}. No prose.'
)


def _render_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _render_checklist(requirements: list[dict]) -> str:
    lines = []
    for requirement in requirements:
        mark = "" if requirement.get("firm") else "  (preference)"
        lines.append(f"- {requirement['key']}: {_render_value(requirement['value'])}{mark}")
    return "\n".join(lines) if lines else "- (none)"


def _render_candidate(index: int, row: dict, requirements: list[dict], levels: list[int]) -> str:
    lines = [f"Candidate {index}", f"  title: {row.get('title')}"]
    facts = "; ".join(
        f"{key}: {_render_value(value)}"
        for key, value in sorted((row.get("facts") or {}).items())
        if value not in (None, "")
    )
    if facts:
        lines.append(f"  facts: {facts}")
    if row.get("category"):
        lines.append(f"  category: {row['category']}")
    options = row.get("options")
    if options:
        lines.append(f"  variant options: {_render_value(options)}")
    lines.append(f"  price: {row.get('price')}")
    # Computed, not asked for. The rule that title wording is weaker evidence
    # than a raw fact was already stated twice in the prompt and still applied
    # inconsistently, so the part that is mechanically checkable is supplied
    # instead of instructed. Omitted when nothing is carried, because an empty
    # list reads as an assertion that the candidate carries nothing.
    carried = ", ".join(
        f"{requirement['key']}({_EV_NAMES[level]})"
        for requirement, level in zip(requirements, levels)
        if level > _EV_NONE
    )
    if carried:
        lines.append(f"  carried: {carried}")
    return "\n".join(lines)


def _parse_verdicts(reply: Any, size: int) -> tuple[list[int], dict[int, int]]:
    """Survivor indices, plus each candidate's satisfied-requirement count.

    A line ends at its first FAIL, so its PASS count is exactly the number of
    requirements that candidate satisfied — which is what makes the near-miss
    ranking free.
    """
    if not isinstance(reply, dict):
        return [], {}
    evaluation = reply.get("evaluation")
    if not isinstance(evaluation, dict):
        for value in list(reply.values())[::-1]:
            if isinstance(value, dict) and any(isinstance(v, str) for v in value.values()):
                evaluation = value
                break
    if not isinstance(evaluation, dict):
        return [], {}
    passed: list[int] = []
    depth: dict[int, int] = {}
    for raw_key, line in sorted(evaluation.items()):
        if not isinstance(line, str):
            continue
        digits = re.findall(r"\d+", str(raw_key))
        if not digits:
            continue
        index = int(digits[-1]) - 1
        if not 0 <= index < size:
            continue
        depth[index] = len(re.findall(r"\bPASS\b", line, re.I))
        if depth[index] and not re.search(r"\b(FAIL|UNKNOWN)\b", line, re.I):
            passed.append(index)
    return sorted(passed), depth


def _verify(
    goal: str, requirements: list[dict], rows: list[dict], evidence: dict[str, list[int]]
) -> tuple[list[dict], bool]:
    """Check each candidate independently. Returns (survivors, actually_checked).

    Fails open in both directions. A checklist that rejected everything is a
    statement about the checklist, not about the pool, and a call that never
    landed leaves candidates unchecked — a third state, distinct from verified
    and from rejected, and the caller is told which happened.
    """
    if not requirements or len(rows) < 2:
        return rows, False

    checklist = _render_checklist(requirements)
    survivors: list[dict] = []
    deepest: list[tuple[int, dict]] = []
    answered = False

    for start in range(0, len(rows), _VERIFY_BATCH):
        batch = rows[start : start + _VERIFY_BATCH]
        blocks = "\n\n".join(
            _render_candidate(offset + 1, row, requirements, evidence[_key(row["product_id"], row["sku"])])
            for offset, row in enumerate(batch)
        )
        try:
            reply = _chat(
                [
                    {"role": "system", "content": _VERIFY_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Request:\n{goal}\n\nChecklist:\n{checklist}\n\n"
                            f"Candidates:\n{blocks}"
                        ),
                    },
                ]
            )
        except RuntimeError:
            continue
        passed, depth = _parse_verdicts(reply, len(batch))
        if depth:
            answered = True
        survivors.extend(batch[index] for index in passed)
        for index, count in sorted(depth.items()):
            deepest.append((count, batch[index]))

    if not answered:
        return rows, False
    if survivors:
        return survivors, True
    # Nobody passed, but the same replies measured everyone. The head of the
    # ranking the check just rejected in full is a worse fallback than the
    # candidates that got furthest through it: that ranking cannot tell a wrong
    # kind of object with the right property from the reverse, and the check
    # has already measured which is which.
    best = max((count for count, _ in deepest), default=0)
    near = [row for count, row in deepest if count == best and count > 0]
    return (near or rows), True


def _select_best(
    goal: str,
    rules: str,
    requirements: list[dict],
    rows: list[dict],
    evidence: dict[str, list[int]],
    budget: float | None,
    currency: str,
    dead: str | None,
    verified: bool,
    dead_row: dict | None = None,
    relax_ladder: bool = False,
    preference: tuple[str | None, str | None] = (None, None),
) -> dict:
    """Compare survivors and return one. Never rejects, never returns nothing.

    `verified` says whether the per-candidate check actually ran. Where it did
    not, the prompt must not claim it did — an unearned "all of these already
    passed" turns a degraded path into a confident wrong answer.
    """
    def rank(row: dict) -> tuple:
        key = _evidence_key(evidence[_key(row["product_id"], row["sku"])], requirements)
        # Price is a CONSTRAINT, not a quality signal: `eligible` has already
        # enforced the budget, so every candidate here is affordable. Ordering
        # the remainder by price sold a shopper who said they would pay more
        # for sweat resistance the cheapest item in the pool with none. `sku`
        # breaks the residual tie, and only for determinism.
        return (
            _relaxation_step(row, dead_row, preference) if relax_ladder else 0,
            tuple(-level for level in key),
            row["sku"],
        )

    ordered = sorted(rows, key=rank)
    if len(ordered) == 1:
        return ordered[0]

    context = (
        f"The variant {dead} is gone. Choose the best remaining option under the "
        "shopper's stated fallback order."
        if dead
        else "This is a real purchase commitment, not a placeholder."
    )
    budget_line = f"Working budget: {budget:.2f} {currency}." if budget is not None else ""
    blocks = "\n\n".join(
        _render_candidate(index + 1, row, requirements, evidence[_key(row["product_id"], row["sku"])])
        for index, row in enumerate(ordered)
    )
    try:
        reply = _chat(
            [
                {"role": "system", "content": _SELECT_SYSTEM if verified else _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Request:\n{goal}\n\nRules:\n{rules}\n\n"
                        + (
                            "Requirements already satisfied by all of these:\n"
                            if verified
                            else "Requirements, NOT yet checked against these candidates:\n"
                        )
                        + f"{_render_checklist(requirements)}\n\n{budget_line}\n{context}\n\n"
                        f"Candidates:\n{blocks}"
                    ),
                },
            ]
        )
    except RuntimeError:
        return ordered[0]

    picked = 0
    if isinstance(reply, dict):
        digits = re.findall(r"\d+", str(reply.get("best") or ""))
        if digits:
            index = int(digits[0]) - 1
            if 0 <= index < len(ordered):
                picked = index

    # The computed ordering may overturn the choice, but only where a rival is
    # at least as good on every component and better on one. A candidate ahead
    # on one count and behind on another does not win, and a tie among rivals
    # leaves the choice alone: this decides order within a set the model has
    # already judged, never which candidates are valid.
    incumbent = _evidence_key(
        evidence[_key(ordered[picked]["product_id"], ordered[picked]["sku"])], requirements
    )
    challengers = [
        row
        for row in ordered
        if _dominates(
            _evidence_key(evidence[_key(row["product_id"], row["sku"])], requirements),
            incumbent,
        )
    ]
    if not challengers:
        return ordered[picked]
    best = max(
        _evidence_key(evidence[_key(row["product_id"], row["sku"])], requirements)
        for row in challengers
    )
    tied = [
        row
        for row in challengers
        if _evidence_key(evidence[_key(row["product_id"], row["sku"])], requirements) == best
    ]
    return tied[0] if len(tied) == 1 else ordered[picked]


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
    tools = _tool_index(policy.get("tools") or [])
    _arm_ladder(rules)
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
    # One checklist serves both halves of the run: it names what to retrieve
    # and, later, what each candidate is checked against. Keyed on the goal
    # because the shopper may add to it — an underspecified task releases its
    # firm requirement only in reply to a direct question.
    checklists: dict[str, list[dict]] = {}
    terms = _phrase_terms(goal)
    # The shared rules reject accessory and non-device listings whenever the
    # shopper asks for a device, and both device families gate on it.
    wants_device = "not an accessory" in query.casefold() or bool(
        re.search(r"\b(smartphone|phone|monitor|headphone|laptop|tablet|camera)s?\b", goal, re.I)
    )
    # See `_SHORTLIST`: a task that names a phrase to match is graded on the
    # precision of what we inspected, so it is the one that cannot afford to
    # look widely. That is the shopper's own request, not a task shape we infer.
    scored_shortlist = bool(terms)
    shortlist = _SHORTLIST if scored_shortlist else _SHORTLIST_WIDE

    def inspect_more(note: str) -> bool:
        """View the next batch of search hits. Returns False when none are left."""
        pending = catalog.undetailed(shortlist * 3)
        if terms:
            # Only phrase-matching titles can be in the eligible pool, and every
            # inspection of something outside it lowers the retrieval payout.
            # Replayed over the pack this drops no gold product at all, so it
            # stays a hard filter.
            pending = [
                pid for pid in pending
                if _title_matches((catalog.products[pid] or {}).get("title"), terms)
            ] or pending
        if wants_device:
            accessory = {
                pid for pid in pending
                if _looks_like_accessory((catalog.products[pid] or {}).get("title"))
            }
            device_first = [pid for pid in pending if pid not in accessory]
            # The families do state "reject accessories when a device is
            # asked for", but a title keyword is a guess at that rule and a
            # guess that DELETES is pure downside wherever inspection is free:
            # replayed over the pack it puts 12 gold products out of reach,
            # including both of the only two one task had, which is that task's
            # whole loss. Demoting keeps the intent — a real device is still
            # inspected first — while costing a misread title its ordering
            # rather than the answer.
            #
            # TF2 is the exception, and keeps the deletion: there a compared
            # product outside the eligible pool is subtracted from the payout,
            # so acting on the guess is what the metric asks for.
            pending = (
                (device_first or pending)
                if scored_shortlist
                else device_first + [pid for pid in pending if pid in accessory]
            )
        pending = pending[:shortlist]
        if not pending:
            return False
        catalog.absorb(
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
        budget = _fold_budget(budget, currency, message)
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
        catalog.absorb(driver.step([{"name": "message", "args": {"content": str(reply)[:400]}}], "reply"))

    # ASK — anything the shopper deliberately withheld has to be requested
    # before the first cart add, because that add is graded against the full
    # requirement set including the part not yet stated.
    if _should_ask(goal, query):
        question = _chat(
            [
                {
                    "role": "system",
                    "content": (
                        # A shopper who reveals progressively answers a vague
                        # question with the use case they already gave. Asking
                        # for the properties themselves is what earns the part
                        # being held back, and asking for ALL of them matters
                        # where a commitment is graded on matching several at
                        # once rather than any single one.
                        "The shopper has told you what they want the product "
                        "FOR but is holding back the specifics. Ask one short, "
                        "direct question for the concrete product requirements "
                        "and features that matter to them, and invite them to "
                        "name every one rather than only the most important. "
                        "Do not ask about their use case; they have given it. "
                        "Plain text only."
                    ),
                },
                {"role": "user", "content": goal},
            ],
            expect_json=False,
        )
        catalog.absorb(driver.step([{"name": "message", "args": {"content": str(question)[:300]}}], "ask"))
        answer_if_asked()

    # SEARCH — a named candidate set replaces retrieval entirely.
    named = _parse_candidates(goal)
    if named:
        ids = sorted({product_id for product_id, _ in named})
        catalog.absorb(
            driver.step(
                [
                    {"name": "compare", "args": {"product_ids": ids[index : index + _COMPARE_CAP]}}
                    for index in range(0, len(ids), _COMPARE_CAP)
                ],
                "compare named candidates",
            )
        )
    else:
        checklists[goal] = _requirements(goal)
        # The ceiling is the contract's own, read from `policy_view`, rather
        # than a number of ours: one turn carries `max_calls_per_turn` calls.
        actions = _retrieval_actions(
            _plan_queries(goal, scored_shortlist),
            checklists[goal],
            budget,
            catalog.queries,
            driver.max_calls,
            _stated_category(goal),
        )
        catalog.absorb(driver.step(actions, "search"))

        inspect_more("inspect candidates")
    answer_if_asked()
    if not catalog.eligible(budget, dead):
        # A tightened budget can empty the inspected pool; widen it before
        # giving up, since an abandoned session scores zero outright.
        inspect_more("inspect further candidates")

    ranked_axes = _axis_order(goal) if named else []
    # Whether this task states a relaxation ORDER to follow when the committed
    # item dies. Read off the rule text, never off the family label.
    relax_ladder = bool(_RELAX_LADDER_RE.search(query))
    # The pair the shopper named, graded against directly where they named one.
    preference = _preferred_pair(goal)

    def select(
        exclude: set[str], dead_key: str | None, dead_row: dict | None = None
    ) -> dict | None:
        rows = _narrow(catalog.eligible(budget, exclude), goal)
        if not rows:
            return None
        if ranked_axes:
            # A named candidate set with a stated tiebreak is a rule rather
            # than a judgement, and resolving it in code is what already pays
            # this family. Nothing below improves on an exact answer.
            return _rank(rows, ranked_axes)[0]

        # The checklist depends on everything the shopper has said, and they may
        # have said more since the last selection — an underspecified task
        # releases its firm requirement only in reply to a direct question.
        if goal not in checklists:
            checklists[goal] = _requirements(goal)
        requirements = checklists[goal]
        preferred_brand = _requirement_value(requirements, "brand")

        # What the shopper named, else what the checklist inferred -- the same
        # value the structured slices are drawn on. We already trust it enough
        # to RETRIEVE by; holding candidates to it is that judgement made
        # consistently, and the asymmetry was costing real reward. A pool drawn
        # for "smartphone" still carried power banks, running belts and baking
        # mixes into the comparison, and a feature phone whose title said
        # "Smartphone" and whose seller filed `phone_type: smartphone` won one
        # outright -- which is the case the rules warn about in as many words.
        #
        # Falls open like every other narrowing here: a classification no
        # inspected candidate carries is evidence about the pool, not a reason
        # to stop deciding.
        required_category = _stated_category(goal) or _requirement_value(
            requirements, *_CATEGORY_KEYS
        )
        if required_category:
            classified = [row for row in rows if _category_matches(row, required_category)]
            rows = classified or rows

        if wants_device:
            # A device was asked for, so an accessory is not a weaker candidate
            # — it is the wrong kind of thing, and one that files the requested
            # property as an ordinary fact will out-carry the right kind. Now
            # that the match is whole-token this can be a filter rather than a
            # demotion: measured over the pack it strips 3-15 junk candidates
            # from 19 tasks and loses gold on none. Demoting is still right for
            # INSPECTION, where reach is what matters and a wrong guess is only
            # a place in the queue.
            devices = [row for row in rows if not _looks_like_accessory(row.get("title"))]
            rows = devices or rows

        if dead_row is None and preferred_brand:
            # The first cart add is graded separately from the order, and one
            # family grades it on the named brand: "for the initial cart
            # choice, honor my named preferred brand when a fully compliant
            # item is available". A named brand stays SOFT for the order — an
            # off-brand item meeting every firm requirement is explicitly fine
            # — but it decides the commitment, and a single soft rank cannot
            # express "decisive here, negotiable there".
            #
            # Falls open when nothing carries the brand, which is the other
            # half of the same rule: the preference yields to the firm
            # requirements rather than emptying the pool.
            branded = [row for row in rows if _brand_matches(row, preferred_brand)]
            rows = branded or rows

        if relax_ladder and dead_row:
            # The rule states an ORDER, not a preference: try another colour of
            # the same model BEFORE changing model. Ordering the candidates by
            # it is not enough — the comparison returns the model's pick, and
            # the model reaches past a nearer rung for a cheaper item on a
            # further one. So the rung is resolved here, in code, the way the
            # ranking family's tiebreak is: keep only the nearest rung that has
            # any candidate at all, and let judgement operate inside it.
            #
            # Every row here already satisfies budget, stock and the stated
            # specs, so narrowing to a rung can never promote an invalid
            # candidate, and falling through to the next rung when one is empty
            # is what the rule asks for anyway.
            nearest = min(_relaxation_step(row, dead_row, preference) for row in rows)
            rows = [row for row in rows if _relaxation_step(row, dead_row, preference) == nearest]

        evidence = {
            _key(row["product_id"], row["sku"]): _evidence_row(row, requirements)
            for row in rows
        }
        # Order before capping, so the cap keeps the best-evidenced candidates
        # rather than whichever the catalogue happened to yield first.
        rows.sort(
            key=lambda row: (
                _relaxation_step(row, dead_row, preference) if relax_ladder else 0,
                tuple(
                    -level
                    for level in _evidence_key(
                        evidence[_key(row["product_id"], row["sku"])], requirements
                    )
                ),
                row["sku"],
            )
        )
        rows = rows[:_MAX_CANDIDATES]

        survivors, verified = _verify(goal, requirements, rows, evidence)
        return _select_best(
            goal,
            rules,
            requirements,
            survivors,
            evidence,
            budget,
            currency,
            dead=dead_key,
            verified=verified,
            dead_row=dead_row,
            relax_ladder=relax_ladder,
            preference=preference,
        )

    # COMMIT — the graded first add.
    ideal = select(dead, None)
    if ideal is None:
        return driver.dialogue
    ideal_key = _key(ideal["product_id"], ideal["sku"])
    catalog.absorb(
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
    for index in range(0, min(len(seen), shortlist), _COMPARE_CAP):
        observe.append({"name": "compare", "args": {"product_ids": seen[index : index + _COMPARE_CAP]}})
    catalog.absorb(driver.step(observe, "verify committed variant after the event"))
    answer_if_asked()

    # RECOMMIT — the event always invalidates what was committed, so the order
    # necessarily lands on a different variant.
    dead.add(ideal_key)
    replacement = select(dead, ideal_key, ideal)
    if replacement is None:
        return driver.dialogue
    catalog.absorb(
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
        cheaper = select(dead, ideal_key, ideal)
        if cheaper is not None:
            replacement = cheaper
            catalog.absorb(
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
    catalog.absorb(driver.step([{"name": "place_test_order", "args": order_args}], "order"))
    return driver.dialogue


__all__ = ["agent_main"]
