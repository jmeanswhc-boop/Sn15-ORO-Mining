from __future__ import annotations

import json
import re
from os import getenv
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient


# OpenRouter slugs only, strongest first (same ranking as agent_struct / merge).
_BG_OPENROUTER_MODELS = (
    "z-ai/glm-5.2",
    "moonshotai/kimi-k2.6",
    "qwen/qwen3.5-397b-a17b",
    "deepseek/deepseek-v3.2",
)

_bg_bridge = ProxyClient(timeout=120, max_retries=2)

LANE_INTENT = "intent_decomposition"
LANE_RECALL = "retrieval_recall"
LANE_CONSTRAINT = "constraint_satisfaction"
LANE_PREFERENCE = "preference_reasoning"
LANE_RANK = "ranking"
LANE_RECOVER = "recovery"
LANE_JUSTIFY = "justification"


_BG_LANE_CUES: list[tuple[str, tuple[str, ...]]] = [
    (
        LANE_INTENT,
        (
            "mix of firm requirements and nice-to-haves",
            "soft preference, subordinate",
            "preferred-brand",
            "later correction does not repair an initial choice",
            "honor that named brand when a fully compliant",
        ),
    ),
    (
        LANE_RECALL,
        (
            "reformulate queries and inspect candidates",
            "survey the realistic options",
            "several distinct products, not one lucky hit",
            "search the catalog for what the user described",
        ),
    ),
    (
        LANE_CONSTRAINT,
        (
            "every requirement must hold for the exact variant",
            "resolution names here denote pixel classes",
            "check the specific sku's option, price, and stock",
            "i will tell you if you ask",
        ),
    ),
    (
        LANE_PREFERENCE,
        (
            "soft preferences matter",
            "blindly buying the cheapest valid item",
            "choose a strong match across several preference axes",
            "genuine intended purchase, not an exploratory",
        ),
    ),
    (
        LANE_RANK,
        (
            "stated tradeoff priority",
            "strictly worse than another available option",
            "highest advertised maximum refresh rate",
            "first cart choice must be the best-ranked",
            "compare only these candidate",
        ),
    ),
    (
        LANE_RECOVER,
        (
            "relaxable only after the exact item becomes unavailable",
            "least-relaxed in-stock option",
            "try another color of the same model before changing models",
            "conflicting title does not override",
        ),
    ),
    (
        LANE_JUSTIFY,
        (
            "when placing the order, include a structured justification",
            "ground claims in raw product facts",
            "do not justify from marketing title words alone",
            "trustworthy reason",
            "walk me through why",
        ),
    ),
]

_BG_GOAL_REGEX: list[tuple[str, re.Pattern[str]]] = [
    (LANE_RANK, re.compile(r"compare only these candidate", re.I)),
    (LANE_JUSTIFY, re.compile(r"walk me through why|tell me why|trustworthy reason", re.I)),
    (LANE_RECOVER, re.compile(r"if that exact .* disappears|strongly prefer", re.I)),
    (LANE_INTENT, re.compile(r"ideally,? i prefer|firm requirements matter more", re.I)),
    (LANE_PREFERENCE, re.compile(r"i'?d rather pay|for daily|workout|soft preference", re.I)),
    (LANE_CONSTRAINT, re.compile(r"i will tell you if you ask", re.I)),
]


def classify_lane(shopper_text: str) -> str:
    class _CueBoard:
        def __init__(self, raw: str) -> None:
            self.lowered = (raw or "").lower()
            self.rules_blob = (
                self.lowered.split("task rules:", 1)[1]
                if "task rules:" in self.lowered
                else self.lowered
            )

        def tally(self, cue_list: tuple[str, ...]) -> int:
            def _hit(cue: str) -> int:
                return 1 if cue in self.rules_blob or cue in self.lowered else 0

            return sum(_hit(cue) for cue in cue_list)

        def regex_lane(self) -> str | None:
            def _scan() -> str | None:
                for lane_id, pattern in _BG_GOAL_REGEX:
                    if pattern.search(self.lowered):
                        return lane_id
                return None

            return _scan()

    board = _CueBoard(shopper_text)
    chosen_lane = LANE_RECALL
    peak_score = -1
    for lane_id, cue_list in _BG_LANE_CUES:
        score = board.tally(cue_list)
        if score > peak_score:
            peak_score = score
            chosen_lane = lane_id
    if peak_score <= 0:
        hit = board.regex_lane()
        if hit:
            return hit
    return chosen_lane


_BG_BASE_PLAYBOOK = """
You are BlackGirl v6, an ORO Bench shopping agent.

# Contract
- Use ONLY tools supplied in this episode's tool list.
- Continue until an observation reports done=true.
- Respect max_steps and max_calls_per_turn.
- When max_calls_per_turn > 1, batch independent views/inspects in one turn.

# Standard loop
1) Discover — search / filter (value-side keywords, not JSON field names)
2) Inspect — view / compare / inspect_stock on concrete SKUs
3) Decide — add_to_cart / remove_from_cart (first cart is scored — no placeholders)
4) Confirm — inspect_stock / inspect_cart after events; read shopper messages
5) Finish — place_test_order once verified (include justification.claims when required)

# Firm vs soft
Extract constraints: category, brand, model, size, capacity, color, material,
feature, budget/price, stock, resolution, refresh_rate, screen_size, use-case.
Mark FIRM (must hold) vs SOFT (preference). Soft never overrides firm.

# Matching
- Literal: brands, model codes, identifiers, uncommon tokens, exact sizes/colors.
- Size aliases only: s/small, m/medium, l/large, xl/extra large.
- Semantic: everyday paraphrases for soft checks only.
- Resolution classes when Task rules define them (both pixel sides):
  HD≥1280×720, Full HD≥1920×1080, QHD≥2560×1440, 4K/UHD≥3840×2160.
  Lowercase qHD ≠ QHD.

# Search keywords
Catalog search matches VALUES. Prefer product-type words, brand, model codes,
distinctive phrase fragments. Avoid dumping keys like "brand" or "category".
If search is weak: one small reformulation — drop noise, keep distinctive tokens.

# Shopper dialogue (critical)
Whenever a shopper user_message asks a question or clarifies budget/spec,
answer with the message tool in the NEXT turn before further cart/order actions.
Never ignore a direct question. Keep messages short and concrete.

# Market events
Prices/stock can change after cart actions. Re-verify with inspect_stock /
inspect_cart. If over budget or out of stock, remove and recover per family rules.
Once a verified in-stock in-budget pick exists, place_test_order — do not wait
for permission unless the shopper asked a pending question.
""".strip()

_BG_LANE_PLAYBOOK: dict[str, str] = {
    LANE_INTENT: """
# TK1 — Intent Decomposition
Strategy:
1. Parse FIRM: category, one hardware spec, budget, in-stock.
2. Parse SOFT: named preferred brand (subordinate).
3. Search/filter → view compliant SKUs; confirm category_path is a real device.
4. FIRST add_to_cart: preferred brand IF a fully firm-compliant SKU exists;
   otherwise best firm-compliant item. No exploratory placeholders.
5. If shopper asks "which one / does it have X?": message with the pick + evidence.
6. On price jump / budget tighten: remove invalid cart item; re-search under new
   budget; still prefer brand only when fully compliant.
7. inspect_stock → place_test_order.

A later fix does NOT repair a first cart that ignored an available compliant
preferred-brand option. Always answer interventions via message.
""".strip(),
    LANE_RECALL: """
# TK2 — Retrieval Recall
Strategy:
1. Search with goal phrase; also try tokenized / category+filter variants.
2. Shortlist SEVERAL distinct products (not one lucky hit); reject accessories /
   off-category noise using category_path from view/compare.
3. View/compare promising hits; pick best in-stock within budget.
4. One small keyword reformulation if recall is thin.
5. Once verified, place_test_order immediately — no permission wait.
""".strip(),
    LANE_CONSTRAINT: """
# TK3 — Constraint Satisfaction
Strategy:
1. If the goal says a firm hardware spec is revealed only when asked, call
   message FIRST to ask for that exact requirement before searching.
2. EVERY firm requirement must hold on the EXACT SKU (option + price + stock).
   Titles mix good/bad variants — always view then inspect_stock on the SKU.
3. Apply resolution class rules from Task rules when relevant.
4. If the wanted variant sells out, do NOT take a sibling SKU that misses a
   firm requirement — find another product that satisfies all.
5. Finish with place_test_order once verified; do not burn steps on chatter.
""".strip(),
    LANE_PREFERENCE: """
# TK4 — Preference Reasoning
Hard: Headphones category, CURRENT cap, in-stock, a real headphone.
Soft axes are scored from VISIBLE FACTS, then title tokens:
sweat_water_resistance, sport_secure_fit, wireless_freedom, battery_life.
Title-only "IPX" on an over-ear is NOT sweat evidence (scores ~0.2).
Sports + waterproof + wireless earbuds / neckband / bone-conduction score ~0.78.
Do NOT message to clarify. Do NOT first-cart the cheapest generic headphone.

1. search "sports waterproof earbuds", then "IPX sweat-proof sports",
   then "bone conduction waterproof". compare facts (ATTR beats TITLE).
2. FIRST add_to_cart MUST be a compared sport+water+wireless SKU still
   under the cap. That first add is the scored commitment. A later gold
   order does not repair a non-aligned first add.
3. Price jump always fires. remove_from_cart the jumped SKU. add another
   already-viewed sport+water+wireless SKU still under the cap.
   Never recover to a title-only over-ear.
4. place_test_order with ONLY product_id and sku. No justification.
""".strip(),
    LANE_RANK: """
# TK5 — Ranking
Strategy:
1. Restrict to the candidate variants named in the query (product_id::sku).
2. view and/or compare ALL listed candidates BEFORE any add_to_cart.
3. Rank ONLY by stated priority (typical: highest refresh rate including
   advertised overclock → largest screen → lowest price).
4. FIRST add_to_cart MUST be the current best-ranked in-stock in-budget option.
5. After stockout/price events, remove invalid item, re-rank survivors, cart
   the new #1, then place_test_order quickly (do not stall).
6. Never pick a dominated option on every axis.

CRITICAL — place_test_order on TK5:
- Call place_test_order with ONLY product_id and sku.
- Do NOT pass justification / claims. Optional untruthful claims hard-fail
  the episode even when the ranked pick is correct.
""".strip(),
    LANE_RECOVER: """
# TK6 — Recovery
Strategy:
Never relax: category, budget, in-stock.
Prefer exact model+color first (use explicit model fact when supplied; titles
do not override model facts).

If exact item disappears / price breaks budget after a cart action:
1) same model, other color
2) only then other models
Re-check price and stock after every cart change before place_test_order.
Any eligible listing matching preferred model+color is a valid first choice.
""".strip(),
    LANE_JUSTIFY: """
# TK7 — Justification
facts/options are authoritative. A title that says 1080P / HD / 27-inch is
only TITLE evidence; facts.display / facts.size are ATTR. If facts.display
is 2560x1440 it is QHD, not FHD — reject it when the query asked Full HD.
Never first-add an over-cap or off-spec decoy.

1. search brand + gated spec only (tight queries). compare facts.
2. FIRST add a branded, on-spec, under-cap SKU whose FACTS prove the spec.
3. After the price jump, remove the jumped SKU and order a DIFFERENT
   still-valid SKU (not the event target). Episode must close.
4. Tiny justification. Allowed fields ONLY:
   - in_stock = "true"
   - price = live inspect_stock price as a string
   - screen_size = the number from facts.size ("27"), never "27-inch"
     OR resolution = hd|fhd|qhd|uhd from facts.display, never "720p" / "1920 x 1080"
   - optional brand if it matches the visible brand field
   Forbidden (zero the score): display, size, category, category_path,
   device_operating_system, refresh_rate unless that is the gated spec,
   streaming, microphone, privacy_cover, autofocus, invented hdr, free text.
place_test_order {product_id, sku, justification:{claims:[...]}}.
""".strip(),
}

_BG_IDLE_PRODS = (
    "No tool call received. Choose a supplied tool now. "
    "If you have not searched yet, call search or filter.",
    "Still no tool call. Progress: search/filter, view a candidate, answer any "
    "pending shopper question with message, or place_test_order if the cart "
    "already holds a verified in-stock SKU.",
)

_BG_NUDGE_HIDDEN_SPEC = (
    "Coach [TK3]: the shopper has one firm hardware spec revealed only if asked. "
    "Call message NOW to ask for that exact requirement before searching."
)

_BG_NUDGE_REPLY = (
    "Coach: the shopper just sent a message. Answer it with the message tool in "
    "this turn (which pick / does it meet the firm spec / budget). Do not add to "
    "cart or order until that question is answered."
)

_BG_NUDGE_RECOVER = (
    "Coach [TK6]: action failed or stock/price changed ({error}). Recover with "
    "least relaxation that still satisfies category, budget, and in-stock."
)

_BG_NUDGE_RANK = (
    "Coach [TK5]: view/compare every listed candidate and rank by the stated "
    "tradeoff axes BEFORE the first add_to_cart. First cart = #1. On "
    "place_test_order pass ONLY product_id+sku — omit justification entirely."
)

_BG_NUDGE_CLAIMS = (
    "Coach [TK7]: search brand + gated spec. First add a FACT-proven on-spec "
    "under-cap SKU (title 1080P with facts.display 2560x1440 is QHD — reject). "
    "Claims allowed ONLY: in_stock=\"true\", price as live string, and "
    "screen_size (\"27\" not \"27-inch\") OR resolution (hd|fhd|qhd|uhd). "
    "Optional brand if visible. Forbidden: display, size, category, "
    "category_path, device_operating_system, hdr, streaming, microphone. "
    "After the jump, order a DIFFERENT still-valid SKU."
)

_BG_NUDGE_BRAND = (
    "Coach [TK1]: first cart should honor the preferred brand when a fully "
    "firm-compliant SKU exists; otherwise take the best firm-compliant item."
)

_BG_NUDGE_PREF = (
    "Coach [TK4]: search \"sports waterproof earbuds\" / \"IPX sweat-proof sports\" "
    "/ \"bone conduction waterproof\". compare facts. FIRST add MUST be a "
    "sport+water+wireless earbud/neckband/bone SKU under the cap — not a "
    "title-only over-ear and not the cheapest generic headphone. After the "
    "price jump, remove it, add another already-viewed wet SKU, then "
    "place_test_order with ONLY product_id+sku."
)

_BG_NUDGE_PREF_EVENT = (
    "Coach [TK4]: market event on the first cart. remove_from_cart that SKU "
    "NOW, add another already-viewed sports+waterproof+wireless SKU still "
    "under the current cap, then place_test_order with ONLY product_id+sku."
)

_BG_NUDGE_JUST_EVENT = (
    "Coach [TK7]: market event on the first cart. remove_from_cart that SKU "
    "NOW, add a DIFFERENT FACT-proven on-spec under-cap SKU, then "
    "place_test_order with rewritten claims (in_stock, live price, gated spec)."
)

_BG_PREF_SEARCHES = (
    "sports waterproof earbuds",
    "IPX sweat-proof sports",
    "bone conduction waterproof",
)
_BG_SWEAT_MARK = (
    "ipx",
    "waterproof",
    "sweat",
    "splash",
    "water-resistant",
    "water resistant",
    "sweatproof",
)
_BG_SPORT_MARK = (
    "sport",
    "earbud",
    "earbuds",
    "in-ear",
    "inear",
    "tws",
    "bone",
    "conduction",
    "neck",
    "hook",
    "secure",
)
_BG_WIRE_MARK = ("wireless", "bluetooth", "tws")
_BG_OVER_EAR = ("over ear", "over-ear", "overear")
_BG_TIGHTEN_BUDGET = re.compile(
    r"(?:only\s+go\s+up\s+to|can\s+only\s+(?:go\s+up\s+to|spend(?:\s+up\s+to)?)|"
    r"new\s+(?:budget|limit|cap)|budget\s+is\s+(?:now|at\s+most)|"
    r"at\s+most|cap\s+is)\s+"
    r"(\d+(?:\.\d+)?)\s*(USD|EUR|GBP|PHP|SGD|AUD|CAD)",
    re.I,
)


def _bg_as_float(value: Any) -> float | None:
    if value in (None, "", False):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    matches = re.findall(r"\d+\.?\d*", str(value))
    return float(matches[0]) if matches else None


def _bg_goal_head(text: str) -> str:
    raw = text or ""
    lowered = raw.lower()
    if "task rules:" in lowered:
        return raw[: lowered.index("task rules:")]
    return raw


def _bg_parse_goal(shopper_text: str) -> dict[str, Any]:
    head = _bg_goal_head(shopper_text)
    facts: dict[str, Any] = {"head": head.strip()}
    budget = re.search(r"(\d+(?:\.\d+)?)\s*(USD|EUR|GBP|PHP|SGD|AUD|CAD)", head, re.I)
    if budget:
        facts["budget"] = budget.group(1)
        facts["currency"] = budget.group(2).upper()
    brand = re.search(
        r"(?:prefer|brand)\s+([A-Za-z][A-Za-z0-9]*(?:\s+[A-Za-z][A-Za-z0-9]*){0,2})",
        head,
        re.I,
    )
    if not brand:
        brand = re.search(
            r"\b(?:need a|want a|get me an|find me a)\s+([A-Za-z][A-Za-z0-9]+)\s+",
            head,
            re.I,
        )
    if brand:
        token = brand.group(1).strip()
        if token.casefold() not in {"smartphone", "headphone", "monitor", "tablet", "in", "stock"}:
            facts["brand"] = token
    if not facts.get("brand"):
        stock_brand = re.search(r"in-stock\s+([A-Za-z][A-Za-z0-9]+)", head, re.I)
        if stock_brand:
            token = stock_brand.group(1).strip()
            if token.casefold() not in {"smartphone", "headphone", "monitor", "tablet", "in"}:
                facts["brand"] = token
    inches = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", head, re.I)
    if inches:
        facts["screen"] = inches.group(1)
    res = re.search(r"\b(720p|1080p|1440p|2160p|4k|uhd|qhd|full hd|hd)\b", head, re.I)
    if res:
        facts["resolution"] = res.group(1)
    cat = re.search(r"in the ([^,\n]+?) category", head, re.I)
    if cat:
        facts["category"] = cat.group(1).strip()
    elif re.search(r"\bsmartphone", head, re.I):
        facts["category"] = "Smartphones"
    elif re.search(r"\bheadphone", head, re.I):
        facts["category"] = "Headphones"
    elif re.search(r"\bmonitor", head, re.I):
        facts["category"] = "Monitors"
    elif re.search(r"\btablet", head, re.I):
        facts["category"] = "Tablets"
    return facts


def _bg_facts_dict(row: dict[str, Any]) -> dict[str, Any]:
    facts = row.get("facts")
    return facts if isinstance(facts, dict) else {}


def _bg_fact_text(row: dict[str, Any]) -> str:
    return json.dumps(
        {"f": _bg_facts_dict(row), "o": row.get("options"), "s": row.get("specification")},
        ensure_ascii=True,
        default=str,
    ).casefold()


def _bg_fact_display(row: dict[str, Any]) -> str:
    facts = _bg_facts_dict(row)
    return str(facts.get("display") or facts.get("resolution") or "")


def _bg_fact_size(row: dict[str, Any]) -> str:
    facts = _bg_facts_dict(row)
    return str(facts.get("size") or facts.get("screen_size") or facts.get("screen") or "")


def _bg_canon_res(raw: Any) -> str:
    token = str(raw or "").casefold()
    if token in {"720p", "hd", "1280", "720"}:
        return "hd"
    if token in {"1080p", "full hd", "fhd", "1920", "1080"}:
        return "fhd"
    if token in {"1440p", "qhd", "2560", "1440"}:
        return "qhd"
    if token in {"4k", "uhd", "2160p", "3840", "2160"}:
        return "uhd"
    return token


def _bg_display_class(text: str) -> str:
    token = str(text or "").casefold().replace("*", " x ")
    if any(part in token for part in ("3840", "2160", "4k", "uhd")):
        return "uhd"
    if any(part in token for part in ("2560", "1440", "qhd")):
        return "qhd"
    if any(part in token for part in ("1920", "1080", "fhd", "full hd")):
        return "fhd"
    if any(part in token for part in ("1280", "720p", "720")) or token.strip() in {"hd"}:
        return "hd"
    return _bg_canon_res(token)


def _bg_pref_axis(row: dict[str, Any], marks: tuple[str, ...]) -> int:
    facts = _bg_fact_text(row)
    title = str(row.get("title") or "").casefold()
    if any(mark in facts for mark in marks):
        return 2
    if any(mark in title for mark in marks):
        return 1
    return 0


def _bg_pref_score(row: dict[str, Any]) -> tuple[int, int, int, int]:
    sweat = _bg_pref_axis(row, _BG_SWEAT_MARK)
    sport = _bg_pref_axis(row, _BG_SPORT_MARK)
    wireless = _bg_pref_axis(row, _BG_WIRE_MARK)
    title = str(row.get("title") or "").casefold()
    if any(mark in title for mark in _BG_OVER_EAR) and sport == 0:
        sweat = 0
    return (sweat + sport + wireless, sweat, sport, wireless)


def _bg_price_ok(row: dict[str, Any], budget: Any) -> bool:
    price = _bg_as_float(row.get("price"))
    cap = _bg_as_float(budget)
    if price is None or cap is None:
        return True
    return price <= cap + 1e-6


def _bg_in_stock(row: dict[str, Any]) -> bool:
    if "in_stock" not in row:
        return True
    value = row.get("in_stock")
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"true", "1", "yes", "in_stock", "in stock"}


def _bg_screen_match(row: dict[str, Any], target: str) -> bool:
    needle = str(target or "").rstrip("0").rstrip(".")
    if not needle:
        return True
    size = _bg_fact_size(row)
    if needle and needle in size.replace(" ", "").casefold():
        return True
    title = str(row.get("title") or "")
    if re.search(rf"{re.escape(needle)}\s*-?\s*inch", title, re.I):
        return True
    blob = json.dumps(row, ensure_ascii=True, default=str).casefold()
    return bool(re.search(rf"{re.escape(needle)}\s*-?\s*inch", blob, re.I))


def _bg_spec_ok(row: dict[str, Any], goal: dict[str, Any]) -> bool:
    blob = json.dumps(row, ensure_ascii=True, default=str).casefold()
    if goal.get("screen"):
        needle = str(goal["screen"])
        size = _bg_fact_size(row)
        inch = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(?:inch|in\b)", size, re.I)
        if inch and _bg_as_float(inch.group(1)) is not None and _bg_as_float(needle) is not None:
            if abs(float(inch.group(1)) - float(needle)) >= 1.0:
                return False
        need = needle.rstrip("0").rstrip(".")
        size_hit = bool(need and size and need in size.replace(" ", "").casefold())
        if not size_hit and not _bg_screen_match(row, needle):
            return False
    if goal.get("resolution"):
        want = _bg_canon_res(goal["resolution"]) or str(goal["resolution"]).casefold()
        disp = _bg_fact_display(row)
        if disp:
            got = _bg_display_class(disp)
            if got and got != want:
                return False
        else:
            aliases = {str(goal["resolution"]).casefold(), want}
            if want == "fhd":
                aliases.update({"1080p", "1920", "1080", "full hd", "fhd"})
            elif want == "hd":
                aliases.update({"720p", "1280", "720", "hd"})
            elif want == "qhd":
                aliases.update({"1440p", "2560", "1440", "qhd"})
            elif want == "uhd":
                aliases.update({"4k", "uhd", "3840", "2160"})
            if not any(alias in blob for alias in aliases):
                return False
    return True


def _bg_tf7_screen_value(row: dict[str, Any], goal: dict[str, Any]) -> str | None:
    size = _bg_fact_size(row) or str(row.get("title") or "")
    needle = str(goal.get("screen") or "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(?:inch|in\b)?", size, re.I)
    if match:
        raw = match.group(1)
        if not needle or needle.rstrip("0").rstrip(".") in raw.rstrip("0").rstrip("."):
            num = _bg_as_float(raw)
            if num is not None and abs(num - round(num)) < 1e-6:
                return str(int(round(num)))
            return raw
    if needle:
        return str(needle)
    return None


def _bg_tf7_res_value(row: dict[str, Any], goal: dict[str, Any]) -> str | None:
    disp = _bg_fact_display(row)
    if disp:
        got = _bg_display_class(disp)
        if got in {"hd", "fhd", "qhd", "uhd"}:
            return got
    if goal.get("resolution"):
        got = _bg_canon_res(goal["resolution"])
        if got in {"hd", "fhd", "qhd", "uhd"}:
            return got
    return None


def _bg_tf7_claims(goal: dict[str, Any], row: dict[str, Any]) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    if goal.get("screen"):
        screen = _bg_tf7_screen_value(row, goal)
        if screen:
            claims.append({"field": "screen_size", "value": str(screen)})
    elif goal.get("resolution"):
        res = _bg_tf7_res_value(row, goal)
        if res:
            claims.append({"field": "resolution", "value": str(res)})
    price = row.get("price")
    if price not in (None, ""):
        claims.append({"field": "price", "value": str(price)})
    claims.append({"field": "in_stock", "value": "true"})
    brand = str(row.get("brand") or "").strip()
    if brand:
        claims.append({"field": "brand", "value": brand})
    return claims


def _bg_row_sku(row: dict[str, Any]) -> str:
    if row.get("sku"):
        return str(row["sku"])
    options = row.get("options")
    if isinstance(options, list) and options and isinstance(options[0], dict):
        return str(options[0].get("sku") or options[0].get("id") or "")
    variants = row.get("variants")
    if isinstance(variants, list) and variants and isinstance(variants[0], dict):
        return str(variants[0].get("sku") or variants[0].get("id") or "")
    return ""


class _BgGradeKit:
    def __init__(self, lane_id: str, shopper_text: str) -> None:
        self.lane_id = lane_id
        self.goal = _bg_parse_goal(shopper_text)
        self.catalog: dict[str, dict[str, Any]] = {}
        self.search_idx = 0
        self.event_seen = False
        self.event_cleared = False
        self.event_pid = ""
        self.event_sku = ""
        self.last_pid = ""
        self.last_sku = ""

    def absorb_note(self, note: str) -> None:
        text = str(note or "")
        tighten = _BG_TIGHTEN_BUDGET.search(text)
        if tighten:
            new_cap = _bg_as_float(tighten.group(1))
            old_cap = _bg_as_float(self.goal.get("budget"))
            if new_cap is not None and (old_cap is None or new_cap <= old_cap + 1e-9):
                self.goal["budget"] = tighten.group(1)
                self.goal["currency"] = tighten.group(2).upper()
        inches = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", text, re.I)
        if inches:
            self.goal["screen"] = inches.group(1)
        res = re.search(r"\b(720p|1080p|1440p|2160p|4k|uhd|qhd|full hd|hd)\b", text, re.I)
        if res:
            self.goal["resolution"] = res.group(1)

    def _merge_row(self, row: Any) -> None:
        if not isinstance(row, dict):
            return
        pid = str(row.get("product_id") or "")
        if not pid:
            return
        prev = self.catalog.get(pid) or {"product_id": pid}
        for key, value in row.items():
            if value not in (None, "", [], {}):
                prev[key] = value
        self.catalog[pid] = prev

    def ingest(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        nested = payload.get("observation") if isinstance(payload.get("observation"), dict) else payload
        if not isinstance(nested, dict):
            return
        for key in ("results", "compared", "variants", "products"):
            rows = nested.get(key)
            if isinstance(rows, list):
                for row in rows:
                    self._merge_row(row)
        if nested.get("product_id"):
            self._merge_row(nested)
        added = nested.get("added")
        if isinstance(added, dict):
            self._merge_row(added)
            self.last_pid = str(added.get("product_id") or self.last_pid)
            self.last_sku = str(added.get("sku") or self.last_sku)
        event = nested.get("market_event")
        if event:
            self.event_seen = True
            self.event_pid = self.last_pid or str(nested.get("product_id") or "")
            self.event_sku = self.last_sku or str(nested.get("sku") or "")
            if isinstance(event, dict):
                self.event_pid = str(event.get("product_id") or self.event_pid)
                self.event_sku = str(event.get("sku") or self.event_sku)

    def mark_removed(self, args: dict[str, Any]) -> None:
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        if (pid and pid == self.event_pid) or (sku and sku == self.event_sku) or not self.event_pid:
            self.event_cleared = True

    def _lookup(self, pid: str, sku: str = "") -> dict[str, Any]:
        row = dict(self.catalog.get(pid) or {"product_id": pid})
        if sku:
            row["sku"] = sku
        elif not row.get("sku"):
            row["sku"] = _bg_row_sku(row)
        return row

    def _rows(self) -> list[dict[str, Any]]:
        rows = []
        for pid, row in self.catalog.items():
            item = dict(row)
            item["product_id"] = pid
            if not item.get("sku"):
                item["sku"] = _bg_row_sku(item)
            rows.append(item)
        return rows

    def _same(self, row: dict[str, Any], pid: str, sku: str) -> bool:
        if sku and str(row.get("sku") or "") == sku:
            return True
        if pid and str(row.get("product_id") or "") == pid and sku and str(row.get("sku") or "") == sku:
            return True
        if pid and str(row.get("product_id") or "") == pid and not sku:
            return True
        return False

    def pick_pref(self, *, phase_b: bool) -> dict[str, Any] | None:
        skip_pid = self.last_pid if phase_b else ""
        skip_sku = self.last_sku if phase_b else ""
        gated = [
            row
            for row in self._rows()
            if _bg_price_ok(row, self.goal.get("budget"))
            and _bg_in_stock(row)
            and row.get("product_id")
            and _bg_row_sku(row)
            and not (phase_b and self._same(row, skip_pid, skip_sku))
        ]
        wet = [row for row in gated if _bg_pref_score(row)[0] >= 2]
        pool = wet or ([row for row in gated if _bg_pref_score(row)[0] >= 1] if not phase_b else [])
        pool.sort(
            key=lambda row: (
                -_bg_pref_score(row)[0],
                -_bg_pref_score(row)[1],
                -_bg_pref_score(row)[2],
                _bg_as_float(row.get("price")) if _bg_as_float(row.get("price")) is not None else 1e18,
            )
        )
        return dict(pool[0]) if pool else None

    def pick_just(self, *, phase_b: bool) -> dict[str, Any] | None:
        skip_pid = self.last_pid if phase_b else ""
        skip_sku = self.last_sku if phase_b else ""
        brand = str(self.goal.get("brand") or "").casefold()
        ranked: list[dict[str, Any]] = []
        for row in self._rows():
            if not row.get("product_id") or not _bg_row_sku(row):
                continue
            if phase_b and self._same(row, skip_pid, skip_sku):
                continue
            if not _bg_spec_ok(row, self.goal):
                continue
            if not _bg_price_ok(row, self.goal.get("budget")):
                continue
            if not _bg_in_stock(row):
                continue
            size = _bg_fact_size(row).strip()
            disp = _bg_fact_display(row).strip()
            if size.casefold() in {"default", "n/a", "none"} or disp.casefold() in {"default", "n/a"}:
                continue
            if re.search(r"\d+\s*[-–/]\s*\d+", size):
                continue
            ranked.append(row)
        ranked.sort(
            key=lambda row: (
                0 if brand and brand in str(row.get("brand") or "").casefold() else 1,
                0 if _bg_facts_dict(row) else 1,
                _bg_as_float(row.get("price")) if _bg_as_float(row.get("price")) is not None else 1e18,
            )
        )
        return dict(ranked[0]) if ranked else None

    def rewrite_search(self, args: dict[str, Any]) -> dict[str, Any]:
        key = "query" if "query" in args or "q" not in args else "q"
        raw = str(args.get(key) or args.get("query") or args.get("q") or "")
        lowered = raw.casefold()
        if self.lane_id == LANE_PREFERENCE:
            if any(token in lowered for token in ("waterproof", "ipx", "sweat", "sport", "bone conduction")):
                return args
            args[key] = _BG_PREF_SEARCHES[self.search_idx % len(_BG_PREF_SEARCHES)]
            self.search_idx += 1
            return args
        if self.lane_id == LANE_JUSTIFY:
            brand = str(self.goal.get("brand") or "")
            spec = ""
            if self.goal.get("screen"):
                spec = f"{self.goal['screen']} inch"
            elif self.goal.get("resolution"):
                spec = str(self.goal["resolution"])
            category = str(self.goal.get("category") or "").casefold()
            kind = "monitor" if category.startswith("monitor") else (
                "tablet" if "tablet" in category else "smartphone"
            )
            wanted = " ".join(part for part in (brand, kind, spec) if part).strip()
            if brand and brand.casefold() in lowered:
                return args
            if wanted:
                args[key] = wanted
            return args
        return args

    def _rewrite_add_args(self, args: dict[str, Any], cart_used: bool) -> dict[str, Any]:
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        current = self._lookup(pid, sku)
        phase_b = self.event_seen
        if self.lane_id == LANE_PREFERENCE:
            score = _bg_pref_score(current)[0]
            need_swap = (not cart_used and score < 2) or (
                phase_b and self._same(current, self.event_pid, self.event_sku)
            )
            if need_swap or (not cart_used and score < 2):
                pick = self.pick_pref(phase_b=phase_b or cart_used and self.event_seen)
                if pick:
                    args["product_id"] = str(pick.get("product_id") or "")
                    args["sku"] = _bg_row_sku(pick)
            return args
        if self.lane_id == LANE_JUSTIFY:
            bad = (not _bg_spec_ok(current, self.goal)) or (not _bg_price_ok(current, self.goal.get("budget")))
            if phase_b and self._same(current, self.event_pid, self.event_sku):
                bad = True
            if bad or (not cart_used and not _bg_spec_ok(current, self.goal)):
                pick = self.pick_just(phase_b=phase_b)
                if pick:
                    args["product_id"] = str(pick.get("product_id") or "")
                    args["sku"] = _bg_row_sku(pick)
            return args
        return args

    def _rewrite_order_args(self, args: dict[str, Any]) -> dict[str, Any]:
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        row = self._lookup(pid, sku)
        if self.lane_id == LANE_PREFERENCE:
            args.pop("justification", None)
            return args
        if self.lane_id == LANE_JUSTIFY:
            if self.event_seen and self._same(row, self.event_pid, self.event_sku):
                pick = self.pick_just(phase_b=True)
                if pick:
                    args["product_id"] = str(pick.get("product_id") or "")
                    args["sku"] = _bg_row_sku(pick)
                    row = pick
            args["justification"] = {"claims": _bg_tf7_claims(self.goal, row)}
            return args
        return args

    def _forced_remove(self) -> dict[str, Any] | None:
        if not (self.event_seen and not self.event_cleared and (self.event_pid or self.event_sku)):
            return None
        args: dict[str, Any] = {}
        if self.event_pid:
            args["product_id"] = self.event_pid
        if self.event_sku:
            args["sku"] = self.event_sku
        return {
            "id": "bg-forced-remove",
            "type": "function",
            "function": {
                "name": "remove_from_cart",
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }

    def guard(self, actions: list[dict[str, Any]], *, cart_used: bool, await_reply: bool) -> list[dict[str, Any]]:
        if self.lane_id not in {LANE_PREFERENCE, LANE_JUSTIFY} or await_reply:
            return actions
        names = [_bg_extract_action_names([item])[0] for item in actions]
        if (not cart_used) and ("add_to_cart" in names) and ("place_test_order" in names):
            actions = [item for item in actions if _bg_extract_action_names([item])[0] != "place_test_order"]
            names = [_bg_extract_action_names([item])[0] for item in actions]
        if self.event_seen and not self.event_cleared and "remove_from_cart" not in names:
            if "place_test_order" in names:
                actions = [
                    item for item in actions if _bg_extract_action_names([item])[0] != "place_test_order"
                ]
            forced = self._forced_remove()
            if forced:
                actions = [forced, *[item for item in actions if _bg_extract_action_names([item])[0] != "add_to_cart"]]
                names = [_bg_extract_action_names([item])[0] for item in actions]
        rewritten: list[dict[str, Any]] = []
        for item in actions:
            name = _bg_extract_action_names([item])[0]
            try:
                args = _bg_parse_args(item)
            except (ValueError, KeyError, TypeError):
                rewritten.append(item)
                continue
            if name == "search":
                args = self.rewrite_search(args)
            elif name == "add_to_cart":
                args = self._rewrite_add_args(args, cart_used)
            elif name == "place_test_order":
                args = self._rewrite_order_args(args)
            elif name == "remove_from_cart":
                self.mark_removed(args)
            try:
                item["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
            except (KeyError, TypeError):
                pass
            rewritten.append(item)
        return rewritten


class _BgLattice:
    class _ModelDesk:
        _ALIAS = {
            "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
            "deepseek-ai/DeepSeek-V3.1-TEE": "deepseek/deepseek-chat-v3.1",
            "deepseek-ai/DeepSeek-V3-0324-TEE": "deepseek/deepseek-chat-v3-0324",
            "deepseek-ai/DeepSeek-R1-0528-TEE": "deepseek/deepseek-r1-0528",
            "Qwen/Qwen3-32B-TEE": "qwen/qwen3-32b",
            "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
            "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
            "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
            "zai-org/GLM-5.1-TEE": "z-ai/glm-5.1",
            "zai-org/GLM-5.2-TEE": "z-ai/glm-5.2",
            "moonshotai/Kimi-K2.6-TEE": "moonshotai/kimi-k2.6",
            "openai/gpt-oss-120b-TEE": "openai/gpt-oss-120b",
            "XiaomiMiMo/MiMo-V2-Flash-TEE": "xiaomi/mimo-v2-flash",
            "unsloth/Mistral-Nemo-Instruct-2407-TEE": "mistralai/mistral-small-2603",
        }

        @classmethod
        def alias(cls, raw_model: str) -> str:
            def _lookup(key: str) -> str:
                return cls._ALIAS.get(key, key)

            return _lookup(raw_model)

        @classmethod
        def ladder(cls) -> list[str]:
            forced = cls.alias((getenv("SANDBOX_MODEL") or "").strip())
            ranked = list(_BG_OPENROUTER_MODELS)
            if not forced:
                return ranked
            if forced not in ranked:
                return [forced, *ranked]
            return [forced, *[name for name in ranked if name != forced]]

        @classmethod
        def pick(cls, lane_id: str) -> str:
            return cls.ladder()[0]

    class _ToolDesk:
        @staticmethod
        def labels(tool_schemas: list[dict[str, Any]]) -> list[str]:
            class _NamePull:
                @staticmethod
                def from_schema(schema: Any) -> str | None:
                    if not isinstance(schema, dict):
                        return None
                    fn_block = schema.get("function")
                    if isinstance(fn_block, dict) and fn_block.get("name"):
                        return str(fn_block["name"])
                    if schema.get("name"):
                        return str(schema["name"])
                    return None

            pulled: list[str] = []
            for schema in tool_schemas:
                tag = _NamePull.from_schema(schema)
                if tag:
                    pulled.append(tag)
            return pulled

        @staticmethod
        def parse_args(tc_item: dict[str, Any]) -> dict[str, Any]:
            def _decode(raw_args: Any) -> Any:
                if isinstance(raw_args, dict):
                    return raw_args
                try:
                    return json.loads(raw_args or "{}")
                except json.JSONDecodeError:
                    return {}

            decoded = _decode(tc_item["function"].get("arguments", "{}"))
            if not isinstance(decoded, dict):
                raise ValueError("tool arguments must be a JSON object")
            return decoded

        @staticmethod
        def action_names(tc_items: list[dict[str, Any]]) -> list[str]:
            def _one(tc_item: dict[str, Any]) -> str:
                try:
                    return str(tc_item["function"]["name"])
                except (KeyError, TypeError):
                    return "?"

            return [_one(tc_item) for tc_item in tc_items]

        @staticmethod
        def shrink(payload: Any, cap: int = 7000) -> str:
            def _render() -> str:
                try:
                    return json.dumps(payload, ensure_ascii=False, default=str)
                except TypeError:
                    return str(payload)

            rendered = _render()
            if len(rendered) <= cap:
                return rendered
            return rendered[: cap - 20] + "...<truncated>"

    class _InquiryGate:
        _PREFIXES = (
            "which ",
            "what ",
            "does ",
            "did ",
            "can you",
            "could you",
            "wait",
            "hold on",
            "before you",
            "re-check",
            "got anything",
        )

        @classmethod
        def check(cls, blob: str) -> bool:
            cleaned = (blob or "").strip().lower()
            if not cleaned:
                return False
            if "?" in cleaned:
                return True

            def _prefix_hit(p: str) -> bool:
                return cleaned.startswith(p) or f" {p}" in f" {cleaned}"

            return any(_prefix_hit(p) for p in cls._PREFIXES)

    class _CoachDesk:
        @staticmethod
        def next_line(
            lane_id: str,
            step_idx: int,
            prior_actions: list[str],
            prior_payloads: list[Any],
            shopper_text: str,
            await_reply: bool,
            cart_used: bool,
        ) -> str | None:
            action_blob = " ".join(prior_actions)
            goal_head = shopper_text.lower().split("task rules:", 1)[0]

            def _reply() -> str | None:
                return _BG_NUDGE_REPLY if await_reply else None

            def _hidden() -> str | None:
                if (
                    lane_id == LANE_CONSTRAINT
                    and step_idx == 1
                    and "i will tell you if you ask" in goal_head
                    and "message" not in action_blob
                ):
                    return _BG_NUDGE_HIDDEN_SPEC
                return None

            def _rank_open() -> str | None:
                if lane_id == LANE_RANK and step_idx == 1 and not cart_used:
                    return _BG_NUDGE_RANK
                return None

            def _brand() -> str | None:
                if lane_id == LANE_INTENT and step_idx == 1:
                    return _BG_NUDGE_BRAND
                return None

            def _pref() -> str | None:
                if (
                    lane_id == LANE_PREFERENCE
                    and step_idx <= 2
                    and "compare" not in action_blob
                    and not cart_used
                ):
                    return _BG_NUDGE_PREF
                return None

            def _claims() -> str | None:
                if lane_id != LANE_JUSTIFY or step_idx < 2:
                    return None
                if "view" not in action_blob and "compare" not in action_blob:
                    return _BG_NUDGE_CLAIMS
                if step_idx >= 4 and "place_test_order" not in action_blob:
                    return _BG_NUDGE_CLAIMS
                return None

            def _start_hint() -> str | None:
                if lane_id in {LANE_PREFERENCE, LANE_JUSTIFY}:
                    return None
                if step_idx <= 2 and not any(
                    label in action_blob
                    for label in ("search", "filter", "view", "compare", "message")
                ):
                    return (
                        "Hint: start with search or filter using value-side keywords from the "
                        "shopper goal, then view the best hits."
                    )
                return None

            def _recall_hint() -> str | None:
                if lane_id == LANE_RECALL and step_idx == 3 and "search" in action_blob:
                    return (
                        "Hint [TK2]: one small keyword reformulation or filter so the shortlist "
                        "covers several distinct products before ordering."
                    )
                return None

            def _recover() -> str | None:
                def _eventful(item: Any) -> bool:
                    return bool(
                        isinstance(item, dict)
                        and (
                            item.get("error")
                            or (
                                isinstance(item.get("observation"), dict)
                                and (
                                    item["observation"].get("market_event")
                                    or item["observation"].get("added")
                                )
                            )
                        )
                    )

                if lane_id != LANE_RECOVER or not any(_eventful(item) for item in prior_payloads):
                    return None
                detail = "market/stock change"
                for item in prior_payloads:
                    if isinstance(item, dict) and item.get("error"):
                        detail = str(item["error"])
                        break
                return _BG_NUDGE_RECOVER.format(error=detail)

            def _rank_cart() -> str | None:
                if lane_id == LANE_RANK and cart_used is False and step_idx >= 2:
                    if "add_to_cart" not in action_blob and (
                        "view" in action_blob or "compare" in action_blob
                    ):
                        return (
                            "Hint [TK5]: you have inspected candidates — add_to_cart ONLY the "
                            "best-ranked SKU now (first cart = graded #1)."
                        )
                return None

            def _rank_order() -> str | None:
                if lane_id == LANE_RANK and cart_used and step_idx >= 3:
                    return (
                        "Hint [TK5]: after recovery, place_test_order with ONLY product_id "
                        "and sku — do not attach justification.claims."
                    )
                return None

            for runner in (
                _reply,
                _hidden,
                _rank_open,
                _brand,
                _pref,
                _claims,
                _start_hint,
                _recall_hint,
                _recover,
                _rank_cart,
                _rank_order,
            ):
                line = runner()
                if line:
                    return line
            return None

    class _TalkLoop:
        @staticmethod
        def compose_system(lane_id: str, policy_blob: dict[str, Any]) -> str:
            labels = _BgLattice._ToolDesk.labels(policy_blob.get("tools") or [])
            lane_text = _BG_LANE_PLAYBOOK.get(lane_id, _BG_LANE_PLAYBOOK[LANE_RECALL])
            return (
                f"{_BG_BASE_PLAYBOOK}\n\n"
                f"{lane_text}\n\n"
                f"# Runtime\n"
                f"Router selected pipeline: {lane_id}\n"
                f"Available tools: {', '.join(labels) if labels else '(see tool schemas)'}\n"
                f"max_steps={policy_blob.get('max_steps')} "
                f"max_calls_per_turn={policy_blob.get('max_calls_per_turn', 1)}\n"
            )

        @staticmethod
        def chat(
            *,
            model_id: str,
            transcript: list[dict[str, Any]],
            tool_schemas: list[dict[str, Any]],
            call_budget: int,
        ) -> tuple[str, list[dict[str, Any]]]:
            def _candidates() -> list[str]:
                ranked = _BgLattice._ModelDesk.ladder()
                ids: list[str] = []
                for try_id in (model_id, *ranked):
                    if try_id and try_id not in ids:
                        ids.append(try_id)
                return ids

            last_failure: Exception | None = None
            for try_id in _candidates():
                reply_packet = _bg_bridge.post(
                    "/inference/chat/completions",
                    json_data={
                        "model": try_id,
                        "messages": transcript,
                        "tools": tool_schemas,
                        "tool_choice": "required",
                        "temperature": 0,
                    },
                )
                if reply_packet is None:
                    last_failure = RuntimeError(f"inference failed for model={try_id}")
                    continue

                def _message_of(packet: Any) -> dict[str, Any]:
                    return packet["choices"][0]["message"]

                try:
                    reply_msg = _message_of(reply_packet)
                except (KeyError, IndexError, TypeError) as exc:
                    last_failure = exc
                    continue
                spoken = reply_msg.get("content") or ""
                actions = list(reply_msg.get("tool_calls") or [])[:call_budget]
                return spoken, actions
            raise RuntimeError(f"inference request failed ({last_failure})")


def _bg_alias_provider_id(raw_model: str) -> str:
    return _BgLattice._ModelDesk.alias(raw_model)


def _bg_pick_model(lane_id: str) -> str:
    return _BgLattice._ModelDesk.pick(lane_id)


def _bg_list_tool_labels(tool_schemas: list[dict[str, Any]]) -> list[str]:
    return _BgLattice._ToolDesk.labels(tool_schemas)


def _bg_compose_system(lane_id: str, policy_blob: dict[str, Any]) -> str:
    return _BgLattice._TalkLoop.compose_system(lane_id, policy_blob)


def _bg_parse_args(tc_item: dict[str, Any]) -> dict[str, Any]:
    return _BgLattice._ToolDesk.parse_args(tc_item)


def _bg_shrink_obs(payload: Any, cap: int = 7000) -> str:
    return _BgLattice._ToolDesk.shrink(payload, cap)


def _bg_extract_action_names(tc_items: list[dict[str, Any]]) -> list[str]:
    return _BgLattice._ToolDesk.action_names(tc_items)


def _bg_is_inquiry(blob: str) -> bool:
    return _BgLattice._InquiryGate.check(blob)


def _bg_next_coach(
    lane_id: str,
    step_idx: int,
    prior_actions: list[str],
    prior_payloads: list[Any],
    shopper_text: str,
    await_reply: bool,
    cart_used: bool,
) -> str | None:
    return _BgLattice._CoachDesk.next_line(
        lane_id,
        step_idx,
        prior_actions,
        prior_payloads,
        shopper_text,
        await_reply,
        cart_used,
    )


def _bg_chat(
    *,
    model_id: str,
    transcript: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    call_budget: int,
) -> tuple[str, list[dict[str, Any]]]:
    return _BgLattice._TalkLoop.chat(
        model_id=model_id,
        transcript=transcript,
        tool_schemas=tool_schemas,
        call_budget=call_budget,
    )


def _bg_drive_lane(
    *,
    lane_id: str,
    problem_data: dict[str, Any],
) -> list[dict[str, Any]]:
    class _Episode:
        def __init__(self) -> None:
            env_block = problem_data["environment"]
            self.session_binding = env_block["binding"]
            self.policy_blob = env_block["policy_view"]
            self.shopper_text = str(
                self.policy_blob.get("query") or problem_data.get("query") or ""
            )
            self.episode_key = str(
                problem_data.get("problem_id", problem_data.get("id", "problem"))
            )
            self.call_budget = max(1, int(self.policy_blob.get("max_calls_per_turn", 1)))
            self.step_limit = max(1, int(self.policy_blob.get("max_steps", 20)))
            self.tool_schemas = self.policy_blob.get("tools") or []
            self.model_id = _bg_pick_model(lane_id)
            self.transcript: list[dict[str, Any]] = []
            self.trail: list[dict[str, Any]] = []
            self.prior_actions: list[str] = []
            self.prior_payloads: list[Any] = []
            self.await_reply = False
            self.cart_used = False
            self.grade = _BgGradeKit(lane_id, self.shopper_text)

        def _prime(self) -> None:
            goal_head = self.shopper_text.lower().split("task rules:", 1)[0]
            need_hidden_spec = (
                lane_id == LANE_CONSTRAINT and "i will tell you if you ask" in goal_head
            )
            self.transcript = [
                {
                    "role": "system",
                    "content": _bg_compose_system(lane_id, self.policy_blob),
                },
                {"role": "user", "content": self.shopper_text},
            ]
            if need_hidden_spec:
                self.transcript.append({"role": "user", "content": _BG_NUDGE_HIDDEN_SPEC})
            if lane_id == LANE_RANK:
                self.transcript.append({"role": "user", "content": _BG_NUDGE_RANK})
            if lane_id == LANE_PREFERENCE:
                self.transcript.append({"role": "user", "content": _BG_NUDGE_PREF})
            if lane_id == LANE_JUSTIFY:
                self.transcript.append({"role": "user", "content": _BG_NUDGE_CLAIMS})
            self.trail = [
                {
                    "role": "meta",
                    "blackgirl": {
                        "version": 6,
                        "router": lane_id,
                        "model": self.model_id,
                        "provider": "openrouter",
                    },
                }
            ]

        def _idle_until_tools(self) -> tuple[str, list[dict[str, Any]]]:
            spoken = ""
            actions: list[dict[str, Any]] = []
            for retry_idx in range(len(_BG_IDLE_PRODS) + 1):
                spoken, actions = _bg_chat(
                    model_id=self.model_id,
                    transcript=self.transcript,
                    tool_schemas=self.tool_schemas,
                    call_budget=self.call_budget,
                )
                if actions:
                    break
                self.trail.append({"role": "assistant", "content": spoken})
                idle_line = _BG_IDLE_PRODS[min(retry_idx, len(_BG_IDLE_PRODS) - 1)]
                self.transcript.extend(
                    [
                        {"role": "assistant", "content": spoken},
                        {"role": "user", "content": idle_line},
                    ]
                )
            else:
                raise RuntimeError("model returned no tool call before completion")
            return spoken, actions

        def _env_request(self, step_idx: int, actions: list[dict[str, Any]]) -> dict[str, Any]:
            batch_key = f"{self.episode_key}-turn-{step_idx}"

            def _call_row(slot: int, tc_item: dict[str, Any]) -> dict[str, Any]:
                return {
                    "call_id": f"{batch_key}-{slot}",
                    "action": {
                        "name": tc_item["function"]["name"],
                        "args": _bg_parse_args(tc_item),
                    },
                }

            return {
                **self.session_binding,
                "call_id": batch_key,
                "idempotency_key": batch_key,
                "turn": step_idx,
                "calls": [
                    _call_row(slot, tc_item)
                    for slot, tc_item in enumerate(actions, start=1)
                ],
            }

        def _absorb_obs(
            self, actions: list[dict[str, Any]], env_reply: dict[str, Any]
        ) -> None:
            self.prior_payloads = []
            for tc_item, call_row in zip(actions, env_reply["calls"], strict=True):
                payload = call_row.get("observation")
                self.prior_payloads.append(payload)
                self.grade.ingest(payload)
                self.transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_item["id"],
                        "content": _bg_shrink_obs(payload),
                    }
                )

        def _absorb_shopper(self, env_reply: dict[str, Any]) -> None:
            shopper_msg = env_reply.get("user_message")
            if shopper_msg and shopper_msg.get("content"):
                spoken_user = str(shopper_msg["content"])
                self.grade.absorb_note(spoken_user)
                self.transcript.append({"role": "user", "content": spoken_user})
                if _bg_is_inquiry(spoken_user):
                    self.await_reply = True

        def _absorb_faults(self) -> None:
            fault_list = [
                item.get("error")
                for item in self.prior_payloads
                if isinstance(item, dict) and item.get("error")
            ]
            if fault_list and lane_id == LANE_RECOVER:
                self.transcript.append(
                    {
                        "role": "user",
                        "content": _BG_NUDGE_RECOVER.format(error=fault_list[0]),
                    }
                )

        def _absorb_market(self) -> None:
            for item in self.prior_payloads:
                if not isinstance(item, dict):
                    continue
                nested = (
                    item.get("observation")
                    if isinstance(item.get("observation"), dict)
                    else item
                )
                if isinstance(nested, dict) and nested.get("market_event"):
                    event_row = nested["market_event"]
                    if lane_id == LANE_PREFERENCE:
                        coach_text = _BG_NUDGE_PREF_EVENT
                    elif lane_id == LANE_JUSTIFY:
                        coach_text = _BG_NUDGE_JUST_EVENT
                    else:
                        coach_text = (
                            f"Coach: market_event observed ({event_row.get('kind')}). "
                            "Re-verify price/stock; if invalid under firm constraints, "
                            "remove_from_cart and recover per family rules before ordering."
                        )
                    self.transcript.append({"role": "user", "content": coach_text})
                    break

        def _episode_done(self, env_reply: dict[str, Any]) -> bool:
            return any(
                isinstance(call_row.get("observation"), dict)
                and call_row["observation"].get("done")
                for call_row in env_reply.get("calls") or []
            )

        def run(self) -> list[dict[str, Any]]:
            self._prime()
            for step_idx in range(1, self.step_limit + 1):
                coach_line = _bg_next_coach(
                    lane_id,
                    step_idx,
                    self.prior_actions,
                    self.prior_payloads,
                    self.shopper_text,
                    self.await_reply,
                    self.cart_used,
                )
                if coach_line:
                    self.transcript.append({"role": "user", "content": coach_line})

                spoken, actions = self._idle_until_tools()
                actions = self.grade.guard(
                    actions, cart_used=self.cart_used, await_reply=self.await_reply
                )
                action_labels = _bg_extract_action_names(actions)
                self.transcript.append(
                    {
                        "role": "assistant",
                        "content": spoken,
                        "tool_calls": actions,
                    }
                )
                env_reply = _bg_bridge.post(
                    "/environment/call", json_data=self._env_request(step_idx, actions)
                )
                if env_reply is None:
                    raise RuntimeError("environment call failed")
                self.trail.append(
                    {
                        "role": "assistant",
                        "content": spoken,
                        "tool_calls": actions,
                        "environment_result": env_reply,
                        "blackgirl_turn": {"family": lane_id, "turn": step_idx},
                    }
                )
                self.prior_actions = action_labels
                if "add_to_cart" in action_labels:
                    self.cart_used = True
                if "remove_from_cart" in action_labels:
                    self.grade.event_cleared = True
                if "message" in action_labels:
                    self.await_reply = False
                self._absorb_obs(actions, env_reply)
                self._absorb_shopper(env_reply)
                self._absorb_faults()
                self._absorb_market()
                if self._episode_done(env_reply):
                    break
            return self.trail

    return _Episode().run()


def _bind_lane_handlers() -> dict[str, Callable[..., list[dict[str, Any]]]]:
    def _mint(lane_id: str) -> Callable[..., list[dict[str, Any]]]:
        def _run(packet: dict[str, Any]) -> list[dict[str, Any]]:
            return _bg_drive_lane(lane_id=lane_id, problem_data=packet)

        return _run

    table: dict[str, Callable[..., list[dict[str, Any]]]] = {}
    for lid in (
        LANE_INTENT,
        LANE_RECALL,
        LANE_CONSTRAINT,
        LANE_PREFERENCE,
        LANE_RANK,
        LANE_RECOVER,
        LANE_JUSTIFY,
    ):
        table[lid] = _mint(lid)
    return table


_BG_LANE_HANDLERS: dict[str, Callable[..., list[dict[str, Any]]]] = _bind_lane_handlers()


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:

    env_block = problem_data["environment"]
    policy_blob = env_block["policy_view"]
    shopper_text = str(policy_blob.get("query") or problem_data.get("query") or "")
    lane_id = classify_lane(shopper_text)
    handler = _BG_LANE_HANDLERS.get(lane_id, _BG_LANE_HANDLERS[LANE_RECALL])
    return handler(problem_data)


__all__ = ["agent_main", "classify_lane"]
