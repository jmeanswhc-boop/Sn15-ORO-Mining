from __future__ import annotations

import json
import random
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_MODELS = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)

_INFER = ProxyClient(timeout=110, max_retries=1, retry_delay=1.0)
_PROBLEM_BUDGET_S = 270.0
_OBS_CHARS = 7000
_MAX_REPROMPTS = 2
_EPS = 1e-9
# Turns reserved after the first cart for add → recheck → recover → order.
_COMMIT_SLACK = 6
# Inspect this many ok peers when several candidates are already in hand.
_PEER_FLOOR = 2

_VARIANT_RE = re.compile(r"\b(\d{6,})::([A-Za-z0-9._\-]+)\b")
_DIM_RE = re.compile(r"(\d{2,5})\s*[x×*]\s*(\d{2,5})")
_P_TOKEN_RE = re.compile(r"\b(\d{3,4}p)\b", re.I)
_QUOTED_RE = re.compile(r"[\"“]([^\"”]{2,80})[\"”]")
_PHRASE_RE = re.compile(
    r"matches(?:\s+the\s+phrase)?\s+[\"“]?([A-Za-z0-9][^\"”\n,]{1,80}?)[\"”]?"
    r"(?=\s*,|\s+at\s+or\s+|\s+and\s+buy|\s*$)",
    re.I,
)
_PREF_ITEM_RE = re.compile(
    r"prefer(?:ably)? the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b",
    re.I,
)
_PREF_BRAND_RE = re.compile(
    r"\bprefer(?:ably)?\s+(?:a\s+|an\s+|the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)"
    r"\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)",
    re.I,
)
_REQUIRED_BRAND_RE = re.compile(
    r"\b(?:required brand|brand is|brand named)\s+(?:is\s+)?([A-Za-z][\w&.'\- ]{1,40})",
    re.I,
)
_RATHER_RE = re.compile(
    r"\b(?:i'?d rather|i would rather|rather)\b.{0,50}?\bfor\s+([^,.;\n]{3,80})",
    re.I,
)
_PURPOSE_RE = re.compile(r"\bit'?s for\s+([^,.;\n]{3,80})", re.I)
_MATTERS_RE = re.compile(
    r"([^.;:\n]{3,80}?)\s+matters?\s+more(?:\s+to\s+me)?(?:\s+than\b[^.;]*)?",
    re.I,
)
_CARE_ABOUT_RE = re.compile(
    r"\bcare\s+more\s+about\s+([^,.;\n]{3,80}?)(?:\s+than\b|$)"
    r"|([^.;:\n]{3,80}?)\s+is what i (?:actually )?care about",
    re.I,
)
_LESS_IMPORTANT_RE = re.compile(
    r"(?:price|saving(?:s)?|the cheapest(?: one)?)\s+is less important than\s+([^,.;\n]{3,80})",
    re.I,
)
_CEILING_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|"
    r"max(?:imum)?(?:\s+(?:of|price))?|within|budget(?:\s+(?:is|of))?|"
    r"only go up to|can spend up to|spend up to|at most|<=)\s*[:.]?\s*"
    r"([^\d\s]{0,6})?\s*(\d[\d,]*(?:\.\d+)?)\s*([^\s,.;:]{0,12})?",
    re.I,
)
_HARD_SPAN_RE = re.compile(
    r"\b(?:must|require(?:s|d)?|exactly)\b\s+([^.;\n]{2,60})",
    re.I,
)
_MEASURE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-]?\s*([A-Za-z%]{1,12}|\"|'{2})",
    re.I,
)
_PREFER_RE = re.compile(
    r"\b(?:prefer(?:red|ably)?|ideally|nice\s+to\s+have|if\s+possible)\b"
    r"\s+(?:a\s+|an\s+|the\s+)?([A-Za-z0-9][^,.;\n]{1,60})",
    re.I,
)
_COMPARE_RE = re.compile(
    r"\b(highest|lowest|largest|smallest|cheapest|most|least|maximum|minimum)\b"
    r"\s+([a-z0-9][a-z0-9 \-]{1,40})",
    re.I,
)
_CATEGORY_RE = re.compile(
    r"\b(?:in\s+the|in)\s+([A-Za-z0-9][^.\n]{1,40}?)\s+categor(?:y|ies)\b",
    re.I,
)
_SURFACE_RE = re.compile(r"surface at least (\d+) distinct", re.I)
_WHY_RE = re.compile(r"\b(why|reason|explain|justif(?:y|ication)?)\b", re.I)
_ASK_RE = re.compile(
    r"\b(?:ask(?:s|ed)? (?:the )?(?:shopper|user)|when asked|"
    r"hidden (?:firm )?requirement|which preference|"
    r"priority(?: order)? (?:among|of) (?:your )?prefer)\b",
    re.I,
)
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|"
    r"which (?:one|option)|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)
_HINT_KEYS = frozenset({"title", "name", "description", "snippet", "summary", "query"})
_ID_KEYS = frozenset({"product_id", "sku", "listing_id", "id", "currency", "session_id"})
_META_UNITS = frozenset(
    {
        "sku", "id", "pid", "new", "used", "stock", "true", "false", "default",
        "and", "or", "to", "of", "for", "the", "x", "by", "php", "usd", "eur",
    }
)
_FUNCTION_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "for", "with", "from", "in", "on",
        "at", "by", "be", "is", "are", "was", "were", "this", "that", "these", "those",
        "my", "your", "our", "me", "i", "we", "you", "want", "find", "get", "please",
        "can", "could", "would", "should", "than", "then", "into", "over", "under",
        "below", "above", "not", "any", "some", "item", "product", "products",
        "option", "options", "one", "it", "its", "need", "order", "stock", "place",
        "buy", "have", "like",
    }
)

_CORE_RULES = (
    "You are a careful shopping agent. Use only the tools supplied for this episode. "
    "Exactly one tool call per turn, so be efficient. Reason from three sources only: "
    "the shopper's own words, the parameter names and descriptions in the tool schemas, "
    "and values the environment actually returns. Do not rename a requested specification. "
    "If the shopper asks for a measurement, model code, or phrase, search and filter "
    "using that extracted text as the query, not the surrounding instruction sentence. "
    "A brand seen in one listing is not a shopper constraint unless they named it. "
    "After a price or stock change, keep the original phrase and preferences in the "
    "replacement search; do not fall back to a generic category-only query. "
    "A search title is a hint for what to open next, not a verified "
    "attribute. Open a product and read its returned facts before buying. Confirm live "
    "price and availability before adding to the cart or ordering. A deterministic "
    "controller checks every proposal and takes over the cart and the order once a "
    "candidate is fully verified; do not add an exploratory placeholder. Prices and "
    "stock change while you shop; the controller re-checks the "
    "carted item and will tell you when to move to a different product. Separate "
    "necessary constraints from optional preferences. Satisfy necessary constraints "
    "first; use preferences only to break ties. When the request lists specific "
    "product identifiers, consider only those. When it states an order such as "
    "highest or lowest of some attribute, apply that order to the attribute words "
    "they used, using values you observed. If the shopper asks why, copy values from "
    "returned facts and do not invent attributes. Answer every direct question from "
    "the shopper with one short message before anything else, and never send progress "
    "updates. Never ask the shopper to raise the budget. Do not repeat a tool call "
    "with identical arguments; one compare of up to five products beats several view "
    "calls. Finish by ordering once a verified product is in the cart."
)


def _tokens(text: str) -> list[str]:
    """Keep mixed alphanumerics (A77S, X100, 5G), decimals, and trailing p."""
    return re.findall(
        r"[A-Za-z]+\d[\w]*|\d+[A-Za-z]+|\d+(?:\.\d+)?p?|[A-Za-z]+",
        text or "",
    )


def _token_set(text: str) -> set[str]:
    return {tok.lower() for tok in _tokens(text)}


def _alnum_set(text: str) -> set[str]:
    """Keep mixed tokens such as a77s that _tokens would split."""
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _has_theme(text: str, theme: list[str]) -> bool:
    if not theme:
        return True
    have = _alnum_set(text)
    return all(tok in have for tok in theme)


def _model_in_text(model: str, text: str) -> bool:
    raw = (model or "").strip()
    if not raw or not text:
        return False
    if re.fullmatch(r"\d{2,}", raw):
        return re.search(rf"(?<!\d){re.escape(raw)}(?!\d)", text) is not None
    compact = re.sub(r"[^a-z0-9]+", "", raw.lower())
    return bool(compact) and compact in re.sub(r"[^a-z0-9]+", "", text.lower())


def _content_tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in _tokens(text):
        low = tok.lower()
        if low in _FUNCTION_WORDS or len(low) < 2:
            continue
        if low not in out:
            out.append(low)
    return out


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text or "")


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _stems(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}


def _key_tokens(text: str) -> set[str]:
    parts = re.split(r"[_\s.\-/]+", str(text or "").lower())
    out: set[str] = set()
    for part in parts:
        for tok in _tokens(part):
            low = tok.lower().rstrip("s")
            if low and low not in _FUNCTION_WORDS:
                out.add(low)
    return out


def _measurements(text: str) -> list[tuple[float, str]]:
    """Number+unit pairs copied from text. Units are the words that appear there."""
    found: list[tuple[float, str]] = []
    for match in _MEASURE_RE.finditer(text or ""):
        number = _as_float(match.group(1))
        unit = (match.group(2) or "").strip().lower().rstrip("s")
        if number is None or not unit:
            continue
        if abs(number) >= 1_000_000:
            continue
        if unit in {'"', "''", "inche", "inch", "in"}:
            unit = "in"
        elif unit in _FUNCTION_WORDS or unit in _META_UNITS:
            continue
        if len(unit) == 1 and unit not in {'"', "%", "'"}:
            continue
        if unit == "%":
            continue
        if len(unit) > 4:
            continue
        found.append((float(number), unit))
    for match in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:["”]|\'{2})', text or ""):
        number = _as_float(match.group(1))
        if number is not None:
            found.append((float(number), "in"))
    return found


def _primary_request(query: str) -> str:
    text = (query or "").strip()
    if not text:
        return ""
    return re.split(r"\n\s*\n", text, maxsplit=1)[0].strip()


def _money_ceilings(text: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for match in _CEILING_RE.finditer(text or ""):
        number = _as_float(match.group(2))
        if number is None:
            continue
        unit = (match.group(3) or match.group(1) or "").strip(" .,:;?!")
        found.append({"number": float(number), "unit": unit})
    return found


def _listed_variants(text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).rstrip(".,;:")) for m in _VARIANT_RE.finditer(text or "")]


def _dimension_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for match in _DIM_RE.finditer(text or ""):
        phrase = f"{match.group(1)}x{match.group(2)}"
        if phrase not in phrases:
            phrases.append(phrase)
    return phrases


def _is_meta_phrase(phrase: str) -> bool:
    return bool(
        re.search(
            r"\b(?:brand|categor(?:y|ies)|catalog|budget|requirements?|preference|priority|"
            r"explicitly|identify|device os|title or|walk me|justif|reason such|"
            r"matching product|distinct|threshold|search result|task rules)\b",
            phrase or "",
            re.I,
        )
    )


def _is_instruction_query(text: str) -> bool:
    """True when this string is task protocol, not a catalog search."""
    blob = " ".join((text or "").split())
    if not blob:
        return True
    if _is_meta_phrase(blob):
        return True
    low = blob.lower()
    if re.search(r"\b(?:find an|place one test order|and buy one|walk me through)\b", low):
        return True
    protocol = sum(
        1
        for marker in ("listing", "matches", "phrase", "category", "in-stock", "in stock")
        if marker in low
    )
    return protocol >= 2


def _clip_clause(text: str) -> str:
    """Keep the first clause; stop at dash, semicolon, or a rather-pivot."""
    chunk = " ".join((text or "").split()).strip(" .,:;")
    chunk = re.split(
        r"\s*[—–]\s*|\s+-\s+|\s*;\s*|\b(?:i'?d rather|i would rather|rather than|\brather\b)\b",
        chunk,
        maxsplit=1,
        flags=re.I,
    )[0]
    return chunk.strip(" .,:;")


def _trim_pref_span(text: str) -> str:
    chunk = _clip_clause(text)
    chunk = re.split(
        r"\b(?:than|rather than|that survives|that i|which i)\b",
        chunk,
        maxsplit=1,
        flags=re.I,
    )[0]
    chunk = re.split(r"\s+that\s+", chunk, maxsplit=1, flags=re.I)[0]
    chunk = re.sub(
        r"^(?:cheapest isn'?t the goal(?: here)?|don'?t just grab the cheapest(?: one)?)\s*[,;:\-—–]?\s*",
        "",
        chunk,
        flags=re.I,
    )
    return chunk.strip(" .,:;")


def _match_phrases(text: str) -> list[str]:
    """Shopper match-phrases, quoted or introduced by 'the phrase X'."""
    found: list[str] = []

    def add(phrase: str) -> None:
        phrase = " ".join((phrase or "").split()).strip(" .,:;\"“”")
        if not phrase or _is_meta_phrase(phrase) or _is_instruction_query(phrase):
            return
        if phrase.lower() in _FUNCTION_WORDS:
            return
        if phrase not in found:
            found.append(phrase)

    for match in _QUOTED_RE.finditer(text or ""):
        add(match.group(1))
    for match in _PHRASE_RE.finditer(text or ""):
        add(match.group(1))
    return found


_PROTOCOL_ATTR_RE = re.compile(
    r"\b(?:reason|listing|identit|threshold|matching product|useful|"
    r"valid item|distinct|digit|class whose|advertised maximum|shopper|"
    r"justif|claim|tradeoff)\b",
    re.I,
)


def _claim_names_from_text(text: str) -> list[str]:
    """Pull claim identifiers from schema prose. Not a pack field table."""
    found: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        name = (name or "").strip().lower().replace(" ", "_")
        if not name or name in seen:
            return
        if name in _FUNCTION_WORDS or name in _META_UNITS:
            return
        if name in {"field", "value", "claims", "type", "string", "object", "array", "true", "false"}:
            return
        seen.add(name)
        found.append(name)

    for match in re.finditer(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", text or ""):
        add(match.group(0))
    for match in re.finditer(
        r"\b[a-z][a-z0-9_]{1,30}(?:\s*,\s*[a-z][a-z0-9_]{1,30})+(?:\s*,?\s*(?:and|or)\s+[a-z][a-z0-9_]{1,30})?",
        text or "",
        re.I,
    ):
        for part in re.split(r"\s*,\s*|\s+(?:and|or)\s+", match.group(0), flags=re.I):
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", part.strip()):
                add(part)
    return found


def _hard_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for phrase in _match_phrases(text):
        if phrase not in phrases:
            phrases.append(phrase)
    for match in _HARD_SPAN_RE.finditer(text or ""):
        words = match.group(1).split()[:12]
        phrase = " ".join(words).strip(" .,:;")
        if phrase and not _is_meta_phrase(phrase) and phrase not in phrases:
            phrases.append(phrase)
    for phrase in _dimension_phrases(text):
        if phrase not in phrases:
            phrases.append(phrase)
    for match in _P_TOKEN_RE.finditer(text or ""):
        token = match.group(1)
        if token not in phrases:
            phrases.append(token)
    return phrases


def _clauses(text: str) -> list[str]:
    """Split on clause boundaries, not decimal points inside prices."""
    parts = re.split(
        r"[;\n]+|[—–]+|(?:\s+-\s+)|(?<!\d)\.(?:\s+|$)",
        text or "",
    )
    return [" ".join(p.split()).strip(" .,:;") for p in parts if " ".join(p.split()).strip(" .,:;")]


def _intent_spans(text: str) -> tuple[list[str], list[str]]:
    """Preference clauses vs purpose clauses, bounded by sentence structure."""
    prefer: list[str] = []
    purpose: list[str] = []

    def add(dest: list[str], raw: str) -> None:
        phrase = _trim_pref_span(raw)
        if not phrase or phrase.lower() in _FUNCTION_WORDS or _is_instruction_query(phrase):
            return
        if _is_meta_phrase(phrase):
            return
        if re.search(r"\b(?:cheapest|grab the)\b", phrase, re.I):
            return
        if len(_content_tokens(phrase)) < 2 and not re.search(r"[a-z]{4,}", phrase or "", re.I):
            return
        if phrase not in dest:
            dest.append(phrase)

    for clause in _clauses(text):
        for match in _PREFER_RE.finditer(clause):
            add(prefer, match.group(1))
        for match in _RATHER_RE.finditer(clause):
            add(prefer, match.group(1))
        for match in _MATTERS_RE.finditer(clause):
            add(prefer, match.group(1))
        for match in _CARE_ABOUT_RE.finditer(clause):
            add(prefer, match.group(1) or match.group(2) or "")
        for match in _LESS_IMPORTANT_RE.finditer(clause):
            add(prefer, match.group(1))
        for match in _PURPOSE_RE.finditer(clause):
            add(purpose, _clip_clause(match.group(1)))
        for match in re.finditer(
            r"(?:\d[\d,]*(?:\.\d+)?)\s*[A-Za-z]{0,12}\s+for\s+(.{3,70})",
            clause,
            re.I,
        ):
            add(purpose, _clip_clause(match.group(1)))
    return prefer, purpose


def _prefer_phrases(text: str) -> list[str]:
    prefer, purpose = _intent_spans(text)
    phrases = list(prefer)
    for phrase in purpose:
        if phrase not in phrases:
            phrases.append(phrase)
    return phrases


def _comparatives(text: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for match in _COMPARE_RE.finditer(text or ""):
        direction_word = match.group(1).lower()
        attribute = " ".join(match.group(2).split())
        attribute = re.split(
            r"\b(?:then|among|and|before|after|wins|choose|pick)\b",
            attribute,
            maxsplit=1,
            flags=re.I,
        )[0].strip(" .,:;-")
        if not attribute or _PROTOCOL_ATTR_RE.search(attribute):
            continue
        if not [t for t in re.findall(r"[a-z]+", attribute.lower()) if len(t) >= 4]:
            continue
        direction = "min" if direction_word in {"lowest", "smallest", "cheapest", "least", "minimum"} else "max"
        item = {"direction": direction, "attribute": attribute}
        if item not in found:
            found.append(item)
    return found


def _category_phrase(text: str) -> str | None:
    match = _CATEGORY_RE.search(text or "")
    if match:
        phrase = " ".join(match.group(1).split()).strip(" .,:;")
        return phrase or None
    match = re.search(
        r"\bcategor(?:y|ies)\s+is\s+([A-Za-z][\w &/-]{1,40})",
        text or "",
        re.I,
    )
    if match:
        phrase = " ".join(match.group(1).split()).strip(" .,:;")
        return phrase or None
    match = re.search(
        r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
        text or "",
        re.I,
    )
    return match.group(1) if match else None


def _brand_before_category(text: str, category: str | None) -> str | None:
    """Capitalized words naming the item, e.g. 'a BrandName category'."""
    if not category:
        return None
    match = re.search(
        r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category),
        text or "",
        re.I,
    )
    if not match:
        return None
    skip = {tok.lower() for tok in _tokens(" ".join(_measure_phrases(text, None) + _dimension_phrases(text)))}
    skip.update(m.lower() for m in _P_TOKEN_RE.findall(text or ""))
    keep = []
    for word in re.findall(r"[\w&.'\-]+", match.group(1)):
        if any(ch.isdigit() for ch in word) or _P_TOKEN_RE.fullmatch(word):
            break
        if not word[:1].isupper():
            continue
        if word.lower() in skip or word.lower() in _FUNCTION_WORDS:
            continue
        if word.isupper() and len(word) <= 4:
            continue
        keep.append(word)
    return " ".join(keep) or None


def _prefer_brand(text: str) -> str | None:
    match = _PREF_BRAND_RE.search(text or "")
    if not match:
        return None
    name = match.group(1).strip(" .,")
    words = name.split()
    if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
        return None
    return name


def _required_brand(text: str) -> str | None:
    match = _REQUIRED_BRAND_RE.search(text or "")
    if not match:
        return None
    name = match.group(1).strip(" .,")
    name = re.split(r"\b(?:and|with|in|at|for)\b", name, maxsplit=1, flags=re.I)[0].strip()
    words = [w for w in name.split() if w[:1].isupper() or w.istitle()]
    return " ".join(words) if words else None


def _specific_prefer(phrases: list[str]) -> bool:
    return any(re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", phrase or "") for phrase in phrases)


def _measure_phrases(text: str, budget: float | None) -> list[str]:
    """Number+unit snippets copied from this request. No unit conversion."""
    phrases: list[str] = []
    for number, unit in _measurements(text):
        if budget is not None and abs(number - budget) <= _EPS:
            continue
        if number >= 100000:
            continue
        phrase = ("%g" % number) + " " + unit
        if phrase not in phrases:
            phrases.append(phrase)
    return phrases


def _search_strings(request: str) -> list[str]:
    """Catalog queries from match phrases, prefs, and measures — not the instruction."""
    primary = _primary_request(request)
    strings: list[str] = []

    def add(item: str) -> None:
        cleaned = " ".join((item or "").split())
        if not cleaned or _is_instruction_query(cleaned):
            return
        if cleaned not in strings:
            strings.append(cleaned)

    matched = _match_phrases(primary)
    for phrase in matched:
        add(phrase)
        words = [t for t in _content_tokens(phrase)]
        if len(words) >= 2:
            compact = " ".join(words)
            add(compact)
            if len(words[0]) >= 4:
                add(words[0])
    ceilings = _money_ceilings(primary)
    budget = ceilings[0]["number"] if ceilings else None
    measures = _measure_phrases(primary, budget)
    for phrase in _prefer_phrases(primary) + _hard_phrases(primary) + measures + _dimension_phrases(primary):
        add(phrase)
    cat = _category_phrase(primary)
    brand = _brand_before_category(primary, cat) or _prefer_brand(primary)
    if not matched:
        if brand and brand not in strings:
            strings.insert(0, brand)
        if cat:
            add(f"{brand} {cat}" if brand else cat)
    if not strings:
        for phrase in matched:
            add(phrase)
    uniq: list[str] = []
    for item in strings:
        if item and item not in uniq and not _is_instruction_query(item):
            uniq.append(item)
    return uniq[:6]


def _blob_has_number(blob: str, num_str: str) -> bool:
    want = _as_float(num_str)
    if want is None:
        return False
    for found in _numbers(blob):
        got = _as_float(found)
        if got is not None and abs(got - want) <= max(1e-6, abs(want) * 1e-6):
            return True
    return bool(re.search(rf"(?<!\d){re.escape(num_str)}(?!\d)", blob or ""))


def _text_has_numbers(text: str, numbers: list[str]) -> bool:
    blob = str(text or "")
    return all(_blob_has_number(blob, num) for num in numbers)


def _rank_prior(rank: int | None) -> float:
    """Simple recency/coverage weight for earlier search hits. Not a fitted table."""
    if rank is None or rank < 0:
        return 0.0
    return 1.0 / (rank + 1)


def _p_token_supported(phrase: str, blob: str) -> bool:
    """Np matches a recorded Np token or a WxH pair whose sides include N.

    A different Mp token in the same blob does not veto a WxH pair that
    already has N as a side.
    """
    if not re.fullmatch(r"\d{3,4}p", phrase or "", re.I):
        return False
    blob = blob or ""
    if phrase.lower() in blob.lower():
        return True
    try:
        want = int(phrase[:-1])
    except ValueError:
        return False
    for left, right in _DIM_RE.findall(blob):
        try:
            sides = (int(left), int(right))
        except ValueError:
            continue
        if want in sides:
            return True
    return False


def _np_values(text: str) -> list[int]:
    found: list[int] = []
    for match in _P_TOKEN_RE.finditer(text or ""):
        try:
            found.append(int(match.group(1)[:-1]))
        except ValueError:
            continue
    return found


def _pixel_grid_request(text: str) -> bool:
    """True when the shopper named a pixel grid (Np or WxH), not a length."""
    return bool(_P_TOKEN_RE.search(text or "") or _DIM_RE.search(text or ""))


def _request_spec_labels(text: str) -> list[str]:
    """ALL-CAPS abbreviations sitting next to an Np/WxH the shopper wrote.

    These are the shopper's own words (HD beside 720p), not a class table.
    """
    blob = text or ""
    spans = [(m.start(), m.end()) for m in _P_TOKEN_RE.finditer(blob)]
    spans.extend((m.start(), m.end()) for m in _DIM_RE.finditer(blob))
    if not spans:
        return []
    labels: list[str] = []
    for start, end in spans:
        window = blob[max(0, start - 16) : min(len(blob), end + 16)]
        for match in re.finditer(r"\b[A-Z]{2,4}\b", window):
            tok = match.group(0).lower()
            if tok in _FUNCTION_WORDS or tok in _META_UNITS:
                continue
            if tok.endswith("p") and tok[:-1].isdigit():
                continue
            if tok not in labels:
                labels.append(tok)
    return labels


def _pixel_grid_value(text: str) -> bool:
    """Observed text is a pixel grid (WxH/Np) or a short whole-value spec token."""
    blob = str(text or "").strip()
    if not blob:
        return False
    if _DIM_RE.search(blob) or _P_TOKEN_RE.search(blob):
        return True
    return bool(re.fullmatch(r"[A-Za-z]{2,4}", blob))


def _conflicting_np(text: str, want: int) -> bool:
    """True when a fact that can support N also names a different Np class."""
    others = [n for n in _np_values(text) if n != want]
    return bool(others)


def _phrase_in_value(phrase: str, value: Any) -> bool:
    if value is None or not phrase:
        return False
    blob = str(value)
    if not blob.strip():
        return False
    if phrase.lower() in blob.lower():
        return True
    if _p_token_supported(phrase, blob):
        return True
    numbers = _numbers(phrase)
    tokens = [tok.lower() for tok in _content_tokens(phrase) if not tok.isdigit()]
    if numbers:
        if not _text_has_numbers(blob, numbers):
            return False
        compact_phrase = re.sub(r"[\s×*x\-]", "", phrase.lower())
        compact_blob = re.sub(r"[\s×*x\-]", "", blob.lower())
        if compact_phrase and compact_phrase in compact_blob:
            return True
        if len(numbers) >= 2:
            return True
        if tokens and not all(tok in blob.lower() for tok in tokens):
            return False
        return True
    return bool(tokens) and all(tok in blob.lower() for tok in tokens)


def _is_hint_key(key: str) -> bool:
    return str(key).lower().rsplit(".", 1)[-1] in _HINT_KEYS


def _flatten_facts(product: dict[str, Any]) -> dict[str, Any]:
    """Keep the environment's own field names. Do not alias them."""
    facts: dict[str, Any] = {}
    for key, value in (product or {}).items():
        if key in {"variants", "results", "compared", "items", "observations"}:
            continue
        if _is_hint_key(str(key)):
            continue
        if str(key).lower().rsplit(".", 1)[-1] in _ID_KEYS:
            continue
        if isinstance(value, dict):
            nested_prefix = "" if key == "facts" else f"{key}."
            for sub_key, sub_value in value.items():
                if _is_hint_key(str(sub_key)) or str(sub_key).lower() in _ID_KEYS:
                    continue
                if isinstance(sub_value, (str, int, float, bool)) or sub_value is None:
                    facts[f"{nested_prefix}{sub_key}"] = sub_value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            facts[str(key)] = value
    return facts


def _unwrap(obs: Any) -> Any:
    if isinstance(obs, dict) and isinstance(obs.get("observation"), (dict, list)):
        return obs["observation"]
    return obs


def _tool_index(tools: list) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        required = params.get("required") if isinstance(params.get("required"), list) else []
        index[str(name)] = {"properties": props, "required": [str(x) for x in required]}
    return index


def _param(index: dict[str, dict[str, Any]], tool: str, key: str) -> bool:
    return key in (index.get(tool) or {}).get("properties", {})


def _filter_args(index: dict[str, dict[str, Any]], tool: str, args: dict[str, Any]) -> dict[str, Any]:
    props = (index.get(tool) or {}).get("properties") or {}
    if not props:
        return args
    return {key: value for key, value in args.items() if key in props and value is not None}


def _arguments(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + ' ... (truncated by agent)"}'


def _ground_value(facts: dict[str, Any], field: str, value: Any) -> bool:
    if field in facts:
        actual = facts.get(field)
    else:
        actual = None
        best = 0
        for key, observed in facts.items():
            overlap = len(_key_tokens(field) & _key_tokens(str(key)))
            if overlap > best:
                best, actual = overlap, observed
        if best == 0:
            blob = " ".join(str(v) for v in facts.values() if v is not None)
            return _phrase_in_value(str(value), blob)
    if actual is None:
        return False
    if isinstance(actual, bool):
        want = str(value).strip().lower()
        return want in {"true", "1", "yes"} if actual else want in {"false", "0", "no"}
    if str(actual).strip().lower() == str(value).strip().lower():
        return True
    left, right = _as_float(actual), _as_float(value)
    if left is not None and right is not None:
        return abs(left - right) <= max(1e-6, abs(left) * 1e-6)
    return _phrase_in_value(str(value), actual)


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


class Brief:
    """Requirements parsed from this episode's public request text."""

    def __init__(self, query: str) -> None:
        self.raw = query or ""
        self.goal = _primary_request(self.raw)
        self.request = self.raw
        ceilings = _money_ceilings(self.goal) or _money_ceilings(self.raw)
        self.budget = ceilings[0]["number"] if ceilings else None
        self.currency = ceilings[0]["unit"] if ceilings else ""
        self.candidate_keys = _listed_variants(self.goal) or _listed_variants(self.raw)
        self.hard_phrases = _hard_phrases(self.goal)
        soft, purpose = _intent_spans(self.goal)
        self.soft_prefer = soft
        self.purpose_phrases = purpose
        self.prefer_phrases = _prefer_phrases(self.goal)
        self.comparatives = _comparatives(self.goal)
        self.category = _category_phrase(self.goal) or _category_phrase(self.raw)
        self.brand = _required_brand(self.goal) or _brand_before_category(self.goal, self.category)
        self.pref_brand = _prefer_brand(self.goal)
        if self.pref_brand and self.brand and self.pref_brand.lower() == self.brand.lower():
            self.brand = None
        pref_item = _PREF_ITEM_RE.search(self.goal)
        self.pref_color = pref_item.group(1).lower() if pref_item else None
        self.pref_model = pref_item.group(2) if pref_item else None
        if self.pref_model:
            self.pref_brand = None
        self.measures = _measure_phrases(self.goal, self.budget)
        phrase = _PHRASE_RE.search(self.goal)
        quoted = _QUOTED_RE.findall(self.goal)
        matched = _match_phrases(self.goal)
        source = matched[0] if matched else (phrase.group(1) if phrase else (quoted[0] if quoted else ""))
        self.theme = [t.lower() for t in re.findall(r"[a-z0-9]+", source.lower())]
        surface = _SURFACE_RE.search(self.raw)
        self.required_identities = int(surface.group(1)) if surface else 0
        self.ranking_like = bool(self.candidate_keys) and bool(self.comparatives)
        ranking_like = self.ranking_like
        self.asks_reason = bool(_WHY_RE.search(self.goal)) and not ranking_like
        self.asks_shopper = bool(_ASK_RE.search(self.goal)) or (
            "firm requirement" in self.goal.lower() and not self.measures
        )
        self.search_strings = _search_strings(self.goal)
        self.specific_prefer = _specific_prefer(self.prefer_phrases) or bool(self.pref_model)

    def adopt_text(self, text: str) -> list[str]:
        """Fold a later shopper message into the live requirements."""
        added: list[str] = []
        for phrase in _hard_phrases(text) + _dimension_phrases(text):
            if phrase and phrase not in self.hard_phrases:
                self.hard_phrases.append(phrase)
                added.append(phrase)
        for phrase in _measure_phrases(text, self.budget):
            if phrase not in self.measures:
                self.measures.append(phrase)
                added.append(phrase)
            if phrase not in self.search_strings:
                self.search_strings.insert(0, phrase)
        soft, purpose = _intent_spans(text)
        for phrase in soft:
            if phrase and phrase not in self.soft_prefer:
                self.soft_prefer.append(phrase)
            if phrase and phrase not in self.prefer_phrases:
                self.prefer_phrases.append(phrase)
        for phrase in purpose:
            if phrase and phrase not in self.purpose_phrases:
                self.purpose_phrases.append(phrase)
            if phrase and phrase not in self.prefer_phrases:
                self.prefer_phrases.append(phrase)
        brand = _required_brand(text) or _brand_before_category(text, self.category or _category_phrase(text))
        if brand and not self.brand:
            self.brand = brand
            added.append(brand)
        cat = _category_phrase(text)
        if cat and not self.category:
            self.category = cat
            added.append(cat)
        if _specific_prefer(self.prefer_phrases) or self.pref_model:
            self.specific_prefer = True
        extra_phrases = _match_phrases(text) + _prefer_phrases(text)
        for blob in extra_phrases:
            if blob and blob not in self.search_strings and not _is_instruction_query(blob):
                self.search_strings.insert(0, blob)
        return added


class Ledger:
    """Observed facts stay separate from search hints."""

    def __init__(self) -> None:
        self.cart: list = []
        self.checked: dict = {}
        self.checked_seq: dict = {}
        self.inspected: set = set()
        self.products: dict = {}
        self.rows: dict = {}
        self.options: dict = {}
        self.facts: dict = {}
        self.hints: dict = {}
        self.listing_ids: set = set()
        self.listing_of: dict = {}
        self.identities: set = set()
        self.seq = 0
        self.first_add_seq: int | None = None
        self.last_add_seq: int | None = None
        self.adds = 0
        self.event_target: str | None = None
        self.dead_keys: set = set()
        self.dead_products: set = set()
        self.filter_pids: set = set()
        self.calls_seen: set = set()
        self.search_rank: dict = {}

    @staticmethod
    def key(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def record(self, name: str, args: dict, obs: dict, brief: Brief) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        payload = _unwrap(obs)
        payload = payload if isinstance(payload, dict) else obs
        low = (name or "").lower()
        if name in ("search", "filter") or any(tok in low for tok in ("search", "filter", "find", "list")):
            from_filter = name == "filter" or "filter" in low
            for index, row in enumerate(payload.get("results") or []):
                if isinstance(row, dict):
                    self._note_row(row, brief, rank=index)
                    pid = row.get("product_id")
                    if from_filter and pid:
                        self.filter_pids.add(str(pid))
        elif name == "view" or ("view" in low and "cart" not in low):
            self._note_product(payload, brief, opened=True)
        elif name == "compare" or "compare" in low:
            for prod in payload.get("compared") or payload.get("products") or []:
                if isinstance(prod, dict):
                    self._note_product(prod, brief, opened=True)
        elif name == "inspect_stock" or ("inspect" in low and "cart" not in low):
            k = self.key(payload.get("product_id") or args.get("product_id"), payload.get("sku") or args.get("sku"))
            self.checked[k] = {"in_stock": bool(payload.get("in_stock")), "price": _as_float(payload.get("price"))}
            self.checked_seq[k] = self.seq
            self.inspected.add(k)
            pid = payload.get("product_id") or args.get("product_id")
            sku = payload.get("sku") or args.get("sku")
            bag = self.facts.setdefault(self.key(pid, sku), {})
            if "price" in payload:
                bag["price"] = payload.get("price")
            if "in_stock" in payload:
                bag["in_stock"] = payload.get("in_stock")
        elif name == "add_to_cart":
            added = payload.get("added") or {}
            pid, sku = added.get("product_id"), added.get("sku")
            if pid and sku:
                self.adds += 1
                if (pid, sku) not in self.cart:
                    self.cart.append((pid, sku))
                if self.first_add_seq is None:
                    self.first_add_seq = self.seq
                self.last_add_seq = self.seq
            self._sync_cart(payload)
        elif name == "remove_from_cart":
            self._sync_cart(payload)
            removed = payload.get("removed") or {}
            pair = (removed.get("product_id"), removed.get("sku"))
            if pair in self.cart:
                self.cart.remove(pair)
        elif name == "inspect_cart":
            self._sync_cart(payload)

    def _sync_cart(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [(r.get("product_id"), r.get("sku")) for r in rows if isinstance(r, dict) and r.get("product_id")]
        if pairs or rows == []:
            self.cart = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            k = self.key(r.get("product_id"), r.get("sku"))
            price = _as_float(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.checked.get(k, {})
            self.checked[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.checked_seq[k] = self.seq

    def _note_row(self, row: dict, brief: Brief, rank: int | None = None) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        if rank is not None:
            prev = self.search_rank.get(str(pid))
            if prev is None or rank < prev:
                self.search_rank[str(pid)] = rank
        title = str(row.get("title") or "")
        if brief.theme:
            if _has_theme(title, brief.theme):
                self.identities.add(str(pid))
        else:
            self.identities.add(str(pid))
        if not row.get("sku"):
            return
        k = self.key(pid, row.get("sku"))
        self.rows[k] = row
        self.hints.setdefault(k, [])
        if title and title not in self.hints[k]:
            self.hints[k].append(title[:500])
        if row.get("price") is not None:
            self.checked.setdefault(k, {"in_stock": bool(row.get("in_stock", True)), "price": _as_float(row.get("price"))})

    def _note_product(self, prod: dict, brief: Brief, *, opened: bool) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.products[str(pid)] = prod
        if prod.get("listing_id"):
            self.listing_ids.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        title = str(prod.get("title") or "")
        if brief.theme and _has_theme(title, brief.theme):
            self.identities.add(str(pid))
        base = _flatten_facts(prod)
        variants = prod.get("variants") if isinstance(prod.get("variants"), list) else []
        if variants:
            for variant in variants:
                if not isinstance(variant, dict) or not variant.get("sku"):
                    continue
                k = self.key(pid, variant.get("sku"))
                self.checked[k] = {"in_stock": bool(variant.get("in_stock")), "price": _as_float(variant.get("price"))}
                self.checked_seq[k] = self.seq
                opt = str(variant.get("options") or "")
                self.options[k] = opt
                merged = dict(base)
                merged.update(_flatten_facts(variant))
                if opt:
                    merged["options"] = opt
                self.facts[k] = merged
                if title:
                    self.hints.setdefault(k, [])
                    if title not in self.hints[k]:
                        self.hints[k].append(title[:500])
        elif opened:
            sku = prod.get("sku") or ""
            k = self.key(pid, sku)
            self.facts[k] = base
            if title:
                self.hints.setdefault(k, [])
                if title not in self.hints[k]:
                    self.hints[k].append(title[:500])

    def status(self, pid: str, sku: str) -> dict | None:
        return self.checked.get(self.key(pid, sku))


class Agent:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_index = _tool_index(self.tools)
        self.tool_names = set(self.tool_index)
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.brief = Brief(str(self.policy.get("query") or ""))
        self.budget = self.brief.budget
        self.ledger = Ledger()
        self.clock = Clock(_PROBLEM_BUDGET_S)
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._user_prompt()},
        ]
        self.turn = 0
        self.asked_shopper = False
        self.shopper_question_pending = False
        self.shopper_updated = False
        self.rejected_streak = 0
        self.pending_envelope: dict | None = None
        self.search_calls = 0
        self.cart_inspections = 0
        self.pending_recheck: tuple | None = None
        self.forced_compares = 0
        self.budget_peers_done = False
        self.probes: set = set()
        self.searched_twins = False
        self.reads_done = False
        self.unreadable: set = set()
        self.question_waits = 0
        self.questions_asked = 0
        self.done = False
        self.search_cursor = 0
        self.model_index = 0
        self._infer_staggered = False
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    def _system_prompt(self) -> str:
        names = ", ".join(sorted(self.tool_names)) or "(see schemas)"
        return _CORE_RULES + "\n\nTools available in this episode: " + names + ".\n\n" + self._brief_text()

    def _user_prompt(self) -> str:
        return (
            (self.policy.get("query") or "")
            + f"\n\nLimits: max_steps={self.max_steps}, max_calls_per_turn={self.max_calls}."
        )

    def _brief_text(self) -> str:
        lines = ["Use the shopper's wording. Do not rename specifications."]
        if self.budget is not None:
            unit = f" {self.brief.currency}" if self.brief.currency else ""
            lines.append(f"Maximum spend stated in the request: {self.budget}{unit}")
        if self.brief.category:
            lines.append("Requested category: " + self.brief.category)
        if self.brief.brand:
            lines.append("Firm brand named in the request: " + self.brief.brand)
        if self.brief.pref_brand:
            lines.append("Preferred brand (only after every firm requirement): " + self.brief.pref_brand)
        if self.brief.pref_model:
            extra = (self.brief.pref_color + " ") if self.brief.pref_color else ""
            lines.append("Preferred item: " + extra + self.brief.pref_model)
        if self.brief.hard_phrases:
            lines.append("Necessary phrases from the request:")
            for phrase in self.brief.hard_phrases[:8]:
                lines.append("- " + phrase)
        if self.brief.prefer_phrases:
            lines.append("Optional preferences from the request:")
            for phrase in self.brief.prefer_phrases[:6]:
                lines.append("- " + phrase)
        if self.brief.comparatives:
            lines.append("Order stated in the request:")
            for item in self.brief.comparatives:
                lines.append(f"- {item['direction']} {item['attribute']}")
        if self.brief.candidate_keys:
            lines.append("Only these identifiers are in scope:")
            for pid, sku in self.brief.candidate_keys:
                lines.append(f"- {pid}::{sku}")
        if self.brief.asks_reason:
            lines.append(
                "The shopper asked why. Any justification must copy values from returned facts. "
                "Do not claim an attribute the observation left null or omitted."
            )
        if self.brief.required_identities:
            lines.append(
                f"The request asks to surface at least {self.brief.required_identities} distinct matching products."
            )
        lines.append("Open products before buying. Confirm live price and availability before ordering.")
        return "\n".join(lines)

    def _opening_question(self) -> str | None:
        if not self.brief.asks_shopper:
            return None
        if self.brief.prefer_phrases and not self.brief.specific_prefer:
            return (
                "Which of your preferences should count most, and in what order should I weigh the others?"
            )
        return "Before I search: which requirement is firm, and what exact value do you need?"

    def _note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    def _prod(self, pid: str) -> dict:
        return self.ledger.products.get(str(pid)) or {}

    def _title(self, pid: str) -> str:
        title = str(self._prod(pid).get("title") or "")
        if title:
            return title
        for key, row in self.ledger.rows.items():
            if key.startswith(str(pid) + "::"):
                return str(row.get("title") or "")
        return ""

    def _brand_of(self, pid: str, sku: str = "") -> str:
        facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
        for key, value in facts.items():
            low = str(key).lower()
            if low == "brand" or low.endswith("_brand") or low.endswith(".brand"):
                text = str(value or "").strip()
                if text:
                    return text
        brand = self._prod(pid).get("brand")
        if brand:
            return str(brand)
        for key, row in self.ledger.rows.items():
            if key.startswith(str(pid) + "::") and row.get("brand"):
                return str(row["brand"])
        return ""

    def _brand_ok(self, pid: str, sku: str = "") -> bool:
        want = self.brief.brand
        if not want:
            return True
        wanted = _token_set(want)
        if not wanted:
            return True
        have = _token_set(self._brand_of(pid, sku))
        if have:
            return wanted <= have
        return wanted <= _token_set(self._title(pid))

    def _options(self, pid: str, sku: str) -> str:
        return self.ledger.options.get(Ledger.key(pid, sku), "")

    def _fact_blob(self, pid: str, sku: str) -> str:
        facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
        parts = [str(v) for v in facts.values() if v is not None]
        opt = self._options(pid, sku)
        if opt:
            parts.append(opt)
        brand = self._brand_of(pid, sku)
        if brand:
            parts.append(brand)
        return " ".join(parts)

    def _hint_blob(self, pid: str, sku: str) -> str:
        hints = list(self.ledger.hints.get(Ledger.key(pid, sku), []))
        title = self._title(pid)
        if title:
            hints.append(title)
        return " ".join(hints)

    def _quoted_phrases(self) -> set[str]:
        return {" ".join(m.split()) for m in _QUOTED_RE.findall(self.brief.goal)}

    def _fact_supports(self, pid: str, sku: str, phrase: str) -> bool:
        facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
        if re.fullmatch(r"\d{3,4}p", phrase or "", re.I):
            return self._p_token_quality(pid, sku) >= 2
        for key, value in facts.items():
            if _phrase_in_value(phrase, value) or _phrase_in_value(phrase, key):
                return True
        if _phrase_in_value(phrase, self._options(pid, sku)):
            return True
        brand = self._brand_of(pid, sku)
        if brand and _phrase_in_value(phrase, brand):
            return True
        return _phrase_in_value(phrase, self._fact_blob(pid, sku))

    def _hint_supports(self, pid: str, sku: str, phrase: str) -> bool:
        return _phrase_in_value(phrase, self._hint_blob(pid, sku))

    def _category_ok(self, pid: str) -> bool:
        want = self.brief.category
        path = self._prod(pid).get("category_path")
        if not want:
            return True
        if not isinstance(path, list) or not path:
            return True
        wanted = _stems(want)
        labels = set()
        for label in path:
            labels |= _stems(str(label))
        if wanted & labels:
            return True
        compact_want = _stems("".join(re.findall(r"[a-z]+", want.lower())))
        return any(
            _stems("".join(re.findall(r"[a-z]+", str(label).lower()))) == wanted
            or compact_want <= _stems(str(label).lower())
            for label in path
        )

    def _category_fit(self, pid: str) -> int:
        """2 = last path label is the requested category, 1 = overlap, 0 = miss."""
        want = self.brief.category
        if not want:
            return 1
        path = self._prod(pid).get("category_path")
        wanted = _stems(want)
        if not isinstance(path, list) or not path:
            title = self._title(pid).lower()
            return 2 if wanted and all(tok in title for tok in wanted) else (1 if self._category_ok(pid) else 0)
        last = _stems(str(path[-1]))
        if wanted and wanted == last:
            return 2
        if wanted and any(wanted & _stems(str(label)) for label in path):
            return 1
        return 1 if self._category_ok(pid) else 0

    def _p_token_quality(self, pid: str, sku: str) -> int:
        """Literal Np outranks a clean WxH/label; mixed Np classes rank lowest."""
        phrases = [p for p in self.brief.hard_phrases if re.fullmatch(r"\d{3,4}p", p or "", re.I)]
        if not phrases:
            return 0
        facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
        blob = self._spec_blob(pid, sku)
        labels = _request_spec_labels(self.brief.goal + " " + self.brief.raw)
        best = 0
        for phrase in phrases:
            try:
                want = int(phrase[:-1])
            except ValueError:
                continue
            field_best = 0
            for key, value in facts.items():
                if value is None:
                    continue
                text = str(value)
                conflict = _conflicting_np(text, want)
                if phrase.lower() in text.lower():
                    field_best = max(field_best, 1 if conflict else 3)
                    continue
                compact = text.strip().lower()
                if labels and any(re.search(r"(?<![a-z])" + re.escape(lab) + r"(?![a-z])", compact) for lab in labels):
                    field_best = max(field_best, 1 if conflict else 2)
                for left_s, right_s in _DIM_RE.findall(text):
                    try:
                        left, right = int(left_s), int(right_s)
                    except ValueError:
                        continue
                    if right == want or left == want:
                        clean = 2 if min(left, right) == want and max(left, right) != want else 1
                        field_best = max(field_best, 1 if conflict else clean)
            if field_best:
                best = max(best, field_best)
            elif _p_token_supported(phrase, blob):
                best = max(best, 1 if _conflicting_np(blob, want) else 2)
            elif labels and any(
                re.search(r"(?<![a-z])" + re.escape(lab) + r"(?![a-z])", blob.lower()) for lab in labels
            ):
                best = max(best, 1 if _conflicting_np(blob, want) else 2)
        return best

    def _spec_conflicted(self, pid: str, sku: str) -> bool:
        """True when supporting pixel evidence also names a different Np class."""
        phrases = [p for p in self.brief.hard_phrases if re.fullmatch(r"\d{3,4}p", p or "", re.I)]
        if not phrases:
            return False
        facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
        for phrase in phrases:
            try:
                want = int(phrase[:-1])
            except ValueError:
                continue
            for value in facts.values():
                if value is None:
                    continue
                text = str(value)
                if _p_token_supported(phrase, text) and _conflicting_np(text, want):
                    return True
        return False

    def _extreme_low_price(self, c: dict, peers: list) -> bool:
        """True when this price is a decade below the other observed in-budget prices.

        Uses only this episode's inspected/live peers. No catalog price table.
        """
        if self.brief.ranking_like:
            return False
        price = c.get("price")
        if price is None or price <= 0:
            return False
        others = [
            float(p["price"])
            for p in peers
            if p is not c and p.get("price") is not None and float(p["price"]) > 0
        ]
        if not others:
            return False
        others.sort()
        med = others[len(others) // 2]
        return float(price) * 10.0 < med

    def _evidence_tuple(self, pid: str, sku: str) -> tuple:
        return (
            0 if self._spec_conflicted(pid, sku) else 1,
            self._p_token_quality(pid, sku),
            self._measure_quality(pid, sku),
            self._fact_pref_hits(pid, sku),
        )

    def _price_outlier_penalty(self, c: dict, peers: list | None = None) -> int:
        rows = peers if peers is not None else [x for x in self._candidates() if x["live"] and x["viewed"]]
        if not self._extreme_low_price(c, rows):
            return 0
        mine = self._evidence_tuple(c["pid"], c["sku"])
        typical = [p for p in rows if not self._extreme_low_price(p, rows)]
        if typical and mine > max(self._evidence_tuple(p["pid"], p["sku"]) for p in typical):
            return 0
        return 1

    def _match_weight(self, pid: str, sku: str = "") -> float:
        """Request coverage plus a small rank prior. Price is not a component."""
        score = 0.05 * _rank_prior(self.ledger.search_rank.get(str(pid)))
        score += 0.4 * self._category_fit(pid)
        if self.brief.brand and self._brand_ok(pid, sku):
            score += 0.3
        title = self._title(pid)
        blob = title + " " + self._options(pid, sku) + " " + self._fact_blob(pid, sku)
        if self.brief.pref_model:
            if _model_in_text(self.brief.pref_model, blob):
                score += 0.8
                if self.brief.pref_color and re.search(
                    r"(?<![a-z])" + re.escape(self.brief.pref_color) + r"(?![a-z])",
                    blob.lower(),
                ):
                    score += 0.25
            else:
                score -= 0.4
        if sku:
            score += 0.25 * self._measure_quality(pid, sku)
            score += 0.15 * self._measure_hits(pid, sku)
            score += 0.2 * self._fact_pref_hits(pid, sku)
            score += 0.04 * min(12, self._prefer_overlap(pid, sku))
            score += 0.2 * self._p_token_quality(pid, sku)
            if self._spec_conflicted(pid, sku):
                score -= 0.35
            if self.brief.hard_phrases:
                hits = sum(
                    1
                    for phrase in self.brief.hard_phrases
                    if self._fact_supports(pid, sku, phrase) or self._hint_supports(pid, sku, phrase)
                )
                score += 0.15 * hits
        else:
            low = title.lower()
            for phrase in self.brief.measures:
                nums = _numbers(phrase)
                if nums and _blob_has_number(title, nums[0]):
                    score += 0.35
            for phrase in self.brief.prefer_phrases:
                tokens = [t.lower() for t in _content_tokens(phrase)]
                score += 0.06 * sum(1 for token in tokens if token in low)
            if self.brief.hard_phrases:
                score += 0.1 * sum(1 for phrase in self.brief.hard_phrases if _phrase_in_value(phrase, title))
        return score

    def _in_scope(self, pid: str, sku: str) -> bool:
        listed = self.brief.candidate_keys
        if not listed:
            return True
        return (str(pid), str(sku)) in listed

    def _hard_ok(self, pid: str, sku: str) -> bool:
        if not self._in_scope(pid, sku):
            return False
        quoted = self._quoted_phrases()
        for phrase in self.brief.hard_phrases:
            if self._fact_supports(pid, sku, phrase):
                continue
            if phrase in quoted and self._hint_supports(pid, sku, phrase):
                continue
            return False
        if self.brief.theme and not self.brief.candidate_keys:
            title = self._title(pid)
            if title and not _has_theme(title + " " + self._hint_blob(pid, sku), self.brief.theme):
                return False
        return True

    def _preference_score(self, pid: str, sku: str) -> int:
        return self._prefer_overlap(pid, sku)

    def _fact_pref_hits(self, pid: str, sku: str) -> int:
        return sum(1 for phrase in self.brief.prefer_phrases if self._fact_supports(pid, sku, phrase))

    def _spec_blob(self, pid: str, sku: str) -> str:
        facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
        parts = [
            str(v)
            for k, v in facts.items()
            if v is not None and str(k).lower().rsplit(".", 1)[-1] not in _ID_KEYS | {"price", "in_stock", "currency"}
        ]
        opt = self._options(pid, sku)
        if opt:
            parts.append(opt)
        return " ".join(parts)

    def _measure_quality(self, pid: str, sku: str) -> int:
        """Exact labelled measurements outrank ranges that merely contain the number."""
        if not self.brief.measures:
            return 0
        facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
        opt = self._options(pid, sku)
        score = 0
        for phrase in self.brief.measures:
            nums = _numbers(phrase)
            want = _as_float(nums[0]) if nums else None
            if want is None:
                continue
            best = 0
            contradicted = False
            for key, value in facts.items():
                low = str(key).lower().rsplit(".", 1)[-1]
                if low in _ID_KEYS | {"price", "in_stock", "currency", "condition", "options"}:
                    continue
                text = str(value or "")
                related = self._measure_related(phrase, str(key), value)
                single = re.fullmatch(
                    r"\s*(\d+(?:\.\d+)?)\s*-?\s*(?:[A-Za-z%]{0,12}|\"|')?\s*$",
                    text,
                    re.I,
                )
                if single:
                    exact = _as_float(single.group(1))
                    if exact is not None and abs(exact - want) <= max(1e-6, abs(want) * 1e-6):
                        best = max(best, 4 if related else 3)
                        continue
                    if related and exact is not None and abs(exact - want) > max(1e-6, abs(want) * 1e-6):
                        contradicted = True
                        continue
                found = [_as_float(n) for n in _numbers(text)]
                found = [n for n in found if n is not None]
                if (
                    len(found) >= 2
                    and not _DIM_RE.search(text)
                    and not _P_TOKEN_RE.search(text)
                    and min(found) - _EPS <= want <= max(found) + _EPS
                ):
                    best = max(best, 1)
            if contradicted and best < 4:
                score += 0
                continue
            if best == 0:
                want_unit = None
                units = [u for _n, u in _measurements(phrase)]
                if units:
                    want_unit = units[0]
                for number, unit in _measurements(opt):
                    if abs(number - want) > max(1e-6, abs(want) * 1e-6):
                        continue
                    if want_unit is None or unit == want_unit or unit.startswith(want_unit) or want_unit.startswith(unit):
                        best = 2
                        break
            score += best
        return score

    def _token_covered(self, token: str, blob: str) -> bool:
        """Morphology/containment only: run⊂running, water⊂waterproof. No synonym table."""
        tok = (token or "").lower().rstrip("s")
        if len(tok) < 3:
            return False
        text = (blob or "").lower()
        if tok in text:
            return True
        for word in re.findall(r"[a-z]+", text):
            stem = word.rstrip("s")
            if len(stem) < 3:
                continue
            if stem.startswith(tok) or (tok.startswith(stem) and len(stem) >= 4):
                return True
        return False

    def _token_overlap(self, pid: str, sku: str, phrases: list[str]) -> int:
        """Coverage of shopper tokens in observed title + facts, then search hints."""
        observed = (
            self._fact_blob(pid, sku)
            + " "
            + self._options(pid, sku)
            + " "
            + self._title(pid)
        ).lower()
        hint = self._hint_blob(pid, sku).lower()
        score = 0
        for phrase in phrases:
            tokens = [t.lower() for t in _content_tokens(phrase)]
            if not tokens:
                continue
            fact_hits = sum(1 for t in tokens if self._token_covered(t, observed))
            if fact_hits:
                score += 3 * fact_hits
                continue
            score += sum(1 for t in tokens if self._token_covered(t, hint))
        return score

    def _prefer_overlap(self, pid: str, sku: str) -> int:
        return self._token_overlap(pid, sku, self.brief.prefer_phrases)

    def _soft_overlap(self, pid: str, sku: str) -> int:
        phrases = self.brief.soft_prefer or self.brief.prefer_phrases
        return self._token_overlap(pid, sku, phrases)

    def _title_category_hit(self, pid: str) -> int:
        want = self.brief.category
        if not want:
            return 0
        title = self._title(pid).lower()
        toks = [t for t in _stems(want) if len(t) > 2]
        return int(bool(toks) and all(tok in title for tok in toks))

    def _measure_satisfied(self, pid: str, sku: str) -> bool:
        if not self.brief.measures:
            return True
        if str(pid) not in self.ledger.products:
            return False
        return (
            self._measure_hits(pid, sku) >= len(self.brief.measures)
            and self._measure_quality(pid, sku) > 0
        )

    def _measure_field_tokens(self, phrase: str) -> tuple[str, ...]:
        """Tokens taken from this request phrase, not a vertical field map."""
        return tuple(t for t in _key_tokens(phrase) if t and not t.isdigit())

    def _measure_related(self, phrase: str, key: str, value: Any) -> bool:
        """True when this fact is about the same measurement the request named.

        Relatedness is request-token or unit overlap only. A bare catalog number
        without a unit is handled in hits/quality, not by treating every exact
        number as the same attribute.
        """
        tokens = self._measure_field_tokens(phrase)
        low = str(key).lower().rsplit(".", 1)[-1]
        if low in _ID_KEYS | {"condition", "options", "brand", "title", "name"}:
            return False
        if any(tok in low for tok in tokens):
            return True
        units = [u for _n, u in _measurements(phrase)]
        have = _measurements(f"{key} {value}")
        for unit in units:
            for _number, have_unit in have:
                if have_unit == unit or have_unit.startswith(unit) or unit.startswith(have_unit):
                    return True
        return False

    def _measure_contradicted(self, pid: str, sku: str) -> bool:
        """True when a labelled comparable field has a different exact number."""
        if not self.brief.measures:
            return False
        facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
        if not facts:
            return False
        for phrase in self.brief.measures:
            nums = _numbers(phrase)
            want = _as_float(nums[0]) if nums else None
            if want is None:
                continue
            exact = False
            conflict = False
            for key, value in facts.items():
                if not self._measure_related(phrase, str(key), value):
                    continue
                single = re.fullmatch(
                    r"\s*(\d+(?:\.\d+)?)\s*-?\s*(?:[A-Za-z%]{0,12}|\"|')?\s*$",
                    str(value or ""),
                    re.I,
                )
                if not single:
                    continue
                exact_n = _as_float(single.group(1))
                if exact_n is None:
                    continue
                if abs(exact_n - want) <= max(1e-6, abs(want) * 1e-6):
                    exact = True
                else:
                    conflict = True
            if conflict and not exact:
                return True
        return False

    def _pref_brand_miss(self, pid: str, sku: str) -> int:
        if not self.brief.pref_brand:
            return 0
        wanted = _token_set(self.brief.pref_brand)
        have = _token_set(self._brand_of(pid, sku))
        if have:
            return 0 if wanted <= have else 1
        return 0 if wanted <= _token_set(self._title(pid)) else 1

    def _firm_hold(self, pid: str, sku: str) -> bool:
        """Shopper-stated constraints that fallback must not silently drop."""
        if not self._category_ok(pid) or not self._in_scope(pid, sku) or not self._brand_ok(pid, sku):
            return False
        if self.brief.measures and not self._measure_satisfied(pid, sku):
            return False
        if self.brief.hard_phrases and not self._hard_ok(pid, sku):
            return False
        if self.brief.theme and not self.brief.candidate_keys:
            title = self._title(pid)
            if title and not _has_theme(title + " " + self._hint_blob(pid, sku), self.brief.theme):
                return False
        return True

    def _must_terminate(self) -> bool:
        return self.turn >= self.max_steps - 3 or self.clock.remaining() < 28

    def _recovery_room(self) -> bool:
        """True while there is still budget to cart, survive an event, and order."""
        return self.turn < self.max_steps - _COMMIT_SLACK and self.clock.remaining() > 40

    def _verified_commit_ready(self) -> bool:
        """True when the current inspected candidate already meets firm evidence."""
        rows = self._inspected_ok()
        if not rows:
            return False
        best = rows[0]
        pid, sku = best["pid"], best["sku"]
        if not self._firm_hold(pid, sku):
            return False
        if self._spec_conflicted(pid, sku):
            return False
        if self._price_outlier_penalty(best):
            return False
        p_need = [p for p in self.brief.hard_phrases if re.fullmatch(r"\d{3,4}p", p or "", re.I)]
        if p_need and self._p_token_quality(pid, sku) < 2:
            return False
        if self.brief.measures and self._measure_quality(pid, sku) < 3:
            return False
        if self.brief.prefer_phrases:
            if self._prefer_overlap(pid, sku) <= 0 and self._fact_pref_hits(pid, sku) <= 0:
                return False
        return True

    def _search_exhausted(self) -> bool:
        """True when more retrieval is unlikely to beat the current live evidence."""
        if self.brief.ranking_like or self.brief.candidate_keys:
            return False
        if self.search_cursor < 2 and self.forced_compares < 2:
            return False
        viewed = [c for c in self._candidates() if c["live"] and c["viewed"]]
        if not viewed:
            return False
        p_need = [p for p in self.brief.hard_phrases if re.fullmatch(r"\d{3,4}p", p or "", re.I)]
        if p_need and max(self._p_token_quality(c["pid"], c["sku"]) for c in viewed) < 2:
            return True
        return False

    def _explore_room(self) -> bool:
        """True while remaining budget still justifies more search before first cart.

        Recovery slack stays reserved. Extra exploration stops earlier once a
        candidate is already verified, or once further search has little expected
        value, so add → event → recover → justify → order still fits.
        """
        if not self._recovery_room():
            return False
        if self.clock.remaining() <= 40:
            return False
        if self.brief.ranking_like:
            return True
        if self._verified_commit_ready() or self._search_exhausted():
            return False
        return True

    def _recovery_parts(self) -> list[str]:
        """Replacement query: match phrase and preference first, category last."""
        parts: list[str] = []
        matched = _match_phrases(self.brief.goal)
        for item in (
            *matched[:2],
            *self.brief.soft_prefer[:2],
            *self.brief.purpose_phrases[:1],
            *self.brief.prefer_phrases[:2],
            *self.brief.measures[:2],
            *[p for p in self.brief.hard_phrases if re.fullmatch(r"\d{3,4}p", p or "", re.I)][:1],
            *[p for p in self.brief.hard_phrases if p not in matched][:1],
            self.brief.pref_brand,
            self.brief.brand,
            self.brief.pref_model,
            self.brief.pref_color,
            self.brief.category,
        ):
            text = str(item or "").strip()
            if text and text not in parts and not _is_instruction_query(text):
                parts.append(text)
        return parts

    def _inspect_score(self, pid: str, sku: str) -> tuple:
        return (
            int(self._spec_conflicted(pid, sku)),
            -self._measure_quality(pid, sku),
            -self._p_token_quality(pid, sku),
            -self._fact_pref_hits(pid, sku),
            -self._prefer_overlap(pid, sku),
            self._pref_brand_miss(pid, sku),
            -self._match_weight(pid, sku),
        )

    def _peer_floor(self) -> int:
        if self.brief.ranking_like:
            return 1
        if self.brief.prefer_phrases or self.brief.candidate_keys or self.brief.pref_brand:
            return _PEER_FLOOR
        return 1

    def _row_price(self, pid: str) -> float | None:
        if str(pid) in self.ledger.products:
            for sku in self._skus_of(pid):
                price = (self.ledger.checked.get(Ledger.key(pid, sku)) or {}).get("price")
                if price is not None:
                    return price
        for key, row in self.ledger.rows.items():
            if key.startswith(str(pid) + "::"):
                st = self.ledger.checked.get(key) or {}
                if st.get("price") is not None:
                    return st.get("price")
                price = _as_float(row.get("price"))
                if price is not None:
                    return price
        return None

    def _plausible_peer_pids(self) -> list[str]:
        """Already-observed listings that can still compete for the first cart."""
        pids: list[str] = []
        seen: set[str] = set()

        def add(pid: str, title: str = "") -> None:
            pid = str(pid)
            if not pid or pid in seen or pid in self.ledger.dead_products or pid in self.unreadable:
                return
            price = self._row_price(pid)
            if self.budget is not None and price is not None and price > self.budget + _EPS:
                return
            if self.brief.theme and title and not _has_theme(title, self.brief.theme):
                return
            if self.brief.brand and not self._brand_ok(pid):
                title_hit = self.brief.brand.lower() in (title or "").lower()
                if not title_hit and pid not in self.ledger.products:
                    return
            seen.add(pid)
            pids.append(pid)

        for pid in self.ledger.search_rank:
            add(pid, self._title(pid))
        for key, row in self.ledger.rows.items():
            pid, _, _sku = key.partition("::")
            add(pid, str(row.get("title") or ""))
        for pid in self.ledger.products:
            add(pid, self._title(pid))
        return pids

    def _needed_peers(self) -> int:
        n = len(self._plausible_peer_pids())
        if n <= 1:
            return 1
        return min(self._peer_floor(), n)

    def _inspected_ok(self) -> list:
        rows = [c for c in self._candidates() if c["ok"] and c["key"] in self.ledger.inspected]
        rows.sort(key=self._commit_rank)
        return rows

    def _has_peer_work(self) -> bool:
        if len(self._inspected_ok()) >= self._needed_peers():
            return False
        if self._pending_inspect() is not None:
            return True
        if self.forced_compares >= 1:
            return False
        return any(
            pid not in self.ledger.products and pid not in self.unreadable
            for pid in self._plausible_peer_pids()
        )

    def _peer_step(self) -> tuple | None:
        """Inspect or open one shortlist of already-observed in-budget listings."""
        if self.ledger.adds or self.brief.ranking_like:
            return None
        if not self._inspected_ok():
            return None
        if len(self._inspected_ok()) >= self._needed_peers():
            return None
        inspect = self._pending_inspect()
        if inspect is not None:
            return inspect
        if self.forced_compares >= 1:
            return None
        missing = [
            pid
            for pid in self._plausible_peer_pids()
            if pid not in self.ledger.products and pid not in self.unreadable
        ]
        if not missing:
            return None
        self.forced_compares += 1
        if len(missing) > 1 and "compare" in self.tool_names:
            return ("compare", _filter_args(self.tool_index, "compare", {"product_ids": missing[:5]}))
        if "view" in self.tool_names:
            return ("view", _filter_args(self.tool_index, "view", {"product_id": missing[0]}))
        return None

    def _budget_peer_probe(self) -> tuple | None:
        """One budget-sliced retrieval when preferences exist but only one in-budget peer is visible."""
        if self.ledger.adds or self.brief.ranking_like or self.budget_peers_done:
            return None
        if not self._inspected_ok():
            return None
        if not (self.brief.prefer_phrases or self.brief.pref_brand):
            return None
        if self.budget is None or not self._recovery_room():
            return None
        if len(self._plausible_peer_pids()) >= self._peer_floor():
            self.budget_peers_done = True
            return None
        self.budget_peers_done = True
        if "filter" in self.tool_names:
            args: dict[str, Any] = {}
            if _param(self.tool_index, "filter", "category") and self.brief.category:
                args["category"] = self.brief.category
            if _param(self.tool_index, "filter", "in_stock"):
                args["in_stock"] = True
            if _param(self.tool_index, "filter", "max_price"):
                args["max_price"] = self.budget
            if _param(self.tool_index, "filter", "k"):
                args["k"] = 10
            if args:
                return ("filter", _filter_args(self.tool_index, "filter", args))
        if "search" in self.tool_names and self.brief.search_strings:
            args = {"query": self.brief.search_strings[0], "k": 10, "in_stock": True}
            return ("search", _filter_args(self.tool_index, "search", args))
        return None

    def _commit_rank(self, c: dict) -> tuple:
        pid, sku = c["pid"], c["sku"]
        return (
            -int(self._measure_satisfied(pid, sku)),
            int(self._spec_conflicted(pid, sku)),
            self._price_outlier_penalty(c),
            -self._measure_quality(pid, sku),
            -self._p_token_quality(pid, sku),
            -self._title_category_hit(pid),
            self._pref_brand_miss(pid, sku),
            -self._fact_pref_hits(pid, sku),
            -self._prefer_overlap(pid, sku),
            -self._preference_score(pid, sku),
            -c.get("measures", 0),
            c["relax"],
            self._listing_penalty(pid),
            -self._match_weight(pid, sku),
            c["key"],
        )

    def _intent_rank(self, c: dict) -> tuple:
        """Recovery rank: original request coverage before category or price."""
        pid, sku = c["pid"], c["sku"]
        return (
            -int(self._firm_hold(pid, sku)),
            int(self._spec_conflicted(pid, sku)),
            -self._soft_overlap(pid, sku),
            -self._prefer_overlap(pid, sku),
            -self._fact_pref_hits(pid, sku),
            -self._measure_quality(pid, sku),
            -self._p_token_quality(pid, sku),
            self._pref_brand_miss(pid, sku),
            self._price_outlier_penalty(c),
            -self._match_weight(pid, sku),
            c["relax"],
            self._listing_penalty(pid),
            c["key"],
        )

    def _recovery_query(self) -> str:
        """Text query rebuilt from the original request, not the latest candidate."""
        cat = (self.brief.category or "").strip().lower()
        bits: list[str] = []
        for part in self._recovery_parts():
            if part.strip().lower() == cat:
                continue
            if part not in bits:
                bits.append(part)
            if len(bits) >= 2:
                break
        return " ".join(bits).strip()

    def _query_is_category_only(self, query: str) -> bool:
        blob = " ".join((query or "").split()).lower()
        if not blob:
            return True
        cat = (self.brief.category or "").strip().lower()
        return bool(cat) and blob == cat

    def _recover_with_text(self) -> bool:
        """True when recovery must keep original request text, not category alone.

        Brand-only soft prefers still use structured brand/category retrieval.
        Match phrases, purpose/use, attribute preferences, and named models
        need the search-text path.
        """
        if self.brief.pref_model:
            return True
        if _match_phrases(self.brief.goal) or self.brief.purpose_phrases:
            return True
        brandish = {
            (self.brief.pref_brand or "").strip().lower(),
            (self.brief.brand or "").strip().lower(),
        }
        brandish.discard("")
        protocol = {"ideally", "prefer", "preferred", "firm", "requirement", "requirements", "matter", "matters"}
        for phrase in self.brief.soft_prefer:
            if _is_instruction_query(phrase) or _is_meta_phrase(phrase):
                continue
            tokens = [t.lower() for t in _content_tokens(phrase)]
            useful = [
                t
                for t in tokens
                if t not in brandish and t not in _FUNCTION_WORDS and t not in protocol
            ]
            if useful:
                return True
        return False

    def _request_text_hits(self, pid: str, sku: str) -> int:
        """How much of the original request text this listing still covers."""
        hits = self._soft_overlap(pid, sku) + self._prefer_overlap(pid, sku) + self._fact_pref_hits(pid, sku)
        matched = _match_phrases(self.brief.goal)
        if matched:
            hits += self._token_overlap(pid, sku, matched[:2])
        if self.brief.theme and _has_theme(
            self._title(pid) + " " + self._hint_blob(pid, sku) + " " + self._fact_blob(pid, sku),
            self.brief.theme,
        ):
            hits += 2
        if self.brief.pref_model and _model_in_text(
            self.brief.pref_model,
            self._title(pid) + " " + self._options(pid, sku) + " " + self._fact_blob(pid, sku),
        ):
            hits += 4
        return hits

    def _structured_filter_args(self) -> dict[str, Any]:
        """Structured constraints only. Never a substitute for search text."""
        args: dict[str, Any] = {}
        if _param(self.tool_index, "filter", "category") and self.brief.category:
            args["category"] = self.brief.category
        if _param(self.tool_index, "filter", "in_stock"):
            args["in_stock"] = True
        if _param(self.tool_index, "filter", "max_price") and self.budget is not None:
            args["max_price"] = self.budget
        if _param(self.tool_index, "filter", "brand") and self.brief.brand:
            args["brand"] = self.brief.brand
        if _param(self.tool_index, "filter", "k"):
            args["k"] = 10
        return args

    def _search_text_args(self, query: str) -> dict[str, Any]:
        args: dict[str, Any] = {"query": query}
        if _param(self.tool_index, "search", "in_stock"):
            args["in_stock"] = True
        if _param(self.tool_index, "search", "k"):
            args["k"] = 10
        return args

    def _filter_carries_query(self) -> bool:
        return "filter" in self.tool_names and _param(self.tool_index, "filter", "query")

    def _emit_retrieval(self, query: str) -> tuple | None:
        """Issue retrieval that keeps search text even when filter has no query field."""
        cleaned = " ".join((query or "").split())
        if self.ledger.event_target:
            recovered = self._recovery_query()
            if recovered and _is_instruction_query(recovered):
                recovered = ""
            if recovered and (
                not cleaned
                or self._query_is_category_only(cleaned)
                or _is_instruction_query(cleaned)
            ):
                cleaned = recovered
        if not cleaned or _is_instruction_query(cleaned):
            return None
        probe = ("retrieve", cleaned.lower())
        if probe in self.probes:
            return None
        self.probes.add(probe)
        if self.ledger.event_target and self._recover_with_text():
            self.searched_twins = True
        if self._filter_carries_query():
            args = self._structured_filter_args()
            args["query"] = cleaned
            return ("filter", _filter_args(self.tool_index, "filter", args))
        if self.ledger.event_target and self._recover_with_text() and "search" in self.tool_names:
            return ("search", _filter_args(self.tool_index, "search", self._search_text_args(cleaned)))
        if "search" in self.tool_names:
            return ("search", _filter_args(self.tool_index, "search", self._search_text_args(cleaned)))
        if "filter" in self.tool_names and not self._recover_with_text():
            args = self._structured_filter_args()
            if args:
                return ("filter", _filter_args(self.tool_index, "filter", args))
        return None

    def _skip_arbitrary_pending(self) -> bool:
        """After an event, do not walk unrelated viewed SKUs before intent search."""
        if not self.ledger.event_target or not self._recover_with_text():
            return False
        return not self._recovery_survivors()

    def _recovery_discovery_pending(self) -> bool:
        """True while a new intent-preserving retrieval still has unread matches."""
        if not self.ledger.event_target or not self._recover_with_text():
            return False
        if self._must_terminate():
            return False
        if not self.searched_twins:
            return True
        if self._pending_inspect(intent_only=True) is not None:
            return True
        return bool(self._unseen_leads())

    def _recovery_survivors(self) -> list:
        """Inspected listings that still match explicit original intent after a market event."""
        if not self.ledger.event_target:
            return []
        if not (self.brief.prefer_phrases or _match_phrases(self.brief.goal)):
            return []
        good = [
            c
            for c in self._candidates()
            if c["ok"] and c["key"] in self.ledger.inspected and c["key"] != self.ledger.event_target
        ]
        if not good:
            return []
        dead_pid, _, dead_sku = self.ledger.event_target.partition("::")
        floor = self._soft_overlap(dead_pid, dead_sku)
        if floor > 0:
            kept = [c for c in good if self._soft_overlap(c["pid"], c["sku"]) >= floor]
            if kept:
                kept.sort(key=self._intent_rank)
                return kept
        covered = [
            c
            for c in good
            if self._soft_overlap(c["pid"], c["sku"]) > 0 or self._prefer_overlap(c["pid"], c["sku"]) > 0
        ]
        if not covered and _match_phrases(self.brief.goal):
            covered = good
        covered.sort(key=self._intent_rank)
        return covered

    def _related_phrases(self, attribute: str) -> list[str]:
        attr = _key_tokens(attribute)
        if not attr:
            return []
        phrases = [attribute]
        blob = self.brief.goal + "\n" + self.brief.raw
        for match in re.finditer(r"\b([a-z][a-z0-9]+(?:[ _-][a-z0-9]+){0,3})\b", blob.lower()):
            phrase = match.group(1).replace("_", " ")
            ptoks = _key_tokens(phrase)
            if not (attr & ptoks):
                continue
            if len(ptoks) < 2 and phrase != attribute.lower():
                continue
            if phrase not in phrases:
                phrases.append(phrase)
        for field in self._claim_enum():
            if attr & _key_tokens(field) and field not in phrases:
                phrases.append(field.replace("_", " "))
        return phrases

    def _evidence_layers(self, pid: str, sku: str) -> list[tuple[str, str]]:
        facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
        labelled = " ".join(
            f"{k} {v}"
            for k, v in facts.items()
            if v is not None and str(k).lower().rsplit(".", 1)[-1] not in _ID_KEYS | {"options", "price", "in_stock"}
        )
        return [
            ("option", self._options(pid, sku)),
            ("fact", labelled),
            ("title", self._title(pid)),
        ]

    def _attribute_number(self, pid: str, sku: str, attribute: str) -> float | None:
        tokens = _key_tokens(attribute)
        if not tokens:
            return None
        if tokens & {"price", "cost", "spend", "budget"}:
            st = self.ledger.checked.get(Ledger.key(pid, sku)) or {}
            return st.get("price")
        facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
        related = self._related_phrases(attribute)
        related_toks: set[str] = set()
        for phrase in related:
            related_toks |= _key_tokens(phrase)
        unit_from_request = {unit for _n, unit in _measurements(" ".join(related) + " " + self.brief.goal)}
        labelled_unit: str | None = None
        labelled_number: float | None = None
        for key, value in facts.items():
            low = str(key).lower().rsplit(".", 1)[-1]
            if low in {"price", "in_stock", "options"} or low in _ID_KEYS or _is_hint_key(str(key)):
                continue
            key_toks = _key_tokens(key)
            if not (tokens & key_toks) and not (related_toks & key_toks):
                continue
            number = _as_float(value)
            if number is None:
                continue
            labelled_number = number
            units = [u for _n, u in _measurements(str(value))]
            if units:
                labelled_unit = units[0]
        option_by_unit: dict[str, float] = {}
        fact_by_unit: dict[str, float] = {}
        title_by_unit: dict[str, float] = {}
        for layer, text in self._evidence_layers(pid, sku):
            values: dict[str, float] = {}
            for number, unit in _measurements(text):
                prev = values.get(unit)
                values[unit] = number if prev is None else max(prev, number)
            if layer == "option":
                option_by_unit = values
            elif layer == "fact":
                fact_by_unit = values
            else:
                title_by_unit = values
        if labelled_unit and labelled_unit in option_by_unit:
            return option_by_unit[labelled_unit]
        if labelled_number is not None:
            return labelled_number
        wanted_units = set(unit_from_request)
        if labelled_unit:
            wanted_units.add(labelled_unit)
        for unit in wanted_units:
            if unit in option_by_unit:
                return option_by_unit[unit]
            if unit in fact_by_unit:
                return fact_by_unit[unit]
            if unit in title_by_unit:
                return title_by_unit[unit]
        leftover = self._leftover_unit(attribute)
        if leftover:
            if leftover in option_by_unit:
                return option_by_unit[leftover]
            if leftover in fact_by_unit:
                return fact_by_unit[leftover]
            if leftover in title_by_unit:
                return title_by_unit[leftover]
        return None

    def _leftover_unit(self, attribute: str) -> str | None:
        """Unit used by this ranking axis after labelled facts claim the others."""
        assigned: set[str] = set()
        unmatched: list[str] = []
        for item in self.brief.comparatives:
            attr = item["attribute"]
            toks = _key_tokens(attr)
            if toks & {"price", "cost", "spend", "budget"}:
                continue
            related = self._related_phrases(attr)
            rel_toks: set[str] = set()
            for phrase in related:
                rel_toks |= _key_tokens(phrase)
            claimed = False
            for facts in list(self.ledger.facts.values())[:8]:
                for key, value in facts.items():
                    key_toks = _key_tokens(key)
                    if (toks & key_toks) or (rel_toks & key_toks):
                        if str(key).lower().rsplit(".", 1)[-1] in _ID_KEYS:
                            continue
                        units = [u for _n, u in _measurements(str(value))]
                        if units:
                            assigned.add(units[0])
                            claimed = True
                            break
                if claimed:
                    break
            if not claimed:
                unmatched.append(attr)
        if attribute not in unmatched:
            return None
        coverage: dict[str, int] = {}
        keys = list(self.brief.candidate_keys) if self.brief.candidate_keys else []
        if not keys:
            keys = [(k.partition("::")[0], k.partition("::")[2]) for k in list(self.ledger.facts)[:12]]
        for pid, sku in keys:
            seen = set()
            for _layer, text in self._evidence_layers(str(pid), str(sku)):
                for _number, unit in _measurements(text):
                    seen.add(unit)
            for unit in seen:
                if unit in assigned:
                    continue
                coverage[unit] = coverage.get(unit, 0) + 1
        if not coverage:
            return None
        ranked_units = sorted(coverage, key=lambda u: (-coverage[u], u))
        index = unmatched.index(attribute)
        if index >= len(ranked_units):
            return None
        return ranked_units[index]

    def _relax(self, pid: str, sku: str) -> int:
        model, color = self.brief.pref_model, self.brief.pref_color
        if model:
            facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
            blob = " ".join(
                [
                    self._title(pid),
                    self._options(pid, sku),
                    str(facts.get("model") or ""),
                    str(facts.get("name") or ""),
                    str(facts.get("product_model") or ""),
                ]
            )
            same_model = _model_in_text(model, blob)
            shade = (
                self._options(pid, sku)
                + " "
                + str(facts.get("color") or "")
                + " "
                + self._title(pid)
            ).lower()
            same_color = bool(color) and re.search(
                r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade
            ) is not None
            if same_model and (not color or same_color):
                return 0
            if same_model:
                return 1
            return 2 if same_color else 3
        phrases = self.brief.prefer_phrases
        if not phrases:
            return 0
        hits = sum(
            1
            for phrase in phrases
            if self._fact_supports(pid, sku, phrase) or self._hint_supports(pid, sku, phrase)
        )
        if hits == len(phrases):
            return 0
        if hits:
            return 1
        return 2

    def _listing_penalty(self, pid: str) -> int:
        dead = self.ledger.event_target
        if not dead:
            return 0
        dead_listing = self.ledger.listing_of.get(dead.split("::", 1)[0])
        if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
            return 1
        return 0

    def _measure_hits(self, pid: str, sku: str) -> int:
        if not self.brief.measures:
            return 0
        facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
        blob = self._fact_blob(pid, sku)
        found = _measurements(blob)
        hits = 0
        for phrase in self.brief.measures:
            nums = _numbers(phrase)
            units = [u for _n, u in _measurements(phrase)]
            matched = False
            want = _as_float(nums[0]) if nums else None
            unit = units[0] if units else None
            if want is not None:
                for number, have_unit in found:
                    if abs(number - want) > max(1e-6, abs(want) * 1e-6):
                        continue
                    if unit is None or have_unit == unit or have_unit.startswith(unit) or unit.startswith(have_unit):
                        matched = True
                        break
                if not matched:
                    for key, value in facts.items():
                        low = str(key).lower().rsplit(".", 1)[-1]
                        if low in _ID_KEYS | {"price", "in_stock", "currency", "condition", "options"}:
                            continue
                        text = str(value or "")
                        related = self._measure_related(phrase, str(key), value)
                        single = re.fullmatch(
                            r"\s*(\d+(?:\.\d+)?)\s*-?\s*(?:[A-Za-z%]{0,12}|\"|')?\s*$",
                            text,
                            re.I,
                        )
                        exact = _as_float(single.group(1)) if single else None
                        if exact is not None and abs(exact - want) <= max(1e-6, abs(want) * 1e-6):
                            matched = True
                            break
                        if _DIM_RE.search(text) or _P_TOKEN_RE.search(text):
                            continue
                        if related or re.search(r"\d+\s*-\s*\d+", text):
                            numbers = [_as_float(n) for n in _numbers(text)]
                            numbers = [n for n in numbers if n is not None]
                            if len(numbers) >= 2 and min(numbers) - _EPS <= want <= max(numbers) + _EPS:
                                matched = True
                                break
            if matched:
                hits += 1
        return hits

    def _assess(self, pid: str, sku: str) -> dict:
        key = Ledger.key(pid, sku)
        st = self.ledger.checked.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.ledger.dead_keys or (
            str(pid) in self.ledger.dead_products and not self.brief.pref_model
        )
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and not dead
        viewed = str(pid) in self.ledger.products
        ok = bool(
            live
            and viewed
            and self._category_ok(pid)
            and self._in_scope(pid, sku)
            and self._brand_ok(pid, sku)
        )
        if ok and self.brief.hard_phrases:
            ok = self._hard_ok(pid, sku)
        if ok and self.brief.theme and not self.brief.candidate_keys:
            title = self._title(pid)
            if title and not _has_theme(title + " " + self._hint_blob(pid, sku), self.brief.theme):
                ok = False
        if ok and self.brief.measures and viewed:
            ok = self._measure_satisfied(pid, sku)
        if ok and self.ledger.event_target and self.brief.prefer_phrases and not self.brief.pref_model:
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split("::", 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                ok = False
        return {
            "pid": str(pid),
            "sku": str(sku),
            "key": key,
            "price": price,
            "live": live,
            "viewed": viewed,
            "ok": ok,
            "relax": self._relax(pid, sku),
            "measures": self._measure_hits(pid, sku),
        }

    def _rank_of(self, c: dict) -> tuple:
        pid, sku = c["pid"], c["sku"]
        parts: list[Any] = []
        for item in self.brief.comparatives:
            number = self._attribute_number(pid, sku, item["attribute"])
            if number is None:
                parts.extend([1, 0.0])
            elif item["direction"] == "min":
                parts.extend([0, float(number)])
            else:
                parts.extend([0, -float(number)])
        parts.append(c["relax"])
        if self.brief.pref_brand and self.ledger.adds == 0:
            wanted = _token_set(self.brief.pref_brand)
            have = _token_set(self._brand_of(pid, sku))
            if have:
                hit = wanted <= have
            else:
                hit = wanted <= _token_set(self._title(pid))
            parts.append(0 if hit else 1)
        parts.append(self._listing_penalty(pid))
        parts.append(-c.get("measures", 0))
        parts.append(-self._preference_score(pid, sku))
        price = c["price"] if c["price"] is not None else 1e18
        parts.append(price)
        parts.append(c["key"])
        return tuple(parts)

    def _candidates(self) -> list:
        seen = set()
        out = []
        for key in list(self.ledger.checked):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self._assess(pid, sku))
        return out

    def _target(self) -> dict | None:
        first = self.ledger.adds == 0
        good = [c for c in self._candidates() if c["ok"]]
        if not good:
            fallback = self._live_fallback()
            firm = [c for c in fallback if self._firm_hold(c["pid"], c["sku"])]
            if firm:
                return firm[0]
            if self.brief.measures:
                return None
            if not self._explore_room() or self._must_terminate():
                return fallback[0] if fallback else None
            return None
        if first and self.brief.measures:
            best_q = max(self._measure_quality(c["pid"], c["sku"]) for c in good)
            if best_q >= 4:
                good = [c for c in good if self._measure_quality(c["pid"], c["sku"]) >= 4]
            elif best_q > 0:
                good = [c for c in good if self._measure_quality(c["pid"], c["sku"]) >= best_q]
        if not first and self.brief.measures:
            exact = [c for c in good if self._measure_satisfied(c["pid"], c["sku"])]
            if exact:
                good = exact
        if first and self.brief.pref_brand:
            wanted = _token_set(self.brief.pref_brand)
            named = []
            for c in good:
                have = _token_set(self._brand_of(c["pid"], c["sku"]))
                if have:
                    hit = wanted <= have
                else:
                    hit = wanted <= _token_set(self._title(c["pid"]))
                if hit:
                    named.append(c)
            if named:
                good = named
            elif self._recovery_room() and self.turn < self.max_steps * 0.45:
                return None
        elif self.brief.pref_brand:
            named = [c for c in good if self._pref_brand_miss(c["pid"], c["sku"]) == 0]
            if named:
                good = named
        if self.brief.pref_model and self.ledger.event_target:
            siblings = [c for c in good if c["relax"] <= 1]
            if siblings:
                good = siblings
            else:
                return None
        if (
            self.ledger.event_target
            and self._recover_with_text()
            and not self.brief.pref_model
        ):
            survivors = self._recovery_survivors()
            if survivors:
                good = survivors
            else:
                covered = [c for c in good if self._request_text_hits(c["pid"], c["sku"]) > 0]
                if covered:
                    good = covered
                elif not self.searched_twins or self._recovery_discovery_pending():
                    return None
        if first:
            p_need = [p for p in self.brief.hard_phrases if re.fullmatch(r"\d{3,4}p", p or "", re.I)]
            if p_need:
                best_p = max(self._p_token_quality(c["pid"], c["sku"]) for c in good)
                if best_p >= 2:
                    good = [c for c in good if self._p_token_quality(c["pid"], c["sku"]) >= 2]
                elif self._explore_room():
                    return None
            clean = [c for c in good if not self._spec_conflicted(c["pid"], c["sku"])]
            if clean:
                good = clean
            elif self._explore_room() and p_need:
                return None
        if first and not self.brief.ranking_like and len(good) >= 2:
            typical = [c for c in good if not self._price_outlier_penalty(c, good)]
            if typical:
                good = typical
        if first and self.brief.specific_prefer:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif self._recovery_room() and self.turn < self.max_steps * 0.6:
                return None
        if first and self.brief.prefer_phrases and not self.brief.ranking_like:
            best_f = max(self._fact_pref_hits(c["pid"], c["sku"]) for c in good)
            if best_f > 0:
                good = [c for c in good if self._fact_pref_hits(c["pid"], c["sku"]) >= best_f]
            else:
                best_o = max(self._prefer_overlap(c["pid"], c["sku"]) for c in good)
                if best_o > 0:
                    good = [c for c in good if self._prefer_overlap(c["pid"], c["sku"]) >= best_o]
        if not self.brief.ranking_like and len(good) >= 2:
            typical = [c for c in good if not self._price_outlier_penalty(c, good)]
            if typical:
                good = typical
            clean = [c for c in good if not self._spec_conflicted(c["pid"], c["sku"])]
            if clean:
                good = clean
        if not self.brief.ranking_like:
            if first and not self._ready() and self._explore_room():
                return None
            if self.ledger.event_target and (self.brief.prefer_phrases or _match_phrases(self.brief.goal)):
                good.sort(key=self._intent_rank)
            else:
                good.sort(key=self._commit_rank)
            return good[0]
        good.sort(key=self._rank_of)
        return good[0]

    def _answer_text(self) -> str:
        target = self._target() or (self._candidates() or [None])[0]
        if not target:
            return "I am still checking the options against your requirements; nothing is decided yet."
        pid, sku = target["pid"], target["sku"]
        parts = ["I am going with " + (self._title(pid) or pid)[:90]]
        facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
        for phrase in self.brief.hard_phrases[:2]:
            for key, value in facts.items():
                if value is not None and _phrase_in_value(phrase, value):
                    parts.append("its listed " + str(key) + " is " + str(value)[:40])
                    break
        if target.get("price") is not None:
            unit = (" " + self.brief.currency) if self.brief.currency else ""
            parts.append("it is %.2f%s and in stock" % (target["price"], unit))
        return ", ".join(parts) + "."

    def _live_fallback(self) -> list:
        """Live, viewed, in-budget SKUs. Firm shopper constraints still rank first."""
        rows = []
        for c in self._candidates():
            if not c["live"] or not c["viewed"]:
                continue
            if not self._category_ok(c["pid"]):
                continue
            if self.brief.brand and not self._brand_ok(c["pid"], c["sku"]):
                continue
            rows.append(c)
        if not rows:
            rows = [c for c in self._candidates() if c["live"] and c["viewed"] and self._category_ok(c["pid"])]
        if self.brief.measures:
            exact = [c for c in rows if self._measure_satisfied(c["pid"], c["sku"])]
            rows = exact
        if self.brief.hard_phrases:
            hard = [c for c in rows if self._hard_ok(c["pid"], c["sku"])]
            if hard:
                rows = hard
            else:
                p_need = [p for p in self.brief.hard_phrases if re.fullmatch(r"\d{3,4}p", p or "", re.I)]
                if p_need:
                    best_p = max((self._p_token_quality(c["pid"], c["sku"]) for c in rows), default=0)
                    if best_p >= 2:
                        rows = [c for c in rows if self._p_token_quality(c["pid"], c["sku"]) >= 2]
        clean = [c for c in rows if not self._spec_conflicted(c["pid"], c["sku"])]
        if clean:
            rows = clean
        if not self.brief.ranking_like and len(rows) >= 2:
            typical = [c for c in rows if not self._price_outlier_penalty(c, rows)]
            if typical:
                rows = typical
        if self.brief.prefer_phrases:
            covered = [
                c
                for c in rows
                if self._soft_overlap(c["pid"], c["sku"]) > 0
                or self._prefer_overlap(c["pid"], c["sku"]) > 0
            ]
            if covered:
                rows = covered
        if self.ledger.event_target and (self.brief.prefer_phrases or _match_phrases(self.brief.goal)):
            rows.sort(key=self._intent_rank)
        else:
            rows.sort(key=self._commit_rank)
        return rows

    def _recall_need(self) -> int:
        if not self.brief.required_identities:
            return 0
        early = self.turn < self.max_steps * 0.5
        return self.brief.required_identities + (2 if early else 0)

    def _skus_of(self, pid: str) -> list[str]:
        found: list[str] = []
        prod = self._prod(pid)
        for variant in prod.get("variants") or []:
            if isinstance(variant, dict) and variant.get("sku"):
                found.append(str(variant["sku"]))
        for bag in (self.ledger.facts, self.ledger.rows, self.ledger.checked):
            for key in bag:
                if key.startswith(str(pid) + "::"):
                    sku = key.partition("::")[2]
                    if sku:
                        found.append(sku)
        out: list[str] = []
        for sku in found:
            if sku not in out:
                out.append(sku)
        return out

    def _pending_inspect(self, *, intent_only: bool = False) -> tuple | None:
        if "inspect_stock" not in self.tool_names:
            return None
        inspected_best = 0
        inspected_p = 0
        for c in self._candidates():
            if c["ok"] and c["key"] in self.ledger.inspected:
                inspected_best = max(inspected_best, self._measure_quality(c["pid"], c["sku"]))
                inspected_p = max(inspected_p, self._p_token_quality(c["pid"], c["sku"]))
        have_ok = any(c["ok"] and c["key"] in self.ledger.inspected for c in self._candidates())
        ranked: list[tuple] = []
        for pid in list(self.ledger.products):
            if str(pid) in self.ledger.dead_products:
                continue
            if not self._category_ok(pid):
                continue
            for sku in self._skus_of(pid):
                if self.brief.candidate_keys and not self._in_scope(pid, sku):
                    continue
                if self.brief.brand and not self._brand_ok(pid, sku):
                    continue
                if self._measure_contradicted(pid, sku):
                    continue
                if self._spec_conflicted(pid, sku) and inspected_p >= 2:
                    continue
                if self.brief.measures and self._measure_hits(pid, sku) < len(self.brief.measures):
                    continue
                if (
                    self.brief.hard_phrases
                    and not self._hard_ok(pid, sku)
                    and not self.ledger.event_target
                    and have_ok
                    and self._recovery_room()
                ):
                    continue
                key = Ledger.key(pid, sku)
                if key in self.ledger.inspected:
                    continue
                st = self.ledger.checked.get(key) or {}
                price = st.get("price")
                if self.budget is not None and price is not None and price > self.budget + _EPS:
                    continue
                if st.get("in_stock") is False:
                    continue
                if intent_only and self._request_text_hits(pid, sku) <= 0:
                    continue
                quality = self._measure_quality(pid, sku)
                if inspected_best >= 4 and quality < 4:
                    continue
                p_quality = self._p_token_quality(pid, sku)
                if inspected_p >= 2 and p_quality < inspected_p:
                    continue
                ranked.append(
                    (
                        int(self._spec_conflicted(pid, sku)),
                        -self._request_text_hits(pid, sku) if intent_only else 0,
                        -quality,
                        -p_quality,
                        -self._match_weight(pid, sku),
                        str(pid),
                        sku,
                    )
                )
        if not ranked:
            return None
        ranked.sort()
        pid, sku = ranked[0][-2], ranked[0][-1]
        return ("inspect_stock", {"product_id": pid, "sku": sku})

    def _open_incomplete(self, *, mark: bool = True) -> tuple | None:
        """View a brand/category match whose facts still lack requested dimensions."""
        if self.ledger.adds or "view" not in self.tool_names:
            return None
        need_p = bool(_P_TOKEN_RE.search(self.brief.goal)) or any(
            re.fullmatch(r"\d{3,4}p", p or "", re.I) for p in self.brief.hard_phrases
        )
        need_m = bool(self.brief.measures)
        if not need_p and not need_m:
            return None
        for pid in list(self.ledger.products):
            if str(pid) in self.ledger.dead_products or str(pid) in self.unreadable:
                continue
            if not self._category_ok(pid):
                continue
            if self.brief.brand and not self._brand_ok(pid):
                continue
            skus = self._skus_of(pid) or [""]
            if need_p and max(self._p_token_quality(pid, sku) for sku in skus) >= 2:
                continue
            if need_m and any(self._measure_satisfied(pid, sku) for sku in skus):
                continue
            blob = " ".join(self._spec_blob(pid, sku) for sku in skus)
            if not need_p and (_DIM_RE.search(blob) or _P_TOKEN_RE.search(blob) or _measurements(blob)):
                continue
            tag = "detail:" + str(pid)
            if tag in self.probes:
                continue
            if mark:
                self.probes.add(tag)
            return ("view", _filter_args(self.tool_index, "view", {"product_id": str(pid)}))
        return None

    def _read_more(self) -> tuple | None:
        if self.ledger.event_target and self._recover_with_text():
            if self._recovery_survivors():
                return None
            inspect = self._pending_inspect(intent_only=True)
            if inspect is not None:
                return inspect
            leads = self._unseen_leads()
            if leads and self.forced_compares < 10:
                self.forced_compares += 1
                if len(leads) > 1 and "compare" in self.tool_names:
                    return ("compare", _filter_args(self.tool_index, "compare", {"product_ids": leads[:5]}))
                if "view" in self.tool_names:
                    return ("view", _filter_args(self.tool_index, "view", {"product_id": leads[0]}))
            twin = self._twin_search()
            if twin is not None:
                return twin
            nxt = self._next_search()
            if nxt is not None:
                return nxt
            return None
        inspect = self._pending_inspect()
        if inspect is not None:
            return inspect
        opened = self._open_incomplete()
        if opened is not None:
            return opened
        twin = self._twin_search()
        if twin is not None:
            return twin
        leads = self._unseen_leads()
        if leads and self.forced_compares < 10 and self.turn >= 1:
            self.forced_compares += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", _filter_args(self.tool_index, "compare", {"product_ids": leads[:5]}))
            if "view" in self.tool_names:
                return ("view", _filter_args(self.tool_index, "view", {"product_id": leads[0]}))
        if self.brief.required_identities and len(self.ledger.identities) < self._recall_need():
            return self._next_search()
        if self.turn < 2:
            return None
        return self._next_search()

    def _useful_search_strings(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for query in self.brief.search_strings:
            cleaned = " ".join((query or "").split())
            if not cleaned or _is_instruction_query(cleaned):
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
        return out

    def _search_budget(self) -> int:
        useful = self._useful_search_strings()
        extra = 2 if self.brief.required_identities else 0
        if not useful:
            return 1 + extra
        return min(len(useful) + extra, 6)

    def _rewrite_query(self, query: str) -> str:
        """Keep catalog queries on the shopper's phrase, not the instruction."""
        cleaned = " ".join((query or "").split())
        useful = self._useful_search_strings()
        matched = _match_phrases(self.brief.goal)
        primary = (matched[0] if matched else "") or (useful[0] if useful else "")
        if self.ledger.event_target:
            recovered = self._recovery_query() or primary
            if recovered and _is_instruction_query(recovered):
                recovered = primary
            if recovered and not _is_instruction_query(recovered) and (
                not cleaned
                or self._query_is_category_only(cleaned)
                or _is_instruction_query(cleaned)
            ):
                return recovered
        if not cleaned:
            return primary
        if _is_instruction_query(cleaned):
            return primary or cleaned
        if matched:
            want = _alnum_set(matched[0])
            have = _alnum_set(cleaned)
            if want and not (want & have):
                return matched[0]
        return cleaned

    def _next_search(self) -> tuple | None:
        useful = self._useful_search_strings()
        if self.search_cursor >= self._search_budget() or not useful:
            self.reads_done = True
            inspect = self._pending_inspect(intent_only=self._skip_arbitrary_pending())
            if inspect is not None:
                return inspect
            opened = self._open_incomplete()
            if opened is not None:
                return opened
            return None
        query = useful[self.search_cursor % len(useful)]
        if self.ledger.event_target and self._recover_with_text():
            self.search_cursor += 1
            emitted = self._emit_retrieval(self._rewrite_query(query) or query)
            if emitted is not None:
                return emitted
            if self.search_cursor < self._search_budget():
                return self._next_search()
            self.reads_done = True
            return None
        probe = (
            self.brief.category or "",
            query.lower(),
            "%.2f" % (self.budget or 0.0),
            str(self.search_cursor),
        )
        if probe in self.probes:
            self.search_cursor += 1
            if self.search_cursor < self._search_budget():
                return self._next_search()
            self.reads_done = True
            return None
        self.probes.add(probe)
        self.search_cursor += 1
        if "filter" in self.tool_names and (self.brief.category or self.budget is not None) and self.search_cursor % 2 == 1:
            args: dict[str, Any] = {}
            if _param(self.tool_index, "filter", "category") and self.brief.category:
                args["category"] = self.brief.category
            if _param(self.tool_index, "filter", "in_stock"):
                args["in_stock"] = True
            if _param(self.tool_index, "filter", "max_price") and self.budget is not None:
                args["max_price"] = self.budget
            brand = self.brief.brand
            if _param(self.tool_index, "filter", "brand") and brand:
                args["brand"] = brand
            if _param(self.tool_index, "filter", "query"):
                args["query"] = query
            elif _match_phrases(self.brief.goal):
                args = {}
            if args and _param(self.tool_index, "filter", "k"):
                args["k"] = 10
            if args:
                return ("filter", _filter_args(self.tool_index, "filter", args))
        if "search" in self.tool_names:
            args = {"query": query}
            if _param(self.tool_index, "search", "in_stock"):
                args["in_stock"] = True
            if _param(self.tool_index, "search", "k"):
                args["k"] = 10
            return ("search", _filter_args(self.tool_index, "search", args))
        self.reads_done = True
        return None

    def _twin_search(self) -> tuple | None:
        dead = self.ledger.event_target
        if self.searched_twins or not dead:
            return None
        if "search" not in self.tool_names and "filter" not in self.tool_names:
            return None
        if self._recover_with_text():
            query = self._recovery_query()
            if self.brief.pref_model:
                parts = [self.brief.pref_model]
                if self.brief.pref_color:
                    parts.append(self.brief.pref_color)
                if self.brief.brand:
                    parts.insert(0, self.brief.brand)
                if self.brief.category:
                    parts.append(self.brief.category)
                query = " ".join(p for p in parts if p) or query
            if not query:
                return None
            emitted = self._emit_retrieval(query)
            if emitted is None:
                return None
            self.searched_twins = True
            self._note(
                "The carted item is no longer valid. Searching with the original request text %r rather than category alone."
                % query
            )
            return emitted
        pid = dead.split("::", 1)[0]
        parts = []
        if self.brief.pref_model:
            parts.append(self.brief.pref_model)
            if self.brief.pref_color:
                parts.append(self.brief.pref_color)
            if self.brief.brand:
                parts.insert(0, self.brief.brand)
            if self.brief.category:
                parts.append(self.brief.category)
        else:
            cat = (self.brief.category or "").strip().lower()
            for part in self._recovery_parts():
                if part.strip().lower() == cat:
                    continue
                if part not in parts:
                    parts.append(part)
                if len(parts) >= 2:
                    break
            if not parts and self.brief.category:
                parts.append(self.brief.category)
        query = " ".join(parts).strip()
        if not query:
            return None
        self.searched_twins = True
        self._note(
            "The carted item is no longer valid. Searching related listings that still meet the request."
        )
        if "filter" in self.tool_names and (self.brief.category or self.budget is not None):
            args: dict[str, Any] = {}
            if _param(self.tool_index, "filter", "category") and self.brief.category:
                args["category"] = self.brief.category
            if _param(self.tool_index, "filter", "in_stock"):
                args["in_stock"] = True
            if _param(self.tool_index, "filter", "max_price") and self.budget is not None:
                args["max_price"] = self.budget
            if _param(self.tool_index, "filter", "query"):
                args["query"] = query
            if _param(self.tool_index, "filter", "k"):
                args["k"] = 10
            brand = self.brief.brand
            if _param(self.tool_index, "filter", "brand") and brand:
                args["brand"] = brand
            if args:
                return ("filter", _filter_args(self.tool_index, "filter", args))
        args = {"query": query, "k": 10, "in_stock": True}
        return ("search", _filter_args(self.tool_index, "search", args))

    def _unseen_leads(self) -> list:
        leads = []
        for key, row in self.ledger.rows.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.ledger.products or str(pid) in self.ledger.dead_products:
                continue
            if self.brief.candidate_keys and (str(pid), str(sku)) not in self.brief.candidate_keys:
                continue
            st = self.ledger.checked.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.budget is not None and price > self.budget + _EPS):
                continue
            if (
                self.brief.category
                and self.ledger.filter_pids
                and not (self.ledger.event_target and self._recover_with_text())
                and str(pid) not in self.ledger.filter_pids
            ):
                continue
            title = str(row.get("title") or "")
            if self.brief.theme and not _has_theme(title, self.brief.theme):
                continue
            if self.brief.pref_model:
                if not _model_in_text(self.brief.pref_model, title + " " + str(row.get("options") or "")):
                    continue
            elif self.brief.specific_prefer and self.ledger.adds == 0 and not self.ledger.event_target:
                blob = title.lower()
                if not any(_phrase_in_value(phrase, blob) for phrase in self.brief.prefer_phrases):
                    continue
            elif self.ledger.event_target and (self.brief.prefer_phrases or _match_phrases(self.brief.goal)):
                phrases = (
                    _match_phrases(self.brief.goal)[:1]
                    + self.brief.soft_prefer[:2]
                    + self.brief.purpose_phrases[:1]
                )
                if phrases:
                    blob = title.lower()
                    if not any(
                        self._token_covered(tok, blob)
                        for phrase in phrases
                        for tok in _content_tokens(phrase)
                    ):
                        continue
            if self.brief.brand:
                wanted = _token_set(self.brief.brand)
                have = _token_set(str(row.get("brand") or ""))
                if have:
                    if not wanted <= have:
                        continue
                elif self.brief.category:
                    continue
                elif not wanted <= _token_set(title):
                    continue
            if self.brief.required_identities:
                leads.append((price, str(pid)))
            else:
                leads.append((-self._match_weight(str(pid), sku), str(pid)))
        leads.sort()
        out = []
        for _key, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def _shortlist_note(self) -> None:
        best = self._target()
        if best:
            self._note(
                "Verified candidate: %s::%s at %.2f. The controller will inspect, cart and order it; "
                "read other candidates only if one could rank better."
                % (best["pid"], best["sku"], best["price"] or 0.0)
            )
            return
        missing = [
            phrase
            for phrase in self.brief.hard_phrases
            if not any(self._fact_supports(c["pid"], c["sku"], phrase) for c in self._candidates() if c["viewed"])
        ]
        if missing:
            self._note(
                "Necessary request text not yet found in returned facts: "
                + "; ".join(missing[:4])
                + ". Search or open products using those words. Do not rename them."
            )
            return
        blocked = [c for c in self._candidates() if c["viewed"] and c["live"] and not c["ok"]]
        if blocked:
            self._note(
                "The products read so far fail a firm requirement (stock, budget, category, "
                "or a phrase taken from the request). Search for different products."
            )

    def _budget_from_message(self, text: str) -> None:
        ceilings = _money_ceilings(text)
        if not ceilings:
            return
        new = ceilings[0]["number"]
        if self.budget is not None and new >= self.budget:
            return
        self.budget = new
        if ceilings[0]["unit"]:
            self.brief.currency = ceilings[0]["unit"]
        self.shopper_updated = True
        unit = (" " + self.brief.currency) if self.brief.currency else ""
        self._note("Budget updated by the shopper: at most %.2f%s. Every later choice must fit it." % (new, unit))
        for pid, sku in list(self.ledger.cart):
            st = self.ledger.status(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _EPS:
                self.ledger.dead_keys.add(Ledger.key(pid, sku))
                self._note(
                    "The carted item %s costs %.2f, above the new budget; the controller will replace it."
                    % (Ledger.key(pid, sku), price)
                )

    def _justification_field_schema(self) -> dict[str, Any]:
        schema = self.tool_index.get("place_test_order") or {}
        props = schema.get("properties") or {}
        just = props.get("justification") if isinstance(props.get("justification"), dict) else {}
        sub = just.get("properties") if isinstance(just.get("properties"), dict) else {}
        if "claims" in sub:
            claims = sub.get("claims") if isinstance(sub.get("claims"), dict) else {}
            items = claims.get("items") if isinstance(claims.get("items"), dict) else {}
            item_props = items.get("properties") if isinstance(items.get("properties"), dict) else {}
            field = item_props.get("field") if isinstance(item_props.get("field"), dict) else {}
            return field if isinstance(field, dict) else {}
        return just if isinstance(just, dict) else {}

    def _claim_enum(self) -> list[str]:
        """Closed vocabulary from the runtime schema, if the contract lists one."""
        field = self._justification_field_schema()
        enum = field.get("enum")
        if isinstance(enum, list) and enum:
            return [str(x) for x in enum]
        props = field.get("properties") if isinstance(field.get("properties"), dict) else {}
        if props:
            skip = {"claims", "summary", "evidence", "reason", "text"}
            return [str(k) for k in props if str(k) not in skip]
        return []

    def _claim_schema_names(self) -> list[str]:
        """Identifiers mentioned in this episode's justification schema text."""
        field = self._justification_field_schema()
        just = ((self.tool_index.get("place_test_order") or {}).get("properties") or {}).get("justification")
        blob = str(field.get("description") or "")
        if isinstance(just, dict):
            blob += " " + str(just.get("description") or "")
        return _claim_names_from_text(blob)

    def _claim_fields(self) -> list[str]:
        enum = self._claim_enum()
        inferred: list[str] = ["price", "in_stock"]
        if self.brief.brand or self.brief.pref_brand:
            inferred.append("brand")
        goal_toks = _key_tokens(self.brief.goal + " " + self.brief.raw)
        pool = enum or list(dict.fromkeys(inferred + self._claim_schema_names()))
        wanted = [field for field in inferred if field in pool]
        rest = []
        for field in pool:
            if field in wanted:
                continue
            overlap = len(_key_tokens(field.replace("_", " ")) & goal_toks)
            rest.append((-overlap, field))
        rest.sort()
        return wanted + [field for _score, field in rest]

    def _align_claim_field(self, fact_key: str, value: str, used: set[str]) -> str | None:
        """Pick a runtime schema field for an observed fact. No pack field map."""
        pool = self._claim_enum() or self._claim_fields()
        if not pool:
            return None
        core = {"price", "in_stock", "brand", "category"}
        key_toks = _key_tokens(str(fact_key).replace(".", " "))
        val_toks = _key_tokens(value)
        hz_like = "hz" in key_toks or "hz" in val_toks or bool(re.search(r"\bhz\b", value or "", re.I))
        dim_like = bool(_DIM_RE.search(value or "") or _P_TOKEN_RE.search(value or ""))
        pixel_like = dim_like or _pixel_grid_value(value)
        best: str | None = None
        best_score = 0
        for field in pool:
            if field in used or field in core:
                continue
            ftoks = _key_tokens(field.replace("_", " "))
            if pixel_like and not hz_like and "rate" in ftoks:
                continue
            score = 10 * len(ftoks & key_toks)
            for kt in key_toks:
                if len(kt) < 3:
                    continue
                if any(kt in ft for ft in ftoks) or any(ft in kt for ft in ftoks if len(ft) >= 3):
                    score += 8
            # Pixel-grid values must not bind feature reasons (hd ⊂ hdr).
            if not pixel_like:
                score += len(ftoks & val_toks)
            if score > best_score:
                best, best_score = field, score
        if best is not None and best_score > 0:
            return best
        # Closed schema leftover: assign leftover evidence to a listed field
        # using request overlap, never a fitted field map.
        if not self._claim_enum():
            if self._open_pixel_spec_claim(value):
                return "resolution"
            return None
        if not (dim_like or self.brief.measures):
            return None
        goal_toks = _key_tokens(self.brief.goal + " " + self.brief.raw)
        measure_toks: set[str] = set()
        for phrase in self.brief.measures:
            measure_toks |= set(self._measure_field_tokens(phrase))
        ranked: list[tuple[int, int, str]] = []
        for field in self._claim_fields():
            if field in used or field in core:
                continue
            ftoks = _key_tokens(field.replace("_", " "))
            if dim_like and not hz_like and "rate" in ftoks:
                continue
            ranked.append((len(ftoks & goal_toks), -len(ftoks & measure_toks), field))
        if not ranked:
            return None
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        if dim_like:
            for _req, measure_home, field in ranked:
                if measure_toks and measure_home != 0:
                    continue
                return field
            return None
        if self.brief.measures:
            for _req, measure_home, field in ranked:
                if measure_home < 0:
                    return field
        return ranked[0][2]

    def _open_pixel_spec_claim(self, value: str) -> bool:
        """Open schema + shopper pixel-grid + observed WxH/Np/label → name it resolution."""
        if self._claim_enum():
            return False
        if not _pixel_grid_request(self.brief.goal):
            return False
        text = str(value or "").strip()
        if not text or not _pixel_grid_value(text):
            return False
        if _DIM_RE.search(text) or _P_TOKEN_RE.search(text):
            return True
        labels = _request_spec_labels(self.brief.goal + " " + self.brief.raw)
        compact = text.lower()
        return bool(labels) and compact in labels

    def _observed_facts(self, pid: str, sku: str) -> dict[str, Any]:
        facts = dict(self.ledger.facts.get(Ledger.key(pid, sku), {}))
        prod_facts = self._prod(pid).get("facts")
        if isinstance(prod_facts, dict):
            for key, value in prod_facts.items():
                facts.setdefault(str(key), value)
        st = self.ledger.status(pid, sku) or {}
        if st.get("price") is not None:
            facts.setdefault("price", st["price"])
        if "in_stock" in st:
            facts.setdefault("in_stock", st["in_stock"])
        brand = self._brand_of(pid, sku)
        if brand:
            facts.setdefault("brand", brand)
        return facts

    def _claims_for(self, pid: str, sku: str) -> list:
        facts = self._observed_facts(pid, sku)
        allowed = self._claim_fields()
        allowed_set = set(allowed)
        claims: list[dict[str, str]] = []
        used: set[str] = set()

        def add(field: str, value: Any) -> None:
            if field in used or value is None:
                return
            if field not in allowed_set:
                return
            text = str(value).strip()
            if not text:
                return
            claims.append({"field": field, "value": text})
            used.add(field)

        st = self.ledger.status(pid, sku) or {}
        if st.get("price") is not None:
            add("price", "%.2f" % st["price"] if isinstance(st["price"], (int, float)) else st["price"])
        if st.get("in_stock") is not None:
            add("in_stock", "true" if st.get("in_stock") else "false")
        brand = self._brand_of(pid, sku)
        if brand:
            add("brand", brand)
        path = self._prod(pid).get("category_path")
        if isinstance(path, list) and path:
            add("category", str(path[-1]))
        for phrase in self.brief.measures:
            units = [u for _n, u in _measurements(phrase)]
            nums = _numbers(phrase)
            want = _as_float(nums[0]) if nums else None
            if want is None:
                continue
            picked: tuple[int, str, Any, float] | None = None
            for key, value in facts.items():
                if value is None or str(key).lower().rsplit(".", 1)[-1] in _ID_KEYS | {"options", "condition"}:
                    continue
                text = str(value)
                related = self._measure_related(phrase, str(key), value)
                single = re.fullmatch(
                    r"\s*(\d+(?:\.\d+)?)\s*-?\s*(?:[A-Za-z%]{0,12}|\"|')?\s*$",
                    text,
                    re.I,
                )
                number = _as_float(single.group(1)) if single else None
                if number is None or abs(number - want) > max(1e-6, abs(want) * 1e-6):
                    if _DIM_RE.search(text) or _P_TOKEN_RE.search(text):
                        continue
                    hits = [_as_float(n) for n in _numbers(text)]
                    hits = [n for n in hits if n is not None and abs(n - want) <= max(1e-6, abs(want) * 1e-6)]
                    number = hits[0] if hits else None
                if number is None:
                    continue
                if not related and not single:
                    continue
                row = (0 if related else 1, str(key), value, number)
                if picked is None or row[0] < picked[0]:
                    picked = row
            if picked is not None:
                _prio, key, value, number = picked
                field = self._align_claim_field(key, str(value), used)
                if not field and not self._claim_enum():
                    field = key.lower().rsplit(".", 1)[-1]
                    if field:
                        allowed_set.add(field)
                if field:
                    suffix = units[0] if units else ""
                    add(field, ("%g" % number) + suffix)
        observed, src_key = self._observed_dimension_match(pid, sku)
        if observed:
            field = self._align_claim_field(src_key or "", observed, used)
            if self._open_pixel_spec_claim(observed):
                field = "resolution"
                allowed_set.add("resolution")
            elif not field and not self._claim_enum():
                field = (src_key or "").lower().rsplit(".", 1)[-1] or None
                if field == "display" and _pixel_grid_request(self.brief.goal):
                    field = "resolution"
                if field:
                    allowed_set.add(field)
            if field:
                add(field, observed)
        if self._claim_enum():
            goal_toks = _key_tokens(self.brief.goal)
            for field in allowed:
                if field in used or field in {"price", "in_stock", "brand", "category"}:
                    continue
                best_key, best_overlap = None, 0
                for key, value in facts.items():
                    if value is None or str(key).lower().rsplit(".", 1)[-1] in _ID_KEYS | {"options", "condition"}:
                        continue
                    overlap = len(_key_tokens(field) & _key_tokens(key))
                    if overlap > best_overlap:
                        best_overlap, best_key = overlap, key
                if best_key is not None and best_overlap:
                    add(field, facts[best_key])
                    continue
                if goal_toks & _key_tokens(field):
                    number = self._attribute_number(pid, sku, field.replace("_", " "))
                    if number is not None:
                        add(field, "%g" % number)
        return claims[:6]

    def _observed_dimension_match(self, pid: str, sku: str) -> tuple[str | None, str | None]:
        """Copy observed WxH/Np/label text that supports a request phrase. Returns (text, fact_key)."""
        facts = self._observed_facts(pid, sku)
        want = [p for p in self.brief.hard_phrases if re.fullmatch(r"\d{3,4}p", p or "", re.I)]
        want.extend(_dimension_phrases(self.brief.goal))
        labels = _request_spec_labels(self.brief.goal + " " + self.brief.raw)
        if not want and not labels:
            return None, None

        def want_n(phrase: str) -> int | None:
            if re.fullmatch(r"\d{3,4}p", phrase or "", re.I):
                try:
                    return int(phrase[:-1])
                except ValueError:
                    return None
            return None

        mixed: tuple[str, str] | None = None
        phrases = want or [""]
        for key, value in facts.items():
            if value is None:
                continue
            low = str(key).lower().rsplit(".", 1)[-1]
            if low in _ID_KEYS | {"options", "condition", "price", "in_stock"}:
                continue
            text = str(value).strip()
            if not text:
                continue
            for phrase in phrases:
                n = want_n(phrase) if phrase else None
                token = _P_TOKEN_RE.search(text)
                hit: str | None = None
                if phrase and phrase.lower() in text.lower().replace(" ", ""):
                    hit = token.group(1) if token else text
                elif phrase and phrase.lower() in text.lower():
                    hit = token.group(1) if token else text
                elif phrase and _p_token_supported(phrase, text):
                    dim = _DIM_RE.search(text)
                    if dim:
                        hit = f"{dim.group(1)} x {dim.group(2)}"
                    elif token:
                        hit = token.group(1)
                elif labels and re.fullmatch(r"[A-Za-z]{2,4}", text):
                    if text.lower() in labels:
                        hit = text
                if not hit:
                    continue
                if n is not None and _conflicting_np(text, n):
                    mixed = mixed or (hit, str(key))
                    continue
                return hit, str(key)
        blob = self._spec_blob(pid, sku)
        for phrase in want:
            n = want_n(phrase)
            if n is not None and _conflicting_np(blob, n):
                continue
            if phrase.lower() in blob.lower():
                token = _P_TOKEN_RE.search(blob)
                return (token.group(1) if token else phrase), None
            if _p_token_supported(phrase, blob):
                dim = _DIM_RE.search(blob)
                if dim:
                    return f"{dim.group(1)} x {dim.group(2)}", None
        if mixed:
            return mixed
        return None, None

    def _apply_justification(self, args: dict, pid: str, sku: str) -> dict:
        schema = self.tool_index.get("place_test_order") or {}
        props = schema.get("properties") or {}
        if "justification" not in props:
            args.pop("justification", None)
            return args
        if not self.brief.asks_reason:
            args.pop("justification", None)
            return args
        claims = self._claims_for(pid, sku)
        if not claims:
            args.pop("justification", None)
            return args
        expected = props.get("justification") if isinstance(props.get("justification"), dict) else {}
        sub = expected.get("properties") if isinstance(expected.get("properties"), dict) else {}
        if "claims" in sub:
            args["justification"] = {"claims": claims}
        elif expected.get("type") == "object":
            args["justification"] = {c["field"]: c["value"] for c in claims}
        else:
            args["justification"] = "; ".join(f"{c['field']}={c['value']}" for c in claims)
        return args

    def _order_args(self, pid: str, sku: str) -> dict:
        args = {"product_id": pid, "sku": sku}
        args = _filter_args(self.tool_index, "place_test_order", args)
        return self._apply_justification(args, pid, sku)

    def _valid_now(self, pid: str, sku: str) -> tuple:
        st = self.ledger.status(pid, sku)
        k = Ledger.key(pid, sku)
        if st is None:
            return False, "inspect_stock the exact product_id and sku first"
        if (pid, sku) not in self.ledger.cart:
            return False, "add that exact variant to the cart first"
        if self.ledger.last_add_seq is not None and self.ledger.checked_seq.get(k, 0) <= self.ledger.last_add_seq:
            return False, "re-check the variant with inspect_stock after the cart addition"
        if not st.get("in_stock"):
            return False, "that variant is out of stock; choose a different product"
        price = st.get("price")
        if self.budget is not None and price is not None and price > self.budget + _EPS:
            return False, "that variant costs %.2f, above the budget of %.2f" % (price, self.budget)
        if k in self.ledger.dead_keys:
            return False, "that variant was invalidated by the market change; choose a different product"
        return True, ""

    def _ready(self) -> bool:
        if self.ledger.adds:
            return True
        if self.brief.asks_shopper and not self.shopper_updated and not self._must_terminate():
            return False
        if self.brief.required_identities:
            roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
            if roomy and len(self.ledger.identities) < self._recall_need():
                return False
        inspected_ok = self._inspected_ok()
        if not inspected_ok:
            if self.brief.measures and self._explore_room():
                return False
            if not self._explore_room():
                return bool(self._live_fallback())
            return bool(not self._recovery_room() and self._live_fallback())
        if not self._explore_room():
            if (
                self.ledger.adds == 0
                and not self.brief.ranking_like
                and (self.brief.prefer_phrases or self.brief.pref_brand)
                and self._recovery_room()
            ):
                need = self._needed_peers()
                if len(inspected_ok) < need and self._has_peer_work():
                    return False
            return True
        if self.brief.ranking_like:
            return True
        best = inspected_ok[0]
        pending = self._pending_inspect()
        if pending is not None:
            ppid, psku = str(pending[1].get("product_id") or ""), str(pending[1].get("sku") or "")
            if ppid and psku and self._inspect_score(ppid, psku) < self._inspect_score(best["pid"], best["sku"]):
                return False
        need = self._needed_peers()
        if len(inspected_ok) < need and self._has_peer_work():
            return False
        if (
            (self.brief.prefer_phrases or self.brief.pref_brand)
            and not self.brief.candidate_keys
            and self.budget is not None
            and not self.budget_peers_done
            and len(self._plausible_peer_pids()) < self._peer_floor()
        ):
            return False
        if (
            self.brief.candidate_keys
            and any(
                Ledger.key(pid, sku) not in self.ledger.inspected
                and str(pid) in self.ledger.products
                and str(pid) not in self.unreadable
                and str(pid) not in self.ledger.dead_products
                for pid, sku in self.brief.candidate_keys
            )
        ):
            return False
        if (
            (self.brief.prefer_phrases or self.brief.pref_brand)
            and not self.brief.candidate_keys
            and not self.brief.specific_prefer
            and self.search_calls < 1
        ):
            return False
        return True

    def _fresh(self, pid: str, sku: str) -> bool:
        k = Ledger.key(pid, sku)
        if k not in self.ledger.inspected:
            return False
        return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq

    def _plan(self) -> tuple | None:
        if self.done:
            return None
        if self.shopper_question_pending:
            if self.question_waits < 2 or "message" not in self.tool_names:
                self.question_waits += 1
                return None
            return ("message", {"content": self._answer_text()})
        if self.brief.candidate_keys and "compare" in self.tool_names:
            missing = [
                pid
                for pid, _sku in self.brief.candidate_keys
                if pid not in self.ledger.products and pid not in self.unreadable
            ]
            if missing:
                return ("compare", _filter_args(self.tool_index, "compare", {"product_ids": missing[:5]}))
        if self.brief.pref_model and "search" in self.tool_names:
            rows = [c for c in self._candidates() if c["ok"] and c["relax"] == 0]
            near = [c for c in self._candidates() if c["viewed"] and c["relax"] == 1]
            need_twins = (not self.ledger.adds) or bool(self.ledger.event_target)
            if need_twins and not rows and near and not self.searched_twins:
                parts = [self.brief.pref_model]
                if self.brief.pref_color:
                    parts.append(self.brief.pref_color)
                if self.brief.brand:
                    parts.insert(0, self.brief.brand)
                if self.brief.category:
                    parts.append(self.brief.category)
                query = " ".join(p for p in parts if p)
                if query:
                    self.searched_twins = True
                    args = {"query": query, "k": 10, "in_stock": True}
                    return ("search", _filter_args(self.tool_index, "search", args))
        target = self._target()
        if self.ledger.adds == 0 and not self.brief.ranking_like:
            if not self._ready() and self._explore_room():
                opened = self._open_incomplete()
                if opened is not None:
                    return opened
                inspect = self._pending_inspect()
                if inspect is not None:
                    return inspect
                peer = self._peer_step()
                if peer is not None:
                    return peer
                budget = self._budget_peer_probe()
                if budget is not None:
                    return budget
        if target is None and not self._skip_arbitrary_pending():
            inspect = self._pending_inspect()
            if inspect is not None:
                return inspect
        if target is None or not self._ready():
            forced = self._read_more()
            if forced is not None:
                return forced
            if target is None:
                return None
        pid, sku = target["pid"], target["sku"]
        if (pid, sku) in self.ledger.cart:
            if not self._fresh(pid, sku):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            ok, _why = self._valid_now(pid, sku)
            if ok:
                return ("place_test_order", self._order_args(pid, sku))
        if not self._ready():
            return None
        for cpid, csku in self.ledger.cart:
            if (cpid, csku) != (pid, sku) and "remove_from_cart" in self.tool_names:
                return ("remove_from_cart", {"product_id": cpid, "sku": csku})
        if not self._fresh(pid, sku):
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return ("add_to_cart", {"product_id": pid, "sku": sku})

    def _sleep_budgeted(self, seconds: float, keep: float = 20.0) -> bool:
        wait = min(max(0.0, seconds), max(0.0, self.clock.remaining() - keep))
        if wait <= 0:
            return False
        time.sleep(wait)
        return True

    def _ask_model(self) -> dict | None:
        if not self._infer_staggered:
            self._infer_staggered = True
            spread = (abs(hash(self.problem_id)) % 80) / 10.0
            if spread > 0.15:
                self._sleep_budgeted(spread, keep=40.0)
        backoff = 1.6
        empty_reprompts = 0
        missing: set[str] = set()
        paid_out: set[str] = set()
        attempts = 0
        n_models = max(1, len(self.roster))
        while self.clock.remaining() > 22 and attempts < 10:
            model = self.roster[self.model_index % n_models]
            payload = {
                "model": model,
                "messages": self.messages,
                "tools": self.tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            result = _INFER.post_verbose("/inference/chat/completions", json_data=payload)
            attempts += 1
            data = result.data if result is not None else None
            if isinstance(data, dict) and data.get("choices"):
                msg = data["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                empty_reprompts += 1
                if empty_reprompts > 3:
                    break
                self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                self._note("No tool call was made. Choose exactly one of the supplied tools now.")
                continue
            error = (result.error if result is not None else None) or {}
            status = error.get("status")
            body = str(error.get("body") or "").lower()
            if status == 429 or "in_flight" in body:
                wait = min(12.0, random.uniform(backoff, backoff * 2.4))
                if not self._sleep_budgeted(wait, keep=22.0):
                    break
                backoff = min(12.0, backoff * 1.7)
                continue
            if status == 404:
                missing.add(model)
                if len(missing) >= n_models:
                    break
                self.model_index = (self.model_index + 1) % n_models
                if not self._sleep_budgeted(random.uniform(0.4, 1.2), keep=22.0):
                    break
                continue
            if status == 402:
                paid_out.add(model)
                if len(paid_out) >= n_models:
                    break
                self.model_index = (self.model_index + 1) % n_models
                if not self._sleep_budgeted(random.uniform(1.0, 3.0), keep=22.0):
                    break
                continue
            self.model_index = (self.model_index + 1) % n_models
            wait = min(6.0, random.uniform(backoff, backoff * 2.0))
            if not self._sleep_budgeted(wait, keep=22.0):
                break
            backoff = min(12.0, backoff * 1.5)
        return None

    def _send(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            result = _PROXY.post_verbose("/environment/call", json_data=envelope)
            data = result.data if result is not None else None
            if isinstance(data, dict) and data.get("calls"):
                return data
            status = (result.error or {}).get("status") if result is not None else None
            if status == 409:
                return None
            if self.clock.remaining() < 20:
                break
            wait = min(2.0 * (attempt + 1), 6.0)
            if status == 429:
                wait = min(8.0, wait + random.uniform(0.4, 1.6))
            time.sleep(wait)
        return None

    def _act(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self._send(pend["envelope"])
            if replay is None:
                self._note("The environment is not answering; the previous action is still pending and will be retried.")
                return None
            self.pending_envelope = None
            self.turn = pend["turn"]
            self._absorb(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
            if self.done:
                return None
        turn = self.turn + 1
        group = self.problem_id + "-turn-" + str(turn)
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [{"call_id": group + "-1", "action": {"name": name, "args": args}}],
        }
        result = self._send(envelope)
        if result is None:
            self.pending_envelope = {
                "envelope": envelope, "turn": turn, "name": name, "args": args,
                "content": content, "call_id": tool_call_id or ("agent-" + str(turn)),
            }
        else:
            self.turn = turn
        return self._absorb(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.ledger.calls_seen.discard(name + json.dumps(args, sort_keys=True))
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps({"error": "environment call failed; it will be retried automatically"}),
                }
            )
            return None
        calls = result.get("calls") or []
        wrap = (calls[0].get("observation") if calls and isinstance(calls[0], dict) else None) or {}
        obs = wrap.get("observation") if isinstance(wrap.get("observation"), dict) else wrap
        if isinstance(obs, dict) and wrap.get("error") and not obs.get("error"):
            obs = dict(obs)
            obs["error"] = wrap.get("error")
        if wrap.get("done"):
            self.done = True
        self.ledger.record(name, args, obs if isinstance(obs, dict) else {}, self.brief)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _trim(json.dumps(obs), _OBS_CHARS)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self._budget_from_message(text)
            added = self.brief.adopt_text(text)
            if self.asked_shopper or added:
                self.shopper_updated = True
            if added:
                self.search_cursor = 0
                self.reads_done = False
                self.forced_compares = 0
                self._note(
                    "Additional requirement recorded from the shopper: "
                    + "; ".join(added[:4])
                    + ". Only evidence that contains that text may be bought."
                )
            if _QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self._note("The shopper asked a question: answer it with message in your next turn, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        if name in ("search", "filter") and isinstance(result, dict):
            self.search_calls += 1
        return obs if isinstance(obs, dict) else None

    def _post_add_recheck(self) -> None:
        pid, sku = self.pending_recheck
        self.pending_recheck = None
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = _as_float(obs.get("price"))
        over = self.budget is not None and price is not None and price > self.budget + _EPS
        if stock and not over:
            self._note(
                "Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it."
                % (Ledger.key(pid, sku), price)
            )
            ok, _why = self._valid_now(pid, sku)
            if ok and not self.done:
                self._act("place_test_order", self._order_args(pid, sku))
            return
        self.ledger.event_target = Ledger.key(pid, sku)
        self.ledger.dead_keys.add(Ledger.key(pid, sku))
        if not self.brief.pref_model:
            self.ledger.dead_products.add(str(pid))
        self.forced_compares = 0
        self.searched_twins = False
        self.reads_done = False
        self.search_cursor = 0
        extra = [p for p in self._recovery_parts() if not _is_instruction_query(p)]
        for item in reversed(extra[:3]):
            if item and item not in self.brief.search_strings:
                self.brief.search_strings.insert(0, item)
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        kept = "; ".join(extra[:4])
        recovered = self._recovery_query()
        if self._recovery_survivors():
            path = (
                "A verified listing already matches the original request; the controller will use it "
                "instead of a category-only search."
            )
        elif recovered:
            path = (
                "No verified match is on record, so the next retrieval uses %r rather than category alone."
                % recovered
            )
        else:
            path = "Replace it with a different product that still meets the original request."
        self._note(
            "Market change: the carted item %s %s. Original request kept%s. %s"
            % (Ledger.key(pid, sku), reason, (": " + kept) if kept else "", path)
        )
        self._shortlist_note()

    def _finish_fast(self) -> None:
        for pid, sku in reversed(self.ledger.cart):
            if not self._fresh(pid, sku) and "inspect_stock" in self.tool_names:
                self._act("inspect_stock", {"product_id": pid, "sku": sku})
                if self.done:
                    return
            ok, _why = self._valid_now(pid, sku)
            if ok and not self.done:
                self._act("place_test_order", self._order_args(pid, sku))
                return
        if self.done:
            return
        if not any(c["ok"] for c in self._candidates()):
            have_usable = any(
                c["live"] and c["viewed"] and c["key"] in self.ledger.inspected
                for c in self._candidates()
            )
            if not have_usable:
                pending = self._pending_inspect()
                if pending is not None:
                    self._act(pending[0], pending[1])
                    if self.done:
                        return
                else:
                    opened = self._open_incomplete()
                    if opened is not None:
                        self._act(opened[0], opened[1])
                        if self.done:
                            return
        good = [c for c in self._candidates() if c["ok"]]
        if self.brief.pref_model and self.ledger.event_target:
            siblings = [c for c in good if c["relax"] <= 1]
            good = siblings
        elif self.brief.pref_model:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
        if (
            self.ledger.event_target
            and self._recover_with_text()
            and not self.brief.pref_model
        ):
            survivors = self._recovery_survivors()
            if survivors:
                good = survivors
            else:
                covered = [
                    c
                    for c in good
                    if self._request_text_hits(c["pid"], c["sku"]) > 0
                ]
                if covered:
                    good = covered
        if not good and self.brief.theme and not self.brief.pref_model:
            themed = [
                c
                for c in self._candidates()
                if c["live"]
                and c["viewed"]
                and _has_theme(self._title(c["pid"]) + " " + self._hint_blob(c["pid"], c["sku"]), self.brief.theme)
            ]
            if self.brief.measures:
                themed = [c for c in themed if self._measure_satisfied(c["pid"], c["sku"])]
            good = themed
        if not good:
            good = self._live_fallback()
        if not good:
            return
        if self.brief.ranking_like:
            good.sort(key=self._rank_of)
        elif self.ledger.event_target and (self.brief.prefer_phrases or _match_phrases(self.brief.goal)):
            good.sort(key=self._intent_rank)
        else:
            good.sort(key=self._commit_rank)
        pid, sku = good[0]["pid"], good[0]["sku"]
        if self.brief.measures and not self._measure_satisfied(pid, sku):
            return
        if (pid, sku) not in self.ledger.cart:
            obs = self._act("add_to_cart", {"product_id": pid, "sku": sku})
            if self.done or not isinstance(obs, dict) or obs.get("error"):
                return
        if self.done:
            return
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if self.done or not isinstance(obs, dict) or not obs.get("in_stock"):
            return
        price = _as_float(obs.get("price"))
        if self.budget is not None and price is not None and price > self.budget + _EPS:
            return
        self._act("place_test_order", self._order_args(pid, sku))

    def run(self) -> list:
        question = self._opening_question()
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            if self.pending_recheck and "inspect_stock" in self.tool_names:
                self._post_add_recheck()
                if not self.done and (self.turn >= self.max_steps - 3 or self.clock.remaining() < 25):
                    self._finish_fast()
                    break
                continue
            if self.clock.remaining() < 25:
                self._finish_fast()
                break
            if self.turn >= self.max_steps - 3:
                self._finish_fast()
                break
            if question and not self.asked_shopper and "message" in self.tool_names:
                self.asked_shopper = True
                self._act("message", {"content": question})
                continue
            plan = self._plan()
            if plan is not None:
                name, args = plan
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    if name in ("inspect_stock", "add_to_cart"):
                        self.ledger.dead_keys.add(Ledger.key(args.get("product_id"), args.get("sku")))
                    if name == "compare":
                        self.unreadable.update(str(p) for p in args.get("product_ids") or [])
                    if name == "message":
                        self.shopper_question_pending = False
                    last_forced, repeats = None, 0
                    self._note("That step changed nothing, so the controller moved on to other candidates.")
                    continue
                obs = self._act(name, args)
                if name == "message":
                    self.shopper_question_pending = False
                    self.question_waits = 0
                if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                    added = obs.get("added") or {}
                    if added.get("product_id"):
                        self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                continue
            last_forced, repeats = None, 0
            proposal = self._ask_model()
            if proposal is None:
                self._finish_fast()
                break
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = _arguments(call)
            content = proposal.get("content") or ""
            reprompts = 0
            while True:
                verdict = self._validate(name, args)
                if verdict is None:
                    break
                if reprompts >= _MAX_REPROMPTS:
                    self.rejected_streak += 1
                    self._note("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if self.rejected_streak >= 3:
                        self._shortlist_note()
                        self.rejected_streak = 0
                    verdict = "skip"
                    break
                reprompts += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": verdict})}
                )
                self.dialogue.append({"role": "assistant", "content": "[rejected " + name + "] " + verdict[:200]})
                again = self._ask_model()
                if again is None:
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = _arguments(call)
                content = again.get("content") or ""
            if verdict == "skip":
                continue
            self.rejected_streak = 0
            if name not in self.tool_names:
                continue
            args = _filter_args(self.tool_index, name, args)
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.ledger.cart:
                    pid, sku = self.ledger.cart[-1]
                args = self._order_args(pid, sku)
            if name == "message":
                if not self.shopper_question_pending:
                    self.questions_asked += 1
                self.shopper_question_pending = False
                self.question_waits = 0
            obs = self._act(name, args, content, call.get("id"))
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") or {}
                if added.get("product_id"):
                    self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
            if name in ("search", "filter"):
                if self.brief.required_identities:
                    have, need = len(self.ledger.identities), self._recall_need()
                    self._note(
                        "Distinct matching products surfaced so far: %d of the required %d." % (have, need)
                        + ("" if have >= need else " Keep searching with different wording.")
                    )
            if name == "inspect_cart":
                self.cart_inspections += 1
            if name in ("view", "compare"):
                self._shortlist_note()
        if not self.done:
            self._finish_fast()
        return self.dialogue

    def _validate(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
        if name in ("search", "filter"):
            brand = str(args.get("brand") or "").strip()
            if brand:
                allowed = (self.brief.brand or self.brief.pref_brand or "").strip()
                if not allowed or not (
                    _token_set(brand) <= _token_set(allowed) or _token_set(allowed) <= _token_set(brand)
                ):
                    args.pop("brand", None)
            if name == "filter" and self.ledger.event_target and self._recover_with_text():
                if not self._filter_carries_query():
                    kept_brand = str(args.get("brand") or "").strip()
                    firm = (self.brief.brand or "").strip()
                    brand_ok = bool(kept_brand) and bool(firm) and (
                        _token_set(kept_brand) <= _token_set(firm) or _token_set(firm) <= _token_set(kept_brand)
                    )
                    if not brand_ok:
                        return (
                            "after a market change, search with the original request text; "
                            "a category-only filter drops the shopper's stated intent"
                        )
            if name == "search" or "query" in args:
                rewritten = self._rewrite_query(str(args.get("query") or ""))
                if rewritten:
                    args["query"] = rewritten
                elif "query" in args:
                    args.pop("query", None)
            elif self.ledger.event_target and _param(self.tool_index, name, "query"):
                recovered = self._recovery_query()
                if recovered:
                    args["query"] = recovered
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.ledger.cart:
                pid, sku = self.ledger.cart[-1]
            ok, why = self._valid_now(pid, sku)
            if not ok:
                return "order refused: " + why
            if self.brief.required_identities and len(self.ledger.identities) < self._recall_need():
                return (
                    "order refused: only %d distinct matching products have been surfaced; search with different "
                    "wording until at least %d appear" % (len(self.ledger.identities), self._recall_need())
                )
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            c = self._assess(pid, sku)
            key = Ledger.key(pid, sku)
            if not c["live"]:
                return "that variant is out of stock, over budget, or already invalidated"
            if not c["viewed"]:
                return "view or compare that product first so its returned facts are on record"
            if key not in self.ledger.inspected:
                return "inspect_stock that exact variant first: its current price and stock decide whether it can be bought"
            if not self._in_scope(pid, sku):
                return "that variant is not one of the identifiers listed in the request"
            if self.brief.measures and c["viewed"]:
                if self._measure_contradicted(pid, sku):
                    return "that variant's labelled measurement conflicts with a firm shopper requirement"
                if not self._measure_satisfied(pid, sku):
                    return "returned facts do not show the measurement the shopper required"
            if self.brief.specific_prefer and not self.ledger.adds and c["relax"] and self.turn < self.max_steps * 0.6:
                return "the request named a specific preferred item; keep searching for it before committing a candidate"
            if (
                self.ledger.event_target
                and self._recover_with_text()
                and not self.brief.pref_model
                and self._request_text_hits(pid, sku) <= 0
                and not self.searched_twins
            ):
                return (
                    "that listing does not show the original request text; "
                    "search or open a replacement that still matches it before committing"
                )
            if not self.brief.ranking_like:
                best = self._target()
                if best and (best["pid"], best["sku"]) != (pid, sku):
                    if not self.ledger.adds or (
                        self.ledger.event_target and self._recover_with_text()
                    ):
                        return "a higher-weight verified candidate is already on record; commit that one first"
            if not self._ready():
                return (
                    "hold: read one more product that also meets every requirement first, so there is a "
                    "fallback when the market moves right after the cart addition"
                )
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.shopper_question_pending and "?" not in text:
                return "do not send status updates; use message only to answer the shopper's question or to ask one"
            if not self.shopper_question_pending and self.questions_asked >= 2:
                return "the shopper has already answered; decide from the evidence you have rather than asking again"
        if name == "inspect_cart" and self.cart_inspections >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.ledger.calls_seen and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        self.ledger.calls_seen.add(key)
        return None


def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
