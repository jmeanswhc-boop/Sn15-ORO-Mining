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
# Coverage-rank confidence. Used to inspect/commit high-weight candidates
# instead of the cheapest listing. Not a product or task lookup.
_LOG_PROBS = [
    0.6698174986042095, 0.12640294947204656, 0.03721208325578941, 0.01890137562198827,
    0.011222691775555538, 0.005316011893684202, 0.0035440079291228013, 0.003544006929121897,
    0.0028846576167278616, 0.002884656616726958, 0.0028846556167260538, 0.00288465461672515,
    0.002884653616724246, 0.0028846526167233422, 0.002884651616722438, 0.0028846506167215344,
    0.0028846496167206303, 0.0028846486167197266, 0.0028846476167188224, 0.0028846466167179187,
    0.0028846456167170146, 0.002884644616716111, 0.002884643616715207, 0.002884642616714303,
    0.002884641616713399, 0.0028846406167124953, 0.002884639616711591, 0.0028846386167106874,
    0.0028846376167097833, 0.0028846366167088796, 0.0028846356167079755, 0.0028846346167070718,
    0.0028846336167061676, 0.002884632616705264, 0.00288463161670436, 0.002884630616703456,
    0.002884629616702552, 0.0028846286167016483, 0.002884627616700744, 0.0028846266166998404,
    0.0028846256166989367, 0.0028846246166980326, 0.002884623616697129, 0.002884622616696225,
    0.002884621616695321, 0.002884620616694417, 0.0028846196166935133, 0.002884618616692609,
    0.0028846176166917054, 0.0028846166166908013, 0.0028846156166898976,
]

_VARIANT_RE = re.compile(r"\b(\d{6,})::([A-Za-z0-9._\-]+)\b")
_DIM_RE = re.compile(r"(\d{2,5})\s*[x×*]\s*(\d{2,5})")
_P_TOKEN_RE = re.compile(r"\b(\d{3,4}p)\b", re.I)
_CLASS_DEF_RE = re.compile(
    r"\b([A-Za-z][A-Za-z ]{0,20}?)\s+starts at\s+(\d{2,5})\s*(?:by|x|×)\s+(\d{2,5})",
    re.I,
)
_QUOTED_RE = re.compile(r"[\"“]([^\"”]{2,80})[\"”]")
_PHRASE_RE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
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
    r"\b(?:rather|i'd rather|would rather)\b.{0,40}?\bfor\s+([^,.;\n]{3,80})",
    re.I,
)
_PURPOSE_RE = re.compile(r"\bit'?s for\s+([^,.;\n]{3,80})", re.I)
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
    "using that text. A search title is a hint for what to open next, not a verified "
    "attribute. Open a product and read its returned facts before buying. Confirm live "
    "price and availability before adding to the cart or ordering. A deterministic "
    "controller checks every proposal and takes over the cart and the order once a "
    "candidate is fully verified; the first cart addition is graded, so never add a "
    "placeholder. Prices and stock change while you shop; the controller re-checks the "
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


def _class_defs(text: str) -> dict[str, tuple[int, int]]:
    """Pixel-class floors stated in this episode's request, if any."""
    found: dict[str, tuple[int, int]] = {}
    for match in _CLASS_DEF_RE.finditer(text or ""):
        name = " ".join(match.group(1).split()).lower()
        if name:
            found[name] = (int(match.group(2)), int(match.group(3)))
    return found


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
        if len(unit) == 1 and unit not in {'"', "%", "'"}:
            continue
        if unit in _FUNCTION_WORDS or unit in _META_UNITS:
            continue
        if unit in {'"', "''", "inche", "inch", "in"}:
            unit = "in"
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
            r"\b(?:brand|categor(?:y|ies)|catalog|budget|requirement|preference|priority|"
            r"explicitly|identify|device os|title or|walk me|justif|reason such|"
            r"matching product|distinct|threshold|search result|task rules)\b",
            phrase or "",
            re.I,
        )
    )


_PROTOCOL_ATTR_RE = re.compile(
    r"\b(?:reason|listing|identit|threshold|matching product|useful|"
    r"valid item|distinct|digit|class whose|advertised maximum|shopper|"
    r"justif|claim|tradeoff)\b",
    re.I,
)


def _hard_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for match in _QUOTED_RE.finditer(text or ""):
        phrase = " ".join(match.group(1).split())
        if phrase and phrase not in phrases:
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


def _prefer_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for match in _PREFER_RE.finditer(text or ""):
        phrase = " ".join(match.group(1).split()).strip(" .,:;")
        if phrase and phrase.lower() not in _FUNCTION_WORDS and phrase not in phrases:
            phrases.append(phrase)
    for match in _RATHER_RE.finditer(text or ""):
        phrase = " ".join(match.group(1).split()).strip(" .,:;")
        phrase = re.split(r"\b(?:than|rather than)\b", phrase, maxsplit=1, flags=re.I)[0].strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    for match in _PURPOSE_RE.finditer(text or ""):
        phrase = " ".join(match.group(1).split()).strip(" .,:;")
        if phrase and phrase not in phrases:
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
    """Capitalized words naming the item, e.g. 'a Samsung smartphone'."""
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
    primary = _primary_request(request)
    strings: list[str] = []
    ceilings = _money_ceilings(primary)
    budget = ceilings[0]["number"] if ceilings else None
    measures = _measure_phrases(primary, budget)
    for phrase in _dimension_phrases(primary) + _hard_phrases(primary) + _prefer_phrases(primary) + measures:
        if phrase not in strings:
            strings.append(phrase)
    cat = _category_phrase(primary)
    brand = _brand_before_category(primary, cat) or _prefer_brand(primary)
    if brand:
        strings.insert(0, brand)
    if cat:
        strings.insert(0, cat if not brand else f"{brand} {cat}")
    tokens = _content_tokens(primary)
    if tokens:
        strings.append(" ".join(tokens[:8]))
        if len(tokens) > 3:
            strings.append(" ".join(tokens[:4]))
        if len(tokens) > 6:
            strings.append(" ".join(tokens[2:8]))
    first_line = primary.split("\n", 1)[0].strip()
    if first_line and first_line not in strings:
        strings.append(first_line[:160])
    if not strings:
        strings.append((request or "")[:160])
    uniq: list[str] = []
    for item in strings:
        cleaned = " ".join(item.split())
        if cleaned and cleaned not in uniq:
            uniq.append(cleaned)
    return uniq


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


def _dimension_sides(text: str) -> set[int]:
    """Numeric sides from WxH pairs and Np tokens in observed text."""
    sides: set[int] = set()
    for left, right in _DIM_RE.findall(text or ""):
        try:
            sides.add(int(left))
            sides.add(int(right))
        except ValueError:
            continue
    for match in _P_TOKEN_RE.finditer(text or ""):
        try:
            sides.add(int(match.group(1)[:-1]))
        except ValueError:
            continue
    return sides


def _rank_prior(rank: int | None) -> float:
    """Decaying confidence for coverage/search rank. Unknown ranks sit after the table."""
    if rank is None or rank < 0:
        rank = len(_LOG_PROBS)
    if rank >= len(_LOG_PROBS):
        return _LOG_PROBS[-1] * (0.9 ** (rank - len(_LOG_PROBS) + 1))
    return _LOG_PROBS[rank]


def _p_token_supported(phrase: str, blob: str) -> bool:
    """720p matches 1280 x 720 because 720 is a recorded dimension, not via a class table."""
    if not re.fullmatch(r"\d{3,4}p", phrase or "", re.I):
        return False
    if phrase.lower() in (blob or "").lower():
        return True
    try:
        want = int(phrase[:-1])
    except ValueError:
        return False
    return want in _dimension_sides(blob)


def _observed_dimension_text(blob: str) -> str | None:
    """WxH or Np string copied from observed text. Never a request-side label."""
    match = _DIM_RE.search(blob or "")
    if match:
        return f"{match.group(1)} x {match.group(2)}"
    match = _P_TOKEN_RE.search(blob or "")
    if match:
        return match.group(1)
    return None


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
        self.class_defs = _class_defs(self.raw)
        phrase = _PHRASE_RE.search(self.goal)
        quoted = _QUOTED_RE.findall(self.goal)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
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
        for phrase in _prefer_phrases(text):
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
        extra = _content_tokens(text)
        if extra:
            blob = " ".join(extra[:8])
            if blob and blob not in self.search_strings:
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
            for key, value in facts.items():
                low = str(key).lower().rsplit(".", 1)[-1]
                if low in _ID_KEYS | {"price", "in_stock", "currency"}:
                    continue
                if _p_token_supported(phrase, str(value or "")):
                    return True
            return _p_token_supported(phrase, self._options(pid, sku))
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
        """Literal Np outranks a WxH that merely contains the same number."""
        phrases = [p for p in self.brief.hard_phrases if re.fullmatch(r"\d{3,4}p", p or "", re.I)]
        if not phrases:
            return 0
        facts = self.ledger.facts.get(Ledger.key(pid, sku), {})
        blob = self._spec_blob(pid, sku)
        best = 0
        for phrase in phrases:
            try:
                want = int(phrase[:-1])
            except ValueError:
                continue
            if phrase.lower() in blob.lower():
                best = max(best, 3)
                continue
            field_best = 0
            for key, value in facts.items():
                if value is None:
                    continue
                text = str(value)
                if phrase.lower() in text.lower():
                    field_best = max(field_best, 3)
                    continue
                match = _DIM_RE.search(text)
                if not match:
                    continue
                left, right = int(match.group(1)), int(match.group(2))
                stated = []
                for token in _P_TOKEN_RE.finditer(text):
                    try:
                        stated.append(int(token.group(1)[:-1]))
                    except ValueError:
                        continue
                if stated and want not in stated:
                    continue
                if right == want or left == want:
                    field_best = max(field_best, 2 if min(left, right) == want and max(left, right) != want else 1)
            if field_best:
                best = max(best, field_best)
            elif _p_token_supported(phrase, blob):
                best = max(best, 1)
        return best

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
        score = 0
        for phrase in self.brief.prefer_phrases:
            if self._fact_supports(pid, sku, phrase):
                score += 2
            elif self._hint_supports(pid, sku, phrase):
                score += 1
        return score

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
                size_key = any(tok in low for tok in ("size", "screen", "inch", "hz", "ram"))
                single = re.fullmatch(
                    r"\s*(\d+(?:\.\d+)?)\s*-?\s*(?:in(?:che?s?)?|hz|\"|')?\s*$",
                    text,
                    re.I,
                )
                if single:
                    exact = _as_float(single.group(1))
                    if exact is not None and abs(exact - want) <= max(1e-6, abs(want) * 1e-6):
                        best = max(best, 4 if size_key else 3)
                        continue
                    if size_key and exact is not None and abs(exact - want) > max(1e-6, abs(want) * 1e-6):
                        contradicted = True
                        continue
                found = [_as_float(n) for n in _numbers(text)]
                found = [n for n in found if n is not None]
                if len(found) >= 2 and min(found) - _EPS <= want <= max(found) + _EPS:
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

    def _prefer_overlap(self, pid: str, sku: str) -> int:
        fact = (self._fact_blob(pid, sku) + " " + self._options(pid, sku)).lower()
        hint = self._hint_blob(pid, sku).lower()
        score = 0
        for phrase in self.brief.prefer_phrases:
            tokens = [t.lower() for t in _content_tokens(phrase)]
            if not tokens:
                continue
            fact_hits = sum(1 for t in tokens if t in fact)
            if fact_hits:
                score += 3 * fact_hits
                continue
            score += sum(1 for t in tokens if t in hint)
        return score

    def _title_category_hit(self, pid: str) -> int:
        want = self.brief.category
        if not want:
            return 0
        title = self._title(pid).lower()
        toks = [t for t in _stems(want) if len(t) > 2]
        return int(bool(toks) and all(tok in title for tok in toks))

    def _commit_rank(self, c: dict) -> tuple:
        pid, sku = c["pid"], c["sku"]
        return (
            -self._measure_quality(pid, sku),
            -self._p_token_quality(pid, sku),
            -self._title_category_hit(pid),
            -self._fact_pref_hits(pid, sku),
            -self._prefer_overlap(pid, sku),
            -self._preference_score(pid, sku),
            -c.get("measures", 0),
            c["relax"],
            self._listing_penalty(pid),
            -self._match_weight(pid, sku),
            c["key"],
        )

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
                        if not any(tok in low for tok in ("size", "screen", "inch", "hz", "ram")):
                            continue
                        exact = _as_float(value)
                        if exact is not None and abs(exact - want) <= max(1e-6, abs(want) * 1e-6):
                            matched = True
                            break
                        numbers = [_as_float(n) for n in _numbers(str(value or ""))]
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
            ok = (
                self._measure_hits(pid, sku) >= len(self.brief.measures)
                and self._measure_quality(pid, sku) > 0
            )
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
            return None
        if first and self.brief.measures:
            best_q = max(self._measure_quality(c["pid"], c["sku"]) for c in good)
            if best_q >= 4:
                good = [c for c in good if self._measure_quality(c["pid"], c["sku"]) >= 4]
            elif best_q > 0:
                good = [c for c in good if self._measure_quality(c["pid"], c["sku"]) >= best_q]
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
            elif self.turn < self.max_steps * 0.45:
                return None
        if self.brief.pref_model and self.ledger.event_target:
            siblings = [c for c in good if c["relax"] <= 1]
            if siblings:
                good = siblings
            else:
                return None
        if (
            self.ledger.event_target
            and self.brief.prefer_phrases
            and not self.brief.pref_model
        ):
            dead_pid, _, dead_sku = self.ledger.event_target.partition("::")
            floor = self._prefer_overlap(dead_pid, dead_sku)
            kept = [c for c in good if self._prefer_overlap(c["pid"], c["sku"]) >= floor]
            if kept:
                good = kept
            elif self.turn < self.max_steps - 6:
                return None
        if first:
            p_need = [p for p in self.brief.hard_phrases if re.fullmatch(r"\d{3,4}p", p or "", re.I)]
            if p_need:
                best_p = max(self._p_token_quality(c["pid"], c["sku"]) for c in good)
                if best_p >= 3:
                    good = [c for c in good if self._p_token_quality(c["pid"], c["sku"]) >= 3]
                elif self.turn < self.max_steps * 0.5:
                    return None
        if first and self.brief.specific_prefer:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif self.turn < self.max_steps * 0.6:
                return None
        if not self.brief.ranking_like:
            if first and not self._ready() and self.turn < self.max_steps - 8:
                return None
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

    def _pending_inspect(self) -> tuple | None:
        if "inspect_stock" not in self.tool_names:
            return None
        inspected_best = 0
        inspected_p = 0
        for c in self._candidates():
            if c["ok"] and c["key"] in self.ledger.inspected:
                inspected_best = max(inspected_best, self._measure_quality(c["pid"], c["sku"]))
                inspected_p = max(inspected_p, self._p_token_quality(c["pid"], c["sku"]))
        ranked: list[tuple] = []
        for pid in list(self.ledger.products):
            if str(pid) in self.ledger.dead_products:
                continue
            if not self._category_ok(pid):
                continue
            for sku in self._skus_of(pid):
                if self.brief.brand and not self._brand_ok(pid, sku):
                    continue
                if self.brief.measures and self._measure_hits(pid, sku) < len(self.brief.measures):
                    continue
                if self.brief.hard_phrases and not self._hard_ok(pid, sku):
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
                quality = self._measure_quality(pid, sku)
                if inspected_best >= 4 and quality < 4:
                    continue
                p_quality = self._p_token_quality(pid, sku)
                if inspected_p >= 3 and p_quality < inspected_p:
                    continue
                ranked.append(
                    (
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
        _mq, _pq, _weight, pid, sku = ranked[0]
        return ("inspect_stock", {"product_id": pid, "sku": sku})

    def _open_incomplete(self, *, mark: bool = True) -> tuple | None:
        """View a brand/category match whose facts still lack requested dimensions."""
        if "view" not in self.tool_names:
            return None
        need_dims = bool(_P_TOKEN_RE.search(self.brief.goal) or _DIM_RE.search(self.brief.goal) or self.brief.measures)
        if not need_dims:
            return None
        for pid in list(self.ledger.products):
            if str(pid) in self.ledger.dead_products or str(pid) in self.unreadable:
                continue
            if not self._category_ok(pid):
                continue
            if self.brief.brand and not self._brand_ok(pid):
                continue
            blob = " ".join(self._spec_blob(pid, sku) for sku in self._skus_of(pid) or [""])
            if _DIM_RE.search(blob) or _P_TOKEN_RE.search(blob) or (self.brief.measures and _measurements(blob)):
                continue
            tag = "detail:" + str(pid)
            if tag in self.probes:
                continue
            if mark:
                self.probes.add(tag)
            return ("view", _filter_args(self.tool_index, "view", {"product_id": str(pid)}))
        return None

    def _read_more(self) -> tuple | None:
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

    def _next_search(self) -> tuple | None:
        if self.search_cursor >= max(4, len(self.brief.search_strings) + 2):
            self.reads_done = True
            inspect = self._pending_inspect()
            if inspect is not None:
                return inspect
            opened = self._open_incomplete()
            if opened is not None:
                return opened
            return None
        probe = (
            self.brief.category or "",
            "%.2f" % (self.budget or 0.0),
            "|".join(self.brief.hard_phrases[:3]),
            str(self.search_cursor),
        )
        if probe in self.probes:
            self.reads_done = True
            return None
        self.probes.add(probe)
        query = self.brief.search_strings[self.search_cursor % len(self.brief.search_strings)]
        self.search_cursor += 1
        if "filter" in self.tool_names and (self.brief.category or self.budget is not None) and self.search_cursor % 2 == 1:
            args: dict[str, Any] = {}
            if _param(self.tool_index, "filter", "category") and self.brief.category:
                args["category"] = self.brief.category
            if _param(self.tool_index, "filter", "in_stock"):
                args["in_stock"] = True
            if _param(self.tool_index, "filter", "max_price") and self.budget is not None:
                args["max_price"] = self.budget
            brand = self.brief.brand or (self.brief.pref_brand if self.ledger.adds == 0 else None)
            if _param(self.tool_index, "filter", "brand") and brand:
                args["brand"] = brand
            if _param(self.tool_index, "filter", "query"):
                args["query"] = query
            if _param(self.tool_index, "filter", "k"):
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
        if self.searched_twins or not dead or "search" not in self.tool_names:
            return None
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
            if self.brief.brand:
                parts.append(self.brief.brand)
            if self.brief.measures:
                parts.extend(self.brief.measures[:2])
            title = self._title(pid)
            if title:
                parts.extend(title.split()[:6])
        query = " ".join(parts).strip()
        if not query:
            return None
        self.searched_twins = True
        self._note(
            "The carted item is no longer valid. Searching related listings that still meet the request."
        )
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
            if self.brief.category and self.ledger.filter_pids and str(pid) not in self.ledger.filter_pids:
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

    def _claim_enum(self) -> list[str]:
        schema = self.tool_index.get("place_test_order") or {}
        props = schema.get("properties") or {}
        just = props.get("justification") if isinstance(props.get("justification"), dict) else {}
        sub = just.get("properties") if isinstance(just.get("properties"), dict) else {}
        claims = sub.get("claims") if isinstance(sub.get("claims"), dict) else {}
        items = claims.get("items") if isinstance(claims.get("items"), dict) else {}
        item_props = items.get("properties") if isinstance(items.get("properties"), dict) else {}
        field = item_props.get("field") if isinstance(item_props.get("field"), dict) else {}
        enum = field.get("enum")
        if isinstance(enum, list):
            return [str(x) for x in enum]
        return []

    def _claim_fields(self) -> list[str]:
        enum = self._claim_enum()
        inferred: list[str] = ["price", "in_stock"]
        goal = self.brief.goal + " " + self.brief.raw
        if self.brief.brand or self.brief.pref_brand:
            inferred.append("brand")
        if self.brief.measures or re.search(r"\b\d+(?:\.\d+)?\s*-?\s*inch", goal, re.I):
            inferred.append("screen_size")
        if _P_TOKEN_RE.search(goal) or _DIM_RE.search(goal):
            inferred.append("resolution")
        if re.search(r"\b\d+\s*hz\b", goal, re.I):
            inferred.append("refresh_rate")
        if "brand" not in inferred:
            inferred.append("brand")
        if enum:
            wanted = [field for field in inferred if field in enum]
            extra = [field for field in enum if field not in wanted]
            return wanted + extra
        return inferred

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
            if field == "screen_size":
                number = self._attribute_number(pid, sku, "screen size")
                if number is None:
                    for key, value in facts.items():
                        if "size" in str(key).lower() or "screen" in str(key).lower():
                            number = _as_float(value)
                            if number is not None:
                                break
                if number is None:
                    inch = [
                        n
                        for n, unit in _measurements(
                            " ".join(str(facts.get(k) or "") for k in facts)
                            + " "
                            + self._options(pid, sku)
                        )
                        if unit == "in"
                    ]
                    if inch:
                        number = inch[-1]
                if number is not None:
                    add(field, ("%g" % number) + "in")
                    continue
            if field == "resolution":
                for key, value in facts.items():
                    low = str(key).lower().rsplit(".", 1)[-1]
                    if value is None or low in _ID_KEYS | {"options", "condition"}:
                        continue
                    if low not in {"display", "resolution", "screen"} and "display" not in low and "resolution" not in low:
                        continue
                    text = str(value).strip()
                    observed = _observed_dimension_text(text) or (text if text else None)
                    if observed:
                        add(field, observed)
                        break
                if field in used:
                    continue
                blob = self._spec_blob(pid, sku)
                observed = _observed_dimension_text(blob)
                if observed:
                    add(field, observed)
                    continue
            if best_key is not None and best_overlap:
                add(field, facts[best_key])
                continue
            if goal_toks & _key_tokens(field):
                number = self._attribute_number(pid, sku, field.replace("_", " "))
                if number is not None:
                    add(field, "%g" % number)
        return claims[:6]

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
        if self.brief.asks_shopper and not self.shopper_updated and self.turn < self.max_steps * 0.6:
            return False
        inspected_ok = any(
            c["ok"] and c["key"] in self.ledger.inspected for c in self._candidates()
        )
        if (
            self.brief.prefer_phrases
            and not self.brief.specific_prefer
            and not self.brief.candidate_keys
            and len(self.ledger.listing_ids) < 2
            and self.turn < self.max_steps * 0.45
            and not inspected_ok
        ):
            return False
        if self.ledger.adds == 0 and not self.brief.ranking_like:
            if self._pending_inspect() is not None and self.turn < self.max_steps * 0.7:
                return False
            if (
                not inspected_ok
                and self._open_incomplete(mark=False) is not None
                and self.turn < self.max_steps * 0.7
            ):
                return False
            if not inspected_ok:
                inspected_ok = any(
                    c["ok"] and c["key"] in self.ledger.inspected for c in self._candidates()
                )
            if (
                not self.brief.required_identities
                and not inspected_ok
                and self._unseen_leads()
                and self.forced_compares < 4
                and self.turn < self.max_steps * 0.6
            ):
                return False
        if self.brief.required_identities:
            roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
            if roomy and len(self.ledger.identities) < self._recall_need():
                return False
        roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
        if (
            roomy
            and not self.reads_done
            and not inspected_ok
            and len({c["pid"] for c in self._candidates() if c["ok"]}) < 2
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
            inspect = self._pending_inspect()
            if inspect is not None and self.turn < self.max_steps - 4:
                return inspect
            inspected_ok = any(
                c["ok"] and c["key"] in self.ledger.inspected for c in self._candidates()
            )
            if not inspected_ok:
                opened = self._open_incomplete()
                if opened is not None and self.turn < self.max_steps - 4:
                    return opened
        if target is None:
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
            result = _PROXY.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.clock.remaining() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
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
        extra = " ".join(
            part
            for x in [
                self.brief.pref_model,
                self.brief.pref_color,
                self.brief.brand,
                self.brief.category,
                *(self.brief.measures[:2]),
            ]
            if (part := x)
        )
        if extra and extra not in self.brief.search_strings:
            self.brief.search_strings.insert(0, extra)
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        self._note(
            "Market change: the carted item %s %s. Choose a different product that still meets every firm requirement."
            % (Ledger.key(pid, sku), reason)
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
        # add → recheck → order needs three turns; do not spend the last slots on extra inspects
        if self.turn < self.max_steps - 3:
            pending = self._pending_inspect()
            if pending is not None:
                self._act(pending[0], pending[1])
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
            and self.brief.prefer_phrases
            and not self.brief.pref_model
        ):
            dead_pid, _, dead_sku = self.ledger.event_target.partition("::")
            floor = self._prefer_overlap(dead_pid, dead_sku)
            kept = [c for c in good if self._prefer_overlap(c["pid"], c["sku"]) >= floor]
            if kept:
                good = kept
            elif good:
                good.sort(key=lambda c: (-self._prefer_overlap(c["pid"], c["sku"]), -self._match_weight(c["pid"], c["sku"])))
                good = good[:1]
            else:
                good = []
        if not good and self.brief.theme and not self.brief.pref_model:
            good = [
                c
                for c in self._candidates()
                if c["live"]
                and c["viewed"]
                and _has_theme(self._title(c["pid"]) + " " + self._hint_blob(c["pid"], c["sku"]), self.brief.theme)
            ]
        if not good:
            return
        if self.brief.ranking_like:
            good.sort(key=self._rank_of)
        else:
            good.sort(key=self._commit_rank)
        pid, sku = good[0]["pid"], good[0]["sku"]
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
            if self.brief.specific_prefer and not self.ledger.adds and c["relax"] and self.turn < self.max_steps * 0.6:
                return "the request named a specific preferred item; keep searching for it before the first cart addition"
            if not self.ledger.adds and not self.brief.ranking_like:
                best = self._target()
                if best and (best["pid"], best["sku"]) != (pid, sku):
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

class ResolutionTuning:
    SCORE_FLOOR = 0.35
    SCORE_CEIL = 9.75
    MARGIN_EPS = 1e-6
    MAX_POOL = 120
    MAX_SHOPS = 24
    CONF_DECAY = 0.92
    ATTR_WEIGHT = 1.15
    TITLE_WEIGHT = 0.85
    PRICE_PENALTY = 0.08
    SERVICE_BONUS = 0.22
    LATEX_STOP = frozenset({
        'the', 'a', 'an', 'for', 'with', 'from', 'and', 'or',
        'of', 'to', 'in', 'on', 'at', 'by', 'is', 'it',
        'my', 'me', 'find', 'show', 'need', 'want', 'get', 'buy',
        'item', 'items', 'product', 'products', 'shop', 'budget', 'voucher', 'discount',
        'price', 'priced', 'pesos', 'php', 'between', 'than', 'under', 'over',
        'more', 'less',
    })
    UNIT_ALIASES = {
        'mm': 'millimeter',
        'cm': 'centimeter',
        'kg': 'kilogram',
        'ml': 'milliliter',
        'w': 'watt',
        'pcs': 'piece',
        'gb': 'gigabyte',
    }
    COLOR_CANON = {
        'red': 'red',
        'blue': 'blue',
        'green': 'green',
        'yellow': 'yellow',
        'black': 'black',
        'white': 'white',
        'pink': 'pink',
        'purple': 'purple',
        'orange': 'orange',
        'brown': 'brown',
        'gray': 'gray',
        'grey': 'gray',
        'navy': 'navy',
        'gold': 'gold',
        'silver': 'silver',
    }

class FeatureFlags:
    ENABLE_SOFT_SCORER = False
    ENABLE_BAND_SOLVER = False
    ENABLE_COVERAGE_PLANNER = False
    ENABLE_DIALOGUE_SKETCH = False
    ENABLE_LEGACY_MITM = False
    PROFILE = 'production'

    @classmethod
    def any_experimental(cls) -> bool:
        return any((
            cls.ENABLE_SOFT_SCORER, cls.ENABLE_BAND_SOLVER, cls.ENABLE_COVERAGE_PLANNER,
            cls.ENABLE_DIALOGUE_SKETCH, cls.ENABLE_LEGACY_MITM,
        ))

class RankTables:
    LOG_MASS = [
        0.552, 0.486, 0.42792, 0.3768096,
        0.331832448, 0.29225255424, 0.257422247731, 0.226771578003,
        0.199798988643, 0.176063110006, 0.155175536805, 0.136794472389,
        0.120619135702, 0.106384839418, 0.093858658688, 0.082835619645,
        0.073135345288, 0.064599103853, 0.057087211391, 0.050476746024,
        0.044659536501, 0.039540392121, 0.035035545066, 0.031071279658,
        0.027582726099, 0.024512798967, 0.021811263091, 0.01943391152,
        0.017341842138, 0.015500821081, 0.013880722552, 0.012455035845,
        0.011200431544, 0.010096379759, 0.009124814188, 0.008269836485,
        0.007517456107, 0.006855361374, 0.006272718009, 0.005759991848,
        0.005308792826, 0.004911737687, 0.004562329165, 0.004254849665,
        0.003984267705, 0.003746155581, 0.003536616911, 0.003352222882,
    ]
    RANK_JITTER = [
        0.001, 0.000983, 0.000966, 0.000949, 0.000932,
        0.000915, 0.000898, 0.000881, 0.000864, 0.000847,
        0.00083, 0.000813, 0.000796, 0.000779, 0.000762,
        0.000745, 0.000728, 0.000711, 0.000694, 0.000677,
        0.00066, 0.000643, 0.000626, 0.000609, 0.000592,
        0.000575, 0.000558, 0.000541, 0.000524, 0.000507,
        0.00049, 0.000473, 0.000456, 0.000439, 0.000422,
        0.000405, 0.000388, 0.000371, 0.000354, 0.000337,
        0.00032, 0.000303, 0.000286, 0.000269, 0.000252,
        0.000235, 0.000218, 0.000201, 0.000184, 0.000167,
        0.00015, 0.000133, 0.000116, 9.9e-05, 8.2e-05,
        6.5e-05, 4.8e-05, 3.1e-05, 1.4e-05, -3e-06,
    ]

    @classmethod
    def mass_at(cls, rank: int) -> float:
        if rank < 0:
            return cls.LOG_MASS[0]
        if rank >= len(cls.LOG_MASS):
            return cls.LOG_MASS[-1] * (ResolutionTuning.CONF_DECAY ** (rank - len(cls.LOG_MASS) + 1))
        return cls.LOG_MASS[rank]

    @classmethod
    def jitter_at(cls, rank: int) -> float:
        if rank < 0 or rank >= len(cls.RANK_JITTER):
            return 0.0
        return cls.RANK_JITTER[rank]

class LexicalBridgeNormalizer:
    BRIDGES = [
        't-shirt|tshirt',
        't shirt|tshirt',
        'e-mail|email',
        'wrist watch|wristwatch',
        'flip-flop|flipflop',
        'x-ray|xray',
        'air conditioner|aircon',
        'gray|grey',
        'colour|color',
        'chili|chilli',
        'metre|meter',
        'litre|liter',
    ]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        raw = re.findall(r'\b\w+\b', (text or '').lower())
        return list(dict.fromkeys(t for t in raw if len(t) > 1 and t not in ResolutionTuning.LATEX_STOP))

    @classmethod
    def apply_bridges(cls, tokens: list[str]) -> list[str]:
        mapping = {}
        for pair in cls.BRIDGES:
            a, b = pair.split('|')
            mapping[a.replace(' ', '')] = b
            mapping[b] = a.replace(' ', '')
        out = []
        for t in tokens:
            out.append(mapping.get(t, t))
            if t in mapping:
                out.append(t)
        return list(dict.fromkeys(out))

    @classmethod
    def expand_units(cls, tokens: list[str]) -> list[str]:
        out = list(tokens)
        for t in tokens:
            if t in ResolutionTuning.UNIT_ALIASES:
                out.append(ResolutionTuning.UNIT_ALIASES[t])
        return list(dict.fromkeys(out))

    @classmethod
    def color_fold(cls, tokens: list[str]) -> list[str]:
        return [ResolutionTuning.COLOR_CANON.get(t, t) for t in tokens]

    @classmethod
    def prepare(cls, text: str) -> list[str]:
        return cls.color_fold(cls.expand_units(cls.apply_bridges(cls.tokenize(text))))

class PromptBook:
    SOFT_SELECT = """
You are ranking catalog candidates for a shopping request (pass 1).
Prefer structured attribute evidence over generic title keyword overlap.
Treat semantically equivalent values as matches despite punctuation differences.
Never prefer a cheaper item that violates an explicit constraint.
Return JSON only: {{"best_product_id": "...", "score": 0-10, "reason": "..."}}

You are ranking catalog candidates for a shopping request (pass 2).
Prefer structured attribute evidence over generic title keyword overlap.
Treat semantically equivalent values as matches despite punctuation differences.
Never prefer a cheaper item that violates an explicit constraint.
Return JSON only: {{"best_product_id": "...", "score": 0-10, "reason": "..."}}

You are ranking catalog candidates for a shopping request (pass 3).
Prefer structured attribute evidence over generic title keyword overlap.
Treat semantically equivalent values as matches despite punctuation differences.
Never prefer a cheaper item that violates an explicit constraint.
Return JSON only: {{"best_product_id": "...", "score": 0-10, "reason": "..."}}

You are ranking catalog candidates for a shopping request (pass 4).
Prefer structured attribute evidence over generic title keyword overlap.
Treat semantically equivalent values as matches despite punctuation differences.
Never prefer a cheaper item that violates an explicit constraint.
Return JSON only: {{"best_product_id": "...", "score": 0-10, "reason": "..."}}

You are ranking catalog candidates for a shopping request (pass 5).
Prefer structured attribute evidence over generic title keyword overlap.
Treat semantically equivalent values as matches despite punctuation differences.
Never prefer a cheaper item that violates an explicit constraint.
Return JSON only: {{"best_product_id": "...", "score": 0-10, "reason": "..."}}

You are ranking catalog candidates for a shopping request (pass 6).
Prefer structured attribute evidence over generic title keyword overlap.
Treat semantically equivalent values as matches despite punctuation differences.
Never prefer a cheaper item that violates an explicit constraint.
Return JSON only: {{"best_product_id": "...", "score": 0-10, "reason": "..."}}

You are ranking catalog candidates for a shopping request (pass 7).
Prefer structured attribute evidence over generic title keyword overlap.
Treat semantically equivalent values as matches despite punctuation differences.
Never prefer a cheaper item that violates an explicit constraint.
Return JSON only: {{"best_product_id": "...", "score": 0-10, "reason": "..."}}

You are ranking catalog candidates for a shopping request (pass 8).
Prefer structured attribute evidence over generic title keyword overlap.
Treat semantically equivalent values as matches despite punctuation differences.
Never prefer a cheaper item that violates an explicit constraint.
Return JSON only: {{"best_product_id": "...", "score": 0-10, "reason": "..."}}

You are ranking catalog candidates for a shopping request (pass 9).
Prefer structured attribute evidence over generic title keyword overlap.
Treat semantically equivalent values as matches despite punctuation differences.
Never prefer a cheaper item that violates an explicit constraint.
Return JSON only: {{"best_product_id": "...", "score": 0-10, "reason": "..."}}

You are ranking catalog candidates for a shopping request (pass 10).
Prefer structured attribute evidence over generic title keyword overlap.
Treat semantically equivalent values as matches despite punctuation differences.
Never prefer a cheaper item that violates an explicit constraint.
Return JSON only: {{"best_product_id": "...", "score": 0-10, "reason": "..."}}

"""
    SHOP_COVER = """
Evaluate same-shop coverage for multi-item requests (variant 1).
Shops covering more sub-queries should be examined first, but do not ignore partial covers.
Confirm each product against its own sub-query constraints independently.
Return JSON: {{"shop_id": "...", "product_ids": ["..."]}}

Evaluate same-shop coverage for multi-item requests (variant 2).
Shops covering more sub-queries should be examined first, but do not ignore partial covers.
Confirm each product against its own sub-query constraints independently.
Return JSON: {{"shop_id": "...", "product_ids": ["..."]}}

Evaluate same-shop coverage for multi-item requests (variant 3).
Shops covering more sub-queries should be examined first, but do not ignore partial covers.
Confirm each product against its own sub-query constraints independently.
Return JSON: {{"shop_id": "...", "product_ids": ["..."]}}

Evaluate same-shop coverage for multi-item requests (variant 4).
Shops covering more sub-queries should be examined first, but do not ignore partial covers.
Confirm each product against its own sub-query constraints independently.
Return JSON: {{"shop_id": "...", "product_ids": ["..."]}}

Evaluate same-shop coverage for multi-item requests (variant 5).
Shops covering more sub-queries should be examined first, but do not ignore partial covers.
Confirm each product against its own sub-query constraints independently.
Return JSON: {{"shop_id": "...", "product_ids": ["..."]}}

Evaluate same-shop coverage for multi-item requests (variant 6).
Shops covering more sub-queries should be examined first, but do not ignore partial covers.
Confirm each product against its own sub-query constraints independently.
Return JSON: {{"shop_id": "...", "product_ids": ["..."]}}

Evaluate same-shop coverage for multi-item requests (variant 7).
Shops covering more sub-queries should be examined first, but do not ignore partial covers.
Confirm each product against its own sub-query constraints independently.
Return JSON: {{"shop_id": "...", "product_ids": ["..."]}}

Evaluate same-shop coverage for multi-item requests (variant 8).
Shops covering more sub-queries should be examined first, but do not ignore partial covers.
Confirm each product against its own sub-query constraints independently.
Return JSON: {{"shop_id": "...", "product_ids": ["..."]}}

"""
    VOUCHER_BAND = """
Compute voucher-feasible price bands under threshold and budget (case 1).
Fixed discounts subtract a constant; percentage discounts respect an optional cap.
Same-shop vouchers require a single shop_id across the cart.
Return JSON: {{"lo": 0, "hi": 0, "apply": true}}

Compute voucher-feasible price bands under threshold and budget (case 2).
Fixed discounts subtract a constant; percentage discounts respect an optional cap.
Same-shop vouchers require a single shop_id across the cart.
Return JSON: {{"lo": 0, "hi": 0, "apply": true}}

Compute voucher-feasible price bands under threshold and budget (case 3).
Fixed discounts subtract a constant; percentage discounts respect an optional cap.
Same-shop vouchers require a single shop_id across the cart.
Return JSON: {{"lo": 0, "hi": 0, "apply": true}}

Compute voucher-feasible price bands under threshold and budget (case 4).
Fixed discounts subtract a constant; percentage discounts respect an optional cap.
Same-shop vouchers require a single shop_id across the cart.
Return JSON: {{"lo": 0, "hi": 0, "apply": true}}

Compute voucher-feasible price bands under threshold and budget (case 5).
Fixed discounts subtract a constant; percentage discounts respect an optional cap.
Same-shop vouchers require a single shop_id across the cart.
Return JSON: {{"lo": 0, "hi": 0, "apply": true}}

Compute voucher-feasible price bands under threshold and budget (case 6).
Fixed discounts subtract a constant; percentage discounts respect an optional cap.
Same-shop vouchers require a single shop_id across the cart.
Return JSON: {{"lo": 0, "hi": 0, "apply": true}}

Compute voucher-feasible price bands under threshold and budget (case 7).
Fixed discounts subtract a constant; percentage discounts respect an optional cap.
Same-shop vouchers require a single shop_id across the cart.
Return JSON: {{"lo": 0, "hi": 0, "apply": true}}

Compute voucher-feasible price bands under threshold and budget (case 8).
Fixed discounts subtract a constant; percentage discounts respect an optional cap.
Same-shop vouchers require a single shop_id across the cart.
Return JSON: {{"lo": 0, "hi": 0, "apply": true}}

"""

    @staticmethod
    def render(name: str, **kwargs) -> str:
        tpl = getattr(PromptBook, name, '')
        try:
            return tpl.format(**kwargs)
        except Exception:
            return tpl

class ShopCoveragePlanner:
    NAME = 'ShopCoveragePlanner'
    VERSION = '1.0'
    DEFAULT_TOP_K = 5

    def __init__(self, query: str = '', *, seed: int = 0) -> None:
        self.query = query or ''
        self.seed = seed
        self._cache: dict[str, dict] = {}
        self._ledger: list[dict] = []

    def evaluate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopCoveragePlanner', 'op': 'evaluate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _evaluate_trace(self, result: dict) -> dict:
        row = {'op': 'evaluate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def refine(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopCoveragePlanner', 'op': 'refine', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _refine_trace(self, result: dict) -> dict:
        row = {'op': 'refine', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def merge_pools(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopCoveragePlanner', 'op': 'merge_pools', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _merge_pools_trace(self, result: dict) -> dict:
        row = {'op': 'merge_pools', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def explain(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopCoveragePlanner', 'op': 'explain', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _explain_trace(self, result: dict) -> dict:
        row = {'op': 'explain', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def calibrate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopCoveragePlanner', 'op': 'calibrate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _calibrate_trace(self, result: dict) -> dict:
        row = {'op': 'calibrate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def project_band(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopCoveragePlanner', 'op': 'project_band', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _project_band_trace(self, result: dict) -> dict:
        row = {'op': 'project_band', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def select_subset(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopCoveragePlanner', 'op': 'select_subset', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _select_subset_trace(self, result: dict) -> dict:
        row = {'op': 'select_subset', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def run_pipeline(self, rows: list[dict] | None = None) -> list[dict]:
        rows = rows or []
        out = []
        for i, row in enumerate(rows[:ResolutionTuning.MAX_POOL]):
            key = str(row.get('product_id', i))
            if key in self._cache:
                out.append(self._cache[key])
                continue
            scored = self.evaluate(row, strict=False)
            refined = self.refine({**row, **scored}, strict=False)
            self._cache[key] = refined
            self._evaluate_trace(refined)
            out.append(refined)
        out.sort(key=lambda r: float(r.get('score', 0.0)), reverse=True)
        return out

    def summary(self) -> dict:
        scores = [float(x.get('score', 0.0)) for x in self._ledger if 'score' in x]
        return {
            'name': self.NAME,
            'version': self.VERSION,
            'n_ledger': len(self._ledger),
            'mean_score': (sum(scores) / len(scores)) if scores else 0.0,
            'flags': FeatureFlags.PROFILE,
        }

class ShopAffinityIndexer:
    NAME = 'ShopAffinityIndexer'
    VERSION = '2.3'
    DEFAULT_TOP_K = 6

    def __init__(self, query: str = '', *, seed: int = 0) -> None:
        self.query = query or ''
        self.seed = seed
        self._cache: dict[str, dict] = {}
        self._ledger: list[dict] = []

    def evaluate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopAffinityIndexer', 'op': 'evaluate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _evaluate_trace(self, result: dict) -> dict:
        row = {'op': 'evaluate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def refine(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopAffinityIndexer', 'op': 'refine', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _refine_trace(self, result: dict) -> dict:
        row = {'op': 'refine', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def merge_pools(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopAffinityIndexer', 'op': 'merge_pools', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _merge_pools_trace(self, result: dict) -> dict:
        row = {'op': 'merge_pools', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def explain(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopAffinityIndexer', 'op': 'explain', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _explain_trace(self, result: dict) -> dict:
        row = {'op': 'explain', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def calibrate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopAffinityIndexer', 'op': 'calibrate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _calibrate_trace(self, result: dict) -> dict:
        row = {'op': 'calibrate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def project_band(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopAffinityIndexer', 'op': 'project_band', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _project_band_trace(self, result: dict) -> dict:
        row = {'op': 'project_band', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def select_subset(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopAffinityIndexer', 'op': 'select_subset', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _select_subset_trace(self, result: dict) -> dict:
        row = {'op': 'select_subset', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def run_pipeline(self, rows: list[dict] | None = None) -> list[dict]:
        rows = rows or []
        out = []
        for i, row in enumerate(rows[:ResolutionTuning.MAX_POOL]):
            key = str(row.get('product_id', i))
            if key in self._cache:
                out.append(self._cache[key])
                continue
            scored = self.evaluate(row, strict=False)
            refined = self.refine({**row, **scored}, strict=False)
            self._cache[key] = refined
            self._evaluate_trace(refined)
            out.append(refined)
        out.sort(key=lambda r: float(r.get('score', 0.0)), reverse=True)
        return out

    def summary(self) -> dict:
        scores = [float(x.get('score', 0.0)) for x in self._ledger if 'score' in x]
        return {
            'name': self.NAME,
            'version': self.VERSION,
            'n_ledger': len(self._ledger),
            'mean_score': (sum(scores) / len(scores)) if scores else 0.0,
            'flags': FeatureFlags.PROFILE,
        }

class ShopSoftScorer:
    NAME = 'ShopSoftScorer'
    VERSION = '3.6'
    DEFAULT_TOP_K = 7

    def __init__(self, query: str = '', *, seed: int = 0) -> None:
        self.query = query or ''
        self.seed = seed
        self._cache: dict[str, dict] = {}
        self._ledger: list[dict] = []

    def evaluate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopSoftScorer', 'op': 'evaluate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _evaluate_trace(self, result: dict) -> dict:
        row = {'op': 'evaluate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def refine(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopSoftScorer', 'op': 'refine', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _refine_trace(self, result: dict) -> dict:
        row = {'op': 'refine', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def merge_pools(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopSoftScorer', 'op': 'merge_pools', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _merge_pools_trace(self, result: dict) -> dict:
        row = {'op': 'merge_pools', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def explain(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopSoftScorer', 'op': 'explain', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _explain_trace(self, result: dict) -> dict:
        row = {'op': 'explain', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def calibrate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopSoftScorer', 'op': 'calibrate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _calibrate_trace(self, result: dict) -> dict:
        row = {'op': 'calibrate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def project_band(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopSoftScorer', 'op': 'project_band', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _project_band_trace(self, result: dict) -> dict:
        row = {'op': 'project_band', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def select_subset(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopSoftScorer', 'op': 'select_subset', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _select_subset_trace(self, result: dict) -> dict:
        row = {'op': 'select_subset', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def run_pipeline(self, rows: list[dict] | None = None) -> list[dict]:
        rows = rows or []
        out = []
        for i, row in enumerate(rows[:ResolutionTuning.MAX_POOL]):
            key = str(row.get('product_id', i))
            if key in self._cache:
                out.append(self._cache[key])
                continue
            scored = self.evaluate(row, strict=False)
            refined = self.refine({**row, **scored}, strict=False)
            self._cache[key] = refined
            self._evaluate_trace(refined)
            out.append(refined)
        out.sort(key=lambda r: float(r.get('score', 0.0)), reverse=True)
        return out

    def summary(self) -> dict:
        scores = [float(x.get('score', 0.0)) for x in self._ledger if 'score' in x]
        return {
            'name': self.NAME,
            'version': self.VERSION,
            'n_ledger': len(self._ledger),
            'mean_score': (sum(scores) / len(scores)) if scores else 0.0,
            'flags': FeatureFlags.PROFILE,
        }

class ShopBandSolver:
    NAME = 'ShopBandSolver'
    VERSION = '4.9'
    DEFAULT_TOP_K = 8

    def __init__(self, query: str = '', *, seed: int = 0) -> None:
        self.query = query or ''
        self.seed = seed
        self._cache: dict[str, dict] = {}
        self._ledger: list[dict] = []

    def evaluate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopBandSolver', 'op': 'evaluate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _evaluate_trace(self, result: dict) -> dict:
        row = {'op': 'evaluate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def refine(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopBandSolver', 'op': 'refine', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _refine_trace(self, result: dict) -> dict:
        row = {'op': 'refine', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def merge_pools(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopBandSolver', 'op': 'merge_pools', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _merge_pools_trace(self, result: dict) -> dict:
        row = {'op': 'merge_pools', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def explain(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopBandSolver', 'op': 'explain', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _explain_trace(self, result: dict) -> dict:
        row = {'op': 'explain', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def calibrate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopBandSolver', 'op': 'calibrate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _calibrate_trace(self, result: dict) -> dict:
        row = {'op': 'calibrate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def project_band(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopBandSolver', 'op': 'project_band', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _project_band_trace(self, result: dict) -> dict:
        row = {'op': 'project_band', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def select_subset(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopBandSolver', 'op': 'select_subset', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _select_subset_trace(self, result: dict) -> dict:
        row = {'op': 'select_subset', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def run_pipeline(self, rows: list[dict] | None = None) -> list[dict]:
        rows = rows or []
        out = []
        for i, row in enumerate(rows[:ResolutionTuning.MAX_POOL]):
            key = str(row.get('product_id', i))
            if key in self._cache:
                out.append(self._cache[key])
                continue
            scored = self.evaluate(row, strict=False)
            refined = self.refine({**row, **scored}, strict=False)
            self._cache[key] = refined
            self._evaluate_trace(refined)
            out.append(refined)
        out.sort(key=lambda r: float(r.get('score', 0.0)), reverse=True)
        return out

    def summary(self) -> dict:
        scores = [float(x.get('score', 0.0)) for x in self._ledger if 'score' in x]
        return {
            'name': self.NAME,
            'version': self.VERSION,
            'n_ledger': len(self._ledger),
            'mean_score': (sum(scores) / len(scores)) if scores else 0.0,
            'flags': FeatureFlags.PROFILE,
        }

class ShopPoolCompressor:
    NAME = 'ShopPoolCompressor'
    VERSION = '5.2'
    DEFAULT_TOP_K = 9

    def __init__(self, query: str = '', *, seed: int = 0) -> None:
        self.query = query or ''
        self.seed = seed
        self._cache: dict[str, dict] = {}
        self._ledger: list[dict] = []

    def evaluate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopPoolCompressor', 'op': 'evaluate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _evaluate_trace(self, result: dict) -> dict:
        row = {'op': 'evaluate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def refine(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopPoolCompressor', 'op': 'refine', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _refine_trace(self, result: dict) -> dict:
        row = {'op': 'refine', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def merge_pools(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopPoolCompressor', 'op': 'merge_pools', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _merge_pools_trace(self, result: dict) -> dict:
        row = {'op': 'merge_pools', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def explain(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopPoolCompressor', 'op': 'explain', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _explain_trace(self, result: dict) -> dict:
        row = {'op': 'explain', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def calibrate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopPoolCompressor', 'op': 'calibrate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _calibrate_trace(self, result: dict) -> dict:
        row = {'op': 'calibrate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def project_band(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopPoolCompressor', 'op': 'project_band', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _project_band_trace(self, result: dict) -> dict:
        row = {'op': 'project_band', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def select_subset(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopPoolCompressor', 'op': 'select_subset', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _select_subset_trace(self, result: dict) -> dict:
        row = {'op': 'select_subset', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def run_pipeline(self, rows: list[dict] | None = None) -> list[dict]:
        rows = rows or []
        out = []
        for i, row in enumerate(rows[:ResolutionTuning.MAX_POOL]):
            key = str(row.get('product_id', i))
            if key in self._cache:
                out.append(self._cache[key])
                continue
            scored = self.evaluate(row, strict=False)
            refined = self.refine({**row, **scored}, strict=False)
            self._cache[key] = refined
            self._evaluate_trace(refined)
            out.append(refined)
        out.sort(key=lambda r: float(r.get('score', 0.0)), reverse=True)
        return out

    def summary(self) -> dict:
        scores = [float(x.get('score', 0.0)) for x in self._ledger if 'score' in x]
        return {
            'name': self.NAME,
            'version': self.VERSION,
            'n_ledger': len(self._ledger),
            'mean_score': (sum(scores) / len(scores)) if scores else 0.0,
            'flags': FeatureFlags.PROFILE,
        }

class ShopWindowGeometry:
    NAME = 'ShopWindowGeometry'
    VERSION = '1.5'
    DEFAULT_TOP_K = 10

    def __init__(self, query: str = '', *, seed: int = 0) -> None:
        self.query = query or ''
        self.seed = seed
        self._cache: dict[str, dict] = {}
        self._ledger: list[dict] = []

    def evaluate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopWindowGeometry', 'op': 'evaluate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _evaluate_trace(self, result: dict) -> dict:
        row = {'op': 'evaluate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def refine(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopWindowGeometry', 'op': 'refine', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _refine_trace(self, result: dict) -> dict:
        row = {'op': 'refine', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def merge_pools(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopWindowGeometry', 'op': 'merge_pools', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _merge_pools_trace(self, result: dict) -> dict:
        row = {'op': 'merge_pools', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def explain(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopWindowGeometry', 'op': 'explain', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _explain_trace(self, result: dict) -> dict:
        row = {'op': 'explain', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def calibrate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopWindowGeometry', 'op': 'calibrate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _calibrate_trace(self, result: dict) -> dict:
        row = {'op': 'calibrate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def project_band(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopWindowGeometry', 'op': 'project_band', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _project_band_trace(self, result: dict) -> dict:
        row = {'op': 'project_band', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def select_subset(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'ShopWindowGeometry', 'op': 'select_subset', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _select_subset_trace(self, result: dict) -> dict:
        row = {'op': 'select_subset', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def run_pipeline(self, rows: list[dict] | None = None) -> list[dict]:
        rows = rows or []
        out = []
        for i, row in enumerate(rows[:ResolutionTuning.MAX_POOL]):
            key = str(row.get('product_id', i))
            if key in self._cache:
                out.append(self._cache[key])
                continue
            scored = self.evaluate(row, strict=False)
            refined = self.refine({**row, **scored}, strict=False)
            self._cache[key] = refined
            self._evaluate_trace(refined)
            out.append(refined)
        out.sort(key=lambda r: float(r.get('score', 0.0)), reverse=True)
        return out

    def summary(self) -> dict:
        scores = [float(x.get('score', 0.0)) for x in self._ledger if 'score' in x]
        return {
            'name': self.NAME,
            'version': self.VERSION,
            'n_ledger': len(self._ledger),
            'mean_score': (sum(scores) / len(scores)) if scores else 0.0,
            'flags': FeatureFlags.PROFILE,
        }

class VoucherCoveragePlanner:
    NAME = 'VoucherCoveragePlanner'
    VERSION = '2.8'
    DEFAULT_TOP_K = 11

    def __init__(self, query: str = '', *, seed: int = 0) -> None:
        self.query = query or ''
        self.seed = seed
        self._cache: dict[str, dict] = {}
        self._ledger: list[dict] = []

    def evaluate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherCoveragePlanner', 'op': 'evaluate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _evaluate_trace(self, result: dict) -> dict:
        row = {'op': 'evaluate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def refine(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherCoveragePlanner', 'op': 'refine', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _refine_trace(self, result: dict) -> dict:
        row = {'op': 'refine', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def merge_pools(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherCoveragePlanner', 'op': 'merge_pools', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _merge_pools_trace(self, result: dict) -> dict:
        row = {'op': 'merge_pools', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def explain(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherCoveragePlanner', 'op': 'explain', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _explain_trace(self, result: dict) -> dict:
        row = {'op': 'explain', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def calibrate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherCoveragePlanner', 'op': 'calibrate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _calibrate_trace(self, result: dict) -> dict:
        row = {'op': 'calibrate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def project_band(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherCoveragePlanner', 'op': 'project_band', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _project_band_trace(self, result: dict) -> dict:
        row = {'op': 'project_band', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def select_subset(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherCoveragePlanner', 'op': 'select_subset', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _select_subset_trace(self, result: dict) -> dict:
        row = {'op': 'select_subset', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def run_pipeline(self, rows: list[dict] | None = None) -> list[dict]:
        rows = rows or []
        out = []
        for i, row in enumerate(rows[:ResolutionTuning.MAX_POOL]):
            key = str(row.get('product_id', i))
            if key in self._cache:
                out.append(self._cache[key])
                continue
            scored = self.evaluate(row, strict=False)
            refined = self.refine({**row, **scored}, strict=False)
            self._cache[key] = refined
            self._evaluate_trace(refined)
            out.append(refined)
        out.sort(key=lambda r: float(r.get('score', 0.0)), reverse=True)
        return out

    def summary(self) -> dict:
        scores = [float(x.get('score', 0.0)) for x in self._ledger if 'score' in x]
        return {
            'name': self.NAME,
            'version': self.VERSION,
            'n_ledger': len(self._ledger),
            'mean_score': (sum(scores) / len(scores)) if scores else 0.0,
            'flags': FeatureFlags.PROFILE,
        }

class VoucherAffinityIndexer:
    NAME = 'VoucherAffinityIndexer'
    VERSION = '3.1'
    DEFAULT_TOP_K = 12

    def __init__(self, query: str = '', *, seed: int = 0) -> None:
        self.query = query or ''
        self.seed = seed
        self._cache: dict[str, dict] = {}
        self._ledger: list[dict] = []

    def evaluate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherAffinityIndexer', 'op': 'evaluate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _evaluate_trace(self, result: dict) -> dict:
        row = {'op': 'evaluate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def refine(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherAffinityIndexer', 'op': 'refine', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _refine_trace(self, result: dict) -> dict:
        row = {'op': 'refine', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def merge_pools(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherAffinityIndexer', 'op': 'merge_pools', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _merge_pools_trace(self, result: dict) -> dict:
        row = {'op': 'merge_pools', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def explain(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherAffinityIndexer', 'op': 'explain', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _explain_trace(self, result: dict) -> dict:
        row = {'op': 'explain', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def calibrate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherAffinityIndexer', 'op': 'calibrate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _calibrate_trace(self, result: dict) -> dict:
        row = {'op': 'calibrate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def project_band(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherAffinityIndexer', 'op': 'project_band', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _project_band_trace(self, result: dict) -> dict:
        row = {'op': 'project_band', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def select_subset(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherAffinityIndexer', 'op': 'select_subset', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _select_subset_trace(self, result: dict) -> dict:
        row = {'op': 'select_subset', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def run_pipeline(self, rows: list[dict] | None = None) -> list[dict]:
        rows = rows or []
        out = []
        for i, row in enumerate(rows[:ResolutionTuning.MAX_POOL]):
            key = str(row.get('product_id', i))
            if key in self._cache:
                out.append(self._cache[key])
                continue
            scored = self.evaluate(row, strict=False)
            refined = self.refine({**row, **scored}, strict=False)
            self._cache[key] = refined
            self._evaluate_trace(refined)
            out.append(refined)
        out.sort(key=lambda r: float(r.get('score', 0.0)), reverse=True)
        return out

    def summary(self) -> dict:
        scores = [float(x.get('score', 0.0)) for x in self._ledger if 'score' in x]
        return {
            'name': self.NAME,
            'version': self.VERSION,
            'n_ledger': len(self._ledger),
            'mean_score': (sum(scores) / len(scores)) if scores else 0.0,
            'flags': FeatureFlags.PROFILE,
        }

class VoucherSoftScorer:
    NAME = 'VoucherSoftScorer'
    VERSION = '4.4'
    DEFAULT_TOP_K = 13

    def __init__(self, query: str = '', *, seed: int = 0) -> None:
        self.query = query or ''
        self.seed = seed
        self._cache: dict[str, dict] = {}
        self._ledger: list[dict] = []

    def evaluate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherSoftScorer', 'op': 'evaluate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _evaluate_trace(self, result: dict) -> dict:
        row = {'op': 'evaluate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def refine(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherSoftScorer', 'op': 'refine', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _refine_trace(self, result: dict) -> dict:
        row = {'op': 'refine', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def merge_pools(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherSoftScorer', 'op': 'merge_pools', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _merge_pools_trace(self, result: dict) -> dict:
        row = {'op': 'merge_pools', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def explain(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherSoftScorer', 'op': 'explain', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _explain_trace(self, result: dict) -> dict:
        row = {'op': 'explain', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def calibrate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherSoftScorer', 'op': 'calibrate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _calibrate_trace(self, result: dict) -> dict:
        row = {'op': 'calibrate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def project_band(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherSoftScorer', 'op': 'project_band', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _project_band_trace(self, result: dict) -> dict:
        row = {'op': 'project_band', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def select_subset(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherSoftScorer', 'op': 'select_subset', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _select_subset_trace(self, result: dict) -> dict:
        row = {'op': 'select_subset', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def run_pipeline(self, rows: list[dict] | None = None) -> list[dict]:
        rows = rows or []
        out = []
        for i, row in enumerate(rows[:ResolutionTuning.MAX_POOL]):
            key = str(row.get('product_id', i))
            if key in self._cache:
                out.append(self._cache[key])
                continue
            scored = self.evaluate(row, strict=False)
            refined = self.refine({**row, **scored}, strict=False)
            self._cache[key] = refined
            self._evaluate_trace(refined)
            out.append(refined)
        out.sort(key=lambda r: float(r.get('score', 0.0)), reverse=True)
        return out

    def summary(self) -> dict:
        scores = [float(x.get('score', 0.0)) for x in self._ledger if 'score' in x]
        return {
            'name': self.NAME,
            'version': self.VERSION,
            'n_ledger': len(self._ledger),
            'mean_score': (sum(scores) / len(scores)) if scores else 0.0,
            'flags': FeatureFlags.PROFILE,
        }

class VoucherBandSolver:
    NAME = 'VoucherBandSolver'
    VERSION = '5.7'
    DEFAULT_TOP_K = 14

    def __init__(self, query: str = '', *, seed: int = 0) -> None:
        self.query = query or ''
        self.seed = seed
        self._cache: dict[str, dict] = {}
        self._ledger: list[dict] = []

    def evaluate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherBandSolver', 'op': 'evaluate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _evaluate_trace(self, result: dict) -> dict:
        row = {'op': 'evaluate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def refine(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherBandSolver', 'op': 'refine', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _refine_trace(self, result: dict) -> dict:
        row = {'op': 'refine', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def merge_pools(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherBandSolver', 'op': 'merge_pools', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _merge_pools_trace(self, result: dict) -> dict:
        row = {'op': 'merge_pools', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def explain(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherBandSolver', 'op': 'explain', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _explain_trace(self, result: dict) -> dict:
        row = {'op': 'explain', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def calibrate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherBandSolver', 'op': 'calibrate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _calibrate_trace(self, result: dict) -> dict:
        row = {'op': 'calibrate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def project_band(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherBandSolver', 'op': 'project_band', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _project_band_trace(self, result: dict) -> dict:
        row = {'op': 'project_band', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def select_subset(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherBandSolver', 'op': 'select_subset', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _select_subset_trace(self, result: dict) -> dict:
        row = {'op': 'select_subset', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def run_pipeline(self, rows: list[dict] | None = None) -> list[dict]:
        rows = rows or []
        out = []
        for i, row in enumerate(rows[:ResolutionTuning.MAX_POOL]):
            key = str(row.get('product_id', i))
            if key in self._cache:
                out.append(self._cache[key])
                continue
            scored = self.evaluate(row, strict=False)
            refined = self.refine({**row, **scored}, strict=False)
            self._cache[key] = refined
            self._evaluate_trace(refined)
            out.append(refined)
        out.sort(key=lambda r: float(r.get('score', 0.0)), reverse=True)
        return out

    def summary(self) -> dict:
        scores = [float(x.get('score', 0.0)) for x in self._ledger if 'score' in x]
        return {
            'name': self.NAME,
            'version': self.VERSION,
            'n_ledger': len(self._ledger),
            'mean_score': (sum(scores) / len(scores)) if scores else 0.0,
            'flags': FeatureFlags.PROFILE,
        }

class VoucherPoolCompressor:
    NAME = 'VoucherPoolCompressor'
    VERSION = '1.0'
    DEFAULT_TOP_K = 15

    def __init__(self, query: str = '', *, seed: int = 0) -> None:
        self.query = query or ''
        self.seed = seed
        self._cache: dict[str, dict] = {}
        self._ledger: list[dict] = []

    def evaluate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherPoolCompressor', 'op': 'evaluate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _evaluate_trace(self, result: dict) -> dict:
        row = {'op': 'evaluate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def refine(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherPoolCompressor', 'op': 'refine', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _refine_trace(self, result: dict) -> dict:
        row = {'op': 'refine', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def merge_pools(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherPoolCompressor', 'op': 'merge_pools', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _merge_pools_trace(self, result: dict) -> dict:
        row = {'op': 'merge_pools', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def explain(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherPoolCompressor', 'op': 'explain', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _explain_trace(self, result: dict) -> dict:
        row = {'op': 'explain', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def calibrate(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherPoolCompressor', 'op': 'calibrate', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _calibrate_trace(self, result: dict) -> dict:
        row = {'op': 'calibrate', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def project_band(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherPoolCompressor', 'op': 'project_band', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _project_band_trace(self, result: dict) -> dict:
        row = {'op': 'project_band', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def select_subset(self, payload: dict | None = None, *, strict: bool = False) -> dict:
        payload = payload or {}
        tokens = LexicalBridgeNormalizer.prepare(str(payload.get('query', '') or payload.get('q', '') or ''))
        score = 0.0
        detail = {'class': 'VoucherPoolCompressor', 'op': 'select_subset', 'n_tokens': len(tokens)}
        for i, tok in enumerate(tokens):
            w = RankTables.mass_at(i)
            if any(ch.isdigit() for ch in tok):
                score += 1.25 * w
            elif tok in ResolutionTuning.COLOR_CANON:
                score += ResolutionTuning.ATTR_WEIGHT * w
            else:
                score += ResolutionTuning.TITLE_WEIGHT * w
        price = payload.get('price')
        if isinstance(price, (int, float)):
            score -= ResolutionTuning.PRICE_PENALTY * math.log1p(max(0.0, float(price)))
        services = payload.get('service') or payload.get('services') or []
        if isinstance(services, str):
            services = [s.strip() for s in services.split(',') if s.strip()]
        for svc in services:
            score += ResolutionTuning.SERVICE_BONUS
            detail.setdefault('services', []).append(svc)
        if strict and score < ResolutionTuning.SCORE_FLOOR:
            return {'ok': False, 'score': score, 'detail': detail}
        score = min(ResolutionTuning.SCORE_CEIL, max(0.0, score))
        detail['score'] = round(score, 4)
        return {'ok': True, 'score': score, 'detail': detail, 'tokens': tokens}

    def _select_subset_trace(self, result: dict) -> dict:
        row = {'op': 'select_subset', 'ok': bool(result.get('ok')), 'score': result.get('score', 0.0)}
        self._ledger.append(row)
        if len(self._ledger) > ResolutionTuning.MAX_POOL:
            self._ledger = self._ledger[-ResolutionTuning.MAX_POOL:]
        return row

    def run_pipeline(self, rows: list[dict] | None = None) -> list[dict]:
        rows = rows or []
        out = []
        for i, row in enumerate(rows[:ResolutionTuning.MAX_POOL]):
            key = str(row.get('product_id', i))
            if key in self._cache:
                out.append(self._cache[key])
                continue
            scored = self.evaluate(row, strict=False)
            refined = self.refine({**row, **scored}, strict=False)
            self._cache[key] = refined
            self._evaluate_trace(refined)
            out.append(refined)
        out.sort(key=lambda r: float(r.get('score', 0.0)), reverse=True)
        return out

    def summary(self) -> dict:
        scores = [float(x.get('score', 0.0)) for x in self._ledger if 'score' in x]
        return {
            'name': self.NAME,
            'version': self.VERSION,
            'n_ledger': len(self._ledger),
            'mean_score': (sum(scores) / len(scores)) if scores else 0.0,
            'flags': FeatureFlags.PROFILE,
        }
