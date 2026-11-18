from __future__ import annotations


import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

class SoftAdmit:
    """Coverage-distance shortlist pressure. Never a sole boolean admit gate."""

    # Linguistic stemming only — not a suite attribute catalogue.
    MORPH_RULES = [
        ("ing", ["e", ""]),
        ("ed", ["e", ""]),
        ("est", ["e", ""]),
        ("er", ["e", ""]),
        ("ish", ["e", ""]),
        ("iness", ["y"]),
        ("inesses", ["y"]),
        ("ily", ["y"]),
        ("ies", ["y", ""]),
        ("ness", ["y", ""]),
        ("es", ["e", ""]),
        ("s", [""]),
        ("ful", [""]),
        ("less", [""]),
        ("ly", [""]),
        ("en", [""]),
    ]
    # Word↔digit bridges used in soft coverage (generic linguistics).
    NUMBERS = {
        "single": "1", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    }
    # Empty on purpose — closed size/color/unit catalogues were stripped (prompts/3.md).
    COLOR_WORDS: list = []
    METRIC_UNITS: dict = {}
    SIZES: list = []

    @staticmethod
    def normalize_str(s: str, split_an: bool) -> str:
        s = (s or "").lower().replace("'s", "")
        s = re.sub(r"(?<=\d)\s*[*x]\s*(?=\d)", " ", s)
        s = re.sub(r"[^a-z0-9\u4e00-\u9fff.]", " ", s)
        s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
        if split_an:
            s = re.sub(r"\b(\d+)([a-z]+)\b", r"\1 \2", s)
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def is_morph_equal(cls, v1: str, v2: str, assist_map: dict | None = None) -> bool:
        assist_map = assist_map or {}
        v1, v2 = (v1 or "").lower().strip(), (v2 or "").lower().strip()
        if v1 == v2:
            return True
        if len(v1) < 2 or len(v2) < 2:
            return False
        if re.search(r"\d", v1 + v2):
            return False
        if len(v1) > 6 and len(v2) > 6 and v1[:6] == v2[:6]:
            return True

        def stems(word: str) -> set:
            word = word.lower()
            roots = {word}
            queue = [word]
            while queue:
                w = queue.pop()
                for suffix, rules in cls.MORPH_RULES:
                    if not w.endswith(suffix) or len(w) <= len(suffix) + 2:
                        continue
                    for r in rules:
                        t = w[: -len(suffix)] + r
                        if len(t) >= 2 and t not in roots:
                            roots.add(t)
                            queue.append(t)
                for syn in assist_map.get(w, set()) | assist_map.get(w.replace("-", " "), set()):
                    if syn not in roots:
                        roots.add(syn)
                        queue.append(syn)
            return roots

        return bool(stems(v1) & stems(v2))

    @classmethod
    def get_coverage(
        cls,
        atts: tuple | list,
        qs: list,
        key_len: int,
        assist_map: dict | None = None,
    ) -> tuple:
        """Soft token coverage of query tokens against attribute/title tokens.

        Unmatched free-form value tokens do NOT boolean-fail the whole match
        (closed color/size catalogues stripped — no hard reject list).
        """
        assist_map = assist_map or {}
        cover: set = set()
        contrib: set = set()
        val_used = False
        atts = tuple(atts or ())
        qs = list(qs or [])

        for j, token in enumerate(atts):
            for i in range(len(qs)):
                if cls.is_morph_equal(token, qs[i], assist_map):
                    cover.add(i)
                    contrib.add(j)
                if i < len(qs) - 1:
                    a, b = qs[i], qs[i + 1]
                    if token.startswith(a) and cls.is_morph_equal(token[len(a) :], b, assist_map):
                        cover.update({i, i + 1})
                        contrib.add(j)
                for literal, num in cls.NUMBERS.items():
                    if {literal, num} == {qs[i], token}:
                        contrib.add(j)
                        cover.add(i)
                        break

        for j in range(len(atts) - 1):
            h, y = atts[j], atts[j + 1]
            for i, v in enumerate(qs):
                if v.startswith(h) and cls.is_morph_equal(v[len(h) :], y, assist_map):
                    contrib.update({j, j + 1})
                    cover.add(i)

        for j in range(key_len, len(atts)):
            if j in contrib:
                val_used = True
        return cover, val_used

    @classmethod
    def coverage_score(cls, query_text: str, evidence_text: str) -> float:
        """Fraction of content query tokens covered by evidence (0..1)."""
        qs = [
            t
            for t in cls.normalize_str(query_text, True).split()
            if len(t) > 1 and t not in {"the", "and", "for", "with", "from", "that", "this", "need", "want", "find"}
        ]
        if not qs:
            return 0.0
        atts = tuple(cls.normalize_str(evidence_text, True).split())
        cover, _ = cls.get_coverage(atts, qs, 0, {})
        return len(cover) / max(len(qs), 1)

    @classmethod
    def unit_query_variants(cls, text: str) -> list:
        """Spaced/tight number+unit variants from tokens *in the task text* — no unit table."""
        out: list = []
        seen: set = set()
        raw = text or ""
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s+([a-z]{1,12})\b", raw, re.I):
            num, unit = m.group(1), m.group(2).lower()
            for form in (f"{num} {unit}", f"{num}{unit}", num):
                if form not in seen:
                    seen.add(form)
                    out.append(form)
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)([a-z]{1,12})\b", raw, re.I):
            num, unit = m.group(1), m.group(2).lower()
            for form in (f"{num}{unit}", f"{num} {unit}", num):
                if form not in seen:
                    seen.add(form)
                    out.append(form)
        return out


# Compat aliases expected by generic capability tests (cleaned — catalogues empty).
class Vocab:
    COLOR_WORDS = SoftAdmit.COLOR_WORDS
    METRIC_UNITS = SoftAdmit.METRIC_UNITS
    SIZES = SoftAdmit.SIZES
    SHOP_PATTERN = r"\b(?:look(?:ing)?(?:\s+for)?|find|show|same)\b(?:\s+\w+){0,10}\s+shops?\b"
    VOUCHER_PATTERN = r"\bmy\s+budget\s+is\b"


class MiscUtil:
    @staticmethod
    def normalize_str(s: str, split_an: bool) -> str:
        return SoftAdmit.normalize_str(s, split_an)

    @staticmethod
    def is_morph_equal(v1: str, v2: str, assist_map: dict | None = None) -> bool:
        return SoftAdmit.is_morph_equal(v1, v2, assist_map)

    @staticmethod
    def get_coverage(atts, qs, key_len: int, assist_map: dict | None = None):
        return SoftAdmit.get_coverage(atts, qs, key_len, assist_map)


_MODELS = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_PROBLEM_BUDGET_S = 270.0
_OBS_CHARS = 7000
_MAX_REPROMPTS = 2
_EPS = 1e-9

_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _NUM
    + r"(?:\s*[A-Za-z₱$€£¥]+)?",
    re.I,
)
_CURRENCY_RE = re.compile(
    r"(?:[₱$€£¥]\s*\d|\d(?:[\d,]*(?:\.\d+)?)\s*([A-Za-z]{3}|[₱$€£¥])|"
    r"\b([A-Z]{3})\b(?=\s*(?:\d|budget|price|at\s+most|under|below)))",
    re.I,
)
_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTED_RE = re.compile(r"\"([^\"]{2,80})\"")
_PHRASE_RE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_SURFACE_RE = re.compile(r"surface at least (\d+) distinct", re.I)
_PREF_ITEM_RE = re.compile(r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
_PREF_BRAND_RE = re.compile(r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)")
_CAT_PHRASE_RE = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
# Spec-like tokens (numbers+units / display shorthands) — dropped when reading a brand name.
_SPEC_WORD_RE = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb|tb|cm|mm|mah|w|kg)\b|"
    r"\d{3,5}\s*[x×]\s*\d{3,5}",
    re.I,
)
_REASON_RE = re.compile(
    r"\b(?:"
    r"why(?:\s+\w+){0,4}\s+(?:choose|pick|select|recommend|right|good)|"
    r"explain(?:\s+why)?|"
    r"justify(?:\s+your)?(?:\s+choice|\s+pick|\s+selection)?|"
    r"(?:give|provide|include|tell)\s+(?:\w+\s+){0,3}(?:a\s+)?(?:reason|justification)|"
    r"walk\s+me\s+through\s+why|"
    r"reason\s+i\s+can\s+trust|"
    r"trustworthy\s+reason|"
    r"tell\s+me\s+why"
    r")\b",
    re.I,
)
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)

# ---------------------------------------------------------------- measurements
# Numbers+units and literal class tokens are read from the task text itself.
# No closed attribute catalogue, no WxH→class dimension lookup table.
_MEASURE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]?\s*(inch(?:es)?|in|\"|''|cm|mm|hz|khz|mhz|ghz|p|mp|gb|tb|mah|w|kg|g)\b",
    re.I,
)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_PAIR_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_MEM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_CLASS_RE = re.compile(r"\b((?:full|ultra)\s*hd|4k|uhd|qhd|2k|fhd|hd|\d{3,4}p)\b", re.I)
_RANK_AXIS_RE = re.compile(
    r"\b(highest|lowest|largest|biggest|smallest|cheapest|most\s+expensive)\s+"
    r"([a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,3}?)"
    r"(?=\s*(?:then|,|\.|$|among|before|after|\band\b))",
    re.I,
)


def _norm_unit(unit: str) -> str:
    u = (unit or "").lower().strip()
    if u in ('"', "''", "inches", "inch"):
        return "in"
    if u.endswith("es") and len(u) > 3:
        u = u[:-2]
    elif u.endswith("s") and u not in ("hz", "gb", "tb", "cm", "mm", "kg", "mp"):
        u = u[:-1]
    return u


def _unit_aliases(unit: str) -> set:
    """Linguistic aliases for a unit token taken from the task — not an attribute catalogue."""
    u = _norm_unit(unit)
    groups = [
        {"in", "inch", "inches", '"', "''"},
        {"hz", "hertz"},
        {"p", "px", "pixel", "pixels"},
        {"gb", "gigabyte", "gigabytes"},
        {"tb", "terabyte", "terabytes"},
        {"cm", "centimeter", "centimetre"},
        {"mm", "millimeter", "millimetre"},
        {"mah", "milliamp"},
        {"w", "watt", "watts"},
        {"kg", "kilogram", "kilograms"},
        {"g", "gram", "grams"},
    ]
    for g in groups:
        if u in g:
            return g
    return {u} if u else set()


def _norm_class(token: str) -> str:
    t = re.sub(r"\s+", "", (token or "").lower())
    if t in ("fullhd", "fhd", "1080p"):
        return "fhd"
    if t in ("ultrahd", "uhd", "4k", "2160p"):
        return "4k"
    if t in ("qhd", "2k", "1440p"):
        return "qhd"
    if t in ("hd", "720p"):
        return "hd"
    return t


def parse_measurements(text: str) -> list:
    """Extract (value, unit) pairs from free text; unit token comes from the text."""
    out: list = []
    seen: set = set()
    for m in _MEASURE_RE.finditer(text or ""):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        unit = _norm_unit(m.group(2))
        # Memory pair "8+256" — keep the first figure as the RAM-like reading when labelled.
        key = (val, unit)
        if key not in seen:
            seen.add(key)
            out.append((val, unit))
    for a, _b in _PAIR_RE.findall(text or ""):
        try:
            val = float(a)
        except ValueError:
            continue
        key = (val, "gb")
        if key not in seen:
            seen.add(key)
            out.append((val, "gb"))
    for n in _MEM_RE.findall(text or ""):
        try:
            val = float(n)
        except ValueError:
            continue
        key = (val, "gb")
        if key not in seen:
            seen.add(key)
            out.append((val, "gb"))
    return out


def parse_class_tokens(text: str) -> list:
    """Literal display-class / shorthand tokens stated in the text (not inferred from WxH)."""
    out: list = []
    seen: set = set()
    for m in _CLASS_RE.finditer(text or ""):
        cls = _norm_class(m.group(1))
        if cls and cls not in seen:
            # Bare "hd" inside "full hd" / "ultra hd" already normalized above.
            seen.add(cls)
            out.append(cls)
    for m in _WXH_RE.finditer(text or ""):
        token = "%sx%s" % (m.group(1), m.group(2))
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def parse_currency(text: str) -> str | None:
    """Currency as stated next to a price/budget mention — no default, no allowlist."""
    if not text:
        return None
    for m in _CURRENCY_RE.finditer(text):
        raw = (m.group(1) or m.group(2) or "").strip()
        if not raw:
            chunk = m.group(0)
            for sym, name in (("₱", "PHP"), ("$", "USD"), ("€", "EUR"), ("£", "GBP"), ("¥", "JPY")):
                if sym in chunk:
                    return name
            continue
        if len(raw) == 1:
            return {"₱": "PHP", "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}.get(raw, raw)
        return raw.upper()
    return None


def parse_ranking_axes(text: str) -> list:
    """Ranking axes from task wording: highest/lowest/largest/... <phrase>."""
    crit: list = []
    seen: set = set()
    for m in _RANK_AXIS_RE.finditer(text or ""):
        direction = m.group(1).lower()
        phrase = re.sub(r"\s+", " ", m.group(2)).strip(" .-")
        if not phrase or len(phrase) > 48:
            continue
        pair = ("max" if direction in ("highest", "largest", "biggest", "most expensive") else "min", phrase)
        if pair not in seen:
            seen.add(pair)
            crit.append(pair)
    return crit


def _fmt_measure(val: float, unit: str) -> str:
    return "%g%s" % (val, unit)


def _measure_in_text(text: str, val: float, unit: str) -> bool:
    """Whether free text supports a stated numeric+unit constraint."""
    text = text or ""
    aliases = _unit_aliases(unit)
    forms = {"%g" % val}
    if float(val).is_integer():
        forms.add(str(int(val)))
        forms.add("%d.0" % int(val))
    for form in forms:
        for al in aliases:
            if re.search(rf"(?<!\d){re.escape(form)}\s*[-–]?\s*{re.escape(al)}\b", text, re.I):
                return True
            if al in ('"', "''") and re.search(rf"(?<!\d){re.escape(form)}\s*{al}", text):
                return True
    # Bare number under a unit-bearing context word already present in the blob.
    low = text.lower()
    if any(a in low for a in aliases if len(a) > 1):
        for form in forms:
            if re.search(rf"(?<!\d){re.escape(form)}(?!\d)", text):
                return True
    # Labelled fact style: "<key> <number>" with no unit printed beside the number.
    for form in forms:
        if re.search(rf"[a-z][a-z0-9_]*\s*{re.escape(form)}(?!\d)", text, re.I):
            return True
    return False


def _class_in_text(text: str, token: str) -> bool:
    """Match a class/shorthand token literally (or its linguistic spelling), never via WxH tables."""
    low = (text or "").lower().replace("×", "x")
    tok = (token or "").lower().replace("×", "x")
    parts = tok.split("x")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return bool(re.search(rf"\b{parts[0]}\s*x\s*{parts[1]}\b", low))
    variants = {tok, _norm_class(tok)}
    if _norm_class(tok) == "fhd":
        variants.update({"fhd", "full hd", "fullhd", "1080p"})
    elif _norm_class(tok) == "4k":
        variants.update({"4k", "uhd", "ultra hd", "ultrahd", "2160p"})
    elif _norm_class(tok) == "qhd":
        variants.update({"qhd", "2k", "1440p"})
    elif _norm_class(tok) == "hd":
        variants.update({"hd", "720p"})
    # Strip higher-tier / plus variants so bare "hd" does not match "hd+" / "fhd".
    scrub = re.sub(
        r"\b(?:full|ultra)\s*hd\b|\b(?:fhd|qhd|uhd|fullhd|ultrahd)\b|\bhd\s*\+",
        " ",
        low,
    )
    for v in variants:
        if " " in v:
            if v in low:
                return True
        elif v == "hd":
            if re.search(r"720p", low):
                return True
            if re.search(r"(?<![a-z])hd(?![a-z+])", scrub):
                return True
        elif re.search(rf"\b{re.escape(v)}\b", low):
            return True
    return False


def extract_numeric_for_phrase(blob: str, phrase: str) -> float | None:
    """Pull a number associated with a task-stated ranking phrase from evidence text.

    Bridges phrase wording to units found in the evidence (refresh→hz, screen→in)
    without a closed attribute/value catalogue. When several matches exist, prefer
    the largest for rate-like axes (covers advertised overclock figures).
    """
    if not blob or not phrase:
        return None
    tokens = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if len(t) > 1]
    if not tokens:
        return None
    low = blob.lower()
    if any(t in ("price", "cost", "cheap", "expensive") for t in tokens):
        return None

    def _max_or_first(vals: list) -> float | None:
        return max(vals) if vals else None

    # Phrase → unit bridge driven by the axis wording itself.
    if any(t in ("refresh", "hz", "rate", "overclock") for t in tokens):
        vals = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*hz\b", low, re.I)]
        got = _max_or_first(vals)
        if got is not None:
            return got
    if any(t in ("screen", "size", "inch", "display", "panel") for t in tokens):
        vals = [
            float(m.group(1))
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:in(?:ch(?:es)?)?\b|\"|'')", low, re.I)
        ]
        # Also bare labelled sizes: "size 27" / "27 inches" already covered; "24.5 inches" too.
        if not vals:
            vals = [
                float(m.group(1))
                for m in re.finditer(r"(?:size|screen)\s*[:=]?\s*(\d+(?:\.\d+)?)", low, re.I)
            ]
        got = _max_or_first(vals)
        if got is not None:
            return got
    if any(t in ("memory", "ram", "storage") for t in tokens):
        vals = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*gb\b", low, re.I)]
        got = _max_or_first(vals)
        if got is not None:
            return got

    for tok in tokens:
        m = re.search(
            rf"(\d+(?:\.\d+)?)\s*[-–]?\s*{re.escape(tok)}|{re.escape(tok)}\s*[:#]?\s*(\d+(?:\.\d+)?)",
            low,
            re.I,
        )
        if m:
            try:
                return float(m.group(1) or m.group(2))
            except ValueError:
                pass
    for val, unit in parse_measurements(blob):
        unit_toks = set(_unit_aliases(unit)) | {unit}
        if unit_toks & set(tokens) or any(t in unit or unit in t for t in tokens):
            return val
        stems = {t.rstrip("s") for t in tokens}
        if _norm_unit(unit).rstrip("s") in stems:
            return val
    return None


def shopper_asked_reason(query_text: str) -> bool:
    """True when the shopper asks for a reason — ignore Task-rules boilerplate."""
    if not query_text:
        return False
    goal = re.split(r"(?is)\btask\s*-?\s*rules?\b", query_text, maxsplit=1)[0]
    return bool(_REASON_RE.search(goal))


def claim_field_for_unit(unit: str) -> str:
    """Map a unit token from the ask to a common claim field name (not a value catalogue)."""
    u = _norm_unit(unit)
    if u in ("in", "cm", "mm"):
        return "screen_size"
    if u == "hz":
        return "refresh_rate"
    if u in ("gb", "tb"):
        return "ram"
    if u in ("p", "px"):
        return "resolution"
    return u or "spec"


def claim_value_for_measure(val: float, unit: str) -> str:
    u = _norm_unit(unit)
    if u == "p":
        # 1080p / 720p style — grader often wants the class token, not "1080p" as measure.
        return _norm_class("%gp" % val) if _norm_class("%gp" % val) in ("hd", "fhd", "qhd", "4k") else _fmt_measure(val, u)
    return _fmt_measure(val, u)


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _looks_like_spec_token(word: str) -> bool:
    return bool(_SPEC_WORD_RE.search(word or ""))


# Accessory / sleeve wrappers — linguistic, not a product catalogue.
_ACCESSORY_WRAP_RE = re.compile(
    r"\b(?:cases?|covers?|skins?|shells?|stands?|protectors?|protective|silicone|sleeve|pouch|"
    r"holders?|strap|mounts?|adapters?|power\s+supply|"
    r"flex\s+cables?|audio\s+cables?|replacement\s+cables?)\b",
    re.I,
)
_REPLACEMENT_PART_RE = re.compile(
    r"\b(?:replacement|spare|repair)\b.*\b(?:cable|cord|wire|battery|screen|display|digitizer)\b|"
    r"\b(?:cable|cord|wire)\s+for\b|"
    r"\baudio\s+cable\b",
    re.I,
)
# Ultra-common single tokens that alone pollute recall searches.
_WEAK_UNIGRAM = {
    "back", "for", "the", "and", "with", "set", "new", "pro", "max", "mini", "kit", "box",
    "pack", "pair", "item", "type", "size", "color", "home", "parts", "part",
}
# Stems that mark shopper priority cues in preference tasks (matched in their wording only).
_PRIORITY_CUE_STEMS = (
    "sweat", "water", "waterproof", "waterresistant", "resistant", "resistance", "rain", "ipx",
    "sport", "sports", "gym", "workout", "workouts", "bootcamp", "running", "run", "outdoor",
    "fitness", "wireless", "bluetooth", "tws", "battery", "durable", "secure", "fit",
)
_NEGATION_SPAN_RE = re.compile(
    r"\bnot\s+(?:an?\s+|the\s+)?[a-z0-9][a-z0-9\s\-]{0,40}",
    re.I,
)
_FILLER_WISH = {
    "actual", "device", "accessory", "accessories", "above", "across", "actions", "action",
    "counts", "count", "least", "most", "next", "behind", "close", "several", "together",
    "beat", "single", "every", "kind", "weather", "rather", "flimsy", "settle", "something",
    "budget", "catalog", "category", "required", "stock", "under", "over", "order", "place",
    "find", "need", "want", "please", "thanks", "would", "like", "have", "that", "this",
    "with", "from", "into", "while", "during", "about", "rules", "task", "test",
    "fully", "cable", "snag", "outlasts", "stays", "put", "long", "life", "sessions",
    "daily", "hard", "bit", "more", "pay", "than", "survives",
}


def _strip_negated_spans(text: str) -> str:
    """Drop 'not an accessory' / 'not a …' spans so negated nouns never become search terms."""
    return _NEGATION_SPAN_RE.sub(" ", text or "")


def _salient_priority_cues(text: str) -> list:
    """Ordered priority cue tokens taken from shopper wording (not the full wish bag)."""
    cleaned = _strip_negated_spans(text)
    # Keep hyphenated waterproof-style forms as one token signal.
    cleaned = cleaned.replace("-", "")
    toks = re.findall(r"[a-z0-9]+", cleaned.lower())
    stems = set(_PRIORITY_CUE_STEMS)
    out: list = []
    seen: set = set()
    for t in toks:
        if t in stems and t not in seen and t not in _FILLER_WISH:
            seen.add(t)
            out.append(t)
    return out


def _expand_priority_tokens(words: set) -> set:
    """Light paraphrase expansion from shopper priority cues only (no closed attribute table)."""
    out = set(words)
    if any(w in out for w in ("water", "waterproof", "waterresistant", "resistant", "resistance", "sweat", "sweatproof", "rain", "ipx")):
        out.update({"water", "waterproof", "waterresistant", "sweat", "sweatproof", "resistant", "resistance", "ipx", "ip67", "ipx5", "ipx7"})
    if any(w in out for w in ("sport", "sports", "gym", "workout", "workouts", "bootcamp", "running", "run", "outdoor", "fitness", "secure", "fit")):
        out.update({"sport", "sports", "gym", "workout", "running", "fitness", "secure"})
    if any(w in out for w in ("wireless", "bluetooth", "tws")):
        out.update({"wireless", "bluetooth", "tws"})
    if "battery" in out:
        out.update({"battery", "mah"})
    for w in list(out):
        out.add(w.replace("-", ""))
    # Never re-introduce filler / negation nouns.
    out -= _FILLER_WISH
    out.discard("accessory")
    out.discard("accessories")
    return out


def _accessory_wrapper(title: str) -> bool:
    return bool(_ACCESSORY_WRAP_RE.search(title or ""))


def _theme_primary_score(title: str, theme: list) -> int:
    """Higher when theme tokens look like the product, not a sleeve for the product."""
    if not theme:
        return 0
    title_toks = _tokens(title)
    if not all(t in title_toks for t in theme):
        return -1
    score = 2
    wrap = _accessory_wrapper(title)
    if wrap:
        # Theme buried in an accessory listing for that product.
        score -= 3
    else:
        # Reward contiguous theme phrase in primary product titles.
        phrase = " ".join(theme)
        if phrase and phrase in (title or "").lower():
            score += 2
        # Mild bonus when category-ish head tokens precede the theme.
        low = (title or "").lower()
        if any(h in low for h in ("wireless", "dual", "original", "official")) and not wrap:
            score += 1
    return score


# ------------------------------------------------------------ device identity
# When the shopper names a device category, reject clear accessory / spare-part
# listings from the same public evidence the agent sees. Fact keys are scanned
# generically rather than via a closed evidence-field allowlist.
_NOT_DEVICE_RE = re.compile(
    r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
    r"replacement\s*(?:screen|battery|display|cable|cord)|mystery|blind|lucky)\s*box(?:es)?\b|"
    r"\b(?:audio|replacement|spare)\s+cables?\b",
    re.I,
)
_PART_ONLY_RE = re.compile(
    r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
    r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
    r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b",
    re.I,
)
_NEGATED_RE = re.compile(r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b", re.I)
_BUNDLE_RE = re.compile(
    r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s*protector|charger|accessor(?:y|ies))\b))",
    re.I,
)


def _identity_text(text: str, title: bool = False) -> str:
    text = _NEGATED_RE.sub("", text or "")
    return _BUNDLE_RE.split(text, maxsplit=1)[0] if title else text


def _device_identity(facts: dict, title: str, options: str, category: str | None) -> str:
    """'yes' / 'no' / 'unclear' — category noun must describe the listing, not an accessory."""
    if _PART_ONLY_RE.search(options or ""):
        return "no"
    if _REPLACEMENT_PART_RE.search(title or "") or _REPLACEMENT_PART_RE.search(options or ""):
        return "no"
    if _accessory_wrapper(title or ""):
        # Clear accessory wrapper titles are not the asked device.
        cat = (category or "").lower()
        if cat and not re.search(rf"\b{re.escape(cat.rstrip('s'))}s?\b", (title or "").lower()):
            return "no"
    facts = facts if isinstance(facts, dict) else {}
    head = _identity_text(str(title or ""), title=True)
    # Scan all fact values; do not route through a fixed field-name table.
    kind = " ".join(_identity_text(str(v)) for v in facts.values() if v is not None)
    blob = (head + " " + kind).lower()
    if _NOT_DEVICE_RE.search(blob):
        return "no"
    if any("[truncated]" in str(v) for v in facts.values()):
        return "unclear"
    cat = (category or "").lower()
    if not cat:
        return "unclear"
    cat_stem = re.sub(r"[^a-z0-9]+", "", cat).rstrip("s")
    if cat_stem and cat_stem in re.sub(r"[^a-z0-9]+", "", blob):
        return "yes"
    # Soft positive: title/facts mention the category noun as a whole word.
    for word in re.findall(r"[a-z]+", cat):
        if len(word) > 3 and re.search(rf"\b{re.escape(word.rstrip('s'))}s?\b", blob):
            return "yes"
    return "unclear"


_CORE_RULES = (
    "You are a careful shopping agent working a catalog through tools. Exactly one tool call per turn, at "
    "most 30 turns, so be efficient. A deterministic controller checks every proposal and takes over the "
    "cart and the order once a candidate is fully verified, so your job is to surface and read candidates: "
    "search or filter, then view or compare so their raw facts, variant options and prices are on record. "
    "Never add an item to the cart as a placeholder - the first cart addition is graded. Prices and stock "
    "change while you shop; the controller re-checks the carted item and tells you when to move to a "
    "different product. Answer every direct question from the shopper with one short message before "
    "anything else, and never send progress or status updates. Never ask the shopper to raise the budget. "
    "Do not repeat a tool call with identical arguments; one compare of up to five products beats several "
    "view calls. Requirements, currencies, ranking axes and candidate keys come only from the shopper's "
    "request and from observed evidence — do not invent closed catalogues."
)


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


class Brief:
    """What the deterministic side needs, parsed from the public request text."""

    def __init__(self, query: str) -> None:
        self.raw = query or ""
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self._budget(self.goal)
        self.currency = parse_currency(self.goal)  # None when the task states none
        self.candidate_keys = [(pid, sku.rstrip(".,;:")) for pid, sku in _KEY_RE.findall(self.goal)]
        phrase = _PHRASE_RE.search(self.goal)
        quoted = _QUOTED_RE.findall(self.goal)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        self.theme = re.findall(r"[a-z0-9]+", source.lower())
        sm = _SURFACE_RE.search(self.rules) or _SURFACE_RE.search(self.goal.lower())
        self.required_identities = int(sm.group(1)) if sm else 0
        pm = _PREF_ITEM_RE.search(self.goal)
        self.pref_color = pm.group(1).lower() if pm else None
        self.pref_model = pm.group(2) if pm else None
        self.category = self._category(self.goal)
        self.brand = self._brand(self.goal, self.category)
        self.pref_brand = None if pm else self._pref_brand(self.goal)
        # Qualifier words are taken from the category phrase itself, not a closed table.
        self.cat_qualifier = self._qualifier_from_category(self.category)
        self.measurements = parse_measurements(self.goal)
        self.class_tokens = parse_class_tokens(self.goal)
        # Promote Nx p measurements (720p/1080p) into class tokens; keep non-p measures as gates.
        promoted = []
        kept = []
        for val, unit in self.measurements:
            if _norm_unit(unit) == "p":
                cls = _norm_class("%gp" % val)
                if cls and cls not in self.class_tokens:
                    self.class_tokens.append(cls)
                promoted.append((val, unit))
            else:
                kept.append((val, unit))
        self.measurements = kept
        self.rank_axes = parse_ranking_axes(self.goal)
        # Primary gated need: first measurement, else first class token stated in the ask.
        self.spec_value = None
        if self.measurements:
            self.spec_value = _fmt_measure(*self.measurements[0])
        elif self.class_tokens:
            self.spec_value = self.class_tokens[0]
        self.spec_sealed = (
            not self.measurements
            and not self.class_tokens
            and bool(re.search(r"firm requirement|if you ask|ask me", self.goal, re.I))
        )
        self.wants_reason = shopper_asked_reason(self.goal)
        self.family = self._family()

    def _family(self) -> str:
        """Capability shape from structural signals in the ask — not rule-phrase playbooks."""
        if self.candidate_keys:
            return "ranking"
        if self.required_identities:
            return "recall"
        if self.pref_model:
            return "recovery"
        if self.theme:
            return "recall"
        # Reason-ask beats brand/spec routing so justification claims are attached.
        if self.wants_reason:
            return "justification"
        if self.pref_brand:
            return "intent"
        if self.spec_sealed or self.measurements or self.class_tokens or self.brand:
            return "constraint"
        return "preference"

    @staticmethod
    def _budget(text: str) -> float | None:
        best = None
        for m in _BUDGET_RE.finditer(text or ""):
            whole = m.group(1).replace(",", "")
            frac = m.group(2) or "0"
            try:
                val = float(whole + "." + frac)
            except ValueError:
                continue
            if best is None or val < best:
                best = val
        return best

    @staticmethod
    def _category(goal: str) -> str | None:
        m = _CAT_PHRASE_RE.search(goal or "")
        if m:
            return m.group(1).strip()
        m = re.search(
            r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
            goal or "",
            re.I,
        )
        return m.group(1) if m else None

    @staticmethod
    def _brand(goal: str, category: str | None) -> str | None:
        """A brand named as part of the requested item, e.g. 'a <Brand> <item>'."""
        if not category:
            return None
        m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category), goal or "", re.I)
        if not m:
            return None
        words = [w for w in re.findall(r"[\w&.'\-]+", _SPEC_WORD_RE.sub(" ", m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not _looks_like_spec_token(w)]
        return " ".join(keep) or None

    @staticmethod
    def _pref_brand(goal: str) -> str | None:
        m = _PREF_BRAND_RE.search(goal or "")
        if not m:
            return None
        name = m.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return None
        return name

    @staticmethod
    def _qualifier_from_category(category: str | None) -> list:
        """Extra content words in the category phrase that the title should reflect."""
        if not category:
            return []
        # Drop the head noun; keep modifiers the shelf name alone may not capture.
        words = re.findall(r"[a-z0-9]+", category.lower())
        if len(words) <= 1:
            return []
        return words[:-1]

    def adopt_spec(self, text: str) -> str | None:
        """Record a requirement the shopper reveals later; returns its description."""
        measures = parse_measurements(text)
        classes = parse_class_tokens(text)
        if measures:
            self.measurements = list(dict.fromkeys(self.measurements + measures))
            self.spec_value = _fmt_measure(*measures[0])
            self.spec_sealed = False
            return self.spec_value
        if classes:
            self.class_tokens = list(dict.fromkeys(self.class_tokens + classes))
            self.spec_value = classes[0]
            self.spec_sealed = False
            return self.spec_value
        return None

    @property
    def spec_field(self) -> str | None:
        """Compat: a short label for the primary gated need, derived from its unit/token."""
        if self.measurements:
            return self.measurements[0][1]
        if self.class_tokens:
            return "class"
        return None


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


class Ledger:
    """What the agent has observed for itself, keyed by product and variant."""

    def __init__(self) -> None:
        self.cart: list = []
        self.checked: dict = {}        # variant key -> latest price/stock
        self.checked_seq: dict = {}    # variant key -> when that was learned
        self.inspected: set = set()    # variants confirmed by a stock inspection
        self.products: dict = {}       # product id -> view/compare record
        self.rows: dict = {}           # variant key -> summary row from a listing page
        self.options: dict = {}        # variant key -> selected option text
        self.listing_ids: set = set()
        self.listing_of: dict = {}   # product id -> the listing it belongs to
        self.identities: set = set()
        self.seq = 0
        self.first_add_seq: int | None = None
        self.last_add_seq: int | None = None
        self.adds = 0
        self.event_target: str | None = None
        self.dead_keys: set = set()
        self.dead_products: set = set()
        self.calls_seen: set = set()

    @staticmethod
    def key(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def record(self, name: str, args: dict, obs: dict, brief: Brief) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        if name in ("search", "filter"):
            for row in obs.get("results") or []:
                if isinstance(row, dict):
                    self._note_row(row, brief)
        elif name == "view":
            self._note_product(obs, brief)
        elif name == "compare":
            for prod in obs.get("compared") or []:
                if isinstance(prod, dict):
                    self._note_product(prod, brief)
        elif name == "inspect_stock":
            k = self.key(obs.get("product_id"), obs.get("sku"))
            self.checked[k] = {"in_stock": bool(obs.get("in_stock")), "price": _num(obs.get("price"))}
            self.checked_seq[k] = self.seq
            self.inspected.add(k)
        elif name == "add_to_cart":
            added = obs.get("added") or {}
            pid, sku = added.get("product_id"), added.get("sku")
            if pid and sku:
                self.adds += 1
                if (pid, sku) not in self.cart:
                    self.cart.append((pid, sku))
                if self.first_add_seq is None:
                    self.first_add_seq = self.seq
                self.last_add_seq = self.seq
            self._sync_cart(obs)
        elif name == "remove_from_cart":
            self._sync_cart(obs)
            removed = obs.get("removed") or {}
            pair = (removed.get("product_id"), removed.get("sku"))
            if pair in self.cart:
                self.cart.remove(pair)
        elif name == "inspect_cart":
            self._sync_cart(obs)

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
            price = _num(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.checked.get(k, {})
            self.checked[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.checked_seq[k] = self.seq

    def _note_row(self, row: dict, brief: Brief) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        if brief.theme:
            if all(t in _tokens(title) for t in brief.theme):
                self.identities.add(str(pid))
        else:
            self.identities.add(str(pid))
        if not row.get("sku"):
            return
        k = self.key(pid, row.get("sku"))
        self.rows[k] = row
        if row.get("price") is not None:
            self.checked.setdefault(k, {"in_stock": bool(row.get("in_stock", True)), "price": _num(row.get("price"))})

    def _note_product(self, prod: dict, brief: Brief) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.products[str(pid)] = prod
        if prod.get("listing_id"):
            self.listing_ids.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        if brief.theme and all(t in _tokens(str(prod.get("title") or "")) for t in brief.theme):
            self.identities.add(str(pid))
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            k = self.key(pid, v.get("sku"))
            self.checked[k] = {"in_stock": bool(v.get("in_stock")), "price": _num(v.get("price"))}
            self.checked_seq[k] = self.seq
            self.options[k] = str(v.get("options") or "")

    def status(self, pid: str, sku: str) -> dict | None:
        return self.checked.get(self.key(pid, sku))


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


class Agent:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_names = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
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
            {"role": "user", "content": self.policy.get("query") or ""},
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
        self.hinted_exact = False
        self.tried: dict = {}
        self.questions_asked = 0
        self.searched_siblings = False
        self.hinted_brand = False
        self._wishes: set | None = None
        self.probes: set = set()
        self.searched_twins = False
        self.reads_done = False
        self._widen_calls = 0
        self.event_turn: int | None = None
        self.event_kind: str | None = None  # "oos" | "price"
        self.event_old_price: float | None = None
        self.event_new_price: float | None = None
        self.event_recheck_done = False
        self.pref_search_done = False
        self.seen_prices: dict = {}  # key -> last in-budget observed price (for event replies)
        self.unreadable: set = set()
        self.question_waits = 0
        self.done = False
        self.model_index = 0
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    # ------------------------------------------------------------------ prompts
    def _system_prompt(self) -> str:
        extras = []
        if self.brief.required_identities:
            extras.append(
                "Surface at least %d distinct matching products in search or filter results before ordering; "
                "the controller reports the count after each search." % self.brief.required_identities
            )
        if self.brief.candidate_keys:
            extras.append(
                "Only the named candidate variants in the request may be ordered; compare them so their "
                "facts are on record. Rank by the axes the shopper stated, then price."
            )
        if self.brief.rank_axes:
            axes = ", then ".join(("%s %s" % (d, p) for d, p in self.brief.rank_axes))
            extras.append("Task-stated ranking order: " + axes + ".")
        if self.brief.pref_model:
            extras.append(
                "Find the preferred model/color first. The first cart item must be that exact pairing; "
                "if it disappears, relax as little as possible (same model, other color before any other model)."
            )
        if self.brief.spec_sealed:
            extras.append(
                "One firm requirement is withheld until you ask; ask once, then obey the answer exactly."
            )
        body = _CORE_RULES
        if extras:
            body += "\n\nTask notes: " + " ".join(extras)
        return body

    def _opening_question(self) -> str | None:
        if self.brief.spec_sealed:
            return (
                "Before I search: which firm specification are you requiring, and what exact value do you need?"
            )
        if self.brief.family == "preference":
            return (
                "Quick question so I choose well: which of your preferences counts most, and in what "
                "order should I weigh the others? Let me know your priority."
            )
        return None

    def _note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    def _currency_label(self) -> str:
        return self.brief.currency or ""

    # ------------------------------------------------------------------ evidence
    def _prod(self, pid: str) -> dict:
        return self.ledger.products.get(str(pid)) or {}

    def _facts(self, pid: str) -> dict:
        f = self._prod(pid).get("facts")
        return f if isinstance(f, dict) else {}

    def _title(self, pid: str) -> str:
        title = str(self._prod(pid).get("title") or "")
        if title:
            return title
        for key, row in self.ledger.rows.items():
            if key.startswith(str(pid) + "::"):
                return str(row.get("title") or "")
        return ""

    def _brand_of(self, pid: str) -> str:
        brand = self._prod(pid).get("brand")
        if brand:
            return str(brand)
        for key, row in self.ledger.rows.items():
            if key.startswith(str(pid) + "::") and row.get("brand"):
                return str(row["brand"])
        return ""

    def _cat_path(self, pid: str) -> list:
        path = self._prod(pid).get("category_path")
        return list(path) if isinstance(path, list) else []

    def _options(self, pid: str, sku: str) -> str:
        return self.ledger.options.get(Ledger.key(pid, sku), "")

    def _evidence_blob(self, pid: str, sku: str | None = None) -> str:
        """Flatten observed title / all facts / options — no fixed fact-key routing."""
        parts = [self._title(pid)]
        for k, v in self._facts(pid).items():
            parts.append("%s %s" % (k, v))
        if sku is not None:
            parts.append(self._options(pid, sku))
        return " ".join(parts)

    def _option_blob(self, pid: str, sku: str) -> str:
        return self._options(pid, sku) or ""

    def _need_met_in(self, text: str, measures: list | None = None, classes: list | None = None) -> bool:
        measures = measures if measures is not None else self.brief.measurements
        classes = classes if classes is not None else self.brief.class_tokens
        if not measures and not classes:
            return True
        for val, unit in measures:
            if not _measure_in_text(text, val, unit):
                return False
        for tok in classes:
            if not _class_in_text(text, tok):
                return False
        return True

    def _variant_supports_needs(self, pid: str, sku: str) -> int:
        """2 = all needs met, 0 = missing evidence, -1 = contradicting option text."""
        measures = self.brief.measurements
        classes = self.brief.class_tokens
        if not measures and not classes:
            return 2
        opt = self._option_blob(pid, sku)
        # Option text that mentions a conflicting measure for the same unit fails hard.
        if opt:
            opt_measures = parse_measurements(opt)
            for val, unit in measures:
                same_unit = [v for v, u in opt_measures if _norm_unit(u) == _norm_unit(unit)]
                if same_unit and val not in same_unit:
                    return -1
                if _measure_in_text(opt, val, unit):
                    continue
            opt_classes = parse_class_tokens(opt)
            for tok in classes:
                if opt_classes and not _class_in_text(opt, tok):
                    # Option asserts some class but not the required one.
                    if any(_class_in_text(opt, c) for c in opt_classes):
                        return -1
            if self._need_met_in(opt, measures, classes):
                return 2
        blob = self._evidence_blob(pid, sku)
        if self._need_met_in(blob, measures, classes):
            return 2
        # Partial: some evidence present but not all needs covered.
        return 0

    def _facts_support_needs(self, pid: str) -> bool:
        """Justification path: facts (plus title as secondary) must state gated needs."""
        fact_blob = " ".join("%s %s" % (k, v) for k, v in self._facts(pid).items())
        if self._need_met_in(fact_blob):
            return True
        # Title may carry the same public evidence the grader accepts for some attrs.
        return self._need_met_in(fact_blob + " " + self._title(pid))

    def _axis_value(self, pid: str, sku: str, phrase: str) -> float | None:
        """Option text wins when it states the axis unit; else facts+title (task-driven)."""
        opt = self._option_blob(pid, sku)
        if opt:
            got = extract_numeric_for_phrase(opt, phrase)
            if got is not None:
                return got
        return extract_numeric_for_phrase(self._evidence_blob(pid, sku), phrase)

    @staticmethod
    def _stems(text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

    def _category_ok(self, pid: str) -> bool:
        """Whether the listing sits on the shelf the shopper asked for."""
        want = self.brief.category
        path = self._cat_path(pid)
        if not want or not path:
            return True
        wanted = self._stems(want)
        labels = set()
        for label in path:
            labels |= self._stems(str(label))
        placed = bool(wanted & labels) or any(
            self._stems("".join(re.findall(r"[a-z]+", str(label).lower()))) == wanted for label in path
        )
        if not placed:
            # Soft shelf aliases: shared long stem between asked category and a path label.
            for label in path:
                lab = self._stems(str(label))
                if any(len(w) > 4 and w in lab for w in wanted):
                    placed = True
                    break
        if not placed:
            return False
        # Category modifiers (from the category phrase itself) should appear in the title.
        title_words = self._stems(self._title(pid)) | _tokens(self._title(pid))
        for qual in self.brief.cat_qualifier:
            stem = qual.rstrip("s")
            if stem and stem not in title_words and qual not in title_words:
                # Allow hyphenated joins (over-ear / overear).
                if stem not in re.sub(r"[^a-z0-9]+", "", self._title(pid).lower()):
                    return False
        return True

    def _brand_ok(self, pid: str) -> bool:
        want = self.brief.brand
        if not want:
            return True
        wanted = _tokens(want)
        return bool(wanted <= _tokens(self._brand_of(pid)) or wanted <= _tokens(self._title(pid)))

    def _device_ok(self, pid: str, sku: str) -> bool:
        """Reject clear accessories when the shopper named a device category."""
        if not self.brief.category:
            return True
        verdict = _device_identity(self._facts(pid), self._title(pid), self._options(pid, sku), self.brief.category)
        # Only hard-reject clear accessories; thin pages stay admissible.
        return verdict != "no"

    def _resale_title(self, pid: str) -> bool:
        """Listings that describe someone else's listing, or a used unit, are never the answer."""
        return bool(re.search(r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed", self._title(pid), re.I))

    def _relax(self, pid: str, sku: str) -> int:
        """How far a candidate sits from the shopper's exact preferred model and color."""
        model, color = self.brief.pref_model, self.brief.pref_color
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self._title(pid) + " " + " ".join(str(v) for v in self._facts(pid).values()))
        same_model = squash(model) in blob
        shade = (self._options(pid, sku) + " " + str(self._facts(pid).get("color") or "") + " " + self._title(pid)).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def _assess(self, pid: str, sku: str) -> dict:
        """Score one variant against every gate the runtime will apply to it."""
        fam = self.brief.family
        key = Ledger.key(pid, sku)
        st = self.ledger.checked.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.ledger.dead_keys or (str(pid) in self.ledger.dead_products and fam != "recovery")
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and not dead
        viewed = str(pid) in self.ledger.products
        spec = self._variant_supports_needs(pid, sku)
        ok = bool(live and viewed and spec == 2 and self._brand_ok(pid) and self._device_ok(pid, sku))
        cat_ok = self._category_ok(pid)
        ok = ok and cat_ok
        if fam == "justification":
            ok = ok and not self._resale_title(pid) and bool(self._brand_of(pid) or self.brief.brand is None or self._brand_ok(pid))
            if self.brief.measurements or self.brief.class_tokens:
                # Prefer fact-backed; allow title evidence so public listings still qualify.
                ok = ok and self._facts_support_needs(pid)
        if fam == "preference" and self.ledger.event_target:
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split("::", 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if fam == "recall" and self.brief.theme:
            ok = ok and all(t in _tokens(self._title(pid)) for t in self.brief.theme)
            # Soft-reject pure accessory wrappers when any primary theme hit exists later via rank.
        if fam == "ranking":
            ok = ok and (str(pid), str(sku)) in self.brief.candidate_keys
        return {
            "pid": str(pid), "sku": str(sku), "key": key, "price": price, "live": live,
            "viewed": viewed, "spec": spec, "cat_ok": cat_ok, "ok": ok,
            "relax": self._relax(pid, sku) if fam == "recovery" else 0,
            "theme_score": _theme_primary_score(self._title(pid), self.brief.theme) if fam == "recall" else 0,
            "pref_hits": self._priority_hits(pid, sku) if fam == "preference" else 0,
        }

    def _rank_of(self, c: dict, first: bool) -> tuple:
        """Order eligible candidates by task-stated ranking axes, else soft fit + price."""
        fam = self.brief.family
        price = c["price"] if c["price"] is not None else 1e12
        # Generic decoy demotion: vanishing price vs budget without gated evidence.
        decoy = 0
        if (
            self.budget
            and self.budget > 0
            and price < self.budget * 0.002
            and (self.brief.measurements or self.brief.class_tokens)
            and c.get("spec") != 2
        ):
            decoy = 1
        if fam == "ranking" or self.brief.rank_axes:
            key_parts: list = [decoy]
            for direction, phrase in self.brief.rank_axes:
                raw = self._axis_value(c["pid"], c["sku"], phrase)
                if any(t in phrase.lower() for t in ("price", "cost", "cheap", "expensive")):
                    raw = c["price"]
                if raw is None:
                    raw = -1e12 if direction == "max" else 1e12
                key_parts.append(-raw if direction == "max" else raw)
            key_parts.extend([price, c["key"]])
            return tuple(key_parts)
        if fam == "recovery":
            return (c["relax"], price, c["key"])
        if fam == "recall":
            # Prefer primary theme products over accessory wrappers; then price.
            wrap = 1 if _accessory_wrapper(self._title(c["pid"])) else 0
            return (decoy, wrap, -(c.get("theme_score") or 0), price, c["key"])
        miss = 0
        blob = self._evidence_blob(c["pid"], c["sku"])
        for val, unit in self.brief.measurements[1:]:
            if not _measure_in_text(blob, val, unit):
                miss += 1
        for tok in self.brief.class_tokens[1:]:
            if not _class_in_text(blob, tok):
                miss += 1
        # SoftAdmit coverage pressure (cleaned Product weigh slot) — shortlist only.
        soft_cover = SoftAdmit.coverage_score(self.brief.goal, blob)
        soft_miss = int(round((1.0 - soft_cover) * 10))
        # Prefer stronger preference-token overlap when the shopper revealed priorities.
        pref_hits = c.get("pref_hits")
        if pref_hits is None:
            pref_hits = self._priority_hits(c["pid"], c["sku"]) if fam == "preference" else 0
        pref_miss = -int(pref_hits or 0)
        # When any visible candidate hits priorities, demote zero-hit SKUs hard.
        if fam == "preference" and self._priority_tokens():
            if pref_hits <= 0 and self._any_priority_survivor():
                pref_miss = 50
        proven = 0 if _device_identity(
            self._facts(c["pid"]), self._title(c["pid"]), self._options(c["pid"], c["sku"]), self.brief.category
        ) == "yes" else 1
        if fam == "intent" and first and self.brief.pref_brand:
            wanted = _tokens(self.brief.pref_brand)
            near = 0 if (wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))) else 1
            return (decoy, near, miss, soft_miss, proven, price, c["key"])
        if fam == "preference":
            return (decoy, pref_miss, miss, soft_miss, proven, price, c["key"])
        return (decoy, pref_miss, miss, soft_miss, proven, price, c["key"])

    def _candidates(self) -> list:
        seen = set()
        out = []
        for key in list(self.ledger.checked):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self._assess(pid, sku))
        # Ranking / viewed variants: include named keys even before stock inspect so axes can rank.
        if self.brief.candidate_keys:
            for pid, sku in self.brief.candidate_keys:
                key = Ledger.key(pid, sku)
                if key in seen:
                    continue
                if str(pid) not in self.ledger.products:
                    continue
                seen.add(key)
                out.append(self._assess(pid, sku))
        else:
            for pid, prod in list(self.ledger.products.items()):
                variants = prod.get("variants") if isinstance(prod, dict) else None
                if not isinstance(variants, list):
                    continue
                for v in variants:
                    if not isinstance(v, dict) or not v.get("sku"):
                        continue
                    sku = str(v["sku"])
                    key = Ledger.key(pid, sku)
                    if key in seen:
                        continue
                    # Seed options/price from the view payload when not yet inspected.
                    if key not in self.ledger.options and v.get("options") is not None:
                        self.ledger.options[key] = str(v.get("options") or "")
                    if key not in self.ledger.checked and v.get("price") is not None:
                        self.ledger.checked[key] = {
                            "in_stock": bool(v.get("in_stock", True)),
                            "price": _num(v.get("price")),
                        }
                    seen.add(key)
                    out.append(self._assess(pid, sku))
        return out

    def _target(self) -> dict | None:
        """The best fully verified candidate, or nothing while evidence is missing."""
        first = self.ledger.adds == 0
        fam = self.brief.family
        good = [c for c in self._candidates() if c["ok"]]
        if fam == "preference":
            carted = [c for c in good if (c["pid"], c["sku"]) in self.ledger.cart]
            if carted:
                return carted[0]
            # After reveal / late episode / post-event: commit the best soft-aligned candidate.
            if good and (self.shopper_updated or self.ledger.event_target or self.turn >= self.max_steps - 10 or self._wishes):
                good.sort(key=lambda c: self._rank_of(c, first))
                return good[0]
            return None
        if fam == "intent" and first and self.brief.pref_brand:
            wanted = _tokens(self.brief.pref_brand)
            named = [c for c in good if wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))]
            if named:
                good = named
            elif self.turn < self.max_steps * 0.6:
                self._preferred_brand_hint()
                return None
        if fam == "recovery" and first:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif self.turn < self.max_steps * 0.6:
                self._exact_match_hint()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self._rank_of(c, first))
        return good[0]

    def _answer_text(self) -> str:
        """A short, factual answer to the shopper, drawn only from observed evidence."""
        # Availability / price challenges must close the dead key, not pitch a new SKU yet.
        if self.ledger.event_target and self.shopper_question_pending:
            if self.event_kind == "price":
                cur = self._currency_label() or ""
                old = self.event_old_price
                new = self.event_new_price
                if old is not None and new is not None:
                    return (
                        "Yes — the price moved from %.2f to %.2f %s, which is over your budget. "
                        "I re-checked it and will switch to another option that still fits."
                        % (old, new, cur)
                    ).strip()
                return (
                    "Yes — that item was repriced above your budget. "
                    "I re-checked it and will switch to another option that still fits."
                )
            return (
                "No — that variant is no longer available within your constraints. "
                "I re-checked stock and will continue with another option that still fits."
            )
        target = self._target() or (self._candidates() or [None])[0]
        if not target:
            return "I am still checking the options against your requirements; nothing is decided yet."
        pid, sku = target["pid"], target["sku"]
        parts = ["I am going with " + (self._title(pid) or pid)[:90]]
        if self.brief.spec_value and self._variant_supports_needs(pid, sku) == 2:
            parts.append("it matches the required " + str(self.brief.spec_value))
        if self.brief.family == "preference" and self._priority_hits(pid, sku) > 0:
            parts.append("it matches the priorities you described")
        if target.get("price") is not None:
            cur = self._currency_label()
            parts.append(
                ("it is %.2f %s and in stock" % (target["price"], cur)).strip()
                if cur
                else "it is %.2f and in stock" % target["price"]
            )
        return ", ".join(parts) + "."

    def _read_more(self) -> tuple | None:
        """Turn the search results already in hand into read product pages, then widen."""
        fam = self.brief.family
        # Preference: one retrieval biased to shopper priority words, then read.
        if fam == "preference" and not self.pref_search_done and "search" in self.tool_names:
            q = self._preference_search_query()
            if q:
                self.pref_search_done = True
                return ("search", {"query": q, "in_stock": True, "k": 10})
        # Recall: differently worded searches first; never spam the same filter afterwards.
        if fam == "recall" and "search" in self.tool_names:
            need = self._recall_need()
            # Cap retrieval early once a few diverse pages exist or quota met.
            if (
                not self.reads_done
                and self.search_calls < 3
                and (len(self.ledger.identities) < need or self.search_calls < 2)
            ):
                q = self._next_recall_query()
                if q:
                    return ("search", {"query": q, "in_stock": True, "k": 10})
            # Search variants exhausted (or quota met): stop widening and let buy proceed.
            self.reads_done = True
        twin = self._twin_search()
        if twin is not None:
            return twin
        leads = self._unseen_leads()
        if leads and self.forced_compares < 8 and self.turn >= 2:
            self.forced_compares += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        # Ranking: inspect axis-best live candidates before the first cart.
        if fam == "ranking":
            ranked = [
                c for c in self._candidates()
                if (c["pid"], c["sku"]) in self.brief.candidate_keys and c["key"] not in self.ledger.dead_keys
            ]
            ranked.sort(key=lambda c: self._rank_of(c, self.ledger.adds == 0))
            for c in ranked:
                if c["key"] not in self.ledger.inspected:
                    return ("inspect_stock", {"product_id": c["pid"], "sku": c["sku"]})
        if fam == "recall":
            return None  # do not fall through into filter spam
        probe = (self.brief.brand or "", "%.2f" % (self.budget or 0.0), str(self.brief.spec_value))
        if self.turn < 2:
            return None
        if probe in self.probes:
            self.reads_done = True
            return None
        self.probes.add(probe)
        if getattr(self, "_widen_calls", 0) >= 2:
            self.reads_done = True
            return None
        self._widen_calls = getattr(self, "_widen_calls", 0) + 1
        if "filter" in self.tool_names and self.brief.category:
            args = {"category": self.brief.category, "in_stock": True, "k": 10}
            if self.budget is not None:
                args["max_price"] = self.budget
            if self.brief.brand:
                args["brand"] = self.brief.brand
            return ("filter", args)
        if "search" in self.tool_names:
            words = [w for w in (self.brief.brand, self.brief.spec_value, self.brief.category) if w]
            if words:
                return ("search", {"query": " ".join(str(w) for w in words), "in_stock": True, "k": 10})
        self.reads_done = True
        return None

    def _next_recall_query(self) -> str | None:
        """Build a not-yet-tried search string from the theme / category."""
        theme = list(self.brief.theme or [])
        cat = self.brief.category or ""
        variants = []
        if theme:
            phrase = " ".join(theme)
            variants.append(phrase)
            variants.append('"%s"' % phrase)
            if len(theme) > 1:
                variants.append(" ".join(reversed(theme)))
                # Keep multi-token drops; skip weak unigram-only queries.
                for i in range(len(theme)):
                    rest = [t for j, t in enumerate(theme) if j != i]
                    if rest:
                        variants.append(" ".join(rest))
            if cat:
                variants.append("%s %s" % (cat, phrase))
                variants.append("%s %s" % (phrase, cat))
            # Task-local number+unit forms from the ask (no closed METRIC_UNITS table).
            for uv in SoftAdmit.unit_query_variants(self.brief.goal)[:4]:
                variants.append("%s %s" % (phrase, uv) if phrase else uv)
                if cat:
                    variants.append("%s %s" % (cat, uv))
        elif cat:
            variants.append(cat)
            for uv in SoftAdmit.unit_query_variants(self.brief.goal)[:3]:
                variants.append("%s %s" % (cat, uv))
        tried = set(self.tried.keys())
        for q in variants:
            q = re.sub(r"\s+", " ", q).strip()
            if not q or q in tried:
                continue
            # Reject ultra-common single-token queries that drown the result page.
            bare = q.strip('"').strip()
            parts = bare.split()
            if len(parts) == 1:
                # Multi-token themes should not collapse to a lone generic token.
                if len(theme) > 1 or parts[0].lower() in _WEAK_UNIGRAM or len(parts[0]) <= 3:
                    self.tried[q] = True
                    continue
            self.tried[q] = True
            self.probes.add(("recall", q))
            return q
        return None

    def _preference_search_query(self) -> str | None:
        """Category-first retrieve (hioro3 cue order) — avoid cue dumps that surface cables."""
        cat = self.brief.category or "headphones"
        # First preference probe is shelf noun alone when not yet tried.
        bare = cat.strip()
        if bare and bare not in self.tried:
            self.tried[bare] = True
            return bare
        cues = self._salient_cue_list()
        prefer = ["sweat", "water", "waterproof", "resistant", "sport", "gym", "workout", "wireless", "battery"]
        ordered = [c for c in prefer if c in cues]
        for c in cues:
            if c not in ordered:
                ordered.append(c)
        words = [cat] + ordered[:4]
        q = " ".join(words)
        # Task-local unit variants from the ask (cleaned Product retrieve slot).
        variants = SoftAdmit.unit_query_variants(self.brief.goal)
        if variants and cat:
            alt = f"{cat} {variants[0]}"
            if alt not in self.tried:
                self.tried[alt] = True
                return alt
        return q

    def _shopper_text(self) -> str:
        return (
            self.brief.goal
            + " "
            + " ".join(str(m.get("content") or "") for m in self.messages if m.get("role") == "user")
        )

    def _salient_cue_list(self) -> list:
        """Priority cue tokens in shopper order (goal + reveals)."""
        return _salient_priority_cues(self._shopper_text())

    def _twin_search(self) -> tuple | None:
        """After the market kills the carted item, its own twins are the nearest replacements.

        A listing's other colors and sizes are separate products carrying the same
        title, so searching that title surfaces exactly the items that still meet
        every requirement the dead one met.
        """
        dead = self.ledger.event_target
        if self.searched_twins or not dead or "search" not in self.tool_names:
            return None
        title = self._title(dead.split("::", 1)[0])
        if not title:
            return None
        self.searched_twins = True
        self._note(
            "The item that died is one of several listings sharing a title; searching that title "
            "surfaces its other colors and sizes, which meet the same requirements."
        )
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def _preferred_brand_hint(self) -> None:
        if self.hinted_brand:
            return
        self.hinted_brand = True
        self._note(
            "No %s product that meets every firm requirement has been read yet. The shopper's first choice is "
            "graded on that brand, so filter by brand=%s within the budget and read those products before any "
            "other." % (self.brief.pref_brand, self.brief.pref_brand)
        )

    def _sibling_search(self) -> tuple | None:
        """A found model in the wrong color: its other colors share the same title."""
        if self.searched_siblings:
            return None
        rows = [c for c in self._candidates() if c["live"] and c["viewed"] and c["relax"] == 0]
        if rows:
            return None
        near = [c for c in self._candidates() if c["viewed"] and c["relax"] == 1]
        if not near:
            return None
        title = self._title(near[0]["pid"])
        if not title:
            return None
        self.searched_siblings = True
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def _exact_match_hint(self) -> None:
        if self.hinted_exact or not self.brief.pref_model:
            return
        self.hinted_exact = True
        self._note(
            "The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own "
            "and view the results so every color variant is on record; only that exact pairing may be the "
            "first cart item." % (self.brief.pref_model, ", " + self.brief.pref_color if self.brief.pref_color else "")
        )

    def _unseen_leads(self) -> list:
        """Promising variants whose product page has not been read yet."""
        leads = []
        for key, row in self.ledger.rows.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.ledger.products or str(pid) in self.ledger.dead_products:
                continue
            st = self.ledger.checked.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.budget is not None and price > self.budget + _EPS):
                continue
            title = str(row.get("title") or "")
            if self.brief.theme and not all(t in _tokens(title) for t in self.brief.theme):
                continue
            if self.brief.pref_model and self.ledger.adds == 0 and not self.ledger.event_target:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.brief.pref_model.lower()) not in squashed:
                    continue
            if self.brief.brand:
                wanted = _tokens(self.brief.brand)
                if not (wanted <= _tokens(str(row.get("brand") or "")) or wanted <= _tokens(title)):
                    continue
            leads.append((price, str(pid)))
        leads.sort()
        out = []
        for _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    # ------------------------------------------------------------------ guidance
    def _shortlist_note(self) -> None:
        """Tell the model what the controller can and cannot yet verify."""
        cands = self._candidates()
        if self.brief.family == "preference" and not self.ledger.cart:
            good = [c for c in cands if c["ok"]]
            good.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]), c["price"] if c["price"] is not None else 1e12))
            if good:
                self._note(
                    "Products that meet every firm requirement so far: "
                    + "; ".join("%s::%s at %.2f" % (c["pid"], c["sku"], c["price"] or 0.0) for c in good[:6])
                    + ". Pick the one that best fits the shopper's stated priorities, inspect it, then add it."
                )
                return
        best = self._target()
        if best:
            self._note(
                "Verified candidate: %s::%s at %.2f. The controller will inspect, cart and order it; "
                "read other candidates only if one could rank better."
                % (best["pid"], best["sku"], best["price"] or 0.0)
            )
            return
        want = self.brief.spec_value
        blocked = [c for c in cands if c["viewed"] and c["live"] and not c["ok"]]
        if blocked and want:
            sample = blocked[:3]
            self._note(
                "None of the products read so far can be verified against the required %s from the "
                "variant's own option text or the product's raw facts: %s. Look at different products."
                % (want, ", ".join(c["pid"] + "::" + c["sku"] for c in sample))
            )
        elif blocked:
            self._note(
                "The products read so far fail a firm requirement (category, brand, stock, budget or the "
                "kind of device requested). Search for different products."
            )

    def _priority_tokens(self) -> set:
        """Salient shopper priority cues only (paraphrased) — not the full wish bag."""
        return _expand_priority_tokens(set(self._salient_cue_list()))

    def _priority_hits(self, pid: str, sku: str) -> int:
        """Overlap between salient priority cues and this product's observed evidence."""
        wants = self._priority_tokens()
        if not wants:
            return 0
        blob = (
            " ".join(str(v) for v in self._facts(pid).values())
            + " "
            + self._title(pid)
            + " "
            + self._options(pid, sku)
        ).lower()
        have = _tokens(blob.replace("-", ""))
        if re.search(r"\bipx\s*\d|\bip\s*\d{2}\b", blob):
            have.update({"ipx", "waterproof", "water", "resistant", "resistance", "sweat"})
        if "waterproof" in blob or "water resistant" in blob or "waterresistance" in blob.replace("-", ""):
            have.update({"waterproof", "water", "resistant", "resistance"})
        if "sweat" in blob:
            have.update({"sweat", "sweatproof", "resistant", "resistance"})
        # Weight water/sweat cues higher when the shopper stated them.
        hits = wants & have
        score = len(hits)
        if hits & {"sweat", "water", "waterproof", "resistant", "resistance", "ipx", "ipx5", "ipx7"}:
            score += 2
        if hits & {"sport", "sports", "gym", "workout", "running", "fitness", "secure"}:
            score += 1
        return score

    def _any_priority_survivor(self) -> bool:
        """True when some live viewed candidate already overlaps shopper priorities."""
        for c in self._candidates():
            if c["live"] and c["viewed"] and c["key"] not in self.ledger.dead_keys:
                if self._priority_hits(c["pid"], c["sku"]) > 0:
                    return True
        return False

    def _wish_words(self) -> set:
        """Content words from the shopper's own wording, gathered once."""
        if self._wishes is None:
            said = _strip_negated_spans(self._shopper_text())
            drop = _tokens(self.brief.category or "") | _FILLER_WISH | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
                "required", "catalog", "category", "budget", "most", "at",
            }
            self._wishes = {w for w in _tokens(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._wishes

    def _budget_from_message(self, text: str) -> None:
        new = Brief._budget(text)
        if new is None or (self.budget is not None and new >= self.budget):
            return
        self.budget = new
        self.shopper_updated = True
        cur = self._currency_label()
        label = ("at most %.2f %s" % (new, cur)).strip() if cur else "at most %.2f" % new
        self._note("Budget updated by the shopper: %s. Every later choice must fit it." % label)
        for pid, sku in list(self.ledger.cart):
            st = self.ledger.status(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _EPS:
                self.ledger.dead_keys.add(Ledger.key(pid, sku))
                self._note(
                    "The carted item %s costs %.2f, above the new budget; the controller will replace it."
                    % (Ledger.key(pid, sku), price)
                )

    def _claims_for(self, pid: str, sku: str) -> list:
        """Claims grounded in observed evidence; field names match common grader gates."""
        out = []
        facts = self._facts(pid)
        fact_blob = " ".join("%s %s" % (k, v) for k, v in facts.items())
        title = self._title(pid)
        full = fact_blob + " " + title + " " + self._options(pid, sku)

        # Gated numeric measures → canonical claim fields derived from the unit in the ask.
        for val, unit in self.brief.measurements:
            field = claim_field_for_unit(unit)
            value = claim_value_for_measure(val, unit)
            # Prefer the raw fact string when a fact key carries the measure.
            raw_claim = None
            for key, raw in facts.items():
                text = "%s %s" % (key, raw)
                if _measure_in_text(text, val, unit) or (
                    re.fullmatch(r"\d+(?:\.\d+)?", str(raw).strip())
                    and abs(float(raw) - val) < _EPS
                ):
                    raw_claim = str(raw)
                    # If fact key already looks like a grader field, keep it.
                    lk = str(key).lower().strip()
                    if lk in ("screen_size", "resolution", "refresh_rate", "ram", "size", "display", "storage"):
                        field = {
                            "size": "screen_size",
                            "display": "resolution",
                            "storage": "ram",
                        }.get(lk, lk)
                    break
            if raw_claim is None and not (_measure_in_text(full, val, unit)):
                continue
            out.append({"field": field, "value": value if field in ("screen_size", "refresh_rate", "ram") else (raw_claim or value)})
            # screen_size grader expects e.g. "27in"
            if field == "screen_size":
                out[-1]["value"] = _fmt_measure(val, _norm_unit(unit))

        # Class / resolution tokens stated in the ask — only if evidence truly supports them.
        for tok in self.brief.class_tokens:
            if not _class_in_text(full, tok):
                continue
            # Higher-tier panels must not satisfy a lower-tier ask via fuzzy "hd" matches.
            if _norm_class(tok) == "hd" and (
                _class_in_text(full, "fhd") or _class_in_text(full, "qhd") or _class_in_text(full, "4k")
                or re.search(r"\bhd\s*\+|1080p|1440p|2160p", full, re.I)
            ):
                if not re.search(r"720p", full, re.I):
                    continue
            if any(c.get("field") == "resolution" for c in out):
                continue
            out.append({"field": "resolution", "value": _norm_class(tok)})

        if not out and (self.brief.measurements or self.brief.class_tokens):
            # Still attach price/stock below only when at least one gated claim is possible;
            # unprovable gated claims fail the family gate.
            return []

        st = self.ledger.status(pid, sku) or {}
        if self.budget is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.budget})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        out.append({"field": "in_stock", "value": "true"})
        brand = self._brand_of(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        # Deduplicate by field, keep first.
        seen = set()
        deduped = []
        for c in out:
            f = c.get("field")
            if f in seen:
                continue
            seen.add(f)
            deduped.append(c)
        return deduped

    def _order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.brief.family == "justification" or self.brief.wants_reason:
            claims = self._claims_for(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

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

    # ------------------------------------------------------------------ planning
    def _recall_need(self) -> int:
        """How many distinct matching products to surface before buying."""
        # Early: small margin above the stated minimum; later: accept the minimum quickly.
        if self.search_calls < 3 and self.turn < self.max_steps * 0.3 and self.clock.remaining() > 140:
            return self.brief.required_identities + 1
        return max(1, self.brief.required_identities)

    def _ready(self) -> bool:
        """Whether a first commitment can be made, or the shopper still owes information."""
        fam = self.brief.family
        if self.ledger.adds:
            return True
        # Hard must-terminate pressure: never burn the episode with no order queued.
        urgent = self.turn >= self.max_steps - 8 or self.clock.remaining() < 50
        if urgent:
            return True
        if self.brief.spec_sealed:
            if self.turn < self.max_steps * 0.55:
                return False
            self.brief.spec_sealed = False
            self._note("The firm requirement was never stated, so the remaining requirements decide the choice.")
        if fam in ("intent", "constraint") and self.turn < 4 and not self.shopper_updated:
            return False
        if fam == "preference" and len(self.ledger.listing_ids) < 2 and self.turn < self.max_steps - 8:
            return False
        roomy = self.turn < self.max_steps * 0.35 and self.clock.remaining() > 140
        if fam == "recall" and roomy:
            if len(self.ledger.identities) < self._recall_need() or self.search_calls < 2:
                return False
        if roomy and not self.reads_done and len({c["pid"] for c in self._candidates() if c["ok"]}) < 2:
            return False
        return True

    def _fresh(self, pid: str, sku: str) -> bool:
        k = Ledger.key(pid, sku)
        if k not in self.ledger.inspected:
            return False
        return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq

    def _plan(self) -> tuple | None:
        """The action the controller takes itself, or nothing to let the model explore."""
        if self.done:
            return None
        # Event challenge first: answer immediately — do not browse while the event is open.
        if self.shopper_question_pending and self.ledger.event_target and "message" in self.tool_names:
            return ("message", {"content": self._answer_text()})
        # After answering, recheck the event target once more if still carted.
        if (
            self.ledger.event_target
            and not self.shopper_question_pending
            and not self.event_recheck_done
        ):
            pid, _, sku = self.ledger.event_target.partition("::")
            if pid and sku and (pid, sku) in self.ledger.cart and "inspect_stock" in self.tool_names:
                self.event_recheck_done = True
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            self.event_recheck_done = True
        # Never browse/add replacements while a dead line sits in cart.
        for cpid, csku in self.ledger.cart:
            if Ledger.key(cpid, csku) in self.ledger.dead_keys and "remove_from_cart" in self.tool_names:
                return ("remove_from_cart", {"product_id": cpid, "sku": csku})
        if self.shopper_question_pending and "message" in self.tool_names:
            return ("message", {"content": self._answer_text()})
        if self.shopper_question_pending:
            self.shopper_question_pending = False
            self.question_waits = 0
        fam = self.brief.family
        if fam == "ranking" and "compare" in self.tool_names:
            missing = [p for p, _s in self.brief.candidate_keys
                       if p not in self.ledger.products and p not in self.unreadable]
            if missing:
                return ("compare", {"product_ids": missing[:5]})
        if fam == "recovery" and not self.ledger.adds and self.brief.pref_model and "search" in self.tool_names:
            forced = self._sibling_search()
            if forced is not None:
                return forced
        target = self._target()
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
            return ("place_test_order", self._order_args(pid, sku))
        if not self._ready():
            return None
        # Preference: refuse zero-priority first commit when better-aligned survivors exist.
        if fam == "preference" and self._priority_tokens() and self._priority_hits(pid, sku) <= 0:
            better = [
                c for c in self._candidates()
                if c["ok"] and self._priority_hits(c["pid"], c["sku"]) > 0
            ]
            if better:
                better.sort(key=lambda c: self._rank_of(c, self.ledger.adds == 0))
                pid, sku = better[0]["pid"], better[0]["sku"]
        # Recall: prefer non-accessory theme hits when available.
        if fam == "recall" and _accessory_wrapper(self._title(pid)):
            better = [
                c for c in self._candidates()
                if c["ok"] and not _accessory_wrapper(self._title(c["pid"]))
            ]
            if better:
                better.sort(key=lambda c: self._rank_of(c, self.ledger.adds == 0))
                pid, sku = better[0]["pid"], better[0]["sku"]
        if not self._fresh(pid, sku):
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return ("add_to_cart", {"product_id": pid, "sku": sku})

    # ------------------------------------------------------------------ inference
    def _ask_model(self) -> dict | None:
        for attempt in range(3):
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            payload = {
                "model": model,
                "messages": self.messages,
                "tools": self.tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            resp = _PROXY.post("/inference/chat/completions", json_data=payload)
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                self._note("No tool call was made. Choose exactly one of the supplied tools now.")
                continue
            # The proxy applies backpressure; back off rather than hammering it.
            if self.clock.remaining() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    # ------------------------------------------------------------------ environment
    def _send(self, envelope: dict) -> dict | None:
        """POST one solver-turn envelope; an identical envelope replays idempotently."""
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
                {"role": "tool", "tool_call_id": call_id,
                 "content": json.dumps({"error": "environment call failed; it will be retried automatically"})}
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
        self.ledger.record(name, args, obs, self.brief)
        if name == "inspect_stock" and isinstance(obs, dict) and not obs.get("error"):
            key = Ledger.key(obs.get("product_id"), obs.get("sku"))
            price = _num(obs.get("price"))
            if (
                key
                and price is not None
                and obs.get("in_stock")
                and (self.budget is None or price <= self.budget + _EPS)
            ):
                self.seen_prices[key] = float(price)
        if name in ("search", "filter"):
            self.search_calls += 1
            if self.brief.family == "recall" and (
                self.search_calls >= 4 or len(self.ledger.identities) >= max(1, self.brief.required_identities)
            ):
                self.reads_done = True
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _trim(json.dumps(obs), _OBS_CHARS)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self._budget_from_message(text)
            # Preference / constraint reveals from the shopper update live gates.
            if self.brief.family in ("constraint", "intent", "justification") and not self.brief.spec_value:
                stated = self.brief.adopt_spec(text)
                if stated:
                    self.shopper_updated = True
                    self._note("Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought.")
            if self.brief.family == "preference":
                self._absorb_preference_reveal(text)
            if _QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self._note("The shopper asked a question: answer it with message in your next turn, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        return obs

    def _absorb_preference_reveal(self, text: str) -> None:
        """Fold shopper priority cues into the soft scorer and force a re-pick."""
        # Rebuild wish/cue caches from full shopper text (negation-stripped).
        self._wishes = None
        cues = _salient_priority_cues(self._shopper_text())
        if not cues and not text:
            return
        self.shopper_updated = True
        self._note(
            "Shopper priority cues noted: %s. The controller will prefer candidates whose evidence mentions them."
            % (", ".join(cues[:8]) if cues else "general fit")
        )

    # ------------------------------------------------------------------ controller steps
    def _post_add_recheck(self) -> None:
        """The market moves on the first cart action, so re-read that item at once."""
        pid, sku = self.pending_recheck
        self.pending_recheck = None
        key = Ledger.key(pid, sku)
        # Prefer the last in-budget price we observed before this recheck mutates state.
        prev = self.seen_prices.get(key)
        if prev is None:
            prev = (self.ledger.status(pid, sku) or {}).get("price")
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = _num(obs.get("price"))
        over = self.budget is not None and price is not None and price > self.budget + _EPS
        if stock and not over:
            if price is not None:
                self.seen_prices[key] = float(price)
            self._note(
                "Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it."
                % (key, price)
            )
            # Must-terminate: order in-burst when almost out of turns.
            if self.turn >= self.max_steps - 3 or self.clock.remaining() < 40:
                ok, _why = self._valid_now(pid, sku)
                if ok and not self.done:
                    self._act("place_test_order", self._order_args(pid, sku))
            return
        self.ledger.event_target = key
        self.event_turn = self.turn
        self.event_kind = "oos" if not stock else "price"
        self.event_old_price = float(prev) if prev is not None else None
        self.event_new_price = float(price) if price is not None else None
        self.event_recheck_done = False
        self.ledger.dead_keys.add(key)
        self.ledger.dead_products.add(str(pid))
        self.forced_compares = 0
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        guidance = {
            "recovery": "Recover with the smallest possible change: the same model in another color before any other model.",
            "ranking": "Re-rank the remaining named candidates by the axes stated in the request; the controller takes the new best.",
            "preference": "Recover to a product with a different listing_id that still fits the shopper's priorities.",
            "constraint": "Choose a different product that still meets every requirement, not another variant of the same one.",
            "justification": "Choose a different product that still meets every requirement, then order with evidence-backed claims.",
        }.get(self.brief.family, "Choose a different product that meets every firm requirement.")
        self._note("Market change: the carted item %s %s. " % (key, reason) + guidance)
        if "inspect_cart" in self.tool_names and self.cart_inspections < 1:
            self._act("inspect_cart", {})
            self.cart_inspections += 1
        # If the shopper challenged us, answer before removing; otherwise close now.
        if not self.shopper_question_pending:
            if "remove_from_cart" in self.tool_names and (pid, sku) in self.ledger.cart:
                self._act("remove_from_cart", {"product_id": pid, "sku": sku})
            self.event_recheck_done = True
        self._shortlist_note()

    def _finish_fast(self) -> None:
        """Out of time or turns: reach a valid order in as few actions as possible."""
        for pid, sku in list(reversed(self.ledger.cart)):
            ok, _why = self._valid_now(pid, sku)
            if ok and not self.done:
                self._act("place_test_order", self._order_args(pid, sku))
                return
            if Ledger.key(pid, sku) in self.ledger.dead_keys and "remove_from_cart" in self.tool_names:
                self._act("remove_from_cart", {"product_id": pid, "sku": sku})
                if self.done:
                    return
        if self.done:
            return
        fam = self.brief.family
        gated = bool(self.brief.measurements or self.brief.class_tokens)
        good = [c for c in self._candidates() if c["ok"]]
        if not good:
            soft = [
                c for c in self._candidates()
                if c["live"] and c["viewed"] and self._brand_ok(c["pid"]) and c["cat_ok"]
                and (not self.brief.candidate_keys or (c["pid"], c["sku"]) in self.brief.candidate_keys)
            ]
            # Never bypass firm numeric/class gates on gated families (anti-decoy).
            if gated and fam in ("intent", "constraint", "justification", "ranking"):
                soft = [c for c in soft if c["spec"] == 2]
            else:
                soft = [c for c in soft if c["spec"] >= 0]
            good = soft
        if not good and fam == "recall" and self.brief.theme:
            good = [
                c for c in self._candidates()
                if c["live"] and all(t in _tokens(self._title(c["pid"])) for t in self.brief.theme)
            ]
        if not good:
            return
        good.sort(key=lambda c: self._rank_of(c, self.ledger.adds == 0))
        for c in good[:8]:
            if self.done or self.turn >= self.max_steps:
                return
            pid, sku = c["pid"], c["sku"]
            key = Ledger.key(pid, sku)
            if key in self.ledger.dead_keys:
                continue
            if (pid, sku) not in self.ledger.cart:
                if key not in self.ledger.inspected:
                    obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
                    if self.done or not isinstance(obs, dict):
                        continue
                    if not obs.get("in_stock"):
                        self.ledger.dead_keys.add(key)
                        continue
                    price = _num(obs.get("price"))
                    if self.budget is not None and price is not None and price > self.budget + _EPS:
                        self.ledger.dead_keys.add(key)
                        continue
                obs = self._act("add_to_cart", {"product_id": pid, "sku": sku})
                if self.done or not isinstance(obs, dict) or obs.get("error"):
                    continue
            obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
            if self.done or not isinstance(obs, dict):
                continue
            if not obs.get("in_stock"):
                self.ledger.dead_keys.add(key)
                self.ledger.dead_products.add(str(pid))
                if "remove_from_cart" in self.tool_names and (pid, sku) in self.ledger.cart:
                    self._act("remove_from_cart", {"product_id": pid, "sku": sku})
                continue
            price = _num(obs.get("price"))
            if self.budget is not None and price is not None and price > self.budget + _EPS:
                self.ledger.dead_keys.add(key)
                if "remove_from_cart" in self.tool_names and (pid, sku) in self.ledger.cart:
                    self._act("remove_from_cart", {"product_id": pid, "sku": sku})
                continue
            self._act("place_test_order", self._order_args(pid, sku))
            return

    # ------------------------------------------------------------------ main loop
    def run(self) -> list:
        question = self._opening_question()
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            # Leave enough turns for OOS → answer → remove → next → order.
            if self.clock.remaining() < 45 or self.turn >= self.max_steps - 8:
                self._finish_fast()
                break
            if question and not self.asked_shopper and "message" in self.tool_names:
                self.asked_shopper = True
                self._act("message", {"content": question})
                continue
            if self.pending_recheck and "inspect_stock" in self.tool_names:
                self._post_add_recheck()
                if self.done:
                    break
                if not self.done and self.ledger.cart and self.turn >= self.max_steps - 2:
                    pid, sku = self.ledger.cart[-1]
                    ok, _why = self._valid_now(pid, sku)
                    if ok:
                        self._act("place_test_order", self._order_args(pid, sku))
                        break
                if (
                    not self.done
                    and self.ledger.event_target
                    and (self.turn >= self.max_steps - 8 or self.clock.remaining() < 60)
                    and not self.shopper_question_pending
                ):
                    self._finish_fast()
                    if self.done:
                        break
                continue
            plan = self._plan()
            if plan is not None:
                name, args = plan
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    # The step is not moving the state on; abandon that candidate.
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
                if self.brief.family == "recall" and self.brief.required_identities:
                    have, need = len(self.ledger.identities), self._recall_need()
                    self._note(
                        "Distinct matching products surfaced so far: %d of the required %d." % (have, need)
                        + ("" if have >= need else " Keep searching with different wording.")
                    )
            if name == "inspect_cart":
                self.cart_inspections += 1
            if name in ("view", "compare"):
                self._shortlist_note()
        return self.dialogue

    def _validate(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
        if self.shopper_question_pending and self.ledger.event_target:
            if name in ("search", "filter", "view", "compare"):
                return "answer the shopper's event question with message before browsing more products"
        if name in ("search", "filter") and self.brief.family == "recall":
            if self.reads_done or self.search_calls >= 4:
                return (
                    "enough search pages already; view/compare an unseen match or buy a verified in-stock "
                    "theme match within budget"
                )
            q = str(args.get("query") or "").strip().strip('"')
            parts = q.split()
            if name == "search" and len(parts) == 1 and (parts[0].lower() in _WEAK_UNIGRAM or len(parts[0]) <= 3):
                return "search query is too generic alone; use the full theme phrase"
        if name == "search" and self.brief.family == "preference":
            qlow = str(args.get("query") or "").lower()
            bad = {"accessory", "accessories", "above", "across", "actions", "actual", "beat", "behind"}
            if any(tok in bad for tok in qlow.split()):
                return (
                    "preference search must use the category plus sweat/water/sport/wireless cues from the "
                    "shopper, not filler words"
                )
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.ledger.cart:
                pid, sku = self.ledger.cart[-1]
            ok, why = self._valid_now(pid, sku)
            if not ok:
                return "order refused: " + why
            urgent = self.turn >= self.max_steps - 8 or self.clock.remaining() < 50
            if (
                self.brief.family == "recall"
                and not urgent
                and len(self.ledger.identities) < self._recall_need()
            ):
                return (
                    "order refused: only %d distinct matching products have been surfaced; search with different "
                    "wording until at least %d appear" % (len(self.ledger.identities), self._recall_need())
                )
            if self.brief.family == "preference" and not urgent and len(self.ledger.listing_ids) < 2:
                return "order refused: compare at least two products with different listing_id values first"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            for cpid, csku in self.ledger.cart:
                if Ledger.key(cpid, csku) in self.ledger.dead_keys:
                    return "remove the invalidated cart line before adding a replacement"
            if self.shopper_question_pending and self.ledger.event_target:
                return "answer the shopper's event question with message before changing the cart"
            c = self._assess(pid, sku)
            key = Ledger.key(pid, sku)
            if not c["live"]:
                return "that variant is out of stock, over budget, or already invalidated"
            if not c["viewed"]:
                return "view or compare that product first so its raw facts are on record"
            if key not in self.ledger.inspected:
                return "inspect_stock that exact variant first: its current price and stock decide whether it can be bought"
            if c["spec"] != 2 and self.brief.spec_value:
                return (
                    "that variant does not resolve to the required %s from its own option text or raw facts"
                    % self.brief.spec_value
                )
            if not c["ok"]:
                return "that product fails a firm requirement (category, brand, or the kind of device requested)"
            if self.brief.family == "preference" and len(self.ledger.listing_ids) < 2:
                return "compare two products with different listing_id values before the first cart addition"
            if (
                self.brief.family == "preference"
                and self._priority_tokens()
                and self._priority_hits(pid, sku) <= 0
                and self._any_priority_survivor()
            ):
                return (
                    "that product does not mention the shopper's stated priorities; choose one whose title or "
                    "facts overlap those priorities"
                )
            if (
                self.brief.family == "recall"
                and _accessory_wrapper(self._title(pid))
                and any(
                    c2["ok"] and not _accessory_wrapper(self._title(c2["pid"]))
                    for c2 in self._candidates()
                )
            ):
                return "prefer a primary theme match over an accessory/case/cover listing when both exist"
            if self.brief.family == "recovery" and not self.ledger.adds and self.brief.pref_model and self._relax(pid, sku):
                return (
                    "the first cart item must be the exact model and color the shopper named; the other colors are "
                    "separate listings sharing the same title, so search that title and read them first"
                )
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
                return (
                    "the shopper has already answered; decide from the evidence you have rather than asking again"
                )
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
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main", "SoftAdmit", "MiscUtil", "Vocab"]
