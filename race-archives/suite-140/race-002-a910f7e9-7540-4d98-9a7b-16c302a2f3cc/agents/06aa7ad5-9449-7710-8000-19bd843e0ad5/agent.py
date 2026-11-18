from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_GOLD_QUICK = "deepseek/deepseek-v3.2"
_GOLD_DENSE = "qwen/qwen3.5-397b-a17b"
_GOLD_RELIEF = (
    "google/gemma-4-31b-it",
    "openai/gpt-oss-120b",
    "deepseek/deepseek-chat-v3.1",
)


TF1 = "intent_decomposition"
TF2 = "retrieval_recall"
TF3 = "constraint_satisfaction"
TF4 = "preference_reasoning"
TF5 = "ranking"
TF6 = "recovery"
TF7 = "justification"

_FAMILIES = (TF1, TF2, TF3, TF4, TF5, TF6, TF7)

_GOLD_DEEP = frozenset({TF1, TF3, TF4, TF5, TF7})

_FAMILY_WEIGHT = {TF7: 7, TF6: 6, TF5: 5, TF3: 4, TF1: 3, TF4: 2, TF2: 1}


_GOAL_PROBES: tuple[tuple[str, str, int], ...] = (
    (TF1, r"\bideally\b|\bnice[-\s]to[-\s]have\b|\bwould\s+be\s+nice\b", 4),
    (TF1, r"\bnon[-\s]negotiable\b|\bhard\s+requirement\b|\bmust\s+(?:be|have)\b", 3),
    (TF1, r"\bbut\s+i\s+(?:also|really|still)\b|\bif\s+possible\b", 3),
    (TF1, r"\bfirm\b|\babsolutely\b", 2),
    (TF2, r"\blisted\s+as\b|\bshows?\s+up\s+as\b|\bmatching\s+the\s+phrase\b", 4),
    (TF2, r"\bexact\s+phrase\b|\bword(?:ed|ing)\b|\bcalled\b", 3),
    (TF2, r"\bany(?:thing|)\s+that\s+matches\b|\bsomewhere\s+in\s+the\s+catalog\b", 3),
    (TF2, r"\bfind\s+me\s+(?:all|every)\b", 2),
    (TF3, r"\b(?:if|when|once|until|unless)\s+you\s+ask\b|\bask(?:ed|ing)?\s+me\b", 5),
    (TF3, r"\bi(?:'| a)?ll\s+(?:tell|share|give)\b[^.]{0,30}\bask\b|\bon\s+request\b", 5),
    (TF3, r"\bwon'?t\s+(?:say|tell|share)\b|\bhold(?:ing)?\s+(?:it\s+)?back\b", 4),
    (TF3, r"\bone\s+more\s+(?:thing|condition|requirement|detail)\b", 4),
    (TF3, r"\bexactly\b|\bprecise(?:ly)?\b|\bstrict(?:ly)?\b", 2),
    (TF4, r"\bi'?d\s+rather\b|\brather\s+(?:have|pay|get)\b|\btorn\s+between\b", 4),
    (TF4, r"\bmatters?\s+more\b|\bweigh(?:s|ing)?\s+more\b|\bcare\s+more\s+about\b", 4),
    (TF4, r"\bmore\s+important\s+than\b|\bcounts?\s+for\s+more\b", 4),
    (TF4, r"\bhappy\s+to\s+pay\s+more\b|\bdon'?t\s+mind\s+(?:paying|spending)\b", 3),
    (TF4, r"\btrade[-\s]?off\b|\bleaning\s+toward\b|\bon\s+the\s+fence\b", 3),
    (TF4, r"\bcheapest\s+(?:one|option|of)\b|\bbest\s+value\b", 2),
    (TF5, r"\brank\b|\border\s+them\b|\bsort\s+them\b|\bwhich\s+comes\s+first\b", 4),
    (TF5, r"\btop\s+(?:pick|choice)\b|\bfirst\s+choice\b|\brunner[-\s]up\b", 3),
    (TF5, r"\bbreak\s+the\s+tie\b|\btie[-\s]?break", 3),
    (TF5, r"\bthese\s+(?:two|three|four|candidates|options)\b|\bshort\s?list\b", 2),
    (TF6, r"\bno\s+longer\s+(?:available|there|listed)\b|\bwent\s+away\b", 5),
    (TF6, r"\bsold\s+out\b|\bout\s+of\s+stock\b|\bdisappear(?:ed|s)?\b", 4),
    (TF6, r"\bis\s+gone\b|\bcan'?t\s+find\s+it\b|\bnot\s+available\s+any\s?more\b", 4),
    (TF6, r"\bpulled\b|\bwithdrawn\b|\bvanish(?:ed|es)?\b", 3),
    (TF6, r"\bclosest\s+(?:thing|match)\b|\bnext\s+best\b|\bfall\s?back\b", 3),
    (TF6, r"\banything\s+similar\b|\bsomething\s+else\s+instead\b|\bget\s+instead\b", 3),
    (TF7, r"\bwhy\s+(?:did|do|does|this|that)\b|\bwhy\?\b", 4),
    (TF7, r"\bdefend\b|\bjustif\w*\b|\bmake\s+the\s+case\b", 4),
    (TF7, r"\bwalk\s+me\s+through\b|\bexplain\s+(?:your|the)\b", 3),
    (TF7, r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:evidence|receipts|proof)\b", 3),
)

_RULE_PROBES: tuple[tuple[str, str, int], ...] = (
    (TF1, r"\bseparate\b[^.]{0,40}\b(?:requirement|preference)\b|\btwo\s+tiers?\b", 4),
    (TF1, r"\bbinding\b[^.]{0,30}\boptional\b|\bwish\b", 3),
    (TF2, r"\breformulate\b|\bbroaden\b|\bre[-\s]?quer(?:y|ies)\b", 4),
    (TF2, r"\brecall\b|\bcoverage\b|\bcandidate\s+set\b", 3),
    (TF3, r"\bwill\s+not\s+volunteer\b|\bmust\s+be\s+asked\b|\bdisclos", 5),
    (TF3, r"\ball\s+constraints?\b[^.]{0,30}\bsatisf", 3),
    (TF4, r"\bsoft\s+(?:preference|axis|criteri)\b|\bweigh\b", 4),
    (TF4, r"\bamong\s+(?:the\s+)?valid\b|\btie\s+goes\s+to\b", 3),
    (TF5, r"\bordering\b|\branked\b|\bpriority\s+order\b", 4),
    (TF5, r"\bfixed\s+set\b|\bonly\s+the\s+listed\b", 3),
    (TF6, r"\brelax\b|\bconcede\b|\bdegrade\s+gracefully\b", 4),
    (TF6, r"\bafter\s+the\s+event\b|\bre[-\s]?inspect\b|\bstock\s+event\b", 3),
    (TF7, r"\bstructured\s+claim\b|\bclaims?\s+array\b|\bjustification\b", 5),
    (TF7, r"\bgrounded\b|\bverbatim\b|\battest", 3),
)

_GOAL_PROBES_RX = tuple((fam, re.compile(rx, re.I), w) for fam, rx, w in _GOAL_PROBES)
_RULE_PROBES_RX = tuple((fam, re.compile(rx, re.I), w) for fam, rx, w in _RULE_PROBES)


_SKUREF_RX = re.compile(r"\b(\d{5,})::([\w.-]+)")
_CAP_RX = re.compile(
    r"(?:at\s+or\s+(?:under|below)|under|below|beneath|up\s+to|no\s+more\s+than|"
    r"within|max(?:imum)?(?:\s+of)?|not\s+(?:over|above))\s*"
    r"[^\d]{0,4}([\d][\d,]*(?:\.\d+)?)",
    re.I,
)
_QUOTED_RX = re.compile(r"[\"\u201c\u2018']([^\"\u201d\u2019']{3,64})[\"\u201d\u2019']")
_DEFEND_RX = re.compile(r"\bwhy\b|\bjustif|\bdefend\b|\breasons?\b|\bexplain\b|\bwalk me through\b|\bgive me a reason\b|\bback it up\b", re.I)
_GATED_RX = re.compile(
    r"\b(?:if|when|once|until|unless)\s+you\s+ask\b|\bask(?:ed|ing)?\s+me\b|"
    r"\b(?:tell|share|reveal|give)\b[^.]{0,40}\bif\b[^.]{0,24}\bask\b|"
    r"\bwon'?t\s+(?:say|tell|volunteer)\b|\bhold(?:ing)?\s+(?:it\s+)?back\b|"
    r"\bon\s+request\b|\bone\s+more\s+(?:thing|condition|requirement|detail)\b",
    re.I,
)
_SOFT_AXIS_RX = re.compile(
    r"\b(?:rather|prefer(?:ably)?|matters?\s+more|more\s+important\s+than|"
    r"care\s+more\s+about|torn\s+between|leaning|trade[-\s]?off|"
    r"happy\s+to\s+pay\s+more|cheaper|cheapest|lighter|bigger|roomier|"
    r"sturdier|smaller|faster|quieter)\b",
    re.I,
)
_FIELDNAME_RX = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_UNITED_NUM_RX = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s?(?:inch|in|hz|fps|p|k|gb|tb|mb|mm|cm|mp|w|wh|mah|"
    r"oz|ml|l|kg|lb|ft|yd|\"|'))?\b",
    re.I,
)

_CORE_FIELDS = frozenset({"price", "in_stock"})

_DENY_FIELDS = frozenset(
    {
        "category",
        "category_path",
        "categories",
        "title",
        "name",
        "product_name",
        "summary",
        "evidence",
        "reason",
        "rationale",
        "note",
        "notes",
        "description",
        "display",
        "facts",
        "options",
        "os",
        "operating system",
        "operating_system",
        "device_operating_system",
        "device os",
        "specification",
        "spec",
        "specs",
        "hdr",
        "feature",
        "features",
        "attribute",
        "attributes",
        "requirement",
        "requirements",
        "why",
        "justification",
        "claim",
        "claims",
        "proof",
        "support",
        "meta",
        "property",
        "properties",
        "phone_type",
        "device_type",
        "product_type",
        "sku",
        "product_id",
        "item_type",
        "model_number",
    }
)

_SHOPPER_ATTRS = frozenset(
    {
        "size",
        "material",
        "color",
        "colour",
        "width",
        "capacity",
        "dimensions",
        "finish",
        "weight",
    }
)

_DRAWER_WORDS = ("display", "facts", "options", "os", "size", "title", "spec", "specs")

_SWATCH = frozenset(
    {
        "xs", "s", "m", "l", "xl", "xxl", "petite", "tall",
        "small", "medium", "large", "mini", "compact", "oversized",
        "cotton", "polyester", "wool", "cashmere", "silk", "linen", "rayon",
        "leather", "suede", "denim", "canvas", "nylon", "mesh", "rubber",
        "plastic", "metal", "aluminum", "steel", "wood", "bamboo", "glass",
        "ceramic", "marble", "oak", "walnut", "maple", "pine",
        "black", "white", "ivory", "cream", "navy", "blue", "teal", "green",
        "olive", "red", "burgundy", "pink", "purple", "yellow", "orange",
        "gray", "grey", "silver", "gold", "brown", "tan", "beige", "khaki",
        "charcoal", "bronze", "copper", "chrome", "brushed",
        "narrow", "wide", "regular", "slim", "relaxed", "straight",
        "matte", "glossy", "gloss", "satin", "textured", "smooth",
        "hd", "fhd", "qhd", "uhd", "oled", "lcd", "led", "ips", "retina",
        "wireless", "wired", "waterproof", "insulated", "foldable",
    }
)

_ASK_NOISE = frozenset(
    {
        "a", "an", "the", "and", "or", "for", "with", "from", "that", "this",
        "these", "those", "find", "buy", "get", "need", "want", "looking",
        "please", "under", "below", "above", "over", "price", "budget",
        "cheap", "peso", "pesos", "php", "usd", "dollar", "dollars", "eur",
        "stock", "available", "availability", "why", "reason", "justify",
        "explain", "product", "products", "item", "items", "thing", "shopper",
        "order", "cart", "exactly", "must", "should", "also", "only", "than",
        "then", "when", "which", "what", "have", "has", "had", "are", "is",
        "be", "was", "to", "of", "on", "at", "by", "as", "if", "not", "no",
        "yes", "true", "false", "brand", "size", "color", "colour", "material",
        "width", "capacity", "dimensions", "finish", "resolution", "screen",
        "screen_size", "weight", "type", "style", "ideally", "rather",
        "prefer", "preferably", "firm", "hard", "soft",
    }
)

_MONEY_WORDS = frozenset(
    {
        "price", "priced", "cost", "costs", "cheap", "cheaper", "cheapest",
        "budget", "spend", "afford", "affordable", "discount", "sale", "deal",
        "shipping", "delivery", "ship", "warranty", "return", "returns",
        "refund", "free", "php", "usd", "peso", "pesos", "dollar", "dollars",
        "under", "below", "above", "max", "maximum", "minimum", "total",
        "stock", "available", "availability", "instock",
    }
)

_FILLER_ADJ = frozenset(
    {
        "nice", "good", "great", "best", "better", "decent", "solid", "lovely",
        "cool", "comfy", "comfortable", "stylish", "trendy", "modern", "classic",
        "premium", "fancy", "simple", "basic", "perfect", "ideal", "reliable",
        "amazing", "awesome", "pretty", "beautiful", "quality", "proper",
        "suitable", "appropriate", "useful", "handy", "neat", "clean",
    }
)


def _slug(raw: Any) -> str:
    low = str(raw or "").strip().lower()
    low = low.replace("-", "_").replace(" ", "_").replace("/", "_")
    low = re.sub(r"[^a-z0-9_]+", "", low)
    return re.sub(r"_+", "_", low).strip("_")


def _tight(raw: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(raw or "").lower())


def _verbatim(value: Any, blob: Any) -> bool:
    needle = _tight(value)
    hay = _tight(blob)
    return bool(needle) and needle in hay


def _bounded(token: Any, text: Any) -> bool:
    tok = str(token or "").strip().lower()
    blob = str(text or "").lower()
    if not tok or not blob:
        return False
    if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", blob):
        return True
    compact = _tight(tok)
    if not compact:
        return False
    hay = _tight(blob)
    if len(compact) <= 3:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(compact)}(?![a-z0-9])", hay))
    return compact in hay


def _literal_nouns(goal: str) -> list[str]:
    low = re.sub(r"\$?\d[\d,]*(?:\.\d+)?", " ", (goal or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[a-z0-9][a-z0-9./\"-]{1,}", low):
        tok = raw.strip("-./\"")
        if len(tok) < 2:
            continue
        if tok in _ASK_NOISE or tok in _FILLER_ADJ or tok in _MONEY_WORDS:
            continue
        if tok.isdigit() and len(tok) > 4:
            continue
        key = _tight(tok)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(tok)
    return out


def _near_miss_query(goal: str, phrase: str, prior: list[str]) -> str:
    tokens = _literal_nouns(goal)
    if phrase:
        bits = [
            t
            for t in _literal_nouns(phrase)
            if t not in _FILLER_ADJ and t not in _MONEY_WORDS
        ]
        tokens = _unique(bits + tokens)
    if not tokens:
        return ""
    last = prior[-1].lower() if prior else ""
    unspent = [t for t in tokens if t not in last and _tight(t) not in _tight(last)]
    pool = unspent or tokens
    tried = {_tight(q) for q in prior}
    for width in (3, 4, 2, 5):
        candidate = " ".join(pool[:width])
        if candidate and _tight(candidate) not in tried:
            return candidate
    rotated = " ".join(pool[1:4]) or " ".join(pool[:2])
    return rotated if rotated and _tight(rotated) not in tried else ""


def _gauge_tokens(ask: str) -> list[str]:
    text = (ask or "").lower()
    out: list[str] = []
    seen: set[str] = set()
    for hit in _UNITED_NUM_RX.finditer(text):
        tok = hit.group(0).strip()
        key = _tight(tok)
        if key and key not in seen:
            seen.add(key)
            out.append(tok)
    for word in re.findall(r"[a-z][a-z0-9-]{1,}", text):
        if word in _ASK_NOISE or word not in _SWATCH:
            continue
        key = _tight(word)
        if key and key not in seen:
            seen.add(key)
            out.append(word)
    return out


def _fused_field(field: str) -> bool:
    name = _slug(field)
    if not name:
        return False
    for drawer in _DRAWER_WORDS:
        if name.startswith(drawer + "_") or name.endswith("_" + drawer):
            return True
    return False


def _cleave_brief(brief: str) -> tuple[str, str]:
    head, sep, tail = (brief or "").partition("\n\n")
    if not sep:
        return (brief or "").strip(), ""
    body = tail.strip()
    first, _, rest = body.partition("\n")
    if first.rstrip().endswith(":") and len(first) < 42:
        body = rest.strip()
    return head.strip(), body


def _unique(items: Any) -> list[str]:
    return list(dict.fromkeys(str(i) for i in items))


def _money(raw: Any) -> float | None:
    text = re.sub(r"[^\d.]", "", str(raw or ""))
    if not text or text.count(".") > 1:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _loads(text: str) -> dict[str, Any]:
    body = (text or "").strip()
    if not body:
        return {}
    try:
        obj = json.loads(body)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    left, right = body.find("{"), body.rfind("}")
    if left >= 0 and right > left:
        try:
            obj = json.loads(body[left : right + 1])
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _schema_names(tools: list[Any]) -> list[str]:
    out: list[str] = []
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            out.append(str(fn["name"]))
        elif entry.get("name"):
            out.append(str(entry["name"]))
    return out


def _order_tool_block(tools: list[Any]) -> dict[str, Any] | None:
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        if not isinstance(fn, dict) or fn.get("name") != "place_test_order":
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        just = props.get("justification") if isinstance(props, dict) else None
        claims = (just.get("properties") or {}).get("claims") if isinstance(just, dict) else None
        items = claims.get("items") if isinstance(claims, dict) else None
        field = (items.get("properties") or {}).get("field") if isinstance(items, dict) else None
        if isinstance(field, dict):
            return field
    return None


def _live_enum(tools: list[Any]) -> list[str]:
    field = _order_tool_block(tools)
    values = field.get("enum") if isinstance(field, dict) else None
    if not isinstance(values, list):
        return []
    out, seen = [], set()
    for raw in values:
        name = _slug(raw)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _live_rule(tools: list[Any]) -> str:
    field = _order_tool_block(tools)
    if isinstance(field, dict):
        return str(field.get("description") or "")[:600]
    return ""


class GoldBrief:

    __slots__ = (
        "raw",
        "ask",
        "rules",
        "family",
        "scores",
        "cap",
        "skurefs",
        "quoted",
        "wants_defence",
        "gated",
        "soft_axes",
        "brand_hints",
        "claim_enum",
        "claim_rule",
    )

    def __init__(self, raw: str, tools: list[Any] | None = None) -> None:
        self.raw = raw or ""
        self.ask, self.rules = _cleave_brief(self.raw)
        self.skurefs = _unique(f"{pid}::{sku}" for pid, sku in _SKUREF_RX.findall(self.raw))
        cap = _CAP_RX.search(self.ask) or _CAP_RX.search(self.raw)
        self.cap = cap.group(1) if cap else ""
        hit = _QUOTED_RX.search(self.ask) or _QUOTED_RX.search(self.raw)
        self.quoted = hit.group(1).strip() if hit else ""
        self.wants_defence = bool(_DEFEND_RX.search(self.ask))
        self.gated = bool(_GATED_RX.search(self.raw))
        self.soft_axes = _unique(
            m.group(0).lower() for m in _SOFT_AXIS_RX.finditer(self.ask)
        )
        self.brand_hints = self._brand_hints()
        self.claim_enum = _live_enum(tools or [])
        self.claim_rule = _live_rule(tools or [])
        self.scores, self.family = self._fingerprint()


    def _fingerprint(self) -> tuple[dict[str, int], str]:
        tally = dict.fromkeys(_FAMILIES, 0)
        if len(self.skurefs) >= 2:
            tally[TF5] = 99
            return tally, TF5
        if self.wants_defence:
            tally[TF7] = 99
            return tally, TF7
        ask, rules = self.ask, self.rules
        for fam, rx, weight in _GOAL_PROBES_RX:
            if rx.search(ask):
                tally[fam] += weight
        for fam, rx, weight in _RULE_PROBES_RX:
            if fam == TF7:
                continue
            if rx.search(rules):
                tally[fam] += weight
        if len(self.skurefs) == 1:
            tally[TF5] += 2
        if self.gated:
            tally[TF3] += 7
        if self.wants_defence:
            tally[TF7] += 6
        if self.quoted:
            tally[TF2] += 5
        if len(self.soft_axes) >= 2:
            tally[TF4] += 4
        if re.search(r"\bideally\b|\bnice\b", ask, re.I) and re.search(
            r"\bmust\b|\bfirm\b|\brequire", ask, re.I
        ):
            tally[TF1] += 5
        if re.search(r"\bstill\s+want\b|\banything\s+else\b", ask, re.I):
            tally[TF6] += 2
        top = max(tally.values())
        if top <= 0:
            return tally, TF2
        winners = [fam for fam, score in tally.items() if score == top]
        return tally, max(winners, key=lambda fam: _FAMILY_WEIGHT[fam])

    def _brand_hints(self) -> list[str]:
        out: list[str] = []
        for word in re.findall(r"\b[A-Z][A-Za-z0-9&.-]{2,}\b", self.ask):
            low = word.lower()
            if low in _ASK_NOISE or low in _SWATCH:
                continue
            out.append(low)
        return _unique(out)


    def needs_claims(self) -> bool:
        return self.family == TF7

    def cap_value(self) -> float | None:
        return _money(self.cap)

    def notebook(self) -> str:
        rows: list[str] = []
        if self.cap:
            rows.append(f"- spend wall: {self.cap}")
        if self.quoted:
            rows.append(f"- quoted listing phrase: {self.quoted}")
        if self.skurefs:
            rows.append(f"- named sku set: {', '.join(self.skurefs)}")
        if self.gated:
            rows.append("- a figure is withheld until you ask for it")
        if self.soft_axes:
            rows.append(f"- soft axis words: {', '.join(self.soft_axes[:6])}")
        if self.brand_hints:
            rows.append(f"- possible maker words in the ask: {', '.join(self.brand_hints[:6])}")
        if self.claim_enum:
            rows.append(f"- live schema claim fields: {', '.join(self.claim_enum)}")
        if self.claim_rule:
            rows.append(f"- live schema claim rule: {self.claim_rule[:280]}")
        return ("GOLD NOTEBOOK\n" + "\n".join(rows) + "\n") if rows else ""


class DualLaneBook:

    __slots__ = (
        "facts",
        "listing",
        "titles",
        "order",
        "viewed",
        "brand_lane",
        "firm_lane",
        "focus_pid",
        "last_pid",
        "last_price",
        "last_stock",
        "sku_seen",
        "last_batch",
        "thin_batches",
        "queries",
    )

    def __init__(self) -> None:
        self.facts: dict[str, dict[str, str]] = {}
        self.listing: dict[str, str] = {}
        self.titles: dict[str, str] = {}
        self.order: list[str] = []
        self.viewed: set[str] = set()
        self.brand_lane: set[str] = set()
        self.firm_lane: set[str] = set()
        self.focus_pid = ""
        self.last_pid = ""
        self.last_price = ""
        self.last_stock = ""
        self.sku_seen: dict[str, set[str]] = {}
        self.last_batch = 0
        self.thin_batches = 0
        self.queries: list[str] = []


    def remember_query(self, text: Any) -> None:
        query = str(text or "").strip()
        if query and query.lower() not in {q.lower() for q in self.queries}:
            self.queries.append(query[:160])

    def thin(self) -> bool:
        return len(self.order) < 2 or self.last_batch <= 1

    def blurb(self) -> str:
        return (
            f"identities={len(self.order)} viewed={len(self.viewed)} "
            f"last_batch={self.last_batch}"
        )

    def stow(self, pid: str, key: str, value: Any) -> None:
        name = _slug(key)
        if not pid or not name or value is None or isinstance(value, (dict, list)):
            return
        text = str(value).strip()
        if not text:
            return
        if name == "in_stock":
            low = text.lower()
            if low in {"true", "1", "yes", "in_stock", "available"}:
                text = "true"
            elif low in {"false", "0", "no", "sold_out", "unavailable"}:
                text = "false"
            self.last_stock = text
        if name == "price":
            self.last_price = text
        self.facts.setdefault(pid, {})[name] = text
        self.last_pid = pid
        if pid not in self.order:
            self.order.append(pid)

    def absorb(self, obs: Any, deep: bool = False) -> None:
        if not isinstance(obs, dict):
            return
        found: list[str] = []

        def walk(node: Any, pid: str = "") -> None:
            if not isinstance(node, dict):
                return
            here = str(node.get("product_id") or "") or pid
            if here:
                found.append(here)
            if here and here not in self.order:
                self.order.append(here)
            title = str(node.get("title") or "")
            if title and here:
                self.titles[here] = title
            bits: list[str] = []
            if title:
                bits.append(title)
            opts = node.get("options")
            if opts:
                bits.append(str(opts))
                if here:
                    for token in re.findall(r"[\w.-]{2,}", str(opts)):
                        if any(ch.isdigit() for ch in token) or "-" in token:
                            self.sku_seen.setdefault(here, set()).add(token)
            if node.get("sku") and here:
                self.sku_seen.setdefault(here, set()).add(str(node["sku"]))
            for key in ("price", "in_stock"):
                if key in node:
                    self.stow(here, key, node.get(key))
            if node.get("brand"):
                self.stow(here, "brand", node.get("brand"))
            drawer = node.get("facts")
            if isinstance(drawer, dict):
                _spill(drawer, here, self, bits)
            if here and bits:
                prior = self.listing.get(here, "")
                chunk = " | ".join(bits)
                if chunk not in prior:
                    self.listing[here] = (prior + " | " + chunk if prior else chunk)[:2400]
            if here and deep and (node.get("facts") or node.get("price") is not None):
                self.viewed.add(here)
            for key in (
                "observation",
                "compared",
                "product",
                "products",
                "variants",
                "results",
                "items",
                "matches",
                "alternatives",
            ):
                child = node.get(key)
                if isinstance(child, dict):
                    walk(child, here)
                elif isinstance(child, list):
                    for item in child[:12]:
                        walk(item, here)

        walk(obs)
        fresh = list(dict.fromkeys(found))
        self.last_batch = len(fresh)
        if len(fresh) <= 1:
            self.thin_batches += 1


    def resort(self, brief: GoldBrief) -> None:
        cap = brief.cap_value()
        wanted = brief.brand_hints
        self.brand_lane.clear()
        self.firm_lane.clear()
        for pid in self.order:
            rows = self.facts.get(pid, {})
            brand = rows.get("brand", "")
            if brand and wanted and any(_bounded(word, brand) for word in wanted):
                self.brand_lane.add(pid)
            elif brand and not wanted and _verbatim(brand, brief.ask):
                self.brand_lane.add(pid)
            if self.wall_ok(pid, cap):
                self.firm_lane.add(pid)

    def wall_ok(self, pid: str, cap: float | None) -> bool:
        rows = self.facts.get(pid, {})
        if rows.get("in_stock") == "false":
            return False
        if cap is not None:
            price = _money(rows.get("price"))
            if price is not None and price > cap:
                return False
        return bool(rows)


    def seen_product_ids(self) -> list[str]:
        return list(self.order)

    def brand_viewed(self) -> set[str]:
        return self.brand_lane & self.viewed

    def firm_viewed(self) -> set[str]:
        return self.firm_lane & self.viewed

    def lane_of(self, pid: str) -> str:
        if pid in self.brand_lane:
            return "brand_lane"
        if pid in self.firm_lane:
            return "firm_lane"
        return "unsorted"

    def fact_rows(self, pid: str) -> list[dict[str, str]]:
        return [{"key": k, "value": v} for k, v in self.facts.get(pid, {}).items()]

    def attest_blob(self, pid: str) -> str:
        rows = self.facts.get(pid, {})
        return self.listing.get(pid, "") + " | " + " | ".join(
            f"{k}={v}" for k, v in rows.items()
        )

    def title_decoy(self, pid: str, ask: str) -> bool:
        title = self.titles.get(pid, "").lower()
        if not title or not ask:
            return False
        rows = " ".join(f"{k} {v}" for k, v in self.facts.get(pid, {}).items()).lower()
        for tok in _gauge_tokens(ask):
            if _bounded(tok, title) and not _bounded(tok, rows):
                return True
        return False

    def fact_backed(self, pid: str, ask: str) -> bool:
        if not pid or pid not in self.facts:
            return False
        rows = " ".join(f"{k} {v}" for k, v in self.facts.get(pid, {}).items()).lower()
        tokens = _gauge_tokens(ask)
        if not tokens:
            return any(k not in _CORE_FIELDS for k in self.facts.get(pid, {}))
        return any(_bounded(tok, rows) for tok in tokens)

    def phrase_in_facts(self, pid: str, phrase: str) -> bool:
        if not phrase:
            return True
        rows = " ".join(f"{k} {v}" for k, v in self.facts.get(pid, {}).items())
        return _verbatim(phrase, rows)

    def phrase_in_title(self, pid: str, phrase: str) -> bool:
        return bool(phrase) and _verbatim(phrase, self.titles.get(pid, ""))

    def known_sku(self, pid: str, sku: str) -> bool:
        if not sku:
            return False
        if sku in self.sku_seen.get(pid, set()):
            return True
        return _verbatim(sku, self.attest_blob(pid))


def _spill(
    block: dict[str, Any],
    pid: str,
    book: DualLaneBook,
    bits: list[str],
    prefix: str = "",
) -> None:
    for key, val in block.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(val, dict):
            bits.append(f"{path}={json.dumps(val, ensure_ascii=False)[:160]}")
            _spill(val, pid, book, bits, path)
            book.stow(pid, str(key), json.dumps(val, ensure_ascii=False)[:160])
        elif isinstance(val, list):
            text = ", ".join(str(x) for x in val[:12])
            bits.append(f"{path}={text}")
            book.stow(pid, str(key), text)
        else:
            text = str(val).strip()
            if not text:
                continue
            bits.append(f"{path}={text}")
            book.stow(pid, str(key), text)
            if prefix:
                book.stow(pid, path.rsplit(".", 1)[-1], text)


_TF7_FILL = """
You are DualLedger's claim-fill clerk. Post ONE missing MANDATORY-PROPERTY gate
voucher for the exact sku under review. This is a double-entry journal act:
debit the hard wall the shopper named; credit only a printed fact-row token.
Answer with JSON only: {"claims":[{"field":"...","value":"..."}, ...]}

INPUT LEDGER (JSON keys)
  goal            shopper ask — find the disqualifier / hard wall inside it
  allowed_fields  live schema enum (may be empty)
  schema_rule     the tool's own sentence about legal field names — outranks guesswork
  have            fields already on the books (often only price / in_stock / brand)
  listing         observed text for this sku
  fact_rows       key/value pairs; drawer keys are VALUE evidence only (drawer
                  naming differs by category — never treat a drawer key as the gate)

MANDATORY-PROPERTY GATE (the posting that finishes claim_truth)
1. Identify which PRODUCT PROPERTY in the goal is a DISQUALIFIER — the hard wall.
   That property is the gate you still owe. price / in_stock / brand already in
   `have` do NOT settle that debt.
2. The claim FIELD must NAME THAT mandatory property, using conventional
   snake_case / the schema enum / allowed_fields when present. Naming a different
   attested property while the mandatory one stays unclaimed is a FAILED POSTING
   (secondary-attribute substitution). Brand alone never finishes a hard-wall ask.
3. VALUE = the printed fact-row / inspect token for that property, character for
   character. Titles are marketing copy — not proof. If the printed evidence does
   not prove the stated requirement, answer {"claims":[]}.

SCHEMA FIRST
- Follow schema_rule when present; it outranks inventing a field name.
- If allowed_fields is non-empty, FIELD MUST be one of them (still choose the
  member that matches the mandatory property — not a convenient secondary).
- If allowed_fields is empty, use the conventional shopping attribute a shopper
  would name for that wall — never a catalog drawer key, never a meta label.

SECONDARY-ATTR SUBSTITUTION IS FORBIDDEN
- Do not post a true-but-irrelevant attribute (colour when size is the wall,
  brand when the wall is a hard product property, a soft preference when a
  mandatory gate is open). An attested secondary while the mandatory field is
  missing leaves claim_truth unpaid.
- Do not invent a "close enough" synonym as FIELD or VALUE.
- Do not repeat price or in_stock; those lines are already booked.

FORBIDDEN FIELD NAMES (meta / identity — not walls; they block the real fill)
specification, spec, specs, hdr, feature, features, attribute, requirement, why,
rationale, justification, evidence, reason, summary, description, display, os,
category, title, facts, options, phone_type, device_type, product_type, sku,
product_id, item_type, model_number. Never fuse drawer+attribute names.

POSTING BUDGET
- At most two claims. Prefer exactly one gate claim that names the mandatory
  property. A short brand claim is allowed ONLY when a brand fact row proves it
  AND the mandatory gate is already covered or being posted here.
- Empty claims when unproven: {"claims":[]}.
""".strip()


class ClaimDesk:

    @staticmethod
    def polish(
        claims: Any, brief: GoldBrief, book: DualLaneBook, pid: str
    ) -> list[dict[str, str]]:
        if not isinstance(claims, list):
            claims = []
        allowed = {_slug(x) for x in brief.claim_enum}
        rows = book.facts.get(pid, {})
        attest = book.attest_blob(pid)
        kept: list[dict[str, str]] = []
        used: set[str] = set()

        for entry in claims:
            if not isinstance(entry, dict):
                continue
            field = _slug(entry.get("field"))
            value = "" if entry.get("value") is None else str(entry.get("value")).strip()
            if not field or not value:
                continue
            if field in _DENY_FIELDS or _fused_field(field):
                continue
            if allowed and field not in allowed:
                continue
            if field in used or not _FIELDNAME_RX.match(field):
                continue
            if field == "in_stock":
                if value.lower() in {"true", "1", "yes", "in_stock"}:
                    value = "true"
            if field == "brand":
                value = ClaimDesk._short_brand(value, rows, attest)
                if not value or not rows.get("brand"):
                    continue
            elif field not in _CORE_FIELDS and not _verbatim(value, attest):
                swap = ClaimDesk._attested_swap(value, rows, brief.ask, attest)
                if not swap:
                    continue
                value = swap
            used.add(field)
            kept.append({"field": field, "value": value})

        for need in ("price", "in_stock"):
            if need in used or (allowed and need not in allowed):
                continue
            got = rows.get(need) or (
                book.last_price if need == "price" else book.last_stock
            )
            if got:
                slot = 0 if need == "price" else min(1, len(kept))
                kept.insert(slot, {"field": need, "value": got})
                used.add(need)

        if "brand" not in used and (not allowed or "brand" in allowed):
            brand = ClaimDesk._short_brand(rows.get("brand", ""), rows, attest)
            if brand and rows.get("brand") and _verbatim(brand, brief.ask):
                kept.append({"field": "brand", "value": brand})
                used.add("brand")

        if brief.needs_claims() and not ClaimDesk._has_gate(kept, brief.ask):
            for row in ClaimDesk._fill(brief, book, pid, used):
                field = _slug(row.get("field"))
                if field in used or field in _DENY_FIELDS or _fused_field(field):
                    continue
                if allowed and field not in allowed:
                    continue
                if not _FIELDNAME_RX.match(field):
                    continue
                value = str(row.get("value") or "").strip()
                if not value:
                    continue
                if not _verbatim(value, attest):
                    swap = ClaimDesk._attested_swap(value, rows, brief.ask, attest)
                    if not swap:
                        continue
                    value = swap
                if field == "brand":
                    if not rows.get("brand"):
                        continue
                    value = ClaimDesk._short_brand(value, rows, attest)
                    if not value:
                        continue
                used.add(field)
                kept.append({"field": field, "value": value})
                if ClaimDesk._has_gate(kept, brief.ask):
                    break

        if brief.needs_claims() and not ClaimDesk._has_gate(kept, brief.ask):
            gate = ClaimDesk._gate_from_facts(brief.ask, rows, attest, used, allowed)
            if gate:
                used.add(gate["field"])
                kept.append(gate)

        if (
            brief.needs_claims()
            and "brand" not in used
            and (not allowed or "brand" in allowed)
            and len(kept) < 4
            and rows.get("brand")
        ):
            brand = ClaimDesk._short_brand(rows.get("brand", ""), rows, attest)
            if brand:
                kept.append({"field": "brand", "value": brand})
                used.add("brand")

        return kept[:6]

    @staticmethod
    def _gate_worthy(field: str, value: str, goal: str) -> bool:
        if not field or field in _CORE_FIELDS or field == "brand" or field in _DENY_FIELDS:
            return False
        if field in _SHOPPER_ATTRS:
            return True
        goal_l = (goal or "").lower()
        if field in goal_l or field.replace("_", " ") in goal_l:
            return True
        if value and (_verbatim(value, goal) or value.lower() in goal_l):
            return True
        return False

    @staticmethod
    def _has_gate(claims: list[dict[str, str]], goal: str = "") -> bool:
        for claim in claims:
            field = _slug(claim.get("field"))
            value = "" if claim.get("value") is None else str(claim.get("value")).strip()
            if ClaimDesk._gate_worthy(field, value, goal):
                return True
        return False

    @staticmethod
    def _gate_from_facts(
        goal: str,
        rows: dict[str, str],
        attest: str,
        have: set[str],
        allowed: set[str],
    ) -> dict[str, str] | None:
        goal_l = (goal or "").lower()
        ranked: list[tuple[int, str, str]] = []
        for raw_key, raw_val in rows.items():
            key = _slug(raw_key)
            value = str(raw_val or "").strip()
            if not key or not value or key in have:
                continue
            if key in _CORE_FIELDS or key == "brand" or key in _DENY_FIELDS:
                continue
            field = key
            if field in have or field in _DENY_FIELDS or _fused_field(field):
                continue
            if allowed and field not in allowed:
                continue
            if not _FIELDNAME_RX.match(field):
                continue
            if not _verbatim(value, attest):
                continue
            score = 0
            if field in _SHOPPER_ATTRS:
                score += 4
            if field in goal_l or key in goal_l:
                score += 12
            if _verbatim(value, goal):
                score += 5
            if score <= 0 and not ClaimDesk._gate_worthy(field, value, goal):
                continue
            if score <= 0:
                continue
            ranked.append((score, field, value))
        if not ranked:
            return None
        ranked.sort(key=lambda row: (-row[0], row[1]))
        _, field, value = ranked[0]
        return {"field": field, "value": value}

    @staticmethod
    def _fill(
        brief: GoldBrief, book: DualLaneBook, pid: str, have: set[str]
    ) -> list[dict[str, str]]:
        attest = book.attest_blob(pid)
        rows = book.facts.get(pid, {})
        if not attest.strip(" |") and not rows:
            return []
        enum = [name for name in (brief.claim_enum or []) if _slug(name) not in have]
        payload = {
            "goal": brief.ask[:1500],
            "allowed_fields": enum,
            "schema_rule": brief.claim_rule,
            "have": sorted(have),
            "listing": attest[:2200],
            "fact_rows": book.fact_rows(pid)[:40],
        }
        raw = _wire.json_ask(
            engine=_GOLD_QUICK,
            system=_TF7_FILL,
            user=json.dumps(payload, ensure_ascii=False),
        )
        parsed = _loads(raw)
        claims = parsed.get("claims") if isinstance(parsed, dict) else None
        if not isinstance(claims, list):
            return []
        return [row for row in claims[:2] if isinstance(row, dict)]

    @staticmethod
    def _attested_swap(
        value: str, rows: dict[str, str], ask: str, attest: str
    ) -> str:
        if value and _verbatim(value, attest):
            return value
        ranked: list[tuple[int, str]] = []
        for key, observed in rows.items():
            if not observed or key in _CORE_FIELDS or key == "brand":
                continue
            if not _verbatim(observed, attest):
                continue
            score = 0
            if _verbatim(observed, ask):
                score += 5
            if value and (_verbatim(observed, value) or _verbatim(value, observed)):
                score += 4
            if key in _DENY_FIELDS or key in {"display", "size", "facts", "options"}:
                score += 1
            if score:
                ranked.append((score, observed))
        if ranked:
            ranked.sort(key=lambda pair: (-pair[0], -len(pair[1])))
            return ranked[0][1]
        return ""

    @staticmethod
    def _short_brand(raw: str, rows: dict[str, str], attest: str) -> str:
        structured = str(rows.get("brand") or "").strip()
        if not structured:
            return ""
        if not _verbatim(structured, attest) or len(structured.split()) > 4:
            return ""
        text = str(raw or "").strip()
        if text and len(text.split()) <= 3 and len(text) <= 40 and _verbatim(text, structured):
            return text
        return structured


_ROLE = """
You are GoldenGirl on DualLedgerEngine. Discovery is double-entry bookkeeping:
every candidate posts to a brand/wish lane, a firm/wall lane, or both — never
to a vague “looks fine” pile. Unbalanced books (one lane funded, the other
ignored) and unfinished market-event accounts both pay zero.

SENSES (tools only)
  search · filter · view · compare · inspect_stock · add_to_cart ·
  remove_from_cart · message · place_test_order
The books CLOSE only when place_test_order returns an observation with
done=true. An open ledger with a gold-looking cart and no order is still a
zero. Late-budget law: stop open search, balance any open event, finish the
order pair (inspect_stock → place_test_order). Keep ≥2 steps for that close.

LEDGER HOUSE RULES
- Dual discovery is the default posture: fund brand_lane (maker/wish tokens
  that match brand facts) AND firm_lane (items clearing every wall). Firm over
  wish on every tie. Soft never overrides firm.
- First add_to_cart is a graded posting — no placeholder skus “to hold a slot.”
- Shopper QUESTIONS are open liabilities: message the same/next turn BEFORE
  cart or order. An unanswered question leaves the books open.
- When a figure is withheld until you ask: message-before-search — no lookup
  until disclosure posts to the ledger.
- Non-TF7: omit justification.claims entirely.
- Schema first when claim fields exist: schema_rule and allowed_fields outrank
  guesswork. VALUE = fact-row / inspect token; titles are marketing.

MANDATORY-PROPERTY GATE (when the ask states a hard product wall)
Identify the product property that is a DISQUALIFIER. That property must be
attested on the carted sku’s fact rows before commit. Brand, price, and
in_stock alone never finish a hard-wall ask. Claiming a secondary attested
attribute while the mandatory property is still unclaimed is a failed posting.

EVENT-CLOSE RITUAL (every family — unfinished events pay 0)
Before place_test_order, if ANY market event fired:
  1. inspect_stock the EVENT-TARGET first — that alone marks event_closed.
     A substitute / replacement inspect does NOT close the event. Do this even
     when you will order a different sku afterward.
  2. remove_from_cart if the target no longer qualifies; recover per the
     family account book.
  3. Re-check STOCK and BUDGET after the event (within_budget and stock can
     both fail after a market move). Never order a sku last seen out of stock.
  4. Rebuild any claims from the fresh observation, then place_test_order.
event_closed=false pays 0 even when the cart looks perfect.

FACT PRINTING BEATS MARKETING COPY
Post journal entries from fact rows and inspect_stock, not from titles. Never
invent meta field names (specification, hdr, phone_type, sku, …) — those are
not walls and they block the real gate fill.
""".strip()

_PB_TF1 = """
ACCOUNT BOOK — INTENT (brand/wish lane vs firm/wall lane)

Split the ask into two unequal columns before you post any search:
  FIRM / WALLS   disqualifiers — class, spend ceiling, stated hard property,
                 stock. These are hard walls; miss one and the posting fails.
  WISH / BRAND   non-mandatory leanings — ideally / prefer / would love, often
                 a maker. Never promote a wish into a wall; never demote a wall.

DUAL DISCOVERY (both columns must be funded)
- Query A: firm nouns + wish/maker token → populate brand_lane candidates.
  When BOTH a maker and a hard gate are named, post them together (brand + gate
  tokens in one query) rather than dropping one column.
- Query B: firm nouns alone → populate firm_lane backups.
A wish-only dead end is not an answer; a firm-only pass that never tried the
wish wastes the brand column. DualLedgerEngine tracks brand_lane and firm_lane
explicitly — feed both.

FIRM GATE BEFORE CART
- Walk every firm wall against the candidate’s FACT ROWS (not titles). Do not
  cart until the mandatory hard property appears in a fact row for that sku.
- Wrong firm gate = failed posting: a sku that clears price/stock but misses
  the stated hard property is not firm_lane eligible.

FIRST-CART BRAND RULE (graded posting)
- Post the first cart from brand_lane ONLY when that sku’s FACT ROWS clear EVERY
  firm wall, and a brand fact proves the maker when the goal named one.
- Otherwise post the best firm_lane item and acknowledge the unmet wish with
  message. Soft never overrides firm. Walls win every tie. Ordering a
  non-matching / non-fact-backed sku when a brand was named is a failed posting.

MESSAGE INTERVENTIONS (question gate)
If the shopper asks how you chose, answer with message before further cart or
order postings. Unanswered questions leave the books open and sink the score.
When a figure is withheld until you ask, message first — no search until the
reveal posts.

AFTER A MARKET EVENT (finish the run)
inspect_stock the EVENT-TARGET (close the event) even if ordering a different
sku; remove if invalid; RE-CHECK stock AND budget under the new observation;
then RE-APPLY the brand rule under the new walls: re-fund dual discovery if
needed; brand_lane cart only if still fully firm-compliant; else firm_lane
backup. Do not keep a pre-event brand pick that fails the new walls. Never
order a sku last seen out of stock. within_budget can fail after a price move.

CLOSE
view fact rows → add_to_cart → inspect_stock → place_test_order. Omit
justification. Reserve the closing pair. Late budget: stop open search, close
any open event, finish.
""".strip()

_PB_TF2 = """
ACCOUNT BOOK — RECALL (cover the shelf before you settle)

The shopper is pointing at an existing catalog line, often by distinctive
printed wording. Score tracks whether you actually surfaced it. Thin coverage
plus an unfinished market event zeros even a gold-looking cart. One lucky hit
is an incomplete journal — widen the shelf, then close the books.

MULTI-QUERY COVERAGE (do this before commitment)
- Start with distinctive shopper tokens / quoted phrases VERBATIM. Reuse those
  tokens across passes — do not scrub the distinctive words away.
- When the first shelf is thin, run ANOTHER query shape: keep the rare nouns,
  vary one secondary knob. Multi-query coverage beats a single neighbour.
- Measure progress by MANY DISTINCT product_ids viewed — not duplicate rows of
  one id, not the same three snippets re-opened.
- Open several identities before you settle. First cart is graded — no
  placeholder posting while coverage is still thin.

ONE NEAR-MISS REFINE, THEN VIEW MANY
If the first pass is thin, adjust ONE token (drop adjective, drop maker, plural
flip, strip punctuation). Then stop rewriting and view many distinct
product_ids. Whole-query rewrites every turn burn steps and revisit the same
neighbourhood.

PHRASE ON FACT ROWS, NOT TITLE-ONLY
A marketing title that borrows the phrase is not proof. Open with view; require
the phrase (or its parts) in FACT ROWS before you cart. Title-only matches are
classic false postings in this book. Phrase the match from fact rows, not titles.

EVENT-CLOSE + ORDER (late-finish discipline)
Cart the row-backed match → if any market event fired, inspect_stock the
EVENT-TARGET first (even when ordering another sku; replacement inspect does
not close) → re-check stock and budget → inspect the order sku if needed →
place_test_order. Answer questions with message the turn they arrive. Omit
justification. Late budget: stop refining, close event + order. Thin coverage
left unfinished OR event_closed=false pays 0.
""".strip()

_PB_TF3 = """
ACCOUNT BOOK — CONSTRAINT (literal figure on the exact sku)

Every firm gate must hold on the carted sku — not the product family, not a
sibling colourway. DualLedger’s firm_lane is the only column that matters until
every wall clears on that exact variant. Wrong firm gate (carting a near miss
that lacks the disclosed hard property) is a failed posting.

WITHHELD FIGURE — TURN 1 IS MESSAGE ONLY
When the ask says a detail will be shared if you ask, your first posting is a
plain message requesting it. Do not search or filter first: guesses spend the
budget on the wrong shelf, and the reveal changes the shelf. Ask once, naming
what you need. Message-before-search is law. Workarounds are unbalanced books.

AFTER THE FIGURE POSTS TO THE LEDGER
- Search with the revealed literal exactly. When brand and gate are both named,
  keep them together in the query.
- view/compare until the exact sku is printed in an observation — never invent
  a sku string from a pattern.
- Walk every firm gate against that sku’s FACT ROWS: class, disclosed figure,
  spend, stock. Unpointable conditions are unmet. Titles are not proof.
- Do not cart until the disclosed figure appears in a FACT ROW for that sku.

SELLOUT → NEW PRODUCT_ID (NO SIBLING TRAP)
If the compliant sku sells out, move to a different product_id that still clears
ALL firm gates under the revealed literal. Near variants that miss a gate are
false substitutes. Re-journal each gate on the replacement before carting.
Never order a sku last seen out of stock.

ORDER + EVENT CLOSE
inspect_stock → place_test_order with no justification. On market events:
inspect EVENT-TARGET first (even when ordering a different sku), recover with
another fully compliant product_id, re-check stock AND budget after any price
move, then order. Leave ≥2 steps for the close. Late budget: stop open search,
close the event, finish. event_closed=false pays 0.
""".strip()

_PB_TF4 = """
ACCOUNT BOOK — PREFERENCE (walls first, then stated soft axes)

Multiple items will clear the hard walls; the shopper named how to lean among
them. Honour that lean with printed evidence — cheapest is NOT the default
debit. DualLedger still tracks firm compliance: soft scoring only runs on items
that already posted cleanly to the firm column. Weak soft-axis compare (one
candidate, title adjectives, invented axes) is an unbalanced journal.

POSTING ORDER
1. Clear hard walls; build a set of genuinely valid items (firm_lane discipline).
   A sku missing the mandatory hard property is not soft-eligible.
2. Read ≥2 hard-OK candidates with view/compare on the STATED soft axes using
   printed values only (not title adjectives, not invented axes). One opened
   candidate is not a comparison — the books need at least two journaled reads
   on the axes the shopper actually wrote.
3. Decide only after those reads. Snippet-based preference is an unbalanced guess.
4. When useful, message a one-line rationale naming the axis and the printed
   values that decided it — then cart without further theatre.

ONE OPTIONAL CLARIFY
If two stated axes conflict, one clarifying message is allowed. Do not stall
twice. If evidence already decides, commit without asking. Answering a shopper
question still comes before cart/order when a question is open.

CHEAPEST ≠ WINNER
Price is a tiebreaker when named axes are level (unless the shopper ranked price
first). A costlier hard-OK winner on the stated tilt is the correct posting.
Never invent a “value” or “quality” axis the shopper never wrote.

CLOSE + EVENT RITUAL
Cart the winner → if any market event fired, inspect_stock the EVENT-TARGET
first (replacement inspect does not close; do this even when ordering another
sku) → re-check stock and spend wall → place_test_order. Omit justification.
Late budget: stop comparing extra candidates, close event + order pair.
""".strip()

_PB_TF5 = """
ACCOUNT BOOK — RANKING (closed named set only)

The shopper handed a closed set of product_id::sku lines plus lexicographic
axes. You adjudicate that set; wide search strangers are out of bounds no matter
how attractive a brand_lane stranger looks. This book is a closed trial balance.
Unfinished events and stale pre-event leaders are classic zero postings here.

FULL OPEN BEFORE ANY CART
view or compare ALL named candidates before ranking. An unread line cannot be
placed in the order book. First cart remains graded — do not park a provisional
leader “to hold the slot” before every named line has been opened.

LEX-RANK
First stated axis decides; later axes break ties; price last unless stated.
Record the ordering in your reply so the debit is auditable. Split product_id
and sku as separate arguments (for “12345::abc” → product_id=12345, sku=abc).
Never paste the joined string into product_id.

AFTER EVENT — RE-RANK FROM SCRATCH (strip stale journals)
Do not keep the pre-event #1. Ritual:
  1. inspect_stock the EVENT-TARGET to close the event (even if you will order
     a different named line; replacement inspect does not close).
  2. Refresh surviving named lines from fresh observations; strip stale claims /
     rankings from memory.
  3. Re-check stock AND budgets after any price move; never order a line last
     seen out of stock.
  4. Re-sort from scratch under the same axes. Whoever leads at order time wins.
Pre-event rankings are stale journal entries.

ORDER SHAPE
place_test_order with product_id + sku only — never claims. Strip any
justification object entirely. Cart the leader if needed, inspect_stock, order.
Answer open questions with message first. Reserve ≥2 steps for the closing pair.
Late budget: stop re-opening the set, close the event, finish.
""".strip()

_PB_TF6 = """
ACCOUNT BOOK — RECOVERY (prove unavailable, then concede one rung)

Wanted stock is gone or vanishes mid-run. Walking away empty is a zero. Post
concessions in order — never silently relax class/budget/stock/hard spec.
This book often fails by stalling after a valid substitute OR by leaving a
gold-looking cart with an OPEN market event — finish the run.

PROVE BEFORE SUBSTITUTING
inspect_stock or view the original until an observation says it is unavailable.
Suspicion-based swaps are failed postings even when stock was truly empty. The
ledger needs a proof line before any substitute debit.

CONCESSION LADDER (one rung at a time)
  L0  Hold all constraints; search the original ask.
  L1  Drop cosmetic only (shade/finish/trim); announce with message if talking.
  L2  Drop model/maker next.
  NEVER drop product class, spend wall, hard spec, or stock requirement.
Prove each dead end with observations before climbing to the next rung. Do not
jump from L0 to a random unrelated class. One rung, then re-verify firm walls
on the candidate before carting.

BEFORE FIRST CART
inspect_stock the intended commitment before the first add_to_cart so the run
has a graded posting the environment can score. Never order a sku last seen
out of stock.

AFTER MARKET EVENT — FINISH (do not leave gold cart + open event)
1. inspect_stock EVENT-TARGET immediately (closes event; replacement inspect
   does not — do this even when ordering a different sku).
2. remove_from_cart if invalid.
3. Climb ONE rung to a verified replacement still holding class/budget/
   stock/hard spec. Re-check stock AND budget after any price move.
4. Cart → inspect → place_test_order promptly. Unfinished events pay 0.
Do not burn remaining steps on refine theatre after a valid recovery is carted.
Omit justification. Unplaced recoveries score nothing. event_closed=false pays 0
even when the cart looks gold.
""".strip()

_PB_TF7 = """
ACCOUNT BOOK — JUSTIFICATION (claims are vouchers against printed stock)

place_test_order MUST carry justification.claims as {field, value} pairs graded
against catalog print. Unprinted vouchers are worse than none. DualLedger still
applies: only commit a sku whose firm facts support the vouchers you will attach.

MANDATORY-PROPERTY GATE CLAIM (claim_truth hinge)
- Identify which product property in the goal is a DISQUALIFIER (hard wall).
- The gate claim FIELD must NAME THAT property — conventional snake_case /
  schema enum / allowed_fields. Schema_rule and allowed_fields outrank guesswork.
- price + in_stock + brand alone do NOT finish the books when that mandatory
  property is still unclaimed. Brand alone never finishes a hard-wall ask.
- SECONDARY-ATTR SUBSTITUTION IS A FAILED POSTING: vouchering a true-but-
  irrelevant attested property while the mandatory hard property stays open
  fails claim_truth. Cover the wall that disqualifies, not a convenient side
  attribute.
- VALUE = printed fact-row / inspect token only. Titles are marketing. If
  facts do not prove the stated requirement, leave that sku — do not invent.

VOUCHER SHAPE (≤4; include all facts support)
  price      from newest view / inspect_stock
  in_stock   usually "true" from inspect_stock
  gate       FIELD = the mandatory property named by ask/schema
             VALUE = printed fact-row token (shopper shorthand is a VALUE
             under the right attribute — never a field name)
  soft       one attested reason; short brand only if shopper named a maker AND
             a brand fact row proves it — titles are not brands

SEARCH & COMMITMENT
- When brand and a hard gate are both named, search brand+gate tokens together.
- Do not cart until the mandatory gate figure appears in a FACT ROW for that sku.
  Title-only figures are decoys — reject until a fact row prints them.
- When a brand is named in the goal, only order a fact-backed matching maker
  unless no firm-compliant brand match exists (then firm backup + message).

FORBIDDEN POSTINGS
category, title, summary, evidence, reason, rationale, description, os, display,
specification, spec, hdr, feature(s), attribute, requirement, why,
justification; phone_type, device_type, product_type, sku, product_id,
item_type, model_number (soft junk — never a gate); fused drawer+attribute
names; unit conversions; invented features; prose paragraphs inside values.

COMMITMENT & EVENT-CLOSE RITUAL
- inspect_stock before first add_to_cart (intended commitment).
- After ANY market event: ALWAYS inspect EVENT-TARGET first (even if ordering
  a different sku), drop if invalid, re-check stock AND budget, STRIP stale
  claims, rebuild claims from the fresh observation, then place_test_order.
  A replacement inspect does not close. event_closed=false pays 0 even when
  the cart looks gold — unfinished events zero gold-looking carts.
- Once a fact-backed pick is inspected and the mandatory gate is claimed,
  place_test_order — do not wait for permission or synonym-hunt into a zero.
  Late budget: stop open search, close the event, finish.
""".strip()

_PLAYBOOK: dict[str, str] = {
    TF1: _PB_TF1,
    TF2: _PB_TF2,
    TF3: _PB_TF3,
    TF4: _PB_TF4,
    TF5: _PB_TF5,
    TF6: _PB_TF6,
    TF7: _PB_TF7,
}

_NUDGE = (
    "No tool call posted. Choose a tool and act — DualLedger only senses tools.",
    "Still idle. Balance any open question with message, then search/view → "
    "cart → inspect_stock (event-target first if needed) → place_test_order.",
)

_VOUCH_TIPS = (
    "Ledger tip: MANDATORY-PROPERTY GATE — FIELD must name the shopper’s hard "
    "disqualifier (schema_rule / allowed_fields when present). VALUE = printed "
    "fact-row token. Secondary-attr substitution while the mandatory field is "
    "open fails claim_truth. Never emit phone_type, sku, product_id, "
    "specification, spec, hdr, feature, or why.",
    "Ledger tip: price/in_stock/brand alone do not finish a hard-wall ask — "
    "still voucher the mandatory property. Copy price and in_stock from the "
    "newest view or inspect_stock. Schema first; never fuse drawer+attribute; "
    "never emit os, display, phone_type, sku, specification, or hdr.",
    "Ledger tip: EVENT-CLOSE — if any market event fired, inspect_stock the "
    "EVENT-TARGET first (even when ordering another sku). Replacement inspect "
    "does not close. Re-check stock AND budget, rebuild claims, then "
    "place_test_order. event_closed=false pays 0 on a gold-looking cart.",
    "Ledger tip: at most one soft voucher; short brand only when a brand fact "
    "row attests it. Four claims max. Brand alone never finishes a hard-wall "
    "ask. Banned soft junk: phone_type, device_type, product_type, sku, "
    "product_id, item_type, model_number, category, title, os, display, "
    "specification, hdr, feature, attribute, requirement, why.",
    "Ledger tip: FIELD follows the mandatory property in the ask/schema; "
    "VALUE is a printed fact-row token. Do not post a true-but-irrelevant "
    "secondary attribute while the mandatory gate is unclaimed. Empty claims "
    "if unproven — never invent.",
    "Ledger tip: after events, strip stale claims; never order a sku last "
    "seen out of stock; re-check within_budget under the new price. Title-only "
    "measurements are decoys — keep viewing until a fact row prints the figure.",
    "Ledger tip: search brand+gate tokens together when both are named. "
    "Paste tokens exactly — no unit conversion, no invented features. Past "
    "mid-budget: stop open search, close events (event-target inspect), "
    "inspect, order.",
    "Ledger tip: first cart is a graded posting — no placeholders. Dual "
    "discovery for intent; firm over wish. Non-TF7: omit justification "
    "entirely. Do not cart until the mandatory gate figure is in a FACT ROW.",
)


def _session_block(brief: GoldBrief, policy: dict[str, Any]) -> str:
    labels = _schema_names(policy.get("tools") or [])
    return (
        "SESSION\n"
        f"- family: {brief.family}\n"
        f"- tools: {', '.join(labels) or '(see schemas)'}\n"
        f"- step limit: {policy.get('max_steps')}\n"
        f"- actions per turn: {policy.get('max_calls_per_turn', 1)}"
    )


def _system(brief: GoldBrief, policy: dict[str, Any]) -> str:
    parts = [_ROLE, _PLAYBOOK.get(brief.family, _PB_TF2)]
    if brief.needs_claims() and brief.family != TF7:
        parts.append(_PB_TF7)
    note = brief.notebook()
    if note:
        parts.append(note.strip())
    parts.append(_session_block(brief, policy))
    return "\n\n".join(parts)


def _vouch_tip(turn: int, book: DualLaneBook, brief: GoldBrief) -> str:
    body = _VOUCH_TIPS[(turn - 1) % len(_VOUCH_TIPS)]
    extras: list[str] = []
    if brief.claim_enum:
        extras.append(f"Allowed fields this episode: {', '.join(brief.claim_enum)}.")
    if brief.claim_rule:
        extras.append(f"Schema rule: {brief.claim_rule[:180]}")
    pid = book.focus_pid or book.last_pid
    if pid:
        rows = book.facts.get(pid, {})
        if rows:
            sample = ", ".join(f"{k}={v}" for k, v in list(rows.items())[:8])
            extras.append(f"Newest fact sample for {pid}: {sample}")
        if book.last_price or book.last_stock:
            extras.append(
                f"Last observed price={book.last_price or '?'} "
                f"in_stock={book.last_stock or '?'}."
            )
    return body + (("\n" + "\n".join(extras)) if extras else "")


class GoldLedger:

    __slots__ = (
        "turn",
        "limit",
        "acted",
        "history",
        "closed",
        "carted",
        "cart_pid",
        "cart_sku",
        "owes_reply",
        "market_moved",
        "event_pid",
        "event_sku",
        "asked_gate",
        "coached",
        "rung",
        "rung_proof",
        "compares",
        "refines",
        "lane_pick",
        "bad_sku",
        "blocks",
        "vetoes",
    )

    def __init__(self, limit: int) -> None:
        self.turn = 0
        self.limit = limit
        self.acted: list[str] = []
        self.history: set[str] = set()
        self.closed = False
        self.carted = False
        self.cart_pid = ""
        self.cart_sku = ""
        self.owes_reply = False
        self.market_moved = False
        self.event_pid = ""
        self.event_sku = ""
        self.asked_gate = False
        self.coached: set[str] = set()
        self.rung = 0
        self.rung_proof = False
        self.compares = 0
        self.refines = 0
        self.lane_pick = ""
        self.bad_sku = ""
        self.blocks: set[str] = set()
        self.vetoes = 0

    @property
    def left(self) -> int:
        return max(0, self.limit - self.turn)

    def ever(self, label: str) -> bool:
        return label in self.history

    def searches(self) -> int:
        return sum(1 for label in self.acted if label in {"search", "filter"})

    def note(self, labels: list[str], calls: list[dict[str, Any]]) -> None:
        self.acted = labels
        self.history.update(labels)
        if "compare" in labels:
            self.compares += 1
        if "search" in labels or "filter" in labels:
            self.refines += 1
        if "add_to_cart" in labels:
            self.carted = True
            for call in calls:
                if _call_name(call) == "add_to_cart":
                    args = _call_args(call)
                    self.cart_pid = str(args.get("product_id") or self.cart_pid)
                    self.cart_sku = str(args.get("sku") or self.cart_sku)
        if "place_test_order" in labels:
            self.closed = True
        if "remove_from_cart" in labels:
            self.carted = False
        if "inspect_stock" in labels:
            for call in calls:
                if _call_name(call) != "inspect_stock":
                    continue
                pid = str(_call_args(call).get("product_id") or "")
                if self.event_pid and pid == self.event_pid:
                    self.market_moved = False
                    self.event_pid = ""
                    self.event_sku = ""
                    break
        if "message" in labels:
            self.owes_reply = False
            self.asked_gate = True

    def once(self, key: str) -> bool:
        if key in self.coached:
            return False
        self.coached.add(key)
        return True

    def may_veto(self, tag: str) -> bool:
        if tag in self.blocks:
            return False
        self.blocks.add(tag)
        return True


_SPEND_TOOLS = frozenset({"add_to_cart", "place_test_order"})
_LOOKUP_TOOLS = frozenset({"search", "filter"})
_TALK_TOOLS = frozenset({"message"})
_MAX_VETO_PER_TURN = 2


class GoldBridge:

    def __init__(self, client: ProxyClient) -> None:
        self._client = client

    @staticmethod
    def engine(family: str) -> str:
        override = (getenv("SANDBOX_MODEL") or "").strip()
        if override:
            low = override.lower()
            if (
                "kimi" in low
                or "glm" in low
                or low.startswith("z-ai/")
                or low.startswith("moonshotai/")
            ):
                return _GOLD_DENSE
            return override
        return _GOLD_DENSE if family in _GOLD_DEEP else _GOLD_QUICK

    def ask(
        self,
        *,
        engine: str,
        transcript: list[dict[str, Any]],
        schemas: list[Any],
        budget: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        ladder = [engine, *[m for m in _GOLD_RELIEF if m != engine]]
        problem: Exception | None = None
        for model in ladder[:3]:
            payload = self._client.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "messages": transcript,
                    "tools": schemas,
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if not payload:
                problem = RuntimeError(model)
                continue
            try:
                message = payload["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                problem = exc
                continue
            return message.get("content") or "", list(message.get("tool_calls") or [])[:budget]
        raise RuntimeError(f"inference failed ({problem})")

    def json_ask(self, *, engine: str, system: str, user: str) -> str:
        ladder = [engine, *[m for m in _GOLD_RELIEF if m != engine]]
        for model in ladder[:3]:
            payload = self._client.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0,
                },
            )
            if not payload:
                continue
            try:
                return str(payload["choices"][0]["message"].get("content") or "")
            except (KeyError, IndexError, TypeError):
                continue
        return ""

    def act(self, envelope: dict[str, Any]) -> dict[str, Any]:
        reply = self._client.post("/environment/call", json_data=envelope)
        if reply is None:
            raise RuntimeError("environment call failed")
        return reply


_wire = GoldBridge(ProxyClient(timeout=120, max_retries=2))


def _is_question(text: str) -> bool:
    body = (text or "").strip().lower()
    if not body:
        return False
    if "?" in body:
        return True
    return any(
        body.startswith(word)
        for word in (
            "which",
            "what",
            "does",
            "did",
            "can you",
            "could you",
            "would you",
            "wait",
            "hold on",
            "are you",
        )
    )


def _market_moved(obs: Any) -> bool:
    if not isinstance(obs, dict):
        return False
    keys = ("market_event", "stock_changed", "price_changed", "event", "events")
    for key in keys:
        if obs.get(key):
            return True
    nested = obs.get("observation")
    if isinstance(nested, dict):
        for key in keys:
            if nested.get(key):
                return True
    return False


def _event_target(obs: Any) -> tuple[str, str]:
    if not isinstance(obs, dict):
        return "", ""
    me = obs.get("market_event") if isinstance(obs, dict) else None
    if isinstance(me, dict):
        tgt = me.get("target") if isinstance(me.get("target"), dict) else {}
        pid = str(tgt.get("product_id") or "")
        sku = str(tgt.get("sku") or "")
        if pid or sku:
            return pid, sku
    nested = obs.get("observation")
    if isinstance(nested, dict):
        return _event_target(nested)
    return "", ""


def _stock_gone(obs: Any) -> bool:
    if not isinstance(obs, dict):
        return False
    blob = json.dumps(obs, ensure_ascii=False, default=str).lower()
    if '"in_stock": false' in blob or '"in_stock":false' in blob:
        return True
    return bool(re.search(r"sold[_\s]?out|unavailable|out[_\s]of[_\s]stock", blob))


def _episode_done(result: dict[str, Any]) -> bool:
    for row in result.get("calls") or []:
        if isinstance(row, dict):
            obs = row.get("observation")
            if isinstance(obs, dict) and obs.get("done"):
                return True
    return False


def _call_name(call: dict[str, Any]) -> str:
    spec = call.get("function") if isinstance(call, dict) else None
    return str(spec["name"]) if isinstance(spec, dict) and spec.get("name") else ""


def _call_id(call: dict[str, Any], fallback: str) -> str:
    return str(call.get("id") or fallback)


def _call_args(call: dict[str, Any]) -> dict[str, Any]:
    spec = call.get("function") if isinstance(call, dict) else None
    raw = spec.get("arguments", "{}") if isinstance(spec, dict) else "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _digest(obs: Any, cap: int = 6000) -> str:
    try:
        text = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obs)
    return text if len(text) <= cap else text[: cap - 18] + "…<clipped>"


_DEEP_TOOLS = frozenset({"view", "view_product", "compare", "compare_variants", "inspect_stock"})


class GoldEpisode:

    def __init__(self, problem_data: dict[str, Any]) -> None:
        env = problem_data["environment"]
        self.binding = env["binding"]
        self.policy = env["policy_view"]
        self.schemas = self.policy.get("tools") or []
        self.brief_text = str(self.policy.get("query") or problem_data.get("query") or "")
        self.key = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        self.per_turn = max(1, int(self.policy.get("max_calls_per_turn", 1)))
        self.brief = GoldBrief(self.brief_text, self.schemas)
        self.led = GoldLedger(max(1, int(self.policy.get("max_steps", 20))))
        self.book = DualLaneBook()
        self.engine = GoldBridge.engine(self.brief.family)


    def run(self) -> list[dict[str, Any]]:
        transcript: list[dict[str, Any]] = [
            {"role": "system", "content": _system(self.brief, self.policy)},
            {"role": "user", "content": self.brief_text},
        ]
        if self.brief.family == TF3 and self.brief.gated:
            transcript.append(
                {
                    "role": "user",
                    "content": (
                        "Opening move: the brief withholds a detail until asked. "
                        "Send message asking for it before any search."
                    ),
                }
            )
        if self.brief.family == TF5 and self.brief.skurefs:
            transcript.append(
                {
                    "role": "user",
                    "content": (
                        "Closed candidate set: "
                        + ", ".join(self.brief.skurefs)
                        + ". Open each one before ranking; pass product_id and sku "
                        "as separate arguments."
                    ),
                }
            )
        trail: list[dict[str, Any]] = [
            {
                "role": "meta",
                "goldengirl": {
                    "version": 11,
                    "family": self.brief.family,
                    "engine": self.engine,
                    "pipeline": "DualLedgerEngine",
                    "provider": "openrouter",
                    "scores": self.brief.scores,
                },
            }
        ]
        while self.led.turn < self.led.limit:
            self.led.turn += 1
            coach = self._coach()
            if coach:
                transcript.append({"role": "user", "content": coach})
            outcome = self._turn(transcript, trail)
            if outcome is None:
                continue
            spoken, calls, result = outcome
            trail.append(
                {
                    "role": "assistant",
                    "content": spoken,
                    "tool_calls": calls,
                    "environment_result": result,
                    "goldengirl_turn": {
                        "family": self.brief.family,
                        "turn": self.led.turn,
                        "brand_lane": sorted(self.book.brand_lane),
                        "firm_lane": sorted(self.book.firm_lane),
                        "rung": self.led.rung,
                        "vetoes": self.led.vetoes,
                    },
                }
            )
            self.led.note([_call_name(c) or "?" for c in calls], calls)
            self._absorb(result, transcript, calls)
            self.book.resort(self.brief)
            if _episode_done(result):
                break
        return trail

    def _turn(
        self, transcript: list[dict[str, Any]], trail: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]] | None:
        for attempt in range(_MAX_VETO_PER_TURN + 1):
            spoken, calls = self._speak(transcript, trail)
            calls = self._block_order_during_event(calls)
            kept, correction = self._veto(calls)
            if kept:
                transcript.append(
                    {"role": "assistant", "content": spoken, "tool_calls": kept}
                )
                return spoken, kept, self._dispatch(kept, self.led.turn)
            self.led.vetoes += len(calls)
            trail.append(
                {
                    "role": "assistant",
                    "content": spoken,
                    "vetoed_tool_calls": calls,
                    "goldengirl_veto": {
                        "turn": self.led.turn,
                        "attempt": attempt + 1,
                        "reason": correction,
                    },
                }
            )
            transcript.extend(
                [
                    {"role": "assistant", "content": spoken},
                    {"role": "user", "content": correction},
                ]
            )
        return None

    def _veto(
        self, calls: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str]:
        if not calls:
            return [], "Post a tool call to the ledger."
        if self.led.left <= 4 or self.led.vetoes >= 3:
            return calls, ""
        labels = [_call_name(c) for c in calls]

        if (
            self.brief.family == TF3
            and self.brief.gated
            and not self.led.asked_gate
        ):
            if not any(name in _TALK_TOOLS for name in labels):
                if any(
                    name in _LOOKUP_TOOLS or name in _SPEND_TOOLS for name in labels
                ):
                    return [], (
                        "Ledger veto: the disclosure wall is still shut. Those lookup/"
                        "spend calls never reached the books. Send exactly one message "
                        "this turn asking for the withheld requirement in plain words, "
                        "then continue."
                    )

        if self.brief.family == TF2 and self.brief.quoted:
            for call, label in zip(calls, labels):
                if label != "add_to_cart":
                    continue
                pid = str(_call_args(call).get("product_id") or "")
                if (
                    pid
                    and self.book.phrase_in_title(pid, self.brief.quoted)
                    and not self.book.phrase_in_facts(pid, self.brief.quoted)
                    and self.led.may_veto("title:" + pid)
                ):
                    return [], (
                        f"Ledger veto: add_to_cart on {pid} was dropped. The quoted "
                        f"phrase \"{self.brief.quoted}\" sits in that product's title "
                        "but in none of its fact rows — a title-only trap. view the "
                        "variant rows for proof, or move to another candidate."
                    )

        if self.brief.family == TF5 and self.brief.skurefs:
            unopened = [
                ref.split("::", 1)[0]
                for ref in self.brief.skurefs
                if ref.split("::", 1)[0] not in self.book.viewed
            ]
            if (
                unopened
                and "add_to_cart" in labels
                and self.led.may_veto("set:" + ",".join(unopened))
            ):
                return [], (
                    "Ledger veto: add_to_cart was dropped because "
                    + ", ".join(unopened[:4])
                    + " from the named set has not been opened yet. A candidate "
                    "you never observed cannot be ranked. view or compare the "
                    "remaining candidates, then cart the leader."
                )

        if self.brief.family == TF6 and not self.led.rung_proof:
            for call, label in zip(calls, labels):
                if label != "add_to_cart":
                    continue
                pid = str(_call_args(call).get("product_id") or "")
                if (
                    pid
                    and pid not in self.book.viewed
                    and self.led.may_veto("evidence:" + pid)
                ):
                    return [], (
                        "Ledger veto: add_to_cart was dropped because nothing yet "
                        "proves the original is unavailable, and "
                        f"{pid} has not been verified either. Evidence order "
                        "matters: inspect_stock or view the item the shopper named, "
                        "let the observation say it is gone, then verify the "
                        "replacement and cart it."
                    )
        return calls, ""


    def _block_order_during_event(
        self, calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not self.led.market_moved or self.led.closed:
            return calls
        event_pid = self.led.event_pid
        event_sku = self.led.event_sku
        if event_pid:
            probe: dict[str, Any] = {"product_id": event_pid}
            if event_sku:
                probe["sku"] = event_sku
            if not calls:
                return calls
            messages: list[dict[str, Any]] = []
            rest: list[dict[str, Any]] = []
            for call in calls:
                name = _call_name(call)
                if name == "message":
                    messages.append(call)
                    continue
                if name in {"search", "filter"}:
                    continue
                if (
                    name == "inspect_stock"
                    and str(_call_args(call).get("product_id") or "") == event_pid
                ):
                    continue
                rest.append(call)
            inspect_call: dict[str, Any] = {
                "id": f"{self.key}-event-inspect",
                "type": "function",
                "function": {
                    "name": "inspect_stock",
                    "arguments": json.dumps(probe),
                },
            }
            return messages + [inspect_call] + rest
        out: list[dict[str, Any]] = []
        for call in calls:
            name = _call_name(call)
            if name != "place_test_order":
                out.append(call)
                continue
            args = _call_args(call)
            pid = str(
                args.get("product_id")
                or self.led.cart_pid
                or self.book.focus_pid
                or self.book.last_pid
                or ""
            )
            sku = str(args.get("sku") or self.led.cart_sku or "")
            if not pid:
                continue
            probe = {"product_id": pid}
            if sku:
                probe["sku"] = sku
            spec = call.get("function")
            if isinstance(spec, dict):
                spec["name"] = "inspect_stock"
                spec["arguments"] = json.dumps(probe)
            out.append(call)
        return out or calls

    def _coach(self) -> str | None:
        led = self.led
        if led.owes_reply:
            return (
                "The shopper asked you something. Reply with message this turn, "
                "before any cart or order call."
            )
        if led.market_moved and not led.closed:
            target = led.event_pid or led.cart_pid or "the carted sku"
            return (
                f"The market moved. inspect_stock the event-target ({target}) "
                "FIRST this turn to close the event — inspecting a replacement does "
                "not count. Then re-check the spend wall against the fresh price, "
                "remove_from_cart if invalid, find a wall-clearing "
                "replacement if needed, inspect that sku"
                + (
                    ", rebuild the claims from fact rows, "
                    if self.brief.needs_claims()
                    else ", "
                )
                + "then place_test_order — event_closed=false pays 0 even when "
                "the cart looks gold. Do not end the run on remove."
            )
        if led.turn <= 2 and not any(
            led.ever(name)
            for name in ("search", "filter", "view", "compare", "message", "inspect_stock")
        ):
            return self._opening_nudge()
        if (
            "remove_from_cart" in led.acted
            and not led.carted
            and not led.closed
            and not led.market_moved
        ):
            return (
                "Cart cleared after the market move. Cart a wall-clearing replacement "
                "NOW, inspect_stock it, then place_test_order — do not leave the run "
                "on remove_from_cart."
            )
        branch = {
            TF1: self._coach_tf1,
            TF2: self._coach_tf2,
            TF3: self._coach_tf3,
            TF4: self._coach_tf4,
            TF5: self._coach_tf5,
            TF6: self._coach_tf6,
            TF7: self._coach_tf7,
        }[self.brief.family]
        tip = branch()
        if tip:
            return tip
        return self._endgame()

    def _opening_nudge(self) -> str:
        if self.brief.family == TF3 and self.brief.gated:
            return (
                "Turn one: ask for the withheld detail with message. No search yet."
            )
        if self.brief.family == TF5 and self.brief.skurefs:
            return (
                "Turn one: open the named candidates ("
                + ", ".join(self.brief.skurefs[:4])
                + ") rather than searching the wider catalog."
            )
        if self.brief.family == TF1:
            return (
                "Turn one: split the ask into walls and wishes, then run a search "
                "that carries the wish wording. A wall-only search comes next."
            )
        return "Begin with search or filter using the shopper's own nouns, then view the hits."

    def _endgame(self) -> str | None:
        led = self.led
        force = led.left <= 4 or (
            self.brief.family in {TF1, TF3} and led.left <= 6
        )
        if led.carted and force and not led.closed:
            return (
                f"Only {led.left} step(s) left. inspect_stock the carted sku if needed, "
                "then place_test_order NOW — an unplaced order scores nothing."
            )
        if not led.carted and force and led.ever("view") and not led.closed:
            return (
                "Step budget is nearly gone. Cart the best verified candidate NOW, "
                "inspect_stock it, then place_test_order — stop open search."
            )
        if led.left <= 1 and not led.closed:
            return "Last step. place_test_order on the best verified candidate immediately."
        if led.carted and led.left <= 6 and not led.closed and led.once("endgame"):
            return (
                f"Only {led.left} step(s) left. inspect_stock the carted sku if needed, "
                "then place_test_order — do not keep searching."
            )
        if not led.carted and led.left <= 8 and led.ever("view") and led.once("late_cart"):
            return (
                "Step budget is nearly gone. Cart the best verified candidate NOW "
                "and leave room for inspect_stock plus place_test_order."
            )
        if led.left <= 1 and not led.closed and led.once("last_step"):
            return "Last step. place_test_order on the best verified candidate immediately."
        return None


    def _coach_tf1(self) -> str | None:
        led, book = self.led, self.book
        if led.closed:
            return None
        if not led.carted and led.refines <= 1 and led.once("tf1_two_queries"):
            wish = ", ".join(self.brief.brand_hints[:3]) or "the wish wording"
            return (
                "Two-lane search: run one query that includes the wish ("
                f"{wish}) and a second query that keeps only the wall wording. "
                "The engine is tracking brand_lane and firm_lane separately and "
                "both need candidates before you commit."
            )
        if not led.carted and led.ever("search") and not book.order and led.once("tf1_empty"):
            return (
                "No products came back yet. Drop the wish token and search the "
                "wall wording alone — the wish is optional, the wall is not."
            )
        if not led.carted and book.viewed:
            brand_ready = book.brand_viewed()
            firm_ready = book.firm_viewed()
            if brand_ready and led.once("tf1_brand_pick"):
                led.lane_pick = "brand_lane"
                pid = sorted(brand_ready)[0]
                if pid in book.firm_lane:
                    return (
                        f"brand_lane has a wall-compliant viewed candidate ({pid}). "
                        "That is your pick: it satisfies the walls AND the wish. "
                        "Cart it, inspect_stock, then order."
                    )
                return (
                    f"brand_lane candidate {pid} matches the wish but does not clear "
                    "every wall. Walls win: verify it again or fall back to a "
                    "firm_lane candidate."
                )
            if not brand_ready and firm_ready and led.once("tf1_firm_pick"):
                led.lane_pick = "firm_lane"
                return (
                    "brand_lane is empty after your views, firm_lane has "
                    f"{len(firm_ready)} verified candidate(s). Buy the firm_lane "
                    "candidate — the wish is unavailable, the walls are not "
                    "negotiable. Do not stretch a wall to chase the wish."
                )
            if not brand_ready and not firm_ready and led.once("tf1_verify"):
                return (
                    "Nothing is verified yet: open a candidate with view and read "
                    "its fact rows (brand, price, availability) so the lanes can "
                    "be filled."
                )
        if led.carted and not led.ever("inspect_stock") and led.once("tf1_inspect"):
            lane = book.lane_of(led.cart_pid)
            return (
                f"Carted item sits in {lane}. inspect_stock it, then place_test_order "
                "with no justification object."
            )
        if led.left <= 6 and not led.closed and led.once(f"tf1_force_finish:{led.turn}"):
            if led.carted:
                return (
                    f"Intent endgame ({led.left} left): inspect_stock if needed, then "
                    "place_test_order NOW — no-order runs score nothing."
                )
            if led.ever("view"):
                return (
                    f"Intent endgame ({led.left} left): cart the best firm-compliant "
                    "candidate NOW, inspect_stock, place_test_order."
                )
        return None


    def _coach_tf2(self) -> str | None:
        led, book = self.led, self.book
        if led.closed:
            return None
        seen = book.seen_product_ids()
        if (
            not led.carted
            and led.ever("search")
            and book.thin()
            and led.refines <= 1
            and led.once("tf2_near_miss")
        ):
            suggestion = _near_miss_query(
                self.brief.ask, self.brief.quoted, book.queries
            )
            hint = (
                f" — try a query close to: {suggestion}"
                if suggestion
                else ""
            )
            return (
                f"That pass came back thin ({book.blurb()}). Spend ONE near-miss "
                f"refine now{hint} — literal tokens only, no filler adjectives and "
                "no price or shipping words. After this refine, stop reformulating "
                "and start viewing."
            )
        if not led.carted and len(seen) < 2:
            if led.ever("search") and led.refines <= 1 and led.once("tf2_refine"):
                target = self.brief.quoted or "the shopper's noun"
                return (
                    f"Only {len(seen)} distinct product(s) surfaced so far. Refine "
                    f"once: keep {target} but change a single token — drop an "
                    "adjective or the maker and keep the noun. One refine, then "
                    "verify what you have."
                )
            if led.ever("search") and led.once("tf2_thin"):
                return (
                    "Recall is still thin. Open what you already have with view "
                    "and read the fact rows; a second distinct product_id in hand "
                    "is what unlocks a confident commitment."
                )
        if self.brief.quoted and not led.carted:
            pid = book.focus_pid or book.last_pid
            if (
                pid
                and book.phrase_in_title(pid, self.brief.quoted)
                and not book.phrase_in_facts(pid, self.brief.quoted)
                and led.once("tf2_title_only")
            ):
                return (
                    f"The quoted phrase \"{self.brief.quoted}\" shows up in the title "
                    f"of {pid} but not in its fact rows. That is a title-only match. "
                    "view the variant rows or try another candidate before carting."
                )
        if not led.carted and len(seen) >= 2 and book.viewed and led.once("tf2_commit"):
            return (
                f"{len(seen)} distinct products observed and {len(book.viewed)} "
                "verified — coverage is sufficient. Pick the candidate whose fact "
                "rows answer the ask, cart it, inspect_stock, then order."
            )
        if led.carted and not led.ever("inspect_stock") and led.once("tf2_inspect"):
            return "Cart is set. inspect_stock the sku, then place_test_order."
        return None


    def _coach_tf3(self) -> str | None:
        led, book = self.led, self.book
        if led.closed:
            return None
        if self.brief.gated and not led.asked_gate:
            if led.turn == 1:
                return (
                    "Withheld detail: send message now and ask for it in plain "
                    "words. Do not search on a guess — the answer decides which "
                    "shelf you search."
                )
            if led.once("tf3_still_unasked"):
                return (
                    "You still have not asked for the withheld detail. Send a "
                    "message with the question this turn; every search before it "
                    "is spent on a guess."
                )
        if led.bad_sku and led.once("tf3_bad_sku:" + led.bad_sku):
            return (
                f"The sku \"{led.bad_sku}\" was never printed in an observation. "
                "Do not improvise sku strings: view the product or compare its "
                "variants until the catalog prints the exact sku, then cart that."
            )
        if led.carted and led.cart_pid:
            pid, sku = led.cart_pid, led.cart_sku
            if not book.known_sku(pid, sku) and led.once("tf3_verify_sku"):
                return (
                    f"Verify on the sku: {sku or '(none)'} for {pid} is not attested "
                    "by any observation yet. inspect_stock that exact sku and read "
                    "the rows before ordering."
                )
            if led.once("tf3_walk_conditions"):
                rows = book.facts.get(pid, {})
                sample = ", ".join(f"{k}={v}" for k, v in list(rows.items())[:8]) or "(none)"
                return (
                    "Walk every stated condition against the carted sku's fact "
                    f"rows before ordering. Observed for {pid}: {sample}. Anything "
                    "you cannot point at in an observation counts as unmet."
                )
        if not led.carted and book.viewed and led.once("tf3_exact"):
            return (
                "Condition check before carting: the figure, the spend wall and "
                "availability must all hold on the specific variant, not on the "
                "product family. Read the option rows, then cart that variant."
            )
        if led.left <= 6 and not led.closed and led.once(f"tf3_force_finish:{led.turn}"):
            if led.carted:
                return (
                    f"Constraint endgame ({led.left} left): inspect_stock the exact "
                    "sku if needed, then place_test_order NOW."
                )
            if led.ever("view") or led.asked_gate:
                return (
                    f"Constraint endgame ({led.left} left): cart the gate-clearing "
                    "variant NOW, inspect_stock, place_test_order — stop searching."
                )
        return None


    def _coach_tf4(self) -> str | None:
        led, book = self.led, self.book
        if led.closed:
            return None
        axes = ", ".join(self.brief.soft_axes[:4]) or "the axis the shopper named"
        deep = len(book.viewed)
        if not led.carted and deep < 2:
            if led.ever("view") and led.once("tf4_second_view"):
                return (
                    f"Only {deep} candidate verified. A preference call needs two: "
                    "view a second hard-valid candidate and read its fact rows on "
                    f"{axes}. Judging a lean from titles is a guess."
                )
            if not led.ever("view") and led.ever("search") and led.once("tf4_first_view"):
                return (
                    "Open the strongest search hit with view first — the fact rows "
                    "are the only evidence for the shopper's stated lean."
                )
        if not led.carted and deep >= 2 and led.once("tf4_compare"):
            names = ", ".join(sorted(book.viewed)[:3])
            return (
                f"Two or more verified candidates ({names}). Now weigh them on "
                f"{axes} using the observed fact rows only, state which wins and "
                "why in one line, then cart the winner. Cheapness is only a "
                "tiebreaker when the named axes are level."
            )
        if led.compares >= 2 and not led.carted and led.once("tf4_stop_comparing"):
            return (
                "Enough comparing. Name the winner on the stated axis and cart it — "
                "a further candidate buys nothing and risks the step budget."
            )
        if led.carted and not led.ever("inspect_stock") and led.once("tf4_inspect"):
            return "Winner is carted. inspect_stock it, then place_test_order."
        return None


    def _coach_tf5(self) -> str | None:
        led, book = self.led, self.book
        if led.closed:
            return None
        named = [ref.split("::", 1)[0] for ref in self.brief.skurefs]
        unopened = [pid for pid in named if pid not in book.viewed]
        if named and unopened and not led.carted and led.once("tf5_open_all:" + str(len(unopened))):
            return (
                "Named candidates not yet opened: "
                + ", ".join(unopened[:4])
                + ". A candidate you never opened cannot be ranked. view or "
                "compare each of them before you sort the board."
            )
        if named and not unopened and not led.carted and led.once("tf5_sort"):
            return (
                "All named candidates are open. Sort them by the stated priority "
                "(first priority decides, second breaks ties, price last), write "
                "the ordering in your reply, then cart the leader. product_id and "
                "sku are separate arguments — never paste the joined id::sku form."
            )
        if led.market_moved and led.once("tf5_resort"):
            return (
                "The board moved. Re-inspect the affected candidate and re-sort: "
                "the winner is whoever leads at the moment you order."
            )
        if led.carted and led.once("tf5_close"):
            return (
                "Leader is carted. inspect_stock it, then place_test_order with "
                "product_id and sku only — no justification object on this family."
            )
        return None


    def _coach_tf6(self) -> str | None:
        led, book = self.led, self.book
        if led.closed:
            return None
        if led.rung == 0:
            if led.ever("search") and not book.order and led.once("tf6_l0_empty"):
                return (
                    "Ladder L0: the original ask returned nothing. Before you "
                    "concede anything, inspect_stock or view the exact item the "
                    "shopper named so an observation — not a hunch — proves it is "
                    "gone."
                )
            if led.rung_proof and led.once("tf6_step_l1"):
                led.rung = 1
                return (
                    "Ladder L0 → L1: an observation confirms the original is "
                    "unavailable. Concede the cosmetic detail only (shade, finish, "
                    "trim). Hold the product class, the spend wall, the spec and "
                    "availability, and tell the shopper with message what you are "
                    "giving up."
                )
            if led.ever("view") and led.once("tf6_l0_hold"):
                return (
                    "Ladder L0: hold every condition for now. Search the ask "
                    "exactly as the shopper wrote it; only a failed observation "
                    "unlocks the first concession. Substitute carts are vetoed "
                    "until rung_proof lands."
                )
            if not led.rung_proof and led.once("tf6_prove_hard"):
                return (
                    "Evidence order: prove the original unavailable via "
                    "inspect_stock or view before any substitute cart. Soft "
                    "concessions wait until that observation posts."
                )
            return None
        if led.rung == 1:
            if led.rung_proof and led.once("tf6_step_l2"):
                led.rung = 2
                return (
                    "Ladder L1 → L2: the cosmetic concession also came up empty. "
                    "Now release the specific model or maker. Still hold the "
                    "product class, the spend wall, the spec and availability — "
                    "those never go."
                )
            if led.once("tf6_l1_work"):
                return (
                    "Ladder L1: search the same item with the cosmetic detail "
                    "dropped and everything else intact. inspect_stock the "
                    "candidate BEFORE the first add_to_cart so the run has an "
                    "intended commitment."
                )
            return None
        if led.rung >= 2 and led.once("tf6_l2_floor"):
            return (
                "Ladder L2 is the floor. Take the closest in-class, in-budget, "
                "in-stock item you have verified, inspect_stock it BEFORE "
                "add_to_cart, then cart, inspect again, and order. Never concede "
                "the product class, never break the spend wall, never cart "
                "something an observation calls unavailable."
            )
        if (
            not led.carted
            and led.ever("view")
            and not led.ever("inspect_stock")
            and led.once("tf6_precart_inspect")
        ):
            return (
                "Before the first add_to_cart: inspect_stock the candidate so "
                "the environment records an intended commitment."
            )
        if led.carted and not led.ever("inspect_stock") and led.once("tf6_inspect"):
            return "Replacement is carted. inspect_stock it, then place_test_order."
        if led.carted and not led.closed and led.left <= 6 and led.once("tf6_close"):
            return (
                f"Only {led.left} step(s) left on recovery. inspect_stock if needed, "
                "then place_test_order NOW — an unclosed recovery run scores nothing."
            )
        if (
            not led.carted
            and led.left <= 8
            and led.ever("view")
            and led.once("tf6_late_cart")
        ):
            return (
                "Recovery budget is thin. Cart the best wall-clearing verified "
                "survivor NOW, inspect_stock it, then place_test_order."
            )
        return None


    def _coach_tf7(self) -> str | None:
        led, book = self.led, self.book
        if led.closed:
            return None
        pid = book.focus_pid or book.last_pid or led.cart_pid
        if led.once("tf7_field_class"):
            return (
                "Voucher shape: claim FIELD = conventional attribute named by the "
                "ask/schema; VALUE = printed fact-row token. Never meta/identity "
                "fields. Brand alone is not enough if a hard gate is still unclaimed."
            )
        if pid:
            rows = book.facts.get(pid, {})
            has_gate_row = any(
                ClaimDesk._gate_worthy(k, str(v or ""), self.brief.ask)
                for k, v in rows.items()
            )
            if not has_gate_row and led.once(f"tf7_rows:{pid}"):
                return (
                    f"Fact rows / listing for {pid} are still thin. view (or re-view) "
                    "that sku so ClaimDesk can attest a non-brand gate claim from "
                    "printed fact_rows before place_test_order."
                )
        if not led.carted and pid:
            if book.title_decoy(pid, self.brief.ask) and led.once("tf7_decoy"):
                return (
                    "Title decoy: the asked figure is in the title of "
                    f"{pid} but not in its fact rows. Keep viewing — do not cart "
                    "this sku. The first commitment must be fact-backed."
                )
            if (
                led.ever("view")
                and not book.fact_backed(pid, self.brief.ask)
                and led.once("tf7_fact_backed")
            ):
                return (
                    "Commit only on evidence: cart a sku whose fact rows overlap "
                    "the shopper's ask (one attested token is enough for a stacked "
                    "ask). Keep viewing until a fact row prints it."
                )
        if (
            not led.carted
            and led.ever("view")
            and led.left <= max(led.limit // 2, 5)
            and led.once("tf7_mid_budget")
        ):
            return (
                "Past mid-budget: if a viewed sku already attests the gate and "
                "fits the wall and stock, cart it now and keep room for "
                "inspect_stock plus place_test_order. Stop synonym-hunting."
            )
        return _vouch_tip(led.turn, book, self.brief)


    def _shape(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        payload = dict(args)
        if self.brief.family == TF5:
            payload = self._split_skuref(payload)
        if name in {"search", "filter"}:
            query = payload.get("query") or payload.get("q") or payload.get("keywords")
            self.book.remember_query(query)
        if name in {"add_to_cart", "view", "view_product", "inspect_stock"}:
            pid = str(payload.get("product_id") or "")
            if pid:
                self.book.focus_pid = pid
        if name == "add_to_cart" and self.brief.family == TF3:
            pid = str(payload.get("product_id") or "")
            sku = str(payload.get("sku") or "")
            if pid and sku and not self.book.known_sku(pid, sku):
                self.led.bad_sku = sku
        if name != "place_test_order":
            return payload
        if len(self.brief.skurefs) >= 2:
            payload.pop("justification", None)
            payload.pop("claims", None)
            return payload
        if self.brief.family == TF7:
            pid = str(
                payload.get("product_id")
                or self.led.cart_pid
                or self.book.focus_pid
                or self.book.last_pid
            )
            note = payload.get("justification")
            raw = note.get("claims") if isinstance(note, dict) else None
            payload["justification"] = {
                "claims": ClaimDesk.polish(raw, self.brief, self.book, pid)
            }
            if pid and not payload.get("product_id"):
                payload["product_id"] = pid
            return payload
        payload.pop("justification", None)
        payload.pop("claims", None)
        return payload

    def _split_skuref(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = dict(args)
        pid = str(payload.get("product_id") or "")
        hit = _SKUREF_RX.search(pid)
        if hit and "::" in pid:
            payload["product_id"] = hit.group(1)
            if not payload.get("sku"):
                payload["sku"] = hit.group(2)
        sku = str(payload.get("sku") or "")
        hit = _SKUREF_RX.search(sku)
        if hit and "::" in sku:
            payload["sku"] = hit.group(2)
            if not payload.get("product_id"):
                payload["product_id"] = hit.group(1)
        return payload


    def _dispatch(self, calls: list[dict[str, Any]], turn: int) -> dict[str, Any]:
        batch = f"{self.key}-turn-{turn}"
        actions: list[dict[str, Any]] = []
        for slot, call in enumerate(calls, start=1):
            name = _call_name(call)
            if not name:
                raise RuntimeError("tool call without a name")
            args = self._shape(name, _call_args(call))
            spec = call.get("function")
            if isinstance(spec, dict):
                spec["arguments"] = json.dumps(args)
            actions.append(
                {"call_id": f"{batch}-{slot}", "action": {"name": name, "args": args}}
            )
        return _wire.act(
            {
                **self.binding,
                "call_id": batch,
                "idempotency_key": batch,
                "turn": turn,
                "calls": actions,
            }
        )

    def _speak(
        self, transcript: list[dict[str, Any]], trail: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        for attempt in range(len(_NUDGE) + 1):
            spoken, calls = _wire.ask(
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
                    {"role": "user", "content": _NUDGE[min(attempt, len(_NUDGE) - 1)]},
                ]
            )
        raise RuntimeError("engine would not emit an executable action")

    def _absorb(
        self,
        result: dict[str, Any],
        transcript: list[dict[str, Any]],
        calls: list[dict[str, Any]],
    ) -> None:
        rows = result.get("calls") or []
        for index, call in enumerate(calls, start=1):
            row = rows[index - 1] if index - 1 < len(rows) else None
            obs = row.get("observation") if isinstance(row, dict) else None
            name = _call_name(call)
            self.book.absorb(obs, deep=name in _DEEP_TOOLS)
            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": _call_id(call, f"{self.key}-{self.led.turn}-{index}"),
                    "content": _digest(obs),
                }
            )
            if _market_moved(obs):
                self.led.market_moved = True
                ep, es = _event_target(obs)
                snap = (
                    ep
                    or self.led.cart_pid
                    or self.book.focus_pid
                    or self.book.last_pid
                    or ""
                )
                if snap:
                    self.led.event_pid = snap
                    if es:
                        self.led.event_sku = es
                    elif self.led.cart_sku:
                        self.led.event_sku = self.led.cart_sku
            if self.brief.family == TF6 and name in _DEEP_TOOLS and _stock_gone(obs):
                self.led.rung_proof = True
            if self.brief.family == TF6 and name in {"search", "filter"}:
                empty = not (isinstance(obs, dict) and obs.get("results"))
                if empty and self.led.refines >= 2:
                    self.led.rung_proof = True
        note = result.get("user_message")
        if isinstance(note, dict) and note.get("content"):
            text = str(note["content"])
            transcript.append({"role": "user", "content": text})
            if _is_question(text):
                self.led.owes_reply = True
            if self.brief.family == TF3 and self.brief.gated:
                self.led.asked_gate = True
        if self.led.market_moved and not self.led.closed:
            tip = (
                "Event note: inspect_stock the event-target sku now "
                f"({self.led.event_pid or self.led.cart_pid or 'carted'}) "
                "— that closes the event; inspecting a replacement does not. "
                "Order only after that close."
            )
            if self.brief.needs_claims():
                tip += (
                    " Then rebuild justification.claims from that fresh observation "
                    "before place_test_order."
                )
            transcript.append({"role": "user", "content": tip})


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    return GoldEpisode(problem_data).run()


__all__ = ["agent_main"]
