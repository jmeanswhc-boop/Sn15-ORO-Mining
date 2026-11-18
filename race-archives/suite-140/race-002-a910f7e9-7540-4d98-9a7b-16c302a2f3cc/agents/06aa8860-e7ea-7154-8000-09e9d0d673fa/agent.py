from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_HG_FAST = "deepseek/deepseek-v3.2"
_HG_HEAVY = "qwen/qwen3.5-397b-a17b"
_HG_SPARE = (
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

_ALL_FAMILIES = (TF3, TF7, TF5, TF6, TF1, TF4, TF2)
_HEAVY_FAMILIES = frozenset({TF3, TF4, TF5, TF7})

_TIE_ORDER = {TF3: 7, TF7: 6, TF6: 5, TF5: 4, TF4: 3, TF1: 2, TF2: 1}

_ASK_SHAPE: tuple[tuple[str, str, int], ...] = (

    (TF3, r"\bask\s+me\b|\bif\s+you\s+ask\b|\buntil\s+you\s+ask\b", 6),
    (TF3, r"\bi\s+have\s+(?:a|one)\s+more\b|\bthere'?s\s+something\s+else\b", 4),
    (TF3, r"\bnot\s+going\s+to\s+(?:say|mention|tell)\b|\bkeeping\s+\w+\s+to\s+myself\b", 5),
    (TF3, r"\bhas\s+to\s+be\b|\bneeds?\s+to\s+be\s+exactly\b|\bno\s+substitut", 3),
    (TF3, r"\bevery\s+(?:one\s+of\s+)?(?:my\s+)?(?:condition|requirement)s?\b", 3),

    (TF1, r"\band\s+(?:also|ideally|if\s+possible)\b|\bon\s+top\s+of\s+that\b", 4),
    (TF1, r"\bwould\s+love\b|\bwould\s+be\s+great\b|\bbonus\s+if\b", 4),
    (TF1, r"\bas\s+long\s+as\b|\bprovided\s+that\b", 3),
    (TF1, r"\bdeal[-\s]?breaker\b|\bcan'?t\s+live\s+without\b", 3),

    (TF2, r"\bit\s+says\b|\bthe\s+page\s+says\b|\bthe\s+listing\s+(?:says|reads)\b", 5),
    (TF2, r"\bsomething\s+(?:like|called|named)\b|\bgoes\s+by\b", 4),
    (TF2, r"\bshow\s+me\s+(?:everything|all)\b|\bwhat\s+else\s+is\s+there\b", 3),
    (TF2, r"\bi\s+saw\s+(?:one|it)\b|\bi\s+remember\s+seeing\b", 3),

    (TF4, r"\bif\s+i\s+had\s+to\s+(?:choose|pick)\b|\btorn\s+between\b", 5),
    (TF4, r"\bmore\s+important\s+than\b|\bover\s+the\s+other\b|\bbeats\b", 4),
    (TF4, r"\bhappy\s+to\s+pay\s+more\b|\bdon'?t\s+mind\s+spending\b", 4),
    (TF4, r"\bwhichever\b|\beither\s+is\s+fine\s+but\b", 3),

    (TF5, r"\bhere\s+are\b[^.]{0,30}\b(?:options|candidates|picks)\b", 5),
    (TF5, r"\bput\s+them\s+in\s+order\b|\bnumber\s+them\b|\bfrom\s+best\b", 5),
    (TF5, r"\bmy\s+priority\s+is\b|\bin\s+this\s+order\b|\bthen\s+by\b", 4),
    (TF5, r"\bpick\s+(?:the\s+)?(?:winner|best\s+of\s+these)\b", 3),

    (TF6, r"\bit'?s\s+gone\b|\bcan'?t\s+find\s+it\s+any\s?more\b|\bnot\s+there\b", 6),
    (TF6, r"\banything\s+(?:similar|like\s+it|close)\b|\bwhat\s+can\s+i\s+get\s+instead\b", 4),
    (TF6, r"\bran\s+out\b|\bstock(?:ed)?\s+out\b|\bdiscontinued\b", 4),
    (TF6, r"\bsold\s+out\b|\bdisappear(?:ed|s)?\b|\bno\s+longer\b|\bwithdrawn\b", 4),
    (TF6, r"\bclosest\s+(?:match|thing|one)\b|\bnearest\s+alternative\b", 4),
    (TF6, r"\bi'?ll\s+settle\s+for\b|\bgood\s+enough\b|\bcompromise\b", 3),

    (TF7, r"\btell\s+me\s+why\b|\bhow\s+come\b|\bon\s+what\s+basis\b", 5),
    (TF7, r"\bback\s+(?:it|that)\s+up\b|\bprove\s+it\b|\bcite\b", 5),
    (TF7, r"\bwhat\s+makes\s+(?:it|this)\b|\bconvince\s+me\b", 4),
    (TF7, r"\bexplain\s+(?:your|the|why)\b|\bshow\s+me\s+the\s+evidence\b", 5),
    (TF7, r"\bfor\s+the\s+record\b|\bwrite\s+(?:it|that)\s+down\b", 3),
)

_RULE_SHAPE: tuple[tuple[str, str, int], ...] = (
    (TF3, r"\bhidden\b|\bnot\s+stated\s+up\s?front\b|\bon\s+request\b", 6),
    (TF3, r"\bevery\s+constraint\b|\bfull\s+compliance\b|\bexact\s+variant\b", 4),
    (TF1, r"\bmandatory\b[^.]{0,40}\boptional\b|\brank\s+the\s+requirements\b", 4),
    (TF1, r"\bdo\s+not\s+trade\b[^.]{0,30}\brequirement\b", 3),
    (TF2, r"\bquery\s+again\b|\bsecond\s+pass\b|\bwiden\s+the\s+search\b", 4),
    (TF2, r"\bsurface\b|\bmiss(?:ed|ing)\s+candidates?\b", 3),
    (TF4, r"\bstated\s+preference\b|\bhonour\b|\bhonor\b|\blean\b", 4),
    (TF4, r"\bequally\s+valid\b|\bboth\s+qualify\b", 3),
    (TF5, r"\bexactly\s+these\b|\bclosed\s+set\b|\bdo\s+not\s+add\b", 5),
    (TF5, r"\btie[-\s]?break\b|\bsecondary\s+criteri", 3),
    (TF6, r"\bgraceful\b|\bnearest\s+acceptable\b|\bsubstitution\b", 5),
    (TF6, r"\bmid[-\s]?episode\b|\bchanges\s+while\b", 3),
    (TF7, r"\bclaims?\b[^.]{0,30}\b(?:field|value)\b|\bjustification\s+object\b", 6),
    (TF7, r"\bsupported\s+by\s+observ|\bno\s+unverified\b", 4),
)

_ASK_RX = tuple((fam, re.compile(rx, re.I), w) for fam, rx, w in _ASK_SHAPE)
_RULE_RX = tuple((fam, re.compile(rx, re.I), w) for fam, rx, w in _RULE_SHAPE)

_PAIR_RX = re.compile(r"\b(\d{5,})::([\w.-]+)")
_LID_RX = re.compile(
    r"(?:keep\s+it\s+under|stay\s+(?:under|below)|under|below|less\s+than|"
    r"up\s+to|no\s+more\s+than|at\s+most|cap(?:ped)?\s+at|budget\s+of|"
    r"max(?:imum)?(?:\s+of)?)\s*[^\d]{0,4}([\d][\d,]*(?:\.\d+)?)",
    re.I,
)
_QUOTE_RX = re.compile(r"[\"\u201c\u2018']([^\"\u201d\u2019']{3,64})[\"\u201d\u2019']")
_RECEIPT_RX = re.compile(
    r"\bwhy\b|\bjustif|\bprove\b|\bback\s+it\s+up\b|\bbasis\b|\bconvince\b|"
    r"\bexplain\b|\bevidence\b|\breceipts?\b|\breasons?\b|\bdefend\b|"
    r"\bwalk me through\b|\bgive me a reason\b",
    re.I,
)

_HIDDEN_RX = re.compile(
    r"\bask\s+me\b|\bif\s+you\s+ask\b|\buntil\s+you\s+ask\b|\bon\s+request\b|"
    r"\b(?:tell|share|reveal|mention|say)\b[^.]{0,44}\b(?:if|when|once)\b[^.]{0,26}\bask\b|"
    r"\bnot\s+going\s+to\s+(?:say|volunteer|mention)\b|\bkeeping\s+\w+\s+to\s+myself\b|"
    r"\bhidden\b|\bon\s+my\s+mind\b",
    re.I,
)
_LEAN_RX = re.compile(
    r"\b(?:torn\s+between|if\s+i\s+had\s+to\s+choose|more\s+important\s+than|"
    r"happy\s+to\s+pay\s+more|rather|prefer(?:ably)?|matters?\s+more|lighter|"
    r"quieter|roomier|sturdier|cheaper|cheapest)\b",
    re.I,
)
_LEGAL_FIELD_RX = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_SPECNUM_RX = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s?(?:inch|in|hz|fps|p|k|gb|tb|mb|mm|cm|mp|w|wh|mah|"
    r"oz|ml|l|kg|lb|ft|yd|\"|'))?\b",
    re.I,
)

_BASE_FIELDS = frozenset({"price", "in_stock"})

_NO_FIELD = frozenset(
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
    }
)

_CONTAINERS = ("display", "facts", "options", "os", "size", "title", "spec", "specs")

_PALETTE = frozenset(
    {
        "xs", "s", "m", "l", "xl", "xxl", "petite", "tall", "plus",
        "small", "medium", "large", "mini", "compact", "oversized", "roomy",
        "cotton", "polyester", "wool", "cashmere", "silk", "linen", "rayon",
        "leather", "suede", "denim", "canvas", "nylon", "mesh", "rubber",
        "plastic", "metal", "aluminum", "steel", "wood", "bamboo", "glass",
        "ceramic", "marble", "oak", "walnut", "maple", "pine", "rattan",
        "black", "white", "ivory", "cream", "navy", "blue", "teal", "green",
        "olive", "red", "burgundy", "pink", "purple", "yellow", "orange",
        "gray", "grey", "silver", "gold", "brown", "tan", "beige", "khaki",
        "charcoal", "bronze", "copper", "chrome", "brushed", "sand", "stone",
        "narrow", "wide", "regular", "slim", "relaxed", "straight", "cropped",
        "matte", "glossy", "gloss", "satin", "textured", "smooth", "ribbed",
        "hd", "fhd", "qhd", "uhd", "oled", "lcd", "led", "ips", "retina",
        "wireless", "wired", "waterproof", "insulated", "foldable", "padded",
    }
)

_CHATTER = frozenset(
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
        "prefer", "preferably", "torn", "between", "ask", "hidden",
    }
)

_LOOKUP_TOOLS = frozenset({"search", "filter", "find_product", "search_products"})
_PROBE_TOOLS = frozenset({"view", "view_product", "compare", "compare_variants", "inspect_stock"})
_SPEND_TOOLS = frozenset({"add_to_cart", "place_test_order", "remove_from_cart", "update_cart"})
_TALK_TOOLS = frozenset({"message", "send_message", "ask_user"})

def _key(raw: Any) -> str:
    low = str(raw or "").strip().lower()
    low = low.replace("-", "_").replace(" ", "_").replace("/", "_")
    low = re.sub(r"[^a-z0-9_]+", "", low)
    return re.sub(r"_+", "_", low).strip("_")

def _squash(raw: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(raw or "").lower())

def _printed(value: Any, blob: Any) -> bool:
    needle = _squash(value)
    return bool(needle) and needle in _squash(blob)

def _inside(token: Any, text: Any) -> bool:
    tok = str(token or "").strip().lower()
    blob = str(text or "").lower()
    if not tok or not blob:
        return False
    if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", blob):
        return True
    packed = _squash(tok)
    if not packed:
        return False
    hay = _squash(blob)
    if len(packed) <= 3:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(packed)}(?![a-z0-9])", hay))
    return packed in hay

def _spec_tokens(text: str) -> list[str]:
    body = (text or "").lower()
    out: list[str] = []
    seen: set[str] = set()
    for hit in _SPECNUM_RX.finditer(body):
        tok = hit.group(0).strip()
        packed = _squash(tok)
        if packed and packed not in seen:
            seen.add(packed)
            out.append(tok)
    for word in re.findall(r"[a-z][a-z0-9-]{1,}", body):
        if word in _CHATTER or word not in _PALETTE:
            continue
        packed = _squash(word)
        if packed and packed not in seen:
            seen.add(packed)
            out.append(word)
    return out

def _welded_field(field: str) -> bool:
    name = _key(field)
    if not name:
        return False
    for box in _CONTAINERS:
        if name.startswith(box + "_") or name.endswith("_" + box):
            return True
    return False

def _halve(brief: str) -> tuple[str, str]:
    head, sep, tail = (brief or "").partition("\n\n")
    if not sep:
        return (brief or "").strip(), ""
    body = tail.strip()
    first, _, rest = body.partition("\n")
    if first.rstrip().endswith(":") and len(first) < 42:
        body = rest.strip()
    return head.strip(), body

def _once_each(items: Any) -> list[str]:
    return list(dict.fromkeys(str(i) for i in items))

def _amount(raw: Any) -> float | None:
    text = re.sub(r"[^\d.]", "", str(raw or ""))
    if not text or text.count(".") > 1:
        return None
    try:
        return float(text)
    except ValueError:
        return None

def _json_obj(text: str) -> dict[str, Any]:
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

def _tool_labels(tools: list[Any]) -> list[str]:
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

def _claim_field_schema(tools: list[Any]) -> dict[str, Any] | None:
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

def _schema_enum(tools: list[Any]) -> list[str]:
    field = _claim_field_schema(tools)
    values = field.get("enum") if isinstance(field, dict) else None
    if not isinstance(values, list):
        return []
    out, seen = [], set()
    for raw in values:
        name = _key(raw)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out

def _schema_sentence(tools: list[Any]) -> str:
    field = _claim_field_schema(tools)
    if isinstance(field, dict):
        return str(field.get("description") or "")[:600]
    return ""

class GateBrief:

    __slots__ = (
        "raw",
        "ask",
        "rules",
        "family",
        "tally",
        "lid",
        "pairs",
        "quote",
        "wants_receipt",
        "hidden_spec",
        "leans",
        "maker_words",
        "claim_enum",
        "claim_sentence",
    )

    def __init__(self, raw: str, tools: list[Any] | None = None) -> None:
        self.raw = raw or ""
        self.ask, self.rules = _halve(self.raw)
        self.pairs = _once_each(f"{pid}::{sku}" for pid, sku in _PAIR_RX.findall(self.raw))
        lid = _LID_RX.search(self.ask) or _LID_RX.search(self.raw)
        self.lid = lid.group(1) if lid else ""
        quote = _QUOTE_RX.search(self.ask) or _QUOTE_RX.search(self.raw)
        self.quote = quote.group(1).strip() if quote else ""

        self.wants_receipt = bool(_RECEIPT_RX.search(self.ask))
        self.hidden_spec = bool(_HIDDEN_RX.search(self.raw))
        self.leans = _once_each(m.group(0).lower() for m in _LEAN_RX.finditer(self.ask))
        self.maker_words = self._maker_words()
        self.claim_enum = _schema_enum(tools or [])
        self.claim_sentence = _schema_sentence(tools or [])
        self.tally, self.family = self._read()

    def _read(self) -> tuple[dict[str, int], str]:
        tally = dict.fromkeys(_ALL_FAMILIES, 0)

        if len(self.pairs) >= 2:
            tally[TF5] = 99
            return tally, TF5

        if self.wants_receipt:
            tally[TF7] = 99
            return tally, TF7
        for fam, rx, weight in _ASK_RX:
            if rx.search(self.ask):
                tally[fam] += weight
        for fam, rx, weight in _RULE_RX:
            if fam == TF7:
                continue
            if rx.search(self.rules):
                tally[fam] += weight

        if self.hidden_spec:
            tally[TF3] += 8
        if len(self.pairs) == 1:
            tally[TF5] += 2
        if self.wants_receipt:
            tally[TF7] += 6
        if self.quote:
            tally[TF2] += 5
        if len(self.leans) >= 2:
            tally[TF4] += 5
        if re.search(r"\bwould\s+love\b|\bbonus\b|\bideally\b", self.ask, re.I) and re.search(
            r"\bmust\b|\bneed\b|\bdeal[-\s]?breaker\b", self.ask, re.I
        ):
            tally[TF1] += 5
        if re.search(r"\binstead\b|\balternative\b", self.ask, re.I):
            tally[TF6] += 3
        top = max(tally.values())
        if top <= 0:
            return tally, TF1
        front = [fam for fam, score in tally.items() if score == top]
        return tally, max(front, key=lambda fam: _TIE_ORDER[fam])

    def _maker_words(self) -> list[str]:
        out: list[str] = []
        for word in re.findall(r"\b[A-Z][A-Za-z0-9&.-]{2,}\b", self.ask):
            low = word.lower()
            if low in _CHATTER or low in _PALETTE:
                continue
            out.append(low)
        return _once_each(out)

    def needs_claims(self) -> bool:

        return self.family == TF7

    def lid_value(self) -> float | None:
        return _amount(self.lid)

    def dossier(self) -> str:
        rows: list[str] = []
        if self.lid:
            rows.append(f"- spend lid: {self.lid}")
        if self.quote:
            rows.append(f"- quoted listing text: {self.quote}")
        if self.pairs:
            rows.append(f"- candidates handed to you: {', '.join(self.pairs)}")
        if self.hidden_spec:
            rows.append("- the shopper is holding a specification back until asked")
        if self.leans:
            rows.append(f"- lean words: {', '.join(self.leans[:6])}")
        if self.maker_words:
            rows.append(f"- proper nouns in the ask: {', '.join(self.maker_words[:6])}")
        if self.claim_enum:
            rows.append(f"- live schema claim fields: {', '.join(self.claim_enum)}")
        if self.claim_sentence:
            rows.append(f"- live schema claim rule: {self.claim_sentence[:280]}")
        return ("GATE DOSSIER\n" + "\n".join(rows) + "\n") if rows else ""

class SpecVault:

    __slots__ = ("gate_open", "asked", "revealed_tokens", "question_turn", "answer_text")

    def __init__(self, armed: bool) -> None:

        self.gate_open = not armed
        self.asked = False
        self.revealed_tokens: list[str] = []
        self.question_turn = 0
        self.answer_text = ""

    def mark_asked(self, turn: int) -> None:
        if not self.asked:
            self.question_turn = turn
        self.asked = True

    def take_answer(self, text: str) -> None:
        body = str(text or "").strip()
        if not body:
            return
        self.answer_text = (self.answer_text + " " + body).strip()[:1200]
        for tok in _spec_tokens(body):
            if tok not in self.revealed_tokens:
                self.revealed_tokens.append(tok)
        if self.asked or self.revealed_tokens:
            self.gate_open = True

    def blocked(self) -> bool:
        return not self.gate_open and not self.asked

    def digest(self) -> str:
        if self.revealed_tokens:
            return "revealed: " + ", ".join(self.revealed_tokens[:8])
        if self.asked:
            return "question sent, waiting on the answer"
        return "nothing disclosed yet"

class GateWell:

    __slots__ = (
        "rows",
        "blurb",
        "headline",
        "seen",
        "probed",
        "skus",
        "pin",
        "last_pid",
        "last_price",
        "last_stock",
        "gone",
    )

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, str]] = {}
        self.blurb: dict[str, str] = {}
        self.headline: dict[str, str] = {}
        self.seen: list[str] = []
        self.probed: set[str] = set()
        self.skus: dict[str, set[str]] = {}
        self.pin = ""
        self.last_pid = ""
        self.last_price = ""
        self.last_stock = ""
        self.gone: set[str] = set()

    def keep(self, pid: str, label: str, value: Any) -> None:
        name = _key(label)
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
                self.gone.add(pid)
            self.last_stock = text
        if name == "price":
            self.last_price = text
        self.rows.setdefault(pid, {})[name] = text
        self.last_pid = pid
        if pid not in self.seen:
            self.seen.append(pid)

    def soak(self, obs: Any, probed: bool = False) -> None:
        if not isinstance(obs, dict):
            return

        def walk(node: Any, pid: str = "") -> None:
            if not isinstance(node, dict):
                return
            here = str(node.get("product_id") or "") or pid
            if here and here not in self.seen:
                self.seen.append(here)
            headline = str(node.get("title") or "")
            if headline and here:
                self.headline[here] = headline
            bits: list[str] = []
            if headline:
                bits.append(headline)
            opts = node.get("options")
            if opts:
                bits.append(str(opts))
                if here:
                    for token in re.findall(r"[\w.-]{2,}", str(opts)):
                        if any(ch.isdigit() for ch in token) or "-" in token:
                            self.skus.setdefault(here, set()).add(token)
            if here and node.get("sku"):
                self.skus.setdefault(here, set()).add(str(node["sku"]))
            for label in ("price", "in_stock"):
                if label in node:
                    self.keep(here, label, node.get(label))
            if node.get("brand"):
                self.keep(here, "brand", node.get("brand"))
            drawer = node.get("facts")
            if isinstance(drawer, dict):
                _pour(drawer, here, self, bits)
            if here and bits:
                prior = self.blurb.get(here, "")
                chunk = " | ".join(bits)
                if chunk not in prior:
                    self.blurb[here] = (prior + " | " + chunk if prior else chunk)[:2400]
            if here and probed and (node.get("facts") or node.get("price") is not None):
                self.probed.add(here)
            for label in (
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
                child = node.get(label)
                if isinstance(child, dict):
                    walk(child, here)
                elif isinstance(child, list):
                    for item in child[:12]:
                        walk(item, here)

        walk(obs)

    def row_list(self, pid: str) -> list[dict[str, str]]:
        return [{"key": k, "value": v} for k, v in self.rows.get(pid, {}).items()]

    def receipt(self, pid: str) -> str:
        observed = self.rows.get(pid, {})
        return self.blurb.get(pid, "") + " | " + " | ".join(
            f"{k}={v}" for k, v in observed.items()
        )

    def row_blob(self, pid: str) -> str:
        return " ".join(f"{k} {v}" for k, v in self.rows.get(pid, {}).items()).lower()

    def headline_only(self, pid: str, needle: str) -> bool:
        if not needle:
            return False
        title = self.headline.get(pid, "")
        if not title or not _printed(needle, title):
            return False
        return not _printed(needle, self.row_blob(pid))

    def decoy(self, pid: str, ask: str) -> bool:
        title = self.headline.get(pid, "").lower()
        if not title or not ask:
            return False
        blob = self.row_blob(pid)
        return any(
            _inside(tok, title) and not _inside(tok, blob) for tok in _spec_tokens(ask)
        )

    def row_backed(self, pid: str, ask: str) -> bool:
        if not pid or pid not in self.rows:
            return False
        blob = self.row_blob(pid)
        tokens = _spec_tokens(ask)
        if not tokens:
            return any(k not in _BASE_FIELDS for k in self.rows.get(pid, {}))
        return any(_inside(tok, blob) for tok in tokens)

    def covers_revealed(self, pid: str, tokens: list[str]) -> bool:
        if not tokens:
            return True
        blob = self.row_blob(pid)
        return any(_inside(tok, blob) for tok in tokens)

    def lid_ok(self, pid: str, lid: float | None) -> bool:
        observed = self.rows.get(pid, {})
        if observed.get("in_stock") == "false":
            return False
        if lid is not None:
            price = _amount(observed.get("price"))
            if price is not None and price > lid:
                return False
        return True

    def sku_attested(self, pid: str, sku: str) -> bool:
        if not sku:
            return False
        if sku in self.skus.get(pid, set()):
            return True
        return _printed(sku, self.receipt(pid))

def _pour(
    drawer: dict[str, Any],
    pid: str,
    well: GateWell,
    bits: list[str],
    prefix: str = "",
) -> None:
    for label, val in drawer.items():
        path = f"{prefix}.{label}" if prefix else str(label)
        if isinstance(val, dict):
            bits.append(f"{path}={json.dumps(val, ensure_ascii=False)[:160]}")
            _pour(val, pid, well, bits, path)
            well.keep(pid, str(label), json.dumps(val, ensure_ascii=False)[:160])
        elif isinstance(val, list):
            text = ", ".join(str(x) for x in val[:12])
            bits.append(f"{path}={text}")
            well.keep(pid, str(label), text)
        else:
            text = str(val).strip()
            if not text:
                continue
            bits.append(f"{path}={text}")
            well.keep(pid, str(label), text)
            if prefix:
                well.keep(pid, path.rsplit(".", 1)[-1], text)

_TF7_FILL = """
You are the AskGate receipt desk for ONE missing hard-gate claim.
Input JSON keys: goal, allowed_fields (live schema enum; may be empty),
schema_rule (the tool's own wording about which field names are legal), have
(fields already claimed — often only price / in_stock / brand), listing (text
observed for the exact sku), fact_rows (key/value pairs; catalog drawer keys
here are VALUE evidence only, and drawer naming differs by category).
Reply with JSON only: {"claims":[{"field":"...","value":"..."}, ...]}

WHAT COUNTS AS THE GATE
- Cover the shopper's HARD gate that `have` has not covered yet. price,
  in_stock, and brand alone do NOT satisfy a gate ask — you still owe the
  conventional product attribute.
- FIELD = a conventional snake_case shopping attribute, never a meta label.
  Examples of the right kind of field: size, material, color, width, capacity,
  dimensions, finish, resolution, screen_size. Pick the one that matches what
  the shopper actually required.
- VALUE = the printed token from listing/fact_rows, copied character-for-
  character. Do not paraphrase, do not substitute the shopper's preferred
  wording when the catalog printed something else.

FIELD CHOICE RULES
- Follow schema_rule. When allowed_fields is non-empty the field MUST be taken
  from it. When it is empty, name the conventional attribute a shopper would
  use for that kind of ask — never the catalog drawer key.
- Inch / diagonal / "N-inch" asks → size or screen_size (whichever the schema
  or fact rows support). Do not invent resolution for an inch ask.
- HD / 720p / 1080p / FHD / 4K / UHD / pixel-class asks → resolution. Do not
  invent screen_size for a resolution-class ask.
- Apparel → size / material / color; footwear → size / material / width;
  bags and home → capacity / dimensions / finish; electronics → resolution /
  screen_size as above. Other categories: the ordinary English attribute.
- Shopper shorthand (large, cotton, navy, 720p, 27-inch, ...) is always a
  VALUE under the conventional attribute — never itself a field name.

RESOLUTION CLASS CONFLICT (wrong sku)
- If the shopper stated a resolution class (e.g. 1080p / FHD / 720p / HD) and
  the printed resolution on this sku is a different class (e.g. a 2560x1440
  or QHD figure when they asked 1080p/FHD), this sku is WRONG. Return
  {"claims":[]} — do not invent a matching claim and do not "force" the ask
  onto mismatched print.

BANS (these are NOT gate fields; emitting them blocks the real fill)
- Never emit: specification, spec, specs, hdr, feature, features, attribute,
  requirement, why, rationale, justification, evidence, reason, summary,
  description, display, os, category, title, facts, options.
- Never weld a container onto an attribute (display_resolution, size_material).
- Leave price and in_stock alone; they are already handled.
- Claim brand only when a brand fact row proves it, and keep it short.
- When listing/fact_rows do not prove the hard gate, reply {"claims":[]}.
- No more than two claims.
""".strip()

class GateClaimDesk:

    @staticmethod
    def polish(
        claims: Any, brief: GateBrief, well: GateWell, pid: str
    ) -> list[dict[str, str]]:
        if not isinstance(claims, list):
            claims = []
        allowed = {_key(x) for x in brief.claim_enum}
        observed = well.rows.get(pid, {})
        receipt = well.receipt(pid)
        kept: list[dict[str, str]] = []
        taken: set[str] = set()

        for entry in claims:
            if not isinstance(entry, dict):
                continue
            field = _key(entry.get("field"))
            value = "" if entry.get("value") is None else str(entry.get("value")).strip()
            if not field or not value:
                continue
            if field in _NO_FIELD or _welded_field(field):
                continue
            if allowed and field not in allowed:
                continue
            if field in taken or not _LEGAL_FIELD_RX.match(field):
                continue
            if field == "in_stock" and value.lower() in {"true", "1", "yes", "in_stock"}:
                value = "true"
            if field == "brand":
                value = GateClaimDesk._tight_brand(value, observed, receipt)
                if not value or not observed.get("brand"):
                    continue
            elif field not in _BASE_FIELDS and not _printed(value, receipt):
                swap = GateClaimDesk._printed_swap(value, observed, brief.ask, receipt)
                if not swap:
                    continue
                value = swap
            taken.add(field)
            kept.append({"field": field, "value": value})

        for need in ("price", "in_stock"):
            if need in taken or (allowed and need not in allowed):
                continue
            got = observed.get(need) or (
                well.last_price if need == "price" else well.last_stock
            )
            if got:
                slot = 0 if need == "price" else min(1, len(kept))
                kept.insert(slot, {"field": need, "value": got})
                taken.add(need)

        if "brand" not in taken and (not allowed or "brand" in allowed):
            brand = GateClaimDesk._tight_brand(observed.get("brand", ""), observed, receipt)
            if brand and observed.get("brand") and _printed(brand, brief.ask):
                kept.append({"field": "brand", "value": brand})
                taken.add("brand")

        if brief.needs_claims() and not GateClaimDesk._gate_claimed(kept):
            for row in GateClaimDesk._fill(brief, well, pid, taken):
                field = _key(row.get("field"))
                if field in taken or field in _NO_FIELD or _welded_field(field):
                    continue
                if allowed and field not in allowed:
                    continue
                if not _LEGAL_FIELD_RX.match(field):
                    continue
                value = str(row.get("value") or "").strip()
                if not value:
                    continue
                if not _printed(value, receipt):
                    swap = GateClaimDesk._printed_swap(value, observed, brief.ask, receipt)
                    if not swap:
                        continue
                    value = swap
                if field == "brand":
                    if not observed.get("brand"):
                        continue
                    value = GateClaimDesk._tight_brand(value, observed, receipt)
                    if not value:
                        continue
                taken.add(field)
                kept.append({"field": field, "value": value})
                if GateClaimDesk._gate_claimed(kept):
                    break

        if brief.needs_claims() and not GateClaimDesk._gate_claimed(kept):
            gate = GateClaimDesk._gate_from_facts(brief.ask, observed, receipt, taken, allowed)
            if gate:
                taken.add(gate["field"])
                kept.append(gate)

        if (
            brief.needs_claims()
            and "brand" not in taken
            and (not allowed or "brand" in allowed)
            and len(kept) < 4
            and observed.get("brand")
        ):
            brand = GateClaimDesk._tight_brand(observed.get("brand", ""), observed, receipt)
            if brand:
                kept.append({"field": "brand", "value": brand})
                taken.add("brand")

        return kept[:6]

    @staticmethod
    def _gate_claimed(claims: list[dict[str, str]]) -> bool:
        fields = {_key(c["field"]) for c in claims}
        rest = fields - _BASE_FIELDS - _NO_FIELD
        return bool(rest - {"brand"})

    @staticmethod
    def _gate_from_facts(
        goal: str,
        rows: dict[str, str],
        receipt: str,
        have: set[str],
        allowed: set[str],
    ) -> dict[str, str] | None:
        goal_l = (goal or "").lower()
        ranked: list[tuple[int, str, str]] = []
        for raw_key, raw_val in rows.items():
            key = _key(raw_key)
            value = str(raw_val or "").strip()
            if not key or not value or key in have:
                continue
            if key in _BASE_FIELDS or key == "brand" or key in _NO_FIELD:
                continue
            field = key
            if field in have or field in _NO_FIELD or _welded_field(field):
                continue
            if allowed and field not in allowed:
                continue
            if not _LEGAL_FIELD_RX.match(field):
                continue
            if not _printed(value, receipt):
                continue
            score = 0
            if field in goal_l or key in goal_l:
                score += 12
            if _printed(value, goal):
                score += 5
            if not score:
                continue
            ranked.append((score, field, value))
        if not ranked:
            return None
        ranked.sort(key=lambda row: (-row[0], row[1]))
        _, field, value = ranked[0]
        return {"field": field, "value": value}

    @staticmethod
    def _fill(
        brief: GateBrief, well: GateWell, pid: str, have: set[str]
    ) -> list[dict[str, str]]:
        receipt = well.receipt(pid)
        rows = well.rows.get(pid, {})

        if not receipt.strip(" |") and not rows:
            return []
        enum = [name for name in (brief.claim_enum or []) if _key(name) not in have]
        payload = {
            "goal": brief.ask[:1500],
            "allowed_fields": enum,
            "schema_rule": brief.claim_sentence,
            "have": sorted(have),
            "listing": receipt[:2200],
            "fact_rows": well.row_list(pid)[:40],
        }
        raw = _link.json_ask(
            engine=_HG_FAST,
            system=_TF7_FILL,
            user=json.dumps(payload, ensure_ascii=False),
        )
        parsed = _json_obj(raw)
        claims = parsed.get("claims") if isinstance(parsed, dict) else None

        if not isinstance(claims, list):
            return []
        return [row for row in claims[:2] if isinstance(row, dict)]

    @staticmethod
    def _printed_swap(
        value: str, observed: dict[str, str], ask: str, receipt: str
    ) -> str:
        if value and _printed(value, receipt):
            return value
        ranked: list[tuple[int, str]] = []
        for label, seen in observed.items():
            if not seen or label in _BASE_FIELDS or label == "brand":
                continue
            if not _printed(seen, receipt):
                continue
            score = 0
            if _printed(seen, ask):
                score += 5
            if value and (_printed(seen, value) or _printed(value, seen)):
                score += 4
            if label in _NO_FIELD or label in {"display", "size", "facts", "options"}:
                score += 1
            if score:
                ranked.append((score, seen))
        if ranked:
            ranked.sort(key=lambda pair: (-pair[0], -len(pair[1])))
            return ranked[0][1]
        return ""

    @staticmethod
    def _tight_brand(raw: str, observed: dict[str, str], receipt: str) -> str:
        structured = str(observed.get("brand") or "").strip()
        if not structured:
            return ""
        if not _printed(structured, receipt) or len(structured.split()) > 4:
            return ""
        text = str(raw or "").strip()
        if text and len(text.split()) <= 3 and len(text) <= 40 and _printed(text, structured):
            return text
        return structured

_ROLE = """
You are HiddenGirl on the AskGateEngine. Think of the run as a vault desk with a
disclosure latch and a final receipt tape — not as free-form browsing.

SENSES (tools only; nothing else is real)
  search · filter · view · compare · inspect_stock · add_to_cart ·
  remove_from_cart · message · place_test_order
The episode scores only after an observation returns done=true from
place_test_order. Searching forever without ordering is a zero.

ASKGATE DISCIPLINE
- The engine may discard or withhold tool calls. Treat refusals as law, not as
  suggestions to work around.
- Shopper questions: answer with message on the same or next turn BEFORE any
  cart or order. An open question blocks progress until you reply.
- SpecVault disclosure (when armed): while the gate is shut and you have not
  asked, search/filter attempts are thrown away before they reach the catalog.
  Cart stays locked until disclosure has been requested and applied where the
  family requires it. Message-before-search when a figure is withheld.
- First add_to_cart is graded. Do not park a placeholder sku “just to have
  something”; commit only when the vault rules for this family are met.
- Non-TF7 families: omit justification.claims entirely from place_test_order.
- Always leave ≥2 steps for inspect_stock + place_test_order. Late budget: stop
  open-ended search, close any market event, finish the order.

MARKET EVENT PROTOCOL (every family)
1. inspect_stock the EVENT-TARGET sku first — that is what sets event_closed.
   Inspecting a substitute does not close the event. Do this even when you plan
   to order a different sku afterward.
2. If the target is invalid, remove_from_cart, then recover per the family book.
3. After any price move, re-check the spend lid (within_budget can fail after a
   market event) before you order.
4. Order only after the event is closed. An unfinished event (event_closed=false)
   pays 0 even if the cart looks gold.

RECEIPT HABIT
Titles are marketing. Fact rows and inspect_stock observations are what the
desk will accept. When claims are required (TF7), values must be printed tokens
on the receipt, not paraphrases. Never invent meta field names such as
specification or hdr — those are not gates.
""".strip()

_BOOK_TF1 = """
FAMILY LEDGER — BUNDLED INTENT (vault sort: firm vs wish)

The shopper stacked unequal statements in one ask. Before any search, sort them
into two piles that never trade places:
  FIRM / DEMANDS   disqualifiers — product class, spend lid, stated spec,
                   purchasability / in-stock, any hard numeric gate
  SOFT / WISHES    extras marked with ideally / would love / bonus / prefer —
                   often a maker or look, never a licence to break firm walls

DUAL DISCOVERY (both passes matter)
- Pass A: include the wish token with the firm nouns so brand-ish candidates
  can surface. When BOTH a maker and a hard gate are named, search them
  together (brand + gate tokens in one query) rather than dropping one.
- Pass B: firm nouns alone, so a wall-clearing backup exists when the wish
  shelf is empty or non-compliant.
Do not collapse into a single lazy query. Coverage on both piles is how the
desk avoids “wish-only dead ends” and “firm-only blindness.”

FIRST-CART BRAND RULE (graded commitment)
- Cart the preferred-brand / wish-honouring item ONLY when that exact sku clears
  every firm gate on observed FACT ROWS (not titles), and a brand fact proves
  the maker when the goal named one.
- If no wish-honouring sku is fully firm-compliant, cart the best firm backup
  and say so with message — never soft-over-firm. Ordering a non-matching /
  non-fact-backed sku when a brand was named is a failed vault posting.
- Never demote a firm wall into a wish because the wish shelf looked prettier.
- Do not cart until the hard-gate figure appears in a fact row for that sku.

MESSAGE INTERVENTIONS
If the shopper asks what you found, or challenges the trade-off, answer with
message before touching cart/order. AskGate withholds cart while a question is
open; ignore that and the turn is wasted. When a figure is withheld until you
ask, message first — no search until disclosure.

AFTER A MARKET EVENT
Close the event on the event-target sku (inspect_stock) even if you will order
a different sku. Then remove if invalid, RE-CHECK budget under the new price,
and RE-APPLY the brand rule under the new walls: dual discovery again if needed,
wish-honouring only if still fully firm-compliant, else firm backup. Do not keep
a pre-event wish pick that no longer clears firm gates. within_budget can fail
after a price move — re-read price before place_test_order.

WORK RHYTHM
search/filter → view fact rows (not titles) → add_to_cart compliant sku →
inspect_stock → place_test_order. Reserve the closing pair. Omit justification.
""".strip()

_BOOK_TF2 = """
FAMILY LEDGER — RECALL (surface the printed item, not a neighbour)

This is a coverage problem. The shopper is pointing at something already in the
catalog, often by distinctive wording. One plausible hit is a thin answer; the
desk wants the real listing surfaced and verified across multiple queries.

MULTI-QUERY COVERAGE
- Open with the shopper’s distinctive tokens and any quoted phrase verbatim.
  Reuse those tokens across passes — do not scrub the distinctive words away.
- Run more than one query shape when the first shelf is thin: keep the rare
  nouns, vary one secondary knob. Coverage beats a single lucky neighbour.
- Judge progress by DISTINCT product_ids viewed, not by how many times the same
  listing reappears. Aim to open several different identities before you settle.

ONE NEAR-MISS REFINE, THEN STOP REWRITING
If the first pass is thin, change exactly ONE knob: drop an adjective, drop a
proper noun, plural↔singular, strip punctuation from a phrase. Then view many
distinct product_ids. Do not spend the budget reinventing the whole query every
turn — that returns the same neighbourhood.

PHRASE MUST LIVE IN FACT ROWS
AskGate will withhold cart when the quoted text exists only in a marketing
title. That is the trap: headlines borrow words the item does not own. Open the
candidate, confirm the phrase (or its parts) in fact rows, then cart. Title-only
matches are wrong answers even when they “feel” right. Phrase the match from
fact rows in your reasoning, not from titles.

CLOSING SEQUENCE
Once you have a row-backed match among a covered shelf: add_to_cart →
inspect_stock → place_test_order. Answer shopper questions with message the turn
they land. Close market events on the event-target before finishing (even if
ordering a different sku). Omit justification. Late budget: stop refining and
finish. event_closed=false pays 0.
""".strip()

_BOOK_TF3 = """
FAMILY LEDGER — LOCKED SPEC (disclosure before sense)

The shopper is deliberately withholding a figure and said you will hear it if
you ask. Until disclosure, you are shopping blind. AskGate’s SpecVault makes
that mechanical.

TURN-1 RULE (gate shut)
- First action: message ONLY — one plain ask naming the withheld detail.
- No search, no filter, no “sneaky” browse. While the gate is shut and you have
  not asked, search/filter calls are DISCARDED before they reach the catalog and
  you are re-prompted. Workarounds waste the step. Message-before-search is law.

AFTER THE VAULT OPENS
- Take the revealed literal exactly as given. Search/filter with that token.
  When a brand and a gate figure are both named, keep them together in the query.
- view / compare until the exact sku is printed in an observation — never
  invent or pattern-assemble a sku string.
- Every firm gate must hold on THAT sku’s fact rows: class, disclosed figure,
  spend lid, stock. A sibling colourway or near code is a different product.
- Do not cart until the disclosed figure appears in a FACT ROW for that sku.

SELLOUT / INVALID → NEW PRODUCT_ID
If the disclosed-compliant sku sells out or fails stock, move to a NEW
product_id that still meets ALL firm gates under the revealed literal. Do not
fall into the sibling trap (same family, wrong variant). Prove each gate again
on the replacement’s observations.

ORDER
inspect_stock the carted sku, then place_test_order with no justification
object. If a market event fires, inspect the event-target first (even when
ordering a different sku), recover with another fully gate-meeting product_id,
re-check budget after any price move, then order. Leave ≥2 steps for the close.
""".strip()

_BOOK_TF4 = """
FAMILY LEDGER — PREFERENCE (hard walls, then stated soft axes)

Several items will clear the hard walls; the shopper told you how to tilt among
them. Reward is for honouring that tilt with printed evidence — not for cheapest.
AskGate still enforces message-before-cart when a question is open; answer first.

HARD WALLS FIRST
Clear budget, stock, class, and every stated hard requirement. Soft axes never
excuse a broken wall. Do not treat “would love” language as a hard gate unless
the brief made it binding.

≥2 HARD-OK READS BEFORE CART
Open at least two wall-clearing candidates with view (or compare). One read is
not a comparison. Score them on the STATED soft axes using printed fact values
only — never title adjectives, never axes you invented. First cart is graded;
no placeholder sku while you “keep looking.” Compare ≥2 hard-OK on the stated
soft axes before committing.

ONE OPTIONAL CLARIFY
If two stated axes genuinely conflict, you may send exactly one clarifying
message. AskGate budgets that: a second stall clarification is withheld. If you
already have enough evidence, do not ask — commit.

CHEAPEST ≠ WINNER
Price decides only when named axes are level (or the shopper said price leads).
A more expensive hard-OK item that wins the stated tilt is the correct cart.
Never invent a “value” or “quality” axis the shopper never wrote.

CLOSING
Name the winner and the printed rows that decided it when useful, then
add_to_cart → inspect_stock → place_test_order. Omit justification. Close any
market event on the event-target before ordering (replacement inspect does not
close — inspect the event-target even if ordering a different sku). After a
price move, re-check the spend lid. Late budget: stop browsing, finish the
order pair.
""".strip()

_BOOK_TF5 = """
FAMILY LEDGER — CLOSED RANK (hand of named cards)

The shopper dealt a closed set, usually product_id::sku, plus lexicographic
priorities. You are adjudicating that set — not opening the wider catalog.
AskGate withholds cart while a named card is still unopened.

BOUNDS
- Only named product_id::sku pairs may be ranked or bought. Attractive strangers
  from search are out of bounds.
- view or compare ALL named candidates before any add_to_cart. Unopened cards
  cannot be placed. First cart is graded — no provisional leader before the
  full open.

LEX-RANK
- First stated axis decides; later axes break ties only; price last unless stated
  otherwise. Write the ordering in your reply so the pick is traceable.
- product_id and sku are separate arguments. For “12345::abc”, product_id=12345
  and sku=abc — never paste the joined string into product_id.

AFTER A MARKET EVENT — RE-RANK FROM SCRATCH
Do not keep the pre-event #1. inspect_stock the event-target to close the event
(even if you will order a different named card), refresh observations on the
surviving named set, re-check budgets after any price move, and re-sort from
scratch under the same axes. Whoever leads at order time wins. Pre-event
rankings are stale.

ORDER SHAPE
place_test_order carries product_id + sku only — never claims / justification.
Strip any justification object entirely. Cart the leader if needed,
inspect_stock, order. Answer open questions with message first. Reserve the
closing pair. Late budget: stop re-opening cards, finish.
""".strip()

_BOOK_TF6 = """
FAMILY LEDGER — RECOVERY (prove gone, then climb one rung)

The wanted item is unavailable, or becomes so mid-run. Empty-handed is a zero.
Controlled concession — never silent class/budget collapse. Finish the run once
a valid substitute is proven; unfinished recoveries score nothing. Do not stall.

PROVE ORIGINAL UNAVAILABLE BEFORE SUBSTITUTING
AskGate withholds substitution carts until an observation shows the original is
gone. inspect_stock or view the named item and let the environment say it is
unavailable. Suspicion-based swaps fail even when stock really was empty.

CONCESSION LADDER (one rung at a time; prove each dead end)
  1) Cosmetic: shade / finish / trim — announce the trade with message if the
     shopper is in conversation.
  2) Model or maker next.
  NEVER relax: product class, spend lid, stock requirement, or stated hard spec.
Prove each dead end with observations before climbing. Do not jump from
cosmetic straight into an unrelated class.

BEFORE FIRST CART
inspect_stock the intended commitment candidate before the first add_to_cart.

AFTER A MARKET EVENT — FINISH THE RUN
1. inspect_stock the EVENT-TARGET sku immediately (closes event; replacement
   inspect does not — do this even when ordering a different sku).
2. remove_from_cart if invalid.
3. Climb the ladder to a verified replacement that still holds class/budget/
   stock/spec. Re-check budget after any price move.
4. Cart → inspect_stock → place_test_order promptly.
Do not burn leftover steps on extra refine theatre after a valid recovery is
carted. event_closed=false pays 0 even if the cart looks gold. Omit
justification. An unplaced recovery is a zero.
""".strip()

_BOOK_TF7 = """
FAMILY LEDGER — RECEIPT JUSTIFICATION (claims must match the tape)

place_test_order MUST attach justification.claims: a list of {field, value}
objects. Every pair is graded against catalog print. Pretty prose with unprinted
values scores worse than an empty list. price/in_stock alone do NOT finish the
receipt when the shopper stated a hard gate — you still owe that attribute.

CLAIM SHAPE (≤4; include all that facts support)
  price      verbatim from newest view or inspect_stock
  in_stock   normally "true", from inspect_stock
  gate attr  FIELD = conventional shopper attribute name for the ask
             (apparel: size/material/color; footwear: size/material/width;
             bags/home: capacity/dimensions/finish; electronics:
             resolution for HD/720p/1080p/FHD asks, size/screen_size for inch
             asks; among others) — NEVER catalog drawer keys, NEVER shopper
             shorthand used as the field name, NEVER meta labels
             (specification / spec / hdr / feature / attribute / requirement /
             why / rationale / justification / evidence)
             VALUE = printed fact-row token character-for-character
             (large, cotton, navy, 720p, 27-inch are values under the right
             attribute — not field names)
  soft       at most one attested reason; short brand only when the shopper
             named a maker AND a brand fact row proves it. Titles are not brands.

SEARCH & COMMITMENT
- When brand and a hard gate are both named, search brand+gate tokens together.
- Do not cart until the gate figure appears in a FACT ROW for that sku.
  Title-only figures are decoys — reject that sku.
- If a printed resolution class conflicts with the shopper’s stated class
  (e.g. 2560x1440 when they asked 1080p/FHD), reject the sku; do not claim it.
- When a brand is named in the goal, only order a fact-backed matching maker
  unless no firm-compliant brand match exists (then firm backup + message).

BANS
- category, title, summary, evidence, reason, rationale, description, os,
  display, specification, spec, hdr, feature(s), attribute, requirement, why,
  justification
- fused drawer+attribute names (e.g. display_resolution)
- unit conversions, invented features, paragraph-length claim values

EVENTS & CLOSE
- inspect_stock a candidate BEFORE the first add_to_cart (intended commitment).
- After a market event: ALWAYS inspect_stock the EVENT-TARGET sku first to
  close it — even if you will order a different sku. Replacement inspect does
  not close. Drop if invalid, re-check budget after any price move, rebuild
  claims from the fresh observation, then order.
- Once a row-backed pick is inspected, place_test_order. Waiting for permission
  or hunting synonyms into a zero budget is how receipts become worthless.
""".strip()

_CORPUS: dict[str, str] = {
    TF1: _BOOK_TF1,
    TF2: _BOOK_TF2,
    TF3: _BOOK_TF3,
    TF4: _BOOK_TF4,
    TF5: _BOOK_TF5,
    TF6: _BOOK_TF6,
    TF7: _BOOK_TF7,
}

_POKE = (
    "No tool call landed. Choose a tool and act this turn — AskGate only senses tools.",
    "Still idle. Close any open question with message, then search/view → cart → "
    "inspect_stock (event-target first if an event fired) → place_test_order.",
)

_RECEIPT_TIPS = (
    "Receipt tip: FIELD = conventional attribute (size/material/color clothing; "
    "size/material/width footwear; capacity/dimensions/finish bags-home; "
    "resolution for HD/1080p/FHD, size/screen_size for inch asks). VALUE = "
    "printed fact-row token. Shorthand like 720p or 27-inch is a value, never "
    "a field name. Never emit specification, spec, hdr, feature, or why.",
    "Receipt tip: price/in_stock alone do not finish a gate ask — still claim "
    "the hard attribute. Copy price and in_stock from the newest view or "
    "inspect_stock. Never weld drawer keys onto attributes; never claim os, "
    "display, specification, or hdr as fields.",
    "Receipt tip: at most one soft claim; short brand only when a brand fact "
    "row attests it. Ceiling four claims. Banned: category, title, os, display, "
    "specification, hdr, feature, attribute, requirement, why, rationale, "
    "justification, title-as-brand.",
    "Receipt tip: market event → ALWAYS inspect_stock the EVENT-TARGET sku "
    "first to close it, even if ordering a different sku (replacement inspect "
    "does not). Re-check budget, rebuild claims from the fresh observation, "
    "then place_test_order.",
    "Receipt tip: title-only measurements are decoys and grade false. Keep "
    "viewing until a fact row prints the figure, then cart that sku. If printed "
    "resolution class conflicts with the shopper’s class, reject the sku.",
    "Receipt tip: search brand+gate tokens together when both are named. No "
    "unit conversion, no invented features — paste tokens exactly. Past "
    "mid-budget: stop open search, close events, inspect_stock, order.",
    "Vault tip: unanswered shopper questions block cart/order — message first. "
    "First cart is graded; no placeholders. Non-TF7: omit justification "
    "entirely. Do not cart until the gate figure is in a FACT ROW.",
)

def _brief_block(brief: GateBrief, policy: dict[str, Any], vault: SpecVault) -> str:
    labels = _tool_labels(policy.get("tools") or [])
    return (
        "RUN STATE\n"
        f"- family: {brief.family}\n"
        f"- disclosure gate: {'open' if vault.gate_open else 'shut'} "
        f"({vault.digest()})\n"
        f"- tools: {', '.join(labels) or '(see schemas)'}\n"
        f"- step limit: {policy.get('max_steps')}\n"
        f"- actions per turn: {policy.get('max_calls_per_turn', 1)}"
    )

def _system(brief: GateBrief, policy: dict[str, Any], vault: SpecVault) -> str:
    parts = [_ROLE, _CORPUS.get(brief.family, _BOOK_TF1)]
    if brief.needs_claims() and brief.family != TF7:
        parts.append(_BOOK_TF7)
    dossier = brief.dossier()
    if dossier:
        parts.append(dossier.strip())
    parts.append(_brief_block(brief, policy, vault))
    return "\n\n".join(parts)

def _receipt_tip(turn: int, well: GateWell, brief: GateBrief) -> str:
    body = _RECEIPT_TIPS[(turn - 1) % len(_RECEIPT_TIPS)]
    extras: list[str] = []
    if brief.claim_enum:
        extras.append(f"Fields the schema allows: {', '.join(brief.claim_enum)}.")
    if brief.claim_sentence:
        extras.append(f"Schema rule: {brief.claim_sentence[:180]}")
    pid = well.pin or well.last_pid
    if pid:
        observed = well.rows.get(pid, {})
        if observed:
            sample = ", ".join(f"{k}={v}" for k, v in list(observed.items())[:8])
            extras.append(f"Rows observed for {pid}: {sample}")
        if well.last_price or well.last_stock:
            extras.append(
                f"Freshest price={well.last_price or '?'} "
                f"in_stock={well.last_stock or '?'}."
            )
    return body + (("\n" + "\n".join(extras)) if extras else "")

class GateState:

    __slots__ = (
        "turn",
        "limit",
        "last_labels",
        "history",
        "closed",
        "carted",
        "cart_pid",
        "cart_sku",
        "question_open",
        "event_open",
        "event_pid",
        "event_sku",
        "clarifies",
        "lookups",
        "probes",
        "nagged",
        "blocks",
        "withheld",
        "unavailable_proof",
        "conceded",
        "faked_sku",
    )

    def __init__(self, limit: int) -> None:
        self.turn = 0
        self.limit = limit
        self.last_labels: list[str] = []
        self.history: set[str] = set()
        self.closed = False
        self.carted = False
        self.cart_pid = ""
        self.cart_sku = ""
        self.question_open = False
        self.event_open = False
        self.event_pid = ""
        self.event_sku = ""
        self.clarifies = 0
        self.lookups = 0
        self.probes = 0
        self.nagged: set[str] = set()
        self.blocks: set[str] = set()
        self.withheld = 0
        self.unavailable_proof = False
        self.conceded = 0
        self.faked_sku = ""

    @property
    def left(self) -> int:
        return max(0, self.limit - self.turn)

    def ever(self, label: str) -> bool:
        return label in self.history

    def record(self, labels: list[str], calls: list[dict[str, Any]]) -> None:
        self.last_labels = labels
        self.history.update(labels)
        for label in labels:
            if label in _LOOKUP_TOOLS:
                self.lookups += 1
            if label in _PROBE_TOOLS:
                self.probes += 1
            if label in _TALK_TOOLS:
                self.clarifies += 1
                self.question_open = False
        if "add_to_cart" in labels:
            self.carted = True
            for call in calls:
                if _named(call) == "add_to_cart":
                    args = _arguments(call)
                    self.cart_pid = str(args.get("product_id") or self.cart_pid)
                    self.cart_sku = str(args.get("sku") or self.cart_sku)
        if "remove_from_cart" in labels:
            self.carted = False
        if "place_test_order" in labels:
            self.closed = True
        if "inspect_stock" in labels:
            for call in calls:
                if _named(call) != "inspect_stock":
                    continue
                pid = str(_arguments(call).get("product_id") or "")
                if self.event_pid and pid == self.event_pid:
                    self.event_open = False
                    self.event_pid = ""
                    self.event_sku = ""
                    break

    def first_time(self, tag: str) -> bool:
        if tag in self.nagged:
            return False
        self.nagged.add(tag)
        return True

    def may_block(self, tag: str) -> bool:
        if tag in self.blocks:
            return False
        self.blocks.add(tag)
        return True

class GateBridge:

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
                return _HG_HEAVY
            return override
        return _HG_HEAVY if family in _HEAVY_FAMILIES else _HG_FAST

    def ask(
        self,
        *,
        engine: str,
        transcript: list[dict[str, Any]],
        schemas: list[Any],
        budget: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        ladder = [engine, *[m for m in _HG_SPARE if m != engine]]
        trouble: Exception | None = None
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
                trouble = RuntimeError(model)
                continue
            try:
                message = payload["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                trouble = exc
                continue
            return message.get("content") or "", list(message.get("tool_calls") or [])[:budget]
        raise RuntimeError(f"inference failed ({trouble})")

    def json_ask(self, *, engine: str, system: str, user: str) -> str:
        ladder = [engine, *[m for m in _HG_SPARE if m != engine]]
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

_link = GateBridge(ProxyClient(timeout=120, max_retries=2))

def _reads_as_question(text: str) -> bool:
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
            "are you",
            "wait",
            "hold on",
            "before you",
        )
    )

def _event_in(obs: Any) -> bool:
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

def _says_unavailable(obs: Any) -> bool:
    if not isinstance(obs, dict):
        return False
    blob = json.dumps(obs, ensure_ascii=False, default=str).lower()
    if '"in_stock": false' in blob or '"in_stock":false' in blob:
        return True
    return bool(re.search(r"sold[_\s]?out|unavailable|out[_\s]of[_\s]stock|discontinued", blob))

def _empty_result(obs: Any) -> bool:
    if not isinstance(obs, dict):
        return True
    for key in ("results", "items", "products", "matches"):
        value = obs.get(key)
        if isinstance(value, list) and value:
            return False
    nested = obs.get("observation")
    if isinstance(nested, dict):
        return _empty_result(nested)
    return not obs.get("product_id")

def _run_finished(result: dict[str, Any]) -> bool:
    for row in result.get("calls") or []:
        if isinstance(row, dict):
            obs = row.get("observation")
            if isinstance(obs, dict) and obs.get("done"):
                return True
    return False

def _named(call: dict[str, Any]) -> str:
    spec = call.get("function") if isinstance(call, dict) else None
    return str(spec["name"]) if isinstance(spec, dict) and spec.get("name") else ""

def _reference(call: dict[str, Any], fallback: str) -> str:
    return str(call.get("id") or fallback)

def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    spec = call.get("function") if isinstance(call, dict) else None
    raw = spec.get("arguments", "{}") if isinstance(spec, dict) else "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}

def _condense(obs: Any, cap: int = 6000) -> str:
    try:
        text = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obs)
    return text if len(text) <= cap else text[: cap - 18] + "…<clipped>"

_MAX_WITHHOLD_PER_TURN = 2

class GateEpisode:

    def __init__(self, problem_data: dict[str, Any]) -> None:
        env = problem_data["environment"]
        self.binding = env["binding"]
        self.policy = env["policy_view"]
        self.schemas = self.policy.get("tools") or []
        self.brief_text = str(self.policy.get("query") or problem_data.get("query") or "")
        self.tag = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        self.per_turn = max(1, int(self.policy.get("max_calls_per_turn", 1)))
        self.brief = GateBrief(self.brief_text, self.schemas)
        self.state = GateState(max(1, int(self.policy.get("max_steps", 20))))
        self.vault = SpecVault(armed=self.brief.hidden_spec and self.brief.family == TF3)
        self.well = GateWell()
        self.engine = GateBridge.engine(self.brief.family)

    def run(self) -> list[dict[str, Any]]:
        transcript: list[dict[str, Any]] = [
            {"role": "system", "content": _system(self.brief, self.policy, self.vault)},
            {"role": "user", "content": self.brief_text},
        ]
        if self.vault.blocked():
            transcript.append(
                {
                    "role": "user",
                    "content": (
                        "Gate is shut: search and filter calls will be discarded "
                        "until you have asked the shopper for the withheld "
                        "requirement. Send that message first."
                    ),
                }
            )
        if self.brief.family == TF5 and self.brief.pairs:
            transcript.append(
                {
                    "role": "user",
                    "content": (
                        "Closed candidate set: "
                        + ", ".join(self.brief.pairs)
                        + ". Open each before ranking; product_id and sku are "
                        "separate arguments."
                    ),
                }
            )
        trail: list[dict[str, Any]] = [
            {
                "role": "meta",
                "hiddengirl": {
                    "version": 1,
                    "family": self.brief.family,
                    "engine": self.engine,
                    "pipeline": "AskGateEngine",
                    "provider": "openrouter",
                    "gate_armed": not self.vault.gate_open,
                    "tally": self.brief.tally,
                },
            }
        ]
        while self.state.turn < self.state.limit:
            self.state.turn += 1
            nudge = self._coach()
            if nudge:
                transcript.append({"role": "user", "content": nudge})
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
                    "hiddengirl_turn": {
                        "family": self.brief.family,
                        "turn": self.state.turn,
                        "gate_open": self.vault.gate_open,
                        "asked": self.vault.asked,
                        "revealed": list(self.vault.revealed_tokens),
                        "withheld_calls": self.state.withheld,
                    },
                }
            )
            self.state.record([_named(c) or "?" for c in calls], calls)
            self._ingest(result, transcript, calls)
            if _run_finished(result):
                break
        return trail

    def _turn(
        self, transcript: list[dict[str, Any]], trail: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]] | None:
        for attempt in range(_MAX_WITHHOLD_PER_TURN + 1):
            spoken, calls = self._speak(transcript, trail)
            calls = self._block_order_during_event(calls)
            kept, correction = self._intercept(calls)
            if kept:
                transcript.append(
                    {"role": "assistant", "content": spoken, "tool_calls": kept}
                )
                return spoken, kept, self._dispatch(kept, self.state.turn)

            self.state.withheld += len(calls)
            trail.append(
                {
                    "role": "assistant",
                    "content": spoken,
                    "withheld_tool_calls": calls,
                    "hiddengirl_withhold": {
                        "turn": self.state.turn,
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

    def _intercept(
        self, calls: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str]:
        if not calls:
            return [], "Emit a tool call."
        labels = [_named(c) for c in calls]

        if self.state.left <= 4 or self.state.withheld >= 3:
            return calls, ""

        if self.vault.blocked() and self.brief.family == TF3:
            if not any(label in _TALK_TOOLS for label in labels):
                if any(label in _LOOKUP_TOOLS or label in _SPEND_TOOLS for label in labels):
                    return [], (
                        "Withheld: "
                        + ", ".join(sorted({name for name in labels if name}))
                        + " never reached the environment because the disclosure "
                        "gate is shut. The shopper is holding a requirement back "
                        "until you ask for it, and a search built on a guess "
                        "searches the wrong shelf. Send exactly one message this "
                        "turn that asks for the withheld requirement in plain "
                        "words, then continue."
                    )

        if self.brief.family == TF1 and self.state.question_open:
            if not any(label in _TALK_TOOLS for label in labels):
                spend = [name for name in labels if name in _SPEND_TOOLS]
                if spend:
                    return [], (
                        "Withheld: " + ", ".join(sorted(set(spend))) + " was dropped "
                        "because the shopper's question is still unanswered. "
                        "Intervene first — reply with message in this turn, then "
                        "resume the purchase."
                    )

        if self.brief.family == TF2 and self.brief.quote:
            for call, label in zip(calls, labels):
                if label != "add_to_cart":
                    continue
                pid = str(_arguments(call).get("product_id") or "")
                if (
                    pid
                    and self.well.headline_only(pid, self.brief.quote)
                    and self.state.may_block("title:" + pid)
                ):
                    return [], (
                        f"Withheld: add_to_cart on {pid} was dropped. The quoted "
                        f"text \"{self.brief.quote}\" appears in that product's "
                        "title but in none of its fact rows, which is exactly the "
                        "title-only trap. view the variant rows for proof, or move "
                        "to another candidate."
                    )

        if self.brief.family == TF4 and self.state.clarifies >= 1:
            talk = [name for name in labels if name in _TALK_TOOLS]
            if talk and not self.state.question_open:
                keep = [
                    call
                    for call, name in zip(calls, labels)
                    if name not in _TALK_TOOLS
                ]
                correction = (
                    "Withheld: a second clarification message was dropped — you "
                    "have already used your one question. Decide from the observed "
                    "fact rows on the stated axis and act."
                )
                if keep:
                    return keep, correction
                return [], correction

        if self.brief.family == TF5 and self.brief.pairs:
            unopened = [
                ref.split("::", 1)[0]
                for ref in self.brief.pairs
                if ref.split("::", 1)[0] not in self.well.probed
            ]
            if (
                unopened
                and "add_to_cart" in labels
                and self.state.may_block("set:" + ",".join(unopened))
            ):
                return [], (
                    "Withheld: add_to_cart was dropped because "
                    + ", ".join(unopened[:4])
                    + " from the named set has not been opened yet. A candidate "
                    "you never observed cannot be ranked. view or compare the "
                    "remaining candidates, then cart the leader."
                )

        if self.brief.family == TF6 and not self.state.unavailable_proof:
            for call, label in zip(calls, labels):
                if label != "add_to_cart":
                    continue
                pid = str(_arguments(call).get("product_id") or "")
                if (
                    pid
                    and pid not in self.well.probed
                    and self.state.may_block("evidence:" + pid)
                ):
                    return [], (
                        "Withheld: add_to_cart was dropped because nothing yet "
                        "proves the original is unavailable, and "
                        f"{pid} has not been verified either. Evidence order "
                        "matters in this family: inspect_stock or view the item "
                        "the shopper named, let the observation say it is gone, "
                        "then verify the replacement and cart it."
                    )
        return calls, ""

    def _coach(self) -> str | None:
        state = self.state
        if state.question_open and self.brief.family != TF1:
            return (
                "The shopper is waiting on an answer. Reply with message this "
                "turn before any cart or order call."
            )
        if state.event_open and not state.closed:
            target = state.event_pid or state.cart_pid or "the carted sku"
            return (
                f"A market event is open. inspect_stock the event-target ({target}) "
                "FIRST this turn to close it — inspecting a replacement does not count. "
                "Then remove_from_cart if invalid, find a wall-clearing replacement if "
                "needed, inspect that sku, "
                + ("rebuild the claims from fact rows, " if self.brief.needs_claims() else "")
                + "then place_test_order — do not end the run on remove."
            )
        if state.turn <= 2 and not state.history:
            return self._first_move()

        if (
            "remove_from_cart" in state.last_labels
            and not state.carted
            and not state.closed
            and not state.event_open
        ):
            return (
                "Cart cleared after the market move. Cart a wall-clearing replacement "
                "NOW (or re-cart a survivor), inspect_stock it, then place_test_order — "
                "do not burn the remaining steps without an order."
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
        return branch() or self._last_steps()

    def _first_move(self) -> str:
        if self.vault.blocked():
            return (
                "First move: one message asking for the requirement the shopper "
                "is holding back. Lookup calls are withheld until then."
            )
        if self.brief.family == TF5 and self.brief.pairs:
            return (
                "First move: open the handed candidates ("
                + ", ".join(self.brief.pairs[:4])
                + ") instead of searching the catalog."
            )
        if self.brief.family == TF6:
            return (
                "First move: look for the item exactly as the shopper named it. "
                "A substitution is only allowed once an observation proves that "
                "item is gone."
            )
        return "Open with search or filter using the shopper's own words, then view the hits."

    def _last_steps(self) -> str | None:
        state = self.state
        if state.carted and state.left <= 6 and not state.closed and state.first_time("close"):
            return (
                f"{state.left} step(s) remain. inspect_stock the carted sku if needed, then "
                "place_test_order — an unclosed run scores nothing."
            )
        if (
            not state.carted
            and state.left <= 8
            and state.ever("view")
            and state.first_time("late_cart")
        ):
            return (
                "Step budget is low. If a viewed candidate already clears the walls, "
                "cart it NOW and keep two steps for inspect_stock and the order."
            )
        if state.left <= 1 and not state.closed and state.first_time("last_step"):
            return "Last step. place_test_order on the best verified candidate immediately."
        return None

    def _block_order_during_event(
        self, calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not self.state.event_open or self.state.closed:
            return calls
        event_pid = self.state.event_pid
        event_sku = self.state.event_sku
        if event_pid:
            probe: dict[str, Any] = {"product_id": event_pid}
            if event_sku:
                probe["sku"] = event_sku
            if not calls:
                return calls

            messages: list[dict[str, Any]] = []
            rest: list[dict[str, Any]] = []
            for call in calls:
                name = _named(call)
                if name == "message":
                    messages.append(call)
                    continue
                if name in {"search", "filter"}:
                    continue
                if (
                    name == "inspect_stock"
                    and str(_arguments(call).get("product_id") or "") == event_pid
                ):
                    continue
                rest.append(call)
            inspect_call: dict[str, Any] = {
                "id": f"{self.tag}-event-inspect",
                "type": "function",
                "function": {
                    "name": "inspect_stock",
                    "arguments": json.dumps(probe),
                },
            }
            return messages + [inspect_call] + rest
        out: list[dict[str, Any]] = []
        for call in calls:
            name = _named(call)
            if name != "place_test_order":
                out.append(call)
                continue
            args = _arguments(call)
            pid = str(
                args.get("product_id")
                or self.state.cart_pid
                or self.well.pin
                or self.well.last_pid
                or ""
            )
            sku = str(args.get("sku") or self.state.cart_sku or "")
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

    def _coach_tf1(self) -> str | None:
        state, well = self.state, self.well
        if state.closed:
            return None
        if state.question_open:
            return (
                "Intervention first: the shopper asked something and the engine is "
                "holding your cart and order calls. Answer with message now, in "
                "your own words, then go back to shopping."
            )
        if not state.carted and state.lookups <= 1 and state.first_time("tf1_bundle"):
            wish = ", ".join(self.brief.maker_words[:3]) or "the wish wording"
            return (
                f"Sort the bundle: demands on one side, wishes ({wish}) on the "
                "other. Search with a wish term included; if nothing comes back, "
                "search the demands alone. Never trade a demand for a wish."
            )
        if not state.carted and state.probes and state.first_time("tf1_rows"):
            lid = self.brief.lid or "the stated lid"
            return (
                "Check the fact rows of your candidate against each demand "
                f"(class, spec, spend {lid}, availability) before carting. If a "
                "demand fails, the wish is irrelevant."
            )
        if state.carted and not state.ever("inspect_stock") and state.first_time("tf1_stock"):
            return "Confirm the carted sku with inspect_stock, then place_test_order."
        return None

    def _coach_tf2(self) -> str | None:
        state, well = self.state, self.well
        if state.closed:
            return None
        distinct = len(well.seen)
        phrase = self.brief.quote
        if not state.carted and distinct < 2 and state.ever("search"):
            if state.lookups <= 1 and state.first_time("tf2_one_knob"):
                target = phrase or "the shopper's noun"
                return (
                    f"{distinct} distinct product(s) so far. Turn one knob: keep "
                    f"{target} and drop a single adjective or the proper noun. One "
                    "adjustment, then verify."
                )
            if state.first_time("tf2_verify"):
                return (
                    "Recall is still thin. Open what you have with view and read "
                    "the fact rows — provenance in the rows is what lets you cart."
                )
        pid = well.pin or well.last_pid
        if phrase and pid and not state.carted:
            if well.headline_only(pid, phrase) and state.first_time("tf2_title"):
                return (
                    f"\"{phrase}\" is title-only on {pid}. The engine will withhold "
                    "a cart there. Find the phrase among the fact rows of a "
                    "variant, or try the next candidate."
                )
        if not state.carted and distinct >= 2 and well.probed and state.first_time("tf2_go"):
            return (
                f"{distinct} products seen, {len(well.probed)} verified. Coverage is "
                "enough: cart the candidate whose rows carry the shopper's wording, "
                "inspect_stock, then order."
            )
        return None

    def _coach_tf3(self) -> str | None:
        state, well, vault = self.state, self.well, self.vault
        if state.closed:
            return None
        if vault.blocked():
            return (
                "The disclosure gate is shut: send one message asking the shopper "
                "for the requirement they are holding back. Any search you emit "
                "this turn is discarded before it reaches the environment."
            )
        if vault.asked and not vault.revealed_tokens and state.first_time("tf3_wait"):
            return (
                "You asked — now use whatever the shopper says literally. Do not "
                "start guessing figures while waiting."
            )
        if vault.revealed_tokens and not state.carted and state.first_time("tf3_apply"):
            return (
                "Disclosed: " + ", ".join(vault.revealed_tokens[:6]) + ". Search "
                "with that wording, then open the candidate and confirm the figure "
                "in the variant's own option or fact rows — not in the title."
            )
        if state.faked_sku and state.first_time("tf3_fake:" + state.faked_sku):
            return (
                f"The sku \"{state.faked_sku}\" has never been printed in an "
                "observation. Never assemble a sku from a pattern: view the "
                "product or compare its variants until the catalog shows the sku, "
                "then cart exactly that."
            )
        if state.carted and state.cart_pid:
            pid, sku = state.cart_pid, state.cart_sku
            if not well.sku_attested(pid, sku) and state.first_time("tf3_attest"):
                return (
                    f"Exact-sku verification: {sku or '(none)'} on {pid} is not "
                    "attested by any observation. inspect_stock that sku and read "
                    "what comes back before ordering."
                )
            if vault.revealed_tokens and not well.covers_revealed(pid, vault.revealed_tokens):
                if state.first_time("tf3_cover"):
                    return (
                        "The carted sku's rows do not show the disclosed "
                        f"requirement ({', '.join(vault.revealed_tokens[:4])}). "
                        "Either find it in a fresh observation or replace the item."
                    )
            if state.first_time("tf3_sweep"):
                observed = well.rows.get(pid, {})
                sample = ", ".join(f"{k}={v}" for k, v in list(observed.items())[:8]) or "(none)"
                return (
                    "Final sweep before ordering — match every condition to a row "
                    f"of {pid}: {sample}. Anything unmatched is unmet."
                )
        return None

    def _coach_tf4(self) -> str | None:
        state, well = self.state, self.well
        if state.closed:
            return None
        axis = ", ".join(self.brief.leans[:4]) or "the axis the shopper named"
        deep = len(well.probed)
        if state.clarifies >= 1 and state.first_time("tf4_spent"):
            return (
                "Your one clarification is spent — further message calls are "
                f"withheld. Decide from the observed rows on {axis}."
            )
        if not state.carted and deep < 2:
            if state.probes and state.first_time("tf4_second"):
                return (
                    f"Only {deep} candidate verified. Open a second qualifying "
                    f"candidate and read its rows on {axis}; a tilt judged from "
                    "snippets is a guess."
                )
            if not state.probes and state.ever("search") and state.first_time("tf4_first"):
                return (
                    "Open the strongest hit with view — fact rows are the only "
                    "evidence for the shopper's stated tilt."
                )
        if not state.carted and deep >= 2 and state.first_time("tf4_weigh"):
            names = ", ".join(sorted(well.probed)[:3])
            return (
                f"Verified: {names}. Weigh them on {axis} using observed rows only, "
                "say in one line which wins and why, then cart it. Price decides "
                "only when the named axes tie."
            )
        if state.carted and not state.ever("inspect_stock") and state.first_time("tf4_stock"):
            return "Winner carted. inspect_stock it, then place_test_order."
        return None

    def _coach_tf5(self) -> str | None:
        state, well = self.state, self.well
        if state.closed:
            return None
        named = [ref.split("::", 1)[0] for ref in self.brief.pairs]
        unopened = [pid for pid in named if pid not in well.probed]
        if named and unopened and state.first_time("tf5_open:" + str(len(unopened))):
            return (
                "Still unopened from the named set: "
                + ", ".join(unopened[:4])
                + ". Cart calls stay withheld until each one has been observed. "
                "view or compare them now."
            )
        if named and not unopened and not state.carted and state.first_time("tf5_rank"):
            return (
                "All candidates observed. Rank them by the stated priority (first "
                "priority decides, second breaks ties, price last), write the "
                "ordering in your reply, then cart the leader with product_id and "
                "sku as separate arguments."
            )
        if state.event_open and state.first_time("tf5_reshuffle"):
            return (
                "The board shifted: re-inspect the affected candidate and re-sort "
                "before ordering."
            )
        if state.carted and state.first_time("tf5_close"):
            return (
                "Leader carted. inspect_stock, then place_test_order with "
                "product_id and sku only — no justification object here."
            )
        return None

    def _coach_tf6(self) -> str | None:
        state, well = self.state, self.well
        if state.closed:
            return None
        if not state.unavailable_proof:
            if state.lookups and state.first_time("tf6_prove"):
                return (
                    "Evidence order: cart calls for substitutes are withheld until "
                    "an observation says the original is gone. inspect_stock or "
                    "view the item the shopper named and let the environment say "
                    "it first."
                )
            if state.first_time("tf6_hold"):
                return (
                    "Hold every condition for now and look for exactly what the "
                    "shopper asked for. Concessions come after proof, not before."
                )
            return None
        if state.conceded == 0:
            state.conceded = 1
            return (
                "Proof is in: the original is unavailable. Concede the cosmetic "
                "layer only — shade, finish, trim — and keep the product class, "
                "the spend lid, the spec and availability. Tell the shopper with "
                "message what you are trading away, then inspect_stock the "
                "candidate BEFORE the first add_to_cart (intended commitment)."
            )
        if state.conceded == 1 and not state.carted and state.lookups >= 3:
            state.conceded = 2
            return (
                "The cosmetic concession found nothing either. Release the specific "
                "model or maker now, still holding the class, the lid, the spec and "
                "availability. That is the floor — never concede the class itself."
            )
        if (
            not state.carted
            and state.ever("view")
            and not state.ever("inspect_stock")
            and state.first_time("tf6_precart")
        ):
            return (
                "Before the first add_to_cart: inspect_stock the candidate so "
                "the environment records an intended commitment."
            )
        if state.carted and not state.ever("inspect_stock") and state.first_time("tf6_stock"):
            return "Replacement carted. inspect_stock it, then place_test_order."
        return None

    def _coach_tf7(self) -> str | None:
        state, well = self.state, self.well
        if state.closed:
            return None
        pid = well.pin or well.last_pid or state.cart_pid
        if pid:
            observed = well.rows.get(pid, {})
            has_gate_row = any(
                k not in _BASE_FIELDS and k != "brand" and k not in _NO_FIELD
                for k in observed
            )
            if not has_gate_row and state.first_time(f"tf7_rows:{pid}"):
                return (
                    f"Fact rows / listing for {pid} are still thin. view (or re-view) "
                    "that sku so ClaimDesk can attest a non-brand gate claim from "
                    "printed fact_rows before place_test_order."
                )
        if not state.carted and pid:
            if well.decoy(pid, self.brief.ask) and state.first_time("tf7_decoy"):
                return (
                    f"Decoy on {pid}: the asked figure sits in the title and in no "
                    "fact row. Keep viewing — a claim built on that would grade "
                    "false and the first commitment must be row-backed."
                )
            if (
                state.probes
                and not well.row_backed(pid, self.brief.ask)
                and state.first_time("tf7_backed")
            ):
                return (
                    "Commit on evidence only: cart a sku whose fact rows overlap "
                    "the ask. One attested overlapping token is enough when the "
                    "request stacks traits."
                )
        if (
            not state.carted
            and state.probes
            and state.left <= max(state.limit // 2, 5)
            and state.first_time("tf7_half")
        ):
            return (
                "Half the budget is gone: if a viewed sku already attests the gate "
                "requirement and clears the lid and stock, cart it and keep room "
                "for inspect_stock plus place_test_order."
            )

        if (
            "remove_from_cart" in state.last_labels
            and not state.carted
            and not state.closed
        ):
            return (
                "Cart was cleared after the market move. Find a wall-clearing "
                "replacement, view its fact rows, inspect_stock, then "
                "place_test_order — do not leave the run on remove_from_cart."
            )
        return _receipt_tip(state.turn, well, self.brief)

    def _shape(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        payload = dict(args)
        if self.brief.family == TF5:
            payload = self._split_pair(payload)
        if name in {"add_to_cart", "view", "view_product", "inspect_stock"}:
            pid = str(payload.get("product_id") or "")
            if pid:
                self.well.pin = pid
        if name == "add_to_cart" and self.brief.family == TF3:
            pid = str(payload.get("product_id") or "")
            sku = str(payload.get("sku") or "")
            if pid and sku and not self.well.sku_attested(pid, sku):
                self.state.faked_sku = sku
        if name in _TALK_TOOLS:
            self.vault.mark_asked(self.state.turn)
        if name != "place_test_order":
            return payload
        if len(self.brief.pairs) >= 2:
            payload.pop("justification", None)
            payload.pop("claims", None)
            return payload
        if self.brief.family == TF7:
            pid = str(
                payload.get("product_id")
                or self.state.cart_pid
                or self.well.pin
                or self.well.last_pid
            )
            note = payload.get("justification")
            raw = note.get("claims") if isinstance(note, dict) else None
            payload["justification"] = {
                "claims": GateClaimDesk.polish(raw, self.brief, self.well, pid)
            }
            if pid and not payload.get("product_id"):
                payload["product_id"] = pid
            return payload
        payload.pop("justification", None)
        payload.pop("claims", None)
        return payload

    def _split_pair(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = dict(args)
        pid = str(payload.get("product_id") or "")
        if "::" in pid:
            hit = _PAIR_RX.search(pid)
            if hit:
                payload["product_id"] = hit.group(1)
                if not payload.get("sku"):
                    payload["sku"] = hit.group(2)
        sku = str(payload.get("sku") or "")
        if "::" in sku:
            hit = _PAIR_RX.search(sku)
            if hit:
                payload["sku"] = hit.group(2)
                if not payload.get("product_id"):
                    payload["product_id"] = hit.group(1)
        return payload

    def _dispatch(self, calls: list[dict[str, Any]], turn: int) -> dict[str, Any]:
        batch = f"{self.tag}-turn-{turn}"
        actions: list[dict[str, Any]] = []
        for slot, call in enumerate(calls, start=1):
            name = _named(call)
            if not name:
                raise RuntimeError("tool call without a name")
            args = self._shape(name, _arguments(call))
            spec = call.get("function")
            if isinstance(spec, dict):
                spec["arguments"] = json.dumps(args)
            actions.append(
                {"call_id": f"{batch}-{slot}", "action": {"name": name, "args": args}}
            )
        return _link.act(
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
        for attempt in range(len(_POKE) + 1):
            spoken, calls = _link.ask(
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
                    {"role": "user", "content": _POKE[min(attempt, len(_POKE) - 1)]},
                ]
            )
        raise RuntimeError("engine would not emit an executable action")

    def _ingest(
        self,
        result: dict[str, Any],
        transcript: list[dict[str, Any]],
        calls: list[dict[str, Any]],
    ) -> None:
        rows = result.get("calls") or []
        for index, call in enumerate(calls, start=1):
            row = rows[index - 1] if index - 1 < len(rows) else None
            obs = row.get("observation") if isinstance(row, dict) else None
            name = _named(call)
            self.well.soak(obs, probed=name in _PROBE_TOOLS)
            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": _reference(call, f"{self.tag}-{self.state.turn}-{index}"),
                    "content": _condense(obs),
                }
            )
            if _event_in(obs):
                self.state.event_open = True
                ep, es = _event_target(obs)
                snap = (
                    ep
                    or self.state.cart_pid
                    or self.well.pin
                    or self.well.last_pid
                    or ""
                )
                if snap:
                    self.state.event_pid = snap
                    if es:
                        self.state.event_sku = es
                    elif self.state.cart_sku:
                        self.state.event_sku = self.state.cart_sku

            if self.brief.family == TF6:
                if name in _PROBE_TOOLS and _says_unavailable(obs):
                    self.state.unavailable_proof = True
                elif name in _LOOKUP_TOOLS and _empty_result(obs) and self.state.lookups >= 2:
                    self.state.unavailable_proof = True
        note = result.get("user_message")
        if isinstance(note, dict) and note.get("content"):
            text = str(note["content"])
            transcript.append({"role": "user", "content": text})
            if _reads_as_question(text):
                self.state.question_open = True

            self.vault.take_answer(text)
            if self.vault.revealed_tokens:
                transcript.append(
                    {
                        "role": "user",
                        "content": (
                            "Gate open. Disclosed tokens: "
                            + ", ".join(self.vault.revealed_tokens[:8])
                            + ". Use them exactly as given."
                        ),
                    }
                )
        if self.state.event_open and not self.state.closed:
            tip = (
                "Event note: inspect_stock the event-target sku "
                f"({self.state.event_pid or self.state.cart_pid or 'carted'}) "
                "to close the event — inspecting a replacement does not. "
                "Order only after that close."
            )
            if self.brief.needs_claims():
                tip += (
                    " Then rebuild justification.claims from the fresh observation "
                    "before place_test_order."
                )
            transcript.append({"role": "user", "content": tip})

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    return GateEpisode(problem_data).run()

__all__ = ["agent_main"]
