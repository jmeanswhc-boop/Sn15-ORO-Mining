

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_SWIFT_MODEL = "deepseek/deepseek-v3.2"
_DENSE_MODEL = "qwen/qwen3.5-397b-a17b"
_BACKUP_MODELS = (
    "google/gemma-4-31b-it",
    "openai/gpt-oss-120b",
    "deepseek/deepseek-chat-v3.1",
)

DECREE, DRAGNET, CLAMP, WEIGH, SLATE, RELAX, WARRANT = (
    "decree",
    "dragnet",
    "clamp",
    "weigh",
    "slate",
    "relax",
    "warrant",
)
_FAMILIES = (DECREE, DRAGNET, CLAMP, WEIGH, SLATE, RELAX, WARRANT)

_FAMILY_PRIORITY = {RELAX: 7, SLATE: 6, WARRANT: 5, CLAMP: 4, DECREE: 3, WEIGH: 2, DRAGNET: 1}

_DENSE_FAMILIES = frozenset({RELAX, SLATE, WARRANT, CLAMP, DECREE})

_GOAL_SIGNS: dict[str, tuple[tuple[str, int], ...]] = {
    RELAX: (
        ("if it is gone", 4),
        ("if that happens", 4),
        ("closest thing", 4),
        ("nearest", 3),
        ("relax", 3),
        ("fall back", 3),
        ("otherwise", 2),
        ("still want", 2),
    ),
    SLATE: (
        ("shortlist these", 4),
        ("these three", 4),
        ("put them in order", 4),
        ("rank", 3),
        ("top pick", 3),
        ("second choice", 3),
        ("tie", 2),
    ),
    WARRANT: (
        ("justification", 5),
        ("prove", 4),
        ("show me why", 4),
        ("on the record", 3),
        ("justif", 3),
        ("why", 2),
    ),
    CLAMP: (
        ("has to be exactly", 4),
        ("only tell you", 4),
        ("i will tell you", 3),
        ("strictly", 3),
        ("no substitutions", 4),
        ("exact", 2),
    ),
    DECREE: (
        ("house rule", 4),
        ("non-negotiable", 4),
        ("brand rule", 4),
        ("always buy", 3),
        ("stick to", 3),
        ("only from", 3),
        ("hard requirement", 3),
        ("ideally", 2),
        ("allowed", 2),
    ),
    WEIGH: (
        ("weigh", 3),
        ("worth paying", 4),
        ("i would pay", 4),
        ("unless it costs", 4),
        ("prefer the cheaper", 3),
        ("value", 2),
    ),
    DRAGNET: (
        ("somewhere in", 3),
        ("cannot find", 4),
        ("hard to find", 4),
        ("search for", 2),
        ("wording", 3),
        ("keyword", 3),
    ),
}

_RULE_SIGNS: dict[str, tuple[tuple[str, int], ...]] = {
    RELAX: (("least relax", 5), ("recovery", 4), ("unavailable", 3), ("event", 2)),
    SLATE: (("ranking", 4), ("ordered list", 3), ("priority order", 3)),
    WARRANT: (("claims", 5), ("justification", 5), ("field", 2)),
    CLAMP: (("constraint", 4), ("literal", 3), ("must equal", 3)),
    DECREE: (("intent", 4), ("binding", 3), ("preference vs", 3)),
    WEIGH: (("preference", 4), ("tradeoff", 3), ("soft", 2)),
    DRAGNET: (("retrieval", 4), ("recall", 4), ("reformulate", 3)),
}

_LEGAL_FIELD_RX = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_SLATE_ID_RX = re.compile(r"\b\d{6,}::[\w-]+(?:\.[\w-]+)*")
_MODEL_RX = re.compile(r"\b(?=[a-z0-9-]*\d)[a-z][a-z0-9]{1,}(?:-[a-z0-9]+)*\b", re.I)
_QUOTE_RX = re.compile(r"[\"\u201c]([^\"\u201d]{3,70})[\"\u201d]")
_LIMIT_RX = re.compile(
    r"(?:at or (?:under|below)|under|below|up to|no more than|max(?:imum)?(?:\s+of)?|"
    r"not exceed(?:ing)?)\s*([\d][\d,]*(?:\.\d+)?)",
    re.I,
)
_PROOF_RX = re.compile(
    r"\bjustif|\bprove\b|\bshow me why\b|\bon the record\b|\bwhy\b|"
    r"\breasons?\b|\bexplain\b|\bdefend\b|\bwalk me through\b|\bgive me a reason\b",
    re.I,
)
_HELD_RX = re.compile(
    r"\bask\s+(?:me|for)\b|\bonly\s+tell\s+you\b|\bi\s+will\s+tell\s+you\b|"
    r"\bwithh(?:old|eld)\b|\bif\s+you\s+ask\b",
    re.I,
)
_UNIT_RX = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s?(?:inch|in|hz|p|k|gb|tb|mm|cm|mp|w|mah|oz|ml|"
    r"l|kg|lb|ft|yd|\"))?\b",
    re.I,
)
_PLAIN_WORD_RX = re.compile(r"[a-z][a-z0-9-]{1,}")

_BASE_CLAIMS = frozenset({"price", "in_stock"})

_BLOCKED_FIELDS = frozenset(
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
        "display",
        "facts",
        "options",
    }
)

_COSMETIC_WORDS = frozenset(
    {
        "black", "white", "navy", "blue", "red", "green", "yellow", "gray",
        "grey", "brown", "beige", "khaki", "cream", "ivory", "pink", "purple",
        "orange", "gold", "silver", "bronze", "tan", "charcoal", "olive",
        "matte", "glossy", "gloss", "satin", "brushed", "polished", "chrome",
        "oak", "walnut", "maple", "cherry", "espresso", "natural", "clear",
        "shade", "colour", "color", "finish", "tone", "pattern", "print",
    }
)

_SPEC_WORDS = frozenset(
    {
        "xs", "s", "m", "l", "xl", "xxl", "small", "medium", "large",
        "cotton", "polyester", "wool", "silk", "linen", "leather", "suede",
        "denim", "canvas", "nylon", "plastic", "metal", "wood", "glass",
        "narrow", "wide", "regular", "slim", "hd", "uhd", "qhd", "fhd",
        "oled", "lcd", "led", "matte", "glossy", "navy", "black", "white",
        "gray", "grey", "brown", "beige", "khaki",
    }
)

_NIGHT_STOP = frozenset(
    {
        "a", "an", "the", "and", "or", "for", "with", "from", "that", "this",
        "these", "those", "please", "need", "want", "find", "buy", "get",
        "looking", "look", "help", "me", "my", "i", "you", "your", "we",
        "it", "its", "them", "they", "something", "anything", "some", "any",
        "one", "ones", "product", "products", "item", "items", "thing",
        "order", "cart", "checkout", "store", "shop", "shopper", "must",
        "should", "would", "could", "also", "only", "than", "then", "when",
        "which", "what", "who", "have", "has", "had", "are", "is", "be",
        "was", "were", "to", "of", "on", "at", "by", "as", "if", "not", "no",
        "yes", "true", "false", "very", "really", "much", "more", "most",
        "less", "least", "about", "around", "over", "into", "out", "up",
        "down", "in", "so", "but", "because", "while", "there", "here",
        "now", "still", "just", "like", "price", "budget", "cheap", "cheaper",
        "cheapest", "cost", "stock", "available", "shipping", "delivery",
        "warranty", "php", "usd", "peso", "dollars", "under", "below",
        "above", "why", "reason", "justify", "explain", "brand", "size",
        "color", "colour", "material", "width", "capacity", "dimensions",
        "finish", "resolution", "screen", "type", "kind",
    }
)

def _slug(raw: Any) -> str:

    text = str(raw or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_").replace("/", "_")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "", text)).strip("_")

def _squash(raw: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(raw or "").lower())

def _backed(value: Any, blob: Any) -> bool:

    needle, hay = _squash(value), _squash(blob)
    return bool(needle) and needle in hay

def _token_seen(token: str, text: str) -> bool:

    tok = str(token or "").strip().lower()
    blob = str(text or "").lower()
    if not tok or not blob:
        return False
    if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", blob):
        return True
    packed, hay = _squash(tok), _squash(blob)
    if not packed:
        return False
    if len(packed) <= 3:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(packed)}(?![a-z0-9])", hay))
    return packed in hay

def _asked_specs(goal: str) -> list[str]:

    low = _LIMIT_RX.sub(" ", (goal or "").lower())
    picked: list[str] = []
    seen: set[str] = set()
    for hit in _UNIT_RX.finditer(low):
        token = hit.group(0).strip()
        key = _squash(token)
        if key and key not in seen:
            seen.add(key)
            picked.append(token)
    for word in _PLAIN_WORD_RX.findall(low):
        if word in _NIGHT_STOP or word not in _SPEC_WORDS:
            continue
        key = _squash(word)
        if key and key not in seen:
            seen.add(key)
            picked.append(word)
    return picked

def _joined_drawer_name(field: str) -> bool:

    flat = _slug(field)
    if not flat:
        return False
    for drawer in ("display", "facts", "options", "os", "size", "title"):
        if flat.startswith(drawer + "_") or flat.endswith("_" + drawer):
            return True
    return False

def _read_json(text: str) -> dict[str, Any]:
    body = (text or "").strip()
    if not body:
        return {}
    try:
        loaded = json.loads(body)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        pass
    head, tail = body.find("{"), body.rfind("}")
    if head >= 0 and tail > head:
        try:
            loaded = json.loads(body[head : tail + 1])
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}

def _halve_brief(text: str) -> tuple[str, str]:
    head, sep, tail = (text or "").partition("\n\n")
    if not sep:
        return (text or "").strip(), ""
    body = tail.strip()
    first, _, rest = body.partition("\n")
    if first.rstrip().endswith(":") and len(first) < 40:
        body = rest.strip()
    return head.strip(), body

def _distinct(items: Any) -> list[str]:
    return list(dict.fromkeys(str(i) for i in items))

def _order_tool_fn(tools: list[Any]) -> dict[str, Any] | None:
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        if isinstance(fn, dict) and fn.get("name") == "place_test_order":
            return fn
    return None

def _claim_field_spec(tools: list[Any]) -> dict[str, Any] | None:
    fn = _order_tool_fn(tools)
    if not fn:
        return None
    props = (fn.get("parameters") or {}).get("properties") or {}
    warrant = props.get("justification") if isinstance(props, dict) else None
    claims = (warrant.get("properties") or {}).get("claims") if isinstance(warrant, dict) else None
    items = claims.get("items") if isinstance(claims, dict) else None
    field = (items.get("properties") or {}).get("field") if isinstance(items, dict) else None
    return field if isinstance(field, dict) else None

def _live_claim_fields(tools: list[Any]) -> list[str]:

    field = _claim_field_spec(tools)
    enum = field.get("enum") if isinstance(field, dict) else None
    if not isinstance(enum, list):
        return []
    out, seen = [], set()
    for raw in enum:
        flat = _slug(raw)
        if flat and flat not in seen:
            seen.add(flat)
            out.append(flat)
    return out

def _live_claim_rule(tools: list[Any]) -> str:
    field = _claim_field_spec(tools)
    return str(field.get("description") or "")[:600] if isinstance(field, dict) else ""

def _offered_tools(tools: list[Any]) -> list[str]:
    names: list[str] = []
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
        elif entry.get("name"):
            names.append(str(entry["name"]))
    return names

def _call_label(call: Any) -> str:
    spec = call.get("function") if isinstance(call, dict) else None
    return str(spec["name"]) if isinstance(spec, dict) and spec.get("name") else ""

def _call_args(call: Any) -> dict[str, Any]:
    spec = call.get("function") if isinstance(call, dict) else None
    raw = spec.get("arguments", "{}") if isinstance(spec, dict) else "{}"
    if isinstance(raw, dict):
        return raw
    try:
        loaded = json.loads(raw or "{}")
    except json.JSONDecodeError:
        loaded = {}
    return loaded if isinstance(loaded, dict) else {}

def _call_key(call: Any, fallback: str) -> str:
    return str(call.get("id") or fallback) if isinstance(call, dict) else fallback

def _trim(obs: Any, cap: int = 6000) -> str:
    try:
        text = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obs)
    return text if len(text) <= cap else text[: cap - 18] + "…<clipped>"

def _episode_done(result: dict[str, Any]) -> bool:
    for row in result.get("calls") or []:
        if isinstance(row, dict):
            obs = row.get("observation")
            if isinstance(obs, dict) and obs.get("done"):
                return True
    return False

def _event_fired(obs: Any) -> bool:

    if not isinstance(obs, dict):
        return False
    signals = ("market_event", "stock_changed", "price_changed", "event")
    if any(obs.get(key) for key in signals):
        return True
    nested = obs.get("observation")
    if isinstance(nested, dict):
        return any(nested.get(key) for key in signals[:3])
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

def _price_moved(obs: Any) -> bool:
    if not isinstance(obs, dict):
        return False
    if obs.get("price_changed"):
        return True
    nested = obs.get("observation")
    if isinstance(nested, dict) and nested.get("price_changed"):
        return True
    text = str(obs.get("message") or "").lower()
    return "price" in text and ("chang" in text or "increase" in text or "rose" in text)

def _rejected(obs: Any) -> bool:

    if not isinstance(obs, dict):
        return False
    if obs.get("error") or obs.get("ok") is False or obs.get("success") is False:
        return True
    blob = str(obs.get("message") or obs.get("status") or "").lower()
    return any(
        cue in blob
        for cue in (
            "unavailable",
            "out of stock",
            "sold out",
            "not found",
            "invalid",
            "failed",
            "cannot",
            "no longer",
        )
    )

def _shopper_question(text: str) -> bool:
    body = (text or "").strip().lower()
    if not body:
        return False
    if "?" in body:
        return True
    openers = ("which", "what", "does", "did", "can you", "could you", "would you", "wait")
    return any(body.startswith(opener) for opener in openers)

class NightBrief:

    __slots__ = (
        "raw",
        "goal",
        "rules",
        "family",
        "limit_figure",
        "slate_ids",
        "quoted",
        "wants_proof",
        "held_figure",
        "specs",
        "claim_fields",
        "claim_rule",
    )

    def __init__(self, raw: str, tools: list[Any] | None = None) -> None:
        self.raw = raw or ""
        self.goal, self.rules = _halve_brief(self.raw)
        self.slate_ids = _distinct(_SLATE_ID_RX.findall(self.raw))
        limit = _LIMIT_RX.search(self.goal) or _LIMIT_RX.search(self.raw)
        self.limit_figure = limit.group(1) if limit else ""
        quoted = _QUOTE_RX.search(self.goal)
        self.quoted = quoted.group(1) if quoted else ""
        self.wants_proof = bool(_PROOF_RX.search(self.goal))
        self.held_figure = bool(_HELD_RX.search(self.goal))
        self.specs = _asked_specs(self.goal)
        self.claim_fields = _live_claim_fields(tools or [])
        self.claim_rule = _live_claim_rule(tools or [])
        self.family = self._pick_family()

    def _pick_family(self) -> str:

        if len(self.slate_ids) >= 2:
            return SLATE

        if self.wants_proof:
            return WARRANT
        tally = dict.fromkeys(_FAMILIES, 0)
        ask, rules = self.goal.lower(), self.rules.lower()
        for family, signs in _GOAL_SIGNS.items():
            tally[family] += sum(weight for sign, weight in signs if sign in ask)
        for family, signs in _RULE_SIGNS.items():

            if family == WARRANT:
                continue
            tally[family] += sum(weight for sign, weight in signs if sign in rules)
        if self.held_figure:
            tally[CLAMP] += 7

        if self.wants_proof:
            tally[WARRANT] += 8
        if self.quoted:
            tally[DRAGNET] += 4
        if re.search(r"\bgone\b|\bunavailable\b|\bsold out\b|\bstock\b", ask):
            tally[RELAX] += 5
        if re.search(r"\bideally\b|\bnice to have\b|\bprefer\b", ask) and re.search(
            r"\bmust\b|\bnon-negotiable\b|\bfirm\b|\bhard requirement\b", ask
        ):
            tally[DECREE] += 4
        tally[RELAX] += 2
        best = max(tally.values())
        if best <= 0:
            return RELAX
        return max(
            [f for f, score in tally.items() if score == best],
            key=lambda f: _FAMILY_PRIORITY[f],
        )

    def needs_warrant(self) -> bool:

        return self.family == WARRANT

    def sheet(self) -> str:
        lines: list[str] = []
        if self.limit_figure:
            lines.append(f"- ceiling figure in the ask: {self.limit_figure}")
        if self.quoted:
            lines.append(f"- quoted wording to verify on facts: {self.quoted}")
        if self.slate_ids:
            lines.append(f"- named slate: {', '.join(self.slate_ids)}")
        if self.specs:
            lines.append(f"- specification tokens to attest: {', '.join(self.specs)}")
        if self.held_figure:
            lines.append("- one figure stays hidden until you ask for it")
        if self.claim_fields:
            lines.append(f"- live claim-field enum: {', '.join(self.claim_fields)}")
        if self.claim_rule:
            lines.append(f"- live claim rule: {self.claim_rule[:280]}")
        return ("NIGHT SHEET\n" + "\n".join(lines) + "\n") if lines else ""

class ShadeBook:

    __slots__ = ("shades", "models", "core", "conceded")

    def __init__(self, goal: str) -> None:
        low = (goal or "").lower()
        words = [w for w in _PLAIN_WORD_RX.findall(low) if w not in _NIGHT_STOP]
        self.shades = _distinct([w for w in words if w in _COSMETIC_WORDS])
        model_like: list[str] = []
        for hit in _MODEL_RX.finditer(low):
            token = hit.group(0)
            if token in _NIGHT_STOP or token in _COSMETIC_WORDS:
                continue
            if not any(ch.isdigit() for ch in token):
                continue
            model_like.append(token)
        self.models = _distinct(model_like)[:4]
        self.core = _distinct(
            [
                w
                for w in words
                if w not in _COSMETIC_WORDS and len(w) >= 4 and w not in self.models
            ]
        )[:4]
        self.conceded: list[str] = []

    def concede_for(self, rung: int) -> str:

        if rung <= 0:
            return ""
        pool = self.shades if rung == 1 else self.models or self.shades
        for token in pool:
            if token not in self.conceded:
                return token
        return ""

    def concede(self, rung: int) -> str:
        token = self.concede_for(rung)
        if token:
            self.conceded.append(token)
        return token

    def hold_tokens(self, rung: int) -> list[str]:

        keep = list(self.core)
        if rung < 2:
            keep += [m for m in self.models if m not in self.conceded]
        if rung < 1:
            keep += [s for s in self.shades if s not in self.conceded]
        return _distinct(keep)[:6]

    def blurb(self, rung: int) -> str:
        parts = []
        if self.core:
            parts.append("class tokens " + "/".join(self.core[:3]))
        if self.models:
            parts.append("model tokens " + "/".join(self.models[:2]))
        if self.shades:
            parts.append("shade tokens " + "/".join(self.shades[:2]))
        if self.conceded:
            parts.append("already conceded " + "/".join(self.conceded))
        return f"rung {rung}: " + ("; ".join(parts) if parts else "no distinctive tokens read")

class LadderState:

    __slots__ = (
        "level",
        "rung1_tried",
        "failures",
        "last_failure",
        "events_opened",
        "events_closed",
        "reranked",
        "repriced",
        "turn",
        "limit",
        "history",
        "last_labels",
        "carted",
        "cart_pid",
        "cart_sku",
        "event_pid",
        "event_sku",
        "removed",
        "closed",
        "owes_reply",
        "asked_held",
        "queries",
        "spoken",
    )

    def __init__(self, limit: int) -> None:
        self.level = 0
        self.rung1_tried = False
        self.failures = 0
        self.last_failure = ""
        self.events_opened = 0
        self.events_closed = 0
        self.reranked = False
        self.repriced = False
        self.turn = 0
        self.limit = max(1, limit)
        self.history: set[str] = set()
        self.last_labels: list[str] = []
        self.carted = False
        self.cart_pid = ""
        self.cart_sku = ""
        self.event_pid = ""
        self.event_sku = ""
        self.removed = 0
        self.closed = False
        self.owes_reply = False
        self.asked_held = False
        self.queries: list[str] = []
        self.spoken: set[str] = set()

    def bump_on_failure(self, reason: str = "") -> int:

        self.failures += 1
        self.last_failure = reason or self.last_failure
        if self.level == 0:
            self.level = 1
            self.rung1_tried = True
        elif self.level == 1:
            self.level = 2
        return self.level

    def may_reach_rung2(self) -> bool:

        return self.rung1_tried

    def coach_text(self, book: ShadeBook) -> str:

        hold = ", ".join(book.hold_tokens(self.level)) or "the product class"
        if self.level == 0:
            return (
                "Ladder rung 0: every constraint the shopper stated is still firm. "
                f"Hold {hold}. Nothing is conceded until an observation actually "
                "proves the exact item cannot be bought."
            )
        if self.level == 1:
            giving = book.concede_for(1) or "the cosmetic attribute (shade or finish)"
            return (
                "Ladder rung 1: concede the cosmetic attribute ONLY — give up "
                f"{giving} and keep everything else, including {hold}, the ceiling "
                "figure and in-stock. Do not touch the model or the variant yet."
            )
        return (
            "Ladder rung 2: the cosmetic concession was already tried and failed, so "
            f"the variant or model may now move. Keep {hold}, the product class, the "
            "ceiling figure and in-stock firm — those never bend, and you never skip "
            "back down a rung to re-try what already failed."
        )

    @property
    def left(self) -> int:
        return max(0, self.limit - self.turn)

    @property
    def event_open(self) -> bool:
        return self.events_opened > self.events_closed

    def past_half(self) -> bool:
        return self.turn * 2 >= self.limit

    def ever(self, label: str) -> bool:
        return label in self.history

    def hunted(self) -> bool:
        return bool({"search", "filter"} & self.history)

    def mark_event(self, priced: bool = False, pid: str = "", sku: str = "") -> None:
        self.events_opened += 1
        if priced:
            self.repriced = True
        snap = pid or self.cart_pid
        if snap:
            self.event_pid = snap
            self.event_sku = sku or self.cart_sku or self.event_sku

    def close_event(self) -> None:
        if self.event_open:
            self.events_closed = self.events_opened
            self.event_pid = ""
            self.event_sku = ""

    def note(self, labels: list[str], calls: list[dict[str, Any]]) -> None:
        self.last_labels = labels
        self.history.update(labels)
        if "add_to_cart" in labels:
            self.carted = True
            for call in calls:
                if _call_label(call) != "add_to_cart":
                    continue
                args = _call_args(call)
                self.cart_pid = str(args.get("product_id") or self.cart_pid)
                self.cart_sku = str(args.get("sku") or self.cart_sku)
        if "remove_from_cart" in labels:
            self.carted = False
            self.removed += 1
        if "inspect_stock" in labels:
            for call in calls:
                if _call_label(call) != "inspect_stock":
                    continue
                pid = str(_call_args(call).get("product_id") or "")
                if self.event_pid and pid == self.event_pid:
                    self.close_event()
                    break
        if "place_test_order" in labels:
            self.closed = True
        if "message" in labels:
            self.owes_reply = False
            self.asked_held = True
        if {"search", "filter", "compare"} & set(labels):
            self.reranked = True

    def note_query(self, text: Any) -> None:
        query = str(text or "").strip()
        if query:
            self.queries.append(query[:160])

    def say_once(self, key: str) -> bool:
        if key in self.spoken:
            return False
        self.spoken.add(key)
        return True

class NightWell:

    __slots__ = ("rows", "blurbs", "titles", "last_pid", "target_pid", "price", "stock")

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, str]] = {}
        self.blurbs: dict[str, str] = {}
        self.titles: dict[str, str] = {}
        self.last_pid = ""
        self.target_pid = ""
        self.price = ""
        self.stock = ""

    def stash(self, pid: str, key: Any, value: Any) -> None:
        flat = _slug(key)
        if not pid or not flat or value is None or isinstance(value, (dict, list)):
            return
        text = str(value).strip()
        if not text:
            return
        if flat == "in_stock":
            low = text.lower()
            if low in {"true", "1", "yes", "in_stock"}:
                text = "true"
            elif low in {"false", "0", "no"}:
                text = "false"
            self.stock = text
        if flat == "price":
            self.price = text
        self.rows.setdefault(pid, {})[flat] = text
        self.last_pid = pid

    def absorb(self, obs: Any) -> None:
        if not isinstance(obs, dict):
            return

        def descend(node: Any, inherited: str = "") -> None:
            if not isinstance(node, dict):
                return
            pid = str(node.get("product_id") or "") or inherited
            title = str(node.get("title") or "")
            if pid and title:
                self.titles[pid] = title
            pieces: list[str] = []
            if title:
                pieces.append(title)
            if node.get("options"):
                pieces.append(str(node["options"]))
            for key in ("price", "in_stock"):
                if key in node:
                    self.stash(pid, key, node.get(key))
            if node.get("brand"):
                self.stash(pid, "brand", node.get("brand"))
            drawer = node.get("facts")
            if isinstance(drawer, dict):
                self._unfold(drawer, pid, pieces)
            if pid and pieces:
                prior = self.blurbs.get(pid, "")
                chunk = " | ".join(pieces)
                if chunk not in prior:
                    self.blurbs[pid] = (prior + " | " + chunk if prior else chunk)[:2200]
            for key in (
                "observation",
                "product",
                "compared",
                "variants",
                "results",
                "items",
                "alternatives",
            ):
                child = node.get(key)
                if isinstance(child, dict):
                    descend(child, pid)
                elif isinstance(child, list):
                    for item in child[:12]:
                        descend(item, pid)

        descend(obs)

    def _unfold(
        self, block: dict[str, Any], pid: str, pieces: list[str], prefix: str = ""
    ) -> None:
        for key, value in block.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                packed = json.dumps(value, ensure_ascii=False)[:160]
                pieces.append(f"{path}={packed}")
                self.stash(pid, key, packed)
                self._unfold(value, pid, pieces, path)
            elif isinstance(value, list):
                text = ", ".join(str(x) for x in value[:12])
                pieces.append(f"{path}={text}")
                self.stash(pid, key, text)
            else:
                text = str(value).strip()
                if not text:
                    continue
                pieces.append(f"{path}={text}")
                self.stash(pid, key, text)
                if prefix:
                    self.stash(pid, path.rsplit(".", 1)[-1], text)

    def rows_for(self, pid: str) -> list[dict[str, str]]:
        return [{"key": k, "value": v} for k, v in self.rows.get(pid, {}).items()]

    def proof_blob(self, pid: str) -> str:
        return self.blurbs.get(pid, "") + " | " + " | ".join(
            f"{k}={v}" for k, v in self.rows.get(pid, {}).items()
        )

    def title_only_spec(self, pid: str, goal: str) -> bool:
        title = self.titles.get(pid, "")
        if not title or not goal:
            return False
        blob = " ".join(f"{k} {v}" for k, v in self.rows.get(pid, {}).items())
        return any(
            _token_seen(token, title) and not _token_seen(token, blob)
            for token in _asked_specs(goal)
        )

    def spec_attested(self, pid: str, goal: str) -> bool:
        if not pid or pid not in self.rows:
            return False
        blob = " ".join(f"{k} {v}" for k, v in self.rows.get(pid, {}).items())
        tokens = _asked_specs(goal)
        if not tokens:
            return any(k not in _BASE_CLAIMS for k in self.rows.get(pid, {}))
        return any(_token_seen(token, blob) for token in tokens)

_WARRANT_FILL = """
Pick the justification claim covering ONE requirement the shopper stated.

You receive JSON with: goal, allowed_fields (live schema enum, possibly empty),
schema_rule (the tool's own wording for legal fields), have (fields already
claimed), listing (observed text for this exact sku), fact_rows (key/value
pairs; catalogue drawer keys are VALUE evidence only and their names differ by
category).

Answer with JSON only: {"claims":[{"field":"...","value":"..."}]}

Requirements:
- Address the specification `have` does not cover yet.
- Follow schema_rule. A non-empty allowed_fields list is exhaustive: pick from
  it. When it is empty, name the conventional snake_case attribute for that
  kind of purchase (apparel size/material/color, shoes size/material/width,
  bags and home capacity/dimensions/finish, electronics resolution/screen_size)
  rather than the drawer key.
- The value must be copied verbatim from listing or fact_rows. Shopper
  shorthand (large, cotton, navy, 720p, 27-inch) is still that attribute's
  value — claim the printed token, never the shorthand as a field name.
- Never concatenate a drawer name with an attribute (no display_resolution).
- Never use display, os, category, title, summary, evidence or reason.
- Leave price and in_stock alone; they are handled elsewhere.
- A short brand claim is allowed only when a brand fact row prints it.
- Return {"claims":[]} when the evidence does not support the specification.
- Never more than two claims.
""".strip()

class NightClaims:

    @staticmethod
    def refine(
        incoming: Any, brief: NightBrief, well: NightWell, pid: str
    ) -> list[dict[str, str]]:
        legal = {_slug(f) for f in brief.claim_fields}
        observed = well.rows.get(pid, {})
        proof = well.proof_blob(pid)
        out: list[dict[str, str]] = []
        taken: set[str] = set()

        for entry in incoming if isinstance(incoming, list) else []:
            row = NightClaims._vet(entry, legal, observed, proof, brief, taken)
            if row:
                taken.add(row["field"])
                out.append(row)

        NightClaims._add_base(out, taken, legal, observed, well)

        if "brand" not in taken and (not legal or "brand" in legal):
            brand = NightClaims._brand_value(observed.get("brand", ""), observed, proof)
            if brand and _backed(brand, brief.goal):
                out.append({"field": "brand", "value": brand})
                taken.add("brand")

        if brief.needs_warrant() and not NightClaims._gate_present(out):
            for candidate in NightClaims._light_fill(brief, well, pid, taken):
                row = NightClaims._vet(candidate, legal, observed, proof, brief, taken)
                if not row:
                    continue
                taken.add(row["field"])
                out.append(row)
                if NightClaims._gate_present(out):
                    break

        if brief.needs_warrant() and not NightClaims._gate_present(out):
            gate = NightClaims._gate_from_facts(brief.goal, observed, proof, taken, legal)
            if gate:
                taken.add(gate["field"])
                out.append(gate)

        if (
            brief.needs_warrant()
            and "brand" not in taken
            and (not legal or "brand" in legal)
            and observed.get("brand")
            and len(out) < 4
        ):
            brand = NightClaims._brand_value(observed.get("brand", ""), observed, proof)
            if brand:
                out.append({"field": "brand", "value": brand})
                taken.add("brand")

        return out[:6]

    @staticmethod
    def _vet(
        entry: Any,
        legal: set[str],
        observed: dict[str, str],
        proof: str,
        brief: NightBrief,
        taken: set[str],
    ) -> dict[str, str] | None:
        if not isinstance(entry, dict):
            return None
        field = _slug(entry.get("field"))
        value = "" if entry.get("value") is None else str(entry.get("value")).strip()
        if not field or not value or field in taken:
            return None
        if field in _BLOCKED_FIELDS or _joined_drawer_name(field):
            return None
        if legal and field not in legal:
            return None
        if not _LEGAL_FIELD_RX.match(field):
            return None
        if field == "in_stock":
            if value.lower() in {"true", "1", "yes", "in_stock"}:
                value = "true"
            return {"field": field, "value": value}
        if field == "price":
            return {"field": field, "value": value}
        if field == "brand":
            brand = NightClaims._brand_value(value, observed, proof)
            return {"field": field, "value": brand} if brand else None
        if not _backed(value, proof):
            swap = NightClaims._closest_fact(value, observed, brief.goal, proof)
            if not swap:
                return None
            value = swap
        return {"field": field, "value": value}

    @staticmethod
    def _add_base(
        out: list[dict[str, str]],
        taken: set[str],
        legal: set[str],
        observed: dict[str, str],
        well: NightWell,
    ) -> None:

        for need in ("price", "in_stock"):
            if need in taken or (legal and need not in legal):
                continue
            value = observed.get(need) or (well.price if need == "price" else well.stock)
            if not value:
                continue
            out.insert(0 if need == "price" else min(1, len(out)), {"field": need, "value": value})
            taken.add(need)

    @staticmethod
    def _gate_present(claims: list[dict[str, str]]) -> bool:
        fields = {_slug(c["field"]) for c in claims}
        return bool(fields - _BASE_CLAIMS - _BLOCKED_FIELDS - {"brand"})

    @staticmethod
    def _gate_from_facts(
        goal: str,
        rows: dict[str, str],
        proof: str,
        have: set[str],
        legal: set[str],
    ) -> dict[str, str] | None:

        goal_l = (goal or "").lower()
        ranked: list[tuple[int, str, str]] = []
        for raw_key, raw_val in rows.items():
            key = _slug(raw_key)
            value = str(raw_val or "").strip()
            if not key or not value or key in have:
                continue
            if key in _BASE_CLAIMS or key == "brand" or key in _BLOCKED_FIELDS:
                continue
            field = key
            if field in have or field in _BLOCKED_FIELDS or _joined_drawer_name(field):
                continue
            if legal and field not in legal:
                continue
            if not _LEGAL_FIELD_RX.match(field):
                continue
            if not _backed(value, proof):
                continue
            score = 0
            if field in goal_l or key in goal_l:
                score += 12
            if _backed(value, goal):
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
    def _light_fill(
        brief: NightBrief, well: NightWell, pid: str, taken: set[str]
    ) -> list[dict[str, Any]]:
        proof = well.proof_blob(pid)
        if not proof.strip(" |"):
            return []
        payload = {
            "goal": brief.goal[:1500],
            "allowed_fields": [f for f in brief.claim_fields if f not in taken],
            "schema_rule": brief.claim_rule,
            "have": sorted(taken),
            "listing": proof[:2200],
            "fact_rows": well.rows_for(pid)[:40],
        }
        reply = _bridge.json_turn(
            engine=_SWIFT_MODEL,
            system=_WARRANT_FILL,
            user=json.dumps(payload, ensure_ascii=False),
        )
        rows = _read_json(reply).get("claims")
        return [r for r in rows if isinstance(r, dict)][:2] if isinstance(rows, list) else []

    @staticmethod
    def _closest_fact(
        value: str, observed: dict[str, str], goal: str, proof: str
    ) -> str:

        if value and _backed(value, proof):
            return value
        scored: list[tuple[int, str]] = []
        for key, printed in observed.items():
            if not printed or key in _BASE_CLAIMS or key == "brand":
                continue
            if not _backed(printed, proof):
                continue
            score = 0
            if _backed(printed, goal):
                score += 5
            if value and (_backed(printed, value) or _backed(value, printed)):
                score += 4
            if key in _BLOCKED_FIELDS or key in {"display", "size", "facts", "options"}:
                score += 1
            if score:
                scored.append((score, printed))
        if not scored:
            return ""
        scored.sort(key=lambda pair: (-pair[0], -len(pair[1])))
        return scored[0][1]

    @staticmethod
    def _brand_value(raw: str, observed: dict[str, str], proof: str) -> str:

        printed = str(observed.get("brand") or "").strip()
        if not printed or not _backed(printed, proof) or len(printed.split()) > 4:
            return ""
        candidate = str(raw or "").strip()
        if (
            candidate
            and len(candidate.split()) <= 3
            and len(candidate) <= 40
            and _backed(candidate, printed)
        ):
            return candidate
        return printed

_CHARTER = """
You are NightGirl, an EnvPack purchasing operator who expects the catalogue to
change mid-run. Tools are your only source of truth; nothing is real until an
observation printed it. The run ends when an observation returns done=true
after place_test_order.

RUN SHAPE
  Hunt      search or filter with printed nouns from the ask
  Read      view or compare and read FACT ROWS — titles are advertising
  Hold      add_to_cart only a fact-backed, in-budget, in-stock variant
  Settle    inspect_stock after carting and after ANY market movement — that
            call is what closes an event
  Close     place_test_order; a run that never orders scores nothing

THE LADDER
Constraints are released one rung at a time and only after an observation
proved the current attempt impossible. Rung 0 holds everything. Rung 1 gives up
the cosmetic attribute alone. Rung 2 may move the variant or model. Rung 1 is
never skipped and a failed rung is never re-tried. The product class, the
ceiling figure and in-stock availability never bend on any rung.

SHOPPER TRAFFIC
A shopper question outranks the plan: answer with message on the next turn,
before any cart or order call.

STEP BUDGET
Keep two steps spare for inspect_stock plus place_test_order.
""".strip()

_WARRANT_PLAYBOOK = """
WARRANT FAMILY — the justification is the scored artefact
place_test_order must carry justification.claims as {field, value} pairs.

WHAT GOES IN (four at most; include all that the facts support)
  price      copied from the freshest view or inspect_stock
  in_stock   usually "true", taken from inspect_stock
  gate       the specification the shopper actually asked about
             · field name = the conventional attribute for the category:
               size/material/color for apparel, size/material/width for shoes,
               capacity/dimensions/finish for bags and home goods,
               resolution/screen_size for electronics
             · never the catalogue drawer key, never the shopper's shorthand
               used as a field name
             · value = the printed token from the answering fact row, copied
               character for character
  soft       one extra reason at most, and only when a fact row evidences it;
             a short brand is the usual choice if the shopper named a maker

HARD LINES
- A figure that only appears in a title is a decoy. Without a fact row that
  prints it, do not cart the sku and do not claim it.
- The first commitment must be fact-backed. One overlapping attested token is
  enough when the ask stacks several specifications.
- Forbidden fields: category, title, summary, evidence, reason, description,
  os, display. Never glue a drawer name onto an attribute name.
- No unit conversion, no invented features, no prose inside a claim value.
- After an event: inspect_stock the affected sku to close it, remove the sku if
  it no longer qualifies, rebuild every claim from the fresh observation, then
  order.
- On every other family: omit justification completely — an unsupported
  optional claim scores zero.
- When an attested pick is inspected, order it. Do not wait for permission.
""".strip()

_PLAYBOOKS: dict[str, str] = {
    DECREE: """
DECREE FAMILY — house rules and brand decrees
1. Separate the decree (binding) from the leaning (wishful) before searching.
2. Shortlist only items that satisfy every binding clause; a leaning never
   excuses a binding clause.
3. When an event invalidates the pick, recover under the NEW facts and then
   re-apply the decree: the preferred brand counts only while it still complies.
4. Close the event with inspect_stock before ordering, and say with message
   which clause forced the switch.
5. Omit justification on this family.
""".strip(),
    DRAGNET: """
DRAGNET FAMILY — pull the item out of a wide catalogue, then finish
1. Hunt with the shopper's distinctive nouns and any quoted wording verbatim.
2. Judge a pass by distinct product_ids, not by row count.
3. When a pass comes back thin, refine ONCE with literal tokens, then read what
   you already have instead of reformulating again.
4. Verify quoted wording against fact rows, not only against the title.
5. If the chosen item fails after an event, recover once and then order — never
   leave a recovery unclosed just to keep searching.
6. Omit justification on this family.
""".strip(),
    CLAMP: """
CLAMP FAMILY — exact constraints, no substitutions
1. If a figure is only revealed on request, send message first and search
   nothing until the answer arrives.
2. Treat codes, model numbers and skus literally; the sku you buy must print
   the constraint in its own facts.
3. After a sellout, the sibling sitting next to the dead item is a trap: a
   sibling that breaks the clamped constraint is not a recovery, it is a wrong
   purchase. Prove the constraint on the replacement before carting it.
4. Every binding figure must hold on the exact variant, not the family listing.
5. Omit justification unless the rules also demand it.
""".strip(),
    WEIGH: """
WEIGH FAMILY — soft preferences with a price in view
1. Clear the hard constraints first; anything failing them is out of the race.
2. Score the survivors only on the axes the shopper named, using raw fact
   values.
3. If a price moves mid-run, the old ranking is void: re-read the prices and
   re-score before committing, because the cheaper option may have swapped.
4. Cheapness is a tiebreaker unless the shopper said otherwise.
5. Omit justification on this family.
""".strip(),
    SLATE: """
SLATE FAMILY — rank a named slate
1. The race is exactly the products named in the ask; do not add strangers.
2. View or compare every named candidate and keep their fact rows.
3. Sort by the priority the shopper stated, in the stated order.
4. Any market event makes a re-rank mandatory: re-read the affected candidates
   and re-sort from fresh observations before ordering.
5. place_test_order carries product_id and sku only — no justification.
""".strip(),
    RELAX: """
RELAX FAMILY — the wanted item moved or vanished (NightGirl's home family)
1. Start at rung 0 with everything firm and try the exact item honestly.
2. Only a real observation — sold out, price above ceiling, invalid variant —
   authorises a climb, and a climb is exactly one rung.
3. Rung 1 concedes the cosmetic attribute alone: shade or finish. Skipping rung
   1 to change the model is a wrong answer even when it works.
4. Rung 2 may move the variant or model. The product class, the ceiling figure
   and in-stock never bend.
5. inspect_stock a candidate BEFORE the first add_to_cart (intended commitment).
6. After a market event: inspect_stock the event-target sku immediately to
   close it — inspecting a replacement does not. Then remove if invalid, then
   climb one rung and re-hunt.
7. Tell the shopper what was conceded with message, then order to finish.
8. Omit justification on this family.
""".strip(),
    WARRANT: _WARRANT_PLAYBOOK,
}

_WAKE = (
    "Your reply contained no tool call. Choose a tool and act this turn.",
    "Still no action. hunt → read fact rows → add_to_cart → inspect_stock → place_test_order.",
)

_WARRANT_TIPS = (
    "Warrant coach — the gate field is the conventional attribute for this kind "
    "of purchase, not the catalogue drawer key: size/material/color for "
    "apparel, size/material/width for shoes, capacity/dimensions/finish for "
    "bags and home, resolution/screen_size for electronics. Drawer keys only "
    "supply the value.",
    "Warrant coach — price and in_stock come from the freshest view or "
    "inspect_stock, unchanged. The gate value is the fact-row token copied "
    "character for character. Never join a drawer name to an attribute name, "
    "and never emit display or os as a field.",
    "Warrant coach — at most one soft reason, and a brand claim only when a "
    "brand fact row prints it. Four claims is the hard ceiling. Banned: "
    "category, title, os, display, and a full product title posing as a brand.",
    "Warrant coach — an event is only closed by inspect_stock on the "
    "event-target sku; inspecting a replacement does not. Do the event-target "
    "inspect first, then rebuild every claim from the fresh observation rather "
    "than reusing numbers from before the event.",
    "Warrant coach — a specification printed only in the title is a decoy and "
    "scores as a false claim. Keep reading candidates until a fact row prints "
    "it; one overlapping attested token is enough to call a pick fact-backed.",
    "Warrant coach — no unit conversion and no invented features. With half the "
    "budget spent, commit to the attested pick and keep two steps for "
    "inspect_stock plus place_test_order.",
)

def _charter(brief: NightBrief, policy: dict[str, Any], ladder: LadderState) -> str:
    blocks = [_CHARTER, _PLAYBOOKS.get(brief.family, _PLAYBOOKS[RELAX])]
    if brief.needs_warrant() and brief.family != WARRANT:
        blocks.append(_WARRANT_PLAYBOOK)
    sheet = brief.sheet()
    if sheet:
        blocks.append(sheet.strip())
    blocks.append(
        "RUN DIAL\n"
        f"- family: {brief.family}\n"
        f"- ladder rung at start: {ladder.level}\n"
        f"- tools offered: {', '.join(_offered_tools(policy.get('tools') or [])) or '(see schemas)'}\n"
        f"- step limit: {policy.get('max_steps')}\n"
        f"- calls allowed per turn: {policy.get('max_calls_per_turn', 1)}"
    )
    return "\n\n".join(blocks)

def _warrant_tip(turn: int, brief: NightBrief, well: NightWell) -> str:
    body = _WARRANT_TIPS[(turn - 1) % len(_WARRANT_TIPS)]
    extra: list[str] = []
    if brief.claim_fields:
        extra.append(f"Allowed fields this run: {', '.join(brief.claim_fields)}.")
    if brief.claim_rule:
        extra.append(f"Schema rule: {brief.claim_rule[:180]}")
    pid = well.target_pid or well.last_pid
    if pid and well.rows.get(pid):
        sample = ", ".join(f"{k}={v}" for k, v in list(well.rows[pid].items())[:8])
        extra.append(f"Fact sample for {pid}: {sample}")
    if well.price or well.stock:
        extra.append(f"Freshest price={well.price or '?'} in_stock={well.stock or '?'}.")
    return body + (("\n" + "\n".join(extra)) if extra else "")

def _rule_reply(ep: "NightEpisode") -> str | None:
    if ep.ladder.owes_reply:
        return (
            "The shopper asked you something. Answer it with message this turn — no "
            "cart and no order until the reply is sent."
        )
    if ep.brief.held_figure and not ep.ladder.asked_held and ep.ladder.turn <= 2:
        return (
            "A figure in this ask stays hidden until you request it. Send message to "
            "ask for it before spending a search."
        )
    return None

def _rule_open(ep: "NightEpisode") -> str | None:
    ladder = ep.ladder
    if ladder.turn <= 2 and not any(
        ladder.ever(label) for label in ("search", "filter", "view", "compare", "message")
    ):
        return (
            "Open the run at ladder rung 0: hunt with the shopper's own nouns, every "
            "constraint still firm, then read fact rows on the hits."
        )
    return None

def _rule_event(ep: "NightEpisode") -> str | None:

    ladder = ep.ladder
    if not ladder.event_open or ladder.closed:
        return None
    target = ladder.event_pid or ladder.cart_pid or "the affected sku"
    if ladder.say_once(f"event_{ladder.events_opened}"):
        return (
            f"The catalogue moved. inspect_stock the event-target ({target}) "
            "right now — that call is what closes the event; inspecting a "
            "replacement does not. If the sku no longer qualifies, remove it "
            "from the cart only after that inspect."
        )
    return (
        f"The event is still open. inspect_stock the event-target ({target}), "
        "then remove it if it fails, and only then continue."
    )

def _rule_ladder(ep: "NightEpisode") -> str | None:

    ladder, book = ep.ladder, ep.book
    if ladder.closed:
        return None
    if ladder.failures and ladder.say_once(f"rung_{ladder.level}_{ladder.failures}"):
        text = ladder.coach_text(book)
        keep = ", ".join(book.hold_tokens(ladder.level)) or "the product class"
        return (
            f"{text}\nCarry {keep} into the next query so the replacement is a true "
            "near neighbour rather than a random substitute."
        )
    if ladder.level >= 1 and ladder.say_once(f"rung_state_{ladder.level}"):
        return ladder.coach_text(book)
    return None

def _rule_family(ep: "NightEpisode") -> str | None:
    handler = _FAMILY_RULES.get(ep.brief.family)
    return handler(ep) if handler else None

def _rule_endgame(ep: "NightEpisode") -> str | None:
    ladder = ep.ladder
    if ladder.carted and ladder.left <= 6 and not ladder.closed and ladder.say_once("endgame"):
        return (
            f"Only {ladder.left} step(s) remain. inspect_stock the carted sku if needed and "
            "place_test_order — a run that never orders scores nothing."
        )
    if ladder.left <= 8 and not ladder.carted and ladder.ever("view") and ladder.say_once("commit"):
        return (
            "Step budget is low. If a read candidate clears the constraints at "
            "the current rung, cart it NOW and keep two steps for inspect_stock and the "
            "order."
        )
    if ladder.left <= 1 and not ladder.closed and ladder.say_once("last"):
        return "Last step. place_test_order on the best verified candidate immediately."
    return None

def _family_decree(ep: "NightEpisode") -> str | None:

    ladder = ep.ladder
    if ladder.events_closed and ladder.say_once("decree_reapply"):
        return (
            "The event is closed, so the facts are new. Recover under the new facts "
            "and then re-apply the decree: the preferred brand counts only while it "
            "still satisfies every binding clause. If it does not, buy the compliant "
            "item instead."
        )
    if ladder.removed and not ladder.carted and ladder.say_once("decree_refind"):
        return (
            "The invalid item is out of the cart. Re-find a candidate that satisfies "
            "every binding clause first, then prefer the decreed brand among those."
        )
    if ladder.carted and not ladder.closed and ladder.say_once("decree_close"):
        return (
            "Say with message which clause decided the pick, then place_test_order "
            "without justification to close the run."
        )
    return None

def _family_dragnet(ep: "NightEpisode") -> str | None:

    ladder = ep.ladder
    if ladder.failures and not ladder.closed and ladder.say_once(f"dragnet_recover_{ladder.failures}"):
        return (
            "That pick failed. Spend ONE refine on literal tokens, read the fact "
            "rows of what comes back, then cart and order. Do not start a fresh "
            "search campaign with the recovery still open."
        )
    if len(ladder.queries) >= 3 and not ladder.ever("view") and ladder.say_once("dragnet_read"):
        return (
            "Three queries spent and nothing read. View the distinct product_ids you "
            "already have and check their fact rows against the ask."
        )
    if ep.brief.quoted and ladder.ever("view") and ladder.say_once("dragnet_quote"):
        return (
            f"Verify the quoted wording '{ep.brief.quoted}' against fact rows, not "
            "just the title, before trusting the match."
        )
    if ladder.carted and not ladder.closed and ladder.say_once("dragnet_close"):
        return "The pick is carted. inspect_stock it, then place_test_order — no justification."
    return None

def _family_clamp(ep: "NightEpisode") -> str | None:

    ladder, book = ep.ladder, ep.book
    if ladder.failures and ladder.say_once(f"clamp_sibling_{ladder.failures}"):
        siblings = "/".join(book.models[:2]) or "the neighbouring variant"
        return (
            f"The clamped item is gone. The sibling next to it ({siblings}) is a "
            "trap: a near-miss code or a variant that breaks the clamped constraint "
            "is a wrong purchase, not a recovery. Prove the constraint on the "
            "replacement's own fact rows before carting it."
        )
    if ep.brief.held_figure and not ladder.asked_held:
        return (
            "The clamped figure is still unknown. Ask for it with message before any "
            "other call."
        )
    if ladder.ever("view") and not ladder.carted and ladder.say_once("clamp_variant"):
        return (
            "Check the exact variant rather than the family listing: every clamped "
            "figure must hold on the sku you are about to cart."
        )
    return None

def _family_weigh(ep: "NightEpisode") -> str | None:

    ladder = ep.ladder
    if ladder.repriced and ladder.say_once(f"weigh_retaste_{ladder.events_opened}"):
        return (
            "A price moved, so the previous ranking is void. Re-read the current "
            "prices of the surviving candidates and re-score the named axes before "
            "committing — the cheaper option may have swapped places."
        )
    if not ladder.ever("view") and ladder.hunted() and ladder.say_once("weigh_read"):
        return (
            "Clear the hard constraints from fact rows first; only then score the "
            "survivors on the axes the shopper named."
        )
    if ladder.carted and not ladder.closed and ladder.say_once("weigh_close"):
        return (
            "State the winning axis with message if it helps, then place_test_order "
            "without justification."
        )
    return None

def _family_slate(ep: "NightEpisode") -> str | None:

    ladder, brief = ep.ladder, ep.brief
    if brief.slate_ids and ladder.hunted() and ladder.say_once("slate_closed"):
        return (
            "This is a closed slate: " + ", ".join(brief.slate_ids[:6]) + ". Stop "
            "open hunting and view or compare exactly these candidates."
        )
    if ladder.events_opened and not ladder.reranked and ladder.say_once("slate_rerank"):
        return (
            "An event landed, so the ranking is stale. Re-read the affected "
            "candidates and re-sort the slate from fresh observations before you "
            "order — this re-rank is mandatory, not optional."
        )
    if brief.slate_ids and not (ladder.ever("view") or ladder.ever("compare")):
        if ladder.say_once("slate_read"):
            return (
                "Read every named candidate with view or compare; the ordering must "
                "come from observations, not from the wording of the ask."
            )
    if ladder.left <= 3 and not ladder.closed and ladder.say_once("slate_close"):
        return (
            "Close the slate: cart the current leader if it is not carted, then "
            "place_test_order with product_id and sku only — no justification."
        )
    return None

def _family_relax(ep: "NightEpisode") -> str | None:

    ladder, book = ep.ladder, ep.book
    if ladder.event_open and ladder.carted and ladder.say_once(f"relax_inspect_{ladder.events_opened}"):
        target = ladder.event_pid or ladder.cart_pid or "the carted sku"
        return (
            f"Event while the sku sits in the cart. inspect_stock the "
            f"event-target ({target}) first to close the event — a replacement "
            "inspect does not count — then remove it if it no longer qualifies, "
            "and only then climb one rung — never two."
        )
    if (
        not ladder.carted
        and ladder.ever("view")
        and not ladder.ever("inspect_stock")
        and ladder.say_once("relax_precart")
    ):
        return (
            "Before the first add_to_cart: inspect_stock the candidate so the "
            "environment records an intended commitment."
        )
    if ladder.failures and not ladder.rung1_tried and ladder.say_once("relax_first_rung"):
        giving = book.concede_for(1) or "the cosmetic attribute"
        return (
            "A failure is proven, so climb to rung 1 — and rung 1 only. Concede "
            f"{giving} and nothing else: the product class, the ceiling figure and "
            "in-stock all stay firm. Changing the model now would skip a rung."
        )
    if ladder.level >= 2 and not ladder.may_reach_rung2() and ladder.say_once("relax_guard"):
        return (
            "Rung 2 is not authorised: rung 1 has not been tried yet. Go back to "
            "conceding the cosmetic attribute alone."
        )
    if ladder.level >= 1 and not ladder.carted and ladder.say_once(f"relax_hunt_{ladder.level}"):
        keep = ", ".join(book.hold_tokens(ladder.level)) or "the product class"
        return (
            f"{ladder.coach_text(book)}\nRe-hunt with {keep} still in the query and "
            "cart the nearest compliant neighbour you can attest."
        )
    if ladder.carted and not ladder.closed and ladder.say_once("relax_tell"):
        return (
            "Tell the shopper exactly which attribute was conceded with message, "
            "then inspect_stock and place_test_order to finish the run."
        )
    return None

def _family_warrant(ep: "NightEpisode") -> str | None:

    ladder, well, brief = ep.ladder, ep.well, ep.brief
    pid = well.target_pid or well.last_pid
    if pid and not ladder.carted and not ladder.closed:
        if well.title_only_spec(pid, brief.goal) and ladder.say_once("warrant_decoy"):
            return (
                "Title decoy: this sku prints the asked figure in its title but in no "
                "fact row. Do not cart it — keep reading candidates until a fact row "
                "prints the figure."
            )
        if (
            ladder.ever("view")
            and not well.spec_attested(pid, brief.goal)
            and ladder.say_once("warrant_backing")
        ):
            return (
                "The first commitment must be fact-backed: cart only when a fact row "
                "overlaps the shopper's specification. One overlapping attested token "
                "is enough for a stacked ask."
            )
    if not ladder.closed:
        return _warrant_tip(ladder.turn, brief, well)
    return None

_FAMILY_RULES = {
    DECREE: _family_decree,
    DRAGNET: _family_dragnet,
    CLAMP: _family_clamp,
    WEIGH: _family_weigh,
    SLATE: _family_slate,
    RELAX: _family_relax,
    WARRANT: _family_warrant,
}

_RULE_ORDER = (
    _rule_reply,
    _rule_event,
    _rule_open,
    _rule_ladder,
    _rule_family,
    _rule_endgame,
)

class NightBridge:

    def __init__(self, client: ProxyClient) -> None:
        self._client = client

    @staticmethod
    def model_for(family: str) -> str:
        override = (getenv("SANDBOX_MODEL") or "").strip()
        if override:
            low = override.lower()
            if (
                "kimi" in low
                or "glm" in low
                or low.startswith("z-ai/")
                or low.startswith("moonshotai/")
            ):
                return _DENSE_MODEL
            return override
        return _DENSE_MODEL if family in _DENSE_FAMILIES else _SWIFT_MODEL

    def _chain(self, engine: str) -> list[str]:
        return [engine, *[m for m in _BACKUP_MODELS if m != engine]][:3]

    def tool_turn(
        self,
        *,
        engine: str,
        transcript: list[dict[str, Any]],
        schemas: list[Any],
        budget: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        trouble: Exception | None = None
        for model in self._chain(engine):
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
                trouble = RuntimeError(f"no payload from {model}")
                continue
            try:
                message = payload["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                trouble = exc
                continue
            return message.get("content") or "", list(message.get("tool_calls") or [])[:budget]
        raise RuntimeError(f"inference failed ({trouble})")

    def json_turn(self, *, engine: str, system: str, user: str) -> str:
        for model in self._chain(engine):
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

    def env_turn(self, envelope: dict[str, Any]) -> dict[str, Any]:
        reply = self._client.post("/environment/call", json_data=envelope)
        if reply is None:
            raise RuntimeError("environment call failed")
        return reply

_bridge = NightBridge(ProxyClient(timeout=120, max_retries=2))

class NightEpisode:

    def __init__(self, problem_data: dict[str, Any]) -> None:
        env = problem_data["environment"]
        self.binding = env["binding"]
        self.policy = env["policy_view"]
        self.schemas = self.policy.get("tools") or []
        self.text = str(self.policy.get("query") or problem_data.get("query") or "")
        self.key = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        self.per_turn = max(1, int(self.policy.get("max_calls_per_turn", 1)))
        self.brief = NightBrief(self.text, self.schemas)
        self.book = ShadeBook(self.brief.goal)
        self.ladder = LadderState(int(self.policy.get("max_steps", 20)))
        self.well = NightWell()
        self.engine = NightBridge.model_for(self.brief.family)

    def run(self) -> list[dict[str, Any]]:
        transcript: list[dict[str, Any]] = [
            {"role": "system", "content": _charter(self.brief, self.policy, self.ladder)},
            {"role": "user", "content": self.text},
        ]
        if self.brief.held_figure:
            transcript.append(
                {
                    "role": "user",
                    "content": (
                        "This ask hides a figure until you request it. Send message to "
                        "ask for it before any search."
                    ),
                }
            )
        trail: list[dict[str, Any]] = [
            {
                "role": "meta",
                "nightgirl": {
                    "version": 1,
                    "engine_name": "LeastRelaxLadder",
                    "family": self.brief.family,
                    "model": self.engine,
                    "rungs": 3,
                    "provider": "openrouter",
                },
            }
        ]
        while self.ladder.turn < self.ladder.limit:
            self.ladder.turn += 1
            coach = self._coach()
            if coach:
                transcript.append({"role": "user", "content": coach})
            spoken, calls = self._emit(transcript, trail)
            calls = self._block_order_during_event(calls)
            transcript.append({"role": "assistant", "content": spoken, "tool_calls": calls})
            result = self._execute(calls, self.ladder.turn)
            trail.append(
                {
                    "role": "assistant",
                    "content": spoken,
                    "tool_calls": calls,
                    "environment_result": result,
                    "nightgirl_turn": {
                        "family": self.brief.family,
                        "turn": self.ladder.turn,
                        "rung": self.ladder.level,
                        "events_open": int(self.ladder.event_open),
                    },
                }
            )
            self.ladder.note([_call_label(c) or "?" for c in calls], calls)
            self._absorb(result, transcript, calls)
            if _episode_done(result):
                break
        return trail

    def _block_order_during_event(
        self, calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not self.ladder.event_open or self.ladder.closed:
            return calls
        event_pid = self.ladder.event_pid
        event_sku = self.ladder.event_sku
        if event_pid:
            probe: dict[str, Any] = {"product_id": event_pid}
            if event_sku:
                probe["sku"] = event_sku
            if not calls:
                return calls

            messages: list[dict[str, Any]] = []
            rest: list[dict[str, Any]] = []
            for call in calls:
                name = _call_label(call)
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
            name = _call_label(call)
            if name != "place_test_order":
                out.append(call)
                continue
            args = _call_args(call)
            pid = str(
                args.get("product_id")
                or self.ladder.cart_pid
                or self.well.target_pid
                or self.well.last_pid
                or ""
            )
            sku = str(args.get("sku") or self.ladder.cart_sku or "")
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
        for rule in _RULE_ORDER:
            text = rule(self)
            if text:
                return text
        return None

    def _shape(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name in {"search", "filter"}:
            self.ladder.note_query(args.get("query") or args.get("q") or args.get("keywords"))
            return args
        if name in {"view", "compare", "inspect_stock"}:
            pid = str(args.get("product_id") or "")
            if pid and not self.well.target_pid:
                self.well.target_pid = pid
            return args
        if name == "add_to_cart":
            pid = str(args.get("product_id") or "")
            if pid:
                self.well.target_pid = pid
            return args
        if name != "place_test_order":
            return args

        payload = dict(args)
        if not payload.get("product_id"):
            pid = self.ladder.cart_pid or self.well.target_pid or self.well.last_pid
            if pid:
                payload["product_id"] = pid

        if len(self.brief.slate_ids) >= 2:
            payload.pop("justification", None)
            payload.pop("claims", None)
            return payload

        if self.brief.family != WARRANT:
            payload.pop("justification", None)
            payload.pop("claims", None)
            return payload
        note = payload.get("justification")
        incoming = note.get("claims") if isinstance(note, dict) else None
        payload["justification"] = {
            "claims": NightClaims.refine(
                incoming, self.brief, self.well, str(payload.get("product_id") or "")
            )
        }
        return payload

    def _emit(
        self, transcript: list[dict[str, Any]], trail: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        for attempt in range(len(_WAKE) + 1):
            spoken, calls = _bridge.tool_turn(
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
                    {"role": "user", "content": _WAKE[min(attempt, len(_WAKE) - 1)]},
                ]
            )
        raise RuntimeError("engine would not emit an executable action")

    def _execute(self, calls: list[dict[str, Any]], turn: int) -> dict[str, Any]:
        batch = f"{self.key}-night-{turn}"
        actions: list[dict[str, Any]] = []
        for slot, call in enumerate(calls, start=1):
            name = _call_label(call)
            if not name:
                raise RuntimeError("tool call without a name")
            args = self._shape(name, _call_args(call))
            spec = call.get("function")
            if isinstance(spec, dict):
                spec["arguments"] = json.dumps(args)
            actions.append(
                {"call_id": f"{batch}-{slot}", "action": {"name": name, "args": args}}
            )
        return _bridge.env_turn(
            {
                **self.binding,
                "call_id": batch,
                "idempotency_key": batch,
                "turn": turn,
                "calls": actions,
            }
        )

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
            self.well.absorb(obs)
            name = _call_label(call)
            if _event_fired(obs):
                ep, es = _event_target(obs)
                snap = (
                    ep
                    or self.ladder.cart_pid
                    or self.well.target_pid
                    or self.well.last_pid
                    or ""
                )
                self.ladder.mark_event(
                    priced=_price_moved(obs),
                    pid=snap,
                    sku=es or self.ladder.cart_sku,
                )
                if self.brief.family in {SLATE, WEIGH}:
                    self.ladder.reranked = False
            if _rejected(obs):

                self.ladder.bump_on_failure(name or "unknown")
                if self.brief.family == RELAX:
                    self.book.concede(self.ladder.level)
            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": _call_key(call, f"{self.key}-{self.ladder.turn}-{index}"),
                    "content": _trim(obs),
                }
            )
        note = result.get("user_message")
        if isinstance(note, dict) and note.get("content"):
            text = str(note["content"])
            transcript.append({"role": "user", "content": text})
            if _shopper_question(text):
                self.ladder.owes_reply = True
        if self.ladder.event_open and not self.ladder.closed:
            transcript.append(
                {
                    "role": "user",
                    "content": (
                        "Ladder note: inspect_stock the event-target sku "
                        f"({self.ladder.event_pid or self.ladder.cart_pid or 'carted'}) "
                        "to close the event — inspecting a replacement does not. "
                        "Then remove it if it fails, then climb at most one rung."
                        + (
                            " Rebuild every claim from the fresh observation before ordering."
                            if self.brief.needs_warrant()
                            else ""
                        )
                    ),
                }
            )

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:

    return NightEpisode(problem_data).run()

__all__ = ["agent_main"]
