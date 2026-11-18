

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_ENGINE_FAST = "deepseek/deepseek-v3.2"
_ENGINE_HEAVY = "qwen/qwen3.5-397b-a17b"
_ENGINE_RELIEF = (
    "google/gemma-4-31b-it",
    "openai/gpt-oss-120b",
    "deepseek/deepseek-chat-v3.1",
)

FORK, SCOUT, CIPHER, SAVOR, TIER, RESCUE, VOW = (
    "fork",
    "scout",
    "cipher",
    "savor",
    "tier",
    "rescue",
    "vow",
)
_TRACKS = (FORK, SCOUT, CIPHER, SAVOR, TIER, RESCUE, VOW)

_TRACK_WEIGHT = {TIER: 7, RESCUE: 6, VOW: 5, CIPHER: 4, FORK: 3, SAVOR: 2, SCOUT: 1}

_HEAVY_TRACKS = frozenset({CIPHER, SAVOR, TIER, RESCUE, VOW, FORK})

_ASK_CUES: dict[str, tuple[tuple[str, int], ...]] = {
    TIER: (
        ("rank", 4),
        ("order them", 4),
        ("break the tie", 4),
        ("priority", 3),
        ("comes first", 3),
        ("between these", 3),
        ("tie", 2),
    ),
    RESCUE: (
        ("no longer", 4),
        ("out of stock", 4),
        ("sold out", 4),
        ("vanish", 3),
        ("disappear", 3),
        ("closest", 3),
        ("substitute", 3),
        ("instead", 2),
    ),
    VOW: (
        ("justif", 4),
        ("back it up", 4),
        ("why you", 4),
        ("why did", 3),
        ("evidence", 3),
        ("explain", 3),
        ("reason", 2),
    ),
    CIPHER: (
        ("won't tell", 4),
        ("ask me", 4),
        ("if you ask", 4),
        ("model number", 3),
        ("exact code", 3),
        ("sku", 2),
        ("exactly", 2),
    ),
    FORK: (
        ("non-negotiable", 4),
        ("nice to have", 4),
        ("ideally", 3),
        ("would like", 3),
        ("hard requirement", 3),
        ("prefer", 2),
        ("but", 1),
    ),
    SAVOR: (
        ("care more", 4),
        ("rather have", 4),
        ("matters more", 4),
        ("prefer the cheaper", 4),
        ("would pay", 3),
        ("cheapest", 3),
        ("trade", 2),
        ("lean toward", 2),
        ("worth it", 2),
    ),
    SCOUT: (
        ("described as", 3),
        ("labeled", 3),
        ("mentions", 3),
        ("listing says", 3),
        ("anything that", 2),
        ("phrase", 2),
    ),
}

_RULE_CUES: dict[str, tuple[tuple[str, int], ...]] = {
    TIER: (("rank", 4), ("ordering", 3), ("tradeoff", 2)),
    RESCUE: (("relax", 4), ("unavailable", 3), ("concede", 2)),
    VOW: (("justification", 5), ("claims", 4), ("structured", 3)),
    CIPHER: (("exact match", 4), ("literal", 3), ("ask the shopper", 3)),
    FORK: (("binding", 4), ("soft preference", 3), ("firm", 2)),
    SAVOR: (("preference", 3), ("tiebreak", 3), ("cheaper", 2)),
    SCOUT: (("reformulate", 5), ("recall", 4), ("shortlist", 3), ("query", 2)),
}

_FIELD_RX = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_NAMED_SET_RX = re.compile(r"\b\d{6,}::[\w-]+(?:\.[\w-]+)*")
_CODE_RX = re.compile(r"\b[A-Za-z]{1,8}[-_ ]?\d{2,}[A-Za-z0-9-]*\b")
_QUOTED_RX = re.compile(r"[\"\u201c]([^\"\u201d]{3,70})[\"\u201d]")
_CAP_RX = re.compile(r"\b(?:at or (?:under|below)|under|below|up to|no more than|"
                     r"max(?:imum)?(?:\s+of)?|within)\s*([\d][\d,]*(?:\.\d+)?)", re.I)
_WANTS_WHY_RX = re.compile(r"\bwhy\b|\bjustif|\breasons?\b|\bback it up\b|\bevidence\b|\bexplain\b|\bdefend\b|\bwalk me through\b|\bgive me a reason\b", re.I)
_WITHHELD_RX = re.compile(
    r"\b(?:if|when|unless)\s+you\s+ask\b|\bask\s+me\b|"
    r"\bwon'?t\s+(?:say|tell|share)\b|\bnot\s+telling\b",
    re.I,
)
_MEASURE_RX = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s?(?:inch|in|hz|p|k|gb|tb|mm|cm|mp|w|mah|oz|ml|"
    r"l|kg|lb|ft|yd|\"))?\b",
    re.I,
)
_WORD_RX = re.compile(r"[a-z][a-z0-9-]{1,}")

_CORE_CLAIM = frozenset({"price", "in_stock"})

_DENY_FIELDS = frozenset(
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

_ATTR_WORDS = frozenset(
    {
        "xs", "s", "m", "l", "xl", "xxl", "small", "medium", "large",
        "cotton", "polyester", "wool", "silk", "linen", "leather", "suede",
        "denim", "canvas", "nylon", "plastic", "metal", "wood", "glass",
        "navy", "black", "white", "red", "blue", "green", "yellow", "gray",
        "grey", "brown", "beige", "khaki", "narrow", "wide", "regular",
        "slim", "matte", "glossy", "gloss", "oak", "walnut", "chrome",
        "brushed", "hd", "uhd", "qhd", "fhd", "oled", "lcd", "led",
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

_SCOUT_STOP = frozenset(
    {
        "a", "an", "the", "and", "or", "for", "with", "from", "that", "this",
        "these", "those", "find", "buy", "get", "need", "want", "wants",
        "looking", "look", "please", "help", "me", "my", "i", "you", "your",
        "we", "it", "its", "them", "they", "something", "anything", "some",
        "any", "one", "ones", "product", "products", "item", "items", "thing",
        "things", "shopper", "order", "cart", "checkout", "store", "shop",
        "exactly", "must", "should", "would", "could", "also", "only", "than",
        "then", "when", "which", "what", "who", "have", "has", "had", "are",
        "is", "be", "been", "was", "were", "to", "of", "on", "at", "by", "as",
        "if", "not", "no", "yes", "true", "false", "very", "really", "much",
        "more", "most", "less", "least", "about", "around", "over", "into",
        "out", "up", "down", "in", "so", "but", "because", "while", "there",
        "here", "now", "still", "just", "like", "prefer", "prefers", "why",
        "reason", "justify", "explain", "brand", "size", "color", "colour",
        "material", "width", "capacity", "dimensions", "finish", "resolution",
        "screen", "screen_size", "type", "kind", "sort",
    }
)

def _norm(raw: Any) -> str:

    text = str(raw or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_").replace("/", "_")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "", text)).strip("_")

def _tight(raw: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(raw or "").lower())

def _evidenced(value: Any, blob: Any) -> bool:

    needle, hay = _tight(value), _tight(blob)
    return bool(needle) and needle in hay

def _word_hit(token: str, text: str) -> bool:

    tok = str(token or "").strip().lower()
    blob = str(text or "").lower()
    if not tok or not blob:
        return False
    if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", blob):
        return True
    packed, hay = _tight(tok), _tight(blob)
    if not packed:
        return False
    if len(packed) <= 3:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(packed)}(?![a-z0-9])", hay))
    return packed in hay

def _spec_tokens(goal: str) -> list[str]:

    low = _CAP_RX.sub(" ", (goal or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for hit in _MEASURE_RX.finditer(low):
        tok = hit.group(0).strip()
        key = _tight(tok)
        if key and key not in seen:
            seen.add(key)
            out.append(tok)
    for word in _WORD_RX.findall(low):
        if word in _SCOUT_STOP or word not in _ATTR_WORDS:
            continue
        key = _tight(word)
        if key and key not in seen:
            seen.add(key)
            out.append(word)
    return out

def _literal_tokens(goal: str) -> list[str]:

    low = _CAP_RX.sub(" ", (goal or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[a-z0-9][a-z0-9./\"-]{1,}", low):
        tok = raw.strip("-./\"")
        if len(tok) < 2:
            continue
        if tok in _SCOUT_STOP or tok in _FILLER_ADJ or tok in _MONEY_WORDS:
            continue
        if tok.isdigit() and len(tok) > 4:
            continue
        key = _tight(tok)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(tok)
    return out

def _drawer_concat(field: str) -> bool:

    flat = _norm(field)
    if not flat:
        return False
    for drawer in ("display", "facts", "options", "os", "size", "title"):
        if flat.startswith(drawer + "_") or flat.endswith("_" + drawer):
            return True
    return False

def _json_obj(text: str) -> dict[str, Any]:
    body = (text or "").strip()
    if not body:
        return {}
    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start, end = body.find("{"), body.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(body[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}

def _split_brief(brief: str) -> tuple[str, str]:
    head, sep, tail = (brief or "").partition("\n\n")
    if not sep:
        return (brief or "").strip(), ""
    body = tail.strip()
    first, _, rest = body.partition("\n")
    if first.rstrip().endswith(":") and len(first) < 40:
        body = rest.strip()
    return head.strip(), body

def _uniq(items: Any) -> list[str]:
    return list(dict.fromkeys(str(i) for i in items))

def _schema_claim_enum(tools: list[Any]) -> list[str]:

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
        enum = field.get("enum") if isinstance(field, dict) else None
        if not isinstance(enum, list) or not enum:
            continue
        out, seen = [], set()
        for raw in enum:
            flat = _norm(raw)
            if flat and flat not in seen:
                seen.add(flat)
                out.append(flat)
        return out
    return []

def _schema_claim_rule(tools: list[Any]) -> str:
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
            return str(field.get("description") or "")[:600]
    return ""

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

class ScoutBrief:

    __slots__ = (
        "raw",
        "goal",
        "rules",
        "track",
        "ceiling",
        "named_set",
        "codes",
        "phrase",
        "wants_why",
        "withheld",
        "claim_enum",
        "claim_rule",
        "spec_tokens",
    )

    def __init__(self, raw: str, tools: list[Any] | None = None) -> None:
        self.raw = raw or ""
        self.goal, self.rules = _split_brief(self.raw)
        self.named_set = _uniq(_NAMED_SET_RX.findall(self.raw))
        self.codes = [c for c in _uniq(_CODE_RX.findall(self.goal)) if not c.isdigit()][:6]
        cap = _CAP_RX.search(self.goal) or _CAP_RX.search(self.raw)
        self.ceiling = cap.group(1) if cap else ""
        quoted = _QUOTED_RX.search(self.goal)
        self.phrase = quoted.group(1) if quoted else ""

        self.wants_why = bool(_WANTS_WHY_RX.search(self.goal))
        self.withheld = bool(_WITHHELD_RX.search(self.goal))
        self.claim_enum = _schema_claim_enum(tools or [])
        self.claim_rule = _schema_claim_rule(tools or [])
        self.spec_tokens = _spec_tokens(self.goal)
        self.track = self._pick_track()

    def _pick_track(self) -> str:

        if len(self.named_set) >= 2:
            return TIER

        if self.wants_why:
            return VOW
        score = dict.fromkeys(_TRACKS, 0)
        ask, rules = self.goal.lower(), self.rules.lower()
        for track, cues in _ASK_CUES.items():
            score[track] += sum(weight for cue, weight in cues if cue in ask)
        for track, cues in _RULE_CUES.items():
            if track == VOW:
                continue
            score[track] += sum(weight for cue, weight in cues if cue in rules)
        if self.withheld:
            score[CIPHER] += 7
        if self.wants_why:
            score[VOW] += 6
        if self.phrase:
            score[SCOUT] += 5
        if self.codes and not self.named_set:
            score[CIPHER] += 3
        if "ideally" in ask and ("must" in ask or "non-negotiable" in ask):
            score[FORK] += 5
        if self.ceiling and ("rather" in ask or "care more" in ask):
            score[SAVOR] += 3

        score[SCOUT] += 2
        top = max(score.values())
        if top <= 0:
            return SCOUT
        return max(
            [t for t, s in score.items() if s == top], key=lambda t: _TRACK_WEIGHT[t]
        )

    def needs_claims(self) -> bool:

        return self.track == VOW

    def closed_set(self) -> bool:
        return len(self.named_set) >= 2

    def sheet(self) -> str:
        rows: list[str] = []
        if self.ceiling:
            rows.append(f"- ceiling figure read from the ask: {self.ceiling}")
        if self.phrase:
            rows.append(f"- quoted listing phrase to verify: {self.phrase}")
        if self.codes:
            rows.append(f"- literal codes typed by the shopper: {', '.join(self.codes)}")
        if self.named_set:
            rows.append(f"- closed candidate set: {', '.join(self.named_set)}")
        if self.spec_tokens:
            rows.append(f"- specification tokens to attest: {', '.join(self.spec_tokens)}")
        if self.withheld:
            rows.append("- a figure is withheld until you ASK for it")
        if self.claim_enum:
            rows.append(f"- live claim-field enum: {', '.join(self.claim_enum)}")
        if self.claim_rule:
            rows.append(f"- live claim rule: {self.claim_rule[:280]}")
        return ("SCOUT SHEET\n" + "\n".join(rows) + "\n") if rows else ""

class IdentityBag:

    __slots__ = ("listed", "viewed", "titles", "last_batch", "thin_batches")

    def __init__(self) -> None:
        self.listed: list[str] = []
        self.viewed: list[str] = []
        self.titles: dict[str, str] = {}
        self.last_batch = 0
        self.thin_batches = 0

    def absorb(self, obs: Any) -> int:

        found: list[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                pid = str(node.get("product_id") or "").strip()
                if pid:
                    found.append(pid)
                    title = str(node.get("title") or "").strip()
                    if title:
                        self.titles.setdefault(pid, title)
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        walk(value)
            elif isinstance(node, list):
                for item in node[:40]:
                    walk(item)

        walk(obs)
        fresh = list(dict.fromkeys(found))
        for pid in fresh:
            if pid not in self.listed:
                self.listed.append(pid)
        self.last_batch = len(fresh)
        if len(fresh) <= 1:
            self.thin_batches += 1
        return len(fresh)

    def mark_viewed(self, pid: str) -> None:
        pid = str(pid or "").strip()
        if pid and pid not in self.viewed:
            self.viewed.append(pid)

    @property
    def identities(self) -> int:
        return len(self.listed)

    @property
    def seen_enough(self) -> bool:

        if len(self.viewed) >= 2:
            return True
        return bool(self.viewed) and self.identities <= len(self.viewed)

    def thin(self) -> bool:
        return self.identities < 2 or self.last_batch <= 1

    def blurb(self) -> str:
        return (
            f"identities listed={self.identities} viewed={len(self.viewed)} "
            f"last batch={self.last_batch}"
        )

class RefineKit:

    __slots__ = ("queries", "proposed")

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.proposed: list[str] = []

    def remember(self, text: Any) -> None:
        query = str(text or "").strip()
        if query and query.lower() not in {q.lower() for q in self.queries}:
            self.queries.append(query[:160])

    def spent(self) -> int:
        return len(self.queries)

    def tried(self, candidate: str) -> bool:
        packed = _tight(candidate)
        return bool(packed) and any(_tight(q) == packed for q in self.queries)

    def propose(self, brief: ScoutBrief) -> str:

        tokens = _literal_tokens(brief.goal)
        if brief.phrase:
            phrase_bits = [
                t
                for t in _literal_tokens(brief.phrase)
                if t not in _FILLER_ADJ and t not in _MONEY_WORDS
            ]
            tokens = _uniq(phrase_bits + tokens)
        if not tokens:
            return ""
        last = self.queries[-1].lower() if self.queries else ""
        unspent = [t for t in tokens if not _word_hit(t, last)]
        pool = unspent or tokens
        for width in (3, 4, 2, 5):
            candidate = " ".join(pool[:width])
            if candidate and not self.tried(candidate) and candidate not in self.proposed:
                self.proposed.append(candidate)
                return candidate
        rotated = " ".join(pool[1:4]) or " ".join(pool[:2])
        if rotated:
            self.proposed.append(rotated)
        return rotated

    def sticky(self, brief: ScoutBrief) -> list[str]:

        tokens = _literal_tokens(brief.goal)
        ranked = sorted(
            {t for t in tokens if len(t) >= 4 or t in _ATTR_WORDS},
            key=lambda t: (-len(t), t),
        )
        keep = [t for t in ranked if t not in _ATTR_WORDS][:2]
        keep += [t for t in ranked if t in _ATTR_WORDS][:1]
        return keep[:3]

class ScoutWell:

    __slots__ = ("rows", "listing", "titles", "last_pid", "focus_pid", "price", "stock")

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, str]] = {}
        self.listing: dict[str, str] = {}
        self.titles: dict[str, str] = {}
        self.last_pid = ""
        self.focus_pid = ""
        self.price = ""
        self.stock = ""

    def put(self, pid: str, key: Any, value: Any) -> None:
        flat = _norm(key)
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

        def walk(node: Any, inherited: str = "") -> None:
            if not isinstance(node, dict):
                return
            pid = str(node.get("product_id") or "") or inherited
            title = str(node.get("title") or "")
            if pid and title:
                self.titles[pid] = title
            bits: list[str] = []
            if title:
                bits.append(title)
            if node.get("options"):
                bits.append(str(node["options"]))
            for key in ("price", "in_stock"):
                if key in node:
                    self.put(pid, key, node.get(key))
            if node.get("brand"):
                self.put(pid, "brand", node.get("brand"))
            drawer = node.get("facts")
            if isinstance(drawer, dict):
                self._flatten(drawer, pid, bits)
            if pid and bits:
                prior = self.listing.get(pid, "")
                chunk = " | ".join(bits)
                if chunk not in prior:
                    self.listing[pid] = (prior + " | " + chunk if prior else chunk)[:2200]
            for key in (
                "observation",
                "product",
                "compared",
                "variants",
                "results",
                "items",
                "candidates",
            ):
                child = node.get(key)
                if isinstance(child, dict):
                    walk(child, pid)
                elif isinstance(child, list):
                    for item in child[:12]:
                        walk(item, pid)

        walk(obs)

    def _flatten(
        self, block: dict[str, Any], pid: str, bits: list[str], prefix: str = ""
    ) -> None:
        for key, value in block.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                packed = json.dumps(value, ensure_ascii=False)[:160]
                bits.append(f"{path}={packed}")
                self.put(pid, key, packed)
                self._flatten(value, pid, bits, path)
            elif isinstance(value, list):
                text = ", ".join(str(x) for x in value[:12])
                bits.append(f"{path}={text}")
                self.put(pid, key, text)
            else:
                text = str(value).strip()
                if not text:
                    continue
                bits.append(f"{path}={text}")
                self.put(pid, key, text)
                if prefix:
                    self.put(pid, path.rsplit(".", 1)[-1], text)

    def fact_rows(self, pid: str) -> list[dict[str, str]]:
        return [{"key": k, "value": v} for k, v in self.rows.get(pid, {}).items()]

    def evidence(self, pid: str) -> str:
        observed = self.rows.get(pid, {})
        return self.listing.get(pid, "") + " | " + " | ".join(
            f"{k}={v}" for k, v in observed.items()
        )

    def title_trap(self, pid: str, goal: str) -> bool:

        title = self.titles.get(pid, "")
        if not title or not goal:
            return False
        blob = " ".join(f"{k} {v}" for k, v in self.rows.get(pid, {}).items())
        for token in _spec_tokens(goal):
            if _word_hit(token, title) and not _word_hit(token, blob):
                return True
        return False

    def fact_supported(self, pid: str, goal: str) -> bool:
        if not pid or pid not in self.rows:
            return False
        blob = " ".join(f"{k} {v}" for k, v in self.rows.get(pid, {}).items())
        tokens = _spec_tokens(goal)
        if not tokens:
            return any(k not in _CORE_CLAIM for k in self.rows.get(pid, {}))
        return any(_word_hit(token, blob) for token in tokens)

_VOW_FILL = """
Name the justification claim that covers ONE stated shopping requirement.
Input JSON keys: goal, allowed_fields (live schema enum; may be empty),
schema_rule (the tool's own wording), have (fields already claimed), listing
(observed text for this exact sku), fact_rows (key/value pairs — catalog drawer
keys are VALUE evidence only; drawer names differ per category).

Reply with JSON only: {"claims":[{"field":"...","value":"..."}]}

Constraints:
- Cover the specification the shopper stated that `have` does not cover yet.
- Respect schema_rule. If allowed_fields is non-empty the field MUST be one of
  them. If it is empty, use the conventional snake_case attribute name for that
  kind of ask (apparel: size/material/color; shoes: size/material/width;
  bags & home: capacity/dimensions/finish; electronics: resolution/screen_size)
  — never the catalog drawer key.
- value MUST be the printed token from listing/fact_rows, copied verbatim.
  Shopper shorthand (large, cotton, navy, 720p, 27-inch, …) still belongs to
  that attribute: claim the printed fact token, not the shorthand.
- Never build a field by joining drawer+attribute (display_resolution, …).
- Never emit display, os, category, title, summary, evidence or reason.
- Do not restate price or in_stock.
- One optional short brand only when a brand fact row evidences it.
- If nothing in listing/fact_rows evidences the specification, reply
  {"claims":[]}.
- Two claims maximum.
""".strip()

class ScoutClaims:

    @staticmethod
    def polish(
        raw_claims: Any, brief: ScoutBrief, well: ScoutWell, pid: str
    ) -> list[dict[str, str]]:
        allowed = {_norm(x) for x in brief.claim_enum}
        observed = well.rows.get(pid, {})
        evidence = well.evidence(pid)
        kept: list[dict[str, str]] = []
        used: set[str] = set()

        for entry in raw_claims if isinstance(raw_claims, list) else []:
            if not isinstance(entry, dict):
                continue
            field = _norm(entry.get("field"))
            value = "" if entry.get("value") is None else str(entry.get("value")).strip()
            if not field or not value or field in used:
                continue
            if field in _DENY_FIELDS or _drawer_concat(field):
                continue
            if allowed and field not in allowed:
                continue
            if not _FIELD_RX.match(field):
                continue
            if field == "in_stock":
                if value.lower() in {"true", "1", "yes", "in_stock"}:
                    value = "true"
            elif field == "brand":
                value = ScoutClaims._brand(value, observed, evidence)
                if not value:
                    continue
            elif field not in _CORE_CLAIM and not _evidenced(value, evidence):
                swap = ScoutClaims._nearest_fact(value, observed, brief.goal, evidence)
                if not swap:
                    continue
                value = swap
            used.add(field)
            kept.append({"field": field, "value": value})

        ScoutClaims._fill_core(kept, used, allowed, observed, well)

        if "brand" not in used and (not allowed or "brand" in allowed):
            brand = ScoutClaims._brand(observed.get("brand", ""), observed, evidence)
            if brand and _evidenced(brand, brief.goal):
                kept.append({"field": "brand", "value": brand})
                used.add("brand")

        if brief.needs_claims() and not ScoutClaims._has_gate(kept):
            for row in ScoutClaims._ask_fill(brief, well, pid, used):
                field = _norm(row.get("field"))
                value = str(row.get("value") or "").strip()
                if not field or not value or field in used:
                    continue
                if field in _DENY_FIELDS or _drawer_concat(field):
                    continue
                if allowed and field not in allowed:
                    continue
                if not _FIELD_RX.match(field):
                    continue
                if field == "brand":
                    value = ScoutClaims._brand(value, observed, evidence)
                    if not value:
                        continue
                elif not _evidenced(value, evidence):
                    swap = ScoutClaims._nearest_fact(value, observed, brief.goal, evidence)
                    if not swap:
                        continue
                    value = swap
                used.add(field)
                kept.append({"field": field, "value": value})
                if ScoutClaims._has_gate(kept):
                    break

        if brief.needs_claims() and not ScoutClaims._has_gate(kept):
            gate = ScoutClaims._gate_from_facts(brief.goal, observed, evidence, used, allowed)
            if gate:
                used.add(gate["field"])
                kept.append(gate)

        if (
            brief.needs_claims()
            and "brand" not in used
            and (not allowed or "brand" in allowed)
            and observed.get("brand")
            and len(kept) < 4
        ):
            brand = ScoutClaims._brand(observed.get("brand", ""), observed, evidence)
            if brand:
                kept.append({"field": "brand", "value": brand})
                used.add("brand")

        return kept[:6]

    @staticmethod
    def _fill_core(
        kept: list[dict[str, str]],
        used: set[str],
        allowed: set[str],
        observed: dict[str, str],
        well: ScoutWell,
    ) -> None:

        for need in ("price", "in_stock"):
            if need in used or (allowed and need not in allowed):
                continue
            value = observed.get(need) or (well.price if need == "price" else well.stock)
            if not value:
                continue
            slot = 0 if need == "price" else min(1, len(kept))
            kept.insert(slot, {"field": need, "value": value})
            used.add(need)

    @staticmethod
    def _has_gate(claims: list[dict[str, str]]) -> bool:
        fields = {_norm(c["field"]) for c in claims}
        return bool(fields - _CORE_CLAIM - _DENY_FIELDS - {"brand"})

    @staticmethod
    def _gate_from_facts(
        goal: str,
        rows: dict[str, str],
        evidence: str,
        have: set[str],
        allowed: set[str],
    ) -> dict[str, str] | None:

        goal_l = (goal or "").lower()
        ranked: list[tuple[int, str, str]] = []
        for raw_key, raw_val in rows.items():
            key = _norm(raw_key)
            value = str(raw_val or "").strip()
            if not key or not value or key in have:
                continue
            if key in _CORE_CLAIM or key == "brand" or key in _DENY_FIELDS:
                continue
            field = key
            if field in have or field in _DENY_FIELDS or _drawer_concat(field):
                continue
            if allowed and field not in allowed:
                continue
            if not _FIELD_RX.match(field):
                continue
            if not _evidenced(value, evidence):
                continue
            score = 0
            if field in goal_l or key in goal_l:
                score += 12
            if _evidenced(value, goal):
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
    def _ask_fill(
        brief: ScoutBrief, well: ScoutWell, pid: str, have: set[str]
    ) -> list[dict[str, str]]:
        evidence = well.evidence(pid)
        if not evidence.strip(" |"):
            return []
        enum = [f for f in brief.claim_enum if f not in have]
        payload = {
            "goal": brief.goal[:1500],
            "allowed_fields": enum,
            "schema_rule": brief.claim_rule,
            "have": sorted(have),
            "listing": evidence[:2200],
            "fact_rows": well.fact_rows(pid)[:40],
        }
        reply = _bridge.ask_json(
            engine=_ENGINE_FAST,
            system=_VOW_FILL,
            user=json.dumps(payload, ensure_ascii=False),
        )
        rows = _json_obj(reply).get("claims")
        return [r for r in rows if isinstance(r, dict)][:2] if isinstance(rows, list) else []

    @staticmethod
    def _nearest_fact(
        value: str, observed: dict[str, str], goal: str, evidence: str
    ) -> str:

        if value and _evidenced(value, evidence):
            return value
        ranked: list[tuple[int, str]] = []
        for key, observed_value in observed.items():
            if not observed_value or key in _CORE_CLAIM or key == "brand":
                continue
            if not _evidenced(observed_value, evidence):
                continue
            score = 0
            if _evidenced(observed_value, goal):
                score += 5
            if value and (_evidenced(observed_value, value) or _evidenced(value, observed_value)):
                score += 4
            if key in _DENY_FIELDS or key in {"display", "size", "facts", "options"}:
                score += 1
            if score:
                ranked.append((score, observed_value))
        if not ranked:
            return ""
        ranked.sort(key=lambda pair: (-pair[0], -len(pair[1])))
        return ranked[0][1]

    @staticmethod
    def _brand(raw: str, observed: dict[str, str], evidence: str) -> str:

        structured = str(observed.get("brand") or "").strip()
        if not structured or not _evidenced(structured, evidence):
            return ""
        if len(structured.split()) > 4:
            return ""
        candidate = str(raw or "").strip()
        if (
            candidate
            and len(candidate.split()) <= 3
            and len(candidate) <= 40
            and _evidenced(candidate, structured)
        ):
            return candidate
        return structured

_ROLE = """
You are LuckyGirl, an EnvPack purchasing scout. Tools are your only senses:
nothing is true until an observation printed it. The run ends when an
observation comes back done=true after place_test_order.

SCOUTING RHYTHM
  Sweep    search/filter with the shopper's own printed nouns
  Spread   collect DISTINCT product_ids — two listings of one product is one find
  Read     view/compare and read FACT ROWS; titles are advertising copy
  Take     add_to_cart only a fact-backed, in-budget, in-stock variant
  Settle   inspect_stock after carting or after any market movement
  Close    place_test_order — an unclosed run scores nothing

QUERY DISCIPLINE
Catalogs index printed values, not schema key names. Reuse shopper wording
first. When a pass comes back thin, change ONE token and try again — never
rewrite the whole query five times in five turns.

SHOPPER TRAFFIC
A question from the shopper outranks every other plan: answer it with message
on the very next turn, before any cart or order call.

BUDGET SENSE
Always keep two steps in hand for inspect_stock plus place_test_order.
""".strip()

_VOW_PLAYBOOK = """
VOW TRACK — justification claims (this is the scored part)
place_test_order must carry justification.claims as {field, value} pairs.

CLAIM SET (four at most, all of them when the facts exist)
  price      copied from the latest view or inspect_stock
  in_stock   normally "true", from inspect_stock
  gate       the specification the shopper actually asked about
             · the field name is the conventional attribute for the category —
               size/material/color for apparel, size/material/width for shoes,
               capacity/dimensions/finish for bags and home goods,
               resolution/screen_size for electronics
             · it is NEVER the catalog drawer key and NEVER the shopper's
               shorthand used as a field name
             · the value is the printed token from the answering fact row,
               copied character for character
  soft       at most one extra reason, and only if a fact row evidences it —
             short brand is the usual choice when the shopper named a maker

NON-NEGOTIABLE
- A figure that appears only in a title is a decoy. If no fact row prints it,
  do not cart and do not claim it — keep reading candidates.
- The first commitment must be fact-backed; one overlapping attested token is
  enough when the ask stacks several specs.
- Banned fields: category, title, summary, evidence, reason, description, os,
  display. Never join a drawer name to an attribute name.
- No unit conversion, no invented features, no prose paragraphs inside a claim.
- After a market event: inspect_stock the event-target sku immediately (that
  closes the event); inspecting a replacement does not. Then remove if it no
  longer qualifies, rebuild the claims from the fresh observation, then order.
- inspect_stock a candidate BEFORE the first add_to_cart (intended commitment).
- On every other track: leave justification out entirely — an untruthful
  optional claim scores zero.
- Once a fact-backed pick is inspected, order it. Do not wait for permission
  and do not keep hunting synonyms with the step budget running down.
""".strip()

_PLAYBOOK: dict[str, str] = {
    FORK: """
FORK TRACK — split the ask into walls and wishes
1. Write the BINDING walls out first, then the WISH leanings, separately.
2. Run two discovery passes: one that includes the preferred wish token, one
   that only carries the walls. Both passes matter; do not collapse into one.
3. Shortlist only items that clear every wall. A wish never excuses a wall.
4. If a wall-clearing item also satisfies the wish, that is the pick.
5. If nothing satisfies the wish, buy the best wall-clearing item and say so
   with message.
6. Omit justification on this track.
""".strip(),
    SCOUT: """
SCOUT TRACK — recall by reformulation (LuckyGirl's home track)
1. Pass one: the shopper's distinctive nouns, plus any quoted phrase verbatim.
2. Judge the pass by DISTINCT product_ids, not by row count. One product shown
   twice is one identity.
3. If the pass is thin, spend exactly ONE near-miss refine: literal tokens from
   the ask, no filler adjectives, no price or shipping words, two to five words.
4. Then stop reformulating. View the distinct candidates you found.
5. Verify any quoted phrase against title AND fact rows before trusting it.
6. Discard accessories and bundles by reading classification facts, never by
   guessing from a title.
7. Cart only after at least two identities have been read, when two exist.
8. Omit justification on this track.
""".strip(),
    CIPHER: """
CIPHER TRACK — literal codes and withheld figures
1. If the ask says a figure will only be revealed when you ask, send message
   FIRST and search nothing until the answer lands.
2. Match codes, model numbers and skus literally. Spacing or hyphen variants
   are fine inside a search string, but the item you select must carry the
   code exactly as printed in its facts.
3. A near-miss code is a different product. Never settle for a sibling.
4. Every binding figure must hold on the exact variant you order, not on the
   family listing.
5. Read option facts rather than believing the title.
6. Omit justification unless the rules also demand it.
""".strip(),
    SAVOR: """
SAVOR TRACK — preference scoring after the walls are clear
1. Apply the hard walls first and throw out everything that fails them.
2. Gather at least two wall-valid candidates across at least two observation
   batches before you score anything. One batch is not a comparison.
3. Score the candidates only on the axes the shopper actually named, using raw
   fact values, not titles.
4. Cheapness is a tiebreaker, never the goal, unless the shopper said so.
5. Name your winner and the axis it won on with message when useful.
6. Omit justification on this track.
""".strip(),
    TIER: """
TIER TRACK — rank a closed set
1. The candidate set is exactly the products named in the ask. Do not open a
   wide search to add strangers to the race.
2. View or compare each named candidate and keep their fact rows.
3. Sort strictly by the priority the shopper stated, in the order stated.
4. If a market event lands, re-sort from fresh observations before ordering.
5. place_test_order carries product_id and sku only. No justification.
""".strip(),
    RESCUE: """
RESCUE TRACK — the wanted item is gone
1. Keep the sticky core: product class, ceiling figure, availability. Those
   never bend.
2. Concede in order: finish or shade first, then variant or model. Never drop
   the product class.
3. Carry the sticky tokens into every refine query so the replacement is a
   true near neighbour and not a random substitute.
4. inspect_stock a candidate BEFORE the first add_to_cart (intended commitment).
5. After a market event: inspect_stock the event-target sku immediately to
   close the event — inspecting a replacement does not. Then remove if invalid,
   then find a replacement. Re-inspect after any cart change.
6. Tell the shopper what you conceded with message.
7. Omit justification on this track.
""".strip(),
    VOW: _VOW_PLAYBOOK,
}

_NUDGE = (
    "That reply had no tool call in it. Choose a tool and act this turn.",
    "Still idle. search → view fact rows → add_to_cart → inspect_stock → place_test_order.",
)

_VOW_TIPS = (
    "Claim coach — the gate field is the conventional attribute for the "
    "category (size/material/color for apparel, size/material/width for shoes, "
    "capacity/dimensions/finish for bags and home, resolution/screen_size for "
    "electronics). Drawer keys differ per category and are value evidence only. "
    "Shopper shorthand is a value, never a field name.",
    "Claim coach — copy price and in_stock from the freshest view or "
    "inspect_stock. The gate value must be the printed fact-row token, letter "
    "for letter. Never invent a field by joining a drawer to an attribute, and "
    "never emit display or os as a field.",
    "Claim coach — one soft reason at most, and short brand only when a brand "
    "fact row prints it. Four claims is the ceiling. Banned: category, title, "
    "os, display, and a full product title pretending to be a brand.",
    "Claim coach — after any market event, inspect_stock the event-target sku "
    "before anything else; that closes the event — a replacement inspect does "
    "not. Then rebuild every claim from the fresh observation instead of "
    "reusing stale numbers.",
    "Claim coach — if the asked figure shows up only in the title, keep reading "
    "candidates. Title decoys score as false claims. Wait for a fact row that "
    "prints it, cart that sku, and remember one overlapping attested token is "
    "enough to call a pick fact-backed.",
    "Claim coach — no unit conversion and no invented features. With the budget "
    "half spent, commit to the attested pick and close: leave room for "
    "inspect_stock plus place_test_order rather than hunting one more synonym.",
)

def _system_prompt(brief: ScoutBrief, policy: dict[str, Any]) -> str:
    parts = [_ROLE, _PLAYBOOK.get(brief.track, _PLAYBOOK[SCOUT])]
    if brief.needs_claims() and brief.track != VOW:
        parts.append(_VOW_PLAYBOOK)
    sheet = brief.sheet()
    if sheet:
        parts.append(sheet.strip())
    labels = _tool_labels(policy.get("tools") or [])
    parts.append(
        "SESSION DIAL\n"
        f"- track: {brief.track}\n"
        f"- tools offered: {', '.join(labels) or '(see schemas)'}\n"
        f"- step limit: {policy.get('max_steps')}\n"
        f"- calls allowed per turn: {policy.get('max_calls_per_turn', 1)}"
    )
    return "\n\n".join(parts)

def _vow_tip(turn: int, brief: ScoutBrief, well: ScoutWell) -> str:
    body = _VOW_TIPS[(turn - 1) % len(_VOW_TIPS)]
    extras: list[str] = []
    if brief.claim_enum:
        extras.append(f"Allowed fields this run: {', '.join(brief.claim_enum)}.")
    if brief.claim_rule:
        extras.append(f"Schema rule: {brief.claim_rule[:180]}")
    pid = well.focus_pid or well.last_pid
    if pid:
        observed = well.rows.get(pid, {})
        if observed:
            sample = ", ".join(f"{k}={v}" for k, v in list(observed.items())[:8])
            extras.append(f"Fact sample for {pid}: {sample}")
    if well.price or well.stock:
        extras.append(
            f"Freshest price={well.price or '?'} in_stock={well.stock or '?'}."
        )
    return body + (("\n" + "\n".join(extras)) if extras else "")

class ScoutLog:

    __slots__ = (
        "turn",
        "limit",
        "last_labels",
        "history",
        "query_sigs",
        "probe_batches",
        "wish_probe",
        "wall_probe",
        "carted",
        "closed",
        "owes_reply",
        "event_open",
        "event_pid",
        "event_sku",
        "asked_gate",
        "coached",
        "cart_pid",
        "cart_sku",
        "failures",
    )

    def __init__(self, limit: int) -> None:
        self.turn = 0
        self.limit = max(1, limit)
        self.last_labels: list[str] = []
        self.history: set[str] = set()
        self.query_sigs: set[str] = set()
        self.probe_batches = 0
        self.wish_probe = False
        self.wall_probe = False
        self.carted = False
        self.closed = False
        self.owes_reply = False
        self.event_open = False
        self.event_pid = ""
        self.event_sku = ""
        self.asked_gate = False
        self.coached: set[str] = set()
        self.cart_pid = ""
        self.cart_sku = ""
        self.failures = 0

    @property
    def left(self) -> int:
        return max(0, self.limit - self.turn)

    @property
    def halfway(self) -> bool:
        return self.turn * 2 >= self.limit

    def ever(self, label: str) -> bool:
        return label in self.history

    def searched(self) -> bool:
        return bool({"search", "filter"} & self.history)

    def dual_probed(self) -> bool:

        return len(self.query_sigs) >= 2 or (self.wish_probe and self.wall_probe)

    def note_turn(self, labels: list[str], calls: list[dict[str, Any]]) -> None:
        self.last_labels = labels
        self.history.update(labels)
        if {"search", "filter", "compare", "view"} & set(labels):
            self.probe_batches += 1
        if "add_to_cart" in labels:
            self.carted = True
            for call in calls:
                if _tool_name(call) != "add_to_cart":
                    continue
                args = _tool_args(call)
                self.cart_pid = str(args.get("product_id") or self.cart_pid)
                self.cart_sku = str(args.get("sku") or self.cart_sku)
        if "remove_from_cart" in labels:
            self.carted = False
        if "place_test_order" in labels:
            self.closed = True
        if "message" in labels:
            self.owes_reply = False
            self.asked_gate = True
        if "inspect_stock" in labels:
            for call in calls:
                if _tool_name(call) != "inspect_stock":
                    continue
                pid = str(_tool_args(call).get("product_id") or "")
                if self.event_pid and pid == self.event_pid:
                    self.event_open = False
                    self.event_pid = ""
                    self.event_sku = ""
                    break

    def note_query(self, text: str, wish_token: str) -> None:
        query = str(text or "").strip().lower()
        if not query:
            return
        self.query_sigs.add(" ".join(sorted(_tight(w) for w in query.split())[:6]))
        if wish_token and _word_hit(wish_token, query):
            self.wish_probe = True
        else:
            self.wall_probe = True

    def once(self, key: str) -> bool:
        if key in self.coached:
            return False
        self.coached.add(key)
        return True

class ScoutBridge:

    def __init__(self, client: ProxyClient) -> None:
        self._client = client

    @staticmethod
    def engine_for(track: str) -> str:
        override = (getenv("SANDBOX_MODEL") or "").strip()
        if override:
            low = override.lower()
            if (
                "kimi" in low
                or "glm" in low
                or low.startswith("z-ai/")
                or low.startswith("moonshotai/")
            ):
                return _ENGINE_HEAVY
            return override
        return _ENGINE_HEAVY if track in _HEAVY_TRACKS else _ENGINE_FAST

    def _ladder(self, engine: str) -> list[str]:
        return [engine, *[m for m in _ENGINE_RELIEF if m != engine]][:3]

    def ask_tools(
        self,
        *,
        engine: str,
        transcript: list[dict[str, Any]],
        schemas: list[Any],
        budget: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        problem: Exception | None = None
        for model in self._ladder(engine):
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
                problem = RuntimeError(f"empty reply from {model}")
                continue
            try:
                message = payload["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                problem = exc
                continue
            return message.get("content") or "", list(message.get("tool_calls") or [])[:budget]
        raise RuntimeError(f"inference failed ({problem})")

    def ask_json(self, *, engine: str, system: str, user: str) -> str:
        for model in self._ladder(engine):
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

    def run_tools(self, envelope: dict[str, Any]) -> dict[str, Any]:
        reply = self._client.post("/environment/call", json_data=envelope)
        if reply is None:
            raise RuntimeError("environment call failed")
        return reply

_bridge = ScoutBridge(ProxyClient(timeout=120, max_retries=2))

def _tool_name(call: Any) -> str:
    spec = call.get("function") if isinstance(call, dict) else None
    return str(spec["name"]) if isinstance(spec, dict) and spec.get("name") else ""

def _tool_args(call: Any) -> dict[str, Any]:
    spec = call.get("function") if isinstance(call, dict) else None
    raw = spec.get("arguments", "{}") if isinstance(spec, dict) else "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}

def _call_id(call: Any, fallback: str) -> str:
    return str(call.get("id") or fallback) if isinstance(call, dict) else fallback

def _clip(obs: Any, cap: int = 6000) -> str:
    try:
        text = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obs)
    return text if len(text) <= cap else text[: cap - 18] + "…<clipped>"

def _run_done(result: dict[str, Any]) -> bool:
    for row in result.get("calls") or []:
        if isinstance(row, dict):
            obs = row.get("observation")
            if isinstance(obs, dict) and obs.get("done"):
                return True
    return False

def _market_moved(obs: Any) -> bool:
    if not isinstance(obs, dict):
        return False
    keys = ("market_event", "stock_changed", "price_changed", "event")
    if any(obs.get(k) for k in keys):
        return True
    nested = obs.get("observation")
    if isinstance(nested, dict):
        return any(nested.get(k) for k in keys[:3])
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

def _looks_failed(obs: Any) -> bool:

    if not isinstance(obs, dict):
        return False
    if obs.get("error") or obs.get("ok") is False or obs.get("success") is False:
        return True
    text = str(obs.get("message") or obs.get("status") or "").lower()
    return any(
        cue in text
        for cue in ("unavailable", "out of stock", "sold out", "not found", "invalid", "failed")
    )

def _is_question(text: str) -> bool:
    body = (text or "").strip().lower()
    if not body:
        return False
    if "?" in body:
        return True
    starters = ("which", "what", "does", "did", "can you", "could you", "wait", "hold on", "are you")
    return any(body.startswith(s) for s in starters)

class ScoutEpisode:

    def __init__(self, problem_data: dict[str, Any]) -> None:
        env = problem_data["environment"]
        self.binding = env["binding"]
        self.policy = env["policy_view"]
        self.schemas = self.policy.get("tools") or []
        self.text = str(self.policy.get("query") or problem_data.get("query") or "")
        self.key = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        self.per_turn = max(1, int(self.policy.get("max_calls_per_turn", 1)))
        self.brief = ScoutBrief(self.text, self.schemas)
        self.log = ScoutLog(int(self.policy.get("max_steps", 20)))
        self.bag = IdentityBag()
        self.kit = RefineKit()
        self.well = ScoutWell()
        self.engine = ScoutBridge.engine_for(self.brief.track)

        self._coaches = {
            FORK: self._track_fork,
            SCOUT: self._track_scout,
            CIPHER: self._track_cipher,
            SAVOR: self._track_savor,
            TIER: self._track_tier,
            RESCUE: self._track_rescue,
            VOW: self._track_vow,
        }

        self._shapers = {
            TIER: self._shape_tier,
            VOW: self._shape_vow,
            RESCUE: self._shape_rescue,
        }

    def run(self) -> list[dict[str, Any]]:
        transcript: list[dict[str, Any]] = [
            {"role": "system", "content": _system_prompt(self.brief, self.policy)},
            {"role": "user", "content": self.text},
        ]
        if self.brief.track == CIPHER and self.brief.withheld:
            transcript.append(
                {
                    "role": "user",
                    "content": (
                        "A figure in this ask is withheld until you ask for it. Send "
                        "message to ask for it before spending a search."
                    ),
                }
            )
        trail: list[dict[str, Any]] = [
            {
                "role": "meta",
                "luckygirl": {
                    "version": 6,
                    "engine_name": "ScoutRefineEngine",
                    "track": self.brief.track,
                    "model": self.engine,
                    "identities_required": 2,
                    "provider": "openrouter",
                },
            }
        ]
        while self.log.turn < self.log.limit:
            self.log.turn += 1
            coach = self._coach()
            if coach:
                transcript.append({"role": "user", "content": coach})
            spoken, calls = self._emit(transcript, trail)
            calls = self._block_order_during_event(calls)
            transcript.append({"role": "assistant", "content": spoken, "tool_calls": calls})
            result = self._execute(calls, self.log.turn)
            trail.append(
                {
                    "role": "assistant",
                    "content": spoken,
                    "tool_calls": calls,
                    "environment_result": result,
                    "luckygirl_turn": {
                        "track": self.brief.track,
                        "turn": self.log.turn,
                        "identities": self.bag.identities,
                        "queries_spent": self.kit.spent(),
                    },
                }
            )
            self.log.note_turn([_tool_name(c) or "?" for c in calls], calls)
            self._absorb(result, transcript, calls)
            if _run_done(result):
                break
        return trail

    def _coach(self) -> str | None:

        floor = self._floor_coach()
        if floor:
            return floor
        handler = self._coaches.get(self.brief.track, self._track_scout)
        chosen = handler()
        if chosen:
            return chosen
        return self._closing_coach()

    def _floor_coach(self) -> str | None:
        log = self.log
        if log.owes_reply:
            return (
                "The shopper just asked you something. Answer it with message this "
                "turn — no cart, no order until you have replied."
            )
        if log.event_open and not log.closed:
            target = log.event_pid or log.cart_pid or "the carted sku"
            return (
                f"The market moved. inspect_stock the event-target ({target}) "
                "immediately to close the event — inspecting a replacement does "
                "not count. Then remove it if invalid, then continue."
            )
        if log.turn <= 2 and not any(
            log.ever(label) for label in ("search", "filter", "view", "compare", "message")
        ):
            return (
                "Open the run: search or filter with the shopper's own nouns, then "
                "read fact rows on the distinct hits you get back."
            )
        return None

    def _closing_coach(self) -> str | None:
        log = self.log
        if log.carted and log.left <= 6 and not log.closed and log.once("closing"):
            return (
                f"Only {log.left} step(s) left. inspect_stock the carted sku if needed, then "
                "place_test_order — an unclosed run scores nothing."
            )
        if log.left <= 8 and not log.carted and log.ever("view") and log.once("commit"):
            return (
                "Step budget is low. If a viewed candidate already clears the "
                "walls, cart it NOW and keep two steps for inspect_stock and the order."
            )
        if log.left <= 1 and not log.closed and log.once("last"):
            return "Last step. place_test_order on the best verified candidate immediately."
        return None

    def _track_fork(self) -> str | None:

        log = self.log
        if log.searched() and not log.dual_probed() and log.once("fork_dual"):
            missing = "walls-only" if log.wish_probe else "preferred-token"
            return (
                "Fork discovery is one-sided: you have spent only one kind of pass. "
                f"Run the {missing} pass next so both lanes are scouted, then "
                "shortlist items that clear every binding wall."
            )
        if log.dual_probed() and not log.ever("view") and log.once("fork_read"):
            return (
                "Both discovery lanes are spent. Now view the candidates and check "
                "each binding wall against fact rows — a wish never excuses a wall."
            )
        if log.dual_probed() and log.ever("view") and not log.carted and log.once("fork_pick"):
            return (
                "Pick from wall-clearing items only, preferring the one that also "
                "carries the shopper's leaning. Say what you traded with message."
            )
        return None

    def _track_scout(self) -> str | None:

        log, bag = self.log, self.bag
        if log.searched() and bag.thin() and self.kit.spent() <= 1:
            suggestion = self.kit.propose(self.brief)
            if suggestion and log.once("scout_refine"):
                return (
                    f"That pass came back thin ({bag.blurb()}). Spend ONE near-miss "
                    f"refine now — try a query close to: {suggestion} — literal "
                    "tokens only, no filler adjectives and no price or shipping "
                    "words. After this refine, stop reformulating and start viewing."
                )
        if self.kit.spent() >= 3 and not log.ever("view") and log.once("scout_stop"):
            return (
                "Enough reformulating. You have spent several queries; view the "
                "distinct product_ids you already found and read their fact rows."
            )
        if bag.identities >= 2 and len(bag.viewed) < 2 and log.once("scout_spread"):
            return (
                f"You have {bag.identities} distinct identities but only "
                f"{len(bag.viewed)} read. View a second distinct product before "
                "committing — duplicates of one listing are not a shortlist."
            )
        if bag.seen_enough and not log.carted and log.once("scout_ready"):
            phrase = self.brief.phrase
            tail = (
                f" Confirm the quoted phrase '{phrase}' against fact rows, not just the title."
                if phrase
                else ""
            )
            return (
                "The shortlist is read. Cart the candidate whose fact rows match the "
                "ask, then inspect_stock and order." + tail
            )
        return None

    def _track_cipher(self) -> str | None:

        log = self.log
        if self.brief.withheld and not log.asked_gate:
            if log.once("cipher_ask"):
                return (
                    "The withheld figure is still unknown. Send message asking for it "
                    "now and do not search until the shopper answers."
                )
            return (
                "Still missing the withheld figure — ask for it with message before "
                "any other call."
            )
        codes = self.brief.codes
        if codes and log.ever("view") and log.once("cipher_literal"):
            return (
                f"Match literally: {', '.join(codes)}. Spacing or hyphen variants are "
                "fine inside a search string, but the sku you select must print the "
                "code in its own facts. A near-miss code is a different product."
            )
        if log.ever("view") and not log.carted and log.once("cipher_variant"):
            return (
                "Check the exact variant, not the family listing: every binding "
                "figure must hold on the sku you are about to cart."
            )
        return None

    def _track_savor(self) -> str | None:

        log = self.log
        if log.probe_batches < 2 and log.once("savor_batches"):
            return (
                f"Only {log.probe_batches} observation batch so far. Gather a second "
                "batch of wall-valid candidates before scoring preferences — one "
                "batch is not a comparison."
            )
        if log.probe_batches >= 2 and not log.carted and log.once("savor_score"):
            return (
                "Two batches are in. Drop everything that fails a hard wall, then "
                "rank what is left only on the axes the shopper named, reading raw "
                "fact values. Cheapness is a tiebreaker, not the goal."
            )
        if log.carted and not log.closed and log.once("savor_close"):
            return "Winner is carted. inspect_stock it, then place_test_order without justification."
        return None

    def _track_tier(self) -> str | None:

        log = self.log
        named = self.brief.named_set
        if named and log.searched() and log.once("tier_closed"):
            return (
                "This is a closed set: " + ", ".join(named[:6]) + ". Stop open "
                "searching and view or compare exactly these candidates."
            )
        if named and not log.ever("view") and not log.ever("compare") and log.once("tier_view"):
            return (
                "View or compare each named candidate and keep their fact rows — the "
                "ranking must come from observations, not from the ask's wording."
            )
        if log.event_open and log.once("tier_resort"):
            return "Market moved: re-sort the named set from fresh observations before ordering."
        if log.left <= 3 and not log.closed and log.once("tier_close"):
            return (
                "Close the ranking: cart the current leader if it is not carted, then "
                "place_test_order with product_id and sku only — no justification."
            )
        return None

    def _track_rescue(self) -> str | None:

        log = self.log
        sticky = self.kit.sticky(self.brief)
        if log.failures and log.once(f"rescue_fail_{log.failures}"):
            keep = ", ".join(sticky) if sticky else "the product class"
            return (
                "That candidate failed. Keep the sticky tokens in the next query "
                f"({keep}) and concede only the softest attribute — finish or shade "
                "first, model second. Never drop the product class or the ceiling."
            )
        if log.searched() and self.bag.thin() and log.once("rescue_refine"):
            suggestion = self.kit.propose(self.brief)
            hint = f" Try something near: {suggestion}." if suggestion else ""
            return (
                "Thin recovery pass. Refine once while keeping the sticky tokens "
                f"({', '.join(sticky) or 'product class'}) in the query." + hint
            )
        if (
            not log.carted
            and log.ever("view")
            and not log.ever("inspect_stock")
            and log.once("rescue_precart")
        ):
            return (
                "Before the first add_to_cart: inspect_stock the candidate so "
                "the environment records an intended commitment."
            )
        if log.carted and not log.ever("inspect_stock") and log.once("rescue_inspect"):
            return "Re-inspect the replacement's stock before ordering; unclosed events ruin the run."
        if log.carted and not log.closed and log.once("rescue_tell"):
            return (
                "Tell the shopper which attribute you conceded with message, then "
                "place_test_order to close the run."
            )
        return None

    def _track_vow(self) -> str | None:

        log, well = self.log, self.well
        pid = well.focus_pid or well.last_pid
        if pid and not log.carted and not log.closed:
            if well.title_trap(pid, self.brief.goal) and log.once("vow_decoy"):
                return (
                    "Title decoy: the asked figure is in the title of this sku but in "
                    "none of its fact rows. Do not cart it — keep reading candidates "
                    "until a fact row prints the figure."
                )
            if (
                log.ever("view")
                and not well.fact_supported(pid, self.brief.goal)
                and log.once("vow_backing")
            ):
                return (
                    "The first commitment must be fact-backed: cart only when a fact "
                    "row overlaps the shopper's specification. One overlapping "
                    "attested token is enough for a stacked ask."
                )
        if not log.closed:
            return _vow_tip(log.turn, self.brief, well)
        return None

    def _shape_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name in {"search", "filter"}:
            query = args.get("query") or args.get("q") or args.get("keywords")
            self.kit.remember(query)
            wish = self.brief.spec_tokens[0] if self.brief.spec_tokens else ""
            self.log.note_query(str(query or ""), wish)
            return args
        if name in {"view", "compare", "inspect_stock"}:
            pid = str(args.get("product_id") or "")
            if pid:
                self.bag.mark_viewed(pid)
                self.well.focus_pid = self.well.focus_pid or pid
            for pid in args.get("product_ids") or []:
                self.bag.mark_viewed(str(pid))
            return args
        if name == "add_to_cart":
            pid = str(args.get("product_id") or "")
            if pid:
                self.well.focus_pid = pid
                self.bag.mark_viewed(pid)
            return args
        shaper = self._shapers.get(self.brief.track)
        if name == "place_test_order":
            if len(self.brief.named_set) >= 2:
                return self._strip_claims(dict(args))
            if self.brief.track == VOW:
                return self._shape_vow(dict(args))
            if shaper:
                return shaper(dict(args))
            return self._strip_claims(dict(args))
        return args

    def _block_order_during_event(
        self, calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not self.log.event_open or self.log.closed:
            return calls
        event_pid = self.log.event_pid
        event_sku = self.log.event_sku
        if event_pid:
            probe: dict[str, Any] = {"product_id": event_pid}
            if event_sku:
                probe["sku"] = event_sku
            if not calls:
                return calls

            messages: list[dict[str, Any]] = []
            rest: list[dict[str, Any]] = []
            for call in calls:
                name = _tool_name(call)
                if name == "message":
                    messages.append(call)
                    continue
                if name in {"search", "filter"}:
                    continue
                if (
                    name == "inspect_stock"
                    and str(_tool_args(call).get("product_id") or "") == event_pid
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
            name = _tool_name(call)
            if name != "place_test_order":
                out.append(call)
                continue
            args = _tool_args(call)
            pid = str(
                args.get("product_id")
                or self.log.cart_pid
                or self.well.focus_pid
                or self.well.last_pid
                or ""
            )
            sku = str(args.get("sku") or self.log.cart_sku or "")
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

    def _strip_claims(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload.pop("justification", None)
        return self._with_pid(payload)

    def _shape_tier(self, payload: dict[str, Any]) -> dict[str, Any]:

        payload.pop("justification", None)
        payload.pop("claims", None)
        return self._with_pid(payload)

    def _shape_rescue(self, payload: dict[str, Any]) -> dict[str, Any]:

        payload.pop("justification", None)
        return self._with_pid(payload)

    def _shape_vow(self, payload: dict[str, Any]) -> dict[str, Any]:

        if self.brief.track != VOW:
            return self._strip_claims(payload)
        payload = self._with_pid(payload)
        pid = str(payload.get("product_id") or "")
        note = payload.get("justification")
        raw = note.get("claims") if isinstance(note, dict) else None
        payload["justification"] = {"claims": ScoutClaims.polish(raw, self.brief, self.well, pid)}
        return payload

    def _with_pid(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not payload.get("product_id"):
            pid = self.log.cart_pid or self.well.focus_pid or self.well.last_pid
            if pid:
                payload["product_id"] = pid
        return payload

    def _emit(
        self, transcript: list[dict[str, Any]], trail: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        for attempt in range(len(_NUDGE) + 1):
            spoken, calls = _bridge.ask_tools(
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

    def _execute(self, calls: list[dict[str, Any]], turn: int) -> dict[str, Any]:
        batch = f"{self.key}-scout-{turn}"
        actions: list[dict[str, Any]] = []
        for slot, call in enumerate(calls, start=1):
            name = _tool_name(call)
            if not name:
                raise RuntimeError("tool call without a name")
            args = self._shape_args(name, _tool_args(call))
            spec = call.get("function")
            if isinstance(spec, dict):
                spec["arguments"] = json.dumps(args)
            actions.append(
                {"call_id": f"{batch}-{slot}", "action": {"name": name, "args": args}}
            )
        return _bridge.run_tools(
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
            name = _tool_name(call)
            if name in {"search", "filter", "compare"}:
                self.bag.absorb(obs)
            elif name == "view":
                self.bag.absorb(obs)
                pid = str(_tool_args(call).get("product_id") or "")
                if pid:
                    self.bag.mark_viewed(pid)
            if _market_moved(obs):
                self.log.event_open = True
                ep, es = _event_target(obs)
                snap = (
                    ep
                    or self.log.cart_pid
                    or self.well.focus_pid
                    or self.well.last_pid
                    or ""
                )
                if snap:
                    self.log.event_pid = snap
                    if es:
                        self.log.event_sku = es
                    elif self.log.cart_sku:
                        self.log.event_sku = self.log.cart_sku
            if _looks_failed(obs):
                self.log.failures += 1
            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": _call_id(call, f"{self.key}-{self.log.turn}-{index}"),
                    "content": _clip(obs),
                }
            )
        note = result.get("user_message")
        if isinstance(note, dict) and note.get("content"):
            text = str(note["content"])
            transcript.append({"role": "user", "content": text})
            if _is_question(text):
                self.log.owes_reply = True
        if self.log.event_open and self.brief.needs_claims() and not self.log.closed:
            transcript.append(
                {
                    "role": "user",
                    "content": (
                        "Event note: inspect_stock the event-target sku "
                        f"({self.log.event_pid or self.log.cart_pid or 'carted'}) "
                        "to close the event — inspecting a replacement does not. "
                        "Then rebuild every claim from the fresh observation before "
                        "place_test_order."
                    ),
                }
            )

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:

    return ScoutEpisode(problem_data).run()

__all__ = ["agent_main"]
