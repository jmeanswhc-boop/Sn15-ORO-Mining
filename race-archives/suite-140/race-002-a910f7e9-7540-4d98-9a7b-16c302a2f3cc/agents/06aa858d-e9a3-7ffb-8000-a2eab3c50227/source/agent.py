

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_QUICK_ENGINE = "deepseek/deepseek-v3.2"
_HEAVY_ENGINE = "qwen/qwen3.5-397b-a17b"
_RELIEF_ENGINES = (
    "google/gemma-4-31b-it",
    "openai/gpt-oss-120b",
    "deepseek/deepseek-chat-v3.1",
)

SLATE = "slate"
ROSTER = "roster"
PLINTH = "plinth"
SCALES = "scales"
PODIUM = "podium"
LADDER = "ladder"
SEAL = "seal"

_HEAVY_LANES = frozenset({PODIUM, SEAL, LADDER, PLINTH, SCALES})

_CANDIDATE_RX = re.compile(r"\b\d{4,}::[\w.-]+")
_CEILING_RX = re.compile(
    r"(?:at or (?:under|below)|under|below|beneath|up to|no more than|not exceed(?:ing)?|"
    r"within|max(?:imum)?(?:\s+of)?|budget(?:\s+of)?)\s*(?:[^\d\n]{0,12})?"
    r"([\d][\d,]*(?:\.\d+)?)",
    re.I,
)
_QUOTED_RX = re.compile(r"[\"\u201c]([^\"\u201d]{3,70})[\"\u201d]")
_SEAL_DUTY_RX = re.compile(
    r"structured\s+claim|justification\s*(?:\.|\s)*claims|justification\s+is\s+required|"
    r"must\s+include\s+justification|state\s+(?:your|the)\s+reasons|"
    r"(?:explain|justify)\s+(?:your|the|each)\s+(?:choice|pick|decision|selection)|"
    r"walk me through why|tell me why|give me a reason|why the one you pick",
    re.I,
)
_REASON_RX = re.compile(
    r"\bwhy\b|\breasons?\b|\bjustif\w*|\bexplain\b|\bdefend\b|"
    r"\bwalk me through\b|\bgive me a reason\b|\bback it up\b",
    re.I,
)
_GONE_RX = re.compile(
    r"\bsold\s*out\b|\bout\s+of\s+stock\b|\bunavailable\b|\bdisappear\w*|\bno longer\b|"
    r"\bvanish\w*|\bif (?:it|that|the \w+) (?:is|goes|becomes) (?:gone|unavailable)\b|"
    r"\bfall(?:s|en)?\s+through\b|\bbackup\b|\bnext best\b",
    re.I,
)
_RELAX_RX = re.compile(
    r"(?:flexible (?:on|about)|can (?:compromise|bend|give) on|willing to (?:drop|lose|"
    r"give up|accept)|okay (?:to|with) (?:dropping|losing)|relax|concede|settle for|"
    r"could live without|least important)[:\s]+([a-z][a-z0-9 _/'-]{2,48})",
    re.I,
)
_EXACT_RX = re.compile(
    r"\bexact(?:ly)?\b|\bprecise(?:ly)?\b|\bthat (?:exact|specific) (?:sku|variant|option)\b|"
    r"\bspecific (?:sku|variant|configuration)\b|\bmust be the\b",
    re.I,
)
_WITHHELD_RX = re.compile(
    r"\bif you ask\b|\bwhen you ask\b|\bonly if asked\b|"
    r"(?:tell|share|reveal|give)[^.]{0,44}\bif\b[^.]{0,24}\bask\w*",
    re.I,
)
_WISH_RX = re.compile(
    r"\bideally\b|\bprefer(?:ably|red|ence)?\b|\bwould like\b|\bnice to have\b|"
    r"\bleaning\b|\bhoping for\b|\brather\b",
    re.I,
)
_FIRM_RX = re.compile(
    r"\bfirm\b|\bmust\b|\brequired?\b|\bnon[- ]?negotiable\b|\bhard requirement\b|"
    r"\bhas to\b|\bneeds? to\b|\bstrict(?:ly)?\b|\bmandatory\b",
    re.I,
)
_SOFT_RX = re.compile(
    r"\bmatters more\b|\bcare(?:s)? more\b|\bmore important\b|\bvalue\b|\btrade[- ]?off\b|"
    r"\brather (?:have|pay)\b|\bnot the cheapest\b|\bworth paying\b|\bpriorit\w+",
    re.I,
)
_AXIS_LEAD_RX = re.compile(
    r"(?:priorit(?:y|ise|ize|izing|ising)|rank(?:ed)?\s+by|sort(?:ed)?\s+by|order(?:ed)?\s+by|"
    r"judge(?:d)?\s+(?:by|on)|most important(?:ly)?|first(?:ly)?|second(?:ly)?|third(?:ly)?|"
    r"then|next|followed by|after that|tie[-\s]?break(?:er)?s?(?:\s+by|\s+on)?|"
    r"break ties (?:by|on|with)|highest|lowest|largest|biggest|smallest|cheapest|"
    r"maximi[sz]e|minimi[sz]e|"
    r"cares? more about|matters? more than|weigh(?:ts?|ing)? more|value(?:s)? \w+ over)"
    r"[:\s]+([a-z][a-z0-9 _/'-]{2,44})",
    re.I,
)

_AXIS_DIR_RX = re.compile(
    r"(?P<dir>highest|lowest|largest|biggest|smallest|cheapest|maximi[sz]e|minimi[sz]e)"
    r"[:\s]+(?P<axis>[a-z][a-z0-9 _/'-]{2,44})",
    re.I,
)
_AXIS_MAX_DIRS = frozenset(
    {"highest", "largest", "biggest", "maximize", "maximise"}
)
_AXIS_MIN_DIRS = frozenset(
    {"lowest", "smallest", "cheapest", "minimize", "minimise"}
)
_AXIS_CUT_RX = re.compile(
    r"\b(?:and|but|then|or|while|with|for|from|that|which|if|so|because|unless|as|than|"
    r"next|after|before|under|over|within|plus)\b|[,.;:!?()]"
)
_TRADEOFF_RX = re.compile(
    r"matters? more|cares? more|rather (?:pay|have|spend)|worth paying|not the cheapest|"
    r"trade[- ]?off|quality over|more important than|willing to pay more",
    re.I,
)
_IDLE_TAIL = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "be",
        "of",
        "to",
        "in",
        "on",
        "at",
        "as",
        "by",
        "with",
        "using",
        "it",
        "its",
        "my",
        "your",
        "their",
        "this",
        "that",
        "one",
        "any",
        "most",
        "more",
        "less",
        "very",
        "really",
        "please",
        "thanks",
    }
)

_FIELD_SHAPE_RX = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_CORE_FIELDS = frozenset({"price", "in_stock"})

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
_DRAWERS = ("display", "facts", "options", "os", "size", "title")
_SPEC_NUM_RX = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s?(?:inch|in|hz|p|k|gb|tb|mb|mm|cm|mp|w|wh|mah|oz|ml|"
    r"l|kg|lb|lbs|ft|yd|\"))?\b",
    re.I,
)
_SPEC_WORDS = frozenset(
    {
        "xs",
        "s",
        "m",
        "l",
        "xl",
        "xxl",
        "small",
        "medium",
        "large",
        "petite",
        "tall",
        "cotton",
        "polyester",
        "wool",
        "merino",
        "silk",
        "linen",
        "leather",
        "suede",
        "denim",
        "canvas",
        "nylon",
        "mesh",
        "rubber",
        "plastic",
        "metal",
        "steel",
        "aluminum",
        "aluminium",
        "wood",
        "bamboo",
        "glass",
        "ceramic",
        "navy",
        "black",
        "white",
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "gray",
        "grey",
        "brown",
        "beige",
        "cream",
        "khaki",
        "olive",
        "tan",
        "silver",
        "gold",
        "narrow",
        "wide",
        "regular",
        "slim",
        "relaxed",
        "matte",
        "glossy",
        "gloss",
        "satin",
        "brushed",
        "polished",
        "oak",
        "walnut",
        "maple",
        "chrome",
        "brass",
        "hd",
        "fhd",
        "qhd",
        "uhd",
        "oled",
        "lcd",
        "led",
        "qled",
    }
)
_ASK_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "with",
        "from",
        "that",
        "this",
        "find",
        "buy",
        "get",
        "need",
        "want",
        "looking",
        "please",
        "under",
        "below",
        "above",
        "price",
        "budget",
        "stock",
        "available",
        "why",
        "reason",
        "justify",
        "explain",
        "product",
        "item",
        "items",
        "shopper",
        "order",
        "cart",
        "exactly",
        "must",
        "should",
        "also",
        "only",
        "than",
        "then",
        "when",
        "which",
        "what",
        "have",
        "has",
        "are",
        "is",
        "be",
        "to",
        "of",
        "on",
        "at",
        "by",
        "as",
        "if",
        "not",
        "no",
        "yes",
        "true",
        "false",
        "brand",
        "size",
        "color",
        "colour",
        "material",
        "width",
        "capacity",
        "dimensions",
        "finish",
        "resolution",
        "screen",
        "screen_size",
    }
)

def _slug(raw: Any) -> str:

    return re.sub(r"[^a-z0-9]+", "", str(raw or "").lower())

def _metric_num(raw: Any) -> float | None:

    text = str(raw or "")
    if not text:
        return None
    hits = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    if not hits:
        return None
    try:
        return max(float(h) for h in hits)
    except ValueError:
        return None

def _field_key(raw: Any) -> str:

    low = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "", low)).strip("_")

def _printed(value: Any, blob: Any) -> bool:

    tight = _slug(value)
    return bool(tight) and tight in _slug(blob)

def _word_hit(token: str, text: str) -> bool:

    tok = str(token or "").strip().lower()
    blob = str(text or "").lower()
    if not tok or not blob:
        return False
    if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", blob):
        return True
    tight = _slug(tok)
    if not tight:
        return False
    hay = _slug(blob)
    if len(tight) <= 3:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(tight)}(?![a-z0-9])", hay))
    return tight in hay

def _spec_tokens(goal: str) -> list[str]:

    text = (goal or "").lower()
    out: list[str] = []
    seen: set[str] = set()
    for hit in _SPEC_NUM_RX.finditer(text):
        tok = hit.group(0).strip()
        key = _slug(tok)
        if key and key not in seen:
            seen.add(key)
            out.append(tok)
    for word in re.findall(r"[a-z][a-z0-9-]+", text):
        if word in _ASK_STOP or word not in _SPEC_WORDS:
            continue
        key = _slug(word)
        if key and key not in seen:
            seen.add(key)
            out.append(word)
    return out

def _drawer_concat(field: str) -> bool:

    key = _field_key(field)
    if not key:
        return False
    return any(key.startswith(d + "_") or key.endswith("_" + d) for d in _DRAWERS)

def _read_json(text: str) -> dict[str, Any]:
    body = (text or "").strip()
    if not body:
        return {}
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        head, tail = body.find("{"), body.rfind("}")
        if head < 0 or tail <= head:
            return {}
        try:
            obj = json.loads(body[head : tail + 1])
        except json.JSONDecodeError:
            return {}
    return obj if isinstance(obj, dict) else {}

def _uniq(items: Any) -> list[str]:
    return list(dict.fromkeys(str(i) for i in items))

def _tool_names(tools: list[Any]) -> list[str]:
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

def _claim_field_spec(tools: list[Any]) -> dict[str, Any]:

    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        if not isinstance(fn, dict) or fn.get("name") != "place_test_order":
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        note = props.get("justification") if isinstance(props, dict) else None
        claims = (note.get("properties") or {}).get("claims") if isinstance(note, dict) else None
        items = claims.get("items") if isinstance(claims, dict) else None
        field = (items.get("properties") or {}).get("field") if isinstance(items, dict) else None
        if isinstance(field, dict):
            return field
    return {}

class SilverBrief:

    __slots__ = (
        "raw",
        "goal",
        "rules",
        "lane",
        "gate",
        "tokens",
        "ceiling",
        "phrase",
        "axes",
        "rungs",
        "wants_reason",
        "withheld",
        "exact_talk",
        "claim_enum",
        "claim_rule",
    )

    def __init__(self, raw: str, tools: list[Any] | None = None) -> None:
        self.raw = raw or ""
        self.goal, self.rules = self._cleave(self.raw)
        self.tokens = _uniq(_CANDIDATE_RX.findall(self.raw))
        self.ceiling = self._ceiling()
        quoted = _QUOTED_RX.search(self.goal)
        self.phrase = quoted.group(1) if quoted else ""
        self.axes = self._axes()
        self.rungs = self._rungs()

        self.wants_reason = bool(_REASON_RX.search(self.goal))
        self.withheld = bool(_WITHHELD_RX.search(self.raw))
        self.exact_talk = bool(_EXACT_RX.search(self.goal))
        field = _claim_field_spec(tools or [])
        enum = field.get("enum") if isinstance(field, dict) else None
        self.claim_enum = (
            _uniq(_field_key(x) for x in enum if _field_key(x)) if isinstance(enum, list) else []
        )
        self.claim_rule = str(field.get("description") or "")[:600] if field else ""
        self.lane, self.gate = self._cascade()

    @staticmethod
    def _cleave(raw: str) -> tuple[str, str]:
        head, sep, tail = raw.partition("\n\n")
        if not sep:
            return raw.strip(), ""
        body = tail.strip()
        first, _, rest = body.partition("\n")
        if first.rstrip().endswith(":") and len(first) < 44:
            body = rest.strip()
        return head.strip(), body

    def _ceiling(self) -> str:
        for text in (self.goal, self.raw):
            hit = _CEILING_RX.search(text or "")
            if hit:
                return hit.group(1)
        return ""

    def _axes(self) -> list[str]:

        found: list[str] = []
        for hit in _AXIS_LEAD_RX.finditer(self.goal):
            found.append(hit.group(1))
        for hit in _AXIS_LEAD_RX.finditer(self.rules):
            found.append(hit.group(1))
        out: list[str] = []
        seen: set[str] = set()
        for chunk in found:
            phrase = _AXIS_CUT_RX.split(chunk, maxsplit=1)[0].strip(" -_/'")
            words = [w for w in phrase.split() if w]
            while words and words[-1].lower() in _IDLE_TAIL:
                words.pop()
            while words and words[0].lower() in _IDLE_TAIL:
                words.pop(0)
            phrase = " ".join(words)[:40].strip()
            key = _slug(phrase)
            if not phrase or len(key) < 3 or key in seen:
                continue
            seen.add(key)
            out.append(phrase)
        return out[:5]

    def axis_plan(self) -> list[tuple[str, str]]:

        plan: list[tuple[str, str]] = []
        seen: set[str] = set()
        for text in (self.goal, self.rules):
            for hit in _AXIS_DIR_RX.finditer(text or ""):
                direction = hit.group("dir").lower()
                chunk = hit.group("axis")
                phrase = _AXIS_CUT_RX.split(chunk, maxsplit=1)[0].strip(" -_/'")
                words = [w for w in phrase.split() if w]
                while words and words[-1].lower() in _IDLE_TAIL:
                    words.pop()
                while words and words[0].lower() in _IDLE_TAIL:
                    words.pop(0)
                phrase = " ".join(words)[:40].strip()
                key = _slug(phrase)
                if not phrase or len(key) < 3:
                    continue

                if key in seen or any(key in s or s in key for s in seen):
                    continue
                if direction in _AXIS_MAX_DIRS:
                    sense = "max"
                elif direction in _AXIS_MIN_DIRS:
                    sense = "min"
                else:
                    continue
                seen.add(key)
                plan.append((sense, phrase))
        if plan:
            return plan[:5]

        out: list[tuple[str, str]] = []
        for axis in self.axes:
            sense = "min" if "price" in axis.lower() or "cost" in axis.lower() else "max"
            out.append((sense, axis))
        return out

    def _rungs(self) -> list[str]:

        out: list[str] = []
        seen: set[str] = set()
        for text in (self.goal, self.rules):
            for hit in _RELAX_RX.finditer(text or ""):
                phrase = _AXIS_CUT_RX.split(hit.group(1), maxsplit=1)[0].strip(" -_/'")
                key = _slug(phrase)
                if not phrase or len(key) < 3 or key in seen:
                    continue
                seen.add(key)
                out.append(phrase[:40])
        return out[:4]

    def _cascade(self) -> tuple[str, str]:

        goal, rules = self.goal.lower(), self.rules.lower()
        both = goal + "\n" + rules

        if len(self.tokens) >= 2:
            return PODIUM, "closed-set"

        if _SEAL_DUTY_RX.search(goal) or _REASON_RX.search(goal):
            return SEAL, "claim-duty"

        if _SEAL_DUTY_RX.search(rules) and "omit" not in rules and "do not" not in rules:
            return SEAL, "claim-duty-rules"

        if _GONE_RX.search(both) or self.rungs:
            return LADDER, "loss-talk"

        if self.withheld or self.exact_talk:
            return PLINTH, "exact-wall"

        if self.axes and _SOFT_RX.search(both):
            return SCALES, "stated-axes"

        if _TRADEOFF_RX.search(both):
            return SCALES, "trade-off"

        if self.wants_reason and self.claim_enum:
            return SEAL, "reason-ask"

        if _WISH_RX.search(both) and _FIRM_RX.search(both):
            return SLATE, "firm-wish"

        if len(self.tokens) == 1:
            return PLINTH, "single-token"

        if _SOFT_RX.search(both):
            return SCALES, "soft-talk"

        return ROSTER, "phrase" if self.phrase else "discovery"

    def needs_claims(self) -> bool:

        return self.lane == SEAL

    def notes(self) -> str:
        rows: list[str] = []
        if self.ceiling:
            rows.append(f"- spend wall: {self.ceiling}")
        if self.tokens:
            rows.append(f"- closed candidate set ({len(self.tokens)}): {', '.join(self.tokens)}")
        if self.axes:
            rows.append(f"- stated axes, in stated order: {' > '.join(self.axes)}")
        if self.rungs:
            rows.append(f"- relax rungs offered: {' then '.join(self.rungs)}")
        if self.phrase:
            rows.append(f"- listing wording to reuse: {self.phrase}")
        if self.withheld:
            rows.append("- a figure is withheld until you ask for it")
        if self.claim_enum:
            rows.append(f"- live claim fields: {', '.join(self.claim_enum)}")
        if self.claim_rule:
            rows.append(f"- live claim rule: {self.claim_rule[:260]}")
        return ("SILVER SHEET\n" + "\n".join(rows) + "\n") if rows else ""

class CandidateSet:

    __slots__ = ("tokens", "seat", "pid_of", "sku_of", "viewed", "dropped")

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = list(tokens)
        self.seat = {tok: i for i, tok in enumerate(self.tokens)}
        self.pid_of: dict[str, str] = {}
        self.sku_of: dict[str, str] = {}
        for tok in self.tokens:
            pid, _, sku = tok.partition("::")
            self.pid_of[tok] = pid
            self.sku_of[tok] = sku
        self.viewed: set[str] = set()
        self.dropped: set[str] = set()

    def __bool__(self) -> bool:
        return bool(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def holds_pid(self, pid: str) -> bool:
        return bool(pid) and any(self.pid_of[t] == pid for t in self.tokens)

    def note_read(self, pid: str, sku: str = "") -> None:

        if not pid:
            return
        want = _slug(sku)
        for tok in self.tokens:
            if self.pid_of[tok] != pid:
                continue
            have = _slug(self.sku_of[tok])
            if want and have and want != have:
                continue
            self.viewed.add(tok)

    def note_gone(self, pid: str, sku: str = "") -> None:
        want = _slug(sku)
        for tok in self.tokens:
            if self.pid_of[tok] != pid:
                continue
            if want and _slug(self.sku_of[tok]) and want != _slug(self.sku_of[tok]):
                continue
            self.dropped.add(tok)

    def pending(self) -> list[str]:
        return [t for t in self.tokens if t not in self.viewed and t not in self.dropped]

    def live(self) -> list[str]:
        return [t for t in self.tokens if t not in self.dropped]

    def covered(self) -> bool:
        return not self.pending()

    def coverage(self) -> str:
        return f"{len(self.viewed)}/{len(self.tokens)} named candidates read"

class PodiumBoard:

    __slots__ = (
        "turn",
        "limit",
        "acted",
        "ever_acted",
        "cart_pid",
        "cart_sku",
        "committed",
        "closed",
        "owes_reply",
        "asked_withheld",
        "event_at",
        "event_pid",
        "event_sku",
        "cleared_at",
        "reranks",
        "rung",
        "stage",
        "said",
    )

    def __init__(self, limit: int) -> None:
        self.turn = 0
        self.limit = max(1, limit)
        self.acted: list[str] = []
        self.ever_acted: set[str] = set()
        self.cart_pid = ""
        self.cart_sku = ""
        self.committed = False
        self.closed = False
        self.owes_reply = False
        self.asked_withheld = False
        self.event_at = 0
        self.event_pid = ""
        self.event_sku = ""
        self.cleared_at = 0
        self.reranks = 0
        self.rung = 0
        self.stage = 0
        self.said: set[str] = set()

    @property
    def left(self) -> int:
        return max(0, self.limit - self.turn)

    def did(self, name: str) -> bool:
        return name in self.ever_acted

    def once(self, key: str) -> bool:
        if key in self.said:
            return False
        self.said.add(key)
        return True

    def log(self, names: list[str], calls: list[dict[str, Any]]) -> None:
        self.acted = names
        self.ever_acted.update(names)
        if "add_to_cart" in names:
            self.committed = True
            for call in calls:
                if _call_name(call) != "add_to_cart":
                    continue
                args = _call_args(call)
                self.cart_pid = str(args.get("product_id") or self.cart_pid)
                self.cart_sku = str(args.get("sku") or self.cart_sku)
        if "remove_from_cart" in names:
            self.committed = False
        if "place_test_order" in names:
            self.closed = True
        if "message" in names:
            self.owes_reply = False
            self.asked_withheld = True
        if "inspect_stock" in names and self.event_at:
            for call in calls:
                if _call_name(call) != "inspect_stock":
                    continue
                pid = str(_call_args(call).get("product_id") or "")
                if self.event_pid and pid == self.event_pid:
                    self.cleared_at = self.turn
                    self.event_at = 0
                    self.event_pid = ""
                    self.event_sku = ""
                    self.reranks += 1
                    break

    def flag_event(self, pid: str = "", sku: str = "") -> None:
        if self.turn > self.event_at:
            self.event_at = self.turn
        snap = pid or self.cart_pid
        if snap:
            self.event_pid = snap
            self.event_sku = sku or self.cart_sku or self.event_sku

    def event_open(self) -> bool:
        return bool(self.event_at) and not self.closed

    def climb(self) -> int:
        self.rung += 1
        return self.rung

    def advance_stage(self, stage: int) -> bool:
        if stage <= self.stage:
            return False
        self.stage = stage
        return True

class SilverWell:

    __slots__ = ("rows", "blurb", "titles", "last_pid", "focus_pid", "last_price", "last_stock")

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, str]] = {}
        self.blurb: dict[str, str] = {}
        self.titles: dict[str, str] = {}
        self.last_pid = ""
        self.focus_pid = ""
        self.last_price = ""
        self.last_stock = ""

    def keep(self, pid: str, key: str, value: Any) -> None:
        field = _field_key(key)
        if not pid or not field or value is None or isinstance(value, (dict, list)):
            return
        text = str(value).strip()
        if not text:
            return
        if field == "in_stock":
            low = text.lower()
            if low in {"true", "1", "yes", "in_stock"}:
                text = "true"
            elif low in {"false", "0", "no"}:
                text = "false"
            self.last_stock = text
        if field == "price":
            self.last_price = text
        self.rows.setdefault(pid, {})[field] = text
        self.last_pid = pid

    def soak(self, obs: Any) -> None:
        if not isinstance(obs, dict):
            return
        self._walk(obs, "")

    def _walk(self, node: Any, inherited: str) -> None:
        if not isinstance(node, dict):
            return
        pid = str(node.get("product_id") or "") or inherited
        bits: list[str] = []
        title = str(node.get("title") or "")
        if title and pid:
            self.titles[pid] = title
        if title:
            bits.append(title)
        if node.get("options"):
            bits.append(str(node["options"]))
        for key in ("price", "in_stock", "brand", "sku"):
            if key in node:
                self.keep(pid, key, node.get(key))
        facts = node.get("facts")
        if isinstance(facts, dict):
            self._spread(facts, pid, bits, "")
        if pid and bits:
            prior = self.blurb.get(pid, "")
            chunk = " | ".join(bits)
            if chunk not in prior:
                self.blurb[pid] = (prior + " | " + chunk if prior else chunk)[:2200]
        for key in ("observation", "product", "compared", "variants", "results", "items", "candidates"):
            child = node.get(key)
            if isinstance(child, dict):
                self._walk(child, pid)
            elif isinstance(child, list):
                for item in child[:12]:
                    self._walk(item, pid)

    def _spread(self, block: dict[str, Any], pid: str, bits: list[str], prefix: str) -> None:
        for key, val in block.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(val, dict):
                bits.append(f"{path}={json.dumps(val, ensure_ascii=False)[:160]}")
                self.keep(pid, str(key), json.dumps(val, ensure_ascii=False)[:160])
                self._spread(val, pid, bits, path)
            elif isinstance(val, list):
                text = ", ".join(str(x) for x in val[:12])
                bits.append(f"{path}={text}")
                self.keep(pid, str(key), text)
            else:
                text = str(val).strip()
                if not text:
                    continue
                bits.append(f"{path}={text}")
                self.keep(pid, str(key), text)
                if prefix:
                    self.keep(pid, path.rsplit(".", 1)[-1], text)

    def fact_rows(self, pid: str) -> list[dict[str, str]]:
        return [{"key": k, "value": v} for k, v in self.rows.get(pid, {}).items()]

    def attest(self, pid: str) -> str:
        seen = self.rows.get(pid, {})
        return self.blurb.get(pid, "") + " | " + " | ".join(f"{k}={v}" for k, v in seen.items())

    def facts_blob(self, pid: str) -> str:
        return " ".join(f"{k} {v}" for k, v in self.rows.get(pid, {}).items()).lower()

    def title_only_figure(self, pid: str, goal: str) -> bool:

        title = self.titles.get(pid, "").lower()
        if not title or not goal:
            return False
        blob = self.facts_blob(pid)
        return any(_word_hit(t, title) and not _word_hit(t, blob) for t in _spec_tokens(goal))

    def fact_backed(self, pid: str, goal: str) -> bool:
        if not pid or pid not in self.rows:
            return False
        blob = self.facts_blob(pid)
        tokens = _spec_tokens(goal)
        if not tokens:
            return any(k not in _CORE_FIELDS for k in self.rows.get(pid, {}))
        return any(_word_hit(t, blob) for t in tokens)

    def identity_key(self, pid: str) -> str:

        row = self.rows.get(pid, {})
        brand = _slug(row.get("brand", ""))
        title = _slug(self.titles.get(pid, ""))[:48]
        return f"{brand}:{title}" if (brand or title) else pid

    def identity_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for pid in self.rows:
            groups.setdefault(self.identity_key(pid), []).append(pid)
        return groups

    def axis_evidence(self, pid: str, axis: str) -> str:

        want = _slug(axis)
        if not want:
            return ""
        for key, val in self.rows.get(pid, {}).items():
            if want in _slug(key) or _slug(key) in want:
                return val
        for key, val in self.rows.get(pid, {}).items():
            if _word_hit(axis, f"{key} {val}"):
                return val
        return ""

_SEAL_FILL = """
Name the justification claim for ONE shopping requirement.
Input JSON: goal, allowed_fields (live enum, may be empty), schema_rule (the
tool's own wording for legal fields), have (fields already claimed), listing
(observed text for the exact sku), fact_rows (key/value pairs including catalog
drawer keys — those are VALUE evidence only; drawer names vary by category).
Return JSON only: {"claims":[{"field":"...","value":"..."}, ...]}
Rules:
- Cover the specification the shopper stated that `have` is missing.
- Obey schema_rule. When allowed_fields is non-empty, field MUST come from it.
  When it is empty, name the specification field from the shopper's own request
  in snake_case — the conventional attribute for that kind of ask, e.g.
  size/material/color (apparel), size/material/width (shoes), capacity/
  dimensions/finish (bags/home), resolution/screen_size (electronics) —
  NOT the catalog drawer key.
- Catalog drawer keys vary by category; claim field is the conventional
  attribute; value is copied from the answering fact row. Shopper shorthand
  (large, cotton, navy, 720p, 27-inch, …) still maps to that attribute —
  do not invent a different field name for the shorthand.
- value MUST be the printed fact token from listing/fact_rows (verbatim) —
  never a preferred synonym invented from the ask.
- Never invent field names by concatenating drawer+attribute
  (e.g. display_resolution, size_material). Use the conventional attribute alone.
- Never emit display, os, category, title, summary, evidence, or reason as field.
- Do not restate price or in_stock — those are already handled.
- Optional short brand ONLY if a brand fact row evidences it (never invent).
- No guessing: if listing/fact_rows do not evidence the specification, return
  {"claims":[]}.
- At most 2 claims (one gate preferred; optional short brand if evidenced).
""".strip()

class SilverClaims:

    @staticmethod
    def polish(
        raw: Any, brief: SilverBrief, well: SilverWell, pid: str
    ) -> list[dict[str, str]]:
        entries = raw if isinstance(raw, list) else []
        allowed = {_field_key(x) for x in brief.claim_enum}
        seen = well.rows.get(pid, {})
        attest = well.attest(pid)
        kept: list[dict[str, str]] = []
        used: set[str] = set()

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            field = _field_key(entry.get("field"))
            value = "" if entry.get("value") is None else str(entry.get("value")).strip()
            if not field or not value or field in used:
                continue
            if field in _DENY_FIELDS or _drawer_concat(field):
                continue
            if allowed and field not in allowed:
                continue
            if not _FIELD_SHAPE_RX.match(field):
                continue
            if field == "in_stock" and value.lower() in {"true", "1", "yes", "in_stock"}:
                value = "true"
            if field == "brand":
                value = SilverClaims._brand(value, seen, attest)
                if not value or not seen.get("brand"):
                    continue
            elif field not in _CORE_FIELDS and not _printed(value, attest):
                swap = SilverClaims._nearest_printed(value, seen, brief.goal, attest)
                if not swap:
                    continue
                value = swap
            used.add(field)
            kept.append({"field": field, "value": value})

        for need in ("price", "in_stock"):
            if need in used or (allowed and need not in allowed):
                continue
            got = seen.get(need) or (well.last_price if need == "price" else well.last_stock)
            if got:
                kept.insert(0 if need == "price" else min(1, len(kept)), {"field": need, "value": got})
                used.add(need)

        if "brand" not in used and (not allowed or "brand" in allowed):
            brand = SilverClaims._brand(seen.get("brand", ""), seen, attest)
            if brand and seen.get("brand") and _printed(brand, brief.goal):
                kept.append({"field": "brand", "value": brand})
                used.add("brand")

        if brief.needs_claims() and not SilverClaims._has_gate(kept):
            for row in SilverClaims._fill(brief, well, pid, used):
                field = _field_key(row.get("field"))
                value = str(row.get("value") or "").strip()
                if not field or not value or field in used:
                    continue
                if field in _DENY_FIELDS or _drawer_concat(field):
                    continue
                if allowed and field not in allowed:
                    continue
                if not _FIELD_SHAPE_RX.match(field):
                    continue
                if not _printed(value, attest):
                    swap = SilverClaims._nearest_printed(value, seen, brief.goal, attest)
                    if not swap:
                        continue
                    value = swap
                if field == "brand":
                    if not seen.get("brand"):
                        continue
                    value = SilverClaims._brand(value, seen, attest)
                    if not value:
                        continue
                used.add(field)
                kept.append({"field": field, "value": value})
                if SilverClaims._has_gate(kept):
                    break

        if brief.needs_claims() and not SilverClaims._has_gate(kept):
            gate = SilverClaims._gate_from_facts(brief.goal, seen, attest, used, allowed)
            if gate:
                used.add(gate["field"])
                kept.append(gate)

        if (
            brief.needs_claims()
            and "brand" not in used
            and (not allowed or "brand" in allowed)
            and len(kept) < 4
            and seen.get("brand")
        ):
            brand = SilverClaims._brand(seen.get("brand", ""), seen, attest)
            if brand:
                kept.append({"field": "brand", "value": brand})
                used.add("brand")

        return kept[:6]

    @staticmethod
    def _has_gate(kept: list[dict[str, str]]) -> bool:
        fields = {_field_key(c["field"]) for c in kept}
        return bool(fields - _CORE_FIELDS - _DENY_FIELDS - {"brand"})

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
            key = _field_key(raw_key)
            value = str(raw_val or "").strip()
            if not key or not value or key in have:
                continue
            if key in _CORE_FIELDS or key == "brand" or key in _DENY_FIELDS:
                continue
            field = key
            if field in have or field in _DENY_FIELDS or _drawer_concat(field):
                continue
            if allowed and field not in allowed:
                continue
            if not _FIELD_SHAPE_RX.match(field):
                continue
            if not _printed(value, attest):
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
        brief: SilverBrief, well: SilverWell, pid: str, have: set[str]
    ) -> list[dict[str, str]]:
        attest = well.attest(pid)
        if not attest.strip(" |"):
            return []
        enum = [f for f in brief.claim_enum if f not in have]
        payload = {
            "goal": brief.goal[:1500],
            "allowed_fields": enum,
            "schema_rule": brief.claim_rule,
            "have": sorted(have),
            "listing": attest[:2200],
            "fact_rows": well.fact_rows(pid)[:40],
        }
        answer = _WIRE.ask_json(
            engine=_QUICK_ENGINE,
            system=_SEAL_FILL,
            user=json.dumps(payload, ensure_ascii=False),
        )
        rows = _read_json(answer).get("claims")
        if not isinstance(rows, list):
            return []
        allowed = {_field_key(x) for x in enum}
        out: list[dict[str, str]] = []
        for item in rows[:2]:
            if not isinstance(item, dict):
                continue
            field = _field_key(item.get("field"))
            value = str(item.get("value") or "").strip()
            if not field or not value or field in have or field in _DENY_FIELDS:
                continue
            if _drawer_concat(field) or not _FIELD_SHAPE_RX.match(field):
                continue
            if allowed and field not in allowed:
                continue
            if field == "brand" and "brand" not in well.rows.get(pid, {}):
                continue
            out.append({"field": field, "value": value})
        return out

    @staticmethod
    def _nearest_printed(
        value: str, seen: dict[str, str], goal: str, attest: str
    ) -> str:

        if value and _printed(value, attest):
            return value
        ranked: list[tuple[int, str]] = []
        for key, val in seen.items():
            if not val or key in _CORE_FIELDS or key == "brand":
                continue
            if not _printed(val, attest):
                continue
            score = 0
            if _printed(val, goal):
                score += 5
            if value and (_printed(val, value) or _printed(value, val)):
                score += 4
            if key in _DENY_FIELDS or key in {"display", "size", "facts", "options"}:
                score += 1
            if score:
                ranked.append((score, val))
        if not ranked:
            return ""
        ranked.sort(key=lambda row: (-row[0], -len(row[1])))
        return ranked[0][1]

    @staticmethod
    def _brand(raw: str, seen: dict[str, str], attest: str) -> str:

        structured = str(seen.get("brand") or "").strip()
        if not structured or not _printed(structured, attest):
            return ""
        if len(structured.split()) > 4:
            return ""
        text = str(raw or "").strip()
        if text and len(text.split()) <= 3 and len(text) <= 40 and _printed(text, structured):
            return text
        return structured

_CHARTER = """
You are SilverGirl, an EnvPack purchasing operator running a closed-set desk.
Tools are your only sensors. The episode ends when an observation returns
done=true after place_test_order.

LOOP
  Read    search / filter / view — the shopper's printed nouns, not schema keys
  Prove   fact rows decide; titles are advertising and routinely lie
  Commit  add_to_cart one proven, in-budget, in-stock variant
  Confirm inspect_stock after any cart change or market move
  Close   place_test_order

DISCIPLINE
- Never rewrite a query wholesale: change ONE token and look again.
- An unanswered shopper question sinks the episode — reply with message first.
- Keep at least two steps for inspect_stock + place_test_order. Not ordering scores zero.
- Never invent a figure, convert a unit, or promote a title phrase into a fact.
""".strip()

_SEAL_PLAYBOOK = """
FAMILY — SEAL (justification)
place_test_order MUST carry justification.claims as {field, value} pairs.

CLAIM SHAPE (max 4)
  1) price     — copied from the latest view / inspect_stock
  2) in_stock  — usually "true" from inspect_stock
  3) the GATE specification the shopper asked about
       field = the conventional attribute for the ask (size/material/color for
       apparel; size/material/width for shoes; capacity/dimensions/finish for
       bags and home; resolution/screen_size for electronics, among others)
       — never the catalog drawer key, never shopper shorthand as a field name
       value = the VERBATIM printed token from the answering fact row
  4) at most one SOFT reason, attested in the listing/facts — short brand only
       when the shopper named a maker AND a brand fact row prints it

WALLS
- Title-only figures are decoys. No fact row, no cart, no claim.
- Forbidden fields: category, title, summary, evidence, reason, description, os, display.
- Never build a field by gluing drawer+attribute together.
- After a market event: inspect_stock the event-target sku immediately (that
  closes the event); inspecting a replacement does not. Then drop it if invalid,
  rebuild claims from the fresh observation, then order.
- inspect_stock a candidate BEFORE the first add_to_cart (intended commitment).
- Once a proven pick is inspected, order. Do not wait for permission.
""".strip()

_PLAYBOOK: dict[str, str] = {
    PODIUM: """
FAMILY — PODIUM (ranking, Silver specialty)
The named candidates are a CLOSED SET. Nothing outside it can win.

SEQUENCE
1. view every named product_id::sku — all of them, before any cart. A ranking
   built on unread candidates is a guess, and guesses score nothing here.
2. Sort the read set by the shopper's stated priority order (prefer / best /
   highest / largest / lowest language). The first axis decides; later axes only
   break ties. Price is an axis only if stated. First-listed is NOT automatic
   winner — lexicographic rank on the stated axes is.
3. Never cart a dominated candidate (strictly worse on an earlier axis, or
   worse on a later axis while tied earlier). Re-compare the closed set, then
   add_to_cart ONLY the leader, inspect_stock it, then place_test_order.
4. place_test_order carries product_id and sku ONLY. Do NOT attach a
   justification field on a ranking order — optional claims here score against you.

AFTER A MARKET MOVE
Re-rank from scratch over the surviving candidates. The pre-event leader has no
standing. Re-read the moved sku with inspect_stock first, then re-sort, then close.
Never open-search for alternatives: the set is closed.
""".strip(),
    SLATE: """
FAMILY — SLATE (intent)
Work the decision in a fixed order and do not skip ahead.

STEP 1  Write the BINDING walls and the WISH leanings as two separate lists.
STEP 2  Shortlist only items that satisfy every binding wall.
STEP 3  Prove the shortlist from fact rows — one view per serious contender.
STEP 4  Cart the intentional winner: a compliant item that also honours the
        leaning when one exists. A leaning never excuses a broken wall.

The first cart is a statement of intent, not a probe. Do not cart to "see what
happens". Answer shopper questions with message before committing. Omit
justification from the order.
""".strip(),
    ROSTER: """
FAMILY — ROSTER (recall)
Breadth first, then hygiene.

- Query with the shopper's own wording; if a quoted phrase exists, reuse it verbatim.
- Thin results mean one token is wrong — swap that token, do not restart.
- Collect DISTINCT identities. The same product repeated under different skus is
  one entry on the roster, not three. Read brand + title + fact rows to tell
  duplicates from genuinely different items.
- Reject accessories and look-alikes by reading classification fact rows, never
  by trusting the title.
- Prove the pick, cart it, inspect, order. Omit justification.
""".strip(),
    PLINTH: """
FAMILY — PLINTH (constraint)
Facts dominate. Nothing else is admissible.

- If a requirement is withheld until asked, send message and ask BEFORE searching.
- When the shopper names an exact sku or variant, the ordered sku must be that
  one, character for character. A sibling variant is a failure, not a near miss.
- Every binding figure must appear in a fact row on the exact variant you buy.
  A figure that appears only in the title is a decoy — keep viewing.
- Invent no class floors, no implied minimums, no "typical for this category".
- Omit justification unless the environment separately requires it.
""".strip(),
    SCALES: """
FAMILY — SCALES (preference)
Weigh only the axes the shopper stated. Nothing else has weight.

- Hard walls first: budget, stock, category, named requirements. A preference
  never rescues an item that fails a wall.
- Then compare at least two wall-clearing candidates on the stated axes, using
  printed fact values — not on tone, not on title adjectives.
- An axis the shopper never mentioned carries zero weight, however attractive.
- If the catalog is silent on a stated axis for a candidate, say so and lean on
  the next stated axis rather than imagining evidence.
- Cheapness is a tiebreaker only, and only when nothing stated separates the pair.
- Omit justification unless separately required.
""".strip(),
    LADDER: """
FAMILY — LADDER (recovery)
When the wanted item is gone, climb one rung at a time.

- Hold the load-bearing walls: product class, spend wall, availability.
- Concede in the order the shopper offered. If they offered nothing, concede the
  most cosmetic attribute first and the defining one last. One rung per attempt;
  never jump to "anything similar".
- inspect_stock a candidate BEFORE the first add_to_cart (intended commitment).
- After a market move: inspect_stock the event-target sku immediately to close
  the event — inspecting a replacement does not. Then remove if invalid, then
  find a replacement. After every cart change, inspect_stock before ordering.
- State the rung you climbed in your message so the shopper is not surprised.
- Keep the order payload clean: omit justification.
""".strip(),
    SEAL: _SEAL_PLAYBOOK,
}

_PULSE: dict[str, tuple[str, ...]] = {
    PODIUM: (
        "Podium pulse: read every named candidate before ranking; unread candidates cannot be ranked.",
        "Podium pulse: lex-rank on stated priority (highest/largest first, lowest price last). Never cart a dominated candidate.",
        "Podium pulse: order with product_id + sku only — no justification field on a ranking close.",
    ),
    SLATE: (
        "Slate pulse: binding walls list, then wish list, then shortlist. In that order.",
        "Slate pulse: prove the contender from fact rows before the first cart.",
        "Slate pulse: the leaning is a bonus on a compliant item, never a substitute for compliance.",
    ),
    ROSTER: (
        "Roster pulse: reuse the shopper's wording; swap one token when results run thin.",
        "Roster pulse: dedupe by brand + title + facts — repeated skus are one identity.",
        "Roster pulse: classification fact rows separate the real item from accessories.",
    ),
    PLINTH: (
        "Plinth pulse: the exact named variant or nothing; siblings fail.",
        "Plinth pulse: every binding figure must show in a fact row on the bought variant.",
        "Plinth pulse: no invented floors, no category assumptions, no unit conversion.",
    ),
    SCALES: (
        "Scales pulse: walls first, then the stated axes, then price as a tiebreaker.",
        "Scales pulse: compare two wall-clearing candidates on printed values before carting.",
        "Scales pulse: an unstated axis has zero weight no matter how appealing.",
    ),
    LADDER: (
        "Ladder pulse: one rung per attempt, cosmetic before defining.",
        "Ladder pulse: class, spend wall and availability stay fixed while you concede.",
        "Ladder pulse: inspect_stock after every cart change, then close.",
    ),
    SEAL: (
        "Seal pulse: claim field = the conventional attribute for the ask; value = the printed "
        "fact-row token. Drawer keys are evidence, never field names.",
        "Seal pulse: price and in_stock copied from the newest observation; one soft reason at most; "
        "short brand only when a brand fact row prints it.",
        "Seal pulse: forbidden fields — category, title, summary, evidence, reason, os, display. "
        "Never glue drawer+attribute into a field name.",
        "Seal pulse: after a market event inspect_stock the carted sku, rebuild claims, then order.",
        "Seal pulse: a figure printed only in the title is a decoy — keep viewing until a fact row prints it.",
    ),
}

_WAKE = (
    "That reply carried no tool call. Choose a tool and act now.",
    "Still idle. Read → prove → cart → inspect_stock → place_test_order.",
)

def _compose_system(brief: SilverBrief, policy: dict[str, Any]) -> str:
    parts = [_CHARTER, _PLAYBOOK.get(brief.lane, _PLAYBOOK[ROSTER])]
    if brief.needs_claims() and brief.lane != SEAL:
        parts.append(_SEAL_PLAYBOOK)
    sheet = brief.notes()
    if sheet:
        parts.append(sheet.strip())
    parts.append(
        "SESSION\n"
        f"- family: {brief.lane} (gate: {brief.gate})\n"
        f"- tools: {', '.join(_tool_names(policy.get('tools') or [])) or '(schemas)'}\n"
        f"- step limit: {policy.get('max_steps')}\n"
        f"- actions per turn: {policy.get('max_calls_per_turn', 1)}\n"
        f"- optional claims: {'required' if brief.needs_claims() else 'omit entirely'}"
    )
    return "\n\n".join(parts)

def _pulse(lane: str, turn: int) -> str:
    bank = _PULSE.get(lane) or _PULSE[ROSTER]
    return bank[(turn - 1) % len(bank)]

class SilverBridge:
    def __init__(self, client: ProxyClient) -> None:
        self._client = client

    @staticmethod
    def engine_for(lane: str) -> str:
        override = (getenv("SANDBOX_MODEL") or "").strip()
        if override:
            low = override.lower()
            if (
                "kimi" in low
                or "glm" in low
                or low.startswith("z-ai/")
                or low.startswith("moonshotai/")
            ):
                return _HEAVY_ENGINE
            return override
        return _HEAVY_ENGINE if lane in _HEAVY_LANES else _QUICK_ENGINE

    def _ladder(self, engine: str) -> list[str]:
        return [engine, *[m for m in _RELIEF_ENGINES if m != engine]][:3]

    def ask_tools(
        self,
        *,
        engine: str,
        transcript: list[dict[str, Any]],
        schemas: list[Any],
        budget: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        failure: Exception | None = None
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
                failure = RuntimeError(model)
                continue
            try:
                message = payload["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                failure = exc
                continue
            return message.get("content") or "", list(message.get("tool_calls") or [])[:budget]
        raise RuntimeError(f"inference failed ({failure})")

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

    def act(self, envelope: dict[str, Any]) -> dict[str, Any]:
        reply = self._client.post("/environment/call", json_data=envelope)
        if reply is None:
            raise RuntimeError("environment call failed")
        return reply

_WIRE = SilverBridge(ProxyClient(timeout=120, max_retries=2))

def _call_name(call: Any) -> str:
    spec = call.get("function") if isinstance(call, dict) else None
    return str(spec["name"]) if isinstance(spec, dict) and spec.get("name") else ""

def _call_args(call: Any) -> dict[str, Any]:
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

def _digest(obs: Any, cap: int = 6000) -> str:
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

def _market_moved(obs: Any) -> bool:
    if not isinstance(obs, dict):
        return False
    keys = ("market_event", "stock_changed", "price_changed", "event")
    if any(obs.get(k) for k in keys):
        return True
    nested = obs.get("observation")
    return isinstance(nested, dict) and any(nested.get(k) for k in keys)

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

def _is_question(text: str) -> bool:
    body = (text or "").strip().lower()
    if not body:
        return False
    if "?" in body:
        return True
    return body.startswith(
        ("which", "what", "does", "did", "can you", "could you", "would you", "wait", "hold on")
    )

class SilverEpisode:
    def __init__(self, problem_data: dict[str, Any]) -> None:
        env = problem_data["environment"]
        self.binding = env["binding"]
        self.policy = env["policy_view"]
        self.schemas = self.policy.get("tools") or []
        self.ask = str(self.policy.get("query") or problem_data.get("query") or "")
        self.key = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        self.per_turn = max(1, int(self.policy.get("max_calls_per_turn", 1)))
        self.brief = SilverBrief(self.ask, self.schemas)
        self.cands = CandidateSet(self.brief.tokens)
        self.board = PodiumBoard(int(self.policy.get("max_steps", 20)))
        self.well = SilverWell()
        self.engine = SilverBridge.engine_for(self.brief.lane)

    def run(self) -> list[dict[str, Any]]:
        transcript: list[dict[str, Any]] = [
            {"role": "system", "content": _compose_system(self.brief, self.policy)},
            {"role": "user", "content": self.ask},
        ]
        trail: list[dict[str, Any]] = [
            {
                "role": "meta",
                "silvergirl": {
                    "version": 11,
                    "architecture": "ClosedSetPodium",
                    "family": self.brief.lane,
                    "gate": self.brief.gate,
                    "candidates": len(self.cands),
                    "axes": self.brief.axes,
                    "engine": self.engine,
                    "claims": self.brief.needs_claims(),
                    "provider": "openrouter",
                },
            }
        ]
        while self.board.turn < self.board.limit:
            self.board.turn += 1
            line = self._coach_line()
            if line:
                transcript.append({"role": "user", "content": line})
            spoken, calls = self._speak(transcript, trail)
            calls = self._block_order_during_event(calls)
            transcript.append({"role": "assistant", "content": spoken, "tool_calls": calls})
            result = self._dispatch(calls, self.board.turn)
            trail.append(
                {
                    "role": "assistant",
                    "content": spoken,
                    "tool_calls": calls,
                    "environment_result": result,
                    "silvergirl_turn": {
                        "family": self.brief.lane,
                        "turn": self.board.turn,
                        "coverage": self.cands.coverage() if self.cands else "",
                        "rung": self.board.rung,
                        "reranks": self.board.reranks,
                    },
                }
            )
            self.board.log([_call_name(c) or "?" for c in calls], calls)
            self._absorb(result, transcript, calls)
            if _episode_done(result):
                break
        return trail

    def _block_order_during_event(
        self, calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not self.board.event_open() or self.board.closed:
            return calls
        event_pid = self.board.event_pid
        event_sku = self.board.event_sku
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
                or self.board.cart_pid
                or self.well.focus_pid
                or self.well.last_pid
                or ""
            )
            sku = str(args.get("sku") or self.board.cart_sku or "")
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

    def _coach_line(self) -> str | None:
        board = self.board
        if board.owes_reply:
            return "The shopper asked you something. Reply with message this turn, before anything else."
        if self.brief.withheld and not board.asked_withheld and board.turn == 1:
            return (
                "A figure is withheld until asked. Send message requesting it now — "
                "searching before you know it wastes the episode."
            )
        if board.event_open():
            return self._event_line()
        lane = self.brief.lane
        if lane == PODIUM:
            line = self._podium_coach()
        elif lane == SLATE:
            line = self._slate_coach()
        elif lane == ROSTER:
            line = self._roster_coach()
        elif lane == PLINTH:
            line = self._plinth_coach()
        elif lane == SCALES:
            line = self._scales_coach()
        elif lane == LADDER:
            line = self._ladder_coach()
        else:
            line = self._seal_coach()
        if line:
            return line
        return self._tail_line()

    def _event_line(self) -> str:
        board = self.board
        target = board.event_pid or board.cart_pid or "the moved sku"
        if self.brief.lane == PODIUM:
            live = ", ".join(self.cands.live()[:6])
            return (
                "Market move on a ranked candidate. Re-rank from scratch: the pre-event "
                f"leader has no standing. inspect_stock the event-target ({target}) "
                "to close the event — inspecting a replacement does not — "
                f"drop it if it is invalid, re-sort the survivors ({live or 'the read set'}) "
                "by the stated axes, then cart and close with product_id + sku only."
            )
        if self.brief.lane == LADDER:
            rung = board.climb() if board.once(f"rung-{board.turn}") else board.rung
            offered = self.brief.rungs[rung - 1] if 0 < rung <= len(self.brief.rungs) else ""
            target_concede = f"Concede '{offered}' next." if offered else (
                "Concede the most cosmetic attribute next; keep class, spend wall and stock."
            )
            return (
                f"The wanted item moved — this is ladder rung {rung}. {target_concede} "
                f"inspect_stock the event-target ({target}) to close the event, remove "
                "it if dead, then search one rung down and close."
            )
        if self.brief.needs_claims():
            return (
                f"Market move. inspect_stock the event-target ({target}) to close the "
                "event — a replacement inspect does not. Then remove it if it no longer "
                "qualifies, rebuild every claim from the fresh observation, then order."
            )
        return (
            f"Market move. inspect_stock the event-target ({target}) to close the event, "
            "remove it if it no longer satisfies a wall, re-pick if needed, then "
            "place_test_order."
        )

    def _podium_scorecard(self) -> str:

        plan = self.brief.axis_plan()
        rows: list[str] = []
        for tok in self.cands.live():
            if tok not in self.cands.viewed:
                continue
            pid = self.cands.pid_of[tok]
            if plan:
                cells = []
                for sense, axis in plan[:3]:
                    val = self.well.axis_evidence(pid, axis) or "not printed"
                    cells.append(f"{sense}:{axis}={val}")
                rows.append(f"  - {tok}: " + "; ".join(cells))
            else:
                price = self.well.rows.get(pid, {}).get("price", "?")
                stock = self.well.rows.get(pid, {}).get("in_stock", "?")
                rows.append(f"  - {tok}: price={price}; in_stock={stock}")
        return "\n".join(rows) if rows else "  (no printed values yet)"

    def _podium_coach(self) -> str | None:
        board, cands = self.board, self.cands
        if not cands:
            if board.once("podium-open"):
                return (
                    "No candidate tokens in the ask, so build the set yourself: search once, "
                    "shortlist the distinct hits, then rank them by the stated axes in order."
                )
            return None
        pending = cands.pending()
        if pending and not board.committed:
            head = ", ".join(pending[:5])
            return (
                f"Closed set: {cands.coverage()}. Still unread: {head}. view the next unread "
                "named product_id::sku now — no add_to_cart, no filtering, no open search "
                "until every named candidate has been read. A rank over unread candidates is a guess."
            )
        if pending and board.committed and board.once("podium-late-read"):
            return (
                f"You carted before finishing the reads ({cands.coverage()}). view the remaining "
                f"named candidate(s) {', '.join(pending[:4])}, then confirm or swap the cart."
            )

        plan = self.brief.axis_plan()
        if plan:
            axes = " > ".join(
                f"{'highest' if s == 'max' else 'lowest'} {a}" for s, a in plan
            )
        else:
            axes = (
                " > ".join(self.brief.axes)
                if self.brief.axes
                else "the shopper's stated priority (prefer/best/rank language)"
            )
        scorecard = self._podium_scorecard()

        if not board.committed:
            if board.once("podium-rank"):
                return (
                    f"Every named candidate is read ({cands.coverage()}). Lex-sort by [{axes}] — "
                    "first axis decides, later axes only break ties. Do NOT cart a dominated "
                    f"candidate (strictly worse on an earlier axis). Scorecard:\n{scorecard}\n"
                    "add_to_cart the leader now."
                )
            return (
                f"Cart the leader of the read set (sorted by [{axes}]) — never a dominated "
                "candidate — then inspect_stock."
            )
        if not board.did("inspect_stock") and board.once("podium-inspect"):
            return (
                "Leader is carted. inspect_stock it, then place_test_order with product_id + sku only."
            )
        if not board.closed and board.once("podium-close"):
            return (
                "Close now: place_test_order with product_id and sku only. Do NOT attach a "
                "justification field — optional claims on a ranking order count against you."
            )
        return None

    def _slate_coach(self) -> str | None:
        board = self.board
        stage = self._slate_stage()
        if not board.advance_stage(stage):
            return None
        if stage == 1:
            wish = "the leaning" if _WISH_RX.search(self.brief.goal) else "any soft leaning"
            wall = f" Spend wall: {self.brief.ceiling}." if self.brief.ceiling else ""
            return (
                "Stage 1 of 4 — split the ask. Write the BINDING walls in one list and "
                f"{wish} in a second list.{wall} Then search with the binding wording only."
            )
        if stage == 2:
            return (
                "Stage 2 of 4 — shortlist. Keep only hits that satisfy every binding wall. "
                "Drop anything that merely looks close; do not cart yet."
            )
        if stage == 3:
            return (
                "Stage 3 of 4 — prove. view the serious contenders and read their fact rows. "
                "A wall is satisfied only when a fact row says so on that exact variant."
            )
        if stage == 4:
            return (
                "Stage 4 of 4 — commit with intent. Cart the compliant item that also honours "
                "the leaning; if nothing compliant offers the leaning, cart the compliant item "
                "anyway and say so. Then inspect_stock and place_test_order without justification."
            )
        return None

    def _slate_stage(self) -> int:
        board = self.board
        if not (board.did("search") or board.did("filter") or board.did("message")):
            return 1
        if not board.did("view") and not board.did("compare"):
            return 2
        if not board.committed:
            proven = any(
                self.well.fact_backed(pid, self.brief.goal) for pid in self.well.rows
            )
            return 4 if proven else 3
        return 0

    def _roster_coach(self) -> str | None:
        board = self.board
        groups = self.well.identity_groups()
        dupes = {k: v for k, v in groups.items() if len(v) > 1}
        if not board.did("search") and not board.did("filter") and board.once("roster-open"):
            seed = f'Reuse the quoted wording "{self.brief.phrase}" verbatim.' if self.brief.phrase else (
                "Query with the shopper's own nouns."
            )
            return f"Open the roster: search first. {seed} Breadth now, hygiene next."
        if dupes and not board.committed and board.once("roster-dupes"):
            sample = "; ".join(
                f"{' / '.join(pids[:3])}" for pids in list(dupes.values())[:2]
            )
            return (
                f"Identity hygiene: {len(dupes)} duplicate identity group(s) on the roster "
                f"({sample}). Collapse each group to ONE entry before comparing — repeated skus "
                "of the same product are not separate options."
            )
        distinct = len(groups)
        if distinct == 1 and board.did("search") and not board.committed and board.once("roster-thin"):
            return (
                "Only one distinct identity on the roster. Swap exactly ONE token in the query "
                "and search again — do not rewrite the whole query — then read the new hits."
            )
        if distinct >= 2 and not board.did("view") and board.once("roster-prove"):
            return (
                f"{distinct} distinct identities collected. view the strongest ones and read "
                "classification fact rows to separate the real item from accessories and look-alikes."
            )
        if board.committed and not board.closed and board.once("roster-close"):
            return "Roster settled. inspect_stock the carted sku, then place_test_order without justification."
        return None

    def _plinth_coach(self) -> str | None:
        board = self.board
        named = self.cands.tokens
        pid = self.well.focus_pid or self.well.last_pid
        if named and not board.committed and board.once("plinth-exact"):
            token = named[0]
            return (
                f"The shopper named {token}. The ordered sku must be exactly that — a sibling "
                "variant is a failure, not a near miss. view it, confirm each binding figure in "
                "its fact rows, then cart that exact sku."
            )
        if board.committed and named:
            want_pid = self.cands.pid_of[named[0]]
            want_sku = self.cands.sku_of[named[0]]
            drift = (board.cart_pid and board.cart_pid != want_pid) or (
                want_sku and board.cart_sku and _slug(board.cart_sku) != _slug(want_sku)
            )
            if drift and board.once("plinth-drift"):
                return (
                    f"The cart holds {board.cart_pid}::{board.cart_sku} but the shopper demanded "
                    f"{named[0]}. Remove the wrong variant and cart the exact named sku."
                )
        if (
            pid

            and not self.cands.holds_pid(pid)
            and self.well.title_only_figure(pid, self.brief.goal)
            and board.once("plinth-decoy")
        ):
            return (
                "Title decoy: the asked figure is printed in the title of the current product but "
                "in none of its fact rows. Do not cart it. Keep viewing until a fact row prints "
                "the figure on the exact variant."
            )
        if (
            pid
            and board.did("view")
            and not board.committed
            and not self.well.fact_backed(pid, self.brief.goal)
            and board.once("plinth-unproven")
        ):
            return (
                "Nothing in the viewed fact rows answers the binding requirement yet. Do not "
                "assume a category default or convert a unit — view another variant until a fact "
                "row states it."
            )
        if board.committed and not board.did("inspect_stock") and board.once("plinth-inspect"):
            return "Proven and carted. inspect_stock the exact sku, then place_test_order."
        return None

    def _scales_coach(self) -> str | None:
        board = self.board
        axes = self.brief.axes
        if not axes and board.once("scales-noaxis"):
            return (
                "No axis was stated explicitly, so weigh nothing beyond the walls: satisfy every "
                "hard requirement, compare two compliant candidates on printed facts, and let "
                "price break the tie only if nothing stated separates them."
            )
        readable = [pid for pid in self.well.rows if self.well.fact_backed(pid, self.brief.goal)]
        if axes and not board.committed and len(readable) < 2 and board.did("view"):
            if board.once("scales-second"):
                return (
                    f"Only {len(readable)} wall-clearing candidate has been read. view a second one "
                    f"before carting — the stated axes ({' > '.join(axes)}) cannot be weighed "
                    "against a single option."
                )
            return None
        if axes and len(readable) >= 2 and not board.committed and board.once("scales-weigh"):
            lines: list[str] = []
            for pid in readable[:3]:
                cells = [f"{axis}={self.well.axis_evidence(pid, axis) or 'not printed'}" for axis in axes[:3]]
                lines.append(f"{pid}: " + ", ".join(cells))
            return (
                f"Weigh on the stated axes in order ({' > '.join(axes)}) using printed values only:\n"
                + "\n".join(lines)
                + "\nAn axis the shopper never named has zero weight. Where a value is not printed, "
                "move to the next stated axis instead of guessing. Then cart the winner."
            )
        if board.committed and not board.closed and board.once("scales-close"):
            return (
                "Winner carted on the stated axes. inspect_stock it, then place_test_order with no "
                "justification field."
            )
        return None

    def _ladder_coach(self) -> str | None:
        board = self.board
        rungs = self.brief.rungs
        if board.once("ladder-open"):
            keep = ["product class", "availability"]
            if self.brief.ceiling:
                keep.insert(1, f"spend wall {self.brief.ceiling}")
            plan = (
                "Concede in this order: " + " then ".join(rungs) + "."
                if rungs
                else "The shopper offered no concession order, so concede the most cosmetic "
                "attribute first and the defining one last."
            )
            return (
                f"Recovery run. Fixed and non-negotiable: {', '.join(keep)}. {plan} "
                "One rung per attempt — never jump to 'anything similar'."
            )
        if board.did("search") and not board.committed and not board.did("view") and board.once("ladder-read"):
            return (
                "Read before conceding again: view the surviving hits and confirm the fixed walls "
                "hold on the exact variant. A concession is only valid if the walls still stand."
            )
        if (
            not board.committed
            and board.did("view")
            and not board.did("inspect_stock")
            and board.once("ladder-precart")
        ):
            return (
                "Before the first add_to_cart: inspect_stock the candidate so the "
                "environment records an intended commitment."
            )
        if not board.committed and board.left <= max(3, board.limit // 3) and board.once("ladder-hurry"):
            rung = board.climb()
            nxt = rungs[rung - 1] if 0 < rung <= len(rungs) else "the next most cosmetic attribute"
            return (
                f"Steps are short. Climb to rung {rung} now — concede {nxt} — inspect_stock "
                "BEFORE add_to_cart, cart the best surviving item that keeps the fixed walls, "
                "and close."
            )
        if board.committed and not board.closed and board.once("ladder-close"):
            return (
                "Recovery pick carted. Say which rung you climbed in your message, inspect_stock "
                "the sku, then place_test_order with a clean payload."
            )
        return None

    def _seal_coach(self) -> str | None:
        board = self.board
        if board.closed:
            return None
        pid = self.well.focus_pid or self.well.last_pid
        if pid and not board.committed and self.well.title_only_figure(pid, self.brief.goal):
            if board.once("seal-decoy"):
                return (
                    "Title decoy on the current product: the asked figure appears in the title but "
                    "in no fact row. Do not cart it — a claim built on title text is untruthful and "
                    "scores zero. Keep viewing."
                )
        if (
            pid
            and board.did("view")
            and not board.committed
            and not self.well.fact_backed(pid, self.brief.goal)
            and board.once("seal-proof")
        ):
            return (
                "First commitment must be fact-backed: cart only when a fact row overlaps the "
                "shopper's ask. Keep viewing until one does."
            )
        if (
            not board.committed
            and board.did("view")
            and board.left <= max(board.limit // 2, 5)
            and board.once("seal-midpoint")
        ):
            return (
                "Midpoint: if a viewed sku already prints the gate specification and clears budget "
                "and stock, cart it now and leave room for inspect_stock + place_test_order. Stop "
                "hunting synonyms."
            )
        return self._seal_pulse()

    def _seal_pulse(self) -> str:
        base = _pulse(SEAL, self.board.turn)
        extra: list[str] = []
        if self.brief.claim_enum:
            extra.append(f"Allowed claim fields this episode: {', '.join(self.brief.claim_enum)}.")
        if self.brief.claim_rule:
            extra.append(f"Schema rule: {self.brief.claim_rule[:180]}")
        pid = self.well.focus_pid or self.well.last_pid
        if pid:
            rows = self.well.rows.get(pid, {})
            if rows:
                sample = ", ".join(f"{k}={v}" for k, v in list(rows.items())[:8])
                extra.append(f"Fact sample for {pid}: {sample}")
            if self.well.last_price or self.well.last_stock:
                extra.append(
                    f"Freshest price={self.well.last_price or '?'} "
                    f"in_stock={self.well.last_stock or '?'}."
                )
        return base + (("\n" + "\n".join(extra)) if extra else "")

    def _tail_line(self) -> str | None:
        board = self.board
        if board.turn <= 2 and not any(
            board.did(name) for name in ("search", "filter", "view", "compare", "message")
        ):
            return "Open with search or filter using the shopper's own nouns, then view the hits."
        if self.brief.lane == PODIUM and self.cands.pending():

            return None
        if board.committed and not board.closed and board.left <= 6 and board.once("deadline"):
            return (
                f"Only {board.left} step(s) remain. inspect_stock if unconfirmed, then "
                "place_test_order — an unplaced order scores nothing."
            )
        if not board.closed and board.left <= 1 and board.once("last-step"):
            return "Last step. place_test_order on the best proven candidate now."
        if not board.committed and board.left <= 8 and board.did("view") and board.once("hurry-cart"):
            return (
                "Step budget is low. Cart the best proven candidate NOW and leave "
                "room for inspect_stock + place_test_order."
            )
        if board.turn > 2 and board.once(f"pulse-{board.turn}") and board.turn % 2 == 1:
            return _pulse(self.brief.lane, board.turn)
        return None

    def _shape(self, name: str, args: dict[str, Any]) -> dict[str, Any]:

        if name in {"view", "inspect_stock", "compare", "add_to_cart"}:
            pid = str(args.get("product_id") or "")
            if pid:
                self.well.focus_pid = pid
            return args
        if name != "place_test_order":
            return args
        payload = dict(args)

        if len(self.brief.tokens) >= 2:
            payload.pop("justification", None)
            payload.pop("claims", None)
            return payload
        if self.brief.lane != SEAL:

            payload.pop("justification", None)
            payload.pop("claims", None)
            return payload
        pid = str(
            payload.get("product_id")
            or self.board.cart_pid
            or self.well.focus_pid
            or self.well.last_pid
        )
        note = payload.get("justification")
        raw = note.get("claims") if isinstance(note, dict) else None
        payload["justification"] = {"claims": SilverClaims.polish(raw, self.brief, self.well, pid)}
        if pid and not payload.get("product_id"):
            payload["product_id"] = pid
        return payload

    def _dispatch(self, calls: list[dict[str, Any]], turn: int) -> dict[str, Any]:
        batch = f"{self.key}-turn-{turn}"
        actions: list[dict[str, Any]] = []
        for slot, call in enumerate(calls, start=1):
            name = _call_name(call)
            if not name:
                raise RuntimeError("nameless tool call")
            args = self._shape(name, _call_args(call))
            spec = call.get("function")
            if isinstance(spec, dict):
                spec["arguments"] = json.dumps(args)
            actions.append({"call_id": f"{batch}-{slot}", "action": {"name": name, "args": args}})
        return _WIRE.act(
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
        for attempt in range(len(_WAKE) + 1):
            spoken, calls = _WIRE.ask_tools(
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
            self.well.soak(obs)
            if isinstance(obs, dict) and not obs.get("error"):

                if _call_name(call) in {"view", "compare", "inspect_stock"}:
                    args = _call_args(call)
                    self.cands.note_read(
                        str(args.get("product_id") or ""), str(args.get("sku") or "")
                    )
            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": _call_id(call, f"{self.key}-{self.board.turn}-{index}"),
                    "content": _digest(obs),
                }
            )
            if _market_moved(obs):
                ep, es = _event_target(obs)
                snap = (
                    ep
                    or self.board.cart_pid
                    or self.well.focus_pid
                    or self.well.last_pid
                    or ""
                )
                self.board.flag_event(pid=snap, sku=es or self.board.cart_sku)

        for pid, facts in self.well.rows.items():
            if not self.cands.holds_pid(pid):
                continue
            if facts.get("in_stock") == "false":
                self.cands.note_gone(pid)
                continue
            self.cands.dropped.difference_update(
                {t for t in self.cands.tokens if self.cands.pid_of[t] == pid}
            )
            if len(facts) > 1:
                self.cands.note_read(pid)
        note = result.get("user_message")
        if isinstance(note, dict) and note.get("content"):
            text = str(note["content"])
            transcript.append({"role": "user", "content": text})
            if _is_question(text):
                self.board.owes_reply = True

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    return SilverEpisode(problem_data).run()

__all__ = ["agent_main"]
