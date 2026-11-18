"""ORO-Bench shopping agent: a candidate × gate SAT matrix.

Buying is filling a table. Each row is one catalog offer (product_id::sku).
Each column is a named gate: live stock-and-budget, viewed page, category,
brand, device identity, hardware spec, theme words, listing diversity,
justification fact-claim, ranking named-key, and a live stock inspection.
Every cell starts as unknown, then becomes pass or fail as public evidence
arrives. An unknown cell is not a guess — it is the next tool call: view
this id, inspect this sku, or ask the shopper. The closer fires only when
a row is all-pass. Preference leaves the model to pick among all-pass rows
until late; ranking fills every named-key row before any add; recovery
keeps the first all-pass at relax 0 until the hunt is exhausted; intent
stays on a preferred brand while that brand still has an all-pass row.
Evidence is read the way the environment grades it: labelled raw facts,
then the selected variant's option string, then the title. Conflict
collapses to none. The model explores; the matrix decides when a row
is actually buyable.
"""

from __future__ import annotations
import json
import re
import time
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
ROSTER = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
PIPE = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
HORIZON_S = 270.0
OBS_CAP = 7000
NUDGE_CAP = 2
MONEY_EPS = 1e-9
MONEY_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
LANE_SIGNS = (
    ("preference", "listing_id"),
    ("constraint", "exact variant"),
    ("justification", "trustworthy reason"),
    ("intent", "nice-to-haves"),
)
GATE_ORDER = (
    "live",
    "viewed",
    "category",
    "brand",
    "device",
    "spec",
    "theme",
    "listing_diversity",
    "fact_claim",
    "named_key",
    "inspected",
)
LANE_HINTS = {
    "intent": "Firm gates: category, named hardware, budget, stock. A named brand is soft: keep it while an all-pass row of that brand exists. Filter then view so option text fills the spec cell. Answer with message. A later lower budget applies to every row.",
    "recall": "Search several wordings (k=10, in_stock=true) until enough titles contain every requested word as a whole word. Then view those rows so the matrix can pass them.",
    "constraint": "Ask for the hidden hardware value, then treat it as a hard gate on the variant option text. Filter and view. If a carted variant dies, switch products, not just color.",
    "preference": "Use the shopper's priority answer. Compare different listing_id values. The first cart add is graded. Recover to a different listing_id if the carted row is repriced.",
    "ranking": "Only named product_id::sku pairs are real rows. Compare them all so every named-key cell is filled before any add. Highest refresh, then largest screen, then lowest price.",
    "recovery": "Search the model code and view every color. The first all-pass commit must be relax 0. If it dies, same model other color before any other model.",
    "justification": "Need labelled facts that state the spec, plus category, brand, stock and budget. View first. Claims come only from that evidence.",
    "generic": "Surface rows, resolve unknown cells, let the closer buy the elected all-pass offer.",
}
SHOP_RULES = (
    "Propose exactly one tool call per turn. A SAT gate matrix scores stock, budget, page evidence, "
    "category, brand, device, spec, theme, listing diversity, fact claims, named keys, and inspection. "
    "Surface rows and resolve unknown cells (search/filter, then view/compare). Never add a placeholder; "
    "the first cart add is graded. Never send status updates. Never ask to raise the budget. Answer a "
    "direct shopper question with message first. Do not repeat identical tool arguments."
)

def allowed_roster() -> tuple:
    provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    names = ROSTER.get(provider) or ROSTER["chutes"]
    pinned = getenv("SANDBOX_MODEL")
    if pinned:
        names = (pinned,) + tuple(item for item in names if item != pinned)
    return names
def money(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
def decode_args(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}
def clip_obs(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + ' ... (truncated by agent)"}'
def words_of(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))
def noun_match(left: str, right: str) -> bool:
    a = re.sub(r"[^a-z0-9]+", "", (left or "").lower()).rstrip("s")
    b = re.sub(r"[^a-z0-9]+", "", (right or "").lower()).rstrip("s")
    return bool(a) and a == b
class SpecReader:
    """Facts, then option text, then title. Conflict -> none. RAM needs an adjacent memory word."""
    INCH = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
    HERTZ = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
    WXH = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
    GIGS = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
    PAIR = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
    PAIR_HINT = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
    MEM = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
    QUARTER = re.compile(r"\bqHD\b")
    OS = re.compile(
        r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian|harmony\s*os|"
        r"hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
        re.I,
    )
    IS_PHONE = re.compile(r"\bsmart\s*phones?\b", re.I)
    NOT_PHONE = re.compile(
        r"\btablets?\b|\b(?:feature|basic)\s*phones?\b|\b(?:mystery|blind|lucky)\s*box(?:es)?\b|"
        r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
        r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b",
        re.I,
    )
    PART_ONLY = re.compile(
        r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
        r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
        r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b",
        re.I,
    )
    NEGATED = re.compile(
        r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b",
        re.I,
    )
    BUNDLE = re.compile(
        r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
        r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s*protector|charger|accessor(?:y|ies))\b))",
        re.I,
    )
    LABEL = {"screen_size": "size", "resolution": "display", "refresh_rate": "refresh_rate", "ram": "storage"}
    UNIT = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}
    @classmethod
    def inches(cls, blob: str) -> set:
        found = set()
        for hit in cls.INCH.finditer(blob or ""):
            found.add("%gin" % float(hit.group(1)))
        return found
    @classmethod
    def hertz(cls, blob: str) -> set:
        found = set()
        for hit in cls.HERTZ.finditer(blob or ""):
            found.add("%ghz" % float(hit.group(1)))
        return found
    @classmethod
    def gigs(cls, blob: str) -> set:
        found = set()
        for hit in cls.GIGS.finditer(blob or ""):
            scale = 1024 if hit.group(2).lower() == "tb" else 1
            found.add("%ggb" % (float(hit.group(1)) * scale))
        return found
    @classmethod
    def panel_class(cls, long_side: int, short_side: int) -> str | None:
        if long_side >= 3840 and short_side >= 2160:
            return "4k"
        if long_side >= 2560 and short_side >= 1440:
            return "qhd"
        if long_side >= 1920 and short_side >= 1080:
            return "fhd"
        if long_side >= 1280 and short_side >= 720:
            return "hd"
        return None
    @classmethod
    def pixels(cls, blob: str) -> set:
        blob = blob or ""
        if cls.QUARTER.search(blob):
            return set()
        low = blob.lower()
        found = set()
        for hit in cls.WXH.finditer(low):
            sides = [int(hit.group(1)), int(hit.group(2))]
            sides.sort(reverse=True)
            kind = cls.panel_class(sides[0], sides[1])
            if kind:
                found.add(kind)
        if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
            found.add("4k")
        if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
            found.add("qhd")
        if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
            found.add("fhd")
        stripped = re.sub(r"\b(?:full|ultra) hd\b", "", low)
        if re.search(r"\b(?:hd|720p?)\b", stripped):
            found.add("hd")
        return found
    @classmethod
    def harvest(cls, field: str, blob: str) -> set:
        if field == "screen_size":
            return cls.inches(blob)
        if field == "refresh_rate":
            return cls.hertz(blob)
        if field == "resolution":
            return cls.pixels(blob)
        return cls.gigs(blob)
    @classmethod
    def near(cls, field: str, blob: str) -> set:
        blob = blob or ""
        if field != "ram":
            return cls.harvest(field, blob)
        paired = set()
        for left, _right in cls.PAIR.findall(blob):
            paired.add("%ggb" % float(left))
        if paired:
            return paired
        mem = set()
        for num in cls.MEM.findall(blob):
            mem.add("%ggb" % float(num))
        return mem
    @classmethod
    def talks(cls, field: str, blob: str) -> bool:
        blob = blob or ""
        if field == "ram":
            return bool(cls.PAIR_HINT.search(blob) or cls.MEM.search(blob))
        if field == "screen_size":
            return bool(cls.INCH.search(blob))
        if field == "refresh_rate":
            return bool(cls.HERTZ.search(blob))
        if cls.WXH.search(blob):
            return True
        return bool(
            re.search(
                r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b",
                blob,
                re.I,
            )
        )
    @classmethod
    def lone(cls, values: set) -> str | None:
        if len(values) != 1:
            return None
        return next(iter(values))
    @classmethod
    def asked(cls, field: str | None, blob: str) -> str | None:
        if not field:
            return None
        return cls.lone(cls.harvest(field, blob or ""))
    @classmethod
    def stems(cls, blob: str) -> set:
        out = set()
        for piece in re.findall(r"[a-z]+", (blob or "").lower()):
            out.add(piece.rstrip("s"))
        return out
    @classmethod
    def scrub_identity(cls, blob: str, title: bool = False) -> str:
        cleaned = cls.NEGATED.sub("", blob or "")
        if not title:
            return cleaned
        return cls.BUNDLE.split(cleaned, maxsplit=1)[0]
    @classmethod
    def handset(cls, facts: dict, title: str, options: str) -> str:
        if cls.PART_ONLY.search(options or ""):
            return "no"
        facts = facts if isinstance(facts, dict) else {}
        raw_title = str(title or "")
        head = cls.scrub_identity(raw_title, title=True)
        kind_bits = []
        for key in ("phone_type", "device_type", "description_device_type"):
            kind_bits.append(cls.scrub_identity(str(facts.get(key) or "")))
        kind = " ".join(kind_bits)
        blob = head + " " + kind
        if cls.NOT_PHONE.search(blob):
            return "no"
        for key in ("phone_type", "device_type", "device_operating_system"):
            if "[truncated]" in str(facts.get(key) or ""):
                return "unclear"
        if cls.IS_PHONE.search(blob):
            return "yes"
        os_bits = []
        for key in ("device_operating_system", "description_operating_system"):
            raw = str(facts.get(key) or "")
            if re.search(r"\bcompatib", raw, re.I):
                continue
            os_bits.append(cls.scrub_identity(raw))
        if cls.OS.search(" ".join(os_bits)):
            return "yes"
        if cls.OS.search(head) and not re.search(r"\bcompatib", raw_title, re.I):
            return "yes"
        return "unclear"
class Need:
    """Shopper request turned into the gates the matrix will fill."""
    BUDGET = re.compile(
        r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
        r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
        + MONEY_NUM
        + r"\s*(PHP|php|₱|pesos?)?",
        re.I,
    )
    KEY = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
    QUOTED = re.compile(r"\"([^\"]{2,80})\"")
    PHRASE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
    SURFACE = re.compile(r"surface at least (\d+) distinct", re.I)
    PREF_ITEM = re.compile(r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
    PREF_BRAND = re.compile(
        r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)"
    )
    CAT_PHRASE = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
    SPEC_WORD = re.compile(
        r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
        r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b",
        re.I,
    )
    WHY = re.compile(r"\b(?:why|reason|explain|justif)", re.I)
    ASK = re.compile(
        r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
        r"|tell me (?:which|whether|if)|confirm whether)\b",
        re.I,
    )
    def __init__(self, query: str) -> None:
        self.raw = query or ""
        goal, _sep, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.ceiling = self.take_budget(self.goal)
        self.currency = "PHP"
        iso = re.search(r"\b([A-Z]{3})\b", self.goal)
        if iso and iso.group(1) in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
            self.currency = iso.group(1)
        self.named: list = []
        for pid, sku in self.KEY.findall(self.goal):
            self.named.append((pid, sku.rstrip(".,;:")))
        phrase = self.PHRASE.search(self.goal)
        quoted = self.QUOTED.findall(self.goal)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        self.theme = re.findall(r"[a-z0-9]+", source.lower())
        surface = self.SURFACE.search(self.rules)
        self.need_ids = int(surface.group(1)) if surface else 0
        pref = self.PREF_ITEM.search(self.goal)
        self.want_color = pref.group(1).lower() if pref else None
        self.want_model = pref.group(2) if pref else None
        self.category = self.take_category(self.goal)
        self.brand = self.take_brand(self.goal, self.category)
        self.want_brand = None if pref else self.take_pref_brand(self.goal)
        self.shape = self.take_shape(self.goal)
        self.specs = self.take_specs(self.goal)
        self.spec_field, self.spec_value = self.take_gate_spec()
        self.sealed = self.spec_field is None and "firm requirement" in self.goal.lower()
        self.family = self.sniff_lane()
    def sniff_lane(self) -> str:
        if self.named:
            return "ranking"
        if self.need_ids:
            return "recall"
        if self.want_model:
            return "recovery"
        for lane, needle in LANE_SIGNS:
            if needle in self.rules:
                return lane
        if self.theme:
            return "recall"
        if self.WHY.search(self.goal):
            return "justification"
        if self.want_brand:
            return "intent"
        if self.sealed or self.spec_field or self.brand:
            return "constraint"
        return "preference"
    @staticmethod
    def take_budget(text: str) -> float | None:
        best = None
        for hit in Need.BUDGET.finditer(text or ""):
            whole = hit.group(1).replace(",", "")
            frac = hit.group(2) or "0"
            try:
                val = float(whole + "." + frac)
            except ValueError:
                continue
            if best is None or val < best:
                best = val
        return best
    @staticmethod
    def take_category(goal: str) -> str | None:
        hit = Need.CAT_PHRASE.search(goal or "")
        if hit:
            return hit.group(1).strip()
        hit = re.search(
            r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
            goal or "",
            re.I,
        )
        return hit.group(1) if hit else None
    @staticmethod
    def take_brand(goal: str, category: str | None) -> str | None:
        if not category:
            return None
        hit = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category), goal or "", re.I)
        if not hit:
            return None
        keep = []
        for token in re.findall(r"[\w&.'\-]+", Need.SPEC_WORD.sub(" ", hit.group(1))):
            if token and token[:1].isupper() and not any(ch.isdigit() for ch in token) and not SpecReader.pixels(token):
                keep.append(token)
        return " ".join(keep) or None
    @staticmethod
    def take_pref_brand(goal: str) -> str | None:
        hit = Need.PREF_BRAND.search(goal or "")
        if not hit:
            return None
        name = hit.group(1).strip(" .,")
        bits = name.split()
        if not bits or len(bits) > 4 or not any(bit[:1].isupper() for bit in bits):
            return None
        return name
    @staticmethod
    def take_shape(goal: str) -> str | None:
        if re.search(r"\bgaming\s+monitors?\b", goal or "", re.I):
            return "gaming"
        if re.search(r"\bover[\s-]?ear\b", goal or "", re.I):
            return "over ear"
        return None
    @staticmethod
    def take_specs(goal: str) -> dict:
        out = {}
        for field in ("screen_size", "refresh_rate", "resolution", "ram"):
            value = SpecReader.asked(field, goal)
            if value:
                out[field] = value
        return out
    def take_gate_spec(self) -> tuple:
        for field, value in self.specs.items():
            return field, value
        return None, None
    def adopt_spec(self, text: str) -> str | None:
        for field in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = SpecReader.asked(field, text)
            if value:
                self.spec_field, self.spec_value = field, value
                self.specs[field] = value
                self.sealed = False
                return field + " = " + value
        return None
class Offer:
    """One product_id::sku row the gate matrix can score."""
    def __init__(self, pid: str, sku: str) -> None:
        self.pid = str(pid)
        self.sku = str(sku or "")
        self.key = self.pid + "::" + self.sku
        self.price: float | None = None
        self.stock: bool | None = None
        self.options = ""
        self.title = ""
        self.brand = ""
        self.path: list = []
        self.facts: dict = {}
        self.listing: str | None = None
        self.seen_page = False
        self.quoted = False
    @property
    def pair(self) -> tuple:
        return (self.pid, self.sku)
class Gate:
    """One named column of the candidate × gate table."""
    def __init__(self, column: str) -> None:
        self.column = column
    def score(self, offer: Offer, shop: "MatrixShop") -> str:
        fn = getattr(self, "chk_" + self.column)
        return fn(offer, shop)
    def probe(self, offer: Offer, shop: "MatrixShop") -> tuple | None:
        col = self.column
        if col == "viewed":
            return shop.peek(offer.pid)
        if col == "inspected":
            return ("inspect_stock", {"product_id": offer.pid, "sku": offer.sku})
        if col == "live" and not offer.quoted:
            if not offer.seen_page:
                return shop.peek(offer.pid)
            return ("inspect_stock", {"product_id": offer.pid, "sku": offer.sku})
        if col == "spec" and shop.need.sealed:
            return ("message", {"content": shop.constraint_prompt()})
        if col in ("named_key", "fact_claim", "theme"):
            return shop.peek(offer.pid)
        if col == "listing_diversity":
            return shop.widen_listings()
        return None
    def chk_live(self, offer: Offer, shop: "MatrixShop") -> str:
        dead = offer.key in shop.dead_keys
        if str(offer.pid) in shop.dead_pids and shop.need.family != "recovery":
            dead = True
        if not offer.quoted and not offer.seen_page:
            return "unknown"
        if dead or shop.listing_clash(offer):
            return "fail"
        if not offer.stock or offer.price is None:
            return "fail"
        if shop.ceiling is not None and offer.price > shop.ceiling + MONEY_EPS:
            return "fail"
        return "pass"
    def chk_viewed(self, offer: Offer, shop: "MatrixShop") -> str:
        if offer.pid in shop.unreadable:
            return "fail"
        if offer.seen_page:
            return "pass"
        return "unknown"
    def chk_category(self, offer: Offer, shop: "MatrixShop") -> str:
        return "pass" if shop.shelf_fits(offer) else "fail"
    def chk_brand(self, offer: Offer, shop: "MatrixShop") -> str:
        return "pass" if shop.maker_fits(offer) else "fail"
    def chk_device(self, offer: Offer, shop: "MatrixShop") -> str:
        return "pass" if shop.device_fits(offer) else "fail"
    def chk_spec(self, offer: Offer, shop: "MatrixShop") -> str:
        if shop.need.sealed:
            return "unknown"
        field, want = shop.need.spec_field, shop.need.spec_value
        if not field or not want:
            return "pass"
        got = shop.variant_value(offer, field)
        if got == want:
            return "pass"
        if got:
            return "fail"
        if not offer.seen_page:
            return "unknown"
        return "fail"
    def chk_theme(self, offer: Offer, shop: "MatrixShop") -> str:
        theme = shop.need.theme
        if not theme:
            return "pass"
        title = shop.title_of(offer)
        if not title and not offer.seen_page:
            return "unknown"
        have = words_of(title)
        for token in theme:
            if token not in have:
                return "fail"
        return "pass"
    def chk_listing_diversity(self, offer: Offer, shop: "MatrixShop") -> str:
        if shop.need.family != "preference":
            return "pass"
        if shop.add_count:
            return "pass"
        if len(shop.listing_ids) >= 2:
            return "pass"
        return "unknown"
    def chk_fact_claim(self, offer: Offer, shop: "MatrixShop") -> str:
        if shop.need.family != "justification":
            return "pass"
        if shop.resale(offer):
            return "fail"
        brand = shop.brand_of(offer)
        if not brand:
            return "unknown" if not offer.seen_page else "fail"
        field, want = shop.need.spec_field, shop.need.spec_value
        if not field or not want:
            return "pass"
        if not offer.seen_page:
            return "unknown"
        if shop.fact_value(offer, field) == want and shop.product_value(offer, field) == want:
            return "pass"
        return "fail"
    def chk_named_key(self, offer: Offer, shop: "MatrixShop") -> str:
        if shop.need.family != "ranking":
            return "pass"
        if offer.pair not in shop.need.named:
            return "fail"
        if not offer.seen_page:
            return "unknown"
        hz = shop.measure(offer, "refresh_rate")
        inch = shop.measure(offer, "screen_size")
        if hz is None or inch is None:
            return "fail"
        return "pass"
    def chk_inspected(self, offer: Offer, shop: "MatrixShop") -> str:
        if offer.key not in shop.inspected:
            return "unknown"
        if shop.last_add_tick is not None and shop.quote_tick.get(offer.key, 0) <= shop.last_add_tick:
            return "unknown"
        return "pass"
class GateTable:
    """The SAT table: offers as rows, named gates as columns, cells as states."""
    def __init__(self) -> None:
        self.gates = [Gate(name) for name in GATE_ORDER]
        self.cells: dict = {}
    def paint(self, offers: list, shop: "MatrixShop") -> None:
        self.cells = {}
        for offer in offers:
            for gate in self.gates:
                self.cells[(offer.key, gate.column)] = gate.score(offer, shop)
    def state(self, offer: Offer, column: str) -> str:
        return self.cells.get((offer.key, column), "unknown")
    def row_all_pass(self, offer: Offer) -> bool:
        for gate in self.gates:
            if self.state(offer, gate.column) != "pass":
                return False
        return True
    def row_has_fail(self, offer: Offer) -> bool:
        for gate in self.gates:
            if self.state(offer, gate.column) == "fail":
                return True
        return False
    def viable(self, offer: Offer) -> bool:
        for gate in self.gates:
            if gate.column == "inspected":
                continue
            if self.state(offer, gate.column) != "pass":
                return False
        return True
    def any_viable(self, offers: list) -> bool:
        for offer in offers:
            if self.viable(offer):
                return True
        return False
    def unlock_first(self, offers: list, shop: "MatrixShop") -> tuple | None:
        """Walk the matrix; the first unknown cell that can act becomes a tool call."""
        if shop.need.family == "ranking" and "compare" in shop.toolset:
            missing = []
            for pid, _sku in shop.need.named:
                if pid in shop.pages or pid in shop.unreadable:
                    continue
                if pid not in missing:
                    missing.append(pid)
            if missing:
                return ("compare", {"product_ids": missing[:5]})
        if shop.need.sealed and "message" in shop.toolset and not shop.asked:
            return ("message", {"content": shop.constraint_prompt()})
        for offer in offers:
            failed = self.row_has_fail(offer)
            if failed and shop.need.family != "ranking":
                viewed = self.state(offer, "viewed")
                if viewed != "unknown":
                    continue
            for gate in self.gates:
                if self.state(offer, gate.column) != "unknown":
                    continue
                if gate.column == "inspected":
                    if not self.viable(offer):
                        continue
                    if shop.need.family == "preference" and shop.turn < shop.cap - 8 and not shop.cart:
                        continue
                action = gate.probe(offer, shop)
                if action is not None:
                    return action
        return None
    def passed_rows(self, offers: list) -> list:
        rows = []
        for offer in offers:
            if self.row_all_pass(offer):
                rows.append(offer)
        return rows
    def elect(self, offers: list, shop: "MatrixShop") -> Offer | None:
        """Walk all-pass rows and keep a running champion. No sort(key=tuple)."""
        fam = shop.need.family
        first = shop.add_count == 0
        passed = self.passed_rows(offers)
        if fam == "preference":
            for offer in passed:
                if offer.pair in shop.cart:
                    return offer
            if passed and shop.turn >= shop.cap - 8:
                return self._wish_then_cheap(passed, shop)
            return None
        if fam == "intent" and first and shop.need.want_brand:
            branded = []
            for offer in passed:
                if shop.pref_brand_hits(offer):
                    branded.append(offer)
            if branded:
                passed = branded
            elif shop.turn < shop.cap * 0.6:
                shop.nudge_brand()
                return None
        if fam == "recovery" and first:
            exact = []
            for offer in passed:
                if shop.relax_of(offer) == 0:
                    exact.append(offer)
            if exact:
                passed = exact
            elif shop.turn < shop.cap * 0.6:
                shop.nudge_exact()
                return None
        if not passed:
            return None
        return self._champion(passed, shop, first)
    def _wish_then_cheap(self, rows: list, shop: "MatrixShop") -> Offer | None:
        champ = None
        best_hits = -1
        best_price = 1e18
        for offer in rows:
            hits = shop.wish_hits(offer)
            price = offer.price if offer.price is not None else 1e12
            take = False
            if champ is None:
                take = True
            elif hits > best_hits:
                take = True
            elif hits == best_hits and price < best_price:
                take = True
            if take:
                champ = offer
                best_hits = hits
                best_price = price
        return champ
    def _champion(self, rows: list, shop: "MatrixShop", first: bool) -> Offer | None:
        champ = None
        for offer in rows:
            if champ is None or self._beats(offer, champ, shop, first):
                champ = offer
        return champ
    def _beats(self, a: Offer, b: Offer, shop: "MatrixShop", first: bool) -> bool:
        fam = shop.need.family
        if fam == "ranking":
            return self._rank_beats(a, b, shop)
        if fam == "recovery":
            ra, rb = shop.relax_of(a), shop.relax_of(b)
            if ra != rb:
                return ra < rb
            return self._cheaper(a, b)
        extra_a = shop.extra_misses(a)
        extra_b = shop.extra_misses(b)
        if fam == "intent" and first and shop.need.want_brand:
            near_a = 0 if shop.pref_brand_hits(a) else 1
            near_b = 0 if shop.pref_brand_hits(b) else 1
            if near_a != near_b:
                return near_a < near_b
        if extra_a != extra_b:
            return extra_a < extra_b
        proven_a = 0 if shop.handset_of(a) == "yes" else 1
        proven_b = 0 if shop.handset_of(b) == "yes" else 1
        if proven_a != proven_b:
            return proven_a < proven_b
        return self._cheaper(a, b)
    def _rank_beats(self, a: Offer, b: Offer, shop: "MatrixShop") -> bool:
        ha = shop.measure(a, "refresh_rate") or 0.0
        hb = shop.measure(b, "refresh_rate") or 0.0
        if ha != hb:
            return ha > hb
        ia = shop.measure(a, "screen_size") or 0.0
        ib = shop.measure(b, "screen_size") or 0.0
        if ia != ib:
            return ia > ib
        if self._cheaper(a, b):
            return True
        if self._cheaper(b, a):
            return False
        return a.key < b.key
    @staticmethod
    def _cheaper(a: Offer, b: Offer) -> bool:
        pa = a.price if a.price is not None else 1e12
        pb = b.price if b.price is not None else 1e12
        if pa != pb:
            return pa < pb
        return a.key < b.key
class MatrixShop:
    """Runs the episode: unknown cells become tool calls; all-pass rows close."""
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.toolset = set()
        for item in self.tools:
            if not isinstance(item, dict):
                continue
            fn = item.get("function") or {}
            name = fn.get("name")
            if name:
                self.toolset.add(name)
        self.cap = int(self.policy.get("max_steps") or 30)
        self.call_cap = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.need = Need(str(self.policy.get("query") or ""))
        self.ceiling = self.need.ceiling
        self.table = GateTable()
        self.offers: dict = {}
        self.pages: dict = {}
        self.listing_rows: dict = {}
        self.listing_ids: set = set()
        self.listing_of: dict = {}
        self.identities: set = set()
        self.cart: list = []
        self.inspected: set = set()
        self.quote_tick: dict = {}
        self.dead_keys: set = set()
        self.dead_pids: set = set()
        self.seen_calls: set = set()
        self.tick = 0
        self.first_add_tick: int | None = None
        self.last_add_tick: int | None = None
        self.add_count = 0
        self.event_target: str | None = None
        self.started = time.monotonic()
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": self.system_text()},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.turn = 0
        self.asked = False
        self.q_pending = False
        self.shopper_wrote = False
        self.reject_run = 0
        self.held: dict | None = None
        self.searches = 0
        self.cart_looks = 0
        self.after_add: tuple | None = None
        self.forced_reads = 0
        self.told_exact = False
        self.questions = 0
        self.looked_siblings = False
        self.told_brand = False
        self.wishes: set | None = None
        self.probes: set = set()
        self.looked_twins = False
        self.reads_done = False
        self.unreadable: set = set()
        self.q_waits = 0
        self.done = False
        self.model_i = 0
        self.roster = allowed_roster()
        for pid, sku in self.need.named:
            self.ensure(pid, sku)
    def left(self) -> float:
        return HORIZON_S - (time.monotonic() - self.started)
    def system_text(self) -> str:
        play = LANE_HINTS.get(self.need.family, LANE_HINTS["generic"])
        extra = ""
        if self.need.family == "recall" and self.need.need_ids:
            extra = (
                " You must surface at least %d distinct matching products in search or "
                "filter results before ordering; the matrix reports the count after each search."
                % self.need.need_ids
            )
        return SHOP_RULES + "\n\nLane: " + play + extra
    def constraint_prompt(self) -> str:
        return (
            "Before I search: which hardware specification is your firm requirement "
            "(screen size, resolution class, refresh rate, or memory), and what value do you need?"
        )
    def preference_prompt(self) -> str:
        return (
            "Quick question so I choose well: which of your preferences counts most, and in what "
            "order should I weigh the others? Let me know your priority."
        )
    def opening_prompt(self) -> str | None:
        if self.need.family == "constraint" or self.need.sealed:
            return self.constraint_prompt()
        if self.need.family == "preference":
            return self.preference_prompt()
        return None
    def whisper(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})
    def ensure(self, pid: Any, sku: Any) -> Offer:
        key = str(pid) + "::" + str(sku or "")
        offer = self.offers.get(key)
        if offer is None:
            offer = Offer(str(pid), str(sku or ""))
            self.offers[key] = offer
        return offer
    def walk_offers(self) -> list:
        return [offer for offer in self.offers.values() if offer.sku]
    def page(self, pid: str) -> dict:
        return self.pages.get(str(pid)) or {}
    def title_of(self, offer: Offer) -> str:
        titled = offer.title or str(self.page(offer.pid).get("title") or "")
        if titled:
            return titled
        for row in self.listing_rows.values():
            if str(row.get("product_id")) == offer.pid:
                return str(row.get("title") or "")
        return ""
    def brand_of(self, offer: Offer) -> str:
        branded = offer.brand or str(self.page(offer.pid).get("brand") or "")
        if branded:
            return branded
        for row in self.listing_rows.values():
            if str(row.get("product_id")) == offer.pid and row.get("brand"):
                return str(row["brand"])
        return ""
    def path_of(self, offer: Offer) -> list:
        if offer.path:
            return list(offer.path)
        path = self.page(offer.pid).get("category_path")
        return list(path) if isinstance(path, list) else []
    def facts_of(self, offer: Offer) -> dict:
        if offer.facts:
            return offer.facts
        raw = self.page(offer.pid).get("facts")
        return raw if isinstance(raw, dict) else {}
    def drop_codes(self, offer: Offer, title: str) -> str:
        model = str(self.facts_of(offer).get("model") or "")
        for token in re.split(r"[;,]", model):
            token = token.strip()
            stem = re.split(r"[-_]", token)[0] if token else ""
            coded = bool(token) and re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)+", token, re.I)
            mixed = bool(stem) and re.search(r"[a-z]", stem, re.I) and re.search(r"\d", stem)
            panel = bool(stem) and re.fullmatch(r"(?:[48]k|[ufq]?hd|fullhd|ultrahd|\d+p)\d*", stem, re.I)
            if coded and mixed and not panel:
                title = re.sub(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", "", title, flags=re.I)
        return title
    def fact_value(self, offer: Offer, field: str) -> str | None:
        raw = self.facts_of(offer).get(SpecReader.LABEL[field])
        if not raw:
            return None
        text = str(raw).strip()
        found = SpecReader.harvest(field, text)
        if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text):
            return "%g" % float(text) + SpecReader.UNIT[field]
        return SpecReader.lone(found)
    def product_value(self, offer: Offer, field: str) -> str | None:
        vals: set = set()
        clash = False
        raw = self.facts_of(offer).get(SpecReader.LABEL[field])
        if raw:
            found = SpecReader.harvest(field, str(raw))
            clash = clash or len(found) > 1
            if field == "resolution" and SpecReader.QUARTER.search(str(raw)):
                clash = True
            vals |= found
        title = self.title_of(offer)
        if field == "resolution":
            title = self.drop_codes(offer, title)
            if SpecReader.QUARTER.search(title):
                clash = True
        vals |= SpecReader.near(field, title)
        if clash:
            return None
        return SpecReader.lone(vals)
    def variant_value(self, offer: Offer, field: str) -> str | None:
        opt = offer.options
        vals = SpecReader.near(field, opt)
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if SpecReader.talks(field, opt):
            return None
        return self.product_value(offer, field)
    def measure(self, offer: Offer, field: str) -> float | None:
        value = self.variant_value(offer, field)
        if not value:
            return None
        try:
            return float(value[:-2])
        except ValueError:
            return None
    def shelf_fits(self, offer: Offer) -> bool:
        want = self.need.category
        path = self.path_of(offer)
        if not want or not path:
            return True
        wanted = SpecReader.stems(want)
        labels = set()
        for label in path:
            labels |= SpecReader.stems(str(label))
        placed = bool(wanted & labels)
        if not placed:
            for label in path:
                glued = SpecReader.stems("".join(re.findall(r"[a-z]+", str(label).lower())))
                if glued == wanted:
                    placed = True
                    break
        if not placed and "smartphone" in wanted:
            for label in path:
                if re.fullmatch(r"(?:smart|cell|mobile|unlocked)?\s*phones?", str(label).strip(), re.I):
                    placed = True
                    break
        if not placed:
            return False
        shape = self.need.shape
        if not shape:
            return True
        title = self.title_of(offer)
        bag = SpecReader.stems(title) | words_of(title)
        if shape == "gaming":
            fast = False
            for item in SpecReader.hertz(title):
                if float(item[:-2]) >= 120:
                    fast = True
                    break
            return "monitor" in bag and ("gaming" in bag or fast)
        return ("over" in bag and "ear" in bag) or "overear" in bag or "circumaural" in bag
    def maker_fits(self, offer: Offer) -> bool:
        want = self.need.brand
        if not want:
            return True
        wanted = words_of(want)
        return bool(wanted <= words_of(self.brand_of(offer)) or wanted <= words_of(self.title_of(offer)))
    def device_fits(self, offer: Offer) -> bool:
        if not noun_match(self.need.category or "", "smartphone"):
            return True
        verdict = self.handset_of(offer)
        if self.need.family in ("constraint", "justification", "recovery"):
            return verdict == "yes"
        return verdict != "no"
    def handset_of(self, offer: Offer) -> str:
        return SpecReader.handset(self.facts_of(offer), self.title_of(offer), offer.options)
    def resale(self, offer: Offer) -> bool:
        return bool(re.search(r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed", self.title_of(offer), re.I))
    def relax_of(self, offer: Offer) -> int:
        model, color = self.need.want_model, self.need.want_color
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self.title_of(offer) + " " + " ".join(str(v) for v in self.facts_of(offer).values()))
        same_model = squash(model) in blob
        shade = (offer.options + " " + str(self.facts_of(offer).get("color") or "") + " " + self.title_of(offer)).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3
    def extra_misses(self, offer: Offer) -> int:
        misses = 0
        for field, value in self.need.specs.items():
            if field == self.need.spec_field:
                continue
            if self.variant_value(offer, field) != value:
                misses += 1
        return misses
    def pref_brand_hits(self, offer: Offer) -> bool:
        wanted = words_of(self.need.want_brand or "")
        if not wanted:
            return False
        return bool(wanted <= words_of(self.brand_of(offer)) or wanted <= words_of(self.title_of(offer)))
    def listing_clash(self, offer: Offer) -> bool:
        if self.need.family != "preference" or not self.event_target:
            return False
        dead_pid = self.event_target.split("::", 1)[0]
        dead_listing = self.listing_of.get(dead_pid)
        if dead_listing and self.listing_of.get(str(offer.pid)) == dead_listing:
            return True
        return False
    def wish_words(self) -> set:
        if self.wishes is not None:
            return self.wishes
        said = self.need.goal
        for msg in self.messages:
            if msg.get("role") == "user":
                said += " " + str(msg.get("content") or "")
        drop = words_of(self.need.category or "") | {
            "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
            "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
            "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
        }
        kept = set()
        for token in words_of(said):
            if len(token) > 3 and not token.isdigit() and token not in drop:
                kept.add(token)
        self.wishes = kept
        return kept
    def wish_hits(self, offer: Offer) -> int:
        wants = self.wish_words()
        if not wants:
            return 0
        blob = " ".join(str(v) for v in self.facts_of(offer).values())
        have = words_of(blob + " " + self.title_of(offer) + " " + offer.options)
        return len(wants & have)
    def refresh_table(self) -> list:
        rows = self.walk_offers()
        self.table.paint(rows, self)
        return rows
    def peek(self, pid: str) -> tuple:
        unseen = []
        for offer in self.walk_offers():
            if offer.seen_page or offer.pid in self.unreadable:
                continue
            if offer.pid not in unseen:
                unseen.append(offer.pid)
        if pid not in unseen:
            unseen.insert(0, pid)
        if len(unseen) > 1 and "compare" in self.toolset:
            return ("compare", {"product_ids": unseen[:5]})
        return ("view", {"product_id": pid})
    def widen_listings(self) -> tuple | None:
        leads = self.unread_leads()
        if leads and "compare" in self.toolset:
            return ("compare", {"product_ids": leads[:5]})
        if leads:
            return ("view", {"product_id": leads[0]})
        return self.seed_catalog()
    def recall_need(self) -> int:
        early = self.searches < 8 and self.turn < self.cap * 0.5
        return self.need.need_ids + (2 if early else 0)
    def may_commit(self) -> bool:
        if self.add_count:
            return True
        if self.need.sealed:
            if self.turn < self.cap * 0.6:
                return False
            self.need.sealed = False
            self.whisper("The firm requirement was never stated, so the remaining gates decide the row.")
        fam = self.need.family
        if fam in ("intent", "constraint") and self.turn < 4 and not self.shopper_wrote:
            return False
        if fam == "preference" and len(self.listing_ids) < 2:
            return False
        roomy = self.turn < self.cap * 0.45 and self.left() > 140
        if fam == "recall" and roomy:
            if len(self.identities) < self.recall_need() or self.searches < 2:
                return False
        rows = self.walk_offers()
        unique = set()
        for offer in rows:
            if self.table.viable(offer):
                unique.add(offer.pid)
        if roomy and not self.reads_done and len(unique) < 2:
            return False
        return True
    def unread_leads(self) -> list:
        cheapest = None
        chosen: list = []
        for key, row in self.listing_rows.items():
            pid, _sep, sku = key.partition("::")
            if not sku or str(pid) in self.pages or str(pid) in self.dead_pids:
                continue
            offer = self.offers.get(key)
            if offer is None:
                continue
            if not offer.stock or offer.price is None:
                continue
            if self.ceiling is not None and offer.price > self.ceiling + MONEY_EPS:
                continue
            title = str(row.get("title") or "")
            if self.need.theme and not all(token in words_of(title) for token in self.need.theme):
                continue
            if self.need.want_model and self.add_count == 0 and not self.event_target:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.need.want_model.lower()) not in squashed:
                    continue
            if self.need.brand:
                wanted = words_of(self.need.brand)
                if not (wanted <= words_of(str(row.get("brand") or "")) or wanted <= words_of(title)):
                    continue
            price = offer.price
            if cheapest is None or price < cheapest:
                cheapest = price
                chosen.insert(0, str(pid))
            elif str(pid) not in chosen:
                chosen.append(str(pid))
        out = []
        for pid in chosen:
            if pid not in out:
                out.append(pid)
        return out
    def twin_search(self) -> tuple | None:
        dead = self.event_target
        if self.looked_twins or not dead or "search" not in self.toolset:
            return None
        pid = dead.split("::", 1)[0]
        title = ""
        page = self.pages.get(pid) or {}
        title = str(page.get("title") or "")
        if not title:
            for offer in self.walk_offers():
                if offer.pid == pid and offer.title:
                    title = offer.title
                    break
        if not title:
            return None
        self.looked_twins = True
        self.whisper(
            "The dead row shares a title with sibling listings; searching that title "
            "surfaces other colors and sizes that still pass the same hard gates."
        )
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})
    def sibling_search(self) -> tuple | None:
        if self.looked_siblings:
            return None
        rows = self.walk_offers()
        self.table.paint(rows, self)
        exact_live = False
        near = None
        for offer in rows:
            if not offer.seen_page:
                continue
            if self.relax_of(offer) == 0 and offer.stock and offer.price is not None:
                exact_live = True
            if self.relax_of(offer) == 1 and near is None:
                near = offer
        if exact_live or near is None:
            return None
        title = self.title_of(near)
        if not title:
            return None
        self.looked_siblings = True
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})
    def seed_catalog(self) -> tuple | None:
        fam = self.need.family
        if fam == "recall" and len(self.identities) < self.recall_need():
            return None
        twins = self.twin_search()
        if twins is not None:
            return twins
        leads = self.unread_leads()
        if leads and self.forced_reads < 8 and self.turn >= 2:
            self.forced_reads += 1
            if len(leads) > 1 and "compare" in self.toolset:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        if self.turn < 2:
            return None
        probe = (self.need.brand or "", "%.2f" % (self.ceiling or 0.0), str(self.need.spec_value))
        if probe in self.probes:
            self.reads_done = True
            return None
        self.probes.add(probe)
        if "filter" in self.toolset and self.need.category:
            args = {"category": self.need.category, "in_stock": True, "k": 10}
            if self.ceiling is not None:
                args["max_price"] = self.ceiling
            if self.need.brand:
                args["brand"] = self.need.brand
            return ("filter", args)
        if "search" in self.toolset:
            bits = []
            for piece in (self.need.brand, self.need.spec_value, self.need.category):
                if piece:
                    bits.append(str(piece))
            if bits:
                return ("search", {"query": " ".join(bits), "in_stock": True, "k": 10})
        self.reads_done = True
        return None
    def nudge_brand(self) -> None:
        if self.told_brand:
            return
        self.told_brand = True
        self.whisper(
            "No %s row that passes every hard gate has been read yet. Filter by brand=%s "
            "within the budget and view those products before any other."
            % (self.need.want_brand, self.need.want_brand)
        )
    def nudge_exact(self) -> None:
        if self.told_exact or not self.need.want_model:
            return
        self.told_exact = True
        color = ", " + self.need.want_color if self.need.want_color else ""
        self.whisper(
            "The shopper's exact preference (%s%s) is not a row yet. Search the model code "
            "on its own and view the results so every color variant is on the table; only "
            "that exact pairing may be the first cart add." % (self.need.want_model, color)
        )
    def shortlist(self) -> None:
        rows = self.refresh_table()
        if self.need.family == "preference" and not self.cart:
            good = [offer for offer in rows if self.table.viable(offer)]
            if good:
                bits = ["%s::%s at %.2f" % (o.pid, o.sku, o.price or 0.0) for o in good[:6]]
                self.whisper(
                    "Rows that currently pass every hard gate: " + "; ".join(bits)
                    + ". Pick the one that best fits the shopper's stated priorities, inspect it, then add it."
                )
                return
        pick = self.table.elect(rows, self)
        if pick:
            self.whisper(
                "All-pass row: %s::%s at %.2f. The closer will inspect, cart and order it; "
                "read other rows only if one could beat it." % (pick.pid, pick.sku, pick.price or 0.0)
            )
            return
        field, want = self.need.spec_field, self.need.spec_value
        blocked = [o for o in rows if o.seen_page and o.stock and o.price is not None and not self.table.viable(o)]
        if blocked and field:
            sample = [o.pid + "::" + o.sku for o in blocked[:3]]
            self.whisper(
                "No row is all-pass yet. For each, the %s cell must resolve to exactly %s "
                "from the variant's own option text or the product's raw facts: %s. "
                "Look at different products." % (field, want, ", ".join(sample))
            )
        elif blocked:
            self.whisper(
                "Rows read so far fail a hard gate (category, brand, stock, budget or the "
                "kind of device requested). Search for different products."
            )
    def lower_budget(self, text: str) -> None:
        new = Need.take_budget(text)
        if new is None or (self.ceiling is not None and new >= self.ceiling):
            return
        self.ceiling = new
        self.need.ceiling = new
        self.shopper_wrote = True
        self.whisper(
            "Budget updated by the shopper: at most %.2f %s. Every later row must fit it."
            % (new, self.need.currency)
        )
        for pid, sku in list(self.cart):
            offer = self.offers.get(pid + "::" + sku)
            if offer is None or offer.price is None:
                continue
            if offer.price > new + MONEY_EPS:
                self.dead_keys.add(offer.key)
                self.whisper(
                    "The carted row %s costs %.2f, above the new budget; the closer will replace it."
                    % (offer.key, offer.price)
                )
    def claims_for(self, offer: Offer) -> list:
        out = []
        field, want = self.need.spec_field, self.need.spec_value
        if not field and self.need.family == "justification":
            field, want = "resolution", self.fact_value(offer, "resolution")
        if not field or not want:
            return []
        stated = dict(self.need.specs) or {field: want}
        stated.setdefault(field, want)
        for name, value in stated.items():
            if self.fact_value(offer, name) == value and self.product_value(offer, name) == value:
                out.append({"field": name, "value": value})
        if not out:
            return []
        if self.ceiling is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.ceiling})
        elif offer.price is not None:
            out.append({"field": "price", "value": "%.2f" % offer.price})
        out.append({"field": "in_stock", "value": "true"})
        brand = self.brand_of(offer)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out
    def order_args(self, offer: Offer) -> dict:
        out = {"product_id": offer.pid, "sku": offer.sku}
        if self.need.family == "justification":
            claims = self.claims_for(offer)
            if claims:
                out["justification"] = {"claims": claims}
        return out
    def order_ready(self, offer: Offer) -> tuple:
        if offer.price is None and offer.stock is None:
            return False, "inspect_stock the exact product_id and sku first"
        if offer.pair not in self.cart:
            return False, "add that exact variant to the cart first"
        if self.last_add_tick is not None and self.quote_tick.get(offer.key, 0) <= self.last_add_tick:
            return False, "re-check the variant with inspect_stock after the cart addition"
        if not offer.stock:
            return False, "that variant is out of stock; choose a different product"
        if self.ceiling is not None and offer.price is not None and offer.price > self.ceiling + MONEY_EPS:
            return False, "that variant costs %.2f, above the budget of %.2f" % (offer.price, self.ceiling)
        if offer.key in self.dead_keys:
            return False, "that variant was invalidated by the market change; choose a different product"
        return True, ""
    def spoken_answer(self) -> str:
        rows = self.refresh_table()
        pick = self.table.elect(rows, self)
        if pick is None:
            for offer in rows:
                pick = offer
                break
        if pick is None:
            return "I am still checking the options against your requirements; nothing is decided yet."
        parts = ["I am going with " + (self.title_of(pick) or pick.pid)[:90]]
        field, want = self.need.spec_field, self.need.spec_value
        if field and want and self.variant_value(pick, field) == want:
            parts.append("its listed " + field.replace("_", " ") + " is " + want)
        if pick.price is not None:
            parts.append("it is %.2f %s and in stock" % (pick.price, self.need.currency))
        return ", ".join(parts) + "."
    def next_move(self) -> tuple | None:
        if self.done:
            return None
        if self.q_pending:
            if self.q_waits < 2 or "message" not in self.toolset:
                self.q_waits += 1
                return None
            return ("message", {"content": self.spoken_answer()})
        if self.need.family == "recovery" and not self.add_count and self.need.want_model and "search" in self.toolset:
            forced = self.sibling_search()
            if forced is not None:
                return forced
        rows = self.refresh_table()
        unknown = self.table.unlock_first(rows, self)
        if unknown is not None:
            name, _args = unknown
            if name in ("view", "compare", "inspect_stock", "message"):
                pick = self.table.elect(rows, self)
                if pick is None or not self.may_commit() or name != "inspect_stock":
                    return unknown
        pick = self.table.elect(rows, self)
        if pick is None or not self.may_commit():
            seeded = self.seed_catalog()
            if seeded is not None:
                return seeded
            if pick is None:
                return None
            if not self.may_commit():
                return None
        if pick.pair in self.cart:
            if self.table.state(pick, "inspected") != "pass":
                return ("inspect_stock", {"product_id": pick.pid, "sku": pick.sku})
            return ("place_test_order", self.order_args(pick))
        if not self.may_commit():
            return None
        for pid, sku in self.cart:
            key = pid + "::" + sku
            if key in self.dead_keys and "remove_from_cart" in self.toolset:
                return ("remove_from_cart", {"product_id": pid, "sku": sku})
        if self.table.state(pick, "inspected") != "pass":
            return ("inspect_stock", {"product_id": pick.pid, "sku": pick.sku})
        return ("add_to_cart", {"product_id": pick.pid, "sku": pick.sku})
    def ask_model(self) -> dict | None:
        attempt = 0
        while attempt < 3:
            model = self.roster[min(self.model_i, len(self.roster) - 1)]
            payload = {
                "model": model,
                "messages": self.messages,
                "tools": self.tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            resp = PIPE.post("/inference/chat/completions", json_data=payload)
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.call_cap]}
                self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                self.whisper("No tool call was made. Choose exactly one of the supplied tools now.")
                attempt += 1
                continue
            if self.left() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_i = min(self.model_i + 1, len(self.roster) - 1)
            attempt += 1
        return None
    def post_env(self, envelope: dict) -> dict | None:
        attempt = 0
        while attempt < 4:
            result = PIPE.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.left() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
            attempt += 1
        return None
    def enact(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if self.held is not None:
            pend = self.held
            replay = self.post_env(pend["envelope"])
            if replay is None:
                self.whisper("The environment is not answering; the previous action is still pending and will be retried.")
                return None
            self.held = None
            self.turn = pend["turn"]
            self.remember(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
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
        result = self.post_env(envelope)
        if result is None:
            self.held = {
                "envelope": envelope,
                "turn": turn,
                "name": name,
                "args": args,
                "content": content,
                "call_id": tool_call_id or ("agent-" + str(turn)),
            }
        else:
            self.turn = turn
        return self.remember(name, args, content, tool_call_id or ("agent-" + str(turn)), result)
    def remember(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.seen_calls.discard(name + json.dumps(args, sort_keys=True))
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
        self.ingest(name, args, obs)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": clip_obs(json.dumps(obs), OBS_CAP)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self.lower_budget(text)
            if self.need.family in ("constraint", "intent") and not self.need.spec_value:
                stated = self.need.adopt_spec(text)
                if stated:
                    self.shopper_wrote = True
                    self.whisper(
                        "Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought."
                    )
            if Need.ASK.search(text):
                self.q_pending = True
                self.whisper("The shopper asked a question: answer it with message in your next turn, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        return obs
    def ingest(self, name: str, args: dict, obs: dict) -> None:
        self.tick += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        sink = {
            "search": self._hits,
            "filter": self._hits,
            "view": self._page,
            "compare": self._cmp,
            "inspect_stock": self._insp,
            "add_to_cart": self._added,
            "remove_from_cart": self._pulled,
            "inspect_cart": self._cart_sync,
        }.get(name)
        if sink:
            sink(obs)
    def _hits(self, obs: dict) -> None:
        for row in obs.get("results") or []:
            if isinstance(row, dict):
                self._row(row)
    def _page(self, obs: dict) -> None:
        if isinstance(obs, dict):
            self._prod(obs)
    def _cmp(self, obs: dict) -> None:
        for prod in obs.get("compared") or []:
            if isinstance(prod, dict):
                self._prod(prod)
    def _insp(self, obs: dict) -> None:
        offer = self.ensure(obs.get("product_id"), obs.get("sku"))
        offer.price = money(obs.get("price"))
        offer.stock = bool(obs.get("in_stock"))
        offer.quoted = True
        self.quote_tick[offer.key] = self.tick
        self.inspected.add(offer.key)
    def _added(self, obs: dict) -> None:
        added = obs.get("added") or {}
        pid, sku = added.get("product_id"), added.get("sku")
        if pid and sku:
            self.add_count += 1
            pair = (str(pid), str(sku))
            if pair not in self.cart:
                self.cart.append(pair)
            if self.first_add_tick is None:
                self.first_add_tick = self.tick
            self.last_add_tick = self.tick
        self._cart_sync(obs)
    def _pulled(self, obs: dict) -> None:
        self._cart_sync(obs)
        removed = obs.get("removed") or {}
        pair = (str(removed.get("product_id") or ""), str(removed.get("sku") or ""))
        if pair in self.cart:
            self.cart.remove(pair)
    def _cart_sync(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = []
        for row in rows:
            if isinstance(row, dict) and row.get("product_id"):
                pairs.append((str(row.get("product_id")), str(row.get("sku") or "")))
        if pairs or rows == []:
            self.cart = pairs
        for row in rows:
            if not isinstance(row, dict) or not row.get("product_id"):
                continue
            offer = self.ensure(row.get("product_id"), row.get("sku"))
            price = money(row.get("current_price"))
            stock = row.get("current_in_stock")
            if price is None and stock is None:
                continue
            if stock is not None:
                offer.stock = bool(stock)
            if price is not None:
                offer.price = price
            offer.quoted = True
            self.quote_tick[offer.key] = self.tick
    def _row(self, row: dict) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        if not self.need.theme or all(token in words_of(title) for token in self.need.theme):
            self.identities.add(str(pid))
        if not row.get("sku"):
            return
        offer = self.ensure(pid, row.get("sku"))
        offer.title = title or offer.title
        if row.get("brand"):
            offer.brand = str(row["brand"])
        self.listing_rows[offer.key] = row
        if row.get("price") is not None and not offer.quoted:
            offer.price = money(row.get("price"))
            offer.stock = bool(row.get("in_stock", True))
            offer.quoted = True
    def _prod(self, prod: dict) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.pages[str(pid)] = prod
        if prod.get("listing_id"):
            self.listing_ids.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        title = str(prod.get("title") or "")
        if self.need.theme and all(token in words_of(title) for token in self.need.theme):
            self.identities.add(str(pid))
        facts = prod.get("facts") if isinstance(prod.get("facts"), dict) else {}
        path = prod.get("category_path") if isinstance(prod.get("category_path"), list) else []
        brand = str(prod.get("brand") or "")
        for var in prod.get("variants") or []:
            if not isinstance(var, dict) or not var.get("sku"):
                continue
            offer = self.ensure(pid, var.get("sku"))
            offer.seen_page = True
            offer.title = title
            offer.brand = brand
            offer.path = path
            offer.facts = facts
            offer.options = str(var.get("options") or "")
            offer.price = money(var.get("price"))
            offer.stock = bool(var.get("in_stock"))
            offer.quoted = True
            if prod.get("listing_id"):
                offer.listing = str(prod["listing_id"])
            self.quote_tick[offer.key] = self.tick
    def after_cart(self) -> None:
        pid, sku = self.after_add
        self.after_add = None
        obs = self.enact("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = money(obs.get("price"))
        over = self.ceiling is not None and price is not None and price > self.ceiling + MONEY_EPS
        key = pid + "::" + sku
        if stock and not over:
            self.whisper(
                "Re-check after the cart addition: %s is in stock at %s, unchanged. The closer will order it."
                % (key, price)
            )
            return
        self.event_target = key
        self.dead_keys.add(key)
        self.dead_pids.add(str(pid))
        self.forced_reads = 0
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        tip = {
            "recovery": "Recover with the smallest possible change: the same model in another color before any other model.",
            "ranking": "The remaining named rows are walked again; the closer takes the new champion.",
            "preference": "Recover to a product with a different listing_id that still fits the shopper's priorities.",
            "constraint": "Choose a different product that still meets every requirement, not another variant of the same one.",
        }.get(self.need.family, "Choose a different product that meets every firm requirement.")
        self.whisper("Market change: the carted row %s %s. " % (key, reason) + tip)
        self.shortlist()
    def sprint(self) -> None:
        for pid, sku in reversed(self.cart):
            offer = self.offers.get(pid + "::" + sku)
            if offer is None:
                continue
            ok, _why = self.order_ready(offer)
            if ok and not self.done:
                self.enact("place_test_order", self.order_args(offer))
                return
        if self.done or self.turn > self.cap - 3:
            return
        rows = self.refresh_table()
        good = []
        for offer in rows:
            if self.table.viable(offer):
                good.append(offer)
        if not good and self.need.family != "justification":
            for offer in rows:
                spec_ok = self.table.state(offer, "spec") == "pass"
                named_ok = (not self.need.named) or offer.pair in self.need.named
                if offer.stock and spec_ok and self.maker_fits(offer) and self.shelf_fits(offer) and named_ok:
                    good.append(offer)
        if not good:
            return
        pick = self.table._champion(good, self, self.add_count == 0)
        if pick is None:
            return
        if pick.pair not in self.cart:
            obs = self.enact("add_to_cart", {"product_id": pick.pid, "sku": pick.sku})
            if self.done or not isinstance(obs, dict) or obs.get("error"):
                return
        obs = self.enact("inspect_stock", {"product_id": pick.pid, "sku": pick.sku})
        if self.done or not isinstance(obs, dict) or not obs.get("in_stock"):
            return
        price = money(obs.get("price"))
        if self.ceiling is not None and price is not None and price > self.ceiling + MONEY_EPS:
            return
        self.enact("place_test_order", self.order_args(pick))
    def admit(self, name: str, args: dict) -> str | None:
        if name not in self.toolset:
            return "unknown tool; choose one of " + str(sorted(n for n in self.toolset if n))
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.cart:
                pid, sku = self.cart[-1]
            offer = self.offers.get(pid + "::" + sku) or Offer(pid, sku)
            ok, why = self.order_ready(offer)
            if not ok:
                return "order refused: " + why
            if self.need.family == "recall" and len(self.identities) < self.recall_need():
                return (
                    "order refused: only %d distinct matching products have been surfaced; search with different "
                    "wording until at least %d appear" % (len(self.identities), self.recall_need())
                )
            if self.need.family == "preference" and len(self.listing_ids) < 2:
                return "order refused: compare at least two products with different listing_id values first"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            offer = self.ensure(pid, sku)
            self.refresh_table()
            if self.table.state(offer, "live") != "pass":
                return "that variant is out of stock, over budget, or already invalidated"
            if self.table.state(offer, "viewed") != "pass":
                return "view or compare that product first so its raw facts are on record"
            if offer.key not in self.inspected:
                return "inspect_stock that exact variant first: its current price and stock decide whether it can be bought"
            if self.table.state(offer, "spec") != "pass" and self.need.spec_field:
                return (
                    "that variant does not resolve to %s = %s from its own option text or raw facts"
                    % (self.need.spec_field, self.need.spec_value)
                )
            if not self.table.viable(offer):
                return "that product fails a firm requirement (category, brand, or the kind of device requested)"
            if self.need.family == "preference" and len(self.listing_ids) < 2:
                return "compare two products with different listing_id values before the first cart addition"
            if self.need.family == "recovery" and not self.add_count and self.need.want_model and self.relax_of(offer):
                return (
                    "the first cart item must be the exact model and color the shopper named; the other colors are "
                    "separate listings sharing the same title, so search that title and read them first"
                )
            if not self.may_commit():
                return (
                    "hold: read one more product that also meets every requirement first, so there is a "
                    "fallback when the market moves right after the cart addition"
                )
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.q_pending and "?" not in text:
                return "do not send status updates; use message only to answer the shopper's question or to ask one"
            if not self.q_pending and self.questions >= 2:
                return "the shopper has already answered; decide from the evidence you have rather than asking again"
        if name == "inspect_cart" and self.cart_looks >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.seen_calls and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        self.seen_calls.add(key)
        return None
    def run(self) -> list:
        question = self.opening_prompt()
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.cap:
            if self.left() < 25 or self.turn >= self.cap - 3:
                self.sprint()
                break
            if question and not self.asked and "message" in self.toolset:
                self.asked = True
                self.enact("message", {"content": question})
                continue
            if self.after_add and "inspect_stock" in self.toolset:
                self.after_cart()
                continue
            move = self.next_move()
            if move is not None:
                name, args = move
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    if name in ("inspect_stock", "add_to_cart"):
                        self.dead_keys.add(str(args.get("product_id")) + "::" + str(args.get("sku")))
                    if name == "compare":
                        self.unreadable.update(str(p) for p in args.get("product_ids") or [])
                    if name == "message":
                        self.q_pending = False
                    last_forced, repeats = None, 0
                    self.whisper("That step changed nothing, so the matrix moved on to other rows.")
                    continue
                obs = self.enact(name, args)
                if name == "message":
                    self.q_pending = False
                    self.q_waits = 0
                if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                    added = obs.get("added") or {}
                    if added.get("product_id"):
                        self.after_add = (str(added["product_id"]), str(added["sku"]))
                continue
            last_forced, repeats = None, 0
            proposal = self.ask_model()
            if proposal is None:
                self.sprint()
                break
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = decode_args(call)
            content = proposal.get("content") or ""
            nudges = 0
            verdict = None
            while True:
                verdict = self.admit(name, args)
                if verdict is None:
                    break
                if nudges >= NUDGE_CAP:
                    self.reject_run += 1
                    self.whisper("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if self.reject_run >= 3:
                        self.shortlist()
                        self.reject_run = 0
                    verdict = "skip"
                    break
                nudges += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": verdict})}
                )
                self.dialogue.append({"role": "assistant", "content": "[rejected " + name + "] " + verdict[:200]})
                again = self.ask_model()
                if again is None:
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = decode_args(call)
                content = again.get("content") or ""
            if verdict == "skip":
                continue
            self.reject_run = 0
            if name not in self.toolset:
                continue
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.cart:
                    pid, sku = self.cart[-1]
                offer = self.offers.get(pid + "::" + sku) or Offer(pid, sku)
                args = self.order_args(offer)
            if name == "message":
                if not self.q_pending:
                    self.questions += 1
                self.q_pending = False
                self.q_waits = 0
            obs = self.enact(name, args, content, call.get("id"))
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") or {}
                if added.get("product_id"):
                    self.after_add = (str(added["product_id"]), str(added["sku"]))
            if name in ("search", "filter"):
                self.searches += 1
                if self.need.family == "recall" and self.need.need_ids:
                    have, needn = len(self.identities), self.recall_need()
                    extra = "" if have >= needn else " Keep searching with different wording."
                    self.whisper(
                        "Distinct matching products surfaced so far: %d of the required %d." % (have, needn) + extra
                    )
            if name == "inspect_cart":
                self.cart_looks += 1
            if name in ("view", "compare"):
                self.shortlist()
        return self.dialogue
def agent_main(problem_data: dict) -> list:
    try:
        return MatrixShop(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]
__all__ = ["agent_main"]
