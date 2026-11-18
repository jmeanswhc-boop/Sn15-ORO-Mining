"""ORO-Bench agent: three-deep fallback bench.

A model scouts; a bench of at most three verified (product_id, sku) seats is
what the controller buys from. First add waits for two seats so a post-cart
shock has a successor. Evidence is read as the environment grades it.
Stepper chain over ShopOrder + SeenWorld + SpecEye + Bench — not Brief/Ledger/_plan.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient


# Allowlisted models only; SANDBOX_MODEL may pin first.
_ROSTER = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_HTTP = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_SAND_S = 270.0
_OBS_CAP = 7000
_NUDGE_CAP = 2
_SLACK = 1e-9
_BENCH_MAX = 3
_LATE_FRAC = 0.40
_SHOCK_LEFT = 40.0
_SEAT_TURN = 8

_MONEY_ATOM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_MONEY_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _MONEY_ATOM
    + r"\s*(PHP|php|₱|pesos?)?",
    re.I,
)
_PAIR_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTE_RE = re.compile(r"\"([^\"]{2,80})\"")
_THEME_RE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_NEED_RE = re.compile(r"surface at least (\d+) distinct", re.I)
_PREF_RE = re.compile(r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
_SOFT_RE = re.compile(
    r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)"
)
_SHELF_RE = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
_SPECISH_RE = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b",
    re.I,
)
_WHY_RE = re.compile(r"\b(?:why|reason|explain|justif)", re.I)
_ASK_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)

# Family signs are structural (rules text), same detection contract as the source.
_LANE_MARKS = (
    ("preference", "listing_id"),
    ("constraint", "exact variant"),
    ("justification", "trustworthy reason"),
    ("intent", "nice-to-haves"),
)

_LANE_BRIEFS = {
    "intent": (
        "Hard gates: shelf, named hardware, budget, stock. A brand is a soft lean — "
        "take it when a legal unit of that house exists, else any legal house. Filter, "
        "then view so SpecEye can read the variant. A later cheaper budget binds later seats."
    ),
    "recall": (
        "Issue differently worded in-stock searches (k=10) until enough distinct titles "
        "carry every requested word as a whole token. The bench will not commit while "
        "that count is short. View the strongest hits so a seat can form."
    ),
    "constraint": (
        "One hardware gate is hidden until you ask. After it is spoken, the chosen SKU's "
        "own option string beats any product-wide number. Filter and view. If the seated "
        "SKU dies, move to another product, not a twin of the dead listing."
    ),
    "preference": (
        "Ask which wish ranks first. Compare products that do not share a listing_id. "
        "The first cart line is graded. A shock replacement must come from a different "
        "listing_id than the dead seat."
    ),
    "ranking": (
        "Only the named product_id::sku pairs are legal. Compare them so facts land in "
        "SeenWorld. The bench ranks by advertised max Hz, then inches, then price. After "
        "a shock the survivors are re-ranked."
    ),
    "recovery": (
        "Hunt the named model plus color first. Seat 0 stays at relax 0 until that hunt "
        "is exhausted. Same model, other color is the first relaxation; category, budget "
        "and stock never relax."
    ),
    "justification": (
        "Need a unit whose own labelled facts state the asked spec, on the asked shelf "
        "and house, in stock and inside the cap. View it, seat it, order with claims "
        "copied only from those facts."
    ),
    "generic": "Read the hard gates, scout, view candidates, and let the bench seat and buy.",
}

_CHARTER = (
    "You scout a shop catalog through tools. One tool call per turn, thirty turns at "
    "most. A three-seat bench controller inspects, carts and orders; you surface and "
    "read products so two verified seats exist before anyone carts. Never placeholder-"
    "add — the first add_to_cart is graded. Prices and stock move after that add; the "
    "controller re-reads the seated SKU. Answer every direct shopper question with "
    "message and nothing else that turn. Do not narrate progress or ask for more budget. "
    "Do not repeat an identical tool call; one compare of up to five ids beats stacked views."
)


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + ' ... (truncated by agent)"}'


def _fn_args(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


class Move:
    """One environment action the controller or the model wants to fire."""

    __slots__ = ("name", "args", "note")

    def __init__(self, name: str, args: dict | None = None, note: str = "") -> None:
        self.name = name
        self.args = args or {}
        self.note = note


class SpecEye:
    """Environment-faithful evidence. Class methods only; disagreement is nothing."""
    INCH = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
    HZ = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
    WXH = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
    GB = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
    PAIR = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
    PAIR_HINT = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
    MEM = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
    QHD_BAD = re.compile(r"\bqHD\b")
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
    FACT_TAG = {
        "screen_size": "size",
        "resolution": "display",
        "refresh_rate": "refresh_rate",
        "ram": "storage",
    }

    @classmethod
    def inch_set(cls, text: str) -> set:
        return {"%gin" % float(m.group(1)) for m in cls.INCH.finditer(text or "")}

    @classmethod
    def hz_set(cls, text: str) -> set:
        return {"%ghz" % float(m.group(1)) for m in cls.HZ.finditer(text or "")}

    @classmethod
    def gb_set(cls, text: str) -> set:
        out = set()
        for m in cls.GB.finditer(text or ""):
            n = float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1)
            out.add("%ggb" % n)
        return out

    @classmethod
    def pixel_band(cls, long_side: int, short_side: int) -> str | None:
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
    def res_set(cls, text: str) -> set:
        text = text or ""
        if cls.QHD_BAD.search(text):
            return set()
        low = text.lower()
        found: set = set()
        for m in cls.WXH.finditer(low):
            pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
            band = cls.pixel_band(pair[0], pair[1])
            if band:
                found.add(band)
        if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
            found.add("4k")
        if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
            found.add("qhd")
        if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
            found.add("fhd")
        if re.search(r"\b(?:hd|720p?)\b", re.sub(r"\b(?:full|ultra) hd\b", "", low)):
            found.add("hd")
        return found

    @classmethod
    def _sets(cls) -> dict:
        return {
            "screen_size": cls.inch_set,
            "refresh_rate": cls.hz_set,
            "resolution": cls.res_set,
            "ram": cls.gb_set,
        }

    @classmethod
    def adjacent(cls, field: str, text: str) -> set:
        """Values a free-form string asserts; RAM needs a memory word or a pair."""
        text = text or ""
        if field != "ram":
            return cls._sets()[field](text)
        paired = {"%ggb" % float(a) for a, _b in cls.PAIR.findall(text)}
        return paired or {"%ggb" % float(n) for n in cls.MEM.findall(text)}

    @classmethod
    def mentions(cls, field: str, text: str) -> bool:
        text = text or ""
        if field == "ram":
            return bool(cls.PAIR_HINT.search(text) or cls.MEM.search(text))
        if field == "screen_size":
            return bool(cls.INCH.search(text))
        if field == "refresh_rate":
            return bool(cls.HZ.search(text))
        return bool(
            cls.WXH.search(text)
            or re.search(
                r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b",
                text,
                re.I,
            )
        )

    @classmethod
    def only(cls, values: set) -> str | None:
        return next(iter(values)) if len(values) == 1 else None

    @classmethod
    def wanted(cls, field: str | None, text: str) -> str | None:
        if not field:
            return None
        return cls.only(cls._sets()[field](text or ""))

    @classmethod
    def tokens(cls, text: str) -> set:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    @classmethod
    def same_noun(cls, a: str, b: str) -> bool:
        x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
        y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
        return bool(x) and x == y

    @classmethod
    def clean_identity(cls, text: str, title: bool = False) -> str:
        text = cls.NEGATED.sub("", text or "")
        return cls.BUNDLE.split(text, maxsplit=1)[0] if title else text

    @classmethod
    def phone_verdict(cls, facts: dict, title: str, options: str) -> str:
        """'yes' / 'no' / 'unclear' from public evidence only."""
        if cls.PART_ONLY.search(options or ""):
            return "no"
        facts = facts if isinstance(facts, dict) else {}
        raw_title = str(title or "")
        head = cls.clean_identity(raw_title, title=True)
        kind = " ".join(
            cls.clean_identity(str(facts.get(k) or ""))
            for k in ("phone_type", "device_type", "description_device_type")
        )
        blob = head + " " + kind
        if cls.NOT_PHONE.search(blob):
            return "no"
        if any(
            "[truncated]" in str(facts.get(k) or "")
            for k in ("phone_type", "device_type", "device_operating_system")
        ):
            return "unclear"
        if cls.IS_PHONE.search(blob):
            return "yes"
        os_blob = " ".join(
            cls.clean_identity(str(facts.get(k) or ""))
            for k in ("device_operating_system", "description_operating_system")
            if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
        )
        if cls.OS.search(os_blob):
            return "yes"
        if cls.OS.search(head) and not re.search(r"\bcompatib", raw_title, re.I):
            return "yes"
        return "unclear"

    @classmethod
    def peel_model_codes(cls, model_blob: str, title: str) -> str:
        """Drop model codes so a product name is not read as a panel class."""
        for token in re.split(r"[;,]", str(model_blob or "")):
            token = token.strip()
            stem = re.split(r"[-_]", token)[0] if token else ""
            if (
                token
                and re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)+", token, re.I)
                and re.search(r"[a-z]", stem, re.I)
                and re.search(r"\d", stem)
                and not re.fullmatch(r"(?:[48]k|[ufq]?hd|fullhd|ultrahd|\d+p)\d*", stem, re.I)
            ):
                title = re.sub(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", "", title, flags=re.I)
        return title

    @classmethod
    def fact_read(cls, facts: dict, field: str) -> str | None:
        """Labelled fact, including a bare number that inherits the label's unit."""
        raw = facts.get(cls.FACT_TAG[field]) if isinstance(facts, dict) else None
        if not raw:
            return None
        text = str(raw).strip()
        found = cls._sets()[field](text)
        if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text):
            unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[field]
            return "%g" % float(text) + unit
        return cls.only(found)

    @classmethod
    def page_read(cls, facts: dict, title: str, field: str, model_blob: str = "") -> str | None:
        """Product-wide value: labelled facts plus title; disagreement is nothing."""
        vals: set = set()
        conflict = False
        raw = facts.get(cls.FACT_TAG[field]) if isinstance(facts, dict) else None
        if raw:
            found = cls._sets()[field](str(raw))
            conflict = conflict or len(found) > 1
            if field == "resolution" and cls.QHD_BAD.search(str(raw)):
                conflict = True
            vals |= found
        use_title = title or ""
        if field == "resolution":
            use_title = cls.peel_model_codes(model_blob, use_title)
            if cls.QHD_BAD.search(use_title):
                conflict = True
        vals |= cls.adjacent(field, use_title)
        return None if conflict else cls.only(vals)

    @classmethod
    def sku_read(cls, facts: dict, title: str, options: str, field: str, model_blob: str = "") -> str | None:
        """Variant value: its own option text wins; an ambiguous mention blocks."""
        vals = cls.adjacent(field, options or "")
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if cls.mentions(field, options or ""):
            return None
        return cls.page_read(facts, title, field, model_blob)

    @classmethod
    def measure(cls, facts: dict, title: str, options: str, field: str, model_blob: str = "") -> float | None:
        value = cls.sku_read(facts, title, options, field, model_blob)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    @classmethod
    def stems(cls, text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

    @classmethod
    def shelf_ok(cls, want: str | None, qualifier: str | None, path: list, title: str) -> bool:
        if not want or not path:
            return True
        wanted = cls.stems(want)
        labels: set = set()
        for label in path:
            labels |= cls.stems(str(label))
        placed = bool(wanted & labels) or any(
            cls.stems("".join(re.findall(r"[a-z]+", str(label).lower()))) == wanted for label in path
        )
        if not placed and "smartphone" in wanted:
            placed = any(
                re.fullmatch(r"(?:smart|cell|mobile|unlocked)?\s*phones?", str(label).strip(), re.I)
                for label in path
            )
        if not placed:
            return False
        if not qualifier:
            return True
        words = cls.stems(title) | cls.tokens(title)
        if qualifier == "gaming":
            fast = any(float(v[:-2]) >= 120 for v in cls.hz_set(title))
            return "monitor" in words and ("gaming" in words or fast)
        return "over" in words and "ear" in words or "overear" in words or "circumaural" in words

    @classmethod
    def maker_ok(cls, want: str | None, brand: str, title: str) -> bool:
        if not want:
            return True
        wanted = cls.tokens(want)
        return bool(wanted <= cls.tokens(brand) or wanted <= cls.tokens(title))

    @classmethod
    def gadget_ok(cls, category: str | None, family: str, facts: dict, title: str, options: str) -> bool:
        if not cls.same_noun(category or "", "smartphone"):
            return True
        verdict = cls.phone_verdict(facts, title, options)
        if family in ("constraint", "justification", "recovery"):
            return verdict == "yes"
        return verdict != "no"

    @classmethod
    def resale(cls, title: str) -> bool:
        return bool(
            re.search(
                r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed",
                title or "",
                re.I,
            )
        )

    @classmethod
    def closeness(cls, pref_model: str | None, pref_color: str | None, title: str, facts: dict, options: str) -> int:
        """0 exact model+color, 1 same model, 2 same color only, 3 neither."""
        if not pref_model:
            return 0

        def squash(t: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", (t or "").lower())

        blob = squash(title + " " + " ".join(str(v) for v in (facts or {}).values()))
        same_model = squash(pref_model) in blob
        shade = ((options or "") + " " + str((facts or {}).get("color") or "") + " " + (title or "")).lower()
        same_color = bool(pref_color) and re.search(r"(?<![a-z])" + re.escape(pref_color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    @classmethod
    def extras_missed(cls, specs: dict, gate_field: str | None, facts: dict, title: str, options: str, model_blob: str) -> int:
        return sum(
            1
            for f, v in specs.items()
            if f != gate_field and cls.sku_read(facts, title, options, f, model_blob) != v
        )

    @classmethod
    def proven_flag(cls, facts: dict, title: str, options: str) -> int:
        return 0 if cls.phone_verdict(facts, title, options) == "yes" else 1

    @classmethod
    def brand_near(cls, pref_brand: str | None, brand: str, title: str) -> int:
        if not pref_brand:
            return 1
        wanted = cls.tokens(pref_brand)
        return 0 if (wanted <= cls.tokens(brand) or wanted <= cls.tokens(title)) else 1

    @classmethod
    def score(cls, order: "ShopOrder", world: "SeenWorld", pid: str, sku: str) -> dict:
        """Grade one SKU against every gate the runtime will apply."""
        key = (str(pid), str(sku))
        st = world.snap_of(pid, sku) or {}
        price, stock = st.get("price"), st.get("in_stock")
        fam = order.family
        dead = key in world.dead_pairs or (str(pid) in world.dead_pids and fam != "recovery")
        live = bool(stock) and price is not None and (order.cap is None or price <= order.cap + _SLACK) and not dead
        viewed = str(pid) in world.pages
        field, want = order.gate_field, order.gate_value
        facts = world.facts_of(pid)
        title = world.title_of(pid)
        options = world.options_of(pid, sku)
        model_blob = str(facts.get("model") or "")
        spec = 2
        if field and want:
            got = cls.sku_read(facts, title, options, field, model_blob)
            spec = 2 if got == want else (-1 if got else 0)
        brand = world.brand_of(pid)
        path = world.path_of(pid)
        ok = bool(
            live
            and viewed
            and spec == 2
            and cls.maker_ok(order.house, brand, title)
            and cls.gadget_ok(order.shelf, fam, facts, title, options)
        )
        cat_ok = cls.shelf_ok(order.shelf, order.flavor, path, title)
        ok = ok and cat_ok
        if fam == "justification":
            ok = ok and not cls.resale(title) and bool(brand)
            if field and want:
                ok = ok and cls.fact_read(facts, field) == want and cls.page_read(facts, title, field, model_blob) == want
        if fam == "preference" and world.shock_pair:
            dead_listing = world.listing_of(world.shock_pair[0])
            if dead_listing and world.listing_of(str(pid)) == dead_listing:
                ok = False
        if fam == "recall" and order.theme:
            ok = ok and all(t in cls.tokens(title) for t in order.theme)
        if fam == "ranking":
            ok = ok and (str(pid), str(sku)) in order.named
            hz = cls.measure(facts, title, options, "refresh_rate", model_blob)
            inch = cls.measure(facts, title, options, "screen_size", model_blob)
            ok = ok and hz is not None and inch is not None
        return {
            "pid": str(pid),
            "sku": str(sku),
            "key": key,
            "price": price,
            "live": live,
            "viewed": viewed,
            "spec": spec,
            "cat_ok": cat_ok,
            "ok": ok,
            "relax": cls.closeness(order.pref_model, order.pref_color, title, facts, options) if fam == "recovery" else 0,
        }

    @classmethod
    def rank_tuple(cls, order: "ShopOrder", world: "SeenWorld", row: dict, first: bool) -> tuple:
        """Family sort: ranking Hz/inch/price; recovery relax/price; intent near/extra/proof/price."""
        fam = order.family
        price = row["price"] if row["price"] is not None else 1e12
        pid, sku = row["pid"], row["sku"]
        facts = world.facts_of(pid)
        title = world.title_of(pid)
        options = world.options_of(pid, sku)
        model_blob = str(facts.get("model") or "")
        if fam == "ranking":
            hz = cls.measure(facts, title, options, "refresh_rate", model_blob) or 0.0
            inch = cls.measure(facts, title, options, "screen_size", model_blob) or 0.0
            return (-hz, -inch, price, row["key"])
        if fam == "recovery":
            return (row["relax"], price, row["key"])
        extra = cls.extras_missed(order.hardware, order.gate_field, facts, title, options, model_blob)
        proven = cls.proven_flag(facts, title, options)
        kin = 0
        if world.shock_pair and not first:
            dead_title = world.title_of(world.shock_pair[0])
            kin = -len(cls.tokens(title) & cls.tokens(dead_title))
        if fam == "intent" and first and order.soft_house:
            near = cls.brand_near(order.soft_house, world.brand_of(pid), title)
            return (near, extra, proven, price, row["key"])
        return (kin, extra, proven, price, row["key"])

    @classmethod
    def claims(cls, order: "ShopOrder", world: "SeenWorld", pid: str, sku: str) -> list:
        out = []
        field, want = order.gate_field, order.gate_value
        facts = world.facts_of(pid)
        title = world.title_of(pid)
        model_blob = str(facts.get("model") or "")
        if not field and order.family == "justification":
            field, want = "resolution", cls.fact_read(facts, "resolution")
        if not field or not want:
            return []
        stated = dict(order.hardware) or {field: want}
        stated.setdefault(field, want)
        for name, value in stated.items():
            if cls.fact_read(facts, name) == value and cls.page_read(facts, title, name, model_blob) == value:
                out.append({"field": name, "value": value})
        if not out:
            return []
        st = world.snap_of(pid, sku) or {}
        if order.cap is not None:
            out.append({"field": "price", "value": "at most %.2f" % order.cap})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        out.append({"field": "in_stock", "value": "true"})
        brand = world.brand_of(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out


class ShopOrder:
    """Parsed public request. Same family detection, wants bag instead of Brief."""
    def __init__(self, query: str) -> None:
        self.raw = query or ""
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.wants: dict[str, Any] = {}
        self._ingest()

    def _ingest(self) -> None:
        self.wants["cap"] = self._money(self.goal)
        self.wants["ccy"] = self._ccy(self.goal)
        self.wants["named"] = [(pid, sku.rstrip(".,;:")) for pid, sku in _PAIR_RE.findall(self.goal)]
        phrase = _THEME_RE.search(self.goal)
        quoted = _QUOTE_RE.findall(self.goal)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        self.wants["theme"] = re.findall(r"[a-z0-9]+", source.lower())
        sm = _NEED_RE.search(self.rules)
        self.wants["need_n"] = int(sm.group(1)) if sm else 0
        pm = _PREF_RE.search(self.goal)
        self.wants["pref_color"] = pm.group(1).lower() if pm else None
        self.wants["pref_model"] = pm.group(2) if pm else None
        self.wants["shelf"] = self._shelf(self.goal)
        self.wants["house"] = self._house(self.goal, self.wants["shelf"])
        self.wants["soft_house"] = None if pm else self._soft_house(self.goal)
        self.wants["flavor"] = self._flavor(self.goal)
        self.wants["hardware"] = self._hardware(self.goal)
        gate = next(iter(self.wants["hardware"].items()), (None, None))
        self.wants["gate_field"], self.wants["gate_value"] = gate
        self.wants["hidden"] = self.wants["gate_field"] is None and "firm requirement" in self.goal.lower()
        self.wants["family"] = self._lane()
        for key in (
            "cap", "ccy", "named", "theme", "need_n", "pref_color", "pref_model",
            "shelf", "house", "soft_house", "flavor", "hardware", "gate_field",
            "gate_value", "hidden", "family",
        ):
            setattr(self, key, self.wants[key])

    @staticmethod
    def _money(text: str) -> float | None:
        best = None
        for m in _MONEY_RE.finditer(text or ""):
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
    def _ccy(goal: str) -> str:
        m = re.search(r"\b([A-Z]{3})\b", goal or "")
        if m and m.group(1) in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
            return m.group(1)
        return "PHP"

    @staticmethod
    def _shelf(goal: str) -> str | None:
        m = _SHELF_RE.search(goal or "")
        if m:
            return m.group(1).strip()
        m = re.search(
            r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
            goal or "",
            re.I,
        )
        return m.group(1) if m else None

    @staticmethod
    def _house(goal: str, shelf: str | None) -> str | None:
        if not shelf:
            return None
        m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(shelf), goal or "", re.I)
        if not m:
            return None
        words = [w for w in re.findall(r"[\w&.'\-]+", _SPECISH_RE.sub(" ", m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not SpecEye.res_set(w)]
        return " ".join(keep) or None

    @staticmethod
    def _soft_house(goal: str) -> str | None:
        m = _SOFT_RE.search(goal or "")
        if not m:
            return None
        name = m.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return None
        return name

    @staticmethod
    def _flavor(goal: str) -> str | None:
        if re.search(r"\bgaming\s+monitors?\b", goal or "", re.I):
            return "gaming"
        if re.search(r"\bover[\s-]?ear\b", goal or "", re.I):
            return "over ear"
        return None

    @staticmethod
    def _hardware(goal: str) -> dict:
        out = {}
        for field in ("screen_size", "refresh_rate", "resolution", "ram"):
            value = SpecEye.wanted(field, goal)
            if value:
                out[field] = value
        return out

    def _lane(self) -> str:
        if self.wants["named"]:
            return "ranking"
        if self.wants["need_n"]:
            return "recall"
        if self.wants["pref_model"]:
            return "recovery"
        for name, sign in _LANE_MARKS:
            if sign in self.rules:
                return name
        if self.wants["theme"]:
            return "recall"
        if _WHY_RE.search(self.goal):
            return "justification"
        if self.wants["soft_house"]:
            return "intent"
        if self.wants["hidden"] or self.wants["gate_field"] or self.wants["house"]:
            return "constraint"
        return "preference"

    def adopt_hint(self, text: str) -> str | None:
        for field in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = SpecEye.wanted(field, text)
            if value:
                self.wants["gate_field"] = self.gate_field = field
                self.wants["gate_value"] = self.gate_value = value
                self.wants["hardware"][field] = value
                self.hardware[field] = value
                self.wants["hidden"] = self.hidden = False
                return field + " = " + value
        return None

    def shrink_cap(self, text: str) -> float | None:
        new = self._money(text)
        if new is None:
            return None
        old = self.wants.get("cap")
        if old is not None and new >= old:
            return None
        self.wants["cap"] = self.cap = new
        return new


class SeenWorld:
    """Observations keyed by (product_id, sku) pairs, not a Ledger."""
    def __init__(self) -> None:
        self.basket: list = []
        self.snaps: dict = {}
        self.snap_tick: dict = {}
        self.probed: set = set()
        self.pages: dict = {}
        self.rows: dict = {}
        self.opt_text: dict = {}
        self.listings: set = set()
        self.listed_as: dict = {}
        self.hits: set = set()
        self.tick = 0
        self.first_add_tick: int | None = None
        self.last_add_tick: int | None = None
        self.add_n = 0
        self.shock_pair: tuple | None = None
        self.dead_pairs: set = set()
        self.dead_pids: set = set()
        self.fingerprints: set = set()

    @staticmethod
    def pair(pid: Any, sku: Any) -> tuple:
        return (str(pid), str(sku or ""))

    def remember(self, tool: str, args: dict, obs: dict, order: ShopOrder) -> None:
        self.tick += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        handler = {
            "search": self._from_listing,
            "filter": self._from_listing,
            "view": self._from_page,
            "compare": self._from_compare,
            "inspect_stock": self._from_inspect,
            "add_to_cart": self._from_cart_add,
            "remove_from_cart": self._from_remove,
            "inspect_cart": self._from_cart_look,
        }.get(tool)
        if handler:
            handler(obs, order)

    def _from_listing(self, obs: dict, order: ShopOrder) -> None:
        for row in obs.get("results") or []:
            if isinstance(row, dict):
                self._note_row(row, order)

    def _from_page(self, obs: dict, order: ShopOrder) -> None:
        self._note_page(obs, order)

    def _from_compare(self, obs: dict, order: ShopOrder) -> None:
        for prod in obs.get("compared") or []:
            if isinstance(prod, dict):
                self._note_page(prod, order)

    def _from_inspect(self, obs: dict, order: ShopOrder) -> None:
        key = self.pair(obs.get("product_id"), obs.get("sku"))
        self.snaps[key] = {"in_stock": bool(obs.get("in_stock")), "price": _as_float(obs.get("price"))}
        self.snap_tick[key] = self.tick
        self.probed.add(key)

    def _from_cart_add(self, obs: dict, order: ShopOrder) -> None:
        added = obs.get("added") or {}
        pid, sku = added.get("product_id"), added.get("sku")
        if pid and sku:
            self.add_n += 1
            key = (pid, sku)
            if key not in self.basket:
                self.basket.append(key)
            if self.first_add_tick is None:
                self.first_add_tick = self.tick
            self.last_add_tick = self.tick
        self._sync_basket(obs)

    def _from_remove(self, obs: dict, order: ShopOrder) -> None:
        self._sync_basket(obs)
        removed = obs.get("removed") or {}
        pair = (removed.get("product_id"), removed.get("sku"))
        if pair in self.basket:
            self.basket.remove(pair)

    def _from_cart_look(self, obs: dict, order: ShopOrder) -> None:
        self._sync_basket(obs)

    def _sync_basket(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [(r.get("product_id"), r.get("sku")) for r in rows if isinstance(r, dict) and r.get("product_id")]
        if pairs or rows == []:
            self.basket = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            key = self.pair(r.get("product_id"), r.get("sku"))
            price = _as_float(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.snaps.get(key, {})
            self.snaps[key] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.snap_tick[key] = self.tick

    def _note_row(self, row: dict, order: ShopOrder) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        if order.theme:
            if all(t in SpecEye.tokens(title) for t in order.theme):
                self.hits.add(str(pid))
        else:
            self.hits.add(str(pid))
        if not row.get("sku"):
            return
        key = self.pair(pid, row.get("sku"))
        self.rows[key] = row
        if row.get("price") is not None:
            self.snaps.setdefault(key, {"in_stock": bool(row.get("in_stock", True)), "price": _as_float(row.get("price"))})

    def _note_page(self, prod: dict, order: ShopOrder) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.pages[str(pid)] = prod
        if prod.get("listing_id"):
            self.listings.add(str(prod["listing_id"]))
            self.listed_as[str(pid)] = str(prod["listing_id"])
        if order.theme and all(t in SpecEye.tokens(str(prod.get("title") or "")) for t in order.theme):
            self.hits.add(str(pid))
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            key = self.pair(pid, v.get("sku"))
            self.snaps[key] = {"in_stock": bool(v.get("in_stock")), "price": _as_float(v.get("price"))}
            self.snap_tick[key] = self.tick
            self.opt_text[key] = str(v.get("options") or "")

    def snap_of(self, pid: str, sku: str) -> dict | None:
        return self.snaps.get(self.pair(pid, sku))

    def title_of(self, pid: str) -> str:
        title = str((self.pages.get(str(pid)) or {}).get("title") or "")
        if title:
            return title
        for key, row in self.rows.items():
            if key[0] == str(pid):
                return str(row.get("title") or "")
        return ""

    def brand_of(self, pid: str) -> str:
        brand = (self.pages.get(str(pid)) or {}).get("brand")
        if brand:
            return str(brand)
        for key, row in self.rows.items():
            if key[0] == str(pid) and row.get("brand"):
                return str(row["brand"])
        return ""

    def facts_of(self, pid: str) -> dict:
        f = (self.pages.get(str(pid)) or {}).get("facts")
        return f if isinstance(f, dict) else {}

    def path_of(self, pid: str) -> list:
        path = (self.pages.get(str(pid)) or {}).get("category_path")
        return list(path) if isinstance(path, list) else []

    def options_of(self, pid: str, sku: str) -> str:
        return self.opt_text.get(self.pair(pid, sku), "")

    def listing_of(self, pid: str) -> str | None:
        return self.listed_as.get(str(pid))

    def all_pairs(self) -> list:
        return [k for k in self.snaps if k[1]]

    def unread_ids(self, order: ShopOrder, cap: float | None) -> list:
        leads = []
        for key, row in self.rows.items():
            pid, sku = key
            if not sku or str(pid) in self.pages or str(pid) in self.dead_pids:
                continue
            st = self.snaps.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (cap is not None and price > cap + _SLACK):
                continue
            title = str(row.get("title") or "")
            if order.theme and not all(t in SpecEye.tokens(title) for t in order.theme):
                continue
            if order.pref_model and self.add_n == 0 and not self.shock_pair:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", order.pref_model.lower()) not in squashed:
                    continue
            if order.house:
                wanted = SpecEye.tokens(order.house)
                if not (wanted <= SpecEye.tokens(str(row.get("brand") or "")) or wanted <= SpecEye.tokens(title)):
                    continue
            leads.append((price, str(pid)))
        leads.sort()
        out = []
        for _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def mark_dead(self, pid: str, sku: str) -> None:
        self.dead_pairs.add(self.pair(pid, sku))
        self.dead_pids.add(str(pid))
        self.shock_pair = self.pair(pid, sku)


class Bench:
    """Up to three verified seats. Fill: score, family-filter, sort, take 3."""
    def __init__(self) -> None:
        self.seats: list = []

    def depth(self) -> int:
        return len(self.seats)

    def head(self) -> dict | None:
        return self.seats[0] if self.seats else None

    def backup(self) -> dict | None:
        return self.seats[1] if len(self.seats) > 1 else None

    def clear(self) -> None:
        self.seats = []

    def kill_head(self) -> dict | None:
        """Mark-and-promote: drop seat 0 so seat 1 becomes the new head."""
        if not self.seats:
            return None
        return self.seats.pop(0)

    def refill(
        self,
        order: ShopOrder,
        world: SeenWorld,
        first: bool,
        hunt_open: bool,
    ) -> None:
        """Score known pairs, apply family filters, sort, keep three unique seats."""
        rows = []
        for pid, sku in world.all_pairs():
            row = SpecEye.score(order, world, pid, sku)
            if row["ok"]:
                rows.append(row)
        if order.family == "recovery" and first and hunt_open:
            exact = [r for r in rows if r["relax"] == 0]
            rows = exact
        if order.family == "intent" and first and order.soft_house:
            wanted = SpecEye.tokens(order.soft_house)
            named = [
                r
                for r in rows
                if wanted <= SpecEye.tokens(world.brand_of(r["pid"]))
                or wanted <= SpecEye.tokens(world.title_of(r["pid"]))
            ]
            if named:
                rows = named
        rows.sort(key=lambda r: SpecEye.rank_tuple(order, world, r, first))
        seen: set = set()
        filled = []
        for row in rows:
            if row["key"] in seen:
                continue
            seen.add(row["key"])
            filled.append(row)
            if len(filled) >= _BENCH_MAX:
                break
        self.seats = filled

    def may_first_add(self, turn: int, max_steps: int, sand_left: float) -> bool:
        """Forbid first add until two seats exist, unless the episode is already late."""
        if self.depth() >= 2:
            return True
        if self.depth() >= 1 and turn >= _SEAT_TURN:
            return True
        if sand_left < _SHOCK_LEFT:
            return True
        if max_steps and turn >= max_steps * _LATE_FRAC:
            return True
        return False

    def harvest_query(self, world: SeenWorld) -> str | None:
        """First eight title tokens of the dead seat, for a twin/sibling sweep."""
        if not world.shock_pair:
            return None
        title = world.title_of(world.shock_pair[0])
        if not title:
            return None
        words = title.split()[:8]
        return " ".join(words) if words else None

    def bulletin(self) -> str:
        if not self.seats:
            return "Bench is empty: no verified (product_id, sku) seats yet."
        bits = [
            "seat %d %s::%s at %s relax=%s" % (i, r["pid"], r["sku"], r["price"], r.get("relax", 0))
            for i, r in enumerate(self.seats)
        ]
        return "Bench: " + " | ".join(bits)


class BenchAgent:
    """Stepper chain, not a _plan method. Fill bench, inspect, seat, recheck, order."""
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.toolset = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.order = ShopOrder(str(self.policy.get("query") or ""))
        self.world = SeenWorld()
        self.bench = Bench()
        self.t0 = time.monotonic()
        self.transcript: list = []
        self.chat: list = [
            {"role": "system", "content": self._charter()},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.turn = 0
        self.asked = False
        self.owe_reply = False
        self.shopper_moved = False
        self.reject_run = 0
        self.held: dict | None = None
        self.searches = 0
        self.cart_looks = 0
        self.recheck_pair: tuple | None = None
        self.page_pushes = 0
        self.exact_nudge = False
        self.brand_nudge = False
        self.sibling_swept = False
        self.twin_swept = False
        self.catalog_keys: set = set()
        self.pages_exhausted = False
        self.shock_turn: int | None = None
        self.opaque: set = set()
        self.reply_waits = 0
        self.questions = 0
        self.finished = False
        self.wish_bag: set | None = None
        self.slot_ix = 0
        self.hunt_closed = False
        self.shock_stage = ""
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _ROSTER.get(provider) or _ROSTER["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    def sand_left(self) -> float:
        return _SAND_S - (time.monotonic() - self.t0)

    def _charter(self) -> str:
        brief = _LANE_BRIEFS.get(self.order.family, _LANE_BRIEFS["generic"])
        extra = ""
        if self.order.family == "recall" and self.order.need_n:
            extra = (
                " Surface at least %d distinct matching products in search or filter "
                "hits before any bench commit; the controller reports the running count."
                % self.order.need_n
            )
        return _CHARTER + "\n\nLane brief: " + brief + extra

    def _whisper(self, text: str) -> None:
        self.chat.append({"role": "user", "content": "[note] " + text})
        self.transcript.append({"role": "assistant", "content": "[note] " + text[:300]})

    def _first_ask(self) -> str | None:
        if self.order.family == "constraint" or self.order.hidden:
            return (
                "Before I scout: which hardware gate is firm (screen size, resolution "
                "class, refresh rate, or memory), and what value must it resolve to?"
            )
        if self.order.family == "preference":
            return (
                "So I seat the right listing: which preference should dominate, and in "
                "what order should I weigh the rest?"
            )
        return None

    def _rebuild(self) -> None:
        first = self.world.add_n == 0
        hunt_open = (
            self.order.family == "recovery"
            and first
            and not self.hunt_closed
            and self.turn < self.max_steps * 0.6
            and self.sand_left() >= _SHOCK_LEFT
        )
        self.bench.refill(self.order, self.world, first, hunt_open)

    def _recall_bar(self) -> int:
        early = self.searches < 8 and self.turn < self.max_steps * 0.5
        return self.order.need_n + (2 if early else 0)

    def _late_window(self) -> bool:
        return self.sand_left() < _SHOCK_LEFT or (self.max_steps and self.turn >= self.max_steps * _LATE_FRAC)

    def _want_depth(self) -> int:
        if self.world.add_n or self._late_window() or self.turn >= _SEAT_TURN:
            return 1
        if self.sand_left() > 80 and self.turn < self.max_steps * 0.6:
            return 3
        return 2

    def _commit_ok(self) -> bool:
        """Gates that block the first add_to_cart besides bench depth."""
        if self.world.add_n:
            return True
        if self.order.hidden:
            if self.turn < 8:
                return False
            self.order.hidden = False
            self._whisper("The hidden gate was never spoken; remaining gates decide the seat.")
        if self.order.family in ("intent", "constraint") and self.turn < 4 and not self.shopper_moved:
            return False
        if self.order.family == "preference" and len(self.world.listings) < 2:
            return False
        if self.order.family == "recall":
            if len(self.world.hits) < self._recall_bar():
                return False
            if self.searches < 2 and not self._late_window():
                return False
        if not self.bench.may_first_add(self.turn, self.max_steps, self.sand_left()):
            return False
        return True

    def _fresh(self, pid: str, sku: str) -> bool:
        key = self.world.pair(pid, sku)
        if key not in self.world.probed:
            return False
        return self.world.last_add_tick is None or self.world.snap_tick.get(key, 0) > self.world.last_add_tick

    def _order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.order.family == "justification":
            claims = SpecEye.claims(self.order, self.world, pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def _legal_now(self, pid: str, sku: str) -> tuple:
        st = self.world.snap_of(pid, sku)
        key = self.world.pair(pid, sku)
        if st is None:
            return False, "inspect_stock that exact product_id and sku first"
        if (pid, sku) not in self.world.basket:
            return False, "add that exact variant to the cart first"
        if self.world.last_add_tick is not None and self.world.snap_tick.get(key, 0) <= self.world.last_add_tick:
            return False, "re-check the variant with inspect_stock after the cart addition"
        if not st.get("in_stock"):
            return False, "that variant is out of stock; choose a different product"
        price = st.get("price")
        if self.order.cap is not None and price is not None and price > self.order.cap + _SLACK:
            return False, "that variant costs %.2f, above the budget of %.2f" % (price, self.order.cap)
        if key in self.world.dead_pairs:
            return False, "that variant was invalidated by the market change; choose a different product"
        return True, ""

    def _scripted_reply(self) -> str:
        self._rebuild()
        head = self.bench.head()
        if not head:
            return "I am still matching the catalog against your gates; nothing is seated yet."
        pid, sku = head["pid"], head["sku"]
        bits = ["I am going with " + (self.world.title_of(pid) or pid)[:90]]
        field, want = self.order.gate_field, self.order.gate_value
        facts = self.world.facts_of(pid)
        title = self.world.title_of(pid)
        options = self.world.options_of(pid, sku)
        model_blob = str(facts.get("model") or "")
        if field and want and SpecEye.sku_read(facts, title, options, field, model_blob) == want:
            bits.append("its listed " + field.replace("_", " ") + " is " + want)
        if head.get("price") is not None:
            bits.append("it is %.2f %s and in stock" % (head["price"], self.order.ccy))
        return ", ".join(bits) + "."

    def _wish_words(self) -> set:
        if self.wish_bag is None:
            said = self.order.goal + " " + " ".join(
                str(m.get("content") or "") for m in self.chat if m.get("role") == "user"
            )
            drop = SpecEye.tokens(self.order.shelf or "") | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
            }
            self.wish_bag = {w for w in SpecEye.tokens(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self.wish_bag

    def _wish_hits(self, pid: str, sku: str) -> int:
        wants = self._wish_words()
        if not wants:
            return 0
        have = SpecEye.tokens(
            " ".join(str(v) for v in self.world.facts_of(pid).values())
            + " "
            + self.world.title_of(pid)
            + " "
            + self.world.options_of(pid, sku)
        )
        return len(wants & have)

    def _post_shopper(self, text: str) -> None:
        new = self.order.shrink_cap(text)
        if new is not None:
            self.shopper_moved = True
            self._whisper(
                "Budget updated by the shopper: at most %.2f %s. Every later seat must fit it."
                % (new, self.order.ccy)
            )
            for pid, sku in list(self.world.basket):
                st = self.world.snap_of(pid, sku) or {}
                price = st.get("price")
                if price is not None and price > new + _SLACK:
                    self.world.dead_pairs.add(self.world.pair(pid, sku))
                    self._whisper(
                        "The seated item %s::%s costs %.2f, above the new cap; a successor will be seated."
                        % (pid, sku, price)
                    )
        if self.order.family in ("constraint", "intent") and not self.order.gate_value:
            stated = self.order.adopt_hint(text)
            if stated:
                self.shopper_moved = True
                self._whisper("Firm gate recorded: " + stated + ". Only a SKU that resolves to it may be seated.")
        if _ASK_RE.search(text):
            self.owe_reply = True
            self._whisper("The shopper asked a question: answer it with message next, then continue.")

    def _announce_bench(self) -> None:
        self._rebuild()
        if self.order.family == "preference" and not self.world.basket:
            good = [r for pid, sku in self.world.all_pairs() if (r := SpecEye.score(self.order, self.world, pid, sku))["ok"]]
            good.sort(key=lambda r: (-self._wish_hits(r["pid"], r["sku"]), r["price"] if r["price"] is not None else 1e12))
            if good:
                self._whisper(
                    "Legal listings so far: "
                    + "; ".join("%s::%s at %.2f" % (r["pid"], r["sku"], r["price"] or 0.0) for r in good[:6])
                    + ". Pick the one that best fits the stated priorities; the bench will inspect then seat it."
                )
                return
        if self.bench.head():
            self._whisper(self.bench.bulletin() + ". The controller inspects, seats and orders the head.")
            return
        field, want = self.order.gate_field, self.order.gate_value
        blocked = [
            r for pid, sku in self.world.all_pairs()
            if (r := SpecEye.score(self.order, self.world, pid, sku))["viewed"] and r["live"] and not r["ok"]
        ]
        if blocked and field:
            self._whisper(
                "No seat can be verified yet. Evidence for %s must resolve to exactly %s from the "
                "SKU option text or labelled facts: %s. Read different products."
                % (field, want, ", ".join(r["pid"] + "::" + r["sku"] for r in blocked[:3]))
            )
        elif blocked:
            self._whisper("Read products fail a firm gate (shelf, house, stock, budget or device). Scout elsewhere.")

    # ------------------------------------------------------------------ steppers
    def _step_talk(self) -> Move | None:
        if not self.owe_reply:
            return None
        # After a cart line exists the shopper is asking about that line — answer now.
        if self.world.add_n or self.reply_waits >= 2 or "message" not in self.toolset:
            return Move("message", {"content": self._scripted_reply()}, "shopper is owed an answer")
        self.reply_waits += 1
        return None

    def _step_rank_read(self) -> Move | None:
        if self.order.family != "ranking" or "compare" not in self.toolset:
            return None
        missing = [p for p, _s in self.order.named if p not in self.world.pages and p not in self.opaque]
        if not missing:
            return None
        return Move("compare", {"product_ids": missing[:5]}, "named ranking pairs still unread")

    def _step_shock(self) -> Move | None:
        """Death path: promote, harvest title twins, refill, then inspect/order."""
        if not self.world.shock_pair or self.world.add_n == 0:
            return None
        if self.shock_stage == "search":
            q = self.bench.harvest_query(self.world)
            self.shock_stage = "read"
            if q and "search" in self.toolset and not self.twin_swept:
                self.twin_swept = True
                self._whisper("Seat 0 died; seat 1 promoted. Searching the dead title (8 tokens) for twins.")
                return Move("search", {"query": q, "k": 10, "in_stock": True}, "twin harvest")
            return None
        if self.shock_stage == "read":
            leads = self.world.unread_ids(self.order, self.order.cap)
            self.shock_stage = "buy"
            if leads and self.page_pushes < 10:
                self.page_pushes += 1
                if len(leads) > 1 and "compare" in self.toolset:
                    return Move("compare", {"product_ids": leads[:5]}, "read harvested twins")
                return Move("view", {"product_id": leads[0]}, "read harvested twin")
            return None
        return None

    def _step_recovery_hunt(self) -> Move | None:
        if self.order.family != "recovery" or self.world.add_n or not self.order.pref_model:
            return None
        if self.sibling_swept or "search" not in self.toolset:
            return None
        self._rebuild()
        if any(r.get("relax") == 0 for r in self.bench.seats):
            return None
        near = [
            r for pid, sku in self.world.all_pairs()
            if (r := SpecEye.score(self.order, self.world, pid, sku))["viewed"] and r["relax"] == 1
        ]
        title = self.world.title_of(near[0]["pid"]) if near else ""
        if not title:
            return None
        self.sibling_swept = True
        return Move("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True}, "sibling colors")

    def _step_fill_more(self) -> Move | None:
        """Grow the bench toward two (or three when time is ample) verified seats."""
        if self.order.family == "recall" and len(self.world.hits) < self._recall_bar():
            return self._widen()
        self._rebuild()
        if self.world.add_n and self.bench.depth() >= 1:
            return None
        if self.bench.depth() >= self._want_depth():
            return None
        if self._late_window() and self.bench.depth() >= 1:
            return None
        leads = self.world.unread_ids(self.order, self.order.cap)
        if leads and self.page_pushes < 8 and self.turn >= 2:
            self.page_pushes += 1
            if len(leads) > 1 and "compare" in self.toolset:
                return Move("compare", {"product_ids": leads[:5]}, "fill unread seats")
            return Move("view", {"product_id": leads[0]}, "fill unread seat")
        if self.turn < 2:
            return None
        return self._widen()

    def _widen(self) -> Move | None:
        probe = (self.order.house or "", "%.2f" % (self.order.cap or 0.0), str(self.order.gate_value))
        if probe in self.catalog_keys:
            self.pages_exhausted = True
            return None
        self.catalog_keys.add(probe)
        if "filter" in self.toolset and self.order.shelf:
            args: dict[str, Any] = {"category": self.order.shelf, "in_stock": True, "k": 10}
            if self.order.cap is not None:
                args["max_price"] = self.order.cap
            if self.order.house:
                args["brand"] = self.order.house
            return Move("filter", args, "widen catalog under current gates")
        if "search" in self.toolset:
            words = [w for w in (self.order.house, self.order.gate_value, self.order.shelf) if w]
            if words:
                return Move("search", {"query": " ".join(str(w) for w in words), "in_stock": True, "k": 10}, "widen")
        self.pages_exhausted = True
        return None

    def _step_seat(self) -> Move | None:
        self._rebuild()
        head = self.bench.head()
        if head is None:
            if self.order.family == "intent" and self.world.add_n == 0 and self.order.soft_house:
                if self.turn < self.max_steps * 0.6:
                    self._nudge_brand()
            if self.order.family == "recovery" and self.world.add_n == 0 and self.order.pref_model:
                if self.turn < self.max_steps * 0.6:
                    self._nudge_exact()
            return None
        if not self._commit_ok():
            return None
        pid, sku = head["pid"], head["sku"]
        if (pid, sku) in self.world.basket:
            if not self._fresh(pid, sku):
                return Move("inspect_stock", {"product_id": pid, "sku": sku}, "re-read seated head")
            return Move("place_test_order", self._order_args(pid, sku), "order seated head")
        for cpid, csku in self.world.basket:
            if self.world.pair(cpid, csku) in self.world.dead_pairs and "remove_from_cart" in self.toolset:
                return Move("remove_from_cart", {"product_id": cpid, "sku": csku}, "drop dead basket line")
        if not self._fresh(pid, sku):
            return Move("inspect_stock", {"product_id": pid, "sku": sku}, "inspect before add")
        return Move("add_to_cart", {"product_id": pid, "sku": sku}, "seat bench head")

    def _nudge_brand(self) -> None:
        if self.brand_nudge or not self.order.soft_house:
            return
        self.brand_nudge = True
        self._whisper(
            "No %s unit that clears every firm gate has been read. Filter brand=%s "
            "inside the cap and read those pages before any other house."
            % (self.order.soft_house, self.order.soft_house)
        )

    def _nudge_exact(self) -> None:
        if self.exact_nudge or not self.order.pref_model:
            return
        self.exact_nudge = True
        extra = ", " + self.order.pref_color if self.order.pref_color else ""
        self._whisper(
            "The exact preference (%s%s) is not on the bench. Search the model code "
            "alone and view every color; seat 0 must stay at relax 0 until that hunt ends."
            % (self.order.pref_model, extra)
        )

    def next_action(self) -> Move | None:
        if self.finished:
            return None
        steppers: tuple[Callable[[], Move | None], ...] = (
            self._step_talk,
            self._step_rank_read,
            self._step_shock,
            self._step_recovery_hunt,
            self._step_fill_more,
            self._step_seat,
        )
        for stepper in steppers:
            move = stepper()
            if move is not None:
                return move
        return None

    # ------------------------------------------------------------------ inference
    def _query_lm(self) -> dict | None:
        for attempt in range(3):
            model = self.roster[min(self.slot_ix, len(self.roster) - 1)]
            payload = {
                "model": model,
                "messages": self.chat,
                "tools": self.tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            resp = _HTTP.post("/inference/chat/completions", json_data=payload)
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.chat.append({"role": "assistant", "content": msg.get("content") or ""})
                self._whisper("No tool call was made. Choose exactly one of the supplied tools now.")
                continue
            if self.sand_left() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.slot_ix = min(self.slot_ix + 1, len(self.roster) - 1)
        return None

    # ------------------------------------------------------------------ environment
    def _push_env(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            result = _HTTP.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.sand_left() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def _fire(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if self.held is not None:
            pend = self.held
            replay = self._push_env(pend["envelope"])
            if replay is None:
                self._whisper("The environment is silent; the previous action is still pending and will be retried.")
                return None
            self.held = None
            self.turn = pend["turn"]
            self._ingest(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
            if self.finished:
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
        result = self._push_env(envelope)
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
        return self._ingest(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def _ingest(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.chat.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.transcript.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.world.fingerprints.discard(name + json.dumps(args, sort_keys=True))
            self.chat.append(
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
            self.finished = True
        self.world.remember(name, args, obs if isinstance(obs, dict) else {}, self.order)
        self.chat.append({"role": "tool", "tool_call_id": call_id, "content": _clip(json.dumps(obs), _OBS_CAP)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.chat.append({"role": "user", "content": text})
            self._post_shopper(text)
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.finished = True
        return obs if isinstance(obs, dict) else None

    def _after_add_recheck(self) -> None:
        """The market moves on the first cart action; re-read that SKU at once."""
        pid, sku = self.recheck_pair
        self.recheck_pair = None
        obs = self._fire("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = _as_float(obs.get("price"))
        over = self.order.cap is not None and price is not None and price > self.order.cap + _SLACK
        if stock and not over:
            self._whisper(
                "Post-add recheck: %s::%s is in stock at %s. The controller will order it."
                % (pid, sku, price)
            )
            return
        self.world.mark_dead(pid, sku)
        fallen = self.bench.kill_head()
        if fallen and (fallen["pid"], fallen["sku"]) != (str(pid), str(sku)):
            self.bench.seats.insert(0, fallen)
            self.bench.kill_head()
        self.page_pushes = 0
        self.twin_swept = False
        self.shock_stage = "search"
        self.shock_turn = self.turn
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the cap" % (price or 0.0)
        tips = {
            "recovery": "Smallest change first: same model, other color, then any other model.",
            "ranking": "Surviving named pairs are re-ranked; the new head is taken.",
            "preference": "Successor must have a different listing_id than the dead seat.",
            "constraint": "Another product that still meets every gate, not a twin of the dead one.",
        }.get(self.order.family, "Seat a different product that still clears every firm gate.")
        self._whisper("Market shock: seated %s::%s %s. " % (pid, sku, reason) + tips)
        self._rebuild()
        self._announce_bench()

    def _sprint_one(self, pid: str, sku: str) -> bool:
        """Inspect → add → post-add inspect → order one seat. False if that seat dies."""
        if not self._fresh(pid, sku):
            obs = self._fire("inspect_stock", {"product_id": pid, "sku": sku})
            if self.finished or not isinstance(obs, dict) or not obs.get("in_stock"):
                return False
            price = _as_float(obs.get("price"))
            if self.order.cap is not None and price is not None and price > self.order.cap + _SLACK:
                return False
        if (pid, sku) not in self.world.basket:
            obs = self._fire("add_to_cart", {"product_id": pid, "sku": sku})
            if self.finished or not isinstance(obs, dict) or obs.get("error"):
                return False
            obs = self._fire("inspect_stock", {"product_id": pid, "sku": sku})
            if self.finished or not isinstance(obs, dict) or not obs.get("in_stock"):
                return False
            price = _as_float(obs.get("price"))
            if self.order.cap is not None and price is not None and price > self.order.cap + _SLACK:
                return False
        ok, _why = self._legal_now(pid, sku)
        if not ok:
            return False
        self._fire("place_test_order", self._order_args(pid, sku))
        return self.finished

    def _sprint(self) -> None:
        """Out of sand or turns: reach a legal order in as few fires as possible."""
        for pid, sku in reversed(self.world.basket):
            ok, _why = self._legal_now(pid, sku)
            if ok and not self.finished:
                self._fire("place_test_order", self._order_args(pid, sku))
                return
        if self.finished:
            return
        self._rebuild()
        good = list(self.bench.seats)
        if not good:
            scored = [SpecEye.score(self.order, self.world, pid, sku) for pid, sku in self.world.all_pairs()]
            good = [r for r in scored if r["ok"]]
            if not good and self.order.family != "justification":
                good = [
                    r
                    for r in scored
                    if r["live"]
                    and r["spec"] == 2
                    and SpecEye.maker_ok(self.order.house, self.world.brand_of(r["pid"]), self.world.title_of(r["pid"]))
                    and r["cat_ok"]
                    and (not self.order.named or (r["pid"], r["sku"]) in self.order.named)
                ]
            good.sort(key=lambda r: SpecEye.rank_tuple(self.order, self.world, r, self.world.add_n == 0))
        for row in good[:4]:
            if self.finished:
                return
            if self._sprint_one(row["pid"], row["sku"]):
                return

    def _gate(self, name: str, args: dict) -> str | None:
        if name not in self.toolset:
            return "unknown tool; choose one of " + str(sorted(n for n in self.toolset if n))
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.world.basket:
                pid, sku = self.world.basket[-1]
            ok, why = self._legal_now(pid, sku)
            if not ok:
                return "order refused: " + why
            if self.order.family == "recall" and len(self.world.hits) < self._recall_bar():
                return (
                    "order refused: only %d distinct matching products have been surfaced; search with different "
                    "wording until at least %d appear" % (len(self.world.hits), self._recall_bar())
                )
            if self.order.family == "preference" and len(self.world.listings) < 2:
                return "order refused: compare at least two products with different listing_id values first"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            row = SpecEye.score(self.order, self.world, pid, sku)
            key = self.world.pair(pid, sku)
            if not row["live"]:
                return "that variant is out of stock, over budget, or already invalidated"
            if not row["viewed"]:
                return "view or compare that product first so its raw facts are on record"
            if key not in self.world.probed:
                return "inspect_stock that exact variant first: current price and stock decide whether it can be bought"
            if row["spec"] != 2 and self.order.gate_field:
                return (
                    "that variant does not resolve to %s = %s from its own option text or raw facts"
                    % (self.order.gate_field, self.order.gate_value)
                )
            if not row["ok"]:
                return "that product fails a firm requirement (category, brand, or the kind of device requested)"
            if self.order.family == "preference" and len(self.world.listings) < 2:
                return "compare two products with different listing_id values before the first cart addition"
            if self.order.family == "recovery" and not self.world.add_n and self.order.pref_model and row["relax"]:
                return (
                    "seat 0 must be the exact model and color; other colors are separate listings sharing "
                    "the title, so search that title and read them first"
                )
            if self.order.family == "recall" and len(self.world.hits) < self._recall_bar():
                return (
                    "hold: only %d distinct matching identities so far; need %d before a bench commit"
                    % (len(self.world.hits), self._recall_bar())
                )
            if not self._commit_ok():
                return (
                    "hold: the bench needs two verified seats before the first add_to_cart "
                    "(last 20 percent of turns or under 40s remaining may waive this)"
                )
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.owe_reply and "?" not in text:
                return "do not send status updates; use message only to answer the shopper or to ask one question"
            if not self.owe_reply and self.questions >= 2:
                return "the shopper has already answered; decide from the evidence rather than asking again"
        if name == "inspect_cart" and self.cart_looks >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.world.fingerprints and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        self.world.fingerprints.add(key)
        return None

    def _controller_step(self, opener: str | None) -> Move | None:
        if opener and not self.asked and "message" in self.toolset:
            self.asked = True
            return Move("message", {"content": opener}, "opening question")
        if self.recheck_pair and "inspect_stock" in self.toolset:
            return Move("__recheck__", {}, "post-add recheck")
        return self.next_action()

    def _apply(self, move: Move) -> None:
        if move.name == "__recheck__":
            self._after_add_recheck()
            return
        obs = self._fire(move.name, move.args)
        if move.name == "message":
            self.owe_reply = False
            self.reply_waits = 0
        if move.name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
            added = obs.get("added") or {}
            if added.get("product_id"):
                self.recheck_pair = (str(added["product_id"]), str(added["sku"]))
        if move.name in ("search", "filter"):
            self.searches += 1
            if self.order.family == "recall" and self.order.need_n:
                have, need = len(self.world.hits), self._recall_bar()
                self._whisper(
                    "Distinct matching products surfaced so far: %d of the required %d."
                    % (have, need)
                    + ("" if have >= need else " Keep searching with different wording.")
                )
        if move.name == "inspect_cart":
            self.cart_looks += 1
        if move.name in ("view", "compare"):
            self._announce_bench()

    def run(self) -> list:
        opener = self._first_ask()
        last_sig, repeats = None, 0
        while not self.finished and self.turn < self.max_steps:
            if self.sand_left() < 25 or self.turn >= self.max_steps - 3:
                self._sprint()
                if self.finished:
                    break
            move = self._controller_step(opener)
            opener = None
            if move is not None and move.name != "__skip__":
                signature = move.name + json.dumps(move.args, sort_keys=True)
                repeats = repeats + 1 if signature == last_sig else 1
                last_sig = signature
                if repeats > 2:
                    if move.name in ("inspect_stock", "add_to_cart"):
                        self.world.dead_pairs.add(
                            self.world.pair(move.args.get("product_id"), move.args.get("sku"))
                        )
                    if move.name == "compare":
                        self.opaque.update(str(p) for p in move.args.get("product_ids") or [])
                    if move.name == "message":
                        self.owe_reply = False
                    last_sig, repeats = None, 0
                    self._whisper("That step changed nothing, so the bench moved on to other seats.")
                    continue
                self._apply(move)
                continue
            last_sig, repeats = None, 0
            proposal = self._query_lm()
            if proposal is None:
                self._sprint()
                if self.finished:
                    break
                continue
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = _fn_args(call)
            content = proposal.get("content") or ""
            nudges = 0
            verdict: str | None = None
            while True:
                verdict = self._gate(name, args)
                if verdict is None:
                    break
                if nudges >= _NUDGE_CAP:
                    self.reject_run += 1
                    self._whisper("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if self.reject_run >= 3:
                        self._announce_bench()
                        self.reject_run = 0
                    verdict = "skip"
                    break
                nudges += 1
                self.chat.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.chat.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id") or "x",
                        "content": json.dumps({"rejected": verdict}),
                    }
                )
                self.transcript.append({"role": "assistant", "content": "[rejected " + name + "] " + verdict[:200]})
                again = self._query_lm()
                if again is None:
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = _fn_args(call)
                content = again.get("content") or ""
            if verdict == "skip":
                continue
            self.reject_run = 0
            if name not in self.toolset:
                continue
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.world.basket:
                    pid, sku = self.world.basket[-1]
                args = self._order_args(pid, sku)
            if name == "message":
                if not self.owe_reply:
                    self.questions += 1
                self.owe_reply = False
                self.reply_waits = 0
            obs = self._fire(name, args, content, call.get("id"))
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") or {}
                if added.get("product_id"):
                    self.recheck_pair = (str(added["product_id"]), str(added["sku"]))
            if name in ("search", "filter"):
                self.searches += 1
                if self.order.family == "recall" and self.order.need_n:
                    have, need = len(self.world.hits), self._recall_bar()
                    self._whisper(
                        "Distinct matching products surfaced so far: %d of the required %d."
                        % (have, need)
                        + ("" if have >= need else " Keep searching with different wording.")
                    )
            if name == "inspect_cart":
                self.cart_looks += 1
            if name in ("view", "compare"):
                self._announce_bench()
        return self.transcript


def agent_main(problem_data: dict) -> list:
    try:
        return BenchAgent(problem_data).run()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]

