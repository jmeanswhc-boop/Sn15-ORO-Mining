from __future__ import annotations

import json
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
_PROBLEM_BUDGET_S = 270.0
_OBS_CHARS = 7000
_MAX_REPROMPTS = 2
_EPS = 1e-9

_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|at most|no higher than|ceiling of|keep (?:it )?below)\s*"
    + _NUM,
    re.I,
)
_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_CURRENCY_RE = re.compile(r"\b([A-Z]{3})\b")
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)

# Generic unit extractors (normalize stated numbers — not attribute ontologies).
_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_PAIR_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_PAIR_HINT_RE = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
_MEM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_QUARTER_RE = re.compile(r"\bqHD\b")
_CAPACITY_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(ml|millilit(?:er|re)s?|l|lit(?:er|re)s?)(?!\w)", re.I)
_WATTAGE_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(kw|kilowatts?|w|watts?)(?!\w)", re.I)
_WARRANTY_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(years?|yrs?|months?|mos?)(?!\w)", re.I)
_PACK_RE = re.compile(r"(?<![\w.])(\d+)\s*(?:-\s*)?(?:cts?|pcs?|pieces?|count|units?|pack(?!s))(?!\w)", re.I)
# Thin accessory-title gate (generic shopping English — not suite synonym banks).
_ACCESSORY_TITLE_RE = re.compile(
    r"\b(?:case for|cover for|screen protectors?|replacement (?:screen|battery|display)|"
    r"compatible with|for iphone|for samsung|fits? (?:iphone|samsung))\b",
    re.I,
)

# Stable EnvPack public fact labels / claim aliases (environment vocabulary).
_PUBLIC_FACT_LABEL = {
    "resolution": "display",
    "screen_size": "size",
    "refresh_rate": "refresh_rate",
    "capacity": "capacity",
    "wattage": "wattage",
    "warranty_duration": "warranty_duration",
    "pack_quantity": "pack_quantity",
}
_CLAIM_FIELD_ALIAS = {
    "display": "resolution",
    "panel": "resolution",
    "size": "screen_size",
    "screen": "screen_size",
    "screensize": "screen_size",
    "hz": "refresh_rate",
    "refresh": "refresh_rate",
    "availability": "in_stock",
    "stock": "in_stock",
    "current_price": "price",
}
_GATE_CLAIM_FIELDS = frozenset(_PUBLIC_FACT_LABEL)

# Family-stable phrases from ORO public Task rules (agent_system), not suite products.
_FAMILY_RULE_SIGNS = (
    ("justification", "trustworthy reason"),
    ("preference", "listing_id"),
    ("constraint", "exact variant"),
    ("intent", "nice-to-haves"),
)

_FAMILY_IDS = (
    "intent", "recall", "constraint", "preference", "ranking", "recovery", "justification", "generic",
)

_EXTRACT_SCHEMA = """\
Return ONLY a JSON object with these keys (use null/[]/{} when unknown):
{
  "family": one of intent|recall|constraint|preference|ranking|recovery|justification|generic,
  "category": string or null,
  "brand": string or null,
  "pref_brand": string or null,
  "budget": number or null,
  "currency": ISO 4217 code string or null,
  "theme_tokens": [string],
  "required_identities": integer (0 if none),
  "pref_model": string or null,
  "pref_color": string or null,
  "cat_qualifier": free-text product-shape adjective from the query or null,
  "require_device_identity": true if the shopper wants a real device not an accessory/part,
  "spec_sealed": true if a firm hardware requirement is hidden until the shopper is asked,
  "firm_specs": [{"concept": snake_case_name, "values": ["canonical", "aliases..."]}],
  "ranking_axes": ordered list of concepts to maximize/minimize, e.g. ["refresh_rate","screen_size","price"],
  "gate_concept": primary firm concept to cite/justify or null
}
Rules for firm_specs.values: expand natural synonyms yourself (e.g. if the shopper says 4K/UHD,
include equivalent pixel-pair forms and class-name synonyms). Do not invent requirements absent from the query.
For ranking_axes: extract the order the query states; if none stated, use [].
"""

_CORE_RULES = (
    "You are a careful shopping agent working a catalog through tools. Exactly one tool call per turn. "
    "A deterministic controller checks every proposal and takes over the cart and the order once a "
    "candidate is fully verified, so your job is to surface and read candidates: search or filter, then "
    "view or compare so their raw facts, variant options and prices are on record. Never add an item to "
    "the cart as a placeholder - the first cart addition is graded. Prices and stock change while you "
    "shop; the controller re-checks the carted item and tells you when to move to a different product. "
    "Answer every direct question from the shopper with one short message before anything else, and never "
    "send progress or status updates. Never ask the shopper to raise the budget. Do not repeat a tool "
    "call with identical arguments; one compare of up to five products beats several view calls."
)

_REASONING_POLICIES = """
Reasoning policies (apply on every task; these are linguistic patterns, not suite answers):

1) Quantity coalescing
When the query states an explicit piece/pack count (e.g. "2-piece", "3-pack", "5pcs") together with a
collective/quantity noun for the same product (e.g. "set", "pack", "kit", "bundle", "lot") — such as
"2-piece ... hair oil set" or "3-pack cable bundle" — these two words describe the SAME single quantity
fact, not two separate constraints. Drop the now-redundant collective noun from the rewritten sub-query
and keep only the explicit count, so the sub-query reads as one unambiguous quantity mention.
e.g. "2-piece organic hair oil set" -> "2-piece organic hair oil" (NOT "2-piece organic hair oil set");
"3-pack cable bundle" -> "3-pack cable". Only keep the collective noun when NO explicit piece/pack count
is stated (e.g. "hair oil set" with no number stays "hair oil set").

2) Multi-product boundaries in one utterance
- A completing spec or quantity in the middle (e.g. "5 seats", "1 meter", "3 pcs", "2-pack", or a
  color/size that naturally concludes a product description) often signals a boundary. After such a
  spec, a new product name may begin.
  Example: "car seat cushion cover 5 seats [END of product 1] blue patterned leatherette [product 2]"
- Two product names from different categories placed back to back: when two noun phrases each
  independently name a product (e.g. "golf bag" and "stand bag"), treat them as two separate products —
  even without a connector. Be conservative: only split when BOTH parts are standalone product names.
  Do NOT split "wireless bluetooth mechanical keyboard" — modifiers of one product noun.
- Different color/attribute clusters assigned to different names: "golf bag WHITE ... stand bag BLUE".

3) Evidence precedence
For any gated specification, prefer the selected variant's own option text, then labelled raw facts,
then the listing title. If those sources disagree, treat the value as unverified and keep looking.

4) Device vs accessory
When the shopper asked for a device, do not buy a case, protector, spare part, or "compatible with"
accessory that merely mentions the device. Prefer listings whose facts describe the device itself.

5) Display / panel naming
Map stated pixel pairs and panel names to the same display class the listing asserts. Do not invent a
second independent constraint from synonyms of one class. Quarter-HD style names are not full HD.

6) Ranking
Extract the ordered ranking criteria from the query (and only those). Do not assume a fixed attribute
order unless the query states it. After a market change, re-rank the remaining live candidates.

7) Recovery
Relax as little as possible: same model in another color before any other model. Never relax category,
budget, or stock.

8) Justification
Cite only facts you have viewed or compared for the ordered item, plus live price and stock. Do not
invent claims from titles alone when raw facts are available.

9) Shelf names
Treat retailer shelf synonyms as the same product class when they clearly name the same kind of goods
the shopper asked for; do not invent cross-category equivalences.
"""

_PLAYBOOK = {
    "intent": (
        "Firm requirements: category, named hardware specification, budget and stock. A named brand is "
        "only a preference when a fully compliant item of that brand exists. Filter then view/compare. "
        "Answer the shopper; a later lower budget applies to every later choice."
    ),
    "recall": (
        "Run several differently worded searches (k=10, in_stock=true) until enough distinct matching "
        "products have appeared; requested theme words must appear as whole words of the title. Then "
        "view the strongest candidates so the controller can verify one and buy it."
    ),
    "constraint": (
        "The shopper may hide one firm hardware requirement and answer it when asked; obey it exactly for "
        "the specific variant, reading the variant's own option text before any product-wide value. If the "
        "carted variant sells out, move to a different product that still meets every requirement."
    ),
    "preference": (
        "Ask which preference matters most if unclear. Search or filter within budget, compare candidates "
        "with different listing_id values, reading raw facts. The first cart item is graded. If it is "
        "repriced above budget, recover to a different listing_id."
    ),
    "ranking": (
        "Only the named candidate variants count. Compare them so their facts are on record. Rank by the "
        "axes stated in the query (controller sorts). After the market moves, re-rank remaining candidates."
    ),
    "recovery": (
        "Find the preferred model and color first; view so every variant/color is on record. First cart "
        "item must be that exact pairing. If it disappears, relax minimally."
    ),
    "justification": (
        "Find a product whose raw facts state the requested specification, in category/brand, in stock and "
        "within budget. View or compare it; the controller orders with claims taken only from that evidence."
    ),
    "generic": (
        "Understand the firm requirements, filter or search, view candidates, and let the controller "
        "verify and buy the best one."
    ),
}


def _inch_vals(text: str) -> set:
    return {"%gin" % float(m.group(1)) for m in _INCH_RE.finditer(text or "")}


def _hz_vals(text: str) -> set:
    return {"%ghz" % float(m.group(1)) for m in _HZ_RE.finditer(text or "")}


def _gb_vals(text: str) -> set:
    return {
        "%ggb" % (float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1))
        for m in _GB_RE.finditer(text or "")
    }


def _pixel_class(long_side: int, short_side: int) -> str | None:
    """Display-class vocabulary used by the EnvPack attribute contract."""
    if long_side >= 3840 and short_side >= 2160:
        return "4k"
    if long_side >= 2560 and short_side >= 1440:
        return "qhd"
    if long_side >= 1920 and short_side >= 1080:
        return "fhd"
    if long_side >= 1280 and short_side >= 720:
        return "hd"
    return None


def _res_class_vals(text: str) -> set:
    """Canonical display classes asserted by text (4k/qhd/fhd/hd)."""
    text = text or ""
    if _QUARTER_RE.search(text):
        return set()
    low = text.lower()
    out = set()
    for m in _WXH_RE.finditer(low):
        pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        cls = _pixel_class(pair[0], pair[1])
        if cls:
            out.add(cls)
    if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
        out.add("4k")
    if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
        out.add("qhd")
    if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
        out.add("fhd")
    if re.search(r"\b(?:hd|720p?)\b", re.sub(r"\b(?:full|ultra) hd\b", "", low)):
        out.add("hd")
    return out


def _capacity_vals(text: str) -> set:
    out = set()
    for m in _CAPACITY_RE.finditer(text or ""):
        n = float(m.group(1))
        unit = m.group(2).lower()
        ml = n * 1000.0 if unit.startswith("l") and not unit.startswith("ml") else n
        out.add(f"{ml:g}ml")
    return out


def _wattage_vals(text: str) -> set:
    out = set()
    for m in _WATTAGE_RE.finditer(text or ""):
        n = float(m.group(1))
        unit = m.group(2).lower()
        watts = n * 1000.0 if unit.startswith("k") else n
        out.add("%gw" % watts)
    return out


def _warranty_vals(text: str) -> set:
    out = set()
    for m in _WARRANTY_RE.finditer(text or ""):
        n = float(m.group(1))
        unit = m.group(2).lower()
        months = n * 12.0 if unit.startswith("y") else n
        out.add("%gmo" % months)
    return out


def _pack_vals(text: str) -> set:
    return {"%dct" % int(n) for n in _PACK_RE.findall(text or "")}


def _canonicalize_claim_field(field: str | None) -> str:
    key = (field or "").strip().lower().replace(" ", "_")
    return _CLAIM_FIELD_ALIAS.get(key, key)


def _adjacent_vals(concept: str, text: str) -> set:
    text = text or ""
    key = _canonicalize_claim_field(concept)
    if key in ("ram", "memory", "storage_ram"):
        paired = {"%ggb" % float(a) for a, _b in _PAIR_RE.findall(text)}
        return paired or {"%ggb" % float(n) for n in _MEM_RE.findall(text)}
    if key in ("screen_size",):
        return _inch_vals(text)
    if key in ("refresh_rate",):
        return _hz_vals(text)
    if key in ("resolution",):
        return _res_class_vals(text)
    if key == "capacity":
        return _capacity_vals(text)
    if key == "wattage":
        return _wattage_vals(text)
    if key == "warranty_duration":
        return _warranty_vals(text)
    if key == "pack_quantity":
        return _pack_vals(text)
    if key in ("storage",) and _GB_RE.search(text):
        return _gb_vals(text)
    return set()


def _mentions(concept: str, text: str) -> bool:
    text = text or ""
    key = _canonicalize_claim_field(concept)
    if key in ("ram", "memory", "storage_ram"):
        return bool(_PAIR_HINT_RE.search(text) or _MEM_RE.search(text))
    if key == "screen_size":
        return bool(_INCH_RE.search(text))
    if key == "refresh_rate":
        return bool(_HZ_RE.search(text))
    if key == "resolution":
        return bool(_res_class_vals(text) or _WXH_RE.search(text))
    if key == "capacity":
        return bool(_CAPACITY_RE.search(text))
    if key == "wattage":
        return bool(_WATTAGE_RE.search(text))
    if key == "warranty_duration":
        return bool(_WARRANTY_RE.search(text))
    if key == "pack_quantity":
        return bool(_PACK_RE.search(text))
    want = _tokens(concept)
    return bool(want and want <= _tokens(text))


def _one(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _same_word(a: str, b: str) -> bool:
    x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
    y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
    return bool(x) and x == y


def _parse_budget(text: str) -> float | None:
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


def _parse_currency(text: str) -> str:
    m = _CURRENCY_RE.search(text or "")
    return m.group(1) if m else ""


def _json_object(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def _extract_brief_payload(query: str, model: str) -> dict:
    """One-shot LLM parse of the public query into a structured Brief payload."""
    messages = [
        {
            "role": "system",
            "content": (
                "You extract shopping-task structure from a public shopper query. "
                + _EXTRACT_SCHEMA
                + "\nRespond with JSON only."
            ),
        },
        {"role": "user", "content": query or ""},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    resp = _PROXY.post("/inference/chat/completions", json_data=payload)
    if not isinstance(resp, dict) or not resp.get("choices"):
        return {}
    msg = resp["choices"][0].get("message") or {}
    return _json_object(str(msg.get("content") or ""))


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


class Brief:
    """What the deterministic side needs — LLM-extracted, with structural fallbacks."""

    def __init__(self, query: str, extracted: dict | None = None) -> None:
        self.raw = query or ""
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules
        extracted = extracted if isinstance(extracted, dict) else {}

        self.budget = extracted.get("budget")
        if self.budget is not None:
            try:
                self.budget = float(self.budget)
            except (TypeError, ValueError):
                self.budget = None
        if self.budget is None:
            self.budget = _parse_budget(self.goal)

        cur = extracted.get("currency")
        self.currency = str(cur).upper() if cur else _parse_currency(self.goal)

        self.candidate_keys = [(pid, sku.rstrip(".,;:")) for pid, sku in _KEY_RE.findall(self.goal)]
        theme = extracted.get("theme_tokens") or []
        self.theme = [str(t).lower() for t in theme if str(t).strip()] if isinstance(theme, list) else []

        try:
            self.required_identities = int(extracted.get("required_identities") or 0)
        except (TypeError, ValueError):
            self.required_identities = 0

        self.pref_color = (str(extracted["pref_color"]).lower() if extracted.get("pref_color") else None)
        self.pref_model = str(extracted["pref_model"]) if extracted.get("pref_model") else None
        self.category = str(extracted["category"]).strip() if extracted.get("category") else None
        self.brand = str(extracted["brand"]).strip() if extracted.get("brand") else None
        self.pref_brand = str(extracted["pref_brand"]).strip() if extracted.get("pref_brand") else None
        self.cat_qualifier = str(extracted["cat_qualifier"]).strip().lower() if extracted.get("cat_qualifier") else None
        self.require_device_identity = bool(extracted.get("require_device_identity"))
        self.spec_sealed = bool(extracted.get("spec_sealed"))

        self.specs: dict[str, set[str]] = {}
        for item in extracted.get("firm_specs") or []:
            if not isinstance(item, dict):
                continue
            concept = _canonicalize_claim_field(str(item.get("concept") or ""))
            values = item.get("values") or []
            if not concept or not isinstance(values, list):
                continue
            cleaned = set()
            for v in values:
                if not str(v).strip():
                    continue
                cleaned |= _adjacent_vals(concept, str(v)) or {_norm_value(concept, str(v))}
            cleaned.discard("")
            if cleaned:
                self.specs[concept] = cleaned

        # Fallback: pull unambiguous unit values from the goal if the model missed them.
        if not self.specs:
            for concept, parser in (
                ("screen_size", _inch_vals),
                ("refresh_rate", _hz_vals),
                ("ram", _gb_vals),
                ("resolution", _res_class_vals),
                ("capacity", _capacity_vals),
                ("wattage", _wattage_vals),
                ("warranty_duration", _warranty_vals),
                ("pack_quantity", _pack_vals),
            ):
                found = parser(self.goal)
                if len(found) == 1:
                    self.specs[concept] = found

        gate = extracted.get("gate_concept")
        if gate:
            gate = _canonicalize_claim_field(str(gate))
        # Prefer a gate-claimable concept for justification scoring.
        if gate in self.specs:
            self.spec_field = gate
        else:
            self.spec_field = next((k for k in self.specs if k in _GATE_CLAIM_FIELDS), None)
            if self.spec_field is None:
                self.spec_field = next(iter(self.specs), None)
        self.spec_value = next(iter(self.specs[self.spec_field]), None) if self.spec_field else None

        axes = extracted.get("ranking_axes") or []
        self.ranking_axes = [
            _canonicalize_claim_field(str(a)) for a in axes if str(a).strip()
        ] if isinstance(axes, list) else []

        fam = str(extracted.get("family") or "").strip().lower()
        # Map runtime family ids if present.
        if fam in ("preference_reasoning",):
            fam = "preference"
        if fam in ("constraint_satisfaction",):
            fam = "constraint"
        if fam in ("intent_decomposition",):
            fam = "intent"
        if fam in ("retrieval_recall",):
            fam = "recall"
        # Public Task rules are the family contract; they beat a mis-extracted label.
        for name, sign in _FAMILY_RULE_SIGNS:
            if sign in (self.rules or "").lower():
                fam = name
                break
        self.family = fam if fam in _FAMILY_IDS else self._family_fallback()

    def _family_fallback(self) -> str:
        rules = (self.rules or "").lower()
        for name, sign in _FAMILY_RULE_SIGNS:
            if sign in rules:
                return name
        if self.candidate_keys:
            return "ranking"
        if self.required_identities:
            return "recall"
        if self.pref_model:
            return "recovery"
        if self.theme:
            return "recall"
        if re.search(r"\b(?:why|reason|explain|justif)", self.goal, re.I):
            return "justification"
        if self.spec_sealed or self.spec_field or self.brand:
            return "constraint"
        if self.pref_brand:
            return "intent"
        return "generic"

    def adopt_spec(self, text: str) -> str | None:
        """Record a requirement the shopper reveals later; returns its description."""
        for concept, parser in (
            ("screen_size", _inch_vals),
            ("refresh_rate", _hz_vals),
            ("ram", _gb_vals),
            ("resolution", _res_class_vals),
            ("capacity", _capacity_vals),
            ("wattage", _wattage_vals),
            ("warranty_duration", _warranty_vals),
            ("pack_quantity", _pack_vals),
        ):
            found = parser(text)
            if len(found) == 1:
                value = next(iter(found))
                self.specs[concept] = found
                self.spec_field, self.spec_value = concept, value
                self.spec_sealed = False
                return concept + " = " + value
        return None


def _norm_value(concept: str, value: str) -> str:
    key = _canonicalize_claim_field(concept)
    text = (value or "").strip()
    found = _adjacent_vals(key, text)
    if len(found) == 1:
        return next(iter(found))
    if key == "screen_size" and re.fullmatch(r"\d+(?:\.\d+)?", text):
        return "%gin" % float(text)
    if key == "refresh_rate" and re.fullmatch(r"\d+(?:\.\d+)?", text):
        return "%ghz" % float(text)
    if key in ("ram", "memory", "storage_ram") and re.fullmatch(r"\d+(?:\.\d+)?", text):
        return "%ggb" % float(text)
    if key == "resolution":
        classes = _res_class_vals(text)
        return next(iter(classes)) if len(classes) == 1 else text.lower()
    return text.lower()


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
        self.checked: dict = {}
        self.checked_seq: dict = {}
        self.inspected: set = set()
        self.products: dict = {}
        self.rows: dict = {}
        self.options: dict = {}
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

        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)
        self.model_index = 0

        query = str(self.policy.get("query") or "")
        extracted = _extract_brief_payload(query, self.roster[0])
        # Public family hint from problem_data when present (never suite phrase matching).
        for key in ("family", "task_family", "family_id"):
            if problem.get(key) and not extracted.get("family"):
                extracted["family"] = problem.get(key)
        self.brief = Brief(query, extracted)
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
        self.event_turn: int | None = None
        self.unreadable: set = set()
        self.question_waits = 0
        self.done = False

    def _system_prompt(self) -> str:
        play = _PLAYBOOK.get(self.brief.family, _PLAYBOOK["generic"])
        extra = ""
        if self.brief.family == "recall" and self.brief.required_identities:
            extra = (
                " You must surface at least %d distinct matching products in search or filter "
                "results before ordering; the controller reports the count after each search."
                % self.brief.required_identities
            )
        steps = " At most %d turns, so be efficient." % self.max_steps
        return _CORE_RULES + steps + "\n\n" + _REASONING_POLICIES + "\n\nFamily playbook: " + play + extra

    def _opening_question(self) -> str | None:
        if self.brief.family == "constraint" or self.brief.spec_sealed:
            return (
                "Before I search: which hardware specification is your firm requirement, "
                "and what exact value do you need?"
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

    def _fact_blob(self, pid: str) -> str:
        return " ".join("%s %s" % (k, v) for k, v in self._facts(pid).items())

    def _concept_from_facts(self, pid: str, concept: str) -> set:
        """Values for a concept from labelled product facts."""
        facts = self._facts(pid)
        key = _canonicalize_claim_field(concept)
        out: set = set()
        texts: list[str] = []
        label = _PUBLIC_FACT_LABEL.get(key)
        if label and facts.get(label) is not None:
            texts.append(str(facts.get(label)))
        # Also accept keys whose name overlaps the concept.
        for fk, fv in facts.items():
            ftoks = _tokens(str(fk).replace("_", " "))
            ctoks = _tokens(key.replace("_", " "))
            if label and str(fk) == label:
                continue
            if ctoks & ftoks or key in str(fk).lower():
                texts.append(str(fv))
        if not texts:
            texts = [str(v) for v in facts.values()]
        for text in texts:
            found = _adjacent_vals(key, text)
            if found:
                out |= found
            elif re.fullmatch(r"\d+(?:\.\d+)?", text.strip()):
                unit_map = {
                    "screen_size": "in",
                    "refresh_rate": "hz",
                    "ram": "gb",
                    "capacity": "ml",
                    "wattage": "w",
                    "warranty_duration": "mo",
                    "pack_quantity": "ct",
                }
                if key in unit_map:
                    out.add("%g%s" % (float(text.strip()), unit_map[key]))
                elif key == "resolution":
                    out |= _res_class_vals(text)
                else:
                    out.add(text.strip().lower())
            else:
                normed = _norm_value(key, text)
                if normed:
                    out.add(normed)
        return {v for v in out if v}


    def _product_values(self, pid: str, concept: str) -> set:
        vals = set(self._concept_from_facts(pid, concept))
        vals |= _adjacent_vals(concept, self._title(pid))
        # Free-text concepts: also allow token presence of acceptable wants later.
        return vals

    def _variant_values(self, pid: str, sku: str, concept: str) -> set:
        opt = self._options(pid, sku)
        vals = _adjacent_vals(concept, opt)
        if len(vals) > 1:
            return set()
        if vals:
            return vals
        if _mentions(concept, opt):
            return set()
        return self._product_values(pid, concept)

    def _spec_match(self, pid: str, sku: str, concept: str, wants: set) -> int:
        """2 exact, 0 unknown, -1 conflict."""
        if not concept or not wants:
            return 2
        got = self._variant_values(pid, sku, concept)
        if not got:
            # Free-text fallback: any want token appears in options/facts/title.
            blob = (
                self._options(pid, sku) + " " + self._fact_blob(pid) + " " + self._title(pid)
            ).lower()
            if any(_norm_value(concept, w) in blob or w.lower() in blob for w in wants):
                return 2
            return 0
        norms = {_norm_value(concept, w) for w in wants}
        if got & norms:
            return 2
        return -1

    def _variant_value(self, pid: str, sku: str, concept: str) -> str | None:
        return _one(self._variant_values(pid, sku, concept))

    def _fact_value(self, pid: str, concept: str) -> str | None:
        return _one(self._concept_from_facts(pid, concept))

    def _product_value(self, pid: str, concept: str) -> str | None:
        return _one(self._product_values(pid, concept))

    def _measure(self, pid: str, sku: str, concept: str) -> float | None:
        value = self._variant_value(pid, sku, concept)
        if not value:
            return None
        m = re.match(r"^(\d+(?:\.\d+)?)", value)
        try:
            return float(m.group(1)) if m else None
        except ValueError:
            return None

    @staticmethod
    def _stems(text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

    def _category_ok(self, pid: str) -> bool:
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
            # Soft shelf synonymy: shared content word of length > 3.
            placed = bool({w for w in wanted if len(w) > 3} & {w for w in labels if len(w) > 3})
        if not placed:
            return False
        qualifier = self.brief.cat_qualifier
        if not qualifier:
            return True
        title = self._title(pid)
        qtoks = _tokens(qualifier)
        return bool(qtoks) and qtoks <= (_tokens(title) | self._stems(title))

    def _brand_ok(self, pid: str) -> bool:
        want = self.brief.brand
        if not want:
            return True
        wanted = _tokens(want)
        return bool(wanted <= _tokens(self._brand_of(pid)) or wanted <= _tokens(self._title(pid)))

    def _device_ok(self, pid: str, sku: str) -> bool:
        if not self.brief.require_device_identity:
            return True
        title = self._title(pid)
        options = self._options(pid, sku)
        if _ACCESSORY_TITLE_RE.search(title) or _ACCESSORY_TITLE_RE.search(options):
            return False
        return True

    def _resale_title(self, pid: str) -> bool:
        return bool(
            re.search(
                r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed",
                self._title(pid),
                re.I,
            )
        )

    def _relax(self, pid: str, sku: str) -> int:
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
        fam = self.brief.family
        key = Ledger.key(pid, sku)
        st = self.ledger.checked.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.ledger.dead_keys or (str(pid) in self.ledger.dead_products and fam != "recovery")
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and not dead
        viewed = str(pid) in self.ledger.products
        field, wants = self.brief.spec_field, self.brief.specs.get(self.brief.spec_field or "", set())
        want = self.brief.spec_value
        spec = self._spec_match(pid, sku, field, wants) if field and wants else 2
        ok = bool(live and viewed and spec == 2 and self._brand_ok(pid) and self._device_ok(pid, sku))
        cat_ok = self._category_ok(pid)
        ok = ok and cat_ok
        if fam == "justification":
            ok = ok and not self._resale_title(pid) and bool(self._brand_of(pid))
            gate = _canonicalize_claim_field(field)
            if gate in _GATE_CLAIM_FIELDS:
                claimed = self._gate_claim_value(pid, gate, wants)
                ok = ok and bool(claimed)
        if fam == "preference" and self.ledger.event_target:
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split("::", 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if fam == "recall" and self.brief.theme:
            ok = ok and all(t in _tokens(self._title(pid)) for t in self.brief.theme)
        if fam == "ranking":
            ok = ok and (str(pid), str(sku)) in self.brief.candidate_keys
            for axis in self.brief.ranking_axes:
                if axis in ("price",):
                    continue
                if self._measure(pid, sku, axis) is None and not self._variant_values(pid, sku, axis):
                    ok = False
                    break
        return {
            "pid": str(pid), "sku": str(sku), "key": key, "price": price, "live": live,
            "viewed": viewed, "spec": spec, "cat_ok": cat_ok, "ok": ok,
            "relax": self._relax(pid, sku) if fam == "recovery" else 0,
        }

    def _axis_key(self, c: dict, axis: str) -> float:
        if axis == "price":
            return c["price"] if c["price"] is not None else 1e12
        measured = self._measure(c["pid"], c["sku"], axis)
        if measured is not None:
            # Higher is better for non-price axes unless name suggests otherwise.
            if any(x in axis for x in ("price", "cost", "weight", "latency")):
                return measured
            return -measured
        return 0.0

    def _rank_of(self, c: dict, first: bool) -> tuple:
        fam = self.brief.family
        price = c["price"] if c["price"] is not None else 1e12
        if fam == "ranking":
            parts = []
            axes = self.brief.ranking_axes or ["price"]
            for axis in axes:
                parts.append(self._axis_key(c, axis))
            parts.append(c["key"])
            return tuple(parts)
        if fam == "recovery":
            return (c["relax"], price, c["key"])
        extra = 0
        for concept, wants in self.brief.specs.items():
            if concept == self.brief.spec_field:
                continue
            if self._spec_match(c["pid"], c["sku"], concept, wants) != 2:
                extra += 1
        if fam == "intent" and first and self.brief.pref_brand:
            wanted = _tokens(self.brief.pref_brand)
            near = 0 if (wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))) else 1
            return (near, extra, price, c["key"])
        return (extra, price, c["key"])

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
        fam = self.brief.family
        good = [c for c in self._candidates() if c["ok"]]
        if fam == "preference":
            carted = [c for c in good if (c["pid"], c["sku"]) in self.ledger.cart]
            if carted:
                return carted[0]
            if good and self.turn >= self.max_steps - 8:
                good.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]),
                                         c["price"] if c["price"] is not None else 1e12))
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
        target = self._target() or (self._candidates() or [None])[0]
        if not target:
            return "I am still checking the options against your requirements; nothing is decided yet."
        pid, sku = target["pid"], target["sku"]
        parts = ["I am going with " + (self._title(pid) or pid)[:90]]
        field, wants = self.brief.spec_field, self.brief.specs.get(self.brief.spec_field or "", set())
        if field and wants and self._spec_match(pid, sku, field, wants) == 2:
            parts.append("its listed " + field.replace("_", " ") + " matches " + (self.brief.spec_value or next(iter(wants))))
        if target.get("price") is not None:
            cur = self.brief.currency or ""
            parts.append("it is %.2f%s and in stock" % (target["price"], (" " + cur) if cur else ""))
        return ", ".join(parts) + "."

    def _read_more(self) -> tuple | None:
        fam = self.brief.family
        if fam == "recall" and len(self.ledger.identities) < self._recall_need():
            return None
        twin = self._twin_search()
        if twin is not None:
            return twin
        leads = self._unseen_leads()
        if leads and self.forced_compares < 8 and self.turn >= 2:
            self.forced_compares += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        probe = (self.brief.brand or "", "%.2f" % (self.budget or 0.0), str(self.brief.spec_value))
        if self.turn < 2:
            return None
        if probe in self.probes:
            self.reads_done = True
            return None
        self.probes.add(probe)
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
                return ("search", {"query": " ".join(words), "in_stock": True, "k": 10})
        self.reads_done = True
        return None

    def _twin_search(self) -> tuple | None:
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

    def _shortlist_note(self) -> None:
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
        field, want = self.brief.spec_field, self.brief.spec_value
        blocked = [c for c in cands if c["viewed"] and c["live"] and not c["ok"]]
        if blocked and field:
            sample = blocked[:3]
            self._note(
                "None of the products read so far can be verified. For each, the evidence for %s must "
                "resolve to exactly %s from the variant's own option text or the product's raw facts: %s. "
                "Look at different products." % (field, want, ", ".join(c["pid"] + "::" + c["sku"] for c in sample))
            )
        elif blocked:
            self._note(
                "The products read so far fail a firm requirement (category, brand, stock, budget or the "
                "kind of device requested). Search for different products."
            )

    def _priority_hits(self, pid: str, sku: str) -> int:
        wants = self._wish_words()
        if not wants:
            return 0
        have = _tokens(" ".join(str(v) for v in self._facts(pid).values()) + " " + self._title(pid) + " " + self._options(pid, sku))
        return len(wants & have)

    def _wish_words(self) -> set:
        if self._wishes is None:
            said = self.brief.goal + " " + " ".join(
                str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
            )
            drop = _tokens(self.brief.category or "") | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "test", "buy", "something", "its",
                "would", "like", "have", "from", "into", "about", "please", "thanks",
            }
            self._wishes = {w for w in _tokens(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._wishes

    def _budget_from_message(self, text: str) -> None:
        new = _parse_budget(text)
        if new is None or (self.budget is not None and new >= self.budget):
            return
        self.budget = new
        self.shopper_updated = True
        cur = self.brief.currency or ""
        self._note(
            "Budget updated by the shopper: at most %.2f%s. Every later choice must fit it."
            % (new, (" " + cur) if cur else "")
        )
        for pid, sku in list(self.ledger.cart):
            st = self.ledger.status(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _EPS:
                self.ledger.dead_keys.add(Ledger.key(pid, sku))
                self._note(
                    "The carted item %s costs %.2f, above the new budget; the controller will replace it."
                    % (Ledger.key(pid, sku), price)
                )

    def _gate_claim_value(self, pid: str, field: str, wants: set[str]) -> str | None:
        """Canonical gate value that visible facts support and the shopper asked for."""
        field = _canonicalize_claim_field(field)
        fact_vals = self._concept_from_facts(pid, field)
        if not fact_vals:
            return None
        want_norms = {_norm_value(field, w) for w in wants} if wants else set()
        want_norms |= set(wants or ())
        overlap = fact_vals & want_norms
        if overlap:
            return sorted(overlap)[0]
        # Fact uniquely states one value and no shopper want was parsed — still claim the fact.
        if len(fact_vals) == 1 and not wants:
            return next(iter(fact_vals))
        return None

    def _claims_for(self, pid: str, sku: str) -> list:
        """TF7 requires gate_attr + price + in_stock, plus a soft reason (brand)."""
        out: list[dict] = []
        field = _canonicalize_claim_field(self.brief.spec_field)
        wants = set(self.brief.specs.get(self.brief.spec_field or "", set()))
        if field not in _GATE_CLAIM_FIELDS:
            # Try any gate-capable concept we already verified on this product.
            for candidate in list(self.brief.specs) + list(_GATE_CLAIM_FIELDS):
                cand = _canonicalize_claim_field(candidate)
                if cand not in _GATE_CLAIM_FIELDS:
                    continue
                cand_wants = set(self.brief.specs.get(candidate, set())) or set(self.brief.specs.get(cand, set()))
                value = self._gate_claim_value(pid, cand, cand_wants)
                if value:
                    field, wants = cand, cand_wants or {value}
                    break
            else:
                field = None

        if field:
            value = self._gate_claim_value(pid, field, wants)
            if value:
                out.append({"field": str(field), "value": str(value)})

        if not out:
            return []  # never submit a justification missing the gate claim

        st = self.ledger.status(pid, sku) or {}
        # Inclusive budget threshold is accepted by the verifier; safer than exact float match.
        if self.budget is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.budget})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        else:
            return []

        out.append({"field": "in_stock", "value": "true"})

        brand = self._brand_of(pid)
        if not brand:
            return []  # soft claim required for faithfulness
        out.append({"field": "brand", "value": str(brand)})

        # Ensure every claim value is a non-empty string (runtime rejects otherwise).
        cleaned = []
        for claim in out:
            field_s = str(claim.get("field") or "").strip()
            value_s = str(claim.get("value") or "").strip()
            if field_s and value_s:
                cleaned.append({"field": field_s, "value": value_s})
        return cleaned

    def _order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.brief.family == "justification":
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

    def _recall_need(self) -> int:
        early = self.search_calls < 8 and self.turn < self.max_steps * 0.5
        return self.brief.required_identities + (2 if early else 0)

    def _ready(self) -> bool:
        fam = self.brief.family
        if self.ledger.adds:
            return True
        if self.brief.spec_sealed:
            if self.turn < self.max_steps * 0.6:
                return False
            self.brief.spec_sealed = False
            self._note("The firm requirement was never stated, so the remaining requirements decide the choice.")
        if fam in ("intent", "constraint") and self.turn < 4 and not self.shopper_updated:
            return False
        if fam == "preference" and len(self.ledger.listing_ids) < 2:
            return False
        roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
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
        if self.done:
            return None
        if self.shopper_question_pending:
            if self.question_waits < 2 or "message" not in self.tool_names:
                self.question_waits += 1
                return None
            return ("message", {"content": self._answer_text()})
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
            if self.brief.family == "justification" and not self._claims_for(pid, sku):
                return None
            return ("place_test_order", self._order_args(pid, sku))
        if not self._ready():
            return None
        for cpid, csku in self.ledger.cart:
            if Ledger.key(cpid, csku) in self.ledger.dead_keys and "remove_from_cart" in self.tool_names:
                return ("remove_from_cart", {"product_id": cpid, "sku": csku})
        if not self._fresh(pid, sku):
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return ("add_to_cart", {"product_id": pid, "sku": sku})

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
            if self.clock.remaining() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
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
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _trim(json.dumps(obs), _OBS_CHARS)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self._budget_from_message(text)
            if self.brief.family in ("constraint", "intent") and not self.brief.spec_value:
                stated = self.brief.adopt_spec(text)
                if stated:
                    self.shopper_updated = True
                    self._note("Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought.")
            if _QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self._note("The shopper asked a question: answer it with message in your next turn, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        return obs

    def _post_add_recheck(self) -> None:
        pid, sku = self.pending_recheck
        self.pending_recheck = None
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = _num(obs.get("price"))
        over = self.budget is not None and price is not None and price > self.budget + _EPS
        if stock and not over:
            self._note(
                "Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it."
                % (Ledger.key(pid, sku), price)
            )
            return
        self.ledger.event_target = Ledger.key(pid, sku)
        self.event_turn = self.turn
        self.ledger.dead_keys.add(Ledger.key(pid, sku))
        self.ledger.dead_products.add(str(pid))
        self.forced_compares = 0
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        guidance = {
            "recovery": "Recover with the smallest possible change: the same model in another color before any other model.",
            "ranking": "The remaining named candidates are re-ranked by the query's stated axes; the controller takes the new best.",
            "preference": "Recover to a product with a different listing_id that still fits the shopper's priorities.",
            "constraint": "Choose a different product that still meets every requirement, not another variant of the same one.",
        }.get(self.brief.family, "Choose a different product that meets every firm requirement.")
        self._note("Market change: the carted item %s %s. " % (Ledger.key(pid, sku), reason) + guidance)
        self._shortlist_note()

    def _finish_fast(self) -> None:
        for pid, sku in reversed(self.ledger.cart):
            ok, _why = self._valid_now(pid, sku)
            if ok and not self.done:
                if self.brief.family == "justification" and not self._claims_for(pid, sku):
                    continue
                self._act("place_test_order", self._order_args(pid, sku))
                return
        if self.done or self.turn > self.max_steps - 3:
            return
        good = [c for c in self._candidates() if c["ok"]]
        if not good and self.brief.family != "justification":
            good = [
                c for c in self._candidates()
                if c["live"] and c["spec"] == 2 and self._brand_ok(c["pid"]) and c["cat_ok"]
                and (not self.brief.candidate_keys or (c["pid"], c["sku"]) in self.brief.candidate_keys)
            ]
        if not good:
            return
        good.sort(key=lambda c: self._rank_of(c, self.ledger.adds == 0))
        pid, sku = good[0]["pid"], good[0]["sku"]
        if (pid, sku) not in self.ledger.cart:
            obs = self._act("add_to_cart", {"product_id": pid, "sku": sku})
            if self.done or not isinstance(obs, dict) or obs.get("error"):
                return
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if self.done or not isinstance(obs, dict) or not obs.get("in_stock"):
            return
        price = _num(obs.get("price"))
        if self.budget is not None and price is not None and price > self.budget + _EPS:
            return
        if self.brief.family == "justification" and not self._claims_for(pid, sku):
            return
        self._act("place_test_order", self._order_args(pid, sku))

    def run(self) -> list:
        question = self._opening_question()
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            if self.clock.remaining() < 25 or self.turn >= self.max_steps - 3:
                self._finish_fast()
                break
            if question and not self.asked_shopper and "message" in self.tool_names:
                self.asked_shopper = True
                self._act("message", {"content": question})
                continue
            if self.pending_recheck and "inspect_stock" in self.tool_names:
                self._post_add_recheck()
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
                self.search_calls += 1
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
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.ledger.cart:
                pid, sku = self.ledger.cart[-1]
            ok, why = self._valid_now(pid, sku)
            if not ok:
                return "order refused: " + why
            if self.brief.family == "recall" and len(self.ledger.identities) < self._recall_need():
                return (
                    "order refused: only %d distinct matching products have been surfaced; search with different "
                    "wording until at least %d appear" % (len(self.ledger.identities), self._recall_need())
                )
            if self.brief.family == "preference" and len(self.ledger.listing_ids) < 2:
                return "order refused: compare at least two products with different listing_id values first"
            if self.brief.family == "justification" and not self._claims_for(pid, sku):
                return (
                    "order refused: cannot attach a fact-backed justification yet "
                    "(need the gate specification from raw facts, current price/stock, and brand)"
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
                return "view or compare that product first so its raw facts are on record"
            if key not in self.ledger.inspected:
                return "inspect_stock that exact variant first: its current price and stock decide whether it can be bought"
            if c["spec"] != 2 and self.brief.spec_field:
                return (
                    "that variant does not resolve to %s = %s from its own option text or raw facts"
                    % (self.brief.spec_field, self.brief.spec_value)
                )
            if not c["ok"]:
                return "that product fails a firm requirement (category, brand, or the kind of device requested)"
            if self.brief.family == "preference" and len(self.ledger.listing_ids) < 2:
                return "compare two products with different listing_id values before the first cart addition"
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


__all__ = ["agent_main"]
