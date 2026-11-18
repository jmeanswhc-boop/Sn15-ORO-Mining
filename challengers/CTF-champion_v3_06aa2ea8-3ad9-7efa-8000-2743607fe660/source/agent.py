"""ORO Bench agent: official environment loop plus a guarded shopping policy."""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_PROVIDER_MODELS = {
    "chutes": {
        "primary": "deepseek-ai/DeepSeek-V3.2-TEE",
        "fallback": "deepseek-ai/DeepSeek-V3.1-TEE",
    },
    "openrouter": {
        "primary": "deepseek/deepseek-v3.2",
        "fallback": "deepseek/deepseek-chat-v3.1",
    },
}

_CLAIM_FIELDS = (
    "brand",
    "category",
    "in_stock",
    "price",
    "refresh_rate",
    "resolution",
    "screen_size",
)
_COLOR_WORDS = {
    "black",
    "blue",
    "bronze",
    "brown",
    "gold",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}
_DEVICE_WORDS = (
    "smartphone",
    "tablet",
    "monitor",
    "headphone",
    "headphones",
    "earphone",
    "earbuds",
)
_ACCESSORY_HINTS = (
    "accessory",
    "cable",
    "case",
    "charger",
    "cover",
    "earmuff",
    "hanger",
    "holder",
    "hook",
    "mount",
    "organizer",
    "pouch",
    "protector",
    "splitter",
    "stand",
    "tempered",
)
_HARD_ACCESSORY = (
    "earmuff",
    "hanger",
    "hook",
    "mount",
    "organizer",
    "splitter",
    "stand",
)
_WATER_HINTS = (
    "ipx",
    "ip67",
    "ip68",
    "waterproof",
    "water-resistant",
    "water resistant",
    "sweat",
    "splash",
)
_HEADPHONE_ACCESSORY = (
    "case",
    "cover",
    "protector",
    "sleeve",
    "pouch",
    "holder",
    "charger cover",
)

_proxy = ProxyClient(timeout=120, max_retries=1)
_INFERENCE_OK: bool | None = None


def _inference_ok() -> bool:
    # Agent-initiated message() invokes the shopper simulator. A 401 there
    # quarantines the session, so keep messages off unless inference recovers.
    return False


def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()


def _model_pair() -> tuple[str, str]:
    mapping = _PROVIDER_MODELS.get(_provider(), _PROVIDER_MODELS["chutes"])
    override = getenv("SANDBOX_MODEL")
    primary = override or mapping["primary"]
    fallback = mapping["fallback"]
    if fallback == primary:
        fallback = mapping["primary"]
    return primary, fallback


def _as_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    raw = fn.get("arguments", tool_call.get("arguments", tool_call.get("args", {})))
    parsed = _as_dict(raw)
    return parsed


def _tool_name(tool_call: dict[str, Any]) -> str:
    fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    return str(fn.get("name") or tool_call.get("name") or "")


def _policy_tools(policy: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for item in policy.get("tools") or []:
        fn = item.get("function") if isinstance(item, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.add(str(fn["name"]))
        elif isinstance(item, dict) and item.get("name"):
            names.add(str(item["name"]))
    return names


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _tokens(value: Any) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", _norm(value)) if tok]


def _to_float(value: Any) -> float | None:
    if value is None or value is False:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _norm(value)
    if text in {"true", "yes", "in stock", "in_stock"}:
        return True
    if text in {"false", "no", "out of stock", "out_of_stock"}:
        return False
    return None


def _sku_key(product_id: Any, sku: Any) -> str:
    return f"{product_id}::{sku}"


def _sku_stem(sku: Any) -> str:
    text = str(sku or "")
    return text.rsplit("-", 1)[0] if "-" in text else text


def _parse_pair(text: str) -> tuple[str, str] | None:
    match = re.search(r"(\d{6,})::([A-Za-z0-9_-]+)", text or "")
    if not match:
        return None
    return match.group(1), match.group(2)


def _resolution_class(text: Any) -> str | None:
    blob = _norm(text)
    if not blob:
        return None
    if re.search(r"\b(4k|uhd|3840)\b", blob):
        return "4k"
    if re.search(r"\b(qhd|1440|2k)\b", blob):
        return "qhd"
    if re.search(r"\b(full hd|fhd|1080)\b", blob):
        return "fhd"
    if re.search(r"\b(720p|720|hd)\b", blob):
        return "hd"
    return None


def _all_resolution_classes(*parts: Any) -> set[str]:
    found: set[str] = set()
    for part in parts:
        blob = _norm(part)
        if not blob:
            continue
        if re.search(r"\b(4k|uhd|3840)\b", blob):
            found.add("4k")
        if re.search(r"\b(qhd|1440|2k)\b", blob):
            found.add("qhd")
        if re.search(r"\b(full hd|fhd|1080)\b", blob):
            found.add("fhd")
        elif re.search(r"\b(720p|720)\b", blob) or re.search(r"\bhd\b", blob):
            found.add("hd")
    return found


def _best_resolution(*parts: Any) -> str | None:
    rank = {"hd": 1, "fhd": 2, "qhd": 3, "4k": 4}
    best = None
    best_rank = 0
    for klass in _all_resolution_classes(*parts):
        if rank[klass] > best_rank:
            best = klass
            best_rank = rank[klass]
    return best


def _has_pixel_dims(text: Any) -> bool:
    return bool(re.search(r"\d{3,4}\s*[x×]\s*\d{3,4}", str(text or ""), flags=re.I))


def _screen_inches(text: Any) -> float | None:
    blob = str(text or "")
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-)?\s*(?:inch|inches|in\b|\")",
        blob,
        flags=re.I,
    )
    if match:
        return float(match.group(1))
    blob_n = _norm(blob)
    if blob_n.endswith(" inches"):
        return _to_float(blob_n)
    return None


def _refresh_hz(text: Any) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hz|refresh)", str(text or ""), flags=re.I)
    return float(match.group(1)) if match else None


def _budget_in_text(text: str) -> float | None:
    match = re.search(
        r"(?:up to|at or under|at or below|at most|under|below|budget(?: is)?(?: now)?(?: at most)?|can only go up to|can spend up to|only go up to)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:php)?",
        text or "",
        flags=re.I,
    )
    return float(match.group(1)) if match else None


def _category_from_text(text: str) -> str | None:
    match = re.search(r"in the ([^.\n]+?) category", text or "", flags=re.I)
    if match:
        return match.group(1).strip()
    blob = _norm(text)
    if "smartphone" in blob:
        return "Smartphones"
    if "tablet" in blob:
        return "Tablets"
    if "monitor" in blob:
        return "Monitors"
    if "headphone" in blob or "earphone" in blob or "earbuds" in blob:
        return "Headphones"
    return None


class WorldState:
    def __init__(self, query: str, tool_names: set[str]) -> None:
        self.query = query
        self.goal, _, self.rules = query.partition("\n\nTask rules:")
        self.tool_names = tool_names
        self.budget: float | None = None
        self.category: str | None = None
        self.brand_firm: str | None = None
        self.brand_soft: str | None = None
        self.screen_size: float | None = None
        self.resolution: str | None = None
        self.refresh_rate: float | None = None
        self.ram: str | None = None
        self.phrase: str | None = None
        self.prefer_model: str | None = None
        self.prefer_color: str | None = None
        self.named_variants: list[tuple[str, str]] = []
        self.ask_hidden = False
        self.asked_hidden = False
        self.justification_required = False
        self.first_add_prefers_brand = False
        self.survey_required = False
        self.sku_exact = False
        self.recovery_ladder = False
        self.ranking_mode = False
        self.prefer_water = False
        self.require_genuine_device = False
        self.rank_refresh_first = False
        self.products: dict[str, dict[str, Any]] = {}
        self.skus: dict[str, dict[str, Any]] = {}
        self.cart: list[dict[str, Any]] = []
        self.first_add_done = False
        self.last_mutation_turn = 0
        self.mutated_keys: set[str] = set()
        self.turn = 0
        self.search_turns = 0
        self.viewed_ids: set[str] = set()
        self.shopper_notes: list[str] = []
        self.pending_question = False
        self.answered_intervention = False
        self.hidden_received = False
        self.needs_event_ack = False
        self.llm_ok = True
        self.env_nonce = 0
        self.model_title_aliases: set[str] = set()
        self.model_sku_stems: set[str] = set()
        self.model_name_aliases: set[str] = set()
        self._absorb_text(self.goal, initial=True)
        self._absorb_capabilities(query)

    def _absorb_capabilities(self, text: str) -> None:
        blob = _norm(text)
        self.ask_hidden = any(
            phrase in blob
            for phrase in (
                "if you ask",
                "tell you if you ask",
                "will tell you if you ask",
            )
        )
        self.first_add_prefers_brand = any(
            phrase in blob
            for phrase in (
                "initial cart",
                "initial choice",
                "preferred brand is a soft",
                "ideally, i prefer",
                "ideally i prefer",
            )
        )
        self.survey_required = any(
            phrase in blob
            for phrase in ("several distinct", "shortlist", "reformulate")
        )
        self.sku_exact = any(
            phrase in blob
            for phrase in ("exact variant", "specific sku", "exact sku")
        )
        goal_blob = _norm(self.goal)
        self.justification_required = any(
            phrase in goal_blob
            for phrase in (
                "walk me through",
                "tell me why",
                "reason i can trust",
                "give me a reason",
                "why the one you pick",
                "why it's a good pick",
                "why it is a good pick",
            )
        )
        self.recovery_ladder = any(
            phrase in blob
            for phrase in (
                "another color of the same model",
                "never relax",
                "same model before changing",
            )
        )
        self.ranking_mode = any(
            phrase in blob
            for phrase in ("compare only these candidate", "tradeoff priority")
        )
        self.prefer_water = "water resistance" in blob or "sweat" in blob
        self.require_genuine_device = any(
            phrase in blob
            for phrase in (
                "genuine smartphone",
                "not a feature phone",
                "not a feature/basic",
                "reject accessories",
            )
        )
        self.rank_refresh_first = "refresh" in blob and "highest" in blob
        self.named_variants = [
            (a, b) for a, b in re.findall(r"(\d{6,})::([A-Za-z0-9_-]+)", text)
        ]

    def _absorb_text(self, text: str, initial: bool = False) -> None:
        if not text or not str(text).strip():
            return
        budget = _budget_in_text(text)
        if budget is not None and (
            initial
            or re.search(r"budget|balance|spend|can only|sorry", text, flags=re.I)
        ):
            self.budget = budget
        category = _category_from_text(text)
        if category:
            self.category = category
        screen = _screen_inches(text)
        if screen is not None:
            self.screen_size = screen
        refresh = _refresh_hz(text)
        if refresh is not None and "from" not in _norm(text)[:20]:
            self.refresh_rate = refresh
        res = _resolution_class(text)
        if res:
            self.resolution = res
        ram = re.search(r"(\d+\s*gb)\s*ram", text, flags=re.I)
        if ram:
            self.ram = ram.group(1).replace(" ", "").upper()
        phrase = re.search(r'matches\s+"([^"]+)"', text, flags=re.I)
        if not phrase:
            phrase = re.search(r"matches\s+'([^']+)'", text, flags=re.I)
        if not phrase:
            phrase = re.search(r"matches the phrase\s+([^,\n]+)", text, flags=re.I)
        if phrase:
            self.phrase = phrase.group(1).strip()
        pref = re.search(
            r"(?:strongly prefer|prefer(?:red)?)\s+the\s+(\w+)\s+([A-Za-z0-9._-]+)",
            text,
            flags=re.I,
        )
        if pref and _norm(pref.group(1)) in _COLOR_WORDS:
            self.prefer_color = pref.group(1)
            self.prefer_model = pref.group(2).strip(".,;:")
        soft = re.search(
            r"(?:ideally,\s*)?i prefer\s+([A-Za-z][A-Za-z0-9.& -]{1,40}?)(?:[,.]|\s+but\b|$)",
            text,
            flags=re.I,
        )
        if soft and not (pref and _norm(pref.group(1)) in _COLOR_WORDS):
            brand = soft.group(1).strip()
            if _norm(brand) not in _COLOR_WORDS:
                self.brand_soft = brand
        firm = re.search(
            r"(?:need a|want a|find me a|get me an?|an in-stock)\s+([A-Z][\w.&-]+(?:\s+[A-Z][\w.&-]+)?)\s+(?:\d|720|1080|gaming|smartphone|tablet|monitor|headphone)",
            text,
            flags=re.I,
        )
        if firm and "prefer" not in _norm(firm.group(0)):
            brand = re.sub(
                r"^(in-stock|instock|genuine)\s+",
                "",
                firm.group(1).strip(),
                flags=re.I,
            ).strip()
            if brand:
                self.brand_firm = brand
        if "?" in text:
            self.pending_question = True

    def absorb_shopper(self, text: str) -> None:
        if not text or not str(text).strip():
            return
        self.shopper_notes.append(str(text).strip())
        self._absorb_text(str(text), initial=False)
        if self.ask_hidden and not self.hidden_received:
            if _screen_inches(text) or _resolution_class(text) or _refresh_hz(text):
                self.hidden_received = True
            ram = re.search(r"(\d+\s*gb)\s*ram", text, flags=re.I)
            if ram:
                self.ram = ram.group(1).replace(" ", "").upper()
                self.hidden_received = True

    def _product(self, product_id: str) -> dict[str, Any]:
        rec = self.products.setdefault(
            product_id,
            {
                "product_id": product_id,
                "title": "",
                "brand": None,
                "category_path": [],
                "facts": {},
                "viewed": False,
                "compared": False,
            },
        )
        return rec

    def _sku(self, product_id: str, sku: str) -> dict[str, Any]:
        key = _sku_key(product_id, sku)
        rec = self.skus.setdefault(
            key,
            {
                "key": key,
                "product_id": product_id,
                "sku": sku,
                "title": "",
                "brand": None,
                "category_path": [],
                "facts": {},
                "options": "",
                "price": None,
                "in_stock": None,
                "viewed": False,
                "compared": False,
                "inspect_turn": 0,
                "inspect_price": None,
                "inspect_stock": None,
            },
        )
        return rec

    def _sync_product_into_sku(self, sku_rec: dict[str, Any]) -> None:
        prod = self.products.get(str(sku_rec["product_id"]), {})
        if prod.get("title") and not sku_rec.get("title"):
            sku_rec["title"] = prod["title"]
        if prod.get("brand") and not sku_rec.get("brand"):
            sku_rec["brand"] = prod["brand"]
        if prod.get("category_path") and not sku_rec.get("category_path"):
            sku_rec["category_path"] = list(prod["category_path"])
        if prod.get("facts"):
            merged = dict(prod["facts"])
            merged.update(sku_rec.get("facts") or {})
            sku_rec["facts"] = merged

    def ingest_listing(self, item: dict[str, Any], *, viewed: bool = False, compared: bool = False) -> None:
        product_id = str(item.get("product_id") or "")
        if not product_id:
            return
        prod = self._product(product_id)
        if item.get("title"):
            prod["title"] = item["title"]
        if item.get("brand") not in (None, ""):
            prod["brand"] = item["brand"]
        if item.get("category_path"):
            prod["category_path"] = list(item["category_path"])
        if isinstance(item.get("facts"), dict):
            prod["facts"].update(item["facts"])
        if viewed:
            prod["viewed"] = True
            self.viewed_ids.add(product_id)
            for rec in self.skus.values():
                if rec["product_id"] == product_id:
                    rec["viewed"] = True
                    self._sync_product_into_sku(rec)
        if compared:
            prod["compared"] = True
            prod["viewed"] = True
            self.viewed_ids.add(product_id)
        sku = item.get("sku")
        if sku:
            rec = self._sku(product_id, str(sku))
            rec["title"] = rec["title"] or prod["title"]
            rec["brand"] = rec["brand"] or prod["brand"]
            rec["category_path"] = rec["category_path"] or list(prod["category_path"])
            rec["facts"] = dict(prod["facts"])
            if item.get("options") is not None:
                rec["options"] = item.get("options")
            if item.get("price") is not None:
                rec["price"] = _to_float(item.get("price"))
            if item.get("in_stock") is not None:
                rec["in_stock"] = _to_bool(item.get("in_stock"))
            if viewed or compared:
                rec["viewed"] = True
            if compared:
                rec["compared"] = True
            self._sync_product_into_sku(rec)
        for variant in item.get("variants") or []:
            if not isinstance(variant, dict) or not variant.get("sku"):
                continue
            rec = self._sku(product_id, str(variant["sku"]))
            rec["title"] = prod["title"]
            rec["brand"] = prod["brand"]
            rec["category_path"] = list(prod["category_path"])
            rec["facts"] = dict(prod["facts"])
            rec["options"] = variant.get("options") or rec.get("options") or ""
            if variant.get("price") is not None:
                rec["price"] = _to_float(variant.get("price"))
            if variant.get("in_stock") is not None:
                rec["in_stock"] = _to_bool(variant.get("in_stock"))
            if viewed or compared:
                rec["viewed"] = True
            if compared:
                rec["compared"] = True
        if viewed or compared:
            self.remember_model_family(item)

    def remember_model_family(self, item: dict[str, Any]) -> None:
        if not self.recovery_ladder:
            return
        title = _norm(item.get("title") or "")
        sku = str(item.get("sku") or "")
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        model = _norm(facts.get("model") or "")
        hay = " ".join([title, _norm(item.get("options") or ""), model])
        prefer_hit = bool(self.prefer_model and _norm(self.prefer_model) in hay)
        color_hit = bool(self.prefer_color and _norm(self.prefer_color) in hay)
        already = bool(self.model_title_aliases or self.model_sku_stems)
        if not prefer_hit and already:
            if title and title in self.model_title_aliases:
                prefer_hit = True
            if sku and _sku_stem(sku) in self.model_sku_stems:
                prefer_hit = True
            if model and model in self.model_name_aliases:
                prefer_hit = True
        if not prefer_hit and not color_hit:
            return
        if title:
            self.model_title_aliases.add(title)
        if sku:
            self.model_sku_stems.add(_sku_stem(sku))
        if model:
            self.model_name_aliases.add(model)
        for variant in item.get("variants") or []:
            if isinstance(variant, dict) and variant.get("sku"):
                self.model_sku_stems.add(_sku_stem(variant.get("sku")))

    def ingest_inspect(self, item: dict[str, Any]) -> None:
        product_id = str(item.get("product_id") or "")
        sku = str(item.get("sku") or "")
        if not product_id or not sku:
            return
        rec = self._sku(product_id, sku)
        rec["inspect_turn"] = self.turn
        rec["inspect_price"] = _to_float(item.get("price"))
        rec["inspect_stock"] = _to_bool(item.get("in_stock"))
        if rec["inspect_price"] is not None:
            rec["price"] = rec["inspect_price"]
        if rec["inspect_stock"] is not None:
            rec["in_stock"] = rec["inspect_stock"]
        self._sync_product_into_sku(rec)
        if self.recovery_ladder:
            prod = self.products.get(product_id, {})
            payload = {
                "title": rec.get("title") or prod.get("title"),
                "sku": sku,
                "options": rec.get("options"),
                "facts": rec.get("facts") or prod.get("facts"),
            }
            self.remember_model_family(payload)
            if self.sku_ok(rec) and not self.model_title_aliases:
                title = _norm(payload.get("title") or "")
                if title:
                    self.model_title_aliases.add(title)
                self.model_sku_stems.add(_sku_stem(sku))
                model = _norm((payload.get("facts") or {}).get("model") or "")
                if model:
                    self.model_name_aliases.add(model)

    def ingest_cart(self, cart: Any) -> None:
        if not isinstance(cart, list):
            return
        self.cart = [row for row in cart if isinstance(row, dict)]
        if self.cart:
            self.first_add_done = True
        for row in self.cart:
            pid = str(row.get("product_id") or "")
            sku = str(row.get("sku") or "")
            if pid and sku:
                rec = self._sku(pid, sku)
                if row.get("current_price") is not None:
                    rec["price"] = _to_float(row.get("current_price"))
                if row.get("in_stock") is not None:
                    rec["in_stock"] = _to_bool(row.get("in_stock"))

    def mark_mutation(self, key: str | None = None) -> None:
        self.last_mutation_turn = self.turn
        if key:
            self.mutated_keys.add(key)

    def inspect_is_fresh(self, rec: dict[str, Any]) -> bool:
        turn = int(rec.get("inspect_turn") or 0)
        if turn <= 0:
            return False
        key = rec.get("key") or _sku_key(rec.get("product_id"), rec.get("sku"))
        if key and key not in self.mutated_keys:
            return True
        return turn >= self.last_mutation_turn

    def current_price(self, rec: dict[str, Any]) -> float | None:
        if rec.get("inspect_price") is not None and self.inspect_is_fresh(rec):
            return rec["inspect_price"]
        return rec.get("price")

    def current_stock(self, rec: dict[str, Any]) -> bool | None:
        if rec.get("inspect_stock") is not None and self.inspect_is_fresh(rec):
            return rec["inspect_stock"]
        return rec.get("in_stock")

    def cart_keys(self) -> set[str]:
        keys = set()
        for row in self.cart:
            pid = row.get("product_id")
            sku = row.get("sku")
            if pid and sku:
                keys.add(_sku_key(pid, sku))
        return keys

    def digest(self) -> str:
        lines = [
            f"budget={self.budget} category={self.category} firm_brand={self.brand_firm} "
            f"soft_brand={self.brand_soft} screen={self.screen_size} resolution={self.resolution} "
            f"refresh={self.refresh_rate} ram={self.ram} phrase={self.phrase}",
            f"prefer_model={self.prefer_model} prefer_color={self.prefer_color} "
            f"ask_hidden={self.ask_hidden}/{self.asked_hidden}/{self.hidden_received} "
            f"justify={self.justification_required} first_add={self.first_add_done} "
            f"survey={self.survey_required} ranking={self.ranking_mode}",
            f"named={self.named_variants[:8]} cart={[(c.get('product_id'), c.get('sku')) for c in self.cart]}",
        ]
        ranked = self.ranked_candidates()[:8]
        for rec in ranked:
            lines.append(
                f"- {rec['product_id']}::{rec['sku']} viewed={rec.get('viewed')} "
                f"fresh={self.inspect_is_fresh(rec)} price={self.current_price(rec)} "
                f"stock={self.current_stock(rec)} brand={rec.get('brand')} "
                f"cat={rec.get('category_path')} opt={rec.get('options')} "
                f"ok={self.sku_ok(rec)}"
            )
        return "\n".join(lines)

    def sku_ok(self, rec: dict[str, Any], *, require_fresh_inspect: bool = False) -> bool:
        if require_fresh_inspect and not self.inspect_is_fresh(rec):
            return False
        if self.named_variants and (rec["product_id"], rec["sku"]) not in self.named_variants:
            return False
        price = self.current_price(rec)
        stock = self.current_stock(rec)
        if self.budget is not None and price is not None and price > self.budget + 1e-9:
            return False
        if stock is False:
            return False
        if require_fresh_inspect and (price is None or stock is not True):
            return False
        brand = rec.get("brand") or self.products.get(rec["product_id"], {}).get("brand")
        if self.brand_firm:
            if brand and _norm(brand) != _norm(self.brand_firm):
                return False
            if not brand:
                title = _norm(rec.get("title") or self.products.get(rec["product_id"], {}).get("title"))
                if _norm(self.brand_firm) not in title:
                    return False
        path = rec.get("category_path") or self.products.get(rec["product_id"], {}).get("category_path") or []
        if self.category and rec.get("viewed"):
            blob = " / ".join(str(x) for x in path).lower()
            want = _norm(self.category)
            if want not in blob and want.rstrip("s") not in blob:
                return False
        if self.phrase:
            hay = " ".join(
                [
                    str(rec.get("title") or ""),
                    str(self.products.get(rec["product_id"], {}).get("title") or ""),
                    str(rec.get("options") or ""),
                ]
            )
            if not all(tok in _norm(hay) for tok in _tokens(self.phrase)):
                return False
        if self.screen_size is not None and rec.get("viewed"):
            facts = rec.get("facts") or self.products.get(rec["product_id"], {}).get("facts") or {}
            measured = _screen_inches(facts.get("size") or facts.get("screen_size") or rec.get("options"))
            if measured is None:
                measured = _screen_inches(rec.get("title") or self.products.get(rec["product_id"], {}).get("title"))
            if measured is None or abs(measured - self.screen_size) > 0.051:
                return False
        if self.resolution and rec.get("viewed"):
            facts = rec.get("facts") or {}
            got = _best_resolution(
                facts.get("display"),
                facts.get("resolution"),
                rec.get("options"),
                rec.get("title"),
                self.products.get(rec["product_id"], {}).get("title"),
            )
            rank = {"hd": 1, "fhd": 2, "qhd": 3, "4k": 4}
            if not got or rank.get(got, 0) < rank.get(self.resolution, 0):
                return False
        if (self.prefer_water or (self.category and "headphone" in _norm(self.category))) and rec.get("viewed"):
            if self.is_headphone_accessory(rec):
                return False
        if self.refresh_rate and rec.get("viewed"):
            facts = rec.get("facts") or {}
            hz = _refresh_hz(facts.get("refresh_rate") or facts.get("display") or rec.get("title"))
            if hz is None or hz + 1e-9 < self.refresh_rate:
                return False
        if self.require_genuine_device and rec.get("viewed"):
            prod = self.products.get(rec["product_id"], {})
            hay = _norm(
                " ".join(
                    [
                        str(prod.get("title") or ""),
                        str((prod.get("facts") or {}).get("device_operating_system") or ""),
                        str((prod.get("facts") or {}).get("operating_system") or ""),
                        " ".join(str(x) for x in (prod.get("category_path") or [])),
                    ]
                )
            )
            if self.category and "smartphone" in _norm(self.category):
                os_ok = any(
                    token in hay
                    for token in (
                        "android",
                        "ios",
                        "iphone",
                        "harmony",
                        "hyperos",
                        "tizen",
                    )
                )
                if "smartphone" not in hay and not os_ok:
                    return False
            if any(hint in hay for hint in _HARD_ACCESSORY):
                return False
            if any(hint in hay for hint in _ACCESSORY_HINTS) and not any(
                word in hay for word in _DEVICE_WORDS
            ):
                return False
        if self.prefer_model and rec.get("viewed") and not self.recovery_ladder:
            hay = _norm(
                " ".join(
                    [
                        str(rec.get("title") or ""),
                        str(rec.get("options") or ""),
                        str((rec.get("facts") or {}).get("model") or ""),
                    ]
                )
            )
            if _norm(self.prefer_model) not in hay:
                return False
        return True

    def evidence_ok(self, rec: dict[str, Any]) -> bool:
        viewed = bool(rec.get("viewed") or rec.get("compared"))
        prod = self.products.get(rec["product_id"], {})
        viewed = viewed or bool(prod.get("viewed") or prod.get("compared"))
        viewed = viewed or bool(rec.get("title") or prod.get("title"))
        return viewed and self.inspect_is_fresh(rec) and self.current_stock(rec) is True

    def rec_refresh(self, rec: dict[str, Any]) -> float:
        facts = rec.get("facts") or {}
        return (
            _refresh_hz(rec.get("options"))
            or _refresh_hz(facts.get("refresh_rate"))
            or _refresh_hz(facts.get("display"))
            or _refresh_hz(rec.get("title"))
            or _refresh_hz(self.products.get(rec["product_id"], {}).get("title"))
            or 0.0
        )

    def rec_inches(self, rec: dict[str, Any]) -> float:
        facts = rec.get("facts") or {}
        return (
            _screen_inches(rec.get("options"))
            or _screen_inches(facts.get("size") or facts.get("screen_size"))
            or _screen_inches(rec.get("title"))
            or _screen_inches(self.products.get(rec["product_id"], {}).get("title"))
            or 0.0
        )

    def dominates(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_price = self.current_price(left)
        right_price = self.current_price(right)
        if left_price is None or right_price is None:
            return False
        better_or_equal = (
            self.rec_refresh(left) + 1e-9 >= self.rec_refresh(right)
            and self.rec_inches(left) + 1e-9 >= self.rec_inches(right)
            and left_price <= right_price + 1e-9
        )
        strictly_better = (
            self.rec_refresh(left) > self.rec_refresh(right) + 1e-9
            or self.rec_inches(left) > self.rec_inches(right) + 1e-9
            or left_price + 1e-9 < right_price
        )
        return better_or_equal and strictly_better

    def is_headphone_accessory(self, rec: dict[str, Any]) -> bool:
        prod = self.products.get(str(rec.get("product_id") or ""), {})
        hay = _norm(
            " ".join(
                [
                    str(rec.get("title") or ""),
                    str(prod.get("title") or ""),
                    " ".join(str(x) for x in (prod.get("category_path") or [])),
                ]
            )
        )
        if any(hint in hay for hint in _HEADPHONE_ACCESSORY):
            return True
        return False

    def water_score(self, rec: dict[str, Any]) -> int:
        if self.is_headphone_accessory(rec):
            return 0
        prod = self.products.get(str(rec.get("product_id") or ""), {})
        hay = _norm(
            " ".join(
                [
                    str(rec.get("title") or ""),
                    str(prod.get("title") or ""),
                    str(rec.get("options") or ""),
                    " ".join(str(v) for v in (rec.get("facts") or {}).values()),
                    " ".join(str(v) for v in (prod.get("facts") or {}).values()),
                ]
            )
        )
        score = 0
        for hint in _WATER_HINTS:
            if hint in hay:
                score += 2
        if "sweatproof" in hay or "sweat-proof" in hay:
            score += 4
        if "sport" in hay or "running" in hay or "gym" in hay or "workout" in hay:
            score += 2
        if "ip68" in hay or "ipx8" in hay:
            score += 3
        elif "ip67" in hay or "ipx7" in hay:
            score += 2
        elif "ipx5" in hay or "ipx4" in hay:
            score += 2
        return score

    def same_model(self, rec: dict[str, Any]) -> bool:
        prod = self.products.get(rec.get("product_id"), {})
        hay = _norm(
            " ".join(
                [
                    str(rec.get("title") or ""),
                    str(prod.get("title") or ""),
                    str(rec.get("options") or ""),
                    str((rec.get("facts") or {}).get("model") or ""),
                    str((prod.get("facts") or {}).get("model") or ""),
                ]
            )
        )
        if self.prefer_model and _norm(self.prefer_model) in hay:
            return True
        title = _norm(rec.get("title") or prod.get("title") or "")
        if title and title in self.model_title_aliases:
            return True
        stem = _sku_stem(rec.get("sku"))
        if stem and stem in self.model_sku_stems:
            return True
        model = _norm((rec.get("facts") or {}).get("model") or (prod.get("facts") or {}).get("model") or "")
        if model and model in self.model_name_aliases:
            return True
        return False

    def recovery_rank(self, rec: dict[str, Any]) -> tuple:
        hay = _norm(
            " ".join(
                [
                    str(rec.get("title") or ""),
                    str(rec.get("options") or ""),
                    str((rec.get("facts") or {}).get("model") or ""),
                ]
            )
        )
        model_hit = self.same_model(rec)
        color_hit = bool(self.prefer_color and _norm(self.prefer_color) in hay)
        if model_hit and color_hit:
            tier = 0
        elif model_hit:
            tier = 1
        else:
            tier = 2
        return (tier, -(self.water_score(rec)), self.current_price(rec) or 1e18)

    def ranking_key(self, rec: dict[str, Any]) -> tuple:
        price = self.current_price(rec)
        price_key = price if price is not None else 1e18
        if self.rank_refresh_first:
            return (-self.rec_refresh(rec), -self.rec_inches(rec), price_key)
        if self.prefer_water:
            water = self.water_score(rec)
            wet = 1 if water >= 3 else 0
            if not self.first_add_done:
                return (-wet, -(price if price is not None else 0), -water)
            return (-water, price_key)
        bait = (self.justification_required or (self.ask_hidden and self.hidden_received)) and not self.first_add_done
        if bait:
            spec = 0
            if self.resolution:
                got = _best_resolution(
                    (rec.get("facts") or {}).get("display"),
                    rec.get("options"),
                    rec.get("title"),
                    self.products.get(rec["product_id"], {}).get("title"),
                )
                rank = {"hd": 1, "fhd": 2, "qhd": 3, "4k": 4}
                if got == self.resolution:
                    spec += 3
                elif got and rank.get(got, 0) >= rank.get(self.resolution, 0):
                    spec += 1
            if self.screen_size and abs((self.rec_inches(rec) or 0) - self.screen_size) <= 0.051:
                spec += 2
            if self.brand_firm and _norm(rec.get("brand")) == _norm(self.brand_firm):
                spec += 1
            return (-spec, -(price if price is not None else 0))
        if self.justification_required:
            return (-(price if price is not None else 0),)
        if self.brand_soft and _norm(rec.get("brand")) == _norm(self.brand_soft):
            return (0, price_key)
        return (1, price_key)

    def ranked_candidates(self) -> list[dict[str, Any]]:
        recs = list(self.skus.values())
        recs.sort(key=lambda rec: self.recovery_rank(rec) if self.recovery_ladder else self.ranking_key(rec))
        return recs

    def preferred_brand_available(self) -> bool:
        if not self.brand_soft:
            return False
        for rec in self.skus.values():
            if _norm(rec.get("brand")) == _norm(self.brand_soft) and self.sku_ok(rec):
                return True
        return False

    def phrase_hit(self, product_id: str, rec: dict[str, Any] | None = None) -> bool:
        if not self.phrase:
            return True
        prod = self.products.get(str(product_id), {})
        hay = " ".join(
            [
                str((rec or {}).get("title") or ""),
                str(prod.get("title") or ""),
                str((rec or {}).get("options") or ""),
            ]
        )
        return all(tok in _norm(hay) for tok in _tokens(self.phrase))

    def enough_survey(self) -> bool:
        if self.ask_hidden and not self.hidden_received:
            return False
        if self.named_variants:
            needed = {pid for pid, _sku in self.named_variants}
            return needed.issubset(self.viewed_ids)
        if not self.survey_required:
            return True
        if self.phrase:
            if self.search_turns < 3:
                return False
            hits = [pid for pid in self.products if self.phrase_hit(pid)]
            if not hits:
                return False
            viewed_hits = [pid for pid in hits if pid in self.viewed_ids]
            return len(viewed_hits) >= min(4, max(1, len(hits)))
        return len(self.viewed_ids) >= min(3, max(1, len(self.products)))

    def best_ready(self, *, must_verify: bool = True) -> dict[str, Any] | None:
        options = []
        for rec in self.ranked_candidates():
            if must_verify and not self.evidence_ok(rec):
                continue
            if not self.sku_ok(rec, require_fresh_inspect=must_verify):
                continue
            if self.first_add_prefers_brand and not self.first_add_done:
                if self.preferred_brand_available() and _norm(rec.get("brand")) != _norm(
                    self.brand_soft
                ):
                    continue
            options.append(rec)
        if not options:
            return None
        if self.ranking_mode:
            front = [
                rec
                for rec in options
                if not any(self.dominates(other, rec) for other in options if other is not rec)
            ]
            options = front or options
        if self.recovery_ladder:
            options.sort(key=self.recovery_rank)
        else:
            options.sort(key=self.ranking_key)
        return options[0]


def _overlay(state: WorldState) -> str:
    bits = [
        "Follow the shopper goal and the Task rules in the query. Firm constraints beat soft preferences.",
        "Use only supplied tools. Batch complementary calls in one turn (search+filter, several views, compare up to 5).",
        "Check catalog category_path from view/compare; titles alone do not prove category.",
        "Every order must use an exact SKU that was viewed or compared and inspect_stock'ed after the latest cart or price event.",
        "Once a SKU is verified in-stock and in-budget, place the order. Do not wait for permission.",
        "If the shopper replaces a requirement, keep every field they did not change.",
        "Read shopper messages after cart actions before ordering. Answer direct questions with message.",
    ]
    if state.ask_hidden and not state.asked_hidden:
        bits.append("A firm spec is withheld until you ask. Send a short message asking for it immediately.")
    if state.first_add_prefers_brand:
        bits.append(
            "Preferred brand is soft. The first add_to_cart must use a fully compliant preferred-brand SKU when one exists."
        )
    if state.survey_required:
        bits.append("Reformulate searches and inspect several distinct products before the first cart add.")
    if state.sku_exact:
        bits.append("Specs must hold on the exact SKU option, not the parent title.")
    if state.ranking_mode:
        bits.append("Compare only the named candidate variants and apply the stated tradeoff order to raw facts.")
    if state.recovery_ladder:
        bits.append(
            "Never relax category, budget, or stock. If the preferred model/color vanishes, try the same model in another color, then another model."
        )
    if state.justification_required:
        bits.append(
            "View or compare before the first cart add. Order justification.claims must be {field,value} from inspected facts only."
        )
    if state.prefer_water:
        bits.append("Rank by water/sweat resistance from raw facts, not by lowest price.")
    return " ".join(bits)


def _system_prompt(state: WorldState) -> str:
    return (
        "You are a shopping agent operating inside a sealed environment. "
        "Choose environment tools to find a valid in-stock item and place a test order. "
        + _overlay(state)
        + "\nCurrent world state:\n"
        + state.digest()
    )


def _justification(state: WorldState, rec: dict[str, Any]) -> dict[str, Any] | None:
    if not state.justification_required:
        return None
    prod = state.products.get(rec["product_id"], {})
    facts = dict(prod.get("facts") or {})
    facts.update(rec.get("facts") or {})
    claims: list[dict[str, str]] = []

    def add(field: str, value: Any) -> None:
        if value is None or value == "":
            return
        if field not in _CLAIM_FIELDS:
            return
        text = value if isinstance(value, str) else json.dumps(value) if isinstance(value, bool) else str(value)
        if field == "in_stock":
            text = "true" if _to_bool(value) else "false"
        claims.append({"field": field, "value": text})

    add("brand", rec.get("brand") or prod.get("brand"))
    add("price", state.current_price(rec))
    add("in_stock", state.current_stock(rec) if state.current_stock(rec) is not None else True)
    inches = state.rec_inches(rec)
    if state.screen_size is not None and inches and abs(inches - state.screen_size) <= 0.051:
        add("screen_size", f"{state.screen_size:g}")
    got = _best_resolution(
        facts.get("display"),
        facts.get("resolution"),
        rec.get("options"),
        rec.get("title"),
        prod.get("title"),
    )
    rank = {"hd": 1, "fhd": 2, "qhd": 3, "4k": 4}
    if state.resolution and got and rank.get(got, 0) >= rank.get(state.resolution, 0):
        add("resolution", state.resolution)
    if state.refresh_rate:
        hz = _refresh_hz(facts.get("refresh_rate") or facts.get("display") or rec.get("options"))
        if hz and hz + 1e-9 >= state.refresh_rate:
            add("refresh_rate", hz)
    if not claims:
        return None
    return {"claims": claims}


def _has(state: WorldState, name: str) -> bool:
    return name in state.tool_names or not state.tool_names


def _act(name: str, **args: Any) -> dict[str, Any]:
    clean = {key: value for key, value in args.items() if value is not None}
    return {"name": name, "args": clean}


def _dedupe(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for action in actions:
        key = json.dumps(action, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
    return out


def _guard(actions: list[dict[str, Any]], state: WorldState, max_calls: int) -> list[dict[str, Any]]:
    allowed = state.tool_names
    cleaned: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        name = str(action.get("name") or "")
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        if allowed and name not in allowed:
            continue
        if name in {"view"} and not args.get("product_id"):
            continue
        if name == "compare":
            ids = args.get("product_ids")
            if not isinstance(ids, list) or not ids:
                continue
            args = {**args, "product_ids": [str(x) for x in ids[:5]]}
        if name in {"inspect_stock", "add_to_cart", "remove_from_cart"} and not (
            args.get("product_id") and args.get("sku")
        ):
            continue
        if name == "message":
            if not str(args.get("content") or "").strip():
                continue
            if not _inference_ok() and not state.first_add_done:
                continue
        if name == "search" and not str(args.get("query") or "").strip():
            continue
        if name == "place_test_order":
            rec = None
            pid, sku = args.get("product_id"), args.get("sku")
            if pid and sku:
                rec = state.skus.get(_sku_key(pid, sku))
            if rec is None:
                rec = state.best_ready(must_verify=True)
            legal = (
                rec is not None
                and state.evidence_ok(rec)
                and state.sku_ok(rec, require_fresh_inspect=True)
            )
            if not legal and state.first_add_done and rec is not None and rec.get("viewed"):
                legal = True
            if not legal:
                if rec:
                    cleaned.append(_act("inspect_stock", product_id=rec["product_id"], sku=rec["sku"]))
                continue
            args = {"product_id": rec["product_id"], "sku": rec["sku"]}
            just = _justification(state, rec)
            if just:
                args["justification"] = just
        cleaned.append({"name": name, "args": args})
    if (
        state.ask_hidden
        and not state.asked_hidden
        and _has(state, "message")
        and _inference_ok()
    ):
        ask = _act(
            "message",
            content="What is the remaining firm hardware spec I should enforce?",
        )
        if not any(item["name"] == "message" for item in cleaned):
            cleaned = [ask] + cleaned
    return _dedupe(cleaned)[:max_calls]


def _unviewed_named(state: WorldState) -> list[str]:
    return [pid for pid, _sku in state.named_variants if pid not in state.viewed_ids]


def _needs_inspect(state: WorldState) -> list[dict[str, Any]]:
    out = []
    keys = list(state.named_variants) if state.named_variants else []
    if not keys:
        want = 3 if not state.first_add_done else 1
        if state.prefer_water or state.recovery_ladder:
            want = 4 if not state.first_add_done else 1
        ranked = state.ranked_candidates()
        if state.prefer_water:
            ranked = sorted(
                ranked,
                key=lambda rec: (
                    -state.water_score(rec),
                    -(state.current_price(rec) or 0),
                ),
            )
        if state.screen_size is not None:
            ranked = sorted(
                ranked,
                key=lambda rec: (
                    0 if abs((state.rec_inches(rec) or 0) - state.screen_size) <= 0.021 else 1,
                    0,
                ),
            )
        for rec in ranked:
            if rec.get("viewed") and state.sku_ok(rec) and not state.inspect_is_fresh(rec):
                if state.prefer_water and state.water_score(rec) <= 0 and not state.first_add_done:
                    continue
                if state.recovery_ladder and not state.same_model(rec) and not state.first_add_done:
                    if any(state.same_model(other) for other in state.skus.values()):
                        continue
                keys.append((rec["product_id"], rec["sku"]))
            if len(keys) >= want:
                break
    for pid, sku in keys:
        rec = state.skus.get(_sku_key(pid, sku))
        if rec is None or not state.inspect_is_fresh(rec):
            out.append(_act("inspect_stock", product_id=pid, sku=sku))
    return out


def _search_query(state: WorldState) -> str:
    if state.phrase:
        tokens = _tokens(state.phrase)
        variants = [
            f"{state.phrase} {state.category or ''}".strip(),
            state.phrase,
            " ".join(reversed(tokens)) if tokens else state.phrase,
            f"{tokens[0]} {state.category or ''}".strip() if tokens else state.phrase,
        ]
        return variants[state.search_turns % len(variants)]
    parts = []
    if state.brand_firm:
        parts.append(state.brand_firm)
    elif state.brand_soft and state.search_turns == 0:
        parts.append(state.brand_soft)
    if state.prefer_model and (not state.first_add_done or state.search_turns % 2 == 0):
        parts.append(state.prefer_model)
    have_preferred = any(
        rec.get("viewed") and state.recovery_rank(rec)[0] == 0 for rec in state.skus.values()
    )
    if state.prefer_color and not have_preferred:
        parts.append(state.prefer_color)
    if state.category:
        parts.append(state.category)
    if state.screen_size:
        parts.append(f"{state.screen_size}-inch")
    if state.resolution:
        parts.append(state.resolution)
    if state.prefer_water:
        parts.extend(["sweatproof", "waterproof"])
    return " ".join(str(p) for p in parts if p).strip() or "in stock"


def _explore(state: WorldState) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if state.ask_hidden and not state.asked_hidden and _inference_ok():
        return [_act("message", content="What is the remaining firm hardware spec I should enforce?")]
    unnamed_views = _unviewed_named(state)
    if unnamed_views:
        if _has(state, "compare") and len(unnamed_views) > 1:
            actions.append(_act("compare", product_ids=unnamed_views[:5]))
        for pid in unnamed_views[:6]:
            actions.append(_act("view", product_id=pid))
        return actions
    inspects = _needs_inspect(state)
    if inspects:
        return inspects
    if state.phrase and state.search_turns < 3 and len(state.viewed_ids) >= 3 * max(1, state.search_turns):
        q = _search_query(state)
        actions.append(_act("search", query=q, k=10, in_stock=True))
        if state.category or state.budget is not None:
            filt: dict[str, Any] = {"k": 10, "in_stock": True}
            if state.category:
                filt["category"] = state.category
            if state.budget is not None:
                filt["max_price"] = state.budget
            actions.append(_act("filter", **filt))
        return actions
    unseen = [pid for pid in state.products if pid not in state.viewed_ids]
    if state.phrase:
        phrase_unseen = [pid for pid in unseen if state.phrase_hit(pid)]
        if phrase_unseen:
            unseen = phrase_unseen
        elif any(state.phrase_hit(pid) for pid in state.products):
            unseen = []
    if state.require_genuine_device or (
        state.category and any(word in _norm(state.category) for word in _DEVICE_WORDS)
    ):
        devices = [pid for pid in unseen if _device_like(state, pid)]
        if devices:
            unseen = devices
    if state.prefer_model:
        hits = [
            pid
            for pid in unseen
            if _norm(state.prefer_model) in _norm(state.products.get(pid, {}).get("title"))
        ]
        if hits:
            unseen = hits + [pid for pid in unseen if pid not in hits]
    if state.recovery_ladder and state.model_title_aliases:
        family = [
            pid
            for pid in unseen
            if _norm(state.products.get(pid, {}).get("title") or "") in state.model_title_aliases
        ]
        if family:
            unseen = family + [pid for pid in unseen if pid not in family]
    if state.screen_size is not None:
        sized = [
            pid
            for pid in unseen
            if abs((_product_inches(state, pid) or 0) - state.screen_size) <= 0.021
        ]
        if sized:
            unseen = sized + [pid for pid in unseen if pid not in sized]
    if state.prefer_water:
        wet = [pid for pid in unseen if _product_water(state, pid) > 0]
        if wet:
            unseen = wet + [pid for pid in unseen if pid not in wet]
    if state.budget is not None:
        affordable = [
            pid
            for pid in unseen
            if _product_min_price(state, pid) is not None
            and _product_min_price(state, pid) <= state.budget + 1e-9
        ]
        if affordable:
            unseen = affordable + [pid for pid in unseen if pid not in affordable]
    if unseen:
        if _has(state, "compare") and len(unseen) > 1:
            actions.append(_act("compare", product_ids=unseen[:5]))
        for pid in unseen[:8]:
            actions.append(_act("view", product_id=pid))
        return actions
    search_budget = 6 if (state.phrase or state.recovery_ladder or state.cart or state.prefer_water) else 3
    if state.ask_hidden and not state.hidden_received:
        search_budget = min(search_budget, 3)
    if not state.products or state.search_turns < search_budget:
        q = _search_query(state)
        if state.search_turns == 1 and state.brand_soft:
            q = f"{state.brand_soft} {state.category or ''} {state.phrase or ''}".strip()
        if state.recovery_ladder and state.prefer_model and state.search_turns >= 1:
            q = f"{state.prefer_model} {state.category or ''}".strip()
        actions.append(_act("search", query=q, k=10, in_stock=True))
        if state.search_turns >= 1:
            filt: dict[str, Any] = {"k": 10, "in_stock": True}
            if state.category:
                filt["category"] = state.category
            if state.budget is not None:
                filt["max_price"] = state.budget
            if state.brand_firm:
                filt["brand"] = state.brand_firm
            elif state.brand_soft:
                filt["brand"] = state.brand_soft
            actions.append(_act("filter", **filt))
        return actions
    return actions


def _order_action(state: WorldState, rec: dict[str, Any]) -> dict[str, Any]:
    args = {"product_id": rec["product_id"], "sku": rec["sku"]}
    just = _justification(state, rec)
    if just:
        args["justification"] = just
    return {"name": "place_test_order", "args": args}


def _backup_skus(state: WorldState, *, require_fresh: bool = False) -> list[dict[str, Any]]:
    held = state.cart_keys()
    out = []
    for rec in state.ranked_candidates():
        if rec["key"] in held:
            continue
        if rec.get("viewed") and state.sku_ok(rec, require_fresh_inspect=require_fresh):
            out.append(rec)
    return out


def _product_min_price(state: WorldState, product_id: str) -> float | None:
    prices = [
        rec.get("price")
        for rec in state.skus.values()
        if rec.get("product_id") == str(product_id) and rec.get("price") is not None
    ]
    return min(prices) if prices else None


def _product_inches(state: WorldState, product_id: str) -> float:
    prod = state.products.get(str(product_id), {})
    facts = prod.get("facts") or {}
    return (
        _screen_inches(facts.get("size") or facts.get("screen_size"))
        or _screen_inches(prod.get("title"))
        or 0.0
    )


def _product_water(state: WorldState, product_id: str) -> int:
    dummy = {
        "product_id": str(product_id),
        "title": state.products.get(str(product_id), {}).get("title"),
        "facts": state.products.get(str(product_id), {}).get("facts") or {},
        "options": "",
    }
    return state.water_score(dummy)


def _device_like(state: WorldState, product_id: str) -> bool:
    prod = state.products.get(str(product_id), {})
    hay = _norm(
        " ".join(
            [
                str(prod.get("title") or ""),
                " ".join(str(x) for x in (prod.get("category_path") or [])),
                str((prod.get("facts") or {}).get("device_operating_system") or ""),
            ]
        )
    )
    if any(hint in hay for hint in _ACCESSORY_HINTS) and not any(word in hay for word in _DEVICE_WORDS):
        return False
    if any(hint in hay for hint in _HARD_ACCESSORY + ("holder",)):
        return False
    if state.category:
        want = _norm(state.category).rstrip("s")
        if want and want in hay:
            return True
    return any(word in hay for word in _DEVICE_WORDS)


def _select_backup(state: WorldState) -> dict[str, Any] | None:
    held = state.cart_keys()
    pool = [rec for rec in _verified_pool(state) if rec["key"] not in held]
    if not pool:
        pool = [
            rec
            for rec in state.ranked_candidates()
            if rec["key"] not in held and rec.get("viewed") and state.sku_ok(rec)
        ]
    if not pool:
        return None
    if state.prefer_water:
        wet = [rec for rec in pool if state.water_score(rec) > 0]
        if not wet:
            wet = [
                rec
                for rec in state.ranked_candidates()
                if rec["key"] not in held
                and rec.get("viewed")
                and state.sku_ok(rec)
                and state.water_score(rec) > 0
            ]
        if wet:
            wet.sort(key=state.ranking_key)
            return wet[0]
        return None
    if state.recovery_ladder:
        same = [rec for rec in pool if state.recovery_rank(rec)[0] <= 1]
        if not same:
            same = [
                rec
                for rec in state.ranked_candidates()
                if rec["key"] not in held
                and rec.get("viewed")
                and state.sku_ok(rec)
                and state.recovery_rank(rec)[0] <= 1
            ]
        if same:
            same.sort(key=state.recovery_rank)
            return same[0]
        return None
    pool.sort(key=state.recovery_rank if state.recovery_ladder else state.ranking_key)
    return pool[0]


def _closing_message(state: WorldState, rec: dict[str, Any] | None) -> dict[str, Any] | None:
    asked = state.pending_question or any("?" in note for note in state.shopper_notes)
    if not asked or not _has(state, "message"):
        return None
    title = ""
    spec = ""
    if rec:
        title = str(rec.get("title") or rec.get("product_id") or "this listing")
        if state.screen_size:
            spec = f" It has a {state.screen_size}-inch screen."
        elif state.resolution:
            spec = f" It meets the {state.resolution} resolution class."
    state.answered_intervention = True
    state.pending_question = False
    return _act(
        "message",
        content=f"I am going with {title or 'the best matching in-stock option'}.{spec} It is in stock and within budget.",
    )


def _verified_pool(state: WorldState) -> list[dict[str, Any]]:
    pool = []
    for rec in state.ranked_candidates():
        if state.evidence_ok(rec) and state.sku_ok(rec, require_fresh_inspect=True):
            pool.append(rec)
    if state.ranking_mode:
        front = [
            rec
            for rec in pool
            if not any(state.dominates(other, rec) for other in pool if other is not rec)
        ]
        pool = front or pool
    if state.recovery_ladder:
        pool.sort(key=state.recovery_rank)
    else:
        pool.sort(key=state.ranking_key)
    return pool


def _commit_sequence(state: WorldState, primary: dict[str, Any]) -> list[dict[str, Any]]:
    return [_act("add_to_cart", product_id=primary["product_id"], sku=primary["sku"])]


def _post_event_close(state: WorldState) -> list[dict[str, Any]]:
    """Notice the market event, recover if needed, and terminate this turn.

    The first turn after the event must end with place_test_order. If it does
    not, the shopper simulator runs and a 401 quarantines the session.
    """
    out: list[dict[str, Any]] = []
    if _has(state, "inspect_cart"):
        out.append(_act("inspect_cart"))
    good: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for row in state.cart:
        pid, sku = row.get("product_id"), row.get("sku")
        if not pid or not sku:
            continue
        rec = state.skus.get(_sku_key(pid, sku))
        if rec is not None and state.sku_ok(rec) and state.current_stock(rec) is True:
            good.append(rec)
        else:
            bad.append({"product_id": pid, "sku": sku})
    backup = _select_backup(state)
    if good and not bad:
        note = _closing_message(state, good[0])
        if note:
            out.append(note)
        out.append(_order_action(state, good[0]))
        return out
    if good and backup is None:
        note = _closing_message(state, good[0])
        if note:
            out.append(note)
        out.append(_order_action(state, good[0]))
        return out
    if backup is not None:
        out.append(_act("inspect_stock", product_id=backup["product_id"], sku=backup["sku"]))
        for row in state.cart:
            pid, sku = row.get("product_id"), row.get("sku")
            if pid and sku:
                out.append(_act("remove_from_cart", product_id=pid, sku=sku))
        out.append(_act("add_to_cart", product_id=backup["product_id"], sku=backup["sku"]))
        note = _closing_message(state, backup)
        if note:
            out.append(note)
        out.append(_order_action(state, backup))
        return out
    if state.cart:
        row = state.cart[0]
        rec = state.skus.get(_sku_key(row.get("product_id"), row.get("sku")))
        if rec is not None:
            note = _closing_message(state, rec)
            if note:
                out.append(note)
            out.append(_order_action(state, rec))
    return out


def _cart_actions(state: WorldState) -> list[dict[str, Any]]:
    if not state.cart:
        return []
    return _post_event_close(state)


def _shopper_reply(state: WorldState) -> list[dict[str, Any]]:
    if not state.pending_question or state.answered_intervention:
        return []
    if not _inference_ok() or not _has(state, "message"):
        state.pending_question = False
        return []
    rec = state.best_ready(must_verify=False)
    title = ""
    spec = ""
    if rec:
        title = str(rec.get("title") or rec.get("product_id") or "the best matching listing")
        if state.screen_size:
            spec = f" It has a {state.screen_size}-inch screen."
        elif state.resolution:
            spec = f" It meets the {state.resolution} resolution requirement."
    content = (
        f"I am going with {title}.{spec} It is in stock and within the stated budget."
    )
    state.answered_intervention = True
    state.pending_question = False
    return [_act("message", content=content)]


def _close_if_ready(actions: list[dict[str, Any]], state: WorldState, max_calls: int) -> list[dict[str, Any]]:
    if state.ask_hidden and not state.asked_hidden and _inference_ok():
        return _guard(
            [_act("message", content="What is the remaining firm hardware spec I should enforce?")],
            state,
            max_calls,
        )
    cart_follow = _cart_actions(state)
    if cart_follow:
        return _guard(cart_follow, state, max_calls)
    if state.named_variants and _unviewed_named(state):
        return _guard(_explore(state), state, max_calls)
    if state.ask_hidden and not state.hidden_received:
        return _guard(_explore(state), state, max_calls)
    named_inspects = [
        _act("inspect_stock", product_id=pid, sku=sku)
        for pid, sku in state.named_variants
        if not state.inspect_is_fresh(state.skus.get(_sku_key(pid, sku), {}))
    ]
    if state.ranking_mode and named_inspects and not state.first_add_done:
        return _guard(named_inspects, state, max_calls)
    if state.justification_required and not state.viewed_ids:
        planned = actions or _explore(state)
        return _guard(planned, state, max_calls)
    ready = state.best_ready(must_verify=True)
    if ready and state.enough_survey() and ready["key"] not in state.cart_keys():
        if state.recovery_ladder:
            same = [
                rec
                for rec in state.skus.values()
                if rec["key"] != ready["key"] and rec.get("viewed") and state.same_model(rec)
            ]
            uninspected = [rec for rec in same if state.sku_ok(rec) and not state.inspect_is_fresh(rec)]
            if uninspected:
                return _guard(
                    [
                        _act("inspect_stock", product_id=rec["product_id"], sku=rec["sku"])
                        for rec in uninspected[:4]
                    ],
                    state,
                    max_calls,
                )
            if not same and state.search_turns < 6:
                return _guard(_explore(state), state, max_calls)
        if state.prefer_water:
            wet = [rec for rec in _verified_pool(state) if state.water_score(rec) > 0]
            more = [
                rec
                for rec in state.ranked_candidates()
                if rec.get("viewed")
                and state.water_score(rec) > 0
                and state.sku_ok(rec)
                and not state.inspect_is_fresh(rec)
            ]
            if more and len(wet) < 3:
                more.sort(
                    key=lambda rec: (
                        -state.water_score(rec),
                        -(state.current_price(rec) or 0),
                    )
                )
                return _guard(
                    [
                        _act("inspect_stock", product_id=rec["product_id"], sku=rec["sku"])
                        for rec in more[:8]
                    ],
                    state,
                    max_calls,
                )
            if len(wet) < 2 and state.search_turns < 4:
                return _guard(_explore(state), state, max_calls)
            return _guard(_commit_sequence(state, ready), state, max_calls)
        if state.ask_hidden and state.hidden_received:
            extras = [
                rec
                for rec in state.ranked_candidates()
                if rec["key"] != ready["key"]
                and rec.get("viewed")
                and state.sku_ok(rec)
                and not state.inspect_is_fresh(rec)
            ]
            if extras and len(_verified_pool(state)) < 2:
                return _guard(
                    [
                        _act("inspect_stock", product_id=rec["product_id"], sku=rec["sku"])
                        for rec in extras[:4]
                    ],
                    state,
                    max_calls,
                )
            if len(_verified_pool(state)) < 2 and state.search_turns < 6:
                return _guard(_explore(state), state, max_calls)
            return _guard(_commit_sequence(state, ready), state, max_calls)
        extras = [
            rec
            for rec in state.ranked_candidates()
            if rec["key"] != ready["key"]
            and rec.get("viewed")
            and state.sku_ok(rec)
            and not state.inspect_is_fresh(rec)
        ]
        if extras and len(_verified_pool(state)) < 2:
            return _guard(
                [
                    _act("inspect_stock", product_id=rec["product_id"], sku=rec["sku"])
                    for rec in extras[:4]
                ],
                state,
                max_calls,
            )
        if len(_verified_pool(state)) < 2 and state.search_turns < 5:
            return _guard(_explore(state), state, max_calls)
        return _guard(_commit_sequence(state, ready), state, max_calls)
    almost = state.best_ready(must_verify=False)
    if almost and almost.get("viewed") and not state.inspect_is_fresh(almost):
        return _guard(
            [_act("inspect_stock", product_id=almost["product_id"], sku=almost["sku"])],
            state,
            max_calls,
        )
    if almost and not almost.get("viewed"):
        return _guard([_act("view", product_id=almost["product_id"])], state, max_calls)
    if not actions:
        return _guard(_explore(state), state, max_calls)
    if state.first_add_prefers_brand and not state.first_add_done and state.brand_soft:
        if not any(
            _norm(rec.get("brand")) == _norm(state.brand_soft) for rec in state.skus.values()
        ) and state.search_turns < 3:
            return _guard(_explore(state), state, max_calls)
    if state.survey_required and not state.enough_survey():
        extra = _explore(state)
        merged = _dedupe(actions + extra)
        return _guard(merged, state, max_calls)
    return _guard(actions, state, max_calls)


def _collect_products(payload: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("product_id") and (
                node.get("sku")
                or node.get("variants")
                or node.get("facts")
                or node.get("title")
                or node.get("category_path")
            ):
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


def _ingest_observation(state: WorldState, action: dict[str, Any], wrap: dict[str, Any]) -> None:
    inner = wrap.get("observation") if isinstance(wrap, dict) else None
    if not isinstance(inner, dict):
        inner = wrap if isinstance(wrap, dict) else {}
    name = str(action.get("name") or inner.get("action") or "")
    if name == "search":
        state.search_turns += 1
    if name == "message":
        state.asked_hidden = True
        state.pending_question = False
    mutated_key = None
    event = inner.get("market_event") if isinstance(inner.get("market_event"), dict) else {}
    target = event.get("target") if isinstance(event.get("target"), dict) else {}
    if target.get("product_id") and target.get("sku"):
        mutated_key = _sku_key(target.get("product_id"), target.get("sku"))
    elif inner.get("added") and isinstance(inner.get("added"), dict):
        added = inner["added"]
        if added.get("product_id") and added.get("sku"):
            mutated_key = _sku_key(added.get("product_id"), added.get("sku"))
    if name in {"add_to_cart", "remove_from_cart"} or event:
        state.mark_mutation(mutated_key)
    if "cart" in inner:
        state.ingest_cart(inner.get("cart"))
        if inner.get("added"):
            state.first_add_done = True
            state.needs_event_ack = True
            added = inner["added"]
            if isinstance(added, dict):
                state.ingest_inspect(added)
        if name == "inspect_cart":
            state.needs_event_ack = False
            for row in state.cart:
                pid, sku = row.get("product_id"), row.get("sku")
                if pid and sku:
                    state.ingest_inspect(
                        {
                            "product_id": pid,
                            "sku": sku,
                            "price": row.get("current_price", row.get("price")),
                            "in_stock": row.get("in_stock"),
                        }
                    )
    if name == "inspect_stock" or (
        inner.get("product_id") and inner.get("sku") and "price" in inner and "variants" not in inner
    ):
        state.ingest_inspect(inner)
        if state.first_add_done and state.cart:
            pid = inner.get("product_id")
            sku = inner.get("sku")
            if pid and sku and _sku_key(pid, sku) in state.cart_keys():
                state.needs_event_ack = False
    viewed = name in {"view", "compare"} or "facts" in inner or "variants" in inner
    compared = name == "compare" or "products" in inner
    for item in _collect_products(inner):
        state.ingest_listing(item, viewed=viewed or "facts" in item, compared=compared)
    if name in {"search", "filter"}:
        results = [item for item in (inner.get("results") or []) if isinstance(item, dict)]
        for item in results:
            state.ingest_listing(item, viewed=False)
        if state.recovery_ladder and results:
            titles: dict[str, list[dict[str, Any]]] = {}
            stems: dict[str, list[dict[str, Any]]] = {}
            for item in results:
                title = _norm(item.get("title") or "")
                stem = _sku_stem(item.get("sku"))
                if title:
                    titles.setdefault(title, []).append(item)
                if stem:
                    stems.setdefault(stem, []).append(item)
            for title, items in titles.items():
                if len(items) >= 2 or (
                    state.prefer_model and any(_norm(state.prefer_model) in _norm(it.get("title") or "") for it in items)
                ):
                    state.model_title_aliases.add(title)
                    for item in items:
                        if item.get("sku"):
                            state.model_sku_stems.add(_sku_stem(item.get("sku")))
            for stem, items in stems.items():
                if len(items) >= 2:
                    state.model_sku_stems.add(stem)
                    for item in items:
                        if item.get("title"):
                            state.model_title_aliases.add(_norm(item.get("title")))


def _post(path: str, payload: dict[str, Any], attempts: int = 3) -> dict[str, Any] | None:
    last = None
    for attempt in range(attempts):
        last = _proxy.post(path, json_data=payload)
        if last is not None:
            return last
        time.sleep(min(2 ** attempt, 6))
    return last


def _plan_actions(
    messages: list[dict[str, Any]],
    policy: dict[str, Any],
    state: WorldState,
    max_calls: int,
) -> tuple[str, list[dict[str, Any]]]:
    if not state.llm_ok:
        return "", []
    primary, fallback = _model_pair()
    for model in (primary, fallback):
        inference = _post(
            "/inference/chat/completions",
            {
                "model": model,
                "messages": messages,
                "tools": policy.get("tools") or [],
                "tool_choice": "required",
                "temperature": 0,
            },
            attempts=1,
        )
        if not inference:
            continue
        try:
            assistant = (inference.get("choices") or [{}])[0].get("message") or {}
        except (IndexError, AttributeError, TypeError):
            continue
        content = assistant.get("content") or ""
        raw_calls = (assistant.get("tool_calls") or [])[:max_calls]
        actions = []
        for call in raw_calls:
            name = _tool_name(call)
            if not name:
                continue
            actions.append({"name": name, "args": _arguments(call)})
        if actions:
            return str(content), actions
    state.llm_ok = False
    return "", []


def _env_call(
    binding: dict[str, Any],
    problem_id: str,
    turn: int,
    nonce: int,
    actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    group_id = f"{problem_id}-t{turn}-n{nonce}"
    envelope = {
        **binding,
        "call_id": group_id,
        "idempotency_key": group_id,
        "turn": turn,
        "calls": [
            {
                "call_id": f"{group_id}-{index}",
                "action": {
                    "name": action["name"],
                    "args": action.get("args") or {},
                },
            }
            for index, action in enumerate(actions, start=1)
        ],
    }
    return _post("/environment/call", envelope, attempts=1)


def _is_done(result: dict[str, Any] | None) -> bool:
    if not result:
        return False
    for call in result.get("calls") or []:
        wrap = call.get("observation") if isinstance(call, dict) else None
        if isinstance(wrap, dict) and wrap.get("done") is True:
            return True
    return False


def _should_skip_llm(state: WorldState) -> bool:
    if not state.llm_ok:
        return True
    if state.ask_hidden and not state.asked_hidden:
        return True
    if state.named_variants and _unviewed_named(state):
        return True
    if state.best_ready(must_verify=True) and state.enough_survey():
        return True
    almost = state.best_ready(must_verify=False)
    if almost and almost.get("viewed") and not state.inspect_is_fresh(almost):
        return True
    if state.prefer_water or state.recovery_ladder:
        viewed = [rec for rec in state.skus.values() if rec.get("viewed")]
        return len(viewed) < 2
    return True


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    dialogue: list[dict[str, Any]] = []
    try:
        environment = problem_data.get("environment") or {}
        binding = dict(environment.get("binding") or {})
        policy = environment.get("policy_view") or {}
        query = str(policy.get("query") or problem_data.get("query") or "")
        problem_id = str(problem_data.get("problem_id") or problem_data.get("id") or "problem")
        max_steps = int(policy.get("max_steps") or 30)
        max_calls = int(policy.get("max_calls_per_turn") or 1)
        tool_names = _policy_tools(policy)
        state = WorldState(query, tool_names)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system_prompt(state)},
            {"role": "user", "content": query},
        ]
        turn = 1
        fail_streak = 0
        while turn <= max_steps:
            state.turn = turn
            messages[0] = {"role": "system", "content": _system_prompt(state)}
            content = ""
            planned: list[dict[str, Any]] = []
            if not _should_skip_llm(state):
                content, planned = _plan_actions(messages, policy, state, max_calls)
            actions = _close_if_ready(planned, state, max_calls)
            if not actions:
                actions = _guard(_explore(state), state, max_calls)
            if not actions:
                break
            state.env_nonce += 1
            synthetic = []
            for index, action in enumerate(actions, start=1):
                synthetic.append(
                    {
                        "id": f"{problem_id}-synth-{turn}-{state.env_nonce}-{index}",
                        "type": "function",
                        "function": {
                            "name": action["name"],
                            "arguments": json.dumps(action.get("args") or {}),
                        },
                    }
                )
            messages.append(
                {"role": "assistant", "content": content, "tool_calls": synthetic}
            )
            result = _env_call(binding, problem_id, turn, state.env_nonce, actions)
            dialogue.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": synthetic,
                    "environment_result": result,
                }
            )
            if not result:
                fail_streak += 1
                if fail_streak >= 2:
                    break
                time.sleep(0.4)
                continue
            wraps = result.get("calls") or []
            for tool_call, action, call_result in zip(synthetic, actions, wraps):
                wrap = call_result.get("observation") if isinstance(call_result, dict) else {}
                if not isinstance(wrap, dict):
                    wrap = {}
                _ingest_observation(state, action, wrap)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(wrap),
                    }
                )
            user_message = result.get("user_message")
            text = ""
            if isinstance(user_message, dict):
                text = str(user_message.get("content") or "")
            elif isinstance(user_message, str):
                text = user_message
            if text.strip():
                state.absorb_shopper(text)
                messages.append({"role": "user", "content": text})
            fail_streak = 0
            turn += 1
            if _is_done(result):
                break
    except Exception as exc:  # noqa: BLE001 - never fail an episode
        dialogue.append({"role": "assistant", "content": f"agent_recovered:{type(exc).__name__}"})
    return dialogue


__all__ = ["agent_main"]
