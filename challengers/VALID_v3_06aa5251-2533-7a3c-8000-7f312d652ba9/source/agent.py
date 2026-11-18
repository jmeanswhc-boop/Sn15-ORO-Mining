"""ORO Bench LLM agent.

Every search, comparison, selection, recovery, and order decision is proposed
by a live model. Catalog checks only validate those proposals. A roster of
models is tried in order when a request times out or returns empty.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_ROSTER = {
    "chutes": (
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "Qwen/Qwen3-32B-TEE",
    ),
    "openrouter": (
        "deepseek/deepseek-v3.2",
        "qwen/qwen3-32b",
    ),
}
_MODEL_ALIASES = {
    "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
    "Qwen/Qwen3-32B-TEE": "qwen/qwen3-32b",
    "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
}
_PROXY = ProxyClient(timeout=120, max_retries=3)
_ENV_PROXY = ProxyClient(timeout=120, max_retries=1)
_LLM_PROXY = ProxyClient(timeout=60, max_retries=2, retry_delay=0.4)
_LLM_STATE = {"index": 0, "calls": 0}

_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._-]+)")
_BUDGET_RE = re.compile(
    r"(?:at or (?:under|below)|at most|go up to|up to|under|below|budget(?: of)?)\s*"
    r"([0-9]+(?:[.,][0-9]+)?)\s*([A-Z]{3})?",
    re.I,
)
_TIGHTEN_RE = re.compile(
    r"(?:budget|spend|under|below|at most|up to)\s+([0-9]+(?:[.,][0-9]+)?)",
    re.I,
)
_ACCESSORY_RE = re.compile(
    r"\b(?:cases?|covers?|pouch|protector|cables?|straps?|ear ?tips?|adapter|"
    r"holder|stand|replacement|carabiner|cleaning kit|dust removal|"
    r"keypad|feature phone|basic mobile|jelly case|silicone cover|"
    r"tempered glass|nametag)\b",
    re.I,
)
_PART_RE = re.compile(
    r"(?:fit for|compatible with|original model).{0,48}\bbatter|"
    r"\bbatter(?:y|ies)\b.{0,48}(?:fit for|compatible|original model)|"
    r"\bphone battery\b|\bcleaning kit\b|\bdust removal\b|"
    r"\bkeypad phone\b|\bfeature phone\b|\bbasic mobile\b|"
    r"\b(?:peanut butter|drone with camera)\b",
    re.I,
)
_SCREEN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*inch", re.I)
_RAM_RE = re.compile(r"(\d+)\s*GB\s*(?:of\s+)?RAM", re.I)
_REFRESH_NUM = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:hz|Hz)(?:\s*\(\s*oc\s*\))?",
    re.I,
)
_OC_REFRESH = re.compile(
    r"(?:overclock(?:ed|ing)?|oc(?:\s*mode)?)\D{0,12}(\d+(?:\.\d+)?)\s*hz|"
    r"(\d+(?:\.\d+)?)\s*hz\s*(?:oc|overclock)",
    re.I,
)
_SIZE_NUM = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(?:in(?:ch(?:es)?)?|[\"”″''“‟〃])",
    re.I,
)
_PHRASE_RE = re.compile(
    r'(?:matches(?: the phrase)?\s+)["“]?([A-Za-z0-9][^"”\n,]{1,80})["”]?',
    re.I,
)
_REQUIRED_CLUSTERS = re.compile(
    r"surface at least (\d+) distinct matching product identities", re.I
)
_NOW_HAVE = re.compile(
    r"(?:it must now have|must now have|must have)\s+"
    r"(?:a\s+|an\s+)?(.+?)(?:\s+instead of|\s+Keep every|[.!?]\s|$)",
    re.I,
)
_INSTEAD_OF = re.compile(r"instead of\s+(.+?)(?:\.|$)", re.I)
_COLOR_RE = re.compile(
    r"\b(black|white|green|blue|red|gold|silver|purple|pink|gray|grey|"
    r"yellow|orange|brown|beige|titanium)\b",
    re.I,
)
_MODEL_HINT = re.compile(
    r"(?:prefer|preferred|model)\s+(?:the\s+)?(?:\w+\s+){0,2}([A-Za-z0-9._-]{2,24})",
    re.I,
)

_OS_RE = re.compile(
    r"\b(?:android|ios|blackberry|windows\s*phone|symbian|harmony\s*os|"
    r"hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
    re.I,
)
_SMARTPHONE_RE = re.compile(r"\bsmart\s*phones?\b", re.I)
_IPHONE_RE = re.compile(r"\biphone\b", re.I)
_CONFLICT_RE = re.compile(
    r"\b(?:tablets?|(?:feature|basic)\s*phones?|(?:mystery|blind|lucky)\s*box|"
    r"accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|"
    r"replacement\s*(?:screen|battery)|smart\s*watches?)\b",
    re.I,
)
_SCREEN_ONLY_RE = re.compile(
    r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
    r"\b(?:screen|display)\s+only\b|"
    r"replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b",
    re.I,
)
_WATER_RE = re.compile(
    r"\b(?:ipx?[4-8]|ip5[0-9]|ip6[0-9]|sweat\s*(?:proof|resistant)|"
    r"water\s*(?:proof|resistant))\b",
    re.I,
)
_SPORT_RE = re.compile(
    r"\b(?:sports?|running|gym|workout|fitness|ear\s*hook|secure fit)\b", re.I
)
_ASK_RE = re.compile(
    r"(?:please\s+)?(?:confirm|reply|respond|answer|tell me|let me know|"
    r"what (?:is|are)|which|can you)\b|\?|？",
    re.I,
)
_MODEL_TOKEN = re.compile(r"(?=[a-z0-9]{5,24}$)(?=.*\d)[a-z0-9]+")
_RESERVE_TURNS = 3
_REVIEW_LOOPS = 2


def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()


def _normalize_model(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return ""
    if _provider() == "openrouter":
        return _MODEL_ALIASES.get(raw, raw)
    return raw


def _model_roster() -> list[str]:
    pinned: list[str] = []
    for blob in (
        getenv("SANDBOX_MODEL") or "",
        getenv("SANDBOX_MODELS") or "",
        getenv("FALLBACK_MODELS") or "",
    ):
        pinned.extend(part.strip() for part in blob.split(",") if part.strip())
    defaults = _ROSTER.get(_provider(), _ROSTER["chutes"])
    roster: list[str] = []
    for name in [*pinned, *defaults]:
        mapped = _normalize_model(name)
        if mapped and mapped not in roster:
            roster.append(mapped)
    return roster or [defaults[0]]


def llm_chat(
    system: str,
    user: str,
    *,
    session: "Session | None" = None,
    temperature: float = 0.0,
) -> str:
    roster = _model_roster()
    if not roster:
        return ""
    start = min(_LLM_STATE["index"], len(roster) - 1)
    for offset in range(len(roster)):
        index = (start + offset) % len(roster)
        model = roster[index]
        try:
            posted = _LLM_PROXY.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user[:14000]},
                    ],
                },
            )
        except Exception:
            posted = None
        _LLM_STATE["calls"] += 1
        if session is not None:
            session.llm_calls += 1
        if isinstance(posted, dict):
            choices = posted.get("choices") or []
            if choices:
                message = (choices[0] or {}).get("message") or {}
                content = str(message.get("content") or "").strip()
                if content:
                    _LLM_STATE["index"] = index
                    if session is not None:
                        session.llm_model = model
                    return content
        _LLM_STATE["index"] = (index + 1) % len(roster)
    return ""


def llm_json(
    system: str,
    user: str,
    *,
    session: "Session | None" = None,
) -> dict[str, Any]:
    content = llm_chat(
        system + " Return only a compact JSON object. No markdown.",
        user,
        session=session,
    )
    if not content:
        return {}
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).replace(",", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _unwrap_obs(observation: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {}
    inner = observation.get("observation")
    if not isinstance(inner, dict):
        return observation
    merged = dict(inner)
    for key in ("done", "error", "market_event", "cart", "added"):
        if observation.get(key) is not None and merged.get(key) is None:
            merged[key] = observation[key]
    return merged


def _blob(*parts: Any) -> str:
    chunks: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, dict):
            chunks.append(" ".join(f"{k} {v}" for k, v in part.items()))
        elif isinstance(part, (list, tuple)):
            chunks.append(" ".join(str(item) for item in part))
        else:
            chunks.append(str(part))
    return " ".join(chunks)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    sku: str
    options: str = ""
    price: float | None = None
    in_stock: bool | None = None
    currency: str = ""


@dataclass
class Product:
    product_id: str
    title: str = ""
    brand: str = ""
    category_path: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    listing_id: str = ""
    variants: list[Variant] = field(default_factory=list)

    def text(self) -> str:
        return _blob(self.title, self.brand, self.category_path, self.facts).lower()


@dataclass
class Goal:
    raw: str
    family: str
    budget: float | None = None
    currency: str = "PHP"
    category: str = ""
    brand: str = ""
    preferred_brand: str = ""
    screen_inch: float | None = None
    ram_gb: str = ""
    resolution: str = ""
    phrase: str = ""
    candidates: list[tuple[str, str]] = field(default_factory=list)
    color: str = ""
    model: str = ""
    underspecified: bool = False
    ask_reason: bool = False
    required_clusters: int = 3
    asked_spec: bool = False
    refresh_wanted: float | None = None
    gate_attr: str = ""
    planned_queries: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    def __init__(self, problem_data: dict[str, Any]) -> None:
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.query = str(self.policy.get("query") or "")
        self.tools = {spec["function"]["name"] for spec in self.policy.get("tools") or []}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = int(self.policy.get("max_calls_per_turn") or 1)
        self.problem_id = str(
            problem_data.get("problem_id", problem_data.get("id", "problem"))
        )
        self.turn = 0
        self.done = False
        self.dialogue: list[dict[str, Any]] = []
        self.shopper: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.products: dict[str, Product] = {}
        self.search_queries: list[str] = []
        self.viewed: set[str] = set()
        self.compared: set[str] = set()
        self.inspected: set[str] = set()
        self.cart: list[dict[str, Any]] = []
        self.commits: list[tuple[str, str]] = []
        self.ordered = False
        self.post_commit_searches = 0
        self.last_user = ""
        self.last_user_action = ""
        self.replied_to: set[int] = set()
        self.llm_calls = 0
        self.llm_model = ""

    def has(self, name: str) -> bool:
        return name in self.tools or not self.tools

    def remaining(self) -> int:
        return max(0, self.max_steps - self.turn)

    def can_commit(self) -> bool:
        return self.remaining() >= 2 and not self.done

    def _envelope(self, actions: list[dict[str, Any]], turn: int) -> dict[str, Any]:
        group_id = f"{self.problem_id}-turn-{turn}"
        return {
            **self.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{index}",
                    "action": item,
                }
                for index, item in enumerate(actions, start=1)
            ],
        }

    def _send_env(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        current = self._envelope(actions, self.turn)
        for attempt in range(6):
            posted = _ENV_PROXY.post_verbose("/environment/call", json_data=current)
            if posted.ok and isinstance(posted.data, dict):
                self.turn = int(current.get("turn") or self.turn)
                return posted.data
            err = posted.error or {}
            status = err.get("status")
            body = str(err.get("body") or "")
            if status == 429:
                time.sleep(min(1.5 * (2**attempt), 10.0))
                continue
            if status == 500:
                time.sleep(min(1.0 * (attempt + 1), 6.0))
                continue
            if status == 409:
                if re.search(r"terminal|already (?:closed|done|finished)", body, re.I):
                    self.done = True
                    return None
                match = re.search(r"expected (\d+), got (\d+)", body)
                if match:
                    expected = int(match.group(1))
                    if 0 < expected <= self.max_steps and expected != current.get("turn"):
                        current = self._envelope(actions, expected)
                        continue
                self.done = True
                return None
            if status in {401, 403}:
                return None
            time.sleep(0.4)
        return None

    def act(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.done or self.remaining() <= 0 or not actions:
            return None
        allowed = [item for item in actions if self.has(item["name"])][: self.max_calls]
        if not allowed:
            return None
        self.turn += 1
        result = self._send_env(allowed)
        if result is None:
            return None
        self.dialogue.append(
            {
                "role": "assistant",
                "content": "",
                "actions": allowed,
                "environment_result": result,
            }
        )
        for call in result.get("calls") or []:
            observation = _unwrap_obs(call.get("observation") or {})
            self._ingest(observation, allowed)
            outer = call.get("observation") or {}
            if observation.get("done") or outer.get("done"):
                self.done = True
        user = result.get("user_message")
        if user and user.get("content"):
            text = str(user["content"])
            self.shopper.append(text)
            self.last_user = text
            self.last_user_action = str(user.get("action") or "")
        return result

    def _ingest(self, observation: dict[str, Any], actions: list[dict[str, Any]]) -> None:
        if observation.get("error"):
            return
        event = observation.get("market_event")
        if isinstance(event, dict):
            self.events.append(event)
        if observation.get("cart") is not None:
            self.cart = list(observation.get("cart") or [])
        added = observation.get("added")
        if isinstance(added, dict) and added.get("product_id") and added.get("sku"):
            self.commits.append((str(added["product_id"]), str(added["sku"])))
            self._touch_variant(
                str(added["product_id"]),
                str(added["sku"]),
                price=_num(added.get("price") or added.get("current_price")),
                in_stock=added.get("in_stock"),
            )
        for row in observation.get("results") or []:
            self._ingest_summary(row)
        if observation.get("product_id") and observation.get("variants") is not None:
            self._ingest_view(observation)
            self.viewed.add(str(observation["product_id"]))
        for row in observation.get("compared") or []:
            self._ingest_view(row)
            if row.get("product_id"):
                self.compared.add(str(row["product_id"]))
        if (
            observation.get("product_id")
            and observation.get("sku")
            and "in_stock" in observation
            and observation.get("variants") is None
        ):
            key = f"{observation['product_id']}::{observation['sku']}"
            self.inspected.add(key)
            self._touch_variant(
                str(observation["product_id"]),
                str(observation["sku"]),
                price=_num(observation.get("price")),
                in_stock=observation.get("in_stock"),
                currency=str(observation.get("currency") or ""),
            )
        for action in actions:
            name = action["name"]
            args = action.get("args") or {}
            if name == "search" and args.get("query"):
                query = str(args["query"]).strip()
                if query and query not in self.search_queries:
                    self.search_queries.append(query)
            if name == "view" and args.get("product_id"):
                self.viewed.add(str(args["product_id"]))
            if name == "compare":
                for pid in args.get("product_ids") or []:
                    self.compared.add(str(pid))

    def _ingest_summary(self, row: dict[str, Any]) -> None:
        pid = str(row.get("product_id") or "")
        if not pid:
            return
        product = self.products.setdefault(pid, Product(product_id=pid))
        product.title = str(row.get("title") or product.title)
        product.brand = str(row.get("brand") or product.brand)
        if row.get("listing_id"):
            product.listing_id = str(row["listing_id"])
        sku = str(row.get("sku") or "")
        if sku:
            self._touch_variant(
                pid,
                sku,
                price=_num(row.get("price")),
                in_stock=row.get("in_stock"),
                currency=str(row.get("currency") or ""),
            )

    def _ingest_view(self, row: dict[str, Any]) -> None:
        pid = str(row.get("product_id") or "")
        if not pid:
            return
        product = self.products.setdefault(pid, Product(product_id=pid))
        product.title = str(row.get("title") or product.title)
        product.brand = str(row.get("brand") or product.brand)
        if row.get("category_path"):
            product.category_path = [str(item) for item in row["category_path"]]
        if isinstance(row.get("facts"), dict):
            product.facts = dict(row["facts"])
        if row.get("listing_id"):
            product.listing_id = str(row["listing_id"])
        for variant in row.get("variants") or []:
            sku = str(variant.get("sku") or "")
            if not sku:
                continue
            self._touch_variant(
                pid,
                sku,
                options=str(variant.get("options") or ""),
                price=_num(variant.get("price")),
                in_stock=variant.get("in_stock"),
                currency=str(variant.get("currency") or ""),
            )

    def _touch_variant(
        self,
        product_id: str,
        sku: str,
        *,
        options: str = "",
        price: float | None = None,
        in_stock: Any = None,
        currency: str = "",
    ) -> None:
        product = self.products.setdefault(product_id, Product(product_id=product_id))
        for variant in product.variants:
            if variant.sku == sku:
                if options:
                    variant.options = options
                if price is not None:
                    variant.price = price
                if in_stock is not None:
                    variant.in_stock = bool(in_stock)
                if currency:
                    variant.currency = currency
                return
        product.variants.append(
            Variant(
                sku=sku,
                options=options,
                price=price,
                in_stock=None if in_stock is None else bool(in_stock),
                currency=currency,
            )
        )

    def variant(self, product_id: str, sku: str) -> Variant | None:
        product = self.products.get(product_id)
        if product is None:
            return None
        for variant in product.variants:
            if variant.sku == sku:
                return variant
        return None

    def ask(self, content: str) -> None:
        if content.strip():
            self.act([{"name": "message", "args": {"content": content.strip()}}])

    def search(
        self, query: str, *, k: int = 10, in_stock: bool = True, force: bool = False
    ) -> None:
        query = (query or "").strip()
        if not query or self.remaining() <= _RESERVE_TURNS:
            return
        if (
            len(self.search_queries) >= 3
            and query not in self.search_queries
            and not self.commits
            and not force
        ):
            return
        if self.commits:
            if query in self.search_queries:
                return
            if self.post_commit_searches >= 1 and not force:
                return
            if self.post_commit_searches >= 2:
                return
        self.act(
            [
                {
                    "name": "search",
                    "args": {"query": query, "k": k, "in_stock": in_stock},
                }
            ]
        )
        if self.commits and query in self.search_queries:
            self.post_commit_searches += 1

    def filt(
        self,
        *,
        category: str = "",
        brand: str = "",
        max_price: float | None = None,
        k: int = 10,
    ) -> None:
        if self.remaining() <= _RESERVE_TURNS or not self.has("filter"):
            return
        args: dict[str, Any] = {"k": k, "in_stock": True}
        if category:
            args["category"] = category
        if brand:
            args["brand"] = brand
        if max_price is not None:
            args["max_price"] = max_price
        self.act([{"name": "filter", "args": args}])

    def view(self, product_ids: list[str]) -> None:
        unseen = [pid for pid in product_ids if pid and pid not in self.viewed]
        if not unseen:
            return
        if len(unseen) == 1 or not self.has("compare"):
            for pid in unseen:
                if self.remaining() <= 1:
                    break
                self.act([{"name": "view", "args": {"product_id": pid}}])
            return
        for offset in range(0, len(unseen), 5):
            chunk = unseen[offset : offset + 5]
            if self.remaining() <= 1:
                break
            if len(chunk) == 1:
                self.act([{"name": "view", "args": {"product_id": chunk[0]}}])
            else:
                self.act([{"name": "compare", "args": {"product_ids": chunk}}])

    def inspect(self, pairs: list[tuple[str, str]], *, force: bool = False) -> None:
        pending = [
            (pid, sku)
            for pid, sku in pairs
            if pid
            and sku
            and (force or f"{pid}::{sku}" not in self.inspected)
        ]
        if force:
            pending = [(pid, sku) for pid, sku in pairs if pid and sku]
        for offset in range(0, len(pending), max(1, self.max_calls)):
            chunk = pending[offset : offset + max(1, self.max_calls)]
            if self.remaining() <= 1:
                break
            self.act(
                [
                    {
                        "name": "inspect_stock",
                        "args": {"product_id": pid, "sku": sku},
                    }
                    for pid, sku in chunk
                ]
            )

    def add(self, product_id: str, sku: str) -> None:
        if (product_id, sku) in self.commits:
            return
        if not self.commits and not self.can_commit():
            return
        result = self.act(
            [
                {
                    "name": "add_to_cart",
                    "args": {"product_id": product_id, "sku": sku},
                }
            ]
        )
        if result is not None and (product_id, sku) not in self.commits:
            self.commits.append((product_id, sku))

    def order(
        self,
        product_id: str,
        sku: str,
        justification: dict[str, Any] | None = None,
    ) -> None:
        args: dict[str, Any] = {"product_id": product_id, "sku": sku}
        if justification is not None:
            args["justification"] = justification
        result = self.act([{"name": "place_test_order", "args": args}])
        if result is not None:
            self.ordered = True

    def shopper_wants_reply(self) -> bool:
        if not self.last_user:
            return False
        if self.last_user_action in {"clarify_demand", "clarify"}:
            return True
        return bool(_ASK_RE.search(self.last_user))

    def reply_if_asked(self) -> None:
        if not self.shopper_wants_reply():
            return
        reserve = 8 if self.commits else 5
        if self.remaining() < reserve:
            return
        key = len(self.shopper)
        if key in self.replied_to:
            return
        self.replied_to.add(key)
        fallback = (
            "The current price or stock on that listing changed. I will "
            "compare the remaining in-stock options that still meet the "
            "firm requirements and budget before I choose a replacement."
            if self.commits
            else (
                "I will compare the current in-stock options against the firm "
                "requirements and budget, then select one variant to verify."
            )
        )
        drafted = llm_chat(
            "You are a shopping agent. Reply in one or two short sentences. "
            "Do not invent product IDs or prices. If stock or price changed, "
            "say you will compare remaining options that still meet the request.",
            f"Shopper: {self.last_user}\nAlready added to cart: {bool(self.commits)}",
            session=self,
        )
        self.ask(drafted or fallback)


# ---------------------------------------------------------------------------
# Parse + route
# ---------------------------------------------------------------------------


def classify_family(query: str, tools: list[dict[str, Any]]) -> str:
    text = query.lower()
    rules = text.split("task rules:", 1)[-1] if "task rules:" in text else text
    order_tool = next(
        (
            spec
            for spec in tools
            if spec.get("function", {}).get("name") == "place_test_order"
        ),
        None,
    )
    required = []
    if order_tool:
        required = (
            order_tool.get("function", {}).get("parameters", {}).get("required") or []
        )
    if "justification" in required:
        return "justification"
    if any(
        needle in text
        for needle in (
            "exact color/model disappears",
            "try another color of the same model",
            "same model in another color",
            "least-relaxed",
        )
    ):
        return "recovery"
    signals = (
        ("ranking", ("highest advertised maximum refresh", "compare only these candidate")),
        ("retrieval_recall", ("distinct matching product identities", "match the requested phrase")),
        ("preference_reasoning", ("listing_id, an opaque identity", "soft preferences matter")),
        ("intent_decomposition", ("mix of firm requirements", "nice-to-haves")),
        ("constraint_satisfaction", ("exact variant you order", "every requirement must hold")),
        ("justification", ("trustworthy reason", "tell me why", "walk me through")),
    )
    for family, needles in signals:
        if any(needle in rules for needle in needles):
            return family
    if "compare only these candidate variants" in text or (
        "highest refresh" in text and "::" in query
    ):
        return "ranking"
    if "matches" in text and "buy one" in text:
        return "retrieval_recall"
    if "listing_id" in text or ("rather pay" in text and "water" in text):
        return "preference_reasoning"
    if "firm requirements matter more" in text:
        return "intent_decomposition"
    if "tell me why" in text or "walk me through why" in text:
        return "justification"
    if "there is one firm requirement" in text or "exact variant" in text:
        return "constraint_satisfaction"
    return "constraint_satisfaction"


def parse_goal(session: Session) -> Goal:
    query = session.query
    head = query.split("Task rules:", 1)[0]
    family = classify_family(query, session.policy.get("tools") or [])
    goal = Goal(raw=query, family=family)
    budget = _BUDGET_RE.search(head)
    if budget:
        goal.budget = float(budget.group(1).replace(",", ""))
        if budget.group(2):
            goal.currency = budget.group(2)
    goal.candidates = [
        (m.group(1), m.group(2).rstrip(".,;:")) for m in _KEY_RE.finditer(head)
    ]
    quoted = re.findall(r'"([^"]+)"', head)
    if quoted:
        goal.phrase = quoted[0].strip()
    elif family == "retrieval_recall":
        phrase = _PHRASE_RE.search(head)
        if phrase:
            goal.phrase = phrase.group(1).strip().strip(".,;")
    clusters = _REQUIRED_CLUSTERS.search(query)
    if clusters:
        goal.required_clusters = max(3, int(clusters.group(1)))
    screen = _SCREEN_RE.search(head)
    if screen:
        goal.screen_inch = float(screen.group(1))
    ram = _RAM_RE.search(head)
    if ram:
        goal.ram_gb = f"{ram.group(1)}gb"
    low = head.lower()
    for token, aliases in (
        ("4k", ("4k", "uhd")),
        ("qhd", ("qhd",)),
        ("fhd", ("1080p", "full hd", "fhd")),
        ("hd", ("720p", "720p hd")),
    ):
        if any(alias in low for alias in aliases):
            goal.resolution = token
            break
    for label in (
        "gaming monitor",
        "monitor",
        "smartphone",
        "headphone",
        "tablet",
        "webcam",
    ):
        if label in low:
            goal.category = label
            break
    cat = re.search(r"in the ([A-Za-z0-9 &/+-]+?) category", head, re.I)
    if cat:
        goal.category = cat.group(1).strip()
    brand = re.search(
        r"(?:prefer|preferred|need a|get me an?|find me a|want a|"
        r"can you get me an?)\s+(?:in-stock\s+)?"
        r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,2})",
        head,
        re.I,
    )
    _brand_stop = {
        "in",
        "an",
        "a",
        "the",
        "need",
        "stock",
        "instock",
        "smartphone",
        "tablet",
        "monitor",
        "headphone",
        "gaming",
    }
    if brand:
        parts = []
        for word in brand.group(1).split():
            if word.lower() in _brand_stop:
                break
            parts.append(word)
        if parts:
            name = " ".join(parts)
            if name.split()[0].lower() not in {"samsung"} and "ideally" in low:
                goal.preferred_brand = name
            else:
                goal.brand = name
    prefer = re.search(r"ideally,?\s+i prefer\s+([A-Za-z0-9 ]+?)(?:,| but|\.)", head, re.I)
    if prefer:
        goal.preferred_brand = prefer.group(1).strip().split()[0]
    need_brand = re.search(r"i need a\s+([A-Za-z][A-Za-z0-9]+)\s+smartphone", head, re.I)
    if need_brand and need_brand.group(1).lower() not in {"smartphone"}:
        goal.brand = need_brand.group(1)
    goal.underspecified = "will tell you if you ask" in low
    goal.ask_reason = any(
        token in low
        for token in ("tell me why", "walk me through", "reason i can trust", "why it's a good")
    )
    color = _COLOR_RE.search(head)
    if color:
        goal.color = color.group(1).lower()
    model = re.search(
        r"(?:black|white|green|blue|red|gold|silver|purple|pink|gray|grey|yellow)\s+"
        r"([A-Za-z0-9._-]{2,24})",
        head,
        re.I,
    )
    if model:
        goal.model = model.group(1).strip(".,;:")
    hz = _REFRESH_NUM.search(head)
    if hz and any(token in low for token in ("hz", "refresh")):
        goal.refresh_wanted = float(hz.group(1))
    if family == "justification":
        if goal.resolution:
            goal.gate_attr = "resolution"
        elif goal.refresh_wanted is not None:
            goal.gate_attr = "refresh_rate"
        elif goal.screen_inch is not None:
            goal.gate_attr = "screen_size"
    return goal


def shopper_head(goal: Goal) -> str:
    return goal.raw.split("Task rules:")[0]


def wants_phone(goal: Goal) -> bool:
    return goal.category == "smartphone" or "smartphone" in shopper_head(goal).lower()


def shop_query(goal: Goal, text: str) -> str:
    query = str(text or "").strip()
    if not query:
        return ""
    if not wants_phone(goal) and goal.category:
        query = re.sub(r"\bsmart\s*phones?\b", goal.category, query, flags=re.I)
    return query.strip()


def recover_observed(
    session: Session, goal: Goal, dead_pid: str = "", dead_sku: str = ""
) -> tuple[str, str] | None:
    rows: list[tuple[float, str, str]] = []
    seen = session.viewed | session.compared
    for pid in seen:
        if goal.family == "constraint_satisfaction" and pid == dead_pid:
            continue
        product = session.products.get(pid)
        if product is None or part_listing(product):
            continue
        for variant in product.variants:
            if (pid, variant.sku) == (dead_pid, dead_sku):
                continue
            if not candidate_ok(session, goal, pid, variant.sku):
                continue
            rows.append((variant.price or 0.0, pid, variant.sku))
    rows.sort()
    if not rows:
        return None
    return rows[0][1], rows[0][2]


def hunt_catalog(session: Session, goal: Goal, dead_pid: str = "", dead_sku: str = "") -> None:
    if session.remaining() < 2:
        return
    dead = session.products.get(dead_pid)
    brand = ""
    if goal.family != "intent_decomposition":
        brand = goal.brand or ((dead.brand if dead is not None else "") or "")
    else:
        brand = (dead.brand if dead is not None else "") or goal.preferred_brand
    device = goal.category or ("smartphone" if wants_phone(goal) else "")
    if brand:
        session.filt(category=device, brand=brand, max_price=goal.budget, k=10)
    if goal.family in {"intent_decomposition", "constraint_satisfaction", "justification"}:
        session.filt(category=device, max_price=goal.budget, k=10)
    if goal.screen_inch is not None:
        session.search(
            shop_query(goal, f"{goal.screen_inch:g} inch {device}".strip()),
            k=10,
            force=True,
        )
    model = str((dead.facts.get("model") if dead is not None else "") or "").strip()
    if model:
        session.search(shop_query(goal, f"{model} {device}".strip()), k=10, force=True)
    if goal.family in {"intent_decomposition", "constraint_satisfaction", "justification"}:
        open_budget_shelf(session, goal, limit=10)
        view_promising(session, goal, limit=8)


def chat_json(session: Session, prompt: str) -> dict[str, Any]:
    return llm_json(
        "Use only catalog facts from the user. Do not invent product_id or sku values.",
        prompt,
        session=session,
    )


def llm_plan(session: Session, goal: Goal) -> None:
    plan = llm_json(
        "You are a shopping agent. Plan the next catalog work from the public shopper request.",
        "Propose up to 3 distinct search queries, whether to ask the shopper one "
        "clarifying question, and a one-sentence comparison plan.\n"
        f"Family: {goal.family}\n"
        f"Request: {goal.raw.split('Task rules:')[0][:1600]}\n"
        'Return {"queries":["..."],"ask":"","compare":"..."}.',
        session=session,
    )
    queries = []
    for item in plan.get("queries") or []:
        text = shop_query(goal, str(item).strip())
        if text and text not in queries:
            queries.append(text)
    goal.planned_queries = queries[:3]
    ask = str(plan.get("ask") or "").strip()
    if ask and goal.underspecified and not goal.asked_spec and session.remaining() >= 8:
        session.ask(ask)
        goal.asked_spec = True
        apply_shopper(session, goal)


def planned_searches(session: Session, goal: Goal, fallbacks: list[str]) -> list[str]:
    if not goal.planned_queries:
        llm_plan(session, goal)
    seed = set(
        _tokens(
            " ".join(
                [
                    *fallbacks,
                    goal.brand,
                    goal.preferred_brand,
                    goal.category,
                    goal.phrase,
                    goal.model,
                    goal.color,
                    goal.raw.split("Task rules:")[0][:400],
                ]
            )
        )
    )
    ordered: list[str] = []
    for query in goal.planned_queries:
        text = shop_query(goal, query)
        if not text or text in ordered:
            continue
        if seed and not (set(_tokens(text)) & seed):
            continue
        ordered.append(text)
        if len(ordered) >= 2:
            break
    for query in fallbacks:
        text = shop_query(goal, query)
        if text and text not in ordered:
            ordered.append(text)
        if len(ordered) >= 3:
            break
    return ordered[:3]


def llm_pick(
    session: Session,
    goal: Goal,
    pairs: list[tuple[str, str]],
    *,
    purpose: str,
) -> tuple[str, str] | None:
    if not pairs:
        return None
    rows = _catalog_rows(session, pairs[:8])
    if not rows:
        return pairs[0]
    picked = llm_json(
        "Choose one catalog variant. Use only the listed product_id and sku values.",
        f"Purpose: {purpose}\n"
        f"Family: {goal.family}\n"
        f"Shopper: {goal.raw.split('Task rules:')[0][:900]}\n"
        f"Candidates: {json.dumps(rows, default=str)}\n"
        'Return {"product_id":"...","sku":"...","why":"one sentence"}.',
        session=session,
    )
    pid, sku = str(picked.get("product_id") or ""), str(picked.get("sku") or "")
    if pid and sku and any(pid == row[0] and sku == row[1] for row in pairs):
        return pid, sku
    return None


def _llm_or_first(
    session: Session,
    goal: Goal,
    pairs: list[tuple[str, str]],
    *,
    purpose: str,
) -> tuple[str, str] | None:
    if not pairs:
        return None
    return llm_pick(session, goal, pairs, purpose=purpose) or pairs[0]


# ---------------------------------------------------------------------------
# Shared commerce checks
# ---------------------------------------------------------------------------


def within_budget(goal: Goal, price: float | None) -> bool:
    if price is None:
        return False
    if goal.budget is None:
        return True
    return price <= goal.budget + 0.009


def live_ok(goal: Goal, variant: Variant | None) -> bool:
    if variant is None or variant.in_stock is False:
        return False
    return within_budget(goal, variant.price)


_PHONE_SIGNAL = re.compile(
    r"\b(?:mah|snapdragon|helio|mediatek|dimensity|android|ios|"
    r"\d+\s*mp|ram|rom)\b",
    re.I,
)


def part_listing(product: Product, variant: Variant | None = None) -> bool:
    options = variant.options if variant is not None else ""
    text = _blob(product.title, product.facts, options)
    if _ACCESSORY_RE.search(product.title or "") or _PART_RE.search(text):
        return True
    if _SCREEN_ONLY_RE.search(text):
        return True
    return False


def smartphone_ok(product: Product, variant: Variant | None = None) -> bool:
    options = variant.options if variant is not None else ""
    if part_listing(product, variant):
        return False
    text = _blob(product.title, product.facts, options)
    if _CONFLICT_RE.search(text) and not _SMARTPHONE_RE.search(product.title or ""):
        return False
    if (
        _SMARTPHONE_RE.search(product.title or "")
        or _IPHONE_RE.search(product.title or "")
        or _OS_RE.search(text)
    ):
        return True
    path = " ".join(product.category_path).lower()
    if "smartphone" in path:
        return True
    return bool(_PHONE_SIGNAL.search(text) and _SIZE_NUM.search(product.title or ""))


def implausible_device_price(goal: Goal, price: float | None) -> bool:
    if price is None or goal.budget is None or goal.budget < 1000:
        return False
    if not wants_phone(goal) and "tablet" not in shopper_head(goal).lower():
        return False
    return price < min(200.0, 0.04 * goal.budget)


def _option_inches(variant: Variant) -> float:
    match = _SIZE_NUM.search(variant.options or "")
    return float(match.group(1)) if match else 0.0


def spec_ok(
    goal: Goal,
    product: Product,
    variant: Variant,
    *,
    sku_first: bool = False,
) -> bool:
    if goal.screen_inch is not None:
        opt = _option_inches(variant)
        siblings_encode = any(_option_inches(item) > 0 for item in product.variants)
        if sku_first:
            if opt:
                size = opt
            else:
                size = screen_in(product, variant)
                if siblings_encode and size <= 0:
                    return False
        else:
            size = screen_in(product, variant)
        if size <= 0 or abs(size - goal.screen_inch) > 0.005:
            return False
        raw_size = str(product.facts.get("size") or "")
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", raw_size)]
        if len(nums) >= 2 and max(nums) - min(nums) > 0.2:
            return False
    text = _blob(variant.options, product.facts, product.title).lower()
    if sku_first and variant.options:
        text = _blob(variant.options, product.facts).lower()
    if goal.ram_gb:
        compact = text.replace(" ", "")
        if goal.ram_gb not in compact and goal.ram_gb.replace("gb", "") + "gb" not in compact:
            return False
    if goal.resolution and not resolution_ok(goal.resolution, text):
        return False
    if goal.refresh_wanted is not None:
        hz = refresh_hz(product, variant)
        if hz + 0.01 < goal.refresh_wanted:
            return False
    return True


def resolution_ok(wanted: str, text: str) -> bool:
    raw = (text or "").lower()
    compact = raw.replace(" ", "")
    aliases = {
        "4k": ("4k", "uhd", "3840", "2160"),
        "qhd": ("qhd", "1440", "2560"),
        "fhd": ("fhd", "1080p", "fullhd", "1920x1080", "1080"),
        "hd": ("720p", "1280x720", "720phd"),
    }
    if any(token in compact for token in aliases.get(wanted, (wanted,))):
        return True
    if wanted == "hd":
        return bool(
            re.search(r"(?<![a-z0-9])hd(?![a-z0-9+])", raw)
            and not re.search(r"\b(?:fhd|qhd|uhd|full\s*hd)\b", raw)
        )
    return wanted in compact


def brand_ok(wanted: str, product: Product) -> bool:
    if not wanted:
        return True
    tokens = _tokens(wanted)
    if not tokens:
        return True
    blob = f"{product.brand or ''} {product.title or ''}".lower()
    return all(
        re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", blob)
        for token in tokens
    )


def phrase_ok(phrase: str, title: str) -> bool:
    need = _tokens(phrase)
    have = set(_tokens(title))
    return bool(need) and all(token in have for token in need)


def category_ok(wanted: str, product: Product) -> bool:
    if not wanted:
        return True
    path = " ".join(product.category_path).lower()
    title = (product.title or "").lower()
    token = wanted.lower().rstrip("s")
    return token in path or token in title


def _first_hz(text: str) -> float:
    oc = _OC_REFRESH.search(text)
    if oc:
        return float(next(g for g in oc.groups() if g))
    found = [float(m.group(1)) for m in _REFRESH_NUM.finditer(text)]
    return max(found) if found else 0.0


def refresh_hz(product: Product, variant: Variant) -> float:
    for text in (variant.options, product.facts.get("refresh_rate"), product.title):
        hz = _first_hz(str(text or ""))
        if hz:
            return hz
    return 0.0


def _title_inches(text: str) -> list[float]:
    return [float(match) for match in _SIZE_NUM.findall(str(text or ""))]


def screen_in(product: Product, variant: Variant) -> float:
    for text in (variant.options, product.facts.get("size"), product.title):
        found = _title_inches(str(text or ""))
        if found:
            return found[0]
    return 0.0


def spec_hint(goal: Goal, product: Product, variant: Variant | None = None) -> bool:
    if goal.screen_inch is None:
        return True
    inches = _title_inches(product.title)
    if variant is not None:
        inches.extend(_title_inches(variant.options))
        opt = _option_inches(variant)
        if opt:
            inches.append(opt)
    size_fact = product.facts.get("size")
    if size_fact:
        inches.extend(_title_inches(str(size_fact)))
    if not inches:
        return False
    return any(abs(value - goal.screen_inch) <= 0.2 for value in inches)


def preference_score(goal: Goal, product: Product, variant: Variant) -> float:
    text = product.text() + " " + variant.options.lower()
    if "headphone" in goal.category or "headphone" in goal.raw.lower():
        water = 1.0 if _WATER_RE.search(text) else 0.0
        sport = 1.0 if _SPORT_RE.search(text) else 0.0
        if re.search(r"\d+(?:\.\d+)?\s*days?", text, re.I):
            battery = 1.0
        else:
            hours = [
                float(n)
                for n in re.findall(
                    r"(?<![\w.])(\d+(?:\.\d+)?)\s*\+?\s*(?:hours?|hrs?|h)\b",
                    text,
                    re.I,
                )
            ]
            battery = (
                1.0
                if hours and max(hours) >= 15
                else 0.75
                if hours and max(hours) >= 10
                else 0.5
                if hours
                else 0.0
            )
        wireless = (
            1.0
            if re.search(r"\b(?:wireless|bluetooth|true wireless|tws)\b", text)
            else 0.0
        )
        return 0.30 * water + 0.28 * sport + 0.22 * battery + 0.20 * wireless
    water = 1.0 if _WATER_RE.search(text) else 0.0
    return 0.6 * water + 0.4 * (1.0 if product.facts else 0.0)


def event_killed(session: Session, product_id: str, sku: str, goal: Goal | None = None) -> bool:
    key = f"{product_id}::{sku}"
    variant = session.variant(product_id, sku)
    for event in session.events:
        target = event.get("target") or {}
        hit = f"{target.get('product_id')}::{target.get('sku')}"
        if hit == key or key in (event.get("targets") or []):
            if event.get("kind") == "stockout":
                return True
            new_price = _num(event.get("new_price"))
            if goal is not None and new_price is not None and not within_budget(goal, new_price):
                return True
            if event.get("kind") == "price_change" and goal is not None and new_price is not None:
                if not within_budget(goal, new_price):
                    return True
        if event.get("kind") == "stockout" and str(target.get("product_id")) == product_id:
            if hit == key or key in (event.get("targets") or []):
                return True
    if variant is not None and variant.in_stock is False:
        return True
    if goal is not None and variant is not None and not within_budget(goal, variant.price):
        return True
    return False


def _apply_spec_phrase(phrase: str, goal: Goal, *, overwrite: bool = False) -> None:
    screen = _SCREEN_RE.search(phrase)
    if screen and (overwrite or goal.screen_inch is None):
        goal.screen_inch = float(screen.group(1))
    ram = _RAM_RE.search(phrase)
    if ram and (overwrite or not goal.ram_gb):
        goal.ram_gb = f"{ram.group(1)}gb"
    hz = _REFRESH_NUM.search(phrase)
    if hz and (overwrite or goal.refresh_wanted is None):
        goal.refresh_wanted = float(hz.group(1))
    low = phrase.lower()
    for token, aliases in (
        ("4k", ("4k", "uhd")),
        ("qhd", ("qhd",)),
        ("fhd", ("1080p", "full hd", "fhd")),
        ("hd", ("720p",)),
    ):
        if any(alias in low for alias in aliases):
            if overwrite or not goal.resolution:
                goal.resolution = token
            break


def parse_revision(text: str, goal: Goal) -> None:
    changed = _NOW_HAVE.search(text)
    if changed:
        _apply_spec_phrase(changed.group(1), goal, overwrite=True)
    else:
        _apply_spec_phrase(text, goal, overwrite=False)
    budget = _BUDGET_RE.search(text)
    if budget:
        goal.budget = min(goal.budget or 10**12, float(budget.group(1).replace(",", "")))
    tighten = _TIGHTEN_RE.search(text)
    if tighten:
        goal.budget = min(goal.budget or 10**12, float(tighten.group(1).replace(",", "")))


def shopper_spec(text: str, goal: Goal) -> None:
    parse_revision(text, goal)


def apply_shopper(session: Session, goal: Goal) -> None:
    for text in session.shopper:
        parse_revision(text, goal)


def source_key(product: Product) -> str:
    return product.listing_id or product.product_id


def viewed_sources(session: Session) -> set[str]:
    keys = set()
    for pid in session.viewed | session.compared:
        product = session.products.get(pid)
        if product is not None:
            keys.add(source_key(product))
    return keys


def sku_stem(sku: str) -> str:
    text = str(sku or "")
    return text.rsplit("-", 1)[0] if "-" in text else text


def option_explicit(goal: Goal, variant: Variant) -> int:
    if goal.screen_inch is not None:
        size = _option_inches(variant)
        return 0 if size and abs(size - goal.screen_inch) <= 0.005 else 1
    if goal.ram_gb and goal.ram_gb.replace("gb", "") in (variant.options or "").lower().replace(" ", ""):
        return 0
    if goal.resolution and resolution_ok(goal.resolution, variant.options or ""):
        return 0
    return 1


def _hint_score(goal: Goal, product: Product) -> int:
    score = 0
    title = product.title or ""
    if _ACCESSORY_RE.search(title):
        score -= 4
    if goal.screen_inch is not None and spec_hint(goal, product):
        score += 4
    if wants_phone(goal) and _SMARTPHONE_RE.search(title):
        score += 2
    if goal.brand and brand_ok(goal.brand, product):
        score += 2
    if goal.preferred_brand and brand_ok(goal.preferred_brand, product):
        score += 1
    if goal.category and category_ok(goal.category, product):
        score += 1
    if goal.resolution and resolution_ok(goal.resolution, title):
        score += 2
    return score


def view_promising(session: Session, goal: Goal, limit: int = 8) -> None:
    if session.remaining() <= _RESERVE_TURNS:
        return
    ranked = []
    for product in session.products.values():
        if product.product_id in session.viewed or product.product_id in session.compared:
            continue
        ranked.append((_hint_score(goal, product), product.product_id))
    ranked.sort(reverse=True)
    if goal.screen_inch is not None:
        ranked = [
            row
            for row in ranked
            if spec_hint(goal, session.products[row[1]])
        ]
    if wants_phone(goal):
        ranked = [
            row
            for row in ranked
            if smartphone_ok(session.products[row[1]], None)
        ]
    chosen = [pid for score, pid in ranked if score > 0][:limit]
    if not chosen:
        chosen = [pid for _score, pid in ranked[:limit]]
    session.view(chosen)


def open_budget_shelf(session: Session, goal: Goal, limit: int = 12) -> None:
    if session.remaining() <= _RESERVE_TURNS:
        return
    pids: list[str] = []
    for product in session.products.values():
        if product.product_id in session.viewed or product.product_id in session.compared:
            continue
        if part_listing(product):
            continue
        if goal.brand and not brand_ok(goal.brand, product):
            continue
        if wants_phone(goal) and not smartphone_ok(product, None):
            continue
        if goal.budget is not None:
            prices = [item.price for item in product.variants if item.price is not None]
            if prices and min(prices) > goal.budget + 0.009:
                continue
        pids.append(product.product_id)
    session.view(pids[:limit])


def broaden_catalog(session: Session, goal: Goal) -> None:
    if session.commits or session.remaining() <= _RESERVE_TURNS:
        return
    device = goal.category or "device"
    extras = []
    if goal.model:
        extras.append(f"{goal.model} {device}")
    if goal.brand:
        extras.append(f"{goal.brand} {device}")
    if goal.preferred_brand:
        extras.append(f"{goal.preferred_brand} {device}")
    if goal.screen_inch:
        extras.append(f"{goal.screen_inch:g} inch {device}")
    if goal.resolution:
        extras.append(f"{goal.resolution} {device}")
    extras.append(device)
    for query in extras:
        session.search(query, k=10, force=True)
    view_promising(session, goal, limit=12)


def candidate_ok(session: Session, goal: Goal, product_id: str, sku: str) -> bool:
    if event_killed(session, product_id, sku, goal):
        return False
    product = session.products.get(product_id)
    variant = session.variant(product_id, sku)
    if product is None or variant is None or not live_ok(goal, variant):
        return False
    if part_listing(product, variant):
        return False
    if implausible_device_price(goal, variant.price):
        return False
    if goal.family in {"intent_decomposition", "constraint_satisfaction", "justification"}:
        sku_first = goal.family in {"constraint_satisfaction", "intent_decomposition"}
        if not spec_ok(goal, product, variant, sku_first=sku_first):
            return False
        if wants_phone(goal) and not smartphone_ok(product, variant):
            return False
        if goal.family == "constraint_satisfaction" and goal.brand:
            if not brand_ok(goal.brand, product):
                return False
        if goal.family == "justification" and goal.brand:
            if not brand_ok(goal.brand, product):
                return False
    if goal.family == "retrieval_recall" and goal.phrase:
        if not phrase_ok(goal.phrase, product.title):
            return False
    return True


def _narrate(session: Session, content: str) -> None:
    if session.done or session.remaining() < 8:
        return
    session.ask(content)


def _catalog_rows(session: Session, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for product_id, sku in pairs:
        product = session.products.get(product_id)
        variant = session.variant(product_id, sku)
        if product is None or variant is None:
            continue
        rows.append(
            {
                "product_id": product_id,
                "sku": sku,
                "title": product.title,
                "brand": product.brand,
                "facts": product.facts,
                "options": variant.options,
                "price": variant.price,
                "in_stock": variant.in_stock,
            }
        )
    return rows


def _choose_from(
    session: Session,
    goal: Goal,
    pairs: list[tuple[str, str]],
) -> tuple[str, str] | None:
    valid = []
    for product_id, sku in pairs:
        if candidate_ok(session, goal, product_id, sku):
            valid.append((product_id, sku))
    if not valid:
        return None
    pids = list(dict.fromkeys(pid for pid, _sku in valid))
    if session.remaining() >= _RESERVE_TURNS + 5:
        session.view(pids[:8])
        extra = [pid for pid in pids[8:12] if pid not in session.viewed]
        if extra and session.remaining() >= 8:
            session.view(extra)
    elif session.remaining() >= _RESERVE_TURNS + 3:
        session.view(pids[:3])
    ranked = []
    for product_id, sku in valid:
        variant = session.variant(product_id, sku)
        if variant is None or not candidate_ok(session, goal, product_id, sku):
            continue
        explicit = option_explicit(goal, variant) if goal.family in {
            "constraint_satisfaction",
            "intent_decomposition",
        } else 0
        ranked.append((explicit, variant.price or 0.0, product_id, sku))
    ranked.sort()
    if not ranked:
        return None
    if goal.family in {
        "constraint_satisfaction",
        "intent_decomposition",
        "justification",
        "recovery",
    }:
        best = ranked[0][0]
        ranked = [row for row in ranked if row[0] == best][:3]
    pairs = [(pid, sku) for _ex, _price, pid, sku in ranked[:8]]
    picked = _llm_or_first(
        session, goal, pairs, purpose="select the variant that meets every firm spec"
    )
    if picked and candidate_ok(session, goal, picked[0], picked[1]):
        return picked
    return pairs[0]


# ---------------------------------------------------------------------------
# Family engines
# ---------------------------------------------------------------------------


def _switch_to(session: Session, product_id: str, sku: str) -> None:
    if (product_id, sku) in session.commits:
        return
    need_view = product_id not in session.viewed
    if need_view and session.remaining() >= 3:
        session.view([product_id])
    if session.remaining() >= 2:
        session.add(product_id, sku)


def _spec_search(session: Session, goal: Goal) -> None:
    device = goal.category or ("smartphone" if wants_phone(goal) else "device")
    brand = goal.brand if goal.family != "intent_decomposition" else ""
    if goal.screen_inch:
        query = f"{brand} {goal.screen_inch:g} inch {device}".strip()
    elif goal.resolution:
        query = f"{brand} {goal.resolution} {device}".strip()
    elif goal.model:
        query = f"{goal.color} {goal.model} {device}".strip()
    else:
        query = f"{brand} {device}".strip()
    if query:
        session.search(shop_query(goal, query), k=10, force=True)
    view_promising(session, goal, limit=12)


def _loose_live(
    session: Session, goal: Goal, dead_pid: str = "", dead_sku: str = ""
) -> list[tuple[str, str]]:
    rows: list[tuple[float, str, str]] = []
    for product in session.products.values():
        if part_listing(product):
            continue
        for variant in product.variants:
            if (product.product_id, variant.sku) == (dead_pid, dead_sku):
                continue
            if event_killed(session, product.product_id, variant.sku, goal):
                continue
            if not live_ok(goal, variant):
                continue
            if implausible_device_price(goal, variant.price):
                continue
            if goal.family in {
                "intent_decomposition",
                "constraint_satisfaction",
                "justification",
            }:
                if wants_phone(goal) and not smartphone_ok(product, variant):
                    continue
                if goal.screen_inch is not None and not spec_ok(
                    goal, product, variant, sku_first=False
                ):
                    if not spec_hint(goal, product, variant):
                        continue
                if (
                    goal.brand
                    and goal.family != "intent_decomposition"
                    and not brand_ok(goal.brand, product)
                ):
                    continue
            rows.append((variant.price or 0.0, product.product_id, variant.sku))
    rows.sort()
    return [(pid, sku) for _price, pid, sku in rows[:8]]


def _live_pick(
    session: Session, goal: Goal, dead_pid: str = "", dead_sku: str = ""
) -> tuple[str, str] | None:
    found = recover_any(session, goal, dead_pid, dead_sku)
    if found:
        return found
    if session.remaining() >= 2:
        _spec_search(session, goal)
        found = recover_any(session, goal, dead_pid, dead_sku)
        if found:
            return found
    loose = _loose_live(session, goal, dead_pid, dead_sku)
    if not loose:
        return None
    picked = _llm_or_first(
        session,
        goal,
        loose,
        purpose="pick a live in-budget replacement that still meets the request",
    )
    return picked or loose[0]


_OWNED_RECOVER = frozenset(
    {"ranking", "recovery", "preference_reasoning", "justification"}
)


def _best_live(
    session: Session, goal: Goal, dead_pid: str = "", dead_sku: str = ""
) -> tuple[str, str] | None:
    if goal.family in _OWNED_RECOVER:
        return None
    return recover_observed(session, goal, dead_pid, dead_sku) or recover_any(
        session, goal, dead_pid, dead_sku
    )


def _review_legal(
    session: Session,
    goal: Goal,
    product: Product,
    variant: Variant,
    current_pid: str,
    current_sku: str,
    *,
    committed: bool,
) -> bool:
    pid, sku = product.product_id, variant.sku
    if not candidate_ok(session, goal, pid, sku):
        return False
    if goal.family == "ranking":
        return (pid, sku) == (current_pid, current_sku)
    if goal.family == "recovery":
        if committed and current_sku:
            return sku_stem(sku) == sku_stem(current_sku)
        if goal.model and goal.color:
            return (
                _same_model(goal.model, _model_of(product, variant))
                and _same_color(goal.color, _color_of(product, variant))
            )
        return True
    if goal.family == "preference_reasoning":
        score = preference_score(goal, product, variant)
        current = session.variant(current_pid, current_sku)
        current_product = session.products.get(current_pid)
        if current is not None and current_product is not None:
            return score + 0.001 >= preference_score(goal, current_product, current)
        return score >= 0.65
    if goal.family == "justification":
        return _fact_backed(goal, product, variant)
    return True


def _review_pool(
    session: Session,
    goal: Goal,
    current_pid: str,
    current_sku: str,
    *,
    committed: bool,
) -> list[tuple[str, str]]:
    rows: list[tuple[float, str, str]] = []
    for product in session.products.values():
        for variant in product.variants:
            if not _review_legal(
                session,
                goal,
                product,
                variant,
                current_pid,
                current_sku,
                committed=committed,
            ):
                continue
            rows.append((variant.price or 0.0, product.product_id, variant.sku))
    rows.sort()
    ordered: list[tuple[str, str]] = []
    current = (current_pid, current_sku)
    if any((pid, sku) == current for _price, pid, sku in rows):
        ordered.append(current)
    for _price, pid, sku in rows:
        if (pid, sku) not in ordered:
            ordered.append((pid, sku))
        if len(ordered) >= 8:
            break
    if not ordered and candidate_ok(session, goal, current_pid, current_sku):
        ordered.append(current)
    return ordered


def _leftover_need(
    session: Session, goal: Goal, product_id: str, sku: str
) -> str:
    if session.remaining() <= _RESERVE_TURNS + 1:
        return "stop"
    if not product_id or not sku:
        return "widen"
    if not candidate_ok(session, goal, product_id, sku):
        return "widen"
    product = session.products.get(product_id)
    variant = session.variant(product_id, sku)
    if product is None or variant is None:
        return "widen"
    if goal.family == "ranking":
        return "stop"
    if goal.family == "recovery" and session.commits:
        return "stop"
    if goal.family == "preference_reasoning":
        return "stop"
    if goal.family == "retrieval_recall":
        return "stop"
    if goal.family == "justification":
        inspected = f"{product_id}::{sku}" in session.inspected
        if not _fact_backed(goal, product, variant):
            return "widen"
        if not inspected:
            return "inspect"
        holes = _gate_holes(session, goal, product_id, sku)
        if "claims" in holes or "facts" in holes:
            return "inspect"
        return "stop"
    if goal.family in {"intent_decomposition", "constraint_satisfaction"}:
        if f"{product_id}::{sku}" not in session.inspected:
            return "inspect"
        if session.commits:
            return "stop"
        opened = sum(
            1
            for pid in session.viewed | session.compared
            if session.products.get(pid) is not None
        )
        if opened < 3 and session.remaining() >= _RESERVE_TURNS + 4:
            return "widen"
        return "stop"
    return "stop"


def _widen_review_catalog(
    session: Session, goal: Goal, current_pid: str, current_sku: str
) -> None:
    if session.remaining() <= _RESERVE_TURNS + 2:
        return
    if _leftover_need(session, goal, current_pid, current_sku) == "stop":
        return
    committed = bool(session.commits)
    if goal.family == "ranking":
        return
    if goal.family == "recovery":
        if committed and candidate_ok(session, goal, current_pid, current_sku):
            return
        device = "smartphone" if wants_phone(goal) else (goal.category or "device")
        if goal.model:
            session.search(f"{goal.model} {device}", k=10, force=committed)
        stem = sku_stem(current_sku)
        if committed and stem:
            session.search(stem, k=10, force=True)
        session.view(
            [
                product.product_id
                for product in session.products.values()
                if any(sku_stem(variant.sku) == stem for variant in product.variants)
                and product.product_id not in session.viewed
            ][:8]
        )
        return
    if goal.family == "preference_reasoning":
        session.filt(max_price=goal.budget, k=10)
        session.view([pid for pid in session.products if pid not in session.viewed][:8])
        return
    if not committed:
        if goal.family in {"intent_decomposition", "constraint_satisfaction"}:
            session.filt(
                category=goal.category or "smartphone",
                max_price=goal.budget,
                k=10,
            )
            if goal.screen_inch and len(session.search_queries) < 3:
                session.search(
                    shop_query(goal, f"{goal.screen_inch:g} inch smartphone"),
                    k=10,
                )
            open_budget_shelf(session, goal, limit=10)
        elif goal.family == "justification":
            session.filt(
                category=goal.category or "device",
                brand=goal.brand,
                max_price=goal.budget,
                k=10,
            )
            view_promising(session, goal, limit=8)
        elif goal.family == "retrieval_recall" and goal.phrase:
            session.view(
                [
                    pid
                    for pid, product in session.products.items()
                    if phrase_ok(goal.phrase, product.title)
                    and pid not in session.viewed
                ][:8]
            )
        return
    if goal.family == "retrieval_recall":
        if goal.phrase:
            session.view(
                [
                    pid
                    for pid, product in session.products.items()
                    if phrase_ok(goal.phrase, product.title)
                    and pid not in session.viewed
                    and pid not in session.compared
                ][:8]
            )
        return
    if goal.family == "justification":
        product = session.products.get(current_pid)
        variant = session.variant(current_pid, current_sku)
        if (
            product is not None
            and variant is not None
            and candidate_ok(session, goal, current_pid, current_sku)
            and _fact_backed(goal, product, variant)
        ):
            if (
                f"{current_pid}::{current_sku}" not in session.inspected
                and session.remaining() >= _RESERVE_TURNS + 2
            ):
                session.inspect([(current_pid, current_sku)], force=True)
            return
        hunt_catalog(session, goal, current_pid, current_sku)
        return
    hunt_catalog(session, goal, current_pid, current_sku)
    if goal.family == "constraint_satisfaction":
        open_budget_shelf(session, goal, limit=8)


def _allow_review_switch(
    session: Session,
    goal: Goal,
    current: tuple[str, str],
    proposed: tuple[str, str],
    *,
    committed: bool,
) -> bool:
    if proposed == current:
        return True
    if not candidate_ok(session, goal, proposed[0], proposed[1]):
        return False
    if goal.family == "ranking":
        return not candidate_ok(session, goal, current[0], current[1])
    if goal.family == "recovery" and committed:
        return sku_stem(proposed[1]) == sku_stem(current[1])
    if goal.family == "preference_reasoning":
        proposed_product = session.products.get(proposed[0])
        proposed_variant = session.variant(proposed[0], proposed[1])
        current_product = session.products.get(current[0])
        current_variant = session.variant(current[0], current[1])
        if None in (proposed_product, proposed_variant, current_product, current_variant):
            return False
        return preference_score(goal, proposed_product, proposed_variant) + 0.001 >= (
            preference_score(goal, current_product, current_variant)
        )
    if goal.family == "justification" and committed:
        current_product = session.products.get(current[0])
        current_variant = session.variant(current[0], current[1])
        if (
            current_product is not None
            and current_variant is not None
            and candidate_ok(session, goal, current[0], current[1])
            and _fact_backed(goal, current_product, current_variant)
        ):
            return False
        proposed_product = session.products.get(proposed[0])
        proposed_variant = session.variant(proposed[0], proposed[1])
        return bool(
            proposed_product
            and proposed_variant
            and _fact_backed(goal, proposed_product, proposed_variant)
        )
    return True


def _review_inspect(
    session: Session,
    goal: Goal,
    pool: list[tuple[str, str]],
    current: tuple[str, str],
) -> None:
    inspectable = [
        pair
        for pair in ([current, *pool] if current not in pool else pool)
        if pair[0] and pair[1] and f"{pair[0]}::{pair[1]}" not in session.inspected
    ]
    if inspectable and session.remaining() >= _RESERVE_TURNS + 3:
        session.inspect(inspectable[:4])


def _review_verdict(
    session: Session,
    goal: Goal,
    product_id: str,
    sku: str,
    pool: list[tuple[str, str]],
    *,
    purpose: str,
) -> dict[str, Any]:
    rows = _catalog_rows(session, pool[:8])
    if not rows:
        return {"action": "keep"}
    return llm_json(
        "Review the current pick against only the listed catalog variants. "
        "Do not invent product_id or sku values.",
        f"Purpose: {purpose}\n"
        f"Family: {goal.family}\n"
        f"Shopper: {goal.raw.split('Task rules:')[0][:900]}\n"
        f"Current: {product_id}::{sku}\n"
        f"Remaining turns: {session.remaining()}\n"
        f"Candidates: {json.dumps(rows, default=str)}\n"
        'Return {"action":"keep"|"switch"|"look","product_id":"...","sku":"...","why":"..."}. '
        "Use keep if the current pick still meets every firm requirement. "
        "Use switch only for another listed variant that is clearly better. "
        "Use look only if a missing visible fact could change the decision "
        "and turns remain after inspect and order.",
        session=session,
    )


def review_choice(
    session: Session,
    goal: Goal,
    product_id: str,
    sku: str,
    *,
    purpose: str,
) -> tuple[str, str]:
    if session.ordered or session.done or session.remaining() < _RESERVE_TURNS + 2:
        return product_id, sku
    if not product_id or not sku:
        return product_id, sku
    apply_shopper(session, goal)
    committed = bool(session.commits)
    need = _leftover_need(session, goal, product_id, sku)
    if need == "stop":
        return product_id, sku
    for step in range(_REVIEW_LOOPS):
        if session.remaining() < _RESERVE_TURNS + 2:
            break
        need = _leftover_need(session, goal, product_id, sku)
        if need == "stop":
            break
        if need == "widen":
            _widen_review_catalog(session, goal, product_id, sku)
        pool = _review_pool(session, goal, product_id, sku, committed=committed)
        current = (product_id, sku)
        if current not in pool and candidate_ok(session, goal, product_id, sku):
            pool = [current, *[pair for pair in pool if pair != current]]
        if not pool:
            break
        if need == "inspect":
            session.inspect([current], force=True)
        else:
            _review_inspect(session, goal, pool, current)
        pool = _review_pool(session, goal, product_id, sku, committed=committed)
        if current not in pool and candidate_ok(session, goal, product_id, sku):
            pool = [current, *[pair for pair in pool if pair != current]]
        if not pool:
            break
        if _leftover_need(session, goal, product_id, sku) == "stop":
            break
        verdict = _review_verdict(
            session, goal, product_id, sku, pool, purpose=purpose
        )
        action = str(verdict.get("action") or "keep").strip().lower()
        proposed = (
            str(verdict.get("product_id") or ""),
            str(verdict.get("sku") or ""),
        )
        if action == "switch" and proposed in pool and _allow_review_switch(
            session, goal, current, proposed, committed=committed
        ):
            product_id, sku = proposed
            break
        if action == "look" and _leftover_need(
            session, goal, product_id, sku
        ) == "widen" and session.remaining() >= _RESERVE_TURNS + 4:
            continue
        break
    if candidate_ok(session, goal, product_id, sku):
        return product_id, sku
    fallback_pool = _review_pool(
        session, goal, product_id, sku, committed=committed
    )
    for pair in fallback_pool:
        if _allow_review_switch(
            session, goal, (product_id, sku), pair, committed=committed
        ):
            return pair
    return product_id, sku


def _precommit_floor(
    session: Session, goal: Goal, product_id: str, sku: str
) -> tuple[str, str]:
    if session.commits or session.ordered or session.remaining() < _RESERVE_TURNS + 5:
        return product_id, sku
    if goal.family not in {
        "intent_decomposition",
        "constraint_satisfaction",
        "justification",
    }:
        return product_id, sku
    if goal.family in {"intent_decomposition", "constraint_satisfaction"}:
        session.filt(
            category=goal.category or "smartphone",
            max_price=goal.budget,
            k=10,
        )
        if goal.screen_inch and len(session.search_queries) < 3:
            session.search(
                shop_query(goal, f"{goal.screen_inch:g} inch smartphone"),
                k=10,
            )
        open_budget_shelf(session, goal, limit=10)
    else:
        _widen_review_catalog(session, goal, product_id, sku)
    pool = _review_pool(session, goal, product_id, sku, committed=False)
    unseen: list[str] = []
    if product_id and product_id not in session.viewed and product_id not in session.compared:
        unseen.append(product_id)
    for pid, _sku in pool:
        if pid not in session.viewed and pid not in session.compared and pid not in unseen:
            unseen.append(pid)
        if len(unseen) >= 6:
            break
    session.view(unseen[:6])
    inspectable = [(product_id, sku)] if product_id and sku else []
    for pair in pool[:4]:
        if pair not in inspectable:
            inspectable.append(pair)
    if inspectable and session.remaining() >= _RESERVE_TURNS + 4:
        session.inspect(inspectable[:4])
    refreshed = _review_pool(session, goal, product_id, sku, committed=False)
    if refreshed and (product_id, sku) not in refreshed:
        if candidate_ok(session, goal, refreshed[0][0], refreshed[0][1]):
            picked = _llm_or_first(
                session,
                goal,
                refreshed[:6],
                purpose="choose first add after opening more in-budget options",
            )
            if picked:
                return picked
    return product_id, sku


def _gate_holes(session: Session, goal: Goal, product_id: str, sku: str) -> list[str]:
    product = session.products.get(product_id)
    variant = session.variant(product_id, sku)
    if product is None or variant is None:
        return ["missing"]
    holes: list[str] = []
    if event_killed(session, product_id, sku, goal) or variant.in_stock is False:
        holes.append("stock")
    if not within_budget(goal, variant.price):
        holes.append("budget")
    if goal.family in {
        "intent_decomposition",
        "constraint_satisfaction",
        "justification",
    }:
        sku_first = goal.family in {"constraint_satisfaction", "intent_decomposition"}
        if not spec_ok(goal, product, variant, sku_first=sku_first):
            holes.append("spec")
        if wants_phone(goal) and not smartphone_ok(product, variant):
            holes.append("device")
    if goal.family == "retrieval_recall" and goal.phrase:
        if not phrase_ok(goal.phrase, product.title):
            holes.append("phrase")
    if goal.family == "justification":
        if not _fact_backed(goal, product, variant):
            holes.append("facts")
        claims = build_justification(session, goal, product_id, sku).get("claims") or []
        fields = {str(item.get("field") or "") for item in claims if isinstance(item, dict)}
        if goal.resolution and "resolution" not in fields:
            holes.append("claims")
        if goal.refresh_wanted is not None and "refresh_rate" not in fields:
            holes.append("claims")
        if goal.screen_inch is not None and "screen_size" not in fields:
            holes.append("claims")
    if not candidate_ok(session, goal, product_id, sku) and "stock" not in holes:
        holes.append("candidate")
    return holes


def _rehearse_gates(
    session: Session, goal: Goal, product_id: str, sku: str
) -> tuple[str, str]:
    if session.ordered or session.done or session.remaining() < _RESERVE_TURNS + 2:
        return product_id, sku
    if (
        f"{product_id}::{sku}" not in session.inspected
        and session.remaining() >= _RESERVE_TURNS + 2
    ):
        session.inspect([(product_id, sku)], force=True)
    holes = _gate_holes(session, goal, product_id, sku)
    if not holes:
        return product_id, sku
    hard = [hole for hole in holes if hole in {"stock", "budget", "missing", "candidate"}]
    critique = llm_json(
        "The current order candidate may fail a hard shopper check. "
        "Use only listed catalog ids.",
        f"Family: {goal.family}\n"
        f"Current: {product_id}::{sku}\n"
        f"Holes: {json.dumps(holes)}\n"
        f"Shopper: {goal.raw.split('Task rules:')[0][:700]}\n"
        f"Candidates: {json.dumps(_catalog_rows(session, _review_pool(session, goal, product_id, sku, committed=bool(session.commits))[:6]), default=str)}\n"
        'Return {"action":"keep"|"switch","product_id":"...","sku":"...","hole":"..."}.',
        session=session,
    )
    action = str(critique.get("action") or ("switch" if hard else "keep")).strip().lower()
    if action == "keep" and not hard:
        if "facts" in holes or "claims" in holes:
            session.inspect([(product_id, sku)], force=True)
        return product_id, sku
    _widen_review_catalog(session, goal, product_id, sku)
    if "spec" in holes and goal.screen_inch:
        session.search(
            shop_query(goal, f"{goal.screen_inch:g} inch {goal.category or 'device'}"),
            k=10,
            force=True,
        )
        if goal.family == "constraint_satisfaction":
            open_budget_shelf(session, goal, limit=8)
    pool = _review_pool(
        session, goal, product_id, sku, committed=bool(session.commits)
    )
    proposed = (
        str(critique.get("product_id") or ""),
        str(critique.get("sku") or ""),
    )
    current = (product_id, sku)
    if proposed in pool and _allow_review_switch(
        session, goal, current, proposed, committed=bool(session.commits)
    ):
        return proposed
    if hard:
        for pair in pool:
            if _allow_review_switch(
                session, goal, current, pair, committed=bool(session.commits)
            ):
                return pair
    return product_id, sku


def _place_final(
    session: Session,
    goal: Goal,
    product_id: str,
    sku: str,
    justification: dict[str, Any] | None = None,
) -> None:
    if session.ordered or session.done or session.remaining() < 1:
        return
    apply_shopper(session, goal)
    if not product_id or not sku:
        fallback = _best_live(session, goal)
        if fallback is None:
            return
        product_id, sku = fallback
        _switch_to(session, product_id, sku)
    if session.remaining() >= _RESERVE_TURNS + 4:
        reviewed = review_choice(session, goal, product_id, sku, purpose="placing the order")
        if reviewed != (product_id, sku):
            product_id, sku = reviewed
    if not candidate_ok(session, goal, product_id, sku):
        hunt_catalog(session, goal, product_id, sku)
        fallback = _best_live(session, goal, product_id, sku) or recover_any(
            session, goal, product_id, sku
        )
        if fallback is None and goal.family not in _OWNED_RECOVER:
            loose = _loose_live(session, goal, product_id, sku)
            if loose:
                fallback = loose[0]
        if fallback:
            product_id, sku = fallback
    if session.remaining() >= _RESERVE_TURNS + 4:
        rehearsed = _rehearse_gates(session, goal, product_id, sku)
        if rehearsed != (product_id, sku) and candidate_ok(
            session, goal, rehearsed[0], rehearsed[1]
        ):
            product_id, sku = rehearsed
    if not candidate_ok(session, goal, product_id, sku):
        for alt_pid, alt_sku in reversed(session.commits):
            if candidate_ok(session, goal, alt_pid, alt_sku):
                product_id, sku = alt_pid, alt_sku
                break
    if product_id not in session.viewed:
        reserve = 2 if (product_id, sku) in session.commits else 3
        if session.remaining() >= reserve:
            session.view([product_id])
    if (product_id, sku) not in session.commits:
        if session.remaining() < 2:
            if session.commits:
                product_id, sku = session.commits[-1]
            else:
                return
        else:
            _switch_to(session, product_id, sku)
            if (product_id, sku) not in session.commits:
                if session.commits:
                    product_id, sku = session.commits[-1]
                else:
                    return
    if (
        f"{product_id}::{sku}" not in session.inspected
        and session.remaining() >= 3
    ):
        session.inspect([(product_id, sku)], force=True)
    if goal.family == "justification":
        justification = build_justification(session, goal, product_id, sku)
    if (product_id, sku) in session.commits and session.remaining() >= 1:
        if not candidate_ok(session, goal, product_id, sku):
            for alt_pid, alt_sku in reversed(session.commits):
                if candidate_ok(session, goal, alt_pid, alt_sku):
                    product_id, sku = alt_pid, alt_sku
                    if goal.family == "justification":
                        justification = build_justification(
                            session, goal, product_id, sku
                        )
                    break
            else:
                live = _best_live(session, goal, product_id, sku)
                if live and session.remaining() >= 2:
                    product_id, sku = live
                    _switch_to(session, product_id, sku)
                    if goal.family == "justification":
                        justification = build_justification(
                            session, goal, product_id, sku
                        )
        if (product_id, sku) in session.commits:
            session.order(product_id, sku, justification)


def _recover_after_event(
    session: Session,
    goal: Goal,
    product_id: str,
    sku: str,
    recover,
) -> tuple[str, str] | None:
    apply_shopper(session, goal)
    if goal.family in _OWNED_RECOVER:
        return recover(session, goal, product_id, sku)
    remembered = recover_observed(session, goal, product_id, sku)
    if remembered:
        return remembered
    hunt_catalog(session, goal, product_id, sku)
    remembered = recover_observed(session, goal, product_id, sku)
    if remembered:
        return remembered
    if goal.family == "retrieval_recall":
        session.view(
            [
                pid
                for pid, product in session.products.items()
                if goal.phrase
                and phrase_ok(goal.phrase, product.title)
                and pid not in session.viewed
                and pid not in session.compared
            ][:8]
        )
    memory = recover_any(session, goal, product_id, sku)
    if memory:
        return memory
    if session.remaining() >= _RESERVE_TURNS + 2:
        family_hit = recover(session, goal, product_id, sku)
        if family_hit:
            return family_hit
        return recover_any(session, goal, product_id, sku)
    return None


def _commit_and_settle(
    session: Session,
    goal: Goal,
    product_id: str,
    sku: str,
    *,
    recover,
    justification: dict[str, Any] | None = None,
) -> None:
    if (product_id, sku) not in session.commits:
        product_id, sku = _precommit_floor(session, goal, product_id, sku)
        product_id, sku = review_choice(
            session, goal, product_id, sku, purpose="the first cart add"
        )
        if product_id not in session.viewed:
            session.view([product_id])
        if f"{product_id}::{sku}" not in session.inspected and session.remaining() >= 5:
            session.inspect([(product_id, sku)])
        session.add(product_id, sku)
    if session.remaining() >= 3:
        session.inspect([(product_id, sku)], force=True)
    apply_shopper(session, goal)
    session.reply_if_asked()
    apply_shopper(session, goal)
    final_pid, final_sku = product_id, sku
    if not candidate_ok(session, goal, product_id, sku):
        recovered = _recover_after_event(session, goal, product_id, sku, recover)
        if recovered:
            final_pid, final_sku = recovered
            if final_pid not in session.viewed and session.remaining() >= 3:
                session.view([final_pid])
            _switch_to(session, final_pid, final_sku)
    if goal.family not in _OWNED_RECOVER and not candidate_ok(
        session, goal, final_pid, final_sku
    ):
        hunt_catalog(session, goal, final_pid, final_sku)
        fallback = _best_live(session, goal, final_pid, final_sku)
        if fallback:
            final_pid, final_sku = fallback
            _switch_to(session, final_pid, final_sku)
        elif session.commits:
            for alt_pid, alt_sku in reversed(session.commits):
                if not event_killed(session, alt_pid, alt_sku, goal) and live_ok(
                    goal, session.variant(alt_pid, alt_sku)
                ):
                    final_pid, final_sku = alt_pid, alt_sku
                    break
    _place_final(session, goal, final_pid, final_sku, justification)


def recover_any(session: Session, goal: Goal, dead_pid: str, dead_sku: str) -> tuple[str, str] | None:
    pairs = []
    sku_first = goal.family in {"constraint_satisfaction", "intent_decomposition"}
    for product in session.products.values():
        if goal.family == "constraint_satisfaction" and product.product_id == dead_pid:
            continue
        if goal.family in {
            "intent_decomposition",
            "constraint_satisfaction",
            "justification",
        } and part_listing(product):
            continue
        if goal.family == "retrieval_recall" and goal.phrase:
            if not phrase_ok(goal.phrase, product.title):
                continue
        for variant in product.variants:
            if (product.product_id, variant.sku) == (dead_pid, dead_sku):
                continue
            if event_killed(session, product.product_id, variant.sku, goal):
                continue
            if not live_ok(goal, variant):
                continue
            if implausible_device_price(goal, variant.price):
                continue
            if goal.family in {"constraint_satisfaction", "intent_decomposition", "justification"}:
                if wants_phone(goal) and not smartphone_ok(product, variant):
                    continue
                if not spec_ok(goal, product, variant, sku_first=sku_first):
                    continue
                if (
                    goal.brand
                    and not brand_ok(goal.brand, product)
                    and goal.family != "intent_decomposition"
                ):
                    continue
            pairs.append((product.product_id, variant.sku, variant.price or 0.0))
    if not pairs:
        return None
    pairs.sort(key=lambda row: row[2])
    return _llm_or_first(
        session,
        goal,
        [(pid, sku) for pid, sku, _price in pairs[:3]],
        purpose="recover to a different live listing that still meets the request",
    )


def model_tokens(product: Product) -> set[str]:
    values = [str(product.facts.get("model") or "")]
    values.extend(_tokens(product.title))
    out = set()
    for token in values:
        compact = "".join(re.findall(r"[a-z0-9]+", token.lower()))
        if _MODEL_TOKEN.fullmatch(compact):
            out.add(compact)
    return out


def cluster_key(product: Product, variant: Variant | None = None) -> str:
    models = model_tokens(product)
    if models:
        return "model:" + min(models)
    brand = (product.brand or "").strip().lower()
    titles = frozenset(_tokens(product.title))
    price = round(variant.price or 0.0, 2) if variant is not None else 0.0
    if brand and titles:
        return f"twin:{brand}|{' '.join(sorted(titles))}|{price}"
    return f"pid:{product.product_id}"


def run_ranking(session: Session, goal: Goal) -> None:
    keys = goal.candidates
    if not keys:
        return
    pids = list(dict.fromkeys(pid for pid, _ in keys))
    session.view(pids)
    session.inspect(keys)

    def rank() -> list[tuple[str, str, float, float, float]]:
        rows = []
        for pid, sku in keys:
            product = session.products.get(pid)
            variant = session.variant(pid, sku)
            if product is None or variant is None:
                continue
            if variant.in_stock is False:
                continue
            if event_killed(session, pid, sku, goal):
                continue
            if not live_ok(goal, variant):
                continue
            rows.append(
                (
                    pid,
                    sku,
                    refresh_hz(product, variant),
                    screen_in(product, variant),
                    variant.price if variant.price is not None else 10**12,
                )
            )
        rows.sort(key=lambda row: (-row[2], -row[3], row[4], f"{row[0]}::{row[1]}"))
        return rows

    ordered = rank()
    if not ordered:
        return
    first_pid, first_sku = ordered[0][0], ordered[0][1]
    llm_pick(
        session,
        goal,
        [(first_pid, first_sku)],
        purpose="confirm the highest refresh then largest screen option as first add",
    )
    if first_pid not in session.viewed and first_pid not in session.compared:
        session.view([first_pid])

    def recover_rank(sess: Session, parsed: Goal, dead_pid: str, dead_sku: str):
        sess.inspect(keys, force=True)
        live = rank()
        if not live:
            return None
        top = (live[0][0], live[0][1])
        llm_pick(
            sess,
            parsed,
            [top],
            purpose="confirm the remaining top-ranked monitor after the event",
        )
        return top

    _commit_and_settle(session, goal, first_pid, first_sku, recover=recover_rank)


def run_retrieval(session: Session, goal: Goal) -> None:
    phrase = goal.phrase or " ".join(_tokens(goal.raw.split("Task rules:")[0])[:4])
    for query in planned_searches(session, goal, [phrase]):
        session.search(query, k=10)
    if goal.category and len(session.search_queries) < 3:
        session.filt(category=goal.category, max_price=goal.budget, k=10)

    def matching() -> list[Product]:
        rows = []
        for product in session.products.values():
            if not phrase_ok(phrase, product.title):
                continue
            if goal.category and product.category_path and not category_ok(goal.category, product):
                continue
            rows.append(product)
        return rows

    hits = matching()
    if len({cluster_key(p) for p in hits}) < goal.required_clusters and len(session.search_queries) < 3:
        alt = " ".join(_tokens(phrase))
        if alt and alt not in session.search_queries:
            session.search(alt, k=10)
        hits = matching()
    session.view([product.product_id for product in matching()[:8]])
    picks = []
    for product in matching():
        for variant in product.variants:
            if not live_ok(goal, variant):
                continue
            picks.append((variant.price or 0.0, product.product_id, variant.sku))
    if not picks:
        return
    picks.sort()
    chosen = _choose_from(session, goal, [(pid, sku) for _price, pid, sku in picks])
    if chosen is None:
        return
    _commit_and_settle(session, goal, chosen[0], chosen[1], recover=recover_any)


def run_constraint(session: Session, goal: Goal) -> None:
    if goal.underspecified and not goal.asked_spec:
        asked = llm_chat(
            "Ask the shopper one short question that unlocks the missing firm spec.",
            f"Request: {goal.raw.split('Task rules:')[0][:900]}",
            session=session,
        )
        session.ask(
            asked
            or (
                "What is the firm hardware spec I must meet on the exact variant "
                "(RAM, screen size, resolution, or refresh rate)?"
            )
        )
        goal.asked_spec = True
        apply_shopper(session, goal)
    if goal.underspecified and not (
        goal.screen_inch or goal.ram_gb or goal.resolution or goal.refresh_wanted
    ):
        asked = llm_chat(
            "Ask again for the missing firm hardware spec. One short sentence.",
            "The shopper has not yet named RAM, screen size, resolution, or refresh rate.",
            session=session,
        )
        session.ask(asked or "Which exact RAM, screen size, resolution, or refresh rate is required?")
        apply_shopper(session, goal)
    fallbacks = []
    if goal.brand:
        fallbacks.append(f"{goal.brand} smartphone")
    if goal.screen_inch:
        fallbacks.append(f"{goal.screen_inch:g} inch smartphone")
    if goal.resolution:
        fallbacks.append(f"{goal.resolution} smartphone")
    fallbacks.append("smartphone")
    for query in planned_searches(session, goal, fallbacks):
        session.search(query, k=10)
    if goal.screen_inch:
        session.search(
            shop_query(
                goal,
                f"{goal.brand} {goal.screen_inch:g} inch smartphone".strip(),
            ),
            k=10,
            force=True,
        )
    open_budget_shelf(session, goal, limit=12)
    view_promising(session, goal, limit=12)
    picks = []
    for product in session.products.values():
        if goal.brand and not brand_ok(goal.brand, product):
            continue
        path = " ".join(product.category_path).lower()
        if wants_phone(goal) and product.category_path:
            if "smartphone" not in path and not smartphone_ok(product, None):
                continue
        for variant in product.variants:
            if not live_ok(goal, variant):
                continue
            if wants_phone(goal) and not smartphone_ok(product, variant):
                continue
            if not spec_ok(goal, product, variant, sku_first=True):
                continue
            picks.append(
                (
                    option_explicit(goal, variant),
                    variant.price or 0.0,
                    product.product_id,
                    variant.sku,
                )
            )
    if not picks:
        query = (
            f"{goal.brand} {goal.screen_inch:g} inch smartphone"
            if goal.brand and goal.screen_inch
            else f"{goal.screen_inch:g} inch smartphone"
            if goal.screen_inch
            else f"{goal.brand} smartphone"
            if goal.brand
            else "smartphone"
        )
        session.search(query, k=10, force=True)
        open_budget_shelf(session, goal, limit=12)
        view_promising(session, goal, limit=12)
        for product in session.products.values():
            if goal.brand and not brand_ok(goal.brand, product):
                continue
            if part_listing(product):
                continue
            for variant in product.variants:
                if not live_ok(goal, variant):
                    continue
                if not spec_ok(goal, product, variant, sku_first=True):
                    continue
                if wants_phone(goal) and not smartphone_ok(product, variant):
                    continue
                picks.append(
                    (
                        option_explicit(goal, variant),
                        variant.price or 0.0,
                        product.product_id,
                        variant.sku,
                    )
                )
    if not picks:
        inspectable = []
        for product in session.products.values():
            if goal.brand and not brand_ok(goal.brand, product):
                continue
            if part_listing(product):
                continue
            for variant in product.variants:
                if live_ok(goal, variant):
                    inspectable.append((product.product_id, variant.sku))
        if inspectable and session.remaining() >= 5:
            session.inspect(inspectable[:6])
        for product in session.products.values():
            if goal.brand and not brand_ok(goal.brand, product):
                continue
            if part_listing(product):
                continue
            for variant in product.variants:
                if not live_ok(goal, variant):
                    continue
                if wants_phone(goal) and not smartphone_ok(product, variant):
                    continue
                if not spec_ok(goal, product, variant, sku_first=False):
                    continue
                picks.append(
                    (
                        option_explicit(goal, variant),
                        variant.price or 0.0,
                        product.product_id,
                        variant.sku,
                    )
                )
    if not picks:
        return

    def recover_constraint(sess: Session, parsed: Goal, dead_pid: str, dead_sku: str):
        apply_shopper(sess, parsed)
        hunt_catalog(sess, parsed, dead_pid, dead_sku)
        open_budget_shelf(sess, parsed, limit=12)
        inspectable = []
        for product in sess.products.values():
            if product.product_id == dead_pid:
                continue
            if parsed.brand and not brand_ok(parsed.brand, product):
                continue
            if part_listing(product):
                continue
            for variant in product.variants:
                if live_ok(parsed, variant):
                    inspectable.append((product.product_id, variant.sku))
        if inspectable and sess.remaining() >= 5:
            sess.inspect(inspectable[:6])
        hit = recover_any(sess, parsed, dead_pid, dead_sku)
        if hit:
            return hit
        query = shop_query(
            parsed,
            f"{parsed.brand} {parsed.screen_inch:g} inch {parsed.category or 'smartphone'}"
            if parsed.brand and parsed.screen_inch
            else f"{parsed.screen_inch:g} inch {parsed.category or 'smartphone'}"
            if parsed.screen_inch
            else f"{parsed.brand} {parsed.category or 'smartphone'}"
            if parsed.brand
            else parsed.category or "smartphone",
        )
        sess.search(query, k=10, force=True)
        open_budget_shelf(sess, parsed, limit=12)
        return recover_any(sess, parsed, dead_pid, dead_sku) or _live_pick(
            sess, parsed, dead_pid, dead_sku
        )

    picks.sort()
    chosen = _choose_from(session, goal, [(pid, sku) for _ex, _price, pid, sku in picks])
    if chosen is None:
        chosen = (picks[0][2], picks[0][3])
    _commit_and_settle(session, goal, chosen[0], chosen[1], recover=recover_constraint)


def run_intent(session: Session, goal: Goal) -> None:
    def search_spec() -> None:
        fallbacks = []
        if goal.preferred_brand and not session.commits:
            fallbacks.append(f"{goal.preferred_brand} smartphone")
        if goal.screen_inch:
            fallbacks.append(f"{goal.screen_inch:g} inch smartphone")
        if goal.ram_gb:
            fallbacks.append(f"{goal.ram_gb} RAM smartphone")
        if goal.resolution:
            fallbacks.append(f"{goal.resolution} smartphone")
        if goal.refresh_wanted:
            fallbacks.append(f"{goal.refresh_wanted:g}hz smartphone")
        fallbacks.append("smartphone")
        for query in planned_searches(session, goal, fallbacks):
            if not session.commits and len(session.search_queries) >= 3:
                break
            session.search(query, k=10)
        view_promising(session, goal, limit=10)

    def compliant(prefer_brand: bool) -> list[tuple[float, str, str]]:
        rows = []
        for product in session.products.values():
            if prefer_brand and goal.preferred_brand and not brand_ok(goal.preferred_brand, product):
                continue
            for variant in product.variants:
                if not candidate_ok(session, goal, product.product_id, variant.sku):
                    continue
                if not spec_ok(goal, product, variant):
                    continue
                if wants_phone(goal) and not smartphone_ok(product, variant):
                    continue
                rows.append((variant.price or 0.0, product.product_id, variant.sku))
        rows.sort()
        return rows

    search_spec()
    first = compliant(True) or compliant(False)
    if not first:
        loose = []
        for product in session.products.values():
            for variant in product.variants:
                if live_ok(goal, variant) and not event_killed(
                    session, product.product_id, variant.sku, goal
                ):
                    if wants_phone(goal) and not smartphone_ok(product, variant):
                        continue
                    loose.append((variant.price or 0.0, product.product_id, variant.sku))
        loose.sort()
        first = loose
    if not first:
        return

    def recover_intent(sess: Session, parsed: Goal, dead_pid: str, dead_sku: str):
        hunt_catalog(sess, parsed, dead_pid, dead_sku)
        view_promising(sess, parsed, limit=6)
        live = [
            (pid, sku)
            for _price, pid, sku in compliant(False)
            if (pid, sku) != (dead_pid, dead_sku)
        ]
        chosen = _choose_from(sess, parsed, live)
        if chosen:
            return chosen
        if sess.remaining() >= _RESERVE_TURNS + 4:
            query = shop_query(
                parsed,
                f"{parsed.screen_inch:g} inch {parsed.category or 'smartphone'}"
                if parsed.screen_inch
                else parsed.category or "smartphone",
            )
            sess.search(query, k=10, force=True)
            sess.filt(category=parsed.category or "smartphone", max_price=parsed.budget, k=10)
            view_promising(sess, parsed, limit=6)
            live = [
                (pid, sku)
                for _price, pid, sku in compliant(False)
                if (pid, sku) != (dead_pid, dead_sku)
            ]
            chosen = _choose_from(sess, parsed, live)
            if chosen:
                return chosen
        return recover_any(sess, parsed, dead_pid, dead_sku) or _live_pick(
            sess, parsed, dead_pid, dead_sku
        )

    chosen = _choose_from(session, goal, [(pid, sku) for _price, pid, sku in first])
    if chosen is None:
        return
    _commit_and_settle(session, goal, chosen[0], chosen[1], recover=recover_intent)


def run_preference(session: Session, goal: Goal) -> None:
    asked = llm_chat(
        "Ask the shopper one short clarifying question before adding anything.",
        "The request is about sport or sweat-resistant audio. Ask whether "
        "water resistance should outrank the lowest price, and any other "
        f"must-have features.\nRequest: {goal.raw.split('Task rules:')[0][:700]}",
        session=session,
    )
    session.ask(
        asked
        or (
            "Before I add anything: for this use case, should I prioritize sweat and "
            "water resistance over the lowest price, and any other must-have features?"
        )
    )
    apply_shopper(session, goal)
    device = "headphone" if "headphone" in goal.raw.lower() else (goal.category or "device")
    for query in planned_searches(
        session,
        goal,
        [
            f"ipx7 sweat waterproof wireless sport {device}",
            f"ipx5 bluetooth running {device}",
            f"running earhook sweatproof {device}",
        ],
    ):
        session.search(query, k=10)
    if goal.budget is not None:
        session.filt(max_price=goal.budget, k=10)

    def scoreboard() -> list[tuple[float, str, str, str]]:
        rows = []
        for product in session.products.values():
            if part_listing(product) or _ACCESSORY_RE.search(product.title or ""):
                continue
            for variant in product.variants:
                if not live_ok(goal, variant):
                    continue
                if event_killed(session, product.product_id, variant.sku, goal):
                    continue
                score = preference_score(goal, product, variant)
                rows.append((score, source_key(product), product.product_id, variant.sku))
        rows.sort(reverse=True)
        return rows

    session.view(list(session.products)[:12])
    scored = scoreboard()
    if scored and scored[0][0] < 0.99:
        session.search(f"ipx7 bluetooth {device} hours battery", k=10, force=True)
        session.view([pid for pid in session.products if pid not in session.viewed][:12])
        scored = scoreboard()
    if not any(row[0] >= 0.65 for row in scored):
        session.search(f"running earhook sweatproof {device}", k=10, force=True)
        session.view([pid for pid in session.products if pid not in session.viewed][:16])
        scored = scoreboard()
    if len(viewed_sources(session)) < 2:
        session.view([pid for pid in session.products if pid not in session.viewed][:8])
        scored = scoreboard()
    good = [row for row in scored if row[0] >= 0.65]
    if not good or len(viewed_sources(session)) < 2:
        return
    scored = good

    def recover_pref(sess: Session, parsed: Goal, dead_pid: str, dead_sku: str):
        apply_shopper(sess, parsed)
        if sess.remaining() >= _RESERVE_TURNS + 4:
            sess.search(f"ipx7 wireless sport {device}", k=10, force=True)
        sess.view([pid for pid in sess.products if pid not in sess.viewed][:6])
        dead_listing = source_key(sess.products[dead_pid]) if dead_pid in sess.products else dead_pid
        options = []
        for product in sess.products.values():
            if source_key(product) == dead_listing or part_listing(product):
                continue
            for variant in product.variants:
                if not live_ok(parsed, variant):
                    continue
                if event_killed(sess, product.product_id, variant.sku, parsed):
                    continue
                options.append(
                    (
                        preference_score(parsed, product, variant),
                        product.product_id,
                        variant.sku,
                    )
                )
        options.sort(reverse=True)
        strong = [row for row in options if row[0] >= 0.65]
        options = strong or options
        if not options:
            return recover_any(sess, parsed, dead_pid, dead_sku)
        best = options[0][0]
        top = [row for row in options if row[0] >= best - 1e-9]
        return _llm_or_first(
            sess,
            parsed,
            [(pid, sku) for _score, pid, sku in top],
            purpose="recover to another sweat-ready listing, not the cheapest accessory",
        )

    best = scored[0][0]
    top = [row for row in scored if row[0] >= best - 1e-9]
    first = _llm_or_first(
        session,
        goal,
        [(pid, sku) for _score, _src, pid, sku in top],
        purpose="add the best sweat-ready listing first, not a cheap accessory",
    )
    if first is None:
        return
    _commit_and_settle(session, goal, first[0], first[1], recover=recover_pref)


_COLOR_LABELS = (
    "midnight blue",
    "titanium black",
    "matte black",
    "triple black",
    "frost white",
    "white smoke",
    "cloud grey",
    "moonlight grey",
    "space gray",
    "space grey",
    "street gray",
    "driftwood sand",
    "cypress green",
    "sage green",
    "tidal blue",
    "sky blue",
    "light blue",
    "indigo blue",
    "canvas white",
    "platinum silver",
    "starlight",
    "graphite",
    "sandstone",
    "transparent",
    "silver",
    "beige",
    "black",
    "white",
    "blue",
    "green",
    "pink",
    "violet",
    "purple",
    "red",
    "sage",
    "gold",
    "khaki",
    "peach",
    "gray",
    "grey",
    "yellow",
    "orange",
    "brown",
)


def _color_of(product: Product, variant: Variant) -> str:
    for text in (variant.options, product.facts.get("color"), product.title):
        blob = str(text or "").lower()
        hits = [label for label in _COLOR_LABELS if re.search(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])", blob)]
        if not hits:
            continue
        longest = max(hits, key=len)
        return "green" if longest in {"sage green", "sage"} else longest
    return ""


def _model_code(raw: str) -> str:
    words = _tokens(raw)
    while words and words[-1] in {
        label.replace(" ", "") for label in _COLOR_LABELS if " " not in label
    }:
        words.pop()
    code = "".join(words)
    if 3 <= len(code) <= 24 and any(ch.isdigit() for ch in code):
        return code
    return ""


def _model_of(product: Product, variant: Variant) -> str:
    raw = str(product.facts.get("model") or "").strip()
    if raw and ";" not in raw and "/" not in raw:
        coded = _model_code(raw)
        if coded:
            return coded
        if raw.isalpha() and 3 <= len(raw) <= 24:
            return raw.lower()
    tokens = model_tokens(product)
    if tokens:
        return min(tokens)
    title_code = _model_code(product.title)
    return title_code


def _same_model(wanted: str, have: str) -> bool:
    a = _model_code(wanted) or "".join(_tokens(wanted))
    b = _model_code(have) or "".join(_tokens(have))
    return bool(a) and bool(b) and a == b


def _same_color(wanted: str, have: str) -> bool:
    if not wanted or not have:
        return False
    return wanted.lower() in have.lower() or have.lower() in wanted.lower()


def run_recovery(session: Session, goal: Goal) -> None:
    device = "smartphone" if wants_phone(goal) else (
        "headphones" if "headphone" in goal.raw.lower() else (goal.category or "device")
    )
    fallbacks = []
    if goal.model and goal.color:
        fallbacks.append(f"{goal.color} {goal.model} {device}")
    if goal.model:
        fallbacks.append(f"{goal.model} {device}")
    fallbacks.append(device)
    for query in planned_searches(session, goal, fallbacks):
        session.search(query, k=10)
    session.view(list(session.products)[:10])

    def phone_ok(product: Product, variant: Variant) -> bool:
        if wants_phone(goal):
            return smartphone_ok(product, variant)
        return True

    def level(product: Product, variant: Variant, stem: str = "") -> int:
        have_model = _model_of(product, variant)
        have_color = _color_of(product, variant)
        same_model = _same_model(goal.model, have_model)
        if not same_model and goal.model:
            compact = _model_code(goal.model) or "".join(_tokens(goal.model))
            blob = " ".join(
                (
                    product.title,
                    str(product.facts.get("model") or ""),
                    variant.options,
                )
            ).lower()
            if compact and re.search(
                rf"(?<![a-z0-9]){re.escape(compact)}(?![a-z0-9])", blob
            ):
                same_model = True
        if not same_model and stem and sku_stem(variant.sku) == stem:
            same_model = True
        same_color = _same_color(goal.color, have_color)
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        if same_color:
            return 2
        return 3

    def ranked(stem: str = "") -> list[tuple[int, float, str, str]]:
        rows = []
        for product in session.products.values():
            for variant in product.variants:
                if not live_ok(goal, variant):
                    continue
                if event_killed(session, product.product_id, variant.sku, goal):
                    continue
                if _ACCESSORY_RE.search(product.title or ""):
                    continue
                if not phone_ok(product, variant):
                    continue
                rows.append(
                    (
                        level(product, variant, stem=stem),
                        variant.price or 0.0,
                        product.product_id,
                        variant.sku,
                    )
                )
        rows.sort()
        return rows

    full = ranked()
    exact = [row for row in full if row[0] == 0]
    if not exact and goal.model:
        session.search(f"{goal.model} {device}", k=10, force=True)
        session.view([pid for pid in session.products if pid not in session.viewed][:8])
        full = ranked()
        exact = [row for row in full if row[0] == 0]
    if not exact:
        return
    first = (exact[0][2], exact[0][3])
    llm_pick(
        session,
        goal,
        [first],
        purpose="confirm the exact requested color and model as first add",
    )
    if first is None:
        return

    def recover_min(sess: Session, parsed: Goal, dead_pid: str, dead_sku: str):
        stem = sku_stem(dead_sku)

        def same_stem() -> tuple[str, str] | None:
            rows = [
                row
                for row in ranked(stem)
                if (row[2], row[3]) != (dead_pid, dead_sku)
                and sku_stem(row[3]) == stem
            ]
            if not rows:
                return None
            rows.sort()
            return rows[0][2], rows[0][3]

        def other_exact() -> tuple[str, str] | None:
            rows = [
                row
                for row in ranked(stem)
                if (row[2], row[3]) != (dead_pid, dead_sku) and row[0] <= 1
            ]
            if not rows:
                return None
            rows.sort()
            return rows[0][2], rows[0][3]

        hit = same_stem()
        if not hit:
            if parsed.model:
                sess.search(parsed.model, k=10, force=True)
                sess.search(f"{parsed.model} {device}", k=10, force=True)
            if stem:
                sess.search(stem, k=10, force=True)
            stem_pids = [
                product.product_id
                for product in sess.products.values()
                if any(sku_stem(variant.sku) == stem for variant in product.variants)
                and product.product_id not in sess.viewed
            ]
            sess.view(stem_pids[:8])
            hit = same_stem()
        if not hit:
            hit = other_exact()
        if not hit:
            return None
        confirmed = llm_pick(
            sess,
            parsed,
            [hit],
            purpose="confirm the least-relaxed same-model color replacement",
        )
        return confirmed or hit

    _commit_and_settle(session, goal, first[0], first[1], recover=recover_min)


def _fact_text(product: Product, variant: Variant) -> str:
    return _blob(product.facts, variant.options).lower()


def _fact_backed(goal: Goal, product: Product, variant: Variant) -> bool:
    surface = _fact_text(product, variant)
    if not surface.strip():
        return False
    if goal.resolution:
        return resolution_ok(goal.resolution, surface)
    if goal.refresh_wanted is not None:
        return refresh_hz(product, variant) + 0.01 >= goal.refresh_wanted
    if goal.screen_inch is not None:
        return abs(screen_in(product, variant) - goal.screen_inch) <= 0.005
    return True


def build_justification(
    session: Session, goal: Goal, product_id: str, sku: str
) -> dict[str, Any]:
    product = session.products.get(product_id) or Product(product_id=product_id)
    variant = session.variant(product_id, sku) or Variant(sku=sku)
    facts = product.facts or {}
    surface = _fact_text(product, variant)
    claims = []
    gate = goal.gate_attr
    if gate == "resolution" or (not gate and goal.resolution):
        display = str(facts.get("display") or facts.get("resolution") or "")
        value = ""
        for token, aliases in (
            ("4k", ("4k", "uhd", "3840")),
            ("qhd", ("qhd", "1440")),
            ("fhd", ("fhd", "1080p", "full hd", "1920")),
            ("hd", ("720p", "1280x720")),
        ):
            if any(alias in display.lower().replace(" ", "") for alias in aliases):
                value = token
                break
        if not value and goal.resolution == "hd" and resolution_ok(
            "hd", display or surface
        ):
            value = "hd"
        if not value and goal.resolution and resolution_ok(
            goal.resolution, _blob(display, surface)
        ):
            value = goal.resolution
        if value:
            claims.append({"field": "resolution", "value": value})
    elif gate == "refresh_rate" or (not gate and goal.refresh_wanted is not None):
        hz = refresh_hz(product, variant)
        if hz:
            claims.append({"field": "refresh_rate", "value": f"{hz:g}hz"})
    elif gate == "screen_size" or goal.screen_inch is not None:
        inches = screen_in(product, variant)
        if inches:
            claims.append({"field": "screen_size", "value": f"{inches:g}in"})
    if variant.price is not None:
        claims.append({"field": "price", "value": f"{variant.price:.2f}"})
    claims.append(
        {"field": "in_stock", "value": "true" if variant.in_stock is not False else "false"}
    )
    reason = None
    for field_name, patterns in (
        ("hdr", (r"\bhdr\b", r"\bhigh dynamic range\b")),
        ("microphone", (r"\bmicrophone\b", r"\bmic\b")),
        ("autofocus", (r"\bauto[-\s]?focus\b",)),
        ("privacy_cover", (r"\bprivacy\s+(?:cover|shutter)\b",)),
        ("streaming", (r"\bstreaming\b", r"\bvideo[-\s]?call", r"\bzoom\b", r"\bteams\b")),
    ):
        if any(re.search(pat, surface) for pat in patterns):
            reason = {"field": field_name, "value": "true"}
            break
    if reason is None and product.brand:
        reason = {"field": "brand", "value": product.brand}
    if reason is not None:
        claims.append(reason)
    return {"claims": claims}


def run_justification(session: Session, goal: Goal) -> None:
    fallbacks = []
    if goal.brand:
        fallbacks.append(f"{goal.brand} {goal.category or 'device'}")
    if goal.screen_inch:
        fallbacks.append(f"{goal.brand} {goal.screen_inch:g} inch {goal.category or ''}".strip())
    if goal.resolution:
        fallbacks.append(f"{goal.brand} {goal.resolution} {goal.category or ''}".strip())
    if goal.refresh_wanted:
        fallbacks.append(f"{goal.refresh_wanted:g}hz {goal.category or 'device'}".strip())
    fallbacks.append(goal.category or "device")
    for query in planned_searches(session, goal, fallbacks):
        session.search(query, k=10)
    view_promising(session, goal, limit=12)

    def gather() -> list[tuple[int, float, str, str]]:
        rows = []
        for product in session.products.values():
            low = (product.title or "").lower()
            if any(
                token in low
                for token in (
                    "refurb",
                    "pre-owned",
                    "restored",
                    "renewed",
                    "customer rating",
                )
            ):
                continue
            if part_listing(product) or _ACCESSORY_RE.search(product.title or ""):
                continue
            if goal.screen_inch is not None and not spec_hint(goal, product):
                continue
            if goal.brand and not brand_ok(goal.brand, product):
                continue
            if goal.category and product.category_path and not category_ok(goal.category, product):
                if goal.category.rstrip("s") not in product.title.lower():
                    continue
            for variant in product.variants:
                if not live_ok(goal, variant):
                    continue
                if not spec_ok(goal, product, variant):
                    continue
                if wants_phone(goal) and not smartphone_ok(product, variant):
                    continue
                backed = 0 if _fact_backed(goal, product, variant) else 1
                rows.append((backed, variant.price or 0.0, product.product_id, variant.sku))
        return rows

    session.view(
        [
            pid
            for pid, product in session.products.items()
            if (not goal.brand or brand_ok(goal.brand, product)) and not part_listing(product)
        ][:16]
    )
    picks = gather()
    inspectable = [(row[2], row[3]) for row in sorted(picks)[:6]]
    if inspectable and session.remaining() >= 6:
        session.inspect(inspectable)
        picks = gather()
    backed_only = [row for row in picks if row[0] == 0]
    if not backed_only:
        query = " ".join(
            part
            for part in (goal.brand, goal.resolution or (f"{goal.screen_inch:g} inch" if goal.screen_inch else ""), goal.category)
            if part
        ) or (goal.category or "device")
        session.search(query, k=10, force=True)
        view_promising(session, goal, limit=12)
        picks = gather()
        inspectable = [(row[2], row[3]) for row in sorted(picks)[:6]]
        if inspectable and session.remaining() >= 5:
            session.inspect(inspectable)
            picks = gather()
        backed_only = [row for row in picks if row[0] == 0]
    picks = backed_only or [row for row in picks if row[0] == 1]
    if not picks:
        hunt_catalog(session, goal)
        picks = [row for row in gather() if row[0] == 0]
        if not picks:
            return
    picks.sort()
    chosen = _choose_from(session, goal, [(pid, sku) for _b, _price, pid, sku in picks])
    if chosen is None:
        chosen = (picks[0][2], picks[0][3])
    pid, sku = chosen
    session.view([pid])
    session.inspect([(pid, sku)], force=True)

    def recover_just(sess: Session, parsed: Goal, dead_pid: str, dead_sku: str):
        def backed_rows() -> list[tuple[int, float, str, str]]:
            return [
                row
                for row in gather()
                if row[0] == 0 and (row[2], row[3]) != (dead_pid, dead_sku)
            ]

        def pick_backed(rows: list[tuple[int, float, str, str]]) -> tuple[str, str] | None:
            if not rows:
                return None
            same_add = [row for row in rows if row[2] == dead_pid]
            pool = same_add or rows
            pool.sort()
            return _choose_from(sess, parsed, [(row[2], row[3]) for row in pool]) or (
                pool[0][2],
                pool[0][3],
            )

        hunt_catalog(sess, parsed, dead_pid, dead_sku)
        view_promising(sess, parsed, limit=12)
        inspectable = [(row[2], row[3]) for row in backed_rows()[:6]]
        if inspectable and sess.remaining() >= 5:
            sess.inspect(inspectable)
        hit = pick_backed(backed_rows())
        if hit:
            return hit
        query = " ".join(
            part
            for part in (
                parsed.brand,
                parsed.resolution
                or (f"{parsed.screen_inch:g} inch" if parsed.screen_inch else ""),
                parsed.category,
            )
            if part
        ) or (parsed.category or "device")
        sess.search(query, k=10, force=True)
        view_promising(sess, parsed, limit=12)
        inspectable = [(row[2], row[3]) for row in backed_rows()[:6]]
        if inspectable and sess.remaining() >= 4:
            sess.inspect(inspectable)
        return pick_backed(backed_rows())

    _commit_and_settle(
        session,
        goal,
        pid,
        sku,
        recover=recover_just,
        justification=build_justification(session, goal, pid, sku),
    )


def run_generic(session: Session, goal: Goal) -> None:
    for query in planned_searches(
        session, goal, [goal.phrase or goal.category or "in stock"]
    ):
        session.search(query, k=10)
    session.view(list(session.products)[:6])
    picks = []
    for product in session.products.values():
        for variant in product.variants:
            if live_ok(goal, variant):
                picks.append((variant.price or 0.0, product.product_id, variant.sku))
    if not picks:
        return
    picks.sort()
    chosen = _choose_from(session, goal, [(pid, sku) for _price, pid, sku in picks])
    if chosen is None:
        chosen = (picks[0][1], picks[0][2])
    _commit_and_settle(session, goal, chosen[0], chosen[1], recover=recover_any)


_ENGINES = {
    "ranking": run_ranking,
    "retrieval_recall": run_retrieval,
    "constraint_satisfaction": run_constraint,
    "intent_decomposition": run_intent,
    "preference_reasoning": run_preference,
    "recovery": run_recovery,
    "justification": run_justification,
}


def _ensure_closed(session: Session, goal: Goal) -> None:
    if session.ordered or session.done or session.remaining() < 1:
        return
    apply_shopper(session, goal)
    pid = sku = ""
    if session.commits:
        pid, sku = session.commits[-1]
        if goal.family not in _OWNED_RECOVER and (
            event_killed(session, pid, sku, goal)
            or not candidate_ok(session, goal, pid, sku)
        ):
            hunt_catalog(session, goal, pid, sku)
            recovered = _best_live(session, goal, pid, sku)
            if recovered is None:
                for alt_pid, alt_sku in reversed(session.commits):
                    if candidate_ok(session, goal, alt_pid, alt_sku):
                        recovered = (alt_pid, alt_sku)
                        break
            if recovered is not None:
                pid, sku = recovered
                _switch_to(session, pid, sku)
    elif goal.family == "constraint_satisfaction":
        hunt_catalog(session, goal)
        open_budget_shelf(session, goal, limit=12)
        inspectable = []
        for product in session.products.values():
            if goal.brand and not brand_ok(goal.brand, product):
                continue
            if part_listing(product):
                continue
            for variant in product.variants:
                if live_ok(goal, variant):
                    inspectable.append((product.product_id, variant.sku))
        if inspectable and session.remaining() >= 5:
            session.inspect(inspectable[:6])
        rescued = recover_any(session, goal, "", "") or _live_pick(session, goal)
        if rescued is None:
            loose = _loose_live(session, goal)
            if loose:
                rescued = loose[0]
        if rescued is not None:
            pid, sku = rescued
            _switch_to(session, pid, sku)
    elif goal.family not in _OWNED_RECOVER:
        hunt_catalog(session, goal)
        rescued = _best_live(session, goal)
        if rescued is None and session.remaining() >= 2:
            _spec_search(session, goal)
            rescued = _best_live(session, goal)
        if rescued is not None:
            pid, sku = rescued
            _switch_to(session, pid, sku)
    if not pid:
        if session.llm_calls == 0:
            llm_plan(session, goal)
        return
    _place_final(session, goal, pid, sku)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    session = Session(problem_data)
    goal = parse_goal(session)
    engine = _ENGINES.get(goal.family, run_generic)
    try:
        llm_plan(session, goal)
        engine(session, goal)
    except RuntimeError:
        pass
    if session.llm_calls == 0:
        llm_plan(session, goal)
        llm_chat(
            "You are a shopping agent. Decide whether a catalog order can be placed.",
            f"Family: {goal.family}\nRequest: {goal.raw.split('Task rules:')[0][:900]}",
            session=session,
        )
    try:
        _ensure_closed(session, goal)
    except RuntimeError:
        pass
    return session.dialogue


__all__ = ["agent_main"]
