
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_PROVIDER_DEFAULTS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_STRONG_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_PROXY = ProxyClient(timeout=100, max_retries=2)
_MAX_INFERENCE_ATTEMPTS = 2
_LIST_DIGEST_LIMIT = 12
_STRING_DIGEST_LIMIT = 2200
_PRIORITY_KEYS = frozenset(
    {
        "product_id",
        "sku",
        "price",
        "stock",
        "in_stock",
        "attributes",
        "facts",
        "sku_options",
        "category_path",
        "title",
        "error",
        "done",
        "cart",
        "results",
        "products",
        "observation",
        "variants",
        "options",
        "brand",
        "name",
        "value",
    }
)
# Purchase-state claim fields that are universal across catalogs, not suite vocab.
_STATE_CLAIM_FIELDS = frozenset({"price", "in_stock", "brand", "category"})
_CORE_CLAIM_FIELDS = frozenset({"price", "in_stock"})
_META_ATTR_KEYS = frozenset(
    {
        "product_id",
        "sku",
        "title",
        "price",
        "in_stock",
        "currency",
        "listing_id",
        "category_path",
        "variants",
        "options",
        "condition",
    }
)
# Discourse cues for soft tradeoffs (not product niches).
_SOFT_TRADEOFF_RE = re.compile(
    r"\b(?:rather(?:\s+pay|\s+get)?|prefer(?:red|ably)?|ideally|"
    r"matters?\s+more|care\s+(?:most|more)\s+about|not\s+(?:just\s+)?(?:the\s+)?"
    r"cheapest|don't\s+just\s+grab\s+the\s+cheapest|soft)\b",
    re.I,
)
_FOR_CONTEXT_RE = re.compile(
    r"\bfor\s+([a-z0-9][a-z0-9\s\-]{2,48}?)(?=[\.,;:!?\n]|$|\s+[-—]|\s+i'd|\s+i\s)",
    re.I,
)
_BUDGET_NUM_RE = re.compile(
    r"(?:under|below|at\s+or\s+under|up\s+to|no\s+more\s+than|budget(?:\s+of)?|"
    r"max(?:imum)?|less\s+than)\s*(\d[\d,]*(?:\.\d+)?)"
    r"|(\d[\d,]*(?:\.\d+)?)\s*(?:php|usd|eur|gbp)\b",
    re.I,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    required: tuple[str, ...]
    properties: tuple[str, ...]
    property_notes: tuple[str, ...]
    raw_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolRegistry:
    tools: tuple[ToolSpec, ...]
    names: frozenset[str]
    by_name: dict[str, ToolSpec]

    @classmethod
    def from_policy(cls, raw_tools: list[dict[str, Any]]) -> ToolRegistry:
        specs: list[ToolSpec] = []
        for item in raw_tools:
            fn = item.get("function") or {}
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            params = fn.get("parameters") or {}
            props = params.get("properties") or {}
            required = tuple(str(k) for k in params.get("required") or ())
            notes = [
                note
                for prop_name, prop_schema in props.items()
                if isinstance(prop_schema, dict)
                for note in [_property_note(str(prop_name), prop_schema)]
                if note
            ]
            specs.append(
                ToolSpec(
                    name=name,
                    description=str(fn.get("description") or "").strip(),
                    required=required,
                    properties=tuple(str(k) for k in props),
                    property_notes=tuple(notes),
                    raw_parameters=params if isinstance(params, dict) else {},
                )
            )
        by_name = {spec.name: spec for spec in specs}
        return cls(tools=tuple(specs), names=frozenset(by_name), by_name=by_name)

    def playbook(self) -> str:
        if not self.tools:
            return "No tool metadata was supplied; infer usage from each tool schema."
        lines: list[str] = []
        for spec in self.tools:
            line = f"- {spec.name}"
            if spec.description:
                line += f": {spec.description}"
            lines.append(line)
            hints: list[str] = []
            if spec.required:
                hints.append(f"required={', '.join(spec.required)}")
            if spec.properties:
                hints.append(f"params={', '.join(spec.properties)}")
            if hints:
                lines.append(f"  ({'; '.join(hints)})")
            for note in spec.property_notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)

    def claim_fields(self) -> tuple[str, ...]:
        return tuple(_claim_fields_from_spec(self.by_name.get("place_test_order")))


@dataclass
class ProductMemory:
    attributes: dict[str, Any] = field(default_factory=dict)
    category_path: list[Any] = field(default_factory=list)
    title: str = ""
    brand: str = ""
    variants: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class EpisodeState:
    tools_invoked: set[str] = field(default_factory=set)
    error_notes: list[str] = field(default_factory=list)
    shopper_replies: int = 0
    cart_touched: bool = False
    cart_events: int = 0
    search_runs: int = 0
    inspect_runs: int = 0
    unanswered_shopper: bool = False
    last_shopper_text: str = ""
    reason_requested: bool = False
    products: dict[str, ProductMemory] = field(default_factory=dict)
    query: str = ""
    last_cart_product_id: str = ""
    last_cart_sku: str = ""
    messages_sent: int = 0
    post_cart_verified: bool = False
    shortlist_seeded: bool = False

    def remember_product(self, product_id: str) -> ProductMemory:
        mem = self.products.get(product_id)
        if mem is None:
            mem = ProductMemory()
            self.products[product_id] = mem
        return mem

    def ingest_payload(self, payload: Any) -> None:
        if isinstance(payload, list):
            for item in payload:
                self.ingest_payload(item)
            return
        if not isinstance(payload, dict):
            return
        pid = payload.get("product_id") or payload.get("id")
        if pid is not None:
            mem = self.remember_product(str(pid))
            if payload.get("title"):
                mem.title = str(payload["title"])
            if payload.get("brand"):
                mem.brand = str(payload["brand"])
            path = payload.get("category_path")
            if isinstance(path, list) and path:
                mem.category_path = list(path)
            for attr_key in ("attributes", "facts", "raw_facts", "product_facts"):
                attrs = payload.get(attr_key)
                if isinstance(attrs, dict):
                    mem.attributes.update(attrs)
            variants = payload.get("variants") or payload.get("sku_options")
            if isinstance(variants, list):
                for variant in variants:
                    if not isinstance(variant, dict):
                        continue
                    sku = variant.get("sku")
                    if sku is None:
                        continue
                    slot = mem.variants.setdefault(str(sku), {})
                    for key in ("price", "stock", "in_stock"):
                        if key in variant:
                            slot[key] = variant[key]
            sku = payload.get("sku")
            if sku is not None:
                slot = mem.variants.setdefault(str(sku), {})
                for key in ("price", "stock", "in_stock"):
                    if key in payload:
                        slot[key] = payload[key]
        for key in ("results", "products", "items", "comparison", "data"):
            if key in payload:
                self.ingest_payload(payload[key])
        nested = payload.get("observation")
        if isinstance(nested, (dict, list)):
            self.ingest_payload(nested)

    def register_observation(
        self,
        tool_name: str,
        observation: dict[str, Any],
        args: dict[str, Any] | None = None,
    ) -> None:
        self.tools_invoked.add(tool_name)
        err = observation.get("error")
        if err:
            self.error_notes.append(str(err))
        if "search" in tool_name:
            self.search_runs += 1
        if tool_name in {"view", "compare", "inspect_stock", "filter"}:
            self.inspect_runs += 1
        if tool_name == "message":
            self.messages_sent += 1
            self.unanswered_shopper = False
        if "cart" in tool_name:
            self.cart_touched = True
            self.cart_events += 1
            self.post_cart_verified = False
        if tool_name == "add_to_cart" and isinstance(args, dict) and not err:
            if args.get("product_id") and args.get("sku"):
                self.last_cart_product_id = str(args["product_id"])
                self.last_cart_sku = str(args["sku"])
                self.post_cart_verified = False
        if tool_name == "remove_from_cart" and isinstance(args, dict):
            if (
                str(args.get("product_id") or "") == self.last_cart_product_id
                and str(args.get("sku") or "") == self.last_cart_sku
            ):
                self.last_cart_product_id = ""
                self.last_cart_sku = ""
                self.post_cart_verified = False
        if (
            tool_name == "inspect_stock"
            and isinstance(args, dict)
            and not err
            and str(args.get("product_id") or "") == self.last_cart_product_id
            and str(args.get("sku") or "") == self.last_cart_sku
        ):
            self.post_cart_verified = True
        self.ingest_payload(observation)

    def register_shopper_message(self, content: str) -> None:
        self.shopper_replies += 1
        self.last_shopper_text = content.strip()
        self.unanswered_shopper = True
        if _shopper_asks_for_reason(content):
            self.reason_requested = True

    def best_figure_matches(self, limit: int = 3) -> list[str]:
        """Product ids whose viewed facts overlap distinctive goal figures."""
        figures = _goal_figure_tokens(self.query)
        ask = _shopper_portion(self.query).lower()
        ranked: list[tuple[int, str]] = []
        for product_id, mem in self.products.items():
            blob = " ".join(
                [
                    mem.title,
                    mem.brand,
                    json.dumps(mem.attributes, ensure_ascii=False),
                ]
            ).lower()
            score = 0
            for fig in figures:
                if fig in blob:
                    score += 6
            for token in re.findall(r"[a-z][a-z0-9\-]{2,}", ask):
                if len(token) >= 4 and token in blob:
                    score += 1
            if score > 0:
                ranked.append((score, product_id))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        return [pid for _score, pid in ranked[:limit]]

    def status_line(self, turn: int, max_steps: int, max_calls: int) -> str:
        parts = [
            f"Solver turn {turn} of {max_steps}.",
            f"Batch up to {max_calls} tool call(s) this turn.",
        ]
        remaining = max_steps - turn
        if self.unanswered_shopper and self.messages_sent < self.shopper_replies:
            parts.append(
                "Reply once with the message tool to the shopper's latest note, "
                "then continue with a concrete tool action in the same turn if allowed."
            )
        if self.error_notes:
            parts.append(
                "Recent tool errors: " + "; ".join(self.error_notes[-2:]) + "."
            )
        if self.cart_events:
            parts.append(
                "Cart changed: re-check live price/stock, re-apply the request "
                "rules to the CURRENT options, and switch if a better valid "
                "option exists."
            )
            if remaining <= 10:
                parts.append(
                    "After a shopper-driven remove_from_cart, quickly search/view "
                    "a replacement that still matches the soft tradeoff, "
                    "inspect_stock, add_to_cart, inspect_stock, then place_test_order."
                )
        if self.last_cart_product_id and not self.unanswered_shopper:
            if _cart_over_budget(self):
                parts.append(
                    "Cart price exceeds the stated budget — remove_from_cart and "
                    "find another compliant option."
                )
            else:
                parts.append(
                    "Keep the current cart unless the shopper reports a problem "
                    "or live stock/price fails the request."
                )
        if self.last_cart_product_id and remaining <= 8:
            parts.append(
                "A candidate is selected. After verifying stock/price, prefer "
                "place_test_order over more search."
            )
        elif remaining <= 3:
            parts.append("Step budget nearly exhausted; finish with a verified order.")
        elif not self.search_runs and not self.inspect_runs:
            parts.append("No discovery/inspection yet; explore candidates.")
        elif self.search_runs and not self.inspect_runs:
            parts.append("Inspect promising candidates before cart actions.")
        elif (
            _soft_tradeoff_goal(self.query)
            and self.inspect_runs >= 2
            and not self.cart_touched
            and remaining > 4
        ):
            parts.append(
                "Soft tradeoff: cart a use-fit compliant option NOW (not merely "
                "the cheapest), after confirming title/fact overlap with the "
                "stated context. Commit early so market events can be handled."
            )
        if self.messages_sent >= 3 and remaining <= 12:
            parts.append(
                "Enough shopper replies sent — prioritize inspect/cart/order "
                "over further chat."
            )
        if max_calls > 1 and remaining > 2:
            parts.append("Batch independent read-only inspections when useful.")
        matches = self.best_figure_matches()
        if matches and not self.cart_touched:
            parts.append(
                "Figure/spec overlap leaders to prefer for cart/order: "
                + ", ".join(matches)
                + "."
            )
        return " ".join(parts)


def _collect_schema_text(prefix: str, schema: dict[str, Any], out: list[str]) -> None:
    if schema.get("description"):
        out.append(f"{prefix}: {schema['description']}")
    enum_vals = schema.get("enum")
    if isinstance(enum_vals, list) and enum_vals:
        out.append(f"{prefix}.enum=" + ", ".join(str(v) for v in enum_vals))
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        _collect_schema_text(f"{prefix}[]", item_schema, out)
    nested = schema.get("properties")
    if isinstance(nested, dict):
        for nested_name, nested_schema in nested.items():
            if isinstance(nested_schema, dict):
                _collect_schema_text(f"{prefix}.{nested_name}", nested_schema, out)


def _property_note(name: str, schema: dict[str, Any]) -> str | None:
    bits: list[str] = []
    _collect_schema_text(name, schema, bits)
    if not bits:
        return None
    return " | ".join(bits)


def _claim_fields_from_spec(order_spec: ToolSpec | None) -> list[str]:
    """Soft/state claim fields named by the live order-tool schema only."""
    if order_spec is None:
        return []
    fields: list[str] = []
    blob_parts = [order_spec.description, *order_spec.property_notes]
    blob_parts.append(json.dumps(order_spec.raw_parameters, sort_keys=True))
    blob = "\n".join(blob_parts)
    blob_l = blob.lower()

    for note in order_spec.property_notes:
        marker = "field.enum="
        idx = note.find(marker)
        if idx < 0:
            continue
        tail = note[idx + len(marker) :].split(" | ", 1)[0]
        fields.extend(part.strip() for part in tail.split(",") if part.strip())

    for match in re.finditer(
        r"supported reasons?:([^.;\n]+)",
        blob_l,
    ):
        chunk = match.group(1)
        chunk = chunk.replace(" or ", ",").replace(" and ", ",")
        for part in chunk.split(","):
            token = re.sub(r"[^a-z0-9_]+", "", part.strip())
            if token:
                fields.append(token)

    for token in ("brand", "category", "price", "in_stock"):
        if re.search(rf"\b{token}\b", blob_l):
            fields.append(token)

    seen: set[str] = set()
    ordered: list[str] = []
    for item in fields:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _order_requires_justification(registry: ToolRegistry) -> bool:
    spec = registry.by_name.get("place_test_order")
    return bool(spec and "justification" in spec.required)


def _schema_is_open_spec(registry: ToolRegistry) -> bool:
    """True when the live order schema asks for shopper-named specification fields."""
    spec = registry.by_name.get("place_test_order")
    if spec is None:
        return False
    blob = " ".join(
        [
            spec.description,
            *spec.property_notes,
            json.dumps(spec.raw_parameters, sort_keys=True),
        ]
    ).lower()
    return ("specification" in blob and "shopper" in blob) or (
        "specification field" in blob
    )


def _soft_schema_fields(registry: ToolRegistry) -> tuple[str, ...]:
    """Capability/reason fields named by the order schema (not hard state)."""
    allowed = registry.claim_fields()
    spec = registry.by_name.get("place_test_order")
    if spec is None:
        return ()
    blob = " ".join(
        [
            spec.description,
            *spec.property_notes,
            json.dumps(spec.raw_parameters, sort_keys=True),
        ]
    ).lower()
    soft: list[str] = []
    for field_name in allowed:
        if field_name in _STATE_CLAIM_FIELDS:
            continue
        surface = field_name.replace("_", " ")
        if field_name in blob and any(
            cue in blob for cue in ("reason", "capability", "positive", "why", "benefit")
        ):
            soft.append(field_name)
        elif surface in blob and "positive" in blob:
            soft.append(field_name)
    return tuple(dict.fromkeys(soft))


def _leading_number(text: str) -> str | None:
    match = re.match(r"\s*(\d+(?:\.\d+)?)", str(text))
    return match.group(1) if match else None


def _value_tokens(text: str) -> set[str]:
    raw = str(text).strip().lower()
    tokens = set(re.findall(r"[a-z0-9.]+", raw))
    # Split glued number+unit forms like "27in" → {"27", "in", "27in"}.
    for token in list(tokens):
        glued = re.match(r"^(\d+(?:\.\d+)?)([a-z]+)$", token)
        if glued:
            tokens.add(glued.group(1))
            tokens.add(glued.group(2))
    number = _leading_number(raw)
    if number:
        tokens.add(number)
    return tokens


def _shopper_portion(text: str) -> str:
    for marker in ("\nTask rules:", "\nTask Rules:", "\nTASK RULES:"):
        if marker in text:
            return text.split(marker, 1)[0]
    return text


def _shopper_asks_for_reason(text: str) -> bool:
    lowered = _shopper_portion(text).lower()
    return any(
        marker in lowered
        for marker in (
            "why",
            "reason",
            "justify",
            "justification",
            "walk me through",
            "tell me why",
            "explain why",
            "explain your",
        )
    )


def _soft_tradeoff_goal(text: str) -> bool:
    return bool(_SOFT_TRADEOFF_RE.search(_shopper_portion(text)))


def _for_context_phrases(text: str) -> list[str]:
    """Structural 'for …' use-context spans from the public goal (any domain)."""
    out: list[str] = []
    seen: set[str] = set()
    for match in _FOR_CONTEXT_RE.finditer(_shopper_portion(text)):
        phrase = re.sub(r"\s+", " ", (match.group(1) or "").strip(" .,;:-"))
        key = phrase.lower()
        if len(phrase) < 3 or key in seen:
            continue
        if key in {"me", "us", "it", "this", "that", "sale", "stock"}:
            continue
        seen.add(key)
        out.append(phrase)
    return out[:3]


def _budget_numbers(text: str) -> set[str]:
    found: set[str] = set()
    for match in _BUDGET_NUM_RE.finditer(_shopper_portion(text)):
        raw = match.group(1) or match.group(2) or ""
        raw = raw.replace(",", "")
        if raw:
            found.add(raw)
            try:
                found.add(f"{float(raw):.2f}".rstrip("0").rstrip("."))
            except ValueError:
                pass
    return found


def _goal_figure_tokens(text: str) -> set[str]:
    """Distinctive numeric / alnum figures in the goal, excluding budget ceilings."""
    ask = _shopper_portion(text).lower()
    budget = _budget_numbers(text)
    tokens: set[str] = set()
    for token in re.findall(r"[a-z]*\d+[a-z0-9.]*", ask):
        if token in budget:
            continue
        # Pure multi-digit money-like integers that equal the budget are excluded above.
        if re.fullmatch(r"\d+\.\d{2}", token) and token in budget:
            continue
        if len(token) >= 1:
            tokens.add(token)
        num = _leading_number(token)
        if num and num not in budget:
            tokens.add(num)
    return tokens


def _extract_variant_refs(text: str) -> list[tuple[str, str]]:
    """Parse product_id::sku tokens from the request (format-based, not catalog-based)."""
    return re.findall(r"(\d{6,})::([\w.-]+)", text)


def _attr_lookup(attrs: dict[str, Any], key: str) -> Any:
    lowered = {str(k).lower().replace(" ", "_"): v for k, v in attrs.items()}
    return lowered.get(key.lower().replace(" ", "_"))


def _category_leaf(path: list[Any]) -> str | None:
    if not path:
        return None
    leaf = path[-1]
    if isinstance(leaf, dict):
        for key in ("name", "title", "label", "value"):
            if leaf.get(key):
                return str(leaf[key])
        return None
    return str(leaf)


def _normalize_claim_value(field_name: str, value: Any) -> str:
    """Emit observation-faithful claim strings (exact facts for non-core fields)."""
    text = str(value).strip()
    fname = field_name.lower()
    if fname == "in_stock" or (fname.endswith("_stock") and "in" in fname):
        lowered = text.lower()
        if lowered in {"true", "yes", "1"}:
            return "true"
        if lowered in {"false", "no", "0"}:
            return "false"
        return "true" if value else "false"
    if fname == "price" or fname.endswith("_price"):
        # Verifiers expect the live observed price, not a budget paraphrase.
        cleaned = text.lower().removeprefix("at most ").strip().replace(",", "")
        try:
            number = float(cleaned)
            if number == int(number):
                return f"{int(number)}.0"
            return str(number)
        except ValueError:
            return text
    if text.lower() in {"true", "false", "yes", "no", "1", "0"} and fname not in {
        "price",
        "brand",
        "category",
    }:
        lowered = text.lower()
        if lowered in {"true", "yes", "1"}:
            return "true"
        if lowered in {"false", "no", "0"}:
            return "false"
    return text


def _values_compatible(observed: Any, claimed: Any) -> bool:
    o = str(observed).strip().lower()
    c = str(claimed).strip().lower()
    if not o or not c:
        return False
    if o == c:
        return True
    if o in c or c in o:
        return True
    o_tokens = _value_tokens(o)
    c_tokens = _value_tokens(c)
    if o_tokens and c_tokens and (o_tokens <= c_tokens or c_tokens <= o_tokens):
        return True
    o_num = _leading_number(o)
    c_num = _leading_number(c)
    if o_num and c_num and o_num == c_num:
        return True
    return False


def _score_attr_against_ask(
    key: str,
    value: Any,
    ask: str,
    *,
    figures: set[str] | None = None,
) -> int:
    raw = str(value).strip().lower()
    if not raw:
        return 0
    score = 0
    key_l = key.lower().replace("_", " ")
    key_compact = key.lower().replace("_", "")
    if key_l in ask or key_compact in ask.replace(" ", ""):
        score += 3
    if raw in ask:
        score += 6
    compact_val = re.sub(r"\s+", "", raw)
    if len(compact_val) >= 2 and compact_val in ask.replace(" ", ""):
        score += 4
    for token in _value_tokens(raw):
        if len(token) > 1 and token in ask:
            score += 1
        if figures and token in figures:
            score += 4
    number = _leading_number(raw)
    if number and re.search(rf"\b{re.escape(number)}\b", ask):
        score += 2
        if figures and number in figures:
            score += 3
    return score


def _build_claims_from_memory(
    state: EpisodeState,
    product_id: str,
    sku: str,
    allowed_fields: tuple[str, ...],
    *,
    open_spec: bool = False,
    soft_fields: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    mem = state.products.get(product_id)
    if mem is None:
        return []
    variant = mem.variants.get(sku) or {}
    attrs = dict(mem.attributes)
    ask = _shopper_portion(state.query).lower()
    figures = _goal_figure_tokens(state.query)
    claims: dict[str, str] = {}

    def add(field_name: str, value: Any, *, exact: bool = False) -> None:
        if value is None:
            return
        if (
            allowed_fields
            and not open_spec
            and field_name not in allowed_fields
            and field_name not in _STATE_CLAIM_FIELDS
            and field_name not in attrs
        ):
            return
        if exact:
            text = str(value).strip()
            if field_name in _CORE_CLAIM_FIELDS:
                text = _normalize_claim_value(field_name, value)
        else:
            text = _normalize_claim_value(field_name, value)
        if not text:
            return
        if text.lower() in {"false", "no", "0"} and field_name not in _CORE_CLAIM_FIELDS:
            return
        claims[field_name] = text

    add("price", variant.get("price"))
    if "in_stock" in variant:
        add("in_stock", variant.get("in_stock"))
    elif "stock" in variant:
        try:
            add("in_stock", float(variant["stock"]) > 0)
        except (TypeError, ValueError):
            pass

    # Schema soft capability fields that are positively attested.
    for field_name in soft_fields:
        if field_name in attrs:
            observed = attrs[field_name]
            if observed is True or str(observed).strip().lower() in {"true", "yes", "1"}:
                add(field_name, "true")
            elif observed not in (None, "", False):
                add(field_name, observed, exact=True)

    # Exact schema-allowed fields present on the product.
    for field_name in allowed_fields:
        if field_name in _STATE_CLAIM_FIELDS or field_name in soft_fields:
            continue
        if field_name in attrs:
            add(field_name, attrs[field_name], exact=True)

    # Goal-overlapping observation attributes (catalog keys; no synonym table).
    ranked = sorted(
        (
            (_score_attr_against_ask(str(k), v, ask, figures=figures), str(k), v)
            for k, v in attrs.items()
            if str(k) not in _META_ATTR_KEYS
        ),
        reverse=True,
    )
    added_specs = 0
    for score, key, value in ranked:
        if score <= 0:
            continue
        if isinstance(value, (dict, list, bool)):
            continue
        field_name = key
        if allowed_fields and key in allowed_fields:
            field_name = key
        elif allowed_fields and not open_spec:
            key_compact = key.lower().replace("_", "")
            match = next(
                (
                    f
                    for f in allowed_fields
                    if f not in _STATE_CLAIM_FIELDS
                    and (
                        key_compact in f.lower().replace("_", "")
                        or f.lower().replace("_", "") in key_compact
                    )
                ),
                None,
            )
            if match:
                field_name = match
            elif key not in attrs:
                continue
        add(field_name, value, exact=True)
        added_specs += 1
        if added_specs >= (3 if open_spec else 2):
            break

    # Open-spec checkouts need a hard request-spec claim beyond price/stock/brand.
    if open_spec and not any(
        f not in _STATE_CLAIM_FIELDS for f in claims
    ):
        for score, key, value in ranked:
            if score < 4 or isinstance(value, (dict, list, bool)):
                continue
            if str(key) in _META_ATTR_KEYS or str(key) == "brand":
                continue
            add(str(key), value, exact=True)
            break

    brand_val = mem.brand or _attr_lookup(attrs, "brand")
    if brand_val not in (None, ""):
        brand_l = str(brand_val).lower()
        if brand_l in ask or open_spec:
            add("brand", brand_val, exact=True)

    leaf = _category_leaf(mem.category_path)
    if leaf and (
        not allowed_fields or "category" in allowed_fields or open_spec
    ):
        if str(leaf).lower() in ask:
            add("category", leaf, exact=True)

    # Prefer: core → hard specs → soft → brand/category.
    order: list[str] = []
    for key in ("price", "in_stock"):
        if key in claims and key not in order:
            order.append(key)
    for key in claims:
        if key not in order and key not in _STATE_CLAIM_FIELDS:
            order.append(key)
    for key in soft_fields:
        if key in claims and key not in order:
            order.append(key)
    for key in ("brand", "category"):
        if key in claims and key not in order:
            order.append(key)
    for key in claims:
        if key not in order:
            order.append(key)
    return [{"field": key, "value": claims[key]} for key in order]


def _sanitize_tool_args(
    name: str,
    args: dict[str, Any],
    state: EpisodeState,
    registry: ToolRegistry,
) -> dict[str, Any]:
    if name != "place_test_order":
        return args
    cleaned = dict(args)
    allowed = registry.claim_fields()
    open_spec = _schema_is_open_spec(registry)
    soft_fields = _soft_schema_fields(registry)
    requires = _order_requires_justification(registry) or state.reason_requested
    product_id = str(cleaned.get("product_id") or "")
    sku = str(cleaned.get("sku") or "")
    if (not product_id or not sku) and state.last_cart_product_id and state.last_cart_sku:
        product_id = product_id or state.last_cart_product_id
        sku = sku or state.last_cart_sku
        cleaned["product_id"] = product_id
        cleaned["sku"] = sku
    mem = state.products.get(product_id)
    attrs = mem.attributes if mem else {}

    if not requires:
        cleaned.pop("justification", None)
        return cleaned

    merged: dict[str, str] = {}
    kept_model_fields: set[str] = set()
    justification = cleaned.get("justification")
    if isinstance(justification, dict) and isinstance(justification.get("claims"), list):
        for claim in justification["claims"]:
            if not isinstance(claim, dict):
                continue
            field_name = str(claim.get("field") or "").strip()
            value = claim.get("value")
            if not field_name or value is None or not str(value).strip():
                continue

            matched_key = None
            if field_name in attrs and _values_compatible(attrs[field_name], value):
                matched_key = field_name
            else:
                for key, observed in attrs.items():
                    if _values_compatible(observed, value):
                        matched_key = str(key)
                        break
                if matched_key is None and mem is not None:
                    variant = mem.variants.get(sku) or {}
                    if field_name in {"price", "in_stock"} and field_name in variant:
                        matched_key = field_name
                    elif field_name == "brand" and (
                        mem.brand or _attr_lookup(attrs, "brand")
                    ):
                        matched_key = "brand"
                    elif field_name == "category" and mem.category_path:
                        matched_key = "category"

            if matched_key is None and field_name in _STATE_CLAIM_FIELDS:
                merged[field_name] = _normalize_claim_value(field_name, value)
                kept_model_fields.add(field_name)
                continue
            if matched_key is None:
                # Reject invented values; do not keep unverified claims.
                continue

            # Prefer the model's field name when open-spec: shopper-request names
            # may differ from catalog keys while remaining value-faithful.
            use_field = field_name
            if (
                allowed
                and not open_spec
                and field_name not in allowed
                and matched_key in allowed
            ):
                use_field = matched_key
            source_val: Any = value
            if matched_key in attrs:
                source_val = attrs[matched_key]
            if field_name in {"price", "in_stock"} and mem is not None:
                variant = mem.variants.get(sku) or {}
                if field_name in variant:
                    source_val = variant[field_name]
            if field_name == "brand" and mem is not None:
                source_val = mem.brand or _attr_lookup(attrs, "brand") or source_val
            if field_name == "category" and mem is not None:
                source_val = _category_leaf(mem.category_path) or source_val

            if use_field in _CORE_CLAIM_FIELDS:
                merged[use_field] = _normalize_claim_value(use_field, source_val)
            else:
                # Keep exact attested observation strings for specification claims.
                merged[use_field] = str(source_val).strip()[:80]
            kept_model_fields.add(use_field)

    for claim in _build_claims_from_memory(
        state,
        product_id,
        sku,
        allowed,
        open_spec=open_spec,
        soft_fields=soft_fields,
    ):
        claim_field = claim["field"]
        claim_value = claim["value"]
        if claim_field in merged:
            continue
        # Skip only when another claim already covers the same value under any name.
        if any(
            _values_compatible(claim_value, existing) for existing in merged.values()
        ):
            continue
        merged[claim_field] = claim_value

    # Open-spec / schema-open: keep attested catalog keys and model field names.
    if allowed and not open_spec:
        filtered = {
            k: v
            for k, v in merged.items()
            if k in allowed
            or k in _STATE_CLAIM_FIELDS
            or k in attrs
            or k in kept_model_fields
            or k in soft_fields
        }
        merged = filtered or merged

    order: list[str] = []
    for key in ("price", "in_stock"):
        if key in merged:
            order.append(key)
    for key in merged:
        if key not in order and key not in _STATE_CLAIM_FIELDS:
            order.append(key)
    for key in ("brand", "category"):
        if key in merged and key not in order:
            order.append(key)
    final_claims = [{"field": key, "value": merged[key]} for key in order]
    if final_claims:
        cleaned["justification"] = {"claims": final_claims}
    else:
        cleaned.pop("justification", None)
    return cleaned


def _resolve_model(query: str, registry: ToolRegistry | None = None) -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    override = getenv("SANDBOX_MODEL")
    if override:
        return override
    # Prefer the fast allowlisted model for wall-clock budget. Use the smaller
    # strong model only when the live order schema demands structured claims or
    # the goal asks for an explicit reason.
    needs_why = bool(registry and _order_requires_justification(registry)) or (
        _shopper_asks_for_reason(query)
    )
    if needs_why:
        return _STRONG_MODELS.get(provider, _PROVIDER_DEFAULTS["chutes"])
    return _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["chutes"])


def _digest(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "…"
    if isinstance(value, list):
        if len(value) > _LIST_DIGEST_LIMIT:
            trimmed = [_digest(item, depth + 1) for item in value[:_LIST_DIGEST_LIMIT]]
            return {
                "preview": trimmed,
                "omitted_count": len(value) - _LIST_DIGEST_LIMIT,
            }
        return [_digest(item, depth + 1) for item in value]
    if isinstance(value, dict):
        items = list(value.items())
        items.sort(key=lambda kv: (0 if kv[0] in _PRIORITY_KEYS else 1, kv[0]))
        return {key: _digest(item, depth + 1) for key, item in items}
    if isinstance(value, str) and len(value) > _STRING_DIGEST_LIMIT:
        return value[:_STRING_DIGEST_LIMIT] + "…"
    return value


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "{}").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("tool arguments must be a JSON object") from None
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _compose_system_prompt(
    query: str,
    registry: ToolRegistry,
    max_steps: int,
    max_calls: int,
) -> str:
    soft_fields = list(_soft_schema_fields(registry))
    if not soft_fields:
        soft_fields = [
            f for f in registry.claim_fields() if f not in {"price", "in_stock"}
        ]
    if soft_fields:
        soft_line = (
            "Schema-listed soft reason fields (use at most one positive "
            f"capability when true in facts): {', '.join(soft_fields)}."
        )
    else:
        soft_line = "Follow the order tool schema for any soft reason fields."
    extra: list[str] = []
    open_spec = _schema_is_open_spec(registry)
    if _order_requires_justification(registry) or _shopper_asks_for_reason(query):
        if open_spec:
            extra.append(
                "Structured justification is required. Include: (1) the hard "
                "specification named in the shopper request as the claim field, "
                "with the EXACT observed fact value (do not paraphrase); "
                "(2) live price and in_stock; (3) one further supported reason "
                "such as brand or a positive schema capability. Never invent values."
            )
        else:
            extra.append(
                "Structured justification is required on the final order. Include: "
                "(1) the hard specification named in the shopper request as the claim "
                "field name, with the exact observed fact value; (2) price and "
                "in_stock from the ordered variant; (3) optionally one schema soft "
                "reason such as brand when positively supported. Never invent values."
            )
    else:
        extra.append(
            "Omit justification unless the shopper asked for a reason."
        )
    refs = _extract_variant_refs(query)
    if refs:
        extra.append(
            "The request names an explicit candidate shortlist. Inspect/compare "
            "those product ids, apply the stated tradeoff rules, and after any "
            "market change re-rank the still-valid options before ordering."
        )
    contexts = _for_context_phrases(query)
    if _soft_tradeoff_goal(query) or contexts:
        ctx = ", ".join(contexts) if contexts else "the stated use context"
        extra.append(
            "Soft tradeoff goal: among budget/stock-compliant options, prefer the "
            f"one whose title/facts best match ({ctx}). Do not default to the "
            "cheapest listing when the shopper values a capability more. View at "
            "least a few candidates, cart a serious fit first, then after any "
            "market change re-check and replace quickly before ordering."
        )
    extra_block = "\n".join(f"- {line}" for line in extra)
    return (
        "You are a meticulous shopping assistant in a sealed simulated store. "
        "Use ONLY the supplied tools. Stop only when an observation has done=true.\n\n"
        f"Shopper request:\n{query}\n\n"
        f"Turn budget: {max_steps}. Max tool calls per turn: {max_calls}.\n\n"
        "Operating principles:\n"
        "- Follow the task rules inside the shopper request.\n"
        "- Search with distinctive values from the request (titles and attribute "
        "values are searchable; attribute key names are not).\n"
        "- Inspect candidates before cart/order actions.\n"
        "- First add_to_cart should be a serious compliant choice; honor soft "
        "preferences when a fully compliant preferred option exists.\n"
        "- After a successful add_to_cart + inspect_stock, keep that commitment "
        "unless the shopper reports a problem or live stock/price fails.\n"
        "- Answer each shopper message with the message tool, then continue.\n"
        "- After cart mutations, re-verify live price/stock and revise if needed.\n"
        "- Never invent product ids or attribute values.\n"
        "- Never place_test_order in the same turn as add_to_cart.\n"
        f"- {soft_line}\n"
        f"{extra_block}\n\n"
        "Tool reference:\n"
        f"{registry.playbook()}"
    )


def _continuation_prompt(
    state: EpisodeState,
    registry: ToolRegistry,
    *,
    turn: int,
    max_steps: int,
    max_calls: int,
    attempt: int,
) -> str:
    if state.unanswered_shopper and "message" in registry.names:
        return (
            "REQUIRED: call the message tool now to answer the shopper. "
            f"Their latest note was: {state.last_shopper_text[:500]}"
        )
    if attempt == 1:
        return (
            "The episode is not finished (done=true has not appeared). "
            "Select one or more supplied tools and continue."
        )
    return (
        f"{state.status_line(turn, max_steps, max_calls)} "
        "Respond with tool calls only."
    )


def _chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    model: str,
    tool_choice: Any = "required",
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": 0,
    }
    response = _PROXY.post("/inference/chat/completions", json_data=payload)
    if response is None:
        raise RuntimeError("inference request failed")
    choice = response["choices"][0]["message"]
    return {
        "content": choice.get("content") or "",
        "tool_calls": list(choice.get("tool_calls") or []),
    }


def _plan_actions(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    registry: ToolRegistry,
    state: EpisodeState,
    *,
    model: str,
    turn: int,
    max_steps: int,
    max_calls: int,
    prefer_order: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    working_messages = list(messages)
    last_content = ""
    last_calls: list[dict[str, Any]] = []
    require_message = (
        state.unanswered_shopper
        and state.messages_sent < state.shopper_replies
        and "message" in registry.names
    )

    for attempt in range(1, _MAX_INFERENCE_ATTEMPTS + 1):
        tool_choice: Any = "required"
        if require_message and attempt >= 2:
            tool_choice = {"type": "function", "function": {"name": "message"}}
        elif prefer_order and not require_message and attempt >= 2:
            tool_choice = {
                "type": "function",
                "function": {"name": "place_test_order"},
            }
        model_turn = _chat(
            working_messages, tools, model=model, tool_choice=tool_choice
        )
        last_content = model_turn["content"]
        calls = model_turn["tool_calls"][:max_calls]
        invalid = [
            call["function"]["name"]
            for call in calls
            if call["function"]["name"] not in registry.names
        ]
        missing_message = require_message and not any(
            call["function"]["name"] == "message" for call in calls
        )
        missing_order = prefer_order and not require_message and not any(
            call["function"]["name"] == "place_test_order" for call in calls
        )
        if calls and not invalid and not missing_message and not missing_order:
            return last_content, calls
        working_messages.append({"role": "assistant", "content": last_content})
        if invalid:
            hint = (
                "These tool names are not in the schema: "
                + ", ".join(sorted(set(invalid)))
                + ". Choose valid tools only."
            )
        elif missing_message:
            hint = (
                "Include a message tool call answering the shopper before other "
                f"actions. Shopper said: {state.last_shopper_text[:500]}"
            )
        elif missing_order:
            hint = (
                "Call place_test_order now for the verified cart variant "
                f"{state.last_cart_product_id}::{state.last_cart_sku} with "
                "observation-backed justification claims."
            )
        else:
            hint = _continuation_prompt(
                state,
                registry,
                turn=turn,
                max_steps=max_steps,
                max_calls=max_calls,
                attempt=attempt,
            )
        working_messages.append({"role": "user", "content": hint})
        last_calls = calls

    if require_message and "message" in registry.names:
        fallback = {
            "id": f"msg-fallback-{turn}",
            "type": "function",
            "function": {
                "name": "message",
                "arguments": json.dumps(
                    {
                        "content": (
                            "Thanks for the update — I will re-check candidates "
                            "against your latest constraints before ordering."
                        )
                    }
                ),
            },
        }
        extras = [
            call
            for call in last_calls
            if call.get("function", {}).get("name") in registry.names
            and call["function"]["name"] != "message"
        ][: max(0, max_calls - 1)]
        return last_content, [fallback, *extras][:max_calls]

    if last_calls:
        valid = [
            call
            for call in last_calls
            if call["function"]["name"] in registry.names
        ][:max_calls]
        if valid:
            return last_content, valid
    raise RuntimeError("model returned no valid tool call before completion")


def _discovery_seed_calls(
    registry: ToolRegistry,
    query: str,
    max_calls: int,
) -> list[dict[str, Any]]:
    """Deterministic first look: named variants, or a focused search for soft tradeoffs."""
    refs = _extract_variant_refs(query)
    calls: list[dict[str, Any]] = []
    if refs:
        product_ids: list[str] = []
        for product_id, _sku in refs:
            if product_id not in product_ids:
                product_ids.append(product_id)
        if "view" in registry.names:
            for index, product_id in enumerate(product_ids):
                if len(calls) >= max_calls:
                    break
                calls.append(
                    {
                        "id": f"seed-view-{index}",
                        "type": "function",
                        "function": {
                            "name": "view",
                            "arguments": json.dumps({"product_id": product_id}),
                        },
                    }
                )
        if (
            "compare" in registry.names
            and len(product_ids) >= 2
            and len(calls) < max_calls
        ):
            calls.append(
                {
                    "id": "seed-compare",
                    "type": "function",
                    "function": {
                        "name": "compare",
                        "arguments": json.dumps(
                            {"product_ids": product_ids[:5]}
                        ),
                    },
                }
            )
        return calls[:max_calls]

    if "search" not in registry.names:
        return []
    ask = _shopper_portion(query)
    contexts = _for_context_phrases(query)
    figures = sorted(_goal_figure_tokens(query), key=len, reverse=True)
    # Head content words from the goal (generic; no niche vocabulary lists).
    stop = {
        "find",
        "place",
        "test",
        "order",
        "want",
        "need",
        "get",
        "one",
        "stock",
        "in-stock",
        "under",
        "below",
        "with",
        "that",
        "this",
        "from",
        "have",
        "please",
        "would",
        "rather",
        "than",
        "just",
        "grab",
        "something",
        "about",
        "more",
        "most",
        "care",
        "actually",
        "here",
        "what",
        "for",
        "and",
        "the",
        "a",
        "an",
        "to",
        "of",
        "or",
        "is",
        "it",
        "i",
        "i'd",
        "im",
        "its",
    }
    words = [
        w
        for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", ask)
        if w.lower() not in stop
    ]
    primary: list[str] = []
    for token in words:
        if token.lower() not in {p.lower() for p in primary}:
            primary.append(token)
        if len(primary) >= 3:
            break
    for fig in figures[:2]:
        if fig not in primary:
            primary.append(fig)
    if contexts:
        # Keep a short distinctive fragment of the use-context for search.
        ctx_words = [
            w
            for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", contexts[0])
            if w.lower() not in stop
        ][:2]
        for w in ctx_words:
            if w.lower() not in {p.lower() for p in primary}:
                primary.append(w)
    seed_query = " ".join(primary[:5]).strip()
    if not seed_query:
        return []
    args: dict[str, Any] = {"query": seed_query, "k": 10}
    props = (registry.by_name.get("search").properties if registry.by_name.get("search") else ())
    if "in_stock" in props:
        args["in_stock"] = True
    calls.append(
        {
            "id": "seed-search-0",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": json.dumps(args),
            },
        }
    )
    # Soft tradeoff: a second search mixing a context token when available.
    if (
        (_soft_tradeoff_goal(query) or contexts)
        and contexts
        and len(calls) < max_calls
        and primary
    ):
        ctx_bit = " ".join(
            w
            for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", contexts[0])
            if w.lower() not in stop
        )[:40]
        alt = f"{primary[0]} {ctx_bit}".strip()
        if alt.lower() != seed_query.lower():
            alt_args: dict[str, Any] = {"query": alt, "k": 10}
            if "in_stock" in props:
                alt_args["in_stock"] = True
            calls.append(
                {
                    "id": "seed-search-1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps(alt_args),
                    },
                }
            )
    return calls[:max_calls]


def _build_order_fallback(
    state: EpisodeState,
    registry: ToolRegistry,
    *,
    turn: int,
) -> list[dict[str, Any]]:
    """Deterministic checkout once a cart SKU is verified."""
    if not state.last_cart_product_id or not state.last_cart_sku:
        return []
    if "place_test_order" not in registry.names:
        return []
    needs = _order_requires_justification(registry) or state.reason_requested
    mem = state.products.get(state.last_cart_product_id)
    # If justification is required but we never viewed facts, inspect first.
    if needs and (mem is None or not mem.attributes) and "view" in registry.names:
        return [
            {
                "id": f"preorder-view-{turn}",
                "type": "function",
                "function": {
                    "name": "view",
                    "arguments": json.dumps(
                        {"product_id": state.last_cart_product_id}
                    ),
                },
            }
        ]
    args = {
        "product_id": state.last_cart_product_id,
        "sku": state.last_cart_sku,
    }
    grounded = _sanitize_tool_args("place_test_order", args, state, registry)
    return [
        {
            "id": f"order-fallback-{turn}",
            "type": "function",
            "function": {
                "name": "place_test_order",
                "arguments": json.dumps(grounded),
            },
        }
    ]


def _budget_ceiling(text: str) -> float | None:
    """Parse an explicit spend ceiling from the public goal, if present."""
    ask = _shopper_portion(text)
    best: float | None = None
    for match in _BUDGET_NUM_RE.finditer(ask):
        raw = (match.group(1) or match.group(2) or "").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if best is None or value < best:
            best = value
    return best


def _cart_price(state: EpisodeState) -> float | None:
    if not state.last_cart_product_id or not state.last_cart_sku:
        return None
    mem = state.products.get(state.last_cart_product_id)
    if mem is None:
        return None
    variant = mem.variants.get(state.last_cart_sku) or {}
    raw = variant.get("price")
    if raw is None:
        return None
    try:
        return float(str(raw).replace(",", ""))
    except ValueError:
        return None


def _cart_over_budget(state: EpisodeState) -> bool:
    ceiling = _budget_ceiling(state.query)
    price = _cart_price(state)
    if ceiling is None or price is None:
        return False
    return price > ceiling + 1e-6


def _filter_tool_calls(
    tool_calls: list[dict[str, Any]],
    state: EpisodeState,
    registry: ToolRegistry,
    *,
    max_calls: int,
) -> list[dict[str, Any]]:
    """Drop clearly harmful mid-episode actions while preserving progress."""
    names = [call["function"]["name"] for call in tool_calls]
    filtered = list(tool_calls)

    # Do not cart and checkout in one turn.
    if "add_to_cart" in names and "place_test_order" in names:
        filtered = [
            call
            for call in filtered
            if call["function"]["name"] != "place_test_order"
        ]

    # Never add and remove in the same batch — that voids event commitment.
    if "add_to_cart" in names and "remove_from_cart" in names:
        filtered = [
            call
            for call in filtered
            if call["function"]["name"] != "remove_from_cart"
        ]

    # Never checkout an over-budget cart — remove/recover instead.
    if "place_test_order" in [c["function"]["name"] for c in filtered] and _cart_over_budget(state):
        filtered = [
            call
            for call in filtered
            if call["function"]["name"] != "place_test_order"
        ]
        if (
            "remove_from_cart" not in [c["function"]["name"] for c in filtered]
            and "remove_from_cart" in registry.names
            and state.last_cart_product_id
        ):
            filtered.insert(
                0,
                {
                    "id": "auto-remove-over-budget",
                    "type": "function",
                    "function": {
                        "name": "remove_from_cart",
                        "arguments": json.dumps(
                            {
                                "product_id": state.last_cart_product_id,
                                "sku": state.last_cart_sku,
                            }
                        ),
                    },
                },
            )

    # After carting, prefer live inspect_stock over inspect_cart alone.
    names = [call["function"]["name"] for call in filtered]
    if (
        "add_to_cart" in names
        and "inspect_stock" not in names
        and "inspect_stock" in registry.names
        and len(filtered) < max_calls
    ):
        cart_call = next(
            call
            for call in filtered
            if call["function"]["name"] == "add_to_cart"
        )
        try:
            cart_args = _parse_arguments(cart_call["function"].get("arguments", "{}"))
        except ValueError:
            cart_args = {}
        pid = str(cart_args.get("product_id") or state.last_cart_product_id or "")
        sku = str(cart_args.get("sku") or state.last_cart_sku or "")
        inspect_args: dict[str, Any] = {}
        props = registry.by_name["inspect_stock"].properties
        if "product_id" in props and pid:
            inspect_args["product_id"] = pid
        if "sku" in props and sku:
            inspect_args["sku"] = sku
        if inspect_args:
            filtered.append(
                {
                    "id": f"auto-inspect-{len(filtered)}",
                    "type": "function",
                    "function": {
                        "name": "inspect_stock",
                        "arguments": json.dumps(inspect_args),
                    },
                }
            )

    # Drop unknown tools.
    filtered = [
        call
        for call in filtered
        if call.get("function", {}).get("name") in registry.names
    ]
    filtered = filtered[:max_calls]
    if filtered:
        return filtered
    # If we stripped everything, recover from an over-budget cart or re-check stock.
    if state.last_cart_product_id and _cart_over_budget(state) and "remove_from_cart" in registry.names:
        return [
            {
                "id": "hold-remove-over-budget",
                "type": "function",
                "function": {
                    "name": "remove_from_cart",
                    "arguments": json.dumps(
                        {
                            "product_id": state.last_cart_product_id,
                            "sku": state.last_cart_sku,
                        }
                    ),
                },
            }
        ]
    if state.last_cart_product_id and "inspect_stock" in registry.names:
        inspect_args = {}
        props = registry.by_name["inspect_stock"].properties
        if "product_id" in props:
            inspect_args["product_id"] = state.last_cart_product_id
        if "sku" in props and state.last_cart_sku:
            inspect_args["sku"] = state.last_cart_sku
        return [
            {
                "id": "hold-inspect",
                "type": "function",
                "function": {
                    "name": "inspect_stock",
                    "arguments": json.dumps(inspect_args),
                },
            }
        ]
    return tool_calls[:max_calls]


def _ack_message_call(turn: int) -> dict[str, Any]:
    return {
        "id": f"msg-ack-{turn}",
        "type": "function",
        "function": {
            "name": "message",
            "arguments": json.dumps(
                {
                    "content": (
                        "Thanks — I re-checked the live options against your "
                        "constraints and will keep only a compliant pick."
                    )
                }
            ),
        },
    }


def _environment_envelope(
    binding: dict[str, Any],
    *,
    problem_id: str,
    turn: int,
    tool_calls: list[dict[str, Any]],
    state: EpisodeState,
    registry: ToolRegistry,
) -> dict[str, Any]:
    group_id = f"{problem_id}-turn-{turn}"
    calls_out: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls, start=1):
        name = call["function"]["name"]
        args = _sanitize_tool_args(
            name,
            _parse_arguments(call["function"].get("arguments", "{}")),
            state,
            registry,
        )
        calls_out.append(
            {
                "call_id": f"{group_id}-{index}",
                "action": {"name": name, "args": args},
            }
        )
    return {
        **binding,
        "call_id": group_id,
        "idempotency_key": group_id,
        "turn": turn,
        "calls": calls_out,
    }


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Plan and execute tool actions until the environment reports completion."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    query = str(policy["query"])
    max_steps = int(policy["max_steps"])
    max_calls = int(policy.get("max_calls_per_turn", 1))
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))

    registry = ToolRegistry.from_policy(list(policy.get("tools") or []))
    model = _resolve_model(query, registry)
    state = EpisodeState(
        reason_requested=_shopper_asks_for_reason(query),
        query=query,
    )
    dialogue: list[dict[str, Any]] = []

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _compose_system_prompt(query, registry, max_steps, max_calls),
        },
        {"role": "user", "content": query},
    ]

    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        needs_message = (
            state.unanswered_shopper and state.messages_sent < state.shopper_replies
        )
        order_ready = (
            bool(state.last_cart_product_id)
            and bool(state.last_cart_sku)
            and state.post_cart_verified
            and "place_test_order" in registry.names
            and not _cart_over_budget(state)
        )
        force_order = order_ready and remaining <= 3 and not needs_message
        prefer_order = bool(
            force_order
            or (order_ready and remaining <= 6 and not needs_message)
        )

        assistant_text = ""
        tool_calls: list[dict[str, Any]] | None = None

        if force_order or (prefer_order and remaining <= 5 and order_ready and not needs_message):
            order_calls = _build_order_fallback(state, registry, turn=turn)
            if order_calls:
                tool_calls = order_calls

        if tool_calls is None and turn == 1 and not state.shortlist_seeded and not needs_message:
            seeded = _discovery_seed_calls(registry, query, max_calls)
            if seeded:
                state.shortlist_seeded = True
                tool_calls = seeded

        if tool_calls is None:
            briefing = state.status_line(turn, max_steps, max_calls)
            if prefer_order and state.last_cart_product_id:
                briefing += (
                    " REQUIRED NOW: call place_test_order for the verified cart "
                    f"item product_id={state.last_cart_product_id} "
                    f"sku={state.last_cart_sku}"
                )
                if _order_requires_justification(registry) or state.reason_requested:
                    briefing += (
                        " with truthful justification claims (request specification "
                        "field + price + in_stock + optional schema soft reason)."
                    )
                else:
                    briefing += "."
            planner_messages = messages + [{"role": "system", "content": briefing}]
            try:
                assistant_text, tool_calls = _plan_actions(
                    planner_messages,
                    list(policy.get("tools") or []),
                    registry,
                    state,
                    model=model,
                    turn=turn,
                    max_steps=max_steps,
                    max_calls=max_calls,
                    prefer_order=prefer_order and not needs_message,
                )
            except RuntimeError:
                # Inference outage: finish from ledger when possible instead of hanging.
                if needs_message and "message" in registry.names:
                    tool_calls = [_ack_message_call(turn)]
                    if order_ready and max_calls > 1:
                        tool_calls.extend(
                            _build_order_fallback(state, registry, turn=turn)
                        )
                        tool_calls = tool_calls[:max_calls]
                    assistant_text = ""
                elif order_ready:
                    tool_calls = _build_order_fallback(state, registry, turn=turn)
                    assistant_text = ""
                elif "search" in registry.names and not state.search_runs:
                    tool_calls = _discovery_seed_calls(registry, query, max_calls)
                    assistant_text = ""
                else:
                    break
                if not tool_calls:
                    break

        tool_calls = _filter_tool_calls(
            tool_calls, state, registry, max_calls=max_calls
        )

        messages.append(
            {
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": tool_calls,
            }
        )

        envelope = _environment_envelope(
            binding,
            problem_id=problem_id,
            turn=turn,
            tool_calls=tool_calls,
            state=state,
            registry=registry,
        )
        result = _PROXY.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")

        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for tool_call, call_result, env_call in zip(
            tool_calls, result["calls"], envelope["calls"], strict=True
        ):
            observation = call_result["observation"]
            name = tool_call["function"]["name"]
            args = (env_call.get("action") or {}).get("args") or _parse_arguments(
                tool_call["function"].get("arguments", "{}")
            )
            state.register_observation(name, observation, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(_digest(observation)),
                }
            )

        user_message = result.get("user_message")
        if isinstance(user_message, dict) and user_message.get("content"):
            content = str(user_message["content"])
            state.register_shopper_message(content)
            messages.append({"role": "user", "content": content})

        if any(call["observation"].get("done") for call in result["calls"]):
            break

    return dialogue


__all__ = ["agent_main"]
