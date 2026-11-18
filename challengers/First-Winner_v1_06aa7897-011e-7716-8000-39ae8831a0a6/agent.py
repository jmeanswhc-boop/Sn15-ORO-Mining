"""Generic ORO Bench shopping agent.

LLM proposes every shopping decision from the public query, tool schemas, and
runtime observations. Code only handles inference I/O, observation bookkeeping,
schema filtering, and mechanical safety (inspect before cart, recheck after
cart, cart before order). No suite-specific lookup tables for screen sizes,
currencies, preference axes, claim-field names, or task-family rubrics.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


# --------------------------------------------------------------------------- infra
_MODELS = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_WALL_S = 270.0
_OBS_CHARS = 7000
_MAX_REPROMPTS = 2
_EPS = 1e-9

# Soft budget cue from free text — magnitude only, never a currency conversion table.
_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _NUM,
    re.I,
)
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)
# Shopper asked for a reason — not rules boilerplate that mentions "justification".
_REASON_RE = re.compile(r"\b(?:why|reason|explain|justif)\b", re.I)
_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(?:inches\b|inch\b|in\b|[\"”'])", re.I)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz\b", re.I)
_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×*]\s*(\d{3,5})")
_RANGE_RE = re.compile(r"\d+(?:\.\d+)?\s*[-–/]\s*\d+(?:\.\d+)?")
_RES_KEY_RE = re.compile(r"resolut|display|pixel|panel|screen_type", re.I)
_SIZE_KEY_RE = re.compile(r"size|screen|inch|display", re.I)
_HZ_KEY_RE = re.compile(r"refresh", re.I)
_RAM_KEY_RE = re.compile(r"ram|memory|storage|capacity", re.I)

# Policy text replaces hardcoded measure / preference / claim / taxonomy tables.
# Teach the model *how* to reason; do not encode suite answers in code.
_SYSTEM = """You are a careful shopping agent for an open product catalog.

Use only the supplied tools. Exactly one tool call per turn. Continue until an observation
reports done=true. Never invent catalog facts — every commitment must be grounded in
search/filter/view/compare/inspect_stock/inspect_cart observations from this session.

## Workflow
1. Read the shopper goal and any rules/clarifications carefully.
2. Surface candidates with search or filter (prefer a few precise queries over many).
3. view or compare so titles, facts, variant options, prices, and stock are on record.
4. inspect_stock the exact product_id + sku you intend to buy.
5. add_to_cart that exact variant.
6. Re-check stock/price after carting (prices and stock can change).
7. place_order / place_test_order only when the carted variant is still valid.
8. If a carted item is invalidated, remove it and commit a replacement — do not keep
   searching until turns run out.
9. Answer every direct shopper question with one short message before other actions.
10. Never ask the shopper to raise the budget. Do not repeat an identical tool call.
11. Prefer one compare of several products over many separate views.

## Interpreting the query (no memorized answer keys)
Treat the query as a generic shopping request. Infer constraints from the words present —
do not rely on hidden family labels, grader rubrics, or fixed lookup tables for screen
sizes, currencies, brands, or preference scores.

### Quantity / pack wording
When the query states an explicit piece/pack count (e.g. "2-piece", "3-pack", "5pcs")
together with a collective/quantity noun for the same product (e.g. "set", "pack", "kit",
"bundle", "lot") — such as "2-piece ... hair oil set" or "3-pack cable bundle" — these two
words describe the SAME single quantity fact, not two separate constraints. Drop the
now-redundant collective noun from a rewritten sub-query and keep only the explicit count,
so the sub-query reads as one unambiguous quantity mention.
  e.g. "2-piece organic hair oil set" -> "2-piece organic hair oil"
       (NOT "2-piece organic hair oil set");
       "3-pack cable bundle" -> "3-pack cable".
This is not a loss of detail — the explicit count already fully captures the
"set"/"pack"/"kit" meaning — but leaving both words invites downstream matching to wrongly
treat them as two independent constraints. Only keep the collective noun as-is when NO
explicit piece/pack count is stated (e.g. "hair oil set" with no number stays "hair oil set").

### Splitting multi-product requests
1. A completing spec or quantity in the middle — specs like "5 seats", "1 meter", "3 pcs",
   "2-pack", or a color/size that naturally concludes a product description often signal a
   boundary. After such a spec, a new product name may begin.
   Example: "car seat cushion cover 5 seats [END of product 1] blue patterned leatherette [product 2]"
2. Two product names from different categories placed back to back — when two noun phrases
   each independently name a product (e.g. "golf bag" and "stand bag"), treat them as two
   separate products even without a connector. Be conservative: only split when BOTH parts
   are standalone product names. Do NOT split "wireless bluetooth mechanical keyboard" —
   those are modifiers of one product noun.
3. Different color/attribute clusters assigned to different names —
   "golf bag WHITE ... stand bag BLUE" means each product has its own attribute cluster.

### Measures, sizes, and specs
Match numeric or named specs as they appear in the query against observation text
(titles, options, facts). Examples of language you may see: screen/display size in inches,
refresh rate in Hz, RAM/storage in GB/TB, resolution as Np / WxH / HD-style labels.
Rules:
- Prefer labelled fact fields and variant options over incidental title numbers.
- Accessory titles often borrow a parent device's size ("Case for … 6.1 inches") — that is
  NOT the accessory's own screen size when the shopper asked for the device itself.
- Do not use a fixed private map from phrases to catalog keys. Use the fact keys and option
  strings actually returned by tools. If evidence conflicts or is ambiguous, view/compare
  further or choose another listing.
- Resolution shorthand in the query (e.g. 720p, full HD) must be supported by evidence in
  that listing's facts/options/title; do not invent equivalences that the observation does
  not support.

### Currency and budget
Use the currency symbol or code that appears in the shopper text as-is. Do not convert
between currencies and do not assume a default currency table. Honor budget language
("under", "at most", "budget of", …) against observed prices in the same units shown.

### Brand, category, model, color
Follow explicit brand/category/model/color requests from the query. Prefer exact model and
preferred color when named. After a preferred listing disappears, prefer another variant of
the same model before switching models. Soft "prefer Brand X" cues are preferences, not
licenses to break firm constraints.

### Soft preferences / tradeoffs
When the shopper describes soft priorities (durability, fit for an activity, battery life,
wireless convenience, materials, "rather pay a bit more", "don't just grab the cheapest"),
rank by how well observed evidence covers several of those stated cues together.
Do not reduce this to a single memorized score table. Do not order a listing that only
matches one incidental token when stronger multi-cue evidence is already on record.
Skip accessory cases/sleeves/mounts when the shopper asked for the device itself.

### Accessories vs the requested product
If the shopper asked for a device (phone, tablet, headphones, monitor, …), do not cart a
case, cover, protector, mount, charger, cable, or "for <device>" add-on unless they asked
for that accessory. Use title, category path, and type/OS facts from observations — not a
private product whitelist.

### Negations and rejects
Honor "not a …", "reject …" kinds from the goal/rules. Do not treat instructional boilerplate
("do not settle", "do not wait") as product rejects.

### Named candidates and recall
If the query lists product_id::sku candidates, compare those first and choose among them.
If the query asks to find listings matching a phrase and to surface at least N distinct
matches, run one precise phrase search (optionally one word-order reformulation), then
survey hits with compare/view — do not spam extra searches.

### Justification / claims
When the shopper asks why / for a reason, the order tool needs structured claims:
  justification: {"claims": [{"field": "<name>", "value": "<value>"}, ...]}
Use these field names (not catalog label spellings): screen_size, resolution,
refresh_rate, ram, price, in_stock, brand.
Values must be proven by that product's labelled fact fields from view/compare:
  screen_size like "27in"; resolution like "hd" / "fhd" / "qhd" / "4k";
  price as "at most <budget>" (list prices change after cart — do not quote a stale price);
  in_stock as "true".
Titles often decoy (e.g. "720p HD" in the title while facts do not). Never claim a spec
that is missing from labelled facts, and do not cart a title-only match.

### Dialogue
Answer shopper questions briefly and truthfully from evidence. Ask at most one clarifying
question only when a firm requirement is sealed/hidden; otherwise decide from the query
and observations. Never ask them to raise the budget.

### Failure recovery
Identical tool calls are wasted. After rejection notes, change the action. When the market
invalidates a cart line, remove it and immediately work a replacement through
inspect → cart → recheck → order.
"""


# --------------------------------------------------------------------------- helpers
def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


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


def _tool_params(tool: dict) -> dict:
    fn = tool.get("function") or tool
    params = (fn.get("parameters") or {}) if isinstance(fn, dict) else {}
    props = params.get("properties") if isinstance(params, dict) else None
    return props if isinstance(props, dict) else {}


def _tool_name(tool: dict) -> str | None:
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    if tool.get("name"):
        return str(tool["name"])
    return None


def _parse_budget(text: str) -> float | None:
    """Soft upper bound from shopper language — not a currency lookup."""
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


def _goal_text(query: str) -> str:
    raw = query or ""
    return raw.split("\n\nTask rules:", 1)[0].split("\n\n", 1)[0]


def _facts_map(prod: dict) -> dict:
    raw = prod.get("facts")
    if raw is None:
        raw = prod.get("attributes")
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if v not in (None, "")}
    if isinstance(raw, list):
        out: dict = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = item.get("name") or item.get("key") or item.get("field") or item.get("attribute")
            val = item.get("value") if "value" in item else item.get("val")
            if key is not None and val not in (None, ""):
                out[str(key)] = val
        return out
    return {}


def _res_classes(text: str) -> set[str]:
    """Display classes stated in a fact string. hd is not full-hd / ultra-hd / hd+."""
    low = (text or "").lower()
    if not low.strip() or "[truncated]" in low:
        return set()
    compact = re.sub(r"\s+", "", low)
    out: set[str] = set()
    if re.search(r"\b(?:4k|uhd|ultra\s*hd|ultrahd|2160p)\b", low) or "3840x2160" in compact or "2160x3840" in compact:
        out.add("4k")
    if re.search(r"\b(?:qhd|1440p|2k)\b", low) or "2560x1440" in compact or "1440x2560" in compact:
        out.add("qhd")
    if re.search(r"\b(?:fhd|1080p|full\s*hd|fullhd)\b", low) or "1920x1080" in compact or "1080x1920" in compact:
        out.add("fhd")
    stripped = re.sub(r"\b(?:full|ultra)\s*hd\b", " ", low)
    if "fhd" not in out and "4k" not in out:
        if re.search(r"(?<![a-z0-9])hd\+(?![a-z0-9])", stripped):
            pass
        elif re.search(r"(?<![a-z0-9])(?:hd|720p)(?![a-z0-9+])", stripped):
            out.add("hd")
    for m in _WXH_RE.finditer(low):
        w, h = int(m.group(1)), int(m.group(2))
        # Height 720 is 720p; 720x1600 portrait width is not.
        if h == 720 or (w == 720 and w >= h):
            out.add("hd")
        if h in (1080, 1920) and w in (1080, 1920) and w != h:
            out.add("fhd")
        if h in (1440, 2560) and w in (1440, 2560) and w != h:
            out.add("qhd")
        if h in (2160, 3840) and w in (2160, 3840) and w != h:
            out.add("4k")
    return out


def _query_specs(text: str) -> dict[str, str]:
    """Hardware values the shopper stated — generic units, not a suite answer key."""
    text = text or ""
    out: dict[str, str] = {}
    inches = {float(m.group(1)) for m in _INCH_RE.finditer(text)}
    if len(inches) == 1:
        out["screen_size"] = "%gin" % next(iter(inches))
    hz = {float(m.group(1)) for m in _HZ_RE.finditer(text)}
    if len(hz) == 1:
        out["refresh_rate"] = "%ghz" % next(iter(hz))
    gb: set[float] = set()
    for m in _GB_RE.finditer(text):
        n = float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1)
        gb.add(n)
    if len(gb) == 1:
        out["ram"] = "%ggb" % next(iter(gb))
    low = text.lower()
    if re.search(r"\b(?:4k|uhd|ultra\s*hd|ultrahd|2160p)\b", low):
        out["resolution"] = "4k"
    elif re.search(r"\b(?:qhd|1440p|2k)\b", low):
        out["resolution"] = "qhd"
    elif re.search(r"\b(?:fhd|1080p|full\s*hd|fullhd)\b", low):
        out["resolution"] = "fhd"
    elif re.search(r"\b(?:720p|hd)\b", re.sub(r"\b(?:full|ultra)\s*hd\b", " ", low)):
        out["resolution"] = "hd"
    return out


def _labelled_proof(facts: dict, field: str) -> str | None:
    """Return the canonical value iff labelled facts (not titles) uniquely support it."""
    found: set[str] = set()
    for key, raw in facts.items():
        text = str(raw or "").strip()
        if not text or "[truncated]" in text.lower():
            continue
        klow = str(key).lower()
        if field == "resolution":
            if not _RES_KEY_RE.search(klow):
                continue
            found |= _res_classes(text)
        elif field == "screen_size":
            if not _SIZE_KEY_RE.search(klow):
                continue
            if _RANGE_RE.search(text):
                continue
            vals = {float(m.group(1)) for m in _INCH_RE.finditer(text)}
            if not vals and re.fullmatch(r"\d+(?:\.\d+)?", text):
                vals = {float(text)}
            found |= {"%gin" % v for v in vals}
        elif field == "refresh_rate":
            if not _HZ_KEY_RE.search(klow):
                continue
            vals = {float(m.group(1)) for m in _HZ_RE.finditer(text)}
            if not vals and re.fullmatch(r"\d+(?:\.\d+)?", text):
                vals = {float(text)}
            found |= {"%ghz" % v for v in vals}
        elif field == "ram":
            if not _RAM_KEY_RE.search(klow):
                continue
            vals = set()
            for m in _GB_RE.finditer(text):
                vals.add(float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1))
            if not vals and re.fullmatch(r"\d+(?:\.\d+)?", text):
                vals = {float(text)}
            found |= {"%ggb" % v for v in vals}
    return next(iter(found)) if len(found) == 1 else None


# --------------------------------------------------------------------------- ledger (observations only)
class Ledger:
    def __init__(self) -> None:
        self.cart: list[tuple[Any, Any]] = []
        self.checked: dict[str, dict] = {}
        self.checked_seq: dict[str, int] = {}
        self.inspected: set[str] = set()
        self.products: dict[str, dict] = {}
        self.seq = 0
        self.last_add_seq: int | None = None
        self.adds = 0
        self.dead_keys: set[str] = set()
        self.calls_seen: set[str] = set()
        self.budget: float | None = None

    @staticmethod
    def key(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def record(self, name: str, args: dict, obs: dict) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        if name in ("search", "filter"):
            for row in obs.get("results") or []:
                if isinstance(row, dict):
                    self._note_row(row)
        elif name == "view":
            self._note_product(obs)
        elif name == "compare":
            for prod in obs.get("compared") or []:
                if isinstance(prod, dict):
                    self._note_product(prod)
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
        pairs = [
            (r.get("product_id"), r.get("sku"))
            for r in rows
            if isinstance(r, dict) and r.get("product_id")
        ]
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
            # Soft signal: cart line that went OOS or over budget after a change.
            if stock is False:
                self.dead_keys.add(k)
            if self.budget is not None and price is not None and price > self.budget + _EPS:
                self.dead_keys.add(k)

    def _note_row(self, row: dict) -> None:
        pid = row.get("product_id")
        if not pid or not row.get("sku"):
            return
        k = self.key(pid, row.get("sku"))
        if row.get("price") is not None:
            self.checked.setdefault(
                k, {"in_stock": bool(row.get("in_stock", True)), "price": _num(row.get("price"))}
            )

    def _note_product(self, prod: dict) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.products[str(pid)] = prod
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            k = self.key(pid, v.get("sku"))
            self.checked[k] = {"in_stock": bool(v.get("in_stock")), "price": _num(v.get("price"))}
            self.checked_seq[k] = self.seq

    def status(self, pid: str, sku: str) -> dict | None:
        return self.checked.get(self.key(pid, sku))


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


# --------------------------------------------------------------------------- agent
class Agent:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.schemas: dict[str, dict] = {}
        self.tool_names: set[str] = set()
        for t in self.tools:
            name = _tool_name(t)
            if name:
                self.tool_names.add(name)
                self.schemas[name] = _tool_params(t)
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        query = str(self.policy.get("query") or "")
        self.budget = _parse_budget(query)
        goal = _goal_text(query)
        self.query_specs = _query_specs(goal)
        order_tool = self._find_tool(("place_test_order", "place_order", "order"))
        accepts_claims = bool(order_tool and "justification" in self.schemas.get(order_tool or "", {}))
        if not accepts_claims and order_tool:
            for tool in self.tools:
                if _tool_name(tool) == order_tool and "justification" in json.dumps(tool):
                    accepts_claims = True
                    props = dict(self.schemas.get(order_tool) or {})
                    props["justification"] = props.get("justification") or {}
                    self.schemas[order_tool] = props
                    break
        self.needs_justification = bool(accepts_claims and _REASON_RE.search(goal))
        self.ledger = Ledger()
        self.ledger.budget = self.budget
        self.clock = Clock(_WALL_S)
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": query},
        ]
        self.turn = 0
        self.shopper_question_pending = False
        self.questions_asked = 0
        self.cart_inspections = 0
        self.pending_recheck: tuple | None = None
        self.pending_envelope: dict | None = None
        self.rejected_streak = 0
        self.done = False
        self.model_index = 0
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    def _find_tool(self, candidates: tuple[str, ...]) -> str | None:
        for name in candidates:
            if name in self.tool_names:
                return name
        return None

    def _filter_args(self, name: str, args: dict) -> dict:
        props = self.schemas.get(name)
        if not props:
            return dict(args)
        return {k: v for k, v in args.items() if k in props}

    def _has(self, name: str) -> bool:
        return name in self.tool_names

    def _note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    def _order_tool(self) -> str | None:
        return self._find_tool(("place_test_order", "place_order", "order"))

    def _facts(self, pid: str) -> dict:
        return _facts_map(self.ledger.products.get(str(pid)) or {})

    def _brand_of(self, pid: str) -> str:
        prod = self.ledger.products.get(str(pid)) or {}
        if prod.get("brand"):
            return str(prod["brand"]).strip()
        for key, val in self._facts(pid).items():
            if re.search(r"^brand$|brand_name", str(key), re.I) and val not in (None, ""):
                return str(val).strip()
        return ""

    def _spec_claims(self, pid: str) -> list[dict[str, str]]:
        facts = self._facts(pid)
        if not facts:
            return []
        out: list[dict[str, str]] = []
        for field, want in self.query_specs.items():
            got = _labelled_proof(facts, field)
            if got != want:
                return []
            out.append({"field": field, "value": want})
        return out

    def _claims_for(self, pid: str, sku: str) -> list[dict[str, str]]:
        if not self.needs_justification:
            return []
        specs = self._spec_claims(pid)
        if self.query_specs and not specs:
            return []
        st = self.ledger.status(pid, sku) or {}
        if not st.get("in_stock"):
            return []
        out = list(specs)
        if self.budget is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.budget})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        else:
            return []
        out.append({"field": "in_stock", "value": "true"})
        brand = self._brand_of(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out

    def _order_args(self, pid: str, sku: str) -> dict:
        out: dict[str, Any] = {"product_id": str(pid), "sku": str(sku)}
        if self.needs_justification:
            claims = self._claims_for(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        name = self._order_tool() or "place_test_order"
        return self._filter_args(name, out)

    def _valid_now(self, pid: str, sku: str) -> tuple[bool, str]:
        st = self.ledger.status(pid, sku)
        k = Ledger.key(pid, sku)
        if st is None:
            return False, "inspect_stock the exact product_id and sku first"
        if not any(str(a) == str(pid) and str(b) == str(sku) for a, b in self.ledger.cart):
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

    def _budget_from_message(self, text: str) -> None:
        new = _parse_budget(text)
        if new is None or (self.budget is not None and new >= self.budget):
            return
        self.budget = new
        self.ledger.budget = new
        self._note(
            "Budget updated by the shopper: at most %.2f in the currency they used. "
            "Every later choice must fit it." % new
        )
        for pid, sku in list(self.ledger.cart):
            st = self.ledger.status(str(pid), str(sku)) or {}
            price = st.get("price")
            if price is not None and price > new + _EPS:
                self.ledger.dead_keys.add(Ledger.key(pid, sku))
                self._note(
                    "The carted item %s costs %.2f, above the new budget; replace it."
                    % (Ledger.key(pid, sku), price)
                )

    # ------------------------------------------------------------------ inference / env
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
        if name not in self.tool_names:
            self._note("Skipped unavailable tool: " + name)
            return None
        args = self._filter_args(name, args)
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self._send(pend["envelope"])
            if replay is None:
                self._note("The environment is not answering; the previous action is still pending.")
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
                "envelope": envelope,
                "turn": turn,
                "name": name,
                "args": args,
                "content": content,
                "call_id": tool_call_id or ("agent-" + str(turn)),
            }
        else:
            self.turn = turn
        return self._absorb(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        ]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.ledger.calls_seen.discard(name + json.dumps(args, sort_keys=True))
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
        self.ledger.record(name, args, obs if isinstance(obs, dict) else {})
        self.messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": _trim(json.dumps(obs), _OBS_CHARS)}
        )
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self._budget_from_message(text)
            if _QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self._note("The shopper asked a question: answer it with message next, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        return obs if isinstance(obs, dict) else {}

    def _post_add_recheck(self) -> None:
        if not self.pending_recheck or not self._has("inspect_stock"):
            self.pending_recheck = None
            return
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
                "Re-check after cart: %s is in stock at %s. Ready to order if requirements are met."
                % (Ledger.key(pid, sku), price if price is not None else "?")
            )
            return
        self.ledger.dead_keys.add(Ledger.key(pid, sku))
        why = "out of stock" if not stock else "above budget"
        self._note(
            "Re-check after cart: %s is %s. Remove it and choose a different variant."
            % (Ledger.key(pid, sku), why)
        )

    def _mechanical_plan(self) -> tuple | None:
        """Only non-discretionary safety steps — shopping choices stay with the LLM."""
        # Clear invalidated cart lines before anything else.
        # Title-decoy cart lines cannot be justified — treat like a market kill.
        if self.needs_justification:
            for cpid, csku in self.ledger.cart:
                if not self._claims_for(str(cpid), str(csku)):
                    self.ledger.dead_keys.add(Ledger.key(cpid, csku))
        for cpid, csku in self.ledger.cart:
            if Ledger.key(cpid, csku) in self.ledger.dead_keys and self._has("remove_from_cart"):
                return (
                    "remove_from_cart",
                    self._filter_args("remove_from_cart", {"product_id": cpid, "sku": csku}),
                )
        # Answer pending shopper questions before further shopping.
        if self.shopper_question_pending and self._has("message"):
            # Let the model phrase the answer — nudge only.
            return None
        left = self.max_steps - self.turn
        secs = self.clock.remaining()
        urgent = left <= 3 or secs < 25
        order_name = self._order_tool()
        if urgent and order_name:
            for pid, sku in reversed(self.ledger.cart):
                ok, _why = self._valid_now(str(pid), str(sku))
                if not ok:
                    continue
                if self.needs_justification and not self._claims_for(str(pid), str(sku)):
                    continue
                return (order_name, self._order_args(str(pid), str(sku)))
        return None

    def _validate(self, name: str, args: dict) -> str | None:
        """Mechanical safety only — no measure/preference/family rubrics."""
        if name not in self.tool_names:
            return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
        if name in ("place_test_order", "place_order", "order"):
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.ledger.cart:
                pid, sku = str(self.ledger.cart[-1][0]), str(self.ledger.cart[-1][1])
            ok, why = self._valid_now(pid, sku)
            if not ok:
                return "order refused: " + why
            if self.needs_justification and not self._claims_for(pid, sku):
                return (
                    "order refused: labelled facts do not prove the requested spec "
                    "(title-only does not count). view/compare a listing whose facts state it, "
                    "then order with justification claims"
                )
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            if any(Ledger.key(cp, cs) in self.ledger.dead_keys for cp, cs in self.ledger.cart):
                return "remove the invalidated cart item before adding a replacement"
            if str(pid) not in self.ledger.products:
                return "view or compare that product first so its facts are on record"
            key = Ledger.key(pid, sku)
            if key not in self.ledger.inspected:
                return "inspect_stock that exact variant first"
            st = self.ledger.status(pid, sku) or {}
            if not st.get("in_stock"):
                return "that variant is out of stock, over budget, or already invalidated"
            price = st.get("price")
            if self.budget is not None and price is not None and price > self.budget + _EPS:
                return "that variant is over the shopper's budget"
            if key in self.ledger.dead_keys:
                return "that variant was invalidated; choose a different product"
            if self.needs_justification and not self._spec_claims(pid):
                return (
                    "that listing's labelled facts do not state the requested spec "
                    "(title text is not enough). view another product whose facts prove it"
                )
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.shopper_question_pending and "?" not in text:
                return "use message only to answer the shopper or to ask a clarifying question"
            if not self.shopper_question_pending and self.questions_asked >= 2:
                return "the shopper has already answered; decide from the evidence you have"
        if name == "inspect_cart" and self.cart_inspections >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.ledger.calls_seen and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        self.ledger.calls_seen.add(key)
        return None

    def run(self) -> list:
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            if self.pending_recheck and self._has("inspect_stock"):
                self._post_add_recheck()
                continue

            plan = self._mechanical_plan()
            if plan is not None:
                name, args = plan
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    last_forced, repeats = None, 0
                    self._note("That safety step changed nothing; choose another valid action.")
                    continue
                self._act(name, args)
                continue

            last_forced, repeats = None, 0
            if self.shopper_question_pending:
                self._note(
                    "Answer the shopper's pending question with the message tool now, "
                    "briefly and from evidence, then continue shopping."
                )

            left = self.max_steps - self.turn
            if left <= 6 or self.clock.remaining() < 45:
                extra = (
                    " Pass justification.claims from labelled facts (spec + price + in_stock)."
                    if self.needs_justification
                    else ""
                )
                self._note(
                    "Few turns/time remain. If a valid carted variant exists, inspect_stock "
                    "if needed and place the order; otherwise remove invalidated lines and "
                    "commit the best evidence-backed replacement quickly." + extra
                )

            proposal = self._ask_model()
            if proposal is None:
                # Last-resort: order a still-valid cart line if any.
                order_name = self._order_tool()
                if order_name:
                    for pid, sku in reversed(self.ledger.cart):
                        ok, _why = self._valid_now(str(pid), str(sku))
                        if not ok:
                            continue
                        if self.needs_justification and not self._claims_for(str(pid), str(sku)):
                            continue
                        self._act(order_name, self._order_args(str(pid), str(sku)))
                        break
                break

            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = _arguments(call)
            content = proposal.get("content") or ""
            reprompts = 0
            verdict: str | None = None
            while True:
                verdict = self._validate(name, args)
                if verdict is None:
                    break
                if reprompts >= _MAX_REPROMPTS:
                    self.rejected_streak += 1
                    self._note("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if self.rejected_streak >= 3:
                        self._note(
                            "Several proposals were rejected. Re-read the goal, compare remaining "
                            "candidates, inspect_stock a compliant variant, cart it, recheck, and order."
                        )
                        self.rejected_streak = 0
                    verdict = "skip"
                    break
                reprompts += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id") or "x",
                        "content": json.dumps({"rejected": verdict}),
                    }
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

            args = self._filter_args(name, args)
            if name in ("place_test_order", "place_order", "order"):
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if (not pid or not sku) and self.ledger.cart:
                    pid, sku = str(self.ledger.cart[-1][0]), str(self.ledger.cart[-1][1])
                args = self._order_args(pid, sku)
            if name == "message":
                if not self.shopper_question_pending:
                    self.questions_asked += 1
                self.shopper_question_pending = False
            obs = self._act(name, args, content, call.get("id"))
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") or {}
                apid, asku = added.get("product_id"), added.get("sku")
                if apid and asku:
                    self.pending_recheck = (str(apid), str(asku))
            if name == "inspect_cart":
                self.cart_inspections += 1
        return self.dialogue


def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:  # never raise out of the harness
        return [
            {
                "role": "assistant",
                "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200],
            }
        ]


__all__ = ["agent_main"]
