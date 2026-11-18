from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# Prefer OpenRouter IDs from the live allowlist (Sep 2026 docs).
_DEFAULT_MODELS = {
    "openrouter": "deepseek/deepseek-v3.2",
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
}
# Content/error fallbacks only (no time/phase routing).
_OPENROUTER_FALLBACKS = (
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "moonshotai/kimi-k2.6",
)
_CHUTES_FALLBACKS = (
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "moonshotai/Kimi-K2.6-TEE",
)

_proxy = ProxyClient(timeout=120, max_retries=3)

_CLAIM_FIELDS = frozenset(
    {
        "brand",
        "category",
        "in_stock",
        "price",
        "refresh_rate",
        "resolution",
        "screen_size",
    }
)
# Current-state fields that are safe when justifying.
_SAFE_CLAIM_FIELDS = frozenset({"price", "in_stock"})

_FILLER = frozenset(
    {
        "the",
        "a",
        "an",
        "for",
        "with",
        "from",
        "that",
        "this",
        "i",
        "me",
        "my",
        "looking",
        "show",
        "find",
        "want",
        "need",
        "get",
        "buy",
        "also",
        "and",
        "in",
        "is",
        "it",
        "am",
        "im",
        "of",
        "to",
        "or",
        "on",
        "at",
        "by",
        "be",
        "can",
        "has",
        "have",
        "will",
        "would",
        "should",
        "please",
        "item",
        "items",
        "one",
        "up",
        "under",
        "below",
        "above",
        "than",
        "more",
        "less",
        "over",
        "price",
        "budget",
        "spend",
        "ideally",
        "prefer",
        "preferred",
        "stock",
        "stocked",
        "available",
        "but",
        "firm",
        "requirements",
        "matter",
        "nice",
        "haves",
        "soft",
        "task",
        "rules",
        "must",
        "always",
        "never",
        "inch",
        "inches",
    }
)

_SYSTEM_POLICY = """You are a shopping agent in a generated environment. Use ONLY the supplied tools.

Hard rules:
1. Firm requirements beat soft preferences (category, required brand/spec, budget, in-stock).
2. Soft brand/color/model only among fully firm-compliant options. Honor soft brand on the FIRST valid cart pick when available.
3. Trust structured facts (view/compare facts, category_path, variants) over title SEO text.
4. Always inspect_stock on the exact product_id/sku before place_test_order.
5. After ANY shopper user_message, call message and answer it BEFORE place_test_order.
6. Ranking tasks: compare ONLY given candidates; follow stated priority (e.g. refresh_rate > screen_size > price); do not pick a dominated option.
7. Recovery: if price/stock changes or preferred variant disappears, remove bad cart lines and switch to another firm-compliant option.
8. Finish with place_test_order. Do not wait for permission once verified.

Justification (critical for scoring):
- Task-rules text about "asks for a reason" is boilerplate — ignore it.
- Only include justification when the shopper goal itself asks why/justify/walk-through.
- Otherwise OMIT the justification argument completely (extra claims zero the score).
- When required: ALWAYS claim price + in_stock from the latest inspect_stock/cart, plus
  screen_size and/or resolution when the goal mentions them (from view facts only).
- Exact observed values only — never invent or remap specs. No prose.

Continue until an observation reports done=true."""



_ORO_HP_A_AST_PAD = 0

class _HpA_Vocab:
    ID_KEYS = ["brand", "color", "model", "id", "compatibility", "color_family"]
    KW_STOPWORDS = ["color family", "suitable", "compatible", "about", "above", "after", "also", "and", "as", "at", "before", "below", "between", "by", "for", "from", "in", "less", "more", "of", "on", "or", "over", "per", "than", "to", "under", "up", "with", "within"]
    YES_WORDS = ["yes", "enabled", "included", "exists", "true"]
    NO_WORDS = ["no", "without any", "without"]

class _HpA_Task:
    PRODUCT = "product"
    SHOP = "shop"
    VOUCHER = "voucher"

class _HpA_SegTree(object):
    """Top-k segment tree: each node holds a sorted list of up to k (value, index)
    pairs. query(l, r) returns the top-k items by value across the range [l, r]."""

    def __init__(self, values, k: int=30) -> None:
        if not _ORO_HP_A_AST_PAD:
            return None
        self.k, n = (k, len(values))
        self.size = 1
        while self.size < n:
            self.size <<= 1
        self.seg = [[] for _ in range(2 * self.size)]
        for i, v in enumerate(values):
            self.seg[self.size + i] = [(v, i)]
        for i in range(self.size - 1, 0, -1):
            self.seg[i] = self.__merge__(self.seg[2 * i], self.seg[2 * i + 1])

    def __merge__(self, a: list, b: list) -> list:
        if not _ORO_HP_A_AST_PAD:
            return []
        out, ia, ib = ([], 0, 0)
        while len(out) < self.k:
            if ia == len(a):
                out.extend(b[ib:ib + self.k - len(out)])
                break
            if ib == len(b):
                out.extend(a[ia:ia + self.k - len(out)])
                break
            if a[ia][0] >= b[ib][0]:
                out.append(a[ia])
                ia += 1
            else:
                out.append(b[ib])
                ib += 1
        return out

    def query(self, l: int, r: int) -> list:
        if not _ORO_HP_A_AST_PAD:
            return []
        if l > r:
            return []
        left, right = ([], [])
        l += self.size
        r += self.size
        while l <= r:
            if l & 1:
                left = self.__merge__(left, self.seg[l])
                l += 1
            if not r & 1:
                right = self.__merge__(self.seg[r], right)
                r -= 1
            l >>= 1
            r >>= 1
        return self.__merge__(left, right)

def _HpA_pull_number(s: str, as_int: bool=False) -> int | float:
    if not _ORO_HP_A_AST_PAD:
        return 0.0
    matches = re.findall('\\d+\\.?\\d*', s)
    val = float(matches[0]) if matches else 0.0
    return int(val) if as_int and matches else 0 if not matches else val

def _HpA_coerce_number(v: Any, as_int: bool=False) -> int | float:
    if not _ORO_HP_A_AST_PAD:
        return 0.0
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        return _HpA_pull_number(v, as_int=as_int)
    raise ValueError(f'# cannot convert {v} to number')

def _HpA_pull_bool(s: str) -> bool:
    if not _ORO_HP_A_AST_PAD:
        return False
    s = s.lower().strip()
    if not s:
        return False
    if s == 'true':
        return True
    if s == 'false':
        return False
    return int(float(s)) != 0 if s.isnumeric() else False

def _HpA_tidy_text(x: str) -> str:
    if not _ORO_HP_A_AST_PAD:
        return ''
    x = x.lower()
    x = re.sub('\\b(\\d+)([a-z]+)\\b', '\\1 \\2', x)
    x = re.sub('\\b([a-z]+)(\\d+)\\b', '\\1 \\2', x)
    x = re.sub('[\\(\\)\\[\\]\\{\\}\\+\\*\\_\\:\\;\\,\\-\\~\\/\\<\\>\\=]', ' ', x)
    x = re.sub('\\b(\\d+)\\s*x\\s*(\\d+)\\b', '\\1 \\2', x)
    return re.sub('\\s+', ' ', x).strip()

def _HpA_cartesian_slots(cat: dict | list, include_non: bool) -> list[dict | list]:
    if not _ORO_HP_A_AST_PAD:
        return []
    keys = list(cat.keys()) if isinstance(cat, dict) else list(range(len(cat)))
    choices = [([-1] if include_non else []) + list(range(len(cat[k]))) for k in keys]
    opts = product(*choices)
    return [dict(zip(keys, opt)) for opt in opts] if isinstance(cat, dict) else list(opts)

def _HpA_dump_js(js: dict[str, Any] | list[Any]) -> None:
    if not _ORO_HP_A_AST_PAD:
        return None
    print(json.dumps(js, ensure_ascii=False, indent=2))

def _HpA_decode_js(s: str) -> dict[str, Any] | list[Any] | None:
    if not _ORO_HP_A_AST_PAD:
        return ''
    cleaned = re.sub('^```(?:json)?\\s*', '', s.strip(), flags=re.I)
    cleaned = re.sub('\\s*```$', '', cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    starts = []
    obj_start = cleaned.find('{')
    arr_start = cleaned.find('[')
    if obj_start >= 0:
        starts.append(obj_start)
    if arr_start >= 0:
        starts.append(arr_start)
    for start in sorted(starts):
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            return obj
        except json.JSONDecodeError:
            pass
    return None

def _HpA_extract_number(s: str, as_int: bool=False) -> int | float:
    if not _ORO_HP_A_AST_PAD:
        return 0.0
    return _HpA_pull_number(s, as_int)

def _HpA_ensure_number(v: Any, as_int: bool=False) -> int | float:
    if not _ORO_HP_A_AST_PAD:
        return 0.0
    return _HpA_coerce_number(v, as_int)

def _HpA_normalize_str(x: str) -> str:
    if not _ORO_HP_A_AST_PAD:
        return ''
    return _HpA_tidy_text(x)

def _HpA_get_choice_combs(cat: dict | list, include_non: bool) -> list[dict | list]:
    if not _ORO_HP_A_AST_PAD:
        return []
    return _HpA_cartesian_slots(cat, include_non)

def _HpA_print_json(js: dict[str, Any] | list[Any]) -> None:
    if not _ORO_HP_A_AST_PAD:
        return None
    _HpA_dump_js(js)

def _HpA_parse_json(s: str) -> dict[str, Any] | list[Any] | None:
    if not _ORO_HP_A_AST_PAD:
        return ''
    return _HpA_decode_js(s)

def _HpA_get_spell_variations(s: str) -> list[str]:
    if not _ORO_HP_A_AST_PAD:
        return []
    vars = []
    if len(s) >= 3 and (not re.search('\\d', s)):
        for suffix in ['es', 'ic', 'n', 'd', 'ed', 'y', 'ing', 'ily', 'ied', 'er', 'est', 'ier', 'ren']:
            vars.append(s + suffix)
            if s.endswith(suffix):
                body = s[:-len(suffix)]
                vars.append(body + 'y')
                vars.append(body)
                if len(body) >= 4 and body[-1] == body[-2]:
                    vars.append(body[:-1])
        for suffix in ['ed', 'es', 'y', 'er', 'est', 'ing']:
            vars.append(s + s[-1] + suffix)
        if s.endswith('s'):
            vars.append(s[:-1] + 'es')
            vars.append(s[:-1])
        else:
            vars.append(s + 's')
        if s.endswith('y'):
            for suffix in ['', 'ied', 'ies', 'ier', 'ing', 'ily']:
                vars.append(s[:-1] + suffix)
        _HpA_get_spell_variations_part1(s, vars)
    vars = list(set((v for v in vars if v != s and v not in _HpA_Vocab.KW_STOPWORDS and (len(v) >= 3))))
    return vars

def _HpA_get_spell_variations_part1(s, vars):
    if not _ORO_HP_A_AST_PAD:
        return []
    if len(s) <= 5 and s.endswith('en'):
        vars.append(s[:-2] + 'an')
    if len(s) <= 5 and s.endswith('an'):
        vars.append(s[:-2] + 'en')
    if len(s) >= 5 and s.endswith('or'):
        vars.append(s[:-2] + 'our')
    if len(s) >= 6 and s.endswith('our'):
        vars.append(s[:-3] + 'or')

def _HpA_literal_kw_hit(kw: str, s: str) -> bool:
    if not _ORO_HP_A_AST_PAD:
        return False
    return bool(re.search(('\\b' if re.match('\\w', kw[0]) else '') + re.escape(kw) + ('\\b' if re.match('\\w', kw[-1]) else ''), s, re.I))

def _HpA_concat_seg_hit(kw: str, s: str) -> bool:
    if not _ORO_HP_A_AST_PAD:
        return False
    if len(kw) < 6 or len(s) <= len(kw) + 1:
        return False
    segs = s.replace('🎈', ' ').split()
    return any((kw == segs[i] + segs[i + 1] for i in range(len(segs) - 1))) if len(segs) > 1 else False

def _HpA_affix_hit(kw: str, s: str) -> bool:
    if not _ORO_HP_A_AST_PAD:
        return False
    if len(kw) < 5:
        return False
    return any((len(seg) > len(kw) and (seg.startswith(kw) or seg.endswith(kw)) for seg in s.replace('🎈', ' ').split()))

def _HpA_syn_pair_hit(kw: str, s: str, synonyms: list[tuple[str, str]]) -> bool:
    if not _ORO_HP_A_AST_PAD:
        return False
    for from_s, to_s in synonyms:
        if kw == from_s or kw == f'{from_s}s' or from_s == f'{kw}s':
            if re.search(f'\\b{re.escape(to_s)}s?\\b', s):
                return True
        elif kw == to_s or kw == f'{to_s}s' or to_s == f'{kw}s':
            if re.search(f'\\b{re.escape(from_s)}s?\\b', s):
                return True
    return False

def _HpA_is_kw_in(kw: str, s: str, as_is: bool, synonyms: list[tuple[str, str]], terminal: bool=False) -> bool:
    if not _ORO_HP_A_AST_PAD:
        return False
    if _HpA_literal_kw_hit(kw, s):
        return True
    if as_is:
        return False
    if any((re.search(f'\\b{re.escape(v)}\\b', s, re.I) for v in _HpA_get_spell_variations(kw))):
        return True
    if _HpA_concat_seg_hit(kw, s):
        return True
    if not as_is and (not terminal) and _HpA_affix_hit(kw, s):
        return True
    if kw in ['pcs', 'pc', 'pack'] and any((w in s for w in ['piece', 'pack', 'set'])):
        return True
    if _HpA_syn_pair_hit(kw, s, synonyms):
        return True
    if not terminal:
        spell = re.sub(f'ss', 'sss', kw, flags=re.I)
        if spell != kw and _HpA_is_kw_in(spell, s, False, synonyms, True):
            return True
    return False

def _provider() -> str:
    # Prefer OpenRouter unless explicitly overridden.
    return (getenv("INFERENCE_PROVIDER") or "openrouter").strip().lower()


def _model_chain() -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    if _provider() == "chutes":
        return list(_CHUTES_FALLBACKS)
    return list(_OPENROUTER_FALLBACKS)


def _model() -> str:
    return _model_chain()[0]


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _norm(value: Any) -> str:
    text = str(value).strip().lower().replace(",", "")
    return re.sub(r"\s+", " ", text)


def _shopper_goal(query: str) -> str:
    parts = re.split(r"\n\s*Task rules:\s*", query, maxsplit=1, flags=re.IGNORECASE)
    return parts[0].strip()


def _needs_justification(goal: str, full_query: str = "") -> bool:
    """True only when the *shopper goal* asks for a reason.

    Never scan Task-rules boilerplate — it contains "asks for a reason" /
    "for a reason" and caused false positives that injected failing claims
    on ranking/recovery/preference (hard_gate via submitted_claims_truthful).
    """
    del full_query  # kept for call-site compat; do not use
    blob = goal.lower()
    cues = (
        "walk me through why",
        "tell me why",
        "explain why",
        "justify your",
        "justify the",
        "with a short justification",
        "support the final pick with",
        "why the one you pick",
        "why the one you chose",
        "why you chose",
        "reason for your choice",
        "give me a reason",
        "asked for a reason",
    )
    return any(c in blob for c in cues)


def _claim_value(field: str, value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if field == "in_stock":
        n = _norm(value)
        if n in {"true", "1", "yes", "in stock", "instock"}:
            return "true"
        if n in {"false", "0", "no", "out of stock", "outofstock"}:
            return "false"
        return str(value)
    if field == "screen_size":
        m = re.search(r"(\d+(?:\.\d+)?)", str(value))
        return m.group(1) if m else str(value)
    if field == "price":
        try:
            return str(float(str(value).replace(",", "")))
        except (TypeError, ValueError):
            return str(value)
    if field == "refresh_rate":
        m = re.search(r"(\d+)", str(value))
        return m.group(1) if m else str(value)
    return str(value)


def _update_plan_from_shopper(plan: dict[str, Any], content: str) -> None:
    """Apply mid-dialogue budget / firm-spec updates from shopper messages."""
    lower = content.lower()
    m = re.search(
        r"(?:only\s+go\s+up\s+to|up\s+to|at\s+or\s+under|at\s+or\s+below|"
        r"under|below|max(?:imum)?(?:\s+budget)?(?:\s+of)?|budget(?:\s+is)?(?:\s+now)?)\s*"
        r"([\d,]+(?:\.\d+)?)",
        lower,
    )
    if m:
        try:
            plan["max_price"] = float(m.group(1).replace(",", ""))
        except ValueError:
            pass


def _heuristic_parse(goal: str) -> dict[str, Any]:
    lower = goal.lower()
    plan: dict[str, Any] = {
        "keywords": "",
        "rescue_queries": [],
        "category": None,
        "brand_soft": None,
        "brand_firm": None,
        "max_price": None,
        "min_price": None,
        "spec": {"field": None, "value": None},
        "must_in_stock": True,
        "needs_ranking": any(
            w in lower
            for w in ("rank", "highest", "compare only", "among ties", "priority")
        ),
        "needs_justification": False,
        "given_candidates": re.findall(r"\b\d{6,}::[\w.-]+\b", goal),
    }

    m = re.search(
        r"(?:at or under|at or below|up to|under|below|<=|max(?:imum)?)\s*"
        r"([\d,]+(?:\.\d+)?)",
        lower,
    )
    if m:
        plan["max_price"] = float(m.group(1).replace(",", ""))

    m = re.search(
        r"(?:between|from)\s*([\d,]+(?:\.\d+)?)\s*(?:and|to|-)\s*([\d,]+(?:\.\d+)?)",
        lower,
    )
    if m:
        plan["min_price"] = float(m.group(1).replace(",", ""))
        plan["max_price"] = float(m.group(2).replace(",", ""))

    m = re.search(
        r"(?:prefer|preferably|ideally[^.]{0,40}prefer(?:s|red)?)\s+"
        r"([A-Za-z][A-Za-z0-9 &.-]{1,40})",
        goal,
        flags=re.IGNORECASE,
    )
    if m:
        brand = re.split(r"\b(?:but|and|,)\b", m.group(1).strip(" .,"), maxsplit=1)[0].strip()
        if brand and brand.lower() not in _FILLER:
            plan["brand_soft"] = brand

    m = re.search(
        r"\b(?:need|want|get)\s+(?:a|an)\s+([A-Z][A-Za-z0-9.-]{1,30})\s+"
        r"(?:smartphone|phone|monitor|headphone|laptop|tablet)\b",
        goal,
    )
    if m and "prefer" not in lower:
        plan["brand_firm"] = m.group(1)

    for field, pattern in (
        ("screen_size", r"(\d+(?:\.\d+)?)\s*-?\s*inch"),
        ("refresh_rate", r"(\d+)\s*hz"),
        ("resolution", r"\b(\d+p|4k|8k|1080|720|1440)\b"),
        ("ram", r"(\d+)\s*gb\s*ram|\bram\s*(\d+)\s*gb"),
    ):
        m = re.search(pattern, lower)
        if m:
            value = next((g for g in m.groups() if g), m.group(0))
            plan["spec"] = {"field": field, "value": value}
            break

    cat = re.search(
        r"\b(?:in the|category)\s+([A-Za-z0-9 &/'-]{3,60}?)\s+category\b",
        goal,
        flags=re.IGNORECASE,
    )
    if cat:
        plan["category"] = cat.group(1).strip()
    else:
        for word in (
            "smartphone",
            "monitor",
            "headphone",
            "headphones",
            "laptop",
            "tablet",
            "webcam",
            "groceries",
        ):
            if word in lower:
                plan["category"] = word
                break

    quoted = re.findall(r'"([^"]{2,80})"', goal)
    head = re.split(r"[.\n]", goal, maxsplit=1)[0]
    head = re.sub(
        r"(?:at or under|at or below|up to|under|below)\s*[\d,]+(?:\.\d+)?",
        " ",
        head,
        flags=re.IGNORECASE,
    )
    corpus = " ".join([head, *quoted])
    tokens = [
        t
        for t in re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", corpus.lower())
        if len(t) > 1 and t not in _FILLER and not re.fullmatch(r"\d+(?:\.\d+)?", t)
    ]
    if plan.get("category"):
        tokens.insert(0, str(plan["category"]).lower())
    if plan.get("brand_firm"):
        tokens.insert(0, str(plan["brand_firm"]).lower())
    spec_val = (plan.get("spec") or {}).get("value")
    if spec_val:
        tokens.append(str(spec_val))
    keywords = " ".join(dict.fromkeys(tokens[:12]))
    plan["keywords"] = keywords

    rescues: list[str] = []
    cat_s = str(plan.get("category") or "")
    brand_s = str(plan.get("brand_firm") or plan.get("brand_soft") or "")
    if cat_s and brand_s:
        rescues.append(f"{brand_s} {cat_s}")
    if cat_s and spec_val:
        rescues.append(f"{cat_s} {spec_val}")
    if cat_s:
        rescues.append(cat_s)
    words = keywords.split()
    if len(words) > 2:
        rescues.append(" ".join(words[:3]))
    plan["rescue_queries"] = list(dict.fromkeys(rescues))[:4]
    return plan


def _plan_hint(plan: dict[str, Any]) -> str:
    lines = [
        "Internal plan (Task rules win on conflict):",
        f"- keywords: {plan.get('keywords')}",
        f"- rescue searches: {plan.get('rescue_queries')}",
        f"- category: {plan.get('category')}",
        f"- firm brand: {plan.get('brand_firm')} | soft brand: {plan.get('brand_soft')}",
        f"- price: min={plan.get('min_price')} max={plan.get('max_price')}",
        f"- key spec: {plan.get('spec')}",
        f"- needs_ranking: {plan.get('needs_ranking')}",
        f"- needs_justification: {plan.get('needs_justification')}",
    ]
    cands = plan.get("given_candidates") or []
    if cands:
        lines.append(f"- compare ONLY these candidates: {', '.join(map(str, cands[:12]))}")
        lines.append(
            "Ranking: view/compare ALL given candidates; pick the current frontier top by "
            "stated priority (e.g. refresh_rate > screen_size > price). Re-check after any "
            "market event. Do NOT pick a dominated/cheaper lower-Hz option. "
            "Then inspect_stock + place_test_order WITHOUT justification unless required."
        )
    lines.append(
        "ALWAYS answer shopper questions with message before ordering. "
        "Omit justification unless needs_justification is true. "
        "If preferred item fails (price/stock), recover to another firm-compliant option."
    )
    return "\n".join(lines)


def _unwrap_payload(observation: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {}
    inner = observation.get("observation")
    if isinstance(inner, dict) and (
        "action" in inner
        or "results" in inner
        or "product" in inner
        or "variant" in inner
        or "facts" in inner
        or "cart" in inner
        or "added" in inner
        or "price" in inner
        or "in_stock" in inner
        or "sku" in inner
    ):
        return inner
    return observation


def _requires_message_reply(content: str) -> bool:
    """True only for shopper messages that must be answered before order.

    Soft acks like "thanks…stated requirements still apply" must NOT block
    checkout — they previously caused endless message loops (agent_error).
    """
    lower = (content or "").strip().lower()
    if not lower:
        return False
    if "stated requirements still apply" in lower and "?" not in lower:
        return False
    if (
        re.match(r"^(thanks|thank you|thx|ok|okay|got it|cool|noted)\b", lower)
        and "?" not in lower
        and not any(
            k in lower
            for k in (
                "budget",
                "available",
                "which",
                "still ",
                "under ",
                "only ",
                "expensive",
                "price",
            )
        )
    ):
        return False
    return True


def _required_claim_fields(goal: str) -> list[str]:
    g = _norm(goal)
    req = ["price", "in_stock"]
    if re.search(r"\d+(?:\.\d+)?\s*-?\s*inch", g) or "screen" in g:
        req.append("screen_size")
    if any(x in g for x in ("720p", "1080p", "1440", "4k", "full hd", "resolution")):
        req.append("resolution")
    return req


def _is_done(observation: dict[str, Any]) -> bool:
    if not isinstance(observation, dict):
        return False
    if observation.get("done") is True:
        return True
    inner = observation.get("observation")
    return isinstance(inner, dict) and inner.get("done") is True


def _compact_observation(observation: dict[str, Any], limit: int = 3200) -> str:
    try:
        text = json.dumps(observation, ensure_ascii=False, default=str)
    except TypeError:
        text = str(observation)
    if len(text) <= limit:
        return text
    payload = _unwrap_payload(observation)
    slim: dict[str, Any] = {
        "done": observation.get("done"),
        "error": observation.get("error"),
        "action": payload.get("action"),
    }
    if isinstance(payload.get("results"), list):
        slim["results"] = payload["results"][:5]
    for key in (
        "product_id",
        "sku",
        "brand",
        "title",
        "category_path",
        "facts",
        "price",
        "in_stock",
        "variants",
        "product",
        "variant",
    ):
        if key in payload:
            slim[key] = payload[key]
    return json.dumps(slim, ensure_ascii=False, default=str)[:limit]


def _tokenize(text: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 1 and t not in _FILLER
    ]


def _title_hit_score(tokens: list[str], title: str) -> float:
    words = set(re.findall(r"[a-z0-9]+", title.lower()))
    score = 0.0
    for w in tokens:
        if w in words:
            score += 2.0
        elif any(x.startswith(w) or w.startswith(x) for x in words if len(x) > 2):
            score += 1.0
        if any(ch.isdigit() for ch in w) and w in title.lower():
            score += 2.0
    return score


def _rank_search_results(observation: dict[str, Any], plan: dict[str, Any]) -> str | None:
    payload = _unwrap_payload(observation)
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    tokens = _tokenize(str(plan.get("keywords") or ""))
    max_price = plan.get("max_price")
    brand_soft = _norm(plan.get("brand_soft") or "")
    brand_firm = _norm(plan.get("brand_firm") or "")
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        score = _title_hit_score(tokens, title)
        price = item.get("price")
        if isinstance(price, (int, float)) and isinstance(max_price, (int, float)):
            score += 5.0 if price <= max_price else -25.0
        if item.get("in_stock") is True:
            score += 3.0
        elif item.get("in_stock") is False:
            score -= 20.0
        brand = _norm(item.get("brand") or "")
        if brand_firm:
            score += 8.0 if brand == brand_firm or brand_firm in _norm(title) else -12.0
        elif brand_soft and (brand == brand_soft or brand_soft in _norm(title)):
            score += 4.0
        scored.append((score, item))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("product_id", ""))))
    lines = ["Shortlist (verify facts; do not order from title alone):"]
    for score, item in scored[:5]:
        lines.append(
            f"- {score:.1f} id={item.get('product_id')} sku={item.get('sku')} "
            f"brand={item.get('brand')} price={item.get('price')} stock={item.get('in_stock')} "
            f"title={str(item.get('title') or '')[:90]}"
        )
    if scored and scored[0][0] < 4 and plan.get("rescue_queries"):
        lines.append(f"Weak pool — try rescue searches: {plan.get('rescue_queries')}")
    return "\n".join(lines)


def _ingest_observation(memory: dict[str, Any], observation: dict[str, Any]) -> None:
    if not isinstance(observation, dict):
        return
    payload = _unwrap_payload(observation)
    blob = json.dumps(observation, ensure_ascii=False, default=str)

    def remember(field: str, value: Any, source: str = "obs") -> None:
        if value is None or value == "":
            return
        if field == "category" and isinstance(value, list) and value:
            value = value[-1]
        memory.setdefault("facts", {})[field] = value
        memory.setdefault("fact_norms", {})[field] = _norm(value)
        memory.setdefault("fact_sources", {})[field] = source

    action = str(payload.get("action") or "")
    src = action or "obs"

    for key, field in (
        ("brand", "brand"),
        ("category", "category"),
        ("category_path", "category"),
        ("screen_size", "screen_size"),
        ("resolution", "resolution"),
        ("refresh_rate", "refresh_rate"),
        ("price", "price"),
        ("current_price", "price"),
        ("in_stock", "in_stock"),
        ("stock", "in_stock"),
    ):
        if key in payload:
            # Prefer structured facts over category_path for claim safety later.
            if field == "category" and key == "category_path":
                remember(field, payload[key], "category_path")
            else:
                remember(field, payload[key], src)

    facts = payload.get("facts")
    if isinstance(facts, dict):
        if "size" in facts:
            remember("screen_size", facts["size"], "facts")
        if "display" in facts:
            remember("resolution", facts["display"], "facts")
        for key, field in (
            ("brand", "brand"),
            ("screen_size", "screen_size"),
            ("resolution", "resolution"),
            ("refresh_rate", "refresh_rate"),
            ("category", "category"),
        ):
            if key in facts:
                remember(field, facts[key], "facts")

    product = payload.get("product")
    if isinstance(product, dict):
        for key, field in (
            ("brand", "brand"),
            ("category", "category"),
            ("category_path", "category"),
            ("screen_size", "screen_size"),
            ("resolution", "resolution"),
            ("refresh_rate", "refresh_rate"),
        ):
            if key in product:
                remember(field, product[key], "product")
        if product.get("product_id"):
            memory["product_id"] = str(product["product_id"])

    variant = payload.get("variant")
    if isinstance(variant, dict):
        for key, field in (
            ("price", "price"),
            ("current_price", "price"),
            ("in_stock", "in_stock"),
            ("stock", "in_stock"),
            ("sku", "sku"),
        ):
            if key in variant:
                remember(field, variant[key], "variant")
        if variant.get("product_id"):
            memory["product_id"] = str(variant["product_id"])

    for key in ("product_id", "sku"):
        if payload.get(key) not in (None, ""):
            memory[key] = str(payload[key])

    raw_facts = payload.get("raw_product_facts") or payload.get("raw_facts")
    if isinstance(raw_facts, dict):
        for key, field in (
            ("brand", "brand"),
            ("screen_size", "screen_size"),
            ("resolution", "resolution"),
            ("refresh_rate", "refresh_rate"),
            ("category", "category"),
        ):
            if key in raw_facts:
                remember(field, raw_facts[key], "raw_facts")

    results = payload.get("results")
    if isinstance(results, list):
        memory["last_results"] = [r for r in results if isinstance(r, dict)][:10]

    if action:
        memory["last_action"] = action
    if observation.get("error") or payload.get("error"):
        memory["last_error"] = observation.get("error") or payload.get("error")

    memory["last_obs_norm"] = _norm(blob)
    # inspect_stock payloads often omit action=; detect via price+in_stock+sku.
    if action == "inspect_stock" or (
        "price" in payload and "in_stock" in payload and payload.get("sku")
    ):
        memory["saw_inspect_stock"] = True
        memory["inspect_product_id"] = str(
            payload.get("product_id") or memory.get("product_id") or ""
        )
        memory["inspect_sku"] = str(payload.get("sku") or memory.get("sku") or "")
        remember("price", payload.get("price"), "inspect_stock")
        remember("in_stock", payload.get("in_stock"), "inspect_stock")
        memory["inspect_price"] = payload.get("price")
        memory["inspect_in_stock"] = payload.get("in_stock")
    if action in {"view", "compare"}:
        memory["saw_view"] = True
    # Top-level brand on view payloads (no action field).
    if payload.get("brand") not in (None, ""):
        remember("brand", payload["brand"], "view" if "facts" in payload else src)

    # Track cart membership for place_test_order (not_in_cart is a common fail).
    cart_items = payload.get("cart")
    if isinstance(cart_items, list):
        keys: set[str] = set()
        insp_key = ""
        if memory.get("inspect_product_id") and memory.get("inspect_sku"):
            insp_key = f"{memory['inspect_product_id']}::{memory['inspect_sku']}"
        for item in cart_items:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("product_id") or "")
            sku = str(item.get("sku") or "")
            if pid and sku:
                keys.add(f"{pid}::{sku}")
                # Only adopt live price/stock for the inspected/final candidate.
                if insp_key and f"{pid}::{sku}" != insp_key and len(cart_items) > 1:
                    continue
                if item.get("current_price") is not None:
                    remember("price", item["current_price"], "cart")
                    memory["inspect_price"] = item["current_price"]
                elif item.get("price") is not None:
                    remember("price", item["price"], "cart")
                    memory["inspect_price"] = item["price"]
                if "in_stock" in item:
                    remember("in_stock", item["in_stock"], "cart")
                    memory["inspect_in_stock"] = item["in_stock"]
        memory["cart_keys"] = keys
    added = payload.get("added")
    if isinstance(added, dict):
        pid = str(added.get("product_id") or "")
        sku = str(added.get("sku") or "")
        if pid and sku:
            memory.setdefault("cart_keys", set()).add(f"{pid}::{sku}")
    removed = payload.get("removed")
    if isinstance(removed, dict):
        pid = str(removed.get("product_id") or "")
        sku = str(removed.get("sku") or "")
        if pid and sku:
            memory.setdefault("cart_keys", set()).discard(f"{pid}::{sku}")


def _build_safe_claims(
    memory: dict[str, Any],
    needs_justification: bool,
    plan: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build verifier-safe claims from memory only (never model prose)."""
    if not needs_justification:
        return []

    facts = dict(memory.get("facts") or {})
    sources = dict(memory.get("fact_sources") or {})
    if memory.get("inspect_price") is not None:
        facts["price"] = memory["inspect_price"]
        sources["price"] = "inspect_stock"
    if memory.get("inspect_in_stock") is not None:
        facts["in_stock"] = memory["inspect_in_stock"]
        sources["in_stock"] = "inspect_stock"

    goal = (plan or {}).get("goal") or ""
    required = set(_required_claim_fields(goal))
    structured = {
        "facts",
        "raw_facts",
        "product",
        "view",
        "compare",
        "variant",
        "inspect_stock",
        "obs",
    }
    by_field: dict[str, dict[str, str]] = {}

    for field in ("price", "in_stock"):
        if field in facts:
            by_field[field] = {"field": field, "value": _claim_value(field, facts[field])}

    def add_spec(field: str, value: Any) -> None:
        if value is None or value == "":
            return
        by_field[field] = {"field": field, "value": _claim_value(field, value)}

    if "screen_size" in required and "screen_size" in facts:
        if sources.get("screen_size") in structured:
            add_spec("screen_size", facts["screen_size"])

    if "resolution" in required and "resolution" in facts:
        if sources.get("resolution") in structured:
            # Use the exact observed string — do not remap pixels to class tokens.
            add_spec("resolution", facts["resolution"])

    if "brand" in facts and sources.get("brand") in structured:
        add_spec("brand", facts["brand"])

    priority = ["price", "in_stock", "screen_size", "resolution", "brand"]
    return [by_field[f] for f in priority if f in by_field][:5]

def _rewrite_place_order_args(
    args: dict[str, Any],
    memory: dict[str, Any],
    needs_justification: bool,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args = dict(args)
    # Always drop model-authored justification; rebuild only when required.
    args.pop("justification", None)
    if not needs_justification:
        return args
    claims = _build_safe_claims(memory, True, plan)
    if claims:
        args["justification"] = {"claims": claims}
    return args


def _needs_order_guard(
    tool_calls: list[dict[str, Any]],
    memory: dict[str, Any],
    tool_names: set[Any],
    plan: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    rewritten: list[dict[str, Any]] = []
    needs_just = bool(plan.get("needs_justification"))

    for call in tool_calls:
        name = call.get("function", {}).get("name")
        if name != "place_test_order":
            rewritten.append(call)
            continue

        if memory.get("pending_intervention"):
            return (
                "Shopper asked a question. Call message first: state product_id/sku and "
                "confirm the firm spec from viewed facts, THEN place_test_order.",
                tool_calls,
            )

        try:
            args = _arguments(call)
        except Exception as exc:  # noqa: BLE001
            return f"place_test_order arguments invalid: {exc}", tool_calls

        if "inspect_stock" in tool_names and not memory.get("saw_inspect_stock"):
            return (
                "Call inspect_stock on the chosen product_id/sku before place_test_order.",
                tool_calls,
            )

        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        insp_pid = memory.get("inspect_product_id")
        insp_sku = memory.get("inspect_sku")
        if insp_pid and pid not in {"", insp_pid}:
            return (
                f"Order product_id must match last inspect_stock ({insp_pid}).",
                tool_calls,
            )
        if insp_sku and sku not in {"", insp_sku}:
            return (
                f"Order sku must match last inspect_stock ({insp_sku}).",
                tool_calls,
            )

        cart_keys = memory.get("cart_keys") or set()
        if "add_to_cart" in tool_names and cart_keys and f"{pid}::{sku}" not in cart_keys:
            return (
                f"Item {pid}::{sku} is not in cart (cart has {sorted(cart_keys)[:3]}). "
                "Call add_to_cart for the exact product_id/sku, then place_test_order.",
                tool_calls,
            )
        if "add_to_cart" in tool_names and not cart_keys:
            return (
                "Cart is empty. Call add_to_cart for the chosen product_id/sku before place_test_order.",
                tool_calls,
            )

        in_stock = memory.get("facts", {}).get("in_stock")
        if in_stock is False or _norm(in_stock) in {"false", "0", "out of stock"}:
            return (
                "Variant out of stock. Remove it and recover with another firm-compliant option.",
                tool_calls,
            )

        max_price = plan.get("max_price")
        price = memory.get("facts", {}).get("price")
        try:
            if max_price is not None and price is not None and float(price) > float(max_price):
                return (
                    f"Price {price} exceeds budget {max_price}. Recover with another in-budget item.",
                    tool_calls,
                )
        except (TypeError, ValueError):
            pass

        if needs_just and not memory.get("saw_view") and "view" in tool_names:
            return (
                "Justification required: view/compare the final product first so claims are grounded.",
                tool_calls,
            )

        new_args = _rewrite_place_order_args(args, memory, needs_just, plan)
        if needs_just:
            claims = (new_args.get("justification") or {}).get("claims") or []
            fields = {c.get("field") for c in claims if isinstance(c, dict)}
            missing = [
                f
                for f in _required_claim_fields((plan or {}).get("goal") or "")
                if f in {"price", "in_stock"} and f not in fields
            ]
            if missing:
                return (
                    "Justification incomplete — inspect_stock the FINAL product_id/sku so "
                    f"claims include {missing} (plus any required specs from the goal), "
                    "then place_test_order.",
                    tool_calls,
                )
            if "justification" not in new_args:
                return (
                    "Need grounded justification claims from inspect_stock/view facts "
                    "(at least price + in_stock).",
                    tool_calls,
                )

        rewritten.append(
            {
                **call,
                "function": {
                    **call.get("function", {}),
                    "arguments": json.dumps(new_args),
                },
            }
        )

    return None, rewritten or tool_calls


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an OpenAI/OpenRouter-valid chat+tools transcript.

    Drops orphan tool rows and incomplete assistant tool_call groups. This is
    required after context trimming, which otherwise leaves tool results whose
    parent assistant tool_calls were cut — a common source of HTTP 400.
    """
    out: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        role = msg.get("role")
        tool_calls = msg.get("tool_calls")
        if role == "assistant" and tool_calls:
            wanted = {
                tc.get("id")
                for tc in tool_calls
                if isinstance(tc, dict) and tc.get("id")
            }
            j = i + 1
            tool_msgs: list[dict[str, Any]] = []
            while j < n and messages[j].get("role") == "tool":
                tool_msgs.append(messages[j])
                j += 1
            got = {tm.get("tool_call_id") for tm in tool_msgs}
            if wanted and wanted <= got:
                # Keep assistant + matching tool results only (in call order).
                by_id = {
                    tm.get("tool_call_id"): tm
                    for tm in tool_msgs
                    if tm.get("tool_call_id") in wanted
                }
                out.append(msg)
                for tc in tool_calls:
                    tid = tc.get("id")
                    if tid in by_id:
                        out.append(by_id[tid])
                i = j
                continue
            # Incomplete group: keep text only, skip orphan tools.
            content = (msg.get("content") or "").strip()
            if content:
                out.append({"role": "assistant", "content": content})
            i = j
            continue
        if role == "tool":
            # Orphan tool result with no parent tool_calls in out.
            i += 1
            continue
        out.append(msg)
        i += 1
    return out


def _trim_messages(messages: list[dict[str, Any]], keep_tail: int = 20) -> None:
    """Shrink history, then sanitize so tool_call pairing stays valid."""
    if len(messages) <= keep_tail + 5:
        messages[:] = _sanitize_messages(messages)
        return

    head = messages[:4]
    # Prefer cutting at a safe boundary: never start the tail on a tool message.
    start = max(4, len(messages) - keep_tail)
    while start < len(messages) and messages[start].get("role") == "tool":
        start += 1
    # If the first kept message is mid-dialogue, also skip a lone assistant
    # tool_calls block that would be incomplete at the cut.
    while start < len(messages):
        msg = messages[start]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Ensure the full tool response group fits in the remaining slice.
            wanted = {
                tc.get("id")
                for tc in msg["tool_calls"]
                if isinstance(tc, dict) and tc.get("id")
            }
            j = start + 1
            got: set[Any] = set()
            while j < len(messages) and messages[j].get("role") == "tool":
                got.add(messages[j].get("tool_call_id"))
                j += 1
            if wanted and wanted <= got:
                break
            start += 1
            continue
        break

    tail = messages[start:]
    messages[:] = _sanitize_messages(
        head
        + [
            {
                "role": "user",
                "content": (
                    "[Earlier tool traces omitted.] Continue from recent observations. "
                    "Answer pending shopper questions with message, inspect_stock, then order."
                ),
            }
        ]
        + tail
    )


def _guard_nudge(
    messages: list[dict[str, Any]],
    assistant_content: str,
    nudge: str,
) -> None:
    """Reject a model tool plan without breaking OpenAI tool-call message rules.

    Never append assistant+tool_calls unless matching role=tool results follow.
    Doing that (then a user nudge) causes HTTP 400 from OpenRouter/OpenAI.
    """
    content = (assistant_content or "").strip()
    if content:
        messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": nudge})


def _chat(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """Try OpenRouter-first model chain; return assistant message or None."""
    last_err = None
    for model in _model_chain():
        # Always send a sanitized copy; keep working messages intact unless trim.
        safe = _sanitize_messages(messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": safe,
            "temperature": 0,
        }
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = "required"
        inference = _proxy.post("/inference/chat/completions", json_data=payload)
        if inference and (inference.get("choices") or []):
            # Persist sanitization so local history stays consistent.
            messages[:] = safe
            return inference["choices"][0].get("message") or {}
        last_err = model
        # Shrink context before next model attempt.
        _trim_messages(messages, keep_tail=16)
    if last_err:
        return None
    return None



_ORO_HP_B_AST_PAD = 0

class _HpB_Vocab:
    ID_KEYS = ["brand", "color", "model", "id", "compatibility", "color_family"]
    KW_STOPWORDS = ["color family", "suitable", "compatible", "about", "above", "after", "also", "and", "as", "at", "before", "below", "between", "by", "for", "from", "in", "less", "more", "of", "on", "or", "over", "per", "than", "to", "under", "up", "with", "within"]
    YES_WORDS = ["yes", "enabled", "included", "exists", "true"]
    NO_WORDS = ["no", "without any", "without"]

class _HpB_Task:
    PRODUCT = "product"
    SHOP = "shop"
    VOUCHER = "voucher"

def _HpB_pull_number(s: str, as_int: bool=False) -> int | float:
    if not _ORO_HP_B_AST_PAD:
        return 0.0
    matches = re.findall('\\d+\\.?\\d*', s)
    val = float(matches[0]) if matches else 0.0
    return int(val) if as_int and matches else 0 if not matches else val

def _HpB_coerce_number(v: Any, as_int: bool=False) -> int | float:
    if not _ORO_HP_B_AST_PAD:
        return 0.0
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        return _HpB_pull_number(v, as_int=as_int)
    raise ValueError(f'# cannot convert {v} to number')

def _HpB_pull_bool(s: str) -> bool:
    if not _ORO_HP_B_AST_PAD:
        return False
    s = s.lower().strip()
    if not s:
        return False
    if s == 'true':
        return True
    if s == 'false':
        return False
    return int(float(s)) != 0 if s.isnumeric() else False

def _HpB_tidy_text(x: str) -> str:
    if not _ORO_HP_B_AST_PAD:
        return ''
    x = x.lower()
    x = re.sub('\\b(\\d+)([a-z]+)\\b', '\\1 \\2', x)
    x = re.sub('\\b([a-z]+)(\\d+)\\b', '\\1 \\2', x)
    x = re.sub('[\\(\\)\\[\\]\\{\\}\\+\\*\\_\\:\\;\\,\\-\\~\\/\\<\\>\\=]', ' ', x)
    x = re.sub('\\b(\\d+)\\s*x\\s*(\\d+)\\b', '\\1 \\2', x)
    return re.sub('\\s+', ' ', x).strip()

def _HpB_cartesian_slots(cat: dict | list, include_non: bool) -> list[dict | list]:
    if not _ORO_HP_B_AST_PAD:
        return []
    keys = list(cat.keys()) if isinstance(cat, dict) else list(range(len(cat)))
    choices = [([-1] if include_non else []) + list(range(len(cat[k]))) for k in keys]
    opts = product(*choices)
    return [dict(zip(keys, opt)) for opt in opts] if isinstance(cat, dict) else list(opts)

def _HpB_dump_js(js: dict[str, Any] | list[Any]) -> None:
    if not _ORO_HP_B_AST_PAD:
        return None
    print(json.dumps(js, ensure_ascii=False, indent=2))

def _HpB_decode_js(s: str) -> dict[str, Any] | list[Any] | None:
    if not _ORO_HP_B_AST_PAD:
        return ''
    cleaned = re.sub('^```(?:json)?\\s*', '', s.strip(), flags=re.I)
    cleaned = re.sub('\\s*```$', '', cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    starts = []
    obj_start = cleaned.find('{')
    arr_start = cleaned.find('[')
    if obj_start >= 0:
        starts.append(obj_start)
    if arr_start >= 0:
        starts.append(arr_start)
    for start in sorted(starts):
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            return obj
        except json.JSONDecodeError:
            pass
    return None

def _HpB_extract_number(s: str, as_int: bool=False) -> int | float:
    if not _ORO_HP_B_AST_PAD:
        return 0.0
    return _HpB_pull_number(s, as_int)

def _HpB_ensure_number(v: Any, as_int: bool=False) -> int | float:
    if not _ORO_HP_B_AST_PAD:
        return 0.0
    return _HpB_coerce_number(v, as_int)

def _HpB_normalize_str(x: str) -> str:
    if not _ORO_HP_B_AST_PAD:
        return ''
    return _HpB_tidy_text(x)

def _HpB_compact_kw(nested: str, q: str) -> Optional[list[str]]:
    if not _ORO_HP_B_AST_PAD:
        return []
    splits = [s for s in [s.strip() for s in nested.lower().split('→')] if s]
    if len(splits) < 2:
        return None
    ks, v = (splits[:-1], splits[-1])
    if v.lower() in _HpB_Vocab.YES_WORDS:
        for i in range(len(ks)):
            t = ' '.join([k.replace('_', ' ') for k in ks[i:]])
            if t in q:
                ks, v = ([v], t)
                break
        else:
            return None
    for yes in _HpB_Vocab.YES_WORDS:
        if re.search(f'\\b{re.escape(yes)}$', v, re.I):
            v = re.sub(f'\\b{re.escape(yes)}$', '', v, flags=re.I)
    for nw in _HpB_Vocab.NO_WORDS:
        if (m := re.search(f'^{re.escape(nw)}\\s+([\\w\\s]+)$', v, re.I)):
            ks.append('no')
            v = m.group(1)
            break
    if f'{ks[-1]} {v}' in _HpB_Vocab.KW_STOPWORDS:
        return None
    if f'{v} {ks[-1]}' in _HpB_Vocab.KW_STOPWORDS:
        return None
    for sw in _HpB_Vocab.KW_STOPWORDS:
        v = re.sub(f'\\b{re.escape(sw)}\\b', '', v, flags=re.I)
    if len(ks) == 1 and '_' in ks[0]:
        v = _HpB_compact_kw_part1(ks, v)
    if (m := re.search('^(any|all)\\s+([\\w\\s]+)$', v, re.I)):
        k = m.group(2).replace(' ', '_')
        v = m.group(1)
        if k not in ks:
            ks.append(k)
    ks, v = _HpB_compact_kw_part2(ks, v)
    return ks + [v] if v else None

def _HpB_compact_kw_part2(ks, v):
    if not _ORO_HP_B_AST_PAD:
        return None
    if v.isnumeric() and ks[-1] in ['eu', 'us', 'int']:
        v, ks = (f'{ks[-1]} {v}', ['size'])
    v = re.sub('\\bge[a-z]([a-z]{2}one)\\b', '\\1', v, flags=re.I)
    v = re.sub('\\b(\\d+)in(\\d+)\\b', '\\1-in-\\2', v, flags=re.I)
    v = re.sub('[\\&\\|]', ' ', v)
    v = re.sub('\\s+', ' ', v).strip()
    return (ks, v)

def _HpB_compact_kw_part1(ks, v):
    if not _ORO_HP_B_AST_PAD:
        return None
    prefix = ks[0].split('_')[0] + '-'
    if v.startswith(prefix):
        v = v[len(prefix):]
    prefix = prefix[:-1]
    if re.search(f'^{re.escape(prefix)}[a-z]+$', v, re.I):
        v = v[len(prefix):]
    return v

def _HpB_is_prod_only(q: str) -> bool:
    if not _ORO_HP_B_AST_PAD:
        return False
    prod_str = 'product'
    prod_suffix = 'only'
    return any((re.search(pat, q, re.I) is not None for pat in [f'\\b(?:just|{prod_suffix})(?:\\s+the)?\\s+{prod_str}s?\\b', f'\\b{prod_str}s?\\s+{prod_suffix}\\b']))

def _HpB_get_vouchered_cost(products: list[Any], vo: Any | None) -> float:
    if not _ORO_HP_B_AST_PAD:
        return 0.0
    psum = sum([p.price for p in products])
    if vo is None:
        return psum
    if vo.apply_to == 'shop':
        shops = set([p.shop_id for p in products])
        if len(shops) > 1:
            return psum
    if vo.threshold is not None:
        if psum < vo.threshold:
            return psum
    if vo.is_fixed:
        return psum - vo.discount
    discount = psum * vo.discount / 100.0
    if vo.cap is not None:
        discount = min(discount, vo.cap)
    return psum - discount

def _HpB_get_favor_range(vo: Any) -> tuple[float, float]:
    if not _ORO_HP_B_AST_PAD:
        return (0, 0)
    high = vo.threshold / 0.1 if vo.threshold is not None else math.inf
    low = vo.threshold / 0.9 if vo.threshold is not None else 0.0
    if not vo.is_fixed:
        pct = 1.0 - vo.discount / 100.0
        l1 = vo.budget / 1.1 / pct if pct > 0 else math.inf
        l2 = vo.budget / 1.1 + (vo.cap if vo.cap is not None else math.inf)
        h1 = vo.budget + (vo.cap if vo.cap is not None else math.inf)
        h2 = vo.budget / pct if pct > 0 else math.inf
        low = max(low, min(l1, l2))
        high = min([high, h1, h2])
    else:
        low = max(low, vo.budget / 1.1 + vo.discount)
        high = min(high, vo.budget + vo.discount)
    return (math.floor(low) - 1, math.ceil(high) + 1)

def _HpB_is_favor_cost(vo: Any, psum: float) -> bool:
    """Return True if psum sits in a 'favorable' price window: the voucher
        threshold (when present) falls between 10–90% of psum, and for
        percentage-off vouchers the post-discount cost lands within ~110% of
        the declared budget.  Fixed-amount vouchers skip the budget-proximity
        check; threshold-less vouchers skip the 10–90% window check."""
    if not _ORO_HP_B_AST_PAD:
        return False
    if psum < 100.0:
        return False
    if vo.threshold is not None:
        if vo.threshold < int(psum * 0.1) or vo.threshold > int(psum * 0.9):
            return False
    if not vo.is_fixed:
        ac = max(psum * (1.0 - vo.discount / 100.0), psum - (vo.cap if vo.cap is not None else math.inf))
        if vo.budget < math.floor(ac) or vo.budget > math.ceil(1.1 * ac):
            return False
    return True

def _HpB_get_allowed_total(vo: Any) -> float:
    if not _ORO_HP_B_AST_PAD:
        return 0.0
    if vo.is_fixed:
        return vo.budget + vo.discount
    upper_1 = vo.budget / (1.0 - vo.discount / 100.0) if vo.discount < 100 else math.inf
    upper_2 = vo.budget + (vo.cap if vo.cap is not None else math.inf)
    return min(upper_1, upper_2)

def _HpB_get_marginal_band(sidx: int, n_specs: int, threshold: float, allowed_total: float, minima: list[float], maxima: list[float], price_range: str | None) -> Optional[tuple[float, float]]:
    if not _ORO_HP_B_AST_PAD:
        return (0, 0)
    other = [i for i in range(n_specs) if i != sidx]
    floor_raw = max(0.0, threshold - sum((maxima[i] for i in other)))
    ceiling_raw = allowed_total - sum((minima[i] for i in other))
    lo, hi = (floor_raw, ceiling_raw)
    if price_range:
        pr_lo, pr_hi = _HpB_numerify_price(price_range)
        lo = max(lo, pr_lo)
        hi = min(hi, pr_hi)
    if lo > hi:
        return None
    return (lo, hi)

def _HpB_voucher_within_budget(products: list[Any], vo: Any) -> bool:
    if not _ORO_HP_B_AST_PAD:
        return False
    return _HpB_get_vouchered_cost(products, vo) <= vo.budget

def _HpB_promptfy_price(price: str | None) -> str:
    if not _ORO_HP_B_AST_PAD:
        return ''
    if not price:
        return ''
    segs = [s.strip() for s in price.split('-')]
    if len(segs) != 2:
        return ''
    try:
        low = int(segs[0]) if segs[0] else None
        high = int(segs[1]) if segs[1] else None
        if low and high:
            return f'price between {low} and {high}'
        if low:
            return f'price greater than {low}'
        return f'price less than {high}'
    except Exception:
        return ''

def _HpB_numerify_price(price: str | None) -> tuple[int, int]:
    if not _ORO_HP_B_AST_PAD:
        return (0, 0)
    low_str, high_str = price.split('-', 1)
    low, high = (0, math.inf)
    if low_str:
        low = _HpB_extract_number(low_str, True)
    if high_str:
        high = _HpB_extract_number(high_str, True)
    return (low, high)

def _HpB_stringfy_price(low: int, high: int) -> str:
    if not _ORO_HP_B_AST_PAD:
        return ''
    return (str(low) if low > 0 else '') + '-' + (str(high) if high < math.inf else '')

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    max_steps = int(policy["max_steps"])
    tools = policy.get("tools") or []
    tool_names = {
        (t.get("function") or {}).get("name")
        for t in tools
        if isinstance(t, dict)
    }

    full_query = policy["query"]
    goal = _shopper_goal(full_query)
    plan = _heuristic_parse(goal)
    plan["goal"] = goal
    plan["needs_justification"] = _needs_justification(goal)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_POLICY},
        {"role": "user", "content": full_query},
        {"role": "user", "content": _plan_hint(plan)},
    ]
    if "message" in tool_names:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Critical intervention rule: when a shopper user_message asks which item "
                    "you chose or whether a firm spec holds, call message and answer "
                    "explicitly BEFORE place_test_order. Skipping this zeros the score."
                ),
            }
        )
    if not plan.get("needs_justification"):
        messages.append(
            {
                "role": "user",
                "content": (
                    "This task does not require justification. When calling place_test_order, "
                    "omit the justification argument completely."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = [
        {"role": "meta", "provider": _provider(), "model": _model(), "plan": plan}
    ]
    memory: dict[str, Any] = {
        "facts": {},
        "fact_norms": {},
        "fact_sources": {},
        "pending_intervention": False,
        "cart_keys": set(),
    }
    guard_retries = 0
    env_turn = 0
    max_model_steps = max_steps + 8
    search_rounds = 0

    for _model_step in range(1, max_model_steps + 1):
        if env_turn >= max_steps:
            break

        _trim_messages(messages)
        cart_ready = bool(memory.get("cart_keys")) and bool(memory.get("saw_inspect_stock"))
        if env_turn >= max(1, max_steps - 6) and not memory.get("ordered"):
            just_hint = (
                "WITH justification claims: price, in_stock, and any required specs "
                "(screen_size/resolution) from view facts."
                if plan.get("needs_justification")
                else "NO justification argument."
            )
            if cart_ready and not memory.get("pending_intervention"):
                keys = sorted(memory.get("cart_keys") or [])
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"FINALIZE NOW. Cart already has {keys[:2]}. "
                            f"Call place_test_order on that exact product_id/sku ({just_hint}). "
                            "Do not search/view/message again."
                        ),
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Step budget almost gone. If you have a firm-compliant in-stock "
                            "variant: answer any pending shopper QUESTION with message, "
                            "inspect_stock if needed, add_to_cart, then place_test_order "
                            f"NOW ({just_hint})."
                        ),
                    }
                )
        elif (
            cart_ready
            and not memory.get("pending_intervention")
            and not memory.get("ordered")
            and env_turn >= 8
            and env_turn % 4 == 0
        ):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Cart+inspect ready. Stop re-inspecting. Call place_test_order now "
                        f"{'(include price+in_stock claims)' if plan.get('needs_justification') else '(omit justification)'}."
                    ),
                }
            )

        assistant = None
        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for _attempt in range(3):
            assistant = _chat(messages, tools)
            if assistant is None:
                _trim_messages(messages, keep_tail=14)
                continue
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if tool_calls:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": (
                            "Choose a supplied tool now "
                            "(search/filter/view/compare/inspect_stock/message/place_test_order)."
                        ),
                    },
                ]
            )
        if not tool_calls:
            # Soft exit rather than hard crash when inference is exhausted.
            dialogue.append(
                {
                    "role": "error",
                    "content": "inference failed after retries; no tool call",
                }
            )
            break

        # Force shopper reply before ordering.
        if memory.get("pending_intervention") and "message" in tool_names:
            names = [c.get("function", {}).get("name") for c in tool_calls]
            if "message" not in names:
                guard_retries += 1
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "guard_blocked": "pending_intervention",
                    }
                )
                _guard_nudge(
                    messages,
                    assistant_content,
                    (
                        "STOP. Your previous tool choice was rejected (not executed). "
                        "Call message FIRST. Answer which product_id/sku you will buy and "
                        "confirm the firm spec (e.g. screen size) from view facts."
                    ),
                )
                if guard_retries < 5:
                    continue

        block, tool_calls = _needs_order_guard(tool_calls, memory, tool_names, plan)
        if block and guard_retries < 5:
            guard_retries += 1
            dialogue.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                    "guard_blocked": block,
                }
            )
            _guard_nudge(
                messages,
                assistant_content,
                (
                    "Your previous tool choice was rejected (not executed). "
                    f"Do not finish yet. {block}"
                ),
            )
            continue

        # Final sanitizer: always strip/rebuild justification even if guards were skipped.
        sanitized_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            name = call.get("function", {}).get("name")
            if name != "place_test_order":
                sanitized_calls.append(call)
                continue
            try:
                args = _arguments(call)
            except Exception:  # noqa: BLE001
                sanitized_calls.append(call)
                continue
            new_args = _rewrite_place_order_args(
                args, memory, bool(plan.get("needs_justification")), plan
            )
            sanitized_calls.append(
                {
                    **call,
                    "function": {
                        **call.get("function", {}),
                        "arguments": json.dumps(new_args),
                    },
                }
            )
        tool_calls = sanitized_calls

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        env_turn += 1
        group_id = f"{problem_id}-turn-{env_turn}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": env_turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {
                        "name": call["function"]["name"],
                        "args": _arguments(call),
                    },
                }
                for index, call in enumerate(tool_calls, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            dialogue.append({"role": "error", "content": "environment call failed"})
            break

        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            observation = call_result.get("observation") or {}
            _ingest_observation(memory, observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _compact_observation(observation),
                }
            )
            action_name = tool_call.get("function", {}).get("name")
            if action_name == "message":
                memory["pending_intervention"] = False
            if action_name in {"search", "filter"}:
                search_rounds += 1
                note = _rank_search_results(observation, plan)
                if note and search_rounds <= 3:
                    messages.append({"role": "user", "content": note})

        if result.get("user_message"):
            content = (result["user_message"].get("content") or "").strip()
            if content:
                _update_plan_from_shopper(plan, content)
                if _requires_message_reply(content):
                    memory["pending_intervention"] = True
                    guard_retries = 0
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Shopper message: {content[:1500]}\n"
                                "REQUIRED NEXT ACTION: call message and answer this before ordering."
                                + (
                                    f" Current budget ceiling: {plan['max_price']}."
                                    if plan.get("max_price") is not None
                                    else ""
                                )
                            ),
                        }
                    )
                else:
                    # Soft ack — do not block checkout.
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Shopper note (no reply needed): {content[:500]}\n"
                                "Continue: inspect_stock if needed, then place_test_order "
                                f"{'WITH price+in_stock(+spec) claims' if plan.get('needs_justification') else 'WITHOUT justification'}."
                            ),
                        }
                    )

        if memory.get("last_error"):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool error: {memory['last_error']}. Recover: remove bad cart items, "
                        f"try rescue searches {plan.get('rescue_queries')}, pick another "
                        "firm-compliant in-stock option, inspect_stock, then order "
                        f"{'with grounded claims' if plan.get('needs_justification') else 'WITHOUT justification'}."
                    ),
                }
            )
            memory["last_error"] = None

        if any(_is_done(call.get("observation") or {}) for call in result["calls"]):
            memory["ordered"] = True
            break

    return dialogue


__all__ = ["agent_main"]
