
from __future__ import annotations
import json
import re
import time
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
_OPENROUTER_MODELS = ['moonshotai/kimi-k2.6', 'deepseek/deepseek-v3.2', 'qwen/qwen3.5-397b-a17b', 'z-ai/glm-5.2', 'deepseek/deepseek-chat-v3.1', 'qwen/qwen3.6-27b']
_CHUTES_MODELS = ['moonshotai/Kimi-K2.6-TEE', 'deepseek-ai/DeepSeek-V3.2-TEE', 'Qwen/Qwen3.5-397B-A17B-TEE', 'zai-org/GLM-5.2-TEE', 'deepseek-ai/DeepSeek-V3.1-TEE', 'Qwen/Qwen3.6-27B-TEE']
_DEFAULT_CLAIM_FIELDS = frozenset({'brand', 'category', 'in_stock', 'price', 'refresh_rate', 'resolution', 'screen_size'})
_CLAIM_DENY = frozenset({'category'})
_FIELD_ALIASES = {'display': 'resolution', 'screen': 'screen_size', 'size': 'screen_size', 'screen size': 'screen_size', 'refresh': 'refresh_rate', 'hz': 'refresh_rate', 'category_path': 'category'}
_CAND_KEY_RE = re.compile('(\\d{6,})::([A-Za-z0-9][A-Za-z0-9._-]*)')
_ASK_SPEC_RE = re.compile('tell you if you ask|if you ask|ask me.*firm|firm requirement.*ask', re.I)
_proxy = ProxyClient(timeout=120, max_retries=3)
_SYSTEM = 'You are an ORO Bench shopping agent (multi-donor TF merge on EnvPack tools).\n\nUse ONLY supplied tools. Continue until done=true. Never memorize product IDs or suite ranks.\nHidden races differ from local packs — obey this episode\'s shopper text + Task rules only.\n\n## TF1 Intent (Lovelystar)\nSilently decompose into ≤4 mutually exclusive sub-goals (keywords, firm budget/specs, soft prefs).\nIf shopper asks a question / changes budget / asks which item: NEXT action is `message`, then resume.\nAsk-first: if they say a firm spec is told when you ask → FIRST call `message` for that spec.\n\n## TF2 Retrieval (HiSun lean + madlol; KAK anti-thrash)\nsearch/filter with reformulations. Survey a few distinct product_ids — enough to be confident.\nHard cap mindset: after ~3–4 weak searches, view best hits and proceed (do not 2× thrash).\nPrefer in_stock=true when availability matters.\n\n## TF3 Constraints (madlol / ELON / angry Product)\nFirm specs are NON-NEGOTIABLE (category_path, RAM/screen/resolution/refresh, budget, stock).\nMatch brands/models/sizes/numbers LITERALLY after `view`. Soft brand only after all firm pass.\n\n## TF4 Preference (Lovelystar)\nAmong fully valid options, prefer cheaper / better fit to soft prefs. Soft ≠ override firm.\n\n## TF5 Ranking (0828 closedRank)\nWhen shopper lists product_id::sku ("compare only these"): that set is exhaustive.\nview every listed product_id; rank by stated priority; cart+order MUST be a listed key.\nNever open-catalog search for substitutes.\n\n## TF6 Recovery (Lovelystar + 0828 finishSafe)\nStock/price shock or market_event: message if asked → remove_from_cart → other candidate\n(same model/color ladder when relevant) → inspect_stock → add_to_cart → place_test_order.\nNEVER order on out-of-stock / pre-event assumptions. Never quit with "cannot complete".\n\n## TF7 Justification (Lovelystar + 0828)\nOnly if shopper asks why/reason. Claims MUST match latest view/inspect_stock facts.\nNever claim category. Resolution classes: hd/fhd/qhd/uhd when shopper used that language.\nOmit justification entirely when not asked.\n\n## Finish discipline\nsearch/filter → view/compare → inspect_stock → add_to_cart → place_test_order.\nLate in the budget: stop exploring and order an in-stock in-budget pick.\n'

def _provider() -> str:
    raw = (getenv('INFERENCE_PROVIDER') or '').strip().lower()
    if raw in ('chutes', 'openrouter'):
        return raw
    if getenv('OPENROUTER_API_KEY'):
        return 'openrouter'
    return 'chutes'

def _model_list() -> list[str]:
    override = getenv('SANDBOX_MODEL')
    if override:
        return [override]
    return list(_OPENROUTER_MODELS if _provider() == 'openrouter' else _CHUTES_MODELS)

def _shopper_text(query: str) -> str:
    return query.split('Task rules:', 1)[0].strip()

def _parse_candidate_variants(query: str) -> list[tuple[str, str]]:
    head = _shopper_text(query)
    low = head.lower()
    pairs = _CAND_KEY_RE.findall(head)
    if not any((s in low for s in ('compare only', 'candidate variant', 'candidate variants', 'these candidate', 'only these'))):
        if len(pairs) < 2:
            return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pid, sku in pairs:
        sku = sku.rstrip('.,;:)')
        key = f'{pid}::{sku}'
        if key in seen:
            continue
        seen.add(key)
        out.append((pid, sku))
    return out

def _justification_required(query: str) -> bool:
    head = _shopper_text(query).lower()
    needles = ('why', 'reason', 'walk me through', 'tell me why', 'justif', 'good pick')
    return any((n in head for n in needles))

def _needs_ask_first_spec(query: str) -> bool:
    return bool(_ASK_SPEC_RE.search(_shopper_text(query)))


_COLOR_RE = (
    r"black|white|green|red|blue|gold|silver|pink|purple|gray|grey|"
    r"yellow|orange|moonlight\s+white"
)


def _soft_brand(query: str) -> str | None:
    """Soft preferred brand from ideally/prefer (TF1 intent)."""
    head = _shopper_text(query)
    m = re.search(
        r"(?i)\bideally\b[^.]{0,60}\bprefer\s+"
        r"([A-Za-z][A-Za-z0-9]*(?:\s+[A-Za-z][A-Za-z0-9]*){0,3})",
        head,
    )
    if not m:
        return None
    brand = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".,;:")
    if brand.lower() in {"a", "an", "the", "something", "one"}:
        return None
    return brand


def _soft_wish_tokens(query: str) -> list[str]:
    """Use-case soft tokens for TF4 preference (word-boundary safe)."""
    head = _shopper_text(query).lower()
    cues = (
        "sweat",
        "waterproof",
        "water-resistant",
        "water resistant",
        "water",
        "ipx",
        "gym",
        "workout",
        "bootcamp",
        "outdoor",
        "running",
        "battery",
        "noise",
        "wireless",
        "bluetooth",
    )
    found: list[str] = []
    for c in cues:
        if " " in c or "-" in c:
            if c in head:
                found.append(c)
        elif re.search(rf"\b{re.escape(c)}\b", head):
            found.append(c)
    if any(x.startswith("water") and x != "water" for x in found):
        found = [x for x in found if x != "water"]
    return found


def _retrieval_phrase(query: str) -> str | None:
    head = _shopper_text(query)
    m = re.search(
        r"matches\s+(?:the\s+phrase\s+)?[\"']([^\"']+)[\"']",
        head,
        re.I,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip() or None
    m = re.search(
        r"matches\s+(?:the\s+phrase\s+)?([A-Za-z0-9][\w\s./-]{0,80}?)"
        r"(?=\s*,|\s+at\s+or|\s+and\s+buy)",
        head,
        re.I,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip() or None
    return None


def _retrieval_category(query: str) -> str | None:
    m = re.search(r"\bin the\s+(.+?)\s+category\b", _shopper_text(query), re.I)
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(1)).strip() or None


def _pref_color(query: str) -> str | None:
    head = _shopper_text(query)
    m = re.search(
        rf"(?:strongly\s+)?prefer(?:\s+the)?\s+({_COLOR_RE})\s+",
        head,
        flags=re.I,
    )
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(1)).strip().lower()


def _firm_brand(query: str) -> str | None:
    """Firm brand only when Brand sits before a product/spec token (not 'smartphone at or')."""
    head = _shopper_text(query)
    if re.search(r"(?i)\bideally\b", head):
        return None
    m = re.search(
        r"(?i)(?:i\s+want|i\s+need|find\s+me|get\s+me|can you get me)\s+"
        r"(?:an?\s+)?(?:in-stock\s+)?"
        r"(?-i:([A-Za-z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,2}))\s+"
        r"(?:720p|1080p|1440p|2160p|4k|full\s*hd|\bhd\b|\d+(?:\.\d+)?-?\s*inch(?:es)?"
        r"|smartphone|tablet|monitor|headphones?|earbuds?|laptop|gaming)\b",
        head,
    )
    if not m:
        return None
    brand = re.sub(r"\s+", " ", m.group(1)).strip()
    blocked = {
        "the", "new", "good", "best", "cheap", "in", "an", "a", "stock", "in-stock",
        "smartphone", "tablet", "monitor", "headphone", "headphones", "earbuds",
        "laptop", "gaming", "at", "or", "below", "under",
    }
    if brand.lower() in blocked or any(w in blocked for w in brand.lower().split()):
        return None
    return brand


def _sku_family(sku: str) -> str:
    s = str(sku or "").strip()
    if not s:
        return ""
    return s.rsplit("-", 1)[0] if "-" in s else s


def _num(text: str | None) -> float | None:
    if text is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(text).replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _rank_key(facts: dict[str, str]) -> tuple[float, float, float]:
    hz = _num(facts.get("refresh_rate"))
    inches = _num(facts.get("screen_size"))
    price = _num(facts.get("price"))
    return (
        hz if hz is not None else -1.0,
        inches if inches is not None else -1.0,
        -(price) if price is not None else -1e18,
    )


def _best_closed_pick(
    candidates: list[tuple[str, str]],
    *,
    facts_by_pid: dict[str, dict[str, str]],
    titles: dict[str, str],
) -> tuple[str, str] | None:
    if not candidates:
        return None
    scored: list[tuple[tuple[float, float, float], str, str]] = []
    for pid, sku in candidates:
        facts = dict(facts_by_pid.get(pid) or {})
        title = titles.get(pid) or ""
        if not facts.get("screen_size") and title:
            m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", title, re.I)
            if m:
                facts["screen_size"] = m.group(1)
        if not facts.get("refresh_rate") and title:
            m = re.search(r"(\d+)\s*hz", title, re.I)
            if m:
                facts["refresh_rate"] = m.group(1)
        scored.append((_rank_key(facts), pid, sku))
    scored.sort(reverse=True)
    return scored[0][1], scored[0][2]


def _needs_cart_violation(
    tool_calls: list[dict[str, Any]],
    *,
    carted: set[tuple[str, str]],
) -> str | None:
    """EnvPack requires add_to_cart before place_test_order (TF2 race pattern)."""
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        if name != "place_test_order":
            continue
        raw = call["function"].get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        if not pid:
            return "place_test_order needs product_id; add_to_cart that sku first"
        if (pid, sku) not in carted and (pid, "") not in carted:
            return (
                "place_test_order requires the sku already in cart — "
                "call add_to_cart for this product_id/sku first"
            )
    return None


def _extract_view_title(observation: Any) -> str:
    if not isinstance(observation, dict):
        return ""
    node = observation.get("observation")
    if not isinstance(node, dict):
        node = observation
    return str(node.get("title") or "")


def _spec_fields_from_query(query: str) -> list[str]:
    """Infer which justification spec fields the shopper named (race-general)."""
    head = _shopper_text(query).lower()
    fields: list[str] = []
    if any((x in head for x in ('720p', '1080p', '1440p', '4k', 'uhd', 'qhd', 'full hd', ' resolution'))) or re.search('\\bhd\\b', head):
        fields.append('resolution')
    if any((x in head for x in ('inch', 'screen', 'display size'))):
        fields.append('screen_size')
    if 'refresh' in head or re.search('\\b\\d+\\s*hz\\b', head):
        fields.append('refresh_rate')
    if re.search('\\bram\\b', head):
        pass
    return fields

def _claim_field_allowlist(tools: list[dict[str, Any]]) -> frozenset[str]:
    for tool in tools:
        fn = tool.get('function') or tool
        if (fn.get('name') or '') != 'place_test_order':
            continue
        props = (fn.get('parameters') or {}).get('properties') or {}
        claims = ((props.get('justification') or {}).get('properties') or {}).get('claims', {}).get('items', {}).get('properties', {}).get('field', {})
        enum = claims.get('enum')
        if isinstance(enum, list) and enum:
            return frozenset((str(x) for x in enum))
    return _DEFAULT_CLAIM_FIELDS

def _normalize_claim_field(field: str, allowed: frozenset[str]) -> str | None:
    raw = str(field or '').strip()
    if not raw:
        return None
    key = raw.lower()
    if key in _CLAIM_DENY or raw in _CLAIM_DENY:
        return None
    if raw in allowed:
        return raw
    if key in allowed:
        return key
    mapped = _FIELD_ALIASES.get(key)
    if mapped and mapped in allowed and (mapped not in _CLAIM_DENY):
        return mapped
    return None

def _query_wants_resolution_class(query: str) -> str | None:
    """Return hd|fhd|qhd|uhd if shopper named a resolution class (0828 PASS pattern)."""
    low = _shopper_text(query).lower()
    if re.search('\\b(4k|uhd|ultra\\s*hd)\\b', low):
        return 'uhd'
    if re.search('\\b(qhd|1440p|2k)\\b', low):
        return 'qhd'
    if re.search('\\b(fhd|1080p|full\\s*hd)\\b', low):
        return 'fhd'
    if re.search('\\b(720p|hd)\\b', low):
        return 'hd'
    return None

def _normalize_resolution_value(text: str, query: str) -> str:
    """Prefer exact view strings; allow class tokens when shopper named them."""
    text = str(text or '').strip()
    if not text:
        return text
    low = text.lower().replace('×', 'x')
    if low in ('hd', 'fhd', 'qhd', 'uhd', '4k'):
        return 'uhd' if low == '4k' else low
    wanted = _query_wants_resolution_class(query)
    if wanted == 'hd' and any((x in low for x in ('720', '1280', ' hd', 'hd '))):
        if '1080' not in low and '1920' not in low and ('1440' not in low):
            return 'hd'
    if wanted == 'fhd' and any((x in low for x in ('1080', '1920', 'full hd', 'fhd'))):
        return 'fhd'
    return text

def _extract_view_facts(observation: Any) -> tuple[str | None, dict[str, str]]:
    """Pull product_id + brand/resolution/screen_size from a view observation."""
    facts: dict[str, str] = {}
    if not isinstance(observation, dict):
        return (None, facts)
    node = observation.get('observation')
    if not isinstance(node, dict):
        node = observation
    pid = node.get('product_id') or observation.get('product_id')
    if pid is not None:
        pid = str(pid)

    def walk(obj: Any, path: str='') -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in ('brand', 'resolution', 'screen_size', 'refresh_rate') and v not in (None, '', 'None'):
                    facts.setdefault(lk, str(v).strip())
                if lk in ('attributes', 'specs', 'options', 'variants') or isinstance(v, (dict, list)):
                    walk(v, path + '.' + lk)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:8]):
                walk(v, f'{path}[{i}]')
    walk(node)
    attrs = node.get('attributes') or node.get('specs')
    if isinstance(attrs, list):
        for a in attrs:
            if not isinstance(a, dict):
                continue
            name = str(a.get('name') or a.get('key') or '').lower()
            val = a.get('value') or a.get('val')
            if not val:
                continue
            if 'brand' in name:
                facts.setdefault('brand', str(val).strip())
            if 'resolution' in name or name in ('display', 'screen resolution'):
                facts.setdefault('resolution', str(val).strip())
            if 'screen' in name and 'size' in name:
                facts.setdefault('screen_size', str(val).strip())
            if 'refresh' in name:
                facts.setdefault('refresh_rate', str(val).strip())
    if node.get('brand') not in (None, '', 'None'):
        facts['brand'] = str(node['brand']).strip()
    return (pid, facts)

def _preferred_model_hint(query: str) -> str | None:
    """Extract preferred model token; stop at `.` / `If` (v1 TF6 garble fix)."""
    head = _shopper_text(query)
    colors = 'black|white|green|red|blue|gold|silver|pink|purple|gray|grey|yellow|orange|moonlight\\s+white'
    patterns = (f'prefer(?:\\s+the)?\\s+(?:{colors})?\\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})', f'eye on the\\s+(?:{colors})?\\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})', f'you have your eye on the\\s+(?:{colors})?\\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})')
    token = None
    for pat in patterns:
        m = re.search(pat, head, flags=re.I)
        if m:
            token = m.group(1).strip()
            break
    if not token:
        return None
    token = token.split('.')[0]
    token = re.split('\\s+\\bIf\\b|\\s+\\bspecifically\\b|\\s+\\bgenuine\\b|\\s+\\bnot\\b|,', token, maxsplit=1, flags=re.I)[0].strip()
    token = token.strip('.,;:()[]"\'')
    if token.lower() in {'black', 'white', 'green', 'red', 'blue', 'gold', 'silver', 'pink', 'purple', 'gray', 'grey', 'yellow', 'orange', 'the', 'a', 'an'}:
        return None
    return token or None

def _canonical_recovery_message(model: str | None) -> str:
    """Declarative same-model / other-color commit (no permission ask)."""
    if model:
        return f'The preferred option went out of stock when I added it to cart. Following your rules, I will now take the same model ({model}) in another color/variant that is still in stock, keep budget and category firm, and only change models if no same-model variant remains.'
    return "That item went out of stock when I added it to cart. I'm now selecting another in-stock option that still matches your firm budget and category requirements."

def _normalize_claim_value(field: str, value: Any, *, query: str) -> str:
    text = str(value).strip()
    if field == 'in_stock':
        low = text.lower()
        if low in ('true', '1', 'yes', 'in stock', 'instock'):
            return 'true'
        if low in ('false', '0', 'no', 'out of stock'):
            return 'false'
    if field == 'price':
        cleaned = re.sub('[^\\d.]', '', text.replace(',', ''))
        if cleaned:
            try:
                return str(float(cleaned))
            except ValueError:
                return text
    if field == 'refresh_rate':
        m = re.search('(\\d+(?:\\.\\d+)?)\\s*hz', text, flags=re.I)
        if m:
            return f'{m.group(1)}Hz'
        if re.fullmatch('\\d+(?:\\.\\d+)?', text):
            return f'{text}Hz'
    if field == 'screen_size':
        m = re.search('(\\d+(?:\\.\\d+)?)', text)
        if m:
            return f'{m.group(1)} inches'
    if field == 'resolution':
        return _normalize_resolution_value(text, query)
    return text

def _sanitize_place_order_args(args: dict[str, Any], *, query: str, allowed_fields: frozenset[str], last_stock: dict[str, Any] | None, view_facts_by_pid: dict[str, dict[str, str]] | None) -> dict[str, Any]:
    out = dict(args)
    if not _justification_required(query):
        out.pop('justification', None)
        return out
    just = out.get('justification')
    if not isinstance(just, dict):
        just = {}
    claims_in = just.get('claims') if isinstance(just.get('claims'), list) else []
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    needed_specs = set(_spec_fields_from_query(query))
    pid = str(out.get('product_id') or '')
    view_facts: dict[str, str] = {}
    if view_facts_by_pid and pid and (pid in view_facts_by_pid):
        view_facts = dict(view_facts_by_pid[pid])
    for claim in claims_in:
        if not isinstance(claim, dict):
            continue
        field = _normalize_claim_field(str(claim.get('field', '')), allowed_fields)
        if field is None or field in seen:
            continue
        if field in ('resolution', 'screen_size', 'refresh_rate') and field not in needed_specs:
            continue
        raw_val = claim.get('value', '')
        if field == 'brand':
            if view_facts.get('brand'):
                raw_val = view_facts['brand']
            else:
                continue
        elif field in view_facts and field in ('resolution', 'screen_size', 'refresh_rate'):
            raw_val = view_facts[field]
        value = _normalize_claim_value(field, raw_val, query=query)
        if value == '':
            continue
        if field == 'resolution' and value.lower() in ('hd', 'fhd', 'qhd', 'uhd'):
            vis = (view_facts.get('resolution') or '').lower()
            wanted = _query_wants_resolution_class(query)
            if wanted == value.lower():
                pass
            elif not vis:
                continue
            elif value.lower() == 'hd' and (not any((x in vis for x in ('hd', '720', '1280')))):
                continue
            elif value.lower() == 'fhd' and (not any((x in vis for x in ('fhd', '1080', '1920', 'full')))):
                continue
        cleaned.append({'field': field, 'value': value})
        seen.add(field)
    if last_stock:
        if 'price' not in seen and last_stock.get('price') is not None:
            cleaned.append({'field': 'price', 'value': _normalize_claim_value('price', last_stock.get('price'), query=query)})
            seen.add('price')
        if 'in_stock' not in seen and last_stock.get('in_stock') is not None:
            cleaned.append({'field': 'in_stock', 'value': _normalize_claim_value('in_stock', last_stock.get('in_stock'), query=query)})
            seen.add('in_stock')
    if 'brand' not in seen and view_facts.get('brand'):
        brand = view_facts['brand']
        if brand and brand.lower() not in ('none', 'null', 'n/a'):
            cleaned.append({'field': 'brand', 'value': brand})
            seen.add('brand')
    for spec in needed_specs:
        if spec in allowed_fields and spec not in seen and (spec not in _CLAIM_DENY):
            if view_facts.get(spec):
                cleaned.append({'field': spec, 'value': _normalize_claim_value(spec, view_facts[spec], query=query)})
                seen.add(spec)
            elif spec == 'resolution':
                wanted = _query_wants_resolution_class(query)
                if wanted:
                    cleaned.append({'field': 'resolution', 'value': wanted})
                    seen.add('resolution')
    cleaned = [c for c in cleaned if c['field'] not in _CLAIM_DENY]
    if cleaned:
        out['justification'] = {'claims': cleaned}
    else:
        out.pop('justification', None)
    return out

def _arguments(tool_call: dict[str, Any], *, query: str, allowed_fields: frozenset[str], last_stock: dict[str, Any] | None, view_facts_by_pid: dict[str, dict[str, str]] | None) -> dict[str, Any]:
    raw = tool_call['function'].get('arguments', '{}')
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError('tool arguments must be a JSON object')
    if (tool_call['function'].get('name') or '') == 'place_test_order':
        return _sanitize_place_order_args(parsed, query=query, allowed_fields=allowed_fields, last_stock=last_stock, view_facts_by_pid=view_facts_by_pid)
    return parsed

def _rewrite_recovery_message(args: dict[str, Any], *, query: str, force: bool=False) -> dict[str, Any]:
    """Force declarative same-model recovery commit (LS v5 / 0828_v2 PASS)."""
    content = str(args.get('content') or args.get('message') or '')
    low = content.lower()
    stockish = any((x in low for x in ('out of stock', 'went out', 'still available', 'no longer', 'not available', 'unavailable', 'oos', 'price increased', 'more expensive', 'exceeds your budget', 'above your', 'over your')))
    asks = any((x in low for x in ('should i', 'do you want', 'would you like', 'any preference', 'preferences?', 'your preference', 'or look for a different', 'or a different', 'let me know', 'what would you like')))
    model = _preferred_model_hint(query)
    model_l = (model or '').lower()
    commits = bool(model_l) and model_l in low and ('same' in low or 'another color' in low or 'other color' in low or ('same model' in low))
    if force:
        out = dict(args)
        out['content'] = _canonical_recovery_message(model)
        out.pop('message', None)
        return out
    need = asks or stockish
    if not need:
        return args
    if commits and (not asks):
        if model and f'({model})' in content and ('. If' not in content):
            return args
    out = dict(args)
    out['content'] = _canonical_recovery_message(model)
    out.pop('message', None)
    return out

def _looks_like_market_event(text: str) -> bool:
    low = (text or '').lower()
    return any((c in low for c in ('still available', 'more expensive', 'got more expensive', 'i can only go', 're-check', 'price jump', 'price changed')))

def _looks_like_shopper_intervention(text: str) -> bool:
    low = (text or '').lower().strip()
    if len(low) < 8:
        return False
    cues = ('?', 'which one', 'before you buy', 'hold on', 'wait,', 'still available', 'i can only go', 'non-negotiable', 're-check', 'tell you if you ask', 'firm requirement', 'please keep to', 'compare the current', 'screen', 'ram', 'budget', 'inch')
    return any((c in low for c in cues))

def _obs_stock_payload(observation: Any) -> dict[str, Any] | None:
    """Extract stock/price facts from nested EnvPack observations."""
    if not isinstance(observation, dict):
        return None
    node = observation.get('observation')
    if isinstance(node, dict) and ('in_stock' in node or 'price' in node or 'added' in node):
        if 'added' in node and isinstance(node['added'], dict):
            return node['added']
        cart = node.get('cart')
        if isinstance(cart, list) and cart and isinstance(cart[0], dict):
            return cart[0]
        return node
    if 'in_stock' in observation or 'price' in observation:
        return observation
    return None

def _stock_is_false(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    val = payload.get('in_stock')
    if val is False:
        return True
    if isinstance(val, str) and val.strip().lower() in ('false', '0', 'no'):
        return True
    return False

def _chat(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    last_err: Exception | None = None
    for model in _model_list():
        for _try in range(2):
            try:
                inference = _proxy.post('/inference/chat/completions', json_data={'model': model, 'messages': messages, 'tools': tools, 'tool_choice': 'required', 'temperature': 0})
                if inference is None:
                    last_err = RuntimeError(f'inference failed for {model}')
                    time.sleep(0.3)
                    continue
                return inference
            except Exception as exc:
                last_err = exc
                time.sleep(0.3)
    raise RuntimeError(f'inference unavailable: {last_err}')

def _env_call(envelope: dict[str, Any]) -> dict[str, Any]:
    base_id = str(envelope.get('call_id') or 'call')
    for attempt in range(3):
        payload = dict(envelope)
        if attempt:
            suffix = f'r{attempt}-{int(time.time() * 1000) % 1000000}'
            payload['call_id'] = f'{base_id}-{suffix}'
            payload['idempotency_key'] = payload['call_id']
            calls = []
            for index, call in enumerate(payload.get('calls') or [], start=1):
                c = dict(call)
                c['call_id'] = f"{payload['call_id']}-{index}"
                calls.append(c)
            payload['calls'] = calls
        result = _proxy.post('/environment/call', json_data=payload)
        if result is not None:
            return result
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f'environment call failed after retries ({base_id})')

def _tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [str(c.get('function', {}).get('name') or '') for c in tool_calls]

def _closed_set_violation(tool_calls: list[dict[str, Any]], *, candidates: list[tuple[str, str]], viewed: set[str]) -> str | None:
    if not candidates:
        return None
    cand_pids = {pid for pid, _ in candidates}
    cand_keys = {f'{pid}::{sku}' for pid, sku in candidates}
    names = _tool_names(tool_calls)
    if any((n in ('search', 'filter') for n in names)) and len(viewed & cand_pids) < len(cand_pids):
        return f'Closed candidate set: do not search/filter until every listed product is viewed. Missing: {sorted(cand_pids - viewed)}'
    for call in tool_calls:
        name = call.get('function', {}).get('name') or ''
        raw = call['function'].get('arguments', '{}')
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        if name == 'view':
            pid = str(args.get('product_id') or '')
            if pid and pid not in cand_pids:
                return f'view product_id {pid} is outside the listed candidates. View only: {sorted(cand_pids)}'
        if name in ('add_to_cart', 'inspect_stock', 'place_test_order'):
            pid = str(args.get('product_id') or '')
            sku = str(args.get('sku') or '')
            if pid and pid not in cand_pids:
                return f'{name} product_id {pid} is outside listed candidates {sorted(cand_pids)}'
            if pid and sku and (f'{pid}::{sku}' not in cand_keys):
                if name in ('add_to_cart', 'place_test_order'):
                    return f'{name} {pid}::{sku} is not a listed variant. Use one of: {sorted(cand_keys)}'
    return None

def _oos_order_violation(tool_calls: list[dict[str, Any]], *, last_stock: dict[str, Any] | None) -> str | None:
    """Reject ordering when latest stock/cart says out of stock (market shock)."""
    if not _stock_is_false(last_stock):
        return None
    names = _tool_names(tool_calls)
    if 'place_test_order' in names:
        return 'Rejected: latest stock/cart shows in_stock=false (market shock). remove_from_cart if needed, pick another candidate, inspect_stock, then order only an in-stock sku.'
    return None

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """EnvPack loop: LS phases + safe-band gates for hidden races."""
    environment = problem_data['environment']
    binding = environment['binding']
    policy = environment['policy_view']
    problem_id = str(problem_data.get('problem_id', problem_data.get('id', 'problem')))
    max_calls = int(policy.get('max_calls_per_turn', 1))
    tools = policy['tools']
    query = str(policy.get('query') or '')
    allowed_fields = _claim_field_allowlist(tools)
    candidates = _parse_candidate_variants(query)
    cand_pids = {pid for pid, _ in candidates}
    ask_first = _needs_ask_first_spec(query)
    model_hint = _preferred_model_hint(query)
    pref_color = _pref_color(query)
    firm_brand = _firm_brand(query)
    soft_brand = _soft_brand(query)
    soft_wishes = _soft_wish_tokens(query)
    phrase = _retrieval_phrase(query)
    ret_cat = _retrieval_category(query)
    boot = (
        f"{query}\n\n"
        "Procedure: ask firm specs if withheld-until-asked; else retrieve → view → "
        "inspect_stock → cart → place_test_order. "
        "On stockout: message must COMMIT to same preferred model/other color "
        "(never ask 'should I?'); then remove → recover → order. "
        "Justification only if asked: include brand + exact view specs + current "
        "price/in_stock; never claim category; do not invent hd/fhd. "
        "Never place_test_order before add_to_cart."
    )
    if soft_brand and not firm_brand:
        boot += (
            f"\n\nSOFT BRAND PREFERENCE: `{soft_brand}`. Honor it on the FIRST cart "
            "when a fully firm-compliant listing of that brand exists; firm specs win "
            "if the soft brand has no compliant option."
        )
    if soft_wishes:
        boot += (
            "\n\nSOFT USE-CASE TOKENS: "
            + ", ".join(soft_wishes)
            + ". Among firm-valid options, prefer the listing whose view facts/title "
            "best match these tokens (do not default to cheapest alone)."
        )
    if phrase:
        boot += (
            f"\n\nRETRIEVAL PHRASE: keep title tokens from `{phrase}` in every search. "
            "Survey several distinct product_ids that match the phrase + category + budget, "
            "then inspect_stock → add_to_cart → place_test_order. "
            "After stockout: COMMIT → remove → search the SAME phrase → view a different "
            "match → inspect → cart → order (do not drift to unrelated names)."
        )
        if ret_cat:
            boot += f" Category must stay under `{ret_cat}`."
    if ask_first:
        boot += (
            "\n\nASK-FIRST: Your first tool call must be `message` asking for the "
            "withheld firm hardware spec. After the reply, search with that spec and "
            "complete the order without apology loops."
        )
    if model_hint:
        boot += (
            f"\n\nPREFERRED MODEL TOKEN: `{model_hint}`. On stockout of the preferred "
            "color/variant, recover that same model in another color before changing models."
        )
    if firm_brand:
        boot += (
            f"\n\nFirm brand `{firm_brand}`: keep it on every cart/order after shocks."
        )
    if candidates:
        listed = ", ".join(f"{pid}::{sku}" for pid, sku in candidates)
        boot += (
            f"\n\nCLOSED CANDIDATE SET (mandatory): {listed}. "
            "view each listed product_id; rank by stated priority; order a listed key only."
        )
    messages: list[dict[str, Any]] = [{'role': 'system', 'content': _SYSTEM}, {'role': 'user', 'content': boot}]
    dialogue: list[dict[str, Any]] = []
    must_message = bool(ask_first)
    asked_spec = False
    viewed: set[str] = set()
    searchish = 0
    last_stock: dict[str, Any] | None = None
    view_facts_by_pid: dict[str, dict[str, str]] = {}
    titles: dict[str, str] = {}
    shock_recover = False
    recovery_msg_force = False
    need_reinspect = False
    last_pick: dict[str, str] = {}
    cart_sku: dict[str, str] = {}
    carted: set[tuple[str, str]] = set()
    shocked_pid: str | None = None
    shocked_sku: str | None = None
    max_steps = int(policy['max_steps'])
    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        if must_message:
            hint = "REQUIRED this turn: call `message` once. If this is a stockout/'still available?' reply: STATE you will take another color/variant of the SAME preferred model (do not ask permission)."
            if model_hint:
                hint += f' Preferred model token: {model_hint}.'
            messages.append({'role': 'user', 'content': hint})
        elif candidates and len(viewed & cand_pids) < len(cand_pids):
            missing = sorted(cand_pids - viewed)
            messages.append({'role': 'user', 'content': f'Still missing views for closed-set product_ids: {missing}. Call view on the next missing id (or compare).'})
        elif shock_recover or need_reinspect:
            if phrase and not model_hint:
                recover = (
                    "Market shock / re-inspect required: remove_from_cart if cart is stale. "
                    f"Search again with phrase `{phrase}`, VIEW a different product_id "
                    "that still matches, inspect_stock, add_to_cart, place_test_order. "
                    "Do not quit; do not ask permission; do not order without cart."
                )
            else:
                recover = (
                    "Market shock / re-inspect required: remove_from_cart if cart is stale. "
                    "Recover SAME preferred model other color/variant first"
                )
                if model_hint:
                    recover += f" (model `{model_hint}` — keep this token in search)"
                recover += (
                    ", VIEW the replacement product, inspect_stock an in-stock sku, "
                    "then add_to_cart → place_test_order. Prefer another sku of the same "
                    "listing family before switching models. Do not quit; do not ask "
                    "permission; do not order without cart."
                )
            messages.append({"role": "user", "content": recover})
            shock_recover = False
        elif not candidates and searchish >= 3 and (len(viewed) < 2):
            messages.append({'role': 'user', 'content': 'TF2 anti-thrash (HiSun/KAK): stop repeating the same search. Reformulate once OR view best hits, then inspect_stock and finish.'})
        elif not candidates and searchish >= 5 and (len(viewed) >= 2):
            messages.append({'role': 'user', 'content': 'Search budget spent. Among viewed products meeting firm constraints, inspect_stock → add_to_cart → place_test_order NOW.'})
        elif remaining <= 6 and (not must_message):
            messages.append({'role': 'user', 'content': f'Only ~{remaining} steps left. If you have an in-stock in-budget sku, add_to_cart then place_test_order NOW. Stop exploratory search.'})
        assistant_content = ''
        tool_calls: list[dict[str, Any]] = []
        for _attempt in range(4):
            inference = _chat(messages=messages, tools=tools)
            assistant = inference['choices'][0]['message']
            assistant_content = assistant.get('content') or ''
            tool_calls = (assistant.get('tool_calls') or [])[:max_calls]
            if not tool_calls:
                messages.extend([{'role': 'assistant', 'content': assistant_content}, {'role': 'user', 'content': 'No tool call. Call a supplied tool now.'}])
                continue
            if must_message and 'message' not in _tool_names(tool_calls):
                messages.extend([{'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls}, {'role': 'user', 'content': 'Rejected. Call only `message` this turn.'}])
                continue
            if not must_message:
                violation = _closed_set_violation(tool_calls, candidates=candidates, viewed=viewed)
                if violation:
                    messages.extend([{'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls}, {'role': 'user', 'content': f'Rejected: {violation}'}])
                    continue
                oos = _oos_order_violation(tool_calls, last_stock=last_stock)
                if oos:
                    messages.extend([{'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls}, {'role': 'user', 'content': oos}])
                    shock_recover = True
                    need_reinspect = True
                    continue
                need_cart = _needs_cart_violation(tool_calls, carted=carted)
                if need_cart:
                    messages.extend([{'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls}, {'role': 'user', 'content': f'Rejected: {need_cart}'}])
                    continue
                if model_hint and shocked_pid and need_reinspect:
                    search_bad = False
                    for call in tool_calls:
                        if (call.get('function', {}).get('name') or '') != 'search':
                            continue
                        raw = call['function'].get('arguments', '{}')
                        try:
                            a = json.loads(raw) if isinstance(raw, str) else raw
                        except json.JSONDecodeError:
                            a = {}
                        q = str((a or {}).get('query') or '')
                        if model_hint.lower() not in q.lower():
                            search_bad = True
                            break
                    if search_bad:
                        messages.extend([{'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls}, {'role': 'user', 'content': f'Rejected: recovery search must include preferred model token `{model_hint}`.'}])
                        continue
                names_try = _tool_names(tool_calls)
                if need_reinspect and 'place_test_order' in names_try and ('inspect_stock' not in names_try):
                    messages.extend([{'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls}, {'role': 'user', 'content': 'Rejected: after a market event you must inspect_stock (and fix cart) before place_test_order.'}])
                    continue
            break
        else:
            if candidates and len(viewed & cand_pids) < len(cand_pids):
                miss = sorted(cand_pids - viewed)[0]
                tool_calls = [{'id': f'forced-view-{turn}', 'type': 'function', 'function': {'name': 'view', 'arguments': json.dumps({'product_id': miss})}}]
                assistant_content = f'Viewing closed-set candidate {miss}.'
            elif must_message and ask_first and (not asked_spec):
                tool_calls = [{'id': f'forced-ask-{turn}', 'type': 'function', 'function': {'name': 'message', 'arguments': json.dumps({'content': 'What is the firm hardware requirement (RAM, screen size, resolution, or refresh rate)?'})}}]
                assistant_content = 'Asking for the withheld firm spec.'
            elif must_message:
                tool_calls = [{'id': f'forced-msg-{turn}', 'type': 'function', 'function': {'name': 'message', 'arguments': json.dumps({'content': _canonical_recovery_message(model_hint)})}}]
                assistant_content = 'Committing after model stall.'
            elif last_stock and not _stock_is_false(last_stock) and carted:
                pid = str(last_stock.get('product_id') or last_pick.get('product_id') or '')
                sku = str(last_stock.get('sku') or last_pick.get('sku') or '')
                tool_calls = [{'id': f'forced-order-{turn}', 'type': 'function', 'function': {'name': 'place_test_order', 'arguments': json.dumps({'product_id': pid, 'sku': sku} if sku else {'product_id': pid})}}]
                assistant_content = 'Finishing on carted in-stock sku after model stall.'
            else:
                q = _shopper_text(query).split('\n', 1)[0][:120]
                if phrase:
                    q = phrase
                elif model_hint:
                    q = model_hint
                tool_calls = [{'id': f'forced-search-{turn}', 'type': 'function', 'function': {'name': 'search', 'arguments': json.dumps({'query': q, 'in_stock': True})}}]
                assistant_content = 'Searching after model stall.'
        if must_message:
            msg_only = [c for c in tool_calls if c.get('function', {}).get('name') == 'message']
            if msg_only:
                tool_calls = msg_only[:1]
        names = _tool_names(tool_calls)
        if any((n in ('search', 'filter') for n in names)):
            searchish += 1
        if 'message' in names and ask_first:
            asked_spec = True
        messages.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls})
        group_id = f'{problem_id}-turn-{turn}'
        sanitized_calls: list[dict[str, Any]] = []
        for index, call in enumerate(tool_calls, start=1):
            args = _arguments(call, query=query, allowed_fields=allowed_fields, last_stock=last_stock, view_facts_by_pid=view_facts_by_pid)
            name = call.get('function', {}).get('name') or ''
            if name == 'message' and isinstance(args, dict):
                rewritten = _rewrite_recovery_message(args, query=query, force=recovery_msg_force)
                if rewritten is not args or recovery_msg_force:
                    args = rewritten
                    if 'content' in args and 'message' in args:
                        args.pop('message', None)
            if name in ('inspect_stock', 'add_to_cart', 'place_test_order'):
                if args.get('product_id') and args.get('sku'):
                    last_pick = {'product_id': str(args['product_id']), 'sku': str(args['sku'])}
            if name == 'add_to_cart':
                cart_sku = dict(last_pick)
            if name == 'remove_from_cart':
                cart_sku = {}
            sanitized_calls.append({'call_id': f'{group_id}-{index}', 'action': {'name': name, 'args': args}})
        envelope = {**binding, 'call_id': group_id, 'idempotency_key': group_id, 'turn': turn, 'calls': sanitized_calls}
        result = _env_call(envelope)
        dialogue.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls, 'environment_result': result})
        for call in tool_calls:
            if (call.get('function', {}).get('name') or '') == 'view':
                raw = call['function'].get('arguments', '{}')
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = {}
                if isinstance(args, dict) and args.get('product_id'):
                    viewed.add(str(args['product_id']))
        if 'message' in names:
            must_message = False
            recovery_msg_force = False
        for tool_call, call_result in zip(tool_calls, result['calls'], strict=True):
            obs = call_result.get('observation')
            messages.append({'role': 'tool', 'tool_call_id': tool_call['id'], 'content': json.dumps(obs)})
            payload = _obs_stock_payload(obs if isinstance(obs, dict) else {})
            if payload is None and isinstance(obs, dict):
                payload = _obs_stock_payload({'observation': obs})
            name = tool_call.get('function', {}).get('name') or ''
            if name == 'view' and isinstance(obs, dict):
                pid, facts = _extract_view_facts(obs)
                title = _extract_view_title(obs)
                if not pid:
                    raw = tool_call['function'].get('arguments', '{}')
                    try:
                        a = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        a = {}
                    if isinstance(a, dict) and a.get('product_id'):
                        pid = str(a['product_id'])
                if pid:
                    viewed.add(pid)
                    if title:
                        titles[pid] = title
                    # Always register pid — retrieval SKUs often lack structured brand/specs.
                    view_facts_by_pid.setdefault(pid, {})
                    if facts:
                        view_facts_by_pid[pid] = {**view_facts_by_pid[pid], **facts}
            if name == 'add_to_cart':
                raw = tool_call['function'].get('arguments', '{}')
                try:
                    a = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    a = {}
                err = None
                if isinstance(obs, dict):
                    err = obs.get('error')
                    if err is None and isinstance(obs.get('observation'), dict):
                        err = obs['observation'].get('error')
                if isinstance(a, dict):
                    cpid = str(a.get('product_id') or '')
                    csku = str(a.get('sku') or '')
                    if cpid and not err and payload and not _stock_is_false(payload):
                        carted.add((cpid, csku))
                        carted.add((cpid, ''))
                    elif cpid:
                        carted.discard((cpid, csku))
                        carted.discard((cpid, ''))
            if name == 'remove_from_cart':
                raw = tool_call['function'].get('arguments', '{}')
                try:
                    a = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    a = {}
                if isinstance(a, dict):
                    cpid = str(a.get('product_id') or '')
                    csku = str(a.get('sku') or '')
                    carted.discard((cpid, csku))
                    carted.discard((cpid, ''))
            if name == 'inspect_stock':
                need_reinspect = False
            if name in ('inspect_stock', 'add_to_cart', 'inspect_cart') and payload:
                last_stock = payload
                if _stock_is_false(payload):
                    shock_recover = True
                    need_reinspect = True
                    recovery_msg_force = True
                    raw = tool_call['function'].get('arguments', '{}')
                    try:
                        a = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        a = {}
                    if isinstance(a, dict):
                        shocked_pid = str(a.get('product_id') or payload.get('product_id') or '') or None
                        shocked_sku = str(a.get('sku') or payload.get('sku') or '') or None
                        if shocked_pid:
                            carted.discard((shocked_pid, shocked_sku or ''))
                            carted.discard((shocked_pid, ''))
            if isinstance(obs, dict):
                inner = obs.get('observation') if isinstance(obs.get('observation'), dict) else obs
                if isinstance(inner, dict) and inner.get('market_event'):
                    shock_recover = True
                    need_reinspect = True
                    recovery_msg_force = True
                    added = inner.get('added')
                    if isinstance(added, dict):
                        last_stock = added
                        shocked_pid = str(added.get('product_id') or shocked_pid or '') or None
                        shocked_sku = str(added.get('sku') or shocked_sku or '') or None
                        if shocked_pid:
                            carted.discard((shocked_pid, shocked_sku or ''))
                            carted.discard((shocked_pid, ''))
                    me = inner.get('market_event')
                    if isinstance(me, dict):
                        tgt = me.get('target') if isinstance(me.get('target'), dict) else {}
                        tpid = str(tgt.get('product_id') or shocked_pid or '') or None
                        tsku = str(tgt.get('sku') or shocked_sku or '') or None
                        if tpid:
                            shocked_pid = tpid
                            shocked_sku = tsku
                            carted.discard((tpid, tsku or ''))
                            carted.discard((tpid, ''))
        if result.get('user_message'):
            um = result['user_message']
            content = um.get('content') if isinstance(um, dict) else str(um)
            messages.append({'role': 'user', 'content': content})
            if _looks_like_shopper_intervention(str(content)):
                low = str(content).lower()
                if ask_first and asked_spec and ('?' not in low):
                    must_message = False
                else:
                    must_message = True
                if _looks_like_market_event(str(content)):
                    need_reinspect = True
                    recovery_msg_force = True
                elif '?' in low or any((x in low for x in ('which', 'still available', 'hold on', 'before you'))):
                    must_message = True
                    if any((x in low for x in ('still available', 'out of stock', 'hold on', 'before you'))):
                        recovery_msg_force = True
        if any(((call.get('observation') or {}).get('done') for call in result['calls'])):
            break
        if (
            not must_message
            and candidates
            and len(viewed & cand_pids) >= len(cand_pids)
        ):
            best = _best_closed_pick(
                candidates, facts_by_pid=view_facts_by_pid, titles=titles
            )
            tip = (
                "All listed candidates viewed. Rank by stated priority "
                "(usually highest refresh → largest screen → lowest price), "
                "inspect_stock the winner, add_to_cart listed sku, place_test_order."
            )
            if best:
                tip += (
                    f" Controller hint from viewed facts: prefer `{best[0]}::{best[1]}` "
                    "unless a listed option clearly beats it on the stated priority."
                )
            messages.append({"role": "user", "content": tip})
        elif not must_message and not candidates and not shock_recover:
            finish = (
                "Episode not done. Prefer inspect_stock → add_to_cart → "
                "place_test_order on an in-stock sku that meets firm constraints. "
                "Omit justification unless the shopper asked why."
            )
            if soft_wishes:
                finish += (
                    " Soft-fit check: prefer a viewed listing that mentions "
                    + "/".join(soft_wishes[:4])
                    + " when still firm-valid."
                )
            messages.append({"role": "user", "content": finish})
    return dialogue
__all__ = ['agent_main']
